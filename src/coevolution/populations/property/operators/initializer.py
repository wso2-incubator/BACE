"""PropertyTestInitializer — creates Gen-0 property test individuals."""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import replace

from loguru import logger

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import OPERATION_INITIAL, PopulationConfig, Problem
from coevolution.core.interfaces.language import ICodeParser
from coevolution.strategies.llm_base import (
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    LLMSyntaxError,
    llm_retry,
)
from infrastructure.languages import PythonLanguage
from infrastructure.sandbox import SandboxConfig, create_sandbox

from ..types import IOPairCache
from .helpers import transform_public_tests
from .validator import validate_property_test


class PropertyTestInitializer(BaseLLMInitializer[TestIndividual]):
    """Creates Gen-0 property test individuals via two LLM calls.

    Call 1: gen_inputs.j2 → Python input-generator script.
            The script is validated and stored in ``io_pair_cache`` so the
            evaluator can run it when ``execute_tests`` is first called.

    Call 2: describe_properties.j2 → List of property descriptions.
    Call 3: convert_description_to_property.j2 → Implementation for each description.
            Each snippet is validated against the problem's public test cases;
            invalid snippets are discarded.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
        sandbox_config: SandboxConfig,
        io_pair_cache: IOPairCache,
        llm_workers: int = 8,
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config)
        self.sandbox_config = sandbox_config
        self.io_pair_cache = io_pair_cache
        self.llm_workers = llm_workers
        self._python_lang = PythonLanguage()
        self._python_sandbox_config = replace(sandbox_config, language="python")

    # ── IPopulationInitializer ────────────────────────────────────────────────

    def initialize(self, problem: Problem) -> list[TestIndividual]:
        # LLM call 1: generate and cache the input-generator script
        self._generate_and_cache_generator(problem)

        # LLM call 2: generate and prune property test snippets
        return self._generate_property_tests(problem)

    # ── LLM call 1 ───────────────────────────────────────────────────────────

    def _generate_and_cache_generator(self, problem: Problem) -> None:
        try:
            script = self._call_gen_inputs(problem)
        except Exception as exc:
            logger.warning(
                f"PropertyTestInitializer: gen_inputs LLM call failed: {exc}. "
                "Input-generator script omitted; property tests may not run effectively."
            )
            return

        if not self._python_lang.parser.is_syntax_valid(script):
            logger.warning(
                "PropertyTestInitializer: gen_inputs produced invalid Python after retries. "
                "Input-generator script omitted; property tests may not run effectively."
            )
            return

        self.io_pair_cache.store_generator_script(script)
        logger.debug("PropertyTestInitializer: generator script cached.")

    @llm_retry((LLMGenerationError, LLMSyntaxError, ValueError))
    def _call_gen_inputs(self, problem: Problem) -> str:
        prompt = self.prompt_manager.render_prompt(
            "operators/property/gen_inputs.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)
        blocks = self._python_lang.parser.extract_code_blocks(response)
        if not blocks:
            raise LLMGenerationError("gen_inputs: no code block found in LLM response.")
        script = blocks[0]
        script = self._python_lang.parser.remove_main_block(script)
        if not self._python_lang.parser.is_syntax_valid(script):
            raise LLMSyntaxError("gen_inputs produced invalid Python")
        return script

    # ── LLM call 2 ───────────────────────────────────────────────────────────

    def _generate_property_tests(self, problem: Problem) -> list[TestIndividual]:
        try:
            candidates = self._call_gen_property(problem)
        except Exception as exc:
            logger.warning(
                f"PropertyTestInitializer: gen_property LLM call failed: {exc}."
            )
            return []

        python_sandbox = create_sandbox(self._python_sandbox_config)

        transformed_tests = transform_public_tests(
            problem.public_test_cases, problem.starter_code, self.parser
        )

        individuals: list[TestIndividual] = []

        for snippet, description in candidates:
            try:
                valid = validate_property_test(
                    snippet, transformed_tests, python_sandbox
                )
            except Exception as exc:
                logger.debug(
                    f"PropertyTestInitializer: validate_property_test raised {exc}"
                )
                continue

            if not valid:
                logger.debug("PropertyTestInitializer: snippet rejected by validation.")
                continue

            individuals.append(
                TestIndividual(
                    snippet=snippet,
                    probability=self.pop_config.initial_prior,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    explanation=description,
                    metadata={
                        "pruning": "passed_public_io",
                        "description": description,
                    },
                )
            )

        logger.debug(
            f"PropertyTestInitializer: {len(individuals)}/{len(candidates)} "
            "snippets passed validation."
        )
        return individuals

    @llm_retry((LLMGenerationError, ValueError))
    def _call_gen_property(self, problem: Problem) -> list[tuple[str, str]]:
        """Two-stage generation: 1. Describe properties, 2. Convert to code."""
        # Stage 1: Brainstorm descriptions
        descriptions = self._call_describe_properties(problem)
        if not descriptions:
            return []

        logger.debug(
            f"PropertyTestInitializer: Generated {len(descriptions)} descriptions."
        )
        logger.trace(f"Descriptions: {descriptions}")

        # Stage 2: Convert each description to code in parallel
        results: list[tuple[str, str]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.llm_workers
        ) as executor:
            future_to_desc = {
                executor.submit(self._call_convert_to_property, desc, problem): desc
                for desc in descriptions
            }
            for future in concurrent.futures.as_completed(future_to_desc):
                desc = future_to_desc[future]
                try:
                    snippet = future.result()
                    results.append((snippet, desc))
                except Exception as exc:
                    logger.warning(
                        f"PropertyTestInitializer: Failed to convert description to code: {exc}"
                    )

        return results

    @llm_retry((LLMGenerationError, ValueError))
    def _call_describe_properties(self, problem: Problem) -> list[str]:
        prompt = self.prompt_manager.render_prompt(
            "operators/property/describe_properties.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)
        # Extract content between <property_description> tags
        descriptions = re.findall(
            r"<property_description>(.*?)</property_description>", response, re.DOTALL
        )
        if not descriptions:
            raise LLMGenerationError(
                "PropertyTestInitializer: No property descriptions found in LLM response."
            )
        return [d.strip() for d in descriptions if d.strip()]

    @llm_retry((LLMGenerationError, LLMSyntaxError, ValueError))
    def _call_convert_to_property(self, description: str, problem: Problem) -> str:
        prompt = self.prompt_manager.render_prompt(
            "operators/property/convert_description_to_property.j2",
            description=description,
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)
        blocks = self.parser.extract_code_blocks(response)
        if not blocks:
            raise LLMGenerationError(
                "convert_to_property: no code blocks found in LLM response."
            )

        block = blocks[0]
        block = self.parser.remove_main_block(block)
        if not self.parser.is_syntax_valid(block):
            raise LLMSyntaxError("convert_to_property: invalid syntax")

        return block


__all__ = ["PropertyTestInitializer"]
