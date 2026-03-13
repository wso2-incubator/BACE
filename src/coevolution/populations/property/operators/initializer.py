"""PropertyTestInitializer — creates Gen-0 property test individuals."""

from __future__ import annotations

from dataclasses import replace

from loguru import logger

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import OPERATION_INITIAL, PopulationConfig, Problem
from coevolution.core.interfaces.language import (
    ICodeParser,
    LanguageTransformationError,
)
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

    Call 2: gen_property.j2 → N candidate ``def property_<name>(...)`` snippets.
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
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config)
        self.sandbox_config = sandbox_config
        self.io_pair_cache = io_pair_cache
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

        for snippet in candidates:
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
                    metadata={"pruning": "passed_public_io"},
                )
            )

        logger.debug(
            f"PropertyTestInitializer: {len(individuals)}/{len(candidates)} "
            "snippets passed validation."
        )
        return individuals

    @llm_retry((LLMGenerationError, ValueError))
    def _call_gen_property(self, problem: Problem) -> list[str]:
        prompt = self.prompt_manager.render_prompt(
            "operators/property/gen_property.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)
        blocks = self.parser.extract_code_blocks(response)
        if not blocks:
            raise LLMGenerationError(
                "gen_property: no code blocks found in LLM response."
            )

        candidates: list[str] = []
        for block in blocks:
            try:
                block = self.parser.remove_main_block(block)
                self.parser.is_syntax_valid(block)
                candidates.append(block)
            except (LanguageTransformationError, LLMSyntaxError):
                pass

        return candidates


__all__ = ["PropertyTestInitializer"]
