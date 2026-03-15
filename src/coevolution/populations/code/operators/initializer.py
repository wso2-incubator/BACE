"""Code initializers — create Gen-0 code individuals via LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

from loguru import logger

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    LanguageParsingError,
    LanguageTransformationError,
    PopulationConfig,
    Problem,
)
from coevolution.core.interfaces.language import ICodeParser
from coevolution.strategies.llm_base import (
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    LLMSyntaxError,
    llm_retry,
)

from ._helpers import _CodeLLMHelpers


class BaseCodeInitializer(_CodeLLMHelpers, BaseLLMInitializer[CodeIndividual], ABC):
    """Base class for code population initializers."""

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
        llm_workers: int = 4,
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config)
        self.llm_workers = llm_workers

    @abstractmethod
    def initialize(self, problem: Problem) -> list[CodeIndividual]:
        """Create Gen-0 code individuals."""
        ...


class StandardCodeInitializer(BaseCodeInitializer):
    """Creates Gen-0 code individuals via batched LLM calls."""

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
        init_batch_size: int = 2,
        llm_workers: int = 4,
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config, llm_workers)
        self.init_batch_size = min(
            init_batch_size, pop_config.initial_population_size or 1
        )

    def initialize(self, problem: Problem) -> list[CodeIndividual]:
        target = self.pop_config.initial_population_size
        individuals: list[CodeIndividual] = []
        num_batches = (target + self.init_batch_size - 1) // self.init_batch_size

        def _generate_batch(batch_size: int) -> list[str]:
            if batch_size == 1:
                prompt = self.prompt_manager.render_prompt(
                    "operators/code/initial_single.j2",
                    question_content=problem.question_content,
                    starter_code=problem.starter_code,
                )
                response = self._generate(prompt)
                code = self._extract_code_block(response)
                code = self._validated_code(code, problem.starter_code, "initial")
                return [code]
            else:
                prompt = self.prompt_manager.render_prompt(
                    "operators/code/initial_population.j2",
                    question_content=problem.question_content,
                    starter_code=problem.starter_code,
                    population_size=batch_size,
                )
                response = self._generate(prompt)
                blocks = self._extract_all_code_blocks(response)
                validated_blocks = []
                for b in blocks:
                    validated_blocks.append(
                        self._validated_code(b, problem.starter_code, "initial")
                    )
                if len(validated_blocks) != batch_size:
                    raise ValueError(
                        f"Expected {batch_size} code blocks, got {len(validated_blocks)}"
                    )
                return validated_blocks

        logger.info(
            f"StandardCodeInitializer: initializing {target} individuals in {num_batches} batches"
        )
        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            futures = [
                executor.submit(_generate_batch, self.init_batch_size)
                for _ in range(num_batches)
            ]
            for future in as_completed(futures):
                try:
                    snippets = future.result()
                    for snip in snippets:
                        individuals.append(
                            CodeIndividual(
                                snippet=snip,
                                probability=self.pop_config.initial_prior,
                                creation_op=OPERATION_INITIAL,
                                generation_born=0,
                                explanation=self.parser.get_docstring(snip),
                            )
                        )
                except Exception as e:
                    logger.error(f"Batch init failed: {e}")

        if not individuals:
            raise RuntimeError("StandardCodeInitializer: failed to generate any individuals")
        return individuals[:target]


class PlanningCodeInitializer(BaseCodeInitializer):
    """Two-phase: plan per individual, then code from plan."""

    def initialize(self, problem: Problem) -> list[CodeIndividual]:
        target = self.pop_config.initial_population_size

        @llm_retry(
            (
                ValueError,
                LanguageParsingError,
                LanguageTransformationError,
                LLMGenerationError,
                LLMSyntaxError,
            )
        )
        def _make_plan() -> str:
            prompt = self.prompt_manager.render_prompt(
                "operators/code/plan_generate.j2",
                question_content=problem.question_content,
                starter_code=problem.starter_code,
            )
            return self._generate(prompt).strip()

        @llm_retry(
            (
                ValueError,
                LanguageParsingError,
                LanguageTransformationError,
                LLMGenerationError,
                LLMSyntaxError,
            )
        )
        def _code_from_plan(plan: str) -> tuple[str, str]:
            prompt = self.prompt_manager.render_prompt(
                "operators/code/plan_to_code.j2",
                question_content=problem.question_content,
                plan=plan,
                starter_code=problem.starter_code,
            )
            response = self._generate(prompt)
            code = self._extract_code_block(response)
            code = self._validated_code(code, problem.starter_code, "plan_to_code")
            return plan, code

        # Phase A: plans
        logger.info(f"PlanningCodeInitializer: generating {target} plans")
        plans: list[str] = []
        with ThreadPoolExecutor(max_workers=self.llm_workers) as ex:
            for f in as_completed([ex.submit(_make_plan) for _ in range(target)]):
                try:
                    plans.append(f.result())
                except Exception as e:
                    logger.error(f"Plan generation failed: {e}")

        if not plans:
            raise RuntimeError("PlanningCodeInitializer: no plans generated")

        # Phase B: code from plans
        logger.info(f"PlanningCodeInitializer: generating code for {len(plans)} plans")
        individuals: list[CodeIndividual] = []
        with ThreadPoolExecutor(max_workers=self.llm_workers) as ex:
            for f in as_completed([ex.submit(_code_from_plan, p) for p in plans]):  # type: ignore[assignment]
                try:
                    plan, code = cast(tuple[str, str], f.result())
                    individuals.append(
                        CodeIndividual(
                            snippet=code,
                            probability=self.pop_config.initial_prior,
                            creation_op=OPERATION_INITIAL,
                            generation_born=0,
                            explanation=plan,
                            metadata={"plan": plan},
                        )
                    )
                except Exception as e:
                    logger.error(f"Plan-to-code failed: {e}")

        if not individuals:
            raise RuntimeError("PlanningCodeInitializer: no individuals generated")
        return individuals[:target]


__all__ = ["BaseCodeInitializer", "StandardCodeInitializer", "PlanningCodeInitializer"]
