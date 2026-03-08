"""CodeInitializer — creates Gen-0 code individuals via batched LLM calls."""

from __future__ import annotations

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
    llm_retry,
)

from ._helpers import _CodeLLMHelpers


class CodeInitializer(_CodeLLMHelpers, BaseLLMInitializer[CodeIndividual]):
    """Creates Gen-0 code individuals via batched LLM calls.

    Supports both standard and planning modes.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
        init_batch_size: int = 2,
        llm_workers: int = 4,
        planning_enabled: bool = False,
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config)
        self.init_batch_size = min(
            init_batch_size, pop_config.initial_population_size or 1
        )
        self.llm_workers = llm_workers
        self.planning_enabled = planning_enabled

    def initialize(self, problem: Problem) -> list[CodeIndividual]:
        target = self.pop_config.initial_population_size
        if self.planning_enabled:
            return self._init_with_planning(problem, target)
        return self._init_standard(problem, target)

    def _init_standard(self, problem: Problem, target: int) -> list[CodeIndividual]:
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
                self._validated_code(code, problem.starter_code, "initial")
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
                for b in blocks:
                    self._validated_code(b, problem.starter_code, "initial")
                if len(blocks) != batch_size:
                    raise ValueError(
                        f"Expected {batch_size} code blocks, got {len(blocks)}"
                    )
                return blocks

        logger.info(
            f"CodeInitializer: initializing {target} individuals in {num_batches} batches"
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
                            )
                        )
                except Exception as e:
                    logger.error(f"Batch init failed: {e}")

        if not individuals:
            raise RuntimeError("CodeInitializer: failed to generate any individuals")
        return individuals[:target]

    def _init_with_planning(
        self, problem: Problem, target: int
    ) -> list[CodeIndividual]:
        """Two-phase: plan per individual, then code from plan."""

        @llm_retry(
            (
                ValueError,
                LanguageParsingError,
                LanguageTransformationError,
                LLMGenerationError,
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
            self._validated_code(code, problem.starter_code, "plan_to_code")
            return plan, code

        # Phase A: plans
        plans: list[str] = []
        with ThreadPoolExecutor(max_workers=self.llm_workers) as ex:
            for f in as_completed([ex.submit(_make_plan) for _ in range(target)]):
                try:
                    plans.append(f.result())
                except Exception as e:
                    logger.error(f"Plan generation failed: {e}")

        if not plans:
            raise RuntimeError("CodeInitializer (planning): no plans generated")

        # Phase B: code from plans
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
                            metadata={"plan": plan},
                        )
                    )
                except Exception as e:
                    logger.error(f"Plan-to-code failed: {e}")

        if not individuals:
            raise RuntimeError("CodeInitializer (planning): no individuals generated")
        return individuals[:target]


__all__ = ["CodeInitializer"]
__all__ = ["CodeInitializer"]
