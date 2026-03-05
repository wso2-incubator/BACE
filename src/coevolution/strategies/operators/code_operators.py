"""Concrete Code Operator implementations.

Each operator is self-sufficient:
- Inherits LLM service from BaseEvolutionaryOperator / BaseLLMInitializer
- Performs parent selection, LLM generation, probability assignment,
  and CodeIndividual construction inside execute() / initialize().

Operators:
- CodeMutationOperator   — mutates a single parent
- CodeCrossoverOperator  — combines two parents
- CodeEditOperator       — edits parent guided by failing tests
- CodeInitializer        — creates Gen-0 individuals (standard + planning mode)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol, cast

from loguru import logger

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    CoevolutionContext,
    PopulationConfig,
    Problem,
)
from coevolution.core.interfaces.language import ILanguage
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy
from infrastructure.code_preprocessing.exceptions import (
    CodeParsingError,
    CodeTransformationError,
)

from .base_llm_service import (
    BaseEvolutionaryOperator,
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    llm_retry,
)

type TestPopulationType = str


class IFailingTestSelector(Protocol):
    """Protocol for selecting failing tests for code individuals."""

    @staticmethod
    def select_k_failing_tests(
        coevolution_context: CoevolutionContext,
        code_individual: CodeIndividual,
        k: int = 10,
    ) -> list[tuple[TestIndividual, TestPopulationType]]: ...


# ---------------------------------------------------------------------------
# Shared helpers mixin
# ---------------------------------------------------------------------------

class _CodeLLMHelpers:
    language_adapter: ILanguage  # provided by BaseLLMService

    def _extract_all_code_blocks(self, response: str) -> list[str]:
        return self.language_adapter.extract_code_blocks(response)

    def _contains_starter_code(self, code: str, starter_code: str) -> bool:
        return self.language_adapter.contains_starter_code(code, starter_code)

    def _validated_code(self, code: str, starter_code: str, op: str) -> str:
        if not self._contains_starter_code(code, starter_code):
            raise ValueError(f"{op} result does not contain starter code structure.")
        return code


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

class CodeMutationOperator(_CodeLLMHelpers, BaseEvolutionaryOperator[CodeIndividual]):
    """Mutation: select one parent → LLM rephrase → new CodeIndividual."""

    def operation_name(self) -> str:
        return OPERATION_MUTATION

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        code_pop = context.code_population
        problem = context.problem

        parents = self.parent_selector.select_parents(code_pop, 1, context)
        if not parents:
            logger.warning("CodeMutationOperator: no parents available")
            return []
        parent = parents[0]

        prompt = self.prompt_manager.render_prompt(
            "operators/code/mutate.j2",
            question_content=problem.question_content,
            individual=parent.snippet,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)
        mutated_code = self._extract_code_block(response)
        mutated_code = self._validated_code(mutated_code, problem.starter_code, "mutation")

        probability = self.prob_assigner.assign_probability(
            OPERATION_MUTATION, [parent.probability]
        )
        return [
            CodeIndividual(
                snippet=mutated_code,
                probability=probability,
                creation_op=OPERATION_MUTATION,
                generation_born=code_pop.generation + 1,
                parents={"code": [parent.id], "test": []},
            )
        ]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class CodeCrossoverOperator(_CodeLLMHelpers, BaseEvolutionaryOperator[CodeIndividual]):
    """Crossover: select two parents → LLM combine → new CodeIndividual."""

    def operation_name(self) -> str:
        return OPERATION_CROSSOVER

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        code_pop = context.code_population
        problem = context.problem

        parents = self.parent_selector.select_parents(code_pop, 2, context)
        if len(parents) < 2:
            logger.warning("CodeCrossoverOperator: need 2 parents, got fewer")
            return []
        p1, p2 = parents[0], parents[1]

        prompt = self.prompt_manager.render_prompt(
            "operators/code/crossover.j2",
            question_content=problem.question_content,
            parent1=p1.snippet,
            parent2=p2.snippet,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)
        child_code = self._extract_code_block(response)
        child_code = self._validated_code(child_code, problem.starter_code, "crossover")

        probability = self.prob_assigner.assign_probability(
            OPERATION_CROSSOVER, [p1.probability, p2.probability]
        )
        return [
            CodeIndividual(
                snippet=child_code,
                probability=probability,
                creation_op=OPERATION_CROSSOVER,
                generation_born=code_pop.generation + 1,
                parents={"code": [p1.id, p2.id], "test": []},
            )
        ]


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------

class CodeEditOperator(_CodeLLMHelpers, BaseEvolutionaryOperator[CodeIndividual]):
    """Edit: select parent + failing tests → targeted LLM fix → new CodeIndividual."""

    def __init__(
        self,
        llm: ILanguageModel,
        language_adapter: ILanguage,
        parent_selector: IParentSelectionStrategy[CodeIndividual],
        prob_assigner: IProbabilityAssigner,
        failing_test_selector: IFailingTestSelector,
        k_failing_tests: int = 10,
    ) -> None:
        super().__init__(llm, language_adapter, parent_selector, prob_assigner)
        self.failing_test_selector = failing_test_selector
        self.k_failing_tests = k_failing_tests

    def operation_name(self) -> str:
        return OPERATION_EDIT

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        code_pop = context.code_population
        problem = context.problem

        parents = self.parent_selector.select_parents(code_pop, 1, context)
        if not parents:
            logger.warning("CodeEditOperator: no parents available")
            return []
        parent = parents[0]

        failing = self.failing_test_selector.select_k_failing_tests(
            context, parent, k=self.k_failing_tests
        )
        if not failing:
            logger.debug("CodeEditOperator: no failing tests for this parent, skipping")
            return []

        feedback_parts = []
        for idx, (test_ind, _pop_type) in enumerate(failing, start=1):
            # Retrieve the error trace from the interaction store
            exec_result = context.interactions["unittest"].execution_results
            trace = "No trace available"
            if parent.id in exec_result and test_ind.id in exec_result[parent.id]:
                trace = exec_result[parent.id][test_ind.id].error_log or trace
            feedback_parts.append(
                f"Failing Test #{idx}:\n{test_ind.snippet}\n\nError Trace:\n{trace}"
            )
        feedback = "\n\n" + "=" * 80 + "\n\n".join(feedback_parts)

        prompt = self.prompt_manager.render_prompt(
            "operators/code/edit_multiple.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            individual=parent.snippet,
            feedback=feedback,
        )
        response = self._generate(prompt)
        edited_code = self._extract_code_block(response)
        edited_code = self._validated_code(edited_code, problem.starter_code, "edit")

        probability = self.prob_assigner.assign_probability(
            OPERATION_EDIT, [parent.probability]
        )
        return [
            CodeIndividual(
                snippet=edited_code,
                probability=probability,
                creation_op=OPERATION_EDIT,
                generation_born=code_pop.generation + 1,
                parents={
                    "code": [parent.id],
                    "test": [t.id for t, _ in failing],
                },
                metadata={"num_failing_tests": len(failing)},
            )
        ]


# ---------------------------------------------------------------------------
# Initializer
# ---------------------------------------------------------------------------

class CodeInitializer(_CodeLLMHelpers, BaseLLMInitializer[CodeIndividual]):
    """Creates Gen-0 code individuals via batched LLM calls.

    Supports both standard and planning modes.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        language_adapter: ILanguage,
        pop_config: PopulationConfig,
        init_batch_size: int = 2,
        llm_workers: int = 4,
        planning_enabled: bool = False,
    ) -> None:
        super().__init__(llm, language_adapter, pop_config)
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

    # ------------------------------------------------------------------

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
                    raise ValueError(f"Expected {batch_size} code blocks, got {len(blocks)}")
                return blocks

        logger.info(f"CodeInitializer: initializing {target} individuals in {num_batches} batches")
        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            futures = [executor.submit(_generate_batch, self.init_batch_size) for _ in range(num_batches)]
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

    def _init_with_planning(self, problem: Problem, target: int) -> list[CodeIndividual]:
        """Two-phase: plan per individual, then code from plan."""

        @llm_retry((ValueError, CodeParsingError, CodeTransformationError, LLMGenerationError))
        def _make_plan() -> str:
            prompt = self.prompt_manager.render_prompt(
                "operators/code/plan_generate.j2",
                question_content=problem.question_content,
                starter_code=problem.starter_code,
            )
            return self._generate(prompt).strip()

        @llm_retry((ValueError, CodeParsingError, CodeTransformationError, LLMGenerationError))
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


__all__ = [
    "CodeMutationOperator",
    "CodeCrossoverOperator",
    "CodeEditOperator",
    "CodeInitializer",
    "IFailingTestSelector",
]
