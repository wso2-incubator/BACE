"""Concrete Unittest Operator implementations.

Each operator is self-sufficient:
- Inherits LLM service from BaseEvolutionaryOperator / BaseLLMInitializer
- Performs parent selection, interaction-context lookup, LLM generation,
  probability assignment, and TestIndividual construction inside execute().

Operators:
- UnittestMutationOperator  — mutates a single test
- UnittestCrossoverOperator — combines two tests
- UnittestEditOperator      — edits test guided by passing/failing code pairs
- UnittestInitializer       — creates Gen-0 test individuals
"""

from __future__ import annotations

import random

from loguru import logger

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    CoevolutionContext,
    PopulationConfig,
    Problem,
)
from coevolution.core.interfaces.language import (
    ILanguage,
    LanguageParsingError,
    LanguageTransformationError,
)

from .base_llm_service import (
    BaseEvolutionaryOperator,
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    llm_retry,
)


# ---------------------------------------------------------------------------
# Shared extraction helpers
# ---------------------------------------------------------------------------

class _TestLLMHelpers:
    """Mixin: test-specific extraction utilities."""

    language_adapter: ILanguage  # provided by BaseLLMService

    def _extract_test_functions(self, code: str) -> list[str]:
        return self.language_adapter.split_tests(code)

    def _extract_first_test_function(self, code: str) -> str:
        functions = self._extract_test_functions(code)
        if not functions:
            raise LanguageParsingError("No test functions found in generated code")
        return functions[0]


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

class UnittestMutationOperator(_TestLLMHelpers, BaseEvolutionaryOperator[TestIndividual]):
    """Mutation: select one parent test → LLM rephrase → new TestIndividual."""

    def operation_name(self) -> str:
        return OPERATION_MUTATION

    @llm_retry((ValueError, LanguageParsingError, LanguageTransformationError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        test_pop = context.test_populations["unittest"]
        parents = self.parent_selector.select_parents(test_pop, 1, context)
        if not parents:
            logger.warning("UnittestMutationOperator: no parents available")
            return []
        parent = parents[0]

        prompt = self.prompt_manager.render_prompt(
            "operators/unittest/mutate.j2",
            question_content=context.problem.question_content,
            individual=parent.snippet,
        )
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        mutated = self._extract_first_test_function(extracted)

        probability = self.prob_assigner.assign_probability(
            OPERATION_MUTATION, [parent.probability]
        )
        return [
            TestIndividual(
                snippet=mutated,
                probability=probability,
                creation_op=OPERATION_MUTATION,
                generation_born=test_pop.generation + 1,
                parents={"code": [], "test": [parent.id]},
            )
        ]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class UnittestCrossoverOperator(_TestLLMHelpers, BaseEvolutionaryOperator[TestIndividual]):
    """Crossover: combine two parent tests → new TestIndividual."""

    def operation_name(self) -> str:
        return OPERATION_CROSSOVER

    @llm_retry((ValueError, LanguageParsingError, LanguageTransformationError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        test_pop = context.test_populations["unittest"]
        parents = self.parent_selector.select_parents(test_pop, 2, context)
        if len(parents) < 2:
            logger.warning("UnittestCrossoverOperator: need 2 parents, got fewer")
            return []
        p1, p2 = parents[0], parents[1]

        prompt = self.prompt_manager.render_prompt(
            "operators/unittest/crossover.j2",
            question_content=context.problem.question_content,
            parent1=p1.snippet,
            parent2=p2.snippet,
        )
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        child = self._extract_first_test_function(extracted)

        probability = self.prob_assigner.assign_probability(
            OPERATION_CROSSOVER, [p1.probability, p2.probability]
        )
        return [
            TestIndividual(
                snippet=child,
                probability=probability,
                creation_op=OPERATION_CROSSOVER,
                generation_born=test_pop.generation + 1,
                parents={"code": [], "test": [p1.id, p2.id]},
            )
        ]


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------

class UnittestEditOperator(_TestLLMHelpers, BaseEvolutionaryOperator[TestIndividual]):
    """Edit: improve a test's discriminating power using passing/failing code context.

    Three edit modes (auto-selected by what interaction data is available):
    - discriminating  : has passing AND failing code → edit to discriminate harder
    - all-failing     : no passing code → edit to be more lenient/correct
    - all-passing     : no failing code → edit to break one of the passing ones
    """

    def operation_name(self) -> str:
        return OPERATION_EDIT

    @llm_retry((ValueError, LanguageParsingError, LanguageTransformationError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        test_pop = context.test_populations["unittest"]
        code_pop = context.code_population
        interactions = context.interactions["unittest"]

        # 1. Select parent test
        parents = self.parent_selector.select_parents(test_pop, 1, context)
        if not parents:
            return []
        parent = parents[0]

        # 2. Look up which code individuals pass/fail this test
        parent_test_idx = test_pop.get_index_of_individual(parent)
        if parent_test_idx == -1:
            logger.error(f"Parent test ID {parent.id} not found in test pop index")
            return []

        if interactions.observation_matrix.size == 0:
            logger.debug("Interaction matrix empty — skipping edit")
            return []

        test_col = interactions.observation_matrix[:, parent_test_idx]
        passing_indices = random.sample(
            [i for i, r in enumerate(test_col) if r == 1],
            min(2, sum(1 for r in test_col if r == 1)),
        )
        failing_indices = random.sample(
            [i for i, r in enumerate(test_col) if r == 0],
            min(2, sum(1 for r in test_col if r == 0)),
        )

        if not passing_indices and not failing_indices:
            logger.debug(f"No code interaction data for test {parent.id}")
            return []

        passing_inds = [code_pop[i] for i in passing_indices]
        failing_inds = [code_pop[i] for i in failing_indices]

        error_traces = [
            interactions.execution_results[fi.id][parent.id].error_log or "No trace available"
            for fi in failing_inds
        ]

        # 3. Choose the right prompt
        if passing_inds and failing_inds:
            edit_type = "discriminating"
            prompt = self.prompt_manager.render_prompt(
                "operators/unittest/edit_discriminating.j2",
                question_content=context.problem.question_content,
                current_test_snippet=parent.snippet,
                passing_code_snippet=passing_inds[0].snippet,
                failing_code_snippet=failing_inds[0].snippet,
                failing_code_trace=error_traces[0],
            )
        elif not passing_inds:
            edit_type = "all-failing"
            if len(failing_inds) < 2:
                logger.debug("Not enough failing inds for all-failing edit")
                return []
            prompt = self.prompt_manager.render_prompt(
                "operators/unittest/edit_all_failing.j2",
                question_content=context.problem.question_content,
                current_test_snippet=parent.snippet,
                failing_code_snippet_P=failing_inds[0].snippet,
                failing_code_trace_P=error_traces[0],
                failing_code_snippet_Q=failing_inds[1].snippet,
                failing_code_trace_Q=error_traces[1],
            )
        else:
            edit_type = "all-passing"
            if len(passing_inds) < 2:
                logger.debug("Not enough passing inds for all-passing edit")
                return []
            prompt = self.prompt_manager.render_prompt(
                "operators/unittest/edit_all_passing.j2",
                question_content=context.problem.question_content,
                current_test_snippet=parent.snippet,
                passing_code_snippet_P=passing_inds[0].snippet,
                passing_code_snippet_Q=passing_inds[1].snippet,
            )

        logger.debug(f"UnittestEditOperator: using '{edit_type}' edit mode")
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        edited = self._extract_first_test_function(extracted)

        probability = self.prob_assigner.assign_probability(
            OPERATION_EDIT, [parent.probability]
        )
        return [
            TestIndividual(
                snippet=edited,
                probability=probability,
                creation_op=OPERATION_EDIT,
                generation_born=test_pop.generation + 1,
                parents={
                    "code": [ind.id for ind in passing_inds + failing_inds],
                    "test": [parent.id],
                },
                metadata={"edit_type": edit_type},
            )
        ]


# ---------------------------------------------------------------------------
# Initializer
# ---------------------------------------------------------------------------

class UnittestInitializer(_TestLLMHelpers, BaseLLMInitializer[TestIndividual]):
    """Creates Gen-0 test individuals via LLM.

    Asks for `population_size` tests in one shot and recovers
    gracefully if the count doesn't match.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        language_adapter: ILanguage,
        pop_config: PopulationConfig,
        llm_workers: int = 1,
    ) -> None:
        super().__init__(llm, language_adapter, pop_config)
        self.llm_workers = llm_workers

    def initialize(self, problem: Problem) -> list[TestIndividual]:
        target = self.pop_config.initial_population_size
        test_functions = self._generate_test_functions(problem, target)

        individuals: list[TestIndividual] = []
        for fn in test_functions:
            individuals.append(
                TestIndividual(
                    snippet=fn,
                    probability=self.pop_config.initial_prior,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                )
            )
        logger.debug(f"UnittestInitializer: created {len(individuals)} individuals")
        return individuals

    @llm_retry((ValueError, LanguageParsingError, LanguageTransformationError, LLMGenerationError))
    def _generate_test_functions(self, problem: Problem, target: int) -> list[str]:
        prompt = self.prompt_manager.render_prompt(
            "operators/unittest/initial.j2",
            population_size=target,
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)

        code_blocks = self.language_adapter.extract_code_blocks(response)
        test_functions: list[str] = []
        for block in code_blocks:
            test_functions.extend(self._extract_test_functions(block))

        # Trim over-generation
        if len(test_functions) > target:
            test_functions = test_functions[:target]

        # Top-up under-generation with an additional LLM call
        if len(test_functions) < target:
            additional = target - len(test_functions)
            logger.info(f"UnittestInitializer: topping up {additional} more tests")
            extra_prompt = self.prompt_manager.render_prompt(
                "operators/unittest/initial.j2",
                population_size=additional,
                question_content=problem.question_content,
                starter_code=problem.starter_code,
            )
            extra_response = self._generate(extra_prompt)
            for block in self.language_adapter.extract_code_blocks(extra_response):
                test_functions.extend(self._extract_test_functions(block))
                if len(test_functions) >= target:
                    break
            test_functions = test_functions[:target]

        return test_functions


__all__ = [
    "UnittestMutationOperator",
    "UnittestCrossoverOperator",
    "UnittestEditOperator",
    "UnittestInitializer",
]
