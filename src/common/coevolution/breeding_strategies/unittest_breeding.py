"""
Concrete UnittestBreedingStrategy.

This class implements the specific handlers for unittest operations (Selection -> Execution -> Construction).
It inherits the robust parallel breeding loop, circuit breakers, and batching logic from BaseBreedingStrategy.
"""

from __future__ import annotations

import random
from typing import Protocol

from loguru import logger

from ..core.individual import TestIndividual
from ..core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    CoevolutionContext,
    InitialInput,
    IParentSelectionStrategy,
    IProbabilityAssigner,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
)
from ..operators.unittest_llm_operator import (
    UnittestCrossoverInput,
    UnittestEditInput,
    UnittestLLMOperator,
    UnittestMutationInput,
)
from .base_breeding import BaseBreedingStrategy


class ITestFeedbackSelector(Protocol):
    """Protocol for generating feedback for test edits."""

    def select_feedback(
        self,
        coevolution_context: CoevolutionContext,
        test_individual: TestIndividual,
    ) -> str | None:
        """
        Generate feedback for editing a test case.
        Could be based on code coverage, redundancy, or ability to kill mutants.
        """
        ...


class UnittestBreedingStrategy(BaseBreedingStrategy[TestIndividual]):
    """
    Concrete UnittestBreedingStrategy using a UnittestLLMOperator.
    """

    def __init__(
        self,
        operator: UnittestLLMOperator,
        op_rates_config: OperatorRatesConfig,
        pop_config: PopulationConfig,
        probability_assigner: IProbabilityAssigner,
        parent_selector: IParentSelectionStrategy[TestIndividual],
        feedback_selector: ITestFeedbackSelector,
        max_workers: int = 1,
    ) -> None:
        # Initialize Base Class (sets up op_rates, max_workers, and strategies dict)
        super().__init__(op_rates_config, max_workers)

        self.operator = operator
        self.pop_config = pop_config
        self.probability_assigner = probability_assigner
        self.parent_selector = parent_selector
        self.feedback_selector = feedback_selector

        # Validate operations
        for op in self.op_rates_config.operation_rates.keys():
            if op not in self.operator.supported_operations():
                raise ValueError(
                    f"Operation '{op}' in rates config is not supported by the operator."
                )

        # Register Handlers
        # These are used by the BaseBreedingStrategy's robust loop
        self._strategies = {
            OPERATION_MUTATION: self._breed_via_mutation,
            OPERATION_CROSSOVER: self._breed_via_crossover,
            OPERATION_EDIT: self._breed_via_edit,
        }

    def initialize_individuals(
        self, problem: Problem
    ) -> tuple[list[TestIndividual], str | None]:
        """Create initial individuals via LLM."""

        initial_outputs, context_code = self.operator.generate_initial_snippets(
            InitialInput(
                operation=OPERATION_INITIAL,
                question_content=problem.question_content,
                starter_code=problem.starter_code,
                population_size=self.pop_config.initial_population_size,
            )
        )
        individuals: list[TestIndividual] = []

        # Robustness check for initial generation
        if not initial_outputs or not initial_outputs.results:
            logger.error("No initial unittest snippets generated.")
            return [], None

        for operator_result in initial_outputs.results:
            individual = TestIndividual(
                snippet=operator_result.snippet,
                probability=self.pop_config.initial_prior,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
            )
            individuals.append(individual)

        return individuals, context_code

    # --- Handlers (Selection -> Execution -> Construction) ---

    def _breed_via_mutation(self, context: CoevolutionContext) -> list[TestIndividual]:
        """Full-cycle handler for Mutation."""
        parents = self.parent_selector.select_parents(
            context.test_populations["unittest"], count=1, coevolution_context=context
        )
        if not parents:
            return []
        parent = parents[0]

        dto = UnittestMutationInput(
            operation=OPERATION_MUTATION,
            question_content=context.problem.question_content,
            parent_snippet=parent.snippet,
        )

        try:
            output = self.operator.apply(dto)
        except Exception as e:
            logger.warning(f"Unittest mutation operator failed: {e}")
            return []

        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_MUTATION,
                parent_probs=[parent.probability],
                initial_prior=self.pop_config.initial_prior,
            )
            offspring.append(
                TestIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_MUTATION,
                    generation_born=context.test_populations["unittest"].generation + 1,
                    parents={"code": [], "test": [parent.id]},
                    metadata=res.metadata,
                )
            )
        return offspring

    def _breed_via_crossover(self, context: CoevolutionContext) -> list[TestIndividual]:
        """Full-cycle handler for Crossover."""
        parents = self.parent_selector.select_parents(
            context.test_populations["unittest"], count=2, coevolution_context=context
        )
        if len(parents) < 2:
            return []
        p1, p2 = parents[0], parents[1]

        dto = UnittestCrossoverInput(
            operation=OPERATION_CROSSOVER,
            question_content=context.problem.question_content,
            parent1_snippet=p1.snippet,
            parent2_snippet=p2.snippet,
        )

        try:
            output = self.operator.apply(dto)
        except Exception as e:
            logger.warning(f"Unittest crossover operator failed: {e}")
            return []

        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_CROSSOVER,
                parent_probs=[p1.probability, p2.probability],
                initial_prior=self.pop_config.initial_prior,
            )
            offspring.append(
                TestIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_CROSSOVER,
                    generation_born=context.test_populations["unittest"].generation + 1,
                    parents={"code": [], "test": [p1.id, p2.id]},
                    metadata=res.metadata,
                )
            )
        return offspring

    def _breed_via_edit(self, context: CoevolutionContext) -> list[TestIndividual]:
        """
        Full-cycle handler for Edit (Discrimination-driven).
        Fixes logic errors regarding variable scope.
        """
        # 1. Select Parent Test
        parents = self.parent_selector.select_parents(
            context.test_populations["unittest"], count=1, coevolution_context=context
        )
        if not parents:
            return []
        parent = parents[0]

        # 2. Get Interaction Data
        interactions = context.interactions["unittest"]
        test_pop = context.test_populations["unittest"]
        code_pop = context.code_population

        parent_test_idx = test_pop.get_index_of_individual(parent)
        if parent_test_idx == -1:
            logger.error(f"Parent test individual ID {parent.id} not found.")
            return []

        if interactions.observation_matrix.size == 0:
            logger.error("Interaction observation matrix is empty.")
            return []

        test_results_col = interactions.observation_matrix[:, parent_test_idx]

        # 3. Identify Candidates
        passing_code_indices = [i for i, res in enumerate(test_results_col) if res == 1]
        failing_code_indices = [i for i, res in enumerate(test_results_col) if res == 0]

        code_parent_ids: list[str] = []

        passing_snippet = "No passing code available."
        failing_snippet = "No failing code available."
        error_trace = "No error trace available."

        if passing_code_indices:
            passing_code_idx = random.choice(passing_code_indices)  # Use random.choice
            passing_code_ind = code_pop[passing_code_idx]
            passing_snippet = passing_code_ind.snippet
            code_parent_ids.append(passing_code_ind.id)

        if failing_code_indices:
            failing_code_idx = random.choice(failing_code_indices)  # Use random.choice
            failing_code_ind = code_pop[failing_code_idx]
            failing_snippet = failing_code_ind.snippet
            code_parent_ids.append(failing_code_ind.id)

            # Get Error Trace
            try:
                result = interactions.execution_results[
                    failing_code_ind.id
                ].test_results[parent.id]
                error_trace = result.details or "No error trace available."
            except (KeyError, AttributeError):
                logger.error(
                    f"Error trace not found for code ID {failing_code_ind.id} "
                    f"and test ID {parent.id}."
                )
                raise ValueError(
                    f"Error trace not found for code ID {failing_code_ind.id} "
                    f"and test ID {parent.id}."
                )

        # 4. Prepare DTO
        dto = UnittestEditInput(
            operation=OPERATION_EDIT,
            question_content=context.problem.question_content,
            parent_snippet=parent.snippet,
            passing_code_snippet=passing_snippet,
            failing_code_snippet=failing_snippet,
            failing_code_trace=error_trace,
        )

        try:
            output = self.operator.apply(dto)
        except Exception as e:
            logger.warning(f"Unittest edit operator failed: {e}")
            return []

        # 5. Construct Offspring
        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_EDIT,
                parent_probs=[parent.probability],
                initial_prior=self.pop_config.initial_prior,
            )
            offspring.append(
                TestIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_EDIT,
                    generation_born=context.test_populations["unittest"].generation + 1,
                    parents={"code": code_parent_ids, "test": [parent.id]},
                    metadata=res.metadata,
                )
            )
        return offspring


__all__ = ["UnittestBreedingStrategy"]
