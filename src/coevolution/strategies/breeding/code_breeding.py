"""
This module defines the `CodeBreedingStrategy` class, a concrete implementation of a breeding strategy
for code individuals in a coevolutionary framework. It leverages a language model operator to perform
genetic operations such as mutation, crossover, and edit, facilitating the evolution of code solutions
in response to a given problem and associated test cases.

Classes and Protocols:
----------------------
- IFailingTestSelector (Protocol): Defines the interface for selecting failing tests for a given code individual.
- CodeBreedingStrategy: Implements the main breeding strategy for code individuals, supporting mutation,
    crossover, and edit operations using a provided operator and configuration.

Key Methods:
------------
- __init__: Initializes the strategy with the operator, configuration, probability assigner, parent selector,
    and failing test selector.
- initialize_individuals: Generates the initial population of code individuals using the operator.
- _breed_via_mutation: Produces offspring via mutation of a selected parent.
- _breed_via_crossover: Produces offspring via crossover between two selected parents.
- _breed_via_edit: Produces offspring by editing a parent individual in response to a failing test case.

Usage:
------
This strategy is intended to be used within a coevolutionary algorithm where code and test populations
are evolved together. The strategy relies on external components for parent selection, probability assignment,
and failing test selection, allowing for flexible integration with different evolutionary setups.

Raises:
-------
- ValueError: If the operator rates configuration includes operations not supported by the provided operator.
- RuntimeError: If the operator fails to generate initial code snippets.

Exports:
--------
- CodeBreedingStrategy
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from loguru import logger

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
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

from ..operators.code_llm_operator import (
    CodeCrossoverInput,
    CodeEditInput,
    CodeLLMOperator,
    CodeMutationInput,
)
from .base_breeding import BaseBreedingStrategy

type TestPopulationType = str  # e.g., "unit", "differential", "public"


class IFailingTestSelector(Protocol):
    """Protocol for selecting failing tests for code individuals."""

    @staticmethod
    def select_k_failing_tests(
        coevolution_context: CoevolutionContext,
        code_individual: CodeIndividual,
        k: int = 10,
    ) -> list[tuple[TestIndividual, TestPopulationType]]:
        """Select up to k failing tests for the given code individual.

        Args:
            coevolution_context: Current coevolution context with populations and interactions.
            code_individual: The code individual for which to select failing tests.
            k: Maximum number of failing tests to select.

        Returns:
            A list of tuples (selected_test_individual, test_population_type).
            Empty list if no failing tests are found.
        """
        ...


class CodeBreedingStrategy(BaseBreedingStrategy[CodeIndividual]):
    """
    Concrete CodeBreedingStrategy using a CodeLLMOperator.
    """

    def __init__(
        self,
        operator: CodeLLMOperator,
        op_rates_config: OperatorRatesConfig,
        pop_config: PopulationConfig,
        probability_assigner: IProbabilityAssigner,
        parent_selector: IParentSelectionStrategy[CodeIndividual],
        failing_test_selector: IFailingTestSelector,
        init_pop_batch_size: int = 2,
        llm_workers: int = 1,
        k_failing_tests: int = 10,
    ) -> None:
        """
        Initialize the CodeBreedingStrategy.
        Args:
            operator: The CodeLLMOperator to use for breeding operations.
            op_rates_config: Configuration for operation rates.
            pop_config: Population configuration parameters.
            probability_assigner: Strategy for assigning probabilities to offspring.
            parent_selector: Strategy for selecting parent individuals.
            failing_test_selector: Strategy for selecting failing tests for edit operations.
            init_pop_batch_size: Number of individuals to generate per batch during initialization.
            llm_workers: Maximum number of parallel workers for initialization.
            k_failing_tests: Maximum number of failing tests to select for edit operations (default: 10).
        """
        self.operator = operator
        self.op_rates_config = op_rates_config
        self.pop_config = pop_config
        self.probability_assigner = probability_assigner
        self.parent_selector = parent_selector
        self.failing_test_selector = failing_test_selector
        self.llm_workers = llm_workers
        self.init_pop_batch_size = init_pop_batch_size
        self.k_failing_tests = k_failing_tests

        if pop_config.initial_population_size < init_pop_batch_size:
            self.init_pop_batch_size = pop_config.initial_population_size
            logger.info(
                f"Adjusted init_pop_batch_size to {self.init_pop_batch_size} "
                f"to not exceed initial_population_size."
            )

        # validate operations in rates config are supported by the operator
        for op in self.op_rates_config.operation_rates.keys():
            if op not in self.operator.supported_operations():
                raise ValueError(
                    f"Operation '{op}' in rates config is not supported by the operator."
                )

        self._strategies = {
            OPERATION_MUTATION: self._breed_via_mutation,
            OPERATION_CROSSOVER: self._breed_via_crossover,
            OPERATION_EDIT: self._breed_via_edit,
        }

    def initialize_individuals(self, problem: Problem) -> list[CodeIndividual]:
        """Create initial individuals using parallel execution.

        Submits multiple batch generation tasks to a ThreadPoolExecutor.
        """
        individuals: list[CodeIndividual] = []
        initial_pop_size = self.pop_config.initial_population_size

        # Calculate number of batches needed
        # equivalent to math.ceil(initial_pop_size / pop_batch_size)
        num_batches = (
            initial_pop_size + self.init_pop_batch_size - 1
        ) // self.init_pop_batch_size

        input_dto = InitialInput(
            operation=OPERATION_INITIAL,
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            population_size=self.init_pop_batch_size,
        )

        logger.info(
            f"Initializing population of size {initial_pop_size} in {num_batches} parallel batches."
        )

        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            # Submit all batch tasks at once
            future_to_batch = {
                executor.submit(self.operator.generate_initial_snippets, input_dto): i
                for i in range(num_batches)
            }

            for future in as_completed(future_to_batch):
                try:
                    initial_outputs = future.result()

                    if not initial_outputs or not initial_outputs.results:
                        logger.warning("A batch task returned no results.")
                        continue

                    for operator_result in initial_outputs.results:
                        individual = CodeIndividual(
                            snippet=operator_result.snippet,
                            probability=self.pop_config.initial_prior,
                            creation_op=OPERATION_INITIAL,
                            generation_born=0,
                        )
                        individuals.append(individual)

                    logger.debug(
                        f"Population progress: {len(individuals)}/{initial_pop_size}"
                    )

                except Exception as e:
                    logger.error(f"Error in initialization batch: {e}")

        # Validation: Ensure we generated at least some individuals
        if not individuals:
            logger.error(
                "CodeBreedingStrategy.initialize_individuals: No initial snippets generated"
            )
            raise RuntimeError("Failed to generate initial code snippets")

        # Trim excess individuals if any
        if len(individuals) > initial_pop_size:
            individuals = individuals[:initial_pop_size]

        logger.debug(f"Created {len(individuals)} code individuals (Gen 0)")
        return individuals

    def _breed_via_mutation(self, context: CoevolutionContext) -> list[CodeIndividual]:
        """Breed new individuals via mutation."""
        # 1. Selection (Context is local!)
        parents: list[CodeIndividual] = self.parent_selector.select_parents(
            context.code_population, count=1, coevolution_context=context
        )
        if not parents:
            return []
        parent: CodeIndividual = parents[0]

        # 2. Execution
        dto = CodeMutationInput(
            operation=OPERATION_MUTATION,
            question_content=context.problem.question_content,
            parent_snippet=parent.snippet,
            starter_code=context.problem.starter_code,
        )

        try:
            output = self.operator.apply(dto)
        except Exception as e:
            logger.warning(f"Mutation operator failed: {e}")
            return []

        # 3. Construction
        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_MUTATION,
                parent_probs=[parent.probability],
                initial_prior=self.pop_config.initial_prior,
            )

            offspring.append(
                CodeIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_MUTATION,
                    generation_born=context.code_population.generation + 1,
                    parents={"code": [parent.id], "test": []},
                    metadata=res.metadata,
                )
            )

        return offspring

    def _breed_via_crossover(self, context: CoevolutionContext) -> list[CodeIndividual]:
        # Select two parents
        parents: list[CodeIndividual] = self.parent_selector.select_parents(
            context.code_population, count=2, coevolution_context=context
        )
        if len(parents) < 2:
            logger.warning("Crossover requires two parents, but less were selected.")
            return []

        parent1, parent2 = parents[0], parents[1]

        # Prepare DTO for crossover
        dto = CodeCrossoverInput(
            operation=OPERATION_CROSSOVER,
            question_content=context.problem.question_content,
            parent1_snippet=parent1.snippet,
            parent2_snippet=parent2.snippet,
            starter_code=context.problem.starter_code,
        )

        try:
            output = self.operator.apply(dto)
        except Exception as e:
            logger.warning(f"Crossover operator failed: {e}")
            return []

        # Construct offspring individuals
        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_CROSSOVER,
                parent_probs=[parent1.probability, parent2.probability],
                initial_prior=self.pop_config.initial_prior,
            )

            offspring.append(
                CodeIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_CROSSOVER,
                    generation_born=context.code_population.generation + 1,
                    parents={"code": [parent1.id, parent2.id], "test": []},
                    metadata=res.metadata,
                )
            )

        return offspring

    def _breed_via_edit(
        self, coevolution_context: CoevolutionContext
    ) -> list[CodeIndividual]:
        """Breed new individuals via edit using failing tests."""
        # 1. Select Parent
        parents: list[CodeIndividual] = self.parent_selector.select_parents(
            coevolution_context.code_population,
            count=1,
            coevolution_context=coevolution_context,
        )
        if not parents:
            return []
        parent: CodeIndividual = parents[0]

        # 2. Select K Failing Tests
        failing_test_selections = self.failing_test_selector.select_k_failing_tests(
            coevolution_context, parent, k=self.k_failing_tests
        )
        if not failing_test_selections:
            logger.warning(
                f"No failing tests found for code individual '{parent.id}' during edit."
            )
            return []

        # 3. Collect test cases with error traces
        failing_tests_with_trace: list[tuple[str, str]] = []
        test_ids: list[str] = []

        for failing_test_ind, test_population_type in failing_test_selections:
            error_trace = (
                coevolution_context.interactions[test_population_type]
                .execution_results[parent.id]
                .test_results[failing_test_ind.id]
                .details
            )

            if not error_trace:
                logger.warning(
                    f"No error trace found for failing test '{failing_test_ind.id}' "
                    f"and code individual '{parent.id}'."
                )
                error_trace = "No error trace available."

            failing_tests_with_trace.append((failing_test_ind.snippet, error_trace))
            test_ids.append(failing_test_ind.id)

        # 4. Execution
        dto = CodeEditInput(
            operation=OPERATION_EDIT,
            question_content=coevolution_context.problem.question_content,
            parent_snippet=parent.snippet,
            failing_tests_with_trace=failing_tests_with_trace,
            starter_code=coevolution_context.problem.starter_code,
        )

        try:
            output = self.operator.apply(dto)
        except Exception as e:
            logger.warning(f"Edit operator failed: {e}")
            return []

        # 5. Construction
        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_EDIT,
                parent_probs=[parent.probability],
                initial_prior=self.pop_config.initial_prior,
            )

            offspring.append(
                CodeIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_EDIT,
                    generation_born=coevolution_context.code_population.generation + 1,
                    parents={"code": [parent.id], "test": test_ids},
                    metadata={
                        **res.metadata,
                        "num_failing_tests_used": len(failing_tests_with_trace),
                    },
                )
            )

        return offspring


__all__ = ["CodeBreedingStrategy"]
