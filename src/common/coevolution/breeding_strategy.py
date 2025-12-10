# src/common/coevolution/breeding_strategy.py
import numpy as np
from loguru import logger

from .core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_MUTATION,
    OPERATION_REPRODUCTION,
    BaseIndividual,
    BasePopulation,
    ExecutionResults,
    IGeneticOperator,
    IIndividualFactory,
    IProbabilityAssigner,
    ISelectionStrategy,
    Operation,
    OperationContext,
    OperatorRatesConfig,
)


class BreedingStrategy[T_self: BaseIndividual, T_other: BaseIndividual]:
    """
    Breeding strategy for generating a single offspring.

    This class implements the genetic algorithm logic for one reproductive
    act: selecting parent(s), applying a genetic operator
    (crossover, edit, reproduction), and applying mutation.
    """

    def __init__(
        self,
        selector: ISelectionStrategy,
        operator: IGeneticOperator,
        individual_factory: IIndividualFactory[T_self],
        probability_assigner: IProbabilityAssigner,
        initial_prior: float,
    ) -> None:
        """
        Initialize the reproduction strategy.

        Args:
            selector: An object implementing the ISelectionStrategy interface
                      for parent selection.
            operator: An object implementing the IGeneticOperator interface
                      for crossover/mutation/edit.
            individual_factory: A factory function (e.g., a class constructor)
                                that creates a new individual object.
            initial_prior: The default probability to assign to new offspring
                           created via crossover or mutation.
        """
        self.selector = selector
        self.operator = operator
        self.individual_factory = individual_factory
        self.initial_prior = initial_prior
        self.probability_assigner = probability_assigner
        logger.debug(
            f"ReproductionStrategy initialized with initial_prior={initial_prior}"
        )

    def _get_population_type(self, population: BasePopulation[T_self]) -> str:
        """Helper to get a clean name for logging."""
        return population.__class__.__name__.replace("Population", "")

    def _perform_crossover(
        self, population: BasePopulation[T_self], current_generation: int
    ) -> tuple[str, list[T_self]]:
        """Selects two parents and performs crossover."""
        logger.trace("Worker selecting parents for crossover")
        p1_idx, p2_idx = self.selector.select_parents(population.probabilities)
        parent1 = population[p1_idx]
        parent2 = population[p2_idx]

        context = OperationContext(
            operation=OPERATION_CROSSOVER,
            code_individuals=[parent1, parent2],  # type: ignore[list-item]
            test_individuals=[],
            current_generation=current_generation,
        )
        new_snippet = self.operator.apply(context)

        pop_type = self._get_population_type(population)
        logger.trace(
            f"{pop_type} offspring: crossover (parents: {parent1.id}, {parent2.id})"
        )
        parents = [parent1, parent2]
        return new_snippet, parents

    def _perform_edit(
        self,
        population: BasePopulation[T_self],
        other_population: BasePopulation[T_other],
        execution_results: ExecutionResults,
        observation_matrix: np.ndarray,
        current_generation: int,
    ) -> tuple[str, list[T_self]]:
        """Selects one parent and performs a feedback-driven edit."""
        logger.trace("Worker selecting parent for edit")
        parent_idx = self.selector.select(population.probabilities)
        parent = population[parent_idx]

        # Find a failing test (if any) to provide context
        failing_test_idx = None
        if observation_matrix is not None and parent_idx < observation_matrix.shape[0]:
            # Find first failing test (0 in observation matrix)
            failing_tests = np.where(observation_matrix[parent_idx] == 0)[0]
            if len(failing_tests) > 0:
                failing_test_idx = failing_tests[0]

        # Build context with parent code and optionally a failing test
        test_individuals = []
        if failing_test_idx is not None and failing_test_idx < len(other_population):
            test_individuals = [other_population[failing_test_idx]]

        context = OperationContext(
            operation=OPERATION_EDIT,
            code_individuals=[parent],  # type: ignore[list-item]
            test_individuals=test_individuals,
            code_population=population,  # type: ignore[arg-type]
            test_population=other_population,  # type: ignore[arg-type]
            observation_matrix=observation_matrix,
            execution_results=execution_results,
            current_generation=current_generation,
        )

        new_snippet = self.operator.apply(context)

        pop_type = self._get_population_type(population)
        logger.trace(
            f"{pop_type} offspring: edit (parent: {parent.id}, test_idx: {failing_test_idx})"
        )
        return new_snippet, [parent]

    def _perform_reproduction(
        self, population: BasePopulation[T_self]
    ) -> tuple[str, list[T_self]]:
        """Selects one parent and copies it (reproduction)."""
        logger.trace("Worker selecting parent for reproduction")
        parent_idx = self.selector.select(population.probabilities)
        parent = population[parent_idx]

        logger.trace(f"{self._get_population_type(population)} offspring: reproduction")

        # Return the original snippet, not a copy
        return parent.snippet, [parent]

    def _apply_mutation(
        self, individual: T_self, prev_operation: Operation, current_generation: int
    ) -> str:
        """
        Applies mutation to an individual based on the mutation rate.
        """
        logger.trace("Worker applying mutation")
        context = OperationContext(
            operation=OPERATION_MUTATION,
            code_individuals=[individual],  # type: ignore[list-item]
            test_individuals=[],
            current_generation=current_generation,
        )
        mutated_snippet = self.operator.apply(context)
        logger.trace(f"Offspring: mutated after {prev_operation}")
        return mutated_snippet

    def _notify_parents(
        self,
        parents: list[T_self],
        operation: Operation,
        offspring_id: str,
        generation: int,
    ) -> None:
        """
        Notifies parent individuals that they have produced offspring.
        Each parent handles its own logging internally.

        Args:
            parents: List of parent individuals.
            operation: The genetic operation used.
            offspring_id: The ID of the offspring produced.
            generation: The generation when this parenting occurred.
        """
        for parent in parents:
            parent.notify_parent_of(offspring_id, operation, generation)

    def generate_single_offspring(
        self,
        population: BasePopulation[T_self],
        other_population: BasePopulation[T_other],
        execution_results: ExecutionResults,
        observation_matrix: np.ndarray,
        operation_rates: OperatorRatesConfig,
    ) -> T_self:
        """
        Generates a single offspring using appropriate genetic operators.

        Args:
            population: Population to generate offspring from.
            other_population: The complementary population (for edit operations).
            execution_results: Execution results data (passed to operators via context).
            observation_matrix: Observation matrix (passed to operators via context).
            operation_rates: Configuration for crossover/edit/mutation rates.

        Returns:
            A new Individual object.

        Raises:
            Exception: Re-raises exceptions from operators (e.g., LLMGenerationError)
                       to be handled by the parallel executor.
        """
        rand = np.random.random()
        base_operation: Operation = OPERATION_REPRODUCTION  # Default
        new_snippet: str = ""
        offspring: T_self
        final_operation: Operation
        final_prob: float
        parents: list[T_self]
        parent_ids: list[str]
        parent_probs: list[float]

        current_gen = population.generation

        # Step 1: Determine base genetic operation (crossover/edit/reproduction)
        if rand < operation_rates.crossover_rate:
            base_operation = OPERATION_CROSSOVER
            new_snippet, parents = self._perform_crossover(population, current_gen)

        elif rand < operation_rates.crossover_rate + operation_rates.edit_rate:
            base_operation = OPERATION_EDIT
            new_snippet, parents = self._perform_edit(
                population,
                other_population,
                execution_results,
                observation_matrix,
                current_gen,
            )

        else:
            base_operation = OPERATION_REPRODUCTION
            new_snippet, parents = self._perform_reproduction(population)

        parent_ids = [parent.id for parent in parents]
        parent_probs = [parent.probability for parent in parents]

        # Step 2: Determine if mutation will be applied
        will_mutate = np.random.random() < operation_rates.mutation_rate

        # Step 3: Determine final operation and snippet
        if will_mutate:
            final_operation = OPERATION_MUTATION
            # Pass the parent (not just snippet) to get full individual context
            final_snippet = self._apply_mutation(
                parents[0], base_operation, current_gen
            )
        else:
            final_operation = base_operation
            final_snippet = new_snippet

        # Step 4: Calculate probability based on final operation
        final_prob = self.probability_assigner.assign_probability(
            operation=final_operation,
            parent_probs=parent_probs,
            initial_prior=self.initial_prior,
        )

        # Step 5: Create offspring with final values
        offspring = self.individual_factory(
            snippet=final_snippet,
            probability=final_prob,
            creation_op=final_operation,
            generation_born=current_gen + 1,  # increment generation for offspring
            parent_ids=parent_ids,
        )

        # Step 6: Notify parents of their role in producing offspring
        self._notify_parents(parents, base_operation, offspring.id, current_gen)

        return offspring
