# src/common/coevolution/core/breeding_strategy.py
import numpy as np
from loguru import logger

from .interfaces import (
    BaseIndividual,
    BasePopulation,
    ExecutionResults,
    IFeedbackGenerator,
    IGeneticOperator,
    IIndividualFactory,
    IProbabilityAssigner,
    ISelectionStrategy,
    Operations,
    OperatorRatesConfig,
    ParentProbabilities,
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
        self, population: BasePopulation[T_self]
    ) -> tuple[str, list[str], ParentProbabilities]:
        """Selects two parents and performs crossover."""
        logger.trace("Worker selecting parents for crossover")
        p1_idx, p2_idx = self.selector.select_parents(population.probabilities)
        parent1 = population[p1_idx]
        parent2 = population[p2_idx]

        new_snippet = self.operator.crossover(parent1.snippet, parent2.snippet)

        pop_type = self._get_population_type(population)
        logger.trace(
            f"{pop_type} offspring: crossover (parents: {parent1.id}, {parent2.id})"
        )

        parent_ids = [parent1.id, parent2.id]
        parent_probs = [parent1.probability, parent2.probability]
        return new_snippet, parent_ids, parent_probs

    def _perform_edit(
        self,
        population: BasePopulation[T_self],
        other_population: BasePopulation[T_other],
        execution_results: ExecutionResults,
        observation_matrix: np.ndarray,
        feedback_generator: IFeedbackGenerator[T_other],
    ) -> tuple[str, list[str], ParentProbabilities]:
        """Selects one parent and performs a feedback-driven edit."""
        if not feedback_generator:
            # This is a safeguard; __init__ check should prevent this.
            logger.warning(
                "Edit op selected, but no feedback generator. Defaulting to reproduction."
            )
            return self._perform_reproduction(population)

        logger.trace("Worker selecting parent for edit")
        parent_idx = self.selector.select(population.probabilities)
        parent = population[parent_idx]

        feedback = feedback_generator(
            observation_matrix=observation_matrix,
            execution_results=execution_results,
            other_population=other_population,
            individual_idx=parent_idx,
        )

        new_snippet = self.operator.edit(parent.snippet, feedback)

        pop_type = self._get_population_type(population)
        logger.trace(
            f"{pop_type} offspring: edit ({len(feedback)} chars feedback, parent: {parent.id})"
        )

        parent_ids = [parent.id]
        parent_probs = [parent.probability]
        return new_snippet, parent_ids, parent_probs

    def _perform_reproduction(
        self, population: BasePopulation[T_self]
    ) -> tuple[str, list[str], ParentProbabilities]:
        """Selects one parent and copies it (reproduction)."""
        logger.trace("Worker selecting parent for reproduction")
        parent_idx = self.selector.select(population.probabilities)
        parent = population[parent_idx]

        logger.trace(f"{self._get_population_type(population)} offspring: reproduction")

        parent_ids = [parent.id]
        parent_probs = [parent.probability]
        # Return the original snippet, not a copy
        return parent.snippet, parent_ids, parent_probs

    def _apply_mutation(self, snippet: str, operation: Operations) -> str:
        """
        Applies mutation to a snippet based on the mutation rate.
        Returns the (potentially) mutated snippet and the final operation name.
        """

        logger.trace("Worker applying mutation")
        mutated_snippet = self.operator.mutate(snippet)
        logger.trace(f"Offspring: {operation} (after mutation)")
        return mutated_snippet

    def generate_single_offspring(
        self,
        population: BasePopulation[T_self],
        other_population: BasePopulation[T_other],
        execution_results: ExecutionResults,
        feedback_generator: IFeedbackGenerator[T_other],
        observation_matrix: np.ndarray,
        operation_rates: OperatorRatesConfig,
    ) -> T_self:
        """
        Generates a single offspring using appropriate genetic operators.

        Args:
            population: Population to generate offspring from.
            other_population: The complementary population (for feedback generation).
            execution_results: Opaque execution results data, which will be
                               passed directly to the feedback_generator.
            feedback_generator: Function to generate feedback string for edit ops.
            observation_matrix: Observation matrix (for feedback generation).
            crossover_rate: Probability of crossover (0.0 to 1.0).
            mutation_rate: Probability of mutation (0.0 to 1.0).
            edit_rate: Probability of edit (0.0 to 1.0).
            population_type: Type name for logging (e.g., "code", "test").

        Returns:
            A new Individual object.

        Raises:
            Exception: Re-raises exceptions from operators (e.g., LLMGenerationError)
                       or feedback generator to be handled by the parallel executor.
        """
        rand = np.random.random()
        operation: Operations = "reproduction"  # Default
        new_snippet: str = ""
        new_prob: float = self.initial_prior
        parent_ids: list[str] = []
        parent_probs: ParentProbabilities = []

        current_gen = population.generation

        if rand < operation_rates.crossover_rate:
            operation = "crossover"
            new_snippet, parent_ids, parent_probs = self._perform_crossover(population)

        elif rand < operation_rates.crossover_rate + operation_rates.edit_rate:
            operation = "edit"
            new_snippet, parent_ids, parent_probs = self._perform_edit(
                population,
                other_population,
                execution_results,
                observation_matrix,
                feedback_generator,
            )

        else:
            operation = "reproduction"
            new_snippet, parent_ids, parent_probs = self._perform_reproduction(
                population
            )

        if np.random.random() < operation_rates.mutation_rate:
            operation = "mutation"
            new_snippet = self._apply_mutation(new_snippet, operation)

        new_prob = self.probability_assigner(
            operation=operation,
            parent_probs=parent_probs,
            initial_prior=self.initial_prior,
        )

        logger.trace(
            f"Assigned new probability for {operation} offspring: {new_prob:.4f}"
        )

        offspring = self.individual_factory(
            snippet=new_snippet,
            probability=new_prob,
            creation_op=operation,
            generation_born=current_gen,
            parent_ids=parent_ids,
        )

        return offspring
