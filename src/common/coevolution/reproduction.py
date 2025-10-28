"""
Reproduction strategies for evolutionary algorithms.

This module provides generic reproduction strategies that generate offspring
from populations using genetic operators (crossover, mutation, edit, reproduction).

The reproduction strategy implements a roulette wheel approach where operator
selection is based on configured probabilities (crossover_rate, mutation_rate,
edit_rate, and implicit reproduction_rate).

Design:
- Generic: Works with any population type (code, test, etc.)
- Uses LLM-based operators (BaseLLMOperator) for genetic operations
- Selection-agnostic: Uses provided selection strategy
- Symmetric: Code and test offspring generation use the same interface
  (both require the "other" population for feedback in edit operations)
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
from loguru import logger

from common.sandbox import TestExecutionResult

from .operators import BaseLLMOperator
from .population import BasePopulation
from .selection import SelectionStrategy


class ReproductionStrategy:
    """
    Reproduction strategy for generating offspring from populations.

    This class implements the genetic algorithm reproduction phase where
    offspring are generated from a population using genetic operators
    based on configured probabilities.

    The strategy uses a roulette wheel approach for primary operations:
    - [0, crossover_rate): Apply crossover
    - [crossover_rate, crossover_rate + edit_rate): Apply edit
    - [crossover_rate + edit_rate, 1.0]: Apply reproduction (copy)

    After the primary operation, mutation is applied independently with probability mutation_rate.

    Prior probability assignment:
    - Crossover: max(min(parent1_prob, parent2_prob), initial_prior)
    - Edit: max(parent_prob, initial_prior)
    - Reproduction: parent_prob (unchanged)
    - Mutation: Keeps the prior from the primary operation

    Attributes:
        selector: Selection strategy for choosing parents
        operator: LLM-based genetic operator implementing crossover/mutation/edit
        initial_prior: Prior probability for new offspring
    """

    def __init__(
        self,
        selector: SelectionStrategy,
        operator: BaseLLMOperator,
        initial_prior: float,
    ) -> None:
        """
        Initialize the reproduction strategy.

        Args:
            selector: Selection strategy for parent selection
            operator: LLM-based genetic operator for genetic operations (CodeOperator or TestOperator)
            initial_prior: Prior probability assigned to offspring
        """
        self.selector = selector
        self.operator = operator
        self.initial_prior = initial_prior

    def generate_offspring(
        self,
        population: BasePopulation,
        other_population: BasePopulation,
        execution_results: Dict[int, TestExecutionResult],
        feedback_generator: Callable[..., str],
        observation_matrix: np.ndarray | None,
        offspring_size: int,
        crossover_rate: float,
        mutation_rate: float,
        edit_rate: float,
        population_type: str = "generic",
    ) -> List[Tuple[str, float]]:
        """
        Generate offspring for a population using genetic operators.

        This unified method works for both code and test populations. The workflow is:
        1. Apply primary operation (crossover, edit, or reproduction) based on rates
        2. Apply mutation independently with probability mutation_rate
        3. Assign prior probability based on operation and parent probabilities

        The edit operation requires:
        - execution_results: Results from executing this population against the other
        - other_population: The complementary population (test for code, code for test)
        - feedback_generator: Function to generate feedback for edit operations

        Args:
            population: Population to generate offspring from
            other_population: The complementary population for feedback generation
            execution_results: Execution results for feedback generation
            feedback_generator: Function to generate feedback for edit operations.
                              Expected signature: (execution_result, other_population, individual_idx) -> feedback_str
                              The actual population types can be specific (TestPopulation, CodePopulation)
            offspring_size: Number of offspring to generate
            crossover_rate: Probability of applying crossover as primary operation
            mutation_rate: Probability of applying mutation after primary operation
            edit_rate: Probability of applying edit as primary operation
            population_type: Type name for logging (e.g., "code", "test")

        Returns:
            List of (offspring_individual, probability) tuples
        """
        logger.info(
            f"Generating {offspring_size} {population_type} offspring "
            f"(crossover={crossover_rate:.2f}, "
            f"mutation={mutation_rate:.2f}, "
            f"edit={edit_rate:.2f})"
        )

        offspring = []
        for i in range(offspring_size):
            rand = np.random.random()
            operation = ""

            if rand < crossover_rate:
                # Apply crossover
                logger.debug(f"offspring {i + 1}: selecting parents for crossover")
                parent1_idx, parent2_idx = self.selector.select_parents(
                    population.probabilities
                )
                parent1 = population.individuals[parent1_idx]
                parent2 = population.individuals[parent2_idx]
                parent1_prob = float(population.probabilities[parent1_idx])
                parent2_prob = float(population.probabilities[parent2_idx])

                child = self.operator.crossover(parent1, parent2)

                # Prior is max of (min of parent probs, initial_prior)
                child_prob = max(min(parent1_prob, parent2_prob), self.initial_prior)
                operation = "crossover"
                logger.trace(
                    f"{population_type.capitalize()} offspring {i + 1}: crossover "
                    f"(parent probs: {parent1_prob:.4f}, {parent2_prob:.4f} → {child_prob:.4f})"
                )

            elif rand < crossover_rate + edit_rate:
                # Apply edit operation using feedback
                logger.debug(f"offspring {i + 1}: selecting parent for edit")
                parent_idx = self.selector.select(population.probabilities)
                parent = population.individuals[parent_idx]
                parent_prob = float(population.probabilities[parent_idx])

                # Generate feedback for the parent individual
                feedback = feedback_generator(
                    observation_matrix,
                    execution_results,
                    other_population,
                    parent_idx,
                )

                # Use LLM to edit based on feedback
                child = self.operator.edit(parent, feedback)

                # Prior is max of (parent prob, initial_prior)
                child_prob = max(parent_prob, self.initial_prior)
                operation = "edit"
                logger.trace(
                    f"{population_type.capitalize()} offspring {i + 1}: edit with feedback "
                    f"({len(feedback)} chars, parent prob: {parent_prob:.4f} → {child_prob:.4f})"
                )

            else:
                # Reproduction (copy without modification)
                logger.debug(f"offspring {i + 1}: selecting parent for reproduction")
                child_idx = self.selector.select(population.probabilities)
                child = population.individuals[child_idx]
                child_prob = float(population.probabilities[child_idx])
                operation = "reproduction"
                logger.trace(
                    f"{population_type.capitalize()} offspring {i + 1}: reproduction "
                    f"(prob: {child_prob:.4f})"
                )

            # Apply mutation independently with probability mutation_rate
            if np.random.random() < mutation_rate:
                logger.debug(f"offspring {i + 1}: chosen for mutation")
                child = self.operator.mutate(child)
                operation += "+mutation"
                logger.trace(
                    f"{population_type.capitalize()} offspring {i + 1}: mutation applied "
                    f"(after {operation.split('+')[0]})"
                )

            offspring.append((child, child_prob))

        logger.debug(f"Generated {len(offspring)} {population_type} offspring")
        return offspring
