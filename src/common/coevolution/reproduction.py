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
- Single responsibility: Only handles offspring generation logic
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

    The strategy uses a roulette wheel approach:
    - [0, crossover_rate): Apply crossover
    - [crossover_rate, crossover_rate + mutation_rate): Apply mutation
    - [crossover_rate + mutation_rate, crossover_rate + mutation_rate + edit_rate): Apply edit
    - [crossover_rate + mutation_rate + edit_rate, 1.0]: Apply reproduction (copy)

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
        offspring_size: int,
        crossover_rate: float,
        mutation_rate: float,
        edit_rate: float,
        population_type: str = "generic",
    ) -> List[Tuple[str, float]]:
        """
        Generate offspring for a population using genetic operators.

        This unified method works for both code and test populations. The edit
        operation requires:
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
            crossover_rate: Probability of applying crossover
            mutation_rate: Probability of applying mutation
            edit_rate: Probability of applying edit
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

            if rand < crossover_rate:
                # Apply crossover
                parent1_idx, parent2_idx = self.selector.select_parents(
                    population.probabilities
                )
                parent1 = population.individuals[parent1_idx]
                parent2 = population.individuals[parent2_idx]
                child = self.operator.crossover(parent1, parent2)
                logger.trace(
                    f"{population_type.capitalize()} offspring {i + 1}: crossover"
                )

            elif rand < crossover_rate + mutation_rate:
                # Apply mutation
                parent_idx = self.selector.select(population.probabilities)
                parent = population.individuals[parent_idx]
                child = self.operator.mutate(parent)
                logger.trace(
                    f"{population_type.capitalize()} offspring {i + 1}: mutation"
                )

            elif rand < crossover_rate + mutation_rate + edit_rate:
                # Apply edit operation using feedback
                parent_idx = self.selector.select(population.probabilities)
                parent = population.individuals[parent_idx]

                # Generate feedback using the provided feedback generator
                feedback = feedback_generator(
                    execution_results[parent_idx],
                    other_population,
                    parent_idx,
                )

                # Use LLM to edit based on feedback
                child = self.operator.edit(parent, feedback)
                logger.trace(
                    f"{population_type.capitalize()} offspring {i + 1}: edit with feedback "
                    f"({len(feedback)} chars)"
                )

            else:
                # Reproduction (copy without modification)
                child_idx = self.selector.select(population.probabilities)
                child = population.individuals[child_idx]
                logger.trace(
                    f"{population_type.capitalize()} offspring {i + 1}: reproduction"
                )

            # Offspring gets neutral prior probability
            offspring.append((child, self.initial_prior))

        logger.debug(f"Generated {len(offspring)} {population_type} offspring")
        return offspring
