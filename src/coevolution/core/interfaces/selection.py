# coevolution/core/interfaces/selection.py
"""
Selection strategy protocols for elite and parent selection.
"""

from typing import Protocol

from .base import BaseIndividual, BasePopulation
from .config import PopulationConfig
from .context import CoevolutionContext


class IEliteSelectionStrategy[T: BaseIndividual](Protocol):
    """
    Protocol for selecting elite individuals to preserve unchanged to next generation.

    This interface enables pluggable, sophisticated selection strategies that can consider:
    - Individual probabilities (belief in correctness)
    - Test performance (passing rates across multiple test populations for code)
    - Discrimination ability (for tests - how well they distinguish good/bad code)
    - Diversity metrics (avoiding redundant individuals)
    - Any other custom criteria

    The strategy receives the target population, its configuration, and full coevolution
    context, providing maximum flexibility to implement different selection strategies.

    Example strategies:
    - Code selection: Multi-objective (probability + test performance)
    - Test selection: Pareto front (probability + discrimination)
    - Simple selection: Top-k by probability only
    - Diversity-based: Select diverse individuals covering different behaviors
    """

    def select_elites(
        self,
        population: BasePopulation[T],
        population_config: PopulationConfig,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select elite individuals to preserve unchanged to next generation.

        Args:
            population: The population to select elites from
            population_config: Configuration containing selection preferences
                              (e.g., diversity_selection flag, elitism_rate, elite_size)
            coevolution_context: Complete system state including all populations
                                and their interactions for context-aware selection

        Returns:
            List of elite individuals to preserve unchanged.
            The number of elites is typically determined by population_config
            (e.g., elite_size, survival_rate, or elitism_rate).

        Empty State Behavior (size=0):
            - Identity Operation: When population size is 0, should return an empty
              list [] (no individuals to select from, no elites to preserve).
            - This supports differential testing scenarios where populations may
              start empty and grow through bootstrapping operations.
        """
        ...


class IParentSelectionStrategy[T: BaseIndividual](Protocol):
    """
    Protocol for selecting parent individuals for breeding.

    Defines the contract for any class that can select parents from a population
    based on fitness, behavior, or other criteria. Strategies receive the full
    population and coevolution context to enable sophisticated selection logic.

    Example strategies:
    - Roulette wheel: Probability-proportional selection
    - Tournament: Select best from random subset
    - Rank-based: Select based on rank rather than raw probability
    - Behavior-based: Select parents with complementary behaviors
    - DET-specific: Select code pairs with similar behavior for differential testing
    """

    def select_parents(
        self,
        population: BasePopulation[T],
        count: int,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select parent individuals for breeding.

        Args:
            population: Population to select parents from
            count: Number of parents to select:
                   - 1 for mutation/reproduction
                   - 2 for crossover
                   - N for batch operations
            coevolution_context: Full coevolution state for context-aware selection
                                (e.g., interaction data, other populations)

        Returns:
            List of selected parent individuals. May contain duplicates if
            count > 1 and the same individual is selected multiple times.

        Note:
            For cross-species operations (e.g., DET selecting code parents for
            test offspring), the strategy can access other populations via
            coevolution_context and return individuals of different type than
            the target population.
        """
        ...
