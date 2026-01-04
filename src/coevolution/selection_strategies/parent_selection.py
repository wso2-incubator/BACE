"""
Parent selection strategies for breeding in coevolutionary algorithms.

Implements fitness-proportionate (roulette wheel) selection.
"""

import numpy as np
from loguru import logger

from ..core.interfaces import (
    BaseIndividual,
    BasePopulation,
    CoevolutionContext,
    IParentSelectionStrategy,
)


class RouletteWheelParentSelection[T: BaseIndividual](IParentSelectionStrategy[T]):
    """
    Roulette wheel (fitness-proportionate) parent selection strategy.

    Selects parents with probability proportional to their fitness/probability
    values. Also known as fitness-proportionate selection. Individuals with
    higher probabilities have a greater chance of being selected as parents.

    This strategy is suitable when:
    - Probability values represent meaningful fitness
    - You want to maintain selection pressure proportional to fitness
    - Population diversity needs to be preserved (weak individuals still have a chance)

    The strategy handles edge cases gracefully:
    - All probabilities zero: Falls back to uniform random selection
    - Single individual: Returns that individual
    - Count > population size: Allows duplicates (sampling with replacement)

    Examples:
        >>> strategy = RouletteWheelParentSelection()
        >>> # Select 2 parents for crossover
        >>> parents = strategy.select_parents(population, count=2, context)
        >>> # Select 1 parent for mutation
        >>> parent = strategy.select_parents(population, count=1, context)[0]
    """

    def select_parents(
        self,
        population: BasePopulation[T],
        count: int,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select parent individuals using roulette wheel selection.

        Parents are selected with probability proportional to their fitness/
        probability values. This is done by normalizing probabilities to sum
        to 1.0 and using them as sampling weights.

        Args:
            population: Population to select parents from
            count: Number of parents to select (typically 1 for mutation, 2 for crossover)
            coevolution_context: Full coevolution state (unused in basic roulette wheel,
                                but available for context-aware extensions)

        Returns:
            List of selected parent individuals. May contain duplicates if the
            same individual is selected multiple times (sampling with replacement).

        Raises:
            ValueError: If count < 1 or population is empty

        Note:
            Selection is done with replacement - the same individual can be
            selected multiple times if count > 1.
        """
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}")

        if population.size == 0:
            raise ValueError("Cannot select parents from empty population")

        # count greater than population size
        if population.size < count:
            raise ValueError("Population size must be at least equal to count")

        # Extract probabilities from population
        probabilities = population.probabilities

        # Handle all-zero probabilities (degenerate case)
        total_prob = np.sum(probabilities)
        if total_prob == 0:
            logger.warning(
                "Roulette wheel: all probabilities are zero, using uniform random selection"
            )
            # Uniform random selection
            indices = np.random.choice(population.size, size=count, replace=True)
            selected = [population.individuals[int(idx)] for idx in indices]
            logger.debug(
                f"Selected {count} parents via uniform random: "
                f"{[ind.id for ind in selected]}"
            )
            return selected

        # Normalize probabilities to create a valid probability distribution
        normalized_probs = probabilities / total_prob

        # Vectorized selection using numpy's built-in weighted choice
        # replace=True allows the same individual to be selected multiple times
        indices = np.random.choice(
            population.size, size=count, p=normalized_probs, replace=True
        )

        # Convert indices to Individual objects
        selected = [population.individuals[int(idx)] for idx in indices]

        logger.debug(
            f"Roulette wheel selected {count} parents: "
            f"{[f'{ind.id} (p={ind.probability:.4f})' for ind in selected]}"
        )

        return selected

    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return "RouletteWheelParentSelection()"
