"""
Top-K elite selection strategy.

This module implements a simple top-k elite selection based on probability
values, suitable for any population type (code, test, differential test, etc.).
"""

from loguru import logger

from coevolution.core.interfaces import (
    BaseIndividual,
    BasePopulation,
    CoevolutionContext,
    IEliteSelectionStrategy,
    PopulationConfig,
)


class TopKEliteSelector[T: BaseIndividual](IEliteSelectionStrategy[T]):
    """
    Simple top-k elite selection strategy based on probability.

    Selects the k individuals with highest probabilities to preserve
    into the next generation. This is a generic strategy that works
    for any population type (code, test, differential test, etc.).

    The number of elites is determined from PopulationConfig, typically
    using elitism_rate or elite_size parameters.

    This strategy is suitable when:
    - Probability values are the primary fitness metric
    - Simple greedy selection is desired
    - No diversity or multi-objective concerns exist

    Examples:
        >>> # Works for any population type
        >>> selector = TopKEliteSelector()
        >>> code_elites = selector.select_elites(code_pop, config, context)
        >>> test_elites = selector.select_elites(test_pop, config, context)
    """

    def select_elites(
        self,
        population: BasePopulation[T],
        population_config: PopulationConfig,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select top-k elite individuals by probability.

        Args:
            population: Population to select elites from
            population_config: Configuration containing elitism_rate or elite_size
            coevolution_context: Full system state (unused in simple top-k)

        Returns:
            List of elite individuals with highest probabilities.

        Empty State Behavior:
            Returns empty list [] for empty populations (size=0).
        """
        # Empty state guard
        if population.size == 0:
            logger.debug("TopKEliteSelector: Population is empty, returning []")
            return []

        # Determine number of elites from config
        num_elites = int(population.size * population_config.elitism_rate)

        # Use population's built-in method for top-k selection
        elites = population.get_top_k_individuals(num_elites)

        if elites:
            logger.debug(
                f"TopKEliteSelector: Selected {len(elites)} elites from "
                f"population (size={population.size}), "
                f"prob range=[{elites[-1].probability:.4f}, {elites[0].probability:.4f}]"
            )
        else:
            logger.debug(
                f"TopKEliteSelector: Selected 0 elites from population (size={population.size})"
            )

        return elites

    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return "TopKEliteSelector()"
