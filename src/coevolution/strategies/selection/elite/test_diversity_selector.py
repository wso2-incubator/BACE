"""
Diversity-based elite selection for test populations.

This module implements elite selection that maximizes behavioral diversity
based on interaction patterns with the code population.
"""

import numpy as np
from loguru import logger

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    BasePopulation,
    CoevolutionContext,
    IEliteSelectionStrategy,
    PopulationConfig,
)


class TestDiversityEliteSelector[T: TestIndividual](IEliteSelectionStrategy[T]):
    """
    Diversity-based elite selection for test populations.

    Selects test individuals that maximize behavioral diversity based on
    their interaction patterns with the code population. Uses the observation
    matrix to identify tests with non-overlapping discrimination patterns.

    This strategy is specifically designed for test populations and requires
    access to the observation matrix via the CoevolutionContext. The test
    population key identifies which interaction matrix to use.

    The strategy aims to select tests that:
    - Have high discrimination ability (probability)
    - Cover different code behaviors (diversity)
    - Avoid redundant tests that check the same things

    Design:
    - Parameterized with test_population_key to identify which matrix to use
    - Works with any TestIndividual subclass (unittest, differential, property)
    - Falls back to top-k selection if matrix unavailable

    Examples:
        >>> # For unittest population
        >>> unittest_selector = TestDiversityEliteSelector(
        ...     test_population_key="unittest"
        ... )
        >>> elites = unittest_selector.select_elites(
        ...     unittest_pop, config, context
        ... )
        >>>
        >>> # For differential test population
        >>> diff_selector = TestDiversityEliteSelector(
        ...     test_population_key="differential"
        ... )
        >>> elites = diff_selector.select_elites(diff_pop, config, context)
    """

    def __init__(self, test_population_key: str) -> None:
        """
        Initialize the diversity selector for a specific test population.

        Args:
            test_population_key: Key to lookup interaction data in
                                CoevolutionContext.interactions.
                                Must match a key in context.test_populations
                                (e.g., "unittest", "differential", "property").

        Examples:
            >>> selector = TestDiversityEliteSelector("unittest")
            >>> selector = TestDiversityEliteSelector("differential")
        """
        self.test_population_key = test_population_key
        logger.debug(
            f"Initialized TestDiversityEliteSelector with key: {test_population_key}"
        )

    def select_elites(
        self,
        population: BasePopulation[T],
        population_config: PopulationConfig,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select diverse elite test individuals using observation matrix analysis.

        Groups tests by their unique discrimination patterns (observation matrix columns).
        Within each group of tests with identical patterns, selects the one with
        highest probability. ALWAYS returns best from each unique group (maximizes diversity).

        Args:
            population: Test population to select elites from
            population_config: Configuration (elitism_rate ignored for this strategy)
            coevolution_context: Full system state including interaction matrices

        Returns:
            List of elite test individuals with diverse discrimination patterns.
            Each selected test has a unique discrimination pattern, and within
            each pattern group, the highest probability test is chosen.
            Size = number of unique patterns (may be less than or greater than elitism_rate).

        Empty State Behavior:
            Returns empty list [] for empty populations (size=0).

        Fallback Behavior:
            If observation matrix is unavailable or invalid, falls back to
            simple top-k selection by probability.

        Algorithm:
            1. Extract observation matrix (rows=code, columns=tests)
            2. Group test columns by unique patterns
            3. Within each unique group, select test with highest probability
            4. Return ALL selected tests (one per unique pattern)
        """
        # Empty state guard
        if population.size == 0:
            logger.debug(
                f"TestDiversityEliteSelector[{self.test_population_key}]: "
                "Population is empty, returning []"
            )
            return []

        # Try to get the observation matrix
        try:
            interaction = coevolution_context.interactions[self.test_population_key]
            observation_matrix = interaction.observation_matrix
        except (KeyError, AttributeError) as e:
            logger.warning(
                f"TestDiversityEliteSelector[{self.test_population_key}]: "
                f"Could not access observation matrix ({e}), falling back to top-k selection"
            )
            return self._fallback_top_k_selection(population, population_config)

        # Validate matrix dimensions
        if observation_matrix.shape[1] != population.size:
            logger.warning(
                f"TestDiversityEliteSelector[{self.test_population_key}]: "
                f"Matrix column count ({observation_matrix.shape[1]}) != "
                f"population size ({population.size}), falling back to top-k selection"
            )
            return self._fallback_top_k_selection(population, population_config)

        # Group tests by unique column patterns
        unique_groups = self._group_by_unique_columns(observation_matrix)

        logger.debug(
            f"TestDiversityEliteSelector[{self.test_population_key}]: "
            f"Found {len(unique_groups)} unique discrimination patterns among "
            f"{population.size} tests"
        )

        # Select highest probability test from each unique group
        elites = []
        probabilities = population.probabilities

        for group_indices in unique_groups:
            # Get probabilities for tests in this group
            group_probs = probabilities[group_indices]
            # Find index within group with highest probability
            best_in_group_idx = int(np.argmax(group_probs))
            # Map back to original population index
            best_test_idx = group_indices[best_in_group_idx]
            # Add to elites
            elites.append(population[best_test_idx])

        logger.info(
            f"TestDiversityEliteSelector[{self.test_population_key}]: "
            f"Selected {len(elites)} diverse elites from {population.size} tests "
            f"({len(unique_groups)} unique patterns, one best from each)"
        )

        if elites:
            elite_probs = [e.probability for e in elites]
            logger.debug(
                f"Elite probability range: [{min(elite_probs):.4f}, {max(elite_probs):.4f}], "
                f"avg={np.mean(elite_probs):.4f}"
            )

        return elites

    def _group_by_unique_columns(
        self, observation_matrix: np.ndarray
    ) -> list[np.ndarray]:
        """
        Group column indices by unique column patterns.

        Args:
            observation_matrix: Matrix where columns represent tests

        Returns:
            List of arrays, where each array contains indices of tests
            that share the same discrimination pattern (identical columns).

        Example:
            Matrix columns: [A, B, A, C, B] (where A, B, C are unique patterns)
            Returns: [[0, 2], [1, 4], [3]]  (groups of indices)
        """
        num_tests = observation_matrix.shape[1]

        # Transpose to work with columns as rows for easier comparison
        columns = observation_matrix.T  # Shape: (num_tests, num_code)

        # Track which tests have been grouped
        grouped = set()
        unique_groups = []

        for i in range(num_tests):
            if i in grouped:
                continue

            # Start a new group with this test
            current_group = [i]
            grouped.add(i)

            # Find all other tests with identical pattern
            for j in range(i + 1, num_tests):
                if j in grouped:
                    continue

                # Check if columns are identical
                if np.array_equal(columns[i], columns[j]):
                    current_group.append(j)
                    grouped.add(j)

            unique_groups.append(np.array(current_group))

        logger.trace(
            f"Grouped {num_tests} tests into {len(unique_groups)} unique patterns. "
            f"Group sizes: {[len(g) for g in unique_groups]}"
        )

        return unique_groups

    def _fallback_top_k_selection(
        self,
        population: BasePopulation[T],
        population_config: PopulationConfig,
    ) -> list[T]:
        """
        Fallback to simple top-k selection when matrix analysis is unavailable.

        Args:
            population: Population to select from
            population_config: Configuration with elitism_rate

        Returns:
            Top-k individuals by probability
        """
        num_elites = int(population.size * population_config.elitism_rate)
        elites = population.get_top_k_individuals(num_elites)

        logger.debug(
            f"TestDiversityEliteSelector[{self.test_population_key}]: "
            f"Fallback top-k selected {len(elites)} elites"
        )

        return elites

    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return f"TestDiversityEliteSelector(test_population_key='{self.test_population_key}')"
