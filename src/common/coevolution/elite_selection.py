"""
Elite selection strategies for coevolutionary algorithms.

This module implements the IEliteSelectionStrategy protocol, providing
selection mechanisms for choosing elite individuals to preserve unchanged
into the next generation.

The selection strategies work with Individual objects and can utilize the
full CoevolutionContext including interaction matrices for sophisticated
selection criteria.

Available Strategies:
- TopKEliteSelector: Simple top-k selection by probability (generic)
- TestDiversityEliteSelector: Diversity-based selection for test populations
  using observation matrix analysis
- CodeDiversityEliteSelector: Diversity-based selection for code populations
  combining diversity with top probability elites

Design Principles:
- Protocol-based: Implements IEliteSelectionStrategy[T]
- Context-aware: Receives full CoevolutionContext for advanced selection logic
- Type-safe: Uses generic type parameters bounded by BaseIndividual
- Matrix-aware: Can access interaction matrices for diversity calculations
"""

import numpy as np
from loguru import logger

from .core.individual import CodeIndividual, TestIndividual
from .core.interfaces import (
    BaseIndividual,
    BasePopulation,
    CoevolutionContext,
    PopulationConfig,
)


class TopKEliteSelector[T: BaseIndividual]:
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


class TestDiversityEliteSelector[T: TestIndividual]:
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

    def __init__(self, test_population_key: str):
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


class CodeDiversityEliteSelector[T: CodeIndividual]:
    """
    Diversity-based elite selection for code populations.

    Combines two selection criteria:
    1. Behavioral diversity: Selects code with diverse test interaction patterns
    2. Quality guarantee: Ensures top probability individuals are always included

    The strategy analyzes ALL observation matrices (concatenated horizontally) to
    identify diverse behavioral patterns across all test populations, then ensures
    the absolute best individuals (by probability) are included in the final elite set.

    This dual-objective approach ensures:
    - Population maintains behavioral diversity (exploration)
    - Best solutions are always preserved (exploitation)

    Design:
    - Uses ALL test population matrices (public, unittest, differential, etc.)
    - Concatenates matrices horizontally to form comprehensive behavioral signature
    - Works with any CodeIndividual subclass
    - Falls back to top-k selection if matrices unavailable

    Algorithm:
    1. Concatenate all observation matrices: [public | unittest | differential | ...]
    2. Group code by unique row patterns in concatenated matrix
    3. Select highest probability code from each unique group (diversity)
    4. Add top-k individuals by probability (quality guarantee)
    5. Return union, ensuring elitism_rate * population_size elites total

    Examples:
        >>> # For code population using all test matrices
        >>> code_selector = CodeDiversityEliteSelector()
        >>> elites = code_selector.select_elites(code_pop, config, context)
    """

    def __init__(self):
        """
        Initialize the diversity selector for code population.

        Uses all available test population matrices for comprehensive
        behavioral diversity analysis.
        """
        logger.debug("Initialized CodeDiversityEliteSelector (uses all test matrices)")

    def select_elites(
        self,
        population: BasePopulation[T],
        population_config: PopulationConfig,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select diverse elite code individuals with quality guarantee.

        Combines diversity-based selection (unique behavioral patterns across
        ALL test populations) with top-k selection (absolute best by probability)
        to ensure both exploration and exploitation.

        Strategy:
        1. Select best from each unique behavioral group (diversity)
        2. Also select top-k individuals by probability (quality)
        3. Deduplicate to avoid selecting same individual twice
        4. Return combined list

        Args:
            population: Code population to select elites from
            population_config: Configuration containing elitism_rate
            coevolution_context: Full system state including interaction matrices

        Returns:
            List of elite code individuals combining:
            - Best from each unique behavioral pattern (diversity)
            - Top-k by probability (quality guarantee)
            No duplicates. Size may exceed elitism_rate if many unique patterns.

        Empty State Behavior:
            Returns empty list [] for empty populations (size=0).

        Fallback Behavior:
            If observation matrices are unavailable or invalid, falls back to
            simple top-k selection by probability.

        Algorithm:
            1. Concatenate all observation matrices horizontally
               [public_matrix | unittest_matrix | differential_matrix | ...]
            2. Group code rows by unique patterns in concatenated matrix
            3. Within each unique group, select code with highest probability
            4. Add top-k individuals by probability to guarantee quality
            5. Deduplicate and return
        """
        # Empty state guard
        if population.size == 0:
            logger.debug(
                "CodeDiversityEliteSelector: Population is empty, returning []"
            )
            return []

        # Determine target number of elites for top-k quality guarantee
        num_top_elites = int(population.size * population_config.elitism_rate)

        # Try to concatenate all observation matrices
        try:
            concatenated_matrix = self._concatenate_all_matrices(
                coevolution_context, population.size
            )
        except (KeyError, AttributeError, ValueError) as e:
            logger.warning(
                f"CodeDiversityEliteSelector: "
                f"Could not concatenate observation matrices ({e}), "
                f"falling back to top-k selection"
            )
            return self._fallback_top_k_selection(population, num_top_elites)

        # Group code by unique row patterns in concatenated matrix
        unique_groups = self._group_by_unique_rows(concatenated_matrix)

        logger.debug(
            f"CodeDiversityEliteSelector: "
            f"Found {len(unique_groups)} unique behavioral patterns among "
            f"{population.size} code individuals (across {concatenated_matrix.shape[1]} tests)"
        )

        # Get probability array
        probabilities = population.probabilities

        # Step 1: Get best code from each unique group (diversity)
        diverse_elites = []
        for group_indices in unique_groups:
            group_probs = probabilities[group_indices]
            best_in_group_idx = int(np.argmax(group_probs))
            best_code_idx = group_indices[best_in_group_idx]
            diverse_elites.append(population[best_code_idx])

        # Step 2: Get top-k individuals by probability (quality guarantee)
        top_k_elites = population.get_top_k_individuals(num_top_elites)

        # Step 3: Combine and deduplicate
        elite_ids = set()
        final_elites = []

        # Add all diverse elites first
        for code in diverse_elites:
            if code.id not in elite_ids:
                elite_ids.add(code.id)
                final_elites.append(code)

        # Add top-k elites (skip duplicates)
        for code in top_k_elites:
            if code.id not in elite_ids:
                elite_ids.add(code.id)
                final_elites.append(code)

        logger.info(
            f"CodeDiversityEliteSelector: "
            f"Selected {len(final_elites)} elites from {population.size} code: "
            f"{len(diverse_elites)} from unique patterns + "
            f"{len(final_elites) - len(diverse_elites)} additional top elites "
            f"(target top-k={num_top_elites}, {len(unique_groups)} unique patterns)"
        )

        if final_elites:
            elite_probs = [e.probability for e in final_elites]
            logger.debug(
                f"Elite probability range: [{min(elite_probs):.4f}, {max(elite_probs):.4f}], "
                f"avg={np.mean(elite_probs):.4f}"
            )

        return final_elites

    def _concatenate_all_matrices(
        self, coevolution_context: CoevolutionContext, expected_rows: int
    ) -> np.ndarray:
        """
        Concatenate all observation matrices horizontally.

        Args:
            coevolution_context: Context containing all interaction matrices
            expected_rows: Expected number of rows (code population size)

        Returns:
            Concatenated matrix: [public_matrix | unittest_matrix | differential_matrix | ...]
            Shape: (num_code, total_tests_across_all_populations)

        Raises:
            ValueError: If no matrices available or dimension mismatch
        """
        matrices = []
        matrix_keys = []

        for key, interaction_data in coevolution_context.interactions.items():
            matrix = interaction_data.observation_matrix

            # Validate row count matches code population size
            if matrix.shape[0] != expected_rows:
                raise ValueError(
                    f"Matrix '{key}' has {matrix.shape[0]} rows, "
                    f"expected {expected_rows}"
                )

            matrices.append(matrix)
            matrix_keys.append(key)

        if not matrices:
            raise ValueError("No observation matrices available in context")

        # Concatenate horizontally: [matrix1 | matrix2 | matrix3 | ...]
        concatenated = np.concatenate(matrices, axis=1)

        logger.debug(
            f"Concatenated {len(matrices)} matrices: {matrix_keys} → "
            f"shape {concatenated.shape} "
            f"(rows={concatenated.shape[0]}, total_cols={concatenated.shape[1]})"
        )

        return concatenated

    def _group_by_unique_rows(self, observation_matrix: np.ndarray) -> list[np.ndarray]:
        """
        Group row indices by unique row patterns.

        Args:
            observation_matrix: Matrix where rows represent code

        Returns:
            List of arrays, where each array contains indices of code
            that share the same behavioral pattern (identical rows).

        Example:
            Matrix rows: [A, B, A, C, B] (where A, B, C are unique patterns)
            Returns: [[0, 2], [1, 4], [3]]  (groups of indices)
        """
        num_code = observation_matrix.shape[0]

        # Track which code have been grouped
        grouped = set()
        unique_groups = []

        for i in range(num_code):
            if i in grouped:
                continue

            # Start a new group with this code
            current_group = [i]
            grouped.add(i)

            # Find all other code with identical pattern
            for j in range(i + 1, num_code):
                if j in grouped:
                    continue

                # Check if rows are identical
                if np.array_equal(observation_matrix[i], observation_matrix[j]):
                    current_group.append(j)
                    grouped.add(j)

            unique_groups.append(np.array(current_group))

        logger.trace(
            f"Grouped {num_code} code into {len(unique_groups)} unique patterns. "
            f"Group sizes: {[len(g) for g in unique_groups]}"
        )

        return unique_groups

    def _fallback_top_k_selection(
        self,
        population: BasePopulation[T],
        num_elites: int,
    ) -> list[T]:
        """
        Fallback to simple top-k selection when matrix analysis is unavailable.

        Args:
            population: Population to select from
            num_elites: Number of elites to select

        Returns:
            Top-k individuals by probability
        """
        elites = population.get_top_k_individuals(num_elites)

        logger.debug(
            f"CodeDiversityEliteSelector: Fallback top-k selected {len(elites)} elites"
        )

        return elites

    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return "CodeDiversityEliteSelector()"
