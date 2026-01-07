"""
Diversity-based elite selection for code populations.

This module implements elite selection that combines behavioral diversity
(unique test interaction patterns) with quality guarantees (top probabilities).
"""

import numpy as np
from loguru import logger

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    BasePopulation,
    CoevolutionContext,
    IEliteSelectionStrategy,
    PopulationConfig,
)


class CodeDiversityEliteSelector(IEliteSelectionStrategy[CodeIndividual]):
    """
    Diversity-based elite selection for code populations.

    Core idea:
    I will save the best version of every distinct strategy we have found
    so far. Then, I will fill the remaining slots with the highest-probability
    individuals to ensure we remain strong. Finally, if this combined group
    is too large to support new offspring, I will strictly cut the
    lowest-probability individuals to fit the population limit."

    Combines two selection criteria:
    1. Behavioral diversity: Selects code with diverse test interaction patterns
    2. Quality guarantee: Ensures top probability individuals are always included

    The strategy analyzes ALL observation matrices (concatenated horizontally) to
    identify diverse behavioral patterns across all test populations, then ensures
    the absolute best individuals (by probability) are included in the final elite set.
    Finally, it enforces strict population limits by truncating lowest-probability
    individuals if the elite set grows too large.

    This dual-objective approach ensures:
    - Population maintains behavioral diversity (exploration)
    - Best solutions are always preserved (exploitation)
    - Population size constraints are respected (feasibility)

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
    5. Union and deduplicate the two sets
    6. Truncate: If total elites > (max_pop - offspring_count), keep only the top-N by probability

    Examples:
        >>> # For code population using all test matrices
        >>> code_selector = CodeDiversityEliteSelector()
        >>> elites = code_selector.select_elites(code_pop, config, context)
    """

    def __init__(self) -> None:
        """
        Initialize the diversity selector for code population.

        Uses all available test population matrices for comprehensive
        behavioral diversity analysis.
        """
        logger.debug("Initialized CodeDiversityEliteSelector (uses all test matrices)")

    def _sort_by_probability(
        self, individuals: list[CodeIndividual]
    ) -> list[CodeIndividual]:
        """Helper to sort individuals by probability descending."""
        return sorted(individuals, key=lambda ind: ind.probability, reverse=True)

    def select_elites(
        self,
        population: BasePopulation[CodeIndividual],
        population_config: PopulationConfig,
        coevolution_context: CoevolutionContext,
    ) -> list[CodeIndividual]:
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

        # Determine the offspring number to leave room for breeding
        num_offspring = int(
            population_config.offspring_rate * population_config.max_population_size
        )

        maximum_num_elites = population_config.max_population_size - num_offspring

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

        # Truncate if exceeding maximum allowed elites
        if maximum_num_elites < len(final_elites):
            logger.debug(
                f"CodeDiversityEliteSelector: Number of selected elites ({len(final_elites)}) "
                f"exceeds available slots after offspring ({maximum_num_elites}). "
                f"Truncating elite list to fit."
            )

            final_elites = self._sort_by_probability(final_elites)[:maximum_num_elites]

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

        return np.asarray(concatenated)

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
        population: BasePopulation[CodeIndividual],
        num_elites: int,
    ) -> list[CodeIndividual]:
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
