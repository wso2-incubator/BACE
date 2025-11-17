"""
Pareto front selection system for multi-objective optimization.

This module implements the IPareto protocol for selecting test individuals
based on two objectives: probability (correctness) and discrimination (ability
to distinguish between good and bad code).

The implementation is stateless and uses static methods for all operations,
making it easy to test and use without instantiation.
"""

import numpy as np
from loguru import logger

from .core.interfaces import IPareto


class ParetoSystem(IPareto):
    """
    Static implementation of the IPareto protocol.

    Provides Pareto front selection for multi-objective optimization
    of test populations based on:
    1. Probability (belief in correctness)
    2. Discrimination (ability to distinguish good from bad code)

    All methods are static - no instantiation needed.
    """

    @staticmethod
    def calculate_discrimination(observation_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates discrimination scores for each test using entropy.

        A test with maximum discrimination (1.0) passes for exactly 50% of codes.
        A test with minimum discrimination (0.0) passes for 0% or 100% of codes.

        The discrimination is calculated as binary entropy:
        H(p) = -p * log2(p) - (1-p) * log2(1-p)

        where p is the pass rate (fraction of codes that pass the test).

        Args:
            observation_matrix: A 2D array where rows are code individuals
                                and columns are tests. Values are 1 (pass) or 0 (fail).

        Returns:
            A 1D array of discrimination scores for each test, in range [0, 1].

        Raises:
            ValueError: If the observation matrix is completely empty (0x0).
        """
        # Check for completely empty matrix (0x0) or invalid shape
        if observation_matrix.size == 0 and observation_matrix.shape == (0, 0):
            logger.error("Cannot calculate discrimination for empty observation matrix")
            raise ValueError("Observation matrix cannot be empty")

        num_codes, num_tests = observation_matrix.shape
        logger.debug(
            f"Calculating discrimination for {num_tests} tests across {num_codes} codes"
        )

        if num_codes == 0:
            logger.warning(
                f"Observation matrix has 0 codes. Returning zero discrimination for {num_tests} tests."
            )
            return np.zeros(num_tests, dtype=float)

        # Calculate pass rate for each test (column)
        pass_rates = np.sum(observation_matrix, axis=0) / float(num_codes)
        logger.trace(f"Pass rates: {pass_rates}")

        # Avoid log2(0) by clipping to [eps, 1-eps]
        eps = 1e-12
        clipped_rates = np.clip(pass_rates, eps, 1 - eps)
        logger.trace(f"Clipped pass rates with eps={eps}: {clipped_rates}")

        # Calculate binary entropy (discrimination)
        # H(p) = -p*log2(p) - (1-p)*log2(1-p)
        entropy = -clipped_rates * np.log2(clipped_rates) - (
            1 - clipped_rates
        ) * np.log2(1 - clipped_rates)
        logger.trace(f"Computed entropy: {entropy}")

        # Validate that all discriminations are in [0, 1]
        if not (np.all(entropy >= 0) and np.all(entropy <= 1)):
            logger.error(
                f"Invalid discriminations computed: min={np.min(entropy)}, max={np.max(entropy)}"
            )
            raise ValueError(
                "Discriminations (entropy) must be in [0, 1] for all tests"
            )

        logger.debug(
            f"Computed discriminations: min={np.min(entropy):.4f}, max={np.max(entropy):.4f}, "
            f"mean={np.mean(entropy):.4f}"
        )
        return np.asarray(entropy)

    @staticmethod
    def calculate_pareto_front(
        probabilities: np.ndarray, discriminations: np.ndarray
    ) -> list[int]:
        """
        Calculates the Pareto front for two-objective optimization.

        An individual is on the Pareto front if no other individual dominates it.
        Individual A dominates individual B if A is at least as good as B in all
        objectives and strictly better in at least one objective.

        Both objectives are maximized (higher is better).

        Args:
            probabilities: 1D array of probabilities (Objective 1, to maximize).
            discriminations: 1D array of discriminations (Objective 2, to maximize).

        Returns:
            A list of integer indices for the individuals on the Pareto front.

        Raises:
            ValueError: If arrays have different lengths or are empty.
        """
        if len(probabilities) != len(discriminations):
            logger.error(
                f"Length mismatch: {len(probabilities)} probabilities vs "
                f"{len(discriminations)} discriminations"
            )
            raise ValueError(
                "Probabilities and discriminations must have the same length"
            )

        if len(probabilities) == 0:
            logger.warning("Empty arrays provided to calculate_pareto_front")
            return []

        logger.debug(f"Calculating Pareto front for {len(probabilities)} individuals")

        # Stack objectives into (n_points, 2) array for maximization
        objectives = np.column_stack((probabilities, discriminations))

        num_points = objectives.shape[0]
        candidate_indices = np.arange(num_points)
        current_objectives = objectives.copy()
        next_comparison_idx = 0

        # Iteratively remove dominated points
        while next_comparison_idx < len(current_objectives):
            comparison_point = current_objectives[next_comparison_idx]

            # A point is NOT dominated by comparison_point if:
            # 1. It's strictly better in at least one objective, OR
            # 2. It's equal in all objectives (identical points should all be kept)
            is_strictly_better = np.any(current_objectives > comparison_point, axis=1)
            is_identical = np.all(current_objectives == comparison_point, axis=1)
            is_not_dominated_mask = is_strictly_better | is_identical

            # Filter to non-dominated points
            candidate_indices = candidate_indices[is_not_dominated_mask]
            current_objectives = current_objectives[is_not_dominated_mask]

            # Update next comparison index accounting for removed points
            next_comparison_idx = (
                np.sum(is_not_dominated_mask[:next_comparison_idx]) + 1
            )

        # Log statistics about the Pareto front
        if len(candidate_indices) > 0:
            selected_probs = probabilities[candidate_indices]
            selected_discs = discriminations[candidate_indices]
            logger.debug(
                f"Pareto front: selected {len(candidate_indices)} of {num_points} points "
                f"({100 * len(candidate_indices) / num_points:.1f}%)"
            )
            logger.debug(
                f"Pareto front probability range: [{np.min(selected_probs):.4f}, {np.max(selected_probs):.4f}]"
            )
            logger.debug(
                f"Pareto front discrimination range: [{np.min(selected_discs):.4f}, {np.max(selected_discs):.4f}]"
            )
        else:
            logger.warning("Pareto front is empty (unexpected)")

        return [int(idx) for idx in candidate_indices]

    @staticmethod
    def filter_by_diversity(
        selected_indices: list[int],
        probabilities: np.ndarray,
        discriminations: np.ndarray,
        observation_matrix: np.ndarray,
    ) -> list[int]:
        """
        Filters selected individuals to ensure diversity based on minimum distance.

        Args:
            selected_indices: List of integer indices for selected individuals.
            probabilities: 1D array of probabilities for all individuals.
            discriminations: 1D array of discriminations for all individuals.
            observation_matrix: 2D array of test results for all individuals.
        Returns:
            A filtered list of integer indices ensuring diversity.
        """

        def _duplicate_check(idx1: int, idx2: int) -> bool:
            """Checks if two individuals are duplicates based on objectives and results."""
            if (
                probabilities[idx1] == probabilities[idx2]
                and discriminations[idx1] == discriminations[idx2]
                and np.array_equal(
                    observation_matrix[:, idx1], observation_matrix[:, idx2]
                )
            ):
                return True
            return False

        filtered_indices: list[int] = []

        # Sort selected_indices to ensure a deterministic outcome
        # (though not strictly necessary, it's good practice).
        for idx in sorted(selected_indices):
            is_duplicate = False

            # Check if this individual is a duplicate of one we've *already added*
            for existing_idx in filtered_indices:
                if _duplicate_check(idx, existing_idx):
                    is_duplicate = True
                    break  # It's a duplicate, stop checking

            if not is_duplicate:
                # This is the first one of its kind we've seen
                filtered_indices.append(idx)

        # Log the change
        if len(selected_indices) != len(filtered_indices):
            logger.debug(
                f"Diversity filtering reduced {len(selected_indices)} indices "
                f"to {len(filtered_indices)} (removed exact duplicates)."
            )
        else:
            logger.debug("Diversity filtering found no exact duplicates.")

        return filtered_indices

    @staticmethod
    def get_pareto_indices(
        probabilities: np.ndarray, observation_matrix: np.ndarray
    ) -> list[int]:
        """
        Returns indices of Pareto-optimal individuals.

        This is the main public API method that orchestrates the full Pareto
        selection process by:
        1. Calculating discrimination scores from the observation matrix
        2. Finding the Pareto front based on both objectives

        Args:
            probabilities: 1D array of probabilities (Objective 1, to maximize).
            observation_matrix: 2D array where rows are code individuals and
                              columns are tests. Used to calculate discrimination
                              scores (Objective 2, to maximize).

        Returns:
            A list of integer indices for the individuals on the Pareto front.

        Raises:
            ValueError: If array dimensions are inconsistent.
        """
        if len(probabilities) != observation_matrix.shape[1]:
            logger.error(
                f"Dimension mismatch: {len(probabilities)} probabilities vs "
                f"{observation_matrix.shape[1]} tests in observation matrix"
            )
            raise ValueError(
                "Number of probabilities must match number of tests (columns) "
                "in observation matrix"
            )

        logger.info(
            f"Computing Pareto front for {len(probabilities)} tests "
            f"against {observation_matrix.shape[0]} codes"
        )

        # Calculate discrimination scores
        discriminations = ParetoSystem.calculate_discrimination(observation_matrix)

        logger.debug("Rounding probabilities and discriminations to 4 decimal places")
        discriminations = np.round(discriminations, 4)
        probabilities = np.round(probabilities, 4)

        # Calculate Pareto front
        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        pareto_indices = ParetoSystem.filter_by_diversity(
            pareto_indices, probabilities, discriminations, observation_matrix
        )

        logger.info(
            f"Pareto selection complete: {len(pareto_indices)} of {len(probabilities)} "
            f"tests selected ({100 * len(pareto_indices) / len(probabilities):.1f}%)"
        )

        return pareto_indices
