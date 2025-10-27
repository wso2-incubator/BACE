"""
Selection strategies for evolutionary algorithms.

This module provides various selection methods used in evolutionary algorithms,
including tournament selection, roulette wheel selection, rank selection,
random selection, and elitism.

Works directly with probability arrays and returns indices for single source of truth.
"""

import numpy as np
from loguru import logger


class SelectionStrategy:
    """
    A class that encapsulates various selection strategies for evolutionary algorithms.

    All methods work directly with probability arrays and return indices, maintaining
    a single source of truth - the population objects themselves hold the individuals.

    Design Notes:
    - Individual selection algorithms (binary_tournament, roulette_wheel, etc.) are
      @staticmethod because they don't depend on instance state - they're pure functions
      that operate on probability arrays.
    - The configured method is stored in instance state (self.method) for consistent
      selection throughout the evolutionary process.
    - Returns indices instead of individuals, allowing callers to retrieve individuals
      from their own population objects (single source of truth).
    - This design separates algorithm implementation (static) from configuration (instance),
      making it easy to add new selection methods and test them independently.

    Args:
        method: The selection method to use. Available: "binary_tournament",
                "roulette_wheel", "rank_selection", "random_selection"

    Raises:
        ValueError: If an invalid selection method is specified
    """

    def __init__(self, method: str = "binary_tournament"):
        """
        Initialize the selection strategy with a specific method.

        Args:
            method: Selection method to use for all selections

        Raises:
            ValueError: If the method is not valid
        """
        if method not in SelectionStrategy.get_available_methods():
            available = ", ".join(SelectionStrategy.get_available_methods())
            raise ValueError(
                f"Invalid selection method: '{method}'. Available methods: {available}"
            )
        self.method = method
        logger.debug(f"Initialized SelectionStrategy with method: {method}")

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """
        Returns a list of available selection method names.

        Returns:
            List of available selection method names.
        """
        return [
            "binary_tournament",
            "roulette_wheel",
            "rank_selection",
            "random_selection",
        ]

    @staticmethod
    def binary_tournament(probabilities: np.ndarray) -> int:
        """
        Performs binary tournament selection based on probabilities.

        Args:
            probabilities: Array of selection probabilities

        Returns:
            Index of the selected individual
        """
        idx1, idx2 = np.random.choice(len(probabilities), size=2, replace=False)
        prob1 = probabilities[idx1]
        prob2 = probabilities[idx2]

        if prob1 > prob2:
            winner_idx = idx1
            winner_prob = prob1
        else:
            winner_idx = idx2
            winner_prob = prob2

        logger.trace(
            f"Binary tournament: idx {idx1} (prob={prob1:.4f}) vs "
            f"idx {idx2} (prob={prob2:.4f}) → winner idx {winner_idx} (prob={winner_prob:.4f})"
        )

        return int(winner_idx)

    @staticmethod
    def elitism(probabilities: np.ndarray, num_elites: int) -> list[int]:
        """
        Selects the top individuals based on probabilities.

        Args:
            probabilities: Array of selection probabilities
            num_elites: Number of top individuals to select

        Returns:
            List of indices for elite individuals (sorted by probability, descending)
        """
        # Get indices sorted by probability (descending)
        elite_indices = np.argsort(probabilities)[::-1][:num_elites]

        if len(elite_indices) > 0:
            elite_probs = probabilities[elite_indices]
            logger.debug(
                f"Elitism: selected {len(elite_indices)} elites, "
                f"prob range=[{np.min(elite_probs):.4f}, {np.max(elite_probs):.4f}], "
                f"avg={np.mean(elite_probs):.4f}"
            )
            logger.trace(f"Elite probabilities: {[f'{p:.4f}' for p in elite_probs]}")

        return [int(idx) for idx in elite_indices]

    @staticmethod
    def pareto_front(
        probabilities: np.ndarray, discriminations: np.ndarray
    ) -> list[int]:
        """
        Selects the Pareto-efficient individuals by maximizing two objectives:
        probabilities and discriminations. Returns the list of indices corresponding to Pareto-optimal points.

        Args:
            probabilities: 1-D array of selection probabilities (to maximize)
            discriminations: 1-D array of discriminative power values (to maximize)

        Returns:
            List of integer indices of Pareto-efficient individuals.
        """
        if len(probabilities) != len(discriminations):
            raise ValueError(
                "Probabilities and discriminations must have the same length"
            )

        # Build objectives array (n_points, 2) for maximization
        objectives = np.column_stack((probabilities, discriminations))

        num_points = objectives.shape[0]
        candidate_indices = np.arange(num_points)
        current_objectives = objectives.copy()
        next_comparison_idx = 0

        while next_comparison_idx < len(current_objectives):
            comp = current_objectives[next_comparison_idx]
            # A point is NOT dominated by comp if any objective is strictly greater
            is_not_dominated_mask = np.any(current_objectives > comp, axis=1)
            # keep the comparison point itself
            is_not_dominated_mask[next_comparison_idx] = True

            candidate_indices = candidate_indices[is_not_dominated_mask]
            current_objectives = current_objectives[is_not_dominated_mask]

            next_comparison_idx = (
                np.sum(is_not_dominated_mask[:next_comparison_idx]) + 1
            )

        if len(candidate_indices) > 0:
            sel_probs = probabilities[candidate_indices]
            sel_discs = discriminations[candidate_indices]
            logger.debug(
                f"Pareto front: selected {len(candidate_indices)} points, "
                f"prob range=[{np.min(sel_probs):.4f}, {np.max(sel_probs):.4f}], "
                f"disc range=[{np.min(sel_discs):.4f}, {np.max(sel_discs):.4f}]"
            )

        return [int(idx) for idx in candidate_indices]

    @staticmethod
    def roulette_wheel(probabilities: np.ndarray) -> int:
        """
        Performs roulette wheel selection based on probabilities.

        Args:
            probabilities: Array of selection probabilities

        Returns:
            Index of the selected individual
        """
        total_prob = np.sum(probabilities)

        if total_prob == 0:
            # Handle edge case where all probabilities are zero
            logger.warning(
                "Roulette wheel: all probabilities are zero, using random selection"
            )
            idx = np.random.choice(len(probabilities))
            return int(idx)

        pick = np.random.rand() * total_prob
        current = 0
        for i in range(len(probabilities)):
            current += probabilities[i]
            if current > pick:
                selected_prob = probabilities[i]
                logger.trace(
                    f"Roulette wheel: pick={pick:.4f}/{total_prob:.4f}, "
                    f"selected idx {i} (prob={selected_prob:.4f})"
                )
                return int(i)

        # Fallback in case of numerical errors
        logger.trace("Roulette wheel: numerical fallback, selecting last individual")
        return len(probabilities) - 1

    @staticmethod
    def rank_selection(probabilities: np.ndarray) -> int:
        """
        Performs rank-based selection.

        Individuals are selected based on their rank (position after sorting by probability)
        rather than their raw probability values. This helps when probability values have a large
        range or when there are outliers.

        Args:
            probabilities: Array of selection probabilities

        Returns:
            Index of the selected individual
        """
        # Get ranks (1 to n, where n is population size)
        # Lower probability gets lower rank, higher probability gets higher rank
        ranks = np.argsort(np.argsort(probabilities)) + 1

        # Use ranks as selection probabilities
        total_rank = np.sum(ranks)
        if total_rank == 0:
            logger.warning("Rank selection: all ranks are zero, using random selection")
            idx = np.random.choice(len(probabilities))
            return int(idx)

        pick = np.random.rand() * total_rank
        current = 0
        for i, rank in enumerate(ranks):
            current += rank
            if current > pick:
                selected_prob = probabilities[i]
                logger.trace(
                    f"Rank selection: pick={pick:.1f}/{total_rank:.1f}, "
                    f"selected idx {i} (rank={rank}, prob={selected_prob:.4f})"
                )
                return int(i)

        # Fallback in case of numerical errors
        logger.trace("Rank selection: numerical fallback, selecting last individual")
        return len(probabilities) - 1

    @staticmethod
    def random_selection(probabilities: np.ndarray) -> int:
        """
        Performs uniform random selection.

        Selects an individual uniformly at random, ignoring probability values.
        This can be useful for maintaining diversity or as a baseline comparison.

        Args:
            probabilities: Array of selection probabilities (not used, but kept for API consistency)

        Returns:
            Index of the selected individual
        """
        idx = np.random.choice(len(probabilities))
        selected_prob = probabilities[idx]
        logger.trace(f"Random selection: selected idx {idx} (prob={selected_prob:.4f})")
        return int(idx)

    def select(self, probabilities: np.ndarray) -> int:
        """
        Selects a single individual using the configured selection method.

        Args:
            probabilities: Array of selection probabilities

        Returns:
            Index of the selected individual
        """
        selection_func = getattr(self, self.method)
        result: int = selection_func(probabilities)
        return result

    def select_parents(self, probabilities: np.ndarray) -> tuple[int, int]:
        """
        Selects two different parents using the configured selection method.

        Args:
            probabilities: Array of selection probabilities

        Returns:
            Tuple containing indices of two different selected parents

        Raises:
            ValueError: If population size < 2.
        """
        logger.trace(f"Selecting two parents using method: {self.method}")

        if len(probabilities) < 2:
            raise ValueError(
                "Population must have at least 2 individuals to select different parents"
            )

        parent1_idx = self.select(probabilities)

        # Keep selecting until we get a different parent
        max_attempts = 100
        attempts = 0
        for attempts in range(1, max_attempts + 1):
            parent2_idx = self.select(probabilities)
            # Check if parents are different indices
            if parent1_idx != parent2_idx:
                if attempts > 10:
                    logger.debug(f"Found different parent after {attempts} attempts")
                break
        else:
            # If all attempts failed (unlikely), force selection of different individual
            logger.warning(
                f"Failed to find different parent after {max_attempts} attempts, "
                "forcing selection of different individual"
            )
            # Select a random index different from parent1
            available_indices = [
                i for i in range(len(probabilities)) if i != parent1_idx
            ]
            parent2_idx = int(np.random.choice(available_indices))

        logger.trace(
            f"Selected parent indices: {parent1_idx} (prob={probabilities[parent1_idx]:.4f}), "
            f"{parent2_idx} (prob={probabilities[parent2_idx]:.4f})"
        )

        return parent1_idx, parent2_idx
