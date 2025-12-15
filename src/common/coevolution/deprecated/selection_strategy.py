"""
Selection strategies for evolutionary algorithms.

This module provides various selection methods used in parent selection during
coevolutionary algorithms. All strategies work with probability arrays and return
indices, maintaining a single source of truth in the population objects.

Available Methods:
- Binary Tournament: Selects the better of two randomly chosen individuals
- Roulette Wheel: Probability-proportional selection (fitness-proportionate)
- Rank Selection: Selection based on rank rather than raw probabilities
- Random Selection: Uniform random selection (ignores probabilities)

Design:
- Configuration-based: Selection method is specified at initialization
- Stateless operations: All selection methods are pure functions
- Returns indices: Allows callers to manage their own population objects
"""

from enum import Enum

import numpy as np
from loguru import logger

from .core.interfaces import ISelectionStrategy


class SelectionMethod(Enum):
    """Enumeration of available selection methods."""

    BINARY_TOURNAMENT = "binary_tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_SELECTION = "rank_selection"
    RANDOM_SELECTION = "random_selection"


class SelectionStrategy(ISelectionStrategy):
    """
    Configurable selection strategy for evolutionary algorithms.

    This class implements various selection methods for parent selection.
    The method is configured at initialization and used consistently
    throughout the evolutionary process.

    All selection methods are implemented as static methods for independent
    testing and reusability. The instance stores the configured method.

    Examples:
        >>> # Using enum directly
        >>> strategy = SelectionStrategy(SelectionMethod.BINARY_TOURNAMENT)
        >>> # Using string (convenient for config files)
        >>> strategy = SelectionStrategy("binary_tournament")
        >>> # Using the method
        >>> parent_idx = strategy.select(probabilities)
        >>> parent1_idx, parent2_idx = strategy.select_parents(probabilities)

    Args:
        method: The selection method to use. Can be either:
                - A SelectionMethod enum value (e.g., SelectionMethod.BINARY_TOURNAMENT)
                - A string (e.g., "binary_tournament", "roulette_wheel")

    Raises:
        ValueError: If method string is not a valid selection method name
        TypeError: If method is neither a SelectionMethod enum nor a string
    """

    # Class-level mapping of string names to enum values for easy lookup
    _STRING_TO_ENUM = {
        "binary_tournament": SelectionMethod.BINARY_TOURNAMENT,
        "roulette_wheel": SelectionMethod.ROULETTE_WHEEL,
        "rank_selection": SelectionMethod.RANK_SELECTION,
        "random_selection": SelectionMethod.RANDOM_SELECTION,
    }

    def __init__(
        self, method: SelectionMethod | str = SelectionMethod.BINARY_TOURNAMENT
    ):
        """
        Initialize the selection strategy with a specific method.

        Args:
            method: Selection method to use. Can be either:
                    - A SelectionMethod enum value
                    - A string name (e.g., "binary_tournament")

        Raises:
            ValueError: If method string is not valid
            TypeError: If method is neither enum nor string
        """
        # Convert string to enum if necessary
        if isinstance(method, str):
            method_lower = method.lower()
            if method_lower not in self._STRING_TO_ENUM:
                available = ", ".join(f"'{m}'" for m in self._STRING_TO_ENUM.keys())
                raise ValueError(
                    f"Invalid selection method: '{method}'. "
                    f"Available methods: {available}"
                )
            method = self._STRING_TO_ENUM[method_lower]
        elif not isinstance(method, SelectionMethod):
            raise TypeError(
                f"method must be a SelectionMethod enum or string, "
                f"got {type(method).__name__}"
            )

        self._method = method

        # Map enum values to static method implementations
        self._method_map = {
            SelectionMethod.BINARY_TOURNAMENT: self.binary_tournament,
            SelectionMethod.ROULETTE_WHEEL: self.roulette_wheel,
            SelectionMethod.RANK_SELECTION: self.rank_selection,
            SelectionMethod.RANDOM_SELECTION: self.random_selection,
        }

        logger.debug(f"Initialized SelectionStrategy with method: {method.value}")

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """
        Returns a list of available selection method names as strings.

        This is useful for validation in configuration files or user interfaces.

        Returns:
            List of available selection method names (strings).
        """
        return list(cls._STRING_TO_ENUM.keys())

    @property
    def method(self) -> SelectionMethod:
        """Returns the configured selection method."""
        return self._method

    @staticmethod
    def binary_tournament(probabilities: np.ndarray) -> int:
        """
        Performs binary tournament selection based on probabilities.

        Randomly selects two individuals and returns the one with higher probability.
        For single-element populations, returns that element.

        Args:
            probabilities: Array of selection probabilities

        Returns:
            Index of the selected individual (winner of tournament)
        """
        # Handle single-element case
        if len(probabilities) == 1:
            logger.trace("Binary tournament: only one individual, returning index 0")
            return 0

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
    def roulette_wheel(probabilities: np.ndarray) -> int:
        """
        Performs roulette wheel (fitness-proportionate) selection.

        Individuals are selected with probability proportional to their
        fitness scores. Also known as fitness-proportionate selection.

        Uses vectorized numpy operations for efficiency.

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

        # Vectorized selection using numpy's built-in weighted choice
        normalized_probs = probabilities / total_prob
        idx = np.random.choice(len(probabilities), p=normalized_probs)

        logger.trace(
            f"Roulette wheel: selected idx {idx} (prob={probabilities[idx]:.4f})"
        )

        return int(idx)

    @staticmethod
    def rank_selection(probabilities: np.ndarray) -> int:
        """
        Performs rank-based selection.

        Individuals are selected based on their rank (position after sorting
        by probability) rather than raw probability values. This helps when
        probability values have a large range or when there are outliers.

        Uses vectorized numpy operations for efficiency.

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

        # Vectorized selection using numpy's built-in weighted choice
        normalized_ranks = ranks / total_rank
        idx = np.random.choice(len(probabilities), p=normalized_ranks)

        logger.trace(
            f"Rank selection: selected idx {idx} "
            f"(rank={ranks[idx]}, prob={probabilities[idx]:.4f})"
        )

        return int(idx)

    @staticmethod
    def random_selection(probabilities: np.ndarray) -> int:
        """
        Performs uniform random selection.

        Selects an individual uniformly at random, ignoring probability values.
        This can be useful for maintaining diversity or as a baseline comparison.

        Args:
            probabilities: Array of selection probabilities (not used, kept for API consistency)

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
            probabilities: A 1D array of fitness/probability scores
                           for the population.

        Returns:
            The integer index of the selected individual.
        """
        # Use method map to call the appropriate selection function
        selection_func = self._method_map[self._method]
        return selection_func(probabilities)

    def select_parents(self, probabilities: np.ndarray) -> tuple[int, int]:
        """
        Selects two *different* parent indices from the population.

        Args:
            probabilities: A 1D array of fitness/probability scores
                           for the population.

        Returns:
            A tuple of two different integer indices (parent1_idx, parent2_idx).

        Raises:
            ValueError: If the population size (inferred from `probabilities`)
                        is less than 2.
        """
        logger.trace(f"Selecting two parents using method: {self._method.value}")

        if len(probabilities) < 2:
            raise ValueError(
                "Population must have at least 2 individuals to select different parents"
            )

        parent1_idx = self.select(probabilities)

        # Keep selecting until we get a different parent
        max_attempts = 100
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
