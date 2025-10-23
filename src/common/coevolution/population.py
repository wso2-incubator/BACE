"""
Population management for coevolutionary algorithms.

This module provides the Population class for managing code solutions or test cases
in evolutionary algorithms with Bayesian belief updating.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class Population:
    """
    Represents a mutable population of code solutions or test cases.

    This class encapsulates both the actual code/test strings and their
    associated Bayesian correctness probabilities. It's designed to work
    with the Bayesian belief updating system and supports dynamic population
    updates across generations.

    Attributes:
        individuals: List of code strings or test case strings
        probabilities: Numpy array of correctness probabilities for each individual
        size: Population size (automatically updated when individuals change)
        generation: Current generation number
        metadata: Dictionary for tracking additional information
    """

    def __init__(
        self,
        individuals: List[str],
        probabilities: np.ndarray,
        generation: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a Population.

        Args:
            individuals: List of code strings or test case strings
            probabilities: Numpy array of correctness probabilities
            generation: Current generation number (default: 0)
            metadata: Optional dictionary for additional information

        Raises:
            ValueError: If validation checks fail
        """
        # Validate inputs
        if len(individuals) != len(probabilities):
            raise ValueError(
                f"Number of individuals ({len(individuals)}) must match "
                f"number of probabilities ({len(probabilities)})"
            )
        if len(individuals) == 0:
            raise ValueError("Population cannot be empty")

        self._individuals = individuals
        self._probabilities = probabilities
        self.generation = generation
        self.metadata = metadata if metadata is not None else {}

    @property
    def individuals(self) -> List[str]:
        """Get the list of individuals."""
        return self._individuals

    @property
    def probabilities(self) -> np.ndarray:
        """Get the probability array."""
        return self._probabilities

    @property
    def size(self) -> int:
        """Get the current population size."""
        return len(self._individuals)

    def _validate_consistency(self) -> None:
        """Internal method to validate population consistency."""
        if len(self._individuals) != len(self._probabilities):
            raise ValueError(
                f"Population consistency violated: {len(self._individuals)} individuals "
                f"but {len(self._probabilities)} probabilities"
            )

    def get_best_individual(self) -> tuple[str, float]:
        """
        Get the individual with the highest correctness probability.

        Returns:
            Tuple of (best_individual, best_probability)
        """
        best_idx = int(np.argmax(self._probabilities))
        return self._individuals[best_idx], float(self._probabilities[best_idx])

    def get_top_k_individuals(self, k: int) -> List[tuple[str, float]]:
        """
        Get the top k individuals by correctness probability.

        Args:
            k: Number of top individuals to retrieve

        Returns:
            List of (individual, probability) tuples, sorted by probability (descending)
        """
        if k > len(self._individuals):
            k = len(self._individuals)
        top_indices = np.argsort(self._probabilities)[-k:][::-1]
        return [
            (self._individuals[int(idx)], float(self._probabilities[idx]))
            for idx in top_indices
        ]

    def update_probabilities(self, new_probabilities: np.ndarray) -> None:
        """
        Update the correctness probabilities for all individuals.

        Args:
            new_probabilities: New probability array

        Raises:
            ValueError: If the size doesn't match the population size
        """
        if len(new_probabilities) != len(self._individuals):
            raise ValueError(
                f"New probabilities size ({len(new_probabilities)}) must match "
                f"population size ({len(self._individuals)})"
            )
        self._probabilities = new_probabilities

    def replace_individuals(
        self,
        new_individuals: List[str],
        new_probabilities: np.ndarray,
    ) -> None:
        """
        Replace the entire population with new individuals and probabilities.

        This is useful for generational replacement in evolutionary algorithms.

        Args:
            new_individuals: New list of individuals
            new_probabilities: New probability array

        Raises:
            ValueError: If validation checks fail
        """
        if len(new_individuals) != len(new_probabilities):
            raise ValueError(
                f"Number of new individuals ({len(new_individuals)}) must match "
                f"number of new probabilities ({len(new_probabilities)})"
            )
        if len(new_individuals) == 0:
            raise ValueError("Cannot replace with empty population")

        self._individuals = new_individuals
        self._probabilities = new_probabilities
        self._validate_consistency()

    def add_individual(self, individual: str, probability: float) -> None:
        """
        Add a single individual to the population.

        Args:
            individual: Code string to add
            probability: Correctness probability for this individual

        Raises:
            ValueError: If probability is not in [0, 1]
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")

        self._individuals.append(individual)
        self._probabilities = np.append(self._probabilities, probability)
        self._validate_consistency()

    def remove_individual(self, index: int) -> tuple[str, float]:
        """
        Remove an individual at the specified index.

        Args:
            index: Index of individual to remove

        Returns:
            Tuple of (removed_individual, removed_probability)

        Raises:
            ValueError: If population would become empty or index is invalid
            IndexError: If index is out of bounds
        """
        if len(self._individuals) <= 1:
            raise ValueError("Cannot remove from population with only one individual")
        if index < 0 or index >= len(self._individuals):
            raise IndexError(
                f"Index {index} out of bounds for population size {len(self._individuals)}"
            )

        removed_individual = self._individuals.pop(index)
        removed_probability = float(self._probabilities[index])
        self._probabilities = np.delete(self._probabilities, index)
        self._validate_consistency()

        return removed_individual, removed_probability

    def replace_individual(
        self, index: int, individual: str, probability: float
    ) -> None:
        """
        Replace a single individual at the specified index.

        Args:
            index: Index of individual to replace
            individual: New code string
            probability: New correctness probability

        Raises:
            ValueError: If probability is not in [0, 1]
            IndexError: If index is out of bounds
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        if index < 0 or index >= len(self._individuals):
            raise IndexError(
                f"Index {index} out of bounds for population size {len(self._individuals)}"
            )

        self._individuals[index] = individual
        self._probabilities[index] = probability

    def increment_generation(self) -> None:
        """Increment the generation counter."""
        self.generation += 1

    def __len__(self) -> int:
        """Get the size of the population."""
        return len(self._individuals)

    def __repr__(self) -> str:
        """String representation of the population."""
        return (
            f"Population(size={len(self._individuals)}, generation={self.generation}, "
            f"avg_prob={np.mean(self._probabilities):.4f})"
        )

    def __getitem__(self, index: int) -> tuple[str, float]:
        """
        Get an individual and its probability by index.

        Args:
            index: Index of individual to retrieve

        Returns:
            Tuple of (individual, probability)
        """
        return self._individuals[index], float(self._probabilities[index])
