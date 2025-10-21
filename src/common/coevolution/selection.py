"""
Selection strategies for evolutionary algorithms.

This module provides various selection methods used in evolutionary algorithms,
including tournament selection, roulette wheel selection, rank selection,
random selection, and elitism.
"""

from typing import Callable, Optional

import numpy as np


class SelectionStrategy:
    """
    A class that encapsulates various selection strategies for evolutionary algorithms.
    """

    @staticmethod
    def binary_tournament(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Performs binary tournament selection on a population based on fitness scores.

        Args:
            population: Array of individuals in the population.
            fitness: Array of fitness scores corresponding to each individual.

        Returns:
            Selected individual from the population.
        """
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
        if fitness[idx1] > fitness[idx2]:
            return np.asarray(population[idx1])
        else:
            return np.asarray(population[idx2])

    @staticmethod
    def elitism(
        population: np.ndarray, fitness: np.ndarray, num_elites: int
    ) -> np.ndarray:
        """
        Selects the top individuals from the population based on fitness scores.

        Args:
            population: Array of individuals in the population.
            fitness: Array of fitness scores corresponding to each individual.
            num_elites: Number of top individuals to select.

        Returns:
            Array of selected elite individuals.
        """
        elite_indices = np.argsort(fitness)[-num_elites:]
        return population[elite_indices]

    @staticmethod
    def roulette_wheel(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Performs roulette wheel selection on a population based on fitness scores.

        Args:
            population: Array of individuals in the population.
            fitness: Array of fitness scores corresponding to each individual.

        Returns:
            Selected individual from the population.
        """
        total_fitness = np.sum(fitness)
        if total_fitness == 0:
            # Handle edge case where all fitness values are zero
            return np.asarray(population[np.random.choice(len(population))])

        pick = np.random.rand() * total_fitness
        current = 0
        for individual, fit in zip(population, fitness):
            current += fit
            if current > pick:
                return np.asarray(individual)

        # Fallback in case of numerical errors
        return np.asarray(population[-1])

    @staticmethod
    def rank_selection(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Performs rank-based selection on a population.

        Individuals are selected based on their rank (position after sorting by fitness)
        rather than their raw fitness values. This helps when fitness values have a large
        range or when there are outliers.

        Args:
            population: Array of individuals in the population.
            fitness: Array of fitness scores corresponding to each individual.

        Returns:
            Selected individual from the population.
        """
        # Get ranks (1 to n, where n is population size)
        # Lower fitness gets lower rank, higher fitness gets higher rank
        ranks = np.argsort(np.argsort(fitness)) + 1

        # Use ranks as selection probabilities
        total_rank = np.sum(ranks)
        if total_rank == 0:
            return np.asarray(population[np.random.choice(len(population))])

        pick = np.random.rand() * total_rank
        current = 0
        for individual, rank in zip(population, ranks):
            current += rank
            if current > pick:
                return np.asarray(individual)

        # Fallback in case of numerical errors
        return np.asarray(population[-1])

    @staticmethod
    def random_selection(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Performs uniform random selection from the population.

        Selects an individual uniformly at random, ignoring fitness values.
        This can be useful for maintaining diversity or as a baseline comparison.

        Args:
            population: Array of individuals in the population.
            fitness: Array of fitness scores (unused, but kept for interface consistency).

        Returns:
            Randomly selected individual from the population.
        """
        random_idx = np.random.choice(len(population))
        return np.asarray(population[random_idx])

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

    @classmethod
    def _get_selection_function(
        cls, method: str
    ) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Internal method to get selection function by name."""
        methods: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
            "binary_tournament": cls.binary_tournament,
            "roulette_wheel": cls.roulette_wheel,
            "rank_selection": cls.rank_selection,
            "random_selection": cls.random_selection,
        }
        return methods.get(method)

    @classmethod
    def select_parents(
        cls,
        population: np.ndarray,
        fitness: np.ndarray,
        method: str = "binary_tournament",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Selects two different parents from the population using the specified selection method.

        Args:
            population: Array of individuals in the population.
            fitness: Array of fitness scores corresponding to each individual.
            method: Selection method to use. Available methods can be retrieved using
                   get_available_methods(). Default is "binary_tournament".

        Returns:
            Tuple containing two different selected parents.

        Raises:
            ValueError: If an invalid selection method is specified or if population size < 2.
        """
        if len(population) < 2:
            raise ValueError(
                "Population must have at least 2 individuals to select different parents"
            )

        selection_func = cls._get_selection_function(method)
        if selection_func is None:
            available = ", ".join(cls.get_available_methods())
            raise ValueError(
                f"Invalid selection method: '{method}'. Available methods: {available}"
            )

        parent1 = selection_func(population, fitness)

        # Keep selecting until we get a different parent
        max_attempts = 100
        for _ in range(max_attempts):
            parent2 = selection_func(population, fitness)
            # Check if parents are different (compare by value, not reference)
            if not np.array_equal(parent1, parent2):
                break
        else:
            # If all attempts failed (unlikely), select a random different individual
            available_indices = np.arange(len(population))
            parent1_idx = np.where(
                np.all(population == parent1, axis=tuple(range(1, parent1.ndim)))
            )[0][0]
            available_indices = available_indices[available_indices != parent1_idx]
            parent2 = population[np.random.choice(available_indices)]

        return parent1, parent2
