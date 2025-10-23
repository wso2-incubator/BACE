"""
Selection strategies for evolutionary algorithms.

This module provides various selection methods used in evolutionary algorithms,
including tournament selection, roulette wheel selection, rank selection,
random selection, and elitism.

Uses the BasePopulation class for cleaner API.
"""

from typing import Callable, List, Optional

import numpy as np

from .population import BasePopulation


class SelectionStrategy:
    """
    A class that encapsulates various selection strategies for evolutionary algorithms.

    All methods work with BaseBasePopulation objects (CodePopulation or TestPopulation).
    """

    @staticmethod
    def binary_tournament(population: BasePopulation) -> tuple[str, float]:
        """
        Performs binary tournament selection on a population based on probabilities.

        Args:
            population: BaseBasePopulation object containing individuals and their probabilities

        Returns:
            Tuple of (selected_individual, selected_probability)
        """
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
        prob1 = population.probabilities[idx1]
        prob2 = population.probabilities[idx2]

        if prob1 > prob2:
            return population[idx1]
        else:
            return population[idx2]

    @staticmethod
    def elitism(population: BasePopulation, num_elites: int) -> List[tuple[str, float]]:
        """
        Selects the top individuals from the population based on probabilities.

        Args:
            population: BaseBasePopulation object containing individuals and their probabilities
            num_elites: Number of top individuals to select

        Returns:
            List of (individual, probability) tuples for elite individuals
        """
        return population.get_top_k_individuals(num_elites)

    @staticmethod
    def roulette_wheel(population: BasePopulation) -> tuple[str, float]:
        """
        Performs roulette wheel selection on a population based on probabilities.

        Args:
            population: BaseBasePopulation object containing individuals and their probabilities

        Returns:
            Tuple of (selected_individual, selected_probability)
        """
        probabilities = population.probabilities
        total_prob = np.sum(probabilities)

        if total_prob == 0:
            # Handle edge case where all probabilities are zero
            idx = np.random.choice(len(population))
            return population[idx]

        pick = np.random.rand() * total_prob
        current = 0
        for i in range(len(population)):
            current += probabilities[i]
            if current > pick:
                return population[i]

        # Fallback in case of numerical errors
        return population[len(population) - 1]

    @staticmethod
    def rank_selection(population: BasePopulation) -> tuple[str, float]:
        """
        Performs rank-based selection on a population.

        Individuals are selected based on their rank (position after sorting by probability)
        rather than their raw probability values. This helps when probability values have a large
        range or when there are outliers.

        Args:
            population: BaseBasePopulation object containing individuals and their probabilities

        Returns:
            Tuple of (selected_individual, selected_probability)
        """
        probabilities = population.probabilities
        # Get ranks (1 to n, where n is population size)
        # Lower probability gets lower rank, higher probability gets higher rank
        ranks = np.argsort(np.argsort(probabilities)) + 1

        # Use ranks as selection probabilities
        total_rank = np.sum(ranks)
        if total_rank == 0:
            idx = np.random.choice(len(population))
            return population[idx]

        pick = np.random.rand() * total_rank
        current = 0
        for i, rank in enumerate(ranks):
            current += rank
            if current > pick:
                return population[i]

        # Fallback in case of numerical errors
        return population[len(population) - 1]

    @staticmethod
    def random_selection(population: BasePopulation) -> tuple[str, float]:
        """
        Performs uniform random selection from the population.

        Selects an individual uniformly at random, ignoring probability values.
        This can be useful for maintaining diversity or as a baseline comparison.

        Args:
            population: BaseBasePopulation object containing individuals and their probabilities

        Returns:
            Tuple of (selected_individual, selected_probability)
        """
        idx = np.random.choice(len(population))
        return population[idx]

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
    ) -> Optional[Callable[[BasePopulation], tuple[str, float]]]:
        """Internal method to get selection function by name."""
        methods: dict[str, Callable[[BasePopulation], tuple[str, float]]] = {
            "binary_tournament": cls.binary_tournament,
            "roulette_wheel": cls.roulette_wheel,
            "rank_selection": cls.rank_selection,
            "random_selection": cls.random_selection,
        }
        return methods.get(method)

    @classmethod
    def select_parents(
        cls,
        population: BasePopulation,
        method: str = "binary_tournament",
    ) -> tuple[tuple[str, float], tuple[str, float]]:
        """
        Selects two different parents from the population using the specified selection method.

        Args:
            population: BaseBasePopulation object to select from
            method: Selection method to use. Available methods can be retrieved using
                   get_available_methods(). Default is "binary_tournament".

        Returns:
            Tuple containing two different selected parents, each as (individual, probability)

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

        parent1_individual, parent1_prob = selection_func(population)

        # Keep selecting until we get a different parent
        max_attempts = 100
        for _ in range(max_attempts):
            parent2_individual, parent2_prob = selection_func(population)
            # Check if parents are different (compare by value)
            if parent1_individual != parent2_individual:
                break
        else:
            # If all attempts failed (unlikely), select a random different individual
            # Find index of parent1
            parent1_idx = None
            for i in range(len(population)):
                if population.individuals[i] == parent1_individual:
                    parent1_idx = i
                    break

            if parent1_idx is not None:
                available_indices = [
                    i for i in range(len(population)) if i != parent1_idx
                ]
                if available_indices:
                    idx = np.random.choice(available_indices)
                    parent2_individual, parent2_prob = population[idx]

        return (parent1_individual, parent1_prob), (parent2_individual, parent2_prob)
        return (parent1_individual, parent1_prob), (parent2_individual, parent2_prob)
