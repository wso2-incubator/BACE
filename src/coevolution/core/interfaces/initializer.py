# coevolution/core/interfaces/initializer.py
"""
Population initializer protocol — separated from breeding.
"""

from typing import Protocol

from .base import BaseIndividual
from .data import Problem


class IPopulationInitializer[T: BaseIndividual](Protocol):
    """
    Protocol for creating Generation 0 individuals.

    Separated from IBreedingStrategy so that initialization logic
    (batching, planning mode, retries) can evolve independently from
    the per-generation breeding loop.
    """

    def initialize(self, problem: Problem) -> list[T]:
        """
        Create the initial population for a given problem.

        Args:
            problem: Problem context (description, starter code, test cases).

        Returns:
            List of initial individuals (Generation 0).
            May return [] for populations that start empty (e.g. differential).
        """
        ...


__all__ = ["IPopulationInitializer"]
