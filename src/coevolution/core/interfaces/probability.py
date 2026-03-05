# coevolution/core/interfaces/probability.py
"""
Probability-assignment protocol for offspring in coevolutionary algorithms.
"""

from typing import Protocol

from .types import Operation, ParentProbabilities


class IProbabilityAssigner(Protocol):
    """
    Protocol defining the contract for a probability assignment policy.
    Injected into operators at construction time.

    initial_prior is owned by the assigner — it must be set at construction
    time and readable by operators that need the Gen-0 default.

    The concrete implementation decides how to combine parent probabilities
    (mean, min, max, etc.) and handles the initial-population edge case.
    """

    initial_prior: float

    def assign_probability(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
    ) -> float:
        """
        Calculates the probability for a new offspring.

        Args:
            operation: The op that created the child (e.g., "crossover", "mutation").
            parent_probs: A list of the parent(s)' probabilities.

        Returns:
            The new probability for the offspring.
        """
        ...


__all__ = ["IProbabilityAssigner"]
