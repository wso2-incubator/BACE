"""
Probability assignment strategies for offspring in coevolutionary algorithms.

This module implements the IProbabilityAssigner interface, providing various
strategies for assigning correctness probabilities to newly created offspring
based on their parents' probabilities and the genetic operation used.

Available Strategies:
    - MeanProbabilityAssigner: Inherits average of parent probabilities
    - MaxProbabilityAssigner: Inherits maximum of parent probabilities (optimistic)
    - MinProbabilityAssigner: Inherits minimum of parent probabilities (pessimistic)
"""

from enum import Enum
from typing import Union

import numpy as np
from loguru import logger

from .core.interfaces import IProbabilityAssigner, Operations, ParentProbabilities


class AssignmentStrategy(str, Enum):
    """
    Enumeration of available probability assignment strategies.

    These strategies determine how offspring inherit probability from their parents.
    """

    MEAN = "mean"  # Average of parent probabilities
    MAX = "max"  # Maximum of parent probabilities (optimistic)
    MIN = "min"  # Minimum of parent probabilities (pessimistic)


class ProbabilityAssigner(IProbabilityAssigner):
    """
    Configurable probability assigner supporting multiple inheritance strategies.

    This class assigns correctness probabilities to new offspring based on:
    1. The genetic operation that created them (crossover, mutation, edit, etc.)
    2. Their parent(s)' probabilities
    3. A configurable assignment strategy

    The strategy determines how parents' probabilities are combined:
    - MEAN: Simple average (balanced, default)
    - MAX: Most optimistic parent (exploration-focused)
    - MIN: Most pessimistic parent (conservative)

    Args:
        strategy: Assignment strategy to use (enum or string)

    Example:
        >>> # Conservative strategy (inherits min of parents)
        >>> assigner = ProbabilityAssigner(AssignmentStrategy.MIN)
        >>>
        >>> # Optimistic strategy with max
        >>> assigner = ProbabilityAssigner("max")
    """

    def __init__(
        self,
        strategy: Union[AssignmentStrategy, str] = AssignmentStrategy.MEAN,
    ) -> None:
        """
        Initialize the probability assigner.

        Args:
            strategy: Assignment strategy (enum or string name)

        Raises:
            ValueError: If strategy is invalid
        """
        # Convert string to enum if necessary
        if isinstance(strategy, str):
            try:
                self.strategy = AssignmentStrategy(strategy.lower())
            except ValueError as e:
                valid = [s.value for s in AssignmentStrategy]
                msg = f"Invalid strategy '{strategy}'. Valid options: {valid}"
                logger.error(msg)
                raise ValueError(msg) from e
        else:
            self.strategy = strategy

        logger.debug(
            f"Initialized ProbabilityAssigner with strategy={self.strategy.value}"
        )

        # Dispatch table for assignment strategies
        self._strategy_methods = {
            AssignmentStrategy.MEAN: self._assign_mean,
            AssignmentStrategy.MAX: self._assign_max,
            AssignmentStrategy.MIN: self._assign_min,
        }

    def assign_probability(
        self,
        operation: Operations,
        parent_probs: ParentProbabilities,
        initial_prior: float,
    ) -> float:
        """
        Calculate the probability for a new offspring.

        For initial population members (Operations.INITIAL), returns the initial_prior.
        For offspring created by genetic operations, applies the configured strategy
        to combine parent probabilities.

        Args:
            operation: The operation that created the offspring
            parent_probs: List of parent probability values (may be empty)
            initial_prior: Default prior probability for initial population

        Returns:
            Assigned probability for the offspring

        Raises:
            ValueError: If parent_probs is empty for non-initial operations
        """
        # Initial population members get the prior
        if operation == Operations.INITIAL:
            logger.trace(f"Assigning initial prior: {initial_prior:.4f}")
            return initial_prior

        # Validate we have parent probabilities for genetic operations
        if not parent_probs:
            msg = f"Cannot assign probability for {operation.value}: no parent probabilities provided"
            logger.error(msg)
            raise ValueError(msg)

        # Apply the configured strategy
        strategy_func = self._strategy_methods[self.strategy]
        assigned_prob = strategy_func(operation, parent_probs, initial_prior)

        if assigned_prob < initial_prior:
            # TODO: Change back to warning after debugging population degradation issue
            logger.warning(
                f"Assigned probability {assigned_prob:.4f} is less than initial prior {initial_prior:.4f}"
            )

        logger.trace(
            f"Assigned probability {assigned_prob:.4f} for {operation.value} "
            f"(strategy={self.strategy.value}, parent_probs={parent_probs})"
        )

        return assigned_prob

    # --- Strategy Implementations ---

    def _assign_mean(
        self,
        operation: Operations,
        parent_probs: ParentProbabilities,
        initial_prior: float,
    ) -> float:
        """
        Assign the mean (average) of parent probabilities.

        This is a balanced strategy that treats all parents equally.
        Works well for both crossover (2 parents) and mutation (1 parent).

        Args:
            operation: The genetic operation (unused in this strategy)
            parent_probs: Parent probability values
            initial_prior: Default prior (unused in this strategy)

        Returns:
            Mean of parent probabilities
        """
        return float(np.mean(parent_probs))

    def _assign_max(
        self,
        operation: Operations,
        parent_probs: ParentProbabilities,
        initial_prior: float,
    ) -> float:
        """
        Assign the maximum of parent probabilities (optimistic strategy).

        This strategy assumes offspring inherit the best traits from their parents.
        Encourages exploration by being optimistic about genetic combinations.

        Args:
            operation: The genetic operation (unused in this strategy)
            parent_probs: Parent probability values
            initial_prior: Default prior (unused in this strategy)

        Returns:
            Maximum of parent probabilities
        """
        return float(np.max(parent_probs))

    def _assign_min(
        self,
        operation: Operations,
        parent_probs: ParentProbabilities,
        initial_prior: float,
    ) -> float:
        """
        Assign the minimum of parent probabilities (pessimistic strategy).

        This is a conservative strategy assuming offspring can only be as good
        as the worst parent. Useful for maintaining high-quality populations.

        Args:
            operation: The genetic operation (unused in this strategy)
            parent_probs: Parent probability values
            initial_prior: Default prior (unused in this strategy)

        Returns:
            Minimum of parent probabilities
        """
        return float(np.min(parent_probs))
