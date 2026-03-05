"""
Probability assignment strategies for offspring in coevolutionary algorithms.

This module implements the IProbabilityAssigner interface, providing various
strategies for assigning correctness probabilities to newly created offspring
based on their parents' probabilities and the genetic operation used.

Available Strategies:
    - mean: Inherits average of parent probabilities
    - max: Inherits maximum of parent probabilities (optimistic)
    - min: Inherits minimum of parent probabilities (pessimistic)
"""

from enum import Enum
from typing import Union

import numpy as np
from loguru import logger

from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    IProbabilityAssigner,
    Operation,
    ParentProbabilities,
)


class AssignmentStrategy(str, Enum):
    """Enumeration of available probability assignment strategies."""

    MEAN = "mean"  # Average of parent probabilities
    MAX = "max"    # Maximum of parent probabilities (optimistic)
    MIN = "min"    # Minimum of parent probabilities (pessimistic)
    INIT = "init"  # Always return initial_prior (rarely needed directly)


class ProbabilityAssigner(IProbabilityAssigner):
    """
    Configurable probability assigner supporting multiple inheritance strategies.

    initial_prior is owned by the assigner: it is set once at construction and
    used internally whenever the OPERATION_INITIAL case or the INIT strategy
    is encountered. Callers do NOT pass initial_prior to assign_probability().

    Args:
        strategy: Assignment strategy to use ('mean', 'max', 'min', 'init').
        initial_prior: Default prior for Gen-0 individuals (owned here).
    """

    def __init__(
        self,
        strategy: Union[AssignmentStrategy, str] = AssignmentStrategy.MIN,
        initial_prior: float = 0.2,
    ) -> None:
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

        self.initial_prior = initial_prior

        logger.debug(
            f"Initialized ProbabilityAssigner "
            f"strategy={self.strategy.value}, initial_prior={initial_prior:.4f}"
        )

        self._strategy_methods = {
            AssignmentStrategy.MEAN: self._assign_mean,
            AssignmentStrategy.MAX: self._assign_max,
            AssignmentStrategy.MIN: self._assign_min,
            AssignmentStrategy.INIT: self._assign_init,
        }

    def assign_probability(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
    ) -> float:
        """
        Calculate the probability for a new offspring.

        For OPERATION_INITIAL returns self.initial_prior.
        For genetic ops applies the configured strategy, clamped to >= initial_prior.

        Args:
            operation: The operation that created the offspring.
            parent_probs: List of parent probability values (may be empty for initial).

        Returns:
            Assigned probability for the offspring.

        Raises:
            ValueError: If parent_probs is empty for non-initial operations.
        """
        if operation == OPERATION_INITIAL:
            logger.trace(f"Assigning initial prior: {self.initial_prior:.4f}")
            return self.initial_prior

        if not parent_probs:
            msg = f"Cannot assign probability for '{operation}': no parent probabilities provided"
            logger.error(msg)
            raise ValueError(msg)

        strategy_func = self._strategy_methods[self.strategy]
        assigned_prob = strategy_func(operation, parent_probs)

        if assigned_prob < self.initial_prior:
            logger.debug(
                f"Assigned {assigned_prob:.4f} < initial_prior {self.initial_prior:.4f}; "
                "clamping to initial_prior."
            )
            return self.initial_prior

        logger.trace(
            f"Assigned {assigned_prob:.4f} for '{operation}' "
            f"(strategy={self.strategy.value}, parents={parent_probs})"
        )
        return assigned_prob

    # --- Strategy Implementations ---

    def _assign_init(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
    ) -> float:
        """Always return the initial prior, regardless of parents."""
        return self.initial_prior

    def _assign_mean(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
    ) -> float:
        """Balanced: inherits the average of parent probabilities."""
        return float(np.mean(parent_probs))

    def _assign_max(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
    ) -> float:
        """Optimistic: inherits the maximum of parent probabilities."""
        return float(np.max(parent_probs))

    def _assign_min(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
    ) -> float:
        """Conservative: inherits the minimum of parent probabilities."""
        return float(np.min(parent_probs))
