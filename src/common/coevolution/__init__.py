"""
Coevolution module for Bayesian code-test coevolution.

This module provides tools for coevolutionary algorithms with Bayesian updates,
specifically designed for code-test coevolution where both populations evolve
simultaneously with mutual evaluation and belief updates.
"""

from .bayesian import (
    CoevolutionConfig,
    initialize_populations,
    run_evaluation,
    update_population_beliefs,
)
from .selection import SelectionStrategy

__all__ = [
    "CoevolutionConfig",
    "initialize_populations",
    "run_evaluation",
    "update_population_beliefs",
    "SelectionStrategy",
]
