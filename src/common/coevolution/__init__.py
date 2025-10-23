"""
Coevolution module for Bayesian code-test coevolution.

This module provides tools for coevolutionary algorithms with Bayesian updates,
specifically designed for code-test coevolution where both populations evolve
simultaneously with mutual evaluation and belief updates.

Note: Uses CodeGenerationProblem from lcb_runner for problem representation.
"""

from .bayesian import (
    initialize_prior_beliefs,
    run_evaluation,
    update_population_beliefs,
)
from .config import CoevolutionConfig
from .operators import BaseLLMOperator, CodeOperator, TestOperator
from .population import Population
from .selection import SelectionStrategy

__all__ = [
    "CoevolutionConfig",
    "initialize_prior_beliefs",
    "run_evaluation",
    "update_population_beliefs",
    "SelectionStrategy",
    "Population",
    "BaseLLMOperator",
    "CodeOperator",
    "TestOperator",
]
