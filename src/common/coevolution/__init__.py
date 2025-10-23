"""
Coevolution module for Bayesian code-test coevolution.

This module provides tools for coevolutionary algorithms with Bayesian updates,
specifically designed for code-test coevolution where both populations evolve
simultaneously with mutual evaluation and belief updates.

Main components:
- config: Configuration classes for coevolution experiments
- bayesian: Bayesian belief update and evaluation functions
- operators: LLM-based genetic operators (mutation, crossover)
- population: Population management for evolutionary algorithms (BasePopulation, CodePopulation, TestPopulation)
- selection: Selection strategies for evolutionary algorithms
- evaluation: Fitness evaluation and testing functions

Usage (Hierarchical Imports):
    from common.coevolution.config import CoevolutionConfig
    from common.coevolution.bayesian import initialize_prior_beliefs, run_evaluation
    from common.coevolution.operators import CodeOperator, TestOperator
    from common.coevolution.selection import SelectionStrategy
    from common.coevolution.population import BasePopulation, CodePopulation, TestPopulation

Note: Uses CodeGenerationProblem from lcb_runner for problem representation.
"""

# No re-exports - use hierarchical imports from submodules

__all__: list[str] = []
