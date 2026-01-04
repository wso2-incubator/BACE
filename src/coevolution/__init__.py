"""
Coevolution module for Bayesian code-test coevolution.

This module provides tools for coevolutionary algorithms with Bayesian updates,
specifically designed for code-test coevolution where both populations evolve
simultaneously with mutual evaluation and belief updates.

Main components:
- config: Configuration classes for coevolution experiments
- bayesian: Bayesian belief update and evaluation functions
- operators: LLM-based genetic operators (mutation, crossover, edit)
- population: Population management (BasePopulation, CodePopulation, TestPopulation)
- selection: Selection strategies for evolutionary algorithms
- evaluation: Code execution and observation matrix generation
- feedback: Natural language feedback generation for LLM editing
- orchestrator: Main orchestrator for running the coevolution algorithm

Usage Patterns:

1. Direct submodule access (recommended):
    import coevolution as coevo
    config = coevo.core.interfaces.EvolutionConfig.simple(generations=50)
    orchestrator = coevo.core.orchestrator.Orchestrator(config, ...)
    selector = coevo.selection.SelectionStrategy(method="binary_tournament")
    code_pop = coevo.population.CodePopulation(...)

2. Hierarchical imports:
    from coevolution.config import CoevolutionConfig
    from coevolution.bayesian import initialize_prior_beliefs, update_population_beliefs
    from coevolution.operators import CodeOperator, TestOperator
    from coevolution.selection import SelectionStrategy
    from coevolution.population import BasePopulation, CodePopulation, TestPopulation
    from coevolution.orchestrator import CoevolutionOrchestrator

Note: Uses CodeGenerationProblem from lcb_runner for problem representation.
"""

from . import core

__all__ = [
    "core",
]
