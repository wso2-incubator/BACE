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
    import common.coevolution as coevo
    config = coevo.config.CoevolutionConfig(num_generations=50)
    orchestrator = coevo.orchestrator.CoevolutionOrchestrator(config, problem, llm, sandbox)
    selector = coevo.selection.SelectionStrategy(method="binary_tournament")
    code_pop = coevo.population.CodePopulation(...)

2. Hierarchical imports:
    from common.coevolution.config import CoevolutionConfig
    from common.coevolution.bayesian import initialize_prior_beliefs, update_population_beliefs
    from common.coevolution.operators import CodeOperator, TestOperator
    from common.coevolution.selection import SelectionStrategy
    from common.coevolution.population import BasePopulation, CodePopulation, TestPopulation
    from common.coevolution.orchestrator import CoevolutionOrchestrator

Note: Uses CodeGenerationProblem from lcb_runner for problem representation.
"""

# Note: Most files have been moved to deprecated/
# They will be replaced by new implementations based on core/ interfaces
from . import core, deprecated
from .bayesian_system import BayesianSystem
from .pareto_system import ParetoSystem
from .selection_strategy import SelectionMethod, SelectionStrategy

__all__ = [
    "core",
    "deprecated",
    "BayesianSystem",
    "ParetoSystem",
    "SelectionMethod",
    "SelectionStrategy",
]
