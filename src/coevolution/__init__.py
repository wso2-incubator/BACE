"""
Coevolution module for Bayesian code-test coevolution.

This module provides tools for coevolutionary algorithms with Bayesian updates,
specifically designed for code-test coevolution where both populations evolve
simultaneously with mutual evaluation and belief updates.

Phase 2 Architecture - Layered Implementation:

Core Components:
    - core/: Domain kernel (Orchestrator, Individual, Population, Interfaces)
    - factories/: The Construction Layer (Builders, Factory functions)

Implementation Layers:
    - services/: Engine mechanisms (Bayesian, Execution, Ledger)
    - strategies/: Pluggable policies (Operators, Breeding, Selection, Probability)
    - adapters/: External data connectors (LiveCodeBench dataset)
    - utils/: Cross-cutting helpers (Logging, Prompts)

Usage Patterns:

1. Direct layer access (recommended):
    from coevolution.factories import OrchestratorBuilder, ScheduleBuilder
    from coevolution.services import execution, bayesian, ledger
    from coevolution.strategies import operators, breeding, selection
    from coevolution.adapters import lcb
    from coevolution import utils

2. Hierarchical imports:
    from coevolution.factories import OrchestratorBuilder
    from coevolution.services.execution import ExecutionSystem
    from coevolution.services.bayesian import BayesianSystem
    from coevolution.strategies.breeding.code_breeding import CodeBreedingStrategy
    from coevolution.adapters.lcb import load_code_generation_dataset

Note: Phase 3 will refactor the core kernel for improved domain clarity.
"""

# from . import adapters, core, factories, services, strategies, utils

__all__ = [
    "core",
    "factories",
    "services",
    "strategies",
    "adapters",
    "utils",
]
