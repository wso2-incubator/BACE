# coevolution/core/interfaces/__init__.py
"""
Core interfaces for the coevolution framework.

This module defines the protocol-based architecture for the coevolution system.
It provides both fine-grained and grouped interfaces to support different
implementation strategies.

The interfaces are organized into focused modules:
- types: Type aliases, enums, and constants
- data: Domain data structures and DTOs
- config: Configuration dataclasses
- base: Abstract base classes for individuals and populations
- context: Context objects for passing state
- language: Protocol for language-specific operations
- sandbox: Protocol for language-specific sandbox execution
- profiles: Profile classes bundling configurations and strategies
- operators: Operator protocols and DTOs
- breeding: Breeding strategy protocols
- selection: Selection strategy protocols
- systems: System-level protocols

All symbols are re-exported at the package level for backward compatibility.
"""

# Import in dependency order to avoid circular imports

# 4. Base classes (depends on types, data)
from .base import BaseIndividual, BasePopulation

# 7. Breeding (depends on base, context, data, types)
from .breeding import IBreedingStrategy, IIndividualFactory, IProbabilityAssigner

# 3. Configuration (depends on types)
from .config import (
    BayesianConfig,
    EvolutionConfig,
    EvolutionPhase,
    EvolutionSchedule,
    OperatorRatesConfig,
    PopulationConfig,
)

# 5. Context (depends on data, forward refs to population)
from .context import CoevolutionContext

# 2. Data structures (depends on types)
from .data import (
    BasicExecutionResult,
    EvaluationResult,
    ExecutionResults,
    InteractionData,
    LogEntry,
    Problem,
    SandboxConfig,
    Test,
)

# 6. Language and Sandbox Adapters (depends on data)
from .language import ILanguage, LanguageParsingError, LanguageTransformationError

# 7. Operators (depends on data, types)
from .operators import (
    BaseOperatorInput,
    InitialInput,
    IOperator,
    OperatorOutput,
    OperatorResult,
)

# 10. Profiles (depends on config, forward refs to breeding, selection, systems)
from .profiles import CodeProfile, OrchestratorConfig, PublicTestProfile, TestProfile
from .sandbox import ISandboxAdapter

# 8. Selection (depends on base, config, context)
from .selection import IEliteSelectionStrategy, IParentSelectionStrategy

# 9. Systems (depends on config, data, forward refs to population)
from .systems import IBeliefUpdater, IExecutionSystem, IInteractionLedger, LedgerFactory

# 1. Types and constants (no dependencies)
from .types import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    OPERATION_REPRODUCTION,
    InteractionKey,
    LifecycleEvent,
    Operation,
    ParentDict,
    ParentProbabilities,
)

__all__ = [
    # Types
    "ExecutionResults",
    "InteractionKey",
    "LifecycleEvent",
    "Operation",
    "OPERATION_CROSSOVER",
    "OPERATION_EDIT",
    "OPERATION_INITIAL",
    "OPERATION_MUTATION",
    "OPERATION_REPRODUCTION",
    "ParentDict",
    "ParentProbabilities",
    # Data
    "BasicExecutionResult",
    "EvaluationResult",
    "ExecutionResults",
    "InteractionData",
    "LogEntry",
    "Problem",
    "SandboxConfig",
    "Test",
    # Config
    "BayesianConfig",
    "EvolutionConfig",
    "EvolutionPhase",
    "EvolutionSchedule",
    "OperatorRatesConfig",
    "PopulationConfig",
    # Base
    "BaseIndividual",
    "BasePopulation",
    # Context
    "CoevolutionContext",
    # Operators
    "BaseOperatorInput",
    "IOperator",
    "InitialInput",
    "OperatorOutput",
    "OperatorResult",
    # Breeding
    "IBreedingStrategy",
    "IIndividualFactory",
    "IProbabilityAssigner",
    # Selection
    "IEliteSelectionStrategy",
    "IParentSelectionStrategy",
    # Systems
    "IBeliefUpdater",
    "IExecutionSystem",
    "IInteractionLedger",
    "LedgerFactory",
    # Profiles
    "CodeProfile",
    "OrchestratorConfig",
    "PublicTestProfile",
    "TestProfile",
    # Language Adapter
    "ILanguage",
    "LanguageParsingError",
    "LanguageTransformationError",
    # Sandbox Adapter
    "ISandboxAdapter",
]
