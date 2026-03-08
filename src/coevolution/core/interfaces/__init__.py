# coevolution/core/interfaces/__init__.py
"""
Core interfaces for the coevolution framework.

Organized into focused modules:
- types:       Type aliases, enums, and constants
- data:        Domain data structures
- config:      Configuration dataclasses
- base:        Abstract base classes for individuals and populations
- context:     Context objects for passing state
- operators:   IOperator protocol
- initializer: IPopulationInitializer
- probability: IProbabilityAssigner
- selection:   Selection strategy protocols
- language:    Protocol for language-specific operations
- sandbox:     Protocol for sandbox execution
- profiles:    Profile classes bundling configs and strategies
- systems:     System-level protocols
"""

# 4. Base classes (depends on types, data)
from .base import BaseIndividual, BasePopulation

# 6b. Breeder interface (depends on base, context)
from .breeder import IBreeder

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

# 8. Initializer (depends on base, data)
from .initializer import IPopulationInitializer

# 10. Language and Sandbox Adapters (depends on data)
from .language import (
    ILanguage,
    LanguageError,
    LanguageParsingError,
    LanguageTransformationError,
)

# 6. Operators (depends on base, context)
from .operators import IOperator

# 7. Probability (depends on types)
from .probability import IProbabilityAssigner

# 11. Profiles (depends on config, forward refs to selection, systems)
from .profiles import CodeProfile, OrchestratorConfig, PublicTestProfile, TestProfile
from .sandbox import ISandbox

# 9. Selection (depends on base, config, context)
from .selection import IEliteSelectionStrategy, IParentSelectionStrategy

# 12. Systems (depends on config, data, forward refs to population)
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
    "IOperator",
    # Breeder
    "IBreeder",
    # Probability
    "IProbabilityAssigner",
    # Initializer
    "IPopulationInitializer",
    # Selection
    "IEliteSelectionStrategy",
    "IParentSelectionStrategy",
    # Language
    "ILanguage",
    "LanguageError",
    "LanguageParsingError",
    "LanguageTransformationError",
    # Sandbox
    "ISandbox",
    # Profiles
    "CodeProfile",
    "OrchestratorConfig",
    "PublicTestProfile",
    "TestProfile",
    # Systems
    "IBeliefUpdater",
    "IExecutionSystem",
    "IInteractionLedger",
    "LedgerFactory",
]
