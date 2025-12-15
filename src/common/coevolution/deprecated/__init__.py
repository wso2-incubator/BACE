"""
Deprecated implementations - DO NOT USE

This package contains deprecated implementations that have been replaced by
the core/ module. These files are kept temporarily for backwards compatibility
during migration.

IMPORTANT:
- These implementations are NOT maintained
- They will be removed in a future version
- All new code should use core/ module instead

Migration Guide:
- Use core.orchestrator.Orchestrator instead of orchestrator.CoevolutionOrchestrator
- Use core.population.CodePopulation instead of population.CodePopulation
- Use core.population.TestPopulation instead of population.TestPopulation
- Use core.breeding_strategy.BreedingStrategy instead of reproduction.ReproductionStrategy

For more details, see README.md in this directory.
"""

import warnings

# Issue deprecation warning when this package is imported
warnings.warn(
    "The common.coevolution.deprecated package contains DEPRECATED implementations. "
    "Use common.coevolution.core instead. "
    "These deprecated modules will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["orchestrator", "population", "reproduction", "selection_strategy"]
