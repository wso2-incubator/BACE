"""
The Construction Layer.

This package contains all logic for instantiating and wiring complex objects.
It includes Builders (for step-by-step construction) and Factory functions (for preset configurations).

Adding a new population:
1. Create populations/<name>/ with operators/ and profile.py
2. Import the factory function below.
"""

from ..populations import registry
from .orchestrator import OrchestratorBuilder, build_orchestrator_from_config
from .population_discovery import PopulationDiscoveryService
from .schedule import ScheduleBuilder

__all__ = [
    # Builders and Services
    "OrchestratorBuilder",
    "ScheduleBuilder",
    "PopulationDiscoveryService",
    # Registry
    "registry",
    # Helper functions
    "build_orchestrator_from_config",
]
