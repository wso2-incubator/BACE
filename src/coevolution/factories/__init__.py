"""
The Construction Layer.

This package contains all logic for instantiating and wiring complex objects.
It includes Builders (for step-by-step construction) and Factory functions (for preset configurations).

Adding a new population:
1. Create populations/<name>/ with operators/ and profile.py
2. Import the factory function below.
"""

from .orchestrator import OrchestratorBuilder, build_orchestrator_from_config
from .schedule import ScheduleBuilder
from ..populations.code.profile import create_default_code_profile
from ..populations.agent_coder.profile import create_agent_coder_code_profile
from ..populations.unittest.profile import (
    create_unittest_test_profile,
    create_public_test_profile,
)
from ..populations.differential.profile import create_differential_test_profile

__all__ = [
    # Builders
    "OrchestratorBuilder",
    "ScheduleBuilder",
    # Factory functions
    "create_default_code_profile",
    "create_agent_coder_code_profile",
    "create_unittest_test_profile",
    "create_public_test_profile",
    "create_differential_test_profile",
    "build_orchestrator_from_config",
]
