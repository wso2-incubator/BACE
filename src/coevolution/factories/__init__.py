"""
The Construction Layer.

This package contains all logic for instantiating and wiring complex objects.
It includes Builders (for step-by-step construction) and Factory functions (for preset configurations).
"""

from .orchestrator import OrchestratorBuilder, build_orchestrator_from_config
from .profiles import (
    create_agent_coder_code_profile,
    create_default_code_profile,
    create_differential_test_profile,
    create_public_test_profile,
    create_unittest_test_profile,
)
from .schedule import ScheduleBuilder

__all__ = [
    # Builders
    "OrchestratorBuilder",
    "ScheduleBuilder",
    # Factory functions
    "create_default_code_profile",
    "create_unittest_test_profile",
    "create_differential_test_profile",
    "create_public_test_profile",
    "build_orchestrator_from_config",
    "create_agent_coder_code_profile",
]
