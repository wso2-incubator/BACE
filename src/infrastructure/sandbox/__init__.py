"""
Sandbox Module

Safe code execution environment for testing generated code with comprehensive analysis.

Structure:
- types: Data classes for execution results
- exceptions: Custom exceptions for sandbox operations
- core: SafeCodeSandbox implementation
- analyzer: PytestXmlAnalyzer for test result parsing
- executor: TestExecutor for high-level test orchestration
- utils: Utility functions and factory methods

usage:
    from infrastructure.sandbox import SubprocessSandbox, create_sandbox
    from infrastructure.sandbox import create_safe_test_environment

    # High-level API (recommended)
    sandbox = create_safe_test_environment()
    test_result = sandbox.execute_test_script(test_code, runtime, analyzer)

    # Standard command execution
    basic_result = sandbox.execute_command(cmd)
"""

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.sandbox import ISandbox

from .adapters.generic import SubprocessSandbox
from .exceptions import CodeExecutionError, CodeExecutionTimeoutError
from .types import BasicExecutionResult, SandboxConfig
from .utils import (
    check_test_execution_status,
    create_safe_test_environment,
)

# Alias for backward compatibility
SafeCodeSandbox = SubprocessSandbox


def create_sandbox(config: SandboxConfig) -> ISandbox:
    """
    Factory function to create a generic subprocess sandbox adapter.
    """
    return SubprocessSandbox(config)


__all__ = [
    # Core classes
    "SubprocessSandbox",
    "SafeCodeSandbox",
    "ISandbox",
    # Data types
    "EvaluationResult",
    "BasicExecutionResult",
    "SandboxConfig",
    # Exceptions
    "CodeExecutionError",
    "CodeExecutionTimeoutError",
    # Factory
    "create_sandbox",
    # Utility functions
    "create_safe_test_environment",
    "check_test_execution_status",
]
