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

Usage:
    from infrastructure.sandbox import SafeCodeSandbox, TestExecutor
    from infrastructure.sandbox import create_safe_test_environment, create_test_executor

    # High-level API (recommended)
    executor = create_test_executor()
    result = executor.execute_test_script(test_code)

    # Low-level API (for more control)
    sandbox = create_safe_test_environment()
    basic_result = sandbox.execute_code(code)
    test_result = sandbox.execute_test_script(test_code)
"""

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.sandbox import ISandboxAdapter

from .adapters.python import PythonSandbox
from .adapters.ballerina import BallerinaSandbox
from .analyzer import PytestXmlAnalyzer
from .exceptions import CodeExecutionError, CodeExecutionTimeoutError
from .executor import TestExecutor
from .types import BasicExecutionResult, SandboxConfig
from .utils import (
    check_test_execution_status,
    create_safe_test_environment,
    create_test_executor,
)

# Alias for backward compatibility
SafeCodeSandbox = PythonSandbox


def create_sandbox(config: SandboxConfig) -> ISandboxAdapter:
    """
    Factory function to create the appropriate sandbox adapter based on config.
    """
    if config.language == "python":
        return PythonSandbox(config)
    elif config.language == "ballerina":
        return BallerinaSandbox(config)
    else:
        raise ValueError(f"Unsupported sandbox language: {config.language}")


__all__ = [
    # Core classes
    "PythonSandbox",
    "BallerinaSandbox",
    "SafeCodeSandbox",
    "ISandboxAdapter",
    "TestExecutor",
    "PytestXmlAnalyzer",
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
    "create_test_executor",
    "check_test_execution_status",
]
