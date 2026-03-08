"""Utility functions for sandbox operations."""

from typing import Optional

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.sandbox import ISandbox

from .adapters.python import PythonSandbox
from .executor import TestExecutor
from .types import SandboxConfig


def create_safe_test_environment(
    config: Optional[SandboxConfig] = None,
) -> ISandbox:
    """
    Create a default safe test environment (Python by default).

    Args:
        config: Sandbox configuration parameters

    Returns:
        Configured ISandbox instance
    """
    config = config or SandboxConfig(
        timeout=300,
        max_memory_mb=256,
        max_output_size=1_000_000,
        test_method_timeout=30,
    )
    return PythonSandbox(config)


def create_test_executor(
    sandbox_adapter: Optional[ISandbox] = None,
    config: Optional[SandboxConfig] = None,
) -> TestExecutor:
    """
    Create a test executor with a specific sandbox or default configuration.

    Args:
        sandbox_adapter: Optional specialized sandbox adapter
        config: Optional configuration for default sandbox

    Returns:
        Configured TestExecutor instance
    """
    if not sandbox_adapter:
        config = config or SandboxConfig(
            timeout=180,
            max_memory_mb=256,
            max_output_size=1_000_000,
            test_method_timeout=30,
        )
        sandbox_adapter = PythonSandbox(config)

    return TestExecutor(sandbox_adapter=sandbox_adapter)


def check_test_execution_status(result: EvaluationResult) -> str:
    """
    Helper function to get a human-readable status of test execution.

    Args:
        result: EvaluationResult from execute_test_script

    Returns:
        String describing the execution status
    """
    if result.status == "passed":
        return "TEST PASSED"
    elif result.status == "failed":
        return f"TEST FAILED. Error: {(result.error_log or '')[:100]}..."
    elif result.status == "error":
        return f"TEST ERROR/SCRIPT ERROR. Error: {(result.error_log or '')[:100]}..."
    else:
        return f"UNKNOWN STATUS: {result.status}"
