"""Utility functions for sandbox operations."""

from typing import Optional

from coevolution.core.interfaces.data import EvaluationResult

from .core import SafeCodeSandbox
from .executor import TestExecutor


def create_safe_test_environment(
    test_method_timeout: Optional[int] = 30,
    script_timeout: int = 300,  # 300 seconds (5 mins) max for entire script
) -> SafeCodeSandbox:
    """
    Create a default safe test environment.

    Args:
        test_method_timeout: Maximum execution time in seconds for individual test methods (default: 30)
        script_timeout: Maximum execution time in seconds for entire script (default: 300)

    Returns:
        Configured SafeCodeSandbox instance
    """
    return SafeCodeSandbox(
        timeout=script_timeout,
        max_memory_mb=100,  # 100MB max memory
        max_output_size=1_000_000,  # 1MB max output
        test_method_timeout=test_method_timeout,  # 30 seconds max per test method
        allowed_imports=[
            "math",
            "random",
            "itertools",
            "collections",
            "functools",
            "operator",
            "copy",
            "json",
            "re",
            "string",
            "unittest",
            "pytest",
            "pytest_timeout",  # Add pytest-timeout plugin
            "typing",
            "dataclasses",
            "enum",
            "heapq",
            "bisect",
        ],
    )


def create_test_executor(test_method_timeout: Optional[int] = 30) -> TestExecutor:
    """
    Create a default test executor with safe configuration.

    Args:
        test_method_timeout: Maximum execution time in seconds for individual test methods (default: 30)

    Returns:
        Configured TestExecutor instance
    """
    return TestExecutor(
        timeout=180,  # 180 seconds max for entire script
        max_memory_mb=100,  # 100MB max memory
        max_output_size=1_000_000,  # 1MB max output
        test_method_timeout=test_method_timeout,  # 30 seconds max per test method
        allowed_imports=[
            "math",
            "random",
            "itertools",
            "collections",
            "functools",
            "operator",
            "copy",
            "json",
            "re",
            "string",
            "unittest",
            "pytest",
            "pytest_timeout",  # Add pytest-timeout plugin
            "typing",
            "dataclasses",
            "enum",
            "heapq",
            "bisect",
        ],
    )


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
        return f"UNKNOWN STATUS: {result.status}"
