"""Utility functions for sandbox operations."""

from typing import Optional

from .core import SafeCodeSandbox
from .executor import TestExecutor
from .types import TestExecutionResult


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


def check_test_execution_status(result: TestExecutionResult) -> str:
    """
    Helper function to get a human-readable status of test execution.

    Args:
        result: TestExecutionResult from execute_test_script

    Returns:
        String describing the execution status
    """
    if result.script_error:
        return f"SCRIPT ERROR: {result.summary}"
    elif result.has_failures:
        return f"TESTS FAILED: {result.tests_passed} passed, {result.tests_failed} failed, {result.tests_errors} errors"
    elif result.all_tests_passed:
        return f"ALL TESTS PASSED: {result.tests_passed} tests successful"
    elif result.total_tests == 0:
        return "NO TESTS: No test cases found or executed"
    else:
        return f"UNKNOWN STATUS: {result.summary}"
