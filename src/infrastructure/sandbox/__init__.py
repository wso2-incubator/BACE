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

from .analyzer import PytestXmlAnalyzer
from .core import SafeCodeSandbox
from .exceptions import CodeExecutionError, CodeExecutionTimeoutError
from .executor import TestExecutor
from .types import (
    BasicExecutionResult,
    SandboxConfig,
    TestAnalysis,
    TestExecutionResult,
    TestResult,
)
from .utils import (
    check_test_execution_status,
    create_safe_test_environment,
    create_test_executor,
)

__all__ = [
    # Core classes
    "SafeCodeSandbox",
    "TestExecutor",
    "PytestXmlAnalyzer",
    # Data types
    "TestResult",
    "BasicExecutionResult",
    "TestExecutionResult",
    "TestAnalysis",
    "SandboxConfig",
    # Exceptions
    "CodeExecutionError",
    "CodeExecutionTimeoutError",
    # Utility functions
    "create_safe_test_environment",
    "create_test_executor",
    "check_test_execution_status",
]
