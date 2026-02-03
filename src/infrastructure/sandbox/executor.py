"""Test executor for orchestrating code execution and analysis."""

from typing import List, Optional

from loguru import logger

from .analyzer import PytestXmlAnalyzer
from .core import SafeCodeSandbox
from .types import BasicExecutionResult, TestResult


class TestExecutor:
    """
    High-level interface for executing a single test function in a sandbox.

    This class orchestrates safe code execution and test result analysis,
    providing a clean separation between execution and analysis concerns.
    """

    __test__ = False

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 100,
        max_output_size: int = 10000,
        allowed_imports: Optional[List[str]] = None,
        python_executable: Optional[str] = None,
        test_method_timeout: Optional[int] = None,
    ):
        """
        Initialize test executor with sandbox configuration.

        Args:
            timeout: Maximum execution time in seconds for the entire script
            max_memory_mb: Maximum memory usage in MB
            max_output_size: Maximum output size in characters
            allowed_imports: List of allowed import modules
            python_executable: Path to Python executable
            test_method_timeout: Maximum execution time in seconds for individual test methods (None = no limit)
        """
        self.sandbox = SafeCodeSandbox(
            timeout=timeout,
            max_memory_mb=max_memory_mb,
            max_output_size=max_output_size,
            allowed_imports=allowed_imports,
            python_executable=python_executable,
            test_method_timeout=test_method_timeout,
        )
        self.analyzer = PytestXmlAnalyzer()
        logger.debug(
            f"Initialized TestExecutor(timeout={timeout}, max_memory_mb={max_memory_mb}, max_output_size={max_output_size}, test_method_timeout={test_method_timeout})"
        )

    def execute_test_script(self, test_script: str) -> TestResult:
        """
        Execute a test script and return a single test result.

        Args:
            test_script: The test script to execute

        Returns:
            TestResult with detailed analysis
        """
        logger.debug(f"TestExecutor: executing test script (len={len(test_script)})")

        # Delegate to sandbox
        return self.sandbox.execute_test_script(test_script)

    def execute_code(self, code: str) -> BasicExecutionResult:
        """
        Execute arbitrary code safely (delegates to sandbox).

        Args:
            code: Python code to execute

        Returns:
            BasicExecutionResult
        """
        return self.sandbox.execute_code(code)
