"""Test executor for orchestrating code execution and analysis."""

import os
import subprocess
import tempfile
import time
from typing import List, Optional

from loguru import logger

from .analyzer import PytestXmlAnalyzer
from .core import SafeCodeSandbox
from .types import BasicExecutionResult, TestExecutionResult


class TestExecutor:
    """
    High-level interface for executing test scripts with comprehensive analysis.

    This class orchestrates safe code execution and test result analysis,
    providing a clean separation between execution and analysis concerns
    while maintaining the convenience of the original execute_test_script API.
    """

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

    def execute_test_script(self, test_script: str) -> TestExecutionResult:
        """
        Execute a test script and return comprehensive results.

        IMPORTANT: The returned TestExecutionResult.test_results list is guaranteed
        to be in the same order as the test methods appear in the test_script,
        regardless of the order pytest executes them. This ensures consistent
        indexing when building observation matrices and generating feedback.

        Args:
            test_script: The test script to execute

        Returns:
            TestExecutionResult with detailed analysis and properly ordered test results
        """
        logger.debug(f"TestExecutor: executing test script (len={len(test_script)})")

        # Create temporary files for script and XML output
        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as script_file,
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".xml", delete=False
            ) as xml_file,
        ):
            script_file_path = script_file.name
            xml_file_path = xml_file.name
            script_file.write(test_script)

        try:
            # Execute pytest with JUnit XML output
            start_time = time.time()

            # Build pytest command
            cmd = [
                self.sandbox.python_executable,
                "-m",
                "pytest",
                script_file_path,
                "--junitxml",
                xml_file_path,
                "--color=no",  # Force disable ANSI color codes
                "-o",
                "console_output_style=classic",  # Use simple output style (no progress bars etc)
            ]

            # Add per-test timeout if configured
            if self.sandbox.test_method_timeout is not None:
                cmd.extend(["--timeout", str(self.sandbox.test_method_timeout)])

            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.sandbox.timeout,
                cwd=tempfile.gettempdir(),
            )

            execution_time = time.time() - start_time

            # Read XML content if file exists
            xml_content = None
            if os.path.exists(xml_file_path):
                try:
                    with open(xml_file_path, "r", encoding="utf-8") as f:
                        xml_content = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read XML file: {e}")

            # Create basic result for error analysis fallback
            basic_result = BasicExecutionResult(
                success=proc_result.returncode == 0,
                output=proc_result.stdout,
                error=proc_result.stderr,
                execution_time=execution_time,
                timeout=False,
                return_code=proc_result.returncode,
            )

            # Analyze the results
            execution_result = self.analyzer.analyze_pytest_xml(
                xml_content, basic_result
            )

            # Reorder test results to match script order (use sandbox's method)
            execution_result = self.sandbox._reorder_test_results_to_match_script(
                test_script, execution_result
            )

            logger.debug(
                f"TestExecutor: finished execution: {execution_result.summary}"
            )
            return execution_result

        except subprocess.TimeoutExpired:
            logger.warning(
                f"Test execution timed out after {self.sandbox.timeout} seconds"
            )
            basic_result = BasicExecutionResult(
                success=False,
                output="",
                error=f"Test execution timed out after {self.sandbox.timeout} seconds",
                execution_time=self.sandbox.timeout,
                timeout=True,
                return_code=-1,
            )
            return self.analyzer.analyze_pytest_xml(None, basic_result)

        finally:
            # Clean up temporary files
            try:
                os.unlink(script_file_path)
                os.unlink(xml_file_path)
            except OSError:
                pass

    def execute_code(self, code: str) -> BasicExecutionResult:
        """
        Execute arbitrary code safely (delegates to sandbox).

        Args:
            code: Python code to execute

        Returns:
            BasicExecutionResult
        """
        return self.sandbox.execute_code(code)
