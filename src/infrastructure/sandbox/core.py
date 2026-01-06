"""Core sandbox implementation for safe code execution."""

import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from .analyzer import PytestXmlAnalyzer
from .types import BasicExecutionResult, TestExecutionResult, TestResult


class SafeCodeSandbox:
    """
    A safe sandbox environment for executing Python code with restrictions.
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
        Initialize the sandbox with safety parameters.

        Args:
            timeout: Maximum execution time in seconds for the entire script
            max_memory_mb: Maximum memory usage in MB (not enforced on all systems)
            max_output_size: Maximum output size in characters
            allowed_imports: List of allowed import modules
            python_executable: Path to Python executable (defaults to sys.executable)
            test_method_timeout: Maximum execution time in seconds for individual test methods (None = no limit)
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size
        self.python_executable = python_executable or sys.executable
        self.test_method_timeout = test_method_timeout
        self.allowed_imports = allowed_imports or [
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
        ]

        # Dangerous modules/functions to block
        # self.blocked_patterns = [
        #     "import os",
        #     # "import sys", # TODO: Consider allowing sys with restrictions
        #     "import subprocess",
        #     "import shutil",
        #     "import socket",
        #     "import urllib",
        #     "import requests",
        #     "import http",
        #     "import ftplib",
        #     "import smtplib",
        #     "import telnetlib",
        #     "import tempfile",
        #     "import pickle",
        #     "import marshal",
        #     "import importlib",
        #     "import __import__",
        #     "exec(",
        #     # "eval(", # TODO: Consider allowing eval with restrictions, this was needed for generating test with actual lcb test cases
        #     "compile(",
        #     "open(",
        #     "file(",
        #     # "input(",
        #     "raw_input(",
        #     "__builtins__",
        #     "__globals__",
        #     "__locals__",
        #     # "globals()",
        #     # "locals()", # TODO: Consider allowing globals/locals with restrictions
        #     "vars()",
        #     "dir()",
        #     "hasattr(",
        #     "getattr(",
        #     "setattr(",
        #     "delattr(",
        #     "exit(",
        #     "quit(",
        #     "reload(",
        # ]

        # TODO: Re-evaluate blocked patterns, for now we disable blocking to allow more flexibility
        self.blocked_patterns = []

    def _check_code_safety(self, code: str) -> bool:
        """
        Check if code contains potentially dangerous patterns.

        Args:
            code: The code string to check

        Returns:
            True if code appears safe, False otherwise
        """
        code_lower = code.lower()

        for pattern in self.blocked_patterns:
            if pattern.lower() in code_lower:
                logger.debug(f"Code blocked by pattern '{pattern}'")
                return False

        logger.trace(
            f"Code passed safety checks (checked {len(self.blocked_patterns)} patterns)"
        )
        return True

    def _create_restricted_environment(self) -> Dict[str, Any]:
        """
        Create a restricted environment for code execution.

        Returns:
            Dictionary with restricted built-ins
        """
        # Create a minimal safe environment
        safe_builtins = {
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "sum": sum,
            "min": max,
            "max": max,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "set": set,
            "frozenset": frozenset,
            "type": type,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "print": print,
            "repr": repr,
            "ord": ord,
            "chr": chr,
            "bin": bin,
            "oct": oct,
            "hex": hex,
            "any": any,
            "all": all,
            "pow": pow,
            "divmod": divmod,
            "reversed": reversed,
            "slice": slice,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "IndexError": IndexError,
            "KeyError": KeyError,
            "AttributeError": AttributeError,
        }

        return {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
            "__doc__": None,
        }

    def execute_code(
        self, code: str, capture_output: bool = True
    ) -> BasicExecutionResult:
        """
        Safely execute Python code in a restricted environment.

        Args:
            code: The Python code to execute
            capture_output: Whether to capture stdout/stderr

        Returns:
            BasicExecutionResult containing:
            - success: bool indicating if execution succeeded
            - output: captured stdout
            - error: captured stderr or error message
            - execution_time: time taken in seconds
            - timeout: whether execution timed out
            - return_code: process return code
        """
        # Check code safety first
        if not self._check_code_safety(code):
            return BasicExecutionResult(
                success=False,
                output="",
                error="Code contains potentially dangerous patterns",
                execution_time=0,
                timeout=False,
                return_code=-1,
            )

        # Create temporary file for code execution
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        logger.trace(
            f"Executing code in sandbox: temp_file={temp_file_path} capture_output={capture_output} code_len={len(code)}",
        )

        try:
            # Execute code in subprocess with timeout
            start_time = time.time()

            result = subprocess.run(
                [self.python_executable, temp_file_path],
                capture_output=capture_output,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),  # Run in temp directory
            )

            logger.trace(f"Execution finished: returncode={result.returncode}")

            execution_time = time.time() - start_time

            # Limit output size
            if len(result.stdout) > self.max_output_size:
                logger.warning(
                    f"Truncating stdout from {len(result.stdout)} to {self.max_output_size} characters"
                )

            if len(result.stderr) > self.max_output_size:
                logger.warning(
                    f"Truncating stderr from {len(result.stderr)} to {self.max_output_size} characters"
                )

            stdout = result.stdout[: self.max_output_size] if result.stdout else ""
            stderr = result.stderr[: self.max_output_size] if result.stderr else ""

            logger.trace(
                f"Captured output sizes: stdout={len(stdout)} stderr={len(stderr)}"
            )

            return BasicExecutionResult(
                success=result.returncode == 0,
                output=stdout,
                error=stderr,
                execution_time=execution_time,
                timeout=False,
                return_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            logger.warning(f"Code execution timed out after {self.timeout} seconds")
            return BasicExecutionResult(
                success=False,
                output="",
                error=f"Code execution timed out after {self.timeout} seconds",
                execution_time=self.timeout,
                timeout=True,
                return_code=-1,
            )

        except Exception as e:
            logger.exception(f"Unhandled exception during code execution: {e}")
            return BasicExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=0,
                timeout=False,
                return_code=-1,
            )

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

    def _reorder_test_results_to_match_script(
        self, test_script: str, execution_result: TestExecutionResult
    ) -> TestExecutionResult:
        """
        Reorder test results to match the order of test methods in the script.

        This is critical for ensuring that test_results[i] corresponds to the i-th
        test method in the script, regardless of unittest's execution order.

        Args:
            test_script: The original test script
            execution_result: The result from test execution with potentially reordered tests

        Returns:
            TestExecutionResult with test_results reordered to match script order
        """
        # Import here to avoid circular dependency
        try:
            from infrastructure.code_preprocessing.analysis import analyze_test_methods
        except ImportError:
            logger.warning(
                "Could not import analyze_test_methods, skipping test result reordering"
            )
            return execution_result

        try:
            # Extract test method names in script order
            test_method_names_in_order = analyze_test_methods(test_script)

            if not test_method_names_in_order:
                logger.warning(
                    "No test methods found in script, returning results as-is"
                )
                return execution_result

            # Build lookup dict from execution results
            results_by_name = {
                result.name: result for result in execution_result.test_results
            }

            # Reorder to match script order
            ordered_test_results = []
            for expected_name in test_method_names_in_order:
                if expected_name in results_by_name:
                    ordered_test_results.append(results_by_name[expected_name])
                else:
                    # Test was in script but not in execution results
                    # This shouldn't happen normally, but handle gracefully
                    logger.warning(
                        f"Test '{expected_name}' found in script but not in execution results"
                    )
                    # Create a placeholder error result
                    ordered_test_results.append(
                        TestResult(
                            name=expected_name,
                            description=f"Test {expected_name}",
                            status="error",
                            details="Test was not executed (collection error or skipped)",
                        )
                    )

            # Check if there are any executed tests not in the script
            # (This would indicate a problem with our analysis)
            executed_names = set(results_by_name.keys())
            expected_names = set(test_method_names_in_order)
            unexpected_tests = executed_names - expected_names

            if unexpected_tests:
                logger.warning(
                    f"Found {len(unexpected_tests)} tests in execution results "
                    f"that were not found in script: {unexpected_tests}"
                )

            # Create new TestExecutionResult with reordered test_results
            return TestExecutionResult(
                script_error=execution_result.script_error,
                tests_passed=execution_result.tests_passed,
                tests_failed=execution_result.tests_failed,
                tests_errors=execution_result.tests_errors,
                test_results=ordered_test_results,
                summary=execution_result.summary,
            )

        except Exception as e:
            logger.error(
                f"Error reordering test results: {e}. Returning original order.",
                exc_info=True,
            )
            return execution_result

    def execute_test_script(self, test_script: str) -> TestExecutionResult:
        """
        Execute a test script and return detailed results with test categorization.

        This method is a convenience wrapper that combines safe code execution
        with pytest XML analysis. For more control over the process, use
        execute_code() and PytestXmlAnalyzer separately.

        IMPORTANT: The returned TestExecutionResult.test_results list is guaranteed
        to be in the same order as the test methods appear in the test_script,
        regardless of the order pytest executes them. This ensures consistent
        indexing when building observation matrices and generating feedback.

        Args:
            test_script: The test script to execute

        Returns:
            TestExecutionResult containing comprehensive execution results,
            with test_results ordered to match the script's test method order
        """
        logger.debug(f"SafeCodeSandbox: executing test script (len={len(test_script)})")

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
                self.python_executable,
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
            if self.test_method_timeout is not None:
                cmd.extend(["--timeout", str(self.test_method_timeout)])

            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
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

            # Create analyzer and analyze the test results
            analyzer = PytestXmlAnalyzer()
            execution_result = analyzer.analyze_pytest_xml(xml_content, basic_result)

            # Reorder test results to match the script order (CRITICAL FIX)
            execution_result = self._reorder_test_results_to_match_script(
                test_script, execution_result
            )

            logger.debug(
                f"SafeCodeSandbox: test script result summary: {execution_result.summary}"
            )
            return execution_result

        except subprocess.TimeoutExpired:
            logger.warning(f"Test execution timed out after {self.timeout} seconds")
            basic_result = BasicExecutionResult(
                success=False,
                output="",
                error=f"Test execution timed out after {self.timeout} seconds",
                execution_time=self.timeout,
                timeout=True,
                return_code=-1,
            )
            analyzer = PytestXmlAnalyzer()
            return analyzer.analyze_pytest_xml(None, basic_result)

        finally:
            # Clean up temporary files
            try:
                os.unlink(script_file_path)
                os.unlink(xml_file_path)
            except OSError:
                pass
