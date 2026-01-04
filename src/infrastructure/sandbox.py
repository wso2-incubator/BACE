"""
Safe code execution sandbox for testing generated code.

This module provides a secure environment to execute Python code snippets
with safety restrictions to prevent harm to the local machine.
"""

import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from loguru import logger


class CodeExecutionTimeoutError(Exception):
    """Raised when code execution times out."""

    pass


class CodeExecutionError(Exception):
    """Raised when code execution fails."""

    pass


# Type definitions for better type safety


@dataclass
class TestResult:
    """Individual test result with detailed information."""

    name: str
    description: str  # Usually the docstring or test description
    status: Literal["passed", "failed", "error"]
    # Full error message/traceback for failed/error tests
    details: Optional[str] = None


@dataclass
class BasicExecutionResult:
    """Result of basic code execution."""

    success: bool
    output: str
    error: str
    execution_time: float
    timeout: bool
    return_code: int


@dataclass
class TestExecutionResult:
    """Simplified result of test script execution focusing on test analysis."""

    # Test-specific analysis
    script_error: bool
    tests_passed: int
    tests_failed: int
    tests_errors: int

    # Ordered list of test results (in execution order)
    test_results: List[TestResult]

    # Simple summary message
    summary: str

    @property
    def total_tests(self) -> int:
        """Total number of tests that ran."""
        return self.tests_passed + self.tests_failed + self.tests_errors

    @property
    def success_rate(self) -> float:
        """Percentage of tests that passed (0.0 to 1.0)."""
        if self.total_tests == 0:
            return 0.0
        return self.tests_passed / self.total_tests

    @property
    def has_failures(self) -> bool:
        """Whether any tests failed or had errors."""
        return self.tests_failed > 0 or self.tests_errors > 0

    @property
    def all_tests_passed(self) -> bool:
        """Whether all tests passed successfully."""
        return (
            self.total_tests > 0
            and self.tests_failed == 0
            and self.tests_errors == 0
            and not self.script_error
        )


@dataclass
class TestAnalysis:
    """Internal analysis of unittest output."""

    passed: int
    failed: int
    errors: int
    script_error: bool
    test_results: List[TestResult]


class PytestXmlAnalyzer:
    """
    Analyzes pytest JUnit XML output to extract detailed test results.

    This class parses the structured XML output from pytest --junitxml
    to provide robust and reliable test result analysis.
    """

    # Regex to identify ANSI escape sequences (both raw and hex-encoded)
    ANSI_ESCAPE_PATTERN = re.compile(
        r"(?:\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])|#x1B\[[0-?]*[ -/]*[@-~])"
    )

    # Regex to capture absolute paths to temporary python files (handles Unix and Windows)
    # Matches examples like: /var/.../tmppjtsol6j.py:425 or C:\Temp\tmppjtsol6j.py:425
    TEMP_PATH_PATTERN = re.compile(r"(?:(?:[A-Za-z]:\\)|/)[^\s:<>\"]+\.py(?::\d+)?")

    # Regex to capture module-like prefixes in instance/class representations
    # Example: "<tmppjtsol6j.TestMakeAEqualB testMethod=test_x>" -> "<TestMakeAEqualB testMethod=test_x>"
    MODULE_PREFIX_PATTERN = re.compile(r"<[A-Za-z0-9_]+\.([A-Za-z_][A-Za-z0-9_]*)")

    def __init__(self) -> None:
        """Initialize the pytest XML analyzer."""
        pass

    def _remove_ansi_codes(self, text: Optional[str]) -> Optional[str]:
        """Strip ANSI escape codes from string."""
        if not text:
            return text
        return self.ANSI_ESCAPE_PATTERN.sub("", text)

    def _sanitize_details(self, text: Optional[str]) -> Optional[str]:
        """
        Strip ANSI codes AND sanitize temporary file paths/module names.

        Replaces absolute temporary file paths with a generic `test_script.py` while
        preserving optional line numbers, and removes random module prefixes from
        class/instance representations.
        """
        if not text:
            return text

        # 1. Remove ANSI codes
        sanitized = self.ANSI_ESCAPE_PATTERN.sub("", text)

        # 2. Replace full temporary file paths with a generic name, preserving line numbers
        def _path_repl(m: re.Match[str]) -> str:
            match_text = m.group(0)
            # If there's a colon with a line number, preserve it
            colon_idx = match_text.rfind(":")
            if colon_idx != -1 and match_text[colon_idx + 1 :].isdigit():
                return f"test_script.py:{match_text[colon_idx + 1 :]}"
            return "test_script.py"

        sanitized = self.TEMP_PATH_PATTERN.sub(_path_repl, sanitized)

        # 3. Clean up the class/module instance representation
        sanitized = self.MODULE_PREFIX_PATTERN.sub(r"<\1", sanitized)

        return sanitized

    def analyze_pytest_xml(
        self, xml_content: Optional[str], basic_result: BasicExecutionResult
    ) -> TestExecutionResult:
        """
        Analyze pytest XML output and return detailed test information.

        Args:
            xml_content: The XML content from pytest --junitxml output, or None if XML wasn't generated
            basic_result: Basic execution result from code execution

        Returns:
            TestExecutionResult with detailed test analysis
        """
        logger.debug(
            f"Analyzing pytest XML output: xml_available={xml_content is not None}, success={basic_result.success}, return_code={basic_result.return_code}"
        )

        # If XML content is available, parse it
        if xml_content:
            try:
                return self._parse_xml_content(xml_content)
            except Exception as e:
                logger.warning(
                    f"Failed to parse XML content: {e}. Falling back to error analysis."
                )
                # Fall through to error analysis

        # No XML or parsing failed - analyze based on basic result
        return self._analyze_execution_error(basic_result)

    def _parse_xml_content(self, xml_content: str) -> TestExecutionResult:
        """Parse JUnit XML content and extract test results."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise

        total_tests = 0
        total_failures = 0
        total_errors = 0
        test_results = []
        script_error = False

        # Parse each testsuite
        for testsuite in root.findall(".//testsuite"):
            suite_tests = int(testsuite.get("tests", 0))
            suite_failures = int(testsuite.get("failures", 0))
            suite_errors = int(testsuite.get("errors", 0))

            total_tests += suite_tests
            total_failures += suite_failures
            total_errors += suite_errors

            # Parse individual test cases
            for testcase in testsuite.findall("testcase"):
                test_name = testcase.get("name", "unknown")
                classname = testcase.get("classname", "")

                # Determine status and details
                status: Literal["passed", "failed", "error"] = "passed"
                details = None

                # Check for failure
                failure = testcase.find("failure")
                if failure is not None:
                    status = "failed"
                    raw_details = failure.text or failure.get("message", "")
                    details = self._sanitize_details(raw_details)

                # Check for error
                error = testcase.find("error")
                if error is not None:
                    status = "error"
                    raw_details = error.text or error.get("message", "")
                    details = self._sanitize_details(raw_details)
                    # Check if this is a syntax error (script-level error)
                    if details and (
                        "SyntaxError" in details
                        or "was never closed" in details
                        or "invalid syntax" in details
                    ):
                        script_error = True

                # Create description from classname and test name
                description = f"{classname}.{test_name}" if classname else test_name

                test_result = TestResult(
                    name=test_name,
                    description=description,
                    status=status,
                    details=details,
                )
                test_results.append(test_result)

        # Create summary
        if script_error:
            summary = "Script execution failed: syntax error"
        elif total_failures > 0 or total_errors > 0:
            summary = f"Tests completed: {total_tests - total_failures - total_errors} passed, {total_failures} failed, {total_errors} errors"
        elif total_tests > 0:
            summary = f"All tests passed: {total_tests} tests"
        else:
            summary = "No tests were found or executed"

        return TestExecutionResult(
            script_error=script_error,
            tests_passed=total_tests - total_failures - total_errors,
            tests_failed=total_failures,
            tests_errors=total_errors,
            test_results=test_results,
            summary=summary,
        )

    def _analyze_execution_error(
        self, basic_result: BasicExecutionResult
    ) -> TestExecutionResult:
        """Analyze execution results when XML parsing failed or wasn't available."""
        script_error = False
        summary = ""

        # Check if this was a script-level error (syntax, import, etc.)
        if not basic_result.success:
            # Look for common error patterns in stderr
            error_output = (basic_result.error or "").lower()
            if any(
                error_type in error_output
                for error_type in [
                    "syntaxerror",
                    "indentationerror",
                    "importerror",
                    "modulenotfounderror",
                    "attributeerror",
                    "nameerror",
                    "typeerror",
                ]
            ):
                script_error = True
                summary = f"Script execution failed: {basic_result.error}"
            else:
                # Other execution failure
                script_error = True
                summary = f"Test execution failed: {basic_result.error}"
        else:
            # Execution succeeded but no XML was generated
            summary = "Test execution completed but no XML output was generated"

        return TestExecutionResult(
            script_error=script_error,
            tests_passed=0,
            tests_failed=0,
            tests_errors=0,
            test_results=[],
            summary=summary,
        )


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
        self.blocked_patterns = [
            "import os",
            # "import sys", # TODO: Consider allowing sys with restrictions
            "import subprocess",
            "import shutil",
            "import socket",
            "import urllib",
            "import requests",
            "import http",
            "import ftplib",
            "import smtplib",
            "import telnetlib",
            "import tempfile",
            "import pickle",
            "import marshal",
            "import importlib",
            "import __import__",
            "exec(",
            # "eval(", # TODO: Consider allowing eval with restrictions, this was needed for generating test with actual lcb test cases
            "compile(",
            "open(",
            "file(",
            # "input(",
            "raw_input(",
            "__builtins__",
            "__globals__",
            "__locals__",
            # "globals()",
            # "locals()", # TODO: Consider allowing globals/locals with restrictions
            "vars()",
            "dir()",
            "hasattr(",
            "getattr(",
            "setattr(",
            "delattr(",
            "exit(",
            "quit(",
            "reload(",
        ]

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


def create_safe_test_environment(
    test_method_timeout: Optional[int] = 30,
    script_timeout: int = 300,  # 300 seconds (5 mins) max for entire script
) -> SafeCodeSandbox:
    """
    Create a default safe test environment.

    Args:
        test_method_timeout: Maximum execution time in seconds for individual test methods (default: 30)

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


# Example usage
if __name__ == "__main__":
    # Example 1: Using the high-level TestExecutor (recommended for most cases)
    executor = create_test_executor()

    safe_code = """
import unittest
import math

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class TestFactorial(unittest.TestCase):
    def test_base_case(self):
        '''Test factorial of 0 and 1'''
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)
    
    def test_normal_case(self):
        '''Test factorial of larger numbers'''
        self.assertEqual(factorial(5), 120)

if __name__ == '__main__':
    unittest.main()
"""

    # Execute with full test analysis
    result = executor.execute_test_script(safe_code)
    print("Test execution result:", result)
    print("Status:", check_test_execution_status(result))

    # Example 2: Using components separately for more control
    sandbox = create_safe_test_environment()
    analyzer = PytestXmlAnalyzer()

    # Execute test script directly
    detailed_result = sandbox.execute_test_script(safe_code)

    print("\nDetailed analysis:")
    print(f"- Passed: {detailed_result.tests_passed}")
    print(f"- Failed: {detailed_result.tests_failed}")
    print(f"- Errors: {detailed_result.tests_errors}")
    print(f"- Summary: {detailed_result.summary}")
    print(
        f"- Test results (in order): {[f'{t.name}:{t.status}' for t in detailed_result.test_results]}"
    )

    # Example 3: Using sandbox for non-test code
    simple_result = sandbox.execute_code("print('Hello from sandbox!')")
    print(
        f"\nSimple execution: {simple_result.success}, output: {simple_result.output.strip()}"
    )
