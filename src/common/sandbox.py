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


class TestResultAnalyzer:
    """
    Analyzes unittest execution output to extract detailed test results.

    This class is responsible for parsing unittest output formats and
    categorizing test results. It works independently of the execution
    environment and can analyze any unittest output.
    """

    def __init__(self) -> None:
        """Initialize the test result analyzer."""
        pass

    def analyze_unittest_output(
        self, basic_result: BasicExecutionResult
    ) -> TestExecutionResult:
        """
        Analyze unittest execution results and return detailed test information.

        Args:
            basic_result: Basic execution result from code execution

        Returns:
            TestExecutionResult with detailed test analysis
        """
        logger.debug(
            f"Analyzing unittest output: success={basic_result.success} return_code={basic_result.return_code} elapsed={basic_result.execution_time:.3f}s"
        )

        # Parse unittest output regardless of success/failure
        # because unittest failures cause non-zero exit codes
        test_analysis = self._parse_unittest_output(
            basic_result.output, basic_result.error or ""
        )

        # Determine script error and create summary
        script_error = False
        summary = ""

        # Check if failure is due to script errors vs test failures
        if not basic_result.success and test_analysis.script_error:
            # True script error (syntax, import, etc.)
            script_error = True
            summary = f"Script execution failed: {basic_result.error}"
            logger.debug(
                f"Detected script error while analyzing unittest output: {basic_result.error}",
            )
        elif test_analysis.failed > 0 or test_analysis.errors > 0:
            # Tests ran but some failed
            summary = f"Tests completed: {test_analysis.passed} passed, {test_analysis.failed} failed, {test_analysis.errors} errors"
            logger.debug(f"Test run had failures/errors: {summary}")
        elif test_analysis.passed > 0:
            # All tests passed
            summary = f"All tests passed: {test_analysis.passed} tests"
            logger.debug(summary)
        elif "NO TESTS RAN" in basic_result.error or (
            basic_result.return_code == 5 and "Ran 0 tests" in basic_result.error
        ):
            # No tests found scenario (unittest exit code 5)
            summary = "No tests were found or executed"
        elif not basic_result.success:
            # Script failed but no unittest patterns found - likely script error
            script_error = True
            summary = f"Script execution failed: {basic_result.error}"
        else:
            # Script succeeded but no tests found
            summary = "No tests were found or executed"

        return TestExecutionResult(
            script_error=script_error,
            tests_passed=test_analysis.passed,
            tests_failed=test_analysis.failed,
            tests_errors=test_analysis.errors,
            test_results=test_analysis.test_results,
            summary=summary,
        )

    def ensure_unittest_verbosity(self, test_script: str) -> str:
        """
        Ensure that unittest.main() calls in the script use verbosity=2.

        Args:
            test_script: The original test script

        Returns:
            Modified test script with verbosity=2 for unittest.main() calls
        """
        modified_script = test_script
        logger.trace(
            f"Ensuring unittest verbosity for test script (len={len(test_script)})"
        )

        # Replace unittest.main() with unittest.main(verbosity=2)
        modified_script = re.sub(
            r"unittest\.main\(\)", "unittest.main(verbosity=2)", modified_script
        )

        # Handle all unittest.main() calls - upgrade verbosity and add if missing
        def process_unittest_main(match: re.Match[str]) -> str:
            params = match.group(1).strip()

            # Check if verbosity is already present
            verbosity_match = re.search(r"verbosity\s*=\s*([01])", params)
            if verbosity_match:
                # Replace verbosity=0 or verbosity=1 with verbosity=2
                new_params = re.sub(r"verbosity\s*=\s*[01]", "verbosity=2", params)
                return f"unittest.main({new_params})"
            elif "verbosity" in params:
                # verbosity=2 already present, keep unchanged
                return match.group(0)
            else:
                # No verbosity parameter, add it
                if params:
                    return f"unittest.main({params}, verbosity=2)"
                else:
                    return "unittest.main(verbosity=2)"

        modified_script = re.sub(
            r"unittest\.main\(([^)]*)\)", process_unittest_main, modified_script
        )

        if modified_script != test_script:
            logger.trace("Modified test script to increase unittest verbosity")

        return modified_script

    def _parse_unittest_output(self, stdout: str, stderr: str) -> TestAnalysis:
        """
        Parse unittest output to extract detailed test results.

        Args:
            stdout: Standard output from test execution
            stderr: Standard error from test execution

        Returns:
            TestAnalysis with parsed test results
        """
        # Check for script-level errors first (syntax, import errors)
        # But exclude unittest failure messages which also appear in stderr
        script_error = False
        if stderr and any(
            error_type in stderr.lower()
            for error_type in [
                "syntaxerror",
                "indentationerror",
                "importerror",
                "modulenotfounderror",
            ]
        ):
            # Make sure it's not just unittest output in stderr
            if not ("Ran " in stderr and ("FAILED" in stderr or "OK" in stderr)):
                script_error = True
                logger.debug(
                    "Detected script-level error in stderr while parsing unittest output"
                )
                return TestAnalysis(
                    passed=0,
                    failed=0,
                    errors=0,
                    script_error=script_error,
                    test_results=[],
                )

        # Combine stdout and stderr for comprehensive parsing
        full_output = stdout + "\n" + stderr

        # Initialize counters
        passed = 0
        failed = 0
        errors = 0

        # Pattern for unittest summary line (e.g., "Ran 5 tests in 0.001s")
        ran_pattern = r"Ran (\d+) tests? in ([\d.]+)s"
        ran_match = re.search(ran_pattern, full_output)

        if ran_match:
            total_tests = int(ran_match.group(1))
            logger.trace(f"Parsed unittest summary: total_tests={total_tests}")

            # Look for OK (all passed)
            if re.search(r"\nOK\s*$", full_output, re.MULTILINE):
                passed = total_tests

            # Look for FAILED with details
            failed_pattern = r"FAILED \((.+?)\)"
            failed_match = re.search(failed_pattern, full_output)

            if failed_match:
                failure_info = failed_match.group(1)

                # Parse failures and errors
                failures_match = re.search(r"failures=(\d+)", failure_info)
                errors_match = re.search(r"errors=(\d+)", failure_info)

                failures = int(failures_match.group(1)) if failures_match else 0
                errors = int(errors_match.group(1)) if errors_match else 0

                failed = failures
                errors = errors
                passed = total_tests - failures - errors
            else:
                # Check if we have failure output but no explicit FAILED line
                # This can happen when tests fail but the format is different
                if "FAIL:" in full_output or "ERROR:" in full_output:
                    # Count failures and errors manually
                    fail_count = len(re.findall(r"FAIL:", full_output))
                    error_count = len(re.findall(r"ERROR:", full_output))

                    failed = fail_count
                    errors = error_count
                    passed = max(0, total_tests - fail_count - error_count)
        else:
            # No "Ran X tests" found, try to parse test dots/letters
            passed, failed, errors = self._parse_test_dots(full_output)

        # Parse individual test results from detailed output (in execution order)
        test_results = self._parse_individual_tests(full_output)

        return TestAnalysis(
            passed=passed,
            failed=failed,
            errors=errors,
            script_error=script_error,
            test_results=test_results,
        )

    def _parse_individual_tests(self, output: str) -> List[TestResult]:
        """Parse individual test method results from verbose output, maintaining execution order."""

        test_results = []

        # Pattern for verbose unittest output - handle both cases:
        # Case 1: test_name (module.path)\nDescription ... STATUS  (with docstring)
        # Case 2: test_name (module.path) ... STATUS  (without docstring)

        # Split output into lines and process sequentially to maintain order
        lines = output.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Look for test execution lines
            # Pattern: test_name (module.TestClass) ... ok/FAIL/ERROR
            test_match = re.match(r"(\w+) \([^)]+\) \.\.\. (ok|FAIL|ERROR)", line)
            if test_match:
                test_name, status = test_match.groups()
                logger.trace(f"Found individual test line: {test_name} -> {status}")

                # Look for description on previous line (if it's not a test line)
                description = f"Test method: {test_name}"  # Default
                if i > 0:
                    prev_line = lines[i - 1].strip()
                    # Check if previous line looks like a description (not a test result line)
                    if prev_line and not re.match(
                        r"\w+ \([^)]+\) \.\.\. (ok|FAIL|ERROR)", prev_line
                    ):
                        description = prev_line

                # Create TestResult with basic info
                test_result = TestResult(
                    name=test_name,
                    description=description,
                    status="passed"
                    if status == "ok"
                    else "failed"
                    if status == "FAIL"
                    else "error",
                    details=None,
                )

                test_results.append(test_result)
                logger.trace(
                    f"Appended TestResult: {test_name} status={test_result.status}"
                )

            i += 1

        # Now extract detailed failure/error information
        self._extract_detailed_failures(output, test_results)

        return test_results

    def _extract_detailed_failures(
        self, output: str, test_results: List[TestResult]
    ) -> None:
        """Extract detailed failure and error information from unittest output."""

        # Pattern for detailed failure/error sections:
        # ======================================================================
        # FAIL/ERROR: test_name (module.path)
        # ----------------------------------------------------------------------
        # Traceback or error details...

        # Find all detailed sections
        detailed_sections = re.split(r"={70,}", output)
        logger.trace(
            f"Found {len(detailed_sections)} detailed sections when extracting failures/errors"
        )

        for section in detailed_sections:
            if not section.strip():
                continue

            # Look for FAIL or ERROR headers
            fail_match = re.search(
                r"FAIL: (\w+) \([^)]+\)\n.*?-{70,}\n(.+?)(?=\n={70,}|\n-{70,}|$)",
                section,
                re.DOTALL,
            )
            error_match = re.search(
                r"ERROR: (\w+) \([^)]+\)\n.*?-{70,}\n(.+?)(?=\n={70,}|\n-{70,}|$)",
                section,
                re.DOTALL,
            )

            if fail_match:
                test_name, error_details = fail_match.groups()
                logger.trace(f"Found detailed FAIL for test {test_name}")
                self._update_test_details(
                    test_results, test_name.strip(), error_details.strip()
                )

            elif error_match:
                test_name, error_details = error_match.groups()
                logger.trace(f"Found detailed ERROR for test {test_name}")
                self._update_test_details(
                    test_results, test_name.strip(), error_details.strip()
                )

    def _update_test_details(
        self, test_results: List[TestResult], test_name: str, error_details: str
    ) -> None:
        """Update the details field for a specific test."""
        for test_result in test_results:
            if test_result.name == test_name:
                # Clean the error details to extract only relevant parts
                test_result.details = self._clean_error_details(error_details)
                break

    def _clean_error_details(self, raw_details: str) -> str:
        """
        Extract only the relevant assertion line and error message from traceback.

        Args:
            raw_details: Raw traceback string

        Returns:
            Cleaned error details with just the assertion and error message
        """
        if not raw_details:
            return ""

        lines = raw_details.strip().split("\n")

        # Find the assertion line and error message
        assertion_line = ""
        error_message = ""

        for i, line in enumerate(lines):
            # Look for assertion lines (indented and contain self.assert)
            if line.strip().startswith("self.assert") or "~~~" in line:
                # Get the actual assertion (previous line if current is ~~~)
                if "~~~" in line and i > 0:
                    assertion_line = lines[i - 1].strip()
                elif line.strip().startswith("self.assert"):
                    assertion_line = line.strip()

            # Look for error messages (AssertionError, ValueError, etc.)
            elif any(
                error_type in line
                for error_type in [
                    "AssertionError:",
                    "ValueError:",
                    "TypeError:",
                    "IndexError:",
                    "KeyError:",
                    "AttributeError:",
                ]
            ):
                error_message = line.strip()

        # Combine assertion and error message
        result_parts = []
        if assertion_line:
            result_parts.append(assertion_line)
        if error_message:
            result_parts.append(error_message)

        return "\n".join(result_parts) if result_parts else raw_details

    def _parse_test_dots(self, output: str) -> tuple[int, int, int]:
        """Parse test results from dot notation (e.g., '..F.E.')

        Returns:
            Tuple of (passed, failed, errors) counts
        """
        passed = 0
        failed = 0
        errors = 0

        # Find test result dots/letters
        dot_pattern = r"^([.FE]+)$"
        matches = re.findall(dot_pattern, output, re.MULTILINE)

        for match in matches:
            for char in match:
                if char == ".":
                    passed += 1
                elif char == "F":
                    failed += 1
                elif char == "E":
                    errors += 1

        return passed, failed, errors


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
    ):
        """
        Initialize the sandbox with safety parameters.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB (not enforced on all systems)
            max_output_size: Maximum output size in characters
            allowed_imports: List of allowed import modules
            python_executable: Path to Python executable (defaults to sys.executable)
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size
        self.python_executable = python_executable or sys.executable
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
            "input(",
            "raw_input(",
            "__builtins__",
            "__globals__",
            "__locals__",
            "globals()",
            "locals()",
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

        logger.debug(
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

            logger.debug(f"Execution finished: returncode={result.returncode}")

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

    def execute_test_script(self, test_script: str) -> TestExecutionResult:
        """
        Execute a test script and return detailed results with test categorization.

        This method is a convenience wrapper that combines safe code execution
        with unittest output analysis. For more control over the process, use
        execute_code() and TestResultAnalyzer separately.

        Args:
            test_script: The test script to execute

        Returns:
            TestExecutionResult containing comprehensive execution results
        """
        # Create analyzer and prepare the test script
        logger.debug(f"SafeCodeSandbox: executing test script (len={len(test_script)})")
        analyzer = TestResultAnalyzer()
        modified_script = analyzer.ensure_unittest_verbosity(test_script)

        # Execute the code safely
        basic_result = self.execute_code(modified_script)

        # Analyze the test results
        result = analyzer.analyze_unittest_output(basic_result)
        logger.debug(f"SafeCodeSandbox: test script result summary: {result.summary}")
        return result


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
    ):
        """
        Initialize test executor with sandbox configuration.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            max_output_size: Maximum output size in characters
            allowed_imports: List of allowed import modules
            python_executable: Path to Python executable
        """
        self.sandbox = SafeCodeSandbox(
            timeout=timeout,
            max_memory_mb=max_memory_mb,
            max_output_size=max_output_size,
            allowed_imports=allowed_imports,
            python_executable=python_executable,
        )
        self.analyzer = TestResultAnalyzer()
        logger.debug(
            f"Initialized TestExecutor(timeout={timeout}, max_memory_mb={max_memory_mb}, max_output_size={max_output_size})"
        )

    def execute_test_script(self, test_script: str) -> TestExecutionResult:
        """
        Execute a test script and return comprehensive results.

        Args:
            test_script: The test script to execute

        Returns:
            TestExecutionResult with detailed analysis
        """
        logger.debug(f"TestExecutor: executing test script (len={len(test_script)})")
        # Prepare the test script with proper verbosity
        modified_script = self.analyzer.ensure_unittest_verbosity(test_script)

        # Execute safely in the sandbox
        basic_result = self.sandbox.execute_code(modified_script)

        # Analyze the results
        result = self.analyzer.analyze_unittest_output(basic_result)
        logger.debug(f"TestExecutor: finished execution: {result.summary}")
        return result

    def execute_code(self, code: str) -> BasicExecutionResult:
        """
        Execute arbitrary code safely (delegates to sandbox).

        Args:
            code: Python code to execute

        Returns:
            BasicExecutionResult
        """
        return self.sandbox.execute_code(code)


def create_safe_test_environment() -> SafeCodeSandbox:
    """
    Create a default safe test environment.

    Returns:
        Configured SafeCodeSandbox instance
    """
    return SafeCodeSandbox(
        timeout=120,  # 120 seconds max
        max_memory_mb=100,  # 100MB max memory
        max_output_size=1_000_000,  # 1MB max output
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
            "typing",
            "dataclasses",
            "enum",
            "heapq",
            "bisect",
        ],
    )


def create_test_executor() -> TestExecutor:
    """
    Create a default test executor with safe configuration.

    Returns:
        Configured TestExecutor instance
    """
    return TestExecutor(
        timeout=30,  # 30 seconds max
        max_memory_mb=100,  # 100MB max memory
        max_output_size=1_000_000,  # 1MB max output
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
    analyzer = TestResultAnalyzer()

    # Prepare and execute
    modified_code = analyzer.ensure_unittest_verbosity(safe_code)
    basic_result = sandbox.execute_code(modified_code)
    detailed_result = analyzer.analyze_unittest_output(basic_result)

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
