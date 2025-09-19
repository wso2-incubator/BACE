"""
Safe code execution sandbox for testing generated code.

This module provides a secure environment to execute Python code snippets
with safety restrictions to prevent harm to the local machine.
"""

import subprocess
import tempfile
import os
import signal
import time
import sys
import re
from typing import Dict, Any, Optional, List, Literal
from pathlib import Path
import shutil
from dataclasses import dataclass


class CodeExecutionTimeoutError(Exception):
    """Raised when code execution times out."""
    pass


class CodeExecutionError(Exception):
    """Raised when code execution fails."""
    pass


# Type definitions for better type safety
ExecutionCategory = Literal['SCRIPT_ERROR', 'TESTS_FAILED',
                            'TESTS_PASSED', 'NO_TESTS_FOUND', 'UNKNOWN']


@dataclass
class TestDetails:
    """Detailed breakdown of individual test results."""
    passed_tests: List[str]
    failed_tests: List[str]
    error_tests: List[str]
    raw_output: str
    raw_error: str


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
    """Comprehensive result of test script execution with detailed categorization."""
    # Basic execution info
    success: bool
    output: str
    error: str
    execution_time: float
    timeout: bool
    return_code: int

    # Test-specific analysis
    script_error: bool
    tests_passed: int
    tests_failed: int
    tests_errors: int
    test_details: TestDetails
    execution_category: ExecutionCategory
    test_summary: str

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
    def is_script_level_error(self) -> bool:
        """Whether the failure is at the script level (not test failures)."""
        return self.script_error or self.execution_category == 'SCRIPT_ERROR'


@dataclass
class TestAnalysis:
    """Internal analysis of unittest output."""
    passed: int
    failed: int
    errors: int
    script_error: bool
    details: TestDetails


class SafeCodeSandbox:
    """
    A safe sandbox environment for executing Python code with restrictions.
    """

    def __init__(self,
                 timeout: int = 30,
                 max_memory_mb: int = 100,
                 max_output_size: int = 10000,
                 allowed_imports: Optional[List[str]] = None,
                 python_executable: Optional[str] = None):
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
            'math', 'random', 'itertools', 'collections', 'functools',
            'operator', 'copy', 'json', 're', 'string', 'unittest',
            'typing', 'dataclasses', 'enum', 'heapq', 'bisect'
        ]

        # Dangerous modules/functions to block
        self.blocked_patterns = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'import socket', 'import urllib', 'import requests', 'import http',
            'import ftplib', 'import smtplib', 'import telnetlib',
            'import tempfile', 'import pickle', 'import marshal',
            'import importlib', 'import __import__', 'exec(', 'eval(',
            'compile(', 'open(', 'file(', 'input(', 'raw_input(',
            '__builtins__', '__globals__', '__locals__', 'globals()',
            'locals()', 'vars()', 'dir()', 'hasattr(', 'getattr(',
            'setattr(', 'delattr(', 'exit(', 'quit(', 'reload('
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
                return False

        return True

    def _create_restricted_environment(self) -> Dict[str, Any]:
        """
        Create a restricted environment for code execution.

        Returns:
            Dictionary with restricted built-ins
        """
        # Create a minimal safe environment
        safe_builtins = {
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
            'sum': sum, 'min': max, 'max': max, 'abs': abs,
            'round': round, 'int': int, 'float': float, 'str': str,
            'bool': bool, 'list': list, 'tuple': tuple, 'dict': dict,
            'set': set, 'frozenset': frozenset, 'type': type,
            'isinstance': isinstance, 'issubclass': issubclass,
            'print': print, 'repr': repr, 'ord': ord, 'chr': chr,
            'bin': bin, 'oct': oct, 'hex': hex, 'any': any, 'all': all,
            'pow': pow, 'divmod': divmod, 'reversed': reversed,
            'slice': slice, 'Exception': Exception, 'ValueError': ValueError,
            'TypeError': TypeError, 'IndexError': IndexError,
            'KeyError': KeyError, 'AttributeError': AttributeError,
        }

        return {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
        }

    def execute_code(self, code: str, capture_output: bool = True) -> BasicExecutionResult:
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
                output='',
                error='Code contains potentially dangerous patterns',
                execution_time=0,
                timeout=False,
                return_code=-1
            )

        # Create temporary file for code execution
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

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

            execution_time = time.time() - start_time

            # Limit output size
            stdout = result.stdout[:self.max_output_size] if result.stdout else ''
            stderr = result.stderr[:self.max_output_size] if result.stderr else ''

            return BasicExecutionResult(
                success=result.returncode == 0,
                output=stdout,
                error=stderr,
                execution_time=execution_time,
                timeout=False,
                return_code=result.returncode
            )

        except subprocess.TimeoutExpired:
            return BasicExecutionResult(
                success=False,
                output='',
                error=f'Code execution timed out after {self.timeout} seconds',
                execution_time=self.timeout,
                timeout=True,
                return_code=-1
            )

        except Exception as e:
            return BasicExecutionResult(
                success=False,
                output='',
                error=f'Execution error: {str(e)}',
                execution_time=0,
                timeout=False,
                return_code=-1
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
        The test script is expected to use the unittest framework.
        The verbosity of unittest output is set to 2 for detailed results.

        Args:
            test_script: The test script to execute

        Returns:
            TestExecutionResult containing comprehensive execution results including:
            - success: bool indicating if script executed without errors
            - output: str - Standard output
            - error: str - Standard error  
            - execution_time: float - Execution time in seconds
            - timeout: bool - Whether execution timed out
            - return_code: int - Process return code
            - script_error: bool indicating if there was a script execution error
            - tests_passed: int number of tests that passed
            - tests_failed: int number of tests that failed
            - tests_errors: int number of tests with errors
            - test_details: TestDetails - detailed breakdown of test results
            - execution_category: ExecutionCategory - categorized execution result
            - test_summary: str - human-readable summary
        """
        basic_result = self.execute_code(test_script)

        # Parse unittest output regardless of success/failure
        # because unittest failures cause non-zero exit codes
        test_analysis = self._parse_unittest_output(
            basic_result.output, basic_result.error or "")

        # Determine execution category and script error status
        script_error = False
        execution_category: ExecutionCategory = 'UNKNOWN'
        test_summary = ""

        # Check if failure is due to script errors vs test failures
        if not basic_result.success and test_analysis.script_error:
            # True script error (syntax, import, etc.)
            script_error = True
            execution_category = 'SCRIPT_ERROR'
            test_summary = f"Script execution failed: {basic_result.error}"
        elif test_analysis.failed > 0 or test_analysis.errors > 0:
            # Tests ran but some failed
            execution_category = 'TESTS_FAILED'
            test_summary = f"Tests completed: {test_analysis.passed} passed, {test_analysis.failed} failed, {test_analysis.errors} errors"
        elif test_analysis.passed > 0:
            # All tests passed
            execution_category = 'TESTS_PASSED'
            test_summary = f"All tests passed: {test_analysis.passed} tests"
        elif 'NO TESTS RAN' in basic_result.error or (basic_result.return_code == 5 and 'Ran 0 tests' in basic_result.error):
            # No tests found scenario (unittest exit code 5)
            execution_category = 'NO_TESTS_FOUND'
            test_summary = "No tests were found or executed"
        elif not basic_result.success:
            # Script failed but no unittest patterns found - likely script error
            script_error = True
            execution_category = 'SCRIPT_ERROR'
            test_summary = f"Script execution failed: {basic_result.error}"
        else:
            # Script succeeded but no tests found
            execution_category = 'NO_TESTS_FOUND'
            test_summary = "No tests were found or executed"

        return TestExecutionResult(
            # Basic execution info
            success=basic_result.success,
            output=basic_result.output,
            error=basic_result.error,
            execution_time=basic_result.execution_time,
            timeout=basic_result.timeout,
            return_code=basic_result.return_code,

            # Test-specific analysis
            script_error=script_error,
            tests_passed=test_analysis.passed,
            tests_failed=test_analysis.failed,
            tests_errors=test_analysis.errors,
            test_details=test_analysis.details,
            execution_category=execution_category,
            test_summary=test_summary
        )

    def _parse_unittest_output(self, stdout: str, stderr: str) -> TestAnalysis:
        """
        Parse unittest output to extract detailed test results.

        Args:
            stdout: Standard output from test execution
            stderr: Standard error from test execution

        Returns:
            TestAnalysis with parsed test results
        """
        details = TestDetails(
            passed_tests=[],
            failed_tests=[],
            error_tests=[],
            raw_output=stdout,
            raw_error=stderr
        )

        # Check for script-level errors first (syntax, import errors)
        # But exclude unittest failure messages which also appear in stderr
        script_error = False
        if stderr and any(error_type in stderr.lower() for error_type in
                          ['syntaxerror', 'indentationerror', 'importerror', 'modulenotfounderror']):
            # Make sure it's not just unittest output in stderr
            if not ('Ran ' in stderr and ('FAILED' in stderr or 'OK' in stderr)):
                script_error = True
                return TestAnalysis(
                    passed=0,
                    failed=0,
                    errors=0,
                    script_error=script_error,
                    details=details
                )

        # Combine stdout and stderr for comprehensive parsing
        full_output = stdout + "\n" + stderr

        # Initialize counters
        passed = 0
        failed = 0
        errors = 0

        # Pattern for unittest summary line (e.g., "Ran 5 tests in 0.001s")
        ran_pattern = r'Ran (\d+) tests? in ([\d.]+)s'
        ran_match = re.search(ran_pattern, full_output)

        if ran_match:
            total_tests = int(ran_match.group(1))

            # Look for OK (all passed)
            if re.search(r'\nOK\s*$', full_output, re.MULTILINE):
                passed = total_tests

            # Look for FAILED with details
            failed_pattern = r'FAILED \((.+?)\)'
            failed_match = re.search(failed_pattern, full_output)

            if failed_match:
                failure_info = failed_match.group(1)

                # Parse failures and errors
                failures_match = re.search(r'failures=(\d+)', failure_info)
                errors_match = re.search(r'errors=(\d+)', failure_info)

                failures = int(failures_match.group(1)
                               ) if failures_match else 0
                errors = int(errors_match.group(1)) if errors_match else 0

                failed = failures
                errors = errors
                passed = total_tests - failures - errors
            else:
                # Check if we have failure output but no explicit FAILED line
                # This can happen when tests fail but the format is different
                if 'FAIL:' in full_output or 'ERROR:' in full_output:
                    # Count failures and errors manually
                    fail_count = len(re.findall(r'FAIL:', full_output))
                    error_count = len(re.findall(r'ERROR:', full_output))

                    failed = fail_count
                    errors = error_count
                    passed = max(0, total_tests - fail_count - error_count)
        else:
            # No "Ran X tests" found, try to parse test dots/letters
            passed, failed, errors = self._parse_test_dots(full_output)

        # Parse individual test results from detailed output
        self._parse_individual_tests(full_output, details)

        return TestAnalysis(
            passed=passed,
            failed=failed,
            errors=errors,
            script_error=script_error,
            details=details
        )

    def _parse_individual_tests(self, output: str, details: TestDetails) -> None:
        """Parse individual test method results from verbose output."""
        # Patterns for unittest verbosity=2 output format
        # Format can be:
        # 1. test_name (module.path) ... result  (simple case)
        # 2. test_name (module.path)\nDocstring ... result  (with docstring)

        test_patterns = [
            # Verbose format with verbosity=2: test_name (module.path) ... ok/FAIL/ERROR
            (r'(\w+) \([^)]+\) \.\.\. ok', 'passed'),
            (r'(\w+) \([^)]+\) \.\.\. FAIL', 'failed'),
            (r'(\w+) \([^)]+\) \.\.\. ERROR', 'error'),

            # Verbose format with docstring: test_name (module.path)\nDocstring ... ok/FAIL/ERROR
            (r'(\w+) \([^)]+\)\n[^\n]+ \.\.\. ok', 'passed'),
            (r'(\w+) \([^)]+\)\n[^\n]+ \.\.\. FAIL', 'failed'),
            (r'(\w+) \([^)]+\)\n[^\n]+ \.\.\. ERROR', 'error'),

            # Standard format: FAIL: test_name (class.test_name)
            (r'FAIL: (\w+) \([^)]+\)', 'failed'),
            (r'ERROR: (\w+) \([^)]+\)', 'error'),

            # Fallback patterns for other formats
            (r'(\w+\.\w+) \.\.\. ok', 'passed'),
            (r'(\w+\.\w+) \.\.\. FAIL', 'failed'),
            (r'(\w+\.\w+) \.\.\. ERROR', 'error'),
        ]

        for pattern, result_type in test_patterns:
            matches = re.findall(pattern, output)
            for match in matches:
                # Handle both single capture group and tuple results
                if isinstance(match, tuple):
                    test_name = match[0] if match[0] else match[1]
                else:
                    test_name = match

                test_name = test_name.strip()
                if test_name and result_type == 'failed' and test_name not in details.failed_tests:
                    details.failed_tests.append(test_name)
                elif test_name and result_type == 'error' and test_name not in details.error_tests:
                    details.error_tests.append(test_name)
                elif test_name and result_type == 'passed' and test_name not in details.passed_tests:
                    details.passed_tests.append(test_name)

    def _parse_test_dots(self, output: str) -> tuple[int, int, int]:
        """Parse test results from dot notation (e.g., '..F.E.')

        Returns:
            Tuple of (passed, failed, errors) counts
        """
        passed = 0
        failed = 0
        errors = 0

        # Find test result dots/letters
        dot_pattern = r'^([.FE]+)$'
        matches = re.findall(dot_pattern, output, re.MULTILINE)

        for match in matches:
            for char in match:
                if char == '.':
                    passed += 1
                elif char == 'F':
                    failed += 1
                elif char == 'E':
                    errors += 1

        return passed, failed, errors


def create_safe_test_environment() -> SafeCodeSandbox:
    """
    Create a default safe test environment.

    Returns:
        Configured SafeCodeSandbox instance
    """
    return SafeCodeSandbox(
        timeout=30,  # 30 seconds max
        max_memory_mb=100,  # 100MB max memory
        max_output_size=10000,  # 10KB max output
        allowed_imports=[
            'math', 'random', 'itertools', 'collections', 'functools',
            'operator', 'copy', 'json', 're', 'string', 'unittest',
            'typing', 'dataclasses', 'enum', 'heapq', 'bisect'
        ]
    )


def check_test_execution_status(result: TestExecutionResult) -> str:
    """
    Helper function to get a human-readable status of test execution.

    Args:
        result: TestExecutionResult from execute_test_script

    Returns:
        String describing the execution status
    """
    category = result.execution_category

    if category == 'SCRIPT_ERROR':
        return f"❌ SCRIPT ERROR: {result.error}"
    elif category == 'TESTS_FAILED':
        return f"⚠️  TESTS FAILED: {result.tests_passed} passed, {result.tests_failed} failed, {result.tests_errors} errors"
    elif category == 'TESTS_PASSED':
        return f"✅ ALL TESTS PASSED: {result.tests_passed} tests successful"
    elif category == 'NO_TESTS_FOUND':
        return "⚪ NO TESTS: No test cases found or executed"
    else:
        return f"❓ UNKNOWN STATUS: {result.test_summary}"


# Example usage
if __name__ == "__main__":
    sandbox = create_safe_test_environment()

    # Test safe code
    safe_code = """
import unittest
import math

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class TestFactorial(unittest.TestCase):
    def test_base_case(self):
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)
    
    def test_normal_case(self):
        self.assertEqual(factorial(5), 120)

if __name__ == '__main__':
    unittest.main()
"""

    result = sandbox.execute_test_script(safe_code)
    print("Execution result:", result)
