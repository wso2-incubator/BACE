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
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil


class CodeExecutionTimeoutError(Exception):
    """Raised when code execution times out."""
    pass


class CodeExecutionError(Exception):
    """Raised when code execution fails."""
    pass


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

    def execute_code(self, code: str, capture_output: bool = True) -> Dict[str, Any]:
        """
        Safely execute Python code in a restricted environment.

        Args:
            code: The Python code to execute
            capture_output: Whether to capture stdout/stderr

        Returns:
            Dictionary containing execution results:
            - success: bool indicating if execution succeeded
            - output: captured stdout
            - error: captured stderr or error message
            - execution_time: time taken in seconds
            - timeout: whether execution timed out
        """
        # Check code safety first
        if not self._check_code_safety(code):
            return {
                'success': False,
                'output': '',
                'error': 'Code contains potentially dangerous patterns',
                'execution_time': 0,
                'timeout': False
            }

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

            return {
                'success': result.returncode == 0,
                'output': stdout,
                'error': stderr,
                'execution_time': execution_time,
                'timeout': False,
                'return_code': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': f'Code execution timed out after {self.timeout} seconds',
                'execution_time': self.timeout,
                'timeout': True,
                'return_code': -1
            }

        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Execution error: {str(e)}',
                'execution_time': 0,
                'timeout': False,
                'return_code': -1
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

    def execute_test_script(self, test_script: str) -> Dict[str, Any]:
        """
        Execute a test script and return detailed results with test categorization.

        Args:
            test_script: The test script to execute

        Returns:
            Dictionary with execution results and detailed test outcomes:
            - success: bool indicating if script executed without errors
            - script_error: bool indicating if there was a script execution error
            - tests_passed: int number of tests that passed
            - tests_failed: int number of tests that failed
            - tests_errors: int number of tests with errors
            - test_results: detailed breakdown of test results
            - execution_category: 'SCRIPT_ERROR', 'TESTS_FAILED', 'TESTS_PASSED', 'NO_TESTS_FOUND'
            - test_summary: human-readable summary
        """
        result = self.execute_code(test_script)

        # Initialize test-specific fields
        result['script_error'] = False
        result['tests_passed'] = 0
        result['tests_failed'] = 0
        result['tests_errors'] = 0
        result['test_results'] = {}
        result['execution_category'] = 'UNKNOWN'

        # Check if failure is due to script errors vs test failures
        output = result['output']
        stderr = result['error'] or ""

        # Parse unittest output regardless of success/failure
        # because unittest failures cause non-zero exit codes
        test_analysis = self._parse_unittest_output(output, stderr)

        result['tests_passed'] = test_analysis['passed']
        result['tests_failed'] = test_analysis['failed']
        result['tests_errors'] = test_analysis['errors']
        result['test_results'] = test_analysis['details']

        # Determine if this was a real script error or just test failures
        if not result['success'] and test_analysis['script_error']:
            # True script error (syntax, import, etc.)
            result['script_error'] = True
            result['execution_category'] = 'SCRIPT_ERROR'
            result['test_summary'] = f"Script execution failed: {result['error']}"
        elif test_analysis['failed'] > 0 or test_analysis['errors'] > 0:
            # Tests ran but some failed
            result['execution_category'] = 'TESTS_FAILED'
            result['test_summary'] = f"Tests completed: {test_analysis['passed']} passed, {test_analysis['failed']} failed, {test_analysis['errors']} errors"
        elif test_analysis['passed'] > 0:
            # All tests passed
            result['execution_category'] = 'TESTS_PASSED'
            result['test_summary'] = f"All tests passed: {test_analysis['passed']} tests"
        elif 'NO TESTS RAN' in result['error'] or (result['return_code'] == 5 and 'Ran 0 tests' in result['error']):
            # No tests found scenario (unittest exit code 5)
            result['execution_category'] = 'NO_TESTS_FOUND'
            result['test_summary'] = "No tests were found or executed"
        elif not result['success']:
            # Script failed but no unittest patterns found - likely script error
            result['script_error'] = True
            result['execution_category'] = 'SCRIPT_ERROR'
            result['test_summary'] = f"Script execution failed: {result['error']}"
        else:
            # Script succeeded but no tests found
            result['execution_category'] = 'NO_TESTS_FOUND'
            result['test_summary'] = "No tests were found or executed"

        return result

    def _parse_unittest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Parse unittest output to extract detailed test results.

        Args:
            stdout: Standard output from test execution
            stderr: Standard error from test execution

        Returns:
            Dictionary with parsed test results
        """
        analysis = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'script_error': False,
            'details': {
                'passed_tests': [],
                'failed_tests': [],
                'error_tests': [],
                'raw_output': stdout,
                'raw_error': stderr
            }
        }

        # Check for script-level errors first (syntax, import errors)
        # But exclude unittest failure messages which also appear in stderr
        if stderr and any(error_type in stderr.lower() for error_type in
                          ['syntaxerror', 'indentationerror', 'importerror', 'modulenotfounderror']):
            # Make sure it's not just unittest output in stderr
            if not ('Ran ' in stderr and ('FAILED' in stderr or 'OK' in stderr)):
                analysis['script_error'] = True
                return analysis

        # Combine stdout and stderr for comprehensive parsing
        full_output = stdout + "\n" + stderr

        # Pattern for unittest summary line (e.g., "Ran 5 tests in 0.001s")
        ran_pattern = r'Ran (\d+) tests? in ([\d.]+)s'
        ran_match = re.search(ran_pattern, full_output)

        if ran_match:
            total_tests = int(ran_match.group(1))

            # Look for OK (all passed)
            if re.search(r'\nOK\s*$', full_output, re.MULTILINE):
                analysis['passed'] = total_tests

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

                analysis['failed'] = failures
                analysis['errors'] = errors
                analysis['passed'] = total_tests - failures - errors
            else:
                # Check if we have failure output but no explicit FAILED line
                # This can happen when tests fail but the format is different
                if 'FAIL:' in full_output or 'ERROR:' in full_output:
                    # Count failures and errors manually
                    fail_count = len(re.findall(r'FAIL:', full_output))
                    error_count = len(re.findall(r'ERROR:', full_output))

                    analysis['failed'] = fail_count
                    analysis['errors'] = error_count
                    analysis['passed'] = max(
                        0, total_tests - fail_count - error_count)
        else:
            # No "Ran X tests" found, try to parse test dots/letters
            self._parse_test_dots(full_output, analysis)

        # Parse individual test results from detailed output
        self._parse_individual_tests(full_output, analysis)

        return analysis

    def _parse_individual_tests(self, output: str, analysis: Dict[str, Any]) -> None:
        """Parse individual test method results from verbose output."""
        # Pattern for individual test results
        test_patterns = [
            (r'(\w+\.\w+) \.\.\. ok', 'passed'),
            (r'(\w+\.\w+) \.\.\. FAIL', 'failed'),
            (r'(\w+\.\w+) \.\.\. ERROR', 'error'),
        ]

        for pattern, result_type in test_patterns:
            matches = re.findall(pattern, output)
            for test_name in matches:
                analysis['details'][f'{result_type}_tests'].append(test_name)

    def _parse_test_dots(self, output: str, analysis: Dict[str, Any]) -> None:
        """Parse test results from dot notation (e.g., '..F.E.')"""
        # Find test result dots/letters
        dot_pattern = r'^([.FE]+)$'
        matches = re.findall(dot_pattern, output, re.MULTILINE)

        for match in matches:
            for char in match:
                if char == '.':
                    analysis['passed'] += 1
                elif char == 'F':
                    analysis['failed'] += 1
                elif char == 'E':
                    analysis['errors'] += 1


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


def check_test_execution_status(result: Dict[str, Any]) -> str:
    """
    Helper function to get a human-readable status of test execution.

    Args:
        result: Result dictionary from execute_test_script

    Returns:
        String describing the execution status
    """
    category = result.get('execution_category', 'UNKNOWN')

    if category == 'SCRIPT_ERROR':
        return f"❌ SCRIPT ERROR: {result.get('error', 'Unknown error')}"
    elif category == 'TESTS_FAILED':
        passed = result.get('tests_passed', 0)
        failed = result.get('tests_failed', 0)
        errors = result.get('tests_errors', 0)
        return f"⚠️  TESTS FAILED: {passed} passed, {failed} failed, {errors} errors"
    elif category == 'TESTS_PASSED':
        passed = result.get('tests_passed', 0)
        return f"✅ ALL TESTS PASSED: {passed} tests successful"
    elif category == 'NO_TESTS_FOUND':
        return "⚪ NO TESTS: No test cases found or executed"
    else:
        return f"❓ UNKNOWN STATUS: {result.get('test_summary', 'Unknown')}"


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
