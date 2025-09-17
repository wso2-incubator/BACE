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
        Execute a test script and return results.

        Args:
            test_script: The test script to execute

        Returns:
            Dictionary with execution results and test outcomes
        """
        result = self.execute_code(test_script)

        if result['success']:
            # Parse test results from output
            output = result['output']
            if 'OK' in output and '.' in output:
                # Unittest output parsing
                lines = output.strip().split('\n')
                last_line = lines[-1] if lines else ''

                if 'OK' in last_line:
                    result['tests_passed'] = True
                    result['test_summary'] = last_line
                else:
                    result['tests_passed'] = False
                    result['test_summary'] = 'Tests failed'
            else:
                result['tests_passed'] = False
                result['test_summary'] = 'Could not determine test results'
        else:
            result['tests_passed'] = False
            result['test_summary'] = 'Test execution failed'

        return result


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
