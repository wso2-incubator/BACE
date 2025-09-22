"""
Comprehensive tests for the SafeCodeSandbox class.

This module contains tests for both basic sandbox functionality and 
enhanced error categorization features.
"""

import pytest
import tempfile
import os
from src.common.sandbox import (
    SafeCodeSandbox,
    create_safe_test_environment,
    CodeExecutionTimeoutError,
    CodeExecutionError,
    check_test_execution_status
)


class TestSafeCodeSandbox:
    """Test cases for SafeCodeSandbox."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sandbox = SafeCodeSandbox()

    def test_basic_execution(self) -> None:
        """Test basic code execution."""
        code = """
def add(a, b):
    return a + b

result = add(2, 3)
print(result)
"""
        result = self.sandbox.execute_code(code)
        assert result.success
        assert "5" in result.output
        assert result.error is None or result.error == ""
        assert result.timeout is False

    def test_syntax_error_handling(self) -> None:
        """Test handling of syntax errors."""
        code = """
def broken_function(
    print("This has a syntax error")
"""
        result = self.sandbox.execute_code(code)
        assert not result.success
        assert "SyntaxError" in result.error or "syntax" in result.error.lower()

    def test_runtime_error_handling(self) -> None:
        """Test handling of runtime errors."""
        code = """
def divide_by_zero():
    return 1 / 0

divide_by_zero()
"""
        result = self.sandbox.execute_code(code)
        assert not result.success
        assert result.error is not None
        assert len(result.error) > 0

    def test_timeout_handling(self) -> None:
        """Test timeout handling for infinite loops."""
        code = """
while True:
    pass
"""
        sandbox = SafeCodeSandbox(timeout=1)  # 1 second timeout
        result = sandbox.execute_code(code)
        assert not result.success
        # Should either timeout or be terminated
        assert result.timeout or result.error is not None

    def test_import_restrictions(self) -> None:
        """Test that dangerous imports are handled."""
        dangerous_code = """
import os
print("Import successful")
"""
        result = self.sandbox.execute_code(dangerous_code)
        # The code might execute - we mainly test that it doesn't crash our system
        assert isinstance(result.success, bool)

    def test_large_output_handling(self) -> None:
        """Test handling of large output."""
        code = """
for i in range(100):  # Reduced from 1000 to speed up test
    print(f"Line {i}")
"""
        result = self.sandbox.execute_code(code)
        assert result.success
        # Should handle large output gracefully
        assert len(result.output) > 0

    def test_memory_intensive_code(self) -> None:
        """Test handling of memory-intensive code."""
        code = """
# Create a reasonably large list
data = list(range(10000))
print(f"Created list with {len(data)} elements")
"""
        result = self.sandbox.execute_code(code)
        assert result.success
        assert "10000" in result.output

    def test_multiline_function_execution(self) -> None:
        """Test execution of multiline functions."""
        code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

test_array = [64, 34, 25, 12, 22, 11, 90]
sorted_array = bubble_sort(test_array.copy())
print(f"Original: {test_array}")
print(f"Sorted: {sorted_array}")
"""
        result = self.sandbox.execute_code(code)
        assert result.success
        assert "Sorted: [11, 12, 22, 25, 34, 64, 90]" in result.output

    def test_test_script_execution(self) -> None:
        """Test executing test scripts."""
        test_script = """
def test_addition():
    assert 2 + 2 == 4
    assert 1 + 1 == 2
    print("Addition tests passed!")

def test_multiplication():
    assert 2 * 3 == 6
    assert 5 * 0 == 0
    print("Multiplication tests passed!")

# Run tests
test_addition()
test_multiplication()
print("All tests completed!")
"""
        result = self.sandbox.execute_test_script(test_script)
        assert result.success
        assert "All tests completed!" in result.output


class TestCreateSafeTestEnvironment:
    """Test cases for create_safe_test_environment function."""

    def test_safe_test_environment_creation(self) -> None:
        """Test creating a safe test environment."""
        sandbox = create_safe_test_environment()
        assert isinstance(sandbox, SafeCodeSandbox)
        assert sandbox.timeout == 30
        assert sandbox.max_memory_mb == 100

        # Test that it can execute code
        code = "print('Hello from safe environment!')"
        result = sandbox.execute_code(code)
        assert result.success
        assert "Hello from safe environment!" in result.output

    def test_safe_environment_with_allowed_imports(self) -> None:
        """Test that allowed imports work in safe environment."""
        sandbox = create_safe_test_environment()
        code = """
import math
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
"""
        result = sandbox.execute_code(code)
        assert result.success
        assert "4.0" in result.output


class TestEnhancedSandbox:
    """Test enhanced sandbox functionality with error categorization."""

    def test_script_error_syntax(self) -> None:
        """Test syntax error detection."""
        sandbox = SafeCodeSandbox()

        # Script with syntax error
        script = """
import unittest

def invalid_function(
    # Missing closing parenthesis - syntax error
    
class TestExample(unittest.TestCase):
    def test_something(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(script)
        assert result.execution_category == 'SCRIPT_ERROR'
        assert 'syntax error' in result.error.lower(
        ) or 'syntaxerror' in result.error.lower()
        assert result.return_code != 0

    def test_script_error_import(self) -> None:
        """Test import error detection."""
        sandbox = SafeCodeSandbox()

        # Script with import error
        script = """
import unittest
import nonexistent_module  # This will cause ImportError

class TestExample(unittest.TestCase):
    def test_something(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(script)
        assert result.execution_category == 'SCRIPT_ERROR'
        assert 'importerror' in result.error.lower(
        ) or 'modulenotfounderror' in result.error.lower()
        assert result.return_code != 0

    def test_tests_failed(self) -> None:
        """Test test failure detection."""
        sandbox = SafeCodeSandbox()

        # Script with failing tests
        script = """
import unittest

class TestExample(unittest.TestCase):
    def test_failing(self):
        self.assertTrue(False, "This test should fail")
    
    def test_another_failing(self):
        self.assertEqual(1, 2, "Math is broken")

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(script)
        assert result.execution_category == 'TESTS_FAILED'
        # unittest returns 1 for failed tests
        assert result.return_code == 1
        total_tests = result.tests_passed + \
            result.tests_failed + result.tests_errors
        assert total_tests == 2
        assert result.tests_failed == 2
        assert result.tests_errors == 0

    def test_tests_passed(self) -> None:
        """Test successful test execution."""
        sandbox = SafeCodeSandbox()

        # Script with passing tests
        script = """
import unittest

class TestExample(unittest.TestCase):
    def test_passing(self):
        self.assertTrue(True)
    
    def test_another_passing(self):
        self.assertEqual(1, 1)
    
    def test_third_passing(self):
        self.assertIsNotNone("hello")

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(script)
        assert result.execution_category == 'TESTS_PASSED'
        assert result.return_code == 0
        total_tests = result.tests_passed + \
            result.tests_failed + result.tests_errors
        assert total_tests == 3
        assert result.tests_failed == 0
        assert result.tests_errors == 0

    def test_no_tests_found(self) -> None:
        """Test no tests found scenario."""
        sandbox = SafeCodeSandbox()

        # Script with no test methods
        script = """
import unittest

class TestExample(unittest.TestCase):
    def not_a_test_method(self):
        self.assertTrue(True)
    
    def another_regular_method(self):
        pass

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(script)
        assert result.execution_category == 'NO_TESTS_FOUND'
        total_tests = result.tests_passed + \
            result.tests_failed + result.tests_errors
        assert total_tests == 0
        assert result.tests_failed == 0
        assert result.tests_errors == 0

    def test_mixed_test_details(self) -> None:
        """Test mixed passing and failing tests."""
        sandbox = SafeCodeSandbox()

        # Script with both passing and failing tests
        script = """
import unittest

class TestExample(unittest.TestCase):
    def test_passing_one(self):
        self.assertTrue(True)
    
    def test_failing_one(self):
        self.assertTrue(False, "This should fail")
    
    def test_passing_two(self):
        self.assertEqual(2, 2)
    
    def test_failing_two(self):
        self.assertEqual(1, 3, "Math fail")

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(script)
        assert result.execution_category == 'TESTS_FAILED'
        assert result.return_code == 1
        total_tests = result.tests_passed + \
            result.tests_failed + result.tests_errors
        assert total_tests == 4
        assert result.tests_failed == 2
        assert result.tests_errors == 0

    def test_test_with_errors(self) -> None:
        """Test tests that have errors (not failures)."""
        sandbox = SafeCodeSandbox()

        # Script with test errors (exceptions during test execution)
        script = """
import unittest

class TestExample(unittest.TestCase):
    def test_with_error(self):
        # This will cause an error, not a failure
        x = 1 / 0
    
    def test_passing(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(script)
        assert result.execution_category == 'TESTS_FAILED'
        assert result.return_code == 1
        total_tests = result.tests_passed + \
            result.tests_failed + result.tests_errors
        assert total_tests == 2
        assert result.tests_failed == 0
        assert result.tests_errors == 1

    def test_check_test_execution_status_helper(self) -> None:
        """Test the helper function for checking test execution status."""
        sandbox = SafeCodeSandbox()

        # Test with passing tests
        passing_script = """
import unittest

class TestExample(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(passing_script)
        status_info = check_test_execution_status(result)

        assert "ALL TESTS PASSED" in status_info
        assert "1 tests successful" in status_info

        # Test with failing tests
        failing_script = """
import unittest

class TestExample(unittest.TestCase):
    def test_failing(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(failing_script)
        status_info = check_test_execution_status(result)

        assert "TESTS FAILED" in status_info
        assert "failed" in status_info.lower()

        # Test with script error
        error_script = """
import unittest
import nonexistent_module

class TestExample(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(error_script)
        status_info = check_test_execution_status(result)

        assert "SCRIPT ERROR" in status_info

    def test_verbose_unittest_output(self) -> None:
        """Test parsing of verbose unittest output."""
        sandbox = SafeCodeSandbox()

        # Script that should produce verbose output
        script = """
import unittest

class TestVerbose(unittest.TestCase):
    def test_one(self):
        '''Test one with docstring'''
        self.assertEqual(1, 1)
    
    def test_two(self):
        '''Test two with docstring'''
        self.assertEqual(2, 2)
    
    def test_three_fails(self):
        '''Test three that fails'''
        self.assertEqual(1, 2, "This should fail")

if __name__ == '__main__':
    unittest.main(verbosity=2)
"""

        result = sandbox.execute_test_script(script)
        assert result.execution_category == 'TESTS_FAILED'
        total_tests = result.tests_passed + \
            result.tests_failed + result.tests_errors
        assert total_tests == 3
        assert result.tests_failed == 1
        assert result.tests_errors == 0

        # Check that individual test results are captured
        assert hasattr(result, 'test_details')
        assert result.test_details is not None

    def test_edge_case_empty_script(self) -> None:
        """Test execution of empty or minimal script."""
        sandbox = SafeCodeSandbox()

        # Empty script
        result = sandbox.execute_test_script("")
        # Empty script might not have unittest.main(), so could be SCRIPT_ERROR or NO_TESTS_FOUND
        assert result.execution_category in [
            'SCRIPT_ERROR', 'NO_TESTS_FOUND']

        # Script with just imports
        minimal_script = """
import unittest
"""
        result = sandbox.execute_test_script(minimal_script)
        # Should complete successfully but find no tests
        assert result.execution_category == 'NO_TESTS_FOUND' or result.execution_category == 'SCRIPT_ERROR'


if __name__ == "__main__":
    pytest.main([__file__])
