"""
Comprehensive tests for the SafeCodeSandbox class.

This module contains tests for both basic sandbox functionality and
enhanced error categorization features.
"""

import pytest

from common.sandbox import (
    BasicExecutionResult,
    PytestXmlAnalyzer,
    SafeCodeSandbox,
    TestExecutionResult,
    TestExecutor,
    check_test_execution_status,
    create_safe_test_environment,
    create_test_executor,
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
def addition():
    assert 2 + 2 == 4
    assert 1 + 1 == 2
    print("Addition tests passed!")

def multiplication():
    assert 2 * 3 == 6
    assert 5 * 0 == 0
    print("Multiplication tests passed!")

# Run tests
addition()
multiplication()
print("All tests completed!")
"""
        result = self.sandbox.execute_test_script(test_script)
        # Note: execute_test_script returns TestExecutionResult, not BasicExecutionResult
        # This script doesn't use unittest/pytest conventions, so it won't be parsed as tests
        assert result.summary == "No tests were found or executed"
        assert result.total_tests == 0


class TestCreateSafeTestEnvironment:
    """Test cases for create_safe_test_environment function."""

    def test_safe_test_environment_creation(self) -> None:
        """Test creating a safe test environment."""
        sandbox = create_safe_test_environment()
        assert isinstance(sandbox, SafeCodeSandbox)
        assert sandbox.timeout == 300  # Safe environment has longer timeout
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
        assert result.script_error is True
        assert "syntax" in result.summary.lower()
        assert result.tests_passed == 0
        assert result.tests_failed == 0

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
        # With pytest, import errors during collection are treated as test errors, not script errors
        assert result.script_error is False
        assert result.has_failures is True
        assert result.tests_errors == 1
        assert result.tests_passed == 0
        assert result.tests_failed == 0

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

        result: TestExecutionResult = sandbox.execute_test_script(script)
        assert result.script_error is False
        assert result.has_failures is True
        assert result.total_tests == 2
        assert result.tests_failed == 2
        assert result.tests_passed == 0
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
        assert result.all_tests_passed is True
        assert result.script_error is False
        assert result.total_tests == 3
        assert result.tests_passed == 3
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
        assert result.total_tests == 0
        assert result.tests_passed == 0
        assert result.tests_failed == 0
        assert result.tests_errors == 0
        assert (
            "no tests" in result.summary.lower() or "0 tests" in result.summary.lower()
        )

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
        assert result.has_failures is True
        assert result.script_error is False
        assert result.total_tests == 4
        assert result.tests_passed == 2
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
        # With pytest, exceptions during test execution are treated as failures
        assert result.has_failures is True
        assert result.script_error is False
        assert result.total_tests == 2
        assert result.tests_passed == 1
        assert result.tests_failed == 1  # ZeroDivisionError is treated as a failure
        assert result.tests_errors == 0

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

        # With pytest, import errors are treated as test errors, not script errors
        assert "TESTS FAILED" in status_info

    def test_failure_message_is_clean_of_ansi_codes(self) -> None:
        """
        Ensure that when a test fails, the error message in 'details'
        does not contain ANSI color codes (raw \x1b characters).
        """
        sandbox = create_safe_test_environment()

        # A script that definitely fails with an assertion error
        failing_script = """
import unittest
class TestFail(unittest.TestCase):
    def test_failure(self):
        # Pytest usually colors this comparison red/green
        self.assertEqual("foo", "bar")
if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(failing_script)

        assert result.tests_failed == 1
        failure_details = result.test_results[0].details

        # 1. Ensure the logic is still there
        assert (
            failure_details is not None
            and "foo" in failure_details
            and "bar" in failure_details
        )

        # 2. THE NEW CHECK: Ensure no escape characters exist
        # \x1b is the ESC character used in ANSI sequences
        assert failure_details is not None and "\x1b" not in failure_details, (
            "Failure details contain ANSI escape sequences!"
        )

    def test_explicit_ansi_sanitization(self) -> None:
        """
        Ensure that if a user script explicitly prints or raises errors
        with ANSI codes, the analyzer strips them out.
        """
        sandbox = create_safe_test_environment()

        # A script that uses ANSI codes in the exception message
        ansi_script = """
import unittest

# ANSI codes
RED = '\\033[91m'
RESET = '\\033[0m'

class TestANSI(unittest.TestCase):
    def test_ansi_error(self):
        # We manually inject color codes into the failure message
        raise ValueError(f"{RED}Critical Failure{RESET}")

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(ansi_script)

        assert (
            result.tests_failed == 1
        )  # Raising ValueError in test method is treated as failure
        details = result.test_results[0].details
        assert details is not None  # Ensure details exist

        # Assert we can read the text
        assert "Critical Failure" in details

        # Assert the color codes are GONE
        assert "[91m" not in details
        assert "[0m" not in details
        assert "\x1b" not in details
        assert "#x1B" not in details

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
        assert result.has_failures is True
        assert result.script_error is False
        assert result.total_tests == 3
        assert result.tests_passed == 2
        assert result.tests_failed == 1
        assert result.tests_errors == 0

        # Check that individual test results are captured
        assert len(result.test_results) == 3
        # Verify test names are captured
        test_names = [t.name for t in result.test_results]
        assert "test_one" in test_names
        assert "test_two" in test_names
        assert "test_three_fails" in test_names

    def test_verbose_unittest_output_details_and_ansi_cleaning(self) -> None:
        """Test that verbose unittest output details are captured and ANSI-clean."""
        sandbox = SafeCodeSandbox()

        # Script with verbose output and detailed failure messages
        script = """
import unittest

class TestVerboseDetails(unittest.TestCase):
    def test_passing_with_details(self):
        '''A passing test with some details'''
        result = 2 + 2
        self.assertEqual(result, 4, "Basic math should work")
    
    def test_failing_with_detailed_message(self):
        '''A failing test with detailed assertion message'''
        actual = "hello world"
        expected = "goodbye world"
        self.assertEqual(actual, expected, 
                        f"Expected '{expected}' but got '{actual}'. "
                        f"Length check: {len(actual)} vs {len(expected)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
"""

        result = sandbox.execute_test_script(script)
        assert result.has_failures is True
        assert result.script_error is False
        assert result.total_tests == 2
        assert result.tests_passed == 1
        assert result.tests_failed == 1
        assert result.tests_errors == 0

        # Check that individual test results are captured
        assert len(result.test_results) == 2

        # Find the failing test
        failing_test = next(
            t
            for t in result.test_results
            if t.name == "test_failing_with_detailed_message"
        )
        assert failing_test is not None
        assert failing_test.status == "failed"

        # Check that details contain expected verbose output elements
        details = failing_test.details
        assert details is not None

        # Should contain the assertion error details
        assert "AssertionError" in details
        assert "Expected 'goodbye world' but got 'hello world'" in details
        assert "Length check:" in details

        # Should contain verbose unittest output indicators
        # Note: pytest captures unittest output, so we look for test failure indicators
        assert "AssertionError" in details  # The exception type
        assert "test_failing_with_detailed_message" in details  # Test name in traceback

        # CRITICAL: Ensure no ANSI escape sequences
        assert "\x1b" not in details, (
            f"Found ANSI escape sequences in details: {repr(details)}"
        )
        assert "[91m" not in details, (
            f"Found ANSI color codes in details: {repr(details)}"
        )
        assert "[0m" not in details, (
            f"Found ANSI reset codes in details: {repr(details)}"
        )

        # Check that the passing test also has clean details
        passing_test = next(
            t for t in result.test_results if t.name == "test_passing_with_details"
        )
        assert passing_test is not None
        assert passing_test.status == "passed"
        if passing_test.details:
            assert "\x1b" not in passing_test.details
            assert "[91m" not in passing_test.details
            assert "[0m" not in passing_test.details

    def test_edge_case_empty_script(self) -> None:
        """Test execution of empty or minimal script."""
        sandbox = SafeCodeSandbox()

        # Empty script
        result = sandbox.execute_test_script("")
        # Empty script has no tests
        assert result.total_tests == 0
        assert "no tests" in result.summary.lower() or result.script_error

        # Script with just imports
        minimal_script = """
import unittest
"""
        result = sandbox.execute_test_script(minimal_script)
        # Should complete successfully but find no tests
        assert result.total_tests == 0
        assert result.script_error is False or "no tests" in result.summary.lower()


class TestPytestXmlAnalyzer:
    """Test cases for the new PytestXmlAnalyzer."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.analyzer = PytestXmlAnalyzer()

    def test_analyze_pytest_xml_success(self) -> None:
        """Test XML analysis for successful test execution."""
        # Mock XML content for successful tests
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="test_module" tests="2" failures="0" errors="0" skipped="0">
        <testcase name="test_one" classname="TestClass" time="0.001">
        </testcase>
        <testcase name="test_two" classname="TestClass" time="0.001">
        </testcase>
    </testsuite>
</testsuites>"""

        basic_result = BasicExecutionResult(
            success=True,
            output="",
            error="",
            execution_time=0.1,
            timeout=False,
            return_code=0,
        )

        result = self.analyzer.analyze_pytest_xml(xml_content, basic_result)

        assert result.script_error is False
        assert result.all_tests_passed is True
        assert result.total_tests == 2
        assert result.tests_passed == 2
        assert result.tests_failed == 0
        assert result.tests_errors == 0

    def test_analyze_pytest_xml_with_failures(self) -> None:
        """Test XML analysis for tests with failures."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="test_module" tests="3" failures="1" errors="1" skipped="0">
        <testcase name="test_pass" classname="TestClass" time="0.001">
        </testcase>
        <testcase name="test_fail" classname="TestClass" time="0.001">
            <failure message="AssertionError">Test failed</failure>
        </testcase>
        <testcase name="test_error" classname="TestClass" time="0.001">
            <error message="ValueError">Test error</error>
        </testcase>
    </testsuite>
</testsuites>"""

        basic_result = BasicExecutionResult(
            success=False,
            output="",
            error="",
            execution_time=0.1,
            timeout=False,
            return_code=1,
        )

        result = self.analyzer.analyze_pytest_xml(xml_content, basic_result)

        assert result.script_error is False
        assert result.has_failures is True
        assert result.total_tests == 3
        assert result.tests_passed == 1
        assert result.tests_failed == 1
        assert result.tests_errors == 1

    def test_analyze_pytest_xml_script_error(self) -> None:
        """Test XML analysis fallback for script errors."""
        # No XML content (script error)
        basic_result = BasicExecutionResult(
            success=False,
            output="",
            error="SyntaxError: invalid syntax",
            execution_time=0.1,
            timeout=False,
            return_code=1,
        )

        result = self.analyzer.analyze_pytest_xml(None, basic_result)

        assert result.script_error is True
        assert "syntax" in result.summary.lower()
        assert result.total_tests == 0

    def test_sanitizes_temp_paths_and_module_prefixes(self) -> None:
        """Ensure temporary file paths and module prefixes are sanitized."""
        # Construct XML where failure details include an absolute temp path
        # and an instance representation with a random module prefix.
        failure_text = (
            "Traceback (most recent call last):\n"
            '  File "/var/folders/bl/ydbmym3d04qb5y3mvth453g40000gp/T/tmppjtsol6j.py", line 425, in test_large_mixed_bits_min_cost\n'
            "    assert something\n"
            "AssertionError\n"
            "&lt;tmppjtsol6j.TestMakeAEqualB testMethod=test_large_mixed_bits_min_cost&gt;\n"
        )

        xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="test_module" tests="1" failures="1" errors="0" skipped="0">
        <testcase name="test_fail" classname="TestClass" time="0.001">
            <failure message="AssertionError">{failure_text}</failure>
        </testcase>
    </testsuite>
</testsuites>"""

        basic_result = BasicExecutionResult(
            success=False,
            output="",
            error="",
            execution_time=0.1,
            timeout=False,
            return_code=1,
        )

        result = self.analyzer.analyze_pytest_xml(xml_content, basic_result)

        assert result.tests_failed == 1
        assert len(result.test_results) == 1
        details = result.test_results[0].details
        assert details is not None

        # Ensure absolute temp path is not leaked and replaced with generic name
        assert "/var/folders" not in details
        assert "tmppjtsol6j" not in details
        assert "test_script.py" in details

        # Ensure module prefix was cleaned (we expect <TestMakeAEqualB ...>
        assert "<TestMakeAEqualB" in details


class TestTestExecutor:
    """Test cases for the new TestExecutor class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.executor = TestExecutor()

    def test_executor_creation(self) -> None:
        """Test TestExecutor creation and configuration."""
        assert isinstance(self.executor, TestExecutor)
        assert isinstance(self.executor.sandbox, SafeCodeSandbox)
        assert isinstance(self.executor.analyzer, PytestXmlAnalyzer)

    def test_executor_execute_test_script(self) -> None:
        """Test TestExecutor.execute_test_script method."""
        test_script = """
import unittest

class TestExample(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
"""

        result = self.executor.execute_test_script(test_script)

        assert result.script_error is False
        assert result.all_tests_passed is True
        assert result.total_tests == 1
        assert result.tests_passed == 1

    def test_executor_execute_code(self) -> None:
        """Test TestExecutor.execute_code method (delegates to sandbox)."""
        code = "print('Hello from executor!')"
        result = self.executor.execute_code(code)

        assert result.success
        assert "Hello from executor!" in result.output

    def test_create_test_executor_function(self) -> None:
        """Test the create_test_executor factory function."""
        executor = create_test_executor()
        assert isinstance(executor, TestExecutor)
        assert isinstance(executor.sandbox, SafeCodeSandbox)


class TestSandboxPytestIntegration:
    """Test cases specifically for pytest/XML integration."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sandbox = SafeCodeSandbox()

    def test_pytest_discovers_test_functions(self) -> None:
        """Test that pytest discovers functions starting with 'test_'."""
        script = """
def test_addition():
    assert 2 + 2 == 4

def test_multiplication():
    assert 3 * 3 == 9

def regular_function():
    return "not a test"
"""

        result = self.sandbox.execute_test_script(script)

        # pytest should discover and run the test_ functions
        assert result.total_tests == 2
        assert result.tests_passed == 2
        assert result.all_tests_passed is True

    def test_pytest_handles_assertions(self) -> None:
        """Test that pytest handles assertion failures correctly."""
        script = """
def test_passing():
    assert 1 == 1

def test_failing():
    assert 1 == 2, "This should fail"
"""

        result = self.sandbox.execute_test_script(script)

        assert result.has_failures is True
        assert result.total_tests == 2
        assert result.tests_passed == 1
        assert result.tests_failed == 1

    def test_pytest_syntax_error_detection(self) -> None:
        """Test that syntax errors are detected as script errors."""
        script = """
import unittest

class TestExample(unittest.TestCase):
    def test_something(self):
        self.assertTrue(True)

# Missing closing parenthesis - syntax error
def broken_function(
"""

        result = self.sandbox.execute_test_script(script)

        assert result.script_error is True
        assert "syntax" in result.summary.lower()

    def test_pytest_import_error_handling(self) -> None:
        """Test how pytest handles import errors."""
        script = """
import unittest
import nonexistent_module_12345

class TestExample(unittest.TestCase):
    def test_something(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""

        result = self.sandbox.execute_test_script(script)

        # Import errors during collection are treated as test errors
        assert result.script_error is False
        assert result.has_failures is True
        assert result.tests_errors >= 1

    def test_test_result_ordering(self) -> None:
        """Test that test results are ordered to match script order."""
        script = """
import unittest

class TestExample(unittest.TestCase):
    def test_z_last(self):
        self.assertTrue(True)

    def test_a_first(self):
        self.assertTrue(True)

    def test_m_middle(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""

        result = self.sandbox.execute_test_script(script)

        # Results should be ordered as they appear in the script
        test_names = [t.name for t in result.test_results]
        expected_order = ["test_z_last", "test_a_first", "test_m_middle"]

        # The ordering should match the script order, not pytest's execution order
        assert test_names == expected_order

    def test_empty_test_class(self) -> None:
        """Test handling of test classes with no test methods."""
        script = """
import unittest

class TestExample(unittest.TestCase):
    def not_a_test_method(self):
        pass

    def helper_method(self):
        return True

if __name__ == '__main__':
    unittest.main()
"""

        result = self.sandbox.execute_test_script(script)

        assert result.total_tests == 0
        assert "no tests" in result.summary.lower() or result.script_error is False

    def test_duplicate_test_methods(self) -> None:
        """
        Test behavior when a script defines two test methods with the same name.
        Python behavior: The second definition overwrites the first.
        Expectation: The test suite runs the second definition and reports its status.
        """
        script = """
import unittest

class TestDuplicate(unittest.TestCase):
    def test_duplicate_name(self):
        '''First definition - should be overwritten'''
        self.assertTrue(True)
        
    def test_duplicate_name(self):
        '''Second definition - should execute'''
        self.fail("The second definition runs and overwrites the first")

if __name__ == '__main__':
    unittest.main()
"""

        result = self.sandbox.execute_test_script(script)

        # 1. Verification: The script execution itself was valid (no syntax error)
        assert result.script_error is False

        # 2. Verification: Python overwrites the first method, so only 1 test actually exists at runtime.
        # Pytest XML will report 1 test total.
        assert result.total_tests == 1

        # 3. Verification: The SECOND definition is the one that runs.
        # Since the second definition calls self.fail(), the test should fail.
        assert result.tests_failed == 1
        assert result.tests_passed == 0

        # 4. Verification: The result details should confirm it was the duplicate
        assert len(result.test_results) >= 1
        # We check the first available result for the specific name
        test_result = next(
            t for t in result.test_results if t.name == "test_duplicate_name"
        )
        assert test_result.status == "failed"
        assert "The second definition runs" in (test_result.details or "")


class TestPerTestTimeout:
    """Test cases for per-test timeout functionality."""

    def test_per_test_timeout_configuration(self) -> None:
        """Test that per-test timeout can be configured."""
        sandbox = SafeCodeSandbox(test_method_timeout=2)
        assert sandbox.test_method_timeout == 2

        executor = TestExecutor(test_method_timeout=3)
        assert executor.sandbox.test_method_timeout == 3

    def test_per_test_timeout_default(self) -> None:
        """Test default per-test timeout values."""
        sandbox = create_safe_test_environment()
        assert sandbox.test_method_timeout == 30  # Default parameter value

        executor = create_test_executor()
        assert executor.sandbox.test_method_timeout == 30  # Default parameter value

    def test_per_test_timeout_disabled(self) -> None:
        """Test that per-test timeout can be disabled."""
        sandbox = SafeCodeSandbox(test_method_timeout=None)
        assert sandbox.test_method_timeout is None

        executor = TestExecutor(test_method_timeout=None)
        assert executor.sandbox.test_method_timeout is None

    def test_per_test_timeout_functionality(self) -> None:
        """Test that per-test timeout actually works."""
        sandbox = SafeCodeSandbox(test_method_timeout=1)  # 1 second per test

        # Script with one test that should pass and one that should timeout
        script = """
import unittest
import time

class TestTimeout(unittest.TestCase):
    def test_quick_pass(self):
        self.assertEqual(1, 1)

    def test_slow_timeout(self):
        time.sleep(5)  # This should timeout after 1 second
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""

        result = sandbox.execute_test_script(script)

        # Should have 2 tests total
        assert result.total_tests == 2

        # One should pass, one should fail due to timeout
        assert result.tests_passed == 1
        assert result.tests_failed == 1

        # Check that one test result mentions timeout
        timeout_found = any(
            "timeout" in (t.details or "").lower() for t in result.test_results
        )
        assert timeout_found, (
            f"No timeout found in test results: {[t.details for t in result.test_results]}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
