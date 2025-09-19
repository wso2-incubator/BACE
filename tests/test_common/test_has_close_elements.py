"""
Test case specifically for the has_close_elements code string
to demonstrate the enhanced typing system.
"""

import pytest
from common.sandbox import SafeCodeSandbox, TestExecutionResult, TestDetails


class TestHasCloseElementsDemo:
    """Test case for the has_close_elements code string provided by user."""

    def setup_method(self):
        """Set up test environment."""
        self.sandbox = SafeCodeSandbox(timeout=30)

        # The exact code string provided by the user
        self.code_string = '''
from typing import List    
import unittest

# Programmer Code
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # Sort the list of numbers
    sorted_numbers = sorted(numbers)

    # Iterate through each element in the sorted list (except the last one)
    for i in range(len(sorted_numbers) - 1):
        # If absolute difference between current and next is less than threshold
        if abs(sorted_numbers[i] - sorted_numbers[i+1]) < threshold:
            return True
    
    # Return False if no such pair found
    return False

# Tester Code
class TestHasCloseElements(unittest.TestCase):

    # Basic Test Cases
    def test_basic_no_close_elements(self):
        """Test with no two numbers being close to each other."""
        self.assertFalse(has_close_elements([1.0, 2.0, 3.0], 0.5), "Should return False if no numbers are close")

    def test_basic_with_close_elements(self):
        """Test with at least two numbers being close to each other."""
        self.assertTrue(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3), "Should return True if at least one pair of numbers is close")

    # Edge Test Cases
    def test_edge_empty_list(self):
        """Test with an empty list."""
        self.assertFalse(has_close_elements([], 0.1), "Should return False for an empty list")

    def test_edge_threshold_zero(self):
        """Test with threshold set to zero (should always return True)."""
        self.assertTrue(has_close_elements([1.0, 1.0], 0.0), "Should return True when threshold is zero")

    def test_edge_single_number_list(self):
        """Test with a list containing only one number."""
        self.assertFalse(has_close_elements([5.0], 2.0), "Should return False for a single element list")

    # Large Scale Test Cases
    def test_large_scale_random_numbers(self):
        """Test with a large set of random numbers and a small threshold to ensure scalability."""
        import random
        test_data = [random.uniform(1, 100) for _ in range(1000)]
        self.assertFalse(has_close_elements(test_data, 0.5), "Should return False if no numbers are close for large data")

    def test_large_scale_consecutive_numbers(self):
        """Test with a list of consecutive numbers to stress the function's performance."""
        consecutive_numbers = [i * 0.1 for i in range(1000)]
        self.assertTrue(has_close_elements(consecutive_numbers, 0.1), "Should return True if consecutive numbers are close")


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''

    def test_enhanced_typing_with_has_close_elements(self):
        """Test that demonstrates the enhanced typing system with the user's code."""

        # Execute the test script
        result: TestExecutionResult = self.sandbox.execute_test_script(
            self.code_string)

        # Verify we get the correct typed result
        assert isinstance(
            result, TestExecutionResult), f"Expected TestExecutionResult, got {type(result)}"

        # Test the basic properties
        assert result.execution_category == 'TESTS_FAILED'
        assert not result.success  # Should fail due to the 2 failing tests
        assert result.total_tests == 7
        assert result.tests_passed == 5
        assert result.tests_failed == 2
        assert result.tests_errors == 0

        # Verify enhanced typed access to test_details
        test_details: TestDetails = result.test_details
        assert isinstance(
            test_details, TestDetails), f"Expected TestDetails, got {type(test_details)}"

        # Test the test_details properties - these are now properly typed lists
        assert isinstance(test_details.failed_tests, list)
        assert isinstance(test_details.passed_tests, list)
        assert isinstance(test_details.error_tests, list)

        # Verify we can access the individual test results
        assert len(test_details.failed_tests) == 2
        # Now captured with verbosity=2
        assert len(test_details.passed_tests) == 5
        assert len(test_details.error_tests) == 0

        # Check that we can identify specific failed tests
        failed_test_names = test_details.failed_tests

        # Look for the specific failing tests we expect
        assert 'test_edge_threshold_zero' in failed_test_names
        assert 'test_large_scale_random_numbers' in failed_test_names

        # Check that passed tests are also captured
        passed_test_names = test_details.passed_tests
        expected_passed_tests = [
            'test_basic_no_close_elements',
            'test_basic_with_close_elements',
            'test_edge_empty_list',
            'test_edge_single_number_list',
            'test_large_scale_consecutive_numbers'
        ]

        for expected_test in expected_passed_tests:
            assert expected_test in passed_test_names, f"Expected {expected_test} in passed tests"

        # Test computed properties
        assert result.success_rate == 5/7  # 5 passed out of 7 total
        assert result.has_failures == True
        assert result.is_script_level_error == False

        print("✅ Enhanced typing test passed!")
        print(f"📊 Found {len(test_details.failed_tests)} failed tests:")
        for test_name in test_details.failed_tests:
            print(f"   • {test_name}")

        print(f"📈 Found {len(test_details.passed_tests)} passed tests:")
        for test_name in test_details.passed_tests:
            print(f"   • {test_name}")

        print(
            f"📈 Total test statistics: {result.tests_passed} passed, {result.tests_failed} failed, {result.tests_errors} errors")
