"""
Test case specifically for the has_close_elements code string
to demonstrate the enhanced typing system.
"""

import pytest
from common.sandbox import SafeCodeSandbox, TestExecutionResult, TestDetails


class TestHasCloseElementsDemo:
    """Test case for the has_close_elements code string provided by user."""

    def setup_method(self) -> None:
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
    
    Args:
        numbers (List[float]): The list of numbers to check.
        threshold (float): The threshold value.
        
    Returns:
        bool: True if there is at least one pair of numbers closer than the threshold, False otherwise.
    """
    # Sort the list of numbers
    numbers.sort()
    
    # Iterate through the sorted list and compare adjacent elements
    for i in range(len(numbers) - 1):
        if abs(numbers[i] - numbers[i + 1]) < threshold:
            return True
    
    # If no such pair is found, return False
    return False


# Tester Code
class TestHasCloseElements(unittest.TestCase):

    def test_basic_functionality(self):
        # Basic scenario: Two numbers are closer than the threshold
        self.assertTrue(has_close_elements([1.0, 2.8, 3.0], 0.5))
        
        # Basic scenario: No two numbers are closer than the threshold
        self.assertFalse(has_close_elements([1.0, 2.0, 3.0], 0.5))

    def test_edge_cases(self):
        # Edge scenario: Empty list
        self.assertFalse(has_close_elements([], 0.5))
        
        # Edge scenario: List with a single element
        self.assertFalse(has_close_elements([1.0], 0.5))
        
        # Edge scenario: Threshold is zero
        with self.assertRaises(ValueError):
            has_close_elements([1.0, 2.8, 3.0], 0.0)
    
    def test_large_scale_performance(self):
        # Large scale scenario: Testing with a large list of numbers
        large_list = [i / 10 for i in range(10000)]
        self.assertTrue(has_close_elements(large_list, 0.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''

    def test_enhanced_typing_with_has_close_elements(self) -> None:
        """Test that demonstrates the enhanced typing system with the user's code."""

        # Execute the test script
        result: TestExecutionResult = self.sandbox.execute_test_script(
            self.code_string)

        # Verify we get the correct typed result
        assert isinstance(
            result, TestExecutionResult), f"Expected TestExecutionResult, got {type(result)}"

        # Test the basic properties
        assert result.execution_category == 'TESTS_FAILED'
        assert not result.success  # Should fail due to the failing test
        assert result.total_tests == 3
        assert result.tests_passed == 2
        assert result.tests_failed == 1
        assert result.tests_errors == 0

        # Verify enhanced typed access to test_details
        test_details: TestDetails = result.test_details
        assert isinstance(
            test_details, TestDetails), f"Expected TestDetails, got {type(test_details)}"

        # Test the test_details properties - these are now properly typed lists
        assert isinstance(test_details.failed_test_details, list)
        assert isinstance(test_details.passed_test_details, list)
        assert isinstance(test_details.error_test_details, list)

        # Verify we can access the individual test results
        assert len(test_details.failed_test_details) == 1
        # Now captured with verbosity=2
        assert len(test_details.passed_test_details) == 2
        assert len(test_details.error_test_details) == 0

        # Check that we can identify specific failed tests
        failed_test_names = [
            test.name for test in test_details.failed_test_details]

        # Look for the specific failing test we expect
        assert 'test_edge_cases' in failed_test_names

        # Check that passed tests are also captured
        passed_test_names = [
            test.name for test in test_details.passed_test_details]
        expected_passed_tests = [
            'test_basic_functionality',
            'test_large_scale_performance'
        ]

        for expected_test in expected_passed_tests:
            assert expected_test in passed_test_names, f"Expected {expected_test} in passed tests"

        # Detailed validation of failed test case details
        self._validate_failed_test_details(test_details)

        # Detailed validation of passed test case details
        self._validate_passed_test_details(test_details)

        # Test computed properties
        assert result.success_rate == 2/3  # 2 passed out of 3 total
        assert result.has_failures == True
        assert result.is_script_level_error == False

        print("✅ Enhanced typing test passed!")
        print(f"📊 Found {len(test_details.failed_test_details)} failed tests:")
        for test in test_details.failed_test_details:
            print(f"   • {test.name}: {test.description}")
            if test.details:
                error_lines = test.details.split('\n')
                print(f"     Error: {error_lines[-1]}")

        print(f"📈 Found {len(test_details.passed_test_details)} passed tests:")
        for test in test_details.passed_test_details:
            print(f"   • {test.name}: {test.description}")

        print(
            f"📈 Total test statistics: {result.tests_passed} passed, {result.tests_failed} failed, {result.tests_errors} errors")
        print(f"📊 Found {len(test_details.failed_test_details)} failed tests:")
        for test in test_details.failed_test_details:
            print(f"   • {test.name}")

        print(f"📈 Found {len(test_details.passed_test_details)} passed tests:")
        for test in test_details.passed_test_details:
            print(f"   • {test.name}")

        print(
            f"📈 Total test statistics: {result.tests_passed} passed, {result.tests_failed} failed, {result.tests_errors} errors")

    def _validate_failed_test_details(self, test_details: TestDetails) -> None:
        """Validate that failed test details contain expected information."""
        failed_tests = test_details.failed_test_details

        # Verify we have exactly one failed test
        assert len(
            failed_tests) == 1, f"Expected 1 failed test, got {len(failed_tests)}"

        failed_test = failed_tests[0]

        # Validate failed test structure and content
        assert failed_test.name == 'test_edge_cases', f"Expected test_edge_cases, got {failed_test.name}"
        assert failed_test.description == 'Test method: test_edge_cases', f"Unexpected description: {failed_test.description}"
        assert failed_test.details is not None, "Failed test should have error details"
        assert isinstance(
            failed_test.details, str), f"Expected string details, got {type(failed_test.details)}"

        # Validate that error details contain expected failure information
        error_details = failed_test.details.lower()
        assert 'assertionerror' in error_details or 'valueerror not raised' in error_details, \
            f"Expected assertion error details, got: {failed_test.details}"

        # Validate that the test failure is related to the expected ValueError
        assert 'valueerror' in error_details, \
            f"Expected ValueError related failure, got: {failed_test.details}"

        print(f"✅ Failed test validation passed for: {failed_test.name}")
        print(
            f"   Error details captured: {len(failed_test.details)} characters")

    def _validate_passed_test_details(self, test_details: TestDetails) -> None:
        """Validate that passed test details contain expected information."""
        passed_tests = test_details.passed_test_details

        # Verify we have exactly two passed tests
        assert len(
            passed_tests) == 2, f"Expected 2 passed tests, got {len(passed_tests)}"

        expected_passed_test_names = [
            'test_basic_functionality', 'test_large_scale_performance']
        actual_passed_test_names = [test.name for test in passed_tests]

        # Validate each passed test
        for expected_name in expected_passed_test_names:
            matching_tests = [
                test for test in passed_tests if test.name == expected_name]
            assert len(
                matching_tests) == 1, f"Expected exactly 1 test named {expected_name}, got {len(matching_tests)}"

            passed_test = matching_tests[0]

            # Validate passed test structure
            assert passed_test.name == expected_name, f"Test name mismatch: {passed_test.name}"
            assert passed_test.description == f'Test method: {expected_name}', \
                f"Unexpected description for {expected_name}: {passed_test.description}"

            # Passed tests might have details (output) or might be None
            if passed_test.details is not None:
                assert isinstance(passed_test.details, str), \
                    f"Expected string details for {expected_name}, got {type(passed_test.details)}"

            print(f"✅ Passed test validation passed for: {passed_test.name}")
            if passed_test.details:
                print(
                    f"   Test output captured: {len(passed_test.details)} characters")
            else:
                print(
                    f"   No additional output captured (as expected for passing tests)")

    def test_comprehensive_test_result_details(self) -> None:
        """Test that ensures comprehensive coverage of test result details for both passed and failed tests."""

        # Execute the test script
        result: TestExecutionResult = self.sandbox.execute_test_script(
            self.code_string)

        # Get test details
        test_details: TestDetails = result.test_details

        # Test comprehensive coverage of all test result details
        all_tests = (test_details.passed_test_details +
                     test_details.failed_test_details +
                     test_details.error_test_details)

        # Verify total test count matches
        assert len(all_tests) == result.total_tests, \
            f"Total test count mismatch: {len(all_tests)} vs {result.total_tests}"

        # Verify each test has required attributes
        for test in all_tests:
            assert hasattr(
                test, 'name'), f"Test missing name attribute: {test}"
            assert hasattr(
                test, 'description'), f"Test missing description attribute: {test}"
            assert hasattr(
                test, 'details'), f"Test missing details attribute: {test}"

            assert isinstance(
                test.name, str), f"Test name should be string, got {type(test.name)}"
            assert isinstance(
                test.description, str), f"Test description should be string, got {type(test.description)}"
            assert test.details is None or isinstance(test.details, str), \
                f"Test details should be None or string, got {type(test.details)}"

        # Verify no duplicate test names
        all_test_names = [test.name for test in all_tests]
        unique_test_names = set(all_test_names)
        assert len(all_test_names) == len(unique_test_names), \
            f"Duplicate test names found: {[name for name in all_test_names if all_test_names.count(name) > 1]}"

        # Verify test categorization is correct
        expected_test_names = {'test_basic_functionality',
                               'test_edge_cases', 'test_large_scale_performance'}
        actual_test_names = set(all_test_names)
        assert expected_test_names == actual_test_names, \
            f"Test name mismatch. Expected: {expected_test_names}, Got: {actual_test_names}"

        # Detailed validation of specific test results
        self._validate_specific_test_outcomes(test_details)

        print("✅ Comprehensive test result details validation passed!")
        print(
            f"📊 Validated {len(all_tests)} total tests with complete detail coverage")

    def _validate_specific_test_outcomes(self, test_details: TestDetails) -> None:
        """Validate specific expected outcomes for each test."""

        # Create lookup dictionaries for easy access
        passed_tests_dict = {
            test.name: test for test in test_details.passed_test_details}
        failed_tests_dict = {
            test.name: test for test in test_details.failed_test_details}
        error_tests_dict = {
            test.name: test for test in test_details.error_test_details}

        # Validate test_basic_functionality (should pass)
        assert 'test_basic_functionality' in passed_tests_dict, \
            "test_basic_functionality should be in passed tests"
        basic_test = passed_tests_dict['test_basic_functionality']
        assert 'basic_functionality' in basic_test.description.lower(), \
            f"Unexpected description for basic test: {basic_test.description}"

        # Validate test_large_scale_performance (should pass)
        assert 'test_large_scale_performance' in passed_tests_dict, \
            "test_large_scale_performance should be in passed tests"
        performance_test = passed_tests_dict['test_large_scale_performance']
        assert 'large_scale_performance' in performance_test.description.lower(), \
            f"Unexpected description for performance test: {performance_test.description}"

        # Validate test_edge_cases (should fail)
        assert 'test_edge_cases' in failed_tests_dict, \
            "test_edge_cases should be in failed tests"
        edge_test = failed_tests_dict['test_edge_cases']
        assert 'edge_cases' in edge_test.description.lower(), \
            f"Unexpected description for edge test: {edge_test.description}"

        # Verify no tests are categorized as errors (for this specific code)
        assert len(error_tests_dict) == 0, \
            f"Expected no error tests, but found: {list(error_tests_dict.keys())}"

        print("✅ Specific test outcome validation passed!")
        print(f"   - Passed tests: {list(passed_tests_dict.keys())}")
        print(f"   - Failed tests: {list(failed_tests_dict.keys())}")
        print(f"   - Error tests: {list(error_tests_dict.keys())}")
