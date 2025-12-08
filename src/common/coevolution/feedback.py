"""
Feedback generation for LLM-based edit operations.

This module provides concrete implementations of the IFeedbackGenerator interface
for both code and test populations. These generators convert test execution results
into natural language feedback suitable for consumption by LLMs in edit operations.

The module provides two implementations:
- CodeFeedbackGenerator: Generates feedback for code snippets based on test failures
- TestFeedbackGenerator: Generates feedback for test cases based on code behavior
"""

import random

import numpy as np
from loguru import logger

from common.sandbox import TestResult

from .core.individual import CodeIndividual, TestIndividual
from .core.interfaces import BasePopulation, ExecutionResults, IFeedbackGenerator

# --- Helper Function ---


def _format_test_entry(
    test_result: TestResult,
    test_code: str,
    entry_num: int,
    include_error: bool = False,
) -> str:
    """
    Formats a single test entry (passed or failed) for the feedback string.

    Args:
        test_result: The TestResult object for the specific test.
        test_code: The corresponding code string for the test method.
        entry_num: The sequential number for the entry (e.g., 1, 2, 3).
        include_error: Whether to include the error details (for failures/errors).

    Returns:
        A formatted string for this test entry.
    """
    parts = []

    # 1. Identifier (Name + Optional Description)
    identifier = test_result.name
    if test_result.description and test_result.description.strip():
        identifier = f"{test_result.name}"
    parts.append(f"{entry_num}. {identifier}")

    # 2. Code Block
    parts.append("Test code:")
    parts.append("```python")
    parts.append(test_code.strip())
    parts.append("```")

    # 3. Optional Error Details
    if include_error:
        error_info = test_result.details or "No details available"
        parts.append(f"Error: {error_info}")

    # Add a trailing newline for separation between entries
    parts.append("")
    return "\n".join(parts)


# --- Concrete Implementations ---


class CodeFeedbackGenerator(IFeedbackGenerator[TestIndividual]):
    """
    Generates natural language feedback for code snippets based on test execution results.

    This generator analyzes test failures and successes to produce actionable feedback
    that can be used by LLM-based code editing operations to improve code quality.

    The feedback includes:
    - Test failure details with error messages
    - Code of failing tests for context
    - Summary of passing tests for reference
    """

    def generate_feedback(
        self,
        observation_matrix: np.ndarray,
        execution_results: ExecutionResults,
        other_population: BasePopulation[TestIndividual],
        individual_idx: int,
    ) -> str:
        """
        Generate natural language feedback from test execution results for LLM editing.

        Formats test failures, errors, and successes into a clear, actionable feedback
        string that can be used by LLM-based code editing operations to improve code.
        Includes the actual test code for all tests to give the LLM full context.

        Args:
            observation_matrix: Binary matrix where entry [i,j] = 1 if code i passed test j.
                               Not used directly but available for advanced analysis.
            execution_results: Dictionary mapping code index to TestExecutionResult.
            other_population: The test population (TestPopulation) containing test snippets.
            individual_idx: Index of the code snippet to generate feedback for.

        Returns:
            Formatted feedback string describing test outcomes with test code.
            Empty string if all tests passed or no useful feedback available.

        Example output:
            "This code failed 2 out of 5 tests:

            1. test_add (Test addition of two numbers)
            Test code:
            ```python
            def test_add(self):
                self.assertEqual(add(2, 3), 5)
            ```
            Error: AssertionError - Expected 5 but got 3

            Passed 3 test(s):

            1. test_multiply
            Test code:
            ```python
            def test_multiply(self):
                self.assertEqual(multiply(2, 3), 6)
            ```"

        Note:
            This feedback is designed for consumption by LLMs in the edit operation.
            It prioritizes clarity and actionability over brevity.
        """
        # --- Initial Checks ---
        if individual_idx not in execution_results:
            logger.debug(
                f"Code {individual_idx}: No execution result found, no feedback generated"
            )
            raise KeyError(f"No execution result for code index {individual_idx}")

        execution_result = execution_results[individual_idx]

        # Check for script errors first (takes precedence)
        if execution_result.script_error:
            logger.debug(
                f"Code {individual_idx}: Script error detected, providing syntax feedback"
            )
            return (
                "This code may have syntax errors that prevented tests "
                "from running. Fix the basic code structure and syntax before "
                "addressing test failures."
                "Another possibility is that the code is inefficient and caused a timeout."
                "Optimization is needed to execute within time limits."
            )

        # Then check if all tests passed
        if execution_result.tests_failed == 0 and execution_result.tests_errors == 0:
            logger.trace(f"Code {individual_idx}: All tests passed, no feedback needed")
            return ""

        logger.debug(
            f"Code {individual_idx}: Generating feedback - "
            f"{execution_result.tests_failed + execution_result.tests_errors} failures, "
            f"{execution_result.tests_passed} passed out of {execution_result.total_tests} total"
        )

        # --- Process Results in a Single Pass ---
        failed_entries: list[str] = []
        passed_entries: list[str] = []

        # Use zip to iterate through results and corresponding test code simultaneously
        # Assumes perfect alignment, guaranteed by Population validation
        for test_res, test_individual in zip(
            execution_result.test_results, other_population
        ):
            if test_res.status in ("failed", "error"):
                failed_entries.append(
                    _format_test_entry(
                        test_res,
                        test_individual.snippet,
                        entry_num=len(failed_entries) + 1,
                        include_error=True,
                    )
                )
                logger.trace(
                    f"Code {individual_idx}: Formatted failure for test '{test_res.name}' "
                    f"({test_res.status})"
                )
            elif test_res.status == "passed":
                passed_entries.append(
                    _format_test_entry(
                        test_res,
                        test_individual.snippet,
                        entry_num=len(passed_entries) + 1,
                        include_error=False,
                    )
                )

        logger.debug(
            f"Code {individual_idx}: Processed {len(failed_entries)} failed and "
            f"{len(passed_entries)} passed test entries"
        )

        # --- Assemble Feedback String ---
        feedback_builder: list[str] = []

        total_passed: int = len(passed_entries)
        feedback_builder.append(
            f"This code passed {total_passed} out of {execution_result.total_tests} tests:\n"
        )

        if len(failed_entries) > 0:
            feedback_builder.append("The following test(s) failed:\n")
            feedback_builder.extend(
                random.sample(failed_entries, k=min(len(failed_entries), 2))
            )

        if len(passed_entries) > 0:
            feedback_builder.append("\nThe following test(s) passed:\n")
            feedback_builder.extend(
                random.sample(passed_entries, k=min(len(passed_entries), 2))
            )
        # Use strip() to remove any potential trailing newline added by the last entry
        feedback = "\n".join(feedback_builder).strip()
        logger.debug(
            f"Code {individual_idx}: Generated feedback ({len(feedback)} chars)"
        )
        logger.trace(f"Code {individual_idx} Feedback:\n{feedback}")
        return feedback


class TestFeedbackGenerator(IFeedbackGenerator[CodeIndividual]):
    """
    Generates natural language feedback for test cases based on code execution results.

    This generator analyzes which code snippets passed or failed a given test and
    produces feedback to help improve test quality, particularly by identifying
    tests that are either too weak (pass buggy code) or too strict (fail correct code).

    The feedback includes:
    - Representative code snippets that passed/failed the test
    - Guidance on whether the test is too weak or too strict
    """

    def generate_feedback(
        self,
        observation_matrix: np.ndarray,
        execution_results: ExecutionResults,
        other_population: BasePopulation[CodeIndividual],
        individual_idx: int,
    ) -> str:
        """
        Generate natural language feedback for a test case based on code execution results.

        Analyzes which code snippets passed or failed the given test and formats this
        information into a feedback string suitable for LLM-based test editing operations.

        The feedback strategy:
        - If no code passed the test: Shows a failed code snippet and suggests the test
          may be too strict or incorrect (assumes the code is correct).
        - If some code passed the test: Shows a passing code snippet and suggests the
          test is too weak (assumes the passing code is buggy).

        Args:
            observation_matrix: Binary numpy array where entry [i,j] = 1 if code i
                               passed test j, else 0.
            execution_results: Dictionary mapping code index to TestExecutionResult.
                              Not used directly by this generator.
            other_population: The code population (CodePopulation) containing code snippets.
            individual_idx: Index of the test in the test population.

        Returns:
            A feedback string summarizing the test results for the given code snippets.

        Example output:
            "The following code snippet passed the test:

            ```python
            def add(a, b):
                return a - b  # Bug: should be + not -
            ```

            However, the above code snippet is buggy and needs improvement.
            The test case could not identify the bugs in this snippet."

        Note:
            This feedback helps the LLM improve test quality by either making tests
            more discriminating (catching bugs) or more appropriate (not too strict).
        """
        feedback_builder: list[str] = []

        # Identify which code snippets passed this test
        passed_code_indices = np.where(observation_matrix[:, individual_idx] == 1)[0]
        failed_code_indices = np.where(observation_matrix[:, individual_idx] == 0)[0]
        logger.debug(
            f"Test {individual_idx}: Found {len(passed_code_indices)} passing code snippet(s) "
            f"out of {observation_matrix.shape[0]} total code snippets"
        )

        if len(passed_code_indices) == 0:
            logger.debug(f"Test {individual_idx}: No code snippets passed this test")
            feedback_builder.append("No code snippets passed this test case")
            feedback_builder.append(
                "This indicates that the test case is incorrect, or too hard."
            )

        # passing code snippets
        if len(passed_code_indices) > 0:
            feedback_builder.append(
                "The following code candidate snippet(s) passed the test:\n"
            )
            code_indices = random.sample(
                list(passed_code_indices), k=min(len(passed_code_indices), 2)
            )

            for i, code_idx in enumerate(code_indices):
                code_snippet = other_population[code_idx].snippet
                feedback_builder.append(f"Passing snippet {i + 1}:")
                feedback_builder.append("```python")
                feedback_builder.append(code_snippet.strip())
                feedback_builder.append("```")
                feedback_builder.append("")

            # Trace-level logging for each snippet added to feedback
            logger.trace(
                f"Test {individual_idx}: Added passing snippet from code index {code_idx} "
                f"(chars={len(code_snippet)})"
            )

            feedback_builder.append(
                "The test case could not identify the bugs in the above snippet(s)."
            )

        # failing code snippets
        if len(failed_code_indices) > 0:
            feedback_builder.append(
                "The following code candidate snippet(s) failed the test:\n"
            )

            code_indices = random.sample(
                list(failed_code_indices), k=min(len(failed_code_indices), 2)
            )
            for i, code_idx in enumerate(code_indices):
                code_snippet = other_population[code_idx].snippet
                feedback_builder.append(f"Failing snippet {i + 1}:")
                feedback_builder.append("```python")
                feedback_builder.append(code_snippet.strip())
                feedback_builder.append("```")
                feedback_builder.append("")

                # We need to write the reason for failure using the test results
                exec_result = execution_results[code_idx]
                try:
                    test_result = exec_result.test_results[individual_idx]
                except IndexError:
                    test_result = None
                error_info = (
                    test_result.details if test_result else "No details available"
                )
                feedback_builder.append(f"Reason for failure: {error_info}")
                feedback_builder.append("")

            # Trace-level logging for each snippet added to feedback
            logger.trace(
                f"Test {individual_idx}: Added failing snippet from code index {code_idx} "
                f"(chars={len(code_snippet)})"
            )

            feedback_builder.append(
                "The test case identified the bugs in the above snippet(s)."
            )

        feedback = "\n".join(feedback_builder).strip()
        logger.debug(
            f"Test {individual_idx}: Generated feedback ({len(feedback)} chars)"
        )
        logger.trace(f"Test {individual_idx} Feedback:\n{feedback}")
        return feedback
