"""
Feedback generation for LLM-based code editing operations.

This module provides functions to convert test execution results into natural
language feedback suitable for consumption by LLMs in edit operations.

"""

import random

import numpy as np
from loguru import logger

# TODO: Update to use core.population after creating adapters
from common.coevolution.deprecated.population import CodePopulation, TestPopulation
from common.sandbox import TestExecutionResult, TestResult

# --- Constants ---
_MAX_ERROR_LENGTH = 300


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
        identifier = f"{test_result.name} ({test_result.description.strip()})"
    parts.append(f"{entry_num}. {identifier}")

    # 2. Code Block
    parts.append("Test code:")
    parts.append("```python")
    parts.append(test_code.strip())
    parts.append("```")

    # 3. Optional Error Details
    if include_error:
        error_info = test_result.details or "No details available"
        if len(error_info) > _MAX_ERROR_LENGTH:
            error_info = error_info[:_MAX_ERROR_LENGTH] + "..."
        parts.append(f"Error: {error_info}")

    # Add a trailing newline for separation between entries
    parts.append("")
    return "\n".join(parts)


# --- Main Feedback Generation Function ---
# TODO: Define a common interface for feedback generation functions, the reproduction module expects a specific signature


def generate_feedback_for_code(
    observation_matrix: np.ndarray,
    execution_results: dict[int, TestExecutionResult],
    test_population: TestPopulation,
    code_idx: int,
) -> str:
    """
    Generate natural language feedback from test execution results for LLM editing.

    Formats test failures, errors, and successes into a clear, actionable feedback
    string that can be used by LLM-based code editing operations to improve code.
    Includes the actual test code for all tests to give the LLM full context.

    Args:
        execution_result: TestExecutionResult for a specific code snippet
        test_population: TestPopulation containing the test cases
        code_idx: Index of the code in the population (for logging)

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

        2. test_divide (Test division operation)
        Test code:
        ```python
        def test_divide(self):
            self.assertEqual(divide(10, 2), 5)
        ```
        Error: ZeroDivisionError - division by zero

        Passed 3 test(s):

        1. test_multiply
        Test code:
        ```python
        def test_multiply(self):
            self.assertEqual(multiply(2, 3), 6)
        ```

        2. test_subtract
        Test code:
        ```python
        def test_subtract(self):
            self.assertEqual(subtract(5, 3), 2)
        ```

        3. test_modulo
        Test code:
        ```python
        def test_modulo(self):
            self.assertEqual(modulo(10, 3), 1)
        ```"

    Note:
        This feedback is designed for consumption by LLMs in the edit operation.
        It prioritizes clarity and actionability over brevity.
        The execution_result.test_results order matches test_population.individuals order.
    """
    # --- Initial Checks ---
    if code_idx not in execution_results:
        logger.debug(
            f"Code {code_idx}: No execution result found, no feedback generated"
        )
        raise KeyError(f"No execution result for code index {code_idx}")

    execution_result = execution_results[code_idx]

    if execution_result.tests_failed == 0 and execution_result.tests_errors == 0:
        logger.trace(f"Code {code_idx}: All tests passed, no feedback needed")
        return ""

    if execution_result.script_error:
        logger.debug(
            f"Code {code_idx}: Script error detected, providing syntax feedback"
        )
        return (
            "This code has syntax errors or import failures that prevented tests "
            "from running. Fix the basic code structure and syntax before "
            "addressing test failures."
        )

    logger.debug(
        f"Code {code_idx}: Generating feedback - "
        f"{execution_result.tests_failed + execution_result.tests_errors} failures, "
        f"{execution_result.tests_passed} passed out of {execution_result.total_tests} total"
    )

    # --- Process Results in a Single Pass ---
    failed_entries: list[str] = []
    passed_entries: list[str] = []

    # Use zip to iterate through results and corresponding code simultaneously
    # Assumes perfect alignment, guaranteed by Population validation
    for test_res, test_code in zip(
        execution_result.test_results, test_population.individuals
    ):
        if test_res.status in ("failed", "error"):
            failed_entries.append(
                _format_test_entry(
                    test_res,
                    test_code,
                    entry_num=len(failed_entries) + 1,
                    include_error=True,
                )
            )
            logger.trace(
                f"Code {code_idx}: Formatted failure for test '{test_res.name}' "
                f"({test_res.status})"
            )
        elif test_res.status == "passed":
            passed_entries.append(
                _format_test_entry(
                    test_res,
                    test_code,
                    entry_num=len(passed_entries) + 1,
                    include_error=False,
                )
            )

    logger.debug(
        f"Code {code_idx}: Processed {len(failed_entries)} failed and "
        f"{len(passed_entries)} passed test entries"
    )

    # --- Assemble Feedback String ---
    feedback_builder: list[str] = []

    total_failed = len(failed_entries)  # Can use this count directly now
    feedback_builder.append(
        f"This code failed {total_failed} out of {execution_result.total_tests} tests:\n"
    )
    feedback_builder.extend(failed_entries)

    if passed_entries:
        feedback_builder.append(f"\nPassed {len(passed_entries)} test(s):\n")
        feedback_builder.extend(passed_entries)

    # Use strip() to remove any potential trailing newline added by the last entry
    feedback = "\n".join(feedback_builder).strip()
    logger.debug(f"Code {code_idx}: Generated feedback ({len(feedback)} chars)")
    logger.trace(f"Code {code_idx} Feedback:\n{feedback}")
    return feedback


def generate_feedback_for_test(
    observation_matrix: np.ndarray,
    execution_result: TestExecutionResult,
    code_population: CodePopulation,
    test_idx: int,
) -> str:
    """
    Generate natural language feedback for a test case based on code execution results.

    Analyzes which code snippets passed or failed the given test and formats this
    information into a feedback string suitable for LLM-based test editing operations.

    Args:
        observation_matrix: Binary numpy array where entry [i,j] = 1 if code i
                            passed test j, else 0.
        test_idx: Index of the test in the population.
        code_population: CodePopulation containing the code snippets.

    Returns:
        A feedback string summarizing the test results for the given code snippets.
    """
    feedback_builder: list[str] = []

    # Identify which code snippets passed this test
    passed_code_indices = np.where(observation_matrix[:, test_idx] == 1)[0]
    failed_code_indices = np.where(observation_matrix[:, test_idx] == 0)[0]
    logger.debug(
        f"Test {test_idx}: Found {len(passed_code_indices)} passing code snippet(s) "
        f"out of {observation_matrix.shape[0]} total code snippets"
    )

    if len(passed_code_indices) == 0:
        logger.debug(f"Test {test_idx}: No code snippets passed this test")
        feedback_builder.append("No code snippets passed this test case")
        feedback_builder.append("The following code snippet failed the test:\n")

        code_idx = random.choice(failed_code_indices)
        code_snippet = code_population.individuals[code_idx]
        feedback_builder.append("```python")
        feedback_builder.append(code_snippet.strip())
        feedback_builder.append("```")
        feedback_builder.append("")
        feedback_builder.append("However, the above code snippet is correct.")
        feedback_builder.append(
            "This indicates that the test case is incorrect, or too hard."
        )

    else:
        feedback_builder.append("The following code snippet passed the test:\n")
        code_idx = random.choice(passed_code_indices)
        code_snippet = code_population.individuals[code_idx]

        feedback_builder.append("```python")
        feedback_builder.append(code_snippet.strip())
        feedback_builder.append("```")
        feedback_builder.append("")

        # Trace-level logging for each snippet added to feedback
        logger.trace(
            f"Test {test_idx}: Added passing snippet from code index {code_idx} "
            f"(chars={len(code_snippet)})"
        )

        feedback_builder.append(
            "However, the above code snippet is buggy and needs improvement."
        )
        feedback_builder.append(
            "The test case could not identify the bugs in this snippet."
        )

    feedback = "\n".join(feedback_builder).strip()
    logger.debug(f"Test {test_idx}: Generated feedback ({len(feedback)} chars)")
    logger.trace(f"Test {test_idx} Feedback:\n{feedback}")
    return feedback
