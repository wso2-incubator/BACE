"""
Code and test evaluation using a sandbox environment.

This module has been refactored into a two-phase process:

1. execute_code_against_tests(): Execute code against tests and return detailed results
2. generate_observation_matrix(): Transform execution results into a binary matrix

This separation allows:
- Reusing execution results for multiple purposes (observation matrix, feedback, etc.)
- Pure functions that are easier to test and reason about
- Flexible result consumption by different parts of the system

The observation matrix is used for Bayesian belief updates.
For LLM feedback generation, see feedback.py.
"""

from typing import Dict

import numpy as np
from loguru import logger

from common.code_preprocessing import builders
from common.coevolution.population import CodePopulation, TestPopulation
from common.sandbox import SafeCodeSandbox, TestExecutionResult


def execute_code_against_tests(
    code_population: CodePopulation,
    test_population: TestPopulation,
    sandbox: SafeCodeSandbox,
) -> Dict[int, TestExecutionResult]:
    """
    Execute each code snippet against all tests and collect detailed results.

    This function performs the actual code execution and returns comprehensive
    results including test outcomes, error messages, and execution details.
    The results can be used both for generating observation matrices and for
    providing feedback to LLM-based editing operations.

    Args:
        code_population: Population of code snippets to test
        test_population: Population of test cases to run
        sandbox: Safe execution environment for testing code

    Returns:
        Dictionary mapping code index to TestExecutionResult.
        Each TestExecutionResult contains:
        - test_results: List of individual test outcomes
        - tests_passed/failed/errors: Aggregate counts
        - Detailed error messages and tracebacks for failures

    Example:
        >>> results = execute_code_against_tests(codes, tests, sandbox)
        >>> results[0].test_results[0].status  # 'passed', 'failed', or 'error'
        >>> results[0].test_results[0].details  # Error message if failed
    """
    execution_results: Dict[int, TestExecutionResult] = {}

    logger.info(
        f"Executing code against tests: {code_population.size} code × "
        f"{test_population.size} tests = "
        f"{code_population.size * test_population.size} evaluations"
    )

    for code_idx, (code_snippet, _) in enumerate(code_population):
        logger.debug(f"Evaluating code snippet {code_idx + 1}/{code_population.size}")

        script = builders.build_test_script_for_lcb(
            code_snippet, test_population.test_class_block
        )
        execution_result = sandbox.execute_test_script(script)

        # Store the complete execution result
        execution_results[code_idx] = execution_result

        # Log individual test outcomes at trace level
        for test_idx, test in enumerate(execution_result.test_results):
            logger.trace(
                f"Code {code_idx}, Test {test_idx} ({test.name}): {test.status}"
            )

    logger.debug(f"Completed executing {len(execution_results)} code snippets")
    return execution_results


def generate_observation_matrix(
    execution_results: Dict[int, TestExecutionResult],
    num_codes: int,
    num_tests: int,
) -> np.ndarray:
    """
    Generate a binary observation matrix from execution results.

    Transforms detailed test execution results into a simple binary matrix
    where entry [i,j] is 1 if code i passed test j, else 0.

    Args:
        execution_results: Dictionary mapping code index to TestExecutionResult
                          (output from execute_code_against_tests)
        num_codes: Size of code population (number of rows)
        num_tests: Size of test population (number of columns)

    Returns:
        Binary numpy array with shape (num_codes, num_tests).
        Entry [i,j] = 1 if code i passed test j, else 0.
        Failed and error tests are both marked as 0.

    Example:
        >>> matrix = generate_observation_matrix(results, 5, 10)
        >>> matrix.shape  # (5, 10)
        >>> matrix[0, 3]  # 1 if code 0 passed test 3, else 0
    """
    observation_matrix = np.zeros((num_codes, num_tests), dtype=int)

    logger.info(f"Generating observation matrix: {num_codes} code × {num_tests} tests")

    for code_idx, execution_result in execution_results.items():
        for test_idx, test in enumerate(execution_result.test_results):
            if test.status == "passed":
                observation_matrix[code_idx, test_idx] = 1
            # elif test.status == "failed" or test.status == "error":
            #     observation_matrix[code_idx, test_idx] = 0  # Explicit, but default
            # Treat failures and errors as 0 (default value)

    logger.debug("Completed generating observation matrix")
    logger.trace(f"Observation Matrix:\n{observation_matrix}")

    # Log pass rates for each code (fraction of tests passed)
    if num_tests > 0:
        pass_rates = np.sum(observation_matrix, axis=1) / num_tests
        logger.trace(f"Code pass rates: {pass_rates}")

    return observation_matrix
