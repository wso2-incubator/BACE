"""
Code and test evaluation using a sandbox environment.


1. execute_code_against_tests(): Execute code against tests and return detailed results
2. generate_observation_matrix(): Transform execution results into a binary matrix
3. compute_code_pass_rates(): Compute pass rates for each code snippet
4. compute_test_pass_rates(): Compute pass rates for each test case
5. compute_test_discriminations(): Compute discrimination (entropy) for each test case

The observation matrix is used for Bayesian belief updates.
For LLM feedback generation, see feedback.py.
"""

import multiprocessing
import os
from typing import Optional

import numpy as np
from loguru import logger

from common.code_preprocessing import builders
from common.coevolution.population import CodePopulation, TestPopulation
from common.sandbox import SafeCodeSandbox, TestExecutionResult


def _execute_single_code(
    code_idx: int,
    code_snippet: str,
    test_class_block: str,
    sandbox: SafeCodeSandbox,
) -> tuple[int, Optional[TestExecutionResult]]:
    """
    Worker function to execute a single code snippet against the test suite.

    Args:
        args: A tuple containing (code_idx, code_snippet, test_class_block, sandbox)

    Returns:
        A tuple containing (code_idx, execution_result) or (code_idx, None) if execution failed.
    """
    try:
        logger.trace(f"Worker (PID {os.getpid()}): Evaluating code {code_idx}")
        script = builders.build_test_script_for_lcb(code_snippet, test_class_block)
        execution_result = sandbox.execute_test_script(script)
        logger.trace(f"Worker (PID {os.getpid()}): Finished code {code_idx}")
        return code_idx, execution_result
    except Exception as e:
        # Log the error and return None for the result part
        logger.error(
            f"Worker (PID {os.getpid()}): Error evaluating code {code_idx}: {e}",
            exc_info=True,
        )
        return code_idx, None  # Return None on failure


# --- Main Evaluation Function ---


def execute_code_against_tests(
    code_population: CodePopulation,
    test_population: TestPopulation,
    sandbox: SafeCodeSandbox,
) -> dict[int, TestExecutionResult]:
    """
    Execute each code snippet against all tests using multiprocessing and collect detailed results.
    Failed executions within workers are excluded from the results dictionary.

    Args:
        code_population: Population of code snippets to test.
        test_population: Population of test cases (provides the test_class_block).
        sandbox: Safe execution environment instance.

    Returns:
        dictionary mapping code index to TestExecutionResult for successfully executed codes.
    """
    execution_results_dict: dict[int, TestExecutionResult] = {}
    total_evaluations = code_population.size * test_population.size
    num_codes_to_run = code_population.size

    # Determine the number of workers: use CPU count, capped by population size
    try:
        cpu_count = os.cpu_count() or 1
    except NotImplementedError:
        cpu_count = 1
    num_workers = min(cpu_count, num_codes_to_run)

    logger.info(
        f"Executing code against tests: {num_codes_to_run} code x "
        f"{test_population.size} tests = {total_evaluations} evaluations "
        f"using {num_workers} workers."
    )

    # Prepare arguments for each task
    tasks = [
        (code_idx, code_snippet, test_population.test_class_block, sandbox)
        for code_idx, (code_snippet, _) in enumerate(code_population)
    ]

    results: list[tuple[int, Optional[TestExecutionResult]]] = []
    failed_indices = []
    try:
        # Use context manager for pool cleanup
        # Using 'spawn' context can improve stability on some OSes
        # context = multiprocessing.get_context("spawn")
        # with context.Pool(processes=num_workers) as pool:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # starmap applies the worker function to each item in tasks
            results = pool.starmap(_execute_single_code, tasks)

    except Exception as e:
        logger.error(
            f"Multiprocessing pool encountered a fatal error: {e}", exc_info=True
        )
        # In case of catastrophic pool failure, return empty results
        return {}
    finally:
        logger.debug("Multiprocessing pool processing complete.")

    # Convert list of results back to dictionary, handling None for failures
    for code_idx, execution_result in results:
        if execution_result is not None:
            execution_results_dict[code_idx] = execution_result
        else:
            # Keep track of indices that failed in the worker
            failed_indices.append(code_idx)
            logger.warning(
                f"Execution failed for code index {code_idx} in worker process."
            )

    successful_evals = len(execution_results_dict)
    failed_evals = len(failed_indices)

    logger.info(
        f"Completed executing {num_codes_to_run} code snippets. Successful executions: {successful_evals}, Failed executions: {failed_evals}."
    )
    # Detailed logging for failures if any occurred
    if failed_evals > 0:
        logger.warning(f"Failed execution indices: {failed_indices}")

    return execution_results_dict


def generate_observation_matrix(
    execution_results: dict[int, TestExecutionResult],
    num_codes: int,
    num_tests: int,
) -> np.ndarray:
    """
    Generate a binary observation matrix from execution results.

    Transforms detailed test execution results into a simple binary matrix
    where entry [i,j] is 1 if code i passed test j, else 0.

    Args:
        execution_results: dictionary mapping code index to TestExecutionResult
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


def compute_code_pass_rates(observation_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the pass rate for each code (row) in the observation matrix.

    Args:
        observation_matrix: Binary numpy array (codes x tests), 1 if code passed test, else 0

    Returns:
        1D numpy array of pass rates for each code (fraction of tests passed)
    """
    num_codes, num_tests = observation_matrix.shape
    logger.debug(
        f"Computing code pass rates for matrix with shape ({num_codes}, {num_tests})"
    )

    if num_tests == 0:
        logger.warning(
            f"Observation matrix has 0 tests. Returning zero pass rates for {num_codes} codes."
        )
        return np.zeros(num_codes, dtype=float)

    pass_rates = np.sum(observation_matrix, axis=1) / float(num_tests)
    logger.debug(
        f"Computed code pass rates, returning array with shape {pass_rates.shape}"
    )
    return np.asarray(pass_rates)


def compute_test_pass_rates(observation_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the pass rate for each test (column) in the observation matrix.

    Args:
        observation_matrix: Binary numpy array (codes x tests), 1 if code passed test, else 0

    Returns:
        1D numpy array of pass rates for each test (fraction of codes that passed)
    """
    num_codes, num_tests = observation_matrix.shape
    logger.debug(
        f"Computing test pass rates for matrix with shape ({num_codes}, {num_tests})"
    )

    if num_codes == 0:
        logger.warning(
            f"Observation matrix has 0 codes. Returning zero pass rates for {num_tests} tests."
        )
        return np.zeros(num_tests, dtype=float)

    pass_rates = np.sum(observation_matrix, axis=0) / float(num_codes)
    logger.debug(
        f"Computed test pass rates, returning array with shape {pass_rates.shape}"
    )
    return np.asarray(pass_rates)


def compute_test_discriminations(observation_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the discrimination (entropy) for each test case (column) given the observation matrix.

    A test that passes for 50% of codes has maximum discrimination (1.0).
    A test that passes for 0% or 100% of codes has minimum discrimination (0.0).

    Args:
        observation_matrix: Binary numpy array (codes x tests), 1 if code passed test, else 0

    Returns:
        1D numpy array of discrimination values (entropy) for each test, in [0, 1]

    Raises:
        ValueError: If any computed discrimination is not in [0, 1]
    """
    logger.debug(
        f"Computing test discriminations for matrix with shape {observation_matrix.shape}"
    )

    pass_rates = compute_test_pass_rates(observation_matrix)
    logger.trace(f"Computed pass rates: {pass_rates}")

    # Avoid log2(0) by clipping x to [eps, 1-eps]
    eps = 1e-12
    x = np.clip(pass_rates, eps, 1 - eps)
    logger.trace(f"Clipped pass rates with eps={eps}: {x}")

    entropy = -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    logger.trace(f"Computed entropy: {entropy}")

    # Normalize: entropy is in [0,1] for binary variable
    if not (np.all(entropy >= 0) and np.all(entropy <= 1)):
        msg = "Discriminations (entropy) must be in [0, 1] for all tests"
        logger.error(msg)
        raise ValueError(msg)

    logger.debug(
        f"Computed test discriminations, returning array with shape {entropy.shape}"
    )
    return np.asarray(entropy)
