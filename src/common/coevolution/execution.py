"""
Execution system for running code against tests and building observation matrices.

This module provides a unified system for:
1. Executing code populations against test populations in parallel
2. Converting execution results into observation matrices for Bayesian updates
"""

import multiprocessing
import os
from typing import Optional

import numpy as np
from loguru import logger

import common.code_preprocessing as cpp
from common.sandbox import SafeCodeSandbox, TestExecutionResult

from .core.interfaces import ExecutionResults, Sandbox
from .core.population import CodePopulation, TestPopulation


def _execute_single_code(
    code_idx: int,
    code_snippet: str,
    test_class_block: str,
    sandbox: SafeCodeSandbox,
) -> tuple[int, Optional[TestExecutionResult]]:
    """
    Worker function to execute a single code snippet against the test suite.

    This function is designed to run in a separate process via multiprocessing.

    IMPORTANT: This function runs in a child process and needs to reconfigure logging
    because multiprocessing child processes don't inherit logger handlers from the parent.
    Without this, TRACE level logs from sandbox.execute_test_script() won't be recorded.

    Args:
        code_idx: Index of the code snippet in the population
        code_snippet: Code string to test
        test_class_block: Test class block to run against the code
        sandbox: Sandbox instance for safe execution

    Returns:
        Tuple of (code_idx, TestExecutionResult or None)
    """
    try:
        # CRITICAL: Reconfigure logging in child process with proper context
        # Child processes from multiprocessing.Pool don't inherit logger handlers
        # from the parent process, so we need to set up logging again here.
        # This ensures TRACE level logs from sandbox are captured with proper context.
        from common.coevolution.logging_utils import setup_logging

        setup_logging(console_level="DEBUG", file_level="TRACE")

        # Compose the complete test script
        script = cpp.composition.compose_lcb_test_script(code_snippet, test_class_block)

        # Execute in sandbox
        result = sandbox.execute_test_script(script)

        logger.debug(
            f"Worker (PID {os.getpid()}): Code {code_idx} executed successfully"
        )
        return code_idx, result

    except Exception as e:
        # Log the error and return None for the result part
        logger.error(
            f"Worker (PID {os.getpid()}): Error evaluating code {code_idx}: {e}",
            exc_info=True,
        )
        return code_idx, None  # Return None on failure


class ExecutionSystem:
    """
    Concrete implementation of IExecutionSystem for test execution and observation.

    This class handles:
    - Parallel execution of code against tests using multiprocessing
    - Conversion of execution results into binary observation matrices
    - Pass rate and discrimination computations

    The system automatically determines optimal worker count based on CPU cores
    and handles execution failures gracefully.
    """

    def __init__(
        self, enable_multiprocessing: bool = True, num_workers: int | None = None
    ):
        """
        Initialize the execution system.

        Args:
            enable_multiprocessing: Whether to use multiprocessing for parallel execution.
                                   Set to False for debugging or sequential execution.
            num_workers: Number of worker processes. If None, uses CPU count.
                        Ignored if enable_multiprocessing is False.
        """
        self.enable_multiprocessing = enable_multiprocessing
        self._num_workers = num_workers

    def _get_num_workers(self, num_codes: int) -> int:
        """Determine optimal number of workers based on CPU count and population size."""
        if not self.enable_multiprocessing:
            return 1

        if self._num_workers is not None:
            return min(self._num_workers, num_codes)

        try:
            cpu_count = os.cpu_count() or 1
        except NotImplementedError:
            cpu_count = 1

        return min(cpu_count, num_codes)

    def _validate_number_of_execution_results(
        self,
        execution_results: ExecutionResults,
        num_codes: int,
        num_tests: int,
    ) -> None:
        """Validate the number of execution results for consistency."""

        if len(execution_results.keys()) != num_codes:
            logger.error(
                f"Expected execution results for {num_codes} codes, "
                f"got {len(execution_results.keys())}"
            )
            raise ValueError("Mismatch in number of execution results")

        for code_idx, result in execution_results.items():
            if not isinstance(result, TestExecutionResult):
                logger.error(
                    f"Execution result for code {code_idx} is not a TestExecutionResult"
                )
                raise ValueError(f"Invalid execution result type for code {code_idx}")

            if len(result.test_results) == 0:
                logger.warning(
                    f"Code {code_idx} has zero test results\n. Likely timeout or execution failure."
                )
            elif len(result.test_results) != num_tests:
                logger.error(
                    f"Code {code_idx}: Expected {num_tests} test results, "
                    f"got {len(result.test_results)}"
                )
                raise ValueError(
                    f"Mismatch in number of test results for code {code_idx}"
                )

    def execute_tests(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
        sandbox: Sandbox,
    ) -> ExecutionResults:
        """
        Execute each code snippet against all tests using multiprocessing.

        This method runs each code snippet in the population against the complete
        test suite in parallel using worker processes. Failed executions are logged
        but excluded from results.

        Args:
            code_population: Population of code snippets to test
            test_population: Population of test cases (provides test_class_block)
            sandbox: Safe execution environment instance

        Returns:
            Dictionary mapping code_idx to TestExecutionResult for successfully executed codes.
            Failed executions will not have entries in the dictionary.

        Example:
            >>> system = ExecutionSystem()
            >>> results = system.execute_tests(codes, tests, sandbox)
            >>> len(results)  # May be less than len(codes) if failures occurred
            >>> 0 in results  # Check if code at index 0 executed successfully
        """
        execution_results_dict: dict[int, TestExecutionResult] = {}
        num_codes = code_population.size
        num_tests = test_population.size
        total_evaluations = num_codes * num_tests
        num_workers = self._get_num_workers(num_codes)

        logger.info(
            f"Executing code against tests: {num_codes} code × {num_tests} tests = "
            f"{total_evaluations} evaluations using {num_workers} workers."
        )

        # Prepare tasks for each code snippet
        tasks = [
            (
                code_idx,
                code_individual.snippet,
                test_population.test_class_block,
                sandbox,
            )
            for code_idx, (code_individual) in enumerate(code_population)
        ]

        if self.enable_multiprocessing and num_workers > 1:
            results = self._execute_with_multiprocessing(tasks, num_workers)
        else:
            results = self._execute_sequentially(tasks)

        # Convert results to dictionary, filtering out failures
        failed_indices = []
        for code_idx, execution_result in results:
            if execution_result is not None:
                execution_results_dict[code_idx] = execution_result
            else:
                failed_indices.append(code_idx)
                logger.warning(
                    f"Execution failed for code index {code_idx} in worker process."
                )

        successful = len(execution_results_dict)
        failed = len(failed_indices)

        logger.info(
            f"Completed executing {num_codes} code snippets. "
            f"Execution Successful: {successful}, Execution Failed: {failed}."
        )

        if failed > 0:
            logger.warning(f"Failed execution indices: {failed_indices}")

        self._validate_number_of_execution_results(
            execution_results_dict, num_codes, num_tests
        )

        # Return dict directly to preserve code_idx mapping
        # This ensures that build_observation_matrix can correctly map results to code indices
        return execution_results_dict

    def _execute_with_multiprocessing(
        self,
        tasks: list[tuple[int, str, str, SafeCodeSandbox]],
        num_workers: int,
    ) -> list[tuple[int, Optional[TestExecutionResult]]]:
        """Execute tasks using multiprocessing pool."""
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.starmap(_execute_single_code, tasks)
            return results
        except Exception as e:
            logger.error(
                f"Multiprocessing pool encountered a fatal error: {e}", exc_info=True
            )
            return []
        finally:
            logger.debug("Multiprocessing pool processing complete.")

    def _execute_sequentially(
        self,
        tasks: list[tuple[int, str, str, SafeCodeSandbox]],
    ) -> list[tuple[int, Optional[TestExecutionResult]]]:
        """Execute tasks sequentially (for debugging or single-threaded mode)."""
        logger.debug("Running sequential execution (no multiprocessing)")
        results = []
        for task in tasks:
            result = _execute_single_code(*task)
            results.append(result)
        return results

    def build_observation_matrix(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
        execution_results: ExecutionResults,
    ) -> np.ndarray:
        """
        Build a binary observation matrix from execution results.

        Transforms detailed test execution results into a binary matrix where
        entry [i,j] is 1 if code i passed test j, else 0.

        IMPORTANT: This method relies on TestExecutionResult.test_results being
        in the same order as test methods in the test script (guaranteed by
        SafeCodeSandbox.execute_test_script). This ensures direct index mapping:
        test_results[j] corresponds to test_population[j].

        Args:
            code_population: Population of code snippets (determines #rows)
            test_population: Population of test cases (determines #columns)
            execution_results: Dict mapping code_idx to TestExecutionResult from execute_tests()

        Returns:
            Binary numpy array with shape (num_codes, num_tests).
            Entry [i,j] = 1 if code i passed test j, else 0.
            Failed/error tests are marked as 0.

        Example:
            >>> matrix = system.build_observation_matrix(codes, tests, results)
            >>> matrix.shape  # (num_codes, num_tests)
            >>> matrix[0, 3]  # 1 if code 0 passed test 3, else 0

        Note:
            If some codes failed to execute, their rows will remain all zeros.
        """
        num_codes = code_population.size
        num_tests = test_population.size
        observation_matrix = np.zeros((num_codes, num_tests), dtype=int)

        logger.info(
            f"Generating observation matrix: {num_codes} code x {num_tests} tests"
        )

        # Fill in the observation matrix based on execution results
        # Direct index mapping: test_results[j] corresponds to test_population[j]
        for code_idx, execution_result in execution_results.items():
            if code_idx >= num_codes:
                logger.warning(
                    f"Code index {code_idx} exceeds population size {num_codes}, skipping"
                )
                continue

            # Validate that we have the expected number of test results
            if len(execution_result.test_results) != num_tests:
                logger.error(
                    f"Code {code_idx}: Expected {num_tests} test results, "
                    f"got {len(execution_result.test_results)}. This indicates a problem "
                    f"with test execution or result ordering."
                )
                # Continue processing but log the issue

            # Direct index mapping: test_results[test_idx] -> matrix[code_idx, test_idx]
            for test_idx, test_result in enumerate(execution_result.test_results):
                if test_idx >= num_tests:
                    logger.warning(
                        f"Test index {test_idx} exceeds population size {num_tests}, skipping"
                    )
                    continue

                if test_result.status == "passed":
                    observation_matrix[code_idx, test_idx] = 1
                # Failures and errors default to 0

        logger.debug("Completed generating observation matrix")
        return observation_matrix
