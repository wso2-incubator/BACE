"""
Execution system for running code against tests and building observation matrices.

This module provides a unified system for:
1. Executing code populations against test populations in parallel
2. Converting execution results into observation matrices for Bayesian updates
"""

import multiprocessing
import os

import numpy as np
from loguru import logger

from coevolution.core.interfaces.language import ILanguage
from infrastructure.sandbox import SandboxConfig, create_sandbox

from ..core.interfaces import (
    EvaluationResult,
    ExecutionResults,
    IExecutionSystem,
    InteractionData,
)
from ..core.population import CodePopulation, TestPopulation


def _execute_atomic_interaction(
    code_idx: int,
    test_idx: int,
    code_snippet: str,
    test_snippet: str,
    sandbox_config: SandboxConfig,
    language_adapter: ILanguage,
) -> tuple[int, int, EvaluationResult]:
    """
    Worker function to execute a single code snippet against a single test function.

    Args:
        code_idx: Index of the code individual in the population
        test_idx: Index of the test individual in the population
        code_snippet: The code snippet to test
        test_snippet: The test function snippet
        sandbox_config: Serializable configuration for creating a fresh sandbox
        language_adapter: Adapter for composing the test script

    Returns:
        Tuple of (code_id, test_id, EvaluationResult)
    """
    try:
        # CRITICAL: Reconfigure logging in child process
        from coevolution.utils.logging import setup_logging

        setup_logging(console_level="DEBUG", file_level="TRACE")

        # Create fresh sandbox instance in this worker process using the factory
        sandbox = create_sandbox(sandbox_config)

        # Compose the complete test script using the language adapter
        script = language_adapter.compose_test_script(code_snippet, test_snippet)

        # Execute in sandbox
        result: EvaluationResult = sandbox.execute_test_script(script)

        return code_idx, test_idx, result

    except Exception as e:
        logger.error(
            f"Worker (PID {os.getpid()}): Fatal error evaluating interaction ({code_idx}, {test_idx}): {e}",
            exc_info=True,
        )
        # Return a failure result
        fail_result = EvaluationResult(
            status="error",
            error_log=f"Fatal worker error: {str(e)}",
            execution_time=0.0,
        )
        return code_idx, test_idx, fail_result


class ExecutionSystem(IExecutionSystem):
    """
    Concrete implementation of IExecutionSystem for test execution and observation.
    """

    def __init__(
        self,
        sandbox_config: "SandboxConfig",
        language_adapter: ILanguage,
        enable_multiprocessing: bool = True,
        cpu_workers: int | None = None,
    ):
        """
        Initialize the execution system with sandbox configuration.

        Args:
            sandbox_config: Configuration for creating sandbox instances
            language_adapter: Adapter for language-specific operations
            enable_multiprocessing: Whether to use multiprocessing for parallel execution
            cpu_workers: Number of worker processes (None = auto-detect from CPU count)
        """
        self.sandbox_config = sandbox_config
        self.language_adapter = language_adapter
        self.enable_multiprocessing = enable_multiprocessing
        self._cpu_workers = cpu_workers

        # Create a local sandbox instance for sequential execution
        # (multiprocessing workers will create their own)
        self._local_sandbox = create_sandbox(sandbox_config)

    def _get_num_workers(self, num_tasks: int) -> int:
        """Determine optimal number of workers based on CPU count and task count."""
        if not self.enable_multiprocessing:
            return 1

        if self._cpu_workers is not None:
            return min(self._cpu_workers, num_tasks)

        try:
            cpu_count = os.cpu_count() or 1
        except NotImplementedError:
            cpu_count = 1

        # Use at most total tasks or CPU count
        return min(cpu_count, num_tasks)

    def execute_tests(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
    ) -> InteractionData:
        """
        Execute each code snippet against all tests and build the atomic InteractionData.

        Ensures strict alignment between matrix columns and test population indices.
        """
        num_codes = code_population.size
        num_tests = test_population.size

        # 1. Pre-allocate the Matrix
        observation_matrix = np.zeros((num_codes, num_tests), dtype=int)

        # 2. Prepare the Dictionary (The Human/Log Truth)
        execution_results: ExecutionResults = ExecutionResults(results={})
        # Initialize dictionary with empty results for all code individuals
        for code in code_population:
            execution_results.results[code.id] = {}

        total_evaluations = num_codes * num_tests
        num_workers = self._get_num_workers(total_evaluations)

        logger.info(
            f"Executing code against tests: {num_codes} code × {num_tests} tests = "
            f"{total_evaluations} evaluations using {num_workers} workers."
        )

        # 3. Run the workers (atomic tasks: (code_snippet, test_snippet))
        tasks = [
            (
                i,
                j,
                code.snippet,
                test.snippet,
                self.sandbox_config,
                self.language_adapter,
            )
            for i, code in enumerate(code_population)
            for j, test in enumerate(test_population)
        ]

        if self.enable_multiprocessing and num_workers > 1:
            raw_results = self._execute_with_multiprocessing(tasks, num_workers)
        else:
            raw_results = self._execute_sequentially(tasks)

        # 4. THE ADAPTER LOOP: Unify the data
        for code_idx, test_idx, sb_test_res in raw_results:
            # Update matrix
            if sb_test_res.status == "passed":
                observation_matrix[code_idx, test_idx] = 1

            # Update results dictionary
            code_id = code_population[code_idx].id
            test_id = test_population[test_idx].id

            # We know execution_results.results[code_id] exists because we initialized it
            execution_results.results[code_id][test_id] = sb_test_res

        # 5. Return the Atomic Artifact
        return InteractionData(
            execution_results=execution_results,
            observation_matrix=observation_matrix,
        )

    def _execute_with_multiprocessing(
        self,
        tasks: list[tuple[int, int, str, str, SandboxConfig, ILanguage]],
        num_workers: int,
    ) -> list[tuple[int, int, EvaluationResult]]:
        """Execute tasks using multiprocessing pool."""
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.starmap(_execute_atomic_interaction, tasks)
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
        tasks: list[tuple[int, int, str, str, SandboxConfig, ILanguage]],
    ) -> list[tuple[int, int, EvaluationResult]]:
        """Execute tasks sequentially (for debugging or single-threaded mode)."""
        logger.debug("Running sequential execution (no multiprocessing)")
        results = []
        for task in tasks:
            result = _execute_atomic_interaction(*task)
            results.append(result)
        return results
