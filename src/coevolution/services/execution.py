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

import infrastructure.code_preprocessing as cpp
# ALIAS the sandbox result type to avoid name collision with core.interfaces.ExecutionResult
from infrastructure.sandbox import SafeCodeSandbox, SandboxConfig
from infrastructure.sandbox import TestResult as SandboxTestResult

from ..core.interfaces import (ExecutionResult, ExecutionResults,
                               IExecutionSystem, InteractionData)
from ..core.interfaces import TestResult as CoreTestResult
from ..core.population import CodePopulation, TestPopulation


def _execute_single_code(
    code_idx: int,
    code_snippet: str,
    test_snippets: list[str],
    sandbox_config: SandboxConfig,
) -> tuple[int, list[SandboxTestResult]]:
    """
    Worker function to execute a single code snippet against multiple test functions.

    Args:
        code_idx: Index of the code individual in the population
        code_snippet: The code snippet to test
        test_snippets: List of test function snippets
        sandbox_config: Serializable configuration for creating a fresh sandbox

    Returns:
        Tuple of (code_idx, list of sandbox_results)
    """
    try:
        # CRITICAL: Reconfigure logging in child process
        from coevolution.utils.logging import setup_logging

        setup_logging(console_level="DEBUG", file_level="TRACE")

        # Create fresh sandbox instance in this worker process
        sandbox = SafeCodeSandbox.from_config(sandbox_config)

        results = []
        for test_snippet in test_snippets:
            # Compose the complete test script using pytest style
            script = cpp.composition.compose_pytest_script(code_snippet, test_snippet)

            # Execute in sandbox
            result: SandboxTestResult = sandbox.execute_test_script(script)
            results.append(result)

            # OPTIMIZATION: If we hit a script-level error (e.g., SyntaxError in programmer code),
            # all subsequent tests for this code snippet will fail with the same error.
            if result.script_error:
                logger.debug(
                    f"Worker (PID {os.getpid()}): Script error detected for code {code_idx}. Skipping remaining {len(test_snippets) - len(results)} tests."
                )
                while len(results) < len(test_snippets):
                    results.append(result)
                break

        logger.debug(
            f"Worker (PID {os.getpid()}): Code {code_idx} executed all tests successfully"
        )
        return code_idx, results

    except Exception as e:
        logger.error(
            f"Worker (PID {os.getpid()}): Fatal error evaluating code {code_idx}: {e}",
            exc_info=True,
        )
        # Return empty list or partially filled list? 
        # Better to return what we have or a failure for all to keep alignment.
        return code_idx, []


class ExecutionSystem(IExecutionSystem):
    """
    Concrete implementation of IExecutionSystem for test execution and observation.
    """

    def __init__(
        self,
        sandbox_config: SandboxConfig,
        enable_multiprocessing: bool = True,
        cpu_workers: int | None = None,
    ):
        """
        Initialize the execution system with sandbox configuration.

        Args:
            sandbox_config: Configuration for creating sandbox instances
            enable_multiprocessing: Whether to use multiprocessing for parallel execution
            cpu_workers: Number of worker processes (None = auto-detect from CPU count)
        """
        self.sandbox_config = sandbox_config
        self.enable_multiprocessing = enable_multiprocessing
        self._cpu_workers = cpu_workers

        # Create a local sandbox instance for sequential execution
        # (multiprocessing workers will create their own)
        self._local_sandbox = SafeCodeSandbox.from_config(sandbox_config)

    def _get_num_workers(self, num_codes: int) -> int:
        """Determine optimal number of workers based on CPU count and population size."""
        if not self.enable_multiprocessing:
            return 1

        if self._cpu_workers is not None:
            return min(self._cpu_workers, num_codes)

        try:
            cpu_count = os.cpu_count() or 1
        except NotImplementedError:
            cpu_count = 1

        return min(cpu_count, num_codes)

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
        execution_results_dict: ExecutionResults = {}

        total_evaluations = num_codes * num_tests
        num_workers = self._get_num_workers(num_codes)

        logger.info(
            f"Executing code against tests: {num_codes} code × {num_tests} tests = "
            f"{total_evaluations} evaluations using {num_workers} workers."
        )

        # 3. Run the workers (pass index so we can map rows directly)
        test_snippets = [t.snippet for t in test_population]
        tasks = [
            (i, code.snippet, test_snippets, self.sandbox_config)
            for i, code in enumerate(code_population)
        ]

        if self.enable_multiprocessing and self._get_num_workers(num_codes) > 1:
            raw_results = self._execute_with_multiprocessing(tasks, num_workers)
        else:
            raw_results = self._execute_sequentially(tasks)

        # 4. THE ADAPTER LOOP: Unify the data
        for code_idx, sandbox_results in raw_results:
            # Lookup Code ID
            try:
                code_id = code_population[code_idx].id
            except Exception:
                logger.error(f"Unable to lookup code id for index {code_idx}")
                continue

            # Case A: Infrastructure Failure (Empty results)
            if not sandbox_results:
                execution_results_dict[code_id] = ExecutionResult(
                    test_results={},
                )
                continue  # Matrix row remains all zeros

            # Case B: Validation Failure (Result count mismatch)
            actual_count = len(sandbox_results)
            if actual_count != num_tests:
                logger.error(
                    f"Result Count Mismatch for Code {code_id}: "
                    f"Expected {num_tests}, got {actual_count}. "
                )
                execution_results_dict[code_id] = ExecutionResult(
                    test_results={},
                )
                continue  # Matrix row remains all zeros

            # Case C: Success (or partial success with failures recorded)
            current_test_results: dict[str, CoreTestResult] = {}

            # Inner Loop: Iterate through the tests
            for test_idx, sb_test_res in enumerate(sandbox_results):
                # --- 1. Fill Matrix (Index-based) ---
                if sb_test_res.status == "passed":
                    observation_matrix[code_idx, test_idx] = 1

                # --- 2. Fill Dictionary (ID-based) ---
                test_id = test_population[test_idx].id

                current_test_results[test_id] = CoreTestResult(
                    details=sb_test_res.details,
                    status=sb_test_res.status,
                    execution_time=sb_test_res.execution_time,
                    script_error=sb_test_res.script_error,
                )

            execution_results_dict[code_id] = ExecutionResult(
                test_results=current_test_results,
            )

        # 5. Return the Atomic Artifact
        return InteractionData(
            execution_results=execution_results_dict,
            observation_matrix=observation_matrix,
        )

    def _execute_with_multiprocessing(
        self,
        tasks: list[tuple[int, str, list[str], SandboxConfig]],
        num_workers: int,
    ) -> list[tuple[int, list[SandboxTestResult]]]:
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
        tasks: list[tuple[int, str, list[str], SandboxConfig]],
    ) -> list[tuple[int, list[SandboxTestResult]]]:
        """Execute tasks sequentially (for debugging or single-threaded mode)."""
        logger.debug("Running sequential execution (no multiprocessing)")
        results = []
        for task in tasks:
            result = _execute_single_code(*task)
            results.append(result)
        return results
            results.append(result)
        return results
