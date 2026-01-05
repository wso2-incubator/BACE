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
from infrastructure.sandbox import SafeCodeSandbox
from infrastructure.sandbox import TestExecutionResult as SandboxResult

from ..core.interfaces import (
    ExecutionResult,
    ExecutionResults,
    IExecutionSystem,
    InteractionData,
    TestResult,
)
from ..core.population import CodePopulation, TestPopulation


def _execute_single_code(
    code_idx: int,
    code_snippet: str,
    test_class_block: str,
    sandbox: SafeCodeSandbox,
) -> tuple[int, Optional[SandboxResult]]:
    """
    Worker function to execute a single code snippet against the test suite.
    """
    try:
        # CRITICAL: Reconfigure logging in child process
        from coevolution.utils.logging import setup_logging

        setup_logging(console_level="DEBUG", file_level="TRACE")

        # Compose the complete test script
        script = cpp.composition.compose_lcb_test_script(code_snippet, test_class_block)

        # Execute in sandbox
        result: SandboxResult = sandbox.execute_test_script(script)

        logger.debug(
            f"Worker (PID {os.getpid()}): Code {code_idx} executed successfully"
        )
        return code_idx, result

    except Exception as e:
        logger.error(
            f"Worker (PID {os.getpid()}): Error evaluating code {code_idx}: {e}",
            exc_info=True,
        )
        return code_idx, None  # Return None on failure


class ExecutionSystem(IExecutionSystem):
    """
    Concrete implementation of IExecutionSystem for test execution and observation.
    """

    def __init__(
        self,
        sandbox: SafeCodeSandbox,
        enable_multiprocessing: bool = True,
        num_workers: int | None = None,
    ):
        self.sandbox = sandbox
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
        tasks = [
            (i, code.snippet, test_population.test_class_block, self.sandbox)
            for i, code in enumerate(code_population)
        ]

        if self.enable_multiprocessing and self._get_num_workers(num_codes) > 1:
            raw_results = self._execute_with_multiprocessing(tasks, num_workers)
        else:
            raw_results = self._execute_sequentially(tasks)

        # 4. THE ADAPTER LOOP: Unify the data
        for code_idx, sandbox_result in raw_results:
            # Lookup Code ID
            try:
                code_id = code_population[code_idx].id
            except Exception:
                logger.error(f"Unable to lookup code id for index {code_idx}")
                continue

            # Case A: Infrastructure Failure (Sandbox crashed/timed out)
            if sandbox_result is None:
                execution_results_dict[code_id] = ExecutionResult(
                    script_error=True,
                    test_results={},
                )
                continue  # Matrix row remains all zeros

            # Case B: Validation Failure (Result count mismatch)
            # Since sandbox results are positional lists, a missing result corrupts
            # the entire row's alignment. We must reject the whole execution.
            actual_count = len(sandbox_result.test_results)
            if actual_count != num_tests:
                logger.error(
                    f"Result Count Mismatch for Code {code_id}: "
                    f"Expected {num_tests}, got {actual_count}. "
                    f"Marking as script error to preserve alignment."
                )
                execution_results_dict[code_id] = ExecutionResult(
                    script_error=True,
                    test_results={},
                )
                continue  # Matrix row remains all zeros

            # Case C: Success
            current_test_results: dict[str, TestResult] = {}

            # Inner Loop: Iterate through the tests
            # We strictly assume sandbox_result.test_results[i] corresponds to test_population[i]
            for test_idx, sb_test_res in enumerate(sandbox_result.test_results):
                # --- 1. Fill Matrix (Index-based) ---
                # TODO: Confirm the status strings are standardized
                if getattr(sb_test_res, "status", None) == "passed":
                    observation_matrix[code_idx, test_idx] = 1

                # --- 2. Fill Dictionary (ID-based) ---
                test_id = test_population[test_idx].id

                current_test_results[test_id] = TestResult(
                    details=getattr(sb_test_res, "details", None),
                    status=getattr(sb_test_res, "status", "error"),
                )

            execution_results_dict[code_id] = ExecutionResult(
                script_error=False,
                test_results=current_test_results,
            )

        # 5. Return the Atomic Artifact
        return InteractionData(
            execution_results=execution_results_dict,
            observation_matrix=observation_matrix,
        )

    def _execute_with_multiprocessing(
        self,
        tasks: list[tuple[int, str, str, SafeCodeSandbox]],
        num_workers: int,
    ) -> list[tuple[int, Optional[SandboxResult]]]:
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
    ) -> list[tuple[int, Optional[SandboxResult]]]:
        """Execute tasks sequentially (for debugging or single-threaded mode)."""
        logger.debug("Running sequential execution (no multiprocessing)")
        results = []
        for task in tasks:
            result = _execute_single_code(*task)
            results.append(result)
        return results
