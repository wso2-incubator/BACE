"""
Concrete implementation of the Divergence Finder.
Uses the execution sandbox to identify differences between two code snippets.
"""

import multiprocessing
from dataclasses import dataclass, replace
from typing import Any, Optional, Union

from loguru import logger

from coevolution.core.interfaces.language import ILanguage
from infrastructure.languages import PythonLanguage
from infrastructure.sandbox import SandboxConfig, create_sandbox

from ..operators.differential_operators import DifferentialResult, IDifferentialFinder


@dataclass
class ExecutionError:
    """Internal helper to propagate worker errors safely."""

    input_idx: int
    error_message: str


# Robust return type for the worker: (index, Result | Error | None)
WorkerResult = Union[
    tuple[int, DifferentialResult], tuple[int, ExecutionError], tuple[int, None]
]


def _worker_entry(
    task_args: tuple[int, dict[str, Any], str, str, SandboxConfig, ILanguage],
) -> WorkerResult:
    """
    Stateless worker function for parallel execution.

    Unpacks arguments, hydrates a fresh sandbox, executes both snippets,
    and checks for divergence.

    Args:
        task_args: A tuple containing (idx, input_data, code_a, code_b, config)
                   Packed as a tuple to support pool.imap_unordered.
    """

    # Unpack arguments
    idx, input_data, code_a_snippet, code_b_snippet, config, language_adapter = (
        task_args
    )

    try:
        # 1. Hydrate Sandbox
        # Ensure we set up logging in the child process if needed
        # No logging for now to reduce overhead
        # setup_logging(console_level="INFO", file_level="TRACE")
        sandbox = create_sandbox(config)

        # Helper to run a single snippet
        def run_snippet(snippet: str) -> Optional[str]:
            test_input_formatted = {"inputdata": input_data}
            script = language_adapter.compose_evaluation_script(
                snippet, str(test_input_formatted)
            )
            exec_result = sandbox.execute_code(script)

            if exec_result.error:
                # We could log the error detail here or bubble it up
                return None
            return exec_result.output.strip()

        # 2. Execute Code A
        out_a = run_snippet(code_a_snippet)

        # 3. Execute Code B
        out_b = run_snippet(code_b_snippet)

        # 4. Check for Runtime Errors
        if out_a is None or out_b is None:
            # If execution failed, we treat it as an error/skip
            return idx, ExecutionError(idx, "Runtime execution failure in sandbox")

        # 5. Compare Outputs
        if out_a != out_b:
            # Create the shared artifact directly
            diff_result = DifferentialResult(
                input_data=input_data, output_a=out_a, output_b=out_b
            )
            return idx, diff_result

        return idx, None

    except Exception as e:
        return idx, ExecutionError(idx, str(e))


class DifferentialFinder(IDifferentialFinder):
    """
    Executes a generator script to produce inputs, then runs two code snippets
    against those inputs to find discrepancies in their outputs.

    Supports both sequential and parallel execution modes.
    """

    def __init__(
        self,
        sandbox_config: SandboxConfig,
        language_adapter: ILanguage,
        enable_multiprocessing: bool = True,
        cpu_workers: int = 4,
    ) -> None:
        """
        Initialize the finder with configuration.

        Args:
            sandbox_config: Configuration to create sandboxes.
            enable_multiprocessing: Whether to use parallel workers.
            cpu_workers: Number of worker processes (default 4).
        """
        self.sandbox_config = sandbox_config
        self.language_adapter = language_adapter
        self.enable_multiprocessing = enable_multiprocessing
        self.cpu_workers = cpu_workers
        self.python = PythonLanguage()

        # Create a local sandbox for generator execution (lightweight task)
        # or for sequential fallback.
        self._local_sandbox = create_sandbox(sandbox_config)

        python_config = replace(sandbox_config, language="python")
        self._python_sandbox = create_sandbox(python_config)

    def _generate_test_inputs(self, generator_script: str) -> list[dict[str, Any]]:
        """
        Executes the generator script to produce a list of test inputs.
        This step is always sequential as it's a single script execution.
        """
        if not self.python.is_syntax_valid(generator_script):
            logger.warning(
                "Input generator script produced invalid Python code. No test inputs generated."
            )
            logger.debug(f"Invalid generator script:\n{generator_script[:1000]}...")
            return []

        # Use the local sandbox instance
        output = self._python_sandbox.execute_code(generator_script).output.strip()

        # Always parse generator output as Python, since generators are always Python
        test_inputs = self.python.parse_test_inputs(output)
        if not test_inputs:
            logger.warning(
                "No test inputs generated or failed to parse generator output."
            )
            logger.debug(f"Generator output was:\n{output[:1000]}...")

        return test_inputs

    def find_differential(
        self,
        code_a_snippet: str,
        code_b_snippet: str,
        input_generator_script: str,
        limit: int = 10,
    ) -> list[DifferentialResult]:
        """
        Finds inputs where code_a and code_b produce different outputs.
        """
        found_divergences: list[DifferentialResult] = []

        # 1. Generate Inputs (Sequential)
        test_inputs = self._generate_test_inputs(input_generator_script)
        if not test_inputs:
            logger.warning("No test inputs generated; skipping differential finding.")
            return found_divergences

        # 2. Check Execution Mode
        # If disabled or strictly 1 worker, use legacy sequential path
        if not self.enable_multiprocessing or self.cpu_workers <= 1:
            return self._find_differential_sequential(
                code_a_snippet, code_b_snippet, test_inputs, limit
            )

        # 3. Parallel Execution Path
        return self._find_differential_parallel(
            code_a_snippet, code_b_snippet, test_inputs, limit
        )

    def _find_differential_sequential(
        self, code_a: str, code_b: str, inputs: list[dict[str, Any]], limit: int
    ) -> list[DifferentialResult]:
        """Legacy sequential implementation using local sandbox."""
        found: list[DifferentialResult] = []
        for idx, ti in enumerate(inputs):
            if len(found) >= limit:
                break

            # Re-use logic from old _generate_output but inline here or use helper
            # For strict correctness with new design, we use the local sandbox directly
            res_a = self._run_single_sequential(code_a, ti)
            res_b = self._run_single_sequential(code_b, ti)

            if res_a is None or res_b is None:
                continue

            if res_a != res_b:
                logger.debug(f"Discrepancy found at input {idx}!")
                found.append(DifferentialResult(ti, res_a, res_b))

        return found

    def _run_single_sequential(
        self, code: str, input_data: dict[str, Any]
    ) -> Optional[str]:
        """Helper for sequential run using self._local_sandbox."""
        test_input_formatted = {"inputdata": input_data}
        script = self.language_adapter.compose_evaluation_script(
            code, str(test_input_formatted)
        )
        exec_result = self._local_sandbox.execute_code(script)
        return None if exec_result.error else exec_result.output.strip()

    def _find_differential_parallel(
        self, code_a: str, code_b: str, inputs: list[dict[str, Any]], limit: int
    ) -> list[DifferentialResult]:
        """High-throughput parallel implementation."""
        found_divergences = []

        # Prepare Tasks
        tasks = [
            (i, inp, code_a, code_b, self.sandbox_config, self.language_adapter)
            for i, inp in enumerate(inputs)
        ]

        # Calculate optimal chunk size
        # Formula: total // (workers * multiplier)
        # e.g., 100 inputs / (4 workers * 4) = 6 items per chunk
        chunk_size = max(1, len(inputs) // (self.cpu_workers * 4))

        try:
            with multiprocessing.Pool(processes=self.cpu_workers) as pool:
                # Use imap_unordered to stream results as they finish
                result_iterator = pool.imap_unordered(
                    _worker_entry, tasks, chunksize=chunk_size
                )

                for idx, result in result_iterator:
                    if isinstance(result, DifferentialResult):
                        found_divergences.append(result)
                        logger.debug(f"Discrepancy found at input {idx} (Parallel)")

                        if len(found_divergences) >= limit:
                            logger.info(f"Limit {limit} reached via parallel search.")
                            pool.terminate()
                            break

                    elif isinstance(result, ExecutionError):
                        # We log but continue, unless error rate is catastrophic
                        # logger.trace(f"Worker execution failed for {idx}: {result.error_message}")
                        pass

        except Exception as e:
            logger.error(f"Parallel execution pool failed: {e}", exc_info=True)
            # In case of pool crash, return whatever we found so far
            return found_divergences

        return found_divergences


__all__ = ["DifferentialFinder"]
