"""DifferentialFinder — concrete IDifferentialFinder using the execution sandbox.

Moved from strategies/breeding/differential_finder.py to this
population-centric location.
"""

import json
import multiprocessing
from dataclasses import dataclass, replace
from typing import Any, Optional, Union

from loguru import logger

from coevolution.core.interfaces.language import (
    ICodeParser,
    ILanguageRuntime,
    IScriptComposer,
)
from infrastructure.languages import PythonLanguage
from infrastructure.sandbox import SandboxConfig, create_sandbox

from .types import DifferentialResult, IDifferentialFinder


@dataclass
class ExecutionError:
    """Internal helper to propagate worker errors safely."""

    input_idx: int
    error_message: str


WorkerResult = Union[
    tuple[int, DifferentialResult], tuple[int, ExecutionError], tuple[int, None]
]


def _worker_entry(
    task_args: tuple[
        int, dict[str, Any], str, str, SandboxConfig, IScriptComposer, ILanguageRuntime
    ],
) -> WorkerResult:
    """Stateless worker function for parallel execution."""
    idx, input_data, code_a_snippet, code_b_snippet, config, composer, runtime = (
        task_args
    )

    try:
        from coevolution.utils.logging import setup_logging

        setup_logging()

        sandbox = create_sandbox(config)

        def run_snippet(snippet: str) -> Optional[str]:
            test_input_formatted = {"inputdata": input_data}
            script = composer.compose_evaluation_script(
                snippet, json.dumps(test_input_formatted)
            )
            import os
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                file_ext = ".bal" if hasattr(runtime, "bal_executable") else ".py"
                script_path = os.path.join(tmpdir, f"eval_script{file_ext}")
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script)

                cmd = runtime.get_execution_command(script_path)
                exec_result = sandbox.execute_command(cmd, cwd=tmpdir)
                if exec_result.error:
                    return None
                return exec_result.output.strip()

        out_a = run_snippet(code_a_snippet)
        out_b = run_snippet(code_b_snippet)

        if out_a is None or out_b is None:
            return idx, ExecutionError(idx, "Runtime execution failure in sandbox")

        if out_a != out_b:
            return idx, DifferentialResult(
                input_data=input_data, output_a=out_a, output_b=out_b
            )

        return idx, None

    except Exception as e:
        return idx, ExecutionError(idx, str(e))


class DifferentialFinder(IDifferentialFinder):
    """Executes a generator script to produce inputs, then runs two code snippets
    against those inputs to find discrepancies in their outputs.

    Supports both sequential and parallel execution modes.
    """

    def __init__(
        self,
        sandbox_config: SandboxConfig,
        parser: ICodeParser,
        composer: IScriptComposer,
        runtime: ILanguageRuntime,
        enable_multiprocessing: bool = True,
        cpu_workers: int = 4,
    ) -> None:
        self.sandbox_config = sandbox_config
        self.parser = parser
        self.composer = composer
        self.runtime = runtime
        self.enable_multiprocessing = enable_multiprocessing
        self.cpu_workers = cpu_workers
        self.python = PythonLanguage()

        self._local_sandbox = create_sandbox(sandbox_config)
        python_config = replace(sandbox_config, language="python")
        self._python_sandbox = create_sandbox(python_config)

    def _generate_test_inputs(self, generator_script: str) -> list[dict[str, Any]]:
        if not self.python.parser.is_syntax_valid(generator_script):
            logger.warning("Input generator script produced invalid Python code.")
            return []

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "generator.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(generator_script)

            cmd = self.python.runtime.get_execution_command(script_path)
            exec_result = self._python_sandbox.execute_command(cmd, cwd=tmpdir)
            output = exec_result.output.strip()

        test_inputs = self.python.parser.parse_test_inputs(output)
        if not test_inputs:
            logger.warning(
                "No test inputs generated or failed to parse generator output."
            )
        return test_inputs

    def find_differential(
        self,
        code_a_snippet: str,
        code_b_snippet: str,
        input_generator_script: str,
        limit: int = 10,
    ) -> list[DifferentialResult]:
        found_divergences: list[DifferentialResult] = []

        test_inputs = self._generate_test_inputs(input_generator_script)
        if not test_inputs:
            return found_divergences

        if not self.enable_multiprocessing or self.cpu_workers <= 1:
            return self._find_differential_sequential(
                code_a_snippet, code_b_snippet, test_inputs, limit
            )

        return self._find_differential_parallel(
            code_a_snippet, code_b_snippet, test_inputs, limit
        )

    def _find_differential_sequential(
        self, code_a: str, code_b: str, inputs: list[dict[str, Any]], limit: int
    ) -> list[DifferentialResult]:
        found: list[DifferentialResult] = []
        for idx, ti in enumerate(inputs):
            if len(found) >= limit:
                break
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
        test_input_formatted = {"inputdata": input_data}
        script = self.composer.compose_evaluation_script(
            code, json.dumps(test_input_formatted)
        )
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            file_ext = ".bal" if hasattr(self.runtime, "bal_executable") else ".py"
            script_path = os.path.join(tmpdir, f"eval_script{file_ext}")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)

            cmd = self.runtime.get_execution_command(script_path)
            exec_result = self._local_sandbox.execute_command(cmd, cwd=tmpdir)
            return None if exec_result.error else exec_result.output.strip()

    def _find_differential_parallel(
        self, code_a: str, code_b: str, inputs: list[dict[str, Any]], limit: int
    ) -> list[DifferentialResult]:
        found_divergences = []
        tasks = [
            (i, inp, code_a, code_b, self.sandbox_config, self.composer, self.runtime)
            for i, inp in enumerate(inputs)
        ]
        chunk_size = max(1, len(inputs) // (self.cpu_workers * 4))

        # Use an explicit pool reference so we can always call join() — even on
        # the early-exit path — to ensure workers are reaped and OS semaphores
        # are released synchronously.
        pool = multiprocessing.Pool(processes=self.cpu_workers)
        try:
            result_iterator = pool.imap_unordered(
                _worker_entry, tasks, chunksize=chunk_size
            )
            for idx, result in result_iterator:
                if isinstance(result, DifferentialResult):
                    found_divergences.append(result)
                    logger.debug(f"Discrepancy found at input {idx} (Parallel)")
                    if len(found_divergences) >= limit:
                        pool.terminate()
                        break
                elif isinstance(result, ExecutionError):
                    pass
        except Exception as e:
            logger.error(f"Parallel execution pool failed: {e}", exc_info=True)
            return found_divergences
        finally:
            # terminate() is idempotent; calling it here covers the normal-exit
            # path where __exit__ was previously the only termination point.
            pool.terminate()
            pool.join()

        return found_divergences


__all__ = ["DifferentialFinder"]
