"""Property test population — execution system.

PropertyTestEvaluator implements IExecutionSystem for property-based tests.
Unlike the standard ExecutionSystem (which runs pytest-style assertions), it:

  Phase A — runs the LLM-generated input-generator script once to obtain a list
             of raw input strings.

  Phase B — executes each code individual against every input in the target
             language to collect (inputdata → actual_output) IOPairs.

  Phase C — evaluates each property test against the collected IOPairs in a
             Python sandbox and builds the observation matrix.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import replace
from typing import Optional

import numpy as np
from loguru import logger

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    EvaluationResult,
    ExecutionResults,
    IExecutionSystem,
    InteractionData,
)
from coevolution.core.interfaces.language import ILanguageRuntime, IScriptComposer
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.utils.logging import setup_logging
from infrastructure.languages import PythonLanguage
from infrastructure.sandbox import SandboxConfig, create_sandbox

from .codegen import compose_property_test_script
from .types import IOPair, IOPairCache

# ── Module-level pure worker functions ───────────────────────────────────────
# These must be module-level (not methods) for multiprocessing pickling.
# IMPORTANT: workers are deliberately pure — they never mutate shared state.
# The parent process is solely responsible for calling io_pair_cache.store()
# and updating _eval_cache after collecting results.


def _io_generation_worker(
    args: tuple[str, str, str, SandboxConfig, IScriptComposer, ILanguageRuntime],
) -> tuple[str, str, Optional[str]]:
    """Execute one code snippet against one input in the target-language sandbox.

    Args:
        args: (code_id, code_snippet, inputdata, sandbox_config, composer, runtime)

    Returns:
        (code_id, inputdata, actual_output) — actual_output is None on error.
    """
    code_id, code_snippet, inputdata, sandbox_config, composer, runtime = args
    try:
        setup_logging()
        sandbox = create_sandbox(sandbox_config)
        script = composer.compose_evaluation_script(code_snippet, inputdata)

        result = sandbox.execute_code(script, runtime)

        if result.error:
            return code_id, inputdata, None
        return code_id, inputdata, result.output.strip()

    except Exception as e:
        logger.debug(f"_io_generation_worker error for code {code_id}: {e}")
        return code_id, inputdata, None


def _property_eval_worker(
    args: tuple[str, str, str, list[IOPair], SandboxConfig],
) -> tuple[str, str, EvaluationResult]:
    """Evaluate one property test against all IOPairs for one code individual.

    Args:
        args: (code_id, test_id, property_snippet, pairs, python_sandbox_config)

    Returns:
        (code_id, test_id, EvaluationResult)
    """
    code_id, test_id, property_snippet, pairs, python_sandbox_config = args
    try:
        setup_logging()
        sandbox = create_sandbox(python_sandbox_config)
        python = PythonLanguage()

        failures: list[dict[str, str]] = []

        for pair in pairs:
            try:
                script = compose_property_test_script(
                    property_snippet, pair["inputdata"], pair["output"]
                )
            except Exception as exc:
                failures.append(
                    {
                        "inputdata": pair["inputdata"],
                        "actual_output": pair["output"],
                        "result": f"error: {exc}",
                    }
                )
                continue

            exec_result = sandbox.execute_code(script, python.runtime)

            if exec_result.error:
                failures.append(
                    {
                        "inputdata": pair["inputdata"],
                        "actual_output": pair["output"],
                        "result": f"error: {exec_result.error}",
                    }
                )
                continue

            stdout = exec_result.output.strip()
            result_line = stdout.splitlines()[-1] if stdout else ""
            if result_line != "True":
                failures.append(
                    {
                        "inputdata": pair["inputdata"],
                        "actual_output": pair["output"],
                        "result": result_line if result_line else "False",
                    }
                )

        if not failures:
            return code_id, test_id, EvaluationResult(status="passed")

        error_log = _format_failure_log(failures)
        return code_id, test_id, EvaluationResult(status="failed", error_log=error_log)

    except Exception as e:
        logger.error(
            f"_property_eval_worker fatal error ({code_id}, {test_id}): {e}",
            exc_info=True,
        )
        return (
            code_id,
            test_id,
            EvaluationResult(
                status="error",
                error_log=f"Property check execution error: {e}",
            ),
        )


def _format_failure_log(failures: list[dict[str, str]]) -> str:
    """Build a structured human-readable failure report."""
    separator = "\u2500" * 58
    lines = [f"PROPERTY CHECK FAILURES\n{separator}"]
    for i, f in enumerate(failures, start=1):
        lines.append(
            f"[{i}] inputdata : {f['inputdata']}\n"
            f"    output    : {f['actual_output']}\n"
            f"    result    : {f['result']}"
        )
    return "\n\n".join(lines)


# ── PropertyTestEvaluator ─────────────────────────────────────────────────────


class PropertyTestEvaluator(IExecutionSystem):
    """IExecutionSystem implementation for property-based test populations.

    Holds a shared ``IOPairCache`` (created once in the profile factory) that
    bridges the initializer (which writes the generator script) and the
    evaluator (which reads it to produce test inputs and IOPairs).
    """

    def __init__(
        self,
        io_pair_cache: IOPairCache,
        sandbox_config: SandboxConfig,
        composer: IScriptComposer,
        runtime: ILanguageRuntime,
        enable_multiprocessing: bool = True,
        cpu_workers: int = 4,
        num_inputs: int = 20,
    ) -> None:
        self.io_pair_cache = io_pair_cache
        self.sandbox_config = sandbox_config
        self.composer = composer
        self.runtime = runtime
        self.enable_multiprocessing = enable_multiprocessing
        self.cpu_workers = cpu_workers
        self.num_inputs = num_inputs

        self._python_lang = PythonLanguage()
        self._python_sandbox_config: SandboxConfig = replace(
            sandbox_config, language="python"
        )

        # Sequential-path sandbox instances (workers create their own)
        self._target_sandbox = create_sandbox(sandbox_config)
        self._python_sandbox = create_sandbox(self._python_sandbox_config)

        # Result cache keyed on (code_id, test_id) — same pattern as ExecutionSystem
        self._eval_cache: dict[tuple[str, str], EvaluationResult] = {}

    # ── IExecutionSystem ──────────────────────────────────────────────────────

    def execute_tests(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
    ) -> InteractionData:
        num_codes = code_population.size
        num_tests = test_population.size

        observation_matrix = np.zeros((num_codes, num_tests), dtype=int)
        execution_results = ExecutionResults(results={})
        for code in code_population:
            execution_results.results[code.id] = {}

        if num_codes == 0 or num_tests == 0:
            return InteractionData(
                execution_results=execution_results,
                observation_matrix=observation_matrix,
            )

        # Phase A — resolve input list (once per problem lifecycle)
        inputs = self._resolve_inputs()

        if not inputs:
            logger.warning(
                "PropertyTestEvaluator: no inputs available — "
                "all property tests will be skipped (status='error')."
            )
            for ci, code in enumerate(code_population):
                for ti, test in enumerate(test_population):
                    result = EvaluationResult(
                        status="error",
                        error_log="No inputs available from generator script.",
                    )
                    execution_results.results[code.id][test.id] = result
                    observation_matrix[ci, ti] = 0
            return InteractionData(
                execution_results=execution_results,
                observation_matrix=observation_matrix,
            )

        # Phase B — populate IOPairs for code individuals not yet cached
        self._populate_io_pairs(code_population, inputs)

        # Phase C — evaluate property tests
        self._evaluate_property_tests(
            code_population, test_population, execution_results, observation_matrix
        )

        return InteractionData(
            execution_results=execution_results,
            observation_matrix=observation_matrix,
        )

    # ── Phase A ───────────────────────────────────────────────────────────────

    def _resolve_inputs(self) -> list[str]:
        """Return the cached input list, generating it on the first call."""
        cached = self.io_pair_cache.get_generated_inputs()
        if cached:
            return cached

        generator_script = self.io_pair_cache.get_generator_script()
        if generator_script is None:
            logger.warning(
                "PropertyTestEvaluator: no generator script in cache — "
                "skipping IO generation."
            )
            return []

        inputs = self._run_generator_script(generator_script)
        if inputs:
            self.io_pair_cache.store_generated_inputs(inputs)
        return inputs

    def _run_generator_script(self, generator_code: str) -> list[str]:
        """Compose and run the generator script in the Python sandbox.

        The cache stores the raw ``generate_inputs`` function body (no top-level
        call).  ``remove_main_block`` strips any ``if __name__ == '__main__':``
        guard the LLM may have included, then the framework appends a controlled
        call with ``self.num_inputs`` — matching the differential generator pattern.
        """
        clean_code = self._python_lang.parser.remove_main_block(generator_code)
        runnable = (
            clean_code
            + f"\n\nfor _inp in generate_inputs({self.num_inputs}):\n    print(_inp)\n"
        )
        try:
            result = self._python_sandbox.execute_code(
                runnable, self._python_lang.runtime
            )

            if result.error:
                logger.warning(
                    f"PropertyTestEvaluator: generator script error: {result.error}"
                )
                return []

            lines = [ln for ln in result.output.splitlines() if ln.strip()]
            logger.debug(
                f"PropertyTestEvaluator: generator produced {len(lines)} inputs."
            )
            return lines
        except Exception as e:
            logger.error(f"PropertyTestEvaluator: generator script raised: {e}")
            return []

    # ── Phase B ───────────────────────────────────────────────────────────────

    def _populate_io_pairs(
        self, code_population: CodePopulation, inputs: list[str]
    ) -> None:
        """Populate IOPairs for any code individual not yet in the cache."""
        new_codes = [c for c in code_population if not self.io_pair_cache.has(c.id)]
        if not new_codes:
            return

        use_mp = (
            self.enable_multiprocessing
            and self.cpu_workers > 1
            and len(new_codes) * len(inputs) > 1
        )

        if use_mp:
            self._populate_io_pairs_parallel(new_codes, inputs)
        else:
            self._populate_io_pairs_sequential(new_codes, inputs)

    def _populate_io_pairs_sequential(
        self, new_codes: list[CodeIndividual], inputs: list[str]
    ) -> None:
        for code in new_codes:
            pairs: list[IOPair] = []
            for inputdata in inputs:
                _, _, actual_output = _io_generation_worker(
                    (
                        code.id,
                        code.snippet,
                        inputdata,
                        self.sandbox_config,
                        self.composer,
                        self.runtime,
                    )
                )
                if actual_output is not None:
                    pairs.append(IOPair(inputdata=inputdata, output=actual_output))
            self.io_pair_cache.store(code.id, pairs)

    def _populate_io_pairs_parallel(
        self, new_codes: list[CodeIndividual], inputs: list[str]
    ) -> None:
        tasks = [
            (
                code.id,
                code.snippet,
                inputdata,
                self.sandbox_config,
                self.composer,
                self.runtime,
            )
            for code in new_codes
            for inputdata in inputs
        ]
        chunk_size = max(1, len(tasks) // (self.cpu_workers * 4))

        # Collect results per code_id before writing to cache
        raw: dict[str, list[IOPair]] = {c.id: [] for c in new_codes}

        pool = multiprocessing.Pool(processes=self.cpu_workers)
        try:
            for code_id, inputdata, actual_output in pool.imap_unordered(
                _io_generation_worker, tasks, chunksize=chunk_size
            ):
                if actual_output is not None:
                    raw[code_id].append(
                        IOPair(inputdata=inputdata, output=actual_output)
                    )
        except Exception as e:
            logger.error(f"PropertyTestEvaluator Phase B parallel error: {e}")
        finally:
            pool.terminate()
            pool.join()

        # Parent is the sole writer — safe after pool is joined
        for code_id, pairs in raw.items():
            self.io_pair_cache.store(code_id, pairs)

    # ── Phase C ───────────────────────────────────────────────────────────────

    def _evaluate_property_tests(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
        execution_results: ExecutionResults,
        observation_matrix: np.ndarray,
    ) -> None:
        """Evaluate all (code, property_test) pairs not already in _eval_cache."""
        # Build task list for cache misses only
        tasks: list[tuple[int, int, str, str, str, list[IOPair], SandboxConfig]] = []
        for ci, code in enumerate(code_population):
            for ti, test in enumerate(test_population):
                key = (code.id, test.id)
                if key in self._eval_cache:
                    result = self._eval_cache[key]
                    execution_results.results[code.id][test.id] = result
                    observation_matrix[ci, ti] = 1 if result.status == "passed" else 0
                else:
                    pairs = self.io_pair_cache.get(code.id)
                    tasks.append(
                        (
                            ci,
                            ti,
                            code.id,
                            test.id,
                            test.snippet,
                            pairs,
                            self._python_sandbox_config,
                        )
                    )

        if not tasks:
            return

        use_mp = self.enable_multiprocessing and self.cpu_workers > 1 and len(tasks) > 1

        if use_mp:
            results = self._eval_parallel(tasks)
        else:
            results = self._eval_sequential(tasks)

        for ci, ti, code_id, test_id, result in results:
            self._eval_cache[(code_id, test_id)] = result
            execution_results.results[code_id][test_id] = result
            observation_matrix[ci, ti] = 1 if result.status == "passed" else 0

    def _eval_sequential(
        self, tasks: list[tuple[int, int, str, str, str, list[IOPair], SandboxConfig]]
    ) -> list[tuple[int, int, str, str, EvaluationResult]]:
        results = []
        for ci, ti, code_id, test_id, snippet, pairs, py_config in tasks:
            _, _, result = _property_eval_worker(
                (code_id, test_id, snippet, pairs, py_config)
            )
            results.append((ci, ti, code_id, test_id, result))
        return results

    def _eval_parallel(
        self, tasks: list[tuple[int, int, str, str, str, list[IOPair], SandboxConfig]]
    ) -> list[tuple[int, int, str, str, EvaluationResult]]:
        # Build index for (code_id, test_id) → (ci, ti)
        index: dict[tuple[str, str], tuple[int, int]] = {
            (t[2], t[3]): (t[0], t[1]) for t in tasks
        }
        worker_tasks = [(t[2], t[3], t[4], t[5], t[6]) for t in tasks]
        chunk_size = max(1, len(worker_tasks) // (self.cpu_workers * 4))

        results: list[tuple[int, int, str, str, EvaluationResult]] = []

        pool = multiprocessing.Pool(processes=self.cpu_workers)
        try:
            for code_id, test_id, result in pool.imap_unordered(
                _property_eval_worker, worker_tasks, chunksize=chunk_size
            ):
                ci, ti = index[(code_id, test_id)]
                results.append((ci, ti, code_id, test_id, result))
        except Exception as e:
            logger.error(f"PropertyTestEvaluator Phase C parallel error: {e}")
        finally:
            pool.terminate()
            pool.join()

        return results


__all__ = ["PropertyTestEvaluator"]
