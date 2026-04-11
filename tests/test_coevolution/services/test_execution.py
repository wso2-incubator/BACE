from typing import Any, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.services.execution import ExecutionSystem, _execute_atomic_interaction
from infrastructure.languages.python import PythonLanguage
from infrastructure.sandbox import SandboxConfig

# --- Fixtures ---


@pytest.fixture
def mock_populations() -> tuple[MagicMock, MagicMock]:
    """Creates dummy Code and Test populations for deterministic testing."""
    c1 = MagicMock()
    c1.id = "c1"
    c1.snippet = "def f(): return 1"

    c2 = MagicMock()
    c2.id = "c2"
    c2.snippet = "def f(): return 2"

    t1 = MagicMock()
    t1.id = "t1"
    t1.snippet = "def test_f(): assert True"

    t2 = MagicMock()
    t2.id = "t2"
    t2.snippet = "def test_f2(): assert True"

    t3 = MagicMock()
    t3.id = "t3"
    t3.snippet = "def test_f3(): assert True"

    code_pop = MagicMock()
    code_pop.__iter__.return_value = [c1, c2]  # 2 Codes

    # Deterministic __getitem__ implementation using the provided index
    def get_code(i: int) -> list[MagicMock]:
        return [c1, c2][i]

    code_pop.__getitem__.side_effect = get_code
    code_pop.size = 2
    code_pop.__len__.return_value = 2

    test_pop = MagicMock()

    def get_test(i: int) -> list[MagicMock]:
        return [t1, t2, t3][i]

    test_pop.__iter__.return_value = [t1, t2, t3]  # 3 Tests
    test_pop.__getitem__.side_effect = get_test
    test_pop.size = 3
    test_pop.__len__.return_value = 3

    return code_pop, test_pop


@pytest.fixture
def basic_config() -> SandboxConfig:
    return SandboxConfig(timeout=5)


# --- Tests ---


class TestExecutionSystemLogic:
    def test_matrix_alignment(
        self, mock_populations: tuple[MagicMock, MagicMock], basic_config: SandboxConfig
    ) -> None:
        """
        Verifies that the scatter-gather logic correctly places results
        into the (N_code x N_test) matrix.
        """
        code_pop, test_pop = mock_populations
        lang = PythonLanguage()
        system = ExecutionSystem(
            basic_config,
            composer=lang.composer,
            runtime=lang.runtime,
            analyzer=lang.analyzer,
            enable_multiprocessing=False,
        )

        # We mock the atomic executor to return a specific pattern
        # Let's say: Code 0 passes everything, Code 1 fails everything.
        with patch(
            "coevolution.services.execution._execute_atomic_interaction"
        ) as mock_worker:

            def side_effect(
                c_idx: int,
                t_idx: int,
                code_snippet: str,
                test_snippet: str,
                config: SandboxConfig,
                composer: Any,
                runtime: Any,
                analyzer: Any,
            ) -> tuple[int, int, EvaluationResult]:
                status: Literal["passed", "failed"] = (
                    "passed" if c_idx == 0 else "failed"
                )
                return (
                    c_idx,
                    t_idx,
                    EvaluationResult(status=status, execution_time=0.1),
                )

            mock_worker.side_effect = side_effect

            # Action
            interaction_data = system.execute_tests(code_pop, test_pop)

            # Assertions
            matrix = interaction_data.observation_matrix

            # Check Dimensions: (2 codes, 3 tests)
            assert matrix.shape == (2, 3)

            # Check Logic: Row 0 should be all 1s (Passed), Row 1 all 0s (Failed)
            assert np.array_equal(matrix[0], [1, 1, 1])
            assert np.array_equal(matrix[1], [0, 0, 0])


class TestWorkerResilience:
    def test_worker_handles_fatal_exception(self, basic_config: SandboxConfig) -> None:
        """
        Ensures that if the sandbox crashes or network fails,
        the worker catches it and returns an Error result, preventing the pool from hanging.
        """
        # We simulate a catastrophic failure in the sandbox creation
        with patch("coevolution.services.execution.create_sandbox") as mock_sandbox:
            mock_sandbox.side_effect = RuntimeError("Docker Daemon unresponsive")

            lang = PythonLanguage()
            c_idx, t_idx, result = _execute_atomic_interaction(
                0,
                0,
                "code",
                "test",
                basic_config,
                lang.composer,
                lang.runtime,
                lang.analyzer,
            )

            assert result.status == "error"
            assert "Docker Daemon unresponsive" in (result.error_log or "")


class TestExecutionCache:
    def test_second_call_with_same_populations_skips_workers(
        self, mock_populations: tuple[MagicMock, MagicMock], basic_config: SandboxConfig
    ) -> None:
        """
        After a first call that executes all 2×3=6 pairs, a second call with the
        same populations must resolve every pair from cache and never invoke the
        atomic worker again.
        """
        code_pop, test_pop = mock_populations
        lang = PythonLanguage()
        system = ExecutionSystem(
            basic_config,
            composer=lang.composer,
            runtime=lang.runtime,
            analyzer=lang.analyzer,
            enable_multiprocessing=False,
        )

        with patch(
            "coevolution.services.execution._execute_atomic_interaction"
        ) as mock_worker:
            mock_worker.return_value = (
                0,
                0,
                EvaluationResult(status="passed", execution_time=0.1),
            )

            def side_effect(
                c_idx: int, t_idx: int, *args: Any, **kwargs: Any
            ) -> tuple[int, int, EvaluationResult]:
                return (
                    c_idx,
                    t_idx,
                    EvaluationResult(status="passed", execution_time=0.1),
                )

            mock_worker.side_effect = side_effect

            # First call: all 6 pairs must be dispatched
            system.execute_tests(code_pop, test_pop)
            assert mock_worker.call_count == 6

            # Reset the iterator so the population is iterable again
            code_pop.__iter__.return_value = (
                list(code_pop.__iter__.return_value)
                if not isinstance(code_pop.__iter__.return_value, list)
                else code_pop.__iter__.return_value
            )

            # Second call: all pairs are cached — worker must not be called again
            system.execute_tests(code_pop, test_pop)
            assert mock_worker.call_count == 6  # unchanged

    def test_partial_miss_dispatches_only_new_pairs(
        self, mock_populations: tuple[MagicMock, MagicMock], basic_config: SandboxConfig
    ) -> None:
        """
        After warming the cache with 2 code individuals, introducing a third code
        individual (c3) should result in exactly 3 new worker calls (c3×t1, c3×t2,
        c3×t3) and the existing 6 pairs served from cache.
        """
        code_pop, test_pop = mock_populations

        # Build a third code individual with a fresh ID not present in the cache
        c3 = MagicMock()
        c3.id = "c3"
        c3.snippet = "def f(): return 3"

        # c1/c2 share the same IDs as the fixture population so their pairs are already cached
        c1 = MagicMock()
        c1.id = "c1"
        c1.snippet = "def f(): return 1"
        c2 = MagicMock()
        c2.id = "c2"
        c2.snippet = "def f(): return 2"

        code_pop_extended = MagicMock()

        code_pop_extended.__iter__.return_value = [c1, c2, c3]
        code_pop_extended.__getitem__ = MagicMock(side_effect=lambda i: [c1, c2, c3][i])
        code_pop_extended.size = 3

        lang = PythonLanguage()
        system = ExecutionSystem(
            basic_config,
            composer=lang.composer,
            runtime=lang.runtime,
            analyzer=lang.analyzer,
            enable_multiprocessing=False,
        )

        with patch(
            "coevolution.services.execution._execute_atomic_interaction"
        ) as mock_worker:

            def side_effect(
                c_idx: int, t_idx: int, *args: Any, **kwargs: Any
            ) -> tuple[int, int, EvaluationResult]:
                return (
                    c_idx,
                    t_idx,
                    EvaluationResult(status="passed", execution_time=0.1),
                )

            mock_worker.side_effect = side_effect

            # First call: warm cache with c1, c2 vs t1, t2, t3 (6 pairs)
            system.execute_tests(code_pop, test_pop)
            assert mock_worker.call_count == 6

            # Refresh iterator for extended population
            code_pop_extended.__iter__.return_value = [c1, c2, c3]

            # Second call with extended population: only c3's 3 pairs are new
            system.execute_tests(code_pop_extended, test_pop)
            assert mock_worker.call_count == 9  # 6 cached + 3 new

    def test_cache_returns_consistent_results(
        self, mock_populations: tuple[MagicMock, MagicMock], basic_config: SandboxConfig
    ) -> None:
        """
        The observation_matrix and execution_results returned on the second (cached)
        call must be identical in values to the first call.
        """
        code_pop, test_pop = mock_populations
        lang = PythonLanguage()
        system = ExecutionSystem(
            basic_config,
            composer=lang.composer,
            runtime=lang.runtime,
            analyzer=lang.analyzer,
            enable_multiprocessing=False,
        )

        with patch(
            "coevolution.services.execution._execute_atomic_interaction"
        ) as mock_worker:

            def side_effect(
                c_idx: int, t_idx: int, *args: Any, **kwargs: Any
            ) -> tuple[int, int, EvaluationResult]:
                # Code 0 always passes, code 1 always fails
                status: Literal["passed"] | Literal["failed"] = (
                    "passed" if c_idx == 0 else "failed"
                )
                return (
                    c_idx,
                    t_idx,
                    EvaluationResult(status=status, execution_time=0.1),
                )

            mock_worker.side_effect = side_effect

            first = system.execute_tests(code_pop, test_pop)
            second = system.execute_tests(code_pop, test_pop)

        assert np.array_equal(first.observation_matrix, second.observation_matrix)
        for code_id in first.execution_results.keys():
            for test_id, result in first.execution_results[code_id].items():
                assert (
                    second.execution_results[code_id][test_id].status == result.status
                )


@pytest.mark.integration
class TestIntegrationConcurrency:
    def test_multiprocessing_serialization(
        self, mock_populations: tuple[MagicMock, MagicMock], basic_config: SandboxConfig
    ) -> None:
        """
        Verifies that the system actually works with multiple processes.
        This catches pickling errors which are common with MagicMocks or complex configs.
        """
        code_pop, test_pop = mock_populations

        # NOTE: For this test to run in a real suite, we need the SandboxConfig
        # to be pickleable. We verify that here.
        lang = PythonLanguage()
        system = ExecutionSystem(
            basic_config,
            composer=lang.composer,
            runtime=lang.runtime,
            analyzer=lang.analyzer,
            enable_multiprocessing=True,
            cpu_workers=2,
        )

        # We patch the sandbox inside the worker process to avoid needing real Docker
        # This requires 'spawn' or 'fork' safety depending on OS, handled by MP.
        with patch(
            "infrastructure.sandbox.SafeCodeSandbox.from_config"
        ) as mock_from_config:
            mock_instance = MagicMock()
            mock_from_config.return_value = mock_instance
            mock_instance.execute_test_script.return_value = EvaluationResult(
                status="passed", execution_time=0.1
            )

            # Action
            data = system.execute_tests(code_pop, test_pop)

            # If we get here without a PickleError, the IPC is working.
            assert data.observation_matrix.sum() == 6  # 2 * 3 all passing
            # If we get here without a PickleError, the IPC is working.
            assert data.observation_matrix.sum() == 6  # 2 * 3 all passing
