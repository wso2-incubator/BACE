"""Unit / regression tests for DifferentialFinder.

Regression test: DIFF-001 — output values must NOT be double-JSON-encoded.

Background
----------
``compose_evaluation_script`` always emits ``print(json.dumps(result))``, so the
raw stdout captured from the sandbox is already a JSON string, e.g.:

    "0\\n"   ← with surrounding quotes on the terminal

Before the fix, ``_run_single_sequential`` and the ``run_snippet`` worker closure
stored this raw JSON string verbatim in ``DifferentialResult.output_a/b``.
``get_test_method_from_io`` then called ``json.dumps`` on it *again*, producing a
doubly-encoded expected_output (``'"0\\\\n"'``) that no code individual could ever
match — causing an all-zero observation matrix regardless of code correctness.

The fix: ``json.loads`` the stdout at capture time so ``DifferentialResult``
holds the decoded Python value.  ``json.dumps`` in ``get_test_method_from_io``
then encodes it exactly once.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from coevolution.core.interfaces.data import SandboxConfig
from coevolution.populations.differential.finder import DifferentialFinder
from coevolution.populations.differential.types import DifferentialResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exec_result(output: str, error: str = "") -> MagicMock:
    """Return a fake sandbox execution result."""
    r: MagicMock = MagicMock()
    r.output = output
    r.error = error
    return r


def _real_sandbox_config() -> SandboxConfig:
    """Return a minimal valid SandboxConfig (no external processes involved)."""
    return SandboxConfig(timeout=10, max_memory_mb=128, language="python")


def _make_finder(local_sandbox: MagicMock) -> DifferentialFinder:
    """Build a DifferentialFinder with all heavy dependencies mocked out.

    Uses a *real* SandboxConfig so that the ``dataclasses.replace()`` call inside
    ``DifferentialFinder.__init__`` succeeds.  The actual sandbox objects are then
    replaced with our controllable mocks after construction.
    """
    composer: MagicMock = MagicMock()
    composer.compose_evaluation_script.return_value = "# fake evaluation script"

    runtime: MagicMock = MagicMock()
    runtime.get_execution_command.return_value = ["python", "eval_script.py"]

    with patch("coevolution.populations.differential.finder.create_sandbox") as mock_create:
        # First call → _local_sandbox, second call → _python_sandbox
        mock_create.side_effect = [local_sandbox, MagicMock()]
        finder = DifferentialFinder(
            sandbox_config=_real_sandbox_config(),
            parser=MagicMock(),
            composer=composer,
            runtime=runtime,
            enable_multiprocessing=False,
            cpu_workers=1,
        )

    return finder


# ---------------------------------------------------------------------------
# DIFF-001 — Regression: stdout must be json.loads-decoded before storing
# ---------------------------------------------------------------------------


class TestOutputDecoding:
    """DIFF-001: ``_run_single_sequential`` must decode the JSON-encoded stdout once."""

    def test_string_output_is_decoded(self) -> None:
        """compose_evaluation_script prints ``json.dumps("0\\n")`` → stdout is ``'"0\\n"'``.
        The finder must ``json.loads`` it → stored value is the Python string ``'0\\n'``.

        If this test fails, the double-JSON-encoding bug has regressed:
        expected_output in differential tests will be ``'"0\\\\n"'`` and every
        assertion will fail.
        """
        # Simulate what the sandbox returns: json.dumps of the actual return value
        stdout: str = json.dumps("0\n")  # → '"0\\n"'
        local_sandbox: MagicMock = MagicMock()
        local_sandbox.execute_command.return_value = _make_exec_result(stdout)

        finder = _make_finder(local_sandbox)
        result: Any = finder._run_single_sequential(
            code="class Solution:\n    def sol(self, input_str): return '0\\n'",
            input_data={"input_str": ""},
        )

        assert result == "0\n", (
            f"Expected decoded Python string '0\\n', got {result!r}.\n"
            "REGRESSION: _run_single_sequential returned raw JSON stdout.\n"
            "DifferentialFinder must json.loads() the stdout from compose_evaluation_script."
        )
        # Guard against the old broken doubly-encoded value
        assert result != '"0\\n"', (
            "Output is '\"0\\\\n\"' — the raw JSON-encoded string is being stored.\n"
            "get_test_method_from_io will json.dumps it again, making expected_output "
            "doubly-encoded and unmatchable by any code individual."
        )

    def test_integer_output_is_decoded(self) -> None:
        """Integer return values must be decoded to Python ``int``, not left as ``'3'``."""
        stdout: str = json.dumps(3)  # → '3'
        local_sandbox: MagicMock = MagicMock()
        local_sandbox.execute_command.return_value = _make_exec_result(stdout)

        finder = _make_finder(local_sandbox)
        result: Any = finder._run_single_sequential(
            code="class Solution:\n    def sol(self, x): return 3",
            input_data={"x": 0},
        )

        assert result == 3, (
            f"Expected int 3, got {result!r} (type {type(result).__name__}).\n"
            "Integer outputs from compose_evaluation_script must be json.loads-decoded."
        )
        assert not isinstance(result, str), (
            "result is still a string; should be a Python int after json.loads."
        )

    def test_list_output_is_decoded(self) -> None:
        """List return values must be decoded to Python ``list``."""
        stdout: str = json.dumps([1, 2, 3])  # → '[1, 2, 3]'
        local_sandbox: MagicMock = MagicMock()
        local_sandbox.execute_command.return_value = _make_exec_result(stdout)

        finder = _make_finder(local_sandbox)
        result: Any = finder._run_single_sequential(
            code="class Solution:\n    def sol(self): return [1,2,3]",
            input_data={},
        )

        assert result == [1, 2, 3], f"Expected list [1, 2, 3], got {result!r}."

    def test_non_json_stdout_falls_back_gracefully(self) -> None:
        """Non-JSON stdout (e.g. a legacy bare ``print``) must be returned as-is."""
        raw: str = "hello world"  # not valid JSON
        local_sandbox: MagicMock = MagicMock()
        local_sandbox.execute_command.return_value = _make_exec_result(raw)

        finder = _make_finder(local_sandbox)
        result: Any = finder._run_single_sequential(
            code="def sol(): pass", input_data={}
        )

        assert result == raw, "Non-JSON stdout should be returned verbatim as a fallback."

    def test_execution_error_returns_none(self) -> None:
        """If the sandbox reports an error, ``_run_single_sequential`` must return ``None``."""
        local_sandbox: MagicMock = MagicMock()
        local_sandbox.execute_command.return_value = _make_exec_result(
            output="", error="Traceback (most recent call last): ..."
        )

        finder = _make_finder(local_sandbox)
        result: Any = finder._run_single_sequential(
            code="def sol(): pass", input_data={}
        )

        assert result is None


# ---------------------------------------------------------------------------
# find_differential — sequential end-to-end
# ---------------------------------------------------------------------------


class TestFindDifferential:
    """End-to-end tests for the sequential ``find_differential`` path."""

    def _make_finder_with_inputs(
        self,
        inputs: list[dict[str, Any]],
        *snippet_outputs: str,
    ) -> DifferentialFinder:
        """Build a finder where:
        - The Python generator sandbox returns *inputs* as JSON.
        - The local sandbox returns *snippet_outputs* in order for each snippet run.
        """
        local_sandbox: MagicMock = MagicMock()
        local_sandbox.execute_command.side_effect = [
            _make_exec_result(o) for o in snippet_outputs
        ]

        composer: MagicMock = MagicMock()
        composer.compose_evaluation_script.return_value = "# script"
        runtime: MagicMock = MagicMock()
        runtime.get_execution_command.return_value = ["python", "x.py"]

        python_sandbox: MagicMock = MagicMock()
        python_sandbox.execute_command.return_value = _make_exec_result(
            json.dumps(inputs)
        )

        with patch("coevolution.populations.differential.finder.create_sandbox") as mock_create:
            mock_create.side_effect = [local_sandbox, python_sandbox]
            finder = DifferentialFinder(
                sandbox_config=_real_sandbox_config(),
                parser=MagicMock(),
                composer=composer,
                runtime=runtime,
                enable_multiprocessing=False,
                cpu_workers=1,
            )

        # Override parse_test_inputs so no real execution occurs for the generator
        finder.python.parser.parse_test_inputs = MagicMock(return_value=inputs)
        return finder

    def test_divergence_detected_and_outputs_are_python_values(self) -> None:
        """When code_a and code_b return different values, a ``DifferentialResult`` is
        produced whose ``output_a``/``output_b`` are decoded Python values — not JSON strings.

        This directly validates the DIFF-001 fix from the end-to-end path.
        """
        inputs: list[dict[str, Any]] = [{"x": 1}]
        out_a: str = json.dumps(1)   # compose_evaluation_script stdout for code_a
        out_b: str = json.dumps(2)   # compose_evaluation_script stdout for code_b

        finder = self._make_finder_with_inputs(inputs, out_a, out_b)
        divergences: list[DifferentialResult] = finder.find_differential(
            "def a(): pass", "def b(): pass", "# gen"
        )

        assert len(divergences) == 1
        d = divergences[0]
        assert isinstance(d, DifferentialResult)
        assert d.output_a == 1, (
            f"output_a should be int 1, got {d.output_a!r}.\n"
            "DIFF-001 regression: DifferentialResult is storing raw JSON stdout."
        )
        assert d.output_b == 2, f"output_b should be int 2, got {d.output_b!r}."

    def test_no_divergence_when_outputs_match(self) -> None:
        """``find_differential`` returns ``[]`` when both snippets produce the same output."""
        inputs: list[dict[str, Any]] = [{"x": 1}]
        same: str = json.dumps(42)

        finder = self._make_finder_with_inputs(inputs, same, same)
        divergences: list[DifferentialResult] = finder.find_differential(
            "def a(): pass", "def b(): pass", "# gen"
        )

        assert divergences == []

    def test_limit_is_respected(self) -> None:
        """``find_differential`` stops after ``limit`` divergences are collected."""
        inputs: list[dict[str, Any]] = [{"x": i} for i in range(10)]
        # Alternate outputs so every input is a divergence (1 vs 2)
        outputs: list[str] = [json.dumps(v) for _ in inputs for v in (1, 2)]

        finder = self._make_finder_with_inputs(inputs, *outputs)
        divergences: list[DifferentialResult] = finder.find_differential(
            "def a(): pass", "def b(): pass", "# gen", limit=3
        )

        assert len(divergences) <= 3

    def test_input_skipped_on_execution_error(self) -> None:
        """If ``_run_single_sequential`` returns ``None``, that input is silently skipped."""
        inputs: list[dict[str, Any]] = [{"x": 1}, {"x": 2}]

        call_counter: list[int] = [0]

        def patched_run(code: str, input_data: dict[str, Any]) -> Any:
            i = call_counter[0]
            call_counter[0] += 1
            # First call (code_a for input 0): simulate a sandbox error
            if i == 0:
                return None
            # Subsequent calls return incrementing decoded integers
            return i

        local_sandbox: MagicMock = MagicMock()
        composer: MagicMock = MagicMock()
        composer.compose_evaluation_script.return_value = "# script"
        runtime: MagicMock = MagicMock()
        runtime.get_execution_command.return_value = ["python", "x.py"]

        with patch("coevolution.populations.differential.finder.create_sandbox") as mock_create:
            mock_create.return_value = local_sandbox
            finder = DifferentialFinder(
                sandbox_config=_real_sandbox_config(),
                parser=MagicMock(),
                composer=composer,
                runtime=runtime,
                enable_multiprocessing=False,
                cpu_workers=1,
            )

        finder.python.parser.parse_test_inputs = MagicMock(return_value=inputs)
        finder._run_single_sequential = patched_run  # type: ignore[method-assign]

        divergences: list[DifferentialResult] = finder.find_differential(
            "def a(): pass", "def b(): pass", "# gen"
        )
        # input 0 is skipped (None); input 1 → out_a=1, out_b=2 → divergence
        assert len(divergences) == 1
        assert divergences[0].output_a != divergences[0].output_b
