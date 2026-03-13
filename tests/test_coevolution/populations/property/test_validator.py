"""Tests for validate_property_test — property test snippet validator.

Problem domain: sorting algorithm.

Public test cases (SORT_TESTS)
------------------------------
Three minimal cases covering the core contract of a sort:

    1.  [3, 1, 2]  →  [1, 2, 3]   — basic reordering
    2.  [5, 5, 1]  →  [1, 5, 5]   — duplicate elements
    3.  [42]       →  [42]         — single-element (boundary for ordering check)

Property snippets under test
-----------------------------
The three properties that together characterise a correct sort:

    1.  property_all_elements_present  — multiset equality (sorted input == sorted output)
    2.  property_same_length           — length is preserved
    3.  property_non_decreasing        — each element ≥ its predecessor
"""

from __future__ import annotations

import io
import textwrap
from contextlib import redirect_stdout
from typing import Any
from unittest.mock import MagicMock

from coevolution.core.interfaces.data import BasicExecutionResult, Test
from coevolution.core.interfaces.sandbox import ISandbox
from coevolution.populations.property.operators.validator import validate_property_test

# ── Shared test cases ──────────────────────────────────────────────────────────

SORT_TESTS: list[Test] = [
    Test(input='{"lst": [3, 1, 2]}', output="[1, 2, 3]"),  # basic reordering
    Test(input='{"lst": [5, 5, 1]}', output="[1, 5, 5]"),  # duplicates
    Test(input='{"lst": [42]}', output="[42]"),  # single element
]


# ── Property snippets ──────────────────────────────────────────────────────────

# 1. All elements in the input are present in the output (multiset equality).
PROPERTY_ELEMENTS_PRESENT = textwrap.dedent("""\
    def property_all_elements_present(inputdata, output):
        import json
        inp = json.loads(inputdata)
        out = json.loads(output)
        return sorted(inp["lst"]) == sorted(out)
""")

# 2. Length is unchanged.
PROPERTY_SAME_LENGTH = textwrap.dedent("""\
    def property_same_length(inputdata, output):
        import json
        inp = json.loads(inputdata)
        out = json.loads(output)
        return len(inp["lst"]) == len(out)
""")

# 3. Output is non-decreasing.
PROPERTY_NON_DECREASING = textwrap.dedent("""\
    def property_non_decreasing(inputdata, output):
        import json
        out = json.loads(output)
        return all(out[i] <= out[i + 1] for i in range(len(out) - 1))
""")

# Buggy: requires reverse-sorted output — wrong for ascending sort.
PROPERTY_REVERSE_SORTED = textwrap.dedent("""\
    def property_reverse_sorted(inputdata, output):
        import json
        out = json.loads(output)
        return out == sorted(out, reverse=True)
""")

# Buggy: unconditionally returns False.
PROPERTY_ALWAYS_FALSE = textwrap.dedent("""\
    def property_always_false(inputdata, output):
        return False
""")

# Buggy: raises at runtime.
PROPERTY_CRASHES = textwrap.dedent("""\
    def property_crashes(inputdata, output):
        raise ValueError("boom")
""")


# ── In-process sandbox ─────────────────────────────────────────────────────────


class InProcessSandbox:
    """ISandbox-compatible helper that runs code in-process via exec().

    Avoids filesystem/subprocess overhead in unit tests while producing
    authentic BasicExecutionResult values.
    """

    def execute_code(self, code: str, runtime: Any) -> BasicExecutionResult:
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, {})  # noqa: S102
        except Exception as exc:
            return BasicExecutionResult(
                success=False,
                output=buf.getvalue(),
                error=str(exc),
                execution_time=0.0,
                timeout=False,
                return_code=1,
            )
        return BasicExecutionResult(
            success=True,
            output=buf.getvalue(),
            error="",
            execution_time=0.0,
            timeout=False,
            return_code=0,
        )

    # The other ISandbox methods are not exercised by validate_property_test.
    def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def execute_test_script(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


# ── Valid sorting properties ───────────────────────────────────────────────────


class TestValidSortingProperties:
    """Each of the three correct sorting properties should pass all SORT_TESTS."""

    def test_elements_present_passes_all_cases(self) -> None:
        assert validate_property_test(
            PROPERTY_ELEMENTS_PRESENT,
            SORT_TESTS,
            InProcessSandbox(),
        )

    def test_same_length_passes_all_cases(self) -> None:
        assert validate_property_test(
            PROPERTY_SAME_LENGTH,
            SORT_TESTS,
            InProcessSandbox(),
        )

    def test_non_decreasing_passes_all_cases(self) -> None:
        assert validate_property_test(
            PROPERTY_NON_DECREASING,
            SORT_TESTS,
            InProcessSandbox(),
        )

    def test_non_decreasing_vacuously_true_for_single_element(self) -> None:
        # [42] -> [42]: range(0) is empty, so all() is trivially True.
        single = [Test(input="[42]", output="[42]")]
        assert validate_property_test(
            PROPERTY_NON_DECREASING,
            single,
            InProcessSandbox(),
        )

    def test_same_length_passes_with_duplicates(self) -> None:
        dup = [Test(input='{"lst": [5, 5, 1]}', output="[1, 5, 5]")]
        assert validate_property_test(
            PROPERTY_SAME_LENGTH,
            dup,
            InProcessSandbox(),
        )


# ── Buggy properties ───────────────────────────────────────────────────────────


class TestBuggyProperties:
    """Properties whose logic is wrong should be rejected."""

    def test_reverse_sorted_rejected(self) -> None:
        # [3,1,2]->[1,2,3]: ascending ≠ descending, so property returns False.
        assert not validate_property_test(
            PROPERTY_REVERSE_SORTED,
            SORT_TESTS,
            InProcessSandbox(),
        )

    def test_always_false_rejected(self) -> None:
        assert not validate_property_test(
            PROPERTY_ALWAYS_FALSE,
            SORT_TESTS,
            InProcessSandbox(),
        )

    def test_crashing_property_rejected(self) -> None:
        # Runtime exception → sandbox returns an error → validator returns False.
        assert not validate_property_test(
            PROPERTY_CRASHES,
            SORT_TESTS,
            InProcessSandbox(),
        )

    def test_property_requiring_at_least_two_elements_rejected(self) -> None:
        # Fails for the single-element test case [42]->[42].
        snippet = textwrap.dedent("""\
            def property_at_least_two(inputdata, output):
                import json
                return len(json.loads(output)) >= 2
        """)
        assert not validate_property_test(
            snippet,
            SORT_TESTS,
            InProcessSandbox(),
        )


# ── Short-circuit behaviour ────────────────────────────────────────────────────


class TestShortCircuit:
    """Validator should stop after the first failing test case."""

    def test_sandbox_not_called_after_first_failure(self) -> None:
        sandbox = MagicMock(spec=ISandbox)
        sandbox.execute_code.return_value = BasicExecutionResult(
            success=True,
            output="False\n",  # property returns False → first case fails
            error="",
            execution_time=0.0,
            timeout=False,
            return_code=0,
        )
        validate_property_test(PROPERTY_ALWAYS_FALSE, SORT_TESTS, sandbox)
        # Only one sandbox call should have been made despite three test cases.
        assert sandbox.execute_code.call_count == 1


# ── Edge cases ─────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Boundary and error-path behaviour of validate_property_test."""

    def test_empty_public_tests_accepted(self) -> None:
        # Nothing to reject against — any snippet (even an always-False one) passes.
        assert validate_property_test(
            PROPERTY_ALWAYS_FALSE,
            [],
            InProcessSandbox(),
        )

    def test_no_property_function_rejected(self) -> None:
        bad_snippet = "def check_output(inputdata, output):\n    return True\n"
        assert not validate_property_test(
            bad_snippet,
            SORT_TESTS,
            InProcessSandbox(),
        )

    def test_empty_snippet_rejected(self) -> None:
        assert not validate_property_test(
            "",
            SORT_TESTS,
            InProcessSandbox(),
        )

    def test_sandbox_execution_error_rejected(self) -> None:
        sandbox = MagicMock(spec=ISandbox)
        sandbox.execute_code.return_value = BasicExecutionResult(
            success=False,
            output="",
            error="Timeout",
            execution_time=5.0,
            timeout=True,
            return_code=-1,
        )
        assert not validate_property_test(PROPERTY_NON_DECREASING, SORT_TESTS, sandbox)

    def test_empty_stdout_rejected(self) -> None:
        # sandbox returns success but prints nothing — not "True".
        sandbox = MagicMock(spec=ISandbox)
        sandbox.execute_code.return_value = BasicExecutionResult(
            success=True,
            output="",
            error="",
            execution_time=0.0,
            timeout=False,
            return_code=0,
        )
        assert not validate_property_test(PROPERTY_NON_DECREASING, SORT_TESTS, sandbox)

    def test_only_last_stdout_line_is_checked(self) -> None:
        # If the snippet prints debug lines before the result, the validator
        # must still accept it as long as the *last* line is "True".
        snippet = textwrap.dedent("""\
            def property_with_debug_output(inputdata, output):
                print("debug: checking")
                return True
        """)
        assert validate_property_test(
            snippet,
            SORT_TESTS,
            InProcessSandbox(),
        )
