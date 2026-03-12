"""Tests for coevolution.populations.property.codegen.compose_property_test_script."""

from __future__ import annotations

import textwrap

import pytest

from coevolution.core.interfaces.language import LanguageTransformationError
from coevolution.populations.property.codegen import compose_property_test_script

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_script(script: str) -> str:
    """Execute a self-contained script string and return stripped stdout."""
    import contextlib
    import io

    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        exec(compile(script, "<string>", "exec"), {})  # noqa: S102
    return stdout_buf.getvalue().strip()


SIMPLE_SNIPPET = textwrap.dedent("""\
    def property_output_is_true(inputdata, output):
        return output == "True"
""")

MULTI_FN_SNIPPET = textwrap.dedent("""\
    def helper():
        pass

    def property_nonempty(inputdata, output):
        return len(output) > 0
""")


# ── Structure tests ───────────────────────────────────────────────────────────


class TestScriptStructure:
    def test_snippet_is_included_verbatim(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "x", "y")
        assert SIMPLE_SNIPPET in script

    def test_result_assignment_present(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "x", "y")
        assert "result = property_output_is_true(" in script

    def test_print_result_present(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "x", "y")
        assert "print(result)" in script

    def test_named_args_in_call(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "a", "b")
        assert "inputdata=" in script
        assert "output=" in script

    def test_snippet_comes_before_call(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "a", "b")
        fn_pos = script.index("def property_output_is_true")
        call_pos = script.index("result =")
        assert fn_pos < call_pos

    def test_helper_functions_retained(self) -> None:
        # The full snippet is embedded verbatim, so any helper defined before
        # the property_* function must still appear in the output script.
        script = compose_property_test_script(MULTI_FN_SNIPPET, "in", "out")
        assert "def helper()" in script


# ── Function name extraction ──────────────────────────────────────────────────


class TestFunctionNameExtraction:
    def test_uses_correct_function_name(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "x", "y")
        assert "result = property_output_is_true(" in script

    def test_picks_first_property_fn_when_multiple_exist(self) -> None:
        snippet = textwrap.dedent("""\
            def property_alpha(inputdata, output):
                return True

            def property_beta(inputdata, output):
                return False
        """)
        script = compose_property_test_script(snippet, "x", "y")
        assert "result = property_alpha(" in script
        assert "property_beta(" not in script.split("result =")[1]

    def test_ignores_non_property_functions_before(self) -> None:
        script = compose_property_test_script(MULTI_FN_SNIPPET, "in", "out")
        assert "result = property_nonempty(" in script

    def test_property_fn_with_underscore_suffix(self) -> None:
        snippet = (
            "def property_sorted_order_check(inputdata, output):\n    return True\n"
        )
        script = compose_property_test_script(snippet, "x", "y")
        assert "result = property_sorted_order_check(" in script

    def test_raises_when_no_property_function(self) -> None:
        snippet = "def check_output(inputdata, output):\n    return True\n"
        with pytest.raises(LanguageTransformationError):
            compose_property_test_script(snippet, "x", "y")

    def test_raises_on_empty_snippet(self) -> None:
        with pytest.raises(LanguageTransformationError):
            compose_property_test_script("", "x", "y")

    def test_raises_when_function_named_property_only(self) -> None:
        # 'property' alone without underscore and a suffix should NOT match
        snippet = "def property(inputdata, output):\n    return True\n"
        with pytest.raises(LanguageTransformationError):
            compose_property_test_script(snippet, "x", "y")


# ── JSON injection / string safety ───────────────────────────────────────────


class TestStringSafety:
    def test_plain_strings_injected(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "hello", "world")
        assert '"hello"' in script
        assert '"world"' in script

    def test_double_quotes_in_inputdata_escaped(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, 'say "hi"', "y")
        # json.dumps produces: "say \"hi\""
        assert 'say \\"hi\\"' in script or r"say \"hi\"" in script

    def test_double_quotes_in_output_escaped(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "x", 'a "b" c')
        assert 'a \\"b\\" c' in script or r"a \"b\" c" in script

    def test_newline_in_inputdata_escaped(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "line1\nline2", "y")
        assert "\\n" in script

    def test_backslash_in_values_escaped(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "a\\b", "c\\d")
        assert "\\\\b" in script
        assert "\\\\d" in script

    def test_empty_strings_valid(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, "", "")
        assert '""' in script

    def test_unicode_values_survive(self) -> None:
        # json.dumps escapes non-ASCII by default (\u00e9, etc.) — that is
        # still valid Python; verify the values roundtrip correctly at runtime.
        snippet = textwrap.dedent("""\
            def property_unicode_roundtrip(inputdata, output):
                return inputdata == "héllo" and output == "wörld"
        """)
        script = compose_property_test_script(snippet, "héllo", "wörld")
        assert _run_script(script).strip() == "True"

    def test_json_injection_does_not_break_syntax(self) -> None:
        # A payload that would break naive string interpolation
        malicious = '"); import os; os.system("rm -rf /'
        script = compose_property_test_script(SIMPLE_SNIPPET, malicious, "y")
        # The script must still be syntactically valid Python
        import ast

        ast.parse(script)  # raises SyntaxError if broken


# ── Execution correctness ─────────────────────────────────────────────────────


class TestExecutionCorrectness:
    def test_prints_true_when_property_holds(self) -> None:
        snippet = textwrap.dedent("""\
            def property_output_nonempty(inputdata, output):
                return len(output) > 0
        """)
        script = compose_property_test_script(snippet, "any input", "some output")
        assert _run_script(script) == "True"

    def test_prints_false_when_property_fails(self) -> None:
        snippet = textwrap.dedent("""\
            def property_output_empty(inputdata, output):
                return len(output) == 0
        """)
        script = compose_property_test_script(snippet, "any input", "non-empty")
        assert _run_script(script) == "False"

    def test_inputdata_received_as_string(self) -> None:
        snippet = textwrap.dedent("""\
            def property_inputdata_is_string(inputdata, output):
                return isinstance(inputdata, str)
        """)
        script = compose_property_test_script(snippet, "42", "anything")
        assert _run_script(script) == "True"

    def test_output_received_as_string(self) -> None:
        snippet = textwrap.dedent("""\
            def property_output_is_string(inputdata, output):
                return isinstance(output, str)
        """)
        script = compose_property_test_script(snippet, "anything", "42")
        assert _run_script(script) == "True"

    def test_exact_values_passed_through(self) -> None:
        snippet = textwrap.dedent("""\
            def property_exact_match(inputdata, output):
                return inputdata == "hello world" and output == "HELLO WORLD"
        """)
        script = compose_property_test_script(snippet, "hello world", "HELLO WORLD")
        assert _run_script(script) == "True"

    def test_multiline_inputdata_preserved(self) -> None:
        snippet = textwrap.dedent("""\
            def property_has_newline(inputdata, output):
                return "\\n" in inputdata
        """)
        script = compose_property_test_script(snippet, "line1\nline2", "anything")
        assert _run_script(script) == "True"

    def test_special_chars_in_output_preserved(self) -> None:
        special = 'output with "quotes" and \\backslash'
        snippet = textwrap.dedent(f"""\
            def property_exact_output(inputdata, output):
                return output == {special!r}
        """)
        script = compose_property_test_script(snippet, "in", special)
        assert _run_script(script) == "True"

    def test_property_using_standard_library(self) -> None:
        snippet = textwrap.dedent("""\
            import json

            def property_valid_json(inputdata, output):
                try:
                    json.loads(output)
                    return True
                except ValueError:
                    return False
        """)
        script = compose_property_test_script(snippet, "x", '{"key": 1}')
        assert _run_script(script) == "True"

        script_bad = compose_property_test_script(snippet, "x", "not json")
        assert _run_script(script_bad) == "False"

    def test_helper_function_callable_at_runtime(self) -> None:
        # A property function that delegates to a helper must work — the helper
        # is retained in the composed script and therefore in scope at runtime.
        snippet = textwrap.dedent("""\
            def _is_sorted(items):
                return items == sorted(items)

            def property_output_sorted(inputdata, output):
                return _is_sorted(output.split(","))
        """)
        script = compose_property_test_script(snippet, "x", "a,b,c")
        assert _run_script(script) == "True"

        script_fail = compose_property_test_script(snippet, "x", "c,a,b")
        assert _run_script(script_fail) == "False"


# ── Return value type ─────────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_str(self) -> None:
        result = compose_property_test_script(SIMPLE_SNIPPET, "x", "y")
        assert isinstance(result, str)

    def test_produced_script_ends_with_newline(self) -> None:
        result = compose_property_test_script(SIMPLE_SNIPPET, "x", "y")
        assert result.endswith("\n")
