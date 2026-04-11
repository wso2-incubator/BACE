"""Tests for helpers in coevolution.populations.property.operators.helpers."""

from __future__ import annotations

import json
import textwrap
import contextlib
import io
import ast
from unittest.mock import MagicMock
import pytest

from coevolution.core.interfaces.data import Test
from coevolution.core.interfaces.language import ICodeParser, LanguageTransformationError
from coevolution.populations.property.operators.helpers import (
    transform_public_tests,
    compose_property_test_script
)
from infrastructure.languages.python.adapter import PythonParser

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_script(script: str) -> str:
    """Execute a self-contained script string and return stripped stdout."""
    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        exec(compile(script, "<string>", "exec"), {})  # noqa: S102
    return stdout_buf.getvalue().strip()

# ── transform_public_tests tests ──────────────────────────────────────────────

def test_transform_empty_list() -> None:
    parser = PythonParser()
    assert transform_public_tests([], "def f(): ...", parser) == []

def test_transform_single_arg_list() -> None:
    parser = PythonParser()
    
    tests = [Test(input="[1, 2, 3]", output="[1, 2, 3]")]
    transformed = transform_public_tests(tests, "def sort(lst): ...", parser)
    
    assert len(transformed) == 1
    assert json.loads(transformed[0].input) == {"lst": [1, 2, 3]}
    assert json.loads(transformed[0].output) == [1, 2, 3]

def test_transform_multiple_args() -> None:
    parser = PythonParser()
    
    # Input has 3 lines
    tests = [Test(input="10\n'hello'\n3.14", output="True")]
    transformed = transform_public_tests(tests, "def f(a, b, c): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"a": 10, "b": "hello", "c": 3.14}
    assert json.loads(transformed[0].output) is True

def test_transform_newline_separated_strings() -> None:
    parser = PythonParser()
    
    tests = [Test(input="first line\nsecond line", output="'ok'")]
    transformed = transform_public_tests(tests, "def f(s1, s2): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"s1": "first line", "s2": "second line"}
    assert json.loads(transformed[0].output) == "ok"

def test_transform_space_separated_fallback() -> None:
    parser = PythonParser()
    
    tests = [Test(input="1\n2", output="3")]
    transformed = transform_public_tests(tests, "def add(x, y): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"x": 1, "y": 2}
    assert json.loads(transformed[0].output) == 3

def test_transform_signature_failure_returns_raw() -> None:
    parser = MagicMock(spec=ICodeParser)
    parser.parse_public_test.side_effect = Exception("Parse error")
    
    tests = [Test(input="raw input", output="raw output")]
    transformed = transform_public_tests(tests, "stutter code", parser)
    
    assert transformed == tests

def test_transform_mismatched_count_best_effort() -> None:
    parser = PythonParser()
    
    # 3 inputs for 2 params
    tests = [Test(input="1\n2\n3", output="6")]
    transformed = transform_public_tests(tests, "def f(a, b): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"a": 1, "b": 2}
    assert json.loads(transformed[0].output) == 6

def test_transform_complex_literal_eval() -> None:
    parser = PythonParser()
    
    tests = [Test(input="{'key': [1, 2, {'inner': 'val'}]}", output="None")]
    transformed = transform_public_tests(tests, "def process(data): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"data": {"key": [1, 2, {"inner": "val"}]}}

# ── compose_property_test_script tests ────────────────────────────────────────

SIMPLE_SNIPPET = textwrap.dedent("""\
    def property_output_is_true(input_arg, output):
        return output == "True"
""")

MULTI_FN_SNIPPET = textwrap.dedent("""\
    def helper():
        pass

    def property_nonempty(input_arg, output):
        return len(output) > 0
""")

class TestScriptStructure:
    def test_snippet_is_included_verbatim(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "x"}, "y")
        assert SIMPLE_SNIPPET in script

    def test_result_assignment_present(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "x"}, "y")
        assert "result = property_output_is_true(" in script

    def test_print_result_present(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "x"}, "y")
        assert "print(result)" in script

    def test_named_args_in_call(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "a"}, "b")
        assert "input_arg=" in script
        assert "output=" in script

    def test_snippet_comes_before_call(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "a"}, "b")
        fn_pos = script.index("def property_output_is_true")
        call_pos = script.index("result =")
        assert fn_pos < call_pos

    def test_helper_functions_retained(self) -> None:
        script = compose_property_test_script(MULTI_FN_SNIPPET, {"arg": "in"}, "out")
        assert "def helper()" in script

class TestFunctionNameExtraction:
    def test_uses_correct_function_name(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "x"}, "y")
        assert "result = property_output_is_true(" in script

    def test_picks_first_property_fn_when_multiple_exist(self) -> None:
        snippet = textwrap.dedent("""\
            def property_alpha(input_arg, output):
                return True

            def property_beta(input_arg, output):
                return False
        """)
        script = compose_property_test_script(snippet, {"arg": "x"}, "y")
        assert "result = property_alpha(" in script
        assert "property_beta(" not in script.split("result =")[1]

    def test_ignores_non_property_functions_before(self) -> None:
        script = compose_property_test_script(MULTI_FN_SNIPPET, {"arg": "in"}, "out")
        assert "result = property_nonempty(" in script

    def test_property_fn_with_underscore_suffix(self) -> None:
        snippet = (
            "def property_sorted_order_check(input_arg, output):\n    return True\n"
        )
        script = compose_property_test_script(snippet, {"arg": "x"}, "y")
        assert "result = property_sorted_order_check(" in script

    def test_raises_when_no_property_function(self) -> None:
        snippet = "def check_output(input_arg, output):\n    return True\n"
        with pytest.raises(LanguageTransformationError):
            compose_property_test_script(snippet, {"arg": "x"}, "y")

    def test_raises_on_empty_snippet(self) -> None:
        with pytest.raises(LanguageTransformationError):
            compose_property_test_script("", {"arg": "x"}, "y")

    def test_raises_when_function_named_property_only(self) -> None:
        snippet = "def property(input_arg, output):\n    return True\n"
        with pytest.raises(LanguageTransformationError):
            compose_property_test_script(snippet, {"arg": "x"}, "y")

class TestStringSafety:
    def test_plain_strings_injected(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "hello"}, "world")
        assert "'hello'" in script
        assert "'world'" in script

    def test_double_quotes_in_input_arg_escaped(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": 'say "hi"'}, "y")
        assert 'say "hi"' in script

    def test_double_quotes_in_output_escaped(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "x"}, 'a "b" c')
        assert 'a "b" c' in script

    def test_newline_in_input_arg_escaped(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "line1\nline2"}, "y")
        assert "\\n" in script

    def test_backslash_in_values_escaped(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "a\\b"}, "c\\d")
        assert "\\\\b" in script
        assert "\\\\d" in script

    def test_empty_strings_valid(self) -> None:
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": ""}, "")
        assert "''" in script

    def test_unicode_values_survive(self) -> None:
        snippet = textwrap.dedent("""\
            def property_unicode_roundtrip(input_arg, output):
                return input_arg["arg"] == "héllo" and output == "wörld"
        """)
        script = compose_property_test_script(snippet, {"arg": "héllo"}, "wörld")
        assert _run_script(script).strip() == "True"

    def test_json_injection_does_not_break_syntax(self) -> None:
        malicious = '"); import os; os.system("rm -rf /'
        script = compose_property_test_script(SIMPLE_SNIPPET, {"arg": malicious}, "y")
        ast.parse(script)  # raises SyntaxError if broken

class TestExecutionCorrectness:
    def test_prints_true_when_property_holds(self) -> None:
        snippet = textwrap.dedent("""\
            def property_output_nonempty(input_arg, output):
                return len(output) > 0
        """)
        script = compose_property_test_script(snippet, {"arg": "any input"}, "some output")
        assert _run_script(script) == "True"

    def test_prints_false_when_property_fails(self) -> None:
        snippet = textwrap.dedent("""\
            def property_output_empty(input_arg, output):
                return len(output) == 0
        """)
        script = compose_property_test_script(snippet, {"arg": "any input"}, "non-empty")
        assert _run_script(script) == "False"

    def test_inputdata_received_as_string(self) -> None:
        snippet = textwrap.dedent("""\
            def property_input_arg_is_dict(input_arg, output):
                return isinstance(input_arg, dict)
        """)
        script = compose_property_test_script(snippet, {"arg": "42"}, "anything")
        assert _run_script(script) == "True"

    def test_output_received_as_string(self) -> None:
        snippet = textwrap.dedent("""\
            def property_output_is_string(input_arg, output):
                return isinstance(output, str)
        """)
        script = compose_property_test_script(snippet, {"arg": "anything"}, "42")
        assert _run_script(script) == "True"

    def test_exact_values_passed_through(self) -> None:
        snippet = textwrap.dedent("""\
            def property_exact_match(input_arg, output):
                return input_arg == {"arg": "hello world"} and output == "HELLO WORLD"
        """)
        script = compose_property_test_script(snippet, {"arg": "hello world"}, "HELLO WORLD")
        assert _run_script(script) == "True"

    def test_multiline_inputdata_preserved(self) -> None:
        snippet = textwrap.dedent("""\
            def property_has_newline(input_arg, output):
                return "\\n" in input_arg["arg"]
        """)
        script = compose_property_test_script(snippet, {"arg": "line1\nline2"}, "anything")
        assert _run_script(script) == "True"

    def test_special_chars_in_output_preserved(self) -> None:
        special = 'output with "quotes" and \\backslash'
        snippet = textwrap.dedent(f"""\
            def property_exact_output(input_arg, output):
                return output == {special!r}
        """)
        script = compose_property_test_script(snippet, {"arg": "in"}, special)
        assert _run_script(script) == "True"

    def test_property_using_standard_library(self) -> None:
        snippet = textwrap.dedent("""\
            import json

            def property_valid_json(input_arg, output):
                try:
                    json.loads(output)
                    return True
                except ValueError:
                    return False
        """)
        script = compose_property_test_script(snippet, {"arg": "x"}, '{"key": 1}')
        assert _run_script(script) == "True"

        script_bad = compose_property_test_script(snippet, {"arg": "x"}, "not json")
        assert _run_script(script_bad) == "False"

    def test_helper_function_callable_at_runtime(self) -> None:
        snippet = textwrap.dedent("""\
            def _is_sorted(items):
                return items == sorted(items)

            def property_output_sorted(input_arg, output):
                return _is_sorted(output.split(","))
        """)
        script = compose_property_test_script(snippet, {"arg": "x"}, "a,b,c")
        assert _run_script(script) == "True"

        script_fail = compose_property_test_script(snippet, {"arg": "x"}, "c,a,b")
        assert _run_script(script_fail) == "False"

class TestReturnType:
    def test_returns_str(self) -> None:
        result = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "x"}, "y")
        assert isinstance(result, str)

    def test_produced_script_ends_with_newline(self) -> None:
        result = compose_property_test_script(SIMPLE_SNIPPET, {"arg": "x"}, "y")
        assert result.endswith("\n")
