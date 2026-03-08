import io
import sys
import textwrap
from typing import Any

import pytest

from coevolution.core.interfaces import (
    LanguageParsingError,
    LanguageTransformationError,
)
from infrastructure.languages.python import PythonLanguage
from infrastructure.languages.python import ast as python_ast
from infrastructure.languages.python.ast import MethodSignature as _MethodSignature


@pytest.fixture
def adapter() -> PythonLanguage:
    """Create PythonLanguage instance for testing."""
    return PythonLanguage()


# ---------------------------------------------------------------------------
# Language property
# ---------------------------------------------------------------------------


class TestLanguageProperty:
    def test_returns_python(self, adapter: PythonLanguage) -> None:
        assert adapter.language == "python"


# ---------------------------------------------------------------------------
# extract_code_blocks
# ---------------------------------------------------------------------------


class TestExtractCodeBlocks:
    def test_extracts_single_python_block(self, adapter: PythonLanguage) -> None:
        response = """Here's the solution:
```python
def add(a, b):
    return a + b
```"""
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "def add" in blocks[0]

    def test_extracts_capital_python_block(self, adapter: PythonLanguage) -> None:
        response = "```Python\ndef foo(): pass\n```"
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "def foo" in blocks[0]

    def test_extracts_multiple_blocks(self, adapter: PythonLanguage) -> None:
        response = """```python
def solution_a():
    return 1
```

```python
def solution_b():
    return 2
```"""
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 2
        assert "solution_a" in blocks[0]
        assert "solution_b" in blocks[1]

    def test_falls_back_to_raw_response_when_valid(
        self, adapter: PythonLanguage
    ) -> None:
        response = "def foo():\n    return 42"
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "def foo" in blocks[0]

    def test_returns_empty_list_when_no_valid_code(
        self, adapter: PythonLanguage
    ) -> None:
        response = "This is just plain text with no code."
        blocks = adapter.extract_code_blocks(response)
        assert blocks == []

    def test_skips_blocks_with_syntax_errors(self, adapter: PythonLanguage) -> None:
        response = """```python
def valid():
    return 1
```
```python
def broken(
```"""
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "def valid" in blocks[0]


# ---------------------------------------------------------------------------
# is_syntax_valid
# ---------------------------------------------------------------------------


class TestIsSyntaxValid:
    def test_valid_function(self, adapter: PythonLanguage) -> None:
        assert adapter.is_syntax_valid("def add(a, b):\n    return a + b")

    def test_valid_class(self, adapter: PythonLanguage) -> None:
        assert adapter.is_syntax_valid("class Solution:\n    def solve(self): pass")

    def test_empty_string_is_valid(self, adapter: PythonLanguage) -> None:
        # ast.parse('') succeeds
        assert adapter.is_syntax_valid("")

    def test_invalid_syntax(self, adapter: PythonLanguage) -> None:
        assert not adapter.is_syntax_valid("def broken(:\n    pass")

    def test_unmatched_paren(self, adapter: PythonLanguage) -> None:
        assert not adapter.is_syntax_valid("result = (1 + 2")


# ---------------------------------------------------------------------------
# extract_test_names
# ---------------------------------------------------------------------------


class TestExtractTestNames:
    def test_extracts_single_test(self, adapter: PythonLanguage) -> None:
        code = "def test_add():\n    assert 1 + 1 == 2"
        assert adapter.extract_test_names(code) == ["test_add"]

    def test_extracts_multiple_tests(self, adapter: PythonLanguage) -> None:
        code = (
            "def test_one():\n    pass\n\ndef test_two():\n    pass\n\n"
            "def helper():\n    pass"
        )
        names = adapter.extract_test_names(code)
        assert names == ["test_one", "test_two"]

    def test_returns_empty_when_no_tests(self, adapter: PythonLanguage) -> None:
        assert adapter.extract_test_names("def helper(): pass") == []

    def test_handles_syntax_error_gracefully(self, adapter: PythonLanguage) -> None:
        assert adapter.extract_test_names("def test_broken(") == []


# ---------------------------------------------------------------------------
# split_tests
# ---------------------------------------------------------------------------


class TestSplitTests:
    def test_splits_single_test(self, adapter: PythonLanguage) -> None:
        code = "def test_foo():\n    assert True"
        parts = adapter.split_tests(code)
        assert len(parts) == 1
        assert "test_foo" in parts[0]

    def test_splits_multiple_tests(self, adapter: PythonLanguage) -> None:
        code = "def test_a():\n    pass\n\ndef test_b():\n    pass"
        parts = adapter.split_tests(code)
        assert len(parts) == 2

    def test_ignores_non_test_functions(self, adapter: PythonLanguage) -> None:
        code = "def helper():\n    pass\n\ndef test_real():\n    pass"
        parts = adapter.split_tests(code)
        assert len(parts) == 1
        assert "test_real" in parts[0]

    def test_returns_empty_on_no_tests(self, adapter: PythonLanguage) -> None:
        assert adapter.split_tests("x = 1") == []


# ---------------------------------------------------------------------------
# compose_test_script
# ---------------------------------------------------------------------------


class TestComposeTestScript:
    def test_combines_code_and_test(self, adapter: PythonLanguage) -> None:
        code = "def add(a, b):\n    return a + b"
        test = "def test_add():\n    assert add(1, 2) == 3"
        script = adapter.compose_test_script(code, test)
        assert "def add" in script
        assert "def test_add" in script

    def test_adds_pytest_import_when_missing(self, adapter: PythonLanguage) -> None:
        script = adapter.compose_test_script("def f(): pass", "def test_f(): pass")
        assert "import pytest" in script

    def test_does_not_duplicate_pytest_import(self, adapter: PythonLanguage) -> None:
        code = "import pytest\ndef f(): pass"
        test = "def test_f(): pass"
        script = adapter.compose_test_script(code, test)
        assert script.count("import pytest") == 1

    def test_includes_main_block(self, adapter: PythonLanguage) -> None:
        script = adapter.compose_test_script("def f(): pass", "def test_f(): pass")
        assert 'if __name__ == "__main__":' in script
        assert 'pytest.main([__file__, "-v"])' in script


# ---------------------------------------------------------------------------
# _parse_method_signature (private helper, tested directly)
# ---------------------------------------------------------------------------


class TestParseMethodSignature:
    def test_stdin_class_signature(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def sol(self, input_str: str) -> str:
                    pass
        """)
        sig = python_ast.parse_method_signature(starter)
        assert sig.class_name == "Solution"
        assert sig.method_name == "sol"
        assert sig.params == [("input_str", "str")]
        assert sig.return_type == "str"
        assert sig.is_standalone is False

    def test_standalone_function(self, adapter: PythonLanguage) -> None:
        starter = "def add(x: int, y: int) -> int:\n    pass\n"
        sig = python_ast.parse_method_signature(starter)
        assert sig.class_name is None
        assert sig.method_name == "add"
        assert sig.params == [("x", "int"), ("y", "int")]
        assert sig.return_type == "int"
        assert sig.is_standalone is True

    def test_functional_class_signature(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def add(self, x: int, y: int) -> int:
                    pass
        """)
        sig = python_ast.parse_method_signature(starter)
        assert sig.class_name == "Solution"
        assert sig.method_name == "add"
        assert sig.params == [("x", "int"), ("y", "int")]
        assert sig.return_type == "int"

    def test_incomplete_signature_auto_completed(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def calculate(self, value: float) -> float:
        """)
        sig = python_ast.parse_method_signature(starter)
        assert sig.method_name == "calculate"
        assert sig.params == [("value", "float")]

    def test_no_type_annotations(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def process(self, data):
                    pass
        """)
        sig = python_ast.parse_method_signature(starter)
        assert sig.params == [("data", None)]
        assert sig.return_type is None

    def test_no_class_or_function_raises_error(self, adapter: PythonLanguage) -> None:
        with pytest.raises(
            LanguageParsingError, match="No class or function definition found"
        ):
            python_ast.parse_method_signature("# just a comment\nx = 5\n")

    def test_class_without_instance_method_raises_error(
        self, adapter: PythonLanguage
    ) -> None:
        starter = textwrap.dedent("""
            class Solution:
                @staticmethod
                def static_method():
                    pass
        """)
        with pytest.raises(LanguageParsingError, match="No instance method found"):
            python_ast.parse_method_signature(starter)

    def test_multiple_methods_selects_first(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def first(self, x: int) -> int:
                    pass
                def second(self, y: str) -> str:
                    pass
        """)
        sig = python_ast.parse_method_signature(starter)
        assert sig.method_name == "first"


# ---------------------------------------------------------------------------
# generate_test_case (ILanguage)
# ---------------------------------------------------------------------------


class TestGenerateTestCase:
    def test_stdin_test_generation(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def sol(self, input_str: str) -> str:
                    pass
        """)
        code = adapter.generate_test_case("5\n3 2 1", "1 2 3", starter, 1)
        assert "def test_case_1():" in code
        assert "solution = Solution()" in code
        assert "solution.sol(input_str)" in code
        assert repr("5\n3 2 1") in code

    def test_functional_class_test_generation(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def add(self, x: int, y: int) -> int:
                    pass
        """)
        code = adapter.generate_test_case("5\n3", "8", starter, 1)
        assert "def test_case_1():" in code
        assert "solution = Solution()" in code
        assert "import ast" in code
        assert "solution.add(*args)" in code

    def test_standalone_function_test_generation(self, adapter: PythonLanguage) -> None:
        starter = "def add(x: int, y: int) -> int:\n    pass\n"
        code = adapter.generate_test_case("5\n3", "8", starter, 1)
        assert "def test_case_1():" in code
        assert "solution = " not in code
        assert "add(*args)" in code

    def test_test_number_in_function_name(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def calculate(self, a: int, b: int, c: int) -> int:
                    pass
        """)
        code = adapter.generate_test_case("1\n2\n3", "6", starter, 42)
        assert "def test_case_42():" in code

    def test_different_class_names_preserved(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class MySolution:
                def compute(self, x: int) -> int:
                    pass
        """)
        code = adapter.generate_test_case("5", "25", starter, 1)
        assert "MySolution()" in code
        assert "solution.compute(*args)" in code

    def test_invalid_starter_raises_error(self, adapter: PythonLanguage) -> None:
        with pytest.raises(LanguageParsingError):
            adapter.generate_test_case("1", "2", "not valid python code {", 1)

    def test_stdin_round_trip_executes(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def sol(self, input_str: str) -> str:
                    return input_str
        """)
        test_code = adapter.generate_test_case("hello", "hello", starter, 1)
        ns: dict[str, Any] = {}
        exec(starter + "\n" + test_code, ns)
        ns["test_case_1"]()  # must not raise

    def test_functional_round_trip_executes(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def add(self, x: int, y: int) -> int:
                    return x + y
        """)
        test_code = adapter.generate_test_case("5\n3", "8", starter, 1)
        ns: dict[str, Any] = {}
        exec(starter + "\n" + test_code, ns)
        ns["test_case_1"]()

    def test_standalone_round_trip_executes(self, adapter: PythonLanguage) -> None:
        starter = "def sort_array(arr):\n    return sorted(arr)\n"
        test_code = adapter.generate_test_case("[3, 1, 2]", "[1, 2, 3]", starter, 1)
        ns: dict[str, Any] = {}
        exec(starter + "\n" + test_code, ns)
        ns["test_case_1"]()


# ---------------------------------------------------------------------------
# remove_main_block (ILanguage)
# ---------------------------------------------------------------------------


class TestRemoveMainBlock:
    def test_removes_main_block(self, adapter: PythonLanguage) -> None:
        code = "def foo():\n    pass\n\nif __name__ == '__main__':\n    foo()\n"
        result = adapter.remove_main_block(code)
        assert "if __name__" not in result
        assert "def foo" in result

    def test_preserves_code_without_main_block(self, adapter: PythonLanguage) -> None:
        code = "def foo():\n    pass"
        result = adapter.remove_main_block(code)
        assert "def foo" in result

    def test_preserves_other_if_blocks(self, adapter: PythonLanguage) -> None:
        code = (
            "x = 1\nif x > 0:\n    print(x)\n\nif __name__ == '__main__':\n    run()\n"
        )
        result = adapter.remove_main_block(code)
        assert "if x > 0" in result
        assert "if __name__" not in result

    def test_removes_double_quoted_main_block(self, adapter: PythonLanguage) -> None:
        code = 'def foo():\n    pass\n\nif __name__ == "__main__":\n    foo()\n'
        result = adapter.remove_main_block(code)
        assert "if __name__" not in result

    def test_removes_reversed_comparison(self, adapter: PythonLanguage) -> None:
        code = "def foo():\n    pass\n\nif '__main__' == __name__:\n    foo()\n"
        result = adapter.remove_main_block(code)
        assert "__main__" not in result
        assert "def foo" in result

    def test_raises_on_syntax_error(self, adapter: PythonLanguage) -> None:
        with pytest.raises(LanguageParsingError):
            adapter.remove_main_block("if invalid syntax")


# ---------------------------------------------------------------------------
# compose_evaluation_script (ILanguage)
# ---------------------------------------------------------------------------


class TestComposeEvaluationScript:
    def test_basic_solution_class(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
        script = adapter.compose_evaluation_script(
            prog, "{'inputdata': {'a': 1, 'b': 2}}"
        )
        assert "class Solution" in script
        assert "sol = Solution()" in script
        assert "sol.add(1, 2)" in script
        assert "print(sol.add(1, 2))" in script

    def test_generated_script_executes_correctly(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
        script = adapter.compose_evaluation_script(
            prog, "{'inputdata': {'a': 1, 'b': 2}}"
        )
        captured = io.StringIO()
        sys.stdout = captured
        try:
            exec(script)
        finally:
            sys.stdout = sys.__stdout__
        assert "3" in captured.getvalue()

    def test_wraps_loose_functions_into_solution(self, adapter: PythonLanguage) -> None:
        prog = "def multiply(x, y):\n    return x * y\n"
        script = adapter.compose_evaluation_script(
            prog, "{'inputdata': {'x': 3, 'y': 4}}"
        )
        assert "class Solution" in script
        assert "def multiply(self, x, y)" in script
        assert "sol.multiply(3, 4)" in script

    def test_string_parameters(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def process(self, text, count):\n        return text * count\n"
        script = adapter.compose_evaluation_script(
            prog, "{'inputdata': {'text': 'hello', 'count': 3}}"
        )
        assert "process('hello', 3)" in script

    def test_preserves_imports(self, adapter: PythonLanguage) -> None:
        prog = "import math\n\nclass Solution:\n    def sqrt_sum(self, a, b):\n        return math.sqrt(a + b)\n"
        script = adapter.compose_evaluation_script(
            prog, "{'inputdata': {'a': 3, 'b': 6}}"
        )
        assert "import math" in script

    def test_preserves_helper_classes(self, adapter: PythonLanguage) -> None:
        prog = (
            "class Helper:\n    def val(self): return 42\n\n"
            "class Solution:\n    def solve(self, x):\n        return Helper().val() + x\n"
        )
        script = adapter.compose_evaluation_script(prog, "{'inputdata': {'x': 8}}")
        assert "class Helper" in script
        assert "class Solution" in script

    def test_no_parameters_function(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def get_answer(self):\n        return 42\n"
        script = adapter.compose_evaluation_script(prog, "{'inputdata': {}}")
        assert "get_answer()" in script

    def test_list_as_parameter(self, adapter: PythonLanguage) -> None:
        prog = (
            "class Solution:\n    def sum_list(self, nums):\n        return sum(nums)\n"
        )
        script = adapter.compose_evaluation_script(
            prog, "{'inputdata': {'nums': [1, 2, 3, 4, 5]}}"
        )
        assert "sum_list([1, 2, 3, 4, 5])" in script

    def test_boolean_parameter(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def toggle(self, flag):\n        return not flag\n"
        script = adapter.compose_evaluation_script(
            prog, "{'inputdata': {'flag': True}}"
        )
        assert "toggle(True)" in script

    def test_raises_on_invalid_programmer_code(self, adapter: PythonLanguage) -> None:
        with pytest.raises(
            LanguageParsingError, match="Failed to parse programmer code"
        ):
            adapter.compose_evaluation_script(
                "class Solution: invalid syntax here", "{'inputdata': {}}"
            )

    def test_raises_when_no_solution_found(self, adapter: PythonLanguage) -> None:
        with pytest.raises(LanguageTransformationError, match="No Solution class"):
            adapter.compose_evaluation_script("x = 1", "{'inputdata': {}}")

    def test_raises_when_missing_inputdata_key(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def solve(self, x):\n        return x\n"
        with pytest.raises(LanguageTransformationError, match="inputdata"):
            adapter.compose_evaluation_script(prog, "{'wrongkey': {'x': 1}}")

    def test_raises_when_inputdata_not_dict(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def solve(self, x):\n        return x\n"
        with pytest.raises(LanguageTransformationError, match="dict object"):
            adapter.compose_evaluation_script(prog, "{'inputdata': 'not a dict'}")

    def test_raises_on_invalid_input_dict(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def solve(self, x):\n        return x\n"
        with pytest.raises(
            LanguageTransformationError, match="Failed to parse input data"
        ):
            adapter.compose_evaluation_script(prog, "this is not valid python dict")

    def test_multiline_string_with_newlines(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def solve(self, input_str):\n        return len(input_str.split('\\n'))\n"
        input_data = "{'inputdata': {'input_str': 'line1\\nline2\\nline3'}}"
        script = adapter.compose_evaluation_script(prog, input_data)
        captured = io.StringIO()
        sys.stdout = captured
        try:
            exec(script)
        finally:
            sys.stdout = sys.__stdout__
        assert "3" in captured.getvalue()

    def test_whitespace_in_input_dict(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
        script = adapter.compose_evaluation_script(
            prog, "{ 'inputdata' : { 'a' : 1 , 'b' : 2 } }"
        )
        assert "add(1, 2)" in script

    def test_methods_called_in_order_first_method(
        self, adapter: PythonLanguage
    ) -> None:
        prog = (
            "class Solution:\n"
            "    def solve(self, x):\n"
            "        return x * 2\n"
            "    def other(self, y):\n"
            "        return y * 3\n"
        )
        script = adapter.compose_evaluation_script(prog, "{'inputdata': {'x': 10}}")
        assert "sol.solve(10)" in script


# ---------------------------------------------------------------------------
# normalize_code
# ---------------------------------------------------------------------------


class TestNormalizeCode:
    def test_removes_comments(self, adapter: PythonLanguage) -> None:
        code = "def f():\n    # comment\n    return 1"
        normalized = adapter.normalize_code(code)
        assert "# comment" not in normalized
        assert "return 1" in normalized

    def test_removes_double_quote_docstrings(self, adapter: PythonLanguage) -> None:
        code = 'def f():\n    """docstring"""\n    return 1'
        normalized = adapter.normalize_code(code)
        assert "docstring" not in normalized

    def test_removes_single_quote_docstrings(self, adapter: PythonLanguage) -> None:
        code = "def f():\n    '''docstring'''\n    return 1"
        normalized = adapter.normalize_code(code)
        assert "docstring" not in normalized

    def test_strips_extra_whitespace(self, adapter: PythonLanguage) -> None:
        code = "def   foo():     pass"
        normalized = adapter.normalize_code(code)
        assert "  " not in normalized


# ---------------------------------------------------------------------------
# contains_starter_code
# ---------------------------------------------------------------------------


class TestContainsStarterCode:
    def test_exact_match(self, adapter: PythonLanguage) -> None:
        starter = "class Solution:\n    def solve(self): pass"
        code = "class Solution:\n    def solve(self):\n        return 42"
        assert adapter.contains_starter_code(code, starter)

    def test_empty_starter_always_true(self, adapter: PythonLanguage) -> None:
        assert adapter.contains_starter_code("def f(): pass", "")

    def test_missing_function_returns_false(self, adapter: PythonLanguage) -> None:
        starter = "class Solution:\n    def missing(self): pass"
        code = "class Solution:\n    def present(self): pass"
        assert not adapter.contains_starter_code(code, starter)


# ---------------------------------------------------------------------------
# get_structural_metadata
# ---------------------------------------------------------------------------


class TestGetStructuralMetadata:
    def test_extracts_classes_and_functions(self, adapter: PythonLanguage) -> None:
        code = "class Foo:\n    pass\n\ndef bar():\n    pass"
        meta = adapter.get_structural_metadata(code)
        assert "Foo" in meta["classes"]
        assert "bar" in meta["functions"]

    def test_extracts_imports(self, adapter: PythonLanguage) -> None:
        code = "import math\nfrom typing import List\n\ndef f(): pass"
        meta = adapter.get_structural_metadata(code)
        assert any("import math" in imp for imp in meta["imports"])
        assert any("List" in imp for imp in meta["imports"])

    def test_returns_empty_for_empty_code(self, adapter: PythonLanguage) -> None:
        meta = adapter.get_structural_metadata("")
        assert meta["classes"] == []
        assert meta["functions"] == []
        assert meta["imports"] == []


# ---------------------------------------------------------------------------
# parse_test_inputs
# ---------------------------------------------------------------------------


class TestParseTestInputs:
    def test_parses_list_of_dicts(self, adapter: PythonLanguage) -> None:
        outputs = '[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'
        result = adapter.parse_test_inputs(outputs)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_wraps_single_dict_in_list(self, adapter: PythonLanguage) -> None:
        outputs = '{"a": 1}'
        result = adapter.parse_test_inputs(outputs)
        assert len(result) == 1
        assert result[0]["a"] == 1

    def test_handles_inf_and_nan(self, adapter: PythonLanguage) -> None:
        import math

        result = adapter.parse_test_inputs("[{'v': inf}]")
        assert len(result) == 1
        assert math.isinf(result[0]["v"])

    def test_returns_empty_on_invalid_input(self, adapter: PythonLanguage) -> None:
        assert adapter.parse_test_inputs("not valid") == []


# ---------------------------------------------------------------------------
# get_docstring
# ---------------------------------------------------------------------------


class TestDocstringExtraction:
    def test_priority_1_class_docstring(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent('''
            class Solution:
                """I am the class docstring"""
                # I am a method comment
                def solve(self):
                    """I am the method docstring"""
                    pass
        ''')
        assert adapter.get_docstring(code) == "I am the class docstring"

    def test_priority_2_method_docstring(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent('''
            class Solution:
                def solve(self):
                    """I am the method docstring"""
                    pass
        ''')
        assert adapter.get_docstring(code) == "I am the method docstring"

    def test_priority_3_method_comments(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent("""
            class Solution:
                # I am the method comment
                def solve(self):
                    pass
        """)
        assert adapter.get_docstring(code) == "I am the method comment"

    def test_priority_4_class_comments(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent("""
            # I am the class comment
            class Solution:
                def solve(self):
                    pass
        """)
        assert adapter.get_docstring(code) == "I am the class comment"

    def test_method_comments_beat_class_comments(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent("""
            # Class comment (should be ignored)
            class Solution:
                # Method comment (should be picked)
                def solve(self):
                    pass
        """)
        assert adapter.get_docstring(code) == "Method comment (should be picked)"

    def test_standalone_function_docstring(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent('''
            def solve():
                """Standalone docstring"""
                pass
        ''')
        assert adapter.get_docstring(code) == "Standalone docstring"

    def test_returns_empty_string_on_failure(self, adapter: PythonLanguage) -> None:
        assert adapter.get_docstring("not valid python {{{") == ""
