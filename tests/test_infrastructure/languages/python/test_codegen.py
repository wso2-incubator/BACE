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


@pytest.fixture
def adapter() -> PythonLanguage:
    """Create PythonLanguage instance for testing."""
    return PythonLanguage()


class TestComposeTestScript:
    def test_combines_code_and_test(self, adapter: PythonLanguage) -> None:
        code = "def add(a, b):\n    return a + b"
        test = "def test_add():\n    assert add(1, 2) == 3"
        script = adapter.composer.compose_test_script(code, test)
        assert "def add" in script
        assert "def test_add" in script

    def test_adds_pytest_import_when_missing(self, adapter: PythonLanguage) -> None:
        script = adapter.composer.compose_test_script(
            "def f(): pass", "def test_f(): pass"
        )
        assert "import pytest" in script

    def test_does_not_duplicate_pytest_import(self, adapter: PythonLanguage) -> None:
        code = "import pytest\ndef f(): pass"
        test = "def test_f(): pass"
        script = adapter.composer.compose_test_script(code, test)
        assert script.count("import pytest") == 1

    def test_includes_main_block(self, adapter: PythonLanguage) -> None:
        script = adapter.composer.compose_test_script(
            "def f(): pass", "def test_f(): pass"
        )
        assert 'if __name__ == "__main__":' in script
        assert 'pytest.main([__file__, "-v"])' in script


class TestGenerateTestCase:
    def test_stdin_test_generation(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def sol(self, input_str: str) -> str:
                    pass
        """)
        code = adapter.composer.generate_test_case("5\n3 2 1", "1 2 3", starter, 1)
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
        code = adapter.composer.generate_test_case("5\n3", "8", starter, 1)
        assert "def test_case_1():" in code
        assert "solution = Solution()" in code
        assert "solution.add(*args)" in code

    def test_standalone_function_test_generation(self, adapter: PythonLanguage) -> None:
        starter = "def add(x: int, y: int) -> int:\n    pass\n"
        code = adapter.composer.generate_test_case("5\n3", "8", starter, 1)
        assert "def test_case_1():" in code
        assert "solution = " not in code
        assert "add(*args)" in code

    def test_test_number_in_function_name(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def calculate(self, a: int, b: int, c: int) -> int:
                    pass
        """)
        code = adapter.composer.generate_test_case("1\n2\n3", "6", starter, 42)
        assert "def test_case_42():" in code

    def test_different_class_names_preserved(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class MySolution:
                def compute(self, x: int) -> int:
                    pass
        """)
        code = adapter.composer.generate_test_case("5", "25", starter, 1)
        assert "MySolution()" in code
        assert "solution.compute(*args)" in code

    def test_invalid_starter_raises_error(self, adapter: PythonLanguage) -> None:
        with pytest.raises(LanguageParsingError):
            adapter.composer.generate_test_case("1", "2", "not valid python code {", 1)

    def test_stdin_round_trip_executes(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def sol(self, input_str: str) -> str:
                    return input_str
        """)
        test_code = adapter.composer.generate_test_case("hello", "hello", starter, 1)
        ns: dict[str, Any] = {}
        exec(starter + "\n" + test_code, ns)
        ns["test_case_1"]()  # must not raise

    def test_functional_round_trip_executes(self, adapter: PythonLanguage) -> None:
        starter = textwrap.dedent("""
            class Solution:
                def add(self, x: int, y: int) -> int:
                    return x + y
        """)
        test_code = adapter.composer.generate_test_case("5\n3", "8", starter, 1)
        ns: dict[str, Any] = {}
        exec(starter + "\n" + test_code, ns)
        ns["test_case_1"]()

    def test_standalone_round_trip_executes(self, adapter: PythonLanguage) -> None:
        starter = "def sort_array(arr):\n    return sorted(arr)\n"
        test_code = adapter.composer.generate_test_case(
            "[3, 1, 2]", "[1, 2, 3]", starter, 1
        )
        ns: dict[str, Any] = {}
        exec(starter + "\n" + test_code, ns)
        ns["test_case_1"]()


class TestRemoveMainBlock:
    def test_removes_main_block(self, adapter: PythonLanguage) -> None:
        code = "def foo():\n    pass\n\nif __name__ == '__main__':\n    foo()\n"
        result = adapter.parser.remove_main_block(code)
        assert "if __name__" not in result
        assert "def foo" in result

    def test_preserves_code_without_main_block(self, adapter: PythonLanguage) -> None:
        code = "def foo():\n    pass"
        result = adapter.parser.remove_main_block(code)
        assert "def foo" in result

    def test_preserves_other_if_blocks(self, adapter: PythonLanguage) -> None:
        code = (
            "x = 1\nif x > 0:\n    print(x)\n\nif __name__ == '__main__':\n    run()\n"
        )
        result = adapter.parser.remove_main_block(code)
        assert "if x > 0" in result
        assert "if __name__" not in result

    def test_removes_double_quoted_main_block(self, adapter: PythonLanguage) -> None:
        code = 'def foo():\n    pass\n\nif __name__ == "__main__":\n    foo()\n'
        result = adapter.parser.remove_main_block(code)
        assert "if __name__" not in result

    def test_removes_reversed_comparison(self, adapter: PythonLanguage) -> None:
        code = "def foo():\n    pass\n\nif '__main__' == __name__:\n    foo()\n"
        result = adapter.parser.remove_main_block(code)
        assert "__main__" not in result
        assert "def foo" in result

    def test_raises_on_syntax_error(self, adapter: PythonLanguage) -> None:
        with pytest.raises(LanguageParsingError):
            adapter.parser.remove_main_block("if invalid syntax")


class TestComposeEvaluationScript:
    def test_basic_solution_class(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
        script = adapter.composer.compose_evaluation_script(
            prog, "{'input_arg': {'a': 1, 'b': 2}}"
        )
        assert "class Solution" in script
        assert "sol = Solution()" in script
        assert "sol.add(" in script
        assert "a=1" in script
        assert "b=2" in script
        assert "json.dumps" in script

    def test_generated_script_executes_correctly(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
        script = adapter.composer.compose_evaluation_script(
            prog, "{'input_arg': {'a': 1, 'b': 2}}"
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
        script = adapter.composer.compose_evaluation_script(
            prog, "{'input_arg': {'x': 3, 'y': 4}}"
        )
        assert "class Solution" in script
        assert "def multiply(self, x, y)" in script
        assert "multiply(" in script
        assert "x=3" in script
        assert "y=4" in script

    def test_string_parameters(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def process(self, text, count):\n        return text * count\n"
        script = adapter.composer.compose_evaluation_script(
            prog, "{'input_arg': {'text': 'hello', 'count': 3}}"
        )
        assert "process(" in script
        assert "'hello'" in script
        assert "count=3" in script

    def test_preserves_imports(self, adapter: PythonLanguage) -> None:
        prog = "import math\n\nclass Solution:\n    def sqrt_sum(self, a, b):\n        return math.sqrt(a + b)\n"
        script = adapter.composer.compose_evaluation_script(
            prog, "{'input_arg': {'a': 3, 'b': 6}}"
        )
        assert "import math" in script

    def test_preserves_helper_classes(self, adapter: PythonLanguage) -> None:
        prog = (
            "class Helper:\n    def val(self): return 42\n\n"
            "class Solution:\n    def solve(self, x):\n        return Helper().val() + x\n"
        )
        script = adapter.composer.compose_evaluation_script(
            prog, "{'input_arg': {'x': 8}}"
        )
        assert "class Helper" in script
        assert "class Solution" in script

    def test_no_parameters_function(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def get_answer(self):\n        return 42\n"
        script = adapter.composer.compose_evaluation_script(prog, "{'input_arg': {}}")
        assert "get_answer()" in script

    def test_list_as_parameter(self, adapter: PythonLanguage) -> None:
        prog = (
            "class Solution:\n    def sum_list(self, nums):\n        return sum(nums)\n"
        )
        script = adapter.composer.compose_evaluation_script(
            prog, "{'input_arg': {'nums': [1, 2, 3, 4, 5]}}"
        )
        assert "sum_list(" in script
        assert "[1, 2, 3, 4, 5]" in script

    def test_boolean_parameter(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def toggle(self, flag):\n        return not flag\n"
        script = adapter.composer.compose_evaluation_script(
            prog, "{'input_arg': {'flag': True}}"
        )
        assert "toggle(" in script
        assert "flag=True" in script

    def test_raises_on_invalid_programmer_code(self, adapter: PythonLanguage) -> None:
        with pytest.raises(
            LanguageParsingError, match="Failed to parse programmer code"
        ):
            adapter.composer.compose_evaluation_script(
                "class Solution: invalid syntax here", "{'input_arg': {}}"
            )

    def test_raises_when_no_solution_found(self, adapter: PythonLanguage) -> None:
        with pytest.raises(LanguageTransformationError, match="No Solution class"):
            adapter.composer.compose_evaluation_script("x = 1", "{'input_arg': {}}")

    def test_raises_when_missing_inputdata_key(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def solve(self, x):\n        return x\n"
        # Robust parsing now handles missing key by assuming the whole dict is inputdata
        script = adapter.composer.compose_evaluation_script(
            prog, "{'wrongkey': {'x': 1}}"
        )
        assert "wrongkey={'x': 1}" in script

    def test_raises_when_inputdata_not_dict(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def solve(self, x):\n        return x\n"
        with pytest.raises(
            LanguageTransformationError, match="Input data must be a dict"
        ):
            adapter.composer.compose_evaluation_script(
                prog, "{'input_arg': 'not a dict'}"
            )

    def test_raises_on_invalid_input_dict(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def solve(self, x):\n        return x\n"
        with pytest.raises(
            LanguageTransformationError, match="Failed to parse input data"
        ):
            adapter.composer.compose_evaluation_script(
                prog, "this is not valid python dict"
            )

    def test_multiline_string_with_newlines(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def solve(self, input_str):\n        return len(input_str.split('\\n'))\n"
        input_data = "{'input_arg': {'input_str': 'line1\\nline2\\nline3'}}"
        script = adapter.composer.compose_evaluation_script(prog, input_data)
        captured = io.StringIO()
        sys.stdout = captured
        try:
            exec(script)
        finally:
            sys.stdout = sys.__stdout__
        assert "3" in captured.getvalue()

    def test_whitespace_in_input_dict(self, adapter: PythonLanguage) -> None:
        prog = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
        script = adapter.composer.compose_evaluation_script(
            prog, "{ 'input_arg' : { 'a' : 1 , 'b' : 2 } }"
        )
        assert "add(" in script
        assert "a=1" in script
        assert "b=2" in script

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
        script = adapter.composer.compose_evaluation_script(
            prog, "{'input_arg': {'x': 10}}"
        )
        assert "sol.solve(x=10)" in script


class TestNormalizeCode:
    def test_removes_comments(self, adapter: PythonLanguage) -> None:
        code = "def f():\n    # comment\n    return 1"
        normalized = adapter.parser.normalize_code(code)
        assert "# comment" not in normalized
        assert "return 1" in normalized

    def test_removes_double_quote_docstrings(self, adapter: PythonLanguage) -> None:
        code = 'def f():\n    """docstring"""\n    return 1'
        normalized = adapter.parser.normalize_code(code)
        assert "docstring" not in normalized

    def test_removes_single_quote_docstrings(self, adapter: PythonLanguage) -> None:
        code = "def f():\n    '''docstring'''\n    return 1"
        normalized = adapter.parser.normalize_code(code)
        assert "docstring" not in normalized

    def test_strips_extra_whitespace(self, adapter: PythonLanguage) -> None:
        code = "def   foo():     pass"
        normalized = adapter.parser.normalize_code(code)
        assert "  " not in normalized
