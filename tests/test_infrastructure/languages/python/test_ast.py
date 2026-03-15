import textwrap
import pytest

from coevolution.core.interfaces import LanguageParsingError
from infrastructure.languages.python import PythonLanguage
from infrastructure.languages.python import ast as python_ast


@pytest.fixture
def adapter() -> PythonLanguage:
    """Create PythonLanguage instance for testing."""
    return PythonLanguage()


class TestIsSyntaxValid:
    def test_valid_function(self, adapter: PythonLanguage) -> None:
        assert adapter.parser.is_syntax_valid("def add(a, b):\n    return a + b")

    def test_valid_class(self, adapter: PythonLanguage) -> None:
        assert adapter.parser.is_syntax_valid("class Solution:\n    def solve(self): pass")

    def test_empty_string_is_valid(self, adapter: PythonLanguage) -> None:
        assert adapter.parser.is_syntax_valid("")

    def test_invalid_syntax(self, adapter: PythonLanguage) -> None:
        assert not adapter.parser.is_syntax_valid("def broken(:\n    pass")

    def test_unmatched_paren(self, adapter: PythonLanguage) -> None:
        assert not adapter.parser.is_syntax_valid("result = (1 + 2")


class TestExtractTestNames:
    def test_extracts_single_test(self, adapter: PythonLanguage) -> None:
        code = "def test_add():\n    assert 1 + 1 == 2"
        assert adapter.parser.extract_test_names(code) == ["test_add"]

    def test_extracts_multiple_tests(self, adapter: PythonLanguage) -> None:
        code = (
            "def test_one():\n    pass\n\ndef test_two():\n    pass\n\n"
            "def helper():\n    pass"
        )
        names = adapter.parser.extract_test_names(code)
        assert names == ["test_one", "test_two"]

    def test_returns_empty_when_no_tests(self, adapter: PythonLanguage) -> None:
        assert adapter.parser.extract_test_names("def helper(): pass") == []

    def test_handles_syntax_error_gracefully(self, adapter: PythonLanguage) -> None:
        assert adapter.parser.extract_test_names("def test_broken(") == []


class TestSplitTests:
    def test_splits_single_test(self, adapter: PythonLanguage) -> None:
        code = "def test_foo():\n    assert True"
        parts = adapter.parser.split_tests(code)
        assert len(parts) == 1
        assert "test_foo" in parts[0]

    def test_splits_multiple_tests(self, adapter: PythonLanguage) -> None:
        code = "def test_a():\n    pass\n\ndef test_b():\n    pass"
        parts = adapter.parser.split_tests(code)
        assert len(parts) == 2

    def test_ignores_non_test_functions(self, adapter: PythonLanguage) -> None:
        code = "def helper():\n    pass\n\ndef test_real():\n    pass"
        parts = adapter.parser.split_tests(code)
        assert len(parts) == 1
        assert "test_real" in parts[0]

    def test_returns_empty_on_no_tests(self, adapter: PythonLanguage) -> None:
        assert adapter.parser.split_tests("x = 1") == []


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


class TestContainsStarterCode:
    def test_exact_match(self, adapter: PythonLanguage) -> None:
        starter = "class Solution:\n    def solve(self): pass"
        code = "class Solution:\n    def solve(self):\n        return 42"
        assert adapter.parser.contains_starter_code(code, starter)

    def test_empty_starter_always_true(self, adapter: PythonLanguage) -> None:
        assert adapter.parser.contains_starter_code("def f(): pass", "")

    def test_missing_function_returns_false(self, adapter: PythonLanguage) -> None:
        starter = "class Solution:\n    def missing(self): pass"
        code = "class Solution:\n    def present(self): pass"
        assert not adapter.parser.contains_starter_code(code, starter)


class TestGetStructuralMetadata:
    def test_extracts_classes_and_functions(self, adapter: PythonLanguage) -> None:
        code = "class Foo:\n    pass\n\ndef bar():\n    pass"
        meta = adapter.parser.get_structural_metadata(code)
        assert "Foo" in meta["classes"]
        assert "bar" in meta["functions"]

    def test_extracts_imports(self, adapter: PythonLanguage) -> None:
        code = "import math\nfrom typing import List\n\ndef f(): pass"
        meta = adapter.parser.get_structural_metadata(code)
        assert any("import math" in imp for imp in meta["imports"])
        assert any("List" in imp for imp in meta["imports"])

    def test_returns_empty_for_empty_code(self, adapter: PythonLanguage) -> None:
        meta = adapter.parser.get_structural_metadata("")
        assert meta["classes"] == []
        assert meta["functions"] == []
        assert meta["imports"] == []


class TestParseTestInputs:
    def test_parses_list_of_dicts(self, adapter: PythonLanguage) -> None:
        outputs = '[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'
        result = adapter.parser.parse_test_inputs(outputs)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_wraps_single_dict_in_list(self, adapter: PythonLanguage) -> None:
        outputs = '{"a": 1}'
        result = adapter.parser.parse_test_inputs(outputs)
        assert len(result) == 1
        assert result[0]["a"] == 1

    def test_handles_inf_and_nan(self, adapter: PythonLanguage) -> None:
        import math

        result = adapter.parser.parse_test_inputs("[{'v': inf}]")
        assert len(result) == 1
        assert math.isinf(result[0]["v"])

    def test_returns_empty_on_invalid_input(self, adapter: PythonLanguage) -> None:
        assert adapter.parser.parse_test_inputs("not valid") == []


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
        assert adapter.parser.get_docstring(code) == "I am the class docstring"

    def test_priority_2_method_docstring(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent('''
            class Solution:
                def solve(self):
                    """I am the method docstring"""
                    pass
        ''')
        assert adapter.parser.get_docstring(code) == "I am the method docstring"

    def test_priority_3_method_comments(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent("""
            class Solution:
                # I am the method comment
                def solve(self):
                    pass
        """)
        assert adapter.parser.get_docstring(code) == "I am the method comment"

    def test_priority_4_class_comments(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent("""
            # I am the class comment
            class Solution:
                def solve(self):
                    pass
        """)
        assert adapter.parser.get_docstring(code) == "I am the class comment"

    def test_method_comments_beat_class_comments(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent("""
            # Class comment (should be ignored)
            class Solution:
                # Method comment (should be picked)
                def solve(self):
                    pass
        """)
        assert adapter.parser.get_docstring(code) == "Method comment (should be picked)"

    def test_standalone_function_docstring(self, adapter: PythonLanguage) -> None:
        code = textwrap.dedent('''
            def solve():
                """Standalone docstring"""
                pass
        ''')
        assert adapter.parser.get_docstring(code) == "Standalone docstring"

    def test_returns_none_on_failure(self, adapter: PythonLanguage) -> None:
        assert adapter.parser.get_docstring("not valid python {{{") is None
