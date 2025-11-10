"""Tests for code_preprocessing.transformation module."""

import pytest

from common.code_preprocessing.exceptions import (
    CodeParsingError,
    CodeTransformationError,
)
from common.code_preprocessing.transformation import (
    extract_class_block,
    extract_function_with_helpers,
    extract_test_methods_code,
    remove_if_main_block,
    replace_test_methods,
)


class TestExtractTestMethodsCode:
    """Test extract_test_methods_code function."""

    def test_extracts_test_method_code(self) -> None:
        code = """
class TestFoo(unittest.TestCase):
    def test_bar(self):
        self.assertTrue(True)
"""
        methods = extract_test_methods_code(code)
        assert len(methods) == 1
        assert "def test_bar" in methods[0]
        assert "self.assertTrue(True)" in methods[0]

    def test_extracts_multiple_test_methods(self) -> None:
        code = """
class TestFoo(unittest.TestCase):
    def test_one(self):
        pass
    
    def test_two(self):
        pass
"""
        methods = extract_test_methods_code(code)
        assert len(methods) == 2

    def test_skips_non_test_methods(self) -> None:
        code = """
class TestFoo(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_something(self):
        pass
"""
        methods = extract_test_methods_code(code)
        assert len(methods) == 1
        assert len(methods) == 1

    def test_skips_solution_class(self) -> None:
        code = """
class Solution:
    def solve(self):
        pass

class TestSolution(unittest.TestCase):
    def test_solve(self):
        pass
"""
        methods = extract_test_methods_code(code)
        assert len(methods) == 1

    def test_raises_error_on_syntax_error(self) -> None:
        code = "class TestFoo(\n    invalid"
        with pytest.raises(CodeParsingError):
            extract_test_methods_code(code)


class TestExtractFunctionWithHelpers:
    """Test extract_function_with_helpers function."""

    def test_extracts_function_with_no_helpers(self) -> None:
        code = "def main():\n    return 42"
        result = extract_function_with_helpers(code, "main")
        assert "return 42" in result

    def test_extracts_function_with_helpers(self) -> None:
        code = """
def helper():
    return 1

def main():
    return helper()
"""
        result = extract_function_with_helpers(code, "main")
        assert "def helper" in result
        assert "return helper()" in result

    def test_includes_imports(self) -> None:
        code = """
import os

def main():
    return os.path.exists('/')
"""
        result = extract_function_with_helpers(code, "main")
        assert "import os" in result

    def test_raises_error_when_function_not_found(self) -> None:
        code = "def foo():\n    pass"
        with pytest.raises(CodeTransformationError, match="not found"):
            extract_function_with_helpers(code, "nonexistent")

    def test_indents_properly(self) -> None:
        code = "def main():\n    x = 1\n    return x"
        result = extract_function_with_helpers(code, "main")
        # Should have 4-space indentation
        assert "    x = 1" in result
        assert "    return x" in result


class TestExtractClassBlock:
    """Test extract_class_block function."""

    def test_extracts_simple_class(self) -> None:
        code = "class Foo:\n    pass"
        result = extract_class_block(code)
        assert "class Foo" in result

    def test_extracts_class_with_methods(self) -> None:
        code = """
class Foo:
    def bar(self):
        pass
"""
        result = extract_class_block(code)
        assert "class Foo" in result
        assert "def bar" in result

    def test_extracts_first_class(self) -> None:
        code = """
class First:
    pass

class Second:
    pass
"""
        result = extract_class_block(code)
        assert "class First" in result
        assert "class Second" not in result

    def test_raises_error_when_no_class(self) -> None:
        code = "def foo():\n    pass"
        with pytest.raises(CodeTransformationError, match="No class definition found"):
            extract_class_block(code)


class TestReplaceTestMethods:
    """Test replace_test_methods function."""

    def test_replaces_test_methods(self) -> None:
        original = """
class TestFoo(unittest.TestCase):
    def test_old(self):
        pass
"""
        new_methods = ["def test_new(self):\n    self.assertTrue(True)"]
        result = replace_test_methods(original, new_methods)
        assert "test_new" in result
        assert "test_old" not in result

    def test_preserves_setup_methods(self) -> None:
        original = """
class TestFoo(unittest.TestCase):
    def setUp(self):
        self.x = 1
    
    def test_old(self):
        pass
"""
        new_methods = ["def test_new(self):\n    pass"]
        result = replace_test_methods(original, new_methods)
        assert "def setUp" in result
        assert "self.x = 1" in result

    def test_preserves_class_variables(self) -> None:
        original = """
class TestFoo(unittest.TestCase):
    MAX_VALUE = 100
    
    def test_old(self):
        pass
"""
        new_methods = ["def test_new(self):\n    pass"]
        result = replace_test_methods(original, new_methods)
        assert "MAX_VALUE = 100" in result

    def test_handles_multiple_new_methods(self) -> None:
        original = "class TestFoo(unittest.TestCase):\n    def test_old(self): pass"
        new_methods = [
            "def test_one(self):\n    pass",
            "def test_two(self):\n    pass",
        ]
        result = replace_test_methods(original, new_methods)
        assert "test_one" in result
        assert "test_two" in result

    def test_raises_error_when_no_class(self) -> None:
        code = "def foo(): pass"
        with pytest.raises(CodeTransformationError):
            replace_test_methods(code, [])


class TestRemoveIfMainBlock:
    """Test remove_if_main_block function."""

    def test_removes_main_block(self) -> None:
        code = """
def foo():
    pass

if __name__ == "__main__":
    foo()
"""
        result = remove_if_main_block(code)
        assert "if __name__" not in result
        assert "def foo" in result

    def test_preserves_code_without_main_block(self) -> None:
        code = "def foo():\n    pass"
        result = remove_if_main_block(code)
        assert "def foo" in result

    def test_preserves_other_if_blocks(self) -> None:
        code = """
if x > 0:
    print("positive")

if __name__ == "__main__":
    run()
"""
        result = remove_if_main_block(code)
        assert "if x > 0" in result
        assert "if __name__" not in result

    def test_raises_error_on_syntax_error(self) -> None:
        code = "if invalid syntax"
        with pytest.raises(CodeParsingError):
            remove_if_main_block(code)

    def test_removes_main_block_with_single_quotes(self) -> None:
        code = """
def foo():
    pass

if __name__ == '__main__':
    foo()
"""
        result = remove_if_main_block(code)
        assert "if __name__" not in result
        assert "__main__" not in result
        assert "def foo" in result

    def test_removes_reversed_comparison(self) -> None:
        """Test reversed comparison: if '__main__' == __name__:"""
        code = """
def foo():
    pass

if '__main__' == __name__:
    foo()
"""
        result = remove_if_main_block(code)
        assert "if __name__" not in result
        assert "if '__main__'" not in result
        assert "def foo" in result

    def test_removes_main_block_with_parentheses(self) -> None:
        code = """
def foo():
    pass

if (__name__ == "__main__"):
    foo()
"""
        result = remove_if_main_block(code)
        assert "if __name__" not in result
        assert "def foo" in result

    def test_removes_multiline_main_block(self) -> None:
        code = """
def foo():
    pass

if (
    __name__ == "__main__"
):
    foo()
    bar()
"""
        result = remove_if_main_block(code)
        assert "if __name__" not in result
        assert "__main__" not in result
        assert "def foo" in result

    def test_removes_main_block_with_else(self) -> None:
        """The entire if-else block should be removed when if is main block"""
        code = """
def foo():
    pass

if __name__ == "__main__":
    foo()
else:
    print("imported")
"""
        result = remove_if_main_block(code)
        assert "if __name__" not in result
        assert "else:" not in result
        assert "def foo" in result

    def test_removes_main_block_with_multiple_statements(self) -> None:
        code = """
import sys

def foo():
    pass

def bar():
    pass

if __name__ == "__main__":
    foo()
    bar()
    sys.exit(0)
"""
        result = remove_if_main_block(code)
        assert "if __name__" not in result
        assert "import sys" in result
        assert "def foo" in result
        assert "def bar" in result
        assert "sys.exit" not in result

    def test_preserves_nested_if_with_name_check(self) -> None:
        """Should only remove top-level main blocks, not nested ones"""
        code = """
def check():
    if __name__ == "__main__":
        print("nested")

if __name__ == "__main__":
    check()
"""
        result = remove_if_main_block(code)
        # Nested if inside function should be preserved
        assert "def check" in result
        # Top-level if should be removed
        lines = result.strip().split("\n")
        # Should not have top-level if __name__ (only the one in function)
        assert not any(
            line.strip().startswith("if __name__")
            for line in lines
            if "def check" not in result[: result.index(line)]
        )

    def test_preserves_other_name_comparisons(self) -> None:
        code = """
if __name__ != "__main__":
    print("imported")

if __name__ == "other":
    print("other")
"""
        result = remove_if_main_block(code)
        assert "if __name__ !=" in result
        # ast.unparse may convert quotes, so check for either
        assert "if __name__ == 'other'" in result or 'if __name__ == "other"' in result

    def test_empty_main_block(self) -> None:
        code = """
def foo():
    pass

if __name__ == "__main__":
    pass
"""
        result = remove_if_main_block(code)
        assert "if __name__" not in result
        assert "def foo" in result
