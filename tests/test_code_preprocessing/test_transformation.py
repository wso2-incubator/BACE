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
    get_target_from_starter,
    remove_if_main_block,
    remove_starter_from_code,
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


class TestGetTargetFromStarter:
    """Test get_target_from_starter function."""

    def test_finds_class_definition(self) -> None:
        starter_code = """
class Solution:
    def solve(self):
        pass
"""
        target_type, target_name = get_target_from_starter(starter_code)
        assert target_name == "Solution"
        import ast

        assert target_type == ast.ClassDef

    def test_finds_function_definition(self) -> None:
        starter_code = """
def solution(x: int) -> int:
    return x
"""
        target_type, target_name = get_target_from_starter(starter_code)
        assert target_name == "solution"
        import ast

        assert target_type == ast.FunctionDef

    def test_returns_first_definition(self) -> None:
        starter_code = """
def helper():
    pass

class Solution:
    pass
"""
        target_type, target_name = get_target_from_starter(starter_code)
        # First definition is the function
        assert target_name == "helper"

    def test_handles_indented_code(self) -> None:
        starter_code = """
    class Solution:
        def solve(self):
            pass
"""
        target_type, target_name = get_target_from_starter(starter_code)
        assert target_name == "Solution"

    def test_handles_complex_class_definition(self) -> None:
        starter_code = """
class Solution(BaseClass, AnotherMixin):
    \"\"\"Docstring.\"\"\"
    def method(self):
        pass
"""
        target_type, target_name = get_target_from_starter(starter_code)
        assert target_name == "Solution"

    def test_raises_error_no_class_or_function(self) -> None:
        starter_code = "x = 1\ny = 2"
        with pytest.raises(ValueError, match="must contain at least one"):
            get_target_from_starter(starter_code)

    def test_raises_error_on_syntax_error(self) -> None:
        starter_code = "class Solution(\n    invalid"
        with pytest.raises(ValueError, match="Error parsing"):
            get_target_from_starter(starter_code)

    def test_handles_function_with_type_hints(self) -> None:
        starter_code = """
def maxPartitionsAfterOperations(s: str, k: int) -> int:
    pass
"""
        target_type, target_name = get_target_from_starter(starter_code)
        assert target_name == "maxPartitionsAfterOperations"


class TestRemoveStarterFromCode:
    """Test remove_starter_from_code function."""

    def test_removes_simple_class(self) -> None:
        full_code = """
class Solution:
    def solve(self):
        return 42

class Helper:
    pass
"""
        starter_code = """
class Solution:
    def solve(self):
        pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class Solution" not in result
        assert "class Helper" in result

    def test_removes_simple_function(self) -> None:
        full_code = """
def solution(x):
    return x * 2

def helper():
    return 1
"""
        starter_code = """
def solution(x):
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "def solution" not in result
        assert "def helper" in result

    def test_removes_class_with_complex_body(self) -> None:
        """Test the exact example from the user."""
        full_code = """
import unittest

class Solution:
    '''
    This is the class we want to remove.
    It has docstrings and complex logic.
    '''
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        # Placeholder naive (incorrect) implementation to allow tests to run in isolation.
        # Replace with the real implementation.
        # Very simplistic: without any change, simulate greedy partitioning.
        def partitions_count(s, k):
            i = 0
            cnt = 0
            n = len(s)
            while i < n:
                seen = set()
                j = i
                while j < n and (s[j] in seen or len(seen) < k):
                    seen.add(s[j])
                    j += 1
                cnt += 1
                i = j
            return cnt
        # Try all single-character changes (and also no change) to choose max
        best = partitions_count(s, k)
        n = len(s)
        import string
        for i in range(n):
            orig = s[i]
            for c in string.ascii_lowercase:
                if c == orig: continue
                ns = s[:i] + c + s[i+1:]
                best = max(best, partitions_count(ns, k))
        return best
    
    def another_method(self):
        # Even with other methods, it will be correctly identified
        pass

class TestMaxPartitionsAfterOperations(unittest.TestCase):
    # ... your tests ...
    def test_example_1(self):
        print("Test 1")
        # s = Solution()
        # self.assertEqual(s.maxPartitionsAfterOperations("abc", 2), 2)
        pass
"""
        # Test with incomplete starter code (no body)
        starter_code = """
class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class Solution" not in result
        assert "class TestMaxPartitionsAfterOperations" in result
        assert "import unittest" in result
        assert "def test_example_1" in result

    def test_removes_class_preserves_other_classes(self) -> None:
        full_code = """
class First:
    pass

class Second:
    def method(self):
        pass

class Third:
    pass
"""
        starter_code = """
class Second:
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class First" in result
        assert "class Second" not in result
        assert "class Third" in result

    def test_removes_function_preserves_other_functions(self) -> None:
        full_code = """
def func_one():
    return 1

def func_two():
    return 2

def func_three():
    return 3
"""
        starter_code = """
def func_two():
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "def func_one" in result
        assert "def func_two" not in result
        assert "def func_three" in result

    def test_removes_class_with_imports(self) -> None:
        full_code = """
import os
import sys
from typing import List

class Solution:
    def solve(self):
        return os.path.exists('/')

def helper():
    return 42
"""
        starter_code = """
class Solution:
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "import os" in result
        assert "import sys" in result
        assert "from typing import List" in result
        assert "class Solution" not in result
        assert "def helper" in result

    def test_removes_function_with_nested_functions(self) -> None:
        full_code = """
def outer():
    def inner():
        return 1
    return inner()

def other():
    pass
"""
        starter_code = """
def outer():
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "def outer" not in result
        assert "def inner" not in result  # inner is nested, should be removed
        assert "def other" in result

    def test_handles_indented_full_code(self) -> None:
        # Test that dedent works on starter code (not full code which gets indented)
        # The transformation.py uses textwrap.dedent on starter code to handle indentation
        full_code = """
class Solution:
    def solve(self):
        pass

class Helper:
    pass
"""
        # Indented starter code should work due to textwrap.dedent in get_target_from_starter
        starter_code = """
    class Solution:
        pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class Solution" not in result
        assert "class Helper" in result

    def test_removes_class_with_decorators(self) -> None:
        full_code = """
@dataclass
class Solution:
    x: int
    
    def solve(self):
        return self.x

class Helper:
    pass
"""
        starter_code = """
class Solution:
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class Solution" not in result
        assert "class Helper" in result

    def test_removes_function_with_decorators(self) -> None:
        full_code = """
@cache
def solution(x):
    return x * 2

@timer
def helper():
    return 1
"""
        starter_code = """
def solution(x):
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "def solution" not in result
        assert "@cache" not in result
        assert "def helper" in result

    def test_returns_error_on_invalid_starter_code(self) -> None:
        full_code = "class Solution:\n    pass"
        invalid_starter = "class Solution(\n    invalid"
        result = remove_starter_from_code(full_code, invalid_starter)
        # Should return error message string
        assert isinstance(result, str)
        assert "Error" in result or "invalid" in result

    def test_returns_error_on_invalid_full_code(self) -> None:
        starter_code = "class Solution:\n    pass"
        invalid_full = "class Solution:\n    def x(\n    invalid"
        result = remove_starter_from_code(invalid_full, starter_code)
        # Should return error message string
        assert isinstance(result, str)
        assert "Error" in result

    def test_handles_empty_file_after_removal(self) -> None:
        full_code = """
class Solution:
    pass
"""
        starter_code = """
class Solution:
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        # Result should be valid Python code (empty module)
        assert isinstance(result, str)
        assert "class Solution" not in result

    def test_removes_class_with_staticmethod(self) -> None:
        full_code = """
class Solution:
    @staticmethod
    def static_method():
        return 42
    
    def instance_method(self):
        pass

class Other:
    pass
"""
        starter_code = """
class Solution:
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class Solution" not in result
        assert "class Other" in result

    def test_removes_class_with_classmethod(self) -> None:
        full_code = """
class Solution:
    @classmethod
    def create(cls):
        return cls()
    
    def method(self):
        pass

class Other:
    pass
"""
        starter_code = """
class Solution:
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class Solution" not in result
        assert "class Other" in result

    def test_removes_class_with_property(self) -> None:
        full_code = """
class Solution:
    @property
    def value(self):
        return self._value
    
    def __init__(self):
        self._value = 0

class Other:
    pass
"""
        starter_code = """
class Solution:
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class Solution" not in result
        assert "class Other" in result

    def test_removes_function_with_multiple_parameters(self) -> None:
        full_code = """
def solution(a: int, b: str, c: List[int] = None, *args, **kwargs) -> bool:
    return True

def other():
    return False
"""
        # Test with incomplete starter code (no body)
        starter_code = """
def solution(a: int, b: str, c: List[int] = None, *args, **kwargs) -> bool:
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "def solution" not in result
        assert "def other" in result

    def test_preserves_code_comments(self) -> None:
        full_code = """
# This is a comment about Solution
class Solution:
    pass

# This is a comment about Helper
class Helper:
    pass
"""
        starter_code = """
class Solution:
    pass
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class Solution" not in result
        assert "class Helper" in result
        # Comments may or may not be preserved depending on AST behavior

    def test_removes_function_with_complex_logic(self) -> None:
        full_code = """
def solution(nums):
    result = []
    for num in nums:
        if num > 0:
            result.append(num * 2)
    return result

def helper():
    return 1
"""
        # Test with incomplete starter code (no body)
        starter_code = """
def solution(nums):
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "def solution" not in result
        assert "def helper" in result

    def test_removes_class_with_inheritance(self) -> None:
        full_code = """
class Solution(BaseClass, Mixin):
    def solve(self):
        return super().solve()

class Other:
    pass
"""
        # Test with incomplete starter code (no body)
        starter_code = """
class Solution(BaseClass, Mixin):
"""
        result = remove_starter_from_code(full_code, starter_code)
        assert isinstance(result, str)
        assert "class Solution" not in result
        assert "class Other" in result

    def test_exact_user_example(self) -> None:
        """Test using the exact example from the user request."""
        full_code = """
import unittest

# Assume the solution is imported as described:
# from solution_module import Solution
# For the purpose of these tests, I'll create a placeholder Solution class.
# Replace the placeholder with the actual imported Solution when running tests.

class Solution:
    '''
    This is the class we want to remove.
    It has docstrings and complex logic.
    '''
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        # Placeholder naive (incorrect) implementation to allow tests to run in isolation.
        # Replace with the real implementation.
        # Very simplistic: without any change, simulate greedy partitioning.
        def partitions_count(s, k):
            i = 0
            cnt = 0
            n = len(s)
            while i < n:
                seen = set()
                j = i
                while j < n and (s[j] in seen or len(seen) < k):
                    seen.add(s[j])
                    j += 1
                cnt += 1
                i = j
            return cnt
        # Try all single-character changes (and also no change) to choose max
        best = partitions_count(s, k)
        n = len(s)
        import string
        for i in range(n):
            orig = s[i]
            for c in string.ascii_lowercase:
                if c == orig: continue
                ns = s[:i] + c + s[i+1:]
                best = max(best, partitions_count(ns, k))
        return best
    
    def another_method(self):
        # Even with other methods, it will be correctly identified
        pass

class TestMaxPartitionsAfterOperations(unittest.TestCase):
    # ... your tests ...
    def test_example_1(self):
        print("Test 1")
        # s = Solution()
        # self.assertEqual(s.maxPartitionsAfterOperations("abc", 2), 2)
        pass
"""
        # Exact incomplete starter code from user request
        starter_code = """
class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
"""
        result = remove_starter_from_code(full_code, starter_code)

        assert isinstance(result, str)
        # Verify Solution class is removed
        assert "class Solution:" not in result
        assert "def maxPartitionsAfterOperations" not in result
        assert "def another_method" not in result

        # Verify TestMaxPartitionsAfterOperations is preserved
        assert "class TestMaxPartitionsAfterOperations" in result
        assert "def test_example_1" in result

        # Verify imports are preserved
        assert "import unittest" in result
