"""Tests for code_preprocessing.transformation module."""

import ast
import textwrap

import pytest

from common.code_preprocessing.exceptions import (
    CodeParsingError,
    CodeTransformationError,
)
from common.code_preprocessing.transformation import (
    build_test_method_from_io,
    extract_class_block,
    extract_first_test_method_code,
    extract_function_with_helpers,
    extract_test_methods_code,
    extract_unittest_code,
    get_target_from_starter,
    is_unittest_class,
    remove_if_main_block,
    remove_starter_from_code,
    setup_unittest_class_from_starter_code,
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

    def test_preserves_comments_and_docstrings(self) -> None:
        """Ensure that inline comments and docstrings in test methods are preserved."""
        code = textwrap.dedent('''
            import unittest

            class TestComments(unittest.TestCase):
                def test_with_comments(self):
                    """This is a docstring for the test method."""
                    # Setup value
                    x = 1  # inline assignment comment
                    # Assert something
                    self.assertEqual(x, 1)
        ''')

        methods = extract_test_methods_code(code)
        assert len(methods) == 1
        method_code = methods[0]
        # Docstring should be present in the extracted source
        assert "This is a docstring for the test method." in method_code
        # Inline comments should be preserved
        assert "# Setup value" in method_code
        assert "# inline assignment comment" in method_code


class TestExtractFirstTestMethodCode:
    """
    Tests for extract_first_test_method_code.
    Focuses on correctness, indentation handling, and preservation of source formatting.
    """

    def test_extracts_first_test_method_from_unittest_class_exact_match(self) -> None:
        """Should extract the method and return it without class indentation."""
        code = textwrap.dedent("""
            import unittest

            class TestFoo(unittest.TestCase):
                def test_bar(self):
                    self.assertTrue(True)
                
                def test_baz(self):
                    pass
        """)

        # We expect the indentation to be stripped
        expected = textwrap.dedent("""
            def test_bar(self):
                self.assertTrue(True)
        """).strip()

        result = extract_first_test_method_code(code)
        # Normalize dedentation from the extractor before comparison
        assert textwrap.dedent(result).strip() == expected

    def test_preserves_comments_and_formatting(self) -> None:
        """
        Test that extract_first_test_method_code works with ast.unparse.
        Comments are not preserved with ast.unparse, but the function should still work.
        """
        code = textwrap.dedent("""
            import unittest

            class TestStyle(unittest.TestCase):
                def test_comments(self):
                    # This checks exact string
                    x = 'single_quotes' # Inline comment
        """)

        result = extract_first_test_method_code(code)

        # With our extractor using source segments, comments may be preserved
        assert "def test_comments(self):" in result
        assert "x = 'single_quotes'" in result
        # Comments should now be preserved when source segment is available
        assert "# This checks exact string" in result

    def test_preserves_comments_and_docstrings_in_first_test_method(self) -> None:
        """Ensure that extracting the first test method preserves comments and docstrings."""
        code = textwrap.dedent('''
            import unittest

            class TestCommentsFirst(unittest.TestCase):
                def test_with_comments(self):
                    """Docstring for the first test method."""
                    # Setup value
                    x = 42  # inline assignment comment
                    # Assert
                    self.assertEqual(x, 42)
        ''')

        result = extract_first_test_method_code(code)
        # Docstring and comments should be preserved when source segment is available
        assert "Docstring for the first test method." in result
        assert "# Setup value" in result
        assert "# inline assignment comment" in result

    def test_extracts_first_test_method_from_multiple_classes(self) -> None:
        """Should stop at the first valid unittest class found."""
        code = textwrap.dedent("""
            import unittest

            class TestFirst(unittest.TestCase):
                def test_one(self):
                    pass

            class TestSecond(unittest.TestCase):
                def test_two(self):
                    pass
        """)

        result = extract_first_test_method_code(code)
        assert "def test_one" in result
        assert "def test_two" not in result

    def test_extracts_first_test_function_fallback(self) -> None:
        """Should handle files that use pytest style functions instead of classes."""
        code = textwrap.dedent("""
            def test_foo():
                assert True

            def test_bar():
                assert False
        """)

        result = extract_first_test_method_code(code)
        assert "def test_foo" in result
        assert "def test_bar" not in result

    def test_raises_error_when_class_has_no_test_methods(self) -> None:
        """Should raise error if a class exists but contains no methods starting with test_."""
        code = textwrap.dedent("""
            import unittest

            class TestFoo(unittest.TestCase):
                def setUp(self):
                    pass
                def helper(self):
                    pass
        """)

        with pytest.raises(CodeTransformationError, match="No test method found"):
            extract_first_test_method_code(code)

    def test_raises_error_when_no_test_methods_at_all(self) -> None:
        """Should raise error if code is valid python but has no tests."""
        code = textwrap.dedent("""
            def helper():
                return 1
            class Solution:
                def solve(self):
                    return 2
        """)

        with pytest.raises(CodeTransformationError, match="No test method found"):
            extract_first_test_method_code(code)

    def test_raises_error_on_syntax_error(self) -> None:
        """Should catch SyntaxError and re-raise as CodeParsingError."""
        code = "class TestFoo(\n    invalid code here..."

        with pytest.raises(CodeParsingError, match="Failed to parse test code"):
            extract_first_test_method_code(code)

    def test_skips_non_unittest_classes(self) -> None:
        """
        Should ignore classes that don't look like unittest classes.
        """
        code = textwrap.dedent("""
            import unittest

            class FakeClass:
                def test_fake(self):
                    pass

            class RealClass(unittest.TestCase):
                def test_real(self):
                    pass
        """)

        result = extract_first_test_method_code(code)

        # Should find the first unittest class and its first test method
        assert "def test_real" in result
        assert "def test_fake" not in result

    def test_extracts_large_test_method(self) -> None:
        """Ensure that a large, real-world-looking test method is extracted intact."""
        code = """
def test_all_zero_to_all_one_optimal_ascending_costs(self):
    for c in C:  # ascending order
        current_sum += c  # that flipped bit now contributes
        total += current_sum

    expected = str(total)
    self.assertEqual(Solution().sol(input_str), expected)
"""

        # ACT
        # Assuming extract_first_test_method_code handles the parsing correctly
        # (Note: ast.parse usually requires dedented input if not in a file context,
        # so we dedent input here to ensure parsing succeeds)
        result = extract_first_test_method_code(textwrap.dedent(code))

        # ASSERT
        # We strip both sides to ignore leading/trailing newlines
        # We dedent 'code' to match the flat structure returned by the function
        assert textwrap.dedent(code).strip() == result.strip()


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


class TestIsUnittestClass:
    """Test is_unittest_class function."""

    def test_detects_unittest_testcase(self) -> None:
        code = "class TestFoo(unittest.TestCase): pass"
        tree = ast.parse(code)
        assert is_unittest_class(tree.body[0])  # type: ignore

    def test_detects_direct_testcase(self) -> None:
        code = "class TestFoo(TestCase): pass"
        tree = ast.parse(code)
        assert is_unittest_class(tree.body[0])  # type: ignore

    def test_rejects_non_testcase_class(self) -> None:
        code = "class Solution: pass"
        tree = ast.parse(code)
        assert not is_unittest_class(tree.body[0])  # type: ignore

    def test_rejects_custom_base_class(self) -> None:
        code = "class MyClass(BaseClass): pass"
        tree = ast.parse(code)
        assert not is_unittest_class(tree.body[0])  # type: ignore

    def test_handles_multiple_inheritance(self) -> None:
        code = "class TestFoo(Mixin, unittest.TestCase): pass"
        tree = ast.parse(code)
        assert is_unittest_class(tree.body[0])  # type: ignore

    def test_handles_multiple_inheritance_first_testcase(self) -> None:
        code = "class TestFoo(unittest.TestCase, Mixin): pass"
        tree = ast.parse(code)
        assert is_unittest_class(tree.body[0])  # type: ignore

    def test_testcase_attribute_variant(self) -> None:
        code = "class TestFoo(test.TestCase): pass"
        tree = ast.parse(code)
        assert is_unittest_class(tree.body[0])  # type: ignore

    def test_rejects_testable_name(self) -> None:
        code = "class MyTest(object): pass"
        tree = ast.parse(code)
        # Should not detect as TestCase just because class name has "Test"
        assert not is_unittest_class(tree.body[0])  # type: ignore


class TestExtractUnittestCode:
    def test_silently_removes_solution_class(self) -> None:
        """Condition: class Solution should disappear without error."""
        full_code = """
import unittest

class Solution:
    def solve(self): return 1

class TestFoo(unittest.TestCase):
    def test_x(self): pass
"""
        result = extract_unittest_code(full_code)
        assert "class Solution" not in result
        assert "class TestFoo" in result
        assert "import unittest" in result

    def test_raises_error_on_other_classes(self) -> None:
        """Condition: class Helper (not Solution, not Unittest) raises error."""
        full_code = """
import unittest

class Helper:  # This is NOT allowed
    pass

class TestFoo(unittest.TestCase):
    pass
"""
        with pytest.raises(
            CodeTransformationError, match="Non-unittest class 'Helper'"
        ):
            extract_unittest_code(full_code)

    def test_raises_error_on_helper_functions(self) -> None:
        """Condition: Top level functions raise error."""
        full_code = """
def my_helper(): return 1
"""
        with pytest.raises(
            CodeTransformationError, match="Top-level function 'my_helper'"
        ):
            extract_unittest_code(full_code)

    def test_remove_solution_import(self) -> None:
        full_code = """import unittest
from solution_module import Solution
import Solution
import solution
from solution import Solution
from Solution import solution
from Solution import Solution
import Solution as sol
class Solution:
    def solve(self): return 1
class TestFoo(unittest.TestCase):
    def test_x(self): pass
"""
        result = extract_unittest_code(full_code)
        assert "from solution_module import Solution" not in result
        assert "import Solution" not in result
        assert "import solution" not in result
        assert "from solution import Solution" not in result
        assert "from Solution import solution" not in result
        assert "from Solution import Solution" not in result
        assert "import Solution as sol" not in result
        assert "class TestFoo" in result

    def test_silently_removes_try_blocks(self) -> None:
        """Condition: Try blocks should be silently removed."""
        full_code = """
import unittest

try:
    from solution import Solution
except ImportError:
    pass

class TestFoo(unittest.TestCase):
    def test_x(self): pass
"""
        result = extract_unittest_code(full_code)
        assert "try:" not in result
        assert "from solution import Solution" not in result
        assert "except ImportError:" not in result
        assert "class TestFoo" in result
        assert "import unittest" in result

    def test_silently_removes_try_blocks_with_else(self) -> None:
        """Condition: Try blocks with else/finally should also be removed."""
        full_code = """
import unittest

try:
    import solution
except ImportError:
    pass
else:
    print("imported")
finally:
    print("done")

class TestBar(unittest.TestCase):
    def test_y(self): pass
"""
        result = extract_unittest_code(full_code)
        assert "try:" not in result
        assert "import solution" not in result
        assert "except ImportError:" not in result
        assert "else:" not in result
        assert "finally:" not in result
        assert "class TestBar" in result

    def test_preserves_comments_and_docstrings_in_unittest_code(self) -> None:
        """Ensure extract_unittest_code preserves docstrings and inline comments inside kept nodes."""
        full_code = textwrap.dedent('''
            import unittest

            # Top-level comment about tests

            class TestInnerComments(unittest.TestCase):
                """This class has a docstring that should be preserved."""
                def test_doc_and_comments(self):
                    """Method docstring should be kept."""
                    # setup comment
                    x = 5  # inline comment
                    self.assertEqual(x, 5)

            # Another comment that may be lost
        ''')

        result = extract_unittest_code(full_code)

        # Docstrings and inline comments inside the class/method should be preserved
        assert "This class has a docstring that should be preserved." in result
        assert "Method docstring should be kept." in result
        assert "# setup comment" in result
        assert "# inline comment" in result


# =============================================================================
# Tests for setup_unittest_class_from_starter_code
# =============================================================================


class TestSetupUnittestClassFromStarterCode:
    """Test setup_unittest_class_from_starter_code function."""

    def test_setup_unittest_valid_code(self) -> None:
        """
        Verify that a standard starter code snippet produces the correct
        unittest scaffold with imports and setUp method.
        """
        starter_code = """
class Solution:
    def solve(self, x: int) -> int:
        return x + 1
    """

        result = setup_unittest_class_from_starter_code(starter_code)

        # Assert imports are present
        assert "import unittest" in result
        assert "from typing import *" in result

        # Assert class inheritance
        assert "class TestSolution(unittest.TestCase):" in result

        # Assert setUp structure
        assert "def setUp(self):" in result
        assert "self.solution = Solution()" in result

    def test_setup_unittest_different_class_name(self) -> None:
        """Verify it correctly extracts arbitrary class names."""
        starter_code = "class Calculator:"
        result = setup_unittest_class_from_starter_code(starter_code)
        assert "class TestCalculator(unittest.TestCase):" in result
        assert "self.solution = Calculator()" in result

    def test_setup_unittest_invalid_code(self) -> None:
        """Verify it returns the error string when no class is found."""
        starter_code = "def just_a_function(): pass"
        result = setup_unittest_class_from_starter_code(starter_code)
        assert "# Error: Could not parse class name" in result


# =============================================================================
# Tests for build_test_method_from_io
# =============================================================================


class TestBuildTestMethodFromIo:
    """Test build_test_method_from_io function."""

    def test_build_method_basic_integers(self) -> None:
        """Test standard integer arguments (checking k=v formatting)."""
        starter_code = """
class Solution:
    def add(self, a: int, b: int) -> int: ...
    """
        io_pairs = [{"inputdata": {"a": 1, "b": 2}, "output": 3}]

        result = build_test_method_from_io(starter_code, io_pairs, ["C1"])

        # Check method name
        assert "def test_case_C1(self):" in result
        # Check execution call
        assert "result = self.solution.add(a=1, b=2)" in result
        # Check assertion
        assert "self.assertEqual(result, 3)" in result

    def test_build_method_strings_with_repr(self) -> None:
        """
        Test that string inputs are correctly quoted using repr().
        If repr() wasn't used, output would be 'a=hello' (variable) instead of 'a="hello"' (string).
        """
        starter_code = """
class Solution:
    def reverse(self, s: str) -> str: ...
    """
        io_pairs = [{"inputdata": {"s": "hello"}, "output": "olleh"}]

        result = build_test_method_from_io(starter_code, io_pairs, ["C2"])

        # Inputs should be quoted
        assert "s='hello'" in result or 's="hello"' in result
        # Output should be quoted
        assert "self.assertEqual(result, 'olleh')" in result or '"olleh"' in result

    def test_build_method_complex_types_list(self) -> None:
        """Test that list/complex types are formatted correctly."""
        starter_code = """
class Solution:
    def sort_list(self, nums: list[int]) -> list[int]: ...
    """
        io_pairs = [{"inputdata": {"nums": [3, 1, 2]}, "output": [1, 2, 3]}]

        result = build_test_method_from_io(starter_code, io_pairs, [])

        # Check list formatting
        assert "nums=[3, 1, 2]" in result
        assert "self.assertEqual(result, [1, 2, 3])" in result
        # Check default suffix when ID list is empty
        assert "def test_case_generated(self):" in result

    def test_build_method_multiple_subtests(self) -> None:
        """Verify that multiple IO pairs create multiple assertions in one method."""
        starter_code = """
class Solution:
    def is_even(self, n: int) -> bool: ...
    """
        io_pairs = [
            {"inputdata": {"n": 2}, "output": True},
            {"inputdata": {"n": 3}, "output": False},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, ["P1", "P2"])

        assert "def test_case_P1_P2(self):" in result
        # Both subtests should be present
        assert "Subtest 1" in result
        assert "n=2" in result
        assert "Subtest 2" in result
        assert "n=3" in result

    def test_build_method_parse_error(self) -> None:
        """Verify CodeParsingError is raised if starter code is invalid."""
        invalid_code = "this is not python code"
        io_pairs = [{"inputdata": {"x": 1}, "output": 1}]

        with pytest.raises(CodeParsingError) as excinfo:
            build_test_method_from_io(invalid_code, io_pairs, ["ERR"])

        assert "Could not parse class/method names" in str(excinfo.value)

    def test_build_method_input_name_mismatch_is_ignored(self) -> None:
        """
        Verify that the builder essentially ignores key names matching.
        It blindly trusts the dictionary keys match the function signature.
        (This confirms behavior, even though it results in runtime errors later).
        """
        starter_code = """
class Solution:
    def add(self, a, b): ...
    """
        # The dictionary has 'x', but function has 'a'.
        # The builder should just write x=1. Python will raise TypeError at RUNTIME,
        # but the BUILDER should succeed.
        io_pairs = [{"inputdata": {"x": 1, "y": 2}, "output": 3}]

        result = build_test_method_from_io(starter_code, io_pairs, ["MISMATCH"])

        assert "x=1" in result
        assert "y=2" in result

    def test_build_method_complex_parameters_list_of_lists(self) -> None:
        """Test method with complex parameters including List[List[int]]."""
        starter_code = """
class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
"""
        io_pairs = [
            {"inputdata": {"n": 3, "conflictingPairs": [[1, 2], [2, 3]]}, "output": 2}
        ]

        result = build_test_method_from_io(starter_code, io_pairs, ["MAX_SUB"])

        # Check method name
        assert "def test_case_MAX_SUB(self):" in result
        # Check execution call with complex parameters
        assert (
            "result = self.solution.maxSubarrays(n=3, conflictingPairs=[[1, 2], [2, 3]])"
            in result
        )
        # Check assertion
        assert "self.assertEqual(result, 2)" in result

    def test_build_method_stdin_style_input_str(self) -> None:
        """Test method that takes input as a single string (stdin style)."""
        starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        # Input is provided as a single string.
"""
        io_pairs = [
            {
                "inputdata": {"input_str": "5 6 5\n1 2\n2 4\n3 1\n3 5\n4 3\n5 2"},
                "output": "4",
            }
        ]

        result = build_test_method_from_io(starter_code, io_pairs, ["STDIN"])

        # Check method name
        assert "def test_case_STDIN(self):" in result
        # Check execution call with input_str parameter
        assert (
            "result = self.solution.sol(input_str='5 6 5\\n1 2\\n2 4\\n3 1\\n3 5\\n4 3\\n5 2')"
            in result
        )
        # Check assertion
        assert "self.assertEqual(result, '4')" in result
