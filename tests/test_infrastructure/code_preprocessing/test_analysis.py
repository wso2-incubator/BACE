"""Tests for code_preprocessing.analysis module."""

import pytest

from infrastructure.code_preprocessing.analysis import (
    analyze_code_classes,
    analyze_code_functions,
    analyze_code_imports,
    analyze_test_methods,
    contains_starter_code,
)
from infrastructure.code_preprocessing.exceptions import CodeParsingError


class TestAnalyzeTestMethods:
    """Test analyze_test_methods function."""

    def test_finds_test_methods(self) -> None:
        code = """
class TestFoo(unittest.TestCase):
    def test_bar(self):
        pass
    
    def test_baz(self):
        pass
"""
        methods = analyze_test_methods(code)
        assert len(methods) == 2
        assert "test_bar" in methods
        assert "test_baz" in methods

    def test_ignores_non_test_methods(self) -> None:
        code = """
class TestFoo(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_something(self):
        pass
    
    def helper_method(self):
        pass
"""
        methods = analyze_test_methods(code)
        assert len(methods) == 1
        assert methods == ["test_something"]

    def test_handles_empty_test_class(self) -> None:
        code = """
class TestFoo(unittest.TestCase):
    pass
"""
        methods = analyze_test_methods(code)
        assert methods == []

    def test_raises_error_on_syntax_error(self) -> None:
        code = "class TestFoo(\n    invalid"
        with pytest.raises(CodeParsingError, match="Failed to parse test code"):
            analyze_test_methods(code)

    def test_only_analyzes_first_class(self) -> None:
        code = """
class TestFirst(unittest.TestCase):
    def test_one(self):
        pass

class TestSecond(unittest.TestCase):
    def test_two(self):
        pass
"""
        methods = analyze_test_methods(code)
        assert methods == ["test_one"]

    def test_handles_no_classes(self) -> None:
        code = "def test_standalone():\n    pass"
        methods = analyze_test_methods(code)
        assert methods == []


class TestAnalyzeCodeImports:
    """Test analyze_code_imports function."""

    def test_extracts_simple_imports(self) -> None:
        code = "import os\nimport sys"
        imports = analyze_code_imports(code)
        assert "import os" in imports
        assert "import sys" in imports

    def test_extracts_from_imports(self) -> None:
        code = "from typing import List\nfrom pathlib import Path"
        imports = analyze_code_imports(code)
        assert "from typing import List" in imports
        assert "from pathlib import Path" in imports

    def test_removes_duplicate_imports(self) -> None:
        code = "import os\nimport os\nimport sys"
        imports = analyze_code_imports(code)
        assert imports.count("import os") == 1
        assert "import sys" in imports

    def test_ignores_non_import_code(self) -> None:
        code = "import os\ndef foo(): pass\nclass Bar: pass"
        imports = analyze_code_imports(code)
        assert len(imports) == 1
        assert imports == ["import os"]

    def test_handles_multiline_imports(self) -> None:
        code = "from typing import (\n    List,\n    Dict\n)"
        imports = analyze_code_imports(code)
        # Returns each line of the multiline import
        assert len(imports) == 4
        assert "from typing import (" in imports[0]
        assert (
            "from typing import (List, Dict)" in imports[0]
            or "from typing import (" in imports[0]
        )

    def test_raises_error_on_syntax_error(self) -> None:
        code = "import os\nfrom invalid"
        with pytest.raises(CodeParsingError):
            analyze_code_imports(code)

    def test_handles_no_imports(self) -> None:
        code = "def foo(): pass"
        imports = analyze_code_imports(code)
        assert imports == []


class TestAnalyzeCodeFunctions:
    """Test analyze_code_functions function."""

    def test_finds_function_names(self) -> None:
        code = "def foo():\n    pass\n\ndef bar():\n    pass"
        functions = analyze_code_functions(code)
        assert len(functions) == 2
        assert "foo" in functions
        assert "bar" in functions

    def test_ignores_class_methods(self) -> None:
        code = """
def standalone():
    pass

class MyClass:
    def method(self):
        pass
"""
        functions = analyze_code_functions(code)
        assert functions == ["standalone"]

    def test_handles_no_functions(self) -> None:
        code = "x = 1\ny = 2"
        functions = analyze_code_functions(code)
        assert functions == []

    def test_raises_error_on_syntax_error(self) -> None:
        code = "def foo(\n    invalid"
        with pytest.raises(CodeParsingError):
            analyze_code_functions(code)

    def test_handles_nested_functions(self) -> None:
        code = """
def outer():
    def inner():
        pass
    return inner
"""
        functions = analyze_code_functions(code)
        # Only top-level functions
        assert functions == ["outer"]


class TestAnalyzeCodeClasses:
    """Test analyze_code_classes function."""

    def test_finds_class_names(self) -> None:
        code = "class Foo:\n    pass\n\nclass Bar:\n    pass"
        classes = analyze_code_classes(code)
        assert len(classes) == 2
        assert "Foo" in classes
        assert "Bar" in classes

    def test_ignores_functions(self) -> None:
        code = """
class MyClass:
    pass

def my_function():
    pass
"""
        classes = analyze_code_classes(code)
        assert classes == ["MyClass"]

    def test_handles_no_classes(self) -> None:
        code = "def foo(): pass\nx = 1"
        classes = analyze_code_classes(code)
        assert classes == []

    def test_raises_error_on_syntax_error(self) -> None:
        code = "class Foo(\n    invalid"
        with pytest.raises(CodeParsingError):
            analyze_code_classes(code)

    def test_handles_inheritance(self) -> None:
        code = """
class Base:
    pass

class Derived(Base):
    pass
"""
        classes = analyze_code_classes(code)
        assert "Base" in classes
        assert "Derived" in classes

    def test_handles_nested_classes(self) -> None:
        code = """
class Outer:
    class Inner:
        pass
"""
        classes = analyze_code_classes(code)
        # Only top-level classes
        assert classes == ["Outer"]


class TestContainsStarterCode:
    """Test contains_starter_code function."""

    def test_complete_match(self) -> None:
        """Test exact complete code match."""
        code = """
class Solution:
    def solve(self, x):
        return x * 2
"""
        starter = """
class Solution:
    def solve(self, x):
        return x * 2
"""
        assert contains_starter_code(code, starter) is True

    def test_incomplete_class_definition(self) -> None:
        """Test incomplete class definition (common in LCB)."""
        code = """
class Solution:
    def solve(self, x: int) -> int:
        return x * 2
"""
        starter = "class Solution:\n    def solve(self"
        assert contains_starter_code(code, starter) is True

    def test_incomplete_with_type_hints(self) -> None:
        """Test starter code with partial type hints."""
        code = """
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # implementation
        return [0, 1]
"""
        starter = "class Solution:\n    def twoSum(self, nums: List[int]"
        assert contains_starter_code(code, starter) is True

    def test_multiple_methods_incomplete(self) -> None:
        """Test incomplete starter with multiple method signatures."""
        code = """
class Solution:
    def method1(self, x):
        return x
    
    def method2(self, y):
        return y * 2
"""
        starter = """
class Solution:
    def method1(self
    def method2(self
"""
        assert contains_starter_code(code, starter) is True

    def test_with_comments_and_whitespace(self) -> None:
        """Test that comments and whitespace don't affect matching."""
        code = """
# This is a solution
class Solution:
    # Main method
    def solve(self, x):
        # Return result
        return x * 2
"""
        starter = """
class Solution:
    def solve(self, x):
"""
        assert contains_starter_code(code, starter) is True

    def test_different_class_name_no_match(self) -> None:
        """Test that different class names don't match."""
        code = "class Solution:\n    def solve(self): pass"
        starter = "class DifferentClass:\n    def solve(self"
        assert contains_starter_code(code, starter) is False

    def test_different_method_name_no_match(self) -> None:
        """Test that different method names don't match."""
        code = "class Solution:\n    def solve(self): pass"
        starter = "class Solution:\n    def different_method(self"
        assert contains_starter_code(code, starter) is False

    def test_only_class_signature(self) -> None:
        """Test with just class signature."""
        code = """
class Solution:
    def __init__(self):
        pass
    
    def solve(self, x):
        return x
"""
        starter = "class Solution:"
        assert contains_starter_code(code, starter) is True

    def test_function_not_in_class(self) -> None:
        """Test standalone function starter code."""
        code = """
def helper(x):
    return x * 2

def main():
    print(helper(5))
"""
        starter = "def helper(x"
        assert contains_starter_code(code, starter) is True

    def test_empty_starter_code(self) -> None:
        """Test with empty starter code."""
        code = "class Solution:\n    pass"
        starter = ""
        assert contains_starter_code(code, starter) is True

    def test_starter_code_with_docstring(self) -> None:
        """Test that docstrings are handled correctly."""
        code = '''
class Solution:
    """This is a solution class."""
    def solve(self, x):
        """Solve the problem."""
        return x * 2
'''
        starter = "class Solution:\n    def solve(self, x):"
        assert contains_starter_code(code, starter) is True

    def test_partial_parameters(self) -> None:
        """Test incomplete parameter list."""
        code = "class Solution:\n    def process(self, data, mode, debug=False):\n        return data"
        starter = "class Solution:\n    def process(self, data"
        assert contains_starter_code(code, starter) is True

    def test_no_match_completely_different(self) -> None:
        """Test completely different code returns False."""
        code = "def foo():\n    return 1"
        starter = "class Bar:\n    def baz(self"
        assert contains_starter_code(code, starter) is False
