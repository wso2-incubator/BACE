"""Tests for code_preprocessing.composition module."""

import pytest

from infrastructure.code_preprocessing.composition import (
    compose_lcb_output_script,
    compose_lcb_test_script,
    compose_pytest_script,
    rebuild_unittest_with_methods,
)
from infrastructure.code_preprocessing.exceptions import (
    CodeParsingError,
    CodeTransformationError,
)


class TestComposeLcbTestScript:
    """Test compose_lcb_test_script function."""

    def test_combines_solution_class_and_tests(self) -> None:
        prog = """
class Solution:
    def solve(self):
        return 42
"""
        test = """
import unittest

class TestSolution(unittest.TestCase):
    def test_solve(self):
        self.assertEqual(Solution().solve(), 42)
"""
        script = compose_lcb_test_script(prog, test)
        assert "class Solution" in script
        assert "class TestSolution" in script

    def test_wraps_functions_into_solution_class(self) -> None:
        prog = "def solve():\n    return 42"
        test = "import unittest\nclass TestSolution(unittest.TestCase): pass"
        script = compose_lcb_test_script(prog, test)
        assert "class Solution" in script
        # Function should be converted to method
        assert "def solve(self)" in script

    def test_removes_solution_imports_from_test(self) -> None:
        prog = "class Solution:\n    pass"
        test = """
from solution import Solution
import unittest

class TestSolution(unittest.TestCase):
    pass
"""
        script = compose_lcb_test_script(prog, test)
        assert "from solution import Solution" not in script

    def test_removes_capitalized_Solution_imports_from_test(self) -> None:
        prog = "class Solution:\n    pass"
        test = """
from Solution import Solution
import unittest
class TestSolution(unittest.TestCase):
    pass
"""
        script = compose_lcb_test_script(prog, test)
        assert "from Solution import Solution" not in script

    def test_adds_unittest_import_if_missing(self) -> None:
        prog = "class Solution:\n    pass"
        test = "class TestSolution:\n    pass"
        script = compose_lcb_test_script(prog, test)
        assert "import unittest" in script

    def test_raises_error_when_no_solution_found(self) -> None:
        prog = "x = 1"  # No Solution class or functions
        test = "import unittest\nclass TestSolution(unittest.TestCase): pass"
        with pytest.raises(CodeTransformationError, match="No Solution class"):
            compose_lcb_test_script(prog, test)

    def test_preserves_helper_classes(self) -> None:
        prog = """
class Helper:
    pass

class Solution:
    pass
"""
        test = "import unittest\nclass TestSolution(unittest.TestCase): pass"
        script = compose_lcb_test_script(prog, test)
        assert "class Helper" in script
        assert "class Solution" in script


class TestRebuildUnittestWithMethods:
    """Test rebuild_unittest_with_methods function."""

    def test_replaces_test_methods(self) -> None:
        original = """
import unittest

class TestFoo(unittest.TestCase):
    def test_old(self):
        pass
"""
        new_methods = ["def test_new(self):\n    self.assertTrue(True)"]
        result = rebuild_unittest_with_methods(original, new_methods)
        assert "test_new" in result
        assert "test_old" not in result

    def test_preserves_imports(self) -> None:
        original = """
import unittest
import os

class TestFoo(unittest.TestCase):
    def test_old(self):
        pass
"""
        new_methods = ["def test_new(self): pass"]
        result = rebuild_unittest_with_methods(original, new_methods)
        assert "import unittest" in result
        assert "import os" in result

    def test_preserves_setup_and_teardown(self) -> None:
        original = """
class TestFoo(unittest.TestCase):
    def setUp(self):
        self.x = 1
    
    def tearDown(self):
        self.x = None
    
    def test_old(self):
        pass
"""
        new_methods = ["def test_new(self): pass"]
        result = rebuild_unittest_with_methods(original, new_methods)
        assert "def setUp" in result
        assert "def tearDown" in result

    def test_handles_multiple_new_methods(self) -> None:
        original = "class TestFoo(unittest.TestCase):\n    def test_old(self): pass"
        new_methods = [
            "def test_one(self): pass",
            "def test_two(self): pass",
            "def test_three(self): pass",
        ]
        result = rebuild_unittest_with_methods(original, new_methods)
        assert "test_one" in result
        assert "test_two" in result
        assert "test_three" in result

    def test_raises_error_when_no_class(self) -> None:
        code = "def foo(): pass"
        with pytest.raises(CodeParsingError, match="No class definition found"):
            rebuild_unittest_with_methods(code, [])

    def test_skips_invalid_method_code(self) -> None:
        original = "class TestFoo(unittest.TestCase):\n    def test_old(self): pass"
        new_methods = [
            "def test_valid(self): pass",
            "invalid syntax here",
            "def test_also_valid(self): pass",
        ]
        result = rebuild_unittest_with_methods(original, new_methods)
        # Should include valid methods, skip invalid
        assert "test_valid" in result
        assert "test_also_valid" in result


class TestComposeLcbOutputScript:
    """Test compose_lcb_output_script function."""

    def test_basic_solution_class_with_simple_params(self) -> None:
        """Test generating output script from Solution class with simple parameters."""
        prog = """
class Solution:
    def add(self, a, b):
        return a + b
"""
        input_data = "{'inputdata': {'a': 1, 'b': 2}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "class Solution" in script
        assert "sol = Solution()" in script
        assert "sol.add(1, 2)" in script
        assert "print(sol.add(1, 2))" in script

    def test_execution_of_generated_script_basic(self) -> None:
        """Test that generated script actually executes and produces correct output."""
        prog = """
class Solution:
    def add(self, a, b):
        return a + b
"""
        input_data = "{'inputdata': {'a': 1, 'b': 2}}"
        script = compose_lcb_output_script(prog, input_data)

        # Capture output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        exec(script)
        sys.stdout = sys.__stdout__

        assert "3" in captured_output.getvalue()

    def test_wraps_loose_functions_into_solution_class(self) -> None:
        """Test that standalone functions are wrapped into Solution class."""
        prog = """
def multiply(x, y):
    return x * y
"""
        input_data = "{'inputdata': {'x': 3, 'y': 4}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "class Solution" in script
        assert "def multiply(self, x, y)" in script
        assert "sol.multiply(3, 4)" in script

    def test_string_parameters(self) -> None:
        """Test handling of string parameters in input data."""
        prog = """
class Solution:
    def process(self, text, count):
        return text * count
"""
        input_data = "{'inputdata': {'text': 'hello', 'count': 3}}"
        script = compose_lcb_output_script(prog, input_data)

        # repr() produces single quotes for strings
        assert "process('hello', 3)" in script

    def test_execution_with_string_parameters(self) -> None:
        """Test execution of generated script with string parameters."""
        prog = """
class Solution:
    def process(self, text, count):
        return text * count
"""
        input_data = "{'inputdata': {'text': 'x', 'count': 5}}"
        script = compose_lcb_output_script(prog, input_data)

        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        exec(script)
        sys.stdout = sys.__stdout__

        assert "xxxxx" in captured_output.getvalue()

    def test_preserves_imports(self) -> None:
        """Test that imports from programmer code are preserved."""
        prog = """
import math

class Solution:
    def sqrt_sum(self, a, b):
        return math.sqrt(a + b)
"""
        input_data = "{'inputdata': {'a': 3, 'b': 6}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "import math" in script

    def test_preserves_helper_classes(self) -> None:
        """Test that helper classes are preserved in the output."""
        prog = """
class Helper:
    def helper_method(self):
        return 42

class Solution:
    def solve(self, x):
        h = Helper()
        return h.helper_method() + x
"""
        input_data = "{'inputdata': {'x': 8}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "class Helper" in script
        assert "class Solution" in script

    def test_multiple_parameters_in_correct_order(self) -> None:
        """Test that parameters are passed in the correct order."""
        prog = """
class Solution:
    def concat(self, a, b, c):
        return a + b + c
"""
        input_data = "{'inputdata': {'a': 'x', 'b': 'y', 'c': 'z'}}"
        script = compose_lcb_output_script(prog, input_data)

        # The order depends on dict iteration, but all values should be present
        assert "concat(" in script
        # repr() produces single quotes for strings
        assert "'x'" in script
        assert "'y'" in script
        assert "'z'" in script

    def test_numeric_types_different_precision(self) -> None:
        """Test handling of different numeric types (int, float)."""
        prog = """
class Solution:
    def calc(self, a, b):
        return a + b
"""
        input_data = "{'inputdata': {'a': 1.5, 'b': 2.7}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "calc(1.5, 2.7)" in script

    def test_execution_with_numeric_precision(self) -> None:
        """Test execution with floating point numbers."""
        prog = """
class Solution:
    def calc(self, a, b):
        return a + b
"""
        input_data = "{'inputdata': {'a': 1.5, 'b': 2.5}}"
        script = compose_lcb_output_script(prog, input_data)

        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        exec(script)
        sys.stdout = sys.__stdout__

        assert "4" in captured_output.getvalue()

    def test_list_as_parameter(self) -> None:
        """Test handling of list as a parameter."""
        prog = """
class Solution:
    def sum_list(self, nums):
        return sum(nums)
"""
        input_data = "{'inputdata': {'nums': [1, 2, 3, 4, 5]}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "sum_list([1, 2, 3, 4, 5])" in script

    def test_dict_as_parameter(self) -> None:
        """Test handling of dict as a parameter."""
        prog = """
class Solution:
    def get_value(self, d):
        return d.get('key', None)
"""
        input_data = "{'inputdata': {'d': {'key': 'value'}}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "get_value({'key': 'value'})" in script

    def test_boolean_parameters(self) -> None:
        """Test handling of boolean parameters."""
        prog = """
class Solution:
    def toggle(self, flag):
        return not flag
"""
        input_data = "{'inputdata': {'flag': True}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "toggle(True)" in script

    def test_none_as_parameter(self) -> None:
        """Test handling of None as a parameter."""
        prog = """
class Solution:
    def check_none(self, x):
        return x is None
"""
        input_data = "{'inputdata': {'x': None}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "check_none(None)" in script

    def test_single_parameter_function(self) -> None:
        """Test function with only one parameter."""
        prog = """
class Solution:
    def square(self, x):
        return x * x
"""
        input_data = "{'inputdata': {'x': 5}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "square(5)" in script

    def test_no_parameters_function(self) -> None:
        """Test function with no parameters (besides self)."""
        prog = """
class Solution:
    def get_answer(self):
        return 42
"""
        input_data = "{'inputdata': {}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "get_answer()" in script

    def test_execution_no_parameters(self) -> None:
        """Test execution of function with no parameters."""
        prog = """
class Solution:
    def get_answer(self):
        return 42
"""
        input_data = "{'inputdata': {}}"
        script = compose_lcb_output_script(prog, input_data)

        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        exec(script)
        sys.stdout = sys.__stdout__

        assert "42" in captured_output.getvalue()

    def test_method_name_detection_in_solution_class(self) -> None:
        """Test that the first non-dunder method is correctly identified."""
        prog = """
class Solution:
    def solve(self, x):
        return x * 2
    
    def other_method(self, y):
        return y * 3
"""
        input_data = "{'inputdata': {'x': 10}}"
        script = compose_lcb_output_script(prog, input_data)

        # Should use the first method (solve)
        assert "sol.solve(10)" in script

    def test_raises_error_on_invalid_programmer_code(self) -> None:
        """Test that CodeParsingError is raised for invalid Python code."""
        prog = "class Solution: invalid syntax here"
        input_data = "{'inputdata': {}}"

        with pytest.raises(CodeParsingError, match="Failed to parse programmer code"):
            compose_lcb_output_script(prog, input_data)

    def test_raises_error_when_no_solution_found(self) -> None:
        """Test that CodeTransformationError is raised when Solution class not found."""
        prog = "x = 1"  # No Solution class or functions
        input_data = "{'inputdata': {}}"

        with pytest.raises(CodeTransformationError, match="No Solution class"):
            compose_lcb_output_script(prog, input_data)

    def test_raises_error_when_no_method_found(self) -> None:
        """Test that CodeTransformationError is raised when no callable method found."""
        prog = "class Solution: pass"  # No methods
        input_data = "{'inputdata': {}}"

        with pytest.raises(CodeTransformationError, match="No callable method"):
            compose_lcb_output_script(prog, input_data)

    def test_raises_error_missing_inputdata_key(self) -> None:
        """Test that error is raised when 'inputdata' key is missing."""
        prog = """
class Solution:
    def solve(self, x):
        return x
"""
        input_data = "{'wrongkey': {'x': 1}}"

        with pytest.raises(CodeTransformationError, match="inputdata"):
            compose_lcb_output_script(prog, input_data)

    def test_raises_error_when_inputdata_not_dict(self) -> None:
        """Test that error is raised when 'inputdata' is not a dictionary."""
        prog = """
class Solution:
    def solve(self, x):
        return x
"""
        input_data = "{'inputdata': 'not a dict'}"

        with pytest.raises(CodeTransformationError, match="dict object"):
            compose_lcb_output_script(prog, input_data)

    def test_raises_error_on_invalid_python_dict(self) -> None:
        """Test that error is raised for invalid Python dict input."""
        prog = """
class Solution:
    def solve(self, x):
        return x
"""
        input_data = "this is not valid python dict"

        with pytest.raises(CodeTransformationError, match="Failed to parse input data"):
            compose_lcb_output_script(prog, input_data)

    def test_complex_nested_structure(self) -> None:
        """Test handling of complex nested data structures."""
        prog = """
class Solution:
    def process(self, data):
        return data['a']['b'] + len(data['c'])
"""
        input_data = "{'inputdata': {'data': {'a': {'b': 10}, 'c': [1, 2, 3]}}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "process(" in script

    def test_execution_complex_nested_structure(self) -> None:
        """Test execution with complex nested data structures."""
        prog = """
class Solution:
    def process(self, data):
        return data['a']['b'] + len(data['c'])
"""
        input_data = "{'inputdata': {'data': {'a': {'b': 10}, 'c': [1, 2, 3]}}}"
        script = compose_lcb_output_script(prog, input_data)

        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        exec(script)
        sys.stdout = sys.__stdout__

        assert "13" in captured_output.getvalue()

    def test_special_method_name_solve(self) -> None:
        """Test common method name 'solve'."""
        prog = """
class Solution:
    def solve(self, n):
        return n * 2
"""
        input_data = "{'inputdata': {'n': 21}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "sol.solve(21)" in script

    def test_special_method_name_lengthOfLongestSubstring(self) -> None:
        """Test camelCase method names."""
        prog = """
class Solution:
    def lengthOfLongestSubstring(self, s):
        return len(s)
"""
        input_data = "{'inputdata': {'s': 'hello'}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "lengthOfLongestSubstring(" in script

    def test_whitespace_handling_in_input_json(self) -> None:
        """Test that whitespace in JSON is handled correctly."""
        prog = """
class Solution:
    def add(self, a, b):
        return a + b
"""
        input_data = "{ 'inputdata' : { 'a' : 1 , 'b' : 2 } }"
        script = compose_lcb_output_script(prog, input_data)

        assert "add(1, 2)" in script

    def test_multiline_string_with_newlines(self) -> None:
        """Test handling of multi-line string input with escaped newlines."""
        prog = """
class Solution:
    def solve(self, input_str):
        lines = input_str.split('\\n')
        return len(lines)
"""
        # Multi-line string input with actual newlines preserved as \\n
        input_data = "{'inputdata': {'input_str': '20\\n1 1 1 1 0 0 1 1 0 0 0 1 0 1 0 1 1 0 1 0\\n0 0 0 1 1 1 0 1 1 0 0 0 0 0 0 1 0 1 0 0\\n52 73 97 72 54 15 79 67 13 55 65 22 36 90 84 46 1 2 27 8'}}"
        script = compose_lcb_output_script(prog, input_data)

        # The composed script should contain the string with escaped newlines
        assert "solve(" in script
        # Check that newlines are preserved in the generated script
        assert "\\n" in script

    def test_execution_multiline_string_with_newlines(self) -> None:
        """Test execution of solution with multi-line string input."""
        prog = """
class Solution:
    def solve(self, input_str):
        lines = input_str.split('\\n')
        return len(lines)
"""
        input_data = "{'inputdata': {'input_str': 'line1\\nline2\\nline3'}}"
        script = compose_lcb_output_script(prog, input_data)

        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        exec(script)
        sys.stdout = sys.__stdout__

        # Should output 3 (three lines)
        assert "3" in captured_output.getvalue()

    def test_complex_multiline_input_with_numbers(self) -> None:
        """Test realistic multi-line input similar to competitive programming."""
        prog = """
class Solution:
    def sol(self, input_str):
        data = list(map(int, input_str.split()))
        return len(data)
"""
        # Realistic input with multiple lines and numbers
        input_data = "{'inputdata': {'input_str': '5\\n1 2 3 4 5\\n10 20 30 40 50'}}"
        script = compose_lcb_output_script(prog, input_data)

        assert "sol(" in script
        # Verify the raw string with newlines is in the script
        assert "\\n" in script

    def test_execution_complex_multiline_input(self) -> None:
        """Test execution with complex multi-line input parsing."""
        prog = """
class Solution:
    def sol(self, input_str):
        lines = input_str.strip().split('\\n')
        n = int(lines[0])
        arr = list(map(int, lines[1].split()))
        return n, len(arr)
"""
        input_data = "{'inputdata': {'input_str': '3\\n1 2 3'}}"
        script = compose_lcb_output_script(prog, input_data)

        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        exec(script)
        sys.stdout = sys.__stdout__

        # Should output (3, 3)
        assert "3" in captured_output.getvalue()

    def test_python_dict_with_string_concatenation(self) -> None:
        """Test handling of Python dict format with string concatenation operators."""
        prog = """
class Solution:
    def solve(self, input_str):
        lines = input_str.strip().split('\\n')
        return len(lines)
"""
        # This is the exact format from the original error - Python dict with + operators
        input_data = """{"inputdata": {"input_str": "21\\n" +
"1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\\n" +
"0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\\n" +
"100 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"}}"""
        script = compose_lcb_output_script(prog, input_data)

        # Should successfully parse and generate script
        assert "class Solution" in script
        assert "sol = Solution()" in script
        assert "solve(" in script

    def test_execution_python_dict_with_string_concatenation(self) -> None:
        """Test execution with Python dict format using string concatenation."""
        prog = """
class Solution:
    def solve(self, input_str):
        lines = input_str.strip().split('\\n')
        return len(lines)
"""
        input_data = """{"inputdata": {"input_str": "21\\n" +
"1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\\n" +
"0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\\n" +
"100 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"}}"""
        script = compose_lcb_output_script(prog, input_data)

        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        exec(script)
        sys.stdout = sys.__stdout__

        # Should output 4 (four lines: "21", line of 1s and 0s, line of 0s and 1s, line of 100 and 1s)
        assert "4" in captured_output.getvalue()


class TestComposePytestScript:
    """Test compose_pytest_script function for individual test execution."""

    def test_combines_solution_class_and_test_function(self) -> None:
        """Test that Solution class and test function are combined correctly."""
        prog = """
class Solution:
    def add(self, a, b):
        return a + b
"""
        test = """
def test_add():
    s = Solution()
    assert s.add(2, 3) == 5
"""
        script = compose_pytest_script(prog, test)
        assert "class Solution" in script
        assert "def test_add():" in script
        assert "def add(self, a, b):" in script

    def test_loose_functions_not_wrapped(self) -> None:
        """Test that standalone functions are left as-is (no wrapping)."""
        prog = """
def multiply(x, y):
    return x * y
"""
        test = """
def test_multiply():
    assert multiply(3, 4) == 12
"""
        script = compose_pytest_script(prog, test)
        # Functions should remain standalone
        assert "def multiply(x, y):" in script
        # No automatic Solution class creation
        assert "class Solution" not in script

    def test_adds_pytest_import(self) -> None:
        """Test that pytest import is added if missing."""
        prog = "def func(): pass"
        test = "def test_something(): pass"
        script = compose_pytest_script(prog, test)
        assert "import pytest" in script

    def test_preserves_pytest_import_if_present(self) -> None:
        """Test that existing pytest import is preserved without duplication."""
        prog = """
import pytest

def func():
    pass
"""
        test = "def test_something(): pass"
        script = compose_pytest_script(prog, test)
        # Should only have one pytest import
        assert script.count("import pytest") == 1

    def test_preserves_helper_classes(self) -> None:
        """Test that helper classes are preserved."""
        prog = """
class Helper:
    def helper_method(self):
        return 'helper'

class Solution:
    def use_helper(self):
        h = Helper()
        return h.helper_method()
"""
        test = """
def test_use_helper():
    s = Solution()
    assert s.use_helper() == 'helper'
"""
        script = compose_pytest_script(prog, test)
        assert "class Helper" in script
        assert "class Solution" in script
        assert "def helper_method(self):" in script

    def test_handles_multiple_test_functions(self) -> None:
        """Test that multiple test functions are included."""
        prog = "class Solution:\n    pass"
        test = """
def test_one():
    pass

def test_two():
    pass
"""
        script = compose_pytest_script(prog, test)
        assert "def test_one():" in script
        assert "def test_two():" in script

    def test_includes_pytest_main_block(self) -> None:
        """Test that pytest.main() execution block is added."""
        prog = "class Solution:\n    pass"
        test = "def test_something():\n    pass"
        script = compose_pytest_script(prog, test)
        assert 'if __name__ == "__main__":' in script
        assert 'pytest.main([__file__, "-v"])' in script

    def test_handles_test_with_imports(self) -> None:
        """Test that test-specific imports are preserved."""
        prog = "def func(): pass"
        test = """
import math

def test_with_math():
    assert math.sqrt(4) == 2
"""
        script = compose_pytest_script(prog, test)
        assert "import math" in script
        assert "def test_with_math():" in script

    def test_preserves_all_imports(self) -> None:
        """Test that all imports from both code and test are preserved."""
        prog = """
import math

def calculate(x):
    return math.sqrt(x)
"""
        test = """
import os

def test_calculate():
    assert calculate(4) == 2
"""
        script = compose_pytest_script(prog, test)
        # Both imports should be present
        assert "import math" in script
        assert "import os" in script

    def test_preserves_programmer_imports(self) -> None:
        """Test that programmer code imports are preserved."""
        prog = """
from typing import List
import collections

def process(items: List[int]):
    return collections.Counter(items)
"""
        test = """
def test_process():
    result = process([1, 2, 2, 3])
    assert result[2] == 2
"""
        script = compose_pytest_script(prog, test)
        assert "from typing import List" in script
        assert "import collections" in script
