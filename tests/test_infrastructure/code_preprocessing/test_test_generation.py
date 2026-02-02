"""Tests for generic test generation from Test dataclass and starter code."""

import pytest

from infrastructure.code_preprocessing.exceptions import CodeParsingError
from infrastructure.code_preprocessing.test_generation import (
    MethodSignature,
    generate_pytest_test,
    is_stdin_signature,
    parse_method_signature,
)


class TestParseMethodSignature:
    """Tests for parse_method_signature function."""

    def test_stdin_signature(self):
        """Test parsing STDIN-style signature."""
        starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""
        sig = parse_method_signature(starter_code)

        assert sig.class_name == "Solution"
        assert sig.method_name == "sol"
        assert len(sig.params) == 1
        assert sig.params[0] == ("input_str", "str")
        assert sig.return_type == "str"

    def test_functional_signature(self):
        """Test parsing FUNCTIONAL-style signature with multiple typed params."""
        starter_code = """
class Solution:
    def add(self, x: int, y: int) -> int:
        pass
"""
        sig = parse_method_signature(starter_code)

        assert sig.class_name == "Solution"
        assert sig.method_name == "add"
        assert len(sig.params) == 2
        assert sig.params[0] == ("x", "int")
        assert sig.params[1] == ("y", "int")
        assert sig.return_type == "int"

    def test_incomplete_signature(self):
        """Test parsing incomplete starter code (missing pass)."""
        starter_code = """
class Solution:
    def calculate(self, value: float) -> float:
"""
        # Should auto-complete with pass
        sig = parse_method_signature(starter_code)

        assert sig.class_name == "Solution"
        assert sig.method_name == "calculate"
        assert len(sig.params) == 1
        assert sig.params[0] == ("value", "float")
        assert sig.return_type == "float"

    def test_no_type_annotations(self):
        """Test parsing signature without type annotations."""
        starter_code = """
class Solution:
    def process(self, data):
        pass
"""
        sig = parse_method_signature(starter_code)

        assert sig.class_name == "Solution"
        assert sig.method_name == "process"
        assert len(sig.params) == 1
        assert sig.params[0] == ("data", None)
        assert sig.return_type is None

    def test_no_class_raises_error(self):
        """Test that missing class raises CodeParsingError."""
        starter_code = """
def standalone_function():
    pass
"""
        with pytest.raises(CodeParsingError, match="No class definition found"):
            parse_method_signature(starter_code)

    def test_no_instance_method_raises_error(self):
        """Test that class without instance method raises error."""
        starter_code = """
class Solution:
    @staticmethod
    def static_method():
        pass
"""
        with pytest.raises(CodeParsingError, match="No instance method found"):
            parse_method_signature(starter_code)

    def test_multiple_methods_selects_first(self):
        """Test that multiple methods selects the first instance method."""
        starter_code = """
class Solution:
    def first_method(self, x: int) -> int:
        pass
    
    def second_method(self, y: str) -> str:
        pass
"""
        sig = parse_method_signature(starter_code)

        # Should select the first instance method
        assert sig.method_name == "first_method"
        assert sig.params[0] == ("x", "int")


class TestIsStdinSignature:
    """Tests for is_stdin_signature function."""

    def test_valid_stdin_signature(self):
        """Test that valid STDIN signature is recognized."""
        sig = MethodSignature(
            class_name="Solution",
            method_name="sol",
            params=[("input_str", "str")],
            return_type="str",
        )
        assert is_stdin_signature(sig) is True

    def test_wrong_param_name(self):
        """Test that wrong parameter name is not STDIN."""
        sig = MethodSignature(
            class_name="Solution",
            method_name="sol",
            params=[("data", "str")],
            return_type="str",
        )
        assert is_stdin_signature(sig) is False

    def test_multiple_params(self):
        """Test that multiple parameters is not STDIN."""
        sig = MethodSignature(
            class_name="Solution",
            method_name="sol",
            params=[("input_str", "str"), ("other", "int")],
            return_type="str",
        )
        assert is_stdin_signature(sig) is False

    def test_wrong_param_type(self):
        """Test that wrong parameter type is not STDIN."""
        sig = MethodSignature(
            class_name="Solution",
            method_name="sol",
            params=[("input_str", "int")],
            return_type="str",
        )
        assert is_stdin_signature(sig) is False

    def test_wrong_return_type(self):
        """Test that wrong return type is not STDIN."""
        sig = MethodSignature(
            class_name="Solution",
            method_name="sol",
            params=[("input_str", "str")],
            return_type="int",
        )
        assert is_stdin_signature(sig) is False


class TestGeneratePytestTest:
    """Tests for generate_pytest_test function."""

    def test_stdin_test_generation(self):
        """Test generating pytest test for STDIN-style problem."""
        starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""
        input_data = "5\n3 2 1 4 5"
        output_data = "1 2 3 4 5"

        test_code = generate_pytest_test(input_data, output_data, starter_code, 1)

        # Verify structure
        assert "def test_case_1():" in test_code
        assert "solution = Solution()" in test_code
        assert "input_str = " in test_code
        assert "expected_output = " in test_code
        assert "assert solution.sol(input_str) == expected_output" in test_code

        # Verify input and output are properly quoted
        assert repr(input_data.rstrip("\n")) in test_code
        assert repr(output_data.rstrip("\n")) in test_code

    def test_functional_test_generation(self):
        """Test generating pytest test for FUNCTIONAL-style problem."""
        starter_code = """
class Solution:
    def add(self, x: int, y: int) -> int:
        pass
"""
        input_data = "5\n3"
        output_data = "8"

        test_code = generate_pytest_test(input_data, output_data, starter_code, 1)

        # Verify structure
        assert "def test_case_1():" in test_code
        assert "import ast" in test_code
        assert "solution = Solution()" in test_code
        assert "input_lines = " in test_code
        assert "args = [ast.literal_eval(line) for line in input_lines]" in test_code
        assert "expected_output = ast.literal_eval(" in test_code
        assert "assert solution.add(*args) == expected_output" in test_code

    def test_functional_with_multiple_args(self):
        """Test functional test with multiple arguments."""
        starter_code = """
class Solution:
    def calculate(self, a: int, b: int, c: int) -> int:
        pass
"""
        input_data = "1\n2\n3"
        output_data = "6"

        test_code = generate_pytest_test(input_data, output_data, starter_code, 42)

        assert "def test_case_42():" in test_code
        assert "solution.calculate(*args)" in test_code

    def test_stdin_multiline_input(self):
        """Test STDIN test with multiline input."""
        starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""
        input_data = "3\\nline1\\nline2\\nline3"
        output_data = "line3\\nline2\\nline1"

        test_code = generate_pytest_test(input_data, output_data, starter_code, 1)

        # Should have proper string representation
        assert "def test_case_1():" in test_code
        assert "solution.sol(input_str)" in test_code

    def test_invalid_starter_code_raises_error(self):
        """Test that invalid starter code raises CodeParsingError."""
        starter_code = "not valid python code {"

        with pytest.raises(CodeParsingError):
            generate_pytest_test("1", "2", starter_code, 1)

    def test_different_class_names(self):
        """Test that different class names are preserved."""
        starter_code = """
class MySolution:
    def compute(self, x: int) -> int:
        pass
"""
        test_code = generate_pytest_test("5", "25", starter_code, 1)

        assert "solution = MySolution()" in test_code
        assert "solution.compute(*args)" in test_code


class TestIntegration:
    """Integration tests for the test generation pipeline."""

    def test_stdin_round_trip(self):
        """Test that generated STDIN test can be executed."""
        starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        # Simple echo implementation
        return input_str
"""

        test_code = generate_pytest_test("hello world", "hello world", starter_code, 1)

        # Execute the generated test
        exec_globals = {}
        exec(starter_code + "\n" + test_code, exec_globals)

        # Call the test function
        exec_globals["test_case_1"]()  # Should not raise

    def test_functional_round_trip(self):
        """Test that generated FUNCTIONAL test can be executed."""
        starter_code = """
class Solution:
    def add(self, x: int, y: int) -> int:
        return x + y
"""

        test_code = generate_pytest_test("5\n3", "8", starter_code, 1)

        # Execute the generated test
        exec_globals = {}
        exec(starter_code + "\n" + test_code, exec_globals)

        # Call the test function
        exec_globals["test_case_1"]()  # Should not raise
