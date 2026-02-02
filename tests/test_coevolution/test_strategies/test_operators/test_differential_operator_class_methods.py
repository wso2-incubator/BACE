"""
Tests for differential operator with class-based starter codes.

These tests focus on verifying that IO pair generation and test method building
work correctly for class methods with various parameter types (not just simple
input_str/output_str based problems).
"""

from unittest.mock import MagicMock

import pytest

from coevolution.strategies.operators.differential_llm_operator import (
    DifferentialInputOutput,
    DifferentialLLMOperator,
)


@pytest.fixture
def mock_llm() -> MagicMock:
    """Returns a mock LLM instance."""
    llm = MagicMock()
    llm.generate.return_value = "```python\ndef generate_test_inputs(n): pass\n```"
    return llm


@pytest.fixture
def operator(mock_llm: MagicMock) -> DifferentialLLMOperator:
    """Returns the DifferentialLLMOperator instance with mocked LLM."""
    return DifferentialLLMOperator(llm=mock_llm)


class TestClassMethodIOPairs:
    """Test suite for class-based methods with various parameter types."""

    def test_simple_two_int_parameters(self, operator: DifferentialLLMOperator) -> None:
        """Test method with two integer parameters (l, r)."""
        starter_code = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        pass
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"l": 1, "r": 10}, "output": 2},
            {"inputdata": {"l": 10, "r": 20}, "output": 5},
        ]

        # Call the method
        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["P1", "P2"], 0
        )

        # Verify it generates a valid pytest test
        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result

    def test_list_parameter(self, operator: DifferentialLLMOperator) -> None:
        """Test method with list parameter."""
        starter_code = """
class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        pass
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"nums": [3, 1, 2]}, "output": [1, 2, 3]},
            {"inputdata": {"nums": [5, 4, 3, 2, 1]}, "output": [1, 2, 3, 4, 5]},
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["C1", "C2"], 1
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result

    def test_nested_list_parameters(self, operator: DifferentialLLMOperator) -> None:
        """Test method with nested list parameters (List[List[int]])."""
        starter_code = """
class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        pass
"""
        io_pairs: list[DifferentialInputOutput] = [
            {
                "inputdata": {"n": 3, "conflictingPairs": [[1, 2], [2, 3]]},
                "output": 2,
            },
            {
                "inputdata": {"n": 5, "conflictingPairs": [[1, 2], [3, 4], [4, 5]]},
                "output": 3,
            },
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["X1", "X2"], 0
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result

    def test_string_parameters(self, operator: DifferentialLLMOperator) -> None:
        """Test method with string parameters."""
        starter_code = """
class Solution:
    def isPalindrome(self, s: str) -> bool:
        pass
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"s": "racecar"}, "output": True},
            {"inputdata": {"s": "hello"}, "output": False},
            {"inputdata": {"s": "A man a plan a canal Panama"}, "output": True},
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["A1", "A2"], 2
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result

    def test_mixed_parameter_types(self, operator: DifferentialLLMOperator) -> None:
        """Test method with mixed parameter types (int, string, list)."""
        starter_code = """
class Solution:
    def findSubstring(self, s: str, k: int, words: list[str]) -> list[int]:
        pass
"""
        io_pairs: list[DifferentialInputOutput] = [
            {
                "inputdata": {"s": "barfoothefoobar", "k": 3, "words": ["foo", "bar"]},
                "output": [0, 9],
            },
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["M1", "M2"], 5
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result

    def test_empty_io_pairs_list(self, operator: DifferentialLLMOperator) -> None:
        """Test that empty IO pairs list raises ValueError."""
        starter_code = """
class Solution:
    def dummy(self, x: int) -> int:
        pass
"""
        io_pairs: list[DifferentialInputOutput] = []

        with pytest.raises(ValueError, match="IO pairs list cannot be empty"):
            operator.get_test_method_from_io(starter_code, io_pairs, ["E1", "E2"], 0)

    def test_dict_parameter(self, operator: DifferentialLLMOperator) -> None:
        """Test method with dict parameter."""
        starter_code = """
class Solution:
    def processData(self, data: dict[str, int]) -> int:
        pass
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"data": {"a": 1, "b": 2, "c": 3}}, "output": 6},
            {"inputdata": {"data": {"x": 10}}, "output": 10},
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["D1", "D2"], 0
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result


class TestStandaloneFunctionIOPairs:
    """Test suite for standalone functions (not class methods)."""

    def test_simple_standalone_function(
        self, operator: DifferentialLLMOperator
    ) -> None:
        """Test standalone function with two integer parameters."""
        starter_code = """def add(x: int, y: int) -> int:
    return x + y
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"x": 5, "y": 3}, "output": 8},
            {"inputdata": {"x": 10, "y": 20}, "output": 30},
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["S1", "S2"], 0
        )

        # Verify it generates a valid pytest test
        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result
        assert "add(" in result

    def test_standalone_function_single_param(
        self, operator: DifferentialLLMOperator
    ) -> None:
        """Test standalone function with single parameter."""
        starter_code = """def square(x: int) -> int:
    return x * x
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"x": 5}, "output": 25},
            {"inputdata": {"x": 7}, "output": 49},
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["Q1", "Q2"], 0
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result
        assert "square(" in result

    def test_standalone_function_list_parameter(
        self, operator: DifferentialLLMOperator
    ) -> None:
        """Test standalone function with list parameter."""
        starter_code = """def sort_array(nums: list[int]) -> list[int]:
    return sorted(nums)
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"nums": [3, 1, 2]}, "output": [1, 2, 3]},
            {"inputdata": {"nums": [5, 4, 3, 2, 1]}, "output": [1, 2, 3, 4, 5]},
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["L1", "L2"], 0
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result
        assert "sort_array(" in result

    def test_standalone_function_string_parameter(
        self, operator: DifferentialLLMOperator
    ) -> None:
        """Test standalone function with string parameter."""
        starter_code = """def reverse_string(s: str) -> str:
    return s[::-1]
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"s": "hello"}, "output": "olleh"},
            {"inputdata": {"s": "world"}, "output": "dlrow"},
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["R1", "R2"], 0
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result
        assert "reverse_string(" in result

    def test_standalone_function_mixed_params(
        self, operator: DifferentialLLMOperator
    ) -> None:
        """Test standalone function with mixed parameter types."""
        starter_code = """def find_in_string(s: str, target: str, start: int) -> int:
    return s.find(target, start)
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"s": "hello world", "target": "o", "start": 0}, "output": 4},
            {"inputdata": {"s": "hello world", "target": "o", "start": 5}, "output": 7},
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["F1", "F2"], 0
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result
        assert "find_in_string(" in result

    def test_standalone_function_nested_list(
        self, operator: DifferentialLLMOperator
    ) -> None:
        """Test standalone function with nested list parameter."""
        starter_code = """def flatten(matrix: list[list[int]]) -> list[int]:
    return [item for row in matrix for item in row]
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"matrix": [[1, 2], [3, 4]]}, "output": [1, 2, 3, 4]},
            {
                "inputdata": {"matrix": [[5], [6, 7], [8, 9, 10]]},
                "output": [5, 6, 7, 8, 9, 10],
            },
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["N1", "N2"], 0
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result
        assert "flatten(" in result

    def test_standalone_function_dict_parameter(
        self, operator: DifferentialLLMOperator
    ) -> None:
        """Test standalone function with dict parameter."""
        starter_code = """def sum_values(data: dict[str, int]) -> int:
    return sum(data.values())
"""
        io_pairs: list[DifferentialInputOutput] = [
            {"inputdata": {"data": {"a": 1, "b": 2, "c": 3}}, "output": 6},
            {"inputdata": {"data": {"x": 10, "y": 20}}, "output": 30},
        ]

        result = operator.get_test_method_from_io(
            starter_code, io_pairs, ["D1", "D2"], 0
        )

        assert isinstance(result, str)
        assert "def test_" in result
        assert "assert" in result
        assert "sum_values(" in result
