"""
Tests for differential operator with class-based starter codes.

These tests focus on verifying that IO pair generation and test method building
work correctly for class methods with various parameter types (not just simple
input_str/output_str based problems).
"""

from unittest.mock import MagicMock, patch

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

    @patch("coevolution.strategies.operators.differential_llm_operator.transformation")
    def test_simple_two_int_parameters(
        self, mock_transform: MagicMock, operator: DifferentialLLMOperator
    ) -> None:
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
        operator.get_test_method_from_io(starter_code, io_pairs, ["P1", "P2"], 0)

        # Verify transformation.build_test_method_from_io was called correctly
        mock_transform.build_test_method_from_io.assert_called_once()
        call_args = mock_transform.build_test_method_from_io.call_args

        # Check arguments: (starter_code, io_pairs, suffix)
        assert call_args[0][0] == starter_code
        assert call_args[0][1] == io_pairs
        assert call_args[0][2] == "P1_P2_0"

    @patch("coevolution.strategies.operators.differential_llm_operator.transformation")
    def test_list_parameter(
        self, mock_transform: MagicMock, operator: DifferentialLLMOperator
    ) -> None:
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

        operator.get_test_method_from_io(starter_code, io_pairs, ["C1", "C2"], 1)

        mock_transform.build_test_method_from_io.assert_called_once()
        call_args = mock_transform.build_test_method_from_io.call_args

        assert call_args[0][0] == starter_code
        assert call_args[0][1] == io_pairs
        assert call_args[0][2] == "C1_C2_1"

    @patch("coevolution.strategies.operators.differential_llm_operator.transformation")
    def test_nested_list_parameters(
        self, mock_transform: MagicMock, operator: DifferentialLLMOperator
    ) -> None:
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

        operator.get_test_method_from_io(starter_code, io_pairs, ["X1", "X2"], 0)

        mock_transform.build_test_method_from_io.assert_called_once()
        call_args = mock_transform.build_test_method_from_io.call_args

        assert call_args[0][0] == starter_code
        assert call_args[0][1] == io_pairs
        assert call_args[0][2] == "X1_X2_0"

    @patch("coevolution.strategies.operators.differential_llm_operator.transformation")
    def test_string_parameters(
        self, mock_transform: MagicMock, operator: DifferentialLLMOperator
    ) -> None:
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

        operator.get_test_method_from_io(starter_code, io_pairs, ["A1", "A2"], 2)

        mock_transform.build_test_method_from_io.assert_called_once()
        call_args = mock_transform.build_test_method_from_io.call_args

        assert call_args[0][0] == starter_code
        assert call_args[0][1] == io_pairs
        assert call_args[0][2] == "A1_A2_2"

    @patch("coevolution.strategies.operators.differential_llm_operator.transformation")
    def test_mixed_parameter_types(
        self, mock_transform: MagicMock, operator: DifferentialLLMOperator
    ) -> None:
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

        operator.get_test_method_from_io(starter_code, io_pairs, ["M1", "M2"], 5)

        mock_transform.build_test_method_from_io.assert_called_once()
        call_args = mock_transform.build_test_method_from_io.call_args

        assert call_args[0][0] == starter_code
        assert call_args[0][1] == io_pairs
        assert call_args[0][2] == "M1_M2_5"

    @patch("coevolution.strategies.operators.differential_llm_operator.transformation")
    def test_empty_io_pairs_list(
        self, mock_transform: MagicMock, operator: DifferentialLLMOperator
    ) -> None:
        """Test that empty IO pairs list is handled gracefully."""
        starter_code = """
class Solution:
    def dummy(self, x: int) -> int:
        pass
"""
        io_pairs: list[DifferentialInputOutput] = []

        operator.get_test_method_from_io(starter_code, io_pairs, ["E1", "E2"], 0)

        mock_transform.build_test_method_from_io.assert_called_once()
        call_args = mock_transform.build_test_method_from_io.call_args

        assert call_args[0][0] == starter_code
        assert call_args[0][1] == []
        assert call_args[0][2] == "E1_E2_0"

    @patch("coevolution.strategies.operators.differential_llm_operator.transformation")
    def test_dict_parameter(
        self, mock_transform: MagicMock, operator: DifferentialLLMOperator
    ) -> None:
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

        operator.get_test_method_from_io(starter_code, io_pairs, ["D1", "D2"], 0)

        mock_transform.build_test_method_from_io.assert_called_once()
        call_args = mock_transform.build_test_method_from_io.call_args

        assert call_args[0][0] == starter_code
        assert call_args[0][1] == io_pairs
        assert call_args[0][2] == "D1_D2_0"
