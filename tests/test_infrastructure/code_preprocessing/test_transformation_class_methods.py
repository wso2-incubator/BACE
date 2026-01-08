"""
Tests for transformation.build_test_method_from_io with class-based starter codes.

Focus on verifying that IO pairs are correctly converted to test methods for
various parameter types beyond simple input_str/output_str patterns.
"""

import pytest

from infrastructure.code_preprocessing import CodeParsingError
from infrastructure.code_preprocessing.transformation import build_test_method_from_io


class TestBuildTestMethodClassMethods:
    """Test suite for building test methods from class-based starter codes."""

    def test_two_int_parameters_beautifulNumbers(self) -> None:
        """Test building method for beautifulNumbers(l: int, r: int)."""
        starter_code = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        pass
"""
        io_pairs = [
            {"inputdata": {"l": 1, "r": 10}, "output": 2},
            {"inputdata": {"l": 10, "r": 20}, "output": 5},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "P1_P2_0")

        # Verify method signature
        assert "def test_case_P1_P2_0(self):" in result

        # Verify first subtest with keyword arguments
        assert "# Subtest 1" in result
        assert "result = self.solution.beautifulNumbers(l=1, r=10)" in result
        assert "self.assertEqual(result, 2)" in result

        # Verify second subtest
        assert "# Subtest 2" in result
        assert "result = self.solution.beautifulNumbers(l=10, r=20)" in result
        assert "self.assertEqual(result, 5)" in result

    def test_list_parameter_sortArray(self) -> None:
        """Test building method for sortArray(nums: list[int])."""
        starter_code = """
class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        pass
"""
        io_pairs = [
            {"inputdata": {"nums": [3, 1, 2]}, "output": [1, 2, 3]},
            {"inputdata": {"nums": [5, 4, 3, 2, 1]}, "output": [1, 2, 3, 4, 5]},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "C1_C2_1")

        assert "def test_case_C1_C2_1(self):" in result

        # Check list formatting with keyword argument
        assert "result = self.solution.sortArray(nums=[3, 1, 2])" in result
        assert "self.assertEqual(result, [1, 2, 3])" in result

        assert "result = self.solution.sortArray(nums=[5, 4, 3, 2, 1])" in result
        assert "self.assertEqual(result, [1, 2, 3, 4, 5])" in result

    def test_nested_list_parameter(self) -> None:
        """Test building method with List[List[int]] parameter."""
        starter_code = """
class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        pass
"""
        io_pairs = [
            {
                "inputdata": {"n": 3, "conflictingPairs": [[1, 2], [2, 3]]},
                "output": 2,
            }
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "X1_X2_0")

        assert "def test_case_X1_X2_0(self):" in result
        assert (
            "result = self.solution.maxSubarrays(n=3, conflictingPairs=[[1, 2], [2, 3]])"
            in result
        )
        assert "self.assertEqual(result, 2)" in result

    def test_string_parameter_isPalindrome(self) -> None:
        """Test building method for isPalindrome(s: str)."""
        starter_code = """
class Solution:
    def isPalindrome(self, s: str) -> bool:
        pass
"""
        io_pairs = [
            {"inputdata": {"s": "racecar"}, "output": True},
            {"inputdata": {"s": "hello"}, "output": False},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "A1_A2_2")

        assert "def test_case_A1_A2_2(self):" in result

        # Check string formatting
        assert "result = self.solution.isPalindrome(s='racecar')" in result
        assert "self.assertEqual(result, True)" in result

        assert "result = self.solution.isPalindrome(s='hello')" in result
        assert "self.assertEqual(result, False)" in result

    def test_mixed_parameter_types(self) -> None:
        """Test building method with mixed parameter types."""
        starter_code = """
class Solution:
    def findSubstring(self, s: str, k: int, words: list[str]) -> list[int]:
        pass
"""
        io_pairs = [
            {
                "inputdata": {"s": "barfoo", "k": 3, "words": ["foo", "bar"]},
                "output": [0, 3],
            }
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "M1_M2_5")

        assert "def test_case_M1_M2_5(self):" in result
        assert (
            "result = self.solution.findSubstring(s='barfoo', k=3, words=['foo', 'bar'])"
            in result
        )
        assert "self.assertEqual(result, [0, 3])" in result

    def test_dict_parameter(self) -> None:
        """Test building method with dict parameter."""
        starter_code = """
class Solution:
    def processData(self, data: dict[str, int]) -> int:
        pass
"""
        io_pairs = [{"inputdata": {"data": {"a": 1, "b": 2, "c": 3}}, "output": 6}]

        result = build_test_method_from_io(starter_code, io_pairs, "D1_D2_0")

        assert "def test_case_D1_D2_0(self):" in result
        assert (
            "result = self.solution.processData(data={'a': 1, 'b': 2, 'c': 3})"
            in result
        )
        assert "self.assertEqual(result, 6)" in result

    def test_single_parameter_single_io_pair(self) -> None:
        """Test with single parameter and single IO pair."""
        starter_code = """
class Solution:
    def square(self, x: int) -> int:
        pass
"""
        io_pairs = [{"inputdata": {"x": 5}, "output": 25}]

        result = build_test_method_from_io(starter_code, io_pairs, "S1_S2_0")

        assert "def test_case_S1_S2_0(self):" in result
        assert "result = self.solution.square(x=5)" in result
        assert "self.assertEqual(result, 25)" in result
        assert "# Subtest 1" in result

    def test_multiple_io_pairs_creates_multiple_assertions(self) -> None:
        """Test that multiple IO pairs create multiple assertions in one method."""
        starter_code = """
class Solution:
    def add(self, a: int, b: int) -> int:
        pass
"""
        io_pairs = [
            {"inputdata": {"a": 1, "b": 2}, "output": 3},
            {"inputdata": {"a": 5, "b": 7}, "output": 12},
            {"inputdata": {"a": 0, "b": 0}, "output": 0},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "ADD_TEST")

        # Single method
        assert result.count("def test_case_") == 1

        # Three subtests
        assert "# Subtest 1" in result
        assert "# Subtest 2" in result
        assert "# Subtest 3" in result

        # All assertions present
        assert "result = self.solution.add(a=1, b=2)" in result
        assert "self.assertEqual(result, 3)" in result
        assert "result = self.solution.add(a=5, b=7)" in result
        assert "self.assertEqual(result, 12)" in result
        assert "result = self.solution.add(a=0, b=0)" in result
        assert "self.assertEqual(result, 0)" in result

    def test_empty_io_pairs_list(self) -> None:
        """Test that empty IO pairs list creates method with no assertions."""
        starter_code = """
class Solution:
    def dummy(self, x: int) -> int:
        pass
"""
        io_pairs = []

        result = build_test_method_from_io(starter_code, io_pairs, "EMPTY")

        assert "def test_case_EMPTY(self):" in result
        # Should have no subtests
        assert "# Subtest" not in result
        assert "assertEqual" not in result

    def test_empty_suffix_uses_default(self) -> None:
        """Test that empty suffix defaults to 'generated'."""
        starter_code = """
class Solution:
    def test(self, x: int) -> int:
        pass
"""
        io_pairs = [{"inputdata": {"x": 1}, "output": 1}]

        result = build_test_method_from_io(starter_code, io_pairs, "")

        assert "def test_case_generated(self):" in result

    def test_invalid_starter_code_raises_error(self) -> None:
        """Test that invalid starter code raises CodeParsingError."""
        invalid_code = "this is not valid python code"
        io_pairs = [{"inputdata": {"x": 1}, "output": 1}]

        with pytest.raises(CodeParsingError) as excinfo:
            build_test_method_from_io(invalid_code, io_pairs, "ERROR")

        assert "Could not parse class/method names" in str(excinfo.value)

    def test_no_class_in_starter_code_raises_error(self) -> None:
        """Test that starter code without a class raises error."""
        no_class_code = "def standalone_function(x): return x"
        io_pairs = [{"inputdata": {"x": 1}, "output": 1}]

        with pytest.raises(CodeParsingError) as excinfo:
            build_test_method_from_io(no_class_code, io_pairs, "NO_CLASS")

        assert "Could not parse class/method names" in str(excinfo.value)

    def test_no_method_in_class_raises_error(self) -> None:
        """Test that class without methods raises error."""
        no_method_code = """
class Solution:
    pass
"""
        io_pairs = [{"inputdata": {"x": 1}, "output": 1}]

        with pytest.raises(CodeParsingError) as excinfo:
            build_test_method_from_io(no_method_code, io_pairs, "NO_METHOD")

        assert "Could not parse class/method names" in str(excinfo.value)

    def test_special_characters_in_string_values(self) -> None:
        """Test that special characters in string values are properly escaped."""
        starter_code = """
class Solution:
    def processString(self, s: str) -> str:
        pass
"""
        io_pairs = [
            {"inputdata": {"s": "Hello\nWorld"}, "output": "Hello World"},
            {"inputdata": {"s": "Tab\tSeparated"}, "output": "Tab Separated"},
            {"inputdata": {"s": "Quote'Test"}, "output": "Quote Test"},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "SPECIAL")

        # repr() should properly escape special characters
        assert "s='Hello\\nWorld'" in result
        assert "s='Tab\\tSeparated'" in result
        # Single quotes in string will be handled by repr()
        assert "'Quote" in result

    def test_none_output_value(self) -> None:
        """Test that None output value is handled correctly."""
        starter_code = """
class Solution:
    def maybeReturn(self, x: int) -> int | None:
        pass
"""
        io_pairs = [
            {"inputdata": {"x": 1}, "output": None},
            {"inputdata": {"x": 2}, "output": 4},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "NONE_TEST")

        assert "self.assertEqual(result, None)" in result
        assert "self.assertEqual(result, 4)" in result

    def test_boolean_output_values(self) -> None:
        """Test that boolean output values are handled correctly."""
        starter_code = """
class Solution:
    def isValid(self, x: int) -> bool:
        pass
"""
        io_pairs = [
            {"inputdata": {"x": 0}, "output": False},
            {"inputdata": {"x": 1}, "output": True},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "BOOL_TEST")

        assert "self.assertEqual(result, False)" in result
        assert "self.assertEqual(result, True)" in result
