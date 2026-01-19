"""
Tests demonstrating the advantages of ast.literal_eval over str() conversion.

These tests show how the current implementation catches type errors and
representation issues that would be missed with string-based comparison.
"""

import pytest

from infrastructure.code_preprocessing.transformation import build_test_method_from_io


class TestLCBStdinStringOutputs:
    """Test that LCB STDIN-style problems preserve string outputs."""

    def test_lcb_stdin_preserves_string_outputs(self) -> None:
        """Verify that when method returns str, outputs are kept as strings."""
        # LCB STDIN problem signature: def sol(self, input_str: str) -> str
        starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""
        # Output from sandbox is string "70000000070" (numeric string)
        # This should NOT be converted to int because return type is str
        io_pairs = [
            {
                "inputdata": {"input_str": "71 70 1000000000\\n2 1\\n..."},
                "output": "70000000070",
            }
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "STDIN_TEST")

        # Should generate: self.assertEqual(result, "70000000070")
        # NOT: self.assertEqual(result, 70000000070)
        assert (
            'self.assertEqual(result, "70000000070")' in result
            or "self.assertEqual(result, '70000000070')" in result
        )
        assert "self.assertEqual(result, 70000000070)" not in result

    def test_lcb_stdin_with_non_numeric_string(self) -> None:
        """Verify string preservation with non-numeric strings."""
        starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""
        io_pairs = [{"inputdata": {"input_str": "hello\\nworld\\n"}, "output": "HELLO"}]

        result = build_test_method_from_io(starter_code, io_pairs, "TEXT_TEST")

        # Should preserve as string
        assert (
            'self.assertEqual(result, "HELLO")' in result
            or "self.assertEqual(result, 'HELLO')" in result
        )

    def test_lcb_stdin_with_boolean_looking_string(self) -> None:
        """Verify that 'True'/'False' strings are preserved when return type is str."""
        starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""
        io_pairs = [
            {"inputdata": {"input_str": "1\\n"}, "output": "True"},
            {"inputdata": {"input_str": "0\\n"}, "output": "False"},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "BOOL_STR_TEST")

        # Should generate string comparisons, not boolean
        assert (
            'self.assertEqual(result, "True")' in result
            or "self.assertEqual(result, 'True')" in result
        )
        assert (
            'self.assertEqual(result, "False")' in result
            or "self.assertEqual(result, 'False')" in result
        )
        # Should NOT generate boolean comparisons
        assert "self.assertEqual(result, True)" not in result
        assert "self.assertEqual(result, False)" not in result


class TestTypeSafetyAdvantages:
    """Tests showing why ast.literal_eval is better than str() conversion."""

    def test_integer_vs_string_type_checking(self) -> None:
        """Verify that integer outputs are compared as integers, not strings."""
        starter_code = """
class Solution:
    def compute(self, x: int) -> int:
        pass
"""
        # Output from sandbox is string "42"
        io_pairs = [{"inputdata": {"x": 7}, "output": "42"}]

        result = build_test_method_from_io(starter_code, io_pairs, "TYPE_TEST")

        # Should generate: self.assertEqual(result, 42) not self.assertEqual(result, "42")
        assert "self.assertEqual(result, 42)" in result
        assert 'self.assertEqual(result, "42")' not in result

    def test_list_type_checking(self) -> None:
        """Verify that list outputs are compared as lists, not strings."""
        starter_code = """
class Solution:
    def getList(self) -> list[int]:
        pass
"""
        # Output from sandbox is string "[1, 2, 3]"
        io_pairs = [{"inputdata": {}, "output": "[1, 2, 3]"}]

        result = build_test_method_from_io(starter_code, io_pairs, "LIST_TEST")

        # Should generate: self.assertEqual(result, [1, 2, 3])
        assert "self.assertEqual(result, [1, 2, 3])" in result
        assert 'self.assertEqual(result, "[1, 2, 3]")' not in result

    def test_boolean_type_checking(self) -> None:
        """Verify that boolean outputs are compared as booleans, not strings."""
        starter_code = """
class Solution:
    def check(self, x: int) -> bool:
        pass
"""
        # Output from sandbox is string "True"
        io_pairs = [
            {"inputdata": {"x": 1}, "output": "True"},
            {"inputdata": {"x": 0}, "output": "False"},
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "BOOL_TEST")

        # Should generate: self.assertEqual(result, True) and self.assertEqual(result, False)
        assert "self.assertEqual(result, True)" in result
        assert "self.assertEqual(result, False)" in result
        # Not string comparisons
        assert 'self.assertEqual(result, "True")' not in result
        assert 'self.assertEqual(result, "False")' not in result

    def test_none_type_checking(self) -> None:
        """Verify that None outputs are compared as None, not strings."""
        starter_code = """
class Solution:
    def maybeReturn(self, x: int) -> int | None:
        pass
"""
        # Output from sandbox is string "None"
        io_pairs = [{"inputdata": {"x": 0}, "output": "None"}]

        result = build_test_method_from_io(starter_code, io_pairs, "NONE_TEST")

        # Should generate: self.assertEqual(result, None)
        assert "self.assertEqual(result, None)" in result
        assert 'self.assertEqual(result, "None")' not in result

    def test_nested_structure_type_checking(self) -> None:
        """Verify that nested structures maintain proper types."""
        starter_code = """
class Solution:
    def process(self, n: int) -> dict[str, list[int]]:
        pass
"""
        # Output from sandbox is string representation of nested structure
        io_pairs = [{"inputdata": {"n": 3}, "output": "{'data': [1, 2, 3]}"}]

        result = build_test_method_from_io(starter_code, io_pairs, "NESTED_TEST")

        # Should generate proper dict comparison, not string
        assert "self.assertEqual(result, {'data': [1, 2, 3]})" in result
        assert "self.assertEqual(result, \"{'data': [1, 2, 3]}\")" not in result

    def test_actual_string_value_preserved(self) -> None:
        """Verify that actual string values work correctly when method returns str."""
        starter_code = """
class Solution:
    def greet(self, name: str) -> str:
        pass
"""
        # When the method returns str and sandbox uses print(result),
        # the output is the string value WITHOUT quotes: "Hello World"
        # Since the method return type is str, we should preserve it as a string
        io_pairs = [{"inputdata": {"name": "World"}, "output": "Hello World"}]

        result = build_test_method_from_io(starter_code, io_pairs, "STRING_TEST")

        # Should preserve the string value
        assert (
            "self.assertEqual(result, 'Hello World')" in result
            or 'self.assertEqual(result, "Hello World")' in result
        )

    def test_mixed_types_in_single_method(self) -> None:
        """Verify that multiple IO pairs with different types work correctly."""
        starter_code = """
class Solution:
    def polymorphic(self, x: int) -> int | str | None:
        pass
"""
        io_pairs = [
            {"inputdata": {"x": 1}, "output": "42"},  # int
            {"inputdata": {"x": 2}, "output": "'text'"},  # string
            {"inputdata": {"x": 3}, "output": "None"},  # None
            {"inputdata": {"x": 4}, "output": "True"},  # bool
        ]

        result = build_test_method_from_io(starter_code, io_pairs, "POLY_TEST")

        # Each type should be properly handled
        assert "self.assertEqual(result, 42)" in result
        assert "self.assertEqual(result, 'text')" in result
        assert "self.assertEqual(result, None)" in result
        assert "self.assertEqual(result, True)" in result

    def test_spacing_differences_dont_matter(self) -> None:
        """
        Verify that spacing differences in list/dict representations don't matter.

        With ast.literal_eval, "[1,2,3]" and "[1, 2, 3]" both become the same list.
        With str(), they would be different strings and cause test failures.
        """
        starter_code = """
class Solution:
    def getList(self) -> list[int]:
        pass
"""
        # Sandbox might output without spaces
        io_pairs_no_space = [{"inputdata": {}, "output": "[1,2,3]"}]
        result_no_space = build_test_method_from_io(
            starter_code, io_pairs_no_space, "NO_SPACE"
        )

        # Should generate the same comparison regardless of spacing
        assert "self.assertEqual(result, [1, 2, 3])" in result_no_space

        # Both formats should produce identical test code
        io_pairs_with_space = [{"inputdata": {}, "output": "[1, 2, 3]"}]
        result_with_space = build_test_method_from_io(
            starter_code, io_pairs_with_space, "WITH_SPACE"
        )

        # Extract just the assertion lines to compare
        no_space_assertion = [
            line for line in result_no_space.split("\n") if "assertEqual" in line
        ][0]
        with_space_assertion = [
            line for line in result_with_space.split("\n") if "assertEqual" in line
        ][0]

        assert no_space_assertion == with_space_assertion

    def test_handles_unparseable_strings_gracefully(self) -> None:
        """
        Verify that if a string can't be parsed by ast.literal_eval,
        it's kept as a string (fallback behavior).
        """
        starter_code = """
class Solution:
    def special(self) -> str:
        pass
"""
        # This is not valid Python literal syntax
        io_pairs = [{"inputdata": {}, "output": "some random text"}]

        result = build_test_method_from_io(starter_code, io_pairs, "UNPARSEABLE")

        # Should fall back to treating it as a string
        assert "self.assertEqual(result, 'some random text')" in result
