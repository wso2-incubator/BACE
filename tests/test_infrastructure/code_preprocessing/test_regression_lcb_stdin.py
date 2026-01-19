"""
Regression test for the specific issue reported:
Differential tests failing because string outputs were being converted to integers.

Issue: When a method signature returns str (like LCB STDIN problems),
the differential test builder was converting numeric string outputs like "70000000070"
to integers using ast.literal_eval. This caused tests to expect integer 70000000070,
but the actual solution returned string "70000000070", causing test failures.

This file contains a test that reproduces the exact scenario from the bug report.
"""

from infrastructure.code_preprocessing.transformation import build_test_method_from_io


def test_regression_lcb_stdin_numeric_string_not_converted() -> None:
    """
    Regression test for the reported issue where test T337 failed.

    The problem was:
    - Solution method: def sol(self, input_str: str) -> str
    - Actual output from code: "70000000070" (string)
    - Test was generated as: self.assertEqual(result, 70000000070) (integer)
    - Test failed because "70000000070" != 70000000070

    After the fix, test should be: self.assertEqual(result, "70000000070") (string)
    """
    starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""

    # This is the exact scenario from the bug report
    # Output is a large numeric string that looks like it could be an integer
    io_pairs = [
        {
            "inputdata": {"input_str": "71 70 1000000000\\n2 1\\n2 3\\n..."},
            "output": "70000000070",
        }
    ]

    test_method = build_test_method_from_io(starter_code, io_pairs, "T337")

    # Verify the test expects a STRING, not an integer
    # Should have: self.assertEqual(result, "70000000070")
    # or: self.assertEqual(result, '70000000070')
    assert (
        'self.assertEqual(result, "70000000070")' in test_method
        or "self.assertEqual(result, '70000000070')" in test_method
    ), f"Test should expect string '70000000070', but got:\\n{test_method}"

    # Verify it does NOT expect an integer
    assert "self.assertEqual(result, 70000000070)" not in test_method, (
        f"Test should NOT expect integer 70000000070, but got:\\n{test_method}"
    )

    print("✓ Regression test passed: Numeric strings preserved for str return type")


def test_non_str_return_type_still_converts() -> None:
    """
    Verify that methods WITHOUT str return type still get type conversion.

    This ensures we didn't break the existing type-safe behavior for methods
    that return int, bool, list, etc.
    """
    starter_code = """
class Solution:
    def compute(self, x: int) -> int:
        pass
"""

    # Output is numeric string that should be converted to int
    io_pairs = [{"inputdata": {"x": 5}, "output": "42"}]

    test_method = build_test_method_from_io(starter_code, io_pairs, "INT_TEST")

    # Should have integer comparison
    assert "self.assertEqual(result, 42)" in test_method, (
        f"Test should expect integer 42, but got:\\n{test_method}"
    )

    # Should NOT have string comparison
    assert 'self.assertEqual(result, "42")' not in test_method, (
        f"Test should NOT expect string '42', but got:\\n{test_method}"
    )

    print("✓ Non-str return types still get proper type conversion")


if __name__ == "__main__":
    test_regression_lcb_stdin_numeric_string_not_converted()
    test_non_str_return_type_still_converts()
    print("\\n✓ All regression tests passed!")
