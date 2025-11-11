"""
Test cases for contains_starter_code function with real LCB dataset examples.
"""

from common.code_preprocessing.analysis import contains_starter_code


class TestContainsStarterCode:
    """Test contains_starter_code with real LiveCodeBench starter codes."""

    def test_exact_match(self) -> None:
        """Test exact match of starter code."""
        starter = "class Solution:\n    def solve(self, x: int) -> int:\n        pass"
        code = "class Solution:\n    def solve(self, x: int) -> int:\n        pass"
        assert contains_starter_code(code, starter) is True

    def test_incomplete_function_signature_leetcode_style(self) -> None:
        """Test LeetCode-style incomplete function signature (real example)."""
        # Real example from LeetCode problems
        starter = "class Solution:\n    def missingInteger(self, nums: List[int]) -> int:\n        "

        # Complete implementation
        code = """class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find sequential prefix
        seq_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                seq_sum += nums[i]
            else:
                break
        
        # Find smallest missing
        result = seq_sum
        nums_set = set(nums)
        while result in nums_set:
            result += 1
        return result
"""
        assert contains_starter_code(code, starter) is True

    def test_incomplete_signature_with_list_type(self) -> None:
        """Test incomplete signature with List type annotation."""
        starter = "class Solution:\n    def maxFrequencyElements(self, nums: List[int]) -> int:\n        "

        code = """class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        from collections import Counter
        freq = Counter(nums)
        max_freq = max(freq.values())
        return sum(1 for f in freq.values() if f == max_freq) * max_freq
"""
        assert contains_starter_code(code, starter) is True

    def test_starter_with_nested_types(self) -> None:
        """Test starter code with nested type annotations."""
        starter = "class Solution:\n    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:\n        "

        code = """class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diag = 0
        max_area = 0
        for length, width in dimensions:
            diag = (length**2 + width**2) ** 0.5
            area = length * width
            if diag > max_diag or (diag == max_diag and area > max_area):
                max_diag = diag
                max_area = area
        return max_area
"""
        assert contains_starter_code(code, starter) is True

    def test_default_starter_code(self) -> None:
        """Test with the default starter code used for STDIN problems."""
        starter = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""

        code = """
class Solution:
    def sol(self, input_str: str) -> str:
        # Parse input
        n = int(input_str.strip())
        # Process
        result = n * 2
        return str(result)
"""
        assert contains_starter_code(code, starter) is True

    def test_starter_code_with_extra_whitespace(self) -> None:
        """Test that extra whitespace doesn't break matching."""
        starter = "class Solution:\n    def solve(self, x: int) -> int:\n        "

        code = """class Solution:
    
    def solve(self, x: int) -> int:
        
        return x * 2
"""
        assert contains_starter_code(code, starter) is True

    def test_starter_code_with_comments(self) -> None:
        """Test that comments in complete code don't break matching."""
        starter = "class Solution:\n    def solve(self, x: int) -> int:\n        "

        code = """class Solution:
    # This is a solution class
    def solve(self, x: int) -> int:
        # Calculate result
        return x * 2
"""
        assert contains_starter_code(code, starter) is True

    def test_different_class_name_should_fail(self) -> None:
        """Test that different class names don't match."""
        starter = "class Solution:\n    def solve(self, x: int) -> int:\n        "

        code = """class MySolution:
    def solve(self, x: int) -> int:
        return x * 2
"""
        assert contains_starter_code(code, starter) is False

    def test_different_method_name_should_fail(self) -> None:
        """Test that different method names don't match."""
        starter = "class Solution:\n    def solve(self, x: int) -> int:\n        "

        code = """class Solution:
    def compute(self, x: int) -> int:
        return x * 2
"""
        assert contains_starter_code(code, starter) is False

    def test_empty_starter_code(self) -> None:
        """Test that empty starter code is always considered contained."""
        starter = ""
        code = "class Solution:\n    def solve(self): return 42"
        assert contains_starter_code(code, starter) is True

    def test_empty_complete_code_with_non_empty_starter(self) -> None:
        """Test that non-empty starter is not in empty code."""
        starter = "class Solution:\n    def solve(self): pass"
        code = ""
        assert contains_starter_code(code, starter) is False

    def test_signature_matching_strategy(self) -> None:
        """Test that signature matching works when direct match fails."""
        # Starter has extra whitespace/formatting that won't match directly
        starter = (
            "class    Solution:\n\n    def solve(self, x: int)  ->  int:\n        "
        )

        code = """class Solution:
    def solve(self, x: int) -> int:
        return x * 2
"""
        # Should match based on signatures
        assert contains_starter_code(code, starter) is True

    def test_multiple_methods_in_starter(self) -> None:
        """Test starter code with multiple method signatures."""
        starter = """class Solution:
    def method1(self, x: int) -> int:
        
    def method2(self, y: str) -> str:
        """

        code = """class Solution:
    def method1(self, x: int) -> int:
        return x * 2
    
    def method2(self, y: str) -> str:
        return y.upper()
    
    def helper(self):
        pass
"""
        assert contains_starter_code(code, starter) is True

    def test_partial_method_signature(self) -> None:
        """Test very incomplete method signature (cut off mid-parameter)."""
        starter = "class Solution:\n    def solve(self, x"

        code = """class Solution:
    def solve(self, x: int, y: int) -> int:
        return x + y
"""
        # Should match based on class and method name signatures
        assert contains_starter_code(code, starter) is True

    def test_docstring_in_complete_code(self) -> None:
        """Test that docstrings in complete code don't break matching."""
        starter = "class Solution:\n    def solve(self, x: int) -> int:\n        "

        code = '''class Solution:
    def solve(self, x: int) -> int:
        """
        This is a docstring.
        It explains the solution.
        """
        return x * 2
'''
        assert contains_starter_code(code, starter) is True

    def test_real_lcb_default_starter(self) -> None:
        """Test the actual default starter code from LCB dataset."""
        # This is what gets used for Codeforces/AtCoder STDIN problems
        starter = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""

        code = """
class Solution:
    def sol(self, input_str: str) -> str:
        lines = input_str.strip().split('\\n')
        n = int(lines[0])
        result = str(n * 2)
        return result
"""
        assert contains_starter_code(code, starter) is True

    def test_missing_both_class_and_method(self) -> None:
        """Test code that has neither the class nor method from starter."""
        starter = "class Solution:\n    def solve(self, x: int) -> int:\n        "

        code = """def main():
    x = 5
    print(x * 2)

main()
"""
        assert contains_starter_code(code, starter) is False

    def test_only_class_matches_not_method(self) -> None:
        """Test when only class name matches but not method."""
        starter = "class Solution:\n    def solve(self, x: int) -> int:\n        "

        code = """class Solution:
    def different_method(self, x: int) -> int:
        return x * 2
"""
        # Should fail - method name doesn't match
        assert contains_starter_code(code, starter) is False

    def test_case_sensitivity(self) -> None:
        """Test that matching is case-sensitive."""
        starter = "class Solution:\n    def solve(self, x: int) -> int:\n        "

        code = """class solution:
    def solve(self, x: int) -> int:
        return x * 2
"""
        # Different case in class name
        assert contains_starter_code(code, starter) is False
