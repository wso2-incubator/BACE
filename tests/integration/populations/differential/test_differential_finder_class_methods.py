import pytest
from coevolution.populations.differential.finder import DifferentialFinder
from infrastructure.languages.python import PythonLanguage
from infrastructure.sandbox import SandboxConfig

pytestmark = pytest.mark.integration


class TestDifferentialFinderClassMethods:
    """Integration tests for class-based method differential testing."""

    def test_two_int_parameters_range_check(self) -> None:
        """
        Test differential finding for a method with two int parameters.

        Scenario:
        - Code A (Correct): Counts numbers in range [l, r] that satisfy a condition.
        - Code B (Buggy): Has an off-by-one error in the range check.
        """
        code_a_snippet = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        count = 0
        for num in range(l, r + 1):
            if num % 2 == 0:
                count += 1
        return count
"""

        code_b_snippet = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        count = 0
        # Bug: Missing +1, so r is not included
        for num in range(l, r):
            if num % 2 == 0:
                count += 1
        return count
"""

        input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"l": 1, "r": 10},   # Should find difference
        {"l": 2, "r": 8},    # Should find difference
        {"l": 5, "r": 5},    # Edge case: single number
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(3)
"""

        sandbox_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000
        )
        lang = PythonLanguage()
        finder = DifferentialFinder(
            parser=lang.parser, composer=lang.composer, runtime=lang.runtime, sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
        )

        results = finder.find_differential(
            code_a_snippet=code_a_snippet,
            code_b_snippet=code_b_snippet,
            input_generator_script=input_generator_script,
            limit=10,
        )

        # We expect differences in all cases because r is not properly included in code B
        assert len(results) >= 1, (
            f"Expected at least 1 divergence, found {len(results)}"
        )

        # Check that we found the expected divergences (order may vary)
        input_data_found = [r.input_data for r in results]
        assert {"l": 1, "r": 10} in input_data_found or {
            "l": 2,
            "r": 8,
        } in input_data_found

        # Verify outputs are strings
        for div in results:
            assert isinstance(div.output_a, str)
            assert isinstance(div.output_b, str)

    def test_list_parameter_sorting_logic(self) -> None:
        """
        Test differential finding for a method with list parameter.

        Scenario:
        - Code A (Correct): Standard sorting.
        - Code B (Buggy): Conditionally reverses sort for specific list lengths.
        """
        code_a_snippet = """
class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        return sorted(nums)
"""

        code_b_snippet = """
class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        # Bug: Reverses for lists longer than 3 elements
        if len(nums) > 3:
            return sorted(nums, reverse=True)
        return sorted(nums)
"""

        input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"nums": [3, 1, 2]},           # Length 3: Should match
        {"nums": [5, 2, 8, 1]},        # Length 4: Should differ
        {"nums": [10, 5, 1, 9, 2]},    # Length 5: Should differ
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(3)
"""

        sandbox_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000
        )
        lang = PythonLanguage()
        finder = DifferentialFinder(
            parser=lang.parser, composer=lang.composer, runtime=lang.runtime, sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
        )

        results = finder.find_differential(
            code_a_snippet, code_b_snippet, input_generator_script
        )

        assert len(results) == 2, f"Expected 2 divergences, found {len(results)}"

        # Check results (order may vary due to parallel execution)
        input_data_list = [r.input_data for r in results]
        assert {"nums": [5, 2, 8, 1]} in input_data_list
        assert {"nums": [10, 5, 1, 9, 2]} in input_data_list

        # Verify outputs are strings and show sorting differences
        for res in results:
            assert isinstance(res.output_a, str)
            assert isinstance(res.output_b, str)
            # Output A should be ascending, Output B should be descending
            assert res.output_a != res.output_b

    def test_string_parameter_case_sensitivity(self) -> None:
        """
        Test differential finding for a method with string parameter.

        Scenario:
        - Code A (Correct): Case-insensitive palindrome check.
        - Code B (Buggy): Case-sensitive palindrome check.
        """
        code_a_snippet = """
class Solution:
    def isPalindrome(self, s: str) -> bool:
        cleaned = s.lower().replace(" ", "")
        return cleaned == cleaned[::-1]
"""

        code_b_snippet = """
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # Bug: Case sensitive comparison
        cleaned = s.replace(" ", "")
        return cleaned == cleaned[::-1]
"""

        input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"s": "aba"},                           # Should match
        {"s": "Racecar"},                       # Should differ (case)
        {"s": "A man a plan a canal Panama"},  # Should differ (case)
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(3)
"""

        sandbox_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000
        )
        lang = PythonLanguage()
        finder = DifferentialFinder(
            parser=lang.parser, composer=lang.composer, runtime=lang.runtime, sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
        )

        results = finder.find_differential(
            code_a_snippet, code_b_snippet, input_generator_script
        )
        # We expect exactly 2 discrepancies
        assert len(results) == 2, f"Expected 2 discrepancies, found {len(results)}"
        
        div_inputs = [r.input_data for r in results]
        assert {"s": "Racecar"} in div_inputs
        assert {"s": "A man a plan a canal Panama"} in div_inputs

    def test_mixed_parameter_types(self) -> None:
        """
        Test differential finding for a method with mixed parameter types.

        Scenario:
        - Code A (Correct): Finds indices where substring of length k starts.
        - Code B (Buggy): Off-by-one error in range calculation.
        """
        code_a_snippet = """
class Solution:
    def findSubstring(self, s: str, k: int) -> list[int]:
        result = []
        for i in range(len(s) - k + 1):
            result.append(i)
        return result
"""

        code_b_snippet = """
class Solution:
    def findSubstring(self, s: str, k: int) -> list[int]:
        result = []
        # Bug: Missing +1, so last valid index is excluded
        for i in range(len(s) - k):
            result.append(i)
        return result
"""

        input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"s": "abcdef", "k": 3},  # Length 6, k=3: Should find difference
        {"s": "hello", "k": 2},   # Length 5, k=2: Should find difference
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(2)
"""

        sandbox_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000
        )
        lang = PythonLanguage()
        finder = DifferentialFinder(
            parser=lang.parser, composer=lang.composer, runtime=lang.runtime, sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
        )

        results = finder.find_differential(
            code_a_snippet, code_b_snippet, input_generator_script
        )

        assert len(results) >= 1, (
            f"Expected at least 1 divergence, found {len(results)}"
        )

        # Check that we found at least one of the expected divergences (order may vary)
        input_data_list = [r.input_data for r in results]
        assert {"s": "abcdef", "k": 3} in input_data_list or {
            "s": "hello",
            "k": 2,
        } in input_data_list

        # Verify outputs are strings and show differences
        for res in results:
            assert isinstance(res.output_a, str)
            assert isinstance(res.output_b, str)
            assert res.output_a != res.output_b

    def test_nested_list_parameter(self) -> None:
        """
        Test differential finding for a method with List[List[int]] parameter.

        Scenario:
        - Code A (Correct): Flattens nested list.
        - Code B (Buggy): Only flattens first level.
        """
        code_a_snippet = """
class Solution:
    def flatten(self, matrix: list[list[int]]) -> list[int]:
        result = []
        for row in matrix:
            result.extend(row)
        return result
"""

        code_b_snippet = """
class Solution:
    def flatten(self, matrix: list[list[int]]) -> list[int]:
        # Bug: Returns concatenation of first elements only
        result = []
        for row in matrix:
            if row:
                result.append(row[0])
        return result
"""

        input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"matrix": [[1, 2], [3, 4], [5, 6]]},
        {"matrix": [[10], [20], [30]]},
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(2)
"""

        sandbox_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000
        )
        lang = PythonLanguage()
        finder = DifferentialFinder(
            parser=lang.parser, composer=lang.composer, runtime=lang.runtime, sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
        )

        results = finder.find_differential(
            code_a_snippet, code_b_snippet, input_generator_script
        )

        assert len(results) >= 1, (
            f"Expected at least 1 divergence, found {len(results)}"
        )

        # Check first divergence
        div = results[0]
        assert div.input_data == {"matrix": [[1, 2], [3, 4], [5, 6]]}
        assert div.output_a == "[1, 2, 3, 4, 5, 6]"
        assert div.output_b == "[1, 3, 5]"

    def test_no_divergence_identical_implementations(self) -> None:
        """
        Test that identical implementations produce no divergences.

        Scenario:
        - Code A and Code B are identical.
        """
        code_snippet = """
class Solution:
    def add(self, a: int, b: int) -> int:
        return a + b
"""

        input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"a": 1, "b": 2},
        {"a": 5, "b": 7},
        {"a": 0, "b": 0},
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(3)
"""

        sandbox_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000
        )
        lang = PythonLanguage()
        finder = DifferentialFinder(
            parser=lang.parser, composer=lang.composer, runtime=lang.runtime, sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
        )

        results = finder.find_differential(
            code_a_snippet=code_snippet,
            code_b_snippet=code_snippet,
            input_generator_script=input_generator_script,
            limit=10,
        )

        assert len(results) == 0, f"Expected no divergences, found {len(results)}"

    def test_dict_parameter(self) -> None:
        """
        Test differential finding for a method with dict parameter.

        Scenario:
        - Code A (Correct): Sums all values in dict.
        - Code B (Buggy): Only sums positive values.
        """
        code_a_snippet = """
class Solution:
    def sumValues(self, data: dict[str, int]) -> int:
        return sum(data.values())
"""

        code_b_snippet = """
class Solution:
    def sumValues(self, data: dict[str, int]) -> int:
        # Bug: Only sums positive values
        return sum(v for v in data.values() if v > 0)
"""

        input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"data": {"a": 1, "b": 2, "c": 3}},        # All positive: Should match
        {"data": {"a": 1, "b": -2, "c": 3}},       # Has negative: Should differ
        {"data": {"x": -5, "y": -10}},             # All negative: Should differ
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(3)
"""

        sandbox_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000
        )
        lang = PythonLanguage()
        finder = DifferentialFinder(
            parser=lang.parser, composer=lang.composer, runtime=lang.runtime, sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
        )

        results = finder.find_differential(
            code_a_snippet, code_b_snippet, input_generator_script
        )

        assert len(results) == 2, f"Expected 2 divergences, found {len(results)}"

        # Check results (order may vary due to parallel execution)
        input_data_list = [r.input_data for r in results]
        assert {"data": {"a": 1, "b": -2, "c": 3}} in input_data_list
        assert {"data": {"x": -5, "y": -10}} in input_data_list

        # Verify outputs are strings and show the difference
        for res in results:
            assert isinstance(res.output_a, str)
            assert isinstance(res.output_b, str)
            assert res.output_a != res.output_b
