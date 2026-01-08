"""
End-to-end test demonstrating the full differential testing workflow
for class-based starter codes like beautifulNumbers.

This test verifies the entire pipeline:
1. Input generator script creation
2. Differential finder execution
3. Test method generation from IO pairs
4. Test class integration
"""

from unittest.mock import MagicMock

import pytest

from coevolution.strategies.breeding.differential_finder import DifferentialFinder
from coevolution.strategies.operators.differential_llm_operator import (
    DifferentialLLMOperator,
)
from infrastructure.code_preprocessing.composition import rebuild_unittest_with_methods
from infrastructure.code_preprocessing.transformation import (
    build_test_method_from_io,
    setup_unittest_class_from_starter_code,
)
from infrastructure.sandbox import SandboxConfig


class TestBeautifulNumbersEndToEnd:
    """End-to-end test for beautifulNumbers-style starter codes."""

    def test_full_workflow_beautiful_numbers(self) -> None:
        """
        Complete workflow test for a method with two int parameters.

        This test simulates the entire differential testing process:
        1. Generate initial test class scaffold
        2. Find differential inputs between two implementations
        3. Build test methods from the differential IO pairs
        4. Integrate test methods into the test class
        """

        # --- Step 1: Setup ---
        starter_code = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        '''Count beautiful numbers in range [l, r]'''
        pass
"""

        # Two different implementations to test
        code_a_snippet = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        '''Correct implementation: includes r in range'''
        count = 0
        for num in range(l, r + 1):
            if num % 2 == 0:
                count += 1
        return count
"""

        code_b_snippet = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        '''Buggy implementation: excludes r from range'''
        count = 0
        for num in range(l, r):  # Bug: should be r + 1
            if num % 2 == 0:
                count += 1
        return count
"""

        # --- Step 2: Generate Initial Test Class Scaffold ---
        test_class_block = setup_unittest_class_from_starter_code(starter_code)

        assert "class TestSolution(unittest.TestCase):" in test_class_block
        assert "self.solution = Solution()" in test_class_block
        assert "def setUp(self):" in test_class_block

        # --- Step 3: Find Differential Inputs ---
        input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"l": 1, "r": 10},   # Expected diff: A=5 (2,4,6,8,10), B=4 (2,4,6,8)
        {"l": 2, "r": 8},    # Expected diff: A=4 (2,4,6,8), B=3 (2,4,6)
        {"l": 5, "r": 5},    # Edge case: single number
        {"l": 1, "r": 2},    # Small range
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(4)
"""

        sandbox_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000
        )
        finder = DifferentialFinder(
            sandbox_config=sandbox_config, enable_multiprocessing=True, num_workers=4
        )

        differential_results = finder.find_differential(
            code_a_snippet=code_a_snippet,
            code_b_snippet=code_b_snippet,
            input_generator_script=input_generator_script,
            limit=10,
        )

        # Verify we found differentials
        assert len(differential_results) >= 2
        print(f"\nFound {len(differential_results)} differential test cases:")
        for i, result in enumerate(differential_results):
            print(
                f"  Case {i + 1}: Input {result.input_data} -> A={result.output_a}, B={result.output_b}"
            )

        # --- Step 4: Build Test Methods from Differential IO Pairs ---
        mock_llm = MagicMock()
        operator = DifferentialLLMOperator(llm=mock_llm)

        test_methods = []
        for idx, diff_result in enumerate(differential_results):
            # Convert DifferentialResult to IO pair format
            io_pair = {
                "inputdata": diff_result.input_data,
                "output": diff_result.output_a,  # Use output_a as the expected output
            }

            # Build test method
            test_method = operator.get_test_method_from_io(
                starter_code=starter_code,
                io_pairs=[io_pair],
                code_parent_ids=["CodeA", "CodeB"],
                io_index=idx,
            )

            test_methods.append(test_method)
            print(f"\nGenerated test method {idx + 1}:")
            print(test_method)

        # Verify test methods were generated correctly
        assert len(test_methods) >= 2

        # Check first test method structure
        first_method = test_methods[0]
        assert "def test_case_CodeA_CodeB_0(self):" in first_method
        assert "result = self.solution.beautifulNumbers(" in first_method
        assert "self.assertEqual(result," in first_method
        assert "l=" in first_method
        assert "r=" in first_method

        # --- Step 5: Integrate Test Methods into Test Class ---
        final_test_class = rebuild_unittest_with_methods(test_class_block, test_methods)

        print("\n=== Final Test Class ===")
        print(final_test_class)

        # Verify final test class structure
        assert "class TestSolution(unittest.TestCase):" in final_test_class
        assert "def setUp(self):" in final_test_class
        assert "self.solution = Solution()" in final_test_class

        # Verify all test methods are present
        for i in range(len(test_methods)):
            assert f"def test_case_CodeA_CodeB_{i}(self):" in final_test_class

        # Verify test calls the correct method with correct parameters
        assert "self.solution.beautifulNumbers(" in final_test_class

        # --- Step 6: Verify Specific Test Content ---
        # Check that first differential (l=1, r=10) is correctly represented
        assert "l=1" in final_test_class
        assert "r=10" in final_test_class
        # Results are converted to strings for comparison with sandbox output
        assert "self.assertEqual(result," in final_test_class

    def test_build_test_method_preserves_parameter_names(self) -> None:
        """
        Verify that parameter names are preserved in generated test methods.

        This is critical for methods like beautifulNumbers(l, r) where
        positional vs keyword arguments matter.
        """
        starter_code = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        pass
"""

        io_pairs = [
            {"inputdata": {"l": 1, "r": 10}, "output": 5},
            {"inputdata": {"l": 100, "r": 200}, "output": 51},
        ]

        test_method = build_test_method_from_io(starter_code, io_pairs, "NAMES_TEST")

        # Verify keyword arguments are used
        assert "l=1" in test_method
        assert "r=10" in test_method
        assert "l=100" in test_method
        assert "r=200" in test_method

        # Verify they're in the correct format
        assert "beautifulNumbers(l=1, r=10)" in test_method
        assert "beautifulNumbers(l=100, r=200)" in test_method

        # Verify assertions with str() conversion
        assert "self.assertEqual(result, 5)" in test_method
        assert "self.assertEqual(result, 51)" in test_method

    def test_parameter_order_independence(self) -> None:
        """
        Verify that keyword arguments handle different parameter orders correctly.

        When using keyword arguments, the order in the input dict shouldn't matter.
        """
        starter_code = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        pass
"""

        # Input dict with parameters in reverse order
        io_pairs = [
            {"inputdata": {"r": 10, "l": 1}, "output": 5},  # r before l
        ]

        test_method = build_test_method_from_io(starter_code, io_pairs, "ORDER_TEST")

        # Both parameters should be present as keyword arguments
        assert "l=1" in test_method
        assert "r=10" in test_method

        # The method call should work regardless of dict order
        # because we use keyword arguments
        assert "beautifulNumbers(" in test_method

    def test_multiple_differential_pairs_create_separate_methods(self) -> None:
        """
        Verify that multiple differential IO pairs create separate test methods
        when processed through the operator.
        """
        starter_code = """
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        pass
"""

        mock_llm = MagicMock()
        operator = DifferentialLLMOperator(llm=mock_llm)

        # Simulate multiple differential findings
        differential_io_pairs = [
            [{"inputdata": {"l": 1, "r": 10}, "output": 5}],
            [{"inputdata": {"l": 10, "r": 20}, "output": 6}],
            [{"inputdata": {"l": 20, "r": 30}, "output": 6}],
        ]

        test_methods = []
        for idx, io_pairs in enumerate(differential_io_pairs):
            method = operator.get_test_method_from_io(
                starter_code=starter_code,
                io_pairs=io_pairs,
                code_parent_ids=["C1", "C2"],
                io_index=idx,
            )
            test_methods.append(method)

        # Verify we have 3 separate test methods
        assert len(test_methods) == 3

        # Verify each has a unique suffix
        assert "test_case_C1_C2_0" in test_methods[0]
        assert "test_case_C1_C2_1" in test_methods[1]
        assert "test_case_C1_C2_2" in test_methods[2]

        # Verify each contains the correct input values
        assert "l=1, r=10" in test_methods[0]
        assert "l=10, r=20" in test_methods[1]
        assert "l=20, r=30" in test_methods[2]
