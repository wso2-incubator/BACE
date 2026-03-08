"""
Integration test for LCB STDIN-style problems with differential testing.

This test verifies that when a method returns str (like LCB STDIN problems),
the differential test generation preserves string outputs correctly.
"""

from coevolution.populations.differential.finder import DifferentialFinder
from coevolution.populations.differential.operators.llm_operator import (
    DifferentialLLMOperator,
)
from infrastructure.sandbox import SandboxConfig


def test_lcb_stdin_differential_preserves_string_outputs() -> None:
    """
    Verify that LCB STDIN-style differential tests preserve string outputs.

    This tests the complete workflow:
    1. Two code snippets that return strings (LCB STDIN style)
    2. Differential finder finds divergences
    3. Test method builder creates tests that expect strings, not converted types
    """
    # LCB STDIN style: def sol(self, input_str: str) -> str
    code_a_snippet = """
class Solution:
    def sol(self, input_str: str) -> str:
        # Returns numeric string
        n = int(input_str.strip())
        return str(n * 2)
"""

    code_b_snippet = """
class Solution:
    def sol(self, input_str: str) -> str:
        # Returns different numeric string
        n = int(input_str.strip())
        return str(n * 3)
"""

    input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"input_str": "5"},
        {"input_str": "10"},
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(2)
"""

    # Find differential outputs
    sandbox_config = SandboxConfig(timeout=5, max_memory_mb=200, max_output_size=50_000)
    finder = DifferentialFinder(
        sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
    )

    results = finder.find_differential(
        code_a_snippet=code_a_snippet,
        code_b_snippet=code_b_snippet,
        input_generator_script=input_generator_script,
        limit=10,
    )

    # Should find divergences (5*2=10 vs 5*3=15, 10*2=20 vs 10*3=30)
    assert len(results) == 2, f"Expected 2 divergences, found {len(results)}"

    # Verify outputs are strings
    for div in results:
        assert isinstance(div.output_a, str), (
            f"output_a should be str, got {type(div.output_a)}"
        )
        assert isinstance(div.output_b, str), (
            f"output_b should be str, got {type(div.output_b)}"
        )

    # Now test the test generation
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    operator = DifferentialLLMOperator(llm=mock_llm)

    # Convert first divergence to IO pair
    div = results[0]
    io_pair = {
        "inputdata": div.input_data,
        "output": div.output_a,
    }

    # Build test method
    starter_code = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""

    test_method = operator.get_test_method_from_io(
        starter_code=starter_code,
        io_pairs=[io_pair],
        code_parent_ids=["CodeA", "CodeB"],
        io_index=0,
    )

    # Verify that the test expects a STRING, not an integer
    # For input "5", output is "10" (string)
    # Test should be: self.assertEqual(result, "10")
    # NOT: self.assertEqual(result, 10)

    # Check that string comparison is used (with quotes)
    assert '"10"' in test_method or "'10'" in test_method, (
        f"Test should expect string '10', but got: {test_method}"
    )

    # Make sure it's NOT comparing to integer
    assert "result, 10)" not in test_method, (
        f"Test should NOT expect integer 10, but got: {test_method}"
    )

    print("✓ LCB STDIN differential test correctly preserves string outputs")
