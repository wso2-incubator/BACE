from infrastructure.languages.python import PythonLanguage
from infrastructure.languages.ballerina import BallerinaLanguage
"""Integration test for dual-language differential testing: Python generators + Ballerina code."""

import pytest

from infrastructure.sandbox import SandboxConfig, create_sandbox


class TestDualLanguageDifferentialTesting:
    """Test differential testing with Python input generators and Ballerina code execution."""

    def test_python_generator_with_ballerina_code(self) -> None:
        """
        End-to-end test demonstrating dual-language architecture.

        Flow:
        1. Python generator script creates test inputs
        2. Test inputs are parsed from Python output
        3. Ballerina code is executed with those inputs
        4. Differential testing finds divergences
        """

        # Step 1: Create Python input generator script
        python_generator = """
def generate_test_inputs(num_inputs):
    import random
    random.seed(42)
    
    inputs = []
    for i in range(num_inputs):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        inputs.append({"a": a, "b": b})
    
    return inputs

print(generate_test_inputs(5))
"""

        # Step 2: Execute Python generator to get test inputs
        python = PythonLanguage()
        python_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000, language="python"
        )
        python_sandbox = create_sandbox(python_config)

        # Compose and execute generator script
        composed_script = python.composer.compose_generator_script(python_generator, 5)
        result = python_sandbox.execute_code(composed_script, python.runtime)

        assert result.success, f"Generator should execute successfully: {result.error}"

        # Parse output (use last line in case of multiple outputs)
        output_lines = result.output.strip().split("\n")
        output_to_parse = output_lines[-1]
        test_inputs = python.parser.parse_test_inputs(output_to_parse)

        # Verify we got test inputs
        assert len(test_inputs) == 5, "Should generate 5 test inputs"
        for inp in test_inputs:
            assert "a" in inp and "b" in inp, "Each input should have 'a' and 'b' keys"

        # Step 3: Create Ballerina code snippets (correct vs buggy)
        ballerina_code_correct = """
public function add(int a, int b) returns int {
    return a + b;
}
"""

        ballerina_code_buggy = """
public function add(int a, int b) returns int {
    return a + b + 1;
}
"""

        # Step 4: Execute Ballerina code with Python-generated inputs
        ballerina = BallerinaLanguage()
        ballerina_config = SandboxConfig(
            timeout=30, max_memory_mb=200, max_output_size=50_000, language="ballerina"
        )
        ballerina_sandbox = create_sandbox(ballerina_config)

        results_a = []
        results_b = []

        for test_input in test_inputs:
            a = test_input["a"]
            b = test_input["b"]

            # Format input for Ballerina
            input_formatted = f"add({a}, {b})"

            # Execute correct code
            script_a = ballerina.composer.compose_evaluation_script(
                ballerina_code_correct, input_formatted
            )
            output_a = ballerina_sandbox.execute_code(script_a, ballerina.runtime)
            assert output_a.success, f"Correct code should execute: {output_a.error}"
            results_a.append(int(output_a.output.strip()))

            # Execute buggy code
            script_b = ballerina.composer.compose_evaluation_script(
                ballerina_code_buggy, input_formatted
            )
            output_b = ballerina_sandbox.execute_code(script_b, ballerina.runtime)
            assert output_b.success, f"Buggy code should execute: {output_b.error}"
            results_b.append(int(output_b.output.strip()))

        # Step 5: Verify differential results
        divergences = [
            i for i in range(len(test_inputs)) if results_a[i] != results_b[i]
        ]

        assert len(divergences) > 0, (
            "Should find divergences between correct and buggy code"
        )

        # Verify the divergence is consistent (always off by 1)
        for idx in divergences:
            expected = results_a[idx]
            actual = results_b[idx]
            assert actual == expected + 1, (
                f"Buggy code should be off by 1 at test {idx}"
            )

        # Verify results match expected calculations
        for idx, test_input in enumerate(test_inputs):
            a = test_input["a"]
            b = test_input["b"]
            assert results_a[idx] == a + b, f"Correct result at index {idx}"
            assert results_b[idx] == a + b + 1, f"Buggy result at index {idx}"

    def test_python_generator_with_special_floats(self) -> None:
        """Test that Python generators can produce special float values for Ballerina."""

        # Python generator with special float values
        python_generator = """
import math

def generate_test_inputs(num_inputs):
    return [
        {"x": 1.5},
        {"x": float('inf')},
        {"x": float('-inf')},
        {"x": float('nan')},
        {"x": 0.0}
    ][:num_inputs]

print(generate_test_inputs(5))
"""

        # Execute generator
        python = PythonLanguage()
        python_config = SandboxConfig(
            timeout=5, max_memory_mb=200, max_output_size=50_000, language="python"
        )
        python_sandbox = create_sandbox(python_config)

        composed_script = python.composer.compose_generator_script(python_generator, 5)
        result = python_sandbox.execute_code(composed_script, python.runtime)

        assert result.success, "Generator should execute successfully"

        # Parse output
        output_lines = result.output.strip().split("\n")
        output_to_parse = output_lines[-1]
        test_inputs = python.parser.parse_test_inputs(output_to_parse)

        # Verify special float values are parsed correctly
        assert len(test_inputs) == 5, "Should parse 5 inputs"

        import math

        assert test_inputs[0]["x"] == 1.5
        assert math.isinf(test_inputs[1]["x"]) and test_inputs[1]["x"] > 0
        assert math.isinf(test_inputs[2]["x"]) and test_inputs[2]["x"] < 0
        assert math.isnan(test_inputs[3]["x"])
        assert test_inputs[4]["x"] == 0.0
