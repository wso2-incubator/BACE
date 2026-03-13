import pytest
import textwrap
from infrastructure.languages.python import PythonLanguage

@pytest.fixture
def adapter() -> PythonLanguage:
    return PythonLanguage()

class TestCodegenSignatures:
    def test_codegen_int_return_type_casting(self, adapter):
        # Using a proper class to match standard dataset patterns
        starter_code = textwrap.dedent("""
            class Solution:
                def sol(self, x: int) -> int:
                    return x + 1
        """)
        
        # Input is JSON compatible, Output is a string "11" in dataset
        input_str = "10"
        output_str = '"11"'
        
        test_case = adapter.composer.generate_test_case(input_str, output_str, starter_code, 1)
        
        # Check if 'expected_output' is cast to int (11) and not string '"11"'
        assert "expected_output = 11" in test_case
        assert "solution = Solution()" in test_case
        assert "actual_output = solution.sol(*args)" in test_case
        assert "assert actual_output == expected_output" in test_case

    def test_codegen_str_return_type_casting(self, adapter):
        starter_code = textwrap.dedent("""
            class Solution:
                def sol(self, x: int) -> str:
                    return str(x)
        """)
        
        # Input 10, Output 10 (int in dataset)
        input_str = "10"
        output_str = "10"
        
        test_case = adapter.composer.generate_test_case(input_str, output_str, starter_code, 1)
        
        # Should cast 10 -> '10'
        assert "expected_output = '10'" in test_case
        assert "actual_output = solution.sol(*args)" in test_case

    def test_codegen_bool_return_type_casting(self, adapter):
        starter_code = "def sol(x: int) -> bool:\n    return x > 0"
        
        # Output "true" (string in dataset)
        input_str = "10"
        output_str = '"true"'
        
        test_case = adapter.composer.generate_test_case(input_str, output_str, starter_code, 1)
        
        # Should cast "true" -> True
        assert "expected_output = True" in test_case

    def test_codegen_complex_signature_fallback(self, adapter):
        starter_code = "from typing import List\ndef sol(nums: List[int]) -> List[int]:\n    return sorted(nums)"
        
        # Signature is List[int], current caster only handles int/str/float/bool strictly.
        # It should fallback to raw output_val (which is a list from _parse_val)
        input_str = "[3, 1, 2]"
        output_str = "[1, 2, 3]"
        
        test_case = adapter.composer.generate_test_case(input_str, output_str, starter_code, 1)
        
        assert "expected_output = [1, 2, 3]" in test_case
        assert "actual_output = sol(*args)" in test_case

    def test_codegen_mixed_functional_arguments(self, adapter):
        starter_code = "def sol(a: int, b: str, c: float) -> str:\n    return f'{a}-{b}-{c}'"
        
        # functional style (newline separated)
        input_str = "1\nhello\n3.14"
        output_str = '"1-hello-3.14"'
        
        test_case = adapter.composer.generate_test_case(input_str, output_str, starter_code, 1)
        
        assert "args = [1, 'hello', 3.14]" in test_case
        assert "expected_output = '1-hello-3.14'" in test_case
        assert "actual_output = sol(*args)" in test_case

    def test_codegen_stdin_style_raw_string(self, adapter):
        # input_str:str -> str pattern in a class
        starter_code = textwrap.dedent("""
            class Solution:
                def sol(self, input_str: str) -> str:
                    return input_str.strip()
        """)
        
        # Multi-line raw string input (not JSON)
        input_str = "line1\nline2"
        output_str = "line1\nline2"
        
        test_case = adapter.composer.generate_test_case(input_str, output_str, starter_code, 1)
        
        assert "input_str = 'line1\\nline2'" in test_case
        assert "expected_output = 'line1\\nline2'" in test_case
        assert "actual_output = solution.sol(input_str)" in test_case
        assert "assert actual_output == expected_output" in test_case

    def test_codegen_standalone_functional(self, adapter):
        # A standalone function (no class) with functional signature
        starter_code = "def sol(x: int) -> int:\n    return x + 1"
        
        input_str = "10"
        output_str = "11"
        
        test_case = adapter.composer.generate_test_case(input_str, output_str, starter_code, 1)
        
        assert "def test_case_1():" in test_case
        assert "args = [10]" in test_case
        assert "actual_output = sol(*args)" in test_case
        assert "expected_output = 11" in test_case
        assert "solution =" not in test_case
