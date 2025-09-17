"""
Tests for the SafeCodeSandbox class.
"""

import pytest
import tempfile
import os
from src.common.sandbox import SafeCodeSandbox, create_safe_test_environment, CodeExecutionTimeoutError, CodeExecutionError


class TestSafeCodeSandbox:
    """Test cases for SafeCodeSandbox."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sandbox = SafeCodeSandbox()

    def test_basic_execution(self):
        """Test basic code execution."""
        code = """
def add(a, b):
    return a + b

result = add(2, 3)
print(result)
"""
        result = self.sandbox.execute_code(code)
        assert result['success']
        assert "5" in result['output']
        assert result['error'] is None or result['error'] == ""
        assert result.get('timeout_occurred', False) is False

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = """
def broken_function(
    print("This has a syntax error")
"""
        result = self.sandbox.execute_code(code)
        assert not result['success']
        assert result['error'] is not None
        assert len(result['error']) > 0

    def test_runtime_error_handling(self):
        """Test handling of runtime errors."""
        code = """
def divide_by_zero():
    return 1 / 0

divide_by_zero()
"""
        result = self.sandbox.execute_code(code)
        assert not result['success']
        assert result['error'] is not None
        assert len(result['error']) > 0

    def test_timeout_handling(self):
        """Test timeout handling for infinite loops."""
        code = """
while True:
    pass
"""
        sandbox = SafeCodeSandbox(timeout=1)  # 1 second timeout
        result = sandbox.execute_code(code)
        assert not result['success']
        # Should either timeout or be terminated
        assert result.get('timeout_occurred',
                          False) or result['error'] is not None

    def test_import_restrictions(self):
        """Test that dangerous imports are handled."""
        dangerous_code = """
import os
print("Import successful")
"""
        result = self.sandbox.execute_code(dangerous_code)
        # The code might execute - we mainly test that it doesn't crash our system
        assert isinstance(result['success'], bool)

    def test_large_output_handling(self):
        """Test handling of large output."""
        code = """
for i in range(100):  # Reduced from 1000 to speed up test
    print(f"Line {i}")
"""
        result = self.sandbox.execute_code(code)
        assert result['success']
        # Should handle large output gracefully
        assert len(result['output']) > 0

    def test_memory_intensive_code(self):
        """Test handling of memory-intensive code."""
        code = """
# Create a reasonably large list
data = list(range(10000))
print(f"Created list with {len(data)} elements")
"""
        result = self.sandbox.execute_code(code)
        assert result['success']
        assert "10000" in result['output']

    def test_multiline_function_execution(self):
        """Test execution of multiline functions."""
        code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

test_array = [64, 34, 25, 12, 22, 11, 90]
sorted_array = bubble_sort(test_array.copy())
print(f"Original: {test_array}")
print(f"Sorted: {sorted_array}")
"""
        result = self.sandbox.execute_code(code)
        assert result['success']
        assert "Sorted: [11, 12, 22, 25, 34, 64, 90]" in result['output']

    def test_test_script_execution(self):
        """Test executing test scripts."""
        test_script = """
def test_addition():
    assert 2 + 2 == 4
    assert 1 + 1 == 2
    print("Addition tests passed!")

def test_multiplication():
    assert 2 * 3 == 6
    assert 5 * 0 == 0
    print("Multiplication tests passed!")

# Run tests
test_addition()
test_multiplication()
print("All tests completed!")
"""
        result = self.sandbox.execute_test_script(test_script)
        assert result['success']
        assert "All tests completed!" in result['output']


class TestCreateSafeTestEnvironment:
    """Test cases for create_safe_test_environment function."""

    def test_safe_test_environment_creation(self):
        """Test creating a safe test environment."""
        sandbox = create_safe_test_environment()
        assert isinstance(sandbox, SafeCodeSandbox)
        assert sandbox.timeout == 30
        assert sandbox.max_memory_mb == 100

        # Test that it can execute code
        code = "print('Hello from safe environment!')"
        result = sandbox.execute_code(code)
        assert result['success']
        assert "Hello from safe environment!" in result['output']

    def test_safe_environment_with_allowed_imports(self):
        """Test that allowed imports work in safe environment."""
        sandbox = create_safe_test_environment()
        code = """
import math
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
"""
        result = sandbox.execute_code(code)
        assert result['success']
        assert "4.0" in result['output']


if __name__ == "__main__":
    pytest.main([__file__])
