"""
Tests for the CodeProcessor class.
"""

import pytest
from common.code_processor import CodeProcessor


class TestCodeProcessor:
    """Test cases for CodeProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = CodeProcessor()

    def test_extract_function_name_from_problem_basic(self):
        """Test basic function name extraction."""
        prompt = '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if numbers are close. """'''

        result = self.processor.extract_function_name_from_problem(prompt)
        assert result == "has_close_elements"

    def test_extract_function_name_from_problem_multiline(self):
        """Test function name extraction with multiline definition."""
        prompt = '''from typing import List

def factorial(n: int) -> int:
    """Calculate factorial of n."""
    pass'''

        result = self.processor.extract_function_name_from_problem(prompt)
        assert result == "factorial"

    def test_extract_function_name_from_problem_no_function(self):
        """Test when no function is found."""
        prompt = '''# Just a comment
x = 5'''

        result = self.processor.extract_function_name_from_problem(prompt)
        assert result == ""

    def test_extract_function_name_from_problem_empty_string(self):
        """Test with empty string."""
        result = self.processor.extract_function_name_from_problem("")
        assert result == ""

    def test_remove_comments_single_line(self):
        """Test single line comment removal."""
        code = '''def test():
    # This is a comment
    return True  # Another comment'''

        result = self.processor.remove_comments(code)
        assert "# This is a comment" not in result
        assert "# Another comment" not in result
        assert "return True" in result
        assert "def test():" in result

    def test_remove_comments_multiline_triple_quotes(self):
        """Test multiline comment removal with triple quotes."""
        code = '''def test():
    """
    This is a multiline
    docstring that should be removed
    """
    return True'''

        result = self.processor.remove_comments(code)
        assert "This is a multiline" not in result
        assert "docstring that should be removed" not in result
        assert "return True" in result
        assert "def test():" in result

    def test_remove_comments_multiline_single_quotes(self):
        """Test multiline comment removal with single quotes."""
        code = """def test():
    '''
    This is a multiline comment
    with single quotes
    '''
    return True"""

        result = self.processor.remove_comments(code)
        assert "This is a multiline comment" not in result
        assert "with single quotes" not in result
        assert "return True" in result

    def test_remove_comments_mixed(self):
        """Test removal of both single line and multiline comments."""
        code = '''def factorial(n):
    """Calculate factorial."""
    # Base case
    if n <= 1:
        return 1  # Return 1 for base case
    return n * factorial(n - 1)  # Recursive call'''

        result = self.processor.remove_comments(code)
        assert "Calculate factorial" not in result
        assert "Base case" not in result
        assert "Return 1 for base case" not in result
        assert "Recursive call" not in result
        assert "if n <= 1:" in result
        assert "return 1" in result
        assert "return n * factorial(n - 1)" in result

    def test_remove_comments_empty_string(self):
        """Test with empty string."""
        result = self.processor.remove_comments("")
        assert result == ""

    def test_extract_code_block_from_response_markdown(self):
        """Test extracting code from markdown code blocks."""
        response = '''Here's the solution:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

This implements factorial recursively.'''

        result = self.processor.extract_code_block_from_response(response)
        expected = '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)'''

        assert result.strip() == expected.strip()

    def test_extract_code_block_from_response_no_markdown(self):
        """Test when there's no markdown code block."""
        response = '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)'''

        result = self.processor.extract_code_block_from_response(response)
        # Should return empty string when no ```python block is found
        assert result == ""

    def test_extract_code_block_from_response_multiple_blocks(self):
        """Test extracting from multiple code blocks (should get first one)."""
        response = '''Here are two solutions:

```python
def solution1(n):
    return n + 1
```

And another:

```python
def solution2(n):
    return n + 2
```'''

        result = self.processor.extract_code_block_from_response(response)
        assert "solution1" in result
        assert "solution2" not in result

    def test_extract_code_block_from_response_empty_string(self):
        """Test with empty string."""
        result = self.processor.extract_code_block_from_response("")
        assert result == ""

    def test_extract_function_with_helpers_simple(self):
        """Test extracting function with no helpers."""
        code = '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)'''

        result = self.processor.extract_function_with_helpers(
            code, "factorial")
        expected_lines = [
            "if n <= 1:",
            "return 1",
            "return n * factorial(n - 1)"
        ]

        for line in expected_lines:
            assert line in result

    def test_extract_function_with_helpers_with_helper(self):
        """Test extracting function with helper functions."""
        code = '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def count_primes(numbers):
    count = 0
    for num in numbers:
        if is_prime(num):
            count += 1
    return count'''

        result = self.processor.extract_function_with_helpers(
            code, "count_primes")

        # Should include the helper function
        assert "def is_prime(n):" in result
        assert "if n < 2:" in result

        # Should include the main function body
        assert "count = 0" in result
        assert "for num in numbers:" in result

    def test_extract_function_with_helpers_with_imports(self):
        """Test extracting function with imports."""
        code = '''import math
from typing import List

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)'''

        result = self.processor.extract_function_with_helpers(
            code, "calculate_distance")

        # Should include imports with proper indentation
        assert "import math" in result
        assert "from typing import List" in result

        # Should include function body
        assert "return math.sqrt" in result

    def test_extract_function_with_helpers_target_not_found(self):
        """Test when target function is not found."""
        code = '''def helper_function():
    return 42'''

        result = self.processor.extract_function_with_helpers(
            code, "nonexistent_function")
        assert result == ""

    def test_extract_function_with_helpers_empty_code(self):
        """Test with empty code."""
        result = self.processor.extract_function_with_helpers(
            "", "some_function")
        assert result == ""
