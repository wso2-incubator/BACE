"""
Tests for the CodeProcessor class.
"""

import pytest
from src.common.code_processor import CodeProcessor


class TestCodeProcessor:
    """Test cases for CodeProcessor."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.processor = CodeProcessor()

    def test_extract_function_name_from_problem_basic(self) -> None:
        """Test basic function name extraction."""
        prompt = '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if numbers are close. """'''

        result = self.processor.extract_function_name_from_problem(prompt)
        assert result == "has_close_elements"

    def test_extract_function_name_from_problem_multiline(self) -> None:
        """Test function name extraction with multiline definition."""
        prompt = '''from typing import List

def factorial(n: int) -> int:
    """Calculate factorial of n."""
    pass'''

        result = self.processor.extract_function_name_from_problem(prompt)
        assert result == "factorial"

    def test_extract_function_name_from_problem_no_function(self) -> None:
        """Test when no function is found."""
        prompt = '''# Just a comment
x = 5'''

        result = self.processor.extract_function_name_from_problem(prompt)
        assert result == ""

    def test_extract_function_name_from_problem_empty_string(self) -> None:
        """Test with empty string."""
        result = self.processor.extract_function_name_from_problem("")
        assert result == ""

    def test_remove_comments_single_line(self) -> None:
        """Test removal of single line comments."""
        code = '''def test():
    # This is a comment
    return 42'''

        result = self.processor.remove_comments(code)
        expected = '''def test():
    
    return 42'''
        assert result == expected

    def test_remove_comments_multiline_triple_quotes(self) -> None:
        """Test removal of multiline comments with triple quotes."""
        code = '''def test():
    """
    This is a multiline
    comment
    """
    return 42'''

        result = self.processor.remove_comments(code)
        expected = '''def test():
    
    return 42'''
        assert result == expected

    def test_remove_comments_multiline_single_quotes(self) -> None:
        """Test removal of multiline comments with single quotes."""
        code = """def test():
    '''
    This is a multiline
    comment
    '''
    return 42"""

        result = self.processor.remove_comments(code)
        expected = '''def test():
    
    return 42'''
        assert result == expected

    def test_remove_comments_mixed(self) -> None:
        """Test removal of mixed comment types."""
        code = '''def test():
    # Single line comment
    """
    Multiline comment
    """
    return 42  # Inline comment'''

        result = self.processor.remove_comments(code)
        expected = '''def test():
    
    
    return 42  '''
        assert result == expected

    def test_remove_comments_empty_string(self) -> None:
        """Test with empty string."""
        result = self.processor.remove_comments("")
        assert result == ""

    def test_extract_code_block_from_response_markdown(self) -> None:
        """Test extraction from markdown code blocks."""
        response = '''Here's the solution:

```python
def add(a, b):
    return a + b
```

That should work!'''

        result = self.processor.extract_code_block_from_response(response)
        expected = "def add(a, b):\n    return a + b"
        assert result == expected

    def test_extract_code_block_from_response_no_markdown(self) -> None:
        """Test when no markdown blocks exist."""
        response = "Just plain text with no code blocks."

        result = self.processor.extract_code_block_from_response(response)
        assert result == response

    def test_extract_code_block_from_response_multiple_blocks(self) -> None:
        """Test extraction when multiple code blocks exist."""
        response = '''Here are two solutions:

```python
def add(a, b):
    return a + b
```

```python
def multiply(a, b):
    return a * b
```'''

        result = self.processor.extract_code_block_from_response(response)
        expected = "def add(a, b):\n    return a + b"
        assert result == expected

    def test_extract_code_block_from_response_from_multile_non_python_blocks(self) -> None:
        """Test extraction when multiple non-Python code blocks exist, but we want Python."""
        response = '''Here's some code:

```javascript
function add(a, b) {
    return a + b;
}
```

```python
def add(a, b):
    return a + b
```

```bash
echo "Hello"
```'''

        result = self.processor.extract_code_block_from_response(response)
        expected = "def add(a, b):\n    return a + b"
        assert result == expected

    def test_extract_code_block_from_response_empty_string(self) -> None:
        """Test with empty string."""
        result = self.processor.extract_code_block_from_response("")
        assert result == ""

    def test_extract_function_with_helpers_simple(self) -> None:
        """Test extraction of function without helpers."""
        code = '''def target_function():
    return 42'''

        result = self.processor.extract_function_with_helpers(
            code, "target_function")
        expected = '''def target_function():
    return 42'''
        assert result == expected

    def test_extract_function_with_helpers_with_helper(self) -> None:
        """Test extraction of function with helper functions."""
        code = '''def helper_function():
    return 10

def target_function():
    return helper_function() + 32

def another_function():
    return 0'''

        result = self.processor.extract_function_with_helpers(
            code, "target_function")

        # Should include the target function
        assert "def target_function():" in result
        assert "return helper_function() + 32" in result

        # Should include the helper function
        assert "def helper_function():" in result
        assert "return 10" in result

        # Should NOT include the unrelated function
        assert "def another_function():" not in result

    def test_extract_function_with_helpers_with_imports(self) -> None:
        """Test extraction with imports."""
        code = '''import math
from typing import List

def helper_function(x):
    return math.sqrt(x)

def target_function(numbers: List[float]):
    return helper_function(sum(numbers))'''

        result = self.processor.extract_function_with_helpers(
            code, "target_function")

        # Should include imports with proper indentation
        assert "import math" in result
        assert "from typing import List" in result

        # Should include function body
        assert "return math.sqrt" in result

    def test_extract_function_with_helpers_target_not_found(self) -> None:
        """Test when target function is not found."""
        code = '''def helper_function():
    return 42'''

        result = self.processor.extract_function_with_helpers(
            code, "nonexistent_function")
        assert result == ""

    def test_extract_function_with_helpers_empty_code(self) -> None:
        """Test with empty code."""
        result = self.processor.extract_function_with_helpers(
            "", "some_function")
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__])
