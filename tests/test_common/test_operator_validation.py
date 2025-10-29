"""Test validation and retry logic in LLM operators."""

from unittest.mock import Mock, patch

import pytest

from common.coevolution.operators import (
    BaseLLMOperator,
    CodeValidationError,
    LLMGenerationError,
)


class ConcreteOperator(BaseLLMOperator):
    """Concrete implementation for testing."""

    def create_initial_individuals(self, population_size: int) -> list[str] | str:
        return []

    def mutate(self, individual: str, population: list[str]) -> str:
        return ""

    def crossover(self, parent1: str, parent2: str) -> str:
        return ""

    def edit(self, individual: str) -> str:
        return ""


class TestOperatorValidation:
    """Test validation methods."""

    def test_validate_empty_code_fails(self):
        """Empty code should fail validation."""
        llm_mock = Mock()
        op = ConcreteOperator(llm_mock, None)

        with pytest.raises(CodeValidationError, match="Empty or whitespace-only code"):
            op._validate_generated_code("")

        with pytest.raises(CodeValidationError, match="Empty or whitespace-only code"):
            op._validate_generated_code("   \n\t  ")

    def test_validate_syntax_error_fails(self):
        """Code with syntax errors should fail validation."""
        llm_mock = Mock()
        op = ConcreteOperator(llm_mock, None)

        with pytest.raises(CodeValidationError, match="Syntax error"):
            op._validate_generated_code(
                "def foo(\n  return 42"
            )  # Missing closing paren

        with pytest.raises(CodeValidationError, match="Syntax error"):
            op._validate_generated_code("if True\n    pass")  # Missing colon

    def test_validate_valid_code_passes(self):
        """Valid Python code should pass validation."""
        llm_mock = Mock()
        op = ConcreteOperator(llm_mock, None)

        # Should not raise (must have at least 2 lines as per current validation)
        op._validate_generated_code("def foo():\n    return 42")
        op._validate_generated_code("x = 1\ny = 2")
        op._validate_generated_code("print('hello')\nprint('world')")


class TestOperatorRetry:
    """Test retry logic."""

    def test_retry_on_empty_response(self):
        """Should retry when LLM returns empty code."""
        llm_mock = Mock()
        # First call returns empty, second call returns valid code
        llm_mock.generate.side_effect = [
            "```python\n\n```",  # Empty code block
            "```python\ndef foo():\n    return 42\n```",  # Valid code
        ]

        op = ConcreteOperator(llm_mock, None, max_retries=3)

        # Mock the extract function to return empty first, then valid
        with patch(
            "common.coevolution.operators.extract_code_block_from_response"
        ) as extract_mock:
            extract_mock.side_effect = ["", "def foo():\n    return 42"]

            result = op._generate_and_extract("test prompt")

            assert result == "def foo():\n    return 42"
            assert llm_mock.generate.call_count == 2  # Should have retried once

    def test_retry_on_syntax_error(self):
        """Should retry when LLM returns code with syntax errors."""
        llm_mock = Mock()
        # First call returns invalid syntax, second call returns valid code
        llm_mock.generate.side_effect = [
            "```python\ndef foo(\n    return 42\n```",  # Syntax error
            "```python\ndef foo():\n    return 42\n```",  # Valid code
        ]

        op = ConcreteOperator(llm_mock, None, max_retries=3)

        with patch(
            "common.coevolution.operators.extract_code_block_from_response"
        ) as extract_mock:
            extract_mock.side_effect = [
                "def foo(\n    return 42",
                "def foo():\n    return 42",
            ]

            result = op._generate_and_extract("test prompt")

            assert result == "def foo():\n    return 42"
            assert llm_mock.generate.call_count == 2  # Should have retried once

    def test_fails_after_max_retries(self):
        """Should fail after max retries exhausted."""
        llm_mock = Mock()
        # Always return empty
        llm_mock.generate.return_value = "```python\n\n```"

        op = ConcreteOperator(llm_mock, None, max_retries=2)

        with patch(
            "common.coevolution.operators.extract_code_block_from_response"
        ) as extract_mock:
            extract_mock.return_value = ""  # Always empty

            # Tenacity wraps the exception in RetryError
            with pytest.raises(Exception):  # Can be RetryError or LLMGenerationError
                op._generate_and_extract("test prompt")

            # Should have tried max_retries times
            assert llm_mock.generate.call_count == 2


class TestOperatorRetryMany:
    """Test retry logic for batch generation."""

    def test_partial_valid_results(self):
        """Should return valid blocks and skip invalid ones."""
        llm_mock = Mock()
        llm_mock.generate.return_value = "Multiple blocks"

        op = ConcreteOperator(llm_mock, None, max_retries=3)

        with patch(
            "common.coevolution.operators.extract_all_code_blocks_from_response"
        ) as extract_mock:
            # Mix of valid and invalid code
            extract_mock.return_value = [
                "def foo():\n    return 42",  # Valid
                "def bar(\n    return 1",  # Syntax error
                "def baz():\n    return 99",  # Valid
            ]

            result = op._generate_and_extract_many("test prompt", n=3)

            # Should return only the 2 valid blocks
            assert len(result) == 2
            assert "def foo():" in result[0]
            assert "def baz():" in result[1]

    def test_all_invalid_retries(self):
        """Should retry if all blocks are invalid."""
        llm_mock = Mock()
        # First call: all invalid, second call: some valid
        llm_mock.generate.side_effect = ["Response 1", "Response 2"]

        op = ConcreteOperator(llm_mock, None, max_retries=3)

        with patch(
            "common.coevolution.operators.extract_all_code_blocks_from_response"
        ) as extract_mock:
            extract_mock.side_effect = [
                ["", "def bad(\n    x"],  # All invalid
                ["def good():\n    return 1"],  # Valid
            ]

            result = op._generate_and_extract_many("test prompt", n=2)

            assert len(result) == 1
            assert "def good():" in result[0]
            assert llm_mock.generate.call_count == 2  # Should have retried
