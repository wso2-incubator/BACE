from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from coevolution.core.interfaces.language import ILanguage
from coevolution.populations.differential.operators.llm_operator import (
    DifferentialInputOutput,
    DifferentialLLMOperator,
)
from coevolution.strategies.llm_base import ILanguageModel


@pytest.fixture
def mock_llm() -> MagicMock:
    """Returns a mock LLM instance."""
    llm = MagicMock()
    llm.generate.return_value = "```python\ndef generate_test_inputs(n): pass\n```"
    return llm


@pytest.fixture
def mock_language() -> MagicMock:
    """Returns a mock language adapter."""
    lang = MagicMock(spec=ILanguage)
    lang.language = "python"
    # Return a string that satisfies the "f(" check if it's there
    lang.generate_test_case.side_effect = (
        lambda input_str,
        output_str,
        starter_code,
        test_number: f"def test_case_{test_number}():\n    assert f({input_str}) == {output_str}"
    )
    lang.extract_code_blocks.side_effect = (
        lambda x: [x]
        if "```python" not in x
        else [x.split("```python")[1].split("```")[0].strip()]
    )
    return lang


@pytest.fixture
def operator(mock_llm: MagicMock, mock_language: MagicMock) -> DifferentialLLMOperator:
    """Returns the DifferentialLLMOperator instance with mocked dependencies."""
    return DifferentialLLMOperator(llm=mock_llm, language_adapter=mock_language)


@pytest.fixture
def sample_io_pairs() -> list[DifferentialInputOutput]:
    return [
        {"inputdata": {"x": 1}, "output": 1},
        {"inputdata": {"x": 2}, "output": 2},
        {"inputdata": {"x": 3}, "output": 3},
        {"inputdata": {"x": 4}, "output": 4},
    ]


def test_get_test_method_from_io(
    operator: DifferentialLLMOperator,
    sample_io_pairs: list[DifferentialInputOutput],
) -> None:
    """Verify generation of test methods from IO pairs."""
    starter_code = "class Solution:\n    def f(self, x):\n        return x"
    parent_ids = ["P1", "P2"]

    # This calls into PythonLanguage.compose_test_case internally in DifferentialLLMOperator.get_test_method_from_io
    # Since DifferentialLLMOperator uses hardcoded PythonLanguage for tests, we can test it directly.
    result = operator.get_test_method_from_io(
        starter_code, sample_io_pairs, parent_ids, io_index=0
    )

    assert isinstance(result, str)
    assert "def test_" in result or "test_" in result
    assert "assert" in result


def test_get_test_method_from_io_standalone(
    operator: DifferentialLLMOperator,
    sample_io_pairs: list[DifferentialInputOutput],
) -> None:
    """Verify test generation for standalone functions."""
    starter_code = "def f(x: int) -> int:\n    return x"
    parent_ids = ["S1", "S2"]

    result = operator.get_test_method_from_io(
        starter_code, sample_io_pairs, parent_ids, io_index=0
    )

    assert isinstance(result, str)
    assert "def test_" in result or "test_" in result
    assert "assert" in result
    assert "f(" in result
