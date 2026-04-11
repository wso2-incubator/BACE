from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from coevolution.populations.differential.operators.llm_operator import (
    DifferentialInputOutput,
    DifferentialLLMOperator,
)


@pytest.fixture
def mock_llm() -> MagicMock:
    """Returns a mock LLM instance."""
    llm = MagicMock()
    llm.generate.return_value = "```python\ndef generate_test_inputs(n): pass\n```"
    return llm


@pytest.fixture
def mock_parser() -> MagicMock:
    """Returns a mock language parser."""
    parser = MagicMock()
    parser.extract_code_blocks.side_effect = (
        lambda x: [x]
        if "```python" not in x
        else [x.split("```python")[1].split("```")[0].strip()]
    )
    return parser


@pytest.fixture
def mock_composer() -> MagicMock:
    """Returns a mock language composer."""
    composer = MagicMock()
    composer.generate_test_case.side_effect = (
        lambda input_str,
        output_str,
        starter_code,
        test_number: (
            f"import json\n\n\n"
            f"def test_case_{test_number}():\n"
            f"    assert f(json.loads({input_str!r})) == {output_str}"
        )
    )
    return composer


@pytest.fixture
def diff_llm_operator(
    mock_llm: MagicMock, mock_parser: MagicMock, mock_composer: MagicMock
) -> DifferentialLLMOperator:
    """Returns the DifferentialLLMOperator instance with mocked dependencies."""
    return DifferentialLLMOperator(
        llm=mock_llm,
        parser=mock_parser,
        composer=mock_composer,
        language_name="python",
    )


@pytest.fixture
def sample_io_pairs() -> list[DifferentialInputOutput]:
    return [
        {"input_arg": {"x": 1}, "output": 1},
        {"input_arg": {"x": 2}, "output": 2},
        {"input_arg": {"x": 3}, "output": 3},
        {"input_arg": {"x": 4}, "output": 4},
    ]


def test_get_test_method_from_io(
    diff_llm_operator: DifferentialLLMOperator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify generation of test methods from IO pairs."""
    starter_code = "class Solution:\n    def f(self, x):\n        return x"
    parent_ids = ["P1", "P2"]
    sample_io_pairs = [
        {"input_arg": {"x": 1}, "output": 1},
        {"input_arg": {"x": 2}, "output": 2},
    ]

    # This calls into PythonLanguage.compose_test_case internally in DifferentialLLMOperator.get_test_method_from_io
    # Since DifferentialLLMOperator uses hardcoded PythonLanguage for tests, we can test it directly.
    result = diff_llm_operator.get_test_method_from_io(
        starter_code, cast(list[DifferentialInputOutput], sample_io_pairs), parent_ids, io_index=0
    )

    assert isinstance(result, str)
    assert "def test_" in result or "test_" in result
    assert "assert" in result


def test_get_test_method_from_io_standalone(
    diff_llm_operator: DifferentialLLMOperator,
    sample_io_pairs: list[DifferentialInputOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify test generation for standalone functions."""
    starter_code = "def f(x: int) -> int:\n    return x"
    parent_ids = ["S1", "S2"]

    result = diff_llm_operator.get_test_method_from_io(
        starter_code, sample_io_pairs, parent_ids, io_index=0
    )

    assert isinstance(result, str)
    assert "def test_" in result or "test_" in result
    assert "assert" in result
    assert "f(" in result
