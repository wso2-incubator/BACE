from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_INITIAL,
    BaseOperatorInput,
    InitialInput,
    OperatorOutput,
)
from coevolution.strategies.operators.base_llm_operator import UnsupportedOperatorInput
from coevolution.strategies.operators.differential_llm_operator import (  # DifferentialCrossoverInput,  # TODO: This class doesn't exist in the source - test needs updating
    OPERATION_DISCOVERY,
    DifferentialGenScriptInput,
    DifferentialInputOutput,
    DifferentialLLMOperator,
)

# --- Fixtures ---


@pytest.fixture
def mock_llm() -> MagicMock:
    """Returns a mock LLM instance."""
    llm = MagicMock()
    # Default behavior: return a simple code block
    llm.generate.return_value = "```python\ndef generate_test_inputs(n): pass\n```"
    return llm


@pytest.fixture
def operator(mock_llm: MagicMock) -> DifferentialLLMOperator:
    """Returns the DifferentialLLMOperator instance with mocked LLM."""
    return DifferentialLLMOperator(llm=mock_llm)


@pytest.fixture
def sample_io_pairs() -> list[DifferentialInputOutput]:
    return [
        {"inputdata": {"x": 1}, "output": 1},
        {"inputdata": {"x": 2}, "output": 2},
        {"inputdata": {"x": 3}, "output": 3},
        {"inputdata": {"x": 4}, "output": 4},
    ]


# --- Tests ---


def test_supported_operations(operator: DifferentialLLMOperator) -> None:
    """Verify the operator declares support for Discovery and Crossover."""
    ops = operator.supported_operations()
    assert OPERATION_DISCOVERY in ops
    assert OPERATION_CROSSOVER in ops


@patch("coevolution.strategies.operators.differential_llm_operator.transformation")
def test_generate_initial_snippets(
    mock_transform_module: MagicMock, operator: DifferentialLLMOperator
) -> None:
    """Verify initialization creates a scaffold class block."""
    # Cast module mock for typing
    mock_transform = mock_transform_module

    # Setup
    input_dto = InitialInput(
        operation=OPERATION_INITIAL,
        question_content="Sort list",
        starter_code="def sort(x): pass",
        population_size=10,
    )
    mock_transform.setup_unittest_class_from_starter_code.return_value = (
        "class TestSolution: ..."
    )

    # Execute
    result, block = operator.generate_initial_snippets(input_dto)

    # Assert
    assert isinstance(result, OperatorOutput)
    assert len(result.results) == 0  # Should be empty initially
    assert block == "class TestSolution: ..."
    mock_transform.setup_unittest_class_from_starter_code.assert_called_with(
        "def sort(x): pass"
    )


@patch("coevolution.strategies.operators.differential_llm_operator.transformation")
def test_get_test_method_from_io(
    mock_transform_module: MagicMock,
    operator: DifferentialLLMOperator,
    sample_io_pairs: list[DifferentialInputOutput],
) -> None:
    """Verify deterministic generation of test methods from IO pairs."""
    mock_transform = mock_transform_module

    starter_code = "class Solution:\n    def f(self, x):\n        return x"
    parent_ids = ["P1", "P2"]

    operator.get_test_method_from_io(
        starter_code, sample_io_pairs, parent_ids, io_index=0
    )

    # Implementation passes suffix string (P1_P2_0) not parent_ids list
    mock_transform.build_test_method_from_io.assert_called_once_with(
        starter_code, sample_io_pairs, "P1_P2_0"
    )


@patch("coevolution.strategies.operators.differential_llm_operator.transformation")
def test_handle_generation_script_success(
    mock_transform_module: MagicMock,
    operator: DifferentialLLMOperator,
    mock_llm: MagicMock,
) -> None:
    """
    Test the flow of generating a fuzzer script:
    1. Formats Prompt
    2. Calls LLM
    3. Extracts Code
    4. Appends execution call (print(generate...))
    """
    mock_transform = mock_transform_module

    # Setup
    mock_llm.generate.return_value = (
        "```python\ndef generate_test_inputs(n):\n    return []\n```"
    )
    mock_transform.remove_if_main_block.return_value = (
        "def generate_test_inputs(n):\n    return []"
    )

    input_dto = DifferentialGenScriptInput(
        operation=OPERATION_DISCOVERY,
        question_content="Question",
        equivalent_code_snippet_1="def f1(): pass",
        equivalent_code_snippet_2="def f2(): pass",
        passing_differential_test_io_pairs=[],
        num_inputs_to_generate=50,
    )

    # Execute
    output = operator.apply(input_dto)

    # Assert
    assert len(output.results) == 1
    script = output.results[0].snippet

    # Check LLM call
    mock_llm.generate.assert_called_once()
    args, _ = mock_llm.generate.call_args
    assert "def f1(): pass" in args[0]
    assert "def f2(): pass" in args[0]

    # Check Post-processing (The Critical Logic)
    # It must append the print statement with the requested number of inputs
    assert "print(generate_test_inputs(50))" in script
    assert "def generate_test_inputs" in script


def test_apply_invalid_input(operator: DifferentialLLMOperator) -> None:
    """Verify that passing an unknown DTO raises UnsupportedOperatorInput."""

    @dataclass(frozen=True)
    class RandomInput(BaseOperatorInput):
        pass

    bad_input = RandomInput(operation="random", question_content="?")

    with pytest.raises(UnsupportedOperatorInput):
        operator.apply(bad_input)
    bad_input = RandomInput(operation="random", question_content="?")

    with pytest.raises(UnsupportedOperatorInput):
        operator.apply(bad_input)
