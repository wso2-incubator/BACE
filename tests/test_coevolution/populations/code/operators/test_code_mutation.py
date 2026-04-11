from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import OPERATION_MUTATION, CoevolutionContext
from coevolution.populations.code.operators.mutation import CodeMutationOperator


@pytest.fixture
def mock_context() -> MagicMock:
    context = MagicMock(spec=CoevolutionContext)
    context.code_population = MagicMock()
    context.code_population.generation = 1
    context.problem = MagicMock()
    context.problem.question_content = "Write a function to add two numbers."
    context.problem.starter_code = "def add(a, b):"
    return context


@pytest.fixture
def mock_parent() -> CodeIndividual:
    return CodeIndividual(
        snippet="def add(a, b):\n    return a + b",
        probability=0.5,
        creation_op="initial",
        generation_born=1,
        parents={"code": [], "test": []},
    )


@pytest.fixture
def mutation_operator() -> CodeMutationOperator:
    mock_llm = MagicMock()
    mock_parser = MagicMock()
    mock_selector = MagicMock()
    mock_prob = MagicMock()

    op = CodeMutationOperator(
        llm=mock_llm,
        parser=mock_parser,
        language_name="python",
        parent_selector=mock_selector,
        prob_assigner=mock_prob,
    )
    return op


def test_code_mutation_execute_success(
    mutation_operator: MagicMock, mock_context: MagicMock, mock_parent: MagicMock
) -> None:
    # Setup
    mutation_operator.parent_selector.select_parents.return_value = [mock_parent]
    mutation_operator._llm.generate.return_value = (
        "```python\ndef add(a, b):\n    return a + b + 0\n```"
    )
    mutation_operator.parser.extract_code_blocks.return_value = [
        "def add(a, b):\n    return a + b + 0"
    ]
    mutation_operator.parser.contains_starter_code.return_value = True
    mutation_operator.parser.remove_main_block.return_value = (
        "def add(a, b):\n    return a + b + 0"
    )
    mutation_operator.prob_assigner.assign_probability.return_value = 0.6

    # Execute
    results = mutation_operator.execute(mock_context)

    # Verify
    assert len(results) == 1
    mutated = results[0]
    assert isinstance(mutated, CodeIndividual)
    assert "return a + b + 0" in mutated.snippet
    assert mutated.probability == 0.6
    assert mutated.creation_op == OPERATION_MUTATION
    assert mutated.generation_born == 2
    assert mutated.parents["code"] == [mock_parent.id]

    # Verify calls
    mutation_operator.parent_selector.select_parents.assert_called_once_with(
        mock_context.code_population, 1, mock_context
    )
    mutation_operator._llm.generate.assert_called_once()
    mutation_operator.prob_assigner.assign_probability.assert_called_once_with(
        OPERATION_MUTATION, [mock_parent.probability]
    )


def test_code_mutation_no_parents(
    mutation_operator: MagicMock, mock_context: MagicMock
) -> None:
    # Setup
    mutation_operator.parent_selector.select_parents.return_value = []

    # Execute
    results = mutation_operator.execute(mock_context)

    # Verify
    assert results == []
    mutation_operator._llm.generate.assert_not_called()
    # Verify
    assert results == []
    mutation_operator._llm.generate.assert_not_called()
