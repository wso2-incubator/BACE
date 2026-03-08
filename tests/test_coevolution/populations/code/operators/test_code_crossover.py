from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import OPERATION_CROSSOVER, CoevolutionContext
from coevolution.populations.code.operators.crossover import CodeCrossoverOperator


@pytest.fixture
def mock_context():
    context = MagicMock(spec=CoevolutionContext)
    context.code_population = MagicMock()
    context.code_population.generation = 1
    context.problem = MagicMock()
    context.problem.question_content = "Combine code."
    context.problem.starter_code = "def foo():"
    return context


@pytest.fixture
def mock_parents():
    p1 = CodeIndividual(
        snippet="p1",
        probability=0.4,
        creation_op="ini",
        generation_born=1,
        parents={"code": [], "test": []},
    )
    p2 = CodeIndividual(
        snippet="p2",
        probability=0.6,
        creation_op="ini",
        generation_born=1,
        parents={"code": [], "test": []},
    )
    return [p1, p2]


@pytest.fixture
def crossover_operator():
    mock_llm = MagicMock()
    mock_parser = MagicMock()
    mock_selector = MagicMock()
    mock_prob = MagicMock()

    op = CodeCrossoverOperator(
        llm=mock_llm,
        parser=mock_parser,
        language_name="python",
        parent_selector=mock_selector,
        prob_assigner=mock_prob,
    )
    return op


def test_code_crossover_execute_success(crossover_operator, mock_context, mock_parents):
    # Setup
    crossover_operator.parent_selector.select_parents.return_value = mock_parents
    crossover_operator._llm.generate.return_value = "```python\nchild code\n```"
    crossover_operator.parser.extract_code_blocks.return_value = [
        "child code"
    ]
    crossover_operator.prob_assigner.assign_probability.return_value = 0.5

    # Execute
    results = crossover_operator.execute(mock_context)

    # Verify
    assert len(results) == 1
    child = results[0]
    assert child.snippet == "child code"
    assert child.probability == 0.5
    assert child.creation_op == OPERATION_CROSSOVER
    assert child.generation_born == 2
    assert set(child.parents["code"]) == {mock_parents[0].id, mock_parents[1].id}


def test_code_crossover_insufficient_parents(crossover_operator, mock_context):
    # Setup
    crossover_operator.parent_selector.select_parents.return_value = [MagicMock()]

    # Execute
    results = crossover_operator.execute(mock_context)

    # Verify
    assert results == []
    crossover_operator._llm.generate.assert_not_called()
    # Verify
    assert results == []
    crossover_operator._llm.generate.assert_not_called()
