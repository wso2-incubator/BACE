from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from coevolution.populations.unittest.operators.mutation import UnittestMutationOperator
from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import CoevolutionContext, OPERATION_MUTATION

@pytest.fixture
def mock_context():
    context = MagicMock(spec=CoevolutionContext)
    # Important: it uses context.test_populations["unittest"]
    mock_pop = MagicMock()
    mock_pop.generation = 1
    context.test_populations = {"unittest": mock_pop}
    context.problem = MagicMock()
    context.problem.question_content = "Test Mutation"
    return context

@pytest.fixture
def mock_parent():
    return TestIndividual(
        snippet="def test_foo(): pass",
        probability=0.5,
        creation_op="initial",
        generation_born=1,
        parents={"code": [], "test": []}
    )

@pytest.fixture
def unittest_mutation_operator():
    mock_llm = MagicMock()
    mock_language = MagicMock()
    mock_language.language = "python"
    mock_selector = MagicMock()
    mock_prob = MagicMock()
    
    op = UnittestMutationOperator(
        llm=mock_llm,
        language_adapter=mock_language,
        parent_selector=mock_selector,
        prob_assigner=mock_prob
    )
    return op

def test_unittest_mutation_execute_success(unittest_mutation_operator, mock_context, mock_parent):
    # Setup
    unittest_mutation_operator.parent_selector.select_parents.return_value = [mock_parent]
    unittest_mutation_operator._llm.generate.return_value = "```python\ndef test_mutated(): pass\n```"
    unittest_mutation_operator.language_adapter.extract_code_blocks.return_value = ["def test_mutated(): pass"]
    unittest_mutation_operator.prob_assigner.assign_probability.return_value = 0.5
    
    # We need to mock _extract_first_test_function because it's called on 'self'
    with patch.object(UnittestMutationOperator, '_extract_first_test_function', return_value="def test_mutated(): pass"):
        # Execute
        results = unittest_mutation_operator.execute(mock_context)
    
        # Verify
        assert len(results) == 1
        mutated = results[0]
        assert isinstance(mutated, TestIndividual)
        assert mutated.snippet == "def test_mutated(): pass"
        assert mutated.creation_op == OPERATION_MUTATION
        assert mutated.generation_born == 2
        assert mutated.parents["test"] == [mock_parent.id]

def test_unittest_mutation_no_parents(unittest_mutation_operator, mock_context):
    # Setup
    unittest_mutation_operator.parent_selector.select_parents.return_value = []
    
    # Execute
    results = unittest_mutation_operator.execute(mock_context)
    
    # Verify
    assert results == []
    unittest_mutation_operator._llm.generate.assert_not_called()
