"""Tests for AdversarialPropertyRefiner."""

import json
from unittest.mock import MagicMock

import pytest

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import CoevolutionContext, Problem, Test
from coevolution.core.population import TestPopulation
from coevolution.populations.property.operators.refiner import AdversarialPropertyRefiner
from coevolution.strategies.probability.assigner import ProbabilityAssigner
from coevolution.strategies.selection.parent_selection import RouletteWheelParentSelection
from infrastructure.llm_client import LLMClient


@pytest.fixture
def mock_llm() -> MagicMock:
    return MagicMock(spec=LLMClient)


@pytest.fixture
def mock_parser() -> MagicMock:
    parser = MagicMock()
    parser.extract_code_blocks.side_effect = lambda x: [x] if "def property" in x else []
    parser.is_syntax_valid.return_value = True
    parser.remove_main_block.side_effect = lambda x: x
    return parser


@pytest.fixture
def mock_context() -> MagicMock:
    context = MagicMock(spec=CoevolutionContext)
    context.problem = Problem(
        question_title="X",
        question_content="Solve X",
        question_id="X",
        starter_code="def solve(): pass",
        public_test_cases=[Test(input="{}", output="null")],
        private_test_cases=[],
    )
    context.test_populations = {}
    context.generation = 1
    return context


@pytest.fixture
def refiner(mock_llm, mock_parser) -> AdversarialPropertyRefiner:
    return AdversarialPropertyRefiner(
        llm=mock_llm,
        parser=mock_parser,
        language_name="python",
        parent_selector=RouletteWheelParentSelection(),
        prob_assigner=ProbabilityAssigner(initial_prior=0.5),
    )


class TestAdversarialPropertyRefiner:
    def test_execute_success(
        self, refiner, mock_llm, mock_context
    ) -> None:
        # 1. Setup population
        parent = TestIndividual(
            snippet="def property_test(i, o): return True",
            probability=0.2,
            creation_op="init",
            generation_born=0,
        )
        pop = TestPopulation(individuals=[parent])
        mock_context.test_populations = {"property": pop}
        
        # 2. Mock LLM responses
        # Response 1: Counter-example with reasoning
        ce_json = json.dumps({"inputdata": "{}", "output": "null"})
        mock_llm.generate.side_effect = [
            f"<reasoning>Test reasoning</reasoning><counter_example>{ce_json}</counter_example>",  # Phase 1
            "def property_refined(i, o): return True"        # Phase 2
        ]
        
        # 3. Execute
        offspring = refiner.execute(mock_context)
        
        assert len(offspring) == 1
        assert offspring[0].creation_op == "adversarial_refinement"
        assert offspring[0].parents["test"] == [parent.id]
        assert offspring[0].snippet == "def property_refined(i, o): return True"
        assert offspring[0].metadata["reasoning"] == "Test reasoning"
        assert "counter_example" in offspring[0].metadata

    def test_execute_no_counter_example(self, refiner, mock_llm, mock_context) -> None:
        # Setup population
        parent = TestIndividual(
            snippet="def property_test(i, o): return True",
            probability=0.2,
            creation_op="init",
            generation_born=0,
        )
        pop = TestPopulation(individuals=[parent])
        mock_context.test_populations = {"property": pop}
        
        # Mock LLM response: Reasoning only, no counter-example tags
        mock_llm.generate.return_value = "<reasoning>The property test correctly validates the sorting algorithm.</reasoning> Everything is fine."
        
        # Execute
        offspring = refiner.execute(mock_context)
        
        assert len(offspring) == 0

    def test_empty_population(self, refiner, mock_context) -> None:
        mock_context.test_populations = {"property": TestPopulation(individuals=[])}
        offspring = refiner.execute(mock_context)
        assert len(offspring) == 0
