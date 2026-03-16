"""Integration tests for AdversarialPropertyRefiner using real LLM."""

import json
import os

import pytest

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import CoevolutionContext, Problem, Test
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.populations.property.operators.refiner import (
    AdversarialPropertyRefiner,
)
from coevolution.strategies.probability.assigner import ProbabilityAssigner
from coevolution.strategies.selection.parent_selection import (
    ReverseRouletteWheelParentSelection,
)
from infrastructure.languages.python import PythonLanguage
from infrastructure.llm_client import LLMClient, create_llm_client

pytestmark = pytest.mark.integration

# --- Skip guard ---
REQUIRES_OPENAI = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-LLM integration tests",
)

# --- Problem definition ---
ADD_PROBLEM = Problem(
    question_title="add",
    question_content="Given two integers x and y, return their sum. Example: add(2, 3) should return 5.",
    question_id="integration/add",
    starter_code="def add(x: int, y: int) -> int:\n    ...\n",
    public_test_cases=[
        Test(input=json.dumps({"x": 2, "y": 3}), output="5"),
        Test(input=json.dumps({"x": 0, "y": 0}), output="0"),
        Test(input=json.dumps({"x": -1, "y": 1}), output="0"),
    ],
    private_test_cases=[],
)


# --- LLM response printer ---
class PrintingLLMClient(LLMClient):
    """Thin proxy that prints every LLM response to stdout as it arrives."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    def generate(self, prompt: object, **kwargs: object) -> str:
        response = self._inner.generate(prompt, **kwargs)
        print(f"\n{'═' * 72}")
        print("[LLM RESPONSE]")
        print(response)
        print(f"{'═' * 72}\n")
        return response


# --- Fixtures ---
@pytest.fixture
def llm_client() -> LLMClient:
    inner = create_llm_client(
        provider="openai",
        model="gpt-5-mini",
        reasoning_effort="minimal",
    )
    return PrintingLLMClient(inner)


@pytest.fixture
def refiner(llm_client: LLMClient) -> AdversarialPropertyRefiner:
    python_lang = PythonLanguage()
    return AdversarialPropertyRefiner(
        llm=llm_client,
        parser=python_lang.parser,
        language_name="python",
        parent_selector=ReverseRouletteWheelParentSelection(),
        prob_assigner=ProbabilityAssigner(initial_prior=0.5),
        max_falsification_attempts=3,
    )


@REQUIRES_OPENAI
class TestAdversarialRefinerIntegration:
    """Integration tests for AdversarialPropertyRefiner."""

    def test_refinement_of_subtle_bad_property(
        self, refiner: AdversarialPropertyRefiner
    ) -> None:
        """Test that a subtle bad property is correctly refined."""
        # This property is incorrect because it fails for non-positive numbers
        bad_snippet = (
            "from typing import Any\n"
            "import json\n"
            "def property_result_greater_than_inputs(input_arg: dict, output: Any) -> bool:\n"
            "    x, y = input_arg['x'], input_arg['y']\n"
            "    res = int(output)\n"
            "    return res > x and res > y\n"
        )

        parent = TestIndividual(
            snippet=bad_snippet,
            probability=0.1,  # Low probability to be selected
            creation_op="init",
            generation_born=0,
            metadata={"falsification_attempts": 0},
        )

        pop = TestPopulation(individuals=[parent])
        context = CoevolutionContext(
            problem=ADD_PROBLEM,
            code_population=CodePopulation(individuals=[]),
            test_populations={"property": pop},
            interactions={},
        )

        offspring = refiner.execute(context)

        assert len(offspring) == 1
        child = offspring[0]
        assert child.creation_op == "adversarial_refinement"
        assert child.parents["test"] == [parent.id]

        print(f"\n[TRACE] Refined snippet:\n{child.snippet}")
        # The refined snippet should handle <= 0 cases.
        # We can't strictly assert the code content as it's LLM generated, but it should exist.
        assert "def property_" in child.snippet

    def test_retry_exhaustion_on_correct_property(
        self, refiner: AdversarialPropertyRefiner
    ) -> None:
        """Test that the refiner gives up after several failed attempts."""
        # This property is vacuously correct and robust
        correct_snippet = (
            "def property_always_true(input_arg: dict, output: Any) -> bool:\n"
            "    return True\n"
        )

        parent = TestIndividual(
            snippet=correct_snippet,
            probability=0.9,
            creation_op="init",
            generation_born=0,
            metadata={"falsification_attempts": 0},
        )

        pop = TestPopulation(individuals=[parent])
        context = CoevolutionContext(
            problem=ADD_PROBLEM,
            code_population=CodePopulation(individuals=[]),
            test_populations={"property": pop},
            interactions={},
        )

        # 1st attempt
        res1 = refiner.execute(context)
        assert len(res1) == 0
        assert parent.metadata["falsification_attempts"] == 1

        # 2nd attempt
        res2 = refiner.execute(context)
        assert len(res2) == 0
        assert parent.metadata["falsification_attempts"] == 2

        # 3rd attempt
        res3 = refiner.execute(context)
        assert len(res3) == 0
        assert parent.metadata["falsification_attempts"] == 3

        # 4th attempt - should hit the limit and skip before calling LLM
        res4 = refiner.execute(context)
        assert len(res4) == 0
        assert parent.metadata["falsification_attempts"] == 3  # Should remain at limit

        # 4th attempt - should hit the limit and skip before calling LLM
        res4 = refiner.execute(context)
        assert len(res4) == 0
        assert parent.metadata["falsification_attempts"] == 3  # Should remain at limit
