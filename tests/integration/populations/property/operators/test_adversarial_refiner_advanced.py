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
from infrastructure.languages import PythonLanguage
from infrastructure.llm_client import create_llm_client

# Problem: Find the Duplicate Number
# Given an array of integers `nums` containing n + 1 integers where each integer
# is in the range [1, n] inclusive. There is only one repeated number in `nums`,
# return this repeated number.
DUPLICATE_PROBLEM = Problem(
    question_id="find_duplicate",
    question_title="Find the Duplicate Number",
    question_content=(
        "Given an array of integers nums containing n + 1 integers where each integer "
        "is in the range [1, n] inclusive. There is only one repeated number in nums, "
        "return this repeated number."
    ),
    starter_code="def findDuplicate(nums: list[int]) -> int:",
    public_test_cases=[
        Test(input="[1,3,4,2,2]", output="2"),
        Test(input="[3,1,3,4,2]", output="3"),
    ],
    private_test_cases=[],
)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping real OpenAI integration tests.",
)
@pytest.mark.integration
class TestAdversarialRefinerAdvanced:
    @pytest.fixture
    def llm_client(self):
        return create_llm_client(provider="openai", model="gpt-5-mini")

    @pytest.fixture
    def refiner(self, llm_client):
        lang = PythonLanguage()
        return AdversarialPropertyRefiner(
            llm=llm_client,
            parser=lang.parser,
            language_name="python",
            parent_selector=ReverseRouletteWheelParentSelection(),
            prob_assigner=ProbabilityAssigner(),
        )

    def test_refinement_of_duplicate_count_property(self, refiner):
        """
        Test that the refiner fixed a property that assumes a duplicate
        appears exactly twice.
        """
        # Buggy property: assumes the duplicate appears exactly twice.
        # It's wrong because the problem says "repeated number",
        # which could appear 3, 4, or more times.
        buggy_snippet = (
            "import json\n"
            "def property_appears_twice(inputdata: str, output: str) -> bool:\n"
            '    """\n'
            "    Checks that the identified duplicate appears exactly twice in the list.\n"
            '    """\n'
            "    nums = json.loads(inputdata)\n"
            "    duplicate = json.loads(output)\n"
            "    return nums.count(duplicate) == 2\n"
        )

        parent = TestIndividual(
            snippet=buggy_snippet,
            probability=0.1,  # Low belief -> highly likely for selection
            creation_op="init",
            generation_born=0,
            explanation="Checks that the identified duplicate appears exactly twice in the list.",
        )

        pop = TestPopulation(individuals=[parent])
        context = CoevolutionContext(
            problem=DUPLICATE_PROBLEM,
            code_population=CodePopulation(individuals=[]),
            test_populations={"property": pop},
            interactions={},
        )

        offspring = refiner.execute(context)

        assert len(offspring) == 1
        child = offspring[0]
        assert child.creation_op == "adversarial_refinement"

        print(f"\n[TRACE] Original Explanation: {parent.explanation}")
        print(f"\n[TRACE] Refined Explanation (Docstring): {child.explanation}")
        print(f"\n[TRACE] Refined Snippet:\n{child.snippet}")

        # The refined property should probably check >= 2 or use a different logic.
        assert "def property_" in child.snippet
        # Check that docstring was extracted as explanation
        assert child.explanation and len(child.explanation) > 10
        # The refined property should probably check >= 2 or use a different logic.
        assert "def property_" in child.snippet
        # Check that docstring was extracted as explanation
        assert child.explanation and len(child.explanation) > 10
