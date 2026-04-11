
import os
import sys
from loguru import logger
from typing import List, Optional

# Add src to path if necessary
sys.path.append(os.path.abspath("src"))

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import PopulationConfig, Problem, Test
from coevolution.populations.unittest.operators.initializer import UnittestInitializer
from infrastructure.languages.python.adapter import PythonLanguage
from infrastructure.llm_client.factory import create_llm_client
import pytest
from infrastructure.llm_client.base import LLMClient

@pytest.mark.integration
def test_unittest_initialization_gpt5() -> Optional[int]:
    """
    Integration test for UnittestInitializer using GPT-5-mini.
    
    This test verifies that the initializer can correctly generate and
    populate a unittest population of size 20.
    """
    # 1. Setup LLM
    # Note: Ensure OPENAI_API_KEY is set in your environment
    try:
        llm: LLMClient = create_llm_client(
            provider="openai",
            model="gpt-5-mini",
            reasoning_effort="minimal"
        )
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        return None

    # 2. Setup Language
    lang: PythonLanguage = PythonLanguage()
    
    # 3. Setup Config
    # Target 20 as requested by the user to reproduce the bug
    pop_config: PopulationConfig = PopulationConfig(
        initial_prior=0.5,
        initial_population_size=20,
        max_population_size=20,
    )
    
    # 4. Setup Problem (Two Sum)
    problem: Problem = Problem(
        question_title="Two Sum",
        question_content="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.",
        question_id="two-sum",
        starter_code="class Solution:\n    def twoSum(self, nums: list[int], target: int) -> list[int]:\n        pass",
        public_test_cases=[
            Test(input="nums = [2,7,11,15], target = 9", output="[0,1]"),
            Test(input="nums = [3,2,4], target = 6", output="[1,2]"),
            Test(input="nums = [3,3], target = 6", output="[0,1]"),
        ],
        private_test_cases=[]
    )
    
    # 5. Initialize
    initializer: UnittestInitializer = UnittestInitializer(
        llm=llm,
        parser=lang.parser,
        language_name="python",
        pop_config=pop_config
    )
    
    logger.info("Starting Unittest initialization with GPT-5-mini...")
    individuals: List[TestIndividual] = initializer.initialize(problem)
    
    logger.info(f"Created {len(individuals)} individuals")
    for i, ind in enumerate(individuals):
        logger.info(f"Individual {i+1} snippet:\n{ind.snippet}")
    
    assert len(individuals) == 20, f"FAILURE: Created only {len(individuals)} individuals (expected 20)."
    logger.success("SUCCESS: Created all 20 individuals.")

if __name__ == "__main__":
    test_unittest_initialization_gpt5()
