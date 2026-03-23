import os
import sys
from unittest.mock import MagicMock
from loguru import logger
from typing import Optional

# Add src to path if necessary
sys.path.append(os.path.abspath("src"))

import pytest
from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import CoevolutionContext, Problem
from coevolution.core.population import CodePopulation
from coevolution.populations.differential.operators.discovery import DifferentialDiscoveryOperator
from coevolution.populations.differential.operators.llm_operator import DifferentialLLMOperator
from coevolution.populations.differential.finder import DifferentialFinder
from coevolution.populations.differential.types import FunctionallyEquivGroup
from infrastructure.languages.python.adapter import PythonLanguage
from infrastructure.llm_client.factory import create_llm_client
from infrastructure.llm_client.base import LLMClient
from infrastructure.sandbox import SandboxConfig

@pytest.mark.integration
def test_discovery_operator_gpt5() -> Optional[int]:
    """
    Integration test for DifferentialDiscoveryOperator using GPT-5-mini.
    
    This test provides two sort functions:
    Code A: Correct (uses sorted())
    Code B: Buggy (returns the list unsorted)
    
    The LLM should generate test inputs for these, the differential finder
    should find divergences, and the operator should produce test offspring.
    """
    # 1. Setup LLM
    try:
        llm: LLMClient = create_llm_client(
            provider="openai",
            model="gpt-5-mini",
            reasoning_effort="minimal"
        )
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        return None

    # 2. Setup Language & Services
    lang: PythonLanguage = PythonLanguage()
    sandbox_config = SandboxConfig(timeout=5, max_memory_mb=200, max_output_size=50_000)
    
    differential_finder = DifferentialFinder(
        parser=lang.parser,
        composer=lang.composer,
        runtime=lang.runtime,
        sandbox_config=sandbox_config,
        enable_multiprocessing=True,
        cpu_workers=4
    )
    
    llm_service = DifferentialLLMOperator(
        llm=llm,
        parser=lang.parser,
        composer=lang.composer,
        language_name="python"
    )

    # 3. Mocks for the operator
    parent_selector = MagicMock()
    prob_assigner = MagicMock()
    prob_assigner.initial_prior = 0.5
    prob_assigner.assign_probability.return_value = 0.5
    
    func_eq_selector = MagicMock()

    discovery_operator = DifferentialDiscoveryOperator(
        llm=llm,
        parser=lang.parser,
        language_name="python",
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
        llm_service=llm_service,
        differential_finder=differential_finder,
        func_eq_selector=func_eq_selector,
        divergence_limit=3,
        max_pairs_per_group=1,
        num_passing_tests_to_sample=0,
        llm_workers=1
    )

    # 4. Setup Problem & Context
    starter_code = '''class Solution:
    def sort_array(self, nums: list[int]) -> list[int]:
        pass
'''
    problem = Problem(
        question_title="Sort Array",
        question_content="Given an array of integers nums, sort the array in ascending order and return it.",
        question_id="sort-array",
        starter_code=starter_code,
        public_test_cases=[],
        private_test_cases=[]
    )
    
    context = MagicMock(spec=CoevolutionContext)
    context.problem = problem
    context.code_population = CodePopulation(individuals=[], generation=0)
    
    # Code A: Correct
    code_a_snippet = '''class Solution:
    def sort_array(self, nums: list[int]) -> list[int]:
        return sorted(nums)
'''
    code_a = CodeIndividual(snippet=code_a_snippet, probability=0.8, creation_op="init", generation_born=0)
    
    # Code B: Buggy (returns unmodified list)
    code_b_snippet = '''class Solution:
    def sort_array(self, nums: list[int]) -> list[int]:
        return nums
'''
    code_b = CodeIndividual(snippet=code_b_snippet, probability=0.7, creation_op="init", generation_born=0)
    
    # Group them together
    group = FunctionallyEquivGroup(
        code_individuals=[code_a, code_b],
        passing_test_individuals={}
    )
    func_eq_selector.select_functionally_equivalent_codes.return_value = [group]
    
    # 5. Execute Pipeline
    logger.info("Starting Differential Discovery Operator with GPT-5-mini...")
    offspring = discovery_operator.execute(context)
    
    logger.info(f"Created {len(offspring)} test individuals from divergence")
    for i, ind in enumerate(offspring):
        logger.info(f"Offspring {i+1} snippet:\n{ind.snippet}")
        
    assert len(offspring) > 0, "Failed to create any offspring test individuals."
    logger.success(f"SUCCESS: Created {len(offspring)} individuals.")
    
if __name__ == "__main__":
    test_discovery_operator_gpt5()
