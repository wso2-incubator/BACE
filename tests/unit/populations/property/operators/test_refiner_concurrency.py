import time
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import CoevolutionContext, Problem
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.populations.property.operators.refiner import AdversarialPropertyRefiner
from coevolution.strategies.selection.parent_selection import ReverseRouletteWheelParentSelection
from coevolution.strategies.probability.assigner import ProbabilityAssigner

class MockLLM:
    def __init__(self):
        self.call_count = 0
        self.lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        with self.lock:
            self.call_count += 1
        # Simulate expensive LLM call with a short delay to avoid slow tests
        time.sleep(0.005)
        # Return a response that will trigger "no counter-example found"
        return "<reasoning>No counter-example found</reasoning><is_valid>true</is_valid>"

def test_refiner_concurrency_stinker_race():
    """
    Test Scenario: "The Stinker Race"
    Verify that multiple threads selecting the same 'stinker' property
    don't trigger redundant LLM calls and update the retry counter correctly.
    """
    mock_llm = MockLLM()
    mock_parser = MagicMock()
    # Mock parser to extract reasoning
    mock_parser.extract_code_blocks.return_value = []
    
    refiner = AdversarialPropertyRefiner(
        llm=mock_llm,
        parser=mock_parser,
        language_name="python",
        parent_selector=ReverseRouletteWheelParentSelection(),
        prob_assigner=ProbabilityAssigner()
    )

    # Setup: Population with a single high-priority (low belief) individual
    stinker = TestIndividual(
        snippet="def prop(): return True",
        probability=0.001, # Low belief -> high selection probability
        creation_op="init",
        generation_born=0,
        explanation="initial"
    )
    pop = TestPopulation(individuals=[stinker])
    problem = Problem(
        question_title="test",
        question_content="test",
        question_id="test",
        starter_code="test",
        public_test_cases=[],
        private_test_cases=[]
    )
    context = CoevolutionContext(
        problem=problem,
        code_population=CodePopulation(individuals=[]),
        test_populations={"property": pop},
        interactions={}
    )

    # Execution: 10 threads hitting the refiner simultaneously
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(refiner.execute, context) for _ in range(10)]
        results = [f.result() for f in futures]

    # Assertions
    # 1. Only 1 LLM call should have been made
    assert mock_llm.call_count == 1
    
    # 2. All 10 threads should have returned empty lists (1 failed CE find, 9 skipped)
    assert all(r == [] for r in results)
    
    # 3. Falsification attempts should be exactly 1, not 10
    assert stinker.metadata.get("falsification_attempts") == 1

def test_refiner_concurrency_starvation_avoidance():
    """
    Verify that if Parent A is busy, a worker re-rolls and picks Parent B.
    """
    mock_llm = MockLLM()
    mock_parser = MagicMock()
    mock_parser.extract_code_blocks.return_value = []
    
    refiner = AdversarialPropertyRefiner(
        llm=mock_llm,
        parser=mock_parser,
        language_name="python",
        parent_selector=ReverseRouletteWheelParentSelection(),
        prob_assigner=ProbabilityAssigner()
    )

    # Two stinkers
    t1 = TestIndividual(snippet="t1", probability=0.001, creation_op="i", generation_born=0, explanation="e1")
    t2 = TestIndividual(snippet="t2", probability=0.001, creation_op="i", generation_born=0, explanation="e2")
    
    pop = TestPopulation(individuals=[t1, t2])
    problem = Problem(
        question_title="test", question_content="test", question_id="test",
        starter_code="test", public_test_cases=[], private_test_cases=[]
    )
    context = CoevolutionContext(
        problem=problem,
        code_population=CodePopulation(individuals=[]),
        test_populations={"property": pop},
        interactions={}
    )

    # 10 threads. In theory, they should be able to process BOTH t1 and t2 in parallel.
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(refiner.execute, context) for _ in range(10)]
        results = [f.result() for f in futures]

    # Assertions
    # Both stinkers should have been processed
    # (Since mock LLM takes 0.5s, the first two threads will claim t1 and t2, others will skip after 5 retries)
    assert mock_llm.call_count == 2
    assert t1.metadata.get("falsification_attempts") == 1
    assert t2.metadata.get("falsification_attempts") == 1

def test_refiner_concurrency_retry_limit_honored():
    """
    Verify that even with concurrent access, the max_falsification_attempts is honored.
    """
    mock_llm = MockLLM()
    mock_parser = MagicMock()
    mock_parser.extract_code_blocks.return_value = []

    refiner = AdversarialPropertyRefiner(
        llm=mock_llm,
        parser=mock_parser,
        language_name="python",
        parent_selector=ReverseRouletteWheelParentSelection(),
        prob_assigner=ProbabilityAssigner(),
        max_falsification_attempts=1 # Tight limit
    )

    stinker = TestIndividual(
        snippet="def prop(): return True",
        probability=0.001,
        creation_op="init",
        generation_born=0,
        explanation="initial"
    )
    # Already at the limit
    stinker.metadata["falsification_attempts"] = 1
    
    pop = TestPopulation(individuals=[stinker])
    problem = Problem(
        question_title="test",
        question_content="test",
        question_id="test",
        starter_code="test",
        public_test_cases=[],
        private_test_cases=[]
    )
    context = CoevolutionContext(
        problem=problem,
        code_population=CodePopulation(individuals=[]),
        test_populations={"property": pop},
        interactions={}
    )

    # 10 threads try to refine it
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(refiner.execute, context) for _ in range(10)]
        results = [f.result() for f in futures]

    # Assertions
    # 1. No LLM calls should be made because it's already capped
    assert mock_llm.call_count == 0
    # 2. All should return empty
    assert all(r == [] for r in results)
