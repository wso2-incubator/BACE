from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock
import pytest

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import CoevolutionContext, Problem
from coevolution.core.population import CodePopulation
from coevolution.populations.differential.operators.discovery import DifferentialDiscoveryOperator
from coevolution.populations.differential.types import DifferentialResult, FunctionallyEquivGroup
from coevolution.populations.differential.operators.llm_operator import (
    DifferentialLLMOperator,
)

@pytest.fixture
def discovery_operator() -> DifferentialDiscoveryOperator:
    llm = MagicMock()
    parser = MagicMock()
    parent_selector = MagicMock()
    prob_assigner = MagicMock()
    llm_service = MagicMock(spec=DifferentialLLMOperator)
    differential_finder = MagicMock()
    func_eq_selector = MagicMock()
    
    # Mock prob_assigner.initial_prior
    prob_assigner.initial_prior = 0.5
    
    return DifferentialDiscoveryOperator(
        llm=llm,
        parser=parser,
        language_name="python",
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
        llm_service=llm_service,
        differential_finder=differential_finder,
        func_eq_selector=func_eq_selector
    )

def test_operation_name(discovery_operator: DifferentialDiscoveryOperator) -> None:
    from coevolution.populations.differential.types import OPERATION_DISCOVERY
    assert discovery_operator.operation_name() == OPERATION_DISCOVERY

def test_execute_full_pileline_success(discovery_operator: DifferentialDiscoveryOperator) -> None:
    # Setup context and problem
    # Problem(question_title, question_content, question_id, starter_code, public_test_cases, private_test_cases)
    problem = Problem(
        question_title="Test Title",
        question_content="Question内容",
        question_id="test_q",
        starter_code="def f(x): pass",
        public_test_cases=[],
        private_test_cases=[]
    )
    context = MagicMock(spec=CoevolutionContext)
    context.problem = problem
    context.code_population = CodePopulation(individuals=[], generation=0)
    
    # Phase 1: Candidates
    code_a = CodeIndividual(snippet="snippet A", probability=0.8, creation_op="init", generation_born=0)
    code_b = CodeIndividual(snippet="snippet B", probability=0.7, creation_op="init", generation_born=0)
    group = FunctionallyEquivGroup(
        code_individuals=[code_a, code_b],
        passing_test_individuals={}
    )
    cast(MagicMock, discovery_operator.func_eq_selector.select_functionally_equivalent_codes).return_value = [group]
    
    # Phase 2: Script generation
    cast(MagicMock, discovery_operator.llm_service.generate_script).return_value = "generator_script_content"
    
    # Phase 3: Divergence finding
    divergence = DifferentialResult(input_data={"x": 1}, output_a="1", output_b="2")
    cast(MagicMock, discovery_operator.differential_finder.find_differential).return_value = [divergence]
    
    # Mock LLM service to return a snippet
    cast(MagicMock, discovery_operator.llm_service.get_test_method_from_io).return_value = "def test_case_0(): assert f(1) == 1"
    
    # Run execute
    offspring = discovery_operator.execute(context)
    
    # Assertions
    assert len(offspring) == 2  # Two hypotheses per divergence
    assert isinstance(offspring[0], TestIndividual)
    assert offspring[0].snippet == "def test_case_0(): assert f(1) == 1"
    assert offspring[0].creation_op == discovery_operator.operation_name()
    assert offspring[0].parents["code"] == [code_a.id, code_b.id]
    
    # Verify internal calls
    cast(MagicMock, discovery_operator.llm_service.generate_script).assert_called_once()
    cast(MagicMock, discovery_operator.differential_finder.find_differential).assert_called_once_with(
        "snippet A", "snippet B", "generator_script_content", limit=discovery_operator.divergence_limit
    )

def test_candidate_selection_interleaving(discovery_operator: DifferentialDiscoveryOperator) -> None:
    # Test that multiple groups are interleaved in round-robin fashion
    context = MagicMock(spec=CoevolutionContext)
    
    group1 = FunctionallyEquivGroup(
        code_individuals=[
            CodeIndividual(snippet="", probability=0.9, creation_op="init", generation_born=0),
            CodeIndividual(snippet="", probability=0.8, creation_op="init", generation_born=0)
        ],
        passing_test_individuals={}
    )
    group2 = FunctionallyEquivGroup(
        code_individuals=[
            CodeIndividual(snippet="", probability=0.8, creation_op="init", generation_born=0),
            CodeIndividual(snippet="", probability=0.8, creation_op="init", generation_born=0)
        ],
        passing_test_individuals={}
    )
    
    cast(MagicMock, discovery_operator.func_eq_selector.select_functionally_equivalent_codes).return_value = [group1, group2]
    
    candidates = discovery_operator._select_candidates(context)
    
    # Should contain tasks from group1 then group2 (interleaved)
    # A1 vs B1 task, A2 vs B2 task
    assert len(candidates) >= 2
    assert {candidates[0].code_a.snippet, candidates[0].code_b.snippet} == {"", ""}
    # IDs are auto-generated (C0, C1, C2, C3)

def test_explored_pairs_cache(discovery_operator: DifferentialDiscoveryOperator) -> None:
    # Test that already explored pairs are not selected again
    context = MagicMock(spec=CoevolutionContext)
    
    code_a = CodeIndividual(snippet="", probability=0.9, creation_op="init", generation_born=0)
    code_b = CodeIndividual(snippet="", probability=0.8, creation_op="init", generation_born=0)
    group = FunctionallyEquivGroup(
        code_individuals=[code_a, code_b],
        passing_test_individuals={}
    )
    cast(MagicMock, discovery_operator.func_eq_selector.select_functionally_equivalent_codes).return_value = [group]
    
    # Manually mark as explored
    discovery_operator._mark_pair_explored(code_a, code_b)
    
    candidates = discovery_operator._select_candidates(context)
    assert len(candidates) == 0

def test_reset_explored_pairs(discovery_operator: DifferentialDiscoveryOperator) -> None:
    discovery_operator._explored_pairs_cache.add(("A", "B"))
    discovery_operator.reset_explored_pairs()
    assert len(discovery_operator._explored_pairs_cache) == 0
