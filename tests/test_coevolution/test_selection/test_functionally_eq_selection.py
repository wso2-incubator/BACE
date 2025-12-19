from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from common.coevolution.breeding_strategies.differential_breeding import (
    DifferentialBreedingStrategy,
    DivergenceResult,
    FunctionallyEquivGroup,
    IDivergenceFinder,
    IFunctionallyEquivalentCodeSelector,
)
from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_INITIAL,
    CoevolutionContext,
    IParentSelectionStrategy,
    IProbabilityAssigner,
    OperatorOutput,
    OperatorRatesConfig,
    OperatorResult,
    PopulationConfig,
    Problem,
)
from common.coevolution.core.population import CodePopulation, TestPopulation
from common.coevolution.operators.differential_llm_operator import (
    OPERATION_DISCOVERY,
    DifferentialLLMOperator,
)

# --- Fixtures ---


@pytest.fixture
def mock_operator() -> MagicMock:
    """Returns a mock that satisfies the DifferentialLLMOperator interface."""
    op = MagicMock(spec=DifferentialLLMOperator)
    op.supported_operations.return_value = {
        OPERATION_INITIAL,
        OPERATION_CROSSOVER,
        OPERATION_DISCOVERY,
    }
    return op


@pytest.fixture
def mock_divergence_finder() -> MagicMock:
    """Returns a mock that satisfies the IDivergenceFinder protocol."""
    return MagicMock(spec=IDivergenceFinder)


@pytest.fixture
def mock_func_eq_selector() -> MagicMock:
    """Returns a mock that satisfies the IFunctionallyEquivalentCodeSelector protocol."""
    return MagicMock(spec=IFunctionallyEquivalentCodeSelector)


@pytest.fixture
def mock_context() -> MagicMock:
    """Returns a mock CoevolutionContext with connected mock populations."""
    ctx = MagicMock(spec=CoevolutionContext)

    # Setup dummy populations
    ctx.code_population = MagicMock(spec=CodePopulation)
    ctx.code_population.generation = 5

    # Setup test populations dict
    test_pop_mock = MagicMock(spec=TestPopulation)
    test_pop_mock.size = 0  # Start empty
    test_pop_mock.generation = 5

    # We must mock the dictionary behavior
    ctx.test_populations = {"differential": test_pop_mock}

    ctx.problem = MagicMock(spec=Problem)
    ctx.problem.question_content = "Sort list"
    ctx.problem.starter_code = "def sort(x): pass"

    return ctx


@pytest.fixture
def strategy(
    mock_operator: MagicMock,
    mock_divergence_finder: MagicMock,
    mock_func_eq_selector: MagicMock,
) -> DifferentialBreedingStrategy:
    """
    Returns the concrete Strategy instance injected with mocks.
    """
    # We cast mocks to their concrete types to satisfy the strategy constructor
    return DifferentialBreedingStrategy(
        operator=cast(DifferentialLLMOperator, mock_operator),
        divergence_finder=cast(IDivergenceFinder, mock_divergence_finder),
        op_rates_config=OperatorRatesConfig(
            operation_rates={OPERATION_DISCOVERY: 0.8, OPERATION_CROSSOVER: 0.2}
        ),
        pop_config=PopulationConfig(
            initial_population_size=10, initial_prior=0.5, max_population_size=20
        ),
        probability_assigner=MagicMock(spec=IProbabilityAssigner),
        parent_selector=MagicMock(spec=IParentSelectionStrategy),
        functionally_equivalent_code_selector=cast(
            IFunctionallyEquivalentCodeSelector, mock_func_eq_selector
        ),
        max_workers=1,
    )


# --- Helper functions ---


def make_code_ind(id_val: str) -> CodeIndividual:
    """Helper to create a properly typed mock CodeIndividual."""
    ind = MagicMock(spec=CodeIndividual)
    ind.id = id_val
    ind.snippet = f"code_{id_val}"
    # Mypy treats MagicMocks as Any, but we cast for clarity if used strictly
    return cast(CodeIndividual, ind)


def make_test_ind(
    id_val: str, io_pairs: list[dict[str, Any]] | None = None
) -> TestIndividual:
    """Helper to create a properly typed mock TestIndividual."""
    ind = MagicMock(spec=TestIndividual)
    ind.id = id_val
    ind.metadata = {"io_pairs": io_pairs} if io_pairs else {}
    ind.probability = 0.5
    return cast(TestIndividual, ind)


# --- Tests ---


def test_initialization_scaffold(
    strategy: DifferentialBreedingStrategy, mock_operator: MagicMock
) -> None:
    """Test that initialize_individuals returns no individuals but sets up the class block."""
    mock_operator.generate_initial_snippets.return_value = (
        MagicMock(spec=OperatorOutput),
        "class TestScaffold: ...",
    )

    problem_mock = MagicMock(spec=Problem)
    problem_mock.question_content = "Q"
    problem_mock.starter_code = "S"

    inds, block = strategy.initialize_individuals(problem_mock)

    assert inds == []
    assert block == "class TestScaffold: ..."
    mock_operator.generate_initial_snippets.assert_called_once()


def test_breed_via_discovery_success(
    strategy: DifferentialBreedingStrategy,
    mock_context: MagicMock,
    mock_func_eq_selector: MagicMock,
    mock_operator: MagicMock,
    mock_divergence_finder: MagicMock,
) -> None:
    """
    Verify the full discovery pipeline:
    Group Selection -> LLM Script -> Divergence Found -> 2 New Individuals Created.
    """
    # 1. Setup Data
    code_a = make_code_ind("A")
    code_b = make_code_ind("B")

    # Mock Selector returning one group with 2 codes
    group = FunctionallyEquivGroup(
        code_individuals=[code_a, code_b], passing_test_individuals={}
    )
    mock_func_eq_selector.select_functionally_equivalent_codes.return_value = [group]

    # Mock LLM Script Generation
    mock_operator.apply.return_value = OperatorOutput(
        results=[OperatorResult(snippet="def fuzz(): ...")]
    )

    # Mock Divergence Finder returning 1 divergence
    divergence_result: DivergenceResult = {
        "input_data": {"x": 1},
        "output_a": 10,
        "output_b": 20,
    }
    # find_divergence returns a LIST of results
    mock_divergence_finder.find_divergence.return_value = [divergence_result]

    # Mock Method Builder
    mock_operator.get_test_method_from_io.return_value = "def test_diff(): ..."

    # 2. Run Breed
    # We pass the mock_context. Note: We must ensure it's typed correctly in the signature
    # or cast it, though Python handles the MagicMock fine at runtime.
    offspring = strategy.breed(cast(CoevolutionContext, mock_context), num_offsprings=2)

    # 3. Assertions
    assert len(offspring) == 2
    assert offspring[0].creation_op == OPERATION_DISCOVERY
    assert offspring[1].creation_op == OPERATION_DISCOVERY

    # Verify Metadata logic
    meta_a = offspring[0].metadata
    # The first individual assumes A (10) is correct
    assert meta_a is not None
    assert meta_a["io_pairs"][0]["output"] == 10
    assert meta_a["divergence_outputs"] == {"A": [20], "B": [10]}

    meta_b = offspring[1].metadata
    assert meta_b is not None
    assert meta_b["io_pairs"][0]["output"] == 20


def test_breed_via_discovery_retries_on_failure(
    strategy: DifferentialBreedingStrategy,
    mock_context: MagicMock,
    mock_func_eq_selector: MagicMock,
    mock_operator: MagicMock,
    mock_divergence_finder: MagicMock,
) -> None:
    """
    Verify the loop retries if:
    1. LLM fails
    2. Divergence Finder returns nothing
    """
    code_a = make_code_ind("A")
    code_b = make_code_ind("B")
    group = FunctionallyEquivGroup(
        code_individuals=[code_a, code_b], passing_test_individuals={}
    )
    mock_func_eq_selector.select_functionally_equivalent_codes.return_value = [group]

    # Setup Operator: Always succeeds
    mock_operator.apply.return_value = OperatorOutput(
        results=[OperatorResult(snippet="script")]
    )
    mock_operator.get_test_method_from_io.return_value = "test_code"

    # Setup Finder:
    # Attempt 1: Returns empty list [] (No divergence) -> Strategy should retry
    # Attempt 2: Returns valid list -> Strategy succeeds
    divergence_result: DivergenceResult = {
        "input_data": {"x": 2},
        "output_a": 1,
        "output_b": 2,
    }
    mock_divergence_finder.find_divergence.side_effect = [
        [],  # Fail
        [divergence_result],  # Success
    ]

    offspring = strategy.breed(cast(CoevolutionContext, mock_context), num_offsprings=2)

    assert len(offspring) == 2
    # Verify finder was called twice because the first time yielded no results
    assert mock_divergence_finder.find_divergence.call_count == 2


def test_breed_via_crossover(
    strategy: DifferentialBreedingStrategy, mock_context: MagicMock
) -> None:
    """Verify crossover correctly merges metadata from parents."""
    # Setup context with existing differential tests
    # Access the specific test population mock stored in the dict
    diff_pop = mock_context.test_populations["differential"]
    diff_pop.size = 10

    # Setup Parents
    p1 = make_test_ind("P1", io_pairs=[{"inputdata": {"x": 1}, "output": 1}])
    p2 = make_test_ind("P2", io_pairs=[{"inputdata": {"y": 2}, "output": 2}])

    # Mock the strategy's parent_selector property
    select_parents_mock: MagicMock = cast(
        MagicMock, strategy.parent_selector.select_parents
    )
    select_parents_mock.return_value = [p1, p2]

    # Mock Operator Crossover Result
    merged_metadata = {"io_pairs": [{"x": 1}, {"y": 2}]}

    # We must cast the operator mock back to MagicMock to configure return values
    # if Mypy gets confused, but usually it's fine.
    op_mock = cast(MagicMock, strategy.operator)
    op_mock.apply.return_value = OperatorOutput(
        results=[OperatorResult(snippet="merged_test", metadata=merged_metadata)]
    )

    # Configure probability assigner to return a float
    prob_assigner_mock = cast(MagicMock, strategy.probability_assigner)
    prob_assigner_mock.assign_probability.return_value = 0.7

    # Call the private handler directly for unit testing
    offspring = strategy._breed_via_crossover(cast(CoevolutionContext, mock_context))

    assert len(offspring) == 1
    assert offspring[0].creation_op == OPERATION_CROSSOVER
    assert offspring[0].metadata == merged_metadata


def test_breed_flow_switching(
    strategy: DifferentialBreedingStrategy, mock_context: MagicMock
) -> None:
    """
    Test the main breed() method switches phases:
    1. Discovery Phase fails (returns empty).
    2. Expansion Phase (Crossover) takes over to fill quota.
    """
    # 1. Setup Discovery to FAIL (No groups)
    func_selector = cast(MagicMock, strategy.func_eq_code_selector)
    func_selector.select_functionally_equivalent_codes.return_value = []

    # 2. Setup Crossover to SUCCEED
    # Ensure population size allows crossover (>=2)
    mock_context.test_populations["differential"].size = 5

    p1 = make_test_ind("P1", io_pairs=[{"inputdata": {"x": 1}, "output": 1}])
    p2 = make_test_ind("P2", io_pairs=[{"inputdata": {"y": 2}, "output": 2}])

    parent_selector = cast(MagicMock, strategy.parent_selector)
    parent_selector.select_parents.return_value = [p1, p2]

    op_mock = cast(MagicMock, strategy.operator)
    op_mock.apply.return_value = OperatorOutput(
        results=[
            OperatorResult(
                snippet="cross",
                metadata={"io_pairs": [{"inputdata": {"z": 3}, "output": 3}]},
            )
        ]
    )

    # Configure probability assigner to return a float
    prob_assigner_mock = cast(MagicMock, strategy.probability_assigner)
    prob_assigner_mock.assign_probability.return_value = 0.7

    # Run breed requesting 1 offspring
    # Discovery will return [], remaining_needed = 1.
    # Crossover loop should run.
    results = strategy.breed(cast(CoevolutionContext, mock_context), num_offsprings=1)

    assert len(results) == 1
    assert results[0].creation_op == OPERATION_CROSSOVER
    assert len(results) == 1
    assert results[0].creation_op == OPERATION_CROSSOVER
