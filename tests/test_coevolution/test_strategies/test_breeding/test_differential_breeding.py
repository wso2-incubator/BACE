from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_INITIAL,
    CoevolutionContext,
    IParentSelectionStrategy,
    OperatorOutput,
    OperatorRatesConfig,
    OperatorResult,
    PopulationConfig,
    Problem,
)
from coevolution.core.population import CodePopulation, TestPopulation

# Adjust imports to match your project structure
from coevolution.strategies.breeding.differential_breeding import (
    DifferentialBreedingStrategy,
    DifferentialResult,
    FunctionallyEquivGroup,
    IDifferentialFinder,
    IFunctionallyEquivalentCodeSelector,
)
from coevolution.strategies.operators.differential_llm_operator import (
    OPERATION_DISCOVERY,
    DifferentialLLMOperator,
)

# --- Fixtures ---


@pytest.fixture
def mock_operator() -> MagicMock:
    """Returns a mock that satisfies the DifferentialLLMOperator interface."""
    op = MagicMock(spec=DifferentialLLMOperator)
    # Ensure checking supported operations passes
    op.supported_operations.return_value = {
        OPERATION_INITIAL,
        OPERATION_CROSSOVER,
        OPERATION_DISCOVERY,
    }
    return op


@pytest.fixture
def mock_differential_finder() -> MagicMock:
    """Returns a mock that satisfies the IDifferentialFinder protocol."""
    return MagicMock(spec=IDifferentialFinder)


@pytest.fixture
def mock_func_eq_selector() -> MagicMock:
    """Returns a mock that satisfies the IFunctionallyEquivalentCodeSelector protocol."""
    return MagicMock(spec=IFunctionallyEquivalentCodeSelector)


@pytest.fixture
def mock_context() -> MagicMock:
    """Returns a mock CoevolutionContext with connected mock populations."""
    ctx = MagicMock(spec=CoevolutionContext)

    # Setup dummy code population
    ctx.code_population = MagicMock(spec=CodePopulation)
    ctx.code_population.generation = 5

    # Setup test populations dict
    test_pop_mock = MagicMock(spec=TestPopulation)
    test_pop_mock.size = 0  # Start empty
    test_pop_mock.generation = 5

    # Mock the dictionary behavior for looking up populations
    ctx.test_populations = {"differential": test_pop_mock}

    # Setup Problem
    ctx.problem = MagicMock(spec=Problem)
    ctx.problem.question_content = "Sort list"
    ctx.problem.starter_code = "def sort(x): pass"

    return ctx


@pytest.fixture
def strategy(
    mock_operator: MagicMock,
    mock_differential_finder: MagicMock,
    mock_func_eq_selector: MagicMock,
) -> DifferentialBreedingStrategy:
    """
    Returns the concrete Strategy instance injected with mocks.
    """
    return DifferentialBreedingStrategy(
        operator=cast(DifferentialLLMOperator, mock_operator),
        differential_finder=cast(IDifferentialFinder, mock_differential_finder),
        op_rates_config=OperatorRatesConfig(
            operation_rates={OPERATION_DISCOVERY: 0.8, OPERATION_CROSSOVER: 0.2}
        ),
        pop_config=PopulationConfig(
            initial_population_size=10, initial_prior=0.5, max_population_size=20
        ),
        probability_assigner=MagicMock(),
        parent_selector=MagicMock(spec=IParentSelectionStrategy),
        functionally_equivalent_code_selector=cast(
            IFunctionallyEquivalentCodeSelector, mock_func_eq_selector
        ),
        llm_workers=1,
    )


# --- Helper functions ---


def make_code_ind(id_val: str) -> CodeIndividual:
    """
    Helper to create a properly typed mock CodeIndividual.
    Crucial: Must have .id and .snippet attributes.
    """
    ind = MagicMock(spec=CodeIndividual)
    ind.id = id_val
    ind.snippet = f"code_snippet_{id_val}"
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
    mock_differential_finder: MagicMock,
) -> None:
    """
    Verify the full discovery pipeline.
    Ensures that CodeIndividual objects (not IDs) are passed through the pipeline.
    """
    # 1. Setup Data: Create Mock Code Individuals
    code_a = make_code_ind("A")
    code_b = make_code_ind("B")

    # 2. Setup Selector to return a Group containing these Objects
    group = FunctionallyEquivGroup(
        code_individuals=[code_a, code_b], passing_test_individuals={}
    )
    mock_func_eq_selector.select_functionally_equivalent_codes.return_value = [group]

    # 3. Mock Operator (LLM Script Gen)
    mock_operator.apply.return_value = OperatorOutput(
        results=[OperatorResult(snippet="def fuzz(): ...")]
    )
    mock_operator.get_test_method_from_io.return_value = "def test_diff(): ..."

    # 4. Mock Divergence Finder
    # differential_finder receives (code_a.snippet, code_b.snippet, script)
    divergence_result = DifferentialResult(
        input_data={"x": 1},
        output_a=10,
        output_b=20,
    )
    mock_differential_finder.find_differential.return_value = [divergence_result]

    # 5. Run Breed
    # We ask for 2 offspring, which aligns with 1 divergence -> 2 scenarios (A wins, B wins)
    offspring = strategy.breed(cast(CoevolutionContext, mock_context), num_offsprings=2)

    # 6. Assertions
    assert len(offspring) == 2

    # Check that differential finder was called with snippets, not IDs or Objects
    mock_differential_finder.find_differential.assert_called_with(
        code_a.snippet, code_b.snippet, "def fuzz(): ...", limit=5
    )

    # Verify Metadata logic for the first offspring (Scenario 1: A is winner)
    meta_a = offspring[0].metadata
    assert meta_a is not None
    assert "divergence_outputs" in meta_a

    # Since 'code_a' mock has id="A", and 'output_a' was 10:
    assert meta_a["divergence_outputs"] == {"A": [10], "B": [20]}


def test_breed_via_discovery_retries_on_failure(
    strategy: DifferentialBreedingStrategy,
    mock_context: MagicMock,
    mock_func_eq_selector: MagicMock,
    mock_operator: MagicMock,
    mock_differential_finder: MagicMock,
) -> None:
    """
    Verify the loop retries if the divergence finder returns nothing initially.
    """
    # 1. Setup Objects
    code_a = make_code_ind("A")
    code_b = make_code_ind("B")

    group = FunctionallyEquivGroup(
        code_individuals=[code_a, code_b], passing_test_individuals={}
    )
    mock_func_eq_selector.select_functionally_equivalent_codes.return_value = [group]

    # 2. Setup Operator
    mock_operator.apply.return_value
