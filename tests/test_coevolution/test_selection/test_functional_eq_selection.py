from unittest.mock import MagicMock

import numpy as np
import pytest

from common.coevolution.breeding_strategies.differential_breeding import (
    FunctionallyEquivGroup,
)
from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import CoevolutionContext, InteractionData
from common.coevolution.core.population import CodePopulation, TestPopulation
from common.coevolution.selection_strategies.functionally_eq_selection import (
    FunctionallyEqSelector,
)

# --- Fixtures ---


@pytest.fixture
def selector() -> FunctionallyEqSelector:
    return FunctionallyEqSelector()


@pytest.fixture
def mock_context() -> MagicMock:
    """Creates a skeleton context we can populate in tests."""
    ctx = MagicMock(spec=CoevolutionContext)

    # Populations
    ctx.code_population = MagicMock(spec=CodePopulation)
    ctx.test_populations = {}

    # Interactions dictionary
    ctx.interactions = {}

    return ctx


# --- Helpers ---


def setup_code_pop(ctx: MagicMock, size: int) -> list[CodeIndividual]:
    """Sets up the code population with dummy individuals."""
    individuals: list[CodeIndividual] = []
    for i in range(size):
        ind = MagicMock(spec=CodeIndividual)
        ind.id = f"C{i}"
        individuals.append(ind)

    ctx.code_population.individuals = individuals
    ctx.code_population.size = size
    # Allow indexing: ctx.code_population[i]
    ctx.code_population.__getitem__.side_effect = lambda i: individuals[i]
    return individuals


def setup_interaction(
    ctx: MagicMock, test_type: str, matrix: list[list[int]]
) -> list[TestIndividual]:
    """
    Sets up a test population and its interaction matrix.
    Args:
        matrix: 2D list (rows=codes, cols=tests)
    """
    matrix_np = np.array(matrix)
    num_tests = matrix_np.shape[1]

    # 1. Create Test Population
    test_individuals: list[TestIndividual] = []
    for i in range(num_tests):
        ind = MagicMock(spec=TestIndividual)
        ind.id = f"{test_type}_T{i}"
        test_individuals.append(ind)

    test_pop = MagicMock(spec=TestPopulation)
    test_pop.individuals = test_individuals
    test_pop.size = num_tests
    test_pop.__getitem__.side_effect = lambda i: test_individuals[i]

    ctx.test_populations[test_type] = test_pop

    # 2. Create InteractionData
    interaction = MagicMock(spec=InteractionData)
    interaction.observation_matrix = matrix_np
    ctx.interactions[test_type] = interaction

    return test_individuals


# --- Tests ---


def test_select_empty_population(
    selector: FunctionallyEqSelector, mock_context: MagicMock
) -> None:
    """If code population is empty, return empty list."""
    setup_code_pop(mock_context, size=0)
    groups = selector.select_functionally_equivalent_codes(mock_context)
    assert groups == []


def test_select_no_tests_defined(
    selector: FunctionallyEqSelector, mock_context: MagicMock
) -> None:
    """
    If there are code individuals but NO tests (Generation 0),
    everyone is theoretically equivalent (indistinguishable).
    Should return 1 giant group.
    """
    codes = setup_code_pop(mock_context, size=3)
    # No interactions setup

    groups = selector.select_functionally_equivalent_codes(mock_context)

    assert len(groups) == 1
    assert len(groups[0].code_individuals) == 3
    assert set(groups[0].code_individuals) == set(codes)
    assert groups[0].passing_test_individuals == {}


def test_select_distinct_groups_single_test_type(
    selector: FunctionallyEqSelector, mock_context: MagicMock
) -> None:
    """
    Scenario:
    - Code 0, 1 pass T0. (Identical behavior [1])
    - Code 2 fails T0. (Distinct behavior [0])
    Expected: 2 Groups.
    """
    codes = setup_code_pop(mock_context, size=3)

    # Matrix: Rows=Codes, Cols=Tests
    # C0: [1]
    # C1: [1]
    # C2: [0]
    matrix = [[1], [1], [0]]
    tests = setup_interaction(mock_context, "public", matrix)

    groups = selector.select_functionally_equivalent_codes(mock_context)

    assert len(groups) == 2

    # Sort groups by size to make assertion deterministic (Group 0 has 2, Group 1 has 1)
    groups.sort(key=lambda g: len(g.code_individuals), reverse=True)

    # Group 1: C0, C1
    g1 = groups[0]
    assert len(g1.code_individuals) == 2
    assert codes[0] in g1.code_individuals
    assert codes[1] in g1.code_individuals
    # Verify passing tests (They passed T0)
    assert g1.passing_test_individuals["public"][0] == tests[0]

    # Group 2: C2
    g2 = groups[1]
    assert len(g2.code_individuals) == 1
    assert g2.code_individuals[0] == codes[2]
    # Verify passing tests (Passed nothing)
    assert "public" not in g2.passing_test_individuals


def test_select_complex_groups_multiple_test_types(
    selector: FunctionallyEqSelector, mock_context: MagicMock
) -> None:
    """
    Scenario: Multi-dimensional behavior vectors.
    Codes must match on ALL test types to be grouped.

    Code | Public (T0, T1) | Private (T2) | Signature
    --------------------------------------------------
    C0   | 1, 0            | 1            | [1, 0, 1]
    C1   | 1, 0            | 1            | [1, 0, 1] -> Group A
    C2   | 1, 0            | 0            | [1, 0, 0] -> Group B
    C3   | 0, 1            | 1            | [0, 1, 1] -> Group C
    """

    # Public Tests Matrix (2 tests)
    mat_public = [
        [1, 0],  # C0
        [1, 0],  # C1
        [1, 0],  # C2
        [0, 1],  # C3
    ]
    tests_pub = setup_interaction(mock_context, "public", mat_public)

    # Private Tests Matrix (1 test)
    mat_private = [
        [1],  # C0
        [1],  # C1
        [0],  # C2
        [1],  # C3
    ]
    tests_priv = setup_interaction(mock_context, "private", mat_private)

    groups = selector.select_functionally_equivalent_codes(mock_context)

    assert len(groups) == 3

    # Helper to find group by representative code ID
    def get_group_by_code(cid: str) -> FunctionallyEquivGroup | None:
        for g in groups:
            if any(c.id == cid for c in g.code_individuals):
                return g
        return None

    # Verify Group A (C0, C1)
    g_a = get_group_by_code("C0")
    assert g_a is not None
    assert len(g_a.code_individuals) == 2
    assert any(c.id == "C1" for c in g_a.code_individuals)
    # Check passing tests logic: Should have Pub_T0 and Priv_T0
    assert tests_pub[0] in g_a.passing_test_individuals["public"]
    assert tests_priv[0] in g_a.passing_test_individuals["private"]

    # Verify Group B (C2) - Differs on Private
    g_b = get_group_by_code("C2")
    assert g_b is not None
    assert len(g_b.code_individuals) == 1
    assert "private" not in g_b.passing_test_individuals  # Passed 0 private tests

    # Verify Group C (C3) - Differs on Public
    g_c = get_group_by_code("C3")
    assert g_c is not None
    assert len(g_c.code_individuals) == 1
    assert tests_pub[1] in g_c.passing_test_individuals["public"]  # Passed Pub_T1


def test_select_mismatch_matrix_shape(
    selector: FunctionallyEqSelector, mock_context: MagicMock
) -> None:
    """
    If interaction matrix rows != population size, should log error and return empty.
    """
    setup_code_pop(mock_context, size=5)

    # Matrix has only 3 rows
    setup_interaction(mock_context, "public", [[1], [1], [1]])

    groups = selector.select_functionally_equivalent_codes(mock_context)

    assert groups == []
