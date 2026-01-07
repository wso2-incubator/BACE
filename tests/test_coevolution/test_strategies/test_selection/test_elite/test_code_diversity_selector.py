from typing import List
from unittest.mock import Mock

import numpy as np
import pytest

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import CoevolutionContext, InteractionData
from coevolution.core.population import BasePopulation, PopulationConfig

# Adjust imports to match your project structure
from coevolution.strategies.selection.elite import CodeDiversityEliteSelector

# -----------------------------------------------------------------------------
# Fixtures & Helpers
# -----------------------------------------------------------------------------


@pytest.fixture
def selector():
    return CodeDiversityEliteSelector()


@pytest.fixture
def mock_config():
    """Default configuration for testing."""
    config = Mock(spec=PopulationConfig)
    config.max_population_size = 10
    config.offspring_rate = (
        0.0  # Default to no offspring (keep all elites) for simpler math
    )
    config.elitism_rate = 0.2  # Default top 20%
    return config


def create_mock_individual(ind_id: str, prob: float) -> Mock:
    """Helper to create a mock CodeIndividual."""
    ind = Mock(spec=CodeIndividual)
    ind.id = ind_id
    ind.probability = prob
    # Make string representation useful for debugging
    ind.__repr__ = lambda x: f"Code({ind_id}, p={prob})"
    return ind


def create_mock_population(individuals: List[Mock]) -> Mock:
    """Helper to create a mock BasePopulation."""
    pop = Mock(spec=BasePopulation)
    pop.size = len(individuals)
    pop.probabilities = np.array([ind.probability for ind in individuals])

    # Enable indexing: pop[i]
    pop.__getitem__ = lambda self, idx: individuals[idx]

    # Mock get_top_k_individuals logic
    def get_top_k(k):
        sorted_inds = sorted(individuals, key=lambda x: x.probability, reverse=True)
        return sorted_inds[:k]

    pop.get_top_k_individuals.side_effect = get_top_k
    return pop


def create_mock_context(matrices: List[np.ndarray]) -> Mock:
    """
    Helper to create a context with multiple interaction matrices.
    matrices: List of numpy arrays to be placed in interactions.
    """
    context = Mock(spec=CoevolutionContext)
    interactions = {}

    for i, matrix in enumerate(matrices):
        interaction = Mock(spec=InteractionData)
        interaction.observation_matrix = matrix
        interactions[f"interaction_{i}"] = interaction

    context.interactions = interactions
    return context


# -----------------------------------------------------------------------------
# 1. Unit Tests: Core Helper Logic
# -----------------------------------------------------------------------------


class TestCoreLogic:
    def test_concatenate_matrices_success(self, selector):
        """Test proper horizontal stacking of matrices."""
        # Matrix A: 3 codes x 2 tests
        mat_a = np.array([[1, 0], [1, 0], [0, 1]])
        # Matrix B: 3 codes x 3 tests
        mat_b = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])

        context = create_mock_context([mat_a, mat_b])

        # Expected: 3 codes x 5 tests
        result = selector._concatenate_all_matrices(context, expected_rows=3)

        assert result.shape == (3, 5)
        # Check first row merged correctly: [1, 0] + [0, 0, 1] -> [1, 0, 0, 0, 1]
        np.testing.assert_array_equal(result[0], np.array([1, 0, 0, 0, 1]))

    def test_concatenate_matrices_dimension_mismatch(self, selector):
        """Test validation error when matrices have different row counts."""
        mat_a = np.zeros((3, 2))
        mat_b = np.zeros((4, 2))  # Mismatch!

        context = create_mock_context([mat_a, mat_b])

        with pytest.raises(ValueError, match="expected 3"):
            selector._concatenate_all_matrices(context, expected_rows=3)

    def test_group_by_unique_rows_logic(self, selector):
        """
        Verify the core diversity grouping logic.

        Scenario:
        Index 0: [1, 0]
        Index 1: [0, 1]
        Index 2: [1, 0] (Same as 0)
        Index 3: [1, 1]
        Index 4: [0, 1] (Same as 1)
        """
        matrix = np.array(
            [
                [1, 0],  # Grp A
                [0, 1],  # Grp B
                [1, 0],  # Grp A
                [1, 1],  # Grp C
                [0, 1],  # Grp B
            ]
        )

        groups = selector._group_by_unique_rows(matrix)

        # Should find 3 unique groups
        assert len(groups) == 3

        # Convert list of arrays to list of lists for easier assert
        groups_list = sorted([sorted(g.tolist()) for g in groups])

        # Expected groupings: [0, 2], [1, 4], [3]
        expected = sorted([[0, 2], [1, 4], [3]])
        assert groups_list == expected


# -----------------------------------------------------------------------------
# 2. Integration Tests: Selection Strategy
# -----------------------------------------------------------------------------


class TestSelectionStrategy:
    def test_basic_diversity_selection(self, selector, mock_config):
        """
        Scenario: 4 individuals.
        Behaviors: 1 & 2 identical (Group A). 3 & 4 identical (Group B).
        Probs: 1=0.9, 2=0.1, 3=0.8, 4=0.2.
        Elitism: 0.0 (Only diversity matters).
        Expect: Indiv 1 (Best of A) and Indiv 3 (Best of B).
        """
        mock_config.elitism_rate = 0.0

        ind1 = create_mock_individual("1", 0.9)
        ind2 = create_mock_individual("2", 0.1)  # Same behavior as 1, lower prob
        ind3 = create_mock_individual("3", 0.8)
        ind4 = create_mock_individual("4", 0.2)  # Same behavior as 3, lower prob

        population = create_mock_population([ind1, ind2, ind3, ind4])

        # Matrix to enforce grouping: Rows 0,1 same; Rows 2,3 same
        matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        context = create_mock_context([matrix])

        elites = selector.select_elites(population, mock_config, context)

        assert len(elites) == 2
        ids = {e.id for e in elites}
        assert ids == {"1", "3"}

    def test_quality_guarantee(self, selector, mock_config):
        """
        Scenario: 5 individuals, ALL Identical behavior.
        Probs: [0.9, 0.8, 0.7, 0.6, 0.5]
        Elitism: 0.4 (Top 2 guaranteed).

        Diversity step selects: top 1 (0.9)
        Quality step selects: top 2 (0.9, 0.8)
        Union: {0.9, 0.8}
        """
        mock_config.elitism_rate = 0.4
        mock_config.max_population_size = 10
        mock_config.offspring_rate = 0.0  # Plenty of room

        inds = [
            create_mock_individual(str(i), p)
            for i, p in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])
        ]
        population = create_mock_population(inds)

        # All identical rows
        matrix = np.zeros((5, 2))
        context = create_mock_context([matrix])

        elites = selector.select_elites(population, mock_config, context)

        # Should return 2 elites (Best diversity + 2nd Best quality)
        assert len(elites) == 2
        ids = {e.id for e in elites}
        assert ids == {"0", "1"}  # 0.9 and 0.8

    def test_population_limit_truncation(self, selector, mock_config):
        """
        Scenario:
        Pop Size: 5. All Unique Behaviors.
        Probs: [0.1, 0.2, 0.3, 0.4, 0.5] (Reverse order)

        Constraints:
        Max Pop: 10
        Offspring Rate: 0.8 -> Offspring = 8.
        Available Slots = 10 - 8 = 2 slots.

        Diversity Step finds 5 unique elites.
        But we only have room for 2.
        Expect: Return ONLY the top 2 highest probability (0.5, 0.4).
        """
        mock_config.max_population_size = 10
        mock_config.offspring_rate = 0.8

        # Indivs 0-4 with increasing probs
        inds = [create_mock_individual(str(i), (i + 1) / 10) for i in range(5)]
        population = create_mock_population(inds)

        # Identity matrix ensures everyone is unique
        matrix = np.eye(5)
        context = create_mock_context([matrix])

        elites = selector.select_elites(population, mock_config, context)

        assert len(elites) == 2
        # Should keep 4 (0.5) and 3 (0.4)
        ids = {e.id for e in elites}
        assert ids == {"4", "3"}

        # Verify strict probability ordering
        probs = [e.probability for e in elites]
        assert max(probs) == 0.5
        assert min(probs) == 0.4


# -----------------------------------------------------------------------------
# 3. Edge Cases & Fallbacks
# -----------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_population(self, selector, mock_config):
        """Test handling of empty population."""
        population = create_mock_population([])
        context = create_mock_context([])

        elites = selector.select_elites(population, mock_config, context)
        assert elites == []

    def test_fallback_on_matrix_error(self, selector, mock_config):
        """Test fallback to top-k when matrices are missing/invalid."""
        mock_config.elitism_rate = 0.5  # Should pick top 2 of 4

        inds = [
            create_mock_individual(str(i), p)
            for i, p in enumerate([0.1, 0.9, 0.2, 0.8])
        ]
        population = create_mock_population(inds)

        # Context has NO matrices -> causes concatenation error
        context = create_mock_context([])

        # This should log warning and use fallback
        elites = selector.select_elites(population, mock_config, context)

        assert len(elites) == 2
        ids = {e.id for e in elites}
        assert ids == {"1", "3"}  # 0.9 and 0.8

    def test_zero_elitism_rate(self, selector, mock_config):
        """
        Test elitism_rate = 0.
        Should strictly rely on diversity (1 per unique group).
        """
        mock_config.elitism_rate = 0.0

        # 2 Groups.
        # Grp 1: Ind A (0.9), Ind B (0.1) -> Pick A
        # Grp 2: Ind C (0.2) -> Pick C
        indA = create_mock_individual("A", 0.9)
        indB = create_mock_individual("B", 0.1)
        indC = create_mock_individual("C", 0.2)

        population = create_mock_population([indA, indB, indC])

        matrix = np.array(
            [
                [1, 1],  # A
                [1, 1],  # B (Same as A)
                [0, 0],  # C (Unique)
            ]
        )
        context = create_mock_context([matrix])

        elites = selector.select_elites(population, mock_config, context)

        assert len(elites) == 2
        ids = {e.id for e in elites}
        assert ids == {"A", "C"}
