"""Tests for ReverseRouletteWheelParentSelection."""

from unittest.mock import Mock

import pytest

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import OPERATION_INITIAL, CoevolutionContext
from coevolution.core.population import CodePopulation
from coevolution.strategies.selection.parent_selection import (
    ReverseRouletteWheelParentSelection,
)


@pytest.fixture
def sample_code_population() -> CodePopulation:
    """Create a sample code population with varying probabilities."""
    individuals = [
        CodeIndividual(
            snippet="def solution1(): pass",
            probability=0.1,  # High priority in Reverse Roulette
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        ),
        CodeIndividual(
            snippet="def solution2(): pass",
            probability=0.3,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        ),
        CodeIndividual(
            snippet="def solution3(): pass",
            probability=0.9,  # Low priority in Reverse Roulette
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        ),
    ]
    return CodePopulation(individuals=individuals)


@pytest.fixture
def mock_coevolution_context() -> CoevolutionContext:
    """Create a mock coevolution context."""
    return Mock(spec=CoevolutionContext)


class TestReverseRouletteWheelParentSelection:
    """Test suite for ReverseRouletteWheelParentSelection."""

    def test_select_single_parent(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test selecting a single parent."""
        strategy: ReverseRouletteWheelParentSelection[CodeIndividual] = (
            ReverseRouletteWheelParentSelection()
        )
        parents = strategy.select_parents(
            sample_code_population,
            count=1,
            coevolution_context=mock_coevolution_context,
        )

        assert len(parents) == 1
        assert parents[0] in sample_code_population.individuals

    def test_probability_proportional_selection(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test that LOWER probability individuals are selected more often."""
        strategy: ReverseRouletteWheelParentSelection[CodeIndividual] = (
            ReverseRouletteWheelParentSelection()
        )

        # Run many selections to verify statistical properties
        num_trials = 1000
        selection_counts = {ind.id: 0 for ind in sample_code_population.individuals}

        for _ in range(num_trials):
            parents = strategy.select_parents(
                sample_code_population,
                count=1,
                coevolution_context=mock_coevolution_context,
            )
            selection_counts[parents[0].id] += 1

        # Inverse probs: 1-0.1=0.9, 1-0.3=0.7, 1-0.9=0.1
        # Individual with prob=0.1 should be selected ~9/17 of the time
        # Individual with prob=0.3 should be selected ~7/17 of the time
        # Individual with prob=0.9 should be selected ~1/17 of the time

        c0_count = selection_counts[sample_code_population.individuals[0].id]  # p=0.1 (high priority)
        c1_count = selection_counts[sample_code_population.individuals[1].id]  # p=0.3
        c2_count = selection_counts[sample_code_population.individuals[2].id]  # p=0.9 (low priority)

        # Verify rough proportions
        assert c0_count > c1_count > c2_count
        assert c0_count > c2_count * 5  # 0.9 vs 0.1 ratio is 9x

    def test_all_one_probabilities(
        self, mock_coevolution_context: CoevolutionContext
    ) -> None:
        """Test selection when all probabilities are one (degenerate case)."""
        individuals = [
            CodeIndividual(
                snippet=f"def solution{i}(): pass",
                probability=1.0,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i in range(3)
        ]
        population = CodePopulation(individuals=individuals)
        strategy: ReverseRouletteWheelParentSelection[CodeIndividual] = (
            ReverseRouletteWheelParentSelection()
        )

        # Should fall back to uniform random selection
        parents = strategy.select_parents(
            population, count=2, coevolution_context=mock_coevolution_context
        )

        assert len(parents) == 2
        assert all(p in population.individuals for p in parents)
