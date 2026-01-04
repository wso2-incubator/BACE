"""
Tests for parent selection strategies.

This module tests the IParentSelectionStrategy implementations,
verifying that they correctly select parents from populations
based on their probability values.
"""

from unittest.mock import Mock

import pytest

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import OPERATION_INITIAL, CoevolutionContext
from coevolution.core.population import CodePopulation
from coevolution.selection_strategies.parent_selection import (
    RouletteWheelParentSelection,
)


@pytest.fixture
def sample_code_population() -> CodePopulation:
    """Create a sample code population with varying probabilities."""
    individuals = [
        CodeIndividual(
            snippet="def solution1(): pass",
            probability=0.1,
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
            probability=0.6,
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


class TestRouletteWheelParentSelection:
    """Test suite for RouletteWheelParentSelection."""

    def test_select_single_parent(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test selecting a single parent."""
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )
        parents = strategy.select_parents(
            sample_code_population,
            count=1,
            coevolution_context=mock_coevolution_context,
        )

        assert len(parents) == 1
        assert parents[0] in sample_code_population.individuals

    def test_select_two_parents(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test selecting two parents for crossover."""
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )
        parents = strategy.select_parents(
            sample_code_population,
            count=2,
            coevolution_context=mock_coevolution_context,
        )

        assert len(parents) == 2
        assert all(p in sample_code_population.individuals for p in parents)

    def test_select_multiple_parents(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test selecting multiple parents (more than 2)."""
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )
        count = 5
        with pytest.raises(
            ValueError, match="Population size must be at least equal to count"
        ):
            strategy.select_parents(
                sample_code_population,
                count=count,
                coevolution_context=mock_coevolution_context,
            )

    def test_allows_duplicate_selection(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test that same individual can be selected multiple times (sampling with replacement)."""
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )
        # Selecting more parents than population size should now raise an error
        with pytest.raises(
            ValueError, match="Population size must be at least equal to count"
        ):
            strategy.select_parents(
                sample_code_population,
                count=100,
                coevolution_context=mock_coevolution_context,
            )

    def test_probability_proportional_selection(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test that higher probability individuals are selected more often."""
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
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

        # Individual with prob=0.6 should be selected ~60% of the time
        # Individual with prob=0.3 should be selected ~30% of the time
        # Individual with prob=0.1 should be selected ~10% of the time

        # Get counts for each individual (in order of increasing probability)
        c0_count = selection_counts[sample_code_population.individuals[0].id]  # p=0.1
        c1_count = selection_counts[sample_code_population.individuals[1].id]  # p=0.3
        c2_count = selection_counts[sample_code_population.individuals[2].id]  # p=0.6

        # Verify rough proportions (with some tolerance for randomness)
        # Higher probability individual should be selected most
        assert c2_count > c1_count > c0_count

        # Check approximate proportions (within reasonable bounds)
        # Individual with 0.6 probability should have ~6x the selections of 0.1
        assert c2_count / c0_count > 3  # At least 3x more (conservative)

    def test_single_individual_population(
        self, mock_coevolution_context: CoevolutionContext
    ) -> None:
        """Test selection from population with only one individual."""
        individuals = [
            CodeIndividual(
                snippet="def only_solution(): pass",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
        ]
        population = CodePopulation(individuals=individuals)
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )
        # Selecting more parents than population size should raise an error
        with pytest.raises(
            ValueError, match="Population size must be at least equal to count"
        ):
            strategy.select_parents(
                population, count=3, coevolution_context=mock_coevolution_context
            )

    def test_all_zero_probabilities(
        self, mock_coevolution_context: CoevolutionContext
    ) -> None:
        """Test selection when all probabilities are zero (degenerate case)."""
        individuals = [
            CodeIndividual(
                snippet=f"def solution{i}(): pass",
                probability=0.0,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i in range(3)
        ]
        population = CodePopulation(individuals=individuals)
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )

        # Should fall back to uniform random selection
        parents = strategy.select_parents(
            population, count=2, coevolution_context=mock_coevolution_context
        )

        assert len(parents) == 2
        assert all(p in population.individuals for p in parents)

    def test_zero_count_raises_error(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test that count=0 raises ValueError."""
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )

        with pytest.raises(ValueError, match="count must be at least 1"):
            strategy.select_parents(
                sample_code_population,
                count=0,
                coevolution_context=mock_coevolution_context,
            )

    def test_negative_count_raises_error(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test that negative count raises ValueError."""
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )

        with pytest.raises(ValueError, match="count must be at least 1"):
            strategy.select_parents(
                sample_code_population,
                count=-1,
                coevolution_context=mock_coevolution_context,
            )

    def test_repr(self) -> None:
        """Test string representation."""
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )
        assert repr(strategy) == "RouletteWheelParentSelection()"

    def test_returns_individual_objects(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test that selected parents are Individual objects, not indices."""
        strategy: RouletteWheelParentSelection[CodeIndividual] = (
            RouletteWheelParentSelection()
        )
        parents = strategy.select_parents(
            sample_code_population,
            count=2,
            coevolution_context=mock_coevolution_context,
        )

        assert all(isinstance(p, CodeIndividual) for p in parents)
        assert all(hasattr(p, "id") for p in parents)
        assert all(hasattr(p, "probability") for p in parents)
        assert all(hasattr(p, "snippet") for p in parents)
        assert all(hasattr(p, "snippet") for p in parents)
