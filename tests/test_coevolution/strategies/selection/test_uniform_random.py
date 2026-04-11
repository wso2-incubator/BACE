"""Tests for UniformRandomParentSelection."""

from unittest.mock import Mock

import numpy as np
import pytest

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import OPERATION_INITIAL, CoevolutionContext
from coevolution.core.population import CodePopulation
from coevolution.strategies.selection.parent_selection import (
    UniformRandomParentSelection,
)


@pytest.fixture
def sample_code_population() -> CodePopulation:
    """Create a sample code population with varying probabilities."""
    individuals = [
        CodeIndividual(
            snippet=f"def solution{i}(): pass",
            probability=p,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )
        for i, p in enumerate([0.1, 0.5, 0.9])
    ]
    return CodePopulation(individuals=individuals)


@pytest.fixture
def mock_coevolution_context() -> CoevolutionContext:
    """Create a mock coevolution context."""
    return Mock(spec=CoevolutionContext)


class TestUniformRandomParentSelection:
    """Test suite for UniformRandomParentSelection."""

    def test_select_single_parent(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test selecting a single parent."""
        strategy: UniformRandomParentSelection[CodeIndividual] = (
            UniformRandomParentSelection()
        )
        parents = strategy.select_parents(
            sample_code_population,
            count=1,
            coevolution_context=mock_coevolution_context,
        )

        assert len(parents) == 1
        assert parents[0] in sample_code_population.individuals

    def test_uniform_random_selection(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test that selection is roughly uniform regardless of probability."""
        # Seed NumPy's RNG for reproducibility
        np.random.seed(42)

        strategy: UniformRandomParentSelection[CodeIndividual] = (
            UniformRandomParentSelection()
        )

        # Run many selections to verify statistical properties
        num_trials = 3000
        selection_counts = {ind.id: 0 for ind in sample_code_population.individuals}

        for _ in range(num_trials):
            parents = strategy.select_parents(
                sample_code_population,
                count=1,
                coevolution_context=mock_coevolution_context,
            )
            selection_counts[parents[0].id] += 1

        # Each individual should be selected ~1000 times
        expected_count = num_trials // len(sample_code_population.individuals)
        tolerance = 100  # Allow some variance for random sampling

        for count in selection_counts.values():
            assert abs(count - expected_count) < tolerance

    def test_empty_population_raises_error(
        self, mock_coevolution_context: CoevolutionContext
    ) -> None:
        """Test selecting from an empty population raises ValueError."""
        population = CodePopulation(individuals=[])
        strategy: UniformRandomParentSelection[CodeIndividual] = (
            UniformRandomParentSelection()
        )
        with pytest.raises(
            ValueError, match="Cannot select parents from empty population"
        ):
            strategy.select_parents(
                population, count=1, coevolution_context=mock_coevolution_context
            )

    def test_count_greater_than_population_size(
        self,
        sample_code_population: CodePopulation,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test selecting more parents than available in population."""
        strategy: UniformRandomParentSelection[CodeIndividual] = (
            UniformRandomParentSelection()
        )
        # Should raise ValueError as per the implementation rule population.size < count
        with pytest.raises(
            ValueError, match="Population size must be at least equal to count"
        ):
            strategy.select_parents(
                sample_code_population,
                count=5,
                coevolution_context=mock_coevolution_context,
            )
            strategy.select_parents(
                sample_code_population,
                count=5,
                coevolution_context=mock_coevolution_context,
            )
