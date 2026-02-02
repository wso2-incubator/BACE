"""
Tests for TopKEliteSelector.

This module tests the TopKEliteSelector implementation,
verifying that it correctly selects top-k elite individuals
based on probability values.
"""

from unittest.mock import Mock

import pytest

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    CoevolutionContext,
    PopulationConfig,
)
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.strategies.selection.elite import TopKEliteSelector


@pytest.fixture
def sample_code_population() -> CodePopulation:
    """Create a sample code population with varying probabilities."""
    individuals = [
        CodeIndividual(
            snippet=f"def solution{i}(): pass",
            probability=0.1 * (i + 1),  # 0.1, 0.2, 0.3, 0.4, 0.5
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )
        for i in range(5)
    ]
    return CodePopulation(individuals=individuals)


@pytest.fixture
def sample_test_population() -> TestPopulation:
    """Create a sample test population with varying probabilities."""
    individuals = [
        TestIndividual(
            snippet=f"def test{i}(): assert True",
            probability=0.15 * (i + 1),  # 0.15, 0.30, 0.45, 0.60, 0.75
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )
        for i in range(5)
    ]
    return TestPopulation(
        individuals=individuals,
    )


@pytest.fixture
def population_config() -> PopulationConfig:
    """Create a sample population configuration."""
    return PopulationConfig(
        initial_prior=0.5,
        initial_population_size=10,
        max_population_size=20,
        elitism_rate=0.4,  # Keep 40% as elites
    )


@pytest.fixture
def mock_coevolution_context() -> CoevolutionContext:
    """Create a mock coevolution context."""
    return Mock(spec=CoevolutionContext)


class TestTopKEliteSelector:
    """Test suite for TopKEliteSelector."""

    def test_select_top_elites(
        self,
        sample_code_population: CodePopulation,
        population_config: PopulationConfig,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test selecting top-k elites by probability."""
        selector: TopKEliteSelector[CodeIndividual] = TopKEliteSelector()
        elites = selector.select_elites(
            sample_code_population, population_config, mock_coevolution_context
        )

        # With elitism_rate=0.4 and population size=5, expect 2 elites (5 * 0.4 = 2.0)
        assert len(elites) == 2

        # Should select the two highest probability individuals
        assert elites[0].probability == 0.5  # Highest
        assert elites[1].probability == 0.4  # Second highest

    def test_empty_population(
        self,
        population_config: PopulationConfig,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test that empty population returns empty elite list."""
        # Create empty population (now supported)
        empty_pop = CodePopulation(individuals=[])
        selector: TopKEliteSelector[CodeIndividual] = TopKEliteSelector()
        elites = selector.select_elites(
            empty_pop, population_config, mock_coevolution_context
        )

        assert len(elites) == 0

    def test_works_with_test_population(
        self,
        sample_test_population: TestPopulation,
        population_config: PopulationConfig,
        mock_coevolution_context: CoevolutionContext,
    ) -> None:
        """Test that TopKEliteSelector works with test populations (generic)."""
        selector: TopKEliteSelector[TestIndividual] = TopKEliteSelector()
        elites = selector.select_elites(
            sample_test_population, population_config, mock_coevolution_context
        )

        # With elitism_rate=0.4 and population size=5, expect 2 elites
        assert len(elites) == 2

        # Should select the two highest probability individuals
        assert elites[0].probability == 0.75  # Highest
        assert elites[1].probability == 0.60  # Second highest

    def test_single_individual_population(
        self,
        population_config: PopulationConfig,
        mock_coevolution_context: CoevolutionContext,
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
        selector: TopKEliteSelector[CodeIndividual] = TopKEliteSelector()

        elites = selector.select_elites(
            population, population_config, mock_coevolution_context
        )

        # With elitism_rate=0.4 and size=1, expect 0 elites (1 * 0.4 = 0.4 -> int(0.4) = 0)
        # Actually, int(1 * 0.4) = 0, so we get 0 elites
        assert len(elites) == 0

    def test_repr(self) -> None:
        """Test string representation."""
        selector: TopKEliteSelector[CodeIndividual] = TopKEliteSelector()
        assert repr(selector) == "TopKEliteSelector()"
