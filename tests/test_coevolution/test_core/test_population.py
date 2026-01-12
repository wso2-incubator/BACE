"""
Comprehensive tests for the population.py module.

This module contains tests for both CodePopulation and TestPopulation classes,
covering initialization, properties, methods, edge cases, and integration scenarios.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    OPERATION_REPRODUCTION,
)
from coevolution.core.population import CodePopulation, TestPopulation

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_code_individuals() -> list[CodeIndividual]:
    """Create a list of sample CodeIndividuals for testing."""
    return [
        CodeIndividual(
            snippet=f"def func_{i}(): return {i}",
            probability=0.1 * (i + 1),
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={},
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_test_individuals() -> list[TestIndividual]:
    """Create a list of sample TestIndividuals for testing."""
    return [
        TestIndividual(
            snippet=f"def test_{i}(self): assert True",
            probability=0.1 * (i + 1),
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={},
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_test_block_builder() -> MagicMock:
    """Mock ITestBlockBuilder that returns a formatted test class."""
    mock = MagicMock()
    mock.return_value = "class TestClass:\n    pass"
    return mock


@pytest.fixture
def sample_code_population(
    sample_code_individuals: list[CodeIndividual],
) -> CodePopulation:
    """Create a sample CodePopulation."""
    return CodePopulation(sample_code_individuals, generation=0)


@pytest.fixture
def sample_test_population(
    sample_test_individuals: list[TestIndividual],
    mock_test_block_builder: MagicMock,
) -> TestPopulation:
    """Create a sample TestPopulation with mocked dependencies."""
    return TestPopulation(
        individuals=sample_test_individuals,
        test_block_rebuilder=mock_test_block_builder,
        test_class_block="class TestOriginal:\n    pass",
        generation=0,
    )


# ============================================================================
# BASE POPULATION SHARED BEHAVIOR TESTS
# ============================================================================


class TestBasePopulationSharedBehavior:
    """Test common behavior inherited from BasePopulation."""

    def test_code_population_initialization(
        self, sample_code_individuals: list[CodeIndividual]
    ) -> None:
        """Test CodePopulation initialization with valid individuals."""
        pop = CodePopulation(sample_code_individuals, generation=0)

        assert pop.size == 5
        assert pop.generation == 0
        assert len(pop._individuals) == 5

    def test_test_population_initialization(
        self,
        sample_test_individuals: list[TestIndividual],
        mock_test_block_builder: MagicMock,
    ) -> None:
        """Test TestPopulation initialization with valid individuals."""
        pop = TestPopulation(
            individuals=sample_test_individuals,
            test_block_rebuilder=mock_test_block_builder,
            test_class_block="class Test:\n    pass",
            generation=3,
        )

        assert pop.size == 5
        assert pop.generation == 3
        assert len(pop._individuals) == 5

    def test_initialization_with_empty_list(
        self, sample_code_individuals: list[CodeIndividual]
    ) -> None:
        """Test that empty list is now allowed (for bootstrap/dynamic populations)."""
        # Empty populations are now supported for dynamic population types
        pop = CodePopulation([], generation=0)
        assert pop.size == 0
        assert pop.generation == 0
        assert len(pop.individuals) == 0
        assert len(pop.probabilities) == 0
        assert len(pop.snippets) == 0
        assert len(pop.ids) == 0

    def test_len_magic_method(self, sample_code_population: CodePopulation) -> None:
        """Test __len__ returns correct size."""
        assert len(sample_code_population) == 5

    def test_getitem_magic_method(self, sample_code_population: CodePopulation) -> None:
        """Test __getitem__ allows index access."""
        individual = sample_code_population[0]
        assert isinstance(individual, CodeIndividual)
        assert individual == sample_code_population._individuals[0]

    def test_getitem_with_negative_index(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test __getitem__ works with negative indices."""
        individual = sample_code_population[-1]
        assert individual == sample_code_population._individuals[-1]

    def test_getitem_with_invalid_index(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test __getitem__ raises IndexError for invalid index."""
        with pytest.raises(IndexError):
            _ = sample_code_population[100]

    def test_iter_magic_method(self, sample_code_population: CodePopulation) -> None:
        """Test __iter__ allows iteration."""
        individuals = list(sample_code_population)
        assert len(individuals) == 5
        assert all(isinstance(ind, CodeIndividual) for ind in individuals)

    def test_size_property(self, sample_code_population: CodePopulation) -> None:
        """Test size property returns correct value."""
        assert sample_code_population.size == 5

    def test_generation_property(self, sample_code_population: CodePopulation) -> None:
        """Test generation property returns correct value."""
        assert sample_code_population.generation == 0

    def test_individuals_property_returns_copy(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test individuals property returns a copy, not reference."""
        individuals = sample_code_population.individuals

        # Modify the returned list
        individuals.append(None)  # type: ignore

        # Original should be unchanged
        assert len(sample_code_population.individuals) == 5
        assert len(individuals) == 6

    def test_probabilities_property(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test probabilities property returns numpy array."""
        probs = sample_code_population.probabilities

        assert isinstance(probs, np.ndarray)
        assert len(probs) == 5
        np.testing.assert_array_almost_equal(probs, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_snippets_property(self, sample_code_population: CodePopulation) -> None:
        """Test snippets property returns list of code snippets."""
        snippets = sample_code_population.snippets

        assert isinstance(snippets, list)
        assert len(snippets) == 5
        assert all("def func_" in s for s in snippets)

    def test_ids_property(self, sample_code_population: CodePopulation) -> None:
        """Test ids property returns list of individual IDs."""
        ids = sample_code_population.ids

        assert isinstance(ids, list)
        assert len(ids) == 5
        assert all(id.startswith("C") for id in ids)

    def test_compute_average_probability(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test compute_average_probability with various probabilities."""
        avg = sample_code_population.compute_average_probability()
        expected = (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5
        assert abs(avg - expected) < 0.0001

    def test_compute_average_probability_all_zeros(
        self, sample_code_individuals: list[CodeIndividual]
    ) -> None:
        """Test compute_average_probability with all 0.0."""
        for ind in sample_code_individuals:
            ind.probability = 0.0

        pop = CodePopulation(sample_code_individuals)
        assert pop.compute_average_probability() == 0.0

    def test_compute_average_probability_all_ones(
        self, sample_code_individuals: list[CodeIndividual]
    ) -> None:
        """Test compute_average_probability with all 1.0."""
        for ind in sample_code_individuals:
            ind.probability = 1.0

        pop = CodePopulation(sample_code_individuals)
        assert pop.compute_average_probability() == 1.0

    def test_update_probabilities_valid(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test update_probabilities with valid array."""
        new_probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        sample_code_population.update_probabilities(new_probs)

        np.testing.assert_array_almost_equal(
            sample_code_population.probabilities, new_probs
        )

    def test_update_probabilities_wrong_size(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test update_probabilities with wrong size raises ValueError."""
        new_probs = np.array([0.9, 0.8, 0.7])  # Only 3 values

        with pytest.raises(ValueError, match="must match population size"):
            sample_code_population.update_probabilities(new_probs)

    def test_get_best_individual(self, sample_code_population: CodePopulation) -> None:
        """Test get_best_individual returns highest probability."""
        best = sample_code_population.get_best_individual()

        assert best is not None
        assert best.probability == 0.5  # Highest in our sample

    def test_get_best_individual_with_ties(
        self, sample_code_individuals: list[CodeIndividual]
    ) -> None:
        """Test get_best_individual with multiple individuals at max probability."""
        # Set all to same probability
        for ind in sample_code_individuals:
            ind.probability = 0.8

        pop = CodePopulation(sample_code_individuals)
        best = pop.get_best_individual()

        assert best is not None
        assert best.probability == 0.8

    def test_get_top_k_individuals(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test get_top_k_individuals returns k highest."""
        top_3 = sample_code_population.get_top_k_individuals(3)

        assert len(top_3) == 3
        # Should be sorted descending by probability
        assert top_3[0].probability >= top_3[1].probability
        assert top_3[1].probability >= top_3[2].probability

    def test_get_top_k_individuals_k_greater_than_size(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test get_top_k_individuals with k > population size."""
        top_10 = sample_code_population.get_top_k_individuals(10)

        # Should return all individuals
        assert len(top_10) == 5

    def test_get_top_k_individuals_k_zero(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test get_top_k_individuals with k=0 returns empty list."""
        top_0 = sample_code_population.get_top_k_individuals(0)
        assert top_0 == []

    def test_get_top_k_individuals_k_negative(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test get_top_k_individuals with negative k returns empty list."""
        top_neg = sample_code_population.get_top_k_individuals(-5)
        assert top_neg == []

    def test_set_next_generation_advances_counter(
        self,
        sample_code_population: CodePopulation,
        sample_code_individuals: list[CodeIndividual],
    ) -> None:
        """Test set_next_generation advances generation counter."""
        initial_gen = sample_code_population.generation

        # Create new individuals for next generation
        new_individuals = sample_code_individuals[:3]  # Keep only 3
        sample_code_population.set_next_generation(new_individuals)

        assert sample_code_population.generation == initial_gen + 1

    def test_set_next_generation_replaces_individuals(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test set_next_generation replaces population."""
        new_individuals = [
            CodeIndividual(
                snippet="def new(): pass",
                probability=0.95,
                creation_op=OPERATION_MUTATION,
                generation_born=1,
                parents={"code": ["C0"]},
            )
        ]

        sample_code_population.set_next_generation(new_individuals)

        assert sample_code_population.size == 1
        assert sample_code_population[0].snippet == "def new(): pass"

    def test_set_next_generation_with_empty_list(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test set_next_generation with empty list results in empty population."""
        sample_code_population.set_next_generation([])

        assert sample_code_population.size == 0
        assert len(sample_code_population.individuals) == 0

    @patch("coevolution.core.interfaces.base.logger")
    def test_set_next_generation_logs_changes(
        self,
        mock_logger: MagicMock,
        sample_code_population: CodePopulation,
        sample_code_individuals: list[CodeIndividual],
    ) -> None:
        """Test set_next_generation logs kept, added, removed."""
        # Keep first 2, add 2 new (total removes 3)
        new_individuals = sample_code_individuals[:2] + [
            CodeIndividual(
                snippet="def new1(): pass",
                probability=0.6,
                creation_op=OPERATION_MUTATION,
                generation_born=1,
                parents={},
            ),
            CodeIndividual(
                snippet="def new2(): pass",
                probability=0.7,
                creation_op=OPERATION_CROSSOVER,
                generation_born=1,
                parents={},
            ),
        ]

        sample_code_population.set_next_generation(new_individuals)

        # Verify logging was called
        mock_logger.debug.assert_called()
        mock_logger.info.assert_called()


# ============================================================================
# CODE POPULATION SPECIFIC TESTS
# ============================================================================


class TestCodePopulation:
    """Test CodePopulation-specific functionality."""

    def test_initialization_with_code_individuals(
        self, sample_code_individuals: list[CodeIndividual]
    ) -> None:
        """Test CodePopulation can be initialized with CodeIndividuals."""
        pop = CodePopulation(sample_code_individuals, generation=5)

        assert pop.size == 5
        assert pop.generation == 5
        assert all(isinstance(ind, CodeIndividual) for ind in pop)

    def test_repr_format(self, sample_code_population: CodePopulation) -> None:
        """Test __repr__ string representation."""
        repr_str = repr(sample_code_population)

        assert "CodePopulation" in repr_str
        assert "gen=0" in repr_str
        assert "size=5" in repr_str
        assert "avg_prob=" in repr_str

    def test_on_generation_advanced_is_noop(
        self,
        sample_code_population: CodePopulation,
        sample_code_individuals: list[CodeIndividual],
    ) -> None:
        """Test _on_generation_advanced does nothing for CodePopulation."""
        # Advance generation
        sample_code_population.set_next_generation(sample_code_individuals[:3])

        # Should just advance counter, no special behavior
        assert sample_code_population.generation == 1
        assert sample_code_population.size == 3

    def test_multiple_generation_advances(
        self, sample_code_individuals: list[CodeIndividual]
    ) -> None:
        """Test advancing through multiple generations."""
        pop = CodePopulation(sample_code_individuals, generation=0)

        for i in range(5):
            new_individuals = sample_code_individuals[: (5 - i)]
            pop.set_next_generation(new_individuals)
            assert pop.generation == i + 1


# ============================================================================
# TEST POPULATION SPECIFIC TESTS
# ============================================================================


class TestTestPopulation:
    """Test TestPopulation-specific functionality."""

    def test_initialization_with_test_class_block(
        self,
        sample_test_individuals: list[TestIndividual],
        mock_test_block_builder: MagicMock,
    ) -> None:
        """Test TestPopulation initialization requires test_class_block."""
        test_block = "class TestExample:\n    def test_one(self): pass"

        pop = TestPopulation(
            individuals=sample_test_individuals,
            test_block_rebuilder=mock_test_block_builder,
            test_class_block=test_block,
            generation=2,
        )

        assert pop.size == 5
        assert pop.generation == 2
        assert pop.test_class_block == test_block

    def test_initialization_without_test_class_block(
        self,
        sample_test_individuals: list[TestIndividual],
        mock_test_block_builder: MagicMock,
    ) -> None:
        """Test initialization without test_class_block raises ValueError."""
        with pytest.raises(ValueError, match="test_class_block is required"):
            TestPopulation(
                individuals=sample_test_individuals,
                test_block_rebuilder=mock_test_block_builder,
                test_class_block="",
                generation=0,
            )

    def test_initialization_with_whitespace_test_class_block(
        self,
        sample_test_individuals: list[TestIndividual],
        mock_test_block_builder: MagicMock,
    ) -> None:
        """Test initialization with whitespace-only test_class_block raises ValueError."""
        with pytest.raises(ValueError, match="test_class_block is required"):
            TestPopulation(
                individuals=sample_test_individuals,
                test_block_rebuilder=mock_test_block_builder,
                test_class_block="   \n\t  ",
                generation=0,
            )

    def test_test_class_block_property(
        self, sample_test_population: TestPopulation
    ) -> None:
        """Test test_class_block property returns current block."""
        assert (
            sample_test_population.test_class_block == "class TestOriginal:\n    pass"
        )

    def test_repr_format(self, sample_test_population: TestPopulation) -> None:
        """Test __repr__ string representation."""
        repr_str = repr(sample_test_population)

        assert "TestPopulation" in repr_str
        assert "gen=0" in repr_str
        assert "size=5" in repr_str
        assert "avg_prob=" in repr_str


# ============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================


class TestPopulationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_population_with_one_individual(self) -> None:
        """Test population with single individual."""
        individual = CodeIndividual(
            snippet="def solo(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={},
        )

        pop = CodePopulation([individual], generation=0)

        assert pop.size == 1
        assert pop.get_best_individual() == individual
        assert pop.compute_average_probability() == 0.5

    def test_population_with_large_size(self) -> None:
        """Test population with many individuals."""
        individuals = [
            CodeIndividual(
                snippet=f"def func_{i}(): pass",
                probability=np.random.random(),
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
            )
            for i in range(100)
        ]

        pop = CodePopulation(individuals, generation=0)

        assert pop.size == 100
        assert len(list(pop)) == 100

    def test_all_individuals_same_probability(
        self, sample_code_individuals: list[CodeIndividual]
    ) -> None:
        """Test population where all have same probability."""
        for ind in sample_code_individuals:
            ind.probability = 0.5

        pop = CodePopulation(sample_code_individuals)

        assert pop.compute_average_probability() == 0.5
        best = pop.get_best_individual()
        assert best is not None
        assert best.probability == 0.5

    def test_empty_snippets(self) -> None:
        """Test individuals with empty code snippets."""
        individuals = [
            CodeIndividual(
                snippet="",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
            )
            for _ in range(3)
        ]

        pop = CodePopulation(individuals)

        assert pop.size == 3
        assert all(s == "" for s in pop.snippets)

    def test_very_long_snippets(self) -> None:
        """Test individuals with very long snippets."""
        long_snippet = "def long_function():\n    " + "x = 1\n    " * 1000 + "return x"
        individuals = [
            CodeIndividual(
                snippet=long_snippet,
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
            )
        ]

        pop = CodePopulation(individuals)

        assert pop.size == 1
        assert len(pop.snippets[0]) > 10000

    def test_probabilities_at_boundaries(self) -> None:
        """Test with probabilities at 0.0 and 1.0."""
        individuals = [
            CodeIndividual(
                snippet="def a(): pass",
                probability=0.0,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
            ),
            CodeIndividual(
                snippet="def b(): pass",
                probability=1.0,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
            ),
        ]

        pop = CodePopulation(individuals)

        best_individual = pop.get_best_individual()
        assert best_individual is not None
        assert best_individual.probability == 1.0
        assert pop.compute_average_probability() == 0.5

    def test_update_probabilities_with_edge_values(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Test updating probabilities with boundary values."""
        new_probs = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        sample_code_population.update_probabilities(new_probs)

        np.testing.assert_array_almost_equal(
            sample_code_population.probabilities, new_probs
        )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestPopulationIntegration:
    """Integration tests for realistic scenarios."""

    def test_evolutionary_workflow_code_population(self) -> None:
        """Test complete evolutionary workflow for CodePopulation."""
        # Initialize
        individuals = [
            CodeIndividual(
                snippet=f"def func_{i}(): return {i}",
                probability=0.3,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
            )
            for i in range(10)
        ]

        pop = CodePopulation(individuals, generation=0)

        # Simulate evolution
        for gen in range(5):
            # Update probabilities (simulate fitness evaluation)
            new_probs = np.random.uniform(0.0, 1.0, pop.size)
            pop.update_probabilities(new_probs)

            # Select best individuals
            top_5 = pop.get_top_k_individuals(5)

            # Create new generation (simulate reproduction)
            new_individuals = [
                CodeIndividual(
                    snippet=ind.snippet,
                    probability=ind.probability,
                    creation_op=OPERATION_REPRODUCTION,
                    generation_born=gen + 1,
                    parents={"code": [ind.id]},
                )
                for ind in top_5
            ]

            pop.set_next_generation(new_individuals)
            assert pop.generation == gen + 1
            assert pop.size == 5

    def test_code_and_test_populations_together(
        self,
        sample_code_individuals: list[CodeIndividual],
        sample_test_individuals: list[TestIndividual],
        mock_test_block_builder: MagicMock,
    ) -> None:
        """Test CodePopulation and TestPopulation working together."""
        code_pop = CodePopulation(sample_code_individuals, generation=0)
        test_pop = TestPopulation(
            individuals=sample_test_individuals,
            test_block_rebuilder=mock_test_block_builder,
            test_class_block="class Test:\n    pass",
            generation=0,
        )

        # Both should have independent state
        assert code_pop.generation == 0
        assert test_pop.generation == 0

        # Advance code population
        code_pop.set_next_generation(sample_code_individuals[:3])
        assert code_pop.generation == 1
        assert test_pop.generation == 0  # Test pop unchanged

        # Advance test population
        test_pop.set_next_generation(sample_test_individuals[:4])
        assert code_pop.generation == 1  # Code pop unchanged
        assert test_pop.generation == 1

    def test_probability_tracking_across_generations(self) -> None:
        """Test tracking probability evolution across generations."""
        individuals = [
            CodeIndividual(
                snippet=f"def func_{i}(): pass",
                probability=0.1,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
            )
            for i in range(5)
        ]

        pop = CodePopulation(individuals, generation=0)

        avg_probs = [pop.compute_average_probability()]

        # Simulate improvement over generations
        for gen in range(3):
            # Increase probabilities
            new_probs = pop.probabilities + 0.2
            new_probs = np.clip(new_probs, 0.0, 1.0)
            pop.update_probabilities(new_probs)

            avg_probs.append(pop.compute_average_probability())

            # Next generation
            pop.set_next_generation(individuals)

        # Average probability should increase
        assert avg_probs[-1] > avg_probs[0]


class TestPopulationAdditionalCoverage:
    """Additional tests to cover edge branches and API flexibility."""

    def test_getitem_slice_returns_list(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Slicing a population should return a list of individuals."""
        subset = sample_code_population[1:4]
        assert isinstance(subset, list)
        assert len(subset) == 3
        assert all(isinstance(ind, CodeIndividual) for ind in subset)

    def test_snippets_property_returns_copy(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Modifying returned snippets must not affect underlying population."""
        snippets = sample_code_population.snippets
        original_len = len(snippets)
        snippets.append("def injected(): pass")
        # Underlying population unchanged
        assert len(sample_code_population.snippets) == original_len

    def test_ids_property_returns_copy(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Modifying returned ids must not affect underlying population."""
        ids = sample_code_population.ids
        original_len = len(ids)
        ids.append("C999999")
        assert len(sample_code_population.ids) == original_len

    def test_update_probabilities_rejects_invalid_values(
        self, sample_code_population: CodePopulation
    ) -> None:
        """Updating with out-of-range probability should raise and not change state."""
        before = sample_code_population.probabilities.copy()
        invalid = np.array([-0.1, 0.2, 0.3, 0.4, 0.5])
        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            sample_code_population.update_probabilities(invalid)
        # Ensure no changes applied
        np.testing.assert_array_almost_equal(
            sample_code_population.probabilities, before
        )

    def test_update_probabilities_accepts_python_list(
        self, sample_code_population: CodePopulation
    ) -> None:
        """update_probabilities should accept plain Python lists."""
        new_probs = [0.5, 0.4, 0.3, 0.2, 0.1]
        sample_code_population.update_probabilities(new_probs)  # type: ignore[arg-type]
        np.testing.assert_array_almost_equal(
            sample_code_population.probabilities, new_probs
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
