"""
Tests for elite selection strategies.

This module tests the IEliteSelectionStrategy implementations,
verifying that they correctly select elite individuals from populations
based on probability, diversity, and other criteria.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    CoevolutionContext,
    InteractionData,
    PopulationConfig,
)
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.strategies.selection.elite_selection import (
    CodeDiversityEliteSelector,
    TestDiversityEliteSelector,
    TopKEliteSelector,
)


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
        test_block_rebuilder=Mock(),
        test_class_block="class TestSuite: pass",
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


class TestTestDiversityEliteSelector:
    """Test suite for TestDiversityEliteSelector."""

    def test_initialization(self) -> None:
        """Test selector initialization with test population key."""
        selector: TestDiversityEliteSelector[TestIndividual] = (
            TestDiversityEliteSelector("unittest")
        )
        assert selector.test_population_key == "unittest"
        assert "unittest" in repr(selector)

    def test_empty_population(
        self,
        population_config: PopulationConfig,
    ) -> None:
        """Test that empty population returns empty elite list."""
        # Create empty test population (now supported)
        empty_pop = TestPopulation(
            individuals=[],
            test_block_rebuilder=Mock(),
            test_class_block="class TestSuite: pass",
        )

        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": Mock()}

        selector: TestDiversityEliteSelector[TestIndividual] = (
            TestDiversityEliteSelector("unittest")
        )
        elites = selector.select_elites(empty_pop, population_config, context)

        assert len(elites) == 0

    def test_diversity_selection_unique_columns(
        self, population_config: PopulationConfig
    ) -> None:
        """Test diversity selection when all test columns are unique."""
        # Create test population with 4 tests
        test_individuals = [
            TestIndividual(
                snippet=f"def test{i}(): assert True",
                probability=0.2 * (i + 1),  # 0.2, 0.4, 0.6, 0.8
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i in range(4)
        ]
        test_pop = TestPopulation(
            individuals=test_individuals,
            test_block_rebuilder=Mock(),
            test_class_block="class TestSuite: pass",
        )

        # Create observation matrix with unique columns (3 code x 4 tests)
        # Each test has a unique discrimination pattern
        observation_matrix = np.array(
            [
                [1, 0, 1, 0],  # code 0
                [0, 1, 1, 0],  # code 1
                [1, 1, 0, 1],  # code 2
            ]
        )

        # Create context with interaction data
        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": interaction_data}

        selector = TestDiversityEliteSelector("unittest")
        elites = selector.select_elites(test_pop, population_config, context)

        # All 4 columns are unique, so all 4 tests should be selected
        assert len(elites) == 4
        # Should return all tests since all are unique
        assert set(e.id for e in elites) == set(t.id for t in test_individuals)

    def test_diversity_selection_duplicate_columns(
        self, population_config: PopulationConfig
    ) -> None:
        """Test diversity selection when some test columns are duplicates."""
        # Create test population with 5 tests
        test_individuals = [
            TestIndividual(
                snippet=f"def test{i}(): assert True",
                probability=prob,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i, prob in enumerate([0.2, 0.8, 0.3, 0.5, 0.7])  # Varying probabilities
        ]
        test_pop = TestPopulation(
            individuals=test_individuals,
            test_block_rebuilder=Mock(),
            test_class_block="class TestSuite: pass",
        )

        # Create observation matrix with duplicate columns
        # Columns: [A, B, A, C, B] where A, B, C are unique patterns
        observation_matrix = np.array(
            [
                [1, 0, 1, 1, 0],  # code 0: tests 0,2 match (A), tests 1,4 match (B)
                [0, 1, 0, 0, 1],  # code 1: tests 0,2 match (A), tests 1,4 match (B)
                [1, 1, 1, 0, 1],  # code 2: tests 0,2 match (A), tests 1,4 match (B)
            ]
        )

        # Create context
        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": interaction_data}

        selector = TestDiversityEliteSelector("unittest")
        elites = selector.select_elites(test_pop, population_config, context)

        # 3 unique patterns (A, B, C), so expect 3 elites
        assert len(elites) == 3

        # Within each group, highest probability should be selected:
        # Group A (tests 0, 2): probs [0.2, 0.3] -> select test 2 (0.3)
        # Group B (tests 1, 4): probs [0.8, 0.7] -> select test 1 (0.8)
        # Group C (test 3): prob [0.5] -> select test 3 (0.5)

        elite_probs = sorted([e.probability for e in elites])
        assert elite_probs == [0.3, 0.5, 0.8]

    def test_diversity_selection_all_identical_columns(
        self, population_config: PopulationConfig
    ) -> None:
        """Test diversity selection when all tests have identical patterns."""
        # Create test population with 4 tests
        test_individuals = [
            TestIndividual(
                snippet=f"def test{i}(): assert True",
                probability=prob,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i, prob in enumerate([0.2, 0.8, 0.5, 0.3])
        ]
        test_pop = TestPopulation(
            individuals=test_individuals,
            test_block_rebuilder=Mock(),
            test_class_block="class TestSuite: pass",
        )

        # All columns identical
        observation_matrix = np.array(
            [
                [1, 1, 1, 1],  # code 0
                [0, 0, 0, 0],  # code 1
                [1, 1, 1, 1],  # code 2
            ]
        )

        # Create context
        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"differential": interaction_data}

        selector = TestDiversityEliteSelector("differential")
        elites = selector.select_elites(test_pop, population_config, context)

        # Only 1 unique pattern, so expect 1 elite
        assert len(elites) == 1

        # Should select the highest probability test (0.8)
        assert elites[0].probability == 0.8

    def test_fallback_on_missing_matrix(
        self,
        sample_test_population: TestPopulation,
        population_config: PopulationConfig,
    ) -> None:
        """Test fallback to top-k selection when matrix is unavailable."""
        # Create context without the expected interaction data
        context = Mock(spec=CoevolutionContext)
        context.interactions = {}  # Missing "unittest" key

        selector = TestDiversityEliteSelector("unittest")
        elites = selector.select_elites(
            sample_test_population, population_config, context
        )

        # Should fallback to top-k selection
        # With elitism_rate=0.4 and size=5, expect 2 elites
        assert len(elites) == 2
        assert elites[0].probability == 0.75  # Highest
        assert elites[1].probability == 0.60  # Second highest

    def test_fallback_on_invalid_matrix_dimensions(
        self,
        sample_test_population: TestPopulation,
        population_config: PopulationConfig,
    ) -> None:
        """Test fallback when matrix dimensions don't match population size."""
        # Create matrix with wrong number of columns
        # Population has 5 tests, but matrix has 3 columns
        observation_matrix = np.array(
            [
                [1, 0, 1],  # Wrong size!
                [0, 1, 0],
            ]
        )

        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": interaction_data}

        selector = TestDiversityEliteSelector("unittest")
        elites = selector.select_elites(
            sample_test_population, population_config, context
        )

        # Should fallback to top-k selection
        assert len(elites) == 2
        assert elites[0].probability == 0.75
        assert elites[1].probability == 0.60

    def test_diversity_selection_all_zeros_matrix(
        self, population_config: PopulationConfig
    ) -> None:
        """Test diversity selection when observation matrix is all zeros."""
        # Create test population with 3 tests
        test_individuals = [
            TestIndividual(
                snippet=f"def test{i}(): assert True",
                probability=prob,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i, prob in enumerate([0.3, 0.7, 0.5])
        ]
        test_pop = TestPopulation(
            individuals=test_individuals,
            test_block_rebuilder=Mock(),
            test_class_block="class TestSuite: pass",
        )

        # All zeros - all tests fail all code
        observation_matrix = np.array(
            [
                [0, 0, 0],  # code 0
                [0, 0, 0],  # code 1
                [0, 0, 0],  # code 2
            ]
        )

        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": interaction_data}

        selector = TestDiversityEliteSelector("unittest")
        elites = selector.select_elites(test_pop, population_config, context)

        # All columns are identical (all zeros), so only 1 unique pattern
        assert len(elites) == 1

        # Should select the highest probability test (0.7)
        assert elites[0].probability == 0.7

    def test_diversity_selection_all_ones_matrix(
        self, population_config: PopulationConfig
    ) -> None:
        """Test diversity selection when observation matrix is all ones."""
        # Create test population with 3 tests
        test_individuals = [
            TestIndividual(
                snippet=f"def test{i}(): assert True",
                probability=prob,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i, prob in enumerate([0.4, 0.9, 0.6])
        ]
        test_pop = TestPopulation(
            individuals=test_individuals,
            test_block_rebuilder=Mock(),
            test_class_block="class TestSuite: pass",
        )

        # All ones - all tests pass all code
        observation_matrix = np.array(
            [
                [1, 1, 1],  # code 0
                [1, 1, 1],  # code 1
                [1, 1, 1],  # code 2
            ]
        )

        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": interaction_data}

        selector = TestDiversityEliteSelector("unittest")
        elites = selector.select_elites(test_pop, population_config, context)

        # All columns are identical (all ones), so only 1 unique pattern
        assert len(elites) == 1

        # Should select the highest probability test (0.9)
        assert elites[0].probability == 0.9

    def test_diversity_selection_completely_unique_patterns(
        self, population_config: PopulationConfig
    ) -> None:
        """Test diversity selection when all tests have completely unique patterns."""
        # Create test population with 3 tests
        test_individuals = [
            TestIndividual(
                snippet=f"def test{i}(): assert True",
                probability=prob,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i, prob in enumerate([0.2, 0.5, 0.8])
        ]
        test_pop = TestPopulation(
            individuals=test_individuals,
            test_block_rebuilder=Mock(),
            test_class_block="class TestSuite: pass",
        )

        # Each test has a completely unique discrimination pattern
        observation_matrix = np.array(
            [
                [1, 0, 1],  # code 0
                [0, 1, 0],  # code 1
                [1, 1, 1],  # code 2
                [0, 0, 1],  # code 3
            ]
        )

        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": interaction_data}

        selector = TestDiversityEliteSelector("unittest")
        elites = selector.select_elites(test_pop, population_config, context)

        # All 3 columns are unique, so expect all 3 tests selected
        assert len(elites) == 3

        # All tests should be in the elite set
        elite_probs = sorted([e.probability for e in elites])
        assert elite_probs == [0.2, 0.5, 0.8]

    def test_repr(self) -> None:
        """Test string representation."""
        selector = TestDiversityEliteSelector("differential")
        assert (
            repr(selector)
            == "TestDiversityEliteSelector(test_population_key='differential')"
        )


class TestCodeDiversityEliteSelector:
    """Test suite for CodeDiversityEliteSelector."""

    def test_initialization(self) -> None:
        """Test selector initialization."""
        selector = CodeDiversityEliteSelector()
        assert repr(selector) == "CodeDiversityEliteSelector()"

    def test_diversity_with_single_matrix(
        self,
        sample_code_population: CodePopulation,
        population_config: PopulationConfig,
    ) -> None:
        """Test diversity selection with a single observation matrix."""
        # Create observation matrix (5 code x 3 tests)
        # Code patterns: [A, B, A, C, B] where A, B, C are unique
        observation_matrix = np.array(
            [
                [1, 0, 1],  # code 0 - pattern A, prob=0.1
                [0, 1, 0],  # code 1 - pattern B, prob=0.2
                [1, 0, 1],  # code 2 - pattern A (duplicate of 0), prob=0.3
                [1, 1, 1],  # code 3 - pattern C, prob=0.4
                [0, 1, 0],  # code 4 - pattern B (duplicate of 1), prob=0.5
            ]
        )

        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": interaction_data}

        selector = CodeDiversityEliteSelector()
        elites = selector.select_elites(
            sample_code_population, population_config, context
        )

        # Strategy: Select best from each unique group + top-k elites
        # 3 unique patterns:
        # Pattern A (code 0, 2): probs [0.1, 0.3] -> select code 2 (0.3)
        # Pattern B (code 1, 4): probs [0.2, 0.5] -> select code 4 (0.5)
        # Pattern C (code 3): prob [0.4] -> select code 3 (0.4)
        # Diverse elites: code 2, 4, 3
        #
        # Top-k elites (k = int(5 * 0.4) = 2): code 4 (0.5), code 3 (0.4)
        #
        # Combined (deduplicated): code 2 (0.3), code 4 (0.5), code 3 (0.4)
        # Total: 3 elites (all unique patterns represented + top quality guaranteed)

        assert len(elites) == 3

        elite_probs = sorted([e.probability for e in elites])
        assert elite_probs == pytest.approx([0.3, 0.4, 0.5])

    def test_diversity_with_multiple_matrices(
        self, population_config: PopulationConfig
    ) -> None:
        """Test diversity selection with multiple concatenated matrices."""
        # Create code population with 4 individuals
        code_individuals = [
            CodeIndividual(
                snippet=f"def solution{i}(): pass",
                probability=prob,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i, prob in enumerate([0.2, 0.5, 0.8, 0.3])
        ]
        code_pop = CodePopulation(individuals=code_individuals)

        # Create two observation matrices
        # unittest matrix (4 code x 2 tests)
        unittest_matrix = np.array(
            [
                [1, 0],  # code 0
                [0, 1],  # code 1
                [1, 0],  # code 2 (same as code 0 in unittest)
                [0, 0],  # code 3
            ]
        )

        # differential matrix (4 code x 2 tests)
        diff_matrix = np.array(
            [
                [1, 1],  # code 0
                [1, 0],  # code 1
                [0, 1],  # code 2 (different from code 0 when combined!)
                [0, 0],  # code 3
            ]
        )

        # Concatenated: [[1,0,1,1], [0,1,1,0], [1,0,0,1], [0,0,0,0]]
        # All 4 patterns are unique when combined!

        context = Mock(spec=CoevolutionContext)
        context.interactions = {
            "unittest": InteractionData(
                execution_results=Mock(), observation_matrix=unittest_matrix
            ),
            "differential": InteractionData(
                execution_results=Mock(), observation_matrix=diff_matrix
            ),
        }

        selector = CodeDiversityEliteSelector()
        elites = selector.select_elites(code_pop, population_config, context)

        # Strategy: best from each unique group + top-k
        # All 4 patterns unique, so diverse elites = all 4: [0.2, 0.5, 0.8, 0.3]
        # Top-k (k = int(4 * 0.4) = 1): code 2 (0.8)
        # Combined (deduplicated): all 4 code (0.2, 0.5, 0.8, 0.3)
        # Total: 4 elites

        assert len(elites) == 4

        elite_probs = sorted([e.probability for e in elites])
        assert elite_probs == [0.2, 0.3, 0.5, 0.8]

    def test_fallback_on_missing_matrices(
        self,
        sample_code_population: CodePopulation,
        population_config: PopulationConfig,
    ) -> None:
        """Test fallback to top-k when no matrices available."""
        context = Mock(spec=CoevolutionContext)
        context.interactions = {}  # No matrices

        selector = CodeDiversityEliteSelector()
        elites = selector.select_elites(
            sample_code_population, population_config, context
        )

        # Should fallback to top-k
        # With elitism_rate=0.4 and size=5, expect 2 elites
        assert len(elites) == 2
        assert elites[0].probability == 0.5  # Highest
        assert elites[1].probability == 0.4  # Second highest

    def test_fallback_on_dimension_mismatch(
        self,
        sample_code_population: CodePopulation,
        population_config: PopulationConfig,
    ) -> None:
        """Test fallback when matrix dimensions don't match."""
        # Matrix with wrong number of rows (3 instead of 5)
        observation_matrix = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )

        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": interaction_data}

        selector = CodeDiversityEliteSelector()
        elites = selector.select_elites(
            sample_code_population, population_config, context
        )

        # Should fallback to top-k
        assert len(elites) == 2
        assert elites[0].probability == 0.5
        assert elites[1].probability == 0.4

    def test_empty_population(self, population_config: PopulationConfig) -> None:
        """Test with empty population."""
        # Create empty code population (now supported)
        empty_pop = CodePopulation(individuals=[])

        context = Mock(spec=CoevolutionContext)
        context.interactions = {}

        selector = CodeDiversityEliteSelector()
        elites = selector.select_elites(empty_pop, population_config, context)

        assert len(elites) == 0

    def test_all_identical_patterns(self, population_config: PopulationConfig) -> None:
        """Test when all code have identical behavioral patterns."""
        # Create code population
        code_individuals = [
            CodeIndividual(
                snippet=f"def solution{i}(): pass",
                probability=prob,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i, prob in enumerate([0.1, 0.9, 0.5, 0.3])
        ]
        code_pop = CodePopulation(individuals=code_individuals)

        # All code have identical pattern
        observation_matrix = np.array(
            [
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
            ]
        )

        interaction_data = InteractionData(
            execution_results=Mock(), observation_matrix=observation_matrix
        )
        context = Mock(spec=CoevolutionContext)
        context.interactions = {"unittest": interaction_data}

        selector = CodeDiversityEliteSelector()
        elites = selector.select_elites(code_pop, population_config, context)

        # Strategy: best from each unique group + top-k
        # Only 1 unique pattern, so diverse elites = [code 1 (0.9)]
        # Top-k (k = int(4 * 0.4) = 1): code 1 (0.9)
        # Combined (deduplicated): code 1 (0.9)
        # Total: 1 elite

        assert len(elites) == 1
        assert elites[0].probability == 0.9

    def test_repr(self) -> None:
        """Test string representation."""
        selector = CodeDiversityEliteSelector()
        assert repr(selector) == "CodeDiversityEliteSelector()"
