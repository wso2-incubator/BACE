"""
Comprehensive tests for the breeding_strategy.py module.

This module contains tests for the BreedingStrategy class, covering:
- Initialization and setup
- Individual genetic operations (crossover, edit, reproduction, mutation)
- Operation selection logic
- Probability assignment
- Offspring creation
- Parent notification
- Integration scenarios
- Edge cases and error handling
- Randomness control
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.coevolution.core.breeding_strategy import BreedingStrategy
from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import Operations, OperatorRatesConfig
from common.coevolution.core.population import CodePopulation, TestPopulation

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_selector() -> MagicMock:
    """Mock ISelectionStrategy."""
    mock = MagicMock()
    # Default: select index 0 for single parent, indices (0, 1) for two parents
    mock.select.return_value = 0
    mock.select_parents.return_value = (0, 1)
    return mock


@pytest.fixture
def mock_operator() -> MagicMock:
    """Mock IGeneticOperator."""
    mock = MagicMock()
    mock.crossover.return_value = "def crossover_result(): pass"
    mock.edit.return_value = "def edited_result(): pass"
    mock.mutate.return_value = "def mutated_result(): pass"
    return mock


@pytest.fixture
def mock_factory() -> MagicMock:
    """Mock IIndividualFactory."""
    mock = MagicMock()
    # Return a new CodeIndividual by default
    mock.side_effect = lambda **kwargs: CodeIndividual(**kwargs)
    return mock


@pytest.fixture
def mock_probability_assigner() -> MagicMock:
    """Mock IProbabilityAssigner."""
    mock = MagicMock()
    mock.assign_probability.return_value = 0.5  # Default probability
    return mock


@pytest.fixture
def mock_feedback_generator() -> MagicMock:
    """Mock IFeedbackGenerator."""
    mock = MagicMock()
    mock.generate_feedback.return_value = "Feedback: improve this code"
    return mock


@pytest.fixture
def sample_code_population() -> CodePopulation:
    """Create a sample CodePopulation."""
    individuals = [
        CodeIndividual(
            snippet=f"def func_{i}(): return {i}",
            probability=0.1 * (i + 1),
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        )
        for i in range(5)
    ]
    return CodePopulation(individuals, generation=0)


@pytest.fixture
def sample_test_population() -> TestPopulation:
    """Create a sample TestPopulation."""
    individuals = [
        TestIndividual(
            snippet=f"def test_{i}(self): assert True",
            probability=0.1 * (i + 1),
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        )
        for i in range(5)
    ]
    # Mock dependencies for TestPopulation
    mock_pareto = MagicMock(return_value=[0, 1])
    mock_builder = MagicMock(return_value="class Test:\n    pass")
    return TestPopulation(
        individuals=individuals,
        pareto=mock_pareto,
        test_block_rebuilder=mock_builder,
        test_class_block="class Test:\n    pass",
        generation=0,
    )


@pytest.fixture
def sample_execution_results() -> MagicMock:
    """Mock ExecutionResults."""
    return MagicMock()


@pytest.fixture
def sample_observation_matrix() -> np.ndarray:
    """Create a sample observation matrix."""
    return np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])


@pytest.fixture
def default_operation_rates() -> OperatorRatesConfig:
    """Default operation rates for testing."""
    return OperatorRatesConfig(
        crossover_rate=0.3,
        edit_rate=0.3,
        mutation_rate=0.2,
    )


@pytest.fixture
def breeding_strategy(
    mock_selector: MagicMock,
    mock_operator: MagicMock,
    mock_factory: MagicMock,
    mock_probability_assigner: MagicMock,
) -> BreedingStrategy[CodeIndividual, TestIndividual]:
    """Create a BreedingStrategy with mocked dependencies."""
    return BreedingStrategy[CodeIndividual, TestIndividual](
        selector=mock_selector,
        operator=mock_operator,
        individual_factory=mock_factory,
        probability_assigner=mock_probability_assigner,
        initial_prior=0.5,
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestBreedingStrategyInitialization:
    """Test initialization and setup."""

    def test_initialization_with_valid_dependencies(
        self,
        mock_selector: MagicMock,
        mock_operator: MagicMock,
        mock_factory: MagicMock,
        mock_probability_assigner: MagicMock,
    ) -> None:
        """Test initialization stores dependencies correctly."""
        strategy: BreedingStrategy[CodeIndividual, TestIndividual] = BreedingStrategy[
            CodeIndividual, TestIndividual
        ](
            selector=mock_selector,
            operator=mock_operator,
            individual_factory=mock_factory,
            probability_assigner=mock_probability_assigner,
            initial_prior=0.5,
        )

        assert strategy.selector is mock_selector
        assert strategy.operator is mock_operator
        assert strategy.individual_factory is mock_factory
        assert strategy.probability_assigner is mock_probability_assigner
        assert strategy.initial_prior == 0.5

    def test_initialization_with_different_initial_prior(
        self,
        mock_selector: MagicMock,
        mock_operator: MagicMock,
        mock_factory: MagicMock,
        mock_probability_assigner: MagicMock,
    ) -> None:
        """Test initialization with different initial_prior values."""
        for prior in [0.0, 0.3, 0.7, 1.0]:
            strategy: BreedingStrategy[CodeIndividual, TestIndividual] = (
                BreedingStrategy[CodeIndividual, TestIndividual](
                    selector=mock_selector,
                    operator=mock_operator,
                    individual_factory=mock_factory,
                    probability_assigner=mock_probability_assigner,
                    initial_prior=prior,
                )
            )
            assert strategy.initial_prior == prior

    @patch("common.coevolution.core.breeding_strategy.logger")
    def test_initialization_logs_debug_message(
        self,
        mock_logger: MagicMock,
        mock_selector: MagicMock,
        mock_operator: MagicMock,
        mock_factory: MagicMock,
        mock_probability_assigner: MagicMock,
    ) -> None:
        """Test that initialization logs debug message."""
        _ = BreedingStrategy(
            selector=mock_selector,
            operator=mock_operator,
            individual_factory=mock_factory,
            probability_assigner=mock_probability_assigner,
            initial_prior=0.5,
        )

        mock_logger.debug.assert_called_once()
        call_str = str(mock_logger.debug.call_args)
        assert "initial_prior=0.5" in call_str


# ============================================================================
# CROSSOVER OPERATION TESTS
# ============================================================================


class TestCrossoverOperation:
    """Test crossover-specific behavior."""

    def test_perform_crossover_selects_two_parents(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        mock_selector: MagicMock,
    ) -> None:
        """Test _perform_crossover selects two parents."""
        breeding_strategy._perform_crossover(sample_code_population)

        mock_selector.select_parents.assert_called_once()
        # Verify it was called with population probabilities
        call_args = mock_selector.select_parents.call_args[0]
        assert len(call_args[0]) == 5  # 5 individuals

    def test_perform_crossover_calls_operator(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        mock_operator: MagicMock,
    ) -> None:
        """Test _perform_crossover calls operator.crossover."""
        breeding_strategy._perform_crossover(sample_code_population)

        mock_operator.crossover.assert_called_once()
        # Verify it was called with parent snippets
        call_args = mock_operator.crossover.call_args[0]
        assert "def func_0" in call_args[0]
        assert "def func_1" in call_args[1]

    def test_perform_crossover_returns_snippet_and_parents(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
    ) -> None:
        """Test _perform_crossover returns snippet and both parents."""
        snippet, parents = breeding_strategy._perform_crossover(sample_code_population)

        assert snippet == "def crossover_result(): pass"
        assert len(parents) == 2
        assert all(isinstance(p, CodeIndividual) for p in parents)

    def test_perform_crossover_with_different_parent_indices(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        mock_selector: MagicMock,
    ) -> None:
        """Test crossover with different parent selections."""
        test_cases = [(0, 2), (1, 4), (2, 3)]

        for idx1, idx2 in test_cases:
            mock_selector.select_parents.return_value = (idx1, idx2)

            snippet, parents = breeding_strategy._perform_crossover(
                sample_code_population
            )

            assert parents[0] == sample_code_population[idx1]
            assert parents[1] == sample_code_population[idx2]


# ============================================================================
# EDIT OPERATION TESTS
# ============================================================================


class TestEditOperation:
    """Test edit-specific behavior."""

    def test_perform_edit_selects_one_parent(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_selector: MagicMock,
    ) -> None:
        """Test _perform_edit selects one parent."""
        breeding_strategy._perform_edit(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            sample_observation_matrix,
            mock_feedback_generator,
        )

        mock_selector.select.assert_called_once()

    def test_perform_edit_calls_feedback_generator(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
    ) -> None:
        """Test _perform_edit calls feedback generator with correct args."""
        breeding_strategy._perform_edit(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            sample_observation_matrix,
            mock_feedback_generator,
        )

        mock_feedback_generator.generate_feedback.assert_called_once()
        call_kwargs = mock_feedback_generator.generate_feedback.call_args[1]
        assert "observation_matrix" in call_kwargs
        assert "execution_results" in call_kwargs
        assert "other_population" in call_kwargs
        assert "individual_idx" in call_kwargs

    def test_perform_edit_calls_operator_edit(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_operator: MagicMock,
    ) -> None:
        """Test _perform_edit calls operator.edit with snippet and feedback."""
        breeding_strategy._perform_edit(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            sample_observation_matrix,
            mock_feedback_generator,
        )

        mock_operator.edit.assert_called_once()
        call_args = mock_operator.edit.call_args[0]
        assert "def func_0" in call_args[0]  # Parent snippet
        # Check that feedback generator was called (feedback is passed to edit)
        mock_feedback_generator.generate_feedback.assert_called_once()

    def test_perform_edit_returns_snippet_and_single_parent(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
    ) -> None:
        """Test _perform_edit returns snippet and single parent."""
        snippet, parents = breeding_strategy._perform_edit(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            sample_observation_matrix,
            mock_feedback_generator,
        )

        assert snippet == "def edited_result(): pass"
        assert len(parents) == 1
        assert isinstance(parents[0], CodeIndividual)

    def test_perform_edit_without_feedback_generator_falls_back(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
    ) -> None:
        """Test _perform_edit falls back to reproduction without feedback generator."""
        # Cast None to satisfy type checker (testing edge case)
        from typing import cast

        from common.coevolution.core.interfaces import IFeedbackGenerator

        snippet, parents = breeding_strategy._perform_edit(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            sample_observation_matrix,
            cast(IFeedbackGenerator[TestIndividual], None),  # No feedback generator
        )

        # Should return parent's snippet (reproduction fallback)
        assert "def func_0" in snippet
        assert len(parents) == 1


# ============================================================================
# REPRODUCTION OPERATION TESTS
# ============================================================================


class TestReproductionOperation:
    """Test reproduction-specific behavior."""

    def test_perform_reproduction_selects_one_parent(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        mock_selector: MagicMock,
    ) -> None:
        """Test _perform_reproduction selects one parent."""
        breeding_strategy._perform_reproduction(sample_code_population)

        mock_selector.select.assert_called_once()

    def test_perform_reproduction_returns_original_snippet(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        mock_selector: MagicMock,
    ) -> None:
        """Test _perform_reproduction returns parent's original snippet."""
        mock_selector.select.return_value = 2

        snippet, parents = breeding_strategy._perform_reproduction(
            sample_code_population
        )

        # Should be parent's exact snippet
        assert snippet == sample_code_population[2].snippet
        assert "def func_2" in snippet

    def test_perform_reproduction_returns_single_parent(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
    ) -> None:
        """Test _perform_reproduction returns single parent."""
        snippet, parents = breeding_strategy._perform_reproduction(
            sample_code_population
        )

        assert len(parents) == 1
        assert isinstance(parents[0], CodeIndividual)

    def test_perform_reproduction_with_different_parent_indices(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        mock_selector: MagicMock,
    ) -> None:
        """Test reproduction with different parent selections."""
        for idx in range(5):
            mock_selector.select.return_value = idx

            snippet, parents = breeding_strategy._perform_reproduction(
                sample_code_population
            )

            assert parents[0] == sample_code_population[idx]
            assert snippet == sample_code_population[idx].snippet


# ============================================================================
# MUTATION OPERATION TESTS
# ============================================================================


class TestMutationOperation:
    """Test mutation application."""

    def test_apply_mutation_calls_operator(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        mock_operator: MagicMock,
    ) -> None:
        """Test _apply_mutation calls operator.mutate."""
        original_snippet = "def original(): pass"

        breeding_strategy._apply_mutation(original_snippet, Operations.CROSSOVER)

        mock_operator.mutate.assert_called_once_with(original_snippet)

    def test_apply_mutation_returns_mutated_snippet(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
    ) -> None:
        """Test _apply_mutation returns mutated snippet."""
        original_snippet = "def original(): pass"

        result = breeding_strategy._apply_mutation(
            original_snippet, Operations.REPRODUCTION
        )

        assert result == "def mutated_result(): pass"

    def test_apply_mutation_with_different_operations(
        self,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        mock_operator: MagicMock,
    ) -> None:
        """Test mutation can be applied after any operation."""
        operations: list[Operations] = [
            Operations.CROSSOVER,
            Operations.EDIT,
            Operations.REPRODUCTION,
        ]

        for op in operations:
            mock_operator.mutate.reset_mock()

            breeding_strategy._apply_mutation("def test(): pass", op)

            mock_operator.mutate.assert_called_once()


# ============================================================================
# OPERATION SELECTION TESTS
# ============================================================================


class TestOperationSelection:
    """Test probabilistic operation selection logic."""

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_crossover_selected_when_rand_below_crossover_rate(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_operator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test crossover is selected when rand < crossover_rate."""
        # First call: 0.1 (< 0.3 crossover_rate) → crossover
        # Second call: 0.5 (>= 0.2 mutation_rate) → no mutation
        mock_random.side_effect = [0.1, 0.5]

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        # Verify crossover was called
        mock_operator.crossover.assert_called_once()
        assert offspring.creation_op == Operations.CROSSOVER

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_edit_selected_when_rand_in_edit_range(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_operator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test edit is selected when crossover_rate <= rand < crossover_rate + edit_rate."""
        # First call: 0.4 (>= 0.3, < 0.6) → edit
        # Second call: 0.5 (>= 0.2) → no mutation
        mock_random.side_effect = [0.4, 0.5]

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        # Verify edit was called
        mock_operator.edit.assert_called_once()
        assert offspring.creation_op == Operations.EDIT

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_reproduction_selected_when_rand_above_edit_range(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_operator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test reproduction is selected when rand >= crossover_rate + edit_rate."""
        # First call: 0.7 (>= 0.6) → reproduction
        # Second call: 0.5 (>= 0.2) → no mutation
        mock_random.side_effect = [0.7, 0.5]

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        # Verify no genetic operations were called (reproduction returns original)
        mock_operator.crossover.assert_not_called()
        mock_operator.edit.assert_not_called()
        assert offspring.creation_op == Operations.REPRODUCTION

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_mutation_applied_when_rand_below_mutation_rate(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_operator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test mutation is applied when rand < mutation_rate."""
        # First call: 0.1 (< 0.3) → crossover
        # Second call: 0.1 (< 0.2) → mutation
        mock_random.side_effect = [0.1, 0.1]

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        # Verify mutation was applied
        mock_operator.mutate.assert_called_once()
        assert offspring.creation_op == Operations.MUTATION

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_mutation_not_applied_when_rand_above_mutation_rate(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_operator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test mutation is not applied when rand >= mutation_rate."""
        # First call: 0.1 (< 0.3) → crossover
        # Second call: 0.5 (>= 0.2) → no mutation
        mock_random.side_effect = [0.1, 0.5]

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        # Verify mutation was not applied
        mock_operator.mutate.assert_not_called()
        assert offspring.creation_op == Operations.CROSSOVER


# ============================================================================
# PROBABILITY ASSIGNMENT TESTS
# ============================================================================


class TestProbabilityAssignment:
    """Test probability calculation for different operations."""

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_probability_assigner_called_with_crossover(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_probability_assigner: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test probability assigner called with operation='crossover'."""
        mock_random.side_effect = [0.1, 0.5]  # crossover, no mutation

        breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        mock_probability_assigner.assign_probability.assert_called_once()
        call_kwargs = mock_probability_assigner.assign_probability.call_args[1]
        assert call_kwargs["operation"] == Operations.CROSSOVER

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_probability_assigner_called_with_mutation(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_probability_assigner: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test probability assigner called with operation='mutation' when mutated."""
        mock_random.side_effect = [0.1, 0.1]  # crossover, then mutation

        breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        mock_probability_assigner.assign_probability.assert_called_once()
        call_kwargs = mock_probability_assigner.assign_probability.call_args[1]
        assert call_kwargs["operation"] == Operations.MUTATION

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_probability_assigner_receives_parent_probabilities(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_probability_assigner: MagicMock,
        mock_selector: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test probability assigner receives parent probabilities."""
        mock_random.side_effect = [0.1, 0.5]  # crossover, no mutation
        mock_selector.select_parents.return_value = (0, 1)

        breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        call_kwargs = mock_probability_assigner.assign_probability.call_args[1]
        parent_probs = call_kwargs["parent_probs"]
        assert len(parent_probs) == 2
        assert parent_probs[0] == 0.1  # Parent 0 probability
        assert parent_probs[1] == 0.2  # Parent 1 probability

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_probability_assigner_receives_initial_prior(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_probability_assigner: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test probability assigner receives initial_prior."""
        mock_random.side_effect = [0.7, 0.5]  # reproduction, no mutation

        breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        call_kwargs = mock_probability_assigner.assign_probability.call_args[1]
        assert call_kwargs["initial_prior"] == 0.5


# ============================================================================
# OFFSPRING CREATION TESTS
# ============================================================================


class TestOffspringCreation:
    """Test offspring individual creation."""

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_offspring_created_with_correct_snippet(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test offspring has correct snippet based on operation."""
        mock_random.side_effect = [0.1, 0.5]  # crossover, no mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        assert offspring.snippet == "def crossover_result(): pass"

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_offspring_generation_incremented(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test offspring generation is parent generation + 1."""
        mock_random.side_effect = [0.7, 0.5]  # reproduction, no mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        assert offspring.generation_born == sample_code_population.generation + 1

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_offspring_has_correct_parent_ids(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_selector: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test offspring has correct parent IDs."""
        mock_random.side_effect = [0.1, 0.5]  # crossover, no mutation
        mock_selector.select_parents.return_value = (0, 1)

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        assert len(offspring.parent_ids) == 2
        assert offspring.parent_ids[0] == sample_code_population[0].id
        assert offspring.parent_ids[1] == sample_code_population[1].id

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_offspring_creation_op_is_final_operation(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test offspring creation_op is the final operation (mutation if applied)."""
        mock_random.side_effect = [0.1, 0.1]  # crossover, then mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        # Should be 'mutation', not 'crossover'
        assert offspring.creation_op == Operations.MUTATION


# ============================================================================
# PARENT NOTIFICATION TESTS
# ============================================================================


class TestParentNotification:
    """Test parent lifecycle event logging."""

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_parents_notified_after_crossover(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_selector: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test both parents are notified after crossover."""
        mock_random.side_effect = [0.1, 0.5]  # crossover, no mutation
        mock_selector.select_parents.return_value = (0, 1)

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        parent1 = sample_code_population[0]
        parent2 = sample_code_population[1]

        # Check parents' lifecycle logs
        parent1_events = [entry.event.value for entry in parent1.lifecycle_log]
        parent2_events = [entry.event.value for entry in parent2.lifecycle_log]

        assert "became_parent" in parent1_events
        assert "became_parent" in parent2_events

        # Check offspring ID is logged in details
        parent1_last = parent1.lifecycle_log[-1]
        assert parent1_last.details["offspring_id"] == offspring.id

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_parent_notified_with_correct_generation(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test parent is notified with current generation, not offspring's."""
        mock_random.side_effect = [0.7, 0.5]  # reproduction, no mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        parent = sample_code_population[0]
        parent_last_event = parent.lifecycle_log[-1]

        # Parent notified with current generation (0), not offspring's (1)
        assert parent_last_event.generation == 0
        assert offspring.generation_born == 1

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_parent_notified_with_final_operation(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test parent is notified with base operation (not final operation after mutation)."""
        mock_random.side_effect = [0.1, 0.1]  # crossover, then mutation

        _ = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        parent1 = sample_code_population[0]
        parent1_last_event = parent1.lifecycle_log[-1]

        # Should be notified with base operation 'crossover', not 'mutation'
        assert parent1_last_event.details["operation"] == Operations.CROSSOVER.value


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestBreedingStrategyIntegration:
    """Test complete breeding workflows end-to-end."""

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_full_crossover_without_mutation(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test complete crossover flow without mutation."""
        mock_random.side_effect = [0.1, 0.5]  # crossover, no mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        assert offspring.creation_op == Operations.CROSSOVER
        assert offspring.snippet == "def crossover_result(): pass"
        assert len(offspring.parent_ids) == 2
        assert offspring.generation_born == 1
        assert offspring.probability == 0.5

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_full_crossover_with_mutation(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test complete crossover + mutation flow."""
        mock_random.side_effect = [0.1, 0.1]  # crossover, then mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        assert offspring.creation_op == Operations.MUTATION
        assert offspring.snippet == "def mutated_result(): pass"
        assert len(offspring.parent_ids) == 2
        assert offspring.generation_born == 1

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_full_edit_without_mutation(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test complete edit flow without mutation."""
        mock_random.side_effect = [0.4, 0.5]  # edit, no mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        assert offspring.creation_op == Operations.EDIT
        assert offspring.snippet == "def edited_result(): pass"
        assert len(offspring.parent_ids) == 1
        assert offspring.generation_born == 1

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_full_reproduction_without_mutation(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test complete reproduction flow without mutation."""
        mock_random.side_effect = [0.7, 0.5]  # reproduction, no mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        assert offspring.creation_op == Operations.REPRODUCTION
        assert "def func_0" in offspring.snippet  # Parent's snippet
        assert len(offspring.parent_ids) == 1

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_multiple_offspring_generation(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test generating multiple offspring produces different results."""
        # Different random values for each offspring
        mock_random.side_effect = [
            0.1,
            0.5,  # offspring 1: crossover, no mutation
            0.4,
            0.5,  # offspring 2: edit, no mutation
            0.7,
            0.1,  # offspring 3: reproduction, mutation
        ]

        offspring1 = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        offspring2 = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        offspring3 = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        # Verify operations
        assert offspring1.creation_op == Operations.CROSSOVER
        assert offspring2.creation_op == Operations.EDIT
        assert offspring3.creation_op == Operations.MUTATION


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestBreedingStrategyEdgeCases:
    """Test boundary conditions and error handling."""

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_all_rates_at_zero(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
    ) -> None:
        """Test with all operation rates at 0.0 (should default to reproduction)."""
        mock_random.side_effect = [0.5, 0.5]  # Any value

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            OperatorRatesConfig(crossover_rate=0.0, edit_rate=0.0, mutation_rate=0.0),
        )

        assert offspring.creation_op == Operations.REPRODUCTION

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_mutation_rate_at_one(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test with mutation rate at 1.0 (always mutate)."""
        mock_random.side_effect = [0.1, 0.5]  # crossover, then check mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            OperatorRatesConfig(
                crossover_rate=default_operation_rates.crossover_rate,
                edit_rate=default_operation_rates.edit_rate,
                mutation_rate=1.0,  # Always mutate
            ),
        )

        assert offspring.creation_op == Operations.MUTATION

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_operator_exception_propagates(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        mock_operator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test that operator exceptions are propagated."""
        mock_random.side_effect = [0.1, 0.5]  # crossover
        mock_operator.crossover.side_effect = Exception("LLM generation failed")

        with pytest.raises(Exception, match="LLM generation failed"):
            breeding_strategy.generate_single_offspring(
                sample_code_population,
                sample_test_population,
                sample_execution_results,
                mock_feedback_generator,
                sample_observation_matrix,
                default_operation_rates,
            )

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_feedback_generator_exception_propagates(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test that feedback generator exceptions are propagated."""
        mock_random.side_effect = [0.4, 0.5]  # edit
        mock_feedback_generator.generate_feedback.side_effect = Exception(
            "Feedback generation failed"
        )

        with pytest.raises(Exception, match="Feedback generation failed"):
            breeding_strategy.generate_single_offspring(
                sample_code_population,
                sample_test_population,
                sample_execution_results,
                mock_feedback_generator,
                sample_observation_matrix,
                default_operation_rates,
            )

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_population_at_different_generations(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test offspring generation with population at different generations."""
        for gen in [0, 5, 100]:
            # Reset mock for each iteration
            mock_random.side_effect = [0.7, 0.5]  # reproduction, no mutation
            sample_code_population._generation = gen

            offspring = breeding_strategy.generate_single_offspring(
                sample_code_population,
                sample_test_population,
                sample_execution_results,
                mock_feedback_generator,
                sample_observation_matrix,
                default_operation_rates,
            )

            assert offspring.generation_born == gen + 1


# ============================================================================
# TYPE SAFETY TESTS
# ============================================================================


class TestBreedingStrategyTypeSafety:
    """Test generic type parameters."""

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_works_with_code_individuals(
        self,
        mock_random: MagicMock,
        breeding_strategy: BreedingStrategy[CodeIndividual, TestIndividual],
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test breeding strategy works with CodeIndividual."""
        mock_random.side_effect = [0.1, 0.5]  # crossover, no mutation

        offspring = breeding_strategy.generate_single_offspring(
            sample_code_population,
            sample_test_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        assert isinstance(offspring, CodeIndividual)
        assert offspring.id.startswith("C")

    @patch("common.coevolution.core.breeding_strategy.np.random.random")
    def test_works_with_test_individuals(
        self,
        mock_random: MagicMock,
        mock_selector: MagicMock,
        mock_operator: MagicMock,
        mock_probability_assigner: MagicMock,
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_execution_results: MagicMock,
        sample_observation_matrix: np.ndarray,
        mock_feedback_generator: MagicMock,
        default_operation_rates: OperatorRatesConfig,
    ) -> None:
        """Test breeding strategy works with TestIndividual."""
        mock_random.side_effect = [0.1, 0.5]  # crossover, no mutation

        # Create breeding strategy for TestIndividual
        test_factory = MagicMock(side_effect=lambda **kwargs: TestIndividual(**kwargs))
        test_strategy: BreedingStrategy[TestIndividual, CodeIndividual] = (
            BreedingStrategy[TestIndividual, CodeIndividual](
                selector=mock_selector,
                operator=mock_operator,
                individual_factory=test_factory,
                probability_assigner=mock_probability_assigner,
                initial_prior=0.5,
            )
        )

        offspring = test_strategy.generate_single_offspring(
            sample_test_population,
            sample_code_population,
            sample_execution_results,
            mock_feedback_generator,
            sample_observation_matrix,
            default_operation_rates,
        )

        assert isinstance(offspring, TestIndividual)
        assert offspring.id.startswith("T")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
