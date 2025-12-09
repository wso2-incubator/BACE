"""
Tests for probability assignment strategies.

This module tests the ProbabilityAssigner implementation, including:
- Initialization with different strategies
- Assignment logic for each strategy
- Edge cases (empty parents, initial population, single parent)
- Parameter validation
- Factory function
"""

import numpy as np
import pytest

from common.coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    OPERATION_REPRODUCTION,
)
from common.coevolution.probability_assigner import (
    AssignmentStrategy,
    ProbabilityAssigner,
)

# --- Initialization Tests ---


class TestProbabilityAssignerInitialization:
    """Tests for ProbabilityAssigner initialization and configuration."""

    def test_initialization_with_enum(self) -> None:
        """Test initialization with AssignmentStrategy enum."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MEAN)
        assert assigner.strategy == AssignmentStrategy.MEAN

    def test_initialization_with_string(self) -> None:
        """Test initialization with string strategy name."""
        assigner = ProbabilityAssigner("max")
        assert assigner.strategy == AssignmentStrategy.MAX

    def test_initialization_with_string_case_insensitive(self) -> None:
        """Test that string initialization is case-insensitive."""
        assigner = ProbabilityAssigner("MIN")
        assert assigner.strategy == AssignmentStrategy.MIN

    def test_initialization_invalid_strategy_string(self) -> None:
        """Test that invalid strategy string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            ProbabilityAssigner("invalid_strategy")

    def test_initialization_default_parameters(self) -> None:
        """Test that default parameters are set correctly."""
        assigner = ProbabilityAssigner()
        assert assigner.strategy == AssignmentStrategy.MEAN


# --- Initial Population Tests ---


class TestInitialPopulationAssignment:
    """Tests for assigning probabilities to initial population members."""

    @pytest.mark.parametrize(
        "strategy",
        [
            AssignmentStrategy.MEAN,
            AssignmentStrategy.MAX,
            AssignmentStrategy.MIN,
        ],
    )
    def test_initial_operation_returns_prior(
        self, strategy: AssignmentStrategy
    ) -> None:
        """Test that INITIAL operation returns initial_prior for all strategies."""
        assigner = ProbabilityAssigner(strategy=strategy)
        initial_prior = 0.5

        prob = assigner.assign_probability(
            operation=OPERATION_INITIAL,
            parent_probs=[],
            initial_prior=initial_prior,
        )

        assert prob == initial_prior

    def test_initial_with_different_priors(self) -> None:
        """Test initial assignment with various prior values."""
        assigner = ProbabilityAssigner()

        for prior in [0.1, 0.3, 0.5, 0.7, 0.9]:
            prob = assigner.assign_probability(
                operation=OPERATION_INITIAL,
                parent_probs=[],
                initial_prior=prior,
            )
            assert prob == prior


# --- Mean Strategy Tests ---


class TestMeanStrategy:
    """Tests for MEAN probability assignment strategy."""

    def test_mean_single_parent(self) -> None:
        """Test mean strategy with single parent (mutation case)."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MEAN)

        prob = assigner.assign_probability(
            operation=OPERATION_MUTATION,
            parent_probs=[0.6],
            initial_prior=0.5,
        )

        assert prob == 0.6

    def test_mean_two_parents(self) -> None:
        """Test mean strategy with two parents (crossover case)."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MEAN)

        prob = assigner.assign_probability(
            operation=OPERATION_CROSSOVER,
            parent_probs=[0.4, 0.8],
            initial_prior=0.5,
        )

        expected = (0.4 + 0.8) / 2
        assert prob == expected

    def test_mean_multiple_parents(self) -> None:
        """Test mean strategy with multiple parents."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MEAN)

        parent_probs = [0.3, 0.5, 0.7, 0.9]
        prob = assigner.assign_probability(
            operation=OPERATION_CROSSOVER,
            parent_probs=parent_probs,
            initial_prior=0.5,
        )

        expected = np.mean(parent_probs)
        assert prob == expected


# --- Max Strategy Tests ---


class TestMaxStrategy:
    """Tests for MAX (optimistic) probability assignment strategy."""

    def test_max_single_parent(self) -> None:
        """Test max strategy with single parent."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MAX)

        prob = assigner.assign_probability(
            operation=OPERATION_MUTATION,
            parent_probs=[0.7],
            initial_prior=0.5,
        )

        assert prob == 0.7

    def test_max_two_parents(self) -> None:
        """Test max strategy selects highest parent probability."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MAX)

        prob = assigner.assign_probability(
            operation=OPERATION_CROSSOVER,
            parent_probs=[0.3, 0.9],
            initial_prior=0.5,
        )

        assert prob == 0.9

    def test_max_multiple_parents(self) -> None:
        """Test max strategy with multiple parents."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MAX)

        parent_probs = [0.2, 0.5, 0.8, 0.6]
        prob = assigner.assign_probability(
            operation=OPERATION_CROSSOVER,
            parent_probs=parent_probs,
            initial_prior=0.5,
        )

        assert prob == 0.8


# --- Min Strategy Tests ---


class TestMinStrategy:
    """Tests for MIN (pessimistic) probability assignment strategy."""

    def test_min_single_parent(self) -> None:
        """Test min strategy with single parent."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MIN)

        prob = assigner.assign_probability(
            operation=OPERATION_MUTATION,
            parent_probs=[0.4],
            initial_prior=0.5,
        )

        assert prob == 0.4

    def test_min_two_parents(self) -> None:
        """Test min strategy selects lowest parent probability."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MIN)

        prob = assigner.assign_probability(
            operation=OPERATION_CROSSOVER,
            parent_probs=[0.3, 0.9],
            initial_prior=0.5,
        )

        assert prob == 0.3

    def test_min_multiple_parents(self) -> None:
        """Test min strategy with multiple parents."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MIN)

        parent_probs = [0.7, 0.5, 0.8, 0.6]
        prob = assigner.assign_probability(
            operation=OPERATION_CROSSOVER,
            parent_probs=parent_probs,
            initial_prior=0.5,
        )

        assert prob == 0.5


# --- Edge Cases and Error Handling ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_parent_probs_for_non_initial(self) -> None:
        """Test that empty parent_probs raises error for genetic operations."""
        assigner = ProbabilityAssigner()

        with pytest.raises(ValueError, match="no parent probabilities"):
            assigner.assign_probability(
                operation=OPERATION_CROSSOVER,
                parent_probs=[],
                initial_prior=0.5,
            )

    def test_extreme_probabilities(self) -> None:
        """Test assignment with extreme probability values (0 and 1)."""
        assigner = ProbabilityAssigner(AssignmentStrategy.MEAN)

        # Test with 0.0
        prob = assigner.assign_probability(
            operation=OPERATION_MUTATION,
            parent_probs=[0.0],
            initial_prior=0.5,
        )
        assert prob == 0.0

        # Test with 1.0
        prob = assigner.assign_probability(
            operation=OPERATION_MUTATION,
            parent_probs=[1.0],
            initial_prior=0.5,
        )
        assert prob == 1.0

        # Test mix
        prob = assigner.assign_probability(
            operation=OPERATION_CROSSOVER,
            parent_probs=[0.0, 1.0],
            initial_prior=0.5,
        )
        assert prob == 0.5

    def test_all_operations(self) -> None:
        """Test assignment works for all defined Operations."""
        assigner = ProbabilityAssigner()

        # Test all operations except INITIAL
        operations = [
            OPERATION_CROSSOVER,
            OPERATION_MUTATION,
            OPERATION_EDIT,
            OPERATION_REPRODUCTION,
        ]

        for op in operations:
            prob = assigner.assign_probability(
                operation=op,
                parent_probs=[0.5, 0.7],
                initial_prior=0.5,
            )
            # Should return a valid probability
            assert 0.0 <= prob <= 1.0


# --- Integration-Style Tests ---


class TestStrategyComparison:
    """Tests comparing behavior across different strategies."""

    def test_strategies_differ_for_crossover(self) -> None:
        """Test that different strategies produce different results for crossover."""
        parent_probs = [0.3, 0.9]
        initial_prior = 0.5

        mean_assigner = ProbabilityAssigner(AssignmentStrategy.MEAN)
        max_assigner = ProbabilityAssigner(AssignmentStrategy.MAX)
        min_assigner = ProbabilityAssigner(AssignmentStrategy.MIN)

        mean_prob = mean_assigner.assign_probability(
            OPERATION_CROSSOVER, parent_probs, initial_prior
        )
        max_prob = max_assigner.assign_probability(
            OPERATION_CROSSOVER, parent_probs, initial_prior
        )
        min_prob = min_assigner.assign_probability(
            OPERATION_CROSSOVER, parent_probs, initial_prior
        )

        # All should be different
        assert mean_prob == 0.6  # (0.3 + 0.9) / 2
        assert max_prob == 0.9
        assert min_prob == 0.3

        # Ordering relationship
        assert min_prob < mean_prob < max_prob

    def test_strategies_identical_for_initial(self) -> None:
        """Test that all strategies return the same value for INITIAL."""
        initial_prior = 0.5

        strategies = [
            AssignmentStrategy.MEAN,
            AssignmentStrategy.MAX,
            AssignmentStrategy.MIN,
        ]

        probs = [
            ProbabilityAssigner(s).assign_probability(
                OPERATION_INITIAL, [], initial_prior
            )
            for s in strategies
        ]

        # All should be identical
        assert all(p == initial_prior for p in probs)
