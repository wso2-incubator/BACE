"""Tests for the SelectionStrategy implementation."""

import numpy as np
import pytest

from common.coevolution.selection_strategy import SelectionMethod, SelectionStrategy


class TestSelectionStrategyInit:
    """Tests for SelectionStrategy initialization."""

    def test_init_with_enum(self) -> None:
        """Test initialization with enum value."""
        strategy = SelectionStrategy(SelectionMethod.BINARY_TOURNAMENT)
        assert strategy._method == SelectionMethod.BINARY_TOURNAMENT

    def test_init_with_string(self) -> None:
        """Test initialization with string value."""
        strategy = SelectionStrategy("roulette_wheel")
        assert strategy._method == SelectionMethod.ROULETTE_WHEEL

    def test_init_with_invalid_string(self) -> None:
        """Test initialization with invalid string."""
        with pytest.raises(ValueError, match="Invalid selection method"):
            SelectionStrategy("invalid_method")

    def test_init_with_all_enum_values(self) -> None:
        """Test initialization with all enum values."""
        for method in SelectionMethod:
            strategy = SelectionStrategy(method)
            assert strategy._method == method

    def test_init_with_all_string_values(self) -> None:
        """Test initialization with all string representations."""
        strings = [
            "binary_tournament",
            "roulette_wheel",
            "rank_selection",
            "random_selection",
        ]
        expected = [
            SelectionMethod.BINARY_TOURNAMENT,
            SelectionMethod.ROULETTE_WHEEL,
            SelectionMethod.RANK_SELECTION,
            SelectionMethod.RANDOM_SELECTION,
        ]
        for string, expected_method in zip(strings, expected):
            strategy = SelectionStrategy(string)
            assert strategy._method == expected_method


class TestBinaryTournament:
    """Tests for binary tournament selection."""

    def test_binary_tournament_basic(self) -> None:
        """Test basic binary tournament selection."""
        np.random.seed(42)
        probabilities = np.array([0.1, 0.5, 0.3, 0.1])
        strategy = SelectionStrategy(SelectionMethod.BINARY_TOURNAMENT)
        selected_indices = [strategy.select(probabilities) for _ in range(100)]
        assert all(0 <= idx < len(probabilities) for idx in selected_indices)

    def test_binary_tournament_single_element(self) -> None:
        """Test binary tournament with single element."""
        probabilities = np.array([1.0])
        strategy = SelectionStrategy(SelectionMethod.BINARY_TOURNAMENT)
        idx = strategy.select(probabilities)
        assert idx == 0

    def test_binary_tournament_two_elements(self) -> None:
        """Test binary tournament with two elements."""
        np.random.seed(42)
        probabilities = np.array([0.9, 0.1])
        strategy = SelectionStrategy(SelectionMethod.BINARY_TOURNAMENT)
        selected_indices = [strategy.select(probabilities) for _ in range(100)]
        count_0 = sum(1 for idx in selected_indices if idx == 0)
        assert count_0 > 50

    def test_binary_tournament_all_equal(self) -> None:
        """Test binary tournament with equal probabilities."""
        np.random.seed(42)
        probabilities = np.array([0.25, 0.25, 0.25, 0.25])
        strategy = SelectionStrategy(SelectionMethod.BINARY_TOURNAMENT)
        selected_indices = [strategy.select(probabilities) for _ in range(100)]
        unique_indices = set(selected_indices)
        assert len(unique_indices) == 4


class TestRouletteWheel:
    """Tests for roulette wheel selection."""

    def test_roulette_wheel_basic(self) -> None:
        """Test basic roulette wheel selection."""
        np.random.seed(42)
        probabilities = np.array([0.1, 0.3, 0.5, 0.1])
        strategy = SelectionStrategy(SelectionMethod.ROULETTE_WHEEL)
        selected_indices = [strategy.select(probabilities) for _ in range(1000)]
        counts = np.bincount(selected_indices, minlength=len(probabilities))
        frequencies = counts / len(selected_indices)
        assert frequencies[2] > frequencies[1]
        assert frequencies[1] > frequencies[0]

    def test_roulette_wheel_single_element(self) -> None:
        """Test roulette wheel with single element."""
        probabilities = np.array([1.0])
        strategy = SelectionStrategy(SelectionMethod.ROULETTE_WHEEL)
        idx = strategy.select(probabilities)
        assert idx == 0

    def test_roulette_wheel_all_zeros(self) -> None:
        """Test roulette wheel with all zero probabilities."""
        np.random.seed(42)
        probabilities = np.array([0.0, 0.0, 0.0, 0.0])
        strategy = SelectionStrategy(SelectionMethod.ROULETTE_WHEEL)
        idx = strategy.select(probabilities)
        assert 0 <= idx < len(probabilities)

    def test_roulette_wheel_one_nonzero(self) -> None:
        """Test roulette wheel with one non-zero probability."""
        probabilities = np.array([0.0, 0.0, 1.0, 0.0])
        strategy = SelectionStrategy(SelectionMethod.ROULETTE_WHEEL)
        idx = strategy.select(probabilities)
        assert idx == 2

    def test_roulette_wheel_vectorized_correctness(self) -> None:
        """Test that vectorized implementation produces correct distribution."""
        np.random.seed(123)
        probabilities = np.array([0.2, 0.5, 0.3])
        strategy = SelectionStrategy(SelectionMethod.ROULETTE_WHEEL)
        selected_indices = [strategy.select(probabilities) for _ in range(10000)]
        counts = np.bincount(selected_indices, minlength=len(probabilities))
        frequencies = counts / len(selected_indices)
        normalized_probs = probabilities / probabilities.sum()
        assert np.allclose(frequencies, normalized_probs, atol=0.05)


class TestRankSelection:
    """Tests for rank-based selection."""

    def test_rank_selection_basic(self) -> None:
        """Test basic rank selection."""
        np.random.seed(42)
        probabilities = np.array([0.1, 0.3, 0.5, 0.2])
        strategy = SelectionStrategy(SelectionMethod.RANK_SELECTION)
        selected_indices = [strategy.select(probabilities) for _ in range(1000)]
        counts = np.bincount(selected_indices, minlength=len(probabilities))
        assert counts[2] > counts[1]
        assert counts[1] > counts[0]

    def test_rank_selection_single_element(self) -> None:
        """Test rank selection with single element."""
        probabilities = np.array([1.0])
        strategy = SelectionStrategy(SelectionMethod.RANK_SELECTION)
        idx = strategy.select(probabilities)
        assert idx == 0

    def test_rank_selection_all_equal(self) -> None:
        """Test rank selection with equal values."""
        np.random.seed(42)
        probabilities = np.array([0.5, 0.5, 0.5, 0.5])
        strategy = SelectionStrategy(SelectionMethod.RANK_SELECTION)
        selected_indices = [strategy.select(probabilities) for _ in range(100)]
        unique_indices = set(selected_indices)
        assert len(unique_indices) == 4

    def test_rank_selection_with_duplicates(self) -> None:
        """Test rank selection with duplicate values."""
        np.random.seed(42)
        probabilities = np.array([0.1, 0.3, 0.3, 0.5])
        strategy = SelectionStrategy(SelectionMethod.RANK_SELECTION)
        selected_indices = [strategy.select(probabilities) for _ in range(1000)]
        counts = np.bincount(selected_indices, minlength=len(probabilities))
        assert counts[3] > counts[1]
        assert counts[3] > counts[2]

    def test_rank_selection_vectorized_correctness(self) -> None:
        """Test that vectorized rank selection produces correct distribution."""
        np.random.seed(456)
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        strategy = SelectionStrategy(SelectionMethod.RANK_SELECTION)
        selected_indices = [strategy.select(probabilities) for _ in range(10000)]
        counts = np.bincount(selected_indices, minlength=len(probabilities))
        frequencies = counts / len(selected_indices)
        sorted_indices = np.argsort(probabilities)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(probabilities) + 1)
        expected_probs = ranks / ranks.sum()
        assert np.allclose(frequencies, expected_probs, atol=0.05)


class TestRandomSelection:
    """Tests for random selection."""

    def test_random_selection_basic(self) -> None:
        """Test basic random selection."""
        np.random.seed(42)
        probabilities = np.array([0.1, 0.3, 0.5, 0.1])
        strategy = SelectionStrategy(SelectionMethod.RANDOM_SELECTION)
        selected_indices = [strategy.select(probabilities) for _ in range(1000)]
        counts = np.bincount(selected_indices, minlength=len(probabilities))
        frequencies = counts / len(selected_indices)
        assert all(0.15 < freq < 0.35 for freq in frequencies)

    def test_random_selection_single_element(self) -> None:
        """Test random selection with single element."""
        probabilities = np.array([1.0])
        strategy = SelectionStrategy(SelectionMethod.RANDOM_SELECTION)
        idx = strategy.select(probabilities)
        assert idx == 0

    def test_random_selection_ignores_probabilities(self) -> None:
        """Test that random selection ignores probability values."""
        np.random.seed(42)
        probabilities = np.array([0.01, 0.01, 0.01, 0.97])
        strategy = SelectionStrategy(SelectionMethod.RANDOM_SELECTION)
        selected_indices = [strategy.select(probabilities) for _ in range(1000)]
        counts = np.bincount(selected_indices, minlength=len(probabilities))
        frequencies = counts / len(selected_indices)
        assert all(0.15 < freq < 0.35 for freq in frequencies)


class TestSelectionMethodDispatch:
    """Tests for method dispatch mechanism."""

    def test_method_map_completeness(self) -> None:
        """Test that all enum values have corresponding methods."""
        strategy = SelectionStrategy(SelectionMethod.BINARY_TOURNAMENT)
        for method in SelectionMethod:
            strategy._method = method
            probabilities = np.array([0.25, 0.25, 0.25, 0.25])
            idx = strategy.select(probabilities)
            assert 0 <= idx < len(probabilities)

    def test_select_uses_correct_method(self) -> None:
        """Test that select() uses the configured method."""
        np.random.seed(42)
        probabilities = np.array([0.1, 0.3, 0.5, 0.1])
        results = {}
        for method in SelectionMethod:
            strategy = SelectionStrategy(method)
            indices = [strategy.select(probabilities) for _ in range(100)]
            results[method] = indices
        assert len(results) == len(SelectionMethod)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_large_population(self) -> None:
        """Test selection with large population."""
        np.random.seed(42)
        probabilities = np.random.rand(1000)
        strategy = SelectionStrategy(SelectionMethod.ROULETTE_WHEEL)
        idx = strategy.select(probabilities)
        assert 0 <= idx < len(probabilities)

    def test_very_small_probabilities(self) -> None:
        """Test with very small but non-zero probabilities."""
        probabilities = np.array([1e-10, 1e-10, 1e-10, 1.0])
        strategy = SelectionStrategy(SelectionMethod.ROULETTE_WHEEL)
        idx = strategy.select(probabilities)
        assert idx == 3

    def test_negative_probabilities_behavior(self) -> None:
        """Test behavior with negative probabilities (edge case)."""
        probabilities = np.array([-0.1, 0.3, 0.5, 0.3])
        strategy = SelectionStrategy(SelectionMethod.BINARY_TOURNAMENT)
        idx = strategy.select(probabilities)
        assert 0 <= idx < len(probabilities)

    def test_empty_array_raises_error(self) -> None:
        """Test that empty array raises appropriate error."""
        probabilities = np.array([])
        strategy = SelectionStrategy(SelectionMethod.BINARY_TOURNAMENT)
        with pytest.raises((ValueError, IndexError)):
            strategy.select(probabilities)
