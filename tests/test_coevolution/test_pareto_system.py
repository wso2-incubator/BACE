"""
Comprehensive tests for ParetoSystem implementation.

Test Coverage:
1. Discrimination calculation (entropy-based)
2. Pareto front calculation (two-objective optimization)
3. Full Pareto selection pipeline (get_pareto_indices)
4. Edge cases and error handling
5. Mathematical properties and invariants
"""

import numpy as np
import pytest

from common.coevolution.pareto_system import ParetoSystem


class TestDiscriminationCalculation:
    """Tests for the calculate_discrimination method."""

    def test_perfect_discrimination_half_pass_rate(self) -> None:
        """Test that 50% pass rate gives maximum discrimination (1.0)."""
        # 4 codes, 1 test: 2 pass, 2 fail -> pass_rate = 0.5
        observation_matrix = np.array([[1], [1], [0], [0]])

        discriminations = ParetoSystem.calculate_discrimination(observation_matrix)

        # Binary entropy is max (1.0) at p=0.5
        assert discriminations.shape == (1,)
        np.testing.assert_allclose(discriminations[0], 1.0, rtol=1e-10)

    def test_zero_discrimination_all_pass(self) -> None:
        """Test that 100% pass rate gives minimum discrimination (0.0)."""
        # All codes pass -> pass_rate = 1.0
        observation_matrix = np.array([[1], [1], [1], [1]])

        discriminations = ParetoSystem.calculate_discrimination(observation_matrix)

        # Binary entropy is 0 at p=1.0
        assert discriminations.shape == (1,)
        np.testing.assert_allclose(discriminations[0], 0.0, atol=1e-10)

    def test_zero_discrimination_all_fail(self) -> None:
        """Test that 0% pass rate gives minimum discrimination (0.0)."""
        # All codes fail -> pass_rate = 0.0
        observation_matrix = np.array([[0], [0], [0], [0]])

        discriminations = ParetoSystem.calculate_discrimination(observation_matrix)

        # Binary entropy is 0 at p=0.0
        assert discriminations.shape == (1,)
        np.testing.assert_allclose(discriminations[0], 0.0, atol=1e-10)

    def test_multiple_tests_different_discriminations(self) -> None:
        """Test discrimination calculation for multiple tests with different pass rates."""
        # Test 0: 2/4 pass (50%) -> disc = 1.0
        # Test 1: 4/4 pass (100%) -> disc = 0.0
        # Test 2: 0/4 pass (0%) -> disc = 0.0
        # Test 3: 3/4 pass (75%) -> disc = H(0.75) ≈ 0.811
        observation_matrix = np.array(
            [
                [1, 1, 0, 1],  # Code 0
                [1, 1, 0, 1],  # Code 1
                [0, 1, 0, 1],  # Code 2
                [0, 1, 0, 0],  # Code 3
            ]
        )

        discriminations = ParetoSystem.calculate_discrimination(observation_matrix)

        assert discriminations.shape == (4,)
        np.testing.assert_allclose(discriminations[0], 1.0, rtol=1e-10)  # 50%
        np.testing.assert_allclose(discriminations[1], 0.0, atol=1e-10)  # 100%
        np.testing.assert_allclose(discriminations[2], 0.0, atol=1e-10)  # 0%

        # H(0.75) = -0.75*log2(0.75) - 0.25*log2(0.25)
        expected_disc_3 = -0.75 * np.log2(0.75) - 0.25 * np.log2(0.25)
        np.testing.assert_allclose(discriminations[3], expected_disc_3, rtol=1e-10)

    def test_discrimination_always_in_range(self) -> None:
        """Test that all discriminations are in [0, 1] for various inputs."""
        # Random binary matrix
        np.random.seed(42)
        observation_matrix = np.random.randint(0, 2, size=(20, 10))

        discriminations = ParetoSystem.calculate_discrimination(observation_matrix)

        assert discriminations.shape == (10,)
        assert np.all(discriminations >= 0.0)
        assert np.all(discriminations <= 1.0)

    def test_discrimination_shape_correctness(self) -> None:
        """Test that discrimination output shape matches number of tests (columns)."""
        observation_matrix = np.random.randint(0, 2, size=(15, 8))

        discriminations = ParetoSystem.calculate_discrimination(observation_matrix)

        assert discriminations.shape == (8,)

    def test_discrimination_with_single_code(self) -> None:
        """Test discrimination calculation with only one code."""
        # With 1 code, all tests have pass_rate = 0.0 or 1.0 -> disc = 0.0
        observation_matrix = np.array([[1, 0, 1, 0]])

        discriminations = ParetoSystem.calculate_discrimination(observation_matrix)

        assert discriminations.shape == (4,)
        # All should be 0.0 (no discrimination with single code)
        np.testing.assert_allclose(discriminations, 0.0, atol=1e-10)

    def test_discrimination_empty_matrix_raises_error(self) -> None:
        """Test that empty observation matrix raises ValueError."""
        observation_matrix = np.array([]).reshape(0, 0)

        with pytest.raises(ValueError, match="Observation matrix cannot be empty"):
            ParetoSystem.calculate_discrimination(observation_matrix)

    def test_discrimination_zero_codes(self) -> None:
        """Test discrimination with zero codes returns zeros."""
        # 0 codes, 5 tests
        observation_matrix = np.array([]).reshape(0, 5)

        discriminations = ParetoSystem.calculate_discrimination(observation_matrix)

        assert discriminations.shape == (5,)
        np.testing.assert_allclose(discriminations, 0.0)


class TestParetoFrontCalculation:
    """Tests for the calculate_pareto_front method."""

    def test_single_point_is_pareto_optimal(self) -> None:
        """Test that a single point is always on the Pareto front."""
        probabilities = np.array([0.5])
        discriminations = np.array([0.7])

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        assert pareto_indices == [0]

    def test_all_dominated_by_one(self) -> None:
        """Test that only the strictly dominating point is selected."""
        # Point 0 dominates all others (higher in both objectives)
        probabilities = np.array([0.9, 0.5, 0.6, 0.4])
        discriminations = np.array([0.9, 0.5, 0.6, 0.4])

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        assert pareto_indices == [0]

    def test_all_points_pareto_optimal(self) -> None:
        """Test when all points are on the Pareto front (no dominance)."""
        # Each point is better in one objective but worse in the other
        # Point 0: (0.2, 0.9) - high disc, low prob
        # Point 1: (0.5, 0.5) - balanced
        # Point 2: (0.9, 0.2) - high prob, low disc
        probabilities = np.array([0.2, 0.5, 0.9])
        discriminations = np.array([0.9, 0.5, 0.2])

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        # All are on the front (sorted by index)
        assert sorted(pareto_indices) == [0, 1, 2]

    def test_pareto_front_with_duplicates(self) -> None:
        """Test Pareto front when some points have identical objectives."""
        # Points 0 and 1 are identical (0.5, 0.7)
        # Point 2 is different (0.3, 0.9) - NOT dominated by 0/1 due to higher disc
        probabilities = np.array([0.5, 0.5, 0.3])
        discriminations = np.array([0.7, 0.7, 0.9])

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        # All three points should be on the front:
        # - Points 0 and 1 are identical, both are Pareto optimal
        # - Point 2 has lower probability but higher discrimination, also Pareto optimal
        assert sorted(pareto_indices) == [0, 1, 2]

    def test_pareto_front_mixed_dominance(self) -> None:
        """Test a realistic scenario with mixed dominance relationships."""
        # Point 0: (0.3, 0.8) - dominated by 1
        # Point 1: (0.6, 0.9) - Pareto optimal
        # Point 2: (0.9, 0.5) - Pareto optimal
        # Point 3: (0.4, 0.4) - dominated by 1 and 2
        probabilities = np.array([0.3, 0.6, 0.9, 0.4])
        discriminations = np.array([0.8, 0.9, 0.5, 0.4])

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        # Only points 1 and 2 are on the Pareto front
        assert sorted(pareto_indices) == [1, 2]

    def test_pareto_front_length_mismatch_raises_error(self) -> None:
        """Test that mismatched array lengths raise ValueError."""
        probabilities = np.array([0.5, 0.6])
        discriminations = np.array([0.7, 0.8, 0.9])

        with pytest.raises(
            ValueError,
            match="Probabilities and discriminations must have the same length",
        ):
            ParetoSystem.calculate_pareto_front(probabilities, discriminations)

    def test_pareto_front_empty_arrays(self) -> None:
        """Test that empty arrays return empty list."""
        probabilities = np.array([])
        discriminations = np.array([])

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        assert pareto_indices == []

    def test_pareto_front_with_zeros(self) -> None:
        """Test Pareto front calculation with zero values."""
        # Point 0: (0.0, 0.0) - dominated
        # Point 1: (0.5, 0.0) - Pareto optimal
        # Point 2: (0.0, 0.5) - Pareto optimal
        probabilities = np.array([0.0, 0.5, 0.0])
        discriminations = np.array([0.0, 0.0, 0.5])

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        assert sorted(pareto_indices) == [1, 2]

    def test_pareto_front_with_ones(self) -> None:
        """Test Pareto front with maximum values."""
        # Point 0: (1.0, 1.0) - strictly dominates all
        # Point 1: (0.5, 0.5)
        # Point 2: (0.8, 0.8)
        probabilities = np.array([1.0, 0.5, 0.8])
        discriminations = np.array([1.0, 0.5, 0.8])

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        assert pareto_indices == [0]


class TestParetoIndicesAPI:
    """Tests for the get_pareto_indices public API method."""

    def test_full_pipeline_simple_case(self) -> None:
        """Test the full Pareto selection pipeline with a simple case."""
        # 4 codes, 3 tests
        # Test 0: 2/4 pass (50%) -> disc = 1.0, prob = 0.6
        # Test 1: 4/4 pass (100%) -> disc = 0.0, prob = 0.9
        # Test 2: 1/4 pass (25%) -> disc ≈ 0.811, prob = 0.4
        observation_matrix = np.array(
            [
                [1, 1, 0],  # Code 0
                [1, 1, 0],  # Code 1
                [0, 1, 1],  # Code 2
                [0, 1, 0],  # Code 3
            ]
        )
        probabilities = np.array([0.6, 0.9, 0.4])

        pareto_indices = ParetoSystem.get_pareto_indices(
            probabilities, observation_matrix
        )

        # Expected:
        # Test 0: (prob=0.6, disc=1.0) - Pareto optimal
        # Test 1: (prob=0.9, disc=0.0) - Pareto optimal (highest prob)
        # Test 2: (prob=0.4, disc≈0.811) - dominated by Test 0
        assert sorted(pareto_indices) == [0, 1]

    def test_full_pipeline_all_selected(self) -> None:
        """Test when all tests are Pareto optimal."""
        # Each test has a different trade-off
        observation_matrix = np.array(
            [
                [1, 0, 0],  # Code 0
                [1, 1, 0],  # Code 1
                [0, 1, 1],  # Code 2
                [0, 0, 1],  # Code 3
            ]
        )
        # Test 0: 2/4 pass (50%) -> disc = 1.0
        # Test 1: 2/4 pass (50%) -> disc = 1.0
        # Test 2: 2/4 pass (50%) -> disc = 1.0
        # All have same discrimination but different probabilities
        probabilities = np.array([0.9, 0.5, 0.3])

        pareto_indices = ParetoSystem.get_pareto_indices(
            probabilities, observation_matrix
        )

        # With same discrimination, higher probability dominates
        # Only Test 0 should be selected
        assert pareto_indices == [0]

    def test_full_pipeline_dimension_mismatch_raises_error(self) -> None:
        """Test that dimension mismatch raises ValueError."""
        observation_matrix = np.array([[1, 0], [0, 1]])
        probabilities = np.array([0.5, 0.6, 0.7])  # 3 tests but matrix has 2 columns

        with pytest.raises(
            ValueError, match="Number of probabilities must match number of tests"
        ):
            ParetoSystem.get_pareto_indices(probabilities, observation_matrix)

    def test_full_pipeline_with_realistic_scenario(self) -> None:
        """Test a realistic scenario with many tests and codes."""
        np.random.seed(123)
        num_codes = 50
        num_tests = 20

        # Create observation matrix with varying pass rates
        observation_matrix = np.random.randint(0, 2, size=(num_codes, num_tests))

        # Create probabilities
        probabilities = np.random.uniform(0.3, 0.9, size=num_tests)

        pareto_indices = ParetoSystem.get_pareto_indices(
            probabilities, observation_matrix
        )

        # Basic sanity checks
        assert len(pareto_indices) > 0
        assert len(pareto_indices) <= num_tests
        assert all(0 <= idx < num_tests for idx in pareto_indices)
        # No duplicates
        assert len(pareto_indices) == len(set(pareto_indices))


class TestMathematicalProperties:
    """Tests for mathematical properties and invariants."""

    def test_discrimination_symmetry(self) -> None:
        """Test that discrimination is symmetric around p=0.5."""
        # H(p) = H(1-p) for binary entropy
        observation_matrix_1 = np.array([[1], [1], [1], [0]])  # p=0.75
        observation_matrix_2 = np.array([[0], [0], [0], [1]])  # p=0.25

        disc_1 = ParetoSystem.calculate_discrimination(observation_matrix_1)
        disc_2 = ParetoSystem.calculate_discrimination(observation_matrix_2)

        np.testing.assert_allclose(disc_1, disc_2, rtol=1e-10)

    def test_discrimination_monotonic_increase_to_half(self) -> None:
        """Test that discrimination increases as pass rate approaches 0.5 from below."""
        # Create matrices with increasing pass rates: 0.25, 0.375, 0.5
        matrices = [
            np.array([[1], [0], [0], [0]]),  # 0.25
            np.array([[1], [1], [0], [0], [0], [0], [0], [0]]),  # 0.25
            np.array([[1], [1], [1], [0], [0], [0], [0], [0]]),  # 0.375
            np.array([[1], [1], [1], [1], [0], [0], [0], [0]]),  # 0.5
        ]

        discs = [ParetoSystem.calculate_discrimination(m)[0] for m in matrices]

        # Discrimination should increase as we approach 0.5
        assert discs[0] == discs[1]  # Same pass rate
        assert discs[1] < discs[2] < discs[3]

    def test_pareto_front_subset_property(self) -> None:
        """Test that Pareto front is always a subset of the original population."""
        probabilities = np.random.uniform(0, 1, size=15)
        discriminations = np.random.uniform(0, 1, size=15)

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        # All indices should be valid
        assert all(0 <= idx < len(probabilities) for idx in pareto_indices)
        # No duplicates
        assert len(pareto_indices) == len(set(pareto_indices))

    def test_pareto_front_non_dominated_property(self) -> None:
        """Test that no point on the Pareto front is dominated by another."""
        probabilities = np.array([0.2, 0.5, 0.8, 0.6, 0.4])
        discriminations = np.array([0.9, 0.7, 0.3, 0.8, 0.5])

        pareto_indices = ParetoSystem.calculate_pareto_front(
            probabilities, discriminations
        )

        # Check that no point dominates another on the front
        for i in pareto_indices:
            for j in pareto_indices:
                if i != j:
                    # i should not dominate j (not both >= and at least one >)
                    prob_i_better = probabilities[i] > probabilities[j]
                    disc_i_better = discriminations[i] > discriminations[j]

                    # i dominates j if: (prob_i >= prob_j AND disc_i >= disc_j) AND
                    # (prob_i > prob_j OR disc_i > disc_j)
                    i_dominates_j = (
                        (probabilities[i] >= probabilities[j])
                        and (discriminations[i] >= discriminations[j])
                        and (prob_i_better or disc_i_better)
                    )

                    assert not i_dominates_j, (
                        f"Point {i} dominates point {j} on Pareto front"
                    )


class TestDiversityFiltering:
    """Tests for the filter_by_diversity method."""

    def test_no_duplicates_returns_all(self) -> None:
        """Test that when there are no duplicates, all indices are returned."""
        selected_indices = [0, 1, 2]
        probabilities = np.array([0.5, 0.6, 0.7])
        discriminations = np.array([0.8, 0.9, 0.4])
        observation_matrix = np.array(
            [
                [1, 0, 1],  # Different results for each test
                [0, 1, 0],
                [1, 1, 0],
            ]
        )

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        assert sorted(result) == [0, 1, 2]

    def test_exact_duplicates_removed(self) -> None:
        """Test that exact duplicates are filtered out."""
        selected_indices = [0, 1, 2]
        probabilities = np.array([0.5, 0.5, 0.6])
        discriminations = np.array([0.8, 0.8, 0.9])
        observation_matrix = np.array(
            [
                [1, 1, 0],  # Test 0 and 1 have identical results
                [1, 1, 0],  # Test 0 and 1 have identical results
                [0, 0, 1],  # Test 2 is different
            ]
        )

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        # Should keep one of the duplicates (0 or 1) and 2
        assert len(result) == 2
        assert 2 in result
        assert (0 in result) or (1 in result)

    def test_multiple_duplicate_groups(self) -> None:
        """Test filtering with multiple groups of duplicates."""
        selected_indices = [0, 1, 2, 3, 4]
        probabilities = np.array([0.5, 0.5, 0.5, 0.7, 0.7])
        discriminations = np.array([0.8, 0.8, 0.9, 0.6, 0.6])
        observation_matrix = np.array(
            [
                [1, 1, 0, 0, 0],  # Tests 0,1 have same results
                [1, 1, 0, 0, 0],  # Tests 0,1 have same results
                [0, 0, 1, 0, 0],  # Test 2 different
                [0, 0, 0, 1, 1],  # Tests 3,4 have same results
                [0, 0, 0, 1, 1],  # Tests 3,4 have same results
            ]
        )

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        # Should keep one from group {0,1}, one from {3,4}, and 2
        assert len(result) == 3
        assert 2 in result
        # One from {0,1}
        assert len(set(result) & {0, 1}) == 1
        # One from {3,4}
        assert len(set(result) & {3, 4}) == 1

    def test_empty_selected_indices(self) -> None:
        """Test that empty input returns empty output."""
        selected_indices: list[int] = []
        probabilities = np.array([0.5, 0.6])
        discriminations = np.array([0.8, 0.9])
        observation_matrix = np.array([[1, 0], [0, 1]])

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        assert result == []

    def test_single_index_no_duplicates(self) -> None:
        """Test with single index (no duplicates possible)."""
        selected_indices = [1]
        probabilities = np.array([0.5, 0.6, 0.7])
        discriminations = np.array([0.8, 0.9, 0.4])
        observation_matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        assert result == [1]

    def test_different_probabilities_same_other_attributes(self) -> None:
        """Test that different probabilities prevent duplicate detection."""
        selected_indices = [0, 1]
        probabilities = np.array([0.5, 0.6])  # Different probabilities
        discriminations = np.array([0.8, 0.8])  # Same discrimination
        observation_matrix = np.array(
            [
                [1, 1],  # Same observation results
                [1, 1],  # Same observation results
            ]
        )

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        # Should keep both since probabilities differ
        assert sorted(result) == [0, 1]

    def test_different_discriminations_same_other_attributes(self) -> None:
        """Test that different discriminations prevent duplicate detection."""
        selected_indices = [0, 1]
        probabilities = np.array([0.5, 0.5])  # Same probability
        discriminations = np.array([0.8, 0.9])  # Different discriminations
        observation_matrix = np.array(
            [
                [1, 1],  # Same observation results
                [1, 1],  # Same observation results
            ]
        )

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        # Should keep both since discriminations differ
        assert sorted(result) == [0, 1]

    def test_different_observations_same_objectives(self) -> None:
        """Test that different observation results prevent duplicate detection."""
        selected_indices = [0, 1]
        probabilities = np.array([0.5, 0.5])  # Same probability
        discriminations = np.array([0.8, 0.8])  # Same discrimination
        observation_matrix = np.array(
            [
                [1, 0],  # Different observation results
                [0, 1],  # Different observation results
            ]
        )

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        # Should keep both since observation results differ
        assert sorted(result) == [0, 1]

    def test_all_identical_duplicates(self) -> None:
        """Test when all selected indices are identical duplicates."""
        selected_indices = [0, 1, 2]
        probabilities = np.array([0.5, 0.5, 0.5])
        discriminations = np.array([0.8, 0.8, 0.8])
        observation_matrix = np.array(
            [
                [1, 1, 1],  # All identical
                [1, 1, 1],  # All identical
                [1, 1, 1],  # All identical
            ]
        )

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        # Should keep only one (the first one encountered)
        assert result == [0]

    def test_unsorted_input_indices(self) -> None:
        """Test that unsorted input indices are handled correctly."""
        selected_indices = [2, 0, 1]  # Unsorted
        probabilities = np.array([0.5, 0.5, 0.6])
        discriminations = np.array([0.8, 0.8, 0.9])
        observation_matrix = np.array(
            [
                [1, 1, 0],  # Test 0 and 1 identical
                [1, 1, 0],  # Test 0 and 1 identical
                [0, 0, 1],  # Test 2 different
            ]
        )

        result = ParetoSystem.filter_by_diversity(
            selected_indices, probabilities, discriminations, observation_matrix
        )

        # Should keep one from {0,1} and 2
        assert len(result) == 2
        assert 2 in result
        assert (0 in result) or (1 in result)
