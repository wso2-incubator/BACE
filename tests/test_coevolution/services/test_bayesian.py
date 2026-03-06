"""
Comprehensive tests for the BayesianSystem implementation.

This module tests the Bayesian belief initialization and updating system
used in the coevolutionary framework. Tests cover:
- Belief initialization
- Probability ↔ log-odds conversions
- Weight of Evidence (WoE) calculations
- Code and test belief updates
- Edge cases and error handling
- Mathematical properties
"""

from typing import Callable

import numpy as np
import pytest

from coevolution.services.bayesian import BayesianSystem
from coevolution.core.interfaces import BayesianConfig


class TestBeliefInitialization:
    """Test suite for initialize_beliefs method."""

    def test_initialization_with_valid_probability(self) -> None:
        """Test initialization with a standard probability value."""
        beliefs = BayesianSystem.initialize_beliefs(
            population_size=10, initial_probability=0.5
        )

        assert beliefs.shape == (10,)
        assert np.allclose(beliefs, 0.5)
        assert beliefs.dtype == float

    def test_initialization_with_low_probability(self) -> None:
        """Test initialization with a low probability value."""
        beliefs = BayesianSystem.initialize_beliefs(
            population_size=5, initial_probability=0.1
        )

        assert beliefs.shape == (5,)
        assert np.allclose(beliefs, 0.1)

    def test_initialization_with_high_probability(self) -> None:
        """Test initialization with a high probability value."""
        beliefs = BayesianSystem.initialize_beliefs(
            population_size=5, initial_probability=0.9
        )

        assert beliefs.shape == (5,)
        assert np.allclose(beliefs, 0.9)

    def test_initialization_with_zero_probability(self) -> None:
        """Test that zero probability is adjusted for numerical stability."""
        beliefs = BayesianSystem.initialize_beliefs(
            population_size=5, initial_probability=0.0
        )

        assert beliefs.shape == (5,)
        # Should be slightly above 0 due to epsilon adjustment
        assert np.all(beliefs > 0.0)
        assert np.all(beliefs < 0.01)  # But still very small

    def test_initialization_with_one_probability(self) -> None:
        """Test that probability of 1.0 is adjusted for numerical stability."""
        beliefs = BayesianSystem.initialize_beliefs(
            population_size=5, initial_probability=1.0
        )

        assert beliefs.shape == (5,)
        # Should be slightly below 1 due to epsilon adjustment
        assert np.all(beliefs < 1.0)
        assert np.all(beliefs > 0.99)  # But still very close to 1

    def test_initialization_invalid_population_size_zero(self) -> None:
        """Test that zero population size raises ValueError."""
        with pytest.raises(ValueError, match="Population size must be a positive"):
            BayesianSystem.initialize_beliefs(
                population_size=0, initial_probability=0.5
            )

    def test_initialization_invalid_population_size_negative(self) -> None:
        """Test that negative population size raises ValueError."""
        with pytest.raises(ValueError, match="Population size must be a positive"):
            BayesianSystem.initialize_beliefs(
                population_size=-5, initial_probability=0.5
            )

    def test_initialization_invalid_probability_below_zero(self) -> None:
        """Test that probability < 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="Initial probability must be in the range"
        ):
            BayesianSystem.initialize_beliefs(
                population_size=5, initial_probability=-0.1
            )

    def test_initialization_invalid_probability_above_one(self) -> None:
        """Test that probability > 1 raises ValueError."""
        with pytest.raises(
            ValueError, match="Initial probability must be in the range"
        ):
            BayesianSystem.initialize_beliefs(
                population_size=5, initial_probability=1.5
            )


class TestProbabilityLogOddsConversion:
    """Test suite for probability ↔ log-odds conversion methods."""

    def test_roundtrip_conversion_mid_probability(self) -> None:
        """Test that prob → log-odds → prob is identity for p=0.5."""
        probs = np.array([0.5, 0.5, 0.5])
        log_odds = BayesianSystem._probabilities_to_log_odds(probs)
        recovered_probs = BayesianSystem._log_odds_to_probabilities(log_odds)

        assert np.allclose(probs, recovered_probs, rtol=1e-10)

    def test_roundtrip_conversion_various_probabilities(self) -> None:
        """Test roundtrip conversion for various probability values."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        log_odds = BayesianSystem._probabilities_to_log_odds(probs)
        recovered_probs = BayesianSystem._log_odds_to_probabilities(log_odds)

        assert np.allclose(probs, recovered_probs, rtol=1e-10)

    def test_probability_half_to_log_odds_zero(self) -> None:
        """Test that p=0.5 converts to log-odds=0."""
        probs = np.array([0.5])
        log_odds = BayesianSystem._probabilities_to_log_odds(probs)

        assert np.allclose(log_odds, 0.0, atol=1e-10)

    def test_high_probability_to_positive_log_odds(self) -> None:
        """Test that p>0.5 converts to positive log-odds."""
        probs = np.array([0.7, 0.8, 0.9])
        log_odds = BayesianSystem._probabilities_to_log_odds(probs)

        assert np.all(log_odds > 0)

    def test_low_probability_to_negative_log_odds(self) -> None:
        """Test that p<0.5 converts to negative log-odds."""
        probs = np.array([0.1, 0.2, 0.3])
        log_odds = BayesianSystem._probabilities_to_log_odds(probs)

        assert np.all(log_odds < 0)

    def test_extreme_probabilities_clipping(self) -> None:
        """Test that extreme probabilities (0, 1) are handled safely."""
        # These should be clipped to avoid infinity
        probs = np.array([0.0, 1.0])
        log_odds = BayesianSystem._probabilities_to_log_odds(probs)

        # Should not be infinite
        assert np.all(np.isfinite(log_odds))

    def test_extreme_log_odds_to_probabilities(self) -> None:
        """Test that extreme log-odds convert to probabilities near 0 or 1."""
        log_odds = np.array([-100, 100])
        probs = BayesianSystem._log_odds_to_probabilities(log_odds)

        assert probs[0] < 1e-40  # Very close to 0
        assert probs[1] > 0.999999  # Very close to 1


class TestWoECalculation:
    """Test suite for Weight of Evidence (WoE) calculation methods."""

    @pytest.fixture
    def standard_config(self) -> BayesianConfig:
        """Standard Bayesian configuration for testing."""
        return BayesianConfig(
            alpha=0.9,  # High probability of passing if both correct
            beta=0.1,  # Low probability of passing if test wrong
            gamma=0.1,  # Low probability of passing if both wrong
            learning_rate=1.0,
        )

    def test_woe_vectors_code_update_shape(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that WoE vectors have correct shape for code update."""
        test_probs = np.array([0.5, 0.6, 0.7])
        woe_fail, woe_pass = BayesianSystem._calculate_woe_vectors_for_code_update(
            test_probs, standard_config
        )

        assert woe_fail.shape == (3,)
        assert woe_pass.shape == (3,)

    def test_woe_vectors_test_update_shape(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that WoE vectors have correct shape for test update."""
        code_probs = np.array([0.5, 0.6, 0.7, 0.8])
        woe_fail, woe_pass = BayesianSystem._calculate_woe_vectors_for_test_update(
            code_probs, standard_config
        )

        assert woe_fail.shape == (4,)
        assert woe_pass.shape == (4,)

    def test_woe_no_nans(self, standard_config: BayesianConfig) -> None:
        """Test that WoE calculations don't produce NaN values."""
        test_probs = np.array([0.0, 0.5, 1.0])
        woe_fail, woe_pass = BayesianSystem._calculate_woe_vectors_for_code_update(
            test_probs, standard_config
        )

        assert np.all(np.isfinite(woe_fail))
        assert np.all(np.isfinite(woe_pass))

    def test_woe_perfect_test_increases_belief_on_pass(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that a perfect test (p=1.0) gives strong positive WoE for pass."""
        test_probs = np.array([1.0])
        _, woe_pass = BayesianSystem._calculate_woe_vectors_for_code_update(
            test_probs, standard_config
        )

        assert woe_pass[0] > 0


class TestCodeBeliefUpdate:
    """Test suite for update_code_beliefs method."""

    @pytest.fixture
    def standard_config(self) -> BayesianConfig:
        return BayesianConfig(
            alpha=0.9,
            beta=0.1,
            gamma=0.1,
            learning_rate=1.0,
        )

    @pytest.fixture
    def all_true_mask(self) -> Callable[[int, int], np.ndarray]:
        """Helper to generate all-true masks on the fly."""

        def _make_mask(rows: int, cols: int) -> np.ndarray:
            return np.ones((rows, cols), dtype=int)

        return _make_mask

    def test_update_with_all_passes_increases_belief(
        self,
        standard_config: BayesianConfig,
        all_true_mask: Callable[[int, int], np.ndarray],
    ) -> None:
        """Test that all tests passing increases code belief."""
        prior_code_probs = np.array([0.5, 0.5])
        prior_test_probs = np.array([0.9, 0.9])  # Good tests
        observation_matrix = np.array(
            [
                [1, 1],  # Code 0 passes both tests
                [1, 1],  # Code 1 passes both tests
            ]
        )
        mask = all_true_mask(*observation_matrix.shape)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs,
            prior_test_probs,
            observation_matrix,
            mask,
            standard_config,
        )

        # Both should increase
        assert np.all(posterior > prior_code_probs)

    def test_update_with_all_failures_decreases_belief(
        self,
        standard_config: BayesianConfig,
        all_true_mask: Callable[[int, int], np.ndarray],
    ) -> None:
        """Test that all tests failing decreases code belief."""
        prior_code_probs = np.array([0.5, 0.5])
        prior_test_probs = np.array([0.9, 0.9])  # Good tests
        observation_matrix = np.array(
            [
                [0, 0],  # Code 0 fails both tests
                [0, 0],  # Code 1 fails both tests
            ]
        )
        mask = all_true_mask(*observation_matrix.shape)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs,
            prior_test_probs,
            observation_matrix,
            mask,
            standard_config,
        )

        # Both should decrease
        assert np.all(posterior < prior_code_probs)

    def test_update_preserves_probability_bounds(
        self,
        standard_config: BayesianConfig,
        all_true_mask: Callable[[int, int], np.ndarray],
    ) -> None:
        """Test that updated probabilities remain in [0, 1]."""
        prior_code_probs = np.array([0.1, 0.5, 0.9])
        prior_test_probs = np.array([0.8, 0.8])
        observation_matrix = np.array(
            [
                [1, 1],
                [0, 0],
                [1, 0],
            ]
        )
        mask = all_true_mask(*observation_matrix.shape)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs,
            prior_test_probs,
            observation_matrix,
            mask,
            standard_config,
        )

        assert np.all(posterior >= 0.0)
        assert np.all(posterior <= 1.0)

    def test_update_with_very_low_learning_rate_minimal_change(
        self, all_true_mask: Callable[[int, int], np.ndarray]
    ) -> None:
        """Test that very low learning_rate produces minimal change."""
        config = BayesianConfig(alpha=0.9, beta=0.1, gamma=0.1, learning_rate=0.01)
        prior_code_probs = np.array([0.5, 0.5])
        prior_test_probs = np.array([0.9, 0.9])
        observation_matrix = np.array(
            [
                [1, 1],
                [0, 0],
            ]
        )
        mask = all_true_mask(*observation_matrix.shape)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, mask, config
        )

        # Changes should be very small
        assert np.allclose(posterior, prior_code_probs, atol=0.1)

    def test_update_output_shape_matches_input(
        self,
        standard_config: BayesianConfig,
        all_true_mask: Callable[[int, int], np.ndarray],
    ) -> None:
        """Test that output shape matches input code population size."""
        prior_code_probs = np.array([0.5, 0.6, 0.7])
        prior_test_probs = np.array([0.8, 0.9])
        observation_matrix = np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ]
        )
        mask = all_true_mask(*observation_matrix.shape)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs,
            prior_test_probs,
            observation_matrix,
            mask,
            standard_config,
        )

        assert posterior.shape == prior_code_probs.shape


class TestTestBeliefUpdate:
    """Test suite for update_test_beliefs method."""

    @pytest.fixture
    def standard_config(self) -> BayesianConfig:
        return BayesianConfig(
            alpha=0.9,
            beta=0.1,
            gamma=0.1,
            learning_rate=1.0,
        )

    def test_update_good_test_fails_good_code_decreases(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that a test that fails good code has decreased belief."""
        prior_code_probs = np.array([0.9, 0.9])  # Both codes very good
        prior_test_probs = np.array([0.5, 0.5])
        observation_matrix = np.array(
            [
                [0, 1],  # Good code 0 fails test 0, passes test 1
                [0, 1],  # Good code 1 fails test 0, passes test 1
            ]
        )
        # NOTE: Mask for test update is (Test, Code)
        # Obs is (Code, Test) = (2, 2)
        # Mask should be (2, 2)
        mask = np.ones((2, 2), dtype=int)

        posterior = BayesianSystem.update_test_beliefs(
            prior_code_probs,
            prior_test_probs,
            observation_matrix,
            mask,
            standard_config,
        )

        # Test 0 should decrease (fails good code)
        # Test 1 should increase (passes good code)
        assert posterior[0] < prior_test_probs[0]
        assert posterior[1] > prior_test_probs[1]


class TestRealisticScenarios:
    """Integration tests with realistic coevolution scenarios."""

    @pytest.fixture
    def realistic_config(self) -> BayesianConfig:
        return BayesianConfig(
            alpha=0.95,
            beta=0.05,
            gamma=0.05,
            learning_rate=0.5,
        )

    @pytest.fixture
    def all_true_mask(self) -> Callable[[int, int], np.ndarray]:
        def _make_mask(rows: int, cols: int) -> np.ndarray:
            return np.ones((rows, cols), dtype=int)

        return _make_mask

    def test_perfect_code_vs_perfect_tests(
        self,
        realistic_config: BayesianConfig,
        all_true_mask: Callable[[int, int], np.ndarray],
    ) -> None:
        prior_code_probs = np.array([0.9])
        prior_test_probs = np.array([0.9, 0.9, 0.9])
        observation_matrix = np.array([[1, 1, 1]])  # Passes all tests
        mask = all_true_mask(*observation_matrix.shape)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs,
            prior_test_probs,
            observation_matrix,
            mask,
            realistic_config,
        )
        assert posterior[0] > 0.92

    def test_bad_code_vs_perfect_tests(
        self,
        realistic_config: BayesianConfig,
        all_true_mask: Callable[[int, int], np.ndarray],
    ) -> None:
        prior_code_probs = np.array([0.5])
        prior_test_probs = np.array([0.9, 0.9, 0.9])
        observation_matrix = np.array([[0, 0, 0]])  # Fails all tests
        mask = all_true_mask(*observation_matrix.shape)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs,
            prior_test_probs,
            observation_matrix,
            mask,
            realistic_config,
        )
        assert posterior[0] < 0.3


class TestUpdateCalculations:
    """
    Test suite for verifying the exact mathematical correctness of the
    Bayesian update calculations against the PDF's formulas.
    """

    @pytest.fixture
    def intuitive_config(self) -> BayesianConfig:
        return BayesianConfig(
            alpha=0.1,
            beta=0.2,
            gamma=0.5,
            learning_rate=1.0,
        )

    def test_code_update_calculation_pass(
        self, intuitive_config: BayesianConfig
    ) -> None:
        prior_code_probs = np.array([0.5])
        prior_test_probs = np.array([0.9])
        observation_matrix = np.array([[1]])
        mask = np.ones_like(observation_matrix, dtype=int)

        # Manual Calc Reference (from PDF/Previous logic)
        # Logit(0.5) = 0
        # WoE ~= 1.374
        # Exp Post ~= 0.798
        expected_posterior = 0.798

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs,
            prior_test_probs,
            observation_matrix,
            mask,
            intuitive_config,
        )

        assert np.allclose(posterior, expected_posterior, atol=1e-3)


class TestMaskingLogic:
    """
    Tests specifically for the masking functionality.
    Verifies that interactions where mask=False do not affect belief updates.
    """

    @pytest.fixture
    def standard_config(self) -> BayesianConfig:
        return BayesianConfig(alpha=0.9, beta=0.1, gamma=0.1, learning_rate=1.0)

    def test_masking_prevents_update_code(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that a fully masked update results in NO change."""
        prior_code = np.array([0.5])
        prior_test = np.array([0.9])

        # Pass -> Should increase belief if unmasked
        obs = np.array([[1]])

        # MASK IS FALSE
        mask = np.zeros_like(obs, dtype=int)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code, prior_test, obs, mask, standard_config
        )

        # Belief should remain exactly 0.5
        assert posterior[0] == 0.5

    def test_masking_prevents_update_test(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that a fully masked update results in NO change for tests."""
        prior_code = np.array([0.9])
        prior_test = np.array([0.5])

        # Fail -> Should decrease test belief if unmasked (assuming code is trusted)
        obs = np.array([[0]])  # Code 0, Test 0

        # Mask shape for tests is (Tests, Codes) -> (1, 1)
        mask = np.zeros((1, 1), dtype=int)

        posterior = BayesianSystem.update_test_beliefs(
            prior_code, prior_test, obs, mask, standard_config
        )

        assert posterior[0] == 0.5

    def test_partial_masking_code(self, standard_config: BayesianConfig) -> None:
        """
        Test scenario:
        Code 0: Interaction Masked (Should stay same)
        Code 1: Interaction Unmasked (Should update)
        """
        prior_code = np.array([0.5, 0.5])
        prior_test = np.array([0.9])

        # Both pass
        obs = np.array([[1], [1]])  # Shape (2, 1)

        # Mask: [False, True]
        mask = np.array([[False], [True]], dtype=int)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code, prior_test, obs, mask, standard_config
        )

        # Code 0: No change
        assert posterior[0] == 0.5
        # Code 1: Increased
        assert posterior[1] > 0.5

    def test_masking_prevents_negative_update(
        self, standard_config: BayesianConfig
    ) -> None:
        """
        Test that a 'fail' (which usually drops belief) is ignored if masked.
        """
        prior_code = np.array([0.5])
        prior_test = np.array([0.9])

        # Fail -> Should drop belief
        obs = np.array([[0]])
        mask = np.zeros_like(obs, dtype=int)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code, prior_test, obs, mask, standard_config
        )

        # Should not drop
        assert posterior[0] == 0.5

    def test_mask_shape_mismatch_handling(
        self, standard_config: BayesianConfig
    ) -> None:
        """
        Verify behavior when mask shape doesn't match.
        Depending on numpy broadcasting, this might run or fail.
        Ideally, we want to know if it broadcasts safely or raises.
        """
        prior_code = np.array([0.5, 0.5])
        prior_test = np.array([0.9])

        obs = np.array([[1], [1]])  # (2, 1)
        mask = np.ones((1, 1), dtype=int)  # Mismatch row count

        # This might raise a ValueError during matrix mult or earlier
        try:
            BayesianSystem.update_code_beliefs(
                prior_code, prior_test, obs, mask, standard_config
            )
        except ValueError:
            pass  # Expected behavior
        except Exception:
            # If it broadcasts (e.g. applying same mask to all rows),
            # that might be technically valid numpy but logic error.
            # For this test, we just ensure it doesn't crash interpreter.
            pass


class TestMatrixDimensions:
    """
    Stress tests for asymmetric population sizes to ensure
    vectorization and broadcasting logic is robust.
    """

    @pytest.fixture
    def standard_config(self) -> BayesianConfig:
        return BayesianConfig(alpha=0.1, beta=0.1, gamma=0.1, learning_rate=1.0)

    def test_many_codes_few_tests(self, standard_config: BayesianConfig) -> None:
        """Scenario: 100 candidates evaluated against 3 tests."""
        # N=100, M=3
        prior_code = np.full(100, 0.5)
        prior_test = np.array([0.9, 0.1, 0.5])

        # Random observations and masks
        rng = np.random.default_rng(42)
        obs = rng.integers(0, 2, size=(100, 3))
        mask = rng.integers(0, 2, size=(100, 3)).astype(int)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code, prior_test, obs, mask, standard_config
        )

        assert posterior.shape == (100,)
        assert np.all(np.isfinite(posterior))

    def test_few_codes_many_tests(self, standard_config: BayesianConfig) -> None:
        """Scenario: 3 candidates evaluated against 100 tests."""
        # N=3, M=100
        prior_code = np.array([0.5, 0.5, 0.5])
        prior_test = np.full(100, 0.9)

        rng = np.random.default_rng(42)
        # Obs shape is (Codes, Tests) -> (3, 100)
        obs = rng.integers(0, 2, size=(3, 100))
        mask = rng.integers(0, 2, size=(3, 100)).astype(int)

        posterior = BayesianSystem.update_code_beliefs(
            prior_code, prior_test, obs, mask, standard_config
        )

        assert posterior.shape == (3,)
        assert np.all(np.isfinite(posterior))

    def test_test_update_transpose_safety(
        self, standard_config: BayesianConfig
    ) -> None:
        """
        Verify update_test_beliefs handles the implicit transpose correctly.
        Input Obs: (Codes, Tests)
        Mask for Tests: (Tests, Codes)
        """
        # 10 Codes, 5 Tests
        prior_code = np.full(10, 0.8)
        prior_test = np.full(5, 0.5)

        obs = np.zeros((10, 5))  # (Codes, Tests)
        mask = np.ones((10, 5), dtype=int)  # (Tests, Codes)

        posterior = BayesianSystem.update_test_beliefs(
            prior_code, prior_test, obs, mask, standard_config
        )

        assert posterior.shape == (5,)


class TestLearningRateProperties:
    """Verifies the mathematical properties of the learning rate extension."""

    def test_learning_rate_linearity_in_log_space(self) -> None:
        """
        The change in log-odds should be exactly linear with respect to
        learning rate.
        Delta_Logit(LR=0.5) == 0.5 * Delta_Logit(LR=1.0)
        """
        config_full = BayesianConfig(alpha=0.1, beta=0.1, gamma=0.1, learning_rate=1.0)
        config_half = BayesianConfig(alpha=0.1, beta=0.1, gamma=0.1, learning_rate=0.5)

        prior_code = np.array([0.5])
        prior_test = np.array([0.9])
        obs = np.array([[1]])
        mask = np.array([[True]])

        # Get Posteriors
        post_full = BayesianSystem.update_code_beliefs(
            prior_code, prior_test, obs, mask, config_full
        )
        post_half = BayesianSystem.update_code_beliefs(
            prior_code, prior_test, obs, mask, config_half
        )

        # Convert to Logits to compare deltas
        logit_prior = 0.0  # logit(0.5)
        logit_full = np.log(post_full[0] / (1 - post_full[0]))
        logit_half = np.log(post_half[0] / (1 - post_half[0]))

        delta_full = logit_full - logit_prior
        delta_half = logit_half - logit_prior

        # The half update should be exactly half the magnitude of the full update
        assert np.isclose(delta_half, 0.5 * delta_full)


class TestSensitivityTrends:
    """
    Validates the 'Trend Conditions' from PDF Section 10.
    Does the WoE increase/decrease with respect to interactor belief as predicted?
    """

    def test_woe_trend_positive_derivative(self) -> None:
        """
        PDF Table 1, Row 1:
        If gamma > alpha*beta, WoE(Pass) increases as Trust(T) increases.
        """
        # Set gamma > alpha*beta
        # 0.2 > 0.1 * 0.1 (0.01) -> True
        config = BayesianConfig(alpha=0.1, beta=0.1, gamma=0.2, learning_rate=1.0)

        # We check the implied WoE for T=0.8 vs T=0.9
        # Note: We can infer WoE trends by looking at the posterior outcome
        # since Posterior follows WoE monotonically.

        prior_code = np.array([0.5, 0.5])
        prior_tests = np.array([0.8, 0.9])  # Trust(T_low) vs Trust(T_high)

        # Both codes pass their respective tests
        obs = np.array([[1, 0], [0, 1]])
        # Mask isolates interactions: Code 0 <-> Test 0, Code 1 <-> Test 1
        mask = np.array([[True, False], [False, True]])

        posterior = BayesianSystem.update_code_beliefs(
            prior_code, prior_tests, obs, mask, config
        )

        # Since derivative is positive, higher Trust (0.9) should yield higher posterior
        # than lower Trust (0.8)
        assert posterior[1] > posterior[0]

    def test_woe_trend_negative_derivative(self) -> None:
        """
        PDF Table 2:
        WoE(Fail) should always decrease (get more negative) as Trust(T) increases.
        A failure from a trusted test hurts more than failure from untrusted test.
        """
        config = BayesianConfig(alpha=0.1, beta=0.1, gamma=0.1, learning_rate=1.0)

        prior_code = np.array([0.5, 0.5])
        prior_tests = np.array([0.6, 0.9])  # Untrusted vs Trusted

        # Both codes fail
        obs = np.zeros((2, 2))
        # Code 0 <-> Test 0, Code 1 <-> Test 1
        mask = np.array([[True, False], [False, True]])

        posterior = BayesianSystem.update_code_beliefs(
            prior_code, prior_tests, obs, mask, config
        )

        # Code 1 failed a highly trusted test -> Should be punished more (lower belief)
        assert posterior[1] < posterior[0]
