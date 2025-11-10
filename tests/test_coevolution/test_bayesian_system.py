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

import numpy as np
import pytest

from common.coevolution.bayesian_system import BayesianSystem
from common.coevolution.core.interfaces import BayesianConfig


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

    def test_woe_code_update_shape(self, standard_config: BayesianConfig) -> None:
        """Test that WoE matrix has correct shape for code update."""
        test_probs = np.array([0.5, 0.6, 0.7])
        woe = BayesianSystem._calculate_woe_for_code_update(test_probs, standard_config)

        assert woe.shape == (3, 2)  # (num_tests, 2) for [fail, pass]

    def test_woe_test_update_shape(self, standard_config: BayesianConfig) -> None:
        """Test that WoE matrix has correct shape for test update."""
        code_probs = np.array([0.5, 0.6, 0.7, 0.8])
        woe = BayesianSystem._calculate_woe_for_test_update(code_probs, standard_config)

        assert woe.shape == (4, 2)  # (num_codes, 2) for [fail, pass]

    def test_woe_no_nans(self, standard_config: BayesianConfig) -> None:
        """Test that WoE calculations don't produce NaN values."""
        test_probs = np.array([0.0, 0.5, 1.0])
        woe = BayesianSystem._calculate_woe_for_code_update(test_probs, standard_config)

        assert np.all(np.isfinite(woe))

    def test_woe_perfect_test_increases_belief_on_pass(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that a perfect test (p=1.0) gives strong positive WoE for pass."""
        test_probs = np.array([1.0])
        woe = BayesianSystem._calculate_woe_for_code_update(test_probs, standard_config)

        # Ensure woe is 2D even with single test
        if woe.ndim == 1:
            woe = woe.reshape(-1, 2)

        # woe[:, 1] is WoE for pass - should be positive
        assert woe[0, 1] > 0

    def test_woe_calculation_is_finite(self, standard_config: BayesianConfig) -> None:
        """Test that WoE calculations remain finite even with extreme probabilities."""
        test_probs = np.array([0.0, 0.5, 1.0])
        woe = BayesianSystem._calculate_woe_for_code_update(test_probs, standard_config)

        # All WoE values should be finite
        assert np.all(np.isfinite(woe))
        # Shape should be (3, 2)
        assert woe.shape == (3, 2)


class TestCodeBeliefUpdate:
    """Test suite for update_code_beliefs method."""

    @pytest.fixture
    def standard_config(self) -> BayesianConfig:
        """Standard Bayesian configuration for testing."""
        return BayesianConfig(
            alpha=0.9,
            beta=0.1,
            gamma=0.1,
            learning_rate=1.0,
        )

    def test_update_with_all_passes_increases_belief(
        self, standard_config: BayesianConfig
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

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, standard_config
        )

        # Both should increase
        assert np.all(posterior > prior_code_probs)

    def test_update_with_all_failures_decreases_belief(
        self, standard_config: BayesianConfig
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

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, standard_config
        )

        # Both should decrease
        assert np.all(posterior < prior_code_probs)

    def test_update_preserves_probability_bounds(
        self, standard_config: BayesianConfig
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

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, standard_config
        )

        assert np.all(posterior >= 0.0)
        assert np.all(posterior <= 1.0)

    def test_update_with_very_low_learning_rate_minimal_change(self) -> None:
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

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, config
        )

        # Changes should be very small
        assert np.allclose(posterior, prior_code_probs, atol=0.1)

    def test_update_with_half_learning_rate_partial_change(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that learning_rate=0.5 produces half the update."""
        prior_code_probs = np.array([0.5])
        prior_test_probs = np.array([0.9])
        observation_matrix = np.array([[1]])

        # Full update
        posterior_full = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, standard_config
        )

        # Half update
        half_config = BayesianConfig(alpha=0.9, beta=0.1, gamma=0.1, learning_rate=0.5)
        posterior_half = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, half_config
        )

        # Half update should be between prior and full update
        assert prior_code_probs[0] < posterior_half[0] < posterior_full[0]

    def test_update_output_shape_matches_input(
        self, standard_config: BayesianConfig
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

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, standard_config
        )

        assert posterior.shape == prior_code_probs.shape


class TestTestBeliefUpdate:
    """Test suite for update_test_beliefs method."""

    @pytest.fixture
    def standard_config(self) -> BayesianConfig:
        """Standard Bayesian configuration for testing."""
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

        posterior = BayesianSystem.update_test_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, standard_config
        )

        # Test 0 should decrease (fails good code)
        # Test 1 should increase (passes good code)
        assert posterior[0] < prior_test_probs[0]
        assert posterior[1] > prior_test_probs[1]

    def test_update_preserves_probability_bounds(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that updated probabilities remain in [0, 1]."""
        prior_code_probs = np.array([0.1, 0.5, 0.9])
        prior_test_probs = np.array([0.2, 0.8])
        observation_matrix = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )

        posterior = BayesianSystem.update_test_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, standard_config
        )

        assert np.all(posterior >= 0.0)
        assert np.all(posterior <= 1.0)

    def test_update_with_very_low_learning_rate_minimal_change(self) -> None:
        """Test that very low learning_rate produces minimal change."""
        config = BayesianConfig(alpha=0.9, beta=0.1, gamma=0.1, learning_rate=0.01)
        prior_code_probs = np.array([0.5, 0.5])
        prior_test_probs = np.array([0.5, 0.5])
        observation_matrix = np.array(
            [
                [1, 0],
                [0, 1],
            ]
        )

        posterior = BayesianSystem.update_test_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, config
        )

        # Changes should be very small
        assert np.allclose(posterior, prior_test_probs, atol=0.1)

    def test_update_output_shape_matches_input(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that output shape matches input test population size."""
        prior_code_probs = np.array([0.5, 0.6, 0.7])
        prior_test_probs = np.array([0.8, 0.9])
        observation_matrix = np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ]
        )

        posterior = BayesianSystem.update_test_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, standard_config
        )

        assert posterior.shape == prior_test_probs.shape


class TestRealisticScenarios:
    """Integration tests with realistic coevolution scenarios."""

    @pytest.fixture
    def realistic_config(self) -> BayesianConfig:
        """Realistic Bayesian configuration."""
        return BayesianConfig(
            alpha=0.95,  # Very high chance both correct → pass
            beta=0.05,  # Low chance test wrong helps
            gamma=0.05,  # Low chance both wrong → pass
            learning_rate=0.5,  # Moderate learning
        )

    def test_perfect_code_vs_perfect_tests(
        self, realistic_config: BayesianConfig
    ) -> None:
        """Test scenario where perfect code passes all perfect tests."""
        prior_code_probs = np.array([0.9])
        prior_test_probs = np.array([0.9, 0.9, 0.9])
        observation_matrix = np.array([[1, 1, 1]])  # Passes all tests

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, realistic_config
        )

        # Should increase belief significantly
        assert posterior[0] > 0.92

    def test_bad_code_vs_perfect_tests(self, realistic_config: BayesianConfig) -> None:
        """Test scenario where bad code fails all perfect tests."""
        prior_code_probs = np.array([0.5])
        prior_test_probs = np.array([0.9, 0.9, 0.9])
        observation_matrix = np.array([[0, 0, 0]])  # Fails all tests

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, realistic_config
        )

        # Should decrease belief significantly
        assert posterior[0] < 0.3

    def test_mixed_scenario(self, realistic_config: BayesianConfig) -> None:
        """Test a mixed scenario with various code and test qualities."""
        prior_code_probs = np.array([0.2, 0.5, 0.8])
        prior_test_probs = np.array([0.3, 0.7, 0.9])
        observation_matrix = np.array(
            [
                [0, 0, 0],  # Bad code fails all
                [0, 1, 1],  # Medium code passes good tests
                [1, 1, 1],  # Good code passes all
            ]
        )

        code_posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, realistic_config
        )

        test_posterior = BayesianSystem.update_test_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, realistic_config
        )

        # Check beliefs are updated reasonably
        assert np.all(code_posterior >= 0.0)
        assert np.all(code_posterior <= 1.0)
        assert np.all(test_posterior >= 0.0)
        assert np.all(test_posterior <= 1.0)

        # Bad code should decrease
        assert code_posterior[0] < prior_code_probs[0]
        # Good code should increase
        assert code_posterior[2] > prior_code_probs[2]


class TestMathematicalProperties:
    """Test mathematical properties of the Bayesian system."""

    @pytest.fixture
    def standard_config(self) -> BayesianConfig:
        """Standard Bayesian configuration for testing."""
        return BayesianConfig(
            alpha=0.9,
            beta=0.1,
            gamma=0.1,
            learning_rate=1.0,
        )

    def test_monotonicity_more_passes_higher_belief(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that more test passes → higher code belief."""
        prior_code_probs = np.array([0.5, 0.5, 0.5])
        prior_test_probs = np.array([0.9, 0.9, 0.9])

        # 0 passes, 1 pass, 2 passes
        obs_matrix = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
            ]
        )

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, obs_matrix, standard_config
        )

        # Should be monotonically increasing
        assert posterior[0] < posterior[1] < posterior[2]

    def test_symmetry_identical_inputs_identical_outputs(
        self, standard_config: BayesianConfig
    ) -> None:
        """Test that identical code members get identical updates."""
        prior_code_probs = np.array([0.5, 0.5, 0.5])
        prior_test_probs = np.array([0.9, 0.9])

        # All codes have same interactions
        obs_matrix = np.array(
            [
                [1, 0],
                [1, 0],
                [1, 0],
            ]
        )

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, obs_matrix, standard_config
        )

        # All should have same posterior
        assert np.allclose(posterior[0], posterior[1])
        assert np.allclose(posterior[1], posterior[2])

    def test_extreme_priors_stay_bounded(self, standard_config: BayesianConfig) -> None:
        """Test that even with extreme priors, posteriors stay in [0, 1]."""
        # Start with very extreme priors
        prior_code_probs = np.array([0.01, 0.99])
        prior_test_probs = np.array([0.01, 0.99])

        obs_matrix = np.array(
            [
                [1, 1],
                [0, 0],
            ]
        )

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, obs_matrix, standard_config
        )

        assert np.all(posterior >= 0.0)
        assert np.all(posterior <= 1.0)
        assert np.all(np.isfinite(posterior))


class TestUpdateCalculations:
    """
    Test suite for verifying the exact mathematical correctness of the
    Bayesian update calculations against the PDF's formulas.

    These tests validate that the implementation produces the exact values
    predicted by the mathematical model, not just directional changes.
    """

    @pytest.fixture
    def intuitive_config(self) -> BayesianConfig:
        """
        A 'common sense' config based on the PDF definitions.

        - alpha = P(pass | C, !T): Correct code passing incorrect test (low)
        - beta = P(pass | !C, T): Incorrect code passing correct test (low)
        - gamma = P(pass | !C, !T): Incorrect code passing incorrect test (coin flip)
        - learning_rate = 1.0 (full Bayesian update)
        """
        return BayesianConfig(
            alpha=0.1,  # Correct code rarely passes incorrect test
            beta=0.2,  # Incorrect code rarely passes correct test
            gamma=0.5,  # When both wrong, 50/50 chance
            learning_rate=1.0,
        )

    def test_code_update_calculation_pass(
        self, intuitive_config: BayesianConfig
    ) -> None:
        """
        Manually verify the posterior probability for a code candidate that
        passes a single, reliable test, based on PDF equations.
        """
        # P(C_i) = 0.5 (Prior logit = 0)
        prior_code_probs = np.array([0.5])
        # P(T_j) = 0.9 (A reliable test)
        prior_test_probs = np.array([0.9])
        observation_matrix = np.array([[1]])  # Code passes the test

        # --- Manual Calculation (from PDF) ---
        # Extract hyperparameters for clarity
        _t, _a, _b, _g = 0.9, 0.1, 0.2, 0.5

        # 1. Likelihoods (Sec 4.1)
        # P(D=1|C) = P(T) + a*(1-P(T)) = 0.9 + 0.1*(0.1) = 0.91
        like_pass_c_correct = 0.91
        # P(D=1|!C) = b*P(T) + g*(1-P(T)) = 0.2*0.9 + 0.5*(0.1) = 0.18 + 0.05 = 0.23
        like_pass_c_incorrect = 0.23

        # 2. Weight of Evidence (Sec 7.2)
        # WoE(D=1) = log( P(D=1|C) / P(D=1|!C) )
        woe = np.log(
            like_pass_c_correct / like_pass_c_incorrect
        )  # log(0.91 / 0.23) ~= 1.374

        # 3. Posterior Logit (Sec 7.1)
        # L_post = L_prior + WoE
        prior_logit = 0.0  # (since P(C_i) = 0.5)
        post_logit = prior_logit + woe

        # 4. Posterior Probability (Sec 7.1)
        # P_post = 1 / (1 + e^(-L))
        expected_posterior = 1 / (1 + np.exp(-post_logit))  # ~= 0.798

        # --- Run Actual Function ---
        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, intuitive_config
        )

        assert np.allclose(posterior, expected_posterior, atol=1e-3)

    def test_code_update_calculation_fail(
        self, intuitive_config: BayesianConfig
    ) -> None:
        """
        Manually verify the posterior probability for a code candidate that
        fails a single, reliable test.
        """
        # P(C_i) = 0.5 (Prior logit = 0)
        prior_code_probs = np.array([0.5])
        # P(T_j) = 0.9 (A reliable test)
        prior_test_probs = np.array([0.9])
        observation_matrix = np.array([[0]])  # Code fails the test

        # --- Manual Calculation (from PDF) ---
        # Extract hyperparameters for clarity
        _t, _a, _b, _g = 0.9, 0.1, 0.2, 0.5

        # 1. Likelihoods (Sec 4.1)
        # P(D=0|C) = (1-a)*(1-P(T)) = (0.9)*(0.1) = 0.09
        like_fail_c_correct = 0.09
        # P(D=0|!C) = (1-b)*P(T) + (1-g)*(1-P(T)) = 0.8*0.9 + 0.5*0.1 = 0.72 + 0.05 = 0.77
        like_fail_c_incorrect = 0.77

        # 2. Weight of Evidence (Sec 7.2)
        # WoE(D=0) = log( P(D=0|C) / P(D=0|!C) )
        woe = np.log(
            like_fail_c_correct / like_fail_c_incorrect
        )  # log(0.09 / 0.77) ~= -2.146

        # 3. Posterior Logit & Prob
        post_logit = 0.0 + woe
        expected_posterior = 1 / (1 + np.exp(-post_logit))  # ~= 0.105

        # --- Run Actual Function ---
        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, intuitive_config
        )

        assert np.allclose(posterior, expected_posterior, atol=1e-3)

    def test_test_update_calculation_pass(
        self, intuitive_config: BayesianConfig
    ) -> None:
        """
        Manually verify the posterior probability for a test that passes
        against a reliable code candidate.
        """
        # P(T_j) = 0.5 (Prior logit = 0)
        prior_test_probs = np.array([0.5])
        # P(C_i) = 0.9 (A reliable code)
        prior_code_probs = np.array([0.9])
        observation_matrix = np.array([[1]])  # Code passes test

        # --- Manual Calculation (from PDF) ---
        # Extract hyperparameters for clarity
        _c, _a, _b, _g = 0.9, 0.1, 0.2, 0.5

        # 1. Likelihoods (Sec 4.1)
        # P(D=1|T) = P(C) + b*(1-P(C)) = 0.9 + 0.2*(0.1) = 0.92
        like_pass_t_correct = 0.92
        # P(D=1|!T) = a*P(C) + g*(1-P(C)) = 0.1*0.9 + 0.5*(0.1) = 0.09 + 0.05 = 0.14
        like_pass_t_incorrect = 0.14

        # 2. Weight of Evidence
        woe = np.log(
            like_pass_t_correct / like_pass_t_incorrect
        )  # log(0.92 / 0.14) ~= 1.884

        # 3. Posterior Logit & Prob
        post_logit = 0.0 + woe
        expected_posterior = 1 / (1 + np.exp(-post_logit))  # ~= 0.868

        # --- Run Actual Function ---
        posterior = BayesianSystem.update_test_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, intuitive_config
        )

        assert np.allclose(posterior, expected_posterior, atol=1e-3)

    def test_multiple_observations_accumulate(
        self, intuitive_config: BayesianConfig
    ) -> None:
        """
        Test that WoE from multiple observations accumulates correctly.
        """
        prior_code_probs = np.array([0.5, 0.5])
        prior_test_probs = np.array([0.9, 0.2])  # Reliable and unreliable test

        # Code 0: Pass reliable (T0), Fail unreliable (T1)
        # Code 1: Fail reliable (T0), Pass unreliable (T1)
        observation_matrix = np.array([[1, 0], [0, 1]])

        # Calculate WoE values manually (for documentation)
        # WoE for T0 (t=0.9, a=0.1, b=0.2, g=0.5): [Fail: -2.146, Pass: 1.374]
        # WoE for T1 (t=0.2): [Fail: 0.251, Pass: -0.452]

        # Calculate WoE values for T1 (t=0.2) to build expected result
        # t1 = 0.2, a = 0.1, b = 0.2, g = 0.5
        # t1 = 0.2
        # For documentation: like_pass_c_correct_t1 = 0.28, like_pass_c_incorrect_t1 = 0.44
        # This gives WoE_pass_t1 ~= -0.452 (not needed in calculation)

        # like_fail_c_correct_t1 = (1-a)*(1-t1) = 0.9 * 0.8 = 0.72
        # like_fail_c_incorrect_t1 = (1-b)*t1 + (1-g)*(1-t1) = 0.8*0.2 + 0.5*0.8 = 0.56
        # WoE_fail_t1 = log(0.72/0.56) ~= 0.251

        # Expected logits (using calculated WoE values from above)
        # Code 0: 0.0 + woe_pass_t0 + woe_fail_t1 = 0.0 + 1.374 + 0.251 = 1.625
        # Code 1: 0.0 + woe_fail_t0 + woe_pass_t1 = 0.0 + (-2.146) + (-0.452) = -2.598

        expected_post_logits = np.array([1.625, -2.598])
        expected_posteriors = 1 / (1 + np.exp(-expected_post_logits))

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, intuitive_config
        )

        assert np.allclose(posterior, expected_posteriors, atol=1e-3)


class TestBeliefDynamics:
    """
    Tests the complex belief dynamics described in Section 8 of the PDF,
    including 'paradoxical' updates where pass=bad and fail=good.

    These tests verify that the system correctly models counter-intuitive
    scenarios that can arise with certain hyperparameter configurations.
    """

    @pytest.fixture
    def paradoxical_config(self) -> BayesianConfig:
        """
        A config designed to trigger non-intuitive updates.

        Based on Sec 8.2, we need t(1-a-b+g) < g-a.
        Let's pick: a=0.1, b=0.9, g=0.8.
        - (1-a-b+g) = 1 - 0.1 - 0.9 + 0.8 = 0.8
        - (g-a) = 0.8 - 0.1 = 0.7
        - Condition becomes: t * 0.8 < 0.7  =>  t < 0.875

        So, any test with P(T_j) < 0.875 will trigger paradoxical behavior.
        """
        return BayesianConfig(alpha=0.1, beta=0.9, gamma=0.8, learning_rate=1.0)

    def test_paradoxical_pass_decreases_belief(
        self, paradoxical_config: BayesianConfig
    ) -> None:
        """
        Tests that a 'pass' can decrease belief under the right
        hyperparameter and interactor (test) probability conditions.

        This validates the counter-intuitive case discussed in PDF Section 8.2.
        """
        prior_code_probs = np.array([0.5])
        # Use an unreliable test (t=0.5), which satisfies t < 0.875
        prior_test_probs = np.array([0.5])
        observation_matrix = np.array([[1]])  # Code *passes* the test

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, paradoxical_config
        )

        # Belief should *decrease* from 0.5
        assert posterior[0] < 0.5

        # Verify the WoE is indeed negative for a pass
        t, a, b, g = 0.5, 0.1, 0.9, 0.8
        like_pass_correct = t + a * (1 - t)  # 0.5 + 0.1*0.5 = 0.55
        like_pass_incorrect = b * t + g * (1 - t)  # 0.9*0.5 + 0.8*0.5 = 0.85
        woe_pass = np.log(like_pass_correct / like_pass_incorrect)
        assert woe_pass < 0  # Negative WoE for a pass!

    def test_paradoxical_fail_increases_belief(
        self, paradoxical_config: BayesianConfig
    ) -> None:
        """
        Tests that a 'fail' can increase belief under the same
        paradoxical conditions.

        This is the complementary case to the paradoxical pass.
        """
        prior_code_probs = np.array([0.5])
        # Use the same unreliable test (t=0.5)
        prior_test_probs = np.array([0.5])
        observation_matrix = np.array([[0]])  # Code *fails* the test

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, prior_test_probs, observation_matrix, paradoxical_config
        )

        # Belief should *increase* from 0.5
        assert posterior[0] > 0.5

        # Verify the WoE is indeed positive for a fail
        t, a, b, g = 0.5, 0.1, 0.9, 0.8
        like_fail_correct = (1 - a) * (1 - t)  # 0.9 * 0.5 = 0.45
        like_fail_incorrect = (1 - b) * t + (1 - g) * (
            1 - t
        )  # 0.1*0.5 + 0.2*0.5 = 0.15
        woe_fail = np.log(like_fail_correct / like_fail_incorrect)
        assert woe_fail > 0  # Positive WoE for a fail!

    def test_intuitive_behavior_with_reliable_test(self) -> None:
        """
        Test that 'intuitive' hyperparameters + reliable interactors
        produce intuitive belief changes:
        - Pass → Belief increases
        - Fail → Belief decreases
        """
        # Config where pass=good, fail=bad
        config = BayesianConfig(alpha=0.01, beta=0.01, gamma=0.01, learning_rate=1.0)

        prior_code_prob = np.array([0.5])
        reliable_test_prob = np.array([0.99])  # P(T) is very high

        # --- Test Pass ---
        obs_pass = np.array([[1]])
        post_pass = BayesianSystem.update_code_beliefs(
            prior_code_prob, reliable_test_prob, obs_pass, config
        )
        assert post_pass[0] > 0.5

        # --- Test Fail ---
        obs_fail = np.array([[0]])
        post_fail = BayesianSystem.update_code_beliefs(
            prior_code_prob, reliable_test_prob, obs_fail, config
        )
        assert post_fail[0] < 0.5

    def test_paradoxical_boundary_condition(
        self, paradoxical_config: BayesianConfig
    ) -> None:
        """
        Test the boundary of the paradoxical condition: t ≈ 0.875.

        At t = 0.875, we're right at the boundary where the paradoxical
        effect should be minimal.
        """
        prior_code_probs = np.array([0.5])
        # Test with t just below the boundary
        boundary_test_prob = np.array([0.87])
        observation_matrix = np.array([[1]])

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, boundary_test_prob, observation_matrix, paradoxical_config
        )

        # Should still show paradoxical behavior (decrease), but less pronounced
        assert posterior[0] < 0.5
        # Should be closer to 0.5 than with t=0.5
        assert posterior[0] > 0.4

    def test_highly_reliable_test_overcomes_paradox(
        self, paradoxical_config: BayesianConfig
    ) -> None:
        """
        Test that a highly reliable test (t > 0.875) produces normal behavior
        even with paradoxical hyperparameters.
        """
        prior_code_probs = np.array([0.5])
        # Test with t above the boundary
        reliable_test_prob = np.array([0.95])
        observation_matrix = np.array([[1]])  # Pass

        posterior = BayesianSystem.update_code_beliefs(
            prior_code_probs, reliable_test_prob, observation_matrix, paradoxical_config
        )

        # Should show normal behavior (increase for a pass)
        assert posterior[0] > 0.5
