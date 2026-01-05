"""
Bayesian belief system for coevolutionary algorithms.

This module implements the IBayesianSystem interface from the core module,
providing Bayesian belief initialization and updating for code-test coevolution.

The implementation operates in probability space for the public API,
while internal calculations are performed in log-odds space for
numerical stability and efficiency.

Implementation based on the deprecated bayesian.py module with
architectural improvements for the protocol-based system.
"""

import numpy as np
from loguru import logger

from ..core.interfaces import BayesianConfig, IBeliefUpdater
from ..utils.logging import (
    log_belief_changes,
    log_belief_update_start,
    log_posterior_statistics,
    log_prior_statistics,
)

# --- Constants ---

# Small value added to probabilities and likelihoods to prevent numerical instabilities
# from log(0) or division by zero in log-odds and WoE calculations
_NUMERICAL_STABILITY_EPSILON = 1e-9


class BayesianSystem(IBeliefUpdater):
    """
    Static implementation of the IBayesianSystem protocol.

    This class provides static methods for initialization and updating of Bayesian beliefs
    for code and test populations in the coevolutionary framework.

    The implementation uses log-odds space internally for numerical stability
    while exposing a probability-based API for ease of use.

    All methods are static and no instantiation is required.
    """

    @staticmethod
    def initialize_beliefs(
        population_size: int,
        initial_probability: float,
    ) -> np.ndarray:
        """
        Initialize uniform prior beliefs for a population.

        Creates an array of prior correctness probabilities with all elements
        set to the same initial value.

        Args:
            population_size: Number of population members.
            initial_probability: Prior probability to assign to each member.

        Returns:
            A numpy array of shape (population_size,) filled with initial_probability.

        Raises:
            ValueError: If population_size <= 0 or initial_probability not in [0, 1].
        """
        logger.debug(
            f"Initializing {population_size} prior beliefs with "
            f"probability {initial_probability:.4f}"
        )

        if population_size <= 0:
            msg = "Population size must be a positive integer"
            logger.error(msg)
            raise ValueError(msg)

        if not (0.0 <= initial_probability <= 1.0):
            msg = "Initial probability must be in the range [0.0, 1.0]"
            logger.error(msg)
            raise ValueError(msg)

        # Adjust extreme probabilities to avoid numerical instability
        adjusted_prob = initial_probability
        if initial_probability == 0.0:
            adjusted_prob += _NUMERICAL_STABILITY_EPSILON
            logger.debug(
                "Initial probability of 0.0 adjusted to avoid numerical instability"
            )
        elif initial_probability == 1.0:
            adjusted_prob -= _NUMERICAL_STABILITY_EPSILON
            logger.debug(
                "Initial probability of 1.0 adjusted to avoid numerical instability"
            )

        beliefs = np.full(population_size, adjusted_prob, dtype=float)
        logger.trace(f"Created prior beliefs array with shape {beliefs.shape}")

        return beliefs

    # --- IBeliefUpdater Implementation ---

    @staticmethod
    def update_code_beliefs(
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        code_update_mask_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        """
        Update beliefs for the code population based on test results.
        Uses vectorized matrix operations for efficiency.
        """
        log_belief_update_start("code", len(prior_code_probs), len(prior_test_probs))
        log_prior_statistics("code", prior_code_probs)

        # 1. Calculate WoE vectors (One value per Test)
        woe_fail, woe_pass = BayesianSystem._calculate_woe_vectors_for_code_update(
            prior_test_probs, config
        )

        # 2. Convert priors to log-odds
        prior_log_odds = BayesianSystem._probabilities_to_log_odds(prior_code_probs)

        # 3. Perform vectorized update using mask and observations
        # We treat observations (0/1) as selectors for WoE values
        posterior_log_odds = BayesianSystem._perform_vectorized_update(
            prior_log_odds=prior_log_odds,
            observation_matrix=observation_matrix,
            mask_matrix=code_update_mask_matrix,
            woe_pass_vector=woe_pass,
            woe_fail_vector=woe_fail,
            learning_rate=config.learning_rate,
        )

        # 4. Convert back to probabilities
        posterior_code_probs = BayesianSystem._log_odds_to_probabilities(
            posterior_log_odds
        )

        log_posterior_statistics("code", prior_code_probs, posterior_code_probs)
        log_belief_changes("code", prior_code_probs, posterior_code_probs)

        return posterior_code_probs

    @staticmethod
    def update_test_beliefs(
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        test_update_mask_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        """
        Update beliefs for the test population based on code results.
        """
        log_belief_update_start("test", len(prior_test_probs), len(prior_code_probs))
        log_prior_statistics("test", prior_test_probs)

        # 1. Calculate WoE vectors (One value per Code)
        woe_fail, woe_pass = BayesianSystem._calculate_woe_vectors_for_test_update(
            prior_code_probs, config
        )

        # 2. Convert priors to log-odds
        prior_log_odds = BayesianSystem._probabilities_to_log_odds(prior_test_probs)

        # 3. Perform vectorized update
        # Note: Transpose observations because Rows=Tests, Cols=Codes for this perspective
        posterior_log_odds = BayesianSystem._perform_vectorized_update(
            prior_log_odds=prior_log_odds,
            observation_matrix=observation_matrix.T,  # Transpose for Test perspective
            mask_matrix=test_update_mask_matrix.T,  # Transpose for Test perspective
            woe_pass_vector=woe_pass,
            woe_fail_vector=woe_fail,
            learning_rate=config.learning_rate,
        )

        # 4. Convert back to probabilities
        posterior_test_probs = BayesianSystem._log_odds_to_probabilities(
            posterior_log_odds
        )

        log_posterior_statistics("test", prior_test_probs, posterior_test_probs)
        log_belief_changes("test", prior_test_probs, posterior_test_probs)

        return posterior_test_probs

    # --- Internal Helper Methods ---

    @staticmethod
    def _probabilities_to_log_odds(probabilities: np.ndarray) -> np.ndarray:
        """Convert probabilities to log-odds with clipping."""
        clipped = np.clip(
            probabilities,
            _NUMERICAL_STABILITY_EPSILON,
            1 - _NUMERICAL_STABILITY_EPSILON,
        )
        return np.asarray(np.log(clipped / (1 - clipped)))

    @staticmethod
    def _log_odds_to_probabilities(log_odds: np.ndarray) -> np.ndarray:
        """Convert log-odds to probabilities using sigmoid."""
        return np.asarray(1 / (1 + np.exp(-log_odds)))

    @staticmethod
    def _perform_vectorized_update(
        prior_log_odds: np.ndarray,
        observation_matrix: np.ndarray,
        mask_matrix: np.ndarray,
        woe_pass_vector: np.ndarray,
        woe_fail_vector: np.ndarray,
        learning_rate: float,
    ) -> np.ndarray:
        """
        Core optimized update logic using matrix multiplication.

        Mathematical Logic:
        Update = LearningRate * Sum_over_j( Mask_ij * [ Obs_ij * WoE_Pass_j + (1-Obs_ij) * WoE_Fail_j ] )

        Args:
            prior_log_odds: (N,) Initial beliefs in log-odds.
            observation_matrix: (N, M) Binary matrix of outcomes.
            mask_matrix: (N, M) Boolean/Binary matrix indicating valid updates.
            woe_pass_vector: (M,) WoE values if outcome is Pass (1).
            woe_fail_vector: (M,) WoE values if outcome is Fail (0).
            learning_rate: Scaling factor.

        Returns:
            (N,) Updated posterior log-odds.
        """
        # Ensure mask is in in for multiplication
        mask_matrix = mask_matrix.astype(int)

        # 1. Calculate contribution from Passing interactions
        # logic: (Mask * Obs) gives 1 only where valid update AND Pass occurred
        pass_interactions = mask_matrix * observation_matrix
        pass_update = pass_interactions @ woe_pass_vector

        # 2. Calculate contribution from Failing interactions
        # logic: (Mask * (1-Obs)) gives 1 only where valid update AND Fail occurred
        fail_interactions = mask_matrix * (1 - observation_matrix)
        fail_update = fail_interactions @ woe_fail_vector

        # 3. Combine
        total_woe_update = pass_update + fail_update
        updated_log_odds = prior_log_odds + (learning_rate * total_woe_update)

        return np.asarray(updated_log_odds)

    @staticmethod
    def _calculate_woe_vectors_for_code_update(
        test_probs: np.ndarray, config: BayesianConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate WoE vectors for code updates.
        Returns (woe_fail_vector, woe_pass_vector).
        """
        t_p = test_probs  # Shape (num_tests,)

        # Likelihoods P(D|C)
        # Pass | Correct / Incorrect
        like_pass_c_corr = t_p + config.alpha * (1 - t_p)
        like_pass_c_inc = config.beta * t_p + config.gamma * (1 - t_p)

        # Fail | Correct / Incorrect
        like_fail_c_corr = (1 - config.alpha) * (1 - t_p)
        like_fail_c_inc = (1 - config.beta) * t_p + (1 - config.gamma) * (1 - t_p)

        # WoE = log(Likelihood_Correct / Likelihood_Incorrect)
        woe_pass = np.log(
            (like_pass_c_corr + _NUMERICAL_STABILITY_EPSILON)
            / (like_pass_c_inc + _NUMERICAL_STABILITY_EPSILON)
        )
        woe_fail = np.log(
            (like_fail_c_corr + _NUMERICAL_STABILITY_EPSILON)
            / (like_fail_c_inc + _NUMERICAL_STABILITY_EPSILON)
        )

        return woe_fail, woe_pass

    @staticmethod
    def _calculate_woe_vectors_for_test_update(
        code_probs: np.ndarray, config: BayesianConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate WoE vectors for test updates.
        Returns (woe_fail_vector, woe_pass_vector).
        """
        c_p = code_probs  # Shape (num_codes,)

        # Likelihoods P(D|T)
        # Pass | Correct / Incorrect
        like_pass_t_corr = c_p + config.beta * (1 - c_p)
        like_pass_t_inc = config.alpha * c_p + config.gamma * (1 - c_p)

        # Fail | Correct / Incorrect
        like_fail_t_corr = (1 - config.beta) * (1 - c_p)
        like_fail_t_inc = (1 - config.alpha) * c_p + (1 - config.gamma) * (1 - c_p)

        woe_pass = np.log(
            (like_pass_t_corr + _NUMERICAL_STABILITY_EPSILON)
            / (like_pass_t_inc + _NUMERICAL_STABILITY_EPSILON)
        )
        woe_fail = np.log(
            (like_fail_t_corr + _NUMERICAL_STABILITY_EPSILON)
            / (like_fail_t_inc + _NUMERICAL_STABILITY_EPSILON)
        )

        return woe_fail, woe_pass
