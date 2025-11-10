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

from .core.interfaces import BayesianConfig, IBayesianSystem
from .logging_utils import (
    log_belief_changes,
    log_belief_update_start,
    log_posterior_statistics,
    log_prior_statistics,
)

# --- Constants ---

# Small value added to probabilities and likelihoods to prevent numerical instabilities
# from log(0) or division by zero in log-odds and WoE calculations
_NUMERICAL_STABILITY_EPSILON = 1e-9


class BayesianSystem(IBayesianSystem):
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
        config: BayesianConfig,
    ) -> np.ndarray:
        """
        Update beliefs for the code population based on test results.

        Args:
            prior_code_probs: Prior probabilities for the code population.
            prior_test_probs: Prior probabilities for the test population.
            observation_matrix: Matrix of interactions (rows=code, cols=tests).
                               Values are 1 for pass, 0 for fail.
            config: BayesianConfig containing hyperparameters
                   (alpha, beta, gamma, learning_rate).

        Returns:
            Updated posterior probabilities for the code population.
        """
        log_belief_update_start("code", len(prior_code_probs), len(prior_test_probs))
        log_prior_statistics("code", prior_code_probs)

        # Calculate WoE matrix for code update
        logger.trace("Calculating WoE matrix for code updates")
        woe_for_code = BayesianSystem._calculate_woe_for_code_update(
            prior_test_probs, config
        )

        # Convert to log-odds space
        logger.trace("Converting prior code probabilities to log-odds")
        prior_code_log_odds = BayesianSystem._probabilities_to_log_odds(
            prior_code_probs
        )

        # Update in log-odds space
        logger.debug("Updating code beliefs in log-odds space")
        posterior_code_log_odds = BayesianSystem._update_log_odds(
            prior_code_log_odds, observation_matrix, woe_for_code, config.learning_rate
        )

        # Convert back to probabilities
        logger.trace("Converting posterior log-odds to probabilities")
        posterior_code_probs = BayesianSystem._log_odds_to_probabilities(
            posterior_code_log_odds
        )

        log_posterior_statistics("code", prior_code_probs, posterior_code_probs)
        log_belief_changes("code", prior_code_probs, posterior_code_probs)

        return posterior_code_probs

    @staticmethod
    def update_test_beliefs(
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        """
        Update beliefs for the test population based on code results.

        Args:
            prior_code_probs: Prior probabilities for the code population.
            prior_test_probs: Prior probabilities for the test population.
            observation_matrix: Matrix of interactions (rows=code, cols=tests).
                               Values are 1 for pass, 0 for fail.
            config: BayesianConfig containing hyperparameters
                   (alpha, beta, gamma, learning_rate).

        Returns:
            Updated posterior probabilities for the test population.
        """
        log_belief_update_start("test", len(prior_test_probs), len(prior_code_probs))
        log_prior_statistics("test", prior_test_probs)

        # Calculate WoE matrix for test update
        logger.trace("Calculating WoE matrix for test updates")
        woe_for_test = BayesianSystem._calculate_woe_for_test_update(
            prior_code_probs, config
        )

        # Convert to log-odds space
        logger.trace("Converting prior test probabilities to log-odds")
        prior_test_log_odds = BayesianSystem._probabilities_to_log_odds(
            prior_test_probs
        )

        # Update in log-odds space (need to transpose observation matrix for tests)
        logger.debug("Updating test beliefs in log-odds space")
        posterior_test_log_odds = BayesianSystem._update_log_odds(
            prior_test_log_odds,
            observation_matrix.T,  # Transpose: rows become tests, cols become codes
            woe_for_test,
            config.learning_rate,
        )

        # Convert back to probabilities
        logger.trace("Converting posterior log-odds to probabilities")
        posterior_test_probs = BayesianSystem._log_odds_to_probabilities(
            posterior_test_log_odds
        )

        log_posterior_statistics("test", prior_test_probs, posterior_test_probs)
        log_belief_changes("test", prior_test_probs, posterior_test_probs)

        return posterior_test_probs

    # --- Internal Helper Methods ---

    @staticmethod
    def _probabilities_to_log_odds(probabilities: np.ndarray) -> np.ndarray:
        """
        Convert array of probabilities to log-odds.

        Clips values to avoid log(0) or division by zero.

        Args:
            probabilities: Array of probability values.

        Returns:
            Array of log-odds values.
        """
        # Clip probabilities to avoid infinity in log-odds
        clipped_probs = np.clip(
            probabilities,
            _NUMERICAL_STABILITY_EPSILON,
            1 - _NUMERICAL_STABILITY_EPSILON,
        )
        return np.asarray(np.log(clipped_probs / (1 - clipped_probs)))

    @staticmethod
    def _log_odds_to_probabilities(log_odds_array: np.ndarray) -> np.ndarray:
        """
        Convert array of log-odds to probabilities.

        Uses the logistic (sigmoid) function.

        Args:
            log_odds_array: Array of log-odds values.

        Returns:
            Array of probability values.
        """
        return np.asarray(1 / (1 + np.exp(-log_odds_array)))

    @staticmethod
    def _calculate_woe_for_code_update(
        test_probs: np.ndarray, config: BayesianConfig
    ) -> np.ndarray:
        """
        Calculate the Weight of Evidence (WoE) matrix for updating code beliefs.

        Corresponds to Equations 16 and 17 in the coevolution documentation.

        Args:
            test_probs: Current correctness probabilities P(T_j) for tests.
            config: BayesianConfig containing alpha, beta, gamma.

        Returns:
            WoE matrix of shape (num_tests, 2) for updating code beliefs.
            Last dimension contains [WoE for fail (0), WoE for pass (1)].
        """
        t_p = test_probs[np.newaxis, :]  # Shape (1, num_tests)

        # Likelihood P(D=1 | C_i=1) and P(D=1 | C_i=0)
        like_pass_c_correct = t_p + config.alpha * (1 - t_p)
        like_pass_c_incorrect = config.beta * t_p + config.gamma * (1 - t_p)

        # Likelihood P(D=0 | C_i=1) and P(D=0 | C_i=0)
        like_fail_c_correct = (1 - config.alpha) * (1 - t_p)
        like_fail_c_incorrect = (1 - config.beta) * t_p + (1 - config.gamma) * (1 - t_p)

        # WoE = log(Likelihood(Correct) / Likelihood(Incorrect))
        # Add epsilon inside log to prevent log(0)
        woe_code_pass = np.log(
            (like_pass_c_correct + _NUMERICAL_STABILITY_EPSILON)
            / (like_pass_c_incorrect + _NUMERICAL_STABILITY_EPSILON)
        )
        woe_code_fail = np.log(
            (like_fail_c_correct + _NUMERICAL_STABILITY_EPSILON)
            / (like_fail_c_incorrect + _NUMERICAL_STABILITY_EPSILON)
        )

        # Stack fail WoE (index 0) and pass WoE (index 1) along last axis
        # Use reshape instead of squeeze to preserve shape even with single test
        woe_code_fail_flat = woe_code_fail.ravel()
        woe_code_pass_flat = woe_code_pass.ravel()
        return np.stack([woe_code_fail_flat, woe_code_pass_flat], axis=-1)

    @staticmethod
    def _calculate_woe_for_test_update(
        code_probs: np.ndarray, config: BayesianConfig
    ) -> np.ndarray:
        """
        Calculate the Weight of Evidence (WoE) matrix for updating test beliefs.

        Corresponds to Equations 18 and 19 in the coevolution documentation.

        Args:
            code_probs: Current correctness probabilities P(C_i) for codes.
            config: BayesianConfig containing alpha, beta, gamma.

        Returns:
            WoE matrix of shape (num_codes, 2) for updating test beliefs.
            Last dimension contains [WoE for fail (0), WoE for pass (1)].
        """
        c_p = code_probs[:, np.newaxis]  # Shape (num_codes, 1)

        # Likelihood P(D=1 | T_j=1) and P(D=1 | T_j=0)
        like_pass_t_correct = c_p + config.beta * (1 - c_p)
        like_pass_t_incorrect = config.alpha * c_p + config.gamma * (1 - c_p)

        # Likelihood P(D=0 | T_j=1) and P(D=0 | T_j=0)
        like_fail_t_correct = (1 - config.beta) * (1 - c_p)
        like_fail_t_incorrect = (1 - config.alpha) * c_p + (1 - config.gamma) * (
            1 - c_p
        )

        # WoE = log(Likelihood(Correct) / Likelihood(Incorrect))
        woe_test_pass = np.log(
            (like_pass_t_correct + _NUMERICAL_STABILITY_EPSILON)
            / (like_pass_t_incorrect + _NUMERICAL_STABILITY_EPSILON)
        )
        woe_test_fail = np.log(
            (like_fail_t_correct + _NUMERICAL_STABILITY_EPSILON)
            / (like_fail_t_incorrect + _NUMERICAL_STABILITY_EPSILON)
        )

        # Stack fail WoE (index 0) and pass WoE (index 1) along last axis
        # Use reshape instead of squeeze to preserve shape even with single code
        woe_test_fail_flat = woe_test_fail.ravel()
        woe_test_pass_flat = woe_test_pass.ravel()
        return np.stack([woe_test_fail_flat, woe_test_pass_flat], axis=-1)

    @staticmethod
    def _update_log_odds(
        prior_log_odds: np.ndarray,
        observation_matrix: np.ndarray,
        woe_matrix: np.ndarray,
        learning_rate: float,
    ) -> np.ndarray:
        """
        Update log-odds using the WoE matrix, scaled by learning rate.

        This is a generic update function that works for both code and test updates.
        The observation_matrix should be oriented such that rows correspond to
        the population being updated.

        Args:
            prior_log_odds: Prior log-odds for the population being updated.
            observation_matrix: Matrix of observations (0 or 1), with rows
                               corresponding to the population being updated.
            woe_matrix: Weight of Evidence matrix with shape (num_observations, 2).
            learning_rate: Scaling factor for the Bayesian update.

        Returns:
            Posterior log-odds for the population.
        """
        # Expand observation matrix to index into WoE matrix
        indices = observation_matrix[..., np.newaxis]
        woe_matrix_expanded = woe_matrix[np.newaxis, :, :]

        # Extract relevant WoE values based on observations
        relevant_woes = np.take_along_axis(
            woe_matrix_expanded, indices, axis=2
        ).squeeze(-1)

        # Sum WoE across all observations for each individual
        total_woe = np.sum(relevant_woes, axis=1)

        # Apply Bayesian update scaled by learning rate
        posterior_log_odds = prior_log_odds + learning_rate * total_woe

        return np.asarray(posterior_log_odds, dtype=float)
