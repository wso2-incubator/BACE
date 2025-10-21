"""
Bayesian belief updating for coevolutionary algorithms.

This module implements Bayesian belief updating for code-test coevolution,
where both code and test populations evolve simultaneously with mutual
evaluation and belief updates.

The public-facing functions operate in probability space for user convenience,
while internal calculations are performed in log-odds space for numerical stability.
"""

from typing import Tuple

import numpy as np


class CoevolutionConfig:
    """Configuration parameters for coevolutionary algorithm with Bayesian updates."""

    def __init__(
        self,
        initial_code_population_size: int = 10,
        initial_test_population_size: int = 20,
        c0_prior: float = 0.5,  # Prior probability of code being correct
        t0_prior: float = 0.5,  # Prior probability of test being correct
        alpha: float = 0.1,  # P(pass | code correct, test incorrect)
        beta: float = 0.2,  # P(pass | code incorrect, test correct)
        gamma: float = 0.5,  # P(pass | code incorrect, test incorrect)
    ):
        """
        Initialize coevolution configuration.

        Args:
            initial_code_population_size: Size of initial code population
            initial_test_population_size: Size of initial test population
            c0_prior: Prior probability that a code is correct
            t0_prior: Prior probability that a test is correct
            alpha: Hyperparameter for a correct code passing an incorrect test.
            beta: Hyperparameter for an incorrect code passing a correct test.
            gamma: Hyperparameter for an incorrect code passing an incorrect test.
        """
        if not (0 < c0_prior < 1):
            raise ValueError("c0_prior must be between 0 and 1")
        if not (0 < t0_prior < 1):
            raise ValueError("t0_prior must be between 0 and 1")
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        if not (0 <= beta <= 1):
            raise ValueError("beta must be between 0 and 1")
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be between 0 and 1")

        self.initial_code_population_size = initial_code_population_size
        self.initial_test_population_size = initial_test_population_size
        self.c0_prior = c0_prior
        self.t0_prior = t0_prior
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


# --- Internal Helper Functions for Log-Odds Conversion ---


def _probabilities_to_log_odds_array(probabilities: np.ndarray) -> np.ndarray:
    """
    Convert array of probabilities to log-odds.
    Clips values to avoid log(0) or division by zero.

    Args:
        probabilities: Array of probability values

    Returns:
        Array of log-odds values
    """
    # Clip probabilities to avoid infinity in log-odds during calculations
    epsilon = 1e-9
    clipped_probs = np.clip(probabilities, epsilon, 1 - epsilon)
    return np.asarray(np.log(clipped_probs / (1 - clipped_probs)))


def _log_odds_array_to_probabilities(log_odds_array: np.ndarray) -> np.ndarray:
    """
    Convert array of log-odds to probabilities.

    Args:
        log_odds_array: Array of log-odds values

    Returns:
        Array of probability values
    """
    # This is the logistic (sigmoid) function
    return np.asarray(1 / (1 + np.exp(-log_odds_array)))


# --- Core Logic Functions (Internal) ---


def _calculate_woe_for_code_update(
    test_probs: np.ndarray, config: CoevolutionConfig
) -> np.ndarray:
    """
    Calculates the Weight of Evidence (WoE) matrix for updating code beliefs.

    Args:
        test_probs: Current correctness probabilities for the test population.
        config: The coevolution configuration object.

    Returns:
        A WoE matrix (num_tests x 2) for updating code beliefs.
        (slice [:,0] for fails, [:,1] for passes)
    """
    epsilon = 1e-9
    t_p = test_probs[np.newaxis, :]

    like_pass_c_correct = t_p + config.alpha * (1 - t_p)
    like_pass_c_incorrect = config.beta * t_p + config.gamma * (1 - t_p)
    like_fail_c_correct = (1 - config.alpha) * (1 - t_p)
    like_fail_c_incorrect = (1 - config.beta) * t_p + (1 - config.gamma) * (1 - t_p)

    woe_code_pass = np.log(
        (like_pass_c_correct + epsilon) / (like_pass_c_incorrect + epsilon)
    )
    woe_code_fail = np.log(
        (like_fail_c_correct + epsilon) / (like_fail_c_incorrect + epsilon)
    )

    # Transpose to get shape (num_tests, 2)
    return np.stack([woe_code_fail.squeeze(), woe_code_pass.squeeze()], axis=-1)


def _calculate_woe_for_test_update(
    code_probs: np.ndarray, config: CoevolutionConfig
) -> np.ndarray:
    """
    Calculates the Weight of Evidence (WoE) matrix for updating test beliefs.

    Args:
        code_probs: Current correctness probabilities for the code population.
        config: The coevolution configuration object.

    Returns:
        A WoE matrix (num_codes x 2) for updating test beliefs.
        (slice [:,0] for fails, [:,1] for passes)
    """
    epsilon = 1e-9
    c_p = code_probs[:, np.newaxis]

    like_pass_t_correct = c_p + config.beta * (1 - c_p)
    like_pass_t_incorrect = config.alpha * c_p + config.gamma * (1 - c_p)
    like_fail_t_correct = (1 - config.beta) * (1 - c_p)
    like_fail_t_incorrect = (1 - config.alpha) * c_p + (1 - config.gamma) * (1 - c_p)

    woe_test_pass = np.log(
        (like_pass_t_correct + epsilon) / (like_pass_t_incorrect + epsilon)
    )
    woe_test_fail = np.log(
        (like_fail_t_correct + epsilon) / (like_fail_t_incorrect + epsilon)
    )

    return np.stack([woe_test_fail.squeeze(), woe_test_pass.squeeze()], axis=-1)


def _update_code_beliefs(
    code_log_odds: np.ndarray,
    observation_matrix: np.ndarray,
    woe_for_code_update: np.ndarray,
) -> np.ndarray:
    """Update beliefs about code correctness using the pre-calculated WoE matrix."""
    # woe_for_code_update has shape (num_tests, 2)
    # observation_matrix has shape (num_codes, num_tests)
    # We want to select the woe for each test based on the observation for each code
    # The result should be a (num_codes, num_tests) matrix of relevant woes
    relevant_woes = woe_for_code_update[
        np.arange(observation_matrix.shape[1]), observation_matrix
    ]

    total_woe_per_code = np.sum(relevant_woes, axis=1)
    posterior_log_odds = code_log_odds + total_woe_per_code
    return np.asarray(posterior_log_odds)


def _update_test_beliefs(
    test_log_odds: np.ndarray,
    observation_matrix: np.ndarray,
    woe_for_test_update: np.ndarray,
) -> np.ndarray:
    """Update beliefs about test correctness using the pre-calculated WoE matrix."""
    # woe_for_test_update has shape (num_codes, 2)
    # observation_matrix has shape (num_codes, num_tests)
    # We want to select the woe for each code based on the observation for each test
    relevant_woes = woe_for_test_update[
        np.arange(observation_matrix.shape[0]), observation_matrix.T
    ].T

    total_woe_per_test = np.sum(relevant_woes, axis=0)
    posterior_log_odds = test_log_odds + total_woe_per_test
    return np.asarray(posterior_log_odds)


# --- Public-Facing API Functions ---


def initialize_populations(
    config: CoevolutionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize code and test populations with prior beliefs.

    Args:
        config: Configuration object with population sizes and priors.

    Returns:
        A tuple of (code_probabilities, test_probabilities).
    """
    code_probs = np.full(config.initial_code_population_size, config.c0_prior)
    test_probs = np.full(config.initial_test_population_size, config.t0_prior)
    return code_probs, test_probs


def run_evaluation(code_population_size: int, test_population_size: int) -> np.ndarray:
    """
    Simulates running all tests against all codes and returns the observation matrix.

    In a real scenario, this function would be replaced by the actual evaluation engine.

    Args:
        code_population_size: The number of code candidates.
        test_population_size: The number of test cases.

    Returns:
        A (code_population_size x test_population_size) matrix of observations (1 for pass, 0 for fail).
    """
    return np.random.randint(0, 2, size=(code_population_size, test_population_size))


def update_population_beliefs(
    prior_code_probs: np.ndarray,
    prior_test_probs: np.ndarray,
    observation_matrix: np.ndarray,
    config: CoevolutionConfig,
    use_intermediate_updates: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a full generation update for both code and test populations.

    Args:
        prior_code_probs: Prior correctness probabilities for the code population.
        prior_test_probs: Prior correctness probabilities for the test population.
        observation_matrix: Matrix of interactions between the two populations.
        config: The coevolution configuration object.
        use_intermediate_updates: If True, uses updated code probabilities when calculating
                                 test updates. If False, uses prior code probabilities.

    Returns:
        A tuple of the posterior correctness probabilities for codes and tests.
    """
    # 1. Pre-calculate the WoE matrices based on current probabilities for code only
    woe_for_code = _calculate_woe_for_code_update(prior_test_probs, config)

    # 2. Convert prior probabilities to log-odds for the internal update step
    prior_code_log_odds = _probabilities_to_log_odds_array(prior_code_probs)
    prior_test_log_odds = _probabilities_to_log_odds_array(prior_test_probs)

    # 3. Update beliefs in log-odds space for code only
    posterior_code_log_odds = _update_code_beliefs(
        prior_code_log_odds, observation_matrix, woe_for_code
    )

    # 4. Pre-calculate WoE for test update based on updated code probabilities
    if use_intermediate_updates:
        updated_code_probs = _log_odds_array_to_probabilities(posterior_code_log_odds)
        woe_for_test = _calculate_woe_for_test_update(updated_code_probs, config)
    else:
        woe_for_test = _calculate_woe_for_test_update(prior_code_probs, config)

    # 5. Update beliefs in log-odds space for tests
    posterior_test_log_odds = _update_test_beliefs(
        prior_test_log_odds, observation_matrix, woe_for_test
    )

    # 6. Convert posterior log-odds back to probabilities for the output
    posterior_code_probs = _log_odds_array_to_probabilities(posterior_code_log_odds)
    posterior_test_probs = _log_odds_array_to_probabilities(posterior_test_log_odds)

    return posterior_code_probs, posterior_test_probs
