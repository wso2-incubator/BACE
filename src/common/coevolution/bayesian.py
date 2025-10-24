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
from loguru import logger

from .config import CoevolutionConfig

# --- Constants ---

# Small value added to probabilities and likelihoods to prevent numerical instabilities
# from log(0) or division by zero in log-odds and WoE calculations
_NUMERICAL_STABILITY_EPSILON = 1e-9

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
    clipped_probs = np.clip(
        probabilities, _NUMERICAL_STABILITY_EPSILON, 1 - _NUMERICAL_STABILITY_EPSILON
    )
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
    Corresponds to Equations 16 and 17 in the documentation.

    Args:
        test_probs: Current correctness probabilities P(T_j) for the test population.
        config: The coevolution configuration object containing alpha, beta, gamma.

    Returns:
        A WoE matrix (num_tests x 2) for updating code beliefs.
        The last dimension contains [WoE for observation 0 (Fail), WoE for observation 1 (Pass)].
    """
    t_p = test_probs[np.newaxis, :]  # Shape (1, num_tests)

    # Likelihood P(D=1 | C_i=1) and P(D=1 | C_i=0)
    like_pass_c_correct = t_p + config.alpha * (1 - t_p)
    like_pass_c_incorrect = config.beta * t_p + config.gamma * (1 - t_p)

    # Likelihood P(D=0 | C_i=1) and P(D=0 | C_i=0)
    like_fail_c_correct = (1 - config.alpha) * (1 - t_p)
    like_fail_c_incorrect = (1 - config.beta) * t_p + (1 - config.gamma) * (1 - t_p)

    # WoE = log( Likelihood(Correct) / Likelihood(Incorrect) )
    # Adding epsilon inside log to prevent log(0) if likelihoods become zero
    woe_code_pass = np.log(
        (like_pass_c_correct + _NUMERICAL_STABILITY_EPSILON)
        / (like_pass_c_incorrect + _NUMERICAL_STABILITY_EPSILON)
    )
    woe_code_fail = np.log(
        (like_fail_c_correct + _NUMERICAL_STABILITY_EPSILON)
        / (like_fail_c_incorrect + _NUMERICAL_STABILITY_EPSILON)
    )

    # Stack fail WoE (index 0) and pass WoE (index 1) along the last axis
    # Squeeze removes the first dimension of size 1, resulting in (num_tests, 2)
    return np.stack([woe_code_fail.squeeze(), woe_code_pass.squeeze()], axis=-1)


def _calculate_woe_for_test_update(
    code_probs: np.ndarray, config: CoevolutionConfig
) -> np.ndarray:
    """
    Calculates the Weight of Evidence (WoE) matrix for updating test beliefs.
    Corresponds to Equations 18 and 19 in the documentation.

    Args:
        code_probs: Current correctness probabilities P(C_i) for the code population.
        config: The coevolution configuration object containing alpha, beta, gamma.

    Returns:
        A WoE matrix (num_codes x 2) for updating test beliefs.
        The last dimension contains [WoE for observation 0 (Fail), WoE for observation 1 (Pass)].
    """
    c_p = code_probs[:, np.newaxis]  # Shape (num_codes, 1)

    # Likelihood P(D=1 | T_j=1) and P(D=1 | T_j=0)
    like_pass_t_correct = c_p + config.beta * (1 - c_p)
    like_pass_t_incorrect = config.alpha * c_p + config.gamma * (1 - c_p)

    # Likelihood P(D=0 | T_j=1) and P(D=0 | T_j=0)
    like_fail_t_correct = (1 - config.beta) * (1 - c_p)
    like_fail_t_incorrect = (1 - config.alpha) * c_p + (1 - config.gamma) * (1 - c_p)

    # WoE = log( Likelihood(Correct) / Likelihood(Incorrect) )
    woe_test_pass = np.log(
        (like_pass_t_correct + _NUMERICAL_STABILITY_EPSILON)
        / (like_pass_t_incorrect + _NUMERICAL_STABILITY_EPSILON)
    )
    woe_test_fail = np.log(
        (like_fail_t_correct + _NUMERICAL_STABILITY_EPSILON)
        / (like_fail_t_incorrect + _NUMERICAL_STABILITY_EPSILON)
    )

    # Stack fail WoE (index 0) and pass WoE (index 1) along the last axis
    # Squeeze removes the second dimension of size 1, resulting in (num_codes, 2)
    return np.stack([woe_test_fail.squeeze(), woe_test_pass.squeeze()], axis=-1)


def _update_code_beliefs(
    code_log_odds: np.ndarray,
    observation_matrix: np.ndarray,
    woe_for_code_update: np.ndarray,
    config: CoevolutionConfig,  # Pass config for learning rate
) -> np.ndarray:
    """
    Update beliefs about code correctness using the pre-calculated WoE matrix,
    scaled by the learning rate. Implements Equation 14 adjusted for learning rate.

    Args:
        code_log_odds: The prior log-odds for each code candidate.
        observation_matrix: The (num_codes x num_tests) matrix of results (0 or 1).
        woe_for_code_update: The (num_tests x 2) WoE matrix for updating codes.
        config: The coevolution configuration object.

    Returns:
        The posterior log-odds for each code candidate.
    """
    indices = observation_matrix[..., np.newaxis]
    woe_matrix_expanded = woe_for_code_update[np.newaxis, :, :]
    relevant_woes = np.take_along_axis(woe_matrix_expanded, indices, axis=2).squeeze(-1)

    total_woe_per_code = np.sum(relevant_woes, axis=1)

    # Apply the Bayesian update, scaled by learning rate
    posterior_log_odds = code_log_odds + config.learning_rate * total_woe_per_code
    return np.asarray(posterior_log_odds, dtype=float)


def _update_test_beliefs(
    test_log_odds: np.ndarray,
    observation_matrix: np.ndarray,
    woe_for_test_update: np.ndarray,
    config: CoevolutionConfig,  # Pass config for learning rate
) -> np.ndarray:
    """
    Update beliefs about test correctness using the pre-calculated WoE matrix,
    scaled by the learning rate. Implements Equation 15 adjusted for learning rate.

    Args:
        test_log_odds: The prior log-odds for each test case.
        observation_matrix: The (num_codes x num_tests) matrix of results (0 or 1).
        woe_for_test_update: The (num_codes x 2) WoE matrix for updating tests.
        config: The coevolution configuration object.

    Returns:
        The posterior log-odds for each test case.
    """
    indices = observation_matrix[..., np.newaxis]
    woe_matrix_expanded = woe_for_test_update[:, np.newaxis, :]
    relevant_woes = np.take_along_axis(woe_matrix_expanded, indices, axis=2).squeeze(-1)

    total_woe_per_test = np.sum(relevant_woes, axis=0)

    # Apply the Bayesian update, scaled by learning rate
    posterior_log_odds = test_log_odds + config.learning_rate * total_woe_per_test
    return np.asarray(posterior_log_odds, dtype=float)


# --- Public-Facing API Functions ---


def initialize_prior_beliefs(
    size: int,
    prior: float,
) -> np.ndarray:
    """
    Create an array of prior correctness probabilities.

    Returns a 1-D numpy array of length `size` with every element set to `prior`.
    This is a convenience helper for initializing uniform prior beliefs for a
    population of members (code or tests).

    Args:
        size: Number of population members (length of the returned array).
        prior: Prior probability to assign to each member.

    Returns:
        A numpy.ndarray of shape (size,) and dtype float filled with `prior`.
    """
    logger.debug(f"Initializing {size} prior beliefs with probability {prior:.4f}")

    if not (0.0 < prior < 1.0):
        raise ValueError("Prior probability must be in the range (0.0, 1.0)")
    if size <= 0:
        raise ValueError("Size must be a positive integer")

    beliefs = np.full(size, prior, dtype=float)
    logger.trace(f"Created prior beliefs array with shape {beliefs.shape}")

    return beliefs


def update_population_beliefs(
    prior_code_probs: np.ndarray,
    prior_test_probs: np.ndarray,
    observation_matrix: np.ndarray,
    config: CoevolutionConfig,
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
    logger.info(
        f"Updating beliefs: {len(prior_code_probs)} codes × "
        f"{len(prior_test_probs)} tests"
    )

    # Log prior statistics
    prior_code_mean = np.mean(prior_code_probs)
    prior_code_std = np.std(prior_code_probs)
    prior_test_mean = np.mean(prior_test_probs)
    prior_test_std = np.std(prior_test_probs)

    logger.debug(
        f"Prior code beliefs: mean={prior_code_mean:.4f}, std={prior_code_std:.4f}, "
        f"min={np.min(prior_code_probs):.4f}, max={np.max(prior_code_probs):.4f}"
    )
    logger.debug(
        f"Prior test beliefs: mean={prior_test_mean:.4f}, std={prior_test_std:.4f}, "
        f"min={np.min(prior_test_probs):.4f}, max={np.max(prior_test_probs):.4f}"
    )
    logger.debug(
        f"Learning rate: {config.learning_rate}, "
        f"Intermediate updates: {config.use_intermediate_updates}"
    )

    # 1. Pre-calculate the WoE matrices based on current probabilities for code only
    logger.trace("Calculating WoE matrix for code updates")
    woe_for_code = _calculate_woe_for_code_update(prior_test_probs, config)

    # 2. Convert prior probabilities to log-odds for the internal update step
    logger.trace("Converting prior probabilities to log-odds")
    prior_code_log_odds = _probabilities_to_log_odds_array(prior_code_probs)
    prior_test_log_odds = _probabilities_to_log_odds_array(prior_test_probs)

    # 3. Update beliefs in log-odds space for code only
    logger.debug("Updating code beliefs in log-odds space")
    posterior_code_log_odds = _update_code_beliefs(
        prior_code_log_odds, observation_matrix, woe_for_code, config
    )

    # 4. Pre-calculate WoE for test update based on updated code probabilities
    if config.use_intermediate_updates:
        logger.trace("Using intermediate code updates for test WoE calculation")
        updated_code_probs = _log_odds_array_to_probabilities(posterior_code_log_odds)
        woe_for_test = _calculate_woe_for_test_update(updated_code_probs, config)
    else:
        logger.trace("Using prior code probabilities for test WoE calculation")
        woe_for_test = _calculate_woe_for_test_update(prior_code_probs, config)

    # 5. Update beliefs in log-odds space for tests
    logger.debug("Updating test beliefs in log-odds space")
    posterior_test_log_odds = _update_test_beliefs(
        prior_test_log_odds, observation_matrix, woe_for_test, config
    )

    # 6. Convert posterior log-odds back to probabilities for the output
    logger.trace("Converting posterior log-odds to probabilities")
    posterior_code_probs = _log_odds_array_to_probabilities(posterior_code_log_odds)
    posterior_test_probs = _log_odds_array_to_probabilities(posterior_test_log_odds)

    # Log posterior statistics and belief changes
    posterior_code_mean = np.mean(posterior_code_probs)
    posterior_code_std = np.std(posterior_code_probs)
    posterior_test_mean = np.mean(posterior_test_probs)
    posterior_test_std = np.std(posterior_test_probs)

    code_belief_delta = posterior_code_mean - prior_code_mean
    test_belief_delta = posterior_test_mean - prior_test_mean

    logger.debug(
        f"Posterior code beliefs: mean={posterior_code_mean:.4f}, std={posterior_code_std:.4f}, "
        f"min={np.min(posterior_code_probs):.4f}, max={np.max(posterior_code_probs):.4f}"
    )
    logger.debug(
        f"Posterior test beliefs: mean={posterior_test_mean:.4f}, std={posterior_test_std:.4f}, "
        f"min={np.min(posterior_test_probs):.4f}, max={np.max(posterior_test_probs):.4f}"
    )
    logger.info(
        f"Belief update complete: code Δ={code_belief_delta:+.4f}, "
        f"test Δ={test_belief_delta:+.4f}"
    )

    # Log individual belief changes at trace level
    code_deltas = posterior_code_probs - prior_code_probs
    test_deltas = posterior_test_probs - prior_test_probs
    logger.trace(
        f"Code belief changes: mean={np.mean(code_deltas):+.4f}, "
        f"std={np.std(code_deltas):.4f}, "
        f"max_increase={np.max(code_deltas):+.4f}, "
        f"max_decrease={np.min(code_deltas):+.4f}"
    )
    logger.trace(
        f"Test belief changes: mean={np.mean(test_deltas):+.4f}, "
        f"std={np.std(test_deltas):.4f}, "
        f"max_increase={np.max(test_deltas):+.4f}, "
        f"max_decrease={np.min(test_deltas):+.4f}"
    )

    return posterior_code_probs, posterior_test_probs
