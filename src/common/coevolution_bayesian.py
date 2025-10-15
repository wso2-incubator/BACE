"""
Coevolution with Bayesian Updates

This module implements Bayesian belief updating for coevolutionary algorithms,
specifically designed for code-test coevolution where both code and test
populations evolve simultaneously with mutual evaluation and belief updates.

The module provides functions for:
- Converting between probabilities and log-odds
- Evaluating code-test pairs
- Bayesian posterior updates using log-odds for numerical stability
- Population-level belief updates
"""

from typing import Tuple

import numpy as np


class CoevolutionConfig:
    """Configuration parameters for coevolutionary algorithm with Bayesian updates."""

    def __init__(
        self,
        initial_code_population_size: int = 10,
        initial_test_population_size: int = 20,
        c0_prior: float = 0.6,  # Prior probability of code being correct
        t0_prior: float = 0.6,  # Prior probability of test being correct
    ):
        """
        Initialize coevolution configuration.

        Args:
            initial_code_population_size: Size of initial code population
            initial_test_population_size: Size of initial test population
            c0_prior: Prior probability that a code is correct
            t0_prior: Prior probability that a test is correct
            probability_of_pass: Probability of code-test pair passing (for simulation)
        """
        if not (0 < c0_prior < 1):
            raise ValueError("c0_prior must be between 0 and 1")
        if not (0 < t0_prior < 1):
            raise ValueError("t0_prior must be between 0 and 1")

        self.initial_code_population_size = initial_code_population_size
        self.initial_test_population_size = initial_test_population_size
        self.c0_prior = c0_prior
        self.t0_prior = t0_prior


# Type aliases for clarity
LogOdds = float
Probability = float


def probability_to_log_odds(probability: Probability) -> LogOdds:
    """
    Convert probability to log-odds.

    Args:
        probability: Probability value between 0 and 1

    Returns:
        Log-odds value

    Raises:
        ValueError: If probability is not in (0, 1)
    """
    if not (0 < probability < 1):
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")

    return float(np.log(probability / (1 - probability)))


def log_odds_to_probability(log_odds: LogOdds) -> Probability:
    """
    Convert log-odds to probability.

    Args:
        log_odds: Log-odds value

    Returns:
        Probability value between 0 and 1
    """
    odds = np.exp(log_odds)
    return float(odds / (1 + odds))


def log_odds_array_to_probabilities(log_odds_array: np.ndarray) -> np.ndarray:
    """
    Convert array of log-odds to probabilities.

    Args:
        log_odds_array: Array of log-odds values

    Returns:
        Array of probability values
    """
    odds: np.ndarray = np.exp(log_odds_array)
    probabilities: np.ndarray = odds / (1 + odds)
    return probabilities.astype(np.float64)


def initialize_populations(
    config: CoevolutionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize code and test populations with prior beliefs.

    Args:
        config: Configuration object with population sizes and priors

    Returns:
        Tuple of (code_log_odds, test_log_odds)
    """

    # Convert to log-odds for numerical stability
    code_log_odds = np.full(
        config.initial_code_population_size, probability_to_log_odds(config.c0_prior)
    )
    test_log_odds = np.full(
        config.initial_test_population_size, probability_to_log_odds(config.t0_prior)
    )

    return code_log_odds, test_log_odds


def compute_posterior_log_odds(
    prior_log_odds: LogOdds, evidence_log_odds: np.ndarray, observations: np.ndarray
) -> LogOdds:
    """
    Calculate posterior log-odds using Bayesian update.

    Args:
        prior_log_odds: Prior log-odds of entity being correct
        evidence_log_odds: Array of log-odds for evidence sources
        observations: Array of observations (1 for pass/true, 0 for fail/false)

    Returns:
        Updated posterior log-odds

    Raises:
        ValueError: If evidence_log_odds and observations have different lengths
    """
    if len(evidence_log_odds) != len(observations):
        raise ValueError(
            f"Length mismatch: evidence_log_odds has {len(evidence_log_odds)} "
            f"elements, observations has {len(observations)} elements"
        )

    # Calculate evidence weights based on observations
    # Pass (1) contributes positive evidence, fail (0) contributes negative evidence
    evidence_weights = np.where(
        observations == 1, evidence_log_odds, -evidence_log_odds
    )

    # Sum all evidence
    total_evidence = np.sum(evidence_weights)

    # Posterior = prior + evidence
    posterior_log_odds = prior_log_odds + total_evidence

    return float(posterior_log_odds)


def update_code_beliefs(
    code_log_odds: np.ndarray, test_log_odds: np.ndarray, evaluation_matrix: np.ndarray
) -> np.ndarray:
    """
    Update beliefs about code correctness using test results.

    Args:
        code_log_odds: Current log-odds for each code
        test_log_odds: Current log-odds for each test
        evaluation_matrix: Matrix of evaluation results

    Returns:
        Updated log-odds for each code
    """
    updated_code_log_odds = np.zeros_like(code_log_odds)

    for i in range(len(code_log_odds)):
        prior = code_log_odds[i]
        evidence = test_log_odds
        observations = evaluation_matrix[i, :]

        updated_code_log_odds[i] = compute_posterior_log_odds(
            prior, evidence, observations
        )

    return updated_code_log_odds


def update_test_beliefs(
    code_log_odds: np.ndarray, test_log_odds: np.ndarray, evaluation_matrix: np.ndarray
) -> np.ndarray:
    """
    Update beliefs about test correctness using code evaluation results.

    Args:
        code_log_odds: Current log-odds for each code
        test_log_odds: Current log-odds for each test
        evaluation_matrix: Matrix of evaluation results

    Returns:
        Updated log-odds for each test
    """
    updated_test_log_odds = np.zeros_like(test_log_odds)

    for j in range(len(test_log_odds)):
        prior = test_log_odds[j]
        evidence = code_log_odds
        observations = evaluation_matrix[:, j]

        updated_test_log_odds[j] = compute_posterior_log_odds(
            prior, evidence, observations
        )

    return updated_test_log_odds


def update_population_beliefs(
    code_log_odds: np.ndarray, test_log_odds: np.ndarray, evaluation_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update beliefs for both code and test populations.

    Args:
        code_log_odds: Current log-odds for each code
        test_log_odds: Current log-odds for each test
        evaluation_matrix: Matrix of evaluation results

    Returns:
        Tuple of (updated_code_log_odds, updated_test_log_odds)
    """
    updated_code_log_odds = update_code_beliefs(
        code_log_odds, test_log_odds, evaluation_matrix
    )
    updated_test_log_odds = update_test_beliefs(
        code_log_odds, test_log_odds, evaluation_matrix
    )

    return updated_code_log_odds, updated_test_log_odds
