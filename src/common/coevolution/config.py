"""
Configuration for coevolutionary algorithms.

This module contains configuration classes for coevolutionary algorithms,
particularly for code-test coevolution with Bayesian belief updating.
"""


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
        learning_rate: float = 1.0,  # learning rate for belief updates
    ) -> None:
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
            learning_rate: Learning rate for belief updates (scales the WoE updates)
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
        if not (initial_code_population_size > 0):
            raise ValueError("initial_code_population_size must be positive")
        if not (initial_test_population_size > 0):
            raise ValueError("initial_test_population_size must be positive")
        if not (learning_rate > 0):
            raise ValueError("learning_rate must be positive")

        self.initial_code_population_size = initial_code_population_size
        self.initial_test_population_size = initial_test_population_size
        self.c0_prior = c0_prior
        self.t0_prior = t0_prior
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
