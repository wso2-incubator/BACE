"""
Configuration for coevolutionary algorithms.

This module contains configuration classes for coevolutionary algorithms,
particularly for code-test coevolution with Bayesian belief updating.
"""

from typing import Optional


class CoevolutionConfig:
    """Complete configuration for coevolutionary algorithm orchestrator."""

    def __init__(
        self,
        # === Bayesian Parameters ===
        initial_code_population_size: int = 10,
        initial_test_population_size: int = 20,
        initial_code_prior: float = 0.5,  # Prior probability of code being correct
        initial_test_prior: float = 0.5,  # Prior probability of test being correct
        alpha: float = 0.1,  # P(pass | code correct, test incorrect)
        beta: float = 0.2,  # P(pass | code incorrect, test correct)
        gamma: float = 0.5,  # P(pass | code incorrect, test incorrect)
        learning_rate: float = 1.0,  # Learning rate for belief updates
        use_intermediate_updates: bool = False,  # Use updated code probs for test updates
        # === Evolution Strategy ===
        num_generations: int = 50,  # Number of evolutionary cycles
        selection_strategy: str = "binary_tournament",  # Selection method
        # Genetic operator rates for CODE population
        code_crossover_rate: float = 0.7,  # Probability of crossover
        code_mutation_rate: float = 0.2,  # Probability of mutation
        code_edit_rate: float = 0.1,  # Probability of edit based on feedback
        code_elitism_count: int = 2,  # Number of elite individuals to preserve
        code_offspring_size: Optional[
            int
        ] = None,  # Offspring per generation (None = pop_size)
        # Genetic operator rates for TEST population
        test_crossover_rate: float = 0.6,
        test_mutation_rate: float = 0.3,
        test_edit_rate: float = 0.1,
        test_elitism_count: int = 1,
        test_offspring_size: Optional[int] = None,
        # === LLM Configuration ===
        llm_model: str = "gpt-4",  # LLM model for genetic operators
    ) -> None:
        """
        Initialize coevolution configuration.

        Args:
            # Bayesian Parameters
            initial_code_population_size: Size of initial code population
            initial_test_population_size: Size of initial test population
            initial_code_prior: Prior probability that a code is correct
            initial_test_prior: Prior probability that a test is correct
            alpha: Hyperparameter for a correct code passing an incorrect test
            beta: Hyperparameter for an incorrect code passing a correct test
            gamma: Hyperparameter for an incorrect code passing an incorrect test
            learning_rate: Learning rate for belief updates (scales the WoE updates)
            use_intermediate_updates: If True, uses updated code probabilities when
                                     calculating test updates

            # Evolution Strategy
            num_generations: Number of evolutionary cycles to run
            selection_strategy: Selection method (e.g., "binary_tournament", "roulette_wheel")

            # Code Population Genetic Operators
            code_crossover_rate: Probability of applying crossover to code
            code_mutation_rate: Probability of applying mutation to code
            code_edit_rate: Probability of applying edit to code (based on test feedback)
            code_elitism_count: Number of top code individuals to preserve unchanged
            code_offspring_size: Number of new code individuals per generation (None = population_size)

            # Test Population Genetic Operators
            test_crossover_rate: Probability of applying crossover to tests
            test_mutation_rate: Probability of applying mutation to tests
            test_edit_rate: Probability of applying edit to tests (based on code feedback)
            test_elitism_count: Number of top test individuals to preserve unchanged
            test_offspring_size: Number of new test individuals per generation (None = population_size)

            # LLM Configuration
            llm_model: LLM model name for genetic operators
        """
        # === Validate Bayesian Parameters ===
        if not (0 < initial_code_prior < 1):
            raise ValueError("initial_code_prior must be between 0 and 1")
        if not (0 < initial_test_prior < 1):
            raise ValueError("initial_test_prior must be between 0 and 1")
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

        # === Validate Evolution Parameters ===
        if num_generations <= 0:
            raise ValueError("num_generations must be positive")

        # Validate code genetic operator rates
        if not (0 <= code_crossover_rate <= 1):
            raise ValueError("code_crossover_rate must be between 0 and 1")
        if not (0 <= code_mutation_rate <= 1):
            raise ValueError("code_mutation_rate must be between 0 and 1")
        if not (0 <= code_edit_rate <= 1):
            raise ValueError("code_edit_rate must be between 0 and 1")
        if code_crossover_rate + code_mutation_rate + code_edit_rate > 1.0:
            raise ValueError(
                "Sum of code_crossover_rate, code_mutation_rate, and code_edit_rate "
                "must not exceed 1.0"
            )
        if code_elitism_count < 0:
            raise ValueError("code_elitism_count must be non-negative")
        if code_elitism_count >= initial_code_population_size:
            raise ValueError(
                "code_elitism_count must be less than initial_code_population_size"
            )

        # Validate test genetic operator rates
        if not (0 <= test_crossover_rate <= 1):
            raise ValueError("test_crossover_rate must be between 0 and 1")
        if not (0 <= test_mutation_rate <= 1):
            raise ValueError("test_mutation_rate must be between 0 and 1")
        if not (0 <= test_edit_rate <= 1):
            raise ValueError("test_edit_rate must be between 0 and 1")
        if test_crossover_rate + test_mutation_rate + test_edit_rate > 1.0:
            raise ValueError(
                "Sum of test_crossover_rate, test_mutation_rate, and test_edit_rate "
                "must not exceed 1.0"
            )
        if test_elitism_count < 0:
            raise ValueError("test_elitism_count must be non-negative")
        if test_elitism_count >= initial_test_population_size:
            raise ValueError(
                "test_elitism_count must be less than initial_test_population_size"
            )

        # Validate offspring sizes
        if code_offspring_size is not None and code_offspring_size <= 0:
            raise ValueError("code_offspring_size must be positive if specified")
        if test_offspring_size is not None and test_offspring_size <= 0:
            raise ValueError("test_offspring_size must be positive if specified")

        # === Assign Bayesian Parameters ===
        self.initial_code_population_size = initial_code_population_size
        self.initial_test_population_size = initial_test_population_size
        self.initial_code_prior = initial_code_prior
        self.initial_test_prior = initial_test_prior
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.use_intermediate_updates = use_intermediate_updates

        # === Assign Evolution Strategy Parameters ===
        self.num_generations = num_generations
        self.selection_strategy = selection_strategy

        # Code population genetic operators
        self.code_crossover_rate = code_crossover_rate
        self.code_mutation_rate = code_mutation_rate
        self.code_edit_rate = code_edit_rate
        self.code_elitism_count = code_elitism_count
        self.code_offspring_size = (
            code_offspring_size
            if code_offspring_size is not None
            else initial_code_population_size
        )

        # Test population genetic operators
        self.test_crossover_rate = test_crossover_rate
        self.test_mutation_rate = test_mutation_rate
        self.test_edit_rate = test_edit_rate
        self.test_elitism_count = test_elitism_count
        self.test_offspring_size = (
            test_offspring_size
            if test_offspring_size is not None
            else initial_test_population_size
        )

        # === Assign LLM Configuration ===
        self.llm_model = llm_model
