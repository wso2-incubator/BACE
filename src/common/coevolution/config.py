"""
Configuration for coevolutionary algorithms.

This module contains configuration classes for coevolutionary algorithms,
particularly for code-test coevolution with Bayesian belief updating.
"""

from typing import Optional

from loguru import logger

from .selection import SelectionStrategy


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
        # === Population Size Control ===
        max_code_population_size: Optional[
            int
        ] = None,  # Max code population (None = initial)
        max_test_population_size: Optional[
            int
        ] = None,  # Max test population (None = initial)
        # === Evolution Strategy ===
        num_generations: int = 50,  # Number of evolutionary cycles
        selection_strategy: str = "binary_tournament",  # Selection method
        # Genetic operator rates for CODE population
        code_crossover_rate: float = 0.7,  # Probability of crossover
        code_mutation_rate: float = 0.2,  # Probability of mutation
        code_edit_rate: float = 0.1,  # Probability of edit based on feedback
        code_elite_proportion: float = 0.5,  # Proportion of population to preserve as elites
        code_offspring_proportion: float = 0.5,  # Proportion of max population for offspring
        # Genetic operator rates for TEST population
        test_crossover_rate: float = 0.6,
        test_mutation_rate: float = 0.3,
        test_edit_rate: float = 0.1,
        test_elite_proportion: float = 0.5,  # Proportion of population to preserve as elites
        test_offspring_proportion: float = 0.5,  # Proportion of max population for offspring
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

            # Population Size Control
            max_code_population_size: Maximum code population size (None = initial_code_population_size)
            max_test_population_size: Maximum test population size (None = initial_test_population_size)

            # Evolution Strategy
            num_generations: Number of evolutionary cycles to run
            selection_strategy: Selection method (e.g., "binary_tournament", "roulette_wheel")

            # Code Population Genetic Operators
            code_crossover_rate: Probability of applying crossover to code
            code_mutation_rate: Probability of applying mutation to code
            code_edit_rate: Probability of applying edit to code (based on test feedback)
            code_elite_proportion: Proportion of current population to preserve as elites (0.0-1.0)
            code_offspring_proportion: Proportion of max population to generate as offspring (0.0-1.0)

            # Test Population Genetic Operators
            test_crossover_rate: Probability of applying crossover to tests
            test_mutation_rate: Probability of applying mutation to tests
            test_edit_rate: Probability of applying edit to tests (based on code feedback)
            test_elite_proportion: Proportion of current population to preserve as elites (0.0-1.0)
            test_offspring_proportion: Proportion of max population to generate as offspring (0.0-1.0)

            # LLM Configuration
            llm_model: LLM model name for genetic operators

        Note:
            For fixed populations (max_population_size == initial_population_size):
            - If elite_proportion + offspring_proportion > 1.0, offspring count will be
              automatically reduced to fit within the available space after elite selection.
            - For predictable behavior, ensure elite_proportion + offspring_proportion <= 1.0
            - A warning will be logged at initialization if this constraint is violated

            For growing populations (max_population_size > initial_population_size):
            - elite_proportion + offspring_proportion can exceed 1.0 without issues
            - The population will naturally grow up to max_population_size over generations
            - Offspring count is always adjusted to fit remaining space after elites
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

        # === Validate Population Size Control ===
        # Set defaults for max population sizes
        if max_code_population_size is None:
            max_code_population_size = initial_code_population_size
        if max_test_population_size is None:
            max_test_population_size = initial_test_population_size

        if max_code_population_size < initial_code_population_size:
            raise ValueError(
                "max_code_population_size must be >= initial_code_population_size"
            )
        if max_test_population_size < initial_test_population_size:
            raise ValueError(
                "max_test_population_size must be >= initial_test_population_size"
            )

        # Validate elite proportions
        if not (0.0 < code_elite_proportion <= 1.0):
            raise ValueError("code_elite_proportion must be in (0.0, 1.0]")
        if not (0.0 < test_elite_proportion <= 1.0):
            raise ValueError("test_elite_proportion must be in (0.0, 1.0]")

        # Validate offspring proportions
        if not (0.0 < code_offspring_proportion <= 1.0):
            raise ValueError("code_offspring_proportion must be in (0.0, 1.0]")
        if not (0.0 < test_offspring_proportion <= 1.0):
            raise ValueError("test_offspring_proportion must be in (0.0, 1.0]")

        # === Validate Evolution Parameters ===
        if num_generations <= 0:
            raise ValueError("num_generations must be positive")

        if selection_strategy not in SelectionStrategy.get_available_methods():
            available = ", ".join(SelectionStrategy.get_available_methods())
            raise ValueError(
                f"Invalid selection_strategy: '{selection_strategy}'. "
                f"Must be one of: {available}"
            )

        # Validate code genetic operator rates
        if not (0 <= code_crossover_rate <= 1):
            raise ValueError("code_crossover_rate must be between 0 and 1")
        if not (0 <= code_mutation_rate <= 1):
            raise ValueError("code_mutation_rate must be between 0 and 1")
        if not (0 <= code_edit_rate <= 1):
            raise ValueError("code_edit_rate must be between 0 and 1")
        if code_crossover_rate + code_edit_rate > 1.0:
            raise ValueError(
                "Sum of code_crossover_rate and code_edit_rate must not exceed 1.0"
            )

        # Validate test genetic operator rates
        if not (0 <= test_crossover_rate <= 1):
            raise ValueError("test_crossover_rate must be between 0 and 1")
        if not (0 <= test_mutation_rate <= 1):
            raise ValueError("test_mutation_rate must be between 0 and 1")
        if not (0 <= test_edit_rate <= 1):
            raise ValueError("test_edit_rate must be between 0 and 1")
        if test_crossover_rate + test_edit_rate > 1.0:
            raise ValueError(
                "Sum of test_crossover_rate and test_edit_rate must not exceed 1.0"
            )

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

        # === Assign Population Size Control ===
        self.max_code_population_size = max_code_population_size
        self.max_test_population_size = max_test_population_size

        # === Assign Evolution Strategy Parameters ===
        self.num_generations = num_generations
        self.selection_strategy = selection_strategy

        # Code population genetic operators
        self.code_crossover_rate = code_crossover_rate
        self.code_mutation_rate = code_mutation_rate
        self.code_edit_rate = code_edit_rate
        self.code_elite_proportion = code_elite_proportion
        self.code_offspring_proportion = code_offspring_proportion
        # Calculate fixed offspring count from max population and proportion
        self.code_offspring_count = int(
            max_code_population_size * code_offspring_proportion
        )

        # Test population genetic operators
        self.test_crossover_rate = test_crossover_rate
        self.test_mutation_rate = test_mutation_rate
        self.test_edit_rate = test_edit_rate
        self.test_elite_proportion = test_elite_proportion
        self.test_offspring_proportion = test_offspring_proportion
        # Calculate fixed offspring count from max population and proportion
        self.test_offspring_count = int(
            max_test_population_size * test_offspring_proportion
        )

        # === Validate proportion compatibility for fixed populations ===
        # For fixed populations (max == initial), warn if proportions might cause
        # offspring to be reduced due to space constraints
        if max_code_population_size == initial_code_population_size:
            # Calculate what would happen with max elites
            max_code_elites = max(
                1, int(max_code_population_size * code_elite_proportion)
            )
            if max_code_elites + self.code_offspring_count > max_code_population_size:
                logger.warning(
                    f"Code population configuration may reduce offspring count: "
                    f"With fixed population size ({max_code_population_size}), "
                    f"elite_proportion ({code_elite_proportion}) and "
                    f"offspring_proportion ({code_offspring_proportion}) can result in "
                    f"up to {max_code_elites} elites + {self.code_offspring_count} target offspring "
                    f"= {max_code_elites + self.code_offspring_count} (exceeds max by "
                    f"{max_code_elites + self.code_offspring_count - max_code_population_size}). "
                    f"Offspring will be automatically reduced to fit. "
                    f"For predictable behavior, ensure elite_proportion + offspring_proportion <= 1.0"
                )

        if max_test_population_size == initial_test_population_size:
            # Calculate what would happen with max elites
            max_test_elites = max(
                1, int(max_test_population_size * test_elite_proportion)
            )
            if max_test_elites + self.test_offspring_count > max_test_population_size:
                logger.warning(
                    f"Test population configuration may reduce offspring count: "
                    f"With fixed population size ({max_test_population_size}), "
                    f"elite_proportion ({test_elite_proportion}) and "
                    f"offspring_proportion ({test_offspring_proportion}) can result in "
                    f"up to {max_test_elites} elites + {self.test_offspring_count} target offspring "
                    f"= {max_test_elites + self.test_offspring_count} (exceeds max by "
                    f"{max_test_elites + self.test_offspring_count - max_test_population_size}). "
                    f"Offspring will be automatically reduced to fit. "
                    f"For predictable behavior, ensure elite_proportion + offspring_proportion <= 1.0"
                )

        # === Assign LLM Configuration ===
        self.llm_model = llm_model
