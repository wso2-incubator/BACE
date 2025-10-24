"""
Coevolution orchestrator for code-test evolutionary algorithms.

This module provides the main orchestrator class that coordinates the coevolution
of code solutions and test cases using Bayesian belief updating and genetic
operators.

The orchestrator manages:
- Initial population generation for both code and test
- Observation matrix generation through code execution
- Bayesian belief updates based on test results
- Genetic operations (crossover, mutation, edit) with configurable rates
- Selection strategies for parent selection
- Elitism to preserve best individuals
- Generation cycling and termination

Example:
    >>> from common.llm_client import LLMClient
    >>> from common.sandbox import SafeCodeSandbox
    >>> from common.coevolution import CoevolutionConfig, CoevolutionOrchestrator
    >>> from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    >>>
    >>> # Setup
    >>> config = CoevolutionConfig(
    ...     initial_code_population_size=10,
    ...     initial_test_population_size=20,
    ...     num_generations=50
    ... )
    >>> llm = LLMClient(model="gpt-4")
    >>> sandbox = SafeCodeSandbox()
    >>> problem = CodeGenerationProblem(...)
    >>>
    >>> # Run coevolution
    >>> orchestrator = CoevolutionOrchestrator(config, problem, llm, sandbox)
    >>> best_code, best_test = orchestrator.run()
"""

from typing import Dict, List, Tuple

import numpy as np
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem  # type: ignore
from loguru import logger

from common.code_preprocessing.analyzers import extract_test_methods_code
from common.llm_client import LLMClient
from common.sandbox import SafeCodeSandbox, TestExecutionResult

from .bayesian import initialize_prior_beliefs, update_population_beliefs
from .config import CoevolutionConfig
from .evaluation import execute_code_against_tests, generate_observation_matrix
from .feedback import generate_feedback_for_code
from .operators import CodeOperator, TestOperator
from .population import BasePopulation, CodePopulation, TestPopulation
from .reproduction import ReproductionStrategy
from .selection import SelectionStrategy


class CoevolutionOrchestrator:
    """
    Main orchestrator for code-test coevolution algorithm.

    This class coordinates the entire coevolution process, managing both
    code and test populations through multiple generations of evolution
    and Bayesian belief updating.

    Attributes:
        config: Configuration object with all algorithm parameters
        problem: The coding problem to solve
        sandbox: Safe execution environment for testing code
        code_operator: LLM operator for code genetic operations
        test_operator: LLM operator for test genetic operations
        selector: Selection strategy for choosing parents
        code_population: Current code population
        test_population: Current test population
        generation: Current generation number
        last_execution_results: Detailed test execution results from previous generation
                              (used for generating feedback in edit operations)
    """

    def __init__(
        self,
        config: CoevolutionConfig,
        problem: CodeGenerationProblem,
        llm_client: LLMClient,
        sandbox: SafeCodeSandbox,
    ) -> None:
        """
        Initialize the coevolution orchestrator.

        Args:
            config: Configuration object with all algorithm parameters
            problem: The coding problem to solve
            llm_client: LLM client for genetic operations
            sandbox: Safe execution environment for testing code
        """
        logger.info("Initializing CoevolutionOrchestrator")

        # Store configuration and dependencies
        self.config = config
        self.problem = problem
        self.sandbox = sandbox

        # Initialize LLM operators for code and test
        logger.info(f"Initializing operators with LLM model: {config.llm_model}")
        self.code_operator = CodeOperator(llm_client, problem=problem)
        self.test_operator = TestOperator(llm_client, problem=problem)

        # Initialize selection strategy with configured method
        logger.info(f"Initializing selection strategy: {config.selection_strategy}")
        self.selector = SelectionStrategy(method=config.selection_strategy)

        # Initialize reproduction strategies for code and test populations
        logger.info("Initializing reproduction strategies")
        self.code_reproduction: ReproductionStrategy = ReproductionStrategy(
            selector=self.selector,
            operator=self.code_operator,
            initial_prior=config.initial_code_prior,
        )
        self.test_reproduction: ReproductionStrategy = ReproductionStrategy(
            selector=self.selector,
            operator=self.test_operator,
            initial_prior=config.initial_test_prior,
        )

        # Populations will be initialized in run()
        self.code_population: CodePopulation | None = None
        self.test_population: TestPopulation | None = None
        self.generation = 0

        # Execution results for feedback (initialized in run())
        self.last_execution_results: Dict[int, TestExecutionResult] = {}

        logger.info("CoevolutionOrchestrator initialized successfully")

    def _create_initial_code_population(self) -> CodePopulation:
        """
        Create the initial code population with uniform priors.

        Returns:
            Initialized CodePopulation with uniform prior probabilities
        """
        logger.info(
            f"Creating initial code population (size={self.config.initial_code_population_size})"
        )

        # Generate initial code solutions using LLM operator
        code_individuals = self.code_operator.create_initial_population(
            self.config.initial_code_population_size
        )

        logger.info(f"Generated {len(code_individuals)} initial code solutions")

        # Initialize prior probabilities
        code_probs = initialize_prior_beliefs(
            len(code_individuals), self.config.initial_code_prior
        )

        # Create population
        code_population = CodePopulation(
            individuals=code_individuals, probabilities=code_probs, generation=0
        )

        logger.info(
            f"Created CodePopulation: size={code_population.size}, "
            f"avg_prob={np.mean(code_probs):.4f}"
        )

        return code_population

    def _create_initial_test_population(self) -> TestPopulation:
        """
        Create the initial test population with uniform priors.

        Returns:
            Initialized TestPopulation with uniform prior probabilities
        """
        logger.info(
            f"Creating initial test population (size={self.config.initial_test_population_size})"
        )

        # Generate initial test class using LLM operator
        # Note: test_operator.create_initial_population returns a single unittest class string
        test_class_block = self.test_operator.create_initial_population(
            self.config.initial_test_population_size
        )

        logger.debug("Generated initial test class, extracting individual test methods")

        # Extract individual test methods from the unittest class
        test_individuals = extract_test_methods_code(test_class_block)

        logger.info(f"Extracted {len(test_individuals)} test methods from test class")

        # Initialize prior probabilities
        test_probs = initialize_prior_beliefs(
            len(test_individuals), self.config.initial_test_prior
        )

        # Create population
        test_population = TestPopulation(
            individuals=test_individuals,
            probabilities=test_probs,
            generation=0,
            test_class_block=test_class_block,
        )

        logger.info(
            f"Created TestPopulation: size={test_population.size}, "
            f"avg_prob={np.mean(test_probs):.4f}"
        )

        return test_population

    def _generate_code_offspring(self, elite_count: int) -> List[Tuple[str, float]]:
        """
        Generate offspring for the code population using genetic operators.

        Delegates to ReproductionStrategy for the actual offspring generation.
        Adjusts offspring count to ensure it fits within remaining population space.

        Args:
            elite_count: Number of elite individuals being preserved

        Returns:
            List of (offspring_code, probability) tuples
        """
        assert self.code_population is not None, "Code population must be initialized"
        assert self.test_population is not None, "Test population must be initialized"

        # Calculate remaining space after elites
        remaining_space = self.config.max_code_population_size - elite_count
        # Adjust offspring count to fit remaining space (Option 3: offspring fills remainder)
        actual_offspring_count = min(self.config.code_offspring_count, remaining_space)

        return self.code_reproduction.generate_offspring(
            population=self.code_population,
            other_population=self.test_population,
            execution_results=self.last_execution_results,
            feedback_generator=generate_feedback_for_code,
            offspring_size=actual_offspring_count,
            crossover_rate=self.config.code_crossover_rate,
            mutation_rate=self.config.code_mutation_rate,
            edit_rate=self.config.code_edit_rate,
            population_type="code",
        )

    def _generate_test_offspring(self, elite_count: int) -> List[Tuple[str, float]]:
        """
        Generate offspring for the test population using genetic operators.

        Delegates to ReproductionStrategy for the actual offspring generation.
        Adjusts offspring count to ensure it fits within remaining population space.

        Note: Edit operation for tests is not yet implemented. The feedback_generator
        is a placeholder that will be replaced when test feedback generation is ready.

        Args:
            elite_count: Number of elite individuals being preserved

        Returns:
            List of (offspring_test, probability) tuples
        """
        assert self.test_population is not None, "Test population must be initialized"
        assert self.code_population is not None, "Code population must be initialized"

        # Calculate remaining space after elites
        remaining_space = self.config.max_test_population_size - elite_count
        # Adjust offspring count to fit remaining space (Option 3: offspring fills remainder)
        actual_offspring_count = min(self.config.test_offspring_count, remaining_space)

        # Placeholder feedback generator for tests (to be implemented)
        def _placeholder_test_feedback(
            execution_result: TestExecutionResult,
            code_population: BasePopulation,
            test_idx: int,
        ) -> str:
            return "Placeholder feedback for test edit operation"

        return self.test_reproduction.generate_offspring(
            population=self.test_population,
            other_population=self.code_population,
            execution_results=self.last_execution_results,
            feedback_generator=_placeholder_test_feedback,
            offspring_size=actual_offspring_count,
            crossover_rate=self.config.test_crossover_rate,
            mutation_rate=self.config.test_mutation_rate,
            edit_rate=self.config.test_edit_rate,
            population_type="test",
        )

    def _create_next_code_generation(
        self, elites: List[Tuple[str, float]], offspring: List[Tuple[str, float]]
    ) -> None:
        """
        Update the code population with the next generation from elites and offspring.

        Combines elite individuals with offspring to form the new population.
        If the combined size exceeds the configured population size, the best
        individuals are selected based on their probabilities.

        Args:
            elites: List of (code, probability) tuples for elite individuals
            offspring: List of (code, probability) tuples for offspring
        """
        assert self.code_population is not None, "Code population must be initialized"

        # Combine elites and offspring
        combined = elites + offspring

        logger.debug(
            f"Creating next code generation: {len(elites)} elites + "
            f"{len(offspring)} offspring = {len(combined)} total"
        )

        # If combined size exceeds max, select best individuals
        if len(combined) > self.config.max_code_population_size:
            logger.debug(
                f"Selecting best {self.config.max_code_population_size} from {len(combined)}"
            )
            # Sort by probability (descending) and take top N
            combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
            combined = combined_sorted[: self.config.max_code_population_size]

        # Unpack into separate lists
        new_individuals = [ind for ind, _ in combined]
        new_probabilities = np.array([prob for _, prob in combined])

        # Replace the entire population and increment generation
        self.code_population.replace_individuals(new_individuals, new_probabilities)
        self.code_population.increment_generation()

        logger.info(
            f"Updated code population to generation {self.code_population.generation}: "
            f"size={self.code_population.size}, avg_prob={np.mean(new_probabilities):.4f}"
        )

    def _create_next_test_generation(
        self, elites: List[Tuple[str, float]], offspring: List[Tuple[str, float]]
    ) -> None:
        """
        Update the test population with the next generation from elites and offspring.

        Combines elite individuals with offspring to form the new population.
        If the combined size exceeds the configured population size, the best
        individuals are selected based on their probabilities.

        Args:
            elites: List of (test_method, probability) tuples for elite individuals
            offspring: List of (test_method, probability) tuples for offspring
        """
        assert self.test_population is not None, "Test population must be initialized"

        # Combine elites and offspring
        combined = elites + offspring

        logger.debug(
            f"Creating next test generation: {len(elites)} elites + "
            f"{len(offspring)} offspring = {len(combined)} total"
        )

        # If combined size exceeds max, select best individuals
        if len(combined) > self.config.max_test_population_size:
            logger.debug(
                f"Selecting best {self.config.max_test_population_size} from {len(combined)}"
            )
            # Sort by probability (descending) and take top N
            combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
            combined = combined_sorted[: self.config.max_test_population_size]

        # Unpack into separate lists
        new_individuals = [ind for ind, _ in combined]
        new_probabilities = np.array([prob for _, prob in combined])

        # Replace the entire population (this will rebuild the test class block automatically)
        self.test_population.replace_individuals(new_individuals, new_probabilities)
        self.test_population.increment_generation()

        logger.info(
            f"Updated test population to generation {self.test_population.generation}: "
            f"size={self.test_population.size}, avg_prob={np.mean(new_probabilities):.4f}"
        )

    def run(self) -> Tuple[str, float, str, float]:
        """
        Run the complete coevolution algorithm.

        This method orchestrates the entire coevolution process:
        1. Creates initial populations for code and tests
        2. Iterates through generations:
           - Generates observation matrix by executing code against tests
           - Updates beliefs using Bayesian updates
           - Selects elite individuals
           - Generates offspring using genetic operators
           - Creates next generation from elites and offspring
        3. Returns the best code solution and test case

        Returns:
            Tuple of (best_code, best_code_prob, best_test, best_test_prob)

        Raises:
            RuntimeError: If algorithm encounters unrecoverable errors
        """
        logger.info("=" * 80)
        logger.info("STARTING COEVOLUTION ALGORITHM")
        logger.info("=" * 80)
        logger.info(f"Configuration: {self.config.num_generations} generations")
        logger.info(
            f"Code population: size={self.config.initial_code_population_size}, "
            f"prior={self.config.initial_code_prior}"
        )
        logger.info(
            f"Test population: size={self.config.initial_test_population_size}, "
            f"prior={self.config.initial_test_prior}"
        )

        # Step 1: Create initial populations
        logger.info("-" * 80)
        logger.info("STEP 1: Creating initial populations")
        logger.info("-" * 80)

        self.code_population = self._create_initial_code_population()
        self.test_population = self._create_initial_test_population()

        # Main evolution loop
        for generation in range(self.config.num_generations):
            logger.info("=" * 80)
            logger.info(f"GENERATION {generation + 1}/{self.config.num_generations}")
            logger.info("=" * 80)

            # Step 2: Execute code against tests and collect detailed results
            logger.info("-" * 80)
            logger.info("STEP 2: Executing code against tests")
            logger.info("-" * 80)

            execution_results = execute_code_against_tests(
                self.code_population, self.test_population, self.sandbox
            )

            # Store execution results for later use in edit operations
            self.last_execution_results = execution_results

            # Step 3: Generate observation matrix from execution results
            logger.info("-" * 80)
            logger.info("STEP 3: Generating observation matrix")
            logger.info("-" * 80)

            observation_matrix = generate_observation_matrix(
                execution_results,
                self.code_population.size,
                self.test_population.size,
            )

            # Log observation matrix statistics
            total_tests = observation_matrix.size
            passed_tests = np.sum(observation_matrix)
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0
            logger.info(
                f"Observation matrix: {observation_matrix.shape}, "
                f"pass_rate={pass_rate:.2%} ({passed_tests}/{total_tests})"
            )

            # Step 4: Update population beliefs using Bayesian updates
            logger.info("-" * 80)
            logger.info("STEP 4: Updating population beliefs")
            logger.info("-" * 80)

            prior_code_probs = self.code_population.probabilities.copy()
            prior_test_probs = self.test_population.probabilities.copy()

            posterior_code_probs, posterior_test_probs = update_population_beliefs(
                prior_code_probs, prior_test_probs, observation_matrix, self.config
            )

            # Update populations with new probabilities
            self.code_population.update_probabilities(posterior_code_probs)
            self.test_population.update_probabilities(posterior_test_probs)

            # Log best individuals
            best_code, best_code_prob = self.code_population.get_best_individual()
            best_test, best_test_prob = self.test_population.get_best_individual()
            logger.info(
                f"Best code probability: {best_code_prob:.4f}, "
                f"Best test probability: {best_test_prob:.4f}"
            )

            logger.trace(f"Best code snippet:\n{best_code}")
            logger.trace(f"Best test method:\n{best_test}")

            # Step 5: Select elites
            logger.info("-" * 80)
            logger.info("STEP 5: Selecting elite individuals")
            logger.info("-" * 80)

            # Calculate elite count dynamically from current population size
            code_elite_count = max(
                1, int(self.code_population.size * self.config.code_elite_proportion)
            )
            test_elite_count = max(
                1, int(self.test_population.size * self.config.test_elite_proportion)
            )

            code_elite_indices = self.selector.elitism(
                self.code_population.probabilities, code_elite_count
            )
            test_elite_indices = self.selector.elitism(
                self.test_population.probabilities, test_elite_count
            )

            # Convert indices to (individual, probability) tuples
            code_elites = [
                (
                    self.code_population.individuals[idx],
                    self.code_population.probabilities[idx],
                )
                for idx in code_elite_indices
            ]
            test_elites = [
                (
                    self.test_population.individuals[idx],
                    self.test_population.probabilities[idx],
                )
                for idx in test_elite_indices
            ]

            logger.info(
                f"Selected {len(code_elites)} code elites, {len(test_elites)} test elites"
            )

            # Step 6: Generate offspring using genetic operators
            logger.info("-" * 80)
            logger.info("STEP 6: Generating offspring")
            logger.info("-" * 80)

            code_offspring = self._generate_code_offspring(elite_count=code_elite_count)
            test_offspring = self._generate_test_offspring(elite_count=test_elite_count)

            # Step 7: Create next generation
            logger.info("-" * 80)
            logger.info("STEP 7: Creating next generation")
            logger.info("-" * 80)

            self._create_next_code_generation(code_elites, code_offspring)
            self._create_next_test_generation(test_elites, test_offspring)

        # Algorithm complete - return best individuals
        logger.info("=" * 80)
        logger.info("COEVOLUTION COMPLETE")
        logger.info("=" * 80)

        best_code, best_code_prob = self.code_population.get_best_individual()
        best_test, best_test_prob = self.test_population.get_best_individual()

        logger.info(f"Final best code probability: {best_code_prob:.4f}")
        logger.info(f"Final best test probability: {best_test_prob:.4f}")
        logger.info(f"Final code population size: {self.code_population.size}")
        logger.info(f"Final test population size: {self.test_population.size}")

        return best_code, best_code_prob, best_test, best_test_prob
