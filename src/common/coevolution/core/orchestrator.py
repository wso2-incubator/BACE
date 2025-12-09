# src/common/coevolution/orchestrator.py

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from loguru import logger

from common.code_preprocessing.transformation import extract_test_methods_code
from common.coevolution import logging_utils

# Import concrete classes
from .breeding_strategy import BreedingStrategy
from .individual import CodeIndividual, TestIndividual

# Import all interfaces, configs, and types
from .interfaces import (
    OPERATION_INITIAL,
    BasePopulation,
    BayesianConfig,
    CodePopulationConfig,
    EvolutionConfig,
    ExecutionResults,
    IBayesianSystem,
    ICodeOperator,
    IDatasetTestBlockBuilder,
    IExecutionSystem,
    IFeedbackGenerator,
    IPareto,
    IProbabilityAssigner,
    ISelectionStrategy,
    ITestBlockRebuilder,
    ITestOperator,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
    Sandbox,
    Test,
)
from .population import CodePopulation, TestPopulation


@dataclass
class InteractionData:
    """Container for execution results and observation matrices for a generation."""

    gen_exec_results: ExecutionResults
    pub_exec_results: ExecutionResults
    gen_obs_matrix: np.ndarray
    pub_obs_matrix: np.ndarray


class Orchestrator:
    """
    Orchestrator for the co-evolutionary algorithm.

    This class coordinates the entire coevolution process by "wiring up"
    all the necessary components (operators, selectors, updaters)
    which are injected as dependencies. It follows the plan of:

    1. Execute (Generated, Public, Private tests)
    2. Update Beliefs (from Generated, then Public)
    3. Select Elites (top-k for code, Pareto for tests)
    4. Generate Offspring (using breeders)
    5. Set Next Generation (notify removed individuals of death)
    6. Log Evolution Progress (generation summaries and individual lifecycles)

    Logging Strategy:
    - Generation summaries: Logged at each generation with aggregate statistics
    - Individual lifecycles: Logged once when individual dies or survives to the end
    """

    def __init__(
        self,
        # --- Configuration Objects ---
        evo_config: EvolutionConfig,
        code_pop_config: CodePopulationConfig,
        test_pop_config: PopulationConfig,
        code_op_rates_config: OperatorRatesConfig,
        test_op_rates_config: OperatorRatesConfig,
        bayesian_config: BayesianConfig,
        # --- Injected Mock Components ---
        problem: Problem,
        sandbox: Sandbox,
        code_operator: ICodeOperator,
        test_operator: ITestOperator,
        code_selector: ISelectionStrategy,
        test_selector: ISelectionStrategy,
        code_prob_assigner: IProbabilityAssigner,
        test_prob_assigner: IProbabilityAssigner,
        execution_system: IExecutionSystem,
        code_bayesian_system: IBayesianSystem,
        test_bayesian_system: IBayesianSystem,
        pareto: IPareto,
        test_block_rebuilder: ITestBlockRebuilder,
        code_feedback_gen: IFeedbackGenerator[TestIndividual],
        test_feedback_gen: IFeedbackGenerator[CodeIndividual],
        dataset_test_block_builder: IDatasetTestBlockBuilder,
    ) -> None:
        """
        Initializes the orchestrator by storing all injected dependencies.

        Also sets up:
        - Concrete breeding strategies for code and test populations (each with its own selector)
        - Generation logger for tracking evolution progress (via logging_utils)
        - Random seed for reproducibility

        Args:
            code_selector: Selection strategy for code population
            test_selector: Selection strategy for test population
            code_prob_assigner: Probability assigner for code offspring
            test_prob_assigner: Probability assigner for test offspring
            ... (other args documented inline)
        """
        logger.info("Initializing MockOrchestrator...")

        # --- Store Configs ---
        self.evo_config = evo_config
        self.code_pop_config = code_pop_config
        self.test_pop_config = test_pop_config
        self.code_op_rates_config = code_op_rates_config
        self.test_op_rates_config = test_op_rates_config
        self.bayesian_config = bayesian_config

        # --- Store max_workers for parallel breeding ---
        self.max_workers = evo_config.max_workers
        logger.info(f"Parallel breeding configured with max_workers={self.max_workers}")

        # --- Store Problem & Sandbox ---
        self.problem = problem
        self.sandbox = sandbox

        # --- Store Injected Components ---
        self.code_operator = code_operator
        self.test_operator = test_operator
        self.code_selector = code_selector
        self.test_selector = test_selector
        self.code_prob_assigner = code_prob_assigner
        self.test_prob_assigner = test_prob_assigner
        self.execution_system = execution_system

        # TODO: These can be the same instance, no need to have two
        self.code_bayesian_system = code_bayesian_system
        self.test_bayesian_system = test_bayesian_system

        self.pareto = pareto
        self.test_block_rebuilder = test_block_rebuilder
        self.code_feedback_gen = code_feedback_gen
        self.test_feedback_gen = test_feedback_gen
        self.dataset_test_block_builder = dataset_test_block_builder

        # --- Create Concrete Breeding Strategies ---
        # Each breeding strategy uses its own dedicated selector and probability assigner
        self.code_breeder = BreedingStrategy[CodeIndividual, TestIndividual](
            selector=self.code_selector,
            operator=self.code_operator,
            individual_factory=CodeIndividual,
            probability_assigner=self.code_prob_assigner,
            initial_prior=self.code_pop_config.initial_prior,
        )
        self.test_breeder = BreedingStrategy[TestIndividual, CodeIndividual](
            selector=self.test_selector,
            operator=self.test_operator,
            individual_factory=TestIndividual,
            probability_assigner=self.test_prob_assigner,
            initial_prior=self.test_pop_config.initial_prior,
        )

        self._set_random_seed(evo_config.random_seed)

    def run(self) -> tuple[CodePopulation, TestPopulation]:
        """
        Runs the main co-evolutionary loop.
        """
        logging_utils.log_section_header("INFO", "STARTING CO-EVOLUTION RUN")

        # 1. Initialization
        code_pop, test_pop, pub_pop, priv_pop = self._initialize_evolution()

        # 2. Main Loop
        for gen in range(self.evo_config.num_generations + 1):
            logging_utils.log_subsection_header(
                "INFO",
                f"GENERATION {gen} / {self.evo_config.num_generations}",
            )

            # A. Execute
            interactions = self._execute_all_interactions(code_pop, test_pop, pub_pop)

            # B. Update Beliefs (Cooperative & Anchored)
            code_pop, test_pop = self._perform_cooperative_updates(
                code_pop,
                test_pop,
                pub_pop,
                interactions.gen_obs_matrix,
                interactions.pub_obs_matrix,
            )

            # C. Evolve (Select -> Breed -> Transition) # Only if not last generation
            if gen < self.evo_config.num_generations:
                code_pop, test_pop = self._produce_next_generation(
                    code_pop, test_pop, pub_pop, interactions
                )

            # D. Log
            logging_utils.log_generation_summary(code_pop, test_pop)

        # 3. Finalization
        self._finalize_evolution(code_pop, test_pop, pub_pop, priv_pop)

        return code_pop, test_pop

    # =========================================================================
    # Lifecycle Phase 1: Initialization
    # =========================================================================

    def _initialize_evolution(
        self,
    ) -> tuple[CodePopulation, TestPopulation, TestPopulation, TestPopulation]:
        """Creates initial populations and logs initial state."""

        # Create Populations
        code_pop = self._create_initial_code_population()
        test_pop = self._create_initial_test_populations()

        pub_pop = self._create_fixed_test_population_from_test_cases(
            self.problem.public_test_cases, self.problem.starter_code
        )
        priv_pop = self._create_fixed_test_population_from_test_cases(
            self.problem.private_test_cases, self.problem.starter_code
        )

        # Initial Private Run (Baseline)
        _, priv_obs = self._get_exec_results_and_obs_matrix(code_pop, priv_pop)

        logging_utils.log_observation_matrix(priv_obs, code_pop, priv_pop, "private")
        logging_utils.log_generation_summary(code_pop, test_pop)

        return code_pop, test_pop, pub_pop, priv_pop

    # =========================================================================
    # Lifecycle Phase 2A: Execution
    # =========================================================================

    def _execute_all_interactions(
        self,
        code_pop: CodePopulation,
        test_pop: TestPopulation,
        pub_pop: TestPopulation,
    ) -> InteractionData:
        """Executes code against both generated and public tests."""
        logger.debug("Step 1: Executing gen and public test sets...")

        gen_exec, gen_obs = self._get_exec_results_and_obs_matrix(code_pop, test_pop)
        pub_exec, pub_obs = self._get_exec_results_and_obs_matrix(code_pop, pub_pop)

        logging_utils.log_observation_matrix(gen_obs, code_pop, test_pop, "generated")
        logging_utils.log_observation_matrix(pub_obs, code_pop, pub_pop, "public")

        return InteractionData(gen_exec, pub_exec, gen_obs, pub_obs)

    # =========================================================================
    # Lifecycle Phase 2B: Belief Updates (The Core Logic)
    # =========================================================================

    def _perform_cooperative_updates(
        self,
        code_pop: CodePopulation,
        test_pop: TestPopulation,
        pub_pop: TestPopulation,
        gen_obs_matrix: np.ndarray,
        pub_obs_matrix: np.ndarray,
    ) -> tuple[CodePopulation, TestPopulation]:
        """
        Performs the Bayesian belief updates in a specific order to prevent
        confirmation bias.

        Strategy:
        1. Masking: Determine which individuals should be updated based on their
           generation born and the current generation.
        2. Anchor: Update Code using Public Tests (Ground Truth).
        3. Calculate: Compute updates for both populations using the generated observations.
           - Tests judge Code (that has seen Public tests).
           - Code judges Tests (using Test Priors).
        4. Apply: Apply updates simultaneously.
        """
        logger.debug("Step 2: Updating beliefs...")

        # 1. Get Masks
        logger.debug("Getting mask matrices for updates...")
        code_mask_gen, code_mask_pub, test_mask = self._get_mask_matrices(
            code_pop, test_pop, pub_pop
        )

        # 2. ANCHOR: Update Code based on Public Tests first
        logger.debug("Anchoring code beliefs with public tests...")
        code_post_public = self.code_bayesian_system.update_code_beliefs(
            prior_code_probs=code_pop.probabilities,
            prior_test_probs=pub_pop.probabilities,
            observation_matrix=pub_obs_matrix,
            code_update_mask_matrix=code_mask_pub,
            config=self.bayesian_config,
        )
        code_pop.update_probabilities(code_post_public)

        # 3. CALCULATE: Compute updates for both populations using the generated observations.
        logger.debug("Calculating test updates based on updated code beliefs...")
        test_post_generated = self.test_bayesian_system.update_test_beliefs(
            prior_code_probs=code_pop.probabilities,
            prior_test_probs=test_pop.probabilities,
            observation_matrix=gen_obs_matrix,
            test_update_mask_matrix=test_mask,
            config=self.bayesian_config,
        )
        logger.debug("Calculating code updates based on test priors...")
        code_post_generated = self.code_bayesian_system.update_code_beliefs(
            prior_code_probs=code_pop.probabilities,
            prior_test_probs=test_pop.probabilities,
            observation_matrix=gen_obs_matrix,
            code_update_mask_matrix=code_mask_gen,
            config=self.bayesian_config,
        )

        # 4. APPLY: Apply simultaneous updates
        logger.debug("Applying simultaneous updates...")
        test_pop.update_probabilities(test_post_generated)
        code_pop.update_probabilities(code_post_generated)

        return code_pop, test_pop

    # =========================================================================
    # Lifecycle Phase 2C: Evolution (Selection & Breeding)
    # =========================================================================

    def _produce_next_generation(
        self,
        code_pop: CodePopulation,
        test_pop: TestPopulation,
        pub_pop: TestPopulation,
        interactions: InteractionData,
    ) -> tuple[CodePopulation, TestPopulation]:
        """
        Handles the full evolution cycle: Selection, Breeding, and Transition.
        Mutates the population objects in-place.
        """
        # 1. Select Elites
        logger.debug("Step 3: Selecting elites...")
        code_elites, test_elites = self._select_elites(
            code_pop, test_pop, interactions.gen_obs_matrix
        )
        self._notify_elites(code_elites, code_pop.generation)
        self._notify_elites(test_elites, test_pop.generation)

        # 2. Breed Code
        logger.debug("Step 4: Generating offspring...")
        code_offsprings = self._breed_code(
            code_pop,
            test_pop,
            interactions.gen_exec_results,
            interactions.gen_obs_matrix,
            len(code_elites),
        )

        # 3. Breed Tests
        test_offsprings = self._breed_tests(
            test_pop,
            code_pop,
            interactions.gen_exec_results,
            interactions.gen_obs_matrix,
            len(test_elites),
        )

        # 4. Transition Code Population
        logger.debug("Step 5: Setting next generation...")
        new_code_gen = code_elites + code_offsprings
        self._notify_removed_individuals(code_pop, new_code_gen)
        code_pop.set_next_generation(new_code_gen)

        # 5. Transition Test Population
        new_test_gen = test_elites + test_offsprings
        self._notify_removed_individuals(test_pop, new_test_gen)
        test_pop.set_next_generation(new_test_gen)

        return code_pop, test_pop

    # =========================================================================
    # Lifecycle Phase 3: Finalization
    # =========================================================================

    def _finalize_evolution(
        self,
        code_pop: CodePopulation,
        test_pop: TestPopulation,
        pub_pop: TestPopulation,
        priv_pop: TestPopulation,
    ) -> None:
        """Runs final private evaluation and logs closing stats."""
        # Run final private tests
        _, priv_obs = self._get_exec_results_and_obs_matrix(code_pop, priv_pop)

        logging_utils.log_observation_matrix(priv_obs, code_pop, priv_pop, "private")

        # Log final survivors
        logging_utils.log_final_survivors(code_pop, test_pop)
        logging_utils.log_section_header("INFO", "CO-EVOLUTION RUN FINISHED.")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _set_random_seed(self, seed: int) -> None:
        """
        Helper to set random seeds for reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"Random seed set to {seed} for reproducibility.")

    def _create_initial_code_population(self) -> CodePopulation:
        """
        Uses the injected operators to create the initial code population
        """
        logger.info("Creating all initial populations...")

        # --- 1. Code Population ---
        code_snippets = self.code_operator.create_initial_snippets(
            self.code_pop_config.initial_population_size
        )
        code_individuals = [
            CodeIndividual(
                snippet=s,
                probability=self.code_pop_config.initial_prior,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parent_ids=[],
            )
            for s in code_snippets
        ]
        code_population = CodePopulation(code_individuals, generation=0)
        logger.info(f"Created {code_population!r}")
        return code_population

    def _create_initial_test_populations(self) -> TestPopulation:
        """
        Uses the injected operators to create the initial test population
        """
        test_snippets, test_block = self.test_operator.create_initial_snippets(
            self.test_pop_config.initial_population_size
        )
        test_individuals = [
            TestIndividual(
                snippet=s,
                probability=self.test_pop_config.initial_prior,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parent_ids=[],
            )
            for s in test_snippets
        ]
        test_population = TestPopulation(
            individuals=test_individuals,
            pareto=self.pareto,
            test_block_rebuilder=self.test_block_rebuilder,
            test_class_block=test_block,
            generation=0,
        )
        logger.info(f"Created {test_population!r}")
        return test_population

    def _create_fixed_test_population_from_test_cases(
        self, test_cases: list[Test], starter_code: str
    ) -> TestPopulation:
        """
        Create a fixed test population from dataset test cases.

        This method handles the complete flow:
        1. Build test class block from dataset test cases
        2. Extract individual test methods
        3. Create TestPopulation with fixed probability

        Used for creating public and private test populations from the dataset.
        These populations have fixed probability and are not evolved.

        Args:
            test_cases: List of Test objects from the dataset (public or private)
            starter_code: The starter code for the problem

        Returns:
            TestPopulation with fixed test individuals
        """
        # Build the test class block from dataset test cases
        test_class_block = self.dataset_test_block_builder.build_test_class_block(
            test_cases, starter_code
        )

        # Extract individual test methods
        test_methods = extract_test_methods_code(test_class_block)

        # Create test individuals with fixed probability
        FIXED_TEST_PROBABILITY = 1.0
        test_individuals = [
            TestIndividual(
                snippet=method,
                probability=FIXED_TEST_PROBABILITY,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parent_ids=[],
            )
            for method in test_methods
        ]

        # Create and return the test population
        test_population = TestPopulation(
            individuals=test_individuals,
            pareto=self.pareto,
            test_block_rebuilder=self.test_block_rebuilder,
            test_class_block=test_class_block,
            generation=0,
        )
        logger.debug(
            f"Created fixed test population with {len(test_individuals)} tests "
            f"(probability={FIXED_TEST_PROBABILITY})"
        )
        return test_population

    def _select_elites(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
        observation_matrix: np.ndarray,
    ) -> tuple[list[CodeIndividual], list[TestIndividual]]:
        """
        Helper to select elite individuals from code and test populations.

        Args:
            observation_matrix: Execution results needed for Pareto front selection
        """
        # --- Code Elites ---
        num_code_elites = int(code_population.size * self.code_pop_config.elitism_rate)
        code_elites = code_population.get_top_k_individuals(num_code_elites)

        # --- Test Elites (Pareto Front) ---
        test_elites = test_population.get_pareto_front(observation_matrix)
        logger.info(
            f"Selected {len(code_elites)} code elites and {len(test_elites)} test elites (Pareto)."
        )

        return code_elites, test_elites

    def _notify_elites[IndT: CodeIndividual | TestIndividual](
        self, elites: list[IndT], generation: int
    ) -> None:
        """
        Helper to notify elite individuals about their selection.
        """
        for elite in elites:
            elite.notify_selected_as_elite(generation=generation)

    def _notify_removed_individuals[IndT: CodeIndividual | TestIndividual](
        self,
        population: BasePopulation[IndT],
        next_gen_individuals: list[IndT],
    ) -> None:
        """
        Helper to log and notify individuals that were removed from the population
        (failed to survive to the next generation).

        Logs the complete lifecycle record BEFORE notifying about death,
        ensuring we capture the complete state before any modifications.

        Args:
            population: The current population before transition
            next_gen_individuals: The list of individuals that will form the next generation
        """
        current_ids = {ind.id for ind in population}
        next_ids = {ind.id for ind in next_gen_individuals}
        removed_ids = current_ids - next_ids

        if removed_ids:
            logger.debug(
                f"Logging and notifying {len(removed_ids)} removed individuals about death in gen {population.generation}"
            )
            for ind in population:
                if ind.id in removed_ids:
                    # Log complete lifecycle record BEFORE notifying
                    logging_utils.log_individual_complete(ind, "DIED")
                    # Then notify the individual
                    ind.notify_died(generation=population.generation)

    def _breed_code(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
        exec_results: ExecutionResults,
        observation_matrix: np.ndarray,
        num_elites: int = 0,
    ) -> list[CodeIndividual]:
        """
        Helper to breed code offspring using the code breeder.

        Uses ThreadPoolExecutor for parallel breeding when max_workers > 1.
        With max_workers=1, falls back to sequential execution automatically.
        """
        num_code_offspring = min(
            int(
                self.code_pop_config.max_population_size
                * self.code_pop_config.offspring_rate
            ),
            self.code_pop_config.max_population_size - num_elites,
        )

        def breed_single_offspring() -> CodeIndividual:
            """Helper function to breed a single offspring."""
            return self.code_breeder.generate_single_offspring(
                population=code_population,
                other_population=test_population,
                execution_results=exec_results,
                feedback_generator=self.code_feedback_gen,
                observation_matrix=observation_matrix,
                operation_rates=self.code_op_rates_config,
            )

        code_offspring: list[CodeIndividual] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all breeding tasks
            futures = [
                executor.submit(breed_single_offspring)
                for _ in range(num_code_offspring)
            ]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    offspring = future.result()
                    code_offspring.append(offspring)
                except Exception as e:
                    logger.error(f"Failed to breed code offspring: {e}")
                    # Re-raise to let the orchestrator handle it
                    raise

        logger.debug(f"Successfully bred {len(code_offspring)} code offspring")
        return code_offspring

    def _breed_tests(
        self,
        test_population: TestPopulation,
        code_population: CodePopulation,
        gen_exec_results: ExecutionResults,
        gen_observation_matrix: np.ndarray,
        num_elites: int = 0,
    ) -> list[TestIndividual]:
        """
        Helper to breed test offspring using the test breeder.

        Uses ThreadPoolExecutor for parallel breeding when max_workers > 1.
        With max_workers=1, falls back to sequential execution automatically.
        """
        num_test_offspring = self.test_pop_config.initial_population_size - num_elites

        def breed_single_offspring() -> TestIndividual:
            """Helper function to breed a single offspring."""
            return self.test_breeder.generate_single_offspring(
                population=test_population,
                other_population=code_population,
                execution_results=gen_exec_results,
                feedback_generator=self.test_feedback_gen,
                observation_matrix=gen_observation_matrix,
                operation_rates=self.test_op_rates_config,
            )

        test_offsprings: list[TestIndividual] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all breeding tasks
            futures = [
                executor.submit(breed_single_offspring)
                for _ in range(num_test_offspring)
            ]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    offspring = future.result()
                    test_offsprings.append(offspring)
                except Exception as e:
                    logger.error(f"Failed to breed test offspring: {e}")
                    # Re-raise to let the orchestrator handle it
                    raise

        logger.debug(f"Successfully bred {len(test_offsprings)} test offspring")
        return test_offsprings

    def _get_exec_results_and_obs_matrix(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
    ) -> tuple[ExecutionResults, np.ndarray]:
        """
        Helper to execute tests and build observation matrix.
        """
        exec_results = self.execution_system.execute_tests(
            code_population, test_population, self.sandbox
        )
        obs_matrix = self.execution_system.build_observation_matrix(
            code_population, test_population, exec_results
        )

        return exec_results, obs_matrix

    def _get_mask_matrices(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
        public_test_population: TestPopulation,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper to get the mask matrices for code and test populations
        based on generation born sequences.

        Args:
            code_population: The current code population
            test_population: The current test population
            public_test_population: The public test population
        Returns:
            Tuple of (code_mask_matrix_generated, code_mask_matrix_public, test_mask_matrix)
        """
        code_mask_matrix_generated = (
            self.code_bayesian_system.get_code_update_mask_generation(
                updating_ind_born_generations=[
                    ind.generation_born for ind in code_population
                ],
                other_ind_born_generations=[
                    ind.generation_born for ind in test_population
                ],
                current_generation=code_population.generation,
            )
        )

        code_mask_matrix_public = (
            self.code_bayesian_system.get_code_update_mask_generation(
                updating_ind_born_generations=[
                    ind.generation_born for ind in code_population
                ],
                other_ind_born_generations=[
                    ind.generation_born for ind in public_test_population
                ],
                current_generation=code_population.generation,
            )
        )

        test_mask_matrix = self.test_bayesian_system.get_test_update_mask_generation(
            updating_ind_born_generations=[
                ind.generation_born for ind in test_population
            ],
            other_ind_born_generations=[ind.generation_born for ind in code_population],
            current_generation=test_population.generation,
        )

        return (
            code_mask_matrix_generated,
            code_mask_matrix_public,
            test_mask_matrix,
        )
