# src/common/coevolution/orchestrator.py

import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from loguru import logger

from common.code_preprocessing.transformation import extract_test_methods_code
from common.coevolution import logging_utils

# Import concrete classes
from .individual import CodeIndividual, TestIndividual

# Import all interfaces, configs, and types
from .interfaces import (
    OPERATION_INITIAL,
    BasePopulation,
    BayesianConfig,
    BreedingContext,
    CodePopulationConfig,
    CoevolutionContext,
    EvolutionConfig,
    ExecutionResults,
    IBayesianSystem,
    IBreedingStrategy,
    ICodeOperator,
    IDatasetTestBlockBuilder,
    IEliteSelector,
    IExecutionSystem,
    InteractionData,
    ITestBlockRebuilder,
    ITestOperator,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
    Sandbox,
    Test,
)
from .population import CodePopulation, TestPopulation


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
        code_op_rates_config: OperatorRatesConfig,
        bayesian_config: BayesianConfig,
        # --- Per-Test-Type Configuration (dict[test_type, config]) ---
        test_pop_configs: dict[str, PopulationConfig],
        test_op_rates_configs: dict[str, OperatorRatesConfig],
        # --- Injected Components ---
        problem: Problem,
        sandbox: Sandbox,
        code_operator: ICodeOperator,
        execution_system: IExecutionSystem,
        bayesian_system: IBayesianSystem,
        test_block_rebuilder: ITestBlockRebuilder,
        dataset_test_block_builder: IDatasetTestBlockBuilder,
        # --- Code Breeding Strategy & Elite Selection ---
        code_breeding_strategy: IBreedingStrategy[CodeIndividual],
        code_elite_selector: IEliteSelector,
        # --- Per-Test-Type Components (dict[test_type, component]) ---
        test_operators: dict[str, ITestOperator],
        test_breeding_strategies: dict[str, IBreedingStrategy[TestIndividual]],
        test_elite_selectors: dict[str, IEliteSelector],
    ) -> None:
        """
        Initializes the orchestrator by storing all injected dependencies.

        Uses dependency injection for all strategies, allowing complete flexibility
        in breeding, selection, and evolution logic without modifying core orchestrator.

        Supports multiple test population types with different operators, breeding strategies,
        and configurations. Test populations are automatically categorized as:
        - Fixed: "public" and "private" (always fixed, used for evaluation/anchoring)
        - Evolved: Any test type with operators/strategies configured (e.g., "unittest", "differential")

        Args:
            test_operators: Map test_type → operator for that test type
                           Keys determine which test types will be evolved
            test_breeding_strategies: Map test_type → breeding strategy for that type
            test_elite_selectors: Map test_type → elite selector for that type
            test_pop_configs: Map test_type → population config for that type
            test_op_rates_configs: Map test_type → operation rates config for that type
            code_breeding_strategy: Strategy for generating code offspring
            code_elite_selector: Strategy for selecting code elites
            ... (other args documented inline)
        """
        logger.info("Initializing Orchestrator...")

        # --- Store Configs ---
        self.evo_config = evo_config
        self.code_pop_config = code_pop_config
        self.code_op_rates_config = code_op_rates_config
        self.bayesian_config = bayesian_config

        # Store test configs per type
        self.test_pop_configs = test_pop_configs
        self.test_op_rates_configs = test_op_rates_configs

        # --- Store max_workers for parallel breeding ---
        self.max_workers = evo_config.max_workers
        logger.info(f"Parallel breeding configured with max_workers={self.max_workers}")

        # --- Store Problem & Sandbox ---
        self.problem = problem
        self.sandbox = sandbox

        # --- Infer Test Population Types ---
        # Fixed populations are always public and private
        self.fixed_test_types = {"public", "private"}

        # Evolved populations are any that have operators/strategies configured
        self.evolved_test_types = set(test_operators.keys())

        # Validate that fixed and evolved don't overlap
        overlap = self.fixed_test_types & self.evolved_test_types
        if overlap:
            raise ValueError(
                f"Test types {overlap} cannot be both fixed and evolved. "
                f"'public' and 'private' are reserved for fixed test populations."
            )

        logger.info(
            f"Test population types - Evolved: {self.evolved_test_types}, Fixed: {self.fixed_test_types}"
        )

        # --- Store Injected Components ---
        self.code_operator = code_operator
        self.test_operators = test_operators
        self.execution_system = execution_system
        self.bayesian_system = bayesian_system

        self.test_block_rebuilder = test_block_rebuilder
        self.dataset_test_block_builder = dataset_test_block_builder

        # --- Store Breeding Strategies & Elite Selectors ---
        self.code_breeding_strategy = code_breeding_strategy
        self.code_elite_selector = code_elite_selector

        self.test_breeding_strategies = test_breeding_strategies
        self.test_elite_selectors = test_elite_selectors

        # --- Validate Configuration ---
        # Ensure that all operations in the rate configs are supported by operators
        self.code_op_rates_config.validate_against_operator(
            code_operator, "code_operator"
        )

        # Validate each test type's rates config against its operator
        for test_type in self.evolved_test_types:
            if test_type not in test_operators:
                raise ValueError(
                    f"Missing operator for evolved test type '{test_type}'"
                )
            if test_type not in test_op_rates_configs:
                raise ValueError(
                    f"Missing op_rates_config for evolved test type '{test_type}'"
                )
            if test_type not in test_breeding_strategies:
                raise ValueError(
                    f"Missing breeding_strategy for evolved test type '{test_type}'"
                )
            if test_type not in test_elite_selectors:
                raise ValueError(
                    f"Missing elite_selector for evolved test type '{test_type}'"
                )
            if test_type not in test_pop_configs:
                raise ValueError(
                    f"Missing pop_config for evolved test type '{test_type}'"
                )

            test_op_rates_configs[test_type].validate_against_operator(
                test_operators[test_type], f"test_operator['{test_type}']"
            )

        self._set_random_seed(evo_config.random_seed)

    def run(self) -> tuple[CodePopulation, dict[str, TestPopulation]]:
        """
        Runs the main co-evolutionary loop with multiple test populations:
        - Fixed populations: "public" (anchoring), "private" (evaluation only)
        - Evolved populations: Any configured (e.g., "unittest", "differential")

        Returns:
            Tuple of (code_population, evolved_test_populations_dict)
        """
        logging_utils.log_section_header("INFO", "STARTING CO-EVOLUTION RUN")

        # 1. Initialization
        code_pop, evolved_test_pops, public_pop, private_pop = (
            self._initialize_evolution()
        )

        # 2. Main Loop
        for gen in range(self.evo_config.num_generations + 1):
            logging_utils.log_subsection_header(
                "INFO",
                f"GENERATION {gen} / {self.evo_config.num_generations}",
            )

            # A. Execute - returns new interaction data (pure function)
            interactions = self._execute_all_interactions(
                code_pop, evolved_test_pops, public_pop
            )

            # Build context with current populations and new interactions
            context = CoevolutionContext(
                code_population=code_pop,
                test_populations={**evolved_test_pops, "public": public_pop},
                interactions=interactions,
            )

            # B. Update Beliefs - mutates population probabilities in context
            context = self._perform_cooperative_updates(context)

            # C. Evolve - mutates population individuals in context
            if gen < self.evo_config.num_generations:
                context = self._produce_next_generation(context)

            # D. Log generation summary
            logging_utils.log_generation_summary(
                context.code_population,
                {
                    k: v
                    for k, v in context.test_populations.items()
                    if k in self.evolved_test_types
                },
            )

        # 3. Finalization
        code_pop = context.code_population
        evolved_test_pops = {
            k: v
            for k, v in context.test_populations.items()
            if k in self.evolved_test_types
        }
        self._finalize_evolution(code_pop, evolved_test_pops, public_pop, private_pop)

        return code_pop, evolved_test_pops

    # =========================================================================
    # Lifecycle Phase 1: Initialization
    # =========================================================================

    def _initialize_evolution(
        self,
    ) -> tuple[
        CodePopulation, dict[str, TestPopulation], TestPopulation, TestPopulation
    ]:
        """Creates initial populations and logs initial state.

        Returns:
            Tuple of (code_pop, evolved_test_pops_dict, public_pop, private_pop)
        """

        # Create Code Population
        code_pop = self._create_initial_code_population()

        # Create Evolved Test Populations
        evolved_test_pops: dict[str, TestPopulation] = {}
        for test_type in self.evolved_test_types:
            evolved_test_pops[test_type] = self._create_initial_test_population(
                test_type
            )

        # Create Fixed Test Populations
        public_pop = self._create_fixed_test_population_from_test_cases(
            self.problem.public_test_cases, self.problem.starter_code
        )
        private_pop = self._create_fixed_test_population_from_test_cases(
            self.problem.private_test_cases, self.problem.starter_code
        )

        # Initial Private Run (Baseline)
        _, private_obs = self._get_exec_results_and_obs_matrix(code_pop, private_pop)

        logging_utils.log_observation_matrix(
            private_obs, code_pop, private_pop, "private"
        )

        # Log generation 0 for all evolved test populations
        logging_utils.log_generation_summary(code_pop, evolved_test_pops)

        return code_pop, evolved_test_pops, public_pop, private_pop

    # =========================================================================
    # Lifecycle Phase 2A: Execution
    # =========================================================================

    def _execute_all_interactions(
        self,
        code_pop: CodePopulation,
        evolved_test_pops: dict[str, TestPopulation],
        public_pop: TestPopulation,
    ) -> dict[str, InteractionData]:
        """Executes code against all test populations, returns interaction data.

        This is a pure function that runs tests and collects results without
        mutating any populations.

        Args:
            code_pop: Code population to execute
            evolved_test_pops: Dict of evolved test populations to execute against
            public_pop: Public (fixed) test population for anchoring

        Returns:
            Dictionary mapping test type to InteractionData (execution results + observation matrix)
        """
        logger.debug("Step 1: Executing all test sets...")

        interactions: dict[str, InteractionData] = {}

        # Execute evolved test populations
        for test_type, test_pop in evolved_test_pops.items():
            exec_results, obs_matrix = self._get_exec_results_and_obs_matrix(
                code_pop, test_pop
            )
            interactions[test_type] = InteractionData(exec_results, obs_matrix)
            logging_utils.log_observation_matrix(
                obs_matrix,
                code_pop,
                test_pop,
                test_type,
            )

        # Execute public (fixed) test population
        public_exec, public_obs = self._get_exec_results_and_obs_matrix(
            code_pop, public_pop
        )
        interactions["public"] = InteractionData(public_exec, public_obs)
        logging_utils.log_observation_matrix(public_obs, code_pop, public_pop, "public")

        return interactions

    # =========================================================================
    # Lifecycle Phase 2B: Belief Updates (The Core Logic)
    # =========================================================================

    def _perform_cooperative_updates(
        self,
        context: CoevolutionContext,
    ) -> CoevolutionContext:
        """
        Performs the Bayesian belief updates in a specific order to prevent
        confirmation bias.

        Strategy:
        1. Masking: Determine which individuals should be updated based on their
           generation born and the current generation.
        2. Anchor: Update Code using Public Tests (Ground Truth).
        3. Calculate: Compute updates for each evolved test population using their observations.
           - Tests judge Code (that has seen Public tests).
           - Code judges Tests (using Test Priors).
        4. Apply: Apply updates simultaneously.

        MUTATES:
            - context.code_population.probabilities
            - context.test_populations[*].probabilities (for evolved types)

        Args:
            context: Contains populations and interaction data

        Returns:
            The same context (with mutated populations) to signal modification occurred

        Note: Currently updates each evolved test population independently against code.
        Future enhancement could support cross-population updates.
        """
        logger.debug("Step 2: Updating beliefs...")

        # Extract populations and interactions
        code_pop = context.code_population
        public_pop = context.test_populations["public"]
        public_obs_matrix = context.interactions["public"].observation_matrix

        # 1. Get Masks for public tests
        logger.debug("Getting mask matrices for updates...")
        # Use first evolved test pop for now to get the test mask template
        first_test_type = next(iter(self.evolved_test_types))
        first_test_pop = context.test_populations[first_test_type]
        code_mask_unittest, code_mask_public, _ = self._get_mask_matrices(
            code_pop, first_test_pop, public_pop
        )

        # 2. ANCHOR: Update Code based on Public Tests first
        logger.debug("Anchoring code beliefs with public tests...")
        code_post_public = self.bayesian_system.update_code_beliefs(
            prior_code_probs=code_pop.probabilities,
            prior_test_probs=public_pop.probabilities,
            observation_matrix=public_obs_matrix,
            code_update_mask_matrix=code_mask_public,
            config=self.bayesian_config,
        )
        code_pop.update_probabilities(code_post_public)

        # 3 & 4. For each evolved test population: Calculate and Apply updates
        for test_type in self.evolved_test_types:
            test_pop = context.test_populations[test_type]
            test_obs_matrix = context.interactions[test_type].observation_matrix

            # Get mask for this specific test population
            _, _, test_mask = self._get_mask_matrices(code_pop, test_pop, public_pop)

            logger.debug(
                f"Calculating {test_type} test updates based on updated code beliefs..."
            )
            test_post = self.bayesian_system.update_test_beliefs(
                prior_code_probs=code_pop.probabilities,
                prior_test_probs=test_pop.probabilities,
                observation_matrix=test_obs_matrix,
                test_update_mask_matrix=test_mask,
                config=self.bayesian_config,
            )

            logger.debug(
                f"Calculating code updates based on {test_type} test priors..."
            )
            code_post = self.bayesian_system.update_code_beliefs(
                prior_code_probs=code_pop.probabilities,
                prior_test_probs=test_pop.probabilities,
                observation_matrix=test_obs_matrix,
                code_update_mask_matrix=code_mask_unittest,
                config=self.bayesian_config,
            )

            # Apply updates
            test_pop.update_probabilities(test_post)
            code_pop.update_probabilities(code_post)

        return context

    # =========================================================================
    # Lifecycle Phase 2C: Evolution (Selection & Breeding)
    # =========================================================================

    def _produce_next_generation(
        self,
        context: CoevolutionContext,
    ) -> CoevolutionContext:
        """
        Handles the full evolution cycle: Selection, Breeding, and Transition.

        MUTATES:
            - context.code_population.individuals (sets next generation)
            - context.test_populations[*].individuals (for evolved types)

        Args:
            context: Contains populations, interaction data, and configurations

        Returns:
            The same context (with mutated populations) to signal modification occurred
        """
        code_pop = context.code_population

        # 1. Select Elites
        logger.debug("Step 3: Selecting elites...")
        code_elites, test_elites_dict = self._select_elites(context)
        self._notify_elites(code_elites, code_pop.generation)

        # Notify test elites for each type
        for test_type, test_elites in test_elites_dict.items():
            test_pop = context.test_populations[test_type]
            self._notify_elites(test_elites, test_pop.generation)

        # 2. Breed Code
        logger.debug("Step 4: Generating offspring...")
        code_offsprings = self._breed_code(context, len(code_elites))

        # 3. Breed Tests for each evolved type
        test_offsprings_dict: dict[str, list[TestIndividual]] = {}
        for test_type in self.evolved_test_types:
            test_elites = test_elites_dict[test_type]
            test_offsprings_dict[test_type] = self._breed_tests(
                context, test_type, len(test_elites)
            )

        # 4. Transition Code Population
        logger.debug("Step 5: Setting next generation...")
        new_code_gen = code_elites + code_offsprings
        self._notify_removed_individuals(code_pop, new_code_gen)
        code_pop.set_next_generation(new_code_gen)

        # 5. Transition Test Populations
        for test_type in self.evolved_test_types:
            test_pop = context.test_populations[test_type]
            test_elites = test_elites_dict[test_type]
            test_offsprings = test_offsprings_dict[test_type]

            new_test_gen = test_elites + test_offsprings
            self._notify_removed_individuals(test_pop, new_test_gen)
            test_pop.set_next_generation(new_test_gen)

        return context

    # =========================================================================
    # Lifecycle Phase 3: Finalization
    # =========================================================================

    def _finalize_evolution(
        self,
        code_pop: CodePopulation,
        evolved_test_pops: dict[str, TestPopulation],
        public_pop: TestPopulation,
        private_pop: TestPopulation,
    ) -> None:
        """Runs final private evaluation and logs closing stats."""
        # Run final private tests
        _, private_obs = self._get_exec_results_and_obs_matrix(code_pop, private_pop)

        logging_utils.log_observation_matrix(
            private_obs, code_pop, private_pop, "private"
        )

        # Log final survivors for the first evolved test population (for now)
        first_test_pop = next(iter(evolved_test_pops.values()))
        logging_utils.log_final_survivors(code_pop, first_test_pop)
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
                parents={},
            )
            for s in code_snippets
        ]
        code_population = CodePopulation(code_individuals, generation=0)
        logger.info(f"Created {code_population!r}")
        return code_population

    def _create_initial_test_population(self, test_type: str) -> TestPopulation:
        """
        Uses the injected operators to create the initial test population for a specific type.

        Args:
            test_type: The type of test population to create (e.g., "unittest", "differential")
        """
        test_operator = self.test_operators[test_type]
        test_pop_config = self.test_pop_configs[test_type]

        test_snippets, test_block = test_operator.create_initial_snippets(
            test_pop_config.initial_population_size
        )
        test_individuals = [
            TestIndividual(
                snippet=s,
                probability=test_pop_config.initial_prior,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
            )
            for s in test_snippets
        ]
        test_population = TestPopulation(
            individuals=test_individuals,
            test_block_rebuilder=self.test_block_rebuilder,
            test_class_block=test_block,
            generation=0,
        )
        logger.info(f"Created {test_type} {test_population!r}")
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
                parents={},
            )
            for method in test_methods
        ]

        # Create and return the test population
        test_population = TestPopulation(
            individuals=test_individuals,
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
        context: CoevolutionContext,
    ) -> tuple[list[CodeIndividual], dict[str, list[TestIndividual]]]:
        """
        Helper to select elite individuals using injected elite selectors.

        Uses:
        - code_elite_selector for code population (e.g., top-k by probability)
        - test_elite_selectors for each evolved test population type

        Returns:
            Tuple of (code_elites, test_elites_dict) where test_elites_dict maps
            test_type → list of elite individuals for that type
        """
        code_pop = context.code_population

        # --- Code Elites ---
        code_elite_indices = self.code_elite_selector.select_elites(
            coevolution_context=context,
            target_population_type="code",
            population_config=self.code_pop_config,
        )
        code_elites = [code_pop[i] for i in code_elite_indices]

        # --- Test Elites for each evolved type ---
        test_elites_dict: dict[str, list[TestIndividual]] = {}
        for test_type in self.evolved_test_types:
            test_pop = context.test_populations[test_type]
            test_elite_selector = self.test_elite_selectors[test_type]
            test_pop_config = self.test_pop_configs[test_type]

            test_elite_indices = test_elite_selector.select_elites(
                coevolution_context=context,
                target_population_type=test_type,
                population_config=test_pop_config,
            )
            test_elites_dict[test_type] = [test_pop[i] for i in test_elite_indices]

            logger.info(
                f"Selected {len(test_elites_dict[test_type])} elites for {test_type} population"
            )

        logger.info(
            f"Selected {len(code_elites)} code elites and {sum(len(elites) for elites in test_elites_dict.values())} total test elites across {len(test_elites_dict)} test types"
        )

        return code_elites, test_elites_dict

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
        context: CoevolutionContext,
        num_elites: int = 0,
    ) -> list[CodeIndividual]:
        """
        Helper to breed code offspring using the injected code breeding strategy.

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

        # Create BreedingContext for code breeding
        breeding_context = BreedingContext(
            coevolution_context=context,
            rates_config=self.code_op_rates_config,
            target_population_type="code",
        )

        def breed_single_offspring() -> CodeIndividual:
            """Helper function to breed a single offspring."""
            return self.code_breeding_strategy.generate_offspring(breeding_context)

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
        context: CoevolutionContext,
        test_type: str,
        num_elites: int = 0,
    ) -> list[TestIndividual]:
        """
        Helper to breed test offspring using the injected test breeding strategy for a specific type.

        Args:
            context: Complete coevolution context
            test_type: The type of test population to breed (e.g., "unittest", "differential")
            num_elites: Number of elites (to calculate how many offspring to generate)

        Uses ThreadPoolExecutor for parallel breeding when max_workers > 1.
        With max_workers=1, falls back to sequential execution automatically.
        """
        test_pop_config = self.test_pop_configs[test_type]
        test_breeding_strategy = self.test_breeding_strategies[test_type]
        test_op_rates_config = self.test_op_rates_configs[test_type]

        num_test_offspring = test_pop_config.initial_population_size - num_elites

        # Create BreedingContext for test breeding
        breeding_context = BreedingContext(
            coevolution_context=context,
            rates_config=test_op_rates_config,
            target_population_type=test_type,
        )

        def breed_single_offspring() -> TestIndividual:
            """Helper function to breed a single offspring."""
            return test_breeding_strategy.generate_offspring(breeding_context)

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
        unittest_population: TestPopulation,
        public_test_population: TestPopulation,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper to get the mask matrices for code and test populations
        based on generation born sequences.

        Args:
            code_population: The current code population
            unittest_population: The current unittest population (evolved tests)
            public_test_population: The public test population (ground truth)
        Returns:
            Tuple of (code_mask_matrix_unittest, code_mask_matrix_public, test_mask_matrix)
        """
        code_mask_matrix_unittest = (
            self.bayesian_system.get_code_update_mask_generation(
                updating_ind_born_generations=[
                    ind.generation_born for ind in code_population
                ],
                other_ind_born_generations=[
                    ind.generation_born for ind in unittest_population
                ],
                current_generation=code_population.generation,
            )
        )

        code_mask_matrix_public = self.bayesian_system.get_code_update_mask_generation(
            updating_ind_born_generations=[
                ind.generation_born for ind in code_population
            ],
            other_ind_born_generations=[
                ind.generation_born for ind in public_test_population
            ],
            current_generation=code_population.generation,
        )

        test_mask_matrix = self.bayesian_system.get_test_update_mask_generation(
            updating_ind_born_generations=[
                ind.generation_born for ind in unittest_population
            ],
            other_ind_born_generations=[ind.generation_born for ind in code_population],
            current_generation=unittest_population.generation,
        )

        return (
            code_mask_matrix_unittest,
            code_mask_matrix_public,
            test_mask_matrix,
        )
