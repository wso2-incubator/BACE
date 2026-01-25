# src/coevolution/orchestrator.py

import numpy as np
from loguru import logger

from coevolution.utils import logging as logging_utils
from infrastructure.code_preprocessing.transformation import extract_test_methods_code

# Import concrete classes
from .individual import CodeIndividual, TestIndividual

# Import all interfaces, configs, and types
from .interfaces import (
    OPERATION_INITIAL,
    BasePopulation,
    CodeProfile,
    CoevolutionContext,
    EvolutionConfig,
    IBeliefUpdater,
    IDatasetTestBlockBuilder,
    IExecutionSystem,
    IInteractionLedger,
    InteractionData,
    ITestBlockRebuilder,
    LedgerFactory,
    Problem,
    PublicTestProfile,
    Test,
    TestProfile,
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
        # --- Population Profiles ---
        code_profile: CodeProfile,
        evolved_test_profiles: dict[str, TestProfile],
        public_test_profile: PublicTestProfile,
        # --- Global Infrastructure ---
        execution_system: IExecutionSystem,
        bayesian_system: IBeliefUpdater,
        test_block_rebuilder: ITestBlockRebuilder,
        dataset_test_block_builder: IDatasetTestBlockBuilder,
        ledger_factory: LedgerFactory,
    ) -> None:
        """
        Initializes the orchestrator by storing all injected dependencies.

        Uses dependency injection for all strategies, allowing complete flexibility
        in breeding, selection, and evolution logic without modifying core orchestrator.

        Supports multiple test population types with different breeding strategies
        and configurations. Test populations are automatically categorized as:
        - Fixed: "public" and "private" (always fixed, used for evaluation/anchoring)
        - Evolved: Any test type with profiles configured (e.g., "unittest", "differential")

        Note: Operators AND their configurations (operator rates) are owned by breeding
        strategies, not by orchestrator. Orchestrator only interacts with breeding strategies,
        which delegate to operators and manage their own operation rate configs internally.

        Important: Test Type Update Order
        The order in which evolved test types update probabilities during cooperative updates
        is determined by the insertion order of keys in evolved_test_profiles. Python 3.7+
        guarantees dict maintains insertion order, so the order you pass profiles controls
        the update sequence. Example:
            evolved_test_profiles = {"differential": ..., "unittest": ...}
            # Updates in order: public (anchor) → differential → unittest

        Args:
            evo_config: Top-level evolution configuration
            code_profile: Complete profile for code population (config + strategies)
            evolved_test_profiles: Map test_type → profile for that evolved test type.
                The order of keys determines the order of probability updates.
            public_test_profile: Profile for public/ground-truth tests
            execution_system: System for executing code against tests
            bayesian_system: System for belief updates
            test_block_rebuilder: Rebuilds test class blocks
            dataset_test_block_builder: Builds test blocks from dataset
        """
        logger.info("Initializing Orchestrator...")

        # --- Store Profiles ---
        self.code_profile = code_profile
        self.evolved_test_profiles = evolved_test_profiles
        self.public_test_profile = public_test_profile

        # --- Store Top-Level Config ---
        self.evo_config = evo_config

        # --- Infer Test Population Types ---
        # Fixed populations are always public and private
        self.fixed_test_types = {"public", "private"}

        # Evolved populations are any that have profiles configured
        self.evolved_test_types = set(evolved_test_profiles.keys())

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
        self.execution_system = execution_system
        self.bayesian_system = bayesian_system
        self.ledger_factory = ledger_factory

        self.test_block_rebuilder = test_block_rebuilder
        self.dataset_test_block_builder = dataset_test_block_builder

    def run(self, problem: Problem) -> tuple[CodePopulation, dict[str, TestPopulation]]:
        """
        Runs the main co-evolutionary loop with multiple test populations:
        - Fixed populations: "public" (anchoring), "private" (evaluation only)
        - Evolved populations: Any configured (e.g., "unittest", "differential")

        Args:
            problem: The problem to solve. Orchestrator is stateless and can work on
                     different problems by calling run() with different problem instances.

        Returns:
            Tuple of (code_population, evolved_test_populations_dict)
        """
        logging_utils.log_section_header("INFO", "STARTING CO-EVOLUTION RUN")

        # 1. Initialization
        code_pop, evolved_test_pops, public_pop, private_pop = (
            self._initialize_evolution(problem)
        )

        # Create ledger for tracking interactions
        ledger = self.ledger_factory()

        # Global counters
        global_gen = 0
        total_gens = self.evo_config.num_generations - 1
        schedule = self.evo_config.schedule

        # 2. Main Loop: Iterate over Phases -> Generations
        for phase in schedule.phases:
            logger.info(
                f"=== Starting Phase: {phase.name} (Duration: {phase.duration}) ==="
            )
            logger.info(
                f"    [Rules] Evolve Code: {phase.evolve_code} | Evolve Tests: {phase.evolve_tests}"
            )

            for _ in range(phase.duration):
                logging_utils.log_subsection_header(
                    "INFO",
                    f"GENERATION {global_gen} / {total_gens} [Phase: {phase.name}]",
                )

                # A. Execute - returns new interaction data (pure function)
                interactions = self._execute_all_interactions(
                    code_pop, evolved_test_pops, public_pop
                )

                # Build context with current populations and new interactions
                context = CoevolutionContext(
                    problem=problem,
                    code_population=code_pop,
                    test_populations={**evolved_test_pops, "public": public_pop},
                    interactions=interactions,
                )

                # B. Update Beliefs - mutates population probabilities in context
                self._perform_fitness_based_updates(
                    context, ledger
                )  # FIXME: ABLATION STUDY

                # C. Evolve - mutates population individuals in context
                if global_gen < total_gens:
                    self._produce_next_generation(
                        context,
                        evolve_code=phase.evolve_code,
                        evolve_tests=phase.evolve_tests,
                    )

                # D. Log generation summary
                logging_utils.log_generation_summary(
                    context.code_population,
                    {
                        k: v
                        for k, v in context.test_populations.items()
                        if k in self.evolved_test_types
                    },
                )

                global_gen += 1

        # 3. Finalization
        self._finalize_evolution(code_pop, evolved_test_pops, public_pop, private_pop)

        return code_pop, evolved_test_pops

    # =========================================================================
    # Lifecycle Phase 1: Initialization
    # =========================================================================

    def _initialize_evolution(
        self,
        problem: Problem,
    ) -> tuple[
        CodePopulation, dict[str, TestPopulation], TestPopulation, TestPopulation
    ]:
        """Creates initial populations and logs initial state.

        Args:
            problem: The problem to solve

        Returns:
            Tuple of (code_pop, evolved_test_pops_dict, public_pop, private_pop)
        """

        # Create Code Population
        code_pop = self._create_initial_code_population(problem)

        # Create Evolved Test Populations
        evolved_test_pops: dict[str, TestPopulation] = {}
        for test_type in self.evolved_test_types:
            evolved_test_pops[test_type] = self._create_initial_test_population(
                test_type, problem
            )

        # Create Fixed Test Populations
        public_pop = self._create_fixed_test_population_from_test_cases(
            problem.public_test_cases, problem.starter_code
        )
        private_pop = self._create_fixed_test_population_from_test_cases(
            problem.private_test_cases, problem.starter_code
        )

        # Initial Private Run (Baseline) - executor returns atomic InteractionData
        private_interaction = self._get_interaction_data(code_pop, private_pop)
        logging_utils.log_observation_matrix(
            private_interaction.observation_matrix, code_pop, private_pop, "private"
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
            interaction = self.execution_system.execute_tests(code_pop, test_pop)
            interactions[test_type] = interaction
            logging_utils.log_observation_matrix(
                interaction.observation_matrix,
                code_pop,
                test_pop,
                test_type,
            )

        # Execute public (fixed) test population
        public_interaction = self.execution_system.execute_tests(code_pop, public_pop)
        interactions["public"] = public_interaction
        logging_utils.log_observation_matrix(
            public_interaction.observation_matrix, code_pop, public_pop, "public"
        )

        return interactions

    # ============================================
    # ABLATION STUDY: NO COOPERATIVE UPDATES, JUST FITNESS SCORES
    # +=========================================
    def _perform_fitness_based_updates(
        self,
        context: CoevolutionContext,
        ledger: IInteractionLedger,
    ) -> None:
        """
        Performs fitness-based updates without cooperative Bayesian updates.

        This method updates the probabilities of code and test individuals
        based solely on their fitness scores derived from interactions.

        MUTATES populations in the context:
            - context.code_population.probabilities
            - context.test_populations[*].probabilities (for evolved types)
        Args:
            context: Contains populations and interaction data. Populations
                    are mutated in-place.
            lsedger: Ledger to track interactions. # NOT USED IN THIS METHOD
        """
        logger.debug("Step 2: Updating beliefs based on fitness scores...")

        # Extract populations and interactions
        code_pop = context.code_population

        # Update Code probabilities based on fitness scores
        code_fitness_scores = np.array(
            [
                sum(
                    1
                    for test_type, interaction in context.interactions.items()
                    for test_idx, test_ind in enumerate(
                        context.test_populations[test_type].individuals
                    )
                    if interaction.observation_matrix[code_idx, test_idx] == 1
                )
                for code_idx, code_ind in enumerate(code_pop.individuals)
            ]
        )
        code_pop.update_probabilities(
            code_fitness_scores / np.sum(code_fitness_scores), test_type="fitness_based"
        )
        logger.debug(
            f"Updated code probabilities based on fitness scores: {np.round(code_pop.probabilities, 4)}"
        )

        # Update Test probabilities for each evolved test population
        for test_type in self.evolved_test_profiles.keys():
            test_pop = context.test_populations[test_type]
            test_fitness_scores = np.array(
                [
                    sum(
                        1
                        for code_idx, code_ind in enumerate(code_pop.individuals)
                        if context.interactions[test_type].observation_matrix[
                            code_idx, test_idx
                        ]
                        == 1
                    )
                    for test_idx, test_ind in enumerate(test_pop.individuals)
                ]
            )
            test_pop.update_probabilities(
                test_fitness_scores / np.sum(test_fitness_scores), test_type=test_type
            )
            logger.debug(
                f"Updated {test_type} test probabilities based on fitness scores: {np.round(test_pop.probabilities, 4)}"
            )

    # =========================================================================

    # Lifecycle Phase 2B: Belief Updates (The Core Logic)
    # =========================================================================

    def _perform_cooperative_updates(
        self,
        context: CoevolutionContext,
        ledger: IInteractionLedger,
    ) -> None:
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

        MUTATES populations in the context:
            - context.code_population.probabilities
            - context.test_populations[*].probabilities (for evolved types)

        Args:
            context: Contains populations and interaction data. Populations
                    are mutated in-place.

        Note: Currently updates each evolved test population independently against code.
        Future enhancement could support cross-population updates.
        """
        logger.debug("Step 2: Updating beliefs...")

        # Extract populations and interactions
        code_pop = context.code_population
        code_ids = [ind.id for ind in code_pop.individuals]
        public_pop = context.test_populations["public"]
        public_ids = [ind.id for ind in public_pop.individuals]
        public_obs_matrix = context.interactions["public"].observation_matrix

        # 1. Get Masks - considering ALL evolved test populations (except private)
        logger.debug("Getting mask matrices for updates...")

        # Code update mask for public tests only
        mask_pub = ledger.get_new_interaction_mask(
            code_ids, public_ids, "public", "CODE"
        )

        # 2. ANCHOR: Update Code based on Public Tests first
        logger.debug("Anchoring code beliefs with public tests...")
        code_post_public = self.bayesian_system.update_code_beliefs(
            prior_code_probs=code_pop.probabilities,
            prior_test_probs=public_pop.probabilities,
            observation_matrix=public_obs_matrix,
            code_update_mask_matrix=mask_pub,
            config=self.public_test_profile.bayesian_config,
        )
        code_pop.update_probabilities(code_post_public, test_type="public")
        logger.debug(
            f"Posterior code probabilities after public update: {np.round(code_post_public, 4)}"
        )
        ledger.commit_interactions(code_ids, public_ids, "public", "CODE", mask_pub)
        # 3 & 4. For each evolved test population: Calculate and Apply updates
        # Iterate in the order profiles were provided (dict maintains insertion order in Python 3.7+)
        for test_type in self.evolved_test_profiles.keys():
            test_pop = context.test_populations[test_type]
            test_ids = [ind.id for ind in test_pop.individuals]
            test_obs_matrix = context.interactions[test_type].observation_matrix
            test_profile = self.evolved_test_profiles[test_type]

            # Get test update mask - tests updated based on ALL code individuals
            test_mask = ledger.get_new_interaction_mask(
                code_ids,
                test_ids,
                test_type,
                "TEST",
            )

            # Get code update mask for this specific test population
            code_mask_for_this_test = ledger.get_new_interaction_mask(
                code_ids,
                test_ids,
                test_type,
                "CODE",
            )

            logger.debug(
                f"Calculating {test_type} test updates based on updated code beliefs on public tests..."
            )
            test_post = self.bayesian_system.update_test_beliefs(
                prior_code_probs=code_post_public,
                prior_test_probs=test_pop.probabilities,
                observation_matrix=test_obs_matrix,
                test_update_mask_matrix=test_mask,
                config=test_profile.bayesian_config,
            )
            logger.debug(
                f"Posterior {test_type} test probabilities after update: {np.round(test_post, 4)}"
            )

            logger.debug(
                f"Calculating code updates based on {test_type} test priors..."
            )
            code_evolved_post = self.bayesian_system.update_code_beliefs(
                prior_code_probs=code_pop.probabilities,
                prior_test_probs=test_pop.probabilities,
                observation_matrix=test_obs_matrix,
                code_update_mask_matrix=code_mask_for_this_test,
                config=test_profile.bayesian_config,
            )

            logger.debug(
                f"Posterior code probabilities after {test_type} test update: {np.round(code_evolved_post, 4)}"
            )
            # Apply updates with test_type tracking
            test_pop.update_probabilities(test_post, test_type=test_type)
            code_pop.update_probabilities(code_evolved_post, test_type=test_type)
            ledger.commit_interactions(code_ids, test_ids, test_type, "TEST", test_mask)
            ledger.commit_interactions(
                code_ids, test_ids, test_type, "CODE", code_mask_for_this_test
            )

    # =========================================================================
    # Lifecycle Phase 2C: Evolution (Selection & Breeding)
    # =========================================================================

    def _produce_next_generation(
        self,
        context: CoevolutionContext,
        evolve_code: bool,
        evolve_tests: bool,
    ) -> None:
        """
        Handles the full evolution cycle: Selection, Breeding, and Transition.

        MUTATES populations in the context:
            - context.code_population.individuals (sets next generation)
            - context.test_populations[*].individuals (for evolved types)

        Args:
            context: Contains populations, interaction data, and configurations.
                    Populations are mutated in-place.
        """
        code_pop = context.code_population

        # --- 1. Evolve Code (If Active) ---
        if evolve_code:
            logger.debug("Active Phase: Evolving Code Population...")

            # Select Elites
            logger.debug("Selecting code elites...")
            code_elites = self.code_profile.elite_selector.select_elites(
                population=code_pop,
                coevolution_context=context,
                population_config=self.code_profile.population_config,
            )

            logger.debug(f"Selected {len(code_elites)} code elites.")
            self._notify_elites(code_elites, code_pop.generation)

            # Breed Offspring
            code_offsprings = self._breed_code(context, len(code_elites))
            logger.debug(f"Generated {len(code_offsprings)} code offsprings.")

            # Transition
            new_code_inds = code_elites + code_offsprings
            self._notify_removed_individuals(code_pop, new_code_inds, "code")
            code_pop.set_next_generation(new_code_inds)
            logger.info(
                f"Code Population transitioned to generation {code_pop.generation}."
            )
        else:
            logger.debug("Frozen Phase: Code Population is static this generation.")

        # --- 2. Evolve Tests (If Active) ---
        if evolve_tests:
            logger.debug("Active Phase: Evolving Test Populations...")

            # We select elites here to ensure we breed from the best *current* tests
            for test_type in self.evolved_test_types:
                logger.info(f"Evolving {test_type} test elites...")
                test_pop = context.test_populations[test_type]
                test_elites = self.evolved_test_profiles[
                    test_type
                ].elite_selector.select_elites(
                    population=test_pop,
                    coevolution_context=context,
                    population_config=self.evolved_test_profiles[
                        test_type
                    ].population_config,
                )

                logger.debug(f"Selected {len(test_elites)} {test_type} test elites.")

                # Notify Elites
                self._notify_elites(test_elites, test_pop.generation)

                # Breed
                test_offsprings = self._breed_tests(
                    context, test_type, len(test_elites)
                )
                logger.debug(
                    f"Generated {len(test_offsprings)} {test_type} test offsprings."
                )
                # Transition
                new_test_inds = test_elites + test_offsprings
                self._notify_removed_individuals(test_pop, new_test_inds, test_type)
                test_pop.set_next_generation(new_test_inds)
                logger.info(
                    f"{test_type.capitalize()} Test Population transitioned to generation {test_pop.generation}."
                )
        else:
            logger.debug("Frozen Phase: Test Populations are static this generation.")

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
        """
        Final execution and logging after all generations are complete.
        """

        self._execute_all_interactions(
            code_pop, evolved_test_pops, public_pop
        )  # Final execution to ensure latest interactions

        # Run final private tests
        private_interaction = self._get_interaction_data(code_pop, private_pop)

        logging_utils.log_observation_matrix(
            private_interaction.observation_matrix, code_pop, private_pop, "private"
        )

        logging_utils.log_final_survivors(code_pop, evolved_test_pops)
        logging_utils.log_section_header("INFO", "CO-EVOLUTION RUN FINISHED.")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _create_initial_code_population(self, problem: Problem) -> CodePopulation:
        """
        Uses the injected breeding strategy to create the initial code population

        Args:
            problem: The problem to solve
        """
        logger.info("Creating all initial populations...")

        # --- 1. Code Population ---
        code_individuals, context_code = (
            self.code_profile.breeding_strategy.initialize_individuals(
                problem,
            )
        )
        # context_code is None for code populations - can be ignored

        code_population = CodePopulation(code_individuals, generation=0)
        logger.info(f"Created {code_population!r}")
        return code_population

    def _create_initial_test_population(
        self, test_type: str, problem: Problem
    ) -> TestPopulation:
        """
        Uses the injected breeding strategy to create the initial test population for a specific type.

        Args:
            test_type: The type of test population to create (e.g., "unittest", "differential")
            problem: The problem to solve
        """
        test_profile = self.evolved_test_profiles[test_type]

        test_individuals, context_code = (
            test_profile.breeding_strategy.initialize_individuals(
                problem,
            )
        )

        # For test populations, context_code should contain the test class block
        # (imports, setUp, helpers). Breeding strategy/operator is responsible for
        # providing this, even for empty populations (e.g., with a template).
        if context_code is None:
            raise ValueError(
                f"Test breeding strategy for '{test_type}' returned None context_code - "
                f"expected test class block (even for empty populations)"
            )

        test_population = TestPopulation(
            individuals=test_individuals,
            test_block_rebuilder=self.test_block_rebuilder,
            test_class_block=context_code,
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
        code_elites = self.code_profile.elite_selector.select_elites(
            population=code_pop,
            population_config=self.code_profile.population_config,
            coevolution_context=context,
        )

        # --- Test Elites for each evolved type ---
        test_elites_dict: dict[str, list[TestIndividual]] = {}
        for test_type in self.evolved_test_types:
            test_pop = context.test_populations[test_type]
            test_profile = self.evolved_test_profiles[test_type]

            test_elites = test_profile.elite_selector.select_elites(
                population=test_pop,
                population_config=test_profile.population_config,
                coevolution_context=context,
            )
            test_elites_dict[test_type] = test_elites

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
        population_type: str,
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
                    ind.notify_died(generation=population.generation)
                    logging_utils.log_individual_complete(ind, population_type, "DIED")

    def _breed_code(
        self,
        context: CoevolutionContext,
        num_elites: int = 0,
    ) -> list[CodeIndividual]:
        """
        Helper to breed code offspring using the injected code breeding strategy.

        The breeding strategy handles parallelization internally.
        """
        num_code_offspring = min(
            int(
                self.code_profile.population_config.max_population_size
                * self.code_profile.population_config.offspring_rate
            ),
            self.code_profile.population_config.max_population_size - num_elites,
        )

        # Breeding strategy handles parallelization and returns exact count
        code_offspring = self.code_profile.breeding_strategy.breed(
            context, num_code_offspring
        )

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

        The breeding strategy handles parallelization internally.
        """
        test_profile = self.evolved_test_profiles[test_type]

        num_test_offspring = (
            test_profile.population_config.max_population_size - num_elites
        )

        # Breeding strategy handles parallelization and returns exact count
        test_offsprings = test_profile.breeding_strategy.breed(
            context, num_test_offspring
        )

        logger.debug(f"Successfully bred {len(test_offsprings)} test offspring")
        return test_offsprings

    def _get_interaction_data(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
    ) -> InteractionData:
        """
        Helper to execute tests and return an atomic InteractionData artifact.

        Implementations of the execution system construct both the ID-keyed
        `execution_results` and an index-aligned `observation_matrix` and
        return them wrapped in an `InteractionData` object.
        """
        return self.execution_system.execute_tests(code_population, test_population)
