# src/common/coevolution/orchestrator.py

import random

import numpy as np
from loguru import logger

import common.coevolution.logging_utils as logging_utils

# Import concrete classes
from .breeding_strategy import BreedingStrategy
from .individual import CodeIndividual, TestIndividual

# Import all interfaces, configs, and types
from .interfaces import (
    BasePopulation,
    BayesianConfig,
    CodePopulationConfig,
    EvolutionConfig,
    ExecutionResults,
    IBeliefUpdater,
    ICodeOperator,
    ICodeTestExecutor,
    IDiscriminationCalculator,
    IFeedbackGenerator,
    IObservationMatrixBuilder,
    IParetoFrontCalculator,
    IProbabilityAssigner,
    ISelectionStrategy,
    ITestBlockBuilder,
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
        test_pop_config: PopulationConfig,
        code_op_rates_config: OperatorRatesConfig,
        test_op_rates_config: OperatorRatesConfig,
        bayesian_config: BayesianConfig,
        # --- Injected Mock Components ---
        problem: Problem,
        sandbox: Sandbox,
        code_operator: ICodeOperator,
        test_operator: ITestOperator,
        selector: ISelectionStrategy,
        prob_assigner: IProbabilityAssigner,
        executor: ICodeTestExecutor,
        obs_builder: IObservationMatrixBuilder,
        code_belief_updater: IBeliefUpdater,
        test_belief_updater: IBeliefUpdater,
        discrimination_calc: IDiscriminationCalculator,
        pareto_calc: IParetoFrontCalculator,
        test_block_builder: ITestBlockBuilder,
        code_feedback_gen: IFeedbackGenerator[TestIndividual],
        test_feedback_gen: IFeedbackGenerator[CodeIndividual],
    ) -> None:
        """
        Initializes the orchestrator by storing all injected dependencies.

        Also sets up:
        - Concrete breeding strategies for code and test populations
        - Generation logger for tracking evolution progress (via logging_utils)
        - Random seed for reproducibility
        """
        logger.info("Initializing MockOrchestrator...")

        # --- Store Configs ---
        self.evo_config = evo_config
        self.code_pop_config = code_pop_config
        self.test_pop_config = test_pop_config
        self.code_op_rates_config = code_op_rates_config
        self.test_op_rates_config = test_op_rates_config
        self.bayesian_config = bayesian_config

        # --- Store Problem & Sandbox ---
        self.problem = problem
        self.sandbox = sandbox

        # --- Store Injected Components ---
        self.code_operator = code_operator
        self.test_operator = test_operator
        self.selector = selector
        self.prob_assigner = prob_assigner
        self.executor = executor
        self.obs_builder = obs_builder
        self.code_belief_updater = code_belief_updater
        self.test_belief_updater = test_belief_updater
        self.discrimination_calc = discrimination_calc
        self.pareto_calc = pareto_calc
        self.test_block_builder = test_block_builder
        self.code_feedback_gen = code_feedback_gen
        self.test_feedback_gen = test_feedback_gen

        # --- Create Concrete Breeding Strategies ---
        self.code_breeder = BreedingStrategy[CodeIndividual, TestIndividual](
            selector=self.selector,
            operator=self.code_operator,
            individual_factory=CodeIndividual,
            probability_assigner=self.prob_assigner,
            initial_prior=self.code_pop_config.initial_prior,
        )
        self.test_breeder = BreedingStrategy[TestIndividual, CodeIndividual](
            selector=self.selector,
            operator=self.test_operator,
            individual_factory=TestIndividual,
            probability_assigner=self.prob_assigner,
            initial_prior=self.test_pop_config.initial_prior,
        )

        self.gen_logger = logging_utils.get_generation_logger()
        self._set_random_seed(evo_config.random_seed)
        logger.info("MockOrchestrator initialized successfully.")

    def _set_random_seed(self, seed: int) -> None:
        """
        Helper to set random seeds for reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"Random seed set to {seed} for reproducibility.")

    def _create_fixed_test_population(
        self, test_cases: list[Test], fixed_prior: float
    ) -> TestPopulation:
        """
        Helper to create a fixed TestPopulation from the problem's test cases.
        """
        individuals: list[TestIndividual] = []
        snippets: list[str] = []
        for i, t in enumerate(test_cases):
            # Create a mock snippet that matches the test case
            snippet = f"def test_fixed_{i}(self):\n    assert sort_array({t.input}) == {t.output}"
            snippets.append(snippet)
            individuals.append(
                TestIndividual(
                    snippet=snippet,
                    probability=fixed_prior,
                    creation_op="initial",
                    generation_born=0,
                    parent_ids=[],
                )
            )

        # Create a dummy class block string for this fixed population
        dummy_block = (
            "import unittest\nfrom a_mock_solution import sort_array\n\n"
            "class FixedTests(unittest.TestCase):\n"
        )
        dummy_block += "\n\n".join([f"    {s}" for s in snippets])

        logger.info(
            f"Created fixed test population with {len(individuals)} individuals."
        )

        return TestPopulation(
            individuals=individuals,
            pareto_fn=self.pareto_calc,  # Injected, though unused
            rebuild_test_block_fn=self.test_block_builder,  # Injected, though unused
            test_class_block=dummy_block,
            generation=0,
        )

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
                creation_op="initial",
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
                creation_op="initial",
                generation_born=0,
                parent_ids=[],
            )
            for s in test_snippets
        ]
        test_population = TestPopulation(
            individuals=test_individuals,
            pareto_fn=self.pareto_calc,
            rebuild_test_block_fn=self.test_block_builder,
            test_class_block=test_block,
            generation=0,
        )
        logger.info(f"Created {test_population!r}")
        return test_population

    def _exec_results(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
        public_test_population: TestPopulation,
        private_test_population: TestPopulation,
    ) -> tuple[ExecutionResults, ExecutionResults, ExecutionResults]:
        """
        Helper to execute code population against generated, public, and private tests.
        """
        gen_exec_results = self.executor(code_population, test_population, self.sandbox)
        pub_exec_results = self.executor(
            code_population, public_test_population, self.sandbox
        )
        priv_exec_results = self.executor(
            code_population, private_test_population, self.sandbox
        )
        return gen_exec_results, pub_exec_results, priv_exec_results

    def _generate_observation_matrices(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
        public_test_population: TestPopulation,
        private_test_population: TestPopulation,
        gen_exec_results: ExecutionResults,
        pub_exec_results: ExecutionResults,
        priv_exec_results: ExecutionResults,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper to generate observation matrices for generated, public, and private tests.
        """
        gen_obs_matrix = self.obs_builder(
            code_population, test_population, gen_exec_results
        )
        pub_obs_matrix = self.obs_builder(
            code_population, public_test_population, pub_exec_results
        )
        priv_obs_matrix = self.obs_builder(
            code_population, private_test_population, priv_exec_results
        )

        logger.info(f"Generated tests pass rate: {gen_obs_matrix.mean():.2%}")
        logger.info(f"Public tests pass rate: {pub_obs_matrix.mean():.2%}")
        logger.info(f"Private tests pass rate: {priv_obs_matrix.mean():.2%}")

        logger.debug(f"Generated Observation Matrix:\n{gen_obs_matrix}")
        logger.debug(f"Public Observation Matrix:\n{pub_obs_matrix}")
        logger.debug(f"Private Observation Matrix:\n{priv_obs_matrix}")

        return gen_obs_matrix, pub_obs_matrix, priv_obs_matrix

    def _compute_posterior_beliefs(
        self,
        code_probabilities: np.ndarray,
        gen_test_probabilities: np.ndarray,
        public_test_probabilities: np.ndarray,
        gen_obs_matrix: np.ndarray,
        pub_obs_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper to update beliefs for code and generated tests.
        """

        logger.debug("Updating code beliefs based on public tests...")
        code_posterior_on_public = self.code_belief_updater(
            prior_code_probs=code_probabilities,
            prior_test_probs=public_test_probabilities,
            observation_matrix=pub_obs_matrix,
            config=self.bayesian_config,
        )

        logger.debug("Updating generated test beliefs based on updated code...")
        test_posterior = self.test_belief_updater(
            prior_code_probs=code_posterior_on_public,
            prior_test_probs=gen_test_probabilities,
            observation_matrix=gen_obs_matrix,
            config=self.bayesian_config,
        )

        logger.debug("Updating code beliefs based on generated tests...")
        code_posterior_on_generated = self.code_belief_updater(
            prior_code_probs=code_posterior_on_public,
            prior_test_probs=test_posterior,
            observation_matrix=gen_obs_matrix,
            config=self.bayesian_config,
        )

        return code_posterior_on_generated, test_posterior

    def _select_elites(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
    ) -> tuple[list[CodeIndividual], list[TestIndividual]]:
        """
        Helper to select elite individuals from code and test populations.
        """
        # --- Code Elites ---
        num_code_elites = int(code_population.size * self.code_pop_config.elitism_rate)
        code_elites = code_population.get_top_k_individuals(num_code_elites)

        # --- Test Elites (Pareto Front) ---
        test_elites = test_population.get_pareto_front()
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
                    logging_utils.log_individual_complete(self.gen_logger, ind, "DIED")
                    # Then notify the individual
                    ind.notify_died(generation=population.generation)

    def _breed_code(
        self,
        code_population: CodePopulation,
        test_population: TestPopulation,
        gen_exec_results: ExecutionResults,
        gen_observation_matrix: np.ndarray,
        num_elites: int = 0,
    ) -> list[CodeIndividual]:
        """
        Helper to breed code offspring using the code breeder.
        """
        num_code_offspring = min(
            int(
                self.code_pop_config.max_population_size
                * self.code_pop_config.offspring_rate
            ),
            self.code_pop_config.max_population_size - num_elites,
        )

        code_offspring: list[CodeIndividual] = []
        for _ in range(num_code_offspring):
            # Feedback for code edits comes from generated tests
            code_child = self.code_breeder.generate_single_offspring(
                population=code_population,
                other_population=test_population,
                execution_results=gen_exec_results,
                feedback_generator=self.code_feedback_gen,
                observation_matrix=gen_observation_matrix,
                operation_rates=self.code_op_rates_config,
            )
            code_offspring.append(code_child)
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
        """
        num_test_offspring = self.test_pop_config.initial_population_size - num_elites
        test_offsprings: list[TestIndividual] = []
        for _ in range(num_test_offspring):
            # Feedback for test edits comes from code population
            test_child = self.test_breeder.generate_single_offspring(
                population=test_population,
                other_population=code_population,
                execution_results=gen_exec_results,
                feedback_generator=self.test_feedback_gen,
                observation_matrix=gen_observation_matrix.T,
                operation_rates=self.test_op_rates_config,
            )
            test_offsprings.append(test_child)
        return test_offsprings

    def run(self) -> tuple[CodePopulation, TestPopulation]:
        """
        Runs the main co-evolutionary loop for the configured number of generations.

        Evolution Process:
        - Creates initial populations (code, test, public test, private test)
        - Iterates through generations performing:
          1. Execution & Observation (run tests, build observation matrices)
          2. Belief Updates (Bayesian updates on probabilities)
          3. Elite Selection (top-k code, Pareto front tests)
          4. Offspring Generation (via breeding strategies)
          5. Population Transition (set next generation, notify removed individuals)
          6. Logging (generation summaries)

        Logging Behavior:
        - Each generation: Logs lightweight summary with aggregate statistics
        - When individuals die: Logs complete lifecycle record (once per individual)
        - After evolution: Logs all final survivors with complete lifecycle records

        This logging strategy ensures each individual is logged exactly once,
        eliminating redundancy from repeated logging across generations.

        Returns:
            Tuple of (final_code_population, final_test_population)
        """
        logging_utils.log_section_header("INFO", "STARTING MOCK CO-EVOLUTION RUN")

        # --- Create Initial Populations ---
        code_population = self._create_initial_code_population()
        test_population = self._create_initial_test_populations()
        public_test_population = self._create_fixed_test_population(
            self.problem.public_test_cases, fixed_prior=0.95
        )
        private_test_population = self._create_fixed_test_population(
            self.problem.private_test_cases, fixed_prior=0.95
        )

        assert code_population is not None
        assert test_population is not None
        assert public_test_population is not None
        assert private_test_population is not None

        # --- Main Evolution Loop ---
        for gen in range(self.evo_config.num_generations):
            current_gen_num = gen + 1
            logging_utils.log_subsection_header(
                "INFO",
                f"GENERATION {current_gen_num} / {self.evo_config.num_generations}",
            )

            # --- Step 1: Execute & Observe ---
            logger.debug("Step 1: Executing all test sets...")
            gen_exec_results, pub_exec_results, priv_exec_results = self._exec_results(
                code_population,
                test_population,
                public_test_population,
                private_test_population,
            )
            gen_obs_matrix, pub_obs_matrix, priv_obs_matrix = (
                self._generate_observation_matrices(
                    code_population,
                    test_population,
                    public_test_population,
                    private_test_population,
                    gen_exec_results,
                    pub_exec_results,
                    priv_exec_results,
                )
            )

            # --- Step 2: Update Beliefs ---
            logger.debug("Step 2: Updating beliefs...")
            # 2a. Compute Posterior Beliefs
            code_posterior, test_posterior = self._compute_posterior_beliefs(
                code_population.probabilities,
                test_population.probabilities,
                public_test_population.probabilities,
                gen_obs_matrix,
                pub_obs_matrix,
            )
            code_population.update_probabilities(code_posterior)
            test_population.update_probabilities(test_posterior)

            # 2b. Calculate Discrimination (only for generated tests)
            discrim_scores = self.discrimination_calc(gen_obs_matrix)
            test_population.set_discriminations(discrim_scores)

            # --- Step 3: Select Elites ---
            logger.debug("Step 3: Selecting elites...")
            code_elites, test_elites = self._select_elites(
                code_population, test_population
            )

            self._notify_elites(code_elites, generation=code_population.generation)
            self._notify_elites(test_elites, generation=test_population.generation)

            # --- Step 4: Generate Offspring ---
            logger.debug("Step 4: Generating offspring...")
            code_offsprings: list[CodeIndividual] = self._breed_code(
                code_population=code_population,
                test_population=test_population,
                gen_exec_results=gen_exec_results,
                gen_observation_matrix=gen_obs_matrix,
                num_elites=len(code_elites),
            )
            test_offsprings: list[TestIndividual] = self._breed_tests(
                test_population=test_population,
                code_population=code_population,
                gen_exec_results=gen_exec_results,
                gen_observation_matrix=gen_obs_matrix,
                num_elites=len(test_elites),
            )

            # --- Step 5: Set Next Generation ---
            logger.debug("Step 5: Setting next generation...")

            # 5a. Code Population
            new_code_gen = code_elites + code_offsprings
            assert len(new_code_gen) <= self.code_pop_config.max_population_size, (
                "Code population exceeded max size"
            )
            # Notify removed individuals before transitioning
            self._notify_removed_individuals(code_population, new_code_gen)
            code_population.set_next_generation(new_code_gen)

            # 5b. Test Population
            new_test_gen = test_elites + test_offsprings
            assert len(new_test_gen) == self.test_pop_config.initial_population_size, (
                "Test population size mismatch"
            )
            # Notify removed individuals before transitioning
            self._notify_removed_individuals(test_population, new_test_gen)
            test_population.set_next_generation(new_test_gen)

            logger.info(
                f"Advanced to Gen {code_population.generation}. "
                f"Code Pop Size: {code_population.size}, "
                f"Test Pop Size: {test_population.size}"
            )

            logging_utils.log_generation_summary(
                self.gen_logger, code_population, test_population
            )

        # After evolution loop completes, log all final survivors
        logging_utils.log_final_survivors(
            self.gen_logger, code_population, test_population
        )

        logging_utils.log_section_header(
            "INFO",
            "CO-EVOLUTION RUN FINISHED."
            f"Final Code Population: {code_population!r}"
            f"Final Test Population: {test_population!r}"
            f"Final Private Test Pass Rate: {priv_obs_matrix.mean():.2%}",
        )

        return code_population, test_population
