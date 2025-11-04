"""
Mock script to run the coevolution orchestrator with mock implementations.

This script demonstrates the core coevolution algorithm using simple mock
implementations of all components. It's useful for testing the architecture
and understanding the algorithm flow without requiring LLM API calls.
"""

from typing import Any

from loguru import logger

import common.logging_utils as logging_utils
from common.coevolution.core.interfaces import (
    BayesianConfig,
    CodePopulationConfig,
    EvolutionConfig,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
)
from common.coevolution.core.mock import (
    MockCodeBeliefUpdater,
    MockCodeOperator,
    MockCodeTestExecutor,
    MockDiscriminationCalculator,
    MockFeedbackGenerator,
    MockObservationMatrixBuilder,
    MockParetoFrontCalculator,
    MockProbabilityAssigner,
    MockSelectionStrategy,
    MockTestBeliefUpdater,
    MockTestBlockBuilder,
    MockTestOperator,
    get_mock_problem,
)
from common.coevolution.core.orchestrator import Orchestrator
from common.coevolution.core.population import CodePopulation, TestPopulation


def create_configurations() -> tuple[
    EvolutionConfig,
    CodePopulationConfig,
    PopulationConfig,
    OperatorRatesConfig,
    OperatorRatesConfig,
    BayesianConfig,
]:
    """Create all configuration objects for the experiment."""

    # Evolution configuration
    evo_config = EvolutionConfig(
        num_generations=5,  # Run for 5 generations
        random_seed=42,
    )

    # Code population configuration
    code_pop_config = CodePopulationConfig(
        initial_prior=0.5,
        initial_population_size=10,
        max_population_size=15,
        elitism_rate=0.5,  # Keep top 40% as elites
        offspring_rate=0.5,  # Generate 60% new offspring
    )

    # Test population configuration
    test_pop_config = PopulationConfig(
        initial_prior=0.5,
        initial_population_size=20,
    )

    # Code genetic operator rates
    code_op_rates = OperatorRatesConfig(
        crossover_rate=0.2,
        mutation_rate=0.2,
        edit_rate=0.8,
    )

    # Test genetic operator rates
    test_op_rates = OperatorRatesConfig(
        crossover_rate=0.4,
        mutation_rate=0.3,
        edit_rate=0.3,
    )

    # Bayesian belief update configuration
    bayesian_config = BayesianConfig(
        alpha=0.01,  # P(pass | code correct, test incorrect)
        beta=0.2,  # P(pass | code incorrect, test correct)
        gamma=0.2,  # P(pass | code incorrect, test incorrect)
        learning_rate=0.01,  # How fast beliefs update
    )

    return (
        evo_config,
        code_pop_config,
        test_pop_config,
        code_op_rates,
        test_op_rates,
        bayesian_config,
    )


def create_mock_components(problem: Problem) -> dict[str, Any]:
    """Create all mock component instances."""

    # Create operators
    code_operator = MockCodeOperator(problem)
    test_operator = MockTestOperator(problem)

    # Create strategies
    selector = MockSelectionStrategy()
    prob_assigner = MockProbabilityAssigner()

    # Create execution and observation components
    executor = MockCodeTestExecutor()
    obs_builder = MockObservationMatrixBuilder()

    # Create belief updaters
    code_belief_updater = MockCodeBeliefUpdater()
    test_belief_updater = MockTestBeliefUpdater()

    # Create test-specific components
    discrimination_calc = MockDiscriminationCalculator()
    pareto_calc = MockParetoFrontCalculator()
    test_block_builder = MockTestBlockBuilder()

    # Create feedback generators
    code_feedback_gen = MockFeedbackGenerator()
    test_feedback_gen = MockFeedbackGenerator()

    # Mock sandbox (not used in mock implementation)
    sandbox = None

    return {
        "code_operator": code_operator,
        "test_operator": test_operator,
        "selector": selector,
        "prob_assigner": prob_assigner,
        "executor": executor,
        "obs_builder": obs_builder,
        "code_belief_updater": code_belief_updater,
        "test_belief_updater": test_belief_updater,
        "discrimination_calc": discrimination_calc,
        "pareto_calc": pareto_calc,
        "test_block_builder": test_block_builder,
        "code_feedback_gen": code_feedback_gen,
        "test_feedback_gen": test_feedback_gen,
        "sandbox": sandbox,
    }


def log_results(
    code_population: CodePopulation, test_population: TestPopulation
) -> None:
    """Display the final results of the coevolution run."""

    logging_utils.log_section_header("INFO", "FINAL RESULTS")

    # Code population statistics
    best_code = code_population.get_best_individual()
    avg_code_prob = code_population.compute_average_probability()

    logger.info(f"Code Population Size: {code_population.size}")
    logger.info(f"Code Average Probability: {avg_code_prob:.4f}")
    if best_code:
        logger.info(
            f"Best Code Individual: {best_code.id} (prob={best_code.probability:.4f})"
        )

    # Test population statistics
    best_test = test_population.get_best_individual()
    avg_test_prob = test_population.compute_average_probability()
    pareto_front = test_population.get_pareto_front()

    logger.info(f"Test Population Size: {test_population.size}")
    logger.info(f"Test Average Probability: {avg_test_prob:.4f}")
    if best_test:
        logger.info(
            f"Best Test Individual: {best_test.id} (prob={best_test.probability:.4f})"
        )
    logger.info(f"Pareto Front Size: {len(pareto_front)}")

    # Display generation history
    logging_utils.log_section_header("INFO", "GENERATION DISTRIBUTION")
    gen_counts: dict[int, int] = {}
    for code_ind in code_population:
        gen_counts[code_ind.generation_born] = (
            gen_counts.get(code_ind.generation_born, 0) + 1
        )
    for gen in sorted(gen_counts.keys()):
        logger.info(f"  Generation {gen}: {gen_counts[gen]} individuals")

    logging_utils.log_section_header("INFO", "TEST POPULATION GENERATION DISTRIBUTION")
    gen_counts = {}
    for test_ind in test_population:
        gen_counts[test_ind.generation_born] = (
            gen_counts.get(test_ind.generation_born, 0) + 1
        )
    for gen in sorted(gen_counts.keys()):
        logger.info(f"  Generation {gen}: {gen_counts[gen]} individuals")


def main() -> None:
    """Main entry point for the mock coevolution experiment."""

    # Setup logging
    logging_utils.setup_logging(
        console_level="DEBUG",
        file_level="TRACE",
        log_file_base_name="mock_coevolution",
        setup_gen_log=True,
    )

    logging_utils.log_section_header("INFO", "MOCK COEVOLUTION ORCHESTRATOR EXPERIMENT")

    # Step 1: Get mock problem
    logger.info("Loading mock problem...")
    problem = get_mock_problem()
    logger.info(f"Problem: {problem.question_title}")
    logger.info(f"Problem ID: {problem.question_id}")
    logger.info(f"Public tests: {len(problem.public_test_cases)}")
    logger.info(f"Private tests: {len(problem.private_test_cases)}")

    # Step 2: Create configurations
    logger.info("\nCreating configurations...")
    (
        evo_config,
        code_pop_config,
        test_pop_config,
        code_op_rates,
        test_op_rates,
        bayesian_config,
    ) = create_configurations()

    logger.info(f"Evolution: {evo_config.num_generations} generations")
    logger.info(
        f"Code Population: {code_pop_config.initial_population_size} → {code_pop_config.max_population_size} individuals"
    )
    logger.info(
        f"Test Population: {test_pop_config.initial_population_size} individuals"
    )

    # Step 3: Create mock components
    logger.info("\nCreating mock components...")
    components = create_mock_components(problem)
    logger.info("All mock components created successfully")

    # Step 4: Create orchestrator
    logger.info("\nInitializing MockOrchestrator...")
    orchestrator = Orchestrator(
        # Configurations
        evo_config=evo_config,
        code_pop_config=code_pop_config,
        test_pop_config=test_pop_config,
        code_op_rates_config=code_op_rates,
        test_op_rates_config=test_op_rates,
        bayesian_config=bayesian_config,
        # Problem and sandbox
        problem=problem,
        sandbox=components["sandbox"],
        # Operators
        code_operator=components["code_operator"],
        test_operator=components["test_operator"],
        # Strategies
        selector=components["selector"],
        prob_assigner=components["prob_assigner"],
        # Execution
        executor=components["executor"],
        obs_builder=components["obs_builder"],
        # Belief updaters
        code_belief_updater=components["code_belief_updater"],
        test_belief_updater=components["test_belief_updater"],
        # Test components
        discrimination_calc=components["discrimination_calc"],
        pareto_calc=components["pareto_calc"],
        test_block_builder=components["test_block_builder"],
        # Feedback generators
        code_feedback_gen=components["code_feedback_gen"],
        test_feedback_gen=components["test_feedback_gen"],
    )

    # Step 5: Run coevolution
    logging_utils.log_section_header("INFO", "STARTING COEVOLUTION RUN")

    try:
        with logger.contextualize(run_id="mock_run", problem_id=problem.question_id):
            code_population, test_population = orchestrator.run()

        # Step 6: Display results
        log_results(code_population, test_population)

        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"Error during coevolution run: {e}")
        raise


if __name__ == "__main__":
    main()
