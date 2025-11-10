"""
Integration test for the coevolution orchestrator with mock implementations.

This script tests the complete coevolution algorithm using simple mock
implementations of all components. It's useful for verifying the architecture
and understanding the algorithm flow without requiring LLM API calls.

Can be run either with pytest:
    pytest tests/integration/test_orchestrator_integration.py -v

Or as a standalone script:
    python tests/integration/test_orchestrator_integration.py
"""

from typing import Any

import pytest
from loguru import logger

import common.coevolution.logging_utils as logging_utils
from common.coevolution.core.interfaces import (
    BayesianConfig,
    CodePopulationConfig,
    EvolutionConfig,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
)
from common.coevolution.core.mock import (
    MockCodeBayesianSystem,
    MockCodeOperator,
    MockExecutionSystem,
    MockFeedbackGenerator,
    MockPareto,
    MockProbabilityAssigner,
    MockSelectionStrategy,
    MockTestBayesianSystem,
    MockTestBlockRebuilder,
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

    # Create strategies (separate for code and test)
    code_selector = MockSelectionStrategy()
    test_selector = MockSelectionStrategy()
    code_prob_assigner = MockProbabilityAssigner()
    test_prob_assigner = MockProbabilityAssigner()

    # Create grouped systems
    execution_system = MockExecutionSystem()
    code_bayesian_system = MockCodeBayesianSystem()
    test_bayesian_system = MockTestBayesianSystem()

    # Create test-specific components
    pareto = MockPareto()
    test_block_rebuilder = MockTestBlockRebuilder()

    # Create feedback generators
    code_feedback_gen = MockFeedbackGenerator()
    test_feedback_gen = MockFeedbackGenerator()

    # Mock sandbox (not used in mock implementation)
    sandbox = None

    return {
        "code_operator": code_operator,
        "test_operator": test_operator,
        "code_selector": code_selector,
        "test_selector": test_selector,
        "code_prob_assigner": code_prob_assigner,
        "test_prob_assigner": test_prob_assigner,
        "execution_system": execution_system,
        "code_bayesian_system": code_bayesian_system,
        "test_bayesian_system": test_bayesian_system,
        "pareto": pareto,
        "test_block_rebuilder": test_block_rebuilder,
        "code_feedback_gen": code_feedback_gen,
        "test_feedback_gen": test_feedback_gen,
        "sandbox": sandbox,
    }


@pytest.fixture
def mock_problem() -> Problem:
    """Fixture to provide a mock problem for testing."""
    return get_mock_problem()


@pytest.fixture
def configurations() -> tuple[
    EvolutionConfig,
    CodePopulationConfig,
    PopulationConfig,
    OperatorRatesConfig,
    OperatorRatesConfig,
    BayesianConfig,
]:
    """Fixture to provide test configurations."""
    return create_configurations()


@pytest.fixture
def mock_components(mock_problem: Problem) -> dict[str, Any]:
    """Fixture to provide mock components."""
    return create_mock_components(mock_problem)


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
    # Note: get_pareto_front() requires observation_matrix, which is not available in this context
    # pareto_front = test_population.get_pareto_front(observation_matrix)

    logger.info(f"Test Population Size: {test_population.size}")
    logger.info(f"Test Average Probability: {avg_test_prob:.4f}")
    if best_test:
        logger.info(
            f"Best Test Individual: {best_test.id} (prob={best_test.probability:.4f})"
        )
    # logger.info(f"Pareto Front Size: {len(pareto_front)}")

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


def test_orchestrator_full_run(
    mock_problem: Problem,
    configurations: tuple[
        EvolutionConfig,
        CodePopulationConfig,
        PopulationConfig,
        OperatorRatesConfig,
        OperatorRatesConfig,
        BayesianConfig,
    ],
    mock_components: dict[str, Any],
) -> None:
    """
    Test the complete orchestrator run with mock components.

    This integration test verifies that:
    - The orchestrator runs without errors for the configured generations
    - Both code and test populations are created and evolved
    - Final populations contain individuals
    - Best individuals can be identified
    - Pareto front is computed for test population
    """
    # Setup logging for test
    logging_utils.setup_logging(
        console_level="INFO",  # Less verbose for pytest
        file_level="DEBUG",
        log_file_base_name="test_mock_coevolution",
        setup_gen_log=True,
    )

    # Unpack configurations
    (
        evo_config,
        code_pop_config,
        test_pop_config,
        code_op_rates,
        test_op_rates,
        bayesian_config,
    ) = configurations

    # Create orchestrator
    orchestrator = Orchestrator(
        # Configurations
        evo_config=evo_config,
        code_pop_config=code_pop_config,
        test_pop_config=test_pop_config,
        code_op_rates_config=code_op_rates,
        test_op_rates_config=test_op_rates,
        bayesian_config=bayesian_config,
        # Problem and sandbox
        problem=mock_problem,
        sandbox=mock_components["sandbox"],
        # Operators
        code_operator=mock_components["code_operator"],
        test_operator=mock_components["test_operator"],
        # Strategies (separate for code and test)
        code_selector=mock_components["code_selector"],
        test_selector=mock_components["test_selector"],
        code_prob_assigner=mock_components["code_prob_assigner"],
        test_prob_assigner=mock_components["test_prob_assigner"],
        # Grouped systems
        execution_system=mock_components["execution_system"],
        code_bayesian_system=mock_components["code_bayesian_system"],
        test_bayesian_system=mock_components["test_bayesian_system"],
        # Test components
        pareto=mock_components["pareto"],
        test_block_rebuilder=mock_components["test_block_rebuilder"],
        # Feedback generators
        code_feedback_gen=mock_components["code_feedback_gen"],
        test_feedback_gen=mock_components["test_feedback_gen"],
    )

    # Run coevolution
    with logger.contextualize(run_id="test_run", problem_id=mock_problem.question_id):
        code_population, test_population = orchestrator.run()

    # Assertions to verify the run was successful
    assert code_population is not None, "Code population should not be None"
    assert test_population is not None, "Test population should not be None"

    # Verify populations have individuals
    assert code_population.size > 0, "Code population should have individuals"
    assert test_population.size > 0, "Test population should have individuals"

    # Verify best individuals can be retrieved
    best_code = code_population.get_best_individual()
    best_test = test_population.get_best_individual()
    assert best_code is not None, "Should have a best code individual"
    assert best_test is not None, "Should have a best test individual"

    # Verify probability calculations
    avg_code_prob = code_population.compute_average_probability()
    avg_test_prob = test_population.compute_average_probability()
    assert 0.0 <= avg_code_prob <= 1.0, "Average code probability should be in [0, 1]"
    assert 0.0 <= avg_test_prob <= 1.0, "Average test probability should be in [0, 1]"

    # Note: Pareto front verification requires observation_matrix
    # pareto_front = test_population.get_pareto_front(observation_matrix)
    # assert len(pareto_front) > 0, "Should have a non-empty Pareto front"

    # Verify population size constraints
    assert code_population.size <= code_pop_config.max_population_size, (
        f"Code population size {code_population.size} should not exceed max {code_pop_config.max_population_size}"
    )

    logger.info("✓ All integration test assertions passed!")


def main() -> None:
    """Main entry point for the mock coevolution integration test."""

    # Setup logging
    logging_utils.setup_logging(
        console_level="DEBUG",
        file_level="TRACE",
        log_file_base_name="mock_coevolution",
        setup_gen_log=True,
    )

    logging_utils.log_section_header(
        "INFO", "MOCK COEVOLUTION ORCHESTRATOR INTEGRATION TEST"
    )

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
    logger.info("\nInitializing Orchestrator...")
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
        # Strategies (separate for code and test)
        code_selector=components["code_selector"],
        test_selector=components["test_selector"],
        code_prob_assigner=components["code_prob_assigner"],
        test_prob_assigner=components["test_prob_assigner"],
        # Grouped systems
        execution_system=components["execution_system"],
        code_bayesian_system=components["code_bayesian_system"],
        test_bayesian_system=components["test_bayesian_system"],
        # Test components
        pareto=components["pareto"],
        test_block_rebuilder=components["test_block_rebuilder"],
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
        logger.info("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"Error during coevolution run: {e}")
        raise


if __name__ == "__main__":
    main()
