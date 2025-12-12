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
from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import (
    BayesianConfig,
    CodeProfile,
    EvolutionConfig,
    IEliteSelectionStrategy,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
    PublicTestProfile,
    TestProfile,
)
from common.coevolution.core.mock import (
    MockBayesianSystem,
    MockBreedingStrategy,
    MockCodeOperator,
    MockDatasetTestBlockBuilder,
    MockEliteSelector,
    MockExecutionSystem,
    MockTestBlockRebuilder,
    MockTestOperator,
    get_mock_problem,
)
from common.coevolution.core.orchestrator import Orchestrator
from common.coevolution.core.population import CodePopulation, TestPopulation


def create_profiles(
    code_pop_config: PopulationConfig,
    test_pop_configs: dict[str, PopulationConfig],
    bayesian_config: BayesianConfig,
    components: dict[str, Any],
) -> tuple[CodeProfile, dict[str, TestProfile], PublicTestProfile]:
    """Create profile objects from configurations and components."""

    # Create code profile
    code_profile = CodeProfile(
        population_config=code_pop_config,
        breeding_strategy=components["code_breeding_strategy"],
        elite_selector=components["code_elite_selector"],
    )

    # Create evolved test profiles
    evolved_test_profiles = {
        test_type: TestProfile(
            population_config=test_pop_configs[test_type],
            breeding_strategy=components["test_breeding_strategies"][test_type],
            elite_selector=components["test_elite_selectors"][test_type],
            bayesian_config=bayesian_config,
        )
        for test_type in ["unittest", "differential"]
    }

    # Create public test profile
    public_test_profile = PublicTestProfile(bayesian_config=bayesian_config)

    return code_profile, evolved_test_profiles, public_test_profile


def create_configurations() -> tuple[
    EvolutionConfig,
    PopulationConfig,
    OperatorRatesConfig,
    dict[str, PopulationConfig],
    dict[str, OperatorRatesConfig],
    BayesianConfig,
]:
    """Create all configuration objects for the experiment."""

    # Evolution configuration
    evo_config = EvolutionConfig(
        num_generations=5,  # Run for 5 generations
        random_seed=42,
        max_workers=1,  # Sequential execution for deterministic testing
    )

    # Code population configuration
    code_pop_config = PopulationConfig(
        initial_prior=0.5,
        initial_population_size=10,
        max_population_size=15,
        offspring_rate=0.5,  # Generate 50% new offspring
    )

    # Code genetic operator rates
    code_op_rates = OperatorRatesConfig(
        operation_rates={"crossover": 0.2, "edit": 0.6},
        mutation_rate=0.2,
    )

    # Per-test-type configurations (unittest and differential)
    test_pop_configs = {
        "unittest": PopulationConfig(
            initial_prior=0.5,
            initial_population_size=20,
        ),
        "differential": PopulationConfig(
            initial_prior=0.5,
            initial_population_size=0,  # Start with empty population for differential testing
            max_population_size=15,  # Allow growth through bootstrapping
        ),
    }

    test_op_rates_configs = {
        "unittest": OperatorRatesConfig(
            operation_rates={"crossover": 0.4, "edit": 0.3},
            mutation_rate=0.3,
        ),
        "differential": OperatorRatesConfig(
            operation_rates={"crossover": 0.3, "edit": 0.4},
            mutation_rate=0.3,
        ),
    }

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
        code_op_rates,
        test_pop_configs,
        test_op_rates_configs,
        bayesian_config,
    )


def create_mock_components(problem: Problem) -> dict[str, Any]:
    """Create all mock component instances."""

    # Create code operator
    code_operator = MockCodeOperator()

    # Create single bayesian system (not separate for code/test)
    bayesian_system = MockBayesianSystem()

    # Create code breeding strategy and elite selector
    code_breeding_strategy = MockBreedingStrategy(
        operator=code_operator, individual_type="code"
    )
    code_elite_selector: IEliteSelectionStrategy[CodeIndividual] = MockEliteSelector()

    # Create per-test-type components (unittest and differential)
    test_operators = {
        "unittest": MockTestOperator(),
        "differential": MockTestOperator(),
    }

    test_breeding_strategies = {
        "unittest": MockBreedingStrategy(
            operator=test_operators["unittest"], individual_type="test"
        ),
        "differential": MockBreedingStrategy(
            operator=test_operators["differential"], individual_type="test"
        ),
    }

    test_elite_selectors: dict[str, IEliteSelectionStrategy[TestIndividual]] = {
        "unittest": MockEliteSelector(),
        "differential": MockEliteSelector(),
    }

    # Create execution system
    execution_system = MockExecutionSystem()

    # Create test-specific components
    test_block_rebuilder = MockTestBlockRebuilder()

    # Create dataset test block builder
    dataset_test_block_builder = MockDatasetTestBlockBuilder()

    # Mock sandbox (not used in mock implementation)
    sandbox = None

    return {
        "code_operator": code_operator,
        "test_operators": test_operators,
        "code_breeding_strategy": code_breeding_strategy,
        "test_breeding_strategies": test_breeding_strategies,
        "code_elite_selector": code_elite_selector,
        "test_elite_selectors": test_elite_selectors,
        "execution_system": execution_system,
        "bayesian_system": bayesian_system,
        "test_block_rebuilder": test_block_rebuilder,
        "dataset_test_block_builder": dataset_test_block_builder,
        "sandbox": sandbox,
    }


@pytest.fixture
def mock_problem() -> Problem:
    """Fixture to provide a mock problem for testing."""
    return get_mock_problem()


@pytest.fixture
def configurations() -> tuple[
    EvolutionConfig,
    PopulationConfig,
    OperatorRatesConfig,
    dict[str, PopulationConfig],
    dict[str, OperatorRatesConfig],
    BayesianConfig,
]:
    """Fixture to provide test configurations."""
    return create_configurations()


@pytest.fixture
def mock_components(mock_problem: Problem) -> dict[str, Any]:
    """Fixture to provide mock components."""
    return create_mock_components(mock_problem)


def log_results(
    code_population: CodePopulation, test_populations: dict[str, TestPopulation]
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

    # Test population statistics (for each type)
    for test_type, test_population in test_populations.items():
        best_test = test_population.get_best_individual()
        avg_test_prob = test_population.compute_average_probability()

        logger.info(
            f"\n{test_type.upper()} Test Population Size: {test_population.size}"
        )
        logger.info(
            f"{test_type.upper()} Test Average Probability: {avg_test_prob:.4f}"
        )
        if best_test:
            logger.info(
                f"Best {test_type.upper()} Test Individual: {best_test.id} (prob={best_test.probability:.4f})"
            )

    # Note: get_pareto_front() requires observation_matrix, which is not available in this context
    # pareto_front = test_population.get_pareto_front(observation_matrix)
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
    for test_type, test_population in test_populations.items():
        logger.info(f"\n{test_type.upper()} Test Population:")
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
        PopulationConfig,
        OperatorRatesConfig,
        dict[str, PopulationConfig],
        dict[str, OperatorRatesConfig],
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
    """
    # Setup logging for test
    logging_utils.setup_logging(
        console_level="INFO",  # Less verbose for pytest
        file_level="DEBUG",
        log_file_base_name="test_mock_coevolution",
    )

    # Unpack configurations
    (
        evo_config,
        code_pop_config,
        code_op_rates,  # Not used - kept for backward compat with fixture
        test_pop_configs,
        test_op_rates_configs,  # Not used - kept for backward compat with fixture
        bayesian_config,
    ) = configurations

    # Create profiles from configurations and components
    code_profile, evolved_test_profiles, public_test_profile = create_profiles(
        code_pop_config=code_pop_config,
        test_pop_configs=test_pop_configs,
        bayesian_config=bayesian_config,
        components=mock_components,
    )

    # Create orchestrator using profiles
    orchestrator = Orchestrator(
        evo_config=evo_config,
        code_profile=code_profile,
        evolved_test_profiles=evolved_test_profiles,
        public_test_profile=public_test_profile,
        execution_system=mock_components["execution_system"],
        bayesian_system=mock_components["bayesian_system"],
        test_block_rebuilder=mock_components["test_block_rebuilder"],
        dataset_test_block_builder=mock_components["dataset_test_block_builder"],
    )

    # Run coevolution - orchestrator is stateless, problem passed at runtime
    with logger.contextualize(run_id="test_run", problem_id=mock_problem.question_id):
        code_population, evolved_test_pops = orchestrator.run(mock_problem)

    # Assertions to verify the run was successful
    assert code_population is not None, "Code population should not be None"
    assert evolved_test_pops is not None, "Evolved test populations should not be None"
    assert "unittest" in evolved_test_pops, (
        "unittest population should be in evolved_test_pops"
    )

    test_population = evolved_test_pops["unittest"]

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
        code_op_rates,
        test_pop_configs,
        test_op_rates_configs,
        bayesian_config,
    ) = create_configurations()

    logger.info(f"Evolution: {evo_config.num_generations} generations")
    logger.info(
        f"Code Population: {code_pop_config.initial_population_size} → {code_pop_config.max_population_size} individuals"
    )
    # Log test population configs
    for test_type, test_config in test_pop_configs.items():
        logger.info(
            f"{test_type.upper()} Test Population: {test_config.initial_population_size} individuals"
        )

    # Step 3: Create mock components
    logger.info("\nCreating mock components...")
    components = create_mock_components(problem)
    logger.info("All mock components created successfully")

    # Step 4: Create profiles
    logger.info("\nCreating profiles...")
    code_profile, evolved_test_profiles, public_test_profile = create_profiles(
        code_pop_config=code_pop_config,
        test_pop_configs=test_pop_configs,
        bayesian_config=bayesian_config,
        components=components,
    )

    # Step 5: Create orchestrator
    logger.info("\nInitializing Orchestrator...")
    orchestrator = Orchestrator(
        evo_config=evo_config,
        code_profile=code_profile,
        evolved_test_profiles=evolved_test_profiles,
        public_test_profile=public_test_profile,
        execution_system=components["execution_system"],
        bayesian_system=components["bayesian_system"],
        test_block_rebuilder=components["test_block_rebuilder"],
        dataset_test_block_builder=components["dataset_test_block_builder"],
    )

    # Step 6: Run coevolution
    logging_utils.log_section_header("INFO", "STARTING COEVOLUTION RUN")

    try:
        with logger.contextualize(run_id="mock_run", problem_id=problem.question_id):
            code_population, test_populations = orchestrator.run(problem)

        # Step 6: Display results
        log_results(code_population, test_populations)

        logger.info("\n" + "=" * 80)
        logger.info("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"Error during coevolution run: {e}")
        raise


if __name__ == "__main__":
    main()
