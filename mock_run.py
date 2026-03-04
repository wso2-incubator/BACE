"""
Mock coevolution run using simple mock implementations of all components.

Useful for verifying the architecture and understanding the algorithm flow
without requiring LLM API calls or a real execution environment.

Run from the workspace root:
    python mock_run.py
"""

from typing import Any

from loguru import logger

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    BayesianConfig,
    CodeProfile,
    EvolutionConfig,
    EvolutionPhase,
    EvolutionSchedule,
    IEliteSelectionStrategy,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
    PublicTestProfile,
    TestProfile,
)
from coevolution.core.mock import (
    MockBeliefUpdater,
    MockBreedingStrategy,
    MockCodeOperator,
    MockEliteSelector,
    MockExecutionSystem,
    MockTestOperator,
    get_mock_problem,
    mock_ledger_factory,
)
from coevolution.core.orchestrator import Orchestrator
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.utils import logging as logging_utils
from infrastructure.languages import PythonLanguage


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
    phase = EvolutionPhase(
        name="evolution",
        duration=5,
        evolve_code=True,
        evolve_tests=True,
    )
    schedule = EvolutionSchedule(phases=[phase])
    evo_config = EvolutionConfig(schedule=schedule)

    # Code population configuration
    code_pop_config = PopulationConfig(
        initial_prior=0.5,
        initial_population_size=10,
        max_population_size=15,
        offspring_rate=0.5,  # Generate 50% new offspring
    )

    # Code genetic operator rates (must sum to 1.0)
    code_op_rates = OperatorRatesConfig(
        operation_rates={
            "mutation": 0.2,
            "crossover": 0.2,
            "edit": 0.6,
        },
    )

    # Per-test-type configurations (unittest and differential)
    test_pop_configs = {
        "unittest": PopulationConfig(
            initial_prior=0.5,
            initial_population_size=20,
            max_population_size=20,
        ),
        "differential": PopulationConfig(
            initial_prior=0.5,
            initial_population_size=0,  # Start with empty population for differential testing
            max_population_size=15,  # Allow growth through bootstrapping
        ),
    }

    test_op_rates_configs = {
        "unittest": OperatorRatesConfig(
            operation_rates={
                "mutation": 0.3,
                "crossover": 0.4,
                "edit": 0.3,
            },
        ),
        "differential": OperatorRatesConfig(
            operation_rates={
                "mutation": 0.3,
                "crossover": 0.3,
                "edit": 0.4,
            },
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
    bayesian_system = MockBeliefUpdater()

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
        "sandbox": sandbox,
    }


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


def main() -> None:
    """Main entry point for the mock coevolution run."""

    # Setup logging
    logging_utils.setup_logging(
        console_level="DEBUG",
        file_level="TRACE",
        log_file_base_name="mock_coevolution",
    )

    logging_utils.log_section_header("INFO", "MOCK COEVOLUTION ORCHESTRATOR RUN")

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
        ledger_factory=mock_ledger_factory,
        language_adapter=PythonLanguage(),
    )

    # Step 6: Run coevolution
    logging_utils.log_section_header("INFO", "STARTING COEVOLUTION RUN")

    try:
        with logger.contextualize(run_id="mock_run", problem_id=problem.question_id):
            code_population, test_populations = orchestrator.run(problem)

        # Step 7: Display results
        log_results(code_population, test_populations)

        logger.info("\n" + "=" * 80)
        logger.info("MOCK RUN COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"Error during coevolution run: {e}")
        raise


if __name__ == "__main__":
    main()
