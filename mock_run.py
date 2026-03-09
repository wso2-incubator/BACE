"""
Mock coevolution run using simple mock implementations of all components.

Useful for verifying the architecture and understanding the algorithm flow
without requiring LLM API calls or a real execution environment.

Run from the workspace root:
    python mock_run.py
"""

from typing import Any

import typer
from loguru import logger

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    BayesianConfig,
    CodeProfile,
    EvolutionConfig,
    EvolutionPhase,
    EvolutionSchedule,
    IEliteSelectionStrategy,
    PopulationConfig,
    PublicTestProfile,
    TestProfile,
)
from coevolution.core.mock import (
    MockBeliefUpdater,
    MockCodeInitializer,
    MockCodeOperator,
    MockEliteSelector,
    MockExecutionSystem,
    MockTestInitializer,
    MockTestOperator,
    get_mock_problem,
    mock_ledger_factory,
)
from coevolution.core.orchestrator import Orchestrator
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.strategies.breeding.breeder import Breeder, RegisteredOperator
from coevolution.utils import logging as logging_utils
from infrastructure.languages import PythonLanguage


def create_profiles(
    code_pop_config: PopulationConfig,
    test_pop_configs: dict[str, PopulationConfig],
    bayesian_config: BayesianConfig,
    components: dict[str, Any],
) -> tuple[CodeProfile, dict[str, TestProfile], PublicTestProfile]:
    """Create profile objects from configurations and components."""

    code_profile = CodeProfile(
        population_config=code_pop_config,
        breeder=components["code_breeder"],
        initializer=components["code_initializer"],
        elite_selector=components["code_elite_selector"],
    )

    evolved_test_profiles = {
        test_type: TestProfile(
            population_config=test_pop_configs[test_type],
            breeder=components["test_breeders"][test_type],
            initializer=components["test_initializers"][test_type],
            elite_selector=components["test_elite_selectors"][test_type],
            bayesian_config=bayesian_config,
        )
        for test_type in ["unittest", "differential"]
    }

    public_test_profile = PublicTestProfile(bayesian_config=bayesian_config)

    return code_profile, evolved_test_profiles, public_test_profile


def create_configurations(
    schedule_type: str = "simultaneous",
) -> tuple[
    EvolutionConfig,
    PopulationConfig,
    dict[str, PopulationConfig],
    BayesianConfig,
]:
    """Create all configuration objects for the experiment based on schedule type."""

    if schedule_type == "alternating":
        schedule = EvolutionSchedule(
            phases=[
                EvolutionPhase(
                    name="test_warmup", duration=1, evolve_code=False, evolve_tests=True
                ),
                EvolutionPhase(
                    name="code_catchup",
                    duration=1,
                    evolve_code=True,
                    evolve_tests=False,
                ),
                EvolutionPhase(
                    name="test_lead", duration=1, evolve_code=False, evolve_tests=True
                ),
                EvolutionPhase(
                    name="code_follow", duration=2, evolve_code=True, evolve_tests=False
                ),
            ]
        )
    elif schedule_type == "warmup_only":
        schedule = EvolutionSchedule(
            phases=[
                EvolutionPhase(
                    name="test_warmup", duration=2, evolve_code=False, evolve_tests=True
                ),
                EvolutionPhase(
                    name="code_evolution",
                    duration=3,
                    evolve_code=True,
                    evolve_tests=False,
                ),
            ]
        )
    else:  # default simultaneous
        schedule = EvolutionSchedule(
            phases=[
                EvolutionPhase(
                    name="evolution", duration=5, evolve_code=True, evolve_tests=True
                ),
            ]
        )

    evo_config = EvolutionConfig(schedule=schedule)

    code_pop_config = PopulationConfig(
        initial_prior=0.5,
        initial_population_size=10,
        max_population_size=15,
        offspring_rate=0.5,
    )

    test_pop_configs = {
        "unittest": PopulationConfig(
            initial_prior=0.5,
            initial_population_size=20,
            max_population_size=20,
        ),
        "differential": PopulationConfig(
            initial_prior=0.5,
            initial_population_size=0,
            max_population_size=15,
        ),
    }

    bayesian_config = BayesianConfig(
        alpha=0.01,
        beta=0.2,
        gamma=0.2,
        learning_rate=0.01,
    )

    return evo_config, code_pop_config, test_pop_configs, bayesian_config


def create_mock_components(
    code_pop_config: PopulationConfig,
    test_pop_configs: dict[str, PopulationConfig],
) -> dict[str, Any]:
    """Create all mock component instances."""

    # Initializers
    code_initializer = MockCodeInitializer(
        population_size=code_pop_config.initial_population_size,
        initial_prior=code_pop_config.initial_prior,
    )

    test_initializers = {
        test_type: MockTestInitializer(
            population_size=cfg.initial_population_size,
            initial_prior=cfg.initial_prior,
        )
        for test_type, cfg in test_pop_configs.items()
    }

    # Operators
    code_operator = MockCodeOperator()

    test_operators = {
        "unittest": MockTestOperator(test_type="unittest"),
        "differential": MockTestOperator(test_type="differential"),
    }

    # Breeders — weights co-registered with operators
    code_breeder = Breeder(
        registered_operators=[
            RegisteredOperator(weight=1.0, operator=code_operator),
        ],
        llm_workers=1,
    )

    test_breeders = {
        test_type: Breeder(
            registered_operators=[
                RegisteredOperator(weight=1.0, operator=test_operators[test_type]),
            ],
            llm_workers=1,
        )
        for test_type in ["unittest", "differential"]
    }

    # Elite selectors
    code_elite_selector: IEliteSelectionStrategy[CodeIndividual] = MockEliteSelector()
    test_elite_selectors: dict[str, IEliteSelectionStrategy[TestIndividual]] = {
        "unittest": MockEliteSelector(),
        "differential": MockEliteSelector(),
    }

    # Infrastructure
    execution_system = MockExecutionSystem()
    bayesian_system = MockBeliefUpdater()

    return {
        "code_initializer": code_initializer,
        "test_initializers": test_initializers,
        "code_operator": code_operator,
        "test_operators": test_operators,
        "code_breeder": code_breeder,
        "test_breeders": test_breeders,
        "code_elite_selector": code_elite_selector,
        "test_elite_selectors": test_elite_selectors,
        "execution_system": execution_system,
        "bayesian_system": bayesian_system,
    }


def log_results(
    code_population: CodePopulation, test_populations: dict[str, TestPopulation]
) -> None:
    """Display the final results of the coevolution run."""

    logging_utils.log_section_header("INFO", "FINAL RESULTS")

    best_code = code_population.get_best_individual()
    avg_code_prob = code_population.compute_average_probability()

    logger.info(f"Code Population Size: {code_population.size}")
    logger.info(f"Code Average Probability: {avg_code_prob:.4f}")
    if best_code:
        logger.info(
            f"Best Code Individual: {best_code.id} (prob={best_code.probability:.4f})"
        )

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


app = typer.Typer(help="Run a mock coevolution experiment.")


@app.command()
def main(
    schedule: str = typer.Option(
        "simultaneous",
        "--schedule",
        "-s",
        help="Schedule type: 'simultaneous', 'alternating', or 'warmup_only'",
    ),
) -> None:
    """Main entry point for the mock coevolution run."""

    run_id = logging_utils.setup_logging(
        console_level="DEBUG",
        file_level="TRACE",
        run_id="mock",
        problem_id="mock-problem",
    )

    logging_utils.log_section_header("INFO", "MOCK COEVOLUTION ORCHESTRATOR RUN")

    logger.info("Loading mock problem...")
    problem = get_mock_problem()
    logger.info(f"Problem: {problem.question_title}")
    logger.info(f"Problem ID: {problem.question_id}")
    logger.info(f"Public tests: {len(problem.public_test_cases)}")
    logger.info(f"Private tests: {len(problem.private_test_cases)}")

    logger.info("\nCreating configurations...")
    (
        evo_config,
        code_pop_config,
        test_pop_configs,
        bayesian_config,
    ) = create_configurations(schedule_type=schedule)

    # Save configuration to the run directory
    logging_utils.save_run_config(
        run_id,
        {
            "experiment": {
                "name": "mock-coevolution",
                "description": "Mock coevolution run for testing dashboard",
                "language": "python",
            },
            "llm": {
                "provider": "mock-provider",
                "model": "mock-gpt-5-mini",
            },
            "evolution_config": evo_config,
            "code_profile": {
                "population_config": code_pop_config,
                "operator_rates": {
                    "crossover": 0.2,
                    "mutation": 0.3,
                    "edit": 0.5,
                },
            },
            "test_profiles": {
                "unittest": {
                    "population_config": test_pop_configs["unittest"],
                    "operator_rates": {
                        "crossover": 0.1,
                        "mutation": 0.4,
                        "discovery": 0.5,
                    },
                },
                "differential": {
                    "population_config": test_pop_configs["differential"],
                    "operator_rates": {
                        "crossover": 0.2,
                        "mutation": 0.2,
                        "discovery": 0.6,
                    },
                },
            },
            "bayesian_config": bayesian_config,
        },
    )

    logger.info(f"Evolution: {evo_config.num_generations} generations")
    logger.info(
        f"Code Population: {code_pop_config.initial_population_size} → {code_pop_config.max_population_size} individuals"
    )
    for test_type, test_config in test_pop_configs.items():
        logger.info(
            f"{test_type.upper()} Test Population: {test_config.initial_population_size} individuals"
        )

    logger.info("\nCreating mock components...")
    components = create_mock_components(code_pop_config, test_pop_configs)
    logger.info("All mock components created successfully")

    logger.info("\nCreating profiles...")
    code_profile, evolved_test_profiles, public_test_profile = create_profiles(
        code_pop_config=code_pop_config,
        test_pop_configs=test_pop_configs,
        bayesian_config=bayesian_config,
        components=components,
    )

    logger.info("\nInitializing Orchestrator...")
    orchestrator = Orchestrator(
        evo_config=evo_config,
        code_profile=code_profile,
        evolved_test_profiles=evolved_test_profiles,
        public_test_profile=public_test_profile,
        execution_system=components["execution_system"],
        bayesian_system=components["bayesian_system"],
        ledger_factory=mock_ledger_factory,
        composer=PythonLanguage().composer,
    )

    logging_utils.log_section_header("INFO", "STARTING COEVOLUTION RUN")

    try:
        with logger.contextualize(run_id=run_id, problem_id=problem.question_id):
            code_population, test_populations = orchestrator.run(problem)

        log_results(code_population, test_populations)

        logger.info("\n" + "=" * 80)
        logger.info("MOCK RUN COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"Error during coevolution run: {e}")
        raise


if __name__ == "__main__":
    app()
