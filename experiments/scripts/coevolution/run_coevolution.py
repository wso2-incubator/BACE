"""
Script to run the coevolution orchestrator on LiveCodeBench problems.

This script demonstrates the new profile-based orchestrator builder pattern:
- Profile factory functions for creating standard configurations
- OrchestratorBuilder with fluent API for flexible assembly
- Multiple test population types (unittest, differential, public)
- Clean separation of concerns via dependency injection

The new architecture supports:
- Multiple evolved test populations with different strategies
- Configurable breeding, selection, and Bayesian parameters per population
- Easy customization through profile factories
- Type-safe configuration with comprehensive validation
"""

from datetime import datetime
from typing import Optional

import typer
from loguru import logger

import coevolution.utils.logging as logging_utils
from coevolution.adapters.lcb import (
    Difficulty,
    LCBCodeGenerationProblem,
    LCBDatasetTestBlockBuilder,
    LCBTestBlockRebuilder,
    load_code_generation_dataset,
)
from coevolution.factories import (
    OrchestratorBuilder,
    ScheduleBuilder,
    build_orchestrator_from_config,
    create_default_code_profile,
    create_differential_test_profile,
    create_public_test_profile,
    create_unittest_test_profile,
)
from coevolution.services.bayesian import BayesianSystem
from coevolution.services.execution import ExecutionSystem
from infrastructure.llm_client import create_llm_client
from infrastructure.sandbox import SafeCodeSandbox, SandboxConfig


def load_problems() -> list[LCBCodeGenerationProblem]:
    """Load a problem from LiveCodeBench dataset."""
    logger.info("Loading problem from LiveCodeBench dataset...")

    problems = load_code_generation_dataset(
        release_version="release_v6",
        start_date="2025-03-01",
        end_date="2025-05-10",
        difficulty=Difficulty.HARD,
    )

    if not problems:
        raise ValueError("No problems found matching criteria")

    return problems


def main(
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Unique identifier for this run. If not provided, one will be auto-generated with timestamp.",
    ),
    problem_ids: Optional[list[str]] = typer.Option(
        None,
        "--problem-ids",
        "-p",
        help="List of problem IDs to run. If not provided, runs on a default set of problems.",
    ),
) -> None:
    """Run a coevolution experiment on a LiveCodeBench problem."""
    logging_utils.setup_logging(console_level="DEBUG", file_level="TRACE")

    logging_utils.log_section_header("INFO", "STARTING COEVOLUTION EXPERIMENT")

    # Generate run_id if not provided
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {run_id}")

    # ====================================
    # Step 1: Create LLM and Sandbox
    # ====================================

    # LLM and Sandbox created outside problem loop to reuse across problems
    # There's token limit in llm and sandbox resource limitations
    logger.info("Creating LLM client and sandbox...")

    llm_model = "gpt-5-mini"
    llm_provider = "openai"
    llm_client = create_llm_client(
        provider=llm_provider, model=llm_model, reasoning_effort="minimal"
    )
    logger.info(f"Using model: {llm_client.model}")

    sandbox_for_differential = SafeCodeSandbox(
        timeout=20, max_memory_mb=256, max_output_size=10_000_000
    )

    exec_sandbox_config = SandboxConfig(
        timeout=180,
        max_memory_mb=100,
        max_output_size=10_000_000,
        test_method_timeout=30,
    )

    logger.info("Sandbox created")

    # ====================================
    # Step 2: Load Problems
    # ====================================
    problems = load_problems()
    if problem_ids is not None:
        selected_problems = [
            problem for problem in problems if problem.question_id in problem_ids
        ]
    else:
        selected_problems = problems[:10]  # Default to first 10 problems

    for problem in selected_problems:
        with logger.contextualize(problem_id=problem.question_id, run_id=run_id):
            logger.info(f"Running coevolution on problem ID: {problem.question_id}")
            logging_utils.log_problem(problem)

            # ====================================
            # Step 3: Create Infrastructure Components
            # ====================================
            logger.info("Creating infrastructure components...")

            # Execution System
            execution_system = ExecutionSystem(
                sandbox_config=exec_sandbox_config,
                enable_multiprocessing=True,
                num_workers=12,
            )

            # Bayesian System (unified for all populations)
            bayesian_system = BayesianSystem()

            # Test Block Rebuilder
            test_block_rebuilder = LCBTestBlockRebuilder()

            # Dataset Test Block Builder
            dataset_test_block_builder = LCBDatasetTestBlockBuilder()

            logger.info("Infrastructure components created successfully")

            # ====================================
            # Step 4: Create Population Profiles
            # ====================================
            logger.info("Creating population profiles using factory functions...")

            # Code Profile: Variable-size population with diversity-aware selection
            code_profile = create_default_code_profile(
                llm_client=llm_client,
                initial_prior=0.2,
                initial_population_size=10,
                max_population_size=15,
                offspring_rate=0.3,
                elitism_rate=0.3,
                mutation_rate=0.2,
                crossover_rate=0.2,
                edit_rate=0.6,
                max_workers=10,  # Parallel breeding
                diversity_enabled=True,  # use top-k selection based on elitism rate only
            )

            # Unittest Profile: Fixed-size population with discrimination-driven edit
            unittest_profile = create_unittest_test_profile(
                llm_client=llm_client,
                initial_prior=0.2,
                initial_population_size=20,
                max_population_size=20,  # Fixed size
                offspring_rate=0.8,
                elitism_rate=0.4,
                mutation_rate=0.3,
                crossover_rate=0.2,
                edit_rate=0.5,
                alpha=0.2,
                beta=0.2,
                gamma=0.1,
                learning_rate=0.05,
                max_workers=10,
                diversity_enabled=True,
            )

            # Differential Profile: Bootstrap from empty with discovery-based growth
            differential_profile = create_differential_test_profile(
                llm_client=llm_client,
                sandbox=sandbox_for_differential,
                initial_prior=0.2,
                initial_population_size=0,  # Bootstrap mode
                max_population_size=100,
                offspring_rate=0.5,
                elitism_rate=0.3,
                discovery_rate=1.0,  # Discovery finds divergent code pairs
                alpha=0.3,  # Lower reliability than unittest
                beta=0.5,
                gamma=0.5,
                learning_rate=0.025,
                max_workers=10,
                diversity_enabled=True,
            )

            # Public Test Profile: Ground-truth anchoring tests
            public_profile = create_public_test_profile(
                alpha=0.001,  # Very high reliability
                beta=0.1,
                gamma=0.1,
                learning_rate=0.05,
            )

            logger.info("All profiles created successfully")
            logger.info(
                f"  Code: {code_profile.population_config.initial_population_size} → "
                f"{code_profile.population_config.max_population_size} individuals"
            )
            logger.info(
                f"  Unittest: {unittest_profile.population_config.initial_population_size} tests (fixed)"
            )
            logger.info(
                f"  Differential: {differential_profile.population_config.initial_population_size} → "
                f"{differential_profile.population_config.max_population_size} tests (bootstrap)"
            )

            # ====================================
            # Step 5: Build Orchestrator Configuration
            # ====================================
            logger.info("Building orchestrator configuration...")

            # build schedule
            schedule = (
                ScheduleBuilder()
                .alternating(
                    total_duration=10,
                    code_step=1,
                    test_step=1,
                    start_with="test",
                )
                .build()
            )

            config = (
                OrchestratorBuilder()
                .with_evolution_config(schedule)
                .with_code_profile(code_profile)
                .add_test_profile("unittest", unittest_profile)
                .add_test_profile("differential", differential_profile)
                .with_public_test_profile(public_profile)
                .with_execution_system(execution_system)
                .with_bayesian_system(bayesian_system)
                .with_test_block_rebuilder(test_block_rebuilder)
                .with_dataset_test_block_builder(dataset_test_block_builder)
                .build()
            )

            logger.info("Configuration validated and built successfully")

            # ====================================
            # Step 6: Create Orchestrator
            # ====================================
            orchestrator = build_orchestrator_from_config(config)
            logger.info("Orchestrator created successfully")

            # ====================================
            # Step 7: Run Coevolution
            # ====================================
            logger.info("Starting coevolution run...")
            code_population, evolved_test_populations = orchestrator.run(problem)

            # ====================================
            # Step 8: Display Results
            # ====================================
            best_code = code_population.get_best_individual()

            if best_code:
                logger.info(f"Best code probability: {best_code.probability:.4f}")
                logger.info(f"Best code ID: {best_code.id}")

            logger.info(f"Final code population size: {code_population.size}")
            logger.info(
                f"Code population avg probability: {code_population.compute_average_probability():.4f}"
            )

            # Display results for each evolved test population
            for test_type, test_population in evolved_test_populations.items():
                best_test = test_population.get_best_individual()

                if best_test:
                    logger.info(
                        f"Best {test_type} test probability: {best_test.probability:.4f}"
                    )
                    logger.info(f"Best {test_type} test ID: {best_test.id}")

                logger.info(
                    f"Final {test_type} population size: {test_population.size}"
                )
                logger.info(
                    f"{test_type} population avg probability: "
                    f"{test_population.compute_average_probability():.4f}"
                )

            # Log best solutions
            logging_utils.log_subsection_header("INFO", "BEST CODE SOLUTION")
            if best_code:
                logger.info(f"ID: {best_code.id}")
                logger.info(best_code.snippet)
            else:
                logger.info("No code solution found")

            # Log best tests for each type
            for test_type, test_population in evolved_test_populations.items():
                best_test = test_population.get_best_individual()
                logging_utils.log_subsection_header(
                    "INFO", f"BEST {test_type.upper()} TEST CASE"
                )
                if best_test:
                    logger.info(f"ID: {best_test.id}")
                    logger.info(best_test.snippet)
                else:
                    logger.info(f"No {test_type} test case found")

                logging_utils.log_subsection_header(
                    "INFO", f"{test_type.upper()} TEST CLASS BLOCK"
                )
                logger.info(test_population.test_class_block)

    logging_utils.log_section_header("INFO", "END OF EXPERIMENT")


if __name__ == "__main__":
    typer.run(main)
