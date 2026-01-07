# run_coevolution.py

"""
Script to run the coevolution orchestrator on LiveCodeBench problems.

Optimized Pattern:
- GLOBAL SCOPE: Initialize heavy infrastructure (LLM, Sandbox, Execution Pools) ONCE.
- RUN SCOPE: Reuse the Orchestrator instance to process multiple problems.
"""

import os
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
from infrastructure.sandbox import SandboxConfig


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
        help="Unique identifier for this run. If not provided, one will be auto-generated.",
    ),
    problem_ids: Optional[list[str]] = typer.Option(
        None,
        "--problem-ids",
        "-p",
        help="List of problem IDs to run. If not provided, runs on a default set.",
    ),
) -> None:
    """Run a coevolution experiment on a LiveCodeBench problem."""
    logging_utils.setup_logging(console_level="DEBUG", file_level="TRACE")

    logging_utils.log_section_header("INFO", "STARTING COEVOLUTION EXPERIMENT")

    # Generate run_id if not provided
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {run_id}")

    # =========================================================================
    # PHASE 1: INFRASTRUCTURE INITIALIZATION (Run Once)
    # =========================================================================
    # These components are heavy and problem-agnostic. We build them once
    # and reuse them to save startup time and resources.

    logger.info("Initializing Global Infrastructure...")

    # 1. LLM Client
    llm_model = "gpt-5-mini"
    llm_client = create_llm_client(
        provider="openai", model=llm_model, reasoning_effort="minimal"
    )
    logger.info(f"Using model: {llm_client.model}")

    # 2. Sandbox Configurations
    differential_sandbox_config = SandboxConfig(
        timeout=20, max_memory_mb=100, max_output_size=10_000_000
    )
    exec_sandbox_config = SandboxConfig(
        timeout=180,
        max_memory_mb=100,
        max_output_size=10_000_000,
        test_method_timeout=30,
    )

    cpu_count = os.cpu_count() or 4
    logger.info(f"Detected CPU Count: {cpu_count}")

    # 3. Execution System (Heavy Resource: Process Pool)
    # Created once so the process pool persists across problems.
    execution_system = ExecutionSystem(
        sandbox_config=exec_sandbox_config,
        enable_multiprocessing=True,
        num_workers=cpu_count,
    )

    # 4. Auxiliary Systems
    bayesian_system = BayesianSystem()
    test_block_rebuilder = LCBTestBlockRebuilder()
    dataset_test_block_builder = LCBDatasetTestBlockBuilder()

    logger.info("Infrastructure components ready.")

    # =========================================================================
    # PHASE 2: STRATEGY CONFIGURATION (Run Once)
    # =========================================================================
    logger.info("Configuring Coevolution Strategy...")

    # 1. Profiles (The "Blueprints" for populations)
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
        max_workers=10,
        diversity_enabled=True,
    )

    unittest_profile = create_unittest_test_profile(
        llm_client=llm_client,
        initial_prior=0.2,
        initial_population_size=20,
        max_population_size=20,
        offspring_rate=0.8,
        elitism_rate=0.4,
        mutation_rate=0.3,
        crossover_rate=0.2,
        edit_rate=0.5,
        alpha=0.2,
        beta=0.2,
        gamma=0.1,
        learning_rate=0.05,
        max_workers=cpu_count,
        diversity_enabled=True,
    )

    differential_profile = create_differential_test_profile(
        llm_client=llm_client,
        sandbox_config=differential_sandbox_config,
        initial_prior=0.2,
        initial_population_size=0,  # Bootstrap
        max_population_size=100,
        offspring_rate=0.5,
        elitism_rate=0.3,
        discovery_rate=1.0,
        alpha=0.3,
        beta=0.5,
        gamma=0.5,
        learning_rate=0.025,
        # Resource Partitioning:
        max_workers=cpu_count,  # Total Budget
        # Logic inside factory splits this (e.g., 4 threads, 3 workers each)
    )

    public_profile = create_public_test_profile(
        alpha=0.001, beta=0.1, gamma=0.1, learning_rate=0.05
    )

    # 2. Schedule
    schedule = (
        ScheduleBuilder()
        .alternating(total_duration=10, code_step=1, test_step=1, start_with="test")
        .build()
    )

    # 3. Orchestrator Configuration
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

    # 4. The Engine (Orchestrator)
    # We build the engine ONCE. It is now ready to accept problems.
    orchestrator = build_orchestrator_from_config(config)
    logger.info("Orchestrator Engine Online.")

    # =========================================================================
    # PHASE 3: PROBLEM EXECUTION LOOP
    # =========================================================================

    all_problems = load_problems()

    # Filter problems
    if problem_ids:
        selected_problems = [p for p in all_problems if p.question_id in problem_ids]
    else:
        selected_problems = all_problems[:10]

    logger.info(f"Processing {len(selected_problems)} problems...")

    for i, problem in enumerate(selected_problems):
        # We use Context Managers to tag logs with the current problem
        # This keeps the logs clean even though we reuse the engine
        with logger.contextualize(problem_id=problem.question_id, run_id=run_id):
            logging_utils.log_section_header(
                "INFO",
                f"PROBLEM {i + 1}/{len(selected_problems)}: {problem.question_id}",
            )
            logging_utils.log_problem(problem)

            try:
                # -------------------------------------------------------------
                # THE CORE EXECUTION
                # -------------------------------------------------------------
                # The orchestrator is stateless regarding the problem.
                # It initializes fresh populations for this specific problem
                # using the profiles we configured earlier.
                code_population, evolved_test_populations = orchestrator.run(problem)

                # -------------------------------------------------------------
                # RESULT LOGGING
                # -------------------------------------------------------------
                best_code = code_population.get_best_individual()

                if best_code:
                    logger.info(f"Best Code P(Correct): {best_code.probability:.4f}")
                    logger.info(f"Best Code ID: {best_code.id}")

                logger.info(f"Final Code Pop Size: {code_population.size}")

                # Log Code Solution
                logging_utils.log_subsection_header("INFO", "BEST CODE SOLUTION")
                if best_code:
                    logger.info(best_code.snippet)
                else:
                    logger.warning("No valid code solution found.")

                # Log Test Stats
                for t_type, t_pop in evolved_test_populations.items():
                    best_t = t_pop.get_best_individual()
                    logger.info(
                        f"[{t_type}] Best Test P(Valid): {best_t.probability:.4f}"
                        if best_t
                        else f"[{t_type}] No valid tests"
                    )
                    logger.info(f"[{t_type}] Final Pop Size: {t_pop.size}")

            except Exception as e:
                logger.error(
                    f"Failed to process problem {problem.question_id}: {e}",
                    exc_info=True,
                )
                # We catch the exception so one bad problem doesn't kill the whole batch
                continue

    logging_utils.log_section_header("INFO", "BATCH EXPERIMENT COMPLETE")


if __name__ == "__main__":
    typer.run(main)
