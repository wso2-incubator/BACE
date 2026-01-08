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


def load_problems(
    release_version: str = "release_v6",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    difficulty: Optional[Difficulty] = None,
) -> list[LCBCodeGenerationProblem]:
    """Load problems from LiveCodeBench dataset.

    Args:
        release_version: Dataset version to load
        start_date: Filter problems from this date onwards (YYYY-MM-DD)
        end_date: Filter problems up to this date (YYYY-MM-DD)
        difficulty: Filter by difficulty level

    Returns:
        List of problems matching the criteria
    """
    logger.info("Loading problems from LiveCodeBench dataset...")

    problems = load_code_generation_dataset(
        release_version=release_version,
        start_date=start_date,
        end_date=end_date,
        difficulty=difficulty,
    )

    if not problems:
        raise ValueError("No problems found matching criteria")

    logger.info(f"Loaded {len(problems)} problems from dataset.")
    return problems


def main(
    # --- Experiment Identity ---
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Unique identifier for this run. If not provided, one will be auto-generated.",
    ),
    # --- Dataset Configuration ---
    dataset_version: str = typer.Option(
        "release_v6",
        "--dataset-version",
        "-v",
        help="LiveCodeBench dataset version to use.",
    ),
    start_date: str = typer.Option(
        "2025-03-01",
        "--start-date",
        help="Filter problems from this date onwards (YYYY-MM-DD).",
    ),
    end_date: str = typer.Option(
        "2025-05-10",
        "--end-date",
        help="Filter problems up to this date (YYYY-MM-DD).",
    ),
    difficulty: Difficulty = typer.Option(
        Difficulty.HARD,
        "--difficulty",
        "-d",
        help="Filter problems by difficulty level.",
    ),
    # --- Selection Filters ---
    problem_ids: Optional[list[str]] = typer.Option(
        None,
        "--problem-ids",
        "-p",
        help="Specific problem IDs to run (overrides index slicing).",
    ),
    # --- Sharding Controls ---
    start_index: int = typer.Option(
        0,
        "--start-index",
        "-s",
        help="Index of the first problem to process (for sharding).",
    ),
    end_index: Optional[int] = typer.Option(
        None,
        "--end-index",
        "-e",
        help="Index of the last problem to process (exclusive). If omitted, processes until the end.",
    ),
) -> None:
    """Run a coevolution experiment on LiveCodeBench problems.

    Supports horizontal scaling via sharding:
    - Use --start-index and --end-index to process a slice of the dataset
    - Use --problem-ids to run specific problems (overrides slicing)

    Examples:
        # Process first 100 problems
        python run_coevolution.py --start-index 0 --end-index 100

        # Process problems 100-200 (for parallel execution)
        python run_coevolution.py --start-index 100 --end-index 200

        # Process from index 500 to end
        python run_coevolution.py --start-index 500

        # Run specific problems
        python run_coevolution.py --problem-ids Q123 Q456 Q789
    """
    logging_utils.setup_logging(console_level="DEBUG", file_level="DEBUG")

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
    # PHASE 3: PROBLEM LOADING & SELECTION
    # =========================================================================

    # 1. Load the raw dataset
    all_problems = load_problems(
        release_version=dataset_version,
        start_date=start_date,
        end_date=end_date,
        difficulty=difficulty,
    )

    if not all_problems:
        logger.error("Dataset empty. Aborting run.")
        return

    # 2. Select specific problems (Priority 1)
    if problem_ids:
        logger.info(f"Running specific list of {len(problem_ids)} problems.")
        selected_problems = [p for p in all_problems if p.question_id in problem_ids]

        if not selected_problems:
            logger.warning("None of the requested problem IDs found in dataset.")
            return

    # 3. Slice for Sharding (Priority 2)
    else:
        # Validate indices
        total_available = len(all_problems)

        # Handle 'None' end_index (process till end)
        actual_end = end_index if end_index is not None else total_available

        # Clamp values to prevent index errors
        actual_start = max(0, start_index)
        actual_end = min(total_available, actual_end)

        if actual_start >= actual_end:
            logger.warning(
                f"Start index ({actual_start}) >= End index ({actual_end}). "
                "No problems to process in this shard."
            )
            return

        # Python List Slicing
        selected_problems = all_problems[actual_start:actual_end]

        logger.info(
            f"Shard Configuration:\n"
            f"  - Total Problems Available: {total_available}\n"
            f"  - Shard Range: [{actual_start}:{actual_end}]\n"
            f"  - Shard Size: {len(selected_problems)}"
        )

    # =========================================================================
    # PHASE 4: EXECUTION LOOP
    # =========================================================================

    logger.info(f"Processing {len(selected_problems)} problems...")

    for i, problem in enumerate(selected_problems):
        # Calculate global index for logging clarity
        global_idx = (start_index if not problem_ids else 0) + i

        # We use Context Managers to tag logs with the current problem
        # This keeps the logs clean even though we reuse the engine
        with logger.contextualize(problem_id=problem.question_id, run_id=run_id):
            logging_utils.log_section_header(
                "INFO",
                f"PROBLEM {i + 1}/{len(selected_problems)} "
                f"(Global Index: {global_idx}): {problem.question_id}",
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
