#!/usr/bin/env python3
"""
Main entry point for APR coevolution experiments.

Supports YAML-based configuration with modular, composable configs.
Allows CLI overrides for common parameters.

Usage:
    # Use full experiment config
    uv run python main.py run --config configs/experiments/default.yam    uv run python main.py run --config configs/experiments/default.yamll

    # Override specific components
    uv run python main.py run --config configs/experiments/default.yaml \
        --llm configs/llm/gpt-4.yaml

    # Override problem subset parameters
    uv run python main.py run --config configs/experiments/default.yaml \
        --start-index 100 --end-index 200

    # Quick test with specific problems
    uv run python main.py run --config configs/experiments/quick-test.yaml \
        --problem-ids HumanEval/1 HumanEval/2

Note: Dataset-specific parameters (version, difficulty, dates) are now configured
      in configs/datasets/*.yaml instead of via CLI flags.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
from loguru import logger

from coevolution.dataset import get_adapter
from coevolution.factories import (
    OrchestratorBuilder,
    PopulationDiscoveryService,
    ScheduleBuilder,
    build_orchestrator_from_config,
)
from coevolution.services.bayesian import BayesianSystem
from coevolution.services.execution import ExecutionSystem
from coevolution.utils import config as config_utils
from coevolution.utils import logging as logging_utils
from coevolution.utils.rephrasing import ProblemRephraser
from infrastructure.languages import create_language_adapter
from infrastructure.llm_client import create_llm_client
from infrastructure.sandbox import SandboxConfig

app = typer.Typer(help="APR Coevolution Experiment Runner")


@app.command()
def run(
    # === Core Configuration ===
    config: str = typer.Option(
        "configs/experiments/default.yaml",
        "--config",
        "-c",
        help="Path to experiment configuration file",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Unique identifier for this run (auto-generated if not provided)",
    ),
    # === Component Overrides ===
    llm: Optional[str] = typer.Option(
        None,
        "--llm",
        help="Override LLM config (e.g., configs/llm/gpt-4.yaml)",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Override dataset config (e.g., configs/datasets/lcb.yaml or configs/datasets/humaneval_ballerina.yaml)",
    ),
    code_profile: Optional[str] = typer.Option(
        None,
        "--code-profile",
        help="Override code profile config",
    ),
    # === Subset Overrides ===
    problem_ids: Optional[list[str]] = typer.Option(
        None,
        "--problem-ids",
        "-p",
        help="Specific problem IDs to run (overrides index range)",
    ),
    start_index: Optional[int] = typer.Option(
        None,
        "--start-index",
        "-s",
        help="Override start index for problem subset",
    ),
    end_index: Optional[int] = typer.Option(
        None,
        "--end-index",
        "-e",
        help="Override end index for problem subset",
    ),
    # === Logging Overrides ===
    console_level: Optional[str] = typer.Option(
        None,
        "--console-level",
        help="Override console log level (DEBUG, INFO, WARNING, ERROR)",
    ),
    language: str = typer.Option(
        None,
        "--language",
        "-l",
        help="Target programming language for the experiment",
    ),
    # === Dry Run ===
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Load and validate config without running experiment",
    ),
    # === Resume Functionality ===
    resume: bool = typer.Option(
        False,
        "--resume",
        "-R",
        help="Resume an incomplete run by skipping already processed problems.",
    ),
) -> None:
    """
    Run a coevolution experiment using YAML configuration.

    The config file can reference other modular config files (e.g., LLM, profiles).
    CLI arguments override config file values for easy experimentation.
    """

    # =================================================================
    # 1. LOAD CONFIGURATION
    # =================================================================
    try:
        experiment_config = config_utils.load_experiment_config(config)
    except config_utils.ConfigError as e:
        typer.echo(f"Error loading config: {e}", err=True)
        raise typer.Exit(code=1)

    # =================================================================
    # 2. APPLY CLI OVERRIDES
    # =================================================================
    overrides: dict[str, Any] = {}

    # Component overrides
    if llm:
        llm_config = config_utils._load_yaml_file(Path(llm))
        overrides["llm"] = llm_config

    if dataset:
        dataset_config = config_utils._load_yaml_file(Path(dataset))
        overrides["dataset"] = dataset_config

    if code_profile:
        code_profile_config = config_utils._load_yaml_file(Path(code_profile))
        overrides["code_profile"] = code_profile_config

    # Subset overrides
    if problem_ids is not None:
        overrides["subset.problem_ids"] = problem_ids
    if start_index is not None:
        overrides["subset.start_index"] = start_index
    if end_index is not None:
        overrides["subset.end_index"] = end_index

    # Logging overrides
    if console_level:
        overrides["logging.console_level"] = console_level

    if language:
        overrides["language"] = language

    # Apply overrides
    if overrides:
        experiment_config = config_utils.apply_cli_overrides(
            experiment_config, overrides
        )

    # =================================================================
    # 3. VALIDATE CONFIGURATION
    # =================================================================

    # =================================================================
    # 4. DRY RUN MODE
    # =================================================================
    if dry_run:
        typer.echo("=== Dry Run: Configuration Loaded Successfully ===")
        typer.echo(json.dumps(experiment_config, indent=2, default=str))
        return

    # =================================================================
    # 5. SETUP LOGGING
    # =================================================================
    # Generate run_id if not provided
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    log_config = experiment_config.get("logging", {})
    run_id = logging_utils.setup_logging(
        console_level=log_config.get("console_level", "INFO"),
        file_level=log_config.get("file_level", "TRACE"),
        log_file_base_name=log_config.get("log_file_base_name", "coevolution_run"),
        run_id=run_id,
        resume=resume,
    )

    # Save run config for dashboard accessibility
    logging_utils.save_run_config(run_id, experiment_config)

    logger.bind(run_id=run_id)
    logging_utils.log_section_header("INFO", "STARTING COEVOLUTION EXPERIMENT")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Config: {config}")
    logger.info(
        f"Experiment: {experiment_config.get('experiment', {}).get('name', 'unnamed')}"
    )

    # =================================================================
    # 7. RUN EXPERIMENT
    # =================================================================
    try:
        _run_experiment(experiment_config, run_id, resume=resume)

        logging_utils.log_section_header("INFO", "EXPERIMENT COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


def _run_experiment(config: dict[str, Any], run_id: str, resume: bool = False) -> None:
    """
    Execute the coevolution experiment using the provided configuration.

    Args:
        config: Parsed configuration dictionary
        run_id: Unique identifier for this run
    """
    import os

    # Extract config sections
    llm_config = config.get("llm", {})
    sandbox_config = config.get("sandbox", {})
    schedule_config = config.get("schedule", {})
    dataset_config = config.get("dataset", {})
    subset_config = config.get("subset", {})
    log_config = config.get("logging", {})

    # =========================================================================
    # PHASE 1: INFRASTRUCTURE INITIALIZATION
    # =========================================================================
    logger.info("Initializing Global Infrastructure from config...")

    # 1. LLM Client
    llm_client = create_llm_client(
        provider=llm_config.get("provider", "openai"),
        model=llm_config.get("model", "gpt-5-mini"),
        reasoning_effort=llm_config.get("reasoning_effort", "minimal"),
        max_output_tokens=llm_config.get("max_output_tokens", None),
        enable_token_limit=llm_config.get("enable_token_limit", None),
    )
    logger.info(f"Using model: {llm_client.model}")

    # 2. Sandbox Configurations
    # Preference order:
    #  1) CLI argument (applied into `experiment.language` via overrides)
    #  2) `experiment.language` in the config
    #  3) default to "python"
    language = (
        config.get("language")
        or config.get("experiment", {}).get("language")
        or "python"
    )
    logger.info(f"Target Language: {language}")

    sandbox_diff_config = sandbox_config.get("differential", {})
    differential_sandbox_config = SandboxConfig(
        language=language,
        timeout=sandbox_diff_config.get("timeout", 30),
        max_memory_mb=sandbox_diff_config.get("max_memory_mb", 1024 * 2),
        max_output_size=sandbox_diff_config.get("max_output_size", 10_000_000),
        test_method_timeout=sandbox_diff_config.get("test_method_timeout", 30),
    )

    sandbox_exec_config = sandbox_config.get("execution", {})
    exec_sandbox_config = SandboxConfig(
        language=language,
        timeout=sandbox_exec_config.get("timeout", 30),
        max_memory_mb=sandbox_exec_config.get("max_memory_mb", 1024 * 2),
        max_output_size=sandbox_exec_config.get("max_output_size", 10_000_000),
        test_method_timeout=sandbox_exec_config.get("test_method_timeout", 30),
    )

    # Worker count from config or environment
    cpu_count = int(
        os.environ.get(
            "COEVOLUTION_WORKERS",
            sandbox_config.get("workers", {}).get("cpu_count", 4),
        )
    )
    logger.info(f"Using worker count: {cpu_count}")

    # 3. Language Adapter
    language_adapter = create_language_adapter(language)

    # 4. Execution System
    execution_system = ExecutionSystem(
        sandbox_config=exec_sandbox_config,
        enable_multiprocessing=True,
        cpu_workers=cpu_count,
        composer=language_adapter.composer,
        runtime=language_adapter.runtime,
        analyzer=language_adapter.analyzer,
    )

    # 4. Auxiliary Systems
    bayesian_system = BayesianSystem()

    logger.info("Infrastructure components ready.")

    # =========================================================================
    # PHASE 2: STRATEGY CONFIGURATION
    # =========================================================================
    logger.info("Configuring Coevolution Strategy from config...")

    # 1. Discovery Service
    discovery_service = PopulationDiscoveryService(
        llm_client=llm_client,
        language_adapter=language_adapter,
        execution_system=execution_system,
        sandbox_config=differential_sandbox_config,
        cpu_workers=cpu_count,
    )

    # 2. Construct Profiles
    profiles = discovery_service.construct_all(config)
    code_profile = profiles["code_profile"]
    evolved_test_profiles = profiles["evolved_test_profiles"]
    public_test_profile = profiles["public_test_profile"]

    if not code_profile:
        raise ValueError("No code profile configured or constructed.")

    # 3. Schedule
    schedule = ScheduleBuilder.from_config(schedule_config)

    # 4. Orchestrator Configuration
    builder = (
        OrchestratorBuilder()
        .with_composer(language_adapter.composer)
        .with_evolution_config(schedule)
        .with_code_profile(code_profile)
    )

    # Add evolved test profiles
    for test_type, profile in evolved_test_profiles.items():
        builder = builder.add_test_profile(test_type, profile)

    # Add public test profile if it exists
    if public_test_profile:
        builder = builder.with_public_test_profile(public_test_profile)

    orchestrator_config = (
        builder.with_execution_system(execution_system)
        .with_bayesian_system(bayesian_system)
        .build()
    )

    # 5. The Engine
    orchestrator = build_orchestrator_from_config(orchestrator_config)
    logger.info("Orchestrator Engine Online.")

    # Runtime rephraser configuration
    rephraser_cfg = config.get("rephraser", {})
    rephraser_enabled = rephraser_cfg.get("enabled", False)
    rephraser: ProblemRephraser | None = None
    if rephraser_enabled:
        rephraser = ProblemRephraser(
            llm_client,
            n_rephrasings=rephraser_cfg.get("n", 3),
        )

    # =========================================================================
    # PHASE 3: PROBLEM LOADING & SELECTION
    # =========================================================================

    # Load dataset using adapter (config resolver inlines referenced files)
    dataset_config = config.get("dataset", {})

    adapter_name = dataset_config.get("adapter", "lcb")
    logger.info(f"Using dataset adapter: {adapter_name}")

    adapter = get_adapter(adapter_name)
    all_problems = adapter.load_dataset(dataset_config)

    if not all_problems:
        logger.error("Dataset empty. Aborting run.")
        return

    logger.info(f"Loaded {len(all_problems)} problems from dataset.")

    # Select subset
    problem_ids = subset_config.get("problem_ids")
    start_index = subset_config.get("start_index", 0)
    end_index = subset_config.get("end_index")

    if problem_ids:
        logger.info(f"Running specific list of {len(problem_ids)} problems.")
        selected_problems = [p for p in all_problems if p.question_id in problem_ids]

        if not selected_problems:
            logger.warning("None of the requested problem IDs found in dataset.")
            return
    else:
        # Select by index range
        total_available = len(all_problems)
        actual_end = end_index if end_index is not None else total_available
        actual_start = max(0, start_index)
        actual_end = min(total_available, actual_end)

        if actual_start >= actual_end:
            logger.warning(
                f"Start index ({actual_start}) >= End index ({actual_end}). "
                "No problems to process in this range."
            )
            return

        selected_problems = all_problems[actual_start:actual_end]

        logger.info(
            f"Subset Configuration:\n"
            f"  - Total Problems Available: {total_available}\n"
            f"  - Selected Range: [{actual_start}:{actual_end}]\n"
            f"  - Subset Size: {len(selected_problems)}"
        )

    # =========================================================================
    # PHASE 4: EXECUTION LOOP
    # =========================================================================

    logger.info(f"Processing {len(selected_problems)} problems...")
    # Token accounting across problems
    total_tokens_used = 0
    successful_problems = 0

    for i, problem in enumerate(selected_problems):
        global_idx = (start_index if not problem_ids else 0) + i

        # Check for completion if resume is enabled
        if resume:
            if logging_utils.is_problem_completed(run_id, problem.question_id):
                logger.info(
                    f"[{i+1}/{len(selected_problems)}] Skipping already completed problem: {problem.question_id}"
                )
                continue
            else:
                # If file exists but not complete, we override it for a fresh start
                # (as requested: "we SHOULD override it")
                from coevolution.utils.paths import sanitize_id

                sanitized_run_id = sanitize_id(run_id)
                sanitized_pid = sanitize_id(problem.question_id)
                history_path = (
                    Path("logs")
                    / sanitized_run_id
                    / sanitized_pid
                    / "evolutionary_history.jsonl"
                )
                if history_path.exists():
                    logger.warning(
                        f"[{i+1}/{len(selected_problems)}] Problem {problem.question_id} was interrupted. Overriding partial history log."
                    )
                    history_path.unlink()

        # Re-initialize logging for this specific problem context.
        # This updates the global environment so child workers know which directory to log to.
        run_id = logging_utils.setup_logging(
            console_level=log_config.get("console_level", "INFO"),
            file_level=log_config.get("file_level", "DEBUG"),
            run_id=run_id,
            problem_id=problem.question_id,
            resume=resume,
        )

        with logger.contextualize(problem_id=problem.question_id, run_id=run_id):
            logging_utils.log_section_header(
                "INFO",
                f"PROBLEM {i + 1}/{len(selected_problems)} "
                f"(Global Index: {global_idx}): {problem.question_id}",
            )
            logging_utils.log_problem(problem)
            # Capture token count before processing this problem
            before_tokens = llm_client.total_output_tokens

            # Generate runtime-only rephrasings if enabled (no caching)
            if rephraser_enabled and rephraser is not None:
                try:
                    problem = rephraser.generate_rephrasings(problem)
                except Exception as e:
                    logger.warning(f"Problem rephrasing failed: {e}")

            try:
                code_population, evolved_test_populations = orchestrator.run(problem)

                best_code = code_population.get_best_individual()

                if best_code:
                    logger.info(f"Best Code P(Correct): {best_code.probability:.4f}")
                    logger.info(f"Best Code ID: {best_code.id}")

                logger.info(f"Final Code Pop Size: {code_population.size}")

                logging_utils.log_subsection_header("INFO", "BEST CODE SOLUTION")
                if best_code:
                    logger.info(best_code.snippet)
                else:
                    logger.warning("No valid code solution found.")

                for t_type, t_pop in evolved_test_populations.items():
                    best_t = t_pop.get_best_individual()
                    logger.info(
                        f"[{t_type}] Best Test P(Valid): {best_t.probability:.4f}"
                        if best_t
                        else f"[{t_type}] No valid tests"
                    )
                    logger.info(f"[{t_type}] Final Pop Size: {t_pop.size}")

            except Exception as e:
                error_msg = str(e).replace("{", "{{").replace("}", "}}")
                logger.error(
                    f"Failed to process problem {problem.question_id}: {error_msg}",
                    exc_info=True,
                )

                # Log tokens consumed by the failed attempt, but do NOT include them in totals
                after_tokens = llm_client.total_output_tokens
                failed_used = max(0, after_tokens - before_tokens)
                if failed_used > 0:
                    logger.info(
                        f"Tokens consumed during failed attempt (not counted): {failed_used}"
                    )
            else:
                # Success path: account tokens and update running average
                after_tokens = llm_client.total_output_tokens
                used = max(0, after_tokens - before_tokens)
                total_tokens_used += used
                successful_problems += 1

                avg_tokens = (
                    total_tokens_used / successful_problems
                    if successful_problems > 0
                    else 0
                )

                logger.info(f"Tokens used for problem {problem.question_id}: {used}")
                logger.info(
                    f"Average tokens per successful problem (so far): {avg_tokens:.2f}"
                )

    logging_utils.log_section_header("INFO", "BATCH EXPERIMENT COMPLETE")


@app.command()
def list_configs(
    config_type: str = typer.Argument(
        ...,
        help="Type of configs to list: llm, code, unittest, differential, public, sandbox, schedule, experiment",
    ),
) -> None:
    """List available configuration files of a specific type."""

    type_mapping = {
        "llm": "llm",
        "code": "profiles/code",
        "unittest": "profiles/unittest",
        "differential": "profiles/differential",
        "public": "profiles/public",
        "sandbox": "sandbox",
        "schedule": "schedules",
        "experiment": "experiments",
    }

    if config_type not in type_mapping:
        typer.echo(f"Unknown config type: {config_type}", err=True)
        typer.echo(f"Available types: {', '.join(type_mapping.keys())}")
        raise typer.Exit(code=1)

    config_dir = Path("configs") / type_mapping[config_type]

    if not config_dir.exists():
        typer.echo(f"Config directory not found: {config_dir}", err=True)
        raise typer.Exit(code=1)

    configs = sorted(config_dir.glob("*.yaml"))

    if not configs:
        typer.echo(f"No configs found in {config_dir}")
        return

    typer.echo(f"Available {config_type} configs:")
    for cfg in configs:
        typer.echo(f"  - {cfg.relative_to('configs')}")


if __name__ == "__main__":
    app()
