"""
Batch script to run the coevolution orchestrator on multiple problems.

This script loads a set of problems from a dataset (e.g., LiveCodeBench),
configures the coevolution algorithm, and runs the orchestrator for each
problem, logging the results.
"""

import sys
import time

from lcb_runner.benchmarks import code_generation
from loguru import logger

# Assuming the following modules exist in the specified structure
from common.coevolution.config import CoevolutionConfig
from common.coevolution.orchestrator import CoevolutionOrchestrator
from common.llm_client import LLMClient, create_llm_client
from common.sandbox import SafeCodeSandbox, create_safe_test_environment


def setup_logging(
    log_level: str = "INFO", file_log_level: str = "TRACE"
) -> None:  # Changed default file level to TRACE
    """Configure Loguru loggers."""
    logger.remove()  # Remove default handler

    # Console logger - Keep custom format for readability
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        colorize=True,
    )
    # File logger (detailed) - Use default format, capture TRACE level
    logger.add(
        "logs/experiment_run_{time:YYYYMMDD}.log",
        level=file_log_level.upper(),  # Set to TRACE level
        rotation="100 MB",
        retention="10 days",
        compression="zip",
        enqueue=True,
        colorize=True,
    )
    logger.info("Logging configured.")


def run_single_problem(
    problem: code_generation.CodeGenerationProblem,
    config: CoevolutionConfig,
    llm_client: LLMClient,
    sandbox: SafeCodeSandbox,
) -> str:
    """
    Runs the coevolution orchestrator for a single problem.

    Args:
        problem: The CodeGenerationProblem instance.
        config: The CoevolutionConfig instance.
        llm_client: The LLMClient instance to use.
        sandbox: The SafeCodeSandbox instance for code execution.

    Returns:
        A tuple (best_code_prob, best_test_prob) or (None, None) if execution fails.
    """
    problem_id = problem.question_id
    logger.info(f"--- Starting Problem: {problem_id} ({problem.question_title}) ---")
    start_time = time.time()

    try:
        # Create and run orchestrator
        orchestrator = CoevolutionOrchestrator(
            config=config, problem=problem, llm_client=llm_client, sandbox=sandbox
        )

        final_code_pop, final_test_pop = orchestrator.run()

        # Get best results
        best_code, best_code_prob = final_code_pop.get_best_individual()
        best_test, best_test_prob = final_test_pop.get_best_individual()

        end_time = time.time()
        duration = end_time - start_time
        logger.success(
            f"--- Finished Problem: {problem_id} in {duration:.2f}s --- "
            f"Best Code Prob: {best_code_prob:.4f}, Best Test Prob: {best_test_prob:.4f}"
        )
        logger.trace(f"Problem {problem_id} - Best Code:\n{best_code}")
        logger.trace(f"Problem {problem_id} - Best Test:\n{best_test}")

        logger.trace("Final code population:")
        for i, (individual, prob) in enumerate(final_code_pop):
            logger.trace(f"Individual {i}:\n{individual}\nProbability: {prob:.4f}")

        logger.trace("Final test population:")
        for i, (individual, prob) in enumerate(final_test_pop):
            logger.trace(f"Individual {i}:\n{individual}\nProbability: {prob:.4f}")

        return best_code

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(
            f"--- Error processing Problem: {problem_id} after {duration:.2f}s --- : {e}",
            exc_info=True,
        )
        return None


def main_batch(num_problems: int = 20) -> None:
    """Run a coevolution experiment over multiple problems."""

    setup_logging(log_level="DEBUG", file_log_level="TRACE")

    logger.info("=" * 80)
    logger.info(f"Starting Batch Coevolution Experiment for {num_problems} Problems")
    logger.info("=" * 80)

    # Step 1: Load dataset
    logger.info("Loading LiveCodeBench dataset...")
    try:
        dataset = code_generation.load_code_generation_dataset(
            release_version="release_v6",
            start_date="2024-02-01",
            difficulty=code_generation.Difficulty.HARD,
        )
        logger.info(f"Loaded {len(dataset)} problems from dataset.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        return

    # Select the first N problems
    problems_to_run = dataset[:num_problems]
    if len(problems_to_run) < num_problems:
        logger.warning(
            f"Requested {num_problems} problems, but dataset only contains {len(problems_to_run)}. Running on available problems."
        )
        num_problems = len(problems_to_run)

    # Step 2: Configure coevolution (parameters used for *all* problems)
    logger.info("Configuring coevolution parameters...")
    llm_model = "gpt-5-mini"
    config = CoevolutionConfig(
        # Population sizes
        initial_code_population_size=10,
        initial_test_population_size=20,
        max_code_population_size=15,  # population grows over time to 15
        # Bayesian parameters
        initial_code_prior=0.5,
        initial_test_prior=0.5,
        alpha=0.01,
        beta=0.2,
        gamma=0.2,
        learning_rate=0.1,
        use_intermediate_updates=False,
        # Evolution parameters
        num_generations=3,
        selection_strategy="roulette_wheel",
        # Code genetic operators
        code_crossover_rate=0.2,
        code_mutation_rate=0.1,
        code_edit_rate=0.8,
        code_elite_proportion=0.5,
        code_offspring_proportion=0.5,
        # Test genetic operators
        test_crossover_rate=0.5,
        test_mutation_rate=0.3,
        test_edit_rate=0.5,
        # LLM configuration
        llm_model=llm_model,
    )

    logger.info(f"Using common configuration for all {num_problems} problems.")
    logger.trace(f"Full configuration: {config.__dict__}")  # Log config details

    # Step 3: Loop through problems and run orchestrator
    code_generations = {}
    total_start_time = time.time()

    sandbox = create_safe_test_environment()
    llm_client = create_llm_client(provider="openai", model=llm_model)

    for i, problem in enumerate(problems_to_run):
        logger.info(f"*** Processing Problem {i + 1}/{num_problems} ***")
        best_code_individual = run_single_problem(problem, config, llm_client, sandbox)
        code_generations[problem.question_id] = best_code_individual

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logger.info("=" * 80)
    logger.info("BATCH EXPERIMENT COMPLETE")
    logger.info(f"Total time: {total_duration:.2f} seconds")
    logger.info("=" * 80)

    logger.info("Code Generations:")
    for problem_id, code in code_generations.items():
        logger.info(f" - Problem ID: {problem_id}, Best Code:\n {code}")

    logger.info(f"Total execution time: {total_duration:.2f} seconds")
    logger.info("Experiment script finished.")


if __name__ == "__main__":
    main_batch(num_problems=20)
