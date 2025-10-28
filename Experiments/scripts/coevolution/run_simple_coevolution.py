"""
Simple script to run the coevolution orchestrator on a sample problem.

This script demonstrates how to use the CoevolutionOrchestrator to evolve
both code solutions and test cases for a programming problem using Bayesian
belief updating and LLM-based genetic operators.
"""

import sys

from lcb_runner.benchmarks import code_generation  # type: ignore
from loguru import logger

from common.coevolution.config import CoevolutionConfig
from common.coevolution.orchestrator import CoevolutionOrchestrator
from common.llm_client import create_llm_client
from common.sandbox import create_safe_test_environment


def main() -> None:
    """Run a simple coevolution experiment."""

    # Configure logging
    logger.remove()  # Remove the default 'DEBUG' level handler
    logger.add(
        sys.stderr,
        level="DEBUG",
    )
    logger.add(
        "logs/experiment_{time:YYYY-MM-DD}.log",
        level="TRACE",  # This captures ALL levels
        rotation="100 MB",  # New file every 100 MB
        retention="10 days",  # Keep logs for 10 days
        compression="zip",  # Compress old logs
        enqueue=True,  # Use a queue for thread safety
    )

    logger.info("=" * 80)
    logger.info("Starting Simple Coevolution Experiment")
    logger.info("=" * 80)

    # Step 1: Load a problem from LiveCodeBench
    logger.info("Loading problem from LiveCodeBench dataset...")
    dataset = code_generation.load_code_generation_dataset(
        release_version="release_v6",
        start_date="2024-01-01",
        difficulty=code_generation.Difficulty.HARD,
    )
    problem = dataset[0]

    logger.info(f"Loaded problem: {problem.question_title}")
    logger.info(f"Problem ID: {problem.question_id}")
    logger.info(f"Difficulty: {problem.difficulty}")

    # Step 2: Create LLM client
    logger.info("Creating LLM client...")
    llm_model = "gpt-5-mini"  # Specify the LLM model to use
    llm_client = create_llm_client(provider="openai", model=llm_model)
    logger.info(f"Using model: {llm_client.model}")

    # Step 3: Create sandbox environment
    logger.info("Creating sandbox environment...")
    sandbox = create_safe_test_environment()

    # Step 4: Configure coevolution
    logger.info("Configuring coevolution parameters...")
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
        num_generations=3,  # Small number for quick testing
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

    logger.info(
        f"Configuration: {config.num_generations} generations, "
        f"{config.initial_code_population_size} code solutions, "
        f"{config.initial_test_population_size} test cases"
    )

    logger.trace(f"Full configuration: {config}")

    # Step 5: Create and run orchestrator
    logger.info("Initializing orchestrator...")
    orchestrator = CoevolutionOrchestrator(
        config=config, problem=problem, llm_client=llm_client, sandbox=sandbox
    )

    logger.info("Starting coevolution...")
    code_population, test_population = orchestrator.run()

    best_code_individual, best_code_prob = code_population.get_best_individual()
    best_test_individual, best_test_prob = test_population.get_best_individual()

    # Step 6: Display results
    logger.info("=" * 80)
    logger.info("COEVOLUTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best code probability: {best_code_prob:.4f}")
    logger.info(f"Best test probability: {best_test_prob:.4f}")

    print("\n" + "=" * 80)
    print("BEST CODE SOLUTION:")
    print("=" * 80)
    print(best_code_individual)

    print("\n" + "=" * 80)
    print("BEST TEST CASE:")
    print("=" * 80)
    print(best_test_individual)

    logger.trace("Final code population:")
    for i, (individual, prob) in enumerate(code_population):
        logger.trace(f"Individual {i}:\n{individual}\nProbability: {prob:.4f}")

    logger.trace("Final test population:")
    for i, (individual, prob) in enumerate(test_population):
        logger.trace(f"Individual {i}:\n{individual}\nProbability: {prob:.4f}")

    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
