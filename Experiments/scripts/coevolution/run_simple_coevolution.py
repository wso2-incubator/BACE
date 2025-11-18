"""
Modern script to run the coevolution orchestrator on LiveCodeBench problems.

This script demonstrates how to use the new modular architecture with:
- OrchestratorBuilder for clean dependency injection
- All new core implementations (BayesianSystem, ExecutionSystem, etc.)
- LLM-based genetic operators
- LCB dataset integration
"""

from datetime import datetime
from typing import Optional

import typer
from loguru import logger

import common.coevolution.logging_utils as logging_utils
from common.coevolution.bayesian_system import BayesianSystem
from common.coevolution.execution import ExecutionSystem
from common.coevolution.feedback import CodeFeedbackGenerator, TestFeedbackGenerator
from common.coevolution.lcb_dataset import (
    Difficulty,
    LCBCodeGenerationProblem,
    LCBDatasetTestBlockBuilder,
    LCBTestBlockRebuilder,
    load_code_generation_dataset,
)
from common.coevolution.llm_operators import CodeLLMOperator, TestLLMOperator
from common.coevolution.orchestrator_builder import OrchestratorBuilder
from common.coevolution.pareto_system import ParetoSystem
from common.coevolution.probability_assigner import ProbabilityAssigner
from common.coevolution.selection_strategy import SelectionStrategy
from common.llm_client import create_llm_client
from common.sandbox import create_safe_test_environment


def load_problem() -> LCBCodeGenerationProblem:
    """Load a problem from LiveCodeBench dataset."""
    logger.info("Loading problem from LiveCodeBench dataset...")

    problems = load_code_generation_dataset(
        release_version="release_v5",
        start_date="2024-01-01",
        end_date="2024-12-31",
        difficulty=Difficulty.HARD,
    )

    if not problems:
        raise ValueError("No problems found matching criteria")

    problem = problems[0]

    logger.info(f"Loaded problem: {problem.question_title}")
    logger.info(f"Problem ID: {problem.question_id}")
    logger.info(f"Platform: {problem.platform.value}")
    logger.info(f"Difficulty: {problem.difficulty.value}")
    logger.info(f"Public tests: {len(problem.public_test_cases)}")
    logger.info(f"Private tests: {len(problem.private_test_cases)}")

    return problem


def main(
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Unique identifier for this run. If not provided, one will be auto-generated with timestamp.",
    ),
) -> None:
    """Run a coevolution experiment on a LiveCodeBench problem."""
    logging_utils.setup_logging(console_level="DEBUG", file_level="TRACE")

    logging_utils.log_section_header("INFO", "STARTING COEVOLUTION EXPERIMENT")

    # ====================================
    # Step 1: Load Problem
    # ====================================
    problem = load_problem()

    # Generate run_id if not provided
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {run_id}")

    with logger.contextualize(problem_id=problem.question_id, run_id=run_id):
        # ====================================
        # Step 2: Create Infrastructure
        # ====================================
        logger.info("Creating LLM client and sandbox...")

        llm_model = "gpt-5-mini"
        llm_client = create_llm_client(provider="openai", model=llm_model)
        logger.info(f"Using model: {llm_client.model}")

        sandbox = create_safe_test_environment()
        logger.info("Sandbox created")

        # ====================================
        # Step 3: Instantiate Components
        # ====================================
        logger.info("Instantiating coevolution components...")

        # LLM Operators
        code_operator = CodeLLMOperator(llm=llm_client, problem=problem)
        test_operator = TestLLMOperator(llm=llm_client, problem=problem)

        # Selection Strategies
        code_selector = SelectionStrategy("roulette_wheel")
        test_selector = SelectionStrategy("roulette_wheel")

        # Probability Assigners
        code_prob_assigner = ProbabilityAssigner("min")
        test_prob_assigner = ProbabilityAssigner("min")

        # Execution System
        execution_system = ExecutionSystem(enable_multiprocessing=True, num_workers=12)

        # Bayesian Systems (same implementation for code and test)
        code_bayesian_system = BayesianSystem()
        test_bayesian_system = BayesianSystem()

        # Pareto System
        pareto = ParetoSystem()

        # Test Block Rebuilder
        test_block_rebuilder = LCBTestBlockRebuilder()

        # Feedback Generators
        code_feedback_gen = CodeFeedbackGenerator()
        test_feedback_gen = TestFeedbackGenerator()

        # Dataset Test Block Builder
        dataset_test_block_builder = LCBDatasetTestBlockBuilder()

        logger.info("All components instantiated successfully")

        # ====================================
        # Step 4: Build Orchestrator
        # ====================================
        logger.info("Building orchestrator with configuration...")

        orchestrator = (
            OrchestratorBuilder()
            # Evolution configuration
            .with_evolution_config(
                num_generations=5,  # Small number for quick testing
                random_seed=42,
                max_workers=12,
            )
            # Code population configuration
            .with_code_population_config(
                initial_prior=0.2,
                initial_population_size=10,
                max_population_size=15,
                elitism_rate=0.5,
                offspring_rate=0.5,
            )
            # Test population configuration
            .with_test_population_config(
                initial_prior=0.2,
                initial_population_size=20,
            )
            # Code operator rates
            .with_code_operator_rates(
                crossover_rate=0.2,
                mutation_rate=0.1,
                edit_rate=0.8,
            )
            # Test operator rates
            .with_test_operator_rates(
                crossover_rate=0.2,
                mutation_rate=0.8,
                edit_rate=0.2,
            )
            # Bayesian configuration
            .with_bayesian_config(
                alpha=0.01,  # P(pass | code correct, test incorrect)
                beta=0.2,  # P(pass | code incorrect, test correct)
                gamma=0.2,  # P(pass | code incorrect, test incorrect)
                learning_rate=0.05,
            )
            # Problem and sandbox
            .with_problem(problem)
            .with_sandbox(sandbox)
            # Operators
            .with_code_operator(code_operator)
            .with_test_operator(test_operator)
            # Selection strategies
            .with_code_selector(code_selector)
            .with_test_selector(test_selector)
            # Probability assigners
            .with_code_prob_assigner(code_prob_assigner)
            .with_test_prob_assigner(test_prob_assigner)
            # Systems
            .with_execution_system(execution_system)
            .with_code_bayesian_system(code_bayesian_system)
            .with_test_bayesian_system(test_bayesian_system)
            .with_pareto(pareto)
            # Feedback and rebuilders
            .with_test_block_rebuilder(test_block_rebuilder)
            .with_code_feedback_gen(code_feedback_gen)
            .with_test_feedback_gen(test_feedback_gen)
            .with_dataset_test_block_builder(dataset_test_block_builder)
            .build()
        )

        logger.info("Orchestrator built successfully")

        # ====================================
        # Step 5: Run Coevolution
        # ====================================
        logger.info("Starting coevolution run...")
        code_population, test_population = orchestrator.run()

        # ====================================
        # Step 6: Display Results
        # ====================================
        logger.info("=" * 80)
        logger.info("COEVOLUTION RESULTS")
        logger.info("=" * 80)

        best_code = code_population.get_best_individual()
        best_test = test_population.get_best_individual()

        if best_code:
            logger.info(f"Best code probability: {best_code.probability:.4f}")
            logger.info(f"Best code ID: {best_code.id}")

        if best_test:
            logger.info(f"Best test probability: {best_test.probability:.4f}")
            logger.info(f"Best test ID: {best_test.id}")

        logger.info(f"Final code population size: {code_population.size}")
        logger.info(f"Final test population size: {test_population.size}")
        logger.info(
            f"Code population avg probability: {code_population.compute_average_probability():.4f}"
        )
        logger.info(
            f"Test population avg probability: {test_population.compute_average_probability():.4f}"
        )

        # Print best solutions
        print("\n" + "=" * 80)
        print("BEST CODE SOLUTION:")
        print("=" * 80)
        if best_code:
            print(best_code.snippet)
        else:
            print("No code solution found")

        print("\n" + "=" * 80)
        print("BEST TEST CASE:")
        print("=" * 80)
        if best_test:
            print(best_test.snippet)
        else:
            print("No test case found")

        print("\n" + "=" * 80)
        print("FULL TEST CLASS BLOCK:")
        print("=" * 80)
        print(test_population.test_class_block)

        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)


if __name__ == "__main__":
    typer.run(main)
