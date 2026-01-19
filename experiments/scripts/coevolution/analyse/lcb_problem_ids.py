#!/usr/bin/env python3
"""
Simple script to display LiveCodeBench problem IDs based on dataset filters.

Usage:
    # Default (release_v6, hard difficulty, all dates)
    python lcb_problem_ids.py

    # Specific date range and difficulty
    python lcb_problem_ids.py --start-date 2025-03-01 --end-date 2025-05-10 --difficulty hard

    # Different dataset version
    python lcb_problem_ids.py --dataset-version release_v5 --difficulty medium

    # Export to file
    python lcb_problem_ids.py --start-date 2025-03-01 --end-date 2025-05-10 > problem_ids.txt
"""

from typing import Optional

import typer
from loguru import logger

from coevolution.adapters.lcb import Difficulty, load_code_generation_dataset

app = typer.Typer()


@app.command()
def main(
    dataset_version: str = typer.Option(
        "release_v6",
        "--dataset-version",
        "-v",
        help="LiveCodeBench dataset version to use",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Filter problems from this date onwards (YYYY-MM-DD)",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="Filter problems up to this date (YYYY-MM-DD)",
    ),
    difficulty: Difficulty = typer.Option(
        Difficulty.HARD,
        "--difficulty",
        "-d",
        help="Filter problems by difficulty level (easy, medium, hard)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Show detailed problem information",
    ),
) -> None:
    """
    Display LiveCodeBench problem IDs matching the specified filters.

    Examples:
        # Get all hard problems from Q2 2025
        python lcb_problem_ids.py --start-date 2025-04-01 --end-date 2025-06-30 --difficulty hard

        # Count problems
        python lcb_problem_ids.py --start-date 2025-03-01 --end-date 2025-05-10 | wc -l

        # Save to file
        python lcb_problem_ids.py --difficulty medium > medium_problems.txt
    """
    # Load problems from dataset
    logger.info(f"Loading problems from {dataset_version}...")
    logger.info(f"  Difficulty: {difficulty.value}")
    logger.info(f"  Start Date: {start_date or 'Any'}")
    logger.info(f"  End Date: {end_date or 'Any'}")

    problems = load_code_generation_dataset(
        release_version=dataset_version,
        start_date=start_date,
        end_date=end_date,
        difficulty=difficulty,
    )

    if not problems:
        logger.error("No problems found matching the criteria")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(problems)} problems\n")

    # Display results
    if verbose:
        print("=" * 80)
        print("PROBLEM DETAILS")
        print("=" * 80)
        for i, problem in enumerate(problems, 1):
            print(f"\n{i}. {problem.question_id}")
            print(f"   Title: {problem.question_title}")
            print(f"   Difficulty: {problem.difficulty}")
            print(f"   Contest: {problem.contest_id}")
            print(f"   Date: {problem.contest_date}")
    else:
        # Simple list output (one ID per line, suitable for piping)
        for problem in problems:
            print(problem.question_id)


if __name__ == "__main__":
    app()
