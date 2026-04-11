#!/usr/bin/env python3
"""
Script to extract all run IDs and their associated problem IDs from log files.
"""

import typer

from coevolution.analysis.log_parser import get_problem_ids, get_run_ids


def main(
    log_dir: str = typer.Option("logs/", "-d", help="Directory containing log files"),
    log_filename_pattern: str = typer.Option(
        "*.log", "-f", help="Pattern to match log files (e.g., *.log, worker_*.jsonl)"
    ),
) -> None:
    """
    Extract all run IDs and their associated problem IDs from log files.
    """
    # Get all run IDs
    print("Scanning logs for run IDs...")
    run_ids = get_run_ids(log_dir, log_filename_pattern)

    if not run_ids:
        print(
            f"No run IDs found in {log_dir} matching pattern '{log_filename_pattern}'"
        )
        return

    print(f"\nFound {len(run_ids)} run ID(s):\n")

    # For each run ID, get the associated problem IDs
    results = {}
    for run_id in sorted(run_ids):
        problem_ids = get_problem_ids(log_dir, log_filename_pattern, run_id)
        results[run_id] = problem_ids

        print(f"Run ID: {run_id}")
        if problem_ids:
            print(
                f"  Problem IDs ({len(problem_ids)}): {', '.join(sorted(problem_ids))}"
            )
        else:
            print("  No problem IDs found")
        print()

    # print results
    for run_id, problem_ids in results.items():
        print(f"{run_id}:\t", end=" ")
        for problem_id in problem_ids:
            print(f"{problem_id},", end=" ")
        print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total run IDs: {len(results)}")
    print(
        f"Total unique problem IDs across all runs: {len(set().union(*results.values()))}"
    )


if __name__ == "__main__":
    typer.run(main)
