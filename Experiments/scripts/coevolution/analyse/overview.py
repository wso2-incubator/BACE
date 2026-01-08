# experiments/scripts/coevolution/analyse/overview.py
import sys

import pandas as pd
import typer
from loguru import logger

from coevolution.analysis.log_parser import get_problem_ids, parse_coevolution_log

app = typer.Typer()


def analyze_matrix(df: pd.DataFrame, label: str) -> None:
    """
    Calculates and prints metrics for a specific observation matrix.

    Metrics:
    1. Count of perfect solutions (rows with all 1s)
    2. Best individual code ID and its pass count
    """
    if df.empty:
        print(f"  - {label}: [Empty Matrix]")
        return

    num_codes, num_tests = df.shape

    # 1. Calculate 'Perfect' Solutions (All Tests Passed)
    # Sum across columns (axis=1) to get pass count per code
    pass_counts = df.sum(axis=1)

    # Check if pass_count equals total number of tests
    perfect_solutions = pass_counts[pass_counts == num_tests]
    perfect_count = len(perfect_solutions)

    # 2. Identify Best Code
    if not pass_counts.empty:
        best_code_id = pass_counts.idxmax()
        best_score = pass_counts.max()
    else:
        best_code_id = "N/A"
        best_score = 0

    print(f"  - {label} (Pop: {num_codes}, Tests: {num_tests}):")
    print(f"    - Perfect Solutions (Pass All): {perfect_count}")
    print(f"    - Best Individual: '{best_code_id}' (Passed {best_score}/{num_tests})")


@app.command()
def main(
    run_id: str = typer.Option(..., help="The Run ID to analyze"),
    log_dir: str = typer.Option("logs", help="Directory containing log files"),
    matrix_type: str = typer.Option(
        "private", help="The matrix type to analyze (e.g., 'public', 'unittest')"
    ),
    file_pattern: str = typer.Option(
        "*.log", help="Pattern to match log files (e.g., '*.log' or '*.zip')"
    ),
) -> None:
    """
    Analyze the First and Last observation matrices for a specific Run ID.
    Calculates solution counts and best performers.
    """
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

    logger.info(f"Scanning {log_dir}/{file_pattern} for Run ID: {run_id}...")

    # 1. Find all problems associated with this Run ID
    problem_ids = get_problem_ids(log_dir, file_pattern, run_id)

    if not problem_ids:
        logger.error(f"No problems found for Run ID: {run_id}")
        return

    logger.info(f"Found {len(problem_ids)} problems. Starting analysis...\n")
    print("=" * 80)
    print(f"ANALYSIS REPORT: {run_id}")
    print("=" * 80)

    # 2. Iterate through each problem
    for pid in sorted(list(problem_ids)):
        print(f"\nPROBLEM: {pid}")
        print("-" * 40)

        # Parse the specific logs for this problem
        parsed_data = parse_coevolution_log(
            log_dir=log_dir,
            log_filename_pattern=file_pattern,
            target_run_id=run_id,
            target_problem_id=pid,
        )

        # Extract the list of matrices for the requested type
        matrices = parsed_data["matrices"].get(matrix_type, [])

        if not matrices:
            logger.warning(f"No '{matrix_type}' matrices found for {pid}")
            continue

        # 3. Analyze First and Last Matrix
        first_matrix = matrices[0]
        last_matrix = matrices[-1]

        analyze_matrix(first_matrix, "START (First Matrix)")

        # Only print 'End' if it's different from 'Start'
        if len(matrices) > 1:
            analyze_matrix(last_matrix, "END (Last Matrix)")
        else:
            print("Run had not progressed; only one matrix present.")

    print("\n" + "=" * 80)
    print("End of Report")


if __name__ == "__main__":
    app()
