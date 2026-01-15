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

    # Track summary data for final table
    summary_data = []

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

        print(first_matrix)
        analyze_matrix(first_matrix, "START (First Matrix)")

        # Only print 'End' if it's different from 'Start'
        if len(matrices) > 1:
            print(last_matrix)
            analyze_matrix(last_matrix, "END (Last Matrix)")
        else:
            print("Run had not progressed; only one matrix present.")

        # 4. Collect summary data
        # Check if best individual (highest probability) in final population passes all tests
        solved = False
        champion_infos = []
        
        if not last_matrix.empty:
            num_tests = last_matrix.shape[1]
            pass_counts = last_matrix.sum(axis=1)

            # Get individuals dataframe and find the one with highest probability in final generation
            individuals_df = parsed_data.get("individuals", pd.DataFrame())
            if not individuals_df.empty:
                # Filter for code individuals in the last generation
                final_codes = individuals_df[
                    (individuals_df["status"].str.lower() == "survived")
                    & (individuals_df["type"] == "code")
                ]

                if not final_codes.empty:
                    # Find all codes with highest probability
                    max_prob = final_codes["probability"].max()
                    champions = final_codes[final_codes["probability"] == max_prob]

                    # Collect champion IDs with their test pass counts
                    any_champion_solved = False

                    for idx, champ_row in champions.iterrows():
                        champ_id = champ_row["id"]
                        champ_pass_count = 0
                        if champ_id in pass_counts.index:
                            champ_pass_count = int(pass_counts.loc[champ_id])
                            if champ_pass_count == num_tests:
                                any_champion_solved = True
                        champion_infos.append(
                            f"{champ_id} ({champ_pass_count}/{num_tests})"
                        )

                    solved = any_champion_solved

        # Count initial and final codes passing all tests
        num_initial_passing = len(
            first_matrix.sum(axis=1)[first_matrix.sum(axis=1) == first_matrix.shape[1]]
        )
        total_initial = len(first_matrix)

        num_final_passing = len(
            last_matrix.sum(axis=1)[last_matrix.sum(axis=1) == last_matrix.shape[1]]
        )
        total_final = len(last_matrix)

        # Format champion code ID with test pass info
        champion_info = ", ".join(champion_infos) if champion_infos else "N/A"

        summary_data.append(
            {
                "problem_id": pid,
                "solved": solved,
                "champion_code_id": champion_info,
                "initial_passing": f"{num_initial_passing}/{total_initial}",
                "final_passing": f"{num_final_passing}/{total_final}",
            }
        )

    # 5. Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Problem ID':<30} {'Solved':<10} {'Champion Code ID':<20} {'Initial Passing':<20} {'Final Passing':<20}"
    )
    print("-" * 80)
    for row in summary_data:
        solved_str = "Yes" if row["solved"] else "No"
        print(
            f"{row['problem_id']:<30} {solved_str:<10} {row['champion_code_id']:<20} {row['initial_passing']:<20} {row['final_passing']:<20}"
        )

    print("\n" + "=" * 80)
    print("End of Report")


if __name__ == "__main__":
    app()
