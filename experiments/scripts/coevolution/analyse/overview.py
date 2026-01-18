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
            # Add N/A entry to summary
            summary_data.append(
                {
                    "run_id": run_id,
                    "problem_id": pid,
                    "solved": False,
                    "champion_code_ids": "N/A",
                    "champion_passing": "N/A",
                    "champion_probability": 0.0,
                    "initial_pass_at_10": "N/A",
                    "final_pass_at_15": "N/A",
                    "final_pass_at_10": "N/A",
                }
            )
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
        champion_ids = []
        champion_passing_info = []
        champion_probability = 0.0

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
                    champion_probability = max_prob
                    champions = final_codes[final_codes["probability"] == max_prob]

                    # Collect champion IDs and their test pass counts separately
                    any_champion_solved = False

                    for idx, champ_row in champions.iterrows():
                        champ_id = champ_row["id"]
                        champ_pass_count = 0
                        if champ_id in pass_counts.index:
                            champ_pass_count = int(pass_counts.loc[champ_id])
                            if champ_pass_count == num_tests:
                                any_champion_solved = True
                        champion_ids.append(champ_id)
                        champion_passing_info.append(f"{champ_pass_count}/{num_tests}")

                    solved = any_champion_solved

        # Count initial and final codes passing all tests
        num_initial_passing = len(
            first_matrix.sum(axis=1)[first_matrix.sum(axis=1) == first_matrix.shape[1]]
        )

        num_final_passing = len(
            last_matrix.sum(axis=1)[last_matrix.sum(axis=1) == last_matrix.shape[1]]
        )

        # Calculate final pass@10 from top 10 survived codes by probability
        num_final_pass_at_10 = 0
        if not individuals_df.empty:
            final_codes = individuals_df[
                (individuals_df["status"].str.lower() == "survived")
                & (individuals_df["type"] == "code")
            ]
            if not final_codes.empty:
                # Sort by probability descending and take top 10
                top_10_codes = final_codes.nlargest(
                    min(10, len(final_codes)), "probability"
                )
                # Count how many of top 10 pass all tests
                for _, code_row in top_10_codes.iterrows():
                    code_id = code_row["id"]
                    if code_id in pass_counts.index:
                        if int(pass_counts.loc[code_id]) == num_tests:
                            num_final_pass_at_10 += 1

        # Format champion code IDs and passing info
        champion_ids_str = ", ".join(champion_ids) if champion_ids else "N/A"
        champion_passing_str = (
            ", ".join(champion_passing_info) if champion_passing_info else "N/A"
        )

        summary_data.append(
            {
                "run_id": run_id,
                "problem_id": pid,
                "solved": solved,
                "champion_code_ids": champion_ids_str,
                "champion_passing": champion_passing_str,
                "champion_probability": round(champion_probability, 4),
                "initial_pass_at_10": num_initial_passing,
                "final_pass_at_15": num_final_passing,
                "final_pass_at_10": num_final_pass_at_10,
            }
        )

    # 5. Print summary table
    print("\n" + "=" * 140)
    print("SUMMARY TABLE")
    print("=" * 140)
    print(
        f"{'Run ID':<15} {'Problem ID':<20} {'Solved':<8} {'Champion ID':<15} {'Champion Pass':<14} {'Champion Prob':<14} {'Init P@10':<11} {'Final P@15':<12} {'Final P@10':<12}"
    )
    print("-" * 140)
    for row in summary_data:
        solved_str = "Yes" if row["solved"] else "No"

        # Handle N/A values for metrics
        init_p10 = (
            row["initial_pass_at_10"] if row["initial_pass_at_10"] != "N/A" else "N/A"
        )
        final_p15 = (
            row["final_pass_at_15"] if row["final_pass_at_15"] != "N/A" else "N/A"
        )
        final_p10 = (
            row["final_pass_at_10"] if row["final_pass_at_10"] != "N/A" else "N/A"
        )

        print(
            f"{row['run_id']:<15} {row['problem_id']:<20} {solved_str:<8} {row['champion_code_ids']:<15} {row['champion_passing']:<14} {row['champion_probability']:<14.4f} {str(init_p10):<11} {str(final_p15):<12} {str(final_p10):<12}"
        )

    print("\n" + "=" * 140)
    print("End of Report")


if __name__ == "__main__":
    app()
