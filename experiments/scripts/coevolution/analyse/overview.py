# experiments/scripts/coevolution/analyse/overview.py
import sys

import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from coevolution.analysis.log_parser import get_problem_ids, parse_coevolution_log

app = typer.Typer()
console = Console()


def analyze_matrix(df: pd.DataFrame, label: str) -> None:
    """
    Calculates and prints metrics for a specific observation matrix.
    """
    if df.empty:
        rprint(f"[yellow]  - {label}: [Empty Matrix][/yellow]")
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

    rprint(Text.assemble(
        (f"  - {label} ", "bold cyan"),
        (f"(Pop: {num_codes}, Tests: {num_tests})", "dim"),
        (f": Perfect Solutions: {perfect_count}, Best: "),
        (f"'{best_code_id}'", "bold yellow"),
        (f" ({best_score}/{num_tests})", "green" if best_score == num_tests else "dim")
    ))


@app.command()
def main(
    run_id: str = typer.Option(..., help="The Run ID to analyze"),
    log_dir: str = typer.Option("logs", help="Directory containing log files"),
    file_pattern: str = typer.Option(
        "*.log", help="Pattern to match legacy log files"
    ),
    problem_id: str = typer.Option(None, help="Specific Problem ID to analyze (optional)"),
    legacy: bool = typer.Option(
        False, "--legacy", help="Whether to scan legacy flat log files"
    ),
) -> None:
    """
    Analyze the Private observation matrices for a specific Run ID.
    Calculates solution counts and best performers based on ground-truth.
    """
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

    console.rule(f"[bold magenta]SCANNING: {run_id}[/bold magenta]")

    # 1. Find all problems associated with this Run ID
    legacy_files = [] # Initialize to ensure it's always defined
    if problem_id:
        # For legacy, we still want to find which files contain the run_id to avoid scanning everything
        _, legacy_files = get_problem_ids(log_dir, file_pattern, run_id, use_legacy=legacy)
        problem_ids = {problem_id}
    else:
        problem_ids, legacy_files = get_problem_ids(log_dir, file_pattern, run_id, use_legacy=legacy)

    if not problem_ids:
        logger.error(f"No problems found for Run ID: {run_id}")
        return

    logger.info(f"Found {len(problem_ids)} problems. Starting analysis...\n")

    # Track summary data for final table
    summary_data = []

    # 2. Iterate through each problem
    for pid in sorted(list(problem_ids)):
        if pid == "SETUP":
            continue
        
        rprint(f"\n[bold green]PROBLEM:[/bold green] [bold white]{pid}[/bold white]")
        rprint("[dim]" + "─" * 40 + "[/dim]")

        # Parse the specific logs for this problem
        parsed_data = parse_coevolution_log(
            log_dir=log_dir,
            log_filename_pattern=file_pattern,
            target_run_id=run_id,
            target_problem_id=pid,
            use_legacy=legacy,
            legacy_files=legacy_files
        )

        # We strictly use 'private' for evaluation metrics
        matrix_type = "private"
        matrices = parsed_data["matrices"].get(matrix_type, [])

        if not matrices:
            logger.warning(f"No '{matrix_type}' matrices found for {pid}")
            # Add N/A entry to summary

            summary_data.append({
                "run_id": run_id, "problem_id": pid, "solved": False,
                "champion_code_ids": "N/A", "champion_passing": "N/A",
                "champion_probability": 0.0, "initial_pass_at_10": "N/A",
                "final_pass_at_15": "N/A", "final_pass_at_10": "N/A",
            })
            continue

        # 3. Analyze First and Last Matrix
        first_matrix = matrices[0]
        last_matrix = matrices[-1]

        analyze_matrix(first_matrix, "START")

        # Only print 'End' if it's different from 'Start'
        if len(matrices) > 1:
            analyze_matrix(last_matrix, "END  ")
        else:
            rprint("[dim]Run had not progressed; only one matrix present.[/dim]")

        # 4. Collect summary data
        solved = False
        champion_ids = []
        champion_passing_info = []
        champion_probability = 0.0
        num_final_pass_at_10 = 0
        num_final_pass_at_15 = 0

        if not last_matrix.empty:
            num_tests = last_matrix.shape[1]
            pass_counts = last_matrix.sum(axis=1)

            # Map of ID -> latest probability
            belief_map = {}
            individuals_df = parsed_data.get("individuals", pd.DataFrame())

            # Get all IDs present in the matrix
            active_ids = last_matrix.index.tolist()

            if not individuals_df.empty:
                # Filter for code entries
                code_data = individuals_df[individuals_df["type"] == "code"].copy()
                if not code_data.empty:
                    # Group by ID and get the latest non-NaN probability
                    # This is crucial because some events (like 'survived') don't carry probabilities
                    latest_beliefs = (
                        code_data.dropna(subset=["probability"])
                        .groupby("id")["probability"]
                        .last()
                    )

                    # Map IDs to their latest known beliefs
                    belief_map = {
                        str(iid): latest_beliefs.get(iid, 0.0) for iid in active_ids
                    }
                else:
                    belief_map = {str(iid): 0.0 for iid in active_ids}
            else:
                # If no belief data, treat all as 0.0
                belief_map = {str(iid): 0.0 for iid in active_ids}

            # Sort active IDs by belief
            sorted_by_belief = sorted(
                active_ids, key=lambda x: belief_map.get(str(x), 0.0), reverse=True
            )

            if sorted_by_belief:
                # 1. Champion logic (Top individual(s) by belief)
                max_belief = belief_map[str(sorted_by_belief[0])]
                champion_probability = max_belief
                champions = [
                    iid
                    for iid in sorted_by_belief
                    if belief_map[str(iid)] == max_belief
                ]

                any_champ_solved = False
                for cid in champions:
                    c_pass = (
                        int(pass_counts.loc[cid]) if cid in pass_counts.index else 0
                    )
                    champion_ids.append(str(cid))
                    champion_passing_info.append(f"{c_pass}/{num_tests}")
                    if c_pass == num_tests:
                        any_champ_solved = True

                solved = any_champ_solved

                # 2. Pass@N counts (How many of Top K pass ALL tests?)
                def count_perfect_in_top(k: int) -> int:
                    perfect = 0
                    for cid in sorted_by_belief[:k]:
                        if (
                            cid in pass_counts.index
                            and int(pass_counts.loc[cid]) == num_tests
                        ):
                            perfect += 1
                    return perfect

                num_final_pass_at_10 = count_perfect_in_top(10)
                num_final_pass_at_15 = count_perfect_in_top(15)

        # Count initial perfect solutions
        num_initial_passing = len(matrices[0].sum(axis=1)[matrices[0].sum(axis=1) == matrices[0].shape[1]])

        # Format champion code IDs and passing info
        champion_ids_str = ", ".join(champion_ids) if champion_ids else "N/A"
        champion_passing_str = (
            ", ".join(champion_passing_info) if champion_passing_info else "N/A"
        )

        summary_data.append({
            "run_id": run_id, "problem_id": pid, "solved": solved,
            "champion_code_ids": champion_ids_str,
            "champion_passing": champion_passing_str,
            "champion_probability": champion_probability,
            "initial_pass_at_10": num_initial_passing,
            "final_pass_at_15": num_final_pass_at_15,
            "final_pass_at_10": num_final_pass_at_10,
        })

    # 5. Print summary table
    print("\n")
    table = Table(title=f"RUN SUMMARY: {run_id}", title_style="bold underline magenta", show_header=True, header_style="bold cyan")
    table.add_column("Problem", style="white")
    table.add_column("Solved", justify="center")
    table.add_column("Champion ID", style="yellow")
    table.add_column("Score", justify="center")
    table.add_column("Prob", justify="right")
    table.add_column("Init P@10", justify="center", style="dim")
    table.add_column("Final P@10", justify="center", style="bold green")
    table.add_column("Final P@15", justify="center", style="dim")

    for row in summary_data:
        solved_str = "[bold green]Yes[/bold green]" if row["solved"] else "[bold red]No[/bold red]"
        table.add_row(
            row["problem_id"],
            solved_str,
            row["champion_code_ids"],
            row["champion_passing"],
            f"{row['champion_probability']:.4f}",
            str(row["initial_pass_at_10"]),
            str(row["final_pass_at_10"]),
            str(row["final_pass_at_15"])
        )
    
    console.print(table)

    # Overall summary statistics: total problems, passed, and pass rate
    total_problems = len(summary_data)
    total_passed = sum(1 for r in summary_data if r.get("solved"))
    pass_rate = (total_passed / total_problems * 100) if total_problems else 0.0

    stats_text = Text.assemble(
        ("Total Problems: ", "bold"), (f"{total_problems}\n", "white"),
        ("Passed:         ", "bold"), (f"{total_passed}\n", "green" if total_passed > 0 else "red"),
        ("Pass Rate:      ", "bold"), (f"{pass_rate:.2f}%", "bold yellow")
    )
    console.print("\n", Panel(stats_text, title="Overall Statistics", border_style="magenta", expand=False))
    console.rule("[bold magenta]End of Report[/bold magenta]")


if __name__ == "__main__":
    app()
