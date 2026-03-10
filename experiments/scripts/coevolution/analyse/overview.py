# experiments/scripts/coevolution/analyse/overview.py
import sys
import fnmatch

import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from coevolution.analysis.log_parser import get_problem_ids, parse_coevolution_log, get_run_ids

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
    run_ids: list[str] = typer.Option(..., "--run-id", help="Run IDs to analyze (can be multiple)"),
    log_dir: str = typer.Option("logs", help="Directory containing log files"),
    file_pattern: str = typer.Option(
        "*.log", help="Pattern to match legacy log files"
    ),
    problem_id: str = typer.Option(None, help="Specific Problem ID to analyze (optional)"),
) -> None:
    """
    Analyze Private observation matrices for multiple Run IDs.
    Merges results, ensuring no problem is completed in multiple runs.
    """
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

    # 0. Expand Globs for run_ids
    all_available_runs = get_run_ids(log_dir, file_pattern)
    expanded_run_ids = set()
    for pattern in run_ids:
        if any(c in pattern for c in "*?[]"):
            matches = [r for r in all_available_runs if fnmatch.fnmatch(r, pattern)]
            if not matches:
                logger.warning(f"No runs matched pattern: {pattern}")
            expanded_run_ids.update(matches)
        else:
            expanded_run_ids.add(pattern)
    
    final_run_ids = sorted(list(expanded_run_ids))
    if not final_run_ids:
        logger.error(f"No valid Run IDs found after expansion of: {', '.join(run_ids)}")
        return

    console.rule(f"[bold magenta]ANALYZING RUNS: {', '.join(final_run_ids)}[/bold magenta]")

    # 1. Discover problems across all Run IDs
    # Map of problem_id -> list of source_dicts
    from collections import defaultdict
    all_problems = defaultdict(list)
    
    for rid in final_run_ids:
        problem_map = get_problem_ids(log_dir, file_pattern, rid)
        for pid, sources in problem_map.items():
            if problem_id and pid != problem_id:
                continue
            all_problems[pid].extend(sources)

    if not all_problems:
        logger.error(f"No problems found for Run IDs: {', '.join(run_ids)}")
        return

    # Track summary data for final table
    summary_data = []

    # 2. Iterate through each discovered problem
    for pid in sorted(all_problems.keys()):
        if pid == "SETUP":
            continue
        
        candidates = all_problems[pid]
        problem_candidates_data = []

        # Analyze each candidate source for this problem to find the "best" (completed) one
        for source_info in candidates:
            rid = source_info["run_id"]
            stype = source_info["type"]
            sfiles = source_info["files"]

            # Parse the specific logs
            parsed_data = parse_coevolution_log(
                log_dir=log_dir,
                log_filename_pattern=file_pattern,
                target_run_id=rid,
                target_problem_id=pid,
                source_type=stype,
                legacy_files=sfiles
            )
            
            # Determine if this run is "complete" (has a champion)
            matrices = parsed_data["matrices"].get("private", [])
            has_champion = False
            champion_id_list = []
            
            if matrices:
                last_matrix = matrices[-1]
                active_ids = last_matrix.index.tolist()
                individuals_df = parsed_data.get("individuals", pd.DataFrame())
                
                if not individuals_df.empty:
                    code_data = individuals_df[individuals_df["type"] == "code"].copy()
                    if not code_data.empty:
                        # Check if any probability exists for these IDs
                        latest_beliefs = code_data.dropna(subset=["probability"]).groupby("id")["probability"].last()
                        valid_beliefs = {str(iid): latest_beliefs.get(iid) for iid in active_ids if iid in latest_beliefs.index}
                        if valid_beliefs:
                            has_champion = True
                            max_belief = max(valid_beliefs.values())
                            champion_id_list = [iid for iid, b in valid_beliefs.items() if b == max_belief]

            problem_candidates_data.append({
                "run_id": rid,
                "type": stype,
                "parsed_data": parsed_data,
                "is_complete": has_champion,
                "problem_id": pid,
                "champion_ids": champion_id_list
            })

        # --- Collision Detection & Selection ---
        completed_candidates = [c for c in problem_candidates_data if c["is_complete"]]
        
        if len(completed_candidates) > 1:
            # Raise error as requested: completed in multiple runs
            run_names = [c["run_id"] for c in completed_candidates]
            rprint(f"\n[bold red]ERROR:[/bold red] Problem [bold]{pid}[/bold] is COMPLETED in multiple runs: {', '.join(run_names)}")
            rprint("[red]Please resolve this ambiguity by isolating the correct run.[/red]")
            raise typer.Exit(code=1)
        
        # Selection: Use the completed one, or prefer structured
        if completed_candidates:
            selected = completed_candidates[0]
        else:
            # If none are complete, prefer structured if available
            structured_cands = [c for c in problem_candidates_data if c["type"] == "structured"]
            selected = structured_cands[0] if structured_cands else problem_candidates_data[0]

        rid = selected["run_id"]
        stype = selected["type"]
        parsed_data = selected["parsed_data"]
        
        rprint(f"\n[bold green]PROBLEM:[/bold green] [bold white]{pid}[/bold white] [dim](from {rid} [{stype}])[/dim]")
        rprint("[dim]" + "─" * 40 + "[/dim]")

        matrices = parsed_data["matrices"].get("private", [])
        if not matrices:
            summary_data.append({
                "run_id": rid, "problem_id": pid, "solved": False,
                "champion_code_ids": "N/A", "champion_passing": "N/A",
                "champion_probability": 0.0, "initial_pass_at_10": "N/A",
                "final_pass_at_15": "N/A", "final_pass_at_10": "N/A",
            })
            continue

        analyze_matrix(matrices[0], "START")
        if len(matrices) > 1:
            analyze_matrix(matrices[-1], "END  ")
        else:
            rprint("[dim]Run had not progressed; only one matrix present.[/dim]")

        # 4. Collect summary data
        solved = False
        champion_ids = []
        champion_passing_info = []
        champion_probability = 0.0
        num_final_pass_at_10 = 0
        num_final_pass_at_15 = 0

        last_matrix = matrices[-1]
        num_tests = last_matrix.shape[1]
        pass_counts = last_matrix.sum(axis=1)
        
        active_ids = last_matrix.index.tolist()
        individuals_df = parsed_data.get("individuals", pd.DataFrame())
        
        belief_map = {}
        if not individuals_df.empty:
            code_data = individuals_df[individuals_df["type"] == "code"].copy()
            if not code_data.empty:
                latest_beliefs = code_data.dropna(subset=["probability"]).groupby("id")["probability"].last()
                belief_map = {str(iid): latest_beliefs.get(iid, 0.0) for iid in active_ids}
            else:
                belief_map = {str(iid): 0.0 for iid in active_ids}
        else:
            belief_map = {str(iid): 0.0 for iid in active_ids}

        sorted_by_belief = sorted(active_ids, key=lambda x: belief_map.get(str(x), 0.0), reverse=True)
        
        if sorted_by_belief and any(v > 0 for v in belief_map.values()):
            max_belief = belief_map[str(sorted_by_belief[0])]
            champion_probability = max_belief
            champions = [iid for iid in sorted_by_belief if belief_map[str(iid)] == max_belief]
            
            any_champ_solved = False
            for cid in champions:
                c_pass = int(pass_counts.loc[cid]) if cid in pass_counts.index else 0
                champion_ids.append(str(cid))
                champion_passing_info.append(f"{c_pass}/{num_tests}")
                if c_pass == num_tests:
                    any_champ_solved = True
            
            solved = any_champ_solved
            
            def count_perfect_in_top(k: int) -> int:
                perfect = 0
                for cid in sorted_by_belief[:k]:
                    if cid in pass_counts.index and int(pass_counts.loc[cid]) == num_tests:
                        perfect += 1
                return perfect

            num_final_pass_at_10 = count_perfect_in_top(10)
            num_final_pass_at_15 = count_perfect_in_top(15)

        num_initial_passing = len(matrices[0].sum(axis=1)[matrices[0].sum(axis=1) == matrices[0].shape[1]])

        summary_data.append({
            "run_id": rid, "problem_id": pid, "solved": solved,
            "champion_code_ids": ", ".join(champion_ids) if champion_ids else "N/A",
            "champion_passing": ", ".join(champion_passing_info) if champion_passing_info else "N/A",
            "champion_probability": champion_probability,
            "initial_pass_at_10": num_initial_passing,
            "final_pass_at_15": num_final_pass_at_15,
            "final_pass_at_10": num_final_pass_at_10,
        })

    # 5. Print summary table
    print("\n")
    
    # Separate complete and incomplete
    complete_runs = [r for r in summary_data if r["champion_code_ids"] != "N/A"]
    incomplete_runs = [r for r in summary_data if r["champion_code_ids"] == "N/A"]

    table = Table(title=f"AGGREGATED RUNS SUMMARY", title_style="bold underline magenta", show_header=True, header_style="bold cyan")
    table.add_column("Problem", style="white")
    table.add_column("Run ID", style="dim")
    table.add_column("Solved", justify="center")
    table.add_column("Champion ID", style="yellow")
    table.add_column("Score", justify="center")
    table.add_column("Prob", justify="right")
    table.add_column("Final P@10", justify="center", style="bold green")

    for row in complete_runs:
        solved_str = "[bold green]Yes[/bold green]" if row["solved"] else "[bold red]No[/bold red]"
        table.add_row(
            row["problem_id"],
            row["run_id"],
            solved_str,
            row["champion_code_ids"],
            row["champion_passing"],
            f"{row['champion_probability']:.4f}",
            str(row["final_pass_at_10"])
        )
    
    console.print(table)

    if incomplete_runs:
        rprint("\n[bold yellow]Incomplete Problems (No Champion Found in any Run):[/bold yellow]")
        rprint(", ".join([f"[red]{r['problem_id']}[/red] [dim]({r['run_id']})[/dim]" for r in incomplete_runs]))

    total_problems = len(summary_data)
    total_incomplete = len(incomplete_runs)
    total_completed = len(complete_runs)
    total_passed = sum(1 for r in complete_runs if r.get("solved"))
    pass_rate = (total_passed / total_completed * 100) if total_completed else 0.0

    stats_text = Text.assemble(
        ("Total Unique Problems: ", "bold"), (f"{total_problems}\n", "white"),
        ("Incomplete Items:     ", "bold"), (f"{total_incomplete}\n", "yellow"),
        ("Completed Items:      ", "bold"), (f"{total_completed}\n", "cyan"),
        ("Successful Solutions: ", "bold"), (f"{total_passed}\n", "green" if total_passed > 0 else "red"),
        ("Pass Rate (Completed):", "bold"), (f"{pass_rate:.2f}%", "bold yellow")
    )
    console.print("\n", Panel(stats_text, title="Aggregated Statistics", border_style="magenta", expand=False))
    console.rule("[bold magenta]End of Aggregated Report[/bold magenta]")


if __name__ == "__main__":
    app()
