#!/usr/bin/env python3
"""
initial_population_eval.py — Evaluates the initial code population by sampling.

For each problem in a run:
  1. Extracts the generation-0 public observation matrix.
  2. Randomly samples ONE code that passes all public tests.
  3. Checks whether that code also passes all private tests (gen-0 private matrix).
  4. Marks the problem as "passed" if both are satisfied.

Prints a summary table similar to overview.py.

Usage:
    python experiments/scripts/coevolution/analyse/initial_population_eval.py \\
        --run-id <RUN_ID> [--run-id <RUN_ID_2> ...] \\
        [--seed 42] [--log-dir logs]
"""

import fnmatch
import random
import sys
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Allow imports from the coevolution src package
sys.path.append(str(Path(__file__).resolve().parents[4] / "src"))

from coevolution.analysis.log_parser import (
    get_problem_ids,
    get_run_ids,
    parse_coevolution_log,
)

app = typer.Typer(help="Evaluate initial populations via public→private sampling.")
console = Console()


# ─── HELPERS ──────────────────────────────────────────────────────────────────


def _first_matrix_of_type(
    matrices: dict[str, list[Any]], matrix_type: str
) -> Any | None:
    """Return the first (generation-0) matrix of the given type, or None."""
    mats = matrices.get(matrix_type, [])
    return mats[0] if mats else None


def sample_public_passing_code(public_matrix: Any) -> str | None:
    """
    From the gen-0 public observation matrix (a pd.DataFrame),
    randomly sample ONE code ID that passes ALL public tests.
    Returns the code ID string, or None if no such code exists.
    """
    num_tests = public_matrix.shape[1]
    if num_tests == 0:
        return None

    pass_counts = public_matrix.sum(axis=1)
    perfect_codes = pass_counts[pass_counts == num_tests].index.tolist()

    if not perfect_codes:
        return None

    return str(random.choice(perfect_codes))


def check_private_pass(private_matrix: Any | None, code_id: str) -> bool:
    """
    Check whether `code_id` passes ALL tests in the gen-0 private matrix.
    Returns False if the matrix is None, the code is not in it, or it fails any test.
    """
    if private_matrix is None:
        return False

    num_tests = private_matrix.shape[1]
    if num_tests == 0:
        return False

    if code_id not in private_matrix.index:
        return False

    row = private_matrix.loc[code_id]
    return int(row.sum()) == num_tests


# ─── CLI ──────────────────────────────────────────────────────────────────────


@app.command()
def main(
    run_ids: list[str] = typer.Option(
        ..., "--run-id", help="Run IDs to analyse (can be multiple, supports globs)"
    ),
    log_dir: str = typer.Option("logs", help="Directory containing log files"),
    file_pattern: str = typer.Option("*.log", help="Pattern to match legacy log files"),
    problem_id: str = typer.Option(
        None, help="Specific Problem ID to analyse (optional)"
    ),
    legacy: bool = typer.Option(
        False, "--legacy", help="Whether to scan legacy flat log files"
    ),
    seed: int = typer.Option(
        None, "--seed", help="Random seed for reproducible sampling (optional)"
    ),
) -> None:
    """
    For each problem, randomly pick a code from the initial population that
    passes all PUBLIC tests, then check whether it also passes all PRIVATE tests.
    """
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="WARNING")

    if seed is not None:
        random.seed(seed)
        rprint(f"[dim]Random seed set to {seed}[/dim]")

    # ── 0. Expand glob patterns in run_ids ────────────────────────────────────
    all_available_runs = get_run_ids(log_dir, file_pattern, use_legacy=legacy)
    expanded_run_ids: set[str] = set()
    for pattern in run_ids:
        if any(c in pattern for c in "*?[]"):
            matches = [r for r in all_available_runs if fnmatch.fnmatch(r, pattern)]
            if not matches:
                logger.warning(f"No runs matched pattern: {pattern}")
            expanded_run_ids.update(matches)
        else:
            expanded_run_ids.add(pattern)

    final_run_ids = sorted(expanded_run_ids)
    if not final_run_ids:
        logger.error(f"No valid Run IDs found after expansion of: {', '.join(run_ids)}")
        raise typer.Exit(1)

    console.rule(
        f"[bold magenta]INITIAL POPULATION EVAL: {', '.join(final_run_ids)}[/bold magenta]"
    )

    # ── 0.5 Sanitize optional problem_id filter ───────────────────────────────
    from coevolution.utils.paths import sanitize_id

    target_pid = sanitize_id(problem_id) if problem_id else None

    # ── 1. Discover problems across all run IDs ────────────────────────────────
    from collections import defaultdict

    all_problems: dict[str, list[tuple[str, list[str] | None]]] = defaultdict(list)

    for rid in final_run_ids:
        pids, lfiles = get_problem_ids(log_dir, file_pattern, rid, use_legacy=legacy)
        for pid in pids:
            if target_pid and pid != target_pid:
                continue
            all_problems[pid].append((rid, lfiles))

    if not all_problems:
        logger.error(f"No problems found for Run IDs: {', '.join(run_ids)}")
        raise typer.Exit(1)

    # ── 2. Per-problem evaluation ──────────────────────────────────────────────
    summary_data: list[dict[str, Any]] = []

    for pid in sorted(all_problems.keys()):
        if pid == "SETUP":
            continue

        # Use the first (or only) run that has this problem
        rid, cand_lfiles = all_problems[pid][0]

        parsed_data = parse_coevolution_log(
            log_dir=log_dir,
            log_filename_pattern=file_pattern,
            target_run_id=rid,
            target_problem_id=pid,
            use_legacy=legacy,
            legacy_files=cand_lfiles,
        )

        rprint(
            f"\n[bold green]PROBLEM:[/bold green] [bold white]{pid}[/bold white]"
            f" [dim](from {rid})[/dim]"
        )
        rprint("[dim]" + "─" * 40 + "[/dim]")

        matrices = parsed_data["matrices"]

        # ── 2a. Get the initial (gen-0) public matrix ──────────────────────────
        public_matrix = _first_matrix_of_type(matrices, "public")

        # Fallback: some setups call it "fixed"
        if public_matrix is None:
            public_matrix = _first_matrix_of_type(matrices, "fixed")

        if public_matrix is None:
            rprint(
                "  [yellow]⚠[/yellow]  No public observation matrix found — skipping."
            )
            summary_data.append(
                {
                    "run_id": rid,
                    "problem_id": pid,
                    "public_pop_size": 0,
                    "public_perfect": 0,
                    "sampled_code": "N/A",
                    "passed_public": False,
                    "passed_private": False,
                    "private_tests": "N/A",
                }
            )
            continue

        pop_size = public_matrix.shape[0]
        num_public_tests = public_matrix.shape[1]
        public_pass_counts = public_matrix.sum(axis=1)
        public_perfect = int((public_pass_counts == num_public_tests).sum())

        rprint(
            f"  [cyan]Public matrix[/cyan]  : {pop_size} codes × "
            f"{num_public_tests} tests — "
            f"[bold]{public_perfect}[/bold] pass all"
        )

        # ── 2b. Randomly sample one public-passing code ────────────────────────
        sampled_code = sample_public_passing_code(public_matrix)

        if sampled_code is None:
            rprint(
                "  [yellow]⚠[/yellow]  No code in the initial population passes "
                "all public tests."
            )
            summary_data.append(
                {
                    "run_id": rid,
                    "problem_id": pid,
                    "public_pop_size": pop_size,
                    "public_perfect": public_perfect,
                    "sampled_code": "none",
                    "passed_public": False,
                    "passed_private": False,
                    "private_tests": "N/A",
                }
            )
            continue

        rprint(
            f"  [cyan]Sampled code[/cyan]   : [bold yellow]{sampled_code}[/bold yellow]"
            " (randomly selected from public-passing codes)"
        )

        # ── 2c. Check private tests for that code ──────────────────────────────
        private_matrix = _first_matrix_of_type(matrices, "private")

        num_private_tests = (
            private_matrix.shape[1] if private_matrix is not None else 0
        )

        if private_matrix is None:
            rprint(
                "  [yellow]⚠[/yellow]  No private observation matrix found — "
                "cannot verify."
            )
            passed_private = False
            private_result_str = "N/A"
        else:
            if sampled_code in private_matrix.index:
                row = private_matrix.loc[sampled_code]
                private_passed = int(row.sum())
                passed_private = private_passed == num_private_tests and num_private_tests > 0
                private_result_str = f"{private_passed}/{num_private_tests}"
            else:
                rprint(
                    f"  [yellow]⚠[/yellow]  Sampled code [bold]{sampled_code}[/bold]"
                    " not found in private matrix."
                )
                passed_private = False
                private_result_str = "not in matrix"

        status_icon = "[bold green]✔ PASSED[/bold green]" if passed_private else "[bold red]✘ FAILED[/bold red]"
        rprint(
            f"  [cyan]Private result[/cyan] : {status_icon} "
            f"[dim]({private_result_str} private tests)[/dim]"
        )

        summary_data.append(
            {
                "run_id": rid,
                "problem_id": pid,
                "public_pop_size": pop_size,
                "public_perfect": public_perfect,
                "sampled_code": sampled_code,
                "passed_public": True,
                "passed_private": passed_private,
                "private_tests": private_result_str,
            }
        )

    # ── 3. Summary table (overview.py style) ──────────────────────────────────
    print("\n")

    solved_rows = [r for r in summary_data if r["passed_private"]]
    failed_rows = [r for r in summary_data if not r["passed_private"]]
    no_candidate_rows = [r for r in summary_data if not r["passed_public"]]

    table = Table(
        title="INITIAL POPULATION SAMPLING RESULT",
        title_style="bold underline magenta",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Problem", style="white")
    table.add_column("Run ID", style="dim")
    table.add_column("Pop", justify="right")
    table.add_column("Public Perfect", justify="center", style="bold yellow")
    table.add_column("Sampled", style="yellow")
    table.add_column("Private Score", justify="center")
    table.add_column("Result", justify="center")

    for row in summary_data:
        result_str = (
            "[bold green]✔ PASSED[/bold green]"
            if row["passed_private"]
            else "[bold red]✘ FAILED[/bold red]"
            if row["passed_public"]
            else "[dim]—[/dim]"
        )
        table.add_row(
            str(row["problem_id"]),
            str(row["run_id"]),
            str(row["public_pop_size"]),
            str(row["public_perfect"]),
            str(row["sampled_code"]),
            str(row["private_tests"]),
            result_str,
        )

    console.print(table)

    # ── 4. Aggregate stats ─────────────────────────────────────────────────────
    total_problems = len(summary_data)
    total_no_candidate = len(no_candidate_rows)
    total_with_candidate = total_problems - total_no_candidate
    total_passed = len(solved_rows)
    pass_rate = (total_passed / total_with_candidate * 100) if total_with_candidate else 0.0

    stats_text = Text.assemble(
        ("Total Problems:             ", "bold"),
        (f"{total_problems}\n", "white"),
        ("No Public-Passing Code:     ", "bold"),
        (f"{total_no_candidate}\n", "yellow"),
        ("Had Public-Passing Code:    ", "bold"),
        (f"{total_with_candidate}\n", "cyan"),
        ("Passed Private Tests:       ", "bold"),
        (f"{total_passed}\n", "green" if total_passed > 0 else "red"),
        ("Pass Rate (w/ candidate):   ", "bold"),
        (f"{pass_rate:.2f}%", "bold yellow"),
    )
    console.print(
        "\n",
        Panel(
            stats_text,
            title="Sampling Statistics",
            border_style="magenta",
            expand=False,
        ),
    )
    console.rule("[bold magenta]End of Initial Population Report[/bold magenta]")


if __name__ == "__main__":
    app()
