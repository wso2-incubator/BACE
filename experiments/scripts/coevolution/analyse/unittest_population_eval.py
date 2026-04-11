#!/usr/bin/env python3
"""
unittest_population_eval.py — Evaluates the initial code population using
a two-stage selection:

  Stage 1 (public filter):  Keep only codes that pass ALL public tests.
  Stage 2 (unittest rank):  From those, pick the code that passes the MOST
                             generated static unittests (first unittest
                             observation matrix).  Ties are broken randomly.
  Verify:                   Check whether the selected code passes ALL
                             private tests (first private observation matrix).

Prints a summary table and aggregate statistics similar to overview.py.

Usage:
    python experiments/scripts/coevolution/analyse/unittest_population_eval.py \\
        --run-id <RUN_ID> [--run-id <RUN_ID_2> ...] \\
        [--seed 42] [--log-dir logs]
"""

import fnmatch
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
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

app = typer.Typer(
    help="Evaluate initial populations via public→unittest ranking→private verification."
)
console = Console()


# ─── HELPERS ──────────────────────────────────────────────────────────────────


def _first_matrix_of_type(
    matrices: dict[str, list[pd.DataFrame]], matrix_type: str
) -> pd.DataFrame | None:
    """Return the first (generation-0) matrix of the given type, or None."""
    mats = matrices.get(matrix_type, [])
    return mats[0] if mats else None


def public_passing_codes(public_matrix: pd.DataFrame) -> list[str]:
    """
    Returns a list of code IDs that pass ALL public tests.
    Returns an empty list if no such code exists or the matrix has no tests.
    """
    num_tests = public_matrix.shape[1]
    if num_tests == 0:
        return []
    pass_counts = public_matrix.sum(axis=1)
    return [str(c) for c in pass_counts[pass_counts == num_tests].index.tolist()]


def select_best_by_unittest(
    candidates: list[str],
    unittest_matrix: pd.DataFrame,
) -> tuple[str | None, int, int]:
    """
    Given a list of candidate code IDs and the gen-0 unittest observation
    matrix, return the code ID that passes the most unittests.

    Ties are broken randomly.

    Returns:
        (selected_code_id, unittest_score, total_unittests)
        If none of the candidates appear in the unittest matrix, returns
        (None, 0, total_unittests).
    """
    num_unittest_tests = unittest_matrix.shape[1]

    # Intersect candidates with the unittest matrix index
    available = [c for c in candidates if c in unittest_matrix.index]
    if not available:
        return None, 0, num_unittest_tests

    scores = unittest_matrix.loc[available].sum(axis=1)
    max_score = int(scores.max())

    # Collect all codes tied at the maximum score, then break ties randomly
    top_codes = scores[scores == max_score].index.tolist()
    selected = str(random.choice(top_codes))

    return selected, max_score, num_unittest_tests


def check_private_pass(
    private_matrix: pd.DataFrame | None, code_id: str
) -> tuple[bool, str]:
    """
    Check whether `code_id` passes ALL tests in the gen-0 private matrix.

    Returns:
        (passed: bool, score_str: str)  e.g. (True, "5/5") or (False, "3/5")
    """
    if private_matrix is None:
        return False, "N/A"

    num_tests = private_matrix.shape[1]
    if num_tests == 0:
        return False, "0/0"

    if code_id not in private_matrix.index:
        return False, "not in matrix"

    row = private_matrix.loc[code_id]
    passed = int(row.sum())
    return passed == num_tests, f"{passed}/{num_tests}"


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
        None, "--seed", help="Random seed for reproducible tie-breaking (optional)"
    ),
) -> None:
    """
    Two-stage initial-population selection:
      1. Filter by public tests (must pass all).
      2. Rank remaining codes by static unittest score; pick the best.
      3. Verify the selected code against private tests.
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
        logger.error(
            f"No valid Run IDs found after expansion of: {', '.join(run_ids)}"
        )
        raise typer.Exit(1)

    console.rule(
        f"[bold magenta]UNITTEST-RANKED INITIAL POP EVAL: "
        f"{', '.join(final_run_ids)}[/bold magenta]"
    )

    # ── 0.5 Optional problem_id filter ────────────────────────────────────────
    from coevolution.utils.paths import sanitize_id

    target_pid = sanitize_id(problem_id) if problem_id else None

    # ── 1. Discover problems ───────────────────────────────────────────────────
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

        def _skip(reason: str) -> dict[str, Any]:
            rprint(f"  [yellow]⚠[/yellow]  {reason}")
            return {
                "run_id": rid,
                "problem_id": pid,
                "public_pop_size": 0,
                "public_perfect": 0,
                "unittest_candidates": 0,
                "selected_code": "N/A",
                "unittest_score": "N/A",
                "passed_public": False,
                "passed_private": False,
                "private_score": "N/A",
            }

        # ── Stage 1: public filter ─────────────────────────────────────────────
        public_matrix = _first_matrix_of_type(matrices, "public")
        if public_matrix is None:
            public_matrix = _first_matrix_of_type(matrices, "fixed")

        if public_matrix is None:
            summary_data.append(_skip("No public observation matrix found — skipping."))
            continue

        pop_size = public_matrix.shape[0]
        num_public_tests = public_matrix.shape[1]
        candidates = public_passing_codes(public_matrix)
        public_perfect = len(candidates)

        rprint(
            f"  [cyan]Public matrix[/cyan]   : {pop_size} codes × "
            f"{num_public_tests} tests — "
            f"[bold]{public_perfect}[/bold] pass all"
        )

        if not candidates:
            rprint(
                "  [yellow]⚠[/yellow]  No code passes all public tests — skipping."
            )
            summary_data.append(
                {
                    "run_id": rid,
                    "problem_id": pid,
                    "public_pop_size": pop_size,
                    "public_perfect": 0,
                    "unittest_candidates": 0,
                    "selected_code": "none",
                    "unittest_score": "N/A",
                    "passed_public": False,
                    "passed_private": False,
                    "private_score": "N/A",
                }
            )
            continue

        # ── Stage 2: unittest ranking ──────────────────────────────────────────
        unittest_matrix = _first_matrix_of_type(matrices, "unittest")

        if unittest_matrix is None:
            rprint(
                "  [yellow]⚠[/yellow]  No unittest observation matrix found — "
                "cannot rank by unittest. Skipping."
            )
            summary_data.append(
                {
                    "run_id": rid,
                    "problem_id": pid,
                    "public_pop_size": pop_size,
                    "public_perfect": public_perfect,
                    "unittest_candidates": public_perfect,
                    "selected_code": "N/A",
                    "unittest_score": "N/A",
                    "passed_public": True,
                    "passed_private": False,
                    "private_score": "N/A",
                }
            )
            continue

        num_unittests = unittest_matrix.shape[1]
        selected_code, unittest_score, _ = select_best_by_unittest(
            candidates, unittest_matrix
        )

        if selected_code is None:
            rprint(
                "  [yellow]⚠[/yellow]  None of the public-passing codes appear "
                "in the unittest matrix — skipping."
            )
            summary_data.append(
                {
                    "run_id": rid,
                    "problem_id": pid,
                    "public_pop_size": pop_size,
                    "public_perfect": public_perfect,
                    "unittest_candidates": 0,
                    "selected_code": "none",
                    "unittest_score": "N/A",
                    "passed_public": True,
                    "passed_private": False,
                    "private_score": "N/A",
                }
            )
            continue

        rprint(
            f"  [cyan]Unittest matrix[/cyan] : {unittest_matrix.shape[0]} codes × "
            f"{num_unittests} tests"
        )
        rprint(
            f"  [cyan]Selected code[/cyan]   : [bold yellow]{selected_code}[/bold yellow]"
            f"  (unittest score: [bold]{unittest_score}/{num_unittests}[/bold])"
        )

        # ── Stage 3: private verification ──────────────────────────────────────
        private_matrix = _first_matrix_of_type(matrices, "private")
        passed_private, private_score = check_private_pass(private_matrix, selected_code)

        status_icon = (
            "[bold green]✔ PASSED[/bold green]"
            if passed_private
            else "[bold red]✘ FAILED[/bold red]"
        )
        rprint(
            f"  [cyan]Private result[/cyan]  : {status_icon} "
            f"[dim]({private_score} private tests)[/dim]"
        )

        summary_data.append(
            {
                "run_id": rid,
                "problem_id": pid,
                "public_pop_size": pop_size,
                "public_perfect": public_perfect,
                "unittest_candidates": len(
                    [c for c in candidates if c in unittest_matrix.index]
                ),
                "selected_code": selected_code,
                "unittest_score": f"{unittest_score}/{num_unittests}",
                "passed_public": True,
                "passed_private": passed_private,
                "private_score": private_score,
            }
        )

    # ── 3. Summary table ───────────────────────────────────────────────────────
    print("\n")

    table = Table(
        title="UNITTEST-RANKED INITIAL POPULATION RESULT",
        title_style="bold underline magenta",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Problem", style="white")
    table.add_column("Run ID", style="dim")
    table.add_column("Pop", justify="right")
    table.add_column("Pub ✔", justify="center", style="bold yellow")
    table.add_column("UT Cands", justify="center")
    table.add_column("Selected", style="yellow")
    table.add_column("UT Score", justify="center")
    table.add_column("Private", justify="center")
    table.add_column("Result", justify="center")

    for row in summary_data:
        if row["passed_private"]:
            result_str = "[bold green]✔ PASSED[/bold green]"
        elif row["passed_public"]:
            result_str = "[bold red]✘ FAILED[/bold red]"
        else:
            result_str = "[dim]— no candidate —[/dim]"

        table.add_row(
            str(row["problem_id"]),
            str(row["run_id"]),
            str(row["public_pop_size"]),
            str(row["public_perfect"]),
            str(row["unittest_candidates"]),
            str(row["selected_code"]),
            str(row["unittest_score"]),
            str(row["private_score"]),
            result_str,
        )

    console.print(table)

    # ── 4. Aggregate stats ─────────────────────────────────────────────────────
    total_problems = len(summary_data)
    no_candidate_rows = [r for r in summary_data if not r["passed_public"]]
    solved_rows = [r for r in summary_data if r["passed_private"]]
    total_no_candidate = len(no_candidate_rows)
    total_with_candidate = total_problems - total_no_candidate
    total_passed = len(solved_rows)
    pass_rate = (total_passed / total_problems * 100) if total_problems else 0.0

    stats_text = Text.assemble(
        ("Total Problems:             ", "bold"),
        (f"{total_problems}\n", "white"),
        ("No Public-Passing Code:     ", "bold"),
        (f"{total_no_candidate}\n", "yellow"),
        ("Had Public-Passing Code:    ", "bold"),
        (f"{total_with_candidate}\n", "cyan"),
        ("Passed Private Tests:       ", "bold"),
        (f"{total_passed}\n", "green" if total_passed > 0 else "red"),
        ("Pass Rate:                  ", "bold"),
        (f"{pass_rate:.2f}%  ", "bold yellow"),
        (f"({total_passed}/{total_problems})", "dim"),
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
    console.rule(
        "[bold magenta]End of Unittest-Ranked Initial Population Report[/bold magenta]"
    )


if __name__ == "__main__":
    app()
