#!/usr/bin/env python3
"""
show_progress.py — Renders a high-fidelity terminal dashboard for coevolution runs.
Parses structured JSONL telemetry from evolutionary_history.jsonl.

Usage:
    python experiments/scripts/coevolution/analyse/show_progress.py --run-id <ID> --problem-id <ID>
"""

import json
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

app = typer.Typer(help="Visualize coevolutionary progress with a premium terminal UI.")
console = Console()

# ─── LOGIC ────────────────────────────────────────────────────────────────────


def fmt_id(s: str, max_len: int = 16) -> str:
    """Middle-truncate an ID so the unique suffix is always visible."""
    if not isinstance(s, str):
        return str(s)
    if len(s) <= max_len:
        return s
    keep_end = 8
    keep_start = max_len - keep_end - 1
    return s[:keep_start] + "…" + s[-keep_end:]


def load_events(run_id: str, problem_id: str) -> list[dict[str, Any]]:
    log_path = Path("logs") / run_id / problem_id / "evolutionary_history.jsonl"
    if not log_path.exists():
        console.print(f"[red]Error: Found no history at {log_path}[/red]")
        raise typer.Exit(1)

    events = []
    with open(log_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    continue
                record = data.get("record", {})
                extra = record.get("extra", {})
                event_data = extra.get("event_data", {})
                event_type = record.get("message", "UNKNOWN")
                if event_type == "LIFECYCLE_EVENT":
                    event_type = event_data.get("event", "UNKNOWN").upper()

                events.append(
                    {
                        "timestamp": record.get("time", {}).get("repr", ""),
                        "event_type": event_type,
                        **event_data,
                    }
                )
            except json.JSONDecodeError:
                continue
    return events


def load_config(run_id: str) -> dict[str, Any] | None:
    config_path = Path("logs") / run_id / "run_config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return None
    except Exception:
        return None


# ─── RENDERERS ────────────────────────────────────────────────────────────────


def display_config(config: dict[str, Any]) -> None:
    """Renders a premium dashboard header with all run parameters."""
    console.rule("[bold cyan]RUN CONFIGURATION[/bold cyan]", style="cyan")

    # 1. Experiment Summary (Metadata & LLM)
    exp_cfg = config.get("experiment", {})
    llm_cfg = config.get("llm", {})

    metadata_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    metadata_table.add_column("Key", style="bold cyan", no_wrap=True)
    metadata_table.add_column("Value")

    metadata_table.add_row("Experiment", exp_cfg.get("name", "N/A"))
    metadata_table.add_row(
        "Description", f"[dim]{exp_cfg.get('description', 'N/A')}[/dim]"
    )
    metadata_table.add_row(
        "LLM Model",
        f"[green]{llm_cfg.get('provider', 'N/A')}/{llm_cfg.get('model', 'N/A')}[/green]",
    )
    metadata_table.add_row(
        "Language", f"[yellow]{exp_cfg.get('language', 'N/A')}[/yellow]"
    )

    console.print(Panel(metadata_table, title="[bold]Metadata", border_style="dim"))

    # 2. Population Strategy (Settings + Operators)
    code_prof = config.get("code_profile", {})
    test_profs = config.get("test_profiles", {})

    pop_table = Table(
        title="Coevolutionary Strategy", box=box.ROUNDED, border_style="dim"
    )
    pop_table.add_column("Population", style="bold")
    pop_table.add_column("Sizes (Init/Max)", justify="center")
    pop_table.add_column("Elitism", justify="center")
    pop_table.add_column("Operators (Rates)", style="dim")

    # Add Code Row
    code_cfg = code_prof.get("population_config", {})
    code_ops = code_prof.get("operator_rates", {})
    ops_str = ", ".join([f"{k}:{v:.1f}" for k, v in code_ops.items()])
    pop_table.add_row(
        "Code",
        f"{code_cfg.get('initial_population_size')} / {code_cfg.get('max_population_size')}",
        f"{code_cfg.get('elitism_rate', 0):.1%}",
        ops_str,
    )

    # Add Test Rows
    for t_type, t_prof in test_profs.items():
        t_cfg = t_prof.get("population_config", {})
        t_ops = t_prof.get("operator_rates", {})
        t_ops_str = ", ".join([f"{k}:{v:.1f}" for k, v in t_ops.items()])
        pop_table.add_row(
            t_type.title(),
            f"{t_cfg.get('initial_population_size')} / {t_cfg.get('max_population_size')}",
            f"{t_cfg.get('elitism_rate', 0):.1%}",
            t_ops_str,
        )

    console.print(pop_table)

    # 3. Evolution Schedule
    schedule_data = config.get("evolution_config", {}).get("schedule", {})
    phases = schedule_data.get("phases", [])

    if phases:
        sched_table = Table(
            title="Evolutionary Schedule", box=box.SIMPLE, border_style="dim"
        )
        sched_table.add_column("Phase", style="cyan")
        sched_table.add_column("Dur", justify="right")
        sched_table.add_column("Evolve Code", justify="center")
        sched_table.add_column("Evolve Tests", justify="center")

        for p in phases:
            sched_table.add_row(
                p.get("name", "Unnamed"),
                str(p.get("duration", 0)),
                "[green]✓[/green]" if p.get("evolve_code") else "[red]✗[/red]",
                "[green]✓[/green]" if p.get("evolve_tests") else "[red]✗[/red]",
            )
    console.print(sched_table)
    console.print()


def display_initialization(events: list[dict[str, Any]]) -> None:
    console.rule("[bold cyan]INITIALIZATION[/bold cyan]", style="cyan")

    gen0_creations = [
        e for e in events if e["event_type"] == "CREATED" and e.get("generation") == 0
    ]

    if not gen0_creations:
        console.print("[dim]No initialization data found.[/dim]")
    else:
        # Group code snippets
        code_creations = [
            e for e in gen0_creations if str(e.get("individual_id", "")).startswith("C")
        ]

        if code_creations:
            console.print("\n[bold]Seed Populations[/bold]")
            for e in code_creations:
                snippet = e.get("snippet", "")
                ind_id = e.get("individual_id", "?")
                console.print(
                    Panel(
                        Syntax(
                            snippet,
                            "python",
                            theme="ansi_dark",
                            line_numbers=True,
                            word_wrap=True,
                        ),
                        title=f"[bold]Seed Individual {fmt_id(ind_id)}[/bold]",
                        border_style="dim",
                    )
                )

    # Show private matrix for Gen 0 if it exists
    gen0_matrices = [
        e
        for e in events
        if e["event_type"] == "OBSERVATION_MATRIX"
        and e.get("generation") == 0
        and e.get("test_type") == "private"
    ]
    if gen0_matrices:
        render_observation_matrix(gen0_matrices[-1])


def display_belief_update(update: dict[str, Any], counter: int) -> None:
    pop_type = update.get("population", "unknown")
    based_on = update.get("based_on", "unknown")
    ids = update.get("ids", [])
    prior = update.get("prior", [])
    posterior = update.get("posterior", [])

    if pop_type == "test":
        title = f"{counter}. [magenta]Evolved {based_on} Tests[/magenta]: updated based on new code performance"
    else:
        title = f"{counter}. [cyan]Code Population[/cyan]: updated based on {based_on} results"

    console.print(Text.from_markup(f"\n{title}"))

    table = Table(show_header=True, box=box.SIMPLE_HEAD)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Prior", justify="right")
    table.add_column("Posterior", justify="right", style="bold green")
    table.add_column("Delta", justify="right")

    for i, ind_id in enumerate(ids):
        p_before = prior[i] if i < len(prior) else 0.0
        p_after = posterior[i] if i < len(posterior) else 0.0
        delta = p_after - p_before
        delta_str = (
            f"[green]+{delta:.4f}[/green]"
            if delta > 0
            else f"[red]{delta:.4f}[/red]"
            if delta < 0
            else "[dim]0.0000[/dim]"
        )
        table.add_row(
            fmt_id(str(ind_id)), f"{p_before:.4f}", f"{p_after:.4f}", delta_str
        )

    console.print(table)


def render_observation_matrix(matrix_event: dict[str, Any]) -> None:
    """Helper to render a single observation matrix table."""
    test_type = str(matrix_event.get("test_type", "Unknown")).upper()
    console.print(
        f"\n[bold underline cyan]{test_type} OBSERVATION MATRIX[/bold underline cyan]"
    )

    matrix_data = matrix_event.get("matrix", [])
    code_ids = matrix_event.get("code_ids", [])
    test_ids = matrix_event.get("test_ids", [])

    if matrix_data and code_ids and test_ids:
        table = Table(show_header=True, box=box.SIMPLE, border_style="dim")
        table.add_column("Code \\ Test", style="bold cyan")
        for tid in test_ids:
            table.add_column(fmt_id(str(tid)))

        for i, row_data in enumerate(matrix_data):
            code_id = fmt_id(str(code_ids[i])) if i < len(code_ids) else "C?"
            # Colorize results: 1 = green, 0 = red
            colored_row = [
                "[green]1[/green]" if x == 1 else "[red]0[/red]" if x == 0 else str(x)
                for x in row_data
            ]
            table.add_row(code_id, *colored_row)
        console.print(table)


def display_succession(events: list[dict[str, Any]], current_gen: int) -> None:
    """Renders elites selected from current_gen and newborns bred for current_gen + 1."""

    # 1. Elites from current_gen (who survive into current_gen + 1)
    elites = [
        e
        for e in events
        if e["event_type"] == "SELECTED_AS_ELITE" and e.get("generation") == current_gen
    ]

    # 2. Newborns for next_gen
    next_gen = current_gen + 1
    newborns = [
        e
        for e in events
        if e["event_type"] == "CREATED" and e.get("generation") == next_gen
    ]

    if not elites and not newborns:
        console.print(
            f"\n[bold dim yellow]SUCCESSION: PREPARING GENERATION {next_gen} (ALL POPULATIONS FROZEN)[/bold dim yellow]"
        )
        return

    console.print(
        f"\n[bold underline yellow]SUCCESSION: PREPARING GENERATION {next_gen}[/bold underline yellow]"
    )

    # Check which populations were frozen
    code_elites = [e for e in elites if str(e.get("individual_id", "")).startswith("C")]
    test_elites = [
        e for e in elites if not str(e.get("individual_id", "")).startswith("C")
    ]
    code_newborns = [
        n for n in newborns if str(n.get("individual_id", "")).startswith("C")
    ]
    test_newborns = [
        n for n in newborns if not str(n.get("individual_id", "")).startswith("C")
    ]

    code_frozen = not code_elites and not code_newborns
    tests_frozen = not test_elites and not test_newborns

    if code_frozen:
        console.print(
            "[dim cyan]Code Population: Frozen (No selection or breeding this generation)[/dim cyan]"
        )
    if tests_frozen:
        console.print(
            "[dim magenta]Test Populations: Frozen (No selection or breeding this generation)[/dim magenta]"
        )

    # --- ELITE SELECTION (BANNERS) ---
    if elites:
        if code_elites:
            ids_str = ", ".join(
                fmt_id(str(e.get("individual_id"))) for e in code_elites
            )
            console.print(
                Panel(
                    Text.from_markup(
                        f"[bold cyan]Elite Code Kept:[/bold cyan] {ids_str}"
                    ),
                    border_style="cyan",
                    padding=(0, 1),
                )
            )

        # Separate each test type elite collection
        if test_elites:
            test_types = sorted(
                list(set(e.get("test_type", "Evolved Tests") for e in test_elites))
            )
            for tt in test_types:
                tt_elites = [
                    e for e in test_elites if e.get("test_type", "Evolved Tests") == tt
                ]
                ids_str = ", ".join(
                    fmt_id(str(e.get("individual_id"))) for e in tt_elites
                )
                console.print(
                    Panel(
                        Text.from_markup(
                            f"[bold magenta]Elite {tt.title()} Kept:[/bold magenta] {ids_str}"
                        ),
                        border_style="magenta",
                        padding=(0, 1),
                    )
                )

    # --- BREEDING OPERATIONS (TABLES) ---
    if newborns:
        if code_newborns:
            table = Table(
                title=f"Bred Code Offspring (for Gen {next_gen})",
                box=box.ROUNDED,
                border_style="cyan",
            )
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Operator", style="magenta")
            table.add_column("Parents", style="dim")
            table.add_column("Initial Belief", justify="right")
            for row in code_newborns:
                ind_id = str(row.get("individual_id", ""))
                op = row.get("creation_op", "UNKNOWN")
                prob = row.get("probability", 0.0)
                parents_dict = row.get("parents", {})
                parents_list = parents_dict.get("code", []) + parents_dict.get(
                    "test", []
                )
                parents_str = (
                    ", ".join(map(fmt_id, parents_list)) if parents_list else "—"
                )
                table.add_row(fmt_id(ind_id), str(op), parents_str, f"{prob:.4f}")
            console.print(table)

        if test_newborns:
            # Strictly separate each test type's breeding table
            test_types = sorted(
                list(set(n.get("test_type", "Evolved Tests") for n in test_newborns))
            )
            for tt in test_types:
                tt_newborns = [
                    n
                    for n in test_newborns
                    if n.get("test_type", "Evolved Tests") == tt
                ]
                table = Table(
                    title=f"Bred {tt.title()} Offspring (for Gen {next_gen})",
                    box=box.ROUNDED,
                    border_style="magenta",
                )
                table.add_column("ID", style="magenta", no_wrap=True)
                table.add_column("Operator", style="magenta")
                table.add_column("Parents", style="dim")
                table.add_column("Initial Belief", justify="right")
                for row in tt_newborns:
                    ind_id = str(row.get("individual_id", ""))
                    op = row.get("creation_op", "UNKNOWN")
                    prob = row.get("probability", 0.0)
                    parents_dict = row.get("parents", {})
                    parents_list = parents_dict.get("code", []) + parents_dict.get(
                        "test", []
                    )
                    parents_str = (
                        ", ".join(map(fmt_id, parents_list)) if parents_list else "—"
                    )
                    table.add_row(fmt_id(ind_id), str(op), parents_str, f"{prob:.4f}")
                console.print(table)


def render_generation(
    events: list[dict[str, Any]], gen: int, show_errors: bool = False
) -> None:
    gen_events = [e for e in events if e.get("generation") == gen]
    if not gen_events:
        return

    console.rule(f"[bold yellow]GENERATION {gen}[/bold yellow]", style="yellow")

    # 1. Matrices & Failures (Work)
    matrices = [e for e in gen_events if e["event_type"] == "OBSERVATION_MATRIX"]
    failures = [e for e in gen_events if e["event_type"] == "EVALUATION_FAILED"]

    if matrices:
        for row in matrices:
            test_type = str(row.get("test_type", "Unknown")).upper()
            if test_type == "PRIVATE":
                continue  # Handled in final/initial summary

            render_observation_matrix(row)

            # Failures for this matrix type
            type_failures = [
                f for f in failures if f.get("test_type") == test_type.lower()
            ]
            if show_errors:
                if type_failures:
                    console.print(
                        f"[bold yellow]Error Logs ({test_type}):[/bold yellow]"
                    )
                    for f_row in type_failures:
                        c_id = fmt_id(str(f_row.get("code_id")))
                        t_id = fmt_id(str(f_row.get("test_id")))
                        err = f_row.get("error_log", "")
                        clean_err = (
                            err.split("\n")[0][:120] + "..." if err else "Unknown Error"
                        )
                        console.print(
                            f"  [red]✖[/red] [bold]{c_id}[/bold] vs [bold]{t_id}[/bold]: [dim]{clean_err}[/dim]"
                        )
            elif type_failures:
                num_fails = len(type_failures)
                if num_fails > 0:
                    console.print(
                        f"[dim yellow italic]({num_fails} evaluation failures hidden. Use --show-errors to see details)[/dim yellow italic]"
                    )

    # 2. Belief Updates (Learning)
    belief_updates = [e for e in gen_events if e["event_type"] == "BELIEF_UPDATE"]
    if belief_updates:
        console.print(
            "\n[bold underline yellow]BAYESIAN BELIEF PROGRESSION (BACE)[/bold underline yellow]"
        )

        anchor_updates = [
            u
            for u in belief_updates
            if u.get("population") == "code" and u.get("based_on") == "public"
        ]
        if anchor_updates:
            console.print(
                "\n[bold cyan]Phase 1: Code Anchoring (on Public Tests)[/bold cyan]"
            )
            for i, update in enumerate(anchor_updates, 1):
                display_belief_update(update, i)

        test_updates = [u for u in belief_updates if u.get("population") == "test"]
        if test_updates:
            console.print(
                "\n[bold magenta]Phase 2: Test Auditing (on Anchored Code)[/bold magenta]"
            )
            test_types = sorted(
                list(set(u.get("test_type", "unknown") for u in test_updates))
            )
            for tt in test_types:
                tt_updates = [u for u in test_updates if u.get("test_type") == tt]
                console.print(
                    f"  [italic yellow]-- {tt.upper()} Population Updates --[/italic yellow]"
                )
                for i, update in enumerate(tt_updates, 1):
                    display_belief_update(update, i)

        backprop_updates = [
            u
            for u in belief_updates
            if u.get("population") == "code" and u.get("based_on") != "public"
        ]
        if backprop_updates:
            console.print(
                "\n[bold green]Phase 3: Back-Propagation (on Test Priors)[/bold green]"
            )
            for i, update in enumerate(backprop_updates, 1):
                source = update.get("based_on", "unknown").upper()
                console.print(
                    f"  [italic dim]Source: {source} interaction evidence[/italic dim]"
                )
                display_belief_update(update, i)

    # 3. Succession (Succession planning for N + 1)
    display_succession(events, gen)


# ─── CLI ENTRY ────────────────────────────────────────────────────────────────


@app.command()
def main(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID of the experiment"),
    problem_id: str = typer.Option(
        ..., "--problem-id", "-p", help="Problem ID (e.g. HumanEval/1)"
    ),
    show_errors: bool = typer.Option(
        False, "--show-errors", "-e", help="Show detailed evaluation error logs"
    ),
) -> None:
    """Historical narrative and dashboard for a coevolution problem."""

    console.print(
        Panel(
            Text.from_markup(
                f"Problem: [bold green]{problem_id}[/bold green]\nRun: [dim]{run_id}[/dim]"
            ),
            title="[bold]COEVOLUTION DASHBOARD[/bold]",
            border_style="bold blue",
            expand=False,
        )
    )

    events = load_events(run_id, problem_id)
    if not events:
        return

    # Snippets cache
    snippets: dict[str, str] = {}
    for e in events:
        if e["event_type"] == "CREATED" and "snippet" in e:
            snippets[e["individual_id"]] = e["snippet"]

    # Phase 0: Configuration
    config = load_config(run_id)
    if config:
        display_config(config)

    # Phase 1: Initialization
    display_initialization(events)

    # Phase 2: Generations
    max_gen = 0
    for e in events:
        g = e.get("generation")
        if isinstance(g, int) and g > max_gen:
            max_gen = g

    for gen in range(0, max_gen + 1):
        render_generation(events, gen, show_errors=show_errors)

    # Final Summary (Champions)
    survivors = [e for e in events if e["event_type"] == "SURVIVED"]

    # Sort survivors by probability for champion identification
    survivors.sort(key=lambda s: s.get("probability", 0), reverse=True)

    if survivors:
        # Champion Analysis (Private Tests)
        # Find the latest private observation matrix
        private_matrices = [
            e
            for e in events
            if e["event_type"] == "OBSERVATION_MATRIX"
            and e.get("test_type") == "private"
        ]

        if private_matrices:
            last_private = private_matrices[-1]
            render_observation_matrix(last_private)

            matrix = last_private.get("matrix", [])
            code_ids = last_private.get("code_ids", [])
            test_ids = last_private.get("test_ids", [])
            num_tests = len(test_ids)

            # Identify champion among survivors (highest probability)
            code_survivors = [
                s for s in survivors if str(s.get("individual_id", "")).startswith("C")
            ]
            if code_survivors:
                champion_meta = code_survivors[0]  # Sorted by prob
                champion_id = champion_meta["individual_id"]
                champion_prob = champion_meta.get("probability", 0.0)

                # Find its row in the matrix
                try:
                    c_idx = code_ids.index(champion_id)
                    results = matrix[c_idx]
                    passed = sum(results)

                    is_solved = passed == num_tests

                    banner_style = "bold green" if is_solved else "bold red"
                    banner_msg = "SOLUTION FOUND" if is_solved else "SOLUTION NOT FOUND"

                    console.print("\n")
                    console.print(
                        Panel(
                            Text.from_markup(
                                f"[{banner_style}]{passed}/{num_tests} PRIVATE TESTS PASSED — {banner_msg}[/{banner_style}]"
                            ),
                            title="[bold yellow]CHAMPION VERIFICATION[/bold yellow]",
                            border_style=banner_style,
                            padding=(1, 2),
                        )
                    )

                    # Show champion code and belief
                    champion_snippet = snippets.get(
                        champion_id, "Code snippet not found in logs."
                    )
                    console.print(
                        Panel(
                            Syntax(
                                champion_snippet,
                                "python",
                                theme="ansi_dark",
                                line_numbers=True,
                                word_wrap=True,
                            ),
                            title=f"[bold]Champion {fmt_id(champion_id)}[/bold] (Belief: {champion_prob:.4f})",
                            border_style="cyan",
                        )
                    )

                except (ValueError, IndexError):
                    console.print(
                        "[yellow]Warning: Champion not found in final private matrix.[/yellow]"
                    )
        else:
            console.print(
                "\n[dim]No private test results available for final verification.[/dim]"
            )


if __name__ == "__main__":
    app()
