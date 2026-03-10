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
from rich.columns import Columns
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


def extract_operator_rates(prof: dict[str, Any]) -> str:
    """Extracts operator rates from both sub-dictionary and root-level keys."""
    rates = prof.get("operator_rates", {})
    if not rates:
        # Fallback to root-level *_rate keys
        rates = {
            k.replace("_rate", ""): v
            for k, v in prof.items()
            if k.endswith("_rate") and isinstance(v, (int, float))
        }
    if not rates:
        return ""
    return ", ".join([f"{k}:{v:.1f}" for k, v in rates.items()])


def reconstruct_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the evolution schedule into a list of epoch definitions."""
    schedule_data = config.get("evolution_config", {}).get("schedule", {})
    phases = schedule_data.get("phases", [])
    epochs = []
    for p in phases:
        for _ in range(p.get("duration", 0)):
            epochs.append(
                {
                    "phase_name": p.get("name", "Unknown"),
                    "evolve_code": p.get("evolve_code", False),
                    "evolve_tests": p.get("evolve_tests", False),
                }
            )
    return epochs


def group_events_into_cycles(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group events into cycles delimited by GENERATION_SUMMARY."""
    cycles = []
    current_cycle = []
    for e in events:
        current_cycle.append(e)
        if e["event_type"] == "GENERATION_SUMMARY":
            cycles.append(current_cycle)
            current_cycle = []
    if current_cycle:
        cycles.append(current_cycle)
    return cycles


# ─── RENDERERS ────────────────────────────────────────────────────────────────


def display_config(config: dict[str, Any]) -> None:
    """Renders a premium dashboard header with all run parameters."""
    console.rule("[bold cyan]RUN CONFIGURATION[/bold cyan]", style="cyan")

    # 1. Experiment Summary (Metadata & LLM)
    exp_cfg = config.get("experiment", {})
    llm_cfg = config.get("llm", {})

    metadata_table = Table(show_header=False, box=box.SIMPLE, border_style="dim")
    metadata_table.add_row(
        "Experiment", f"[bold cyan]{exp_cfg.get('name', 'N/A')}[/bold cyan]"
    )
    metadata_table.add_row(
        "Description", f"[dim]{exp_cfg.get('description', '')}[/dim]"
    )
    metadata_table.add_row(
        "LLM Model",
        f"[green]{llm_cfg.get('provider', 'N/A')}/{llm_cfg.get('model', 'N/A')}[/green]",
    )
    metadata_table.add_row(
        "Language", f"[yellow]{exp_cfg.get('language', 'python')}[/yellow]"
    )

    console.print(Panel(metadata_table, title="[bold]Metadata", border_style="dim"))

    # 2. Population Strategy (Detailed Panels)
    code_prof = config.get("code_profile", {})

    # Collect all profiles
    profiles_to_show = [("Code", code_prof)]

    # Check for test profiles
    test_profs_dict = config.get("test_profiles", {})
    for pt in ["unittest", "differential", "public"]:
        if f"{pt}_profile" in config:
            test_profs_dict[pt] = config[f"{pt}_profile"]

    for t_type, t_prof in test_profs_dict.items():
        profiles_to_show.append((t_type.title(), t_prof))

    panels = []
    for name, prof in profiles_to_show:
        cfg = prof.get("population_config", prof)

        # Base Settings
        settings_table = Table(show_header=False, box=None, padding=(0, 1))

        # Population
        init_size = cfg.get("initial_population_size")
        max_size = cfg.get("max_population_size")
        settings_table.add_row("Pop Sizes", f"[bold]{init_size} → {max_size}[/bold]")
        settings_table.add_row(
            "Init Prior", f"[dim]{cfg.get('initial_prior', 'N/A')}[/dim]"
        )

        elitism = cfg.get("elitism_rate")
        if elitism is not None:
            settings_table.add_row("Elitism", f"{elitism:.1%}")

        # Rates (Operations)
        ops_str = extract_operator_rates(prof)
        if ops_str:
            settings_table.add_row("Op Rates", f"[yellow]{ops_str}[/yellow]")

        # Strategy
        div = cfg.get("diversity_enabled")
        if div is not None:
            settings_table.add_row(
                "Diversity", "[green]ON[/green]" if div else "[red]OFF[/red]"
            )

        strat = cfg.get("prob_assigner_strategy")
        if strat:
            settings_table.add_row("Prob Strat", f"[dim]{strat}[/dim]")

        # Performance
        workers = cfg.get("llm_workers")
        if workers:
            settings_table.add_row("Workers", f"{workers}")

        batch = cfg.get("init_pop_batch_size")
        if batch:
            settings_table.add_row("Batch Size", f"{batch}")

        # Specific Settings (Bayesian/Special)
        for special in ["learning_rate", "alpha", "beta", "gamma", "k_failing_tests"]:
            val = cfg.get(special)
            if val is not None:
                lbl = special.replace("_", " ").title()
                settings_table.add_row(lbl, f"[cyan]{val}[/cyan]")

        panels.append(Panel(settings_table, title=f"[bold]{name}", border_style="blue"))

    console.print(Columns(panels, equal=True, expand=True))

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
            console.print("\n[bold]Seed Code Population[/bold]")
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

        # Group test snippets
        test_creations = [
            e
            for e in gen0_creations
            if not str(e.get("individual_id", "")).startswith("C")
        ]
        if test_creations:
            # Group by test_type
            by_type: dict[str, list[dict[str, Any]]] = {}
            for e in test_creations:
                tt = str(e.get("test_type", "Unknown")).upper()
                if tt not in by_type:
                    by_type[tt] = []
                by_type[tt].append(e)

            for tt, creations in by_type.items():
                console.print(f"\n[bold]Initial {tt} Population[/bold]")
                test_panels = []
                for e in creations:
                    snippet = e.get("snippet", "")
                    ind_id = e.get("individual_id", "?")
                    test_panels.append(
                        Panel(
                            Syntax(
                                snippet,
                                "python",
                                theme="ansi_dark",
                                line_numbers=True,
                                word_wrap=True,
                            ),
                            title=f"[bold]{fmt_id(ind_id)}[/bold]",
                            border_style="dim",
                        )
                    )
                console.print(Columns(test_panels, equal=True))

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





def display_population_overview(
    title: str,
    gen_events: list[dict[str, Any]],
    ind_metadata: dict[str, dict[str, Any]],
    latest_probs: dict[str, float],
) -> None:
    """Displays tables of active individuals at the start of the generation."""
    # Find all individuals being evaluated in this generation
    matrices = [e for e in gen_events if e["event_type"] == "OBSERVATION_MATRIX"]
    if not matrices:
        return

    code_ids = set()
    test_pops: dict[str, set[str]] = {}

    for m in matrices:
        tt = str(m.get("test_type", "Test")).upper()
        if tt in ("PRIVATE", "PUBLIC"):
            continue  # Skip fixed tests in overview

        for cid in m.get("code_ids", []):
            code_ids.add(cid)
        if tt not in test_pops:
            test_pops[tt] = set()
        for tid in m.get("test_ids", []):
            test_pops[tt].add(tid)

    if not code_ids and not test_pops:
        return

    # Render Code Pop
    if code_ids:
        table = Table(
            title=f"Code Population ({title})", box=box.SIMPLE, border_style="cyan"
        )
        table.add_column("ID", style="bold cyan")
        table.add_column("Created By", style="dim")
        table.add_column("Born", justify="center")
        table.add_column("Current Belief", justify="right")

        sorted_code = sorted(
            list(code_ids), key=lambda x: latest_probs.get(x, 0.0), reverse=True
        )
        for iid in sorted_code:
            meta = ind_metadata.get(iid, {})
            table.add_row(
                fmt_id(iid),
                str(meta.get("op", "initial")),
                str(meta.get("gen", 0)),
                f"{latest_probs.get(iid, 0.0):.4f}",
            )
        console.print(table)

    # Render Test Pops
    for tt, tids in test_pops.items():
        table = Table(
            title=f"{tt} Population ({title})", box=box.SIMPLE, border_style="magenta"
        )
        table.add_column("ID", style="bold magenta")
        table.add_column("Created By", style="dim")
        table.add_column("Born", justify="center")
        table.add_column("Current Belief", justify="right")

        sorted_tests = sorted(
            list(tids),
            key=lambda x: latest_probs.get(x, 1.0 if x.startswith("T") else 0.0),
            reverse=True,
        )
        for iid in sorted_tests:
            meta = ind_metadata.get(iid, {})
            # Fixed tests (starts with T and not in ind_metadata) default to initial/0
            table.add_row(
                fmt_id(iid),
                str(meta.get("op", "initial")),
                str(meta.get("gen", 0)),
                f"{latest_probs.get(iid, 1.0 if iid.startswith('T') else 0.0):.4f}",
            )
        console.print(table)
    console.print()


def render_cycle(
    events: list[dict[str, Any]],
    cycle_index: int,
    epoch_meta: dict[str, Any],
    code_gen: int,
    test_gen: int,
    ind_metadata: dict[str, dict[str, Any]],
    latest_probs: dict[str, float],
    show_errors: bool = False,
) -> None:
    phase_name = epoch_meta.get("phase_name", "Unknown")
    title_meta = f"Epoch {cycle_index} [{phase_name}] | Code Gen {code_gen} | Test Gen {test_gen}"
    console.rule(f"[bold yellow]{title_meta}[/bold yellow]", style="yellow")

    # 0. Population Overview
    display_population_overview(f"Gen {code_gen if epoch_meta.get('evolve_code') else code_gen}", events, ind_metadata, latest_probs)

    # 1. Matrices & Failures (Work)
    # Filter to show only the LATEST matrix for each test_type in this cycle
    all_matrices = [e for e in events if e["event_type"] == "OBSERVATION_MATRIX"]
    matrix_map = {}
    for m in all_matrices:
        tt = str(m.get("test_type", "Unknown")).upper()
        matrix_map[tt] = m

    matrices = list(matrix_map.values())
    failures = [e for e in events if e["event_type"] == "EVALUATION_FAILED"]

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
                        err = f_row.get("error_log", "Unknown Error")

                        console.print(
                            f"  [red]✖[/red] [bold]{c_id}[/bold] vs [bold]{t_id}[/bold]:"
                        )
                        console.print(
                            Panel(err, border_style="dim red", padding=(0, 1))
                        )
            elif type_failures:
                num_fails = len(type_failures)
                if num_fails > 0:
                    console.print(
                        f"[dim yellow italic]({num_fails} evaluation failures hidden. Use --show-errors to see details)[/dim yellow italic]"
                    )

    # 2. Belief Updates (Learning)
    belief_updates = [e for e in events if e["event_type"] == "BELIEF_UPDATE"]
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

    # 3. Succession
    # Show selection/breeding events present in this cycle
    display_succession_in_cycle(events)


def display_succession_in_cycle(events: list[dict[str, Any]]) -> None:
    elites = [e for e in events if e["event_type"] == "SELECTED_AS_ELITE"]
    newborns = [e for e in events if e["event_type"] == "CREATED" and e.get("creation_op") != "INITIAL"]

    if not elites and not newborns:
        return

    console.print("\n[bold underline yellow]SUCCESSION & BREEDING[/bold underline yellow]")

    if elites:
        code_elites = [e for e in elites if str(e.get("individual_id", "")).startswith("C")]
        test_elites = [e for e in elites if not str(e.get("individual_id", "")).startswith("C")]

        if code_elites:
            ids_str = ", ".join(fmt_id(str(e.get("individual_id"))) for e in code_elites)
            console.print(Panel(Text.from_markup(f"[bold cyan]Elite Code Kept:[/bold cyan] {ids_str}"), border_style="cyan", padding=(0, 1)))

        if test_elites:
            test_types = sorted(list(set(e.get("test_type", "Evolved Tests") for e in test_elites)))
            for tt in test_types:
                tt_elites = [e for e in test_elites if e.get("test_type", "Evolved Tests") == tt]
                ids_str = ", ".join(fmt_id(str(e.get("individual_id"))) for e in tt_elites)
                console.print(Panel(Text.from_markup(f"[bold magenta]Elite {tt.title()} Kept:[/bold magenta] {ids_str}"), border_style="magenta", padding=(0, 1)))

    if newborns:
        code_newborns = [n for n in newborns if str(n.get("individual_id", "")).startswith("C")]
        test_newborns = [n for n in newborns if not str(n.get("individual_id", "")).startswith("C")]

        if code_newborns:
            table = Table(title="Bred Code Offspring", box=box.ROUNDED, border_style="cyan")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Operator", style="magenta")
            table.add_column("Parents", style="dim")
            table.add_column("Initial Belief", justify="right")
            for row in code_newborns:
                parents_dict = row.get("parents", {})
                parents_list = parents_dict.get("code", []) + parents_dict.get("test", [])
                parents_str = ", ".join(map(fmt_id, parents_list)) if parents_list else "—"
                table.add_row(fmt_id(str(row.get("individual_id"))), str(row.get("creation_op")), parents_str, f"{row.get('probability', 0.0):.4f}")
            console.print(table)

        if test_newborns:
            test_types = sorted(list(set(n.get("test_type", "Evolved Tests") for n in test_newborns)))
            for tt in test_types:
                tt_newborns = [n for n in test_newborns if n.get("test_type", "Evolved Tests") == tt]
                table = Table(title=f"Bred {tt.title()} Offspring", box=box.ROUNDED, border_style="magenta")
                table.add_column("ID", style="magenta", no_wrap=True)
                table.add_column("Operator", style="magenta")
                table.add_column("Parents", style="dim")
                table.add_column("Initial Belief", justify="right")
                for row in tt_newborns:
                    parents_dict = row.get("parents", {})
                    parents_list = parents_dict.get("code", []) + parents_dict.get("test", [])
                    parents_str = ", ".join(map(fmt_id, parents_list)) if parents_list else "—"
                    table.add_row(fmt_id(str(row.get("individual_id"))), str(row.get("creation_op")), parents_str, f"{row.get('probability', 0.0):.4f}")
                console.print(table)


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

    # Metadata & snippets cache
    ind_metadata: dict[str, dict[str, Any]] = {}
    latest_probs: dict[str, float] = {}
    snippets: dict[str, str] = {}

    for e in events:
        iid = e.get("individual_id")
        if not iid:
            continue

        if e["event_type"] == "CREATED":
            ind_metadata[iid] = {
                "op": e.get("creation_op", "initial"),
                "gen": e.get("generation", 0),
            }
            if "snippet" in e:
                snippets[iid] = e["snippet"]

        prob = e.get("probability")
        if prob is not None:
            latest_probs[iid] = float(prob)
        elif iid.startswith("T") and "PUBLIC" in e.get("event_type", ""):
            # Fallback for older logs where public tests didn't have explicit probabilities
            latest_probs[iid] = 1.0

    # Phase 0: Configuration
    config = load_config(run_id)
    if config:
        display_config(config)

    # Phase 1: Initialization
    display_initialization(events)

    # Phase 2: Epoch-Centric Cycles
    config = load_config(run_id)
    if not config:
        console.print("[red]Error: Could not load run config for schedule reconstruction.[/red]")
        return

    expected_epochs = reconstruct_schedule(config)
    cycles = group_events_into_cycles(events)

    # Filter initialization events from Cycle 0 if it exists
    # Cycle 0 starts with CREATED(0) but initialization is handled separately
    # so we might need to skip or re-process.
    # Actually, Orchestrator logs a SUMMARY after init. So Cycle 0 is Init.

    code_gen = 0
    test_gen = 0

    for i, cycle_events in enumerate(cycles):
        if i == 0:
            continue # Initialization handled by display_initialization

        # Determine epoch meta (fallback if logs exceed config schedule)
        epoch_idx = i - 1
        meta = expected_epochs[epoch_idx] if epoch_idx < len(expected_epochs) else {"phase_name": "Final Evaluation", "evolve_code": False, "evolve_tests": False}

        render_cycle(
            cycle_events,
            epoch_idx,
            meta,
            code_gen,
            test_gen,
            ind_metadata,
            latest_probs,
            show_errors=show_errors
        )

        # Update generations after the cycle according to the schedule
        if meta.get("evolve_code"):
            code_gen += 1
        if meta.get("evolve_tests"):
            test_gen += 1

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
                # Use cached latest probability which is more reliable
                champion_prob = latest_probs.get(
                    champion_id, champion_meta.get("probability", 0.0)
                )

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
