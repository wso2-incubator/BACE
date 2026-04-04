#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console

from coevolution.utils.paths import sanitize_id

app = typer.Typer(help="Extract champions from a coevolution run.")
console = Console()


def extract_champions_from_log(history_path: Path) -> Optional[Dict[str, Any]]:
    if not history_path.exists():
        return None

    latest_probs: Dict[str, float] = {}
    snippets: Dict[str, str] = {}
    generation_born: Dict[str, int] = {}
    survived_ids: List[str] = []

    with open(history_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                record = data.get("record", {})
                extra = record.get("extra", {})
                event_data = extra.get("event_data", {})
                message = record.get("message", "UNKNOWN")

                # Normalize message if it's from LIFECYCLE_EVENT
                event_type = message
                if message == "LIFECYCLE_EVENT":
                    event_type = event_data.get("event", "UNKNOWN").upper()

                if event_type == "SURVIVED" or event_type == "SELECTED_AS_ELITE":
                    iid = str(event_data.get("individual_id"))
                    if iid.startswith("C") and iid not in survived_ids:
                        survived_ids.append(iid)

                if event_type == "CREATED":
                    iid = str(event_data.get("individual_id"))
                    if iid.startswith("C"):
                        generation_born[iid] = event_data.get("generation", 0)
                        if "snippet" in event_data:
                            snippets[iid] = event_data["snippet"]

                elif message == "BELIEF_UPDATE":
                    ids = event_data.get("ids", [])
                    posterior = event_data.get("posterior", [])
                    for i, iid in enumerate(ids):
                        if i < len(posterior):
                            latest_probs[str(iid)] = float(posterior[i])

                # Probability can also be in CREATED or other events
                if "probability" in event_data and "individual_id" in event_data:
                    latest_probs[str(event_data["individual_id"])] = float(
                        event_data["probability"]
                    )

            except json.JSONDecodeError:
                continue

    if not survived_ids:
        # Fallback: if no explicit SURVIVED events, look for any code individuals in latest_probs
        survived_ids = [iid for iid in latest_probs.keys() if iid.startswith("C")]
        if not survived_ids:
            return None

    # Champion Selection logic:
    # 1. Max probability
    # 2. Min generation_born

    # Sort survivors: primary key probability (desc), secondary key generation (desc)
    survived_ids.sort(
        key=lambda iid: (latest_probs.get(iid, 0.0), generation_born.get(iid, 0)),
        reverse=True,
    )

    champion_id = survived_ids[0]
    return {
        "id": champion_id,
        "snippet": snippets.get(champion_id, ""),
        "probability": latest_probs.get(champion_id, 0.0),
        "generation": generation_born.get(champion_id, 0),
    }


@app.command()
def main(
    run_id: str = typer.Argument(..., help="Run ID of the experiment"),
    log_dir: Path = typer.Option(Path("logs"), help="Logs directory"),
) -> None:
    run_path = log_dir / sanitize_id(run_id)
    if not run_path.exists():
        console.print(f"[red]Error: Run path {run_path} does not exist.[/red]")
        raise typer.Exit(1)

    results_file = run_path / "results.jsonl"
    champions = []

    # Iterate through problem directories
    for problem_dir in sorted(run_path.iterdir()):
        if not problem_dir.is_dir():
            continue

        history_file = problem_dir / "evolutionary_history.jsonl"
        if history_file.exists():
            champion_info = extract_champions_from_log(history_file)
            if champion_info:
                champions.append(
                    {
                        "question_id": problem_dir.name,
                        "snippet": champion_info["snippet"],
                        "status": None,
                    }
                )
                console.print(
                    f"Extracted champion for {problem_dir.name}: {champion_info['id']} (p={champion_info['probability']:.4f}, gen={champion_info['generation']})"
                )
            else:
                console.print(
                    f"[yellow]Warning: No code champions found for {problem_dir.name}[/yellow]"
                )

    if champions:
        with open(results_file, "w") as f:
            for champ in champions:
                f.write(json.dumps(champ) + "\n")
        console.print(
            f"\n[bold green]Successfully saved {len(champions)} champions to {results_file}[/bold green]"
        )
    else:
        console.print("[red]No champions extracted from any problem.[/red]")


if __name__ == "__main__":
    app()
