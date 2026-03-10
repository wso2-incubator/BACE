# src/coevolution/analysis/engines.py
from typing import Any

def reconstruct_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the evolution schedule into a list of epoch definitions."""
    schedule_data = config.get("evolution_config", {}).get("schedule", {})
    phases = schedule_data.get("phases", [])
    epochs = []
    for p in phases:
        duration = p.get("duration", 0)
        # Handle string durations if any
        if isinstance(duration, str):
            try:
                duration = int(duration)
            except ValueError:
                duration = 0
                
        for _ in range(duration):
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


def get_active_ids_in_cycle(cycle_events: list[dict[str, Any]]) -> dict[str, set[str]]:
    """
    Extracts all code and test IDs that were active in a specific interaction cycle.
    Returns: {"code": {id1, id2}, "test_types": {"UNITTEST": {id3, id4}, ...}}
    """
    code_ids = set()
    test_pops: dict[str, set[str]] = {}

    matrices = [e for e in cycle_events if e["event_type"] == "OBSERVATION_MATRIX"]
    for m in matrices:
        tt = str(m.get("test_type", "Test")).upper()
        if tt in ("PRIVATE", "PUBLIC"):
            continue 

        for cid in m.get("code_ids", []):
            code_ids.add(cid)
            
        if tt not in test_pops:
            test_pops[tt] = set()
        for tid in m.get("test_ids", []):
            test_pops[tt].add(tid)

    return {
        "code": code_ids,
        "test_pops": test_pops
    }
