# src/coevolution/analysis/log_parser.py

import glob
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Generator, TypedDict

import pandas as pd
from loguru import logger

# --- Interface Definitions ---

class ParsedLog(TypedDict):
    gen_stats: pd.DataFrame
    individuals: pd.DataFrame
    matrices: dict[str, list[pd.DataFrame]]

# --- The Stream Reader (IO Abstraction for Legacy) ---

def _log_line_generator(
    log_dir: str, log_filename_pattern: str, limit_to_files: list[str] = None, max_lines: int | None = None
) -> Generator[tuple[str, str], None, None]:
    """
    Yields (file_path, line) one by one from matching log files.
    - if limit_to_files is provided, only those files are read.
    - otherwise, globs the log_dir with log_filename_pattern.
    """
    if limit_to_files:
        found_files = limit_to_files
    else:
        search_path = os.path.join(log_dir, log_filename_pattern)
        found_files = sorted(glob.glob(search_path), key=lambda x: os.path.getmtime(x))

    if not found_files:
        if not limit_to_files:
            logger.warning(f"No log files found at {os.path.join(log_dir, log_filename_pattern)}")
        return

    for file_path in found_files:
        try:
            if file_path.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as z:
                    for internal_name in z.namelist():
                        with z.open(internal_name) as binary_f:
                            for i, raw_line in enumerate(binary_f):
                                if max_lines is not None and i >= max_lines:
                                    break
                                yield file_path, raw_line.decode("utf-8")
            else:
                with open(file_path, "r", encoding="utf-8") as text_f:
                    for i, line in enumerate(text_f):
                        if max_lines is not None and i >= max_lines:
                            break
                        yield file_path, line
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")


# --- Specialized Parsers ---

class LegacyLogParser:
    """Parses old-style flat-file .log files."""
    
    def parse(self, log_dir: str, file_pattern: str, run_id: str, problem_id: str, limit_to_files: list[str] = None) -> ParsedLog:
        gen_data = []
        ind_data = []
        matrix_store: dict[str, list[pd.DataFrame]] = defaultdict(list)

        for fpath, line_str in _log_line_generator(log_dir, file_pattern, limit_to_files=limit_to_files):
            if not line_str.strip():
                continue

            try:
                log_entry = json.loads(line_str)
                record = log_entry.get("record", {})
                extra = record.get("extra", {})

                if extra.get("run_id") != run_id or extra.get("problem_id") != problem_id:
                    continue

                message = record.get("message", "")
                parts = [p.strip() for p in message.split("|", 2)]
                header = parts[0]

                if header == "GEN_SUMMARY":
                    if len(parts) >= 2:
                        gen_data.append(json.loads(parts[-1]))
                elif "_INDIVIDUAL_" in header:
                    data = json.loads(parts[-1])
                    header_parts = header.split("_INDIVIDUAL_")
                    if len(header_parts) == 2:
                        data["type"] = header_parts[0].lower()
                        data["status"] = header_parts[1].lower() # survivied/died
                    ind_data.append(data)
                elif header.endswith(" serialized"):
                    matrix_type = header.split(" ")[0].lower()
                    matrix_dict = json.loads(parts[-1])
                    df = pd.DataFrame.from_dict(matrix_dict)
                    df.index.name = "Code"
                    matrix_store[matrix_type].append(df)

            except Exception:
                continue

        return self._finalize(gen_data, ind_data, matrix_store, run_id)

    def _finalize(self, gen_data, ind_data, matrix_store, run_id) -> ParsedLog:
        gen_df = pd.DataFrame(gen_data)
        if not gen_df.empty and "generation" in gen_df.columns:
            gen_df = gen_df.set_index("generation").sort_index()

        ind_df = pd.DataFrame(ind_data)
        if not ind_df.empty and "run_id" not in ind_df.columns:
            ind_df["run_id"] = run_id

        return {
            "gen_stats": gen_df,
            "individuals": ind_df,
            "matrices": dict(matrix_store),
        }


class StructuredJSONLParser:
    """Parses new-style directory-based JSONL telemetry."""
    
    def parse(self, log_dir: str, run_id: str, problem_id: str) -> ParsedLog:
        history_path = Path(log_dir) / run_id / problem_id / "evolutionary_history.jsonl"
        if not history_path.exists():
            return {"gen_stats": pd.DataFrame(), "individuals": pd.DataFrame(), "matrices": {}}

        gen_data = []
        ind_data = [] # standard lifecycle
        matrix_store: dict[str, list[pd.DataFrame]] = defaultdict(list)

        with open(history_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    record = data.get("record", {})
                    extra = record.get("extra", {})
                    event_data = extra.get("event_data", {})
                    message = record.get("message", "")
                    
                    if message == "LIFECYCLE_EVENT":
                        event_type = event_data.get("event", "UNKNOWN").upper()
                        # Map to legacy-compatible fields
                        row = {**event_data, "id": event_data.get("individual_id")}
                        
                        # Type detection
                        if str(row["id"]).startswith("C"):
                            row["type"] = "code"
                        else:
                            row["type"] = "test"
                        
                        # Status mapping for overview.py compatibility
                        if event_type in ("SELECTED_AS_ELITE", "SURVIVED"):
                            row["status"] = "survived"
                        elif event_type == "DIED":
                            row["status"] = "died"
                        else:
                            row["status"] = event_type.lower()
                            
                        ind_data.append(row)
                    
                    elif message == "BELIEF_UPDATE":
                        # BELIEF_UPDATE contains lists of ids and posterior values
                        ids = event_data.get("ids", [])
                        posteriors = event_data.get("posterior", [])
                        population = event_data.get("population", "code")
                        
                        if ids and posteriors and len(ids) == len(posteriors):
                            for iid, prob in zip(ids, posteriors):
                                row = {
                                    "id": iid,
                                    "probability": prob,
                                    "generation": event_data.get("generation"),
                                    "type": population,
                                    "status": "belief_update"
                                }
                                ind_data.append(row)
                        
                    elif message == "GENERATION_SUMMARY":
                        gen_data.append(event_data)
                        
                    elif message == "OBSERVATION_MATRIX":
                        matrix_type = event_data.get("test_type", "unknown").lower()
                        matrix_dict = event_data.get("matrix", {})
                        code_ids = event_data.get("code_ids", [])
                        
                        if matrix_dict:
                            df = pd.DataFrame.from_dict(matrix_dict)
                            # If we have explicit code_ids, use them as index
                            if code_ids and len(code_ids) == len(df):
                                df.index = code_ids
                            df.index.name = "Code"
                            matrix_store[matrix_type].append(df)
                            
                except Exception:
                    continue

        return self._finalize(gen_data, ind_data, matrix_store, run_id)

    def _finalize(self, gen_data, ind_data, matrix_store, run_id) -> ParsedLog:
        gen_df = pd.DataFrame(gen_data)
        if not gen_df.empty and "generation" in gen_df.columns:
            gen_df = gen_df.set_index("generation").sort_index()

        ind_df = pd.DataFrame(ind_data)
        if not ind_df.empty and "run_id" not in ind_df.columns:
            ind_df["run_id"] = run_id

        return {
            "gen_stats": gen_df,
            "individuals": ind_df,
            "matrices": dict(matrix_store),
        }


# --- Public API (The Dispatcher) ---

def parse_coevolution_log(
    log_dir: str = "logs",
    log_filename_pattern: str = "*.log",
    target_run_id: str = None,
    target_problem_id: str | None = None,
    source_type: str | None = None, # "structured" or "legacy"
    legacy_files: list[str] | None = None,
) -> ParsedLog:
    """
    Unified entry point. Parses a specific run/problem from a given source.
    """
    if not target_run_id or not target_problem_id:
        raise ValueError("target_run_id and target_problem_id are required.")

    # 1. Selection: If source_type is explicit, use it
    if source_type == "structured":
        return StructuredJSONLParser().parse(log_dir, target_run_id, target_problem_id)
    elif source_type == "legacy":
        return LegacyLogParser().parse(log_dir, log_filename_pattern, target_run_id, target_problem_id, limit_to_files=legacy_files)

    # 2. Auto-Discovery (Fallback)
    structured_path = Path(log_dir) / target_run_id / target_problem_id / "evolutionary_history.jsonl"
    if structured_path.exists():
        return StructuredJSONLParser().parse(log_dir, target_run_id, target_problem_id)
    
    return LegacyLogParser().parse(log_dir, log_filename_pattern, target_run_id, target_problem_id, limit_to_files=legacy_files)


# --- Helper Utilities (Shared) ---

def get_problem_ids(
    log_dir: str, 
    log_filename_pattern: str, 
    run_id: str
) -> dict[str, list[dict]]:
    """
    Scans for problem IDs across both structured and legacy sources.
    Returns: { problem_id: [{"run_id": str, "type": "structured"|"legacy", "files": [str]}] }
    """
    results = defaultdict(list)
    HeadReadCount = 10

    # 1. Structured Scan (Fast)
    run_path = Path(log_dir) / run_id
    if run_path.exists() and run_path.is_dir():
        for p_dir in run_path.iterdir():
            if p_dir.is_dir() and (p_dir / "evolutionary_history.jsonl").exists():
                results[p_dir.name].append({
                    "run_id": run_id,
                    "type": "structured",
                    "files": [str(p_dir / "evolutionary_history.jsonl")]
                })

    # 2. Legacy Scan (Slow)
    legacy_run_files = set()
    legacy_problem_map = defaultdict(set)
    
    for fpath, line_str in _log_line_generator(log_dir, log_filename_pattern, max_lines=HeadReadCount):
        try:
            log_entry = json.loads(line_str)
            extra = log_entry.get("record", {}).get("extra", {})
            if extra.get("run_id") == run_id:
                legacy_run_files.add(fpath)
                pid = extra.get("problem_id")
                if pid:
                    legacy_problem_map[pid].add(fpath)
        except Exception: continue

    # Merge legacy results
    for pid, files in legacy_problem_map.items():
        results[pid].append({
            "run_id": run_id,
            "type": "legacy",
            "files": sorted(list(files)) # The specific files for THIS problem
        })

    return dict(results)


def get_run_ids(
    log_dir: str, 
    log_filename_pattern: str
) -> set[str]:
    """Scans for run IDs in both structured directories and legacy logs."""
    run_ids = set()
    HeadReadCount = 10

    # 1. Structured Scan (Fast)
    root_path = Path(log_dir)
    if root_path.exists():
        for r_dir in root_path.iterdir():
            if r_dir.is_dir() and (r_dir / "run_config.json").exists():
                run_ids.add(r_dir.name)

    # 2. Legacy Scan (Slow)
    for fpath, line_str in _log_line_generator(log_dir, log_filename_pattern, max_lines=HeadReadCount):
        try:
            log_entry = json.loads(line_str)
            rid = log_entry.get("record", {}).get("extra", {}).get("run_id")
            if rid: run_ids.add(rid)
        except Exception: continue

    return run_ids
