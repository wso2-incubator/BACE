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
    log_dir: str, log_filename_pattern: str, limit_to_files: list[str] | None = None
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
                            for raw_line in binary_f:
                                yield file_path, raw_line.decode("utf-8")
            else:
                with open(file_path, "r", encoding="utf-8") as text_f:
                    for line in text_f:
                        yield file_path, line
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")


# --- Specialized Parsers ---

class LegacyLogParser:
    """Parses old-style flat-file .log files."""
    
    def parse(self, log_dir: str, file_pattern: str, run_id: str, problem_id: str, limit_to_files: list[str] | None = None) -> ParsedLog:
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

    def _finalize(self, gen_data: list[dict[str, Any]], ind_data: list[dict[str, Any]], matrix_store: dict[str, list[pd.DataFrame]], run_id: str) -> ParsedLog:
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
        from coevolution.utils.paths import sanitize_problem_id
        sanitized_pid = sanitize_problem_id(problem_id)
        history_path = Path(log_dir) / run_id / sanitized_pid / "evolutionary_history.jsonl"
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

    def _finalize(self, gen_data: list[dict[str, Any]], ind_data: list[dict[str, Any]], matrix_store: dict[str, list[pd.DataFrame]], run_id: str) -> ParsedLog:
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
    target_run_id: str | None = None,
    target_problem_id: str | None = None,
    use_legacy: bool = False,
    legacy_files: list[str] | None = None,
) -> ParsedLog:
    """
    Unified entry point. Automatically detects if a run is Structured (New)
    or Legacy (Flat) and dispatches to the correct parser.
    """
    if not target_run_id:
        raise ValueError("target_run_id is required.")

    # 1. Detection: Structured (New) Format
    # If problem_id is provided, check the specific path
    if target_problem_id:
        from coevolution.utils.paths import sanitize_problem_id
        sanitized_pid = sanitize_problem_id(target_problem_id)
        structured_path = Path(log_dir) / target_run_id / sanitized_pid / "evolutionary_history.jsonl"
        if structured_path.exists():
            logger.info(f"Detected structured log format for {target_run_id}/{target_problem_id}")
            return StructuredJSONLParser().parse(log_dir, target_run_id, target_problem_id)

    # 2. Discovery for New format
    if not use_legacy or target_problem_id is None:
        run_path = Path(log_dir) / target_run_id
        if run_path.exists() and run_path.is_dir():
            # Find all subfolders with evolutionary_history.jsonl
            pids = []
            for p_dir in run_path.iterdir():
                if p_dir.is_dir() and (p_dir / "evolutionary_history.jsonl").exists():
                    pids.append(p_dir.name)
            
            if pids:
                actual_pid = target_problem_id if target_problem_id in pids else sorted(pids)[0]
                logger.info(f"Using structured log format for {target_run_id}/{actual_pid}")
                return StructuredJSONLParser().parse(log_dir, target_run_id, actual_pid)

    # 3. Fallback to Legacy discovery (only if requested)
    if use_legacy:
        if target_problem_id is None:
            problem_ids, current_legacy_files = get_problem_ids(log_dir, log_filename_pattern, target_run_id, use_legacy=True)
            if not problem_ids:
                return {"gen_stats": pd.DataFrame(), "individuals": pd.DataFrame(), "matrices": {}}
            target_problem_id = sorted(list(problem_ids))[0]
            if legacy_files is None:
                legacy_files = current_legacy_files
            
        logger.info(f"Falling back to legacy flat log parser for {target_run_id}/{target_problem_id}")
        return LegacyLogParser().parse(log_dir, log_filename_pattern, target_run_id, target_problem_id or "", limit_to_files=legacy_files)

    return {"gen_stats": pd.DataFrame(), "individuals": pd.DataFrame(), "matrices": {}}


# --- Helper Utilities (Shared) ---

def get_problem_ids(
    log_dir: str, 
    log_filename_pattern: str, 
    run_id: str, 
    use_legacy: bool = False
) -> tuple[set[str], list[str]]:
    """
    Scans for problem IDs.
    Returns: (Set of problem_ids, List of legacy files containing the run_id)
    """
    problem_ids = set()
    legacy_run_files = set()

    # 1. Structured Scan (Fast)
    run_path = Path(log_dir) / run_id
    if run_path.exists() and run_path.is_dir():
        from coevolution.utils.paths import sanitize_problem_id
        for p_dir in run_path.iterdir():
            if p_dir.is_dir() and (p_dir / "evolutionary_history.jsonl").exists():
                # Note: Directory name is sanitized version, but the logs inside 
                # might still reference the original ID. However, for discovery 
                # we return the directory name which is used as target_problem_id.
                problem_ids.add(p_dir.name)

    # 2. Legacy Scan (Slow - only if requested)
    if use_legacy:
        for fpath, line_str in _log_line_generator(log_dir, log_filename_pattern):
            try:
                log_entry = json.loads(line_str)
                extra = log_entry.get("record", {}).get("extra", {})
                if extra.get("run_id") == run_id:
                    legacy_run_files.add(fpath)
                    pid = extra.get("problem_id")
                    if pid: problem_ids.add(pid)
            except Exception: continue

    return problem_ids, sorted(list(legacy_run_files))


def get_run_ids(
    log_dir: str, 
    log_filename_pattern: str, 
    use_legacy: bool = False
) -> set[str]:
    """Scans for run IDs in structured directories and optionally legacy logs."""
    run_ids = set()

    # 1. Structured Scan (Fast)
    root_path = Path(log_dir)
    if root_path.exists():
        for r_dir in root_path.iterdir():
            if r_dir.is_dir() and (r_dir / "run_config.json").exists():
                run_ids.add(r_dir.name)

    # 2. Legacy Scan (Slow - only if requested)
    if use_legacy:
        for fpath, line_str in _log_line_generator(log_dir, log_filename_pattern):
            try:
                log_entry = json.loads(line_str)
                rid = log_entry.get("record", {}).get("extra", {}).get("run_id")
                if rid: run_ids.add(rid)
            except Exception: continue

    return run_ids
