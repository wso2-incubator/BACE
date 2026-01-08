# src/coevolution/utils/log_parser.py

import glob
import json
import os
import zipfile
from collections import defaultdict
from typing import Generator, TypedDict

import pandas as pd
from loguru import logger

# --- Interface Definitions ---


class MatrixRegistry(TypedDict):
    """
    Dynamic dictionary mapping test types to lists of DataFrames.
    Keys are dynamic (e.g., 'unittest', 'differential', 'future_test_type').
    """

    # We cannot strictly type dynamic keys in TypedDict,
    # but this documents the intent.
    pass


class ParsedLog(TypedDict):
    gen_stats: pd.DataFrame
    individuals: pd.DataFrame
    matrices: dict[str, list[pd.DataFrame]]


# --- The Stream Reader (IO Abstraction) ---


def _log_line_generator(
    log_dir: str, log_filename_pattern: str
) -> Generator[str, None, None]:
    """
    Yields lines one by one from all matching log files,
    abstracting away Zip/Text differences.
    """
    search_path = os.path.join(log_dir, log_filename_pattern)
    found_files = sorted(glob.glob(search_path))

    if not found_files:
        logger.warning(f"No log files found at {search_path}")
        return

    for file_path in found_files:
        logger.info(f"Streaming: {os.path.basename(file_path)}")
        try:
            if file_path.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as z:
                    for internal_name in z.namelist():
                        with z.open(internal_name) as binary_f:
                            for raw_line in binary_f:
                                yield raw_line.decode("utf-8")
            else:
                with open(file_path, "r", encoding="utf-8") as text_f:
                    for line in text_f:
                        yield line
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")


# --- The Parser (Logic) ---


def parse_coevolution_log(
    log_dir: str,
    log_filename_pattern: str,
    target_run_id: str,
    target_problem_id: str,
) -> ParsedLog:
    """
    Flexible parser that dynamically discovers matrix types.
    """

    # 1. Dynamic Containers
    gen_data = []
    ind_data = []
    # defaultdict handles any new test type automatically
    matrix_store: dict[str, list[pd.DataFrame]] = defaultdict(list)

    # 2. Stream Process
    for line_str in _log_line_generator(log_dir, log_filename_pattern):
        if not line_str.strip():
            continue

        try:
            # --- Base JSON Parsing ---
            try:
                log_entry = json.loads(line_str)
            except json.JSONDecodeError:
                continue  # Skip partial lines

            record = log_entry.get("record", {})
            extra = record.get("extra", {})

            # --- Filtering ---
            # Only filter if target IDs are provided
            if extra.get("run_id") != target_run_id:
                continue
            if extra.get("problem_id") != target_problem_id:
                continue

            message = record.get("message")
            if not message or not isinstance(message, str):
                continue

            # --- Routing ---
            # Expected format: HEADER | [OPTIONAL_ID] | JSON_PAYLOAD
            parts = [p.strip() for p in message.split("|", 2)]
            header = parts[0]

            # Case 1: Generation Summaries
            if header == "GEN_SUMMARY":
                if len(parts) >= 2:
                    gen_data.append(json.loads(parts[-1]))

            # Case 2: Individual Lifecycle (Dynamic Status)
            # Matches "CODE_INDIVIDUAL_DIED", "TEST_INDIVIDUAL_SURVIVED", etc.
            elif "_INDIVIDUAL_" in header:
                payload_str = parts[-1]
                data = json.loads(payload_str)

                # Parse Header: "{TYPE}_INDIVIDUAL_{STATUS}"
                # e.g., "CODE_INDIVIDUAL_DIED" -> type="CODE", status="DIED"
                header_parts = header.split("_INDIVIDUAL_")
                if len(header_parts) == 2:
                    data["type"] = header_parts[0].lower()  # code/test/unittest
                    data["status"] = header_parts[1]  # DIED/SURVIVED

                ind_data.append(data)

            # Case 3: Matrices (Fully Dynamic)
            # Matches "{TYPE} serialized"
            elif header.endswith(" serialized"):
                # "UNITTEST serialized" -> matrix_type = "unittest"
                # "PROPERTY serialized" -> matrix_type = "property"
                matrix_type = header.split(" ")[0].lower()

                payload_str = parts[-1]
                matrix_dict = json.loads(payload_str)

                df = pd.DataFrame.from_dict(matrix_dict)
                df.index.name = "Code"

                # AUTO-DISCOVERY: Just append to the list for this type
                matrix_store[matrix_type].append(df)

        except Exception:
            # Don't crash the whole parse on one bad line
            continue

    # --- Formatting ---
    logger.info("Constructing DataFrames...")

    # Generation Stats
    gen_df = pd.DataFrame(gen_data)
    if not gen_df.empty and "generation" in gen_df.columns:
        gen_df = gen_df.set_index("generation").sort_index()

    # Individual Stats
    ind_df = pd.DataFrame(ind_data)
    if not ind_df.empty:
        # Fill global context if available in data, else fill from targets
        if "run_id" not in ind_df.columns and target_run_id:
            ind_df["run_id"] = target_run_id

    logger.success(f"Parsing Complete. Found matrix types: {list(matrix_store.keys())}")

    return {
        "gen_stats": gen_df,
        "individuals": ind_df,
        "matrices": dict(matrix_store),  # Convert defaultdict back to dict
    }


def get_problem_ids(log_dir: str, log_filename_pattern: str, run_id: str) -> set[str]:
    """
    Scans log files to extract all unique problem IDs present.
    """
    problem_ids = set()

    for line_str in _log_line_generator(log_dir, log_filename_pattern):
        if not line_str.strip():
            continue

        try:
            log_entry = json.loads(line_str)
        except json.JSONDecodeError:
            continue  # Skip partial lines

        record = log_entry.get("record", {})
        extra = record.get("extra", {})

        if extra.get("run_id") != run_id:
            continue

        pid = extra.get("problem_id")
        if pid and isinstance(pid, str):
            problem_ids.add(pid)

    return problem_ids


def get_run_ids(log_dir: str, log_filename_pattern: str) -> set[str]:
    """
    Scans log files to extract all unique run IDs present.
    """
    run_ids = set()

    for line_str in _log_line_generator(log_dir, log_filename_pattern):
        if not line_str.strip():
            continue

        try:
            log_entry = json.loads(line_str)
        except json.JSONDecodeError:
            continue  # Skip partial lines

        record = log_entry.get("record", {})
        extra = record.get("extra", {})

        rid = extra.get("run_id")
        if rid and isinstance(rid, str):
            run_ids.add(rid)

    return run_ids
