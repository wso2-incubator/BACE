"""
Logging utilities for the coevolution project.

Provides a standardized setup function for Loguru and helper functions
for logging formatted section headers and individual/generation logging.

Functions:
    setup_logging: Configures console and file logging with context support.
    log_section_header: Logs a prominent "==== SECTION ====" style header.
    log_subsection_header: Logs a "---- Subsection ----" style header.
    log_generation_summary: Logs lightweight generation statistics.
    log_individual_complete: Logs a single individual's complete lifecycle.
    log_final_survivors: Logs all surviving individuals at evolution end.
"""

import glob
import json
import os
import sys
import zipfile
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    import numpy as np

    from common.coevolution.core.individual import CodeIndividual, TestIndividual
    from common.coevolution.core.interfaces import Problem
    from common.coevolution.core.population import CodePopulation, TestPopulation


class ParsedLog(TypedDict):
    gen_stats: pd.DataFrame
    individuals: pd.DataFrame
    matrices: dict[str, list[pd.DataFrame]]


def get_generation_logger() -> Any:
    """
    Returns a logger bound with GEN_LOG=True for generation-specific logging.

    This logger is filtered to write to a separate generation log file,
    allowing separation of detailed evolution tracking from general application logs.

    Returns:
        A Loguru logger instance bound with GEN_LOG context.
    """
    return logger.bind(GEN_LOG=True)


def setup_logging(
    console_level: str = "INFO",
    file_level: str = "TRACE",
    log_file_base_name: str = "coevolution_run",
) -> None:
    """
    Configures Loguru handlers for console and file logging.

    This function removes the default handler and adds two new ones:
    1. A console (stderr) handler with customizable level and color.
    2. A rotating file handler with customizable level, trace context,
       and support for multiprocessing (enqueue=True).

    Both handlers are formatted to include 'run_id' and 'problem_id'
    from the logger's `extra` context.

    Args:
        console_level: The minimum log level to display on the console (e.g., "DEBUG").
        file_level: The minimum log level to write to the file (e.g., "TRACE").
        log_file_base_name: The base name for the log file.
    """
    logger.remove()  # Remove the default, unconfigured handler

    # Define a default context for logs outside a specific run/problem
    default_context = {"run_id": "GLOBAL", "problem_id": "SETUP"}

    # Define format for the console (more concise)
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>[{extra[problem_id]: <8}]</cyan> | "  # Use extra context
        "<cyan>{name}:{function}:{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Define format for the file (more detailed)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "[{extra[run_id]}:{extra[problem_id]}] | "
        "{process}:{thread} | "
        "{name}:{function}:{line} - {message}"
    )

    # Add the console logger
    logger.add(
        sys.stderr,
        level=console_level.upper(),
        format=console_format,
        colorize=True,
    )

    # Add the file logger
    log_file_path = f"logs/{log_file_base_name}_{{time:YYYYMMDD}}.log"
    logger.add(
        log_file_path,
        level=file_level.upper(),
        format=file_format,
        rotation="1000 MB",
        retention="365 days",
        compression="zip",
        enqueue=True,  # Makes logging safe for multiprocessing
        serialize=True,  # JSON format for easier parsing
    )

    # Configure the extra context BEFORE any logging that uses it
    logger.configure(extra=default_context)
    logger.info(
        f"Logging configured. Console level: {console_level}, File level: {file_level}."
    )
    logger.info(f"Detailed logs will be written to {log_file_path}")


def log_section_header(level: str, message: str) -> None:
    """
    Logs a prominent, centered section header surrounded by '='.

    Args:
        message: The text to display in the header.
        level: The log level to use (e.g., "INFO", "DEBUG").
    """
    width = 80
    logger.log(level, "=" * width)
    logger.log(level, f" {message.upper()} ".center(width, "="))
    logger.log(level, "=" * width)


def log_subsection_header(level: str, message: str) -> None:
    """
    Logs a centered subsection header surrounded by '-'.

    Args:
        message: The text to display in the header.
        level: The log level to use (e.g., "INFO", "DEBUG").
    """
    width = 80
    logger.log(level, f" {message} ".center(width, "-"))


def log_generation_summary(
    code_population: "CodePopulation",
    test_population: "TestPopulation",
) -> None:
    """
    Logs lightweight summary statistics for the current generation.
    This logs aggregate information without full individual details,
    reducing redundancy since individuals are logged once when their lifecycle ends.

    Args:
        code_population: The current code population
        test_population: The current test population
    """
    import json

    gen_num = code_population.generation

    # Collect newly born individuals (created this generation)
    new_code_ids = [ind.id for ind in code_population if ind.generation_born == gen_num]
    new_test_ids = [ind.id for ind in test_population if ind.generation_born == gen_num]

    # Calculate statistics
    code_probs = [ind.probability for ind in code_population]
    test_probs = [ind.probability for ind in test_population]

    summary = {
        "generation": gen_num,
        "code_pop_size": len(code_population),
        "test_pop_size": len(test_population),
        "avg_code_prob": round(sum(code_probs) / len(code_probs), 4),
        "avg_test_prob": round(sum(test_probs) / len(test_probs), 4),
        "min_code_prob": round(min(code_probs), 4),
        "max_code_prob": round(max(code_probs), 4),
        "min_test_prob": round(min(test_probs), 4),
        "max_test_prob": round(max(test_probs), 4),
        "new_code_count": len(new_code_ids),
        "new_test_count": len(new_test_ids),
        "new_code_ids": new_code_ids,
        "new_test_ids": new_test_ids,
    }

    log_subsection_header("INFO", f"--- Generation {gen_num} Summary ---")
    logger.info(f"GEN_SUMMARY|{gen_num}|{json.dumps(summary)}")


def log_individual_complete(
    individual: "CodeIndividual | TestIndividual",
    status: str,
) -> None:
    """
    Logs a single individual's complete lifecycle record.
    Called when individual dies (removed from population) or survives (final generation).

    Args:
        gen_logger: The logger bound with GEN_LOG=True for generation logging
        individual: The individual to log
        status: Either "DIED" or "SURVIVED"
    """
    import json

    # Get complete record from individual (includes all lifecycle events)
    record = individual.get_complete_record()

    for key, value in record.items():
        # Round float values to 4 decimal places for cleaner logs
        if isinstance(value, float):
            record[key] = round(value, 4)

    # Add status to the record
    record["status"] = status

    # Log with structured format for easy parsing
    logger.trace(f"INDIVIDUAL_{status}|{individual.id}|{json.dumps(record)}")
    logger.debug(f"Logged complete record for {individual.id} with status {status}")


def log_final_survivors(
    code_population: "CodePopulation",
    test_population: "TestPopulation",
) -> None:
    """
    Logs all individuals that survived to the final generation.
    Called at the end of run() after evolution loop completes.

    Args:
        gen_logger: The logger bound with GEN_LOG=True for generation logging
        code_population: The final code population
        test_population: The final test population
    """
    log_section_header("INFO", "Final Survivors")

    logger.debug(f"Logging {len(code_population)} code survivors")
    code_ind: "CodeIndividual"
    for code_ind in code_population:
        log_individual_complete(code_ind, "SURVIVED")

    logger.debug(f"Logging {len(test_population)} test survivors")
    test_ind: "TestIndividual"
    for test_ind in test_population:
        log_individual_complete(test_ind, "SURVIVED")

    logger.info(
        f"Logged {len(code_population)} code and {len(test_population)} test survivors"
    )


def log_belief_update_start(
    population_type: str, num_items: int, num_observations: int
) -> None:
    """
    Log the start of a belief update operation.

    Args:
        population_type: Either "code" or "test"
        num_items: Number of items in the population being updated
        num_observations: Number of observations used for the update
    """
    logger.info(
        f"Updating {population_type} beliefs: {num_items} {population_type}s x "
        f"{num_observations} observations"
    )


def log_prior_statistics(population_type: str, probs: "np.ndarray") -> None:
    """
    Log statistics about prior beliefs.

    Args:
        population_type: Either "code" or "test"
        probs: Array of prior probabilities
    """
    import numpy as np

    mean_prob = np.mean(probs)
    logger.debug(
        f"Prior {population_type} beliefs: mean={mean_prob:.4f}, "
        f"min={np.min(probs):.4f}, "
        f"max={np.max(probs):.4f}"
    )


def log_posterior_statistics(
    population_type: str, prior_probs: "np.ndarray", posterior_probs: "np.ndarray"
) -> None:
    """
    Log statistics about posterior beliefs and the change from prior.

    Args:
        population_type: Either "code" or "test"
        prior_probs: Array of prior probabilities
        posterior_probs: Array of posterior probabilities
    """
    import numpy as np

    prior_mean = np.mean(prior_probs)
    posterior_mean = np.mean(posterior_probs)
    delta = posterior_mean - prior_mean

    logger.debug(
        f"Posterior {population_type} beliefs: mean={posterior_mean:.4f}, "
        f"min={np.min(posterior_probs):.4f}, "
        f"max={np.max(posterior_probs):.4f}"
    )
    logger.info(
        f"{population_type.capitalize()} belief update complete: avg Δ={delta:+.4f}"
    )


def log_belief_changes(
    population_type: str, prior_probs: "np.ndarray", posterior_probs: "np.ndarray"
) -> None:
    """
    Log detailed statistics about individual belief changes at trace level.

    Args:
        population_type: Either "code" or "test"
        prior_probs: Array of prior probabilities
        posterior_probs: Array of posterior probabilities
    """
    import numpy as np

    deltas = posterior_probs - prior_probs
    logger.trace(
        f"{population_type.capitalize()} belief changes: mean={np.mean(deltas):+.4f}, "
        f"std={np.std(deltas):.4f}, "
        f"max_increase={np.max(deltas):+.4f}, "
        f"max_decrease={np.min(deltas):+.4f}"
    )


def _compute_pass_rates(matrix: "np.ndarray") -> "np.ndarray":
    """
    Compute the pass rate for each row in the matrix.

    Args:
        matrix: Binary numpy array, 1 if passed, else 0

    Returns:
        1D numpy array of pass rates for each row (fraction of columns that are 1)
    """
    import numpy as np

    num_rows, num_cols = matrix.shape

    if num_cols == 0:
        logger.warning(
            f"Matrix has 0 columns. Returning zero pass rates for {num_rows} rows."
        )
        return np.zeros(num_rows, dtype=float)

    pass_rates = np.sum(matrix, axis=1) / float(num_cols)
    return np.asarray(pass_rates)


def _compute_test_discriminations(observation_matrix: "np.ndarray") -> "np.ndarray":
    """
    Compute discrimination for each test (column) in the observation matrix.

    Discrimination measures how well a test separates good from bad code using entropy.
    Uses the entropy of the test pass rate:
    - High entropy (near 1.0): test clearly distinguishes correct from incorrect code (pass rate near 0.5)
    - Low entropy (near 0.0): test doesn't discriminate (all pass or all fail, pass rate near 0 or 1)

    Formula: entropy = -p*log2(p) - (1-p)*log2(1-p) where p is the test pass rate

    Args:
        observation_matrix: Binary numpy array (codes x tests), 1 if code passed test, else 0

    Returns:
        1D numpy array of discrimination values for each test (entropy of pass rate)
    """
    import numpy as np

    num_codes, num_tests = observation_matrix.shape

    if num_codes == 0 or num_tests == 0:
        logger.warning(
            f"Cannot compute discrimination for empty matrix ({num_codes}, {num_tests})"
        )
        return np.zeros(num_tests, dtype=float)

    # Compute pass rate for each test (fraction of codes that pass)
    test_pass_rates = _compute_pass_rates(observation_matrix.T)

    # Compute binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
    # Handle edge cases where p=0 or p=1 (entropy should be 0)
    eps = 1e-10  # Small epsilon to avoid log(0)
    p = np.clip(test_pass_rates, eps, 1 - eps)

    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    return np.asarray(entropy)


def log_code_pass_rates(observation_matrix: "np.ndarray") -> None:
    """
    Log statistics about code pass rates.

    Args:
        observation_matrix: Binary numpy array (codes x tests), 1 if code passed test, else 0
    """
    import numpy as np

    code_pass_rates = _compute_pass_rates(observation_matrix)

    if len(code_pass_rates) == 0:
        logger.warning("No codes to compute pass rates for")
        return

    logger.trace(f"Code pass rates: {code_pass_rates}")
    logger.info(
        f"Code pass rates: mean={np.mean(code_pass_rates):.3f}, "
        f"min={np.min(code_pass_rates):.3f}, "
        f"max={np.max(code_pass_rates):.3f}, "
        f"std={np.std(code_pass_rates):.3f}"
    )

    # Log distribution
    num_perfect = np.sum(code_pass_rates == 1.0)
    num_zero = np.sum(code_pass_rates == 0.0)
    logger.debug(
        f"Code distribution: {num_perfect} perfect (100%), {num_zero} failed all (0%)"
    )


def log_test_pass_rates(observation_matrix: "np.ndarray") -> None:
    """
    Log statistics about test pass rates.

    Args:
        observation_matrix: Binary numpy array (codes x tests), 1 if code passed test, else 0
    """
    import numpy as np

    # Transpose to compute pass rates for tests (columns become rows)
    test_pass_rates = _compute_pass_rates(observation_matrix.T)

    if len(test_pass_rates) == 0:
        logger.warning("No tests to compute pass rates for")
        return
    logger.trace(f"Test pass rates: {test_pass_rates}")
    logger.info(
        f"Test pass rates: mean={np.mean(test_pass_rates):.3f}, "
        f"min={np.min(test_pass_rates):.3f}, "
        f"max={np.max(test_pass_rates):.3f}, "
        f"std={np.std(test_pass_rates):.3f}"
    )

    # Log distribution
    num_all_pass = np.sum(test_pass_rates == 1.0)
    num_all_fail = np.sum(test_pass_rates == 0.0)
    logger.debug(
        f"Test distribution: {num_all_pass} all codes pass, "
        f"{num_all_fail} all codes fail"
    )


def log_test_discriminations(observation_matrix: "np.ndarray") -> None:
    """
    Log statistics about test discrimination values.

    Args:
        observation_matrix: Binary numpy array (codes x tests), 1 if code passed test, else 0
    """
    import numpy as np

    test_discriminations = _compute_test_discriminations(observation_matrix)

    if len(test_discriminations) == 0:
        logger.warning("No tests to compute discriminations for")
        return

    logger.trace(f"Test discriminations: {test_discriminations}")
    logger.info(
        f"Test discriminations: mean={np.mean(test_discriminations):.3f}, "
        f"min={np.min(test_discriminations):.3f}, "
        f"max={np.max(test_discriminations):.3f}"
    )

    # Identify highly discriminating tests
    high_disc_threshold = 0.4  # Tests with std > 0.4 are good discriminators
    num_good_tests = np.sum(test_discriminations > high_disc_threshold)
    num_tests = len(test_discriminations)
    logger.debug(
        f"Highly discriminating tests (disc > {high_disc_threshold}): "
        f"{num_good_tests}/{num_tests} ({100 * num_good_tests / num_tests:.1f}%)"
    )


def _log_observation_matrix_statistics(observation_matrix: "np.ndarray") -> None:
    """
    Log comprehensive statistics about the observation matrix.

    This includes:
    - Matrix dimensions and sparsity
    - Code pass rates (how many tests each code passes)
    - Test pass rates (how many codes pass each test)
    - Test discriminations (how well each test separates codes)

    Args:
        observation_matrix: Binary numpy array (codes x tests), 1 if code passed test, else 0
    """
    import numpy as np

    num_codes, num_tests = observation_matrix.shape
    total_cells = num_codes * num_tests

    if total_cells == 0:
        logger.warning("Observation matrix is empty")
        return

    num_passes = np.sum(observation_matrix)
    sparsity = 1.0 - (num_passes / total_cells)

    logger.info(
        f"Observation Matrix: {num_codes} codes × {num_tests} tests "
        f"= {total_cells} evaluations"
    )
    logger.info(
        f"Total passes: {num_passes}/{total_cells} ({100 * num_passes / total_cells:.1f}%), "
        f"sparsity: {sparsity:.3f}"
    )

    # Use individual logging functions
    log_code_pass_rates(observation_matrix)
    log_test_pass_rates(observation_matrix)
    log_test_discriminations(observation_matrix)


def log_observation_matrix(
    observation_matrix: "np.ndarray",
    code_population: "CodePopulation",
    test_population: "TestPopulation",
    test_type: Literal["generated", "public", "private"] = "generated",
) -> None:
    code_ids = [ind.id for ind in code_population]
    test_ids = [ind.id for ind in test_population]

    # --- New Pandas Method ---

    # 1. Create the DataFrame
    df = pd.DataFrame(observation_matrix, index=code_ids, columns=test_ids)

    # 2. Give the index column a name for a cleaner header
    df.index.name = "Code\\Test"

    logger.info(f"Logging {test_type.upper()} observation matrix ({df.shape})")

    # 3. Log the entire pre-formatted string.
    #    Add a newline to ensure it starts on its own line.
    logger.debug(f"\n{df.to_string()}")
    logger.trace(f"{test_type.upper()} serialized | {df.to_json()}")

    # --- End New Method ---

    _log_observation_matrix_statistics(observation_matrix)
    return


def log_problem(problem: "Problem") -> None:
    logger.info(f"Loaded problem: {problem.question_title}")
    logger.info(f"Problem ID: {problem.question_id}")
    logger.info(f"Public tests: {len(problem.public_test_cases)}")
    logger.info(f"Private tests: {len(problem.private_test_cases)}")
    logger.debug(f"Problem content:\n{problem.question_content}")
    logger.debug(f"Starter code:\n{problem.starter_code}")


# logging parsing and analysis
def parse_complete_coevolution_log(
    log_dir: str, log_filename_pattern: str, target_run_id: str, target_problem_id: str
) -> ParsedLog:
    """
    Performs a SINGLE PASS over all log files to extract:
    1. Generation Summaries
    2. Individual Lifecycles
    3. Observation Matrices (Generated, Public, Private)

    Returns:
        A dictionary containing:
        - 'gen_stats': DataFrame
        - 'individuals': DataFrame
        - 'matrices': {
            'generated': list[pd.DataFrame],
            'public': list[pd.DataFrame],
            'private': list[pd.DataFrame]
          }
    """

    # --- Data Containers ---
    data_store: dict[str, list[Any]] = {
        "gen_data": [],
        "ind_data": [],
        "mat_generated": [],
        "mat_public": [],
        "mat_private": [],
    }

    # --- File Discovery ---
    search_path = os.path.join(log_dir, log_filename_pattern)
    found_files = sorted(glob.glob(search_path))

    if not found_files:
        logger.error(f"No files found for pattern: {search_path}")
        return {
            "gen_stats": pd.DataFrame(),
            "individuals": pd.DataFrame(),
            "matrices": {"generated": [], "public": [], "private": []},
        }

    logger.info(
        f"Starting unified parse on {len(found_files)} files for Run={target_run_id}"
    )

    # --- The Logic Processor ---
    def process_line(line_str: str) -> None:
        if not line_str.strip():
            return

        try:
            # 1. Base JSON Parse
            log_entry = json.loads(line_str)
            record = log_entry.get("record", {})
            extra = record.get("extra", {})

            # 2. Fast Filter (Check IDs immediately)
            if (
                extra.get("run_id") != target_run_id
                or extra.get("problem_id") != target_problem_id
            ):
                return

            # 3. Message Extraction
            message = record.get("message")
            if not message or not isinstance(message, str):
                return

            # 4. The Routing Logic
            # We split by pipe '|'. Maxsplit=2 covers all current cases.
            # Case A (Stats): KEY | ID | JSON  -> [KEY, ID, JSON]
            # Case B (Stats): KEY | JSON       -> [KEY, JSON]
            # Case C (Matrix): KEY | JSON      -> [KEY, JSON]
            parts = [p.strip() for p in message.split("|", 2)]
            if not parts:
                return

            header = parts[0]

            # === ROUTE: GENERATION SUMMARIES ===
            if header == "GEN_SUMMARY":
                if len(parts) >= 2:
                    data_store["gen_data"].append(json.loads(parts[-1]))

            # === ROUTE: INDIVIDUAL LIFECYCLES ===
            elif header in ("INDIVIDUAL_DIED", "INDIVIDUAL_SURVIVED"):
                payload_str = parts[-1]
                data = json.loads(payload_str)
                data["status"] = header.split("_")[-1]  # DIED or SURVIVED

                # If ID is in the middle token (KEY | ID | JSON), capture it
                if len(parts) >= 3 and parts[1] and "id" not in data:
                    data["id"] = parts[1]

                data_store["ind_data"].append(data)

            # === ROUTE: MATRICES ===
            elif header.endswith(" serialized"):
                # header is like "GENERATED serialized", "PUBLIC serialized"
                matrix_type = header.split(" ")[0]  # Grab "GENERATED"
                payload_str = parts[-1]

                matrix_dict = json.loads(payload_str)
                df = pd.DataFrame.from_dict(matrix_dict)
                df.index.name = "Code"

                if matrix_type == "GENERATED":
                    data_store["mat_generated"].append(df)
                elif matrix_type == "PUBLIC":
                    data_store["mat_public"].append(df)
                elif matrix_type == "PRIVATE":
                    data_store["mat_private"].append(df)

        except json.JSONDecodeError:
            pass  # Skip partial lines
        except Exception:
            # Optional: Log specific errors if needed, but keep parsing
            pass

    # --- The File Loop (Single Pass) ---
    for file_path in found_files:
        logger.info(f"Scanning: {os.path.basename(file_path)}")
        try:
            if file_path.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as z:
                    for internal_name in z.namelist():
                        with z.open(internal_name) as binary_f:
                            for raw_line in binary_f:
                                process_line(raw_line.decode("utf-8"))
            else:
                with open(file_path, "r", encoding="utf-8") as text_f:
                    for line in text_f:
                        process_line(line)
        except Exception:
            logger.error(f"Failed to read {file_path}")

    # --- Final Formatting ---
    logger.info("Constructing DataFrames...")

    # 1. Generation Stats
    gen_df = pd.DataFrame(data_store["gen_data"])
    if not gen_df.empty and "generation" in gen_df.columns:
        gen_df = gen_df.set_index("generation").sort_index()

    # 2. Individual Stats
    ind_df = pd.DataFrame(data_store["ind_data"])
    if not ind_df.empty:
        ind_df["run_id"] = target_run_id
        ind_df["problem_id"] = target_problem_id

    logger.success("Parsing Complete.")

    return {
        "gen_stats": gen_df,
        "individuals": ind_df,
        "matrices": {
            "generated": data_store["mat_generated"],
            "public": data_store["mat_public"],
            "private": data_store["mat_private"],
        },
    }
