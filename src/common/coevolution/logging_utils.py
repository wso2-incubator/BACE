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

import sys
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from common.coevolution.core.individual import CodeIndividual, TestIndividual
    from common.coevolution.core.population import CodePopulation, TestPopulation


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
    setup_gen_log: bool = False,
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

    gen_log_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "[{extra[run_id]}:{extra[problem_id]}] | "
        "{message}"
    )

    # Define a filter to exclude generation logs from the main log
    def exclude_gen_logs(record: Any) -> bool:
        return "GEN_LOG" not in record["extra"]

    # Add the console logger
    logger.add(
        sys.stderr,
        level=console_level.upper(),
        format=console_format,
        colorize=True,
        filter=exclude_gen_logs,
    )

    # Add the file logger
    log_file_path = f"logs/{log_file_base_name}_{{time:YYYYMMDD}}.log"
    logger.add(
        log_file_path,
        level=file_level.upper(),
        format=file_format,
        rotation="100 MB",
        retention="10 days",
        compression="zip",
        enqueue=True,  # Makes logging safe for multiprocessing
        filter=exclude_gen_logs,
    )

    if setup_gen_log:
        gen_log_file = f"logs/generations/{log_file_base_name}_{{time:YYYYMMDD}}.log"
        logger.info(f"Adding generation log sink: {gen_log_file}")
        logger.add(
            gen_log_file,
            level="INFO",
            filter=lambda record: "GEN_LOG" in record["extra"],
            format=gen_log_format,
            enqueue=True,
            rotation="10 MB",
            compression="zip",
        )

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
    gen_logger: Any,
    code_population: "CodePopulation",
    test_population: "TestPopulation",
) -> None:
    """
    Logs lightweight summary statistics for the current generation.
    This logs aggregate information without full individual details,
    reducing redundancy since individuals are logged once when their lifecycle ends.

    Args:
        gen_logger: The logger bound with GEN_LOG=True for generation logging
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

    logger.debug(f"Logging generation {gen_num} summary to generation log.")
    gen_logger.info(f"--- Generation {gen_num} Summary ---")
    gen_logger.info(f"GEN_SUMMARY|{gen_num}|{json.dumps(summary)}")


def log_individual_complete(
    gen_logger: Any,
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
    gen_logger.info(f"INDIVIDUAL_{status}|{individual.id}|{json.dumps(record)}")
    logger.debug(f"Logged complete record for {individual.id} with status {status}")


def log_final_survivors(
    gen_logger: Any,
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
    logger.info("Logging final survivors to generation log...")
    gen_logger.info("=" * 80)
    gen_logger.info("--- FINAL SURVIVORS ---")
    gen_logger.info("=" * 80)

    logger.debug(f"Logging {len(code_population)} code survivors")
    code_ind: "CodeIndividual"
    for code_ind in code_population:
        log_individual_complete(gen_logger, code_ind, "SURVIVED")

    logger.debug(f"Logging {len(test_population)} test survivors")
    test_ind: "TestIndividual"
    for test_ind in test_population:
        log_individual_complete(gen_logger, test_ind, "SURVIVED")

    logger.info(
        f"Logged {len(code_population)} code and {len(test_population)} test survivors"
    )
