"""
Logging utilities for the coevolution project.

Provides a standardized setup function for Loguru and helper functions
for logging formatted section headers.

Functions:
    setup_logging: Configures console and file logging with context support.
    log_section_header: Logs a prominent "==== SECTION ====" style header.
    log_subsection_header: Logs a "---- Subsection ----" style header.
"""

import sys

from loguru import logger


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
        rotation="100 MB",
        retention="10 days",
        compression="zip",
        enqueue=True,  # Makes logging safe for multiprocessing
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
