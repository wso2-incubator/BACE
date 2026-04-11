"""Custom exceptions for sandbox operations."""


class CodeExecutionTimeoutError(Exception):
    """Raised when code execution times out."""

    pass


class CodeExecutionError(Exception):
    """Raised when code execution fails."""

    pass
