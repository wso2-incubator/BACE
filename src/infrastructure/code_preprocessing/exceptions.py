"""Custom exceptions for code preprocessing operations."""


class CodeProcessingError(Exception):
    """Base exception for code processing errors."""

    pass


class CodeParsingError(CodeProcessingError):
    """Raised when code cannot be parsed due to syntax errors."""

    pass


class CodeTransformationError(CodeProcessingError):
    """Raised when code transformation/generation fails."""

    pass
