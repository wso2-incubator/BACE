# src/coevolution/core/exceptions.py
"""
Language-agnostic exception hierarchy for the coevolution framework.

These exceptions are raised by ILanguage adapter implementations and caught
by the coevolution engine operators. Keeping them in core.exceptions (rather
than inside the language interface module) allows any module to import and
handle them without pulling in the full ILanguage protocol.
"""


class LanguageError(Exception):
    """Base exception for all language adapter failures."""

    pass


class LanguageParsingError(LanguageError):
    """Raised when code cannot be parsed by the language toolchain."""

    pass


class LanguageTransformationError(LanguageError):
    """Raised when code transformation or composition fails."""

    pass
