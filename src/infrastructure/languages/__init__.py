"""Language adapter factory and registration."""

from loguru import logger

from coevolution.core.interfaces.language import ILanguage

from .ballerina import BallerinaLanguage
from .python import PythonLanguage


def create_language_adapter(language: str = "python") -> ILanguage:
    """
    Factory function to create the appropriate language adapter based on the language name.

    Args:
        language: Name of the programming language (e.g., "python", "ballerina")

    Returns:
        An instance of ILanguage for the specified language

    Raises:
        ValueError: If the language is not supported
    """
    logger.info(f"Creating language adapter for: {language}")
    lang_lower = language.lower()
    if lang_lower == "python":
        return PythonLanguage()
    elif language == "ballerina":
        return BallerinaLanguage()
    else:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages: python, ballerina."
        )


__all__ = [
    "create_language_adapter",
    "PythonLanguage",
    "BallerinaLanguage",
]
