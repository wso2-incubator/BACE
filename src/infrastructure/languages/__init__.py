"""Language adapter factory and registration."""

from .python import PythonLanguageAdapter
from .ballerina import BallerinaLanguageAdapter
from coevolution.core.interfaces.language import ILanguageAdapter

def create_language_adapter(language: str = "python") -> ILanguageAdapter:
    """
    Factory function to create the appropriate language adapter based on the language name.
    
    Args:
        language: Name of the programming language (e.g., "python", "ballerina")
        
    Returns:
        An instance of ILanguageAdapter for the specified language
        
    Raises:
        ValueError: If the language is not supported
    """
    lang_lower = language.lower()
    if lang_lower == "python":
        return PythonLanguageAdapter()
    elif lang_lower == "ballerina":
        return BallerinaLanguageAdapter()
    else:
        raise ValueError(f"Unsupported language: {language}. Supported languages: python, ballerina.")

__all__ = ["create_language_adapter", "PythonLanguageAdapter", "BallerinaLanguageAdapter"]
