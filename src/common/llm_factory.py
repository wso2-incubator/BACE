"""
LLM Factory Module for APR Experiments

This module provides factory functions and utilities for creating and managing
different types of Language Model instances across APR experiments.
"""

from typing import TYPE_CHECKING, Literal

from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:
    from .config import BaseConfig

# Import LLM libraries conditionally to handle missing dependencies gracefully
try:
    from langchain_ollama.chat_models import ChatOllama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None


def create_llm_instance(config: "BaseConfig") -> BaseChatModel:
    """
    Factory function to create the appropriate LLM instance based on provider configuration.

    Leverages LangGraph's unified interface where both ChatOllama and ChatOpenAI
    inherit from BaseChatModel, providing consistent .invoke(), .stream(), and .bind_tools() methods.

    Args:
        config: BaseConfig containing provider and model settings

    Returns:
        BaseChatModel instance (ChatOllama or ChatOpenAI)

    Raises:
        ValueError: If unsupported provider or missing configuration
        ImportError: If required packages are not installed
    """
    if config.llm_provider == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "langchain_ollama is not installed. "
                "Install it with: pip install langchain-ollama"
            )
        return ChatOllama(model=config.llm_model)

    elif config.llm_provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "langchain_openai is not installed. "
                "Install it with: pip install langchain-openai"
            )
        return ChatOpenAI(model=config.llm_model)

    else:
        raise ValueError(
            f"Unsupported provider: {config.llm_provider}. "
            f"Supported providers: 'ollama', 'openai'"
        )


def create_llm_from_params(
    provider: Literal["ollama", "openai"], model: str, **kwargs
) -> BaseChatModel:
    """
    Create LLM instance directly from parameters without config object.

    Args:
        provider: LLM provider ("ollama" or "openai")
        model: Model name/identifier
        **kwargs: Additional parameters to pass to the LLM constructor

    Returns:
        BaseChatModel instance

    Raises:
        ValueError: If unsupported provider
        ImportError: If required packages are not installed
    """
    if provider == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "langchain_ollama is not installed. "
                "Install it with: pip install langchain-ollama"
            )
        return ChatOllama(model=model, **kwargs)

    elif provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "langchain_openai is not installed. "
                "Install it with: pip install langchain-openai"
            )
        return ChatOpenAI(model=model, **kwargs)

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: 'ollama', 'openai'"
        )


def get_available_providers() -> list[str]:
    """
    Get list of available LLM providers based on installed packages.

    Returns:
        List of available provider names
    """
    available = []
    if OLLAMA_AVAILABLE:
        available.append("ollama")
    if OPENAI_AVAILABLE:
        available.append("openai")
    return available


def check_provider_availability(provider: str) -> bool:
    """
    Check if a specific provider is available.

    Args:
        provider: Provider name to check

    Returns:
        True if provider is available, False otherwise
    """
    if provider == "ollama":
        return OLLAMA_AVAILABLE
    elif provider == "openai":
        return OPENAI_AVAILABLE
    else:
        return False


def validate_provider_model_combination(provider: str, model: str) -> None:
    """
    Validate that a provider and model combination is supported.

    Args:
        provider: LLM provider name
        model: Model name/identifier

    Raises:
        ValueError: If combination is not supported
        ImportError: If required packages are not installed
    """
    if not check_provider_availability(provider):
        available_providers = get_available_providers()
        if not available_providers:
            raise ImportError(
                "No LLM providers are available. Install langchain-ollama "
                "and/or langchain-openai to use LLM functionality."
            )
        raise ImportError(
            f"Provider '{provider}' is not available. "
            f"Available providers: {', '.join(available_providers)}"
        )

    # Provider-specific model validation could be added here
    if provider == "openai" and not model:
        raise ValueError("OpenAI provider requires a model name")

    if provider == "ollama" and not model:
        raise ValueError("Ollama provider requires a model name")
