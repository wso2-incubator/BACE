"""Factory function for creating LLM clients."""

from typing import Any, Optional

from loguru import logger

from .base import LLMClient
from .ollama import OllamaClient
from .openai import OpenAIChatClient, OpenAIClient


def create_llm_client(
    provider: str,
    model: str,
    max_output_tokens: Optional[int] = None,
    enable_token_limit: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    **kwargs: Any,
) -> LLMClient:
    """Create an LLM client with the specified provider and configuration.

    Args:
        provider: The LLM provider ('openai', 'openai-chat', or 'ollama').
        model: The model identifier.
        max_output_tokens: Maximum number of output tokens allowed (default: 1M).
        enable_token_limit: Whether to enforce the token limit. If None, uses provider defaults:
                          - OpenAI clients: True (limits enabled by default)
                          - Ollama clients: False (limits disabled by default)
        reasoning_effort: The reasoning effort level for OpenAI Response API models.
                         Common values: 'minimal', 'low', 'medium', 'high'.
                         Defaults to 'minimal'. Only applies to 'openai' provider.
        **kwargs: Additional provider-specific arguments.

    Returns:
        Configured LLM client instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = provider.lower()

    logger.info(f"Creating LLM client: provider={provider}, model={model}")

    # Extract special parameters from kwargs to pass to constructors
    client_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ["max_output_tokens", "enable_token_limit", "reasoning_effort"]
    }

    client: LLMClient
    if provider == "openai":
        # OpenAI default: token limits enabled
        limit_enabled = True if enable_token_limit is None else enable_token_limit
        client = OpenAIClient(
            model, max_output_tokens, limit_enabled, reasoning_effort, **client_kwargs
        )
    elif provider == "openai-chat":
        # OpenAI Chat default: token limits enabled
        limit_enabled = True if enable_token_limit is None else enable_token_limit
        client = OpenAIChatClient(
            model, max_output_tokens, limit_enabled, **client_kwargs
        )
    elif provider == "ollama":
        # Ollama default: token limits disabled
        limit_enabled = False if enable_token_limit is None else enable_token_limit
        client = OllamaClient(model, max_output_tokens, limit_enabled, **client_kwargs)
    elif provider == "gemini":
        from .gemini import GeminiLLMClient

        # Gemini default: token limits enabled
        limit_enabled = True if enable_token_limit is None else enable_token_limit
        client = GeminiLLMClient(
            model, max_output_tokens, limit_enabled, reasoning_effort, **client_kwargs
        )
    elif provider == "transformers":
        from .transformers import TransformersClient

        # Transformers default: token limits disabled
        limit_enabled = False if enable_token_limit is None else enable_token_limit
        client = TransformersClient(
            model, max_output_tokens, limit_enabled, **client_kwargs
        )
    else:
        logger.error(f"Unsupported provider: {provider}")
        raise ValueError(f"Unsupported provider: {provider}")

    logger.info(f"Successfully created {client.__class__.__name__}")
    return client
