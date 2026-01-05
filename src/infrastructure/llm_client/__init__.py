"""
LLM Client Module

This module provides a unified interface for different LLM providers.

Structure:
- base: Abstract base class and core functionality
- exceptions: Custom exceptions for LLM operations
- openai: OpenAI-specific implementations (Chat and Response APIs)
- ollama: Ollama-specific implementation
- factory: Factory function for creating LLM clients

Usage:
    from infrastructure.llm_client import create_llm_client, LLMClient

    # Create a client using the factory
    client = create_llm_client(
        provider="openai",
        model="gpt-4",
        max_output_tokens=100000,
        enable_token_limit=True
    )

    # Or use specific implementations
    from infrastructure.llm_client import OpenAIClient, OllamaClient
"""

from .base import LLMClient
from .exceptions import TokenLimitExceededError
from .factory import create_llm_client
from .ollama import OllamaClient
from .openai import OpenAIChatClient, OpenAIClient

__all__ = [
    # Base classes
    "LLMClient",
    # Exceptions
    "TokenLimitExceededError",
    # Provider implementations
    "OpenAIClient",
    "OpenAIChatClient",
    "OllamaClient",
    # Factory
    "create_llm_client",
]
