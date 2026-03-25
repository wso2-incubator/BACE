"""Gemini LLM client implementation using OpenAI-compatible API."""

import os
from typing import Any, Optional

from loguru import logger

from .openai import OpenAIChatClient


class GeminiLLMClient(OpenAIChatClient):
    """Gemini Client using the OpenAI-compatible Chat Completions API.

    This client wraps the OpenAI SDK but points to the Gemini API endpoint.
    Reference: https://ai.google.dev/gemini-api/docs/openai
    """

    def __init__(
        self,
        model: str,
        max_output_tokens: Optional[int] = None,
        enable_token_limit: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Gemini client.

        Args:
            model: The Gemini model identifier (e.g., 'gemini-1.5-flash').
            max_output_tokens: Maximum tokens allowed.
            enable_token_limit: Whether to enforce token limits.
            **kwargs: Additional arguments passed to the OpenAI client.
        """
        # Hardcode the Gemini OpenAI-compatible base URL
        kwargs["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
        kwargs["api_key"] = os.environ["GEMINI_API_KEY"]

        super().__init__(
            model=model,
            max_output_tokens=max_output_tokens,
            enable_token_limit=enable_token_limit,
            **kwargs,
        )
        logger.debug(f"Initialized GeminiLLMClient with model: {model}")
        logger.debug(f"Initialized GeminiLLMClient with model: {model}")
