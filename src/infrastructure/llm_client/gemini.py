"""Gemini LLM client implementation using native Google GenAI SDK."""

import os
from typing import Any, Optional

from google import genai
from google.genai import types
from loguru import logger

from .base import LLMClient


class GeminiLLMClient(LLMClient):
    """Gemini Client using the native Google GenAI SDK.

    Reference: https://ai.google.dev/gemini-api/docs/quickstart
    """

    def __init__(
        self,
        model: str,
        max_output_tokens: Optional[int] = None,
        enable_token_limit: bool = True,
        reasoning_effort: str | None = "minimal",
        **kwargs: Any,
    ) -> None:
        """Initialize Gemini client.

        Args:
            model: The Gemini model identifier (e.g., 'gemini-2.0-flash').
            max_output_tokens: Maximum tokens allowed.
            enable_token_limit: Whether to enforce token limits.
            reasoning_effort: The reasoning effort level (thinking level).
                             Common values: 'minimal', 'low', 'medium', 'high'.
                             Mapped to Gemini's thinking_level.
            **kwargs: Additional arguments passed to the genai.Client.
        """
        super().__init__(model, max_output_tokens, enable_token_limit)

        # Use api_key from kwargs if provided, otherwise from environment
        api_key = kwargs.pop("api_key", os.environ.get("GEMINI_API_KEY"))
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not found in environment and no api_key provided!"
            )

        self.client = genai.Client(api_key=api_key, **kwargs)
        self.reasoning_effort = reasoning_effort

        logger.debug(
            f"Initialized GeminiLLMClient with model: {model}, "
            f"reasoning_effort: {reasoning_effort}"
        )

    def _map_reasoning_effort(self, effort: str | None) -> types.ThinkingLevel:
        """Map generic reasoning effort to Gemini thinking level.

        Args:
            effort: The reasoning effort level.

        Returns:
            The corresponding Gemini thinking level.
        """
        if not effort:
            return types.ThinkingLevel.MINIMAL

        # Standard mapping
        mapping = {
            "minimal": types.ThinkingLevel.MINIMAL,
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
        }
        return mapping.get(effort.lower(), types.ThinkingLevel.MINIMAL)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate content using Gemini native API.

        Args:
            prompt: User prompt content.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text content.
        """
        # Extract reasoning_effort if provided, otherwise use default
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)
        thinking_level = self._map_reasoning_effort(reasoning_effort)

        logger.debug(
            f"GeminiLLMClient generating with model: {self.model}, "
            f"thinking_level: {thinking_level}"
        )
        logger.trace(f"Prompt (first 200 chars): {prompt[:200]}...")

        # Configure thinking if enabled
        thinking_config = None
        if reasoning_effort and reasoning_effort != "none":
            thinking_config = types.ThinkingConfig(thinking_level=thinking_level)

        # Build generation config
        gen_config = types.GenerateContentConfig(
            thinking_config=thinking_config,
            # Pass max_output_tokens if specified (conversion might be needed if they naming is different)
            # max_output_tokens=self.max_output_tokens,
            **kwargs,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            tools=None,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=gen_config,
        )

        content = response.text
        if content is None:
            logger.error("Gemini response text is None")
            raise ValueError("Gemini response text is None")

        result = str(content)
        logger.debug(f"Generated {len(result)} characters")
        logger.trace(f"Response (first 200 chars): {result[:200]}...")

        # Extract usage metadata for accurate token tracking
        usage = response.usage_metadata
        if usage:
            # Input tokens
            self._add_input_tokens(usage.prompt_token_count or 0)

            # Output tokens include candidates (final answer) and thoughts (for thinking models)
            output_tokens = (usage.candidates_token_count or 0) + (
                usage.thoughts_token_count or 0
            )

            logger.debug(
                f"Gemini usage: prompt={usage.prompt_token_count}, "
                f"candidates={usage.candidates_token_count}, "
                f"thoughts={usage.thoughts_token_count}, "
                f"total={usage.total_token_count}"
            )
            self._add_output_tokens(output_tokens)
        else:
            # Fallback to estimation if usage_metadata is missing
            self._add_input_tokens(self._estimate_tokens(prompt))
            self._add_output_tokens(self._estimate_tokens(result))

        return result
