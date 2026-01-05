"""Ollama LLM client implementation."""

from typing import Any, Optional

from loguru import logger

from .base import LLMClient


class OllamaClient(LLMClient):
    """Ollama LLM client implementation."""

    def __init__(
        self,
        model: str,
        max_output_tokens: Optional[int] = None,
        enable_token_limit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, max_output_tokens, enable_token_limit)
        import ollama

        self.ollama = ollama
        logger.debug(f"Initialized OllamaClient with model: {model}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        logger.debug(f"OllamaClient generating with model: {self.model}")
        logger.trace(f"Prompt (first 200 chars): {prompt[:200]}...")

        response = self.ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response["message"]["content"]
        if not isinstance(content, str):
            logger.error(f"Expected string content, got {type(content)}")
            raise ValueError(f"Expected string content, got {type(content)}")

        logger.debug(f"Generated {len(content)} characters")
        logger.trace(f"Response (first 200 chars): {content[:200]}...")
        self._add_output_tokens(self._estimate_tokens(content))
        return content
