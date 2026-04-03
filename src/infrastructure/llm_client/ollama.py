"""Ollama LLM client implementation."""

from typing import Any, Optional

from loguru import logger

from .base import LLMClient
from .exceptions import LLMInputFormatError


class OllamaClient(LLMClient):
    """Ollama LLM client implementation."""

    def __init__(
        self,
        model: str,
        max_output_tokens: Optional[int] = None,
        enable_token_limit: bool = False,
        workers: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, max_output_tokens, enable_token_limit, workers=workers)
        import ollama

        self.ollama = ollama
        logger.debug(f"Initialized OllamaClient with model: {model}")

    def generate(self, prompt: Any, **kwargs: Any) -> str:
        logger.debug(f"OllamaClient generating with model: {self.model}")
        logger.trace(f"Prompt (first 200 chars): {prompt[:200]}...")

        if isinstance(prompt, list):
            response = self.ollama.chat(
                model=self.model,
                messages=prompt,
                **kwargs,
            )
        elif isinstance(prompt, str):
            response = self.ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        else:
            logger.error(f"Unsupported prompt type: {type(prompt)}")
            raise LLMInputFormatError(f"Unsupported prompt type: {type(prompt)}")

        content = response["message"]["content"]
        if not isinstance(content, str):
            logger.error(f"Expected string content, got {type(content)}")
            raise ValueError(f"Expected string content, got {type(content)}")

        logger.debug(f"Generated {len(content)} characters")
        logger.trace(f"Response (first 200 chars): {content[:200]}...")

        # Track usage
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        if input_tokens > 0:
            self._add_input_tokens(input_tokens)
        else:
            # Fallback to estimation
            prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
            self._add_input_tokens(self._estimate_tokens(prompt_str))

        if output_tokens > 0:
            self._add_output_tokens(output_tokens)
        else:
            # Fallback to estimation
            self._add_output_tokens(self._estimate_tokens(content))

        return content
