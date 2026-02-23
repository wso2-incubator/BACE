"""Transformers LLM client implementation."""

from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from .base import LLMClient
from .exceptions import LLMInputFormatError


class TransformersClient(LLMClient):
    """Transformers LLM client implementation."""

    pipe: Callable[..., Any]

    def __init__(
        self,
        model: str,
        max_output_tokens: Optional[int] = None,
        enable_token_limit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, max_output_tokens, enable_token_limit)
        from transformers import pipeline

        # Type as Callable[..., Any] to keep mypy satisfied for the dynamic
        # pipeline object from the transformers library.
        self.pipe = pipeline("text-generation", model=model)
        logger.debug(f"Initialized TransformersClient pipe with model: {model}")

    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any) -> str:
        logger.debug(f"TransformersClient generating with model: {self.model}")
        if isinstance(prompt, str):
            logger.trace(f"Prompt (first 200 chars): {prompt[:200]}...")
        elif isinstance(prompt, list):
            logger.trace(f"Prompt is a list with length: {len(prompt)}")
        else:
            logger.trace(f"Prompt type: {type(prompt)}")

        if isinstance(prompt, list):
            messages: List[Dict[str, str]] = prompt
        elif isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            logger.error(f"Unsupported prompt type: {type(prompt)}")
            raise LLMInputFormatError(f"Unsupported prompt type: {type(prompt)}")

        response: Any = self.pipe(messages, **kwargs)
        content: str = response[0]["generated_text"][-1]["content"]

        logger.debug(f"Generated {len(content)} characters")
        logger.trace(f"Response (first 200 chars): {content[:200]}...")
        self._add_output_tokens(self._estimate_tokens(content))
        return content
