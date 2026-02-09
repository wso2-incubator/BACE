"""Shared base classes and helpers for LLM-backed operators.

This module contains protocol definitions, retry decorator factory,
and the `BaseLLMOperator` class which handles LLM invocation and basic
extraction helpers. Concrete operators import these utilities.
"""

from typing import Any, Callable, Protocol, Tuple, Type

from loguru import logger
from tenacity import (
    WrappedFn,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from coevolution.core.interfaces.language import ILanguageAdapter
from coevolution.utils.prompt_manager import get_prompt_manager


def llm_retry(
    exception_types: Tuple[Type[Exception], ...],
) -> Callable[[WrappedFn], WrappedFn]:
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(exception_types),
        reraise=True,
    )


class ILanguageModel(Protocol):
    """Protocol defining the minimal interface for language models."""

    def generate(self, prompt: str) -> str:
        """Generate text response from the language model."""
        ...


class LLMGenerationError(Exception):
    """Raised when LLM fails to generate output."""


class UnsupportedOperatorInput(Exception):
    """Raised when an operator receives an input DTO it doesn't support.

    Message includes the input type and (if available) the declared operation.
    """

    def __init__(self, input_type: type, operation: str | None = None) -> None:
        op_part = f" for operation '{operation}'" if operation else ""
        super().__init__(
            f"Unsupported operator input type: {getattr(input_type, '__name__', str(input_type))}{op_part}. "
            "The operation might not be supported by this operator."
        )


class BaseLLMOperator:
    """Abstract base class providing LLM orchestration for genetic operators.

    Concrete operators should subclass this and implement specific
    prompt construction and postprocessing.
    """

    def __init__(self, llm: ILanguageModel, language_adapter: ILanguageAdapter) -> None:
        self._llm = llm
        self.language_adapter = language_adapter
        self.prompt_manager = get_prompt_manager()
        logger.debug(f"Initialized {self.__class__.__name__}")

    @llm_retry(exception_types=(LLMGenerationError,))
    def _generate(self, prompt: Any) -> str:
        logger.debug("Sending prompt to LLM")
        logger.trace(f"Prompt preview: {prompt[:100]}...")

        try:
            raw_response = self._llm.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            logger.debug(f"Prompt that caused failure: {prompt}")
            raise LLMGenerationError(f"LLM API call failed: {e}") from e

        if not raw_response or not raw_response.strip():
            logger.warning("LLM returned empty response")
            raise LLMGenerationError("LLM returned empty response")

        logger.debug("Received response from LLM")
        logger.trace(f"Raw response preview: {raw_response[:100]}...")

        return raw_response

    def _extract_code_block(self, response: str) -> str:
        """
        Extract code block from LLM response using language adapter.

        Args:
            response: Raw text response from the LLM

        Returns:
            Extracted code block or the original response if no block found
        """
        blocks = self.language_adapter.extract_code_blocks(response)
        if blocks:
            return blocks[0]
        return response


__all__ = [
    "ILanguageModel",
    "llm_retry",
    "LLMGenerationError",
    "UnsupportedOperatorInput",
    "BaseLLMOperator",
]
