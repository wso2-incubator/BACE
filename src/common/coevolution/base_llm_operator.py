"""
BaseLLMOperator: Minimal infrastructure for LLM-based genetic operators.

This module provides a lightweight base class that handles ONLY LLM orchestration.

Design:
- LLM client is responsible for retry logic (network, rate limits, timeouts)
- BaseLLMOperator just calls llm.generate() and trusts it works
- Domain-specific validation/extraction is delegated to external utilities
"""

from abc import ABC
from typing import Protocol

from loguru import logger

from .core.interfaces import Problem


class ILanguageModel(Protocol):
    """Protocol defining the minimal interface for language models."""

    def generate(self, prompt: str) -> str:
        """Generate text response from the language model."""
        ...


class LLMGenerationError(Exception):
    """Raised when LLM fails to generate output."""

    pass


class BaseLLMOperator(ABC):
    """
    Abstract base class providing LLM orchestration for genetic operators.

    Provides access to problem context and handles missing starter code gracefully.
    """

    DEFAULT_STARTER_CODE = """class Solution:
    def sol(self, input_str):
        pass
"""

    def __init__(self, llm: ILanguageModel, problem: Problem) -> None:
        """
        Initialize the BaseLLMOperator with a language model and problem.

        Args:
            llm: Any object implementing ILanguageModel protocol (has generate() method)
                 The client is responsible for retry logic and error handling.
            problem: The problem context containing question, tests, and starter code.
        """
        self._llm = llm
        self._problem = problem
        logger.debug(f"Initialized {self.__class__.__name__}")

    @property
    def starter_code(self) -> str:
        """
        Get the starter code, providing a default if the problem has none.

        Returns:
            The problem's starter code, or a default template if empty.
        """
        if not self._problem.starter_code or not self._problem.starter_code.strip():
            logger.debug("Problem has no starter_code, using default template")
            return self.DEFAULT_STARTER_CODE
        return self._problem.starter_code

    @property
    def problem(self) -> Problem:
        """Get the current problem context."""
        return self._problem

    def _generate(self, prompt: str) -> str:
        """
        Send prompt to LLM.

        The LLM client is responsible for handling retries, timeouts,
        and other infrastructure concerns. This method simply delegates
        to the client and handles logging.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Raw text response from the LLM

        Raises:
            LLMGenerationError: If LLM fails to generate output
            Exception: Any exception raised by the LLM client

        Example:
            >>> class CodeLLMOperator(BaseLLMOperator, ICodeOperator):
            ...     def mutate(self, individual: str) -> str:
            ...         prompt = f"Mutate this code: {individual}"
            ...         response = self._generate(prompt)
            ...         # Now extract and validate using code_preprocessing utilities
            ...         return extract_code_block(response)
        """
        logger.debug("Sending prompt to LLM")
        logger.trace(f"Prompt preview: {prompt[:100]}...")

        try:
            raw_response = self._llm.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMGenerationError(f"LLM API call failed: {e}") from e

        if not raw_response or not raw_response.strip():
            logger.warning("LLM returned empty response")
            raise LLMGenerationError("LLM returned empty response")

        logger.debug("Received response from LLM")
        logger.trace(f"Raw response preview: {raw_response[:100]}...")

        return raw_response
