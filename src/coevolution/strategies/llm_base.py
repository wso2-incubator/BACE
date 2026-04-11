"""Shared base classes and helpers for LLM-backed evolutionary operators.

This module contains protocol definitions, retry decorator factory,
and the `BaseLLMService` class which handles LLM invocation and basic
extraction helpers. It also provides `BaseLLMOperator` (the LLM-backed
extension of IOperator) and `BaseLLMInitializer` to unify dependency
injection for concrete operators.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, Tuple, Type

from loguru import logger
from tenacity import (
    WrappedFn,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from coevolution.core.interfaces.base import BaseIndividual
from coevolution.core.interfaces.config import PopulationConfig
from coevolution.core.interfaces.context import CoevolutionContext
from coevolution.core.interfaces.initializer import IPopulationInitializer
from coevolution.core.interfaces.language import ICodeParser
from coevolution.core.interfaces.operators import IOperator
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy
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


class LLMSyntaxError(Exception):
    """Raised when LLM generates code with invalid syntax."""


class BaseLLMService:
    """Abstract base service providing LLM orchestration.

    Handles prompt rendering tools, LLM calls, retries, and code block extraction.
    """

    def __init__(
        self, llm: ILanguageModel, parser: ICodeParser, language_name: str
    ) -> None:
        self._llm = llm
        self.parser = parser
        self.language_name = language_name
        self.prompt_manager = get_prompt_manager(language=language_name)
        logger.debug(
            f"Initialized {self.__class__.__name__} with prompt manager for {language_name}"
        )

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
        blocks = self.parser.extract_code_blocks(response)
        if blocks:
            return blocks[0]
        return response

    def _validate_syntax(self, code: str) -> None:
        """
        Validate syntax of the provided code using the language parser.

        Args:
            code: The code to validate

        Raises:
            LLMSyntaxError: If the syntax is invalid
        """
        if not self.parser.is_syntax_valid(code):
            raise LLMSyntaxError("Generated code has invalid syntax")


class BaseLLMOperator[T: BaseIndividual](BaseLLMService, IOperator[T], ABC):
    """
    Base class for all LLM-backed genetic operators (Mutation, Crossover, Edit).

    Extends IOperator with LLM generation capabilities and standard evolutionary
    dependencies: parent selection and probability assignment.
    Concrete classes implement `execute()`.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        parent_selector: IParentSelectionStrategy[T],
        prob_assigner: IProbabilityAssigner,
    ) -> None:
        super().__init__(llm, parser, language_name)
        self.parent_selector = parent_selector
        self.prob_assigner = prob_assigner

    @abstractmethod
    def operation_name(self) -> str: ...

    @abstractmethod
    def execute(self, context: CoevolutionContext) -> list[T]: ...


class BaseLLMInitializer[T: BaseIndividual](
    BaseLLMService, IPopulationInitializer[T], ABC
):
    """
    Base class for all population initializers (Gen 0 creators).

    Combines LLM generation capabilities with a population configuration.
    Concrete classes implement `initialize()`.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
    ) -> None:
        super().__init__(llm, parser, language_name)
        self.pop_config = pop_config

    @abstractmethod
    def initialize(self, problem: Any) -> list[T]: ...


__all__ = [
    "ILanguageModel",
    "llm_retry",
    "LLMGenerationError",
    "LLMSyntaxError",
    "BaseLLMService",
    "BaseLLMOperator",
    "BaseLLMInitializer",
]
