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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..code_preprocessing import CodeParsingError, analysis, extraction, transformation
from .core.interfaces import ICodeOperator, ITestOperator, Problem
from .prompt_templates import (
    CROSSOVER_CODE,
    CROSSOVER_TEST,
    EDIT_CODE,
    EDIT_TEST,
    INITIAL_CODE,
    INITIAL_TEST_AGENT_CODER_STYLE,
    MUTATE_CODE,
    MUTATE_TEST,
)


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

    Provides access to problem context. Assumes the Problem instance
    always provides valid starter_code (enforced by the dataset loader).
    """

    def __init__(self, llm: ILanguageModel, problem: Problem) -> None:
        """
        Initialize the BaseLLMOperator with a language model and problem.

        Args:
            llm: Any object implementing ILanguageModel protocol (has generate() method)
                 The client is responsible for retry logic and error handling.
            problem: The problem context containing question, tests, and starter code.
                     The problem MUST have non-empty starter_code.
        """
        self._llm = llm
        self._problem = problem

        # Validate that problem has starter code
        if not problem.starter_code or not problem.starter_code.strip():
            raise ValueError(
                f"Problem '{problem.question_title}' has no starter_code. "
                "The dataset loader should provide a default starter_code."
            )

        logger.debug(f"Initialized {self.__class__.__name__}")

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

    def _extract_code_block(self, response: str) -> str:
        """
        Extract Python code block from LLM response.

        Uses code_preprocessing utilities to extract the first Python code block.
        If no code block is found, returns the original response.

        Args:
            response: Raw text response from the LLM

        Returns:
            Extracted Python code block or original response if not found
        """
        extracted_code: str = extraction.extract_code_block_from_response(response)
        logger.trace(f"Extracted code preview: {extracted_code[:100]}...")
        return extracted_code


class CodeLLMOperator(BaseLLMOperator, ICodeOperator):
    """Concrete LLM-based genetic operator for code snippets."""

    def _extract_all_code_blocks(self, response: str) -> list[str]:
        """
        Extract all Python code blocks from LLM response.

        Uses code_preprocessing utilities to extract all Python code blocks.
        If no code blocks are found, returns an empty list.

        Args:
            response: Raw text response from the LLM
        Returns:
            List of extracted Python code blocks (may be empty)
        """
        return extraction.extract_all_code_blocks_from_response(response)

    def _contains_starter_code(self, code: str) -> bool:
        """
        Check if the given code contains the problem's starter code.

        Args:
            code: Code string to check
        Returns:
            True if starter code is present, False otherwise
        """
        return analysis.contains_starter_code(code, self.problem.starter_code)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValueError, CodeParsingError)),
        reraise=True,
    )
    def create_initial_snippets(self, population_size: int) -> list[str]:
        """
        Create initial code snippets for the population.

        This method generates initial code snippets based on the problem's starter code.
        Uses tenacity to retry up to 3 times if ValueError is raised (e.g., wrong number
        of snippets or missing starter code).

        Args:
            population_size: Number of code snippets to generate
        Returns:
            List of initial code snippets
        Raises:
            ValueError: If after retries, still can't generate valid snippets
        """

        prompt: str = INITIAL_CODE.format(
            population_size=population_size,
            question_content=self.problem.question_content,
            starter_code=self.problem.starter_code,
        )

        response: str = self._generate(prompt)
        code_blocks: list[str] = self._extract_all_code_blocks(response)

        # Validate each code block contains starter code
        for code in code_blocks:
            if not self._contains_starter_code(code):
                logger.error("Generated code does not contain starter code structure.")
                raise ValueError(
                    "Generated code does not contain starter code structure."
                )

        # Validate we got the right number of snippets
        if len(code_blocks) != population_size:
            logger.error(
                f"Generated {len(code_blocks)} code snippets, expected {population_size}."
            )
            raise ValueError(
                f"Generated {len(code_blocks)} code snippets, expected {population_size}."
            )

        logger.info(f"Successfully generated {population_size} initial code snippets")
        logger.trace(f"Generated initial code snippets:\n{'\n\n'.join(code_blocks)}")
        return code_blocks

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValueError, CodeParsingError)),
        reraise=True,
    )
    def crossover(self, parent1: str, parent2: str) -> str:
        """
        Perform crossover between two parent code snippets.

        Args:
            parent1: First parent code snippet
            parent2: Second parent code snippet

        Returns:
            New child code snippet resulting from crossover
        """
        logger.debug("Performing crossover between two parent code snippets")
        prompt = CROSSOVER_CODE.format(
            question_content=self.problem.question_content,
            parent1=parent1,
            parent2=parent2,
        )

        response = self._generate(prompt)
        child_code = self._extract_code_block(response)
        logger.info("Crossover produced a new child code snippet")
        if not self._contains_starter_code(child_code):
            logger.error("Crossover result does not contain starter code structure.")
            raise ValueError(
                "Crossover result does not contain starter code structure."
            )
        logger.trace(f"Generated child code snippet (Crossover):\n{child_code}")
        return child_code

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValueError, CodeParsingError)),
        reraise=True,
    )
    def mutate(self, individual: str) -> str:
        """
        Perform mutation on a individual code snippet.

        Args:
            individual: Individual code snippet to mutate

        Returns:
            New mutated code snippet
        """
        logger.debug("Performing mutation on individual code snippet")
        prompt = MUTATE_CODE.format(
            question_content=self.problem.question_content,
            individual=individual,
        )
        response = self._generate(prompt)
        mutated_code = self._extract_code_block(response)
        if not self._contains_starter_code(mutated_code):
            logger.error("Mutation result does not contain starter code structure.")
            raise ValueError("Mutation result does not contain starter code structure.")
        logger.info("Mutation produced a new code snippet")
        logger.trace(f"Generated mutated code snippet:\n{mutated_code}")
        return mutated_code

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValueError, CodeParsingError)),
        reraise=True,
    )
    def edit(self, individual: str, feedback: str) -> str:
        """
        Edit an individual code snippet based on provided feedback.

        Args:
            individual: Individual code snippet to edit
            feedback: Feedback or suggestions for improvement

        Returns:
            New edited code snippet
        """
        logger.debug("Performing edit on individual code snippet based on feedback")
        prompt = EDIT_CODE.format(
            question_content=self.problem.question_content,
            individual=individual,
            feedback=feedback,
        )
        response = self._generate(prompt)
        edited_code = self._extract_code_block(response)
        if not self._contains_starter_code(edited_code):
            logger.error("Edit result does not contain starter code structure.")
            raise ValueError("Edit result does not contain starter code structure.")
        logger.info("Edit produced a new code snippet")
        logger.trace(f"Generated edited code snippet:\n{edited_code}")
        return edited_code


class TestLLMOperator(BaseLLMOperator, ITestOperator):
    """Concrete LLM-based genetic operator for test cases."""

    def _extract_test_methods(self, test_block: str) -> list[str]:
        """
        Extract test methods from test code block extracted from LLM response.

        Args:
            test_block: extracted test code block from LLM response
        Returns:
            List of test method strings
        """
        return transformation.extract_test_methods_code(test_block)

    def _extract_unittest_block(self, code: str) -> str:
        """
        Extract the unittest code block from the given code snippet.

        Args:
            code: Code snippet potentially containing starter code
        Returns:
            Code snippet with starter code removed
        """
        return transformation.extract_unittest_code(code)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValueError, CodeParsingError)),
        reraise=True,
    )
    def create_initial_snippets(self, population_size: int) -> tuple[list[str], str]:
        """
        Create initial test code snippets for a population of test cases.

        Uses tenacity to retry up to 3 times if ValueError is raised (e.g., wrong number
        of test methods generated).

        Args:
            population_size: Number of initial test snippets to create

        Returns:
            A tuple containing a list of initial test snippets and a string representing the overall test suite
        Raises:
            ValueError: If after retries, still can't generate valid test snippets
        """
        prompt: str = INITIAL_TEST_AGENT_CODER_STYLE.format(
            population_size=population_size,
            question_content=self.problem.question_content,
            starter_code=self.problem.starter_code,
        )

        response: str = self._generate(prompt)
        extracted_code_block: str = self._extract_code_block(response)
        test_block: str = self._extract_unittest_block(extracted_code_block)
        test_methods: list[str] = self._extract_test_methods(test_block)

        if len(test_methods) != population_size:
            logger.error(
                f"Generated {len(test_methods)} test methods, expected {population_size}."
            )
            raise ValueError(
                f"Generated {len(test_methods)} test methods, expected {population_size}."
            )
        logger.info(f"Successfully generated {population_size} initial test snippets")
        logger.trace(f"Generated test block:\n{test_block}")
        return test_methods, test_block

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValueError, CodeParsingError)),
        reraise=True,
    )
    def crossover(self, parent1: str, parent2: str) -> str:
        """
        Perform crossover between two parent test snippets.

        Uses tenacity to retry up to 3 times if ValueError is raised.

        Args:
            parent1: First parent test snippet
            parent2: Second parent test snippet
        Returns:
            New child test snippet resulting from crossover
        Raises:
            ValueError: If after retries, still can't generate valid test snippet
        """
        prompt = CROSSOVER_TEST.format(
            question_content=self.problem.question_content,
            parent1=parent1,
            parent2=parent2,
        )

        response = self._generate(prompt)
        child_test = self._extract_code_block(response)
        logger.info("Crossover produced a new child test snippet")
        logger.trace(f"Generated child test snippet (Crossover):\n{child_test}")
        return child_test

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValueError, CodeParsingError)),
        reraise=True,
    )
    def mutate(self, individual: str) -> str:
        """
        Mutate an individual test snippet by modifying its structure or assertions.

        Uses tenacity to retry up to 3 times if ValueError is raised.

        Args:
            individual: Individual test snippet to mutate
        Returns:
            New mutated test snippet
        Raises:
            ValueError: If after retries, still can't generate valid test snippet
        """
        prompt = MUTATE_TEST.format(
            question_content=self.problem.question_content, individual=individual
        )

        response = self._generate(prompt)
        mutated_test = self._extract_code_block(response)
        logger.info("Mutation produced a new test snippet")
        logger.trace(f"Generated mutated test snippet:\n{mutated_test}")
        return mutated_test

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValueError, CodeParsingError)),
        reraise=True,
    )
    def edit(self, individual: str, feedback: str) -> str:
        """
        Edit an individual test snippet based on provided feedback.

        Uses tenacity to retry up to 3 times if ValueError is raised.

        Args:
            individual: Individual test snippet to edit
            feedback: Feedback or suggestions for improvement
        Returns:
            New edited test snippet
        Raises:
            ValueError: If after retries, still can't generate valid test snippet
        """
        prompt = EDIT_TEST.format(
            question_content=self.problem.question_content,
            individual=individual,
            feedback=feedback,
        )

        response = self._generate(prompt)
        edited_test = self._extract_code_block(response)
        logger.info("Edit produced a new test snippet")
        logger.trace(f"Generated edited test snippet:\n{edited_test}")
        return edited_test
