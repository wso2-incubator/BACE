"""
BaseLLMOperator: Minimal infrastructure for LLM-based genetic operators.

This module provides a lightweight base class that handles ONLY LLM orchestration.

Design:
- LLM client is responsible for retry logic (network, rate limits, timeouts)
- BaseLLMOperator just calls llm.generate() and trusts it works
- Domain-specific validation/extraction is delegated to external utilities
"""

from abc import ABC
from dataclasses import dataclass
from typing import Callable, Protocol, Tuple, Type

from loguru import logger
from tenacity import (
    WrappedFn,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.code_preprocessing.exceptions import CodeTransformationError

from ..code_preprocessing import (
    CodeParsingError,
    analysis,
    composition,
    extraction,
    transformation,
)
from .core.interfaces import (
    BaseOperatorInput,
    InitialInput,
    IOperator,
    OperatorOutput,
    OperatorResult,
)
from .prompt_templates import (
    CROSSOVER_CODE,
    CROSSOVER_TEST,
    EDIT_CODE_FIX_FAIL_ONLY,
    EDIT_TEST,
    INITIAL_CODE,
    INITIAL_TEST_AGENT_CODER_STYLE,
    MUTATE_CODE,
    MUTATE_TEST,
)

# ============================================
# === CODE OPERATOR INPUT DTOs
# ============================================


@dataclass(frozen=True)
class CodeMutationInput(BaseOperatorInput):
    """Input DTO for code mutation operations."""

    parent_snippet: str
    starter_code: str


@dataclass(frozen=True)
class CodeCrossoverInput(BaseOperatorInput):
    """Input DTO for code crossover operations."""

    parent1_snippet: str
    parent2_snippet: str
    starter_code: str


@dataclass(frozen=True)
class CodeEditInput(BaseOperatorInput):
    """Input DTO for code edit operations."""

    parent_snippet: str
    feedback: str
    starter_code: str


# ============================================
# === TEST OPERATOR INPUT DTOs
# ============================================


@dataclass(frozen=True)
class TestMutationInput(BaseOperatorInput):
    """Input DTO for test mutation operations."""

    parent_snippet: str


@dataclass(frozen=True)
class TestCrossoverInput(BaseOperatorInput):
    """Input DTO for test crossover operations."""

    parent1_snippet: str
    parent2_snippet: str


@dataclass(frozen=True)
class TestEditInput(BaseOperatorInput):
    """Input DTO for test edit operations."""

    parent_snippet: str
    feedback: str


# ============================================
# === UTILITY FUNCTIONS
# ============================================


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

    pass


class BaseLLMOperator(ABC):
    """
    Abstract base class providing LLM orchestration for genetic operators.

    No longer stores problem context - problem information is passed
    via method parameters as needed.
    """

    def __init__(self, llm: ILanguageModel) -> None:
        """
        Initialize the BaseLLMOperator with a language model.

        Args:
            llm: Any object implementing ILanguageModel protocol (has generate() method)
                 The client is responsible for retry logic and error handling.
        """
        self._llm = llm
        logger.debug(f"Initialized {self.__class__.__name__}")

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


class CodeLLMOperator(BaseLLMOperator, IOperator):
    """Concrete LLM-based genetic operator for code snippets implementing IOperator protocol."""

    def supported_operations(self) -> set[str]:
        """Return the set of operations this operator supports."""
        return {"mutation", "crossover", "edit"}

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

    def _contains_starter_code(self, code: str, starter_code: str) -> bool:
        """
        Check if the given code contains the starter code.

        Args:
            code: Code string to check
            starter_code: The starter code to look for
        Returns:
            True if starter code is present, raise ValueError otherwise
        Raises:
            ValueError: If starter code is not found in the code snippet
        """
        return analysis.contains_starter_code(code, starter_code)

    @llm_retry((ValueError, CodeParsingError))
    def generate_initial_snippets(
        self, input_dto: InitialInput
    ) -> tuple[OperatorOutput, str | None]:
        """
        Generate initial code snippets for the population (IOperator interface).

        Uses tenacity to retry up to 3 times if ValueError is raised (e.g., wrong number
        of snippets or missing starter code).

        Args:
            input_dto: `InitialInput` containing population_size, question_content, and starter_code

        Returns:
            Tuple of (OperatorOutput with snippets, None for code operators)

        Raises:
            ValueError: If after retries, still can't generate valid snippets
        """
        population_size = input_dto.population_size
        problem_description = input_dto.question_content
        starter_code = input_dto.starter_code

        prompt: str = INITIAL_CODE.format(
            population_size=population_size,
            question_content=problem_description,
            starter_code=starter_code,
        )

        response: str = self._generate(prompt)
        code_blocks: list[str] = self._extract_all_code_blocks(response)

        # Validate each code block contains starter code
        for code in code_blocks:
            if not self._contains_starter_code(code, starter_code):
                logger.error(
                    "One of the generated code snippets does not contain starter code."
                )
                raise ValueError(
                    "One of the generated code snippets does not contain starter code."
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

        results = [OperatorResult(snippet=code, metadata={}) for code in code_blocks]
        return OperatorOutput(results=results), None

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_crossover(self, input_dto: CodeCrossoverInput) -> OperatorOutput:
        """
        Perform crossover between two parent code snippets.

        Args:
            input_dto: CodeCrossoverInput containing parents and context

        Returns:
            OperatorOutput with the child code snippet
        """
        logger.debug("Performing crossover between two parent code snippets")
        prompt = CROSSOVER_CODE.format(
            question_content=input_dto.question_content,
            parent1=input_dto.parent1_snippet,
            parent2=input_dto.parent2_snippet,
            starter_code=input_dto.starter_code,
        )

        response = self._generate(prompt)
        child_code = self._extract_code_block(response)
        logger.info("Crossover produced a new child code snippet")

        if not self._contains_starter_code(child_code, input_dto.starter_code):
            logger.error("Crossover result does not contain starter code structure.")
            raise ValueError(
                "Crossover result does not contain starter code structure."
            )
        logger.trace(f"Generated child code snippet (Crossover):\n{child_code}")

        return OperatorOutput(
            results=[
                OperatorResult(snippet=child_code, metadata={"operation": "crossover"})
            ]
        )

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_mutation(self, input_dto: CodeMutationInput) -> OperatorOutput:
        """
        Perform mutation on a individual code snippet.

        Args:
            input_dto: CodeMutationInput containing parent and context

        Returns:
            OperatorOutput with the mutated code snippet
        """
        logger.debug("Performing mutation on individual code snippet")
        prompt = MUTATE_CODE.format(
            question_content=input_dto.question_content,
            individual=input_dto.parent_snippet,
            starter_code=input_dto.starter_code,
        )
        response = self._generate(prompt)
        mutated_code = self._extract_code_block(response)

        if not self._contains_starter_code(mutated_code, input_dto.starter_code):
            logger.error("Mutation result does not contain starter code structure.")
            raise ValueError("Mutation result does not contain starter code structure.")
        logger.info("Mutation produced a new code snippet")
        logger.trace(f"Generated mutated code snippet:\n{mutated_code}")

        return OperatorOutput(
            results=[
                OperatorResult(snippet=mutated_code, metadata={"operation": "mutation"})
            ]
        )

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_edit(self, input_dto: CodeEditInput) -> OperatorOutput:
        """
        Edit an individual code snippet based on provided feedback.

        Args:
            input_dto: CodeEditInput containing parent, feedback, and context

        Returns:
            OperatorOutput with the edited code snippet
        """
        logger.debug("Performing edit on individual code snippet based on feedback")
        prompt = EDIT_CODE_FIX_FAIL_ONLY.format(
            question_content=input_dto.question_content,
            starter_code=input_dto.starter_code,
            individual=input_dto.parent_snippet,
            feedback=input_dto.feedback,
        )
        response = self._generate(prompt)
        edited_code = self._extract_code_block(response)

        if not self._contains_starter_code(edited_code, input_dto.starter_code):
            logger.error("Edit result does not contain starter code structure.")
            raise ValueError("Edit result does not contain starter code structure.")
        logger.info("Edit produced a new code snippet")
        logger.trace(f"Generated edited code snippet:\n{edited_code}")

        return OperatorOutput(
            results=[
                OperatorResult(snippet=edited_code, metadata={"operation": "edit"})
            ]
        )

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        """
        Dispatcher: Routes the DTO to the specific LLM strategy.

        Args:
            input_dto: Input DTO specifying the operation and context

        Returns:
            OperatorOutput with the generated code snippet(s)

        Raises:
            ValueError: If unsupported input type is provided
        """
        match input_dto:
            case CodeMutationInput():
                return self._handle_mutation(input_dto)
            case CodeCrossoverInput():
                return self._handle_crossover(input_dto)
            case CodeEditInput():
                return self._handle_edit(input_dto)
            case _:
                raise ValueError(f"Unsupported input type: {type(input_dto)}")


class TestLLMOperator(BaseLLMOperator):
    """Concrete LLM-based genetic operator for test cases. (To be refactored to implement IOperator)"""

    def _extract_test_methods(self, test_block: str) -> list[str]:
        """
        Extract test methods from unittest test code block extracted from LLM response.

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

    def _extract_first_test_method(self, code: str) -> str:
        """
        Extract the first test method from a code snippet.
        The code snippet may contain multiple test methods or a full unittest class.
        Args:
            code: Code snippet potentially containing multiple test methods or a full unittest class
        Returns:
            Code string of the first test method found
        """
        return transformation.extract_first_test_method_code(code)

    def _rebuild_unittest_with_methods(
        self, test_block: str, test_methods: list[str]
    ) -> str:
        """
        Rebuild a unittest class by replacing old test methods with new ones.

        Args:
            test_block: String containing the original unittest test class definition
            test_methods: List of code strings for new test methods to insert
        Returns:
            String containing the rebuilt unittest class with new test methods
        """
        return composition.rebuild_unittest_with_methods(test_block, test_methods)

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
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
            logger.warning(
                f"Generated {len(test_methods)} test methods, expected {population_size}."
            )

        if len(test_methods) > population_size:
            logger.debug("Trimming excess test methods to match population size")
            test_methods = test_methods[:population_size]
            test_block = self._rebuild_unittest_with_methods(test_block, test_methods)

        if len(test_methods) < population_size:
            logger.debug("Adding additional test methods to match population size")
            additional_methods_needed = population_size - len(test_methods)
            prompt_additional = INITIAL_TEST_AGENT_CODER_STYLE.format(
                population_size=additional_methods_needed,
                question_content=self.problem.question_content,
                starter_code=self.problem.starter_code,
            )
            response_additional = self._generate(prompt_additional)
            extracted_additional = self._extract_code_block(response_additional)
            additional_test_block = self._extract_unittest_block(extracted_additional)
            additional_test_methods = self._extract_test_methods(additional_test_block)
            test_methods.extend(
                additional_test_methods[:additional_methods_needed]
            )  # Trim if too many
            test_block = self._rebuild_unittest_with_methods(test_block, test_methods)

        logger.info(f"Successfully generated {population_size} initial test snippets")
        logger.trace(f"Generated test block:\n{test_block}")
        return test_methods, test_block

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
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
        extracted_code = self._extract_code_block(response)
        child_test = self._extract_first_test_method(extracted_code)
        logger.info("Crossover produced a new child test snippet")
        logger.trace(f"Generated child test snippet (Crossover):\n{child_test}")
        return child_test

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
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
        extracted_code = self._extract_code_block(response)
        mutated_test = self._extract_first_test_method(extracted_code)
        logger.info("Mutation produced a new test snippet")
        logger.trace(f"Generated mutated test snippet:\n{mutated_test}")
        return mutated_test

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
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
        extracted_code = self._extract_code_block(response)
        edited_test = self._extract_first_test_method(extracted_code)
        logger.info("Edit produced a new test snippet")
        logger.trace(f"Generated edited test snippet:\n{edited_test}")
        return edited_test
