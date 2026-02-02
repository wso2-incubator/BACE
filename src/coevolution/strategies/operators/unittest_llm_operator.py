"""Pytest test operator implementation and DTOs.

Contains `PytestLLMOperator` (formerly UnittestLLMOperator) and the DTOs used for
pytest-based test generation and manipulation. Now generates standalone test
functions instead of unittest.TestCase classes.
"""

from dataclasses import dataclass

from loguru import logger

from coevolution.core.interfaces import (
    BaseOperatorInput,
    InitialInput,
    IOperator,
    OperatorOutput,
    OperatorResult,
)
from coevolution.utils.prompts import (
    CROSSOVER_TEST,
    EDIT_ALL_FAILING_UNITTEST,
    EDIT_ALL_PASSING_UNITTEST,
    EDIT_DISCRIMINATING_UNITTEST,
    INITIAL_TEST_AGENT_CODER_STYLE,
    MUTATE_TEST,
)
from infrastructure.code_preprocessing import (
    CodeParsingError,
    CodeTransformationError,
    extraction,
)

from .base_llm_operator import BaseLLMOperator, UnsupportedOperatorInput, llm_retry


@dataclass(frozen=True)
class UnittestMutationInput(BaseOperatorInput):
    """Input DTO for test mutation. Name kept for backward compatibility."""

    parent_snippet: str


@dataclass(frozen=True)
class UnittestCrossoverInput(BaseOperatorInput):
    """Input DTO for test crossover. Name kept for backward compatibility."""

    parent1_snippet: str
    parent2_snippet: str


@dataclass(frozen=True)
class UnittestEditInput(BaseOperatorInput):
    """Input DTO for test editing. Name kept for backward compatibility."""

    parent_snippet: str
    passing_code_snippets: list[str]  # snippets only
    failing_code_snippets_with_traces: list[tuple[str, str]]  # (snippet, trace)


class PytestLLMOperator(BaseLLMOperator, IOperator):
    """
    Concrete LLM-based genetic operator for pytest standalone test functions.

    Formerly UnittestLLMOperator, now generates pytest functions instead of unittest
    classes. This dramatically simplifies the implementation by removing AST
    manipulation for class reconstruction.
    """

    def supported_operations(self) -> set[str]:
        return {"mutation", "crossover", "edit"}

    def _extract_test_functions(self, test_code: str) -> list[str]:
        """Extract standalone pytest test functions from code."""
        return extraction.extract_test_functions_code(test_code)

    def _extract_first_test_function(self, code: str) -> str:
        """Extract the first test function from code."""
        functions = self._extract_test_functions(code)
        if not functions:
            raise CodeParsingError("No test functions found in generated code")
        return functions[0]

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def generate_initial_snippets(self, input_dto: InitialInput) -> OperatorOutput:
        """
        Generate initial pytest test functions and return them as snippets.

        Returns:
            OperatorOutput with test function snippets. TestPopulation will build
            the complete test block by concatenating imports + test functions.
        """
        population_size = input_dto.population_size
        logger.info(
            f"Generating {population_size} initial test function snippets for question: {input_dto.question_content[:50]}..."
        )
        prompt: str = INITIAL_TEST_AGENT_CODER_STYLE.format(
            population_size=population_size,
            question_content=input_dto.question_content,
            starter_code=input_dto.starter_code,
        )

        response: str = self._generate(prompt)
        extracted_code_block: str = self._extract_code_block(response)
        # No need to extract "unittest block" - we already have standalone functions
        test_functions: list[str] = self._extract_test_functions(extracted_code_block)

        logger.debug(
            f"Extracted {len(test_functions)} test functions from initial response"
        )

        if len(test_functions) != population_size:
            logger.warning(
                f"Generated {len(test_functions)} test functions, expected {population_size}."
            )

        # Truncate if we have too many
        if len(test_functions) > population_size:
            test_functions = test_functions[:population_size]
            logger.debug(f"Truncated to {population_size} test functions")

        # Generate more if we don't have enough
        if len(test_functions) < population_size:
            additional_needed = population_size - len(test_functions)
            logger.info(f"Generating {additional_needed} additional test functions")
            prompt_additional = INITIAL_TEST_AGENT_CODER_STYLE.format(
                population_size=additional_needed,
                question_content=input_dto.question_content,
                starter_code=input_dto.starter_code,
            )
            response_additional = self._generate(prompt_additional)
            extracted_additional = self._extract_code_block(response_additional)
            additional_functions = self._extract_test_functions(extracted_additional)
            test_functions.extend(additional_functions[:additional_needed])
            logger.debug(
                f"Added {len(additional_functions[:additional_needed])} additional test functions"
            )

        results = [OperatorResult(snippet=func, metadata={}) for func in test_functions]
        logger.info(
            f"Successfully generated {len(results)} initial test function snippets"
        )
        return OperatorOutput(results=results)

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_crossover(self, input_dto: UnittestCrossoverInput) -> OperatorOutput:
        logger.debug("Performing test crossover operation")
        prompt = CROSSOVER_TEST.format(
            question_content=input_dto.question_content,
            parent1=input_dto.parent1_snippet,
            parent2=input_dto.parent2_snippet,
        )
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        child = self._extract_first_test_function(extracted)
        logger.debug(f"Generated crossover child with length {len(child)}")
        return OperatorOutput(
            results=[OperatorResult(snippet=child, metadata={"operation": "crossover"})]
        )

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_mutation(self, input_dto: UnittestMutationInput) -> OperatorOutput:
        logger.debug("Performing test mutation operation")
        prompt = MUTATE_TEST.format(
            question_content=input_dto.question_content,
            individual=input_dto.parent_snippet,
        )
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        mutated = self._extract_first_test_function(extracted)
        logger.debug(f"Generated mutated child with length {len(mutated)}")
        return OperatorOutput(
            results=[
                OperatorResult(snippet=mutated, metadata={"operation": "mutation"})
            ]
        )

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_edit(self, input_dto: UnittestEditInput) -> OperatorOutput:
        logger.debug("Performing unittest edit operation")

        edit_operation_type: str
        if (
            len(input_dto.passing_code_snippets) > 0
            and len(input_dto.failing_code_snippets_with_traces) > 0
        ):
            edit_operation_type = "discriminating"
            prompt = EDIT_DISCRIMINATING_UNITTEST.format(
                question_content=input_dto.question_content,
                current_test_snippet=input_dto.parent_snippet,
                passing_code_snippet=input_dto.passing_code_snippets[0],
                failing_code_snippet=input_dto.failing_code_snippets_with_traces[0][0],
                failing_code_trace=input_dto.failing_code_snippets_with_traces[0][1],
            )

        elif len(input_dto.passing_code_snippets) == 0:
            edit_operation_type = "all-failing"
            prompt = EDIT_ALL_FAILING_UNITTEST.format(
                question_content=input_dto.question_content,
                current_test_snippet=input_dto.parent_snippet,
                failing_code_snippet_P=input_dto.failing_code_snippets_with_traces[0][
                    0
                ],
                failing_code_trace_P=input_dto.failing_code_snippets_with_traces[0][1],
                failing_code_snippet_Q=input_dto.failing_code_snippets_with_traces[1][
                    0
                ],
                failing_code_trace_Q=input_dto.failing_code_snippets_with_traces[1][1],
            )

        elif len(input_dto.failing_code_snippets_with_traces) == 0:
            edit_operation_type = "all-passing"
            prompt = EDIT_ALL_PASSING_UNITTEST.format(
                question_content=input_dto.question_content,
                current_test_snippet=input_dto.parent_snippet,
                passing_code_snippet_P=input_dto.passing_code_snippets[0],
                passing_code_snippet_Q=input_dto.passing_code_snippets[1],
            )

        else:
            edit_operation_type = "unknown"

        logger.debug(f"Using edit operation type: {edit_operation_type}")
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        edited = self._extract_first_test_function(extracted)
        logger.debug(f"Generated edited child with length {len(edited)}")
        return OperatorOutput(
            results=[
                OperatorResult(
                    snippet=edited, metadata={"operation": edit_operation_type}
                )
            ]
        )

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        operation = getattr(input_dto, "operation", "unknown")
        logger.info(
            f"Applying pytest test operator for operation '{operation}' with input type {type(input_dto).__name__}"
        )
        match input_dto:
            case UnittestMutationInput():
                result = self._handle_mutation(input_dto)
            case UnittestCrossoverInput():
                result = self._handle_crossover(input_dto)
            case UnittestEditInput():
                result = self._handle_edit(input_dto)
            case _:
                logger.error(
                    f"Unsupported operation input: {type(input_dto)} for operation {operation}"
                )
                raise UnsupportedOperatorInput(
                    type(input_dto), getattr(input_dto, "operation", None)
                )
        logger.info(
            f"Successfully applied operation '{operation}', produced {len(result.results)} results"
        )
        return result


# Backward compatibility alias
UnittestLLMOperator = PytestLLMOperator


__all__ = [
    "UnittestMutationInput",
    "UnittestCrossoverInput",
    "UnittestEditInput",
    "UnittestLLMOperator",  # Legacy name
    "PytestLLMOperator",  # New name
]
