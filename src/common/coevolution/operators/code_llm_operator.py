"""Code LLM operator implementation and DTOs.

This file contains the `CodeLLMOperator` plus its input DTO classes.
It mirrors the logic previously found in `llm_operators.py` but keeps
code-specific DTOs colocated with the operator implementation.
"""

from dataclasses import dataclass
from typing import List

from loguru import logger

from ...code_preprocessing import (
    CodeParsingError,
    analysis,
    composition,
    extraction,
    transformation,
)
from ...code_preprocessing.exceptions import CodeTransformationError
from ..core.interfaces import (
    BaseOperatorInput,
    InitialInput,
    IOperator,
    OperatorOutput,
    OperatorResult,
)
from ..prompt_templates import (
    CROSSOVER_CODE,
    EDIT_CODE_FIX_FAIL_ONLY,
    INITIAL_CODE,
    MUTATE_CODE,
)
from .base_llm_operator import (
    BaseLLMOperator,
    ILanguageModel,
    LLMGenerationError,
    UnsupportedOperatorInput,
    llm_retry,
)


@dataclass(frozen=True)
class CodeMutationInput(BaseOperatorInput):
    parent_snippet: str
    starter_code: str


@dataclass(frozen=True)
class CodeCrossoverInput(BaseOperatorInput):
    parent1_snippet: str
    parent2_snippet: str
    starter_code: str


@dataclass(frozen=True)
class CodeEditInput(BaseOperatorInput):
    parent_snippet: str
    failing_test_case: str
    error_trace: str
    starter_code: str


class CodeLLMOperator(BaseLLMOperator, IOperator):
    def supported_operations(self) -> set[str]:
        return {"mutation", "crossover", "edit"}

    def _extract_all_code_blocks(self, response: str) -> List[str]:
        return extraction.extract_all_code_blocks_from_response(response)

    def _contains_starter_code(self, code: str, starter_code: str) -> bool:
        return analysis.contains_starter_code(code, starter_code)

    @llm_retry((ValueError, CodeParsingError))
    def generate_initial_snippets(
        self, input_dto: InitialInput
    ) -> tuple[OperatorOutput, str | None]:
        population_size = input_dto.population_size
        problem_description = input_dto.question_content
        starter_code = input_dto.starter_code

        prompt: str = INITIAL_CODE.format(
            population_size=population_size,
            question_content=problem_description,
            starter_code=starter_code,
        )

        response: str = self._generate(prompt)
        code_blocks: List[str] = self._extract_all_code_blocks(response)

        for code in code_blocks:
            if not self._contains_starter_code(code, starter_code):
                logger.error(
                    "One of the generated code snippets does not contain starter code."
                )
                raise ValueError(
                    "One of the generated code snippets does not contain starter code."
                )

        if len(code_blocks) != population_size:
            logger.error(
                f"Generated {len(code_blocks)} code snippets, expected {population_size}."
            )
            raise ValueError(
                f"Generated {len(code_blocks)} code snippets, expected {population_size}."
            )

        results = [OperatorResult(snippet=code, metadata={}) for code in code_blocks]
        return OperatorOutput(results=results), None

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_crossover(self, input_dto: CodeCrossoverInput) -> OperatorOutput:
        logger.debug("Performing crossover between two parent code snippets")
        prompt = CROSSOVER_CODE.format(
            question_content=input_dto.question_content,
            parent1=input_dto.parent1_snippet,
            parent2=input_dto.parent2_snippet,
            starter_code=input_dto.starter_code,
        )

        response = self._generate(prompt)
        child_code = extraction.extract_code_block_from_response(response)

        if not self._contains_starter_code(child_code, input_dto.starter_code):
            logger.error("Crossover result does not contain starter code structure.")
            raise ValueError(
                "Crossover result does not contain starter code structure."
            )

        return OperatorOutput(
            results=[
                OperatorResult(snippet=child_code, metadata={"operation": "crossover"})
            ]
        )

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_mutation(self, input_dto: CodeMutationInput) -> OperatorOutput:
        logger.debug("Performing mutation on individual code snippet")
        prompt = MUTATE_CODE.format(
            question_content=input_dto.question_content,
            individual=input_dto.parent_snippet,
            starter_code=input_dto.starter_code,
        )
        response = self._generate(prompt)
        mutated_code = extraction.extract_code_block_from_response(response)

        if not self._contains_starter_code(mutated_code, input_dto.starter_code):
            logger.error("Mutation result does not contain starter code structure.")
            raise ValueError("Mutation result does not contain starter code structure.")

        return OperatorOutput(
            results=[
                OperatorResult(snippet=mutated_code, metadata={"operation": "mutation"})
            ]
        )

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_edit(self, input_dto: CodeEditInput) -> OperatorOutput:
        logger.debug("Performing edit on individual code snippet based on feedback")
        prompt = EDIT_CODE_FIX_FAIL_ONLY.format(
            question_content=input_dto.question_content,
            starter_code=input_dto.starter_code,
            individual=input_dto.parent_snippet,
            failing_test_case=input_dto.failing_test_case,
            error_trace=input_dto.error_trace,
        )
        response = self._generate(prompt)
        edited_code = extraction.extract_code_block_from_response(response)

        if not self._contains_starter_code(edited_code, input_dto.starter_code):
            logger.error("Edit result does not contain starter code structure.")
            raise ValueError("Edit result does not contain starter code structure.")

        return OperatorOutput(
            results=[
                OperatorResult(snippet=edited_code, metadata={"operation": "edit"})
            ]
        )

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        match input_dto:
            case CodeMutationInput():
                return self._handle_mutation(input_dto)
            case CodeCrossoverInput():
                return self._handle_crossover(input_dto)
            case CodeEditInput():
                return self._handle_edit(input_dto)
            case _:
                raise UnsupportedOperatorInput(
                    type(input_dto), getattr(input_dto, "operation", None)
                )


__all__ = [
    "CodeMutationInput",
    "CodeCrossoverInput",
    "CodeEditInput",
    "CodeLLMOperator",
]
