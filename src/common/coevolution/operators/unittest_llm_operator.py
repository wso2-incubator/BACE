"""Unittest operator implementation and DTOs.

Contains `UnittestLLMOperator` and the DTOs used for unittest-based
test generation and manipulation.
"""

from dataclasses import dataclass

from loguru import logger

from ...code_preprocessing import (
    CodeParsingError,
    CodeTransformationError,
    composition,
    transformation,
)
from ..core.interfaces import (
    BaseOperatorInput,
    InitialInput,
    IOperator,
    OperatorOutput,
    OperatorResult,
)
from ..prompt_templates import (
    CROSSOVER_TEST,
    EDIT_TEST,
    INITIAL_TEST_AGENT_CODER_STYLE,
    MUTATE_TEST,
)
from .base_llm_operator import BaseLLMOperator, UnsupportedOperatorInput, llm_retry


@dataclass(frozen=True)
class UnittestMutationInput(BaseOperatorInput):
    parent_snippet: str


@dataclass(frozen=True)
class UnittestCrossoverInput(BaseOperatorInput):
    parent1_snippet: str
    parent2_snippet: str


@dataclass(frozen=True)
class UnittestEditInput(BaseOperatorInput):
    parent_snippet: str
    feedback: str


class UnittestLLMOperator(BaseLLMOperator, IOperator):
    """Concrete LLM-based genetic operator for unittest-based tests implementing IOperator."""

    def supported_operations(self) -> set[str]:
        return {"mutation", "crossover", "edit"}

    def _extract_test_methods(self, test_block: str) -> list[str]:
        return transformation.extract_test_methods_code(test_block)

    def _extract_unittest_block(self, code: str) -> str:
        return transformation.extract_unittest_code(code)

    def _extract_first_test_method(self, code: str) -> str:
        return transformation.extract_first_test_method_code(code)

    def _rebuild_unittest_with_methods(
        self, test_block: str, test_methods: list[str]
    ) -> str:
        return composition.rebuild_unittest_with_methods(test_block, test_methods)

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def generate_initial_snippets(
        self, input_dto: InitialInput
    ) -> tuple[OperatorOutput, str | None]:
        """Generate initial test methods and an overall unittest block (context_code)."""
        population_size = input_dto.population_size
        prompt: str = INITIAL_TEST_AGENT_CODER_STYLE.format(
            population_size=population_size,
            question_content=input_dto.question_content,
            starter_code=input_dto.starter_code,
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
            test_methods = test_methods[:population_size]
            test_block = self._rebuild_unittest_with_methods(test_block, test_methods)

        if len(test_methods) < population_size:
            additional_needed = population_size - len(test_methods)
            prompt_additional = INITIAL_TEST_AGENT_CODER_STYLE.format(
                population_size=additional_needed,
                question_content=input_dto.question_content,
                starter_code=input_dto.starter_code,
            )
            response_additional = self._generate(prompt_additional)
            extracted_additional = self._extract_code_block(response_additional)
            additional_block = self._extract_unittest_block(extracted_additional)
            additional_methods = self._extract_test_methods(additional_block)
            test_methods.extend(additional_methods[:additional_needed])
            test_block = self._rebuild_unittest_with_methods(test_block, test_methods)

        results = [OperatorResult(snippet=m, metadata={}) for m in test_methods]
        logger.info(f"Successfully generated {len(results)} initial test snippets")
        return OperatorOutput(results=results), test_block

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_crossover(self, input_dto: UnittestCrossoverInput) -> OperatorOutput:
        prompt = CROSSOVER_TEST.format(
            question_content=input_dto.question_content,
            parent1=input_dto.parent1_snippet,
            parent2=input_dto.parent2_snippet,
        )
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        child = self._extract_first_test_method(extracted)
        return OperatorOutput(
            results=[OperatorResult(snippet=child, metadata={"operation": "crossover"})]
        )

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_mutation(self, input_dto: UnittestMutationInput) -> OperatorOutput:
        prompt = MUTATE_TEST.format(
            question_content=input_dto.question_content,
            individual=input_dto.parent_snippet,
        )
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        mutated = self._extract_first_test_method(extracted)
        return OperatorOutput(
            results=[
                OperatorResult(snippet=mutated, metadata={"operation": "mutation"})
            ]
        )

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_edit(self, input_dto: UnittestEditInput) -> OperatorOutput:
        prompt = EDIT_TEST.format(
            question_content=input_dto.question_content,
            individual=input_dto.parent_snippet,
            feedback=input_dto.feedback,
        )
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        edited = self._extract_first_test_method(extracted)
        return OperatorOutput(
            results=[OperatorResult(snippet=edited, metadata={"operation": "edit"})]
        )

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        match input_dto:
            case UnittestMutationInput():
                return self._handle_mutation(input_dto)
            case UnittestCrossoverInput():
                return self._handle_crossover(input_dto)
            case UnittestEditInput():
                return self._handle_edit(input_dto)
            case _:
                raise UnsupportedOperatorInput(
                    type(input_dto), getattr(input_dto, "operation", None)
                )


__all__ = [
    "UnittestMutationInput",
    "UnittestCrossoverInput",
    "UnittestEditInput",
    "UnittestLLMOperator",
]
