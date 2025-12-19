"""Unittest operator implementation and DTOs.

Contains `UnittestLLMOperator` and the DTOs used for unittest-based
test generation and manipulation.
"""

from dataclasses import dataclass
from typing import Any, TypedDict, cast

from loguru import logger

from ...code_preprocessing import (
    CodeParsingError,
    CodeTransformationError,
    composition,
    transformation,
)
from ..core.interfaces import (
    OPERATION_CROSSOVER,
    BaseOperatorInput,
    InitialInput,
    IOperator,
    OperatorOutput,
    OperatorResult,
)
from ..prompt_templates import DIFFERENTIAL_INPUT_GENERATOR_PROMPT
from .base_llm_operator import BaseLLMOperator, UnsupportedOperatorInput, llm_retry


class DifferentialInputOutput(TypedDict):
    inputdata: dict[str, Any]
    output: Any


@dataclass(frozen=True)
class DifferentialCrossoverInput(BaseOperatorInput):
    starter_code: str
    differential_parent_1_io_pairs: list[DifferentialInputOutput]
    differential_parent_1_id: str
    differential_parent_2_io_pairs: list[DifferentialInputOutput]
    differential_parent_2_id: str


@dataclass(frozen=True)
class DifferentialGenScriptInput(BaseOperatorInput):
    equivalent_code_snippet_1: str
    equivalent_code_snippet_2: str
    passing_differential_test_io_pairs: list[DifferentialInputOutput]
    num_inputs_to_generate: int


# New Operation Constant
OPERATION_DISCOVERY: str = "discovery"


class DifferentialLLMOperator(BaseLLMOperator, IOperator):
    """Concrete LLM-based genetic operator for differential tests implementing IOperator."""

    def supported_operations(self) -> set[str]:
        return {OPERATION_DISCOVERY, OPERATION_CROSSOVER}

    def _rebuild_unittest_with_methods(
        self, test_block: str, test_methods: list[str]
    ) -> str:
        return composition.rebuild_unittest_with_methods(test_block, test_methods)

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def generate_initial_snippets(
        self, input_dto: InitialInput
    ) -> tuple[OperatorOutput, str | None]:
        """Initially there are no differential tests to generate. The test block will be set with the setup code."""

        test_class_block = transformation.setup_unittest_class_from_starter_code(
            input_dto.starter_code
        )
        result = OperatorOutput(results=[])
        return result, test_class_block

    def get_test_method_from_io(
        self,
        starter_code: str,
        io_pairs: list[DifferentialInputOutput],
        code_parent_ids: list[str],
    ) -> str:
        """Build a test method from differential input-output pairs.
        Note: This is not a LLM operation. or a genetic operation.
        """

        method_code = transformation.build_test_method_from_io(
            starter_code, cast(list[dict[str, Any]], io_pairs), code_parent_ids
        )
        return method_code

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_generation_script(
        self, input_dto: DifferentialGenScriptInput
    ) -> OperatorOutput:
        """Generate a script to generate inputs to differentiate between two equivalent code snippets."""

        prompt = DIFFERENTIAL_INPUT_GENERATOR_PROMPT.format(
            code_snippet_P=input_dto.equivalent_code_snippet_1,
            code_snippet_Q=input_dto.equivalent_code_snippet_2,
            current_tests=input_dto.passing_differential_test_io_pairs,
        )

        llm_response = self._llm.generate(prompt)

        code_block = self._extract_code_block(llm_response)
        generated_script = transformation.remove_if_main_block(code_block)
        generated_script += (
            f"\nprint(generate_test_inputs({input_dto.num_inputs_to_generate}))"
        )

        result = OperatorOutput(results=[OperatorResult(snippet=generated_script)])
        return result

    def _handle_crossover(
        self, input_dto: DifferentialCrossoverInput
    ) -> OperatorOutput:
        """
        No LLM crossover for differential tests; simply cross combines test methods from both parents.
        Produces two children by swapping half the methods.
        """

        parent_1_io_pairs = input_dto.differential_parent_1_io_pairs
        parent_2_io_pairs = input_dto.differential_parent_2_io_pairs

        split_idx_1 = len(parent_1_io_pairs) // 2
        split_idx_2 = len(parent_2_io_pairs) // 2

        child_1_io_pairs = (
            parent_1_io_pairs[:split_idx_1] + parent_2_io_pairs[split_idx_2:]
        )
        child_2_io_pairs = (
            parent_2_io_pairs[:split_idx_2] + parent_1_io_pairs[split_idx_1:]
        )

        child_1_test_method = transformation.build_test_method_from_io(
            starter_code=input_dto.starter_code,
            io_pairs=cast(list[dict[str, Any]], child_1_io_pairs),
            parent_ids=[
                input_dto.differential_parent_1_id,
                input_dto.differential_parent_2_id,
                input_dto.operation,
            ],
        )
        child_2_test_method = transformation.build_test_method_from_io(
            starter_code=input_dto.starter_code,
            io_pairs=cast(list[dict[str, Any]], child_2_io_pairs),
            parent_ids=[
                input_dto.differential_parent_2_id,
                input_dto.differential_parent_1_id,
                input_dto.operation,
            ],
        )

        result = OperatorOutput(
            results=[
                OperatorResult(
                    snippet=child_1_test_method, metadata={"io_pairs": child_1_io_pairs}
                ),
                OperatorResult(
                    snippet=child_2_test_method, metadata={"io_pairs": child_2_io_pairs}
                ),
            ]
        )
        return result

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        match input_dto:
            case DifferentialCrossoverInput():
                return self._handle_crossover(input_dto)
            case DifferentialGenScriptInput():
                return self._handle_generation_script(input_dto)
            case _:
                raise UnsupportedOperatorInput(
                    type(input_dto), getattr(input_dto, "operation", None)
                )


__all__ = [
    "DifferentialInputOutput",
    "DifferentialGenScriptInput",
    "DifferentialCrossoverInput",
    "DifferentialLLMOperator",
]
