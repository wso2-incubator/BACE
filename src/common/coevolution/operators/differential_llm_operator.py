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

        logger.debug("Generating initial test class block for differential tests")
        test_class_block = transformation.setup_unittest_class_from_starter_code(
            input_dto.starter_code
        )
        result = OperatorOutput(results=[])
        logger.debug(
            f"Generated initial test class block with length {len(test_class_block)}"
        )
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

        logger.debug(
            f"Building test method from {len(io_pairs)} IO pairs for parents {code_parent_ids}"
        )
        method_code = transformation.build_test_method_from_io(
            starter_code, cast(list[dict[str, Any]], io_pairs), code_parent_ids
        )
        logger.debug(f"Built test method with length {len(method_code)}")
        return method_code

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def _handle_generation_script(
        self, input_dto: DifferentialGenScriptInput
    ) -> OperatorOutput:
        """Generate a script to generate inputs to differentiate between two equivalent code snippets."""

        logger.info(
            f"Generating differential input script for {len(input_dto.passing_differential_test_io_pairs)} existing tests"
        )
        prompt = DIFFERENTIAL_INPUT_GENERATOR_PROMPT.format(
            question_content=input_dto.question_content,
            code_snippet_P=input_dto.equivalent_code_snippet_1,
            code_snippet_Q=input_dto.equivalent_code_snippet_2,
            current_tests=input_dto.passing_differential_test_io_pairs,
        )

        logger.debug(f"PROMPT:\n{prompt}")
        logger.debug(f"Generated prompt with length {len(prompt)}")

        llm_response = self._llm.generate(prompt)
        logger.debug(f"Received LLM response with length {len(llm_response)}")

        code_block = self._extract_code_block(llm_response)
        logger.debug(f"Extracted code block with length {len(code_block)}")
        generated_script = transformation.remove_if_main_block(code_block)
        generated_script += (
            f"\nprint(generate_test_inputs({input_dto.num_inputs_to_generate}))"
        )
        logger.debug(f"Final generated script with length {len(generated_script)}")
        logger.debug(f"Generated differential input script:\n{generated_script}")
        result = OperatorOutput(results=[OperatorResult(snippet=generated_script)])
        logger.info("Successfully generated differential input script")
        return result

    def _handle_crossover(
        self, input_dto: DifferentialCrossoverInput
    ) -> OperatorOutput:
        """
        No LLM crossover for differential tests; simply cross combines test methods from both parents.
        Produces two children by swapping half the methods.
        """

        logger.info(
            f"Performing crossover between differential parents {input_dto.differential_parent_1_id} and {input_dto.differential_parent_2_id}"
        )
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

        logger.debug(
            f"Child 1 IO pairs: {len(child_1_io_pairs)}, Child 2 IO pairs: {len(child_2_io_pairs)}"
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

        logger.debug(
            f"Generated child 1 test method with length {len(child_1_test_method)}"
        )
        logger.debug(
            f"Generated child 2 test method with length {len(child_2_test_method)}"
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
        logger.info("Successfully performed differential crossover")
        return result

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        operation = getattr(input_dto, "operation", "unknown")
        logger.info(
            f"Applying differential operator for operation '{operation}' with input type {type(input_dto).__name__}"
        )
        match input_dto:
            case DifferentialCrossoverInput():
                result = self._handle_crossover(input_dto)
            case DifferentialGenScriptInput():
                result = self._handle_generation_script(input_dto)
            case _:
                logger.error(
                    f"Unsupported operation input: {type(input_dto)} for operation {operation}"
                )
                raise UnsupportedOperatorInput(
                    type(input_dto), getattr(input_dto, "operation", None)
                )
        return result


__all__ = [
    "DifferentialInputOutput",
    "DifferentialGenScriptInput",
    "DifferentialCrossoverInput",
    "DifferentialLLMOperator",
]
