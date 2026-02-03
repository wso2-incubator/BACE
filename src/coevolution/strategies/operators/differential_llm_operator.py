"""Unittest operator implementation and DTOs.

Contains `UnittestLLMOperator` and the DTOs used for unittest-based
test generation and manipulation.
"""

from dataclasses import dataclass
from typing import Any, TypedDict

from loguru import logger

from coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    BaseOperatorInput,
    InitialInput,
    IOperator,
    OperatorOutput,
    OperatorResult,
)
from coevolution.utils.prompts import DIFFERENTIAL_INPUT_GENERATOR_PROMPT
from infrastructure.code_preprocessing import (
    CodeParsingError,
    CodeTransformationError,
    composition,
    transformation,
)

from .base_llm_operator import BaseLLMOperator, UnsupportedOperatorInput, llm_retry


class DifferentialInputOutput(TypedDict):
    inputdata: dict[str, Any]
    output: Any


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

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError))
    def generate_initial_snippets(self, input_dto: InitialInput) -> OperatorOutput:
        """
        Initially there are no differential tests to generate.

        Note: Previously returned test class block, but with pytest standalone
        functions, TestPopulation builds its own test block.
        """

        logger.debug("Generating empty initial output for differential tests")
        result = OperatorOutput(results=[])
        return result

    def get_test_method_from_io(
        self,
        starter_code: str,
        io_pairs: list[DifferentialInputOutput],
        code_parent_ids: list[str],
        io_index: int,
    ) -> str:
        """
        Build a test method from differential input-output pairs.

        Uses the generic test generation system that supports both class methods
        and standalone functions.

        Note: This is not a LLM operation or a genetic operation.
        """
        from infrastructure.code_preprocessing.test_generation import (
            generate_pytest_test,
        )

        logger.debug(
            f"Building test method from {len(io_pairs)} IO pairs for parents {code_parent_ids}"
        )

        # Handle empty IO pairs
        if len(io_pairs) == 0:
            logger.error("Cannot build test method from empty IO pairs list")
            raise ValueError("IO pairs list cannot be empty")

        # Combine all IO pairs into a single test
        # For differential tests, we typically have one IO pair per test to isolate divergences
        if len(io_pairs) != 1:
            logger.warning(
                f"Expected 1 IO pair for differential test, got {len(io_pairs)}. Using first pair only."
            )

        io_pair = io_pairs[0]
        input_data = io_pair["inputdata"]
        expected_output = io_pair["output"]

        # Convert input dict to newline-separated string format
        # e.g., {"x": 5, "y": 3} -> "5\n3"
        input_lines = [str(v) for v in input_data.values()]
        input_str = "\n".join(input_lines)

        # Convert output to string
        output_str = str(expected_output)

        # Generate unique test number from parent IDs and index
        # This ensures test names are unique and traceable
        test_number = hash(f"{'_'.join(code_parent_ids)}_{io_index}") % 10000

        # Use the generic test generation
        test_function = generate_pytest_test(
            input_str, output_str, starter_code, test_number
        )

        logger.debug(f"Built test method with length {len(test_function)}")
        return test_function

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

        logger.trace(f"PROMPT:\n{prompt}")
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
        logger.trace(f"Generated differential input script:\n{generated_script}")
        result = OperatorOutput(results=[OperatorResult(snippet=generated_script)])
        logger.info("Successfully generated differential input script")
        return result

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        operation = getattr(input_dto, "operation", "unknown")
        logger.info(
            f"Applying differential operator for operation '{operation}' with input type {type(input_dto).__name__}"
        )
        match input_dto:
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
    "DifferentialLLMOperator",
]
