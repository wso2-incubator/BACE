"""LLM service for differential test discovery.

DifferentialLLMOperator is a plain LLM service (not an IOperator).
It is injected into DifferentialDiscoveryOperator and provides two services:

1. generate_script(dto) — ask the LLM to write a Python input-generator script
   that surfaces divergences between two equivalent code snippets.

2. get_test_method_from_io(...) — convert a (input, expected_output) pair into
   a pytest-style test function using the language adapter.

Input generator scripts are always Python regardless of the target language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from loguru import logger

from coevolution.core.interfaces.language import (
    ILanguage,
    LanguageParsingError,
    LanguageTransformationError,
)
from infrastructure.languages import PythonLanguage

from .base_llm_service import BaseLLMService, ILanguageModel, LLMGenerationError, llm_retry

OPERATION_DISCOVERY: str = "discovery"


class DifferentialInputOutput(TypedDict):
    inputdata: dict[str, Any]
    output: Any


@dataclass(frozen=True)
class DifferentialGenScriptInput:
    """Input for the script-generation LLM call."""
    question_content: str
    equivalent_code_snippet_1: str
    equivalent_code_snippet_2: str
    passing_test_cases: list[str]
    num_inputs_to_generate: int = 100


class DifferentialLLMOperator(BaseLLMService):
    """LLM service for differential test generation.

    This is NOT an IOperator — it is an internal service used by
    DifferentialDiscoveryOperator.
    """

    def __init__(self, llm: ILanguageModel, language_adapter: ILanguage) -> None:
        super().__init__(llm, language_adapter)
        # Input generator scripts are always Python regardless of main code language
        self.python_adapter = PythonLanguage()
        logger.debug(
            f"DifferentialLLMOperator: code={language_adapter.language}, "
            "generator-script=Python"
        )

    def _extract_python_code_block(self, response: str) -> str:
        """Extract a Python code block from the LLM response."""
        blocks = self.python_adapter.extract_code_blocks(response)
        return blocks[0] if blocks else response

    @llm_retry((ValueError, LanguageParsingError, LanguageTransformationError, LLMGenerationError))
    def generate_script(self, dto: DifferentialGenScriptInput) -> str:
        """Ask the LLM to write a Python input-generator script.

        The script, when executed, will generate inputs that expose divergences
        between the two equivalent code snippets.

        Returns:
            A ready-to-run Python generator script string.
        """
        logger.info(
            f"Generating differential script for {len(dto.passing_test_cases)} existing tests"
        )
        prompt = self.prompt_manager.render_prompt(
            "operators/differential/gen_script.j2",
            question_content=dto.question_content,
            code_snippet_P=dto.equivalent_code_snippet_1,
            code_snippet_Q=dto.equivalent_code_snippet_2,
            current_tests="\n".join(dto.passing_test_cases),
            language="python",
        )

        response = self._generate(prompt)
        code_block = self._extract_python_code_block(response)
        script = self.python_adapter.compose_generator_script(
            code_block, dto.num_inputs_to_generate
        )
        logger.debug(f"Generated script ({len(script)} chars)")
        return script

    def get_test_method_from_io(
        self,
        starter_code: str,
        io_pairs: list[DifferentialInputOutput],
        code_parent_ids: list[str],
        io_index: int,
    ) -> str:
        """Convert a divergence IO pair into a pytest-style test function.

        Args:
            starter_code: The function signature to test.
            io_pairs: Exactly one (input, expected_output) pair.
            code_parent_ids: IDs of the winner and loser code individuals.
            io_index: Index within the current divergence batch (for unique naming).

        Returns:
            A pytest function string.
        """
        if not io_pairs:
            raise ValueError("io_pairs cannot be empty")
        if len(io_pairs) != 1:
            logger.warning(f"Expected 1 IO pair, got {len(io_pairs)}; using first")

        io_pair = io_pairs[0]
        input_lines = [str(v) for v in io_pair["inputdata"].values()]
        input_str = "\n".join(input_lines)
        output_str = str(io_pair["output"])
        test_number = hash(f"{'_'.join(code_parent_ids)}_{io_index}") % 10000

        return self.language_adapter.generate_test_case(
            input_str, output_str, starter_code, test_number
        )


__all__ = [
    "DifferentialInputOutput",
    "DifferentialGenScriptInput",
    "DifferentialLLMOperator",
    "OPERATION_DISCOVERY",
]
