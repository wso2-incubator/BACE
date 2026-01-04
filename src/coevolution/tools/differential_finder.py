"""
Concrete implementation of the Divergence Finder.
Uses the execution sandbox to identify differences between two code snippets.
"""

import ast
from typing import Any

from loguru import logger

from infrastructure.code_preprocessing.analysis import validate_code_syntax
from infrastructure.code_preprocessing.composition import compose_lcb_output_script
from infrastructure.sandbox import SafeCodeSandbox

from ..breeding_strategies.differential_breeding import (
    DifferentialResult,
    IDifferentialFinder,
)


class DifferentialFinder(IDifferentialFinder):
    """
    Executes a generator script to produce inputs, then runs two code snippets
    against those inputs to find discrepancies in their outputs.
    """

    def __init__(self, sandbox: SafeCodeSandbox) -> None:
        self.sandbox = sandbox

    def _generate_output(
        self, code_snippet: str, input_data: dict[str, Any]
    ) -> str | None:
        test_input_formatted = {"inputdata": input_data}
        script = compose_lcb_output_script(code_snippet, str(test_input_formatted))
        exec_result = self.sandbox.execute_code(script)

        if exec_result.error:
            return None

        return exec_result.output.strip()

    def _generate_test_inputs(self, generator_script: str) -> list[dict[str, Any]]:
        if not validate_code_syntax(generator_script):
            logger.warning(
                "Input generator script produced invalid Python code. No test inputs generated."
            )
            logger.debug(f"Invalid generator script:\n{generator_script}")
            return []

        output = self.sandbox.execute_code(generator_script).output.strip()

        try:
            test_inputs: list[dict[str, Any]] = ast.literal_eval(output)
            return test_inputs
        except (SyntaxError, ValueError) as e:
            logger.warning(
                f"Failed to parse input generator output: {e}. No test inputs generated."
            )
            logger.debug(f"Generator output was:\n{output}")
            return []

    def find_differential(
        self,
        code_a_snippet: str,
        code_b_snippet: str,
        input_generator_script: str,
        limit: int = 10,
    ) -> list[DifferentialResult]:
        found_divergences: list[DifferentialResult] = []

        test_inputs = self._generate_test_inputs(input_generator_script)
        if not test_inputs:
            logger.warning("No test inputs generated; skipping differential finding.")
            return found_divergences

        for idx, ti in enumerate(test_inputs):
            code_a_output = self._generate_output(code_a_snippet, ti)
            code_b_output = self._generate_output(code_b_snippet, ti)

            if code_a_output is None or code_b_output is None:
                logger.debug(
                    f"Skipping test input {idx + 1} due to execution error in one of the code snippets."
                )
                continue

            if code_a_output != code_b_output:
                logger.debug(f"Test Input {idx + 1}:\n")
                logger.debug("Discrepancy found!")
                logger.debug(f"Test Input:\n{ti}")
                logger.debug(f"Code 'a' Output: {code_a_output}")
                logger.debug(f"Code 'b' Output: {code_b_output}")
                found_divergences.append(
                    DifferentialResult(
                        input_data=ti,
                        output_a=code_a_output,
                        output_b=code_b_output,
                    )
                )
            if len(found_divergences) >= limit:
                logger.info(
                    f"Reached divergence limit of {limit}. Stopping further checks."
                )
                break
        return found_divergences


__all__ = ["DifferentialFinder"]
