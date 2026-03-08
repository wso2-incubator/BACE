# src/infrastructure/languages/ballerina.py
"""
Ballerina implementation of the ILanguage protocol.

Refactored to enforce Separation of Concerns. All dense AST parsing and 
code generation logic has been extracted to `utils/ballerina_parser.py` and 
`utils/ballerina_codegen.py`. 

This class now acts as an orchestration facade and execution configuration registry.
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.language import ILanguage
from . import codegen as ballerina_codegen
from . import parser as ballerina_parser
from infrastructure.sandbox.ballerina_analyzer import BallerinaTestAnalyzer
from infrastructure.sandbox.types import BasicExecutionResult


class BallerinaLanguage(ILanguage):
    """
    Adapter for Ballerina-specific code operations and execution parameters.
    """

    def __init__(self, bal_executable: str = "bal") -> None:
        self._block_pattern = re.compile(r"```[Bb]allerina\s*([\s\S]+?)\s*```")
        self.bal_executable = bal_executable
        
        # Composition: Test analyzing is delegated entirely.
        self._test_analyzer = BallerinaTestAnalyzer()
        
        logger.info(f"Initialized Ballerina language adapter with exe: {self.bal_executable}")

    # ------------------------------------------------------------------
    # Execution Commands (Duck-typed Plugin Properties)
    # ------------------------------------------------------------------

    def get_execution_command(self, file_path: str) -> List[str]:
        """Provides the SubprocessSandbox with the command to run standard code."""
        return [self.bal_executable, "run", file_path]

    def get_test_command(self, test_file_path: str, result_xml_path: str, **kwargs: Any) -> List[str]:
        """Provides the SubprocessSandbox with the command to run tests."""
        # Ballerina tests are usually run on modules, but for single files we run `bal test <file>`
        # For simplicity, we define the command here. The orchestrator should pass `result_xml_path` if supported,
        # but Ballerina standard test output might not directly support --junitxml mapping like pytest.
        # The analyzer handles stdout parsing.
        return [self.bal_executable, "test", test_file_path]

    def parse_test_results(
        self, raw_result: BasicExecutionResult, xml_content: Optional[str] = None
    ) -> EvaluationResult:
        """
        Parses the raw stdout/JSON returned by the SubprocessSandbox into a structured EvaluationResult.
        """
        return self._test_analyzer.analyze_test_output(raw_result)

    # ------------------------------------------------------------------
    # ILanguage interface implementations
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        return "ballerina"

    def extract_code_blocks(self, response: str) -> List[str]:
        matches = self._block_pattern.findall(response)
        if matches:
            valid_blocks = [code.strip() for code in matches if self.is_syntax_valid(code.strip())]
            if valid_blocks:
                return valid_blocks

        cleaned = response.strip()
        if cleaned and self.is_syntax_valid(cleaned):
            return [cleaned]

        logger.debug("No valid Ballerina code blocks found in response")
        return []

    def is_syntax_valid(self, code: str) -> bool:
        return ballerina_parser.is_syntax_valid(code)

    def extract_test_names(self, test_code: str) -> List[str]:
        return ballerina_parser.extract_test_names(test_code)

    def split_tests(self, test_code: str) -> List[str]:
        return ballerina_parser.split_tests(test_code)

    def compose_test_script(self, code_snippet: str, test_snippet: str) -> str:
        return ballerina_codegen.compose_test_script(code_snippet, test_snippet)

    def compose_evaluation_script(self, code_snippet: str, input_data: str) -> str:
        return ballerina_codegen.compose_evaluation_script(code_snippet, input_data)

    def generate_test_case(
        self, input_str: str, output_str: str, starter_code: str, test_number: int
    ) -> str:
        return ballerina_codegen.generate_test_case(
            input_str, output_str, starter_code, test_number
        )

    def compose_generator_script(self, generator_code: str, num_inputs: int) -> str:
        return ballerina_codegen.compose_generator_script(generator_code, num_inputs)

    def remove_main_block(self, code: str) -> str:
        return ballerina_parser.remove_main_block(code)

    def normalize_code(self, code: str) -> str:
        return ballerina_parser.normalize_code(code)

    def contains_starter_code(self, code: str, starter_code: str) -> bool:
        norm_code = self.normalize_code(code)
        norm_starter = self.normalize_code(starter_code)

        if norm_starter in norm_code:
            return True

        starter_match = ballerina_parser.FUNCTION_PATTERN.search(starter_code)
        if not starter_match:
            return False

        starter_func_name = starter_match.group(2)
        code_funcs = ballerina_parser.FUNCTION_PATTERN.findall(code)
        code_func_names = [match[1] for match in code_funcs]
        return starter_func_name in code_func_names

    def get_structural_metadata(self, code: str) -> Dict[str, Any]:
        return ballerina_parser.get_structural_metadata(code)

    def parse_test_inputs(self, outputs: str) -> List[Dict[str, Any]]:
        return ballerina_parser.parse_test_inputs(outputs)
