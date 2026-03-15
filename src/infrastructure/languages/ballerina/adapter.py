# src/infrastructure/languages/ballerina/adapter.py
"""
Ballerina implementation of the ILanguage protocol.

Refactored to enforce Separation of Concerns via the Facade pattern.
This class acts as an orchestration facade and execution configuration registry.
"""

import re
from typing import Any, Dict, List

from loguru import logger

from coevolution.core.interfaces.language import (
    ILanguage,
    ICodeParser,
    IScriptComposer,
    ILanguageRuntime,
)
from .analyzer import BallerinaTestAnalyzer
from . import codegen as ballerina_codegen
from . import parser as ballerina_parser


class BallerinaParser(ICodeParser):
    def __init__(self) -> None:
        self._block_pattern = re.compile(r"```[Bb]allerina\s*([\s\S]+?)\s*```")

    def extract_code_blocks(self, response: str) -> List[str]:
        matches = self._block_pattern.findall(response)
        if matches:
            valid_blocks = [
                code.strip() for code in matches if self.is_syntax_valid(code.strip())
            ]
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

    def get_docstring(self, code: str) -> str | None:
        return None

    def get_function_signature(self, code: str) -> Dict[str, str]:
        return ballerina_parser.get_function_signature(code)

    def parse_public_test(
        self, input_str: str, output_str: str, starter_code: str
    ) -> tuple[Dict[str, Any], Any]:
        """Parse raw Ballerina public test input/output."""
        sig = self.get_function_signature(starter_code)
        param_names = list(sig.keys())

        # Ballerina output is usually just a JSON literal
        try:
            import json
            output_val = json.loads(output_str)
        except Exception:
            output_val = output_str

        # Ballerina input is often func(arg1, arg2)
        match = re.search(r"\(+(.*)\)+", input_str.strip())
        args_str = match.group(1) if match else input_str
        
        # Simple split by comma for now
        raw_args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
        
        input_dict = {}
        for i, name in enumerate(param_names):
            if i < len(raw_args):
                try:
                    import json
                    input_dict[name] = json.loads(raw_args[i])
                except Exception:
                    input_dict[name] = raw_args[i]
            else:
                input_dict[name] = None
                
        return input_dict, output_val


class BallerinaComposer(IScriptComposer):
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


class BallerinaRuntime(ILanguageRuntime):
    def __init__(self, bal_executable: str):
        self.bal_executable = bal_executable

    @property
    def file_extension(self) -> str:
        return ".bal"

    def get_execution_command(self, file_path: str) -> List[str]:
        return [self.bal_executable, "run", file_path]

    def get_test_command(
        self, test_file_path: str, result_xml_path: str, **kwargs: Any
    ) -> List[str]:
        return [self.bal_executable, "test", test_file_path]


class BallerinaLanguage(ILanguage):
    """
    Facade adapter for Ballerina-specific code operations and execution parameters.
    """

    def __init__(self, bal_executable: str = "bal") -> None:
        self.bal_executable = bal_executable

        self._parser = BallerinaParser()
        self._composer = BallerinaComposer()
        self._runtime = BallerinaRuntime(self.bal_executable)
        self._analyzer = BallerinaTestAnalyzer()

        logger.info(
            f"Initialized Ballerina language facade adapter with exe: {self.bal_executable}"
        )

    @property
    def language(self) -> str:
        return "ballerina"

    @property
    def parser(self) -> BallerinaParser:
        return self._parser

    @property
    def composer(self) -> BallerinaComposer:
        return self._composer

    @property
    def runtime(self) -> BallerinaRuntime:
        return self._runtime

    @property
    def analyzer(self) -> BallerinaTestAnalyzer:
        return self._analyzer
