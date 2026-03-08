# src/infrastructure/languages/python.py
"""
Python implementation of the ILanguage protocol.

Refactored to enforce Separation of Concerns. All dense AST parsing and 
code generation logic has been extracted to `utils/python_ast.py` and 
`utils/python_codegen.py`. 

This class now acts as an orchestration facade and execution configuration registry.
"""

import re
import sys
from typing import Any, Dict, List, Optional

from loguru import logger

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.language import ILanguage
from . import ast as python_ast
from . import codegen as python_codegen
from infrastructure.sandbox.analyzer import PytestXmlAnalyzer
from infrastructure.sandbox.types import BasicExecutionResult


class PythonLanguage(ILanguage):
    """
    Adapter for Python-specific code operations and execution parameters.
    """

    def __init__(self, python_exe: Optional[str] = None) -> None:
        self._block_pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")
        self.python_exe = python_exe or sys.executable
        
        # Composition: Test analyzing is delegated entirely.
        self._test_analyzer = PytestXmlAnalyzer()
        
        logger.info(f"Initialized Python language adapter with exe: {self.python_exe}")

    # ------------------------------------------------------------------
    # Execution Commands (Duck-typed Plugin Properties)
    # ------------------------------------------------------------------

    def get_execution_command(self, file_path: str) -> List[str]:
        """Provides the SubprocessSandbox with the command to run standard code."""
        return [self.python_exe, file_path]

    def get_test_command(self, test_file_path: str, result_xml_path: str, **kwargs: Any) -> List[str]:
        """Provides the SubprocessSandbox with the command to run tests."""
        cmd = [
            self.python_exe,
            "-m",
            "pytest",
            test_file_path,
            "--junitxml",
            result_xml_path,
            "--color=no",
            "-o",
            "console_output_style=classic",
        ]
        timeout = kwargs.get("timeout")
        if timeout is not None:
            cmd.append(f"--timeout={timeout}")
        return cmd

    def parse_test_results(
        self, raw_result: BasicExecutionResult, xml_content: Optional[str] = None
    ) -> EvaluationResult:
        """
        Parses the raw JSON/Stdout/XML returned by the SubprocessSandbox into a structured EvaluationResult.
        """
        return self._test_analyzer.analyze_pytest_xml(xml_content, raw_result)

    # ------------------------------------------------------------------
    # ILanguage interface implementations
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        return "python"

    def extract_code_blocks(self, response: str) -> List[str]:
        """Extract Python code blocks from an LLM response."""
        matches = self._block_pattern.findall(response)

        if not matches:
            stripped = response.strip()
            if python_ast.is_syntax_valid(stripped):
                return [stripped]
            logger.debug(
                f"No Python code blocks found in response: {response[:100]}..."
            )
            return []

        valid_blocks = [block for block in matches if python_ast.is_syntax_valid(block)]
        if not valid_blocks and matches:
            logger.debug("Python blocks found but all had syntax errors.")
        return valid_blocks

    def is_syntax_valid(self, code: str) -> bool:
        return python_ast.is_syntax_valid(code)

    def extract_test_names(self, test_code: str) -> List[str]:
        return python_ast.extract_test_names(test_code)

    def split_tests(self, test_code: str) -> List[str]:
        return python_ast.split_tests(test_code)

    def compose_test_script(self, code_snippet: str, test_snippet: str) -> str:
        return python_codegen.compose_test_script(code_snippet, test_snippet)

    def compose_evaluation_script(self, code_snippet: str, input_data: str) -> str:
        return python_codegen.compose_evaluation_script(code_snippet, input_data)

    def generate_test_case(
        self, input_str: str, output_str: str, starter_code: str, test_number: int
    ) -> str:
        sig = python_ast.parse_method_signature(starter_code)
        if python_ast.is_stdin_signature(sig):
            return python_codegen.gen_stdin_test(
                input_str,
                output_str,
                sig.class_name,
                sig.method_name,
                test_number,
                sig.is_standalone,
            )
        return python_codegen.gen_functional_test(
            input_str,
            output_str,
            sig.class_name,
            sig.method_name,
            test_number,
            sig.is_standalone,
        )

    def compose_generator_script(self, generator_code: str, num_inputs: int) -> str:
        return python_codegen.compose_generator_script(generator_code, num_inputs)

    def remove_main_block(self, code: str) -> str:
        return python_ast.remove_main_block(code)

    def get_structural_metadata(self, code: str) -> Dict[str, Any]:
        return python_ast.get_structural_metadata(code)

    def parse_test_inputs(self, outputs: str) -> List[Dict[str, Any]]:
        return python_ast.parse_test_inputs(outputs)

    def get_docstring(self, code: str) -> str:
        # Assuming we also extract it to python_ast for symmetry
        return python_ast.get_docstring(code)

    def normalize_code(self, code: str) -> str:
        """Normalize Python code by removing comments, docstrings, and extra whitespace."""
        text = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def contains_starter_code(self, code: str, starter_code: str) -> bool:
        """Check if the starter code structure is present in the provided code."""
        normalized_code = self.normalize_code(code)
        normalized_starter = self.normalize_code(starter_code)

        if not normalized_starter.strip():
            return True
        if normalized_starter in normalized_code:
            return True

        def _signatures(text: str) -> List[str]:
            sigs = [f"class {n}" for n in re.findall(r"class\s+(\w+)", text)]
            sigs += [f"def {n}" for n in re.findall(r"def\s+(\w+)\s*\(", text)]
            return sigs

        code_sigs = _signatures(normalized_code)
        starter_sigs = _signatures(normalized_starter)
        if not starter_sigs:
            return False
        return all(s in code_sigs for s in starter_sigs)
