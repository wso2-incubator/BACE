# src/infrastructure/languages/python/adapter.py
"""
Python implementation of the ILanguage protocol.

Refactored to enforce Separation of Concerns via the Facade pattern.
This class acts as an orchestration facade and execution configuration registry.
"""

import ast
import re
import sys
from typing import Any, Dict, List, Optional

from loguru import logger

from coevolution.core.interfaces.language import (
    ILanguage,
    ICodeParser,
    IScriptComposer,
    ILanguageRuntime,
)
from .analyzer import PythonTestAnalyzer
from . import ast as python_ast
from . import codegen as python_codegen


class PythonParser(ICodeParser):
    def __init__(self) -> None:
        self._block_pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")

    def extract_code_blocks(self, response: str) -> List[str]:
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

    def remove_main_block(self, code: str) -> str:
        return python_ast.remove_main_block(code)

    def get_structural_metadata(self, code: str) -> Dict[str, Any]:
        return python_ast.get_structural_metadata(code)

    def parse_test_inputs(self, outputs: str) -> List[Dict[str, Any]]:
        return python_ast.parse_test_inputs(outputs)

    def get_docstring(self, code: str) -> str:
        return python_ast.get_docstring(code)

    def get_function_signature(self, code: str) -> Dict[str, str]:
        return python_ast.get_function_signature(code)

    def parse_public_test(
        self, input_str: str, output_str: str, starter_code: str
    ) -> tuple[Dict[str, Any], Any]:
        """Parse raw Python public test input/output."""
        sig = python_ast.parse_method_signature(starter_code)
        param_names = [p[0] for p in sig.params]

        # 1. Parse raw output
        try:
            output_val = ast.literal_eval(output_str)
        except (ValueError, SyntaxError):
            output_val = output_str

        # 2. Parse raw input
        # Stdin style: input_str: str -> str
        if python_ast.is_stdin_signature(sig):
            return {param_names[0]: input_str}, output_val

        # Functional style: lines are positional args
        lines = [line.strip() for line in input_str.split("\n") if line.strip()]
        raw_vals = []
        for line in lines:
            try:
                raw_vals.append(ast.literal_eval(line))
            except (ValueError, SyntaxError):
                raw_vals.append(line)

        input_dict = {}
        if len(raw_vals) == len(param_names):
            for i, name in enumerate(param_names):
                input_dict[name] = raw_vals[i]
        elif len(param_names) == 1:
            input_dict[param_names[0]] = raw_vals[0] if raw_vals else input_str
        else:
            # Positional fallback
            for i, name in enumerate(param_names):
                if i < len(raw_vals):
                    input_dict[name] = raw_vals[i]
                else:
                    input_dict[name] = None

        return input_dict, output_val

    def normalize_code(self, code: str) -> str:
        text = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def contains_starter_code(self, code: str, starter_code: str) -> bool:
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


class PythonComposer(IScriptComposer):
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


class PythonRuntime(ILanguageRuntime):
    def __init__(self, python_exe: str):
        self.python_exe = python_exe

    @property
    def file_extension(self) -> str:
        return ".py"

    def get_execution_command(self, file_path: str) -> List[str]:
        return [self.python_exe, file_path]

    def get_test_command(
        self, test_file_path: str, result_xml_path: str, **kwargs: Any
    ) -> List[str]:
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


class PythonLanguage(ILanguage):
    """
    Facade adapter for Python-specific code operations and execution parameters.
    """

    def __init__(self, python_exe: Optional[str] = None) -> None:
        self.python_exe = python_exe or sys.executable

        self._parser = PythonParser()
        self._composer = PythonComposer()
        self._runtime = PythonRuntime(self.python_exe)
        self._analyzer = PythonTestAnalyzer()

        logger.info(
            f"Initialized Python language Facade adapter with exe: {self.python_exe}"
        )

    @property
    def language(self) -> str:
        return "python"

    @property
    def parser(self) -> PythonParser:
        return self._parser

    @property
    def composer(self) -> PythonComposer:
        return self._composer

    @property
    def runtime(self) -> PythonRuntime:
        return self._runtime

    @property
    def analyzer(self) -> PythonTestAnalyzer:
        return self._analyzer
