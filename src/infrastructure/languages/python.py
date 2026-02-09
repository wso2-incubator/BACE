# src/infrastructure/adapters/python.py
"""
Python implementation of the ILanguageAdapter protocol.
"""

import ast
import re
from typing import Any, Dict, List

from loguru import logger

from coevolution.core.interfaces.language import (
    ILanguageAdapter,
    LanguageParsingError,
    LanguageTransformationError,
)
from infrastructure.code_preprocessing.composition import compose_lcb_output_script
from infrastructure.code_preprocessing.exceptions import (
    CodeParsingError,
    CodeTransformationError,
)
from infrastructure.code_preprocessing.test_generation import generate_pytest_test
from infrastructure.code_preprocessing.transformation import remove_if_main_block


class PythonLanguageAdapter(ILanguageAdapter):
    """
    Adapter for Python-specific code operations.
    """

    def __init__(self) -> None:
        self._block_pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")

    def extract_code_blocks(self, response: str) -> List[str]:
        """
        Extract Python code blocks from LLM response.
        """
        matches = self._block_pattern.findall(response)

        if not matches:
            # Check if the entire response is valid Python code (no markdown)
            stripped = response.strip()
            if self.is_syntax_valid(stripped):
                return [stripped]

            logger.debug(
                f"No Python code blocks found in response: {response[:100]}..."
            )
            return []

        # Filter out syntactically invalid blocks
        valid_blocks = [block for block in matches if self.is_syntax_valid(block)]

        if not valid_blocks and matches:
            logger.debug("Python blocks found but all had syntax errors.")

        return valid_blocks

    def is_syntax_valid(self, code: str) -> bool:
        """
        Validate Python syntax using ast.parse.
        """
        try:
            ast.parse(code)
            return True
        except (SyntaxError, ValueError, OverflowError):
            return False

    def extract_test_names(self, test_code: str) -> List[str]:
        """
        Extract pytest-style test function names (starting with test_).
        """
        test_names = []
        try:
            tree = ast.parse(test_code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_names.append(node.name)
        except Exception as e:
            logger.debug(f"Failed to extract test names: {e}")

        return test_names

    def split_tests(self, test_code: str) -> List[str]:
        """
        Split a test block into individual standalone test functions.
        """
        test_functions: List[str] = []

        try:
            tree = ast.parse(test_code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    try:
                        function_code = ast.get_source_segment(test_code, node)
                        if function_code is None:
                            function_code = ast.unparse(node)
                        test_functions.append(function_code)
                    except Exception:
                        try:
                            test_functions.append(ast.unparse(node))
                        except Exception:
                            continue
        except Exception as e:
            logger.debug(f"Failed to split tests: {e}")

        return test_functions

    def compose_test_script(self, code_snippet: str, test_snippet: str) -> str:
        """
        Compose a pytest script by combining code and test snippet.
        """
        parts = []

        # Ensure pytest is imported if needed
        has_pytest = "import pytest" in code_snippet or "import pytest" in test_snippet
        if not has_pytest:
            parts.append("import pytest")
            parts.append("")

        parts.append(code_snippet.strip())
        parts.append("")
        parts.append(test_snippet.strip())
        parts.append("")

        # Add pytest main execution entry point
        parts.append('if __name__ == "__main__":')
        parts.append("    import sys")
        parts.append('    pytest.main([__file__, "-v"])')

        return "\n".join(parts)

    def compose_evaluation_script(self, code_snippet: str, input_data: str) -> str:
        """
        Compose an evaluation script using compose_lcb_output_script utility.
        """
        try:
            return compose_lcb_output_script(code_snippet, input_data)
        except CodeParsingError as e:
            raise LanguageParsingError(str(e)) from e
        except CodeTransformationError as e:
            raise LanguageTransformationError(str(e)) from e

    def generate_test_case(
        self, input_str: str, output_str: str, starter_code: str, test_number: int
    ) -> str:
        """
        Generate a pytest test case using generate_pytest_test utility.
        """
        return generate_pytest_test(input_str, output_str, starter_code, test_number)

    def remove_main_block(self, code: str) -> str:
        """
        Remove the 'if __name__ == "__main__":' block.
        """
        return remove_if_main_block(code)

    def normalize_code(self, code: str) -> str:
        """
        Normalize Python code by removing comments, extra whitespace, and docstrings.
        """
        # Remove single-line comments
        text = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

        # Remove multi-line strings/docstrings (triple quotes)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        return "\n".join(lines)

    def contains_starter_code(self, code: str, starter_code: str) -> bool:
        """
        Check if starter code (possibly incomplete/invalid) is present in complete code.
        """
        # Normalize both strings
        normalized_code = self.normalize_code(code)
        normalized_starter = self.normalize_code(starter_code)

        # If starter is empty, it's contained
        if not normalized_starter.strip():
            return True

        # Strategy 1: Direct substring match after normalization
        if normalized_starter in normalized_code:
            return True

        # Strategy 2: Check for structural similarity (signatures)
        def extract_signatures(text: str) -> List[str]:
            signatures = []
            class_names = re.findall(r"class\s+(\w+)", text)
            signatures.extend([f"class {name}" for name in class_names])
            func_names = re.findall(r"def\s+(\w+)\s*\(", text)
            signatures.extend([f"def {name}" for name in func_names])
            return signatures

        code_signatures = extract_signatures(normalized_code)
        starter_signatures = extract_signatures(normalized_starter)

        if not starter_signatures:
            return False

        return all(sig in code_signatures for sig in starter_signatures)

    def get_structural_metadata(self, code: str) -> Dict[str, Any]:
        """
        Extract Python structural metadata (imports, classes, functions).
        """
        metadata: Dict[str, list[str]] = {"imports": [], "classes": [], "functions": []}

        try:
            tree = ast.parse(code)
            lines = code.split("\n")

            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    start = node.lineno - 1
                    end = (
                        (node.end_lineno - 1) if node.end_lineno is not None else start
                    )
                    metadata["imports"].extend(lines[start : end + 1])
                elif isinstance(node, ast.ClassDef):
                    metadata["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    metadata["functions"].append(node.name)

            metadata["imports"] = list(dict.fromkeys(metadata["imports"]))
        except Exception as e:
            logger.debug(f"Failed to extract structural metadata: {e}")

        return metadata

    def parse_test_inputs(self, outputs: str) -> List[Dict[str, Any]]:
        """
        Parses Python literal outputs using ast.literal_eval.
        """
        import ast

        try:
            val = ast.literal_eval(outputs)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return [val]
            return []
        except Exception as e:
            logger.debug(f"Failed to parse test inputs as literal: {e}")
            return []
