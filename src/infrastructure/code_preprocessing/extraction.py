"""Extract structured data from unstructured strings (LLM responses, problem prompts)."""

import ast
import re
from typing import Dict, List, Tuple, TypedDict

from loguru import logger

from .analysis import validate_code_syntax
from .exceptions import CodeParsingError


class CodeStructure(TypedDict):
    """Type definition for code structure extraction results."""

    import_lines: List[str]
    function_definitions: Dict[str, Tuple[int, int]]
    class_definitions: Dict[str, Tuple[int, int]]


def extract_function_name_from_problem(problem_prompt: str) -> str:
    """
    Extract the function name from the problem prompt.

    Args:
        problem_prompt: Problem description containing function definition

    Returns:
        Function name if found, empty string otherwise

    Example:
        >>> prompt = "def calculate(x, y):\\n    pass"
        >>> extract_function_name_from_problem(prompt)
        'calculate'
    """
    lines = problem_prompt.split("\n")
    for line in lines:
        if line.strip().startswith("def "):
            func_name = line.strip().split("def ")[1].split("(")[0]
            return func_name
    return ""


def extract_all_code_blocks_from_response(response: str) -> List[str]:
    """
    Extract all Python code blocks (```python ... ```) from the LLM response.

    Args:
        response: LLM response text potentially containing markdown code blocks

    Returns:
        List of code strings from all ```python blocks found

    Raises:
        CodeParsingError: If no Python code blocks are found in the response

    Example:
        >>> response = "Here's code:\\n```python\\nprint('hi')\\n```"
        >>> extract_all_code_blocks_from_response(response)
        ["print('hi')"]
    """
    pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")
    matches = pattern.findall(response)

    if not matches:
        logger.debug(f"No Python code blocks found in response: {response}")
        raise CodeParsingError(
            "No Python code blocks found in LLM response. "
            "Expected format: ```python\\n<code>\\n```"
        )

    if not all(validate_code_syntax(code) for code in matches):
        logger.debug(f"One or more extracted code blocks have syntax errors: {matches}")
        raise CodeParsingError("One or more extracted code blocks have syntax errors.")

    return matches


def extract_code_block_from_response(response: str) -> str:
    """
    Extract the first Python code block (```python ... ```) from the LLM response.
    If no valid code block is found, raises CodeParsingError.
    If the entire response is valid code, it is returned as is.

    Args:
        response: LLM response text potentially containing markdown code blocks

    Returns:
        Code string from first ```python block

    Raises:
        CodeParsingError: If no Python code block is found in the response

    Example:
        >>> response = "Solution:\\n```python\\ndef foo(): pass\\n```"
        >>> extract_code_block_from_response(response)
        'def foo(): pass'
    """
    pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")
    match = pattern.search(response)
    if match:
        code_block = match.group(1)
        if validate_code_syntax(code_block):
            return code_block
        else:
            logger.debug(f"Extracted code block has syntax errors: {code_block}")
            raise CodeParsingError("Extracted code block has syntax errors.")

    # Check if the entire response is code (no markdown)
    stripped_response = response.strip()
    if validate_code_syntax(stripped_response):
        return stripped_response

    # No code block found - raise explicit error instead of silent fallback
    logger.debug(f"No Python code block found in response: {response}")
    raise CodeParsingError(
        "No Python code block found in LLM response. "
        "Expected format: ```python\\n<code>\\n```"
    )


def extract_code_structure(code_string: str) -> CodeStructure:
    """
    Extract code structure: all imports, functions, and classes using AST.

    Args:
        code_string: Python source code to analyze

    Returns:
        Dictionary containing:
        - import_lines: List of import statement strings
        - function_definitions: Dict mapping function names to (start_line, end_line)
        - class_definitions: Dict mapping class names to (start_line, end_line)

    Raises:
        CodeParsingError: If code has syntax errors

    Example:
        >>> code = "import os\\n\\ndef foo():\\n    pass"
        >>> result = extract_code_structure(code)
        >>> 'foo' in result['function_definitions']
        True
    """
    lines = code_string.split("\n")
    import_lines: List[str] = []
    function_definitions: Dict[str, Tuple[int, int]] = {}
    class_definitions: Dict[str, Tuple[int, int]] = {}

    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        logger.debug(f"Failed to parse code string: {code_string}")
        raise CodeParsingError(f"Failed to parse code: {e}") from e

    for node in tree.body:
        start_idx = node.lineno - 1
        end_idx = (node.end_lineno - 1) if node.end_lineno is not None else start_idx

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_lines.extend(lines[start_idx : end_idx + 1])
        elif isinstance(node, ast.FunctionDef):
            function_definitions[node.name] = (start_idx, end_idx)
        elif isinstance(node, ast.ClassDef):
            class_definitions[node.name] = (start_idx, end_idx)

    return {
        "import_lines": list(dict.fromkeys(import_lines)),  # Remove duplicates
        "function_definitions": function_definitions,
        "class_definitions": class_definitions,
    }


def extract_test_functions_code(test_code: str) -> List[str]:
    """
    Extract standalone pytest test functions from code.

    Unlike extract_test_methods_code which extracts test methods from unittest classes,
    this function extracts top-level test functions (pytest style).

    Args:
        test_code: Python source code containing pytest test functions

    Returns:
        List of test function code strings (each starting with 'def test_')

    Raises:
        CodeParsingError: If code has syntax errors

    Example:
        >>> code = \"\"\"
        ... import pytest
        ...
        ... def test_addition():
        ...     assert 1 + 1 == 2
        ...
        ... def test_subtraction():
        ...     assert 5 - 3 == 2
        ... \"\"\"
        >>> functions = extract_test_functions_code(code)
        >>> len(functions)
        2
    """
    test_functions: List[str] = []

    try:
        tree = ast.parse(test_code)
    except SyntaxError as e:
        logger.debug(f"Failed to parse test code: {test_code}")
        raise CodeParsingError(f"Failed to parse test code: {e}") from e

    for node in tree.body:
        # Look for top-level functions starting with 'test_'
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # Get the source code for this function
            try:
                function_code = ast.get_source_segment(test_code, node)
                if function_code is None:
                    # Fallback to unparsing if get_source_segment fails
                    function_code = ast.unparse(node)
                test_functions.append(function_code)
            except Exception as e:
                logger.warning(
                    f"Failed to extract source for test function '{node.name}': {e}"
                )
                # Try unparsing as fallback
                try:
                    function_code = ast.unparse(node)
                    test_functions.append(function_code)
                except Exception as unparse_error:
                    logger.error(
                        f"Failed to unparse test function '{node.name}': {unparse_error}"
                    )
                    continue

    if not test_functions:
        logger.debug(f"No test functions found in code: {test_code[:200]}...")

    return test_functions
