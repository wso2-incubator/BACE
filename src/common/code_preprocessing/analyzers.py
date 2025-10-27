"""Analyze Python code structure using AST."""

import ast
import logging
from typing import Dict, List, Tuple, TypedDict

from .exceptions import CodeParsingError

# Setup logging
log = logging.getLogger(__name__)


class CodeStructure(TypedDict):
    """Type definition for code structure analysis results."""

    import_lines: List[str]
    function_definitions: Dict[str, Tuple[int, int]]
    class_definitions: Dict[str, Tuple[int, int]]


def parse_code_structure(code_string: str) -> CodeStructure:
    """
    Parse code string to find all imports, functions, and classes using AST.

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
        >>> result = parse_code_structure(code)
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
        log.error(f"Syntax error during parsing: {e}")
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


def extract_test_case_names(test_code: str) -> List[str]:
    """
    Extract test case method names from a unittest test class.

    Args:
        test_code: String containing a unittest test class definition

    Returns:
        List of test method names (methods starting with 'test_')

    Raises:
        CodeParsingError: If test code has syntax errors

    Example:
        >>> code = "class TestFoo(unittest.TestCase):\\n    def test_bar(self): pass"
        >>> extract_test_case_names(code)
        ['test_bar']
    """
    test_cases = []
    try:
        tree = ast.parse(test_code)
    except SyntaxError as e:
        log.error(f"Syntax error parsing test code: {e}")
        raise CodeParsingError(f"Failed to parse test code: {e}") from e

    # Find the first class definition
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Iterate through class methods
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name.startswith(
                    "test_"
                ):
                    test_cases.append(method.name)
            # Stop after first class
            break

    return test_cases


def extract_test_methods_code(test_code: str) -> List[str]:
    """
    Extract the actual code for each test method from a unittest test class.
    Returns a list of code strings for each test method (methods starting with 'test_')
    in the order they appear.

    Args:
        test_code: String containing a unittest test class definition

    Returns:
        List of code strings for each test method

    Raises:
        CodeParsingError: If test code has syntax errors

    Example:
        >>> code = "class TestFoo(unittest.TestCase):\\n    def test_bar(self): pass"
        >>> methods = extract_test_methods_code(code)
        >>> len(methods)
        1
    """
    test_methods_code = []
    try:
        tree = ast.parse(test_code)
    except SyntaxError as e:
        log.error(f"Syntax error parsing test code: {e}")
        raise CodeParsingError(f"Failed to parse test code: {e}") from e

    # Find the first class definition
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name == "Solution":
                log.debug("Skipping 'Solution' class in test code extraction")
                continue  # Skip Solution class if present
            # Iterate through class methods
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name.startswith(
                    "test_"
                ):
                    # Use ast.unparse for robust code generation
                    method_code = ast.unparse(method)
                    test_methods_code.append(method_code)

            # Stop after first class
            break

    return test_methods_code
