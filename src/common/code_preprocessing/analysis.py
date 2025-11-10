"""Analyze Python code and return metadata (non-code information)."""

import ast
from typing import List

from .exceptions import CodeParsingError


def analyze_test_methods(test_code: str) -> List[str]:
    """
    Analyze a unittest test class and return the names of all test methods.

    Args:
        test_code: String containing a unittest test class definition

    Returns:
        List of test method names (methods starting with 'test_')

    Raises:
        CodeParsingError: If test code has syntax errors

    Example:
        >>> code = "class TestFoo(unittest.TestCase):\\n    def test_bar(self): pass"
        >>> analyze_test_methods(code)
        ['test_bar']
    """
    test_cases = []
    try:
        tree = ast.parse(test_code)
    except SyntaxError as e:
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


def analyze_code_imports(code_string: str) -> List[str]:
    """
    Analyze code and return all import statements.

    Args:
        code_string: Python source code

    Returns:
        List of import statement strings

    Raises:
        CodeParsingError: If code has syntax errors

    Example:
        >>> code = "import os\\nimport sys\\ndef foo(): pass"
        >>> analyze_code_imports(code)
        ['import os', 'import sys']
    """
    lines = code_string.split("\n")
    import_lines: List[str] = []

    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        raise CodeParsingError(f"Failed to parse code: {e}") from e

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start_idx = node.lineno - 1
            end_idx = (
                (node.end_lineno - 1) if node.end_lineno is not None else start_idx
            )
            import_lines.extend(lines[start_idx : end_idx + 1])

    return list(dict.fromkeys(import_lines))  # Remove duplicates


def analyze_code_functions(code_string: str) -> List[str]:
    """
    Analyze code and return all function names.

    Args:
        code_string: Python source code

    Returns:
        List of function names

    Raises:
        CodeParsingError: If code has syntax errors

    Example:
        >>> code = "def foo():\\n    pass\\ndef bar():\\n    pass"
        >>> analyze_code_functions(code)
        ['foo', 'bar']
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        raise CodeParsingError(f"Failed to parse code: {e}") from e

    function_names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)

    return function_names


def analyze_code_classes(code_string: str) -> List[str]:
    """
    Analyze code and return all class names.

    Args:
        code_string: Python source code

    Returns:
        List of class names

    Raises:
        CodeParsingError: If code has syntax errors

    Example:
        >>> code = "class Foo:\\n    pass\\nclass Bar:\\n    pass"
        >>> analyze_code_classes(code)
        ['Foo', 'Bar']
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        raise CodeParsingError(f"Failed to parse code: {e}") from e

    class_names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)

    return class_names
