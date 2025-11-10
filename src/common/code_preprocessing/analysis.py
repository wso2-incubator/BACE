"""Analyze Python code and return metadata (non-code information)."""

import ast
import re
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


def contains_starter_code(code_string: str, starter_code: str) -> bool:
    """
    Check if starter code (possibly incomplete/invalid) is present in complete code.

    This function handles incomplete starter code that may not be syntactically valid
    by normalizing both strings and doing a flexible text-based comparison. It removes
    comments, normalizes whitespace, and checks for structural similarity.

    Args:
        code_string: Complete Python source code
        starter_code: Starter/template code (may be incomplete)

    Returns:
        True if starter code structure is found in code_string, False otherwise

    Example:
        >>> code = "class Solution:\\n    def solve(self, x):\\n        return x * 2"
        >>> starter = "class Solution:\\n    def solve(self"
        >>> contains_starter_code(code, starter)
        True
    """

    def normalize_code(text: str) -> str:
        """Normalize code by removing comments, extra whitespace, and docstrings."""
        # Remove single-line comments
        text = re.sub(r"#.*$", "", text, flags=re.MULTILINE)

        # Remove multi-line strings/docstrings (triple quotes)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)

        # Normalize whitespace: collapse multiple spaces/tabs to single space
        text = re.sub(r"[ \t]+", " ", text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]

        # Remove empty lines
        lines = [line for line in lines if line]

        return "\n".join(lines)

    def extract_signature_pattern(text: str) -> list[str]:
        """
        Extract structural patterns from code (class/function signatures).
        Returns a list of signatures for more precise matching.
        """
        signatures = []

        # Extract class names
        class_names = re.findall(r"class\s+(\w+)", text)
        signatures.extend([f"class {name}" for name in class_names])

        # Extract function/method names
        func_names = re.findall(r"def\s+(\w+)\s*\(", text)
        signatures.extend([f"def {name}" for name in func_names])

        return signatures

    # Normalize both strings
    normalized_code = normalize_code(code_string)
    normalized_starter = normalize_code(starter_code)

    # If starter is empty, it's contained
    if not normalized_starter.strip():
        return True

    # Strategy 1: Direct substring match after normalization
    if normalized_starter in normalized_code:
        return True

    # Strategy 2: Check if all starter signatures are present in code
    # This handles incomplete starter code
    starter_sigs = extract_signature_pattern(normalized_starter)
    code_sigs = extract_signature_pattern(normalized_code)

    if starter_sigs:
        # All starter signatures must be in code
        starter_sig_set = set(starter_sigs)
        code_sig_set = set(code_sigs)

        if starter_sig_set.issubset(code_sig_set):
            return True

    # Strategy 3: Line-by-line fuzzy matching for very incomplete code
    # Check if significant lines from starter are in code
    starter_lines = [line for line in normalized_starter.split("\n") if len(line) > 5]
    code_lines_set = set(normalized_code.split("\n"))

    if starter_lines:
        matches = sum(1 for line in starter_lines if line in code_lines_set)
        # If most significant lines match, consider it contained
        if matches > 0 and matches / len(starter_lines) >= 0.8:  # 80% threshold
            return True

    return False
