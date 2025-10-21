"""Transform and extract code segments using AST."""

import ast
import logging
from typing import List

from .exceptions import CodeParsingError, CodeTransformationError

# Setup logging
log = logging.getLogger(__name__)


def extract_function_with_helpers(code_string: str, target_function_name: str) -> str:
    """
    Extract target function body with helper functions and imports.
    For HumanEval style problems - returns properly indented completion code.

    Args:
        code_string: Python source code
        target_function_name: Name of target function to extract

    Returns:
        Indented code string with imports, helpers, and target function body

    Raises:
        CodeParsingError: If code has syntax errors
        CodeTransformationError: If target function not found

    Example:
        >>> code = "import os\\n\\ndef helper():\\n    pass\\n\\ndef main():\\n    helper()"
        >>> result = extract_function_with_helpers(code, "main")
        >>> "helper" in result
        True
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        log.error(f"Syntax error parsing code: {e}")
        raise CodeParsingError(f"Failed to parse code: {e}") from e

    # Find target function and helpers
    target_function = None
    helper_functions = []
    imports = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.FunctionDef):
            if node.name == target_function_name:
                target_function = node
            else:
                helper_functions.append(node)

    if target_function is None:
        log.error(f"Target function '{target_function_name}' not found")
        raise CodeTransformationError(
            f"Target function '{target_function_name}' not found"
        )

    # Build result with proper indentation
    result_parts = []

    # Add imports with indentation
    if imports:
        for imp in imports:
            result_parts.append("    " + ast.unparse(imp))
        result_parts.append("")

    # Add helper functions with indentation
    for helper in helper_functions:
        helper_code = ast.unparse(helper)
        # Indent each line
        indented_helper = "\n".join("    " + line for line in helper_code.split("\n"))
        result_parts.append(indented_helper)
        result_parts.append("")

    # Add target function body (skip the def line, keep body only)
    for stmt in target_function.body:
        stmt_code = ast.unparse(stmt)
        # Indent each line
        indented_stmt = "\n".join("    " + line for line in stmt_code.split("\n"))
        result_parts.append(indented_stmt)

    return "\n".join(result_parts)


def extract_class_block(code_string: str) -> str:
    """
    Extract the first class block from the code string.

    Args:
        code_string: Python source code containing a class definition

    Returns:
        Class source code as string

    Raises:
        CodeParsingError: If code has syntax errors
        CodeTransformationError: If no class found

    Example:
        >>> code = "class Foo:\\n    def bar(self): pass"
        >>> result = extract_class_block(code)
        >>> "class Foo" in result
        True
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        log.error(f"Syntax error parsing code: {e}")
        raise CodeParsingError(f"Failed to parse code: {e}") from e

    # Find first class definition
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return ast.unparse(node)

    log.error("No class definition found in code")
    raise CodeTransformationError("No class definition found in code")


def replace_test_methods(test_code: str, new_test_methods: List[str]) -> str:
    """
    Replace test methods in a test class with new test methods while preserving
    setup methods, helper methods, and class structure.

    Args:
        test_code: String containing the original unittest test class definition
        new_test_methods: List of new test method code strings to replace the old ones.
                         Each should be a complete method definition (e.g., 'def test_foo(self):\\n    ...')

    Returns:
        New test code string with replaced test methods

    Raises:
        CodeParsingError: If test code has syntax errors
        CodeTransformationError: If no class found or transformation fails

    Example:
        >>> original = "class TestFoo(unittest.TestCase):\\n    def test_old(self): pass"
        >>> new_method = "def test_new(self):\\n    self.assertTrue(True)"
        >>> result = replace_test_methods(original, [new_method])
        >>> "test_new" in result
        True
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError as e:
        log.error(f"Syntax error parsing test code: {e}")
        raise CodeParsingError(f"Failed to parse test code: {e}") from e

    # Find the first class definition
    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_node = node
            break

    if class_node is None:
        log.error("No class definition found in test code")
        raise CodeTransformationError("No class definition found in test code")

    # Build new class body: non-test elements + new test methods
    new_body: List[ast.stmt] = []

    # Add non-test methods and class variables
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef):
            # Keep non-test methods (setUp, tearDown, helpers, etc.)
            if not item.name.startswith("test_"):
                new_body.append(item)
        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
            # Keep class variables
            new_body.append(item)

    # Parse and add new test methods
    for test_method_str in new_test_methods:
        try:
            # Parse the test method string into an AST node
            method_tree = ast.parse(test_method_str)
            if method_tree.body and isinstance(method_tree.body[0], ast.FunctionDef):
                new_body.append(method_tree.body[0])
            else:
                log.warning(f"Invalid test method code: {test_method_str[:50]}...")
        except SyntaxError as e:
            log.warning(f"Failed to parse test method: {e}")
            continue

    # Update class body
    class_node.body = new_body

    # Return the unparsed class
    return ast.unparse(class_node)
