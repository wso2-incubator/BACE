"""Transform existing code into different code structures."""

import ast
from typing import List

from loguru import logger

from .exceptions import CodeParsingError, CodeTransformationError


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
        logger.error(f"Syntax error parsing test code: {e}")
        raise CodeParsingError(f"Failed to parse test code: {e}") from e

    # Find the first class definition
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name == "Solution":
                logger.debug("Skipping 'Solution' class in test code extraction")
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
        logger.error(f"Syntax error parsing code: {e}")
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
        logger.error(f"Target function '{target_function_name}' not found")
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
        logger.error(f"Syntax error parsing code: {e}")
        raise CodeParsingError(f"Failed to parse code: {e}") from e

    # Find first class definition
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return ast.unparse(node)

    logger.error("No class definition found in code")
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
        logger.error(f"Syntax error parsing test code: {e}")
        raise CodeParsingError(f"Failed to parse test code: {e}") from e

    # Find the first class definition
    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_node = node
            break

    if class_node is None:
        logger.error("No class definition found in test code")
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
                logger.warning(f"Invalid test method code: {test_method_str[:50]}...")
        except SyntaxError as e:
            logger.warning(f"Failed to parse test method: {e}")
            continue

    # Update class body
    class_node.body = new_body

    # Return the unparsed class
    return ast.unparse(class_node)


def remove_if_main_block(code_string: str) -> str:
    """
    Remove the 'if __name__ == "__main__":' block from the code string.

    Handles both standard and reversed comparisons:
    - if __name__ == "__main__":
    - if "__main__" == __name__:

    Args:
        code_string: Python source code
    Returns:
        Code string without the main block
    """

    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing code: {e}")
        raise CodeParsingError(f"Failed to parse code: {e}") from e

    # Filter out the if __name__ == "__main__": block
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            # Check for: if __name__ == "__main__":
            is_standard_main = (
                isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"
            )

            # Check for reversed: if "__main__" == __name__:
            is_reversed_main = (
                isinstance(node.test.left, ast.Constant)
                and node.test.left.value == "__main__"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Name)
                and node.test.comparators[0].id == "__name__"
            )

            if is_standard_main or is_reversed_main:
                logger.debug("Found and skipping if __name__ == '__main__' block")
                continue  # Skip this block

        new_body.append(node)

    tree.body = new_body
    return ast.unparse(tree)
