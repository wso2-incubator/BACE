"""Compose complete executable test scripts from programmer and tester code."""

import ast
from typing import List, Optional

from loguru import logger

from .exceptions import CodeParsingError, CodeTransformationError


def compose_pytest_script(programmer_code: str, test_function: str) -> str:
    """
    Compose a pytest script by combining programmer code and test function.

    Simple composition that concatenates code without transformation.
    Assumes test_function is already written to work with the programmer_code structure.

    Args:
        programmer_code: Python code (functions, classes, any structure)
        test_function: A standalone pytest test function

    Returns:
        Complete pytest script as string

    Example:
        >>> prog = "def add(a, b):\\n    return a + b"
        >>> test = "def test_add():\\n    assert add(2, 3) == 5"
        >>> script = compose_pytest_script(prog, test)
        >>> "def add" in script and "def test_add" in script
        True
    """
    parts = []

    # Ensure pytest is available
    has_pytest = "import pytest" in programmer_code or "import pytest" in test_function
    if not has_pytest:
        parts.append("import pytest")
        parts.append("")

    # Add programmer code
    parts.append(programmer_code.strip())
    parts.append("")

    # Add test function
    parts.append(test_function.strip())
    parts.append("")

    # Add pytest main execution
    parts.append('if __name__ == "__main__":')
    parts.append('    pytest.main([__file__, "-v"])')

    test_script = "\n".join(parts)
    logger.trace(f"Composed pytest script:\n{test_script}")
    return test_script


def compose_lcb_output_script(programmer_code: str, input_data: str) -> str:
    """
    Generate an executable script from programmer code and input data.

    Combines programmer code with a Solution class (or creates one from functions)
    and generates code to instantiate the Solution and call its method with the
    provided input data.

    Args:
        programmer_code: Python code with Solution class or functions
        input_data: Python dict literal string with format "{'inputdata': <dict_object>}"
                   where dict_object has keys matching the function parameters.
                   Supports string concatenation (e.g., "str1" + "str2").

    Returns:
        Complete executable script that instantiates Solution and calls the method

    Raises:
        CodeParsingError: If programmer code has syntax errors
        CodeTransformationError: If Solution class/functions not found or input parsing fails

    Example:
        >>> prog = "class Solution:\\n    def add(self, a, b):\\n        return a + b"
        >>> input_data = "{'inputdata': {'a': 1, 'b': 2}}"
        >>> script = compose_lcb_output_script(prog, input_data)
        >>> "sol.add(1, 2)" in script
        True
    """
    # --- 1. Parse Programmer Code ---
    try:
        prog_tree = ast.parse(programmer_code)
    except SyntaxError as e:
        logger.debug(f"Syntax error parsing programmer code: {programmer_code}")
        raise CodeParsingError(f"Failed to parse programmer code: {e}") from e

    prog_imports = []
    prog_classes = []
    prog_funcs = []
    for node in prog_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            prog_imports.append(node)
        elif isinstance(node, ast.ClassDef):
            prog_classes.append(node)
        elif isinstance(node, ast.FunctionDef):
            prog_funcs.append(node)

    # --- 2. Find/Create Solution Class ---
    solution_class_node = None
    helper_class_nodes = []
    solution_method_name = None

    for c in prog_classes:
        if c.name == "Solution":
            solution_class_node = c
            # Find the first non-dunder method in the Solution class
            for member in c.body:
                if isinstance(member, ast.FunctionDef) and not member.name.startswith(
                    "_"
                ):
                    solution_method_name = member.name
                    break
        else:
            helper_class_nodes.append(c)

    # If no 'Solution' class, wrap loose functions into one
    if not solution_class_node and prog_funcs:
        wrapped_funcs = []
        for func in prog_funcs:
            # Record the first function name as the method to call
            if solution_method_name is None:
                solution_method_name = func.name

            # Add 'self' argument if not present
            args = func.args.args
            if not args or args[0].arg != "self":
                func.args.args.insert(0, ast.arg(arg="self"))
            wrapped_funcs.append(func)

        # Create ClassDef with type_params for Python 3.12+
        solution_class_node = ast.ClassDef(
            name="Solution",
            bases=[],
            keywords=[],
            body=wrapped_funcs,  # type: ignore[arg-type]
            decorator_list=[],
            type_params=[],
        )

    if not solution_class_node:
        logger.debug(
            f"No Solution class or functions found in programmer code: {programmer_code}"
        )
        raise CodeTransformationError(
            "No Solution class or functions found in programmer code"
        )

    if not solution_method_name:
        logger.debug("No callable method found in Solution class")
        raise CodeTransformationError("No callable method found in Solution class")

    # --- 3. Parse Input Data ---
    try:
        # Use eval() to parse Python dict format with string concatenation support
        # This is safe because we control the input format and only expect dict literals
        input_dict = eval(input_data, {"__builtins__": {}}, {})

        if "inputdata" not in input_dict:
            raise CodeTransformationError("Input data must contain 'inputdata' key")

        input_params = input_dict["inputdata"]

        if not isinstance(input_params, dict):
            raise CodeTransformationError("'inputdata' value must be a dict object")

    except (ValueError, SyntaxError, NameError, TypeError) as e:
        logger.debug(f"Failed to parse input data as Python dict: {input_data}")
        raise CodeTransformationError(f"Failed to parse input data: {e}") from e
    except Exception as e:
        logger.debug(f"Error processing input data: {input_data}")
        raise CodeTransformationError(f"Error processing input data: {e}") from e

    # --- 4. Assemble Final Script ---

    final_code_parts = []

    # Add Imports
    for node in prog_imports:
        final_code_parts.append(ast.unparse(node))
    if prog_imports:
        final_code_parts.append("")

    # Add Helper Classes
    for node in helper_class_nodes:
        final_code_parts.append(ast.unparse(node))
        final_code_parts.append("")

    # Add Solution Class
    final_code_parts.append(ast.unparse(solution_class_node))
    final_code_parts.append("")

    # Add Execution Code
    final_code_parts.append("# Execute solution")
    final_code_parts.append("sol = Solution()")

    # Build the method call with parameters
    # Use repr() for all values to properly escape special characters like newlines
    param_str = ", ".join(repr(v) for v in input_params.values())
    final_code_parts.append(f"print(sol.{solution_method_name}({param_str}))")

    output_script: str = "\n".join(final_code_parts)
    logger.trace(f"Composed LCB output script:\n{output_script}")
    return output_script
