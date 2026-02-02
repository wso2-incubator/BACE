"""Generic test generation from Test dataclass and starter code.

This module provides dataset-agnostic test generation that infers the test format
from the starter code signature, not from dataset-specific metadata.
"""

import ast
import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from .exceptions import CodeParsingError


@dataclass
class MethodSignature:
    """Parsed method signature information."""

    class_name: str
    method_name: str
    params: list[tuple[str, Optional[str]]]  # [(name, type_annotation), ...]
    return_type: Optional[str]


def parse_method_signature(starter_code: str) -> MethodSignature:
    """
    Parse starter code to extract class and method signature using AST.

    Args:
        starter_code: Python code containing a class with a method

    Returns:
        MethodSignature with parsed information

    Raises:
        CodeParsingError: If parsing fails or no valid class/method found

    Example:
        >>> code = "class Solution:\\n    def add(self, x: int, y: int) -> int:\\n        pass"
        >>> sig = parse_method_signature(code)
        >>> sig.method_name
        'add'
        >>> sig.params
        [('x', 'int'), ('y', 'int')]
    """
    try:
        tree = ast.parse(starter_code)
    except SyntaxError:
        # Try to complete incomplete starter code
        try:
            completed = starter_code.rstrip() + "\n        pass"
            tree = ast.parse(completed)
        except SyntaxError as e:
            raise CodeParsingError(f"Failed to parse starter code: {e}") from e

    # Find first class definition
    class_node: Optional[ast.ClassDef] = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_node = node
            break

    if not class_node:
        raise CodeParsingError("No class definition found in starter code")

    # Find first method with 'self' parameter (instance method)
    method_node: Optional[ast.FunctionDef] = None
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            if node.args.args and node.args.args[0].arg == "self":
                method_node = node
                break

    if not method_node:
        raise CodeParsingError("No instance method found in class")

    # Extract class name
    class_name = class_node.name

    # Extract method name
    method_name = method_node.name

    # Extract parameters (skip 'self')
    params: list[tuple[str, Optional[str]]] = []
    for arg in method_node.args.args[1:]:  # Skip self
        param_name = arg.arg
        param_type = None
        if arg.annotation:
            param_type = ast.unparse(arg.annotation)
        params.append((param_name, param_type))

    # Extract return type
    return_type = None
    if method_node.returns:
        return_type = ast.unparse(method_node.returns)

    return MethodSignature(
        class_name=class_name,
        method_name=method_name,
        params=params,
        return_type=return_type,
    )


def is_stdin_signature(signature: MethodSignature) -> bool:
    """
    Determine if signature represents a STDIN-style problem.

    STDIN problems have signature: def method(self, input_str: str) -> str:
    - Exactly one parameter named 'input_str'
    - Parameter type is 'str'
    - Return type is 'str'

    Args:
        signature: Parsed method signature

    Returns:
        True if signature matches STDIN pattern, False otherwise
    """
    # Must have exactly one parameter
    if len(signature.params) != 1:
        return False

    param_name, param_type = signature.params[0]

    # Parameter should be named 'input_str' and typed as 'str'
    # Return type should be 'str'
    return (
        param_name == "input_str"
        and param_type == "str"
        and signature.return_type == "str"
    )


def generate_pytest_test(
    input: str, output: str, starter_code: str, test_number: int = 1
) -> str:
    """
    Generate a standalone pytest test function from test data and starter code.

    Automatically infers test format (STDIN vs FUNCTIONAL) from the starter code
    signature. No dataset-specific metadata required.

    STDIN format (single string I/O):
        - Signature: def method(self, input_str: str) -> str:
        - Input: Raw string (can be multi-line)
        - Output: String result
        - Generated: assert solution.method(input_str) == expected

    FUNCTIONAL format (multiple typed arguments):
        - Signature: def method(self, arg1: Type1, arg2: Type2) -> Result:
        - Input: Newline-separated Python literals
        - Output: Python literal
        - Generated: assert solution.method(*args) == expected

    Args:
        input: Test input as string
        output: Expected output as string
        starter_code: Starter code containing method signature
        test_number: Test case number for naming (default: 1)

    Returns:
        Complete pytest test function as string

    Raises:
        CodeParsingError: If starter code cannot be parsed

    Example:
        >>> starter = "class Solution:\\n    def add(self, x: int, y: int) -> int:\\n        pass"
        >>> test = generate_pytest_test("1\\n2", "3", starter, 1)
        >>> "def test_case_1():" in test
        True
    """
    # Parse signature to determine format
    try:
        signature = parse_method_signature(starter_code)
    except CodeParsingError as e:
        logger.error(f"Failed to parse starter code for test generation: {e}")
        raise

    class_name = signature.class_name
    method_name = signature.method_name

    # Determine format based on signature
    if is_stdin_signature(signature):
        return _generate_stdin_test(input, output, class_name, method_name, test_number)
    else:
        return _generate_functional_test(
            input, output, class_name, method_name, test_number
        )


def _generate_stdin_test(
    input: str, output: str, class_name: str, method_name: str, test_number: int
) -> str:
    """
    Generate pytest test for STDIN-style problem.

    Args:
        input: Input string (may contain newlines)
        output: Expected output string
        class_name: Name of solution class
        method_name: Name of method to test
        test_number: Test case number

    Returns:
        Pytest test function as string
    """
    # Strip trailing newlines for consistency
    input_literal = repr(input.rstrip("\n"))
    output_literal = repr(output.rstrip("\n"))

    return f"""def test_case_{test_number}():
    solution = {class_name}()
    input_str = {input_literal}
    expected_output = {output_literal}
    assert solution.{method_name}(input_str) == expected_output
"""


def _generate_functional_test(
    input: str, output: str, class_name: str, method_name: str, test_number: int
) -> str:
    """
    Generate pytest test for FUNCTIONAL-style problem.

    Args:
        input: Newline-separated Python literals
        output: Expected output as Python literal
        class_name: Name of solution class
        method_name: Name of method to test
        test_number: Test case number

    Returns:
        Pytest test function as string
    """
    import ast as ast_module

    # Split input by newlines and filter empty lines
    input_lines = [line.strip() for line in input.split("\n") if line.strip()]

    # Use repr() to create string literals for the code
    input_lines_repr = repr(input_lines)
    output_repr = repr(output)

    return f"""def test_case_{test_number}():
    # Parse input arguments
    import ast
    solution = {class_name}()
    input_lines = {input_lines_repr}
    args = [ast.literal_eval(line) for line in input_lines]
    expected_output = ast.literal_eval({output_repr})
    assert solution.{method_name}(*args) == expected_output
"""
