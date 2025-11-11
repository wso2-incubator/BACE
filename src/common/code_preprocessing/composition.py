"""Compose complete executable test scripts from programmer and tester code."""

import ast
from typing import List, Optional

from loguru import logger

from .exceptions import CodeParsingError, CodeTransformationError


def compose_lcb_test_script(programmer_code: str, tester_code: str) -> str:
    """
    Combine programmer and tester code into a single test script for LCB Style Problems.

    Args:
        programmer_code: Python code with Solution class or functions
        tester_code: Python code with test class

    Returns:
        Complete test script as string

    Raises:
        CodeParsingError: If either code has syntax errors
        CodeTransformationError: If Solution class/functions not found

    Example:
        >>> prog = "class Solution:\\n    def solve(self): pass"
        >>> test = "import unittest\\nclass TestSolution(unittest.TestCase):\\n    def test_solve(self): pass"
        >>> script = compose_lcb_test_script(prog, test)
        >>> "class Solution" in script
        True
    """
    # --- 1. Parse Programmer Code ---
    try:
        prog_tree = ast.parse(programmer_code)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing programmer code: {e}")
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
    for c in prog_classes:
        if c.name == "Solution":
            solution_class_node = c
        else:
            helper_class_nodes.append(c)

    # If no 'Solution' class, wrap loose functions into one
    if not solution_class_node and prog_funcs:
        wrapped_funcs = []
        for func in prog_funcs:
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
        logger.error("No Solution class or functions found in programmer code")
        raise CodeTransformationError(
            "No Solution class or functions found in programmer code"
        )

    # --- 3. Parse and Clean Tester Code ---

    class SolutionImportRemover(ast.NodeTransformer):
        """Remove imports of Solution class from tester code."""

        def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
            if node.module == "solution":
                return None
            return node

        def visit_Import(self, node: ast.Import) -> Optional[ast.AST]:
            node.names = [alias for alias in node.names if alias.name != "Solution"]
            if not node.names:
                return None
            return node

    try:
        tester_tree = ast.parse(tester_code)
        tester_tree = SolutionImportRemover().visit(tester_tree)
        ast.fix_missing_locations(tester_tree)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing tester code: {e}")
        raise CodeParsingError(f"Failed to parse tester code: {e}") from e

    tester_imports = []
    tester_classes = []
    tester_funcs = []
    tester_main_block = None

    for node in tester_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            tester_imports.append(node)
        elif isinstance(node, ast.ClassDef):
            if node.name != "Solution":
                tester_classes.append(node)
        elif isinstance(node, ast.FunctionDef):
            tester_funcs.append(node)
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            tester_main_block = node

    # --- 4. Assemble Final Script ---

    all_import_nodes = prog_imports + tester_imports
    has_unittest = False
    import_strs = set()
    final_import_nodes = []

    for node in all_import_nodes:
        import_str = ast.unparse(node)
        if "import unittest" in import_str:
            has_unittest = True
        if import_str not in import_strs:
            import_strs.add(import_str)
            final_import_nodes.append(node)

    if not has_unittest:
        final_import_nodes.insert(0, ast.Import(names=[ast.alias(name="unittest")]))

    final_code_parts = []

    # Add Imports
    for node in final_import_nodes:
        final_code_parts.append(ast.unparse(node))
    final_code_parts.append("")

    # Add Programmer Code
    final_code_parts.append("# Programmer Code")
    for node in helper_class_nodes:
        final_code_parts.append(ast.unparse(node))
        final_code_parts.append("")
    final_code_parts.append(ast.unparse(solution_class_node))
    final_code_parts.append("")

    # Add Tester Code
    final_code_parts.append("# Tester Code")
    for node in tester_funcs:
        final_code_parts.append(ast.unparse(node))
        final_code_parts.append("")
    for node in tester_classes:
        final_code_parts.append(ast.unparse(node))
        final_code_parts.append("")

    # Add Main Block
    if tester_main_block:
        final_code_parts.append(ast.unparse(tester_main_block))
    else:
        final_code_parts.append('if __name__ == "__main__":')
        final_code_parts.append("    unittest.main(verbosity=2)")

    return "\n".join(final_code_parts)


def rebuild_unittest_with_methods(test_code: str, new_test_methods: List[str]) -> str:
    """
    Rebuild a unittest class by replacing old test methods with new ones.

    This function preserves the class structure, imports, and non-test methods
    (like setUp, tearDown, etc.) while replacing all test methods (methods
    starting with 'test_') with the provided new test methods.

    Args:
        test_code: String containing the original unittest test class definition
        new_test_methods: List of code strings for new test methods to insert

    Returns:
        String containing the rebuilt unittest class with new test methods

    Raises:
        CodeParsingError: If test code has syntax errors or no class is found

    Example:
        >>> original = "class TestFoo(unittest.TestCase):\\n    def test_old(self): pass"
        >>> new_methods = ["def test_new(self):\\n    self.assertTrue(True)"]
        >>> rebuilt = rebuild_unittest_with_methods(original, new_methods)
        >>> 'test_new' in rebuilt
        True
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing test code: {e}")
        raise CodeParsingError(f"Failed to parse test code: {e}") from e

    # Find the first class definition
    class_node: Optional[ast.ClassDef] = None
    other_nodes: List[ast.stmt] = []  # To preserve imports and other top-level code

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and class_node is None:
            class_node = node
        else:
            other_nodes.append(node)

    if class_node is None:
        raise CodeParsingError("No class definition found in test code")

    # Separate test methods from non-test methods
    non_test_members: List[ast.stmt] = []
    for member in class_node.body:
        if isinstance(member, ast.FunctionDef):
            # Keep setUp, tearDown, and other non-test methods
            if not member.name.startswith("test_"):
                non_test_members.append(member)
        else:
            # Keep class variables, docstrings, etc.
            non_test_members.append(member)

    # Parse the new test methods and add them to the class
    new_method_nodes: List[ast.stmt] = []
    for method_code in new_test_methods:
        try:
            # Parse the method code
            method_tree = ast.parse(method_code)
            # Extract the function definition
            for node in method_tree.body:
                if isinstance(node, ast.FunctionDef):
                    new_method_nodes.append(node)
                    break
        except SyntaxError as e:
            logger.warning(f"Skipping invalid test method code: {e}")
            continue

    # Rebuild the class body with non-test members first, then new test methods
    class_node.body = non_test_members + new_method_nodes

    # Rebuild the module with preserved imports and the modified class
    tree.body = other_nodes + [class_node]

    # Unparse the modified AST back to code
    rebuilt_code = ast.unparse(tree)

    return rebuilt_code
