"""Build complete test scripts by combining programmer and tester code."""

import ast
import logging
from typing import Optional

from .exceptions import CodeParsingError, CodeTransformationError

# Setup logging
log = logging.getLogger(__name__)


def build_test_script_for_humaneval(programmer_code: str, tester_code: str) -> str:
    """
    Combine programmer and tester code into a single test script for HumanEval Style Problems.

    Args:
        programmer_code: Python code with function implementations
        tester_code: Python code with test class

    Returns:
        Complete test script as string

    Raises:
        CodeParsingError: If either code has syntax errors

    Example:
        >>> prog = "def add(a, b):\\n    return a + b"
        >>> test = "import unittest\\nclass TestAdd(unittest.TestCase):\\n    def test_add(self): pass"
        >>> script = build_test_script_for_humaneval(prog, test)
        >>> "def add" in script and "class TestAdd" in script
        True
    """
    try:
        prog_tree = ast.parse(programmer_code)
        test_tree = ast.parse(tester_code)
    except SyntaxError as e:
        log.error(f"Syntax error parsing code: {e}")
        raise CodeParsingError(f"Failed to parse code: {e}") from e

    # Collect imports from both
    imports = []
    import_strs = set()

    for node in prog_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imp_str = ast.unparse(node)
            if imp_str not in import_strs:
                import_strs.add(imp_str)
                imports.append(node)

    for node in test_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imp_str = ast.unparse(node)
            if imp_str not in import_strs:
                import_strs.add(imp_str)
                imports.append(node)

    # Collect programmer functions
    prog_functions = [
        node for node in prog_tree.body if isinstance(node, ast.FunctionDef)
    ]

    # Collect tester classes
    test_classes = [node for node in test_tree.body if isinstance(node, ast.ClassDef)]

    # Build the script using AST unparsing
    script_parts = []

    # Add imports
    for imp in imports:
        script_parts.append(ast.unparse(imp))
    if imports:
        script_parts.append("")

    # Add programmer code
    script_parts.append("# Programmer Code")
    for func in prog_functions:
        script_parts.append(ast.unparse(func))
        script_parts.append("")

    # Add tester code
    script_parts.append("# Tester Code")
    for cls in test_classes:
        script_parts.append(ast.unparse(cls))
        script_parts.append("")

    # Add main block
    script_parts.append('if __name__ == "__main__":')
    script_parts.append("    unittest.main(verbosity=2)")

    return "\n".join(script_parts)


def build_test_script_for_lcb(programmer_code: str, tester_code: str) -> str:
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
        >>> script = build_test_script_for_lcb(prog, test)
        >>> "class Solution" in script
        True
    """
    # --- 1. Parse Programmer Code ---
    try:
        prog_tree = ast.parse(programmer_code)
    except SyntaxError as e:
        log.error(f"Syntax error parsing programmer code: {e}")
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
        log.error("No Solution class or functions found in programmer code")
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
        log.error(f"Syntax error parsing tester code: {e}")
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
