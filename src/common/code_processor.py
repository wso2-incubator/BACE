import ast
import logging
import re
from typing import Dict, List, Optional, Tuple, TypedDict

# Setup logging
log = logging.getLogger(__name__)


# Custom Exceptions
class CodeProcessingError(Exception):
    """Base exception for code processing errors."""

    pass


class CodeParsingError(CodeProcessingError):
    """Raised when code cannot be parsed due to syntax errors."""

    pass


class CodeTransformationError(CodeProcessingError):
    """Raised when code transformation/generation fails."""

    pass


class CodeStructure(TypedDict):
    import_lines: List[str]
    function_definitions: Dict[str, Tuple[int, int]]
    class_definitions: Dict[str, Tuple[int, int]]


class CodeProcessor:
    """
    A class to process and extract relevant code segments from a given code string
    using Abstract Syntax Trees (AST) for robust parsing.
    """

    def extract_function_name_from_problem(self, problem_prompt: str) -> str:
        """
        Extracts the function name from the problem prompt.
        """
        lines = problem_prompt.split("\n")
        for line in lines:
            if line.strip().startswith("def "):
                func_name = line.strip().split("def ")[1].split("(")[0]
                return func_name
        return ""

    def extract_all_code_blocks(self, response: str) -> List[str]:
        """
        Extracts all Python code blocks (```python ... ```) from the LLM response.
        """
        pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")
        matches = pattern.findall(response)
        return matches if matches else []

    def extract_code_block_from_response(self, response: str) -> str:
        """
        Extracts the first Python code block (```python ... ```) from the LLM response.
        If no Python code block is found, returns the original response.
        """
        pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")
        match = pattern.search(response)
        if match:
            return match.group(1)
        return response  # Return original response if no code block found

    def _parse_code_structure(self, code_string: str) -> CodeStructure:
        """Parse code string to find all imports, functions, and classes using AST."""
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
            end_idx = (
                (node.end_lineno - 1) if node.end_lineno is not None else start_idx
            )

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.extend(lines[start_idx : end_idx + 1])
            elif isinstance(node, ast.FunctionDef):
                function_definitions[node.name] = (start_idx, end_idx)
            elif isinstance(node, ast.ClassDef):
                class_definitions[node.name] = (start_idx, end_idx)

        return {
            "import_lines": list(dict.fromkeys(import_lines)),
            "function_definitions": function_definitions,
            "class_definitions": class_definitions,
        }

    def extract_function_with_helpers(
        self, code_string: str, target_function_name: str
    ) -> str:
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
            indented_helper = "\n".join(
                "    " + line for line in helper_code.split("\n")
            )
            result_parts.append(indented_helper)
            result_parts.append("")

        # Add target function body (skip the def line, keep body only)
        for stmt in target_function.body:
            stmt_code = ast.unparse(stmt)
            # Indent each line
            indented_stmt = "\n".join("    " + line for line in stmt_code.split("\n"))
            result_parts.append(indented_stmt)

        return "\n".join(result_parts)

    def extract_class_block(self, code_string: str) -> str:
        """
        Extract the first class block from the code string.

        Args:
            code_string: Python source code containing a class definition

        Returns:
            Class source code as string

        Raises:
            CodeParsingError: If code has syntax errors
            CodeTransformationError: If no class found
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

    def extract_test_case_names(self, test_code: str) -> List[str]:
        """
        Extract test case method names from a unittest test class.

        Args:
            test_code: String containing a unittest test class definition

        Returns:
            List of test method names

        Raises:
            CodeParsingError: If test code has syntax errors
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

    def extract_test_methods_code(self, test_code: str) -> List[str]:
        """
        Extract the actual code for each test method from a unittest test class.
        Returns a list of code strings for each test method (methods starting with 'test_') in the order they appear.

        Args:
            test_code: String containing a unittest test class definition

        Returns:
            List of code strings for each test method

        Raises:
            CodeParsingError: If test code has syntax errors
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
                # Iterate through class methods
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name.startswith(
                        "test_"
                    ):
                        # Use ast.unparse instead of string slicing - more robust!
                        method_code = ast.unparse(method)
                        test_methods_code.append(method_code)

                # Stop after first class
                break

        return test_methods_code

    def replace_test_methods(self, test_code: str, new_test_methods: List[str]) -> str:
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

        # Add non-test methods and class variables (using ast.unparse - robust!)
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
                if method_tree.body and isinstance(
                    method_tree.body[0], ast.FunctionDef
                ):
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

    def build_test_script_for_humaneval(
        self, programmer_code: str, tester_code: str
    ) -> str:
        """
        Combine programmer and tester code into a single test script for HumanEval Style Problems.

        Args:
            programmer_code: Python code with function implementations
            tester_code: Python code with test class

        Returns:
            Complete test script as string

        Raises:
            CodeParsingError: If either code has syntax errors
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
        test_classes = [
            node for node in test_tree.body if isinstance(node, ast.ClassDef)
        ]

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

    def build_test_script_for_lcb(self, programmer_code: str, tester_code: str) -> str:
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
