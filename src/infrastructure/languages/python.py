# src/infrastructure/languages/python.py
"""
Python implementation of the ILanguage protocol.

Self-contained: all helper logic (evaluation-script composition, pytest test
generation, if-__main__ removal) lives as private methods inside PythonLanguage
so this module has no dependency on any external utility package.
"""

import ast
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from coevolution.core.interfaces.language import (
    ILanguage,
    LanguageParsingError,
    LanguageTransformationError,
)


@dataclass
class _MethodSignature:
    """Parsed method / function signature — used internally by PythonLanguage."""

    class_name: Optional[str]  # None for standalone functions
    method_name: str
    params: list[tuple[str, Optional[str]]]  # [(name, type_annotation), ...]
    return_type: Optional[str]
    is_standalone: bool = False


class PythonLanguage(ILanguage):
    """
    Adapter for Python-specific code operations.
    All logic is self-contained within this class.
    """

    def __init__(self) -> None:
        self._block_pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")
        logger.info("Initialized Python language adapter")

    # ------------------------------------------------------------------
    # ILanguage interface
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        return "python"

    def extract_code_blocks(self, response: str) -> List[str]:
        """Extract Python code blocks from an LLM response."""
        matches = self._block_pattern.findall(response)

        if not matches:
            stripped = response.strip()
            if self.is_syntax_valid(stripped):
                return [stripped]
            logger.debug(
                f"No Python code blocks found in response: {response[:100]}..."
            )
            return []

        valid_blocks = [block for block in matches if self.is_syntax_valid(block)]
        if not valid_blocks and matches:
            logger.debug("Python blocks found but all had syntax errors.")
        return valid_blocks

    def is_syntax_valid(self, code: str) -> bool:
        """Validate Python syntax using ast.parse."""
        try:
            ast.parse(code)
            return True
        except (SyntaxError, ValueError, OverflowError):
            return False

    def extract_test_names(self, test_code: str) -> List[str]:
        """Extract pytest-style test function names (starting with test_)."""
        test_names = []
        try:
            tree = ast.parse(test_code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_names.append(node.name)
        except Exception as e:
            logger.debug(f"Failed to extract test names: {e}")
        return test_names

    def split_tests(self, test_code: str) -> List[str]:
        """Split a test block into individual standalone test functions."""
        test_functions: List[str] = []
        try:
            tree = ast.parse(test_code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    try:
                        function_code = ast.get_source_segment(test_code, node)
                        if function_code is None:
                            function_code = ast.unparse(node)
                        test_functions.append(function_code)
                    except Exception:
                        try:
                            test_functions.append(ast.unparse(node))
                        except Exception:
                            continue
        except Exception as e:
            logger.debug(f"Failed to split tests: {e}")
        return test_functions

    def compose_test_script(self, code_snippet: str, test_snippet: str) -> str:
        """Compose a pytest script by combining code and test snippet."""
        parts = []
        has_pytest = "import pytest" in code_snippet or "import pytest" in test_snippet
        if not has_pytest:
            parts.append("import pytest")
            parts.append("")
        parts.append(code_snippet.strip())
        parts.append("")
        parts.append(test_snippet.strip())
        parts.append("")
        parts.append('if __name__ == "__main__":')
        parts.append("    import sys")
        parts.append('    pytest.main([__file__, "-v"])')
        return "\n".join(parts)

    def compose_evaluation_script(self, code_snippet: str, input_data: str) -> str:
        """
        Compose an evaluation script that instantiates Solution and calls the
        method with the provided input data, printing the result.

        Wraps loose functions into a Solution class if needed, then emits code
        that instantiates Solution and calls the method with the given inputs.
        """
        try:
            prog_tree = ast.parse(code_snippet)
        except SyntaxError as e:
            raise LanguageParsingError(f"Failed to parse programmer code: {e}") from e

        prog_imports: list[ast.stmt] = []
        prog_classes: list[ast.ClassDef] = []
        prog_funcs: list[ast.FunctionDef] = []
        for node in prog_tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                prog_imports.append(node)
            elif isinstance(node, ast.ClassDef):
                prog_classes.append(node)
            elif isinstance(node, ast.FunctionDef):
                prog_funcs.append(node)

        solution_class_node: Optional[ast.ClassDef] = None
        helper_class_nodes: list[ast.ClassDef] = []
        solution_method_name: Optional[str] = None

        for c in prog_classes:
            if c.name == "Solution":
                solution_class_node = c
                for member in c.body:
                    if isinstance(
                        member, ast.FunctionDef
                    ) and not member.name.startswith("_"):
                        solution_method_name = member.name
                        break
            else:
                helper_class_nodes.append(c)

        if not solution_class_node and prog_funcs:
            wrapped: list[ast.FunctionDef] = []
            for func in prog_funcs:
                if solution_method_name is None:
                    solution_method_name = func.name
                if not func.args.args or func.args.args[0].arg != "self":
                    func.args.args.insert(0, ast.arg(arg="self"))
                wrapped.append(func)
            solution_class_node = ast.ClassDef(
                name="Solution",
                bases=[],
                keywords=[],
                body=wrapped,  # type: ignore[arg-type]
                decorator_list=[],
                type_params=[],
            )

        if not solution_class_node:
            raise LanguageTransformationError(
                "No Solution class or functions found in programmer code"
            )
        if not solution_method_name:
            raise LanguageTransformationError(
                "No callable method found in Solution class"
            )

        try:
            input_dict = eval(input_data, {"__builtins__": {}}, {})  # noqa: S307
            if "inputdata" not in input_dict:
                raise LanguageTransformationError(
                    "Input data must contain 'inputdata' key"
                )
            input_params = input_dict["inputdata"]
            if not isinstance(input_params, dict):
                raise LanguageTransformationError(
                    "'inputdata' value must be a dict object"
                )
        except (ValueError, SyntaxError, NameError, TypeError) as e:
            raise LanguageTransformationError(f"Failed to parse input data: {e}") from e
        except LanguageTransformationError:
            raise
        except Exception as e:
            raise LanguageTransformationError(
                f"Error processing input data: {e}"
            ) from e

        parts: list[str] = []
        for node in prog_imports:
            parts.append(ast.unparse(node))
        if prog_imports:
            parts.append("")
        for node in helper_class_nodes:
            parts.append(ast.unparse(node))
            parts.append("")
        parts.append(ast.unparse(solution_class_node))
        parts.append("")
        parts.append("# Execute solution")
        parts.append("sol = Solution()")
        param_str = ", ".join(repr(v) for v in input_params.values())
        parts.append(f"print(sol.{solution_method_name}({param_str}))")

        return "\n".join(parts)

    def generate_test_case(
        self, input_str: str, output_str: str, starter_code: str, test_number: int
    ) -> str:
        """Generate a standalone pytest test case from test data and starter code."""
        sig = self._parse_method_signature(starter_code)
        if self._is_stdin_signature(sig):
            return self._gen_stdin_test(
                input_str,
                output_str,
                sig.class_name,
                sig.method_name,
                test_number,
                sig.is_standalone,
            )
        return self._gen_functional_test(
            input_str,
            output_str,
            sig.class_name,
            sig.method_name,
            test_number,
            sig.is_standalone,
        )

    def compose_generator_script(self, generator_code: str, num_inputs: int) -> str:
        """Compose a Python script that executes the generator and prints results."""
        script = self.remove_main_block(generator_code)
        script += f"\n\nprint(generate_test_inputs({num_inputs}))"
        return script

    def remove_main_block(self, code: str) -> str:
        """Remove the top-level 'if __name__ == "__main__":' block."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise LanguageParsingError(f"Failed to parse code: {e}") from e

        new_body = []
        for node in tree.body:
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                is_standard = (
                    isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"
                    and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq)
                    and len(node.test.comparators) == 1
                    and isinstance(node.test.comparators[0], ast.Constant)
                    and node.test.comparators[0].value == "__main__"
                )
                is_reversed = (
                    isinstance(node.test.left, ast.Constant)
                    and node.test.left.value == "__main__"
                    and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq)
                    and len(node.test.comparators) == 1
                    and isinstance(node.test.comparators[0], ast.Name)
                    and node.test.comparators[0].id == "__name__"
                )
                if is_standard or is_reversed:
                    logger.debug("Removing if __name__ == '__main__' block")
                    continue
            new_body.append(node)

        tree.body = new_body
        return ast.unparse(tree)

    def normalize_code(self, code: str) -> str:
        """Normalize Python code by removing comments, docstrings, and extra whitespace."""
        text = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def contains_starter_code(self, code: str, starter_code: str) -> bool:
        """Check if the starter code structure is present in the provided code."""
        normalized_code = self.normalize_code(code)
        normalized_starter = self.normalize_code(starter_code)

        if not normalized_starter.strip():
            return True
        if normalized_starter in normalized_code:
            return True

        def _signatures(text: str) -> List[str]:
            sigs = [f"class {n}" for n in re.findall(r"class\s+(\w+)", text)]
            sigs += [f"def {n}" for n in re.findall(r"def\s+(\w+)\s*\(", text)]
            return sigs

        code_sigs = _signatures(normalized_code)
        starter_sigs = _signatures(normalized_starter)
        if not starter_sigs:
            return False
        return all(s in code_sigs for s in starter_sigs)

    def get_structural_metadata(self, code: str) -> Dict[str, Any]:
        """Extract Python structural metadata (imports, classes, functions)."""
        metadata: Dict[str, list[str]] = {"imports": [], "classes": [], "functions": []}
        try:
            tree = ast.parse(code)
            lines = code.split("\n")
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    start = node.lineno - 1
                    end = (
                        (node.end_lineno - 1) if node.end_lineno is not None else start
                    )
                    metadata["imports"].extend(lines[start : end + 1])
                elif isinstance(node, ast.ClassDef):
                    metadata["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    metadata["functions"].append(node.name)
            metadata["imports"] = list(dict.fromkeys(metadata["imports"]))
        except Exception as e:
            logger.debug(f"Failed to extract structural metadata: {e}")
        return metadata

    def parse_test_inputs(self, outputs: str) -> List[Dict[str, Any]]:
        """
        Parse Python literal outputs using ast.literal_eval, with a fallback to
        restricted eval() for special float values (inf, nan).
        """
        import math

        try:
            val = ast.literal_eval(outputs)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return [val]
            return []
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse test inputs as literal: {e}")
            try:
                safe_ns = {"inf": math.inf, "nan": math.nan, "__builtins__": {}}
                val = eval(outputs, safe_ns)  # noqa: S307
                if isinstance(val, list):
                    return val
                if isinstance(val, dict):
                    return [val]
                logger.debug(
                    f"Eval succeeded but result type {type(val)} is not list or dict"
                )
                return []
            except Exception as eval_error:
                logger.debug(f"Failed to parse with restricted eval: {eval_error}")
                return []

    def get_docstring(self, code: str) -> str:
        """
        Return the docstring of the first function or class in the code.

        Priority:
            1. Class docstring.
            2. First method's docstring (if class has none).
            3. # comments immediately above the method (if class) or function.
        """
        try:
            dedented_code = textwrap.dedent(code)
            tree = ast.parse(dedented_code)
            lines = dedented_code.splitlines()

            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    return ast.get_docstring(node) or self._extract_comments(
                        node, lines
                    )

                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node)
                    if class_doc:
                        return class_doc

                    first_method = next(
                        (n for n in node.body if isinstance(n, ast.FunctionDef)), None
                    )
                    if first_method:
                        method_doc = ast.get_docstring(first_method)
                        if method_doc:
                            return method_doc
                        method_comments = self._extract_comments(first_method, lines)
                        if method_comments:
                            return method_comments

                    return self._extract_comments(node, lines)

        except Exception as e:
            logger.debug(f"Failed to extract docstring: {e}")

        return ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_comments(self, node: ast.AST, lines: List[str]) -> str:
        """Extract # comments immediately preceding a node."""
        comments = []
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return ""
        for i in range(lineno - 2, -1, -1):
            line = lines[i].strip()
            if line.startswith("#"):
                comments.append(line[1:].strip())
            elif line == "":
                continue
            else:
                break
        return "\n".join(reversed(comments))

    def _parse_method_signature(self, starter_code: str) -> _MethodSignature:
        """Parse starter code to extract function/method signature using AST."""
        try:
            tree = ast.parse(starter_code)
        except SyntaxError:
            try:
                suffix = "\n        pass" if "class " in starter_code else "\n    pass"
                tree = ast.parse(starter_code.rstrip() + suffix)
            except SyntaxError as e:
                raise LanguageParsingError(f"Failed to parse starter code: {e}") from e

        # Standalone function takes priority over a class definition
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                params: list[tuple[str, Optional[str]]] = [
                    (arg.arg, ast.unparse(arg.annotation) if arg.annotation else None)
                    for arg in node.args.args
                ]
                return _MethodSignature(
                    class_name=None,
                    method_name=node.name,
                    params=params,
                    return_type=ast.unparse(node.returns) if node.returns else None,
                    is_standalone=True,
                )

        class_node: Optional[ast.ClassDef] = next(
            (n for n in tree.body if isinstance(n, ast.ClassDef)), None
        )
        if not class_node:
            raise LanguageParsingError(
                "No class or function definition found in starter code"
            )

        method_node: Optional[ast.FunctionDef] = next(
            (
                n
                for n in class_node.body
                if isinstance(n, ast.FunctionDef)
                and n.args.args
                and n.args.args[0].arg == "self"
            ),
            None,
        )
        if not method_node:
            raise LanguageParsingError("No instance method found in class")

        params = [
            (arg.arg, ast.unparse(arg.annotation) if arg.annotation else None)
            for arg in method_node.args.args[1:]  # skip 'self'
        ]
        return _MethodSignature(
            class_name=class_node.name,
            method_name=method_node.name,
            params=params,
            return_type=ast.unparse(method_node.returns)
            if method_node.returns
            else None,
            is_standalone=False,
        )

    @staticmethod
    def _is_stdin_signature(sig: _MethodSignature) -> bool:
        """Return True if signature matches the STDIN pattern (input_str:str -> str)."""
        if len(sig.params) != 1:
            return False
        param_name, param_type = sig.params[0]
        return (
            param_name == "input_str"
            and param_type == "str"
            and sig.return_type == "str"
        )

    @staticmethod
    def _gen_stdin_test(
        input_data: str,
        output_data: str,
        class_name: Optional[str],
        method_name: str,
        test_number: int,
        is_standalone: bool,
    ) -> str:
        input_literal = repr(input_data.rstrip("\n"))
        output_literal = repr(output_data.rstrip("\n"))
        if is_standalone:
            return (
                f"def test_case_{test_number}():\n"
                f"    input_str = {input_literal}\n"
                f"    expected_output = {output_literal}\n"
                f"    assert {method_name}(input_str) == expected_output\n"
            )
        return (
            f"def test_case_{test_number}():\n"
            f"    solution = {class_name}()\n"
            f"    input_str = {input_literal}\n"
            f"    expected_output = {output_literal}\n"
            f"    assert solution.{method_name}(input_str) == expected_output\n"
        )

    @staticmethod
    def _gen_functional_test(
        input_data: str,
        output_data: str,
        class_name: Optional[str],
        method_name: str,
        test_number: int,
        is_standalone: bool,
    ) -> str:
        input_lines = [line.strip() for line in input_data.split("\n") if line.strip()]
        input_lines_repr = repr(input_lines)
        output_repr = repr(output_data)
        if is_standalone:
            return (
                f"def test_case_{test_number}():\n"
                f"    import ast\n"
                f"    input_lines = {input_lines_repr}\n"
                f"    args = [ast.literal_eval(line) for line in input_lines]\n"
                f"    expected_output = ast.literal_eval({output_repr})\n"
                f"    assert {method_name}(*args) == expected_output\n"
            )
        return (
            f"def test_case_{test_number}():\n"
            f"    import ast\n"
            f"    solution = {class_name}()\n"
            f"    input_lines = {input_lines_repr}\n"
            f"    args = [ast.literal_eval(line) for line in input_lines]\n"
            f"    expected_output = ast.literal_eval({output_repr})\n"
            f"    assert solution.{method_name}(*args) == expected_output\n"
        )
