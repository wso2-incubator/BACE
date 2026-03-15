"""AST manipulation utilities for Python language parsing."""

import ast
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from coevolution.core.interfaces.language import LanguageParsingError


@dataclass
class MethodSignature:
    """Parsed method / function signature."""

    class_name: Optional[str]  # None for standalone functions
    method_name: str
    params: list[tuple[str, Optional[str]]]  # [(name, type_annotation), ...]
    return_type: Optional[str]
    is_standalone: bool = False


def is_syntax_valid(code: str) -> bool:
    """Validate Python syntax using ast.parse."""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, OverflowError):
        return False


def extract_test_names(test_code: str) -> List[str]:
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


def split_tests(test_code: str) -> List[str]:
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


def remove_main_block(code: str) -> str:
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


def get_structural_metadata(code: str) -> Dict[str, Any]:
    """Extract Python structural metadata (imports, classes, functions)."""
    metadata: Dict[str, list[str]] = {"imports": [], "classes": [], "functions": []}
    try:
        tree = ast.parse(code)
        lines = code.split("\n")
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = node.lineno - 1
                end = (node.end_lineno - 1) if node.end_lineno is not None else start
                metadata["imports"].extend(lines[start : end + 1])
            elif isinstance(node, ast.ClassDef):
                metadata["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                metadata["functions"].append(node.name)
        metadata["imports"] = list(dict.fromkeys(metadata["imports"]))
    except Exception as e:
        logger.debug(f"Failed to extract structural metadata: {e}")
    return metadata


def parse_test_inputs(outputs: str) -> List[Dict[str, Any]]:
    """
    Parse Python literal or JSON outputs, supporting special float values
    (inf, nan, Infinity, NaN).
    """
    import json
    import math

    # Try JSON first (generator wrapper emits JSON)
    try:
        val = json.loads(outputs)
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            return [val]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback to ast.literal_eval
    try:
        val = ast.literal_eval(outputs)
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            return [val]
        logger.debug(
            f"literal_eval succeeded but result type {type(val)} is not list or dict"
        )
        return []
    except (ValueError, SyntaxError) as e:
        logger.debug(f"Failed to parse test inputs as literal: {e}")

    # Final fallback: restricted eval with support for both Python and JSON float tokens
    try:
        safe_ns = {
            "inf": math.inf,
            "nan": math.nan,
            "Infinity": math.inf,
            "-Infinity": -math.inf,
            "NaN": math.nan,
            "__builtins__": {},
        }
        val = eval(outputs, safe_ns)  # noqa: S307
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            return [val]
        logger.debug(f"Eval succeeded but result type {type(val)} is not list or dict")
        return []
    except Exception as eval_error:
        logger.debug(f"Failed to parse with restricted eval: {eval_error}")
        return []


def _extract_comments(node: ast.AST, lines: List[str]) -> str:
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


def get_docstring(code: str) -> Optional[str]:
    """
    Return the docstring or leading comments for the first function or class
    defined in the given code, if any.

    Returns:
        str: The extracted docstring or preceding `#` comments.
        None: If no suitable docstring/comments are found or if the code
            cannot be parsed.
    """
    try:
        dedented_code = textwrap.dedent(code)
        tree = ast.parse(dedented_code)
        lines = dedented_code.splitlines()

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node) or _extract_comments(node, lines)
                return doc if doc else None

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
                    method_comments = _extract_comments(first_method, lines)
                    if method_comments:
                        return method_comments

                cls_doc = _extract_comments(node, lines)
                return cls_doc if cls_doc else None

    except Exception as e:
        logger.debug(f"Failed to extract docstring: {e}")

    return None


def parse_method_signature(starter_code: str) -> MethodSignature:
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
            return MethodSignature(
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
    return MethodSignature(
        class_name=class_node.name,
        method_name=method_node.name,
        params=params,
        return_type=ast.unparse(method_node.returns) if method_node.returns else None,
        is_standalone=False,
    )


def is_stdin_signature(sig: MethodSignature) -> bool:
    """Return True if signature matches the STDIN pattern (input_str:str -> str)."""
    # Filter out 'self' if this is a method
    params = sig.params
    if params and params[0][0] == "self":
        params = params[1:]

    if len(params) != 1:
        return False
    param_name, param_type = params[0]
    return (
        param_name == "input_str" and param_type == "str" and sig.return_type == "str"
    )


def get_function_signature(code: str) -> Dict[str, str]:
    """Extract parameter names and types from Python code."""
    try:
        sig = parse_method_signature(code)
        return {name: (ptype or "Any") for name, ptype in sig.params}
    except Exception as e:
        logger.debug(f"Failed to get Python function signature: {e}")
        return {}
