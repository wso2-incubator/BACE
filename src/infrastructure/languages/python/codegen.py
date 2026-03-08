"""Codegen logic for Python code snippet composition and file formatting."""

import ast
from typing import Optional

from coevolution.core.interfaces.language import (
    LanguageParsingError,
    LanguageTransformationError,
)
from .ast import (
    remove_main_block,
)


def compose_test_script(code_snippet: str, test_snippet: str) -> str:
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


def compose_evaluation_script(code_snippet: str, input_data: str) -> str:
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


def gen_stdin_test(
    input_data: str,
    output_data: str,
    class_name: Optional[str],
    method_name: str,
    test_number: int,
    is_standalone: bool,
) -> str:
    """Generate a stdin test function from inputs."""
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


def gen_functional_test(
    input_data: str,
    output_data: str,
    class_name: Optional[str],
    method_name: str,
    test_number: int,
    is_standalone: bool,
) -> str:
    """Generate a functional test case from inputs."""
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


def compose_generator_script(generator_code: str, num_inputs: int) -> str:
    """Compose a Python script that executes the generator and prints results."""
    script = remove_main_block(generator_code)
    script += f"\n\nprint(generate_test_inputs({num_inputs}))"
    return script
