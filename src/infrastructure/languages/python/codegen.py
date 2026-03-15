"""Codegen logic for Python code snippet composition and file formatting."""

import ast
from typing import Any, Optional

from coevolution.core.interfaces.language import (
    LanguageParsingError,
    LanguageTransformationError,
)

from .ast import remove_main_block


def _parse_val(val: str) -> Any:
    """Try to parse as JSON, fallback to raw string."""
    try:
        import json

        return json.loads(val)
    except Exception:
        return val


def _try_cast_expected(expected: Any, return_type: Optional[str]) -> Any:
    """Attempt to cast expected output to the return type if possible."""
    if not return_type:
        return expected

    try:
        if return_type == "int":
            return int(expected)
        if return_type == "str":
            return str(expected)
        if return_type == "float":
            return float(expected)
        if return_type == "bool":
            if isinstance(expected, str):
                return expected.lower() == "true"
            return bool(expected)
    except Exception:
        pass
    return expected


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
                if isinstance(member, ast.FunctionDef) and not member.name.startswith(
                    "_"
                ):
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
        raise LanguageTransformationError("No callable method found in Solution class")

    try:
        import json

        input_dict = None
        try:
            input_dict = json.loads(input_data)
        except (json.JSONDecodeError, TypeError):
            try:
                # Try parsing as a Python literal (often used in tests)
                input_dict = ast.literal_eval(input_data)
            except Exception:
                # Last resort: treat input_data as the actual value if it's already a dict
                if isinstance(input_data, dict):
                    input_dict = input_data
                else:
                    # If it's a string that's not JSON/literal, it might be meant as the input_str for stdin style,
                    # but here we expect a dict for functional style.
                    # We'll try to wrap it if it's not a dict.
                    input_dict = {"inputdata": input_data}

        # If we got a dict, check if it has 'inputdata' key, otherwise assume it is the inputdata
        if isinstance(input_dict, dict):
            input_params = input_dict.get("inputdata", input_dict)
        else:
            input_params = input_dict

        if not isinstance(input_params, dict):
            # Some old tests pass non-dict as input_data for functions with 1 param
            # or we might be in a weird state.
            raise LanguageTransformationError(
                f"Input data must be a dict or contain 'inputdata' key, got {type(input_params)}"
            )

    except Exception as e:
        raise LanguageTransformationError(f"Failed to parse input data: {e}") from e

    parts: list[str] = [
        "import json",
        "",
    ]
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
    # Use keyword arguments for safety
    param_str = ", ".join(f"{k}={repr(v)}" for k, v in input_params.items())
    parts.append(f"print(json.dumps(sol.{solution_method_name}({param_str})))")

    return "\n".join(parts)


def gen_stdin_test(
    input_data: str,
    output_data: str,
    class_name: Optional[str],
    method_name: str,
    test_number: int,
    is_standalone: bool,
    return_type: Optional[str],
) -> str:
    """Generate a stdin test function from inputs."""
    input_val = _parse_val(input_data)
    output_val = _parse_val(output_data)

    # Intelligence at generation time: align expected output to signature type
    # This prevents '16' == 16 failures.
    expected_aligned = _try_cast_expected(output_val, return_type)

    input_repr = repr(input_val)
    output_repr = repr(expected_aligned)

    if is_standalone:
        return (
            f"def test_case_{test_number}():\n"
            f"    input_str = {input_repr}\n"
            f"    expected_output = {output_repr}\n"
            f"    actual_output = {method_name}(input_str)\n"
            f"    try:\n"
            f"        assert actual_output == expected_output\n"
            f"    except AssertionError:\n"
            f"        # Type-agnostic fallback\n"
            f"        assert str(actual_output).strip() == str(expected_output).strip()\n"
        )
    return (
        f"def test_case_{test_number}():\n"
        f"    solution = {class_name}()\n"
        f"    input_str = {input_repr}\n"
        f"    expected_output = {output_repr}\n"
        f"    actual_output = solution.{method_name}(input_str)\n"
        f"    try:\n"
        f"        assert actual_output == expected_output\n"
        f"    except AssertionError:\n"
        f"        # Type-agnostic fallback\n"
        f"        assert str(actual_output).strip() == str(expected_output).strip()\n"
    )


def gen_functional_test(
    input_data: str,
    output_data: str,
    class_name: Optional[str],
    method_name: str,
    test_number: int,
    is_standalone: bool,
    return_type: Optional[str],
) -> str:
    """Generate a functional test case from inputs."""
    input_lines = [line.strip() for line in input_data.split("\n") if line.strip()]

    parsed_args = [_parse_val(line) for line in input_lines]
    parsed_output = _parse_val(output_data)

    # Intelligence at generation time: align expected output to signature type
    expected_aligned = _try_cast_expected(parsed_output, return_type)

    args_repr = repr(parsed_args)
    output_repr = repr(expected_aligned)

    if is_standalone:
        return (
            f"def test_case_{test_number}():\n"
            f"    args = {args_repr}\n"
            f"    expected_output = {output_repr}\n"
            f"    actual_output = {method_name}(*args)\n"
            f"    try:\n"
            f"        assert actual_output == expected_output\n"
            f"    except AssertionError:\n"
            f"        # Type-agnostic fallback\n"
            f"        assert str(actual_output).strip() == str(expected_output).strip()\n"
        )
    return (
        f"def test_case_{test_number}():\n"
        f"    solution = {class_name}()\n"
        f"    args = {args_repr}\n"
        f"    expected_output = {output_repr}\n"
        f"    actual_output = solution.{method_name}(*args)\n"
        f"    try:\n"
        f"        assert actual_output == expected_output\n"
        f"    except AssertionError:\n"
        f"        # Type-agnostic fallback\n"
        f"        assert str(actual_output).strip() == str(expected_output).strip()\n"
    )


def compose_generator_script(generator_code: str, num_inputs: int) -> str:
    """Compose a Python script that executes the generator and prints results."""
    script = "import json\n"
    script += remove_main_block(generator_code)
    script += f"\n\nprint(json.dumps(generate_test_inputs({num_inputs})))"
    return script
    return script
