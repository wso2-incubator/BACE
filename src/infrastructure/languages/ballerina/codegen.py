"""String codegen and templating tools for Ballerina language execution."""

import re
from coevolution.core.interfaces.language import (
    LanguageParsingError,
    LanguageTransformationError,
)
from .parser import (
    FUNCTION_PATTERN,
    remove_main_block,
)


def compose_test_script(code_snippet: str, test_snippet: str) -> str:
    """Combine implementation and test cases into an executable test file."""
    has_test_import = "import ballerina/test" in test_snippet
    script_parts = []

    if not has_test_import:
        script_parts.append("import ballerina/test;")
        script_parts.append("")

    code_clean = re.sub(r"import ballerina/test;?\s*", "", code_snippet)
    script_parts.append(code_clean.strip())
    script_parts.append("")
    script_parts.append(test_snippet.strip())

    return "\n".join(script_parts)


def compose_evaluation_script(code_snippet: str, input_data: str) -> str:
    """Create an executable script that runs a function with JSON input data."""
    try:
        import json
        try:
            input_dict = json.loads(input_data)
            inner_input = input_dict.get("inputdata", input_dict)
        except (json.JSONDecodeError, TypeError):
            # Fallback for old tests passing raw strings
            inner_input = input_data
        
        # We need the function name from the code snippet
        match = FUNCTION_PATTERN.search(code_snippet)
        if not match:
            raise LanguageTransformationError("Cannot find function in code snippet")
        function_name = match.group(2)
        
        # Build arguments string in the correct order using the function signature
        from .parser import get_function_signature
        sig = get_function_signature(code_snippet)
        param_names = list(sig.keys())

        args_list = []
        for name in param_names:
            val = inner_input.get(name)
            # Complicated types should be handled as JSON and cast
            # Ballerina's ensureType() is perfect for this
            args_list.append(f"check {json.dumps(val)}.ensureType()")

        args_str = ", ".join(args_list)

        script_parts = [
            "import ballerina/io;",
            "import ballerina/lang.value;",
            "",
            code_snippet.strip(),
            "",
            "public function main() returns error? {",
            f"    var result = {function_name}({args_str});",
            "    io:println(value:toJsonString(result));",
            "}",
        ]
        return "\n".join(script_parts)
    except Exception as e:
        raise LanguageTransformationError(
            f"Failed to compose evaluation script: {e}"
        ) from e


def generate_test_case(
    input_str: str, output_str: str, starter_code: str, test_number: int
) -> str:
    """Compile input strings into `@test:Config` snippets."""
    try:
        func_match = FUNCTION_PATTERN.search(starter_code)
        if not func_match:
            raise LanguageParsingError("Cannot find function in starter code")

        function_name = func_match.group(2)

        input_match = re.match(r"\w+\((.*)\)", input_str.strip())
        args = input_match.group(1) if input_match else input_str

        expected_value = output_str.strip()

        capitalized_name = (
            function_name[0].upper() + function_name[1:]
            if function_name
            else function_name
        )
        test_code = f"""@test:Config
function test{capitalized_name}{test_number}() {{
    var result = {function_name}({args});
    test:assertEquals(result, {expected_value}, msg = "Test case {test_number} failed");
}}"""
        return test_code
    except Exception as e:
        raise LanguageTransformationError(f"Failed to generate test case: {e}") from e


def compose_generator_script(generator_code: str, num_inputs: int) -> str:
    """Create an executable script wrapper invoking `generate_test_inputs`."""
    script = remove_main_block(generator_code)
    script += "\n\nimport ballerina/io;"
    script += "\nimport ballerina/lang.value;"
    script += f"\n\npublic function main() {{\n    io:println(value:toJsonString(generate_test_inputs({num_inputs})));\n}}"
    return script
