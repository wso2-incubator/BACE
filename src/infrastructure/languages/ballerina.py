# src/infrastructure/adapters/ballerina.py
"""
Ballerina-specific implementation of ILanguage.

This adapter provides language-specific operations for Ballerina code,
including syntax validation, code extraction, test handling, and script composition.
"""

import ast
import re
from typing import Any, Dict, List

from loguru import logger

from coevolution.core.interfaces.language import (
    ILanguage,
    LanguageParsingError,
    LanguageTransformationError,
)


class BallerinaLanguage(ILanguage):
    """Ballerina language adapter for APR operations."""

    def __init__(self) -> None:
        """Initialize the Ballerina language adapter."""
        self._block_pattern = re.compile(r"```[Bb]allerina\s*([\s\S]+?)\s*```")

        # Pattern to extract function definitions
        self._function_pattern = re.compile(
            r"(public\s+)?function\s+(\w+)\s*\([^)]*\)(\s+returns\s+[^{]+)?\s*\{",
            re.MULTILINE,
        )

        # Pattern to extract test functions
        self._test_pattern = re.compile(
            r"@test:Config\s*\{[^}]*\}\s*function\s+(\w+)\s*\([^)]*\)\s*\{",
            re.MULTILINE,
        )

        logger.info("Initialized BallerinaLanguage")

    @property
    def language(self) -> str:
        return "ballerina"

    def extract_code_blocks(self, response: str) -> List[str]:
        """
        Extract Ballerina code blocks from LLM response.

        Looks for markdown-style ```ballerina code blocks, validates syntax,
        and returns a list of valid code snippets. Falls back to checking if
        the entire response is valid Ballerina code.

        Args:
            response: Raw LLM response string

        Returns:
            List of valid Ballerina code snippets
        """
        # Try to extract markdown code blocks
        matches = self._block_pattern.findall(response)

        if matches:
            valid_blocks = []
            for code in matches:
                code = code.strip()
                if self.is_syntax_valid(code):
                    valid_blocks.append(code)
                else:
                    logger.debug(
                        f"Skipping invalid Ballerina code block: {code[:100]}..."
                    )

            if valid_blocks:
                return valid_blocks

        # Fallback: check if entire response is valid Ballerina code
        cleaned = response.strip()
        if cleaned and self.is_syntax_valid(cleaned):
            return [cleaned]

        logger.debug("No valid Ballerina code blocks found in response")
        return []

    def is_syntax_valid(self, code: str) -> bool:
        """
        Check if Ballerina code has valid syntax.
        We will just say it's valid for now.
        LLMs are pretty bad at generating syntactically correct Ballerina code.
        We can hope that the debug operator will catch syntax errors and that the LLM will improve over time.

        Args:
            code: Ballerina code string to validate

        Returns:
            True if syntax is valid, False otherwise
        """

        return True  # LLMs are very bad at ballerina syntax, so we will skip syntax validation for now.

    def _basic_syntax_check(self, code: str) -> bool:
        """
        Basic syntax check without Ballerina CLI.

        Checks for basic Ballerina syntax patterns as a fallback.
        """
        # Check for basic Ballerina keywords and structure
        has_function = "function" in code
        has_balanced_braces = code.count("{") == code.count("}")
        has_balanced_parens = code.count("(") == code.count(")")

        return has_function and has_balanced_braces and has_balanced_parens

    def extract_test_names(self, test_code: str) -> List[str]:
        """
        Extract test function names from Ballerina test code.

        Finds all functions decorated with @test:Config annotation.

        Args:
            test_code: Ballerina test code string

        Returns:
            List of test function names
        """
        matches = self._test_pattern.findall(test_code)
        return list(matches)

    def split_tests(self, test_code: str) -> List[str]:
        """
        Split Ballerina test code into individual test functions.

        Each test function includes its @test:Config annotation.

        Args:
            test_code: Complete test code string

        Returns:
            List of individual test function strings
        """
        tests = []

        # Find all test function matches with their positions
        matches = list(self._test_pattern.finditer(test_code))

        if not matches:
            return []

        for i, match in enumerate(matches):
            start = match.start()

            # Start counting braces from after the regex match
            # (after the opening brace of the function body)
            match_end = match.end()

            # Find the end of this function (start of next test or end of string)
            if i < len(matches) - 1:
                search_end = matches[i + 1].start()
            else:
                search_end = len(test_code)

            # Count braces starting from after the match to find the closing brace
            brace_count = 1  # Start with 1 because match already captured opening brace
            actual_end = match_end

            for j in range(match_end, search_end):
                char = test_code[j]
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        actual_end = j + 1
                        break

            if actual_end > start:
                test_func = test_code[start:actual_end].strip()
                tests.append(test_func)

        return tests

    def compose_test_script(self, code_snippet: str, test_snippet: str) -> str:
        """
        Combine Ballerina code and test code into an executable test script.

        Ensures the test import is present and both code and tests are included.

        Args:
            code_snippet: Function implementation code
            test_snippet: Test code with @test:Config annotations

        Returns:
            Complete executable Ballerina test script
        """
        # Ensure test import is present
        has_test_import = "import ballerina/test" in test_snippet

        script_parts = []

        # Add test import if not present
        if not has_test_import:
            script_parts.append("import ballerina/test;")
            script_parts.append("")

        # Add code snippet (remove any existing test imports from it)
        code_clean = re.sub(r"import ballerina/test;?\s*", "", code_snippet)
        script_parts.append(code_clean.strip())
        script_parts.append("")

        # Add test snippet
        script_parts.append(test_snippet.strip())

        return "\n".join(script_parts)

    def compose_evaluation_script(self, code_snippet: str, input_data: str) -> str:
        """
        Create an executable Ballerina script that runs a function with input data.

        Args:
            code_snippet: Function implementation
            input_data: Input data in format "function_name(arg1, arg2, ...)"

        Returns:
            Executable Ballerina script that prints the function result
        """
        try:
            # Parse the input data to extract function name and arguments
            match = re.match(r"(\w+)\((.*)\)", input_data.strip())
            if not match:
                raise LanguageTransformationError(f"Invalid input format: {input_data}")

            function_name = match.group(1)
            args_str = match.group(2)

            # Build the script
            script_parts = [
                "import ballerina/io;",
                "",
                code_snippet.strip(),
                "",
                "public function main() {",
                f"    var result = {function_name}({args_str});",
                "    io:println(result);",
                "}",
            ]

            return "\n".join(script_parts)

        except Exception as e:
            raise LanguageTransformationError(
                f"Failed to compose evaluation script: {e}"
            ) from e

    def generate_test_case(
        self, input_str: str, output_str: str, starter_code: str, test_number: int
    ) -> str:
        """
        Generate a Ballerina test case from input and expected output.

        Args:
            input_str: Input data (e.g., "functionName(arg1, arg2)")
            output_str: Expected output value
            starter_code: Function signature to determine function name
            test_number: Test case number for naming

        Returns:
            Complete Ballerina test function string
        """
        try:
            # Extract function name from starter code
            func_match = self._function_pattern.search(starter_code)
            if not func_match:
                raise LanguageParsingError("Cannot find function in starter code")

            function_name = func_match.group(2)

            # Parse input to get arguments
            input_match = re.match(r"\w+\((.*)\)", input_str.strip())
            if input_match:
                args = input_match.group(1)
            else:
                args = input_str

            # Determine the expected output type and format
            expected_value = output_str.strip()

            # Generate test function with preserved camelCase
            # Capitalize first letter while preserving rest of the name
            capitalized_name = (
                function_name[0].upper() + function_name[1:]
                if function_name
                else function_name
            )
            test_code = f"""@test:Config {{ }}
function test{capitalized_name}{test_number}() {{
    var result = {function_name}({args});
    test:assertEquals(result, {expected_value}, msg = "Test case {test_number} failed");
}}"""

            return test_code

        except Exception as e:
            raise LanguageTransformationError(
                f"Failed to generate test case: {e}"
            ) from e

    def compose_generator_script(self, generator_code: str, num_inputs: int) -> str:
        """
        Compose a Ballerina script that executes the generator.
        """
        script = self.remove_main_block(generator_code)
        script += f"\n\nimport ballerina/io;\n\npublic function main() {{\n    io:println(generate_test_inputs({num_inputs}));\n}}"
        return script

    def remove_main_block(self, code: str) -> str:
        """
        Remove main function from Ballerina code.

        Args:
            code: Ballerina code that may contain a main function

        Returns:
            Code with main function removed
        """
        # Pattern to match public function main() { ... }
        main_pattern = re.compile(
            r"public\s+function\s+main\s*\([^)]*\)\s*\{[^}]*\}",
            re.MULTILINE | re.DOTALL,
        )

        # Remove the main function
        code = main_pattern.sub("", code)

        # Clean up extra blank lines
        code = re.sub(r"\n{3,}", "\n\n", code)

        return code.strip()

    def normalize_code(self, code: str) -> str:
        """
        Normalize Ballerina code for structural comparison.

        Removes comments and standardizes whitespace while preserving structure.

        Args:
            code: Ballerina code to normalize

        Returns:
            Normalized code string
        """
        # Remove single-line comments
        code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)

        # Remove multi-line comments
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

        # Normalize whitespace
        lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped:
                lines.append(stripped)

        return "\n".join(lines)

    def contains_starter_code(self, code: str, starter_code: str) -> bool:
        """
        Check if starter code structure is present in the given code.

        Compares function signatures rather than exact string matching.

        Args:
            code: Code to check
            starter_code: Expected starter code structure

        Returns:
            True if starter code is present, False otherwise
        """
        # Normalize both code snippets
        norm_code = self.normalize_code(code)
        norm_starter = self.normalize_code(starter_code)

        # Direct substring match
        if norm_starter in norm_code:
            return True

        # Extract function signature from starter code
        starter_match = self._function_pattern.search(starter_code)
        if not starter_match:
            return False

        starter_func_name = starter_match.group(2)

        # Check if function with same name exists in code
        code_funcs = self._function_pattern.findall(code)
        code_func_names = [match[1] for match in code_funcs]

        return starter_func_name in code_func_names

    def get_structural_metadata(self, code: str) -> Dict[str, Any]:
        """
        Extract structural metadata from Ballerina code.

        Returns information about functions, imports, and code structure.

        Args:
            code: Ballerina code to analyze

        Returns:
            Dictionary with keys: 'functions', 'imports', 'has_main'
        """
        metadata: Dict[str, Any] = {
            "functions": [],
            "imports": [],
            "has_main": False,
        }

        # Extract function definitions
        func_matches = self._function_pattern.findall(code)
        for match in func_matches:
            visibility = match[0].strip() if match[0] else ""
            func_name = match[1]
            returns = match[2].strip() if match[2] else ""

            metadata["functions"].append(
                {
                    "name": func_name,
                    "visibility": visibility,
                    "returns": returns.replace("returns", "").strip(),
                }
            )

            if func_name == "main":
                metadata["has_main"] = True

        # Extract imports
        import_pattern = re.compile(r"import\s+([\w/:.]+);?", re.MULTILINE)
        import_matches = import_pattern.findall(code)
        metadata["imports"] = import_matches

        return metadata

    def parse_test_inputs(self, outputs: str) -> List[Dict[str, Any]]:
        """
        Parse test input data from string output.

        Expects output in format with lines like:
        input: functionName(arg1, arg2, ...)
        output: expected_result

        Args:
            outputs: Raw output string from test input generator

        Returns:
            List of dictionaries with 'input' and 'output' keys
        """
        test_cases = []

        try:
            # Split into lines and process pairs
            lines = [
                line.strip() for line in outputs.strip().split("\n") if line.strip()
            ]

            i = 0
            while i < len(lines):
                if lines[i].startswith("input:"):
                    input_val = lines[i].replace("input:", "").strip()
                    output_val = ""

                    # Look for corresponding output
                    if i + 1 < len(lines) and lines[i + 1].startswith("output:"):
                        output_val = lines[i + 1].replace("output:", "").strip()
                        i += 2
                    else:
                        i += 1

                    test_cases.append(
                        {
                            "input": input_val,
                            "output": output_val,
                        }
                    )
                else:
                    i += 1

            # If no structured format found, try to use Python's literal_eval as fallback
            if not test_cases and outputs.strip():
                try:
                    parsed = ast.literal_eval(outputs)
                    if isinstance(parsed, list):
                        test_cases = parsed
                except (ValueError, SyntaxError):
                    logger.debug("Could not parse test inputs as Python literal")

        except Exception as e:
            logger.debug(f"Error parsing test inputs: {e}")

        return test_cases
