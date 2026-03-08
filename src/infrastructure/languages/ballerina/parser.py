"""Parser and metadata extraction utilities for Ballerina."""

import ast
import re
from typing import Any, Dict, List

from loguru import logger
from coevolution.core.interfaces.language import LanguageParsingError


# Regex patterns heavily used in parsing
FUNCTION_PATTERN = re.compile(
    r"(public\s+)?function\s+(\w+)\s*\([^)]*\)(\s+returns\s+[^{]+)?\s*\{",
    re.MULTILINE,
)
TEST_PATTERN = re.compile(
    r"@test:Config\s*(?:\{[^}]*\})?\s*function\s+(\w+)\s*\([^)]*\)\s*\{",
    re.MULTILINE,
)
MAIN_PATTERN = re.compile(
    r"public\s+function\s+main\s*\([^)]*\)\s*\{[^}]*\}",
    re.MULTILINE | re.DOTALL,
)


def is_syntax_valid(code: str) -> bool:
    """Validate Ballerina syntax (stubbed for now)."""
    return True


def basic_syntax_check(code: str) -> bool:
    """Basic fallback syntax checking."""
    has_function = "function" in code
    has_balanced_braces = code.count("{") == code.count("}")
    has_balanced_parens = code.count("(") == code.count(")")
    return has_function and has_balanced_braces and has_balanced_parens


def extract_test_names(test_code: str) -> List[str]:
    """Extract test function names annotated with @test:Config."""
    matches = TEST_PATTERN.findall(test_code)
    return list(matches)


def split_tests(test_code: str) -> List[str]:
    """Split Ballerina test code into individual test functions."""
    tests = []
    matches = list(TEST_PATTERN.finditer(test_code))

    if not matches:
        return []

    for i, match in enumerate(matches):
        start = match.start()
        match_end = match.end()
        search_end = matches[i + 1].start() if i < len(matches) - 1 else len(test_code)

        brace_count = 1
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


def remove_main_block(code: str) -> str:
    """Remove `public function main(...) { ... }` block."""
    code = MAIN_PATTERN.sub("", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def normalize_code(code: str) -> str:
    """Normalize code by removing comments and extra whitespace."""
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    lines = [line.strip() for line in code.split("\n") if line.strip()]
    return "\n".join(lines)


def get_structural_metadata(code: str) -> Dict[str, Any]:
    """Extract information about functions, imports, and has_main flag."""
    metadata: Dict[str, Any] = {"functions": [], "imports": [], "has_main": False}
    
    func_matches = FUNCTION_PATTERN.findall(code)
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

    import_pattern = re.compile(r"import\s+([\w/:.]+);?", re.MULTILINE)
    metadata["imports"] = import_pattern.findall(code)
    return metadata


def parse_test_inputs(outputs: str) -> List[Dict[str, Any]]:
    """Parse runtime input data from stdout string outputs."""
    test_cases = []
    try:
        lines = [line.strip() for line in outputs.strip().split("\n") if line.strip()]
        i = 0
        while i < len(lines):
            if lines[i].startswith("input:"):
                input_val = lines[i].replace("input:", "").strip()
                output_val = ""
                if i + 1 < len(lines) and lines[i + 1].startswith("output:"):
                    output_val = lines[i + 1].replace("output:", "").strip()
                    i += 2
                else:
                    i += 1
                test_cases.append({"input": input_val, "output": output_val})
            else:
                i += 1

        if not test_cases and outputs.strip():
            try:
                parsed = ast.literal_eval(outputs)
                if isinstance(parsed, list):
                    test_cases = parsed
            except (ValueError, SyntaxError):
                pass
    except Exception as e:
        logger.debug(f"Error parsing test inputs: {e}")

    return test_cases
