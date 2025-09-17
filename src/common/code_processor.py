import re
from typing import Dict, List, Tuple, Optional


class CodeProcessor:
    """
    A class to process and extract relevant code segments from a given code string.
    """

    def extract_function_name_from_prompt(self, problem_prompt: str) -> str:
        """
        Extracts the function name from the problem prompt.
        """
        lines = problem_prompt.split('\n')
        for line in lines:
            if line.strip().startswith('def '):
                # Extract function name
                func_name = line.strip().split('def ')[1].split('(')[0]
                return func_name
        return ""

    def remove_comments(self, code: str) -> str:
        """
        Remove single line (#) and multiline (''' or \"\"\") comments from Python code.
        """
        # Remove multiline comments (triple quotes)
        # This pattern matches ''' or """ followed by any content until the closing quotes
        code = re.sub(r'(\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\")', '', code)

        # Remove single line comments (# comments)
        # This pattern matches # and everything after it until the end of line
        code = re.sub(r'#.*?$', '', code, flags=re.MULTILINE)

        # Clean up extra whitespace and empty lines
        lines: list[str] = code.split('\n')
        cleaned_lines: list[str] = []

        for line in lines:
            stripped = line.rstrip()  # Remove trailing whitespace
            # Keep line if not empty or previous line wasn't empty
            if stripped or (cleaned_lines and cleaned_lines[-1].strip()):
                cleaned_lines.append(stripped)

        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        return '\n'.join(cleaned_lines)

    def extract_code_block_from_response(self, response: str) -> str:
        """
        Extracts the first Python code block from the LLM response.
        If no code block is found, returns the entire response.
        """
        pattern = re.compile(r'```(?:[Pp]ython|[Pp]y)?\s*([\s\S]+?)\s*```')
        match = pattern.search(response)
        if match:
            return match.group(1)
        return response

    def _find_functions_and_imports(self, lines: List[str]) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
        """Find all function definitions and import statements."""
        import_lines = []
        function_definitions = {}

        i = 0
        while i < len(lines):
            stripped_line = lines[i].strip()

            # Collect import lines
            if stripped_line.startswith(('import ', 'from ')):
                import_lines.append(lines[i])
                i += 1
                continue

            # Find function definitions
            if stripped_line.startswith('def '):
                func_name = stripped_line.split(
                    'def ')[1].split('(')[0].strip()
                start_idx = i

                # Find end of this function
                end_idx = self._find_function_end(lines, i)
                function_definitions[func_name] = (start_idx, end_idx)
                i = end_idx + 1
            else:
                i += 1

        return import_lines, function_definitions

    def _find_function_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end index of a function definition."""
        for j in range(start_idx + 1, len(lines)):
            line = lines[j]
            # Function ends when we hit unindented non-empty line
            if (line.strip() and
                not line.startswith('\t') and
                    not line.startswith('    ')):
                return j - 1

        # If no unindented line found, use end of string
        end_idx = len(lines) - 1

        # Remove trailing empty lines
        while end_idx > start_idx and not lines[end_idx].strip():
            end_idx -= 1

        return end_idx

    def _build_completion(self, import_lines: List[str], helper_functions: List[str],
                          function_definitions: Dict[str, Tuple[int, int]],
                          target_body_lines: List[str], lines: List[str]) -> str:
        """Build the final completion code block."""
        result_lines = []

        # Add indented imports
        if import_lines:
            indented_imports = ['    ' + line.strip() for line in import_lines]
            result_lines.extend(indented_imports)
            result_lines.append('')

        # Add helper functions with proper indentation
        for helper_name in helper_functions:
            helper_start, helper_end = function_definitions[helper_name]
            helper_lines = lines[helper_start:helper_end + 1]
            indented_helper = ['    ' + line for line in helper_lines]
            result_lines.extend(indented_helper)
            result_lines.append('')

        # Add target function body
        result_lines.extend(target_body_lines)

        # Clean up trailing empty lines
        while result_lines and not result_lines[-1].strip():
            result_lines.pop()

        return '\n'.join(result_lines)

    def extract_function_with_helpers(self, code_string: str, target_function_name: str) -> str:
        """Extract target function body with helper functions and imports."""
        lines = code_string.split('\n')
        import_lines, function_definitions = self._find_functions_and_imports(
            lines)

        if target_function_name not in function_definitions:
            return ""

        target_start, target_end = function_definitions[target_function_name]
        target_body_lines = lines[target_start + 1:target_end + 1]

        # Get helper functions (all except target)
        helper_functions = [
            name for name in function_definitions.keys()
            if name != target_function_name
        ]

        return self._build_completion(
            import_lines, helper_functions, function_definitions,
            target_body_lines, lines
        )
