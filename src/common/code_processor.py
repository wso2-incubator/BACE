import re
from typing import Dict, List, Tuple, Optional, Union, TypedDict


class CodeStructure(TypedDict):
    import_lines: List[str]
    function_definitions: Dict[str, Tuple[int, int]]
    class_definitions: Dict[str, Tuple[int, int]]


class CodeProcessor:
    """
    A class to process and extract relevant code segments from a given code string.
    """

    def extract_function_name_from_problem(self, problem_prompt: str) -> str:
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
        Extracts the first Python code block (```python ... ```) from the LLM response.
        If no Python code block is found, returns an empty string.
        """
        pattern = re.compile(r'```[Pp]ython\s*([\s\S]+?)\s*```')
        match = pattern.search(response)
        if match:
            return match.group(1)
        return ""

    def _parse_code_structure(self, lines: List[str]) -> CodeStructure:
        """Parse code lines to find all import statements, function definitions, and class definitions."""
        import_lines: List[str] = []
        function_definitions: Dict[str, Tuple[int, int]] = {}
        class_definitions: Dict[str, Tuple[int, int]] = {}

        i = 0
        while i < len(lines):
            stripped_line = lines[i].strip()

            # Collect import lines
            if stripped_line.startswith(('import ', 'from ')):
                import_lines.append(lines[i])
                i += 1
                continue

            # Collect Classes
            if stripped_line.startswith('class '):
                class_name = stripped_line.split(
                    'class ')[1].split('(')[0].strip()
                start_idx = i

                # Find end of this class
                end_idx = self._find_function_end(lines, i)
                class_definitions[class_name] = (start_idx, end_idx)
                i = end_idx + 1
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

        return {
            'import_lines': import_lines,
            'function_definitions': function_definitions,
            'class_definitions': class_definitions
        }

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
        """Extract target function body with helper functions and imports. --> For HumanEval style problems"""
        lines = code_string.split('\n')
        result = self._parse_code_structure(lines)
        import_lines = result['import_lines']
        function_definitions = result['function_definitions']

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

    def extract_class_block(self, code_string: str) -> Optional[str]:
        """Extract the first class block from the code string."""
        lines = code_string.split('\n')
        result = self._parse_code_structure(lines)
        class_definitions = result['class_definitions']

        if not class_definitions:
            return None

        # Get the first class defined
        first_class_name = next(iter(class_definitions))
        class_start, class_end = class_definitions[first_class_name]
        class_lines = lines[class_start:class_end + 1]

        return '\n'.join(class_lines)

    def build_test_script(self, programmer_code: str, tester_code: str) -> str:
        """Combine programmer and tester code into a single test script."""
        script_lines: List[str] = []
        programmer_code_structure = self._parse_code_structure(
            programmer_code.split('\n'))
        tester_code_structure = self._parse_code_structure(
            tester_code.split('\n'))

        # Handle Imports
        programmer_imports = programmer_code_structure['import_lines']
        tester_imports = tester_code_structure['import_lines']

        # Combine and deduplicate imports
        all_imports = list(dict.fromkeys(programmer_imports + tester_imports))
        if all_imports:
            script_lines.extend(all_imports)
            script_lines.append('')  # Add a blank line after imports

        # Add programmer code
        script_lines.append('# Programmer Code')

        # handling functions and helper function if any
        programmer_functions = programmer_code_structure['function_definitions']

        for func_name, (start, end) in programmer_functions.items():
            func_lines = programmer_code.split('\n')[start:end + 1]
            script_lines.extend(func_lines)
            script_lines.append('')  # Blank line after each function

        # Add tester code
        script_lines.append('# Tester Code')

        # Adding the tester classes if any
        tester_classes = tester_code_structure['class_definitions']

        for class_name, (start, end) in tester_classes.items():
            class_lines = tester_code.split('\n')[start:end + 1]
            script_lines.extend(class_lines)
            script_lines.append('')  # Blank line after each class

        # Add if __name__ == "__main__" block by default
        script_lines.append('if __name__ == "__main__":')
        script_lines.append('    unittest.main(verbosity=2)')
        script_lines.append('')  # Blank line at the end

        return '\n'.join(script_lines)
