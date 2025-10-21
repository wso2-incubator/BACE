"""Parse and extract code from LLM responses and problem prompts."""

import re
from typing import List


def extract_function_name_from_problem(problem_prompt: str) -> str:
    """
    Extract the function name from the problem prompt.

    Args:
        problem_prompt: Problem description containing function definition

    Returns:
        Function name if found, empty string otherwise

    Example:
        >>> prompt = "def calculate(x, y):\\n    pass"
        >>> extract_function_name_from_problem(prompt)
        'calculate'
    """
    lines = problem_prompt.split("\n")
    for line in lines:
        if line.strip().startswith("def "):
            func_name = line.strip().split("def ")[1].split("(")[0]
            return func_name
    return ""


def extract_all_code_blocks_from_response(response: str) -> List[str]:
    """
    Extract all Python code blocks (```python ... ```) from the LLM response.

    Args:
        response: LLM response text potentially containing markdown code blocks

    Returns:
        List of code strings from all ```python blocks found

    Example:
        >>> response = "Here's code:\\n```python\\nprint('hi')\\n```"
        >>> extract_all_code_blocks(response)
        ["print('hi')"]
    """
    pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")
    matches = pattern.findall(response)
    return matches if matches else []


def extract_code_block_from_response(response: str) -> str:
    """
    Extract the first Python code block (```python ... ```) from the LLM response.
    If no Python code block is found, returns the original response.

    Args:
        response: LLM response text potentially containing markdown code blocks

    Returns:
        Code string from first ```python block if found, otherwise original response

    Example:
        >>> response = "Solution:\\n```python\\ndef foo(): pass\\n```"
        >>> extract_code_block_from_response(response)
        'def foo(): pass'
    """
    pattern = re.compile(r"```[Pp]ython\s*([\s\S]+?)\s*```")
    match = pattern.search(response)
    if match:
        return match.group(1)
    return response  # Return original response if no code block found
