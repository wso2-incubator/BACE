from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from LLMClient import LLMClient

import re


def extract_function_block(code_string: str) -> str:
    """
    Extracts the function block from 'def' line to the 'return' line (inclusive).
    """
    lines = code_string.split('\n')
    start_idx = None
    end_idx = None

    # Find the line starting with 'def'
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            start_idx = i + 1  # Start from the next line from 'def'

            break

    if start_idx is None:
        return ""

    # Find the return line (with single tab or equivalent spaces indentation)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        # Check for return with single level indentation (tab or 4 spaces)
        if (line.startswith('\treturn') or
                (line.startswith('    return') and not line.startswith('        '))):
            end_idx = i
            break

    if end_idx is None:
        return ""

    # Extract the function block
    function_lines = lines[start_idx:end_idx + 1]
    return '\n'.join(function_lines)


def remove_comments(code):
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
    lines = code.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.rstrip()  # Remove trailing whitespace
        # Keep line if not empty or previous line wasn't empty
        if stripped or (cleaned_lines and cleaned_lines[-1].strip()):
            cleaned_lines.append(stripped)

    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return '\n'.join(cleaned_lines)


def generate_one_completion(prompt: str, client: LLMClient) -> str:
    response: str = client.generate(prompt)
    pattern = re.compile(r'```(?:[Pp]ython|[Pp]y)\s*([\s\S]+?)\s*```')

    # some responses may not be in a code block
    match = pattern.search(response)
    if match:
        code = match.group(1)
    else:
        code = response

    # Extract the function block from def to return
    completion_code: str = remove_comments(extract_function_block(code))
    print("Generated Completion Code:\n", completion_code)  # Debug print
    return completion_code


if __name__ == "__main__":
    # initialize the LLM Client
    llm_provider = "ollama"
    llm_model = "qwen2.5-coder:7b"
    client = LLMClient(llm_provider, model=llm_model)

    num_samples_per_task: int = 1
    problems: list[str] = read_problems()
    for task_id in tqdm(problems, desc="Problems", unit="problem"):
        for i in range(num_samples_per_task):
            # print(f"Generating code for task: {task_id}, sample: {i}")
            completion = dict(task_id=task_id, completion=generate_one_completion(
                problems[task_id]["prompt"], client))

            write_jsonl(f"{llm_model}.jsonl", [completion], append=True)
