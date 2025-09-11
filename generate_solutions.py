from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from LLMClient import LLMClient
import re


def extract_function_name_from_problem(problem_prompt: str) -> str:
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


def extract_function_completion_with_helpers_and_imports(code_string: str, target_function_name: str) -> str:
    """
    Extracts the target function's body along with all other functions as helpers.
    Returns the combined code block suitable for HumanEval.
    """
    lines = code_string.split('\n')

    # Collect all imports
    import_lines = []
    function_definitions = {}  # {function_name: (start_line, end_line)}

    # First pass: identify all functions and imports
    i = 0
    while i < len(lines):
        stripped_line = lines[i].strip()

        # Collect import lines
        if stripped_line.startswith('import ') or stripped_line.startswith('from '):
            import_lines.append(lines[i])
            i += 1
            continue

        # Find function definitions
        if stripped_line.startswith('def '):
            func_name = stripped_line.split('def ')[1].split('(')[0].strip()
            start_idx = i

            # Find end of this function
            end_idx = None
            for j in range(i + 1, len(lines)):
                line = lines[j]
                # Function ends when we hit unindented non-empty line or another def
                if (line.strip() and
                    not line.startswith('\t') and
                        not line.startswith('    ')):
                    end_idx = j - 1
                    break

            if end_idx is None:
                end_idx = len(lines) - 1

            # Remove trailing empty lines
            while end_idx > start_idx and not lines[end_idx].strip():
                end_idx -= 1

            function_definitions[func_name] = (start_idx, end_idx)
            i = end_idx + 1
        else:
            i += 1

    # Find target function
    if target_function_name not in function_definitions:
        return ""

    target_start, target_end = function_definitions[target_function_name]

    # Extract target function body (excluding def line)
    target_body_lines = lines[target_start + 1:target_end + 1]

    # Get all helper functions (all functions except the target)
    helper_functions = [
        name for name in function_definitions.keys() if name != target_function_name]

    # Build the complete code block
    result_lines = []

    # Add indented imports if any
    if import_lines:
        indented_imports = ['    ' + line.strip() for line in import_lines]
        result_lines.extend(indented_imports)
        if indented_imports:
            result_lines.append('')  # Empty line after imports

    # Add helper functions (with proper indentation)
    for helper_name in helper_functions:
        helper_start, helper_end = function_definitions[helper_name]
        helper_lines = lines[helper_start:helper_end + 1]
        # Indent helper function
        indented_helper = ['    ' + line for line in helper_lines]
        result_lines.extend(indented_helper)
        result_lines.append('')  # Empty line after helper function

    # Add target function body
    result_lines.extend(target_body_lines)

    # Clean up trailing empty lines
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    return '\n'.join(result_lines)


def remove_comments(code: str) -> str:
    """
    Remove single line (#) and multiline (''' or \"\"\") comments from Python code.
    This is optional for the HumanEval format but helps in cleaner evaluation.
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


def generate_one_completion(prompt: str, client: LLMClient) -> str:
    # Extract the target function name from the prompt
    target_function_name: str = extract_function_name_from_problem(prompt)
    if not target_function_name:
        raise ValueError("No function definition found in the prompt.")

    prompt = prompt + "\n# Complete the function above, do not use any libraries. You are only allowed to write code inside the function defined above. Let's think step-by-step and then write the final code.\n"
    response: str = client.generate(prompt)
    print("Raw Response:\n", response)  # Debug print

    pattern = re.compile(r'```(?:[Pp]ython|[Pp]y)\s*([\s\S]+?)\s*```')

    # some responses may not be in a code block
    match = pattern.search(response)
    if match:
        code = match.group(1)
    else:
        code = response

    # Extract the function block from def to return
    completion_code: str = remove_comments(
        extract_function_completion_with_helpers_and_imports(code, target_function_name))
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
