"""
Example script to generate code completions for HumanEval problems using an LLM.
This script reads problems from the HumanEval dataset, generates code completions using
a specified LLM, processes the responses to extract clean function implementations,
and saves the results to a JSONL file.
"""

from common.code_processor import CodeProcessor
from common.llm_client import LLMClient
from human_eval.data import write_jsonl, read_problems
import sys
import os
from typing import Dict, Any, List
from tqdm import tqdm


def generate_one_completion(prompt: str, client: LLMClient, code_processor: CodeProcessor) -> str:
    # Extract the target function name from the prompt
    target_function_name: str = code_processor.extract_function_name_from_problem(
        prompt)
    if not target_function_name:
        raise ValueError("No function definition found in the prompt.")

    prompt = prompt + "\n# Complete the function above, do not use any libraries. You are only allowed to write code inside the function defined above. Let's think step-by-step and then write the final code.\n"
    response: str = client.generate(prompt)
    print("Raw Response:\n", response)  # Debug print

    code = code_processor.extract_code_block_from_response(response)

    # Extract the function block from def to return
    code_without_comments: str = code_processor.remove_comments(code)
    completion_code: str = code_processor.extract_function_with_helpers(
        code_without_comments, target_function_name)

    print("Generated Completion Code:\n", completion_code)  # Debug print
    return completion_code


if __name__ == "__main__":
    # initialize the LLM Client
    llm_provider = "ollama"
    llm_model = "qwen2.5-coder:7b"

    client = LLMClient(llm_provider, model=llm_model)
    code_processor = CodeProcessor()

    num_samples_per_task: int = 1
    problems: Dict[str, Dict[str, Any]] = read_problems()
    for task_id in tqdm(problems, desc="Problems", unit="problem"):
        for i in range(num_samples_per_task):
            # print(f"Generating code for task: {task_id}, sample: {i}")
            completion = dict(task_id=task_id, completion=generate_one_completion(
                problems[task_id]["prompt"], client, code_processor))

            write_jsonl(f"{llm_model}.jsonl", [completion], append=True)
