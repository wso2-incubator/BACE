from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from LLMClient import LLMClient


def extract_code_block(response: str) -> str:
    print("Full response:", response)
    if "```" in response:
        parts = response.split("```")
        if len(parts) >= 3:
            code_block = parts[1]
            if code_block.startswith("python"):
                code_block = code_block[len("python"):].strip()
            return code_block.strip()
    return response.strip()


def generate_one_completion(prompt: str, client: LLMClient) -> str:
    response = client.generate(prompt)
    code = extract_code_block(response)
    return code


if __name__ == "__main__":
    # initialize the LLM Client
    llm_provider = "ollama"
    llm_model = "qwen2.5-coder:7b"
    client = LLMClient(llm_provider, model=llm_model)

    num_samples_per_task = 1
    problems = read_problems()
    for task_id in tqdm(problems, desc="Problems", unit="problem"):
        for i in range(num_samples_per_task):
            # print(f"Generating code for task: {task_id}, sample: {i}")
            completion = dict(task_id=task_id, completion=generate_one_completion(
                problems[task_id]["prompt"], client))

            write_jsonl(f"{llm_model}.jsonl", [completion], append=True)
            break
        break
