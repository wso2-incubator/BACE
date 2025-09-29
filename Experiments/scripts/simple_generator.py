#!/usr/bin/env python3
"""
Simple Generator for HumanEval Problems

A modular but straightforward single-agent code generator with configurable prompts,
output files, and dataset selection (full or subset).

Features:
- Easy prompt customization
- Flexible output file naming
- Support for HumanEval subset
- Clean configuration
- Simple but professional structure
"""

from human_eval.data import write_jsonl, read_problems
from common import LLMClient, CodeProcessor, SimpleConfig

from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm


class SimpleGenerator:
    """Simple but modular code generator."""

    def __init__(self, config: SimpleConfig) -> None:
        self.config = config
        self.client = LLMClient(config.llm_provider, model=config.llm_model)
        self.code_processor = CodeProcessor()

    def generate_completion(self, problem_prompt: str) -> str:
        """Generate a single completion for the given problem."""
        # Extract function name
        target_function_name = self.code_processor.extract_function_name_from_problem(
            problem_prompt)
        if not target_function_name:
            raise ValueError("No function definition found in the prompt.")

        # Generate enhanced prompt using configurable template
        enhanced_prompt = problem_prompt + self.config.prompt_template
        response = self.client.generate(enhanced_prompt)

        if self.config.verbose:
            print(f"Raw Response:\n{response}\n")

        # Process response to extract clean code
        code = self.code_processor.extract_code_block_from_response(response)
        code_without_comments = self.code_processor.remove_comments(code)
        completion_code = self.code_processor.extract_function_with_helpers(
            code_without_comments, target_function_name
        )

        if self.config.verbose:
            print(f"Generated Completion:\n{completion_code}\n")

        return completion_code

    def load_problems(self) -> Dict[str, Any]:
        """Load problems from dataset (full or subset)."""
        dataset_path = self.config.get_dataset_path()
        if dataset_path:
            return read_problems(dataset_path)
        else:
            return read_problems()

    def run_generation(self) -> None:
        """Generate completions for the configured dataset."""
        problems = self.load_problems()
        output_path = self.config.get_output_path("simple",
                                                  f"-{self.config.filename_suffix}" if self.config.filename_suffix else "")

        print(f"Starting generation with {len(problems)} problems")
        print(f"Output file: {output_path}")
        print(f"Model: {self.config.llm_model}")
        print(f"Samples per task: {self.config.num_samples_per_task}")

        # Clear output file if it exists
        if Path(output_path).exists():
            Path(output_path).unlink()

        for task_id in tqdm(problems, desc="Generating completions", unit="problem"):
            for sample_idx in range(self.config.num_samples_per_task):
                try:
                    completion_code = self.generate_completion(
                        problems[task_id]["prompt"])

                    completion = {
                        "task_id": task_id,
                        "completion": completion_code
                    }

                    write_jsonl(output_path, [completion], append=True)

                except Exception as e:
                    print(f"Error generating completion for {task_id}: {e}")
                    continue

        print(f"Generation completed! Results saved to: {output_path}")


def main() -> None:
    """Main execution function with example configurations."""

    # Example 1: Basic configuration with default prompt
    config = SimpleConfig(
        llm_model="gpt-5",
        llm_provider="openai",
        num_samples_per_task=1,
        use_humaneval_subset=True,  # Use HumanEval_20 for faster testing
        verbose=True,
        prompt_template="",
        filename_suffix="no-prompt"
    )

    generator = SimpleGenerator(config)
    generator.run_generation()


if __name__ == "__main__":
    main()
