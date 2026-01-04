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

from pathlib import Path
from typing import Any, Optional

import fire
from human_eval.data import read_problems, write_jsonl
from tqdm import tqdm

from infrastructure import CodeProcessor, SimpleConfig
from infrastructure.llm_client import create_llm_client


class SimpleGenerator:
    """Simple but modular code generator."""

    def __init__(self, config: SimpleConfig) -> None:
        self.config = config
        self.client = create_llm_client(config.llm_provider, model=config.llm_model)
        self.code_processor = CodeProcessor()

    def generate_completion(self, problem_prompt: str) -> str:
        """Generate a single completion for the given problem."""
        # Extract function name
        target_function_name = self.code_processor.extract_function_name_from_problem(
            problem_prompt
        )
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

    def load_problems(self) -> Any:
        """Load problems from dataset (full or subset)."""
        dataset_path = self.config.get_dataset_path()
        if dataset_path:
            return read_problems(dataset_path)
        else:
            return read_problems()

    def run_generation(self) -> None:
        """Generate completions for the configured dataset."""
        problems = self.load_problems()
        output_path = self.config.get_output_path(
            "simple",
            f"-{self.config.filename_suffix}" if self.config.filename_suffix else "",
        )

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
                        problems[task_id]["prompt"]
                    )

                    completion = {"task_id": task_id, "completion": completion_code}

                    write_jsonl(output_path, [completion], append=True)

                except Exception as e:
                    print(f"Error generating completion for {task_id}: {e}")
                    continue

        print(f"Generation completed! Results saved to: {output_path}")


def generate(
    llm_model: str = "qwen2.5-coder:7b",
    llm_provider: str = "ollama",
    num_samples_per_task: int = 1,
    use_subset: bool = True,
    subset_path: Optional[str] = None,
    prompt_template: str = "\n# Complete the function above, do not use any libraries. You are only allowed to write code inside the function defined above. Let's think step-by-step and then write the final code.\n",
    output_filename: Optional[str] = None,
    output_dir: str = "data/human_eval/generations",
    filename_suffix: str = "",
    verbose: bool = False,
) -> None:
    """
    Generate code completions for HumanEval problems.

    Args:
        llm_model: The LLM model to use (e.g., "qwen2.5-coder:7b", "gpt-4")
        llm_provider: The LLM provider ("ollama" or "openai")
        num_samples_per_task: Number of samples to generate per problem
        use_subset: Whether to use HumanEval subset (default: True for faster testing)
        subset_path: Path to custom subset file (if None, uses default HumanEval_20)
        prompt_template: Template to append to each problem prompt
        output_filename: Custom output filename (if None, auto-generates)
        output_dir: Output directory for generated files
        filename_suffix: Additional suffix for auto-generated filenames
        verbose: Enable verbose output

    Examples:
        # Basic usage with defaults
        python simple_generator.py generate

        # Use OpenAI GPT-4
        python simple_generator.py generate --llm_model=gpt-4 --llm_provider=openai

        # Generate multiple samples with custom prompt
        python simple_generator.py generate --num_samples_per_task=3 --prompt_template="Think carefully then code:"

        # Use full dataset instead of subset
        python simple_generator.py generate --use_subset=False

        # Custom output file
        python simple_generator.py generate --output_filename="my_results.jsonl"
    """

    config = SimpleConfig(
        llm_model=llm_model,
        llm_provider=llm_provider,
        num_samples_per_task=num_samples_per_task,
        use_humaneval_subset=use_subset,
        humaneval_subset_path=subset_path,
        prompt_template=prompt_template,
        output_filename=output_filename,
        output_dir=output_dir,
        filename_suffix=filename_suffix,
        verbose=verbose,
    )

    print("🚀 Starting Simple Generator")
    print(f"   Model: {config.llm_model} ({config.llm_provider})")
    print(f"   Dataset: {'Subset' if config.use_humaneval_subset else 'Full'}")
    print(f"   Samples per task: {config.num_samples_per_task}")
    print(f"   Verbose: {config.verbose}")
    print()

    generator = SimpleGenerator(config)
    generator.run_generation()


if __name__ == "__main__":
    fire.Fire(
        {
            "generate": generate,
        }
    )
