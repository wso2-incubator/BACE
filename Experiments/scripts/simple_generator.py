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
from common.llm_client import LLMClient
from common.code_processor import CodeProcessor
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent


@dataclass
class SimpleConfig:
    """Simple configuration for code generation."""
    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5-coder:7b"

    # Generation settings
    num_samples_per_task: int = 1
    use_subset: bool = True
    subset_path: Optional[str] = None

    # Prompt settings
    prompt_template: str = "\n# Complete the function above, do not use any libraries. You are only allowed to write code inside the function defined above. Let's think step-by-step and then write the final code.\n"

    # Output settings
    output_filename: Optional[str] = None
    output_dir: str = "data/generations/simple"
    filename_suffix: str = ""  # Gets appended to auto-generated filenames

    # Debug settings
    verbose: bool = False


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

    def get_output_path(self) -> str:
        """Generate output file path with smart naming."""
        if self.config.output_filename:
            # If custom filename provided, make it relative to project root if it's not absolute
            custom_path = Path(self.config.output_filename)
            if custom_path.is_absolute():
                return str(custom_path)
            else:
                return str(project_root / custom_path)

        # Create descriptive filename
        model_name = self.config.llm_model.replace(":", "_")
        subset_suffix = "_subset20" if self.config.use_subset else ""
        samples_suffix = f"-samples{self.config.num_samples_per_task}"
        user_suffix = f"-{self.config.filename_suffix}" if self.config.filename_suffix else ""

        filename = f"{model_name}{subset_suffix}{samples_suffix}{user_suffix}.jsonl"

        # Create output directory relative to project root
        output_dir = project_root / self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        return str(output_dir / filename)

    def load_problems(self) -> Dict[str, Any]:
        """Load problems from dataset (full or subset)."""
        if self.config.use_subset and self.config.subset_path:
            # Load from custom subset path
            return read_problems(self.config.subset_path)
        elif self.config.use_subset:
            # Load from default subset (HumanEval_20)
            subset_path = project_root / \
                Path("data/human_eval/HumanEval_20.jsonl.gz")
            if subset_path.exists():
                return read_problems(str(subset_path))
            else:
                print(
                    f"⚠️  Subset file not found at {subset_path}, using full dataset")
                return read_problems()
        else:
            # Load full dataset
            return read_problems()

    def run_generation(self) -> None:
        """Generate completions for the configured dataset."""
        problems = self.load_problems()
        output_path = self.get_output_path()

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
        llm_model="qwen2.5-coder:7b",
        num_samples_per_task=1,
        use_subset=True,  # Use HumanEval_20 for faster testing
        verbose=True,
        prompt_template="",
        filename_suffix="no-prompt"
    )

    generator = SimpleGenerator(config)
    generator.run_generation()


if __name__ == "__main__":
    main()
