#!/usr/bin/env python3
"""
Script to generate HumanEval_20.jsonl.gz - a random subset of 20 HumanEval problems.
"""

from human_eval.data import read_problems, write_jsonl
import random
import sys
import os
from pathlib import Path
from typing import List


def generate_humaneval_subset(num_problems: int = 20, seed: int = 42) -> List[str]:
    """
    Generate a random subset of HumanEval problems.

    Args:
        num_problems: Number of problems to sample (default: 20)
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Read all problems
    print("Reading HumanEval problems...")
    all_problems = read_problems()

    print(f"Total problems available: {len(all_problems)}")

    if num_problems > len(all_problems):
        raise ValueError(
            f"Requested {num_problems} problems, but only {len(all_problems)} available")

    # Get task IDs and randomly sample
    task_ids = list(all_problems.keys())
    sampled_task_ids = random.sample(task_ids, num_problems)

    # Sort for consistent ordering
    sampled_task_ids.sort()

    print(f"Sampled {len(sampled_task_ids)} problems:")
    for task_id in sampled_task_ids:
        print(f"  - {task_id}")

    # Create subset data
    subset_data = [all_problems[task_id] for task_id in sampled_task_ids]

    # Define output path - save in APR project's data directory
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "human_eval"
    output_file = output_dir / "HumanEval_20.jsonl.gz"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the subset
    print(f"\nWriting subset to: {output_file}")
    write_jsonl(str(output_file), subset_data)

    print(
        f"✅ Successfully generated HumanEval_20.jsonl.gz with {len(subset_data)} problems")

    return sampled_task_ids


def main() -> None:
    """Main entry point."""
    try:
        sampled_ids = generate_humaneval_subset()

        print(f"\n🎯 Summary:")
        print(f"   • Generated: HumanEval_20.jsonl.gz")
        print(f"   • Problems: {len(sampled_ids)}")
        print(f"   • Seed: 42 (reproducible)")
        print(f"   • Location: data/human_eval/")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
