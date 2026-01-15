#!/usr/bin/env python3
"""
HumanEval Subset Evaluation Script

This script evaluates the functional correctness of generated code completions
against HumanEval datasets, with a default focus on the 20-problem subset for
faster development iterations.

Features:
- Evaluates generated samples using pass@k metrics
- Defaults to HumanEval_20.jsonl.gz subset for quick testing
- Configurable evaluation parameters (k values, workers, timeout)
- Fire-based CLI for easy command-line usage
- Automatic path resolution for APR project structure

Usage Examples:
    # Evaluate a sample file against HumanEval_20 subset
    python evaluate_humaneval_subset.py my_samples.jsonl

    # Evaluate with custom k values
    python evaluate_humaneval_subset.py my_samples.jsonl --k="1,5,10"
    
    # Evaluate against full HumanEval dataset
    python evaluate_humaneval_subset.py my_samples.jsonl --problem_file="path/to/HumanEval.jsonl.gz"
    
    # Evaluate with more workers and custom timeout
    python evaluate_humaneval_subset.py my_samples.jsonl --n_workers=8 --timeout=5.0

Input:
    - sample_file: Path to generated samples (.jsonl format)
    - k: Comma-separated k values for pass@k evaluation (default: "1,10,100")
    - n_workers: Number of parallel workers (default: 4)
    - timeout: Code execution timeout in seconds (default: 3.0)
    - problem_file: Path to problem dataset (default: HumanEval_20.jsonl.gz)

Output:
    - Prints pass@k evaluation results to console
    - Saves detailed results to {sample_file}_results.jsonl.gz

Dependencies:
    - fire: Command-line interface
    - human_eval: HumanEval evaluation framework
    
Author: Kaushitha Silva
Created: 2025
"""

import fire
from pathlib import Path
from typing import List

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

# Define output path - save in APR project's data directory
project_root = Path(__file__).parent.parent.parent
dataset_dir = project_root / "data" / "human_eval"
generations_dir = dataset_dir / "generations"
HUMAN_EVAL_20 = dataset_dir / "HumanEval_20.jsonl.gz"


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = str(HUMAN_EVAL_20),
) -> None:
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k_list: List[int] = list(map(int, k.split(",")))
    input_file = generations_dir / sample_file
    results = evaluate_functional_correctness(
        str(input_file), k_list, n_workers, timeout, problem_file)
    print(results)


def main() -> None:
    fire.Fire(entry_point)


if __name__ == "__main__":
    main()
