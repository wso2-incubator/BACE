import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

# Reuse existing analysis utils
from coevolution.analysis.log_parser import get_problem_ids, parse_coevolution_log

app = typer.Typer()


def get_clean_length(code: str) -> int:
    """
    Calculates the length of code excluding comments, docstrings, and whitespace.
    Used as a tie-breaker for solutions with equal probability.
    """
    if not isinstance(code, str) or not code.strip():
        return float("inf")  # Penalize empty/invalid code

    cleaned_code = code

    # 1. Remove Docstrings using AST
    try:
        parsed = ast.parse(cleaned_code)
        # Iterate and remove docstrings from function/class/module levels
        for node in ast.walk(parsed):
            if not isinstance(
                node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)
            ):
                continue
            if not (node.body and isinstance(node.body[0], ast.Expr)):
                continue
            if not isinstance(node.body[0].value, ast.Constant):
                continue
            if isinstance(node.body[0].value.value, str):
                pass
    except SyntaxError:
        # If code is not valid python (possible in evolution), fall back to raw length
        return len(code)

    # 2. Regex approach (Robust enough for scoring)
    # Remove docstrings (triple-quotes)
    cleaned_code = re.sub(r'""".*?"""', "", cleaned_code, flags=re.DOTALL)
    cleaned_code = re.sub(r"'''.*?'''", "", cleaned_code, flags=re.DOTALL)

    # Remove comments (# ...)
    cleaned_code = re.sub(r"#.*", "", cleaned_code)

    # Remove empty lines and whitespace
    cleaned_code = "".join(cleaned_code.split())

    return len(cleaned_code)


@app.command()
def main(
    run_id: str = typer.Option(..., help="The Run ID to extract champions from"),
    log_dir: str = typer.Option("logs", help="Directory containing log files"),
    file_pattern: str = typer.Option("*.log", help="Pattern to match log files"),
) -> None:
    """
    Extracts the champion code snippet for each problem in a run.

    Output:
        Saves to 'results/{run_id}.jsonl'

    Tie-breaking logic:
        1. Highest Probability
        2. Shortest 'Clean' Code Length (ignoring comments/docstrings)
    """
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

    # 1. Setup Output Path
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{run_id}.jsonl"

    # 2. Discovery
    logger.info(f"Scanning {log_dir} for Run ID: {run_id}...")
    problem_ids = get_problem_ids(log_dir, file_pattern, run_id)

    if not problem_ids:
        logger.error(f"No problems found for Run ID: {run_id}")
        return

    if "SETUP" in problem_ids:
        problem_ids.remove("SETUP")

    logger.info(f"Found {len(problem_ids)} problems. Starting extraction...")

    extracted_count = 0

    # 3. Processing
    with open(output_path, "w") as f_out:
        for pid in tqdm(sorted(list(problem_ids)), desc="Processing Problems"):
            # Parse Log
            parsed_data = parse_coevolution_log(
                log_dir=log_dir,
                log_filename_pattern=file_pattern,
                target_run_id=run_id,
                target_problem_id=pid,
            )

            individuals_df = parsed_data.get("individuals", pd.DataFrame())

            if individuals_df.empty:
                continue

            # Filter Survivors
            survivors = individuals_df[
                (individuals_df["type"] == "code")
                & (individuals_df["status"].str.upper() == "SURVIVED")
            ]

            if survivors.empty:
                continue

            # Find Champion(s) with Max Probability
            max_prob = survivors["probability"].max()
            best_candidates = survivors[survivors["probability"] == max_prob].copy()

            # Tie-Breaker: Clean Code Length
            if len(best_candidates) > 1:
                best_candidates["clean_len"] = best_candidates["snippet"].apply(
                    get_clean_length
                )
                # Sort: shortest clean_len first
                champion = best_candidates.sort_values(
                    "clean_len", ascending=True
                ).iloc[0]
            else:
                champion = best_candidates.iloc[0]

            champion_snippet = champion["snippet"]

            if not isinstance(champion_snippet, str) or not champion_snippet.strip():
                continue

            # Save
            record = {
                "question_id": pid,
                "task_id": pid,
                "language": "Python3",
                "source_codes": [champion_snippet],
                "is_solved": None,
            }

            f_out.write(json.dumps(record) + "\n")
            extracted_count += 1

    logger.success(
        f"Extraction complete. {extracted_count} champions saved to: {output_path.absolute()}"
    )


if __name__ == "__main__":
    app()
