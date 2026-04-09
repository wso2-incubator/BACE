import json
import os
from typing import List, Optional

import typer


def format_results(
    root_path: str,
    runs: Optional[List[int]] = typer.Argument(
        default=[1, 2, 3], help="The run numbers to process (e.g., 1 2 3)"
    ),
) -> None:
    """
    Formats CodeSim Results.jsonl files into Results-for-bace.jsonl.
    
    root_path should be the parent directory containing the Run-X folders.
    """
    for run_num in runs:
        run_dir = f"Run-{run_num}"
        input_file = os.path.join(root_path, run_dir, "Results.jsonl")
        output_file = os.path.join(root_path, run_dir, "Results-for-bace.jsonl")

        if not os.path.exists(input_file):
            print(f"Skipping {run_dir}: {input_file} not found.")
            continue

        print(f"Converting {input_file} to {output_file}...")

        with (
            open(input_file, "r", encoding="utf-8") as f_in,
            open(output_file, "w", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                if not line.strip():
                    continue
                data = json.loads(line)

                # Extract first source code snippet
                snippet = data.get("source_codes", [None])[0]

                # Format new object
                formatted_data = {
                    "question_id": data.get("question_id"),
                    "snippet": snippet,
                    "status": None,  # Set to None regardless of is_solved
                }

                f_out.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")

        print(f"Done with {run_dir}.")


if __name__ == "__main__":
    typer.run(format_results)
