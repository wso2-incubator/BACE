import datetime

import pandas as pd
import typer
from loguru import logger

from coevolution.logging_utils import parse_complete_coevolution_log


def parse_problem_ids(value: str) -> list[str]:
    """Parse comma-separated problem IDs."""
    if not value:
        return []
    return [id.strip() for id in value.split(",")]


def get_indices_of_all_passing(matrix: pd.DataFrame) -> list[int]:
    """
    Returns the indices of individuals that pass all test cases.

    Args:
        matrix: DataFrame where rows are individuals and columns are test cases.
                Each cell contains a boolean indicating pass (True) or fail (False).

    Returns:
        Indices of individuals that passed all test cases.
    """
    passing_mask = matrix.all(axis=1)
    passing_indices = passing_mask[passing_mask].index.tolist()
    return passing_indices


def main(
    run_id: str = typer.Argument(..., help="The run ID to parse logs for"),
    problem_ids: str = typer.Option(
        ",".join(
            [
                "abc395_f",
                "abc395_e",
                "abc396_g",
                "abc396_f",
                "abc396_e",
                "abc397_d",
                "abc397_f",
                "abc397_g",
                "abc397_e",
                "abc398_f",
            ]
        ),
        help="Comma-separated list of problem IDs to parse. Defaults to the specified list.",
        callback=parse_problem_ids,
    ),
    date: str = typer.Option(
        datetime.date.today().isoformat(), help="Date in YYYY-MM-DD format"
    ),
) -> None:
    logger.remove()

    log_dir = "logs"
    date_str = date.replace("-", "")
    log_filename_pattern = f"coevolution_run_{date_str}.*log*"

    num_passed = 0

    for problem_id in problem_ids:
        logger.debug(f"Parsing logs for run_id={run_id}, problem_id={problem_id}")
        data = parse_complete_coevolution_log(
            log_dir, log_filename_pattern, run_id, problem_id
        )
        logger.debug(f"Parsed data keys: {list(data.keys())}")

        try:
            num_initial_passing = len(
                get_indices_of_all_passing(data["matrices"]["private"][0])
            )
            final_ids = get_indices_of_all_passing(data["matrices"]["private"][-1])
            num_final_passing = len(final_ids)
            print(
                f"{problem_id}: INITIAL:{num_initial_passing}/{len(data['matrices']['private'][0])}\tFINAL:{num_final_passing}/{len(data['matrices']['private'][-1])}",
                end="\t",
            )

            survived_code_df = data["individuals"][
                (data["individuals"]["type"] == "CodeIndividual")
                & (data["individuals"]["status"] == "SURVIVED")
            ]

            # inds with highest probability among the final individuals
            final_probs = survived_code_df["probability"]
            max_prob = final_probs.max()
            top_inds = survived_code_df[final_probs == max_prob]["id"].tolist()

            # if top_inds intersect final_ids
            intersection = set(top_inds).intersection(set(final_ids))
            if intersection:
                print("SUCCESS: Coevolution found a Solution")
                num_passed += 1
            else:
                print("FAILURE: Coevolution did not find a solution")

        except (KeyError, IndexError) as e:
            logger.error(f"Missing expected data key: {e}")
            print(
                f"Problem {problem_id}: Data incomplete, cannot determine success/failure"
            )

    print(f"\nSummary: {num_passed}/{len(problem_ids)} problems solved successfully.")


if __name__ == "__main__":
    typer.run(main)
