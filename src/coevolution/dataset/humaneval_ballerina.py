import json
from typing import Any

from loguru import logger

from ..core.interfaces import Problem, Test
from .base import DatasetAdapter


class HumanEvalBallerinaAdapter(DatasetAdapter):
    """
    HumanEval Ballerina dataset adapter implementation.

    Loads code generation problems from the HumanEval Ballerina dataset,
    """

    def load_dataset(self, config: dict[str, Any]) -> list[Problem]:
        """
        Load LCB problems based on the provided configuration.

        Args:
            config: Configuration dict with keys:
                - version (str): Dataset release version (default: "release_v6")
                - difficulty (str|None): Filter by difficulty ("easy", "medium", "hard")
                - start_date (str|None): Filter start date (YYYY-MM-DD format)
                - end_date (str|None): Filter end date (YYYY-MM-DD format)

        Returns:
            List of LCBCodeGenerationProblem instances.
        """

        logger.info("Loading HumanEval Ballerina dataset (placeholder implementation)")

        # Load from data/humaneval_ballerina.jsonl
        dataset_path = "data/humaneval_ballerina.jsonl"
        problems = []
        with open(dataset_path, "r") as f:
            for line in f:
                data = json.loads(line)
                problem = Problem(
                    question_title=data["question_title"],
                    question_id=data["question_id"],
                    question_content=data["question_content"],
                    starter_code=data["starter_code"],
                    public_test_cases=[
                        Test(input=test["input"], output=test["output"])
                        for test in data.get("public_test_cases", [])
                    ],
                    private_test_cases=[
                        Test(input=test["input"], output=test["output"])
                        for test in data.get("private_test_cases", [])
                    ],
                )
                problems.append(problem)

        logger.info(f"Loaded {len(problems)} problems from HumanEval Ballerina dataset")
        return problems
