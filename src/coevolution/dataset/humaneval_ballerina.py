import json
from pathlib import Path
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
        Load HumanEval Ballerina problems based on the provided configuration.

        Args:
            config: Configuration dict with keys:
                - data_dir: Base directory where the dataset file is located (default: "data")
        Returns:
            List of HumanEvalBallerinaProblem instances.
        """

        logger.info("Loading HumanEval Ballerina dataset (placeholder implementation)")

        # Load from data/humaneval_ballerina.jsonl
        # Extract base path from config, defaulting to a sensible fallback or failing
        base_data_dir = Path(config.get("data_dir", "data"))
        dataset_path = base_data_dir / "humaneval_ballerina.jsonl"
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
