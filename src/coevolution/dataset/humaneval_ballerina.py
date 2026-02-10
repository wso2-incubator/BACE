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

        sample_problem = Problem(
            question_title="hasCloseElements",
            question_id="sample-1",
            question_content="Check if in given array of numbers, are any two numbers closer to each other than given threshold.\n// hasCloseElements([1.0, 2.0, 3.0], 0.5) returns false\n// hasCloseElements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) returns true\n\nfunction hasCloseElements(float[] numbers, float threshold) returns boolean {\n}",
            starter_code="function hasCloseElements(float[] numbers, float threshold) returns boolean {\n}",
            public_test_cases=[
                Test(
                    input="[1.0, 2.0, 3.0], 0.5",
                    output="false",
                ),
                Test(
                    input="[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3",
                    output="true",
                ),
            ],
            private_test_cases=[
                Test(
                    input="[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3",
                    output="true",
                ),
                Test(
                    input="[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05",
                    output="false",
                ),
                Test(
                    input="[1.0, 2.0, 5.9, 4.0, 5.0], 0.95",
                    output="true",
                ),
                Test(
                    input="[1.0, 2.0, 5.9, 4.0, 5.0], 0.8",
                    output="false",
                ),
                Test(
                    input="[1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1",
                    output="true",
                ),
                Test(
                    input="[1.1, 2.2, 3.1, 4.1, 5.1], 1.0",
                    output="true",
                ),
                Test(
                    input="[1.1, 2.2, 3.1, 4.1, 5.1], 0.5",
                    output="false",
                ),
            ],
        )
        return [sample_problem]
