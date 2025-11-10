import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from loguru import logger

from .core.interfaces import Problem as BaseProblem
from .core.interfaces import Test as BaseTest


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test(BaseTest):
    testtype: TestType

    def __post_init__(self) -> None:
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem(BaseProblem):
    platform: Platform
    contest_id: str
    contest_date: datetime
    difficulty: Difficulty
    metadata: dict[str, Any]

    # Default starter code for problems without one (e.g., STDIN problems)
    DEFAULT_STARTER_CODE = """
class Solution:
    def sol(self, input_str: str) -> str:
        pass
"""

    def __post_init__(self) -> None:
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        # contest_date is already a string, convert it
        if isinstance(self.contest_date, str):
            self.contest_date = datetime.fromisoformat(self.contest_date)

        # Ensure starter_code is never empty
        # LCB problems without starter code (STDIN problems) get a default
        if not self.starter_code or not self.starter_code.strip():
            logger.debug(
                f"Problem '{self.question_title}' has no starter_code, using default"
            )
            self.starter_code = self.DEFAULT_STARTER_CODE

        # public_test_cases is a JSON string, parse it
        if isinstance(self.public_test_cases, str):
            self.public_test_cases = json.loads(self.public_test_cases)
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]  # type: ignore[arg-type]

        # private_test_cases might be compressed
        if isinstance(self.private_test_cases, str):
            try:
                self.private_test_cases = json.loads(self.private_test_cases)
            except Exception:
                self.private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(self.private_test_cases.encode("utf-8"))
                        )
                    )
                )
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]  # type: ignore[arg-type]

        # metadata is a JSON string, parse it
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)

    def insert_output(
        self, output_list: list[str], code_list: list[str]
    ) -> dict[str, Any]:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs: Any,
    ) -> dict[str, Any]:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self) -> dict[str, str]:
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_code_generation_dataset(
    release_version: str = "release_v5",
    start_date: str | None = None,
    end_date: str | None = None,
    difficulty: Difficulty | None = None,
) -> list[CodeGenerationProblem]:
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag=release_version,
    )

    # Fix: The dataset might need to access a specific split
    if hasattr(dataset, "keys"):
        # If it's a DatasetDict, get the appropriate split
        if "train" in dataset:
            logger.info("Using 'train' split of the dataset.")
            dataset = dataset["train"]
        elif "test" in dataset:
            logger.info("Using 'test' split of the dataset.")
            dataset = dataset["test"]
        else:
            # Get the first available split
            logger.info("Using the first available split of the dataset.")
            dataset = dataset[list(dataset.keys())[0]]

    dataset_items: list[CodeGenerationProblem] = [
        CodeGenerationProblem(**p) for p in dataset
    ]

    # Filter only problems with difficulty hard. TODO: Make this configurable.
    if difficulty is not None:
        dataset_items = [e for e in dataset_items if e.difficulty == difficulty]
        logger.info(f"Filtered problems by difficulty: {difficulty.value}")

    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset_items = [e for e in dataset_items if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset_items = [e for e in dataset_items if e.contest_date <= p_end_date]

    logger.info(f"Loaded {len(dataset_items)} problems")
    return dataset_items


if __name__ == "__main__":
    # When run as a script, use the project's loguru logger configuration (if any).
    logger.info("Loading code generation dataset (script entry)")
    dataset = load_code_generation_dataset()
