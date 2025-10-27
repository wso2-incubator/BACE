import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from datasets import load_dataset
from loguru import logger


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
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self) -> None:
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self) -> None:
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
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
        **kwargs,
    ) -> dict:
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
    diffulty: Difficulty | None = None,
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

    dataset = [CodeGenerationProblem(**p) for p in dataset]

    # Filter only problems with difficulty hard. TODO: Make this configurable.
    if diffulty is not None:
        dataset = [e for e in dataset if e.difficulty == diffulty]
        logger.info("Filtered problems by difficulty: %s", diffulty.value)

    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    logger.info("Loaded %d problems", len(dataset))
    return dataset


def load_code_generation_dataset_not_fast(
    release_version: str = "release_v5",
) -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    logger.info("Loaded %d problems", len(dataset))
    return dataset


if __name__ == "__main__":
    # When run as a script, use the project's loguru logger configuration (if any).
    logger.info("Loading code generation dataset (script entry)")
    dataset = load_code_generation_dataset()
