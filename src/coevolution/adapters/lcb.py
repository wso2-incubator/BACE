import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from loguru import logger

from ..core.interfaces import Problem, Test


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
class LCBTest(Test):
    testtype: TestType

    def __post_init__(self) -> None:
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class LCBCodeGenerationProblem(Problem):
    platform: Platform
    contest_id: str
    contest_date: datetime
    difficulty: Difficulty
    metadata: dict[str, Any]

    # Default starter code for problems without one (e.g., STDIN problems)
    DEFAULT_STARTER_CODE = """
class Solution:
    def sol(self, input_str: str) -> str:
        # Input is provided as a single string.
"""

    def __post_init__(self) -> None:
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        if isinstance(self.contest_date, str):
            self.contest_date = datetime.fromisoformat(self.contest_date)

        # Ensure starter_code is never empty
        # LCB problems without starter code (STDIN problems) get a default
        if not self.starter_code or not self.starter_code.strip():
            logger.trace(
                f"Problem '{self.question_title}' has no starter_code, using default"
            )
            self.starter_code = self.DEFAULT_STARTER_CODE

        # Parse public_test_cases
        if isinstance(self.public_test_cases, str):
            self.public_test_cases = json.loads(self.public_test_cases)
        self.public_test_cases = [LCBTest(**t) for t in self.public_test_cases]  # type: ignore[arg-type]

        # Parse private_test_cases (might be JSON or encoded)
        if isinstance(self.private_test_cases, str):
            try:
                self.private_test_cases = json.loads(self.private_test_cases)
            except (json.JSONDecodeError, ValueError):
                # Encoded format: base64 -> zlib -> pickle -> JSON
                self.private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(self.private_test_cases.encode("utf-8"))
                        )
                    )
                )
        self.private_test_cases = [LCBTest(**t) for t in self.private_test_cases]  # type: ignore[arg-type]

        # Parse metadata
        if isinstance(self.metadata, str):
            if self.metadata.strip():
                self.metadata = json.loads(self.metadata)
            else:
                self.metadata = {}


def load_code_generation_dataset(
    release_version: str = "release_v6",
    start_date: str | None = None,
    end_date: str | None = None,
    difficulty: Difficulty | None = None,
) -> list[LCBCodeGenerationProblem]:
    lcb_cache_path = Path(f"data/lcb_{release_version}.jsonl")

    if lcb_cache_path.exists():
        logger.info(f"Loading LCB dataset from cache: {lcb_cache_path}")
        with open(lcb_cache_path, "r") as f:
            dataset_list = [json.loads(line) for line in f]
    else:
        logger.info("Fetching LCB dataset from Hugging Face.")
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            trust_remote_code=True,
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

        dataset_list = [dict(p) for p in dataset]

        # Save to cache
        logger.info(f"Saving LCB dataset to cache: {lcb_cache_path}")
        lcb_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lcb_cache_path, "w") as f:
            for record in dataset_list:

                def json_serializable(obj: Any) -> str:
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return str(obj)

                f.write(json.dumps(record, default=json_serializable) + "\n")

    dataset_items: list[LCBCodeGenerationProblem] = [
        LCBCodeGenerationProblem(**p) for p in dataset_list
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
