import base64
import json
import pickle
import re
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from loguru import logger

import infrastructure.code_preprocessing as cpp

from ..core.interfaces import (
    IDatasetTestBlockBuilder,
    ITestBlockRebuilder,
    Problem,
    Test,
)


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
    release_version: str = "release_v5",
    start_date: str | None = None,
    end_date: str | None = None,
    difficulty: Difficulty | None = None,
) -> list[LCBCodeGenerationProblem]:
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

    dataset_items: list[LCBCodeGenerationProblem] = [
        LCBCodeGenerationProblem(**p) for p in dataset
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


class LCBDatasetTestBlockBuilder(IDatasetTestBlockBuilder):
    """
    Implementation of IDatasetTestBlockBuilder for LiveCodeBench dataset.

    Builds unittest test class blocks from LCB test cases, handling both
    STDIN and FUNCTIONAL test types.
    """

    @staticmethod
    def _convert_to_pytest_from_stdin_tests(test_cases: list[LCBTest]) -> str:
        """
        Converts a list of STDIN LCBTest objects into pytest standalone test functions.
        Each test is a standalone function that creates its own Solution instance.
        """
        if not test_cases:
            return "# No test cases provided."

        # Start building the output file string
        output_lines = [
            "import pytest",
            "",
        ]

        # Loop through each test object and create a standalone function
        for i, test_obj in enumerate(test_cases, 1):
            function_name = f"test_case_{i}"

            # Get input/output directly from the object attributes
            # Use repr() to create a valid, escaped Python string literal
            input_literal = repr(test_obj.input.rstrip("\n"))
            output_literal = repr(test_obj.output.rstrip("\n"))

            test_function = [
                f"def {function_name}():",
                "    solution = Solution()",
                f"    input_str = {input_literal}",
                f"    expected_output = {output_literal}",
                "    assert solution.sol(input_str) == expected_output",
                "",
            ]
            output_lines.extend(test_function)

        # Join all lines into a single string
        return "\n".join(output_lines)

    @staticmethod
    def _convert_to_pytest_functional_tests(
        test_cases: list[LCBTest], starter_code: str
    ) -> str:
        """
        Converts a list of FUNCTIONAL LCBTest objects into pytest standalone test functions.
        Each test is a standalone function that creates its own Solution instance.
        """
        # 1. Parse the starter code to find class and method names
        class_match = re.search(r"class\s+(\w+):", starter_code)
        # Regex to find the first method defined with 'self'
        method_match = re.search(r"def\s+(\w+)\s*\(\s*self\s*[,)]", starter_code)

        if not class_match or not method_match:
            return "# Error: Could not parse class and method name from starter code."

        class_name = class_match.group(1)
        method_name = method_match.group(1)

        # 2. Build the output file string
        output_lines = [
            "import pytest",
            "from typing import *  # Import common types for function signatures",
            "",
        ]

        # 3. Loop through each test object and create a standalone function
        for i, test_obj in enumerate(test_cases, 1):
            function_name = f"test_case_{i}"

            # Get the list of string-based arguments
            input_args_list = test_obj.input.split("\n")

            # Create repr() strings for the generated code
            input_lines_repr = repr(input_args_list)
            expected_output_repr = repr(test_obj.output)

            test_function = [
                f"def {function_name}():",
                f"    # Original Input: {repr(test_obj.input)}",
                f"    solution = {class_name}()",
                f"    input_lines = {input_lines_repr}",
                "    args = [eval(line) for line in input_lines]",
                f"    expected_output = eval({expected_output_repr})",
                f"    assert solution.{method_name}(*args) == expected_output",
                "",
            ]
            output_lines.extend(test_function)

        return "\n".join(output_lines)

    @staticmethod
    def build_test_class_block(test_cases: list[Test], starter_code: str) -> str:
        """
        Generate a pytest test script from LCB problem test case objects.

        Note: Despite the name 'test_class_block', this now generates pytest
        standalone test functions, not unittest classes. The name is kept for
        backward compatibility with existing interfaces.

        Args:
            test_cases: List of Test objects from LCB problem (will be cast to LCBTest)
            starter_code: Starter code for FUNCTIONAL tests

        Returns:
            Complete pytest test script as string with standalone test functions
        """
        # Cast Test objects to LCBTest for type checking
        lcb_test_cases = [LCBTest(**t.__dict__) for t in test_cases]

        if not lcb_test_cases:
            return "# No test cases provided."

        # Check the type of the first test case
        first_test_type = lcb_test_cases[0].testtype

        if first_test_type == TestType.FUNCTIONAL:
            if not starter_code:
                return "# Error: Functional test cases require starter_code."
            # Call the functional test generator
            return LCBDatasetTestBlockBuilder._convert_to_pytest_functional_tests(
                lcb_test_cases, starter_code
            )

        elif first_test_type == TestType.STDIN:
            # Call the STDIN test generator (starter_code is ignored)
            return LCBDatasetTestBlockBuilder._convert_to_pytest_from_stdin_tests(
                lcb_test_cases
            )

        else:
            return f"# Error: Unknown test type '{first_test_type}'."


class LCBTestBlockRebuilder(ITestBlockRebuilder):
    """
    Implementation of ITestBlockRebuilder for LiveCodeBench dataset.

    Rebuilds LCB test cases from unittest test class blocks.
    """

    @staticmethod
    def rebuild_test_block(
        original_class_str: str,
        new_methods_snippets: list[str],
    ) -> str:
        # This method will reconstruct LCBTest objects from the original class string
        # and the new method snippets provided.

        return cpp.composition.rebuild_unittest_with_methods(
            original_class_str, new_methods_snippets
        )
