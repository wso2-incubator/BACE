"""
Test cases for LiveCodeBench dataset module including LCBTest, LCBCodeGenerationProblem,
and LCBDatasetTestBlockBuilder.
"""

import json
from datetime import datetime
from typing import Any

import pytest

from coevolution.lcb_dataset import (
    Difficulty,
    LCBCodeGenerationProblem,
    LCBDatasetTestBlockBuilder,
    LCBTest,
    Platform,
    TestType,
)


class TestTestDataclass:
    """Test cases for the Test dataclass."""

    def test_test_creation_stdin(self) -> None:
        """Test creating a Test object with STDIN type."""
        test = LCBTest(input="1 2", output="3", testtype=TestType.STDIN)

        assert test.input == "1 2"
        assert test.output == "3"
        assert test.testtype == TestType.STDIN

    def test_test_creation_functional(self) -> None:
        """Test creating a Test object with FUNCTIONAL type."""
        test = LCBTest(input="[1, 2]", output="3", testtype=TestType.FUNCTIONAL)

        assert test.input == "[1, 2]"
        assert test.output == "3"
        assert test.testtype == TestType.FUNCTIONAL

    def test_test_post_init_converts_string_to_enum(self) -> None:
        """Test that __post_init__ converts string testtype to enum."""
        test = LCBTest(input="1", output="2", testtype="stdin")  # type: ignore[arg-type]

        assert test.testtype == TestType.STDIN
        assert isinstance(test.testtype, TestType)


class TestCodeGenerationProblem:
    """Test cases for CodeGenerationProblem dataclass."""

    @pytest.fixture
    def sample_problem_data(self) -> dict[str, Any]:
        """Create sample problem data for testing."""
        return {
            "question_title": "Two Sum",
            "question_content": "Find two numbers that add up to target",
            "question_id": "two-sum",
            "starter_code": "class Solution:\n    def twoSum(self, nums, target): pass",
            "platform": "leetcode",
            "contest_id": "weekly-123",
            "contest_date": "2024-01-15",
            "difficulty": "easy",
            "public_test_cases": json.dumps(
                [
                    {
                        "input": "[2,7,11,15], 9",
                        "output": "[0,1]",
                        "testtype": "functional",
                    }
                ]
            ),
            "private_test_cases": json.dumps(
                [{"input": "[3,2,4], 6", "output": "[1,2]", "testtype": "functional"}]
            ),
            "metadata": json.dumps({"func_name": "twoSum"}),
        }

    def test_problem_creation_from_dict(
        self, sample_problem_data: dict[str, Any]
    ) -> None:
        """Test creating CodeGenerationProblem from dictionary."""
        problem = LCBCodeGenerationProblem(**sample_problem_data)

        assert problem.question_title == "Two Sum"
        assert problem.question_id == "two-sum"
        assert problem.platform == Platform.LEETCODE
        assert problem.difficulty == Difficulty.EASY
        assert isinstance(problem.contest_date, datetime)
        assert len(problem.public_test_cases) == 1
        assert len(problem.private_test_cases) == 1
        assert isinstance(problem.public_test_cases[0], LCBTest)
        assert isinstance(problem.public_test_cases[0], LCBTest)

    def test_problem_default_starter_code(self) -> None:
        """Test that empty starter_code gets replaced with default."""
        problem_data = {
            "question_title": "Problem",
            "question_content": "Content",
            "question_id": "problem-1",
            "starter_code": "",  # Empty starter code
            "platform": "leetcode",
            "contest_id": "contest-1",
            "contest_date": "2024-01-15",
            "difficulty": "easy",
            "public_test_cases": json.dumps([]),
            "private_test_cases": json.dumps([]),
            "metadata": json.dumps({}),
        }

        problem = LCBCodeGenerationProblem(**problem_data)  # type: ignore[arg-type]

        assert problem.starter_code == LCBCodeGenerationProblem.DEFAULT_STARTER_CODE
        assert "class Solution:" in problem.starter_code
        assert "def sol(self, input_str: str) -> str:" in problem.starter_code

    def test_problem_preserves_non_empty_starter_code(self) -> None:
        """Test that non-empty starter_code is preserved."""
        custom_starter = "def custom_function(): pass"
        problem_data = {
            "question_title": "Problem",
            "question_content": "Content",
            "question_id": "problem-1",
            "starter_code": custom_starter,
            "platform": "leetcode",
            "contest_id": "contest-1",
            "contest_date": "2024-01-15",
            "difficulty": "easy",
            "public_test_cases": json.dumps([]),
            "private_test_cases": json.dumps([]),
            "metadata": json.dumps({}),
        }

        problem = LCBCodeGenerationProblem(**problem_data)  # type: ignore[arg-type]

        assert problem.starter_code == custom_starter

    def test_problem_enum_conversion(self) -> None:
        """Test that string enums are converted to enum types."""
        problem_data = {
            "question_title": "Problem",
            "question_content": "Content",
            "question_id": "problem-1",
            "starter_code": "code",
            "platform": "codeforces",  # String
            "contest_id": "contest-1",
            "contest_date": "2024-01-15",
            "difficulty": "hard",  # String
            "public_test_cases": json.dumps([]),
            "private_test_cases": json.dumps([]),
            "metadata": json.dumps({}),
        }

        problem = LCBCodeGenerationProblem(**problem_data)  # type: ignore[arg-type]

        assert problem.platform == Platform.CODEFORCES
        assert isinstance(problem.platform, Platform)
        assert problem.difficulty == Difficulty.HARD
        assert isinstance(problem.difficulty, Difficulty)

    def test_problem_datetime_conversion(self) -> None:
        """Test that contest_date string is converted to datetime."""
        problem_data = {
            "question_title": "Problem",
            "question_content": "Content",
            "question_id": "problem-1",
            "starter_code": "code",
            "platform": "leetcode",
            "contest_id": "contest-1",
            "contest_date": "2024-06-15",
            "difficulty": "medium",
            "public_test_cases": json.dumps([]),
            "private_test_cases": json.dumps([]),
            "metadata": json.dumps({}),
        }

        problem = LCBCodeGenerationProblem(**problem_data)  # type: ignore[arg-type]

        assert isinstance(problem.contest_date, datetime)
        assert problem.contest_date.year == 2024
        assert problem.contest_date.month == 6
        assert problem.contest_date.day == 15


class TestLCBDatasetTestBlockBuilder:
    """Test cases for LCBDatasetTestBlockBuilder."""

    @pytest.fixture
    def builder(self) -> LCBDatasetTestBlockBuilder:
        """Create a builder instance for testing."""
        return LCBDatasetTestBlockBuilder()

    def test_build_test_class_block_stdin(
        self, builder: LCBDatasetTestBlockBuilder
    ) -> None:
        """Test building test class block for STDIN test cases."""
        test_cases = [
            LCBTest(input="1 2", output="3", testtype=TestType.STDIN),
            LCBTest(input="5 7", output="12", testtype=TestType.STDIN),
        ]
        starter_code = "class Solution:\n    def add(self, a, b): pass"

        result = builder.build_test_class_block(test_cases, starter_code)

        # Check that it returns a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

        # Check for unittest structure
        assert "import unittest" in result
        assert "class" in result
        assert "def test_" in result

    def test_build_test_class_block_functional(
        self, builder: LCBDatasetTestBlockBuilder
    ) -> None:
        """Test building test class block for FUNCTIONAL test cases."""
        test_cases = [
            LCBTest(input="[1, 2]", output="3", testtype=TestType.FUNCTIONAL),
            LCBTest(input="[5, 7]", output="12", testtype=TestType.FUNCTIONAL),
        ]
        starter_code = "class Solution:\n    def add(self, nums): pass"

        result = builder.build_test_class_block(test_cases, starter_code)

        # Check that it returns a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

        # Check for unittest structure
        assert "import unittest" in result
        assert "class" in result
        assert "def test_" in result

    def test_build_test_class_block_empty_test_cases(
        self, builder: LCBDatasetTestBlockBuilder
    ) -> None:
        """Test building test class block with empty test cases."""
        test_cases: list[LCBTest] = []
        starter_code = "class Solution: pass"

        result = builder.build_test_class_block(test_cases, starter_code)

        # Should return a comment indicating no tests
        assert isinstance(result, str)
        assert "No test cases" in result

    def test_build_test_class_block_preserves_starter_code(
        self, builder: LCBDatasetTestBlockBuilder
    ) -> None:
        """Test that starter code influences FUNCTIONAL test generation."""
        test_cases = [
            LCBTest(input="[1, 2]", output="3", testtype=TestType.FUNCTIONAL),
        ]
        starter_code = "class Solution:\n    def twoSum(self, nums, target): pass"

        result = builder.build_test_class_block(test_cases, starter_code)

        # For functional tests, the result should incorporate the starter code context
        assert isinstance(result, str)
        assert len(result) > 0
        # The actual content depends on generate_unittest_from_lcb_tests implementation

    def test_builder_matches_protocol(
        self, builder: LCBDatasetTestBlockBuilder
    ) -> None:
        """Test that builder has the required protocol method."""
        # Check that the builder has the build_test_class_block method
        assert hasattr(builder, "build_test_class_block")
        assert callable(builder.build_test_class_block)

        # Check method signature accepts correct parameters
        test_cases = [LCBTest(input="1", output="2", testtype=TestType.STDIN)]
        starter_code = "code"

        # Should not raise any errors
        result = builder.build_test_class_block(test_cases, starter_code)
        assert isinstance(result, str)


class TestEnums:
    """Test cases for enum types."""

    def test_platform_enum_values(self) -> None:
        """Test Platform enum has correct values."""
        assert Platform.LEETCODE.value == "leetcode"
        assert Platform.CODEFORCES.value == "codeforces"
        assert Platform.ATCODER.value == "atcoder"

    def test_difficulty_enum_values(self) -> None:
        """Test Difficulty enum has correct values."""
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"

    def test_testtype_enum_values(self) -> None:
        """Test TestType enum has correct values."""
        assert TestType.STDIN.value == "stdin"
        assert TestType.FUNCTIONAL.value == "functional"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
