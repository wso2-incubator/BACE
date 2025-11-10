"""
Test cases for LLM operators using a mock ILanguageModel protocol for testing."""

from typing import Any

import pytest

from common.coevolution.core.interfaces import Problem
from common.coevolution.llm_operators import (
    CodeLLMOperator,
    LLMGenerationError,
    TestLLMOperator,
)


class MockLLM:
    """Mock implementation of ILanguageModel protocol for testing."""

    def __init__(self, response: str | None = None, should_fail: bool = False) -> None:
        """
        Initialize mock LLM.

        Args:
            response: The response to return from generate()
            should_fail: If True, raise an exception on generate()
        """
        self.response: str | None = response
        self.should_fail: bool = should_fail
        self.call_count: int = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        """Mock generate method that returns preset response."""
        self.call_count += 1
        self.prompts.append(prompt)

        if self.should_fail:
            raise Exception("Mock LLM failure")

        if self.response is None:
            raise ValueError("No response configured")

        return self.response


@pytest.fixture
def sample_problem() -> Problem:
    """Create a sample problem for testing."""
    return Problem(
        question_title="Two Sum",
        question_content="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        question_id="two-sum",
        starter_code="def twoSum(nums, target):\n    pass",
        public_test_cases=[],
        private_test_cases=[],
    )


@pytest.fixture
def sample_problem_with_class() -> Problem:
    """Create a sample problem with class-based starter code."""
    return Problem(
        question_title="Solution Problem",
        question_content="Solve this problem.",
        question_id="solution-problem",
        starter_code="class Solution:\n    def solve(self, x):\n        pass",
        public_test_cases=[],
        private_test_cases=[],
    )


class TestCodeLLMOperator:
    """Test cases for CodeLLMOperator."""

    def test_initialization(self, sample_problem: Any) -> None:
        """Test that CodeLLMOperator initializes correctly."""
        mock_llm = MockLLM()
        operator = CodeLLMOperator(mock_llm, sample_problem)

        assert operator.problem == sample_problem
        assert operator.problem.starter_code == sample_problem.starter_code

    def test_initialization_fails_without_starter_code(self) -> None:
        """Test that CodeLLMOperator raises ValueError when problem has no starter code."""
        problem = Problem(
            question_title="Test",
            question_content="Content",
            question_id="test",
            starter_code="",  # Empty starter code
            public_test_cases=[],
            private_test_cases=[],
        )
        mock_llm = MockLLM()

        with pytest.raises(ValueError, match="Problem 'Test' has no starter_code"):
            CodeLLMOperator(mock_llm, problem)

    def test_create_initial_snippets_success(self, sample_problem: Any) -> None:
        """Test successful creation of initial code snippets."""
        response = """Here are 2 solutions:
```python
def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
```

```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
```
"""
        mock_llm = MockLLM(response=response)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        snippets = operator.create_initial_snippets(population_size=2)

        assert len(snippets) == 2
        assert "def twoSum" in snippets[0]
        assert "def twoSum" in snippets[1]
        assert mock_llm.call_count == 1

    def test_create_initial_snippets_wrong_count_retries(
        self, sample_problem: Any
    ) -> None:
        """Test that create_initial_snippets retries when wrong number of snippets generated."""
        # First call returns 1 snippet, should retry
        # After 3 attempts, should raise ValueError
        response_one = """Here is 1 solution:
```python
def twoSum(nums, target):
    return []
```
"""
        mock_llm = MockLLM(response=response_one)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        with pytest.raises(ValueError, match="Generated 1 code snippets, expected 2"):
            operator.create_initial_snippets(population_size=2)

        # Should have tried 3 times (retry decorator)
        assert mock_llm.call_count == 3

    def test_create_initial_snippets_missing_starter_code_retries(
        self, sample_problem: Any
    ) -> None:
        """Test that create_initial_snippets retries when starter code is missing."""
        response = """Here are 2 solutions:
```python
def wrongFunction(nums, target):
    return []
```

```python
def anotherWrong(nums, target):
    return []
```
"""
        mock_llm = MockLLM(response=response)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        with pytest.raises(ValueError, match="does not contain starter code structure"):
            operator.create_initial_snippets(population_size=2)

        # Should have tried 3 times
        assert mock_llm.call_count == 3

    def test_crossover_success(self, sample_problem: Any) -> None:
        """Test successful crossover of two parent solutions."""
        parent1 = """def twoSum(nums, target):
    # Brute force approach
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
"""
        parent2 = """def twoSum(nums, target):
    # Hash map approach
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
"""
        response = """Here's the combined solution:
```python
def twoSum(nums, target):
    # Optimized hash map approach with clear variable names
    num_to_index = {}
    for current_index, current_num in enumerate(nums):
        complement = target - current_num
        if complement in num_to_index:
            return [num_to_index[complement], current_index]
        num_to_index[current_num] = current_index
```
"""
        mock_llm = MockLLM(response=response)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        child = operator.crossover(parent1, parent2)

        assert "def twoSum" in child
        assert "num_to_index" in child
        assert mock_llm.call_count == 1

    def test_crossover_missing_starter_code_retries(self, sample_problem: Any) -> None:
        """Test that crossover retries when result doesn't contain starter code."""
        parent1 = "def twoSum(nums, target):\n    return []"
        parent2 = "def twoSum(nums, target):\n    return []"

        response = """Here's the combined solution:
```python
def wrongName(nums, target):
    return []
```
"""
        mock_llm = MockLLM(response=response)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        with pytest.raises(ValueError, match="does not contain starter code structure"):
            operator.crossover(parent1, parent2)

        # Should have tried 3 times
        assert mock_llm.call_count == 3

    def test_mutate_success(self, sample_problem: Any) -> None:
        """Test successful mutation of a code snippet."""
        individual = """def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
"""
        response = """Here's the mutated solution:
```python
def twoSum(nums, target):
    # Use enumerate for clearer iteration
    for i, num1 in enumerate(nums):
        for j in range(i + 1, len(nums)):
            if num1 + nums[j] == target:
                return [i, j]
```
"""
        mock_llm = MockLLM(response=response)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        mutated = operator.mutate(individual)

        assert "def twoSum" in mutated
        assert "enumerate" in mutated
        assert mock_llm.call_count == 1

    def test_mutate_missing_starter_code_retries(self, sample_problem: Any) -> None:
        """Test that mutate retries when result doesn't contain starter code."""
        individual = "def twoSum(nums, target):\n    return []"

        response = """Here's the mutated solution:
```python
def different(nums, target):
    return []
```
"""
        mock_llm = MockLLM(response=response)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        with pytest.raises(ValueError, match="does not contain starter code structure"):
            operator.mutate(individual)

        assert mock_llm.call_count == 3

    def test_edit_success(self, sample_problem: Any) -> None:
        """Test successful edit based on feedback."""
        individual = """def twoSum(nums, target):
    for i in range(len(nums)):
        if nums[i] + nums[i] == target:  # Bug: should be i and j
            return [i, i]
"""
        feedback = "IndexError: You're using the same index twice"

        response = """Here's the fixed solution:
```python
def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
```
"""
        mock_llm = MockLLM(response=response)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        edited = operator.edit(individual, feedback)

        assert "def twoSum" in edited
        assert "for j in range" in edited
        assert mock_llm.call_count == 1

    def test_edit_missing_starter_code_retries(self, sample_problem: Any) -> None:
        """Test that edit retries when result doesn't contain starter code."""
        individual = "def twoSum(nums, target):\n    return []"
        feedback = "Fix this"

        response = """Here's the fixed solution:
```python
def wrongFunction(nums, target):
    return []
```
"""
        mock_llm = MockLLM(response=response)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        with pytest.raises(ValueError, match="does not contain starter code structure"):
            operator.edit(individual, feedback)

        assert mock_llm.call_count == 3

    def test_llm_failure_raises_error(self, sample_problem: Any) -> None:
        """Test that LLM failures are properly raised."""
        mock_llm = MockLLM(should_fail=True)
        operator = CodeLLMOperator(mock_llm, sample_problem)

        with pytest.raises(LLMGenerationError, match="LLM API call failed"):
            operator.create_initial_snippets(population_size=1)

    def test_empty_response_raises_error(self, sample_problem: Any) -> None:
        """Test that empty LLM responses raise an error."""
        mock_llm = MockLLM(response="")
        operator = CodeLLMOperator(mock_llm, sample_problem)

        with pytest.raises(LLMGenerationError, match="empty response"):
            operator.create_initial_snippets(population_size=1)


class TestTestLLMOperator:
    """Test cases for TestLLMOperator."""

    def test_initialization(self, sample_problem: Any) -> None:
        """Test that TestLLMOperator initializes correctly."""
        mock_llm = MockLLM()
        operator = TestLLMOperator(mock_llm, sample_problem)

        assert operator.problem == sample_problem

    def test_create_initial_snippets_success(self, sample_problem: Any) -> None:
        """Test successful creation of initial test snippets."""
        response = """Here are 2 test cases:
```python
import unittest

class TestTwoSum(unittest.TestCase):
    def test_basic_case(self) -> None:
        self.assertEqual(twoSum([2, 7, 11, 15], 9), [0, 1])
    
    def test_negative_numbers(self) -> None:
        self.assertEqual(twoSum([-1, -2, -3, -4], -6), [1, 2])
```
"""
        mock_llm = MockLLM(response=response)
        operator = TestLLMOperator(mock_llm, sample_problem)

        snippets, full_code = operator.create_initial_snippets(population_size=2)

        assert len(snippets) == 2
        assert "def test_basic_case" in snippets[0]
        assert "def test_negative_numbers" in snippets[1]
        assert "class TestTwoSum" in full_code
        assert mock_llm.call_count == 1

    def test_create_initial_snippets_wrong_count_retries(
        self, sample_problem: Any
    ) -> None:
        """Test that create_initial_snippets retries when wrong number of tests generated."""
        response = """Here is 1 test case:
```python
import unittest

class TestTwoSum(unittest.TestCase):
    def test_basic_case(self) -> None:
        self.assertEqual(twoSum([2, 7], 9), [0, 1])
```
"""
        mock_llm = MockLLM(response=response)
        operator = TestLLMOperator(mock_llm, sample_problem)

        with pytest.raises(ValueError, match="Generated 1 test methods, expected 2"):
            operator.create_initial_snippets(population_size=2)

        # Should have tried 3 times
        assert mock_llm.call_count == 3

    def test_crossover_success(self, sample_problem: Any) -> None:
        """Test successful crossover of two parent test cases."""
        parent1 = """def test_empty_array(self):
    self.assertEqual(twoSum([], 5), [])
"""
        parent2 = """def test_no_solution(self):
    self.assertEqual(twoSum([1, 2, 3], 10), [])
"""
        response = """Here's the new test:
```python
def test_single_element(self):
    self.assertEqual(twoSum([5], 5), [])
```
"""
        mock_llm = MockLLM(response=response)
        operator = TestLLMOperator(mock_llm, sample_problem)

        child = operator.crossover(parent1, parent2)

        assert "def test_single_element" in child
        assert "assertEqual" in child
        assert mock_llm.call_count == 1

    def test_mutate_success(self, sample_problem: Any) -> None:
        """Test successful mutation of a test case."""
        individual = """def test_basic_case(self):
    self.assertEqual(twoSum([2, 7], 9), [0, 1])
"""
        response = """Here's the mutated test:
```python
def test_larger_array(self):
    self.assertEqual(twoSum([1, 2, 3, 4, 5, 6], 10), [3, 5])
```
"""
        mock_llm = MockLLM(response=response)
        operator = TestLLMOperator(mock_llm, sample_problem)

        mutated = operator.mutate(individual)

        assert "def test_larger_array" in mutated
        assert "assertEqual" in mutated
        assert mock_llm.call_count == 1

    def test_edit_success(self, sample_problem: Any) -> None:
        """Test successful edit of a test case based on feedback."""
        individual = """def test_basic_case(self):
    self.assertEqual(twoSum([2, 7], 9), [0, 1])
"""
        feedback = "Need to test edge case with duplicate numbers"

        response = """Here's the new test:
```python
def test_duplicate_numbers(self):
    self.assertEqual(twoSum([3, 3], 6), [0, 1])
```
"""
        mock_llm = MockLLM(response=response)
        operator = TestLLMOperator(mock_llm, sample_problem)

        edited = operator.edit(individual, feedback)

        assert "def test_duplicate_numbers" in edited
        assert "[3, 3]" in edited
        assert mock_llm.call_count == 1

    def test_llm_failure_raises_error(self, sample_problem: Problem) -> None:
        """Test that LLM failures are properly raised."""
        mock_llm = MockLLM(should_fail=True)
        operator = TestLLMOperator(mock_llm, sample_problem)

        with pytest.raises(LLMGenerationError, match="LLM API call failed"):
            operator.create_initial_snippets(population_size=1)
