"""
Test cases for LLM operators using a mock ILanguageModel protocol for testing.
"""

from typing import Any

import pytest

from common.coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    InitialInput,
    Problem,
)
from common.coevolution.operators.base_llm_operator import LLMGenerationError
from common.coevolution.operators.code_llm_operator import (
    CodeCrossoverInput,
    CodeEditInput,
    CodeLLMOperator,
    CodeMutationInput,
)
from common.coevolution.operators.unittest_llm_operator import (
    UnittestCrossoverInput,
    UnittestEditInput,
    UnittestLLMOperator,
    UnittestMutationInput,
)


class MockLLM:
    """Mock implementation of ILanguageModel protocol for testing."""

    def __init__(
        self,
        response: str | None = None,
        should_fail: bool = False,
        responses: list[str] | None = None,
    ) -> None:
        """
        Initialize mock LLM.

        Args:
            response: The response to return from generate()
            should_fail: If True, raise an exception on generate()
            responses: List of responses to return sequentially
        """
        self.response: str | None = response
        self.responses: list[str] | None = responses
        self.response_index: int = 0
        self.should_fail: bool = should_fail
        self.call_count: int = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        """Mock generate method that returns preset response."""
        self.call_count += 1
        self.prompts.append(prompt)

        if self.should_fail:
            raise Exception("Mock LLM failure")

        if self.responses:
            if self.response_index >= len(self.responses):
                raise ValueError("No more responses configured")
            response = self.responses[self.response_index]
            self.response_index += 1
            return response

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
        operator = CodeLLMOperator(mock_llm)

        assert operator._llm is mock_llm
        assert isinstance(operator, CodeLLMOperator)

    def test_constructor_accepts_no_problem(self) -> None:
        """Test that CodeLLMOperator can be constructed without a Problem."""
        mock_llm = MockLLM()
        operator = CodeLLMOperator(mock_llm)

        assert operator._llm is mock_llm
        assert isinstance(operator, CodeLLMOperator)

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
        operator = CodeLLMOperator(mock_llm)

        output, _ = operator.generate_initial_snippets(
            InitialInput(
                operation=OPERATION_INITIAL,
                question_content=sample_problem.question_content,
                population_size=2,
                starter_code=sample_problem.starter_code,
            )
        )
        snippets = [r.snippet for r in output.results]

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
        operator = CodeLLMOperator(mock_llm)

        with pytest.raises(ValueError, match="Generated 1 code snippets, expected 2"):
            operator.generate_initial_snippets(
                InitialInput(
                    operation=OPERATION_INITIAL,
                    question_content=sample_problem.question_content,
                    population_size=2,
                    starter_code=sample_problem.starter_code,
                )
            )

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
        operator = CodeLLMOperator(mock_llm)

        with pytest.raises(
            ValueError,
            match="One of the generated code snippets does not contain starter code",
        ):
            operator.generate_initial_snippets(
                InitialInput(
                    operation=OPERATION_INITIAL,
                    question_content=sample_problem.question_content,
                    population_size=2,
                    starter_code=sample_problem.starter_code,
                )
            )

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
        operator = CodeLLMOperator(mock_llm)

        dto = CodeCrossoverInput(
            operation=OPERATION_CROSSOVER,
            question_content=sample_problem.question_content,
            parent1_snippet=parent1,
            parent2_snippet=parent2,
            starter_code=sample_problem.starter_code,
        )
        output = operator.apply(dto)
        child = output.results[0].snippet

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
        operator = CodeLLMOperator(mock_llm)

        dto = CodeCrossoverInput(
            operation=OPERATION_CROSSOVER,
            question_content=sample_problem.question_content,
            parent1_snippet=parent1,
            parent2_snippet=parent2,
            starter_code=sample_problem.starter_code,
        )

        with pytest.raises(ValueError, match="does not contain starter code structure"):
            operator.apply(dto)

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
        operator = CodeLLMOperator(mock_llm)

        dto = CodeMutationInput(
            operation=OPERATION_MUTATION,
            question_content=sample_problem.question_content,
            parent_snippet=individual,
            starter_code=sample_problem.starter_code,
        )
        output = operator.apply(dto)
        mutated = output.results[0].snippet

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
        operator = CodeLLMOperator(mock_llm)

        dto = CodeMutationInput(
            operation=OPERATION_MUTATION,
            question_content=sample_problem.question_content,
            parent_snippet=individual,
            starter_code=sample_problem.starter_code,
        )

        with pytest.raises(ValueError, match="does not contain starter code structure"):
            operator.apply(dto)

        assert mock_llm.call_count == 3

    def test_edit_success(self, sample_problem: Any) -> None:
        """Test successful edit based on feedback."""
        individual = """def twoSum(nums, target):
    for i in range(len(nums)):
        if nums[i] + nums[i] == target:  # Bug: should be i and j
            return [i, i]
"""
        failing_test_case = "assert twoSum([2, 7, 11, 15], 9) == [0, 1]"
        error_trace = "IndexError: You're using the same index twice"

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
        operator = CodeLLMOperator(mock_llm)

        dto = CodeEditInput(
            operation=OPERATION_EDIT,
            question_content=sample_problem.question_content,
            parent_snippet=individual,
            failing_test_case=failing_test_case,
            error_trace=error_trace,
            starter_code=sample_problem.starter_code,
        )
        output = operator.apply(dto)
        edited = output.results[0].snippet

        assert "def twoSum" in edited
        assert "for j in range" in edited
        assert mock_llm.call_count == 1

    def test_edit_missing_starter_code_retries(self, sample_problem: Any) -> None:
        """Test that edit retries when result doesn't contain starter code."""
        individual = "def twoSum(nums, target):\n    return []"
        failing_test_case = "assert twoSum([2, 7], 9) == [0, 1]"
        error_trace = "AssertionError: assert None == [0, 1]"

        response = """Here's the fixed solution:
```python
def wrongFunction(nums, target):
    return []
```
"""
        mock_llm = MockLLM(response=response)
        operator = CodeLLMOperator(mock_llm)

        dto = CodeEditInput(
            operation=OPERATION_EDIT,
            question_content=sample_problem.question_content,
            parent_snippet=individual,
            failing_test_case=failing_test_case,
            error_trace=error_trace,
            starter_code=sample_problem.starter_code,
        )

        with pytest.raises(ValueError, match="does not contain starter code structure"):
            operator.apply(dto)

        assert mock_llm.call_count == 3

    def test_llm_failure_raises_error(self, sample_problem: Any) -> None:
        """Test that LLM failures are properly raised."""
        mock_llm = MockLLM(should_fail=True)
        operator = CodeLLMOperator(mock_llm)

        with pytest.raises(LLMGenerationError, match="LLM API call failed"):
            operator.generate_initial_snippets(
                InitialInput(
                    operation=OPERATION_INITIAL,
                    question_content=sample_problem.question_content,
                    population_size=1,
                    starter_code=sample_problem.starter_code,
                )
            )

    def test_empty_response_raises_error(self, sample_problem: Any) -> None:
        """Test that empty LLM responses raise an error."""
        mock_llm = MockLLM(response="")
        operator = CodeLLMOperator(mock_llm)

        with pytest.raises(LLMGenerationError, match="empty response"):
            operator.generate_initial_snippets(
                InitialInput(
                    operation=OPERATION_INITIAL,
                    question_content=sample_problem.question_content,
                    population_size=1,
                    starter_code=sample_problem.starter_code,
                )
            )


class TestTestLLMOperator:
    """Test cases for TestLLMOperator."""

    def test_initialization(self, sample_problem: Any) -> None:
        """Test that UnittestLLMOperator initializes correctly."""
        mock_llm = MockLLM()
        operator = UnittestLLMOperator(mock_llm)

        assert operator._llm is mock_llm

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
        operator = UnittestLLMOperator(mock_llm)

        output, full_code = operator.generate_initial_snippets(
            InitialInput(
                operation=OPERATION_INITIAL,
                question_content=sample_problem.question_content,
                population_size=2,
                starter_code=sample_problem.starter_code,
            )
        )
        snippets = [r.snippet for r in output.results]

        assert len(snippets) == 2
        assert "def test_basic_case" in snippets[0]
        assert "def test_negative_numbers" in snippets[1]
        assert "class TestTwoSum" in full_code
        assert mock_llm.call_count == 1

    def test_create_initial_snippets_adjusts_count(self, sample_problem: Any) -> None:
        """Test that create_initial_snippets adjusts when wrong number of tests generated."""
        # First response has 1 test method, second has 1 more
        response1 = """Here is 1 test case:
```python
import unittest

class TestTwoSum(unittest.TestCase):
    def test_basic_case(self) -> None:
        self.assertEqual(twoSum([2, 7], 9), [0, 1])
```
"""
        response2 = """Here is 1 additional test case:
```python
import unittest

class TestTwoSum(unittest.TestCase):
    def test_additional_case(self) -> None:
        self.assertEqual(twoSum([1, 2], 3), [0, 1])
```
"""
        mock_llm = MockLLM(responses=[response1, response2])
        operator = UnittestLLMOperator(mock_llm)

        output, full_code = operator.generate_initial_snippets(
            InitialInput(
                operation=OPERATION_INITIAL,
                question_content=sample_problem.question_content,
                population_size=2,
                starter_code=sample_problem.starter_code,
            )
        )
        snippets = [r.snippet for r in output.results]

        assert len(snippets) == 2
        assert "def test_basic_case" in snippets[0]
        assert "def test_additional_case" in snippets[1]
        # Should have made 2 calls: initial and additional
        assert mock_llm.call_count == 2

    def test_create_initial_snippets_trims_excess(self, sample_problem: Any) -> None:
        """Test that create_initial_snippets trims excess test methods."""
        response = """Here are 3 test cases:
```python
import unittest

class TestTwoSum(unittest.TestCase):
    def test_basic_case(self) -> None:
        self.assertEqual(twoSum([2, 7, 11, 15], 9), [0, 1])
    
    def test_negative_numbers(self) -> None:
        self.assertEqual(twoSum([-1, -2, -3, -4], -6), [1, 2])
    
    def test_extra_case(self) -> None:
        self.assertEqual(twoSum([1, 3, 5], 4), [0, 1])
```
"""
        mock_llm = MockLLM(response=response)
        operator = UnittestLLMOperator(mock_llm)

        output, full_code = operator.generate_initial_snippets(
            InitialInput(
                operation=OPERATION_INITIAL,
                question_content=sample_problem.question_content,
                population_size=2,
                starter_code=sample_problem.starter_code,
            )
        )
        snippets = [r.snippet for r in output.results]

        assert len(snippets) == 2
        assert "def test_basic_case" in snippets[0]
        assert "def test_negative_numbers" in snippets[1]
        assert "def test_extra_case" not in "\n".join(snippets)
        assert mock_llm.call_count == 1

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
        operator = UnittestLLMOperator(mock_llm)

        dto = UnittestCrossoverInput(
            operation=OPERATION_CROSSOVER,
            question_content=sample_problem.question_content,
            parent1_snippet=parent1,
            parent2_snippet=parent2,
        )
        output = operator.apply(dto)
        child = output.results[0].snippet

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
        operator = UnittestLLMOperator(mock_llm)

        dto = UnittestMutationInput(
            operation=OPERATION_MUTATION,
            question_content=sample_problem.question_content,
            parent_snippet=individual,
        )
        output = operator.apply(dto)
        mutated = output.results[0].snippet

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
        operator = UnittestLLMOperator(mock_llm)

        dto = UnittestEditInput(
            operation=OPERATION_EDIT,
            question_content=sample_problem.question_content,
            parent_snippet=individual,
            feedback=feedback,
        )
        output = operator.apply(dto)
        edited = output.results[0].snippet

        assert "def test_duplicate_numbers" in edited
        assert "[3, 3]" in edited
        assert mock_llm.call_count == 1

    def test_llm_failure_raises_error(self, sample_problem: Problem) -> None:
        """Test that LLM failures are properly raised."""
        mock_llm = MockLLM(should_fail=True)
        operator = UnittestLLMOperator(mock_llm)

        with pytest.raises(LLMGenerationError, match="LLM API call failed"):
            operator.generate_initial_snippets(
                InitialInput(
                    operation=OPERATION_INITIAL,
                    question_content=sample_problem.question_content,
                    population_size=1,
                    starter_code=sample_problem.starter_code,
                )
            )
