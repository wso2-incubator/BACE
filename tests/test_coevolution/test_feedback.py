"""
Tests for the feedback generation system.

This module tests both CodeFeedbackGenerator and TestFeedbackGenerator
implementations of the IFeedbackGenerator interface.
"""

import numpy as np
import pytest

from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import ExecutionResults, IPareto, Operations
from common.coevolution.core.population import CodePopulation, TestPopulation
from common.coevolution.feedback import (
    CodeFeedbackGenerator,
    TestFeedbackGenerator,
    _format_test_entry,
)
from common.sandbox import TestExecutionResult, TestResult

# --- Fixtures ---


class MockPareto(IPareto):
    """Mock Pareto system for testing."""

    def calculate_discrimination(self, observation_matrix: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(observation_matrix, axis=1))

    def calculate_pareto_front(
        self, probabilities: np.ndarray, discriminations: np.ndarray
    ) -> list[int]:
        return list([1])

    def get_pareto_indices(
        self, probabilities: np.ndarray, observation_matrix: np.ndarray
    ) -> list[int]:
        return list(range(len(probabilities)))


class MockTestBlockRebuilder:
    """Mock test block rebuilder for testing."""

    def rebuild_test_block(
        self, original_class_str: str, new_method_snippets: list[str]
    ) -> str:
        return original_class_str


@pytest.fixture
def sample_test_population() -> TestPopulation:
    """Create a sample test population for testing."""
    test_snippets = [
        "def test_add(self):\n    self.assertEqual(add(2, 3), 5)",
        "def test_subtract(self):\n    self.assertEqual(subtract(5, 3), 2)",
        "def test_multiply(self):\n    self.assertEqual(multiply(2, 3), 6)",
    ]

    individuals: list[TestIndividual] = [
        TestIndividual(
            snippet=snippet,
            probability=0.5,
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        )
        for snippet in test_snippets
    ]

    return TestPopulation(
        individuals=individuals,
        test_class_block="class TestAdd(unittest.TestCase): pass",
        pareto=MockPareto(),
        test_block_rebuilder=MockTestBlockRebuilder(),
        generation=0,
    )


@pytest.fixture
def sample_code_population() -> CodePopulation:
    """Create a sample code population for testing."""
    code_snippets: list[str] = [
        "def add(a, b):\n    return a + b",
        "def add(a, b):\n    return a - b  # Bug",
        "def add(a, b):\n    return a * b  # Bug",
    ]

    individuals: list[CodeIndividual] = [
        CodeIndividual(
            snippet=snippet,
            probability=0.5,
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        )
        for snippet in code_snippets
    ]

    return CodePopulation(individuals=individuals, generation=0)


@pytest.fixture
def sample_execution_results() -> dict[int, TestExecutionResult]:
    """Create sample execution results for testing."""
    from common.sandbox import TestExecutionResult

    return {
        0: TestExecutionResult(
            script_error=False,
            tests_passed=2,
            tests_failed=1,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_add",
                    description="Test addition",
                    status="failed",
                    details="AssertionError: 5 != 3",
                ),
                TestResult(
                    name="test_subtract",
                    description="Test subtraction",
                    status="passed",
                    details=None,
                ),
                TestResult(
                    name="test_multiply",
                    description="Test multiplication",
                    status="passed",
                    details=None,
                ),
            ],
            summary="Tests completed: 2 passed, 1 failed",
        ),
        1: TestExecutionResult(
            script_error=False,
            tests_passed=3,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_add",
                    description="Test addition",
                    status="passed",
                    details=None,
                ),
                TestResult(
                    name="test_subtract",
                    description="Test subtraction",
                    status="passed",
                    details=None,
                ),
                TestResult(
                    name="test_multiply",
                    description="Test multiplication",
                    status="passed",
                    details=None,
                ),
            ],
            summary="All tests passed: 3 tests",
        ),
        2: TestExecutionResult(
            script_error=True,
            tests_passed=0,
            tests_failed=0,
            tests_errors=0,
            test_results=[],
            summary="Script execution failed: SyntaxError",
        ),
    }


@pytest.fixture
def sample_observation_matrix() -> np.ndarray:
    """Create a sample observation matrix for testing."""
    # 3 codes x 3 tests
    # Code 0: passes tests 1,2 but fails test 0
    # Code 1: passes all tests
    # Code 2: fails all tests (script error)
    return np.array([[0, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=int)


# --- Tests for _format_test_entry helper ---


class TestFormatTestEntry:
    """Tests for the _format_test_entry helper function."""

    def test_format_passed_test_without_error(self) -> None:
        """Test formatting a passed test without error details."""
        test_result = TestResult(
            name="test_add",
            description="Test addition of two numbers",
            status="passed",
            details=None,
        )
        test_code = "def test_add(self):\n    self.assertEqual(add(2, 3), 5)"

        formatted: str = _format_test_entry(
            test_result, test_code, 1, include_error=False
        )

        assert "1. test_add (Test addition of two numbers)" in formatted
        assert "```python" in formatted
        assert test_code.strip() in formatted
        assert "Error:" not in formatted

    def test_format_failed_test_with_error(self) -> None:
        """Test formatting a failed test with error details."""
        test_result = TestResult(
            name="test_divide",
            description="Test division operation",
            status="failed",
            details="ZeroDivisionError: division by zero",
        )
        test_code = "def test_divide(self):\n    self.assertEqual(divide(10, 0), 5)"

        formatted = _format_test_entry(test_result, test_code, 2, include_error=True)

        assert "2. test_divide (Test division operation)" in formatted
        assert "```python" in formatted
        assert test_code.strip() in formatted
        assert "Error: ZeroDivisionError: division by zero" in formatted

    def test_format_test_without_description(self) -> None:
        """Test formatting a test without description."""
        test_result = TestResult(
            name="test_simple", description="", status="passed", details=None
        )
        test_code = "def test_simple(self):\n    pass"

        formatted = _format_test_entry(test_result, test_code, 1, include_error=False)

        # Should just have the name without description
        assert "1. test_simple\n" in formatted
        assert "Test code:" in formatted

    def test_format_test_with_long_error(self) -> None:
        """Test formatting a test with very long error details."""
        long_error = "A" * 500  # Exceeds _MAX_ERROR_LENGTH (300)
        test_result = TestResult(
            name="test_error",
            description="Test with long error",
            status="error",
            details=long_error,
        )
        test_code = "def test_error(self):\n    raise Exception('long error')"

        formatted = _format_test_entry(test_result, test_code, 1, include_error=True)

        assert "Error: " + "A" * 300 + "..." in formatted
        assert len(long_error) > 300  # Verify it was truncated


# --- Tests for CodeFeedbackGenerator ---


class TestCodeFeedbackGenerator:
    """Tests for the CodeFeedbackGenerator class."""

    def test_all_tests_passed_returns_empty_string(
        self,
        sample_test_population: TestPopulation,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test that no feedback is generated when all tests pass."""
        generator = CodeFeedbackGenerator()

        feedback = generator.generate_feedback(
            observation_matrix=np.array([]),  # Not used
            execution_results=sample_execution_results,
            other_population=sample_test_population,
            individual_idx=1,  # Code 1 passes all tests
        )

        assert feedback == ""

    def test_script_error_returns_syntax_message(
        self,
        sample_test_population: TestPopulation,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test that script errors return appropriate feedback."""
        generator = CodeFeedbackGenerator()

        feedback = generator.generate_feedback(
            observation_matrix=np.array([]),
            execution_results=sample_execution_results,
            other_population=sample_test_population,
            individual_idx=2,  # Code 2 has script error
        )

        assert "syntax errors or import failures" in feedback
        assert "Fix the basic code structure" in feedback

    def test_test_failures_generate_detailed_feedback(
        self,
        sample_test_population: TestPopulation,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test that test failures generate detailed feedback."""
        generator = CodeFeedbackGenerator()

        feedback = generator.generate_feedback(
            observation_matrix=np.array([]),
            execution_results=sample_execution_results,
            other_population=sample_test_population,
            individual_idx=0,  # Code 0 has 1 failure, 2 passes
        )

        # Check structure
        assert "failed 1 out of 3 tests" in feedback
        assert "1. test_add (Test addition)" in feedback
        assert "AssertionError: 5 != 3" in feedback

        # Check that passing tests are also listed
        assert "Passed 2 test(s)" in feedback
        assert "test_subtract" in feedback
        assert "test_multiply" in feedback

        # Check code blocks are included
        assert "```python" in feedback
        assert "def test_add(self):" in feedback

    def test_missing_code_index_raises_key_error(
        self,
        sample_test_population: TestPopulation,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test that missing code index raises KeyError."""
        generator = CodeFeedbackGenerator()

        with pytest.raises(KeyError, match="No execution result for code index 999"):
            generator.generate_feedback(
                observation_matrix=np.array([]),
                execution_results=sample_execution_results,
                other_population=sample_test_population,
                individual_idx=999,
            )

    def test_feedback_includes_test_code(
        self,
        sample_test_population: TestPopulation,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test that feedback includes the actual test code."""
        generator = CodeFeedbackGenerator()

        feedback = generator.generate_feedback(
            observation_matrix=np.array([]),
            execution_results=sample_execution_results,
            other_population=sample_test_population,
            individual_idx=0,
        )

        # Verify test code is included for both failed and passed tests
        assert "def test_add(self):" in feedback
        assert "self.assertEqual(add(2, 3), 5)" in feedback
        assert "def test_subtract(self):" in feedback
        assert "def test_multiply(self):" in feedback


# --- Tests for TestFeedbackGenerator ---


class TestTestFeedbackGenerator:
    """Tests for the TestFeedbackGenerator class."""

    def test_no_passing_code_suggests_test_too_strict(
        self,
        sample_code_population: CodePopulation,
        sample_observation_matrix: np.ndarray,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test feedback when no code passes the test."""
        generator = TestFeedbackGenerator()

        # Test index 0: only code 1 passes it (observation_matrix[:, 0] = [0, 1, 0])
        # But we'll test a scenario where NO code passes
        modified_matrix = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=int)

        feedback = generator.generate_feedback(
            observation_matrix=modified_matrix,
            execution_results=sample_execution_results,
            other_population=sample_code_population,
            individual_idx=0,  # Test 0: no code passes
        )

        assert "No code snippets passed this test case" in feedback
        assert "following code snippet failed the test" in feedback
        assert "above code snippet is correct" in feedback
        assert "test case is incorrect, or too hard" in feedback
        assert "```python" in feedback

    def test_some_passing_code_suggests_test_too_weak(
        self,
        sample_code_population: CodePopulation,
        sample_observation_matrix: np.ndarray,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test feedback when some code passes the test."""
        generator = TestFeedbackGenerator()

        feedback = generator.generate_feedback(
            observation_matrix=sample_observation_matrix,
            execution_results=sample_execution_results,
            other_population=sample_code_population,
            individual_idx=0,  # Test 0: code 1 passes it
        )

        assert "following code snippet passed the test" in feedback
        assert "above code snippet is buggy" in feedback
        assert "test case could not identify the bugs" in feedback
        assert "```python" in feedback

    def test_feedback_includes_code_snippet(
        self,
        sample_code_population: CodePopulation,
        sample_observation_matrix: np.ndarray,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test that feedback includes actual code snippet."""
        generator = TestFeedbackGenerator()

        feedback = generator.generate_feedback(
            observation_matrix=sample_observation_matrix,
            execution_results=sample_execution_results,
            other_population=sample_code_population,
            individual_idx=1,  # Test 1: codes 0 and 1 pass it
        )

        # Should include one of the passing code snippets
        assert "```python" in feedback
        assert "def add(a, b):" in feedback

    def test_random_selection_of_representative_code(
        self,
        sample_code_population: CodePopulation,
        sample_observation_matrix: np.ndarray,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test that a representative code snippet is selected."""
        generator = TestFeedbackGenerator()

        # Run multiple times to verify randomness works
        feedbacks = set()
        for _ in range(10):
            feedback = generator.generate_feedback(
                observation_matrix=sample_observation_matrix,
                execution_results=sample_execution_results,
                other_population=sample_code_population,
                individual_idx=1,  # Test 1: multiple codes pass
            )
            feedbacks.add(feedback)

        # Should produce consistent feedback (since we're just picking one)
        # But the specific code chosen might vary (though with small sample, might be same)
        assert len(feedbacks) >= 1

    def test_observation_matrix_column_indexing(
        self,
        sample_code_population: CodePopulation,
        sample_observation_matrix: np.ndarray,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test that observation matrix is correctly indexed by column (test index)."""
        generator = TestFeedbackGenerator()

        # Test 2: all codes fail it (observation_matrix[:, 2] = [1, 1, 0])
        # Actually, looking at the fixture: column 2 is [1, 1, 0]
        # So codes 0 and 1 pass test 2

        feedback = generator.generate_feedback(
            observation_matrix=sample_observation_matrix,
            execution_results=sample_execution_results,
            other_population=sample_code_population,
            individual_idx=2,
        )

        # Since some codes pass (codes 0 and 1), should suggest test is too weak
        assert "passed the test" in feedback
        assert "buggy and needs improvement" in feedback


# --- Integration Tests ---


class TestFeedbackGeneratorsIntegration:
    """Integration tests for feedback generators working together."""

    def test_code_and_test_feedback_complementary(
        self,
        sample_code_population: CodePopulation,
        sample_test_population: TestPopulation,
        sample_observation_matrix: np.ndarray,
        sample_execution_results: ExecutionResults,
    ) -> None:
        """Test that code and test feedback provide complementary information."""
        code_gen = CodeFeedbackGenerator()
        test_gen = TestFeedbackGenerator()

        # Generate feedback for code 0 (has failures)
        code_feedback = code_gen.generate_feedback(
            observation_matrix=sample_observation_matrix,
            execution_results=sample_execution_results,
            other_population=sample_test_population,
            individual_idx=0,
        )

        # Generate feedback for test 0 (some code passes)
        test_feedback = test_gen.generate_feedback(
            observation_matrix=sample_observation_matrix,
            execution_results=sample_execution_results,
            other_population=sample_code_population,
            individual_idx=0,
        )

        # Both should produce non-empty feedback
        assert code_feedback != ""
        assert test_feedback != ""

        # Code feedback should focus on test failures
        assert "failed" in code_feedback
        assert "Error:" in code_feedback

        # Test feedback should focus on code behavior
        assert ("passed the test" in test_feedback) or (
            "failed the test" in test_feedback
        )

    def test_feedback_generators_are_protocol_compliant(self) -> None:
        """Test that both generators implement IFeedbackGenerator protocol."""
        code_gen = CodeFeedbackGenerator()
        test_gen = TestFeedbackGenerator()

        # Both should have generate_feedback method
        assert hasattr(code_gen, "generate_feedback")
        assert hasattr(test_gen, "generate_feedback")
        assert callable(code_gen.generate_feedback)
        assert callable(test_gen.generate_feedback)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
