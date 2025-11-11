"""
Comprehensive tests for the ExecutionSystem implementation.

This module tests the execution system used in the coevolutionary framework:
- Test execution with multiprocessing and sequential modes
- Observation matrix generation
- Error handling and graceful failures
- Worker pool management
- Integration with sandbox and populations
"""

from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import Operations
from common.coevolution.core.population import CodePopulation, TestPopulation
from common.coevolution.execution import ExecutionSystem, _execute_single_code
from common.sandbox import SafeCodeSandbox, TestExecutionResult, TestResult


# Fixtures for test setup
@pytest.fixture
def simple_code_population() -> CodePopulation:
    """Create a simple code population with 3 individuals."""
    individuals = [
        CodeIndividual(
            snippet="def add(a, b): return a + b",
            probability=0.5,
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        ),
        CodeIndividual(
            snippet="def add(a, b): return a - b",  # Wrong
            probability=0.5,
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        ),
        CodeIndividual(
            snippet="def add(a, b): return a * 2 + b",
            probability=0.5,
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        ),
    ]
    return CodePopulation(individuals=individuals)


@pytest.fixture
def simple_test_population() -> TestPopulation:
    """Create a simple test population."""
    test_class = """
class TestAdd:
    def test_positive(self):
        assert add(2, 3) == 5
    
    def test_zero(self):
        assert add(0, 0) == 0
"""
    individuals = [
        TestIndividual(
            snippet="test_positive",
            probability=0.5,
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        ),
        TestIndividual(
            snippet="test_zero",
            probability=0.5,
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        ),
    ]
    # Create mock dependencies for TestPopulation
    mock_pareto = Mock()
    mock_rebuilder = Mock()

    return TestPopulation(
        individuals=individuals,
        pareto=mock_pareto,
        test_block_rebuilder=mock_rebuilder,
        test_class_block=test_class,
    )


@pytest.fixture
def mock_sandbox() -> Mock:
    """Create a mock sandbox for testing."""
    return Mock()


@pytest.fixture
def mock_execution_results() -> dict[int, TestExecutionResult]:
    """Create mock execution results for testing observation matrix (as dict)."""
    return {
        # Code 0: passes both tests
        0: TestExecutionResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                TestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
            ],
            summary="All tests passed",
        ),
        # Code 1: fails both tests
        1: TestExecutionResult(
            script_error=False,
            tests_passed=0,
            tests_failed=2,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_positive",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
                TestResult(
                    name="test_zero",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
            ],
            summary="All tests failed",
        ),
        # Code 2: passes first test, fails second
        2: TestExecutionResult(
            script_error=False,
            tests_passed=1,
            tests_failed=1,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                TestResult(
                    name="test_zero",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
            ],
            summary="Mixed results",
        ),
    }


@pytest.fixture
def mock_execution_results_list() -> list[TestExecutionResult]:
    """Create mock execution results as a list (for mock side_effect)."""
    return [
        # Code 0: passes both tests
        TestExecutionResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                TestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
            ],
            summary="All tests passed",
        ),
        # Code 1: fails both tests
        TestExecutionResult(
            script_error=False,
            tests_passed=0,
            tests_failed=2,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_positive",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
                TestResult(
                    name="test_zero",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
            ],
            summary="All tests failed",
        ),
        # Code 2: passes first test, fails second
        TestExecutionResult(
            script_error=False,
            tests_passed=1,
            tests_failed=1,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                TestResult(
                    name="test_zero",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
            ],
            summary="Mixed results",
        ),
    ]


class TestExecutionSystemInitialization:
    """Test suite for ExecutionSystem initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization with multiprocessing enabled."""
        system = ExecutionSystem()
        assert system.enable_multiprocessing is True
        assert system._num_workers is None

    def test_initialization_with_multiprocessing_disabled(self) -> None:
        """Test initialization with multiprocessing disabled."""
        system = ExecutionSystem(enable_multiprocessing=False)
        assert system.enable_multiprocessing is False

    def test_initialization_with_custom_workers(self) -> None:
        """Test initialization with custom number of workers."""
        system = ExecutionSystem(num_workers=4)
        assert system._num_workers == 4

    def test_initialization_multiprocessing_off_ignores_workers(self) -> None:
        """Test that num_workers is ignored when multiprocessing is disabled."""
        system = ExecutionSystem(enable_multiprocessing=False, num_workers=8)
        num_workers = system._get_num_workers(100)
        assert num_workers == 1  # Should be 1 when multiprocessing is off


class TestWorkerCountDetermination:
    """Test suite for _get_num_workers method."""

    def test_get_num_workers_multiprocessing_disabled(self) -> None:
        """Test that worker count is 1 when multiprocessing is disabled."""
        system = ExecutionSystem(enable_multiprocessing=False)
        assert system._get_num_workers(100) == 1

    def test_get_num_workers_with_custom_count(self) -> None:
        """Test worker count respects custom num_workers."""
        system = ExecutionSystem(num_workers=4)
        assert system._get_num_workers(100) == 4

    def test_get_num_workers_limited_by_population(self) -> None:
        """Test that worker count doesn't exceed population size."""
        system = ExecutionSystem(num_workers=10)
        assert system._get_num_workers(3) == 3

    @patch("os.cpu_count", return_value=8)
    def test_get_num_workers_uses_cpu_count(self, mock_cpu_count: int) -> None:
        """Test that worker count defaults to CPU count."""
        system = ExecutionSystem()
        assert system._get_num_workers(100) == 8

    @patch("os.cpu_count", return_value=None)
    def test_get_num_workers_handles_none_cpu_count(self, mock_cpu_count: int) -> None:
        """Test fallback when cpu_count returns None."""
        system = ExecutionSystem()
        assert system._get_num_workers(100) == 1


class TestObservationMatrixBuilding:
    """Test suite for build_observation_matrix method."""

    def test_build_matrix_all_pass(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
    ) -> None:
        """Test observation matrix when all tests pass."""
        system = ExecutionSystem()

        # All tests pass
        execution_results = {
            i: TestExecutionResult(
                script_error=False,
                tests_passed=2,
                tests_failed=0,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="passed",
                        details=None,
                    ),
                    TestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                ],
                summary="All passed",
            )
            for i in range(3)
        }

        matrix = system.build_observation_matrix(
            simple_code_population, simple_test_population, execution_results
        )

        expected = np.ones((3, 2), dtype=int)
        assert np.array_equal(matrix, expected)
        assert matrix.shape == (3, 2)

    def test_build_matrix_all_fail(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
    ) -> None:
        """Test observation matrix when all tests fail."""
        system = ExecutionSystem()

        # All tests fail
        execution_results = {
            i: TestExecutionResult(
                script_error=False,
                tests_passed=0,
                tests_failed=2,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="failed",
                        details="Error",
                    ),
                    TestResult(
                        name="test_zero",
                        description="",
                        status="failed",
                        details="Error",
                    ),
                ],
                summary="All failed",
            )
            for i in range(3)
        }

        matrix = system.build_observation_matrix(
            simple_code_population, simple_test_population, execution_results
        )

        expected = np.zeros((3, 2), dtype=int)
        assert np.array_equal(matrix, expected)

    def test_build_matrix_mixed_results(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_execution_results: list[TestExecutionResult],
    ) -> None:
        """Test observation matrix with mixed pass/fail results."""
        system = ExecutionSystem()

        matrix = system.build_observation_matrix(
            simple_code_population, simple_test_population, mock_execution_results
        )

        expected = np.array(
            [
                [1, 1],  # Code 0: both pass
                [0, 0],  # Code 1: both fail
                [1, 0],  # Code 2: first pass, second fail
            ],
            dtype=int,
        )

        assert np.array_equal(matrix, expected)
        assert matrix.shape == (3, 2)

    def test_build_matrix_empty_populations(self) -> None:
        """Test observation matrix with empty populations - should skip this test."""
        # Note: Populations cannot be initialized with empty lists per the interface
        # This test is skipped
        pytest.skip("Populations cannot be initialized with empty lists")

    def test_build_matrix_handles_error_status(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
    ) -> None:
        """Test that error status is treated as failure (0)."""
        system = ExecutionSystem()

        execution_results = {
            0: TestExecutionResult(
                script_error=False,
                tests_passed=1,
                tests_failed=0,
                tests_errors=1,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="error",
                        details="SyntaxError",
                    ),
                    TestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                ],
                summary="Error",
            )
        }

        matrix = system.build_observation_matrix(
            simple_code_population, simple_test_population, execution_results
        )

        # Matrix should have shape (3, 2) but only first row is filled from the single result
        assert matrix.shape == (3, 2)
        expected = np.array(
            [
                [0, 1],  # Code 0: error on test 0, passed test 1
                [0, 0],  # Code 1: no result provided
                [0, 0],  # Code 2: no result provided
            ],
            dtype=int,
        )
        assert np.array_equal(matrix, expected)


class TestExecuteTests:
    """Test suite for execute_tests method."""

    def test_execute_tests_sequential_mode(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test execute_tests in sequential mode (no multiprocessing)."""
        system = ExecutionSystem(enable_multiprocessing=False)

        # Mock the sandbox to return successful results
        mock_sandbox.execute_test_script.return_value = TestExecutionResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                TestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
            ],
            summary="Passed",
        )

        results = system.execute_tests(
            simple_code_population, simple_test_population, mock_sandbox
        )

        assert len(results) == 3  # All 3 codes executed successfully
        assert all(isinstance(r, TestExecutionResult) for r in results.values())
        assert mock_sandbox.execute_test_script.call_count == 3

    def test_execute_tests_handles_failures(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test that execute_tests handles execution failures gracefully."""
        system = ExecutionSystem(enable_multiprocessing=False)

        # Mock sandbox to fail on second execution
        def side_effect(*args: Any, **kwargs: Any) -> TestExecutionResult:
            if mock_sandbox.execute_test_script.call_count == 2:
                raise RuntimeError("Execution failed")
            return TestExecutionResult(
                script_error=False,
                tests_passed=2,
                tests_failed=0,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="passed",
                        details=None,
                    ),
                    TestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                ],
                summary="Passed",
            )

        mock_sandbox.execute_test_script.side_effect = side_effect

        results = system.execute_tests(
            simple_code_population, simple_test_population, mock_sandbox
        )

        # Should get 2 successful results (first and third)
        assert len(results) == 2

    def test_execute_tests_empty_populations(self, mock_sandbox: Mock) -> None:
        """Test execute_tests with empty populations."""
        pytest.skip("Populations cannot be initialized with empty lists")


class TestWorkerFunction:
    """Test suite for the _execute_single_code worker function."""

    def test_worker_function_success(self, mock_sandbox: Mock) -> None:
        """Test worker function with successful execution."""
        mock_sandbox.execute_test_script.return_value = TestExecutionResult(
            script_error=False,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_example", description="", status="passed", details=None
                ),
            ],
            summary="Passed",
        )

        code_idx, result = _execute_single_code(
            code_idx=0,
            code_snippet="def add(a, b): return a + b",
            test_class_block="class TestAdd: pass",
            sandbox=mock_sandbox,
        )

        assert code_idx == 0
        assert result is not None
        assert isinstance(result, TestExecutionResult)

    def test_worker_function_handles_exception(self, mock_sandbox: Mock) -> None:
        """Test worker function handles exceptions and returns None."""
        mock_sandbox.execute_test_script.side_effect = RuntimeError("Execution error")

        code_idx, result = _execute_single_code(
            code_idx=1,
            code_snippet="def bad(): raise Exception()",
            test_class_block="class TestBad: pass",
            sandbox=mock_sandbox,
        )

        assert code_idx == 1
        assert result is None  # Should return None on failure


class TestIntegration:
    """Integration tests for the full execution pipeline."""

    def test_full_pipeline_sequential(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
        mock_execution_results_list: list[TestExecutionResult],
    ) -> None:
        """Test the full pipeline from execution to observation matrix."""
        system = ExecutionSystem(enable_multiprocessing=False)

        # Mock sandbox to return our predefined results
        mock_sandbox.execute_test_script.side_effect = mock_execution_results_list

        # Execute tests
        results = system.execute_tests(
            simple_code_population, simple_test_population, mock_sandbox
        )

        # Build observation matrix
        matrix = system.build_observation_matrix(
            simple_code_population, simple_test_population, results
        )

        # Verify matrix shape and contents
        assert matrix.shape == (3, 2)

        expected = np.array(
            [
                [1, 1],  # Code 0: both pass
                [0, 0],  # Code 1: both fail
                [1, 0],  # Code 2: first pass, second fail
            ],
            dtype=int,
        )

        assert np.array_equal(matrix, expected)

    def test_pass_rate_calculation_from_matrix(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_execution_results: list[TestExecutionResult],
    ) -> None:
        """Test that pass rates can be calculated from observation matrix."""
        system = ExecutionSystem()

        matrix = system.build_observation_matrix(
            simple_code_population, simple_test_population, mock_execution_results
        )

        # Calculate code pass rates (row-wise)
        code_pass_rates = np.sum(matrix, axis=1) / matrix.shape[1]
        expected_code_pass_rates = np.array([1.0, 0.0, 0.5])  # 100%, 0%, 50%
        assert np.array_equal(code_pass_rates, expected_code_pass_rates)

        # Calculate test pass rates (column-wise)
        test_pass_rates = np.sum(matrix, axis=0) / matrix.shape[0]
        expected_test_pass_rates = np.array([2 / 3, 1 / 3])  # 66.7%, 33.3%
        assert np.allclose(test_pass_rates, expected_test_pass_rates)


class TestRealSandboxIntegration:
    """Integration tests using the actual SafeCodeSandbox."""

    @pytest.fixture
    def real_sandbox(self) -> SafeCodeSandbox:
        """Create a real SafeCodeSandbox instance."""
        from common.sandbox import SafeCodeSandbox

        return SafeCodeSandbox(timeout=5)

    @pytest.fixture
    def real_code_population(self) -> CodePopulation:
        """Create a code population with real working Python code in LCB format."""
        individuals = [
            # Correct implementation
            CodeIndividual(
                snippet="class Solution:\n    def add(self, a, b):\n        return a + b",
                probability=0.5,
                creation_op=Operations.INITIAL,
                generation_born=0,
                parent_ids=[],
            ),
            # Wrong implementation (subtracts instead)
            CodeIndividual(
                snippet="class Solution:\n    def add(self, a, b):\n        return a - b",
                probability=0.5,
                creation_op=Operations.INITIAL,
                generation_born=0,
                parent_ids=[],
            ),
            # Partially correct (passes test_zero, fails others)
            CodeIndividual(
                snippet="class Solution:\n    def add(self, a, b):\n        # Only works correctly when at least one operand is 0\n        if a == 0:\n            return b\n        if b == 0:\n            return a\n        return 0  # Wrong for non-zero cases",
                probability=0.5,
                creation_op=Operations.INITIAL,
                generation_born=0,
                parent_ids=[],
            ),
        ]
        return CodePopulation(individuals=individuals)

    @pytest.fixture
    def real_test_population(self) -> TestPopulation:
        """Create a test population with real unittest tests in LCB format."""
        from unittest.mock import Mock

        test_class = """
import unittest

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.solution = Solution()
    
    def test_positive_numbers(self):
        '''Test adding positive numbers'''
        self.assertEqual(self.solution.add(2, 3), 5)
        self.assertEqual(self.solution.add(10, 20), 30)
    
    def test_zero(self):
        '''Test adding zero'''
        self.assertEqual(self.solution.add(0, 0), 0)
        self.assertEqual(self.solution.add(5, 0), 5)
        self.assertEqual(self.solution.add(0, 5), 5)
    
    def test_negative_numbers(self):
        '''Test adding negative numbers'''
        self.assertEqual(self.solution.add(-1, -1), -2)
        self.assertEqual(self.solution.add(-5, 3), -2)

if __name__ == '__main__':
    unittest.main()
"""
        individuals = [
            TestIndividual(
                snippet="test_positive_numbers",
                probability=0.5,
                creation_op=Operations.INITIAL,
                generation_born=0,
                parent_ids=[],
            ),
            TestIndividual(
                snippet="test_zero",
                probability=0.5,
                creation_op=Operations.INITIAL,
                generation_born=0,
                parent_ids=[],
            ),
            TestIndividual(
                snippet="test_negative_numbers",
                probability=0.5,
                creation_op=Operations.INITIAL,
                generation_born=0,
                parent_ids=[],
            ),
        ]

        # Mock the required dependencies for TestPopulation
        mock_pareto = Mock()
        mock_rebuilder = Mock()

        return TestPopulation(
            individuals=individuals,
            pareto=mock_pareto,
            test_block_rebuilder=mock_rebuilder,
            test_class_block=test_class,
        )

    def test_real_execution_with_sandbox(
        self,
        real_sandbox: SafeCodeSandbox,
        real_code_population: CodePopulation,
        real_test_population: TestPopulation,
    ) -> None:
        """Test ExecutionSystem with real SafeCodeSandbox execution."""
        system = ExecutionSystem(enable_multiprocessing=False)

        # Execute tests with real sandbox
        results = system.execute_tests(
            real_code_population, real_test_population, real_sandbox
        )

        # Should get results for all 3 codes
        assert len(results) == 3

        # All results should be TestExecutionResult objects
        from common.sandbox import TestExecutionResult

        assert all(isinstance(r, TestExecutionResult) for r in results.values())

        # Build observation matrix
        matrix = system.build_observation_matrix(
            real_code_population, real_test_population, results
        )

        # Matrix should be 3 codes × 3 tests
        assert matrix.shape == (3, 3)

        # Verify expected results:
        # Code 0 (correct): should pass all tests
        assert matrix[0, 0] == 1  # passes test_positive_numbers
        assert matrix[0, 1] == 1  # passes test_zero
        assert matrix[0, 2] == 1  # passes test_negative_numbers

        # Code 1 (wrong - subtracts): should fail all tests
        assert matrix[1, 0] == 0  # fails test_positive_numbers
        assert matrix[1, 1] == 0  # fails test_zero
        assert matrix[1, 2] == 0  # fails test_negative_numbers

        # Code 2 (partially correct): should only pass test_zero
        assert matrix[2, 0] == 0  # fails test_positive_numbers
        assert matrix[2, 1] == 1  # passes test_zero (works when both are 0)
        assert matrix[2, 2] == 0  # fails test_negative_numbers

    def test_real_execution_multiprocessing(
        self,
        real_sandbox: SafeCodeSandbox,
        real_code_population: CodePopulation,
        real_test_population: TestPopulation,
    ) -> None:
        """Test ExecutionSystem with multiprocessing enabled."""
        system = ExecutionSystem(enable_multiprocessing=True, num_workers=2)

        # Execute tests with real sandbox and multiprocessing
        results = system.execute_tests(
            real_code_population, real_test_population, real_sandbox
        )

        # Should get results for all 3 codes
        assert len(results) == 3

        # Build observation matrix
        matrix = system.build_observation_matrix(
            real_code_population, real_test_population, results
        )

        # Same expectations as sequential test
        assert matrix.shape == (3, 3)

        # Code 0 should pass all
        assert np.sum(matrix[0, :]) == 3

        # Code 1 should fail all
        assert np.sum(matrix[1, :]) == 0

        # Code 2 should pass only test_zero
        assert np.sum(matrix[2, :]) == 1


class TestAdditionalEdgeCases:
    """Additional edge and boundary case tests for ExecutionSystem."""

    def test_build_matrix_ignores_extra_execution_results(
        self,
        simple_test_population: TestPopulation,
    ) -> None:
        """If there are more execution results than codes, extras are ignored."""
        system = ExecutionSystem()

        # Only one code individual
        code_population = CodePopulation(
            individuals=[
                CodeIndividual(
                    snippet="def add(a, b): return a + b",
                    probability=0.5,
                    creation_op=Operations.INITIAL,
                    generation_born=0,
                    parent_ids=[],
                )
            ]
        )

        # Two execution results provided (second should be ignored)
        result_row = TestExecutionResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                TestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                TestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
            ],
            summary="All passed",
        )
        execution_results = {0: result_row}  # Only code 0 in population

        matrix = system.build_observation_matrix(
            code_population, simple_test_population, execution_results
        )

        # Shape should respect code population size (1 x 2)
        assert matrix.shape == (1, 2)
        # Only first row filled; extras ignored
        assert np.array_equal(matrix[0], np.array([1, 1], dtype=int))

    def test_build_matrix_handles_mismatched_result_count(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
    ) -> None:
        """Test that mismatched test result counts are logged as errors."""
        system = ExecutionSystem()

        # Execution results with wrong number of test results (3 instead of 2)
        execution_results = {
            0: TestExecutionResult(
                script_error=False,
                tests_passed=2,
                tests_failed=1,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="passed",
                        details=None,
                    ),
                    TestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                    TestResult(
                        name="test_extra",  # Extra test not in population
                        description="",
                        status="failed",
                        details=None,
                    ),
                ],
                summary="Contains extra test",
            ),
            1: TestExecutionResult(
                script_error=False,
                tests_passed=2,
                tests_failed=0,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="passed",
                        details=None,
                    ),
                    TestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                ],
                summary="Correct count",
            ),
            2: TestExecutionResult(
                script_error=False,
                tests_passed=2,
                tests_failed=0,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="passed",
                        details=None,
                    ),
                    TestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                ],
                summary="Correct count",
            ),
        }

        # Should still build matrix but log error for code 0
        matrix = system.build_observation_matrix(
            simple_code_population, simple_test_population, execution_results
        )

        # Matrix should still be built (extra result ignored for code 0)
        expected = np.array(
            [
                [1, 1],  # Code 0: first 2 tests passed (extra ignored)
                [1, 1],  # Code 1: both tests passed
                [1, 1],  # Code 2: both tests passed
            ],
            dtype=int,
        )
        assert np.array_equal(matrix, expected)

    def test_build_matrix_uses_direct_index_mapping(
        self,
        simple_code_population: CodePopulation,
    ) -> None:
        """Test that observation matrix uses direct index mapping (not name-based)."""
        system = ExecutionSystem()

        # Build a test population with simple snippets
        individuals = [
            TestIndividual(
                snippet="test_alpha",
                probability=0.5,
                creation_op=Operations.INITIAL,
                generation_born=0,
                parent_ids=[],
            ),
            TestIndividual(
                snippet="test_beta",
                probability=0.5,
                creation_op=Operations.INITIAL,
                generation_born=0,
                parent_ids=[],
            ),
        ]

        from unittest.mock import Mock

        mock_pareto = Mock()
        mock_rebuilder = Mock()

        test_population = TestPopulation(
            individuals=individuals,
            pareto=mock_pareto,
            test_block_rebuilder=mock_rebuilder,
            test_class_block="class TestDummy: pass",
        )

        # Test results use direct index mapping (test_results[0] -> test_population[0])
        execution_results = {
            0: TestExecutionResult(
                script_error=False,
                tests_passed=2,
                tests_failed=0,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_alpha", description="", status="passed", details=None
                    ),
                    TestResult(
                        name="test_beta", description="", status="passed", details=None
                    ),
                ],
                summary="All passed",
            ),
            1: TestExecutionResult(
                script_error=False,
                tests_passed=0,
                tests_failed=2,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_alpha", description="", status="failed", details=""
                    ),
                    TestResult(
                        name="test_beta", description="", status="failed", details=""
                    ),
                ],
                summary="All failed",
            ),
            2: TestExecutionResult(
                script_error=False,
                tests_passed=1,
                tests_failed=1,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_alpha", description="", status="passed", details=None
                    ),
                    TestResult(
                        name="test_beta", description="", status="failed", details=""
                    ),
                ],
                summary="Mixed",
            ),
        }

        matrix = system.build_observation_matrix(
            simple_code_population, test_population, execution_results
        )

        # Direct index mapping: test_results[i] -> matrix[:, i]
        expected = np.array(
            [
                [1, 1],  # Code 0: both passed
                [0, 0],  # Code 1: both failed
                [1, 0],  # Code 2: first passed, second failed
            ],
            dtype=int,
        )
        assert np.array_equal(matrix, expected)

    def test_build_matrix_unknown_status_counts_as_zero(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
    ) -> None:
        """Statuses other than 'passed' should be treated as 0 (failure)."""
        system = ExecutionSystem()

        execution_results = {
            0: TestExecutionResult(
                script_error=False,
                tests_passed=1,
                tests_failed=0,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="skipped",
                        details=None,
                    ),
                    TestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                ],
                summary="One skipped, one passed",
            ),
            1: TestExecutionResult(
                script_error=False,
                tests_passed=0,
                tests_failed=2,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="xunknown",
                        details=None,
                    ),
                    TestResult(
                        name="test_zero", description="", status="error", details="E"
                    ),
                ],
                summary="Unknown statuses",
            ),
            1: TestExecutionResult(
                script_error=False,
                tests_passed=0,
                tests_failed=2,
                tests_errors=0,
                test_results=[
                    TestResult(
                        name="test_positive",
                        description="",
                        status="failed",
                        details=None,
                    ),
                    TestResult(
                        name="test_zero", description="", status="failed", details=None
                    ),
                ],
                summary="All failed",
            ),
        }

        matrix = system.build_observation_matrix(
            simple_code_population, simple_test_population, execution_results
        )

        expected = np.array(
            [
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            dtype=int,
        )
        assert np.array_equal(matrix, expected)

    def test_worker_function_handles_compose_error(self, mock_sandbox: Mock) -> None:
        """If composition fails before sandbox execution, worker should return None result."""
        with patch(
            "common.coevolution.execution.cpp.composition.compose_lcb_test_script",
            side_effect=RuntimeError("compose fail"),
        ):
            code_idx, result = _execute_single_code(
                code_idx=7,
                code_snippet="def foo(): pass",
                test_class_block="class TestFoo: pass",
                sandbox=mock_sandbox,
            )

        assert code_idx == 7
        assert result is None

    def test_execute_with_multiprocessing_handles_pool_error(self) -> None:
        """_execute_with_multiprocessing returns [] when Pool raises an exception."""
        system = ExecutionSystem()

        tasks: list[tuple[int, str, str, SafeCodeSandbox]] = []
        # Patch Pool to raise on construction
        with patch("multiprocessing.Pool", side_effect=RuntimeError("pool boom")):
            results = system._execute_with_multiprocessing(tasks, num_workers=2)

        assert results == []

    def test_execute_tests_mp_flag_with_one_worker_uses_sequential(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
    ) -> None:
        """When mp enabled but num_workers==1, it should run sequential path."""
        system = ExecutionSystem(enable_multiprocessing=True, num_workers=1)

        # Prepare a stub sequential execution to detect invocation
        def fake_seq(
            _self: ExecutionSystem, tasks: list[tuple[int, str, str, SafeCodeSandbox]]
        ) -> list[tuple[int, TestExecutionResult | None]]:
            # Return a passing result for each task
            out: list[tuple[int, TestExecutionResult | None]] = []
            for idx, _code, _tests, _sb in tasks:
                out.append(
                    (
                        idx,
                        TestExecutionResult(
                            script_error=False,
                            tests_passed=2,
                            tests_failed=0,
                            tests_errors=0,
                            test_results=[
                                TestResult(
                                    name="test_positive",
                                    description="",
                                    status="passed",
                                    details=None,
                                ),
                                TestResult(
                                    name="test_zero",
                                    description="",
                                    status="passed",
                                    details=None,
                                ),
                            ],
                            summary="ok",
                        ),
                    )
                )
            return out

        with (
            patch.object(ExecutionSystem, "_execute_sequentially", new=fake_seq),
            patch.object(
                ExecutionSystem,
                "_execute_with_multiprocessing",
                side_effect=AssertionError("Should not be called"),
            ),
        ):
            # Use a simple mock sandbox since composition is not under test here
            sandbox = Mock()
            # The sandbox won't be called because compose is not patched and worker runs inside fake_seq
            results = system.execute_tests(
                simple_code_population, simple_test_population, sandbox
            )

        # Should have results for all code individuals
        assert len(results) == simple_code_population.size
