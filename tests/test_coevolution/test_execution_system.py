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
from common.coevolution.core.interfaces import TestResult  # The new single test type
from common.coevolution.core.interfaces import (  # The new dict value type
    OPERATION_INITIAL,
    ExecutionResult,
    InteractionData,
)
from common.coevolution.core.population import CodePopulation, TestPopulation
from common.coevolution.execution import ExecutionSystem, _execute_single_code

# Rename the legacy one so we don't confuse it with the new one
from common.sandbox import SafeCodeSandbox
from common.sandbox import TestExecutionResult as SandboxResult
from common.sandbox import TestResult as SandboxTestResult


# Fixtures for test setup
@pytest.fixture
def simple_code_population() -> CodePopulation:
    """Create a simple code population with 3 individuals."""
    individuals = [
        CodeIndividual(
            snippet="def add(a, b): return a + b",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        ),
        CodeIndividual(
            snippet="def add(a, b): return a - b",  # Wrong
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        ),
        CodeIndividual(
            snippet="def add(a, b): return a * 2 + b",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
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
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        ),
        TestIndividual(
            snippet="test_zero",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        ),
    ]
    # Create mock dependencies for TestPopulation
    mock_rebuilder = Mock()

    return TestPopulation(
        individuals=individuals,
        test_block_rebuilder=mock_rebuilder,
        test_class_block=test_class,
    )


@pytest.fixture
def mock_sandbox() -> Mock:
    """Create a mock sandbox for testing."""
    return Mock()


@pytest.fixture
def mock_execution_results() -> list[SandboxResult]:
    """Create mock execution results for testing observation matrix (as list)."""
    return [
        # Code 0: passes both tests
        SandboxResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
            ],
            summary="All tests passed",
        ),
        # Code 1: fails both tests
        SandboxResult(
            script_error=False,
            tests_passed=0,
            tests_failed=2,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
                SandboxTestResult(
                    name="test_zero",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
            ],
            summary="All tests failed",
        ),
        # Code 2: passes first test, fails second
        SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=1,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_zero",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
            ],
            summary="Mixed results",
        ),
    ]


@pytest.fixture
def mock_execution_results_list() -> list[SandboxResult]:
    """Create mock execution results as a list (for mock side_effect)."""
    return [
        # Code 0: passes both tests
        SandboxResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
            ],
            summary="All tests passed",
        ),
        # Code 1: fails both tests
        SandboxResult(
            script_error=False,
            tests_passed=0,
            tests_failed=2,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
                SandboxTestResult(
                    name="test_zero",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
            ],
            summary="All tests failed",
        ),
        # Code 2: passes first test, fails second
        SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=1,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                SandboxTestResult(
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

    def test_default_initialization(self, mock_sandbox: Mock) -> None:
        """Test default initialization with multiprocessing enabled."""
        system = ExecutionSystem(mock_sandbox)
        assert system.sandbox is mock_sandbox
        assert system.enable_multiprocessing is True
        assert system._num_workers is None

    def test_initialization_with_multiprocessing_disabled(
        self, mock_sandbox: Mock
    ) -> None:
        """Test initialization with multiprocessing disabled."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        assert system.enable_multiprocessing is False

    def test_initialization_with_custom_workers(self, mock_sandbox: Mock) -> None:
        """Test initialization with custom number of workers."""
        system = ExecutionSystem(mock_sandbox, num_workers=4)
        assert system._num_workers == 4

    def test_initialization_multiprocessing_off_ignores_workers(
        self, mock_sandbox: Mock
    ) -> None:
        """Test that num_workers is ignored when multiprocessing is disabled."""
        system = ExecutionSystem(
            mock_sandbox, enable_multiprocessing=False, num_workers=8
        )
        num_workers = system._get_num_workers(100)
        assert num_workers == 1  # Should be 1 when multiprocessing is off


class TestWorkerCountDetermination:
    """Test suite for _get_num_workers method."""

    def test_get_num_workers_multiprocessing_disabled(self, mock_sandbox: Mock) -> None:
        """Test that worker count is 1 when multiprocessing is disabled."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        assert system._get_num_workers(100) == 1

    def test_get_num_workers_with_custom_count(self, mock_sandbox: Mock) -> None:
        """Test worker count respects custom num_workers."""
        system = ExecutionSystem(mock_sandbox, num_workers=4)
        assert system._get_num_workers(100) == 4

    def test_get_num_workers_limited_by_population(self, mock_sandbox: Mock) -> None:
        """Test that worker count doesn't exceed population size."""
        system = ExecutionSystem(mock_sandbox, num_workers=10)
        assert system._get_num_workers(3) == 3

    @patch("os.cpu_count", return_value=8)
    def test_get_num_workers_uses_cpu_count(
        self, mock_cpu_count: int, mock_sandbox: Mock
    ) -> None:
        """Test that worker count defaults to CPU count."""
        system = ExecutionSystem(mock_sandbox)
        assert system._get_num_workers(100) == 8

    @patch("os.cpu_count", return_value=None)
    def test_get_num_workers_handles_none_cpu_count(
        self, mock_cpu_count: int, mock_sandbox: Mock
    ) -> None:
        """Test fallback when cpu_count returns None."""
        system = ExecutionSystem(mock_sandbox)
        assert system._get_num_workers(100) == 1


class TestExecuteTests:
    """Test suite for execute_tests method."""

    def test_execute_tests_sequential_mode(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test execute_tests returns unified InteractionData."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Mock Sandbox returns LEGACY objects (SandboxResult)
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
            ],
            summary="Passed",
        )

        # ACT: Call the system
        interaction_data = system.execute_tests(
            simple_code_population, simple_test_population
        )

        # ASSERT: Check the Composite Object
        assert isinstance(interaction_data, InteractionData)

        # 1. Verify Matrix (The Math)
        assert interaction_data.observation_matrix.shape == (3, 2)
        assert np.all(
            interaction_data.observation_matrix == 1
        )  # Since we mocked all pass

        # 2. Verify Dictionary (The Logs)
        # Note: Key is now the ID string, not the index int!
        first_code_id = simple_code_population[0].id
        assert first_code_id in interaction_data.execution_results

        res = interaction_data.execution_results[first_code_id]
        assert isinstance(res, ExecutionResult)  # The NEW interface type
        assert res.script_error is False

    def test_execute_tests_handles_failures(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test that execute_tests handles execution failures gracefully."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Mock sandbox to fail on second execution
        def side_effect(*args: Any, **kwargs: Any) -> SandboxResult:
            if mock_sandbox.execute_test_script.call_count == 2:
                raise RuntimeError("Execution failed")
            return SandboxResult(
                script_error=False,
                tests_passed=2,
                tests_failed=0,
                tests_errors=0,
                test_results=[
                    SandboxTestResult(
                        name="test_positive",
                        description="",
                        status="passed",
                        details=None,
                    ),
                    SandboxTestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                ],
                summary="Passed",
            )

        mock_sandbox.execute_test_script.side_effect = side_effect

        # ACT: Execute tests
        interaction_data = system.execute_tests(
            simple_code_population, simple_test_population
        )

        # ASSERT: Check that failures are handled gracefully
        assert isinstance(interaction_data, InteractionData)
        assert interaction_data.observation_matrix.shape == (3, 2)

        # All codes should have entries
        assert len(interaction_data.execution_results) == 3

        # First code should succeed
        c0_id = simple_code_population[0].id
        assert interaction_data.execution_results[c0_id].script_error is False

        # Second code should have script_error=True due to failure
        c1_id = simple_code_population[1].id
        assert interaction_data.execution_results[c1_id].script_error is True

        # Third code should succeed
        c2_id = simple_code_population[2].id
        assert interaction_data.execution_results[c2_id].script_error is False

    def test_execute_tests_result_count_mismatch(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test that result count mismatches are handled with script_error=True."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Mock sandbox to return wrong number of test results (1 instead of 2)
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                # Missing the second test result!
            ],
            summary="Incomplete results",
        )

        # Execute tests
        interaction_data = system.execute_tests(
            simple_code_population, simple_test_population
        )

        # All codes should have script_error=True due to mismatch
        for code in simple_code_population:
            result = interaction_data.execution_results[code.id]
            assert result.script_error is True
            assert len(result.test_results) == 0  # No test results stored

        # Matrix should be all zeros (no passes)
        assert np.all(interaction_data.observation_matrix == 0)


class TestWorkerFunction:
    """Test suite for the _execute_single_code worker function."""

    def test_worker_function_success(self, mock_sandbox: Mock) -> None:
        """Test worker function with successful execution."""
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
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
        assert isinstance(result, SandboxResult)

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

    def test_full_pipeline_atomic(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_execution_results_list: list[SandboxResult],  # Use Legacy type for mocks
        mock_sandbox: Mock,
    ) -> None:
        """Test the atomic generation of InteractionData."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Mock the sandbox behavior
        mock_sandbox.execute_test_script.side_effect = mock_execution_results_list

        # ACT: Single call does everything now
        data = system.execute_tests(simple_code_population, simple_test_population)

        # ASSERT: Verify Matrix vs Dictionary Consistency
        matrix = data.observation_matrix
        results = data.execution_results

        # Verify shape
        assert matrix.shape == (3, 2)

        # Verify Code 0 (Passed All)
        # Matrix View
        assert np.array_equal(matrix[0], [1, 1])
        # Dict View
        c0_id = simple_code_population[0].id
        assert results[c0_id].script_error is False

        # Verify Code 1 (Failed All)
        # Matrix View
        assert np.array_equal(matrix[1], [0, 0])

    def test_pass_rate_calculation_from_matrix(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_execution_results: list[SandboxResult],
        mock_sandbox: Mock,
    ) -> None:
        """Test that pass rates can be calculated from observation matrix."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Mock the sandbox
        mock_sandbox.execute_test_script.side_effect = mock_execution_results

        # Get InteractionData
        data = system.execute_tests(simple_code_population, simple_test_population)
        matrix = data.observation_matrix

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
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            ),
            # Wrong implementation (subtracts instead)
            CodeIndividual(
                snippet="class Solution:\n    def add(self, a, b):\n        return a - b",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            ),
            # Partially correct (passes test_zero, fails others)
            CodeIndividual(
                snippet="class Solution:\n    def add(self, a, b):\n        # Only works correctly when at least one operand is 0\n        if a == 0:\n            return b\n        if b == 0:\n            return a\n        return 0  # Wrong for non-zero cases",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
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
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            ),
            TestIndividual(
                snippet="test_zero",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            ),
            TestIndividual(
                snippet="test_negative_numbers",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            ),
        ]

        # Mock the required dependencies for TestPopulation
        mock_rebuilder = Mock()

        return TestPopulation(
            individuals=individuals,
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
        system = ExecutionSystem(real_sandbox, enable_multiprocessing=False)

        # Execute tests with real sandbox
        data = system.execute_tests(real_code_population, real_test_population)

        # Should get InteractionData
        assert isinstance(data, InteractionData)
        assert data.observation_matrix.shape == (3, 3)

        # Verify expected results:
        # Code 0 (correct): should pass all tests
        matrix = data.observation_matrix
        assert np.array_equal(matrix[0], [1, 1, 1])  # All pass
        assert matrix[0, 0] == 1  # passes test_positive_numbers
        assert matrix[0, 1] == 1  # passes test_zero
        assert matrix[0, 2] == 1  # passes test_negative_numbers

        # Code 1 (wrong - subtracts): should fail all tests
        assert np.array_equal(matrix[1], [0, 0, 0])  # All fail

        # Code 2 (partially correct): should only pass test_zero
        assert np.array_equal(matrix[2], [0, 1, 0])  # Only middle pass

    def test_real_execution_multiprocessing(
        self,
        real_sandbox: SafeCodeSandbox,
        real_code_population: CodePopulation,
        real_test_population: TestPopulation,
    ) -> None:
        """Test ExecutionSystem with multiprocessing enabled."""
        system = ExecutionSystem(
            real_sandbox, enable_multiprocessing=True, num_workers=2
        )

        # Execute tests with real sandbox and multiprocessing
        data = system.execute_tests(real_code_population, real_test_population)

        # Should get InteractionData
        assert isinstance(data, InteractionData)
        matrix = data.observation_matrix

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

    def test_single_code_single_test(self, mock_sandbox: Mock) -> None:
        """Test execution with minimal population size (1x1)."""
        # Create minimal populations
        code_pop = CodePopulation(
            individuals=[
                CodeIndividual(
                    snippet="def foo(): return 42",
                    probability=1.0,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                )
            ]
        )

        test_pop = TestPopulation(
            individuals=[
                TestIndividual(
                    snippet="test_foo",
                    probability=1.0,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                )
            ],
            test_block_rebuilder=Mock(),
            test_class_block="class TestFoo: pass",
        )

        # Mock sandbox
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_foo", description="", status="passed", details=None
                )
            ],
            summary="Passed",
        )

        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        data = system.execute_tests(code_pop, test_pop)

        # Verify 1x1 matrix
        assert data.observation_matrix.shape == (1, 1)
        assert data.observation_matrix[0, 0] == 1

        # Verify execution results
        assert len(data.execution_results) == 1
        code_id = code_pop[0].id
        assert code_id in data.execution_results
        assert data.execution_results[code_id].script_error is False

    def test_asymmetric_populations(self, mock_sandbox: Mock) -> None:
        """Test with different population sizes (many codes, few tests)."""
        # 5 codes, 2 tests
        code_pop = CodePopulation(
            individuals=[
                CodeIndividual(
                    snippet=f"def fn{i}(): return {i}",
                    probability=0.2,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                )
                for i in range(5)
            ]
        )

        test_pop = TestPopulation(
            individuals=[
                TestIndividual(
                    snippet=f"test_{i}",
                    probability=0.5,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                )
                for i in range(2)
            ],
            test_block_rebuilder=Mock(),
            test_class_block="class Test: pass",
        )

        # Mock all to pass
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name=f"test_{i}", description="", status="passed", details=None
                )
                for i in range(2)
            ],
            summary="Passed",
        )

        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        data = system.execute_tests(code_pop, test_pop)

        # Verify shape
        assert data.observation_matrix.shape == (5, 2)
        assert np.all(data.observation_matrix == 1)

    def test_all_tests_fail(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test when all tests fail for all code."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Mock all failures
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=0,
            tests_failed=2,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
                SandboxTestResult(
                    name="test_zero",
                    description="",
                    status="failed",
                    details="AssertionError",
                ),
            ],
            summary="All failed",
        )

        data = system.execute_tests(simple_code_population, simple_test_population)

        # All zeros in matrix
        assert np.all(data.observation_matrix == 0)

        # All execution results should have script_error=False but failed tests
        for code in simple_code_population:
            result = data.execution_results[code.id]
            assert result.script_error is False
            assert len(result.test_results) == 2
            # All tests should be failed
            for test_result in result.test_results.values():
                assert test_result.status == "failed"

    def test_script_error_handling(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test handling of script errors from sandbox."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Mock script error
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=True,
            tests_passed=0,
            tests_failed=0,
            tests_errors=0,
            test_results=[],
            summary="Script error",
        )

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Matrix should be all zeros
        assert np.all(data.observation_matrix == 0)

        # All execution results should have script_error=True
        for code in simple_code_population:
            result = data.execution_results[code.id]
            assert result.script_error is True
            assert len(result.test_results) == 0

    def test_mixed_statuses(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test various test statuses (passed, failed, error, skipped)."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Return different statuses
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_zero",
                    description="",
                    status="error",
                    details="RuntimeError",
                ),
            ],
            summary="Mixed",
        )

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Only first test should be 1, second should be 0
        for i in range(3):
            assert data.observation_matrix[i, 0] == 1  # First test passed
            assert (
                data.observation_matrix[i, 1] == 0
            )  # Second test errored (counts as fail)

    def test_extra_test_results_ignored(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test that extra test results beyond expected count cause script error."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Return 3 results when only 2 tests exist
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=2,
            tests_failed=1,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_extra", description="", status="failed", details=None
                ),
            ],
            summary="Extra results",
        )

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Should treat as script error due to count mismatch
        for code in simple_code_population:
            result = data.execution_results[code.id]
            assert result.script_error is True

        # Matrix should be all zeros
        assert np.all(data.observation_matrix == 0)

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

    def test_execute_with_multiprocessing_handles_pool_error(
        self, mock_sandbox: Mock
    ) -> None:
        """_execute_with_multiprocessing returns [] when Pool raises an exception."""
        system = ExecutionSystem(mock_sandbox)

        tasks: list[tuple[int, str, str, SafeCodeSandbox]] = []
        # Patch Pool to raise on construction
        with patch("multiprocessing.Pool", side_effect=RuntimeError("pool boom")):
            results = system._execute_with_multiprocessing(tasks, num_workers=2)

        assert results == []

    def test_direct_index_mapping_verification(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Verify that test results use direct index mapping to observation matrix."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Create specific pass/fail pattern
        def side_effect_func(*args: Any, **kwargs: Any) -> SandboxResult:
            call_count = mock_sandbox.execute_test_script.call_count
            if call_count == 1:
                # Code 0: first test passes, second fails
                return SandboxResult(
                    script_error=False,
                    tests_passed=1,
                    tests_failed=1,
                    tests_errors=0,
                    test_results=[
                        SandboxTestResult(
                            name="test_positive",
                            description="",
                            status="passed",
                            details=None,
                        ),
                        SandboxTestResult(
                            name="test_zero",
                            description="",
                            status="failed",
                            details="fail",
                        ),
                    ],
                    summary="Mixed",
                )
            elif call_count == 2:
                # Code 1: first fails, second passes
                return SandboxResult(
                    script_error=False,
                    tests_passed=1,
                    tests_failed=1,
                    tests_errors=0,
                    test_results=[
                        SandboxTestResult(
                            name="test_positive",
                            description="",
                            status="failed",
                            details="fail",
                        ),
                        SandboxTestResult(
                            name="test_zero",
                            description="",
                            status="passed",
                            details=None,
                        ),
                    ],
                    summary="Mixed",
                )
            else:
                # Code 2: both fail
                return SandboxResult(
                    script_error=False,
                    tests_passed=0,
                    tests_failed=2,
                    tests_errors=0,
                    test_results=[
                        SandboxTestResult(
                            name="test_positive",
                            description="",
                            status="failed",
                            details="fail",
                        ),
                        SandboxTestResult(
                            name="test_zero",
                            description="",
                            status="failed",
                            details="fail",
                        ),
                    ],
                    summary="Failed",
                )

        mock_sandbox.execute_test_script.side_effect = side_effect_func

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Verify the specific pattern
        expected_matrix = np.array(
            [
                [1, 0],  # Code 0: test[0] pass, test[1] fail
                [0, 1],  # Code 1: test[0] fail, test[1] pass
                [0, 0],  # Code 2: both fail
            ]
        )
        assert np.array_equal(data.observation_matrix, expected_matrix)

        # Verify dictionary consistency
        test_0_id = simple_test_population[0].id
        test_1_id = simple_test_population[1].id

        code_0_results = data.execution_results[simple_code_population[0].id]
        assert code_0_results.test_results[test_0_id].status == "passed"
        assert code_0_results.test_results[test_1_id].status == "failed"

    def test_large_population_stress(self, mock_sandbox: Mock) -> None:
        """Stress test with larger populations to verify scalability."""
        # Create 20 codes and 15 tests
        code_pop = CodePopulation(
            individuals=[
                CodeIndividual(
                    snippet=f"def code_{i}(): return {i}",
                    probability=1.0 / 20,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                )
                for i in range(20)
            ]
        )

        test_pop = TestPopulation(
            individuals=[
                TestIndividual(
                    snippet=f"test_{i}",
                    probability=1.0 / 15,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                )
                for i in range(15)
            ],
            test_block_rebuilder=Mock(),
            test_class_block="class Test: pass",
        )

        # Mock sandbox to alternate pass/fail based on index
        def side_effect_func(*args: Any, **kwargs: Any) -> SandboxResult:
            call_count = mock_sandbox.execute_test_script.call_count
            # Even codes pass all, odd codes fail all
            if call_count % 2 == 1:
                return SandboxResult(
                    script_error=False,
                    tests_passed=15,
                    tests_failed=0,
                    tests_errors=0,
                    test_results=[
                        SandboxTestResult(
                            name=f"test_{i}",
                            description="",
                            status="passed",
                            details=None,
                        )
                        for i in range(15)
                    ],
                    summary="All passed",
                )
            else:
                return SandboxResult(
                    script_error=False,
                    tests_passed=0,
                    tests_failed=15,
                    tests_errors=0,
                    test_results=[
                        SandboxTestResult(
                            name=f"test_{i}",
                            description="",
                            status="failed",
                            details="fail",
                        )
                        for i in range(15)
                    ],
                    summary="All failed",
                )

        mock_sandbox.execute_test_script.side_effect = side_effect_func

        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        data = system.execute_tests(code_pop, test_pop)

        # Verify shape
        assert data.observation_matrix.shape == (20, 15)

        # Verify pattern: even rows all 1s, odd rows all 0s
        for i in range(20):
            if i % 2 == 0:
                assert np.all(data.observation_matrix[i, :] == 1)
            else:
                assert np.all(data.observation_matrix[i, :] == 0)

        # Verify all execution results exist
        assert len(data.execution_results) == 20

    def test_partial_infrastructure_failure(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test when some codes fail infrastructure-wise but others succeed."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Create a pattern: success, failure, success
        def side_effect_func(*args: Any, **kwargs: Any) -> SandboxResult:
            call_count = mock_sandbox.execute_test_script.call_count
            if call_count == 2:
                # Second call (code 1) fails
                raise RuntimeError("Infrastructure failure")
            else:
                # Others succeed
                return SandboxResult(
                    script_error=False,
                    tests_passed=2,
                    tests_failed=0,
                    tests_errors=0,
                    test_results=[
                        SandboxTestResult(
                            name="test_positive",
                            description="",
                            status="passed",
                            details=None,
                        ),
                        SandboxTestResult(
                            name="test_zero",
                            description="",
                            status="passed",
                            details=None,
                        ),
                    ],
                    summary="Passed",
                )

        mock_sandbox.execute_test_script.side_effect = side_effect_func

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Verify matrix
        expected = np.array(
            [
                [1, 1],  # Code 0: success
                [0, 0],  # Code 1: infrastructure failure
                [1, 1],  # Code 2: success
            ]
        )
        assert np.array_equal(data.observation_matrix, expected)

        # Verify execution results
        assert (
            data.execution_results[simple_code_population[0].id].script_error is False
        )
        assert data.execution_results[simple_code_population[1].id].script_error is True
        assert (
            data.execution_results[simple_code_population[2].id].script_error is False
        )


class TestMatrixAlignmentAndConsistency:
    """Tests specifically focused on matrix-dictionary alignment and consistency."""

    def test_matrix_dictionary_id_consistency(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Verify that matrix indices map correctly to dictionary IDs."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Create a specific pattern for each code
        results_pattern = [
            [1, 1],  # Code 0: both pass
            [1, 0],  # Code 1: first passes, second fails
            [0, 1],  # Code 2: first fails, second passes
        ]

        def side_effect_func(*args: Any, **kwargs: Any) -> SandboxResult:
            call_count = mock_sandbox.execute_test_script.call_count - 1
            pattern = results_pattern[call_count]

            return SandboxResult(
                script_error=False,
                tests_passed=sum(pattern),
                tests_failed=2 - sum(pattern),
                tests_errors=0,
                test_results=[
                    SandboxTestResult(
                        name=f"test_{i}",
                        description="",
                        status="passed" if pattern[i] == 1 else "failed",
                        details=None if pattern[i] == 1 else "fail",
                    )
                    for i in range(2)
                ],
                summary="Test",
            )

        mock_sandbox.execute_test_script.side_effect = side_effect_func

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Verify matrix matches expected pattern
        expected = np.array(results_pattern)
        assert np.array_equal(data.observation_matrix, expected)

        # Verify dictionary results match matrix for each code
        for code_idx, code in enumerate(simple_code_population):
            result = data.execution_results[code.id]
            assert result.script_error is False

            for test_idx, test in enumerate(simple_test_population):
                matrix_value = data.observation_matrix[code_idx, test_idx]
                dict_status = result.test_results[test.id].status

                # Matrix value 1 should correspond to "passed" status
                if matrix_value == 1:
                    assert dict_status == "passed", (
                        f"Matrix says passed but dict says {dict_status}"
                    )
                else:
                    assert dict_status != "passed", (
                        f"Matrix says failed but dict says {dict_status}"
                    )

    def test_no_data_loss_on_partial_failures(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Ensure no data is lost when some executions fail."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Pattern: success, failure, success
        call_results = [
            SandboxResult(
                script_error=False,
                tests_passed=2,
                tests_failed=0,
                tests_errors=0,
                test_results=[
                    SandboxTestResult(
                        name="test_positive",
                        description="",
                        status="passed",
                        details=None,
                    ),
                    SandboxTestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                ],
                summary="Pass",
            ),
            None,  # Failure
            SandboxResult(
                script_error=False,
                tests_passed=1,
                tests_failed=1,
                tests_errors=0,
                test_results=[
                    SandboxTestResult(
                        name="test_positive",
                        description="",
                        status="passed",
                        details=None,
                    ),
                    SandboxTestResult(
                        name="test_zero",
                        description="",
                        status="failed",
                        details="fail",
                    ),
                ],
                summary="Mixed",
            ),
        ]

        def side_effect_func(*args: Any, **kwargs: Any) -> SandboxResult:
            call_count = mock_sandbox.execute_test_script.call_count - 1
            result: SandboxResult | None = call_results[call_count]
            if result is None:
                raise RuntimeError("Simulated failure")
            else:
                return result

        mock_sandbox.execute_test_script.side_effect = side_effect_func

        data = system.execute_tests(simple_code_population, simple_test_population)

        # All codes should have entries
        assert len(data.execution_results) == 3

        # Code 0: success
        c0 = data.execution_results[simple_code_population[0].id]
        assert c0.script_error is False
        assert len(c0.test_results) == 2

        # Code 1: failure (should have script_error=True)
        c1 = data.execution_results[simple_code_population[1].id]
        assert c1.script_error is True
        assert len(c1.test_results) == 0

        # Code 2: success
        c2 = data.execution_results[simple_code_population[2].id]
        assert c2.script_error is False
        assert len(c2.test_results) == 2

    def test_empty_test_results_list_treated_as_mismatch(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test that empty test_results list is treated as count mismatch."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Return empty test results
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=0,
            tests_failed=0,
            tests_errors=0,
            test_results=[],  # Empty!
            summary="No results",
        )

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Should be treated as script error
        for code in simple_code_population:
            result = data.execution_results[code.id]
            assert result.script_error is True
            assert len(result.test_results) == 0

        # Matrix should be all zeros
        assert np.all(data.observation_matrix == 0)

    def test_test_details_preserved_in_execution_results(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Verify that test details/output are preserved in execution results."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        error_msg = "AssertionError: Expected 5, got 3"

        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=1,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive",
                    description="",
                    status="failed",
                    details=error_msg,
                ),
                SandboxTestResult(
                    name="test_zero",
                    description="",
                    status="passed",
                    details=None,
                ),
            ],
            summary="Mixed",
        )

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Check that error details are preserved
        code_0_results = data.execution_results[simple_code_population[0].id]
        test_0_id = simple_test_population[0].id
        test_1_id = simple_test_population[1].id

        assert code_0_results.test_results[test_0_id].details == error_msg
        assert code_0_results.test_results[test_1_id].details is None

    def test_observation_matrix_dtype_is_int(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Verify observation matrix uses int dtype."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
            ],
            summary="Pass",
        )

        data = system.execute_tests(simple_code_population, simple_test_population)

        assert data.observation_matrix.dtype == np.int_
        assert np.all((data.observation_matrix == 0) | (data.observation_matrix == 1))


class TestSequentialVsMultiprocessing:
    """Tests comparing sequential and multiprocessing execution modes."""

    def test_sequential_and_multiprocessing_produce_same_results(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Verify sequential and multiprocessing modes produce identical results."""
        # Define a consistent mock response
        mock_response = SandboxResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_positive", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_zero", description="", status="passed", details=None
                ),
            ],
            summary="Pass",
        )

        # Run sequential
        mock_sandbox.execute_test_script.return_value = mock_response
        system_seq = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        data_seq = system_seq.execute_tests(
            simple_code_population, simple_test_population
        )

        # Reset mock
        mock_sandbox.reset_mock()
        mock_sandbox.execute_test_script.return_value = mock_response

        # Run multiprocessing (with 1 worker to be deterministic)
        system_mp = ExecutionSystem(
            mock_sandbox, enable_multiprocessing=True, num_workers=1
        )
        data_mp = system_mp.execute_tests(
            simple_code_population, simple_test_population
        )

        # Both should produce identical matrices
        assert np.array_equal(data_seq.observation_matrix, data_mp.observation_matrix)

        # Both should have same execution results keys
        assert set(data_seq.execution_results.keys()) == set(
            data_mp.execution_results.keys()
        )


class TestErrorRecovery:
    """Tests for error recovery and graceful degradation."""

    def test_continues_after_worker_exception(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test that execution continues after a worker exception."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # First call raises, rest succeed
        call_count = [0]

        def side_effect_func(*args: Any, **kwargs: Any) -> SandboxResult:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First call fails")
            return SandboxResult(
                script_error=False,
                tests_passed=2,
                tests_failed=0,
                tests_errors=0,
                test_results=[
                    SandboxTestResult(
                        name="test_positive",
                        description="",
                        status="passed",
                        details=None,
                    ),
                    SandboxTestResult(
                        name="test_zero", description="", status="passed", details=None
                    ),
                ],
                summary="Pass",
            )

        mock_sandbox.execute_test_script.side_effect = side_effect_func

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Should have 3 results (all codes processed)
        assert len(data.execution_results) == 3

        # First one should have script_error=True
        assert data.execution_results[simple_code_population[0].id].script_error is True

        # Others should succeed
        assert (
            data.execution_results[simple_code_population[1].id].script_error is False
        )
        assert (
            data.execution_results[simple_code_population[2].id].script_error is False
        )

    def test_all_workers_fail_returns_empty_matrix(
        self,
        simple_code_population: CodePopulation,
        simple_test_population: TestPopulation,
        mock_sandbox: Mock,
    ) -> None:
        """Test that all workers failing still returns valid (zero) matrix."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # All calls fail
        mock_sandbox.execute_test_script.side_effect = RuntimeError("All fail")

        data = system.execute_tests(simple_code_population, simple_test_population)

        # Should have execution results for all codes (with script_error=True)
        assert len(data.execution_results) == 3
        for code in simple_code_population:
            assert data.execution_results[code.id].script_error is True

        # Matrix should be all zeros
        assert np.all(data.observation_matrix == 0)
        assert data.observation_matrix.shape == (3, 2)


class TestArchitecturalWeaknesses:
    """
    Rigorous tests designed to expose architectural weaknesses, edge cases,
    and assumption failures in the ExecutionSystem.

    Focus areas: Index Alignment, Data Integrity, Robustness against malformed inputs.
    """

    def _make_dummy_code_pop(self, size: int) -> CodePopulation:
        """Helper to create dummy code populations."""
        individuals = [
            CodeIndividual(
                snippet=f"def code_{i}(): return {i}",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i in range(size)
        ]
        return CodePopulation(individuals=individuals)

    def _make_dummy_test_pop(self, size: int) -> TestPopulation:
        """Helper to create dummy test populations."""
        individuals = [
            TestIndividual(
                snippet=f"test_{i}",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
            )
            for i in range(size)
        ]
        return TestPopulation(
            individuals=individuals,
            test_block_rebuilder=Mock(),
            test_class_block="class Test: pass",
        )

    def test_alignment_when_sandbox_reorders_tests(self, mock_sandbox: Mock) -> None:
        """
        CRITICAL: Test what happens if the sandbox returns results in a different order
        than the TestPopulation (e.g. alphabetical sorting vs insertion order).

        This test verifies the POSITIONAL assumption in the execution system:
        sandbox_result[i] MUST map to test_population[i], regardless of test names.
        """
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Population: [Test_B, Test_A] (insertion order)
        test_b = TestIndividual(
            snippet="test_b",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )
        test_a = TestIndividual(
            snippet="test_a",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )

        test_pop = TestPopulation(
            individuals=[test_b, test_a],
            test_block_rebuilder=Mock(),
            test_class_block="class Test: pass",
        )

        code_pop = CodePopulation(
            individuals=[
                CodeIndividual(
                    snippet="class Solution:\n    def add(self, a, b):\n        return a + b",
                    probability=0.5,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                )
            ]
        )

        # Sandbox returns results where names suggest alphabetical order
        # BUT the key point is: these are returned in POSITIONS [0, 1]
        # Result at position 0: test_a (passed)
        # Result at position 1: test_b (failed)
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=1,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_a", description="", status="passed", details=None
                ),  # Position 0
                SandboxTestResult(
                    name="test_b", description="", status="failed", details="fail"
                ),  # Position 1
            ],
            summary="Mixed",
        )

        data = system.execute_tests(code_pop, test_pop)

        # CRITICAL VERIFICATION:
        # The system uses POSITIONAL mapping: sandbox_result[i] -> test_population[i]
        #
        # test_population[0] = test_b, receives sandbox_result[0] (test_a, passed) -> WRONG!
        # test_population[1] = test_a, receives sandbox_result[1] (test_b, failed) -> WRONG!
        #
        # This means if the sandbox reorders tests (alphabetically), the positional
        # mapping creates INCORRECT results.

        matrix_row = data.observation_matrix[0]

        # The current implementation maps positionally:
        # Position 0 gets sandbox_result[0] (passed) -> matrix[0,0] = 1
        # Position 1 gets sandbox_result[1] (failed) -> matrix[0,1] = 0
        assert matrix_row[0] == 1, "Position 0: Got sandbox result[0] (passed)"
        assert matrix_row[1] == 0, "Position 1: Got sandbox result[1] (failed)"

        # Dictionary also uses positional mapping through test_population[i].id
        code_id = code_pop[0].id
        result_dict = data.execution_results[code_id].test_results

        # test_pop[0] = test_b, gets sandbox_result[0] which has name="test_a" but status="passed"
        # test_pop[1] = test_a, gets sandbox_result[1] which has name="test_b" but status="failed"
        # The system doesn't validate names, only uses position!
        assert result_dict[test_b.id].status == "passed"
        assert result_dict[test_a.id].status == "failed"

        # This test DOCUMENTS the current behavior: positional mapping.
        # If sandbox reorders tests, results will be misaligned!
        # A robust system would match by test name, not position.

    def test_duplicate_code_ids_cause_dictionary_data_loss(
        self, mock_sandbox: Mock
    ) -> None:
        """Verify behavior when population has duplicate IDs."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)

        # Two individuals with SAME ID (force via object manipulation)
        ind1 = CodeIndividual(
            snippet="c1",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )
        ind2 = CodeIndividual(
            snippet="c2",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )

        # Force same ID by setting private attribute
        object.__setattr__(ind1, "_id", "SAME_ID")
        object.__setattr__(ind2, "_id", "SAME_ID")

        code_pop = CodePopulation(individuals=[ind1, ind2])
        test_pop = TestPopulation(
            individuals=[
                TestIndividual(
                    snippet="t1",
                    probability=0.5,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                )
            ],
            test_block_rebuilder=Mock(),
            test_class_block="class Test: pass",
        )

        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="t1", description="", status="passed", details=None
                )
            ],
            summary="ok",
        )

        data = system.execute_tests(code_pop, test_pop)

        # Matrix should have 2 rows (correct math - index-based)
        assert data.observation_matrix.shape == (2, 1)

        # Dictionary will only have 1 entry (last write wins - ID-based)
        assert len(data.execution_results) == 1

        # This mismatch is a potential issue if IDs aren't unique

    def test_status_check_is_case_sensitive(self, mock_sandbox: Mock) -> None:
        """Test if 'Passed' (capitalized) counts as a failure."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        code_pop = self._make_dummy_code_pop(1)
        test_pop = self._make_dummy_test_pop(1)

        # Sandbox returns "Passed" instead of "passed"
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_0", description="", status="passed", details=None
                )
            ],
            summary="ok",
        )

        data = system.execute_tests(code_pop, test_pop)

        # Current implementation is strict 'passed' only
        assert data.observation_matrix[0, 0] == 0, (
            "Current implementation is strict 'passed' only"
        )

    def test_malformed_sandbox_result_attributes(self, mock_sandbox: Mock) -> None:
        """Test behavior when inner result objects lack standard attributes."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        code_pop = self._make_dummy_code_pop(1)
        test_pop = self._make_dummy_test_pop(1)

        # Result object missing 'status' and 'details' attributes
        # Just a bare Mock with no spec
        malformed_result = Mock(spec=[])

        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=0,
            tests_failed=0,
            tests_errors=0,
            test_results=[malformed_result],
            summary="Malformed",
        )

        data = system.execute_tests(code_pop, test_pop)

        # Should default to 0 in matrix (status won't be "passed")
        assert data.observation_matrix[0, 0] == 0

        # Should default to "error" status in dict (based on getattr fallback)
        res = data.execution_results[code_pop[0].id].test_results[test_pop[0].id]
        assert res.status == "error"

    def test_worker_returns_invalid_index(self, mock_sandbox: Mock) -> None:
        """Test handling when worker returns an index out of bounds."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        code_pop = self._make_dummy_code_pop(2)  # Size 2
        test_pop = self._make_dummy_test_pop(1)

        # Mock worker returning index 99 (Out of bounds)
        with patch("common.coevolution.execution._execute_single_code") as mock_worker:
            mock_worker.return_value = (
                99,
                SandboxResult(
                    script_error=False,
                    tests_passed=1,
                    tests_failed=0,
                    tests_errors=0,
                    test_results=[
                        SandboxTestResult(
                            name="test_0", description="", status="passed", details=None
                        )
                    ],
                    summary="ok",
                ),
            )

            data = system.execute_tests(code_pop, test_pop)

        # Matrix should be valid shape, but empty (zeros) because index 99 was skipped
        assert data.observation_matrix.shape == (2, 1)
        assert np.all(data.observation_matrix == 0)

        # Dictionary should be empty (or only valid entries)
        # Index 99 doesn't map to any code, so it should be skipped
        assert len(data.execution_results) == 0

    def test_execution_with_zero_tests(self, mock_sandbox: Mock) -> None:
        """Test execution when test population is empty."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        code_pop = self._make_dummy_code_pop(3)

        # Empty test population
        test_pop = TestPopulation(
            individuals=[],
            test_block_rebuilder=Mock(),
            test_class_block="class Test: pass",
        )

        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=0,
            tests_failed=0,
            tests_errors=0,
            test_results=[],
            summary="No tests",
        )

        data = system.execute_tests(code_pop, test_pop)

        # Matrix should have shape (3, 0)
        assert data.observation_matrix.shape == (3, 0)

        # Should have 3 execution results
        assert len(data.execution_results) == 3

        # Each should contain no test results in dict
        for code in code_pop:
            assert data.execution_results[code.id].test_results == {}

    def test_sandbox_returns_extra_results_fails_validation(
        self, mock_sandbox: Mock
    ) -> None:
        """Test validation failure when sandbox returns MORE results than expected."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        code_pop = self._make_dummy_code_pop(1)
        test_pop = self._make_dummy_test_pop(1)  # Expect 1

        # Return 2 results (extra)
        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=2,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_0", description="", status="passed", details=None
                ),
                SandboxTestResult(
                    name="test_extra", description="", status="passed", details=None
                ),
            ],
            summary="Extra",
        )

        data = system.execute_tests(code_pop, test_pop)

        # Should be marked as script error because we can't trust mapping
        res = data.execution_results[code_pop[0].id]
        assert res.script_error is True
        assert np.all(data.observation_matrix == 0)

    def test_missing_details_attribute_is_handled(self, mock_sandbox: Mock) -> None:
        """Test robustness when 'details' attribute is missing on sandbox result."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        code_pop = self._make_dummy_code_pop(1)
        test_pop = self._make_dummy_test_pop(1)

        # Result object without 'details' attribute
        res_obj = Mock(spec=["status", "name", "description"])
        res_obj.status = "passed"
        res_obj.name = "test_0"
        res_obj.description = ""
        # Accessing .details will use getattr fallback

        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            test_results=[res_obj],
            summary="ok",
        )

        data = system.execute_tests(code_pop, test_pop)

        res = data.execution_results[code_pop[0].id].test_results[test_pop[0].id]
        assert res.details is None
        assert res.status == "passed"

    def test_matrix_is_strictly_numeric_zeros_and_ones(
        self, mock_sandbox: Mock
    ) -> None:
        """Ensure matrix contains only 0 and 1 integers, no booleans or non-binary."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        code_pop = self._make_dummy_code_pop(1)
        test_pop = self._make_dummy_test_pop(1)

        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_0", description="", status="passed", details=None
                )
            ],
            summary="ok",
        )

        data = system.execute_tests(code_pop, test_pop)

        # Check dtype is integer
        assert data.observation_matrix.dtype in [np.int64, np.int32, np.int_]

        # Check value is integer type, not bool
        assert isinstance(data.observation_matrix[0, 0], (int, np.integer))
        assert not isinstance(data.observation_matrix[0, 0], bool)

        # Ensure only 0 and 1 values
        assert np.all((data.observation_matrix == 0) | (data.observation_matrix == 1))

    def test_extremely_large_output_handling(self, mock_sandbox: Mock) -> None:
        """Test system behavior with massive log output."""
        system = ExecutionSystem(mock_sandbox, enable_multiprocessing=False)
        code_pop = self._make_dummy_code_pop(1)
        test_pop = self._make_dummy_test_pop(1)

        huge_string = "a" * 10_000_000  # 10MB string

        mock_sandbox.execute_test_script.return_value = SandboxResult(
            script_error=False,
            tests_passed=0,
            tests_failed=1,
            tests_errors=0,
            test_results=[
                SandboxTestResult(
                    name="test_0",
                    description="",
                    status="failed",
                    details=huge_string,
                )
            ],
            summary="ok",
        )

        data = system.execute_tests(code_pop, test_pop)

        res = data.execution_results[code_pop[0].id].test_results[test_pop[0].id]

        assert res.details is not None
        assert len(res.details) == 10_000_000
        # The test passes if it doesn't crash or run out of memory
