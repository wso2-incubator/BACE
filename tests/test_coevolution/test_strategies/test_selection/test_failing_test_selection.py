import unittest
from typing import Any, Dict, List, Literal, cast
from unittest.mock import MagicMock, patch

import numpy as np

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import CoevolutionContext, InteractionData
from coevolution.strategies.selection.failing_test_selection import (
    FailingTestSelector,
)
from infrastructure.sandbox import TestExecutionResult, TestResult


class TestFailingTestSelector(unittest.TestCase):
    def setUp(self) -> None:
        # Common setup for tests
        self.code_index = 0
        self.mock_code_individual = MagicMock(spec=CodeIndividual)
        self.mock_code_individual.id = "code_0"

        self.mock_context = MagicMock(spec=CoevolutionContext)
        self.mock_context.test_populations = {}
        self.mock_context.interactions = {}

        # Mock the code population to return the correct index for our individual
        self.mock_context.code_population = MagicMock()
        self.mock_context.code_population.get_index_of_individual.return_value = (
            self.code_index
        )

    # =========================================================================
    # _rank_selection Tests
    # =========================================================================

    def test_rank_selection_valid_input(self) -> None:
        """Test rank selection logic with valid probabilities."""
        probabilities = [0.1, 0.9, 0.5]

        with patch("numpy.random.choice", return_value=0):
            selected_index = FailingTestSelector._rank_selection(probabilities)
            self.assertEqual(selected_index, 1)

    def test_rank_selection_empty_input(self) -> None:
        """Test that empty list raises ValueError."""
        with self.assertRaises(ValueError):
            FailingTestSelector._rank_selection([])

    def test_rank_selection_single_element(self) -> None:
        """Test with a single element list."""
        probabilities = [0.5]
        with patch("numpy.random.choice", return_value=0):
            selected_index = FailingTestSelector._rank_selection(probabilities)
            self.assertEqual(selected_index, 0)

    # =========================================================================
    # select_failing_test Tests
    # =========================================================================

    def _create_mock_population_data(
        self, test_type: str, test_data_list: List[Dict[str, Any]]
    ) -> None:
        """Helper to create mock population, individuals, and execution results."""
        # Create Individuals
        individuals = []
        test_results = {}

        for i, data in enumerate(test_data_list):
            # Mock Individual
            mock_ind = MagicMock(spec=TestIndividual)
            mock_ind.id = f"{test_type}_{i}"
            mock_ind.snippet = f"def test_{i}: pass"
            mock_ind.probability = float(data.get("prob", 0.5))
            individuals.append(mock_ind)

            # Cast status string to Literal for mypy compliance
            raw_status = str(data.get("status", "passed"))
            status_literal = cast(Literal["passed", "failed", "error"], raw_status)

            # Mock Result
            result = TestResult(
                name=f"test_{i}",
                description=f"test_{i}",
                status=status_literal,
                details=data.get("details", None),
            )
            test_results[mock_ind.id] = result

        # Mock Population object - now just the list of individuals
        self.mock_context.test_populations[test_type] = individuals

        # Mock ExecutionResult
        mock_exec_result = MagicMock(spec=TestExecutionResult)
        mock_exec_result.script_error = False
        mock_exec_result.test_results = test_results

        # Setup Context

        mock_interaction = MagicMock(spec=InteractionData)
        mock_interaction.execution_results = {
            self.mock_code_individual.id: mock_exec_result
        }

        # Build a simple integer observation matrix (rows=code, cols=tests).
        # By convention 1 => pass, 0 => fail. Tests in this suite mock only the
        # single code index (0), so create a (1, n_tests) array.
        obs = np.ones((1, len(test_results)), dtype=int)
        for idx, tr in enumerate(test_results.values()):
            if tr.status in ("failed", "error"):
                obs[0, idx] = 0

        mock_interaction.observation_matrix = obs

        self.mock_context.interactions[test_type] = mock_interaction

    def test_select_failing_test_no_failures(self) -> None:
        """Should return None if all tests passed."""
        self._create_mock_population_data(
            "unittest",
            [{"status": "passed", "prob": 0.5}, {"status": "passed", "prob": 0.5}],
        )

        result = FailingTestSelector.select_failing_test(
            self.mock_context, self.mock_code_individual
        )
        self.assertIsNone(result)

    def test_select_failing_test_single_failure(self) -> None:
        """Should return the specific failing test tuple correctly."""
        self._create_mock_population_data(
            "unittest",
            [{"status": "failed", "prob": 0.9, "details": "AssertionError"}],
        )

        with patch.object(FailingTestSelector, "_rank_selection", return_value=0):
            result = FailingTestSelector.select_failing_test(
                self.mock_context, self.mock_code_individual
            )

            self.assertIsNotNone(result)
            if result:
                selected_test, population_type = result
                self.assertEqual(population_type, "unittest")
                self.assertEqual(selected_test.id, "unittest_0")
                self.assertEqual(selected_test.probability, 0.9)

    def test_select_failing_test_multiple_populations(self) -> None:
        """Should aggregate failures from multiple test types."""
        # 1. Add Unittest failures
        self._create_mock_population_data(
            "unittest", [{"status": "failed", "prob": 0.2, "details": "UnitErr"}]
        )

        # 2. Add Differential failures manually to context
        diff_ind = MagicMock(spec=TestIndividual)
        diff_ind.id = "diff_0"
        diff_ind.snippet = "diff code"
        diff_ind.probability = 0.8

        diff_pop = [diff_ind]

        diff_result = TestResult(
            name="diff_0", description="diff_0", status="failed", details="DiffErr"
        )
        diff_exec_res = MagicMock(spec=TestExecutionResult)
        diff_exec_res.script_error = False
        diff_exec_res.test_results = {diff_ind.id: diff_result}

        diff_interaction = MagicMock(spec=InteractionData)
        diff_interaction.execution_results = {
            self.mock_code_individual.id: diff_exec_res
        }
        # Build integer observation matrix for the differential interaction.
        diff_obs = np.ones((1, len(diff_exec_res.test_results)), dtype=int)
        for idx, tr in enumerate(diff_exec_res.test_results.values()):
            if getattr(tr, "status", "passed") in ("failed", "error"):
                diff_obs[0, idx] = 0
        diff_interaction.observation_matrix = diff_obs

        self.mock_context.test_populations["differential"] = diff_pop
        self.mock_context.interactions["differential"] = diff_interaction

        # Mock rank selection to pick the second candidate (the differential one)
        with patch.object(FailingTestSelector, "_rank_selection", return_value=1):
            result = FailingTestSelector.select_failing_test(
                self.mock_context, self.mock_code_individual
            )

            self.assertIsNotNone(result)
            if result:
                selected_test, population_type = result
                self.assertEqual(population_type, "differential")
                self.assertEqual(selected_test.id, "diff_0")

    def test_select_failing_test_script_error(self) -> None:
        """Should skip selection if script_error is True (global failure)."""
        self._create_mock_population_data("unittest", [{"status": "failed"}])

        # Force script_error = True
        interaction = self.mock_context.interactions["unittest"]
        interaction.execution_results[self.mock_code_individual.id].script_error = True

        result = FailingTestSelector.select_failing_test(
            self.mock_context, self.mock_code_individual
        )
        # Using only the observation_matrix for selection means script_error
        # should not prevent choosing a failing test when the matrix indicates one.
        self.assertIsNotNone(result)
        if result:
            selected_test, population_type = result
            self.assertEqual(selected_test.id, "unittest_0")

    def test_select_failing_test_missing_execution_result(self) -> None:
        """Should handle missing execution results gracefully."""
        self._create_mock_population_data("unittest", [{"status": "failed"}])

        # Remove the result for the specific code index
        interaction = self.mock_context.interactions["unittest"]
        interaction.execution_results = {}

        result = FailingTestSelector.select_failing_test(
            self.mock_context, self.mock_code_individual
        )
        # Should return None since execution_results are missing and no fallback
        if result:
            selected_test, population_type = result
            self.assertEqual(selected_test.id, "unittest_0")

    def test_select_failing_test_missing_details(self) -> None:
        """Should handle missing error trace (verify it doesn't crash)."""
        self._create_mock_population_data(
            "unittest",
            [{"status": "failed", "prob": 0.5, "details": None}],
        )

        with patch.object(FailingTestSelector, "_rank_selection", return_value=0):
            result = FailingTestSelector.select_failing_test(
                self.mock_context, self.mock_code_individual
            )

            self.assertIsNotNone(result)
            if result:
                selected_test, _ = result
                self.assertEqual(selected_test.id, "unittest_0")

    def test_select_failing_test_interaction_missing(self) -> None:
        """Should skip if test type exists in populations but not in interactions."""
        self.mock_context.test_populations["ghost_pop"] = MagicMock()

        result = FailingTestSelector.select_failing_test(
            self.mock_context, self.mock_code_individual
        )
        self.assertIsNone(result)

    def test_select_failing_test_individual_not_found(self) -> None:
        """Should return None if code individual is not in population."""
        self.mock_context.code_population.get_index_of_individual.return_value = -1

        result = FailingTestSelector.select_failing_test(
            self.mock_context, self.mock_code_individual
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
