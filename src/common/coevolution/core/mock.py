# /path/to/your/project/mock.py
from dataclasses import dataclass

import numpy as np
from loguru import logger

from common.coevolution import logging_utils

from .interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    BaseIndividual,
    BasePopulation,
    BayesianConfig,
    ExecutionResults,
    IBayesianSystem,
    ICodeOperator,
    IExecutionSystem,
    IFeedbackGenerator,
    IPareto,
    IProbabilityAssigner,
    ISelectionStrategy,
    ITestBlockRebuilder,
    ITestOperator,
    Operation,
    ParentProbabilities,
    Problem,
    Sandbox,
    Test,
)
from .population import CodePopulation, TestPopulation


@dataclass
class MockUnitTestResult:
    """Mock implementation of UnitTestResult for testing purposes."""

    is_passing: list[bool]
    error_messages: list[str]


# setup logging
logging_utils.setup_logging(
    console_level="DEBUG",
    file_level="TRACE",
    log_file_base_name="mock_orchestrator.log",
)

# ============================================
# === HELPER FUNCTIONS
# ============================================


def get_mock_problem() -> Problem:
    """Creates a mock Problem dataclass instance."""
    return Problem(
        question_title="Mock Problem: Sort an Array",
        question_content="Write a function `sort_array(arr)` that sorts an array.",
        question_id="MOCK-001",
        starter_code="def sort_array(arr):\n    # Your code here\n    return arr\n",
        public_test_cases=[
            Test(input="[3, 1, 2]", output="[1, 2, 3]"),
            Test(input="[5, 5, 1]", output="[1, 5, 5]"),
        ],
        private_test_cases=[
            Test(input="[]", output="[]"),
            Test(input="[1]", output="[1]"),
            Test(input="[6, 5, 4, 3, 2, 1]", output="[1, 2, 3, 4, 5, 6]"),
        ],
    )


# ============================================
# === MOCK INTERFACE IMPLEMENTATIONS
# ============================================

# --- Mock Operators ---


class MockCodeOperator(ICodeOperator):
    """Combines mock code initialization and genetic operations."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def create_initial_snippets(self, population_size: int) -> list[str]:
        logger.debug(f"MockCodeOperator: Creating {population_size} code snippets...")
        return [
            self.problem.starter_code + f"    pass # mock code {i}"
            for i in range(population_size)
        ]

    def apply(
        self,
        operation: Operation,
        parent1: str,
        parent2: str | None = None,
        feedback: str | None = None,
    ) -> str:
        """Apply a genetic operation to code snippets."""
        logger.trace(f"MockCodeOperator.apply called with operation={operation}")

        if operation == OPERATION_MUTATION:
            return parent1 + f" # mutated v{np.random.randint(100)}"
        elif operation == OPERATION_CROSSOVER:
            if parent2 is None:
                raise ValueError("Crossover requires parent2")
            mid = len(parent1) // 2
            return (
                parent1[:mid]
                + parent2[mid:]
                + f" # crossover v{np.random.randint(100)}"
            )
        elif operation == OPERATION_EDIT:
            if feedback is None:
                raise ValueError("Edit requires feedback")
            logger.trace(f"MockCodeOperator.apply with feedback: {feedback}")
            return parent1 + f" # edited based on feedback v{np.random.randint(100)}"
        else:
            # For any other operation (including custom ones like "det"), just return parent1
            return parent1 + f" # {operation} v{np.random.randint(100)}"


class MockTestOperator(ITestOperator):
    """Mocks test genetic operations."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def create_initial_snippets(self, population_size: int) -> tuple[list[str], str]:
        logger.debug(
            f"MockTestInitializer: Creating {population_size} test snippets..."
        )

        snippets = []
        for i in range(population_size):
            test_name = f"test_mock_case_{i}"
            snippet = f"def {test_name}(self):\n        assert sort_array([{i}, {i - 1}]) == [{i - 1}, {i}]"
            snippets.append(snippet)

        # Create a mock class block
        class_header = "import unittest\nfrom a_mock_solution import sort_array\n\nclass GeneratedTests(unittest.TestCase):\n"
        indented_snippets = "\n\n".join([f"    {s}" for s in snippets])
        full_class_block = class_header + indented_snippets

        return snippets, full_class_block

    def apply(
        self,
        operation: Operation,
        parent1: str,
        parent2: str | None = None,
        feedback: str | None = None,
    ) -> str:
        """Apply a genetic operation to test snippets."""
        logger.trace(f"MockTestOperator.apply called with operation={operation}")

        if operation == OPERATION_MUTATION:
            return parent1 + f" # mutated test v{np.random.randint(100)}"
        elif operation == OPERATION_CROSSOVER:
            if parent2 is None:
                raise ValueError("Crossover requires parent2")
            mid = len(parent1) // 2
            return (
                parent1[:mid]
                + parent2[mid:]
                + f" # crossover test v{np.random.randint(100)}"
            )
        elif operation == OPERATION_EDIT:
            if feedback is None:
                raise ValueError("Edit requires feedback")
            logger.trace(f"MockTestOperator.apply with feedback: {feedback}")
            return (
                parent1 + f" # edited test based on feedback v{np.random.randint(100)}"
            )
        else:
            # For any other operation (including custom ones like "det"), just return parent1
            return parent1 + f" # {operation} test v{np.random.randint(100)}"


# --- Mock Strategies & Helpers ---


class MockSelectionStrategy(ISelectionStrategy):
    """Mocks parent selection."""

    def select(self, probabilities: np.ndarray) -> int:
        # Simple roulette wheel selection
        p_normalized = probabilities / probabilities.sum()
        return np.random.choice(len(probabilities), p=p_normalized)

    def select_parents(self, probabilities: np.ndarray) -> tuple[int, int]:
        if len(probabilities) < 2:
            raise ValueError("Population too small to select two parents.")
        p1 = self.select(probabilities)
        p2 = self.select(probabilities)
        while p1 == p2:  # Ensure two *different* parents
            p2 = self.select(probabilities)
        return p1, p2


class MockProbabilityAssigner(IProbabilityAssigner):
    """Mocks the assignment of probabilities to new offspring."""

    def assign_probability(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
        initial_prior: float,
    ) -> float:
        if operation == OPERATION_INITIAL:
            return initial_prior
        if not parent_probs:
            return initial_prior
        # New individuals inherit the mean probability of their parents
        return float(np.mean(parent_probs))


class MockPareto(IPareto):
    """
    Mocks multi-objective optimization for test population.

    This mock implementation calculates simple variance-based discrimination
    and returns individuals with above-median probability OR discrimination.
    """

    def calculate_discrimination(self, observation_matrix: np.ndarray) -> np.ndarray:
        """Calculate variance across code individuals for each test."""
        logger.trace("MockPareto: Calculating discrimination scores...")
        # Each column is a test; variance shows how well it distinguishes code
        discriminations: np.ndarray = np.var(observation_matrix, axis=0)
        return discriminations

    def calculate_pareto_front(
        self, probabilities: np.ndarray, discriminations: np.ndarray
    ) -> list[int]:
        """Find Pareto front (individuals with good prob OR good discrimination)."""
        logger.trace("MockPareto: Calculating Pareto front...")

        # Handle NaNs in discrimination (e.g., first generation)
        if np.all(np.isnan(discriminations)):
            # Fall back to just probability
            median_prob = np.median(probabilities)
            indices = np.where(probabilities >= median_prob)[0]
            return list(indices)

        median_prob = np.median(probabilities)
        median_disc = np.nanmedian(discriminations)

        # Get indices for "good" individuals
        good_prob = set(np.where(probabilities >= median_prob)[0])
        good_disc = set(
            np.where(np.nan_to_num(discriminations, nan=-1) >= median_disc)[0]
        )

        # The "front" is the union of these two sets
        front_indices = list(good_prob.union(good_disc))
        logger.trace(f"Mock Pareto front size: {len(front_indices)}")
        return front_indices

    def get_pareto_indices(
        self, probabilities: np.ndarray, observation_matrix: np.ndarray
    ) -> list[int]:
        """Public API: Calculate discriminations and return Pareto front indices."""
        discriminations = self.calculate_discrimination(observation_matrix)
        return self.calculate_pareto_front(probabilities, discriminations)


class MockTestBlockRebuilder(ITestBlockRebuilder):
    """Mocks rebuilding the test class from snippets."""

    def rebuild_test_block(
        self, original_class_str: str, new_method_snippets: list[str]
    ) -> str:
        logger.trace("MockTestBlockRebuilder: Rebuilding test class...")
        # Find the class header from the original
        header = original_class_str.split("def ")[0]

        indented_snippets = "\n\n".join([f"    {s}" for s in new_method_snippets])
        return header + indented_snippets


class MockFeedbackGenerator(IFeedbackGenerator[BaseIndividual]):
    """Mocks generating feedback for the 'edit' operator."""

    def generate_feedback(
        self,
        observation_matrix: np.ndarray,
        execution_results: ExecutionResults,
        other_population: "BasePopulation[BaseIndividual]",
        individual_idx: int,
    ) -> str:
        # The observation_matrix orientation depends on the caller:
        # - When generating feedback for a code individual: rows=code, cols=tests
        #   so individual_idx indexes rows.
        # - When generating feedback for a test individual: rows=code, cols=tests
        #   so individual_idx indexes columns.
        n_rows, n_cols = observation_matrix.shape

        if individual_idx < n_rows:
            # Code individual: look across tests in this row
            failures = np.where(observation_matrix[individual_idx] == 0)[0]
            if len(failures) > 0:
                failed_test_id = other_population.ids[failures[0]]
                return f"Your code failed test: {failed_test_id}"
            return "Your code passed all tests, but try to optimize it."

        # Otherwise treat as a test individual (index into columns)
        if individual_idx < n_cols:
            failures = np.where(observation_matrix[:, individual_idx] == 0)[0]
            if len(failures) > 0:
                failed_code_id = other_population.ids[failures[0]]
                return f"Your test failed code: {failed_code_id}"
            return "Your test passed all current code samples; consider adding harder cases."

        # Index out of bounds: fall back to a neutral message
        return "No feedback available for the selected individual."


# --- Mock Execution & Belief Updaters ---


# ============================================
# === EXECUTION SYSTEM MOCK
# ============================================


class MockExecutionSystem(IExecutionSystem):
    """
    Mock implementation combining test execution and observation matrix building.
    Implements both ICodeTestExecutor and IObservationMatrixBuilder protocols.
    """

    def execute_tests(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
        sandbox: Sandbox,
    ) -> ExecutionResults:
        """Execute all code against all tests."""
        logger.debug("MockExecutionSystem: 'Running' all tests...")

        execution_results: dict[int, MockUnitTestResult] = {}
        num_tests = len(test_population)

        for code_idx, code_ind in enumerate(code_population):
            logger.trace(f"Executing generated tests for Code ID {code_ind.id}")
            is_passing = np.random.rand(num_tests) < code_ind.probability
            error_messages = [
                "" if pass_ else "Mock failure message" for pass_ in is_passing
            ]
            mock_result: MockUnitTestResult = MockUnitTestResult(
                is_passing=is_passing.tolist(), error_messages=error_messages
            )
            execution_results[code_idx] = mock_result

        return execution_results

    def build_observation_matrix(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
        execution_results: ExecutionResults,
    ) -> np.ndarray:
        """Build observation matrix from execution results."""
        logger.trace("MockExecutionSystem: Building observation matrix...")
        num_code = len(code_population)
        num_tests = len(test_population)
        combined_matrix = np.zeros((num_code, num_tests), dtype=int)
        for code_idx, result in execution_results.items():
            for j in range(num_tests):
                combined_matrix[code_idx, j] = 1 if result.is_passing[j] else 0
        return combined_matrix


# ============================================
# === BAYESIAN SYSTEM MOCKS
# ============================================


def _generic_belief_update(
    prior_probs_self: np.ndarray,
    prior_probs_other: np.ndarray,
    observation_matrix: np.ndarray,
    config: BayesianConfig,
    target_threshold: float,
) -> np.ndarray:
    """
    Generic Bayesian belief update logic shared by both code and test systems.

    Args:
        prior_probs_self: Prior probabilities for the population being updated.
        prior_probs_other: Prior probabilities for the opposing population.
        observation_matrix: Observation matrix with rows=self, columns=other.
        config: Bayesian configuration.
        target_threshold: Threshold for determining if performance is "good".

    Returns:
        Updated posterior probabilities for the population being updated.
    """
    # New: support an explicit mask to select which (self,other) cells are
    # considered "new evidence" for this update. The mask should have the
    # same shape as observation_matrix and be boolean. When mask is None we
    # consider all cells (legacy behaviour).
    #
    # NOTE: We implement mask-aware averaging in a numerically stable,
    # vectorized way. Rows with zero included cells will be left with their
    # prior unchanged (no new evidence).
    #
    # If no opposing population, return priors unchanged
    if observation_matrix.shape[1] == 0:
        return prior_probs_self

    # If a mask was attached to the matrix, it will be expected to be stored
    # as an attribute by the caller; however, to keep this helper generic we
    # expect callers to pass an explicit mask via a keyword parameter in the
    # call site. If no mask is provided, behave as before.
    # (Backward compatibility: callers that do not pass a mask will still
    # work with the older behaviour.)
    #
    # NOTE: The mask handling is implemented by callers passing an already
    # filtered/selected observation_matrix and corresponding mask. This helper
    # accepts a mask param when invoked; to keep the function signature
    # stable we will not add a new parameter here — callers should wrap this
    # helper with mask handling. For simplicity in our mock implementations
    # below we will compute success rates using the mask when provided.

    # Calculate success rate for each individual in self population
    success_rates = np.mean(observation_matrix, axis=1)

    # Simple update rule: adjust based on how far from threshold
    adjustment = (success_rates - target_threshold) * config.learning_rate
    new_probs = prior_probs_self + adjustment

    # Clip to ensure probabilities stay in [0.01, 0.99]
    return np.asarray(np.clip(new_probs, 0.01, 0.99))


class MockBayesianSystem(IBayesianSystem):
    """Single mock Bayesian system used for both code and test updates.

    This class centralizes the logic previously split between separate
    code/test implementations. It provides both `update_code_beliefs` and
    `update_test_beliefs` as required by the protocol, and exposes the
    mask generation helpers (code/test/backwards-compatible) plus a private
    helper `_get_update_mask_generation`.
    """

    def initialize_beliefs(
        self,
        population_size: int,
        initial_probability: float,
    ) -> np.ndarray:
        logger.trace("MockBayesianSystem: Initializing beliefs")
        return np.full(population_size, initial_probability)

    def update_code_beliefs(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        code_update_mask_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        logger.trace("MockBayesianSystem: Updating code beliefs")

        mask_arr = np.asarray(code_update_mask_matrix, dtype=bool)
        if mask_arr.shape != observation_matrix.shape:
            raise ValueError("Mask shape must match observation_matrix shape")

        included_counts = mask_arr.sum(axis=1)
        success_rates = np.zeros_like(included_counts, dtype=float)
        nonzero = included_counts > 0
        if nonzero.any():
            masked_sums = (observation_matrix * mask_arr).sum(axis=1)
            success_rates[nonzero] = masked_sums[nonzero] / included_counts[nonzero]

        new_probs = prior_code_probs.astype(float).copy()
        if nonzero.any():
            adjustment = (success_rates[nonzero] - 0.75) * config.learning_rate
            new_probs[nonzero] = np.clip(
                prior_code_probs[nonzero] + adjustment, 0.01, 0.99
            )

        return new_probs

    def update_test_beliefs(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        test_update_mask_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        logger.trace("MockBayesianSystem: Updating test beliefs")

        mask_arr = np.asarray(test_update_mask_matrix, dtype=bool)
        if mask_arr.shape != observation_matrix.shape:
            raise ValueError("Mask shape must match observation matrix shape")

        mask_arr = mask_arr.T  # Transpose for test view
        transposed_matrix = observation_matrix.T
        failure_matrix = 1 - transposed_matrix

        included_counts = mask_arr.sum(axis=1)
        success_rates = np.zeros_like(included_counts, dtype=float)
        nonzero = included_counts > 0
        if nonzero.any():
            masked_sums = (failure_matrix * mask_arr).sum(axis=1)
            success_rates[nonzero] = masked_sums[nonzero] / included_counts[nonzero]

        new_probs = prior_test_probs.astype(float).copy()
        if nonzero.any():
            adjustment = (success_rates[nonzero] - 0.50) * config.learning_rate
            new_probs[nonzero] = np.clip(
                prior_test_probs[nonzero] + adjustment, 0.01, 0.99
            )

        return new_probs

    # Mask generation helpers
    def get_code_update_mask_generation(
        self,
        updating_ind_born_generations: list[int] | np.ndarray,
        other_ind_born_generations: list[int] | np.ndarray,
        current_generation: int,
    ) -> np.ndarray:
        return self._get_update_mask_generation(
            updating_ind_born_generations,
            other_ind_born_generations,
            current_generation,
        )

    def get_test_update_mask_generation(
        self,
        updating_ind_born_generations: list[int] | np.ndarray,
        other_ind_born_generations: list[int] | np.ndarray,
        current_generation: int,
    ) -> np.ndarray:
        return self._get_update_mask_generation(
            updating_ind_born_generations,
            other_ind_born_generations,
            current_generation,
        ).T

    def _get_update_mask_generation(
        self,
        updating_ind_born_generations: list[int] | np.ndarray,
        other_ind_born_generations: list[int] | np.ndarray,
        current_generation: int,
    ) -> np.ndarray:
        rows = len(updating_ind_born_generations)
        cols = len(other_ind_born_generations)
        mask = np.zeros((rows, cols), dtype=bool)

        upd_gens = np.asarray(updating_ind_born_generations, dtype=int)
        oth_gens = np.asarray(other_ind_born_generations, dtype=int)

        new_rows = upd_gens == int(current_generation)
        if new_rows.any():
            mask[new_rows, :] = True

        old_rows = ~new_rows
        if old_rows.any():
            cols_this_gen = oth_gens == int(current_generation)
            if cols_this_gen.any():
                mask[old_rows[:, None] & cols_this_gen[None, :]] = True

        return mask


# Backwards compatible aliases
MockCodeBayesianSystem = MockBayesianSystem
MockTestBayesianSystem = MockBayesianSystem


class MockDatasetTestBlockBuilder:
    """
    Mock builder for creating test class blocks from dataset test cases.

    Builds simple unittest test class blocks for testing purposes.
    Does NOT create TestPopulation - just returns the test class block string.
    """

    def build_test_class_block(self, test_cases: list[Test], starter_code: str) -> str:
        """
        Build a simple unittest test class block from dataset test cases.

        Args:
            test_cases: List of Test objects from the dataset
            starter_code: The starter code for the problem (ignored in mock)

        Returns:
            A complete unittest test class block as a string
        """
        snippets: list[str] = []

        for i, test_case in enumerate(test_cases):
            # Create a simple mock test snippet
            snippet = f"def test_fixed_{i}(self):\n        assert mock_solution({test_case.input!r}) == {test_case.output!r}"
            snippets.append(snippet)

        # Create a simple test class block
        test_class_block = (
            "import unittest\n"
            "from mock_solution import mock_solution\n\n"
            "class FixedTests(unittest.TestCase):\n"
        )
        test_class_block += "\n\n".join([f"    {s}" for s in snippets])

        logger.info(
            f"MockDatasetTestBlockBuilder: Built test class block with {len(snippets)} test methods"
        )

        return test_class_block
