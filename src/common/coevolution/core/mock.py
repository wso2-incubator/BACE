# /path/to/your/project/mock.py
from dataclasses import dataclass

import numpy as np
from loguru import logger

import common.logging_utils as logging_utils

from .interfaces import (
    BaseIndividual,
    BasePopulation,
    BayesianConfig,
    IBeliefInitializer,
    IBeliefUpdater,
    ICodeOperator,
    ICodeTestExecutor,
    IDiscriminationCalculator,
    IFeedbackGenerator,
    IObservationMatrixBuilder,
    IParetoFrontCalculator,
    IProbabilityAssigner,
    ISelectionStrategy,
    ITestBlockBuilder,
    ITestOperator,
    Operations,
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


type ExecutionResults = list[MockUnitTestResult]

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

    def mutate(self, individual: str) -> str:
        logger.trace("MockCodeOperator.mutate called")
        return individual + f" # mutated v{np.random.randint(100)}"

    def crossover(self, parent1: str, parent2: str) -> str:
        logger.trace("MockCodeOperator.crossover called")
        mid = len(parent1) // 2
        return parent1[:mid] + parent2[mid:] + f" # crossover v{np.random.randint(100)}"

    def edit(self, individual: str, feedback: str) -> str:
        logger.trace(f"MockCodeOperator.edit called with feedback: {feedback}")
        return individual + f" # edited based on feedback v{np.random.randint(100)}"


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

    def mutate(self, individual: str) -> str:
        logger.trace("MockTestOperator.mutate called")
        return individual + f" # mutated test v{np.random.randint(100)}"

    def crossover(self, parent1: str, parent2: str) -> str:
        logger.trace("MockTestOperator.crossover called")
        mid = len(parent1) // 2
        return (
            parent1[:mid]
            + parent2[mid:]
            + f" # crossover test v{np.random.randint(100)}"
        )

    def edit(self, individual: str, feedback: str) -> str:
        logger.trace(f"MockTestOperator.edit called with feedback: {feedback}")
        return (
            individual + f" # edited test based on feedback v{np.random.randint(100)}"
        )


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

    def __call__(
        self,
        operation: Operations,
        parent_probs: ParentProbabilities,
        initial_prior: float,
    ) -> float:
        if operation == "initial":
            return initial_prior
        if not parent_probs:
            return initial_prior
        # New individuals inherit the mean probability of their parents
        return float(np.mean(parent_probs))


class MockParetoFrontCalculator(IParetoFrontCalculator):
    """Mocks Pareto front calculation.

    This mock implementation isn't a true Pareto front.
    It just returns the indices of individuals with above-median
    probability OR above-median discrimination.
    """

    def __call__(
        self, probabilities: np.ndarray, discriminations: np.ndarray
    ) -> list[int]:
        logger.trace("MockParetoFrontCalculator called")

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


class MockTestBlockBuilder(ITestBlockBuilder):
    """Mocks rebuilding the test class from snippets."""

    def __call__(self, original_class_str: str, new_method_snippets: list[str]) -> str:
        logger.trace("MockTestBlockBuilder: Rebuilding test class...")
        # Find the class header from the original
        header = original_class_str.split("def ")[0]

        indented_snippets = "\n\n".join([f"    {s}" for s in new_method_snippets])
        return header + indented_snippets


class MockFeedbackGenerator(IFeedbackGenerator[BaseIndividual]):
    """Mocks generating feedback for the 'edit' operator."""

    def __call__(
        self,
        observation_matrix: np.ndarray,
        execution_results: ExecutionResults,
        other_population: "BasePopulation[BaseIndividual]",
        individual_idx: int,
    ) -> str:
        # Find the first test this code individual failed
        failures = np.where(observation_matrix[individual_idx] == 0)[0]
        if len(failures) > 0:
            failed_test_id = other_population.ids[failures[0]]
            return f"Your code failed test: {failed_test_id}"
        return "Your code passed all tests, but try to optimize it."


# --- Mock Execution & Belief Updaters ---


class MockCodeTestExecutor(ICodeTestExecutor):
    """
    Mocks the execution of all code against all tests (generated, public, private).
    Returns a dictionary of observation matrices.
    """

    def __call__(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
        sandbox: Sandbox,
    ) -> ExecutionResults:
        logger.debug("MockCodeTestExecutor: 'Running' all tests...")

        execution_results: ExecutionResults = []
        num_tests = len(test_population)

        for code_ind in code_population:
            logger.trace(f"Executing generated tests for Code ID {code_ind.id}")
            is_passing = np.random.rand(num_tests) < code_ind.probability
            error_messages = [
                "" if pass_ else "Mock failure message" for pass_ in is_passing
            ]
            mock_result: MockUnitTestResult = MockUnitTestResult(
                is_passing=is_passing.tolist(), error_messages=error_messages
            )
            execution_results.append(mock_result)

        return execution_results


class MockObservationMatrixBuilder(IObservationMatrixBuilder):
    """
    Mocks the building of the single observation matrix for belief updating.
    As requested, it combines "generated" and "public" test results.
    """

    def __call__(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
        execution_results: ExecutionResults,
    ) -> np.ndarray:
        logger.trace("MockObservationMatrixBuilder: Combining results...")
        num_code = len(code_population)
        num_tests = len(test_population)
        combined_matrix = np.zeros((num_code, num_tests), dtype=int)
        for i, result in enumerate(execution_results):
            for j in range(num_tests):
                combined_matrix[i, j] = 1 if result.is_passing[j] else 0
        return combined_matrix


class MockDiscriminationCalculator(IDiscriminationCalculator):
    """Mocks the calculation of test discrimination scores."""

    def __call__(self, observation_matrix: np.ndarray) -> np.ndarray:
        logger.trace("MockDiscriminationCalculator: Calculating scores...")

        # A simple mock: discrimination is the variance of outcomes for that test.
        # High variance (mix of pass/fail) means it's discriminating.
        # Low variance (all pass or all fail) is bad.
        if observation_matrix.shape[0] == 0:  # No code individuals
            return np.zeros(observation_matrix.shape[1])

        # Calculate variance along the 'code' axis (axis 0)
        scores = np.var(observation_matrix, axis=0)
        return np.asarray(scores)


class MockBeliefInitializer(IBeliefInitializer):
    """Mocks the initialization of belief arrays."""

    def __call__(
        self,
        population_size: int,
        initial_probability: float,
    ) -> np.ndarray:
        logger.trace("MockBeliefInitializer called")
        return np.full(population_size, initial_probability)


class MockCodeBeliefUpdater(IBeliefUpdater):
    """Mocks the Bayesian update for the Code Population."""

    def __call__(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        logger.trace("MockCodeBeliefUpdater called")

        # Mock logic: if a code passes > 75% of tests,
        # its probability increases, otherwise it decreases.
        if observation_matrix.shape[1] == 0:  # No tests
            return prior_code_probs

        pass_rates = np.mean(observation_matrix, axis=1)

        # Simple update rule
        adjustment = (pass_rates - 0.75) * config.learning_rate
        new_probs = prior_code_probs + adjustment

        # Clip to ensure probabilities stay in [0.01, 0.99]
        return np.asarray(np.clip(new_probs, 0.01, 0.99))


class MockTestBeliefUpdater(IBeliefUpdater):
    """Mocks the Bayesian update for the Test Population."""

    def __call__(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        logger.trace("MockTestBeliefUpdater called")

        # Mock logic: if a test fails > 50% of code,
        # its probability (of being correct) increases, otherwise decreases.
        if observation_matrix.shape[0] == 0:  # No code
            return prior_test_probs

        fail_rates = 1.0 - np.mean(observation_matrix, axis=0)

        # Simple update rule
        adjustment = (fail_rates - 0.5) * config.learning_rate
        new_probs = prior_test_probs + adjustment

        # Clip to ensure probabilities stay in [0.01, 0.99]
        return np.asarray(np.clip(new_probs, 0.01, 0.99))
