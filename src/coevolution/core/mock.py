# /path/to/your/project/mock.py
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from loguru import logger

from coevolution.utils import logging as logging_utils

from .individual import CodeIndividual, TestIndividual
from .interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    BaseIndividual,
    BasePopulation,
    BayesianConfig,
    CoevolutionContext,
    EvaluationResult,
    ExecutionResults,
    IBeliefUpdater,
    IEliteSelectionStrategy,
    IExecutionSystem,
    IInteractionLedger,
    InteractionData,
    IParentSelectionStrategy,
    IProbabilityAssigner,
    Operation,
    ParentProbabilities,
    PopulationConfig,
    Problem,
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


class MockCodeOperator:
    """
    Mock code operator implementing the new IOperator contract.

    execute(context) -> list[CodeIndividual]  — fully self-contained.
    """

    _INITIAL_PRIOR = 0.5

    def operation_name(self) -> str:
        # Mock supports all three operations; we pick mutation for simplicity
        return OPERATION_MUTATION

    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        """
        Select a parent, generate a mutated snippet, return one CodeIndividual.
        """
        logger.trace("MockCodeOperator.execute called")

        code_pop = context.code_population
        if code_pop.size == 0:
            return []

        # Select a parent by probability
        probs = code_pop.probabilities
        p_normalized = probs / probs.sum()
        parent_idx = int(np.random.choice(len(probs), p=p_normalized))
        parent = code_pop[parent_idx]

        # Generate offspring snippet
        op = np.random.choice([OPERATION_MUTATION, OPERATION_CROSSOVER, OPERATION_EDIT])
        if op == OPERATION_MUTATION:
            snippet = f"# mutated code snippet v{np.random.randint(100)}"
        elif op == OPERATION_CROSSOVER:
            snippet = f"# crossover code snippet v{np.random.randint(100)}"
        else:
            snippet = f"# edited code snippet v{np.random.randint(100)}"

        prob = float(np.mean([parent.probability]))

        return [
            CodeIndividual(
                snippet=snippet,
                probability=prob,
                creation_op=op,
                generation_born=code_pop.generation + 1,
                parents={"code": [parent.id], "test": []},
            )
        ]


class MockTestOperator:
    """
    Mock test operator implementing the new IOperator contract.

    execute(context) -> list[TestIndividual]  — fully self-contained.
    """

    def __init__(self, test_type: str = "unittest") -> None:
        self.test_type = test_type

    def operation_name(self) -> str:
        return OPERATION_MUTATION

    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        """
        Select a parent from the test population, generate an offspring.
        """
        logger.trace(f"MockTestOperator({self.test_type}).execute called")

        test_pop = context.test_populations.get(self.test_type)
        if test_pop is None or test_pop.size == 0:
            # No population yet (generation 0 start) — generate a fresh test
            snippet = f"def test_mock_{np.random.randint(1000)}(): assert True"
            return [
                TestIndividual(
                    snippet=snippet,
                    probability=0.5,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                )
            ]

        probs = test_pop.probabilities
        p_normalized = probs / probs.sum()
        parent_idx = int(np.random.choice(len(probs), p=p_normalized))
        parent = test_pop[parent_idx]

        op = np.random.choice([OPERATION_MUTATION, OPERATION_CROSSOVER, OPERATION_EDIT])
        if op == OPERATION_MUTATION:
            snippet = f"def test_mutated_{np.random.randint(1000)}(): pass  # mutated"
        elif op == OPERATION_CROSSOVER:
            snippet = f"def test_crossover_{np.random.randint(1000)}(): pass  # crossover"
        else:
            snippet = f"def test_edited_{np.random.randint(1000)}(): pass  # edited"

        prob = float(np.mean([parent.probability]))

        return [
            TestIndividual(
                snippet=snippet,
                probability=prob,
                creation_op=op,
                generation_born=test_pop.generation + 1,
                parents={"code": [], "test": [parent.id]},
            )
        ]


# --- Mock Initializers ---


class MockCodeInitializer:
    """
    Implements IPopulationInitializer[CodeIndividual].
    Creates a fixed-size initial code population.
    """

    def __init__(self, population_size: int = 10, initial_prior: float = 0.5) -> None:
        self.population_size = population_size
        self.initial_prior = initial_prior

    def initialize(self, problem: Problem) -> list[CodeIndividual]:
        logger.debug(
            f"MockCodeInitializer: Creating {self.population_size} initial code individuals"
        )
        starter_code = problem.starter_code or ""
        return [
            CodeIndividual(
                snippet=starter_code + f"    pass  # mock code {i}",
                probability=self.initial_prior,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
                metadata={"mock_index": i},
            )
            for i in range(self.population_size)
        ]


class MockTestInitializer:
    """
    Implements IPopulationInitializer[TestIndividual].
    Creates a fixed-size initial test population, or returns [] for empty starts.
    """

    def __init__(self, population_size: int = 10, initial_prior: float = 0.5) -> None:
        self.population_size = population_size
        self.initial_prior = initial_prior

    def initialize(self, problem: Problem) -> list[TestIndividual]:
        if self.population_size == 0:
            logger.debug("MockTestInitializer: population_size=0, returning empty list")
            return []

        logger.debug(
            f"MockTestInitializer: Creating {self.population_size} initial test individuals"
        )
        return [
            TestIndividual(
                snippet=f"def test_mock_case_{i}():\n    assert sort_array([{i}, {i - 1}]) == [{i - 1}, {i}]",
                probability=self.initial_prior,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={"code": [], "test": []},
                metadata={"mock_index": i, "test_name": f"test_mock_case_{i}"},
            )
            for i in range(self.population_size)
        ]


# --- Mock Strategies & Helpers ---


class MockParentSelectionStrategy[T: BaseIndividual](IParentSelectionStrategy[T]):
    """Mock implementation of parent selection using roulette wheel."""

    def select_parents(
        self,
        population: BasePopulation[T],
        count: int,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        probabilities = population.probabilities
        p_normalized = probabilities / probabilities.sum()

        selected_indices = np.random.choice(
            len(probabilities),
            size=count,
            p=p_normalized,
            replace=True,
        )

        return [population[int(idx)] for idx in selected_indices]


# Legacy alias
MockSelectionStrategy = MockParentSelectionStrategy


class MockProbabilityAssigner(IProbabilityAssigner):
    """Mocks the assignment of probabilities to new offspring."""

    initial_prior: float

    def __init__(self, initial_prior: float = 0.5) -> None:
        self.initial_prior = initial_prior

    def assign_probability(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
    ) -> float:
        if operation == OPERATION_INITIAL or not parent_probs:
            return self.initial_prior
        return float(np.mean(parent_probs))


class MockEliteSelectionStrategy[T: BaseIndividual](IEliteSelectionStrategy[T]):
    """Mock implementation of elite selection using simple top-k by probability."""

    def select_elites(
        self,
        population: BasePopulation[T],
        population_config: PopulationConfig,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        if population.size == 0:
            logger.debug(
                "MockEliteSelectionStrategy: Population is empty, returning empty elites list"
            )
            return []

        num_elites = int(population.size * population_config.elitism_rate)
        probabilities = population.probabilities
        sorted_indices = np.argsort(probabilities)[::-1]
        elite_indices = sorted_indices[:num_elites]
        elites = [population[int(idx)] for idx in elite_indices]

        logger.debug(
            f"MockEliteSelectionStrategy: Selected {len(elites)} elites "
            f"({population_config.elitism_rate:.1%} of {population.size}) from population"
        )
        return elites


# Legacy alias
MockEliteSelector = MockEliteSelectionStrategy


# --- Mock Execution & Belief Updaters ---


# ============================================
# === EXECUTION SYSTEM MOCK
# ============================================


class MockExecutionSystem(IExecutionSystem):
    """
    Mock implementation combining test execution and observation matrix building.
    """

    def execute_tests(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
    ) -> InteractionData:
        """Execute all code against all tests and return atomic InteractionData."""
        num_code = len(code_population)
        num_tests = len(test_population)

        if num_code == 0 or num_tests == 0:
            logger.debug(
                f"MockExecutionSystem: Empty population detected (code={num_code}, "
                f"test={num_tests}), returning empty InteractionData"
            )
            empty_matrix = np.zeros((num_code, num_tests), dtype=int)
            return InteractionData(
                execution_results=ExecutionResults(results={}),
                observation_matrix=empty_matrix,
            )

        logger.debug("MockExecutionSystem: 'Running' all tests (mock)...")

        execution_results: ExecutionResults = ExecutionResults(results={})
        observation_matrix = np.zeros((num_code, num_tests), dtype=int)

        for i, code_ind in enumerate(code_population):
            logger.trace(f"Executing generated tests for Code ID {code_ind.id}")
            is_passing = np.random.rand(num_tests) < code_ind.probability

            code_results: dict[str, EvaluationResult] = {}
            for j, test_ind in enumerate(test_population):
                passed = bool(is_passing[j])
                observation_matrix[i, j] = 1 if passed else 0
                error_log = None if passed else "Mock failure message"
                status = "passed" if passed else "failed"
                status_lit = cast(Literal["passed", "failed", "error"], status)
                code_results[test_ind.id] = EvaluationResult(
                    error_log=error_log, status=status_lit
                )

            execution_results.results[code_ind.id] = code_results

        return InteractionData(
            execution_results=execution_results, observation_matrix=observation_matrix
        )

    def build_observation_matrix(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
        execution_results: ExecutionResults,
    ) -> np.ndarray:
        """Build observation matrix from ID-keyed ExecutionResults aligned to populations."""
        logger.trace(
            "MockExecutionSystem: Building observation matrix from ExecutionResults..."
        )
        num_code = len(code_population)
        num_tests = len(test_population)

        if num_code == 0 or num_tests == 0:
            return np.zeros((num_code, num_tests), dtype=int)

        combined_matrix = np.zeros((num_code, num_tests), dtype=int)
        test_id_to_idx = {t.id: idx for idx, t in enumerate(test_population)}

        for row_idx, code_ind in enumerate(code_population):
            code_id = code_ind.id
            code_results = execution_results.get(code_id)
            if code_results is None:
                continue
            for test_id, evaluation_result in code_results.items():
                col_idx = test_id_to_idx.get(test_id)
                if col_idx is None:
                    continue
                combined_matrix[row_idx, col_idx] = (
                    1 if evaluation_result.status == "passed" else 0
                )

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
    if observation_matrix.shape[1] == 0:
        return prior_probs_self

    success_rates = np.mean(observation_matrix, axis=1)
    adjustment = (success_rates - target_threshold) * config.learning_rate
    new_probs = prior_probs_self + adjustment
    return np.asarray(np.clip(new_probs, 0.01, 0.99))


class MockBeliefUpdater(IBeliefUpdater):
    """Mock belief updater for testing purposes."""

    def update_code_beliefs(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        code_update_mask_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        if observation_matrix.shape[1] == 0 or observation_matrix.shape[0] == 0:
            logger.trace(
                f"MockBeliefUpdater: Empty observation matrix {observation_matrix.shape}, "
                "returning prior_code_probs unchanged (identity operation)"
            )
            return prior_code_probs.copy()

        logger.trace("MockBeliefUpdater: Updating code beliefs")

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
        if observation_matrix.shape[0] == 0 or observation_matrix.shape[1] == 0:
            logger.trace(
                f"MockBeliefUpdater: Empty observation matrix {observation_matrix.shape}, "
                "returning prior_test_probs unchanged (identity operation)"
            )
            return prior_test_probs.copy()

        logger.trace("MockBeliefUpdater: Updating test beliefs")

        mask_arr = np.asarray(test_update_mask_matrix, dtype=bool)
        if mask_arr.shape != observation_matrix.shape:
            raise ValueError("Mask shape must match observation matrix shape")

        mask_arr = mask_arr.T
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


class MockInteractionLedger(IInteractionLedger):
    """
    Mock ledger that always permits updates (returns all 1s).
    Use this to bypass history checks in unit tests.
    """

    def __init__(self) -> None:
        self.committed_history: list[tuple[str, str, int]] = []

    def get_new_interaction_mask(
        self, code_ids: list[str], test_ids: list[str], test_type: str, target: str
    ) -> np.ndarray:
        return np.ones((len(code_ids), len(test_ids)), dtype=int)

    def commit_interactions(
        self,
        code_ids: list[str],
        test_ids: list[str],
        test_type: str,
        target: str,
        mask: np.ndarray,
    ) -> None:
        self.committed_history.append((test_type, target, np.sum(mask)))


# Factory for injection
def mock_ledger_factory() -> IInteractionLedger:
    return MockInteractionLedger()
