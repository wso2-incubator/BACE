# /path/to/your/project/mock.py
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from loguru import logger

from common.coevolution import logging_utils

from .interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    BaseIndividual,
    BaseOperatorInput,
    BasePopulation,
    BayesianConfig,
    CoevolutionContext,
    ExecutionResult,
    ExecutionResults,
    IBayesianSystem,
    IEliteSelectionStrategy,
    IExecutionSystem,
    InitialInput,
    InteractionData,
    IOperator,
    IParentSelectionStrategy,
    IProbabilityAssigner,
    ITestBlockRebuilder,
    Operation,
    OperatorOutput,
    OperatorResult,
    ParentProbabilities,
    PopulationConfig,
    Problem,
    Test,
    TestResult,
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


class MockCodeOperator(IOperator):
    """Combines mock code initialization and genetic operations."""

    def __init__(self, initial_prior: float = 0.5) -> None:
        self.initial_prior = initial_prior

    def supported_operations(self) -> set[str]:
        """Return the set of operations this operator supports."""
        return {OPERATION_MUTATION, OPERATION_CROSSOVER, OPERATION_EDIT}

    def generate_initial_snippets(
        self, input_dto: InitialInput
    ) -> tuple[OperatorOutput, str | None]:
        """Generate initial code snippets (generation 0). Returns (OperatorOutput, None) for code."""
        population_size = input_dto.population_size
        starter_code = input_dto.starter_code or ""
        logger.debug(f"MockCodeOperator: Generating {population_size} code snippets...")

        results = []
        for i in range(population_size):
            snippet = starter_code + f"    pass # mock code {i}"
            result = OperatorResult(
                snippet=snippet,
                metadata={"mock_index": i},
            )
            results.append(result)

        return OperatorOutput(results=results), None

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        """Apply a genetic operation to code snippets."""
        logger.trace(
            f"MockCodeOperator.apply called with operation={input_dto.operation}"
        )

        operation = input_dto.operation

        # For simplicity in mock, we'll generate one offspring snippet
        # In real implementation, input_dto would have parent_snippets and other context
        if operation == OPERATION_MUTATION:
            snippet = f"# mutated code snippet v{np.random.randint(100)}"
            metadata = {"operation": "mutation"}
        elif operation == OPERATION_CROSSOVER:
            snippet = f"# crossover code snippet v{np.random.randint(100)}"
            metadata = {"operation": "crossover"}
        elif operation == OPERATION_EDIT:
            snippet = f"# edited code snippet v{np.random.randint(100)}"
            metadata = {"operation": "edit"}
        else:
            snippet = f"# {operation} code snippet v{np.random.randint(100)}"
            metadata = {"operation": operation}

        result = OperatorResult(snippet=snippet, metadata=metadata)
        return OperatorOutput(results=[result])


class MockTestOperator(IOperator):
    """Mocks test genetic operations."""

    def __init__(self, initial_prior: float = 0.5) -> None:
        self.initial_prior = initial_prior

    def supported_operations(self) -> set[str]:
        """Return the set of operations this operator supports."""
        return {OPERATION_MUTATION, OPERATION_CROSSOVER, OPERATION_EDIT}

    def generate_initial_snippets(
        self, input_dto: InitialInput
    ) -> tuple[OperatorOutput, str | None]:
        """Generate initial test snippets (generation 0) with test class block (context_code)."""
        population_size = input_dto.population_size
        logger.debug(f"MockTestOperator: Generating {population_size} test snippets...")

        results = []
        for i in range(population_size):
            test_name = f"test_mock_case_{i}"
            snippet = f"def {test_name}(self):\n        assert sort_array([{i}, {i - 1}]) == [{i - 1}, {i}]"
            result = OperatorResult(
                snippet=snippet,
                metadata={"mock_index": i, "test_name": test_name},
            )
            results.append(result)

        # Create a mock test class block (context_code for tests)
        # Always include the class header even for empty populations
        class_header = "import unittest\nfrom a_mock_solution import sort_array\n\nclass GeneratedTests(unittest.TestCase):\n"
        if results:
            indented_snippets = "\n\n".join([f"    {r.snippet}" for r in results])
            context_code = class_header + indented_snippets
        else:
            # Empty population: return template with pass statement
            context_code = class_header + "    pass\n"

        return OperatorOutput(results=results), context_code

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        """Apply a genetic operation to test snippets."""
        logger.trace(
            f"MockTestOperator.apply called with operation={input_dto.operation}"
        )

        operation = input_dto.operation

        # For simplicity in mock, we'll generate one offspring snippet
        # In real implementation, input_dto would have parent_snippets and other context
        if operation == OPERATION_MUTATION:
            snippet = f"def test_mutated_{np.random.randint(1000)}(self): pass  # mutated test"
            metadata = {"operation": "mutation"}
        elif operation == OPERATION_CROSSOVER:
            snippet = f"def test_crossover_{np.random.randint(1000)}(self): pass  # crossover test"
            metadata = {"operation": "crossover"}
        elif operation == OPERATION_EDIT:
            snippet = (
                f"def test_edited_{np.random.randint(1000)}(self): pass  # edited test"
            )
            metadata = {"operation": "edit"}
        else:
            snippet = f"def test_{operation}_{np.random.randint(1000)}(self): pass  # {operation} test"
            metadata = {"operation": operation}

        result = OperatorResult(snippet=snippet, metadata=metadata)
        return OperatorOutput(results=[result])


# --- Mock Strategies & Helpers ---


class MockParentSelectionStrategy[T: BaseIndividual](IParentSelectionStrategy[T]):
    """Mock implementation of parent selection using roulette wheel."""

    def select_parents(
        self,
        population: BasePopulation[T],
        count: int,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select parents using simple roulette wheel selection.

        Args:
            population: Population to select from
            count: Number of parents to select
            coevolution_context: Full context (unused in mock)

        Returns:
            List of selected parent individuals
        """
        probabilities = population.probabilities
        p_normalized = probabilities / probabilities.sum()

        selected_indices = np.random.choice(
            len(probabilities),
            size=count,
            p=p_normalized,
            replace=True,  # Allow same parent multiple times
        )

        return [population[int(idx)] for idx in selected_indices]


# Legacy alias
MockSelectionStrategy = MockParentSelectionStrategy


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


class MockEliteSelectionStrategy[T: BaseIndividual](IEliteSelectionStrategy[T]):
    """Mock implementation of elite selection using simple top-k by probability."""

    def select_elites(
        self,
        population: BasePopulation[T],
        population_config: PopulationConfig,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select elite individuals using simple top-k selection by probability.

        Args:
            population: Population to select elites from
            population_config: Configuration containing elitism_rate
            coevolution_context: Full context (unused in mock)

        Returns:
            List of elite individuals to preserve
        """
        # Empty state guard: identity operation for empty population
        if population.size == 0:
            logger.debug(
                "MockEliteSelectionStrategy: Population is empty, returning empty elites list"
            )
            return []

        # Determine number of elites based on elitism_rate
        num_elites = int(population.size * population_config.elitism_rate)

        # Simple top-k selection by probability
        probabilities = population.probabilities
        sorted_indices = np.argsort(probabilities)[::-1]  # Descending order
        elite_indices = sorted_indices[:num_elites]

        elites = [population[int(idx)] for idx in elite_indices]

        logger.debug(
            f"MockEliteSelectionStrategy: Selected {len(elites)} elites "
            f"({population_config.elitism_rate:.1%} of {population.size}) from population"
        )

        return elites


# Legacy alias
MockEliteSelector = MockEliteSelectionStrategy


class MockBreedingStrategy:
    """Mock implementation of IBreedingStrategy for testing."""

    def __init__(
        self,
        operator: IOperator,
        individual_type: str = "code",
        test_type: str | None = None,
    ) -> None:
        """
        Initialize mock breeding strategy.

        Args:
            operator: The genetic operator to use for creating individuals (IOperator)
            individual_type: "code" or "test" to determine which individuals to create
            test_type: For test breeding, the test population type (e.g., "unittest")
        """
        self.operator = operator
        self.individual_type = individual_type
        self.test_type = test_type
        self.offspring_count = 0

    def initialize_individuals(
        self,
        population_size: int,
        initial_prior: float,
        problem: Problem,
    ) -> tuple[list[Any], str | None]:
        """
        Create initial population by delegating to operator.

        Args:
            population_size: Number of individuals to create
            initial_prior: Initial probability value for individuals
            problem: The problem context to pass through to operator

        Returns:
            Tuple of (individuals, context_code) from operator.
            Breeder passes this through; orchestrator interprets context_code.
        """
        from .individual import CodeIndividual, TestIndividual

        logger.debug(
            f"MockBreedingStrategy: Creating {population_size} initial {self.individual_type} individuals"
        )
        # Call operator's new generate_initial_snippets method
        from .interfaces import InitialInput

        operator_output, context_code = self.operator.generate_initial_snippets(
            InitialInput(
                operation=OPERATION_INITIAL,
                question_content=problem.question_content,
                population_size=population_size,
                starter_code=problem.starter_code,
            )
        )

        # Wrap the OperatorResults into Individual objects
        individuals = []
        for result in operator_output.results:
            if self.individual_type == "code":
                individual: BaseIndividual = CodeIndividual(
                    snippet=result.snippet,
                    probability=initial_prior,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                    metadata=result.metadata,
                )
            else:
                individual = TestIndividual(
                    snippet=result.snippet,
                    probability=initial_prior,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={"code": [], "test": []},
                    metadata=result.metadata,
                )
            individuals.append(individual)

        return individuals, context_code

    def breed(
        self, coevolution_context: CoevolutionContext, num_offsprings: int
    ) -> Any:
        """
        Generate offspring using mock genetic operations.

        Args:
            coevolution_context: Complete system state for breeding
            num_offsprings: Number of offspring to generate

        Returns:
            List of exactly num_offsprings new individuals
        """
        from .individual import CodeIndividual, TestIndividual

        # Empty state guard: identity operation for num_offsprings=0
        if num_offsprings == 0:
            logger.debug(
                f"MockBreedingStrategy: num_offsprings=0, returning empty {self.individual_type} list"
            )
            return []

        offsprings: list[Any] = []

        logger.debug(
            f"MockBreedingStrategy: Breeding {num_offsprings} {self.individual_type} offspring"
        )

        for _ in range(num_offsprings):
            self.offspring_count += 1

            # Select a random parent for probability inheritance
            if self.individual_type == "code":
                population = coevolution_context.code_population
                parent_idx = np.random.randint(0, population.size)
                parent = population[parent_idx]

                # Create new code individual
                offspring: Any = CodeIndividual(
                    snippet=f"{parent.snippet} # mock offspring {self.offspring_count}",
                    probability=parent.probability * 0.95,  # Slight decrease
                    creation_op="crossover",
                    generation_born=coevolution_context.code_population.generation + 1,
                    parents={"code": [parent.id], "test": []},
                )
            else:
                # Test population - get from context
                test_pop_key = self.test_type or "unittest"
                test_population = coevolution_context.test_populations.get(test_pop_key)
                if test_population is None or test_population.size == 0:
                    # Fallback: create a new test individual
                    offspring = TestIndividual(
                        snippet=f"def test_mock_{self.offspring_count}(self): assert True",
                        probability=0.5,
                        creation_op="initial",
                        generation_born=0,
                        parents={"code": [], "test": []},
                    )
                else:
                    parent_idx = np.random.randint(0, test_population.size)
                    test_parent = test_population[parent_idx]

                    offspring = TestIndividual(
                        snippet=f"{test_parent.snippet} # mock offspring {self.offspring_count}",
                        probability=test_parent.probability * 0.95,
                        creation_op="mutation",
                        generation_born=test_population.generation + 1,
                        parents={"code": [], "test": [test_parent.id]},
                    )

            offsprings.append(offspring)
            logger.trace(
                f"MockBreedingStrategy: Generated {self.individual_type} offspring "
                f"{offspring.id} (prob={offspring.probability:.3f})"
            )

        return offsprings


# MockPareto has been removed - elite selection now uses IEliteSelectionStrategy protocol
# See: IEliteSelectionStrategy in interfaces.py for the new selection architecture


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
    ) -> InteractionData:
        """Execute all code against all tests and return atomic InteractionData."""
        num_code = len(code_population)
        num_tests = len(test_population)

        # Empty state guard: return properly shaped empty InteractionData
        if num_code == 0 or num_tests == 0:
            logger.debug(
                f"MockExecutionSystem: Empty population detected (code={num_code}, "
                f"test={num_tests}), returning empty InteractionData"
            )
            empty_matrix = np.zeros((num_code, num_tests), dtype=int)
            return InteractionData(
                execution_results={}, observation_matrix=empty_matrix
            )

        logger.debug("MockExecutionSystem: 'Running' all tests (mock)...")

        execution_results: dict[str, ExecutionResult] = {}
        observation_matrix = np.zeros((num_code, num_tests), dtype=int)

        # Collect ordered test ids for deterministic mapping
        test_ids = [t.id for t in test_population]

        for i, code_ind in enumerate(code_population):
            logger.trace(f"Executing generated tests for Code ID {code_ind.id}")
            # Determine pass/fail per test (mocked by code probability)
            is_passing = np.random.rand(num_tests) < code_ind.probability

            # Build TestResult mapping keyed by test individual id
            test_results: dict[str, TestResult] = {}
            for j, test_ind in enumerate(test_population):
                passed = bool(is_passing[j])
                observation_matrix[i, j] = 1 if passed else 0
                details = None if passed else "Mock failure message"
                status = "passed" if passed else "failed"
                status_lit = cast(Literal["passed", "failed", "error"], status)
                test_results[test_ind.id] = TestResult(
                    details=details, status=status_lit
                )

            exec_result = ExecutionResult(script_error=False, test_result=test_results)
            execution_results[code_ind.id] = exec_result

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
            logger.trace(
                f"MockExecutionSystem: Empty population (code={num_code}, test={num_tests}), returning matrix with shape ({num_code}, {num_tests})"
            )
            return np.zeros((num_code, num_tests), dtype=int)

        combined_matrix = np.zeros((num_code, num_tests), dtype=int)

        # Map test ids to column indices for alignment
        test_id_to_idx = {t.id: idx for idx, t in enumerate(test_population)}

        for row_idx, code_ind in enumerate(code_population):
            code_id = code_ind.id
            exec_result = execution_results.get(code_id)
            if exec_result is None:
                # Missing entry -> leave zeros
                continue
            for test_id, test_result in exec_result.test_result.items():
                col_idx = test_id_to_idx.get(test_id)
                if col_idx is None:
                    # unknown test id -> skip
                    continue
                combined_matrix[row_idx, col_idx] = (
                    1 if test_result.status == "passed" else 0
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
        # Empty state guard: identity operation for size=0
        if population_size == 0:
            logger.trace(
                "MockBayesianSystem: population_size=0, returning empty belief array"
            )
            return np.array([])

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
        # Empty state guard: identity operation when no tests (N,0) or no code (0,N)
        if observation_matrix.shape[1] == 0 or observation_matrix.shape[0] == 0:
            logger.trace(
                f"MockBayesianSystem: Empty observation matrix {observation_matrix.shape}, "
                "returning prior_code_probs unchanged (identity operation)"
            )
            return prior_code_probs.copy()

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
        # Empty state guard: identity operation when no code (0,N) or no tests (N,0)
        if observation_matrix.shape[0] == 0 or observation_matrix.shape[1] == 0:
            logger.trace(
                f"MockBayesianSystem: Empty observation matrix {observation_matrix.shape}, "
                "returning prior_test_probs unchanged (identity operation)"
            )
            return prior_test_probs.copy()

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
        return test_class_block
