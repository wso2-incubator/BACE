# /path/to/your/project/mock.py
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    CoevolutionContext,
    ExecutionResults,
    IBayesianSystem,
    IEliteSelectionStrategy,
    IExecutionSystem,
    IOperator,
    IParentSelectionStrategy,
    IProbabilityAssigner,
    ITestBlockRebuilder,
    Operation,
    ParentProbabilities,
    PopulationConfig,
    Problem,
    Test,
)
from .population import CodePopulation, TestPopulation

if TYPE_CHECKING:
    from .individual import CodeIndividual, TestIndividual


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


class MockCodeOperator(IOperator["CodeIndividual"]):
    """Combines mock code initialization and genetic operations."""

    def __init__(self, initial_prior: float = 0.5) -> None:
        self.initial_prior = initial_prior

    def supported_operations(self) -> set[str]:
        """Return the set of operations this operator supports."""
        return {OPERATION_MUTATION, OPERATION_CROSSOVER, OPERATION_EDIT}

    def create_initial_individuals(
        self, population_size: int, initial_prior: float, problem: Problem
    ) -> tuple[list["CodeIndividual"], None]:
        """Create initial code individuals (generation 0). Returns (individuals, None) for code."""
        from .individual import CodeIndividual

        logger.debug(
            f"MockCodeOperator: Creating {population_size} code individuals..."
        )

        individuals = []
        for i in range(population_size):
            snippet = problem.starter_code + f"    pass # mock code {i}"
            individual = CodeIndividual(
                snippet=snippet,
                probability=initial_prior,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
                metadata={"mock_index": i},
            )
            individuals.append(individual)

        return individuals, None

    def apply(
        self,
        operation: Operation,
        code_parents: list["CodeIndividual"],
        test_parents: list["TestIndividual"],
        coevolution_context: CoevolutionContext,
    ) -> list["CodeIndividual"]:
        """Apply a genetic operation and return code individuals."""
        from .individual import CodeIndividual

        logger.trace(f"MockCodeOperator.apply called with operation={operation}")

        # Determine generation for offspring
        generation = (
            coevolution_context.code_population.generation + 1
            if coevolution_context
            else 1
        )

        if operation == OPERATION_MUTATION:
            parent = code_parents[0]
            new_snippet = parent.snippet + f" # mutated v{np.random.randint(100)}"
            offspring = CodeIndividual(
                snippet=new_snippet,
                probability=parent.probability * 0.95,  # Slight decrease
                creation_op=OPERATION_MUTATION,
                generation_born=generation,
                parents={parent.id: "code"},
                metadata={"operation": "mutation", "parent_prob": parent.probability},
            )
            return [offspring]

        elif operation == OPERATION_CROSSOVER:
            if len(code_parents) < 2:
                raise ValueError("Crossover requires 2 parents")
            parent1 = code_parents[0]
            parent2 = code_parents[1]
            mid = len(parent1.snippet) // 2
            new_snippet = (
                parent1.snippet[:mid]
                + parent2.snippet[mid:]
                + f" # crossover v{np.random.randint(100)}"
            )
            # Average parent probabilities
            offspring_prob = (parent1.probability + parent2.probability) / 2
            offspring = CodeIndividual(
                snippet=new_snippet,
                probability=offspring_prob,
                creation_op=OPERATION_CROSSOVER,
                generation_born=generation,
                parents={parent1.id: "code", parent2.id: "code"},
                metadata={
                    "operation": "crossover",
                    "parent1_prob": parent1.probability,
                    "parent2_prob": parent2.probability,
                },
            )
            return [offspring]

        elif operation == OPERATION_EDIT:
            if not code_parents:
                raise ValueError("Edit requires at least one code individual")
            parent = code_parents[0]
            new_snippet = (
                parent.snippet + f" # edited from context v{np.random.randint(100)}"
            )
            offspring = CodeIndividual(
                snippet=new_snippet,
                probability=parent.probability * 0.98,
                creation_op=OPERATION_EDIT,
                generation_born=generation,
                parents={parent.id: "code"},
                metadata={
                    "operation": "edit",
                    "has_test_context": bool(test_parents),
                },
            )
            return [offspring]

        else:
            # For any other operation (including custom ones like "det")
            if not code_parents:
                raise ValueError(
                    f"Operation {operation} requires at least one code individual"
                )

            parent = code_parents[0]
            new_snippet = parent.snippet + f" # {operation} v{np.random.randint(100)}"
            offspring = CodeIndividual(
                snippet=new_snippet,
                probability=parent.probability * 0.9,
                creation_op=operation,
                generation_born=generation,
                parents={parent.id: "code"},
                metadata={"operation": operation},
            )
            return [offspring]


class MockTestOperator(IOperator["TestIndividual"]):
    """Mocks test genetic operations."""

    def __init__(self, initial_prior: float = 0.5) -> None:
        self.initial_prior = initial_prior

    def supported_operations(self) -> set[str]:
        """Return the set of operations this operator supports."""
        return {OPERATION_MUTATION, OPERATION_CROSSOVER, OPERATION_EDIT}

    def create_initial_individuals(
        self, population_size: int, initial_prior: float, problem: Problem
    ) -> tuple[list["TestIndividual"], str]:
        """Create initial test individuals (generation 0) with test class block (context_code)."""
        from .individual import TestIndividual

        logger.debug(
            f"MockTestOperator: Creating {population_size} test individuals..."
        )

        individuals = []
        for i in range(population_size):
            test_name = f"test_mock_case_{i}"
            snippet = f"def {test_name}(self):\n        assert sort_array([{i}, {i - 1}]) == [{i - 1}, {i}]"
            individual = TestIndividual(
                snippet=snippet,
                probability=initial_prior,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
                metadata={"mock_index": i, "test_name": test_name},
            )
            individuals.append(individual)

        # Create a mock test class block (context_code for tests)
        class_header = "import unittest\nfrom a_mock_solution import sort_array\n\nclass GeneratedTests(unittest.TestCase):\n"
        indented_snippets = "\n\n".join([f"    {ind.snippet}" for ind in individuals])
        context_code = class_header + indented_snippets

        return individuals, context_code

    def apply(
        self,
        operation: Operation,
        code_parents: list["CodeIndividual"],
        test_parents: list["TestIndividual"],
        coevolution_context: CoevolutionContext,
    ) -> list["TestIndividual"]:
        """Apply a genetic operation and return test individuals."""
        from .individual import TestIndividual

        logger.trace(f"MockTestOperator.apply called with operation={operation}")

        # Determine generation for offspring
        generation = 1
        if coevolution_context:
            # Get generation from test population if available
            test_pops = coevolution_context.test_populations
            if test_pops:
                first_test_pop = next(iter(test_pops.values()))
                generation = first_test_pop.generation + 1

        if operation == OPERATION_MUTATION:
            # Tests are stored in test_parents for test operators
            parent = (
                test_parents[0] if test_parents else code_parents[0]  # Fallback
            )
            new_snippet = parent.snippet + f" # mutated test v{np.random.randint(100)}"
            offspring = TestIndividual(
                snippet=new_snippet,
                probability=parent.probability * 0.95,
                creation_op=OPERATION_MUTATION,
                generation_born=generation,
                parents={parent.id: "test"},
                metadata={"operation": "mutation"},
            )
            return [offspring]

        elif operation == OPERATION_CROSSOVER:
            parents = test_parents if test_parents else code_parents
            if len(parents) < 2:
                raise ValueError("Crossover requires 2 individuals")
            parent1 = parents[0]
            parent2 = parents[1]
            mid = len(parent1.snippet) // 2
            new_snippet = (
                parent1.snippet[:mid]
                + parent2.snippet[mid:]
                + f" # crossover test v{np.random.randint(100)}"
            )
            offspring_prob = (parent1.probability + parent2.probability) / 2
            offspring = TestIndividual(
                snippet=new_snippet,
                probability=offspring_prob,
                creation_op=OPERATION_CROSSOVER,
                generation_born=generation,
                parents={parent1.id: "test", parent2.id: "test"},
                metadata={"operation": "crossover"},
            )
            return [offspring]

        elif operation == OPERATION_EDIT:
            parent = test_parents[0] if test_parents else code_parents[0]
            new_snippet = (
                parent.snippet
                + f" # edited test from context v{np.random.randint(100)}"
            )
            offspring = TestIndividual(
                snippet=new_snippet,
                probability=parent.probability * 0.98,
                creation_op=OPERATION_EDIT,
                generation_born=generation,
                parents={parent.id: "test"},
                metadata={"operation": "edit"},
            )
            return [offspring]

        else:
            # For any other operation (including custom ones like "det")
            # DET might use code parents to create test offspring
            if code_parents:
                # Cross-species operation: code parents → test offspring
                parent1 = code_parents[0]
                parent2 = code_parents[1] if len(code_parents) > 1 else parent1
                new_snippet = f"def test_det_{np.random.randint(1000)}(self): pass  # {operation} test"
                offspring = TestIndividual(
                    snippet=new_snippet,
                    probability=self.initial_prior,
                    creation_op=operation,
                    generation_born=generation,
                    parents={parent1.id: "code", parent2.id: "code"},  # Code parents!
                    metadata={"operation": operation, "cross_species": True},
                )
                return [offspring]
            elif test_parents:
                parent = test_parents[0]
                new_snippet = (
                    parent.snippet + f" # {operation} test v{np.random.randint(100)}"
                )
                offspring = TestIndividual(
                    snippet=new_snippet,
                    probability=parent.probability * 0.9,
                    creation_op=operation,
                    generation_born=generation,
                    parents={parent.id: "test"},
                    metadata={"operation": operation},
                )
                return [offspring]
            else:
                raise ValueError(
                    f"Operation {operation} requires at least one parent individual"
                )


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
            population_config: Configuration (uses elitism_rate if available)
            coevolution_context: Full context (unused in mock)

        Returns:
            List of elite individuals to preserve
        """
        # Determine number of elites based on config
        if hasattr(population_config, "elitism_rate"):
            num_elites = int(population.size * population_config.elitism_rate)
        elif hasattr(population_config, "elite_size"):
            num_elites = population_config.elite_size
        else:
            num_elites = max(1, population.size // 2)  # Default: keep top 50%

        # Simple top-k selection by probability
        probabilities = population.probabilities
        sorted_indices = np.argsort(probabilities)[::-1]  # Descending order
        elite_indices = sorted_indices[:num_elites]

        elites = [population[int(idx)] for idx in elite_indices]

        logger.debug(
            f"MockEliteSelectionStrategy: Selected {len(elites)} elites from "
            f"population (size={population.size})"
        )

        return elites


# Legacy alias
MockEliteSelector = MockEliteSelectionStrategy


class MockBreedingStrategy:
    """Mock implementation of IBreedingStrategy for testing."""

    def __init__(
        self,
        operator: IOperator[Any],
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
        logger.debug(
            f"MockBreedingStrategy: Creating {population_size} initial {self.individual_type} individuals"
        )
        # Operator returns tuple[list[T], str | None] - pass through directly
        return self.operator.create_initial_individuals(
            population_size, initial_prior, problem
        )

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
                    parents={parent.id: "code"},
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
                        parents={},
                    )
                else:
                    parent_idx = np.random.randint(0, test_population.size)
                    test_parent = test_population[parent_idx]

                    offspring = TestIndividual(
                        snippet=f"{test_parent.snippet} # mock offspring {self.offspring_count}",
                        probability=test_parent.probability * 0.95,
                        creation_op="mutation",
                        generation_born=test_population.generation + 1,
                        parents={test_parent.id: "test"},
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
