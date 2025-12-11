# coevolution/core/interfaces.py
"""
Core interfaces for the coevolution framework.

This module defines the protocol-based architecture for the coevolution system.
It provides both fine-grained and grouped interfaces to support different
implementation strategies.

Interface Organization:

    1. Data Structures (dataclasses):
       - Configuration classes (BayesianConfig, OperatorRatesConfig, etc.)
       - Context objects (CoevolutionContext, BreedingContext, OperationContext, InteractionData)
       - Data transfer objects (Test, Problem, LogEntry)
       - Type aliases (Operations, ExecutionResults, etc.)

    2. Base Classes (ABC):
       - BaseIndividual: Abstract base for code/test individuals
       - BasePopulation: Abstract base for populations

    3. Fine-Grained Protocols:
       These are single-responsibility interfaces that can be composed:
       - IGeneticOperator: applies genetic operations using OperationContext
       - IBreedingStrategy: orchestrates breeding process using BreedingContext
       - IEliteSelector: selects elite individuals using CoevolutionContext
       - ICodeInitializer / ITestInitializer: create initial populations
       - ISelectionStrategy: parent selection for breeding
       - IProbabilityAssigner: offspring probability calculation
       - IBeliefInitializer / IBeliefUpdater: Bayesian operations
       - ICodeTestExecutor / IObservationMatrixBuilder: execution operations
       - ITestBlockRebuilder: test class block reconstruction
       - IIndividualFactory: individual creation

    4. Grouped Protocols (Systems):
       These combine related functionality typically implemented together:
       - IBayesianSystem: Belief initialization + updating
       - IExecutionSystem: Code execution + observation matrix building
       - ICodeOperator: Code initialization + genetic operations
       - ITestOperator: Test initialization + genetic operations

    5. Context-Based Architecture:
       The system uses immutable context objects to pass state:
       - CoevolutionContext: Complete system state (all populations + interactions)
       - BreedingContext: Context for breeding strategies (adds rates config)
       - OperationContext: Context for operators (adds selected individuals)
       - InteractionData: Encapsulates code-test interaction (execution + observation)

       This design maximizes flexibility and extensibility while maintaining clean interfaces.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    # We are only importing these for type-hinting, not for runtime logic.
    from .population import CodePopulation, TestPopulation


# Type alias for genetic operations
type Operation = str

# Standard operation names (for convenience, not exhaustive)
OPERATION_INITIAL: Literal["initial"] = "initial"
OPERATION_CROSSOVER: Literal["crossover"] = "crossover"
OPERATION_EDIT: Literal["edit"] = "edit"
OPERATION_REPRODUCTION: Literal["reproduction"] = "reproduction"
OPERATION_MUTATION: Literal["mutation"] = "mutation"


class LifecycleEvent(Enum):
    """Enumeration of lifecycle events for individuals."""

    CREATED = "created"
    BECAME_PARENT = "became_parent"
    SELECTED_AS_ELITE = "selected_as_elite"
    PROBABILITY_UPDATED = "probability_updated"
    DIED = "died"
    SURVIVED = "survived"


type ParentProbabilities = list[float]
type UnitTestResult = Any
type ExecutionResults = dict[int, UnitTestResult]
type Sandbox = Any


@dataclass(frozen=True)
class LogEntry:
    """
    A structured log entry for an individual's lifecycle event.

    Attributes:
        generation: The generation number when this event occurred.
        event: The type of lifecycle event.
        details: Additional event-specific information.
    """

    generation: int
    event: LifecycleEvent
    details: dict[str, Any]


@dataclass
class BayesianConfig:
    """
    A data structure to hold the hyperparameters for Bayesian belief updating.

    Making this a `frozen=True` dataclass ensures its values are immutable
    after creation, which is good practice for configuration objects.
    """

    alpha: float  # P(pass | code correct, test incorrect)
    beta: float  # P(pass | code incorrect, test correct)
    gamma: float  # P(pass | code incorrect, test incorrect)
    learning_rate: float

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in the range (0.0, 1.0)")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must be in the range (0.0, 1.0)")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must be in the range (0.0, 1.0)")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in the range (0.0, 1.0]")


@dataclass
class OperatorRatesConfig:
    """
    Configuration for operation selection probabilities.

    Supports flexible operation types - add new operations without changing this class.
    The breeding strategy samples from operation_rates to select a base operation,
    then applies mutation with mutation_rate probability.

    Example:
        OperatorRatesConfig(
            operation_rates={"crossover": 0.3, "edit": 0.2, "det": 0.1},
            mutation_rate=0.1
        )
    """

    operation_rates: dict[
        Operation, float
    ]  # Maps operation name to selection probability
    mutation_rate: float  # Probability of applying mutation after base operation

    def __post_init__(self) -> None:
        # Validate all operation rates are in [0, 1]
        for op, rate in self.operation_rates.items():
            if not (0.0 <= rate <= 1.0):
                raise ValueError(
                    f"Rate for operation '{op}' must be in [0.0, 1.0], got {rate}"
                )

        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0.0, 1.0]")

        # Validate that rates sum to at most 1.0
        total = sum(self.operation_rates.values())
        if total > 1.0:
            raise ValueError(
                f"Operation rates sum to {total:.4f}, must be ≤ 1.0. "
                f"Remaining probability ({1.0 - total:.4f}) is used for reproduction."
            )

    def validate_against_operator(
        self, operator: "IGeneticOperator", operator_name: str = "operator"
    ) -> None:
        """
        Validate that all operations in this config are supported by the operator.

        Args:
            operator: The genetic operator to validate against.
            operator_name: Descriptive name for error messages (e.g., "code_operator").

        Raises:
            ValueError: If any operation in operation_rates is not supported by the operator.

        Example:
            >>> config = OperatorRatesConfig(
            ...     operation_rates={"crossover": 0.3, "edit": 0.2, "invalid_op": 0.1},
            ...     mutation_rate=0.1
            ... )
            >>> config.validate_against_operator(code_operator, "code_operator")
            # Raises: ValueError: code_operator does not support operations: {'invalid_op'}
        """
        supported_ops = operator.supported_operations()
        configured_ops = set(self.operation_rates.keys())
        unsupported_ops = configured_ops - supported_ops

        if unsupported_ops:
            raise ValueError(
                f"{operator_name} does not support operations: {unsupported_ops}. "
                f"Supported operations: {supported_ops}"
            )


@dataclass
class PopulationConfig:
    """
    A data structure to hold the hyperparameters for population management.

    CodePopulation and TestPopulation will both use this same config.
    """

    initial_prior: float
    initial_population_size: int
    diversity_selection: bool = False  # Whether to use diversity-based selection

    def __post_init__(self) -> None:
        if not (0.0 < self.initial_prior < 1.0):
            raise ValueError("initial_prior must be in the range (0.0, 1.0)")
        if self.initial_population_size <= 0:
            raise ValueError("initial_population_size must be positive.")


@dataclass
class CodePopulationConfig(PopulationConfig):
    """
    A data structure to hold the hyperparameters for Code population management.
    Inherits from PopulationConfig.
    """

    max_population_size: int = 0
    elitism_rate: float = 0.0
    offspring_rate: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_population_size <= 0:
            raise ValueError("max_population_size must be positive.")
        if not (0.0 <= self.elitism_rate <= 1.0):
            raise ValueError("elitism_rate must be in the range [0.0, 1.0]")
        if not (0.0 <= self.offspring_rate <= 1.0):
            raise ValueError("offspring_rate must be in the range [0.0, 1.0]")


@dataclass(frozen=True)
class EvolutionConfig:
    """
    Top-level configuration for controlling the evolutionary run.
    """

    num_generations: int
    random_seed: int
    max_workers: int = 1  # Number of parallel workers for breeding (1 = sequential)

    def __post_init__(self) -> None:
        if self.num_generations <= 0:
            raise ValueError("num_generations must be positive.")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")


@dataclass
class Test:
    input: str
    output: str


@dataclass
class Problem:
    question_title: str
    question_content: str
    question_id: str
    starter_code: str
    public_test_cases: list[Test]
    private_test_cases: list[Test]


if TYPE_CHECKING:
    from .individual import CodeIndividual, TestIndividual
    from .population import CodePopulation, TestPopulation


@dataclass(frozen=True)
class InteractionData:
    """
    Captures the interaction between code population and a specific test population.

    This encapsulates execution results and observation matrix for a single
    code-test population pair, making the data relationship explicit.
    """

    execution_results: ExecutionResults
    observation_matrix: np.ndarray


@dataclass(frozen=True)
class CoevolutionContext:
    """
    Immutable snapshot of the coevolution system state.

    Contains all populations and their interactions at a specific point in time.
    This serves as the single source of truth for the system state, passed to
    breeding strategies and operators.

    Design principles:
    - One code population (primary population being evolved)
    - Multiple test populations with different roles/types
    - Each test population has interaction data with code population
    - Immutable to prevent accidental state mutations

    Example:
        context = CoevolutionContext(
            code_population=code_pop,
            test_populations={
                "public": public_tests,
                "unittest": generated_tests,
                "differential": diff_tests,
            },
            interactions={
                "public": InteractionData(exec_results_1, obs_matrix_1),
                "unittest": InteractionData(exec_results_2, obs_matrix_2),
                "differential": InteractionData(exec_results_3, obs_matrix_3),
            }
        )

    Note: current_generation is accessible via code_population.generation
    """

    code_population: "CodePopulation"
    test_populations: dict[
        str, "TestPopulation"
    ]  # Keys: "public", "unittest", "differential", "property", etc.
    interactions: dict[str, InteractionData]  # Same keys as test_populations


@dataclass(frozen=True)
class BreedingContext:
    """
    Context for breeding strategies with breeding-specific configuration.

    Wraps CoevolutionContext with additional breeding parameters needed to
    generate offspring. This separation keeps the breeding strategy interface
    clean and extensible.

    Args:
        coevolution_context: Complete system state
        rates_config: Operation selection probabilities
        target_population_type: Which population to breed - "code" or test population key
    """

    coevolution_context: CoevolutionContext
    rates_config: OperatorRatesConfig
    target_population_type: str  # "code" or key from test_populations like "unittest"


@dataclass(frozen=True)
class OperationContext:
    """
    Context object passed to genetic operators with operation-specific inputs.

    This design supports:
    - Same-population operations (code→code, test→test)
    - Cross-population operations (code→test for DET, test+code→code for edit)
    - Access to full system state via coevolution context
    - Flexible inputs without assuming genetic lineage

    The breeding strategy selects relevant individuals and passes them explicitly.
    The operator extracts what it needs and ignores the rest.

    Design:
    - code_individuals/test_individuals: Selected inputs for this specific operation
    - coevolution_context: Full system state for operators that need more context

    Examples:
        MUTATION: uses code_individuals[0]
        CROSSOVER: uses code_individuals[0] and [1]
        EDIT: uses code_individuals[0], test_individuals[0],
              accesses coevolution_context.interactions for execution data
        DET: uses code_individuals[0] and [1],
             accesses coevolution_context.test_populations for test context
    """

    operation: Operation
    code_individuals: list["CodeIndividual"]  # Selected inputs for this operation
    test_individuals: list["TestIndividual"]  # Selected inputs for this operation
    coevolution_context: CoevolutionContext  # Full system state


class BaseIndividual(ABC):
    """
    Abstract Base Class for any individual.

    Implements all shared logic and state for individuals,
    such as storing snippets, probabilities, and provenance.

    All core attributes (snippet, probability, creation_op) are immutable
    after creation to maintain consistency and prevent accidental modifications.

    Subclasses are only required to implement the 'id' property
    and the '__repr__' method.
    """

    def __init__(
        self,
        snippet: str,
        probability: float,
        creation_op: Operation,
        generation_born: int,
        parent_ids: list[str],
    ) -> None:
        """
        Initializes the shared state for all individuals.
        """
        self.lifecycle_log: list[LogEntry] = []
        self._snippet = snippet
        self._creation_op = creation_op
        self._generation_born = generation_born
        self._parent_ids = parent_ids

        BaseIndividual._validate_probability(probability)
        self._probability = probability

        # Log creation event
        self._log_event(
            generation=generation_born,
            event=LifecycleEvent.CREATED,
            creation_op=creation_op,
            probability=probability,
        )

    def _log_event(
        self, generation: int, event: LifecycleEvent, **details: Any
    ) -> None:
        """
        Internal method to record structured lifecycle events.

        Args:
            generation: The generation number when this event occurred.
            event: The type of lifecycle event.
            **details: Additional event-specific information.
        """
        # Details are already serializable (Operations is now just a string)
        entry = LogEntry(generation=generation, event=event, details=details)
        self.lifecycle_log.append(entry)
        logger.trace(f"Lifecycle event: {event.value} at gen {generation} - {details}")

    def notify_parent_of(
        self, offspring_id: str, operation: Operation, generation: int
    ) -> None:
        """
        Called when this individual is used as a parent.
        Logs the lifecycle event of producing offspring.

        Args:
            offspring_id: The ID of the offspring produced.
            operation: The genetic operation used (e.g., crossover, edit, reproduction, mutation, det).
            generation: The generation number when this parenting occurred.
        """
        self._log_event(
            generation=generation,
            event=LifecycleEvent.BECAME_PARENT,
            offspring_id=offspring_id,
            operation=operation,
        )

    def notify_selected_as_elite(self, generation: int) -> None:
        """
        Called when this individual is selected as an elite.

        Args:
            generation: The generation number when selected as elite.
        """
        self._log_event(
            generation=generation,
            event=LifecycleEvent.SELECTED_AS_ELITE,
        )

    def notify_died(self, generation: int) -> None:
        """
        Called when this individual is removed from the population.

        Args:
            generation: The generation number when this individual died.
        """
        self._log_event(
            generation=generation,
            event=LifecycleEvent.DIED,
        )

    def notify_survived(self, generation: int) -> None:
        """
        Called when this individual survives to the end of the evolutionary run.

        Args:
            generation: The final generation number.
        """
        self._log_event(
            generation=generation,
            event=LifecycleEvent.SURVIVED,
        )

    def notify_probability_updated(self, generation: int) -> None:
        """
        Called when this individual's probability is updated.

        Args:
            generation: The generation number when this update occurred.
        """
        self._log_event(
            generation=generation,
            event=LifecycleEvent.PROBABILITY_UPDATED,
            probability=self.probability,
        )

    def get_complete_record(self) -> dict[str, Any]:
        """
        Returns a complete record of this individual for logging purposes.

        This method is called when the individual's lifecycle ends (either by
        dying or surviving to the end of the run) to capture all information
        about the individual including its full lifecycle history.

        Returns:
            A dictionary containing all individual attributes and lifecycle events.
        """
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "snippet": self.snippet,
            "creation_op": self.creation_op,  # Already a string
            "generation_born": self.generation_born,
            "probability": self.probability,
            "parent_ids": self.parent_ids,
            "lifecycle_events": [
                {
                    "generation": entry.generation,
                    "event": entry.event.value,
                    "details": entry.details,
                }
                for entry in self.lifecycle_log
            ],
        }

    # --- Abstract Properties (Must be implemented by subclasses) ---

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The unique identifier (e.g., 'C1' or 'T1').
        This is abstract because the prefix is different.
        """

    # -- helper for validation of probability --
    @staticmethod
    def _validate_probability(value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError("Probability must be between 0.0 and 1.0")

    # --- Concrete Properties (Shared implementation) ---
    @property
    def snippet(self) -> str:
        """
        The underlying code or test snippet (immutable after creation).
        """
        return self._snippet

    @property
    def probability(self) -> float:
        """
        The belief in this individual's correctness.

        This value is mutable and gets updated via Bayesian belief updates
        based on test execution results.
        """
        return self._probability

    @probability.setter
    def probability(self, value: float) -> None:
        """
        Setter for the individual's probability.

        Called during Bayesian belief updates to adjust confidence
        based on test execution results.
        """
        BaseIndividual._validate_probability(value)
        logger.trace(
            f"{self.id} probability updated: {self._probability:.4f} -> {value:.4f}"
        )
        self._probability = value

    @property
    def creation_op(self) -> Operation:
        """
        The name of the operation that created this individual (immutable).
        """
        return self._creation_op

    @property
    def parent_ids(self) -> list[str]:
        """A list of parent IDs from which this individual was derived."""
        return self._parent_ids

    @property
    def generation_born(self) -> int:
        """The generation number in which this individual was created."""
        if self._generation_born < 0:
            raise ValueError("Generation born cannot be negative.")
        return self._generation_born

    # --- Common Methods ---

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __eq__(self, other: object) -> bool:
        """
        Individuals are considered equal if they are of the same
        base type and have the same unique ID.
        """
        return isinstance(other, BaseIndividual) and self.id == other.id


class BasePopulation[T_Individual: BaseIndividual](ABC):
    """
    Abstract Base Class for any population.

    Implements all shared logic and state for populations,
    such as storing individuals and their probabilities.

    Subclasses are only required to implement the '__init__' method
    to set up the individuals and probabilities.
    """

    def __init__(self, individuals: list[T_Individual], generation: int = 0) -> None:
        """
        Initializes the shared state for all populations.
        """
        if not individuals:
            logger.error("Attempted to initialize BasePopulation with an empty list.")
            raise ValueError("Population cannot be initialized with an empty list.")

        self._individuals = individuals
        self._generation = generation
        logger.debug(
            f"Initialized {self.__class__.__name__} with {len(individuals)} individuals, gen {generation}"
        )

    def __len__(self) -> int:
        return len(self._individuals)

    @overload
    def __getitem__(self, index: int) -> T_Individual:
        """Gets a single individual by integer index."""

    @overload
    def __getitem__(self, index: slice) -> list[T_Individual]:
        """Gets a list of individuals by slice."""

    def __getitem__(self, index: int | slice) -> T_Individual | list[T_Individual]:
        """
        Gets an individual by index or a list of individuals by slice.

        This relies on the underlying list's __getitem__ which
        already supports both integers and slices.
        """
        # The list's __getitem__ handles both types automatically
        return self._individuals[index]

    def __iter__(self) -> Iterator[T_Individual]:
        for i in self._individuals:
            yield i

    @property
    def size(self) -> int:
        """Returns the size of the population."""
        return len(self._individuals)

    @property
    def generation(self) -> int:
        """Returns the current generation number of the population."""
        return self._generation

    @property
    def individuals(self) -> list[T_Individual]:
        """Returns the list of individuals in the population."""
        return self._individuals.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """Returns the list of probabilities for all individuals."""
        return np.asarray([ind.probability for ind in self._individuals])

    @property
    def snippets(self) -> list[str]:
        """Returns the list of snippets for all individuals."""
        return [ind.snippet for ind in self._individuals]

    @property
    def ids(self) -> list[str]:
        """Returns the list of IDs for all individuals."""
        return [ind.id for ind in self._individuals]

    # Core Public API
    def set_next_generation(self, new_individuals: list[T_Individual]) -> None:
        """
        Replaces the entire current population with a new set of individuals
        and advances the generation counter.

        This method logs which individuals were kept (elites),
        added (offspring), and removed (not selected).

        Note: This method does NOT notify individuals about lifecycle events.
        The caller (typically the Orchestrator) is responsible for notifying
        removed individuals about their death before calling this method.
        """
        if not new_individuals:
            raise ValueError("Cannot set an empty population for the next generation.")

        # Logging the differences between old and new populations
        old_ids_map = {ind.id: ind for ind in self._individuals}
        new_ids = {ind.id for ind in new_individuals}

        # Calculate the differences
        deleted_ids = set(old_ids_map.keys()) - new_ids  # In old_ids but not in new_ids
        added_ids = new_ids - set(old_ids_map.keys())  # In new_ids but not in old_ids
        kept_ids = set(old_ids_map.keys()) & new_ids  # In both (set intersection)

        logger.debug(
            f"Gen {self._generation} -> {self._generation + 1}: "
            f"Kept {len(kept_ids)} elites, "
            f"Added {len(added_ids)} new offspring, "
            f"Removed {len(deleted_ids)} individuals."
        )

        if deleted_ids:
            logger.trace(f"Removed individuals: {sorted(list(deleted_ids))}")
        if added_ids:
            logger.trace(f"Added individuals: {sorted(list(added_ids))}")
        if kept_ids:
            logger.trace(f"Kept elite individuals: {sorted(list(kept_ids))}")

        old_size = self.size
        old_avg_prob = self.compute_average_probability()

        self._individuals = new_individuals
        self._generation += 1
        self._on_generation_advanced()

        new_avg_prob = self.compute_average_probability()
        logger.info(
            f"Advanced {self.__class__.__name__} to gen {self._generation}: "
            f"size {old_size} -> {self.size}, "
            f"avg_prob {old_avg_prob:.4f} -> {new_avg_prob:.4f}"
        )

    def compute_average_probability(self) -> float:
        """Computes and returns the average probability of the population."""
        logger.trace(f"Computing avg probability for {self.size} individuals...")
        if self.size == 0:
            logger.trace("Population is empty, avg prob is 0.0")
            return 0.0
        return float(np.mean(self.probabilities))

    def get_best_individual(self) -> T_Individual | None:
        """Returns the individual with the highest probability."""
        logger.trace("Getting best individual...")
        if self.size == 0:
            logger.warning("get_best_individual called on an empty population.")
            return None

        best_individual = max(self._individuals, key=lambda ind: ind.probability)
        logger.trace(f"Best individual found: {best_individual!r}")
        return best_individual

    def get_top_k_individuals(self, k: int) -> list[T_Individual]:
        """Returns the top k individuals with the highest probabilities."""
        logger.trace(f"Getting top {k} individuals...")
        if k <= 0:
            return []
        k = min(k, self.size)
        return sorted(self._individuals, key=lambda ind: ind.probability, reverse=True)[
            :k
        ]

    def update_probabilities(self, new_probabilities: np.ndarray) -> None:
        """Updates the probabilities of individuals in the population."""
        if len(new_probabilities) != self.size:
            logger.error(
                f"Length mismatch: {len(new_probabilities)} probabilities vs {self.size} individuals."
            )
            raise ValueError("Length of new_probabilities must match population size.")

        old_avg = self.compute_average_probability()

        for ind, new_prob in zip(self._individuals, new_probabilities, strict=False):
            ind.probability = float(new_prob)
            ind.notify_probability_updated(generation=self._generation)

        new_avg = self.compute_average_probability()
        logger.info(
            f"Updated probabilities for {self.__class__.__name__} (gen {self._generation}): "
            f"avg {old_avg:.4f} -> {new_avg:.4f} (Δ{new_avg - old_avg:+.4f})"
        )

    # -- abstract hook for subclasses --
    @abstractmethod
    def _on_generation_advanced(self) -> None:
        """
        Hook method called after the generation is advanced.
        Subclasses can override this to implement custom behavior.
        """

    @abstractmethod
    def __repr__(self) -> str:
        pass


class IGeneticOperator(Protocol):
    """
    Abstract interface (Protocol) for genetic operators.

    Defines the contract for any class that applies genetic operations
    to evolve individuals using a rich context object.
    """

    def supported_operations(self) -> set[Operation]:
        """
        Return the set of operations this operator can handle.

        This method enables validation that OperatorRatesConfig only
        references operations that the operator actually implements.

        Returns:
            Set of operation names (strings) that this operator supports.

        Example:
            {"mutation", "crossover", "edit", "det"}
        """
        ...

    def apply(self, context: OperationContext) -> str:
        """
        Apply a genetic operation using the provided context.

        The operator extracts what it needs from the context:
        - MUTATION: uses context.code_individuals[0]
        - CROSSOVER: uses context.code_individuals[0] and [1]
        - EDIT: uses context.code_individuals[0], context.test_individuals[0],
                context.execution_results, context.feedback
        - DET: uses context.code_individuals[0] and [1], context.observation_matrix
                to generate a test that distinguishes between them

        Args:
            context: Complete context including operation type, input individuals,
                    populations, execution results, and other data.

        Returns:
            A new code or test snippet resulting from the operation.

        Raises:
            ValueError: If required context for the operation is missing.
        """
        ...


class IBreedingStrategy[T_self: BaseIndividual](Protocol):
    """
    Abstract interface (Protocol) for breeding strategies.

    Defines the contract for any class that orchestrates the genetic algorithm
    breeding process: selecting parents, applying operators, and creating offspring.

    This is the primary orchestration interface that implementations can customize
    to define different breeding behaviors.

    The breeding strategy receives a complete BreedingContext with all system state
    and configuration, allowing maximum flexibility in breeding logic.
    """

    def generate_offspring(
        self,
        context: BreedingContext,
    ) -> T_self:
        """
        Generate a single offspring individual using the provided context.

        Args:
            context: Complete breeding context with coevolution state and configuration

        Returns:
            A new offspring individual of type T_self
        """
        ...


class IEliteSelector(Protocol):
    """
    Protocol for selecting elite individuals with access to full system context.

    This interface enables pluggable, sophisticated selection strategies that can consider:
    - Individual probabilities (belief in correctness)
    - Test performance (passing rates across multiple test populations for code)
    - Discrimination ability (for tests - how well they distinguish good/bad code)
    - Diversity metrics (avoiding redundant individuals)
    - Any other custom criteria

    The selector receives complete CoevolutionContext, providing maximum flexibility
    to implement different selection strategies without modifying core architecture.

    Example strategies:
    - Code selection: Multi-objective (probability + test performance)
    - Test selection: Pareto front (probability + discrimination)
    - Simple selection: Top-k by probability only
    - Diversity-based: Select diverse individuals covering different behaviors

    This replaces hardcoded selection logic (e.g., Pareto in TestPopulation),
    making the system extensible and experimentable.
    """

    def select_elites(
        self,
        coevolution_context: CoevolutionContext,
        target_population_type: str,
        population_config: PopulationConfig,
    ) -> list[int]:
        """
        Select elite individuals from the target population.

        Args:
            coevolution_context: Complete system state including all populations
                                and their interactions. Provides full visibility
                                for sophisticated selection decisions.
            target_population_type: Population to select from:
                                   - "code" for code population
                                   - Test population key like "unittest", "differential", etc.
            population_config: Population configuration containing selection preferences
                              (e.g., diversity_selection flag, elitism_rate for code)

        Returns:
            Indices of selected elite individuals.
            For code: typically uses elitism_rate from CodePopulationConfig
            For tests: selector determines size (e.g., Pareto front size)
        """
        ...


class ICodeInitializer(Protocol):
    """
    Abstract interface (Protocol) for code initializers.

    Defines the contract for any class that generates
    initial code snippets.
    """

    def create_initial_snippets(self, population_size: int) -> list[str]:
        """
        Generate an initial population of code snippets.

        Args:
            population_size: The number of individuals to generate.

        Returns:
            A list of code snippet strings.
        """
        ...


class ITestInitializer(Protocol):
    """
    Abstract interface (Protocol) for a Test Population initializer.
    """

    def create_initial_snippets(self, population_size: int) -> tuple[list[str], str]:
        """
        Generate an initial population of test snippets.

        This must be implemented by the concrete (e.g., LLM) operator.
        It is responsible for generating a full runnable block (e.g., a
        unittest class) and extracting the individual test snippets.

        Args:
            population_size: A *hint* for how many test methods to
                             generate within the block.

        Returns:
            A tuple containing:
            1. A list of individual test method snippets (e.g., ["def test_1(self):...", ...])
            2. The full text of the class block they were extracted from.
        """
        ...


class ICodeOperator(ICodeInitializer, IGeneticOperator, Protocol):
    """
    Abstract interface (Protocol) for a Code genetic operator.

    Combines code initialization and genetic operations.
    Handles all aspects of code snippet creation and evolution.
    """


class ITestOperator(ITestInitializer, IGeneticOperator, Protocol):
    """
    Abstract interface (Protocol) for a Test genetic operator.

    Combines test initialization and genetic operations.
    Handles all aspects of test snippet creation and evolution.
    """


class ISelectionStrategy(Protocol):
    """
    Abstract interface (Protocol) for a parent selection strategy.

    Defines the contract for any class that can select one or
    more parents from a population based on their fitness scores
    (probabilities).
    """

    def select(self, probabilities: np.ndarray) -> int:
        """
        Selects a single individual's index from the population.

        Args:
            probabilities: A 1D array of fitness/probability scores
                           for the population.

        Returns:
            The integer index of the selected individual.
        """
        ...

    def select_parents(self, probabilities: np.ndarray) -> tuple[int, int]:
        """
        Selects two *different* parent indices from the population.

        Args:
            probabilities: A 1D array of fitness/probability scores
                           for the population.

        Returns:
            A tuple of two different integer indices (parent1_idx, parent2_idx).

        Raises:
            ValueError: If the population size (inferred from `probabilities`)
                        is less than 2.
        """
        ...


class IProbabilityAssigner(Protocol):
    """
    Protocol defining the contract for a probability assignment policy.
    """

    def assign_probability(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
        initial_prior: float,
    ) -> float:
        """
        Calculates the probability for a new offspring.

        Args:
            operation: The op that created the child (e.g., "crossover", "mutation", "det").
            parent_probs: A list of the parent(s)' probabilities.
            initial_prior: The default prior, to be used if the policy dictates.

        Returns:
            The new probability for the offspring.
        """
        ...


class IIndividualFactory[T_Individual: BaseIndividual](Protocol):
    """
    Protocol defining the contract for an individual factory.
    Generic over the type of individual it creates.
    """

    def __call__(
        self,
        snippet: str,
        probability: float,
        creation_op: Operation,
        generation_born: int,
        parent_ids: list[str],
    ) -> T_Individual:
        """
        Constructs and returns a new individual.

        Args:
            snippet: The code/test snippet.
            probability: The initial probability.
            creation_op: The operation that created this individual.
            generation_born: The generation number.
            parent_ids: A list of parent IDs.

        Returns:
            A new instance of a class that implements BaseIndividual.
        """
        ...


class IDatasetTestBlockBuilder(Protocol):
    """
    Protocol for building test class blocks from dataset test cases (public/private tests).

    This interface is responsible for converting dataset-specific test cases into
    executable unittest test class blocks. Different datasets may have different
    test formats (e.g., STDIN vs FUNCTIONAL), and this interface abstracts that.

    This is specifically for fixed test populations derived from the dataset,
    NOT for general test block building (that's what ITestBlockRebuilder is for).

    Note: Implementations can be stateless (using @staticmethod) if desired,
    but the protocol defines an instance method for maximum flexibility.
    """

    def build_test_class_block(self, test_cases: list[Test], starter_code: str) -> str:
        """
        Build a unittest test class block from dataset test cases.

        Args:
            test_cases: List of Test objects from the dataset (public or private)
            starter_code: The starter code for the problem

        Returns:
            A complete unittest test class block as a string, ready to be executed.
            This should include imports, class definition, and all test methods.

        Example:
            >>> builder = LCBDatasetTestBlockBuilder()
            >>> tests = [Test(input="2 3", output="5", testtype=TestType.STDIN)]
            >>> block = builder.build_test_class_block(tests, "def add(a, b): pass")
            >>> "class" in block and "def test_" in block
            True
        """
        ...


class ICodeTestExecutor(Protocol):
    """
    Protocol defining the contract for a code-test executor for the entire population.
    """

    def execute_tests(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
        sandbox: Sandbox,
    ) -> ExecutionResults:
        """
        Executes the given code snippets against the test snippets.

        Args:
            code_population: The population of code snippets to be tested.
            test_population: The population of test snippets to run against the code.
            sandbox: The sandbox environment for execution.

        Returns:
            An ExecutionResults object containing the results of the execution.
        """
        ...


class IObservationMatrixBuilder(Protocol):
    """
    Protocol defining the contract for building an observation matrix.
    """

    def build_observation_matrix(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
        execution_results: ExecutionResults,
    ) -> np.ndarray:
        """
        Builds the observation matrix from execution results.

        Args:
            code_population: The population of code snippets.
            test_population: The population of test snippets.
            execution_results: The results from executing the code against the tests.

        Returns:
            A 2D numpy array representing the observation matrix.
        """
        ...


class ITestBlockRebuilder(Protocol):
    """
    Protocol defining the contract for rebuilding test class blocks.

    This interface allows for flexible implementations of test block reconstruction,
    whether as a simple utility function or a more sophisticated strategy.
    """

    def rebuild_test_block(
        self, original_class_str: str, new_method_snippets: list[str]
    ) -> str:
        """
        Rebuilds a test class string using new test method snippets.

        Args:
            original_class_str: The full string of the original unittest class.
                                This is used as a template to preserve imports,
                                class name, and helper methods.
            new_method_snippets: A list of new test method snippets (e.g.,
                                 ["def test_new_1(self): ...", ...]) to
                                 replace the old ones.

        Returns:
            A new full unittest class string.
        """
        ...


class IBeliefInitializer(Protocol):
    """
    Protocol defining the contract for a Bayesian belief initialization strategy.
    """

    def initialize_beliefs(
        self,
        population_size: int,
        initial_probability: float,
    ) -> np.ndarray:
        """
        Initializes the prior probabilities for a population.

        Args:
            population_size: The size of the population.
            initial_probability: The initial probability to assign to each individual.

        Returns:
            A numpy array of shape (population_size,) with the initialized probabilities.
        """
        ...


class IBeliefMaskGenerator(Protocol):
    """
    Protocol defining the contract for generating the mask for belief updates.
    This is to not update beliefs on the same observation repeatedly across generations.
    """

    # New explicit public wrappers to disambiguate orientation
    def get_code_update_mask_generation(
        self,
        updating_ind_born_generations: list[int] | np.ndarray,
        other_ind_born_generations: list[int] | np.ndarray,
        current_generation: int,
    ) -> np.ndarray:
        """
        Returns a mask suitable for updating code beliefs. Shape is
        (n_code, n_tests) with rows corresponding to code individuals and
        columns to tests.
        """
        ...

    def get_test_update_mask_generation(
        self,
        updating_ind_born_generations: list[int],
        other_ind_born_generations: list[int],
        current_generation: int,
    ) -> np.ndarray:
        """
        Returns a mask suitable for updating test beliefs. Shape is
        (n_tests, n_code) with rows corresponding to tests and columns to code
        individuals. Implementations may return the transpose of the code
        update mask.
        """
        ...


class IBeliefUpdater(Protocol):
    """
    Protocol defining the contract for Bayesian belief update strategies.

    Provides separate methods for updating code and test beliefs, making it
    explicit which population is being updated and avoiding confusion about
    parameter ordering and matrix transformations.
    """

    def update_code_beliefs(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        code_update_mask_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        """
        Update beliefs for the code population based on test results.

        Args:
            prior_code_probs: Prior probabilities for the code population.
            prior_test_probs: Prior probabilities for the test population.
            observation_matrix: Matrix of interactions (rows=code, cols=tests).
            code_update_mask_matrix: Matrix indicating which observations to consider for updates.
            config: A BayesianConfig object containing hyperparameters
                    (alpha, beta, gamma, learning_rate).

        Returns:
            Updated posterior probabilities for the code population.
        """
        ...

    def update_test_beliefs(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        test_update_mask_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        """
        Update beliefs for the test population based on code results.

        Args:
            prior_code_probs: Prior probabilities for the code population.
            prior_test_probs: Prior probabilities for the test population.
            observation_matrix: Matrix of interactions (rows=code, cols=tests).
            test_update_mask_matrix: Matrix indicating which observations to consider for updates.
            config: A BayesianConfig object containing hyperparameters
                    (alpha, beta, gamma, learning_rate).

        Returns:
            Updated posterior probabilities for the test population.
        """
        ...


# ============================================
# === GROUPED PROTOCOLS (SYSTEMS)
# ============================================
# These protocols combine related fine-grained protocols that are
# typically implemented together as cohesive subsystems.


class IBayesianSystem(
    IBeliefInitializer, IBeliefUpdater, IBeliefMaskGenerator, Protocol
):
    """
    Unified interface for Bayesian belief management.

    Combines initialization and updating of beliefs into a single cohesive system.
    Implementations handle both setting initial priors and performing posterior updates.

    This system is used by the Orchestrator to manage probability updates for both
    code and test populations based on execution results.
    """


class IExecutionSystem(ICodeTestExecutor, IObservationMatrixBuilder, Protocol):
    """
    Unified interface for test execution and observation.

    Combines the execution of code against tests and the building of observation
    matrices into a single cohesive system. Implementations handle both running
    tests in a sandbox and converting results into matrices for belief updates.

    This system is used by the Orchestrator to evaluate code populations against
    test populations (generated, public, and private tests).
    """
