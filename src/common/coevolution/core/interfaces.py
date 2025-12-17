# coevolution/core/interfaces.py
"""
Core interfaces for the coevolution framework.

This module defines the protocol-based architecture for the coevolution system.
It provides both fine-grained and grouped interfaces to support different
implementation strategies.

Interface Organization:

    1. Data Structures (dataclasses):
       - Configuration classes (BayesianConfig, OperatorRatesConfig, etc.)
       - Context objects (CoevolutionContext, InteractionData)
       - Data transfer objects (Test, Problem, LogEntry)
       - Type aliases (Operations, ExecutionResults, etc.)

    2. Base Classes (ABC):
       - BaseIndividual: Abstract base for code/test individuals
       - BasePopulation: Abstract base for populations

    3. Fine-Grained Protocols:
       These are single-responsibility interfaces that can be composed:
       - IOperator: unified operator for initialization and genetic operations
       - IBreedingStrategy: orchestrates breeding process
       - IEliteSelectionStrategy: selects elite individuals using CoevolutionContext
       - IParentSelectionStrategy: parent selection for breeding
       - IProbabilityAssigner: offspring probability calculation
       - IBeliefInitializer / IBeliefUpdater: Bayesian operations
       - ICodeTestExecutor / IObservationMatrixBuilder: execution operations
       - ITestBlockRebuilder: test class block reconstruction
       - IIndividualFactory: individual creation

    4. Grouped Protocols (Systems):
       These combine related functionality typically implemented together:
       - IBayesianSystem: Belief initialization + updating + mask generation

    5. Context-Based Architecture:
       The system uses context objects to pass state:
       - CoevolutionContext: Complete system state (all populations + interactions)
       - InteractionData: Encapsulates code-test interaction (execution + observation)

       This design maximizes flexibility and extensibility while maintaining clean interfaces.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, overload

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    # We are only importing these for type-hinting, not for runtime logic.
    from .population import CodePopulation, TestPopulation


# Type alias for genetic operations
type Operation = str

# Type alias for parent lineage tracking (grouped by type)
type ParentDict = dict[Literal["code", "test"], list[str]]

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

# ExecutionResults is an alias mapping a code-individual ID (str) to its
# ExecutionResult. We use a forward-reference string for `ExecutionResult`
# to avoid evaluation order issues at import time and mark it as a TypeAlias
# for clearer typing semantics.
ExecutionResults: TypeAlias = dict[str, "ExecutionResult"]


@dataclass(frozen=True)
class TestResult:
    """
    Represents the result of a single unit test execution.
    """

    details: str | None
    status: Literal["passed", "failed", "error"]


@dataclass(frozen=True)
class ExecutionResult:
    """
    Represents the result of executing a unit test suite against a code individual.
    """

    script_error: bool
    test_results: dict[str, TestResult] = field(default_factory=dict)


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

    Defines the probability distribution over genetic operations for breeding.
    All breeding produces new individuals via genetic operations - there is no
    reproduction (elite selection handles preservation of unchanged individuals).

    The breeding strategy samples from operation_rates to select which operation
    to apply. Rates must sum to exactly 1.0 to ensure all breeding is productive.

    Supported operations:
    - mutation: Small random changes to a single parent
    - crossover: Combine genetic material from two parents
    - edit: Improve code based on test failure feedback
    - det: Generate differential tests from divergent code pairs
    - (custom operations can be added)

    Example:
        # Code population breeding
        OperatorRatesConfig(
            operation_rates={
                "mutation": 0.4,   # 40% single-parent variation
                "crossover": 0.3,  # 30% two-parent combination
                "edit": 0.3,       # 30% feedback-driven improvement
            }
        )

        # Test population breeding
        OperatorRatesConfig(
            operation_rates={
                "mutation": 0.5,   # 50% test mutation
                "crossover": 0.3,  # 30% test crossover
                "det": 0.2,        # 20% differential test generation
            }
        )
    """

    operation_rates: dict[
        Operation, float
    ]  # Maps operation name to selection probability

    def __post_init__(self) -> None:
        # Validate all operation rates are in [0, 1]
        for op, rate in self.operation_rates.items():
            if not (0.0 <= rate <= 1.0):
                raise ValueError(
                    f"Rate for operation '{op}' must be in [0.0, 1.0], got {rate}"
                )

        # Validate that rates sum to exactly 1.0
        total = sum(self.operation_rates.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Operation rates must sum to 1.0 (got {total:.6f}). "
                f"All breeding should produce new individuals via genetic operations. "
                f"Elite selection (handled separately) preserves unchanged individuals."
            )


@dataclass
class PopulationConfig:
    """
    Unified configuration for population management.

    Supports both fixed-size and variable-size populations:

    Fixed-size (e.g., test populations):
        Set max_population_size = initial_population_size (default behavior)

    Variable-size (e.g., code population):
        Set max_population_size > initial_population_size
        Control growth with offspring_rate and elitism_rate

    Args:
        initial_prior: Initial probability for new individuals (0.0, 1.0)
        initial_population_size: Size of generation 0 population
        max_population_size: Maximum allowed size. If None, defaults to initial_population_size
        offspring_rate: Fraction of remaining capacity to fill with offspring each generation (0.0, 1.0]
                       Controls growth speed toward max_population_size
        elitism_rate: Fraction of next generation that should be elites from current population (0.0, 1.0)
                     Controls elite/offspring ratio in next generation
        diversity_selection: Whether to use diversity-based selection strategies

    Example - Fixed size (tests):
        PopulationConfig(
            initial_prior=0.5,
            initial_population_size=15,
            elitism_rate=0.5
        )
        # max_population_size auto-set to 15, stays constant
        # Each generation: 7-8 elites + 7-8 offspring = 15 total

    Example - Variable size (code):
        PopulationConfig(
            initial_prior=0.5,
            initial_population_size=10,
            max_population_size=20,
            offspring_rate=0.5,
            elitism_rate=0.4
        )
        # Gen 0: 10 individuals
        # Gen 1 target: min(10 + 0.5*(20-10), 20) = 15
        #   - 6 elites (40% of 15) + 9 offspring = 15
        # Gen 2 target: min(15 + 0.5*(20-15), 20) = 17-18
        #   - 7 elites (40% of 17) + 10 offspring = 17
        # Gradually grows to max_population_size
    """

    initial_prior: float
    initial_population_size: int
    max_population_size: int
    offspring_rate: float = 1.0
    elitism_rate: float = 0.5
    diversity_selection: bool = False

    def __post_init__(self) -> None:
        # Validate required fields
        if not (0.0 < self.initial_prior < 1.0):
            raise ValueError("initial_prior must be in the range (0.0, 1.0)")
        if self.initial_population_size < 0:
            raise ValueError("initial_population_size must be non-negative.")

        # Validate max_population_size
        if self.max_population_size < 0:
            raise ValueError("max_population_size must be non-negative.")
        if self.max_population_size < self.initial_population_size:
            raise ValueError(
                f"max_population_size ({self.max_population_size}) must be >= "
                f"initial_population_size ({self.initial_population_size})"
            )

        # Special validation for empty initial populations
        if self.initial_population_size == 0 and self.max_population_size == 0:
            raise ValueError(
                "Population initialized to 0 must have max_population_size > 0 to allow growth."
            )

        # Validate offspring_rate
        if not (0.0 < self.offspring_rate <= 1.0):
            raise ValueError("offspring_rate must be in the range (0.0, 1.0]")

        # Validate elitism_rate
        if not (0.0 < self.elitism_rate < 1.0):
            raise ValueError("elitism_rate must be in the range (0.0, 1.0)")

    @property
    def is_fixed_size(self) -> bool:
        """Returns True if this is a fixed-size population configuration."""
        return self.max_population_size == self.initial_population_size


@dataclass(frozen=True)
class EvolutionConfig:
    """
    Top-level configuration for controlling the evolutionary run.
    """

    num_generations: int = 5
    max_workers: int = 1  # Number of parallel workers for breeding (1 = sequential)

    def __post_init__(self) -> None:
        if self.num_generations <= 0:
            raise ValueError("num_generations must be positive.")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")


@dataclass(frozen=True)
class BaseOperatorInput:
    """Base class for all operator inputs."""

    operation: Operation
    question_content: str


@dataclass(frozen=True)
class InitialInput(BaseOperatorInput):
    """Input DTO for initial population generation.

    Fields:
        population_size: Number of individuals to generate.
        starter_code: Optional starter code scaffold (may be empty for tests).
    """

    population_size: int
    starter_code: str


@dataclass(frozen=True)
class OperatorResult:
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


# The Output is just a list of these results
@dataclass(frozen=True)
class OperatorOutput:
    results: list[OperatorResult]


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


# =========================================================================
# Population Profile Classes (System Specification Pattern)
# =========================================================================


@dataclass(frozen=True)
class CodeProfile:
    """
    Complete configuration profile for the code population.

    Bundles all components needed to manage and evolve the code population:
    - Population parameters (size, offspring rates)
    - Breeding strategy (how offspring are generated)
    - Elite selection strategy (how best individuals are chosen)

    Example:
        code_profile = CodeProfile(
            population_config=PopulationConfig(
                initial_prior=0.5,
                initial_population_size=10,
                max_population_size=20,
                offspring_rate=0.8
            ),
            breeding_strategy=my_breeding_strategy,
            elite_selector=my_elite_selector
        )

        # Direct access
        max_size = code_profile.population_config.max_population_size
        elites = code_profile.elite_selector.select(context, k=5)
    """

    population_config: PopulationConfig
    breeding_strategy: "IBreedingStrategy[CodeIndividual]"
    elite_selector: "IEliteSelectionStrategy[CodeIndividual]"


@dataclass(frozen=True)
class TestProfile:
    """
    Complete configuration profile for an evolved test population.

    Bundles all components needed to manage and evolve a test population:
    - Population parameters (size, initial probabilities)
    - Breeding strategy (how test offspring are generated)
    - Elite selection strategy (typically Pareto-based)
    - Bayesian configuration (belief update parameters)

    Different test types (e.g., 'unittest', 'differential') can have
    different Bayesian configurations reflecting their varying reliability.

    Example:
        unittest_profile = TestProfile(
            population_config=PopulationConfig(
                initial_prior=0.5,
                initial_population_size=20
            ),
            breeding_strategy=unittest_breeding_strategy,
            elite_selector=pareto_selector,
            bayesian_config=BayesianConfig(
                alpha=0.01,  # Very reliable tests
                beta=0.3,
                gamma=0.3,
                learning_rate=0.01
            )
        )

        # Direct access
        size = unittest_profile.population_config.initial_population_size
        offspring = unittest_profile.breeding_strategy.breed(context, 10)
        beliefs = bayesian_system.update_test_beliefs(
            ...,
            config=unittest_profile.bayesian_config
        )
    """

    population_config: PopulationConfig
    breeding_strategy: "IBreedingStrategy[TestIndividual]"
    elite_selector: "IEliteSelectionStrategy[TestIndividual]"
    bayesian_config: BayesianConfig


@dataclass(frozen=True)
class PublicTestProfile:
    """
    Configuration profile for fixed/ground-truth test populations (e.g., 'public').

    These tests don't evolve, they only provide anchoring for code beliefs.
    Contains only Bayesian configuration since there's no breeding or selection.

    Example:
        public_profile = PublicTestProfile(
            bayesian_config=BayesianConfig(
                alpha=0.001,  # Ground truth tests are highly reliable
                beta=0.1,
                gamma=0.1,
                learning_rate=0.05
            )
        )

        # Usage
        code_beliefs = bayesian_system.update_code_beliefs(
            ...,
            config=public_profile.bayesian_config
        )
    """

    bayesian_config: BayesianConfig


@dataclass
class CoevolutionContext:
    """
    Mutable container for the coevolution system state at a specific generation.

    Holds references to populations and their current interactions. Populations
    are mutated during the evolution cycle (belief updates, generation transitions).
    A new context is created at each generation with fresh interaction data.

    Design principles:
    - One code population (primary population being evolved)
    - Multiple test populations with different roles/types
    - Each test population has interaction data with code population
    - Populations are mutated in-place during evolution
    - Context is reconstructed each generation with new interactions
    - Problem context flows through the system via this container

    Lifecycle within a generation:
        1. Context created with populations + new interaction data
        2. Populations mutated during belief updates
        3. Populations mutated during evolution (if not final generation)
        4. Context discarded, new one created for next generation

    Example:
        context = CoevolutionContext(
            problem=problem,
            code_population=code_pop,  # Will be mutated
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

        # Mutations happen here
        update_beliefs(context)  # Mutates populations
        evolve_populations(context)  # Mutates populations

    Note: current_generation is accessible via code_population.generation
    """

    problem: Problem
    code_population: "CodePopulation"
    test_populations: dict[
        str, "TestPopulation"
    ]  # Keys: "public", "unittest", "differential", "property", etc.
    interactions: dict[str, InteractionData]  # Same keys as test_populations


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
        parents: ParentDict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the shared state for all individuals.

        Args:
            snippet: The code or test content
            probability: Initial probability value
            creation_op: The genetic operation that created this individual
            generation_born: Generation when this individual was created
            parents: Parent individuals grouped by type {"code": [ids], "test": [ids]}
            metadata: Additional operation-specific metadata
        """
        self.lifecycle_log: list[LogEntry] = []
        self._snippet = snippet
        self._creation_op = creation_op
        self._generation_born = generation_born
        self._parents = parents if parents is not None else {"code": [], "test": []}
        self._metadata = metadata if metadata is not None else {}

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
            "parents": self.parents,  # New dict format with types
            "metadata": self.metadata,  # Operation-specific metadata
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
    def parents(self) -> ParentDict:
        """
        Dictionary grouping parent IDs by their types.

        This allows tracking cross-species parents (e.g., DET operator
        takes code parents to create test offspring).

        Returns:
            Dict grouping parent IDs: {"code": [id1, id2], "test": [id3]}.
        """
        return self._parents

    @property
    def code_parent_ids(self) -> list[str]:
        """
        Convenience accessor for code parent IDs.

        Returns:
            List of code parent IDs, empty list if none.
        """
        return self._parents.get("code", [])

    @property
    def test_parent_ids(self) -> list[str]:
        """
        Convenience accessor for test parent IDs.

        Returns:
            List of test parent IDs, empty list if none.
        """
        return self._parents.get("test", [])

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Operation-specific metadata for this individual.

        This can store additional information such as:
        - LLM prompts and responses
        - Reasoning chains
        - Execution traces
        - Operation parameters

        Returns:
            Dict containing operation-specific metadata.
        """
        return self._metadata

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

        Note: Empty populations are now supported to enable dynamic/bootstrapped
        population types (e.g., differential testing that starts with zero tests).
        """
        self._individuals = individuals
        self._generation = generation

        if not individuals:
            logger.debug(
                f"Initialized {self.__class__.__name__} with 0 individuals (empty/bootstrap mode), gen {generation}"
            )
        else:
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

    def get_index_of_individual(self, individual: T_Individual) -> int:
        """Returns the index of the given individual in the population."""
        for idx, ind in enumerate(self._individuals):
            if ind.id == individual.id:
                return idx
        return -1

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


class IOperator(Protocol):
    """
    Protocol for genetic operators that transform code/test strings.

    Operators are "stateless workers" that handle pure string transformation.
    They are strictly decoupled from Domain Objects (Individuals, Populations).
    The Breeding Strategy handles the wrapping/unwrapping of Individual objects.

    Capabilities:
    1. Generate initial snippets (Generation 0)
    2. Apply genetic operations to existing snippets (Generation 1+)
    """

    def generate_initial_snippets(
        self,
        input_dto: InitialInput,
    ) -> tuple[OperatorOutput, str | None]:
        """
        Generate the initial batch of code or test snippets.

        Args:
            input_dto: DTO containing generation parameters (`InitialInput`).

        Returns:
            Tuple of:
            - OperatorOutput: A container with the generated snippets and metadata.
            - context_code: Optional auxiliary code (e.g., test class scaffold
              with imports/setUp) needed to run these snippets. None for code ops.

        Empty State Behavior (size=0):
            If population_size is 0, should return an empty OperatorOutput
            (results=[]) and potentially a context string (e.g. empty test suite scaffold).
        """
        ...

    def supported_operations(self) -> set[Operation]:
        """
        Return the set of operations this operator can handle.

        Used to validate that the Breeding Strategy's configuration only
        requests operations that this operator actually supports.

        Returns:
            Set of operation names (strings).
            Example: {"mutation", "crossover", "edit", "det"}
        """
        ...

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        """
        Apply a genetic operation to input strings.

        The operator does not need to know about 'Individuals', 'Populations',
        or 'ObservationMatrices'. It receives exactly what it needs via the DTO.

        Args:
            input_dto: A strongly-typed DTO (e.g., MutationInput, EditInput)
                       containing the operation name, parent snippets, and
                       any required context (like error traces).

        Returns:
            OperatorOutput containing the generated offspring snippets and metadata.
            The Strategy is responsible for wrapping these into new Individuals.
        """
        ...


class IBreedingStrategy[T_self: BaseIndividual](Protocol):
    """
    Protocol for breeding strategies that handle population generation.

    The Breeding Strategy acts as the "Manager" of the evolutionary process.
    It bridges the gap between the Domain Model (Individuals, Populations) and
    the Stateless Workers (Operators).

    Responsibilities:
    1. **Orchestration**: Decides which operations to perform (Mutation, Crossover, Edit, etc.)
       based on configured rates.
    2. **Selection**: Selects parents and operation-specific context.
       - Generic Selection: Uses IParentSelectionStrategy for fitness-based selection.
       - Specialized Selection: Finds specific contexts (e.g., failing tests for edits,
         divergent pairs for differential testing) using internal logic or helpers.
    3. **Data Preparation (Unwrapping)**: Extracts raw strings/primitives from selected
       Individual objects and packages them into strongly-typed DTOs (e.g., MutationInput,
       EditInput) for the Operator.
    4. **Dispatch**: Calls the 'dumb' IOperator with these DTOs.
    5. **Construction (Wrapping)**: Uses an IIndividualFactory to wrap the raw strings
       returned by the Operator into new Individual objects, assigning IDs, probabilities,
       and lineage data.

    Design Pattern: "Smart Strategy, Dumb Operator"
    - The Strategy absorbs the complexity of the domain (Matrices, Objects, History).
    - The Operator handles only pure string transformation.
    """

    def initialize_individuals(
        self,
        population_size: int,
        initial_prior: float,
        problem: Problem,
    ) -> tuple[list[T_self], str | None]:
        """
        Create initial population by delegating to operator.

        This method orchestrates the creation of Generation 0:
        1. Extracts problem description and starter code from the Problem object.
        2. Calls the operator's `generate_initial_snippets` method with these primitives.
        3. Wraps the resulting snippets into Individual objects using the factory.

        Args:
            population_size: Number of individuals to create.
            initial_prior: Initial probability value for individuals.
            problem: The problem context (used to extract question/starter code strings).

        Returns:
            Tuple of (individuals, context_code):
            - individuals: List of newly generated individuals (Gen 0).
            - context_code: Optional auxiliary code returned by the operator
              (e.g., test class scaffold).

        Empty State Behavior (size=0):
            - If population_size is 0, returns ([], context_code).
            - This supports bootstrapping scenarios where a population (like differential tests)
              starts empty and is populated later via breeding.
        """
        ...

    def breed(
        self,
        coevolution_context: CoevolutionContext,
        num_offsprings: int,
    ) -> list[T_self]:
        """
        Generate offspring using genetic operations.

        The workflow for each offspring is:
        1. **Select Operation**: Sample an operation type (e.g., "edit") from rates config.
        2. **Select Context**:
           - If "mutation": Select 1 parent using standard selector.
           - If "crossover": Select 2 parents using standard selector.
           - If "edit": Find a (Code, FailingTest) pair using specialized logic/helpers.
           - If "det": Find divergent code pairs using specialized logic/helpers.
        3. **Prepare Input**: Construct the specific DTO (e.g., EditInput) with raw strings.
        4. **Execute**: Call `operator.apply(dto)` to get `OperatorOutput`.
        5. **Construct**: Iterate through results, calculating new probabilities and
           creating new Individual objects via the factory.

        Args:
            coevolution_context: Complete system state (populations, matrices) used
                                 for context-aware selection.
            num_offsprings: Number of offspring to generate.

        Returns:
            List of exactly num_offsprings new individuals.

        Empty State Behavior (size=0):
            - If num_offsprings is 0, returns [].
            - If the target population is empty (size=0), the strategy may switch
              to "Bootstrap Mode" (e.g., creating the first tests from code ambiguities)
              instead of standard genetic evolution.

        Raises:
            Exception: Re-raises exceptions from operators to allow the Orchestrator
                       to handle failures (e.g., by retrying or logging).
        """
        ...


class IEliteSelectionStrategy[T: BaseIndividual](Protocol):
    """
    Protocol for selecting elite individuals to preserve unchanged to next generation.

    This interface enables pluggable, sophisticated selection strategies that can consider:
    - Individual probabilities (belief in correctness)
    - Test performance (passing rates across multiple test populations for code)
    - Discrimination ability (for tests - how well they distinguish good/bad code)
    - Diversity metrics (avoiding redundant individuals)
    - Any other custom criteria

    The strategy receives the target population, its configuration, and full coevolution
    context, providing maximum flexibility to implement different selection strategies.

    Example strategies:
    - Code selection: Multi-objective (probability + test performance)
    - Test selection: Pareto front (probability + discrimination)
    - Simple selection: Top-k by probability only
    - Diversity-based: Select diverse individuals covering different behaviors
    """

    def select_elites(
        self,
        population: BasePopulation[T],
        population_config: PopulationConfig,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select elite individuals to preserve unchanged to next generation.

        Args:
            population: The population to select elites from
            population_config: Configuration containing selection preferences
                              (e.g., diversity_selection flag, elitism_rate, elite_size)
            coevolution_context: Complete system state including all populations
                                and their interactions for context-aware selection

        Returns:
            List of elite individuals to preserve unchanged.
            The number of elites is typically determined by population_config
            (e.g., elite_size, survival_rate, or elitism_rate).

        Empty State Behavior (size=0):
            - Identity Operation: When population size is 0, should return an empty
              list [] (no individuals to select from, no elites to preserve).
            - This supports differential testing scenarios where populations may
              start empty and grow through bootstrapping operations.
        """
        ...


class IParentSelectionStrategy[T: BaseIndividual](Protocol):
    """
    Protocol for selecting parent individuals for breeding.

    Defines the contract for any class that can select parents from a population
    based on fitness, behavior, or other criteria. Strategies receive the full
    population and coevolution context to enable sophisticated selection logic.

    Example strategies:
    - Roulette wheel: Probability-proportional selection
    - Tournament: Select best from random subset
    - Rank-based: Select based on rank rather than raw probability
    - Behavior-based: Select parents with complementary behaviors
    - DET-specific: Select code pairs with similar behavior for differential testing
    """

    def select_parents(
        self,
        population: BasePopulation[T],
        count: int,
        coevolution_context: CoevolutionContext,
    ) -> list[T]:
        """
        Select parent individuals for breeding.

        Args:
            population: Population to select parents from
            count: Number of parents to select:
                   - 1 for mutation/reproduction
                   - 2 for crossover
                   - N for batch operations
            coevolution_context: Full coevolution state for context-aware selection
                                (e.g., interaction data, other populations)

        Returns:
            List of selected parent individuals. May contain duplicates if
            count > 1 and the same individual is selected multiple times.

        Note:
            For cross-species operations (e.g., DET selecting code parents for
            test offspring), the strategy can access other populations via
            coevolution_context and return individuals of different type than
            the target population.
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


class IExecutionSystem(Protocol):
    """
    Protocol defining the contract for a code-test executor for the entire population.

    The executor now returns an `InteractionData` object which contains both
    the detailed `ExecutionResults` (ID-keyed) and an index-aligned
    `observation_matrix`. This enforces atomic construction of the matrix
    while the executor iterates through populations (guaranteeing index
    alignment and eliminating the split-brain problem).
    """

    def execute_tests(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
    ) -> InteractionData:
        """
        Executes the given code snippets against the test snippets and returns
        a unified `InteractionData` artifact.

        Args:
                code_population: The population of code snippets to be tested.
                test_population: The population of test snippets to run against the code.

        Returns:
                An `InteractionData` instance containing both:
                    - `execution_results`: ID-keyed mapping of execution outcomes
                    - `observation_matrix`: numpy ndarray with shape
                        (len(code_population), len(test_population)) such that
                        `observation_matrix[i, j]` corresponds to
                        `code_population[i]` vs `test_population[j]`.

        Consistency guarantees:
                - The executor constructs the matrix in the same loop that
                    produces the `execution_results`, ensuring strict alignment.

        Empty State Behavior (size=0):
                - When `test_population` is empty, return an `InteractionData` with
                    an empty `execution_results` and an appropriately shaped empty
                    numpy array (shape (n_codes, 0)).
                - When `code_population` is empty, return an `InteractionData` with
                    empty `execution_results` and an empty numpy array (shape (0, n_tests)).
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

        Empty State Behavior (size=0):
            - When population_size is 0, should return an empty numpy array with
              shape (0,) representing an empty prior distribution.
            - This supports differential testing scenarios where populations start
              empty and grow through bootstrapping.
        """


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

        Empty State Behavior (size=0):
            - Identity Operation: When observation_matrix has shape (N, 0) (no tests),
              should return posterior = prior_code_probs unchanged (no evidence).
            - When prior_code_probs is empty (shape (0,)), should return empty array
              (shape (0,)) representing no code individuals to update.
            - This supports differential testing where test population may start empty.
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

        Empty State Behavior (size=0):
            - Identity Operation: When observation_matrix has shape (0, N) (no code),
              should return posterior = prior_test_probs unchanged (no evidence).
            - When prior_test_probs is empty (shape (0,)), should return empty array
              (shape (0,)) representing no test individuals to update.
            - This supports differential testing where test population may start empty.
        """
        ...


# ============================================
# === GROUPED PROTOCOLS (SYSTEMS)
# ============================================
# These protocols combine related fine-grained protocols that are
# typically implemented together as cohesive subsystems.


class IBayesianSystem(IBeliefUpdater, IBeliefMaskGenerator, Protocol):
    """
    Unified interface for Bayesian belief management.

    Combines initialization and updating of beliefs into a single cohesive system.
    Implementations handle both setting initial priors and performing posterior updates.

    This system is used by the Orchestrator to manage probability updates for both
    code and test populations based on execution results.
    """
