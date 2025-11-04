# coevolution/core/interfaces.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Literal, Protocol

import numpy as np
from loguru import logger

type Operations = Literal["initial", "crossover", "edit", "reproduction", "mutation"]
type ParentProbabilities = list[float]
type UnitTestResult = Any
type ExecutionResults = list[UnitTestResult]
type Sandbox = Any

if TYPE_CHECKING:
    # We are only importing these for type-hinting, not for runtime logic.
    # This assumes they are defined in .population and .individual
    from .population import CodePopulation, TestPopulation


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
    A data structure to hold the hyperparameters for genetic operations.
    """

    crossover_rate: float
    mutation_rate: float
    edit_rate: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in the range [0.0, 1.0]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in the range [0.0, 1.0]")
        if not (0.0 <= self.edit_rate <= 1.0):
            raise ValueError("edit_rate must be in the range [0.0, 1.0]")


@dataclass
class PopulationConfig:
    """
    A data structure to hold the hyperparameters for population management.

    CodePopulation and TestPopulation will both use this same config.
    """

    initial_prior: float
    initial_population_size: int

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

    max_population_size: int
    elitism_rate: float
    offspring_rate: float

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

    def __post_init__(self) -> None:
        if self.num_generations <= 0:
            raise ValueError("num_generations must be positive.")


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


class BaseIndividual(ABC):
    """
    Abstract Base Class for any individual.

    Implements all shared logic and state for individuals,
    such as storing snippets, probabilities, and provenance.

    Subclasses are only required to implement the 'id' property
    and the '__repr__' method.
    """

    def __init__(
        self,
        snippet: str,
        probability: float,
        creation_op: Operations,
        generation_born: int,
        parent_ids: list[str],
    ) -> None:
        """
        Initializes the shared state for all individuals.
        """
        self.log: list[str] = []
        self._snippet = snippet
        self._creation_op = creation_op
        self._generation_born = generation_born
        self._parent_ids = parent_ids
        self._probability = probability

    def add_to_log(self, change_description: str) -> None:
        self.log.append(change_description)

    # --- Abstract Properties (Must be implemented by subclasses) ---

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The unique identifier (e.g., 'C1' or 'T1').
        This is abstract because the prefix is different.
        """
        pass

    # --- Concrete Properties (Shared implementation) ---

    @property
    def snippet(self) -> str:
        """The underlying code or test snippet."""
        return self._snippet

    @property
    def probability(self) -> float:
        """The belief in this individual's correctness."""
        return self._probability

    @probability.setter
    def probability(self, value: float) -> None:
        """Setter for the individual's probability."""
        if not (0.0 <= value <= 1.0):
            raise ValueError("Probability must be between 0.0 and 1.0")
        logger.trace(f"{self.id} probability set to {value:.4f}")
        self._probability = value

    @property
    def creation_op(self) -> str:
        """The name of the operation that created this individual."""
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

    def __eq__(self, other: Any) -> bool:
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

    def __getitem__(self, index: int) -> T_Individual:
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

        This method also logs which individuals were kept (elites),
        added (offspring), and removed (not selected).
        """
        if not new_individuals:
            raise ValueError("Cannot set an empty population for the next generation.")

        # Logging the differences between old and new populations
        old_ids = {ind.id for ind in self._individuals}
        new_ids = {ind.id for ind in new_individuals}

        # Calculate the differences
        deleted_ids = old_ids - new_ids  # In old_ids but not in new_ids
        added_ids = new_ids - old_ids  # In new_ids but not in old_ids
        kept_ids = old_ids & new_ids  # In both (set intersection)

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

        for ind, new_prob in zip(self._individuals, new_probabilities):
            ind.probability = float(new_prob)

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
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class IGeneticOperator(Protocol):
    """
    Abstract interface (Protocol) for genetic operators.

    Defines the contract for any class that creates and evolves
    individuals.
    """

    def mutate(self, individual: str) -> str:
        """
        Apply a mutation to a single code snippet.

        Args:
            individual: The code string to mutate.

        Returns:
            A new, mutated code snippet string.
        """
        ...

    def crossover(self, parent1: str, parent2: str) -> str:
        """
        Apply crossover to two code snippets.

        Args:
            parent1: The first parent snippet.
            parent2: The second parent snippet.

        Returns:
            A new, child snippet resulting from crossover.
        """
        ...

    def edit(self, individual: str, feedback: str) -> str:
        """
        Apply an edit to a code snippet based on feedback.

        Args:
            individual: The code string to edit.
            feedback: The feedback (e.g., error message) to guide the edit.

        Returns:
            A new, edited code snippet string.
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


class ICodeOperator(ICodeInitializer, IGeneticOperator):
    """
    Abstract interface (Protocol) for a Code genetic operator.

    Combines the ICodeInitializer and IGeneticOperator protocols.
    """

    ...


class ITestOperator(ITestInitializer, IGeneticOperator):
    """
    Abstract interface (Protocol) for a Test genetic operator.

    Combines the ITestInitializer and IGeneticOperator protocols.
    """

    ...


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

    This allows the implementation to be a simple function OR
    a stateful class instance that implements __call__.
    """

    def __call__(
        self,
        operation: Operations,
        parent_probs: ParentProbabilities,
        initial_prior: float,
    ) -> float:
        """
        Calculates the probability for a new offspring.

        Args:
            operation: The op that created the child (e.g., "crossover").
            parent_probs: A list of the parent(s)' probabilities.
            initial_prior: The default prior, to be used if the policy dictates.

        Returns:
            The new probability for the offspring.
        """
        ...


class IParetoFrontCalculator(Protocol):
    """
    Protocol defining the contract for a Pareto front calculation strategy.
    Implementations should maximize both objectives.
    """

    def __call__(
        self, probabilities: np.ndarray, discriminations: np.ndarray
    ) -> list[int]:
        """
        Calculates the Pareto front.

        Args:
            probabilities: 1D array of probabilities (Objective 1, to maximize).
            discriminations: 1D array of discriminations (Objective 2, to maximize).

        Returns:
            A list of integer indices for the individuals on the Pareto front.
        """
        ...


class ITestBlockBuilder(Protocol):
    """
    Protocol defining the contract for a test class block rebuilder.
    """

    def __call__(self, original_class_str: str, new_method_snippets: list[str]) -> str:
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


class IIndividualFactory[T_Individual: BaseIndividual](Protocol):
    """
    Protocol defining the contract for an individual factory.
    Generic over the type of individual it creates.
    """

    def __call__(
        self,
        snippet: str,
        probability: float,
        creation_op: Operations,
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


class IFeedbackGenerator[T_Individual: BaseIndividual](Protocol):
    """
    Protocol defining the contract for a feedback generator.
    """

    def __call__(
        self,
        observation_matrix: np.ndarray,
        execution_results: ExecutionResults,
        other_population: BasePopulation[T_Individual],
        individual_idx: int,
    ) -> str:
        """
        Generates a feedback string for the 'edit' operator.

        Args:
            observation_matrix: The full observation matrix (D).
            execution_results: The opaque, detailed results from the sandbox.
            other_population: The opposing population (e.g., TestPopulation
                              if we are generating feedback for a CodeIndividual).
            individual_idx: The index of the individual to generate feedback for.

        Returns:
            A formatted feedback string for the LLM 'edit' prompt.
        """
        ...


class ICodeTestExecutor(Protocol):
    """
    Protocol defining the contract for a code-test executor for the entire population.
    """

    def __call__(
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

        Returns:
            An ExecutionResults object containing the results of the execution.
        """
        ...


class IObservationMatrixBuilder(Protocol):
    """
    Protocol defining the contract for building an observation matrix.
    """

    def __call__(
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


class IDiscriminationCalculator(Protocol):
    """
    Protocol defining the contract for calculating test discrimination.
    """

    def __call__(self, observation_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates discrimination scores for each test.
        """
        ...


class IBeliefInitializer(Protocol):
    """
    Protocol defining the contract for a Bayesian belief initialization strategy.
    """

    def __call__(
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


class IBeliefUpdater(Protocol):
    """
    Protocol defining the contract for a Bayesian belief update strategy.
    """

    def __call__(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        """
        Performs a posterior update for the concerned population.

        Args:
            prior_code_probs: Prior probabilities for the code population.
            prior_test_probs: Prior probabilities for the test population.
            observation_matrix: Matrix of interactions.
            config: A BayesianConfig object containing hyperparameters
                    (alpha, beta, gamma, learning_rate).

        Returns:
            The updated posterior probabilities for the concerned population.
        """
        ...
