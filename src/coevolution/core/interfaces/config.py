# coevolution/core/interfaces/config.py
"""
Configuration dataclasses for the coevolution framework.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from .types import Operation


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
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in the range [0.0, 1.0]")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError("beta must be in the range [0.0, 1.0]")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in the range [0.0, 1.0]")
        if not (0.0 <= self.learning_rate <= 1.0):
            raise ValueError(
                "learning_rate must be in the range [0.0, 1.0] (0.0 disables updates)"
            )

        # Warn if learning rate is disabled
        if self.learning_rate == 0.0:
            logger.warning(
                "BayesianConfig has learning_rate=0.0 - Bayesian updates will be disabled. "
                "This is typically used for ablation studies to disable anchoring."
            )


@dataclass(frozen=True)
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

    def __getitem__(self, key: Operation) -> float:
        """Allows direct access via config['mutation']"""
        return self.operation_rates[key]


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
            logger.warning(
                "Population initialized to 0 must have max_population_size > 0 to allow growth."
            )

        # Validate offspring_rate
        if not (0.0 <= self.offspring_rate <= 1.0):
            raise ValueError("offspring_rate must be in the range (0.0, 1.0]")

        # Validate elitism_rate
        if not (0.0 <= self.elitism_rate <= 1.0):
            raise ValueError("elitism_rate must be in the range (0.0, 1.0]")

    @property
    def is_fixed_size(self) -> bool:
        """Returns True if this is a fixed-size population configuration."""
        return self.max_population_size == self.initial_population_size


@dataclass(frozen=True)
class EvolutionPhase:
    """
    Defines a specific time block in the evolutionary run with fixed rules.
    """

    name: str
    duration: int
    evolve_code: bool
    evolve_tests: bool

    def __post_init__(self) -> None:
        if self.duration == 0:
            logger.warning(
                f"Phase '{self.name}' has zero duration. It will not be executed."
            )
        if self.duration < 0:
            raise ValueError(f"Phase '{self.name}' duration must be positive.")


@dataclass(frozen=True)
class EvolutionSchedule:
    """
    The immutable plan for the entire evolutionary run.
    """

    phases: list[EvolutionPhase]

    @property
    def total_generations(self) -> int:
        """Derived property: Sum of all phase durations."""
        return sum(p.duration for p in self.phases)


@dataclass(frozen=True)
class EvolutionConfig:
    """
    Top-level configuration for controlling the evolutionary run.
    Now wraps the Schedule object.
    """

    schedule: EvolutionSchedule

    @property
    def num_generations(self) -> int:
        """Derived property: Total generations in the schedule."""
        return self.schedule.total_generations
