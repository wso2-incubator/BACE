# coevolution/core/interfaces/profiles.py
"""
Profile classes for bundling population configurations and strategies.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import BayesianConfig, PopulationConfig

if TYPE_CHECKING:
    from ..individual import CodeIndividual, TestIndividual
    from .breeding import IBreedingStrategy
    from .config import EvolutionConfig
    from .operators import IDatasetTestBlockBuilder, ITestBlockRebuilder
    from .selection import IEliteSelectionStrategy
    from .systems import IBeliefUpdater, IExecutionSystem, LedgerFactory


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


@dataclass(frozen=True)
class OrchestratorConfig:
    """
    Complete configuration bundle for constructing an Orchestrator.

    This dataclass groups all orchestrator dependencies into a single structured
    object, making it easier to:
    - Understand required components at a glance
    - Validate configuration before orchestrator construction
    - Pass configuration between functions/builders
    - Test orchestrator initialization with mock configs

    Attributes:
        evo_config: Top-level evolution parameters (generations, workers)
        code_profile: Complete code population configuration
        evolved_test_profiles: Map of test_type → profile for each evolved test population
        public_test_profile: Configuration for public/ground-truth tests
        execution_system: System for running code against tests
        bayesian_system: System for belief updates
        ledger_factory: Factory for creating fresh interaction ledgers
        test_block_rebuilder: Rebuilds test class blocks from method snippets
        dataset_test_block_builder: Builds test blocks from dataset test cases

    Example:
        config = OrchestratorConfig(
            evo_config=EvolutionConfig.simple(generations=10),
            code_profile=my_code_profile,
            evolved_test_profiles={
                "unittest": unittest_profile,
                "differential": differential_profile,
            },
            public_test_profile=public_profile,
            execution_system=execution_system,
            bayesian_system=bayesian_system,
            ledger_factory=my_ledger_factory,
            test_block_rebuilder=test_rebuilder,
            dataset_test_block_builder=dataset_builder,
        )

        orchestrator = Orchestrator(
            evo_config=config.evo_config,
            code_profile=config.code_profile,
            evolved_test_profiles=config.evolved_test_profiles,
            public_test_profile=config.public_test_profile,
            execution_system=config.execution_system,
            bayesian_system=config.bayesian_system,
            ledger_factory=config.ledger_factory,
            test_block_rebuilder=config.test_block_rebuilder,
            dataset_test_block_builder=config.dataset_test_block_builder,
        )
    """

    # Top-level configuration
    evo_config: "EvolutionConfig"

    # Population profiles
    code_profile: "CodeProfile"
    evolved_test_profiles: dict[str, "TestProfile"]
    public_test_profile: "PublicTestProfile"

    # Global infrastructure systems
    execution_system: "IExecutionSystem"
    bayesian_system: "IBeliefUpdater"
    ledger_factory: "LedgerFactory"
    test_block_rebuilder: "ITestBlockRebuilder"
    dataset_test_block_builder: "IDatasetTestBlockBuilder"
