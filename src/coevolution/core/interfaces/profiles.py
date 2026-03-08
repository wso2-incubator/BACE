# coevolution/core/interfaces/profiles.py
"""
Profile classes for bundling population configurations and strategies.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import BayesianConfig, PopulationConfig

if TYPE_CHECKING:
    from ..individual import CodeIndividual, TestIndividual
    from .breeder import IBreeder
    from .config import EvolutionConfig
    from .initializer import IPopulationInitializer
    from .language import IScriptComposer
    from .selection import IEliteSelectionStrategy
    from .systems import IBeliefUpdater, IExecutionSystem, LedgerFactory


@dataclass(frozen=True)
class CodeProfile:
    """
    Complete configuration profile for the code population.

    Bundles all components needed to manage and evolve the code population:
    - Population parameters (size, offspring rates)
    - Breeder (weighted router over code operators)
    - Initializer (creates generation 0)
    - Elite selection strategy

    Example:
        code_profile = CodeProfile(
            population_config=PopulationConfig(...),
            breeder=Breeder([RegisteredOperator(0.5, mutation_op), ...], llm_workers=4),
            initializer=CodeInitializer(...),
            elite_selector=my_elite_selector,
        )
    """

    population_config: PopulationConfig
    breeder: "IBreeder[CodeIndividual]"
    initializer: "IPopulationInitializer[CodeIndividual]"
    elite_selector: "IEliteSelectionStrategy[CodeIndividual]"


@dataclass(frozen=True)
class TestProfile:
    """
    Complete configuration profile for an evolved test population.

    Bundles all components needed to manage and evolve a test population:
    - Population parameters (size, initial probabilities)
    - Breeder (weighted router over test operators)
    - Initializer (creates generation 0, may return [] for differential)
    - Elite selection strategy (typically Pareto-based)
    - Bayesian configuration (belief update parameters)

    Example:
        unittest_profile = TestProfile(
            population_config=PopulationConfig(...),
            breeder=Breeder([RegisteredOperator(0.4, mutation_op), ...]),
            initializer=UnittestInitializer(...),
            elite_selector=pareto_selector,
            bayesian_config=BayesianConfig(...),
        )
    """

    population_config: PopulationConfig
    breeder: "IBreeder[TestIndividual]"
    initializer: "IPopulationInitializer[TestIndividual]"
    elite_selector: "IEliteSelectionStrategy[TestIndividual]"
    bayesian_config: BayesianConfig


@dataclass(frozen=True)
class PublicTestProfile:
    """
    Configuration profile for fixed/ground-truth test populations (e.g. 'public').

    These tests don't evolve; they only provide anchoring for code beliefs.
    """

    bayesian_config: BayesianConfig


@dataclass(frozen=True)
class OrchestratorConfig:
    """Complete configuration bundle for constructing an Orchestrator."""

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
    composer: "IScriptComposer"
