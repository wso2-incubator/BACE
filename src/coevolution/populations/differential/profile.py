"""Differential population profile factory."""

from __future__ import annotations

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    BayesianConfig,
    PopulationConfig,
    TestProfile,
)
from coevolution.core.interfaces.language import ILanguage
from infrastructure.llm_client import LLMClient
from infrastructure.sandbox.types import SandboxConfig

from coevolution.strategies.breeding.breeder import Breeder, RegisteredOperator
from coevolution.strategies.probability.assigner import ProbabilityAssigner
from coevolution.strategies.selection.elite import TestDiversityEliteSelector
from coevolution.strategies.selection.parent_selection import (
    RouletteWheelParentSelection,
)

from .finder import DifferentialFinder
from .selector import FunctionallyEqSelector
from .operators.llm_operator import DifferentialLLMOperator
from .operators.discovery import DifferentialDiscoveryOperator
from .operators.initializer import DifferentialInitializer


from ..registry import registry


@registry.test_factory("differential")
def create_differential_test_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    sandbox_config: SandboxConfig,
    # ... (parameters)
    initial_prior: float = 0.5,
    initial_population_size: int = 0,
    max_population_size: int = 20,
    offspring_rate: float = 1.0,
    elitism_rate: float = 0.3,
    discovery_rate: float = 1.0,
    alpha: float = 0.05,
    beta: float = 0.3,
    gamma: float = 0.3,
    learning_rate: float = 0.05,
    llm_workers: int = 4,
    cpu_workers: int = 8,
    prob_assigner_strategy: str = "min",
    diversity_enabled: bool = True,
    max_pairs_per_group: int = 5,
    num_passing_tests_to_sample: int = 5,
) -> TestProfile:
    """Create a differential test population profile."""
    # ... (function body)
    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        offspring_rate=offspring_rate,
        elitism_rate=elitism_rate,
        diversity_selection=diversity_enabled,
    )

    prob_assigner = ProbabilityAssigner(
        strategy=prob_assigner_strategy, initial_prior=initial_prior
    )
    parent_selector: RouletteWheelParentSelection[TestIndividual] = (
        RouletteWheelParentSelection()
    )

    llm_service = DifferentialLLMOperator(
        llm=llm_client,
        parser=language_adapter.parser,
        composer=language_adapter.composer,
        language_name=language_adapter.language,
    )

    differential_finder = DifferentialFinder(
        parser=language_adapter.parser,
        composer=language_adapter.composer,
        runtime=language_adapter.runtime,
        sandbox_config=sandbox_config,
        enable_multiprocessing=cpu_workers > 1,
        cpu_workers=cpu_workers,
    )

    discovery_op = DifferentialDiscoveryOperator(
        llm=llm_client,
        parser=language_adapter.parser,
        language_name=language_adapter.language,
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
        llm_service=llm_service,
        differential_finder=differential_finder,
        func_eq_selector=FunctionallyEqSelector(),
        max_pairs_per_group=max_pairs_per_group,
        num_passing_tests_to_sample=num_passing_tests_to_sample,
        llm_workers=llm_workers,
    )

    breeder: Breeder[TestIndividual] = Breeder(
        registered_operators=[
            RegisteredOperator(weight=discovery_rate, operator=discovery_op)
        ],
        llm_workers=1,  # Phase 2 parallelism handled internally by the operator
    )

    initializer = DifferentialInitializer(
        llm=llm_client,
        parser=language_adapter.parser,
        language_name=language_adapter.language,
        pop_config=population_config,
    )

    elite_selector: TestDiversityEliteSelector[TestIndividual] = (
        TestDiversityEliteSelector(test_population_key="differential")
    )
    bayesian_config = BayesianConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        learning_rate=learning_rate,
    )

    return TestProfile(
        population_config=population_config,
        breeder=breeder,
        initializer=initializer,
        elite_selector=elite_selector,
        bayesian_config=bayesian_config,
    )


__all__ = ["create_differential_test_profile"]
