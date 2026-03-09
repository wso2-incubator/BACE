"""Unittest population profile factories."""

from __future__ import annotations

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    BayesianConfig,
    PopulationConfig,
    PublicTestProfile,
    TestProfile,
)
from coevolution.core.interfaces.language import ILanguage
from infrastructure.llm_client import LLMClient

from coevolution.strategies.breeding.breeder import Breeder, RegisteredOperator
from coevolution.strategies.probability.assigner import ProbabilityAssigner
from coevolution.strategies.selection.elite import TestDiversityEliteSelector
from coevolution.strategies.selection.parent_selection import (
    RouletteWheelParentSelection,
)

from .operators.mutation import UnittestMutationOperator
from .operators.crossover import UnittestCrossoverOperator
from .operators.edit import UnittestEditOperator
from .operators.initializer import UnittestInitializer


def create_unittest_test_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    initial_prior: float = 0.2,
    initial_population_size: int = 20,
    max_population_size: int = 20,
    elitism_rate: float = 0.4,
    mutation_rate: float = 0.2,
    offspring_rate: float = 0.8,
    crossover_rate: float = 0.3,
    edit_rate: float = 0.5,
    alpha: float = 0.01,
    beta: float = 0.3,
    gamma: float = 0.3,
    learning_rate: float = 0.05,
    llm_workers: int = 1,
    prob_assigner_strategy: str = "min",
    diversity_enabled: bool = True,
) -> TestProfile:
    """Create a unittest test population profile."""
    total_rate = mutation_rate + crossover_rate + edit_rate
    if not (0.99 <= total_rate <= 1.01):
        raise ValueError(f"Operation rates must sum to 1.0, got {total_rate:.4f}")

    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        elitism_rate=elitism_rate,
        offspring_rate=offspring_rate,
        diversity_selection=diversity_enabled,
    )

    parent_selector: RouletteWheelParentSelection[TestIndividual] = (
        RouletteWheelParentSelection()
    )
    prob_assigner = ProbabilityAssigner(
        strategy=prob_assigner_strategy, initial_prior=initial_prior
    )

    mutation_op = UnittestMutationOperator(
        llm_client,
        language_adapter.parser,
        language_adapter.language,
        parent_selector,
        prob_assigner,
    )
    crossover_op = UnittestCrossoverOperator(
        llm_client,
        language_adapter.parser,
        language_adapter.language,
        parent_selector,
        prob_assigner,
    )
    edit_op = UnittestEditOperator(
        llm_client,
        language_adapter.parser,
        language_adapter.language,
        parent_selector,
        prob_assigner,
    )

    breeder: Breeder[TestIndividual] = Breeder(
        registered_operators=[
            RegisteredOperator(weight=mutation_rate, operator=mutation_op),
            RegisteredOperator(weight=crossover_rate, operator=crossover_op),
            RegisteredOperator(weight=edit_rate, operator=edit_op),
        ],
        llm_workers=llm_workers,
    )

    initializer = UnittestInitializer(
        llm=llm_client,
        parser=language_adapter.parser,
        language_name=language_adapter.language,
        pop_config=population_config,
        llm_workers=llm_workers,
    )

    elite_selector: TestDiversityEliteSelector[TestIndividual] = (
        TestDiversityEliteSelector(test_population_key="unittest")
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


def create_public_test_profile(
    alpha: float = 0.001,
    beta: float = 0.1,
    gamma: float = 0.1,
    learning_rate: float = 0.05,
) -> PublicTestProfile:
    """Create a public/ground-truth test profile (fixed tests, no evolution)."""
    return PublicTestProfile(
        bayesian_config=BayesianConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            learning_rate=learning_rate,
        )
    )


__all__ = ["create_unittest_test_profile", "create_public_test_profile"]
