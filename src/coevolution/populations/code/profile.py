"""Code population profile factory."""

from __future__ import annotations

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    CodeProfile,
    IEliteSelectionStrategy,
    PopulationConfig,
)
from coevolution.core.interfaces.language import ILanguage
from infrastructure.llm_client import LLMClient

from coevolution.strategies.breeding.breeder import Breeder
from coevolution.core.interfaces.operators import RegisteredOperator
from coevolution.strategies.probability.assigner import ProbabilityAssigner
from coevolution.strategies.selection.elite import (
    CodeDiversityEliteSelector,
    TopKEliteSelector,
)
from coevolution.strategies.selection.failing_test_selection import FailingTestSelector
from coevolution.strategies.selection.parent_selection import (
    RouletteWheelParentSelection,
)

from .operators.mutation import CodeMutationOperator
from .operators.crossover import CodeCrossoverOperator
from .operators.edit import CodeGenericEditOperator
from .operators.initializer import (
    BaseCodeInitializer,
    PlanningCodeInitializer,
    StandardCodeInitializer,
)


from ..registry import registry


@registry.code_factory("default")
def create_default_code_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    # ... (rest of parameters)
    initial_prior: float = 0.2,
    initial_population_size: int = 10,
    max_population_size: int = 15,
    offspring_rate: float = 0.8,
    elitism_rate: float = 0.2,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.2,
    generic_edit_rate: float = 0.6,
    init_pop_batch_size: int = 2,
    llm_workers: int = 4,
    diversity_enabled: bool = True,
    prob_assigner_strategy: str = "min",
    k_failing_tests: int = 10,
    planning_enabled: bool = False,
) -> CodeProfile:
    """Create a standard code population profile."""
    # ... (function body)
    total_rate = mutation_rate + crossover_rate + generic_edit_rate
    if not (0.99 <= total_rate <= 1.01):
        raise ValueError(
            f"Operation rates must sum to 1.0, got {total_rate:.4f} "
            f"(mutation={mutation_rate}, crossover={crossover_rate}, generic_edit={generic_edit_rate})"
        )

    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        offspring_rate=offspring_rate,
        elitism_rate=elitism_rate,
        diversity_selection=diversity_enabled,
    )

    parent_selector: RouletteWheelParentSelection[CodeIndividual] = (
        RouletteWheelParentSelection()
    )
    prob_assigner = ProbabilityAssigner(
        strategy=prob_assigner_strategy, initial_prior=initial_prior
    )

    mutation_op = CodeMutationOperator(
        llm_client,
        language_adapter.parser,
        language_adapter.language,
        parent_selector,
        prob_assigner,
    )
    crossover_op = CodeCrossoverOperator(
        llm_client,
        language_adapter.parser,
        language_adapter.language,
        parent_selector,
        prob_assigner,
    )
    generic_edit_op = CodeGenericEditOperator(
        llm_client,
        language_adapter.parser,
        language_adapter.language,
        parent_selector,
        prob_assigner,
        failing_test_selector=FailingTestSelector,
        k_failing_tests=k_failing_tests,
    )

    breeder: Breeder[CodeIndividual] = Breeder(
        registered_operators=[
            RegisteredOperator(weight=mutation_rate, operator=mutation_op),
            RegisteredOperator(weight=crossover_rate, operator=crossover_op),
            RegisteredOperator(weight=generic_edit_rate, operator=generic_edit_op),
        ],
        llm_workers=llm_workers,
    )

    initializer: BaseCodeInitializer
    if planning_enabled:
        initializer = PlanningCodeInitializer(
            llm=llm_client,
            parser=language_adapter.parser,
            language_name=language_adapter.language,
            pop_config=population_config,
            llm_workers=llm_workers,
        )
    else:
        initializer = StandardCodeInitializer(
            llm=llm_client,
            parser=language_adapter.parser,
            language_name=language_adapter.language,
            pop_config=population_config,
            init_batch_size=init_pop_batch_size,
            llm_workers=llm_workers,
        )

    elite_selector: IEliteSelectionStrategy[CodeIndividual] = (
        CodeDiversityEliteSelector() if diversity_enabled else TopKEliteSelector()
    )

    return CodeProfile(
        population_config=population_config,
        breeder=breeder,
        initializer=initializer,
        elite_selector=elite_selector,
    )


__all__ = ["create_default_code_profile"]
