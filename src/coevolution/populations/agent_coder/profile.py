"""AgentCoder population profile factory."""

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
from coevolution.strategies.selection.elite import TopKEliteSelector
from coevolution.strategies.selection.parent_selection import (
    RouletteWheelParentSelection,
)

from .operators.edit import AgentCoderEditOperator
from .operators.initializer import AgentCoderInitializer


from ..registry import registry


@registry.code_factory("agent_coder")
def create_agent_coder_code_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    # ... (rest of parameters)
    initial_prior: float = 0.2,
    llm_workers: int = 1,
    prob_assigner_strategy: str = "min",
) -> CodeProfile:
    """Create an AgentCoder (iterative repair) code profile."""
    # ... (function body)
    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=1,
        max_population_size=1,
        offspring_rate=1.0,
        elitism_rate=0.0,
        diversity_selection=False,
    )

    prob_assigner = ProbabilityAssigner(
        strategy=prob_assigner_strategy, initial_prior=initial_prior
    )
    parent_selector: RouletteWheelParentSelection[CodeIndividual] = (
        RouletteWheelParentSelection()
    )

    edit_op = AgentCoderEditOperator(
        llm=llm_client,
        parser=language_adapter.parser,
        language_name=language_adapter.language,
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
    )

    breeder: Breeder[CodeIndividual] = Breeder(
        registered_operators=[RegisteredOperator(weight=1.0, operator=edit_op)],
        llm_workers=llm_workers,
    )

    initializer = AgentCoderInitializer(
        llm=llm_client,
        parser=language_adapter.parser,
        language_name=language_adapter.language,
        pop_config=population_config,
        edit_operator=edit_op,  # shares conversation history
    )

    elite_selector: IEliteSelectionStrategy[CodeIndividual] = TopKEliteSelector()

    return CodeProfile(
        population_config=population_config,
        breeder=breeder,
        initializer=initializer,
        elite_selector=elite_selector,
    )


__all__ = ["create_agent_coder_code_profile"]
