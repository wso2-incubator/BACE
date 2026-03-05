"""
Profile factory functions for creating pre-configured population profiles.

Each factory returns a fully-wired Profile using the new Breeder + self-sufficient
operator architecture. Old breeding strategy files are still present and will be
removed in Phase D after this has been verified.
"""

from infrastructure.llm_client import LLMClient
from infrastructure.sandbox.types import SandboxConfig

from ..core.individual import CodeIndividual, TestIndividual
from ..core.interfaces import (
    BayesianConfig,
    CodeProfile,
    IEliteSelectionStrategy,
    PopulationConfig,
    PublicTestProfile,
    TestProfile,
)
from ..core.interfaces.language import ILanguage
from ..strategies.breeding.breeder import Breeder, RegisteredOperator
from ..strategies.operators.agent_coder_operators import (
    AgentCoderEditOperator,
    AgentCoderInitializer,
)
from ..strategies.operators.code_operators import (
    CodeCrossoverOperator,
    CodeEditOperator,
    CodeInitializer,
    CodeMutationOperator,
)
from ..strategies.operators.unittest_operators import (
    UnittestCrossoverOperator,
    UnittestEditOperator,
    UnittestInitializer,
    UnittestMutationOperator,
)
from ..strategies.probability.assigner import ProbabilityAssigner
from ..strategies.selection.elite import (
    CodeDiversityEliteSelector,
    TestDiversityEliteSelector,
    TopKEliteSelector,
)
from ..strategies.selection.failing_test_selection import FailingTestSelector
from ..strategies.selection.parent_selection import RouletteWheelParentSelection


def create_default_code_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    initial_prior: float = 0.2,
    initial_population_size: int = 10,
    max_population_size: int = 15,
    offspring_rate: float = 0.8,
    elitism_rate: float = 0.2,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.2,
    edit_rate: float = 0.6,
    init_pop_batch_size: int = 2,
    llm_workers: int = 4,
    diversity_enabled: bool = True,
    prob_assigner_strategy: str = "min",
    k_failing_tests: int = 10,
    planning_enabled: bool = False,
) -> CodeProfile:
    """Create a standard code population profile."""
    total_rate = mutation_rate + crossover_rate + edit_rate
    if not (0.99 <= total_rate <= 1.01):
        raise ValueError(
            f"Operation rates must sum to 1.0, got {total_rate:.4f} "
            f"(mutation={mutation_rate}, crossover={crossover_rate}, edit={edit_rate})"
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

    mutation_op = CodeMutationOperator(llm_client, language_adapter, parent_selector, prob_assigner)
    crossover_op = CodeCrossoverOperator(llm_client, language_adapter, parent_selector, prob_assigner)
    edit_op = CodeEditOperator(
        llm_client, language_adapter, parent_selector, prob_assigner,
        failing_test_selector=FailingTestSelector,
        k_failing_tests=k_failing_tests,
    )

    breeder: Breeder[CodeIndividual] = Breeder(
        registered_operators=[
            RegisteredOperator(weight=mutation_rate, operator=mutation_op),
            RegisteredOperator(weight=crossover_rate, operator=crossover_op),
            RegisteredOperator(weight=edit_rate, operator=edit_op),
        ],
        llm_workers=llm_workers,
    )

    initializer = CodeInitializer(
        llm=llm_client,
        language_adapter=language_adapter,
        pop_config=population_config,
        init_batch_size=init_pop_batch_size,
        llm_workers=llm_workers,
        planning_enabled=planning_enabled,
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

    mutation_op = UnittestMutationOperator(llm_client, language_adapter, parent_selector, prob_assigner)
    crossover_op = UnittestCrossoverOperator(llm_client, language_adapter, parent_selector, prob_assigner)
    edit_op = UnittestEditOperator(llm_client, language_adapter, parent_selector, prob_assigner)

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
        language_adapter=language_adapter,
        pop_config=population_config,
        llm_workers=llm_workers,
    )

    elite_selector: TestDiversityEliteSelector[TestIndividual] = (
        TestDiversityEliteSelector(test_population_key="unittest")
    )

    bayesian_config = BayesianConfig(
        alpha=alpha, beta=beta, gamma=gamma, learning_rate=learning_rate,
    )

    return TestProfile(
        population_config=population_config,
        breeder=breeder,
        initializer=initializer,
        elite_selector=elite_selector,
        bayesian_config=bayesian_config,
    )


def create_differential_test_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    sandbox_config: SandboxConfig,
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
    from ..strategies.breeding.differential_finder import DifferentialFinder
    from ..strategies.operators.differential_llm_operator import DifferentialLLMOperator
    from ..strategies.operators.differential_operators import (
        DifferentialDiscoveryOperator,
        DifferentialInitializer,
    )
    from ..strategies.selection.functionally_eq_selection import FunctionallyEqSelector

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

    # LLM service (script generation + test-from-IO)
    llm_service = DifferentialLLMOperator(llm=llm_client, language_adapter=language_adapter)

    # Sandbox
    differential_finder = DifferentialFinder(
        language_adapter=language_adapter,
        sandbox_config=sandbox_config,
        enable_multiprocessing=cpu_workers > 1,
        cpu_workers=cpu_workers,
    )

    discovery_op = DifferentialDiscoveryOperator(
        llm=llm_client,
        language_adapter=language_adapter,
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
        llm_workers=1,  # Phase 2 parallelism is handled internally by the operator
    )

    initializer = DifferentialInitializer(
        llm=llm_client,
        language_adapter=language_adapter,
        pop_config=population_config,
    )

    elite_selector: TestDiversityEliteSelector[TestIndividual] = (
        TestDiversityEliteSelector(test_population_key="differential")
    )
    bayesian_config = BayesianConfig(
        alpha=alpha, beta=beta, gamma=gamma, learning_rate=learning_rate,
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
            alpha=alpha, beta=beta, gamma=gamma, learning_rate=learning_rate,
        )
    )


def create_agent_coder_code_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    initial_prior: float = 0.2,
    llm_workers: int = 1,
    prob_assigner_strategy: str = "min",
) -> CodeProfile:
    """Create an AgentCoder (iterative repair) code profile.

    Fixed population of 1, edit-only, stateful conversation history.
    """
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
    # AgentCoder never selects parents from a population (pop size is always 1)
    parent_selector: RouletteWheelParentSelection[CodeIndividual] = (
        RouletteWheelParentSelection()
    )

    edit_op = AgentCoderEditOperator(
        llm=llm_client,
        language_adapter=language_adapter,
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
    )

    breeder: Breeder[CodeIndividual] = Breeder(
        registered_operators=[RegisteredOperator(weight=1.0, operator=edit_op)],
        llm_workers=llm_workers,
    )

    initializer = AgentCoderInitializer(
        llm=llm_client,
        language_adapter=language_adapter,
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
