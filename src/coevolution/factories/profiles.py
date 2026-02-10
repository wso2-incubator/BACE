"""
Profile factory functions for creating pre-configured population profiles.

This module provides factory functions for common coevolution scenarios:
- Code population with LLM-based breeding
- Unittest test population for discrimination-driven evolution
- Differential test population for divergence discovery
- Public test population for ground-truth anchoring

Each factory encapsulates the complete setup for a population type,
including operator configuration, breeding strategies, selection mechanisms,
and Bayesian parameters.

Example Usage:
    >>> from coevolution.factories import (
    ...     create_default_code_profile,
    ...     create_unittest_test_profile,
    ...     create_public_test_profile
    ... )
    >>>
    >>> # Create profiles with custom parameters
    >>> code_profile = create_default_code_profile(
    ...     llm_client=my_llm,
    ...     sandbox=my_sandbox,
    ...     initial_population_size=10,
    ...     max_population_size=15,
    ...     mutation_rate=0.3,
    ...     crossover_rate=0.2,
    ...     edit_rate=0.5
    ... )
    >>>
    >>> unittest_profile = create_unittest_test_profile(
    ...     llm_client=my_llm,
    ...     alpha=0.01,
    ...     beta=0.3
    ... )
    >>>
    >>> public_profile = create_public_test_profile(alpha=0.001)
"""

from coevolution.strategies.breeding.agent_coder_breeding import (
    AgentCoderBreedingStrategy,
)
from coevolution.strategies.operators.agent_coder_llm_operator import (
    AgentCoderLLMOperator,
)
from infrastructure.llm_client import LLMClient
from infrastructure.sandbox.types import SandboxConfig

from ..core.individual import CodeIndividual, TestIndividual
from ..core.interfaces import (
    BayesianConfig,
    CodeProfile,
    IEliteSelectionStrategy,
    OperatorRatesConfig,
    PopulationConfig,
    PublicTestProfile,
    TestProfile,
)
from ..core.interfaces.language import ILanguage
from ..strategies.breeding.code_breeding import CodeBreedingStrategy
from ..strategies.breeding.differential_breeding import DifferentialBreedingStrategy
from ..strategies.breeding.differential_finder import DifferentialFinder
from ..strategies.breeding.unittest_breeding import UnittestBreedingStrategy
from ..strategies.operators.code_llm_operator import CodeLLMOperator
from ..strategies.operators.differential_llm_operator import DifferentialLLMOperator
from ..strategies.operators.unittest_llm_operator import UnittestLLMOperator
from ..strategies.probability.assigner import ProbabilityAssigner
from ..strategies.selection.elite import (
    CodeDiversityEliteSelector,
    TestDiversityEliteSelector,
    TopKEliteSelector,
)
from ..strategies.selection.failing_test_selection import FailingTestSelector
from ..strategies.selection.functionally_eq_selection import FunctionallyEqSelector
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
) -> CodeProfile:
    """
    Create a standard code population profile.

    This factory encapsulates the complete code evolution setup:
    - Variable-size population (can grow from initial to max size)
    - LLM-based operator supporting mutation, crossover, edit
    - Diversity-aware or simple top-k elite selection
    - Roulette wheel parent selection
    - Probability assigner for offspring

    Args:
        llm_client: LLM client for code generation
        sandbox: Sandbox for code execution (used by FailingTestSelector)
        initial_prior: Initial probability for new code (default: 0.5)
        initial_population_size: Starting population size (default: 10)
        max_population_size: Maximum allowed size (default: 20)
        offspring_rate: Fraction of capacity to fill each generation (default: 0.8)
        elitism_rate: Fraction of population to preserve as elites (default: 0.3)
        mutation_rate: Probability of mutation operation (default: 0.2)
        crossover_rate: Probability of crossover operation (default: 0.2)
        edit_rate: Probability of edit operation (default: 0.6)
        init_pop_batch_size: Number of individuals to generate per batch during initialization (default: 2)
        llm_workers: Number of parallel LLM workers for breeding (default: 4)
        diversity_enabled: Use diversity selector vs simple top-k (default: True)
        k_failing_tests: Maximum number of failing tests to select for edit operations (default: 10)

    Returns:
        CodeProfile with configured components

    Raises:
        ValueError: If rates don't sum to 1.0 or parameters are invalid
    """
    # Validate rates sum to 1.0
    total_rate = mutation_rate + crossover_rate + edit_rate
    if not (0.99 <= total_rate <= 1.01):  # Allow small floating point error
        raise ValueError(
            f"Operation rates must sum to 1.0, got {total_rate:.4f} "
            f"(mutation={mutation_rate}, crossover={crossover_rate}, edit={edit_rate})"
        )

    # Create population config
    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        offspring_rate=offspring_rate,
        elitism_rate=elitism_rate,
        diversity_selection=diversity_enabled,
    )

    # Create operator
    code_operator = CodeLLMOperator(llm=llm_client, language_adapter=language_adapter)

    # Create operator rates config
    operator_rates = OperatorRatesConfig(
        operation_rates={
            "mutation": mutation_rate,
            "crossover": crossover_rate,
            "edit": edit_rate,
        }
    )

    # Create breeding strategy components
    parent_selector: RouletteWheelParentSelection[CodeIndividual] = (
        RouletteWheelParentSelection()
    )
    prob_assigner = ProbabilityAssigner(strategy=prob_assigner_strategy)

    # Create breeding strategy
    breeding_strategy = CodeBreedingStrategy(
        operator=code_operator,
        op_rates_config=operator_rates,
        pop_config=population_config,
        probability_assigner=prob_assigner,
        parent_selector=parent_selector,
        failing_test_selector=FailingTestSelector,
        init_pop_batch_size=init_pop_batch_size,
        llm_workers=llm_workers,
        k_failing_tests=k_failing_tests,
    )

    # Create elite selector
    if diversity_enabled:
        elite_selector: IEliteSelectionStrategy[CodeIndividual] = (
            CodeDiversityEliteSelector()
        )
    else:
        elite_selector = TopKEliteSelector()

    return CodeProfile(
        population_config=population_config,
        breeding_strategy=breeding_strategy,
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
    """
    Create a unittest test population profile.

    This factory configures discrimination-driven unittest evolution:
    - Fixed-size population
    - LLM-based operator with mutation, crossover, edit
    - Edit operation uses passing vs failing code pairs for discrimination
    - Diversity-aware elite selection (Pareto-like)
    - Moderate reliability Bayesian parameters

    Args:
        llm_client: LLM client for test generation
        initial_prior: Initial probability for new tests (default: 0.5)
        initial_population_size: Fixed population size (default: 20)
        elitism_rate: Fraction to preserve as elites (default: 0.4)
        mutation_rate: Probability of mutation (default: 0.3)
        crossover_rate: Probability of crossover (default: 0.2)
        edit_rate: Probability of edit (default: 0.5)
        alpha: P(pass | code correct, test incorrect) (default: 0.01)
        beta: P(pass | code incorrect, test correct) (default: 0.3)
        gamma: P(pass | both incorrect) (default: 0.3)
        learning_rate: Belief update learning rate (default: 0.05)
        llm_workers: Number of parallel LLM workers for breeding (default: 1)

    Returns:
        TestProfile with configured components

    Raises:
        ValueError: If rates don't sum to 1.0 or parameters are invalid
    """
    # Validate rates
    total_rate = mutation_rate + crossover_rate + edit_rate
    if not (0.99 <= total_rate <= 1.01):
        raise ValueError(f"Operation rates must sum to 1.0, got {total_rate:.4f}")

    # Create population config (fixed-size)
    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        elitism_rate=elitism_rate,
        offspring_rate=offspring_rate,
        diversity_selection=diversity_enabled,
    )

    # Create operator
    test_operator = UnittestLLMOperator(
        llm=llm_client, language_adapter=language_adapter
    )

    # Create operator rates config
    operator_rates = OperatorRatesConfig(
        operation_rates={
            "mutation": mutation_rate,
            "crossover": crossover_rate,
            "edit": edit_rate,
        }
    )

    # Create breeding strategy components
    parent_selector: RouletteWheelParentSelection[TestIndividual] = (
        RouletteWheelParentSelection()
    )
    prob_assigner = ProbabilityAssigner(strategy=prob_assigner_strategy)

    # Create breeding strategy
    breeding_strategy = UnittestBreedingStrategy(
        operator=test_operator,
        op_rates_config=operator_rates,
        pop_config=population_config,
        probability_assigner=prob_assigner,
        parent_selector=parent_selector,
        llm_workers=llm_workers,
    )

    # Create elite selector (diversity-aware for tests)
    elite_selector: TestDiversityEliteSelector[TestIndividual] = (
        TestDiversityEliteSelector(test_population_key="unittest")
    )

    # Create Bayesian config
    bayesian_config = BayesianConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        learning_rate=learning_rate,
    )

    return TestProfile(
        population_config=population_config,
        breeding_strategy=breeding_strategy,
        elite_selector=elite_selector,
        bayesian_config=bayesian_config,
    )


def create_differential_test_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    sandbox_config: SandboxConfig,
    initial_prior: float = 0.5,
    initial_population_size: int = 0,  # Bootstrap mode
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
    """
    Create a differential test population profile.

    This factory configures DET (Differential Evolution Testing):
    - Starts empty, grows via discovery
    - Discovery operation finds divergent code pairs
    - Crossover combines existing differential tests
    - Lower reliability than unittests (higher alpha)

    Args:
        llm_client: LLM client for differential test generation
        sandbox: Sandbox for code execution (used by FunctionallyEquivalentCodeSelector)
        initial_prior: Initial probability for new tests (default: 0.5)
        initial_population_size: Starting size, usually 0 (default: 0)
        max_population_size: Maximum allowed size (default: 20)
        offspring_rate: Fraction of capacity to fill (default: 1.0 for aggressive growth)
        elitism_rate: Fraction to preserve (default: 0.3)
        discovery_rate: Probability of discovery operation (default: 0.7)
        crossover_rate: Probability of crossover (default: 0.3)
        alpha: P(pass | code correct, test incorrect) (default: 0.05, less reliable than unittest)
        beta: P(pass | code incorrect, test correct) (default: 0.3)
        gamma: P(pass | both incorrect) (default: 0.3)
        learning_rate: Belief update learning rate (default: 0.05)
        llm_workers: Number of parallel LLM workers for script generation (default: 4)
        cpu_workers: Number of parallel CPU workers for sandbox execution (default: 8)
        max_pairs_per_group: Maximum pairs to try per functional group (default: 5)
        num_passing_tests_to_sample: Number of passing test cases to randomly sample for differential input generation (default: 5)

    Returns:
        TestProfile with configured components

    Raises:
        ValueError: If rates don't sum to 1.0 or parameters are invalid
    """
    # Validate rates
    total_rate = discovery_rate
    if not (0.99 <= total_rate <= 1.01):
        raise ValueError(f"Operation rates must sum to 1.0, got {total_rate:.4f}")

    # Create population config (variable-size, bootstrap mode)
    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        offspring_rate=offspring_rate,
        elitism_rate=elitism_rate,
        diversity_selection=diversity_enabled,
    )

    # Create operator
    differential_operator = DifferentialLLMOperator(
        llm=llm_client, language_adapter=language_adapter
    )

    # Create operator rates config
    operator_rates = OperatorRatesConfig(
        operation_rates={
            "discovery": discovery_rate,
        }
    )

    # Create breeding strategy components
    parent_selector: RouletteWheelParentSelection[TestIndividual] = (
        RouletteWheelParentSelection()
    )
    prob_assigner = ProbabilityAssigner(strategy=prob_assigner_strategy)

    differential_finder = DifferentialFinder(
        language_adapter=language_adapter,
        sandbox_config=sandbox_config,
        enable_multiprocessing=cpu_workers > 1,
        cpu_workers=cpu_workers,
    )

    # Create breeding strategy
    breeding_strategy = DifferentialBreedingStrategy(
        operator=differential_operator,
        op_rates_config=operator_rates,
        pop_config=population_config,
        probability_assigner=prob_assigner,
        parent_selector=parent_selector,
        functionally_equivalent_code_selector=FunctionallyEqSelector(),
        differential_finder=differential_finder,
        llm_workers=llm_workers,
        max_pairs_per_group=max_pairs_per_group,
        num_passing_tests_to_sample=num_passing_tests_to_sample,
    )

    # Create elite selector (diversity-aware for differential tests)
    elite_selector: TestDiversityEliteSelector[TestIndividual] = (
        TestDiversityEliteSelector(test_population_key="differential")
    )

    # Create Bayesian config
    bayesian_config = BayesianConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        learning_rate=learning_rate,
    )

    return TestProfile(
        population_config=population_config,
        breeding_strategy=breeding_strategy,
        elite_selector=elite_selector,
        bayesian_config=bayesian_config,
    )


def create_public_test_profile(
    alpha: float = 0.001,
    beta: float = 0.1,
    gamma: float = 0.1,
    learning_rate: float = 0.05,
) -> PublicTestProfile:
    """
    Create a public/ground-truth test profile.

    This factory configures fixed tests from the dataset:
    - Very high reliability (low alpha = 0.001)
    - Used for anchoring code beliefs
    - No breeding or selection (fixed population)

    Args:
        alpha: P(pass | code correct, test incorrect) (default: 0.001, very reliable)
        beta: P(pass | code incorrect, test correct) (default: 0.1)
        gamma: P(pass | both incorrect) (default: 0.1)
        learning_rate: Belief update learning rate (default: 0.05)

    Returns:
        PublicTestProfile with Bayesian configuration
    """
    bayesian_config = BayesianConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        learning_rate=learning_rate,
    )

    return PublicTestProfile(bayesian_config=bayesian_config)


def create_agent_coder_code_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    initial_prior: float = 0.2,
    llm_workers: int = 1,
    prob_assigner_strategy: str = "min",
) -> CodeProfile:
    """
    Create an 'AgentCoder' style code profile (Iterative Repair).

    This factory configures a single-agent evolution loop:
    - Fixed Population Size = 1 (Linear Evolution)
    - Stateful LLM Operator (maintains conversation history)
    - Edit-only evolution (No mutation/crossover)
    - Elitism = 0 (Offspring always replaces parent if valid)

    Args:
        llm_client: LLM client for code generation.
        initial_prior: Initial probability for new code (default: 0.2).
        llm_workers: Number of parallel workers for initialization (default: 1).
        prob_assigner_strategy: Strategy for assigning probabilities (default: "min").

    Returns:
        CodeProfile configured for AgentCoder.
    """
    # 1. Enforce Linear Constraints
    # AgentCoder works by modifying a single file iteratively.
    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=1,
        max_population_size=1,
        offspring_rate=1.0,  # 1 offspring replaces 1 parent
        elitism_rate=0.0,  # No elitism, we rely on the loop
        diversity_selection=False,  # Diversity irrelevant for size 1
    )

    # 2. Create Stateful Operator
    # Note: The Orchestrator MUST manage the lifecycle (reset_session) of this operator.
    agent_operator = AgentCoderLLMOperator(
        llm=llm_client, language_adapter=language_adapter
    )

    # 3. Configure Operations (Edit Only)
    operator_rates = OperatorRatesConfig(operation_rates={"edit": 1.0})

    # 4. Helpers
    prob_assigner = ProbabilityAssigner(strategy=prob_assigner_strategy)

    # 5. Create Strategy
    breeding_strategy = AgentCoderBreedingStrategy(
        operator=agent_operator,
        op_rates_config=operator_rates,
        pop_config=population_config,
        probability_assigner=prob_assigner,
        llm_workers=llm_workers,
    )

    # 6. Selector (TopK is sufficient/fastest for size 1)
    elite_selector: IEliteSelectionStrategy[CodeIndividual] = TopKEliteSelector()

    return CodeProfile(
        population_config=population_config,
        breeding_strategy=breeding_strategy,
        elite_selector=elite_selector,
    )
