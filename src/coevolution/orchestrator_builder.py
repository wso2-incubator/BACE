"""
Orchestrator builder with fluent API and profile factory methods.

This module provides:
1. OrchestratorBuilder: Fluent API for constructing Orchestrator instances
2. Profile factory functions: Pre-configured profiles for common scenarios
3. Validation: Ensures all required components are present before building

The builder supports flexible assembly of coevolution systems with multiple
population types, breeding strategies, and configurations while maintaining
type safety and configuration clarity.

Example Usage:
    >>> from common.coevolution.orchestrator_builder import OrchestratorBuilder
    >>> from common.coevolution.orchestrator_builder import (
    ...     create_default_code_profile,
    ...     create_unittest_test_profile,
    ...     create_public_test_profile
    ... )
    >>>
    >>> # Create profiles using factories
    >>> code_profile = create_default_code_profile(llm_client, sandbox)
    >>> unittest_profile = create_unittest_test_profile(llm_client)
    >>> public_profile = create_public_test_profile()
    >>>
    >>> # Build orchestrator
    >>> builder = OrchestratorBuilder()
    >>> config = (
    ...     builder
    ...     .with_evolution_config(num_generations=10, max_workers=4)
    ...     .with_code_profile(code_profile)
    ...     .add_test_profile("unittest", unittest_profile)
    ...     .with_public_test_profile(public_profile)
    ...     .with_execution_system(execution_system)
    ...     .with_bayesian_system(bayesian_system)
    ...     .with_ledger_factory(ledger_factory)
    ...     .with_test_block_rebuilder(test_rebuilder)
    ...     .with_dataset_test_block_builder(dataset_builder)
    ...     .build()
    ... )
    >>>
    >>> # Create orchestrator from config
    >>> from common.coevolution.core.orchestrator import Orchestrator
    >>> orchestrator = Orchestrator(
    ...     evo_config=config.evo_config,
    ...     code_profile=config.code_profile,
    ...     evolved_test_profiles=config.evolved_test_profiles,
    ...     public_test_profile=config.public_test_profile,
    ...     execution_system=config.execution_system,
    ...     bayesian_system=config.bayesian_system,
    ...     ledger_factory=config.ledger_factory,
    ...     test_block_rebuilder=config.test_block_rebuilder,
    ...     dataset_test_block_builder=config.dataset_test_block_builder,
    ... )
"""

from typing import Any

from loguru import logger

from infrastructure.llm_client import LLMClient
from infrastructure.sandbox import SafeCodeSandbox

from .breeding_strategies.code_breeding import CodeBreedingStrategy
from .breeding_strategies.differential_breeding import DifferentialBreedingStrategy
from .breeding_strategies.unittest_breeding import UnittestBreedingStrategy
from .core.individual import CodeIndividual, TestIndividual
from .core.interfaces import (
    BayesianConfig,
    CodeProfile,
    EvolutionConfig,
    IBeliefUpdater,
    IDatasetTestBlockBuilder,
    IEliteSelectionStrategy,
    IExecutionSystem,
    ITestBlockRebuilder,
    LedgerFactory,
    OperatorRatesConfig,
    PopulationConfig,
    PublicTestProfile,
    TestProfile,
)
from .core.scheduling import EvolutionSchedule
from .ledger import InteractionLedger
from .operators.code_llm_operator import CodeLLMOperator
from .operators.differential_llm_operator import DifferentialLLMOperator
from .operators.unittest_llm_operator import UnittestLLMOperator
from .orchestrator_config import OrchestratorConfig
from .probability_assigner import ProbabilityAssigner
from .selection_strategies.elite_selection import (
    CodeDiversityEliteSelector,
    TestDiversityEliteSelector,
    TopKEliteSelector,
)
from .selection_strategies.failing_test_selection import FailingTestSelector
from .selection_strategies.functionally_eq_selection import FunctionallyEqSelector
from .selection_strategies.parent_selection import RouletteWheelParentSelection
from .tools.differential_finder import DifferentialFinder

# =========================================================================
# Profile Factory Functions
# =========================================================================


def create_default_code_profile(
    llm_client: LLMClient,
    sandbox: SafeCodeSandbox,
    initial_prior: float = 0.2,
    initial_population_size: int = 10,
    max_population_size: int = 15,
    offspring_rate: float = 0.8,
    elitism_rate: float = 0.2,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.2,
    edit_rate: float = 0.6,
    max_workers: int = 10,
    diversity_enabled: bool = True,
    prob_assigner_strategy: str = "min",
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
        max_workers: Parallel workers for breeding (default: 1)
        diversity_enabled: Use diversity selector vs simple top-k (default: True)

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
    code_operator = CodeLLMOperator(llm=llm_client)

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
        max_workers=max_workers,
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
    max_workers: int = 1,
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
        max_workers: Parallel workers for breeding (default: 1)

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
    test_operator = UnittestLLMOperator(llm=llm_client)

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
    prob_assigner = ProbabilityAssigner()

    # Create breeding strategy
    breeding_strategy = UnittestBreedingStrategy(
        operator=test_operator,
        op_rates_config=operator_rates,
        pop_config=population_config,
        probability_assigner=prob_assigner,
        parent_selector=parent_selector,
        max_workers=max_workers,
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
    sandbox: SafeCodeSandbox,
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
    max_workers: int = 1,
    prob_assigner_strategy: str = "min",
    diversity_enabled: bool = True,
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
        max_workers: Parallel workers for breeding (default: 1)

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
    differential_operator = DifferentialLLMOperator(llm=llm_client)

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

    # differential finder
    differential_finder = DifferentialFinder(sandbox=sandbox)

    # Create breeding strategy
    breeding_strategy = DifferentialBreedingStrategy(
        operator=differential_operator,
        op_rates_config=operator_rates,
        pop_config=population_config,
        probability_assigner=prob_assigner,
        parent_selector=parent_selector,
        functionally_equivalent_code_selector=FunctionallyEqSelector(),
        differential_finder=differential_finder,
        max_workers=max_workers,
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


# =========================================================================
# OrchestratorBuilder
# =========================================================================


class OrchestratorBuilder:
    """
    Fluent API builder for constructing OrchestratorConfig instances.

    This builder provides a step-by-step interface for assembling all the
    components needed to create an Orchestrator. It supports:
    - Profile-based configuration (code, test profiles)
    - Multiple test population types with dynamic registration
    - Infrastructure components (execution, Bayesian systems)
    - Comprehensive validation before building

    The builder uses optional attributes internally and validates that all
    required components are set before building the final configuration.

    Example:
        >>> builder = OrchestratorBuilder()
        >>> config = (
        ...     builder
        ...     .with_evolution_config(schedule=EvolutionSchedule.simultaneous(10))
        ...     .with_code_profile(code_profile)
        ...     .add_test_profile("unittest", unittest_profile)
        ...     .add_test_profile("differential", differential_profile)
        ...     .with_public_test_profile(public_profile)
        ...     .with_execution_system(execution_system)
        ...     .with_bayesian_system(bayesian_system)
        ...     .with_ledger_factory(ledger_factory)
        ...     .with_test_block_rebuilder(test_rebuilder)
        ...     .with_dataset_test_block_builder(dataset_builder)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize builder with empty state."""
        # Configuration
        self._evo_config: EvolutionConfig | None = None

        # Profiles
        self._code_profile: CodeProfile | None = None
        self._evolved_test_profiles: dict[str, TestProfile] = {}
        self._public_test_profile: PublicTestProfile | None = None

        # Infrastructure
        self._execution_system: IExecutionSystem | None = None
        self._bayesian_system: IBeliefUpdater | None = None
        self._ledger_factory: LedgerFactory = InteractionLedger  # Default factory
        self._test_block_rebuilder: ITestBlockRebuilder | None = None
        self._dataset_test_block_builder: IDatasetTestBlockBuilder | None = None

    def with_evolution_config(
        self,
        schedule: EvolutionSchedule,
    ) -> "OrchestratorBuilder":
        """
        Set evolution configuration.
        Args:
            schedule: Evolution schedule (simultaneous or alternating)
        Returns:
            Self for method chaining
        """
        self._evo_config = EvolutionConfig(schedule=schedule)
        return self

    def with_code_profile(self, profile: CodeProfile) -> "OrchestratorBuilder":
        """
        Set code population profile.

        Args:
            profile: Complete code population configuration

        Returns:
            Self for method chaining
        """
        self._code_profile = profile
        return self

    def add_test_profile(
        self,
        test_type: str,
        profile: TestProfile,
    ) -> "OrchestratorBuilder":
        """
        Add an evolved test population profile.

        Args:
            test_type: Unique key for this test population (e.g., "unittest", "differential")
            profile: Complete test population configuration

        Returns:
            Self for method chaining

        Raises:
            ValueError: If test_type is "public" or "private" (reserved for fixed tests)
        """
        if test_type in {"public", "private"}:
            raise ValueError(
                f"Test type '{test_type}' is reserved for fixed test populations. "
                f"Use evolved test types like 'unittest', 'differential', 'property'."
            )

        if test_type in self._evolved_test_profiles:
            logger.warning(f"Overwriting existing test profile for type '{test_type}'")

        self._evolved_test_profiles[test_type] = profile
        return self

    def with_public_test_profile(
        self, profile: PublicTestProfile
    ) -> "OrchestratorBuilder":
        """
        Set public test profile.

        Args:
            profile: Public/ground-truth test configuration

        Returns:
            Self for method chaining
        """
        self._public_test_profile = profile
        return self

    def with_execution_system(
        self, execution_system: IExecutionSystem
    ) -> "OrchestratorBuilder":
        """
        Set execution system for running code against tests.

        Args:
            execution_system: System for code-test execution

        Returns:
            Self for method chaining
        """
        self._execution_system = execution_system
        return self

    def with_bayesian_system(
        self, bayesian_system: IBeliefUpdater
    ) -> "OrchestratorBuilder":
        """
        Set Bayesian system for belief updates.

        Args:
            bayesian_system: System for belief management

        Returns:
            Self for method chaining
        """
        self._bayesian_system = bayesian_system
        return self

    def with_ledger_factory(
        self, ledger_factory: LedgerFactory
    ) -> "OrchestratorBuilder":
        """
        Set ledger factory for creating interaction ledgers.

        Args:
            ledger_factory: Factory for creating fresh interaction ledgers

        Returns:
            Self for method chaining
        """
        self._ledger_factory = ledger_factory
        return self

    def with_test_block_rebuilder(
        self, test_block_rebuilder: ITestBlockRebuilder
    ) -> "OrchestratorBuilder":
        """
        Set test block rebuilder for reconstructing test class blocks.

        Args:
            test_block_rebuilder: Rebuilder for test class blocks

        Returns:
            Self for method chaining
        """
        self._test_block_rebuilder = test_block_rebuilder
        return self

    def with_dataset_test_block_builder(
        self, dataset_test_block_builder: IDatasetTestBlockBuilder
    ) -> "OrchestratorBuilder":
        """
        Set dataset test block builder for creating tests from dataset.

        Args:
            dataset_test_block_builder: Builder for dataset test blocks

        Returns:
            Self for method chaining
        """
        self._dataset_test_block_builder = dataset_test_block_builder
        return self

    def _validate(self) -> None:
        """
        Validate that all required components are set.

        Raises:
            ValueError: If any required component is missing or invalid
        """
        errors: list[str] = []

        # Check required configuration
        if self._evo_config is None:
            errors.append("Evolution config not set (use with_evolution_config())")

        # Check required profiles
        if self._code_profile is None:
            errors.append("Code profile not set (use with_code_profile())")

        if self._public_test_profile is None:
            errors.append(
                "Public test profile not set (use with_public_test_profile())"
            )

        # Check at least one evolved population exists
        if not self._evolved_test_profiles:
            logger.warning(
                "No evolved test populations configured. "
                "Consider adding unittest or differential test profiles."
            )

        # Check required infrastructure
        if self._execution_system is None:
            errors.append("Execution system not set (use with_execution_system())")

        if self._bayesian_system is None:
            errors.append("Bayesian system not set (use with_bayesian_system())")

        if self._test_block_rebuilder is None:
            errors.append(
                "Test block rebuilder not set (use with_test_block_rebuilder())"
            )

        if self._dataset_test_block_builder is None:
            errors.append(
                "Dataset test block builder not set (use with_dataset_test_block_builder())"
            )
        if errors:
            error_msg = "Validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

    def build(self) -> OrchestratorConfig:
        """
        Build and return the final OrchestratorConfig.

        Returns:
            Validated OrchestratorConfig ready for Orchestrator construction

        Raises:
            ValueError: If validation fails
        """
        self._validate()

        # Type narrowing: After validation, we know these are not None
        assert self._evo_config is not None
        assert self._code_profile is not None
        assert self._public_test_profile is not None
        assert self._execution_system is not None
        assert self._bayesian_system is not None
        assert self._ledger_factory is not None
        assert self._test_block_rebuilder is not None
        assert self._dataset_test_block_builder is not None

        config = OrchestratorConfig(
            evo_config=self._evo_config,
            code_profile=self._code_profile,
            evolved_test_profiles=self._evolved_test_profiles,
            public_test_profile=self._public_test_profile,
            execution_system=self._execution_system,
            bayesian_system=self._bayesian_system,
            ledger_factory=self._ledger_factory,
            test_block_rebuilder=self._test_block_rebuilder,
            dataset_test_block_builder=self._dataset_test_block_builder,
        )

        logger.info("OrchestratorConfig built successfully")
        logger.info(f"  Generations: {config.evo_config.num_generations}")
        logger.info(
            f"  Code population size: {config.code_profile.population_config.initial_population_size} → "
            f"{config.code_profile.population_config.max_population_size}"
        )
        logger.info(
            f"  Evolved test populations: {list(config.evolved_test_profiles.keys())}"
        )

        return config


# =========================================================================
# Convenience Function
# =========================================================================


def build_orchestrator_from_config(config: OrchestratorConfig) -> Any:
    """
    Convenience function to create Orchestrator from OrchestratorConfig.

    Args:
        config: Validated orchestrator configuration

    Returns:
        Configured Orchestrator instance
    """
    from .core.orchestrator import Orchestrator

    return Orchestrator(
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
