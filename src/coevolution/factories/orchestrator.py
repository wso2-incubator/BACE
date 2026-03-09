"""
Orchestrator builder with fluent API for constructing Orchestrator instances.

This module provides:
- OrchestratorBuilder: Fluent API for step-by-step construction
- build_orchestrator_from_config: Convenience function for final assembly
- Validation: Ensures all required components are present before building

The builder supports flexible assembly of coevolution systems with multiple
population types, breeding strategies, and configurations while maintaining
type safety and configuration clarity.

Example Usage:
    >>> from coevolution.factories import OrchestratorBuilder
    >>> from coevolution.factories import (
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
    ...     .build()
    ... )
"""

from typing import Any

from loguru import logger

from ..core.interfaces import (
    CodeProfile,
    EvolutionConfig,
    EvolutionSchedule,
    IBeliefUpdater,
    IExecutionSystem,
    LedgerFactory,
    OrchestratorConfig,
    PublicTestProfile,
    TestProfile,
)
from ..core.interfaces.language import IScriptComposer
from ..services.ledger import InteractionLedger


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
        self._composer: IScriptComposer | None = None

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

        The ORDER in which you call this method determines the order of probability
        updates during cooperative evolution. Test types are updated in the order
        they were added (after public tests which always update first).

        Example:
            builder.add_test_profile("differential", diff_profile)  # Updates first
            builder.add_test_profile("unittest", unit_profile)      # Updates second
            # Result: public → differential → unittest

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

    def with_composer(self, composer: IScriptComposer) -> "OrchestratorBuilder":
        """
        Set script composer.

        Args:
            composer: Generative tests operations

        Returns:
            Self for method chaining
        """
        self._composer = composer
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

        if self._composer is None:
            errors.append("Composer not set (use with_composer())")

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
        assert self._composer is not None

        config = OrchestratorConfig(
            evo_config=self._evo_config,
            code_profile=self._code_profile,
            evolved_test_profiles=self._evolved_test_profiles,
            public_test_profile=self._public_test_profile,
            execution_system=self._execution_system,
            bayesian_system=self._bayesian_system,
            ledger_factory=self._ledger_factory,
            composer=self._composer,
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


def build_orchestrator_from_config(config: OrchestratorConfig) -> Any:
    """
    Convenience function to create Orchestrator from OrchestratorConfig.

    Args:
        config: Validated orchestrator configuration

    Returns:
        Configured Orchestrator instance
    """
    from ..core.orchestrator import Orchestrator

    return Orchestrator(
        evo_config=config.evo_config,
        code_profile=config.code_profile,
        evolved_test_profiles=config.evolved_test_profiles,
        public_test_profile=config.public_test_profile,
        execution_system=config.execution_system,
        bayesian_system=config.bayesian_system,
        ledger_factory=config.ledger_factory,
        composer=config.composer,
    )
