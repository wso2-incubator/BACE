"""
Orchestrator configuration dataclass for bundling all orchestrator dependencies.

This module provides a single dataclass that encapsulates all the components
needed to construct an Orchestrator, making initialization cleaner and more
maintainable than passing dozens of individual parameters.
"""

from dataclasses import dataclass

from .core.interfaces import (
    CodeProfile,
    EvolutionConfig,
    IBayesianSystem,
    IDatasetTestBlockBuilder,
    IExecutionSystem,
    ITestBlockRebuilder,
    PublicTestProfile,
    TestProfile,
)


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
        test_block_rebuilder: Rebuilds test class blocks from method snippets
        dataset_test_block_builder: Builds test blocks from dataset test cases

    Example:
        config = OrchestratorConfig(
            evo_config=EvolutionConfig(num_generations=10, max_workers=4),
            code_profile=my_code_profile,
            evolved_test_profiles={
                "unittest": unittest_profile,
                "differential": differential_profile,
            },
            public_test_profile=public_profile,
            execution_system=execution_system,
            bayesian_system=bayesian_system,
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
            test_block_rebuilder=config.test_block_rebuilder,
            dataset_test_block_builder=config.dataset_test_block_builder,
        )
    """

    # Top-level configuration
    evo_config: EvolutionConfig

    # Population profiles
    code_profile: CodeProfile
    evolved_test_profiles: dict[str, TestProfile]
    public_test_profile: PublicTestProfile

    # Global infrastructure systems
    execution_system: IExecutionSystem
    bayesian_system: IBayesianSystem
    test_block_rebuilder: ITestBlockRebuilder
    dataset_test_block_builder: IDatasetTestBlockBuilder
