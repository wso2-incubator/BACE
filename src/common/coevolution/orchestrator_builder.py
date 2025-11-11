"""
Builder pattern for constructing Orchestrator instances.

This module provides a fluent API for building Orchestrator objects,
simplifying the complex initialization process by providing:
- Method chaining for readability
- Clear separation of configuration and component setup
- Validation of required dependencies
- Default values for common configurations
"""

from typing import TYPE_CHECKING

from loguru import logger

from .core.interfaces import (
    BayesianConfig,
    CodePopulationConfig,
    EvolutionConfig,
    IBayesianSystem,
    ICodeOperator,
    IDatasetTestBlockBuilder,
    IExecutionSystem,
    IFeedbackGenerator,
    IPareto,
    IProbabilityAssigner,
    ISelectionStrategy,
    ITestBlockRebuilder,
    ITestOperator,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
    Sandbox,
)

if TYPE_CHECKING:
    from .core.individual import CodeIndividual, TestIndividual
    from .core.orchestrator import Orchestrator


class OrchestratorBuilder:
    """
    Builder for constructing Orchestrator instances with fluent API.

    Example usage:
        >>> builder = OrchestratorBuilder()
        >>> orchestrator = (
        ...     builder
        ...     .with_evolution_config(num_generations=10, random_seed=42)
        ...     .with_code_population_config(
        ...         initial_prior=0.5,
        ...         initial_population_size=10,
        ...         max_population_size=20,
        ...         elitism_rate=0.2,
        ...         offspring_rate=0.8
        ...     )
        ...     .with_test_population_config(initial_prior=0.5, initial_population_size=5)
        ...     .with_code_operator_rates(crossover_rate=0.3, mutation_rate=0.3, edit_rate=0.4)
        ...     .with_test_operator_rates(crossover_rate=0.3, mutation_rate=0.3, edit_rate=0.4)
        ...     .with_bayesian_config(alpha=0.9, beta=0.1, gamma=0.1, learning_rate=0.5)
        ...     .with_problem(problem)
        ...     .with_sandbox(sandbox)
        ...     .with_code_operator(code_operator)
        ...     .with_test_operator(test_operator)
        ...     .with_code_selector(code_selector)
        ...     .with_test_selector(test_selector)
        ...     .with_code_prob_assigner(code_prob_assigner)
        ...     .with_test_prob_assigner(test_prob_assigner)
        ...     .with_execution_system(execution_system)
        ...     .with_code_bayesian_system(code_bayesian_system)
        ...     .with_test_bayesian_system(test_bayesian_system)
        ...     .with_pareto(pareto)
        ...     .with_test_block_rebuilder(test_block_rebuilder)
        ...     .with_code_feedback_gen(code_feedback_gen)
        ...     .with_test_feedback_gen(test_feedback_gen)
        ...     .with_dataset_test_block_builder(dataset_test_block_builder)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize the builder with all fields set to None."""
        # Configuration objects
        self._evo_config: EvolutionConfig | None = None
        self._code_pop_config: CodePopulationConfig | None = None
        self._test_pop_config: PopulationConfig | None = None
        self._code_op_rates_config: OperatorRatesConfig | None = None
        self._test_op_rates_config: OperatorRatesConfig | None = None
        self._bayesian_config: BayesianConfig | None = None

        # Problem and sandbox
        self._problem: Problem | None = None
        self._sandbox: Sandbox | None = None

        # Injected components
        self._code_operator: ICodeOperator | None = None
        self._test_operator: ITestOperator | None = None
        self._code_selector: ISelectionStrategy | None = None
        self._test_selector: ISelectionStrategy | None = None
        self._code_prob_assigner: IProbabilityAssigner | None = None
        self._test_prob_assigner: IProbabilityAssigner | None = None
        self._execution_system: IExecutionSystem | None = None
        self._code_bayesian_system: IBayesianSystem | None = None
        self._test_bayesian_system: IBayesianSystem | None = None
        self._pareto: IPareto | None = None
        self._test_block_rebuilder: ITestBlockRebuilder | None = None
        self._code_feedback_gen: IFeedbackGenerator["TestIndividual"] | None = None
        self._test_feedback_gen: IFeedbackGenerator["CodeIndividual"] | None = None
        self._dataset_test_block_builder: IDatasetTestBlockBuilder | None = None

    # Configuration methods
    def with_evolution_config(
        self, num_generations: int, random_seed: int, max_workers: int = 1
    ) -> "OrchestratorBuilder":
        """
        Set the evolution configuration.

        Args:
            num_generations: Number of generations to run
            random_seed: Random seed for reproducibility
            max_workers: Number of parallel workers for breeding (default: 1 = sequential)

        Returns:
            Self for method chaining
        """
        self._evo_config = EvolutionConfig(
            num_generations=num_generations,
            random_seed=random_seed,
            max_workers=max_workers,
        )
        logger.debug(f"Set evolution config: {self._evo_config}")
        return self

    def with_code_population_config(
        self,
        initial_prior: float,
        initial_population_size: int,
        max_population_size: int,
        elitism_rate: float,
        offspring_rate: float,
    ) -> "OrchestratorBuilder":
        """
        Set the code population configuration.

        Args:
            initial_prior: Initial probability for code individuals
            initial_population_size: Initial size of code population
            max_population_size: Maximum size of code population
            elitism_rate: Fraction of population to keep as elites
            offspring_rate: Fraction of max population to generate as offspring

        Returns:
            Self for method chaining
        """
        self._code_pop_config = CodePopulationConfig(
            initial_prior=initial_prior,
            initial_population_size=initial_population_size,
            max_population_size=max_population_size,
            elitism_rate=elitism_rate,
            offspring_rate=offspring_rate,
        )
        logger.debug(f"Set code population config: {self._code_pop_config}")
        return self

    def with_test_population_config(
        self, initial_prior: float, initial_population_size: int
    ) -> "OrchestratorBuilder":
        """
        Set the test population configuration.

        Args:
            initial_prior: Initial probability for test individuals
            initial_population_size: Initial size of test population

        Returns:
            Self for method chaining
        """
        self._test_pop_config = PopulationConfig(
            initial_prior=initial_prior,
            initial_population_size=initial_population_size,
        )
        logger.debug(f"Set test population config: {self._test_pop_config}")
        return self

    def with_code_operator_rates(
        self, crossover_rate: float, mutation_rate: float, edit_rate: float
    ) -> "OrchestratorBuilder":
        """
        Set the code operator rates configuration.

        Args:
            crossover_rate: Rate of crossover operations
            mutation_rate: Rate of mutation operations
            edit_rate: Rate of edit operations

        Returns:
            Self for method chaining
        """
        self._code_op_rates_config = OperatorRatesConfig(
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            edit_rate=edit_rate,
        )
        logger.debug(f"Set code operator rates config: {self._code_op_rates_config}")
        return self

    def with_test_operator_rates(
        self, crossover_rate: float, mutation_rate: float, edit_rate: float
    ) -> "OrchestratorBuilder":
        """
        Set the test operator rates configuration.

        Args:
            crossover_rate: Rate of crossover operations
            mutation_rate: Rate of mutation operations
            edit_rate: Rate of edit operations

        Returns:
            Self for method chaining
        """
        self._test_op_rates_config = OperatorRatesConfig(
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            edit_rate=edit_rate,
        )
        logger.debug(f"Set test operator rates config: {self._test_op_rates_config}")
        return self

    def with_bayesian_config(
        self, alpha: float, beta: float, gamma: float, learning_rate: float
    ) -> "OrchestratorBuilder":
        """
        Set the Bayesian configuration.

        Args:
            alpha: P(pass | code correct, test incorrect)
            beta: P(pass | code incorrect, test correct)
            gamma: P(pass | code incorrect, test incorrect)
            learning_rate: Learning rate for belief updates

        Returns:
            Self for method chaining
        """
        self._bayesian_config = BayesianConfig(
            alpha=alpha, beta=beta, gamma=gamma, learning_rate=learning_rate
        )
        logger.debug(f"Set Bayesian config: {self._bayesian_config}")
        return self

    # Problem and sandbox methods
    def with_problem(self, problem: Problem) -> "OrchestratorBuilder":
        """
        Set the problem to solve.

        Args:
            problem: Problem containing question, test cases, etc.

        Returns:
            Self for method chaining
        """
        self._problem = problem
        logger.debug(f"Set problem: {problem.question_id}")
        return self

    def with_sandbox(self, sandbox: Sandbox) -> "OrchestratorBuilder":
        """
        Set the sandbox for code execution.

        Args:
            sandbox: Sandbox environment for executing code

        Returns:
            Self for method chaining
        """
        self._sandbox = sandbox
        logger.debug("Set sandbox")
        return self

    # Component methods
    def with_code_operator(self, operator: ICodeOperator) -> "OrchestratorBuilder":
        """
        Set the code operator.

        Args:
            operator: Code operator for genetic operations

        Returns:
            Self for method chaining
        """
        self._code_operator = operator
        logger.debug("Set code operator")
        return self

    def with_test_operator(self, operator: ITestOperator) -> "OrchestratorBuilder":
        """
        Set the test operator.

        Args:
            operator: Test operator for genetic operations

        Returns:
            Self for method chaining
        """
        self._test_operator = operator
        logger.debug("Set test operator")
        return self

    def with_code_selector(self, selector: ISelectionStrategy) -> "OrchestratorBuilder":
        """
        Set the code selection strategy.

        Args:
            selector: Selection strategy for code population

        Returns:
            Self for method chaining
        """
        self._code_selector = selector
        logger.debug("Set code selector")
        return self

    def with_test_selector(self, selector: ISelectionStrategy) -> "OrchestratorBuilder":
        """
        Set the test selection strategy.

        Args:
            selector: Selection strategy for test population

        Returns:
            Self for method chaining
        """
        self._test_selector = selector
        logger.debug("Set test selector")
        return self

    def with_code_prob_assigner(
        self, assigner: IProbabilityAssigner
    ) -> "OrchestratorBuilder":
        """
        Set the code probability assigner.

        Args:
            assigner: Probability assigner for code offspring

        Returns:
            Self for method chaining
        """
        self._code_prob_assigner = assigner
        logger.debug("Set code probability assigner")
        return self

    def with_test_prob_assigner(
        self, assigner: IProbabilityAssigner
    ) -> "OrchestratorBuilder":
        """
        Set the test probability assigner.

        Args:
            assigner: Probability assigner for test offspring

        Returns:
            Self for method chaining
        """
        self._test_prob_assigner = assigner
        logger.debug("Set test probability assigner")
        return self

    def with_execution_system(self, system: IExecutionSystem) -> "OrchestratorBuilder":
        """
        Set the execution system.

        Args:
            system: Execution system for running tests

        Returns:
            Self for method chaining
        """
        self._execution_system = system
        logger.debug("Set execution system")
        return self

    def with_code_bayesian_system(
        self, system: IBayesianSystem
    ) -> "OrchestratorBuilder":
        """
        Set the code Bayesian system.

        Args:
            system: Bayesian system for code belief updates

        Returns:
            Self for method chaining
        """
        self._code_bayesian_system = system
        logger.debug("Set code Bayesian system")
        return self

    def with_test_bayesian_system(
        self, system: IBayesianSystem
    ) -> "OrchestratorBuilder":
        """
        Set the test Bayesian system.

        Args:
            system: Bayesian system for test belief updates

        Returns:
            Self for method chaining
        """
        self._test_bayesian_system = system
        logger.debug("Set test Bayesian system")
        return self

    def with_pareto(self, pareto: IPareto) -> "OrchestratorBuilder":
        """
        Set the Pareto system.

        Args:
            pareto: Pareto system for multi-objective optimization

        Returns:
            Self for method chaining
        """
        self._pareto = pareto
        logger.debug("Set Pareto system")
        return self

    def with_test_block_rebuilder(
        self, rebuilder: ITestBlockRebuilder
    ) -> "OrchestratorBuilder":
        """
        Set the test block rebuilder.

        Args:
            rebuilder: Test block rebuilder for reconstructing test classes

        Returns:
            Self for method chaining
        """
        self._test_block_rebuilder = rebuilder
        logger.debug("Set test block rebuilder")
        return self

    def with_code_feedback_gen(
        self, generator: IFeedbackGenerator["TestIndividual"]
    ) -> "OrchestratorBuilder":
        """
        Set the code feedback generator.

        Args:
            generator: Feedback generator for code edits

        Returns:
            Self for method chaining
        """
        self._code_feedback_gen = generator
        logger.debug("Set code feedback generator")
        return self

    def with_test_feedback_gen(
        self, generator: IFeedbackGenerator["CodeIndividual"]
    ) -> "OrchestratorBuilder":
        """
        Set the test feedback generator.

        Args:
            generator: Feedback generator for test edits

        Returns:
            Self for method chaining
        """
        self._test_feedback_gen = generator
        logger.debug("Set test feedback generator")
        return self

    def with_dataset_test_block_builder(
        self, builder: IDatasetTestBlockBuilder
    ) -> "OrchestratorBuilder":
        """
        Set the dataset test block builder.

        Args:
            builder: Builder for creating test blocks from dataset test cases

        Returns:
            Self for method chaining
        """
        self._dataset_test_block_builder = builder
        logger.debug("Set dataset test block builder")
        return self

    def _validate(self) -> None:
        """
        Validate that all required components are set.

        Raises:
            ValueError: If any required component is missing
        """
        missing_components = []

        # Check configurations
        if self._evo_config is None:
            missing_components.append("evolution_config")
        if self._code_pop_config is None:
            missing_components.append("code_population_config")
        if self._test_pop_config is None:
            missing_components.append("test_population_config")
        if self._code_op_rates_config is None:
            missing_components.append("code_operator_rates_config")
        if self._test_op_rates_config is None:
            missing_components.append("test_operator_rates_config")
        if self._bayesian_config is None:
            missing_components.append("bayesian_config")

        # Check problem and sandbox
        if self._problem is None:
            missing_components.append("problem")
        if self._sandbox is None:
            missing_components.append("sandbox")

        # Check components
        if self._code_operator is None:
            missing_components.append("code_operator")
        if self._test_operator is None:
            missing_components.append("test_operator")
        if self._code_selector is None:
            missing_components.append("code_selector")
        if self._test_selector is None:
            missing_components.append("test_selector")
        if self._code_prob_assigner is None:
            missing_components.append("code_prob_assigner")
        if self._test_prob_assigner is None:
            missing_components.append("test_prob_assigner")
        if self._execution_system is None:
            missing_components.append("execution_system")
        if self._code_bayesian_system is None:
            missing_components.append("code_bayesian_system")
        if self._test_bayesian_system is None:
            missing_components.append("test_bayesian_system")
        if self._pareto is None:
            missing_components.append("pareto")
        if self._test_block_rebuilder is None:
            missing_components.append("test_block_rebuilder")
        if self._code_feedback_gen is None:
            missing_components.append("code_feedback_gen")
        if self._test_feedback_gen is None:
            missing_components.append("test_feedback_gen")
        if self._dataset_test_block_builder is None:
            missing_components.append("dataset_test_block_builder")

        if missing_components:
            raise ValueError(
                f"Cannot build Orchestrator: missing required components: {', '.join(missing_components)}"
            )

    def build(self) -> "Orchestrator":
        """
        Build and return the configured Orchestrator instance.

        Returns:
            Configured Orchestrator instance

        Raises:
            ValueError: If any required component is missing
        """
        logger.info("Building Orchestrator...")
        self._validate()

        # Import here to avoid circular dependency
        from .core.orchestrator import Orchestrator

        # All fields are guaranteed to be non-None after _validate()
        assert self._evo_config is not None
        assert self._code_pop_config is not None
        assert self._test_pop_config is not None
        assert self._code_op_rates_config is not None
        assert self._test_op_rates_config is not None
        assert self._bayesian_config is not None
        assert self._problem is not None
        assert self._sandbox is not None
        assert self._code_operator is not None
        assert self._test_operator is not None
        assert self._code_selector is not None
        assert self._test_selector is not None
        assert self._code_prob_assigner is not None
        assert self._test_prob_assigner is not None
        assert self._execution_system is not None
        assert self._code_bayesian_system is not None
        assert self._test_bayesian_system is not None
        assert self._pareto is not None
        assert self._test_block_rebuilder is not None
        assert self._code_feedback_gen is not None
        assert self._test_feedback_gen is not None
        assert self._dataset_test_block_builder is not None

        orchestrator = Orchestrator(
            evo_config=self._evo_config,
            code_pop_config=self._code_pop_config,
            test_pop_config=self._test_pop_config,
            code_op_rates_config=self._code_op_rates_config,
            test_op_rates_config=self._test_op_rates_config,
            bayesian_config=self._bayesian_config,
            problem=self._problem,
            sandbox=self._sandbox,
            code_operator=self._code_operator,
            test_operator=self._test_operator,
            code_selector=self._code_selector,
            test_selector=self._test_selector,
            code_prob_assigner=self._code_prob_assigner,
            test_prob_assigner=self._test_prob_assigner,
            execution_system=self._execution_system,
            code_bayesian_system=self._code_bayesian_system,
            test_bayesian_system=self._test_bayesian_system,
            pareto=self._pareto,
            test_block_rebuilder=self._test_block_rebuilder,
            code_feedback_gen=self._code_feedback_gen,
            test_feedback_gen=self._test_feedback_gen,
            dataset_test_block_builder=self._dataset_test_block_builder,
        )

        logger.info("Orchestrator built successfully")
        return orchestrator
