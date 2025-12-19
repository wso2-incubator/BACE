"""
Tests for the OrchestratorBuilder.

This test suite verifies that the builder pattern implementation
correctly validates and constructs Orchestrator instances.
"""

import pytest

from common.coevolution.core.interfaces import (
    BayesianConfig,
    CodePopulationConfig,
    EvolutionConfig,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
    Test,
)
from common.coevolution.deprecated.orchestrator_builder import OrchestratorBuilder


class TestOrchestratorBuilder:
    """Test suite for OrchestratorBuilder."""

    def test_builder_validates_missing_components(self) -> None:
        """Test that builder raises ValueError when required components are missing."""
        builder = OrchestratorBuilder()

        with pytest.raises(ValueError) as exc_info:
            builder.build()

        error_message = str(exc_info.value)
        assert "missing required components" in error_message.lower()
        # Should list at least some of the missing components
        assert "evolution_config" in error_message

    def test_builder_with_evolution_config(self) -> None:
        """Test setting evolution configuration."""
        builder = OrchestratorBuilder()
        result = builder.with_evolution_config(num_generations=10, random_seed=42)

        assert result is builder  # Test method chaining
        assert builder._evo_config is not None
        assert builder._evo_config.num_generations == 10
        assert builder._evo_config.random_seed == 42

    def test_builder_with_code_population_config(self) -> None:
        """Test setting code population configuration."""
        builder = OrchestratorBuilder()
        result = builder.with_code_population_config(
            initial_prior=0.5,
            initial_population_size=10,
            max_population_size=20,
            elitism_rate=0.2,
            offspring_rate=0.8,
        )

        assert result is builder
        assert builder._code_pop_config is not None
        assert builder._code_pop_config.initial_prior == 0.5
        assert builder._code_pop_config.initial_population_size == 10
        assert builder._code_pop_config.max_population_size == 20
        assert builder._code_pop_config.elitism_rate == 0.2
        assert builder._code_pop_config.offspring_rate == 0.8

    def test_builder_with_test_population_config(self) -> None:
        """Test setting test population configuration."""
        builder = OrchestratorBuilder()
        result = builder.with_test_population_config(
            initial_prior=0.5, initial_population_size=5
        )

        assert result is builder
        assert builder._test_pop_config is not None
        assert builder._test_pop_config.initial_prior == 0.5
        assert builder._test_pop_config.initial_population_size == 5

    def test_builder_with_operator_rates(self) -> None:
        """Test setting operator rates configurations."""
        builder = OrchestratorBuilder()

        # Code operator rates
        result = builder.with_code_operator_rates(
            crossover_rate=0.3, mutation_rate=0.3, edit_rate=0.4
        )
        assert result is builder
        assert builder._code_op_rates_config is not None
        assert builder._code_op_rates_config.crossover_rate == 0.3
        assert builder._code_op_rates_config.mutation_rate == 0.3
        assert builder._code_op_rates_config.edit_rate == 0.4

        # Test operator rates
        result = builder.with_test_operator_rates(
            crossover_rate=0.2, mutation_rate=0.4, edit_rate=0.4
        )
        assert result is builder
        assert builder._test_op_rates_config is not None
        assert builder._test_op_rates_config.crossover_rate == 0.2
        assert builder._test_op_rates_config.mutation_rate == 0.4
        assert builder._test_op_rates_config.edit_rate == 0.4

    def test_builder_with_bayesian_config(self) -> None:
        """Test setting Bayesian configuration."""
        builder = OrchestratorBuilder()
        result = builder.with_bayesian_config(
            alpha=0.9, beta=0.1, gamma=0.1, learning_rate=0.5
        )

        assert result is builder
        assert builder._bayesian_config is not None
        assert builder._bayesian_config.alpha == 0.9
        assert builder._bayesian_config.beta == 0.1
        assert builder._bayesian_config.gamma == 0.1
        assert builder._bayesian_config.learning_rate == 0.5

    def test_builder_with_problem(self) -> None:
        """Test setting problem."""
        builder = OrchestratorBuilder()
        problem = Problem(
            question_title="Test Problem",
            question_content="Solve this",
            question_id="test-1",
            starter_code="def solution(): pass",
            public_test_cases=[Test(input="1", output="1")],
            private_test_cases=[Test(input="2", output="2")],
        )

        result = builder.with_problem(problem)
        assert result is builder
        assert builder._problem is problem

    def test_builder_with_sandbox(self) -> None:
        """Test setting sandbox."""
        builder = OrchestratorBuilder()
        sandbox = object()  # Mock sandbox

        result = builder.with_sandbox(sandbox)
        assert result is builder
        assert builder._sandbox is sandbox

    def test_builder_method_chaining(self) -> None:
        """Test that all methods support method chaining."""
        builder = OrchestratorBuilder()

        # Test chaining multiple configuration methods
        result = (
            builder.with_evolution_config(num_generations=5, random_seed=123)
            .with_code_population_config(
                initial_prior=0.6,
                initial_population_size=8,
                max_population_size=16,
                elitism_rate=0.25,
                offspring_rate=0.75,
            )
            .with_test_population_config(initial_prior=0.6, initial_population_size=4)
            .with_bayesian_config(alpha=0.85, beta=0.15, gamma=0.05, learning_rate=0.6)
        )

        assert result is builder
        assert builder._evo_config is not None
        assert builder._code_pop_config is not None
        assert builder._test_pop_config is not None
        assert builder._bayesian_config is not None

    def test_builder_validates_all_configs_present(self) -> None:
        """Test that builder validates all configuration objects are set."""
        builder = OrchestratorBuilder()

        # Set evolution config only
        builder.with_evolution_config(num_generations=5, random_seed=42)

        with pytest.raises(ValueError) as exc_info:
            builder.build()

        error_message = str(exc_info.value)
        # Should still complain about missing components
        assert "missing required components" in error_message.lower()

    def test_config_dataclasses_validate_inputs(self) -> None:
        """Test that configuration dataclasses validate their inputs."""
        # Test EvolutionConfig validation
        with pytest.raises(ValueError):
            EvolutionConfig(num_generations=0, random_seed=42)

        # Test BayesianConfig validation
        with pytest.raises(ValueError):
            BayesianConfig(alpha=1.5, beta=0.1, gamma=0.1, learning_rate=0.5)

        with pytest.raises(ValueError):
            BayesianConfig(alpha=0.9, beta=0.1, gamma=0.1, learning_rate=0.0)

        # Test PopulationConfig validation
        with pytest.raises(ValueError):
            PopulationConfig(initial_prior=1.5, initial_population_size=10)

        with pytest.raises(ValueError):
            PopulationConfig(initial_prior=0.5, initial_population_size=-1)

        # Test CodePopulationConfig validation
        with pytest.raises(ValueError):
            CodePopulationConfig(
                initial_prior=0.5,
                initial_population_size=10,
                max_population_size=20,
                elitism_rate=1.5,
                offspring_rate=0.8,
            )

        # Test OperatorRatesConfig validation
        with pytest.raises(ValueError):
            OperatorRatesConfig(crossover_rate=1.5, mutation_rate=0.3, edit_rate=0.4)
