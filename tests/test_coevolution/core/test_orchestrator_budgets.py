from unittest.mock import MagicMock

import pytest

from coevolution.core.interfaces.config import EvolutionConfig, PopulationConfig
from coevolution.core.interfaces.context import CoevolutionContext
from coevolution.core.interfaces.profiles import (
    CodeProfile,
    PublicTestProfile,
    TestProfile,
)
from coevolution.core.orchestrator import Orchestrator


@pytest.fixture
def mock_orchestrator() -> Orchestrator:
    # Setup minimal mocks for Orchestrator initialization
    evo_config = MagicMock(spec=EvolutionConfig)
    code_profile = MagicMock(spec=CodeProfile)
    evolved_test_profiles: dict[str, TestProfile] = {}
    public_test_profile = MagicMock(spec=PublicTestProfile)
    execution_system = MagicMock()
    bayesian_system = MagicMock()
    ledger_factory = MagicMock()
    composer = MagicMock()

    orchestrator = Orchestrator(
        evo_config=evo_config,
        code_profile=code_profile,
        evolved_test_profiles=evolved_test_profiles,
        public_test_profile=public_test_profile,
        execution_system=execution_system,
        bayesian_system=bayesian_system,
        ledger_factory=ledger_factory,
        composer=composer,
    )
    return orchestrator


def test_breed_tests_respects_offspring_rate(mock_orchestrator: Orchestrator) -> None:
    # Setup test population profile
    test_type = "property"
    max_size = 30
    offspring_rate = 0.2
    num_elites = 4

    pop_config = PopulationConfig(
        initial_prior=0.5,
        initial_population_size=10,
        max_population_size=max_size,
        offspring_rate=offspring_rate,
        elitism_rate=0.5,
    )

    mock_breeder = MagicMock()
    mock_breeder.breed.return_value = [MagicMock()] * 6  # Expected: 30 * 0.2 = 6

    test_profile = MagicMock(spec=TestProfile)
    test_profile.population_config = pop_config
    test_profile.breeder = mock_breeder

    mock_orchestrator.evolved_test_profiles[test_type] = test_profile

    context = MagicMock(spec=CoevolutionContext)

    # Execute breeding
    offspring = mock_orchestrator._breed_tests(
        context, test_type, num_elites=num_elites
    )

    # Verify offspring count
    expected_offspring_count = 6  # min(int(30 * 0.2), 30 - 4) = 6
    assert len(offspring) == 6
    mock_breeder.breed.assert_called_once_with(context, expected_offspring_count)


def test_breed_tests_caps_at_max_capacity(mock_orchestrator: Orchestrator) -> None:
    # Setup test population profile
    test_type = "unittest"
    max_size = 20
    offspring_rate = 1.0  # Wants to fill completely
    num_elites = 15  # Only 5 spots left

    pop_config = PopulationConfig(
        initial_prior=0.5,
        initial_population_size=20,
        max_population_size=max_size,
        offspring_rate=offspring_rate,
        elitism_rate=0.5,
    )

    mock_breeder = MagicMock()
    mock_breeder.breed.return_value = [MagicMock()] * 5

    test_profile = MagicMock(spec=TestProfile)
    test_profile.population_config = pop_config
    test_profile.breeder = mock_breeder

    mock_orchestrator.evolved_test_profiles[test_type] = test_profile

    context = MagicMock(spec=CoevolutionContext)

    # Execute breeding
    offspring = mock_orchestrator._breed_tests(
        context, test_type, num_elites=num_elites
    )

    # Verify offspring count
    expected_offspring_count = 5  # min(int(20 * 1.0), 20 - 15) = 5
    assert len(offspring) == 5
    mock_breeder.breed.assert_called_once_with(context, expected_offspring_count)


def test_breed_code_still_works_correctly(mock_orchestrator: Orchestrator) -> None:
    # Setup code population profile
    max_size = 15
    offspring_rate = 0.3
    num_elites = 5

    pop_config = PopulationConfig(
        initial_prior=0.5,
        initial_population_size=10,
        max_population_size=max_size,
        offspring_rate=offspring_rate,
        elitism_rate=0.5,
    )

    mock_breeder = MagicMock()
    mock_breeder.breed.return_value = [MagicMock()] * 4  # 15 * 0.3 = 4.5 -> 4

    mock_orchestrator.code_profile.population_config = pop_config  # type: ignore
    mock_orchestrator.code_profile.breeder = mock_breeder  # type: ignore

    context = MagicMock(spec=CoevolutionContext)

    # Execute breeding
    offspring = mock_orchestrator._breed_code(context, num_elites=num_elites)

    # Verify offspring count
    expected_offspring_count = 4  # min(int(15 * 0.3), 15 - 5) = min(4, 10) = 4
    assert len(offspring) == 4
    mock_breeder.breed.assert_called_once_with(context, expected_offspring_count)

    # Use a fresh mock for second call to avoid any interference
    mock_breeder_2 = MagicMock()
    mock_breeder_2.breed.return_value = [MagicMock()] * 4
    mock_orchestrator.code_profile.breeder = mock_breeder_2  # type: ignore

    context_2 = MagicMock(spec=CoevolutionContext)

    # Execute breeding again
    offspring_2 = mock_orchestrator._breed_code(context_2, num_elites=num_elites)

    # Verify offspring count
    assert len(offspring_2) == 4
    mock_breeder_2.breed.assert_called_once_with(context_2, expected_offspring_count)
