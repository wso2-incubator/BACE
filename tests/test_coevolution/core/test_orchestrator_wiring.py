from unittest.mock import MagicMock

from coevolution.core.interfaces.config import EvolutionConfig
from coevolution.core.interfaces.operators import RegisteredOperator
from coevolution.core.interfaces.profiles import (
    CodeProfile,
    PublicTestProfile,
    TestProfile,
)
from coevolution.core.orchestrator import Orchestrator
from coevolution.strategies.breeding.breeder import Breeder


def test_orchestrator_wire_repair_operators() -> None:
    """
    Verifies that the Orchestrator collects repair_operators from all evolved test profiles
    and injects them into the code population's breeder.
    """
    # 1. Setup Mocks
    evo_config = MagicMock(spec=EvolutionConfig)

    # Mock Code Breeder
    mock_code_breeder = MagicMock(spec=Breeder)
    code_profile = MagicMock(spec=CodeProfile)
    code_profile.breeder = mock_code_breeder

    # Mock Test Populations with repair operators
    mock_op1 = MagicMock(spec=RegisteredOperator)
    mock_op2 = MagicMock(spec=RegisteredOperator)
    mock_op3 = MagicMock(spec=RegisteredOperator)

    test_profile_1 = MagicMock(spec=TestProfile)
    test_profile_1.repair_operators = [mock_op1, mock_op2]

    test_profile_2 = MagicMock(spec=TestProfile)
    test_profile_2.repair_operators = [mock_op3]

    test_profile_3 = MagicMock(spec=TestProfile)
    test_profile_3.repair_operators = []  # No operators

    evolved_test_profiles: dict[str, TestProfile] = {
        "unittest": test_profile_1,
        "property": test_profile_2,
        "differential": test_profile_3,
    }

    public_test_profile = MagicMock(spec=PublicTestProfile)

    # Other dependencies
    execution_system = MagicMock()
    bayesian_system = MagicMock()
    ledger_factory = MagicMock()
    composer = MagicMock()

    # 2. Initialize Orchestrator (this triggers _wire_repair_operators)
    Orchestrator(
        evo_config=evo_config,
        code_profile=code_profile,
        evolved_test_profiles=evolved_test_profiles,
        public_test_profile=public_test_profile,
        execution_system=execution_system,
        bayesian_system=bayesian_system,
        ledger_factory=ledger_factory,
        composer=composer,
    )

    # 3. Verify
    # Should be called once with all accumulated operators
    mock_code_breeder.add_operators.assert_called_once()
    added_ops = mock_code_breeder.add_operators.call_args[0][0]

    assert len(added_ops) == 3
    assert mock_op1 in added_ops
    assert mock_op2 in added_ops
    assert mock_op3 in added_ops


def test_orchestrator_wire_repair_operators_no_ops() -> None:
    """Verifies orchestrator doesn't crash or call add_operators if no test repair operators exist."""
    evo_config = MagicMock(spec=EvolutionConfig)

    mock_code_breeder = MagicMock(spec=Breeder)
    code_profile = MagicMock(spec=CodeProfile)
    code_profile.breeder = mock_code_breeder

    test_profile = MagicMock(spec=TestProfile)
    test_profile.repair_operators = []

    evolved_test_profiles: dict[str, TestProfile] = {"unittest": test_profile}
    public_test_profile = MagicMock(spec=PublicTestProfile)

    Orchestrator(
        evo_config=evo_config,
        code_profile=code_profile,
        evolved_test_profiles=evolved_test_profiles,
        public_test_profile=public_test_profile,
        execution_system=MagicMock(),
        bayesian_system=MagicMock(),
        ledger_factory=MagicMock(),
        composer=MagicMock(),
    )

    mock_code_breeder.add_operators.assert_not_called()
