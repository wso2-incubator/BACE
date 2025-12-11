"""Test to verify operator rates validation catches unsupported operations."""

import pytest

from common.coevolution.core.interfaces import OperatorRatesConfig
from common.coevolution.core.mock import MockCodeOperator, get_mock_problem


def test_operator_rates_validation_passes_for_supported_operations() -> None:
    """Validate that supported operations pass validation."""
    problem = get_mock_problem()
    code_operator = MockCodeOperator(problem)

    # These operations are supported by MockCodeOperator
    config = OperatorRatesConfig(
        operation_rates={"crossover": 0.3, "edit": 0.2, "mutation": 0.1},
        mutation_rate=0.1,
    )

    # Should not raise
    config.validate_against_operator(code_operator, "code_operator")


def test_operator_rates_validation_fails_for_unsupported_operations() -> None:
    """Validate that unsupported operations fail validation."""
    problem = get_mock_problem()
    code_operator = MockCodeOperator(problem)

    # "invalid_op" is not supported by MockCodeOperator
    config = OperatorRatesConfig(
        operation_rates={"crossover": 0.3, "invalid_op": 0.2}, mutation_rate=0.1
    )

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        config.validate_against_operator(code_operator, "code_operator")

    assert "code_operator does not support operations: {'invalid_op'}" in str(
        exc_info.value
    )
    assert "Supported operations:" in str(exc_info.value)


def test_operator_rates_validation_allows_empty_rates() -> None:
    """Validate that empty operation_rates is allowed (reproduction only)."""
    problem = get_mock_problem()
    code_operator = MockCodeOperator(problem)

    # No operations configured - everything goes to reproduction
    config = OperatorRatesConfig(operation_rates={}, mutation_rate=0.1)

    # Should not raise
    config.validate_against_operator(code_operator, "code_operator")
