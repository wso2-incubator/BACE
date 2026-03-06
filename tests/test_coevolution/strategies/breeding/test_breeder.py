from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, cast

import pytest

from coevolution.core.interfaces.base import BaseIndividual
from coevolution.core.interfaces.operators import IOperator
from coevolution.strategies.breeding.breeder import Breeder, RegisteredOperator

if TYPE_CHECKING:
    from coevolution.core.interfaces.context import CoevolutionContext


class MockIndividual(BaseIndividual):
    """Mock individual for testing."""

    def __init__(self, content: str, probability: float = 1.0) -> None:
        super().__init__(
            snippet=content,
            probability=probability,
            creation_op="MOCK",
            generation_born=0,
        )
        self.content = content

    @property
    def id(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"MockIndividual(content={self.content!r})"

    def __hash__(self) -> int:
        return hash(self.content)


class MockOperator(IOperator[MockIndividual]):
    """Mock operator that returns a fixed number of individuals."""

    def __init__(self, name: str, behavior: str = "success") -> None:
        self.name = name
        self.behavior = behavior
        self.call_count = 0

    def execute(self, context: CoevolutionContext) -> list[MockIndividual]:
        self.call_count += 1
        if self.behavior == "error":
            raise Exception(f"Context error in {self.name}")
        if self.behavior == "fail":
            raise Exception(f"Unexpected failure in {self.name}")

        return [MockIndividual(content=f"{self.name}_offspring_{self.call_count}")]

    def operation_name(self) -> str:
        return self.name


@pytest.fixture
def mock_context() -> CoevolutionContext:
    """Returns a mock CoevolutionContext."""
    return cast("CoevolutionContext", object())


def test_breeder_initialization_fails_with_no_operators() -> None:
    """Verify Breeder raises ValueError if initialized with empty list."""
    with pytest.raises(
        ValueError, match="Breeder must have at least one registered operator."
    ):
        Breeder[MockIndividual](registered_operators=[])


def test_breeder_single_threaded_success(mock_context: CoevolutionContext) -> None:
    """Verify Breeder correctly generates offspring in single-threaded mode."""
    op = MockOperator("op1")
    reg_op = RegisteredOperator(weight=1.0, operator=op)
    breeder = Breeder[MockIndividual](registered_operators=[reg_op], llm_workers=1)

    num_offsprings = 3
    offspring = breeder.breed(mock_context, num_offsprings=num_offsprings)

    assert len(offspring) == num_offsprings
    assert op.call_count == num_offsprings
    assert all(o.content.startswith("op1_offspring") for o in offspring)


def test_breeder_weighted_sampling(mock_context: CoevolutionContext) -> None:
    """Verify Breeder respects operator weights."""
    op1 = MockOperator("op1")
    op2 = MockOperator("op2")

    reg_ops = [
        RegisteredOperator(weight=1.0, operator=op1),
        RegisteredOperator(weight=0.0, operator=op2),
    ]
    breeder = Breeder[MockIndividual](registered_operators=reg_ops, llm_workers=1)

    offspring = breeder.breed(mock_context, num_offsprings=10)

    assert op1.call_count == 10
    assert op2.call_count == 0
    assert len(offspring) == 10


def test_breeder_retry_on_operator_failure(mock_context: CoevolutionContext) -> None:
    """Verify Breeder retries another operator when one raises an Exception."""
    op1 = MockOperator("op1", behavior="error")
    op2 = MockOperator("op2", behavior="success")

    reg_ops = [
        RegisteredOperator(weight=0.5, operator=op1),
        RegisteredOperator(weight=0.5, operator=op2),
    ]

    breeder = Breeder[MockIndividual](registered_operators=reg_ops, llm_workers=1)

    # Use enough offsprings to ensure op1 is picked at least once with random.choices
    # Seed random for predictability
    random.seed(42)
    offspring = breeder.breed(mock_context, num_offsprings=20)

    assert len(offspring) == 20
    assert op2.call_count == 20
    assert op1.call_count > 0


def test_breeder_circuit_breaker(mock_context: CoevolutionContext) -> None:
    """Verify Breeder stops after too many consecutive Operator failures."""
    op1 = MockOperator("op1", behavior="error")
    reg_op = RegisteredOperator(weight=1.0, operator=op1)

    breeder = Breeder[MockIndividual](registered_operators=[reg_op], llm_workers=1)

    offspring = breeder.breed(mock_context, num_offsprings=5)

    assert len(offspring) == 0
    assert op1.call_count == 10


def test_breeder_multi_threaded_success(mock_context: CoevolutionContext) -> None:
    """Verify Breeder works with multiple workers."""
    op = MockOperator("parallel_op")
    reg_op = RegisteredOperator(weight=1.0, operator=op)
    breeder = Breeder[MockIndividual](registered_operators=[reg_op], llm_workers=4)

    num_offsprings = 20
    offspring = breeder.breed(mock_context, num_offsprings=num_offsprings)

    assert len(offspring) == num_offsprings
    assert op.call_count == num_offsprings


def test_breeder_num_offsprings_zero(mock_context: CoevolutionContext) -> None:
    """Verify Breeder returns empty list for zero offsprings."""
    op = MockOperator("op")
    reg_op = RegisteredOperator(weight=1.0, operator=op)
    breeder = Breeder[MockIndividual](registered_operators=[reg_op])

    offspring = breeder.breed(mock_context, num_offsprings=0)
    assert offspring == []
    assert op.call_count == 0

    offspring = breeder.breed(mock_context, num_offsprings=0)
    assert offspring == []
    assert op.call_count == 0
