# coevolution/core/interfaces/operators.py
"""
Operator protocol for genetic operations.

Operators are self-contained units of evolutionary work.
Each operator owns its own parent selection, LLM call, probability
assignment, and individual construction. The Breeder only routes to them.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .base import BaseIndividual
from .types import Operation

if TYPE_CHECKING:
    from .context import CoevolutionContext


# ---------------------------------------------------------------------------
# DTOs — kept for use by concrete operator internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseOperatorInput:
    """Base class for all operator inputs."""

    operation: Operation
    question_content: str


@dataclass(frozen=True)
class InitialInput(BaseOperatorInput):
    """Input DTO for initial population generation.

    Fields:
        population_size: Number of individuals to generate.
        starter_code: Optional starter code scaffold (may be empty for tests).
    """

    population_size: int
    starter_code: str


@dataclass(frozen=True)
class OperatorResult:
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorOutput:
    results: list[OperatorResult]


# ---------------------------------------------------------------------------
# IOperator Protocol
# ---------------------------------------------------------------------------


class IOperator[T: BaseIndividual](Protocol):
    """
    A self-contained unit of evolutionary work.

    Owns:
    - Context/parent selection (via injected selectors at construction)
    - LLM transformation (via injected LLM at construction)
    - Probability assignment and individual construction (via injected
      factory/assigner at construction)

    The Breeder calls only execute() and operation_name().
    """

    def execute(self, context: "CoevolutionContext") -> list[T]:
        """
        End-to-end operation:
          1. Selects context from CoevolutionContext (parents, failing tests, etc.)
          2. Calls LLM / performs transformation
          3. Assigns probabilities and wraps results into Individual objects

        Raises:
            OperatorContextError: If context is insufficient for this operation
                (e.g. no failing tests available for EDIT). The Breeder will
                catch this and retry with a different sampled operator.
        """
        ...

    def operation_name(self) -> str:
        """The name of the operation this instance handles (e.g. 'mutation')."""
        ...


__all__ = [
    "BaseOperatorInput",
    "InitialInput",
    "IOperator",
    "OperatorOutput",
    "OperatorResult",
]
