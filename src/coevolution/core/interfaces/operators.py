# coevolution/core/interfaces/operators.py
"""
IOperator protocol — the single interface all genetic operators implement.

Operators are self-contained units of evolutionary work. The Breeder only
calls execute(context) and operation_name().
"""

from typing import TYPE_CHECKING, Protocol

from .base import BaseIndividual

if TYPE_CHECKING:
    from .context import CoevolutionContext


class IOperator[T: BaseIndividual](Protocol):
    """
    A self-contained unit of evolutionary work.

    Owns:
    - Context/parent selection (via injected selectors at construction)
    - LLM transformation (via injected LLM at construction)
    - Probability assignment and individual construction

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


__all__ = ["IOperator"]
