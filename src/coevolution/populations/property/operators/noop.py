"""No-op operator placeholder for the property test population."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coevolution.core.individual import TestIndividual

if TYPE_CHECKING:
    from coevolution.core.interfaces.context import CoevolutionContext


class NoOpOperator:
    """An operator that produces no offspring.

    Used as a placeholder while breeding is disabled for the property test
    population (offspring_rate=0 means Breeder.breed() early-exits before
    ever calling execute(), so this is never actually invoked).
    """

    def execute(self, context: "CoevolutionContext") -> list[TestIndividual]:
        return []

    def operation_name(self) -> str:
        return "noop"


__all__ = ["NoOpOperator"]
