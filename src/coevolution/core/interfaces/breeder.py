# coevolution/core/interfaces/breeder.py
"""
IBreeder protocol — the interface the Orchestrator depends on.

The concrete implementation (Breeder in strategies/breeding/breeder.py)
registers N weighted IOperators and routes breed() calls to them.
Core code (Orchestrator, profiles) should depend only on this interface.
"""

from typing import TYPE_CHECKING, Protocol

from .base import BaseIndividual

if TYPE_CHECKING:
    from .context import CoevolutionContext


class IBreeder[T: BaseIndividual](Protocol):
    """
    Routes breed() calls to a weighted set of IOperators.

    The Orchestrator calls breed(context, num_offsprings) once per generation
    per population. The concrete Breeder handles operator sampling, retries,
    and parallel execution internally.
    """

    def breed(self, context: "CoevolutionContext", num_offsprings: int) -> list[T]:
        """
        Generate num_offsprings new individuals.

        Args:
            context: Full coevolution context (populations, interactions, problem).
            num_offsprings: How many offspring to produce.

        Returns:
            list of new individuals, length <= num_offsprings.
        """
        ...


__all__ = ["IBreeder"]
