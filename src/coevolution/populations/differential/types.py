"""Shared data types for the differential population.

FunctionallyEquivGroup, IDifferentialFinder, DifferentialResult and
IFunctionallyEquivalentCodeSelector are used by the selector, finder,
and discovery operator — defined here to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import CoevolutionContext


OPERATION_DISCOVERY: str = "discovery"


@dataclass(frozen=True)
class FunctionallyEquivGroup:
    """A cluster of code individuals that behave identically on current tests."""
    code_individuals: list[CodeIndividual]
    passing_test_individuals: dict[str, list[TestIndividual]]


class IFunctionallyEquivalentCodeSelector(Protocol):
    """Protocol for finding groups of code that behave identically."""
    def select_functionally_equivalent_codes(
        self,
        coevolution_context: CoevolutionContext,
    ) -> list[FunctionallyEquivGroup]: ...


@dataclass(frozen=True)
class DifferentialResult:
    """A single input where two code snippets produced different output."""
    input_data: dict[str, Any]
    output_a: Any
    output_b: Any


class IDifferentialFinder(Protocol):
    """Protocol for the execution sandbox."""
    def find_differential(
        self,
        code_a_snippet: str,
        code_b_snippet: str,
        input_generator_script: str,
        limit: int = 10,
    ) -> list[DifferentialResult]: ...


__all__ = [
    "OPERATION_DISCOVERY",
    "FunctionallyEquivGroup",
    "IFunctionallyEquivalentCodeSelector",
    "DifferentialResult",
    "IDifferentialFinder",
]
