# coevolution/core/interfaces/types.py
"""
Type aliases, enums, and constants for the coevolution framework.
"""

from enum import Enum
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .data import ExecutionResult

# Type alias for genetic operations
type Operation = str

# Type alias for parent lineage tracking (grouped by type)
type ParentDict = dict[Literal["code", "test"], list[str]]

# Standard operation names (for convenience, not exhaustive)
OPERATION_INITIAL: Literal["initial"] = "initial"
OPERATION_CROSSOVER: Literal["crossover"] = "crossover"
OPERATION_EDIT: Literal["edit"] = "edit"
OPERATION_REPRODUCTION: Literal["reproduction"] = "reproduction"
OPERATION_MUTATION: Literal["mutation"] = "mutation"


class LifecycleEvent(Enum):
    """Enumeration of lifecycle events for individuals."""

    CREATED = "created"
    BECAME_PARENT = "became_parent"
    SELECTED_AS_ELITE = "selected_as_elite"
    PROBABILITY_UPDATED = "probability_updated"
    DIED = "died"
    SURVIVED = "survived"


type ParentProbabilities = list[float]

# ExecutionResults is an alias mapping a code-individual ID (str) to its
# ExecutionResult. We use a forward-reference string for `ExecutionResult`
# to avoid evaluation order issues at import time and mark it as a TypeAlias
# for clearer typing semantics.
ExecutionResults: TypeAlias = dict[str, "ExecutionResult"]
InteractionKey: TypeAlias = tuple[str, str, str, str]
