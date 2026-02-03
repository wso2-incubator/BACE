# coevolution/core/interfaces/operators.py
"""
Operator protocols and related DTOs for genetic operations.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from .types import Operation


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


# The Output is just a list of these results
@dataclass(frozen=True)
class OperatorOutput:
    results: list[OperatorResult]


class IOperator(Protocol):
    """
    Protocol for genetic operators that transform code/test strings.

    Operators are "stateless workers" that handle pure string transformation.
    They are strictly decoupled from Domain Objects (Individuals, Populations).
    The Breeding Strategy handles the wrapping/unwrapping of Individual objects.

    Capabilities:
    1. Generate initial snippets (Generation 0)
    2. Apply genetic operations to existing snippets (Generation 1+)
    """

    def generate_initial_snippets(
        self,
        input_dto: InitialInput,
    ) -> OperatorOutput:
        """
        Generate the initial batch of code or test snippets.

        Args:
            input_dto: DTO containing generation parameters (`InitialInput`).

        Returns:
            OperatorOutput: A container with the generated snippets and metadata.

        Note: Previously returned a tuple with context_code (test class scaffold),
        but with pytest standalone functions, TestPopulation now builds its own
        test block through simple concatenation of imports + test functions.

        Empty State Behavior (size=0):
            If population_size is 0, should return an empty OperatorOutput (results=[]).
        """
        ...

    def supported_operations(self) -> set[Operation]:
        """
        Return the set of operations this operator can handle.

        Used to validate that the Breeding Strategy's configuration only
        requests operations that this operator actually supports.

        Returns:
            Set of operation names (strings).
            Example: {"mutation", "crossover", "edit", "det"}
        """
        ...

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        """
        Apply a genetic operation to input strings.

        The operator does not need to know about 'Individuals', 'Populations',
        or 'ObservationMatrices'. It receives exactly what it needs via the DTO.

        Args:
            input_dto: A strongly-typed DTO (e.g., MutationInput, EditInput)
                       containing the operation name, parent snippets, and
                       any required context (like error traces).

        Returns:
            OperatorOutput containing the generated offspring snippets and metadata.
            The Strategy is responsible for wrapping these into new Individuals.
        """
        ...
