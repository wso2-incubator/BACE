# coevolution/core/interfaces/operators.py
"""
Operator protocols and related DTOs for genetic operations.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from .data import Test
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
    ) -> tuple[OperatorOutput, str | None]:
        """
        Generate the initial batch of code or test snippets.

        Args:
            input_dto: DTO containing generation parameters (`InitialInput`).

        Returns:
            Tuple of:
            - OperatorOutput: A container with the generated snippets and metadata.
            - context_code: Optional auxiliary code (e.g., test class scaffold
              with imports/setUp) needed to run these snippets. None for code ops.

        Empty State Behavior (size=0):
            If population_size is 0, should return an empty OperatorOutput
            (results=[]) and potentially a context string (e.g. empty test suite scaffold).
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


class IDatasetTestBlockBuilder(Protocol):
    """
    Protocol for building test class blocks from dataset test cases (public/private tests).

    This interface is responsible for converting dataset-specific test cases into
    executable unittest test class blocks. Different datasets may have different
    test formats (e.g., STDIN vs FUNCTIONAL), and this interface abstracts that.

    This is specifically for fixed test populations derived from the dataset,
    NOT for general test block building (that's what ITestBlockRebuilder is for).

    Note: Implementations can be stateless (using @staticmethod) if desired,
    but the protocol defines an instance method for maximum flexibility.
    """

    def build_test_class_block(self, test_cases: list[Test], starter_code: str) -> str:
        """
        Build a unittest test class block from dataset test cases.

        Args:
            test_cases: List of Test objects from the dataset (public or private)
            starter_code: The starter code for the problem

        Returns:
            A complete unittest test class block as a string, ready to be executed.
            This should include imports, class definition, and all test methods.

        Example:
            >>> builder = LCBDatasetTestBlockBuilder()
            >>> tests = [Test(input="2 3", output="5", testtype=TestType.STDIN)]
            >>> block = builder.build_test_class_block(tests, "def add(a, b): pass")
            >>> "class" in block and "def test_" in block
            True
        """
        ...


class ITestBlockRebuilder(Protocol):
    """
    Protocol defining the contract for rebuilding test class blocks.

    This interface allows for flexible implementations of test block reconstruction,
    whether as a simple utility function or a more sophisticated strategy.
    """

    def rebuild_test_block(
        self, original_class_str: str, new_method_snippets: list[str]
    ) -> str:
        """
        Rebuilds a test class string using new test method snippets.

        Args:
            original_class_str: The full string of the original unittest class.
                                This is used as a template to preserve imports,
                                class name, and helper methods.
            new_method_snippets: A list of new test method snippets (e.g.,
                                 ["def test_new_1(self): ...", ...]) to
                                 replace the old ones.

        Returns:
            A new full unittest class string.
        """
        ...
