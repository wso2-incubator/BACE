# src/common/coevolution/core/individual.py
import itertools
from typing import Any

from loguru import logger

from .interfaces import BaseIndividual, Operation, ParentDict


class CodeIndividual(BaseIndividual):
    """
    Implementation of a Code Individual.

    This is the "simple implementation" for the core kernel.
    The "real" LLM-based class (outside the core) will inherit
    from this one and override methods as needed.
    """

    _code_counter = itertools.count(start=0)

    def __init__(
        self,
        snippet: str,
        probability: float,
        creation_op: Operation,
        generation_born: int,
        parents: ParentDict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the individual by passing all state
        to the base class.
        """
        # --- Pass all arguments to the base class ---
        super().__init__(
            snippet,
            probability,
            creation_op,
            generation_born,
            parents,
            metadata,
        )

        self._id = f"C{next(CodeIndividual._code_counter)}"
        logger.debug(f"Created new {self!r}")

    @property
    def id(self) -> str:
        """The unique identifier for this core code individual."""
        return self._id

    def __repr__(self) -> str:
        return (
            f"<CodeIndividual id={self.id} gen={self.generation_born} "
            f"op={self.creation_op} prob={self.probability:.2f}>"
        )


class TestIndividual(BaseIndividual):
    """
    CONCRETE implementation of a Test Individual.

    This is the "simple implementation" for the core kernel.
    The "real" LLM-based class will inherit from this.
    """

    # Counter for the "core" implementation of test individuals
    _core_test_counter = itertools.count(start=0)

    def __init__(
        self,
        snippet: str,
        probability: float,
        creation_op: Operation,
        generation_born: int,
        parents: ParentDict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the individual by passing all state
        to the base class.
        """

        super().__init__(
            snippet,
            probability,
            creation_op,
            generation_born,
            parents,
            metadata,
        )

        self._id = f"T{next(TestIndividual._core_test_counter)}"

        logger.debug(f"Created new {self!r}")  # Use __repr__ for log

    @property
    def id(self) -> str:
        """The unique identifier for this core test individual."""
        return self._id

    # --- Concrete implementation of '__repr__' ---
    def __repr__(self) -> str:
        return (
            f"<TestIndividual id={self.id} gen={self.generation_born} "
            f"op={self.creation_op} prob={self.probability:.2f}>"
        )
