# src/common/coevolution/core/individual.py
import itertools

from loguru import logger

from .interfaces import BaseIndividual, Operations


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
        creation_op: Operations,
        generation_born: int,
        parent_ids: list[str],
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
            parent_ids,
        )

        self._id = f"C{next(CodeIndividual._code_counter)}"
        self.add_to_log(f"Born as {self._id}")
        logger.debug(f"Created new {self!r}")

    @property
    def id(self) -> str:
        """The unique identifier for this core code individual."""
        return self._id

    def __repr__(self) -> str:
        return (
            f"<CodeIndividual id={self.id} gen={self.generation_born} "
            f"op={self.creation_op} prob={self.probability:.1f}>"
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
        creation_op: Operations,
        generation_born: int,
        parent_ids: list[str],
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
            parent_ids,
        )

        # discrimination is only for test individuals
        self._discrimination: float | None = None

        self._id = f"Core_T{next(TestIndividual._core_test_counter)}"
        self.add_to_log(f"Born as {self._id}")

        logger.debug(f"Created new {self!r}")  # Use __repr__ for log

    @property
    def id(self) -> str:
        """The unique identifier for this core test individual."""
        return self._id

    # --- Concrete implementation of '__repr__' ---
    def __repr__(self) -> str:
        # Check if discrimination is set before trying to print it
        disc_str = (
            f"{self._discrimination:.1f}"
            if self._discrimination is not None
            else "None"
        )
        return (
            f"<TestIndividual id={self.id} gen={self.generation_born} "
            f"prob={self.probability:.1f} disc={disc_str}>"
        )

    # --- Test-Specific Properties ---
    @property
    def discrimination(self) -> float | None:
        if self._discrimination is None:
            logger.warning(
                f"{self.id}: Attempted to access .discrimination, but it is not set."
            )
        return self._discrimination

    @discrimination.setter
    def discrimination(self, value: float | None) -> None:  # Allow setting to None
        log_val = f"{value:.4f}" if value is not None else "None"
        logger.trace(f"{self.id} discrimination set to {log_val}")
        self._discrimination = value
