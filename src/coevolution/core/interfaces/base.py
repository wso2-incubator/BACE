# coevolution/core/interfaces/base.py
"""
Abstract base classes for individuals and populations.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, overload

import numpy as np
from loguru import logger

from .types import Operation, ParentDict


class BaseIndividual(ABC):
    """
    Abstract Base Class for any individual.

    Implements all shared logic and state for individuals,
    such as storing snippets, probabilities, and provenance.

    All core attributes (snippet, probability, creation_op) are immutable
    after creation to maintain consistency and prevent accidental modifications.

    Subclasses are only required to implement the 'id' property
    and the '__repr__' method.
    """

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
        Initializes the shared state for all individuals.

        Args:
            snippet: The code or test content
            probability: Initial probability value
            creation_op: The genetic operation that created this individual
            generation_born: Generation when this individual was created
            parents: Parent individuals grouped by type {"code": [ids], "test": [ids]}
            metadata: Additional operation-specific metadata
        """
        self._snippet = snippet
        self._creation_op = creation_op
        self._generation_born = generation_born
        self._parents = parents if parents is not None else {"code": [], "test": []}
        self._metadata = metadata if metadata is not None else {}

        BaseIndividual._validate_probability(probability)
        self._probability = probability

    # --- Abstract Properties (Must be implemented by subclasses) ---

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The unique identifier (e.g., 'C1' or 'T1').
        This is abstract because the prefix is different.
        """

    # -- helper for validation of probability --
    @staticmethod
    def _validate_probability(value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError("Probability must be between 0.0 and 1.0")

    # --- Concrete Properties (Shared implementation) ---
    @property
    def snippet(self) -> str:
        """
        The underlying code or test snippet (immutable after creation).
        """
        return self._snippet

    @property
    def probability(self) -> float:
        """
        The belief in this individual's correctness.

        This value is mutable and gets updated via Bayesian belief updates
        based on test execution results.
        """
        return self._probability

    @probability.setter
    def probability(self, value: float) -> None:
        """
        Setter for the individual's probability.

        Called during Bayesian belief updates to adjust confidence
        based on test execution results.
        """
        BaseIndividual._validate_probability(value)
        logger.trace(
            f"{self.id} probability updated: {self._probability:.4f} -> {value:.4f}"
        )
        self._probability = value

    @property
    def creation_op(self) -> Operation:
        """
        The name of the operation that created this individual (immutable).
        """
        return self._creation_op

    @property
    def parents(self) -> ParentDict:
        """
        Dictionary grouping parent IDs by their types.

        This allows tracking cross-species parents (e.g., DET operator
        takes code parents to create test offspring).

        Returns:
            Dict grouping parent IDs: {"code": [id1, id2], "test": [id3]}.
        """
        return self._parents

    @property
    def code_parent_ids(self) -> list[str]:
        """
        Convenience accessor for code parent IDs.

        Returns:
            List of code parent IDs, empty list if none.
        """
        return self._parents.get("code", [])

    @property
    def test_parent_ids(self) -> list[str]:
        """
        Convenience accessor for test parent IDs.

        Returns:
            List of test parent IDs, empty list if none.
        """
        return self._parents.get("test", [])

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Operation-specific metadata for this individual.

        This can store additional information such as:
        - LLM prompts and responses
        - Reasoning chains
        - Execution traces
        - Operation parameters

        Returns:
            Dict containing operation-specific metadata.
        """
        return self._metadata

    @property
    def generation_born(self) -> int:
        """The generation number in which this individual was created."""
        if self._generation_born < 0:
            raise ValueError("Generation born cannot be negative.")
        return self._generation_born

    # --- Common Methods ---

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __eq__(self, other: object) -> bool:
        """
        Individuals are considered equal if they are of the same
        base type and have the same unique ID.
        """
        return isinstance(other, BaseIndividual) and self.id == other.id


class BasePopulation[T_Individual: BaseIndividual](ABC):
    """
    Abstract Base Class for any population.

    Implements all shared logic and state for populations,
    such as storing individuals and their probabilities.

    Subclasses are only required to implement the '__init__' method
    to set up the individuals and probabilities.
    """

    def __init__(self, individuals: list[T_Individual], generation: int = 0) -> None:
        """
        Initializes the shared state for all populations.

        Note: Empty populations are now supported to enable dynamic/bootstrapped
        population types (e.g., differential testing that starts with zero tests).
        """
        self._individuals = individuals
        self._generation = generation

        if not individuals:
            logger.debug(
                f"Initialized {self.__class__.__name__} with 0 individuals (empty/bootstrap mode), gen {generation}"
            )
        else:
            logger.debug(
                f"Initialized {self.__class__.__name__} with {len(individuals)} individuals, gen {generation}"
            )

    def __len__(self) -> int:
        return len(self._individuals)

    @overload
    def __getitem__(self, index: int) -> T_Individual:
        """Gets a single individual by integer index."""

    @overload
    def __getitem__(self, index: slice) -> list[T_Individual]:
        """Gets a list of individuals by slice."""

    def __getitem__(self, index: int | slice) -> T_Individual | list[T_Individual]:
        """
        Gets an individual by index or a list of individuals by slice.

        This relies on the underlying list's __getitem__ which
        already supports both integers and slices.
        """
        # The list's __getitem__ handles both types automatically
        return self._individuals[index]

    def __iter__(self) -> Iterator[T_Individual]:
        for i in self._individuals:
            yield i

    @property
    def size(self) -> int:
        """Returns the size of the population."""
        return len(self._individuals)

    @property
    def generation(self) -> int:
        """Returns the current generation number of the population."""
        return self._generation

    @property
    def individuals(self) -> list[T_Individual]:
        """Returns the list of individuals in the population."""
        return self._individuals.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """Returns the list of probabilities for all individuals."""
        return np.asarray([ind.probability for ind in self._individuals])

    @property
    def snippets(self) -> list[str]:
        """Returns the list of snippets for all individuals."""
        return [ind.snippet for ind in self._individuals]

    @property
    def ids(self) -> list[str]:
        """Returns the list of IDs for all individuals."""
        return [ind.id for ind in self._individuals]

    # Core Public API
    def set_next_generation(self, new_individuals: list[T_Individual]) -> None:
        """
        Replaces the entire current population with a new set of individuals
        and advances the generation counter.

        This method logs which individuals were kept (elites),
        added (offspring), and removed (not selected).

        Note: This method does NOT notify individuals about lifecycle events.
        The caller (typically the Orchestrator) is responsible for notifying
        removed individuals about their death before calling this method.
        """
        if not new_individuals:
            logger.warning(
                "set_next_generation called with an empty list of new individuals."
            )

        # Logging the differences between old and new populations
        old_ids_map = {ind.id: ind for ind in self._individuals}
        new_ids = {ind.id for ind in new_individuals}

        # Calculate the differences
        deleted_ids = set(old_ids_map.keys()) - new_ids  # In old_ids but not in new_ids
        added_ids = new_ids - set(old_ids_map.keys())  # In new_ids but not in old_ids
        kept_ids = set(old_ids_map.keys()) & new_ids  # In both (set intersection)

        logger.debug(
            f"Gen {self._generation} -> {self._generation + 1}: "
            f"Kept {len(kept_ids)} elites, "
            f"Added {len(added_ids)} new offspring, "
            f"Removed {len(deleted_ids)} individuals."
        )

        if deleted_ids:
            logger.trace(f"Removed individuals: {sorted(list(deleted_ids))}")
        if added_ids:
            logger.trace(f"Added individuals: {sorted(list(added_ids))}")
        if kept_ids:
            logger.trace(f"Kept elite individuals: {sorted(list(kept_ids))}")

        old_size = self.size
        old_avg_prob = self.compute_average_probability()

        self._individuals = new_individuals
        self._generation += 1
        self._on_generation_advanced()

        new_avg_prob = self.compute_average_probability()
        logger.info(
            f"Advanced {self.__class__.__name__} to gen {self._generation}: "
            f"size {old_size} -> {self.size}, "
            f"avg_prob {old_avg_prob:.4f} -> {new_avg_prob:.4f}"
        )

    def compute_average_probability(self) -> float:
        """Computes and returns the average probability of the population."""
        logger.trace(f"Computing avg probability for {self.size} individuals...")
        if self.size == 0:
            logger.trace("Population is empty, avg prob is 0.0")
            return 0.0
        return float(np.mean(self.probabilities))

    def get_best_individual(self) -> T_Individual | None:
        """Returns the individual with the highest probability."""
        logger.trace("Getting best individual...")
        if self.size == 0:
            logger.warning("get_best_individual called on an empty population.")
            return None

        best_individual = max(self._individuals, key=lambda ind: ind.probability)
        logger.trace(f"Best individual found: {best_individual!r}")
        return best_individual

    def get_top_k_individuals(self, k: int) -> list[T_Individual]:
        """Returns the top k individuals with the highest probabilities."""
        logger.trace(f"Getting top {k} individuals...")
        if k <= 0:
            return []
        k = min(k, self.size)
        return sorted(self._individuals, key=lambda ind: ind.probability, reverse=True)[
            :k
        ]

    def update_probabilities(
        self, new_probabilities: np.ndarray, test_type: str | None = None
    ) -> None:
        """Updates the probabilities of individuals in the population.

        Args:
            new_probabilities: Array of new probability values for each individual.
            test_type: The type of test that triggered the update (e.g., 'public', 'unittest', 'differential').
        """
        if len(new_probabilities) != self.size:
            logger.error(
                f"Length mismatch: {len(new_probabilities)} probabilities vs {self.size} individuals."
            )
            raise ValueError("Length of new_probabilities must match population size.")

        old_avg = self.compute_average_probability()

        for ind, new_prob in zip(self._individuals, new_probabilities, strict=False):
            ind.probability = float(new_prob)

        new_avg = self.compute_average_probability()
        logger.info(
            f"Updated probabilities for {self.__class__.__name__} (gen {self._generation}, test_type={test_type}): "
            f"avg {old_avg:.4f} -> {new_avg:.4f} (Δ{new_avg - old_avg:+.4f})"
        )

    def get_index_of_individual(self, individual: T_Individual) -> int:
        """Returns the index of the given individual in the population."""
        for idx, ind in enumerate(self._individuals):
            if ind.id == individual.id:
                return idx
        return -1

    # -- abstract hook for subclasses --
    @abstractmethod
    def _on_generation_advanced(self) -> None:
        """
        Hook method called after the generation is advanced.
        Subclasses can override this to implement custom behavior.
        """

    @abstractmethod
    def __repr__(self) -> str:
        pass
