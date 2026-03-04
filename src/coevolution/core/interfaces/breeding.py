# coevolution/core/interfaces/breeding.py
"""
Breeding-related protocols retained for use by concrete operators.

IBreedingStrategy has been superseded by the universal Breeder class
(see coevolution/strategies/breeding/breeder.py), but is kept here as a
compatibility shim so that existing concrete breeding strategies
(BaseBreedingStrategy, CodeBreedingStrategy, etc.) continue to import
without modification during the migration period.

IProbabilityAssigner and IIndividualFactory are still used by operators to
assign probabilities and construct individuals within execute().
"""

from typing import Protocol

from .base import BaseIndividual
from .types import Operation, ParentProbabilities


class IProbabilityAssigner(Protocol):
    """
    Protocol defining the contract for a probability assignment policy.
    Injected into operators at construction time.
    """

    def assign_probability(
        self,
        operation: Operation,
        parent_probs: ParentProbabilities,
        initial_prior: float,
    ) -> float:
        """
        Calculates the probability for a new offspring.

        Args:
            operation: The op that created the child (e.g., "crossover", "mutation").
            parent_probs: A list of the parent(s)' probabilities.
            initial_prior: The default prior, to be used if the policy dictates.

        Returns:
            The new probability for the offspring.
        """
        ...


class IIndividualFactory[T_Individual: BaseIndividual](Protocol):
    """
    Protocol defining the contract for an individual factory.
    Generic over the type of individual it creates.
    Injected into operators at construction time.
    """

    def __call__(
        self,
        snippet: str,
        probability: float,
        creation_op: Operation,
        generation_born: int,
        parent_ids: list[str],
    ) -> T_Individual:
        """
        Constructs and returns a new individual.

        Args:
            snippet: The code/test snippet.
            probability: The initial probability.
            creation_op: The operation that created this individual.
            generation_born: The generation number.
            parent_ids: A list of parent IDs.

        Returns:
            A new instance of a class that implements BaseIndividual.
        """
        ...


# ---------------------------------------------------------------------------
# Compatibility shim — kept so existing concrete strategies don't break.
# New code should use Breeder + IPopulationInitializer instead.
# ---------------------------------------------------------------------------

from typing import TYPE_CHECKING, Any


class IBreedingStrategy[T: BaseIndividual](Protocol):
    """
    DEPRECATED — superseded by Breeder + IPopulationInitializer.
    Kept as a compatibility shim for concrete strategies not yet migrated.
    """

    def initialize_individuals(self, problem: Any) -> list[T]:
        """Create generation-0 individuals. Use IPopulationInitializer instead."""
        ...

    def breed(self, coevolution_context: Any, num_offsprings: int) -> list[T]:
        """Generate offspring. Use Breeder.breed() instead."""
        ...


__all__ = ["IBreedingStrategy", "IIndividualFactory", "IProbabilityAssigner"]
