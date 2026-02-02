# coevolution/core/interfaces/breeding.py
"""
Breeding strategy protocols for population evolution.
"""

from typing import Protocol

from .base import BaseIndividual
from .context import CoevolutionContext
from .data import Problem
from .types import Operation, ParentProbabilities


class IBreedingStrategy[T_self: BaseIndividual](Protocol):
    """
    Protocol for breeding strategies that handle population generation.

    The Breeding Strategy acts as the "Manager" of the evolutionary process.
    It bridges the gap between the Domain Model (Individuals, Populations) and
    the Stateless Workers (Operators).

    Responsibilities:
    1. **Orchestration**: Decides which operations to perform (Mutation, Crossover, Edit, etc.)
       based on configured rates.
    2. **Selection**: Selects parents and operation-specific context.
       - Generic Selection: Uses IParentSelectionStrategy for fitness-based selection.
       - Specialized Selection: Finds specific contexts (e.g., failing tests for edits,
         divergent pairs for differential testing) using internal logic or helpers.
    3. **Data Preparation (Unwrapping)**: Extracts raw strings/primitives from selected
       Individual objects and packages them into strongly-typed DTOs (e.g., MutationInput,
       EditInput) for the Operator.
    4. **Dispatch**: Calls the 'dumb' IOperator with these DTOs.
    5. **Construction (Wrapping)**: Uses an IIndividualFactory to wrap the raw strings
       returned by the Operator into new Individual objects, assigning IDs, probabilities,
       and lineage data.

    Design Pattern: "Smart Strategy, Dumb Operator"
    - The Strategy absorbs the complexity of the domain (Matrices, Objects, History).
    - The Operator handles only pure string transformation.
    """

    def initialize_individuals(
        self,
        problem: Problem,
    ) -> list[T_self]:
        """
        Create initial population by delegating to operator.

        This method orchestrates the creation of Generation 0:
        1. Extracts problem description and starter code from the Problem object.
        2. Calls the operator's `generate_initial_snippets` method with these primitives.
        3. Wraps the resulting snippets into Individual objects using the factory.

        Args:
            problem: The problem context (used to extract question/starter code strings).

        Returns:
            List of newly generated individuals (Gen 0).

        Empty State Behavior:
            - If population size is 0, returns [].
            - This supports bootstrapping scenarios where a population (like differential tests)
              starts empty and is populated later via breeding.
        """
        ...

    def breed(
        self,
        coevolution_context: CoevolutionContext,
        num_offsprings: int,
    ) -> list[T_self]:
        """
        Generate offspring using genetic operations.

        The workflow for each offspring is:
        1. **Select Operation**: Sample an operation type (e.g., "edit") from rates config.
        2. **Select Context**:
           - If "mutation": Select 1 parent using standard selector.
           - If "crossover": Select 2 parents using standard selector.
           - If "edit": Find a (Code, FailingTest) pair using specialized logic/helpers.
           - If "det": Find divergent code pairs using specialized logic/helpers.
        3. **Prepare Input**: Construct the specific DTO (e.g., EditInput) with raw strings.
        4. **Execute**: Call `operator.apply(dto)` to get `OperatorOutput`.
        5. **Construct**: Iterate through results, calculating new probabilities and
           creating new Individual objects via the factory.

        Args:
            coevolution_context: Complete system state (populations, matrices) used
                                 for context-aware selection.
            num_offsprings: Number of offspring to generate.

        Returns:
            List of exactly num_offsprings new individuals.

        Empty State Behavior (size=0):
            - If num_offsprings is 0, returns [].
            - If the target population is empty (size=0), the strategy may switch
              to "Bootstrap Mode" (e.g., creating the first tests from code ambiguities)
              instead of standard genetic evolution.

        Raises:
            Exception: Re-raises exceptions from operators to allow the Orchestrator
                       to handle failures (e.g., by retrying or logging).
        """
        ...


class IProbabilityAssigner(Protocol):
    """
    Protocol defining the contract for a probability assignment policy.
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
            operation: The op that created the child (e.g., "crossover", "mutation", "det").
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
