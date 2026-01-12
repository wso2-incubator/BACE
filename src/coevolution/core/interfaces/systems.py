# coevolution/core/interfaces/systems.py
"""
System-level protocols for execution, belief updates, and interaction tracking.
"""

from typing import TYPE_CHECKING, Protocol

import numpy as np

from .config import BayesianConfig
from .data import InteractionData

if TYPE_CHECKING:
    from ..population import CodePopulation, TestPopulation


class IExecutionSystem(Protocol):
    """
    Protocol defining the contract for a code-test executor for the entire population.

    The executor now returns an `InteractionData` object which contains both
    the detailed `ExecutionResults` (ID-keyed) and an index-aligned
    `observation_matrix`. This enforces atomic construction of the matrix
    while the executor iterates through populations (guaranteeing index
    alignment and eliminating the split-brain problem).
    """

    def execute_tests(
        self,
        code_population: "CodePopulation",
        test_population: "TestPopulation",
    ) -> InteractionData:
        """
        Executes the given code snippets against the test snippets and returns
        a unified `InteractionData` artifact.

        Args:
                code_population: The population of code snippets to be tested.
                test_population: The population of test snippets to run against the code.

        Returns:
                An `InteractionData` instance containing both:
                    - `execution_results`: ID-keyed mapping of execution outcomes
                    - `observation_matrix`: numpy ndarray with shape
                        (len(code_population), len(test_population)) such that
                        `observation_matrix[i, j]` corresponds to
                        `code_population[i]` vs `test_population[j]`.

        Consistency guarantees:
                - The executor constructs the matrix in the same loop that
                    produces the `execution_results`, ensuring strict alignment.

        Empty State Behavior (size=0):
                - When `test_population` is empty, return an `InteractionData` with
                    an empty `execution_results` and an appropriately shaped empty
                    numpy array (shape (n_codes, 0)).
                - When `code_population` is empty, return an `InteractionData` with
                    empty `execution_results` and an empty numpy array (shape (0, n_tests)).
        """
        ...


class IBeliefUpdater(Protocol):
    """
    Protocol defining the contract for Bayesian belief update strategies.

    Provides separate methods for updating code and test beliefs, making it
    explicit which population is being updated and avoiding confusion about
    parameter ordering and matrix transformations.
    """

    def update_code_beliefs(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        code_update_mask_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        """
        Update beliefs for the code population based on test results.

        Args:
            prior_code_probs: Prior probabilities for the code population.
            prior_test_probs: Prior probabilities for the test population.
            observation_matrix: Matrix of interactions (rows=code, cols=tests).
            code_update_mask_matrix: Matrix indicating which observations to consider for updates.
            config: A BayesianConfig object containing hyperparameters
                    (alpha, beta, gamma, learning_rate).

        Returns:
            Updated posterior probabilities for the code population.

        Empty State Behavior (size=0):
            - Identity Operation: When observation_matrix has shape (N, 0) (no tests),
              should return posterior = prior_code_probs unchanged (no evidence).
            - When prior_code_probs is empty (shape (0,)), should return empty array
              (shape (0,)) representing no code individuals to update.
            - This supports differential testing where test population may start empty.
        """
        ...

    def update_test_beliefs(
        self,
        prior_code_probs: np.ndarray,
        prior_test_probs: np.ndarray,
        observation_matrix: np.ndarray,
        test_update_mask_matrix: np.ndarray,
        config: BayesianConfig,
    ) -> np.ndarray:
        """
        Update beliefs for the test population based on code results.

        Args:
            prior_code_probs: Prior probabilities for the code population.
            prior_test_probs: Prior probabilities for the test population.
            observation_matrix: Matrix of interactions (rows=code, cols=tests).
            test_update_mask_matrix: Matrix indicating which observations to consider for updates.
            config: A BayesianConfig object containing hyperparameters
                    (alpha, beta, gamma, learning_rate).

        Returns:
            Updated posterior probabilities for the test population.

        Empty State Behavior (size=0):
            - Identity Operation: When observation_matrix has shape (0, N) (no code),
              should return posterior = prior_test_probs unchanged (no evidence).
            - When prior_test_probs is empty (shape (0,)), should return empty array
              (shape (0,)) representing no test individuals to update.
            - This supports differential testing where test population may start empty.
        """
        ...


# TODO: This was implemented in a hurry, revisit and refine later.
class IInteractionLedger(Protocol):
    def get_new_interaction_mask(
        self, code_ids: list[str], test_ids: list[str], test_type: str, target: str
    ) -> np.ndarray: ...

    def commit_interactions(
        self,
        code_ids: list[str],
        test_ids: list[str],
        test_type: str,
        target: str,
        mask: np.ndarray,
    ) -> None: ...


class LedgerFactory(Protocol):
    def __call__(self) -> IInteractionLedger: ...
