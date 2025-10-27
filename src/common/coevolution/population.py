"""
Population management for coevolutionary algorithms.

This module provides population classes for managing code solutions or test cases
in evolutionary algorithms with Bayesian belief updating.

Classes:
    BasePopulation: Abstract base class for all population types
    CodePopulation: Concrete class for code solution populations
    TestPopulation: Concrete class for test case populations
"""

from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np
from loguru import logger

from common.code_preprocessing.builders import rebuild_unittest_with_new_methods

from .selection import SelectionStrategy


class BasePopulation(ABC):
    """
    Abstract base class for population management in coevolutionary algorithms.

    This class encapsulates the common behavior for both code and test populations,
    including individual management, probability tracking, and Bayesian belief updating.

    Attributes:
        individuals: List of code strings or test case strings
        probabilities: Numpy array of correctness probabilities for each individual
        size: Population size (automatically updated when individuals change)
        generation: Current generation number
    """

    def __init__(
        self,
        individuals: List[str],
        probabilities: np.ndarray,
        generation: int = 0,
    ) -> None:
        """
        Initialize a BasePopulation.

        Args:
            individuals: List of code strings or test case strings
            probabilities: Numpy array of correctness probabilities
            generation: Current generation number (default: 0)

        Raises:
            ValueError: If validation checks fail
        """
        # Validate inputs
        if len(individuals) != len(probabilities):
            msg = f"Number of individuals ({len(individuals)}) must match number of probabilities ({len(probabilities)})"
            logger.error(msg)
            raise ValueError(msg)

        if len(individuals) == 0:
            msg = "Population cannot be empty"
            logger.error(msg)
            raise ValueError(msg)

        self._individuals = individuals
        self._probabilities = probabilities
        self.generation = generation

        logger.debug(
            f"Initialized {self.__class__.__name__}: size={len(individuals)}, "
            f"generation={generation}, avg_prob={np.mean(probabilities):.4f}"
        )

    @property
    def individuals(self) -> List[str]:
        """Get the list of individuals."""
        return self._individuals.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """Get the probability array."""
        return self._probabilities.copy()

    @property
    def size(self) -> int:
        """Get the current population size."""
        return len(self._individuals)

    def _validate_consistency(self) -> None:
        """Internal method to validate population consistency."""
        if len(self._individuals) != len(self._probabilities):
            raise ValueError(
                f"Population consistency violated: {len(self._individuals)} individuals "
                f"but {len(self._probabilities)} probabilities"
            )

    def get_average_probability(self) -> float:
        """Return the average correctness probability for this population.

        Returns 0.0 for empty populations (though populations should not be empty).
        """
        if len(self._probabilities) == 0:
            return 0.0
        return float(np.mean(self._probabilities))

    def get_best_individual(self) -> tuple[str, float]:
        """
        Get the individual with the highest correctness probability.

        Returns:
            Tuple of (best_individual, best_probability)
        """
        best_idx = int(np.argmax(self._probabilities))
        best_prob = float(self._probabilities[best_idx])

        logger.trace(
            f"Best individual at generation {self.generation}: "
            f"index={best_idx}, prob={best_prob:.4f}"
        )

        return self._individuals[best_idx], best_prob

    def get_top_k_individuals(self, k: int) -> List[tuple[str, float]]:
        """
        Get the top k individuals by correctness probability.

        Args:
            k: Number of top individuals to retrieve

        Returns:
            List of (individual, probability) tuples, sorted by probability (descending)
        """
        if k > len(self._individuals):
            k = len(self._individuals)
        top_indices = np.argsort(self._probabilities)[-k:][::-1]
        return [
            (self._individuals[int(idx)], float(self._probabilities[idx]))
            for idx in top_indices
        ]

    def update_probabilities(self, new_probabilities: np.ndarray) -> None:
        """
        Update the correctness probabilities for all individuals.

        Args:
            new_probabilities: New probability array

        Raises:
            ValueError: If the size doesn't match the population size
        """
        if len(new_probabilities) != len(self._individuals):
            msg = f"New probabilities size ({len(new_probabilities)}) must match population size ({len(self._individuals)})"
            logger.error(msg)
            raise ValueError(msg)

        old_avg = np.mean(self._probabilities)
        self._probabilities = new_probabilities
        new_avg = np.mean(new_probabilities)

        logger.debug(
            f"Updated probabilities for generation {self.generation}: "
            f"avg {old_avg:.4f} → {new_avg:.4f} (Δ{new_avg - old_avg:+.4f})"
        )

    def increment_generation(self) -> None:
        """Increment the generation counter."""
        self.generation += 1
        logger.info(
            f"Advanced to generation {self.generation}: "
            f"size={len(self._individuals)}, avg_prob={np.mean(self._probabilities):.4f}"
        )

    def __len__(self) -> int:
        """Get the size of the population."""
        return len(self._individuals)

    def __getitem__(self, index: int) -> tuple[str, float]:
        """
        Get an individual and its probability by index.

        Args:
            index: Index of individual to retrieve

        Returns:
            Tuple of (individual, probability)
        """
        return self._individuals[index], float(self._probabilities[index])

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """
        Iterate over individuals and their probabilities.

        Yields:
            Tuple of (individual, probability) for each individual in the population

        Example:
            >>> pop = CodePopulation(['code1', 'code2'], np.array([0.5, 0.7]))
            >>> for individual, prob in pop:
            ...     print(f"{individual}: {prob}")
            code1: 0.5
            code2: 0.7
        """
        for i in range(len(self._individuals)):
            yield self._individuals[i], float(self._probabilities[i])

    # Abstract methods that subclasses must implement
    @abstractmethod
    def add_individual(self, individual: str, probability: float) -> None:
        """
        Add a single individual to the population.

        Args:
            individual: Code string or test method to add
            probability: Correctness probability for this individual

        Raises:
            ValueError: If probability is not in [0, 1]
        """
        pass

    @abstractmethod
    def remove_individual(self, index: int) -> tuple[str, float]:
        """
        Remove an individual at the specified index.

        Args:
            index: Index of individual to remove

        Returns:
            Tuple of (removed_individual, removed_probability)

        Raises:
            ValueError: If population would become empty or index is invalid
            IndexError: If index is out of bounds
        """
        pass

    @abstractmethod
    def replace_individual(
        self, index: int, individual: str, probability: float
    ) -> None:
        """
        Replace a single individual at the specified index.

        Args:
            index: Index of individual to replace
            individual: New code string or test method
            probability: New correctness probability

        Raises:
            ValueError: If probability is not in [0, 1]
            IndexError: If index is out of bounds
        """
        pass

    @abstractmethod
    def replace_individuals(
        self, new_individuals: List[str], new_probabilities: np.ndarray
    ) -> None:
        """
        Replace the entire population with new individuals and probabilities.

        Args:
            new_individuals: New list of individuals
            new_probabilities: New probability array

        Raises:
            ValueError: If validation checks fail
        """
        pass


# ============================================================================
# Concrete Population Implementations
# ============================================================================


class CodePopulation(BasePopulation):
    """
    Population of code solutions for evolutionary algorithms.

    This class represents a collection of independent code solutions, each with
    its own correctness probability. Code solutions are standalone and don't
    have dependencies on each other.

    Example:
        >>> solutions = ["def add(a,b): return a+b", "def add(a,b): return a-b"]
        >>> probs = np.array([0.9, 0.3])
        >>> pop = CodePopulation(solutions, probs)
        >>> pop.size
        2
        >>> best_solution, best_prob = pop.get_best_individual()
    """

    def __repr__(self) -> str:
        """String representation of the code population."""
        return (
            f"CodePopulation(size={len(self._individuals)}, "
            f"generation={self.generation}, avg_prob={np.mean(self._probabilities):.4f})"
        )

    def replace_individuals(
        self, new_individuals: List[str], new_probabilities: np.ndarray
    ) -> None:
        """
        Replace the entire code population with new individuals and probabilities.

        This is useful for generational replacement in evolutionary algorithms.

        Args:
            new_individuals: New list of code solutions
            new_probabilities: New probability array

        Raises:
            ValueError: If validation checks fail
        """
        if len(new_individuals) != len(new_probabilities):
            msg = f"Number of new individuals ({len(new_individuals)}) must match number of new probabilities ({len(new_probabilities)})"
            logger.error(msg)
            raise ValueError(msg)

        if len(new_individuals) == 0:
            msg = "Cannot replace with empty population"
            logger.error(msg)
            raise ValueError(msg)

        old_size = len(self._individuals)
        old_avg = np.mean(self._probabilities)

        self._individuals = new_individuals
        self._probabilities = new_probabilities
        self._validate_consistency()

        new_avg = np.mean(new_probabilities)
        logger.info(
            f"Replaced CodePopulation at generation {self.generation}: "
            f"size {old_size} → {len(new_individuals)}, "
            f"avg_prob {old_avg:.4f} → {new_avg:.4f}"
        )

    def add_individual(self, individual: str, probability: float) -> None:
        """
        Add a single code solution to the population.

        Args:
            individual: Code solution string to add
            probability: Correctness probability for this solution

        Raises:
            ValueError: If probability is not in [0, 1]
        """
        if not 0 <= probability <= 1:
            msg = f"Probability must be in [0, 1], got {probability}"
            logger.error(msg)
            raise ValueError(msg)

        self._individuals.append(individual)
        self._probabilities = np.append(self._probabilities, probability)
        self._validate_consistency()

        logger.debug(
            f"Added individual to CodePopulation (generation {self.generation}): "
            f"new_size={len(self._individuals)}, prob={probability:.4f}"
        )

    def remove_individual(self, index: int) -> tuple[str, float]:
        """
        Remove a code solution at the specified index.

        Args:
            index: Index of solution to remove

        Returns:
            Tuple of (removed_individual, removed_probability)

        Raises:
            ValueError: If population would become empty
            IndexError: If index is out of bounds
        """
        if len(self._individuals) <= 1:
            msg = "Cannot remove from population with only one individual"
            logger.error(msg)
            raise ValueError(msg)
        if index < 0 or index >= len(self._individuals):
            msg = f"Index {index} out of bounds for population size {len(self._individuals)}"
            logger.error(msg)
            raise IndexError(msg)

        removed_individual = self._individuals.pop(index)
        removed_probability = float(self._probabilities[index])
        self._probabilities = np.delete(self._probabilities, index)
        self._validate_consistency()

        logger.debug(
            f"Removed individual at index {index} from CodePopulation "
            f"(generation {self.generation}): prob={removed_probability:.4f}, "
            f"new_size={len(self._individuals)}"
        )

        return removed_individual, removed_probability

    def replace_individual(
        self, index: int, individual: str, probability: float
    ) -> None:
        """
        Replace a code solution at the specified index.

        Args:
            index: Index of solution to replace
            individual: New code solution string
            probability: New correctness probability

        Raises:
            ValueError: If probability is not in [0, 1]
            IndexError: If index is out of bounds
        """
        if not 0 <= probability <= 1:
            msg = f"Probability must be in [0, 1], got {probability}"
            logger.error(msg)
            raise ValueError(msg)
        if index < 0 or index >= len(self._individuals):
            msg = f"Index {index} out of bounds for population size {len(self._individuals)}"
            logger.error(msg)
            raise IndexError(msg)

        old_prob = float(self._probabilities[index])
        self._individuals[index] = individual
        self._probabilities[index] = probability

        logger.debug(
            f"Replaced individual at index {index} in CodePopulation "
            f"(generation {self.generation}): prob {old_prob:.4f} → {probability:.4f}"
        )


class TestPopulation(BasePopulation):
    """
    Population of test cases for evolutionary algorithms.

    This class represents a decomposed unittest class where individual test methods
    are tracked separately but maintain a relationship through the full test_class_block.
    The test_class_block is the single source of truth, and individuals are extracted
    test methods from it.

    Attributes:
        test_class_block: Full unittest class containing all test methods

    Example:
        >>> test_methods = ["def test_add(self): ...", "def test_sub(self): ..."]
        >>> full_class = "class TestMath(unittest.TestCase):\\n    def test_add..."
        >>> probs = np.array([0.8, 0.7])
        >>> pop = TestPopulation(test_methods, probs, test_class_block=full_class)
        >>> pop.test_class_block
        'class TestMath...'
    """

    def __init__(
        self,
        individuals: List[str],
        probabilities: np.ndarray,
        generation: int = 0,
        test_class_block: str = "",
    ) -> None:
        """
        Initialize a TestPopulation.

        Args:
            individuals: List of test method strings (extracted from test_class_block)
            probabilities: Numpy array of correctness probabilities
            generation: Current generation number (default: 0)
            test_class_block: Full unittest class code containing all test methods

        Raises:
            ValueError: If validation checks fail or test_class_block is empty
        """
        if not test_class_block or len(test_class_block.strip()) == 0:
            msg = "test_class_block is required and cannot be empty for TestPopulation"
            logger.error(msg)
            raise ValueError(msg)

        super().__init__(individuals, probabilities, generation)
        self._test_class_block = test_class_block
        # Per-test discriminative power (values in [0, 1]). Initialized to 0.0
        # by default; callers should compute and set real values each iteration
        # using `set_discriminations`.
        self._discriminations = np.zeros(len(self._individuals), dtype=float)
        self._discriminations_set = False
        # Validate the newly created discriminations array
        self._validate_discriminations_consistency()

    @property
    def test_class_block(self) -> str:
        """Get the full unittest class block."""
        return self._test_class_block

    @property
    def discriminations(self) -> np.ndarray:
        """Get the per-test discriminative power array (values in [0, 1])."""
        return self._discriminations

    def set_test_class_block(self, test_class_block: str) -> None:
        """
        Set or update the full unittest class block.

        Args:
            test_class_block: The complete unittest class code

        Raises:
            ValueError: If test_class_block is empty

        Note:
            This should typically only be called when the test methods have changed
            and the class block needs to be regenerated to match.
        """
        if not test_class_block or len(test_class_block.strip()) == 0:
            msg = "test_class_block cannot be empty"
            logger.error(msg)
            raise ValueError(msg)
        self._test_class_block = test_class_block

    def _validate_discriminations_consistency(self) -> None:
        """Validate that discriminations length matches the population individuals.

        Raises:
            ValueError: if lengths do not match
        """
        if len(self._individuals) != len(self._discriminations):
            msg = (
                f"Discriminations consistency violated: {len(self._individuals)} individuals "
                f"but {len(self._discriminations)} discriminations"
            )
            logger.error(msg)
            raise ValueError(msg)

    def set_discriminations(self, new_discriminations: np.ndarray) -> None:
        """Set the discriminative power values for all test methods.

        Args:
            new_discriminations: Sequence or numpy array of floats in [0, 1]

        Raises:
            ValueError: If length doesn't match population size or values are invalid
        """
        logger.debug(
            f"Setting new discriminations for {len(self._individuals)} individuals"
        )
        # --- Validation Checks ---
        if new_discriminations.ndim != 1 or len(new_discriminations) != len(
            self._individuals
        ):
            msg = (
                f"New discriminations size ({new_discriminations.shape}) must match "
                f"population size ({len(self._individuals)})"
            )
            logger.error(msg)
            raise ValueError(msg)

        if not np.all(np.isfinite(new_discriminations)):
            msg = "Discriminations contain non-finite values"
            logger.error(msg)
            raise ValueError(msg)

        if np.any(new_discriminations < 0) or np.any(new_discriminations > 1):
            msg = "Discriminations must be in the range [0, 1]"
            logger.error(msg)
            raise ValueError(msg)

        # --- State Update ---
        old_avg = (
            float(np.mean(self._discriminations))
            if len(self._discriminations) > 0
            else 0.0
        )
        self._discriminations = new_discriminations
        self._discriminations_set = True
        new_avg = (
            float(np.mean(self._discriminations))
            if len(self._discriminations) > 0
            else 0.0
        )

        # This existing log is great, captures the state change perfectly
        logger.debug(
            f"Updated discriminations for generation {self.generation}: "
            f"avg {old_avg:.4f} → {new_avg:.4f} (Δ{new_avg - old_avg:+.4f})"
        )

    def get_pareto_front(self) -> List[tuple[str, float]]:
        """Return Pareto-optimal individuals as (individual, probability) tuples.

        This mirrors the `get_top_k_individuals` API and returns the actual
        (individual, probability) tuples for the Pareto front where both
        probability and discrimination are maximized.

        Returns:
            List of (individual, probability) tuples corresponding to Pareto front.

        Raises:
            ValueError: if discriminations are missing or inconsistent
        """
        logger.debug("Calculating Pareto front...")

        # --- State Check ---
        if not self._discriminations_set:
            msg = (
                "Cannot get Pareto front: Discriminations have not been set. "
                "Call set_discriminations() after modifying the population "
                "and before calling this method."
            )

            logger.error(msg)
            raise RuntimeError(msg)

        # --- Validation ---
        try:
            self._validate_discriminations_consistency()
        except ValueError as e:
            logger.error(f"Cannot get Pareto front: {e}")
            raise  # Re-raise the exception after logging

        # --- Initial Front Calculation ---
        indices = SelectionStrategy.pareto_front(
            self._probabilities, self._discriminations
        )
        logger.debug(f"Found {len(indices)} initial Pareto-optimal individuals")

        # --- Filtering Step ---
        avg_prob = self.get_average_probability()
        logger.debug(f"Filtering front by average probability > {avg_prob:.4f}")

        filtered = [idx for idx in indices if self._probabilities[int(idx)] > avg_prob]

        if filtered:
            logger.debug(
                f"Filtered front from {len(indices)} to {len(filtered)} individuals"
            )
            indices = filtered
        else:
            # TODO: This step may not be required as pareto front should ideally contain
            # at least one individual above average probability, unless all are equal

            logger.debug(
                f"No individuals above average probability; "
                f"returning original front of {len(indices)} individuals"
            )

        # --- Final Result ---
        results = [
            (self._individuals[int(idx)], float(self._probabilities[int(idx)]))
            for idx in indices
        ]

        logger.debug(f"Returning {len(results)} individuals in final Pareto front")
        return results

    def _rebuild_test_class_block(self) -> None:
        """
        Internal method to rebuild the test class block from current individuals.

        This method is called automatically whenever individuals are modified
        (added, removed, or replaced) to keep the test_class_block in sync.

        Uses rebuild_unittest_with_new_methods() from builders.py to preserve
        the class structure, imports, and non-test methods while updating test methods.
        """
        self._test_class_block = rebuild_unittest_with_new_methods(
            self._test_class_block, self._individuals
        )
        logger.trace(
            f"Rebuilt test class block with {len(self._individuals)} test methods"
        )

    def __repr__(self) -> str:
        """String representation of the test population."""
        return (
            f"TestPopulation(size={len(self._individuals)}, "
            f"generation={self.generation}, avg_prob={np.mean(self._probabilities):.4f}, "
            f"has_test_class=True)"
        )

    def replace_individuals(
        self, new_individuals: List[str], new_probabilities: np.ndarray
    ) -> None:
        """
        Replace the entire test population with new individuals and probabilities.

        This is useful for generational replacement in evolutionary algorithms.
        The test_class_block will be automatically rebuilt from the new individuals.

        Args:
            new_individuals: New list of test method strings
            new_probabilities: New probability array

        Raises:
            ValueError: If validation checks fail
        """
        if len(new_individuals) != len(new_probabilities):
            msg = (
                f"Number of new individuals ({len(new_individuals)}) must match "
                f"number of new probabilities ({len(new_probabilities)})"
            )
            logger.error(msg)
            raise ValueError(msg)
        if len(new_individuals) == 0:
            msg = "Cannot replace with empty population"
            logger.error(msg)
            raise ValueError(msg)

        old_size = len(self._individuals)
        old_avg = np.mean(self._probabilities)

        self._individuals = new_individuals
        self._probabilities = new_probabilities
        # Reset discriminations for the new population; they should be recomputed
        # externally each iteration and set via `set_discriminations`.
        self._discriminations = np.zeros(len(new_individuals), dtype=float)
        self._discriminations_set = False
        self._rebuild_test_class_block()
        self._validate_consistency()
        self._validate_discriminations_consistency()

        new_avg = np.mean(new_probabilities)
        logger.info(
            f"Replaced TestPopulation at generation {self.generation}: "
            f"size {old_size} → {len(new_individuals)}, "
            f"avg_prob {old_avg:.4f} → {new_avg:.4f}"
        )

    def add_individual(self, individual: str, probability: float) -> None:
        """
        Add a single test method to the population.

        The test_class_block will be automatically rebuilt to include the new test method.

        Args:
            individual: Test method string to add
            probability: Correctness probability for this test

        Raises:
            ValueError: If probability is not in [0, 1]
        """
        if not 0 <= probability <= 1:
            msg = f"Probability must be in [0, 1], got {probability}"
            logger.error(msg)
            raise ValueError(msg)

        self._individuals.append(individual)
        self._probabilities = np.append(self._probabilities, probability)
        # New test methods start with neutral/unknown discrimination (0.0)
        self._discriminations = np.append(self._discriminations, 0.0)
        self._discriminations_set = False
        self._rebuild_test_class_block()
        self._validate_consistency()
        self._validate_discriminations_consistency()

        logger.debug(
            f"Added individual to TestPopulation (generation {self.generation}): "
            f"new_size={len(self._individuals)}, prob={probability:.4f}"
        )

    def remove_individual(self, index: int) -> tuple[str, float]:
        """
        Remove a test method at the specified index.

        The test_class_block will be automatically rebuilt without the removed test method.

        Args:
            index: Index of test method to remove

        Returns:
            Tuple of (removed_individual, removed_probability)

        Raises:
            ValueError: If population would become empty
            IndexError: If index is out of bounds
        """
        if len(self._individuals) <= 1:
            msg = "Cannot remove from population with only one individual"
            logger.error(msg)
            raise ValueError(msg)
        if index < 0 or index >= len(self._individuals):
            msg = f"Index {index} out of bounds for population size {len(self._individuals)}"
            logger.error(msg)
            raise IndexError(msg)

        removed_individual = self._individuals.pop(index)
        removed_probability = float(self._probabilities[index])
        self._probabilities = np.delete(self._probabilities, index)
        # Keep discriminations aligned with individuals
        self._discriminations = np.delete(self._discriminations, index)
        self._rebuild_test_class_block()
        self._validate_consistency()
        self._validate_discriminations_consistency()

        logger.debug(
            f"Removed individual at index {index} from TestPopulation "
            f"(generation {self.generation}): prob={removed_probability:.4f}, "
            f"new_size={len(self._individuals)}"
        )

        return removed_individual, removed_probability

    def replace_individual(
        self, index: int, individual: str, probability: float
    ) -> None:
        """
        Replace a test method at the specified index.

        The test_class_block will be automatically rebuilt with the replaced test method.

        Args:
            index: Index of test method to replace
            individual: New test method string
            probability: New correctness probability

        Raises:
            ValueError: If probability is not in [0, 1]
            IndexError: If index is out of bounds
        """
        if not 0 <= probability <= 1:
            msg = f"Probability must be in [0, 1], got {probability}"
            logger.error(msg)
            raise ValueError(msg)
        if index < 0 or index >= len(self._individuals):
            msg = f"Index {index} out of bounds for population size {len(self._individuals)}"
            logger.error(msg)
            raise IndexError(msg)

        old_prob = float(self._probabilities[index])
        self._individuals[index] = individual
        self._probabilities[index] = probability
        # Replaced individual likely needs a recomputed discrimination value
        # Reset to 0.0 until externally set.
        self._discriminations[index] = 0.0
        self._discriminations_set = False
        self._rebuild_test_class_block()
        self._validate_discriminations_consistency()
        logger.debug(
            f"Replaced individual at index {index} in TestPopulation "
            f"(generation {self.generation}): prob {old_prob:.4f} → {probability:.4f}"
        )
