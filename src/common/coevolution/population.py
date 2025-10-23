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
from typing import Any, Iterator, List

import numpy as np
from loguru import logger


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
            raise ValueError(
                f"Number of individuals ({len(individuals)}) must match "
                f"number of probabilities ({len(probabilities)})"
            )
        if len(individuals) == 0:
            raise ValueError("Population cannot be empty")

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
        return self._individuals

    @property
    def probabilities(self) -> np.ndarray:
        """Get the probability array."""
        return self._probabilities

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
            raise ValueError(
                f"New probabilities size ({len(new_probabilities)}) must match "
                f"population size ({len(self._individuals)})"
            )

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
    def add_individual(
        self, individual: str, probability: float, **kwargs: Any
    ) -> None:
        """
        Add a single individual to the population.

        Args:
            individual: Code string or test method to add
            probability: Correctness probability for this individual
            **kwargs: Additional subclass-specific parameters

        Raises:
            ValueError: If probability is not in [0, 1]
        """
        pass

    @abstractmethod
    def remove_individual(self, index: int, **kwargs: Any) -> tuple[str, float]:
        """
        Remove an individual at the specified index.

        Args:
            index: Index of individual to remove
            **kwargs: Additional subclass-specific parameters

        Returns:
            Tuple of (removed_individual, removed_probability)

        Raises:
            ValueError: If population would become empty or index is invalid
            IndexError: If index is out of bounds
        """
        pass

    @abstractmethod
    def replace_individual(
        self, index: int, individual: str, probability: float, **kwargs: Any
    ) -> None:
        """
        Replace a single individual at the specified index.

        Args:
            index: Index of individual to replace
            individual: New code string or test method
            probability: New correctness probability
            **kwargs: Additional subclass-specific parameters

        Raises:
            ValueError: If probability is not in [0, 1]
            IndexError: If index is out of bounds
        """
        pass

    @abstractmethod
    def replace_individuals(
        self,
        new_individuals: List[str],
        new_probabilities: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Replace the entire population with new individuals and probabilities.

        Args:
            new_individuals: New list of individuals
            new_probabilities: New probability array
            **kwargs: Additional subclass-specific parameters

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
        self,
        new_individuals: List[str],
        new_probabilities: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Replace the entire code population with new individuals and probabilities.

        This is useful for generational replacement in evolutionary algorithms.

        Args:
            new_individuals: New list of code solutions
            new_probabilities: New probability array
            **kwargs: Ignored for code populations

        Raises:
            ValueError: If validation checks fail
        """
        if len(new_individuals) != len(new_probabilities):
            raise ValueError(
                f"Number of new individuals ({len(new_individuals)}) must match "
                f"number of new probabilities ({len(new_probabilities)})"
            )
        if len(new_individuals) == 0:
            raise ValueError("Cannot replace with empty population")

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

    def add_individual(
        self, individual: str, probability: float, **kwargs: Any
    ) -> None:
        """
        Add a single code solution to the population.

        Args:
            individual: Code solution string to add
            probability: Correctness probability for this solution
            **kwargs: Ignored for code populations

        Raises:
            ValueError: If probability is not in [0, 1]
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")

        self._individuals.append(individual)
        self._probabilities = np.append(self._probabilities, probability)
        self._validate_consistency()

        logger.debug(
            f"Added individual to CodePopulation (generation {self.generation}): "
            f"new_size={len(self._individuals)}, prob={probability:.4f}"
        )

    def remove_individual(self, index: int, **kwargs: Any) -> tuple[str, float]:
        """
        Remove a code solution at the specified index.

        Args:
            index: Index of solution to remove
            **kwargs: Ignored for code populations

        Returns:
            Tuple of (removed_individual, removed_probability)

        Raises:
            ValueError: If population would become empty
            IndexError: If index is out of bounds
        """
        if len(self._individuals) <= 1:
            raise ValueError("Cannot remove from population with only one individual")
        if index < 0 or index >= len(self._individuals):
            raise IndexError(
                f"Index {index} out of bounds for population size {len(self._individuals)}"
            )

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
        self, index: int, individual: str, probability: float, **kwargs: Any
    ) -> None:
        """
        Replace a code solution at the specified index.

        Args:
            index: Index of solution to replace
            individual: New code solution string
            probability: New correctness probability
            **kwargs: Ignored for code populations

        Raises:
            ValueError: If probability is not in [0, 1]
            IndexError: If index is out of bounds
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        if index < 0 or index >= len(self._individuals):
            raise IndexError(
                f"Index {index} out of bounds for population size {len(self._individuals)}"
            )

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
            raise ValueError(
                "test_class_block is required and cannot be empty for TestPopulation. "
                "Test populations must have a complete unittest class."
            )

        super().__init__(individuals, probabilities, generation)
        self._test_class_block = test_class_block

    @property
    def test_class_block(self) -> str:
        """Get the full unittest class block."""
        return self._test_class_block

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
            raise ValueError("test_class_block cannot be empty")
        self._test_class_block = test_class_block

    def __repr__(self) -> str:
        """String representation of the test population."""
        return (
            f"TestPopulation(size={len(self._individuals)}, "
            f"generation={self.generation}, avg_prob={np.mean(self._probabilities):.4f}, "
            f"has_test_class=True)"
        )

    def replace_individuals(
        self,
        new_individuals: List[str],
        new_probabilities: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Replace the entire test population with new individuals and probabilities.

        This is useful for generational replacement in evolutionary algorithms.

        Args:
            new_individuals: New list of test method strings
            new_probabilities: New probability array
            **kwargs: Must include 'new_test_class_block' parameter

        Raises:
            ValueError: If validation checks fail or new_test_class_block is missing
        """
        if len(new_individuals) != len(new_probabilities):
            raise ValueError(
                f"Number of new individuals ({len(new_individuals)}) must match "
                f"number of new probabilities ({len(new_probabilities)})"
            )
        if len(new_individuals) == 0:
            raise ValueError("Cannot replace with empty population")

        # Require new_test_class_block for test populations
        new_test_class_block = kwargs.get("new_test_class_block")
        if new_test_class_block is None:
            raise ValueError(
                "new_test_class_block is required when replacing a TestPopulation. "
                "The test class block must be provided for the new test methods."
            )

        old_size = len(self._individuals)
        old_avg = np.mean(self._probabilities)

        self._individuals = new_individuals
        self._probabilities = new_probabilities
        self._test_class_block = new_test_class_block
        self._validate_consistency()

        new_avg = np.mean(new_probabilities)
        logger.info(
            f"Replaced TestPopulation at generation {self.generation}: "
            f"size {old_size} → {len(new_individuals)}, "
            f"avg_prob {old_avg:.4f} → {new_avg:.4f}"
        )

    def add_individual(
        self, individual: str, probability: float, **kwargs: Any
    ) -> None:
        """
        Add a single test method to the population.

        Args:
            individual: Test method string to add
            probability: Correctness probability for this test
            **kwargs: Must include 'updated_test_class_block' parameter

        Raises:
            ValueError: If probability is not in [0, 1] or updated_test_class_block is missing
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")

        # Require updated test_class_block for test populations
        updated_test_class_block = kwargs.get("updated_test_class_block")
        if updated_test_class_block is None:
            raise ValueError(
                "updated_test_class_block is required when adding to a TestPopulation. "
                "The test class block must be regenerated to include the new test method."
            )

        self._individuals.append(individual)
        self._probabilities = np.append(self._probabilities, probability)
        self._test_class_block = updated_test_class_block
        self._validate_consistency()

        logger.debug(
            f"Added individual to TestPopulation (generation {self.generation}): "
            f"new_size={len(self._individuals)}, prob={probability:.4f}"
        )

    def remove_individual(self, index: int, **kwargs: Any) -> tuple[str, float]:
        """
        Remove a test method at the specified index.

        Args:
            index: Index of test method to remove
            **kwargs: Must include 'updated_test_class_block' parameter

        Returns:
            Tuple of (removed_individual, removed_probability)

        Raises:
            ValueError: If population would become empty or updated_test_class_block is missing
            IndexError: If index is out of bounds
        """
        if len(self._individuals) <= 1:
            raise ValueError("Cannot remove from population with only one individual")
        if index < 0 or index >= len(self._individuals):
            raise IndexError(
                f"Index {index} out of bounds for population size {len(self._individuals)}"
            )

        # Require updated test_class_block for test populations
        updated_test_class_block = kwargs.get("updated_test_class_block")
        if updated_test_class_block is None:
            raise ValueError(
                "updated_test_class_block is required when removing from a TestPopulation. "
                "The test class block must be regenerated without the removed test method."
            )

        removed_individual = self._individuals.pop(index)
        removed_probability = float(self._probabilities[index])
        self._probabilities = np.delete(self._probabilities, index)
        self._test_class_block = updated_test_class_block
        self._validate_consistency()

        logger.debug(
            f"Removed individual at index {index} from TestPopulation "
            f"(generation {self.generation}): prob={removed_probability:.4f}, "
            f"new_size={len(self._individuals)}"
        )

        return removed_individual, removed_probability

    def replace_individual(
        self, index: int, individual: str, probability: float, **kwargs: Any
    ) -> None:
        """
        Replace a test method at the specified index.

        Args:
            index: Index of test method to replace
            individual: New test method string
            probability: New correctness probability
            **kwargs: Must include 'updated_test_class_block' parameter

        Raises:
            ValueError: If probability is not in [0, 1] or updated_test_class_block is missing
            IndexError: If index is out of bounds
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        if index < 0 or index >= len(self._individuals):
            raise IndexError(
                f"Index {index} out of bounds for population size {len(self._individuals)}"
            )

        # Require updated test_class_block for test populations
        updated_test_class_block = kwargs.get("updated_test_class_block")
        if updated_test_class_block is None:
            raise ValueError(
                "updated_test_class_block is required when replacing in a TestPopulation. "
                "The test class block must be regenerated with the replaced test method."
            )

        old_prob = float(self._probabilities[index])
        self._individuals[index] = individual
        self._probabilities[index] = probability
        self._test_class_block = updated_test_class_block

        logger.debug(
            f"Replaced individual at index {index} in TestPopulation "
            f"(generation {self.generation}): prob {old_prob:.4f} → {probability:.4f}"
        )
