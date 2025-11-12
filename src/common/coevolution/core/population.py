# src/common/coevolution/core/population.py

import numpy as np
from loguru import logger

from .individual import CodeIndividual, TestIndividual
from .interfaces import BasePopulation, IPareto, ITestBlockRebuilder


class CodePopulation(BasePopulation[CodeIndividual]):
    """
    Concrete population class for CodeIndividuals.
    """

    def __repr__(self) -> str:
        return (
            f"<CodePopulation gen={self.generation} size={self.size} "
            f"avg_prob={self.compute_average_probability():.4f}>"
        )

    def _on_generation_advanced(self) -> None:
        """CodePopulation has no special actions on generation advance."""


class TestPopulation(BasePopulation[TestIndividual]):
    """
    Multi-objective population with Pareto-based selection.

    Unlike CodePopulation, tests are selected based on TWO objectives:
    1. Probability (belief in correctness)
    2. Discrimination (ability to distinguish correct/incorrect code)
    """

    def __init__(
        self,
        individuals: list[TestIndividual],
        # injected dependencies
        pareto: IPareto,
        test_block_rebuilder: ITestBlockRebuilder,
        # default values
        test_class_block: str = "",
        generation: int = 0,
    ) -> None:
        if not test_class_block or not test_class_block.strip():
            raise ValueError("test_class_block is required for TestPopulation")

        super().__init__(individuals, generation)
        self._test_class_block = test_class_block

        self._pareto = pareto
        self._test_block_rebuilder = test_block_rebuilder

        logger.trace(f"Initialized TestPopulation with {self.size} individuals")
        logger.trace(f"test class block:\n{self._test_class_block}")

    def _build_test_class_block(self) -> None:
        """Implementation of rebuilding the test class block."""

        logger.debug("Rebuilding test class block")
        self._test_class_block = self._test_block_rebuilder.rebuild_test_block(
            self._test_class_block,
            [ind.snippet for ind in self._individuals],
        )

        logger.trace(f"New test class block:\n{self._test_class_block}")

    def _on_generation_advanced(self) -> None:
        """
        Hook called by set_next_generation.
        Rebuilds test class block.
        """
        logger.debug(f"Rebuilding test class block for gen {self.generation}...")
        self._build_test_class_block()

    @property
    def test_class_block(self) -> str:
        """Get the full unittest class block."""
        return self._test_class_block

    def get_pareto_front(self, observation_matrix: np.ndarray) -> list[TestIndividual]:
        """
        Selects Pareto-optimal individuals based on probabilities and test execution results.

        Args:
            observation_matrix: Execution results from running this population's tests
                              against code population (rows=code, cols=tests)

        Returns:
            List of TestIndividuals that are on the Pareto front.
        """
        logger.debug("Computing Pareto front for TestPopulation...")
        indices: list[int] = self._pareto.get_pareto_indices(
            self.probabilities, observation_matrix
        )
        return [self._individuals[i] for i in indices]

    def __repr__(self) -> str:
        return (
            f"<TestPopulation gen={self.generation} size={self.size} "
            f"avg_prob={self.compute_average_probability():.4f}>"
        )
