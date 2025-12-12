# src/common/coevolution/core/population.py

from loguru import logger

from .individual import CodeIndividual, TestIndividual
from .interfaces import BasePopulation, ITestBlockRebuilder


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
    Population class for TestIndividuals.

    Manages test individuals and rebuilds the test class block when the generation advances.

    Note: Elite selection logic has been moved to IEliteSelectionStrategy implementations,
    making this class independent of any specific selection strategy (e.g., Pareto).
    """

    def __init__(
        self,
        individuals: list[TestIndividual],
        # injected dependencies
        test_block_rebuilder: ITestBlockRebuilder,
        # default values
        test_class_block: str = "",
        generation: int = 0,
    ) -> None:
        if not test_class_block or not test_class_block.strip():
            raise ValueError("test_class_block is required for TestPopulation")

        super().__init__(individuals, generation)
        self._test_class_block = test_class_block
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

    def __repr__(self) -> str:
        return (
            f"<TestPopulation gen={self.generation} size={self.size} "
            f"avg_prob={self.compute_average_probability():.4f}>"
        )
