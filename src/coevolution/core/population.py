# src/common/coevolution/core/population.py

from loguru import logger

from .individual import CodeIndividual, TestIndividual
from .interfaces import BasePopulation


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

    Manages test individuals. Each test function is executed individually,
    so there's no need to maintain a combined test block.

    Note: Elite selection logic has been moved to IEliteSelectionStrategy implementations,
    making this class independent of any specific selection strategy (e.g., Pareto).
    """

    def __init__(
        self,
        individuals: list[TestIndividual],
        generation: int = 0,
    ) -> None:
        """
        Initialize TestPopulation.

        Args:
            individuals: List of test individuals (each containing a test function snippet)
            generation: Current generation number
        """
        super().__init__(individuals, generation)
        logger.trace(f"Initialized TestPopulation with {self.size} individuals")

    def __repr__(self) -> str:
        return (
            f"<TestPopulation gen={self.generation} size={self.size} "
            f"avg_prob={self.compute_average_probability():.4f}>"
        )

    def _on_generation_advanced(self) -> None:
        """TestPopulation has no special actions on generation advance."""
