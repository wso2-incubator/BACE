# src/common/coevolution/core/population.py

import numpy as np
from loguru import logger

from .individual import CodeIndividual, TestIndividual
from .interfaces import BasePopulation, IParetoFrontCalculator, ITestBlockBuilder


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
        pass


class TestPopulation(BasePopulation[TestIndividual]):
    """
    Concrete population class for TestIndividuals.
    """

    def __init__(
        self,
        individuals: list[TestIndividual],
        # injected dependencies
        pareto_fn: IParetoFrontCalculator,
        rebuild_test_block_fn: ITestBlockBuilder,
        # default values
        test_class_block: str = "",
        generation: int = 0,
    ) -> None:
        if not test_class_block or not test_class_block.strip():
            raise ValueError("test_class_block is required for TestPopulation")

        super().__init__(individuals, generation)
        self._test_class_block = test_class_block
        self._set_default_discriminations()

        self._pareto_fn = pareto_fn
        self._rebuild_test_block_fn = rebuild_test_block_fn

    def _set_default_discriminations(self) -> None:
        """Internal helper to initialize or reset discrimination scores."""
        for ind in self._individuals:
            ind.discrimination = None

    def _build_test_class_block(self) -> None:
        """Implementation of rebuilding the test class block."""

        logger.debug("Rebuilding test class block")
        self._test_class_block = self._rebuild_test_block_fn(
            self._test_class_block,
            [ind.snippet for ind in self._individuals],
        )

    def _on_generation_advanced(self) -> None:
        """
        Hook called by set_next_generation.
        Rebuilds test class block and resets discrimination scores.
        """
        logger.debug(f"Rebuilding test class block for gen {self.generation}...")
        self._build_test_class_block()
        self._set_default_discriminations()

    @property
    def test_class_block(self) -> str:
        """Get the full unittest class block."""
        return self._test_class_block

    @property
    def discriminations(self) -> np.ndarray:
        """Returns a list of discrimination scores from the individuals."""
        return np.array(
            [
                ind._discrimination if ind._discrimination is not None else np.nan
                for ind in self._individuals
            ],
            dtype=float,
        )

    def set_discriminations(self, new_discriminations: np.ndarray) -> None:
        """Sets the discrimination score for each TestIndividual."""
        if len(new_discriminations) != self.size:
            raise ValueError(
                "Length of new_discriminations must match population size."
            )

        for ind, disc in zip(self._individuals, new_discriminations):
            ind.discrimination = float(disc) if np.isfinite(disc) else None

    def get_pareto_front(self) -> list[TestIndividual]:
        """
        Computes and returns the Pareto front based on
        probability and discrimination (maximize both).
        """
        logger.debug("Computing Pareto front for TestPopulation...")
        indices: list[int] = self._pareto_fn(self.probabilities, self.discriminations)
        return [self._individuals[i] for i in indices]

    def __repr__(self) -> str:
        return (
            f"<TestPopulation gen={self.generation} size={self.size} "
            f"avg_prob={self.compute_average_probability():.4f}>"
        )
