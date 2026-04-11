"""DifferentialInitializer — differential population always starts empty."""

from __future__ import annotations

from loguru import logger

from coevolution.core.interfaces import Problem
from coevolution.core.individual import TestIndividual

from coevolution.strategies.llm_base import BaseLLMInitializer


class DifferentialInitializer(BaseLLMInitializer[TestIndividual]):
    """Differential tests start empty — Gen 0 is always []."""

    def initialize(self, problem: Problem) -> list[TestIndividual]:
        logger.debug("DifferentialInitializer: starting with empty population")
        return []


__all__ = ["DifferentialInitializer"]
