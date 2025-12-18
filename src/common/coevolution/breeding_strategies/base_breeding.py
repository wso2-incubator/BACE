from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..core.interfaces import (
    BaseIndividual,
    CoevolutionContext,
    IBreedingStrategy,
    OperatorRatesConfig,
)


class BaseBreedingStrategy[T: BaseIndividual](IBreedingStrategy[T], ABC):
    """
    Abstract base class for breeding strategies.
    """

    @abstractmethod
    def __init__(
        self,
        op_rates_config: OperatorRatesConfig,
    ) -> None:
        self.op_rates_config = op_rates_config
        self._strategies: dict[str, Callable[[CoevolutionContext], list[T]]] = {}

    def _operator_selector(self) -> str:
        """Select an operator based on configured rates."""
        total = sum(self.op_rates_config.operation_rates.values())
        normalized_probs: list[float] = [
            rate / total for rate in self.op_rates_config.operation_rates.values()
        ]
        selected_op: str = np.random.choice(
            list(self.op_rates_config.operation_rates.keys()), p=normalized_probs
        )
        return selected_op

    def breed(
        self, coevolution_context: CoevolutionContext, num_offsprings: int
    ) -> list[T]:
        """
        Breed new individuals based on the configured strategies and rates.

        Args:
            coevolution_context: The current coevolution context.
            num_offsprings: The number of offspring individuals to produce.
        Returns:
            A list of newly bred individuals.
        """
        offspring_list: list[T] = []

        while len(offspring_list) < num_offsprings:
            # 1. Select Operation
            op_name = self._operator_selector()
            handler = self._strategies.get(op_name)

            if not handler:
                continue

            # 2. Delegate everything to the handler
            new_inds = handler(coevolution_context)

            # 3. Collect results
            offspring_list.extend(new_inds)

        return offspring_list[:num_offsprings]


__all__ = ["BaseBreedingStrategy"]
