"""
Universal Breeder — a pure weighted router.

Replaces all population-specific BreedingStrategy subclasses.
Operators are self-sufficient: each operator.execute(context) handles
parent selection, LLM call, probability assignment, and individual construction.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from coevolution.core.interfaces.base import BaseIndividual

if TYPE_CHECKING:
    from coevolution.core.interfaces.context import CoevolutionContext
    from coevolution.core.interfaces.operators import IOperator


@dataclass
class RegisteredOperator[T: BaseIndividual]:
    """Pairs an operator with its sampling weight."""

    weight: float
    operator: "IOperator[T]"


class Breeder[T: BaseIndividual]:
    """
    Pure weighted router over a set of RegisteredOperators.

    Responsibilities:
    - Sample an operator from the registered set by weight.
    - Call operator.execute(context) in a thread pool.
    - Collect results until num_offsprings is reached.
    - Apply a circuit breaker on consecutive failures.

    Does NOT:
    - Know about parent selection.
    - Know about probability assignment or individual factories.
    - Know about LLM calls or DTOs.
    All of that lives inside each operator.
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        registered_operators: list[RegisteredOperator[T]],
        llm_workers: int = 1,
    ) -> None:
        if not registered_operators:
            raise ValueError("Breeder must have at least one registered operator.")

        self._operators = registered_operators
        self.llm_workers = llm_workers

        # Pre-compute for random.choices
        self._op_weights = [ro.weight for ro in registered_operators]
        self._op_list = [ro.operator for ro in registered_operators]

    def _sample_operator(self) -> "IOperator[T]":
        import random

        return random.choices(self._op_list, weights=self._op_weights, k=1)[0]

    def breed(self, context: "CoevolutionContext", num_offsprings: int) -> list[T]:
        """
        Generate offspring by concurrently calling operator.execute(context).

        Args:
            context: Full coevolution context passed to each operator.
            num_offsprings: Exact number of offspring to produce.

        Returns:
            list[T] of length <= num_offsprings.
        """
        if num_offsprings == 0:
            return []

        offspring: list[T] = []

        # --- Single-threaded fast path ---
        if self.llm_workers <= 1:
            consecutive_failures = 0
            while len(offspring) < num_offsprings:
                if consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        f"Breeder aborting: {consecutive_failures} consecutive failures. "
                        "Check operator implementations."
                    )
                    break
                try:
                    results = self._sample_operator().execute(context)
                    if results:
                        offspring.extend(results)
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                except Exception as e:
                    logger.warning(f"Operator failed: {e}")
                    consecutive_failures += 1
            return offspring[:num_offsprings]

        # --- Parallel path ---
        consecutive_failures = 0

        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            while len(offspring) < num_offsprings:
                if consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                    logger.critical(
                        f"Breeder aborting: {consecutive_failures} consecutive failures."
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                needed = num_offsprings - len(offspring)
                batch_size = min(needed, self.llm_workers * 2)

                futures = [
                    executor.submit(self._sample_operator().execute, context)
                    for _ in range(batch_size)
                ]

                batch_produced = False
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        if results:
                            offspring.extend(results)
                            batch_produced = True
                            if len(offspring) >= num_offsprings:
                                for f in futures:
                                    f.cancel()
                                break
                    except Exception as e:
                        logger.warning(f"Operator failed: {e}")

                if batch_produced:
                    consecutive_failures = 0
                else:
                    consecutive_failures += batch_size

        return offspring[:num_offsprings]


__all__ = ["Breeder", "RegisteredOperator"]
