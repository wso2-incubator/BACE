"""
Universal Breeder — a pure weighted router.

Replaces all population-specific BreedingStrategy subclasses.
Operators are self-sufficient: each operator.execute(context) handles
parent selection, LLM call, probability assignment, and individual construction.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import TYPE_CHECKING

from loguru import logger

from coevolution.core.interfaces.base import BaseIndividual

if TYPE_CHECKING:
    from coevolution.core.interfaces.context import CoevolutionContext
    from coevolution.core.interfaces.operators import IOperator


# RegisteredOperator is now imported from coevolution.core.interfaces.operators
from coevolution.core.interfaces.operators import RegisteredOperator


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

    def add_operators(self, new_operators: list[RegisteredOperator[T]]) -> None:
        """
        Dynamically add new operators to the breeder's pool.
        Forces a re-computation of weights and operator lists.
        """
        if not new_operators:
            return

        self._operators.extend(new_operators)
        self._op_weights = [ro.weight for ro in self._operators]
        self._op_list = [ro.operator for ro in self._operators]
        logger.debug(f"Breeder: added {len(new_operators)} new operators. Total: {len(self._operators)}")

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
                op = None
                try:
                    op = self._sample_operator()
                    results = op.execute(context)
                    if results:
                        offspring.extend(results)
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                except Exception as e:
                    op_name = op.operation_name() if op else "Unknown"
                    logger.warning(f"Operator '{op_name}' failed: {e}")
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
                    # Do not call executor.shutdown() here — the `with` block's
                    # __exit__ will call shutdown(wait=True), which is the correct
                    # single teardown path. Calling shutdown() twice with different
                    # `wait` flags races in Python ≤3.8.
                    break

                needed = num_offsprings - len(offspring)
                batch_size = min(needed, self.llm_workers * 2)

                futures_to_op = {}
                for _ in range(batch_size):
                    op = self._sample_operator()
                    futures_to_op[executor.submit(op.execute, context)] = op

                batch_produced = False
                for future in as_completed(futures_to_op):
                    op = futures_to_op[future]
                    try:
                        results = future.result()
                        if results:
                            offspring.extend(results)
                            batch_produced = True
                            if len(offspring) >= num_offsprings:
                                for f in futures_to_op:
                                    f.cancel()
                                break
                    except Exception as e:
                        logger.warning(f"Operator '{op.operation_name()}' failed: {e}")

                if batch_produced:
                    consecutive_failures = 0
                else:
                    consecutive_failures += batch_size

        return offspring[:num_offsprings]


__all__ = ["Breeder", "RegisteredOperator"]
