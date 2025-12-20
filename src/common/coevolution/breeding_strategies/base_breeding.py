import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from loguru import logger

from ..core.interfaces import (
    BaseIndividual,
    CoevolutionContext,
    IBreedingStrategy,
    OperatorRatesConfig,
)


class BaseBreedingStrategy[T: BaseIndividual](IBreedingStrategy[T], ABC):
    """
    Abstract base class for breeding strategies with robust parallel execution.
    Includes circuit breakers and smart batching to prevent resource exhaustion.
    """

    @abstractmethod
    def __init__(
        self,
        op_rates_config: OperatorRatesConfig,
        max_workers: int = 1,
    ) -> None:
        self.op_rates_config = op_rates_config
        self.max_workers = max_workers
        self._strategies: dict[str, Callable[[CoevolutionContext], list[T]]] = {}

    def _operator_selector(self) -> str:
        """
        Select an operator based on configured rates.
        Uses random.choices for better thread safety than numpy in concurrent contexts.
        """
        ops = list(self.op_rates_config.operation_rates.keys())
        weights = list(self.op_rates_config.operation_rates.values())

        # k=1 returns a list of 1 element
        return random.choices(ops, weights=weights, k=1)[0]

    def _attempt_breeding(self, context: CoevolutionContext) -> list[T]:
        """
        Thread-safe wrapper for a single breeding attempt.
        """
        try:
            op_name = self._operator_selector()
            handler = self._strategies.get(op_name)

            if not handler:
                logger.warning(f"No handler found for operation: {op_name}")
                return []

            return handler(context)

        except Exception as e:
            logger.error(f"Breeding attempt failed in worker: {e}")
            return []

    def breed(
        self, coevolution_context: CoevolutionContext, num_offsprings: int
    ) -> list[T]:
        """
        Breed new individuals in parallel with safeguards.
        """
        offspring_list: list[T] = []

        # Optimization: Bypass overhead if single-threaded
        if self.max_workers <= 1:
            failures = 0
            max_failures = 5  # TODO: make configurable

            while len(offspring_list) < num_offsprings:
                new_inds = self._attempt_breeding(coevolution_context)
                if not new_inds:
                    failures += 1
                    if failures > max_failures:
                        logger.error(
                            f"Breeding aborted: exceeded {max_failures} consecutive failures."
                        )
                        break
                else:
                    offspring_list.extend(new_inds)

            return offspring_list[:num_offsprings]

        # --- Parallel Execution Logic ---

        # Circuit Breaker State
        consecutive_failures = 0
        max_consecutive_failures = 5  # TODO: make configurable

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while len(offspring_list) < num_offsprings:
                # 1. Circuit Breaker Check
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        f"ABORTING BREEDING: {consecutive_failures} consecutive failed attempts. "
                        "Check your operators or prompt configuration."
                    )
                    # Cancel any remaining tasks in the queue (best effort)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                # 2. Smart Batching (The Fix for Over-Generation)
                needed = num_offsprings - len(offspring_list)

                # Don't submit 100 tasks if we only have 5 workers.
                # Submit enough to fill the pool + a buffer.
                # This allows us to stop early if the first batch yields high returns.
                batch_cap = self.max_workers * 2
                batch_size = min(needed, batch_cap)

                futures = [
                    executor.submit(self._attempt_breeding, coevolution_context)
                    for _ in range(batch_size)
                ]

                # 3. Collect Results
                batch_success = False

                for future in as_completed(futures):
                    try:
                        new_inds = future.result()
                        if new_inds:
                            offspring_list.extend(new_inds)
                            batch_success = True

                            # Early Exit: If this specific result pushed us over the edge
                            if len(offspring_list) >= num_offsprings:
                                for f in futures:
                                    f.cancel()
                                break
                    except Exception as e:
                        logger.error(f"Future execution error: {e}")

                # 4. Update Circuit Breaker
                if batch_success:
                    consecutive_failures = 0  # Reset on any success
                else:
                    # If the entire batch failed (e.g. 10 threads all returned []), increment count
                    consecutive_failures += batch_size

        return offspring_list[:num_offsprings]
