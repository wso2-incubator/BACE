"""
Stateless utility for streaming evolutionary lifecycle events to structured telemetry.

The coevolution logging architecture uses `loguru` to stream events dynamically directly
to a compressed `.jsonl` file per problem, preventing in-memory history bloat on the
`BaseIndividual` entities themselves.

Events emitted by this class are tagged with `is_evolution_event=True` so they can be
routed cleanly to the `evolutionary_history.jsonl` sink.
"""

from typing import Any

from loguru import logger

from coevolution.core.interfaces.types import LifecycleEvent, Operation


class LifecycleEmitter:
    """
    100% Stateless static class for emitting standard evolutionary telemetry metrics.
    Do not add internal caching arrays (e.g. self.history) to prevent memory bleeds.
    """

    @staticmethod
    def _emit(generation: int, event: LifecycleEvent, **payload: Any) -> None:
        """
        Internal bound log emitter.
        """
        payload["generation"] = generation
        payload["event"] = event.value
        # Emits pure JSON objects to the specific sink hook
        logger.bind(is_evolution_event=True).info(
            "LIFECYCLE_EVENT", event_data=payload
        )

    @staticmethod
    def log_creation(
        generation: int,
        individual_id: str,
        snippet: str,
        operation: Operation,
        probability: float,
        parents: dict[str, list[str]] | None = None,
        test_type: str | None = None,
    ) -> None:
        """
        Log the birth of a new individual. Includes the raw `snippet`!
        Also automatically logs `BECAME_PARENT` for all recorded parents.
        """
        parents_dict = parents if parents is not None else {"code": [], "test": []}

        LifecycleEmitter._emit(
            generation=generation,
            event=LifecycleEvent.CREATED,
            individual_id=individual_id,
            snippet=snippet,
            creation_op=operation,
            probability=probability,
            parents=parents_dict,
            test_type=test_type,
        )

        for p_type, p_ids in parents_dict.items():
            for p_id in p_ids:
                LifecycleEmitter.log_parenting(
                    generation=generation,
                    parent_id=p_id,
                    offspring_id=individual_id,
                    operation=operation,
                )

    @staticmethod
    def log_parenting(
        generation: int,
        parent_id: str,
        offspring_id: str,
        operation: Operation,
    ) -> None:
        """
        Log when an individual is utilized as a parent to create an offspring.
        """
        LifecycleEmitter._emit(
            generation=generation,
            event=LifecycleEvent.BECAME_PARENT,
            individual_id=parent_id,
            offspring_id=offspring_id,
            operation=operation,
        )

    @staticmethod
    def log_survival(generation: int, individual_id: str) -> None:
        """Log when an individual successfully reaches the end of the evolutionary run."""
        LifecycleEmitter._emit(
            generation=generation,
            event=LifecycleEvent.SURVIVED,
            individual_id=individual_id,
        )

    @staticmethod
    def log_death(generation: int, individual_id: str) -> None:
        """Log when an individual is culled from the population."""
        LifecycleEmitter._emit(
            generation=generation,
            event=LifecycleEvent.DIED,
            individual_id=individual_id,
        )

    @staticmethod
    def log_elite_selection(generation: int, individual_id: str, test_type: str | None = None) -> None:
        """Log when an individual is selected to pass directly into the next generation."""
        LifecycleEmitter._emit(
            generation=generation,
            event=LifecycleEvent.SELECTED_AS_ELITE,
            individual_id=individual_id,
            test_type=test_type,
        )

    @staticmethod
    def log_probability_update(
        generation: int,
        individual_id: str,
        old_probability: float,
        new_probability: float,
        test_type: str | None = None,
    ) -> None:
        """
        Log Bayesian belief updates.
        """
        LifecycleEmitter._emit(
            generation=generation,
            event=LifecycleEvent.PROBABILITY_UPDATED,
            individual_id=individual_id,
            old_probability=old_probability,
            new_probability=new_probability,
            probability_change=new_probability - old_probability,
            test_type=test_type,
        )
