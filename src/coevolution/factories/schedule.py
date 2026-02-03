"""
Schedule builder with fluent API for constructing evolution schedules.

This module provides ScheduleBuilder, a fluent interface for defining
evolution schedules with multiple phases. Schedules control which populations
evolve in each generation, enabling patterns like:
- Warmup phases (one population frozen while other evolves)
- Simultaneous evolution (both evolve together)
- Alternating evolution (turn-taking between populations)

Example Usage:
    >>> from coevolution.factories import ScheduleBuilder
    >>>
    >>> # Build a complex schedule
    >>> schedule = (
    ...     ScheduleBuilder()
    ...     .warmup_code(duration=5)           # Code evolves alone first
    ...     .warmup_tests(duration=5)          # Then tests evolve alone
    ...     .simultaneous(duration=10)         # Then both together
    ...     .alternating(                       # Then alternating phases
    ...         total_duration=20,
    ...         code_step=2,
    ...         test_step=3,
    ...         start_with="test"
    ...     )
    ...     .build()
    ... )
"""

from typing import Any, Literal

from ..core.interfaces import EvolutionPhase, EvolutionSchedule


class ScheduleBuilder:
    """
    Fluent interface for constructing EvolutionSchedules.
    """

    def __init__(self) -> None:
        self._phases: list[EvolutionPhase] = []

    def add_phase(
        self, name: str, duration: int, evolve_code: bool, evolve_tests: bool
    ) -> "ScheduleBuilder":
        """Generic method to add any custom phase."""
        self._phases.append(EvolutionPhase(name, duration, evolve_code, evolve_tests))
        return self

    def warmup_code(self, duration: int) -> "ScheduleBuilder":
        """Phase: Only Code evolves (Tests frozen)."""
        return self.add_phase(
            "Code Warmup", duration, evolve_code=True, evolve_tests=False
        )

    def warmup_tests(self, duration: int) -> "ScheduleBuilder":
        """Phase: Only Tests evolve (Code frozen)."""
        return self.add_phase(
            "Test Warmup", duration, evolve_code=False, evolve_tests=True
        )

    def simultaneous(self, duration: int) -> "ScheduleBuilder":
        """Phase: Both evolve every generation."""
        return self.add_phase(
            "Simultaneous", duration, evolve_code=True, evolve_tests=True
        )

    def alternating(
        self,
        total_duration: int,
        code_step: int = 1,
        test_step: int = 1,
        start_with: Literal["code", "test"] = "test",
    ) -> "ScheduleBuilder":
        """
        Adds a sequence of phases alternating between Code and Test evolution.

        Args:
            total_duration: Total generations this block should last.
            code_step: How many generations Code evolves before switching.
            test_step: How many generations Tests evolve before switching.
            start_with: Who goes first.
        """
        remaining = total_duration
        is_code = start_with == "code"

        step_count = 1
        while remaining > 0:
            # Determine current step size (clamp to remaining duration)
            current_duration = min(remaining, code_step if is_code else test_step)

            name = f"Alt-{step_count} ({'Code' if is_code else 'Test'})"
            self.add_phase(
                name=name,
                duration=current_duration,
                evolve_code=is_code,
                evolve_tests=not is_code,
            )

            remaining -= current_duration
            is_code = not is_code  # Flip turn
            step_count += 1

        return self

    def build(self) -> EvolutionSchedule:
        if not self._phases:
            raise ValueError("Schedule must contain at least one phase.")
        return EvolutionSchedule(phases=list(self._phases))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EvolutionSchedule:
        """
        Create an EvolutionSchedule from a configuration dictionary.

        Args:
            config: Schedule configuration with optional warmup and alternating sections.

        Returns:
            Configured EvolutionSchedule.

        Example configs:
            # Code-only warmup
            {
                "warmup": {"duration": 5, "target": "code"}
            }

            # Alternating only
            {
                "alternating": {
                    "total_duration": 10,
                    "code_step": 1,
                    "test_step": 1,
                    "start_with": "test"
                }
            }

            # Warmup + Alternating
            {
                "warmup": {"duration": 5, "target": "code"},
                "alternating": {
                    "total_duration": 6,
                    "code_step": 1,
                    "test_step": 1,
                    "start_with": "test"
                }
            }
        """
        builder = cls()

        # Add warmup phase if specified
        if "warmup" in config:
            warmup_config = config["warmup"]
            duration = warmup_config.get("duration", 5)
            target = warmup_config.get("target", "code")  # Default to code warmup

            if target == "code":
                builder = builder.warmup_code(duration=duration)
            elif target == "tests":
                builder = builder.warmup_tests(duration=duration)
            else:
                raise ValueError(
                    f"Invalid warmup target: {target}. Must be 'code' or 'tests'."
                )

        # Add alternating phase if specified
        if "alternating" in config:
            alternating_config = config["alternating"]
            builder = builder.alternating(
                total_duration=alternating_config["total_duration"],
                code_step=alternating_config.get("code_step", 1),
                test_step=alternating_config.get("test_step", 1),
                start_with=alternating_config.get("start_with", "test"),
            )

        return builder.build()
