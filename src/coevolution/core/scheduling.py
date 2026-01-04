from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EvolutionPhase:
    """
    Defines a specific time block in the evolutionary run with fixed rules.
    """

    name: str
    duration: int
    evolve_code: bool
    evolve_tests: bool

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"Phase '{self.name}' duration must be positive.")


@dataclass(frozen=True)
class EvolutionSchedule:
    """
    The immutable plan for the entire evolutionary run.
    """

    phases: list[EvolutionPhase]

    @property
    def total_generations(self) -> int:
        """Derived property: Sum of all phase durations."""
        return sum(p.duration for p in self.phases)


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
