# coevolution/core/interfaces/data.py
"""
Domain data structures and DTOs for the coevolution framework.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .types import LifecycleEvent


@dataclass(frozen=True)
class EvaluationResult:
    """
    Represents the result of a single unit test execution.
    """

    status: Literal["passed", "failed", "error"]
    error_log: str | None = None
    execution_time: float = 0.0


@dataclass(frozen=True)
class ExecutionResults:
    """
    Represents the collected results of executing a code population against tests.

    Maps code_id -> {test_id -> EvaluationResult}
    """

    results: dict[str, dict[str, EvaluationResult]] = field(default_factory=dict)

    def __getitem__(self, code_id: str) -> dict[str, EvaluationResult]:
        return self.results[code_id]

    def __iter__(self):
        return iter(self.results)

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def items(self):
        return self.results.items()

    def get(self, code_id: str, default=None):
        return self.results.get(code_id, default)


@dataclass(frozen=True)
class InteractionData:
    """
    Captures the interaction between code population and a specific test population.

    This encapsulates execution results and observation matrix for a single
    code-test population pair, making the data relationship explicit.
    """

    execution_results: ExecutionResults
    observation_matrix: np.ndarray


@dataclass(frozen=True)
class LogEntry:
    """
    A structured log entry for an individual's lifecycle event.

    Attributes:
        generation: The generation number when this event occurred.
        event: The type of lifecycle event.
        details: Additional event-specific information.
    """

    generation: int
    event: LifecycleEvent
    details: dict[str, Any]


@dataclass
class Test:
    input: str
    output: str


@dataclass
class Problem:
    question_title: str
    question_content: str
    question_id: str
    starter_code: str
    public_test_cases: list[Test]
    private_test_cases: list[Test]
