# coevolution/core/interfaces/data.py
"""
Domain data structures and DTOs for the coevolution framework.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .types import ExecutionResults, LifecycleEvent


@dataclass(frozen=True)
class TestResult:
    """
    Represents the result of a single unit test execution.
    """

    details: str | None
    status: Literal["passed", "failed", "error"]


@dataclass(frozen=True)
class ExecutionResult:
    """
    Represents the result of executing a unit test suite against a code individual.
    """

    script_error: bool
    test_results: dict[str, TestResult] = field(default_factory=dict)


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
