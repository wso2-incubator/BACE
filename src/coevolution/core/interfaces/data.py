# coevolution/core/interfaces/data.py
"""
Domain data structures and DTOs for the coevolution framework.
"""

import os
from dataclasses import dataclass, field
from typing import (
    Any,
    ItemsView,
    Iterator,
    KeysView,
    List,
    Literal,
    Optional,
    ValuesView,
)

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

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def keys(self) -> KeysView[str]:
        return self.results.keys()

    def values(self) -> ValuesView[dict[str, EvaluationResult]]:
        return self.results.values()

    def items(self) -> ItemsView[str, dict[str, EvaluationResult]]:
        return self.results.items()

    def get(self, code_id: str, default: Any = None) -> Any:
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
    rephrasings: list[str] | None = field(default=None, init=False)

    def with_rephrasings(self, rephrasings: list[str]) -> "Problem":
        """Return a shallow-copy of this Problem with `rephrasings` set.

        Keeps other fields identical. This helper is useful for attaching
        generated rephrasings without mutating callers' original instances.
        """
        # Since this dataclass is not frozen, we create a new instance to
        # preserve functional style and avoid unintended shared-state mutation.
        new = Problem(
            question_title=self.question_title,
            question_content=self.question_content,
            question_id=self.question_id,
            starter_code=self.starter_code,
            public_test_cases=self.public_test_cases,
            private_test_cases=self.private_test_cases,
        )
        new.rephrasings = rephrasings
        return new


@dataclass(frozen=True)
class SandboxConfig:
    """
    Immutable, serializable configuration for sandbox execution.
    Acts as the 'DNA' for creating worker sandboxes in multiprocessing.
    """

    # 1. Resource Constraints
    timeout: int = 30
    max_memory_mb: int = 256
    max_output_size: int = 10000

    # 2. Environment Settings
    language: str = "python"
    allowed_imports: Optional[List[str]] = None
    language_executable: Optional[str] = None
    test_method_timeout: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Validate configuration integrity immediately upon creation.
        We fail fast here so we don't fail later in a subprocess.
        """
        # Since the class is frozen, we access self.param directly.
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")

        if self.max_memory_mb <= 0:
            raise ValueError(f"Memory limit must be positive, got {self.max_memory_mb}")

        if self.language_executable and os.path.exists(self.language_executable):
            # Warning: This check might fail if the path exists on the worker
            # but not the host (e.g. Docker), but usually valid for local MP.
            # We'll skip strict existence check for flexibility, but could add it.
            pass

        if self.test_method_timeout is not None and self.test_method_timeout <= 0:
            raise ValueError(
                f"Test method timeout must be positive if set, got {self.test_method_timeout}"
            )


@dataclass
class BasicExecutionResult:
    """Result of basic code execution."""

    success: bool
    output: str
    error: str
    execution_time: float
    timeout: bool
    return_code: int
