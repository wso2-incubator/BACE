"""Type definitions for sandbox operations."""

import os
from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass(frozen=True)
class SandboxConfig:
    """
    Immutable, serializable configuration for SafeCodeSandbox.
    Acts as the 'DNA' for creating worker sandboxes in multiprocessing.
    """

    # 1. Resource Constraints
    timeout: int = 30
    max_memory_mb: int = 100
    max_output_size: int = 10000

    # 2. Environment Settings
    allowed_imports: Optional[List[str]] = None
    python_executable: Optional[str] = None
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

        if self.python_executable and not os.path.exists(self.python_executable):
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
