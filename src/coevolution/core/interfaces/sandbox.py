# src/coevolution/core/interfaces/sandbox.py
"""
Protocol for language-specific sandbox execution operations.
"""

from typing import Protocol

from coevolution.core.interfaces.data import (
    BasicExecutionResult,
    SandboxConfig,
)


class ISandbox(Protocol):
    """
    Protocol defining the contract for language-specific sandbox execution.

    This allows the sandbox to remain language-agnostic by delegating
    language-specific execution tasks (running tests, parsing results)
    to concrete language implementations.
    """

    def __init__(self, config: SandboxConfig) -> None:
        """
        Initialize the sandbox adapter with configuration.

        Args:
            config: Sandbox configuration with timeout, memory limits, etc.
        """
        ...

    def execute_command(self, cmd: list[str], cwd: str | None = None) -> BasicExecutionResult:
        """
        Execute an arbitrary command safely in the sandbox.

        Args:
            cmd: Command arguments list (e.g., ["python3", "-c", "print(1)"])
            cwd: Optional working directory for the command

        Returns:
            BasicExecutionResult with success status, output, errors, and timing
        """
        ...
