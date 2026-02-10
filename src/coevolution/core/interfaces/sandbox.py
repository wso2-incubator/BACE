# src/coevolution/core/interfaces/sandbox.py
"""
Protocol for language-specific sandbox execution operations.
"""

from typing import Protocol

from coevolution.core.interfaces.data import (
    BasicExecutionResult,
    EvaluationResult,
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

    def execute_code(self, code: str) -> BasicExecutionResult:
        """
        Execute arbitrary code safely and return basic execution result.

        Args:
            code: Source code to execute

        Returns:
            BasicExecutionResult with success status, output, errors, and timing
        """
        ...

    def execute_test_script(self, test_script: str) -> EvaluationResult:
        """
        Execute a test script and return detailed test analysis.

        Args:
            test_script: Complete test script with tests and implementation

        Returns:
            EvaluationResult with detailed test results, pass/fail counts, and analysis
        """
        ...
