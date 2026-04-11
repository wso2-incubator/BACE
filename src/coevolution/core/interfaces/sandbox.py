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
from coevolution.core.interfaces.language import ILanguageRuntime, ITestAnalyzer


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

    def execute_command(
        self, cmd: list[str], cwd: str | None = None
    ) -> BasicExecutionResult:
        """
        Execute an arbitrary command safely in the sandbox.

        Args:
            cmd: Command arguments list (e.g., ["python3", "-c", "print(1)"])
            cwd: Optional working directory for the command

        Returns:
            BasicExecutionResult with success status, output, errors, and timing
        """
        ...

    def execute_test_script(
        self, test_script: str, runtime: ILanguageRuntime, analyzer: ITestAnalyzer
    ) -> EvaluationResult:
        """
        Execute a complete test script and return analyzed results.

        Args:
            test_script: The complete source code of the test script.
            runtime: Language-specific runtime to get commands.
            analyzer: Language-specific analyzer to parse output.

        Returns:
            EvaluationResult: Structured test execution result.
        """
        ...

    def execute_code(
        self, code: str, runtime: ILanguageRuntime
    ) -> BasicExecutionResult:
        """
        Execute an arbitrary code snippet and return the raw output.

        Args:
            code: The source code to execute.
            runtime: Language-specific runtime to get commands.

        Returns:
            BasicExecutionResult: Raw execution result.
        """
        ...
