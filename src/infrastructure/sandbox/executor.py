"""Test executor for orchestrating code execution and analysis."""

from typing import Any, Optional

from loguru import logger

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.sandbox import ISandbox

from .adapters.python import PythonSandbox
from .types import BasicExecutionResult, SandboxConfig


class TestExecutor:
    """
    High-level interface for executing a single test function in a sandbox.

    This class orchestrates safe code execution and test result analysis,
    providing a clean separation between execution and analysis concerns.
    """

    __test__ = False

    def __init__(
        self,
        sandbox_adapter: Optional[ISandbox] = None,
        config: Optional[SandboxConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize test executor with a sandbox adapter.

        Args:
            sandbox_adapter: Specialized sandbox adapter for a language
            config: Sandbox configuration
            **kwargs: Backward compatibility parameters (timeout, etc.)
        """
        if sandbox_adapter:
            self.sandbox = sandbox_adapter
        else:
            # Fallback to default Python sandbox
            if not config and kwargs:
                # Map old kwargs to SandboxConfig
                config = SandboxConfig(
                    timeout=kwargs.get("timeout", 30),
                    max_memory_mb=kwargs.get("max_memory_mb", 100),
                    max_output_size=kwargs.get("max_output_size", 10000),
                    allowed_imports=kwargs.get("allowed_imports"),
                    language_executable=kwargs.get("python_executable"),
                    test_method_timeout=kwargs.get("test_method_timeout"),
                )
            self.sandbox = PythonSandbox(config or SandboxConfig())

        logger.debug(
            f"Initialized TestExecutor with sandbox: {type(self.sandbox).__name__}"
        )

    def execute_test_script(self, test_script: str) -> EvaluationResult:
        """
        Execute a test script and return a single test result.

        Args:
            test_script: The test script to execute

        Returns:
            EvaluationResult with detailed analysis
        """
        logger.debug(f"TestExecutor: executing test script (len={len(test_script)})")

        # Delegate to sandbox
        return self.sandbox.execute_test_script(test_script)

    def execute_code(self, code: str) -> BasicExecutionResult:
        """
        Execute arbitrary code safely (delegates to sandbox).

        Args:
            code: Source code to execute

        Returns:
            BasicExecutionResult
        """
        return self.sandbox.execute_code(code)
