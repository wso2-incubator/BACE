"""Test executor for orchestrating code execution and analysis."""

from typing import Any, Optional

from loguru import logger

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.sandbox import ISandbox
from coevolution.core.interfaces.language import ILanguage

from infrastructure.languages.python import PythonLanguage
from .adapters.generic import SubprocessSandbox
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
        language_adapter: Optional[ILanguage] = None,
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
            self.sandbox = SubprocessSandbox(config or SandboxConfig())

        self.language_adapter = language_adapter or PythonLanguage()

        logger.debug(
            f"Initialized TestExecutor with sandbox: {type(self.sandbox).__name__}"
        )

    def execute_test_script(self, test_script: str) -> EvaluationResult:
        """Execute a test script and return a single test result."""
        import os
        import tempfile
        
        logger.debug(f"TestExecutor: executing test script (len={len(test_script)})")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_ext = ".bal" if getattr(self.language_adapter, "language", "") == "ballerina" else ".py"
            script_path = os.path.join(tmpdir, f"test_script{file_ext}")
            xml_path = os.path.join(tmpdir, "results.xml")

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(test_script)

            kwargs = {}
            if hasattr(self.sandbox, "config") and hasattr(self.sandbox.config, "test_method_timeout"):
                kwargs["timeout"] = self.sandbox.config.test_method_timeout

            cmd = self.language_adapter.runtime.get_test_command(script_path, xml_path, **kwargs)
            raw_result = self.sandbox.execute_command(cmd, cwd=tmpdir)

            xml_content = None
            if os.path.exists(xml_path):
                with open(xml_path, "r", encoding="utf-8") as f:
                    xml_content = f.read()

            result: EvaluationResult = self.language_adapter.analyzer.analyze(raw_result, xml_content=xml_content)
        
        return result

    def execute_code(self, code: str) -> BasicExecutionResult:
        """Execute arbitrary code safely."""
        import os
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_ext = ".bal" if getattr(self.language_adapter, "language", "") == "ballerina" else ".py"
            script_path = os.path.join(tmpdir, f"eval_script{file_ext}")
            
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            if hasattr(self.language_adapter.runtime, "get_execution_command"):
                cmd = self.language_adapter.runtime.get_execution_command(script_path)
            else:
                cmd = ["python", script_path]
                
            return self.sandbox.execute_command(cmd, cwd=tmpdir)
