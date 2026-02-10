"""Ballerina implementation of the ISandbox protocol."""

import subprocess
import tempfile
import time
from pathlib import Path

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.sandbox import ISandbox
from infrastructure.sandbox.analyzer import PytestXmlAnalyzer
from infrastructure.sandbox.types import BasicExecutionResult, SandboxConfig


class BallerinaSandbox(ISandbox):
    """
    A sandbox environment for executing Ballerina code.
    """

    @classmethod
    def from_config(cls, config: SandboxConfig) -> "BallerinaSandbox":
        """
        Create a sandbox instance from a configuration object.
        """
        return cls(config=config)

    def __init__(self, config: SandboxConfig) -> None:
        """
        Initialize the Ballerina sandbox.

        Args:
            config: Sandbox configuration parameters
        """
        self.config = config
        self.bal_executable = config.language_executable or "bal"

    def execute_code(
        self, code: str, capture_output: bool = True
    ) -> BasicExecutionResult:
        """Execute Ballerina code using 'bal run'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create Ballerina project structure
            self._setup_project(tmpdir_path)

            # Write code to main.bal
            (tmpdir_path / "main.bal").write_text(code)

            try:
                start_time = time.time()
                result = subprocess.run(
                    [self.bal_executable, "run", "--offline"],
                    capture_output=capture_output,
                    text=True,
                    timeout=self.config.timeout,
                    cwd=tmpdir,
                )
                execution_time = time.time() - start_time

                stdout = (
                    result.stdout[: self.config.max_output_size]
                    if result.stdout
                    else ""
                )
                stderr = (
                    result.stderr[: self.config.max_output_size]
                    if result.stderr
                    else ""
                )

                return BasicExecutionResult(
                    success=result.returncode == 0,
                    output=stdout,
                    error=stderr,
                    execution_time=execution_time,
                    timeout=False,
                    return_code=result.returncode,
                )

            except subprocess.TimeoutExpired:
                return BasicExecutionResult(
                    success=False,
                    output="",
                    error=f"Ballerina execution timed out after {self.config.timeout} seconds",
                    execution_time=self.config.timeout,
                    timeout=True,
                    return_code=-1,
                )
            except Exception as e:
                return BasicExecutionResult(
                    success=False,
                    output="",
                    error=f"Ballerina execution error: {str(e)}",
                    execution_time=0,
                    timeout=False,
                    return_code=-1,
                )

    def execute_test_script(self, test_script: str) -> EvaluationResult:
        """Execute Ballerina tests using 'bal test'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create Ballerina project structure
            self._setup_project(tmpdir_path)

            # Ballerina tests usually go in a 'tests' directory or same dir
            # For simplicity, we'll put the test script in the project root as it contains both code and tests
            (tmpdir_path / "main.bal").write_text(test_script)

            try:
                start_time = time.time()
                # Run bal test with JUnit reporter
                result = subprocess.run(
                    [self.bal_executable, "test", "--offline"],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout,
                    cwd=tmpdir,
                )
                execution_time = time.time() - start_time

                # Ballerina test results might be printed to stdout/stderr
                # We also need to check if it generates a JUnit XML file
                # By default 'bal test' might not generate JUnit XML unless configured
                # For now, we'll analyze the output as a basic result

                basic_result = BasicExecutionResult(
                    success=result.returncode == 0,
                    output=result.stdout,
                    error=result.stderr,
                    execution_time=execution_time,
                    timeout=False,
                    return_code=result.returncode,
                )

                # TODO: Implement Ballerina-specific XML analysis if needed
                # For now, use the fallback in PytestXmlAnalyzer which works on basic_result
                analyzer = PytestXmlAnalyzer()
                return analyzer.analyze_pytest_xml(None, basic_result)

            except subprocess.TimeoutExpired:
                basic_result = BasicExecutionResult(
                    success=False,
                    output="",
                    error=f"Ballerina test execution timed out after {self.config.timeout} seconds",
                    execution_time=self.config.timeout,
                    timeout=True,
                    return_code=-1,
                )
                analyzer = PytestXmlAnalyzer()
                return analyzer.analyze_pytest_xml(None, basic_result)

    def _setup_project(self, project_path: Path) -> None:
        """Setup a minimal Ballerina project structure."""
        toml_content = """[package]
org = "test"
name = "sandbox"
version = "0.1.0"
"""
        (project_path / "Ballerina.toml").write_text(toml_content)
