"""Ballerina implementation of the ISandbox protocol."""

import subprocess
import tempfile
import time
from pathlib import Path

from loguru import logger

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.sandbox import ISandbox
from infrastructure.sandbox.ballerina_analyzer import (
    BallerinaTestAnalyzer,
    truncate_preserve_tail,
)
from infrastructure.sandbox.memory import MemoryMonitor
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
                proc = subprocess.Popen(
                    [self.bal_executable, "run", "--offline"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=tmpdir,
                )

                monitor = MemoryMonitor(proc, self.config.max_memory_mb)
                monitor.start()

                try:
                    stdout_data, stderr_data = proc.communicate(timeout=self.config.timeout)
                finally:
                    monitor.stop()

                execution_time = time.time() - start_time

                if monitor.exceeded:
                    return BasicExecutionResult(
                        success=False,
                        output="",
                        error=f"Ballerina execution exceeded memory limit of {self.config.max_memory_mb} MB",
                        execution_time=execution_time,
                        timeout=False,
                        return_code=-1,
                    )

                stdout = (
                    stdout_data[: self.config.max_output_size]
                    if stdout_data
                    else ""
                )
                stderr = (
                    stderr_data[: self.config.max_output_size]
                    if stderr_data
                    else ""
                )

                return BasicExecutionResult(
                    success=proc.returncode == 0,
                    output=stdout,
                    error=stderr,
                    execution_time=execution_time,
                    timeout=False,
                    return_code=proc.returncode,
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

            # Ballerina tests need to be in the 'tests' directory
            tests_dir = tmpdir_path / "tests"
            tests_dir.mkdir(exist_ok=True)
            (tests_dir / "main_test.bal").write_text(test_script)

            try:
                start_time = time.time()
                # Run bal test with JUnit reporter
                proc = subprocess.Popen(
                    [self.bal_executable, "test", "--offline"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=tmpdir,
                )
                
                monitor = MemoryMonitor(proc, self.config.max_memory_mb)
                monitor.start()

                try:
                    stdout_data, stderr_data = proc.communicate(timeout=self.config.timeout)
                finally:
                    monitor.stop()

                execution_time = time.time() - start_time

                if monitor.exceeded:
                    basic_result = BasicExecutionResult(
                        success=False,
                        output="",
                        error=f"Ballerina test exceeded memory limit of {self.config.max_memory_mb} MB",
                        execution_time=execution_time,
                        timeout=False,
                        return_code=-1,
                    )
                    analyzer = BallerinaTestAnalyzer()
                    return analyzer.analyze_test_output(basic_result)

                # Truncate stdout/stderr to avoid propagating huge raw logs
                logger.debug(
                    "bal test produced stdout_len=%d stderr_len=%d",
                    len(stdout_data or ""),
                    len(stderr_data or ""),
                )

                stdout = truncate_preserve_tail(
                    stdout_data, self.config.max_output_size
                )
                stderr = truncate_preserve_tail(
                    stderr_data, self.config.max_output_size
                )

                basic_result = BasicExecutionResult(
                    success=proc.returncode == 0,
                    output=stdout,
                    error=stderr,
                    execution_time=execution_time,
                    timeout=False,
                    return_code=proc.returncode,
                )

                # Use Ballerina-specific analyzer
                analyzer = BallerinaTestAnalyzer()
                return analyzer.analyze_test_output(basic_result)

            except subprocess.TimeoutExpired:
                basic_result = BasicExecutionResult(
                    success=False,
                    output="",
                    error=f"Ballerina test execution timed out after {self.config.timeout} seconds",
                    execution_time=self.config.timeout,
                    timeout=True,
                    return_code=-1,
                )
                analyzer = BallerinaTestAnalyzer()
                return analyzer.analyze_test_output(basic_result)

    def _setup_project(self, project_path: Path) -> None:
        """Setup a minimal Ballerina project structure."""
        toml_content = """[package]
org = "test"
name = "sandbox"
version = "0.1.0"
"""
        (project_path / "Ballerina.toml").write_text(toml_content)
