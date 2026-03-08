"""Python implementation of the ISandbox protocol."""

import os
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

from loguru import logger

from coevolution.core.interfaces.data import EvaluationResult
from coevolution.core.interfaces.sandbox import ISandbox
from infrastructure.sandbox.analyzer import PytestXmlAnalyzer
from infrastructure.sandbox.memory import MemoryMonitor
from infrastructure.sandbox.types import BasicExecutionResult, SandboxConfig


class PythonSandbox(ISandbox):
    """
    A safe sandbox environment for executing Python code with restrictions.
    """

    @classmethod
    def from_config(cls, config: SandboxConfig) -> "PythonSandbox":
        """
        Create a sandbox instance from a configuration object.
        Backward compatibility for legacy SafeCodeSandbox.from_config.
        """
        return cls(config=config)

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        timeout: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
        max_output_size: Optional[int] = None,
        allowed_imports: Optional[List[str]] = None,
        python_executable: Optional[str] = None,
        test_method_timeout: Optional[int] = None,
    ):
        """
        Initialize the sandbox with safety parameters.

        Args:
            config: Sandbox configuration parameters
            timeout: Backward compatibility timeout
            max_memory_mb: Backward compatibility max memory
            max_output_size: Backward compatibility max output
            allowed_imports: Backward compatibility allowed imports
            python_executable: Backward compatibility python executable
            test_method_timeout: Backward compatibility test method timeout
        """
        if config is None:
            # Create config from parameters or defaults
            config = SandboxConfig(
                timeout=timeout if timeout is not None else 30,
                max_memory_mb=max_memory_mb if max_memory_mb is not None else 100,
                max_output_size=max_output_size
                if max_output_size is not None
                else 10000,
                allowed_imports=allowed_imports,
                language_executable=python_executable,
                test_method_timeout=test_method_timeout,
            )

        self.config = config
        # Use language_executable as python_executable
        self.python_executable = config.language_executable or sys.executable

        # Default allowed imports if not specified
        self.allowed_imports = config.allowed_imports or [
            "math",
            "random",
            "itertools",
            "collections",
            "functools",
            "operator",
            "copy",
            "json",
            "re",
            "string",
            "unittest",
            "pytest",
            "pytest_timeout",
            "typing",
            "dataclasses",
            "enum",
            "heapq",
            "bisect",
        ]

        # TODO: Re-evaluate blocked patterns
        self.blocked_patterns: list[str] = []

    @property
    def timeout(self) -> int:
        """Backward compatibility for timeout attribute."""
        return self.config.timeout

    def _check_code_safety(self, code: str) -> bool:
        """Check if code contains potentially dangerous patterns."""
        code_lower = code.lower()
        for pattern in self.blocked_patterns:
            if pattern.lower() in code_lower:
                logger.debug(f"Code blocked by pattern '{pattern}'")
                return False
        return True

    def execute_code(
        self, code: str, capture_output: bool = True
    ) -> BasicExecutionResult:
        """Safely execute Python code in a restricted environment."""
        if not self._check_code_safety(code):
            return BasicExecutionResult(
                success=False,
                output="",
                error="Code contains potentially dangerous patterns",
                execution_time=0,
                timeout=False,
                return_code=-1,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)

            try:
                start_time = time.time()
                proc = subprocess.Popen(
                    [self.python_executable, script_path],
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
                        error=f"Memory limit exceeded ({self.config.max_memory_mb} MB)",
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
                    error=f"Code execution timed out after {self.config.timeout} seconds",
                    execution_time=self.config.timeout,
                    timeout=True,
                    return_code=-1,
                )
            except Exception as e:
                return BasicExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution error: {str(e)}",
                    execution_time=0,
                    timeout=False,
                    return_code=-1,
                )

    def execute_test_script(self, test_script: str) -> EvaluationResult:
        """Execute a test script with pytest and return the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.py")
            xml_path = os.path.join(tmpdir, "results.xml")

            with open(script_path, "w") as f:
                f.write(test_script)

            try:
                start_time = time.time()
                cmd = [
                    self.python_executable,
                    "-m",
                    "pytest",
                    script_path,
                    "--junitxml",
                    xml_path,
                    "--color=no",
                    "-o",
                    "console_output_style=classic",
                ]

                if self.config.test_method_timeout is not None:
                    cmd.extend(["--timeout", str(self.config.test_method_timeout)])

                proc = subprocess.Popen(
                    cmd,
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
                        error=f"Test execution exceeded memory limit of {self.config.max_memory_mb} MB",
                        execution_time=execution_time,
                        timeout=False,
                        return_code=-1,
                    )
                    analyzer = PytestXmlAnalyzer()
                    return analyzer.analyze_pytest_xml(None, basic_result)

                xml_content = None
                if os.path.exists(xml_path):
                    with open(xml_path, "r", encoding="utf-8") as f:
                        xml_content = f.read()

                basic_result = BasicExecutionResult(
                    success=proc.returncode == 0,
                    output=stdout_data,
                    error=stderr_data,
                    execution_time=execution_time,
                    timeout=False,
                    return_code=proc.returncode,
                )

                analyzer = PytestXmlAnalyzer()
                return analyzer.analyze_pytest_xml(xml_content, basic_result)

            except subprocess.TimeoutExpired:
                basic_result = BasicExecutionResult(
                    success=False,
                    output="",
                    error=f"Test execution timed out after {self.config.timeout} seconds",
                    execution_time=self.config.timeout,
                    timeout=True,
                    return_code=-1,
                )
                analyzer = PytestXmlAnalyzer()
                return analyzer.analyze_pytest_xml(None, basic_result)
