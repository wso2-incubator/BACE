"""Generic subprocess implementation of the ISandbox protocol."""

import os
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Optional, TYPE_CHECKING
from coevolution.core.interfaces.sandbox import ISandbox
from infrastructure.sandbox.memory import MemoryMonitor
from infrastructure.sandbox.types import BasicExecutionResult, SandboxConfig

if TYPE_CHECKING:
    from coevolution.core.interfaces.language import ILanguageRuntime, ITestAnalyzer
    from coevolution.core.interfaces.data import EvaluationResult


class SubprocessSandbox(ISandbox):
    """
    A generic sandbox environment for securely executing arbitrary system commands.
    Manages common OS-level constraints such as timeouts, memory limits, and capturing streams.
    """

    @classmethod
    def from_config(cls, config: SandboxConfig) -> "SubprocessSandbox":
        """
        Create a sandbox instance from a configuration object.
        """
        return cls(config=config)

    def __init__(self, config: SandboxConfig) -> None:
        """
        Initialize the sandbox with safety parameters.

        Args:
            config: Sandbox configuration parameters
        """
        self.config = config

    @property
    def timeout(self) -> int:
        """Backward compatibility for timeout attribute."""
        return self.config.timeout

    def execute_command(
        self, cmd: list[str], cwd: Optional[str] = None
    ) -> BasicExecutionResult:
        """
        Execute an arbitrary command safely in a restricted environment.

        Args:
            cmd: The command to execute as a list of arguments.
            cwd: Optional working directory. If None, uses a temporary directory.
        """
        if not cmd:
            return BasicExecutionResult(
                success=False,
                output="",
                error="Received empty command array.",
                execution_time=0.0,
                timeout=False,
                return_code=-1,
            )

        # Ensure the executable exists
        executable = cmd[0]
        if not shutil.which(executable):
            return BasicExecutionResult(
                success=False,
                output="",
                error=f"Executable '{executable}' not found. Please ensure it is installed and available in your PATH.",
                execution_time=0.0,
                timeout=False,
                return_code=-1,
            )

        # Allow execution with a provided cwd, or provision a temporary one
        if cwd is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                return self._run_subprocess(cmd, tmpdir)
        else:
            return self._run_subprocess(cmd, cwd)

    def execute_test_script(
        self,
        test_script: str,
        runtime: "ILanguageRuntime",
        analyzer: "ITestAnalyzer",
    ) -> "EvaluationResult":
        """Execute a test script and return a single test result."""
        from coevolution.core.interfaces.data import EvaluationResult

        with tempfile.TemporaryDirectory() as tmpdir:
            file_ext = runtime.file_extension
            script_path = os.path.join(tmpdir, f"test_script{file_ext}")
            xml_path = os.path.join(tmpdir, "results.xml")

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(test_script)

            kwargs = {}
            if hasattr(self.config, "test_method_timeout"):
                kwargs["timeout"] = self.config.test_method_timeout

            cmd = runtime.get_test_command(script_path, xml_path, **kwargs)
            raw_result = self.execute_command(cmd, cwd=tmpdir)

            xml_content = None
            if os.path.exists(xml_path):
                with open(xml_path, "r", encoding="utf-8") as f:
                    xml_content = f.read()

            result: EvaluationResult = analyzer.analyze(
                raw_result, xml_content=xml_content
            )

        return result

    def execute_code(
        self, code: str, runtime: "ILanguageRuntime"
    ) -> BasicExecutionResult:
        """Execute arbitrary code safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_ext = runtime.file_extension
            script_path = os.path.join(tmpdir, f"eval_script{file_ext}")

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            cmd = runtime.get_execution_command(script_path)
            return self.execute_command(cmd, cwd=tmpdir)

    def _run_subprocess(self, cmd: list[str], cwd: str) -> BasicExecutionResult:
        """Helper to manage the actual Popen execution and resource monitoring."""
        try:
            start_time = time.time()
            # Create a new process group/session so we can kill orphaned children.
            # On macOS/Unix, start_new_session=True uses os.setsid().
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                start_new_session=True,
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

            stdout = stdout_data[: self.config.max_output_size] if stdout_data else ""
            stderr = stderr_data[: self.config.max_output_size] if stderr_data else ""

            return BasicExecutionResult(
                success=proc.returncode == 0,
                output=stdout,
                error=stderr,
                execution_time=execution_time,
                timeout=False,
                return_code=proc.returncode,
            )

        except subprocess.TimeoutExpired:
            # Kill the entire process group to ensure no orphaned children 
            # (e.g. test runner subprocesses) hold open the pipes indefinitely.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            
            proc.communicate()  # reaps the zombie and flushes stdout/stderr buffers
            return BasicExecutionResult(
                success=False,
                output="",
                error=f"Command execution timed out after {self.config.timeout} seconds",
                execution_time=self.config.timeout,
                timeout=True,
                return_code=-1,
            )
        except Exception as e:
            return BasicExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=0.0,
                timeout=False,
                return_code=-1,
            )
