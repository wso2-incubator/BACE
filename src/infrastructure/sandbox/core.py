"""Core sandbox implementation for safe code execution."""

import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from coevolution.core.interfaces.data import EvaluationResult

from .analyzer import PytestXmlAnalyzer
from .types import BasicExecutionResult, SandboxConfig


class SafeCodeSandbox:
    """
    A safe sandbox environment for executing Python code with restrictions.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 100,
        max_output_size: int = 10000,
        allowed_imports: Optional[List[str]] = None,
        python_executable: Optional[str] = None,
        test_method_timeout: Optional[int] = None,
    ):
        """
        Initialize the sandbox with safety parameters.

        Args:
            timeout: Maximum execution time in seconds for the entire script
            max_memory_mb: Maximum memory usage in MB (not enforced on all systems)
            max_output_size: Maximum output size in characters
            allowed_imports: List of allowed import modules
            python_executable: Path to Python executable (defaults to sys.executable)
            test_method_timeout: Maximum execution time in seconds for individual test methods (None = no limit)
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size
        self.python_executable = python_executable or sys.executable
        self.test_method_timeout = test_method_timeout
        self.allowed_imports = allowed_imports or [
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
            "pytest_timeout",  # Add pytest-timeout plugin
            "typing",
            "dataclasses",
            "enum",
            "heapq",
            "bisect",
        ]

        # Dangerous modules/functions to block
        # self.blocked_patterns = [
        #     "import os",
        #     # "import sys", # TODO: Consider allowing sys with restrictions
        #     "import subprocess",
        #     "import shutil",
        #     "import socket",
        #     "import urllib",
        #     "import requests",
        #     "import http",
        #     "import ftplib",
        #     "import smtplib",
        #     "import telnetlib",
        #     "import tempfile",
        #     "import pickle",
        #     "import marshal",
        #     "import importlib",
        #     "import __import__",
        #     "exec(",
        #     # "eval(", # TODO: Consider allowing eval with restrictions, this was needed for generating test with actual lcb test cases
        #     "compile(",
        #     "open(",
        #     "file(",
        #     # "input(",
        #     "raw_input(",
        #     "__builtins__",
        #     "__globals__",
        #     "__locals__",
        #     # "globals()",
        #     # "locals()", # TODO: Consider allowing globals/locals with restrictions
        #     "vars()",
        #     "dir()",
        #     "hasattr(",
        #     "getattr(",
        #     "setattr(",
        #     "delattr(",
        #     "exit(",
        #     "quit(",
        #     "reload(",
        # ]

        # TODO: Re-evaluate blocked patterns, for now we disable blocking to allow more flexibility
        self.blocked_patterns: list[str] = []

    def _check_code_safety(self, code: str) -> bool:
        """
        Check if code contains potentially dangerous patterns.

        Args:
            code: The code string to check

        Returns:
            True if code appears safe, False otherwise
        """
        code_lower = code.lower()

        for pattern in self.blocked_patterns:
            if pattern.lower() in code_lower:
                logger.debug(f"Code blocked by pattern '{pattern}'")
                return False

        logger.trace(
            f"Code passed safety checks (checked {len(self.blocked_patterns)} patterns)"
        )
        return True

    def _create_restricted_environment(self) -> Dict[str, Any]:
        """
        Create a restricted environment for code execution.

        Returns:
            Dictionary with restricted built-ins
        """
        # Create a minimal safe environment
        safe_builtins = {
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "sum": sum,
            "min": max,
            "max": max,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "set": set,
            "frozenset": frozenset,
            "type": type,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "print": print,
            "repr": repr,
            "ord": ord,
            "chr": chr,
            "bin": bin,
            "oct": oct,
            "hex": hex,
            "any": any,
            "all": all,
            "pow": pow,
            "divmod": divmod,
            "reversed": reversed,
            "slice": slice,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "IndexError": IndexError,
            "KeyError": KeyError,
            "AttributeError": AttributeError,
        }

        return {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
            "__doc__": None,
        }

    def execute_code(
        self, code: str, capture_output: bool = True
    ) -> BasicExecutionResult:
        """
        Safely execute Python code in a restricted environment.

        Args:
            code: The Python code to execute
            capture_output: Whether to capture stdout/stderr

        Returns:
            BasicExecutionResult containing:
            - success: bool indicating if execution succeeded
            - output: captured stdout
            - error: captured stderr or error message
            - execution_time: time taken in seconds
            - timeout: whether execution timed out
            - return_code: process return code
        """
        # Check code safety first
        if not self._check_code_safety(code):
            return BasicExecutionResult(
                success=False,
                output="",
                error="Code contains potentially dangerous patterns",
                execution_time=0,
                timeout=False,
                return_code=-1,
            )

        # Create temporary directory for code execution
        # Using TemporaryDirectory ensures automatic cleanup even if process is killed
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")

            # Write code to file
            with open(script_path, "w") as f:
                f.write(code)

            logger.trace(
                f"Executing code in sandbox: script_path={script_path} capture_output={capture_output} code_len={len(code)}",
            )

            try:
                # Execute code in subprocess with timeout
                start_time = time.time()

                result = subprocess.run(
                    [self.python_executable, script_path],
                    capture_output=capture_output,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,  # Run in temp directory
                )

                logger.trace(f"Execution finished: returncode={result.returncode}")

                execution_time = time.time() - start_time

                # Limit output size
                if len(result.stdout) > self.max_output_size:
                    logger.warning(
                        f"Truncating stdout from {len(result.stdout)} to {self.max_output_size} characters"
                    )

                if len(result.stderr) > self.max_output_size:
                    logger.warning(
                        f"Truncating stderr from {len(result.stderr)} to {self.max_output_size} characters"
                    )

                stdout = result.stdout[: self.max_output_size] if result.stdout else ""
                stderr = result.stderr[: self.max_output_size] if result.stderr else ""

                logger.trace(
                    f"Captured output sizes: stdout={len(stdout)} stderr={len(stderr)}"
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
                logger.warning(f"Code execution timed out after {self.timeout} seconds")
                return BasicExecutionResult(
                    success=False,
                    output="",
                    error=f"Code execution timed out after {self.timeout} seconds",
                    execution_time=self.timeout,
                    timeout=True,
                    return_code=-1,
                )

            except Exception as e:
                logger.exception(f"Unhandled exception during code execution: {e}")
                return BasicExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution error: {str(e)}",
                    execution_time=0,
                    timeout=False,
                    return_code=-1,
                )

        # Temporary directory automatically cleaned up here

    def execute_test_script(self, test_script: str) -> EvaluationResult:
        """
        Execute a test script containing a single test function and return the result.

        Args:
            test_script: The test script to execute

        Returns:
            EvaluationResult containing the execution outcome
        """
        logger.debug(f"SafeCodeSandbox: executing test script (len={len(test_script)})")

        # Create temporary directory for script and XML output
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.py")
            xml_path = os.path.join(tmpdir, "results.xml")

            # Write test script to file
            with open(script_path, "w") as f:
                f.write(test_script)

            try:
                # Execute pytest with JUnit XML output
                start_time = time.time()

                # Build pytest command
                cmd = [
                    self.python_executable,
                    "-m",
                    "pytest",
                    script_path,
                    "--junitxml",
                    xml_path,
                    "--color=no",  # Force disable ANSI color codes
                    "-o",
                    "console_output_style=classic",  # Use simple output style
                ]

                # Add per-test timeout if configured
                if self.test_method_timeout is not None:
                    cmd.extend(["--timeout", str(self.test_method_timeout)])

                proc_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )

                execution_time = time.time() - start_time

                # Read XML content if file exists
                xml_content = None
                if os.path.exists(xml_path):
                    try:
                        with open(xml_path, "r", encoding="utf-8") as f:
                            xml_content = f.read()
                    except Exception as e:
                        logger.warning(f"Failed to read XML file: {e}")

                # Create basic result for error analysis fallback
                basic_result = BasicExecutionResult(
                    success=proc_result.returncode == 0,
                    output=proc_result.stdout,
                    error=proc_result.stderr,
                    execution_time=execution_time,
                    timeout=False,
                    return_code=proc_result.returncode,
                )

                # Create analyzer and analyze the test results
                analyzer = PytestXmlAnalyzer()
                result = analyzer.analyze_pytest_xml(xml_content, basic_result)

                logger.debug(f"SafeCodeSandbox: test script result: {result.status}")
                return result

            except subprocess.TimeoutExpired:
                logger.warning(f"Test execution timed out after {self.timeout} seconds")
                basic_result = BasicExecutionResult(
                    success=False,
                    output="",
                    error=f"Test execution timed out after {self.timeout} seconds",
                    execution_time=self.timeout,
                    timeout=True,
                    return_code=-1,
                )
                analyzer = PytestXmlAnalyzer()
                return analyzer.analyze_pytest_xml(None, basic_result)

        # Temporary directory automatically cleaned up here

    @classmethod
    def from_config(cls, config: SandboxConfig) -> "SafeCodeSandbox":
        """
        Create a SafeCodeSandbox instance from a SandboxConfig.

        Args:
            config: SandboxConfig object with configuration parameters

        Returns:
            SafeCodeSandbox instance
        """
        return cls(
            timeout=config.timeout,
            max_memory_mb=config.max_memory_mb,
            max_output_size=config.max_output_size,
            allowed_imports=config.allowed_imports,
            python_executable=config.python_executable,
            test_method_timeout=config.test_method_timeout,
        )
        )
