"""
Comprehensive tests for the SafeCodeSandbox class and associated utilities.

Refactored for single-test pytest architecture.
"""

from coevolution.core.interfaces.data import EvaluationResult
from infrastructure.sandbox import (
    SafeCodeSandbox,
    TestExecutor,
    check_test_execution_status,
    create_safe_test_environment,
)


class TestTestExecutorWrapper:
    """Test cases for the high-level TestExecutor."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        from infrastructure.sandbox.types import SandboxConfig
        config = SandboxConfig(timeout=30)
        self.executor = TestExecutor(config=config)

    def test_basic_execution(self) -> None:
        """Test basic code execution via execute_code."""
        code = """
def add(a, b):
    return a + b

result = add(2, 3)
print(result)
"""
        result = self.executor.execute_code(code)
        assert result.success
        assert "5" in result.output
        assert result.error is None or result.error == ""
        assert result.timeout is False

    def test_syntax_error_handling_basic(self) -> None:
        """Test handling of syntax errors in basic execution."""
        code = """
def broken_function(
    print("This has a syntax error")
"""
        result = self.executor.execute_code(code)
        assert not result.success
        assert "SyntaxError" in result.error or "syntax" in result.error.lower()

    def test_timeout_handling_basic(self) -> None:
        """Test timeout handling for infinite loops in basic execution."""
        code = """
while True:
    pass
"""
        from infrastructure.sandbox.types import SandboxConfig
        config = SandboxConfig(timeout=1)
        executor = TestExecutor(config=config)
        result = executor.execute_code(code)
        assert not result.success
        assert result.timeout is True

    def test_single_test_passing(self) -> None:
        """Test executing a single passing pytest function."""
        test_script = """
def test_addition():
    assert 2 + 2 == 4
"""
        result = self.executor.execute_test_script(test_script)
        assert isinstance(result, EvaluationResult)
        assert result.status == "passed"
        assert result.execution_time >= 0

    def test_single_test_failing(self) -> None:
        """Test executing a single failing pytest function."""
        test_script = """
def test_failing():
    assert 1 + 1 == 3
"""
        result = self.executor.execute_test_script(test_script)
        assert result.status == "failed"
        assert result.error_log is not None
        assert "AssertionError" in result.error_log

    def test_single_test_error(self) -> None:
        """Test executing a single pytest function that raises an error."""
        test_script = """
def test_error():
    raise ValueError("Something went wrong")
"""
        result = self.executor.execute_test_script(test_script)
        assert result.status in ["failed", "error"]
        assert result.error_log is not None
        assert "ValueError: Something went wrong" in result.error_log

    def test_script_level_syntax_error(self) -> None:
        """Test handling of syntax errors at the script level (collection failure)."""
        test_script = """
def test_broken(:
    pass
"""
        result = self.executor.execute_test_script(test_script)
        assert result.status == "error"
        assert result.error_log is not None
        assert (
            "SyntaxError" in result.error_log
            or "invalid syntax" in result.error_log.lower()
        )

    def test_script_level_import_error(self) -> None:
        """Test handling of import errors at the script level."""
        test_script = """
import nonexistent_module_xyz
def test_something():
    pass
"""
        result = self.executor.execute_test_script(test_script)
        assert result.status == "error"
        assert result.error_log is not None
        assert "no module named" in result.error_log.lower()

    def test_per_test_timeout(self) -> None:
        """Test that per-test timeout works if configured."""
        test_script = """
import time
def test_slow():
    time.sleep(5)
"""
        from infrastructure.sandbox.types import SandboxConfig
        config = SandboxConfig(timeout=10, test_method_timeout=2)
        executor = TestExecutor(config=config)
        result = executor.execute_test_script(test_script)
        assert result.status == "error" or result.status == "failed"
        assert result.error_log is not None
        assert (
            "timeout" in result.error_log.lower()
            or "timed out" in result.error_log.lower()
        )


class TestTestExecutor:
    """Test cases for TestExecutor high-level wrapper."""

    def test_executor_delegation(self) -> None:
        """Test that TestExecutor correctly delegates and returns EvaluationResult."""
        executor = TestExecutor(timeout=5)
        test_script = "def test_pass(): assert True"
        result = executor.execute_test_script(test_script)
        assert isinstance(result, EvaluationResult)
        assert result.status == "passed"


class TestSandboxUtils:
    """Test cases for sandbox utility functions."""

    def test_create_safe_test_environment(self) -> None:
        sandbox = create_safe_test_environment()
        assert isinstance(sandbox, SafeCodeSandbox)
        assert sandbox.timeout == 300

    def test_check_test_execution_status_helper(self) -> None:
        # Test passed
        res_pass = EvaluationResult(status="passed")
        assert "TEST PASSED" in check_test_execution_status(res_pass)

        # Test failed
        res_fail = EvaluationResult(status="failed", error_log="AssertionError")
        assert "TEST FAILED" in check_test_execution_status(res_fail)

        # Script error
        res_script = EvaluationResult(status="error", error_log="SyntaxError")
        assert "TEST ERROR" in check_test_execution_status(res_script)
        assert "SCRIPT ERROR" in check_test_execution_status(res_script)

        # Script error
        res_script = EvaluationResult(status="error", error_log="SyntaxError")
        assert "TEST ERROR" in check_test_execution_status(res_script)
        assert "SCRIPT ERROR" in check_test_execution_status(res_script)
        # Script error
        res_script = EvaluationResult(status="error", error_log="SyntaxError")
        assert "TEST ERROR" in check_test_execution_status(res_script)
        assert "SCRIPT ERROR" in check_test_execution_status(res_script)
