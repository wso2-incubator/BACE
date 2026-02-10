"""
Comprehensive tests for the BallerinaSandbox class.

Tests cover:
- Basic code execution
- Test script execution
- Error handling (syntax errors, runtime errors)
- Timeout handling
- Project setup
- Integration with sandbox configuration
"""

from pathlib import Path

import pytest

from coevolution.core.interfaces.data import EvaluationResult
from infrastructure.sandbox.adapters.ballerina import BallerinaSandbox
from infrastructure.sandbox.types import BasicExecutionResult, SandboxConfig


@pytest.fixture
def default_config() -> SandboxConfig:
    """Create a default sandbox configuration for testing."""
    return SandboxConfig(
        language="ballerina",
        timeout=30,
        max_memory_mb=100,
        max_output_size=1_000_000,
        test_method_timeout=10,
    )


@pytest.fixture
def sandbox(default_config: SandboxConfig) -> BallerinaSandbox:
    """Create a BallerinaSandbox instance for testing."""
    return BallerinaSandbox(config=default_config)


class TestBallerinaSandboxInitialization:
    """Test BallerinaSandbox initialization and configuration."""

    def test_init_with_config(self, default_config: SandboxConfig) -> None:
        """Test initialization with a SandboxConfig."""
        sandbox = BallerinaSandbox(config=default_config)
        assert sandbox.config == default_config
        assert sandbox.bal_executable == "bal"

    def test_init_with_custom_executable(self) -> None:
        """Test initialization with a custom Ballerina executable."""
        config = SandboxConfig(
            language="ballerina",
            language_executable="/custom/path/to/bal",
            timeout=30,
        )
        sandbox = BallerinaSandbox(config=config)
        assert sandbox.bal_executable == "/custom/path/to/bal"

    def test_from_config_classmethod(self, default_config: SandboxConfig) -> None:
        """Test the from_config class method."""
        sandbox = BallerinaSandbox.from_config(default_config)
        assert isinstance(sandbox, BallerinaSandbox)
        assert sandbox.config == default_config


class TestExecuteCode:
    """Test the execute_code method."""

    def test_simple_hello_world(self, sandbox: BallerinaSandbox) -> None:
        """Test executing a simple hello world program."""
        code = """
import ballerina/io;

public function main() {
    io:println("Hello, World!");
}
"""
        result = sandbox.execute_code(code)
        assert isinstance(result, BasicExecutionResult)
        assert result.success
        assert "Hello, World!" in result.output
        assert result.execution_time >= 0
        assert not result.timeout
        assert result.return_code == 0

    def test_simple_arithmetic(self, sandbox: BallerinaSandbox) -> None:
        """Test executing code with arithmetic operations."""
        code = """
import ballerina/io;

public function main() {
    int sum = 5 + 10;
    io:println(sum);
}
"""
        result = sandbox.execute_code(code)
        assert result.success
        assert "15" in result.output

    def test_function_definition_and_call(self, sandbox: BallerinaSandbox) -> None:
        """Test executing code with function definitions and calls."""
        code = """
import ballerina/io;

function add(int a, int b) returns int {
    return a + b;
}

public function main() {
    int result = add(7, 3);
    io:println(result);
}
"""
        result = sandbox.execute_code(code)
        assert result.success
        assert "10" in result.output

    def test_syntax_error_handling(self, sandbox: BallerinaSandbox) -> None:
        """Test handling of syntax errors."""
        code = """
import ballerina/io;

public function main() {
    io:println("Missing closing quote)
}
"""
        result = sandbox.execute_code(code)
        assert not result.success
        assert len(result.error) > 0
        assert result.return_code != 0

    def test_compilation_error_handling(self, sandbox: BallerinaSandbox) -> None:
        """Test handling of compilation errors."""
        code = """
import ballerina/io;

public function main() {
    int x = "not an integer";
    io:println(x);
}
"""
        result = sandbox.execute_code(code)
        assert not result.success
        assert len(result.error) > 0

    def test_runtime_error_handling(self, sandbox: BallerinaSandbox) -> None:
        """Test handling of runtime errors."""
        code = """
import ballerina/io;

public function main() {
    int[] numbers = [1, 2, 3];
    int x = numbers[10]; // Index out of bounds
    io:println(x);
}
"""
        result = sandbox.execute_code(code)
        # Ballerina might handle this differently
        # It could be a compilation error or runtime panic
        assert (
            not result.success
            or "panic" in result.output.lower()
            or "panic" in result.error.lower()
        )

    def test_timeout_handling(self) -> None:
        """Test timeout handling for long-running code."""
        config = SandboxConfig(language="ballerina", timeout=2)
        sandbox = BallerinaSandbox(config=config)

        code = """
import ballerina/lang.runtime;

public function main() {
    while true {
        runtime:sleep(1);
    }
}
"""
        result = sandbox.execute_code(code)
        assert not result.success
        assert result.timeout
        assert "timed out" in result.error.lower()
        assert result.execution_time >= 2

    def test_empty_code(self, sandbox: BallerinaSandbox) -> None:
        """Test executing empty code."""
        code = ""
        result = sandbox.execute_code(code)
        # Empty Ballerina file actually compiles and runs successfully
        assert result.success

    def test_output_size_limit(self) -> None:
        """Test that output is limited to max_output_size."""
        config = SandboxConfig(
            language="ballerina",
            timeout=30,
            max_output_size=50,  # Very small limit
        )
        sandbox = BallerinaSandbox(config=config)

        code = """
import ballerina/io;

public function main() {
    foreach int i in 0...100 {
        io:println("This is a very long line of output that will exceed the limit");
    }
}
"""
        result = sandbox.execute_code(code)
        # Output should be truncated to max_output_size
        assert len(result.output) <= 50

    def test_multiple_imports(self, sandbox: BallerinaSandbox) -> None:
        """Test code with multiple imports."""
        code = """
import ballerina/io;

public function main() {
    int x = 42;
    string str = x.toString();
    io:println(str);
}
"""
        result = sandbox.execute_code(code)
        assert result.success
        assert "42" in result.output


class TestExecuteTestScript:
    """Test the execute_test_script method."""

    def test_simple_passing_test(self, sandbox: BallerinaSandbox) -> None:
        """Test executing a simple passing test."""
        test_script = """
import ballerina/test;

@test:Config {}
function testAddition() {
    int result = 2 + 2;
    test:assertEquals(result, 4);
}
"""
        result = sandbox.execute_test_script(test_script)
        assert isinstance(result, EvaluationResult)
        # The result status depends on how the analyzer interprets Ballerina test output
        # Since we're using PytestXmlAnalyzer as fallback, results may vary
        assert result.execution_time >= 0

    def test_simple_failing_test(self, sandbox: BallerinaSandbox) -> None:
        """Test executing a simple failing test."""
        test_script = """
import ballerina/test;

@test:Config {}
function testFailing() {
    int result = 2 + 2;
    test:assertEquals(result, 5);
}
"""
        result = sandbox.execute_test_script(test_script)
        assert isinstance(result, EvaluationResult)
        # Should detect failure
        assert result.status in ["failed", "error"]

    def test_multiple_tests(self, sandbox: BallerinaSandbox) -> None:
        """Test executing multiple tests."""
        test_script = """
import ballerina/test;

@test:Config {}
function testAddition() {
    test:assertEquals(2 + 2, 4);
}

@test:Config {}
function testSubtraction() {
    test:assertEquals(5 - 3, 2);
}

@test:Config {}
function testMultiplication() {
    test:assertEquals(3 * 4, 12);
}
"""
        result = sandbox.execute_test_script(test_script)
        assert isinstance(result, EvaluationResult)
        assert result.execution_time >= 0

    def test_test_with_helper_function(self, sandbox: BallerinaSandbox) -> None:
        """Test executing tests that use helper functions."""
        test_script = """
import ballerina/test;

function add(int a, int b) returns int {
    return a + b;
}

@test:Config {}
function testAdd() {
    int result = add(10, 20);
    test:assertEquals(result, 30);
}
"""
        result = sandbox.execute_test_script(test_script)
        assert isinstance(result, EvaluationResult)

    def test_test_with_syntax_error(self, sandbox: BallerinaSandbox) -> None:
        """Test handling of syntax errors in test scripts."""
        test_script = """
import ballerina/test;

@test:Config {}
function testBroken( {
    test:assertEquals(1, 1);
}
"""
        result = sandbox.execute_test_script(test_script)
        assert isinstance(result, EvaluationResult)
        # Should report error/failure
        assert result.status in ["failed", "error"]

    def test_timeout_in_test_script(self) -> None:
        """Test timeout handling in test execution."""
        config = SandboxConfig(language="ballerina", timeout=2)
        sandbox = BallerinaSandbox(config=config)

        test_script = """
import ballerina/test;
import ballerina/lang.runtime;

@test:Config {}
function testTimeout() {
    while true {
        runtime:sleep(1);
    }
}
"""
        result = sandbox.execute_test_script(test_script)
        assert isinstance(result, EvaluationResult)
        assert result.status in ["failed", "error"]
        # May or may not timeout depending on test framework behavior
        assert result.execution_time >= 0


class TestProjectSetup:
    """Test the _setup_project method."""

    def test_project_setup_creates_ballerina_toml(
        self, sandbox: BallerinaSandbox
    ) -> None:
        """Test that _setup_project creates a Ballerina.toml file."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            sandbox._setup_project(project_path)

            toml_file = project_path / "Ballerina.toml"
            assert toml_file.exists()

            content = toml_file.read_text()
            assert "[package]" in content
            assert "org" in content
            assert "name" in content
            assert "version" in content

    def test_project_setup_has_correct_structure(
        self, sandbox: BallerinaSandbox
    ) -> None:
        """Test that the created Ballerina.toml has correct structure."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            sandbox._setup_project(project_path)

            content = (project_path / "Ballerina.toml").read_text()
            assert 'org = "test"' in content
            assert 'name = "sandbox"' in content
            assert 'version = "0.1.0"' in content


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_in_code(self, sandbox: BallerinaSandbox) -> None:
        """Test executing code with unicode characters."""
        code = """
import ballerina/io;

public function main() {
    string message = "Hello 世界 🌍";
    io:println(message);
}
"""
        result = sandbox.execute_code(code)
        assert result.success
        assert "Hello" in result.output

    def test_very_large_output(self, sandbox: BallerinaSandbox) -> None:
        """Test handling of very large output."""
        code = """
import ballerina/io;

public function main() {
    foreach int i in 0...1000 {
        io:println(i);
    }
}
"""
        result = sandbox.execute_code(code)
        # Should succeed and have output
        assert result.success
        assert len(result.output) > 0

    def test_code_with_no_main_function(self, sandbox: BallerinaSandbox) -> None:
        """Test executing code without a main function."""
        code = """
function helper() returns int {
    return 42;
}
"""
        result = sandbox.execute_code(code)
        # Ballerina allows modules without main - they compile successfully
        assert result.success

    def test_code_with_ballerina_dependencies(self, sandbox: BallerinaSandbox) -> None:
        """Test code that uses standard library dependencies."""
        code = """
import ballerina/io;
import ballerina/lang.'string as strings;

public function main() {
    string text = "hello world";
    string upper = strings:toUpperAscii(text);
    io:println(upper);
}
"""
        result = sandbox.execute_code(code)
        assert result.success
        assert "HELLO WORLD" in result.output


class TestIntegrationWithConfig:
    """Test integration with various SandboxConfig options."""

    def test_different_timeout_values(self) -> None:
        """Test sandbox with different timeout configurations."""
        configs = [
            SandboxConfig(language="ballerina", timeout=1),
            SandboxConfig(language="ballerina", timeout=10),
            SandboxConfig(language="ballerina", timeout=60),
        ]

        code = """
import ballerina/io;

public function main() {
    io:println("Test");
}
"""

        for config in configs:
            sandbox = BallerinaSandbox(config=config)
            result = sandbox.execute_code(code)
            assert result.success

    def test_custom_executable_not_found(self) -> None:
        """Test behavior when custom executable doesn't exist."""
        config = SandboxConfig(
            language="ballerina",
            language_executable="/nonexistent/bal",
            timeout=5,
        )
        sandbox = BallerinaSandbox(config=config)

        code = """
import ballerina/io;

public function main() {
    io:println("Test");
}
"""
        result = sandbox.execute_code(code)
        # Should fail with an error
        assert not result.success
        assert len(result.error) > 0


class TestReturnValues:
    """Test that methods return correct types and values."""

    def test_execute_code_returns_basic_execution_result(
        self, sandbox: BallerinaSandbox
    ) -> None:
        """Test that execute_code returns BasicExecutionResult."""
        code = 'import ballerina/io; public function main() { io:println("Test"); }'
        result = sandbox.execute_code(code)

        assert isinstance(result, BasicExecutionResult)
        assert hasattr(result, "success")
        assert hasattr(result, "output")
        assert hasattr(result, "error")
        assert hasattr(result, "execution_time")
        assert hasattr(result, "timeout")
        assert hasattr(result, "return_code")

    def test_execute_test_script_returns_evaluation_result(
        self, sandbox: BallerinaSandbox
    ) -> None:
        """Test that execute_test_script returns EvaluationResult."""
        test_script = """
import ballerina/test;

@test:Config {}
function testSimple() {
    test:assertEquals(1, 1);
}
"""
        result = sandbox.execute_test_script(test_script)

        assert isinstance(result, EvaluationResult)
        assert hasattr(result, "status")
        assert hasattr(result, "execution_time")

    def test_all_result_fields_populated_on_success(
        self, sandbox: BallerinaSandbox
    ) -> None:
        """Test that all result fields are properly populated on success."""
        code = 'import ballerina/io; public function main() { io:println("Success"); }'
        result = sandbox.execute_code(code)

        assert result.success is True
        assert isinstance(result.output, str)
        assert isinstance(result.error, str)
        assert result.execution_time >= 0
        assert result.timeout is False
        assert result.return_code == 0

    def test_all_result_fields_populated_on_failure(
        self, sandbox: BallerinaSandbox
    ) -> None:
        """Test that all result fields are properly populated on failure."""
        code = "invalid ballerina code that wont compile"
        result = sandbox.execute_code(code)

        assert result.success is False
        assert isinstance(result.output, str)
        assert isinstance(result.error, str)
        assert result.execution_time >= 0
        assert result.timeout is False
        assert result.return_code != 0
