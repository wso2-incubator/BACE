"""
Ballerina test output analyzer.

Parses Ballerina test execution output to extract test results and error information.
"""

import re
from typing import Literal

from loguru import logger

from coevolution.core.interfaces.data import EvaluationResult
from infrastructure.sandbox.types import BasicExecutionResult


def truncate_preserve_tail(s: str | None, max_len: int) -> str:
    """Truncate a string preserving head and tail with a snip marker.

    If `s` is shorter than `max_len`, return it unchanged. Otherwise return
    first half + marker + last half so callers keep both the beginning and
    the end of long logs.
    """
    if not s:
        return ""
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    head = max_len // 2
    tail = max_len - head
    return s[:head] + "\n...<snip>...\n" + s[-tail:]


class BallerinaTestAnalyzer:
    """
    Analyzer for Ballerina test output.

    Ballerina test output format:
    - Test status: "testName: has passed." or "testName: has failed."
    - Error details: Multi-line error with expected/actual values
    - Summary: "X passing, Y failing, Z skipped"
    """

    def analyze_test_output(
        self, basic_result: BasicExecutionResult
    ) -> EvaluationResult:
        """
        Analyze Ballerina test execution output.

        Args:
            basic_result: Basic execution result with stdout/stderr

        Returns:
            EvaluationResult with test status and error details
        """
        output = basic_result.output or ""
        stderr = basic_result.error or ""

        logger.debug(
            f"Analyzing Ballerina test output: "
            f"success={basic_result.success}, "
            f"return_code={basic_result.return_code}, "
            f"output_length={len(output)}, "
            f"stderr_length={len(stderr)}"
        )

        # Check for timeout
        if basic_result.timeout:
            return EvaluationResult(
                status="error",
                error_log=truncate_preserve_tail(
                    f"Test execution timed out after {basic_result.execution_time}s",
                    4000,
                ),
                execution_time=basic_result.execution_time,
            )

        # Check if no tests were found
        if "No tests found" in output:
            return EvaluationResult(
                status="error",
                error_log=truncate_preserve_tail(
                    "No tests found. Ensure test functions are annotated with @test:Config and located in the tests/ directory.",
                    4000,
                ),
                execution_time=basic_result.execution_time,
            )

        # Parse test results
        status = self._determine_status(output, basic_result)
        error_log = self._extract_error_details(output, stderr, basic_result)

        # Always truncate the final error_log to avoid returning huge logs
        error_log = truncate_preserve_tail(error_log, 4000) if error_log else error_log

        return EvaluationResult(
            status=status,
            error_log=error_log,
            execution_time=basic_result.execution_time,
        )

    def _determine_status(
        self, output: str, basic_result: BasicExecutionResult
    ) -> Literal["passed", "failed", "error"]:
        """
        Determine test status from output.

        Args:
            output: Test stdout
            basic_result: Basic execution result

        Returns:
            Test status: "passed", "failed", or "error"
        """
        # Check for compilation errors
        if "error: compilation contains errors" in output.lower():
            return "error"

        # Look for test summary line: "X passing Y failing Z skipped"
        summary_pattern = r"(\d+)\s+passing\s+(\d+)\s+failing\s+(\d+)\s+skipped"
        summary_match = re.search(summary_pattern, output)

        if summary_match:
            passing = int(summary_match.group(1))
            failing = int(summary_match.group(2))

            if failing > 0:
                return "failed"
            elif passing > 0:
                return "passed"

        # Check for individual test failures
        if re.search(r":\s+has failed\.", output):
            return "failed"

        # Check for individual test passes
        if re.search(r":\s+has passed\.", output):
            return "passed"

        # If basic_result indicates success and we see test execution
        if basic_result.success and "Running Tests" in output:
            return "passed"

        # Default to error for unclear cases
        return "error"

    def _extract_error_details(
        self, output: str, stderr: str, basic_result: BasicExecutionResult
    ) -> str:
        """
        Extract detailed error information from test output.

        Args:
            output: Test stdout
            stderr: Test stderr
            basic_result: Basic execution result

        Returns:
            Formatted error details
        """
        error_parts = []

        # Extract test failure details
        # Pattern: [fail] testName: followed by error details until next section
        fail_pattern = r"\[fail\]\s+(\w+):\s*\n\n\s*(.*?)(?=\n\s*\n\s*\d+\s+passing|$)"
        fail_matches = re.finditer(fail_pattern, output, re.DOTALL)

        for match in fail_matches:
            test_name = match.group(1)
            error_detail = match.group(2).strip()

            # Parse the TestError format to extract meaningful information
            error_formatted = self._format_test_error(test_name, error_detail)
            error_parts.append(error_formatted)

        # If no structured failures found, look for error messages
        if not error_parts:
            # Check for expected/actual format
            if "expected:" in output.lower() and "actual:" in output.lower():
                # Extract the assertion error section
                lines = output.split("\n")
                in_error_section = False
                error_section = []

                for line in lines:
                    if "has failed" in line:
                        in_error_section = True
                        error_section.append(line)
                    elif in_error_section:
                        if "passing" in line or "failing" in line:
                            break
                        error_section.append(line)

                if error_section:
                    error_parts.append("\n".join(error_section))

        # Check for compilation errors - these can be in stdout or stderr
        compilation_errors = self._extract_compilation_errors(output, stderr)
        if compilation_errors:
            return compilation_errors

        # If no structured failures found, look for error messages
        if not error_parts:
            # Include the Running Tests section and onwards
            if "Running Tests" in output:
                test_section = output.split("Running Tests")[1]
                error_parts.append(test_section.strip())
            else:
                error_parts.append(output.strip())

        # Filter and add stderr (exclude warnings)
        if stderr:
            stderr_lines = [
                line
                for line in stderr.split("\n")
                if line.strip()
                and not line.startswith("WARNING:")
                and "deprecated" not in line.lower()
                and "Unsafe" not in line
            ]
            if stderr_lines:
                error_parts.append("Errors:\n" + "\n".join(stderr_lines))

        # Combine all error parts
        if error_parts:
            return "\n\n".join(error_parts)

        # If no errors found but status suggests there should be
        if not basic_result.success:
            return output or stderr or "Test execution failed with no output"

        return ""

    def _format_test_error(self, test_name: str, error_detail: str) -> str:
        """
        Format a TestError message into a more readable format.

        Args:
            test_name: Name of the failing test
            error_detail: Raw error detail from output

        Returns:
            Formatted, readable error message
        """
        # Extract the main error message from TestError format
        # Pattern: error {ballerina/test:0}TestError ("message...")
        test_error_pattern = r'error\s+\{ballerina/test:\d+\}TestError\s+\("([^"]+)"\)'
        match = re.search(test_error_pattern, error_detail)

        if match:
            message = match.group(1)

            # Clean up excessive whitespace while preserving line breaks for expected/actual
            # Look for patterns like "message\n     expected: ... actual: ..."
            if "expected:" in message and "actual" in message:
                # Split by newlines and clean each part
                parts = message.split("\n")
                cleaned_parts = []
                for part in parts:
                    cleaned = " ".join(part.split())  # Normalize whitespace
                    if cleaned:
                        cleaned_parts.append(cleaned)
                message = "\n  ".join(cleaned_parts)
                formatted = f"Test '{test_name}' failed:\n  {message}"
            else:
                # Simple message - just normalize whitespace
                message = " ".join(message.split())
                formatted = f"Test '{test_name}' failed: {message}"

            # Try to extract file location from the stack trace
            file_pattern = r"fileName:\s+tests/([^\s]+)\s+lineNumber:\s+(\d+)"
            file_match = re.search(file_pattern, error_detail)
            if file_match:
                filename = file_match.group(1)
                line_num = file_match.group(2)
                formatted += f"\n  at {filename}:{line_num}"

            return formatted

        # Fallback: Clean up whitespace
        error_detail = re.sub(r"\s+", " ", error_detail)
        return f"Test '{test_name}' failed:\n{error_detail}"

    def _extract_compilation_errors(self, output: str, stderr: str) -> str:
        """
        Extract and format compilation errors from output or stderr.

        Args:
            output: Test stdout
            stderr: Test stderr

        Returns:
            Formatted compilation errors or empty string if none found
        """
        # Check both output and stderr for compilation errors
        combined = (output or "") + "\n" + (stderr or "")

        # Use a case-insensitive search for the word 'error' or explicit compilation marker
        if not (
            re.search(r"\berror\b", combined, re.I)
            or "compilation contains errors" in combined.lower()
        ):
            return ""

        lines = combined.split("\n")
        match_indexes: list[int] = []

        file_line_pat = re.compile(r"\S+\.[A-Za-z0-9_]+:\d+")

        for i, line in enumerate(lines):
            if re.search(r"\berror\b", line, re.I) or file_line_pat.search(line):
                match_indexes.append(i)

        if not match_indexes:
            return ""

        # Gather small context windows around each match and deduplicate
        blocks: list[str] = []
        seen = set()
        for idx in match_indexes:
            start = max(0, idx - 3)
            end = min(len(lines), idx + 4)
            block = "\n".join(lines[start:end]).strip()
            if not block or block in seen:
                continue
            seen.add(block)
            blocks.append(block)

        if not blocks:
            return ""

        formatted_errors = "Compilation errors:\n"
        for block in blocks:
            # Indent block for readability
            indented = "\n".join("  " + line for line in block.split("\n"))
            formatted_errors += indented + "\n  ---\n"

        # Truncate overly long formatted error text to keep logs readable
        MAX_CHARS = 4000
        if len(formatted_errors) > MAX_CHARS:
            formatted_errors = formatted_errors[:MAX_CHARS] + "\n...<truncated>..."

        return formatted_errors.rstrip()
