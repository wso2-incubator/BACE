"""Ballerina test output analyzer."""

import re
from typing import Any, Literal

from loguru import logger

from coevolution.core.interfaces.language import ITestAnalyzer
from coevolution.core.interfaces.data import BasicExecutionResult, EvaluationResult


def truncate_preserve_tail(s: str | None, max_len: int) -> str:
    """Truncate a string preserving head and tail with a snip marker."""
    if not s:
        return ""
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    head = max_len // 2
    tail = max_len - head
    return s[:head] + "\n...<snip>...\n" + s[-tail:]


class BallerinaTestAnalyzer(ITestAnalyzer):
    """
    Analyzer for Ballerina test output.
    """

    def analyze(self, raw_result: BasicExecutionResult, **kwargs: Any) -> EvaluationResult:
        """
        Analyze Ballerina test execution output.
        """
        output = getattr(raw_result, "output", "") or ""
        stderr = getattr(raw_result, "error", "") or ""

        logger.debug(
            f"Analyzing Ballerina test output: "
            f"success={getattr(raw_result, 'success', False)}, "
            f"output_length={len(output)}"
        )

        # Check for timeout
        if getattr(raw_result, "timeout", False):
            return EvaluationResult(
                status="error",
                error_log=truncate_preserve_tail(
                    f"Test execution timed out after {getattr(raw_result, 'execution_time', 0)}s",
                    4000,
                ),
                execution_time=getattr(raw_result, "execution_time", 0.0),
            )

        # Check if no tests were found
        if "No tests found" in output:
            return EvaluationResult(
                status="error",
                error_log=truncate_preserve_tail(
                    "No tests found. Ensure test functions are annotated with @test:Config.",
                    4000,
                ),
                execution_time=getattr(raw_result, "execution_time", 0.0),
            )

        # Parse test results
        status = self._determine_status(output, raw_result)
        error_log = self._extract_error_details(output, stderr, raw_result)

        error_log = truncate_preserve_tail(error_log, 4000) if error_log else error_log

        return EvaluationResult(
            status=status,
            error_log=error_log,
            execution_time=getattr(raw_result, "execution_time", 0.0),
        )

    def _determine_status(
        self, output: str, basic_result: Any
    ) -> Literal["passed", "failed", "error"]:
        """Determine test status from output."""
        if "error: compilation contains errors" in output.lower():
            return "error"

        summary_pattern = r"(\d+)\s+passing\s+(\d+)\s+failing\s+(\d+)\s+skipped"
        summary_match = re.search(summary_pattern, output)

        if summary_match:
            passing = int(summary_match.group(1))
            failing = int(summary_match.group(2))

            if failing > 0:
                return "failed"
            elif passing > 0:
                return "passed"

        if re.search(r":\s+has failed\.", output):
            return "failed"

        if re.search(r":\s+has passed\.", output):
            return "passed"

        if getattr(basic_result, "success", False) and "Running Tests" in output:
            return "passed"

        return "error"

    def _extract_error_details(
        self, output: str, stderr: str, basic_result: Any
    ) -> str:
        """Extract detailed error information."""
        error_parts = []
        fail_pattern = r"\[fail\]\s+(\w+):\s*\n\n\s*(.*?)(?=\n\s*\n\s*\d+\s+passing|$)"
        fail_matches = re.finditer(fail_pattern, output, re.DOTALL)

        for match in fail_matches:
            test_name = match.group(1)
            error_detail = match.group(2).strip()
            error_formatted = self._format_test_error(test_name, error_detail)
            error_parts.append(error_formatted)

        if not error_parts:
            if "expected:" in output.lower() and "actual:" in output.lower():
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

        compilation_errors = self._extract_compilation_errors(output, stderr)
        if compilation_errors:
            return compilation_errors

        if not error_parts:
            if "Running Tests" in output:
                test_section = output.split("Running Tests")[1]
                error_parts.append(test_section.strip())
            else:
                error_parts.append(output.strip())

        if stderr:
            stderr_lines = [
                line
                for line in stderr.split("\n")
                if line.strip()
                and not line.startswith("WARNING:")
                and "deprecated" not in line.lower()
            ]
            if stderr_lines:
                error_parts.append("Errors:\n" + "\n".join(stderr_lines))

        if error_parts:
            return "\n\n".join(error_parts)

        if not getattr(basic_result, "success", False):
            return output or stderr or "Test execution failed with no output"

        return ""

    def _format_test_error(self, test_name: str, error_detail: str) -> str:
        """Format a TestError message."""
        test_error_pattern = r'error\s+\{ballerina/test:\d+\}TestError\s+\("([^"]+)"\)'
        match = re.search(test_error_pattern, error_detail)

        if match:
            message = match.group(1)
            if "expected:" in message and "actual" in message:
                parts = message.split("\n")
                cleaned_parts = [" ".join(p.split()) for p in parts if p.strip()]
                message = "\n  ".join(cleaned_parts)
                formatted = f"Test '{test_name}' failed:\n  {message}"
            else:
                message = " ".join(message.split())
                formatted = f"Test '{test_name}' failed: {message}"

            file_pattern = r"fileName:\s+tests/([^\s]+)\s+lineNumber:\s+(\d+)"
            file_match = re.search(file_pattern, error_detail)
            if file_match:
                formatted += f"\n  at {file_match.group(1)}:{file_match.group(2)}"
            return formatted

        return f"Test '{test_name}' failed:\n{re.sub(r'\s+', ' ', error_detail)}"

    def _extract_compilation_errors(self, output: str, stderr: str) -> str:
        """Extract compilation errors."""
        combined = (output or "") + "\n" + (stderr or "")
        if not (re.search(r"\berror\b", combined, re.I)):
            return ""

        lines = combined.split("\n")
        match_indexes = [i for i, line in enumerate(lines) if re.search(r"\berror\b", line, re.I)]
        if not match_indexes:
            return ""

        blocks = []
        seen = set()
        for idx in match_indexes:
            block = "\n".join(lines[max(0, idx - 3):min(len(lines), idx + 4)]).strip()
            if block and block not in seen:
                seen.add(block)
                blocks.append(block)

        if not blocks:
            return ""

        formatted_errors = "Compilation errors:\n"
        for block in blocks:
            indented = "\n".join("  " + line for line in block.split("\n"))
            formatted_errors += indented + "\n  ---\n"
        return formatted_errors[:4000].rstrip()
