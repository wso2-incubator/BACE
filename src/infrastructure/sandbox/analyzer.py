"""Pytest XML analyzer for extracting test results."""

import re
import xml.etree.ElementTree as ET
from typing import Literal, Optional

from loguru import logger

from .types import BasicExecutionResult, TestExecutionResult, TestResult


class PytestXmlAnalyzer:
    """
    Analyzes pytest JUnit XML output to extract detailed test results.

    This class parses the structured XML output from pytest --junitxml
    to provide robust and reliable test result analysis.
    """

    # Regex to identify ANSI escape sequences (both raw and hex-encoded)
    ANSI_ESCAPE_PATTERN = re.compile(
        r"(?:\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])|#x1B\[[0-?]*[ -/]*[@-~])"
    )

    # Regex to capture absolute paths to temporary python files (handles Unix and Windows)
    # Matches examples like: /var/.../tmppjtsol6j.py:425 or C:\Temp\tmppjtsol6j.py:425
    TEMP_PATH_PATTERN = re.compile(r"(?:(?:[A-Za-z]:\\)|/)[^\s:<>\"]+\.py(?::\d+)?")

    # Regex to capture relative temp file references
    # Matches examples like: tmphhckiio3.py:80: or tmphhckiio3.py:64:
    TEMP_FILENAME_PATTERN = re.compile(r"\btmp\w{7,12}\.py(?::\d+)?:?")

    # Regex to capture module-like prefixes in instance/class representations
    # Example: "<tmppjtsol6j.TestMakeAEqualB testMethod=test_x>" -> "<TestMakeAEqualB testMethod=test_x>"
    MODULE_PREFIX_PATTERN = re.compile(r"<[A-Za-z0-9_]+\.([A-Za-z_][A-Za-z0-9_]*)")

    def __init__(self) -> None:
        """Initialize the pytest XML analyzer."""
        pass

    def _remove_ansi_codes(self, text: Optional[str]) -> Optional[str]:
        """Strip ANSI escape codes from string."""
        if not text:
            return text
        return self.ANSI_ESCAPE_PATTERN.sub("", text)

    def _sanitize_details(self, text: Optional[str]) -> Optional[str]:
        """
        Strip ANSI codes AND sanitize temporary file paths/module names.

        Replaces absolute temporary file paths with a generic `test_script.py` while
        preserving optional line numbers, and removes random module prefixes from
        class/instance representations.
        """
        if not text:
            return text

        # 1. Remove ANSI codes
        sanitized = self.ANSI_ESCAPE_PATTERN.sub("", text)

        # 2. Replace full temporary file paths with a generic name, preserving line numbers
        def _path_repl(m: re.Match[str]) -> str:
            match_text = m.group(0)
            # If there's a colon with a line number, preserve it
            colon_idx = match_text.rfind(":")
            if colon_idx != -1 and match_text[colon_idx + 1 :].isdigit():
                return f"test_script.py:{match_text[colon_idx + 1 :]}"
            return "test_script.py"

        sanitized = self.TEMP_PATH_PATTERN.sub(_path_repl, sanitized)

        # 3. Replace relative temp file references with generic name, preserving line numbers
        def _filename_repl(m: re.Match[str]) -> str:
            match_text = m.group(0)
            # Extract line number if present (before the trailing colon)
            # Handle cases like tmpXXX.py:80: or tmpXXX.py:64
            if ":" in match_text:
                parts = match_text.rstrip(":").split(":")
                if len(parts) == 2 and parts[1].isdigit():
                    return f"test_script.py:{parts[1]}"
            return "test_script.py"

        sanitized = self.TEMP_FILENAME_PATTERN.sub(_filename_repl, sanitized)

        # 4. Clean up the class/module instance representation
        sanitized = self.MODULE_PREFIX_PATTERN.sub(r"<\1", sanitized)

        return sanitized

    def analyze_pytest_xml(
        self, xml_content: Optional[str], basic_result: BasicExecutionResult
    ) -> TestExecutionResult:
        """
        Analyze pytest XML output and return detailed test information.

        Args:
            xml_content: The XML content from pytest --junitxml output, or None if XML wasn't generated
            basic_result: Basic execution result from code execution

        Returns:
            TestExecutionResult with detailed test analysis
        """
        logger.debug(
            f"Analyzing pytest XML output: xml_available={xml_content is not None}, success={basic_result.success}, return_code={basic_result.return_code}"
        )

        # If XML content is available, parse it
        if xml_content:
            try:
                return self._parse_xml_content(xml_content)
            except Exception as e:
                logger.warning(
                    f"Failed to parse XML content: {e}. Falling back to error analysis."
                )
                # Fall through to error analysis

        # No XML or parsing failed - analyze based on basic result
        return self._analyze_execution_error(basic_result)

    def _parse_xml_content(self, xml_content: str) -> TestExecutionResult:
        """Parse JUnit XML content and extract test results."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise

        total_tests = 0
        total_failures = 0
        total_errors = 0
        test_results = []
        script_error = False

        # Parse each testsuite
        for testsuite in root.findall(".//testsuite"):
            suite_tests = int(testsuite.get("tests", 0))
            suite_failures = int(testsuite.get("failures", 0))
            suite_errors = int(testsuite.get("errors", 0))

            total_tests += suite_tests
            total_failures += suite_failures
            total_errors += suite_errors

            # Parse individual test cases
            for testcase in testsuite.findall("testcase"):
                test_name = testcase.get("name", "unknown")
                classname = testcase.get("classname", "")

                # Determine status and details
                status: Literal["passed", "failed", "error"] = "passed"
                details = None

                # Check for failure
                failure = testcase.find("failure")
                if failure is not None:
                    status = "failed"
                    raw_details = failure.text or failure.get("message", "")
                    details = self._sanitize_details(raw_details)

                # Check for error
                error = testcase.find("error")
                if error is not None:
                    status = "error"
                    raw_details = error.text or error.get("message", "")
                    details = self._sanitize_details(raw_details)
                    # Check if this is a syntax error (script-level error)
                    if details and (
                        "SyntaxError" in details
                        or "was never closed" in details
                        or "invalid syntax" in details
                    ):
                        script_error = True

                # Create description from classname and test name
                description = f"{classname}.{test_name}" if classname else test_name

                test_result = TestResult(
                    name=test_name,
                    description=description,
                    status=status,
                    details=details,
                )
                test_results.append(test_result)

        # Create summary
        if script_error:
            summary = "Script execution failed: syntax error"
        elif total_failures > 0 or total_errors > 0:
            summary = f"Tests completed: {total_tests - total_failures - total_errors} passed, {total_failures} failed, {total_errors} errors"
        elif total_tests > 0:
            summary = f"All tests passed: {total_tests} tests"
        else:
            summary = "No tests were found or executed"

        return TestExecutionResult(
            script_error=script_error,
            tests_passed=total_tests - total_failures - total_errors,
            tests_failed=total_failures,
            tests_errors=total_errors,
            test_results=test_results,
            summary=summary,
        )

    def _analyze_execution_error(
        self, basic_result: BasicExecutionResult
    ) -> TestExecutionResult:
        """Analyze execution results when XML parsing failed or wasn't available."""
        script_error = False
        summary = ""

        # Check if this was a script-level error (syntax, import, etc.)
        if not basic_result.success:
            # Look for common error patterns in stderr
            error_output = (basic_result.error or "").lower()
            if any(
                error_type in error_output
                for error_type in [
                    "syntaxerror",
                    "indentationerror",
                    "importerror",
                    "modulenotfounderror",
                    "attributeerror",
                    "nameerror",
                    "typeerror",
                ]
            ):
                script_error = True
                summary = f"Script execution failed: {basic_result.error}"
            else:
                # Other execution failure
                script_error = True
                summary = f"Test execution failed: {basic_result.error}"
        else:
            # Execution succeeded but no XML was generated
            summary = "Test execution completed but no XML output was generated"

        return TestExecutionResult(
            script_error=script_error,
            tests_passed=0,
            tests_failed=0,
            tests_errors=0,
            test_results=[],
            summary=summary,
        )
