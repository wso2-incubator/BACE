"""Pytest XML analyzer for extracting test results."""

import re
import xml.etree.ElementTree as ET
from typing import Literal, Optional

from loguru import logger

from coevolution.core.interfaces.data import EvaluationResult

from .types import BasicExecutionResult


class PytestXmlAnalyzer:
    """
    Analyzes pytest JUnit XML output to extract a single test result.

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
    ) -> EvaluationResult:
        """
        Analyze pytest XML output and return a single test result.

        Args:
            xml_content: The XML content from pytest --junitxml output, or None if XML wasn't generated
            basic_result: Basic execution result from code execution

        Returns:
            EvaluationResult with detailed test analysis
        """
        logger.debug(
            f"Analyzing pytest XML output: xml_available={xml_content is not None}, success={basic_result.success}, return_code={basic_result.return_code}"
        )

        # If XML content is available, parse it
        if xml_content:
            try:
                return self._parse_xml_content(xml_content, basic_result)
            except Exception as e:
                logger.warning(
                    f"Failed to parse XML content: {e}. Falling back to error analysis."
                )
                # Fall through to error analysis

        # No XML or parsing failed - analyze based on basic result
        return self._analyze_execution_error(basic_result)

    def _parse_xml_content(
        self, xml_content: str, basic_result: BasicExecutionResult
    ) -> EvaluationResult:
        """Parse JUnit XML content and extract the test result."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise

        # We assume there's exactly one test case as we run one test function
        testcase = root.find(".//testcase")

        if testcase is None:
            # No test case found in XML, check if there's a suite-level error
            testsuite = root.find(".//testsuite")
            if testsuite is not None and int(testsuite.get("errors", 0)) > 0:
                # Look for error tag in testsuite or testcases
                error_node = testsuite.find(".//error")
                if error_node is not None:
                    raw_details = error_node.text or error_node.get("message", "")
                    details = self._sanitize_details(raw_details)
                    return EvaluationResult(
                        status="error",
                        error_log=details,
                        execution_time=basic_result.execution_time,
                    )

            return self._analyze_execution_error(basic_result)

        test_name = testcase.get("name", "unknown")

        # Determine status and details
        status: Literal["passed", "failed", "error"] = "passed"
        details = None
        script_error = False

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
            # Check if this is a script-level error (syntax, import, etc.)
            if details:
                details_lower = details.lower()
                if any(
                    err in details_lower
                    for err in [
                        "syntaxerror",
                        "indentationerror",
                        "importerror",
                        "modulenotfounderror",
                        "invalid syntax",
                        "was never closed",
                    ]
                ):
                    script_error = True

        # Check if this test timed out (already handled by status="error" usually, but can check details)
        if details:
            details_lower = details.lower()
            if "timeout" in details_lower or "timed out" in details_lower:
                logger.warning(f"Test '{test_name}' timed out.")

        try:
            execution_time = float(testcase.get("time", 0.0))
        except (ValueError, TypeError):
            execution_time = basic_result.execution_time

        return EvaluationResult(
            status=status,
            error_log=details,
            execution_time=execution_time,
        )

    def _analyze_execution_error(
        self, basic_result: BasicExecutionResult
    ) -> EvaluationResult:
        """Analyze execution results when XML parsing failed or wasn't available."""
        script_error = False
        details = basic_result.error or basic_result.output or "No output available"

        # Check if this was a script-level error (syntax, import, etc.)
        if not basic_result.success:
            # Look for common error patterns in stderr/details
            error_output = details.lower()
            if any(
                error_type in error_output
                for error_type in [
                    "syntaxerror",
                    "indentationerror",
                    "importerror",
                    "modulenotfounderror",
                ]
            ):
                script_error = True

        status: Literal["error"] = "error"
        if basic_result.timeout:
            details = (
                f"Execution timed out after {basic_result.execution_time}s. {details}"
            )

        return EvaluationResult(
            status=status,
            error_log=self._sanitize_details(details),
            execution_time=basic_result.execution_time,
        )
        )
