"""Python test output analyzer utilizing pytest JUnit XML."""

import re
import xml.etree.ElementTree as ET
from typing import Any, Literal, Optional

from loguru import logger

from coevolution.core.interfaces.language import ITestAnalyzer
from coevolution.core.interfaces.data import BasicExecutionResult, EvaluationResult


class PythonTestAnalyzer(ITestAnalyzer):
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
    TEMP_PATH_PATTERN = re.compile(r"(?:(?:[A-Za-z]:\\)|/)[^\s:<>\"]+\.py(?::\d+)?")

    # Regex to capture relative temp file references
    TEMP_FILENAME_PATTERN = re.compile(r"\btmp\w{7,12}\.py(?::\d+)?:?")

    # Regex to capture module-like prefixes in instance/class representations
    MODULE_PREFIX_PATTERN = re.compile(r"<[A-Za-z0-9_]+\.([A-Za-z_][A-Za-z0-9_]*)")

    def __init__(self) -> None:
        """Initialize the python test analyzer."""
        pass

    def _remove_ansi_codes(self, text: Optional[str]) -> Optional[str]:
        """Strip ANSI escape codes from string."""
        if not text:
            return text
        return self.ANSI_ESCAPE_PATTERN.sub("", text)

    def _sanitize_details(self, text: Optional[str]) -> Optional[str]:
        """
        Strip ANSI codes AND sanitize temporary file paths/module names.
        """
        if not text:
            return text

        # 1. Remove ANSI codes
        sanitized = self.ANSI_ESCAPE_PATTERN.sub("", text)

        # 2. Replace full temporary file paths with a generic name
        def _path_repl(m: re.Match[str]) -> str:
            match_text = m.group(0)
            colon_idx = match_text.rfind(":")
            if colon_idx != -1 and match_text[colon_idx + 1 :].isdigit():
                return f"test_script.py:{match_text[colon_idx + 1 :]}"
            return "test_script.py"

        sanitized = self.TEMP_PATH_PATTERN.sub(_path_repl, sanitized)

        # 3. Replace relative temp file references
        def _filename_repl(m: re.Match[str]) -> str:
            match_text = m.group(0)
            if ":" in match_text:
                parts = match_text.rstrip(":").split(":")
                if len(parts) == 2 and parts[1].isdigit():
                    return f"test_script.py:{parts[1]}"
            return "test_script.py"

        sanitized = self.TEMP_FILENAME_PATTERN.sub(_filename_repl, sanitized)

        # 4. Clean up the class/module instance representation
        sanitized = self.MODULE_PREFIX_PATTERN.sub(r"<\1", sanitized)

        return sanitized

    def analyze(self, raw_result: BasicExecutionResult, **kwargs: Any) -> EvaluationResult:
        """
        Analyze pytest XML output and return a single test result.

        Args:
            raw_result: Basic execution result from code execution
            **kwargs: Can include 'xml_content'
        """
        xml_content = kwargs.get("xml_content")
        
        logger.debug(
            f"Analyzing Python test results: xml_available={xml_content is not None}"
        )

        if xml_content:
            try:
                return self._parse_xml_content(xml_content, raw_result)
            except Exception as e:
                logger.warning(
                    f"Failed to parse XML content: {e}. Falling back to error analysis."
                )

        return self._analyze_execution_error(raw_result)

    def _parse_xml_content(
        self, xml_content: str, basic_result: Any
    ) -> EvaluationResult:
        """Parse JUnit XML content and extract the test result."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise

        testcase = root.find(".//testcase")

        if testcase is None:
            testsuite = root.find(".//testsuite")
            if testsuite is not None and int(testsuite.get("errors", 0)) > 0:
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
        status: Literal["passed", "failed", "error"] = "passed"
        details = None

        failure = testcase.find("failure")
        if failure is not None:
            status = "failed"
            raw_details = failure.text or failure.get("message", "")
            details = self._sanitize_details(raw_details)

        error = testcase.find("error")
        if error is not None:
            status = "error"
            raw_details = error.text or error.get("message", "")
            details = self._sanitize_details(raw_details)

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
        self, basic_result: Any
    ) -> EvaluationResult:
        """Analyze execution results when XML parsing failed or wasn't available."""
        details = getattr(basic_result, "error", None) or getattr(basic_result, "output", None) or "No output available"

        status: Literal["error"] = "error"
        if getattr(basic_result, "timeout", False):
            details = (
                f"Execution timed out after {getattr(basic_result, 'execution_time', 0)}s. {details}"
            )

        return EvaluationResult(
            status=status,
            error_log=self._sanitize_details(details),
            execution_time=getattr(basic_result, "execution_time", 0.0),
        )
