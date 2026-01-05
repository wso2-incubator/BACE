"""Type definitions for sandbox operations."""

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class TestResult:
    """Individual test result with detailed information."""

    name: str
    description: str  # Usually the docstring or test description
    status: Literal["passed", "failed", "error"]
    # Full error message/traceback for failed/error tests
    details: Optional[str] = None


@dataclass
class BasicExecutionResult:
    """Result of basic code execution."""

    success: bool
    output: str
    error: str
    execution_time: float
    timeout: bool
    return_code: int


@dataclass
class TestExecutionResult:
    """Simplified result of test script execution focusing on test analysis."""

    # Test-specific analysis
    script_error: bool
    tests_passed: int
    tests_failed: int
    tests_errors: int

    # Ordered list of test results (in execution order)
    test_results: List[TestResult]

    # Simple summary message
    summary: str

    @property
    def total_tests(self) -> int:
        """Total number of tests that ran."""
        return self.tests_passed + self.tests_failed + self.tests_errors

    @property
    def success_rate(self) -> float:
        """Percentage of tests that passed (0.0 to 1.0)."""
        if self.total_tests == 0:
            return 0.0
        return self.tests_passed / self.total_tests

    @property
    def has_failures(self) -> bool:
        """Whether any tests failed or had errors."""
        return self.tests_failed > 0 or self.tests_errors > 0

    @property
    def all_tests_passed(self) -> bool:
        """Whether all tests passed successfully."""
        return (
            self.total_tests > 0
            and self.tests_failed == 0
            and self.tests_errors == 0
            and not self.script_error
        )


@dataclass
class TestAnalysis:
    """Internal analysis of unittest output."""

    passed: int
    failed: int
    errors: int
    script_error: bool
    test_results: List[TestResult]
