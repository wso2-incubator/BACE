# src/coevolution/core/interfaces/language.py
"""
Protocol for language-specific operations.
"""

from typing import Any, Protocol, runtime_checkable

from coevolution.core.exceptions import (
    LanguageError,
    LanguageParsingError,
    LanguageTransformationError,
)
from coevolution.core.interfaces.data import BasicExecutionResult, EvaluationResult

# Re-export so existing callers that import from this module are unaffected.
__all__ = [
    "ICodeParser",
    "IScriptComposer",
    "ILanguageRuntime",
    "ITestAnalyzer",
    "ILanguage",
    "LanguageError",
    "LanguageParsingError",
    "LanguageTransformationError",
]


@runtime_checkable
class ITestAnalyzer(Protocol):
    """
    Protocol for analyzing test execution output.

    Each language implementation should provide an analyzer that converts
    raw execution results (stdout, stderr, XML) into a structured
    EvaluationResult.
    """

    def analyze(
        self, raw_result: BasicExecutionResult, **kwargs: Any
    ) -> EvaluationResult:
        """
        Analyze a raw execution result and return an EvaluationResult.

        Args:
            raw_result: The raw execution result.
            **kwargs: Additional context, such as XML content for Python.

        Returns:
            EvaluationResult: The structured test result.
        """
        ...


@runtime_checkable
class ICodeParser(Protocol):
    """Protocol for static code analysis and parsing."""

    def extract_code_blocks(self, response: str) -> list[str]: ...
    def is_syntax_valid(self, code: str) -> bool: ...
    def extract_test_names(self, test_code: str) -> list[str]: ...
    def split_tests(self, test_code: str) -> list[str]: ...
    def remove_main_block(self, code: str) -> str: ...
    def normalize_code(self, code: str) -> str: ...
    def contains_starter_code(self, code: str, starter_code: str) -> bool: ...
    def get_structural_metadata(self, code: str) -> dict[str, Any]: ...
    def parse_test_inputs(self, outputs: str) -> list[dict[str, Any]]: ...
    def get_docstring(self, code: str) -> str: ...


@runtime_checkable
class IScriptComposer(Protocol):
    """Protocol for generative script composition."""

    def compose_test_script(self, code_snippet: str, test_snippet: str) -> str: ...
    def compose_evaluation_script(self, code_snippet: str, input_data: str) -> str: ...
    def generate_test_case(
        self, input_str: str, output_str: str, starter_code: str, test_number: int
    ) -> str: ...
    def compose_generator_script(self, generator_code: str, num_inputs: int) -> str: ...


@runtime_checkable
class ILanguageRuntime(Protocol):
    """Protocol for language-specific runtime infrastructure."""

    @property
    def file_extension(self) -> str: ...

    def get_execution_command(self, file_path: str) -> list[str]: ...
    def get_test_command(
        self, test_file_path: str, result_xml_path: str, **kwargs: Any
    ) -> list[str]: ...


@runtime_checkable
class ILanguage(Protocol):
    """
    Facade Protocol defining the aggregate contract for language operations.
    Exposes specialized capabilities as properties.
    """

    @property
    def language(self) -> str: ...

    @property
    def parser(self) -> ICodeParser: ...

    @property
    def composer(self) -> IScriptComposer: ...

    @property
    def runtime(self) -> ILanguageRuntime: ...

    @property
    def analyzer(self) -> ITestAnalyzer: ...
