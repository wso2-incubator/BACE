# src/coevolution/core/interfaces/language.py
"""
Protocol for language-specific operations.
"""

from typing import Any, Protocol

from coevolution.core.exceptions import (
    LanguageError,
    LanguageParsingError,
    LanguageTransformationError,
)

# Re-export so existing callers that import from this module are unaffected.
__all__ = [
    "ILanguage",
    "LanguageError",
    "LanguageParsingError",
    "LanguageTransformationError",
]


class ILanguage(Protocol):
    """
    Protocol defining the contract for language-specific operations.

    This allows the core evolution engine to remain language-agnostic by
    delegating syntax-sensitive tasks (extraction, validation, composition)
    to concrete language implementations.
    """

    @property
    def language(self) -> str:
        """
        Return the name of the programming language.
        """
        ...

    def extract_code_blocks(self, response: str) -> list[str]:
        """
        Extract code snippets from a raw LLM response.
        Should handle Markdown-style code blocks for the specific language.
        """
        ...

    def is_syntax_valid(self, code: str) -> bool:
        """
        Check if the provided code snippet has valid syntax in the target language.
        """
        ...

    def extract_test_names(self, test_code: str) -> list[str]:
        """
        Extract the names of individual test cases from a test suite/snippet.
        """
        ...

    def split_tests(self, test_code: str) -> list[str]:
        """
        Split a block of test code into individual standalone test functions/methods.
        """
        ...

    def compose_test_script(self, code_snippet: str, test_snippet: str) -> str:
        """
        Combine a code snippet and a test snippet into a single executable script.
        """
        ...

    def compose_evaluation_script(self, code_snippet: str, input_data: str) -> str:
        """
        Combine a code snippet and raw input data into an executable script
        that prints the output of the function call.
        """
        ...

    def generate_test_case(
        self, input_str: str, output_str: str, starter_code: str, test_number: int
    ) -> str:
        """
        Generate a test case string based on raw input and expected output data.
        """
        ...

    def remove_main_block(self, code: str) -> str:
        """
        Remove the 'if __name__ == "__main__":' block (or equivalent) from code.
        """
        ...

    def normalize_code(self, code: str) -> str:
        """
        Normalize code by removing comments, docstrings, and standardizing whitespace
        to allow for structural comparison.
        """
        ...

    def contains_starter_code(self, code: str, starter_code: str) -> bool:
        """
        Check if the starter code structure is present in the provided code.
        """
        ...

    def get_structural_metadata(self, code: str) -> dict[str, Any]:
        """
        Extract structural metadata such as function names, class definitions,
        and imports.
        """
        ...

    def compose_generator_script(self, generator_code: str, num_inputs: int) -> str:
        """
        Compose a script that executes the generator function and prints the result.
        """
        ...

    def get_execution_command(self, file_path: str) -> list[str]:
        """
        Provide the command to execute a script in this language.
        """
        ...

    def get_test_command(
        self, test_file_path: str, result_xml_path: str, **kwargs: Any
    ) -> list[str]:
        """
        Provide the command to execute tests for a script in this language.
        """
        ...

    def parse_test_results(
        self, raw_result: Any, xml_content: str | None = None
    ) -> Any:
        # raw_result is BasicExecutionResult, return is EvaluationResult
        # We use Any here to avoid circular imports in the interface layer
        """
        Parse raw execution results into a structured EvaluationResult.
        """
        ...

    def parse_test_inputs(self, outputs: str) -> list[dict[str, Any]]:
        """
        Parses the raw output from a test input generator into a list of input dictionaries.
        """
        ...
