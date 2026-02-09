# src/coevolution/core/interfaces/language.py
"""
Protocol for language-specific operations.
"""

from typing import Any, Protocol


class LanguageParsingError(Exception):
    """Raised when code cannot be parsed by the language toolchain."""

    pass


class LanguageTransformationError(Exception):
    """Raised when code transformation or composition fails."""

    pass


class ILanguageAdapter(Protocol):
    """
    Protocol defining the contract for language-specific operations.

    This allows the core evolution engine to remain language-agnostic by
    delegating syntax-sensitive tasks (extraction, validation, composition)
    to concrete language implementations.
    """

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

    def parse_test_inputs(self, outputs: str) -> list[dict[str, Any]]:
        """
        Parses the raw output from a test input generator into a list of input dictionaries.
        """
        ...
