"""_TestLLMHelpers — private mixin for unittest operator LLM utilities."""

from __future__ import annotations

from coevolution.core.interfaces.language import ICodeParser, LanguageParsingError


class _TestLLMHelpers:
    """Mixin: test-specific extraction utilities."""

    parser: ICodeParser  # provided by BaseLLMService

    def _extract_test_functions(self, code: str) -> list[str]:
        return self.parser.split_tests(code)

    def _extract_first_test_function(self, code: str) -> str:
        functions = self._extract_test_functions(code)
        if not functions:
            raise LanguageParsingError("No test functions found in generated code")
        return functions[0]
