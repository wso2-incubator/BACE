"""_TestLLMHelpers — private mixin for unittest operator LLM utilities."""

from __future__ import annotations

from coevolution.core.interfaces.language import ILanguage, LanguageParsingError


class _TestLLMHelpers:
    """Mixin: test-specific extraction utilities."""

    language_adapter: ILanguage  # provided by BaseLLMService

    def _extract_test_functions(self, code: str) -> list[str]:
        return self.language_adapter.split_tests(code)

    def _extract_first_test_function(self, code: str) -> str:
        functions = self._extract_test_functions(code)
        if not functions:
            raise LanguageParsingError("No test functions found in generated code")
        return functions[0]
