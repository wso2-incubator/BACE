"""_CodeLLMHelpers — private mixin for code operator LLM utilities."""

from __future__ import annotations

from coevolution.core.interfaces.language import ICodeParser


class _CodeLLMHelpers:
    parser: ICodeParser  # provided by BaseLLMService

    def _extract_all_code_blocks(self, response: str) -> list[str]:
        return self.parser.extract_code_blocks(response)

    def _contains_starter_code(self, code: str, starter_code: str) -> bool:
        return self.parser.contains_starter_code(code, starter_code)

    def _validated_code(self, code: str, starter_code: str, op: str) -> str:
        if not self._contains_starter_code(code, starter_code):
            raise ValueError(f"{op} result does not contain starter code structure.")
        return code
