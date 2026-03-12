import pytest
from unittest.mock import MagicMock
from coevolution.strategies.llm_base import BaseLLMService, LLMSyntaxError, ILanguageModel
from coevolution.core.interfaces.language import ICodeParser

class MockLLMService(BaseLLMService):
    def execute(self, prompt: str) -> str:
        return self._generate(prompt)

def test_llm_syntax_retry():
    mock_llm = MagicMock(spec=ILanguageModel)
    mock_llm.generate.return_value = "invalid code"
    
    mock_parser = MagicMock(spec=ICodeParser)
    # First two calls return False (invalid), third call returns True (valid)
    mock_parser.is_syntax_valid.side_effect = [False, False, True]
    mock_parser.extract_code_blocks.side_effect = lambda x: [x]
    
    # We need a method decorated with llm_retry that calls _validate_syntax
    from coevolution.strategies.llm_base import llm_retry
    
    class RetryableService(MockLLMService):
        @llm_retry(exception_types=(LLMSyntaxError,))
        def generate_and_validate(self, prompt: str) -> str:
            val = self._generate(prompt)
            self._validate_syntax(val)
            return val

    retry_service = RetryableService(mock_llm, mock_parser, "python")
    
    # This should succeed after 3 attempts (2 failures, 1 success)
    result = retry_service.generate_and_validate("some prompt")
    
    assert result == "invalid code"
    assert mock_llm.generate.call_count == 3
    assert mock_parser.is_syntax_valid.call_count == 3

def test_llm_syntax_retry_failure():
    mock_llm = MagicMock(spec=ILanguageModel)
    mock_llm.generate.return_value = "invalid code"
    
    mock_parser = MagicMock(spec=ICodeParser)
    # Always return False
    mock_parser.is_syntax_valid.return_value = False
    mock_parser.extract_code_blocks.side_effect = lambda x: [x]

    from coevolution.strategies.llm_base import llm_retry
    
    class RetryableService(MockLLMService):
        @llm_retry(exception_types=(LLMSyntaxError,))
        def generate_and_validate(self, prompt: str) -> str:
            val = self._generate(prompt)
            self._validate_syntax(val)
            return val

    retry_service = RetryableService(mock_llm, mock_parser, "python")
    
    # This should fail after 3 attempts
    with pytest.raises(LLMSyntaxError):
        retry_service.generate_and_validate("some prompt")
    
    assert mock_llm.generate.call_count == 3
