"""Exceptions for LLM client operations."""

from typing import Optional


class TokenLimitExceededError(Exception):
    """Raised when the output token limit is exceeded."""

    def __init__(self, current_tokens: int, limit: int, message: Optional[str] = None):
        self.current_tokens = current_tokens
        self.limit = limit
        if message is None:
            message = f"Token limit exceeded: {current_tokens} tokens generated, limit is {limit}"
        super().__init__(message)
