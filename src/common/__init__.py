"""
Common utilities shared across all APR projects.

This module provides shared functionality including:
- CodeProcessor: For processing and extracting code segments
- LLMClient: Unified interface for different LLM providers
"""

from .code_processor import CodeProcessor
from .llm_client import LLMClient

__all__ = ['CodeProcessor', 'LLMClient']
__version__ = "0.1.0"
