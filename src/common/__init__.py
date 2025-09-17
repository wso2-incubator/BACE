"""
Common utilities shared across all APR projects.

This module provides shared functionality including:
- CodeProcessor: For processing and extracting code segments
- LLMClient: Unified interface for different LLM providers
- SafeCodeSandbox: Safe execution environment for generated code
"""

from .code_processor import CodeProcessor
from .llm_client import LLMClient
from .sandbox import SafeCodeSandbox, create_safe_test_environment

__all__ = ['CodeProcessor', 'LLMClient',
           'SafeCodeSandbox', 'create_safe_test_environment']
__version__ = "0.1.0"
