"""
Common utilities shared across all APR projects.

This module provides shared functionality including:
- CodeProcessor: For processing and extracting code segments
- LLMClient: Unified interface for different LLM providers
- SafeCodeSandbox: Safe execution environment for generated code
- Enhanced typing: TestExecutionResult, BasicExecutionResult, TestDetails for type safety
"""

from .code_processor import CodeProcessor
from .llm_client import LLMClient
from .sandbox import (
    SafeCodeSandbox,
    create_safe_test_environment,
    check_test_execution_status,
    TestExecutionResult,
    BasicExecutionResult,
    TestDetails,
    TestAnalysis,
    ExecutionCategory,
    CodeExecutionError,
    CodeExecutionTimeoutError
)

__all__ = [
    'CodeProcessor',
    'LLMClient',
    'SafeCodeSandbox',
    'create_safe_test_environment',
    'check_test_execution_status',
    'TestExecutionResult',
    'BasicExecutionResult',
    'TestDetails',
    'TestAnalysis',
    'ExecutionCategory',
    'CodeExecutionError',
    'CodeExecutionTimeoutError'
]
__version__ = "0.1.0"
