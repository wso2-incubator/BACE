"""
Common utilities shared across all APR projects.

This module provides shared functionality including:
- Languages: Python and Ballerina language adapters (via infrastructure.languages)
- LLMClient: Unified interface for different LLM providers
- SafeCodeSandbox: Safe execution environment for generated code
- Enhanced typing: EvaluationResult, BasicExecutionResult
- Configuration: BaseConfig and specialized config classes for experiments
- LLM Factory: Factory functions for creating LLM instances
- Coevolution: Bayesian coevolution algorithms with selection strategies

Usage:
    from infrastructure.languages import PythonLanguage, BallerinaLanguage
    from infrastructure import LLMClient, SafeCodeSandbox
    from infrastructure.config import BaseConfig, ExperimentConfig
    from coevolution.core.interfaces import LanguageParsingError, LanguageTransformationError
"""

from coevolution.core.exceptions import (
    LanguageError,
    LanguageParsingError,
    LanguageTransformationError,
)
from coevolution.core.interfaces.data import EvaluationResult

from .llm_client import LLMClient
from .sandbox import (
    BasicExecutionResult,
    CodeExecutionError,
    CodeExecutionTimeoutError,
    SafeCodeSandbox,
    SandboxConfig,
    TestExecutor,
    check_test_execution_status,
    create_safe_test_environment,
    create_test_executor,
)

__all__ = [
    # Language-agnostic exceptions (moved from code_preprocessing)
    "LanguageError",
    "LanguageParsingError",
    "LanguageTransformationError",
    # Core utilities
    "LLMClient",
    "SafeCodeSandbox",
    "SandboxConfig",
    "create_safe_test_environment",
    "check_test_execution_status",
    # Types
    "BasicExecutionResult",
    "EvaluationResult",
    "CodeExecutionError",
    "CodeExecutionTimeoutError",
    # New architecture
    "TestExecutor",
    "create_test_executor",
]
__version__ = "0.1.0"
__version__ = "0.1.0"
