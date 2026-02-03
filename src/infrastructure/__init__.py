"""
Common utilities shared across all APR projects.

This module provides shared functionality including:
- Code Preprocessing: For parsing, analyzing, transforming, and building test code
- LLMClient: Unified interface for different LLM providers
- SafeCodeSandbox: Safe execution environment for generated code
- Enhanced typing: TestExecutionResult, BasicExecutionResult, TestDetails for type safety
- Configuration: BaseConfig and specialized config classes for experiments
- LLM Factory: Factory functions for creating LLM instances
- Coevolution: Bayesian coevolution algorithms with selection strategies

Usage (Hierarchical Imports):
    # Code preprocessing - use submodules
    from infrastructure.code_preprocessing.parsers import extract_code_block_from_response
    from infrastructure.code_preprocessing.analyzers import parse_code_structure
    from infrastructure.code_preprocessing.builders import build_test_script_for_humaneval
    from infrastructure.code_preprocessing import CodeParsingError  # Exceptions available directly

    # Coevolution - use submodules (now at root)
    from coevolution.config import CoevolutionConfig
    from coevolution.bayesian import initialize_prior_beliefs
    from coevolution.operators import CodeOperator, TestOperator

    # Other utilities available directly
    from infrastructure import LLMClient, SafeCodeSandbox
    from infrastructure.config import BaseConfig, ExperimentConfig
"""

# Only re-export exceptions from code_preprocessing for convenience
from .code_preprocessing import (
    CodeParsingError,
    CodeProcessingError,
    CodeTransformationError,
)
from .llm_client import LLMClient
from .sandbox import (
    BasicExecutionResult,
    CodeExecutionError,
    CodeExecutionTimeoutError,
    PytestXmlAnalyzer,
    SafeCodeSandbox,
    SandboxConfig,
    TestExecutor,
    TestResult,
    check_test_execution_status,
    create_safe_test_environment,
    create_test_executor,
)

__all__ = [
    # Code Preprocessing - Exceptions only
    "CodeProcessingError",
    "CodeParsingError",
    "CodeTransformationError",
    # Core utilities
    "LLMClient",
    "SafeCodeSandbox",
    "SandboxConfig",
    "create_safe_test_environment",
    "check_test_execution_status",
    # Types
    "BasicExecutionResult",
    "TestResult",
    "CodeExecutionError",
    "CodeExecutionTimeoutError",
    # New architecture
    "TestExecutor",
    "PytestXmlAnalyzer",
    "create_test_executor",
]
__version__ = "0.1.0"
