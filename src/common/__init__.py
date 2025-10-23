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
    from common.code_preprocessing.parsers import extract_code_block_from_response
    from common.code_preprocessing.analyzers import parse_code_structure
    from common.code_preprocessing.builders import build_test_script_for_humaneval
    from common.code_preprocessing import CodeParsingError  # Exceptions available directly

    # Coevolution - use submodules
    from common.coevolution.config import CoevolutionConfig
    from common.coevolution.bayesian import initialize_prior_beliefs
    from common.coevolution.operators import CodeOperator, TestOperator

    # Other utilities available directly
    from common import LLMClient, SafeCodeSandbox
    from common.config import BaseConfig, ExperimentConfig
"""

# Only re-export exceptions from code_preprocessing for convenience
from .code_preprocessing import (
    CodeParsingError,
    CodeProcessingError,
    CodeTransformationError,
)
from .config import AgentCoderConfig, BaseConfig, ExperimentConfig, SimpleConfig
from .llm_client import LLMClient
from .llm_factory import (
    check_provider_availability,
    create_llm_from_params,
    create_llm_instance,
    get_available_providers,
    validate_provider_model_combination,
)
from .sandbox import (
    BasicExecutionResult,
    CodeExecutionError,
    CodeExecutionTimeoutError,
    SafeCodeSandbox,
    TestAnalysis,
    TestExecutionResult,
    TestExecutor,
    TestResultAnalyzer,
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
    "create_safe_test_environment",
    "check_test_execution_status",
    # Types
    "TestExecutionResult",
    "BasicExecutionResult",
    "TestAnalysis",
    "CodeExecutionError",
    "CodeExecutionTimeoutError",
    # New architecture
    "TestExecutor",
    "create_test_executor",
    "TestResultAnalyzer",
    # Configuration
    "BaseConfig",
    "SimpleConfig",
    "AgentCoderConfig",
    "ExperimentConfig",
    # LLM Factory
    "create_llm_instance",
    "create_llm_from_params",
    "get_available_providers",
    "check_provider_availability",
    "validate_provider_model_combination",
]
__version__ = "0.1.0"
