"""
Common utilities shared across all APR projects.

This module provides shared functionality including:
- CodeProcessor: For processing and extracting code segments
- LLMClient: Unified interface for different LLM providers
- SafeCodeSandbox: Safe execution environment for generated code
- Enhanced typing: TestExecutionResult, BasicExecutionResult, TestDetails for type safety
- Configuration: BaseConfig and specialized config classes for experiments
- LLM Factory: Factory functions for creating LLM instances
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
from .config import (
    BaseConfig,
    SimpleConfig,
    AgentCoderConfig,
    ExperimentConfig
)
from .llm_factory import (
    create_llm_instance,
    create_llm_from_params,
    get_available_providers,
    check_provider_availability,
    validate_provider_model_combination
)

__all__ = [
    # Core utilities
    'CodeProcessor',
    'LLMClient',
    'SafeCodeSandbox',
    'create_safe_test_environment',
    'check_test_execution_status',
    # Types
    'TestExecutionResult',
    'BasicExecutionResult',
    'TestDetails',
    'TestAnalysis',
    'ExecutionCategory',
    'CodeExecutionError',
    'CodeExecutionTimeoutError',
    # Configuration
    'BaseConfig',
    'SimpleConfig',
    'AgentCoderConfig',
    'ExperimentConfig',
    # LLM Factory
    'create_llm_instance',
    'create_llm_from_params',
    'get_available_providers',
    'check_provider_availability',
    'validate_provider_model_combination'
]
__version__ = "0.1.0"
