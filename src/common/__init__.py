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
"""

from .code_preprocessing import (
    CodeParsingError,
    CodeProcessingError,
    CodeStructure,
    CodeTransformationError,
    build_test_script_for_humaneval,
    build_test_script_for_lcb,
    extract_all_code_blocks,
    extract_class_block,
    extract_code_block_from_response,
    extract_function_name_from_problem,
    extract_function_with_helpers,
    extract_test_case_names,
    extract_test_methods_code,
    parse_code_structure,
    replace_test_methods,
)
from .coevolution import (
    CoevolutionConfig,
    SelectionStrategy,
    initialize_populations,
    run_evaluation,
    update_population_beliefs,
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
    # Code Preprocessing
    "CodeProcessingError",
    "CodeParsingError",
    "CodeTransformationError",
    "CodeStructure",
    "extract_function_name_from_problem",
    "extract_all_code_blocks",
    "extract_code_block_from_response",
    "parse_code_structure",
    "extract_test_case_names",
    "extract_test_methods_code",
    "extract_function_with_helpers",
    "extract_class_block",
    "replace_test_methods",
    "build_test_script_for_humaneval",
    "build_test_script_for_lcb",
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
    # Coevolution
    "CoevolutionConfig",
    "SelectionStrategy",
    "initialize_populations",
    "run_evaluation",
    "update_population_beliefs",
]
__version__ = "0.1.0"
