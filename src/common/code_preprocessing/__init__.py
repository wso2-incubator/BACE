"""
Code Preprocessing Module
=========================

This module provides utilities for preprocessing Python code in automated testing workflows.
It handles LLM response parsing, AST-based code analysis, code transformation, and test script assembly.

Main components:
- parsers: Extract code from LLM responses
- analyzers: Analyze code structure using AST
- transformers: Extract and modify code segments
- builders: Assemble complete test scripts

Usage:
    from src.common.code_preprocessing import (
        extract_code_block_from_response,
        parse_code_structure,
        extract_function_with_helpers,
        build_test_script_for_humaneval,
    )
"""

# Analyzers
from .analyzers import (
    CodeStructure,
    extract_test_case_names,
    extract_test_methods_code,
    parse_code_structure,
)

# Builders
from .builders import build_test_script_for_humaneval, build_test_script_for_lcb

# Exceptions
from .exceptions import CodeParsingError, CodeProcessingError, CodeTransformationError

# Parsers
from .parsers import (
    extract_all_code_blocks_from_response,
    extract_code_block_from_response,
    extract_function_name_from_problem,
)

# Transformers
from .transformers import (
    extract_class_block,
    extract_function_with_helpers,
    replace_test_methods,
)

__all__ = [
    # Exceptions
    "CodeProcessingError",
    "CodeParsingError",
    "CodeTransformationError",
    # Parsers
    "extract_function_name_from_problem",
    "extract_all_code_blocks_from_response",
    "extract_code_block_from_response",
    # Analyzers
    "CodeStructure",
    "parse_code_structure",
    "extract_test_case_names",
    "extract_test_methods_code",
    # Transformers
    "extract_function_with_helpers",
    "extract_class_block",
    "replace_test_methods",
    # Builders
    "build_test_script_for_humaneval",
    "build_test_script_for_lcb",
]
