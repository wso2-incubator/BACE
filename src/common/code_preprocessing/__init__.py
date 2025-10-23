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

Usage (Hierarchical Imports):
    from src.common.code_preprocessing.parsers import extract_code_block_from_response
    from src.common.code_preprocessing.analyzers import parse_code_structure
    from src.common.code_preprocessing.transformers import extract_function_with_helpers
    from src.common.code_preprocessing.builders import build_test_script_for_humaneval
    from src.common.code_preprocessing import CodeParsingError  # Exceptions available directly
"""

# Only export exceptions for convenience - they're used across many modules
from .exceptions import CodeParsingError, CodeProcessingError, CodeTransformationError

__all__ = [
    "CodeProcessingError",
    "CodeParsingError",
    "CodeTransformationError",
]
