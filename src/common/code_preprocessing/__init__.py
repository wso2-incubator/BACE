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

Usage Patterns:

1. Direct submodule access (recommended):
    import common.code_preprocessing as cpp
    code = cpp.parsers.extract_code_block_from_response(response)
    structure = cpp.analyzers.parse_code_structure(code)
    script = cpp.builders.build_test_script_for_humaneval(code, tests)

2. Hierarchical imports:
    from common.code_preprocessing.parsers import extract_code_block_from_response
    from common.code_preprocessing.analyzers import parse_code_structure
    from common.code_preprocessing.builders import build_test_script_for_humaneval

3. Exception handling:
    from common.code_preprocessing import CodeParsingError, CodeProcessingError
"""

# Import submodules to make them available as attributes
from . import analyzers, builders, parsers, transformers

# Export exceptions for convenience - they're used across many modules
from .exceptions import CodeParsingError, CodeProcessingError, CodeTransformationError

__all__ = [
    # Submodules
    "parsers",
    "analyzers",
    "transformers",
    "builders",
    # Exceptions
    "CodeProcessingError",
    "CodeParsingError",
    "CodeTransformationError",
]
