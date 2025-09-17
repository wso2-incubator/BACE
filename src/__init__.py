"""
APR (Automated Program Repair) package.

A multi-agent system for automated program repair and code generation.
"""

__version__ = "0.1.0"
__author__ = "Kaushitha Silva"
__email__ = "kaushithamsilva@gmail.com"

# Import main components for easy access
from .common import CodeProcessor, LLMClient

__all__ = [
    "CodeProcessor",
    "LLMClient",
    "__version__",
    "__author__",
    "__email__",
]
