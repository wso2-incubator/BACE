"""Dataset layer: External dataset connectors.

This layer contains adapters that connect to external dataset sources:
- lcb: LiveCodeBench dataset adapter

Use the factory.get_adapter() function to obtain adapter instances.
"""

from .base import DatasetAdapter
from .factory import get_adapter

__all__ = ["DatasetAdapter", "get_adapter", "lcb"]
