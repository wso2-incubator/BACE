"""
Elite selection strategies for coevolutionary algorithms.

This package implements the IEliteSelectionStrategy protocol, providing
selection mechanisms for choosing elite individuals to preserve unchanged
into the next generation.

The selection strategies work with Individual objects and can utilize the
full CoevolutionContext including interaction matrices for sophisticated
selection criteria.
"""

from .code_diversity_selector import CodeDiversityEliteSelector
from .test_diversity_selector import TestDiversityEliteSelector
from .top_k_selector import TopKEliteSelector

__all__ = [
    "TopKEliteSelector",
    "TestDiversityEliteSelector",
    "CodeDiversityEliteSelector",
]
