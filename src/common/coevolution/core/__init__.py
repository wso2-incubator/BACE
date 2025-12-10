"""
Core module for coevolution.

This module contains the fundamental abstractions and implementations
for individuals and populations in the coevolution framework.
"""

from . import individual, interfaces, population

__all__ = [
    "individual",
    "interfaces",
    "population",
]
