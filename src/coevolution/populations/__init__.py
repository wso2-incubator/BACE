"""Populations package — one subpackage per population type.

Adding a new population:
1. Create populations/<name>/ with operators/ and profile.py
2. Import the factory in factories/__init__.py
"""

from . import agent_coder, code, differential, property, unittest
from .registry import registry

__all__ = ["code", "unittest", "differential", "agent_coder", "property", "registry"]
