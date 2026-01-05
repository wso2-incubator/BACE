"""Strategies layer: Pluggable policies.

This layer contains interchangeable behavioral components:
- operators: Crossover and mutation operators
- breeding: Population breeding strategies
- selection: Parent/survivor selection strategies
- probability: Probability assignment policies
"""

__all__ = ["operators", "breeding", "selection", "probability"]
