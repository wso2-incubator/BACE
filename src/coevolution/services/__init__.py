"""Services layer: Core engine mechanisms.

This layer contains the singleton components that perform the heavy lifting:
- bayesian: Pure mathematical posterior update logic
- execution: Code execution and sandbox management
- ledger: State tracking and history management
"""

__all__ = ["bayesian", "execution", "ledger"]
