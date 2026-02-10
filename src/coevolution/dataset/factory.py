"""Factory for creating dataset adapters by name."""

from .base import DatasetAdapter


def get_adapter(name: str) -> DatasetAdapter:
    """
    Get a dataset adapter instance by name.

    Args:
        name: The adapter name (e.g., 'lcb')

    Returns:
        An instance of the requested DatasetAdapter.

    Raises:
        ValueError: If the adapter name is not recognized.
    """
    if name == "lcb":
        from .lcb import LCBAdapter

        return LCBAdapter()

    raise ValueError(f"Unknown dataset adapter: '{name}'. Available adapters: ['lcb']")
