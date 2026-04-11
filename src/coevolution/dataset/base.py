"""Base dataset adapter interface for loading problem datasets."""

from abc import ABC, abstractmethod
from typing import Any

from ..core.interfaces import Problem


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.

    Each adapter is responsible for loading problems from a specific dataset source
    and converting them into the standard Problem format.
    """

    @abstractmethod
    def load_dataset(self, config: dict[str, Any]) -> list[Problem]:
        """
        Load problems from the dataset based on the provided configuration.

        Args:
            config: Dataset-specific configuration dictionary containing parameters
                   like version, difficulty filters, date ranges, cache paths, etc.

        Returns:
            List of Problem instances loaded from the dataset.

        Raises:
            ValueError: If configuration is invalid or required parameters are missing.
            FileNotFoundError: If dataset files cannot be found.
        """
        pass
