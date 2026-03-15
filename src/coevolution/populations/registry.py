"""Central registry for population profile factories.

Allows dynamic discovery and construction of populations based on YAML configuration.
"""

from typing import Callable, Dict, TypeVar

from ..core.interfaces import CodeProfile, PublicTestProfile, TestProfile

T = TypeVar("T", CodeProfile, TestProfile, PublicTestProfile)
FactoryFunc = Callable[..., T]


class PopulationRegistry:
    """Registry for population profile factories."""

    def __init__(self) -> None:
        self._code_factories: Dict[str, FactoryFunc[CodeProfile]] = {}
        self._test_factories: Dict[str, FactoryFunc[TestProfile]] = {}
        self._public_factories: Dict[str, FactoryFunc[PublicTestProfile]] = {}

    def register_code_factory(
        self, name: str, factory: FactoryFunc[CodeProfile]
    ) -> None:
        """Register a factory for a code population profile."""
        self._code_factories[name] = factory

    def register_test_factory(
        self, name: str, factory: FactoryFunc[TestProfile]
    ) -> None:
        """Register a factory for an evolved test population profile."""
        self._test_factories[name] = factory

    def register_public_factory(
        self, name: str, factory: FactoryFunc[PublicTestProfile]
    ) -> None:
        """Register a factory for a public test population profile."""
        self._public_factories[name] = factory

    def get_code_factory(self, name: str) -> FactoryFunc[CodeProfile]:
        """Get the factory for a specific code population profile."""
        if name not in self._code_factories:
            raise KeyError(f"No code factory registered for '{name}'")
        return self._code_factories[name]

    def get_test_factory(self, name: str) -> FactoryFunc[TestProfile]:
        """Get the factory for a specific evolved test population profile."""
        if name not in self._test_factories:
            raise KeyError(f"No test factory registered for '{name}'")
        return self._test_factories[name]

    def get_public_factory(self, name: str) -> FactoryFunc[PublicTestProfile]:
        """Get the factory for a specific public test population profile."""
        if name not in self._public_factories:
            raise KeyError(f"No public factory registered for '{name}'")
        return self._public_factories[name]

    @property
    def registered_code_populations(self) -> list[str]:
        return list(self._code_factories.keys())

    @property
    def registered_test_populations(self) -> list[str]:
        return list(self._test_factories.keys())

    @property
    def registered_public_populations(self) -> list[str]:
        return list(self._public_factories.keys())

    # --- Decorators ---

    def code_factory(
        self, name: str
    ) -> Callable[[Callable[..., CodeProfile]], Callable[..., CodeProfile]]:
        """Decorator to register a code profile factory."""

        def decorator(func: Callable[..., CodeProfile]) -> Callable[..., CodeProfile]:
            self.register_code_factory(name, func)
            return func

        return decorator

    def test_factory(
        self, name: str
    ) -> Callable[[Callable[..., TestProfile]], Callable[..., TestProfile]]:
        """Decorator to register a test profile factory."""

        def decorator(func: Callable[..., TestProfile]) -> Callable[..., TestProfile]:
            self.register_test_factory(name, func)
            return func

        return decorator

    def public_factory(
        self, name: str
    ) -> Callable[[Callable[..., PublicTestProfile]], Callable[..., PublicTestProfile]]:
        """Decorator to register a public test profile factory."""

        def decorator(
            func: Callable[..., PublicTestProfile],
        ) -> Callable[..., PublicTestProfile]:
            self.register_public_factory(name, func)
            return func

        return decorator


# Global registry instance
registry = PopulationRegistry()
