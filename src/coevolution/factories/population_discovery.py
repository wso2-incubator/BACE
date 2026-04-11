"""Population discovery service for dynamic profile construction."""

import inspect

from typing import Any, Dict

from loguru import logger

from ..core.interfaces import IExecutionSystem
from ..core.interfaces.language import ILanguage
from ..populations import registry
from ..populations.registry import PopulationRegistry
from infrastructure.llm_client import LLMClient
from infrastructure.sandbox.types import SandboxConfig


class PopulationDiscoveryService:
    """Service to dynamically discover and construct population profiles."""

    def __init__(
        self,
        llm_client: LLMClient,
        language_adapter: ILanguage,
        execution_system: IExecutionSystem,
        sandbox_config: SandboxConfig,
        cpu_workers: int,
    ) -> None:
        self.llm_client = llm_client
        self.language_adapter = language_adapter
        self.execution_system = execution_system
        self.sandbox_config = sandbox_config
        self.cpu_workers = cpu_workers
        self.registry: PopulationRegistry = registry

    def construct_all(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover and construct all profiles defined in the config.

        Returns:
            Dict containing:
                - code_profile: CodeProfile
                - evolved_test_profiles: Dict[str, TestProfile]
                - public_test_profile: Optional[PublicTestProfile]
        """
        results: Dict[str, Any] = {
            "code_profile": None,
            "evolved_test_profiles": {},
            "public_test_profile": None,
        }

        # 1. Code Profile
        code_cfg = experiment_config.get("code_profile")
        if code_cfg:
            # Determine profile type (e.g., "default", "agent_coder")
            # For now, default to "default" unless specified
            code_type = experiment_config.get("code_profile_type", "default")
            results["code_profile"] = self._construct_profile(
                self.registry.get_code_factory(code_type), code_cfg
            )
            logger.info(f"Constructed code profile of type '{code_type}'")

        # 2. Public Profile
        public_cfg = experiment_config.get("public_profile")
        if public_cfg:
            results["public_test_profile"] = self._construct_profile(
                self.registry.get_public_factory("public"), public_cfg
            )
            logger.info("Constructed public test profile")

        # 3. Evolved Test Populations
        # We iterate through all registered test populations and see if they are in the config
        for test_type in self.registry.registered_test_populations:
            profile_key = f"{test_type}_profile"
            test_cfg = experiment_config.get(profile_key)
            if test_cfg:
                factory = self.registry.get_test_factory(test_type)
                results["evolved_test_profiles"][test_type] = self._construct_profile(
                    factory, test_cfg
                )
                logger.info(f"Constructed evolved test profile for '{test_type}'")

        return results

    def _construct_profile(self, factory: Any, config: Dict[str, Any]) -> Any:
        """Call a factory function with arguments from config + common dependencies."""
        sig = inspect.signature(factory)
        kwargs = {}

        # Common dependencies that factories might need
        common_deps = {
            "llm_client": self.llm_client,
            "language_adapter": self.language_adapter,
            "execution_system": self.execution_system,
            "sandbox_config": self.sandbox_config,
            "cpu_workers": self.cpu_workers,
        }

        for param_name, param in sig.parameters.items():
            if param_name in config:
                kwargs[param_name] = config[param_name]
            elif param_name in common_deps:
                kwargs[param_name] = common_deps[param_name]
            elif param.default is not inspect.Parameter.empty:
                # Use default value if it exists
                continue
            else:
                # Parameter is required but not in config or common deps
                logger.warning(
                    f"Parameter '{param_name}' required by {factory.__name__} "
                    f"not found in config or common dependencies."
                )

        return factory(**kwargs)
