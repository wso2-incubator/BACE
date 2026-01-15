"""
Configuration loader for YAML-based experiment configuration.

Provides utilities to load, merge, and validate modular YAML configuration files
for coevolution experiments. Supports:
- Hierarchical config composition (base + overrides)
- Reference resolution for modular configs (e.g., referencing llm configs)
- Environment variable substitution
- Type validation using dataclasses

Example:
    >>> config = load_experiment_config("configs/experiments/production.yaml")
    >>> llm_config = config["llm"]
    >>> code_profile_config = config["code_profile"]
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


def _resolve_env_vars(value: Any) -> Any:
    """
    Recursively resolve environment variables in config values.

    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.

    Args:
        value: Config value (can be string, dict, list, etc.)

    Returns:
        Value with environment variables substituted
    """
    if isinstance(value, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = r"\$\{([^}:]+)(?::-(.[^}]*))?\}"

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.environ.get(var_name, default_value)

        return re.sub(pattern, replacer, value)

    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]

    return value


def _load_yaml_file(file_path: Path) -> dict[str, Any]:
    """
    Load a YAML file and resolve environment variables.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary

    Raises:
        ConfigError: If file cannot be loaded or parsed
    """
    try:
        with open(file_path, "r") as f:
            content = yaml.safe_load(f)

        if content is None:
            return {}

        # Resolve environment variables
        return _resolve_env_vars(content)

    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file {file_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Failed to load config {file_path}: {e}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = _deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def _resolve_config_references(
    config: dict[str, Any], base_dir: Path
) -> dict[str, Any]:
    """
    Resolve references to other config files.

    References are string values ending in .yaml that point to other config files.
    Example: "llm: llm/gpt-4.yaml" loads and merges configs/llm/gpt-4.yaml

    Args:
        config: Configuration dictionary
        base_dir: Base directory for resolving relative paths (usually configs/)

    Returns:
        Configuration with references resolved
    """
    result = {}

    for key, value in config.items():
        if isinstance(value, str) and value.endswith(".yaml"):
            # This is a reference to another config file
            referenced_path = base_dir / value

            if not referenced_path.exists():
                raise ConfigError(
                    f"Referenced config file not found: {referenced_path} (key: {key})"
                )

            logger.debug(f"Loading referenced config: {key} -> {referenced_path}")
            result[key] = _load_yaml_file(referenced_path)

        elif isinstance(value, dict):
            # Recursively resolve nested dicts
            result[key] = _resolve_config_references(value, base_dir)

        elif isinstance(value, list):
            # Handle lists (in case they contain dicts with references)
            result[key] = [
                _resolve_config_references(item, base_dir)
                if isinstance(item, dict)
                else item
                for item in value
            ]

        else:
            result[key] = value

    return result


def load_experiment_config(
    config_path: str | Path, base_config_path: str | Path | None = None
) -> dict[str, Any]:
    """
    Load an experiment configuration file with support for:
    - Base config inheritance
    - Referenced config files (e.g., llm/gpt-4.yaml)
    - Environment variable substitution

    Args:
        config_path: Path to experiment config file
        base_config_path: Optional base config to merge with (defaults to configs/base.yaml if exists)

    Returns:
        Complete configuration dictionary

    Raises:
        ConfigError: If configuration cannot be loaded or is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Experiment config file not found: {config_path}")

    # Determine base directory for resolving references (usually configs/)
    base_dir = (
        config_path.parent.parent
        if config_path.parent.name == "experiments"
        else config_path.parent
    )

    logger.info(f"Loading experiment config: {config_path}")

    # Load base config if it exists
    if base_config_path is None:
        base_config_path = base_dir / "base.yaml"

    config = {}
    if Path(base_config_path).exists():
        logger.info(f"Loading base config: {base_config_path}")
        config = _load_yaml_file(Path(base_config_path))

    # Load experiment config and merge
    experiment_config = _load_yaml_file(config_path)
    config = _deep_merge(config, experiment_config)

    # Resolve references to other config files
    config = _resolve_config_references(config, base_dir)

    logger.info("Configuration loaded successfully")
    return config


def apply_cli_overrides(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """
    Apply CLI overrides to configuration.

    Handles nested keys using dot notation (e.g., "code_profile.initial_population_size").

    Args:
        config: Base configuration
        overrides: Dictionary of override values

    Returns:
        Configuration with overrides applied
    """
    result = config.copy()

    for key, value in overrides.items():
        if value is None:
            # Skip None values (unset CLI options)
            continue

        # Support dot notation for nested keys
        if "." in key:
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            result[key] = value

    return result


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.

    Args:
        config: Configuration to validate

    Raises:
        ConfigError: If configuration is invalid
    """
    # Check for required top-level keys
    required_keys = ["llm"]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigError(f"Missing required config keys: {missing_keys}")

    # Validate LLM config
    if "llm" in config:
        llm_config = config["llm"]
        required_llm_keys = ["provider", "model"]
        missing_llm = [key for key in required_llm_keys if key not in llm_config]
        if missing_llm:
            raise ConfigError(f"Missing required LLM config keys: {missing_llm}")

    logger.debug("Configuration validation passed")


def get_config_value(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., "llm.model")
        default: Default value if key not found

    Returns:
        Config value or default
    """
    keys = key_path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


__all__ = [
    "load_experiment_config",
    "apply_cli_overrides",
    "validate_config",
    "get_config_value",
    "ConfigError",
]
