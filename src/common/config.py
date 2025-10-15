"""
Base Configuration Classes for APR Experiments

This module provides reusable configuration classes and utilities for APR experiments.
It standardizes common configuration patterns across different experiment types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


@dataclass
class BaseConfig(ABC):
    """
    Abstract base configuration class for all APR experiments.

    Provides common configuration options that are shared across different
    experiment types while allowing for specialized configuration in subclasses.
    """

    # LLM Configuration
    llm_provider: Literal["ollama", "openai"] = "ollama"
    llm_model: str = "qwen2.5-coder:7b"
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal"

    # Dataset Configuration
    use_humaneval_subset: bool = False
    humaneval_subset_path: Optional[str] = None

    # Generation Configuration
    num_samples_per_task: int = 1

    # Output Configuration
    output_dir: str = "data/human_eval/generations"
    output_filename: Optional[str] = None

    # Debug Configuration
    verbose: bool = False

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        self.validate_config()
        self._setup_paths()

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate configuration parameters.

        Subclasses should implement this method to validate their specific
        configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        # Base validation
        if self.num_samples_per_task < 1:
            raise ValueError("num_samples_per_task must be >= 1")

        if self.llm_provider not in ["ollama", "openai"]:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _setup_paths(self) -> None:
        """Setup and normalize path configurations."""
        # Convert relative paths to absolute paths relative to project root
        project_root = self.get_project_root()

        # Setup output directory
        if not Path(self.output_dir).is_absolute():
            self.output_dir = str(project_root / self.output_dir)

        # Setup subset path if using subset
        if self.use_humaneval_subset and not self.humaneval_subset_path:
            self.humaneval_subset_path = str(
                project_root / "data" / "human_eval" / "HumanEval_20.jsonl.gz"
            )

    @staticmethod
    def get_project_root() -> Path:
        """
        Get the project root directory.

        Returns:
            Path to the project root directory
        """
        # Navigate up from src/common to find project root
        current = Path(__file__).parent.parent.parent
        return current

    def get_output_path(self, experiment_name: str, additional_suffix: str = "") -> str:
        """
        Generate standardized output file path.

        Args:
            experiment_name: Name of the experiment (e.g., "simple", "agent_coder")
            additional_suffix: Additional suffix to add to filename

        Returns:
            Absolute path to output file
        """
        if self.output_filename:
            # Use custom filename if provided
            custom_path = Path(self.output_filename)
            if custom_path.is_absolute():
                return str(custom_path)
            else:
                return str(self.get_project_root() / custom_path)

        # Generate standardized filename
        model_name = self.llm_model.replace(":", "_").replace("/", "_")
        dataset_suffix = self._get_dataset_suffix()
        samples_suffix = f"-samples{self.num_samples_per_task}"

        filename_parts = [model_name, dataset_suffix, samples_suffix]
        if additional_suffix:
            filename_parts.append(additional_suffix)

        filename = "".join(filename_parts) + ".jsonl"

        # Create output directory
        output_dir = Path(self.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        return str(output_dir / filename)

    def _get_dataset_suffix(self) -> str:
        """Get dataset suffix for filename."""
        if not self.use_humaneval_subset:
            return "-full"

        if self.humaneval_subset_path:
            # Extract subset size from path if available
            # Use .stem to get filename without extension, then split further if needed
            subset_filename = Path(self.humaneval_subset_path).stem
            # Handle .gz files by removing the .jsonl part as well
            if subset_filename.endswith(".jsonl"):
                subset_filename = subset_filename[:-6]  # Remove .jsonl

            if "HumanEval_" in subset_filename:
                subset_size = subset_filename.split("HumanEval_")[1]
                return f"-subset{subset_size}"

        return "-subset20"  # Default subset size

    def get_dataset_path(self) -> Optional[str]:
        """
        Get the path to the dataset based on configuration.

        Returns:
            Path to dataset file if using subset, None for full dataset
        """
        if self.use_humaneval_subset:
            return self.humaneval_subset_path
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            Configuration instance
        """
        # Filter out unknown keys
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


@dataclass
class SimpleConfig(BaseConfig):
    """Configuration for simple single-agent code generation."""

    # Simple generator specific settings
    prompt_template: str = "\n# Complete the function above, do not use any libraries. You are only allowed to write code inside the function defined above. Let's think step-by-step and then write the final code.\n"
    filename_suffix: str = ""

    def validate_config(self) -> None:
        """Validate SimpleConfig parameters."""
        super().validate_config()

        # Simple generator specific validation
        if not self.prompt_template:
            raise ValueError("prompt_template cannot be empty")


@dataclass
class AgentCoderConfig(BaseConfig):
    """Configuration for multi-agent code generation (AgentCoder)."""

    # AgentCoder specific settings
    max_iterations: int = 5
    save_per_iteration: bool = True

    # Separate LLM configurations for each agent
    programmer_llm_provider: Optional[Literal["ollama", "openai"]] = None
    programmer_llm_model: Optional[str] = None
    programmer_reasoning_effort: Optional[
        Literal["minimal", "low", "medium", "high"]
    ] = None
    tester_llm_provider: Optional[Literal["ollama", "openai"]] = None
    tester_llm_model: Optional[str] = None
    tester_reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = (
        None
    )

    def __post_init__(self) -> None:
        """Post-initialization validation and setup with agent-specific LLM defaults."""
        # Set agent-specific defaults if not provided
        if self.programmer_llm_provider is None:
            self.programmer_llm_provider = self.llm_provider
        if self.programmer_llm_model is None:
            self.programmer_llm_model = self.llm_model
        if self.programmer_reasoning_effort is None:
            self.programmer_reasoning_effort = self.reasoning_effort
        if self.tester_llm_provider is None:
            self.tester_llm_provider = self.llm_provider
        if self.tester_llm_model is None:
            self.tester_llm_model = self.llm_model
        if self.tester_reasoning_effort is None:
            self.tester_reasoning_effort = self.reasoning_effort

        # Call parent post-init
        super().__post_init__()

    def validate_config(self) -> None:
        """Validate AgentCoderConfig parameters."""
        super().validate_config()

        # AgentCoder specific validation
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        # Validate agent-specific LLM configurations
        if self.programmer_llm_provider not in ["ollama", "openai"]:
            raise ValueError(
                f"Unsupported programmer LLM provider: {self.programmer_llm_provider}"
            )
        if self.tester_llm_provider not in ["ollama", "openai"]:
            raise ValueError(
                f"Unsupported tester LLM provider: {self.tester_llm_provider}"
            )

        if not self.programmer_llm_model:
            raise ValueError("programmer_llm_model cannot be empty")
        if not self.tester_llm_model:
            raise ValueError("tester_llm_model cannot be empty")

    def get_iteration_output_path(
        self, iteration: int, experiment_name: str = "agent_coder"
    ) -> str:
        """
        Get output path for a specific iteration.

        Args:
            iteration: Iteration number
            experiment_name: Name of the experiment

        Returns:
            Path to iteration-specific output file
        """
        base_path = Path(self.get_output_path(experiment_name))
        iteration_filename = f"{base_path.stem}-iter{iteration}{base_path.suffix}"
        return str(base_path.parent / iteration_filename)


@dataclass
class ExperimentConfig(BaseConfig):
    """
    Generic configuration for custom experiments.

    This class can be used for experiments that need additional
    configuration parameters not covered by SimpleConfig or AgentCoderConfig.
    """

    # Generic experiment settings
    experiment_params: Dict[str, Any] = field(default_factory=dict)

    def validate_config(self) -> None:
        """Validate ExperimentConfig parameters."""
        super().validate_config()

        # Generic validation - can be extended by users
        pass

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get experiment parameter with default fallback.

        Args:
            key: Parameter key
            default: Default value if key not found

        Returns:
            Parameter value or default
        """
        return self.experiment_params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """
        Set experiment parameter.

        Args:
            key: Parameter key
            value: Parameter value
        """
        self.experiment_params[key] = value
