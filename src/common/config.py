"""
Configuration management for the APR project.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os


@dataclass
class ProjectConfig:
    """Base configuration class for all projects."""
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5-coder:7b"
    num_samples_per_task: int = 1
    debug: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'num_samples_per_task': self.num_samples_per_task,
            'debug': self.debug
        }


@dataclass
class AgentCoderConfig(ProjectConfig):
    """Configuration specific to AgentCoder+ project."""
    memory_limit: int = 3
    parallel_agents: bool = True
    session_timeout: int = 3600  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'memory_limit': self.memory_limit,
            'parallel_agents': self.parallel_agents,
            'session_timeout': self.session_timeout
        })
        return base_dict
