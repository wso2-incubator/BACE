import os
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape


class PromptManager:
    """
    Manages loading and rendering of Jinja2 prompt templates.
    """

    def __init__(
        self, template_dir: Optional[str] = None, language: str = "python"
    ) -> None:
        if template_dir is None:
            # Default to src/coevolution/prompts relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_dir = os.path.join(base_dir, "prompts")

        self.language = language
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_prompt(self, template_path: str, **kwargs: Any) -> str:
        """
        Renders a template with the given context.

        Args:
            template_path: Path to the template file relative to the template directory.
            **kwargs: Context variables for the template.

        Returns:
            The rendered prompt string.
        """
        # Inject language into context if not already present
        if "language" not in kwargs:
            kwargs["language"] = self.language

        template = self.env.get_template(template_path)
        return template.render(**kwargs)


# Global instance for easy access if needed, though dependency injection is preferred
_default_manager: Optional[PromptManager] = None


def get_prompt_manager(language: str = "python") -> PromptManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = PromptManager(language=language)
    return _default_manager


__all__ = ["PromptManager", "get_prompt_manager"]
