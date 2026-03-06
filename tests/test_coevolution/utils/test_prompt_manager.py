from pathlib import Path

from coevolution.utils.prompt_manager import PromptManager


def test_prompt_manager_render(tmp_path: Path) -> None:
    # Create a dummy template
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "test.j2"
    template_file.write_text("Hello {{ name }}!")

    manager = PromptManager(template_dir=str(template_dir))
    result = manager.render_prompt("test.j2", name="World")
    assert result == "Hello World!"


def test_prompt_manager_include(tmp_path: Path) -> None:
    # Create dummy templates
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "common").mkdir()

    base_file = template_dir / "common" / "base.j2"
    base_file.write_text("{% macro greet(name) %}Hello {{ name }}!{% endmacro %}")

    child_file = template_dir / "child.j2"
    child_file.write_text("{% import 'common/base.j2' as base %}{{ base.greet(name) }}")

    manager = PromptManager(template_dir=str(template_dir))
    result = manager.render_prompt("child.j2", name="Jinja")
    assert result == "Hello Jinja!"
