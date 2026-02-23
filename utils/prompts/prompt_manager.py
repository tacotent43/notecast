from __future__ import annotations

import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptManager:
    """
    Self-written class for Jinja2 templates especially for summarization.

    Supports:
    - md
    - latex

    Usage examples:
    ```
    from prompts.prompt_manager import PromptManager

    pm = PromptManager("notecast/prompts")

    markdown_text = pm.render(
        "markdown_default_prompt.md.j2",
        language="Russian",
        title="Calculus - Lecture 1",
        transcription=raw_transcription_text
    )

    latex_text = pm.render(
        "latex_default_prompt.j2",
        language="Russian",
        title="Calculus - Lecture 1",
        transcription=raw_transcription_text
    )
    ```
    """

    def __init__(self, templates_dir: str) -> None:
        if not os.path.isdir(templates_dir):
            raise NotADirectoryError(f"Templates directory not found: {templates_dir}")

        self.templates_dir = templates_dir

        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined
        )

    def list_templates(self) -> list[str]:
        templates = []
        for file in os.listdir(self.templates_dir):
            if file.endswith(".j2"):
                templates.append(file)
        return templates

    def load(self, template_name: str):
        try:
            return self.env.get_template(template_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load template '{template_name}': {e}")

    def render(
        self,
        template_name: str,
        *,
        language: str,
        title: str,
        transcription: str
    ) -> str:
        template = self.load(template_name)

        return template.render(
            language=language,
            title=title,
            transcription=transcription
        )
