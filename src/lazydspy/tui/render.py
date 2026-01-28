"""Rendering helpers for the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown


@dataclass
class MarkdownRenderer:
    _cache: dict[tuple[str, int], str] = field(default_factory=dict)

    def render_markdown(self, text: str, width: int) -> str:
        key = (text, width)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        console = Console(width=width, record=True, force_terminal=True)
        console.print(Markdown(text))
        rendered = console.export_text(styles=True).rstrip()
        self._cache[key] = rendered
        return rendered

    def render_plain(self, text: str) -> str:
        return text

    def clear_cache(self) -> None:
        self._cache.clear()


__all__ = ["MarkdownRenderer"]
