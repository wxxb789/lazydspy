"""Prompt_toolkit completers for slash commands and file references."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

DEFAULT_SLASH_COMMANDS = ["help", "clear", "export", "exit", "status", "reset"]
IGNORE_DIRS = {".git", ".venv", "__pycache__", "generated", "dist", "build"}


@dataclass
class SlashCommandCompleter(Completer):
    commands: list[str] = field(default_factory=lambda: list(DEFAULT_SLASH_COMMANDS))

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.get_word_before_cursor(WORD=True)
        if not text.startswith("/"):
            return
        prefix = text[1:]
        for command in self.commands:
            if command.startswith(prefix):
                yield Completion(f"/{command}", start_position=-len(text))


@dataclass
class FileReferenceCompleter(Completer):
    workspace_root: Path

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.get_word_before_cursor(WORD=True)
        if not text.startswith("@"):
            return

        path_text = text[1:]
        base_dir, prefix = self._resolve_base(path_text)
        if not base_dir.exists() or not base_dir.is_dir():
            return

        for entry in sorted(base_dir.iterdir(), key=lambda item: item.name.lower()):
            if entry.name in IGNORE_DIRS:
                continue
            if not entry.name.lower().startswith(prefix.lower()):
                continue
            rel = entry.relative_to(self.workspace_root)
            suggestion = rel.as_posix()
            if entry.is_dir():
                suggestion = suggestion.rstrip("/") + "/"
            yield Completion(f"@{suggestion}", start_position=-len(text))

    def _resolve_base(self, path_text: str) -> tuple[Path, str]:
        if not path_text:
            return self.workspace_root, ""
        candidate = Path(path_text)
        parent = candidate.parent
        prefix = candidate.name
        if candidate.is_absolute():
            base_dir = parent
        else:
            base_dir = self.workspace_root / parent
        return base_dir, prefix


@dataclass
class CombinedCompleter(Completer):
    completers: list[Completer]

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        for completer in self.completers:
            yield from completer.get_completions(document, complete_event)


__all__ = [
    "CombinedCompleter",
    "FileReferenceCompleter",
    "SlashCommandCompleter",
]
