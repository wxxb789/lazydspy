"""Lightweight Console stub."""

from __future__ import annotations

from typing import Any


class Console:
    def print(self, *objects: Any, **kwargs: Any) -> None:  # noqa: D401 - mimic rich API
        """Print to stdout."""

        # Simple passthrough to builtin print to keep behavior observable.
        print(*objects)
