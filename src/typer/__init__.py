"""Minimal Typer stub for offline execution and testing."""

from __future__ import annotations

from typing import Any, Callable, Iterable


class Exit(SystemExit):
    """Mimic typer.Exit for early termination."""

    def __init__(self, code: int = 0) -> None:
        super().__init__(code)
        self.exit_code = code


def Option(default: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: D401 - mimic typer API
    """Return the default value (no CLI parsing)."""

    return default


class Typer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - mimic typer API
        self._commands: list[Callable[..., Any]] = []

    def command(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._commands.append(func)
            return func

        return decorator

    def __call__(self, args: Iterable[str] | None = None, standalone_mode: bool = True) -> None:
        # Simple dispatcher: run the first registered command if present.
        if self._commands:
            self._commands[0]()
