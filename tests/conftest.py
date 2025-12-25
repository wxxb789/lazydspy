"""Test configuration with lightweight stubs for optional dependencies."""

import sys
import types


def _install_rich_stub() -> None:
    """Register minimal rich stubs so imports succeed without the real package."""

    class _Console:
        def print(self, *args, **kwargs):
            return None

    class _Panel:
        def __init__(self, *args, **kwargs):
            return None

    class _Table:
        def __init__(self, *args, **kwargs):
            return None

        def add_column(self, *args, **kwargs):
            return None

        def add_row(self, *args, **kwargs):
            return None

    class _Markdown:
        def __init__(self, *args, **kwargs):
            return None

    rich = types.ModuleType("rich")
    rich_console = types.ModuleType("rich.console")
    rich_console.Console = _Console

    rich_panel = types.ModuleType("rich.panel")
    rich_panel.Panel = _Panel

    rich_table = types.ModuleType("rich.table")
    rich_table.Table = _Table

    rich_markdown = types.ModuleType("rich.markdown")
    rich_markdown.Markdown = _Markdown

    sys.modules.setdefault("rich", rich)
    sys.modules.setdefault("rich.console", rich_console)
    sys.modules.setdefault("rich.panel", rich_panel)
    sys.modules.setdefault("rich.table", rich_table)
    sys.modules.setdefault("rich.markdown", rich_markdown)


def _install_anthropic_stub() -> None:
    """Register an anthropic stub to avoid network-dependent imports."""

    class _ContentBlock:
        def __init__(self, block_type: str, text: str = "", **kwargs):
            self.type = block_type
            self.text = text
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _Messages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                content=[_ContentBlock("text", text="Mock response")],
                stop_reason="end_turn",
            )

    class _Anthropic:
        def __init__(self, *args, **kwargs):
            self.api_key = kwargs.get("api_key")
            self.base_url = kwargs.get("base_url")
            self.messages = _Messages()

    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = _Anthropic
    sys.modules.setdefault("anthropic", anthropic)


def _install_typer_stub() -> None:
    """Register a Typer stub when the real package is unavailable."""

    try:
        import typer as _real_typer  # noqa: F401
    except Exception:
        typer = types.ModuleType("typer")

        class _Exit(Exception):
            def __init__(self, code: int = 0):
                super().__init__()
                self.exit_code = code

        class _Context:
            def __init__(self):
                self.invoked_subcommand = None

        class _Typer:
            def __init__(self, *args, **kwargs):
                self._commands = {}

            def command(self, *args, **kwargs):
                def decorator(func):
                    name = kwargs.get("name") or getattr(func, "__name__", "command")
                    self._commands[name] = func
                    return func

                return decorator

            def callback(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def __call__(self, *args, **kwargs):
                return None

        def _option(default=None, *args, **kwargs):
            return default

        typer.Typer = _Typer
        typer.Exit = _Exit
        typer.Option = _option
        typer.Context = _Context
        sys.modules.setdefault("typer", typer)


_install_rich_stub()
_install_anthropic_stub()
_install_typer_stub()
