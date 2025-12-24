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

    rich = types.ModuleType("rich")
    rich_console = types.ModuleType("rich.console")
    rich_console.Console = _Console

    rich_panel = types.ModuleType("rich.panel")
    rich_panel.Panel = _Panel

    rich_table = types.ModuleType("rich.table")
    rich_table.Table = _Table

    sys.modules.setdefault("rich", rich)
    sys.modules.setdefault("rich.console", rich_console)
    sys.modules.setdefault("rich.panel", rich_panel)
    sys.modules.setdefault("rich.table", rich_table)


def _install_anthropic_stub() -> None:
    """Register an Anthropic stub to avoid network-dependent imports."""

    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = object  # type: ignore[assignment]
    sys.modules.setdefault("anthropic", anthropic)


_install_rich_stub()
_install_anthropic_stub()
