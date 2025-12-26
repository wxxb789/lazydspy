"""Test configuration for lazydspy.

All dependencies (rich, typer, claude-agent-sdk) are installed via uv sync,
so no stubs are needed.
"""

from __future__ import annotations

import asyncio
from typing import Any


def run_async(coro: Any) -> Any:
    """Helper to run async functions in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
