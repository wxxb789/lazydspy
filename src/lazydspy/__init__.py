"""lazydspy - DSPy 优化脚本生成器.

An Agent-driven CLI tool that helps users create DSPy prompt
optimization scripts through interactive dialogue.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"


# Lazy imports to avoid circular dependencies
def __getattr__(name: str) -> Any:
    """Lazy import of submodules."""
    if name == "main":
        from lazydspy.cli import main

        return main
    if name == "Agent":
        from lazydspy.agent import Agent

        return Agent
    if name == "AgentConfig":
        from lazydspy.agent import AgentConfig

        return AgentConfig
    if name == "run_agent":
        from lazydspy.agent import run_agent

        return run_agent
    raise AttributeError(f"module 'lazydspy' has no attribute {name!r}")


__all__ = [
    "__version__",
    "main",
    "Agent",
    "AgentConfig",
    "run_agent",
]
