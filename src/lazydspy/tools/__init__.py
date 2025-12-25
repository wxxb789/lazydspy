"""Tool registry and utilities for lazydspy Agent."""

from __future__ import annotations

from typing import Any, Callable, Coroutine

from .data_ops import DATA_OPS_TOOLS
from .domain_ops import DOMAIN_OPS_TOOLS
from .file_ops import FILE_OPS_TOOLS

# All available tools
ALL_TOOLS = FILE_OPS_TOOLS + DATA_OPS_TOOLS + DOMAIN_OPS_TOOLS

# Tool name to handler mapping
TOOL_HANDLERS: dict[str, Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]] = {
    tool["name"]: tool["handler"] for tool in ALL_TOOLS
}

# Tool name to schema mapping
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    tool["name"]: {
        "name": tool["name"],
        "description": tool["description"],
        "input_schema": tool["input_schema"],
    }
    for tool in ALL_TOOLS
}


def get_tool_handler(name: str) -> Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]] | None:
    """Get the handler function for a tool by name."""
    return TOOL_HANDLERS.get(name)


def get_tool_schema(name: str) -> dict[str, Any] | None:
    """Get the schema for a tool by name."""
    return TOOL_SCHEMAS.get(name)


def list_tool_names() -> list[str]:
    """List all available tool names."""
    return list(TOOL_HANDLERS.keys())


def get_all_tool_schemas() -> list[dict[str, Any]]:
    """Get schemas for all available tools."""
    return list(TOOL_SCHEMAS.values())


__all__ = [
    "ALL_TOOLS",
    "TOOL_HANDLERS",
    "TOOL_SCHEMAS",
    "get_tool_handler",
    "get_tool_schema",
    "list_tool_names",
    "get_all_tool_schemas",
    # Re-export individual tool modules
    "FILE_OPS_TOOLS",
    "DATA_OPS_TOOLS",
    "DOMAIN_OPS_TOOLS",
]
