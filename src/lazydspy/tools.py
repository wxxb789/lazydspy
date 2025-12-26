"""MCP tools for lazydspy Agent.

Uses @tool decorator from claude-agent-sdk to define business tools.
Business logic is separated into _impl functions for easier testing.
"""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from lazydspy.knowledge.cost_models import estimate_optimization_cost, list_supported_models
from lazydspy.knowledge.optimizers import (
    get_optimizer_info,
    get_recommended_optimizer,
    list_all_optimizers,
)


def _make_text_response(text: str) -> dict[str, Any]:
    """Create a standard MCP tool response."""
    return {"content": [{"type": "text", "text": text}]}


# ============================================================================
# Business Logic (testable without MCP)
# ============================================================================


async def estimate_cost_impl(args: dict[str, Any]) -> dict[str, Any]:
    """Estimate the cost of running a DSPy optimization."""
    optimizer = args.get("optimizer", "gepa")
    mode = args.get("mode", "quick")
    dataset_size = args.get("dataset_size", 50)
    model = args.get("model", "claude-opus-4.5")

    result = estimate_optimization_cost(
        optimizer=optimizer,
        mode=mode,
        dataset_size=dataset_size,
        model=model,
    )

    return _make_text_response(json.dumps(result, ensure_ascii=False, indent=2))


async def list_optimizers_impl(args: dict[str, Any]) -> dict[str, Any]:
    """List all available DSPy optimizers."""
    optimizers = list_all_optimizers()
    models = list_supported_models()

    result = {
        "optimizers": optimizers,
        "supported_models": models,
    }

    return _make_text_response(json.dumps(result, ensure_ascii=False, indent=2))


async def get_defaults_impl(args: dict[str, Any]) -> dict[str, Any]:
    """Get default configuration for an optimizer."""
    scenario = args.get("scenario")
    optimizer_name = args.get("optimizer")
    mode = args.get("mode", "quick")

    # Recommend optimizer based on scenario if not specified
    if not optimizer_name and scenario:
        optimizer_name = get_recommended_optimizer(scenario)

    optimizer_name = optimizer_name or "gepa"
    optimizer_info = get_optimizer_info(optimizer_name)

    if not optimizer_info:
        return _make_text_response(f"未知的优化器: {optimizer_name}")

    defaults = optimizer_info.get_defaults(mode)

    result = {
        "optimizer": optimizer_name,
        "mode": mode,
        "hyperparameters": defaults,
        "description": optimizer_info.description,
        "cost_level": optimizer_info.cost_level,
    }

    if scenario:
        result["recommended_for_scenario"] = scenario

    return _make_text_response(json.dumps(result, ensure_ascii=False, indent=2))


# ============================================================================
# MCP Tool Wrappers
# ============================================================================


@tool(
    "estimate_cost",
    "估算 DSPy 优化的成本（API 调用费用）",
    {
        "optimizer": {"type": "string", "description": "优化器名称: gepa 或 miprov2"},
        "mode": {"type": "string", "description": "运行模式: quick 或 full"},
        "dataset_size": {"type": "integer", "description": "数据集大小（样本数）"},
        "model": {
            "type": "string",
            "description": "模型名称（可选，默认 claude-opus-4.5）",
        },
    },
)
async def estimate_cost(args: dict[str, Any]) -> dict[str, Any]:
    """Estimate the cost of running a DSPy optimization."""
    return await estimate_cost_impl(args)


@tool(
    "list_optimizers",
    "列出所有可用的 DSPy 优化器及其信息",
    {},
)
async def list_optimizers(args: dict[str, Any]) -> dict[str, Any]:
    """List all available DSPy optimizers."""
    return await list_optimizers_impl(args)


@tool(
    "get_defaults",
    "获取优化器的默认配置",
    {
        "optimizer": {
            "type": "string",
            "description": "优化器名称: gepa 或 miprov2（可选）",
        },
        "mode": {
            "type": "string",
            "description": "运行模式: quick 或 full（可选，默认 quick）",
        },
        "scenario": {
            "type": "string",
            "description": "场景类型（可选，用于推荐优化器）",
        },
    },
)
async def get_defaults(args: dict[str, Any]) -> dict[str, Any]:
    """Get default configuration for an optimizer."""
    return await get_defaults_impl(args)


# Tool names for allowed_tools configuration
TOOL_NAMES = [
    "mcp__lazydspy__estimate_cost",
    "mcp__lazydspy__list_optimizers",
    "mcp__lazydspy__get_defaults",
]


def create_mcp_server() -> Any:
    """Create an MCP server with all lazydspy tools."""
    return create_sdk_mcp_server(
        name="lazydspy",
        version="0.2.0",
        tools=[estimate_cost, list_optimizers, get_defaults],
    )


__all__ = [
    "create_mcp_server",
    "estimate_cost",
    "estimate_cost_impl",
    "get_defaults",
    "get_defaults_impl",
    "list_optimizers",
    "list_optimizers_impl",
    "TOOL_NAMES",
]
