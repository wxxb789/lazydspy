"""Domain operation tools for lazydspy Agent."""

from __future__ import annotations

import json
from typing import Any

from lazydspy.knowledge.cost_models import estimate_optimization_cost
from lazydspy.knowledge.optimizers import (
    OPTIMIZER_REGISTRY,
    get_optimizer_info,
    list_all_optimizers,
)


async def estimate_cost(args: dict[str, Any]) -> dict[str, Any]:
    """Estimate the cost of running an optimization.

    Args:
        optimizer: Optimizer name (gepa/miprov2)
        mode: Run mode (quick/full)
        dataset_size: Number of examples in the dataset
        model: Model to use (optional)

    Returns:
        Tool response with cost estimation
    """
    try:
        optimizer = args["optimizer"]
        mode = args["mode"]
        dataset_size = args["dataset_size"]
        model = args.get("model", "claude-opus-4.5")

        result = estimate_optimization_cost(
            optimizer=optimizer,
            mode=mode,
            dataset_size=dataset_size,
            model=model,
        )

        # Format the result nicely
        output = f"""成本估算结果

配置:
- 优化器: {result['optimizer']}
- 模式: {result['mode']}
- 模型: {result['model']}
- 数据集大小: {result['dataset_size']} 条

估算:
- API 调用次数: ~{result['estimated_calls']}
- 输入 tokens: ~{result['estimated_input_tokens']:,}
- 输出 tokens: ~{result['estimated_output_tokens']:,}

预估成本: ${result['estimated_cost_usd']:.4f} USD
- 输入成本: ${result['cost_breakdown']['input_cost_usd']:.4f}
- 输出成本: ${result['cost_breakdown']['output_cost_usd']:.4f}

建议: {result['cost_hint']}"""

        return {
            "content": [
                {
                    "type": "text",
                    "text": output,
                }
            ]
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"成本估算失败: {e}",
                }
            ]
        }


async def list_optimizers(args: dict[str, Any]) -> dict[str, Any]:
    """List all available DSPy optimizers.

    Returns:
        Tool response with optimizer list
    """
    try:
        optimizers = list_all_optimizers()

        lines = ["可用的 DSPy 优化器:\n"]
        for opt in optimizers:
            lines.append(f"## {opt['name']} ({opt['key']})")
            lines.append(f"描述: {opt['description']}")
            lines.append(f"推荐场景: {', '.join(opt['recommended_for'])}")
            lines.append(f"成本级别: {opt['cost_level']}")
            lines.append("")

        return {
            "content": [
                {
                    "type": "text",
                    "text": "\n".join(lines),
                }
            ]
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"获取优化器列表失败: {e}",
                }
            ]
        }


async def get_defaults(args: dict[str, Any]) -> dict[str, Any]:
    """Get default configuration for an optimizer.

    Args:
        optimizer: Optimizer name (gepa/miprov2)
        mode: Run mode (quick/full)

    Returns:
        Tool response with default configuration
    """
    try:
        optimizer = args["optimizer"]
        mode = args["mode"]

        info = get_optimizer_info(optimizer)
        if not info:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"未知优化器: {optimizer}。可用选项: gepa, miprov2",
                    }
                ]
            }

        defaults = info.get_defaults(mode)

        output = f"""{info.name} 在 {mode} 模式下的默认配置:

{json.dumps(defaults, ensure_ascii=False, indent=2)}

说明:
- {info.description}
- 推荐场景: {', '.join(info.recommended_for)}
- 成本级别: {info.cost_level}"""

        return {
            "content": [
                {
                    "type": "text",
                    "text": output,
                }
            ]
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"获取默认配置失败: {e}",
                }
            ]
        }


# Tool definitions for MCP server registration
ESTIMATE_COST_TOOL = {
    "name": "estimate_cost",
    "description": "估算 DSPy 优化的 API 成本，帮助用户了解运行优化所需的费用",
    "input_schema": {
        "type": "object",
        "properties": {
            "optimizer": {
                "type": "string",
                "enum": ["gepa", "miprov2"],
                "description": "优化器名称",
            },
            "mode": {
                "type": "string",
                "enum": ["quick", "full"],
                "description": "运行模式",
            },
            "dataset_size": {
                "type": "integer",
                "description": "数据集大小（条数）",
            },
            "model": {
                "type": "string",
                "description": "使用的模型（可选）",
                "default": "claude-opus-4.5",
            },
        },
        "required": ["optimizer", "mode", "dataset_size"],
    },
    "handler": estimate_cost,
}

LIST_OPTIMIZERS_TOOL = {
    "name": "list_optimizers",
    "description": "列出所有可用的 DSPy 优化器及其特性",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
    "handler": list_optimizers,
}

GET_DEFAULTS_TOOL = {
    "name": "get_defaults",
    "description": "获取指定优化器和模式的默认超参配置",
    "input_schema": {
        "type": "object",
        "properties": {
            "optimizer": {
                "type": "string",
                "enum": ["gepa", "miprov2"],
                "description": "优化器名称",
            },
            "mode": {
                "type": "string",
                "enum": ["quick", "full"],
                "description": "运行模式",
            },
        },
        "required": ["optimizer", "mode"],
    },
    "handler": get_defaults,
}

DOMAIN_OPS_TOOLS = [ESTIMATE_COST_TOOL, LIST_OPTIMIZERS_TOOL, GET_DEFAULTS_TOOL]

__all__ = [
    "estimate_cost",
    "list_optimizers",
    "get_defaults",
    "DOMAIN_OPS_TOOLS",
    "ESTIMATE_COST_TOOL",
    "LIST_OPTIMIZERS_TOOL",
    "GET_DEFAULTS_TOOL",
]
