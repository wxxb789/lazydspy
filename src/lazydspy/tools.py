"""MCP tools for lazydspy Agent.

Uses @tool decorator from claude-agent-sdk to define business tools.
Business logic is separated into _impl functions for easier testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from claude_agent_sdk import create_sdk_mcp_server, tool
from pydantic import ValidationError

from lazydspy.knowledge.cost_models import estimate_optimization_cost, list_supported_models
from lazydspy.knowledge.optimizers import (
    get_optimizer_info,
    get_recommended_optimizer,
    list_all_optimizers,
)
from lazydspy.specs import OptimizationSpec, format_validation_error
from lazydspy.state import AgentState, ConversationStage


def _make_text_response(text: str) -> dict[str, Any]:
    """Create a standard MCP tool response."""
    return {"content": [{"type": "text", "text": text}]}


_STATE: AgentState | None = None


def bind_state(state: AgentState | None) -> None:
    """Bind agent state for tool callbacks."""
    global _STATE
    _STATE = state


def _require_state() -> AgentState:
    if _STATE is None:
        raise RuntimeError("Agent state is not initialized")
    return _STATE


def _normalize_mode(
    value: Any,
    default: Literal["quick", "full"] = "quick",
) -> Literal["quick", "full"]:
    """Normalize and validate mode input."""
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text == "quick":
        return "quick"
    if text == "full":
        return "full"
    raise ValueError("mode 必须是 quick 或 full")


def _normalize_optimizer(value: Any, default: str = "gepa") -> str:
    """Normalize optimizer input."""
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text


# ============================================================================
# Business Logic (testable without MCP)
# ============================================================================


async def estimate_cost_impl(args: dict[str, Any]) -> dict[str, Any]:
    """Estimate the cost of running a DSPy optimization."""
    try:
        optimizer = _normalize_optimizer(args.get("optimizer"), default="gepa")
        mode = _normalize_mode(args.get("mode"), default="quick")
        dataset_size = args.get("dataset_size", 50)
        model = args.get("model")
        if model is None or (isinstance(model, str) and not model.strip()):
            model = "claude-opus-4.5"

        result = estimate_optimization_cost(
            optimizer=optimizer,
            mode=mode,
            dataset_size=dataset_size,
            model=str(model),
        )
    except ValueError as exc:
        return _make_text_response(f"参数错误: {exc}")

    return _make_text_response(json.dumps(result, ensure_ascii=False, indent=2))


async def submit_spec_impl(args: dict[str, Any]) -> dict[str, Any]:
    """Validate and register a structured spec."""
    try:
        spec = OptimizationSpec.model_validate(args)
    except ValidationError as exc:
        return _make_text_response(f"Spec 校验失败: {format_validation_error(exc)}")

    try:
        state = _require_state()
    except RuntimeError as exc:
        return _make_text_response(f"状态错误: {exc}")

    state.spec = spec
    state.stage = ConversationStage.CONFIRM
    state.last_validation_errors = []

    summary = spec.summary()
    return _make_text_response(f"已生成规范，请用户确认:\n{summary}")


async def mark_generation_complete_impl(args: dict[str, Any]) -> dict[str, Any]:
    """Mark generation complete and register generated files."""
    try:
        state = _require_state()
    except RuntimeError as exc:
        return _make_text_response(f"状态错误: {exc}")

    files: list[str] = []
    raw_files = args.get("files")
    if isinstance(raw_files, str):
        files = [raw_files]
    elif isinstance(raw_files, list):
        files = [str(item) for item in raw_files if item]

    output_dir = args.get("output_dir")
    if output_dir:
        output_path = Path(str(output_dir))
        if output_path.is_dir():
            files.extend(str(path) for path in output_path.rglob("*.py"))
        else:
            return _make_text_response(f"output_dir 不存在或不可读: {output_path}")

    normalized = [Path(path).expanduser() for path in files]
    state.generated_files = normalized
    state.stage = ConversationStage.VALIDATE

    return _make_text_response("已记录生成文件，准备校验。")


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
    if isinstance(scenario, str):
        scenario = scenario.strip() or None

    optimizer_name = args.get("optimizer")
    if isinstance(optimizer_name, str):
        optimizer_name = optimizer_name.strip().lower() or None

    try:
        mode = _normalize_mode(args.get("mode"), default="quick")
    except ValueError as exc:
        return _make_text_response(f"参数错误: {exc}")

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
    "submit_spec",
    "提交优化需求的结构化规范",
    {
        "scenario": {"type": "string", "description": "优化场景描述"},
        "scenario_type": {"type": "string", "description": "场景类型（可选）"},
        "input_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "输入字段列表",
        },
        "output_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "输出字段列表",
        },
        "dataset_path": {"type": "string", "description": "JSONL 数据集路径"},
        "dataset_size": {"type": "integer", "description": "数据集大小（可选）"},
        "model": {"type": "string", "description": "模型名称（可选）"},
        "optimizer": {"type": "string", "description": "优化器名称（可选）"},
        "mode": {"type": "string", "description": "运行模式（可选）"},
        "checkpoint_dir": {"type": "string", "description": "Checkpoint 目录（可选）"},
        "resume": {"type": "boolean", "description": "是否断点续跑（可选）"},
        "notes": {"type": "string", "description": "其他说明（可选）"},
    },
)
async def submit_spec(args: dict[str, Any]) -> dict[str, Any]:
    """Submit and validate the spec."""
    return await submit_spec_impl(args)


@tool(
    "mark_generation_complete",
    "标记脚本生成完成并提交文件列表",
    {
        "files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "生成的文件路径列表",
        },
        "output_dir": {
            "type": "string",
            "description": "输出目录（可选，用于扫描 *.py）",
        },
    },
)
async def mark_generation_complete(args: dict[str, Any]) -> dict[str, Any]:
    """Mark generation complete and register output files."""
    return await mark_generation_complete_impl(args)


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
    "mcp__lazydspy__submit_spec",
    "mcp__lazydspy__mark_generation_complete",
    "mcp__lazydspy__estimate_cost",
    "mcp__lazydspy__list_optimizers",
    "mcp__lazydspy__get_defaults",
]


def create_mcp_server() -> Any:
    """Create an MCP server with all lazydspy tools."""
    return create_sdk_mcp_server(
        name="lazydspy",
        version="0.2.0",
        tools=[
            submit_spec,
            mark_generation_complete,
            estimate_cost,
            list_optimizers,
            get_defaults,
        ],
    )


__all__ = [
    "create_mcp_server",
    "estimate_cost",
    "estimate_cost_impl",
    "get_defaults",
    "get_defaults_impl",
    "list_optimizers",
    "list_optimizers_impl",
    "mark_generation_complete",
    "mark_generation_complete_impl",
    "submit_spec",
    "submit_spec_impl",
    "bind_state",
    "TOOL_NAMES",
]
