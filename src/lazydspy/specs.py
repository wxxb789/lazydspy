"""Structured specification models for lazydspy."""

from __future__ import annotations

import os
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from lazydspy.knowledge.optimizers import get_recommended_optimizer


def _normalize_str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        raise ValueError(f"{field_name} 必须是字符串列表")

    items = [item for item in items if item]
    return list(dict.fromkeys(items))


class OptimizationSpec(BaseModel):
    """Validated specification for DSPy optimization script generation."""

    scenario: str = Field(..., description="优化场景描述")
    scenario_type: str = Field(default="general", description="场景类型")
    input_fields: list[str] = Field(..., description="输入字段列表")
    output_fields: list[str] = Field(..., description="输出字段列表")
    dataset_path: str = Field(..., description="数据集路径（JSONL）")
    dataset_size: int | None = Field(default=None, description="数据集大小（可选）")
    model: str | None = Field(default=None, description="模型名称（可选）")
    optimizer: Literal["gepa", "miprov2"] | None = Field(default=None, description="优化器")
    mode: Literal["quick", "full"] = Field(default="quick", description="运行模式")
    checkpoint_dir: str | None = Field(default=None, description="Checkpoint 路径")
    resume: bool = Field(default=False, description="是否断点续跑")
    notes: str | None = Field(default=None, description="其他说明")

    @field_validator("scenario", "dataset_path", mode="before")
    @classmethod
    def _validate_required_text(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise ValueError("字段不能为空")
        return text

    @field_validator("scenario_type", mode="before")
    @classmethod
    def _normalize_scenario_type(cls, value: Any) -> str:
        text = str(value).strip().lower() if value is not None else ""
        return text or "general"

    @field_validator("input_fields", mode="before")
    @classmethod
    def _normalize_input_fields(cls, value: Any) -> list[str]:
        items = _normalize_str_list(value, "input_fields")
        if not items:
            raise ValueError("input_fields 至少包含一个字段")
        return items

    @field_validator("output_fields", mode="before")
    @classmethod
    def _normalize_output_fields(cls, value: Any) -> list[str]:
        items = _normalize_str_list(value, "output_fields")
        if not items:
            raise ValueError("output_fields 至少包含一个字段")
        return items

    @field_validator("dataset_size", mode="before")
    @classmethod
    def _validate_dataset_size(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset_size 必须是整数") from exc
        if number <= 0:
            raise ValueError("dataset_size 必须大于 0")
        return number

    @model_validator(mode="after")
    def _fill_defaults(self) -> "OptimizationSpec":
        if self.optimizer is None:
            self.optimizer = cast(
                Literal["gepa", "miprov2"],
                get_recommended_optimizer(self.scenario_type),
            )
        if not self.model:
            self.model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        return self

    def summary(self) -> str:
        """Return a human-readable summary for confirmation."""
        lines = [
            f"- 场景描述: {self.scenario}",
            f"- 场景类型: {self.scenario_type}",
            f"- 输入字段: {', '.join(self.input_fields)}",
            f"- 输出字段: {', '.join(self.output_fields)}",
            f"- 数据集路径: {self.dataset_path}",
            f"- 模型: {self.model}",
            f"- 优化器: {self.optimizer}",
            f"- 模式: {self.mode}",
        ]
        if self.dataset_size:
            lines.append(f"- 数据集大小: {self.dataset_size}")
        if self.checkpoint_dir:
            lines.append(f"- Checkpoint: {self.checkpoint_dir}")
        if self.resume:
            lines.append("- 断点续跑: 是")
        if self.notes:
            lines.append(f"- 备注: {self.notes}")
        return "\n".join(lines)

    model_config = {"extra": "forbid"}


def format_validation_error(error: ValidationError) -> str:
    """Format Pydantic validation errors for display."""
    messages = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", []))
        message = item.get("msg", "invalid")
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages)


__all__ = ["OptimizationSpec", "format_validation_error"]
