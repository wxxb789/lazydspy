"""Core configuration and runtime models for lazydspy."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

OptimizerChoice = Literal["gepa", "miprov2"]
RunMode = Literal["quick", "full"]


class CheckpointSettings(BaseModel):
    """Checkpoint-related configuration."""

    enabled: bool = Field(False, description="是否启用 checkpoint")
    interval: int = Field(10, ge=1, le=1000, description="保存间隔（步数）")
    max_checkpoints: int = Field(3, ge=1, le=50, description="最多保留的 checkpoint 数量")
    directory: Path | None = Field(default=None, description="checkpoint 存储目录")


class GenerationConfig(BaseModel):
    """High-level generation and optimization settings."""

    name: str = Field(..., description="任务名称或标识")
    optimizer: OptimizerChoice = Field("gepa", description="优化器选择")
    run_mode: RunMode = Field("quick", description="运行模式（quick/full）")
    breadth: int = Field(2, ge=1, le=32, description="候选拓展宽度")
    depth: int = Field(2, ge=1, le=32, description="搜索深度")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="采样温度")
    checkpoint: CheckpointSettings = Field(
        default_factory=CheckpointSettings, description="checkpoint 配置"
    )


class CheckpointState(BaseModel):
    """Recorded state for the latest checkpoint."""

    path: Path | None = Field(default=None, description="最近 checkpoint 路径")
    step: int = Field(0, ge=0, description="已完成的步数")
    recovered: bool = Field(False, description="是否由 checkpoint 恢复")


class OptimizationResultSummary(BaseModel):
    """Summary for an optimization run."""

    run_id: str = Field(..., description="运行标识")
    optimizer: OptimizerChoice = Field(..., description="使用的优化器")
    run_mode: RunMode = Field(..., description="运行模式")
    steps_completed: int = Field(..., ge=0, description="完成的步骤数")
    best_score: float | None = Field(None, ge=0.0, le=1.0, description="最优得分（0-1）")
    checkpoint_state: CheckpointState | None = Field(
        default=None, description="最新 checkpoint 状态"
    )


__all__ = [
    "CheckpointSettings",
    "CheckpointState",
    "GenerationConfig",
    "OptimizationResultSummary",
    "OptimizerChoice",
    "RunMode",
]
