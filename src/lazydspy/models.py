"""Core configuration and runtime models for lazydspy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

OptimizerChoice = Literal["gepa", "miprov2"]
RunMode = Literal["quick", "full"]

GEPA_PRESETS: dict[RunMode, dict[str, Any]] = {
    "quick": {"breadth": 2, "depth": 2, "temperature": 0.3},
    "full": {"breadth": 4, "depth": 4, "temperature": 0.7},
}

MIPROV2_PRESETS: dict[RunMode, dict[str, Any]] = {
    "quick": {"search_size": 8, "temperature": 0.3},
    "full": {"search_size": 16, "temperature": 0.6},
}


class CheckpointSettings(BaseModel):
    """Checkpoint-related configuration."""

    enabled: bool = Field(False, description="是否启用 checkpoint")
    interval: int = Field(1, ge=1, le=1000, description="保存间隔（步数）")
    max_checkpoints: int = Field(20, ge=1, le=50, description="最多保留的 checkpoint 数量")
    directory: Path | None = Field(default=None, description="checkpoint 存储目录")


class GEPAHyperparameters(BaseModel):
    """GEPA 超参配置。"""

    breadth: int = Field(2, ge=1, le=32, description="候选拓展宽度")
    depth: int = Field(2, ge=1, le=32, description="搜索深度")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="采样温度")

    @classmethod
    def from_mode(
        cls, mode: RunMode, overrides: dict[str, Any] | None = None
    ) -> "GEPAHyperparameters":
        base = dict(GEPA_PRESETS[mode])
        updates = {k: v for k, v in (overrides or {}).items() if k in cls.model_fields}
        base.update(updates)
        return cls(**base)


class MIPROv2Hyperparameters(BaseModel):
    """MIPROv2 超参配置。"""

    search_size: int = Field(8, ge=1, le=128, description="搜索宽度")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="采样温度")

    @classmethod
    def from_mode(
        cls, mode: RunMode, overrides: dict[str, Any] | None = None
    ) -> "MIPROv2Hyperparameters":
        base = dict(MIPROV2_PRESETS[mode])
        updates = {k: v for k, v in (overrides or {}).items() if k in cls.model_fields}
        base.update(updates)
        return cls(**base)


class GenerationConfig(BaseModel):
    """High-level generation and optimization settings."""

    name: str = Field(..., description="任务名称或标识")
    optimizer: OptimizerChoice = Field("gepa", description="优化器选择")
    run_mode: RunMode = Field("quick", description="运行模式（quick/full）")
    hyperparameters: GEPAHyperparameters | MIPROv2Hyperparameters | dict[str, Any] = Field(
        default_factory=dict, description="算法特定超参"
    )
    checkpoint: CheckpointSettings = Field(
        default_factory=CheckpointSettings, description="checkpoint 配置"
    )

    @model_validator(mode="after")
    def _apply_hyperparameter_presets(self) -> "GenerationConfig":
        overrides: dict[str, Any]
        if isinstance(self.hyperparameters, BaseModel):
            overrides = self.hyperparameters.model_dump()
        elif isinstance(self.hyperparameters, dict):
            overrides = {k: v for k, v in self.hyperparameters.items() if v is not None}
        else:
            overrides = {}

        if self.optimizer == "gepa":
            self.hyperparameters = GEPAHyperparameters.from_mode(self.run_mode, overrides)
        else:
            self.hyperparameters = MIPROv2Hyperparameters.from_mode(self.run_mode, overrides)
        return self


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
    "GEPAHyperparameters",
    "MIPROv2Hyperparameters",
    "MIPROV2_PRESETS",
    "OptimizationResultSummary",
    "OptimizerChoice",
    "GEPA_PRESETS",
    "RunMode",
]
