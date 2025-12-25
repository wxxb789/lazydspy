"""Unified generation configuration model."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .hyperparams import (
    CheckpointSettings,
    GEPAHyperparameters,
    MIPROv2Hyperparameters,
    RunMode,
)


class GenerationConfig(BaseModel):
    """Unified configuration for script generation.

    This is the single source of truth for all generation settings,
    replacing the duplicate definitions in the old cli.py and models.py.
    """

    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        description="会话唯一标识",
    )
    scenario: str = Field(..., description="场景描述")
    input_fields: list[str] = Field(..., description="输入字段列表")
    output_fields: list[str] = Field(..., description="输出字段列表")
    model_preference: str = Field(
        default="claude-opus-4.5",
        description="模型偏好",
    )
    algorithm: str = Field(default="GEPA", description="优化算法 (GEPA 或 MIPROv2)")
    # NOTE: We use Any here to avoid Pydantic's Union type coercion which would
    # convert dict to model instance before model_validator runs, losing the
    # information about which fields were user-specified vs defaults.
    # The actual type will be set in apply_hyperparameter_presets().
    hyperparameters: Any = Field(
        default_factory=dict,
        description="优化超参（可为空，按模式应用默认值）",
    )
    data_path: Path | None = Field(default=None, description="训练数据路径")
    mode: str = Field(default="quick", description="运行模式 (quick 或 full)")
    subset_size: int | None = Field(default=None, description="quick 模式的子集大小")

    # Checkpoint settings
    checkpoint_enabled: bool = Field(default=False, description="是否启用 checkpoint")
    checkpoint_dir: Path = Field(default=Path("checkpoints"), description="Checkpoint 目录")
    checkpoint_interval: int = Field(default=1, ge=1, description="Checkpoint 间隔（步数）")
    max_checkpoints: int = Field(default=20, ge=1, description="最多保留多少个 checkpoint")
    resume: bool = Field(default=False, description="是否从 checkpoint 恢复")

    # Generation options
    generate_sample_data: bool = Field(
        default=False,
        description="是否生成 sample-data/train.jsonl 示例",
    )

    @field_validator("algorithm", mode="before")
    @classmethod
    def normalize_algorithm(cls, value: str) -> str:
        """Normalize algorithm name to standard format."""
        normalized = str(value).lower().replace("-", "").replace("_", "")
        if "gepa" in normalized:
            return "GEPA"
        if "miprov2" in normalized or "mipro" in normalized:
            return "MIPROv2"
        raise ValueError("algorithm must be GEPA or MIPROv2")

    @field_validator("input_fields", "output_fields", mode="before")
    @classmethod
    def parse_fields(cls, value: Any) -> list[str]:
        """Parse comma-separated string or list to field list."""
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        raise ValueError("fields must be a comma-separated string or list")

    @field_validator("hyperparameters", mode="before")
    @classmethod
    def parse_hyperparameters(cls, value: Any, info: Any) -> dict[str, Any]:
        """Parse hyperparameters from various formats and store raw overrides."""
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed: dict[str, Any] = {}
            for chunk in value.split(","):
                if "=" not in chunk:
                    continue
                key, raw = chunk.split("=", maxsplit=1)
                key = key.strip()
                raw = raw.strip()
                if not key:
                    continue
                if raw.isdigit():
                    parsed[key] = int(raw)
                else:
                    try:
                        parsed[key] = float(raw)
                    except ValueError:
                        parsed[key] = raw
            return parsed
        raise ValueError("hyperparameters must be a mapping or key=value list")

    @field_validator("data_path", mode="before")
    @classmethod
    def parse_data_path(cls, value: Any) -> Path | None:
        """Parse data path."""
        if value is None or value == "":
            return None
        if isinstance(value, Path):
            return value
        return Path(str(value)).expanduser()

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def parse_checkpoint_dir(cls, value: Any) -> Path:
        """Parse checkpoint directory."""
        if isinstance(value, Path):
            return value
        if value is None or value == "":
            return Path("checkpoints")
        return Path(str(value)).expanduser()

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: str) -> str:
        """Normalize run mode."""
        normalized = str(value).strip().lower()
        if normalized not in {"quick", "full"}:
            raise ValueError("mode must be quick or full")
        return normalized

    @field_validator("subset_size", mode="before")
    @classmethod
    def validate_subset_size(cls, value: Any) -> int | None:
        """Validate subset size."""
        if value is None or value == "":
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("subset_size must be an integer") from exc
        if parsed <= 0:
            raise ValueError("subset_size must be positive")
        return parsed

    @model_validator(mode="after")
    def apply_hyperparameter_presets(self) -> "GenerationConfig":
        """Apply default hyperparameters based on algorithm and mode."""
        # At this point, hyperparameters is always a dict (guaranteed by parse_hyperparameters)
        # This dict contains ONLY the user-specified overrides, not defaults
        overrides = {k: v for k, v in self.hyperparameters.items() if v is not None}

        normalized_algo = self.algorithm.lower().replace("-", "").replace("_", "")
        run_mode = cast(RunMode, self.mode)

        new_hyper: GEPAHyperparameters | MIPROv2Hyperparameters
        if normalized_algo == "gepa":
            new_hyper = GEPAHyperparameters.from_mode(run_mode, overrides)
        else:
            new_hyper = MIPROv2Hyperparameters.from_mode(run_mode, overrides)

        # Use object.__setattr__ to bypass frozen validation if any
        object.__setattr__(self, "hyperparameters", new_hyper)

        return self

    @property
    def active_hyperparameters(self) -> dict[str, Any]:
        """Return the resolved hyperparameters dict."""
        if isinstance(self.hyperparameters, BaseModel):
            return self.hyperparameters.model_dump()
        return dict(self.hyperparameters) if self.hyperparameters else {}

    def to_checkpoint_settings(self) -> CheckpointSettings:
        """Convert checkpoint fields to CheckpointSettings model."""
        return CheckpointSettings(
            enabled=self.checkpoint_enabled,
            interval=self.checkpoint_interval,
            max_checkpoints=self.max_checkpoints,
            directory=self.checkpoint_dir,
        )


__all__ = ["GenerationConfig"]
