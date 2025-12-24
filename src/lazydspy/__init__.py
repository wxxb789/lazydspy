"""lazydspy package initialization."""

from __future__ import annotations

from .cli import main
from .schemas import MetricResult, ScoreDetail

__all__ = ["main", "MetricResult", "ScoreDetail"]
