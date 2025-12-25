"""lazydspy - Generate DSPy optimization scripts through conversation.

This package provides an Agent-driven CLI tool that helps users create
DSPy prompt optimization scripts through interactive dialogue.
"""

from __future__ import annotations

from .cli import main
from .models import (
    GEPA_PRESETS,
    MIPROV2_PRESETS,
    CheckpointSettings,
    CheckpointState,
    GenerationConfig,
    GEPAHyperparameters,
    MIPROv2Hyperparameters,
    OptimizationResultSummary,
    OptimizerChoice,
    RunMode,
)
from .schemas import MetricResult, ScoreDetail

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # CLI
    "main",
    # Models
    "GenerationConfig",
    "GEPAHyperparameters",
    "MIPROv2Hyperparameters",
    "GEPA_PRESETS",
    "MIPROV2_PRESETS",
    "CheckpointSettings",
    "CheckpointState",
    "OptimizationResultSummary",
    "OptimizerChoice",
    "RunMode",
    # Schemas
    "MetricResult",
    "ScoreDetail",
]
