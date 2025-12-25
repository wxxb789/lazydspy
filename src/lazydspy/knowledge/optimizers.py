"""Optimizer knowledge base for DSPy optimizers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OptimizerInfo(BaseModel):
    """Information about a DSPy optimizer."""

    name: str = Field(..., description="优化器显示名称")
    description: str = Field(..., description="简短描述")
    recommended_for: list[str] = Field(default_factory=list, description="推荐的场景类型")
    cost_level: Literal["low", "medium", "high"] = Field(..., description="成本级别")
    hyperparameters: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="mode -> 超参默认值映射",
    )

    def get_defaults(self, mode: str) -> dict[str, Any]:
        """Get default hyperparameters for a given mode."""
        return dict(self.hyperparameters.get(mode, self.hyperparameters.get("quick", {})))


OPTIMIZER_REGISTRY: dict[str, OptimizerInfo] = {
    "gepa": OptimizerInfo(
        name="GEPA",
        description="Greedy Evolutionary Prompt Assembly - 基于进化策略的 prompt 优化，成本较低",
        recommended_for=["general", "summary", "classification", "generation"],
        cost_level="low",
        hyperparameters={
            "quick": {"breadth": 2, "depth": 2, "temperature": 0.3},
            "full": {"breadth": 4, "depth": 4, "temperature": 0.7},
        },
    ),
    "miprov2": OptimizerInfo(
        name="MIPROv2",
        description="Model-based Instruction Prompt Refinement Optimizer v2 - 适合复杂任务，成本较高",
        recommended_for=["retrieval", "scoring", "qa", "reasoning"],
        cost_level="medium",
        hyperparameters={
            "quick": {"search_size": 8, "temperature": 0.3},
            "full": {"search_size": 16, "temperature": 0.6},
        },
    ),
}


def get_optimizer_info(name: str) -> OptimizerInfo | None:
    """Get optimizer info by name."""
    return OPTIMIZER_REGISTRY.get(name.lower())


def get_recommended_optimizer(scenario_type: str) -> str:
    """Get recommended optimizer for a scenario type."""
    scenario_map: dict[str, str] = {
        "summary": "gepa",
        "retrieval": "miprov2",
        "scoring": "miprov2",
        "qa": "miprov2",
        "classification": "gepa",
        "generation": "gepa",
        "reasoning": "miprov2",
        "general": "gepa",
    }
    return scenario_map.get(scenario_type.lower(), "gepa")


def list_all_optimizers() -> list[dict[str, Any]]:
    """List all available optimizers with their info."""
    return [
        {
            "name": info.name,
            "key": key,
            "description": info.description,
            "recommended_for": info.recommended_for,
            "cost_level": info.cost_level,
        }
        for key, info in OPTIMIZER_REGISTRY.items()
    ]


__all__ = [
    "OptimizerInfo",
    "OPTIMIZER_REGISTRY",
    "get_optimizer_info",
    "get_recommended_optimizer",
    "list_all_optimizers",
]
