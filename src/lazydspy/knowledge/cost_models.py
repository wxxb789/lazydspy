"""Cost estimation models for DSPy optimization."""

from __future__ import annotations

from typing import Any, Literal

# Model pricing per million tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Claude models
    "claude-opus-4.5": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    # OpenAI models
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
}

# Estimated API calls per optimizer/mode combination
OPTIMIZER_CALL_ESTIMATES: dict[str, dict[str, dict[str, int]]] = {
    "gepa": {
        "quick": {"calls_per_example": 3, "overhead_calls": 10},
        "full": {"calls_per_example": 8, "overhead_calls": 30},
    },
    "miprov2": {
        "quick": {"calls_per_example": 5, "overhead_calls": 20},
        "full": {"calls_per_example": 15, "overhead_calls": 50},
    },
}


def estimate_optimization_cost(
    optimizer: str,
    mode: Literal["quick", "full"],
    dataset_size: int,
    model: str = "claude-opus-4.5",
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200,
) -> dict[str, Any]:
    """Estimate the cost of running an optimization.

    Args:
        optimizer: Optimizer name (gepa/miprov2)
        mode: Run mode (quick/full)
        dataset_size: Number of examples in the dataset
        model: Model to use for optimization
        avg_input_tokens: Average input tokens per call
        avg_output_tokens: Average output tokens per call

    Returns:
        Cost estimation details
    """
    # Get pricing, default to Claude Sonnet if unknown model
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-opus-4.5"])

    # Get call estimates
    call_estimates = OPTIMIZER_CALL_ESTIMATES.get(
        optimizer.lower(),
        OPTIMIZER_CALL_ESTIMATES["gepa"],
    )
    mode_estimates = call_estimates.get(mode, call_estimates["quick"])

    # Calculate total calls
    calls_per_example = mode_estimates["calls_per_example"]
    overhead = mode_estimates["overhead_calls"]
    total_calls = calls_per_example * dataset_size + overhead

    # Calculate tokens
    total_input_tokens = total_calls * avg_input_tokens
    total_output_tokens = total_calls * avg_output_tokens

    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "optimizer": optimizer,
        "mode": mode,
        "model": model,
        "dataset_size": dataset_size,
        "estimated_calls": total_calls,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "cost_breakdown": {
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
        },
        "cost_hint": _generate_cost_hint(total_cost, mode),
    }


def _generate_cost_hint(cost: float, mode: str) -> str:
    """Generate a human-readable cost hint."""
    if cost < 0.10:
        return "成本很低，可以放心运行"
    elif cost < 1.0:
        return "成本适中，建议先用 quick 模式验证"
    elif cost < 10.0:
        if mode == "quick":
            return "成本较高，建议减少数据集大小"
        else:
            return "成本较高，建议先用 quick 模式验证后再运行 full 模式"
    else:
        return "成本很高！强烈建议先用 quick 模式和小数据集验证"


def get_model_pricing(model: str) -> dict[str, float] | None:
    """Get pricing info for a model."""
    return MODEL_PRICING.get(model)


def list_supported_models() -> list[str]:
    """List all supported models."""
    return list(MODEL_PRICING.keys())


__all__ = [
    "MODEL_PRICING",
    "OPTIMIZER_CALL_ESTIMATES",
    "estimate_optimization_cost",
    "get_model_pricing",
    "list_supported_models",
]
