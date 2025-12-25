"""Domain knowledge for lazydspy."""

from .cost_models import (
    MODEL_PRICING,
    OPTIMIZER_CALL_ESTIMATES,
    estimate_optimization_cost,
    get_model_pricing,
    list_supported_models,
)
from .optimizers import (
    OPTIMIZER_REGISTRY,
    OptimizerInfo,
    get_optimizer_info,
    get_recommended_optimizer,
    list_all_optimizers,
)

__all__ = [
    # Optimizers
    "OptimizerInfo",
    "OPTIMIZER_REGISTRY",
    "get_optimizer_info",
    "get_recommended_optimizer",
    "list_all_optimizers",
    # Cost models
    "MODEL_PRICING",
    "OPTIMIZER_CALL_ESTIMATES",
    "estimate_optimization_cost",
    "get_model_pricing",
    "list_supported_models",
]
