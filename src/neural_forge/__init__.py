"""Neural Forge public API."""

from neural_forge.checkpoint import DenseModel, load_model, random_toy_model, save_model
from neural_forge.surgery import (
    expand_depth_identity,
    expand_hidden_layer,
    merge_models,
    plan_widths_for_target_params,
    resize_to_widths,
    shrink_hidden_layer,
)

__all__ = [
    "DenseModel",
    "expand_depth_identity",
    "expand_hidden_layer",
    "load_model",
    "merge_models",
    "plan_widths_for_target_params",
    "random_toy_model",
    "resize_to_widths",
    "save_model",
    "shrink_hidden_layer",
]

