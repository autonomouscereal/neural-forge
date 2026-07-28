"""Shared immutable contract for the current LFM2 exact-4x lane."""

from __future__ import annotations


EXPECTED_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_CHILD_UNIQUE_PARAMETERS = 57_422_440
PARENT_MODEL_ROLE = "parent"
EXPECTED_CHILD_TOPOLOGY = {
    "vocab_size": 4_096,
    "hidden_size": 776,
    "intermediate_size": 700,
    "num_hidden_layers": 14,
    "num_attention_heads": 10,
    "num_key_value_heads": 5,
    "head_dim": 64,
    "tie_word_embeddings": False,
}
CAUSAL_HEAD_RECIPE = {
    "candidate_id": "exact4x-context-residual-attention-causal-head-a1000-r1e2",
    "ridge_fraction": 1e-2,
    "residual_scale_ppm": 1_000_000,
    "row_norm_calibration_ppm": 250_000,
    "protect_specials": True,
}


__all__ = [
    "CAUSAL_HEAD_RECIPE",
    "EXPECTED_CHILD_TOPOLOGY",
    "EXPECTED_CHILD_UNIQUE_PARAMETERS",
    "EXPECTED_PARENT_UNIQUE_PARAMETERS",
    "PARENT_MODEL_ROLE",
]
