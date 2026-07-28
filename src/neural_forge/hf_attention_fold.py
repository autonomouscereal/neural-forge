"""Head- and RoPE-constrained sparse folds for split-QKV/GQA attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from neural_forge.hf_feature_fold import FeatureFoldMap


class AttentionFoldError(ValueError):
    """Raised when a requested attention fold cannot preserve GQA topology."""


@dataclass(frozen=True, slots=True)
class AttentionHeadFoldReport:
    role: Literal["query", "key", "value", "context"]
    source_heads: int
    target_heads: int
    source_head_dim: int
    target_head_dim: int
    source_width: int
    target_width: int
    group_sizes: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "gqa-head-rope-pair-constrained-fold",
            "role": self.role,
            "source_heads": self.source_heads,
            "target_heads": self.target_heads,
            "source_head_dim": self.source_head_dim,
            "target_head_dim": self.target_head_dim,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "source_channels_covered": self.source_width,
            "coverage_fraction": 1.0,
            "full_source_coverage": True,
            "dropped_channels": 0,
            "zero_dropped_channels": True,
            "group_sizes": list(self.group_sizes),
            "min_group_size": min(self.group_sizes),
            "max_group_size": max(self.group_sizes),
            "full_target_rank": True,
            "columns_normalized": True,
            "deterministic": True,
            "gqa_routing_preserved": True,
            "rotary_pairs_preserved": True,
            "sign_flips_allowed": False,
            "negative_coefficient_count": 0,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class AttentionFoldMaps:
    query: FeatureFoldMap
    key: FeatureFoldMap
    value: FeatureFoldMap
    context: FeatureFoldMap


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AttentionFoldError(f"{name} must be a positive integer.")
    return value


def validate_gqa_attention_fold_geometry(
    *,
    source_query_heads: int,
    source_key_value_heads: int,
    source_head_dim: int,
    target_query_heads: int,
    target_key_value_heads: int,
    target_head_dim: int,
) -> None:
    """Prove the deterministic GQA fold can populate every target channel."""

    values = {
        "source_query_heads": source_query_heads,
        "source_key_value_heads": source_key_value_heads,
        "source_head_dim": source_head_dim,
        "target_query_heads": target_query_heads,
        "target_key_value_heads": target_key_value_heads,
        "target_head_dim": target_head_dim,
    }
    values = {name: _positive_int(value, name) for name, value in values.items()}
    source_query_heads = values["source_query_heads"]
    source_key_value_heads = values["source_key_value_heads"]
    source_head_dim = values["source_head_dim"]
    target_query_heads = values["target_query_heads"]
    target_key_value_heads = values["target_key_value_heads"]
    target_head_dim = values["target_head_dim"]

    if source_head_dim % 2 or target_head_dim % 2:
        raise AttentionFoldError("RoPE head dimensions must be even.")
    if source_query_heads % source_key_value_heads:
        raise AttentionFoldError("Source query heads must divide into KV groups.")
    if target_query_heads % target_key_value_heads:
        raise AttentionFoldError("Target query heads must divide into KV groups.")
    if source_query_heads < target_query_heads:
        raise AttentionFoldError("Attention expansion is outside the fold operator.")
    if source_key_value_heads < target_key_value_heads:
        raise AttentionFoldError("KV expansion is outside the fold operator.")
    if source_head_dim < target_head_dim:
        raise AttentionFoldError("Head-dimension expansion is outside this operator.")
    source_queries_per_key_value = source_query_heads // source_key_value_heads
    target_queries_per_key_value = target_query_heads // target_key_value_heads
    if source_queries_per_key_value < target_queries_per_key_value:
        raise AttentionFoldError(
            "GQA fold cannot populate every target local-query head from the "
            "source KV groups."
        )


def _head_assignment(
    torch: Any,
    *,
    role: Literal["query", "key", "value", "context"],
    source_heads: int,
    target_heads: int,
    source_head_dim: int,
    target_head_dim: int,
    source_query_heads: int,
    source_key_value_heads: int,
    target_query_heads: int,
    target_key_value_heads: int,
) -> FeatureFoldMap:
    source_half = source_head_dim // 2
    target_half = target_head_dim // 2
    source_q_per_kv = source_query_heads // source_key_value_heads
    target_q_per_kv = target_query_heads // target_key_value_heads
    assignment = torch.empty(source_heads * source_head_dim, dtype=torch.long)

    for source_head in range(source_heads):
        if role in {"query", "context"}:
            source_kv_head = source_head // source_q_per_kv
            source_local_query = source_head % source_q_per_kv
            target_kv_head = min(
                target_key_value_heads - 1,
                source_kv_head
                * target_key_value_heads
                // source_key_value_heads,
            )
            target_local_query = min(
                target_q_per_kv - 1,
                source_local_query * target_q_per_kv // source_q_per_kv,
            )
            target_head = target_kv_head * target_q_per_kv + target_local_query
        else:
            target_head = min(
                target_heads - 1,
                source_head * target_heads // source_heads,
            )

        for source_channel in range(source_head_dim):
            half = 0 if source_channel < source_half else 1
            source_pair = source_channel % source_half
            target_pair = min(
                target_half - 1,
                source_pair * target_half // source_half,
            )
            target_channel = target_pair + half * target_half
            source_index = source_head * source_head_dim + source_channel
            assignment[source_index] = target_head * target_head_dim + target_channel

    target_width = target_heads * target_head_dim
    coefficients = torch.ones(assignment.numel(), dtype=torch.float64)
    counts = torch.bincount(assignment, minlength=target_width)
    if bool((counts == 0).any()):
        raise AttentionFoldError(
            f"{role} fold left one or more target channels unassigned."
        )
    report = AttentionHeadFoldReport(
        role=role,
        source_heads=source_heads,
        target_heads=target_heads,
        source_head_dim=source_head_dim,
        target_head_dim=target_head_dim,
        source_width=int(assignment.numel()),
        target_width=target_width,
        group_sizes=tuple(int(value) for value in counts.tolist()),
    )
    return FeatureFoldMap(
        assignment,
        coefficients,
        target_width,
        report=report,
    )


def build_gqa_attention_fold_maps(
    *,
    source_query_heads: int,
    source_key_value_heads: int,
    source_head_dim: int,
    target_query_heads: int,
    target_key_value_heads: int,
    target_head_dim: int,
) -> AttentionFoldMaps:
    """Build full-coverage maps that keep GQA heads and rotary pairs together."""

    validate_gqa_attention_fold_geometry(
        source_query_heads=source_query_heads,
        source_key_value_heads=source_key_value_heads,
        source_head_dim=source_head_dim,
        target_query_heads=target_query_heads,
        target_key_value_heads=target_key_value_heads,
        target_head_dim=target_head_dim,
    )

    import torch

    common = {
        "torch": torch,
        "source_head_dim": source_head_dim,
        "target_head_dim": target_head_dim,
        "source_query_heads": source_query_heads,
        "source_key_value_heads": source_key_value_heads,
        "target_query_heads": target_query_heads,
        "target_key_value_heads": target_key_value_heads,
    }
    query = _head_assignment(
        **common,
        role="query",
        source_heads=source_query_heads,
        target_heads=target_query_heads,
    )
    key = _head_assignment(
        **common,
        role="key",
        source_heads=source_key_value_heads,
        target_heads=target_key_value_heads,
    )
    value = _head_assignment(
        **common,
        role="value",
        source_heads=source_key_value_heads,
        target_heads=target_key_value_heads,
    )
    context = _head_assignment(
        **common,
        role="context",
        source_heads=source_query_heads,
        target_heads=target_query_heads,
    )
    return AttentionFoldMaps(query=query, key=key, value=value, context=context)


__all__ = [
    "AttentionFoldError",
    "AttentionFoldMaps",
    "AttentionHeadFoldReport",
    "build_gqa_attention_fold_maps",
    "validate_gqa_attention_fold_geometry",
]
