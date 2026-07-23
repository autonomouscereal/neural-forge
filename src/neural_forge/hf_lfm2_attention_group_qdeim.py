"""Standalone intact-GQA-group Q-DEIM attention folding prototype.

The compiler selects whole parent GQA groups from authenticated cached
post-softmax context.  Its residual-output mode measures empirical groups
after their real source O writer and child residual chart, feeds those original
weighted consequence features directly to POD without covariance refactoring,
and balances them against the authenticated causal tangent family before
selection.  A selected group always carries all of its
query heads and its single K/V head; Q, K, and V producers are therefore copied
intact through the target-to-source input decoder and are never averaged before
softmax.  A dense output projection reconstructs every parent context group
from the selected groups and maps the parent residual output through the
source-to-target residual chart.

The span compiler applies the same intact-group construction jointly across
the attention-bearing members of an existing ``Lfm2ModelFoldSpan``.  It keeps
one real group from every contributing layer, uses that layer's own input and
output charts, and materializes one ordinary dense child Q/K/V/O operator.

This is a structural, zero-training compiler.  Numeric diagnostics are
advisory and never accept, reject, rank, or publish a semantic candidate.  A
true context-rank deficit is returned as an authenticated typed exception
because the requested interpolation operator does not exist; it is not
silently completed with random, held-out, or pseudoinverse modes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_model_fold import Lfm2ModelFoldSpan
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)
from neural_forge import hf_lfm2_mlp_qdeim as _mlp_qdeim


LFM2_ATTENTION_GROUP_QDEIM_SCHEMA = (
    "neural-forge-lfm2-attention-intact-gqa-group-qdeim-v1"
)
LFM2_ATTENTION_GROUP_QDEIM_RANK_DEFICIT_SCHEMA = (
    "neural-forge-lfm2-attention-intact-gqa-group-qdeim-rank-deficit-v1"
)
LFM2_ATTENTION_GROUP_QDEIM_SNAPSHOT_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-attention-group-qdeim-functional-snapshots-v2"
)
_LFM2_ATTENTION_GROUP_QDEIM_LEGACY_SNAPSHOT_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-attention-group-qdeim-functional-snapshots-v1"
)
LFM2_ATTENTION_GROUP_SPAN_QDEIM_SCHEMA = (
    "neural-forge-lfm2-attention-intact-gqa-group-span-qdeim-v1"
)
LFM2_ATTENTION_GROUP_SPAN_QDEIM_CONTEXT_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-attention-group-span-complete-bundle-evidence-v2"
)
LFM2_ATTENTION_GROUP_SPAN_QDEIM_RANK_DEFICIT_SCHEMA = (
    "neural-forge-lfm2-attention-group-span-qdeim-rank-deficit-v1"
)

_ALLOWED_SNAPSHOT_KINDS = frozenset(
    {
        "analytic_child_lifted_parent_attention_tangent",
        "authenticated_structural",
    }
)
LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE = (
    "joint_sampled_token_query_key_value"
)
LFM2_ATTENTION_GROUP_QDEIM_OLDER_PREFIX_KV_PERTURBATION_ROLE = "older_prefix_key_value"
_RESIDUAL_OUTPUT_POD_PERTURBATION_ROLES = frozenset(
    {
        LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE,
        LFM2_ATTENTION_GROUP_QDEIM_OLDER_PREFIX_KV_PERTURBATION_ROLE,
    }
)
LFM2_ATTENTION_GROUP_QDEIM_RAW_CONTEXT_METRIC = "raw-post-softmax-context"
LFM2_ATTENTION_GROUP_QDEIM_RESIDUAL_OUTPUT_BALANCED_METRIC = (
    "residual-output-balanced-empirical-and-causal-tangent"
)
_RAW_CONTEXT_GROUP_METRIC = LFM2_ATTENTION_GROUP_QDEIM_RAW_CONTEXT_METRIC
_RESIDUAL_OUTPUT_BALANCED_GROUP_METRIC = (
    LFM2_ATTENTION_GROUP_QDEIM_RESIDUAL_OUTPUT_BALANCED_METRIC
)
_ALLOWED_FUNCTIONAL_POD_GROUP_METRICS = frozenset(
    {
        _RAW_CONTEXT_GROUP_METRIC,
        _RESIDUAL_OUTPUT_BALANCED_GROUP_METRIC,
    }
)


class Lfm2AttentionGroupQdeimError(ValueError):
    """Authenticated intact-group attention compilation failed closed."""


class Lfm2AttentionGroupQdeimInsufficientRankError(Lfm2AttentionGroupQdeimError):
    """Authenticated context snapshots cannot support the target group count."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = MappingProxyType(dict(report))
        super().__init__(
            "Authenticated functional attention group rank "
            f"{report['observed_context_group_rank']} is below the required "
            f"rank {report['minimum_context_group_rank']}; no random, held-out, "
            "or silent interpolation completion is permitted."
        )


@dataclass(frozen=True, slots=True)
class Lfm2AttentionGroupQdeimSnapshotBlock:
    """Separately authenticated structural/tangent context snapshots.

    ``parent_post_softmax_context_rows`` has shape
    ``[rows, source_query_heads, head_dim]``.  These rows participate only in
    functional POD and group selection.  They are not behavioral holdout rows
    and cannot confer semantic acceptance.  Residual-output POD additionally
    requires a typed perturbation role bound to the exact tangent reduction
    receipt; role-less blocks remain valid only for the legacy raw metric.
    """

    block_id: str
    snapshot_kind: str
    parent_post_softmax_context_rows: Any = field(repr=False, compare=False)
    row_sample_weights: Sequence[float] = field(repr=False, compare=False)
    generator_receipt_sha256: str
    block_weight: float = 1.0
    perturbation_role: str | None = None
    perturbation_receipt_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Lfm2CompiledAttentionGroupQdeim:
    """One dense intact-producer attention candidate and its receipt."""

    selected_source_key_value_groups: tuple[int, ...]
    selected_source_query_heads: tuple[int, ...]
    query_weight: Any = field(repr=False, compare=False)
    key_weight: Any = field(repr=False, compare=False)
    value_weight: Any = field(repr=False, compare=False)
    out_weight: Any = field(repr=False, compare=False)
    group_pod_basis: Any = field(repr=False, compare=False)
    group_interpolation: Any = field(repr=False, compare=False)
    context_reconstruction: Any = field(repr=False, compare=False)
    normalized_input_decoder: Any = field(repr=False, compare=False)
    residual_output_chart: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def state_dict(self) -> Mapping[str, Any]:
        """Return detached CPU clones of the standalone dense candidate."""

        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().contiguous().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().contiguous().clone(),
                "v_proj.weight": self.value_weight.detach().cpu().contiguous().clone(),
                "out_proj.weight": self.out_weight.detach().cpu().contiguous().clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)


@dataclass(frozen=True, slots=True)
class Lfm2AttentionGroupQdeimSpanSource:
    """One authenticated attention-bearing layer inside a model-fold span.

    Cached post-softmax context rows share one authenticated row identity
    across all supplied attention layers.  Q/K/V and O remain in that source
    layer's native hidden chart.  The per-layer decoder and output chart are
    therefore part of the source payload, not global span approximations.

    LFM2 Q/K RMSNorm weights are optional because some compatible attention
    families do not expose them.  They must be supplied as a pair when they
    exist.  The span compiler carries their selected producer lineage but does
    not claim that heterogeneous per-layer norms fit the current shared-norm
    LFM2 target without a later architecture-aware installation policy.
    """

    source_layer_index: int
    source_query_weight: Any = field(repr=False, compare=False)
    source_key_weight: Any = field(repr=False, compare=False)
    source_value_weight: Any = field(repr=False, compare=False)
    source_out_weight: Any = field(repr=False, compare=False)
    expected_source_state_sha256: str
    source_query_head_count: int
    source_key_value_head_count: int
    cached_parent_post_softmax_context_rows: Any = field(repr=False, compare=False)
    expected_cached_parent_post_softmax_context_sha256: str
    cache_receipt_sha256: str
    aligned_row_identity_sha256: str
    normalized_input_decoder: Any = field(repr=False, compare=False)
    expected_normalized_input_decoder_sha256: str
    residual_output_chart: Any = field(repr=False, compare=False)
    expected_residual_output_chart_sha256: str
    source_query_norm_weight: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    source_key_norm_weight: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_layer_index, bool)
            or not isinstance(self.source_layer_index, int)
            or self.source_layer_index < 0
        ):
            raise Lfm2AttentionGroupQdeimError(
                "source_layer_index must be a non-negative integer."
            )
        _positive_int(
            self.source_query_head_count,
            label="source_query_head_count",
        )
        _positive_int(
            self.source_key_value_head_count,
            label="source_key_value_head_count",
        )
        for name in (
            "expected_source_state_sha256",
            "expected_cached_parent_post_softmax_context_sha256",
            "cache_receipt_sha256",
            "aligned_row_identity_sha256",
            "expected_normalized_input_decoder_sha256",
            "expected_residual_output_chart_sha256",
        ):
            _digest(getattr(self, name), label=name)
        if (self.source_query_norm_weight is None) != (
            self.source_key_norm_weight is None
        ):
            raise Lfm2AttentionGroupQdeimError(
                "Source Q/K norm state must be supplied as one complete pair."
            )


@dataclass(frozen=True, slots=True)
class Lfm2CompiledSpanAttentionGroupQdeim:
    """One dense child attention operator compiled from a source depth span."""

    span: Lfm2ModelFoldSpan
    attention_source_layer_indices: tuple[int, ...]
    selected_source_key_value_groups: tuple[tuple[int, int], ...]
    selected_source_query_heads: tuple[tuple[int, int], ...]
    query_weight: Any = field(repr=False, compare=False)
    key_weight: Any = field(repr=False, compare=False)
    value_weight: Any = field(repr=False, compare=False)
    out_weight: Any = field(repr=False, compare=False)
    selected_query_norm_weight_rows: Any | None = field(
        repr=False,
        compare=False,
    )
    selected_key_norm_weight_rows: Any | None = field(
        repr=False,
        compare=False,
    )
    group_pod_basis: Any = field(repr=False, compare=False)
    group_interpolation: Any = field(repr=False, compare=False)
    context_reconstruction: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def state_dict(self) -> Mapping[str, Any]:
        """Return the standalone ordinary dense BF16 Q/K/V/O child state."""

        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().contiguous().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().contiguous().clone(),
                "v_proj.weight": self.value_weight.detach().cpu().contiguous().clone(),
                "out_proj.weight": self.out_weight.detach().cpu().contiguous().clone(),
            }
        )

    def selected_norm_state(self) -> Mapping[str, Any]:
        """Return per-selected-head Q/K norm lineage when source norms exist."""

        if self.selected_query_norm_weight_rows is None:
            return MappingProxyType({})
        return MappingProxyType(
            {
                "selected_query_norm_weight_rows": (
                    self.selected_query_norm_weight_rows.detach()
                    .cpu()
                    .contiguous()
                    .clone()
                ),
                "selected_key_norm_weight_rows": (
                    self.selected_key_norm_weight_rows.detach()
                    .cpu()
                    .contiguous()
                    .clone()
                ),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "PyTorch is required for intact-GQA-group Q-DEIM compilation."
        ) from exc
    return torch


def _resolve_qdeim_compute_device(torch: Any, compute_device: Any) -> Any:
    """Resolve one explicit fold-only algebra device without silent fallback."""

    if compute_device == "cpu":
        return torch.device("cpu")
    if not isinstance(compute_device, str):
        raise Lfm2AttentionGroupQdeimError(
            "compute_device must be 'cpu' or an explicitly indexed CUDA device."
        )
    prefix, separator, index_text = compute_device.partition(":")
    if (
        prefix != "cuda"
        or separator != ":"
        or not index_text.isascii()
        or not index_text.isdecimal()
        or index_text != str(int(index_text))
    ):
        raise Lfm2AttentionGroupQdeimError(
            "CUDA compute_device must be explicitly indexed as 'cuda:N'."
        )
    index = int(index_text)
    if not torch.cuda.is_available() or index >= int(torch.cuda.device_count()):
        raise Lfm2AttentionGroupQdeimError(
            f"CUDA compute_device index {index} is unavailable."
        )
    return torch.device("cuda", index)


def _qdeim_compute_receipt(torch: Any, requested: str, resolved: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": "neural-forge-lfm2-attention-qdeim-compute-v1",
        "requested_device": requested,
        "resolved_device": str(resolved),
        "backend": resolved.type,
        "compute_dtype": "float64",
        "authenticated_source_archive_device": "cpu",
        "active_layer_staged_to_compute_device": resolved.type == "cuda",
        "materialized_candidate_device": "cpu",
        "system_memory_is_immutable_archive": True,
        "gpu_is_ephemeral_dense_algebra_workspace": resolved.type == "cuda",
        "silent_device_fallback_performed": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    if resolved.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved)
        body.update(
            {
                "device_index": int(resolved.index),
                "device_name": str(properties.name),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
            }
        )
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2AttentionGroupQdeimError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2AttentionGroupQdeimError(f"{label} must be a positive integer.")
    return value


def _positive_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Lfm2AttentionGroupQdeimError(f"{label} must be a positive finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2AttentionGroupQdeimError(
            f"{label} must be a positive finite number."
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise Lfm2AttentionGroupQdeimError(f"{label} must be a positive finite number.")
    return parsed


def _nonnegative_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Lfm2AttentionGroupQdeimError(
            f"{label} must be a nonnegative finite number."
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2AttentionGroupQdeimError(
            f"{label} must be a nonnegative finite number."
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise Lfm2AttentionGroupQdeimError(
            f"{label} must be a nonnegative finite number."
        )
    return parsed


def _nonempty_token(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2AttentionGroupQdeimError(f"{label} must be non-empty.")
    return value.strip()


def _float_tensor(value: Any, *, label: str, ndim: int) -> Any:
    torch = _torch()
    allowed = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or value.dtype not in allowed
    ):
        raise Lfm2AttentionGroupQdeimError(
            f"{label} must be one dense materialized floating rank-{ndim} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2AttentionGroupQdeimError(f"{label} contains non-finite values.")
    return result


def _tensor_sha256(value: Any, *, label: str) -> str:
    try:
        return lfm2_authenticated_tensor_sha256(value)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise Lfm2AttentionGroupQdeimError(
            f"{label} must be one hashable materialized tensor."
        ) from exc


def _state_sha256(values: Mapping[str, Any], *, schema: str) -> str:
    return canonical_sha256(
        {
            "schema": schema,
            "tensor_sha256": {
                name: _tensor_sha256(value, label=name)
                for name, value in sorted(values.items())
            },
        }
    )


def lfm2_attention_group_qdeim_source_state_sha256(
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_out_weight: Any,
) -> str:
    """Hash the coupled authenticated parent Q/K/V/O state."""

    return _state_sha256(
        {
            "q_proj.weight": source_query_weight,
            "k_proj.weight": source_key_weight,
            "v_proj.weight": source_value_weight,
            "out_proj.weight": source_out_weight,
        },
        schema="neural-forge-lfm2-attention-group-qdeim-source-state-v1",
    )


def lfm2_attention_group_qdeim_span_source_state_sha256(
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_out_weight: Any,
    source_query_norm_weight: Any | None = None,
    source_key_norm_weight: Any | None = None,
) -> str:
    """Hash one span source's complete Q/K/V/O and optional Q/K norm state."""

    if (source_query_norm_weight is None) != (source_key_norm_weight is None):
        raise Lfm2AttentionGroupQdeimError(
            "Span source Q/K norm state must be supplied as one complete pair."
        )
    values = {
        "q_proj.weight": source_query_weight,
        "k_proj.weight": source_key_weight,
        "v_proj.weight": source_value_weight,
        "out_proj.weight": source_out_weight,
    }
    if source_query_norm_weight is not None:
        values.update(
            {
                "q_layernorm.weight": source_query_norm_weight,
                "k_layernorm.weight": source_key_norm_weight,
            }
        )
    return _state_sha256(
        values,
        schema="neural-forge-lfm2-attention-group-span-source-state-v1",
    )


def _target_template_spec(
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    target_value_weight_template: Any,
    target_out_weight_template: Any,
) -> dict[str, dict[str, Any]]:
    torch = _torch()
    values = {
        "q_proj.weight": target_query_weight_template,
        "k_proj.weight": target_key_weight_template,
        "v_proj.weight": target_value_weight_template,
        "out_proj.weight": target_out_weight_template,
    }
    result: dict[str, dict[str, Any]] = {}
    for name, value in values.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.ndim != 2
            or any(int(size) <= 0 for size in value.shape)
            or value.dtype != torch.bfloat16
        ):
            raise Lfm2AttentionGroupQdeimError(
                f"target {name} must be one dense materialized BF16 rank-2 template."
            )
        # Blank template values are deliberately never consumed.
        result[name] = {
            "shape": [int(size) for size in value.shape],
            "dtype": "bfloat16",
        }
    return result


def lfm2_attention_group_qdeim_target_template_sha256(
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    target_value_weight_template: Any,
    target_out_weight_template: Any,
) -> str:
    """Hash blank-target shape and BF16 dtype metadata, not blank values."""

    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-attention-group-qdeim-target-template-v1",
            "tensors": _target_template_spec(
                target_query_weight_template,
                target_key_weight_template,
                target_value_weight_template,
                target_out_weight_template,
            ),
        }
    )


def _row_weights(
    values: Sequence[float],
    *,
    row_count: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != row_count:
        raise Lfm2AttentionGroupQdeimError(
            f"{label} must contain one explicit positive value per context row."
        )
    return tuple(
        _positive_float(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )


def lfm2_attention_group_qdeim_context_evidence_sha256(
    parent_post_softmax_context_rows: Any,
    row_sample_weights: Sequence[float],
) -> str:
    """Hash cached folding-context rows and their explicit positive weights."""

    context = _float_tensor(
        parent_post_softmax_context_rows,
        label="parent_post_softmax_context_rows",
        ndim=3,
    )
    weights = _row_weights(
        row_sample_weights,
        row_count=int(context.shape[0]),
        label="row_sample_weights",
    )
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-attention-group-qdeim-fit-context-v1",
            "partition": "folding_fit_only_not_holdout",
            "context_rows_sha256": _tensor_sha256(
                parent_post_softmax_context_rows,
                label="parent post-softmax context rows",
            ),
            "row_sample_weights": list(weights),
        }
    )


def _normalized_snapshot_blocks(
    supplied: Sequence[Lfm2AttentionGroupQdeimSnapshotBlock],
) -> tuple[dict[str, Any], ...]:
    if isinstance(supplied, (str, bytes)):
        raise Lfm2AttentionGroupQdeimError(
            "functional_snapshot_blocks must be a sequence of snapshot blocks."
        )
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, block in enumerate(tuple(supplied)):
        prefix = f"functional_snapshot_blocks[{index}]"
        if not isinstance(block, Lfm2AttentionGroupQdeimSnapshotBlock):
            raise Lfm2AttentionGroupQdeimError(
                f"{prefix} must be Lfm2AttentionGroupQdeimSnapshotBlock."
            )
        block_id = _nonempty_token(block.block_id, label=f"{prefix}.block_id")
        if block_id in seen_ids:
            raise Lfm2AttentionGroupQdeimError(
                "functional snapshot block IDs must be unique."
            )
        seen_ids.add(block_id)
        snapshot_kind = _nonempty_token(
            block.snapshot_kind,
            label=f"{prefix}.snapshot_kind",
        )
        if snapshot_kind not in _ALLOWED_SNAPSHOT_KINDS:
            raise Lfm2AttentionGroupQdeimError(
                f"{prefix}.snapshot_kind is not an authenticated structural/tangent kind."
            )
        if block.perturbation_role is None:
            perturbation_role = None
            if block.perturbation_receipt_sha256 is not None:
                raise Lfm2AttentionGroupQdeimError(
                    f"{prefix}.perturbation_receipt_sha256 requires a typed perturbation_role."
                )
            perturbation_receipt_sha256 = None
        else:
            perturbation_role = _nonempty_token(
                block.perturbation_role,
                label=f"{prefix}.perturbation_role",
            )
            if (
                snapshot_kind != "analytic_child_lifted_parent_attention_tangent"
                or perturbation_role not in _RESIDUAL_OUTPUT_POD_PERTURBATION_ROLES
            ):
                raise Lfm2AttentionGroupQdeimError(
                    f"{prefix}.perturbation_role is not one typed causal-attention role."
                )
            perturbation_receipt_sha256 = _digest(
                block.perturbation_receipt_sha256,
                label=f"{prefix}.perturbation_receipt_sha256",
            )
        context = _float_tensor(
            block.parent_post_softmax_context_rows,
            label=f"{prefix}.parent_post_softmax_context_rows",
            ndim=3,
        )
        row_weights = _row_weights(
            block.row_sample_weights,
            row_count=int(context.shape[0]),
            label=f"{prefix}.row_sample_weights",
        )
        normalized.append(
            {
                "block_id": block_id,
                "snapshot_kind": snapshot_kind,
                "perturbation_role": perturbation_role,
                "perturbation_receipt_sha256": perturbation_receipt_sha256,
                "context_source": block.parent_post_softmax_context_rows,
                "context": context,
                "row_sample_weights": row_weights,
                "block_weight": _positive_float(
                    block.block_weight,
                    label=f"{prefix}.block_weight",
                ),
                "generator_receipt_sha256": _digest(
                    block.generator_receipt_sha256,
                    label=f"{prefix}.generator_receipt_sha256",
                ),
            }
        )
    return tuple(normalized)


def lfm2_attention_group_qdeim_snapshot_evidence_sha256(
    functional_snapshot_blocks: Sequence[Lfm2AttentionGroupQdeimSnapshotBlock],
) -> str:
    """Hash all separately receipted structural/tangent snapshot blocks."""

    normalized = _normalized_snapshot_blocks(functional_snapshot_blocks)
    typed_roles_present = any(
        block["perturbation_role"] is not None for block in normalized
    )
    blocks = []
    for block in normalized:
        body = {
            "block_id": block["block_id"],
            "snapshot_kind": block["snapshot_kind"],
            "context_rows_sha256": _tensor_sha256(
                block["context_source"],
                label=f"snapshot block {block['block_id']} context",
            ),
            "row_sample_weights": list(block["row_sample_weights"]),
            "block_weight": block["block_weight"],
            "generator_receipt_sha256": block["generator_receipt_sha256"],
        }
        if typed_roles_present:
            body.update(
                {
                    "perturbation_role": block["perturbation_role"],
                    "perturbation_receipt_sha256": block["perturbation_receipt_sha256"],
                }
            )
        blocks.append(body)
    return canonical_sha256(
        {
            "schema": (
                LFM2_ATTENTION_GROUP_QDEIM_SNAPSHOT_EVIDENCE_SCHEMA
                if typed_roles_present
                else _LFM2_ATTENTION_GROUP_QDEIM_LEGACY_SNAPSHOT_EVIDENCE_SCHEMA
            ),
            "heldout_rows_permitted": False,
            "blocks": blocks,
        }
    )


def _canonicalize_pod_signs(basis: Any) -> Any:
    torch = _torch()
    result = basis.contiguous().clone()
    for column in range(int(result.shape[1])):
        values = result[:, column]
        pivot = int(torch.argmax(values.abs()).item())
        if float(values[pivot].item()) < 0.0:
            result[:, column].neg_()
    return result.contiguous()


def _deterministic_block_pivots(basis: Any) -> tuple[tuple[int, ...], Any]:
    """Stable column-pivoted modified QR on the transposed group basis."""

    torch = _torch()
    source_groups, modes = map(int, basis.shape)
    matrix = basis.T.contiguous()
    selected: list[int] = []
    q_columns = torch.empty(
        (modes, 0),
        dtype=torch.float64,
        device=matrix.device,
    )
    residual_norms: list[float] = []
    epsilon = torch.finfo(torch.float64).eps
    for _step in range(modes):
        if q_columns.shape[1]:
            residual = matrix - q_columns @ (q_columns.T @ matrix)
            residual = residual - q_columns @ (q_columns.T @ residual)
        else:
            residual = matrix
        scores = residual.square().sum(dim=0)
        if selected:
            scores[
                torch.tensor(selected, dtype=torch.long, device=matrix.device)
            ] = -1.0
        maximum = float(scores.max().item())
        tie_tolerance = 128.0 * epsilon * max(1.0, abs(maximum))
        candidates = [
            index
            for index in range(source_groups)
            if index not in selected
            and float(scores[index].item()) >= maximum - tie_tolerance
        ]
        if not candidates:
            raise Lfm2AttentionGroupQdeimError(
                "Block-constrained pivoting found no remaining parent GQA group."
            )
        pivot = min(candidates)
        vector = residual[:, pivot].contiguous()
        norm = float(torch.linalg.vector_norm(vector).item())
        if not math.isfinite(norm) or norm <= epsilon:
            raise Lfm2AttentionGroupQdeimError(
                "Block-constrained Q-DEIM encountered a singular group pivot."
            )
        q_columns = torch.cat((q_columns, (vector / norm).unsqueeze(1)), dim=1)
        selected.append(pivot)
        residual_norms.append(norm)
    return tuple(selected), torch.tensor(residual_norms, dtype=torch.float64)


def _group_feature_matrix(
    contexts: Any,
    *,
    key_value_head_count: int,
    queries_per_key_value: int,
) -> Any:
    """Put GQA groups on columns without mixing coordinates within a group."""

    rows, query_heads, head_dim = map(int, contexts.shape)
    expected_query_heads = key_value_head_count * queries_per_key_value
    if query_heads != expected_query_heads:
        raise Lfm2AttentionGroupQdeimError(
            "Context query-head axis does not match the authenticated GQA topology."
        )
    return (
        contexts.reshape(
            rows,
            key_value_head_count,
            queries_per_key_value,
            head_dim,
        )
        .permute(0, 2, 3, 1)
        .reshape(rows * queries_per_key_value * head_dim, key_value_head_count)
        .contiguous()
    )


def _residual_output_group_feature_matrix(
    contexts: Any,
    *,
    source_out_weight: Any,
    residual_output_chart: Any,
    key_value_head_count: int,
    queries_per_key_value: int,
) -> tuple[Any, Any]:
    """Measure each intact group in the child residual-output consequence space."""

    torch = _torch()
    rows, query_heads, head_dim = map(int, contexts.shape)
    expected_query_heads = key_value_head_count * queries_per_key_value
    group_width = queries_per_key_value * head_dim
    if (
        query_heads != expected_query_heads
        or int(source_out_weight.shape[1]) != query_heads * head_dim
        or int(residual_output_chart.shape[0]) != int(source_out_weight.shape[0])
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Residual-output group features do not match the authenticated GQA/O/chart geometry."
        )
    mapped_parent_out = (residual_output_chart.T @ source_out_weight).contiguous()
    target_hidden = int(mapped_parent_out.shape[0])
    group_residual_maps = mapped_parent_out.T.reshape(
        key_value_head_count,
        group_width,
        target_hidden,
    ).contiguous()
    grouped_context = contexts.reshape(rows, key_value_head_count, group_width)
    mapped = torch.einsum(
        "rgw,gwt->rgt",
        grouped_context,
        group_residual_maps,
    )
    features = mapped.permute(0, 2, 1).reshape(
        rows * target_hidden,
        key_value_head_count,
    )
    return features.contiguous(), group_residual_maps


def _direct_feature_gram_receipt(
    factor: Any,
    *,
    factor_id: str,
) -> tuple[Any, Mapping[str, Any]]:
    """Seal the Gram matrix of original feature rows without eigentruncation."""

    torch = _torch()
    if (
        not isinstance(factor, torch.Tensor)
        or factor.ndim != 2
        or any(int(size) <= 0 for size in factor.shape)
        or factor.dtype != torch.float64
        or factor.requires_grad
        or not bool(torch.isfinite(factor).all().item())
    ):
        raise Lfm2AttentionGroupQdeimError(
            f"Direct functional factor {factor_id!r} must be one finite FP64 matrix."
        )
    gram = (factor.T @ factor).contiguous()
    trace = float(torch.trace(gram).item())
    if not math.isfinite(trace) or trace < 0.0:
        raise Lfm2AttentionGroupQdeimError(
            f"Direct functional factor {factor_id!r} has an invalid Gram trace."
        )
    body = {
        "schema": "neural-forge-lfm2-attention-direct-functional-feature-gram-v1",
        "factor_id": factor_id,
        "factor_sha256": _tensor_sha256(
            factor,
            label=f"{factor_id} direct feature factor",
        ),
        "factor_row_count": int(factor.shape[0]),
        "group_count": int(factor.shape[1]),
        "gram_sha256": _tensor_sha256(
            gram,
            label=f"{factor_id} direct feature Gram matrix",
        ),
        "gram_trace": trace,
        "original_weighted_feature_rows_used_directly": True,
        "covariance_eigendecomposition_performed": False,
        "covariance_refactor_performed": False,
        "positive_finite_feature_modes_removed": 0,
        "roundoff_negative_eigenvalue_clamps_performed": 0,
        "numeric_gram_diagnostics_have_semantic_decision_authority": False,
    }
    return gram, MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _residual_output_balanced_functional_pod(
    *,
    contexts: Any,
    fit_row_weights: Sequence[float],
    normalized_blocks: Sequence[Mapping[str, Any]],
    source_out_weight: Any,
    residual_output_chart: Any,
    key_value_head_count: int,
    queries_per_key_value: int,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Balance empirical consequence covariance against the full causal tangent family."""

    torch = _torch()
    blocks_by_role: dict[str, Mapping[str, Any]] = {}
    for block in normalized_blocks:
        role = block["perturbation_role"]
        if role is None or role in blocks_by_role:
            raise Lfm2AttentionGroupQdeimError(
                "Residual-output-balanced attention POD requires exact distinct "
                "typed joint-QKV and older-prefix-KV perturbation roles."
            )
        blocks_by_role[role] = block
    if set(blocks_by_role) != _RESIDUAL_OUTPUT_POD_PERTURBATION_ROLES:
        raise Lfm2AttentionGroupQdeimError(
            "Residual-output-balanced attention POD requires exact distinct typed "
            "joint-QKV and older-prefix-KV perturbation roles."
        )
    ordered_blocks = tuple(
        blocks_by_role[role]
        for role in (
            LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE,
            LFM2_ATTENTION_GROUP_QDEIM_OLDER_PREFIX_KV_PERTURBATION_ROLE,
        )
    )
    if len({block["generator_receipt_sha256"] for block in ordered_blocks}) != 1:
        raise Lfm2AttentionGroupQdeimError(
            "Residual-output causal tangent roles must bind one authenticated tangent bank."
        )
    if len({block["perturbation_receipt_sha256"] for block in ordered_blocks}) != len(
        ordered_blocks
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Residual-output causal tangent roles must bind distinct reduction receipts."
        )
    empirical_features, group_residual_maps = _residual_output_group_feature_matrix(
        contexts,
        source_out_weight=source_out_weight,
        residual_output_chart=residual_output_chart,
        key_value_head_count=key_value_head_count,
        queries_per_key_value=queries_per_key_value,
    )
    target_hidden = int(residual_output_chart.shape[1])
    empirical_weights = torch.tensor(
        tuple(fit_row_weights),
        dtype=torch.float64,
        device=contexts.device,
    ).repeat_interleave(target_hidden)
    weighted_empirical = (
        empirical_features * torch.sqrt(empirical_weights).unsqueeze(1)
    ).contiguous()
    empirical_gram, empirical_receipt = _direct_feature_gram_receipt(
        weighted_empirical,
        factor_id="empirical-post-softmax-context-in-child-residual-output-space",
    )

    tangent_block_receipts: list[dict[str, Any]] = []
    weighted_tangent_factors: list[Any] = []
    for block in ordered_blocks:
        feature = _group_feature_matrix(
            block["context"].to(device=contexts.device, dtype=torch.float64),
            key_value_head_count=key_value_head_count,
            queries_per_key_value=queries_per_key_value,
        )
        head_dim = int(block["context"].shape[2])
        weights = (
            torch.tensor(
                block["row_sample_weights"],
                dtype=torch.float64,
                device=contexts.device,
            )
            * block["block_weight"]
        ).repeat_interleave(queries_per_key_value * head_dim)
        weighted = (feature * torch.sqrt(weights).unsqueeze(1)).contiguous()
        block_gram, block_gram_receipt = _direct_feature_gram_receipt(
            weighted,
            factor_id=f"causal-tangent-role:{block['perturbation_role']}",
        )
        weighted_tangent_factors.append(weighted)
        tangent_block_receipts.append(
            {
                "block_id": block["block_id"],
                "snapshot_kind": block["snapshot_kind"],
                "perturbation_role": block["perturbation_role"],
                "perturbation_receipt_sha256": block["perturbation_receipt_sha256"],
                "generator_receipt_sha256": block["generator_receipt_sha256"],
                "natural_block_weight": block["block_weight"],
                "direct_feature_gram": _deep_thaw(block_gram_receipt),
                "direct_feature_gram_sha256": block_gram_receipt["result_sha256"],
                "weighted_group_gram_sha256": _tensor_sha256(
                    block_gram,
                    label=f"{block['perturbation_role']} weighted group Gram",
                ),
            }
        )

    natural_tangent_factor = torch.cat(
        weighted_tangent_factors,
        dim=0,
    ).contiguous()
    natural_tangent_gram, natural_tangent_receipt = _direct_feature_gram_receipt(
        natural_tangent_factor,
        factor_id="joint-qkv-plus-older-prefix-kv-natural-causal-tangent-family",
    )
    natural_tangent_trace = float(natural_tangent_receipt["gram_trace"])
    empirical_trace = float(empirical_receipt["gram_trace"])
    scale_tolerance = (
        4096.0
        * torch.finfo(torch.float64).eps
        * max(1, key_value_head_count)
        * max(1.0, empirical_trace, natural_tangent_trace)
    )
    if empirical_trace > scale_tolerance and natural_tangent_trace > scale_tolerance:
        tangent_family_scale = empirical_trace / natural_tangent_trace
        balance_applied = True
    else:
        tangent_family_scale = 1.0
        balance_applied = False
    if not math.isfinite(tangent_family_scale) or tangent_family_scale <= 0.0:
        raise Lfm2AttentionGroupQdeimError(
            "Residual-output attention POD produced an invalid tangent-family scale."
        )
    balanced_tangent_factor = (
        natural_tangent_factor * math.sqrt(tangent_family_scale)
    ).contiguous()
    balanced_tangent_gram, balanced_tangent_receipt = _direct_feature_gram_receipt(
        balanced_tangent_factor,
        factor_id="joint-qkv-plus-older-prefix-kv-balanced-causal-tangent-family",
    )
    combined_factor = torch.cat(
        (weighted_empirical, balanced_tangent_factor),
        dim=0,
    ).contiguous()
    combined_gram, combined_receipt = _direct_feature_gram_receipt(
        combined_factor,
        factor_id="combined-empirical-and-balanced-causal-tangent-family",
    )
    expected_combined_gram = (empirical_gram + balanced_tangent_gram).contiguous()
    combined_gram_maximum_error = float(
        (combined_gram - expected_combined_gram).abs().max().item()
    )
    combined_gram_scale = max(
        1.0,
        float(expected_combined_gram.abs().max().item()),
    )
    combined_gram_roundoff_tolerance = (
        64.0
        * torch.finfo(torch.float64).eps
        * max(1, int(combined_factor.shape[0]))
        * combined_gram_scale
    )
    combined_gram_exact = bool(torch.equal(combined_gram, expected_combined_gram))
    combined_gram_within_roundoff = (
        combined_gram_maximum_error <= combined_gram_roundoff_tolerance
    )
    if not combined_gram_within_roundoff:
        raise Lfm2AttentionGroupQdeimError(
            "Direct residual-output feature concatenation failed its Gram identity."
        )
    combined_weights = torch.ones(
        int(combined_factor.shape[0]),
        dtype=torch.float64,
        device=combined_factor.device,
    )
    body = {
        "schema": (
            "neural-forge-lfm2-attention-residual-output-balanced-functional-pod-v2"
        ),
        "group_metric": _RESIDUAL_OUTPUT_BALANCED_GROUP_METRIC,
        "empirical_post_softmax_context_mapped_through_source_o": True,
        "empirical_post_softmax_context_mapped_through_residual_output_chart": True,
        "raw_empirical_context_used_directly_for_group_selection": False,
        "raw_empirical_context_retained_for_context_reconstruction_diagnostics": True,
        "group_residual_maps_sha256": _tensor_sha256(
            group_residual_maps,
            label="empirical attention group residual maps",
        ),
        "empirical_direct_feature_gram": _deep_thaw(empirical_receipt),
        "empirical_direct_feature_gram_sha256": empirical_receipt["result_sha256"],
        "causal_tangent_blocks": tangent_block_receipts,
        "causal_tangent_bank_receipt_sha256": ordered_blocks[0][
            "generator_receipt_sha256"
        ],
        "causal_tangent_natural_direct_feature_gram": _deep_thaw(
            natural_tangent_receipt
        ),
        "causal_tangent_natural_direct_feature_gram_sha256": (
            natural_tangent_receipt["result_sha256"]
        ),
        "causal_tangent_natural_aggregate_gram_trace": natural_tangent_trace,
        "causal_tangent_aggregate_scale": tangent_family_scale,
        "empirical_to_aggregate_tangent_energy_balance_applied": balance_applied,
        "balance_policy": (
            "preserve-relative-joint-and-prefix-covariance-then-match-total-"
            "tangent-trace-to-empirical-residual-output-trace"
        ),
        "causal_tangent_balanced_direct_feature_gram": _deep_thaw(
            balanced_tangent_receipt
        ),
        "causal_tangent_balanced_direct_feature_gram_sha256": (
            balanced_tangent_receipt["result_sha256"]
        ),
        "combined_direct_feature_gram": _deep_thaw(combined_receipt),
        "combined_direct_feature_gram_sha256": combined_receipt["result_sha256"],
        "combined_feature_row_count": int(combined_factor.shape[0]),
        "combined_gram_expected_sha256": _tensor_sha256(
            expected_combined_gram,
            label="expected combined residual-output functional Gram",
        ),
        "combined_gram_exactly_equals_empirical_plus_scaled_tangent": (
            combined_gram_exact
        ),
        "combined_gram_within_fp64_roundoff": combined_gram_within_roundoff,
        "combined_gram_maximum_absolute_error": combined_gram_maximum_error,
        "combined_gram_roundoff_tolerance": combined_gram_roundoff_tolerance,
        "covariance_eigendecomposition_or_refactor_performed": False,
        "positive_finite_feature_modes_removed": 0,
        "original_weighted_empirical_and_tangent_features_feed_group_pod": True,
        "heldout_rows_used": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "numeric_diagnostics_have_semantic_decision_authority": False,
        "whole_response_intelligence_evaluation_is_primary": True,
    }
    return (
        combined_factor,
        combined_weights,
        MappingProxyType({**body, "result_sha256": canonical_sha256(body)}),
    )


def _relative_rmse(prediction: Any, target: Any) -> float:
    torch = _torch()
    numerator = torch.mean((prediction - target).square())
    denominator = torch.mean(target.square()).clamp_min(torch.finfo(torch.float64).tiny)
    return float(torch.sqrt(numerator / denominator).item())


def _rank_deficit_report(
    *,
    source_state_sha256: str,
    target_template_sha256: str,
    decoder_sha256: str,
    chart_sha256: str,
    context_evidence_sha256: str,
    snapshot_evidence_sha256: str,
    source_state_sha256_after: str,
    weighted_group_features: Any,
    singular_values: Any,
    observed_rank: int,
    required_rank: int,
    rank_tolerance: float,
    source_groups: int,
    fit_row_count: int,
    structural_row_count: int,
    compute_execution: Mapping[str, Any],
    functional_pod_group_metric: str,
    functional_pod_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status = (
        "insufficient-authenticated-residual-output-functional-group-rank"
        if functional_pod_group_metric == _RESIDUAL_OUTPUT_BALANCED_GROUP_METRIC
        else "insufficient-authenticated-context-group-rank"
    )
    body = {
        "schema": LFM2_ATTENTION_GROUP_QDEIM_RANK_DEFICIT_SCHEMA,
        "status": status,
        "source_state_sha256_before": source_state_sha256,
        "source_state_sha256_after": source_state_sha256_after,
        "source_state_immutable": source_state_sha256_after == source_state_sha256,
        "target_template_sha256": target_template_sha256,
        "normalized_input_decoder_sha256": decoder_sha256,
        "residual_output_chart_sha256": chart_sha256,
        "fit_context_evidence_sha256": context_evidence_sha256,
        "functional_snapshot_evidence_sha256": snapshot_evidence_sha256,
        "functional_pod_group_metric": functional_pod_group_metric,
        "functional_pod_receipt": (
            None
            if functional_pod_receipt is None
            else _deep_thaw(functional_pod_receipt)
        ),
        "functional_pod_receipt_sha256": (
            None
            if functional_pod_receipt is None
            else functional_pod_receipt["result_sha256"]
        ),
        "compute_execution": dict(compute_execution),
        "weighted_group_feature_matrix_sha256": _tensor_sha256(
            weighted_group_features,
            label="weighted GQA-group feature matrix",
        ),
        "context_group_singular_values_sha256": _tensor_sha256(
            singular_values,
            label="context group singular values",
        ),
        "source_key_value_group_count": source_groups,
        "target_key_value_group_count": required_rank,
        "minimum_context_group_rank": required_rank,
        "observed_context_group_rank": observed_rank,
        "context_group_rank_deficit": required_rank - observed_rank,
        "context_group_rank_absolute_tolerance": rank_tolerance,
        "fit_context_row_count": fit_row_count,
        "structural_tangent_context_row_count": structural_row_count,
        "selected_source_key_value_groups": [],
        "group_interpolation_materialized": False,
        "random_rank_completion_permitted": False,
        "heldout_rank_completion_permitted": False,
        "silent_pseudoinverse_completion_permitted": False,
        "numeric_diagnostics_have_semantic_decision_authority": False,
        "whole_response_intelligence_evaluation_is_primary": True,
        "binary_manual_acceptance_gate_required": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _candidate_state_sha256(q: Any, k: Any, v: Any, o: Any) -> str:
    return _state_sha256(
        {
            "q_proj.weight": q,
            "k_proj.weight": k,
            "v_proj.weight": v,
            "out_proj.weight": o,
        },
        schema="neural-forge-lfm2-attention-group-qdeim-candidate-state-v1",
    )


def compile_lfm2_attention_group_qdeim(
    *,
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_out_weight: Any,
    expected_source_state_sha256: str,
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    target_value_weight_template: Any,
    target_out_weight_template: Any,
    expected_target_template_sha256: str,
    source_query_head_count: int,
    source_key_value_head_count: int,
    target_query_head_count: int,
    target_key_value_head_count: int,
    normalized_input_decoder: Any,
    expected_normalized_input_decoder_sha256: str,
    residual_output_chart: Any,
    expected_residual_output_chart_sha256: str,
    parent_post_softmax_context_rows: Any,
    row_sample_weights: Sequence[float],
    expected_context_evidence_sha256: str,
    functional_snapshot_blocks: Sequence[Lfm2AttentionGroupQdeimSnapshotBlock] = (),
    expected_functional_snapshot_evidence_sha256: str | None = None,
    functional_pod_group_metric: str = _RAW_CONTEXT_GROUP_METRIC,
    pod_rank_relative_tolerance: float | None = None,
    compute_device: str = "cpu",
) -> Lfm2CompiledAttentionGroupQdeim:
    """Compile one dense intact-GQA-group attention candidate without training.

    The current LFM2 use is 16 Q / 8 KV / 64 dimensions folded to 10 Q / 5 KV
    / 64 dimensions.  None of those numbers are hardcoded: all topology and
    tensor relationships are derived from the supplied head counts and shapes.
    """

    torch = _torch()
    resolved_compute_device = _resolve_qdeim_compute_device(torch, compute_device)
    compute_execution = _qdeim_compute_receipt(
        torch,
        compute_device,
        resolved_compute_device,
    )
    source_q_heads = _positive_int(
        source_query_head_count,
        label="source_query_head_count",
    )
    source_kv_heads = _positive_int(
        source_key_value_head_count,
        label="source_key_value_head_count",
    )
    target_q_heads = _positive_int(
        target_query_head_count,
        label="target_query_head_count",
    )
    target_kv_heads = _positive_int(
        target_key_value_head_count,
        label="target_key_value_head_count",
    )
    if source_q_heads % source_kv_heads or target_q_heads % target_kv_heads:
        raise Lfm2AttentionGroupQdeimError(
            "Source and target query-head counts must divide exactly into GQA groups."
        )
    source_q_per_kv = source_q_heads // source_kv_heads
    target_q_per_kv = target_q_heads // target_kv_heads
    if source_q_per_kv != target_q_per_kv:
        raise Lfm2AttentionGroupQdeimError(
            "Source and target must preserve the same queries-per-KV GQA ratio."
        )
    if target_kv_heads > source_kv_heads:
        raise Lfm2AttentionGroupQdeimError(
            "Target KV-group count cannot exceed source KV-group count."
        )
    if pod_rank_relative_tolerance is not None:
        pod_rank_relative_tolerance = _nonnegative_float(
            pod_rank_relative_tolerance,
            label="pod_rank_relative_tolerance",
        )
    functional_pod_group_metric = _nonempty_token(
        functional_pod_group_metric,
        label="functional_pod_group_metric",
    )
    if functional_pod_group_metric not in _ALLOWED_FUNCTIONAL_POD_GROUP_METRICS:
        raise Lfm2AttentionGroupQdeimError(
            "functional_pod_group_metric is not one supported authenticated group metric."
        )

    expected_source = _digest(
        expected_source_state_sha256,
        label="expected_source_state_sha256",
    )
    expected_template = _digest(
        expected_target_template_sha256,
        label="expected_target_template_sha256",
    )
    expected_decoder = _digest(
        expected_normalized_input_decoder_sha256,
        label="expected_normalized_input_decoder_sha256",
    )
    expected_chart = _digest(
        expected_residual_output_chart_sha256,
        label="expected_residual_output_chart_sha256",
    )
    expected_context = _digest(
        expected_context_evidence_sha256,
        label="expected_context_evidence_sha256",
    )

    blocks = tuple(functional_snapshot_blocks)
    normalized_blocks = _normalized_snapshot_blocks(blocks)
    snapshot_sha256 = lfm2_attention_group_qdeim_snapshot_evidence_sha256(blocks)
    if expected_functional_snapshot_evidence_sha256 is None:
        if blocks:
            raise Lfm2AttentionGroupQdeimError(
                "Non-empty functional snapshot blocks require their expected evidence digest."
            )
    elif snapshot_sha256 != _digest(
        expected_functional_snapshot_evidence_sha256,
        label="expected_functional_snapshot_evidence_sha256",
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Functional structural/tangent snapshot authentication failed."
        )

    source_before = lfm2_attention_group_qdeim_source_state_sha256(
        source_query_weight,
        source_key_weight,
        source_value_weight,
        source_out_weight,
    )
    if source_before != expected_source:
        raise Lfm2AttentionGroupQdeimError("Source Q/K/V/O authentication failed.")
    template_sha256 = lfm2_attention_group_qdeim_target_template_sha256(
        target_query_weight_template,
        target_key_weight_template,
        target_value_weight_template,
        target_out_weight_template,
    )
    if template_sha256 != expected_template:
        raise Lfm2AttentionGroupQdeimError(
            "Blank target attention template authentication failed."
        )
    decoder_sha256 = _tensor_sha256(
        normalized_input_decoder,
        label="normalized_input_decoder",
    )
    chart_sha256 = _tensor_sha256(
        residual_output_chart,
        label="residual_output_chart",
    )
    if decoder_sha256 != expected_decoder:
        raise Lfm2AttentionGroupQdeimError(
            "Normalized-input decoder authentication failed."
        )
    if chart_sha256 != expected_chart:
        raise Lfm2AttentionGroupQdeimError(
            "Residual output chart authentication failed."
        )
    context_sha256 = lfm2_attention_group_qdeim_context_evidence_sha256(
        parent_post_softmax_context_rows,
        row_sample_weights,
    )
    if context_sha256 != expected_context:
        raise Lfm2AttentionGroupQdeimError(
            "Cached parent context evidence authentication failed."
        )

    source_q = _float_tensor(
        source_query_weight,
        label="source_query_weight",
        ndim=2,
    )
    source_k = _float_tensor(
        source_key_weight,
        label="source_key_weight",
        ndim=2,
    )
    source_v = _float_tensor(
        source_value_weight,
        label="source_value_weight",
        ndim=2,
    )
    source_o = _float_tensor(
        source_out_weight,
        label="source_out_weight",
        ndim=2,
    )
    decoder = _float_tensor(
        normalized_input_decoder,
        label="normalized_input_decoder",
        ndim=2,
    )
    chart = _float_tensor(
        residual_output_chart,
        label="residual_output_chart",
        ndim=2,
    )
    contexts = _float_tensor(
        parent_post_softmax_context_rows,
        label="parent_post_softmax_context_rows",
        ndim=3,
    )
    weights = _row_weights(
        row_sample_weights,
        row_count=int(contexts.shape[0]),
        label="row_sample_weights",
    )

    source_query_rows, source_hidden = map(int, source_q.shape)
    if source_query_rows % source_q_heads:
        raise Lfm2AttentionGroupQdeimError(
            "Source Q rows are not divisible by source query-head count."
        )
    head_dim = source_query_rows // source_q_heads
    expected_source_kv_rows = source_kv_heads * head_dim
    if (
        tuple(source_k.shape) != (expected_source_kv_rows, source_hidden)
        or tuple(source_v.shape) != tuple(source_k.shape)
        or tuple(source_o.shape) != (source_hidden, source_query_rows)
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Source Q/K/V/O shapes do not form the declared GQA topology."
        )

    target_spec = _target_template_spec(
        target_query_weight_template,
        target_key_weight_template,
        target_value_weight_template,
        target_out_weight_template,
    )
    target_q_shape = tuple(target_spec["q_proj.weight"]["shape"])
    target_k_shape = tuple(target_spec["k_proj.weight"]["shape"])
    target_v_shape = tuple(target_spec["v_proj.weight"]["shape"])
    target_o_shape = tuple(target_spec["out_proj.weight"]["shape"])
    target_query_rows = target_q_heads * head_dim
    target_key_value_rows = target_kv_heads * head_dim
    target_hidden = int(target_q_shape[1])
    if (
        target_q_shape != (target_query_rows, target_hidden)
        or target_k_shape != (target_key_value_rows, target_hidden)
        or target_v_shape != target_k_shape
        or target_o_shape != (target_hidden, target_query_rows)
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Target BF16 Q/K/V/O templates do not form the declared same-head-dimension GQA topology."
        )
    if tuple(decoder.shape) != (target_hidden, source_hidden):
        raise Lfm2AttentionGroupQdeimError(
            "normalized_input_decoder must have shape [target_hidden, source_hidden]."
        )
    if tuple(chart.shape) != (source_hidden, target_hidden):
        raise Lfm2AttentionGroupQdeimError(
            "residual_output_chart must have shape [source_hidden, target_hidden]."
        )
    if tuple(contexts.shape[1:]) != (source_q_heads, head_dim):
        raise Lfm2AttentionGroupQdeimError(
            "Parent post-softmax contexts must have shape [rows, source_q_heads, head_dim]."
        )
    for block in normalized_blocks:
        if tuple(block["context"].shape[1:]) != (source_q_heads, head_dim):
            raise Lfm2AttentionGroupQdeimError(
                f"Snapshot block {block['block_id']!r} does not match source GQA context shape."
            )

    with torch.no_grad():
        # CPU tensors above are the immutable authenticated archives.  Only
        # one layer's active FP64 algebra is staged, so system memory remains
        # the warm backing store while a selected GPU does the dense work.
        source_q_work = source_q.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        source_k_work = source_k.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        source_v_work = source_v.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        source_o_work = source_o.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        decoder_work = decoder.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        chart_work = chart.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        contexts_work = contexts.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        fit_group_features = _group_feature_matrix(
            contexts_work,
            key_value_head_count=source_kv_heads,
            queries_per_key_value=source_q_per_kv,
        )
        functional_pod_receipt: Mapping[str, Any] | None = None
        if functional_pod_group_metric == _RAW_CONTEXT_GROUP_METRIC:
            fit_feature_weights = torch.tensor(
                weights,
                dtype=torch.float64,
                device=resolved_compute_device,
            ).repeat_interleave(source_q_per_kv * head_dim)
            all_features = [fit_group_features]
            all_feature_weights = [fit_feature_weights]
            for block in normalized_blocks:
                all_features.append(
                    _group_feature_matrix(
                        block["context"].to(
                            device=resolved_compute_device,
                            dtype=torch.float64,
                        ),
                        key_value_head_count=source_kv_heads,
                        queries_per_key_value=source_q_per_kv,
                    )
                )
                all_feature_weights.append(
                    (
                        torch.tensor(
                            block["row_sample_weights"],
                            dtype=torch.float64,
                            device=resolved_compute_device,
                        )
                        * block["block_weight"]
                    ).repeat_interleave(source_q_per_kv * head_dim)
                )
            group_features = torch.cat(all_features, dim=0).contiguous()
            feature_weights = torch.cat(all_feature_weights, dim=0).contiguous()
            weighted_group_features = (
                group_features * torch.sqrt(feature_weights).unsqueeze(1)
            ).contiguous()
        else:
            (
                group_features,
                feature_weights,
                functional_pod_receipt,
            ) = _residual_output_balanced_functional_pod(
                contexts=contexts_work,
                fit_row_weights=weights,
                normalized_blocks=normalized_blocks,
                source_out_weight=source_o_work,
                residual_output_chart=chart_work,
                key_value_head_count=source_kv_heads,
                queries_per_key_value=source_q_per_kv,
            )
            weighted_group_features = group_features

        _left, singular_values, right_h = torch.linalg.svd(
            weighted_group_features,
            full_matrices=False,
        )
        singular_maximum = (
            float(singular_values.max().item()) if singular_values.numel() else 0.0
        )
        default_rank_tolerance = (
            max(weighted_group_features.shape)
            * torch.finfo(torch.float64).eps
            * singular_maximum
        )
        rank_tolerance = (
            default_rank_tolerance
            if pod_rank_relative_tolerance is None
            else pod_rank_relative_tolerance * singular_maximum
        )
        observed_rank = int((singular_values > rank_tolerance).sum().item())
        if observed_rank < target_kv_heads:
            source_after_deficit = lfm2_attention_group_qdeim_source_state_sha256(
                source_query_weight,
                source_key_weight,
                source_value_weight,
                source_out_weight,
            )
            if source_after_deficit != source_before:
                raise Lfm2AttentionGroupQdeimError(
                    "Source state changed while computing the authenticated rank deficit."
                )
            raise Lfm2AttentionGroupQdeimInsufficientRankError(
                _rank_deficit_report(
                    source_state_sha256=source_before,
                    source_state_sha256_after=source_after_deficit,
                    target_template_sha256=template_sha256,
                    decoder_sha256=decoder_sha256,
                    chart_sha256=chart_sha256,
                    context_evidence_sha256=context_sha256,
                    snapshot_evidence_sha256=snapshot_sha256,
                    weighted_group_features=weighted_group_features,
                    singular_values=singular_values,
                    observed_rank=observed_rank,
                    required_rank=target_kv_heads,
                    rank_tolerance=rank_tolerance,
                    source_groups=source_kv_heads,
                    fit_row_count=int(contexts.shape[0]),
                    structural_row_count=sum(
                        int(block["context"].shape[0]) for block in normalized_blocks
                    ),
                    compute_execution=compute_execution,
                    functional_pod_group_metric=functional_pod_group_metric,
                    functional_pod_receipt=functional_pod_receipt,
                )
            )

        basis = _canonicalize_pod_signs(right_h[:target_kv_heads].T.contiguous())
        selected_groups, pivot_residual_norms = _deterministic_block_pivots(basis)
        sampled_basis = basis[list(selected_groups), :].contiguous()
        interpolation_singular_values = torch.linalg.svdvals(sampled_basis)
        interpolation_maximum = float(interpolation_singular_values.max().item())
        interpolation_minimum = float(interpolation_singular_values.min().item())
        interpolation_tolerance = (
            max(sampled_basis.shape)
            * torch.finfo(torch.float64).eps
            * interpolation_maximum
        )
        if (
            not math.isfinite(interpolation_minimum)
            or interpolation_minimum <= interpolation_tolerance
        ):
            raise Lfm2AttentionGroupQdeimError(
                "Selected Q-DEIM group basis is structurally singular."
            )
        group_interpolation = torch.linalg.solve(
            sampled_basis.T,
            basis.T,
        ).contiguous()
        selected_identity = group_interpolation[:, list(selected_groups)]
        selected_identity_error = float(
            (
                selected_identity
                - torch.eye(
                    target_kv_heads,
                    dtype=torch.float64,
                    device=resolved_compute_device,
                )
            )
            .abs()
            .max()
            .item()
        )
        identity_tolerance = (
            512.0
            * torch.finfo(torch.float64).eps
            * max(
                1.0,
                interpolation_maximum / interpolation_minimum,
            )
        )
        if selected_identity_error > identity_tolerance:
            raise Lfm2AttentionGroupQdeimError(
                "Selected-group interpolation failed its structural identity invariant."
            )

        group_block_width = source_q_per_kv * head_dim
        context_reconstruction = torch.kron(
            group_interpolation,
            torch.eye(
                group_block_width,
                dtype=torch.float64,
                device=resolved_compute_device,
            ),
        ).contiguous()
        reconstructed_group_features = (
            group_features[:, list(selected_groups)] @ group_interpolation
        )
        weighted_reconstruction_relative_rmse = _relative_rmse(
            reconstructed_group_features * torch.sqrt(feature_weights).unsqueeze(1),
            weighted_group_features,
        )
        fit_reconstruction_relative_rmse = _relative_rmse(
            fit_group_features[:, list(selected_groups)] @ group_interpolation,
            fit_group_features,
        )

        selected_query_heads = tuple(
            group * source_q_per_kv + query_offset
            for group in selected_groups
            for query_offset in range(source_q_per_kv)
        )
        selected_query_rows = tuple(
            head * head_dim + coordinate
            for head in selected_query_heads
            for coordinate in range(head_dim)
        )
        selected_key_value_rows = tuple(
            group * head_dim + coordinate
            for group in selected_groups
            for coordinate in range(head_dim)
        )
        query_fp64 = (
            source_q_work[list(selected_query_rows), :] @ decoder_work.T
        ).contiguous()
        key_fp64 = (
            source_k_work[list(selected_key_value_rows), :] @ decoder_work.T
        ).contiguous()
        value_fp64 = (
            source_v_work[list(selected_key_value_rows), :] @ decoder_work.T
        ).contiguous()
        mapped_parent_out = (chart_work.T @ source_o_work).contiguous()
        out_fp64 = (mapped_parent_out @ context_reconstruction.T).contiguous()

        query = query_fp64.to(device="cpu", dtype=torch.bfloat16).contiguous()
        key = key_fp64.to(device="cpu", dtype=torch.bfloat16).contiguous()
        value = value_fp64.to(device="cpu", dtype=torch.bfloat16).contiguous()
        out = out_fp64.to(device="cpu", dtype=torch.bfloat16).contiguous()

    source_after = lfm2_attention_group_qdeim_source_state_sha256(
        source_query_weight,
        source_key_weight,
        source_value_weight,
        source_out_weight,
    )
    expected_shapes = {
        "q": (target_query_rows, target_hidden),
        "k": (target_key_value_rows, target_hidden),
        "v": (target_key_value_rows, target_hidden),
        "o": (target_hidden, target_query_rows),
    }
    outputs = {"q": query, "k": key, "v": value, "o": out}
    if (
        source_after != source_before
        or len(selected_groups) != target_kv_heads
        or len(set(selected_groups)) != target_kv_heads
        or len(selected_query_heads) != target_q_heads
        or any(
            tuple(tensor.shape) != expected_shapes[name]
            or tensor.dtype != torch.bfloat16
            or tensor.layout != torch.strided
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not bool(torch.isfinite(tensor).all().item())
            for name, tensor in outputs.items()
        )
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Materialized attention candidate failed dense/BF16/no-grad/source invariants."
        )

    candidate_sha256 = _candidate_state_sha256(query, key, value, out)
    body = {
        "schema": LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
        "status": "materialized-for-holistic-whole-response-intelligence-evaluation",
        "method": "deterministic-block-constrained-qdeim-intact-gqa-producers",
        "compute_execution": compute_execution,
        "topology": {
            "source_hidden_size": source_hidden,
            "target_hidden_size": target_hidden,
            "source_query_head_count": source_q_heads,
            "source_key_value_head_count": source_kv_heads,
            "target_query_head_count": target_q_heads,
            "target_key_value_head_count": target_kv_heads,
            "queries_per_key_value": source_q_per_kv,
            "head_dim": head_dim,
            "source_group_block_width": group_block_width,
        },
        "authentication": {
            "source_state_sha256_before": source_before,
            "source_state_sha256_after": source_after,
            "source_state_immutable": source_before == source_after,
            "target_template_sha256": template_sha256,
            "normalized_input_decoder_sha256": decoder_sha256,
            "residual_output_chart_sha256": chart_sha256,
            "fit_context_evidence_sha256": context_sha256,
            "functional_snapshot_evidence_sha256": snapshot_sha256,
            "functional_pod_group_metric": functional_pod_group_metric,
            "functional_pod_receipt_sha256": (
                None
                if functional_pod_receipt is None
                else functional_pod_receipt["result_sha256"]
            ),
        },
        "evidence": {
            "fit_partition_only": True,
            "heldout_rows_used": False,
            "fit_context_row_count": int(contexts.shape[0]),
            "explicit_positive_fit_row_weights": True,
            "functional_pod_group_metric": functional_pod_group_metric,
            "raw_fit_context_used_directly_for_group_selection": (
                functional_pod_group_metric == _RAW_CONTEXT_GROUP_METRIC
            ),
            "functional_pod_receipt": (
                None
                if functional_pod_receipt is None
                else _deep_thaw(functional_pod_receipt)
            ),
            "functional_pod_receipt_sha256": (
                None
                if functional_pod_receipt is None
                else functional_pod_receipt["result_sha256"]
            ),
            "structural_tangent_snapshot_block_count": len(normalized_blocks),
            "structural_tangent_context_row_count": sum(
                int(block["context"].shape[0]) for block in normalized_blocks
            ),
            "snapshot_blocks": [
                {
                    "block_id": block["block_id"],
                    "snapshot_kind": block["snapshot_kind"],
                    "perturbation_role": block["perturbation_role"],
                    "perturbation_receipt_sha256": block["perturbation_receipt_sha256"],
                    "generator_receipt_sha256": block["generator_receipt_sha256"],
                    "context_row_count": int(block["context"].shape[0]),
                    "block_weight": block["block_weight"],
                }
                for block in normalized_blocks
            ],
        },
        "selection": {
            "selected_source_key_value_groups": list(selected_groups),
            "selected_source_query_heads": list(selected_query_heads),
            "whole_gqa_groups_only": True,
            "q_heads_per_selected_group": source_q_per_kv,
            "k_heads_per_selected_group": 1,
            "v_heads_per_selected_group": 1,
            "qkv_producer_averaging_performed": False,
            "qkv_producer_rows_are_single_source_copies_through_decoder": True,
            "selection_algorithm": (
                "deterministic-column-pivoted-modified-qr-of-group-pod-transpose"
            ),
            "pivot_residual_norms_sha256": _tensor_sha256(
                pivot_residual_norms,
                label="block pivot residual norms",
            ),
        },
        "interpolation": {
            "observed_context_group_rank": observed_rank,
            "minimum_context_group_rank": target_kv_heads,
            "context_group_rank_absolute_tolerance": rank_tolerance,
            "context_group_singular_values_sha256": _tensor_sha256(
                singular_values,
                label="context group singular values",
            ),
            "group_pod_basis_sha256": _tensor_sha256(
                basis,
                label="GQA-group POD basis",
            ),
            "group_interpolation_sha256": _tensor_sha256(
                group_interpolation,
                label="GQA-group interpolation",
            ),
            "context_reconstruction_sha256": _tensor_sha256(
                context_reconstruction,
                label="full context reconstruction",
            ),
            "group_interpolation_shape": list(group_interpolation.shape),
            "context_reconstruction_shape": list(context_reconstruction.shape),
            "selected_group_identity_max_abs_error": selected_identity_error,
            "selected_group_identity_tolerance": identity_tolerance,
            "weighted_context_reconstruction_relative_rmse": (
                weighted_reconstruction_relative_rmse
            ),
            "fit_context_reconstruction_relative_rmse": (
                fit_reconstruction_relative_rmse
            ),
            "sampled_basis_minimum_singular_value": interpolation_minimum,
            "sampled_basis_maximum_singular_value": interpolation_maximum,
            "numeric_diagnostics_are_advisory": True,
            "numeric_diagnostics_have_semantic_decision_authority": False,
        },
        "materialization": {
            "q_formula": "source_q[selected_query_rows,:] @ decoder.T",
            "k_formula": "source_k[selected_kv_rows,:] @ decoder.T",
            "v_formula": "source_v[selected_kv_rows,:] @ decoder.T",
            "o_formula": (
                "residual_chart.T @ source_o @ "
                "kron(group_interpolation, I_group_block).T"
            ),
            "dense_all_parent_output_columns_used": True,
            "pre_softmax_qkv_averaging": False,
            "candidate_state_sha256": candidate_sha256,
            "candidate_dtype": "bfloat16",
            "standalone_dense_tensors": True,
            "decoder_shape": list(decoder.shape),
            "residual_chart_shape": list(chart.shape),
            "decoder_and_chart_numeric_rank_not_used_as_gate": True,
        },
        "semantic_policy": {
            "numeric_threshold_auto_accept_permitted": False,
            "numeric_threshold_auto_veto_permitted": False,
            "automatic_semantic_publication_permitted": False,
            "whole_response_intelligence_evaluation_is_primary": True,
            "binary_manual_acceptance_gate_required": False,
            "retention_targets_are_objectives_not_cutoffs": True,
        },
        "prohibited_operations": {
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "pruning_used": False,
            "quantization_used": False,
            "sparsity_used": False,
            "parameter_sharing_used": False,
            "numeric_parent_child_weight_equality_required": False,
        },
    }
    report = {**body, "result_sha256": canonical_sha256(body)}
    return Lfm2CompiledAttentionGroupQdeim(
        selected_source_key_value_groups=selected_groups,
        selected_source_query_heads=selected_query_heads,
        query_weight=query,
        key_weight=key,
        value_weight=value,
        out_weight=out,
        group_pod_basis=basis.detach().cpu().contiguous().clone(),
        group_interpolation=group_interpolation.detach().cpu().contiguous().clone(),
        context_reconstruction=(
            context_reconstruction.detach().cpu().contiguous().clone()
        ),
        normalized_input_decoder=decoder.detach().cpu().contiguous().clone(),
        residual_output_chart=chart.detach().cpu().contiguous().clone(),
        report=MappingProxyType(report),
    )


def _ordered_lfm2_attention_group_qdeim_span_sources(
    span: Lfm2ModelFoldSpan,
    sources: Sequence[Lfm2AttentionGroupQdeimSpanSource],
) -> tuple[Lfm2AttentionGroupQdeimSpanSource, ...]:
    if not isinstance(span, Lfm2ModelFoldSpan):
        raise Lfm2AttentionGroupQdeimError(
            "span must be one existing Lfm2ModelFoldSpan."
        )
    if isinstance(sources, (str, bytes)):
        raise Lfm2AttentionGroupQdeimError(
            "sources must contain authenticated attention-layer payloads."
        )
    try:
        ordered = tuple(sources)
    except TypeError as exc:
        raise Lfm2AttentionGroupQdeimError(
            "sources must contain authenticated attention-layer payloads."
        ) from exc
    if not ordered or any(
        not isinstance(source, Lfm2AttentionGroupQdeimSpanSource) for source in ordered
    ):
        raise Lfm2AttentionGroupQdeimError(
            "sources must contain authenticated attention-layer payloads."
        )
    indices = tuple(source.source_layer_index for source in ordered)
    span_indices = set(span.source_layer_indices)
    if (
        len(set(indices)) != len(indices)
        or any(index not in span_indices for index in indices)
        or indices
        != tuple(index for index in span.source_layer_indices if index in set(indices))
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Attention sources must be one non-empty ordered subset of the existing "
            "model-fold span."
        )
    return ordered


def assemble_lfm2_attention_group_qdeim_span_sources(
    *,
    span: Lfm2ModelFoldSpan,
    attention_source_layer_indices: Sequence[int],
    sources_by_layer: Mapping[int, Lfm2AttentionGroupQdeimSpanSource],
) -> tuple[Lfm2AttentionGroupQdeimSpanSource, ...]:
    """Assemble the semantic stack's typed attention subset for one span.

    Depth planning remains solely owned by :class:`Lfm2ModelFoldSpan`.  The
    semantic layer stack declares which members are attention-bearing and
    supplies their already-cached payloads.  This seam accepts no model or
    cache reader, so it cannot recapture a parent or invent another span.
    """

    if not isinstance(span, Lfm2ModelFoldSpan):
        raise Lfm2AttentionGroupQdeimError(
            "span must be one existing Lfm2ModelFoldSpan."
        )
    if isinstance(attention_source_layer_indices, (str, bytes)):
        raise Lfm2AttentionGroupQdeimError(
            "attention_source_layer_indices must be an ordered integer sequence."
        )
    try:
        indices = tuple(attention_source_layer_indices)
    except TypeError as exc:
        raise Lfm2AttentionGroupQdeimError(
            "attention_source_layer_indices must be an ordered integer sequence."
        ) from exc
    if (
        not indices
        or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        )
        or len(set(indices)) != len(indices)
        or indices
        != tuple(index for index in span.source_layer_indices if index in set(indices))
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Attention layer indices must be one non-empty ordered subset of the span."
        )
    if not isinstance(sources_by_layer, Mapping) or set(sources_by_layer) != set(
        indices
    ):
        raise Lfm2AttentionGroupQdeimError(
            "sources_by_layer must exactly cover the declared attention layers."
        )
    ordered = tuple(sources_by_layer[index] for index in indices)
    if any(
        not isinstance(source, Lfm2AttentionGroupQdeimSpanSource)
        or source.source_layer_index != layer_index
        for layer_index, source in zip(indices, ordered, strict=True)
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Every attention payload must bind its declared source layer."
        )
    return _ordered_lfm2_attention_group_qdeim_span_sources(span, ordered)


def _span_attention_row_weights(
    row_count: int,
    supplied: Sequence[float] | None,
) -> tuple[float, ...]:
    if supplied is None:
        return tuple(1.0 for _ in range(row_count))
    return _row_weights(
        supplied,
        row_count=row_count,
        label="row_sample_weights",
    )


def lfm2_attention_group_qdeim_span_context_evidence_sha256(
    *,
    span: Lfm2ModelFoldSpan,
    sources: Sequence[Lfm2AttentionGroupQdeimSpanSource],
    row_sample_weights: Sequence[float] | None = None,
) -> str:
    """Root complete attention sources and their opaque receipt digests.

    Q/K/V/O plus optional Q/K norm state, topology, cached context, decoder,
    chart, every paired expected digest, and receipt/row identities enter this
    external root.  The receipt contents remain opaque and require upstream
    authentication before this typed compiler seam is used.
    """

    ordered = _ordered_lfm2_attention_group_qdeim_span_sources(span, sources)
    contexts = tuple(
        _float_tensor(
            source.cached_parent_post_softmax_context_rows,
            label=(
                f"source layer {source.source_layer_index} cached parent "
                "post-softmax context rows"
            ),
            ndim=3,
        )
        for source in ordered
    )
    row_count = int(contexts[0].shape[0])
    if any(int(context.shape[0]) != row_count for context in contexts):
        raise Lfm2AttentionGroupQdeimError(
            "All span attention sources must contain the same aligned row count."
        )
    row_identity = ordered[0].aligned_row_identity_sha256
    if any(source.aligned_row_identity_sha256 != row_identity for source in ordered):
        raise Lfm2AttentionGroupQdeimError(
            "All span attention sources must share one authenticated row identity."
        )
    weights = _span_attention_row_weights(row_count, row_sample_weights)
    return canonical_sha256(
        {
            "schema": LFM2_ATTENTION_GROUP_SPAN_QDEIM_CONTEXT_EVIDENCE_SCHEMA,
            "span": span.as_dict(),
            "root_scope": (
                "complete_qkvo_optional_qk_norm_context_decoder_chart_topology_"
                "expected_hashes_and_opaque_receipt_bindings"
            ),
            "attention_source_layer_indices": [
                source.source_layer_index for source in ordered
            ],
            "aligned_row_count": row_count,
            "aligned_row_identity_sha256": row_identity,
            "row_sample_weights": list(weights),
            "sources": [
                {
                    "source_layer_index": source.source_layer_index,
                    "source_query_head_count": source.source_query_head_count,
                    "source_key_value_head_count": (source.source_key_value_head_count),
                    "source_state_sha256": (
                        lfm2_attention_group_qdeim_span_source_state_sha256(
                            source.source_query_weight,
                            source.source_key_weight,
                            source.source_value_weight,
                            source.source_out_weight,
                            source.source_query_norm_weight,
                            source.source_key_norm_weight,
                        )
                    ),
                    "expected_source_state_sha256": (
                        source.expected_source_state_sha256
                    ),
                    "qk_norm_state": {
                        "present": source.source_query_norm_weight is not None,
                        **(
                            {
                                "query_norm_sha256": _tensor_sha256(
                                    source.source_query_norm_weight,
                                    label=(
                                        f"source layer {source.source_layer_index} "
                                        "query norm"
                                    ),
                                ),
                                "key_norm_sha256": _tensor_sha256(
                                    source.source_key_norm_weight,
                                    label=(
                                        f"source layer {source.source_layer_index} "
                                        "key norm"
                                    ),
                                ),
                            }
                            if source.source_query_norm_weight is not None
                            else {}
                        ),
                    },
                    "context_rows_sha256": _tensor_sha256(
                        source.cached_parent_post_softmax_context_rows,
                        label=(
                            f"source layer {source.source_layer_index} cached "
                            "post-softmax context rows"
                        ),
                    ),
                    "expected_context_rows_sha256": (
                        source.expected_cached_parent_post_softmax_context_sha256
                    ),
                    "normalized_input_decoder_sha256": _tensor_sha256(
                        source.normalized_input_decoder,
                        label=(
                            f"source layer {source.source_layer_index} "
                            "normalized-input decoder"
                        ),
                    ),
                    "expected_normalized_input_decoder_sha256": (
                        source.expected_normalized_input_decoder_sha256
                    ),
                    "residual_output_chart_sha256": _tensor_sha256(
                        source.residual_output_chart,
                        label=(
                            f"source layer {source.source_layer_index} "
                            "residual output chart"
                        ),
                    ),
                    "expected_residual_output_chart_sha256": (
                        source.expected_residual_output_chart_sha256
                    ),
                    "cache_receipt_sha256": source.cache_receipt_sha256,
                    "aligned_row_identity_sha256": (source.aligned_row_identity_sha256),
                }
                for source in ordered
            ],
            "upstream_cache_receipts_are_opaque_content_addresses": True,
            "upstream_cache_receipt_verification_still_required": True,
        }
    )


def _all_tensor_rows_identical(value: Any) -> bool:
    torch = _torch()
    if int(value.shape[0]) <= 1:
        return True
    return bool(torch.equal(value[1:], value[0].unsqueeze(0).expand_as(value[1:])))


def compile_lfm2_attention_group_span_qdeim(
    *,
    span: Lfm2ModelFoldSpan,
    sources: Sequence[Lfm2AttentionGroupQdeimSpanSource],
    expected_context_evidence_sha256: str,
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    target_value_weight_template: Any,
    target_out_weight_template: Any,
    expected_target_template_sha256: str,
    target_query_head_count: int,
    target_key_value_head_count: int,
    row_sample_weights: Sequence[float] | None = None,
    pod_rank_relative_tolerance: float | None = None,
    compute_device: str = "cpu",
) -> Lfm2CompiledSpanAttentionGroupQdeim:
    """Fold every attention layer in one depth span into one dense child.

    A single POD/Q-DEIM basis is built over the union of aligned cached
    post-softmax complete-GQA-group contexts.  Selection retains at least one
    real intact group from every contributing attention layer.  Its Q/K/V rows
    pass through that layer's own normalized-input decoder.  The child output
    writer is the sum of every source layer's complete O projection transported
    by that layer's output chart and slice of the joint reconstruction.

    This is only the authenticated algebra primitive.  It neither verifies the
    contents behind opaque upstream cache-receipt digests nor installs the
    result into a production semantic stack; both remain required before any
    decoded-behavior claim.
    """

    torch = _torch()
    ordered = _ordered_lfm2_attention_group_qdeim_span_sources(span, sources)
    expected_context = _digest(
        expected_context_evidence_sha256,
        label="expected_context_evidence_sha256",
    )
    expected_template = _digest(
        expected_target_template_sha256,
        label="expected_target_template_sha256",
    )
    target_q_heads = _positive_int(
        target_query_head_count,
        label="target_query_head_count",
    )
    target_kv_heads = _positive_int(
        target_key_value_head_count,
        label="target_key_value_head_count",
    )
    if target_q_heads % target_kv_heads:
        raise Lfm2AttentionGroupQdeimError(
            "Target query heads must divide exactly into complete GQA groups."
        )
    target_q_per_kv = target_q_heads // target_kv_heads
    if target_kv_heads < len(ordered):
        raise Lfm2AttentionGroupQdeimError(
            "The child needs at least one intact GQA group per attention source."
        )
    if pod_rank_relative_tolerance is not None:
        pod_rank_relative_tolerance = _nonnegative_float(
            pod_rank_relative_tolerance,
            label="pod_rank_relative_tolerance",
        )
    resolved_compute_device = _resolve_qdeim_compute_device(torch, compute_device)
    compute_execution = _qdeim_compute_receipt(
        torch,
        compute_device,
        resolved_compute_device,
    )

    template_sha256 = lfm2_attention_group_qdeim_target_template_sha256(
        target_query_weight_template,
        target_key_weight_template,
        target_value_weight_template,
        target_out_weight_template,
    )
    if template_sha256 != expected_template:
        raise Lfm2AttentionGroupQdeimError(
            "Blank target attention template authentication failed."
        )
    target_spec = _target_template_spec(
        target_query_weight_template,
        target_key_weight_template,
        target_value_weight_template,
        target_out_weight_template,
    )
    target_q_shape = tuple(target_spec["q_proj.weight"]["shape"])
    target_k_shape = tuple(target_spec["k_proj.weight"]["shape"])
    target_v_shape = tuple(target_spec["v_proj.weight"]["shape"])
    target_o_shape = tuple(target_spec["out_proj.weight"]["shape"])
    if target_q_shape[0] % target_q_heads:
        raise Lfm2AttentionGroupQdeimError(
            "Target query rows are not divisible by target query heads."
        )
    head_dim = target_q_shape[0] // target_q_heads
    target_hidden = target_q_shape[1]
    target_group_width = target_q_per_kv * head_dim
    if (
        target_k_shape != (target_kv_heads * head_dim, target_hidden)
        or target_v_shape != target_k_shape
        or target_o_shape != (target_hidden, target_q_heads * head_dim)
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Target BF16 templates do not form the declared GQA topology."
        )

    context_evidence_sha256 = lfm2_attention_group_qdeim_span_context_evidence_sha256(
        span=span,
        sources=ordered,
        row_sample_weights=row_sample_weights,
    )
    if context_evidence_sha256 != expected_context:
        raise Lfm2AttentionGroupQdeimError(
            "Authenticated span post-softmax context evidence changed."
        )

    norm_presence = tuple(
        source.source_query_norm_weight is not None for source in ordered
    )
    if any(norm_presence) and not all(norm_presence):
        raise Lfm2AttentionGroupQdeimError(
            "All attention sources must either supply Q/K norm state or omit it."
        )
    norms_present = all(norm_presence)
    row_identity = ordered[0].aligned_row_identity_sha256
    normalized_sources: list[dict[str, Any]] = []
    source_state_before: dict[int, str] = {}
    row_count: int | None = None
    total_source_groups = 0
    source_index_groups: list[tuple[int, ...]] = []
    for source in ordered:
        layer_index = source.source_layer_index
        prefix = f"source layer {layer_index}"
        current_source_state = lfm2_attention_group_qdeim_span_source_state_sha256(
            source.source_query_weight,
            source.source_key_weight,
            source.source_value_weight,
            source.source_out_weight,
            source.source_query_norm_weight,
            source.source_key_norm_weight,
        )
        if current_source_state != source.expected_source_state_sha256:
            raise Lfm2AttentionGroupQdeimError(
                f"Authenticated {prefix} Q/K/V/O/norm state changed."
            )
        source_state_before[layer_index] = current_source_state
        source_q = _float_tensor(
            source.source_query_weight,
            label=f"{prefix} query weight",
            ndim=2,
        )
        source_k = _float_tensor(
            source.source_key_weight,
            label=f"{prefix} key weight",
            ndim=2,
        )
        source_v = _float_tensor(
            source.source_value_weight,
            label=f"{prefix} value weight",
            ndim=2,
        )
        source_o = _float_tensor(
            source.source_out_weight,
            label=f"{prefix} output weight",
            ndim=2,
        )
        source_q_heads = source.source_query_head_count
        source_kv_heads = source.source_key_value_head_count
        if source_q_heads % source_kv_heads:
            raise Lfm2AttentionGroupQdeimError(
                f"{prefix} query heads do not form complete GQA groups."
            )
        source_q_per_kv = source_q_heads // source_kv_heads
        source_query_rows, source_hidden = map(int, source_q.shape)
        if source_query_rows % source_q_heads:
            raise Lfm2AttentionGroupQdeimError(
                f"{prefix} query rows do not divide by its query-head count."
            )
        source_head_dim = source_query_rows // source_q_heads
        if source_q_per_kv != target_q_per_kv or source_head_dim != head_dim:
            raise Lfm2AttentionGroupQdeimError(
                "Every source and target must preserve GQA ratio and RoPE head_dim."
            )
        if (
            tuple(source_k.shape) != (source_kv_heads * head_dim, source_hidden)
            or tuple(source_v.shape) != tuple(source_k.shape)
            or tuple(source_o.shape) != (source_hidden, source_q_heads * head_dim)
        ):
            raise Lfm2AttentionGroupQdeimError(
                f"{prefix} weights do not form its declared GQA topology."
            )
        context = _float_tensor(
            source.cached_parent_post_softmax_context_rows,
            label=f"{prefix} cached parent post-softmax context rows",
            ndim=3,
        )
        if tuple(context.shape[1:]) != (source_q_heads, head_dim):
            raise Lfm2AttentionGroupQdeimError(
                f"{prefix} context rows do not match its GQA topology."
            )
        if row_count is None:
            row_count = int(context.shape[0])
        elif int(context.shape[0]) != row_count:
            raise Lfm2AttentionGroupQdeimError(
                "Every source layer must contain the same aligned context rows."
            )
        context_sha256 = _tensor_sha256(
            source.cached_parent_post_softmax_context_rows,
            label=f"{prefix} cached parent post-softmax context rows",
        )
        if context_sha256 != source.expected_cached_parent_post_softmax_context_sha256:
            raise Lfm2AttentionGroupQdeimError(
                f"Authenticated {prefix} cached context rows changed."
            )
        if source.aligned_row_identity_sha256 != row_identity:
            raise Lfm2AttentionGroupQdeimError(
                "Every source layer must share one authenticated row identity."
            )
        decoder_sha256 = _tensor_sha256(
            source.normalized_input_decoder,
            label=f"{prefix} normalized-input decoder",
        )
        chart_sha256 = _tensor_sha256(
            source.residual_output_chart,
            label=f"{prefix} residual output chart",
        )
        if decoder_sha256 != source.expected_normalized_input_decoder_sha256:
            raise Lfm2AttentionGroupQdeimError(
                f"Authenticated {prefix} normalized-input decoder changed."
            )
        if chart_sha256 != source.expected_residual_output_chart_sha256:
            raise Lfm2AttentionGroupQdeimError(
                f"Authenticated {prefix} residual output chart changed."
            )
        decoder = _float_tensor(
            source.normalized_input_decoder,
            label=f"{prefix} normalized-input decoder",
            ndim=2,
        )
        chart = _float_tensor(
            source.residual_output_chart,
            label=f"{prefix} residual output chart",
            ndim=2,
        )
        if tuple(decoder.shape) != (target_hidden, source_hidden):
            raise Lfm2AttentionGroupQdeimError(
                f"{prefix} decoder must have shape [target_hidden, source_hidden]."
            )
        if tuple(chart.shape) != (source_hidden, target_hidden):
            raise Lfm2AttentionGroupQdeimError(
                f"{prefix} chart must have shape [source_hidden, target_hidden]."
            )
        query_norm = None
        key_norm = None
        if norms_present:
            query_norm = _float_tensor(
                source.source_query_norm_weight,
                label=f"{prefix} query norm weight",
                ndim=1,
            )
            key_norm = _float_tensor(
                source.source_key_norm_weight,
                label=f"{prefix} key norm weight",
                ndim=1,
            )
            if tuple(query_norm.shape) != (head_dim,) or tuple(key_norm.shape) != (
                head_dim,
            ):
                raise Lfm2AttentionGroupQdeimError(
                    f"{prefix} Q/K norm state must preserve RoPE head_dim."
                )
        union_start = total_source_groups
        total_source_groups += source_kv_heads
        union_stop = total_source_groups
        source_index_groups.append(tuple(range(union_start, union_stop)))
        normalized_sources.append(
            {
                "source": source,
                "query": source_q,
                "key": source_k,
                "value": source_v,
                "out": source_o,
                "context": context,
                "decoder": decoder,
                "chart": chart,
                "query_norm": query_norm,
                "key_norm": key_norm,
                "source_hidden": source_hidden,
                "source_q_heads": source_q_heads,
                "source_kv_heads": source_kv_heads,
                "union_start": union_start,
                "union_stop": union_stop,
                "context_sha256": context_sha256,
                "decoder_sha256": decoder_sha256,
                "chart_sha256": chart_sha256,
            }
        )
    assert row_count is not None
    if target_kv_heads >= total_source_groups:
        raise Lfm2AttentionGroupQdeimError(
            "Span attention Q-DEIM must reduce the union GQA-group count."
        )
    row_weights = _span_attention_row_weights(row_count, row_sample_weights)

    with torch.no_grad():
        staged_sources: list[dict[str, Any]] = []
        for value in normalized_sources:
            staged_sources.append(
                {
                    **value,
                    "query_work": value["query"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "key_work": value["key"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "value_work": value["value"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "out_work": value["out"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "context_work": value["context"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "decoder_work": value["decoder"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "chart_work": value["chart"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "query_norm_work": (
                        None
                        if value["query_norm"] is None
                        else value["query_norm"].to(
                            device=resolved_compute_device,
                            dtype=torch.float64,
                        )
                    ),
                    "key_norm_work": (
                        None
                        if value["key_norm"] is None
                        else value["key_norm"].to(
                            device=resolved_compute_device,
                            dtype=torch.float64,
                        )
                    ),
                }
            )
        source_group_features = [
            _group_feature_matrix(
                value["context_work"],
                key_value_head_count=value["source_kv_heads"],
                queries_per_key_value=target_q_per_kv,
            )
            for value in staged_sources
        ]
        union_group_features = torch.cat(source_group_features, dim=1).contiguous()
        feature_weights = torch.tensor(
            row_weights,
            dtype=torch.float64,
            device=resolved_compute_device,
        ).repeat_interleave(target_group_width)
        sqrt_feature_weights = torch.sqrt(feature_weights).unsqueeze(1)
        weighted_group_features = (
            union_group_features * sqrt_feature_weights
        ).contiguous()
        _pod_left, singular_values, right_h = torch.linalg.svd(
            weighted_group_features,
            full_matrices=False,
        )
        singular_maximum = float(singular_values.max().item())
        default_rank_tolerance = (
            max(weighted_group_features.shape)
            * torch.finfo(torch.float64).eps
            * singular_maximum
        )
        rank_tolerance = (
            default_rank_tolerance
            if pod_rank_relative_tolerance is None
            else pod_rank_relative_tolerance * singular_maximum
        )
        observed_rank = int((singular_values > rank_tolerance).sum().item())
        if observed_rank < target_kv_heads:
            source_state_after_deficit = {
                source.source_layer_index: (
                    lfm2_attention_group_qdeim_span_source_state_sha256(
                        source.source_query_weight,
                        source.source_key_weight,
                        source.source_value_weight,
                        source.source_out_weight,
                        source.source_query_norm_weight,
                        source.source_key_norm_weight,
                    )
                )
                for source in ordered
            }
            body = {
                "schema": LFM2_ATTENTION_GROUP_SPAN_QDEIM_RANK_DEFICIT_SCHEMA,
                "status": "insufficient-authenticated-span-context-group-rank",
                "span": span.as_dict(),
                "attention_source_layer_indices": [
                    source.source_layer_index for source in ordered
                ],
                "source_state_sha256_before": source_state_before,
                "source_state_sha256_after": source_state_after_deficit,
                "source_state_immutable": (
                    source_state_after_deficit == source_state_before
                ),
                "context_evidence_sha256": context_evidence_sha256,
                "target_template_sha256": template_sha256,
                "union_source_key_value_group_count": total_source_groups,
                "target_key_value_group_count": target_kv_heads,
                "minimum_context_group_rank": target_kv_heads,
                "observed_context_group_rank": observed_rank,
                "context_group_rank_deficit": target_kv_heads - observed_rank,
                "context_group_rank_absolute_tolerance": rank_tolerance,
                "random_rank_completion_permitted": False,
                "silent_pseudoinverse_completion_permitted": False,
                "compute_execution": compute_execution,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
            }
            raise Lfm2AttentionGroupQdeimInsufficientRankError(
                {**body, "result_sha256": canonical_sha256(body)}
            )

        pod_basis = _canonicalize_pod_signs(right_h[:target_kv_heads].T.contiguous())
        try:
            selected_global, pivot_residual_norms = (
                _mlp_qdeim._deterministic_source_covering_pivoted_qr_indices(
                    pod_basis,
                    source_index_groups,
                )
            )
        except _mlp_qdeim.Lfm2MlpQdeimError as exc:
            raise Lfm2AttentionGroupQdeimError(
                "No deterministic independent intact-group selection covers "
                "every attention source."
            ) from exc
        sampled_basis = pod_basis[list(selected_global), :].contiguous()
        interpolation_singular = torch.linalg.svdvals(sampled_basis)
        interpolation_minimum = float(interpolation_singular.min().item())
        interpolation_maximum = float(interpolation_singular.max().item())
        interpolation_tolerance = (
            max(sampled_basis.shape)
            * torch.finfo(torch.float64).eps
            * interpolation_maximum
        )
        if (
            not math.isfinite(interpolation_minimum)
            or interpolation_minimum <= interpolation_tolerance
        ):
            raise Lfm2AttentionGroupQdeimError(
                "Selected span GQA-group POD basis is structurally singular."
            )
        group_interpolation = torch.linalg.solve(
            sampled_basis.T,
            pod_basis.T,
        ).contiguous()
        selected_identity_error = float(
            (
                group_interpolation[:, list(selected_global)]
                - torch.eye(
                    target_kv_heads,
                    dtype=torch.float64,
                    device=resolved_compute_device,
                )
            )
            .abs()
            .max()
            .item()
        )
        identity_tolerance = (
            512.0
            * torch.finfo(torch.float64).eps
            * max(1.0, interpolation_maximum / interpolation_minimum)
        )
        if selected_identity_error > identity_tolerance:
            raise Lfm2AttentionGroupQdeimError(
                "Selected span group interpolation failed its identity invariant."
            )
        context_reconstruction = torch.kron(
            group_interpolation,
            torch.eye(
                target_group_width,
                dtype=torch.float64,
                device=resolved_compute_device,
            ),
        ).contiguous()
        reconstructed_group_features = (
            union_group_features[:, list(selected_global)] @ group_interpolation
        )
        reconstruction_relative_rmse = _relative_rmse(
            reconstructed_group_features * sqrt_feature_weights,
            weighted_group_features,
        )

        selected_pairs: list[tuple[int, int]] = []
        selected_query_heads: list[tuple[int, int]] = []
        selected_counts = {source.source_layer_index: 0 for source in ordered}
        query_blocks = []
        key_blocks = []
        value_blocks = []
        query_norm_blocks = []
        key_norm_blocks = []
        for global_group in selected_global:
            source_value = next(
                value
                for value in staged_sources
                if value["union_start"] <= global_group < value["union_stop"]
            )
            source = source_value["source"]
            layer_index = source.source_layer_index
            local_group = global_group - source_value["union_start"]
            selected_pairs.append((layer_index, local_group))
            selected_counts[layer_index] += 1
            local_query_heads = tuple(
                local_group * target_q_per_kv + offset
                for offset in range(target_q_per_kv)
            )
            selected_query_heads.extend(
                (layer_index, head) for head in local_query_heads
            )
            query_rows = [
                head * head_dim + coordinate
                for head in local_query_heads
                for coordinate in range(head_dim)
            ]
            key_value_rows = [
                local_group * head_dim + coordinate for coordinate in range(head_dim)
            ]
            query_blocks.append(
                source_value["query_work"][query_rows] @ source_value["decoder_work"].T
            )
            key_blocks.append(
                source_value["key_work"][key_value_rows]
                @ source_value["decoder_work"].T
            )
            value_blocks.append(
                source_value["value_work"][key_value_rows]
                @ source_value["decoder_work"].T
            )
            if norms_present:
                query_norm_blocks.append(
                    source_value["query_norm_work"]
                    .unsqueeze(0)
                    .repeat(target_q_per_kv, 1)
                )
                key_norm_blocks.append(source_value["key_norm_work"].unsqueeze(0))
        query_fp64 = torch.cat(query_blocks, dim=0).contiguous()
        key_fp64 = torch.cat(key_blocks, dim=0).contiguous()
        value_fp64 = torch.cat(value_blocks, dim=0).contiguous()

        writer_contributions = []
        writer_reports = []
        for source_value in staged_sources:
            start = source_value["union_start"]
            stop = source_value["union_stop"]
            source_reconstruction = group_interpolation[:, start:stop].contiguous()
            context_transport = torch.kron(
                source_reconstruction,
                torch.eye(
                    target_group_width,
                    dtype=torch.float64,
                    device=resolved_compute_device,
                ),
            ).contiguous()
            mapped_writer = (
                source_value["chart_work"].T @ source_value["out_work"]
            ).contiguous()
            contribution = (mapped_writer @ context_transport.T).contiguous()
            writer_contributions.append(contribution)
            layer_index = source_value["source"].source_layer_index
            writer_reports.append(
                {
                    "source_layer_index": layer_index,
                    "source_output_weight_sha256": _tensor_sha256(
                        source_value["source"].source_out_weight,
                        label=f"source layer {layer_index} output weight",
                    ),
                    "residual_output_chart_sha256": source_value["chart_sha256"],
                    "group_reconstruction_slice_sha256": _tensor_sha256(
                        source_reconstruction,
                        label=f"source layer {layer_index} group reconstruction",
                    ),
                    "context_transport_sha256": _tensor_sha256(
                        context_transport,
                        label=f"source layer {layer_index} context transport",
                    ),
                    "writer_contribution_sha256": _tensor_sha256(
                        contribution,
                        label=f"source layer {layer_index} writer contribution",
                    ),
                    "writer_contribution_frobenius_norm": float(
                        torch.linalg.vector_norm(contribution).item()
                    ),
                    "complete_source_output_projection_consumed": True,
                }
            )
        out_fp64 = torch.stack(writer_contributions, dim=0).sum(dim=0).contiguous()

        query = query_fp64.to(device="cpu", dtype=torch.bfloat16).contiguous()
        key = key_fp64.to(device="cpu", dtype=torch.bfloat16).contiguous()
        value = value_fp64.to(device="cpu", dtype=torch.bfloat16).contiguous()
        out = out_fp64.to(device="cpu", dtype=torch.bfloat16).contiguous()
        selected_query_norm_rows = (
            None
            if not norms_present
            else torch.cat(query_norm_blocks, dim=0)
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
        )
        selected_key_norm_rows = (
            None
            if not norms_present
            else torch.cat(key_norm_blocks, dim=0)
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
        )

    source_state_after = {
        source.source_layer_index: (
            lfm2_attention_group_qdeim_span_source_state_sha256(
                source.source_query_weight,
                source.source_key_weight,
                source.source_value_weight,
                source.source_out_weight,
                source.source_query_norm_weight,
                source.source_key_norm_weight,
            )
        )
        for source in ordered
    }
    context_after = {
        source.source_layer_index: _tensor_sha256(
            source.cached_parent_post_softmax_context_rows,
            label=(
                f"source layer {source.source_layer_index} cached post-softmax rows"
            ),
        )
        for source in ordered
    }
    decoder_after = {
        source.source_layer_index: _tensor_sha256(
            source.normalized_input_decoder,
            label=f"source layer {source.source_layer_index} decoder",
        )
        for source in ordered
    }
    chart_after = {
        source.source_layer_index: _tensor_sha256(
            source.residual_output_chart,
            label=f"source layer {source.source_layer_index} chart",
        )
        for source in ordered
    }
    target_template_after = lfm2_attention_group_qdeim_target_template_sha256(
        target_query_weight_template,
        target_key_weight_template,
        target_value_weight_template,
        target_out_weight_template,
    )
    output_tensors = (query, key, value, out)
    expected_shapes = (
        (target_q_heads * head_dim, target_hidden),
        (target_kv_heads * head_dim, target_hidden),
        (target_kv_heads * head_dim, target_hidden),
        (target_hidden, target_q_heads * head_dim),
    )
    if (
        source_state_after != source_state_before
        or target_template_after != template_sha256
        or any(
            context_after[source.source_layer_index]
            != source.expected_cached_parent_post_softmax_context_sha256
            or decoder_after[source.source_layer_index]
            != source.expected_normalized_input_decoder_sha256
            or chart_after[source.source_layer_index]
            != source.expected_residual_output_chart_sha256
            for source in ordered
        )
        or any(count <= 0 for count in selected_counts.values())
        or len(selected_pairs) != target_kv_heads
        or len(set(selected_pairs)) != target_kv_heads
        or any(
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.bfloat16
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not bool(torch.isfinite(tensor).all().item())
            for tensor, shape in zip(output_tensors, expected_shapes, strict=True)
        )
        or (
            norms_present
            and (
                tuple(selected_query_norm_rows.shape) != (target_q_heads, head_dim)
                or tuple(selected_key_norm_rows.shape) != (target_kv_heads, head_dim)
                or selected_query_norm_rows.dtype != torch.bfloat16
                or selected_key_norm_rows.dtype != torch.bfloat16
            )
        )
    ):
        raise Lfm2AttentionGroupQdeimError(
            "Materialized span attention failed dense, coverage, authentication, "
            "or zero-gradient invariants."
        )

    selected_query_norm_shared = bool(
        norms_present and _all_tensor_rows_identical(selected_query_norm_rows)
    )
    selected_key_norm_shared = bool(
        norms_present and _all_tensor_rows_identical(selected_key_norm_rows)
    )
    candidate_sha256 = _candidate_state_sha256(query, key, value, out)
    source_reports = []
    for source_value in normalized_sources:
        source = source_value["source"]
        layer_index = source.source_layer_index
        tensor_hashes = {
            "q_proj.weight": _tensor_sha256(
                source.source_query_weight,
                label=f"source layer {layer_index} query weight",
            ),
            "k_proj.weight": _tensor_sha256(
                source.source_key_weight,
                label=f"source layer {layer_index} key weight",
            ),
            "v_proj.weight": _tensor_sha256(
                source.source_value_weight,
                label=f"source layer {layer_index} value weight",
            ),
            "out_proj.weight": _tensor_sha256(
                source.source_out_weight,
                label=f"source layer {layer_index} output weight",
            ),
        }
        if norms_present:
            tensor_hashes.update(
                {
                    "q_layernorm.weight": _tensor_sha256(
                        source.source_query_norm_weight,
                        label=f"source layer {layer_index} query norm weight",
                    ),
                    "k_layernorm.weight": _tensor_sha256(
                        source.source_key_norm_weight,
                        label=f"source layer {layer_index} key norm weight",
                    ),
                }
            )
        source_reports.append(
            {
                "source_layer_index": layer_index,
                "source_state_sha256_before": source_state_before[layer_index],
                "source_state_sha256_after": source_state_after[layer_index],
                "source_state_immutable": True,
                "source_tensor_sha256": tensor_hashes,
                "source_query_head_count": source.source_query_head_count,
                "source_key_value_head_count": source.source_key_value_head_count,
                "source_hidden_size": source_value["source_hidden"],
                "queries_per_key_value": target_q_per_kv,
                "head_dim": head_dim,
                "union_group_coordinate_range": [
                    source_value["union_start"],
                    source_value["union_stop"],
                ],
                "selected_real_group_count": selected_counts[layer_index],
                "source_layer_retained_by_real_intact_gqa_group": (
                    selected_counts[layer_index] > 0
                ),
                "cached_post_softmax_context_sha256": source_value["context_sha256"],
                "cache_receipt_sha256": source.cache_receipt_sha256,
                "aligned_row_identity_sha256": source.aligned_row_identity_sha256,
                "normalized_input_decoder_sha256": source_value["decoder_sha256"],
                "residual_output_chart_sha256": source_value["chart_sha256"],
            }
        )
    report_body = {
        "schema": LFM2_ATTENTION_GROUP_SPAN_QDEIM_SCHEMA,
        "status": "structural-primitive-materialized-not-behavior-proven",
        "span": span.as_dict(),
        "target_layer_index": span.target_layer_index,
        "attention_source_layer_indices": [
            source.source_layer_index for source in ordered
        ],
        "attention_source_layer_count": len(ordered),
        "all_declared_attention_sources_consumed": True,
        "selection_scope": "joint_union_of_aligned_complete_gqa_group_contexts",
        "selection_constraint": (
            "deterministic_independent_basis_with_one_real_group_per_source"
        ),
        "selected_source_key_value_groups": [
            {
                "source_layer_index": layer_index,
                "source_key_value_group_index": group_index,
            }
            for layer_index, group_index in selected_pairs
        ],
        "selected_source_query_heads": [
            {
                "source_layer_index": layer_index,
                "source_query_head_index": head_index,
            }
            for layer_index, head_index in selected_query_heads
        ],
        "selected_group_counts_by_layer": {
            str(layer_index): selected_counts[layer_index]
            for layer_index in (source.source_layer_index for source in ordered)
        },
        "source_layers": source_reports,
        "target_template_sha256": template_sha256,
        "target_template_values_read": False,
        "context_evidence_sha256": context_evidence_sha256,
        "context_evidence_schema": (
            LFM2_ATTENTION_GROUP_SPAN_QDEIM_CONTEXT_EVIDENCE_SCHEMA
        ),
        "complete_source_bundle_rooted_by_external_expected_evidence": True,
        "aligned_context_row_count": row_count,
        "aligned_row_identity_sha256": row_identity,
        "union_source_key_value_group_count": total_source_groups,
        "target_query_head_count": target_q_heads,
        "target_key_value_head_count": target_kv_heads,
        "queries_per_key_value": target_q_per_kv,
        "rope_head_dim_preserved": head_dim,
        "observed_context_group_rank": observed_rank,
        "minimum_context_group_rank": target_kv_heads,
        "group_pod_basis_sha256": _tensor_sha256(
            pod_basis,
            label="span attention group POD basis",
        ),
        "group_interpolation_sha256": _tensor_sha256(
            group_interpolation,
            label="span attention group interpolation",
        ),
        "context_reconstruction_sha256": _tensor_sha256(
            context_reconstruction,
            label="span attention context reconstruction",
        ),
        "pivot_residual_norms_sha256": _tensor_sha256(
            pivot_residual_norms,
            label="span attention pivot residual norms",
        ),
        "functional_interpolation": {
            "formula": ("concat_layer_group_contexts ~= selected_union_groups @ R"),
            "group_interpolation_shape": list(group_interpolation.shape),
            "context_reconstruction_shape": list(context_reconstruction.shape),
            "weighted_relative_rmse_diagnostic": reconstruction_relative_rmse,
            "selected_identity_maximum_absolute_error": selected_identity_error,
            "condition_number": interpolation_maximum / interpolation_minimum,
            "numeric_diagnostics_have_semantic_authority": False,
        },
        "producer_transport": {
            "method": ("selected_real_intact_source_qkv_groups_through_own_decoders"),
            "q_formula": "Q_l[selected_group_rows,:] @ decoder_l.T",
            "k_formula": "K_l[selected_group_rows,:] @ decoder_l.T",
            "v_formula": "V_l[selected_group_rows,:] @ decoder_l.T",
            "qkv_producer_averaging": False,
            "cross_layer_weight_averaging": False,
            "gqa_grouping_preserved": True,
            "rope_head_dim_preserved": True,
        },
        "writer_transport": {
            "formula": ("O_child = sum_l(chart_l.T @ O_l @ kron(R_l,I_group).T)"),
            "source_contributions": writer_reports,
            "all_source_output_projections_consumed": True,
            "all_source_output_columns_consumed": True,
        },
        "qk_norm_lineage": {
            "source_norms_present": norms_present,
            "selected_query_norm_rows_sha256": (
                None
                if selected_query_norm_rows is None
                else _tensor_sha256(
                    selected_query_norm_rows,
                    label="selected query norm rows",
                )
            ),
            "selected_key_norm_rows_sha256": (
                None
                if selected_key_norm_rows is None
                else _tensor_sha256(
                    selected_key_norm_rows,
                    label="selected key norm rows",
                )
            ),
            "selected_query_norm_is_shared": selected_query_norm_shared,
            "selected_key_norm_is_shared": selected_key_norm_shared,
            "current_lfm2_shared_norm_directly_representable": (
                norms_present
                and selected_query_norm_shared
                and selected_key_norm_shared
            ),
            "source_norms_absent_requires_family_policy": not norms_present,
            "heterogeneous_selected_norms_were_averaged": False,
            "heterogeneous_selected_norms_require_architecture_aware_wiring": (
                norms_present
                and not (selected_query_norm_shared and selected_key_norm_shared)
            ),
        },
        "materialized_candidate_state_sha256": candidate_sha256,
        "dense_ordinary_child_qkvo_tensors": True,
        "dense_bfloat16_target_state": True,
        "parameter_sharing_used": False,
        "semantic_layer_stack_adapter_seam": {
            "span_owner": "Lfm2ModelFoldSpan",
            "ordered_source_payload": "Lfm2AttentionGroupQdeimSpanSource",
            "assembly_helper": ("assemble_lfm2_attention_group_qdeim_span_sources"),
            "planner_duplicated": False,
            "parent_recapture_required": False,
        },
        "remaining_integration": {
            "upstream_cache_receipt_content_verification_required": True,
            "production_semantic_stack_wiring_required": True,
            "transactional_blank_target_installation_required": True,
            "decoded_whole_response_evaluation_required": True,
            "decoded_behavior_evaluated_here": False,
            "behavior_or_intelligence_retention_claimed_here": False,
        },
        "compute_execution": compute_execution,
        "numeric_diagnostics_are_advisory_only": True,
        "automatic_candidate_acceptance_performed": False,
        "model_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
    }
    report = {**report_body, "result_sha256": canonical_sha256(report_body)}
    return Lfm2CompiledSpanAttentionGroupQdeim(
        span=span,
        attention_source_layer_indices=tuple(
            source.source_layer_index for source in ordered
        ),
        selected_source_key_value_groups=tuple(selected_pairs),
        selected_source_query_heads=tuple(selected_query_heads),
        query_weight=query.detach().cpu().contiguous().clone(),
        key_weight=key.detach().cpu().contiguous().clone(),
        value_weight=value.detach().cpu().contiguous().clone(),
        out_weight=out.detach().cpu().contiguous().clone(),
        selected_query_norm_weight_rows=(
            None
            if selected_query_norm_rows is None
            else selected_query_norm_rows.detach().cpu().contiguous().clone()
        ),
        selected_key_norm_weight_rows=(
            None
            if selected_key_norm_rows is None
            else selected_key_norm_rows.detach().cpu().contiguous().clone()
        ),
        group_pod_basis=pod_basis.detach().cpu().contiguous().clone(),
        group_interpolation=(group_interpolation.detach().cpu().contiguous().clone()),
        context_reconstruction=(
            context_reconstruction.detach().cpu().contiguous().clone()
        ),
        report=MappingProxyType(report),
    )


__all__ = [
    "LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE",
    "LFM2_ATTENTION_GROUP_QDEIM_OLDER_PREFIX_KV_PERTURBATION_ROLE",
    "LFM2_ATTENTION_GROUP_QDEIM_RAW_CONTEXT_METRIC",
    "LFM2_ATTENTION_GROUP_QDEIM_RESIDUAL_OUTPUT_BALANCED_METRIC",
    "LFM2_ATTENTION_GROUP_SPAN_QDEIM_CONTEXT_EVIDENCE_SCHEMA",
    "LFM2_ATTENTION_GROUP_SPAN_QDEIM_RANK_DEFICIT_SCHEMA",
    "LFM2_ATTENTION_GROUP_SPAN_QDEIM_SCHEMA",
    "LFM2_ATTENTION_GROUP_QDEIM_RANK_DEFICIT_SCHEMA",
    "LFM2_ATTENTION_GROUP_QDEIM_SCHEMA",
    "LFM2_ATTENTION_GROUP_QDEIM_SNAPSHOT_EVIDENCE_SCHEMA",
    "Lfm2AttentionGroupQdeimError",
    "Lfm2AttentionGroupQdeimInsufficientRankError",
    "Lfm2AttentionGroupQdeimSnapshotBlock",
    "Lfm2AttentionGroupQdeimSpanSource",
    "Lfm2CompiledAttentionGroupQdeim",
    "Lfm2CompiledSpanAttentionGroupQdeim",
    "assemble_lfm2_attention_group_qdeim_span_sources",
    "compile_lfm2_attention_group_qdeim",
    "compile_lfm2_attention_group_span_qdeim",
    "lfm2_attention_group_qdeim_context_evidence_sha256",
    "lfm2_attention_group_qdeim_snapshot_evidence_sha256",
    "lfm2_attention_group_qdeim_span_context_evidence_sha256",
    "lfm2_attention_group_qdeim_span_source_state_sha256",
    "lfm2_attention_group_qdeim_source_state_sha256",
    "lfm2_attention_group_qdeim_target_template_sha256",
]
