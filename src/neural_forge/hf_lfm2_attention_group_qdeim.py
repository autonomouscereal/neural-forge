"""Standalone intact-GQA-group Q-DEIM attention folding prototype.

The compiler selects whole parent GQA groups from authenticated cached
post-softmax context.  A selected group always carries all of its query heads
and its single K/V head; Q, K, and V producers are therefore copied intact
through the target-to-source input decoder and are never averaged before
softmax.  A dense output projection reconstructs every parent context group
from the selected groups and maps the parent residual output through the
source-to-target residual chart.

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
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_ATTENTION_GROUP_QDEIM_SCHEMA = (
    "neural-forge-lfm2-attention-intact-gqa-group-qdeim-v1"
)
LFM2_ATTENTION_GROUP_QDEIM_RANK_DEFICIT_SCHEMA = (
    "neural-forge-lfm2-attention-intact-gqa-group-qdeim-rank-deficit-v1"
)
LFM2_ATTENTION_GROUP_QDEIM_SNAPSHOT_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-attention-group-qdeim-functional-snapshots-v1"
)

_ALLOWED_SNAPSHOT_KINDS = frozenset(
    {
        "analytic_child_lifted_parent_attention_tangent",
        "authenticated_structural",
    }
)


class Lfm2AttentionGroupQdeimError(ValueError):
    """Authenticated intact-group attention compilation failed closed."""


class Lfm2AttentionGroupQdeimInsufficientRankError(Lfm2AttentionGroupQdeimError):
    """Authenticated context snapshots cannot support the target group count."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = MappingProxyType(dict(report))
        super().__init__(
            "Authenticated parent context group rank "
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
    and cannot confer semantic acceptance.
    """

    block_id: str
    snapshot_kind: str
    parent_post_softmax_context_rows: Any = field(repr=False, compare=False)
    row_sample_weights: Sequence[float] = field(repr=False, compare=False)
    generator_receipt_sha256: str
    block_weight: float = 1.0


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


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "PyTorch is required for intact-GQA-group Q-DEIM compilation."
        ) from exc
    return torch


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
    return canonical_sha256(
        {
            "schema": LFM2_ATTENTION_GROUP_QDEIM_SNAPSHOT_EVIDENCE_SCHEMA,
            "heldout_rows_permitted": False,
            "blocks": [
                {
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
                for block in normalized
            ],
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
    q_columns = torch.empty((modes, 0), dtype=torch.float64)
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
            scores[torch.tensor(selected, dtype=torch.long)] = -1.0
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
) -> dict[str, Any]:
    body = {
        "schema": LFM2_ATTENTION_GROUP_QDEIM_RANK_DEFICIT_SCHEMA,
        "status": "insufficient-authenticated-context-group-rank",
        "source_state_sha256_before": source_state_sha256,
        "source_state_sha256_after": source_state_sha256_after,
        "source_state_immutable": source_state_sha256_after == source_state_sha256,
        "target_template_sha256": target_template_sha256,
        "normalized_input_decoder_sha256": decoder_sha256,
        "residual_output_chart_sha256": chart_sha256,
        "fit_context_evidence_sha256": context_evidence_sha256,
        "functional_snapshot_evidence_sha256": snapshot_evidence_sha256,
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
    pod_rank_relative_tolerance: float | None = None,
) -> Lfm2CompiledAttentionGroupQdeim:
    """Compile one dense intact-GQA-group attention candidate without training.

    The current LFM2 use is 16 Q / 8 KV / 64 dimensions folded to 10 Q / 5 KV
    / 64 dimensions.  None of those numbers are hardcoded: all topology and
    tensor relationships are derived from the supplied head counts and shapes.
    """

    torch = _torch()
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
        fit_group_features = _group_feature_matrix(
            contexts,
            key_value_head_count=source_kv_heads,
            queries_per_key_value=source_q_per_kv,
        )
        fit_feature_weights = torch.tensor(
            weights,
            dtype=torch.float64,
        ).repeat_interleave(source_q_per_kv * head_dim)
        all_features = [fit_group_features]
        all_feature_weights = [fit_feature_weights]
        for block in normalized_blocks:
            all_features.append(
                _group_feature_matrix(
                    block["context"],
                    key_value_head_count=source_kv_heads,
                    queries_per_key_value=source_q_per_kv,
                )
            )
            all_feature_weights.append(
                (
                    torch.tensor(
                        block["row_sample_weights"],
                        dtype=torch.float64,
                    )
                    * block["block_weight"]
                ).repeat_interleave(source_q_per_kv * head_dim)
            )
        group_features = torch.cat(all_features, dim=0).contiguous()
        feature_weights = torch.cat(all_feature_weights, dim=0).contiguous()
        weighted_group_features = (
            group_features * torch.sqrt(feature_weights).unsqueeze(1)
        ).contiguous()

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
            (selected_identity - torch.eye(target_kv_heads, dtype=torch.float64))
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
            torch.eye(group_block_width, dtype=torch.float64),
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
        query_fp64 = (source_q[list(selected_query_rows), :] @ decoder.T).contiguous()
        key_fp64 = (source_k[list(selected_key_value_rows), :] @ decoder.T).contiguous()
        value_fp64 = (
            source_v[list(selected_key_value_rows), :] @ decoder.T
        ).contiguous()
        mapped_parent_out = (chart.T @ source_o).contiguous()
        out_fp64 = (mapped_parent_out @ context_reconstruction.T).contiguous()

        query = query_fp64.to(dtype=torch.bfloat16).contiguous()
        key = key_fp64.to(dtype=torch.bfloat16).contiguous()
        value = value_fp64.to(dtype=torch.bfloat16).contiguous()
        out = out_fp64.to(dtype=torch.bfloat16).contiguous()

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
        },
        "evidence": {
            "fit_partition_only": True,
            "heldout_rows_used": False,
            "fit_context_row_count": int(contexts.shape[0]),
            "explicit_positive_fit_row_weights": True,
            "structural_tangent_snapshot_block_count": len(normalized_blocks),
            "structural_tangent_context_row_count": sum(
                int(block["context"].shape[0]) for block in normalized_blocks
            ),
            "snapshot_blocks": [
                {
                    "block_id": block["block_id"],
                    "snapshot_kind": block["snapshot_kind"],
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


__all__ = [
    "LFM2_ATTENTION_GROUP_QDEIM_RANK_DEFICIT_SCHEMA",
    "LFM2_ATTENTION_GROUP_QDEIM_SCHEMA",
    "LFM2_ATTENTION_GROUP_QDEIM_SNAPSHOT_EVIDENCE_SCHEMA",
    "Lfm2AttentionGroupQdeimError",
    "Lfm2AttentionGroupQdeimInsufficientRankError",
    "Lfm2AttentionGroupQdeimSnapshotBlock",
    "Lfm2CompiledAttentionGroupQdeim",
    "compile_lfm2_attention_group_qdeim",
    "lfm2_attention_group_qdeim_context_evidence_sha256",
    "lfm2_attention_group_qdeim_snapshot_evidence_sha256",
    "lfm2_attention_group_qdeim_source_state_sha256",
    "lfm2_attention_group_qdeim_target_template_sha256",
]
