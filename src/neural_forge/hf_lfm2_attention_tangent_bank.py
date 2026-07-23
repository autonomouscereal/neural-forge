"""Authenticated cached-only causal-attention tangent snapshots for LFM2.

The builder evaluates analytic JVPs of the cached parent attention circuit in
every row direction of an authenticated child-to-parent normalized-input
decoder.  It never calls a model and never differentiates a graph.  Two
partial Jacobian families remain separately authenticated as diagnostics:

* query-local perturbations change Q at one sampled query; and
* sampled-suffix history perturbations change K and V at one authenticated
  current-token history slot.

The functional POD evidence instead uses the joint same-token Q+K+V JVP, so
its covariance retains Q/K/V cross terms, plus a non-overlapping older-prefix
V-only JVP (whose linear derivative needs no activation anchor).  Older-prefix
K is not invented; its authenticated cross-phase stitch remains explicit.
Every group tangent is mapped through the real source O block and residual
chart before covariance accumulation.  Small synthetic context blocks are
factored from those covariances and authenticated by direct equality.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_ATTENTION_TANGENT_BANK_SCHEMA = (
    "neural-forge-lfm2-cached-causal-attention-tangent-bank-v1"
)
_REDUCTION_SCHEMA = "neural-forge-lfm2-attention-tangent-covariance-reduction-v1"
_COMPUTE_SCHEMA = "neural-forge-lfm2-attention-tangent-compute-v1"
_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "sparsity_used": False,
        "parameter_sharing_used": False,
        "randomness_used": False,
    }
)


class Lfm2AttentionTangentBankError(ValueError):
    """Cached attention tangent evidence is malformed or unauthenticated."""


class Lfm2AttentionTangentAuthenticationError(Lfm2AttentionTangentBankError):
    """A source tensor or covariance reduction differs from its seal."""


@dataclass(frozen=True, slots=True)
class Lfm2AttentionTangentFitCase:
    """One authenticated FIT phase of the cached causal attention circuit."""

    case_id: str
    phase_weight: float
    query_linear_suffix: Tensor = field(repr=False, compare=False)
    key_linear_suffix: Tensor = field(repr=False, compare=False)
    query_normalized_pre_rope: Tensor = field(repr=False, compare=False)
    key_normalized_pre_rope: Tensor = field(repr=False, compare=False)
    rope_cos: Tensor = field(repr=False, compare=False)
    rope_sin: Tensor = field(repr=False, compare=False)
    query_rotary_suffix: Tensor = field(repr=False, compare=False)
    key_linear_history: Tensor = field(repr=False, compare=False)
    key_normalized_pre_rope_history: Tensor = field(repr=False, compare=False)
    key_rope_cos_history: Tensor = field(repr=False, compare=False)
    key_rope_sin_history: Tensor = field(repr=False, compare=False)
    key_history: Tensor = field(repr=False, compare=False)
    value_history: Tensor = field(repr=False, compare=False)
    qk_scores: Tensor = field(repr=False, compare=False)
    probabilities: Tensor = field(repr=False, compare=False)
    additive_mask: Tensor = field(repr=False, compare=False)
    source_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2AttentionTangentSnapshot:
    """One covariance-equivalent Q-DEIM context block and its seal."""

    block_id: str
    perturbation_kind: str
    context_rows: Tensor = field(repr=False, compare=False)
    row_sample_weights: tuple[float, ...]
    covariance: Tensor = field(repr=False, compare=False)
    reduction_receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Lfm2AttentionTangentBank:
    """Functional joint tangents plus separately authenticated diagnostics."""

    site_id: str
    query_local: Lfm2AttentionTangentSnapshot
    sampled_suffix_history_key_value: Lfm2AttentionTangentSnapshot
    joint_sampled_token_query_key_value: Lfm2AttentionTangentSnapshot
    older_prefix_key_value: Lfm2AttentionTangentSnapshot
    report: Mapping[str, Any]

    @property
    def bank_sha256(self) -> str:
        return str(self.report["bank_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)


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
        raise Lfm2AttentionTangentAuthenticationError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _plain_tensor(
    value: Any,
    *,
    label: str,
    ndim: int,
    allow_negative_infinity: bool = False,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or isinstance(value, nn.Parameter)
        or value.requires_grad
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or value.numel() <= 0
        or not value.dtype.is_floating_point
        or torch.is_complex(value)
    ):
        raise Lfm2AttentionTangentBankError(
            f"{label} must be one detached materialized floating rank-{ndim} tensor."
        )
    if allow_negative_infinity:
        valid = ~torch.isnan(value) & ~torch.isposinf(value)
    else:
        valid = torch.isfinite(value)
    if not bool(valid.all().item()):
        raise Lfm2AttentionTangentBankError(f"{label} contains invalid values.")
    return value


def _tensor_sha256(value: Tensor) -> str:
    return lfm2_authenticated_tensor_sha256(value)


def _authenticate_tensor(value: Tensor, expected: Any, *, label: str) -> str:
    observed = _tensor_sha256(value)
    if observed != _digest(expected, label=f"expected {label} SHA-256"):
        raise Lfm2AttentionTangentAuthenticationError(
            f"{label} differs from its authenticated digest."
        )
    return observed


def _positive_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Lfm2AttentionTangentBankError(f"{label} must be positive and finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2AttentionTangentBankError(
            f"{label} must be positive and finite."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2AttentionTangentBankError(f"{label} must be positive and finite.")
    return result


def _compute_device(value: Any) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not isinstance(value, str):
        raise Lfm2AttentionTangentBankError(
            "compute_device must be 'cpu' or an explicitly indexed CUDA device."
        )
    prefix, separator, suffix = value.partition(":")
    if (
        prefix != "cuda"
        or separator != ":"
        or not suffix.isascii()
        or not suffix.isdecimal()
        or suffix != str(int(suffix))
    ):
        raise Lfm2AttentionTangentBankError(
            "CUDA compute_device must be explicitly indexed as 'cuda:N'."
        )
    index = int(suffix)
    if not torch.cuda.is_available() or index >= int(torch.cuda.device_count()):
        raise Lfm2AttentionTangentBankError(
            f"CUDA compute_device index {index} is unavailable."
        )
    return torch.device("cuda", index)


def _compute_receipt(requested: str, resolved: torch.device) -> Mapping[str, Any]:
    body: dict[str, Any] = {
        "schema": _COMPUTE_SCHEMA,
        "requested_device": requested,
        "resolved_device": str(resolved),
        "backend": resolved.type,
        "compute_dtype": "float64",
        "authenticated_source_archive_device": "cpu",
        "active_phase_staged_to_compute_device": resolved.type == "cuda",
        "materialized_covariance_and_snapshot_device": "cpu",
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
    return MappingProxyType({**body, "receipt_sha256": canonical_sha256(body)})


def _cross_dtype_replay_diagnostic(
    expected: Tensor,
    observed: Tensor,
    *,
    observed_capture_dtype: torch.dtype,
    rtol: float = 2e-2,
    atol: float = 2e-3,
) -> dict[str, object]:
    """Describe an FP64 replay without granting it authentication authority."""

    difference = expected - observed
    maximum_absolute_error = float(difference.abs().max().item())
    root_mean_square_error = float(difference.square().mean().sqrt().item())
    reference_root_mean_square = float(expected.square().mean().sqrt().item())
    return {
        "observed_capture_dtype": str(observed_capture_dtype),
        "replay_dtype": "torch.float64",
        "maximum_absolute_error": maximum_absolute_error,
        "root_mean_square_error": root_mean_square_error,
        "relative_root_mean_square_error_with_atol_floor": (
            root_mean_square_error / max(reference_root_mean_square, atol)
        ),
        "legacy_allclose_rtol": rtol,
        "legacy_allclose_atol": atol,
        "within_legacy_allclose_envelope": bool(
            torch.allclose(expected, observed, rtol=rtol, atol=atol)
        ),
        "hard_authentication_authority": False,
        "semantic_selection_authority": False,
    }


def _rotate_half(value: Tensor) -> Tensor:
    midpoint = int(value.shape[-1]) // 2
    return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def _apply_rope(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return value * cos + _rotate_half(value) * sin


def _rms_norm_sequence_jvp(
    linear: Tensor,
    direction: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Exact RMSNorm JVP for every independent row in a sequence.

    ``linear`` is ``[batch, rows, heads, head_dim]`` and ``direction`` is
    ``[directions, heads, head_dim]``.  The returned independent JVPs are
    ``[batch, directions, rows, heads, head_dim]``; no sequence positions are
    combined by this batching transformation.
    """

    inverse_rms = torch.rsqrt(linear.square().mean(dim=-1, keepdim=True) + epsilon)
    radial = (
        linear.unsqueeze(1) * direction.reshape(1, -1, 1, *direction.shape[1:])
    ).mean(dim=-1, keepdim=True)
    normalized_direction = (
        direction.reshape(1, -1, 1, *direction.shape[1:]) * inverse_rms.unsqueeze(1)
        - linear.unsqueeze(1) * inverse_rms.pow(3).unsqueeze(1) * radial
    )
    return normalized_direction * weight.reshape(1, 1, 1, 1, -1)


def lfm2_attention_causal_jvp(
    *,
    query: Tensor,
    key_history: Tensor,
    value_history: Tensor,
    probabilities: Tensor,
    delta_query: Tensor,
    delta_key_history: Tensor,
    delta_value_history: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate the exact softmax-attention JVP stated in the project contract.

    Every tensor is rank four ``[batch, query_heads, rows, head_dim]`` except
    probabilities, which is ``[batch, query_heads, query_rows, key_rows]``.
    K/V must already be repeated from GQA groups to query heads.
    """

    values = {
        "query": _plain_tensor(query, label="query", ndim=4),
        "key_history": _plain_tensor(key_history, label="key_history", ndim=4),
        "value_history": _plain_tensor(
            value_history,
            label="value_history",
            ndim=4,
        ),
        "probabilities": _plain_tensor(
            probabilities,
            label="probabilities",
            ndim=4,
        ),
        "delta_query": _plain_tensor(
            delta_query,
            label="delta_query",
            ndim=4,
        ),
        "delta_key_history": _plain_tensor(
            delta_key_history,
            label="delta_key_history",
            ndim=4,
        ),
        "delta_value_history": _plain_tensor(
            delta_value_history,
            label="delta_value_history",
            ndim=4,
        ),
    }
    query = values["query"].to(torch.float64)
    key_history = values["key_history"].to(torch.float64)
    value_history = values["value_history"].to(torch.float64)
    probability = values["probabilities"].to(torch.float64)
    delta_query = values["delta_query"].to(torch.float64)
    delta_key = values["delta_key_history"].to(torch.float64)
    delta_value = values["delta_value_history"].to(torch.float64)
    if (
        tuple(query.shape) != tuple(delta_query.shape)
        or tuple(key_history.shape) != tuple(delta_key.shape)
        or tuple(value_history.shape) != tuple(delta_value.shape)
        or tuple(key_history.shape) != tuple(value_history.shape)
        or tuple(probability.shape)
        != (
            int(query.shape[0]),
            int(query.shape[1]),
            int(query.shape[2]),
            int(key_history.shape[2]),
        )
        or int(query.shape[3]) != int(key_history.shape[3])
    ):
        raise Lfm2AttentionTangentBankError("Causal JVP geometry is inconsistent.")
    scale = float(int(query.shape[-1]) ** -0.5)
    delta_score = (
        delta_query @ key_history.transpose(2, 3) + query @ delta_key.transpose(2, 3)
    ) * scale
    centered = delta_score - (probability * delta_score).sum(
        dim=-1,
        keepdim=True,
    )
    delta_probability = probability * centered
    delta_context = delta_probability @ value_history + probability @ delta_value
    return (
        delta_score.contiguous(),
        delta_probability.contiguous(),
        delta_context.contiguous(),
    )


def lfm2_attention_tangent_case_evidence_sha256(
    case: Lfm2AttentionTangentFitCase,
) -> str:
    if not isinstance(case, Lfm2AttentionTangentFitCase):
        raise Lfm2AttentionTangentBankError(
            "Attention tangent case must use the typed FIT case."
        )
    if not isinstance(case.case_id, str) or not case.case_id.strip():
        raise Lfm2AttentionTangentBankError("Attention tangent case_id is empty.")
    tensors = _validated_case_tensors(case)
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-attention-tangent-fit-case-v1",
            "case_id": case.case_id.strip(),
            "phase_weight": _positive_float(
                case.phase_weight,
                label=f"{case.case_id} phase_weight",
            ),
            "source_receipt_sha256": _digest(
                case.source_receipt_sha256,
                label=f"{case.case_id} source receipt",
            ),
            "tensor_sha256": {
                name: _tensor_sha256(value) for name, value in sorted(tensors.items())
            },
        }
    )


def _validated_case_tensors(case: Lfm2AttentionTangentFitCase) -> dict[str, Tensor]:
    return {
        "query_linear_suffix": _plain_tensor(
            case.query_linear_suffix,
            label=f"{case.case_id} query_linear_suffix",
            ndim=4,
        ),
        "key_linear_suffix": _plain_tensor(
            case.key_linear_suffix,
            label=f"{case.case_id} key_linear_suffix",
            ndim=4,
        ),
        "query_normalized_pre_rope": _plain_tensor(
            case.query_normalized_pre_rope,
            label=f"{case.case_id} query_normalized_pre_rope",
            ndim=4,
        ),
        "key_normalized_pre_rope": _plain_tensor(
            case.key_normalized_pre_rope,
            label=f"{case.case_id} key_normalized_pre_rope",
            ndim=4,
        ),
        "rope_cos": _plain_tensor(
            case.rope_cos,
            label=f"{case.case_id} rope_cos",
            ndim=3,
        ),
        "rope_sin": _plain_tensor(
            case.rope_sin,
            label=f"{case.case_id} rope_sin",
            ndim=3,
        ),
        "query_rotary_suffix": _plain_tensor(
            case.query_rotary_suffix,
            label=f"{case.case_id} query_rotary_suffix",
            ndim=4,
        ),
        "key_linear_history": _plain_tensor(
            case.key_linear_history,
            label=f"{case.case_id} key_linear_history",
            ndim=4,
        ),
        "key_normalized_pre_rope_history": _plain_tensor(
            case.key_normalized_pre_rope_history,
            label=f"{case.case_id} key_normalized_pre_rope_history",
            ndim=4,
        ),
        "key_rope_cos_history": _plain_tensor(
            case.key_rope_cos_history,
            label=f"{case.case_id} key_rope_cos_history",
            ndim=3,
        ),
        "key_rope_sin_history": _plain_tensor(
            case.key_rope_sin_history,
            label=f"{case.case_id} key_rope_sin_history",
            ndim=3,
        ),
        "key_history": _plain_tensor(
            case.key_history,
            label=f"{case.case_id} key_history",
            ndim=4,
        ),
        "value_history": _plain_tensor(
            case.value_history,
            label=f"{case.case_id} value_history",
            ndim=4,
        ),
        "qk_scores": _plain_tensor(
            case.qk_scores,
            label=f"{case.case_id} qk_scores",
            ndim=4,
        ),
        "probabilities": _plain_tensor(
            case.probabilities,
            label=f"{case.case_id} probabilities",
            ndim=4,
        ),
        "additive_mask": _plain_tensor(
            case.additive_mask,
            label=f"{case.case_id} additive_mask",
            ndim=4,
            allow_negative_infinity=True,
        ),
    }


def _raw_group_features(context: Tensor, *, groups: int, q_per_group: int) -> Tensor:
    rows, query_heads, head_dim = map(int, context.shape)
    if query_heads != groups * q_per_group:
        raise Lfm2AttentionTangentBankError(
            "Tangent context does not preserve the authenticated GQA groups."
        )
    return (
        context.reshape(rows, groups, q_per_group, head_dim)
        .permute(0, 2, 3, 1)
        .reshape(rows * q_per_group * head_dim, groups)
        .contiguous()
    )


def _residual_group_features(
    context: Tensor,
    *,
    groups: int,
    q_per_group: int,
    group_residual_maps: Tensor,
) -> Tensor:
    """Map each intact GQA-group delta through source O and residual chart."""

    rows, query_heads, head_dim = map(int, context.shape)
    group_width = q_per_group * head_dim
    if query_heads != groups * q_per_group or tuple(group_residual_maps.shape[:2]) != (
        groups,
        group_width,
    ):
        raise Lfm2AttentionTangentBankError(
            "Residual-mapped tangent context does not preserve GQA geometry."
        )
    grouped = context.reshape(rows, groups, group_width)
    mapped = torch.einsum("rgw,gwt->rgt", grouped, group_residual_maps)
    return mapped.permute(0, 2, 1).reshape(-1, groups).contiguous()


def _accumulate_context_covariance(
    covariance: Tensor,
    context: Tensor,
    *,
    row_weight: float,
    groups: int,
    q_per_group: int,
    group_residual_maps: Tensor,
) -> None:
    features = _residual_group_features(
        context,
        groups=groups,
        q_per_group=q_per_group,
        group_residual_maps=group_residual_maps,
    )
    covariance.addmm_(features.T, features, beta=1.0, alpha=row_weight)


def _covariance_snapshot(
    covariance: Tensor,
    *,
    block_id: str,
    perturbation_kind: str,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    original_tangent_context_row_count: int,
    original_weight_sum: float,
) -> Lfm2AttentionTangentSnapshot:
    covariance = covariance.detach().to(device="cpu", dtype=torch.float64)
    covariance = ((covariance + covariance.T) * 0.5).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    maximum = float(eigenvalues.abs().max().item())
    tolerance = (
        4096.0
        * torch.finfo(torch.float64).eps
        * max(1, key_value_heads)
        * max(1.0, maximum)
    )
    if float(eigenvalues.min().item()) < -tolerance:
        raise Lfm2AttentionTangentAuthenticationError(
            f"{perturbation_kind} tangent covariance is not positive semidefinite."
        )
    retained = eigenvalues > tolerance
    if not bool(retained.any().item()):
        # An exact zero partial Jacobian is valid evidence.  One explicit zero
        # factor row preserves its zero covariance without fabricating rank.
        factor = torch.zeros((1, key_value_heads), dtype=torch.float64)
    else:
        values = eigenvalues[retained]
        vectors = eigenvectors[:, retained]
        factor = (torch.sqrt(values).unsqueeze(1) * vectors.T).contiguous()
        for row_index in range(int(factor.shape[0])):
            row = factor[row_index]
            pivot = int(torch.argmax(row.abs()).item())
            if float(row[pivot].item()) < 0.0:
                factor[row_index].neg_()
    context = torch.zeros(
        (int(factor.shape[0]), query_heads, head_dim),
        dtype=torch.float64,
    )
    q_per_group = query_heads // key_value_heads
    for group in range(key_value_heads):
        context[:, group * q_per_group, 0] = factor[:, group]
    reduced_features = _raw_group_features(
        context,
        groups=key_value_heads,
        q_per_group=q_per_group,
    )
    reduced_covariance = (reduced_features.T @ reduced_features).contiguous()
    maximum_error = float((reduced_covariance - covariance).abs().max().item())
    equality_tolerance = (
        16384.0
        * torch.finfo(torch.float64).eps
        * max(1, key_value_heads)
        * max(1.0, maximum)
    )
    if maximum_error > equality_tolerance:
        raise Lfm2AttentionTangentAuthenticationError(
            f"{perturbation_kind} covariance-equivalent reduction failed."
        )
    body = {
        "schema": _REDUCTION_SCHEMA,
        "block_id": block_id,
        "perturbation_kind": perturbation_kind,
        "original_tangent_context_row_count": original_tangent_context_row_count,
        "original_tangent_row_weight_sum": original_weight_sum,
        "source_group_covariance_sha256": _tensor_sha256(covariance),
        "factor_sha256": _tensor_sha256(factor),
        "reduced_context_rows_sha256": _tensor_sha256(context),
        "reduced_group_covariance_sha256": _tensor_sha256(reduced_covariance),
        "reduced_context_row_count": int(context.shape[0]),
        "row_sample_weights": [1.0] * int(context.shape[0]),
        "covariance_maximum_absolute_error": maximum_error,
        "covariance_equality_tolerance": equality_tolerance,
        "covariance_equality_proved": True,
        "covariance_reduction_changes_qdeim_group_pod_ftf": False,
        "numeric_covariance_check_is_structural_not_semantic_gate": True,
    }
    receipt = MappingProxyType({**body, "result_sha256": canonical_sha256(body)})
    return Lfm2AttentionTangentSnapshot(
        block_id=block_id,
        perturbation_kind=perturbation_kind,
        context_rows=context,
        row_sample_weights=tuple(1.0 for _ in range(int(context.shape[0]))),
        covariance=covariance,
        reduction_receipt=receipt,
    )


def _case_geometry(
    tensors: Mapping[str, Tensor],
    *,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int]:
    q_linear = tensors["query_linear_suffix"]
    k_linear = tensors["key_linear_suffix"]
    q_pre = tensors["query_normalized_pre_rope"]
    k_pre = tensors["key_normalized_pre_rope"]
    cos = tensors["rope_cos"]
    sin = tensors["rope_sin"]
    query = tensors["query_rotary_suffix"]
    k_linear_history = tensors["key_linear_history"]
    k_pre_history = tensors["key_normalized_pre_rope_history"]
    k_cos_history = tensors["key_rope_cos_history"]
    k_sin_history = tensors["key_rope_sin_history"]
    keys = tensors["key_history"]
    values = tensors["value_history"]
    qk_scores = tensors["qk_scores"]
    probabilities = tensors["probabilities"]
    mask = tensors["additive_mask"]
    batch, observed_q_heads, query_rows, observed_head_dim = map(int, q_linear.shape)
    suffix_rows = int(k_linear.shape[2])
    history_rows = int(keys.shape[2])
    if (
        observed_q_heads != query_heads
        or observed_head_dim != head_dim
        or tuple(k_linear.shape) != (batch, key_value_heads, suffix_rows, head_dim)
        or suffix_rows != query_rows
        or int(q_pre.shape[0]) != batch
        or tuple(q_pre.shape[1::2]) != (query_heads, head_dim)
        or int(q_pre.shape[2]) < query_rows
        or int(k_pre.shape[0]) != batch
        or tuple(k_pre.shape[1::2]) != (key_value_heads, head_dim)
        or int(k_pre.shape[2]) < suffix_rows
        or tuple(cos.shape) != tuple(sin.shape)
        or int(cos.shape[0]) != batch
        or int(cos.shape[1]) < max(query_rows, suffix_rows)
        or int(cos.shape[2]) != head_dim
        or tuple(query.shape) != (batch, query_heads, query_rows, head_dim)
        or tuple(k_linear_history.shape)
        != (batch, key_value_heads, history_rows, head_dim)
        or tuple(k_pre_history.shape) != tuple(k_linear_history.shape)
        or tuple(k_cos_history.shape) != (batch, history_rows, head_dim)
        or tuple(k_sin_history.shape) != tuple(k_cos_history.shape)
        or tuple(keys.shape) != (batch, key_value_heads, history_rows, head_dim)
        or tuple(values.shape) != tuple(keys.shape)
        or history_rows < suffix_rows
        or tuple(qk_scores.shape) != (batch, query_heads, query_rows, history_rows)
        or tuple(probabilities.shape) != (batch, query_heads, query_rows, history_rows)
        or int(mask.shape[0]) not in (1, batch)
        or int(mask.shape[-2]) != query_rows
        or int(mask.shape[-1]) != history_rows
        or int(mask.shape[1]) not in (1, query_heads)
    ):
        raise Lfm2AttentionTangentBankError(
            "Cached attention tangent phase has incompatible causal geometry."
        )
    return batch, query_rows, suffix_rows, history_rows


def build_lfm2_attention_tangent_bank(
    *,
    site_id: str,
    source_query_weight: Tensor,
    expected_source_query_weight_sha256: str,
    source_key_weight: Tensor,
    expected_source_key_weight_sha256: str,
    source_value_weight: Tensor,
    expected_source_value_weight_sha256: str,
    source_out_weight: Tensor,
    expected_source_out_weight_sha256: str,
    source_query_norm_weight: Tensor,
    expected_source_query_norm_weight_sha256: str,
    source_key_norm_weight: Tensor,
    expected_source_key_norm_weight_sha256: str,
    normalized_input_decoder: Tensor,
    expected_normalized_input_decoder_sha256: str,
    residual_output_chart: Tensor,
    expected_residual_output_chart_sha256: str,
    norm_epsilon: float,
    source_query_head_count: int,
    source_key_value_head_count: int,
    fit_cases: Sequence[Lfm2AttentionTangentFitCase],
    expected_fit_case_evidence_sha256: str,
    compute_device: str = "cpu",
    direction_chunk_size: int = 64,
    history_position_chunk_size: int = 4,
    older_history_position_chunk_size: int = 64,
) -> Lfm2AttentionTangentBank:
    """Build covariance-equivalent query and sampled-suffix K/V tangent blocks."""

    if not isinstance(site_id, str) or not site_id.strip():
        raise Lfm2AttentionTangentBankError("site_id must be nonempty.")
    if (
        isinstance(source_query_head_count, bool)
        or not isinstance(source_query_head_count, int)
        or source_query_head_count <= 0
        or isinstance(source_key_value_head_count, bool)
        or not isinstance(source_key_value_head_count, int)
        or source_key_value_head_count <= 0
        or source_query_head_count % source_key_value_head_count
    ):
        raise Lfm2AttentionTangentBankError("Source GQA topology is invalid.")
    if (
        isinstance(direction_chunk_size, bool)
        or not isinstance(direction_chunk_size, int)
        or direction_chunk_size <= 0
    ):
        raise Lfm2AttentionTangentBankError(
            "direction_chunk_size must be a positive integer."
        )
    if (
        isinstance(history_position_chunk_size, bool)
        or not isinstance(history_position_chunk_size, int)
        or history_position_chunk_size <= 0
    ):
        raise Lfm2AttentionTangentBankError(
            "history_position_chunk_size must be a positive integer."
        )
    if (
        isinstance(older_history_position_chunk_size, bool)
        or not isinstance(older_history_position_chunk_size, int)
        or older_history_position_chunk_size <= 0
    ):
        raise Lfm2AttentionTangentBankError(
            "older_history_position_chunk_size must be a positive integer."
        )
    epsilon = _positive_float(norm_epsilon, label="norm_epsilon")
    resolved_device = _compute_device(compute_device)
    compute_receipt = _compute_receipt(compute_device, resolved_device)
    source_tensors = {
        "source_query_weight": _plain_tensor(
            source_query_weight,
            label="source_query_weight",
            ndim=2,
        ),
        "source_key_weight": _plain_tensor(
            source_key_weight,
            label="source_key_weight",
            ndim=2,
        ),
        "source_value_weight": _plain_tensor(
            source_value_weight,
            label="source_value_weight",
            ndim=2,
        ),
        "source_out_weight": _plain_tensor(
            source_out_weight,
            label="source_out_weight",
            ndim=2,
        ),
        "source_query_norm_weight": _plain_tensor(
            source_query_norm_weight,
            label="source_query_norm_weight",
            ndim=1,
        ),
        "source_key_norm_weight": _plain_tensor(
            source_key_norm_weight,
            label="source_key_norm_weight",
            ndim=1,
        ),
        "normalized_input_decoder": _plain_tensor(
            normalized_input_decoder,
            label="normalized_input_decoder",
            ndim=2,
        ),
        "residual_output_chart": _plain_tensor(
            residual_output_chart,
            label="residual_output_chart",
            ndim=2,
        ),
    }
    expected_hashes = {
        "source_query_weight": expected_source_query_weight_sha256,
        "source_key_weight": expected_source_key_weight_sha256,
        "source_value_weight": expected_source_value_weight_sha256,
        "source_out_weight": expected_source_out_weight_sha256,
        "source_query_norm_weight": expected_source_query_norm_weight_sha256,
        "source_key_norm_weight": expected_source_key_norm_weight_sha256,
        "normalized_input_decoder": expected_normalized_input_decoder_sha256,
        "residual_output_chart": expected_residual_output_chart_sha256,
    }
    source_hashes_before = {
        name: _authenticate_tensor(value, expected_hashes[name], label=name)
        for name, value in source_tensors.items()
    }
    q_rows, parent_hidden = map(int, source_query_weight.shape)
    if q_rows % source_query_head_count:
        raise Lfm2AttentionTangentBankError("Source Q rows do not preserve heads.")
    head_dim = q_rows // source_query_head_count
    kv_rows = source_key_value_head_count * head_dim
    child_hidden, decoder_parent_hidden = map(int, normalized_input_decoder.shape)
    chart_parent_hidden, chart_child_hidden = map(int, residual_output_chart.shape)
    if (
        tuple(source_key_weight.shape) != (kv_rows, parent_hidden)
        or tuple(source_value_weight.shape) != (kv_rows, parent_hidden)
        or tuple(source_out_weight.shape) != (parent_hidden, q_rows)
        or tuple(source_query_norm_weight.shape) != (head_dim,)
        or tuple(source_key_norm_weight.shape) != (head_dim,)
        or decoder_parent_hidden != parent_hidden
        or chart_parent_hidden != parent_hidden
        or chart_child_hidden <= 0
    ):
        raise Lfm2AttentionTangentBankError(
            "Source Q/K/V/norm/decoder geometry is inconsistent."
        )
    cases = tuple(fit_cases)
    if not cases or any(
        not isinstance(case, Lfm2AttentionTangentFitCase) for case in cases
    ):
        raise Lfm2AttentionTangentBankError("fit_cases must contain typed FIT phases.")
    if len({case.case_id for case in cases}) != len(cases):
        raise Lfm2AttentionTangentBankError("FIT attention tangent case IDs repeat.")
    case_digests = tuple(
        lfm2_attention_tangent_case_evidence_sha256(case) for case in cases
    )
    fit_evidence_sha256 = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-attention-tangent-fit-bank-v1",
            "case_sha256": list(case_digests),
        }
    )
    if fit_evidence_sha256 != _digest(
        expected_fit_case_evidence_sha256,
        label="expected FIT attention tangent evidence",
    ):
        raise Lfm2AttentionTangentAuthenticationError(
            "FIT attention tangent cases differ from their evidence seal."
        )

    q_per_group = source_query_head_count // source_key_value_head_count
    query_covariance = torch.zeros(
        (source_key_value_head_count, source_key_value_head_count),
        dtype=torch.float64,
        device=resolved_device,
    )
    history_covariance = torch.zeros_like(query_covariance)
    joint_covariance = torch.zeros_like(query_covariance)
    older_key_value_covariance = torch.zeros_like(query_covariance)
    decoder = normalized_input_decoder.detach().to(
        device=resolved_device,
        dtype=torch.float64,
    )
    q_weight = source_query_weight.detach().to(
        device=resolved_device,
        dtype=torch.float64,
    )
    k_weight = source_key_weight.detach().to(
        device=resolved_device,
        dtype=torch.float64,
    )
    v_weight = source_value_weight.detach().to(
        device=resolved_device,
        dtype=torch.float64,
    )
    o_weight = source_out_weight.detach().to(
        device=resolved_device,
        dtype=torch.float64,
    )
    q_norm_weight = source_query_norm_weight.detach().to(
        device=resolved_device,
        dtype=torch.float64,
    )
    k_norm_weight = source_key_norm_weight.detach().to(
        device=resolved_device,
        dtype=torch.float64,
    )
    residual_chart = residual_output_chart.detach().to(
        device=resolved_device,
        dtype=torch.float64,
    )
    q_directions = (decoder @ q_weight.T).reshape(
        child_hidden,
        source_query_head_count,
        head_dim,
    )
    k_directions = (decoder @ k_weight.T).reshape(
        child_hidden,
        source_key_value_head_count,
        head_dim,
    )
    v_directions = (decoder @ v_weight.T).reshape(
        child_hidden,
        source_key_value_head_count,
        head_dim,
    )
    group_width = q_per_group * head_dim
    group_residual_maps = (
        (residual_chart.T @ o_weight)
        .reshape(chart_child_hidden, source_key_value_head_count, group_width)
        .permute(1, 2, 0)
        .contiguous()
    )
    query_context_row_count = 0
    history_context_row_count = 0
    joint_context_row_count = 0
    older_key_value_context_row_count = 0
    query_weight_sum = 0.0
    history_weight_sum = 0.0
    joint_weight_sum = 0.0
    older_key_value_weight_sum = 0.0
    case_receipts = []
    with torch.inference_mode():
        for case, case_sha in zip(cases, case_digests, strict=True):
            tensors_cpu = _validated_case_tensors(case)
            batch, query_rows, suffix_rows, history_rows = _case_geometry(
                tensors_cpu,
                query_heads=source_query_head_count,
                key_value_heads=source_key_value_head_count,
                head_dim=head_dim,
            )
            phase_weight = _positive_float(
                case.phase_weight,
                label=f"{case.case_id} phase_weight",
            )
            tensors = {
                name: value.detach().to(device=resolved_device, dtype=torch.float64)
                for name, value in tensors_cpu.items()
            }
            q_linear = tensors["query_linear_suffix"].permute(0, 2, 1, 3)
            k_linear = tensors["key_linear_suffix"].permute(0, 2, 1, 3)
            k_linear_history = tensors["key_linear_history"].permute(0, 2, 1, 3)
            q_pre = tensors["query_normalized_pre_rope"][:, :, -query_rows:, :]
            k_pre = tensors["key_normalized_pre_rope"][:, :, -suffix_rows:, :]
            k_pre_history = tensors["key_normalized_pre_rope_history"]
            cos_q = tensors["rope_cos"][:, -query_rows:, :]
            sin_q = tensors["rope_sin"][:, -query_rows:, :]
            cos_k_history = tensors["key_rope_cos_history"]
            sin_k_history = tensors["key_rope_sin_history"]
            query = tensors["query_rotary_suffix"]
            keys = tensors["key_history"]
            values = tensors["value_history"]
            qk_scores = tensors["qk_scores"]
            cached_probability = tensors["probabilities"]
            mask = tensors["additive_mask"]
            repeated_keys = keys.repeat_interleave(q_per_group, dim=1)
            repeated_values = values.repeat_interleave(q_per_group, dim=1)

            # Replay cached causal primals in FP64 for diagnostics.  The trace
            # collector authenticated the execution-device capture and the
            # selective session authenticated these exact tensor payloads.  A
            # cross-dtype reconstruction is not bitwise equivalent to the
            # fused/BF16 capture path, so its envelope cannot veto the cache.
            replayed_qk_scores = (query @ repeated_keys.transpose(2, 3)) * float(
                head_dim**-0.5
            )
            qk_score_replay_diagnostic = _cross_dtype_replay_diagnostic(
                replayed_qk_scores,
                qk_scores,
                observed_capture_dtype=tensors_cpu["qk_scores"].dtype,
            )
            # The collector defines softmax in FP32 and only then casts the
            # stored probability to the activation dtype.  Reconstruct the
            # normalized primal from sealed pre-softmax scores plus mask so
            # sum(P)=1 is retained by the analytic softmax JVP.
            probability = torch.softmax(
                (qk_scores + mask).to(torch.float32),
                dim=-1,
            ).to(torch.float64)
            probability = probability / probability.sum(dim=-1, keepdim=True)
            analytic_probability_receipt = {
                "source": "authenticated_qk_scores_plus_additive_mask",
                "collector_softmax_compute_dtype_replayed": "torch.float32",
                "fp64_row_normalization_applied_for_analytic_jvp": True,
                "analytic_jvp_storage_dtype": "torch.float64",
                "maximum_probability_row_sum_absolute_error": float(
                    (probability.sum(dim=-1) - 1.0).abs().max().item()
                ),
                "cached_activation_dtype_probability_used_as_jvp_primal": False,
            }
            probability_replay_diagnostic = _cross_dtype_replay_diagnostic(
                probability,
                cached_probability,
                observed_capture_dtype=tensors_cpu["probabilities"].dtype,
            )
            expected_q_pre = (
                q_linear
                * torch.rsqrt(q_linear.square().mean(dim=-1, keepdim=True) + epsilon)
                * q_norm_weight.reshape(1, 1, 1, -1)
            ).permute(0, 2, 1, 3)
            expected_k_pre = (
                k_linear
                * torch.rsqrt(k_linear.square().mean(dim=-1, keepdim=True) + epsilon)
                * k_norm_weight.reshape(1, 1, 1, -1)
            ).permute(0, 2, 1, 3)
            expected_k_pre_history = (
                k_linear_history
                * torch.rsqrt(
                    k_linear_history.square().mean(dim=-1, keepdim=True) + epsilon
                )
                * k_norm_weight.reshape(1, 1, 1, -1)
            ).permute(0, 2, 1, 3)
            replayed_key_history = _apply_rope(
                k_pre_history.permute(0, 2, 1, 3),
                cos_k_history.unsqueeze(2),
                sin_k_history.unsqueeze(2),
            ).permute(0, 2, 1, 3)
            expected_query = _apply_rope(
                q_pre.permute(0, 2, 1, 3),
                cos_q.unsqueeze(2),
                sin_q.unsqueeze(2),
            ).permute(0, 2, 1, 3)
            replay_diagnostics = {
                "qk_scores_from_q_k": qk_score_replay_diagnostic,
                "cached_probability_from_sealed_qk_scores_mask": (
                    probability_replay_diagnostic
                ),
                "query_normalized_pre_rope_from_linear": (
                    _cross_dtype_replay_diagnostic(
                        expected_q_pre,
                        q_pre,
                        observed_capture_dtype=tensors_cpu[
                            "query_normalized_pre_rope"
                        ].dtype,
                    )
                ),
                "key_normalized_pre_rope_from_linear": (
                    _cross_dtype_replay_diagnostic(
                        expected_k_pre,
                        k_pre,
                        observed_capture_dtype=tensors_cpu[
                            "key_normalized_pre_rope"
                        ].dtype,
                    )
                ),
                "stitched_key_history_pre_rope_from_linear": (
                    _cross_dtype_replay_diagnostic(
                        expected_k_pre_history,
                        k_pre_history,
                        observed_capture_dtype=tensors_cpu[
                            "key_normalized_pre_rope_history"
                        ].dtype,
                    )
                ),
                "stitched_key_history_rotary_from_pre_rope": (
                    _cross_dtype_replay_diagnostic(
                        replayed_key_history,
                        keys,
                        observed_capture_dtype=tensors_cpu["key_history"].dtype,
                    )
                ),
                "query_rotary_from_pre_rope": _cross_dtype_replay_diagnostic(
                    expected_query,
                    query,
                    observed_capture_dtype=tensors_cpu["query_rotary_suffix"].dtype,
                ),
            }

            query_row_weight = phase_weight / (batch * query_rows * child_hidden)
            history_row_weight = phase_weight / (
                batch * suffix_rows * child_hidden * query_rows
            )
            older_prefix_rows = history_rows - suffix_rows
            maximum_older_tile_from_workspace = max(
                1,
                8_000_000
                // (
                    batch
                    * min(child_hidden, direction_chunk_size)
                    * source_query_head_count
                    * query_rows
                    * head_dim
                ),
            )
            older_position_tile = min(
                older_history_position_chunk_size,
                maximum_older_tile_from_workspace,
            )
            older_key_value_row_weight = (
                None
                if older_prefix_rows == 0
                else phase_weight
                / (batch * older_prefix_rows * child_hidden * query_rows)
            )
            base_context = probability @ repeated_values
            for direction_start in range(0, child_hidden, direction_chunk_size):
                direction_stop = min(
                    child_hidden,
                    direction_start + direction_chunk_size,
                )
                q_direction_chunk = q_directions[direction_start:direction_stop]
                k_direction_chunk = k_directions[direction_start:direction_stop]
                v_direction_chunk = v_directions[direction_start:direction_stop]
                chunk = direction_stop - direction_start
                # All query positions are independent perturbations: a query
                # row only changes its own attention row.  Batch those exact
                # JVPs into one launch without adding cross-position terms.
                delta_q_pre = _rms_norm_sequence_jvp(
                    q_linear,
                    q_direction_chunk,
                    q_norm_weight,
                    epsilon,
                )
                delta_q = _apply_rope(
                    delta_q_pre,
                    cos_q.reshape(batch, 1, query_rows, 1, head_dim),
                    sin_q.reshape(batch, 1, query_rows, 1, head_dim),
                )
                delta_score = torch.einsum(
                    "bcqhd,bhkd->bcqhk",
                    delta_q,
                    repeated_keys,
                ) * float(head_dim**-0.5)
                p = probability.permute(0, 2, 1, 3).unsqueeze(1)
                delta_probability = p * (
                    delta_score - (p * delta_score).sum(dim=-1, keepdim=True)
                )
                query_delta_context = torch.einsum(
                    "bcqhk,bhkd->bcqhd",
                    delta_probability,
                    repeated_values,
                )
                rows = query_delta_context.reshape(
                    batch * chunk * query_rows,
                    source_query_head_count,
                    head_dim,
                )
                _accumulate_context_covariance(
                    query_covariance,
                    rows,
                    row_weight=query_row_weight,
                    groups=source_key_value_head_count,
                    q_per_group=q_per_group,
                    group_residual_maps=group_residual_maps,
                )
                query_context_row_count += int(rows.shape[0])
                query_weight_sum += query_row_weight * int(rows.shape[0])
                delta_v_heads = v_direction_chunk.repeat_interleave(
                    q_per_group,
                    dim=1,
                )

                # K/V positions are also independent perturbations.  Tile the
                # suffix-position axis to keep peak memory bounded while
                # retaining every direction, position, query, and head.
                for suffix_start in range(
                    0,
                    suffix_rows,
                    history_position_chunk_size,
                ):
                    suffix_stop = min(
                        suffix_rows,
                        suffix_start + history_position_chunk_size,
                    )
                    suffix_chunk = suffix_stop - suffix_start
                    history_anchor_start = history_rows - suffix_rows
                    delta_k_pre = _rms_norm_sequence_jvp(
                        k_linear_history[
                            :,
                            history_anchor_start + suffix_start : history_anchor_start
                            + suffix_stop,
                        ],
                        k_direction_chunk,
                        k_norm_weight,
                        epsilon,
                    )
                    delta_k = _apply_rope(
                        delta_k_pre,
                        cos_k_history[
                            :,
                            history_anchor_start + suffix_start : history_anchor_start
                            + suffix_stop,
                        ].reshape(
                            batch,
                            1,
                            suffix_chunk,
                            1,
                            head_dim,
                        ),
                        sin_k_history[
                            :,
                            history_anchor_start + suffix_start : history_anchor_start
                            + suffix_stop,
                        ].reshape(
                            batch,
                            1,
                            suffix_chunk,
                            1,
                            head_dim,
                        ),
                    ).repeat_interleave(q_per_group, dim=3)
                    delta_v = delta_v_heads.reshape(
                        1,
                        chunk,
                        1,
                        source_query_head_count,
                        head_dim,
                    ).expand(batch, -1, suffix_chunk, -1, -1)
                    history_indices = torch.arange(
                        history_rows - suffix_rows + suffix_start,
                        history_rows - suffix_rows + suffix_stop,
                        device=resolved_device,
                    )
                    score_slot = torch.einsum(
                        "bhqd,bcshd->bcshq",
                        query,
                        delta_k,
                    ) * float(head_dim**-0.5)
                    p_slot = (
                        probability.index_select(-1, history_indices)
                        .permute(0, 3, 1, 2)
                        .unsqueeze(1)
                    )
                    value_slot = (
                        repeated_values.index_select(2, history_indices)
                        .permute(0, 2, 1, 3)
                        .unsqueeze(1)
                        .unsqueeze(4)
                    )
                    delta_context = p_slot.unsqueeze(-1) * (
                        score_slot.unsqueeze(-1)
                        * (value_slot - base_context.unsqueeze(1).unsqueeze(1))
                        + delta_v.unsqueeze(4)
                    )
                    rows = delta_context.permute(0, 1, 2, 4, 3, 5).reshape(
                        batch * chunk * suffix_chunk * query_rows,
                        source_query_head_count,
                        head_dim,
                    )
                    _accumulate_context_covariance(
                        history_covariance,
                        rows,
                        row_weight=history_row_weight,
                        groups=source_key_value_head_count,
                        q_per_group=q_per_group,
                        group_residual_maps=group_residual_maps,
                    )
                    history_context_row_count += int(rows.shape[0])
                    history_weight_sum += history_row_weight * int(rows.shape[0])

                    # The residual perturbation at sampled token s changes
                    # Q_s, K_s, and V_s together.  Add the Q contribution only
                    # at output query row s before accumulating the functional
                    # joint Jacobian covariance; this retains the cross terms.
                    query_row_selector = torch.nn.functional.one_hot(
                        torch.arange(
                            suffix_start,
                            suffix_stop,
                            device=resolved_device,
                        ),
                        num_classes=query_rows,
                    ).to(torch.float64)
                    query_row_selector = query_row_selector.reshape(
                        1,
                        1,
                        suffix_chunk,
                        1,
                        query_rows,
                        1,
                    )
                    joint_delta_context = delta_context + (
                        query_delta_context[
                            :,
                            :,
                            suffix_start:suffix_stop,
                            :,
                            :,
                        ].unsqueeze(4)
                        * query_row_selector
                    )
                    joint_rows = joint_delta_context.permute(
                        0,
                        1,
                        2,
                        4,
                        3,
                        5,
                    ).reshape(
                        batch * chunk * suffix_chunk * query_rows,
                        source_query_head_count,
                        head_dim,
                    )
                    _accumulate_context_covariance(
                        joint_covariance,
                        joint_rows,
                        row_weight=history_row_weight,
                        groups=source_key_value_head_count,
                        q_per_group=q_per_group,
                        group_residual_maps=group_residual_maps,
                    )
                    joint_context_row_count += int(joint_rows.shape[0])
                    joint_weight_sum += history_row_weight * int(joint_rows.shape[0])

                # Prior authenticated phases supply every older K RMSNorm/RoPE
                # anchor.  K and V therefore move together for each older
                # history token, non-overlapping with the sampled suffix block.
                for prefix_start in range(
                    0,
                    older_prefix_rows,
                    older_position_tile,
                ):
                    prefix_stop = min(
                        older_prefix_rows,
                        prefix_start + older_position_tile,
                    )
                    prefix_chunk = prefix_stop - prefix_start
                    prefix_indices = torch.arange(
                        prefix_start,
                        prefix_stop,
                        device=resolved_device,
                    )
                    prefix_delta_k_pre = _rms_norm_sequence_jvp(
                        k_linear_history[:, prefix_start:prefix_stop],
                        k_direction_chunk,
                        k_norm_weight,
                        epsilon,
                    )
                    prefix_delta_k = _apply_rope(
                        prefix_delta_k_pre,
                        cos_k_history[:, prefix_start:prefix_stop].reshape(
                            batch,
                            1,
                            prefix_chunk,
                            1,
                            head_dim,
                        ),
                        sin_k_history[:, prefix_start:prefix_stop].reshape(
                            batch,
                            1,
                            prefix_chunk,
                            1,
                            head_dim,
                        ),
                    ).repeat_interleave(q_per_group, dim=3)
                    prefix_score_slot = torch.einsum(
                        "bhqd,bcshd->bcshq",
                        query,
                        prefix_delta_k,
                    ) * float(head_dim**-0.5)
                    prefix_probability = (
                        probability.index_select(-1, prefix_indices)
                        .permute(0, 3, 1, 2)
                        .unsqueeze(1)
                    )
                    prefix_value_slot = (
                        repeated_values.index_select(2, prefix_indices)
                        .permute(0, 2, 1, 3)
                        .unsqueeze(1)
                        .unsqueeze(4)
                    )
                    prefix_delta_value = delta_v_heads.reshape(
                        1,
                        chunk,
                        1,
                        source_query_head_count,
                        1,
                        head_dim,
                    )
                    older_key_value_delta_context = prefix_probability.unsqueeze(-1) * (
                        prefix_score_slot.unsqueeze(-1)
                        * (prefix_value_slot - base_context.unsqueeze(1).unsqueeze(1))
                        + prefix_delta_value
                    )
                    older_rows = older_key_value_delta_context.permute(
                        0,
                        1,
                        2,
                        4,
                        3,
                        5,
                    ).reshape(
                        batch * chunk * prefix_chunk * query_rows,
                        source_query_head_count,
                        head_dim,
                    )
                    _accumulate_context_covariance(
                        older_key_value_covariance,
                        older_rows,
                        row_weight=older_key_value_row_weight,
                        groups=source_key_value_head_count,
                        q_per_group=q_per_group,
                        group_residual_maps=group_residual_maps,
                    )
                    older_key_value_context_row_count += int(older_rows.shape[0])
                    older_key_value_weight_sum += older_key_value_row_weight * int(
                        older_rows.shape[0]
                    )
            case_receipts.append(
                {
                    "case_id": case.case_id,
                    "case_evidence_sha256": case_sha,
                    "source_receipt_sha256": case.source_receipt_sha256,
                    "phase_weight": phase_weight,
                    "batch_size": batch,
                    "sampled_query_count": query_rows,
                    "sampled_suffix_key_value_count": suffix_rows,
                    "complete_key_value_history_count": history_rows,
                    "older_history_position_tile": older_position_tile,
                    "historical_prefix_cross_phase_k_anchor_count": older_prefix_rows,
                    "older_prefix_key_perturbations_materialized": (
                        older_prefix_rows > 0
                    ),
                    "older_prefix_key_tangent_coverage_complete": True,
                    "older_prefix_value_perturbations_materialized": (
                        older_prefix_rows > 0
                    ),
                    "older_prefix_value_tangent_coverage_complete": True,
                    "cross_dtype_fp64_replay_diagnostics": replay_diagnostics,
                    "analytic_softmax_probability_primal": (
                        analytic_probability_receipt
                    ),
                }
            )

    query_snapshot = _covariance_snapshot(
        query_covariance,
        block_id=f"{site_id.strip()}:query-local",
        perturbation_kind="query_local_q",
        query_heads=source_query_head_count,
        key_value_heads=source_key_value_head_count,
        head_dim=head_dim,
        original_tangent_context_row_count=query_context_row_count,
        original_weight_sum=query_weight_sum,
    )
    history_snapshot = _covariance_snapshot(
        history_covariance,
        block_id=f"{site_id.strip()}:sampled-suffix-history-kv",
        perturbation_kind="sampled_suffix_historical_key_value",
        query_heads=source_query_head_count,
        key_value_heads=source_key_value_head_count,
        head_dim=head_dim,
        original_tangent_context_row_count=history_context_row_count,
        original_weight_sum=history_weight_sum,
    )
    joint_snapshot = _covariance_snapshot(
        joint_covariance,
        block_id=f"{site_id.strip()}:joint-sampled-token-qkv",
        perturbation_kind="joint_sampled_token_query_key_value",
        query_heads=source_query_head_count,
        key_value_heads=source_key_value_head_count,
        head_dim=head_dim,
        original_tangent_context_row_count=joint_context_row_count,
        original_weight_sum=joint_weight_sum,
    )
    older_key_value_snapshot = _covariance_snapshot(
        older_key_value_covariance,
        block_id=f"{site_id.strip()}:older-prefix-key-value",
        perturbation_kind="older_prefix_key_value",
        query_heads=source_query_head_count,
        key_value_heads=source_key_value_head_count,
        head_dim=head_dim,
        original_tangent_context_row_count=older_key_value_context_row_count,
        original_weight_sum=older_key_value_weight_sum,
    )
    case_digests_after = tuple(
        lfm2_attention_tangent_case_evidence_sha256(case) for case in cases
    )
    if case_digests_after != case_digests:
        raise Lfm2AttentionTangentAuthenticationError(
            "A cached FIT attention tensor changed during tangent construction."
        )
    source_hashes_after = {
        name: _tensor_sha256(value) for name, value in source_tensors.items()
    }
    if source_hashes_after != source_hashes_before:
        raise Lfm2AttentionTangentAuthenticationError(
            "A source tensor changed during attention tangent construction."
        )
    stitched_prefix_count = sum(
        int(value["historical_prefix_cross_phase_k_anchor_count"])
        for value in case_receipts
    )
    replay_diagnostic_names = (
        "qk_scores_from_q_k",
        "cached_probability_from_sealed_qk_scores_mask",
        "query_normalized_pre_rope_from_linear",
        "key_normalized_pre_rope_from_linear",
        "stitched_key_history_pre_rope_from_linear",
        "stitched_key_history_rotary_from_pre_rope",
        "query_rotary_from_pre_rope",
    )
    replay_diagnostic_summary = {
        name: {
            "case_count": len(case_receipts),
            "outside_legacy_allclose_envelope_count": sum(
                not bool(
                    value["cross_dtype_fp64_replay_diagnostics"][name][
                        "within_legacy_allclose_envelope"
                    ]
                )
                for value in case_receipts
            ),
            "maximum_absolute_error": max(
                float(
                    value["cross_dtype_fp64_replay_diagnostics"][name][
                        "maximum_absolute_error"
                    ]
                )
                for value in case_receipts
            ),
            "hard_authentication_authority": False,
            "semantic_selection_authority": False,
        }
        for name in replay_diagnostic_names
    }
    body = {
        "schema": LFM2_ATTENTION_TANGENT_BANK_SCHEMA,
        "site_id": site_id.strip(),
        "source_hashes": source_hashes_before,
        "fit_case_evidence_sha256": fit_evidence_sha256,
        "fit_case_evidence_immutable_before_and_after": True,
        "fit_cases": case_receipts,
        "geometry": {
            "source_query_head_count": source_query_head_count,
            "source_key_value_head_count": source_key_value_head_count,
            "queries_per_key_value": q_per_group,
            "head_dim": head_dim,
            "parent_hidden_size": parent_hidden,
            "child_hidden_size": child_hidden,
            "residual_chart_child_hidden_size": chart_child_hidden,
            "authenticated_child_decoder_direction_count": child_hidden,
        },
        "group_feature_semantics": (
            "each_group_context_tangent_mapped_through_source_o_block_then_"
            "residual_output_chart"
        ),
        "source_o_and_residual_chart_applied_before_group_covariance": True,
        "functional_group_pod_is_residual_output_attention_aware": True,
        "norm_epsilon": epsilon,
        "direction_policy": "every_authenticated_child_normalized_input_basis_row",
        "direction_chunk_size": direction_chunk_size,
        "query_position_batching_preserves_independent_jvps": True,
        "history_position_chunk_size": history_position_chunk_size,
        "history_position_chunking_preserves_independent_jvps": True,
        "older_history_position_chunk_size": older_history_position_chunk_size,
        "older_history_workspace_element_cap": 8_000_000,
        "older_history_adaptive_chunking_preserves_independent_jvps": True,
        "analytic_jvp_formula": {
            "delta_score": "(delta_q@k.T + q@delta_k.T)/sqrt(head_dim)",
            "delta_probability": ("p*(delta_score-sum(p*delta_score,key_axis))"),
            "delta_context": "delta_probability@v + p@delta_v",
            "sampled_history_contraction": (
                "p_h*(delta_score_h*(v_h-context)+delta_v_h)"
            ),
            "sampled_history_contraction_is_exact_algebraic_form_of_full_jvp": True,
            "probability_primal": (
                "row_normalize(float64(softmax(authenticated_qk_scores+mask,"
                "dtype=float32)))"
            ),
            "cached_bfloat16_probability_used_as_analytic_primal": False,
        },
        "maximum_analytic_probability_row_sum_absolute_error": max(
            float(
                value["analytic_softmax_probability_primal"][
                    "maximum_probability_row_sum_absolute_error"
                ]
            )
            for value in case_receipts
        ),
        "query_local_reduction_receipt": _deep_thaw(query_snapshot.reduction_receipt),
        "query_local_reduction_sha256": query_snapshot.reduction_receipt[
            "result_sha256"
        ],
        "sampled_suffix_history_key_value_reduction_receipt": _deep_thaw(
            history_snapshot.reduction_receipt
        ),
        "sampled_suffix_history_key_value_reduction_sha256": (
            history_snapshot.reduction_receipt["result_sha256"]
        ),
        "joint_sampled_token_query_key_value_reduction_receipt": _deep_thaw(
            joint_snapshot.reduction_receipt
        ),
        "joint_sampled_token_query_key_value_reduction_sha256": (
            joint_snapshot.reduction_receipt["result_sha256"]
        ),
        "older_prefix_key_value_reduction_receipt": _deep_thaw(
            older_key_value_snapshot.reduction_receipt
        ),
        "older_prefix_key_value_reduction_sha256": (
            older_key_value_snapshot.reduction_receipt["result_sha256"]
        ),
        "query_local_and_history_key_value_evidence_separately_labelled": True,
        "query_local_and_sampled_suffix_key_value_partial_covariances_are_"
        "diagnostic_only": True,
        "functional_pod_uses_joint_sampled_token_qkv_cross_terms": True,
        "functional_pod_double_counts_partial_q_or_kv_covariance": False,
        "functional_pod_uses_nonoverlapping_older_prefix_key_value_tangent": True,
        "complete_value_history_tangent_coverage": True,
        "complete_key_history_tangent_coverage": True,
        "complete_key_value_history_tangent_coverage": True,
        "cross_phase_stitched_historical_prefix_key_position_count": (
            stitched_prefix_count
        ),
        "unsupported_historical_prefix_key_position_count": 0,
        "precise_missing_cached_evidence_field_for_older_history_key_tangent": None,
        "precise_missing_authenticated_transform_for_older_history_key_tangent": None,
        "full_key_history_anchor_evidence_provided_to_bank": True,
        "cross_phase_older_key_anchor_stitch_is_feasible_when_prior_phase_"
        "sequence_is_loaded": True,
        "cross_phase_older_key_anchor_stitch_implemented": True,
        "partial_diagnostic_family_scope_is_explicit_and_not_promoted_to_"
        "functional_complete": True,
        "cross_dtype_fp64_replay_diagnostic_summary": replay_diagnostic_summary,
        "cross_dtype_fp64_replay_has_hard_authentication_authority": False,
        "cross_dtype_fp64_replay_has_semantic_selection_authority": False,
        "fp64_replay_mismatch_does_not_reject_authenticated_capture": True,
        "cached_tensor_hashes_and_authenticated_session_remain_authoritative": True,
        "heldout_tensor_payloads_used": 0,
        "full_tangent_bank_materialized_as_duplicate_payload": False,
        "covariance_equivalent_reduction_used": True,
        "tangent_blocks_affect_group_pod_only": True,
        "functional_group_pod_tangent_block_count": 2,
        "direct_parent_qkv_producer_row_copy_remains_in_standalone_compiler": True,
        "dense_parent_output_reconstruction_remains_in_standalone_compiler": True,
        "compute_execution": dict(compute_receipt),
        "operation_audit": dict(_OPERATION_AUDIT),
        "numeric_diagnostics_have_semantic_decision_authority": False,
        "binary_manual_quality_gate_created": False,
    }
    report = MappingProxyType({**body, "bank_sha256": canonical_sha256(body)})
    return Lfm2AttentionTangentBank(
        site_id=site_id.strip(),
        query_local=query_snapshot,
        sampled_suffix_history_key_value=history_snapshot,
        joint_sampled_token_query_key_value=joint_snapshot,
        older_prefix_key_value=older_key_value_snapshot,
        report=report,
    )


__all__ = [
    "LFM2_ATTENTION_TANGENT_BANK_SCHEMA",
    "Lfm2AttentionTangentAuthenticationError",
    "Lfm2AttentionTangentBank",
    "Lfm2AttentionTangentBankError",
    "Lfm2AttentionTangentFitCase",
    "Lfm2AttentionTangentSnapshot",
    "build_lfm2_attention_tangent_bank",
    "lfm2_attention_causal_jvp",
    "lfm2_attention_tangent_case_evidence_sha256",
]
