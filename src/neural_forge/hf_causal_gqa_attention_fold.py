"""Family-neutral coupled GQA/RoPE residual-attention folding.

This module extracts the architecture-independent ideas from the CPCF
attention-dimension and value-dimension experiments into one dense causal
operator.  Every source Q and KV head participates in a deterministic
barycentric target head, RoPE pairs are contracted as pairs, Q/K are fitted
in the target's pre-RoPE coordinates, and V/O are compiled only after the
installed Q/K probabilities have been replayed.

All solves are closed form.  There is no optimizer, autograd, parent model,
parameter pruning, quantization, sharing, or runtime sidecar.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256


SCHEMA = "neural-forge-hf-causal-coupled-gqa-rope-fold-v1"


class HFCausalGQAAttentionFoldError(ValueError):
    """The coupled attention fold is unsupported or its evidence is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise HFCausalGQAAttentionFoldError(
            "PyTorch is required for causal GQA folding."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalGQAAttentionFoldError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalGQAAttentionFoldError(f"{name} must be positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise HFCausalGQAAttentionFoldError(f"{name} must be positive and finite.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalGQAAttentionFoldError(f"{name} must be non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise HFCausalGQAAttentionFoldError(
            f"{name} must be finite and non-negative."
        )
    return result


def _tensor(value: Any, name: str, *, ndim: int | None = None) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise HFCausalGQAAttentionFoldError(f"{name} must be a tensor.")
    if ndim is not None and value.ndim != ndim:
        raise HFCausalGQAAttentionFoldError(
            f"{name} must be rank {ndim}, got rank {value.ndim}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise HFCausalGQAAttentionFoldError(f"{name} contains non-finite values.")
    return result


def _mask(value: Any, name: str, shape: tuple[int, int]) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise HFCausalGQAAttentionFoldError(f"{name} must have shape {shape}.")
    return value.detach().to(device="cpu", dtype=torch.bool).contiguous()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _report(body: Mapping[str, Any]) -> Mapping[str, Any]:
    # Round-trip first so tensor/numpy scalars cannot enter receipts.
    decoded = json.loads(json.dumps(dict(body), allow_nan=False, sort_keys=True))
    decoded["report_sha256"] = canonical_sha256(decoded)
    return MappingProxyType(decoded)


def _balanced_groups(total: int, count: int) -> tuple[tuple[int, ...], ...]:
    total = _positive_int(total, "group source count")
    count = _positive_int(count, "group target count")
    if count > total:
        raise HFCausalGQAAttentionFoldError(
            "Attention head expansion is outside the contraction-only fold."
        )
    quotient, remainder = divmod(total, count)
    groups = []
    cursor = 0
    for index in range(count):
        width = quotient + (1 if index < remainder else 0)
        groups.append(tuple(range(cursor, cursor + width)))
        cursor += width
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class HFCausalGQATopology:
    query_heads: int
    key_value_heads: int
    head_dim: int
    rope_theta: float
    attention_scaling: float
    query_key_norm: str = "identity"
    rms_norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("query_heads", "key_value_heads", "head_dim"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.query_heads % self.key_value_heads:
            raise HFCausalGQAAttentionFoldError(
                "query_heads must be divisible by key_value_heads."
            )
        if self.head_dim % 2:
            raise HFCausalGQAAttentionFoldError("RoPE head_dim must be even.")
        object.__setattr__(
            self, "rope_theta", _positive_float(self.rope_theta, "rope_theta")
        )
        object.__setattr__(
            self,
            "attention_scaling",
            _positive_float(self.attention_scaling, "attention_scaling"),
        )
        norm = str(self.query_key_norm).strip().lower()
        if norm not in {"identity", "rms_norm"}:
            raise HFCausalGQAAttentionFoldError(
                "query_key_norm must be 'identity' or 'rms_norm'."
            )
        object.__setattr__(self, "query_key_norm", norm)
        object.__setattr__(
            self,
            "rms_norm_epsilon",
            _positive_float(self.rms_norm_epsilon, "rms_norm_epsilon"),
        )

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.key_value_heads * self.head_dim

    @property
    def queries_per_key_value(self) -> int:
        return self.query_heads // self.key_value_heads

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "attention_scaling": self.attention_scaling,
            "query_key_norm": self.query_key_norm,
            "rms_norm_epsilon": self.rms_norm_epsilon,
        }


@dataclass(frozen=True, slots=True)
class HFCausalGQABarycentricAssignment:
    source_query_groups: tuple[tuple[int, ...], ...]
    source_key_value_groups: tuple[tuple[int, ...], ...]
    selected_rope_pairs_by_target_kv: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for name in (
            "source_query_groups",
            "source_key_value_groups",
            "selected_rope_pairs_by_target_kv",
        ):
            values = tuple(tuple(int(item) for item in group) for group in getattr(self, name))
            if not values or any(not group for group in values):
                raise HFCausalGQAAttentionFoldError(f"{name} cannot be empty.")
            object.__setattr__(self, name, values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_query_groups": [list(value) for value in self.source_query_groups],
            "source_key_value_groups": [
                list(value) for value in self.source_key_value_groups
            ],
            "selected_rope_pairs_by_target_kv": [
                list(value) for value in self.selected_rope_pairs_by_target_kv
            ],
        }


def _build_assignment(
    source: HFCausalGQATopology,
    target: HFCausalGQATopology,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    source_kv_groups = _balanced_groups(source.key_value_heads, target.key_value_heads)
    source_q_per_kv = source.queries_per_key_value
    target_q_per_kv = target.queries_per_key_value
    query_groups: list[tuple[int, ...]] = []
    for source_kv_group in source_kv_groups:
        candidates = tuple(
            source_kv * source_q_per_kv + local
            for source_kv in source_kv_group
            for local in range(source_q_per_kv)
        )
        local_groups = _balanced_groups(len(candidates), target_q_per_kv)
        query_groups.extend(tuple(candidates[index] for index in group) for group in local_groups)
    if tuple(sorted(index for group in query_groups for index in group)) != tuple(
        range(source.query_heads)
    ):
        raise HFCausalGQAAttentionFoldError(
            "Barycentric query assignment did not cover every source head."
        )
    if tuple(sorted(index for group in source_kv_groups for index in group)) != tuple(
        range(source.key_value_heads)
    ):
        raise HFCausalGQAAttentionFoldError(
            "Barycentric KV assignment did not cover every source head."
        )
    return tuple(query_groups), source_kv_groups


def _reshape_heads(value: Any, heads: int, dim: int, name: str) -> Any:
    value = _tensor(value, name, ndim=3)
    batch, sequence, width = (int(value.shape[0]), int(value.shape[1]), int(value.shape[2]))
    if width != heads * dim:
        raise HFCausalGQAAttentionFoldError(
            f"{name} width {width} differs from {heads}*{dim}."
        )
    return value.reshape(batch, sequence, heads, dim).permute(0, 2, 1, 3).contiguous()


def _head_rms_norm(value: Any, weight: Any, epsilon: float, name: str) -> Any:
    vector = _tensor(weight, name, ndim=1)
    if int(vector.numel()) != int(value.shape[-1]):
        raise HFCausalGQAAttentionFoldError(f"{name} has the wrong head width.")
    normalized = value * (value.square().mean(dim=-1, keepdim=True) + epsilon).rsqrt()
    return normalized * vector.reshape(1, 1, 1, -1)


def _rope_cos_sin(position_ids: Any, dim: int, theta: float) -> tuple[Any, Any]:
    torch = _torch()
    positions = position_ids.detach().to(device="cpu", dtype=torch.float64)
    frequencies = theta ** (
        -torch.arange(0, dim, 2, dtype=torch.float64) / float(dim)
    )
    angles = positions.unsqueeze(-1) * frequencies.reshape(1, 1, -1)
    embedding = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    return embedding.cos(), embedding.sin()


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def apply_rope(value: Any, position_ids: Any, *, theta: float) -> Any:
    cos, sin = _rope_cos_sin(position_ids, int(value.shape[-1]), theta)
    return (value * cos + _rotate_half(value) * sin).contiguous()


def invert_rope(value: Any, position_ids: Any, *, theta: float) -> Any:
    cos, sin = _rope_cos_sin(position_ids, int(value.shape[-1]), theta)
    return (value * cos - _rotate_half(value) * sin).contiguous()


def _repeat_kv(value: Any, query_heads: int) -> Any:
    repeats = query_heads // int(value.shape[1])
    return value.repeat_interleave(repeats, dim=1)


def causal_attention_probabilities(
    query: Any,
    key: Any,
    *,
    attention_mask: Any,
    scaling: float,
) -> tuple[Any, Any]:
    """Return centered scores and probabilities for one causal GQA replay."""

    torch = _torch()
    batch, heads, sequence, _dim = query.shape
    keys = _repeat_kv(key, heads)
    scores = torch.matmul(query, keys.transpose(-1, -2)) * scaling
    key_mask = attention_mask.reshape(batch, 1, 1, sequence).to(torch.bool)
    causal = torch.ones(sequence, sequence, dtype=torch.bool).tril().reshape(
        1, 1, sequence, sequence
    )
    allowed = key_mask & causal
    # Invalid/padded query rows are removed from every fit metric.  Give them a
    # deterministic diagonal fallback to keep the functional replay finite.
    no_key = ~allowed.any(dim=-1, keepdim=True)
    if bool(no_key.any().item()):
        diagonal = torch.eye(sequence, dtype=torch.bool).reshape(
            1, 1, sequence, sequence
        )
        allowed = allowed | (no_key & diagonal)
    masked_scores = scores.masked_fill(~allowed, -torch.inf)
    probabilities = torch.softmax(masked_scores, dim=-1)
    finite_scores = scores.masked_fill(~allowed, 0.0)
    denominator = allowed.sum(dim=-1, keepdim=True).clamp_min(1)
    centered = finite_scores - (
        finite_scores.sum(dim=-1, keepdim=True) / denominator
    )
    centered = centered.masked_fill(~allowed, 0.0)
    query_valid = attention_mask.reshape(batch, 1, sequence, 1).to(torch.bool)
    probabilities = probabilities * query_valid
    return centered.contiguous(), probabilities.contiguous()


def _ridge(source: Any, target: Any, ridge: float) -> Any:
    torch = _torch()
    if source.ndim != 2 or target.ndim != 2 or int(source.shape[0]) != int(target.shape[0]):
        raise HFCausalGQAAttentionFoldError("Ridge evidence matrices are incompatible.")
    gram = source.T @ source
    return torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=torch.float64),
        source.T @ target,
    )


def _masked_rows(value: Any, mask: Any) -> Any:
    trailing = tuple(int(size) for size in value.shape[2:])
    return value.reshape(int(mask.numel()), *trailing)[mask.reshape(-1)].contiguous()


def _relative_rmse(prediction: Any, target: Any) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _retained_energy(prediction: Any, target: Any) -> float:
    error = (prediction - target).square().sum()
    total = target.square().sum().clamp_min(1e-30)
    return float((1.0 - error / total).clamp(min=0.0, max=1.0).item())


def _select_rope_pairs(
    query_group_values: tuple[Any, ...],
    key_values: Any,
    value_values: Any,
    target_dim: int,
) -> tuple[int, ...]:
    torch = _torch()
    source_dim = int(key_values.shape[-1])
    if target_dim > source_dim or target_dim % 2 or source_dim % 2:
        raise HFCausalGQAAttentionFoldError(
            "RoPE dimension expansion or an odd head dimension is unsupported."
        )
    half = source_dim // 2
    energy = torch.zeros(half, dtype=torch.float64)
    for value in (*query_group_values, key_values, value_values):
        energy += value[..., :half].square().sum(dim=tuple(range(value.ndim - 1)))
        energy += value[..., half:].square().sum(dim=tuple(range(value.ndim - 1)))
    count = target_dim // 2
    selected = torch.argsort(energy, descending=True, stable=True)[:count]
    return tuple(sorted(int(value) for value in selected.tolist()))


def _contract_pairs(value: Any, selected: tuple[int, ...]) -> Any:
    torch = _torch()
    half = int(value.shape[-1]) // 2
    index = torch.tensor((*selected, *(item + half for item in selected)), dtype=torch.long)
    return value.index_select(-1, index).contiguous()


@dataclass(frozen=True, slots=True)
class HFCausalCoupledAttentionFold:
    query_weight: Any
    key_weight: Any
    value_weight: Any
    output_weight: Any
    source_topology: HFCausalGQATopology
    target_topology: HFCausalGQATopology
    assignment: HFCausalGQABarycentricAssignment
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("query_weight", "key_weight", "value_weight", "output_weight"):
            object.__setattr__(self, name, _tensor(getattr(self, name), name, ndim=2).clone())
        if int(self.query_weight.shape[0]) != self.target_topology.query_width:
            raise HFCausalGQAAttentionFoldError("query_weight has the wrong width.")
        if int(self.key_weight.shape[0]) != self.target_topology.key_value_width:
            raise HFCausalGQAAttentionFoldError("key_weight has the wrong width.")
        if self.key_weight.shape != self.value_weight.shape:
            raise HFCausalGQAAttentionFoldError("K/V target shapes differ.")
        if int(self.output_weight.shape[1]) != self.target_topology.query_width:
            raise HFCausalGQAAttentionFoldError("output_weight has the wrong input width.")
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def replay(
        self,
        normalized_child_sequences: Any,
        *,
        attention_mask: Any,
        position_ids: Any,
    ) -> tuple[Any, Any, Any]:
        torch = _torch()
        child = _tensor(normalized_child_sequences, "normalized_child_sequences", ndim=3)
        batch, sequence, width = child.shape
        if int(self.query_weight.shape[1]) != int(width):
            raise HFCausalGQAAttentionFoldError("Child replay width differs from fitted QKV.")
        mask = _mask(attention_mask, "attention_mask", (int(batch), int(sequence)))
        positions = position_ids.detach().to(device="cpu")
        query = _reshape_heads(
            child @ self.query_weight.T,
            self.target_topology.query_heads,
            self.target_topology.head_dim,
            "replayed query",
        )
        key = _reshape_heads(
            child @ self.key_weight.T,
            self.target_topology.key_value_heads,
            self.target_topology.head_dim,
            "replayed key",
        )
        value = _reshape_heads(
            child @ self.value_weight.T,
            self.target_topology.key_value_heads,
            self.target_topology.head_dim,
            "replayed value",
        )
        query = apply_rope(query, positions, theta=self.target_topology.rope_theta)
        key = apply_rope(key, positions, theta=self.target_topology.rope_theta)
        scores, probabilities = causal_attention_probabilities(
            query,
            key,
            attention_mask=mask,
            scaling=self.target_topology.attention_scaling,
        )
        contexts = torch.matmul(probabilities, _repeat_kv(value, self.target_topology.query_heads))
        concatenated = contexts.permute(0, 2, 1, 3).reshape(
            int(batch), int(sequence), self.target_topology.query_width
        )
        delta = concatenated @ self.output_weight.T
        return delta.contiguous(), scores, probabilities


def compile_hf_causal_gqa_attention_fold(
    normalized_child_sequences: Any,
    *,
    source_query_sequences: Any,
    source_key_sequences: Any,
    source_value_sequences: Any,
    target_residual_delta_sequences: Any,
    attention_mask: Any,
    position_ids: Any,
    valid_token_mask: Any,
    fit_token_mask: Any,
    validation_token_mask: Any,
    source_topology: HFCausalGQATopology,
    target_topology: HFCausalGQATopology,
    source_query_norm_weight: Any | None = None,
    source_key_norm_weight: Any | None = None,
    ridge: float = 1e-6,
) -> HFCausalCoupledAttentionFold:
    """Compile QK, replay softmax, then compile VO against a residual endpoint."""

    torch = _torch()
    if not isinstance(source_topology, HFCausalGQATopology) or not isinstance(
        target_topology, HFCausalGQATopology
    ):
        raise HFCausalGQAAttentionFoldError("source/target topology types are invalid.")
    if target_topology.query_key_norm != "identity":
        raise HFCausalGQAAttentionFoldError(
            "The first dense target compiler supports identity target Q/K norm only."
        )
    ridge = _nonnegative_float(ridge, "ridge")
    child = _tensor(normalized_child_sequences, "normalized_child_sequences", ndim=3)
    delta_target = _tensor(
        target_residual_delta_sequences, "target_residual_delta_sequences", ndim=3
    )
    batch, sequence = int(child.shape[0]), int(child.shape[1])
    if tuple(delta_target.shape[:2]) != (batch, sequence):
        raise HFCausalGQAAttentionFoldError("Residual target sequence shape differs.")
    mask = _mask(attention_mask, "attention_mask", (batch, sequence))
    valid = _mask(valid_token_mask, "valid_token_mask", (batch, sequence))
    fit = _mask(fit_token_mask, "fit_token_mask", (batch, sequence))
    validation = _mask(validation_token_mask, "validation_token_mask", (batch, sequence))
    if bool((fit & validation).any().item()) or not bool(fit.any().item()) or not bool(
        validation.any().item()
    ) or bool(((fit | validation) & ~valid).any().item()):
        raise HFCausalGQAAttentionFoldError(
            "Attention fit/validation masks must be disjoint nonempty valid subsets."
        )
    positions = position_ids.detach().to(device="cpu")
    if tuple(positions.shape) != (batch, sequence):
        raise HFCausalGQAAttentionFoldError("position_ids has the wrong shape.")
    source_query = _reshape_heads(
        source_query_sequences,
        source_topology.query_heads,
        source_topology.head_dim,
        "source_query_sequences",
    )
    source_key = _reshape_heads(
        source_key_sequences,
        source_topology.key_value_heads,
        source_topology.head_dim,
        "source_key_sequences",
    )
    source_value = _reshape_heads(
        source_value_sequences,
        source_topology.key_value_heads,
        source_topology.head_dim,
        "source_value_sequences",
    )
    if source_topology.query_key_norm == "rms_norm":
        if source_query_norm_weight is None or source_key_norm_weight is None:
            raise HFCausalGQAAttentionFoldError(
                "Source RMS-normalized Q/K evidence requires both norm weights."
            )
        source_query = _head_rms_norm(
            source_query,
            source_query_norm_weight,
            source_topology.rms_norm_epsilon,
            "source_query_norm_weight",
        )
        source_key = _head_rms_norm(
            source_key,
            source_key_norm_weight,
            source_topology.rms_norm_epsilon,
            "source_key_norm_weight",
        )
    elif source_query_norm_weight is not None or source_key_norm_weight is not None:
        raise HFCausalGQAAttentionFoldError(
            "Identity source Q/K semantics forbid norm weights."
        )
    source_query_post = apply_rope(source_query, positions, theta=source_topology.rope_theta)
    source_key_post = apply_rope(source_key, positions, theta=source_topology.rope_theta)
    _source_scores, source_probabilities = causal_attention_probabilities(
        source_query_post,
        source_key_post,
        attention_mask=mask,
        scaling=source_topology.attention_scaling,
    )
    query_groups, kv_groups = _build_assignment(source_topology, target_topology)
    query_groups_per_kv = target_topology.queries_per_key_value
    selected_by_kv = []
    desired_query_post: list[Any] = []
    desired_key_post: list[Any] = []
    desired_value: list[Any] = []
    desired_probabilities: list[Any] = []
    for target_kv, source_kv_group in enumerate(kv_groups):
        first_query = target_kv * query_groups_per_kv
        local_query_groups = query_groups[
            first_query : first_query + query_groups_per_kv
        ]
        local_queries = tuple(
            source_query_post[:, group, :, :].mean(dim=1) for group in local_query_groups
        )
        local_key = source_key_post[:, source_kv_group, :, :].mean(dim=1)
        local_value = source_value[:, source_kv_group, :, :].mean(dim=1)
        selected = _select_rope_pairs(
            local_queries, local_key, local_value, target_topology.head_dim
        )
        selected_by_kv.append(selected)
        desired_query_post.extend(_contract_pairs(value, selected) for value in local_queries)
        desired_key_post.append(_contract_pairs(local_key, selected))
        desired_value.append(_contract_pairs(local_value, selected))
        desired_probabilities.extend(
            source_probabilities[:, group, :, :].mean(dim=1)
            for group in local_query_groups
        )
    desired_query_post_tensor = torch.stack(desired_query_post, dim=1)
    desired_key_post_tensor = torch.stack(desired_key_post, dim=1)
    desired_value_tensor = torch.stack(desired_value, dim=1)
    desired_probability_tensor = torch.stack(desired_probabilities, dim=1)
    desired_query_raw = invert_rope(
        desired_query_post_tensor, positions, theta=target_topology.rope_theta
    )
    desired_key_raw = invert_rope(
        desired_key_post_tensor, positions, theta=target_topology.rope_theta
    )
    child_fit = _masked_rows(child, fit)

    def fit_projection(desired: Any) -> Any:
        rows = desired.permute(0, 2, 1, 3).reshape(batch, sequence, -1)
        return _ridge(child_fit, _masked_rows(rows, fit), ridge).T.contiguous()

    query_weight = fit_projection(desired_query_raw)
    key_weight = fit_projection(desired_key_raw)
    value_weight = fit_projection(desired_value_tensor)
    actual_query = apply_rope(
        _reshape_heads(
            child @ query_weight.T,
            target_topology.query_heads,
            target_topology.head_dim,
            "fitted query",
        ),
        positions,
        theta=target_topology.rope_theta,
    )
    actual_key = apply_rope(
        _reshape_heads(
            child @ key_weight.T,
            target_topology.key_value_heads,
            target_topology.head_dim,
            "fitted key",
        ),
        positions,
        theta=target_topology.rope_theta,
    )
    _actual_scores, actual_probabilities = causal_attention_probabilities(
        actual_query,
        actual_key,
        attention_mask=mask,
        scaling=target_topology.attention_scaling,
    )
    actual_value = _reshape_heads(
        child @ value_weight.T,
        target_topology.key_value_heads,
        target_topology.head_dim,
        "fitted value",
    )
    contexts = torch.matmul(
        actual_probabilities,
        _repeat_kv(actual_value, target_topology.query_heads),
    )
    context_sequences = contexts.permute(0, 2, 1, 3).reshape(
        batch, sequence, target_topology.query_width
    )
    output_weight = _ridge(
        _masked_rows(context_sequences, fit),
        _masked_rows(delta_target, fit),
        ridge,
    ).T.contiguous()
    predicted_delta = context_sequences @ output_weight.T
    fit_prediction = _masked_rows(predicted_delta, fit)
    fit_target = _masked_rows(delta_target, fit)
    validation_prediction = _masked_rows(predicted_delta, validation)
    validation_target = _masked_rows(delta_target, validation)
    probability_fit = fit.reshape(batch, 1, sequence, 1).expand_as(actual_probabilities)
    probability_validation = validation.reshape(batch, 1, sequence, 1).expand_as(
        actual_probabilities
    )
    assignment = HFCausalGQABarycentricAssignment(
        source_query_groups=query_groups,
        source_key_value_groups=kv_groups,
        selected_rope_pairs_by_target_kv=tuple(selected_by_kv),
    )
    body = {
        "schema": SCHEMA,
        "component": "coupled_gqa_rope_qk_softmax_vo_fold",
        "method": "barycentric_complete_head_rope_pair_preimage_then_fixed_qk_vo",
        "source_topology": source_topology.as_dict(),
        "target_topology": target_topology.as_dict(),
        "assignment": assignment.as_dict(),
        "source_query_head_coverage": source_topology.query_heads,
        "source_key_value_head_coverage": source_topology.key_value_heads,
        "all_source_heads_covered_exactly_once": True,
        "qk_installed_before_vo_fit": True,
        "target_softmax_probabilities_replayed": True,
        "complete_target_residual_endpoint_consumed": True,
        "fit_residual_relative_rmse": _relative_rmse(fit_prediction, fit_target),
        "fit_residual_retained_energy_fraction": _retained_energy(
            fit_prediction, fit_target
        ),
        "validation_residual_relative_rmse": _relative_rmse(
            validation_prediction, validation_target
        ),
        "validation_residual_retained_energy_fraction": _retained_energy(
            validation_prediction, validation_target
        ),
        "fit_probability_relative_rmse": _relative_rmse(
            actual_probabilities[probability_fit],
            desired_probability_tensor[probability_fit],
        ),
        "validation_probability_relative_rmse": _relative_rmse(
            actual_probabilities[probability_validation],
            desired_probability_tensor[probability_validation],
        ),
        "ridge": ridge,
        "numeric_weight_matching_objective": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "query_weight_sha256": _tensor_sha256(query_weight),
        "key_weight_sha256": _tensor_sha256(key_weight),
        "value_weight_sha256": _tensor_sha256(value_weight),
        "output_weight_sha256": _tensor_sha256(output_weight),
    }
    return HFCausalCoupledAttentionFold(
        query_weight=query_weight,
        key_weight=key_weight,
        value_weight=value_weight,
        output_weight=output_weight,
        source_topology=source_topology,
        target_topology=target_topology,
        assignment=assignment,
        report=_report(body),
    )


__all__ = [
    "SCHEMA",
    "HFCausalCoupledAttentionFold",
    "HFCausalGQAAttentionFoldError",
    "HFCausalGQABarycentricAssignment",
    "HFCausalGQATopology",
    "apply_rope",
    "causal_attention_probabilities",
    "compile_hf_causal_gqa_attention_fold",
    "invert_rope",
]
