"""No-training endpoint-aware Q/K residual folding for fixed dense V/O.

This module is intentionally independent from the historical score-residual
compiler.  It consumes only sealed cached evidence and a source-derived dense
target Q/K/V/O seed.  Q/K corrections are compiled against the actual mapped
parent attention endpoint while the fixed V/O path remains inside the
analytical Jacobian:

    Q/K -> masked softmax -> V context -> O -> residual endpoint.

Every update is a deterministic ridge normal-equation solve followed by an
exact-BF16 alpha frontier.  FIT-CHECK is authenticated before the solve but is
not numerically evaluated until the final FIT state has been frozen.  No model
execution, autograd, optimizer, pruning, quantization, or target-template value
read is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Barrier
from types import MappingProxyType
from typing import Any, Literal


LFM2_ENDPOINT_QK_CASE_SCHEMA = "neural-forge-lfm2-endpoint-qk-case-v1"
LFM2_ENDPOINT_QK_BANK_SCHEMA = "neural-forge-lfm2-endpoint-qk-bank-v1"
LFM2_ENDPOINT_SOURCE_INVENTORY_SCHEMA = (
    "neural-forge-lfm2-endpoint-source-qkvo-inventory-v1"
)
LFM2_ENDPOINT_QK_SEED_SCHEMA = "neural-forge-lfm2-endpoint-qkvo-seed-v1"
LFM2_ENDPOINT_QK_POLICY_SCHEMA = "neural-forge-lfm2-endpoint-qk-policy-v1"
LFM2_ENDPOINT_QK_RESULT_SCHEMA = "neural-forge-lfm2-endpoint-qk-result-v1"
LFM2_ENDPOINT_QK_FIT_PROPOSAL_SCHEMA = "neural-forge-lfm2-endpoint-qk-fit-proposal-v1"
LFM2_ENDPOINT_QK_EXECUTION_SCHEMA = (
    "neural-forge-lfm2-endpoint-qk-observed-execution-v4"
)
_FIXED_DEVICE_CACHE_SCHEMA = "neural-forge-lfm2-endpoint-fixed-device-cache-v1"
_STATIC_DEVICE_SHARD_SCHEMA = "neural-forge-lfm2-endpoint-static-device-shards-v1"

_ZERO_OPERATION_AUDIT = {
    "model_forward_calls": 0,
    "model_generate_calls": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "trace_recaptures": 0,
    "blank_target_value_reads": 0,
}


class Lfm2EndpointQkResidualFoldError(ValueError):
    """Endpoint-aware Q/K folding failed a numerical or evidence invariant."""


class Lfm2EndpointQkResidualAuthenticationError(Lfm2EndpointQkResidualFoldError):
    """A seed, evidence bank, or result changed after it was sealed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "PyTorch is required for endpoint-aware Q/K folding."
        ) from exc
    return torch


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2EndpointQkResidualAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2EndpointQkResidualFoldError(
            "Tensor hashing requires a dense non-meta tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_sha256(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_tensor_sha256(values[name])))
    return digest.hexdigest()


def _structural_compatible_tensor_sha256(value: Any) -> str:
    """Digest compatible with the existing structural attention artifacts."""

    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _structural_compatible_state_sha256(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_structural_compatible_tensor_sha256(values[name])))
    return digest.hexdigest()


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2EndpointQkResidualFoldError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2EndpointQkResidualFoldError(
            f"{name} must be finite and positive."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2EndpointQkResidualFoldError(f"{name} must be finite and positive.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2EndpointQkResidualFoldError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2EndpointQkResidualFoldError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2EndpointQkResidualFoldError(f"{name} must be finite and nonnegative.")
    return result


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2EndpointQkResidualFoldError(f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise Lfm2EndpointQkResidualFoldError(
            f"{name} exceeds its hard ceiling {maximum}."
        )
    return value


def _dense_cpu_tensor(
    value: Any,
    name: str,
    *,
    ndim: int,
    dtype: Any | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or not (value.dtype.is_floating_point or value.dtype == torch.bool)
    ):
        raise Lfm2EndpointQkResidualFoldError(
            f"{name} must be one dense non-meta rank-{ndim} tensor."
        )
    if dtype is not None and value.dtype != dtype:
        raise Lfm2EndpointQkResidualFoldError(f"{name} must have exact dtype {dtype}.")
    result = value.detach().to(device="cpu").contiguous().clone()
    if result.dtype.is_floating_point and not bool(torch.isfinite(result).all().item()):
        raise Lfm2EndpointQkResidualFoldError(f"{name} contains a non-finite value.")
    return result


def _template_spec(
    value: Any, name: str, expected_shape: tuple[int, ...]
) -> dict[str, Any]:
    """Read shape/dtype only; target-template numerical values remain unread."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or tuple(value.shape) != expected_shape
        or not value.dtype.is_floating_point
    ):
        raise Lfm2EndpointQkResidualFoldError(
            f"{name} must be a dense floating target template with shape "
            f"{expected_shape!r}."
        )
    return {
        "shape": list(expected_shape),
        "dtype": str(value.dtype),
        "layout": str(value.layout),
    }


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _apply_rope(value: Any, cosine: Any, sine: Any) -> Any:
    return value * cosine.unsqueeze(1) + _rotate_half(value) * sine.unsqueeze(1)


def _apply_rope_transpose(value: Any, cosine: Any, sine: Any) -> Any:
    return value * cosine.unsqueeze(1) - _rotate_half(value) * sine.unsqueeze(1)


def _exact_runtime_rmsnorm(raw_rows: Any, norm_weight: Any, epsilon: float) -> Any:
    torch = _torch()
    dtype = raw_rows.dtype
    direction = raw_rows.to(torch.float32)
    variance = direction.square().mean(dim=-1, keepdim=True)
    normalized = direction * torch.rsqrt(variance + epsilon)
    gain = norm_weight
    if gain.device != raw_rows.device or gain.dtype != dtype:
        gain = gain.to(device=raw_rows.device, dtype=dtype)
    return (normalized.to(dtype=dtype) * gain).contiguous()


def _rmsnorm_jvp(
    raw_rows: Any,
    tangent_rows: Any,
    norm_weight: Any,
    epsilon: float,
) -> Any:
    torch = _torch()
    raw = raw_rows.to(torch.float32)
    tangent = tangent_rows.to(torch.float32)
    gain = norm_weight
    if gain.device != raw.device or gain.dtype != torch.float32:
        gain = gain.to(device=raw.device, dtype=torch.float32)
    width = int(raw.shape[-1])
    radius_squared = raw.square().mean(dim=-1, keepdim=True) + epsilon
    radius = torch.sqrt(radius_squared)
    radial = (raw * tangent).sum(dim=-1, keepdim=True) / (width * radius_squared)
    return (gain / radius) * (tangent - raw * radial)


def _rmsnorm_vjp(
    raw_rows: Any,
    cotangent_rows: Any,
    norm_weight: Any,
    epsilon: float,
) -> Any:
    raw = raw_rows.to(_torch().float32)
    cotangent = cotangent_rows.to(_torch().float32)
    gain = norm_weight
    if gain.device != raw.device or gain.dtype != _torch().float32:
        gain = gain.to(device=raw.device, dtype=_torch().float32)
    width = int(raw.shape[-1])
    radius_squared = raw.square().mean(dim=-1, keepdim=True) + epsilon
    radius = _torch().sqrt(radius_squared)
    gained = cotangent * gain
    radial = (raw * gained).sum(dim=-1, keepdim=True) / (width * radius_squared)
    return (gained - raw * radial) / radius


def _center_scores(value: Any, mask: Any) -> Any:
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (value * mask).sum(dim=-1, keepdim=True) / count
    return _torch().where(mask, value - mean, _torch().zeros_like(value))


def _masked_softmax(scores: Any, mask: Any) -> Any:
    torch = _torch()
    masked = scores.to(torch.float64).masked_fill(~mask, -torch.inf)
    result = torch.softmax(masked, dim=-1, dtype=torch.float64)
    return torch.where(mask, result, torch.zeros_like(result))


def _softmax_jvp(probabilities: Any, score_tangent: Any, mask: Any) -> Any:
    centered = score_tangent.to(_torch().float64) - (
        probabilities * score_tangent.to(_torch().float64)
    ).sum(dim=-1, keepdim=True)
    return _torch().where(mask, probabilities * centered, _torch().zeros_like(centered))


def _softmax_vjp(probabilities: Any, probability_cotangent: Any, mask: Any) -> Any:
    # The rowwise softmax Jacobian is symmetric.
    return _softmax_jvp(probabilities, probability_cotangent, mask)


def _validated_rope(
    cosine: Any,
    sine: Any,
    *,
    name: str,
    expected_shape: tuple[int, int, int],
) -> tuple[Any, Any]:
    torch = _torch()
    cos = _dense_cpu_tensor(
        cosine,
        f"{name}_cosine",
        ndim=3,
        dtype=torch.bfloat16,
    )
    sin = _dense_cpu_tensor(
        sine,
        f"{name}_sine",
        ndim=3,
        dtype=torch.bfloat16,
    )
    if tuple(cos.shape) != expected_shape or tuple(sin.shape) != expected_shape:
        raise Lfm2EndpointQkResidualFoldError(
            f"{name} RoPE terms have incompatible shapes."
        )
    half = expected_shape[-1] // 2
    if (
        half <= 0
        or expected_shape[-1] % 2
        or not torch.equal(cos[..., :half], cos[..., half:])
        or not torch.equal(sin[..., :half], sin[..., half:])
    ):
        raise Lfm2EndpointQkResidualFoldError(
            f"{name} RoPE must preserve duplicated-half frequency pairs."
        )
    radius = (
        cos[..., :half].to(torch.float32).square()
        + sin[..., :half].to(torch.float32).square()
    )
    if bool((radius <= torch.finfo(torch.float32).tiny).any().item()):
        raise Lfm2EndpointQkResidualFoldError(
            f"{name} RoPE contains a zero finite-precision radius."
        )
    return cos, sin


def _derive_probability_barycenter(
    parent_probabilities: Any,
    parent_mask: Any,
    query_groups: tuple[tuple[int, ...], ...],
    query_group_weights: tuple[tuple[float, ...], ...],
) -> tuple[Any, Any]:
    torch = _torch()
    batch, source_heads, query_length, key_length = map(int, parent_probabilities.shape)
    mask_heads = int(parent_mask.shape[1])
    expanded_parent_mask = (
        parent_mask.expand(-1, source_heads, -1, -1) if mask_heads == 1 else parent_mask
    )
    target_probabilities = []
    target_masks = []
    for target_head, (members, weights) in enumerate(
        zip(query_groups, query_group_weights, strict=True)
    ):
        if not members or len(members) != len(weights):
            raise Lfm2EndpointQkResidualFoldError(
                f"query group {target_head} is empty or misweighted."
            )
        index = torch.tensor(members, dtype=torch.long)
        selected_mask = expanded_parent_mask.index_select(1, index)
        reference_mask = selected_mask[:, :1]
        if not torch.equal(selected_mask, reference_mask.expand_as(selected_mask)):
            raise Lfm2EndpointQkResidualFoldError(
                "Parent heads merged into one target head use different masks."
            )
        alpha = torch.tensor(weights, dtype=torch.float64)
        if bool((alpha < 0.0).any().item()) or not math.isclose(
            float(alpha.sum().item()),
            1.0,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "Every query-group weight vector must be nonnegative and sum to one."
            )
        selected = parent_probabilities.index_select(1, index)
        mixture = (selected * alpha.reshape(1, -1, 1, 1)).sum(dim=1)
        mask = reference_mask[:, 0]
        mixture = torch.where(mask, mixture, torch.zeros_like(mixture))
        mixture = mixture / mixture.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        target_probabilities.append(mixture)
        target_masks.append(mask)
    probabilities = torch.stack(target_probabilities, dim=1).contiguous()
    masks = torch.stack(target_masks, dim=1).contiguous()
    if tuple(probabilities.shape) != (
        batch,
        len(query_groups),
        query_length,
        key_length,
    ):
        raise AssertionError("Internal probability barycenter shape is invalid.")
    return probabilities, masks


@dataclass(frozen=True, slots=True)
class Lfm2EndpointQkResidualCase:
    """One cached route and mapped parent attention endpoint."""

    target_query_inputs: Any
    target_key_value_inputs: Any
    parent_probabilities: Any
    parent_causal_mask: Any
    parent_endpoint_target: Any
    query_rope_cos: Any
    query_rope_sin: Any
    key_rope_cos: Any
    key_rope_sin: Any
    query_row_sqrt_weights: Any
    query_groups: tuple[tuple[int, ...], ...]
    query_group_weights: tuple[tuple[float, ...], ...]
    trajectory_sha256: str
    parent_probability_provenance: str = "explicit_parent_probability_tensor"
    parent_context_replay_relative_rmse: float = 0.0
    parent_context_replay_maximum_absolute_error: float = 0.0
    parent_context_replay_rtol: float = 0.0
    parent_context_replay_atol: float = 0.0
    target_probability_barycenter: Any = field(init=False, repr=False)
    target_causal_mask: Any = field(init=False, repr=False)
    target_receipt: Mapping[str, Any] = field(init=False, repr=False)
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        query_inputs = _dense_cpu_tensor(
            self.target_query_inputs,
            "target_query_inputs",
            ndim=3,
            dtype=torch.bfloat16,
        )
        key_inputs = _dense_cpu_tensor(
            self.target_key_value_inputs,
            "target_key_value_inputs",
            ndim=3,
            dtype=torch.bfloat16,
        )
        parent_probabilities = _dense_cpu_tensor(
            self.parent_probabilities,
            "parent_probabilities",
            ndim=4,
        ).to(torch.float64)
        parent_mask = _dense_cpu_tensor(
            self.parent_causal_mask,
            "parent_causal_mask",
            ndim=4,
            dtype=torch.bool,
        )
        endpoint = _dense_cpu_tensor(
            self.parent_endpoint_target,
            "parent_endpoint_target",
            ndim=3,
        )
        batch, query_length, hidden = map(int, query_inputs.shape)
        key_batch, key_length, key_hidden = map(int, key_inputs.shape)
        probability_shape = tuple(parent_probabilities.shape)
        if (
            key_batch != batch
            or key_hidden != hidden
            or probability_shape[0] != batch
            or probability_shape[2:] != (query_length, key_length)
            or tuple(endpoint.shape[:2]) != (batch, query_length)
            or tuple(parent_mask.shape[0:1]) != (batch,)
            or tuple(parent_mask.shape[2:]) != (query_length, key_length)
            or int(parent_mask.shape[1]) not in {1, probability_shape[1]}
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "Cached endpoint case tensors do not share one causal geometry."
            )
        if hidden <= 0 or int(endpoint.shape[-1]) <= 0:
            raise Lfm2EndpointQkResidualFoldError(
                "Cached endpoint case widths must be positive."
            )
        expanded_mask = (
            parent_mask.expand(-1, probability_shape[1], -1, -1)
            if int(parent_mask.shape[1]) == 1
            else parent_mask
        )
        if bool((parent_probabilities < 0.0).any().item()) or bool(
            (parent_probabilities.masked_select(~expanded_mask).abs() > 0.0)
            .any()
            .item()
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "Parent probabilities must be nonnegative and zero outside the mask."
            )
        sums = parent_probabilities.sum(dim=-1)
        if not bool(
            torch.allclose(
                sums,
                torch.ones_like(sums),
                rtol=1e-6,
                atol=1e-7,
            )
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "Every parent probability row must sum to one within the "
                "authenticated finite-precision envelope."
            )
        groups = tuple(
            tuple(int(member) for member in group) for group in self.query_groups
        )
        weights = tuple(
            tuple(float(weight) for weight in group)
            for group in self.query_group_weights
        )
        flattened = sorted(member for group in groups for member in group)
        if (
            len(groups) == 0
            or len(groups) != len(weights)
            or flattened != list(range(probability_shape[1]))
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "Query groups must partition every parent query head exactly once."
            )
        target_probabilities, target_mask = _derive_probability_barycenter(
            parent_probabilities,
            parent_mask,
            groups,
            weights,
        )
        head_dim = int(self.query_rope_cos.shape[-1])
        query_cos, query_sin = _validated_rope(
            self.query_rope_cos,
            self.query_rope_sin,
            name="query",
            expected_shape=(batch, query_length, head_dim),
        )
        key_cos, key_sin = _validated_rope(
            self.key_rope_cos,
            self.key_rope_sin,
            name="key",
            expected_shape=(batch, key_length, head_dim),
        )
        row_weights = _dense_cpu_tensor(
            self.query_row_sqrt_weights,
            "query_row_sqrt_weights",
            ndim=2,
        ).to(torch.float64)
        if tuple(row_weights.shape) != (batch, query_length) or bool(
            (row_weights <= 0.0).any().item()
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "Query row sqrt weights must be positive and query-aligned."
            )
        offset = key_length - query_length
        query_position = torch.arange(query_length).reshape(1, 1, -1, 1)
        key_position = torch.arange(key_length).reshape(1, 1, 1, -1)
        if (
            offset < 0
            or bool(
                (target_mask & (key_position > offset + query_position)).any().item()
            )
            or bool((~target_mask.any(dim=-1)).any().item())
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "Target probability barycenter is not one valid causal route."
            )
        trajectory = _required_sha256(
            self.trajectory_sha256,
            "trajectory_sha256",
        )
        provenance = str(self.parent_probability_provenance)
        if provenance not in {
            "authenticated_cached_parent_attention_probabilities",
            "legacy_recomputed_parent_qk",
            "explicit_parent_probability_tensor",
        }:
            raise Lfm2EndpointQkResidualFoldError(
                "parent_probability_provenance is not an authenticated mode."
            )
        context_replay_relative_rmse = _finite_nonnegative(
            self.parent_context_replay_relative_rmse,
            "parent_context_replay_relative_rmse",
        )
        context_replay_maximum_absolute_error = _finite_nonnegative(
            self.parent_context_replay_maximum_absolute_error,
            "parent_context_replay_maximum_absolute_error",
        )
        context_replay_rtol = _finite_nonnegative(
            self.parent_context_replay_rtol,
            "parent_context_replay_rtol",
        )
        context_replay_atol = _finite_nonnegative(
            self.parent_context_replay_atol,
            "parent_context_replay_atol",
        )
        target_body = {
            "schema": LFM2_ENDPOINT_QK_CASE_SCHEMA,
            "transform": "direct_parent_probability_barycenter_and_endpoint",
            "trajectory_sha256": trajectory,
            "parent_probability_sha256": _tensor_sha256(parent_probabilities),
            "parent_causal_mask_sha256": _tensor_sha256(parent_mask),
            "query_groups": [list(group) for group in groups],
            "query_group_weights": [list(group) for group in weights],
            "target_probability_barycenter_sha256": _tensor_sha256(
                target_probabilities
            ),
            "target_causal_mask_sha256": _tensor_sha256(target_mask),
            "parent_endpoint_target_sha256": _tensor_sha256(endpoint),
            "parent_probability_provenance": provenance,
            "parent_context_replay_relative_rmse": (context_replay_relative_rmse),
            "parent_context_replay_maximum_absolute_error": (
                context_replay_maximum_absolute_error
            ),
            "parent_context_replay_rtol": context_replay_rtol,
            "parent_context_replay_atol": context_replay_atol,
            "probability_target_is_mean_of_parent_probabilities": True,
            "probability_target_is_softmax_of_mean_parent_logits": False,
        }
        target_receipt = {
            **target_body,
            "target_receipt_sha256": _canonical_sha256(target_body),
        }
        evidence_body = {
            "target_receipt_sha256": target_receipt["target_receipt_sha256"],
            "target_query_inputs_sha256": _tensor_sha256(query_inputs),
            "target_key_value_inputs_sha256": _tensor_sha256(key_inputs),
            "query_rope_cos_sha256": _tensor_sha256(query_cos),
            "query_rope_sin_sha256": _tensor_sha256(query_sin),
            "key_rope_cos_sha256": _tensor_sha256(key_cos),
            "key_rope_sin_sha256": _tensor_sha256(key_sin),
            "query_row_sqrt_weights_sha256": _tensor_sha256(row_weights),
        }
        object.__setattr__(self, "target_query_inputs", query_inputs)
        object.__setattr__(self, "target_key_value_inputs", key_inputs)
        object.__setattr__(self, "parent_probabilities", parent_probabilities)
        object.__setattr__(self, "parent_causal_mask", parent_mask)
        object.__setattr__(self, "parent_endpoint_target", endpoint)
        object.__setattr__(self, "query_rope_cos", query_cos)
        object.__setattr__(self, "query_rope_sin", query_sin)
        object.__setattr__(self, "key_rope_cos", key_cos)
        object.__setattr__(self, "key_rope_sin", key_sin)
        object.__setattr__(self, "query_row_sqrt_weights", row_weights)
        object.__setattr__(self, "query_groups", groups)
        object.__setattr__(self, "query_group_weights", weights)
        object.__setattr__(self, "trajectory_sha256", trajectory)
        object.__setattr__(self, "parent_probability_provenance", provenance)
        object.__setattr__(
            self,
            "parent_context_replay_relative_rmse",
            context_replay_relative_rmse,
        )
        object.__setattr__(
            self,
            "parent_context_replay_maximum_absolute_error",
            context_replay_maximum_absolute_error,
        )
        object.__setattr__(
            self,
            "parent_context_replay_rtol",
            context_replay_rtol,
        )
        object.__setattr__(
            self,
            "parent_context_replay_atol",
            context_replay_atol,
        )
        object.__setattr__(
            self,
            "target_probability_barycenter",
            target_probabilities,
        )
        object.__setattr__(self, "target_causal_mask", target_mask)
        object.__setattr__(
            self,
            "target_receipt",
            MappingProxyType(target_receipt),
        )
        object.__setattr__(self, "evidence_sha256", _canonical_sha256(evidence_body))

    @property
    def target_receipt_sha256(self) -> str:
        return str(self.target_receipt["target_receipt_sha256"])

    def verify(self) -> str:
        rebuilt = Lfm2EndpointQkResidualCase(
            target_query_inputs=self.target_query_inputs,
            target_key_value_inputs=self.target_key_value_inputs,
            parent_probabilities=self.parent_probabilities,
            parent_causal_mask=self.parent_causal_mask,
            parent_endpoint_target=self.parent_endpoint_target,
            query_rope_cos=self.query_rope_cos,
            query_rope_sin=self.query_rope_sin,
            key_rope_cos=self.key_rope_cos,
            key_rope_sin=self.key_rope_sin,
            query_row_sqrt_weights=self.query_row_sqrt_weights,
            query_groups=self.query_groups,
            query_group_weights=self.query_group_weights,
            trajectory_sha256=self.trajectory_sha256,
            parent_probability_provenance=self.parent_probability_provenance,
            parent_context_replay_relative_rmse=(
                self.parent_context_replay_relative_rmse
            ),
            parent_context_replay_maximum_absolute_error=(
                self.parent_context_replay_maximum_absolute_error
            ),
            parent_context_replay_rtol=self.parent_context_replay_rtol,
            parent_context_replay_atol=self.parent_context_replay_atol,
        )
        if (
            rebuilt.evidence_sha256 != self.evidence_sha256
            or rebuilt.target_receipt_sha256 != self.target_receipt_sha256
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Endpoint Q/K case changed after sealing."
            )
        return self.evidence_sha256


EvidencePartition = Literal["fit", "fit_check"]


@dataclass(frozen=True, slots=True)
class Lfm2EndpointQkResidualEvidenceBank:
    partition: EvidencePartition
    cases: tuple[Lfm2EndpointQkResidualCase, ...]
    source_split_sha256: str
    source_artifact_sha256: str
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if self.partition not in {"fit", "fit_check"}:
            raise Lfm2EndpointQkResidualFoldError(
                "Evidence partition must be 'fit' or 'fit_check'."
            )
        cases = tuple(self.cases)
        if not cases or not all(
            isinstance(case, Lfm2EndpointQkResidualCase) for case in cases
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "Endpoint evidence banks require at least one typed case."
            )
        case_sha256s = tuple(case.verify() for case in cases)
        trajectories = tuple(case.trajectory_sha256 for case in cases)
        if len(set(case_sha256s)) != len(cases) or len(set(trajectories)) != len(cases):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "An endpoint evidence bank repeats numerical content or trajectories."
            )
        source_split = _required_sha256(
            self.source_split_sha256,
            "source_split_sha256",
        )
        source_artifact = _required_sha256(
            self.source_artifact_sha256,
            "source_artifact_sha256",
        )
        body = {
            "schema": LFM2_ENDPOINT_QK_BANK_SCHEMA,
            "partition": self.partition,
            "source_split_sha256": source_split,
            "source_artifact_sha256": source_artifact,
            "case_sha256s": list(case_sha256s),
            "trajectory_sha256s": list(trajectories),
            "target_receipt_sha256s": [case.target_receipt_sha256 for case in cases],
        }
        computed = _canonical_sha256(body)
        if self.evidence_sha256 and self.evidence_sha256 != computed:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Endpoint evidence bank does not match its supplied seal."
            )
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "source_split_sha256", source_split)
        object.__setattr__(self, "source_artifact_sha256", source_artifact)
        object.__setattr__(self, "evidence_sha256", computed)

    @property
    def trajectory_sha256s(self) -> tuple[str, ...]:
        return tuple(case.trajectory_sha256 for case in self.cases)

    @property
    def target_receipt_sha256s(self) -> tuple[str, ...]:
        return tuple(case.target_receipt_sha256 for case in self.cases)

    def verify(self) -> str:
        rebuilt = Lfm2EndpointQkResidualEvidenceBank(
            partition=self.partition,
            cases=self.cases,
            source_split_sha256=self.source_split_sha256,
            source_artifact_sha256=self.source_artifact_sha256,
        )
        if rebuilt.evidence_sha256 != self.evidence_sha256:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Endpoint evidence bank changed after sealing."
            )
        return self.evidence_sha256


def seal_lfm2_endpoint_qk_fit_bank(
    cases: Sequence[Lfm2EndpointQkResidualCase],
    *,
    source_split_sha256: str,
    source_artifact_sha256: str,
) -> Lfm2EndpointQkResidualEvidenceBank:
    return Lfm2EndpointQkResidualEvidenceBank(
        partition="fit",
        cases=tuple(cases),
        source_split_sha256=source_split_sha256,
        source_artifact_sha256=source_artifact_sha256,
    )


def seal_lfm2_endpoint_qk_fit_check_bank(
    cases: Sequence[Lfm2EndpointQkResidualCase],
    *,
    source_split_sha256: str,
    source_artifact_sha256: str,
) -> Lfm2EndpointQkResidualEvidenceBank:
    return Lfm2EndpointQkResidualEvidenceBank(
        partition="fit_check",
        cases=tuple(cases),
        source_split_sha256=source_split_sha256,
        source_artifact_sha256=source_artifact_sha256,
    )


@dataclass(frozen=True, slots=True)
class Lfm2DenseQkvoSourceInventory:
    """Sealed tensor-derived source inventory behind a compaction claim."""

    tensor_receipts: Mapping[str, Mapping[str, Any]]
    source_state_sha256: str
    structural_source_state_sha256: str
    logical_parameters: int
    inventory_sha256: str = ""

    def __post_init__(self) -> None:
        names = (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
        )
        receipts = {
            str(name): dict(receipt)
            for name, receipt in dict(self.tensor_receipts).items()
        }
        if tuple(sorted(receipts)) != tuple(sorted(names)):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Source inventory must authenticate exactly dense Q/K/V/O."
            )
        count = 0
        normalized: dict[str, Mapping[str, Any]] = {}
        for name in names:
            receipt = receipts[name]
            shape = receipt.get("shape")
            dtype = receipt.get("dtype")
            tensor_sha = receipt.get("tensor_sha256")
            structural_tensor_sha = receipt.get("structural_tensor_sha256")
            parameters = receipt.get("logical_parameters")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in shape
                )
                or not isinstance(dtype, str)
                or not dtype.startswith("torch.")
                or _required_sha256(
                    tensor_sha,
                    f"tensor_receipts[{name!r}].tensor_sha256",
                )
                != tensor_sha
                or _required_sha256(
                    structural_tensor_sha,
                    (f"tensor_receipts[{name!r}].structural_tensor_sha256"),
                )
                != structural_tensor_sha
                or isinstance(parameters, bool)
                or not isinstance(parameters, int)
                or parameters != math.prod(shape)
            ):
                raise Lfm2EndpointQkResidualAuthenticationError(
                    f"Source tensor receipt {name!r} is invalid."
                )
            count += parameters
            normalized[name] = MappingProxyType(
                {
                    "shape": list(shape),
                    "dtype": dtype,
                    "logical_parameters": parameters,
                    "tensor_sha256": tensor_sha,
                    "structural_tensor_sha256": structural_tensor_sha,
                }
            )
        logical = _positive_int(self.logical_parameters, "logical_parameters")
        if logical != count:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Source inventory count is not derived from its tensor shapes."
            )
        state_sha = _required_sha256(
            self.source_state_sha256,
            "source_state_sha256",
        )
        state_digest = hashlib.sha256()
        for name in sorted(names):
            state_digest.update(name.encode("utf-8"))
            state_digest.update(bytes.fromhex(str(normalized[name]["tensor_sha256"])))
        if state_digest.hexdigest() != state_sha:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Source inventory state digest does not match its tensor receipts."
            )
        structural_state_sha = _required_sha256(
            self.structural_source_state_sha256,
            "structural_source_state_sha256",
        )
        structural_digest = hashlib.sha256()
        for name in sorted(names):
            structural_digest.update(name.encode("utf-8"))
            structural_digest.update(
                bytes.fromhex(str(normalized[name]["structural_tensor_sha256"]))
            )
        if structural_digest.hexdigest() != structural_state_sha:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Source inventory structural digest does not match its tensor receipts."
            )
        body = {
            "schema": LFM2_ENDPOINT_SOURCE_INVENTORY_SCHEMA,
            "tensor_receipts": {name: dict(normalized[name]) for name in names},
            "source_state_sha256": state_sha,
            "structural_source_state_sha256": structural_state_sha,
            "logical_parameters": logical,
            "count_derived_from_actual_dense_tensors": True,
        }
        computed = _canonical_sha256(body)
        if self.inventory_sha256 and self.inventory_sha256 != computed:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Source Q/K/V/O inventory does not match its supplied seal."
            )
        object.__setattr__(
            self,
            "tensor_receipts",
            MappingProxyType(normalized),
        )
        object.__setattr__(self, "source_state_sha256", state_sha)
        object.__setattr__(
            self,
            "structural_source_state_sha256",
            structural_state_sha,
        )
        object.__setattr__(self, "logical_parameters", logical)
        object.__setattr__(self, "inventory_sha256", computed)

    def verify(self) -> str:
        rebuilt = Lfm2DenseQkvoSourceInventory(
            tensor_receipts=self.tensor_receipts,
            source_state_sha256=self.source_state_sha256,
            structural_source_state_sha256=(self.structural_source_state_sha256),
            logical_parameters=self.logical_parameters,
        )
        if rebuilt.inventory_sha256 != self.inventory_sha256:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Source Q/K/V/O inventory changed after sealing."
            )
        return self.inventory_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_ENDPOINT_SOURCE_INVENTORY_SCHEMA,
            "tensor_receipts": {
                name: dict(receipt) for name, receipt in self.tensor_receipts.items()
            },
            "source_state_sha256": self.source_state_sha256,
            "structural_source_state_sha256": (self.structural_source_state_sha256),
            "logical_parameters": self.logical_parameters,
            "count_derived_from_actual_dense_tensors": True,
            "inventory_sha256": self.inventory_sha256,
        }


def seal_lfm2_dense_qkvo_source_inventory(
    *,
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_output_weight: Any,
) -> Lfm2DenseQkvoSourceInventory:
    """Hash and count actual dense source tensors; metadata alone is insufficient."""

    values = {
        "q_proj.weight": source_query_weight,
        "k_proj.weight": source_key_weight,
        "v_proj.weight": source_value_weight,
        "out_proj.weight": source_output_weight,
    }
    tensors = {
        name: _dense_cpu_tensor(value, f"source_{name}", ndim=2)
        for name, value in values.items()
    }
    receipts = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "logical_parameters": int(tensor.numel()),
            "tensor_sha256": _tensor_sha256(tensor),
            "structural_tensor_sha256": (_structural_compatible_tensor_sha256(tensor)),
        }
        for name, tensor in tensors.items()
    }
    return Lfm2DenseQkvoSourceInventory(
        tensor_receipts=receipts,
        source_state_sha256=_state_sha256(tensors),
        structural_source_state_sha256=(_structural_compatible_state_sha256(tensors)),
        logical_parameters=sum(int(tensor.numel()) for tensor in tensors.values()),
    )


@dataclass(frozen=True, slots=True)
class Lfm2EndpointQkvoSeed:
    """Source-derived ordinary dense target Q/K/V/O state."""

    query_weight: Any
    key_weight: Any
    value_weight: Any
    output_weight: Any
    query_norm_weight: Any
    key_norm_weight: Any
    norm_epsilon: float
    source_inventory: Lfm2DenseQkvoSourceInventory
    source_artifact_sha256: str
    target_template_spec_sha256: str
    seed_sha256: str = ""

    def __post_init__(self) -> None:
        torch = _torch()
        query = _dense_cpu_tensor(
            self.query_weight,
            "query_weight",
            ndim=2,
            dtype=torch.bfloat16,
        )
        key = _dense_cpu_tensor(
            self.key_weight,
            "key_weight",
            ndim=2,
            dtype=torch.bfloat16,
        )
        value = _dense_cpu_tensor(
            self.value_weight,
            "value_weight",
            ndim=2,
            dtype=torch.bfloat16,
        )
        output = _dense_cpu_tensor(
            self.output_weight,
            "output_weight",
            ndim=2,
            dtype=torch.bfloat16,
        )
        query_norm = _dense_cpu_tensor(
            self.query_norm_weight,
            "query_norm_weight",
            ndim=1,
            dtype=torch.bfloat16,
        )
        key_norm = _dense_cpu_tensor(
            self.key_norm_weight,
            "key_norm_weight",
            ndim=1,
            dtype=torch.bfloat16,
        )
        head_dim = int(query_norm.numel())
        hidden = int(query.shape[1])
        if (
            head_dim <= 0
            or head_dim % 2
            or tuple(key_norm.shape) != (head_dim,)
            or int(query.shape[0]) % head_dim
            or int(key.shape[0]) % head_dim
            or int(value.shape[0]) != int(key.shape[0])
            or int(key.shape[1]) != hidden
            or int(value.shape[1]) != hidden
            or int(output.shape[1]) != int(query.shape[0])
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "Dense Q/K/V/O seed tensors do not define a valid GQA topology."
            )
        query_heads = int(query.shape[0]) // head_dim
        key_value_heads = int(key.shape[0]) // head_dim
        if query_heads <= 0 or key_value_heads <= 0 or query_heads % key_value_heads:
            raise Lfm2EndpointQkResidualFoldError(
                "Dense Q/K/V/O seed does not define valid GQA head counts."
            )
        epsilon = _finite_positive(self.norm_epsilon, "norm_epsilon")
        if not isinstance(self.source_inventory, Lfm2DenseQkvoSourceInventory):
            raise TypeError("source_inventory must be Lfm2DenseQkvoSourceInventory.")
        source_inventory_sha = self.source_inventory.verify()
        source_logical = self.source_inventory.logical_parameters
        target_logical = sum(
            int(tensor.numel()) for tensor in (query, key, value, output)
        )
        if source_logical <= target_logical:
            raise Lfm2EndpointQkResidualFoldError(
                "Endpoint Q/K seed must already occupy a genuinely smaller dense "
                "target Q/K/V/O topology."
            )
        source_artifact = _required_sha256(
            self.source_artifact_sha256,
            "source_artifact_sha256",
        )
        template_spec = _required_sha256(
            self.target_template_spec_sha256,
            "target_template_spec_sha256",
        )
        body = {
            "schema": LFM2_ENDPOINT_QK_SEED_SCHEMA,
            "source_artifact_sha256": source_artifact,
            "source_inventory_sha256": source_inventory_sha,
            "source_state_sha256": self.source_inventory.source_state_sha256,
            "structural_source_state_sha256": (
                self.source_inventory.structural_source_state_sha256
            ),
            "target_template_spec_sha256": template_spec,
            "query_weight_sha256": _tensor_sha256(query),
            "key_weight_sha256": _tensor_sha256(key),
            "value_weight_sha256": _tensor_sha256(value),
            "output_weight_sha256": _tensor_sha256(output),
            "query_norm_weight_sha256": _tensor_sha256(query_norm),
            "key_norm_weight_sha256": _tensor_sha256(key_norm),
            "norm_epsilon": epsilon,
            "source_logical_parameters": source_logical,
            "target_logical_parameters": target_logical,
            "parameter_compaction_ratio": source_logical / target_logical,
            "ordinary_dense_target_qkvo": True,
            "blank_target_values_read": False,
        }
        computed = _canonical_sha256(body)
        if self.seed_sha256 and self.seed_sha256 != computed:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Endpoint Q/K/V/O seed does not match its supplied seal."
            )
        object.__setattr__(self, "query_weight", query)
        object.__setattr__(self, "key_weight", key)
        object.__setattr__(self, "value_weight", value)
        object.__setattr__(self, "output_weight", output)
        object.__setattr__(self, "query_norm_weight", query_norm)
        object.__setattr__(self, "key_norm_weight", key_norm)
        object.__setattr__(self, "norm_epsilon", epsilon)
        object.__setattr__(self, "source_artifact_sha256", source_artifact)
        object.__setattr__(self, "target_template_spec_sha256", template_spec)
        object.__setattr__(self, "seed_sha256", computed)

    @property
    def head_dim(self) -> int:
        return int(self.query_norm_weight.numel())

    @property
    def query_heads(self) -> int:
        return int(self.query_weight.shape[0]) // self.head_dim

    @property
    def key_value_heads(self) -> int:
        return int(self.key_weight.shape[0]) // self.head_dim

    @property
    def queries_per_key_value(self) -> int:
        return self.query_heads // self.key_value_heads

    @property
    def hidden_size(self) -> int:
        return int(self.query_weight.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.output_weight.shape[0])

    @property
    def target_logical_parameters(self) -> int:
        return sum(
            int(tensor.numel())
            for tensor in (
                self.query_weight,
                self.key_weight,
                self.value_weight,
                self.output_weight,
            )
        )

    @property
    def source_logical_parameters(self) -> int:
        return self.source_inventory.logical_parameters

    @property
    def parameter_compaction_ratio(self) -> float:
        return self.source_logical_parameters / self.target_logical_parameters

    def verify(self) -> str:
        rebuilt = Lfm2EndpointQkvoSeed(
            query_weight=self.query_weight,
            key_weight=self.key_weight,
            value_weight=self.value_weight,
            output_weight=self.output_weight,
            query_norm_weight=self.query_norm_weight,
            key_norm_weight=self.key_norm_weight,
            norm_epsilon=self.norm_epsilon,
            source_inventory=self.source_inventory,
            source_artifact_sha256=self.source_artifact_sha256,
            target_template_spec_sha256=self.target_template_spec_sha256,
        )
        if rebuilt.seed_sha256 != self.seed_sha256:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Endpoint Q/K/V/O seed changed after sealing."
            )
        return self.seed_sha256


def build_lfm2_endpoint_qkvo_seed(
    *,
    query_weight: Any,
    key_weight: Any,
    value_weight: Any,
    output_weight: Any,
    query_norm_weight: Any,
    key_norm_weight: Any,
    norm_epsilon: float,
    source_query_weight: Any | None = None,
    source_key_weight: Any | None = None,
    source_value_weight: Any | None = None,
    source_output_weight: Any | None = None,
    source_inventory: Lfm2DenseQkvoSourceInventory | None = None,
    source_artifact_sha256: str,
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    target_value_weight_template: Any,
    target_output_weight_template: Any,
) -> Lfm2EndpointQkvoSeed:
    """Seal a source-derived seed while reading blank templates as specs only."""

    source_values = (
        source_query_weight,
        source_key_weight,
        source_value_weight,
        source_output_weight,
    )
    if source_inventory is None:
        if any(value is None for value in source_values):
            raise Lfm2EndpointQkResidualFoldError(
                "Seed construction requires all actual source Q/K/V/O tensors "
                "or one previously sealed source inventory."
            )
        source_inventory = seal_lfm2_dense_qkvo_source_inventory(
            source_query_weight=source_query_weight,
            source_key_weight=source_key_weight,
            source_value_weight=source_value_weight,
            source_output_weight=source_output_weight,
        )
    elif (
        not isinstance(source_inventory, Lfm2DenseQkvoSourceInventory)
        or any(value is not None for value in source_values)
        or not source_inventory.verify()
    ):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "A sealed source inventory cannot be mixed with unverified tensors."
        )
    shapes = {
        "q_proj.weight": tuple(query_weight.shape),
        "k_proj.weight": tuple(key_weight.shape),
        "v_proj.weight": tuple(value_weight.shape),
        "out_proj.weight": tuple(output_weight.shape),
    }
    specs = {
        "q_proj.weight": _template_spec(
            target_query_weight_template,
            "target_query_weight_template",
            shapes["q_proj.weight"],
        ),
        "k_proj.weight": _template_spec(
            target_key_weight_template,
            "target_key_weight_template",
            shapes["k_proj.weight"],
        ),
        "v_proj.weight": _template_spec(
            target_value_weight_template,
            "target_value_weight_template",
            shapes["v_proj.weight"],
        ),
        "out_proj.weight": _template_spec(
            target_output_weight_template,
            "target_output_weight_template",
            shapes["out_proj.weight"],
        ),
    }
    source_dtypes = {
        "q_proj.weight": str(query_weight.dtype),
        "k_proj.weight": str(key_weight.dtype),
        "v_proj.weight": str(value_weight.dtype),
        "out_proj.weight": str(output_weight.dtype),
    }
    if any(specs[name]["dtype"] != source_dtypes[name] for name in specs):
        raise Lfm2EndpointQkResidualFoldError(
            "Blank target template dtypes differ from the source-derived seed."
        )
    return Lfm2EndpointQkvoSeed(
        query_weight=query_weight,
        key_weight=key_weight,
        value_weight=value_weight,
        output_weight=output_weight,
        query_norm_weight=query_norm_weight,
        key_norm_weight=key_norm_weight,
        norm_epsilon=norm_epsilon,
        source_inventory=source_inventory,
        source_artifact_sha256=source_artifact_sha256,
        target_template_spec_sha256=_canonical_sha256(specs),
    )


@dataclass(frozen=True, slots=True)
class Lfm2EndpointQkResidualFoldPolicy:
    ridge: float = 1e-3
    route_objective_weight: float = 0.10
    trust_radius: float = 0.75
    maximum_sweeps: int = 3
    maximum_cg_iterations: int = 64
    minimum_cg_iterations: int = 4
    cg_restart_iterations: int = 64
    cg_relative_tolerance: float = 1e-2
    alpha_frontier: tuple[float, ...] = (
        0.0,
        0.03125,
        0.0625,
        0.125,
        0.25,
        0.5,
        0.75,
        1.0,
    )
    metric_regression_tolerance: float = 1e-6
    minimum_fit_endpoint_improvement: float = 1e-8
    maximum_fit_cases: int = 256
    maximum_fit_check_cases: int = 64
    maximum_endpoint_elements: int = 268_435_456
    maximum_probability_elements: int = 268_435_456
    maximum_prepared_tensor_bytes: int = 68_719_476_736
    maximum_replay_tensor_bytes: int = 68_719_476_736
    maximum_bytes_per_compute_device: int = 21_474_836_480
    solver_device: str = "cpu"
    compute_devices: tuple[str, ...] = ()
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "ridge",
            "trust_radius",
            "cg_relative_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), name),
            )
        for name in (
            "route_objective_weight",
            "metric_regression_tolerance",
            "minimum_fit_endpoint_improvement",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        if self.route_objective_weight > 1.0 or self.trust_radius > 2.0:
            raise Lfm2EndpointQkResidualFoldError(
                "Route weight or trust radius exceeds its analytical ceiling."
            )
        sweeps = _positive_int(self.maximum_sweeps, "maximum_sweeps")
        if sweeps not in {2, 3, 4}:
            raise Lfm2EndpointQkResidualFoldError(
                "Endpoint-aware folding requires exactly 2–4 analytical sweeps."
            )
        object.__setattr__(self, "maximum_sweeps", sweeps)
        for name, maximum in (
            ("maximum_cg_iterations", 4096),
            ("minimum_cg_iterations", 4096),
            ("cg_restart_iterations", 4096),
            ("maximum_fit_cases", 1024),
            ("maximum_fit_check_cases", 1024),
            ("maximum_endpoint_elements", 1_073_741_824),
            ("maximum_probability_elements", 1_073_741_824),
            ("maximum_prepared_tensor_bytes", 1_099_511_627_776),
            ("maximum_replay_tensor_bytes", 1_099_511_627_776),
            ("maximum_bytes_per_compute_device", 274_877_906_944),
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name, maximum=maximum),
            )
        if (
            self.minimum_cg_iterations > self.maximum_cg_iterations
            or self.cg_restart_iterations > self.maximum_cg_iterations
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "CG minimum/restart iterations exceed the maximum."
            )
        alphas = tuple(
            _finite_nonnegative(value, "alpha_frontier")
            for value in self.alpha_frontier
        )
        if (
            not alphas
            or alphas[0] != 0.0
            or any(value > 1.0 for value in alphas)
            or tuple(sorted(set(alphas))) != alphas
            or len(alphas) > 32
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "alpha_frontier must be sorted, unique, start at zero, and "
                "remain inside [0,1]."
            )
        object.__setattr__(self, "alpha_frontier", alphas)
        if not isinstance(self.solver_device, str) or not self.solver_device:
            raise Lfm2EndpointQkResidualFoldError(
                "solver_device must be a nonempty torch device string."
            )
        try:
            device = _torch().device(self.solver_device)
        except (RuntimeError, ValueError) as exc:
            raise Lfm2EndpointQkResidualFoldError(
                "solver_device is not a valid torch device."
            ) from exc
        if device.type == "cuda" and device.index is None:
            raise Lfm2EndpointQkResidualFoldError(
                "CUDA solver_device must include an explicit device index."
            )
        canonical_solver = str(device)
        object.__setattr__(self, "solver_device", canonical_solver)
        raw_compute = self.compute_devices or (canonical_solver,)
        if (
            not isinstance(raw_compute, tuple)
            or not raw_compute
            or not all(isinstance(value, str) and value for value in raw_compute)
        ):
            raise Lfm2EndpointQkResidualFoldError(
                "compute_devices must be a nonempty tuple of device strings."
            )
        try:
            compute = tuple(_torch().device(value) for value in raw_compute)
        except (RuntimeError, ValueError) as exc:
            raise Lfm2EndpointQkResidualFoldError(
                "compute_devices contains an invalid torch device."
            ) from exc
        if any(value.type == "cuda" and value.index is None for value in compute):
            raise Lfm2EndpointQkResidualFoldError(
                "CUDA compute_devices must include explicit device indices."
            )
        canonical_compute = tuple(str(value) for value in compute)
        if len(set(canonical_compute)) != len(canonical_compute):
            raise Lfm2EndpointQkResidualFoldError(
                "compute_devices must not repeat a logical device."
            )
        object.__setattr__(self, "compute_devices", canonical_compute)
        expected = _canonical_sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Endpoint policy authentication is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_ENDPOINT_QK_POLICY_SCHEMA,
            **{
                name: (
                    list(value)
                    if name in {"alpha_frontier", "compute_devices"}
                    else value
                )
                for name, value in (
                    (field_name, getattr(self, field_name))
                    for field_name in self.__dataclass_fields__
                    if field_name != "authentication_sha256"
                )
            },
            "fixed_value_output": True,
            "fit_check_can_trigger_refit": False,
            "training_allowed": False,
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _canonical_sha256(
            self.authentication_body()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


def _resolve_execution_devices(
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> tuple[Any, tuple[Any, ...]]:
    torch = _torch()
    solver = torch.device(policy.solver_device)
    workers = tuple(torch.device(value) for value in policy.compute_devices)
    requested_cuda = tuple(
        device for device in (solver, *workers) if device.type == "cuda"
    )
    if requested_cuda and not torch.cuda.is_available():
        raise Lfm2EndpointQkResidualFoldError(
            "Requested CUDA execution devices are unavailable."
        )
    count = torch.cuda.device_count() if requested_cuda else 0
    if any(
        device.index is None or device.index < 0 or device.index >= count
        for device in requested_cuda
    ):
        raise Lfm2EndpointQkResidualFoldError(
            "A requested CUDA execution device index is unavailable."
        )
    return solver, workers


def _observed_device_identity(device: Any) -> dict[str, Any]:
    torch = _torch()
    observed = torch.device(device)
    cuda_uuid: str | None = None
    cuda_name: str | None = None
    if observed.type == "cuda":
        if observed.index is None:
            raise Lfm2EndpointQkResidualFoldError(
                "Observed CUDA execution must name an explicit device index."
            )
        try:
            properties = torch.cuda.get_device_properties(observed.index)
            cuda_uuid = str(getattr(properties, "uuid"))
            cuda_name = str(getattr(properties, "name"))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise Lfm2EndpointQkResidualFoldError(
                "Observed CUDA execution lacks a readable UUID or name."
            ) from exc
        if not cuda_uuid or cuda_uuid == "None" or not cuda_name:
            raise Lfm2EndpointQkResidualFoldError(
                "Observed CUDA execution lacks a nonempty UUID or name."
            )
    return {
        "observed_logical_device": str(observed),
        "observed_device_type": observed.type,
        "observed_device_index": observed.index,
        "cuda_uuid": cuda_uuid,
        "cuda_name": cuda_name,
    }


def _force_observed_positive_scalar(
    value: Any,
    *,
    intended_device: Any,
    operation_name: str,
) -> tuple[float, dict[str, Any]]:
    torch = _torch()
    intended = torch.device(intended_device)
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.numel() != 1
        or value.device.type == "meta"
        or str(value.device) != str(intended)
    ):
        raise Lfm2EndpointQkResidualFoldError(
            f"{operation_name} did not run on requested device {intended}."
        )
    if intended.type == "cuda":
        torch.cuda.synchronize(intended)
    scalar = float(value.detach().item())
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise Lfm2EndpointQkResidualFoldError(
            f"{operation_name} did not produce a finite positive witness."
        )
    return scalar, _observed_device_identity(intended)


def _tensor_logical_bytes(value: Any) -> int:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise Lfm2EndpointQkResidualFoldError(
            "Execution accounting encountered a non-dense tensor."
        )
    return int(value.numel()) * int(value.element_size())


def _prepared_persistent_tensor_bytes(case: _PreparedCase) -> int:
    return sum(
        _tensor_logical_bytes(value)
        for value in (
            case.query_inputs,
            case.key_inputs,
            case.target_probabilities,
            case.target_endpoint,
            case.mask,
            case.query_cosine,
            case.query_sine,
            case.key_cosine,
            case.key_sine,
            case.query_row_weights,
        )
    )


def _attached_replay_tensor_bytes(case: _PreparedCase) -> int:
    values = (
        case.query_linear,
        case.key_linear,
        case.query_post,
        case.key_post,
        case.repeated_value,
        case.probabilities,
        case.endpoint,
    )
    if any(value is None for value in values):
        raise Lfm2EndpointQkResidualFoldError(
            "Execution accounting requires one attached exact replay per FIT case."
        )
    return sum(_tensor_logical_bytes(value) for value in values)


def _expected_prepared_tensor_bytes(case: Lfm2EndpointQkResidualCase) -> int:
    return (
        _tensor_logical_bytes(case.target_query_inputs)
        + _tensor_logical_bytes(case.target_key_value_inputs)
        + int(case.target_probability_barycenter.numel()) * 8
        + int(case.parent_endpoint_target.numel()) * 8
        + _tensor_logical_bytes(case.target_causal_mask)
        + _tensor_logical_bytes(case.query_rope_cos)
        + _tensor_logical_bytes(case.query_rope_sin)
        + _tensor_logical_bytes(case.key_rope_cos)
        + _tensor_logical_bytes(case.key_rope_sin)
        + int(case.query_row_sqrt_weights.numel()) * 8
    )


def _expected_replay_tensor_bytes(
    case: Lfm2EndpointQkResidualCase,
    seed: Lfm2EndpointQkvoSeed,
) -> int:
    batch, query_length, _hidden = map(int, case.target_query_inputs.shape)
    key_length = int(case.target_key_value_inputs.shape[1])
    query_head_elements = batch * seed.query_heads * query_length * seed.head_dim
    key_head_elements = batch * seed.key_value_heads * key_length * seed.head_dim
    repeated_value_elements = batch * seed.query_heads * key_length * seed.head_dim
    probability_elements = batch * seed.query_heads * query_length * key_length
    endpoint_elements = batch * query_length * seed.output_size
    return 2 * (2 * query_head_elements + 2 * key_head_elements) + 8 * (
        repeated_value_elements + probability_elements + endpoint_elements
    )


def _expected_semantic_probe_tensor_bytes(
    case: Lfm2EndpointQkResidualCase,
    seed: Lfm2EndpointQkvoSeed,
) -> int:
    batch, query_length, _hidden = map(int, case.target_query_inputs.shape)
    key_length = int(case.target_key_value_inputs.shape[1])
    probability_elements = batch * seed.query_heads * query_length * key_length
    endpoint_elements = batch * query_length * seed.output_size
    return (
        2 * (int(seed.query_weight.numel()) + int(seed.key_weight.numel())) * 4
        + (probability_elements + endpoint_elements) * 8
    )


def _conservative_solver_workspace_bytes(seed: Lfm2EndpointQkvoSeed) -> int:
    # Packed FP32 CG vectors, Q/K normal accumulators, BF16 frontier states,
    # and FP64 semantic aggregate witnesses. The factor deliberately exceeds
    # the simultaneously live tensors in the implementation.
    packed = int(seed.query_weight.numel()) + int(seed.key_weight.numel())
    return packed * 64


def _conservative_static_shard_workspace_bytes(
    seed: Lfm2EndpointQkvoSeed,
) -> int:
    # One device-resident packed Q/K direction, one packed gradient
    # accumulator, and one FP64 semantic aggregate. Concurrent shards occupy
    # distinct devices, so this remains a conservative per-device ceiling.
    packed = int(seed.query_weight.numel()) + int(seed.key_weight.numel())
    return packed * 16


def _build_observed_execution_receipt(
    *,
    seed: Lfm2EndpointQkvoSeed,
    fit_bank: Lfm2EndpointQkResidualEvidenceBank,
    prepared_cases: Sequence[_PreparedCase],
    numerical_sweeps: Sequence[Mapping[str, Any]],
    query_weight: Any,
    key_weight: Any,
    policy: Lfm2EndpointQkResidualFoldPolicy,
    solver_device: Any,
    compute_devices: Sequence[Any],
) -> dict[str, Any]:
    """Seal observed replay/JVP/VJP/normal work and actual device assignments."""

    torch = _torch()
    solver = torch.device(solver_device)
    workers = tuple(torch.device(value) for value in compute_devices)
    cases = tuple(prepared_cases)
    sweeps = tuple(numerical_sweeps)
    if (
        not workers
        or len(fit_bank.cases) < len(workers)
        or len(cases) != len(fit_bank.cases)
    ):
        raise Lfm2EndpointQkResidualFoldError(
            "Observed execution requires one distinct FIT case per compute device."
        )
    numerical_normal_calls = sum(
        int(sweep["solve_report"]["normal_operator_calls"]) for sweep in sweeps
    )
    expected_exact_replay_calls = (
        2 + len(sweeps) + sum(len(sweep.get("frontier", ())) for sweep in sweeps)
    )
    expected_jvp_calls = numerical_normal_calls + 1
    expected_vjp_calls = numerical_normal_calls + len(sweeps) + 1
    expected_normal_calls = numerical_normal_calls + 1
    observations_by_index: dict[int, dict[str, Any]] = {}
    solver_query_probe = torch.zeros(
        tuple(seed.query_weight.shape),
        device=solver,
        dtype=torch.float64,
    )
    solver_key_probe = torch.zeros(
        tuple(seed.key_weight.shape),
        device=solver,
        dtype=torch.float64,
    )
    with torch.no_grad():
        _attach_replay(
            cases,
            seed,
            query_weight,
            key_weight,
        )
        # The semantic probe is independent of candidate values. Each static
        # shard builds one Q/K direction, accumulates all case VJPs locally,
        # and performs one ordered reduction to the solver.
        for shard in _static_device_shards(cases):
            delta_query, delta_key = _build_semantic_shard_directions(shard, seed)
            shard_query_probe = torch.zeros(
                tuple(seed.query_weight.shape),
                device=shard.device,
                dtype=torch.float64,
            )
            shard_key_probe = torch.zeros(
                tuple(seed.key_weight.shape),
                device=shard.device,
                dtype=torch.float64,
            )
            for case in shard.cases:
                index = case.assignment_index
                evidence = fit_bank.cases[index]
                expected_device = workers[index % len(workers)]
                if (
                    case.assigned_device != str(expected_device)
                    or case.source.evidence_sha256 != evidence.evidence_sha256
                ):
                    raise Lfm2EndpointQkResidualAuthenticationError(
                        "Prepared FIT cases do not preserve authenticated bank order."
                    )
                (
                    delta_probabilities,
                    delta_endpoint,
                    query_gradient,
                    key_gradient,
                    witness_tensor,
                ) = _run_deterministic_semantic_normal_probe(
                    case,
                    seed,
                    delta_query=delta_query,
                    delta_key=delta_key,
                )
                shard_query_probe.add_(query_gradient.to(dtype=torch.float64))
                shard_key_probe.add_(key_gradient.to(dtype=torch.float64))
                witness, identity = _force_observed_positive_scalar(
                    witness_tensor,
                    intended_device=shard.device,
                    operation_name="cached_endpoint_jvp_vjp_normal_probe",
                )
                prepared_bytes = _prepared_persistent_tensor_bytes(case)
                replay_bytes = _attached_replay_tensor_bytes(case)
                semantic_bytes = _expected_semantic_probe_tensor_bytes(evidence, seed)
                if (
                    case.exact_replay_calls != expected_exact_replay_calls
                    or case.endpoint_jvp_calls != expected_jvp_calls
                    or case.endpoint_vjp_calls != expected_vjp_calls
                    or case.normal_operator_calls != expected_normal_calls
                    or case.semantic_probe_calls != 1
                ):
                    raise Lfm2EndpointQkResidualAuthenticationError(
                        "Observed endpoint solver counters differ from its analytical "
                        "sweep receipts."
                    )
                case_body = {
                    "assignment_index": index,
                    "intended_logical_device": str(shard.device),
                    **identity,
                    "evidence_sha256": evidence.evidence_sha256,
                    "trajectory_sha256": evidence.trajectory_sha256,
                    "target_receipt_sha256": evidence.target_receipt_sha256,
                    "analytical_operation": (
                        "exact_endpoint_jvp_then_vjp_normal_probe"
                    ),
                    "exact_replay_calls": case.exact_replay_calls,
                    "endpoint_jvp_calls": case.endpoint_jvp_calls,
                    "endpoint_vjp_calls": case.endpoint_vjp_calls,
                    "normal_operator_calls": case.normal_operator_calls,
                    "semantic_probe_calls": case.semantic_probe_calls,
                    "semantic_delta_probability_sha256": _tensor_sha256(
                        delta_probabilities
                    ),
                    "semantic_delta_endpoint_sha256": _tensor_sha256(delta_endpoint),
                    "semantic_query_gradient_sha256": _tensor_sha256(query_gradient),
                    "semantic_key_gradient_sha256": _tensor_sha256(key_gradient),
                    "prepared_persistent_tensor_bytes": prepared_bytes,
                    "attached_replay_tensor_bytes": replay_bytes,
                    "semantic_probe_tensor_bytes": semantic_bytes,
                    "analytical_tensor_element_count": (
                        int(delta_probabilities.numel())
                        + int(delta_endpoint.numel())
                        + int(query_gradient.numel())
                        + int(key_gradient.numel())
                    ),
                    "analytical_scalar": witness,
                    "forced_host_scalar_read": True,
                }
                observations_by_index[index] = {
                    **case_body,
                    "case_execution_receipt_sha256": _canonical_sha256(case_body),
                }
            if shard.device == solver:
                solver_query_probe.add_(shard_query_probe)
                solver_key_probe.add_(shard_key_probe)
            else:
                solver_query_probe.add_(shard_query_probe.to(device=solver))
                solver_key_probe.add_(shard_key_probe.to(device=solver))
            shard.fixed_cache.semantic_gradient_shard_reductions += 1
        observations = [
            observations_by_index[index] for index in range(len(fit_bank.cases))
        ]
        query = seed.query_weight.to(device=solver, dtype=torch.float64)
        key = seed.key_weight.to(device=solver, dtype=torch.float64)
        solver_tensor = (
            query.square().sum()
            + key.square().sum()
            + solver_query_probe.square().sum()
            + solver_key_probe.square().sum()
            + query.new_tensor(
                query.numel()
                + key.numel()
                + solver_query_probe.numel()
                + solver_key_probe.numel(),
            )
        )
        solver_witness, solver_identity = _force_observed_positive_scalar(
            solver_tensor,
            intended_device=solver,
            operation_name="solver_seed_qk_energy",
        )
    solver_body = {
        "intended_logical_device": str(solver),
        **solver_identity,
        "seed_sha256": seed.seed_sha256,
        "query_weight_sha256": _tensor_sha256(seed.query_weight),
        "key_weight_sha256": _tensor_sha256(seed.key_weight),
        "semantic_query_normal_sum_sha256": _tensor_sha256(solver_query_probe),
        "semantic_key_normal_sum_sha256": _tensor_sha256(solver_key_probe),
        "analytical_operation": (
            "dense_seed_qk_plus_worker_normal_sum_energy_and_parameter_count"
        ),
        "analytical_tensor_element_count": (
            int(query.numel())
            + int(key.numel())
            + int(solver_query_probe.numel())
            + int(solver_key_probe.numel())
        ),
        "analytical_scalar": solver_witness,
        "forced_host_scalar_read": True,
    }
    solver_observation = {
        **solver_body,
        "solver_execution_receipt_sha256": _canonical_sha256(solver_body),
    }
    requested = [str(device) for device in workers]
    expected_observed = [
        requested[index % len(requested)] for index in range(len(observations))
    ]
    observed = [
        str(observation["observed_logical_device"]) for observation in observations
    ]
    if observed != expected_observed or solver_observation[
        "observed_logical_device"
    ] != str(solver):
        raise Lfm2EndpointQkResidualFoldError(
            "Observed execution did not preserve round-robin device assignment."
        )
    static_shards = _static_device_shards(cases)
    static_shard_receipt = _static_device_shard_receipt(cases)
    shards_by_device = {str(shard.device): shard for shard in static_shards}
    concurrent_static_cuda_shards = _uses_concurrent_static_cuda_shards(static_shards)
    concurrency_mode = _static_shard_concurrency_mode(concurrent_static_cuda_shards)
    transfer_mode = _static_shard_transfer_mode(concurrent_static_cuda_shards)
    device_summaries = []
    cuda_uuids_by_device: dict[str, str] = {}
    for device in workers:
        matching = [
            observation
            for observation in observations
            if observation["observed_logical_device"] == str(device)
        ]
        if not matching:
            raise Lfm2EndpointQkResidualFoldError(
                "Observed execution did not cover every requested compute device."
            )
        uuids = {
            str(observation["cuda_uuid"])
            for observation in matching
            if observation["observed_device_type"] == "cuda"
        }
        if len(uuids) > 1:
            raise Lfm2EndpointQkResidualFoldError(
                "One logical CUDA worker reported multiple physical UUIDs."
            )
        if uuids:
            cuda_uuids_by_device[str(device)] = next(iter(uuids))
        prepared_device_bytes = sum(
            int(observation["prepared_persistent_tensor_bytes"])
            for observation in matching
        )
        replay_device_bytes = sum(
            int(observation["attached_replay_tensor_bytes"]) for observation in matching
        )
        probe_device_bytes = max(
            int(observation["semantic_probe_tensor_bytes"]) for observation in matching
        )
        solver_workspace_bytes = (
            _conservative_solver_workspace_bytes(seed)
            if str(device) == str(solver)
            else 0
        )
        shard_workspace_bytes = _conservative_static_shard_workspace_bytes(seed)
        shard = shards_by_device[str(device)]
        cache = shard.fixed_cache
        expected_cache_counters = {
            "replay_qk_shard_broadcasts": expected_exact_replay_calls,
            "normal_qk_direction_shard_broadcasts": numerical_normal_calls,
            "rhs_gradient_shard_reductions": len(sweeps),
            "normal_gradient_shard_reductions": numerical_normal_calls,
            "semantic_direction_shard_builds": 1,
            "semantic_gradient_shard_reductions": 1,
            "concurrent_rhs_shard_launches": (
                len(sweeps) if concurrent_static_cuda_shards else 0
            ),
            "concurrent_normal_shard_launches": (
                numerical_normal_calls if concurrent_static_cuda_shards else 0
            ),
        }
        if any(
            getattr(cache, name) != expected
            for name, expected in expected_cache_counters.items()
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Static endpoint shard transfer counters differ from the "
                "authenticated numerical sweep receipts."
            )
        fixed_cache_bytes = _fixed_device_tensor_cache_bytes(cache)
        conservative_peak_bytes = (
            prepared_device_bytes
            + replay_device_bytes
            + probe_device_bytes
            + fixed_cache_bytes
            + shard_workspace_bytes
            + solver_workspace_bytes
        )
        device_summaries.append(
            {
                "logical_device": str(device),
                "case_assignment_indices": [
                    int(observation["assignment_index"]) for observation in matching
                ],
                "case_count": len(matching),
                "cuda_uuid": matching[0]["cuda_uuid"],
                "cuda_name": matching[0]["cuda_name"],
                "exact_replay_calls": sum(
                    int(observation["exact_replay_calls"]) for observation in matching
                ),
                "endpoint_jvp_calls": sum(
                    int(observation["endpoint_jvp_calls"]) for observation in matching
                ),
                "endpoint_vjp_calls": sum(
                    int(observation["endpoint_vjp_calls"]) for observation in matching
                ),
                "normal_operator_calls": sum(
                    int(observation["normal_operator_calls"])
                    for observation in matching
                ),
                "semantic_probe_calls": sum(
                    int(observation["semantic_probe_calls"]) for observation in matching
                ),
                "prepared_persistent_tensor_bytes": prepared_device_bytes,
                "attached_replay_tensor_bytes": replay_device_bytes,
                "shared_fixed_tensor_cache_bytes": fixed_cache_bytes,
                "fixed_cache_receipt": _fixed_device_cache_receipt(cache),
                **expected_cache_counters,
                "qk_transfer_execution": transfer_mode,
                "maximum_serial_semantic_probe_tensor_bytes": probe_device_bytes,
                "conservative_static_shard_workspace_bytes": (shard_workspace_bytes),
                "conservative_solver_workspace_bytes": solver_workspace_bytes,
                "conservative_peak_accounted_bytes": conservative_peak_bytes,
                "byte_accounting_scope": (
                    "deterministic_declared_tensor_bytes_not_allocator_peak"
                ),
                "maximum_bytes_per_compute_device": (
                    policy.maximum_bytes_per_compute_device
                ),
                "within_compute_device_byte_ceiling": (
                    conservative_peak_bytes <= policy.maximum_bytes_per_compute_device
                ),
            }
        )
    if len(set(cuda_uuids_by_device.values())) != len(cuda_uuids_by_device):
        raise Lfm2EndpointQkResidualFoldError(
            "Distinct requested CUDA workers did not report distinct UUIDs."
        )
    endpoint_elements = sum(
        int(case.parent_endpoint_target.numel()) for case in fit_bank.cases
    )
    probability_elements = sum(
        int(case.target_probability_barycenter.numel()) for case in fit_bank.cases
    )
    prepared_bytes = sum(
        int(observation["prepared_persistent_tensor_bytes"])
        for observation in observations
    )
    replay_bytes = sum(
        int(observation["attached_replay_tensor_bytes"]) for observation in observations
    )
    maximum_probe_bytes = max(
        int(observation["semantic_probe_tensor_bytes"]) for observation in observations
    )
    total_fixed_cache_bytes = sum(
        int(summary["shared_fixed_tensor_cache_bytes"]) for summary in device_summaries
    )
    resource_observation = {
        "fit_case_count": len(fit_bank.cases),
        "endpoint_elements": endpoint_elements,
        "probability_elements": probability_elements,
        "prepared_persistent_tensor_bytes": prepared_bytes,
        "attached_replay_tensor_bytes": replay_bytes,
        "shared_fixed_tensor_cache_bytes": total_fixed_cache_bytes,
        "fixed_tensor_cache_count": len(device_summaries),
        "maximum_serial_semantic_probe_tensor_bytes": maximum_probe_bytes,
        "conservative_static_shard_workspace_bytes_per_device": (
            _conservative_static_shard_workspace_bytes(seed)
        ),
        "maximum_fit_cases": policy.maximum_fit_cases,
        "maximum_endpoint_elements": policy.maximum_endpoint_elements,
        "maximum_probability_elements": policy.maximum_probability_elements,
        "maximum_prepared_tensor_bytes": policy.maximum_prepared_tensor_bytes,
        "maximum_replay_tensor_bytes": policy.maximum_replay_tensor_bytes,
        "maximum_bytes_per_compute_device": (policy.maximum_bytes_per_compute_device),
        "byte_accounting_scope": (
            "deterministic_declared_tensor_bytes_not_allocator_peak"
        ),
        "production_allocator_peak_authority": (
            "outer_real_probe_continuous_cuda_peak_reserved_monitor"
        ),
        "every_compute_device_within_byte_ceiling": all(
            bool(summary["within_compute_device_byte_ceiling"])
            for summary in device_summaries
        ),
        "within_all_resource_ceilings": (
            len(fit_bank.cases) <= policy.maximum_fit_cases
            and endpoint_elements <= policy.maximum_endpoint_elements
            and probability_elements <= policy.maximum_probability_elements
            and prepared_bytes <= policy.maximum_prepared_tensor_bytes
            and replay_bytes <= policy.maximum_replay_tensor_bytes
            and all(
                bool(summary["within_compute_device_byte_ceiling"])
                for summary in device_summaries
            )
        ),
    }
    if not resource_observation["within_all_resource_ceilings"]:
        raise Lfm2EndpointQkResidualFoldError(
            "Observed endpoint solver tensors exceed authenticated byte ceilings."
        )
    body = {
        "schema": LFM2_ENDPOINT_QK_EXECUTION_SCHEMA,
        "requested_solver_device": str(solver),
        "requested_compute_devices": requested,
        "requested_compute_device_count": len(requested),
        "case_assignment": "authenticated_bank_order_round_robin",
        "compute_device_concurrency": concurrency_mode,
        "qk_transfer_execution": transfer_mode,
        "static_device_shard_receipt": static_shard_receipt,
        "every_requested_compute_device_observed": True,
        "distinct_cuda_uuid_count": len(set(cuda_uuids_by_device.values())),
        "compute_observations": observations,
        "compute_device_summaries": device_summaries,
        "solver_observation": solver_observation,
        "numerical_sweep_count": len(sweeps),
        "numerical_normal_operator_calls": numerical_normal_calls,
        "total_exact_replay_calls": sum(
            int(observation["exact_replay_calls"]) for observation in observations
        ),
        "total_endpoint_jvp_calls": sum(
            int(observation["endpoint_jvp_calls"]) for observation in observations
        ),
        "total_endpoint_vjp_calls": sum(
            int(observation["endpoint_vjp_calls"]) for observation in observations
        ),
        "total_normal_operator_calls": sum(
            int(observation["normal_operator_calls"]) for observation in observations
        ),
        "total_semantic_probe_calls": sum(
            int(observation["semantic_probe_calls"]) for observation in observations
        ),
        "resource_observation": resource_observation,
        "operation_audit": {
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        },
    }
    return {
        **body,
        "receipt_sha256": _canonical_sha256(body),
    }


def _verify_observed_execution_receipt(
    receipt: Mapping[str, Any],
    *,
    seed: Lfm2EndpointQkvoSeed,
    fit_bank: Lfm2EndpointQkResidualEvidenceBank,
    query_weight: Any,
    key_weight: Any,
    policy: Lfm2EndpointQkResidualFoldPolicy,
    sweeps: Sequence[Mapping[str, Any]],
) -> str:
    if not isinstance(receipt, Mapping):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Observed execution receipt must be a mapping."
        )
    body = dict(receipt)
    supplied = body.pop("receipt_sha256", None)
    if supplied != _canonical_sha256(body):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Observed execution receipt seal is invalid."
        )
    expected_receipt_fields = {
        "schema",
        "requested_solver_device",
        "requested_compute_devices",
        "requested_compute_device_count",
        "case_assignment",
        "compute_device_concurrency",
        "qk_transfer_execution",
        "static_device_shard_receipt",
        "every_requested_compute_device_observed",
        "distinct_cuda_uuid_count",
        "compute_observations",
        "compute_device_summaries",
        "solver_observation",
        "numerical_sweep_count",
        "numerical_normal_operator_calls",
        "total_exact_replay_calls",
        "total_endpoint_jvp_calls",
        "total_endpoint_vjp_calls",
        "total_normal_operator_calls",
        "total_semantic_probe_calls",
        "resource_observation",
        "operation_audit",
    }
    expected_operation_audit = {
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
    }
    expected_workers = list(policy.compute_devices)
    expected_concurrent_static_cuda_shards = _supports_concurrent_static_cuda_devices(
        expected_workers
    )
    expected_concurrency_mode = _static_shard_concurrency_mode(
        expected_concurrent_static_cuda_shards
    )
    expected_transfer_mode = _static_shard_transfer_mode(
        expected_concurrent_static_cuda_shards
    )
    observations = body.get("compute_observations")
    summaries = body.get("compute_device_summaries")
    if (
        set(body) != expected_receipt_fields
        or body.get("schema") != LFM2_ENDPOINT_QK_EXECUTION_SCHEMA
        or body.get("requested_solver_device") != policy.solver_device
        or body.get("requested_compute_devices") != expected_workers
        or body.get("requested_compute_device_count") != len(expected_workers)
        or body.get("case_assignment") != "authenticated_bank_order_round_robin"
        or body.get("compute_device_concurrency") != expected_concurrency_mode
        or body.get("qk_transfer_execution") != expected_transfer_mode
        or body.get("every_requested_compute_device_observed") is not True
        or body.get("operation_audit") != expected_operation_audit
        or not isinstance(observations, list)
        or len(observations) != len(fit_bank.cases)
        or not isinstance(summaries, list)
        or len(summaries) != len(expected_workers)
    ):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Observed execution receipt does not bind the requested devices/cases."
        )
    sweep_values = tuple(sweeps)
    numerical_normal_calls = sum(
        int(sweep["solve_report"]["normal_operator_calls"]) for sweep in sweep_values
    )
    expected_counts = {
        "exact_replay_calls": (
            2
            + len(sweep_values)
            + sum(len(sweep.get("frontier", ())) for sweep in sweep_values)
        ),
        "endpoint_jvp_calls": numerical_normal_calls + 1,
        "endpoint_vjp_calls": numerical_normal_calls + len(sweep_values) + 1,
        "normal_operator_calls": numerical_normal_calls + 1,
        "semantic_probe_calls": 1,
    }
    total_fields = {
        "exact_replay_calls": "total_exact_replay_calls",
        "endpoint_jvp_calls": "total_endpoint_jvp_calls",
        "endpoint_vjp_calls": "total_endpoint_vjp_calls",
        "normal_operator_calls": "total_normal_operator_calls",
        "semantic_probe_calls": "total_semantic_probe_calls",
    }
    cuda_identity_by_device: dict[str, tuple[str, str]] = {}
    expected_case_fields = {
        "assignment_index",
        "intended_logical_device",
        "observed_logical_device",
        "observed_device_type",
        "observed_device_index",
        "cuda_uuid",
        "cuda_name",
        "evidence_sha256",
        "trajectory_sha256",
        "target_receipt_sha256",
        "analytical_operation",
        "exact_replay_calls",
        "endpoint_jvp_calls",
        "endpoint_vjp_calls",
        "normal_operator_calls",
        "semantic_probe_calls",
        "semantic_delta_probability_sha256",
        "semantic_delta_endpoint_sha256",
        "semantic_query_gradient_sha256",
        "semantic_key_gradient_sha256",
        "prepared_persistent_tensor_bytes",
        "attached_replay_tensor_bytes",
        "semantic_probe_tensor_bytes",
        "analytical_tensor_element_count",
        "analytical_scalar",
        "forced_host_scalar_read",
    }
    for index, (observation, case) in enumerate(
        zip(observations, fit_bank.cases, strict=True)
    ):
        if not isinstance(observation, Mapping):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "A compute observation is not a mapping."
            )
        observation_body = dict(observation)
        observation_seal = observation_body.pop(
            "case_execution_receipt_sha256",
            None,
        )
        expected_device = expected_workers[index % len(expected_workers)]
        expected_prepared_bytes = _expected_prepared_tensor_bytes(case)
        expected_replay_bytes = _expected_replay_tensor_bytes(case, seed)
        expected_probe_bytes = _expected_semantic_probe_tensor_bytes(case, seed)
        expected_analytical_elements = (
            int(case.target_probability_barycenter.numel())
            + int(case.parent_endpoint_target.numel())
            + int(seed.query_weight.numel())
            + int(seed.key_weight.numel())
        )
        if (
            observation_seal != _canonical_sha256(observation_body)
            or set(observation_body) != expected_case_fields
            or observation.get("assignment_index") != index
            or observation.get("intended_logical_device") != expected_device
            or observation.get("observed_logical_device") != expected_device
            or observation.get("evidence_sha256") != case.evidence_sha256
            or observation.get("trajectory_sha256") != case.trajectory_sha256
            or observation.get("target_receipt_sha256") != case.target_receipt_sha256
            or observation.get("analytical_operation")
            != "exact_endpoint_jvp_then_vjp_normal_probe"
            or observation.get("forced_host_scalar_read") is not True
            or observation.get("prepared_persistent_tensor_bytes")
            != expected_prepared_bytes
            or observation.get("attached_replay_tensor_bytes") != expected_replay_bytes
            or observation.get("semantic_probe_tensor_bytes") != expected_probe_bytes
            or observation.get("analytical_tensor_element_count")
            != expected_analytical_elements
            or any(
                observation.get(name) != expected
                for name, expected in expected_counts.items()
            )
            or any(
                _required_sha256(observation.get(name), name) is None
                for name in (
                    "semantic_delta_probability_sha256",
                    "semantic_delta_endpoint_sha256",
                    "semantic_query_gradient_sha256",
                    "semantic_key_gradient_sha256",
                )
            )
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "A semantic endpoint solver case receipt is invalid."
            )
        scalar = observation.get("analytical_scalar")
        if (
            isinstance(scalar, bool)
            or not isinstance(scalar, (float, int))
            or not math.isfinite(float(scalar))
            or float(scalar) <= 0.0
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "A semantic endpoint solver witness is not finite and positive."
            )
        if observation.get("observed_device_type") == "cuda":
            uuid = str(observation.get("cuda_uuid"))
            name = str(observation.get("cuda_name"))
            if not uuid or uuid == "None" or not name or name == "None":
                raise Lfm2EndpointQkResidualAuthenticationError(
                    "A CUDA endpoint worker lacks authenticated physical identity."
                )
            identity = cuda_identity_by_device.setdefault(
                expected_device,
                (uuid, name),
            )
            if identity != (uuid, name):
                raise Lfm2EndpointQkResidualAuthenticationError(
                    "A logical CUDA worker changed physical identity mid-receipt."
                )
        elif (
            observation.get("observed_device_type") != "cpu"
            or observation.get("cuda_uuid") is not None
            or observation.get("cuda_name") is not None
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "A CPU endpoint worker reported invalid CUDA identity."
            )
    if len(set(cuda_identity_by_device.values())) != len(cuda_identity_by_device):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Distinct CUDA workers do not have distinct physical identities."
        )
    if body.get("distinct_cuda_uuid_count") != len(cuda_identity_by_device):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Observed execution reports the wrong distinct CUDA worker count."
        )
    torch = _torch()
    solver, workers = _resolve_execution_devices(policy)
    replay_cases = _prepare_cases_round_robin(
        fit_bank.cases,
        seed,
        compute_devices=workers,
    )
    replay_shards = _static_device_shards(replay_cases)
    replay_shards_by_device = {str(shard.device): shard for shard in replay_shards}
    expected_static_shard_receipt = _static_device_shard_receipt(replay_cases)
    if body.get("static_device_shard_receipt") != expected_static_shard_receipt:
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Observed execution does not bind the reconstructed static shards "
            "and shared fixed tensor caches."
        )
    expected_summaries = []
    for device in expected_workers:
        matching = [
            observation
            for observation in observations
            if observation["observed_logical_device"] == device
        ]
        prepared_device_bytes = sum(
            int(observation["prepared_persistent_tensor_bytes"])
            for observation in matching
        )
        replay_device_bytes = sum(
            int(observation["attached_replay_tensor_bytes"]) for observation in matching
        )
        probe_device_bytes = max(
            int(observation["semantic_probe_tensor_bytes"]) for observation in matching
        )
        solver_workspace_bytes = (
            _conservative_solver_workspace_bytes(seed)
            if device == policy.solver_device
            else 0
        )
        shard_workspace_bytes = _conservative_static_shard_workspace_bytes(seed)
        replay_shard = replay_shards_by_device[device]
        fixed_cache_bytes = _fixed_device_tensor_cache_bytes(replay_shard.fixed_cache)
        expected_cache_counters = {
            "replay_qk_shard_broadcasts": expected_counts["exact_replay_calls"],
            "normal_qk_direction_shard_broadcasts": numerical_normal_calls,
            "rhs_gradient_shard_reductions": len(sweep_values),
            "normal_gradient_shard_reductions": numerical_normal_calls,
            "semantic_direction_shard_builds": 1,
            "semantic_gradient_shard_reductions": 1,
            "concurrent_rhs_shard_launches": (
                len(sweep_values) if expected_concurrent_static_cuda_shards else 0
            ),
            "concurrent_normal_shard_launches": (
                numerical_normal_calls if expected_concurrent_static_cuda_shards else 0
            ),
        }
        conservative_peak_bytes = (
            prepared_device_bytes
            + replay_device_bytes
            + probe_device_bytes
            + fixed_cache_bytes
            + shard_workspace_bytes
            + solver_workspace_bytes
        )
        expected_summaries.append(
            {
                "logical_device": device,
                "case_assignment_indices": [
                    int(observation["assignment_index"]) for observation in matching
                ],
                "case_count": len(matching),
                "cuda_uuid": matching[0]["cuda_uuid"],
                "cuda_name": matching[0]["cuda_name"],
                **{
                    name: sum(int(observation[name]) for observation in matching)
                    for name in expected_counts
                },
                "prepared_persistent_tensor_bytes": prepared_device_bytes,
                "attached_replay_tensor_bytes": replay_device_bytes,
                "shared_fixed_tensor_cache_bytes": fixed_cache_bytes,
                "fixed_cache_receipt": _fixed_device_cache_receipt(
                    replay_shard.fixed_cache
                ),
                **expected_cache_counters,
                "qk_transfer_execution": expected_transfer_mode,
                "maximum_serial_semantic_probe_tensor_bytes": probe_device_bytes,
                "conservative_static_shard_workspace_bytes": (shard_workspace_bytes),
                "conservative_solver_workspace_bytes": solver_workspace_bytes,
                "conservative_peak_accounted_bytes": conservative_peak_bytes,
                "byte_accounting_scope": (
                    "deterministic_declared_tensor_bytes_not_allocator_peak"
                ),
                "maximum_bytes_per_compute_device": (
                    policy.maximum_bytes_per_compute_device
                ),
                "within_compute_device_byte_ceiling": (
                    conservative_peak_bytes <= policy.maximum_bytes_per_compute_device
                ),
            }
        )
    endpoint_elements = sum(
        int(case.parent_endpoint_target.numel()) for case in fit_bank.cases
    )
    probability_elements = sum(
        int(case.target_probability_barycenter.numel()) for case in fit_bank.cases
    )
    prepared_bytes = sum(
        _expected_prepared_tensor_bytes(case) for case in fit_bank.cases
    )
    replay_bytes = sum(
        _expected_replay_tensor_bytes(case, seed) for case in fit_bank.cases
    )
    maximum_probe_bytes = max(
        _expected_semantic_probe_tensor_bytes(case, seed) for case in fit_bank.cases
    )
    total_fixed_cache_bytes = sum(
        int(summary["shared_fixed_tensor_cache_bytes"])
        for summary in expected_summaries
    )
    expected_resource = {
        "fit_case_count": len(fit_bank.cases),
        "endpoint_elements": endpoint_elements,
        "probability_elements": probability_elements,
        "prepared_persistent_tensor_bytes": prepared_bytes,
        "attached_replay_tensor_bytes": replay_bytes,
        "shared_fixed_tensor_cache_bytes": total_fixed_cache_bytes,
        "fixed_tensor_cache_count": len(expected_summaries),
        "maximum_serial_semantic_probe_tensor_bytes": maximum_probe_bytes,
        "conservative_static_shard_workspace_bytes_per_device": (
            _conservative_static_shard_workspace_bytes(seed)
        ),
        "maximum_fit_cases": policy.maximum_fit_cases,
        "maximum_endpoint_elements": policy.maximum_endpoint_elements,
        "maximum_probability_elements": policy.maximum_probability_elements,
        "maximum_prepared_tensor_bytes": policy.maximum_prepared_tensor_bytes,
        "maximum_replay_tensor_bytes": policy.maximum_replay_tensor_bytes,
        "maximum_bytes_per_compute_device": (policy.maximum_bytes_per_compute_device),
        "byte_accounting_scope": (
            "deterministic_declared_tensor_bytes_not_allocator_peak"
        ),
        "production_allocator_peak_authority": (
            "outer_real_probe_continuous_cuda_peak_reserved_monitor"
        ),
        "every_compute_device_within_byte_ceiling": all(
            bool(summary["within_compute_device_byte_ceiling"])
            for summary in expected_summaries
        ),
        "within_all_resource_ceilings": (
            len(fit_bank.cases) <= policy.maximum_fit_cases
            and endpoint_elements <= policy.maximum_endpoint_elements
            and probability_elements <= policy.maximum_probability_elements
            and prepared_bytes <= policy.maximum_prepared_tensor_bytes
            and replay_bytes <= policy.maximum_replay_tensor_bytes
            and all(
                bool(summary["within_compute_device_byte_ceiling"])
                for summary in expected_summaries
            )
        ),
    }
    replay_solver_query_probe = torch.zeros(
        tuple(seed.query_weight.shape),
        device=solver,
        dtype=torch.float64,
    )
    replay_solver_key_probe = torch.zeros(
        tuple(seed.key_weight.shape),
        device=solver,
        dtype=torch.float64,
    )
    with torch.no_grad():
        _attach_replay(
            replay_cases,
            seed,
            query_weight,
            key_weight,
        )
        for shard in replay_shards:
            delta_query, delta_key = _build_semantic_shard_directions(shard, seed)
            shard_query_probe = torch.zeros(
                tuple(seed.query_weight.shape),
                device=shard.device,
                dtype=torch.float64,
            )
            shard_key_probe = torch.zeros(
                tuple(seed.key_weight.shape),
                device=shard.device,
                dtype=torch.float64,
            )
            for case in shard.cases:
                index = case.assignment_index
                (
                    delta_probabilities,
                    delta_endpoint,
                    query_gradient,
                    key_gradient,
                    witness_tensor,
                ) = _run_deterministic_semantic_normal_probe(
                    case,
                    seed,
                    delta_query=delta_query,
                    delta_key=delta_key,
                )
                shard_query_probe.add_(query_gradient.to(dtype=torch.float64))
                shard_key_probe.add_(key_gradient.to(dtype=torch.float64))
                observation = observations[index]
                actual_identity = _observed_device_identity(shard.device)
                expected_hashes = {
                    "semantic_delta_probability_sha256": _tensor_sha256(
                        delta_probabilities
                    ),
                    "semantic_delta_endpoint_sha256": _tensor_sha256(delta_endpoint),
                    "semantic_query_gradient_sha256": _tensor_sha256(query_gradient),
                    "semantic_key_gradient_sha256": _tensor_sha256(key_gradient),
                }
                replay_scalar = float(witness_tensor.detach().item())
                if (
                    any(
                        observation.get(name) != digest
                        for name, digest in expected_hashes.items()
                    )
                    or any(
                        observation.get(name) != value
                        for name, value in actual_identity.items()
                    )
                    or replay_scalar != float(observation["analytical_scalar"])
                ):
                    raise Lfm2EndpointQkResidualAuthenticationError(
                        "A semantic endpoint solver receipt does not replay from "
                        "its bound bank, candidate state, and execution device."
                    )
            if shard.device == solver:
                replay_solver_query_probe.add_(shard_query_probe)
                replay_solver_key_probe.add_(shard_key_probe)
            else:
                replay_solver_query_probe.add_(shard_query_probe.to(device=solver))
                replay_solver_key_probe.add_(shard_key_probe.to(device=solver))
            shard.fixed_cache.semantic_gradient_shard_reductions += 1
    solver_observation = body.get("solver_observation")
    if not isinstance(solver_observation, Mapping):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Endpoint solver observation is missing."
        )
    solver_body = dict(solver_observation)
    solver_seal = solver_body.pop("solver_execution_receipt_sha256", None)
    expected_solver_fields = {
        "intended_logical_device",
        "observed_logical_device",
        "observed_device_type",
        "observed_device_index",
        "cuda_uuid",
        "cuda_name",
        "seed_sha256",
        "query_weight_sha256",
        "key_weight_sha256",
        "semantic_query_normal_sum_sha256",
        "semantic_key_normal_sum_sha256",
        "analytical_operation",
        "analytical_tensor_element_count",
        "analytical_scalar",
        "forced_host_scalar_read",
    }
    replay_solver_tensor = (
        seed.query_weight.to(device=solver, dtype=torch.float64).square().sum()
        + seed.key_weight.to(device=solver, dtype=torch.float64).square().sum()
        + replay_solver_query_probe.square().sum()
        + replay_solver_key_probe.square().sum()
        + replay_solver_query_probe.new_tensor(
            2 * (int(seed.query_weight.numel()) + int(seed.key_weight.numel()))
        )
    )
    replay_solver_scalar = float(replay_solver_tensor.detach().item())
    replay_solver_identity = _observed_device_identity(solver)
    if (
        solver_seal != _canonical_sha256(solver_body)
        or set(solver_body) != expected_solver_fields
        or solver_observation.get("intended_logical_device") != policy.solver_device
        or solver_observation.get("observed_logical_device") != policy.solver_device
        or solver_observation.get("seed_sha256") != seed.seed_sha256
        or solver_observation.get("query_weight_sha256")
        != _tensor_sha256(seed.query_weight)
        or solver_observation.get("key_weight_sha256")
        != _tensor_sha256(seed.key_weight)
        or solver_observation.get("semantic_query_normal_sum_sha256")
        != _tensor_sha256(replay_solver_query_probe)
        or solver_observation.get("semantic_key_normal_sum_sha256")
        != _tensor_sha256(replay_solver_key_probe)
        or solver_observation.get("analytical_operation")
        != "dense_seed_qk_plus_worker_normal_sum_energy_and_parameter_count"
        or solver_observation.get("analytical_tensor_element_count")
        != 2 * (int(seed.query_weight.numel()) + int(seed.key_weight.numel()))
        or solver_observation.get("forced_host_scalar_read") is not True
        or float(solver_observation.get("analytical_scalar", math.nan))
        != replay_solver_scalar
        or any(
            solver_observation.get(name) != value
            for name, value in replay_solver_identity.items()
        )
        or body.get("compute_device_summaries") != expected_summaries
        or body.get("numerical_sweep_count") != len(sweep_values)
        or body.get("numerical_normal_operator_calls") != numerical_normal_calls
        or body.get("resource_observation") != expected_resource
        or expected_resource["within_all_resource_ceilings"] is not True
        or any(
            body.get(total_name)
            != sum(int(observation[count_name]) for observation in observations)
            for count_name, total_name in total_fields.items()
        )
    ):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Observed endpoint solver aggregate receipt is invalid."
        )
    return str(supplied)


@dataclass(frozen=True, slots=True)
class Lfm2EndpointQkMetrics:
    case_count: int
    eligible_route_rows: int
    endpoint_relative_rmse: float
    probability_relative_rmse: float
    js_divergence_mean: float
    js_divergence_worst: float
    top1_route_agreement_fraction: float
    maximum_causal_probability_leakage: float

    def __post_init__(self) -> None:
        _positive_int(self.case_count, "case_count")
        _positive_int(self.eligible_route_rows, "eligible_route_rows")
        for name in (
            "endpoint_relative_rmse",
            "probability_relative_rmse",
            "js_divergence_mean",
            "js_divergence_worst",
            "maximum_causal_probability_leakage",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        top1 = _finite_nonnegative(
            self.top1_route_agreement_fraction,
            "top1_route_agreement_fraction",
        )
        if top1 > 1.0:
            raise Lfm2EndpointQkResidualFoldError(
                "top1_route_agreement_fraction must remain inside [0,1]."
            )
        object.__setattr__(self, "top1_route_agreement_fraction", top1)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(slots=True)
class _FixedDeviceTensorCache:
    """One immutable fixed V/O/Q-RMS/K-RMS materialization per worker device."""

    device: Any
    value_weight_bfloat16: Any
    output_weight_float64: Any
    query_norm_weight_bfloat16: Any
    key_norm_weight_bfloat16: Any
    query_norm_weight_float32: Any
    key_norm_weight_float32: Any
    replay_qk_shard_broadcasts: int = 0
    normal_qk_direction_shard_broadcasts: int = 0
    rhs_gradient_shard_reductions: int = 0
    normal_gradient_shard_reductions: int = 0
    semantic_direction_shard_builds: int = 0
    semantic_gradient_shard_reductions: int = 0
    concurrent_rhs_shard_launches: int = 0
    concurrent_normal_shard_launches: int = 0


def _build_fixed_device_tensor_cache(
    seed: Lfm2EndpointQkvoSeed,
    device: Any,
) -> _FixedDeviceTensorCache:
    torch = _torch()
    resolved = torch.device(device)
    return _FixedDeviceTensorCache(
        device=resolved,
        value_weight_bfloat16=seed.value_weight.to(
            device=resolved,
            dtype=torch.bfloat16,
        ),
        output_weight_float64=seed.output_weight.to(
            device=resolved,
            dtype=torch.float64,
        ),
        query_norm_weight_bfloat16=seed.query_norm_weight.to(
            device=resolved,
            dtype=torch.bfloat16,
        ),
        key_norm_weight_bfloat16=seed.key_norm_weight.to(
            device=resolved,
            dtype=torch.bfloat16,
        ),
        query_norm_weight_float32=seed.query_norm_weight.to(
            device=resolved,
            dtype=torch.float32,
        ),
        key_norm_weight_float32=seed.key_norm_weight.to(
            device=resolved,
            dtype=torch.float32,
        ),
    )


def _fixed_device_tensor_cache_bytes(cache: _FixedDeviceTensorCache) -> int:
    return sum(
        _tensor_logical_bytes(value)
        for value in (
            cache.value_weight_bfloat16,
            cache.output_weight_float64,
            cache.query_norm_weight_bfloat16,
            cache.key_norm_weight_bfloat16,
            cache.query_norm_weight_float32,
            cache.key_norm_weight_float32,
        )
    )


@dataclass(slots=True)
class _PreparedCase:
    source: Lfm2EndpointQkResidualCase
    assignment_index: int
    assigned_device: str
    fixed_cache: _FixedDeviceTensorCache
    query_inputs: Any
    key_inputs: Any
    target_probabilities: Any
    target_endpoint: Any
    mask: Any
    query_cosine: Any
    query_sine: Any
    key_cosine: Any
    key_sine: Any
    query_row_weights: Any
    query_linear: Any | None = None
    key_linear: Any | None = None
    query_post: Any | None = None
    key_post: Any | None = None
    repeated_value: Any | None = None
    probabilities: Any | None = None
    endpoint: Any | None = None
    exact_replay_calls: int = 0
    endpoint_jvp_calls: int = 0
    endpoint_vjp_calls: int = 0
    normal_operator_calls: int = 0
    semantic_probe_calls: int = 0


@dataclass(frozen=True, slots=True)
class _Replay:
    query_linear: Any
    key_linear: Any
    query_post: Any
    key_post: Any
    repeated_value: Any
    probabilities: Any
    endpoint: Any


@dataclass(frozen=True, slots=True)
class _StaticDeviceShard:
    device: Any
    fixed_cache: _FixedDeviceTensorCache
    cases: tuple[_PreparedCase, ...]


def _static_device_shards(
    cases: Sequence[_PreparedCase],
) -> tuple[_StaticDeviceShard, ...]:
    ordered_cases = tuple(cases)
    devices: list[str] = []
    for case in ordered_cases:
        if case.assigned_device not in devices:
            devices.append(case.assigned_device)
    shards = []
    for logical_device in devices:
        matching = tuple(
            case for case in ordered_cases if case.assigned_device == logical_device
        )
        if not matching:
            raise Lfm2EndpointQkResidualFoldError(
                "A static endpoint device shard is unexpectedly empty."
            )
        cache = matching[0].fixed_cache
        if (
            str(cache.device) != logical_device
            or any(case.fixed_cache is not cache for case in matching)
            or any(str(case.query_inputs.device) != logical_device for case in matching)
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Prepared cases do not share one fixed tensor cache per device."
            )
        shards.append(
            _StaticDeviceShard(
                device=cache.device,
                fixed_cache=cache,
                cases=matching,
            )
        )
    return tuple(shards)


def _supports_concurrent_static_cuda_devices(devices: Sequence[Any]) -> bool:
    """Return whether distinct CUDA devices can execute independent shard work."""

    torch = _torch()
    resolved = tuple(torch.device(device) for device in devices)
    return (
        len(resolved) > 1
        and all(device.type == "cuda" for device in resolved)
        and len({str(device) for device in resolved}) == len(resolved)
    )


def _uses_concurrent_static_cuda_shards(
    shards: Sequence[_StaticDeviceShard],
) -> bool:
    return _supports_concurrent_static_cuda_devices(
        tuple(shard.device for shard in shards)
    )


def _static_shard_concurrency_mode(concurrent: bool) -> str:
    if concurrent:
        return "thread_barrier_concurrent_static_cuda_shards"
    return "serial_static_device_shards"


def _static_shard_transfer_mode(concurrent: bool) -> str:
    if concurrent:
        return (
            "one_local_accumulation_per_concurrent_static_cuda_shard_then_"
            "canonical_requested_device_reduction"
        )
    return "once_per_static_device_shard_serial_requested_device_order"


def _execute_static_device_shards(
    shards: Sequence[_StaticDeviceShard],
    operation: Any,
) -> tuple[Any, ...]:
    """Launch independent CUDA shards together and return canonical-order results."""

    ordered = tuple(shards)
    if not _uses_concurrent_static_cuda_shards(ordered):
        return tuple(operation(shard) for shard in ordered)

    torch = _torch()
    start_barrier = Barrier(len(ordered))

    def _worker(shard: _StaticDeviceShard) -> Any:
        start_barrier.wait()
        with torch.no_grad():
            if shard.device.type != "cuda":  # test-only forced-concurrency path
                return operation(shard)
            with torch.cuda.device(shard.device):
                result = operation(shard)
                # A returned tensor is not evidence that queued CUDA work completed.
                # Synchronize each independent device before canonical reduction.
                torch.cuda.synchronize(shard.device)
                return result

    with ThreadPoolExecutor(
        max_workers=len(ordered),
        thread_name_prefix="lfm2-endpoint-shard",
    ) as executor:
        futures = tuple(executor.submit(_worker, shard) for shard in ordered)
        # Consume in authenticated shard order, independent of completion order.
        return tuple(future.result() for future in futures)


def _fixed_device_cache_receipt(
    cache: _FixedDeviceTensorCache,
) -> dict[str, Any]:
    body = {
        "schema": _FIXED_DEVICE_CACHE_SCHEMA,
        "logical_device": str(cache.device),
        "fixed_value_bfloat16_sha256": _tensor_sha256(cache.value_weight_bfloat16),
        "fixed_output_float64_sha256": _tensor_sha256(cache.output_weight_float64),
        "query_rms_bfloat16_sha256": _tensor_sha256(cache.query_norm_weight_bfloat16),
        "key_rms_bfloat16_sha256": _tensor_sha256(cache.key_norm_weight_bfloat16),
        "query_rms_float32_sha256": _tensor_sha256(cache.query_norm_weight_float32),
        "key_rms_float32_sha256": _tensor_sha256(cache.key_norm_weight_float32),
        "shared_fixed_tensor_bytes": _fixed_device_tensor_cache_bytes(cache),
        "fixed_cache_builds": 1,
        "fixed_tensor_materialization": "once_per_requested_compute_device",
    }
    return {
        **body,
        "fixed_cache_receipt_sha256": _canonical_sha256(body),
    }


def _static_device_shard_receipt(
    cases: Sequence[_PreparedCase],
) -> dict[str, Any]:
    shards = _static_device_shards(cases)
    body = {
        "schema": _STATIC_DEVICE_SHARD_SCHEMA,
        "assignment": "authenticated_bank_order_round_robin_static_shards",
        "device_order": [str(shard.device) for shard in shards],
        "shards": [
            {
                "logical_device": str(shard.device),
                "case_assignment_indices": [
                    case.assignment_index for case in shard.cases
                ],
                "case_count": len(shard.cases),
                "all_cases_share_fixed_cache_by_identity": all(
                    case.fixed_cache is shard.fixed_cache for case in shard.cases
                ),
                "fixed_cache_receipt": _fixed_device_cache_receipt(shard.fixed_cache),
            }
            for shard in shards
        ],
    }
    return {
        **body,
        "static_device_shards_sha256": _canonical_sha256(body),
    }


def _prepare_case(
    case: Lfm2EndpointQkResidualCase,
    seed: Lfm2EndpointQkvoSeed,
    *,
    device: Any,
    assignment_index: int = 0,
    fixed_cache: _FixedDeviceTensorCache | None = None,
) -> _PreparedCase:
    torch = _torch()
    resolved_device = torch.device(device)
    cache = fixed_cache or _build_fixed_device_tensor_cache(seed, resolved_device)
    if str(cache.device) != str(resolved_device):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "A prepared endpoint case received a fixed cache for another device."
        )
    case.verify()
    batch, query_length, hidden = map(int, case.target_query_inputs.shape)
    key_length = int(case.target_key_value_inputs.shape[1])
    if (
        hidden != seed.hidden_size
        or int(case.parent_endpoint_target.shape[-1]) != seed.output_size
        or tuple(case.target_probability_barycenter.shape)
        != (batch, seed.query_heads, query_length, key_length)
        or int(case.query_rope_cos.shape[-1]) != seed.head_dim
    ):
        raise Lfm2EndpointQkResidualFoldError(
            "Endpoint evidence differs from the dense Q/K/V/O seed topology."
        )
    return _PreparedCase(
        source=case,
        assignment_index=assignment_index,
        assigned_device=str(resolved_device),
        fixed_cache=cache,
        query_inputs=case.target_query_inputs.to(device=resolved_device),
        key_inputs=case.target_key_value_inputs.to(device=resolved_device),
        target_probabilities=case.target_probability_barycenter.to(
            device=resolved_device,
            dtype=torch.float64,
        ),
        target_endpoint=case.parent_endpoint_target.to(
            device=resolved_device,
            dtype=torch.float64,
        ),
        mask=case.target_causal_mask.to(device=resolved_device),
        query_cosine=case.query_rope_cos.to(device=resolved_device),
        query_sine=case.query_rope_sin.to(device=resolved_device),
        key_cosine=case.key_rope_cos.to(device=resolved_device),
        key_sine=case.key_rope_sin.to(device=resolved_device),
        query_row_weights=case.query_row_sqrt_weights.to(
            device=resolved_device,
            dtype=torch.float64,
        ).square(),
    )


def _prepare_cases_round_robin(
    cases: Sequence[Lfm2EndpointQkResidualCase],
    seed: Lfm2EndpointQkvoSeed,
    *,
    compute_devices: Sequence[Any],
) -> tuple[_PreparedCase, ...]:
    devices = tuple(compute_devices)
    if not devices:
        raise Lfm2EndpointQkResidualFoldError(
            "At least one endpoint compute device is required."
        )
    caches = {
        str(device): _build_fixed_device_tensor_cache(seed, device)
        for device in devices
    }
    return tuple(
        _prepare_case(
            case,
            seed,
            device=devices[index % len(devices)],
            assignment_index=index,
            fixed_cache=caches[str(devices[index % len(devices)])],
        )
        for index, case in enumerate(cases)
    )


def _exact_replay(
    case: _PreparedCase,
    seed: Lfm2EndpointQkvoSeed,
    query_weight: Any,
    key_weight: Any,
) -> _Replay:
    torch = _torch()
    case.exact_replay_calls += 1
    batch, query_length, _hidden = map(int, case.query_inputs.shape)
    key_length = int(case.key_inputs.shape[1])
    query = query_weight
    if query.device != case.query_inputs.device or query.dtype != torch.bfloat16:
        query = query.to(device=case.query_inputs.device, dtype=torch.bfloat16)
    key = key_weight
    if key.device != case.key_inputs.device or key.dtype != torch.bfloat16:
        key = key.to(device=case.key_inputs.device, dtype=torch.bfloat16)
    cache = case.fixed_cache
    value = cache.value_weight_bfloat16
    output = cache.output_weight_float64
    query_linear = (
        (case.query_inputs @ query.T)
        .reshape(
            batch,
            query_length,
            seed.query_heads,
            seed.head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key_linear = (
        (case.key_inputs @ key.T)
        .reshape(
            batch,
            key_length,
            seed.key_value_heads,
            seed.head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    value_heads = (
        (case.key_inputs @ value.T)
        .reshape(
            batch,
            key_length,
            seed.key_value_heads,
            seed.head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    query_post = _apply_rope(
        _exact_runtime_rmsnorm(
            query_linear,
            cache.query_norm_weight_bfloat16,
            seed.norm_epsilon,
        ),
        case.query_cosine,
        case.query_sine,
    )
    key_post = _apply_rope(
        _exact_runtime_rmsnorm(
            key_linear,
            cache.key_norm_weight_bfloat16,
            seed.norm_epsilon,
        ),
        case.key_cosine,
        case.key_sine,
    )
    repeated_key = key_post.repeat_interleave(
        seed.queries_per_key_value,
        dim=1,
    )
    repeated_value = value_heads.repeat_interleave(
        seed.queries_per_key_value,
        dim=1,
    ).to(torch.float64)
    scores = _center_scores(
        torch.matmul(query_post, repeated_key.transpose(-2, -1)).to(torch.float32)
        / math.sqrt(seed.head_dim),
        case.mask,
    )
    probabilities = _masked_softmax(scores, case.mask)
    context = torch.matmul(probabilities, repeated_value)
    endpoint = (
        context.permute(0, 2, 1, 3).reshape(
            batch, query_length, seed.query_heads * seed.head_dim
        )
        @ output.T
    )
    return _Replay(
        query_linear=query_linear,
        key_linear=key_linear,
        query_post=query_post,
        key_post=key_post,
        repeated_value=repeated_value,
        probabilities=probabilities,
        endpoint=endpoint,
    )


def replay_lfm2_endpoint_qkvo(
    case: Lfm2EndpointQkResidualCase,
    seed: Lfm2EndpointQkvoSeed,
    *,
    query_weight: Any | None = None,
    key_weight: Any | None = None,
    device: str = "cpu",
) -> tuple[Any, Any]:
    """Public cached-evidence replay used for diagnostics and synthetic proofs."""

    torch = _torch()
    resolved = torch.device(device)
    prepared = _prepare_case(case, seed, device=resolved)
    query, key = _broadcast_replay_qk(
        prepared.fixed_cache,
        seed.query_weight if query_weight is None else query_weight,
        seed.key_weight if key_weight is None else key_weight,
    )
    replay = _exact_replay(
        prepared,
        seed,
        query,
        key,
    )
    return (
        replay.probabilities.detach().cpu(),
        replay.endpoint.detach().cpu(),
    )


def _broadcast_replay_qk(
    cache: _FixedDeviceTensorCache,
    query_weight: Any,
    key_weight: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    query = query_weight
    if query.device != cache.device or query.dtype != torch.bfloat16:
        query = query.to(device=cache.device, dtype=torch.bfloat16)
    key = key_weight
    if key.device != cache.device or key.dtype != torch.bfloat16:
        key = key.to(device=cache.device, dtype=torch.bfloat16)
    cache.replay_qk_shard_broadcasts += 1
    return query, key


def _attach_replay(
    cases: Sequence[_PreparedCase],
    seed: Lfm2EndpointQkvoSeed,
    query_weight: Any,
    key_weight: Any,
) -> None:
    for shard in _static_device_shards(cases):
        query, key = _broadcast_replay_qk(
            shard.fixed_cache,
            query_weight,
            key_weight,
        )
        for case in shard.cases:
            replay = _exact_replay(case, seed, query, key)
            case.query_linear = replay.query_linear
            case.key_linear = replay.key_linear
            case.query_post = replay.query_post
            case.key_post = replay.key_post
            case.repeated_value = replay.repeated_value
            case.probabilities = replay.probabilities
            case.endpoint = replay.endpoint


def _score_jvp(
    case: _PreparedCase,
    seed: Lfm2EndpointQkvoSeed,
    delta_query: Any,
    delta_key: Any,
) -> Any:
    torch = _torch()
    assert case.query_linear is not None
    assert case.key_linear is not None
    assert case.query_post is not None
    assert case.key_post is not None
    if (
        delta_query.device != case.query_inputs.device
        or delta_query.dtype != torch.float32
        or delta_key.device != case.key_inputs.device
        or delta_key.dtype != torch.float32
    ):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Endpoint Q/K directions must be broadcast once to the case shard."
        )
    query_delta = delta_query.reshape(
        seed.query_heads,
        seed.head_dim,
        seed.hidden_size,
    )
    key_delta = delta_key.reshape(
        seed.key_value_heads,
        seed.head_dim,
        seed.hidden_size,
    )
    delta_query_linear = torch.einsum(
        "bqf,hdf->bhqd",
        case.query_inputs.to(torch.float32),
        query_delta,
    )
    delta_key_linear = torch.einsum(
        "bkf,gdf->bgkd",
        case.key_inputs.to(torch.float32),
        key_delta,
    )
    delta_query_post = _apply_rope(
        _rmsnorm_jvp(
            case.query_linear,
            delta_query_linear,
            case.fixed_cache.query_norm_weight_float32,
            seed.norm_epsilon,
        ),
        case.query_cosine.to(torch.float32),
        case.query_sine.to(torch.float32),
    )
    delta_key_post = _apply_rope(
        _rmsnorm_jvp(
            case.key_linear,
            delta_key_linear,
            case.fixed_cache.key_norm_weight_float32,
            seed.norm_epsilon,
        ),
        case.key_cosine.to(torch.float32),
        case.key_sine.to(torch.float32),
    )
    repeated_key = case.key_post.to(torch.float32).repeat_interleave(
        seed.queries_per_key_value,
        dim=1,
    )
    repeated_delta_key = delta_key_post.repeat_interleave(
        seed.queries_per_key_value,
        dim=1,
    )
    delta_scores = (
        torch.matmul(delta_query_post, repeated_key.transpose(-2, -1))
        + torch.matmul(
            case.query_post.to(torch.float32),
            repeated_delta_key.transpose(-2, -1),
        )
    ) / math.sqrt(seed.head_dim)
    return _center_scores(delta_scores, case.mask)


def _score_vjp(
    case: _PreparedCase,
    seed: Lfm2EndpointQkvoSeed,
    score_cotangent: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    assert case.query_linear is not None
    assert case.key_linear is not None
    assert case.query_post is not None
    assert case.key_post is not None
    cotangent = _center_scores(score_cotangent.to(torch.float32), case.mask)
    repeated_key = case.key_post.to(torch.float32).repeat_interleave(
        seed.queries_per_key_value,
        dim=1,
    )
    query_post = case.query_post.to(torch.float32)
    query_post_cotangent = torch.matmul(cotangent, repeated_key) / math.sqrt(
        seed.head_dim
    )
    repeated_key_cotangent = torch.matmul(
        cotangent.transpose(-2, -1),
        query_post,
    ) / math.sqrt(seed.head_dim)
    batch, _heads, key_length, _dim = repeated_key_cotangent.shape
    key_post_cotangent = repeated_key_cotangent.reshape(
        batch,
        seed.key_value_heads,
        seed.queries_per_key_value,
        key_length,
        seed.head_dim,
    ).sum(dim=2)
    query_pre_cotangent = _apply_rope_transpose(
        query_post_cotangent,
        case.query_cosine.to(torch.float32),
        case.query_sine.to(torch.float32),
    )
    key_pre_cotangent = _apply_rope_transpose(
        key_post_cotangent,
        case.key_cosine.to(torch.float32),
        case.key_sine.to(torch.float32),
    )
    query_linear_cotangent = _rmsnorm_vjp(
        case.query_linear,
        query_pre_cotangent,
        case.fixed_cache.query_norm_weight_float32,
        seed.norm_epsilon,
    )
    key_linear_cotangent = _rmsnorm_vjp(
        case.key_linear,
        key_pre_cotangent,
        case.fixed_cache.key_norm_weight_float32,
        seed.norm_epsilon,
    )
    query_gradient = torch.einsum(
        "bhqd,bqf->hdf",
        query_linear_cotangent,
        case.query_inputs.to(torch.float32),
    ).reshape(tuple(seed.query_weight.shape))
    key_gradient = torch.einsum(
        "bgkd,bkf->gdf",
        key_linear_cotangent,
        case.key_inputs.to(torch.float32),
    ).reshape(tuple(seed.key_weight.shape))
    return query_gradient, key_gradient


def _endpoint_jvp(
    case: _PreparedCase,
    seed: Lfm2EndpointQkvoSeed,
    delta_query: Any,
    delta_key: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    case.endpoint_jvp_calls += 1
    assert case.probabilities is not None
    assert case.repeated_value is not None
    delta_scores = _score_jvp(case, seed, delta_query, delta_key)
    delta_probabilities = _softmax_jvp(
        case.probabilities,
        delta_scores,
        case.mask,
    )
    delta_context = torch.matmul(
        delta_probabilities,
        case.repeated_value,
    )
    batch, _heads, query_length, _dim = delta_context.shape
    output = case.fixed_cache.output_weight_float64
    delta_endpoint = (
        delta_context.permute(0, 2, 1, 3).reshape(
            batch, query_length, seed.query_heads * seed.head_dim
        )
        @ output.T
    )
    return delta_probabilities, delta_endpoint


def _endpoint_vjp(
    case: _PreparedCase,
    seed: Lfm2EndpointQkvoSeed,
    *,
    endpoint_cotangent: Any,
    probability_cotangent: Any | None = None,
) -> tuple[Any, Any]:
    torch = _torch()
    case.endpoint_vjp_calls += 1
    assert case.probabilities is not None
    assert case.repeated_value is not None
    batch, query_length, output_size = endpoint_cotangent.shape
    if output_size != seed.output_size:
        raise Lfm2EndpointQkResidualFoldError(
            "Endpoint cotangent has the wrong output width."
        )
    output = case.fixed_cache.output_weight_float64
    context_cotangent = (
        (endpoint_cotangent.to(torch.float64) @ output)
        .reshape(
            batch,
            query_length,
            seed.query_heads,
            seed.head_dim,
        )
        .permute(0, 2, 1, 3)
    )
    route_cotangent = torch.matmul(
        context_cotangent,
        case.repeated_value.transpose(-2, -1),
    )
    if probability_cotangent is not None:
        route_cotangent = route_cotangent + probability_cotangent.to(torch.float64)
    score_cotangent = _softmax_vjp(
        case.probabilities,
        route_cotangent,
        case.mask,
    )
    return _score_vjp(case, seed, score_cotangent)


def _run_deterministic_semantic_normal_probe(
    case: _PreparedCase,
    seed: Lfm2EndpointQkvoSeed,
    *,
    delta_query: Any | None = None,
    delta_key: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    torch = _torch()
    device = case.query_inputs.device
    if delta_query is None:
        delta_query = torch.linspace(
            -0.5,
            0.5,
            int(seed.query_weight.numel()),
            device=device,
            dtype=torch.float32,
        ).reshape(tuple(seed.query_weight.shape))
    if delta_key is None:
        delta_key = torch.linspace(
            0.5,
            -0.5,
            int(seed.key_weight.numel()),
            device=device,
            dtype=torch.float32,
        ).reshape(tuple(seed.key_weight.shape))
    delta_probabilities, delta_endpoint = _endpoint_jvp(
        case,
        seed,
        delta_query,
        delta_key,
    )
    query_gradient, key_gradient = _endpoint_vjp(
        case,
        seed,
        endpoint_cotangent=delta_endpoint,
        probability_cotangent=delta_probabilities,
    )
    case.normal_operator_calls += 1
    case.semantic_probe_calls += 1
    witness = (
        delta_probabilities.square().sum()
        + delta_endpoint.square().sum()
        + query_gradient.to(torch.float64).square().sum()
        + key_gradient.to(torch.float64).square().sum()
        + delta_probabilities.new_tensor(
            delta_probabilities.numel()
            + delta_endpoint.numel()
            + query_gradient.numel()
            + key_gradient.numel(),
        )
    )
    return (
        delta_probabilities,
        delta_endpoint,
        query_gradient,
        key_gradient,
        witness,
    )


def _build_semantic_shard_directions(
    shard: _StaticDeviceShard,
    seed: Lfm2EndpointQkvoSeed,
) -> tuple[Any, Any]:
    torch = _torch()
    shard.fixed_cache.semantic_direction_shard_builds += 1
    return (
        torch.linspace(
            -0.5,
            0.5,
            int(seed.query_weight.numel()),
            device=shard.device,
            dtype=torch.float32,
        ).reshape(tuple(seed.query_weight.shape)),
        torch.linspace(
            0.5,
            -0.5,
            int(seed.key_weight.numel()),
            device=shard.device,
            dtype=torch.float32,
        ).reshape(tuple(seed.key_weight.shape)),
    )


class Lfm2EndpointQkLinearOperator:
    """Matrix-free endpoint-plus-route Gauss-Newton ridge operator."""

    def __init__(
        self,
        *,
        seed: Lfm2EndpointQkvoSeed,
        prepared_cases: Sequence[_PreparedCase],
        current_query_weight: Any,
        current_key_weight: Any,
        policy: Lfm2EndpointQkResidualFoldPolicy,
    ) -> None:
        torch = _torch()
        self.seed = seed
        self.cases = tuple(prepared_cases)
        self.shards = _static_device_shards(self.cases)
        self.policy = policy
        self.device = current_query_weight.device
        self.query_shape = tuple(seed.query_weight.shape)
        self.key_shape = tuple(seed.key_weight.shape)
        self.query_numel = int(seed.query_weight.numel())
        self.key_numel = int(seed.key_weight.numel())
        self.query_scale_squared = max(
            float(current_query_weight.to(torch.float64).square().sum().item()),
            self.query_numel * torch.finfo(torch.float32).eps,
        )
        self.key_scale_squared = max(
            float(current_key_weight.to(torch.float64).square().sum().item()),
            self.key_numel * torch.finfo(torch.float32).eps,
        )
        endpoint_energy = 0.0
        route_energy = 0.0
        for case in self.cases:
            assert case.endpoint is not None
            assert case.probabilities is not None
            endpoint_residual = case.target_endpoint - case.endpoint
            probability_residual = case.target_probabilities - case.probabilities
            endpoint_energy += float(
                (case.query_row_weights.unsqueeze(-1) * endpoint_residual.square())
                .sum()
                .item()
            )
            route_energy += float(
                (
                    case.query_row_weights[:, None, :, None]
                    * probability_residual.square()
                )
                .sum()
                .item()
            )
        tiny = torch.finfo(torch.float64).tiny
        self.endpoint_residual_energy = max(endpoint_energy, tiny)
        self.route_residual_energy = max(route_energy, tiny)
        self.normal_operator_calls = 0

    def pack(self, query: Any, key: Any) -> Any:
        return _torch().cat((query.reshape(-1), key.reshape(-1)))

    def unpack(self, vector: Any) -> tuple[Any, Any]:
        if int(vector.numel()) != self.query_numel + self.key_numel:
            raise Lfm2EndpointQkResidualFoldError(
                "Packed endpoint Q/K correction has the wrong parameter count."
            )
        return (
            vector[: self.query_numel].reshape(self.query_shape),
            vector[self.query_numel :].reshape(self.key_shape),
        )

    def jvp(self, vector: Any) -> tuple[tuple[Any, Any], ...]:
        delta_query, delta_key = self.unpack(vector)
        by_case: dict[int, tuple[Any, Any]] = {}
        for shard in self.shards:
            shard_query, shard_key = self._broadcast_direction(
                shard,
                delta_query,
                delta_key,
                count_normal=False,
            )
            for case in shard.cases:
                by_case[id(case)] = _endpoint_jvp(
                    case,
                    self.seed,
                    shard_query,
                    shard_key,
                )
        return tuple(by_case[id(case)] for case in self.cases)

    def _broadcast_direction(
        self,
        shard: _StaticDeviceShard,
        delta_query: Any,
        delta_key: Any,
        *,
        count_normal: bool,
    ) -> tuple[Any, Any]:
        torch = _torch()
        query = delta_query
        if query.device != shard.device or query.dtype != torch.float32:
            query = query.to(device=shard.device, dtype=torch.float32)
        key = delta_key
        if key.device != shard.device or key.dtype != torch.float32:
            key = key.to(device=shard.device, dtype=torch.float32)
        if count_normal:
            shard.fixed_cache.normal_qk_direction_shard_broadcasts += 1
        return query, key

    def _reduce_shard_gradients(
        self,
        cotangents: Mapping[int, tuple[Any, Any | None]],
        *,
        reduction_kind: Literal["rhs", "normal"],
    ) -> tuple[Any, Any]:
        torch = _torch()
        query = torch.zeros(self.query_shape, device=self.device, dtype=torch.float32)
        key = torch.zeros(self.key_shape, device=self.device, dtype=torch.float32)
        concurrent = _uses_concurrent_static_cuda_shards(self.shards)

        def _accumulate(shard: _StaticDeviceShard) -> tuple[Any, Any]:
            shard_query, shard_key = self._accumulate_shard_vjp(shard, cotangents)
            if concurrent:
                if reduction_kind == "rhs":
                    shard.fixed_cache.concurrent_rhs_shard_launches += 1
                else:
                    shard.fixed_cache.concurrent_normal_shard_launches += 1
            return shard_query, shard_key

        shard_gradients = _execute_static_device_shards(self.shards, _accumulate)
        for shard, (shard_query, shard_key) in zip(
            self.shards,
            shard_gradients,
            strict=True,
        ):
            if shard.device == self.device:
                query.add_(shard_query)
                key.add_(shard_key)
            else:
                query.add_(shard_query.to(device=self.device, dtype=torch.float32))
                key.add_(shard_key.to(device=self.device, dtype=torch.float32))
            if reduction_kind == "rhs":
                shard.fixed_cache.rhs_gradient_shard_reductions += 1
            else:
                shard.fixed_cache.normal_gradient_shard_reductions += 1
        return query, key

    def _accumulate_shard_vjp(
        self,
        shard: _StaticDeviceShard,
        cotangents: Mapping[int, tuple[Any, Any | None]],
    ) -> tuple[Any, Any]:
        torch = _torch()
        shard_query = torch.zeros(
            self.query_shape,
            device=shard.device,
            dtype=torch.float32,
        )
        shard_key = torch.zeros(
            self.key_shape,
            device=shard.device,
            dtype=torch.float32,
        )
        for case in shard.cases:
            endpoint_cotangent, probability_cotangent = cotangents[id(case)]
            query_delta, key_delta = _endpoint_vjp(
                case,
                self.seed,
                endpoint_cotangent=endpoint_cotangent,
                probability_cotangent=probability_cotangent,
            )
            if (
                query_delta.device != shard.device
                or query_delta.dtype != torch.float32
                or key_delta.device != shard.device
                or key_delta.dtype != torch.float32
            ):
                raise Lfm2EndpointQkResidualAuthenticationError(
                    "Endpoint VJP did not remain on its static device shard."
                )
            shard_query.add_(query_delta)
            shard_key.add_(key_delta)
        return shard_query, shard_key

    def vjp(
        self,
        cotangents: Sequence[tuple[Any, Any | None]],
    ) -> Any:
        values = tuple(cotangents)
        if len(values) != len(self.cases):
            raise Lfm2EndpointQkResidualFoldError(
                "Endpoint cotangents do not align with prepared FIT cases."
            )
        by_case = {
            id(case): cotangent
            for case, cotangent in zip(self.cases, values, strict=True)
        }
        query, key = self._reduce_shard_gradients(
            by_case,
            reduction_kind="rhs",
        )
        return self.pack(query, key)

    def right_hand_side(self) -> Any:
        cotangents = []
        for case in self.cases:
            assert case.endpoint is not None
            assert case.probabilities is not None
            endpoint_residual = case.target_endpoint - case.endpoint
            probability_residual = case.target_probabilities - case.probabilities
            endpoint_cotangent = (
                case.query_row_weights.unsqueeze(-1)
                * endpoint_residual
                / self.endpoint_residual_energy
            )
            probability_cotangent = (
                self.policy.route_objective_weight
                * case.query_row_weights[:, None, :, None]
                * probability_residual
                / self.route_residual_energy
            )
            cotangents.append((endpoint_cotangent, probability_cotangent))
        return self.vjp(cotangents)

    def normal(self, vector: Any) -> Any:
        torch = _torch()
        self.normal_operator_calls += 1
        delta_query, delta_key = self.unpack(vector)
        query = torch.zeros(self.query_shape, device=self.device, dtype=torch.float32)
        key = torch.zeros(self.key_shape, device=self.device, dtype=torch.float32)
        concurrent = _uses_concurrent_static_cuda_shards(self.shards)

        def _accumulate_normal(shard: _StaticDeviceShard) -> tuple[Any, Any]:
            shard_query, shard_key = self._broadcast_direction(
                shard,
                delta_query,
                delta_key,
                count_normal=True,
            )
            shard_query_gradient = torch.zeros(
                self.query_shape,
                device=shard.device,
                dtype=torch.float32,
            )
            shard_key_gradient = torch.zeros(
                self.key_shape,
                device=shard.device,
                dtype=torch.float32,
            )
            for case in shard.cases:
                case.normal_operator_calls += 1
                delta_probabilities, delta_endpoint = _endpoint_jvp(
                    case,
                    self.seed,
                    shard_query,
                    shard_key,
                )
                endpoint_cotangent = (
                    case.query_row_weights.unsqueeze(-1)
                    * delta_endpoint
                    / self.endpoint_residual_energy
                )
                probability_cotangent = (
                    self.policy.route_objective_weight
                    * case.query_row_weights[:, None, :, None]
                    * delta_probabilities
                    / self.route_residual_energy
                )
                query_delta, key_delta = _endpoint_vjp(
                    case,
                    self.seed,
                    endpoint_cotangent=endpoint_cotangent,
                    probability_cotangent=probability_cotangent,
                )
                shard_query_gradient.add_(query_delta)
                shard_key_gradient.add_(key_delta)
            if concurrent:
                shard.fixed_cache.concurrent_normal_shard_launches += 1
            return shard_query_gradient, shard_key_gradient

        shard_gradients = _execute_static_device_shards(
            self.shards,
            _accumulate_normal,
        )
        for shard, (shard_query_gradient, shard_key_gradient) in zip(
            self.shards,
            shard_gradients,
            strict=True,
        ):
            if shard.device == self.device:
                query.add_(shard_query_gradient)
                key.add_(shard_key_gradient)
            else:
                query.add_(
                    shard_query_gradient.to(device=self.device, dtype=torch.float32)
                )
                key.add_(shard_key_gradient.to(device=self.device, dtype=torch.float32))
            shard.fixed_cache.normal_gradient_shard_reductions += 1
        query.add_(
            delta_query,
            alpha=self.policy.ridge / self.query_scale_squared,
        )
        key.add_(
            delta_key,
            alpha=self.policy.ridge / self.key_scale_squared,
        )
        return self.pack(query, key)


def _dot(left: Any, right: Any) -> float:
    return float((left.to(_torch().float64) * right.to(_torch().float64)).sum().item())


_CG_USABLE_DIRECTION_RULE = (
    "finite_nonzero_solution_and_relative_residual_or_maximum_iterations"
)


def _cg_solve_report(
    solution: Any,
    *,
    converged: bool,
    iterations: int,
    relative_normal_residual: float,
    termination: str,
    normal_operator_calls: int,
) -> dict[str, Any]:
    torch = _torch()
    direction_is_finite = bool(torch.isfinite(solution).all().item())
    direction_has_nonzero_entry = bool(
        direction_is_finite and (solution != 0).any().item()
    )
    usable_direction = bool(
        direction_is_finite
        and direction_has_nonzero_entry
        and termination in {"relative_normal_residual", "maximum_iterations"}
    )
    return {
        "converged": converged,
        "usable_direction": usable_direction,
        "usable_direction_rule": _CG_USABLE_DIRECTION_RULE,
        "direction_is_finite": direction_is_finite,
        "direction_has_nonzero_entry": direction_has_nonzero_entry,
        "iterations": iterations,
        "relative_normal_residual": relative_normal_residual,
        "termination": termination,
        "normal_operator_calls": normal_operator_calls,
    }


def _ridge_cg(
    operator: Lfm2EndpointQkLinearOperator,
    right_hand_side: Any,
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    solution = torch.zeros_like(right_hand_side, dtype=torch.float32)
    right_norm = math.sqrt(max(_dot(right_hand_side, right_hand_side), 0.0))
    if not math.isfinite(right_norm):
        return solution, _cg_solve_report(
            solution,
            converged=False,
            iterations=0,
            relative_normal_residual=math.inf,
            termination="nonfinite_right_hand_side",
            normal_operator_calls=operator.normal_operator_calls,
        )
    if right_norm <= torch.finfo(torch.float64).tiny:
        return solution, _cg_solve_report(
            solution,
            converged=True,
            iterations=0,
            relative_normal_residual=0.0,
            termination="zero_endpoint_residual_gradient",
            normal_operator_calls=operator.normal_operator_calls,
        )
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_squared = _dot(residual, residual)
    relative = 1.0
    for iteration in range(1, policy.maximum_cg_iterations + 1):
        normal_direction = operator.normal(direction)
        curvature = _dot(direction, normal_direction)
        if not math.isfinite(curvature) or curvature <= 0.0:
            return solution, _cg_solve_report(
                solution,
                converged=False,
                iterations=iteration - 1,
                relative_normal_residual=relative,
                termination="nonpositive_or_nonfinite_curvature",
                normal_operator_calls=operator.normal_operator_calls,
            )
        step = residual_squared / curvature
        if not math.isfinite(step):
            return solution, _cg_solve_report(
                solution,
                converged=False,
                iterations=iteration - 1,
                relative_normal_residual=relative,
                termination="nonfinite_step",
                normal_operator_calls=operator.normal_operator_calls,
            )
        solution.add_(direction, alpha=step)
        if iteration % policy.cg_restart_iterations == 0:
            residual = right_hand_side - operator.normal(solution)
        else:
            residual.add_(normal_direction, alpha=-step)
        next_squared = max(_dot(residual, residual), 0.0)
        relative = math.sqrt(next_squared) / right_norm
        if (
            iteration >= policy.minimum_cg_iterations
            and relative <= policy.cg_relative_tolerance
        ):
            return solution, _cg_solve_report(
                solution,
                converged=True,
                iterations=iteration,
                relative_normal_residual=relative,
                termination="relative_normal_residual",
                normal_operator_calls=operator.normal_operator_calls,
            )
        if not math.isfinite(next_squared):
            return solution, _cg_solve_report(
                solution,
                converged=False,
                iterations=iteration,
                relative_normal_residual=math.inf,
                termination="nonfinite_residual",
                normal_operator_calls=operator.normal_operator_calls,
            )
        if iteration % policy.cg_restart_iterations == 0:
            direction = residual.clone()
        else:
            beta = next_squared / residual_squared
            direction.mul_(beta).add_(residual)
        residual_squared = next_squared
    return solution, _cg_solve_report(
        solution,
        converged=False,
        iterations=policy.maximum_cg_iterations,
        relative_normal_residual=relative,
        termination="maximum_iterations",
        normal_operator_calls=operator.normal_operator_calls,
    )


def _trust_scale(
    delta_query: Any,
    delta_key: Any,
    current_query: Any,
    current_key: Any,
    trust_radius: float,
) -> tuple[Any, Any, float, float, float]:
    torch = _torch()
    tiny = torch.finfo(torch.float64).tiny
    query_delta_norm = float(
        torch.linalg.vector_norm(delta_query.to(torch.float64)).item()
    )
    query_current_norm = float(
        torch.linalg.vector_norm(current_query.to(torch.float64)).item()
    )
    key_delta_norm = float(
        torch.linalg.vector_norm(delta_key.to(torch.float64)).item()
    )
    key_current_norm = float(
        torch.linalg.vector_norm(current_key.to(torch.float64)).item()
    )

    # A relative trust region has no meaningful incumbent scale when an exact
    # compiler blank reaches this solver.  Dividing a real analytic direction
    # by float64.tiny made the trust multiplier itself float64.tiny, so the
    # BF16 frontier silently rounded every cold-start correction back to zero.
    # Use the direction as the unit reference only for the exact-zero branch.
    # Every nonzero incumbent follows the original calculation byte-for-byte.
    query_relative = (
        0.0
        if query_current_norm == 0.0 and query_delta_norm == 0.0
        else (
            1.0
            if query_current_norm == 0.0
            else query_delta_norm / max(query_current_norm, tiny)
        )
    )
    key_relative = (
        0.0
        if key_current_norm == 0.0 and key_delta_norm == 0.0
        else (
            1.0
            if key_current_norm == 0.0
            else key_delta_norm / max(key_current_norm, tiny)
        )
    )
    scale = min(
        1.0,
        trust_radius / max(query_relative, tiny),
        trust_radius / max(key_relative, tiny),
    )
    return (
        delta_query * scale,
        delta_key * scale,
        query_relative,
        key_relative,
        scale,
    )


@dataclass(slots=True)
class _MetricAccumulator:
    case_count: int = 0
    endpoint_error: float = 0.0
    endpoint_target_energy: float = 0.0
    probability_error: float = 0.0
    probability_target_energy: float = 0.0
    js_sum: float = 0.0
    js_weight: float = 0.0
    js_worst: float = 0.0
    top1_matches: float = 0.0
    eligible_weight: float = 0.0
    eligible_rows: int = 0
    leakage: float = 0.0

    @staticmethod
    def _row_routing_metrics(
        prediction_probabilities: Any,
        target_probabilities: Any,
        mask: Any,
        query_row_weights: Any,
    ) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[float, ...]]:
        """Evaluate informative rows without synchronizing once per scalar.

        Grouping rows by their exact valid-key count preserves the operands and
        reduction extent of every former scalar FP64 row reduction.  Results
        are then scattered back into the original ``batch, head, query`` order
        before one batched host transfer, so the caller retains the original
        Python accumulation order and weighted metric definitions.
        """

        torch = _torch()
        batch, heads, query_length, key_width = map(
            int,
            prediction_probabilities.shape,
        )
        flat_prediction = prediction_probabilities.reshape(-1, key_width)
        flat_target = target_probabilities.reshape_as(flat_prediction)
        flat_mask = mask.reshape_as(flat_prediction)
        flat_row_weights = (
            query_row_weights[:, None, :].expand(batch, heads, query_length).reshape(-1)
        )
        valid_counts = flat_mask.sum(dim=-1)
        count_histogram = tuple(
            int(value)
            for value in torch.bincount(
                valid_counts,
                minlength=key_width + 1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        eligible_rows = sum(count_histogram[2:])
        if eligible_rows == 0:
            return (), (), ()

        eligible_indices = torch.nonzero(valid_counts > 1, as_tuple=False).flatten()
        eligible_counts = valid_counts.index_select(0, eligible_indices)
        count_order = torch.argsort(eligible_counts, stable=True)
        grouped_indices = eligible_indices.index_select(0, count_order)
        grouped_js = torch.empty(
            (eligible_rows,),
            device=flat_prediction.device,
            dtype=torch.float64,
        )
        grouped_top1 = torch.empty(
            (eligible_rows,),
            device=flat_prediction.device,
            dtype=torch.bool,
        )
        tiny = torch.finfo(torch.float64).tiny
        cursor = 0
        for valid_count in range(2, key_width + 1):
            group_size = count_histogram[valid_count]
            if group_size == 0:
                continue
            row_indices = grouped_indices.narrow(0, cursor, group_size)
            group_mask = flat_mask.index_select(0, row_indices)
            prediction = (
                flat_prediction.index_select(0, row_indices)
                .masked_select(group_mask)
                .reshape(group_size, valid_count)
            )
            target = (
                flat_target.index_select(0, row_indices)
                .masked_select(group_mask)
                .reshape(group_size, valid_count)
            )
            midpoint = 0.5 * (prediction + target)
            js = (
                0.5
                * (
                    (
                        target
                        * (
                            torch.log(target.clamp_min(tiny))
                            - torch.log(midpoint.clamp_min(tiny))
                        )
                    ).sum(dim=-1)
                    + (
                        prediction
                        * (
                            torch.log(prediction.clamp_min(tiny))
                            - torch.log(midpoint.clamp_min(tiny))
                        )
                    ).sum(dim=-1)
                )
                / math.log(2.0)
            )
            grouped_js.narrow(0, cursor, group_size).copy_(js.clamp_min(0.0))
            target_winners = torch.isclose(
                target,
                target.max(dim=-1, keepdim=True).values,
                rtol=1e-10,
                atol=1e-12,
            )
            grouped_top1.narrow(0, cursor, group_size).copy_(
                target_winners.gather(
                    dim=-1,
                    index=prediction.argmax(dim=-1, keepdim=True),
                ).squeeze(-1)
            )
            cursor += group_size

        js_by_row = torch.empty_like(grouped_js)
        top1_by_row = torch.empty_like(grouped_top1)
        js_by_row.scatter_(0, count_order, grouped_js)
        top1_by_row.scatter_(0, count_order, grouped_top1)
        weights_by_row = flat_row_weights.index_select(0, eligible_indices)
        return (
            tuple(float(value) for value in js_by_row.detach().cpu().tolist()),
            tuple(bool(value) for value in top1_by_row.detach().cpu().tolist()),
            tuple(float(value) for value in weights_by_row.detach().cpu().tolist()),
        )

    def add(self, case: _PreparedCase, replay: _Replay) -> None:
        torch = _torch()
        self.case_count += 1
        row_weights = case.query_row_weights.to(torch.float64)
        endpoint_delta = replay.endpoint - case.target_endpoint
        self.endpoint_error += float(
            (row_weights.unsqueeze(-1) * endpoint_delta.square()).sum().item()
        )
        self.endpoint_target_energy += float(
            (row_weights.unsqueeze(-1) * case.target_endpoint.square()).sum().item()
        )
        probability_delta = replay.probabilities - case.target_probabilities
        probability_weights = row_weights[:, None, :, None]
        self.probability_error += float(
            (probability_weights * probability_delta.square()).sum().item()
        )
        self.probability_target_energy += float(
            (probability_weights * case.target_probabilities.square()).sum().item()
        )
        if bool((~case.mask).any().item()):
            self.leakage = max(
                self.leakage,
                float(replay.probabilities[~case.mask].abs().max().item()),
            )
        js_values, top1_matches, routing_weights = self._row_routing_metrics(
            replay.probabilities,
            case.target_probabilities,
            case.mask,
            row_weights,
        )
        for js_value, matched, weight in zip(
            js_values,
            top1_matches,
            routing_weights,
            strict=True,
        ):
            self.js_sum += weight * js_value
            self.js_weight += weight
            self.js_worst = max(self.js_worst, js_value)
            self.top1_matches += weight * float(matched)
            self.eligible_weight += weight
            self.eligible_rows += 1

    def finish(self) -> Lfm2EndpointQkMetrics:
        torch = _torch()
        if self.eligible_rows <= 0 or self.eligible_weight <= 0.0:
            raise Lfm2EndpointQkResidualFoldError(
                "Endpoint evidence contains no informative causal route row."
            )
        tiny = torch.finfo(torch.float64).tiny
        return Lfm2EndpointQkMetrics(
            case_count=self.case_count,
            eligible_route_rows=self.eligible_rows,
            endpoint_relative_rmse=math.sqrt(
                self.endpoint_error / max(self.endpoint_target_energy, tiny)
            ),
            probability_relative_rmse=math.sqrt(
                self.probability_error / max(self.probability_target_energy, tiny)
            ),
            js_divergence_mean=self.js_sum / max(self.js_weight, tiny),
            js_divergence_worst=self.js_worst,
            top1_route_agreement_fraction=(self.top1_matches / self.eligible_weight),
            maximum_causal_probability_leakage=self.leakage,
        )


def _evaluate(
    cases: Sequence[_PreparedCase],
    seed: Lfm2EndpointQkvoSeed,
    query_weight: Any,
    key_weight: Any,
) -> Lfm2EndpointQkMetrics:
    accumulator = _MetricAccumulator()
    for shard in _static_device_shards(cases):
        query, key = _broadcast_replay_qk(
            shard.fixed_cache,
            query_weight,
            key_weight,
        )
        for case in shard.cases:
            accumulator.add(
                case,
                _exact_replay(case, seed, query, key),
            )
    return accumulator.finish()


def evaluate_lfm2_endpoint_qkvo_bank(
    bank: Lfm2EndpointQkResidualEvidenceBank,
    seed: Lfm2EndpointQkvoSeed,
    *,
    query_weight: Any | None = None,
    key_weight: Any | None = None,
    compute_devices: Sequence[str] | None = None,
) -> Lfm2EndpointQkMetrics:
    """Evaluate one sealed bank by cached analytical replay only."""

    torch = _torch()
    if not isinstance(bank, Lfm2EndpointQkResidualEvidenceBank):
        raise TypeError("bank must be Lfm2EndpointQkResidualEvidenceBank.")
    if not isinstance(seed, Lfm2EndpointQkvoSeed):
        raise TypeError("seed must be Lfm2EndpointQkvoSeed.")
    bank.verify()
    seed.verify()
    if bank.source_artifact_sha256 != seed.source_artifact_sha256:
        raise Lfm2EndpointQkResidualAuthenticationError(
            "Evaluation bank and Q/K/V/O seed name different source artifacts."
        )
    raw_devices = tuple(compute_devices or ("cpu",))
    try:
        devices = tuple(torch.device(value) for value in raw_devices)
    except (RuntimeError, ValueError) as exc:
        raise Lfm2EndpointQkResidualFoldError(
            "Evaluation compute_devices contains an invalid device."
        ) from exc
    requested_cuda = tuple(value for value in devices if value.type == "cuda")
    if requested_cuda and not torch.cuda.is_available():
        raise Lfm2EndpointQkResidualFoldError(
            "Requested CUDA evaluation devices are unavailable."
        )
    if any(
        value.index is None
        or value.index < 0
        or value.index >= torch.cuda.device_count()
        for value in requested_cuda
    ):
        raise Lfm2EndpointQkResidualFoldError(
            "A requested CUDA evaluation device index is unavailable."
        )
    prepared = _prepare_cases_round_robin(
        bank.cases,
        seed,
        compute_devices=devices,
    )
    return _evaluate(
        prepared,
        seed,
        seed.query_weight if query_weight is None else query_weight,
        seed.key_weight if key_weight is None else key_weight,
    )


def _route_nonregressing(
    candidate: Lfm2EndpointQkMetrics,
    baseline: Lfm2EndpointQkMetrics,
    tolerance: float,
) -> bool:
    return (
        candidate.probability_relative_rmse
        <= baseline.probability_relative_rmse + tolerance
        and candidate.js_divergence_mean <= baseline.js_divergence_mean + tolerance
        and candidate.js_divergence_worst <= baseline.js_divergence_worst + tolerance
        and candidate.top1_route_agreement_fraction + tolerance
        >= baseline.top1_route_agreement_fraction
        and candidate.maximum_causal_probability_leakage == 0.0
    )


def _all_nonregressing(
    candidate: Lfm2EndpointQkMetrics,
    baseline: Lfm2EndpointQkMetrics,
    tolerance: float,
) -> bool:
    return (
        candidate.endpoint_relative_rmse <= baseline.endpoint_relative_rmse + tolerance
        and _route_nonregressing(candidate, baseline, tolerance)
    )


@dataclass(frozen=True, slots=True)
class _FitNumericalResult:
    query_weight: Any
    key_weight: Any
    initial_metrics: Lfm2EndpointQkMetrics
    proposed_metrics: Lfm2EndpointQkMetrics
    sweeps: tuple[Mapping[str, Any], ...]
    changed: bool


def _compile_fit_numerics(
    *,
    seed: Lfm2EndpointQkvoSeed,
    fit_cases: Sequence[_PreparedCase],
    solver_device: Any,
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> _FitNumericalResult:
    torch = _torch()
    current_query = seed.query_weight.to(device=solver_device).clone()
    current_key = seed.key_weight.to(device=solver_device).clone()
    initial_metrics = _evaluate(
        fit_cases,
        seed,
        current_query,
        current_key,
    )
    current_metrics = initial_metrics
    sweeps: list[dict[str, Any]] = []
    changed = False
    with torch.no_grad():
        for sweep in range(policy.maximum_sweeps):
            _attach_replay(
                fit_cases,
                seed,
                current_query,
                current_key,
            )
            operator = Lfm2EndpointQkLinearOperator(
                seed=seed,
                prepared_cases=fit_cases,
                current_query_weight=current_query.to(torch.float32),
                current_key_weight=current_key.to(torch.float32),
                policy=policy,
            )
            direction, solve_report = _ridge_cg(
                operator,
                operator.right_hand_side(),
                policy,
            )
            sweep_report: dict[str, Any] = {
                "sweep": sweep,
                "linear_solver": "deterministic_matrix_free_ridge_normal_cg",
                "solve_report": solve_report,
                "endpoint_residual_energy": operator.endpoint_residual_energy,
                "route_residual_energy": operator.route_residual_energy,
            }
            if not bool(solve_report["usable_direction"]):
                sweeps.append(sweep_report)
                break
            delta_query, delta_key = operator.unpack(direction)
            (
                delta_query,
                delta_key,
                query_relative,
                key_relative,
                trust_scale,
            ) = _trust_scale(
                delta_query,
                delta_key,
                current_query.to(torch.float32),
                current_key.to(torch.float32),
                policy.trust_radius,
            )
            frontier = []
            selected: (
                tuple[
                    tuple[float, float, float, float, float, str],
                    float,
                    Any,
                    Any,
                    Lfm2EndpointQkMetrics,
                ]
                | None
            ) = None
            for alpha in policy.alpha_frontier:
                candidate_query = (
                    current_query.to(torch.float32) + alpha * delta_query
                ).to(torch.bfloat16)
                candidate_key = (current_key.to(torch.float32) + alpha * delta_key).to(
                    torch.bfloat16
                )
                metrics = _evaluate(
                    fit_cases,
                    seed,
                    candidate_query,
                    candidate_key,
                )
                state_sha = _state_sha256(
                    {
                        "q_proj.weight": candidate_query.detach().cpu(),
                        "k_proj.weight": candidate_key.detach().cpu(),
                    }
                )
                eligible = alpha == 0.0 or (
                    metrics.endpoint_relative_rmse
                    <= current_metrics.endpoint_relative_rmse
                    - policy.minimum_fit_endpoint_improvement
                    and _route_nonregressing(
                        metrics,
                        current_metrics,
                        policy.metric_regression_tolerance,
                    )
                )
                frontier.append(
                    {
                        "alpha": alpha,
                        "exact_bfloat16_state": True,
                        "eligible": eligible,
                        "state_sha256": state_sha,
                        "metrics": metrics.as_dict(),
                    }
                )
                if eligible:
                    selection_key = (
                        metrics.endpoint_relative_rmse,
                        metrics.probability_relative_rmse,
                        metrics.js_divergence_worst,
                        -metrics.top1_route_agreement_fraction,
                        alpha,
                        state_sha,
                    )
                    candidate = (
                        selection_key,
                        alpha,
                        candidate_query,
                        candidate_key,
                        metrics,
                    )
                    if selected is None or candidate[0] < selected[0]:
                        selected = candidate
            assert selected is not None
            (
                _selection_key,
                selected_alpha,
                selected_query,
                selected_key,
                selected_metrics,
            ) = selected
            sweep_report.update(
                {
                    "unscaled_query_relative_delta": query_relative,
                    "unscaled_key_relative_delta": key_relative,
                    "trust_scale": trust_scale,
                    "frontier": frontier,
                    "selected_alpha": selected_alpha,
                }
            )
            sweeps.append(sweep_report)
            same_state = torch.equal(
                selected_query,
                current_query,
            ) and torch.equal(selected_key, current_key)
            if selected_alpha == 0.0 or same_state:
                break
            current_query = selected_query
            current_key = selected_key
            current_metrics = selected_metrics
            changed = True
    return _FitNumericalResult(
        query_weight=current_query.detach().cpu(),
        key_weight=current_key.detach().cpu(),
        initial_metrics=initial_metrics,
        proposed_metrics=current_metrics,
        sweeps=tuple(MappingProxyType(value) for value in sweeps),
        changed=changed,
    )


def _metrics_from_sweep_report(
    value: Any,
    name: str,
) -> Lfm2EndpointQkMetrics:
    expected = set(Lfm2EndpointQkMetrics.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Lfm2EndpointQkResidualAuthenticationError(
            f"{name} does not contain one exact endpoint metric record."
        )
    try:
        return Lfm2EndpointQkMetrics(**dict(value))
    except (TypeError, ValueError) as exc:
        raise Lfm2EndpointQkResidualAuthenticationError(
            f"{name} contains invalid endpoint metrics."
        ) from exc


def _verify_fit_sweep_semantics(
    sweeps: Any,
    *,
    seed: Lfm2EndpointQkvoSeed,
    query_weight: Any,
    key_weight: Any,
    initial_metrics: Lfm2EndpointQkMetrics,
    proposed_metrics: Lfm2EndpointQkMetrics,
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> None:
    if (
        not isinstance(sweeps, list)
        or not sweeps
        or len(sweeps) > policy.maximum_sweeps
    ):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "FIT sweep inventory is empty, unbounded, or not a list."
        )
    solve_fields = {
        "converged",
        "usable_direction",
        "usable_direction_rule",
        "direction_is_finite",
        "direction_has_nonzero_entry",
        "iterations",
        "relative_normal_residual",
        "termination",
        "normal_operator_calls",
    }
    base_sweep_fields = {
        "sweep",
        "linear_solver",
        "solve_report",
        "endpoint_residual_energy",
        "route_residual_energy",
    }
    usable_sweep_fields = {
        *base_sweep_fields,
        "unscaled_query_relative_delta",
        "unscaled_key_relative_delta",
        "trust_scale",
        "frontier",
        "selected_alpha",
    }
    frontier_fields = {
        "alpha",
        "exact_bfloat16_state",
        "eligible",
        "state_sha256",
        "metrics",
    }
    current_metrics = initial_metrics
    current_state_sha = _state_sha256(
        {
            "q_proj.weight": seed.query_weight,
            "k_proj.weight": seed.key_weight,
        }
    )
    for sweep_index, sweep in enumerate(sweeps):
        if (
            not isinstance(sweep, Mapping)
            or sweep.get("sweep") != sweep_index
            or sweep.get("linear_solver") != "deterministic_matrix_free_ridge_normal_cg"
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweep ordering or solver identity is invalid."
            )
        solve = sweep.get("solve_report")
        if not isinstance(solve, Mapping) or set(solve) != solve_fields:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweep solve report does not have the exact authenticated schema."
            )
        if (
            not all(
                isinstance(solve.get(name), bool)
                for name in (
                    "converged",
                    "usable_direction",
                    "direction_is_finite",
                    "direction_has_nonzero_entry",
                )
            )
            or solve.get("usable_direction_rule") != _CG_USABLE_DIRECTION_RULE
            or isinstance(solve.get("iterations"), bool)
            or not isinstance(solve.get("iterations"), int)
            or not 0 <= int(solve["iterations"]) <= policy.maximum_cg_iterations
            or isinstance(solve.get("normal_operator_calls"), bool)
            or not isinstance(solve.get("normal_operator_calls"), int)
            or int(solve["normal_operator_calls"]) < 0
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweep usable-direction fields are invalid."
            )
        relative = solve.get("relative_normal_residual")
        if (
            isinstance(relative, bool)
            or not isinstance(relative, (float, int))
            or not math.isfinite(float(relative))
            or float(relative) < 0.0
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweep relative normal residual is invalid."
            )
        termination = solve.get("termination")
        converged = bool(solve["converged"])
        iterations = int(solve["iterations"])
        if termination == "relative_normal_residual":
            valid_termination = (
                converged
                and policy.minimum_cg_iterations
                <= iterations
                <= policy.maximum_cg_iterations
                and float(relative) <= policy.cg_relative_tolerance
            )
        elif termination == "zero_endpoint_residual_gradient":
            valid_termination = converged and iterations == 0 and float(relative) == 0.0
        elif termination == "maximum_iterations":
            valid_termination = (
                not converged and iterations == policy.maximum_cg_iterations
            )
        elif termination in {
            "nonfinite_right_hand_side",
            "nonpositive_or_nonfinite_curvature",
            "nonfinite_step",
            "nonfinite_residual",
        }:
            valid_termination = not converged
        else:
            valid_termination = False
        expected_usable = bool(
            solve["direction_is_finite"]
            and solve["direction_has_nonzero_entry"]
            and termination in {"relative_normal_residual", "maximum_iterations"}
        )
        if not valid_termination or bool(solve["usable_direction"]) != expected_usable:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweep convergence and usable-direction semantics disagree."
            )
        for energy_name in ("endpoint_residual_energy", "route_residual_energy"):
            energy = sweep.get(energy_name)
            if (
                isinstance(energy, bool)
                or not isinstance(energy, (float, int))
                or not math.isfinite(float(energy))
                or float(energy) <= 0.0
            ):
                raise Lfm2EndpointQkResidualAuthenticationError(
                    f"FIT sweep {energy_name} is invalid."
                )
        if not expected_usable:
            if set(sweep) != base_sweep_fields or sweep_index != len(sweeps) - 1:
                raise Lfm2EndpointQkResidualAuthenticationError(
                    "An unusable FIT direction exposed a frontier or did not stop."
                )
            continue
        if set(sweep) != usable_sweep_fields:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "A usable FIT direction lacks its exact BF16 frontier."
            )
        for name in (
            "unscaled_query_relative_delta",
            "unscaled_key_relative_delta",
            "trust_scale",
        ):
            value = sweep.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise Lfm2EndpointQkResidualAuthenticationError(
                    f"FIT sweep {name} is invalid."
                )
        if float(sweep["trust_scale"]) > 1.0:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweep trust scaling exceeds one."
            )
        frontier = sweep.get("frontier")
        if not isinstance(frontier, list) or len(frontier) != len(
            policy.alpha_frontier
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweep frontier does not match the authenticated alpha policy."
            )
        eligible_rows = []
        for frontier_index, (row, expected_alpha) in enumerate(
            zip(frontier, policy.alpha_frontier, strict=True)
        ):
            if (
                not isinstance(row, Mapping)
                or set(row) != frontier_fields
                or row.get("alpha") != expected_alpha
                or row.get("exact_bfloat16_state") is not True
                or not isinstance(row.get("eligible"), bool)
            ):
                raise Lfm2EndpointQkResidualAuthenticationError(
                    "FIT sweep frontier row schema or alpha order is invalid."
                )
            state_sha = _required_sha256(
                row.get("state_sha256"),
                f"sweeps[{sweep_index}].frontier[{frontier_index}].state_sha256",
            )
            metrics = _metrics_from_sweep_report(
                row.get("metrics"),
                f"sweeps[{sweep_index}].frontier[{frontier_index}].metrics",
            )
            expected_eligible = expected_alpha == 0.0 or (
                metrics.endpoint_relative_rmse
                <= current_metrics.endpoint_relative_rmse
                - policy.minimum_fit_endpoint_improvement
                and _route_nonregressing(
                    metrics,
                    current_metrics,
                    policy.metric_regression_tolerance,
                )
            )
            if bool(row["eligible"]) != expected_eligible:
                raise Lfm2EndpointQkResidualAuthenticationError(
                    "FIT sweep frontier eligibility bypasses the exact metric gate."
                )
            if expected_alpha == 0.0 and (
                metrics.as_dict() != current_metrics.as_dict()
                or state_sha != current_state_sha
            ):
                raise Lfm2EndpointQkResidualAuthenticationError(
                    "FIT sweep alpha-zero row is not the current dense state."
                )
            selection_key = (
                metrics.endpoint_relative_rmse,
                metrics.probability_relative_rmse,
                metrics.js_divergence_worst,
                -metrics.top1_route_agreement_fraction,
                expected_alpha,
                state_sha,
            )
            if expected_eligible:
                eligible_rows.append(
                    (selection_key, expected_alpha, state_sha, metrics)
                )
        if not eligible_rows:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweep frontier has no alpha-zero fallback."
            )
        _selection, selected_alpha, selected_state_sha, selected_metrics = min(
            eligible_rows,
            key=lambda value: value[0],
        )
        if sweep.get("selected_alpha") != selected_alpha:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweep did not select the exact best eligible BF16 state."
            )
        current_metrics = selected_metrics
        previous_state_sha = current_state_sha
        current_state_sha = selected_state_sha
        if (
            selected_alpha == 0.0 or selected_state_sha == previous_state_sha
        ) and sweep_index != len(sweeps) - 1:
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT sweeps continued after the exact frontier stopped changing."
            )
    proposed_state_sha = _state_sha256(
        {
            "q_proj.weight": query_weight,
            "k_proj.weight": key_weight,
        }
    )
    if (
        current_metrics.as_dict() != proposed_metrics.as_dict()
        or current_state_sha != proposed_state_sha
    ):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "FIT sweep frontier does not terminate at the published state/metrics."
        )


@dataclass(frozen=True, slots=True)
class Lfm2CompiledEndpointQkFitProposal:
    """FIT-only Q/K proposal; no FIT-CHECK interface or acceptance decision."""

    query_weight: Any
    key_weight: Any
    seed: Lfm2EndpointQkvoSeed
    policy: Lfm2EndpointQkResidualFoldPolicy
    fit_bank: Lfm2EndpointQkResidualEvidenceBank
    changed: bool
    initial_fit_metrics: Lfm2EndpointQkMetrics
    proposed_fit_metrics: Lfm2EndpointQkMetrics
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        query = _dense_cpu_tensor(
            self.query_weight,
            "query_weight",
            ndim=2,
            dtype=torch.bfloat16,
        )
        key = _dense_cpu_tensor(
            self.key_weight,
            "key_weight",
            ndim=2,
            dtype=torch.bfloat16,
        )
        if (
            not isinstance(self.policy, Lfm2EndpointQkResidualFoldPolicy)
            or not self.policy.verify_authentication()
            or not isinstance(self.fit_bank, Lfm2EndpointQkResidualEvidenceBank)
            or self.fit_bank.partition != "fit"
            or self.fit_bank.source_artifact_sha256 != self.seed.source_artifact_sha256
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT-only endpoint proposal does not bind its policy/evidence."
            )
        fit_sha = self.fit_bank.verify()
        body = dict(self.report)
        supplied = body.pop("report_sha256", None)
        sweeps = body.get("sweeps")
        expected_fields = {
            "schema",
            "method",
            "seed_sha256",
            "source_inventory_sha256",
            "fit_evidence_sha256",
            "source_split_sha256",
            "source_artifact_sha256",
            "fit_target_receipt_sha256s",
            "policy",
            "policy_sha256",
            "query_weight_sha256",
            "key_weight_sha256",
            "fixed_value_weight_sha256",
            "fixed_output_weight_sha256",
            "changed",
            "initial_fit_metrics",
            "proposed_fit_metrics",
            "sweeps",
            "analytical_sweep_count",
            "solver_device",
            "compute_devices",
            "case_assignment",
            "observed_execution_receipt",
            "fit_check_accessible",
            "fit_check_cases_consumed",
            "exact_bfloat16_frontier",
            "direct_parent_probability_barycenter_target",
            "direct_parent_endpoint_target",
            "operation_audit",
        }
        expected_operation_audit = {
            **_ZERO_OPERATION_AUDIT,
            "fit_cases_consumed": len(self.fit_bank.cases),
            "fit_check_cases_consumed": 0,
            "matrix_free_ridge_normal_solves": (
                len(sweeps) if isinstance(sweeps, list) else -1
            ),
        }
        if (
            tuple(query.shape) != tuple(self.seed.query_weight.shape)
            or tuple(key.shape) != tuple(self.seed.key_weight.shape)
            or supplied != _canonical_sha256(body)
            or set(body) != expected_fields
            or body.get("schema") != LFM2_ENDPOINT_QK_FIT_PROPOSAL_SCHEMA
            or body.get("method")
            != (
                "fit_only_fixed_vo_endpoint_probability_gauss_newton_"
                "ridge_bfloat16_frontier"
            )
            or body.get("seed_sha256") != self.seed.verify()
            or body.get("source_inventory_sha256")
            != self.seed.source_inventory.inventory_sha256
            or body.get("policy") != self.policy.as_dict()
            or body.get("policy_sha256") != self.policy.authentication_sha256
            or body.get("fit_evidence_sha256") != fit_sha
            or body.get("source_split_sha256") != self.fit_bank.source_split_sha256
            or body.get("source_artifact_sha256")
            != self.fit_bank.source_artifact_sha256
            or body.get("fit_target_receipt_sha256s")
            != list(self.fit_bank.target_receipt_sha256s)
            or body.get("query_weight_sha256") != _tensor_sha256(query)
            or body.get("key_weight_sha256") != _tensor_sha256(key)
            or body.get("fixed_value_weight_sha256")
            != _tensor_sha256(self.seed.value_weight)
            or body.get("fixed_output_weight_sha256")
            != _tensor_sha256(self.seed.output_weight)
            or body.get("changed") != self.changed
            or body.get("initial_fit_metrics") != self.initial_fit_metrics.as_dict()
            or body.get("proposed_fit_metrics") != self.proposed_fit_metrics.as_dict()
            or not isinstance(sweeps, list)
            or body.get("analytical_sweep_count") != len(sweeps)
            or body.get("solver_device") != self.policy.solver_device
            or body.get("compute_devices") != list(self.policy.compute_devices)
            or body.get("case_assignment") != "authenticated_bank_order_round_robin"
            or body.get("fit_check_cases_consumed") != 0
            or body.get("fit_check_accessible") is not False
            or body.get("exact_bfloat16_frontier") is not True
            or body.get("direct_parent_probability_barycenter_target") is not True
            or body.get("direct_parent_endpoint_target") is not True
            or body.get("operation_audit") != expected_operation_audit
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT-only endpoint Q/K proposal failed authentication."
            )
        _verify_fit_sweep_semantics(
            sweeps,
            seed=self.seed,
            query_weight=query,
            key_weight=key,
            initial_metrics=self.initial_fit_metrics,
            proposed_metrics=self.proposed_fit_metrics,
            policy=self.policy,
        )
        _verify_observed_execution_receipt(
            body.get("observed_execution_receipt"),
            seed=self.seed,
            fit_bank=self.fit_bank,
            query_weight=query,
            key_weight=key,
            policy=self.policy,
            sweeps=sweeps,
        )
        recomputed_initial = evaluate_lfm2_endpoint_qkvo_bank(
            self.fit_bank,
            self.seed,
            compute_devices=self.policy.compute_devices,
        )
        recomputed_proposed = evaluate_lfm2_endpoint_qkvo_bank(
            self.fit_bank,
            self.seed,
            query_weight=query,
            key_weight=key,
            compute_devices=self.policy.compute_devices,
        )
        recomputed_changed = not (
            torch.equal(query, self.seed.query_weight)
            and torch.equal(key, self.seed.key_weight)
        )
        if (
            recomputed_initial.as_dict() != self.initial_fit_metrics.as_dict()
            or recomputed_proposed.as_dict() != self.proposed_fit_metrics.as_dict()
            or recomputed_changed != self.changed
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "FIT-only endpoint proposal metrics do not replay from its "
                "sealed evidence and dense state."
            )
        object.__setattr__(self, "query_weight", query)
        object.__setattr__(self, "key_weight", key)
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "report_sha256": supplied}),
        )

    @property
    def report_sha256(self) -> str:
        return str(self.report["report_sha256"])

    def verify(self) -> str:
        rebuilt = Lfm2CompiledEndpointQkFitProposal(
            query_weight=self.query_weight,
            key_weight=self.key_weight,
            seed=self.seed,
            policy=self.policy,
            fit_bank=self.fit_bank,
            changed=self.changed,
            initial_fit_metrics=self.initial_fit_metrics,
            proposed_fit_metrics=self.proposed_fit_metrics,
            report=self.report,
        )
        return rebuilt.report_sha256


@dataclass(frozen=True, slots=True)
class Lfm2CompiledEndpointQkResidualFold:
    query_weight: Any
    key_weight: Any
    seed: Lfm2EndpointQkvoSeed
    policy: Lfm2EndpointQkResidualFoldPolicy
    fit_bank: Lfm2EndpointQkResidualEvidenceBank
    fit_check_bank: Lfm2EndpointQkResidualEvidenceBank
    fit_proposal: Lfm2CompiledEndpointQkFitProposal
    accepted: bool
    changed: bool
    status: str
    fit_metrics: Lfm2EndpointQkMetrics
    fit_check_metrics: Lfm2EndpointQkMetrics
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        query = _dense_cpu_tensor(
            self.query_weight,
            "query_weight",
            ndim=2,
            dtype=torch.bfloat16,
        )
        key = _dense_cpu_tensor(
            self.key_weight,
            "key_weight",
            ndim=2,
            dtype=torch.bfloat16,
        )
        if tuple(query.shape) != tuple(self.seed.query_weight.shape) or tuple(
            key.shape
        ) != tuple(self.seed.key_weight.shape):
            raise Lfm2EndpointQkResidualFoldError(
                "Compiled endpoint Q/K tensors differ from the seed topology."
            )
        if (
            not isinstance(self.policy, Lfm2EndpointQkResidualFoldPolicy)
            or not self.policy.verify_authentication()
            or not isinstance(self.fit_bank, Lfm2EndpointQkResidualEvidenceBank)
            or not isinstance(
                self.fit_check_bank,
                Lfm2EndpointQkResidualEvidenceBank,
            )
            or not isinstance(
                self.fit_proposal,
                Lfm2CompiledEndpointQkFitProposal,
            )
            or self.fit_proposal.seed is not self.seed
            or self.fit_proposal.policy is not self.policy
            or self.fit_proposal.fit_bank is not self.fit_bank
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Compiled endpoint result does not bind its policy, banks, and "
                "FIT-only proposal."
            )
        fit_sha, fit_check_sha = _validate_banks_and_resources(
            self.seed,
            self.fit_bank,
            self.fit_check_bank,
            self.policy,
        )
        proposal_sha = self.fit_proposal.verify()
        baseline_fit_check = evaluate_lfm2_endpoint_qkvo_bank(
            self.fit_check_bank,
            self.seed,
            compute_devices=self.policy.compute_devices,
        )
        proposed_fit_check = evaluate_lfm2_endpoint_qkvo_bank(
            self.fit_check_bank,
            self.seed,
            query_weight=self.fit_proposal.query_weight,
            key_weight=self.fit_proposal.key_weight,
            compute_devices=self.policy.compute_devices,
        )
        check_nonregressing = _all_nonregressing(
            proposed_fit_check,
            baseline_fit_check,
            self.policy.metric_regression_tolerance,
        )
        expected_accepted = bool(self.fit_proposal.changed and check_nonregressing)
        expected_query = (
            self.fit_proposal.query_weight
            if expected_accepted
            else self.seed.query_weight
        )
        expected_key = (
            self.fit_proposal.key_weight if expected_accepted else self.seed.key_weight
        )
        expected_fit_metrics = (
            self.fit_proposal.proposed_fit_metrics
            if expected_accepted
            else self.fit_proposal.initial_fit_metrics
        )
        expected_check_metrics = (
            proposed_fit_check if expected_accepted else baseline_fit_check
        )
        expected_status = (
            "fit_check_accepted"
            if expected_accepted
            else "fit_check_rejected_seed_fallback"
            if self.fit_proposal.changed
            else "no_exact_bfloat16_fit_improvement"
        )
        body = dict(self.report)
        supplied = body.pop("report_sha256", None)
        proposal_sweeps = [dict(value) for value in self.fit_proposal.report["sweeps"]]
        expected_fields = {
            "schema",
            "method",
            "policy",
            "policy_sha256",
            "fit_evidence_sha256",
            "fit_check_evidence_sha256",
            "fit_proposal_report_sha256",
            "source_split_sha256",
            "source_artifact_sha256",
            "source_inventory",
            "fit_target_receipt_sha256s",
            "fit_check_target_receipt_sha256s",
            "initial_fit_metrics",
            "proposed_fit_metrics",
            "seed_fit_check_metrics",
            "proposed_fit_check_metrics",
            "heldout_endpoint_and_route_nonregressing",
            "heldout_metric_name_is_compatibility_alias_only",
            "fit_check_endpoint_and_route_nonregressing",
            "fit_check_role",
            "unbiased_generalization_claimed",
            "fresh_whole_child_intelligence_validation_required",
            "fit_check_first_metric_evaluation_after_fit_freeze",
            "fit_check_triggered_refit",
            "sweeps",
            "analytical_sweep_count",
            "solver_device",
            "compute_devices",
            "case_assignment",
            "observed_execution_receipt",
            "fixed_value_output_inside_endpoint_jacobian",
            "direct_parent_probability_barycenter_target",
            "direct_parent_endpoint_target",
            "exact_bfloat16_frontier",
            "ordinary_dense_target_qkvo",
            "source_logical_parameters",
            "target_logical_parameters",
            "parameter_compaction_ratio",
            "blank_target_values_read",
            "operation_audit",
            "accepted",
            "changed",
            "status",
            "seed_sha256",
            "query_weight_sha256",
            "key_weight_sha256",
            "fixed_value_weight_sha256",
            "fixed_output_weight_sha256",
            "fit_metrics",
            "fit_check_metrics",
        }
        expected_operation_audit = {
            **_ZERO_OPERATION_AUDIT,
            "fit_cases_consumed": len(self.fit_bank.cases),
            "fit_check_cases_consumed": len(self.fit_check_bank.cases),
            "matrix_free_ridge_normal_solves": len(proposal_sweeps),
        }
        if (
            supplied != _canonical_sha256(body)
            or set(body) != expected_fields
            or body.get("schema") != LFM2_ENDPOINT_QK_RESULT_SCHEMA
            or body.get("method")
            != (
                "fixed_vo_endpoint_and_parent_probability_barycenter_"
                "gauss_newton_ridge_bfloat16_frontier"
            )
            or body.get("query_weight_sha256") != _tensor_sha256(query)
            or body.get("key_weight_sha256") != _tensor_sha256(key)
            or body.get("fixed_value_weight_sha256")
            != _tensor_sha256(self.seed.value_weight)
            or body.get("fixed_output_weight_sha256")
            != _tensor_sha256(self.seed.output_weight)
            or body.get("seed_sha256") != self.seed.verify()
            or body.get("policy") != self.policy.as_dict()
            or body.get("policy_sha256") != self.policy.authentication_sha256
            or body.get("fit_evidence_sha256") != fit_sha
            or body.get("fit_check_evidence_sha256") != fit_check_sha
            or body.get("fit_proposal_report_sha256") != proposal_sha
            or body.get("source_split_sha256") != self.fit_bank.source_split_sha256
            or body.get("source_artifact_sha256")
            != self.fit_bank.source_artifact_sha256
            or body.get("source_inventory") != self.seed.source_inventory.as_dict()
            or body.get("fit_target_receipt_sha256s")
            != list(self.fit_bank.target_receipt_sha256s)
            or body.get("fit_check_target_receipt_sha256s")
            != list(self.fit_check_bank.target_receipt_sha256s)
            or body.get("initial_fit_metrics")
            != self.fit_proposal.initial_fit_metrics.as_dict()
            or body.get("proposed_fit_metrics")
            != self.fit_proposal.proposed_fit_metrics.as_dict()
            or body.get("seed_fit_check_metrics") != baseline_fit_check.as_dict()
            or body.get("proposed_fit_check_metrics") != proposed_fit_check.as_dict()
            or body.get("heldout_endpoint_and_route_nonregressing")
            is not check_nonregressing
            or body.get("heldout_metric_name_is_compatibility_alias_only") is not True
            or body.get("fit_check_endpoint_and_route_nonregressing")
            is not check_nonregressing
            or body.get("fit_check_role")
            != "reused_development_acceptance_gate_not_unbiased_validation"
            or body.get("unbiased_generalization_claimed") is not False
            or body.get("fresh_whole_child_intelligence_validation_required")
            is not True
            or body.get("fit_check_first_metric_evaluation_after_fit_freeze")
            is not True
            or body.get("fit_check_triggered_refit") is not False
            or body.get("sweeps") != proposal_sweeps
            or body.get("analytical_sweep_count") != len(proposal_sweeps)
            or body.get("solver_device") != self.policy.solver_device
            or body.get("compute_devices") != list(self.policy.compute_devices)
            or body.get("case_assignment") != "authenticated_bank_order_round_robin"
            or body.get("observed_execution_receipt")
            != self.fit_proposal.report["observed_execution_receipt"]
            or body.get("fixed_value_output_inside_endpoint_jacobian") is not True
            or body.get("direct_parent_probability_barycenter_target") is not True
            or body.get("direct_parent_endpoint_target") is not True
            or body.get("exact_bfloat16_frontier") is not True
            or body.get("ordinary_dense_target_qkvo") is not True
            or body.get("source_logical_parameters")
            != self.seed.source_logical_parameters
            or body.get("target_logical_parameters")
            != self.seed.target_logical_parameters
            or body.get("parameter_compaction_ratio")
            != self.seed.parameter_compaction_ratio
            or body.get("blank_target_values_read") is not False
            or body.get("operation_audit") != expected_operation_audit
            or body.get("accepted") != self.accepted
            or body.get("changed") != self.changed
            or body.get("status") != self.status
            or body.get("fit_metrics") != self.fit_metrics.as_dict()
            or body.get("fit_check_metrics") != self.fit_check_metrics.as_dict()
            or self.accepted != expected_accepted
            or self.changed != expected_accepted
            or self.status != expected_status
            or not torch.equal(query, expected_query)
            or not torch.equal(key, expected_key)
            or self.fit_metrics.as_dict() != expected_fit_metrics.as_dict()
            or self.fit_check_metrics.as_dict() != expected_check_metrics.as_dict()
        ):
            raise Lfm2EndpointQkResidualAuthenticationError(
                "Compiled endpoint Q/K result failed authentication."
            )
        object.__setattr__(self, "query_weight", query)
        object.__setattr__(self, "key_weight", key)
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "report_sha256": supplied}),
        )

    @property
    def compiled_state_sha256(self) -> str:
        return _state_sha256(self.state_dict())

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().clone(),
                "v_proj.weight": self.seed.value_weight.detach().cpu().clone(),
                "out_proj.weight": self.seed.output_weight.detach().cpu().clone(),
            }
        )

    def verify(self) -> str:
        rebuilt = Lfm2CompiledEndpointQkResidualFold(
            query_weight=self.query_weight,
            key_weight=self.key_weight,
            seed=self.seed,
            policy=self.policy,
            fit_bank=self.fit_bank,
            fit_check_bank=self.fit_check_bank,
            fit_proposal=self.fit_proposal,
            accepted=self.accepted,
            changed=self.changed,
            status=self.status,
            fit_metrics=self.fit_metrics,
            fit_check_metrics=self.fit_check_metrics,
            report=self.report,
        )
        return str(rebuilt.report["report_sha256"])


def _build_result(
    *,
    query_weight: Any,
    key_weight: Any,
    seed: Lfm2EndpointQkvoSeed,
    policy: Lfm2EndpointQkResidualFoldPolicy,
    fit_bank: Lfm2EndpointQkResidualEvidenceBank,
    fit_check_bank: Lfm2EndpointQkResidualEvidenceBank,
    fit_proposal: Lfm2CompiledEndpointQkFitProposal,
    accepted: bool,
    changed: bool,
    status: str,
    fit_metrics: Lfm2EndpointQkMetrics,
    fit_check_metrics: Lfm2EndpointQkMetrics,
    body: Mapping[str, Any],
) -> Lfm2CompiledEndpointQkResidualFold:
    report_body = {
        **dict(body),
        "accepted": accepted,
        "changed": changed,
        "status": status,
        "seed_sha256": seed.seed_sha256,
        "query_weight_sha256": _tensor_sha256(query_weight),
        "key_weight_sha256": _tensor_sha256(key_weight),
        "fixed_value_weight_sha256": _tensor_sha256(seed.value_weight),
        "fixed_output_weight_sha256": _tensor_sha256(seed.output_weight),
        "fit_metrics": fit_metrics.as_dict(),
        "fit_check_metrics": fit_check_metrics.as_dict(),
    }
    report = {
        **report_body,
        "report_sha256": _canonical_sha256(report_body),
    }
    return Lfm2CompiledEndpointQkResidualFold(
        query_weight=query_weight,
        key_weight=key_weight,
        seed=seed,
        policy=policy,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
        fit_proposal=fit_proposal,
        accepted=accepted,
        changed=changed,
        status=status,
        fit_metrics=fit_metrics,
        fit_check_metrics=fit_check_metrics,
        report=report,
    )


def _validate_banks_and_resources(
    seed: Lfm2EndpointQkvoSeed,
    fit_bank: Lfm2EndpointQkResidualEvidenceBank,
    fit_check_bank: Lfm2EndpointQkResidualEvidenceBank,
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> tuple[str, str]:
    fit_sha = fit_bank.verify()
    check_sha = fit_check_bank.verify()
    if (
        fit_bank.partition != "fit"
        or fit_check_bank.partition != "fit_check"
        or fit_bank.source_split_sha256 != fit_check_bank.source_split_sha256
        or fit_bank.source_artifact_sha256 != fit_check_bank.source_artifact_sha256
        or fit_bank.source_artifact_sha256 != seed.source_artifact_sha256
        or set(fit_bank.trajectory_sha256s) & set(fit_check_bank.trajectory_sha256s)
        or {case.evidence_sha256 for case in fit_bank.cases}
        & {case.evidence_sha256 for case in fit_check_bank.cases}
        or set(fit_bank.target_receipt_sha256s)
        & set(fit_check_bank.target_receipt_sha256s)
    ):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "FIT/FIT-CHECK must be disjoint and name the same source split, "
            "source artifact, and dense Q/K/V/O seed."
        )
    if (
        len(fit_bank.cases) > policy.maximum_fit_cases
        or len(fit_check_bank.cases) > policy.maximum_fit_check_cases
    ):
        raise Lfm2EndpointQkResidualFoldError(
            "Endpoint evidence exceeds its bounded case ceilings."
        )
    all_cases = (*fit_bank.cases, *fit_check_bank.cases)
    endpoint_elements = sum(
        int(case.parent_endpoint_target.numel()) for case in all_cases
    )
    probability_elements = sum(
        int(case.target_probability_barycenter.numel()) for case in all_cases
    )
    if (
        endpoint_elements > policy.maximum_endpoint_elements
        or probability_elements > policy.maximum_probability_elements
    ):
        raise Lfm2EndpointQkResidualFoldError(
            "Endpoint evidence exceeds its bounded tensor ceilings."
        )
    return fit_sha, check_sha


def compile_lfm2_cached_endpoint_qk_fit_proposal(
    *,
    seed: Lfm2EndpointQkvoSeed,
    fit_bank: Lfm2EndpointQkResidualEvidenceBank,
    policy: Lfm2EndpointQkResidualFoldPolicy | None = None,
) -> Lfm2CompiledEndpointQkFitProposal:
    """Compile a FIT-only endpoint proposal without any FIT-CHECK interface."""

    if not isinstance(seed, Lfm2EndpointQkvoSeed):
        raise TypeError("seed must be Lfm2EndpointQkvoSeed.")
    if not isinstance(fit_bank, Lfm2EndpointQkResidualEvidenceBank):
        raise TypeError("fit_bank must be Lfm2EndpointQkResidualEvidenceBank.")
    selected_policy = policy or Lfm2EndpointQkResidualFoldPolicy()
    if (
        not isinstance(selected_policy, Lfm2EndpointQkResidualFoldPolicy)
        or not selected_policy.verify_authentication()
    ):
        raise TypeError("policy must be an authenticated endpoint Q/K policy.")
    seed.verify()
    fit_sha = fit_bank.verify()
    if (
        fit_bank.partition != "fit"
        or fit_bank.source_artifact_sha256 != seed.source_artifact_sha256
    ):
        raise Lfm2EndpointQkResidualAuthenticationError(
            "FIT bank and dense Q/K/V/O seed name different source artifacts."
        )
    endpoint_elements = sum(
        int(case.parent_endpoint_target.numel()) for case in fit_bank.cases
    )
    probability_elements = sum(
        int(case.target_probability_barycenter.numel()) for case in fit_bank.cases
    )
    if (
        len(fit_bank.cases) > selected_policy.maximum_fit_cases
        or endpoint_elements > selected_policy.maximum_endpoint_elements
        or probability_elements > selected_policy.maximum_probability_elements
    ):
        raise Lfm2EndpointQkResidualFoldError(
            "FIT endpoint evidence exceeds its bounded resource ceilings."
        )
    solver, workers = _resolve_execution_devices(selected_policy)
    prepared = _prepare_cases_round_robin(
        fit_bank.cases,
        seed,
        compute_devices=workers,
    )
    prepared_bytes = sum(_prepared_persistent_tensor_bytes(case) for case in prepared)
    if prepared_bytes > selected_policy.maximum_prepared_tensor_bytes:
        raise Lfm2EndpointQkResidualFoldError(
            "Prepared FIT tensors exceed their authenticated byte ceiling."
        )
    numerical = _compile_fit_numerics(
        seed=seed,
        fit_cases=prepared,
        solver_device=solver,
        policy=selected_policy,
    )
    execution_receipt = _build_observed_execution_receipt(
        seed=seed,
        fit_bank=fit_bank,
        prepared_cases=prepared,
        numerical_sweeps=numerical.sweeps,
        query_weight=numerical.query_weight,
        key_weight=numerical.key_weight,
        policy=selected_policy,
        solver_device=solver,
        compute_devices=workers,
    )
    body = {
        "schema": LFM2_ENDPOINT_QK_FIT_PROPOSAL_SCHEMA,
        "method": (
            "fit_only_fixed_vo_endpoint_probability_gauss_newton_"
            "ridge_bfloat16_frontier"
        ),
        "seed_sha256": seed.seed_sha256,
        "source_inventory_sha256": seed.source_inventory.inventory_sha256,
        "fit_evidence_sha256": fit_sha,
        "source_split_sha256": fit_bank.source_split_sha256,
        "source_artifact_sha256": fit_bank.source_artifact_sha256,
        "fit_target_receipt_sha256s": list(fit_bank.target_receipt_sha256s),
        "policy": selected_policy.as_dict(),
        "policy_sha256": selected_policy.authentication_sha256,
        "query_weight_sha256": _tensor_sha256(numerical.query_weight),
        "key_weight_sha256": _tensor_sha256(numerical.key_weight),
        "fixed_value_weight_sha256": _tensor_sha256(seed.value_weight),
        "fixed_output_weight_sha256": _tensor_sha256(seed.output_weight),
        "changed": numerical.changed,
        "initial_fit_metrics": numerical.initial_metrics.as_dict(),
        "proposed_fit_metrics": numerical.proposed_metrics.as_dict(),
        "sweeps": [dict(value) for value in numerical.sweeps],
        "analytical_sweep_count": len(numerical.sweeps),
        "solver_device": str(solver),
        "compute_devices": [str(value) for value in workers],
        "case_assignment": "authenticated_bank_order_round_robin",
        "observed_execution_receipt": execution_receipt,
        "fit_check_accessible": False,
        "fit_check_cases_consumed": 0,
        "exact_bfloat16_frontier": True,
        "direct_parent_probability_barycenter_target": True,
        "direct_parent_endpoint_target": True,
        "operation_audit": {
            **_ZERO_OPERATION_AUDIT,
            "fit_cases_consumed": len(fit_bank.cases),
            "fit_check_cases_consumed": 0,
            "matrix_free_ridge_normal_solves": len(numerical.sweeps),
        },
    }
    report = {
        **body,
        "report_sha256": _canonical_sha256(body),
    }
    return Lfm2CompiledEndpointQkFitProposal(
        query_weight=numerical.query_weight,
        key_weight=numerical.key_weight,
        seed=seed,
        policy=selected_policy,
        fit_bank=fit_bank,
        changed=numerical.changed,
        initial_fit_metrics=numerical.initial_metrics,
        proposed_fit_metrics=numerical.proposed_metrics,
        report=report,
    )


def compile_lfm2_cached_endpoint_qk_residual_fold(
    *,
    seed: Lfm2EndpointQkvoSeed,
    fit_bank: Lfm2EndpointQkResidualEvidenceBank,
    fit_check_bank: Lfm2EndpointQkResidualEvidenceBank,
    policy: Lfm2EndpointQkResidualFoldPolicy | None = None,
) -> Lfm2CompiledEndpointQkResidualFold:
    """Compile a bounded endpoint-aware Q/K correction from cached evidence."""

    if not isinstance(seed, Lfm2EndpointQkvoSeed):
        raise TypeError("seed must be Lfm2EndpointQkvoSeed.")
    if not isinstance(fit_bank, Lfm2EndpointQkResidualEvidenceBank):
        raise TypeError("fit_bank must be Lfm2EndpointQkResidualEvidenceBank.")
    if not isinstance(fit_check_bank, Lfm2EndpointQkResidualEvidenceBank):
        raise TypeError("fit_check_bank must be Lfm2EndpointQkResidualEvidenceBank.")
    selected_policy = policy or Lfm2EndpointQkResidualFoldPolicy()
    if (
        not isinstance(selected_policy, Lfm2EndpointQkResidualFoldPolicy)
        or not selected_policy.verify_authentication()
    ):
        raise TypeError("policy must be an authenticated endpoint Q/K policy.")
    seed.verify()
    fit_sha, fit_check_sha = _validate_banks_and_resources(
        seed,
        fit_bank,
        fit_check_bank,
        selected_policy,
    )
    solver, workers = _resolve_execution_devices(selected_policy)
    fit_proposal = compile_lfm2_cached_endpoint_qk_fit_proposal(
        seed=seed,
        fit_bank=fit_bank,
        policy=selected_policy,
    )
    current_query = fit_proposal.query_weight.to(device=solver)
    current_key = fit_proposal.key_weight.to(device=solver)
    initial_fit_metrics = fit_proposal.initial_fit_metrics
    current_fit_metrics = fit_proposal.proposed_fit_metrics
    sweeps = [dict(value) for value in fit_proposal.report["sweeps"]]
    changed = fit_proposal.changed

    # FIT is frozen here.  FIT-CHECK is first numerically prepared/evaluated
    # after all analytical solves and exact-BF16 frontier decisions are over.
    fit_check_cases = _prepare_cases_round_robin(
        fit_check_bank.cases,
        seed,
        compute_devices=workers,
    )
    seed_fit_check_metrics = _evaluate(
        fit_check_cases,
        seed,
        seed.query_weight.to(device=solver),
        seed.key_weight.to(device=solver),
    )
    proposed_fit_check_metrics = _evaluate(
        fit_check_cases,
        seed,
        current_query,
        current_key,
    )
    fit_check_nonregressing = _all_nonregressing(
        proposed_fit_check_metrics,
        seed_fit_check_metrics,
        selected_policy.metric_regression_tolerance,
    )
    accepted = bool(changed and fit_check_nonregressing)
    published_query = current_query.detach().cpu() if accepted else seed.query_weight
    published_key = current_key.detach().cpu() if accepted else seed.key_weight
    published_fit_metrics = current_fit_metrics if accepted else initial_fit_metrics
    published_fit_check_metrics = (
        proposed_fit_check_metrics if accepted else seed_fit_check_metrics
    )
    body = {
        "schema": LFM2_ENDPOINT_QK_RESULT_SCHEMA,
        "method": (
            "fixed_vo_endpoint_and_parent_probability_barycenter_"
            "gauss_newton_ridge_bfloat16_frontier"
        ),
        "policy": selected_policy.as_dict(),
        "policy_sha256": selected_policy.authentication_sha256,
        "fit_evidence_sha256": fit_sha,
        "fit_check_evidence_sha256": fit_check_sha,
        "fit_proposal_report_sha256": fit_proposal.report_sha256,
        "source_split_sha256": fit_bank.source_split_sha256,
        "source_artifact_sha256": fit_bank.source_artifact_sha256,
        "source_inventory": seed.source_inventory.as_dict(),
        "fit_target_receipt_sha256s": list(fit_bank.target_receipt_sha256s),
        "fit_check_target_receipt_sha256s": list(fit_check_bank.target_receipt_sha256s),
        "initial_fit_metrics": initial_fit_metrics.as_dict(),
        "proposed_fit_metrics": current_fit_metrics.as_dict(),
        "seed_fit_check_metrics": seed_fit_check_metrics.as_dict(),
        "proposed_fit_check_metrics": proposed_fit_check_metrics.as_dict(),
        "heldout_endpoint_and_route_nonregressing": fit_check_nonregressing,
        "heldout_metric_name_is_compatibility_alias_only": True,
        "fit_check_endpoint_and_route_nonregressing": fit_check_nonregressing,
        "fit_check_role": (
            "reused_development_acceptance_gate_not_unbiased_validation"
        ),
        "unbiased_generalization_claimed": False,
        "fresh_whole_child_intelligence_validation_required": True,
        "fit_check_first_metric_evaluation_after_fit_freeze": True,
        "fit_check_triggered_refit": False,
        "sweeps": sweeps,
        "analytical_sweep_count": len(sweeps),
        "solver_device": str(solver),
        "compute_devices": [str(value) for value in workers],
        "case_assignment": "authenticated_bank_order_round_robin",
        "observed_execution_receipt": fit_proposal.report["observed_execution_receipt"],
        "fixed_value_output_inside_endpoint_jacobian": True,
        "direct_parent_probability_barycenter_target": True,
        "direct_parent_endpoint_target": True,
        "exact_bfloat16_frontier": True,
        "ordinary_dense_target_qkvo": True,
        "source_logical_parameters": seed.source_logical_parameters,
        "target_logical_parameters": seed.target_logical_parameters,
        "parameter_compaction_ratio": seed.parameter_compaction_ratio,
        "blank_target_values_read": False,
        "operation_audit": {
            **_ZERO_OPERATION_AUDIT,
            "fit_cases_consumed": len(fit_bank.cases),
            "fit_check_cases_consumed": len(fit_check_bank.cases),
            "matrix_free_ridge_normal_solves": len(sweeps),
        },
    }
    return _build_result(
        query_weight=published_query,
        key_weight=published_key,
        seed=seed,
        policy=selected_policy,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
        fit_proposal=fit_proposal,
        accepted=accepted,
        changed=accepted,
        status=(
            "fit_check_accepted"
            if accepted
            else "fit_check_rejected_seed_fallback"
            if changed
            else "no_exact_bfloat16_fit_improvement"
        ),
        fit_metrics=published_fit_metrics,
        fit_check_metrics=published_fit_check_metrics,
        body=body,
    )


__all__ = [
    "LFM2_ENDPOINT_QK_BANK_SCHEMA",
    "LFM2_ENDPOINT_QK_CASE_SCHEMA",
    "LFM2_ENDPOINT_QK_EXECUTION_SCHEMA",
    "LFM2_ENDPOINT_QK_FIT_PROPOSAL_SCHEMA",
    "LFM2_ENDPOINT_QK_POLICY_SCHEMA",
    "LFM2_ENDPOINT_QK_RESULT_SCHEMA",
    "LFM2_ENDPOINT_QK_SEED_SCHEMA",
    "LFM2_ENDPOINT_SOURCE_INVENTORY_SCHEMA",
    "Lfm2CompiledEndpointQkFitProposal",
    "Lfm2CompiledEndpointQkResidualFold",
    "Lfm2DenseQkvoSourceInventory",
    "Lfm2EndpointQkLinearOperator",
    "Lfm2EndpointQkMetrics",
    "Lfm2EndpointQkResidualAuthenticationError",
    "Lfm2EndpointQkResidualCase",
    "Lfm2EndpointQkResidualEvidenceBank",
    "Lfm2EndpointQkResidualFoldError",
    "Lfm2EndpointQkResidualFoldPolicy",
    "Lfm2EndpointQkvoSeed",
    "build_lfm2_endpoint_qkvo_seed",
    "compile_lfm2_cached_endpoint_qk_fit_proposal",
    "compile_lfm2_cached_endpoint_qk_residual_fold",
    "evaluate_lfm2_endpoint_qkvo_bank",
    "replay_lfm2_endpoint_qkvo",
    "seal_lfm2_dense_qkvo_source_inventory",
    "seal_lfm2_endpoint_qk_fit_bank",
    "seal_lfm2_endpoint_qk_fit_check_bank",
]
