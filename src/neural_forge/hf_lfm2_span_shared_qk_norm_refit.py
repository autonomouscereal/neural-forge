"""Closed-form shared-Q/K RMSNorm refit for an LFM2 depth span.

An LFM2 child exposes one learned Q RMSNorm vector and one learned K RMSNorm
vector per attention layer.  A span-QDEIM fold can select intact heads from
several parent layers whose learned norms differ, so directly installing those
heads is not representable by the child topology.  This module resolves that
topological mismatch without averaging norms, executing a model, or training.

Fit diagnostics are descriptive only.  They never accept, reject, rank, or
select an installed tensor.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_SPAN_SHARED_QK_NORM_REFIT_SCHEMA = (
    "neural-forge-lfm2-span-shared-qk-norm-closed-form-refit-v1"
)
LFM2_SPAN_SHARED_QK_NORM_REFIT_METHOD = (
    "coordinatewise-envelope-rmsnorm-preimage-rowspace-ridge"
)
LFM2_SPAN_SHARED_QK_NORM_REFIT_TENSOR_HASH_SCHEME = (
    "neural-forge-lfm2-model-fold-tensor-sha256-v1"
)

_ATTENTION_STATE_NAMES = frozenset(
    {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
    }
)
_REFIT_STATE_NAMES = frozenset(
    {
        "q_proj.weight",
        "k_proj.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
    }
)
_UNCHANGED_STATE_NAMES = frozenset({"v_proj.weight", "out_proj.weight"})


class Lfm2SpanSharedQkNormRefitError(ValueError):
    """The authenticated closed-form shared-norm refit failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "PyTorch is required for shared Q/K norm refitting."
        ) from exc
    return torch


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise Lfm2SpanSharedQkNormRefitError(f"{label} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} must be finite and positive."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2SpanSharedQkNormRefitError(f"{label} must be finite and positive.")
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} must be finite and non-negative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} must be finite and non-negative."
        )
    return result


def _tensor(value: Any, *, label: str, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or not value.is_floating_point()
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} must be one materialized dense floating tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2SpanSharedQkNormRefitError(f"{label} contains NaN or infinity.")
    return result


def lfm2_span_shared_qk_norm_refit_tensor_sha256(value: Any) -> str:
    """Hash a tensor in the model-fold installed-state digest domain."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Cannot fingerprint a non-materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-model-fold-tensor-v1\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _lineage(
    values: Sequence[tuple[int, int]],
    *,
    expected_count: int,
    label: str,
) -> tuple[tuple[int, int], ...]:
    if isinstance(values, (str, bytes)):
        raise Lfm2SpanSharedQkNormRefitError(f"{label} must be an ordered lineage.")
    try:
        result = tuple(tuple(value) for value in values)
    except TypeError as exc:
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} must be an ordered lineage."
        ) from exc
    if (
        len(result) != expected_count
        or len(set(result)) != len(result)
        or any(
            len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in value
            )
            for value in result
        )
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} must uniquely bind every selected non-negative source index."
        )
    return result


def _relative_rmse(prediction: Any, target: Any) -> float:
    torch = _torch()
    error = torch.sqrt(torch.mean((prediction - target).square()))
    scale = torch.sqrt(torch.mean(target.square())).clamp_min(
        torch.finfo(torch.float64).eps
    )
    return float((error / scale).item())


def _runtime_rms_norm(value: Any, weight: Any, epsilon: float) -> Any:
    torch = _torch()
    input_dtype = value.dtype
    direction = value.to(torch.float32)
    variance = direction.square().mean(dim=-1, keepdim=True)
    direction = direction * torch.rsqrt(variance + epsilon)
    return (weight.to(input_dtype) * direction.to(input_dtype)).contiguous()


def _maximum_direction_mean_square(normalized: Any, target_norm: Any) -> float:
    width = int(normalized.shape[-1])
    result = (
        (
            normalized.to(_torch().float64)
            / target_norm.to(_torch().float64).reshape(1, 1, width)
        )
        .square()
        .mean(dim=-1)
        .max()
        .item()
    )
    return float(result)


def _coordinatewise_absolute_envelope(
    source_norm_rows: Any,
    *,
    target_dtype: Any,
    label: str,
) -> tuple[Any, Mapping[str, Any]]:
    torch = _torch()
    source_norm64 = source_norm_rows.to(torch.float64)
    exact_envelope64 = source_norm64.abs().amax(dim=0).contiguous()
    zero_floor = float(torch.finfo(target_dtype).tiny)
    zero_mask = exact_envelope64 == 0.0
    lower_bound64 = torch.where(
        ~zero_mask,
        exact_envelope64,
        torch.full_like(exact_envelope64, zero_floor),
    )
    nearest = lower_bound64.to(dtype=target_dtype).contiguous()
    infinity = torch.full_like(nearest, float("inf"))
    upward_mask = nearest.to(torch.float64) < lower_bound64
    envelope = torch.where(
        upward_mask,
        torch.nextafter(nearest, infinity),
        nearest,
    ).contiguous()
    if not bool(torch.isfinite(envelope).all().item()) or not bool(
        (envelope > 0).all().item()
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} coordinatewise norm envelope is not finite and positive."
        )
    if bool((envelope.to(torch.float64) < lower_bound64).any().item()):
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} target-dtype norm envelope rounded below its exact lower bound."
        )
    previous = torch.nextafter(envelope, torch.zeros_like(envelope))
    if bool((previous.to(torch.float64) >= lower_bound64).any().item()):
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} target-dtype norm envelope is not the minimal upward rounding."
        )
    smallest_positive = torch.nextafter(
        torch.zeros((), dtype=target_dtype),
        torch.full((), float("inf"), dtype=target_dtype),
    )
    report = MappingProxyType(
        {
            "rule": "minimal-per-coordinate-upward-target-dtype-rounding",
            "source_absolute_max_float64_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(exact_envelope64)
            ),
            "positive_zero_floored_lower_bound_float64_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(lower_bound64)
            ),
            "target_dtype_upward_rounded_envelope_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(envelope)
            ),
            "target_dtype": str(target_dtype).removeprefix("torch."),
            "positive_zero_floor": zero_floor,
            "target_dtype_smallest_positive_subnormal": float(smallest_positive.item()),
            "zero_floored_coordinate_count": int(zero_mask.sum().item()),
            "positive_sub_tiny_source_coordinate_count": int(
                ((exact_envelope64 > 0.0) & (exact_envelope64 < zero_floor))
                .sum()
                .item()
            ),
            "upward_nextafter_coordinate_count": int(upward_mask.sum().item()),
            "upward_nextafter_steps_per_adjusted_coordinate": 1,
            "minimal_per_coordinate_target_dtype_ceiling_verified": True,
            "rounded_envelope_is_not_below_float64_lower_bound": True,
            "envelope_rounding_has_quality_gate_authority": False,
        }
    )
    return envelope, report


def _positive_ulp_distance_from_one(value: Any) -> int:
    torch = _torch()
    integer_dtype = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }.get(value.dtype)
    if integer_dtype is None:
        raise Lfm2SpanSharedQkNormRefitError(
            f"Target dtype {value.dtype} has no authenticated ULP encoding."
        )
    scalar = value.detach().cpu().reshape(1)
    one = torch.ones(1, dtype=value.dtype)
    return int(scalar.view(integer_dtype).item()) - int(one.view(integer_dtype).item())


def _minimal_common_representability_scale(
    *,
    query_envelope: Any,
    key_envelope: Any,
    query_targets: Any,
    key_targets: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Apply only the smallest common native scalar needed for a preimage."""

    torch = _torch()
    if query_envelope.dtype != key_envelope.dtype:
        raise Lfm2SpanSharedQkNormRefitError(
            "Query and key norm envelopes must use one target dtype."
        )
    target_dtype = query_envelope.dtype
    one = torch.ones((), dtype=target_dtype, device="cpu")
    infinity = torch.full((), float("inf"), dtype=target_dtype, device="cpu")
    query_pre = _maximum_direction_mean_square(query_targets, query_envelope)
    key_pre = _maximum_direction_mean_square(key_targets, key_envelope)
    worst_pre = max(query_pre, key_pre)

    def scaled(scale: Any) -> tuple[Any, Any, float, float, float]:
        query = (query_envelope * scale).to(dtype=target_dtype).contiguous()
        key = (key_envelope * scale).to(dtype=target_dtype).contiguous()
        if not bool(torch.isfinite(query).all().item()) or not bool(
            torch.isfinite(key).all().item()
        ):
            raise Lfm2SpanSharedQkNormRefitError(
                "The common Q/K representability scale overflowed the target dtype."
            )
        if bool((query < query_envelope).any().item()) or bool(
            (key < key_envelope).any().item()
        ):
            raise Lfm2SpanSharedQkNormRefitError(
                "The common Q/K representability scale violated an envelope lower bound."
            )
        query_maximum = _maximum_direction_mean_square(query_targets, query)
        key_maximum = _maximum_direction_mean_square(key_targets, key)
        return query, key, query_maximum, key_maximum, max(query_maximum, key_maximum)

    upward_steps = 0
    downward_minimality_steps = 0
    if worst_pre < 1.0:
        scale = one
        query_norm, key_norm, query_post, key_post, worst_post = scaled(scale)
    else:
        required = math.sqrt(worst_pre)
        scale = torch.as_tensor(required, dtype=target_dtype, device="cpu")
        if bool((scale < one).item()):
            scale = one
        if not bool(torch.isfinite(scale).item()):
            raise Lfm2SpanSharedQkNormRefitError(
                "No finite target-dtype common Q/K representability scale exists."
            )
        query_norm, key_norm, query_post, key_post, worst_post = scaled(scale)
        while worst_post >= 1.0:
            next_scale = torch.nextafter(scale, infinity)
            if bool(torch.equal(next_scale, scale)) or not bool(
                torch.isfinite(next_scale).item()
            ):
                raise Lfm2SpanSharedQkNormRefitError(
                    "No finite target-dtype common Q/K representability scale exists."
                )
            scale = next_scale
            upward_steps += 1
            query_norm, key_norm, query_post, key_post, worst_post = scaled(scale)

        while bool((scale > one).item()):
            previous_scale = torch.nextafter(scale, one)
            if bool((previous_scale < one).item()) or bool(
                torch.equal(previous_scale, scale)
            ):
                break
            (
                previous_query,
                previous_key,
                previous_query_post,
                previous_key_post,
                previous_worst_post,
            ) = scaled(previous_scale)
            if previous_worst_post >= 1.0:
                break
            scale = previous_scale
            query_norm = previous_query
            key_norm = previous_key
            query_post = previous_query_post
            key_post = previous_key_post
            worst_post = previous_worst_post
            downward_minimality_steps += 1

    previous_scale_value = None
    previous_worst = None
    if bool((scale > one).item()):
        previous_scale = torch.nextafter(scale, one)
        _, _, _, _, previous_worst_value = scaled(previous_scale)
        previous_scale_value = float(previous_scale.item())
        previous_worst = float(previous_worst_value)
        if previous_worst < 1.0:
            raise Lfm2SpanSharedQkNormRefitError(
                "The selected common Q/K representability scale is not minimal."
            )

    applied = not bool(torch.equal(scale, one))
    report = MappingProxyType(
        {
            "rule": ("smallest-common-upward-target-dtype-representable-scalar"),
            "original_query_coordinatewise_envelope_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(query_envelope)
            ),
            "original_key_coordinatewise_envelope_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(key_envelope)
            ),
            "applied": applied,
            "common_scale_target_dtype_value": float(scale.item()),
            "common_scale_target_dtype_tensor_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(scale.reshape(1))
            ),
            "common_scale_ulp_distance_above_one": (
                _positive_ulp_distance_from_one(scale)
            ),
            "nearest_threshold_upward_nextafter_steps": upward_steps,
            "downward_minimality_verification_steps": downward_minimality_steps,
            "query_pre_repair_maximum_direction_mean_square": query_pre,
            "key_pre_repair_maximum_direction_mean_square": key_pre,
            "worst_pre_repair_maximum_direction_mean_square": worst_pre,
            "query_post_repair_maximum_direction_mean_square": query_post,
            "key_post_repair_maximum_direction_mean_square": key_post,
            "worst_post_repair_maximum_direction_mean_square": worst_post,
            "previous_target_dtype_scale_value": previous_scale_value,
            "previous_scale_worst_direction_mean_square": previous_worst,
            "one_ulp_safety_above_equality_boundary": bool(previous_worst == 1.0),
            "coordinatewise_envelopes_are_lower_bounds": True,
            "one_common_scalar_applied_to_every_query_and_key_coordinate": True,
            "minimum_target_dtype_scale_verified": True,
            "representability_repair_has_quality_gate_authority": False,
        }
    )
    return query_norm, key_norm, report


def _preimage(normalized: Any, target_norm: Any, epsilon: float, label: str) -> Any:
    torch = _torch()
    width = int(normalized.shape[-1])
    values = normalized.to(torch.float64)
    gamma = target_norm.to(torch.float64).reshape(1, 1, width)
    direction = values / gamma
    mean_square = direction.square().mean(dim=-1, keepdim=True)
    if bool((mean_square >= 1.0).any().item()):
        maximum = float(mean_square.max().item())
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} has no finite exact RMSNorm preimage under the authenticated "
            f"coordinatewise envelope (maximum direction mean-square {maximum:.9g})."
        )
    scale = torch.sqrt(
        torch.as_tensor(epsilon, dtype=torch.float64) / (1.0 - mean_square)
    )
    result = (direction * scale).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} produced a non-finite RMSNorm preimage."
        )
    return result


def _rowspace_delta(
    x: Any,
    residual: Any,
    ridge: float,
) -> tuple[Any, Mapping[str, Any]]:
    torch = _torch()
    if ridge > 0.0:
        gram = (x @ x.T).contiguous()
        system = gram + ridge * torch.eye(
            int(gram.shape[0]), dtype=torch.float64, device="cpu"
        )
        try:
            solved = torch.linalg.solve(system, residual)
        except RuntimeError as exc:
            raise Lfm2SpanSharedQkNormRefitError(
                "The authenticated row-space ridge system is not solvable."
            ) from exc
        delta = (x.T @ solved).T.contiguous()
        method = MappingProxyType(
            {
                "method": "positive-ridge-row-gram-linear-solve",
                "ridge": ridge,
                "pseudoinverse_relative_tolerance": None,
                "pseudoinverse_absolute_tolerance": None,
                "minimum_frobenius_norm_solution": False,
            }
        )
        return delta, method

    relative_tolerance = (
        max(int(x.shape[0]), int(x.shape[1])) * torch.finfo(torch.float64).eps
    )
    try:
        pseudoinverse = torch.linalg.pinv(
            x,
            atol=0.0,
            rtol=relative_tolerance,
            hermitian=False,
        )
    except RuntimeError as exc:
        raise Lfm2SpanSharedQkNormRefitError(
            "The authenticated zero-ridge minimum-norm pseudoinverse failed."
        ) from exc
    delta = (pseudoinverse @ residual).T.contiguous()
    method = MappingProxyType(
        {
            "method": "zero-ridge-moore-penrose-minimum-frobenius-norm",
            "ridge": 0.0,
            "pseudoinverse_relative_tolerance": relative_tolerance,
            "pseudoinverse_absolute_tolerance": 0.0,
            "minimum_frobenius_norm_solution": True,
        }
    )
    return delta, method


def _nullspace_leakage(x: Any, correction: Any) -> float:
    torch = _torch()
    _u, singular, vh = torch.linalg.svd(x, full_matrices=True)
    if int(singular.numel()) == 0:
        rank = 0
    else:
        tolerance = (
            max(int(x.shape[0]), int(x.shape[1]))
            * torch.finfo(torch.float64).eps
            * float(singular.max().item())
        )
        rank = int((singular > tolerance).sum().item())
    nullspace = vh[rank:, :]
    if int(nullspace.shape[0]) == 0:
        return 0.0
    return float((correction @ nullspace.T).abs().max().item())


def _compile_branch(
    *,
    label: str,
    transported_weight: Any,
    source_norm_rows: Any,
    normalized_targets: Any,
    child_inputs: Any,
    original_envelope: Any,
    envelope_report: Mapping[str, Any],
    target_norm: Any,
    pre_repair_maximum_direction_mean_square: float,
    epsilon: float,
    ridge: float,
) -> tuple[Any, Any, Mapping[str, Any]]:
    torch = _torch()
    row_count, head_count, head_dim = normalized_targets.shape
    target_dtype = target_norm.dtype
    source_norm64 = source_norm_rows.to(torch.float64)
    if not bool((target_norm > 0).all().item()):
        raise Lfm2SpanSharedQkNormRefitError(
            f"{label} target norm envelope is not strictly positive."
        )

    preimage = _preimage(normalized_targets, target_norm, epsilon, label)
    target_matrix = preimage.reshape(row_count, head_count * head_dim)
    target_gamma_rows = target_norm.to(torch.float64).repeat(head_count)
    source_gamma_rows = source_norm64.reshape(head_count * head_dim)
    compensated_prior = (
        transported_weight.to(torch.float64)
        * (source_gamma_rows / target_gamma_rows).reshape(-1, 1)
    ).contiguous()

    x64 = child_inputs.to(torch.float64)
    prior_prediction = (x64 @ compensated_prior.T).contiguous()
    denominator = float(prior_prediction.square().sum().item())
    if denominator == 0.0:
        scale = 0.0
    else:
        numerator = float((prior_prediction * target_matrix).sum().item())
        scale = max(0.0, numerator / denominator)
    scaled_prior = (compensated_prior * scale).contiguous()
    analytic_residual = target_matrix - x64 @ scaled_prior.T
    analytic_delta, analytic_solver = _rowspace_delta(
        x64,
        analytic_residual,
        ridge,
    )
    analytic_weight = (scaled_prior + analytic_delta).contiguous()

    first_native = analytic_weight.to(dtype=target_dtype).contiguous()
    native_x = child_inputs.to(dtype=target_dtype).contiguous()
    native_projection = (native_x @ first_native.T).to(torch.float64)
    fixed_residual = target_matrix - native_projection
    fixed_delta, fixed_solver = _rowspace_delta(
        native_x.to(torch.float64),
        fixed_residual,
        ridge,
    )
    final_weight = (first_native.to(torch.float64) + fixed_delta).to(dtype=target_dtype)
    final_weight = final_weight.contiguous()

    linear = (native_x @ final_weight.T).reshape(row_count, head_count, head_dim)
    normalized_replay = _runtime_rms_norm(linear, target_norm, epsilon).to(
        torch.float64
    )
    report = MappingProxyType(
        {
            "coordinatewise_envelope": dict(envelope_report),
            "target_norm_coordinatewise_envelope_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(original_envelope)
            ),
            "installed_target_norm_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(target_norm)
            ),
            "pre_repair_maximum_direction_mean_square": (
                pre_repair_maximum_direction_mean_square
            ),
            "preimage_maximum_direction_mean_square": (
                _maximum_direction_mean_square(normalized_targets, target_norm)
            ),
            "nonnegative_prior_scale": scale,
            "analytic_precast_correction_constructed_in_cached_input_rowspace": True,
            "analytic_precast_prior_nullspace_maximum_absolute_leakage_advisory": (
                _nullspace_leakage(x64, analytic_delta)
            ),
            "analytic_precast_rowspace_solver": dict(analytic_solver),
            "final_target_dtype_prior_nullspace_maximum_absolute_leakage_advisory": (
                _nullspace_leakage(
                    x64,
                    final_weight.to(torch.float64) - scaled_prior,
                )
            ),
            "final_target_dtype_prior_nullspace_preservation_claimed": False,
            "fixed_native_residual_rowspace_solver": dict(fixed_solver),
            "nullspace_leakage_diagnostics_have_install_authority": False,
            "fixed_native_residual_compensation_passes": 1,
            "cached_normalized_replay_relative_rmse_diagnostic": _relative_rmse(
                normalized_replay,
                normalized_targets.to(torch.float64),
            ),
            "numeric_diagnostics_have_install_authority": False,
        }
    )
    return final_weight, target_norm, report


def _diagnostic_report(
    *,
    child_inputs: Any | None,
    query_targets: Any | None,
    key_targets: Any | None,
    query_weight: Any,
    key_weight: Any,
    query_norm: Any,
    key_norm: Any,
    epsilon: float,
) -> Mapping[str, Any]:
    torch = _torch()
    supplied = (
        child_inputs is not None,
        query_targets is not None,
        key_targets is not None,
    )
    if not any(supplied):
        return MappingProxyType(
            {
                "supplied": False,
                "numeric_diagnostics_have_install_authority": False,
            }
        )
    if not all(supplied):
        raise Lfm2SpanSharedQkNormRefitError(
            "Diagnostic child/Q/K rows must be supplied as one complete bundle."
        )
    assert child_inputs is not None
    assert query_targets is not None
    assert key_targets is not None
    x = _tensor(child_inputs, label="diagnostic child inputs", ndim=2)
    q = _tensor(query_targets, label="diagnostic query targets", ndim=3)
    k = _tensor(key_targets, label="diagnostic key targets", ndim=3)
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    head_dim = int(query_norm.shape[0])
    if (
        int(x.shape[0]) != int(q.shape[0])
        or int(x.shape[0]) != int(k.shape[0])
        or int(x.shape[1]) != int(query_weight.shape[1])
        or tuple(q.shape[1:]) != (q_heads, head_dim)
        or tuple(k.shape[1:]) != (k_heads, head_dim)
        or q_heads * head_dim != int(query_weight.shape[0])
        or k_heads * head_dim != int(key_weight.shape[0])
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Diagnostic rows differ from the compiled Q/K topology."
        )
    native_x = x.to(query_weight.dtype)
    q_linear = (native_x @ query_weight.T).reshape_as(q)
    k_linear = (native_x @ key_weight.T).reshape_as(k)
    q_replay = _runtime_rms_norm(q_linear, query_norm, epsilon).to(torch.float64)
    k_replay = _runtime_rms_norm(k_linear, key_norm, epsilon).to(torch.float64)
    return MappingProxyType(
        {
            "supplied": True,
            "child_input_rows_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(x)
            ),
            "query_normalized_pre_rope_rows_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(q)
            ),
            "key_normalized_pre_rope_rows_sha256": (
                lfm2_span_shared_qk_norm_refit_tensor_sha256(k)
            ),
            "query_relative_rmse_diagnostic": _relative_rmse(
                q_replay, q.to(torch.float64)
            ),
            "key_relative_rmse_diagnostic": _relative_rmse(
                k_replay, k.to(torch.float64)
            ),
            "numeric_diagnostics_have_install_authority": False,
        }
    )


def verify_lfm2_span_shared_qk_norm_refit_receipt(
    receipt: Mapping[str, Any],
    *,
    installed_attention_state_sha256: Mapping[str, str] | None = None,
) -> str:
    """Verify one serialized refit receipt and optional outer tensor manifest."""

    if not isinstance(receipt, Mapping):
        raise Lfm2SpanSharedQkNormRefitError("Refit receipt must be a mapping.")
    try:
        normalized = json.loads(
            json.dumps(
                dict(receipt),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt is not canonical JSON."
        ) from exc
    claimed = _digest(normalized.pop("result_sha256", None), "result_sha256")
    if canonical_sha256(normalized) != claimed:
        raise Lfm2SpanSharedQkNormRefitError("Refit receipt self-hash is invalid.")
    if (
        normalized.get("schema") != LFM2_SPAN_SHARED_QK_NORM_REFIT_SCHEMA
        or normalized.get("method") != LFM2_SPAN_SHARED_QK_NORM_REFIT_METHOD
        or normalized.get("target_norm_rule")
        != (
            "coordinatewise-absolute-envelope-lower-bound-with-minimal-common-"
            "target-dtype-representability-scale"
        )
        or normalized.get("coordinatewise_envelope_lower_bound_preserved") is not True
        or normalized.get("heterogeneous_source_norms_averaged") is not False
        or normalized.get("representability_repair_has_quality_gate_authority")
        is not False
        or normalized.get("zero_ridge_uses_minimum_frobenius_norm_pseudoinverse")
        is not True
        or normalized.get(
            "analytic_precast_correction_constructed_in_cached_input_rowspace"
        )
        is not True
        or normalized.get("final_target_dtype_prior_nullspace_preservation_claimed")
        is not False
        or normalized.get("nullspace_leakage_diagnostics_have_install_authority")
        is not False
        or "prior_preserved_on_nullspace_of_cached_inputs" in normalized
        or normalized.get("fixed_native_residual_compensation_passes") != 1
        or normalized.get("installed_attention_state_hash_scheme")
        != LFM2_SPAN_SHARED_QK_NORM_REFIT_TENSOR_HASH_SCHEME
        or normalized.get("numeric_diagnostics_are_advisory_only") is not True
        or normalized.get("numeric_diagnostics_have_install_authority") is not False
        or normalized.get("candidate_selection_performed") is not False
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt changed its deterministic non-gating algorithm contract."
        )
    for key in (
        "parent_attention_compiler_result_sha256",
        "child_input_rows_sha256",
        "query_normalized_pre_rope_rows_sha256",
        "key_normalized_pre_rope_rows_sha256",
        "transported_query_prior_sha256",
        "transported_key_prior_sha256",
        "selected_query_source_norm_rows_sha256",
        "selected_key_source_norm_rows_sha256",
        "selected_query_source_norm_rows_semantic_sha256",
        "selected_key_source_norm_rows_semantic_sha256",
    ):
        _digest(normalized.get(key), key)
    output = normalized.get("installed_attention_state_sha256")
    refit = normalized.get("refit_state_sha256")
    unchanged = normalized.get("unchanged_state_sha256")
    if (
        not isinstance(output, dict)
        or set(output) != _ATTENTION_STATE_NAMES
        or not isinstance(refit, dict)
        or set(refit) != _REFIT_STATE_NAMES
        or not isinstance(unchanged, dict)
        or set(unchanged) != _UNCHANGED_STATE_NAMES
        or any(_digest(value, key) != value for key, value in output.items())
        or refit != {key: output[key] for key in _REFIT_STATE_NAMES}
        or unchanged != {key: output[key] for key in _UNCHANGED_STATE_NAMES}
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt does not exactly partition its installed attention state."
        )
    if (
        installed_attention_state_sha256 is not None
        and dict(installed_attention_state_sha256) != output
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt differs from the outer installed attention manifest."
        )
    repair = normalized.get("representability_repair")
    query_refit = normalized.get("query_refit")
    key_refit = normalized.get("key_refit")

    def finite_nonnegative(value: Any) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
        )

    if (
        not isinstance(repair, dict)
        or not isinstance(query_refit, dict)
        or not isinstance(key_refit, dict)
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt lacks its authenticated Q/K representability repair."
        )
    query_pre = repair.get("query_pre_repair_maximum_direction_mean_square")
    key_pre = repair.get("key_pre_repair_maximum_direction_mean_square")
    worst_pre = repair.get("worst_pre_repair_maximum_direction_mean_square")
    query_post = repair.get("query_post_repair_maximum_direction_mean_square")
    key_post = repair.get("key_post_repair_maximum_direction_mean_square")
    worst_post = repair.get("worst_post_repair_maximum_direction_mean_square")
    scale = repair.get("common_scale_target_dtype_value")
    ulps = repair.get("common_scale_ulp_distance_above_one")
    previous_scale = repair.get("previous_target_dtype_scale_value")
    previous_worst = repair.get("previous_scale_worst_direction_mean_square")
    applied = repair.get("applied")
    if (
        repair.get("rule") != "smallest-common-upward-target-dtype-representable-scalar"
        or repair.get("coordinatewise_envelopes_are_lower_bounds") is not True
        or repair.get("one_common_scalar_applied_to_every_query_and_key_coordinate")
        is not True
        or repair.get("minimum_target_dtype_scale_verified") is not True
        or repair.get("representability_repair_has_quality_gate_authority") is not False
        or not isinstance(applied, bool)
        or not all(
            finite_nonnegative(value)
            for value in (
                query_pre,
                key_pre,
                worst_pre,
                query_post,
                key_post,
                worst_post,
            )
        )
        or worst_pre != max(query_pre, key_pre)
        or worst_post != max(query_post, key_post)
        or worst_post >= 1.0
        or not finite_nonnegative(scale)
        or scale < 1.0
        or isinstance(ulps, bool)
        or not isinstance(ulps, int)
        or ulps < 0
        or not isinstance(repair.get("nearest_threshold_upward_nextafter_steps"), int)
        or repair.get("nearest_threshold_upward_nextafter_steps") < 0
        or not isinstance(repair.get("downward_minimality_verification_steps"), int)
        or repair.get("downward_minimality_verification_steps") < 0
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt changed its minimal Q/K representability repair."
        )
    for key in (
        "original_query_coordinatewise_envelope_sha256",
        "original_key_coordinatewise_envelope_sha256",
        "common_scale_target_dtype_tensor_sha256",
    ):
        _digest(repair.get(key), key)
    if applied:
        if (
            scale <= 1.0
            or ulps <= 0
            or not finite_nonnegative(previous_scale)
            or not finite_nonnegative(previous_worst)
            or previous_worst < 1.0
            or repair.get("one_ulp_safety_above_equality_boundary")
            != (previous_worst == 1.0)
        ):
            raise Lfm2SpanSharedQkNormRefitError(
                "Refit receipt does not prove a minimal upward repair scale."
            )
    elif (
        scale != 1.0
        or ulps != 0
        or previous_scale is not None
        or previous_worst is not None
        or repair.get("one_ulp_safety_above_equality_boundary") is not False
        or worst_pre >= 1.0
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt changed its byte-stable exact-envelope path."
        )
    if (
        query_refit.get("target_norm_coordinatewise_envelope_sha256")
        != repair["original_query_coordinatewise_envelope_sha256"]
        or key_refit.get("target_norm_coordinatewise_envelope_sha256")
        != repair["original_key_coordinatewise_envelope_sha256"]
        or query_refit.get("installed_target_norm_sha256")
        != output["q_layernorm.weight"]
        or key_refit.get("installed_target_norm_sha256") != output["k_layernorm.weight"]
        or query_refit.get("pre_repair_maximum_direction_mean_square") != query_pre
        or key_refit.get("pre_repair_maximum_direction_mean_square") != key_pre
        or query_refit.get("preimage_maximum_direction_mean_square") != query_post
        or key_refit.get("preimage_maximum_direction_mean_square") != key_post
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt does not bind its original and installed Q/K norms."
        )
    ridge_value = normalized.get("rowspace_ridge")
    if not finite_nonnegative(ridge_value):
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt changed its non-negative row-space ridge."
        )

    def validate_envelope_and_solver(
        branch: Mapping[str, Any],
        *,
        branch_name: str,
        expected_rounded_hash: str,
    ) -> None:
        envelope = branch.get("coordinatewise_envelope")
        if not isinstance(envelope, dict):
            raise Lfm2SpanSharedQkNormRefitError(
                f"{branch_name} refit lacks its upward-rounded envelope receipt."
            )
        for key in (
            "source_absolute_max_float64_sha256",
            "positive_zero_floored_lower_bound_float64_sha256",
            "target_dtype_upward_rounded_envelope_sha256",
        ):
            _digest(envelope.get(key), f"{branch_name}.{key}")
        count_keys = (
            "zero_floored_coordinate_count",
            "positive_sub_tiny_source_coordinate_count",
            "upward_nextafter_coordinate_count",
        )
        if (
            envelope.get("rule")
            != "minimal-per-coordinate-upward-target-dtype-rounding"
            or envelope.get("target_dtype") != normalized.get("target_dtype")
            or not finite_nonnegative(envelope.get("positive_zero_floor"))
            or envelope.get("positive_zero_floor") <= 0.0
            or not finite_nonnegative(
                envelope.get("target_dtype_smallest_positive_subnormal")
            )
            or envelope.get("target_dtype_smallest_positive_subnormal") <= 0.0
            or any(
                isinstance(envelope.get(key), bool)
                or not isinstance(envelope.get(key), int)
                or envelope.get(key) < 0
                for key in count_keys
            )
            or envelope.get("upward_nextafter_steps_per_adjusted_coordinate") != 1
            or envelope.get("minimal_per_coordinate_target_dtype_ceiling_verified")
            is not True
            or envelope.get("rounded_envelope_is_not_below_float64_lower_bound")
            is not True
            or envelope.get("envelope_rounding_has_quality_gate_authority") is not False
            or envelope.get("target_dtype_upward_rounded_envelope_sha256")
            != expected_rounded_hash
        ):
            raise Lfm2SpanSharedQkNormRefitError(
                f"{branch_name} refit changed its minimal upward envelope rounding."
            )

        expected_method = (
            "zero-ridge-moore-penrose-minimum-frobenius-norm"
            if ridge_value == 0.0
            else "positive-ridge-row-gram-linear-solve"
        )
        for solver_key in (
            "analytic_precast_rowspace_solver",
            "fixed_native_residual_rowspace_solver",
        ):
            solver = branch.get(solver_key)
            if (
                not isinstance(solver, dict)
                or solver.get("method") != expected_method
                or solver.get("ridge") != ridge_value
            ):
                raise Lfm2SpanSharedQkNormRefitError(
                    f"{branch_name} refit changed its authenticated row-space solver."
                )
            if ridge_value == 0.0:
                if (
                    not finite_nonnegative(
                        solver.get("pseudoinverse_relative_tolerance")
                    )
                    or solver.get("pseudoinverse_relative_tolerance") <= 0.0
                    or solver.get("pseudoinverse_absolute_tolerance") != 0.0
                    or solver.get("minimum_frobenius_norm_solution") is not True
                ):
                    raise Lfm2SpanSharedQkNormRefitError(
                        f"{branch_name} refit changed its zero-ridge pseudoinverse."
                    )
            elif (
                solver.get("pseudoinverse_relative_tolerance") is not None
                or solver.get("pseudoinverse_absolute_tolerance") is not None
                or solver.get("minimum_frobenius_norm_solution") is not False
            ):
                raise Lfm2SpanSharedQkNormRefitError(
                    f"{branch_name} refit changed its positive-ridge solve."
                )

        if (
            branch.get(
                "analytic_precast_correction_constructed_in_cached_input_rowspace"
            )
            is not True
            or branch.get("final_target_dtype_prior_nullspace_preservation_claimed")
            is not False
            or branch.get("nullspace_leakage_diagnostics_have_install_authority")
            is not False
            or not finite_nonnegative(
                branch.get(
                    "analytic_precast_prior_nullspace_maximum_absolute_leakage_advisory"
                )
            )
            or not finite_nonnegative(
                branch.get(
                    "final_target_dtype_prior_nullspace_maximum_absolute_leakage_advisory"
                )
            )
            or "analytic_rowspace_nullspace_maximum_absolute_leakage" in branch
        ):
            raise Lfm2SpanSharedQkNormRefitError(
                f"{branch_name} refit overclaims final target-dtype nullspace preservation."
            )

    validate_envelope_and_solver(
        query_refit,
        branch_name="query",
        expected_rounded_hash=repair["original_query_coordinatewise_envelope_sha256"],
    )
    validate_envelope_and_solver(
        key_refit,
        branch_name="key",
        expected_rounded_hash=repair["original_key_coordinatewise_envelope_sha256"],
    )
    audit = normalized.get("operation_audit")
    zero_fields = (
        "parent_forward_calls",
        "parent_trace_recaptures",
        "model_generate_calls",
        "autograd_grad_calls",
        "backward_calls",
        "optimizer_steps",
        "training_steps",
        "parameters_pruned",
        "parameters_quantized",
        "parameters_sparsified",
        "parameter_sharing_sites",
        "parent_sidecars",
    )
    if not isinstance(audit, dict) or any(audit.get(key) != 0 for key in zero_fields):
        raise Lfm2SpanSharedQkNormRefitError(
            "Refit receipt violates the zero-training dense-fold operation contract."
        )
    return claimed


@dataclass(frozen=True, slots=True)
class Lfm2CompiledSpanSharedQkNormRefit:
    """Dense native Q/K plus shared norms and their authenticated receipt."""

    query_weight: Any = field(repr=False, compare=False)
    key_weight: Any = field(repr=False, compare=False)
    query_norm_weight: Any = field(repr=False, compare=False)
    key_norm_weight: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().contiguous().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().contiguous().clone(),
                "q_layernorm.weight": (
                    self.query_norm_weight.detach().cpu().contiguous().clone()
                ),
                "k_layernorm.weight": (
                    self.key_norm_weight.detach().cpu().contiguous().clone()
                ),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.report), allow_nan=False))

    def verify_authentication(self) -> str:
        state_hashes = {
            name: lfm2_span_shared_qk_norm_refit_tensor_sha256(value)
            for name, value in self.state_dict().items()
        }
        receipt_hashes = dict(self.report.get("refit_state_sha256", {}))
        if state_hashes != receipt_hashes:
            raise Lfm2SpanSharedQkNormRefitError(
                "Compiled Q/K refit tensors differ from their receipt."
            )
        return verify_lfm2_span_shared_qk_norm_refit_receipt(self.report)


def compile_lfm2_span_shared_qk_norm_refit(
    *,
    transported_query_weight: Any,
    transported_key_weight: Any,
    unchanged_value_weight: Any,
    unchanged_out_weight: Any,
    selected_query_source_norm_rows: Any,
    selected_key_source_norm_rows: Any,
    cached_child_normalized_input_rows: Any,
    cached_query_normalized_pre_rope_rows: Any,
    cached_key_normalized_pre_rope_rows: Any,
    target_query_norm_weight_template: Any,
    target_key_norm_weight_template: Any,
    child_norm_epsilon: float,
    ridge: float,
    selected_source_query_heads: Sequence[tuple[int, int]],
    selected_source_key_value_groups: Sequence[tuple[int, int]],
    parent_attention_compiler_result_sha256: str,
    diagnostic_child_normalized_input_rows: Any | None = None,
    diagnostic_query_normalized_pre_rope_rows: Any | None = None,
    diagnostic_key_normalized_pre_rope_rows: Any | None = None,
) -> Lfm2CompiledSpanSharedQkNormRefit:
    """Compile one deterministic heterogeneous-norm Q/K installation."""

    torch = _torch()
    epsilon = _finite_positive(child_norm_epsilon, "child_norm_epsilon")
    ridge_value = _finite_nonnegative(ridge, "ridge")
    parent_result = _digest(
        parent_attention_compiler_result_sha256,
        "parent_attention_compiler_result_sha256",
    )
    query_prior = _tensor(
        transported_query_weight, label="transported query weight", ndim=2
    )
    key_prior = _tensor(transported_key_weight, label="transported key weight", ndim=2)
    value = _tensor(unchanged_value_weight, label="unchanged value weight", ndim=2)
    out = _tensor(unchanged_out_weight, label="unchanged output weight", ndim=2)
    query_source_norm = _tensor(
        selected_query_source_norm_rows,
        label="selected query source norm rows",
        ndim=2,
    )
    key_source_norm = _tensor(
        selected_key_source_norm_rows,
        label="selected key source norm rows",
        ndim=2,
    )
    child_inputs = _tensor(
        cached_child_normalized_input_rows,
        label="cached child normalized input rows",
        ndim=2,
    )
    query_targets = _tensor(
        cached_query_normalized_pre_rope_rows,
        label="cached query normalized pre-RoPE rows",
        ndim=3,
    )
    key_targets = _tensor(
        cached_key_normalized_pre_rope_rows,
        label="cached key normalized pre-RoPE rows",
        ndim=3,
    )
    query_template = _tensor(
        target_query_norm_weight_template,
        label="target query norm template",
        ndim=1,
    )
    key_template = _tensor(
        target_key_norm_weight_template,
        label="target key norm template",
        ndim=1,
    )
    target_dtype = query_template.dtype
    q_heads = int(query_source_norm.shape[0])
    k_heads = int(key_source_norm.shape[0])
    head_dim = int(query_template.shape[0])
    query_lineage = _lineage(
        selected_source_query_heads,
        expected_count=q_heads,
        label="selected_source_query_heads",
    )
    key_lineage = _lineage(
        selected_source_key_value_groups,
        expected_count=k_heads,
        label="selected_source_key_value_groups",
    )
    if (
        key_template.dtype != target_dtype
        or int(key_template.shape[0]) != head_dim
        or query_prior.dtype != target_dtype
        or key_prior.dtype != target_dtype
        or value.dtype != target_dtype
        or out.dtype != target_dtype
        or tuple(query_source_norm.shape) != (q_heads, head_dim)
        or tuple(key_source_norm.shape) != (k_heads, head_dim)
        or tuple(query_targets.shape) != (int(child_inputs.shape[0]), q_heads, head_dim)
        or tuple(key_targets.shape) != (int(child_inputs.shape[0]), k_heads, head_dim)
        or tuple(query_prior.shape) != (q_heads * head_dim, int(child_inputs.shape[1]))
        or tuple(key_prior.shape) != (k_heads * head_dim, int(child_inputs.shape[1]))
    ):
        raise Lfm2SpanSharedQkNormRefitError(
            "Shared Q/K refit inputs differ from the target attention topology."
        )

    with torch.no_grad():
        query_envelope, query_envelope_report = _coordinatewise_absolute_envelope(
            query_source_norm,
            target_dtype=target_dtype,
            label="query",
        )
        key_envelope, key_envelope_report = _coordinatewise_absolute_envelope(
            key_source_norm,
            target_dtype=target_dtype,
            label="key",
        )
        query_norm, key_norm, representability_repair = (
            _minimal_common_representability_scale(
                query_envelope=query_envelope,
                key_envelope=key_envelope,
                query_targets=query_targets,
                key_targets=key_targets,
            )
        )
        query_weight, query_norm, query_report = _compile_branch(
            label="query normalized targets",
            transported_weight=query_prior,
            source_norm_rows=query_source_norm,
            normalized_targets=query_targets,
            child_inputs=child_inputs,
            original_envelope=query_envelope,
            envelope_report=query_envelope_report,
            target_norm=query_norm,
            pre_repair_maximum_direction_mean_square=representability_repair[
                "query_pre_repair_maximum_direction_mean_square"
            ],
            epsilon=epsilon,
            ridge=ridge_value,
        )
        key_weight, key_norm, key_report = _compile_branch(
            label="key normalized targets",
            transported_weight=key_prior,
            source_norm_rows=key_source_norm,
            normalized_targets=key_targets,
            child_inputs=child_inputs,
            original_envelope=key_envelope,
            envelope_report=key_envelope_report,
            target_norm=key_norm,
            pre_repair_maximum_direction_mean_square=representability_repair[
                "key_pre_repair_maximum_direction_mean_square"
            ],
            epsilon=epsilon,
            ridge=ridge_value,
        )
        diagnostics = _diagnostic_report(
            child_inputs=diagnostic_child_normalized_input_rows,
            query_targets=diagnostic_query_normalized_pre_rope_rows,
            key_targets=diagnostic_key_normalized_pre_rope_rows,
            query_weight=query_weight,
            key_weight=key_weight,
            query_norm=query_norm,
            key_norm=key_norm,
            epsilon=epsilon,
        )

    refit_state = {
        "q_proj.weight": query_weight,
        "k_proj.weight": key_weight,
        "q_layernorm.weight": query_norm,
        "k_layernorm.weight": key_norm,
    }
    unchanged_state = {
        "v_proj.weight": value,
        "out_proj.weight": out,
    }
    installed_state_sha256 = {
        name: lfm2_span_shared_qk_norm_refit_tensor_sha256(tensor)
        for name, tensor in {**refit_state, **unchanged_state}.items()
    }
    body = {
        "schema": LFM2_SPAN_SHARED_QK_NORM_REFIT_SCHEMA,
        "method": LFM2_SPAN_SHARED_QK_NORM_REFIT_METHOD,
        "parent_attention_compiler_result_sha256": parent_result,
        "selected_source_query_heads": [list(value) for value in query_lineage],
        "selected_source_key_value_groups": [list(value) for value in key_lineage],
        "child_input_rows_sha256": lfm2_span_shared_qk_norm_refit_tensor_sha256(
            child_inputs
        ),
        "query_normalized_pre_rope_rows_sha256": (
            lfm2_span_shared_qk_norm_refit_tensor_sha256(query_targets)
        ),
        "key_normalized_pre_rope_rows_sha256": (
            lfm2_span_shared_qk_norm_refit_tensor_sha256(key_targets)
        ),
        "transported_query_prior_sha256": (
            lfm2_span_shared_qk_norm_refit_tensor_sha256(query_prior)
        ),
        "transported_key_prior_sha256": (
            lfm2_span_shared_qk_norm_refit_tensor_sha256(key_prior)
        ),
        "selected_query_source_norm_rows_sha256": (
            lfm2_span_shared_qk_norm_refit_tensor_sha256(query_source_norm)
        ),
        "selected_key_source_norm_rows_sha256": (
            lfm2_span_shared_qk_norm_refit_tensor_sha256(key_source_norm)
        ),
        "selected_query_source_norm_rows_semantic_sha256": (
            lfm2_authenticated_tensor_sha256(query_source_norm)
        ),
        "selected_key_source_norm_rows_semantic_sha256": (
            lfm2_authenticated_tensor_sha256(key_source_norm)
        ),
        "target_norm_rule": (
            "coordinatewise-absolute-envelope-lower-bound-with-minimal-common-"
            "target-dtype-representability-scale"
        ),
        "coordinatewise_envelope_lower_bound_preserved": True,
        "heterogeneous_source_norms_averaged": False,
        "representability_repair": dict(representability_repair),
        "representability_repair_has_quality_gate_authority": False,
        "rmsnorm_preimage_formula": ("U=Y/gamma_star; Z=U*sqrt(epsilon/(1-mean(U^2)))"),
        "norm_compensated_prior_formula": ("Wbar=diag(gamma_source/gamma_star)@W0"),
        "nonnegative_prior_scale_formula": ("max(0,<X@Wbar.T,Z>_F/||X@Wbar.T||_F^2)"),
        "rowspace_residual_formula": (
            "ridge>0:deltaW.T=X.T@(X@X.T+ridge*I)^-1@R;ridge=0:deltaW.T=pinv(X)@R"
        ),
        "zero_ridge_uses_minimum_frobenius_norm_pseudoinverse": True,
        "analytic_precast_correction_constructed_in_cached_input_rowspace": True,
        "final_target_dtype_prior_nullspace_preservation_claimed": False,
        "nullspace_leakage_diagnostics_have_install_authority": False,
        "child_norm_epsilon": epsilon,
        "rowspace_ridge": ridge_value,
        "target_dtype": str(target_dtype).removeprefix("torch."),
        "fixed_native_residual_compensation_passes": 1,
        "query_refit": dict(query_report),
        "key_refit": dict(key_report),
        "diagnostic_evidence": dict(diagnostics),
        "installed_attention_state_hash_scheme": (
            LFM2_SPAN_SHARED_QK_NORM_REFIT_TENSOR_HASH_SCHEME
        ),
        "installed_attention_state_sha256": installed_state_sha256,
        "refit_state_sha256": {
            key: installed_state_sha256[key] for key in sorted(_REFIT_STATE_NAMES)
        },
        "unchanged_state_sha256": {
            key: installed_state_sha256[key] for key in sorted(_UNCHANGED_STATE_NAMES)
        },
        "dense_ordinary_target_tensors": True,
        "numeric_diagnostics_are_advisory_only": True,
        "numeric_diagnostics_have_install_authority": False,
        "candidate_selection_performed": False,
        "operation_audit": {
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "model_generate_calls": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "parameter_sharing_sites": 0,
            "parent_sidecars": 0,
        },
    }
    report = MappingProxyType({**body, "result_sha256": canonical_sha256(body)})
    result = Lfm2CompiledSpanSharedQkNormRefit(
        query_weight=query_weight,
        key_weight=key_weight,
        query_norm_weight=query_norm,
        key_norm_weight=key_norm,
        report=report,
    )
    result.verify_authentication()
    return result


__all__ = [
    "LFM2_SPAN_SHARED_QK_NORM_REFIT_METHOD",
    "LFM2_SPAN_SHARED_QK_NORM_REFIT_SCHEMA",
    "LFM2_SPAN_SHARED_QK_NORM_REFIT_TENSOR_HASH_SCHEME",
    "Lfm2CompiledSpanSharedQkNormRefit",
    "Lfm2SpanSharedQkNormRefitError",
    "compile_lfm2_span_shared_qk_norm_refit",
    "lfm2_span_shared_qk_norm_refit_tensor_sha256",
    "verify_lfm2_span_shared_qk_norm_refit_receipt",
]
