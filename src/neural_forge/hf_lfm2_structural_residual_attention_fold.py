"""Dense, zero-gradient structural attention folding for LFM2.

This module fills the addressing gap left by a real-medoid-only fold.  A
medoid is still used to make clustering deterministic, but it is *not* the
compiled Q/K/V operator.  Every source query and KV head is assigned exactly
once.  Cached score geometry from every assigned head is combined into a
target-head score barycenter, factored at the legal target head dimension,
and regressed into the already-instantiated target's Q/K projection shapes.
V is fitted from the full assigned source group and O is fitted against the
residual endpoint produced by the newly compiled child attention.

There is no model call, optimizer, backward pass, low-bit representation, or
packed parameter storage here.  The result is an ordinary dense floating
Q/K/V/O state for a smaller target topology.  Installation copies into the
existing target Parameters, preserving their identities and shapes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_activation_cluster import (
    GQAActivationClusterPlan,
    GatedMLPActivationClusterPlan,
    build_gqa_activation_cluster_plan,
    build_gated_mlp_activation_cluster_plan,
)
from neural_forge.hf_multi_anchor_cluster import (
    GatedMLPMultiAnchorPlan,
    build_gated_mlp_multi_anchor_cluster_plan,
)
from neural_forge.hf_gated_mlp_weight_signature import (
    GatedMlpWeightCompleteSignature,
    build_gated_mlp_weight_complete_signature,
    gated_mlp_weight_signature_tensor_sha256,
)


_SCHEMA = "neural-forge-lfm2-structural-residual-attention-fold-v1"
_ATTENTION_SCORE_PRESERVATION_SCHEMA = (
    "neural-forge-lfm2-attention-score-preservation-v1"
)
_PREPARED_MLP_BASIS_SCHEMA = (
    "neural-forge-lfm2-structural-residual-mlp-prepared-basis-v1"
)
_CACHED_VERIFIED_TRACE_MODE = "cached_verified"
_FULL_PREFIX_CANONICAL_TRACE_MODE = "full_prefix_canonical"
_SUPPORTED_ATTENTION_TRACE_MODES = frozenset(
    {
        _CACHED_VERIFIED_TRACE_MODE,
        _FULL_PREFIX_CANONICAL_TRACE_MODE,
    }
)
_STATE_KEYS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "out_proj.weight",
)


class Lfm2StructuralResidualAttentionFoldError(ValueError):
    """Cached evidence, target topology, or a structural invariant is invalid."""


class Lfm2StructuralResidualAttentionInstallError(RuntimeError):
    """An in-place target installation or rollback failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for structural residual-attention folding."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be finite and non-negative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be finite and non-negative."
        )
    return result


def _float_tensor(value: Any, name: str, *, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be a non-empty materialized real rank-{ndim} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} contains NaN or infinity."
        )
    return result


def _boolean_tensor(value: Any, name: str, *, shape: tuple[int, ...]) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or tuple(value.shape) != shape
        or value.dtype != torch.bool
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be a dense boolean tensor with shape {shape}."
        )
    return value.detach().to(device="cpu").contiguous()


def _sqrt_row_weights(
    value: Any | None,
    name: str,
    *,
    shape: tuple[int, int],
) -> Any:
    """Return positive square-root least-squares weights for aligned rows."""

    torch = _torch()
    if value is None:
        return torch.ones(shape, dtype=torch.float64)
    result = _float_tensor(value, name, ndim=2)
    if tuple(result.shape) != shape or not bool((result > 0.0).all().item()):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must have shape {shape} and contain only positive values."
        )
    return result


def _row_weighting_mode(
    supplied: Sequence[bool],
    name: str,
) -> str:
    """Describe whether aligned row weights came from the caller."""

    values = tuple(supplied)
    if not values:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} requires at least one case."
        )
    supplied_count = sum(values)
    if supplied_count == 0:
        return "uniform_default"
    if supplied_count == len(values):
        return "caller_supplied_positive_sqrt_row_weights"
    return "mixed_uniform_default_and_caller_supplied_positive_sqrt_row_weights"


def _dense_template(
    value: Any, name: str, *, ndim: int = 2
) -> tuple[Any, tuple[int, ...]]:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or value.dtype
        not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be a dense FP16/BF16/FP32/FP64 rank-{ndim} target tensor."
        )
    # Deliberately return only dtype and shape.  Blank values are never read.
    return value.dtype, tuple(int(size) for size in value.shape)


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a tensor.")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def lfm2_authenticated_tensor_sha256(value: Any) -> str:
    """Return the public tensor digest used by semantic LFM2 inventories."""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a tensor.")
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _state_sha256(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_tensor_sha256(values[name])))
    return digest.hexdigest()


def _template_sha256(spec: Mapping[str, tuple[tuple[int, ...], str]]) -> str:
    encoded = json.dumps(
        {name: [list(shape), dtype] for name, (shape, dtype) in sorted(spec.items())},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative_rmse(prediction: Any, target: Any) -> float:
    torch = _torch()
    error = float(torch.sqrt(torch.mean((prediction - target).square())).item())
    scale = float(torch.sqrt(torch.mean(target.square())).item())
    return error / max(scale, torch.finfo(torch.float64).eps)


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _apply_rope(value: Any, cosine: Any | None, sine: Any | None) -> Any:
    if cosine is None:
        return value
    return value * cosine.unsqueeze(1) + _rotate_half(value) * sine.unsqueeze(1)


def _undo_rope(value: Any, cosine: Any | None, sine: Any | None) -> Any:
    if cosine is None:
        return value
    # LFM2 rotary coordinates are orthogonal; this is R(theta)^T.
    return value * cosine.unsqueeze(1) - _rotate_half(value) * sine.unsqueeze(1)


def _rope_evidence_dtype(*values: Any, name: str) -> Any:
    """Return one exact row dtype or reject mixed/unsupported evidence."""

    torch = _torch()
    supported = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    dtypes = {value.dtype for value in values if isinstance(value, torch.Tensor)}
    if len(dtypes) != 1 or any(
        not isinstance(value, torch.Tensor) or value.dtype not in supported
        for value in values
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} linear/rotary rows must share one FP16/BF16/FP32/FP64 "
            "evidence dtype."
        )
    return next(iter(dtypes))


def _rope_low_precision_reconstruction_error_bound(
    evidence_dtype: Any,
) -> float | None:
    """Finite-operation envelope for cached FP16/BF16 norm+RoPE replay.

    Production traces are produced by CUDA while the closed-form compiler
    deliberately replays RMSNorm on CPU.  Six low-precision rounding
    frontiers separate those paths: direction cast and learned-weight
    product on CPU, versus direction cast, learned-weight product, rotary
    product, and rotary sum on CUDA.  The standard ``gamma_6`` forward-error
    envelope accounts for their higher-order interaction without admitting
    the O(1) residual produced by a wrong causal position.
    """

    torch = _torch()
    if evidence_dtype not in {torch.float16, torch.bfloat16}:
        return None
    unit_roundoff = float(torch.finfo(evidence_dtype).eps) / 2.0
    return (6.0 * unit_roundoff) / (1.0 - 6.0 * unit_roundoff)


def _rope_low_precision_geometry_error_bound(
    evidence_dtype: Any,
) -> float | None:
    """Tighter envelope for unit radius and shared Q/K position geometry."""

    torch = _torch()
    if evidence_dtype not in {torch.float16, torch.bfloat16}:
        return None
    return float(2.0 * torch.finfo(evidence_dtype).eps)


def _rope_pair_relative_error(
    normalized: Any,
    rotated: Any,
    reconstruction: Any,
) -> float:
    """Maximum scale-normalized error over legal rotary coordinate pairs."""

    torch = _torch()
    half = int(normalized.shape[-1]) // 2
    error = reconstruction - rotated
    error_pairs = torch.stack(
        (error[..., :half], error[..., half:]),
        dim=-1,
    )
    normalized_pairs = torch.stack(
        (normalized[..., :half], normalized[..., half:]),
        dim=-1,
    )
    rotated_pairs = torch.stack(
        (rotated[..., :half], rotated[..., half:]),
        dim=-1,
    )
    scale = torch.maximum(
        torch.linalg.vector_norm(normalized_pairs, dim=-1),
        torch.linalg.vector_norm(rotated_pairs, dim=-1),
    ).clamp_min(torch.finfo(torch.float64).tiny)
    value = torch.linalg.vector_norm(error_pairs, dim=-1) / scale
    return float(value.max().item())


def _validate_rope_reconstruction(
    normalized: Any,
    rotated: Any,
    cosine: Any,
    sine: Any,
    name: str,
    *,
    evidence_dtype: Any,
    raw_radius_maximum_deviation: float | None = None,
) -> None:
    """Require one legal rotation under a dtype-derived cache envelope."""

    torch = _torch()
    reconstruction = _apply_rope(normalized, cosine, sine)
    relative_rmse = _relative_rmse(reconstruction, rotated)
    maximum_pair_relative_error = _rope_pair_relative_error(
        normalized,
        rotated,
        reconstruction,
    )
    permitted_error = _rope_low_precision_reconstruction_error_bound(evidence_dtype)
    if permitted_error is None:
        legal = bool(torch.allclose(reconstruction, rotated, rtol=1e-5, atol=1e-7))
    else:
        legal = (
            math.isfinite(maximum_pair_relative_error)
            and maximum_pair_relative_error <= permitted_error
            and (
                raw_radius_maximum_deviation is None
                or (
                    math.isfinite(raw_radius_maximum_deviation)
                    and raw_radius_maximum_deviation <= permitted_error
                )
            )
        )
    if not legal:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} cached linear/rotary rows do not identify one shared legal "
            "RoPE transform "
            f"(evidence_dtype={evidence_dtype}, "
            f"relative_rmse={relative_rmse:.9g}, "
            f"maximum_pair_relative_error={maximum_pair_relative_error:.9g}, "
            f"raw_radius_maximum_deviation="
            f"{0.0 if raw_radius_maximum_deviation is None else raw_radius_maximum_deviation:.9g}, "
            f"permitted_error="
            f"{0.0 if permitted_error is None else permitted_error:.9g})."
        )


def _infer_rope_from_cached_rows(
    normalized: Any,
    rotated: Any,
    name: str,
    *,
    evidence_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """Recover the shared pairwise rotary transform from sealed Q/K rows."""

    torch = _torch()
    half = int(normalized.shape[-1]) // 2
    first = normalized[..., :half]
    second = normalized[..., half:]
    rotated_first = rotated[..., :half]
    rotated_second = rotated[..., half:]
    denominator = (first.square() + second.square()).sum(dim=1)
    floor = torch.finfo(torch.float64).eps
    if bool((denominator <= floor).any().item()):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} cannot infer RoPE at a zero-energy cached coordinate."
        )
    cosine_half = (first * rotated_first + second * rotated_second).sum(
        dim=1
    ) / denominator
    sine_half = (first * rotated_second - second * rotated_first).sum(
        dim=1
    ) / denominator
    magnitude = torch.sqrt(cosine_half.square() + sine_half.square())
    if bool((magnitude <= floor).any().item()):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} cannot infer RoPE at a degenerate cached coordinate pair."
        )
    raw_radius_maximum_deviation = float((magnitude - 1.0).abs().max().item())
    cosine_half = cosine_half / magnitude
    sine_half = sine_half / magnitude
    cosine = torch.cat((cosine_half, cosine_half), dim=-1).contiguous()
    sine = torch.cat((sine_half, sine_half), dim=-1).contiguous()
    _validate_rope_reconstruction(
        normalized,
        rotated,
        cosine,
        sine,
        name,
        evidence_dtype=evidence_dtype,
        raw_radius_maximum_deviation=raw_radius_maximum_deviation,
    )
    return cosine, sine


def _validate_shared_qk_rope(
    query_cosine: Any | None,
    query_sine: Any | None,
    key_cosine: Any | None,
    key_sine: Any | None,
    name: str,
    *,
    evidence_dtype: Any,
) -> None:
    """Cross-check pair structure, unit radius, and causal Q/K positions."""

    torch = _torch()
    values = (query_cosine, query_sine, key_cosine, key_sine)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} Q/K rows disagree on whether RoPE is active."
        )
    head_dim = int(query_cosine.shape[-1])
    half = head_dim // 2
    pair_structure_exact = all(
        torch.equal(value[..., :half], value[..., half:]) for value in values
    )
    query_length = int(query_cosine.shape[1])
    key_length = int(key_cosine.shape[1])
    if not pair_structure_exact or query_length > key_length:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} inferred Q/K RoPE violates duplicated-pair or causal-length "
            "structure."
        )
    query_radius_error = (
        query_cosine[..., :half].square() + query_sine[..., :half].square() - 1.0
    ).abs()
    key_radius_error = (
        key_cosine[..., :half].square() + key_sine[..., :half].square() - 1.0
    ).abs()
    maximum_radius_error = float(
        max(query_radius_error.max().item(), key_radius_error.max().item())
    )
    query_cosine_half = query_cosine[..., :half]
    query_sine_half = query_sine[..., :half]
    key_cosine_half = key_cosine[:, -query_length:, :half]
    key_sine_half = key_sine[:, -query_length:, :half]
    maximum_qk_position_chord = float(
        torch.sqrt(
            (query_cosine_half - key_cosine_half).square()
            + (query_sine_half - key_sine_half).square()
        )
        .max()
        .item()
    )
    permitted_error = _rope_low_precision_geometry_error_bound(evidence_dtype)
    if permitted_error is None:
        legal = bool(
            torch.allclose(
                query_cosine_half,
                key_cosine_half,
                rtol=1e-5,
                atol=1e-7,
            )
            and torch.allclose(
                query_sine_half,
                key_sine_half,
                rtol=1e-5,
                atol=1e-7,
            )
            and maximum_radius_error <= 1e-7
        )
    else:
        legal = (
            math.isfinite(maximum_radius_error)
            and maximum_radius_error <= permitted_error
            and math.isfinite(maximum_qk_position_chord)
            and maximum_qk_position_chord <= permitted_error
        )
    if not legal:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} inferred Q/K rotations do not identify the same legal "
            "causal positions "
            f"(evidence_dtype={evidence_dtype}, "
            f"maximum_radius_error={maximum_radius_error:.9g}, "
            f"maximum_qk_position_chord={maximum_qk_position_chord:.9g}, "
            f"permitted_error="
            f"{0.0 if permitted_error is None else permitted_error:.9g})."
        )


def _rms_norm(value: Any, weight: Any | None, epsilon: float) -> Any:
    if weight is None:
        return value
    variance = value.square().mean(dim=-1, keepdim=True)
    return value * _torch().rsqrt(variance + epsilon) * weight.reshape(1, 1, 1, -1)


def _cached_lfm2_rms_norm(
    value: Any,
    weight: Any | None,
    epsilon: float,
) -> Any:
    """Reconstruct the LFM2 runtime norm represented by cached low-precision rows.

    Transformers LFM2 computes the RMS statistic in FP32, casts the normalized
    direction back to the projection dtype, and only then multiplies the
    learned head-local weight.  Re-evaluating BF16/FP16 cache rows entirely in
    FP64 invents differences that are larger than their rotary quantization
    noise and can make one legal shared RoPE transform look inconsistent.

    Full-precision synthetic evidence keeps the algebraic FP64 path used by
    the structural compiler.  The low-precision branch deliberately mirrors
    the finite-precision LFM2 operation before returning detached FP64 rows for
    the closed-form solver.
    """

    torch = _torch()
    if weight is None:
        return value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    raw = value.detach().to(device="cpu").contiguous()
    raw_weight = weight.detach().to(device="cpu").contiguous()
    if raw.dtype not in {torch.float16, torch.bfloat16}:
        return _rms_norm(
            raw.to(dtype=torch.float64),
            raw_weight.to(dtype=torch.float64),
            epsilon,
        )
    direction = raw.to(dtype=torch.float32)
    variance = direction.square().mean(dim=-1, keepdim=True)
    direction = direction * torch.rsqrt(variance + epsilon)
    # Match Lfm2RMSNorm.forward: weight * hidden_states.to(input_dtype).
    normalized = raw_weight * direction.to(dtype=raw.dtype)
    return normalized.to(dtype=torch.float64).contiguous()


def _projection_target_from_runtime(
    runtime_rows: Any,
    anchor_linear_rows: Any,
    *,
    norm_weight: Any | None,
    epsilon: float,
    cosine: Any | None,
    sine: Any | None,
) -> Any:
    torch = _torch()
    unrotated = _undo_rope(runtime_rows, cosine, sine)
    if norm_weight is None:
        return unrotated
    safe_weight = torch.where(
        norm_weight.abs() > torch.finfo(torch.float64).eps,
        norm_weight,
        torch.ones_like(norm_weight),
    )
    direction = unrotated / safe_weight.reshape(1, 1, 1, -1)
    direction_rms = torch.sqrt(direction.square().mean(dim=-1, keepdim=True))
    direction = direction / direction_rms.clamp_min(torch.finfo(torch.float64).eps)
    anchor_rms = torch.sqrt(
        anchor_linear_rows.square().mean(dim=-1, keepdim=True)
    ).clamp_min(math.sqrt(epsilon))
    return direction * anchor_rms


@dataclass(frozen=True, slots=True)
class Lfm2PriorCorrectionReport:
    role: str
    rows: int
    input_width: int
    output_width: int
    underdetermined: bool
    prior_provided: bool
    prior_norm: float
    correction_norm: float
    update_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "rows": self.rows,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "underdetermined": self.underdetermined,
            "prior_provided": self.prior_provided,
            "prior_norm": self.prior_norm,
            "correction_norm": self.correction_norm,
            "update_scale": self.update_scale,
            "objective": "source_derived_prior_plus_closed_form_residual_correction",
        }


def _fit_linear(
    input_rows: Any,
    target_rows: Any,
    *,
    role: str,
    ridge: float,
    prior: Any | None,
    max_prior_delta_ratio: float | None,
) -> tuple[Any, Lfm2PriorCorrectionReport]:
    """Fit ``target = input @ weight.T`` around a source-derived prior."""

    torch = _torch()
    x = input_rows.reshape(-1, int(input_rows.shape[-1]))
    y = target_rows.reshape(-1, int(target_rows.shape[-1]))
    if int(x.shape[0]) != int(y.shape[0]):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Linear fit inputs and targets have different row counts."
        )
    ridge = _nonnegative_float(ridge, "ridge")
    underdetermined = int(x.shape[0]) < int(x.shape[1])
    if prior is None:
        if underdetermined:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{role} has only {int(x.shape[0])} rows for input width "
                f"{int(x.shape[1])}; a source-derived target-shaped prior is "
                "required so the uncached nullspace is not reset to zero."
            )
        prior_rows = torch.zeros(
            (int(y.shape[1]), int(x.shape[1])), dtype=torch.float64
        )
    else:
        prior_rows = _float_tensor(prior, f"{role}_weight_prior", ndim=2)
        if tuple(prior_rows.shape) != (int(y.shape[1]), int(x.shape[1])):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{role}_weight_prior must have shape "
                f"{(int(y.shape[1]), int(x.shape[1]))}."
            )
    # With no prior the old expression ``y - x @ zeros.T`` is exactly
    # ``y``.  Avoiding that zero GEMM preserves the closed-form objective and
    # the resulting bytes while removing one output-sized matrix multiply
    # from every unregularized-prior projection.
    residual = y if prior is None else y - x @ prior_rows.T
    if ridge == 0.0:
        solution = torch.linalg.lstsq(x, residual, driver="gelsd").solution
    elif int(x.shape[1]) > int(x.shape[0]):
        system = x @ x.T + ridge * torch.eye(int(x.shape[0]), dtype=torch.float64)
        try:
            dual = torch.linalg.solve(system, residual)
        except RuntimeError:
            dual = torch.linalg.lstsq(system, residual, driver="gelsd").solution
        solution = x.T @ dual
    else:
        system = x.T @ x + ridge * torch.eye(int(x.shape[1]), dtype=torch.float64)
        right = x.T @ residual
        try:
            solution = torch.linalg.solve(system, right)
        except RuntimeError:
            solution = torch.linalg.lstsq(system, right, driver="gelsd").solution
    if not bool(torch.isfinite(solution).all().item()):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Closed-form projection fit produced non-finite coefficients."
        )
    correction = solution.T.contiguous()
    prior_norm = float(torch.linalg.vector_norm(prior_rows).item())
    correction_norm = float(torch.linalg.vector_norm(correction).item())
    update_scale = 1.0
    if (
        prior is not None
        and max_prior_delta_ratio is not None
        and correction_norm > 0.0
    ):
        allowed = max_prior_delta_ratio * max(
            prior_norm, torch.finfo(torch.float64).tiny
        )
        update_scale = min(1.0, allowed / correction_norm)
        if update_scale < 1.0:
            correction = correction * update_scale
            correction_norm *= update_scale
    weight = prior_rows + correction
    return weight, Lfm2PriorCorrectionReport(
        role=role,
        rows=int(x.shape[0]),
        input_width=int(x.shape[1]),
        output_width=int(y.shape[1]),
        underdetermined=underdetermined,
        prior_provided=prior is not None,
        prior_norm=prior_norm,
        correction_norm=correction_norm,
        update_scale=update_scale,
    )


def _fit_affine(
    input_rows: Any,
    target_rows: Any,
    *,
    role: str,
    ridge: float,
    weight_prior: Any | None,
    bias_prior: Any | None,
    max_prior_delta_ratio: float | None,
    fit_bias: bool,
) -> tuple[Any, Any | None, Lfm2PriorCorrectionReport]:
    """Fit an affine projection around one source-derived target-shaped prior."""

    torch = _torch()
    if not fit_bias:
        if bias_prior is not None:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{role}_bias_prior was supplied for a bias-free target."
            )
        weight, report = _fit_linear(
            input_rows,
            target_rows,
            role=role,
            ridge=ridge,
            prior=weight_prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        return weight, None, report
    if (weight_prior is None) != (bias_prior is None):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{role} weight and bias priors must be supplied together."
        )
    x = input_rows.reshape(-1, int(input_rows.shape[-1]))
    augmented = torch.cat(
        (x, torch.ones((int(x.shape[0]), 1), dtype=torch.float64)),
        dim=1,
    )
    joint_prior = None
    if weight_prior is not None:
        normalized_weight = _float_tensor(weight_prior, f"{role}_weight_prior", ndim=2)
        normalized_bias = _float_tensor(bias_prior, f"{role}_bias_prior", ndim=1)
        if int(normalized_weight.shape[0]) != int(normalized_bias.shape[0]):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{role} weight and bias priors have different output widths."
            )
        joint_prior = torch.cat(
            (normalized_weight, normalized_bias.unsqueeze(1)), dim=1
        )
    joint, report = _fit_linear(
        augmented,
        target_rows,
        role=role,
        ridge=ridge,
        prior=joint_prior,
        max_prior_delta_ratio=max_prior_delta_ratio,
    )
    return joint[:, :-1].contiguous(), joint[:, -1].contiguous(), report


def _center_scores(scores: Any, mask: Any | None) -> Any:
    torch = _torch()
    if mask is None:
        return scores - scores.mean(dim=-1, keepdim=True)
    expanded = mask.unsqueeze(1)
    count = expanded.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (scores * expanded).sum(dim=-1, keepdim=True) / count
    return torch.where(expanded, scores - mean, torch.zeros_like(scores))


def _weighted_group_rows(
    rows: Any,
    assignment: tuple[int, ...],
    target_count: int,
    source_weights: Any,
) -> tuple[Any, tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]:
    torch = _torch()
    outputs = []
    groups: list[tuple[int, ...]] = []
    weights: list[tuple[float, ...]] = []
    for target in range(target_count):
        members = tuple(
            source
            for source, destination in enumerate(assignment)
            if destination == target
        )
        if not members:
            raise Lfm2StructuralResidualAttentionFoldError(
                "A target group has no assigned source heads."
            )
        index = torch.tensor(members, dtype=torch.long)
        alpha = source_weights.index_select(0, index).clamp_min(0.0)
        if float(alpha.sum().item()) <= torch.finfo(torch.float64).eps:
            alpha = torch.ones_like(alpha)
        alpha = alpha / alpha.sum()
        selected = rows.index_select(1, index)
        outputs.append(
            (selected * alpha.reshape(1, -1, *([1] * (rows.ndim - 2)))).sum(dim=1)
        )
        groups.append(members)
        weights.append(tuple(float(value) for value in alpha))
    return torch.stack(outputs, dim=1), tuple(groups), tuple(weights)


def _factor_score_barycenter(
    centered_scores: Any,
    key_anchor: Any,
    *,
    target_query_heads: int,
    target_key_value_heads: int,
    head_dim: int,
    query_sqrt_weights: Any,
) -> tuple[Any, Any]:
    """Weighted low-rank score target with deterministic anchor orientation."""

    torch = _torch()
    batch, _heads, query_length, key_length = map(int, centered_scores.shape)
    if tuple(query_sqrt_weights.shape) != (batch, query_length) or not bool(
        (query_sqrt_weights > 0.0).all().item()
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "query_sqrt_weights do not align with centered score rows."
        )
    offsets = target_query_heads // target_key_value_heads
    query = torch.zeros(
        (batch, target_query_heads, query_length, head_dim), dtype=torch.float64
    )
    key = torch.zeros(
        (batch, target_key_value_heads, key_length, head_dim), dtype=torch.float64
    )
    for batch_index in range(batch):
        for group in range(target_key_value_heads):
            start = group * offsets
            stop = start + offsets
            matrix = centered_scores[batch_index, start:stop].reshape(
                offsets * query_length, key_length
            ) * math.sqrt(head_dim)
            row_sqrt_weights = query_sqrt_weights[batch_index].repeat(offsets)
            weighted_matrix = matrix * row_sqrt_weights.unsqueeze(1)
            left, singular, right_h = torch.linalg.svd(
                weighted_matrix,
                full_matrices=False,
            )
            rank = min(head_dim, int(singular.numel()))
            left_factor = torch.zeros(
                (offsets * query_length, head_dim), dtype=torch.float64
            )
            right_factor = torch.zeros((key_length, head_dim), dtype=torch.float64)
            root = torch.sqrt(singular[:rank].clamp_min(0.0))
            left_factor[:, :rank] = left[:, :rank] * root
            right_factor[:, :rank] = right_h[:rank].T * root
            left_factor /= row_sqrt_weights.unsqueeze(1)

            anchor = key_anchor[batch_index, group]
            cross = right_factor.T @ anchor
            orient_left, _orient_s, orient_right_h = torch.linalg.svd(
                cross, full_matrices=False
            )
            rotation = orient_left @ orient_right_h
            left_factor = left_factor @ rotation
            right_factor = right_factor @ rotation
            query[batch_index, start:stop] = left_factor.reshape(
                offsets, query_length, head_dim
            )
            key[batch_index, group] = right_factor
    return query, key


def _optional_score_threshold(
    value: Any | None,
    name: str,
    *,
    unit_interval: bool,
) -> float | None:
    if value is None:
        return None
    result = _nonnegative_float(value, name)
    if unit_interval and result > 1.0:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be within [0, 1] or None."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2AttentionScorePreservationThresholds:
    """Optional held-out acceptance thresholds for functional Q/K scores.

    Error thresholds apply to the worst eligible causal row.  Fraction
    thresholds apply to all eligible comparisons.  ``None`` disables one
    threshold without changing the measured metric.
    """

    maximum_softmax_fisher_error: float | None = None
    maximum_js_divergence: float | None = None
    minimum_top1_key_agreement_fraction: float | None = None
    minimum_top_k_pair_order_agreement_fraction: float | None = None
    minimum_winner_margin_sign_agreement_fraction: float | None = None
    minimum_relative_margin_floor_retention_fraction: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_softmax_fisher_error",
            _optional_score_threshold(
                self.maximum_softmax_fisher_error,
                "maximum_softmax_fisher_error",
                unit_interval=False,
            ),
        )
        object.__setattr__(
            self,
            "maximum_js_divergence",
            _optional_score_threshold(
                self.maximum_js_divergence,
                "maximum_js_divergence",
                unit_interval=True,
            ),
        )
        for name in (
            "minimum_top1_key_agreement_fraction",
            "minimum_top_k_pair_order_agreement_fraction",
            "minimum_winner_margin_sign_agreement_fraction",
            "minimum_relative_margin_floor_retention_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _optional_score_threshold(
                    getattr(self, name),
                    name,
                    unit_interval=True,
                ),
            )

    def as_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Lfm2AttentionScorePreservationReport:
    """Canonical score-only acceptance report for held-out Q/K behavior."""

    case_count: int
    total_row_count: int
    eligible_row_count: int
    excluded_empty_row_count: int
    excluded_singleton_row_count: int
    top_k_pair_comparison_count: int
    excluded_reference_tie_pair_count: int
    winner_margin_comparison_count: int
    excluded_reference_tie_margin_count: int
    top_k: int
    relative_margin_floor: float
    softmax_fisher_error_mean: float
    softmax_fisher_error_worst: float
    js_divergence_mean: float
    js_divergence_worst: float
    top1_key_agreement_fraction: float
    top_k_pair_order_agreement_fraction: float
    winner_margin_sign_agreement_fraction: float
    relative_margin_floor_retention_fraction: float
    reference_score_bundle_sha256: str
    candidate_score_bundle_sha256: str
    key_mask_bundle_sha256: str
    case_reports: tuple[tuple[tuple[str, Any], ...], ...]
    thresholds: Lfm2AttentionScorePreservationThresholds
    checks: tuple[tuple[str, bool | None], ...]

    @property
    def accepted(self) -> bool:
        return all(value is not False for _name, value in self.checks)

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "softmax_fisher_error_mean": self.softmax_fisher_error_mean,
            "softmax_fisher_error_worst": self.softmax_fisher_error_worst,
            "js_divergence_mean": self.js_divergence_mean,
            "js_divergence_worst": self.js_divergence_worst,
            "top1_key_agreement_fraction": (self.top1_key_agreement_fraction),
            "top_k_pair_order_agreement_fraction": (
                self.top_k_pair_order_agreement_fraction
            ),
            "winner_margin_sign_agreement_fraction": (
                self.winner_margin_sign_agreement_fraction
            ),
            "relative_margin_floor_retention_fraction": (
                self.relative_margin_floor_retention_fraction
            ),
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": _ATTENTION_SCORE_PRESERVATION_SCHEMA,
            "case_count": self.case_count,
            "total_row_count": self.total_row_count,
            "eligible_row_count": self.eligible_row_count,
            "excluded_empty_row_count": self.excluded_empty_row_count,
            "excluded_singleton_row_count": (self.excluded_singleton_row_count),
            "top_k_pair_comparison_count": (self.top_k_pair_comparison_count),
            "excluded_reference_tie_pair_count": (
                self.excluded_reference_tie_pair_count
            ),
            "winner_margin_comparison_count": (self.winner_margin_comparison_count),
            "excluded_reference_tie_margin_count": (
                self.excluded_reference_tie_margin_count
            ),
            "top_k": self.top_k,
            "relative_margin_floor": self.relative_margin_floor,
            "metrics": self.metrics,
            "thresholds": self.thresholds.as_dict(),
            "checks": dict(self.checks),
            "accepted": self.accepted,
            "reference_score_bundle_sha256": (self.reference_score_bundle_sha256),
            "candidate_score_bundle_sha256": (self.candidate_score_bundle_sha256),
            "key_mask_bundle_sha256": self.key_mask_bundle_sha256,
            "cases": [dict(value) for value in self.case_reports],
            "score_tensors_only": True,
            "numeric_weight_matching_used": False,
            "model_forward_calls": 0,
            "parent_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "masked_keys_excluded": True,
            "empty_and_singleton_rows_excluded": True,
            "softmax_constant_shift_invariant": True,
            "js_divergence_normalized_by_log_two": True,
        }

    @property
    def report_sha256(self) -> str:
        encoded = json.dumps(
            self._payload(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "report_sha256": self.report_sha256,
        }


def _score_case_sequence(value: Any, name: str) -> tuple[Any, ...]:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be a score tensor or a non-empty score-tensor sequence."
        )
    return tuple(value)


def _score_mask_sequence(
    value: Any | None,
    *,
    case_count: int,
) -> tuple[Any | None, ...]:
    torch = _torch()
    if value is None:
        return (None,) * case_count
    if isinstance(value, torch.Tensor):
        if case_count != 1:
            raise Lfm2StructuralResidualAttentionFoldError(
                "One key mask can only accompany one score case."
            )
        return (value,)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != case_count
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "key_masks must be None, one mask, or one mask per score case."
        )
    return tuple(value)


def _attention_score_case_tensor(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 4
        or any(int(size) <= 0 for size in value.shape)
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be a non-empty materialized real "
            "[batch, heads, queries, keys] tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if bool(torch.isnan(result).any().item()):
        raise Lfm2StructuralResidualAttentionFoldError(f"{name} contains NaN.")
    return result


def _attention_score_case_mask(
    value: Any | None,
    *,
    score_shape: tuple[int, ...],
    name: str,
) -> Any:
    torch = _torch()
    batch, heads, queries, keys = score_shape
    if value is None:
        return torch.ones(score_shape, dtype=torch.bool)
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.bool
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must be a dense boolean key mask."
        )
    mask = value.detach().to(device="cpu").contiguous()
    if tuple(mask.shape) == (batch, queries, keys):
        mask = mask.unsqueeze(1)
    if tuple(mask.shape) not in {
        score_shape,
        (batch, 1, queries, keys),
    }:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must have shape [batch, queries, keys], "
            "[batch, 1, queries, keys], or the exact score shape."
        )
    return mask.expand(batch, heads, queries, keys).contiguous()


def _score_bundle_sha256(values: Sequence[Any], prefix: str) -> str:
    return _state_sha256(
        {f"{prefix}.{index:04d}": value for index, value in enumerate(values)}
    )


def _score_row_tolerance(reference: Any, candidate: Any) -> float:
    torch = _torch()
    scale = max(
        1.0,
        float(reference.abs().max().item()),
        float(candidate.abs().max().item()),
    )
    return (
        torch.finfo(torch.float64).eps * max(2, int(reference.numel())) * 16.0 * scale
    )


def _deterministic_descending_indices(value: Any) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(int(value.numel())),
            key=lambda index: (-float(value[index].item()), index),
        )
    )


def evaluate_lfm2_structural_attention_score_preservation(
    reference_scores: Any,
    candidate_scores: Any,
    *,
    key_masks: Any | None = None,
    top_k: int = 8,
    relative_margin_floor: float = 0.5,
    thresholds: Lfm2AttentionScorePreservationThresholds | None = None,
) -> Lfm2AttentionScorePreservationReport:
    """Evaluate held-out Q/K behavior using scores, never numeric weights.

    Inputs are one tensor or parallel sequences of tensors shaped
    ``[batch, heads, queries, keys]``.  Key lengths may differ between cases.
    Masks exclude unavailable causal keys.  Rows with fewer than two valid
    keys are counted but excluded from every metric.
    """

    torch = _torch()
    requested_top_k = _positive_int(top_k, "top_k")
    if requested_top_k < 2:
        raise Lfm2StructuralResidualAttentionFoldError(
            "top_k must be at least two for order and margin evaluation."
        )
    margin_floor = _nonnegative_float(
        relative_margin_floor,
        "relative_margin_floor",
    )
    if margin_floor > 1.0:
        raise Lfm2StructuralResidualAttentionFoldError(
            "relative_margin_floor must be within [0, 1]."
        )
    policy = thresholds or Lfm2AttentionScorePreservationThresholds()
    if not isinstance(policy, Lfm2AttentionScorePreservationThresholds):
        raise TypeError("thresholds must be Lfm2AttentionScorePreservationThresholds.")
    references_raw = _score_case_sequence(
        reference_scores,
        "reference_scores",
    )
    candidates_raw = _score_case_sequence(
        candidate_scores,
        "candidate_scores",
    )
    if len(references_raw) != len(candidates_raw):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Reference and candidate score case counts differ."
        )
    masks_raw = _score_mask_sequence(
        key_masks,
        case_count=len(references_raw),
    )

    references: list[Any] = []
    candidates: list[Any] = []
    masks: list[Any] = []
    fisher_rows: list[float] = []
    js_rows: list[float] = []
    top1_matches = 0
    pair_matches = 0
    pair_count = 0
    tied_pairs = 0
    winner_sign_matches = 0
    winner_floor_matches = 0
    winner_count = 0
    tied_winner_margins = 0
    total_rows = 0
    eligible_rows = 0
    empty_rows = 0
    singleton_rows = 0
    case_reports: list[tuple[tuple[str, Any], ...]] = []

    for case_index, (raw_reference, raw_candidate, raw_mask) in enumerate(
        zip(references_raw, candidates_raw, masks_raw, strict=True)
    ):
        reference = _attention_score_case_tensor(
            raw_reference,
            f"reference_scores[{case_index}]",
        )
        candidate = _attention_score_case_tensor(
            raw_candidate,
            f"candidate_scores[{case_index}]",
        )
        if tuple(candidate.shape) != tuple(reference.shape):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"Score case {case_index} reference/candidate shapes differ."
            )
        mask = _attention_score_case_mask(
            raw_mask,
            score_shape=tuple(int(value) for value in reference.shape),
            name=f"key_masks[{case_index}]",
        )
        if not bool(torch.isfinite(reference[mask]).all().item()) or not bool(
            torch.isfinite(candidate[mask]).all().item()
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"Score case {case_index} has non-finite values at valid keys."
            )
        references.append(reference)
        candidates.append(candidate)
        masks.append(mask)

        flat_reference = reference.reshape(-1, int(reference.shape[-1]))
        flat_candidate = candidate.reshape_as(flat_reference)
        flat_mask = mask.reshape_as(flat_reference)
        case_eligible = 0
        case_empty = 0
        case_singleton = 0
        case_pairs_before = pair_count
        case_winner_before = winner_count
        total_rows += int(flat_reference.shape[0])
        for row_index in range(int(flat_reference.shape[0])):
            valid = torch.nonzero(
                flat_mask[row_index],
                as_tuple=False,
            ).reshape(-1)
            valid_count = int(valid.numel())
            if valid_count == 0:
                empty_rows += 1
                case_empty += 1
                continue
            if valid_count == 1:
                singleton_rows += 1
                case_singleton += 1
                continue
            eligible_rows += 1
            case_eligible += 1
            reference_row = flat_reference[row_index].index_select(0, valid)
            candidate_row = flat_candidate[row_index].index_select(0, valid)
            tolerance = _score_row_tolerance(reference_row, candidate_row)

            reference_probability = torch.softmax(
                reference_row,
                dim=-1,
                dtype=torch.float64,
            )
            candidate_probability = torch.softmax(
                candidate_row,
                dim=-1,
                dtype=torch.float64,
            )
            delta = candidate_row - reference_row
            mean_delta = (reference_probability * delta).sum()
            fisher = (
                (reference_probability * delta.square()).sum() - mean_delta.square()
            ).clamp_min(0.0)
            fisher_rows.append(float(fisher.item()))

            midpoint = 0.5 * (reference_probability + candidate_probability)
            tiny = torch.finfo(torch.float64).tiny
            reference_term = torch.where(
                reference_probability > 0.0,
                reference_probability
                * (
                    torch.log(reference_probability.clamp_min(tiny))
                    - torch.log(midpoint.clamp_min(tiny))
                ),
                torch.zeros_like(reference_probability),
            )
            candidate_term = torch.where(
                candidate_probability > 0.0,
                candidate_probability
                * (
                    torch.log(candidate_probability.clamp_min(tiny))
                    - torch.log(midpoint.clamp_min(tiny))
                ),
                torch.zeros_like(candidate_probability),
            )
            js = 0.5 * (reference_term.sum() + candidate_term.sum()) / math.log(2.0)
            js_rows.append(float(js.clamp_min(0.0).item()))

            order = _deterministic_descending_indices(reference_row)
            candidate_order = _deterministic_descending_indices(candidate_row)
            reference_winner_score = float(reference_row[order[0]].item())
            reference_winners = {
                index
                for index in order
                if abs(float(reference_row[index].item()) - reference_winner_score)
                <= tolerance
            }
            top1_matches += int(candidate_order[0] in reference_winners)
            selected = order[: min(requested_top_k, valid_count)]
            for left_offset, left in enumerate(selected):
                for right in selected[left_offset + 1 :]:
                    reference_difference = float(
                        (reference_row[left] - reference_row[right]).item()
                    )
                    if abs(reference_difference) <= tolerance:
                        tied_pairs += 1
                        continue
                    candidate_difference = float(
                        (candidate_row[left] - candidate_row[right]).item()
                    )
                    reference_sign = 1 if reference_difference > 0.0 else -1
                    candidate_sign = (
                        1
                        if candidate_difference > tolerance
                        else -1
                        if candidate_difference < -tolerance
                        else 0
                    )
                    pair_count += 1
                    pair_matches += int(candidate_sign == reference_sign)

            winner = selected[0]
            for competitor in selected[1:]:
                reference_margin = float(
                    (reference_row[winner] - reference_row[competitor]).item()
                )
                if reference_margin <= tolerance:
                    tied_winner_margins += 1
                    continue
                candidate_margin = float(
                    (candidate_row[winner] - candidate_row[competitor]).item()
                )
                winner_count += 1
                winner_sign_matches += int(candidate_margin > tolerance)
                winner_floor_matches += int(
                    candidate_margin + tolerance >= margin_floor * reference_margin
                )

        case_reports.append(
            tuple(
                sorted(
                    {
                        "case_index": case_index,
                        "score_shape": [int(value) for value in reference.shape],
                        "total_row_count": int(flat_reference.shape[0]),
                        "eligible_row_count": case_eligible,
                        "excluded_empty_row_count": case_empty,
                        "excluded_singleton_row_count": case_singleton,
                        "top_k_pair_comparison_count": (pair_count - case_pairs_before),
                        "winner_margin_comparison_count": (
                            winner_count - case_winner_before
                        ),
                    }.items()
                )
            )
        )

    if eligible_rows <= 0:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Held-out attention score evidence has no row with two valid keys."
        )
    if len(fisher_rows) != eligible_rows or len(js_rows) != eligible_rows:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Attention score metric row accounting is inconsistent."
        )

    top1_fraction = top1_matches / eligible_rows
    pair_fraction = 1.0 if pair_count == 0 else pair_matches / pair_count
    winner_sign_fraction = (
        1.0 if winner_count == 0 else winner_sign_matches / winner_count
    )
    winner_floor_fraction = (
        1.0 if winner_count == 0 else winner_floor_matches / winner_count
    )
    metrics = {
        "softmax_fisher_error_mean": math.fsum(fisher_rows) / len(fisher_rows),
        "softmax_fisher_error_worst": max(fisher_rows),
        "js_divergence_mean": math.fsum(js_rows) / len(js_rows),
        "js_divergence_worst": max(js_rows),
        "top1_key_agreement_fraction": top1_fraction,
        "top_k_pair_order_agreement_fraction": pair_fraction,
        "winner_margin_sign_agreement_fraction": winner_sign_fraction,
        "relative_margin_floor_retention_fraction": winner_floor_fraction,
    }
    checks = {
        "maximum_softmax_fisher_error": (
            None
            if policy.maximum_softmax_fisher_error is None
            else metrics["softmax_fisher_error_worst"]
            <= policy.maximum_softmax_fisher_error
        ),
        "maximum_js_divergence": (
            None
            if policy.maximum_js_divergence is None
            else metrics["js_divergence_worst"] <= policy.maximum_js_divergence
        ),
        "minimum_top1_key_agreement_fraction": (
            None
            if policy.minimum_top1_key_agreement_fraction is None
            else top1_fraction >= policy.minimum_top1_key_agreement_fraction
        ),
        "minimum_top_k_pair_order_agreement_fraction": (
            None
            if policy.minimum_top_k_pair_order_agreement_fraction is None
            else pair_fraction >= policy.minimum_top_k_pair_order_agreement_fraction
        ),
        "minimum_winner_margin_sign_agreement_fraction": (
            None
            if policy.minimum_winner_margin_sign_agreement_fraction is None
            else winner_sign_fraction
            >= policy.minimum_winner_margin_sign_agreement_fraction
        ),
        "minimum_relative_margin_floor_retention_fraction": (
            None
            if policy.minimum_relative_margin_floor_retention_fraction is None
            else winner_floor_fraction
            >= policy.minimum_relative_margin_floor_retention_fraction
        ),
    }
    return Lfm2AttentionScorePreservationReport(
        case_count=len(references),
        total_row_count=total_rows,
        eligible_row_count=eligible_rows,
        excluded_empty_row_count=empty_rows,
        excluded_singleton_row_count=singleton_rows,
        top_k_pair_comparison_count=pair_count,
        excluded_reference_tie_pair_count=tied_pairs,
        winner_margin_comparison_count=winner_count,
        excluded_reference_tie_margin_count=tied_winner_margins,
        top_k=requested_top_k,
        relative_margin_floor=margin_floor,
        softmax_fisher_error_mean=metrics["softmax_fisher_error_mean"],
        softmax_fisher_error_worst=metrics["softmax_fisher_error_worst"],
        js_divergence_mean=metrics["js_divergence_mean"],
        js_divergence_worst=metrics["js_divergence_worst"],
        top1_key_agreement_fraction=top1_fraction,
        top_k_pair_order_agreement_fraction=pair_fraction,
        winner_margin_sign_agreement_fraction=winner_sign_fraction,
        relative_margin_floor_retention_fraction=winner_floor_fraction,
        reference_score_bundle_sha256=_score_bundle_sha256(
            references,
            "reference",
        ),
        candidate_score_bundle_sha256=_score_bundle_sha256(
            candidates,
            "candidate",
        ),
        key_mask_bundle_sha256=_score_bundle_sha256(masks, "mask"),
        case_reports=tuple(case_reports),
        thresholds=policy,
        checks=tuple(sorted(checks.items())),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CachedAttentionFoldCase:
    """One aligned cached score and residual-endpoint case.

    Parent Q/K are post-normalization and post-RoPE.  ``*_linear`` tensors are
    the pre-normalization projection outputs.  Target inputs are the already
    chosen smaller child's Q and KV projection inputs; their hidden width may
    differ from the parent's.  ``parent_post_attention_residual_target_space``
    is the complete parent attention residual endpoint expressed in the
    target residual space, not the isolated O-branch delta.
    """

    target_query_input: Any
    target_key_value_input: Any
    parent_query_linear: Any
    parent_key_linear: Any
    parent_value: Any
    parent_query: Any
    parent_key: Any
    parent_context: Any
    parent_post_attention_residual_target_space: Any
    target_pre_attention_residual: Any
    query_cosine: Any | None = None
    query_sine: Any | None = None
    key_cosine: Any | None = None
    key_sine: Any | None = None
    key_mask: Any | None = None
    parent_contribution: Any | None = None
    query_row_sqrt_weights: Any | None = None
    key_value_row_sqrt_weights: Any | None = None
    parent_probabilities: Any | None = None


@dataclass(frozen=True, slots=True)
class Lfm2StructuralAttentionTopology:
    source_query_heads: int
    source_key_value_heads: int
    target_query_heads: int
    target_key_value_heads: int
    head_dim: int
    source_hidden_size: int
    target_hidden_size: int
    target_output_size: int

    @property
    def source_query_heads_per_group(self) -> int:
        return self.source_query_heads // self.source_key_value_heads

    @property
    def target_query_heads_per_group(self) -> int:
        return self.target_query_heads // self.target_key_value_heads

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "source_query_heads": self.source_query_heads,
            "source_key_value_heads": self.source_key_value_heads,
            "target_query_heads": self.target_query_heads,
            "target_key_value_heads": self.target_key_value_heads,
            "head_dim": self.head_dim,
            "source_hidden_size": self.source_hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "target_output_size": self.target_output_size,
            "source_query_heads_per_group": self.source_query_heads_per_group,
            "target_query_heads_per_group": self.target_query_heads_per_group,
            "rope_pair_legal": self.head_dim % 2 == 0,
        }


@dataclass(frozen=True, slots=True)
class Lfm2SparseResidualAssignment:
    """One full-source, one-destination-per-channel sparse fold map.

    The short-conv compiler may use this as an operator-local B/C/x channel
    assignment even when the model-wide residual coordinates use a dense
    linear basis.
    """

    target_width: int
    source_to_target: tuple[int, ...]
    coefficients: tuple[float, ...]
    assignment_sha256: str = ""

    def __post_init__(self) -> None:
        target_width = _positive_int(self.target_width, "target_width")
        assignment = tuple(self.source_to_target)
        coefficients = tuple(float(value) for value in self.coefficients)
        if (
            not assignment
            or len(assignment) != len(coefficients)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= target_width
                for value in assignment
            )
            or any(not math.isfinite(value) or value == 0.0 for value in coefficients)
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Sparse residual assignment must map every source channel once "
                "with one finite nonzero coefficient."
            )
        if set(assignment) != set(range(target_width)):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Sparse residual assignment must cover every target channel."
            )
        body = {
            "schema": "neural-forge-lfm2-sparse-residual-assignment-v1",
            "source_width": len(assignment),
            "target_width": target_width,
            "source_to_target": list(assignment),
            "coefficients": list(coefficients),
        }
        digest = hashlib.sha256(
            json.dumps(
                body,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if self.assignment_sha256 and self.assignment_sha256 != digest:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Sparse residual assignment digest does not match its contents."
            )
        object.__setattr__(self, "target_width", target_width)
        object.__setattr__(self, "source_to_target", assignment)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "assignment_sha256", digest)

    @property
    def source_width(self) -> int:
        return len(self.source_to_target)

    @property
    def full_source_coverage(self) -> bool:
        return len(self.source_to_target) == self.source_width

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_width": self.source_width,
            "target_width": self.target_width,
            "source_to_target": list(self.source_to_target),
            "coefficients": list(self.coefficients),
            "assignment_sha256": self.assignment_sha256,
            "full_source_coverage": self.full_source_coverage,
            "source_channels_dropped": 0,
            "dense_rotation": False,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CachedShortConvFoldCase:
    """One chronological cached LFM2 B/C/x and residual-endpoint case."""

    target_input: Any
    parent_b: Any
    parent_c: Any
    parent_x: Any
    parent_post_operator_residual_target_space: Any
    target_pre_operator_residual: Any
    reset_state: bool = False
    parent_contribution: Any | None = None


@dataclass(frozen=True, slots=True)
class Lfm2CompiledStructuralShortConvFold:
    """Dense smaller B/C/x, grouped FIR, and endpoint-fitted output state."""

    residual_assignment: Lfm2SparseResidualAssignment
    in_proj_weight: Any
    in_proj_bias: Any | None
    conv_weight: Any
    conv_bias: Any | None
    out_proj_weight: Any
    out_proj_bias: Any | None
    channel_groups: tuple[tuple[int, ...], ...]
    channel_barycentric_weights: tuple[tuple[float, ...], ...]
    source_hidden_size: int
    target_hidden_size: int
    fir_length: int
    source_logical_parameters: int
    target_logical_parameters: int
    source_state_sha256_before: str
    source_state_sha256_after: str
    target_template_sha256: str
    source_derived_prior_sha256: str
    compiled_state_sha256: str
    residual_endpoint_relative_rmse: float
    prior_correction_reports: tuple[Lfm2PriorCorrectionReport, ...]
    source_residual_width: int | None = None
    target_residual_width: int | None = None
    external_residual_assignment_sha256: str | None = None

    @property
    def bias(self) -> bool:
        return self.in_proj_bias is not None

    @property
    def removed_logical_parameters(self) -> int:
        return self.source_logical_parameters - self.target_logical_parameters

    @property
    def parameter_reduction_fraction(self) -> float:
        return self.removed_logical_parameters / self.source_logical_parameters

    @property
    def full_source_coverage(self) -> bool:
        return sorted(
            source for group in self.channel_groups for source in group
        ) == list(range(self.source_hidden_size))

    @property
    def residual_assignment_sha256(self) -> str:
        return self.residual_assignment.assignment_sha256

    def state_dict(self) -> Mapping[str, Any]:
        values = {
            "in_proj.weight": self.in_proj_weight,
            "conv.weight": self.conv_weight,
            "out_proj.weight": self.out_proj_weight,
        }
        if self.bias:
            values.update(
                {
                    "in_proj.bias": self.in_proj_bias,
                    "conv.bias": self.conv_bias,
                    "out_proj.bias": self.out_proj_bias,
                }
            )
        return MappingProxyType(
            {name: value.detach().cpu().clone() for name, value in values.items()}
        )

    def as_dict(self) -> dict[str, Any]:
        value = {
            "schema": "neural-forge-lfm2-structural-residual-short-conv-fold-v1",
            "source_hidden_size": self.source_hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "fir_length": self.fir_length,
            "bias": self.bias,
            "residual_assignment": self.residual_assignment.as_dict(),
            "channel_groups": [list(value) for value in self.channel_groups],
            "channel_barycentric_weights": [
                list(value) for value in self.channel_barycentric_weights
            ],
            "full_source_coverage": self.full_source_coverage,
            "full_source_assignment_fraction": 1.0,
            "source_channels_dropped": 0,
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "removed_logical_parameters": self.removed_logical_parameters,
            "parameter_reduction_fraction": self.parameter_reduction_fraction,
            "residual_endpoint_relative_rmse": (self.residual_endpoint_relative_rmse),
            "source_state_sha256_before": self.source_state_sha256_before,
            "source_state_sha256_after": self.source_state_sha256_after,
            "target_template_sha256": self.target_template_sha256,
            "source_derived_prior_sha256": self.source_derived_prior_sha256,
            "compiled_state_sha256": self.compiled_state_sha256,
            "prior_correction_reports": [
                value.as_dict() for value in self.prior_correction_reports
            ],
            "producer_transport": (
                "all_source_sparse_channel_barycenter_plus_prior_residual_fit"
            ),
            "kernel_transport": (
                "same_all_source_channel_barycenter_over_grouped_fir_rows"
            ),
            "consumer_transport": (
                "actual_compiled_B_times_x_then_grouped_FIR_then_C_times_v_"
                "to_residual_endpoint_prior_fit"
            ),
            "input_projection_chunk_order": ["B", "C", "x"],
            "blank_target_values_read": False,
            "dense_floating_state": True,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }
        if self.source_residual_width is not None:
            value.update(
                {
                    "source_internal_channel_width": self.source_hidden_size,
                    "target_internal_channel_width": self.target_hidden_size,
                    "source_residual_width": self.source_residual_width,
                    "target_residual_width": self.target_residual_width,
                    "external_residual_assignment_sha256": (
                        self.external_residual_assignment_sha256
                    ),
                    "internal_and_residual_widths_decoupled": True,
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class Lfm2StructuralAttentionFoldAudit:
    source_unchanged: bool
    blank_target_values_read: bool
    full_source_query_coverage: bool
    full_source_key_value_coverage: bool
    source_query_heads_dropped: int
    source_key_value_heads_dropped: int
    target_shapes_exact: bool
    every_projection_structurally_smaller: bool

    @property
    def all_hold(self) -> bool:
        return (
            self.source_unchanged
            and not self.blank_target_values_read
            and self.full_source_query_coverage
            and self.full_source_key_value_coverage
            and self.source_query_heads_dropped == 0
            and self.source_key_value_heads_dropped == 0
            and self.target_shapes_exact
            and self.every_projection_structurally_smaller
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_unchanged": self.source_unchanged,
            "blank_target_values_read": self.blank_target_values_read,
            "full_source_query_coverage": self.full_source_query_coverage,
            "full_source_key_value_coverage": (self.full_source_key_value_coverage),
            "source_query_heads_dropped": self.source_query_heads_dropped,
            "source_key_value_heads_dropped": self.source_key_value_heads_dropped,
            "target_shapes_exact": self.target_shapes_exact,
            "every_projection_structurally_smaller": (
                self.every_projection_structurally_smaller
            ),
            "all_hold": self.all_hold,
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "dense_floating_state": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledStructuralAttentionFold:
    topology: Lfm2StructuralAttentionTopology
    cluster_plan: GQAActivationClusterPlan
    query_weight: Any
    key_weight: Any
    value_weight: Any
    output_weight: Any
    query_groups: tuple[tuple[int, ...], ...]
    query_barycentric_weights: tuple[tuple[float, ...], ...]
    key_value_groups: tuple[tuple[int, ...], ...]
    key_value_barycentric_weights: tuple[tuple[float, ...], ...]
    source_logical_parameters: int
    target_logical_parameters: int
    source_state_sha256_before: str
    source_state_sha256_after: str
    target_template_sha256: str
    residual_assignment_sha256: str | None
    source_derived_prior_sha256: str | None
    compiled_state_sha256: str
    centered_score_relative_rmse: float
    residual_endpoint_relative_rmse: float
    query_row_weighting_mode: str
    key_value_row_weighting_mode: str
    query_row_sqrt_weight_bundle_sha256: str
    key_value_row_sqrt_weight_bundle_sha256: str
    prior_correction_reports: tuple[Lfm2PriorCorrectionReport, ...]
    audit: Lfm2StructuralAttentionFoldAudit

    @property
    def removed_logical_parameters(self) -> int:
        return self.source_logical_parameters - self.target_logical_parameters

    @property
    def parameter_reduction_fraction(self) -> float:
        return self.removed_logical_parameters / self.source_logical_parameters

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().clone(),
                "v_proj.weight": self.value_weight.detach().cpu().clone(),
                "out_proj.weight": self.output_weight.detach().cpu().clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "topology": self.topology.as_dict(),
            "query_groups": [list(value) for value in self.query_groups],
            "query_barycentric_weights": [
                list(value) for value in self.query_barycentric_weights
            ],
            "key_value_groups": [list(value) for value in self.key_value_groups],
            "key_value_barycentric_weights": [
                list(value) for value in self.key_value_barycentric_weights
            ],
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "removed_logical_parameters": self.removed_logical_parameters,
            "parameter_reduction_fraction": self.parameter_reduction_fraction,
            "centered_score_relative_rmse": self.centered_score_relative_rmse,
            "residual_endpoint_relative_rmse": (self.residual_endpoint_relative_rmse),
            "source_state_sha256_before": self.source_state_sha256_before,
            "source_state_sha256_after": self.source_state_sha256_after,
            "target_template_sha256": self.target_template_sha256,
            "residual_assignment_sha256": self.residual_assignment_sha256,
            "source_derived_prior_sha256": self.source_derived_prior_sha256,
            "compiled_state_sha256": self.compiled_state_sha256,
            "prior_correction_reports": [
                value.as_dict() for value in self.prior_correction_reports
            ],
            "full_source_assignment_fraction": 1.0,
            "source_groups_dropped": 0,
            "score_objective": (
                "contribution_weighted_centered_cached_score_barycenter_"
                "low_rank_factorization"
            ),
            "qk_solver": "closed_form_svd_factorization_plus_linear_regression",
            "v_solver": "full_assignment_barycenter_plus_linear_regression",
            "o_solver": "child_attention_context_to_residual_endpoint_regression",
            "query_row_weighting_mode": self.query_row_weighting_mode,
            "key_value_row_weighting_mode": self.key_value_row_weighting_mode,
            "query_row_sqrt_weight_bundle_sha256": (
                self.query_row_sqrt_weight_bundle_sha256
            ),
            "key_value_row_sqrt_weight_bundle_sha256": (
                self.key_value_row_sqrt_weight_bundle_sha256
            ),
            "equal_causal_phase_mass_proven": False,
            "causal_phase_weighting": "authenticated_cpcf_caller_report_required",
            "residual_contribution_signature": (
                "full_vector_when_supplied_context_magnitude_fallback"
            ),
            "audit": self.audit.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledStructuralAttentionProducerFold:
    """Target-shaped Q/K/V state compiled before observing child context."""

    topology: Lfm2StructuralAttentionTopology
    cluster_plan: GQAActivationClusterPlan
    query_weight: Any
    key_weight: Any
    value_weight: Any
    query_groups: tuple[tuple[int, ...], ...]
    query_barycentric_weights: tuple[tuple[float, ...], ...]
    key_value_groups: tuple[tuple[int, ...], ...]
    key_value_barycentric_weights: tuple[tuple[float, ...], ...]
    source_logical_parameters: int
    target_logical_parameters: int
    source_state_sha256_before: str
    source_state_sha256_after: str
    target_template_sha256: str
    residual_assignment_sha256: str | None
    source_derived_prior_sha256: str | None
    compiled_state_sha256: str
    centered_score_relative_rmse: float
    query_row_weighting_mode: str
    key_value_row_weighting_mode: str
    query_row_sqrt_weight_bundle_sha256: str
    key_value_row_sqrt_weight_bundle_sha256: str
    prior_correction_reports: tuple[Lfm2PriorCorrectionReport, ...]

    @property
    def removed_logical_parameters(self) -> int:
        return self.source_logical_parameters - self.target_logical_parameters

    @property
    def parameter_reduction_fraction(self) -> float:
        return self.removed_logical_parameters / self.source_logical_parameters

    @property
    def full_source_coverage(self) -> bool:
        return sorted(
            source for group in self.query_groups for source in group
        ) == list(range(self.topology.source_query_heads)) and sorted(
            source for group in self.key_value_groups for source in group
        ) == list(range(self.topology.source_key_value_heads))

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().clone(),
                "v_proj.weight": self.value_weight.detach().cpu().clone(),
            }
        )

    def score_state_dict(self) -> Mapping[str, Any]:
        """Return only the fixed score-path state for a coupled V/O solve."""

        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().clone(),
            }
        )

    @property
    def score_state_sha256(self) -> str:
        return _state_sha256(self.score_state_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ("neural-forge-lfm2-structural-attention-producer-fold-v1"),
            "topology": self.topology.as_dict(),
            "query_groups": [list(value) for value in self.query_groups],
            "query_barycentric_weights": [
                list(value) for value in self.query_barycentric_weights
            ],
            "key_value_groups": [list(value) for value in self.key_value_groups],
            "key_value_barycentric_weights": [
                list(value) for value in self.key_value_barycentric_weights
            ],
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "removed_logical_parameters": self.removed_logical_parameters,
            "parameter_reduction_fraction": (self.parameter_reduction_fraction),
            "source_state_sha256_before": self.source_state_sha256_before,
            "source_state_sha256_after": self.source_state_sha256_after,
            "target_template_sha256": self.target_template_sha256,
            "residual_assignment_sha256": (self.residual_assignment_sha256),
            "source_derived_prior_sha256": (self.source_derived_prior_sha256),
            "compiled_state_sha256": self.compiled_state_sha256,
            "centered_score_relative_rmse": (self.centered_score_relative_rmse),
            "query_row_weighting_mode": self.query_row_weighting_mode,
            "key_value_row_weighting_mode": self.key_value_row_weighting_mode,
            "query_row_sqrt_weight_bundle_sha256": (
                self.query_row_sqrt_weight_bundle_sha256
            ),
            "key_value_row_sqrt_weight_bundle_sha256": (
                self.key_value_row_sqrt_weight_bundle_sha256
            ),
            "equal_causal_phase_mass_proven": False,
            "causal_phase_weighting": "authenticated_cpcf_caller_report_required",
            "prior_correction_reports": [
                value.as_dict() for value in self.prior_correction_reports
            ],
            "full_source_coverage": self.full_source_coverage,
            "source_query_heads_dropped": 0,
            "source_key_value_heads_dropped": 0,
            "consumer_compiled": False,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledStructuralAttentionConsumerFold:
    """O projection fitted from caller-supplied actual child contexts."""

    output_weight: Any
    context_width: int
    target_output_size: int
    case_count: int
    target_template_sha256: str
    source_derived_prior_sha256: str | None
    actual_child_context_sha256: str
    endpoint_target_sha256: str
    compiled_state_sha256: str
    residual_endpoint_relative_rmse: float
    prior_correction_report: Lfm2PriorCorrectionReport

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {"out_proj.weight": (self.output_weight.detach().cpu().clone())}
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ("neural-forge-lfm2-structural-attention-consumer-fold-v1"),
            "context_width": self.context_width,
            "target_output_size": self.target_output_size,
            "case_count": self.case_count,
            "target_template_sha256": self.target_template_sha256,
            "source_derived_prior_sha256": (self.source_derived_prior_sha256),
            "actual_child_context_sha256": (self.actual_child_context_sha256),
            "endpoint_target_sha256": self.endpoint_target_sha256,
            "compiled_state_sha256": self.compiled_state_sha256,
            "residual_endpoint_relative_rmse": (self.residual_endpoint_relative_rmse),
            "prior_correction_report": (self.prior_correction_report.as_dict()),
            "context_source": "caller-supplied-actual-child-context",
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledStructuralAttentionValueOutputFold:
    """Coupled V/O endpoint fit under one already-compiled fixed Q/K path."""

    value_weight: Any
    output_weight: Any
    topology: Lfm2StructuralAttentionTopology
    case_count: int
    iterations_requested: int
    iterations_accepted: int
    fit_relative_rmse_history: tuple[float, ...]
    target_template_sha256: str
    source_derived_prior_sha256: str
    fixed_query_key_sha256: str
    live_value_sha256: str
    actual_child_context_sha256: str
    reconstructed_live_context_sha256: str
    endpoint_target_sha256: str
    compiled_state_sha256: str
    context_reconstruction_relative_rmse: float
    residual_endpoint_relative_rmse: float
    value_context_solver: str
    prior_correction_reports: tuple[Lfm2PriorCorrectionReport, ...]
    fit_tolerance: float

    @property
    def fit_nonincreasing(self) -> bool:
        return all(
            current <= previous + self.fit_tolerance
            for previous, current in zip(
                self.fit_relative_rmse_history,
                self.fit_relative_rmse_history[1:],
            )
        )

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "v_proj.weight": self.value_weight.detach().cpu().clone(),
                "out_proj.weight": self.output_weight.detach().cpu().clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ("neural-forge-lfm2-structural-attention-value-output-fold-v1"),
            "topology": self.topology.as_dict(),
            "case_count": self.case_count,
            "iterations_requested": self.iterations_requested,
            "iterations_accepted": self.iterations_accepted,
            "fit_relative_rmse_history": list(self.fit_relative_rmse_history),
            "fit_nonincreasing": self.fit_nonincreasing,
            "target_template_sha256": self.target_template_sha256,
            "source_derived_prior_sha256": (self.source_derived_prior_sha256),
            "fixed_query_key_sha256": self.fixed_query_key_sha256,
            "live_value_sha256": self.live_value_sha256,
            "actual_child_context_sha256": (self.actual_child_context_sha256),
            "reconstructed_live_context_sha256": (
                self.reconstructed_live_context_sha256
            ),
            "endpoint_target_sha256": self.endpoint_target_sha256,
            "compiled_state_sha256": self.compiled_state_sha256,
            "context_reconstruction_relative_rmse": (
                self.context_reconstruction_relative_rmse
            ),
            "residual_endpoint_relative_rmse": (self.residual_endpoint_relative_rmse),
            "value_context_solver": self.value_context_solver,
            "prior_correction_reports": [
                value.as_dict() for value in self.prior_correction_reports
            ],
            "score_path_fixed": True,
            "producer_parameter_names": [
                "q_proj.weight",
                "k_proj.weight",
            ],
            "consumer_parameter_names": [
                "v_proj.weight",
                "out_proj.weight",
            ],
            "solver": (
                "actual-context-o-control-plus-fixed-score-alternating-closed-form-v-o"
            ),
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledStructuralShortConvProducerFold:
    """Target B/C/x input projection and grouped FIR, without output fit."""

    residual_assignment: Lfm2SparseResidualAssignment
    in_proj_weight: Any
    in_proj_bias: Any | None
    conv_weight: Any
    conv_bias: Any | None
    channel_groups: tuple[tuple[int, ...], ...]
    channel_barycentric_weights: tuple[tuple[float, ...], ...]
    source_hidden_size: int
    target_hidden_size: int
    fir_length: int
    source_logical_parameters: int
    target_logical_parameters: int
    source_state_sha256_before: str
    source_state_sha256_after: str
    target_template_sha256: str
    source_derived_prior_sha256: str
    compiled_state_sha256: str
    prior_correction_report: Lfm2PriorCorrectionReport
    source_residual_width: int | None = None
    target_residual_width: int | None = None
    external_residual_assignment_sha256: str | None = None

    @property
    def bias(self) -> bool:
        return self.in_proj_bias is not None

    @property
    def full_source_coverage(self) -> bool:
        return sorted(
            source for group in self.channel_groups for source in group
        ) == list(range(self.source_hidden_size))

    @property
    def removed_logical_parameters(self) -> int:
        return self.source_logical_parameters - self.target_logical_parameters

    @property
    def parameter_reduction_fraction(self) -> float:
        return self.removed_logical_parameters / self.source_logical_parameters

    def state_dict(self) -> Mapping[str, Any]:
        values = {
            "in_proj.weight": self.in_proj_weight,
            "conv.weight": self.conv_weight,
        }
        if self.bias:
            values.update(
                {
                    "in_proj.bias": self.in_proj_bias,
                    "conv.bias": self.conv_bias,
                }
            )
        return MappingProxyType(
            {name: value.detach().cpu().clone() for name, value in values.items()}
        )

    def as_dict(self) -> dict[str, Any]:
        value = {
            "schema": ("neural-forge-lfm2-structural-short-conv-producer-fold-v1"),
            "source_hidden_size": self.source_hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "fir_length": self.fir_length,
            "bias": self.bias,
            "residual_assignment": self.residual_assignment.as_dict(),
            "channel_groups": [list(value) for value in self.channel_groups],
            "channel_barycentric_weights": [
                list(value) for value in self.channel_barycentric_weights
            ],
            "full_source_coverage": self.full_source_coverage,
            "source_channels_dropped": 0,
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "removed_logical_parameters": self.removed_logical_parameters,
            "parameter_reduction_fraction": (self.parameter_reduction_fraction),
            "source_state_sha256_before": self.source_state_sha256_before,
            "source_state_sha256_after": self.source_state_sha256_after,
            "target_template_sha256": self.target_template_sha256,
            "source_derived_prior_sha256": (self.source_derived_prior_sha256),
            "compiled_state_sha256": self.compiled_state_sha256,
            "prior_correction_report": (self.prior_correction_report.as_dict()),
            "consumer_compiled": False,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }
        if self.source_residual_width is not None:
            value.update(
                {
                    "source_internal_channel_width": self.source_hidden_size,
                    "target_internal_channel_width": self.target_hidden_size,
                    "source_residual_width": self.source_residual_width,
                    "target_residual_width": self.target_residual_width,
                    "external_residual_assignment_sha256": (
                        self.external_residual_assignment_sha256
                    ),
                    "internal_and_residual_widths_decoupled": True,
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class Lfm2CompiledStructuralShortConvConsumerFold:
    """Output projection fitted from caller-supplied actual child contexts."""

    out_proj_weight: Any
    out_proj_bias: Any | None
    target_hidden_size: int
    case_count: int
    target_template_sha256: str
    source_derived_prior_sha256: str | None
    actual_child_context_sha256: str
    endpoint_target_sha256: str
    compiled_state_sha256: str
    residual_endpoint_relative_rmse: float
    prior_correction_report: Lfm2PriorCorrectionReport
    target_context_width: int | None = None

    @property
    def bias(self) -> bool:
        return self.out_proj_bias is not None

    def state_dict(self) -> Mapping[str, Any]:
        values = {"out_proj.weight": self.out_proj_weight}
        if self.bias:
            values["out_proj.bias"] = self.out_proj_bias
        return MappingProxyType(
            {name: value.detach().cpu().clone() for name, value in values.items()}
        )

    def as_dict(self) -> dict[str, Any]:
        value = {
            "schema": ("neural-forge-lfm2-structural-short-conv-consumer-fold-v1"),
            "target_hidden_size": self.target_hidden_size,
            "case_count": self.case_count,
            "bias": self.bias,
            "target_template_sha256": self.target_template_sha256,
            "source_derived_prior_sha256": (self.source_derived_prior_sha256),
            "actual_child_context_sha256": (self.actual_child_context_sha256),
            "endpoint_target_sha256": self.endpoint_target_sha256,
            "compiled_state_sha256": self.compiled_state_sha256,
            "residual_endpoint_relative_rmse": (self.residual_endpoint_relative_rmse),
            "prior_correction_report": (self.prior_correction_report.as_dict()),
            "context_source": "caller-supplied-actual-child-context",
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }
        if self.target_context_width is not None:
            value.update(
                {
                    "target_context_width": self.target_context_width,
                    "context_and_residual_widths_decoupled": True,
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class Lfm2StructuralAttentionInstallReport:
    parameter_identities_preserved: bool
    parameter_shapes_preserved: bool
    gradient_state_preserved: bool
    target_logical_parameters: int
    compiled_state_sha256: str
    installed_state_sha256: str

    @property
    def accepted(self) -> bool:
        return (
            self.parameter_identities_preserved
            and self.parameter_shapes_preserved
            and self.gradient_state_preserved
            and self.compiled_state_sha256 == self.installed_state_sha256
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "parameter_identities_preserved": self.parameter_identities_preserved,
            "parameter_shapes_preserved": self.parameter_shapes_preserved,
            "gradient_state_preserved": self.gradient_state_preserved,
            "target_logical_parameters": self.target_logical_parameters,
            "compiled_state_sha256": self.compiled_state_sha256,
            "installed_state_sha256": self.installed_state_sha256,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CachedMlpFoldCase:
    """Aligned cached SwiGLU activations and target-space residual endpoint."""

    target_input: Any
    parent_gate: Any
    parent_up: Any
    parent_post_swiglu: Any
    parent_post_mlp_residual_target_space: Any
    target_pre_mlp_residual: Any
    parent_contribution: Any | None = None


@dataclass(frozen=True, slots=True)
class Lfm2CachedNormalizedInputMlpFoldCase:
    """FIT-only MLP evidence including the parent's normalized input rows."""

    target_input: Any
    parent_normalized_input: Any
    parent_gate: Any
    parent_up: Any
    parent_post_swiglu: Any
    parent_post_mlp_residual_target_space: Any
    target_pre_mlp_residual: Any
    parent_contribution: Any | None = None


def _raw_mlp_parent_case_sha256(
    case: Lfm2CachedMlpFoldCase,
    *,
    case_index: int,
) -> str:
    tensors = {
        "parent_gate": case.parent_gate,
        "parent_up": case.parent_up,
        "parent_post_swiglu": case.parent_post_swiglu,
        "parent_post_mlp_residual_target_space": (
            case.parent_post_mlp_residual_target_space
        ),
    }
    body = {
        "case_index": case_index,
        "tensor_sha256": {
            name: _tensor_sha256(value) for name, value in sorted(tensors.items())
        },
        "parent_contribution_mode": (
            "source_down_column_norm"
            if case.parent_contribution is None
            else "caller_supplied"
        ),
        "parent_contribution_sha256": (
            None
            if case.parent_contribution is None
            else _tensor_sha256(case.parent_contribution)
        ),
    }
    return _metadata_sha256(body)


def lfm2_normalized_input_mlp_fit_evidence_sha256(
    cases: Sequence[Lfm2CachedNormalizedInputMlpFoldCase],
    case_sample_weights: Sequence[float],
) -> str:
    body = []
    for case_index, (case, sample_weight) in enumerate(
        zip(cases, case_sample_weights, strict=True)
    ):
        tensors = {
            "target_input": case.target_input,
            "parent_normalized_input": case.parent_normalized_input,
            "parent_gate": case.parent_gate,
            "parent_up": case.parent_up,
            "parent_post_swiglu": case.parent_post_swiglu,
            "parent_post_mlp_residual_target_space": (
                case.parent_post_mlp_residual_target_space
            ),
            "target_pre_mlp_residual": case.target_pre_mlp_residual,
        }
        body.append(
            {
                "case_index": case_index,
                "sample_weight": float(sample_weight),
                "tensor_sha256": {
                    name: _tensor_sha256(value)
                    for name, value in sorted(tensors.items())
                },
                "parent_contribution_sha256": (
                    None
                    if case.parent_contribution is None
                    else _tensor_sha256(case.parent_contribution)
                ),
            }
        )
    return _metadata_sha256(body)


def _prepared_mlp_tensor_sha256(
    gate_targets: Sequence[Any],
    up_targets: Sequence[Any],
    parent_endpoints: Sequence[Any],
) -> str:
    values = {}
    for case_index, (gate, up, endpoint) in enumerate(
        zip(gate_targets, up_targets, parent_endpoints, strict=True)
    ):
        prefix = f"case.{case_index:04d}"
        values[f"{prefix}.gate_target"] = gate
        values[f"{prefix}.up_target"] = up
        values[f"{prefix}.parent_endpoint"] = endpoint
    return _state_sha256(values)


@dataclass(frozen=True, slots=True)
class Lfm2PreparedStructuralMlpFoldBasis:
    """Authenticated parent-side SwiGLU basis reusable within one layer."""

    cluster_plan: GatedMLPActivationClusterPlan
    gate_targets: tuple[Any, ...]
    up_targets: tuple[Any, ...]
    parent_endpoints: tuple[Any, ...]
    parent_case_sha256: tuple[str, ...]
    case_rows: tuple[int, ...]
    neuron_groups: tuple[tuple[int, ...], ...]
    neuron_barycentric_weights: tuple[tuple[float, ...], ...]
    source_hidden_size: int
    target_hidden_size: int
    source_intermediate_size: int
    target_intermediate_size: int
    source_logical_parameters: int
    target_logical_parameters: int
    source_state_sha256: str
    target_template_sha256: str
    residual_assignment_sha256: str | None
    cluster_ridge: float
    cluster_max_iterations: int
    cluster_distance_chunk_size: int
    prepared_tensor_sha256: str
    report: Mapping[str, Any] | None = None

    def _report_body(self) -> dict[str, Any]:
        cluster_report = self.cluster_plan.report.as_dict()
        retained_tensor_bytes = sum(
            int(value.numel()) * int(value.element_size())
            for value in (
                *self.gate_targets,
                *self.up_targets,
                *self.parent_endpoints,
            )
        )
        return {
            "schema": _PREPARED_MLP_BASIS_SCHEMA,
            "source_hidden_size": self.source_hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "source_intermediate_size": self.source_intermediate_size,
            "target_intermediate_size": self.target_intermediate_size,
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "source_state_sha256": self.source_state_sha256,
            "target_template_sha256": self.target_template_sha256,
            "residual_assignment_sha256": self.residual_assignment_sha256,
            "parent_case_sha256": list(self.parent_case_sha256),
            "parent_evidence_bundle_sha256": _metadata_sha256(
                list(self.parent_case_sha256)
            ),
            "case_rows": list(self.case_rows),
            "observation_rows": sum(self.case_rows),
            "cluster_plan": cluster_report,
            "cluster_plan_sha256": _metadata_sha256(cluster_report),
            "neuron_groups": [list(value) for value in self.neuron_groups],
            "neuron_barycentric_weights": [
                list(value) for value in self.neuron_barycentric_weights
            ],
            "cluster_ridge": self.cluster_ridge,
            "cluster_max_iterations": self.cluster_max_iterations,
            "cluster_distance_chunk_size": self.cluster_distance_chunk_size,
            "prepared_tensor_sha256": self.prepared_tensor_sha256,
            "retained_tensor_bytes": retained_tensor_bytes,
            "retained_parent_gate_up_targets": True,
            "retained_parent_residual_endpoints": True,
            "combined_cluster_feature_matrix_retained": False,
            "preparation_calls": 1,
            "candidate_compiles_performed_during_preparation": 0,
            "validation_evidence_consumed": 0,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }

    def _validate(self) -> None:
        torch = _torch()
        case_count = len(self.case_rows)
        if (
            not isinstance(self.cluster_plan, GatedMLPActivationClusterPlan)
            or case_count <= 0
            or len(self.gate_targets) != case_count
            or len(self.up_targets) != case_count
            or len(self.parent_endpoints) != case_count
            or len(self.parent_case_sha256) != case_count
            or any(
                isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
                for rows in self.case_rows
            )
            or not all(_valid_sha256_text(value) for value in self.parent_case_sha256)
            or not _valid_sha256_text(self.source_state_sha256)
            or not _valid_sha256_text(self.target_template_sha256)
            or (
                self.residual_assignment_sha256 is not None
                and not _valid_sha256_text(self.residual_assignment_sha256)
            )
            or not _valid_sha256_text(self.prepared_tensor_sha256)
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Prepared MLP basis metadata is incomplete."
            )
        source_hidden = _positive_int(
            self.source_hidden_size,
            "prepared source_hidden_size",
        )
        target_hidden = _positive_int(
            self.target_hidden_size,
            "prepared target_hidden_size",
        )
        source_intermediate = _positive_int(
            self.source_intermediate_size,
            "prepared source_intermediate_size",
        )
        target_intermediate = _positive_int(
            self.target_intermediate_size,
            "prepared target_intermediate_size",
        )
        if (
            target_hidden > source_hidden
            or target_intermediate > source_intermediate
            or self.cluster_plan.source_neurons != source_intermediate
            or self.cluster_plan.target_neurons != target_intermediate
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Prepared MLP basis cluster topology differs from its layer."
            )
        expected_assignment = [None] * source_intermediate
        for target, group in enumerate(self.neuron_groups):
            for source in group:
                if (
                    isinstance(source, bool)
                    or not isinstance(source, int)
                    or source < 0
                    or source >= source_intermediate
                    or expected_assignment[source] is not None
                ):
                    raise Lfm2StructuralResidualAttentionFoldError(
                        "Prepared MLP neuron groups do not cover source neurons "
                        "exactly once."
                    )
                expected_assignment[source] = target
        if (
            tuple(expected_assignment) != self.cluster_plan.source_to_target
            or len(self.neuron_barycentric_weights) != target_intermediate
            or len(self.neuron_groups) != target_intermediate
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Prepared MLP neuron assignment differs from its cluster plan."
            )
        for group, weights in zip(
            self.neuron_groups,
            self.neuron_barycentric_weights,
            strict=True,
        ):
            if (
                not group
                or len(group) != len(weights)
                or any(not math.isfinite(value) or value < 0.0 for value in weights)
                or not math.isclose(
                    math.fsum(weights),
                    1.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    "Prepared MLP barycentric weights are invalid."
                )
        for case_index, rows in enumerate(self.case_rows):
            expected_shapes = (
                (self.gate_targets[case_index], (rows, target_intermediate)),
                (self.up_targets[case_index], (rows, target_intermediate)),
                (self.parent_endpoints[case_index], (rows, target_hidden)),
            )
            if any(
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or tuple(value.shape) != shape
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
                for value, shape in expected_shapes
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    "Prepared MLP tensors are not finite detached FP64 CPU rows."
                )
        if (
            _prepared_mlp_tensor_sha256(
                self.gate_targets,
                self.up_targets,
                self.parent_endpoints,
            )
            != self.prepared_tensor_sha256
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Prepared MLP tensor authentication failed."
            )
        body = self._report_body()
        expected_report = {**body, "report_sha256": _metadata_sha256(body)}
        if self.report is None or dict(self.report) != expected_report:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Prepared MLP basis report authentication failed."
            )

    def __post_init__(self) -> None:
        if self.report is None:
            body = self._report_body()
            object.__setattr__(
                self,
                "report",
                MappingProxyType({**body, "report_sha256": _metadata_sha256(body)}),
            )
        self._validate()
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def preparation_sha256(self) -> str:
        return str(self.report["report_sha256"])

    def verify_authentication(self) -> bool:
        try:
            self._validate()
        except (TypeError, ValueError, RuntimeError):
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


@dataclass(frozen=True, slots=True)
class Lfm2CompiledStructuralMlpFold:
    cluster_plan: GatedMLPActivationClusterPlan
    gate_weight: Any
    up_weight: Any
    down_weight: Any
    neuron_groups: tuple[tuple[int, ...], ...]
    neuron_barycentric_weights: tuple[tuple[float, ...], ...]
    source_hidden_size: int
    target_hidden_size: int
    source_intermediate_size: int
    target_intermediate_size: int
    source_logical_parameters: int
    target_logical_parameters: int
    source_state_sha256_before: str
    source_state_sha256_after: str
    target_template_sha256: str
    residual_assignment_sha256: str | None
    source_derived_prior_sha256: str | None
    prepared_basis_sha256: str
    prepared_basis_reused: bool
    compiled_state_sha256: str
    residual_endpoint_relative_rmse: float
    prior_correction_reports: tuple[Lfm2PriorCorrectionReport, ...]

    @property
    def removed_logical_parameters(self) -> int:
        return self.source_logical_parameters - self.target_logical_parameters

    @property
    def parameter_reduction_fraction(self) -> float:
        return self.removed_logical_parameters / self.source_logical_parameters

    @property
    def full_source_coverage(self) -> bool:
        return sorted(
            source for group in self.neuron_groups for source in group
        ) == list(range(self.source_intermediate_size))

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "w1.weight": self.gate_weight.detach().cpu().clone(),
                "w3.weight": self.up_weight.detach().cpu().clone(),
                "w2.weight": self.down_weight.detach().cpu().clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-lfm2-structural-residual-mlp-fold-v1",
            "source_hidden_size": self.source_hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "source_intermediate_size": self.source_intermediate_size,
            "target_intermediate_size": self.target_intermediate_size,
            "neuron_groups": [list(value) for value in self.neuron_groups],
            "neuron_barycentric_weights": [
                list(value) for value in self.neuron_barycentric_weights
            ],
            "full_source_coverage": self.full_source_coverage,
            "full_source_assignment_fraction": 1.0,
            "source_neurons_dropped": 0,
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "removed_logical_parameters": self.removed_logical_parameters,
            "parameter_reduction_fraction": self.parameter_reduction_fraction,
            "residual_endpoint_relative_rmse": (self.residual_endpoint_relative_rmse),
            "source_state_sha256_before": self.source_state_sha256_before,
            "source_state_sha256_after": self.source_state_sha256_after,
            "target_template_sha256": self.target_template_sha256,
            "residual_assignment_sha256": self.residual_assignment_sha256,
            "source_derived_prior_sha256": self.source_derived_prior_sha256,
            "prepared_basis_sha256": self.prepared_basis_sha256,
            "prepared_basis_reused": self.prepared_basis_reused,
            "preparation_calls_for_this_compile": (
                0 if self.prepared_basis_reused else 1
            ),
            "compiled_state_sha256": self.compiled_state_sha256,
            "prior_correction_reports": [
                value.as_dict() for value in self.prior_correction_reports
            ],
            "producer_transport": (
                "all_assigned_cached_gate_up_barycenter_then_closed_form_regression"
            ),
            "consumer_transport": (
                "compiled_child_post_swiglu_to_residual_endpoint_regression"
            ),
            "blank_target_values_read": False,
            "dense_floating_state": True,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledRealMedoidStructuralMlpFold:
    """Receipt and dense state for the chart-transported real-medoid lane."""

    cluster_plan: GatedMLPMultiAnchorPlan
    gate_weight: Any
    up_weight: Any
    down_weight: Any
    neuron_groups: tuple[tuple[int, ...], ...]
    source_hidden_size: int
    target_hidden_size: int
    source_intermediate_size: int
    target_intermediate_size: int
    source_logical_parameters: int
    target_logical_parameters: int
    source_state_sha256_before: str
    source_state_sha256_after: str
    target_template_sha256: str
    residual_chart_sha256: str
    residual_chart_decoder_sha256: str
    producer_prior_sha256: str
    consumer_prior_sha256: str
    compiled_state_sha256: str
    residual_chart_rank: int
    residual_chart_condition_number: float
    fit_case_weight_sum: float
    fit_row_weight_sum: float
    fit_row_weight_minimum: float
    fit_row_weight_maximum: float
    fit_row_weights_sha256: str
    residual_endpoint_relative_rmse: float
    prior_correction_reports: tuple[Lfm2PriorCorrectionReport, ...]

    @property
    def removed_logical_parameters(self) -> int:
        return self.source_logical_parameters - self.target_logical_parameters

    @property
    def parameter_reduction_fraction(self) -> float:
        return self.removed_logical_parameters / self.source_logical_parameters

    @property
    def full_source_coverage(self) -> bool:
        return sorted(
            source for group in self.neuron_groups for source in group
        ) == list(range(self.source_intermediate_size))

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "w1.weight": self.gate_weight.detach().cpu().clone(),
                "w3.weight": self.up_weight.detach().cpu().clone(),
                "w2.weight": self.down_weight.detach().cpu().clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ("neural-forge-lfm2-real-medoid-chart-structural-mlp-fold-v1"),
            "source_hidden_size": self.source_hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "source_intermediate_size": self.source_intermediate_size,
            "target_intermediate_size": self.target_intermediate_size,
            "medoid_source_neurons": list(self.cluster_plan.medoid_source_neurons),
            "producer_plan": "sparse_multi_anchor_real_medoid",
            "anchors_per_source": self.cluster_plan.anchors_per_source,
            "total_sparse_consumer_edges": len(self.cluster_plan.anchor_coefficients),
            "multi_anchor_plan": self.cluster_plan.report.as_dict(),
            "multi_anchor_plan_sha256": canonical_sha256(
                self.cluster_plan.report.as_dict()
            ),
            "neuron_groups": [list(value) for value in self.neuron_groups],
            "full_source_coverage": self.full_source_coverage,
            "full_source_assignment_fraction": 1.0,
            "source_neurons_dropped": 0,
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "removed_logical_parameters": self.removed_logical_parameters,
            "parameter_reduction_fraction": self.parameter_reduction_fraction,
            "residual_endpoint_relative_rmse": (self.residual_endpoint_relative_rmse),
            "source_state_sha256_before": self.source_state_sha256_before,
            "source_state_sha256_after": self.source_state_sha256_after,
            "target_template_sha256": self.target_template_sha256,
            "residual_chart_sha256": self.residual_chart_sha256,
            "residual_chart_decoder_sha256": (self.residual_chart_decoder_sha256),
            "residual_chart_rank": self.residual_chart_rank,
            "residual_chart_condition_number": (self.residual_chart_condition_number),
            "fit_weighting": "sqrt_case_weight_repeated_per_case_row",
            "fit_case_weight_sum": self.fit_case_weight_sum,
            "fit_row_weight_sum": self.fit_row_weight_sum,
            "fit_row_weight_minimum": self.fit_row_weight_minimum,
            "fit_row_weight_maximum": self.fit_row_weight_maximum,
            "fit_row_weights_sha256": self.fit_row_weights_sha256,
            "producer_prior_sha256": self.producer_prior_sha256,
            "consumer_prior_sha256": self.consumer_prior_sha256,
            "compiled_state_sha256": self.compiled_state_sha256,
            "prior_correction_reports": [
                value.as_dict() for value in self.prior_correction_reports
            ],
            "producer_transport": (
                "coupled_real_source_gate_up_medoids_through_authenticated_"
                "residual_chart_pseudoinverse_plus_closed_form_fit_correction"
            ),
            "consumer_transport": (
                "every_source_w2_column_through_post_swiglu_consumer_"
                "coefficients_then_residual_chart_plus_closed_form_endpoint_"
                "correction"
            ),
            "endpoint_feature_source": (
                "materialized_target_dtype_w1_w3_cast_back_to_fp64"
            ),
            "plan_evidence_scope": "supplied_fit_cases_only",
            "heldout_rows_used_for_plan": 0,
            "heldout_rows_used_by_solver": 0,
            "producer_averaging": False,
            "blank_target_values_read": False,
            "dense_floating_state": True,
            "source_parameters_pruned": 0,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledNormalizedInputRealMedoidStructuralMlpFold:
    """Dense MLP state planted through separate producer/consumer charts.

    The producer chart acts between the *normalized* MLP input spaces.  The
    consumer chart remains the authenticated raw residual chart.  Keeping the
    two operators separate prevents RMSNorm gamma and scale from leaking into
    the uncached producer nullspace.
    """

    cluster_plan: GatedMLPMultiAnchorPlan
    weight_complete_signature: GatedMlpWeightCompleteSignature
    gate_weight: Any
    up_weight: Any
    down_weight: Any
    normalized_input_decoder: Any
    neuron_groups: tuple[tuple[int, ...], ...]
    source_hidden_size: int
    target_hidden_size: int
    source_intermediate_size: int
    target_intermediate_size: int
    source_logical_parameters: int
    target_logical_parameters: int
    source_state_sha256_before: str
    source_state_sha256_after: str
    target_template_sha256: str
    residual_chart_sha256: str
    parent_ffn_norm_weight_sha256: str
    target_ffn_norm_weight_sha256: str
    normalized_chart_sha256: str
    normalized_polar_sha256: str
    normalized_structural_decoder_prior_sha256: str
    normalized_input_decoder_sha256: str
    normalized_input_correction_sha256: str
    normalized_input_rowspace_projector_sha256: str
    producer_prior_sha256: str
    consumer_prior_sha256: str
    fit_evidence_sha256: str
    fit_row_weights_sha256: str
    compiled_state_sha256: str
    residual_chart_rank: int
    normalized_chart_rank: int
    normalized_input_fit_rank: int
    normalized_input_decoder_rank: int
    normalized_structural_decoder_prior_rank: int
    normalized_structural_decoder_prior_operator_norm: float
    normalized_structural_decoder_left_inverse_error: float
    normalized_input_global_gain: float
    normalized_input_ridge_policy_value: float
    normalized_input_scaled_ridge: float
    normalized_input_supported_relative_cap: float
    normalized_input_correction_scale: float
    normalized_input_fit_relative_rmse: float
    normalized_input_polar_orthogonality_error: float
    normalized_input_correction_rowspace_leakage: float
    normalized_input_null_completion_error: float
    normalized_input_null_completion_operator_norm: float
    normalized_input_null_completion_operator_norm_bound: float
    fit_case_weight_sum: float
    fit_row_weight_sum: float
    fit_row_weight_minimum: float
    fit_row_weight_maximum: float
    residual_endpoint_relative_rmse: float
    consumer_correction_report: Lfm2PriorCorrectionReport
    residual_chart_decoder_method: str
    residual_chart_orthonormal_columns: bool
    residual_chart_orthogonality_maximum_absolute_error: float
    residual_chart_svd_calls: int
    residual_chart_pseudoinverse_calls: int

    @property
    def removed_logical_parameters(self) -> int:
        return self.source_logical_parameters - self.target_logical_parameters

    @property
    def parameter_reduction_fraction(self) -> float:
        return self.removed_logical_parameters / self.source_logical_parameters

    @property
    def full_source_coverage(self) -> bool:
        return sorted(
            source for group in self.neuron_groups for source in group
        ) == list(range(self.source_intermediate_size))

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "w1.weight": self.gate_weight.detach().cpu().clone(),
                "w3.weight": self.up_weight.detach().cpu().clone(),
                "w2.weight": self.down_weight.detach().cpu().clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ("neural-forge-lfm2-normalized-input-real-medoid-mlp-fold-v1"),
            "source_hidden_size": self.source_hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "source_intermediate_size": self.source_intermediate_size,
            "target_intermediate_size": self.target_intermediate_size,
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "removed_logical_parameters": self.removed_logical_parameters,
            "parameter_reduction_fraction": self.parameter_reduction_fraction,
            "medoid_source_neurons": list(self.cluster_plan.medoid_source_neurons),
            "anchors_per_source": self.cluster_plan.anchors_per_source,
            "total_sparse_consumer_edges": len(self.cluster_plan.anchor_coefficients),
            "multi_anchor_plan": self.cluster_plan.report.as_dict(),
            "multi_anchor_plan_sha256": canonical_sha256(
                self.cluster_plan.report.as_dict()
            ),
            "weight_complete_signature": self.weight_complete_signature.as_dict(),
            "weight_complete_signature_sha256": (
                self.weight_complete_signature.signature_sha256
            ),
            "weight_complete_signature_report_sha256": (
                self.weight_complete_signature.report["report_sha256"]
            ),
            "weight_complete_signature_used_for_structural_selection": True,
            "cached_activation_rows_used_for_structural_selection": False,
            "cached_activation_rows_used_for_sparse_anchor_fit": False,
            "neuron_groups": [list(value) for value in self.neuron_groups],
            "full_source_coverage": self.full_source_coverage,
            "full_source_assignment_fraction": 1.0,
            "source_neurons_dropped": 0,
            "source_state_sha256_before": self.source_state_sha256_before,
            "source_state_sha256_after": self.source_state_sha256_after,
            "target_template_sha256": self.target_template_sha256,
            "parent_ffn_norm_weight_sha256": (self.parent_ffn_norm_weight_sha256),
            "target_ffn_norm_weight_sha256": (self.target_ffn_norm_weight_sha256),
            "producer_bridge": {
                "role": "normalized_mlp_input_target_to_parent",
                "construction": (
                    "positive_gain_gamma_aware_raw_chart_left_inverse_plus_"
                    "fit_rowspace_correction_with_bounded_structural_null_"
                    "completion"
                ),
                "normalized_chart_sha256": self.normalized_chart_sha256,
                "thin_polar_diagnostic_sha256": self.normalized_polar_sha256,
                "structural_decoder_prior_sha256": (
                    self.normalized_structural_decoder_prior_sha256
                ),
                "decoder_sha256": self.normalized_input_decoder_sha256,
                "correction_sha256": self.normalized_input_correction_sha256,
                "rowspace_projector_sha256": (
                    self.normalized_input_rowspace_projector_sha256
                ),
                "normalized_chart_rank": self.normalized_chart_rank,
                "fit_rank": self.normalized_input_fit_rank,
                "decoder_rank": self.normalized_input_decoder_rank,
                "structural_decoder_prior_rank": (
                    self.normalized_structural_decoder_prior_rank
                ),
                "structural_decoder_prior_operator_norm": (
                    self.normalized_structural_decoder_prior_operator_norm
                ),
                "structural_decoder_left_inverse_error": (
                    self.normalized_structural_decoder_left_inverse_error
                ),
                "positive_global_gain": self.normalized_input_global_gain,
                "ridge_policy_value": self.normalized_input_ridge_policy_value,
                "scaled_ridge": self.normalized_input_scaled_ridge,
                "supported_relative_cap_rho": (
                    self.normalized_input_supported_relative_cap
                ),
                "correction_scale": self.normalized_input_correction_scale,
                "fit_relative_rmse": self.normalized_input_fit_relative_rmse,
                "polar_orthogonality_error": (
                    self.normalized_input_polar_orthogonality_error
                ),
                "correction_rowspace_leakage": (
                    self.normalized_input_correction_rowspace_leakage
                ),
                "null_completion_error": (self.normalized_input_null_completion_error),
                "null_completion_operator_norm": (
                    self.normalized_input_null_completion_operator_norm
                ),
                "null_completion_operator_norm_bound": (
                    self.normalized_input_null_completion_operator_norm_bound
                ),
            },
            "consumer_bridge": {
                "role": "raw_residual_source_to_target",
                "residual_chart_sha256": self.residual_chart_sha256,
                "residual_chart_rank": self.residual_chart_rank,
                "consumer_prior_sha256": self.consumer_prior_sha256,
                "decoder_method": self.residual_chart_decoder_method,
                "orthonormal_columns": (self.residual_chart_orthonormal_columns),
                "orthogonality_maximum_absolute_error": (
                    self.residual_chart_orthogonality_maximum_absolute_error
                ),
                "residual_chart_svd_calls": self.residual_chart_svd_calls,
                "residual_chart_pseudoinverse_calls": (
                    self.residual_chart_pseudoinverse_calls
                ),
            },
            "producer_consumer_bridge_fields_separate": True,
            "normalized_chart_sha256": self.normalized_chart_sha256,
            "normalized_input_decoder_sha256": (self.normalized_input_decoder_sha256),
            "producer_prior_sha256": self.producer_prior_sha256,
            "consumer_prior_sha256": self.consumer_prior_sha256,
            "fit_evidence_sha256": self.fit_evidence_sha256,
            "fit_weighting": "sqrt_case_weight_repeated_per_case_row",
            "fit_case_weight_sum": self.fit_case_weight_sum,
            "fit_row_weight_sum": self.fit_row_weight_sum,
            "fit_row_weight_minimum": self.fit_row_weight_minimum,
            "fit_row_weight_maximum": self.fit_row_weight_maximum,
            "fit_row_weights_sha256": self.fit_row_weights_sha256,
            "compiled_state_sha256": self.compiled_state_sha256,
            "residual_endpoint_relative_rmse": (self.residual_endpoint_relative_rmse),
            "consumer_correction_report": self.consumer_correction_report.as_dict(),
            "producer_transport": (
                "coupled_real_source_gate_up_medoids_through_one_shared_"
                "normalized_input_decoder"
            ),
            "consumer_transport": (
                "every_source_w2_column_through_sparse_multi_anchor_"
                "aggregation_then_raw_residual_chart_and_closed_form_endpoint_"
                "correction"
            ),
            "endpoint_feature_source": (
                "materialized_target_dtype_w1_w3_cast_back_to_fp64"
            ),
            "null_completion_policy": (
                "fit_orthogonal_subspace_equals_positive_gain_unit_norm_"
                "gamma_aware_structural_inverse_with_operator_norm_at_most_"
                "global_gain"
            ),
            "producer_averaging": False,
            "producer_per_neuron_interpolation": False,
            "producer_shared_decoder": True,
            "plan_evidence_scope": (
                "all_source_w1_w3_w2_weights_in_global_residual_basis"
            ),
            "fit_cases_used_for_plan_diagnostics_only": True,
            "heldout_rows_used_for_plan": 0,
            "heldout_rows_used_by_solver": 0,
            "blank_target_values_read": False,
            "dense_floating_state": True,
            "source_parameters_pruned": 0,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }


def prepare_lfm2_structural_residual_mlp_fold_basis(
    *,
    source_gate_weight: Any,
    source_up_weight: Any,
    source_down_weight: Any,
    target_gate_weight_template: Any,
    target_up_weight_template: Any,
    target_down_weight_template: Any,
    cases: Sequence[Lfm2CachedMlpFoldCase],
    residual_assignment_sha256: str | None = None,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
) -> Lfm2PreparedStructuralMlpFoldBasis:
    """Prepare immutable parent-side SwiGLU evidence once for one target layer."""

    torch = _torch()
    if not cases or not all(
        isinstance(value, Lfm2CachedMlpFoldCase) for value in cases
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "cases must contain at least one Lfm2CachedMlpFoldCase."
        )
    cluster_ridge = _nonnegative_float(cluster_ridge, "cluster_ridge")
    cluster_max_iterations = _positive_int(
        cluster_max_iterations,
        "cluster_max_iterations",
    )
    cluster_distance_chunk_size = _positive_int(
        cluster_distance_chunk_size,
        "cluster_distance_chunk_size",
    )
    source_values = {
        "w1.weight": source_gate_weight,
        "w3.weight": source_up_weight,
        "w2.weight": source_down_weight,
    }
    source_before = _state_sha256(source_values)
    source_gate = _float_tensor(
        source_gate_weight,
        "source_gate_weight",
        ndim=2,
    )
    source_up = _float_tensor(source_up_weight, "source_up_weight", ndim=2)
    source_down = _float_tensor(
        source_down_weight,
        "source_down_weight",
        ndim=2,
    )
    source_intermediate, source_hidden = map(int, source_gate.shape)
    if tuple(source_up.shape) != tuple(source_gate.shape) or tuple(
        source_down.shape
    ) != (source_hidden, source_intermediate):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Source gate/up/down tensors do not form one standard LFM2 SwiGLU."
        )

    target_templates = {
        "w1.weight": target_gate_weight_template,
        "w3.weight": target_up_weight_template,
        "w2.weight": target_down_weight_template,
    }
    template_shapes: dict[str, tuple[int, ...]] = {}
    template_spec: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, value in target_templates.items():
        dtype, shape = _dense_template(value, f"target_{name}")
        template_shapes[name] = shape
        template_spec[name] = (shape, str(dtype))
    target_intermediate, target_hidden = template_shapes["w1.weight"]
    expected_shapes = {
        "w1.weight": (target_intermediate, target_hidden),
        "w3.weight": (target_intermediate, target_hidden),
        "w2.weight": (target_hidden, target_intermediate),
    }
    if template_shapes != expected_shapes:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Target templates are not one standard atomic LFM2 SwiGLU set."
        )
    if target_hidden > source_hidden or target_intermediate > source_intermediate:
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target cannot expand the SwiGLU hidden or "
            "intermediate width."
        )
    if target_hidden == source_hidden and target_intermediate == source_intermediate:
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target must reduce the SwiGLU hidden or intermediate width."
        )
    if residual_assignment_sha256 is not None and not _valid_sha256_text(
        residual_assignment_sha256
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_assignment_sha256 must be a lowercase SHA-256 digest."
        )
    if source_hidden != target_hidden and residual_assignment_sha256 is None:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Cross-hidden SwiGLU compilation must be bound to the one global "
            "full-source sparse residual assignment."
        )

    parent_case_sha256 = tuple(
        _raw_mlp_parent_case_sha256(case, case_index=case_index)
        for case_index, case in enumerate(cases)
    )
    normalized = []
    for case_index, case in enumerate(cases):
        prefix = f"cases[{case_index}]"
        target_input = _float_tensor(
            case.target_input,
            f"{prefix}.target_input",
            ndim=2,
        )
        gate = _float_tensor(
            case.parent_gate,
            f"{prefix}.parent_gate",
            ndim=2,
        )
        up = _float_tensor(case.parent_up, f"{prefix}.parent_up", ndim=2)
        post = _float_tensor(
            case.parent_post_swiglu,
            f"{prefix}.parent_post_swiglu",
            ndim=2,
        )
        parent_endpoint = _float_tensor(
            case.parent_post_mlp_residual_target_space,
            f"{prefix}.parent_post_mlp_residual_target_space",
            ndim=2,
        )
        target_residual = _float_tensor(
            case.target_pre_mlp_residual,
            f"{prefix}.target_pre_mlp_residual",
            ndim=2,
        )
        rows = int(target_input.shape[0])
        if (
            tuple(target_input.shape) != (rows, target_hidden)
            or tuple(gate.shape) != (rows, source_intermediate)
            or tuple(up.shape) != tuple(gate.shape)
            or tuple(post.shape) != tuple(gate.shape)
            or tuple(parent_endpoint.shape) != (rows, target_hidden)
            or tuple(target_residual.shape) != tuple(parent_endpoint.shape)
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{prefix} tensor shapes do not match source/target SwiGLU topology."
            )
        if case.parent_contribution is None:
            contribution = post * torch.linalg.vector_norm(
                source_down,
                dim=0,
            ).unsqueeze(0)
        else:
            raw = case.parent_contribution
            ndim = getattr(raw, "ndim", None)
            if ndim not in {2, 3}:
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix}.parent_contribution must have rank 2 or 3."
                )
            contribution = _float_tensor(
                raw,
                f"{prefix}.parent_contribution",
                ndim=int(ndim),
            )
            if tuple(contribution.shape[:2]) != (rows, source_intermediate):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix}.parent_contribution has incompatible leading axes."
                )
            if contribution.ndim == 3:
                contribution = torch.linalg.vector_norm(
                    contribution,
                    dim=-1,
                )
        normalized.append(
            {
                "gate": gate,
                "up": up,
                "post": post,
                "contribution": contribution,
                "parent_endpoint": parent_endpoint,
                "rows": rows,
            }
        )

    with torch.no_grad():
        all_gate = torch.cat([value["gate"] for value in normalized], dim=0)
        all_up = torch.cat([value["up"] for value in normalized], dim=0)
        all_post = torch.cat([value["post"] for value in normalized], dim=0)
        all_contribution = torch.cat(
            [value["contribution"] for value in normalized],
            dim=0,
        )
        cluster = build_gated_mlp_activation_cluster_plan(
            all_gate,
            all_up,
            all_post,
            all_contribution,
            target_neurons=target_intermediate,
            source_neurons=source_intermediate,
            ridge=cluster_ridge,
            max_iterations=cluster_max_iterations,
            distance_chunk_size=cluster_distance_chunk_size,
        )
        if not cluster.report.full_source_coverage:
            raise Lfm2StructuralResidualAttentionFoldError(
                "SwiGLU clustering did not prove full source coverage."
            )
        importance = all_post.square().mean(dim=0) + all_contribution.square().mean(
            dim=0
        )
        gate_targets = []
        up_targets = []
        neuron_groups: tuple[tuple[int, ...], ...] | None = None
        neuron_weights: tuple[tuple[float, ...], ...] | None = None
        for value in normalized:
            gate_target, groups, weights = _weighted_group_rows(
                value["gate"],
                cluster.source_to_target,
                target_intermediate,
                importance,
            )
            up_target, groups_2, weights_2 = _weighted_group_rows(
                value["up"],
                cluster.source_to_target,
                target_intermediate,
                importance,
            )
            if groups != groups_2 or weights != weights_2:
                raise Lfm2StructuralResidualAttentionFoldError(
                    "Gate and up barycentric assignments differ."
                )
            if neuron_groups is None:
                neuron_groups, neuron_weights = groups, weights
            elif neuron_groups != groups or neuron_weights != weights:
                raise Lfm2StructuralResidualAttentionFoldError(
                    "SwiGLU barycentric assignments changed between cases."
                )
            gate_targets.append(
                gate_target.detach().to(device="cpu").contiguous().clone()
            )
            up_targets.append(up_target.detach().to(device="cpu").contiguous().clone())
        parent_endpoints = tuple(
            value["parent_endpoint"].detach().to(device="cpu").contiguous().clone()
            for value in normalized
        )

    source_after = _state_sha256(source_values)
    source_logical = sum(int(value.numel()) for value in source_values.values())
    target_logical = sum(math.prod(shape) for shape in template_shapes.values())
    if (
        source_before != source_after
        or neuron_groups is None
        or neuron_weights is None
        or sorted(source for group in neuron_groups for source in group)
        != list(range(source_intermediate))
        or target_logical >= source_logical
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The prepared SwiGLU basis failed its structural invariants."
        )
    gate_target_tuple = tuple(gate_targets)
    up_target_tuple = tuple(up_targets)
    prepared_tensor_sha256 = _prepared_mlp_tensor_sha256(
        gate_target_tuple,
        up_target_tuple,
        parent_endpoints,
    )
    return Lfm2PreparedStructuralMlpFoldBasis(
        cluster_plan=cluster,
        gate_targets=gate_target_tuple,
        up_targets=up_target_tuple,
        parent_endpoints=parent_endpoints,
        parent_case_sha256=parent_case_sha256,
        case_rows=tuple(int(value["rows"]) for value in normalized),
        neuron_groups=neuron_groups,
        neuron_barycentric_weights=neuron_weights,
        source_hidden_size=source_hidden,
        target_hidden_size=target_hidden,
        source_intermediate_size=source_intermediate,
        target_intermediate_size=target_intermediate,
        source_logical_parameters=source_logical,
        target_logical_parameters=target_logical,
        source_state_sha256=source_before,
        target_template_sha256=_template_sha256(template_spec),
        residual_assignment_sha256=residual_assignment_sha256,
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
        prepared_tensor_sha256=prepared_tensor_sha256,
    )


def _compile_lfm2_structural_residual_mlp_fold_from_prepared_basis(
    *,
    source_gate_weight: Any,
    source_up_weight: Any,
    source_down_weight: Any,
    target_gate_weight_template: Any,
    target_up_weight_template: Any,
    target_down_weight_template: Any,
    cases: Sequence[Lfm2CachedMlpFoldCase],
    gate_weight_prior: Any | None,
    up_weight_prior: Any | None,
    down_weight_prior: Any | None,
    residual_assignment_sha256: str | None,
    ridge: float,
    max_prior_delta_ratio: float | None,
    cluster_ridge: float,
    cluster_max_iterations: int,
    cluster_distance_chunk_size: int,
    prepared_basis: Lfm2PreparedStructuralMlpFoldBasis,
) -> Lfm2CompiledStructuralMlpFold:
    torch = _torch()
    if (
        not isinstance(prepared_basis, Lfm2PreparedStructuralMlpFoldBasis)
        or not prepared_basis.verify_authentication()
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "prepared_basis is not an authenticated structural MLP basis."
        )
    if (
        cluster_ridge != prepared_basis.cluster_ridge
        or cluster_max_iterations != prepared_basis.cluster_max_iterations
        or cluster_distance_chunk_size != prepared_basis.cluster_distance_chunk_size
        or residual_assignment_sha256 != prepared_basis.residual_assignment_sha256
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Prepared MLP basis policy or residual-chart lineage changed."
        )

    source_values = {
        "w1.weight": source_gate_weight,
        "w3.weight": source_up_weight,
        "w2.weight": source_down_weight,
    }
    source_before = _state_sha256(source_values)
    if source_before != prepared_basis.source_state_sha256:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Prepared MLP basis source state changed."
        )
    target_templates = {
        "w1.weight": target_gate_weight_template,
        "w3.weight": target_up_weight_template,
        "w2.weight": target_down_weight_template,
    }
    template_dtypes = {}
    template_shapes = {}
    template_spec = {}
    for name, value in target_templates.items():
        dtype, shape = _dense_template(value, f"target_{name}")
        template_dtypes[name] = dtype
        template_shapes[name] = shape
        template_spec[name] = (shape, str(dtype))
    if _template_sha256(template_spec) != prepared_basis.target_template_sha256:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Prepared MLP basis target template changed."
        )
    expected_shapes = {
        "w1.weight": (
            prepared_basis.target_intermediate_size,
            prepared_basis.target_hidden_size,
        ),
        "w3.weight": (
            prepared_basis.target_intermediate_size,
            prepared_basis.target_hidden_size,
        ),
        "w2.weight": (
            prepared_basis.target_hidden_size,
            prepared_basis.target_intermediate_size,
        ),
    }
    if template_shapes != expected_shapes:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Prepared MLP basis target topology changed."
        )
    parent_case_sha256 = tuple(
        _raw_mlp_parent_case_sha256(case, case_index=case_index)
        for case_index, case in enumerate(cases)
    )
    if parent_case_sha256 != prepared_basis.parent_case_sha256:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Prepared MLP parent evidence changed between candidate compiles."
        )

    candidate_rows = []
    for case_index, case in enumerate(cases):
        prefix = f"cases[{case_index}]"
        target_input = _float_tensor(
            case.target_input,
            f"{prefix}.target_input",
            ndim=2,
        )
        target_residual = _float_tensor(
            case.target_pre_mlp_residual,
            f"{prefix}.target_pre_mlp_residual",
            ndim=2,
        )
        rows = prepared_basis.case_rows[case_index]
        if tuple(target_input.shape) != (
            rows,
            prepared_basis.target_hidden_size,
        ) or tuple(target_residual.shape) != (rows, prepared_basis.target_hidden_size):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{prefix} actual child tensors changed prepared MLP row topology."
            )
        candidate_rows.append(
            {
                "target_input": target_input,
                "target_residual": target_residual,
            }
        )

    supplied_priors = {
        "w1.weight": gate_weight_prior,
        "w3.weight": up_weight_prior,
        "w2.weight": down_weight_prior,
    }
    normalized_priors = {}
    for name, prior in supplied_priors.items():
        if prior is None:
            normalized_priors[name] = None
            continue
        normalized_prior = _float_tensor(
            prior,
            f"source_derived_{name}_prior",
            ndim=2,
        )
        if tuple(normalized_prior.shape) != template_shapes[name]:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"source-derived {name} prior must have target shape "
                f"{template_shapes[name]}."
            )
        normalized_priors[name] = normalized_prior
    prior_sha = (
        None
        if all(value is None for value in normalized_priors.values())
        else _state_sha256(
            {
                name: value
                for name, value in normalized_priors.items()
                if value is not None
            }
        )
    )

    with torch.no_grad():
        target_input_rows = torch.cat(
            [value["target_input"] for value in candidate_rows],
            dim=0,
        )
        gate_weight, gate_fit = _fit_linear(
            target_input_rows,
            torch.cat(prepared_basis.gate_targets, dim=0),
            role="mlp_gate",
            ridge=ridge,
            prior=normalized_priors["w1.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        up_weight, up_fit = _fit_linear(
            target_input_rows,
            torch.cat(prepared_basis.up_targets, dim=0),
            role="mlp_up",
            ridge=ridge,
            prior=normalized_priors["w3.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        child_post = torch.nn.functional.silu(target_input_rows @ gate_weight.T) * (
            target_input_rows @ up_weight.T
        )
        endpoint_target = torch.cat(
            [
                parent_endpoint - value["target_residual"]
                for parent_endpoint, value in zip(
                    prepared_basis.parent_endpoints,
                    candidate_rows,
                    strict=True,
                )
            ],
            dim=0,
        )
        down_weight, down_fit = _fit_linear(
            child_post,
            endpoint_target,
            role="mlp_down",
            ridge=ridge,
            prior=normalized_priors["w2.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        fp64_state = {
            "w1.weight": gate_weight,
            "w3.weight": up_weight,
            "w2.weight": down_weight,
        }
        compiled = {
            name: fp64_state[name]
            .to(device="cpu", dtype=template_dtypes[name])
            .contiguous()
            .clone()
            for name in ("w1.weight", "w3.weight", "w2.weight")
        }
        materialized_gate = compiled["w1.weight"].to(torch.float64)
        materialized_up = compiled["w3.weight"].to(torch.float64)
        materialized_down = compiled["w2.weight"].to(torch.float64)
        materialized_post = torch.nn.functional.silu(
            target_input_rows @ materialized_gate.T
        ) * (target_input_rows @ materialized_up.T)
        endpoint_prediction = materialized_post @ materialized_down.T
        endpoint_error = _relative_rmse(endpoint_prediction, endpoint_target)

    source_after = _state_sha256(source_values)
    if (
        source_after != source_before
        or _prepared_mlp_tensor_sha256(
            prepared_basis.gate_targets,
            prepared_basis.up_targets,
            prepared_basis.parent_endpoints,
        )
        != prepared_basis.prepared_tensor_sha256
        or any(
            tuple(compiled[name].shape) != template_shapes[name]
            for name in ("w1.weight", "w3.weight", "w2.weight")
        )
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The prepared-basis SwiGLU compile failed its structural invariants."
        )
    return Lfm2CompiledStructuralMlpFold(
        cluster_plan=prepared_basis.cluster_plan,
        gate_weight=compiled["w1.weight"],
        up_weight=compiled["w3.weight"],
        down_weight=compiled["w2.weight"],
        neuron_groups=prepared_basis.neuron_groups,
        neuron_barycentric_weights=(prepared_basis.neuron_barycentric_weights),
        source_hidden_size=prepared_basis.source_hidden_size,
        target_hidden_size=prepared_basis.target_hidden_size,
        source_intermediate_size=prepared_basis.source_intermediate_size,
        target_intermediate_size=prepared_basis.target_intermediate_size,
        source_logical_parameters=prepared_basis.source_logical_parameters,
        target_logical_parameters=prepared_basis.target_logical_parameters,
        source_state_sha256_before=source_before,
        source_state_sha256_after=source_after,
        target_template_sha256=prepared_basis.target_template_sha256,
        residual_assignment_sha256=residual_assignment_sha256,
        source_derived_prior_sha256=prior_sha,
        prepared_basis_sha256=prepared_basis.preparation_sha256,
        prepared_basis_reused=True,
        compiled_state_sha256=_state_sha256(compiled),
        residual_endpoint_relative_rmse=endpoint_error,
        prior_correction_reports=(gate_fit, up_fit, down_fit),
    )


def compile_lfm2_real_medoid_structural_residual_mlp_fold(
    *,
    source_gate_weight: Any,
    source_up_weight: Any,
    source_down_weight: Any,
    target_gate_weight_template: Any,
    target_up_weight_template: Any,
    target_down_weight_template: Any,
    residual_chart: Any,
    residual_chart_sha256: str,
    cases: Sequence[Lfm2CachedMlpFoldCase],
    case_sample_weights: Sequence[float] | None = None,
    ridge: float = 0.0,
    max_prior_delta_ratio: float | None = None,
    anchors_per_source: int = 4,
    max_candidate_targets: int = 64,
    cluster_ridge: float = 1e-8,
    max_coefficient_l1: float = 64.0,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    similarity_chunk_size: int = 512,
    max_similarity_products: int = 64_000_000,
    max_projection_fits: int = 1_000_000,
    max_sparse_entries: int = 1_000_000,
    max_workspace_elements: int = 16_000_000,
) -> Lfm2CompiledRealMedoidStructuralMlpFold:
    """Fold a dense SwiGLU through real medoids and an authenticated chart.

    Producer rows are copied as coupled gate/up medoids and transported through
    the residual chart pseudoinverse.  Every source consumer column is folded
    through a bounded sparse multi-anchor post-SwiGLU plan.  Cached rows only
    contribute closed-form residual corrections; no blank target values are
    read and no gradient operation is performed.  One positive weight per case
    may be supplied; it is repeated over that case's rows and applied through
    square-root weighting to planning and all three closed-form fits.
    """

    torch = _torch()
    if not cases or not all(
        isinstance(value, Lfm2CachedMlpFoldCase) for value in cases
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "cases must contain at least one Lfm2CachedMlpFoldCase."
        )
    ridge = _nonnegative_float(ridge, "ridge")
    if max_prior_delta_ratio is not None:
        max_prior_delta_ratio = _nonnegative_float(
            max_prior_delta_ratio, "max_prior_delta_ratio"
        )
    if not _valid_sha256_text(residual_chart_sha256):
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_chart_sha256 must be a lowercase SHA-256 digest."
        )
    try:
        supplied_chart_sha256 = lfm2_authenticated_tensor_sha256(residual_chart)
    except (TypeError, RuntimeError) as exc:
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_chart must be an authenticated materialized tensor."
        ) from exc
    if supplied_chart_sha256 != residual_chart_sha256:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Residual chart tensor authentication failed."
        )
    if case_sample_weights is None:
        normalized_case_weights = tuple(1.0 for _ in cases)
    else:
        if isinstance(case_sample_weights, (str, bytes)) or len(
            case_sample_weights
        ) != len(cases):
            raise Lfm2StructuralResidualAttentionFoldError(
                "case_sample_weights must contain one value per FIT case."
            )
        parsed_weights = []
        for case_index, value in enumerate(case_sample_weights):
            if isinstance(value, bool):
                raise Lfm2StructuralResidualAttentionFoldError(
                    "case_sample_weights must contain positive finite values."
                )
            try:
                weight = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"case_sample_weights[{case_index}] is not finite."
                ) from exc
            if not math.isfinite(weight) or weight <= 0.0:
                raise Lfm2StructuralResidualAttentionFoldError(
                    "case_sample_weights must contain positive finite values."
                )
            parsed_weights.append(weight)
        normalized_case_weights = tuple(parsed_weights)

    source_values = {
        "w1.weight": source_gate_weight,
        "w3.weight": source_up_weight,
        "w2.weight": source_down_weight,
    }
    source_before = _state_sha256(source_values)
    source_gate = _float_tensor(source_gate_weight, "source_gate_weight", ndim=2)
    source_up = _float_tensor(source_up_weight, "source_up_weight", ndim=2)
    source_down = _float_tensor(source_down_weight, "source_down_weight", ndim=2)
    source_intermediate, source_hidden = map(int, source_gate.shape)
    if tuple(source_up.shape) != tuple(source_gate.shape) or tuple(
        source_down.shape
    ) != (source_hidden, source_intermediate):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Source gate/up/down tensors do not form one standard LFM2 SwiGLU."
        )

    target_templates = {
        "w1.weight": target_gate_weight_template,
        "w3.weight": target_up_weight_template,
        "w2.weight": target_down_weight_template,
    }
    template_dtypes: dict[str, Any] = {}
    template_shapes: dict[str, tuple[int, ...]] = {}
    template_spec: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, value in target_templates.items():
        dtype, shape = _dense_template(value, f"target_{name}")
        template_dtypes[name] = dtype
        template_shapes[name] = shape
        template_spec[name] = (shape, str(dtype))
    target_intermediate, target_hidden = template_shapes["w1.weight"]
    if template_shapes != {
        "w1.weight": (target_intermediate, target_hidden),
        "w3.weight": (target_intermediate, target_hidden),
        "w2.weight": (target_hidden, target_intermediate),
    }:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Target templates are not one standard atomic LFM2 SwiGLU set."
        )
    if target_hidden > source_hidden or target_intermediate > source_intermediate:
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target cannot expand hidden or intermediate width."
        )
    if target_hidden == source_hidden and target_intermediate == source_intermediate:
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target must reduce hidden or intermediate width."
        )
    if not all(
        int(target_templates[name].numel()) < int(source_values[name].numel())
        for name in ("w1.weight", "w3.weight", "w2.weight")
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Every materialized target MLP tensor must be smaller than its source."
        )

    chart = _float_tensor(residual_chart, "residual_chart", ndim=2)
    if tuple(chart.shape) != (source_hidden, target_hidden):
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_chart must have shape [source_hidden, target_hidden]."
        )
    chart_rank = int(torch.linalg.matrix_rank(chart).item())
    if chart_rank != target_hidden:
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_chart must have full target-column rank."
        )

    with torch.no_grad():
        singular_values = torch.linalg.svdvals(chart)
        chart_condition = float((singular_values.max() / singular_values.min()).item())
        chart_decoder = torch.linalg.pinv(chart).contiguous()
        mapped_source_down = (chart.T @ source_down).contiguous()

        normalized: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases):
            prefix = f"cases[{case_index}]"
            target_input = _float_tensor(
                case.target_input, f"{prefix}.target_input", ndim=2
            )
            gate = _float_tensor(case.parent_gate, f"{prefix}.parent_gate", ndim=2)
            up = _float_tensor(case.parent_up, f"{prefix}.parent_up", ndim=2)
            post = _float_tensor(
                case.parent_post_swiglu,
                f"{prefix}.parent_post_swiglu",
                ndim=2,
            )
            endpoint = _float_tensor(
                case.parent_post_mlp_residual_target_space,
                f"{prefix}.parent_post_mlp_residual_target_space",
                ndim=2,
            )
            target_residual = _float_tensor(
                case.target_pre_mlp_residual,
                f"{prefix}.target_pre_mlp_residual",
                ndim=2,
            )
            rows = int(target_input.shape[0])
            if (
                tuple(target_input.shape) != (rows, target_hidden)
                or tuple(gate.shape) != (rows, source_intermediate)
                or tuple(up.shape) != tuple(gate.shape)
                or tuple(post.shape) != tuple(gate.shape)
                or tuple(endpoint.shape) != (rows, target_hidden)
                or tuple(target_residual.shape) != tuple(endpoint.shape)
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix} tensor shapes do not match source/target topology."
                )
            if case.parent_contribution is None:
                contribution = post * torch.linalg.vector_norm(
                    mapped_source_down, dim=0
                ).unsqueeze(0)
            else:
                raw = case.parent_contribution
                ndim = getattr(raw, "ndim", None)
                if ndim not in {2, 3}:
                    raise Lfm2StructuralResidualAttentionFoldError(
                        f"{prefix}.parent_contribution must have rank 2 or 3."
                    )
                contribution = _float_tensor(
                    raw, f"{prefix}.parent_contribution", ndim=int(ndim)
                )
                if tuple(contribution.shape[:2]) != (
                    rows,
                    source_intermediate,
                ):
                    raise Lfm2StructuralResidualAttentionFoldError(
                        f"{prefix}.parent_contribution has incompatible axes."
                    )
                if contribution.ndim == 3:
                    contribution = torch.linalg.vector_norm(contribution, dim=-1)
            normalized.append(
                {
                    "target_input": target_input,
                    "gate": gate,
                    "up": up,
                    "post": post,
                    "contribution": contribution,
                    "endpoint": endpoint,
                    "target_residual": target_residual,
                    "sample_weight": normalized_case_weights[case_index],
                    "sqrt_weight": math.sqrt(normalized_case_weights[case_index]),
                }
            )

        row_weights = torch.cat(
            [
                torch.full(
                    (int(value["target_input"].shape[0]),),
                    float(value["sample_weight"]),
                    dtype=torch.float64,
                )
                for value in normalized
            ],
            dim=0,
        )
        sqrt_row_weights = torch.sqrt(row_weights).unsqueeze(1)
        all_gate_unweighted = torch.cat([value["gate"] for value in normalized], dim=0)
        all_up_unweighted = torch.cat([value["up"] for value in normalized], dim=0)
        all_post_unweighted = torch.cat([value["post"] for value in normalized], dim=0)
        all_gate = all_gate_unweighted * sqrt_row_weights
        all_up = all_up_unweighted * sqrt_row_weights
        all_post = all_post_unweighted * sqrt_row_weights
        all_contribution = (
            torch.cat([value["contribution"] for value in normalized], dim=0)
            * sqrt_row_weights
        )
        plan = build_gated_mlp_multi_anchor_cluster_plan(
            all_gate,
            all_up,
            all_post,
            all_contribution,
            target_neurons=target_intermediate,
            consumer_directions=mapped_source_down,
            consumer_axis=1,
            source_neurons=source_intermediate,
            anchors_per_source=anchors_per_source,
            max_candidate_targets=max_candidate_targets,
            ridge=cluster_ridge,
            max_coefficient_l1=max_coefficient_l1,
            max_iterations=cluster_max_iterations,
            distance_chunk_size=cluster_distance_chunk_size,
            similarity_chunk_size=similarity_chunk_size,
            max_similarity_products=max_similarity_products,
            max_projection_fits=max_projection_fits,
            max_sparse_entries=max_sparse_entries,
            max_workspace_elements=max_workspace_elements,
            calibration_provenance="cached_fit_rows_only",
        )
        if not plan.full_source_coverage:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Multi-anchor MLP plan did not prove full source coverage."
            )

        selected_gate = plan.select_gate_producer(source_gate, axis=0).to(torch.float64)
        selected_up = plan.select_up_producer(source_up, axis=0).to(torch.float64)
        gate_prior = (selected_gate @ chart_decoder.T).contiguous()
        up_prior = (selected_up @ chart_decoder.T).contiguous()
        aggregated_down = plan.aggregate_down_consumer(source_down, axis=1).to(
            torch.float64
        )
        down_prior = (chart.T @ aggregated_down).contiguous()

        target_input_rows = torch.cat(
            [value["target_input"] for value in normalized], dim=0
        )
        medoid_indices = torch.tensor(plan.medoid_source_neurons, dtype=torch.long)
        gate_targets = all_gate_unweighted.index_select(1, medoid_indices)
        up_targets = all_up_unweighted.index_select(1, medoid_indices)
        gate_weight_fp64, gate_fit = _fit_linear(
            target_input_rows * sqrt_row_weights,
            gate_targets * sqrt_row_weights,
            role="mlp_real_medoid_gate",
            ridge=ridge,
            prior=gate_prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        up_weight_fp64, up_fit = _fit_linear(
            target_input_rows * sqrt_row_weights,
            up_targets * sqrt_row_weights,
            role="mlp_real_medoid_up",
            ridge=ridge,
            prior=up_prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
        )

        # W2 is deliberately solved from actual materialized producer bytes.
        # This lets its closed-form correction absorb target-dtype planting
        # error instead of discovering it only after the artifact is sealed.
        materialized_gate = gate_weight_fp64.to(
            device="cpu", dtype=template_dtypes["w1.weight"]
        ).contiguous()
        materialized_up = up_weight_fp64.to(
            device="cpu", dtype=template_dtypes["w3.weight"]
        ).contiguous()
        materialized_post = torch.nn.functional.silu(
            target_input_rows @ materialized_gate.to(torch.float64).T
        ) * (target_input_rows @ materialized_up.to(torch.float64).T)
        endpoint_target = torch.cat(
            [value["endpoint"] - value["target_residual"] for value in normalized],
            dim=0,
        )
        down_weight_fp64, down_fit = _fit_linear(
            materialized_post * sqrt_row_weights,
            endpoint_target * sqrt_row_weights,
            role="mlp_multi_anchor_down",
            ridge=ridge,
            prior=down_prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        materialized_down = down_weight_fp64.to(
            device="cpu", dtype=template_dtypes["w2.weight"]
        ).contiguous()
        compiled = {
            "w1.weight": materialized_gate.clone(),
            "w3.weight": materialized_up.clone(),
            "w2.weight": materialized_down.clone(),
        }
        endpoint_prediction = materialized_post @ materialized_down.to(torch.float64).T
        endpoint_error = _relative_rmse(endpoint_prediction, endpoint_target)

    neuron_groups_lists: list[list[int]] = [[] for _ in range(target_intermediate)]
    for source_index, target_index in enumerate(plan.source_to_target):
        neuron_groups_lists[target_index].append(source_index)
    neuron_groups = tuple(tuple(value) for value in neuron_groups_lists)
    source_after = _state_sha256(source_values)
    source_logical = sum(int(value.numel()) for value in source_values.values())
    target_logical = sum(math.prod(shape) for shape in template_shapes.values())
    if (
        source_after != source_before
        or not plan.report.invariants.all_hold
        or sorted(source for group in neuron_groups for source in group)
        != list(range(source_intermediate))
        or target_logical >= source_logical
        or any(
            tuple(compiled[name].shape) != template_shapes[name]
            or compiled[name].requires_grad
            or compiled[name].grad_fn is not None
            or not bool(torch.isfinite(compiled[name]).all().item())
            for name in ("w1.weight", "w3.weight", "w2.weight")
        )
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The real-medoid chart MLP compile failed its invariants."
        )
    return Lfm2CompiledRealMedoidStructuralMlpFold(
        cluster_plan=plan,
        gate_weight=compiled["w1.weight"],
        up_weight=compiled["w3.weight"],
        down_weight=compiled["w2.weight"],
        neuron_groups=neuron_groups,
        source_hidden_size=source_hidden,
        target_hidden_size=target_hidden,
        source_intermediate_size=source_intermediate,
        target_intermediate_size=target_intermediate,
        source_logical_parameters=source_logical,
        target_logical_parameters=target_logical,
        source_state_sha256_before=source_before,
        source_state_sha256_after=source_after,
        target_template_sha256=_template_sha256(template_spec),
        residual_chart_sha256=residual_chart_sha256,
        residual_chart_decoder_sha256=lfm2_authenticated_tensor_sha256(chart_decoder),
        producer_prior_sha256=_state_sha256(
            {"w1.weight": gate_prior, "w3.weight": up_prior}
        ),
        consumer_prior_sha256=_tensor_sha256(down_prior),
        compiled_state_sha256=_state_sha256(compiled),
        residual_chart_rank=chart_rank,
        residual_chart_condition_number=chart_condition,
        fit_case_weight_sum=math.fsum(normalized_case_weights),
        fit_row_weight_sum=float(row_weights.sum().item()),
        fit_row_weight_minimum=float(row_weights.min().item()),
        fit_row_weight_maximum=float(row_weights.max().item()),
        fit_row_weights_sha256=lfm2_authenticated_tensor_sha256(row_weights),
        residual_endpoint_relative_rmse=endpoint_error,
        prior_correction_reports=(gate_fit, up_fit, down_fit),
    )


def compile_lfm2_normalized_input_real_medoid_structural_mlp_fold(
    *,
    source_gate_weight: Any,
    source_up_weight: Any,
    source_down_weight: Any,
    target_gate_weight_template: Any,
    target_up_weight_template: Any,
    target_down_weight_template: Any,
    residual_chart: Any,
    residual_chart_sha256: str,
    parent_ffn_norm_weight: Any,
    parent_ffn_norm_weight_sha256: str,
    target_ffn_norm_weight: Any,
    target_ffn_norm_weight_sha256: str,
    cases: Sequence[Lfm2CachedNormalizedInputMlpFoldCase],
    case_sample_weights: Sequence[float] | None = None,
    normalized_decoder_ridge: float = 1e-8,
    ridge: float = 0.0,
    max_prior_delta_ratio: float | None = None,
    anchors_per_source: int = 4,
    max_candidate_targets: int = 64,
    cluster_ridge: float = 1e-8,
    max_coefficient_l1: float = 64.0,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    similarity_chunk_size: int = 512,
    max_similarity_products: int = 64_000_000,
    max_projection_fits: int = 1_000_000,
    max_sparse_entries: int = 1_000_000,
    max_workspace_elements: int = 16_000_000,
    fold_compute_device: str = "cpu",
) -> Lfm2CompiledNormalizedInputRealMedoidStructuralMlpFold:
    """Plant a smaller dense SwiGLU through separate normalized/raw charts.

    ``residual_chart`` has the established source-by-target orientation.  It
    is converted to a normalized-input chart with the authenticated parent
    and target RMSNorm gamma vectors::

        A = diag(gamma_parent)^-1 @ residual_chart @ diag(gamma_target)

    The raw chart pseudoinverse is conjugated by the two RMSNorm gamma
    vectors, yielding the exact normalized-coordinate left inverse before
    rowwise RMS scale.  Its unit-operator-norm form supplies the bounded
    target-to-parent completion.  One strictly positive gain and one shared
    FIT-rowspace correction are solved from
    ``target_input -> parent_normalized_input``.  The correction is
    ridge-scaled by FIT design energy and capped to rho=1 relative to the
    supported structural prior.  The orthogonal FIT nullspace remains exactly
    that gain-scaled gamma-aware completion, whose operator norm is bounded by
    the gain.

    Coupled real source W1/W3 medoids are transported by that one decoder;
    neither producer receives a per-neuron interpolation or independent fit.
    Every source W2 column is still represented by the sparse multi-anchor
    consumer plan and transported through the raw residual chart.  The final
    W2 solve uses the actual materialized target-dtype W1/W3 bytes.  No model
    forward, backward, optimizer, heldout row, or target-template value is
    used.
    """

    torch = _torch()
    if not cases or not all(
        isinstance(value, Lfm2CachedNormalizedInputMlpFoldCase) for value in cases
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "cases must contain at least one Lfm2CachedNormalizedInputMlpFoldCase."
        )
    normalized_decoder_ridge = _nonnegative_float(
        normalized_decoder_ridge, "normalized_decoder_ridge"
    )
    ridge = _nonnegative_float(ridge, "ridge")
    if max_prior_delta_ratio is not None:
        max_prior_delta_ratio = _nonnegative_float(
            max_prior_delta_ratio, "max_prior_delta_ratio"
        )

    authenticated_inputs = (
        (residual_chart, residual_chart_sha256, "residual_chart"),
        (
            parent_ffn_norm_weight,
            parent_ffn_norm_weight_sha256,
            "parent_ffn_norm_weight",
        ),
        (
            target_ffn_norm_weight,
            target_ffn_norm_weight_sha256,
            "target_ffn_norm_weight",
        ),
    )
    for tensor, expected_sha256, name in authenticated_inputs:
        if not _valid_sha256_text(expected_sha256):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{name}_sha256 must be a lowercase SHA-256 digest."
            )
        try:
            actual_sha256 = lfm2_authenticated_tensor_sha256(tensor)
        except (TypeError, RuntimeError) as exc:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{name} must be an authenticated materialized tensor."
            ) from exc
        if actual_sha256 != expected_sha256:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{name} tensor authentication failed."
            )

    if case_sample_weights is None:
        normalized_case_weights = tuple(1.0 for _ in cases)
    else:
        if isinstance(case_sample_weights, (str, bytes)) or len(
            case_sample_weights
        ) != len(cases):
            raise Lfm2StructuralResidualAttentionFoldError(
                "case_sample_weights must contain one value per FIT case."
            )
        parsed_weights = []
        for case_index, value in enumerate(case_sample_weights):
            if isinstance(value, bool):
                raise Lfm2StructuralResidualAttentionFoldError(
                    "case_sample_weights must contain positive finite values."
                )
            try:
                weight = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"case_sample_weights[{case_index}] is not finite."
                ) from exc
            if not math.isfinite(weight) or weight <= 0.0:
                raise Lfm2StructuralResidualAttentionFoldError(
                    "case_sample_weights must contain positive finite values."
                )
            parsed_weights.append(weight)
        normalized_case_weights = tuple(parsed_weights)

    source_values = {
        "w1.weight": source_gate_weight,
        "w3.weight": source_up_weight,
        "w2.weight": source_down_weight,
    }
    source_before = _state_sha256(source_values)
    source_gate = _float_tensor(source_gate_weight, "source_gate_weight", ndim=2)
    source_up = _float_tensor(source_up_weight, "source_up_weight", ndim=2)
    source_down = _float_tensor(source_down_weight, "source_down_weight", ndim=2)
    source_intermediate, source_hidden = map(int, source_gate.shape)
    if tuple(source_up.shape) != tuple(source_gate.shape) or tuple(
        source_down.shape
    ) != (source_hidden, source_intermediate):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Source gate/up/down tensors do not form one standard LFM2 SwiGLU."
        )

    target_templates = {
        "w1.weight": target_gate_weight_template,
        "w3.weight": target_up_weight_template,
        "w2.weight": target_down_weight_template,
    }
    template_dtypes: dict[str, Any] = {}
    template_shapes: dict[str, tuple[int, ...]] = {}
    template_spec: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, value in target_templates.items():
        dtype, shape = _dense_template(value, f"target_{name}")
        template_dtypes[name] = dtype
        template_shapes[name] = shape
        template_spec[name] = (shape, str(dtype))
    target_intermediate, target_hidden = template_shapes["w1.weight"]
    if template_shapes != {
        "w1.weight": (target_intermediate, target_hidden),
        "w3.weight": (target_intermediate, target_hidden),
        "w2.weight": (target_hidden, target_intermediate),
    }:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Target templates are not one standard atomic LFM2 SwiGLU set."
        )
    if target_hidden > source_hidden or target_intermediate > source_intermediate:
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target cannot expand hidden or intermediate width."
        )
    if target_hidden == source_hidden and target_intermediate == source_intermediate:
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target must reduce hidden or intermediate width."
        )
    if not all(
        int(target_templates[name].numel()) < int(source_values[name].numel())
        for name in ("w1.weight", "w3.weight", "w2.weight")
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Every materialized target MLP tensor must be smaller than its source."
        )

    chart = _float_tensor(residual_chart, "residual_chart", ndim=2)
    parent_gamma = _float_tensor(
        parent_ffn_norm_weight, "parent_ffn_norm_weight", ndim=1
    )
    target_gamma = _float_tensor(
        target_ffn_norm_weight, "target_ffn_norm_weight", ndim=1
    )
    if tuple(chart.shape) != (source_hidden, target_hidden):
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_chart must have shape [source_hidden, target_hidden]."
        )
    if tuple(parent_gamma.shape) != (source_hidden,) or tuple(target_gamma.shape) != (
        target_hidden,
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Authenticated parent/target ffn_norm weights do not match the "
            "source/target hidden widths."
        )
    if bool((parent_gamma == 0.0).any().item()) or bool(
        (target_gamma == 0.0).any().item()
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Normalized-input chart construction requires nonzero ffn_norm weights."
        )
    with torch.no_grad():
        chart_identity = torch.eye(target_hidden, dtype=torch.float64)
        chart_gram = (chart.T @ chart).contiguous()
        chart_orthogonality_error = float(
            (chart_gram - chart_identity).abs().max().item()
        )
        chart_is_orthonormal = bool(
            torch.allclose(
                chart_gram,
                chart_identity,
                rtol=1e-8,
                atol=1e-8,
            )
        )
        if chart_is_orthonormal:
            chart_rank = target_hidden
            weight_signature_basis = chart.contiguous().clone()
            raw_chart_decoder = chart.T.contiguous()
            raw_chart_decoder_method = "authenticated_orthonormal_chart_transpose"
            raw_chart_svd_calls = 0
            raw_chart_pseudoinverse_calls = 0
        else:
            raw_chart_u, raw_chart_singular, raw_chart_vh = torch.linalg.svd(
                chart,
                full_matrices=False,
            )
            singular_threshold = (
                max(source_hidden, target_hidden)
                * torch.finfo(torch.float64).eps
                * raw_chart_singular.max()
            )
            supported_singular = raw_chart_singular > singular_threshold
            chart_rank = int(supported_singular.sum().item())
            weight_signature_basis = (raw_chart_u @ raw_chart_vh).contiguous()
            raw_chart_decoder = (
                (
                    raw_chart_vh.T[:, supported_singular]
                    * raw_chart_singular[supported_singular].reciprocal().unsqueeze(0)
                )
                @ raw_chart_u[:, supported_singular].T
            ).contiguous()
            raw_chart_decoder_method = "single_authenticated_thin_svd_pseudoinverse"
            raw_chart_svd_calls = 1
            raw_chart_pseudoinverse_calls = 0
    if chart_rank != target_hidden:
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_chart must have full target-column rank."
        )
    weight_signature = build_gated_mlp_weight_complete_signature(
        gate_weight=source_gate,
        up_weight=source_up,
        down_weight=source_down,
        residual_basis=weight_signature_basis,
        expected_input_sha256={
            "w1.weight": gated_mlp_weight_signature_tensor_sha256(source_gate),
            "w3.weight": gated_mlp_weight_signature_tensor_sha256(source_up),
            "w2.weight": gated_mlp_weight_signature_tensor_sha256(source_down),
            "residual_basis": gated_mlp_weight_signature_tensor_sha256(
                weight_signature_basis
            ),
        },
        compute_device=fold_compute_device,
        compute_dtype="float64",
    )
    if (
        weight_signature.source_neurons != source_intermediate
        or weight_signature.source_hidden_size != source_hidden
        or weight_signature.target_hidden_size != target_hidden
        or weight_signature.verify()["accepted"] is not True
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Weight-complete MLP signature does not match the source/target topology."
        )

    with torch.no_grad():
        # A maps parent normalized coordinates into child normalized
        # coordinates up to the unavoidable positive rowwise RMS ratio.
        normalized_chart = (
            chart * parent_gamma.reciprocal().unsqueeze(1) * target_gamma.unsqueeze(0)
        ).contiguous()
        if not bool(torch.isfinite(normalized_chart).all().item()):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Normalized-input chart contains non-finite coefficients."
            )
        normalized_chart_rank = int(torch.linalg.matrix_rank(normalized_chart).item())
        if normalized_chart_rank != target_hidden:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Normalized-input chart must have full target-column rank."
            )
        polar_u, _, polar_vh = torch.linalg.svd(normalized_chart, full_matrices=False)
        normalized_polar = (polar_u @ polar_vh).contiguous()
        polar_orthogonality_error = float(
            torch.linalg.matrix_norm(
                normalized_polar.T @ normalized_polar
                - torch.eye(target_hidden, dtype=torch.float64),
                ord=2,
            ).item()
        )
        # The polar factor is a useful orientation diagnostic, but it is not
        # the decoder: polarizing A discards the nonuniform gamma scaling.
        # Conjugating the raw chart left inverse instead gives an exact inverse
        # of A on child coordinates.  A self-dual chart uses its authenticated
        # transpose directly; a generic full-column chart reuses exactly one
        # thin SVD shared with the structural signature basis.
        gamma_aware_structural_decoder = (
            target_gamma.reciprocal().unsqueeze(1)
            * raw_chart_decoder
            * parent_gamma.unsqueeze(0)
        ).contiguous()
        structural_decoder_prior_rank = int(
            torch.linalg.matrix_rank(gamma_aware_structural_decoder).item()
        )
        structural_decoder_prior_operator_norm = float(
            torch.linalg.matrix_norm(
                gamma_aware_structural_decoder,
                ord=2,
            ).item()
        )
        structural_decoder_left_inverse_error = float(
            torch.linalg.matrix_norm(
                gamma_aware_structural_decoder @ normalized_chart
                - torch.eye(target_hidden, dtype=torch.float64),
                ord=2,
            ).item()
        )
        if (
            structural_decoder_prior_rank != target_hidden
            or not math.isfinite(structural_decoder_prior_operator_norm)
            or structural_decoder_prior_operator_norm <= torch.finfo(torch.float64).tiny
            or not math.isfinite(structural_decoder_left_inverse_error)
            or structural_decoder_left_inverse_error > 1e-8
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Gamma-aware normalized-input structural decoder is not a "
                "finite full-rank left inverse of the normalized chart."
            )
        unit_structural_decoder = (
            gamma_aware_structural_decoder / structural_decoder_prior_operator_norm
        ).contiguous()

        mapped_source_down = (chart.T @ source_down).contiguous()
        normalized: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases):
            prefix = f"cases[{case_index}]"
            target_input = _float_tensor(
                case.target_input, f"{prefix}.target_input", ndim=2
            )
            parent_input = _float_tensor(
                case.parent_normalized_input,
                f"{prefix}.parent_normalized_input",
                ndim=2,
            )
            gate = _float_tensor(case.parent_gate, f"{prefix}.parent_gate", ndim=2)
            up = _float_tensor(case.parent_up, f"{prefix}.parent_up", ndim=2)
            post = _float_tensor(
                case.parent_post_swiglu,
                f"{prefix}.parent_post_swiglu",
                ndim=2,
            )
            endpoint = _float_tensor(
                case.parent_post_mlp_residual_target_space,
                f"{prefix}.parent_post_mlp_residual_target_space",
                ndim=2,
            )
            target_residual = _float_tensor(
                case.target_pre_mlp_residual,
                f"{prefix}.target_pre_mlp_residual",
                ndim=2,
            )
            rows = int(target_input.shape[0])
            if (
                tuple(target_input.shape) != (rows, target_hidden)
                or tuple(parent_input.shape) != (rows, source_hidden)
                or tuple(gate.shape) != (rows, source_intermediate)
                or tuple(up.shape) != tuple(gate.shape)
                or tuple(post.shape) != tuple(gate.shape)
                or tuple(endpoint.shape) != (rows, target_hidden)
                or tuple(target_residual.shape) != tuple(endpoint.shape)
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix} tensor shapes do not match source/target topology."
                )
            if case.parent_contribution is None:
                contribution = post * torch.linalg.vector_norm(
                    mapped_source_down, dim=0
                ).unsqueeze(0)
            else:
                raw = case.parent_contribution
                ndim = getattr(raw, "ndim", None)
                if ndim not in {2, 3}:
                    raise Lfm2StructuralResidualAttentionFoldError(
                        f"{prefix}.parent_contribution must have rank 2 or 3."
                    )
                contribution = _float_tensor(
                    raw, f"{prefix}.parent_contribution", ndim=int(ndim)
                )
                if tuple(contribution.shape[:2]) != (
                    rows,
                    source_intermediate,
                ):
                    raise Lfm2StructuralResidualAttentionFoldError(
                        f"{prefix}.parent_contribution has incompatible axes."
                    )
                if contribution.ndim == 3:
                    contribution = torch.linalg.vector_norm(contribution, dim=-1)
            normalized.append(
                {
                    "target_input": target_input,
                    "parent_input": parent_input,
                    "gate": gate,
                    "up": up,
                    "post": post,
                    "contribution": contribution,
                    "endpoint": endpoint,
                    "target_residual": target_residual,
                    "sample_weight": normalized_case_weights[case_index],
                }
            )

        row_weights = torch.cat(
            [
                torch.full(
                    (int(value["target_input"].shape[0]),),
                    float(value["sample_weight"]),
                    dtype=torch.float64,
                )
                for value in normalized
            ],
            dim=0,
        )
        sqrt_row_weights = torch.sqrt(row_weights).unsqueeze(1)
        target_input_rows = torch.cat(
            [value["target_input"] for value in normalized], dim=0
        )
        parent_input_rows = torch.cat(
            [value["parent_input"] for value in normalized], dim=0
        )
        weighted_target_input = target_input_rows * sqrt_row_weights
        weighted_parent_input = parent_input_rows * sqrt_row_weights

        # The gamma-aware structural inverse is normalized once so the only
        # remaining scalar degree of freedom is fitted from all FIT rows and
        # must retain positive orientation.
        unit_structural_prediction = weighted_target_input @ unit_structural_decoder
        gain_denominator = float(unit_structural_prediction.square().sum().item())
        gain_numerator = float(
            (unit_structural_prediction * weighted_parent_input).sum().item()
        )
        if (
            not math.isfinite(gain_numerator)
            or not math.isfinite(gain_denominator)
            or gain_denominator <= torch.finfo(torch.float64).tiny
            or gain_numerator <= 0.0
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "FIT rows do not identify one strictly positive normalized-input "
                "structural gain."
            )
        global_gain = gain_numerator / gain_denominator
        structural_decoder = (unit_structural_decoder * global_gain).contiguous()

        design_u, design_s, design_vh = torch.linalg.svd(
            weighted_target_input, full_matrices=False
        )
        if int(design_s.numel()) == 0 or float(design_s.max().item()) <= 0.0:
            raise Lfm2StructuralResidualAttentionFoldError(
                "FIT normalized child inputs have zero rank."
            )
        design_tolerance = (
            max(weighted_target_input.shape)
            * torch.finfo(torch.float64).eps
            * float(design_s.max().item())
        )
        fit_rank = int((design_s > design_tolerance).sum().item())
        if fit_rank <= 0:
            raise Lfm2StructuralResidualAttentionFoldError(
                "FIT normalized child inputs have zero numerical rank."
            )
        supported_u = design_u[:, :fit_rank]
        supported_s = design_s[:fit_rank]
        supported_vh = design_vh[:fit_rank]
        rowspace_projector = (supported_vh.T @ supported_vh).contiguous()
        design_energy_scale = float((supported_s.square().sum() / fit_rank).item())
        scaled_decoder_ridge = normalized_decoder_ridge * design_energy_scale
        residual = weighted_parent_input - weighted_target_input @ structural_decoder
        spectral_filter = supported_s / (supported_s.square() + scaled_decoder_ridge)
        correction = supported_vh.T @ (
            spectral_filter.unsqueeze(1) * (supported_u.T @ residual)
        )

        # rho=1 is fixed rather than tuned: the FIT-supported correction may
        # be no larger than the polar prior already supported by those rows.
        supported_relative_cap = 1.0
        supported_prior = rowspace_projector @ structural_decoder
        supported_prior_norm = float(torch.linalg.vector_norm(supported_prior).item())
        correction_norm = float(torch.linalg.vector_norm(correction).item())
        allowed_correction_norm = supported_relative_cap * supported_prior_norm
        correction_scale = 1.0
        if correction_norm > allowed_correction_norm:
            correction_scale = allowed_correction_norm / max(
                correction_norm, torch.finfo(torch.float64).tiny
            )
            correction = correction * correction_scale
        correction = correction.contiguous()
        normalized_input_decoder = (structural_decoder + correction).contiguous()

        identity = torch.eye(target_hidden, dtype=torch.float64)
        null_projector = identity - rowspace_projector
        correction_rowspace_leakage = float(
            torch.linalg.vector_norm(null_projector @ correction).item()
        )
        null_completion = null_projector @ normalized_input_decoder
        expected_null_completion = null_projector @ structural_decoder
        null_completion_error = float(
            torch.linalg.vector_norm(null_completion - expected_null_completion).item()
        )
        null_completion_operator_norm = float(
            torch.linalg.matrix_norm(null_completion, ord=2).item()
        )
        decoder_rank = int(torch.linalg.matrix_rank(normalized_input_decoder).item())
        decoder_prediction = target_input_rows @ normalized_input_decoder
        normalized_input_fit_error = _relative_rmse(
            decoder_prediction * sqrt_row_weights,
            parent_input_rows * sqrt_row_weights,
        )

        numerical_scale = max(
            float(torch.linalg.vector_norm(normalized_input_decoder).item()),
            global_gain,
            1.0,
        )
        numerical_tolerance = 1e-10 * numerical_scale
        if (
            decoder_rank != target_hidden
            or polar_orthogonality_error > 1e-10 * max(1.0, math.sqrt(target_hidden))
            or correction_rowspace_leakage > numerical_tolerance
            or null_completion_error > numerical_tolerance
            or null_completion_operator_norm > global_gain + numerical_tolerance
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Normalized-input decoder failed rank, polar, rowspace, or "
                "bounded-null-completion invariants."
            )

        all_gate_unweighted = torch.cat([value["gate"] for value in normalized], dim=0)
        all_up_unweighted = torch.cat([value["up"] for value in normalized], dim=0)
        all_post_unweighted = torch.cat([value["post"] for value in normalized], dim=0)
        all_gate = all_gate_unweighted * sqrt_row_weights
        all_up = all_up_unweighted * sqrt_row_weights
        all_post = all_post_unweighted * sqrt_row_weights
        all_contribution = (
            torch.cat([value["contribution"] for value in normalized], dim=0)
            * sqrt_row_weights
        )
        plan = build_gated_mlp_multi_anchor_cluster_plan(
            all_gate,
            all_up,
            all_post,
            all_contribution,
            target_neurons=target_intermediate,
            consumer_directions=mapped_source_down,
            consumer_axis=1,
            source_neurons=source_intermediate,
            anchors_per_source=anchors_per_source,
            max_candidate_targets=max_candidate_targets,
            ridge=cluster_ridge,
            max_coefficient_l1=max_coefficient_l1,
            max_iterations=cluster_max_iterations,
            distance_chunk_size=cluster_distance_chunk_size,
            similarity_chunk_size=similarity_chunk_size,
            max_similarity_products=max_similarity_products,
            max_projection_fits=max_projection_fits,
            max_sparse_entries=max_sparse_entries,
            max_workspace_elements=max_workspace_elements,
            structural_selection_signatures=weight_signature.selection_signature,
            structural_selection_signature_sha256=(weight_signature.signature_sha256),
            structural_selection_provenance=(weight_signature.report["report_sha256"]),
            calibration_provenance="cached_fit_rows_diagnostics_only",
        )
        if not plan.full_source_coverage:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Multi-anchor MLP plan did not prove full source coverage."
            )

        selected_gate = plan.select_gate_producer(source_gate, axis=0).to(torch.float64)
        selected_up = plan.select_up_producer(source_up, axis=0).to(torch.float64)
        gate_prior = (selected_gate @ normalized_input_decoder.T).contiguous()
        up_prior = (selected_up @ normalized_input_decoder.T).contiguous()
        aggregated_down = plan.aggregate_down_consumer(source_down, axis=1).to(
            torch.float64
        )
        down_prior = (chart.T @ aggregated_down).contiguous()

        # W1 and W3 are planted directly from the same decoder.  Only W2 gets
        # an endpoint correction, after target-dtype producer materialization.
        materialized_gate = gate_prior.to(
            device="cpu", dtype=template_dtypes["w1.weight"]
        ).contiguous()
        materialized_up = up_prior.to(
            device="cpu", dtype=template_dtypes["w3.weight"]
        ).contiguous()
        materialized_post = torch.nn.functional.silu(
            target_input_rows @ materialized_gate.to(torch.float64).T
        ) * (target_input_rows @ materialized_up.to(torch.float64).T)
        endpoint_target = torch.cat(
            [value["endpoint"] - value["target_residual"] for value in normalized],
            dim=0,
        )
        down_weight_fp64, down_fit = _fit_linear(
            materialized_post * sqrt_row_weights,
            endpoint_target * sqrt_row_weights,
            role="mlp_normalized_bridge_multi_anchor_down",
            ridge=ridge,
            prior=down_prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        materialized_down = down_weight_fp64.to(
            device="cpu", dtype=template_dtypes["w2.weight"]
        ).contiguous()
        compiled = {
            "w1.weight": materialized_gate.clone(),
            "w3.weight": materialized_up.clone(),
            "w2.weight": materialized_down.clone(),
        }
        endpoint_prediction = materialized_post @ materialized_down.to(torch.float64).T
        endpoint_error = _relative_rmse(
            endpoint_prediction * sqrt_row_weights,
            endpoint_target * sqrt_row_weights,
        )

    neuron_groups_lists: list[list[int]] = [[] for _ in range(target_intermediate)]
    for source_index, target_index in enumerate(plan.source_to_target):
        neuron_groups_lists[target_index].append(source_index)
    neuron_groups = tuple(tuple(value) for value in neuron_groups_lists)
    source_after = _state_sha256(source_values)
    source_logical = sum(int(value.numel()) for value in source_values.values())
    target_logical = sum(math.prod(shape) for shape in template_shapes.values())
    if (
        source_after != source_before
        or not plan.report.invariants.all_hold
        or sorted(source for group in neuron_groups for source in group)
        != list(range(source_intermediate))
        or target_logical >= source_logical
        or any(
            tuple(compiled[name].shape) != template_shapes[name]
            or compiled[name].requires_grad
            or compiled[name].grad_fn is not None
            or not bool(torch.isfinite(compiled[name]).all().item())
            for name in ("w1.weight", "w3.weight", "w2.weight")
        )
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The normalized-input real-medoid MLP compile failed its invariants."
        )
    fit_evidence_sha256 = lfm2_normalized_input_mlp_fit_evidence_sha256(
        cases, normalized_case_weights
    )
    return Lfm2CompiledNormalizedInputRealMedoidStructuralMlpFold(
        cluster_plan=plan,
        weight_complete_signature=weight_signature,
        gate_weight=compiled["w1.weight"],
        up_weight=compiled["w3.weight"],
        down_weight=compiled["w2.weight"],
        normalized_input_decoder=(
            normalized_input_decoder.detach().cpu().contiguous().clone()
        ),
        neuron_groups=neuron_groups,
        source_hidden_size=source_hidden,
        target_hidden_size=target_hidden,
        source_intermediate_size=source_intermediate,
        target_intermediate_size=target_intermediate,
        source_logical_parameters=source_logical,
        target_logical_parameters=target_logical,
        source_state_sha256_before=source_before,
        source_state_sha256_after=source_after,
        target_template_sha256=_template_sha256(template_spec),
        residual_chart_sha256=residual_chart_sha256,
        parent_ffn_norm_weight_sha256=parent_ffn_norm_weight_sha256,
        target_ffn_norm_weight_sha256=target_ffn_norm_weight_sha256,
        normalized_chart_sha256=lfm2_authenticated_tensor_sha256(normalized_chart),
        normalized_polar_sha256=lfm2_authenticated_tensor_sha256(normalized_polar),
        normalized_structural_decoder_prior_sha256=(
            lfm2_authenticated_tensor_sha256(gamma_aware_structural_decoder)
        ),
        normalized_input_decoder_sha256=lfm2_authenticated_tensor_sha256(
            normalized_input_decoder
        ),
        normalized_input_correction_sha256=lfm2_authenticated_tensor_sha256(correction),
        normalized_input_rowspace_projector_sha256=(
            lfm2_authenticated_tensor_sha256(rowspace_projector)
        ),
        producer_prior_sha256=_state_sha256(
            {"w1.weight": gate_prior, "w3.weight": up_prior}
        ),
        consumer_prior_sha256=_tensor_sha256(down_prior),
        fit_evidence_sha256=fit_evidence_sha256,
        fit_row_weights_sha256=lfm2_authenticated_tensor_sha256(row_weights),
        compiled_state_sha256=_state_sha256(compiled),
        residual_chart_rank=chart_rank,
        normalized_chart_rank=normalized_chart_rank,
        normalized_input_fit_rank=fit_rank,
        normalized_input_decoder_rank=decoder_rank,
        normalized_structural_decoder_prior_rank=(structural_decoder_prior_rank),
        normalized_structural_decoder_prior_operator_norm=(
            structural_decoder_prior_operator_norm
        ),
        normalized_structural_decoder_left_inverse_error=(
            structural_decoder_left_inverse_error
        ),
        normalized_input_global_gain=global_gain,
        normalized_input_ridge_policy_value=normalized_decoder_ridge,
        normalized_input_scaled_ridge=scaled_decoder_ridge,
        normalized_input_supported_relative_cap=supported_relative_cap,
        normalized_input_correction_scale=correction_scale,
        normalized_input_fit_relative_rmse=normalized_input_fit_error,
        normalized_input_polar_orthogonality_error=polar_orthogonality_error,
        normalized_input_correction_rowspace_leakage=(correction_rowspace_leakage),
        normalized_input_null_completion_error=null_completion_error,
        normalized_input_null_completion_operator_norm=(null_completion_operator_norm),
        normalized_input_null_completion_operator_norm_bound=global_gain,
        fit_case_weight_sum=math.fsum(normalized_case_weights),
        fit_row_weight_sum=float(row_weights.sum().item()),
        fit_row_weight_minimum=float(row_weights.min().item()),
        fit_row_weight_maximum=float(row_weights.max().item()),
        residual_endpoint_relative_rmse=endpoint_error,
        consumer_correction_report=down_fit,
        residual_chart_decoder_method=raw_chart_decoder_method,
        residual_chart_orthonormal_columns=chart_is_orthonormal,
        residual_chart_orthogonality_maximum_absolute_error=(chart_orthogonality_error),
        residual_chart_svd_calls=raw_chart_svd_calls,
        residual_chart_pseudoinverse_calls=raw_chart_pseudoinverse_calls,
    )


def compile_lfm2_structural_residual_mlp_fold(
    *,
    source_gate_weight: Any,
    source_up_weight: Any,
    source_down_weight: Any,
    target_gate_weight_template: Any,
    target_up_weight_template: Any,
    target_down_weight_template: Any,
    cases: Sequence[Lfm2CachedMlpFoldCase],
    gate_weight_prior: Any | None = None,
    up_weight_prior: Any | None = None,
    down_weight_prior: Any | None = None,
    residual_assignment_sha256: str | None = None,
    ridge: float = 0.0,
    max_prior_delta_ratio: float | None = 1.0,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    prepared_basis: Lfm2PreparedStructuralMlpFoldBasis | None = None,
) -> Lfm2CompiledStructuralMlpFold:
    """Compile a smaller dense SwiGLU from every assigned source neuron."""

    torch = _torch()
    if not cases or not all(
        isinstance(value, Lfm2CachedMlpFoldCase) for value in cases
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "cases must contain at least one Lfm2CachedMlpFoldCase."
        )
    ridge = _nonnegative_float(ridge, "ridge")
    cluster_ridge = _nonnegative_float(cluster_ridge, "cluster_ridge")
    cluster_max_iterations = _positive_int(
        cluster_max_iterations,
        "cluster_max_iterations",
    )
    cluster_distance_chunk_size = _positive_int(
        cluster_distance_chunk_size,
        "cluster_distance_chunk_size",
    )
    if max_prior_delta_ratio is not None:
        max_prior_delta_ratio = _nonnegative_float(
            max_prior_delta_ratio, "max_prior_delta_ratio"
        )
    if residual_assignment_sha256 is not None and not _valid_sha256_text(
        residual_assignment_sha256
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_assignment_sha256 must be a lowercase SHA-256 digest."
        )
    if prepared_basis is not None:
        return _compile_lfm2_structural_residual_mlp_fold_from_prepared_basis(
            source_gate_weight=source_gate_weight,
            source_up_weight=source_up_weight,
            source_down_weight=source_down_weight,
            target_gate_weight_template=target_gate_weight_template,
            target_up_weight_template=target_up_weight_template,
            target_down_weight_template=target_down_weight_template,
            cases=cases,
            gate_weight_prior=gate_weight_prior,
            up_weight_prior=up_weight_prior,
            down_weight_prior=down_weight_prior,
            residual_assignment_sha256=residual_assignment_sha256,
            ridge=ridge,
            max_prior_delta_ratio=max_prior_delta_ratio,
            cluster_ridge=cluster_ridge,
            cluster_max_iterations=cluster_max_iterations,
            cluster_distance_chunk_size=cluster_distance_chunk_size,
            prepared_basis=prepared_basis,
        )
    source_values = {
        "w1.weight": source_gate_weight,
        "w3.weight": source_up_weight,
        "w2.weight": source_down_weight,
    }
    source_before = _state_sha256(source_values)
    source_gate = _float_tensor(source_gate_weight, "source_gate_weight", ndim=2)
    source_up = _float_tensor(source_up_weight, "source_up_weight", ndim=2)
    source_down = _float_tensor(source_down_weight, "source_down_weight", ndim=2)
    source_intermediate, source_hidden = map(int, source_gate.shape)
    if tuple(source_up.shape) != tuple(source_gate.shape) or tuple(
        source_down.shape
    ) != (source_hidden, source_intermediate):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Source gate/up/down tensors do not form one standard LFM2 SwiGLU."
        )

    target_templates = {
        "w1.weight": target_gate_weight_template,
        "w3.weight": target_up_weight_template,
        "w2.weight": target_down_weight_template,
    }
    template_dtypes: dict[str, Any] = {}
    template_shapes: dict[str, tuple[int, ...]] = {}
    template_spec: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, value in target_templates.items():
        dtype, shape = _dense_template(value, f"target_{name}")
        template_dtypes[name] = dtype
        template_shapes[name] = shape
        template_spec[name] = (shape, str(dtype))
    target_intermediate, target_hidden = template_shapes["w1.weight"]
    expected_shapes = {
        "w1.weight": (target_intermediate, target_hidden),
        "w3.weight": (target_intermediate, target_hidden),
        "w2.weight": (target_hidden, target_intermediate),
    }
    if template_shapes != expected_shapes:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Target templates are not one standard atomic LFM2 SwiGLU set."
        )
    if target_hidden > source_hidden or target_intermediate > source_intermediate:
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target cannot expand the SwiGLU hidden or "
            "intermediate width."
        )
    if target_hidden == source_hidden and target_intermediate == source_intermediate:
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target must reduce the SwiGLU hidden or intermediate width."
        )
    if residual_assignment_sha256 is not None and (
        not isinstance(residual_assignment_sha256, str)
        or len(residual_assignment_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in residual_assignment_sha256
        )
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_assignment_sha256 must be a lowercase SHA-256 digest."
        )
    if source_hidden != target_hidden and residual_assignment_sha256 is None:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Cross-hidden SwiGLU compilation must be bound to the one global "
            "full-source sparse residual assignment."
        )

    supplied_priors = {
        "w1.weight": gate_weight_prior,
        "w3.weight": up_weight_prior,
        "w2.weight": down_weight_prior,
    }
    normalized_priors: dict[str, Any | None] = {}
    for name, prior in supplied_priors.items():
        if prior is None:
            normalized_priors[name] = None
            continue
        normalized_prior = _float_tensor(prior, f"source_derived_{name}_prior", ndim=2)
        if tuple(normalized_prior.shape) != template_shapes[name]:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"source-derived {name} prior must have target shape "
                f"{template_shapes[name]}."
            )
        normalized_priors[name] = normalized_prior
    prior_sha = (
        None
        if all(value is None for value in normalized_priors.values())
        else _state_sha256(
            {
                name: value
                for name, value in normalized_priors.items()
                if value is not None
            }
        )
    )

    normalized: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        prefix = f"cases[{case_index}]"
        target_input = _float_tensor(
            case.target_input, f"{prefix}.target_input", ndim=2
        )
        gate = _float_tensor(case.parent_gate, f"{prefix}.parent_gate", ndim=2)
        up = _float_tensor(case.parent_up, f"{prefix}.parent_up", ndim=2)
        post = _float_tensor(
            case.parent_post_swiglu, f"{prefix}.parent_post_swiglu", ndim=2
        )
        parent_endpoint = _float_tensor(
            case.parent_post_mlp_residual_target_space,
            f"{prefix}.parent_post_mlp_residual_target_space",
            ndim=2,
        )
        target_residual = _float_tensor(
            case.target_pre_mlp_residual,
            f"{prefix}.target_pre_mlp_residual",
            ndim=2,
        )
        rows = int(target_input.shape[0])
        if (
            tuple(target_input.shape) != (rows, target_hidden)
            or tuple(gate.shape) != (rows, source_intermediate)
            or tuple(up.shape) != tuple(gate.shape)
            or tuple(post.shape) != tuple(gate.shape)
            or tuple(parent_endpoint.shape) != (rows, target_hidden)
            or tuple(target_residual.shape) != tuple(parent_endpoint.shape)
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{prefix} tensor shapes do not match source/target SwiGLU topology."
            )
        if case.parent_contribution is None:
            contribution = post * torch.linalg.vector_norm(
                source_down, dim=0
            ).unsqueeze(0)
        else:
            raw = case.parent_contribution
            ndim = getattr(raw, "ndim", None)
            if ndim not in {2, 3}:
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix}.parent_contribution must have rank 2 or 3."
                )
            contribution = _float_tensor(
                raw, f"{prefix}.parent_contribution", ndim=int(ndim)
            )
            if tuple(contribution.shape[:2]) != (rows, source_intermediate):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix}.parent_contribution has incompatible leading axes."
                )
            if contribution.ndim == 3:
                contribution = torch.linalg.vector_norm(contribution, dim=-1)
        normalized.append(
            {
                "target_input": target_input,
                "gate": gate,
                "up": up,
                "post": post,
                "contribution": contribution,
                "parent_endpoint": parent_endpoint,
                "target_residual": target_residual,
            }
        )

    with torch.no_grad():
        all_gate = torch.cat([value["gate"] for value in normalized], dim=0)
        all_up = torch.cat([value["up"] for value in normalized], dim=0)
        all_post = torch.cat([value["post"] for value in normalized], dim=0)
        all_contribution = torch.cat(
            [value["contribution"] for value in normalized], dim=0
        )
        cluster = build_gated_mlp_activation_cluster_plan(
            all_gate,
            all_up,
            all_post,
            all_contribution,
            target_neurons=target_intermediate,
            source_neurons=source_intermediate,
            ridge=cluster_ridge,
            max_iterations=_positive_int(
                cluster_max_iterations, "cluster_max_iterations"
            ),
            distance_chunk_size=_positive_int(
                cluster_distance_chunk_size, "cluster_distance_chunk_size"
            ),
        )
        if not cluster.report.full_source_coverage:
            raise Lfm2StructuralResidualAttentionFoldError(
                "SwiGLU clustering did not prove full source coverage."
            )
        importance = all_post.square().mean(dim=0) + all_contribution.square().mean(
            dim=0
        )
        gate_targets = []
        up_targets = []
        neuron_groups: tuple[tuple[int, ...], ...] | None = None
        neuron_weights: tuple[tuple[float, ...], ...] | None = None
        for value in normalized:
            gate_target, groups, weights = _weighted_group_rows(
                value["gate"],
                cluster.source_to_target,
                target_intermediate,
                importance,
            )
            up_target, groups_2, weights_2 = _weighted_group_rows(
                value["up"],
                cluster.source_to_target,
                target_intermediate,
                importance,
            )
            if groups != groups_2 or weights != weights_2:
                raise Lfm2StructuralResidualAttentionFoldError(
                    "Gate and up barycentric assignments differ."
                )
            if neuron_groups is None:
                neuron_groups, neuron_weights = groups, weights
            elif neuron_groups != groups or neuron_weights != weights:
                raise Lfm2StructuralResidualAttentionFoldError(
                    "SwiGLU barycentric assignments changed between cases."
                )
            gate_targets.append(gate_target)
            up_targets.append(up_target)

        target_input_rows = torch.cat(
            [value["target_input"] for value in normalized], dim=0
        )
        gate_weight, gate_fit = _fit_linear(
            target_input_rows,
            torch.cat(gate_targets, dim=0),
            role="mlp_gate",
            ridge=ridge,
            prior=normalized_priors["w1.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        up_weight, up_fit = _fit_linear(
            target_input_rows,
            torch.cat(up_targets, dim=0),
            role="mlp_up",
            ridge=ridge,
            prior=normalized_priors["w3.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        child_post = torch.nn.functional.silu(target_input_rows @ gate_weight.T) * (
            target_input_rows @ up_weight.T
        )
        endpoint_target = torch.cat(
            [
                value["parent_endpoint"] - value["target_residual"]
                for value in normalized
            ],
            dim=0,
        )
        down_weight, down_fit = _fit_linear(
            child_post,
            endpoint_target,
            role="mlp_down",
            ridge=ridge,
            prior=normalized_priors["w2.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        fp64_state = {
            "w1.weight": gate_weight,
            "w3.weight": up_weight,
            "w2.weight": down_weight,
        }
        compiled = {
            name: fp64_state[name]
            .to(device="cpu", dtype=template_dtypes[name])
            .contiguous()
            .clone()
            for name in ("w1.weight", "w3.weight", "w2.weight")
        }
        materialized_gate = compiled["w1.weight"].to(torch.float64)
        materialized_up = compiled["w3.weight"].to(torch.float64)
        materialized_down = compiled["w2.weight"].to(torch.float64)
        materialized_post = torch.nn.functional.silu(
            target_input_rows @ materialized_gate.T
        ) * (target_input_rows @ materialized_up.T)
        endpoint_prediction = materialized_post @ materialized_down.T
        endpoint_error = _relative_rmse(endpoint_prediction, endpoint_target)

    source_after = _state_sha256(source_values)
    source_logical = sum(int(value.numel()) for value in source_values.values())
    target_logical = sum(math.prod(shape) for shape in template_shapes.values())
    if (
        source_before != source_after
        or neuron_groups is None
        or neuron_weights is None
        or sorted(source for group in neuron_groups for source in group)
        != list(range(source_intermediate))
        or target_logical >= source_logical
        or not all(
            int(target_templates[name].numel()) < int(source_values[name].numel())
            for name in ("w1.weight", "w3.weight", "w2.weight")
        )
        or any(
            tuple(compiled[name].shape) != template_shapes[name]
            for name in ("w1.weight", "w3.weight", "w2.weight")
        )
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The compiled SwiGLU failed its structural invariants."
        )
    prepared_gate_targets = tuple(
        value.detach().to(device="cpu").contiguous().clone() for value in gate_targets
    )
    prepared_up_targets = tuple(
        value.detach().to(device="cpu").contiguous().clone() for value in up_targets
    )
    prepared_parent_endpoints = tuple(
        value["parent_endpoint"].detach().to(device="cpu").contiguous().clone()
        for value in normalized
    )
    prepared_basis_for_report = Lfm2PreparedStructuralMlpFoldBasis(
        cluster_plan=cluster,
        gate_targets=prepared_gate_targets,
        up_targets=prepared_up_targets,
        parent_endpoints=prepared_parent_endpoints,
        parent_case_sha256=tuple(
            _raw_mlp_parent_case_sha256(case, case_index=case_index)
            for case_index, case in enumerate(cases)
        ),
        case_rows=tuple(int(value["target_input"].shape[0]) for value in normalized),
        neuron_groups=neuron_groups,
        neuron_barycentric_weights=neuron_weights,
        source_hidden_size=source_hidden,
        target_hidden_size=target_hidden,
        source_intermediate_size=source_intermediate,
        target_intermediate_size=target_intermediate,
        source_logical_parameters=source_logical,
        target_logical_parameters=target_logical,
        source_state_sha256=source_before,
        target_template_sha256=_template_sha256(template_spec),
        residual_assignment_sha256=residual_assignment_sha256,
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
        prepared_tensor_sha256=_prepared_mlp_tensor_sha256(
            prepared_gate_targets,
            prepared_up_targets,
            prepared_parent_endpoints,
        ),
    )
    return Lfm2CompiledStructuralMlpFold(
        cluster_plan=cluster,
        gate_weight=compiled["w1.weight"],
        up_weight=compiled["w3.weight"],
        down_weight=compiled["w2.weight"],
        neuron_groups=neuron_groups,
        neuron_barycentric_weights=neuron_weights,
        source_hidden_size=source_hidden,
        target_hidden_size=target_hidden,
        source_intermediate_size=source_intermediate,
        target_intermediate_size=target_intermediate,
        source_logical_parameters=source_logical,
        target_logical_parameters=target_logical,
        source_state_sha256_before=source_before,
        source_state_sha256_after=source_after,
        target_template_sha256=_template_sha256(template_spec),
        residual_assignment_sha256=residual_assignment_sha256,
        source_derived_prior_sha256=prior_sha,
        prepared_basis_sha256=prepared_basis_for_report.preparation_sha256,
        prepared_basis_reused=False,
        compiled_state_sha256=_state_sha256(compiled),
        residual_endpoint_relative_rmse=endpoint_error,
        prior_correction_reports=(gate_fit, up_fit, down_fit),
    )


def _run_compiled_short_conv_contexts(
    cases: Sequence[Mapping[str, Any]],
    *,
    in_proj_weight: Any,
    in_proj_bias: Any | None,
    conv_weight: Any,
    conv_bias: Any | None,
) -> list[Any]:
    """Run the actual causal B*x -> depthwise FIR -> C*v child path."""

    torch = _torch()
    target_hidden = int(conv_weight.shape[0])
    fir_length = int(conv_weight.shape[2])
    state = None
    contexts = []
    for case_index, value in enumerate(cases):
        target_input = value["target_input"]
        batch, sequence, _hidden = map(int, target_input.shape)
        if case_index == 0 or value["reset_state"]:
            state = torch.zeros(
                (batch, target_hidden, fir_length - 1),
                dtype=torch.float64,
            )
        elif state is None or int(state.shape[0]) != batch:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Chronological short-conv cases changed batch size without "
                "resetting FIR state."
            )
        projected = target_input @ in_proj_weight.T
        if in_proj_bias is not None:
            projected = projected + in_proj_bias
        b_value, c_value, x_value = projected.chunk(3, dim=-1)
        gated = (b_value * x_value).transpose(1, 2).contiguous()
        history = torch.cat((state, gated), dim=2)
        filtered = torch.nn.functional.conv1d(
            history,
            conv_weight,
            bias=conv_bias,
            groups=target_hidden,
        )
        if int(filtered.shape[2]) != sequence:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Compiled grouped FIR produced a non-causal sequence length."
            )
        if fir_length > 1:
            state = history[:, :, -(fir_length - 1) :].contiguous()
        else:
            state = history[:, :, :0].contiguous()
        contexts.append(c_value * filtered.transpose(1, 2))
    return contexts


def compile_lfm2_structural_residual_short_conv_fold(
    *,
    source_in_proj_weight: Any,
    source_conv_weight: Any,
    source_out_proj_weight: Any,
    target_in_proj_weight_template: Any,
    target_conv_weight_template: Any,
    target_out_proj_weight_template: Any,
    cases: Sequence[Lfm2CachedShortConvFoldCase],
    residual_assignment: Lfm2SparseResidualAssignment,
    external_residual_assignment: Lfm2SparseResidualAssignment | None = None,
    source_in_proj_bias: Any | None = None,
    source_conv_bias: Any | None = None,
    source_out_proj_bias: Any | None = None,
    target_in_proj_bias_template: Any | None = None,
    target_conv_bias_template: Any | None = None,
    target_out_proj_bias_template: Any | None = None,
    in_proj_weight_prior: Any | None = None,
    in_proj_bias_prior: Any | None = None,
    out_proj_weight_prior: Any | None = None,
    out_proj_bias_prior: Any | None = None,
    ridge: float = 0.0,
    max_prior_delta_ratio: float | None = 1.0,
    _producer_only: bool = False,
) -> Lfm2CompiledStructuralShortConvFold | Lfm2CompiledStructuralShortConvProducerFold:
    """Compile a smaller dense LFM2 short-conv from every source channel.

    One sparse operator-local assignment is shared by B/C/x activation
    transport and the grouped FIR rows.  It need not represent the model-wide
    residual coordinates: dense global bases must supply a separate sparse
    B/C/x assignment here.  The input and output projections are fitted only
    after executing the compiled child's real multiplicative and causal FIR
    path.
    """

    torch = _torch()
    if not isinstance(_producer_only, bool):
        raise Lfm2StructuralResidualAttentionFoldError(
            "_producer_only must be boolean."
        )
    if not isinstance(residual_assignment, Lfm2SparseResidualAssignment):
        raise TypeError("residual_assignment must be Lfm2SparseResidualAssignment.")
    if external_residual_assignment is not None and not isinstance(
        external_residual_assignment, Lfm2SparseResidualAssignment
    ):
        raise TypeError(
            "external_residual_assignment must be Lfm2SparseResidualAssignment or None."
        )
    if not cases or not all(
        isinstance(value, Lfm2CachedShortConvFoldCase) for value in cases
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "cases must contain at least one Lfm2CachedShortConvFoldCase."
        )
    ridge = _nonnegative_float(ridge, "ridge")
    if max_prior_delta_ratio is not None:
        max_prior_delta_ratio = _nonnegative_float(
            max_prior_delta_ratio, "max_prior_delta_ratio"
        )

    source_bias_values = (
        source_in_proj_bias,
        source_conv_bias,
        source_out_proj_bias,
    )
    target_bias_templates = (
        target_in_proj_bias_template,
        target_conv_bias_template,
        target_out_proj_bias_template,
    )
    if any(value is not None for value in source_bias_values) and not all(
        value is not None for value in source_bias_values
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "All source LFM2 short-conv biases must be present or absent together."
        )
    if any(value is not None for value in target_bias_templates) and not all(
        value is not None for value in target_bias_templates
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "All target LFM2 short-conv bias templates must be present or "
            "absent together."
        )
    source_bias = source_in_proj_bias is not None
    target_bias = target_in_proj_bias_template is not None
    if source_bias != target_bias:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Source and target LFM2 short-conv bias topology must match."
        )

    source_values = {
        "in_proj.weight": source_in_proj_weight,
        "conv.weight": source_conv_weight,
        "out_proj.weight": source_out_proj_weight,
    }
    if source_bias:
        source_values.update(
            {
                "in_proj.bias": source_in_proj_bias,
                "conv.bias": source_conv_bias,
                "out_proj.bias": source_out_proj_bias,
            }
        )
    source_before = _state_sha256(source_values)
    source_in = _float_tensor(source_in_proj_weight, "source_in_proj_weight", ndim=2)
    source_conv = _float_tensor(source_conv_weight, "source_conv_weight", ndim=3)
    source_out = _float_tensor(source_out_proj_weight, "source_out_proj_weight", ndim=2)
    source_residual_width = int(source_out.shape[0])
    source_channel_width = int(source_conv.shape[0])
    fir_length = int(source_conv.shape[2])
    if (
        tuple(source_in.shape) != (3 * source_channel_width, source_residual_width)
        or tuple(source_conv.shape) != (source_channel_width, 1, fir_length)
        or tuple(source_out.shape) != (source_residual_width, source_channel_width)
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Source tensors do not form one B/C/x depthwise LFM2 short-conv "
            "operator with a common internal channel width."
        )
    normalized_source_biases: dict[str, Any | None] = {
        "in_proj.bias": None,
        "conv.bias": None,
        "out_proj.bias": None,
    }
    if source_bias:
        for name, value, width in (
            ("in_proj.bias", source_in_proj_bias, 3 * source_channel_width),
            ("conv.bias", source_conv_bias, source_channel_width),
            ("out_proj.bias", source_out_proj_bias, source_residual_width),
        ):
            normalized = _float_tensor(value, f"source_{name}", ndim=1)
            if tuple(normalized.shape) != (width,):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"Source {name} has the wrong canonical shape."
                )
            normalized_source_biases[name] = normalized

    target_templates = {
        "in_proj.weight": target_in_proj_weight_template,
        "conv.weight": target_conv_weight_template,
        "out_proj.weight": target_out_proj_weight_template,
    }
    if target_bias:
        target_templates.update(
            {
                "in_proj.bias": target_in_proj_bias_template,
                "conv.bias": target_conv_bias_template,
                "out_proj.bias": target_out_proj_bias_template,
            }
        )
    template_dtypes: dict[str, Any] = {}
    template_shapes: dict[str, tuple[int, ...]] = {}
    template_spec: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, value in target_templates.items():
        ndim = 3 if name == "conv.weight" else (1 if name.endswith(".bias") else 2)
        dtype, shape = _dense_template(value, f"target_{name}", ndim=ndim)
        template_dtypes[name] = dtype
        template_shapes[name] = shape
        template_spec[name] = (shape, str(dtype))
    target_residual_width = int(template_shapes["out_proj.weight"][0])
    target_channel_width = int(template_shapes["conv.weight"][0])
    expected_shapes = {
        "in_proj.weight": (3 * target_channel_width, target_residual_width),
        "conv.weight": (target_channel_width, 1, fir_length),
        "out_proj.weight": (target_residual_width, target_channel_width),
    }
    if target_bias:
        expected_shapes.update(
            {
                "in_proj.bias": (3 * target_channel_width,),
                "conv.bias": (target_channel_width,),
                "out_proj.bias": (target_residual_width,),
            }
        )
    if template_shapes != expected_shapes:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Target templates are not one atomic LFM2 short-conv set with a "
            "common internal channel width and the source FIR length."
        )
    if (
        target_channel_width > source_channel_width
        or target_residual_width > source_residual_width
        or (
            target_channel_width == source_channel_width
            and target_residual_width == source_residual_width
        )
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural short-conv target must reduce at least one width "
            "without enlarging either width."
        )
    if (
        residual_assignment.source_width != source_channel_width
        or residual_assignment.target_width != target_channel_width
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The sparse B/C/x channel assignment does not match the source "
            "and target short-conv widths."
        )
    if external_residual_assignment is None:
        if (
            residual_assignment.source_width != source_residual_width
            or residual_assignment.target_width != target_residual_width
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "An external_residual_assignment is required when LFM2 "
                "short-conv internal and residual widths are decoupled."
            )
        external_residual_assignment = residual_assignment
    if (
        external_residual_assignment.source_width != source_residual_width
        or external_residual_assignment.target_width != target_residual_width
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The external residual assignment does not match the source and "
            "target residual widths."
        )
    decoupled_geometry = (
        source_channel_width != source_residual_width
        or target_channel_width != target_residual_width
    )

    normalized: list[dict[str, Any]] = []
    source_out_column_norm = torch.linalg.vector_norm(source_out, dim=0)
    for case_index, case in enumerate(cases):
        prefix = f"cases[{case_index}]"
        if not isinstance(case.reset_state, bool):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{prefix}.reset_state must be boolean."
            )
        target_input = _float_tensor(
            case.target_input, f"{prefix}.target_input", ndim=3
        )
        parent_b = _float_tensor(case.parent_b, f"{prefix}.parent_b", ndim=3)
        parent_c = _float_tensor(case.parent_c, f"{prefix}.parent_c", ndim=3)
        parent_x = _float_tensor(case.parent_x, f"{prefix}.parent_x", ndim=3)
        parent_endpoint = _float_tensor(
            case.parent_post_operator_residual_target_space,
            f"{prefix}.parent_post_operator_residual_target_space",
            ndim=3,
        )
        target_residual = _float_tensor(
            case.target_pre_operator_residual,
            f"{prefix}.target_pre_operator_residual",
            ndim=3,
        )
        batch, sequence, observed_target_hidden = map(int, target_input.shape)
        if (
            observed_target_hidden != target_residual_width
            or tuple(parent_b.shape) != (batch, sequence, source_channel_width)
            or tuple(parent_c.shape) != tuple(parent_b.shape)
            or tuple(parent_x.shape) != tuple(parent_b.shape)
            or tuple(parent_endpoint.shape) != (batch, sequence, target_residual_width)
            or tuple(target_residual.shape) != tuple(parent_endpoint.shape)
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{prefix} tensor shapes do not match the source/target "
                "short-conv topology."
            )
        if case.parent_contribution is None:
            contribution = (
                parent_c
                * parent_b
                * parent_x
                * source_out_column_norm.reshape(1, 1, -1)
            )
        else:
            raw = case.parent_contribution
            ndim = getattr(raw, "ndim", None)
            if ndim not in {3, 4}:
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix}.parent_contribution must have rank 3 or 4."
                )
            contribution = _float_tensor(
                raw, f"{prefix}.parent_contribution", ndim=int(ndim)
            )
            if tuple(contribution.shape[:3]) != (
                batch,
                sequence,
                source_channel_width,
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix}.parent_contribution has incompatible leading axes."
                )
            if contribution.ndim == 4:
                contribution = torch.linalg.vector_norm(contribution, dim=-1)
        normalized.append(
            {
                "target_input": target_input,
                "parent_b": parent_b,
                "parent_c": parent_c,
                "parent_x": parent_x,
                "parent_endpoint": parent_endpoint,
                "target_residual": target_residual,
                "contribution": contribution,
                "reset_state": case.reset_state,
            }
        )

    with torch.no_grad():
        all_b = torch.cat(
            [
                value["parent_b"].reshape(-1, source_channel_width)
                for value in normalized
            ],
            dim=0,
        )
        all_c = torch.cat(
            [
                value["parent_c"].reshape(-1, source_channel_width)
                for value in normalized
            ],
            dim=0,
        )
        all_x = torch.cat(
            [
                value["parent_x"].reshape(-1, source_channel_width)
                for value in normalized
            ],
            dim=0,
        )
        all_contribution = torch.cat(
            [
                value["contribution"].reshape(-1, source_channel_width)
                for value in normalized
            ],
            dim=0,
        )
        importance = (
            all_b.square().mean(dim=0)
            + all_c.square().mean(dim=0)
            + all_x.square().mean(dim=0)
            + (all_b * all_x).square().mean(dim=0)
            + all_contribution.square().mean(dim=0)
        )
        coefficients = torch.tensor(
            residual_assignment.coefficients, dtype=torch.float64
        )
        weighted_importance = importance * coefficients.abs()

        b_targets = []
        c_targets = []
        x_targets = []
        channel_groups: tuple[tuple[int, ...], ...] | None = None
        channel_weights: tuple[tuple[float, ...], ...] | None = None
        for value in normalized:
            batch, sequence, _width = map(int, value["target_input"].shape)
            case_targets = []
            observed_groups = None
            observed_weights = None
            for role in ("parent_b", "parent_c", "parent_x"):
                folded, groups, weights = _weighted_group_rows(
                    value[role].reshape(batch * sequence, source_channel_width),
                    residual_assignment.source_to_target,
                    target_channel_width,
                    weighted_importance,
                )
                case_targets.append(
                    folded.reshape(batch, sequence, target_channel_width)
                )
                if observed_groups is None:
                    observed_groups, observed_weights = groups, weights
                elif observed_groups != groups or observed_weights != weights:
                    raise Lfm2StructuralResidualAttentionFoldError(
                        "B/C/x sparse channel barycenters disagree."
                    )
            if channel_groups is None:
                channel_groups = observed_groups
                channel_weights = observed_weights
            elif (
                channel_groups != observed_groups or channel_weights != observed_weights
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    "Short-conv channel barycenters changed between cases."
                )
            b_targets.append(case_targets[0])
            c_targets.append(case_targets[1])
            x_targets.append(case_targets[2])
        assert channel_groups is not None
        assert channel_weights is not None

        producer_reduce = torch.zeros(
            (target_channel_width, source_channel_width), dtype=torch.float64
        )
        for target, (members, weights) in enumerate(
            zip(channel_groups, channel_weights)
        ):
            for source, weight in zip(members, weights):
                producer_reduce[target, source] = weight
        channel_reduce = torch.zeros(
            (target_channel_width, source_channel_width), dtype=torch.float64
        )
        for source, target in enumerate(residual_assignment.source_to_target):
            channel_reduce[target, source] = coefficients[source]
        channel_row_energy = channel_reduce.square().sum(dim=1)
        if bool((channel_row_energy <= torch.finfo(torch.float64).eps).any().item()):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Sparse channel assignment cannot construct a right inverse."
            )
        channel_lift = channel_reduce.T / channel_row_energy.reshape(1, -1)
        if not torch.allclose(
            channel_reduce @ channel_lift,
            torch.eye(target_channel_width, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Sparse channel assignment failed its right-inverse invariant."
            )

        external_coefficients = torch.tensor(
            external_residual_assignment.coefficients, dtype=torch.float64
        )
        residual_reduce = torch.zeros(
            (target_residual_width, source_residual_width), dtype=torch.float64
        )
        for source, target in enumerate(external_residual_assignment.source_to_target):
            residual_reduce[target, source] = external_coefficients[source]
        row_energy = residual_reduce.square().sum(dim=1)
        if bool((row_energy <= torch.finfo(torch.float64).eps).any().item()):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Sparse residual assignment cannot construct a right inverse."
            )
        residual_lift = residual_reduce.T / row_energy.reshape(1, -1)
        if not torch.allclose(
            residual_reduce @ residual_lift,
            torch.eye(target_residual_width, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Sparse residual assignment failed its right-inverse invariant."
            )

        source_in_blocks = source_in.chunk(3, dim=0)
        derived_in_prior = torch.cat(
            [producer_reduce @ block @ residual_lift for block in source_in_blocks],
            dim=0,
        )
        derived_out_prior = (residual_reduce @ source_out @ channel_lift).contiguous()
        kernel_rows, kernel_groups, kernel_weights = _weighted_group_rows(
            source_conv.unsqueeze(0),
            residual_assignment.source_to_target,
            target_channel_width,
            weighted_importance,
        )
        if kernel_groups != channel_groups or kernel_weights != channel_weights:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Grouped FIR rows do not use the B/C/x channel assignment."
            )
        conv_weight = kernel_rows.squeeze(0).contiguous()

        derived_in_bias = None
        conv_bias = None
        derived_out_bias = None
        if target_bias:
            source_in_bias = normalized_source_biases["in_proj.bias"]
            source_conv_bias = normalized_source_biases["conv.bias"]
            source_out_bias = normalized_source_biases["out_proj.bias"]
            derived_in_bias = torch.cat(
                [producer_reduce @ block for block in source_in_bias.chunk(3, dim=0)],
                dim=0,
            )
            folded_conv_bias, bias_groups, bias_weights = _weighted_group_rows(
                source_conv_bias.reshape(1, source_channel_width),
                residual_assignment.source_to_target,
                target_channel_width,
                weighted_importance,
            )
            if bias_groups != channel_groups or bias_weights != channel_weights:
                raise Lfm2StructuralResidualAttentionFoldError(
                    "FIR bias does not use the B/C/x channel assignment."
                )
            conv_bias = folded_conv_bias.squeeze(0).contiguous()
            derived_out_bias = (residual_reduce @ source_out_bias).contiguous()

        def normalized_prior(
            supplied: Any | None,
            derived: Any,
            name: str,
            ndim: int,
        ) -> Any:
            if supplied is None:
                return derived
            result = _float_tensor(supplied, name, ndim=ndim)
            if tuple(result.shape) != tuple(derived.shape):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{name} must have source-derived target shape "
                    f"{tuple(derived.shape)}."
                )
            return result

        actual_in_prior = normalized_prior(
            in_proj_weight_prior,
            derived_in_prior,
            "in_proj_weight_prior",
            2,
        )
        actual_out_prior = normalized_prior(
            out_proj_weight_prior,
            derived_out_prior,
            "out_proj_weight_prior",
            2,
        )
        actual_in_bias_prior = None
        actual_out_bias_prior = None
        if target_bias:
            actual_in_bias_prior = normalized_prior(
                in_proj_bias_prior,
                derived_in_bias,
                "in_proj_bias_prior",
                1,
            )
            actual_out_bias_prior = normalized_prior(
                out_proj_bias_prior,
                derived_out_bias,
                "out_proj_bias_prior",
                1,
            )
        elif in_proj_bias_prior is not None or out_proj_bias_prior is not None:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Bias priors were supplied for a bias-free short-conv target."
            )
        prior_values = {
            "in_proj.weight": actual_in_prior,
            "out_proj.weight": actual_out_prior,
            "conv.weight": conv_weight,
        }
        if target_bias:
            prior_values.update(
                {
                    "in_proj.bias": actual_in_bias_prior,
                    "out_proj.bias": actual_out_bias_prior,
                    "conv.bias": conv_bias,
                }
            )
        prior_sha = _state_sha256(prior_values)

        target_input_rows = torch.cat(
            [
                value["target_input"].reshape(-1, target_residual_width)
                for value in normalized
            ],
            dim=0,
        )
        bcx_targets = torch.cat(
            [
                torch.cat((b_value, c_value, x_value), dim=-1).reshape(
                    -1, 3 * target_channel_width
                )
                for b_value, c_value, x_value in zip(b_targets, c_targets, x_targets)
            ],
            dim=0,
        )
        in_proj_weight, in_proj_bias, in_fit = _fit_affine(
            target_input_rows,
            bcx_targets,
            role="short_conv_in_proj",
            ridge=ridge,
            weight_prior=actual_in_prior,
            bias_prior=actual_in_bias_prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
            fit_bias=target_bias,
        )
        if _producer_only:
            producer_fp64 = {
                "in_proj.weight": in_proj_weight,
                "conv.weight": conv_weight,
            }
            producer_prior_values = {
                "in_proj.weight": actual_in_prior,
                "conv.weight": conv_weight,
            }
            if target_bias:
                producer_fp64.update(
                    {
                        "in_proj.bias": in_proj_bias,
                        "conv.bias": conv_bias,
                    }
                )
                producer_prior_values.update(
                    {
                        "in_proj.bias": actual_in_bias_prior,
                        "conv.bias": conv_bias,
                    }
                )
            producer_compiled = {
                name: value.to(device="cpu", dtype=template_dtypes[name])
                .contiguous()
                .clone()
                for name, value in producer_fp64.items()
            }
            producer_source_values = {
                "in_proj.weight": source_in_proj_weight,
                "conv.weight": source_conv_weight,
            }
            if source_bias:
                producer_source_values.update(
                    {
                        "in_proj.bias": source_in_proj_bias,
                        "conv.bias": source_conv_bias,
                    }
                )
            producer_template_spec = {
                name: template_spec[name] for name in producer_compiled
            }
            source_after = _state_sha256(source_values)
            source_logical = sum(
                int(value.numel()) for value in producer_source_values.values()
            )
            target_logical = sum(
                int(value.numel()) for value in producer_compiled.values()
            )
            if (
                source_before != source_after
                or channel_groups is None
                or channel_weights is None
                or sorted(source for group in channel_groups for source in group)
                != list(range(source_channel_width))
                or target_logical >= source_logical
                or any(
                    tuple(producer_compiled[name].shape) != template_shapes[name]
                    for name in producer_compiled
                )
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    "The split short-conv producer failed its invariants."
                )
            return Lfm2CompiledStructuralShortConvProducerFold(
                residual_assignment=residual_assignment,
                in_proj_weight=producer_compiled["in_proj.weight"],
                in_proj_bias=(
                    None if not target_bias else producer_compiled["in_proj.bias"]
                ),
                conv_weight=producer_compiled["conv.weight"],
                conv_bias=(None if not target_bias else producer_compiled["conv.bias"]),
                channel_groups=channel_groups,
                channel_barycentric_weights=channel_weights,
                source_hidden_size=source_channel_width,
                target_hidden_size=target_channel_width,
                fir_length=fir_length,
                source_logical_parameters=source_logical,
                target_logical_parameters=target_logical,
                source_state_sha256_before=source_before,
                source_state_sha256_after=source_after,
                target_template_sha256=_template_sha256(producer_template_spec),
                source_derived_prior_sha256=_state_sha256(producer_prior_values),
                compiled_state_sha256=_state_sha256(producer_compiled),
                prior_correction_report=in_fit,
                source_residual_width=(
                    source_residual_width if decoupled_geometry else None
                ),
                target_residual_width=(
                    target_residual_width if decoupled_geometry else None
                ),
                external_residual_assignment_sha256=(
                    external_residual_assignment.assignment_sha256
                    if decoupled_geometry
                    else None
                ),
            )
        child_contexts = _run_compiled_short_conv_contexts(
            normalized,
            in_proj_weight=in_proj_weight,
            in_proj_bias=in_proj_bias,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
        )
        child_context_rows = torch.cat(
            [value.reshape(-1, target_channel_width) for value in child_contexts],
            dim=0,
        )
        endpoint_target = torch.cat(
            [
                (value["parent_endpoint"] - value["target_residual"]).reshape(
                    -1, target_residual_width
                )
                for value in normalized
            ],
            dim=0,
        )
        out_proj_weight, out_proj_bias, out_fit = _fit_affine(
            child_context_rows,
            endpoint_target,
            role="short_conv_out_proj",
            ridge=ridge,
            weight_prior=actual_out_prior,
            bias_prior=actual_out_bias_prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
            fit_bias=target_bias,
        )

        fp64_state = {
            "in_proj.weight": in_proj_weight,
            "conv.weight": conv_weight,
            "out_proj.weight": out_proj_weight,
        }
        if target_bias:
            fp64_state.update(
                {
                    "in_proj.bias": in_proj_bias,
                    "conv.bias": conv_bias,
                    "out_proj.bias": out_proj_bias,
                }
            )
        compiled = {
            name: fp64_state[name]
            .to(device="cpu", dtype=template_dtypes[name])
            .contiguous()
            .clone()
            for name in target_templates
        }
        materialized_contexts = _run_compiled_short_conv_contexts(
            normalized,
            in_proj_weight=compiled["in_proj.weight"].to(torch.float64),
            in_proj_bias=(
                None if not target_bias else compiled["in_proj.bias"].to(torch.float64)
            ),
            conv_weight=compiled["conv.weight"].to(torch.float64),
            conv_bias=(
                None if not target_bias else compiled["conv.bias"].to(torch.float64)
            ),
        )
        materialized_context_rows = torch.cat(
            [
                value.reshape(-1, target_channel_width)
                for value in materialized_contexts
            ],
            dim=0,
        )
        endpoint_prediction = (
            materialized_context_rows @ compiled["out_proj.weight"].to(torch.float64).T
        )
        if target_bias:
            endpoint_prediction = endpoint_prediction + compiled["out_proj.bias"].to(
                torch.float64
            )
        endpoint_error = _relative_rmse(endpoint_prediction, endpoint_target)

    source_after = _state_sha256(source_values)
    source_logical = sum(int(value.numel()) for value in source_values.values())
    target_logical = sum(math.prod(shape) for shape in template_shapes.values())
    if (
        source_before != source_after
        or not residual_assignment.full_source_coverage
        or not (
            sorted(source for group in channel_groups for source in group)
            == list(range(source_channel_width))
        )
        or target_logical >= source_logical
        or any(
            tuple(compiled[name].shape) != template_shapes[name]
            for name in target_templates
        )
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The compiled short-conv failed its structural invariants."
        )
    return Lfm2CompiledStructuralShortConvFold(
        residual_assignment=residual_assignment,
        in_proj_weight=compiled["in_proj.weight"],
        in_proj_bias=(None if not target_bias else compiled["in_proj.bias"]),
        conv_weight=compiled["conv.weight"],
        conv_bias=None if not target_bias else compiled["conv.bias"],
        out_proj_weight=compiled["out_proj.weight"],
        out_proj_bias=(None if not target_bias else compiled["out_proj.bias"]),
        channel_groups=channel_groups,
        channel_barycentric_weights=channel_weights,
        source_hidden_size=source_channel_width,
        target_hidden_size=target_channel_width,
        fir_length=fir_length,
        source_logical_parameters=source_logical,
        target_logical_parameters=target_logical,
        source_state_sha256_before=source_before,
        source_state_sha256_after=source_after,
        target_template_sha256=_template_sha256(template_spec),
        source_derived_prior_sha256=prior_sha,
        compiled_state_sha256=_state_sha256(compiled),
        residual_endpoint_relative_rmse=endpoint_error,
        prior_correction_reports=(in_fit, out_fit),
        source_residual_width=(source_residual_width if decoupled_geometry else None),
        target_residual_width=(target_residual_width if decoupled_geometry else None),
        external_residual_assignment_sha256=(
            external_residual_assignment.assignment_sha256
            if decoupled_geometry
            else None
        ),
    )


def compile_lfm2_structural_residual_attention_fold(
    *,
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_output_weight: Any,
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    target_value_weight_template: Any,
    target_output_weight_template: Any,
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    head_dim: int,
    query_norm_weight: Any | None = None,
    key_norm_weight: Any | None = None,
    query_weight_prior: Any | None = None,
    key_weight_prior: Any | None = None,
    value_weight_prior: Any | None = None,
    output_weight_prior: Any | None = None,
    residual_assignment_sha256: str | None = None,
    norm_epsilon: float = 1e-6,
    ridge: float = 0.0,
    max_prior_delta_ratio: float | None = 1.0,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    _producer_only: bool = False,
) -> Lfm2CompiledStructuralAttentionFold | Lfm2CompiledStructuralAttentionProducerFold:
    """Compile dense target-shaped Q/K/V/O tensors from sealed cached evidence."""

    torch = _torch()
    if not isinstance(_producer_only, bool):
        raise Lfm2StructuralResidualAttentionFoldError(
            "_producer_only must be boolean."
        )
    head_dim = _positive_int(head_dim, "head_dim")
    if head_dim % 2:
        raise Lfm2StructuralResidualAttentionFoldError(
            "LFM2 RoPE head_dim must be even."
        )
    ridge = _nonnegative_float(ridge, "ridge")
    cluster_ridge = _nonnegative_float(cluster_ridge, "cluster_ridge")
    norm_epsilon = _nonnegative_float(norm_epsilon, "norm_epsilon")
    if max_prior_delta_ratio is not None:
        max_prior_delta_ratio = _nonnegative_float(
            max_prior_delta_ratio, "max_prior_delta_ratio"
        )
    if norm_epsilon == 0.0:
        raise Lfm2StructuralResidualAttentionFoldError("norm_epsilon must be positive.")
    if not cases or not all(
        isinstance(value, Lfm2CachedAttentionFoldCase) for value in cases
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "cases must contain at least one Lfm2CachedAttentionFoldCase."
        )
    query_row_weighting_mode = _row_weighting_mode(
        tuple(case.query_row_sqrt_weights is not None for case in cases),
        "query row weighting",
    )
    key_value_row_weighting_mode = _row_weighting_mode(
        tuple(case.key_value_row_sqrt_weights is not None for case in cases),
        "key/value row weighting",
    )

    source_values = {
        "q_proj.weight": source_query_weight,
        "k_proj.weight": source_key_weight,
        "v_proj.weight": source_value_weight,
        "out_proj.weight": source_output_weight,
    }
    source_before = _state_sha256(source_values)
    sq = _float_tensor(source_query_weight, "source_query_weight", ndim=2)
    sk = _float_tensor(source_key_weight, "source_key_weight", ndim=2)
    sv = _float_tensor(source_value_weight, "source_value_weight", ndim=2)
    so = _float_tensor(source_output_weight, "source_output_weight", ndim=2)
    source_hidden = int(sq.shape[1])
    if int(sk.shape[1]) != source_hidden or int(sv.shape[1]) != source_hidden:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Source Q/K/V input widths differ."
        )
    if int(sq.shape[0]) % head_dim or int(sk.shape[0]) % head_dim:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Source Q/K rows do not divide by head_dim."
        )
    source_query_heads = int(sq.shape[0]) // head_dim
    source_key_value_heads = int(sk.shape[0]) // head_dim
    if (
        tuple(sv.shape) != tuple(sk.shape)
        or tuple(so.shape) != (source_hidden, source_query_heads * head_dim)
        or source_query_heads % source_key_value_heads
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Source Q/K/V/O tensors do not form a valid GQA topology."
        )

    target_templates = {
        "q_proj.weight": target_query_weight_template,
        "k_proj.weight": target_key_weight_template,
        "v_proj.weight": target_value_weight_template,
        "out_proj.weight": target_output_weight_template,
    }
    template_dtypes: dict[str, Any] = {}
    template_shapes: dict[str, tuple[int, ...]] = {}
    template_spec: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, value in target_templates.items():
        dtype, shape = _dense_template(value, f"target_{name}")
        template_dtypes[name] = dtype
        template_shapes[name] = shape
        template_spec[name] = (shape, str(dtype))
    target_hidden = template_shapes["q_proj.weight"][1]
    target_output = template_shapes["out_proj.weight"][0]
    if (
        template_shapes["q_proj.weight"][0] % head_dim
        or template_shapes["k_proj.weight"][0] % head_dim
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Target Q/K rows do not divide by preserved head_dim."
        )
    target_query_heads = template_shapes["q_proj.weight"][0] // head_dim
    target_key_value_heads = template_shapes["k_proj.weight"][0] // head_dim
    expected_shapes = {
        "q_proj.weight": (target_query_heads * head_dim, target_hidden),
        "k_proj.weight": (target_key_value_heads * head_dim, target_hidden),
        "v_proj.weight": (target_key_value_heads * head_dim, target_hidden),
        "out_proj.weight": (target_output, target_query_heads * head_dim),
    }
    if template_shapes != expected_shapes:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"Target templates are not a standard atomic GQA Q/K/V/O set: "
            f"{template_shapes} != {expected_shapes}."
        )
    if (
        target_hidden > source_hidden
        or target_output > source_hidden
        or target_query_heads > source_query_heads
        or target_key_value_heads > source_key_value_heads
        or target_query_heads % target_key_value_heads
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target cannot expand residual width or GQA heads "
            "and must remain valid GQA."
        )
    if (
        target_hidden == source_hidden
        and target_output == source_hidden
        and target_query_heads == source_query_heads
        and target_key_value_heads == source_key_value_heads
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "The structural target must reduce residual width or GQA heads."
        )
    if residual_assignment_sha256 is not None and (
        not isinstance(residual_assignment_sha256, str)
        or len(residual_assignment_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in residual_assignment_sha256
        )
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "residual_assignment_sha256 must be a lowercase SHA-256 digest."
        )
    if (
        source_hidden != target_hidden or source_hidden != target_output
    ) and residual_assignment_sha256 is None:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Cross-hidden structural compilation must be bound to the one global "
            "full-source sparse residual assignment."
        )
    supplied_priors = {
        "q_proj.weight": query_weight_prior,
        "k_proj.weight": key_weight_prior,
        "v_proj.weight": value_weight_prior,
        "out_proj.weight": output_weight_prior,
    }
    normalized_priors: dict[str, Any | None] = {}
    for name, prior in supplied_priors.items():
        if prior is None:
            normalized_priors[name] = None
            continue
        normalized_prior = _float_tensor(prior, f"source_derived_{name}_prior", ndim=2)
        if tuple(normalized_prior.shape) != template_shapes[name]:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"source-derived {name} prior must have target shape "
                f"{template_shapes[name]}."
            )
        normalized_priors[name] = normalized_prior
    prior_sha = (
        None
        if all(value is None for value in normalized_priors.values())
        else _state_sha256(
            {
                name: value
                for name, value in normalized_priors.items()
                if value is not None
            }
        )
    )

    topology = Lfm2StructuralAttentionTopology(
        source_query_heads=source_query_heads,
        source_key_value_heads=source_key_value_heads,
        target_query_heads=target_query_heads,
        target_key_value_heads=target_key_value_heads,
        head_dim=head_dim,
        source_hidden_size=source_hidden,
        target_hidden_size=target_hidden,
        target_output_size=target_output,
    )
    q_norm = (
        None
        if query_norm_weight is None
        else _float_tensor(query_norm_weight, "query_norm_weight", ndim=1)
    )
    k_norm = (
        None
        if key_norm_weight is None
        else _float_tensor(key_norm_weight, "key_norm_weight", ndim=1)
    )
    if q_norm is not None and tuple(q_norm.shape) != (head_dim,):
        raise Lfm2StructuralResidualAttentionFoldError(
            "query_norm_weight must have shape [head_dim]."
        )
    if k_norm is not None and tuple(k_norm.shape) != (head_dim,):
        raise Lfm2StructuralResidualAttentionFoldError(
            "key_norm_weight must have shape [head_dim]."
        )

    normalized: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        prefix = f"cases[{case_index}]"
        rope_evidence_dtype = _rope_evidence_dtype(
            case.parent_query_linear,
            case.parent_query,
            case.parent_key_linear,
            case.parent_key,
            name=prefix,
        )
        target_q_input = _float_tensor(
            case.target_query_input, f"{prefix}.target_query_input", ndim=3
        )
        target_kv_input = _float_tensor(
            case.target_key_value_input,
            f"{prefix}.target_key_value_input",
            ndim=3,
        )
        parent_q_linear = _float_tensor(
            case.parent_query_linear, f"{prefix}.parent_query_linear", ndim=4
        )
        parent_k_linear = _float_tensor(
            case.parent_key_linear, f"{prefix}.parent_key_linear", ndim=4
        )
        parent_v = _float_tensor(case.parent_value, f"{prefix}.parent_value", ndim=4)
        parent_q = _float_tensor(case.parent_query, f"{prefix}.parent_query", ndim=4)
        parent_k = _float_tensor(case.parent_key, f"{prefix}.parent_key", ndim=4)
        parent_context = _float_tensor(
            case.parent_context, f"{prefix}.parent_context", ndim=4
        )
        parent_output = _float_tensor(
            case.parent_post_attention_residual_target_space,
            f"{prefix}.parent_post_attention_residual_target_space",
            ndim=3,
        )
        target_residual = _float_tensor(
            case.target_pre_attention_residual,
            f"{prefix}.target_pre_attention_residual",
            ndim=3,
        )
        batch, query_length, observed_target_hidden = map(int, target_q_input.shape)
        key_length = int(target_kv_input.shape[1])
        if (
            int(target_kv_input.shape[0]) != batch
            or int(target_kv_input.shape[2]) != target_hidden
            or observed_target_hidden != target_hidden
            or tuple(parent_q_linear.shape)
            != (batch, source_query_heads, query_length, head_dim)
            or tuple(parent_q.shape) != tuple(parent_q_linear.shape)
            or tuple(parent_context.shape) != tuple(parent_q_linear.shape)
            or tuple(parent_k_linear.shape)
            != (batch, source_key_value_heads, key_length, head_dim)
            or tuple(parent_k.shape) != tuple(parent_k_linear.shape)
            or tuple(parent_v.shape) != tuple(parent_k_linear.shape)
            or tuple(parent_output.shape) != (batch, query_length, target_output)
            or tuple(target_residual.shape) != tuple(parent_output.shape)
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{prefix} tensor shapes do not match source/target topology."
            )
        mask = (
            None
            if case.key_mask is None
            else _boolean_tensor(
                case.key_mask,
                f"{prefix}.key_mask",
                shape=(batch, query_length, key_length),
            )
        )
        query_sqrt_weights = _sqrt_row_weights(
            case.query_row_sqrt_weights,
            f"{prefix}.query_row_sqrt_weights",
            shape=(batch, query_length),
        )
        key_value_sqrt_weights = _sqrt_row_weights(
            case.key_value_row_sqrt_weights,
            f"{prefix}.key_value_row_sqrt_weights",
            shape=(batch, key_length),
        )
        rope_values: dict[str, Any | None] = {}
        supplied_rope = (
            case.query_cosine,
            case.query_sine,
            case.key_cosine,
            case.key_sine,
        )
        if any(value is not None for value in supplied_rope) and not all(
            value is not None for value in supplied_rope
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{prefix} must supply all four rotary cosine/sine tensors together."
            )
        for name, value, length in (
            ("query_cosine", case.query_cosine, query_length),
            ("query_sine", case.query_sine, query_length),
            ("key_cosine", case.key_cosine, key_length),
            ("key_sine", case.key_sine, key_length),
        ):
            rope_values[name] = (
                None
                if value is None
                else _float_tensor(value, f"{prefix}.{name}", ndim=3)
            )
            if rope_values[name] is not None and tuple(rope_values[name].shape) != (
                batch,
                length,
                head_dim,
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix}.{name} has the wrong rotary shape."
                )
        normalized_query = _cached_lfm2_rms_norm(
            case.parent_query_linear,
            query_norm_weight,
            norm_epsilon,
        )
        normalized_key = _cached_lfm2_rms_norm(
            case.parent_key_linear,
            key_norm_weight,
            norm_epsilon,
        )
        if all(value is None for value in supplied_rope):
            if q_norm is None and torch.allclose(
                normalized_query, parent_q, rtol=1e-7, atol=1e-9
            ):
                query_cosine = query_sine = None
            else:
                query_cosine, query_sine = _infer_rope_from_cached_rows(
                    normalized_query,
                    parent_q,
                    f"{prefix}.query",
                    evidence_dtype=rope_evidence_dtype,
                )
            if k_norm is None and torch.allclose(
                normalized_key, parent_k, rtol=1e-7, atol=1e-9
            ):
                key_cosine = key_sine = None
            else:
                key_cosine, key_sine = _infer_rope_from_cached_rows(
                    normalized_key,
                    parent_k,
                    f"{prefix}.key",
                    evidence_dtype=rope_evidence_dtype,
                )
            rope_values = {
                "query_cosine": query_cosine,
                "query_sine": query_sine,
                "key_cosine": key_cosine,
                "key_sine": key_sine,
            }
        else:
            _validate_rope_reconstruction(
                normalized_query,
                parent_q,
                rope_values["query_cosine"],
                rope_values["query_sine"],
                f"{prefix}.query",
                evidence_dtype=rope_evidence_dtype,
            )
            _validate_rope_reconstruction(
                normalized_key,
                parent_k,
                rope_values["key_cosine"],
                rope_values["key_sine"],
                f"{prefix}.key",
                evidence_dtype=rope_evidence_dtype,
            )
        _validate_shared_qk_rope(
            rope_values["query_cosine"],
            rope_values["query_sine"],
            rope_values["key_cosine"],
            rope_values["key_sine"],
            prefix,
            evidence_dtype=rope_evidence_dtype,
        )
        contribution = None
        if case.parent_contribution is not None:
            contribution = _float_tensor(
                case.parent_contribution,
                f"{prefix}.parent_contribution",
                ndim=4,
            )
            if tuple(contribution.shape[:3]) != (
                batch,
                source_query_heads,
                query_length,
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{prefix}.parent_contribution has incompatible leading axes."
                )
        normalized.append(
            {
                "target_q_input": target_q_input,
                "target_kv_input": target_kv_input,
                "q_linear": parent_q_linear,
                "k_linear": parent_k_linear,
                "value": parent_v,
                "query": parent_q,
                "key": parent_k,
                "context": parent_context,
                "parent_output": parent_output,
                "target_residual": target_residual,
                "mask": mask,
                "contribution": contribution,
                "query_sqrt_weights": query_sqrt_weights,
                "key_value_sqrt_weights": key_value_sqrt_weights,
                **rope_values,
            }
        )
    query_row_sqrt_weight_bundle_sha256 = _state_sha256(
        {
            f"case.{index:04d}": value["query_sqrt_weights"]
            for index, value in enumerate(normalized)
        }
    )
    key_value_row_sqrt_weight_bundle_sha256 = _state_sha256(
        {
            f"case.{index:04d}": value["key_value_sqrt_weights"]
            for index, value in enumerate(normalized)
        }
    )

    with torch.no_grad():
        query_signatures = torch.cat(
            [
                (value["query"] * value["query_sqrt_weights"][:, None, :, None])
                .permute(1, 0, 2, 3)
                .reshape(source_query_heads, -1)
                for value in normalized
            ],
            dim=1,
        )
        key_signatures = torch.cat(
            [
                (value["key"] * value["key_value_sqrt_weights"][:, None, :, None])
                .permute(1, 0, 2, 3)
                .reshape(source_key_value_heads, -1)
                for value in normalized
            ],
            dim=1,
        )
        value_signatures = torch.cat(
            [
                (value["value"] * value["key_value_sqrt_weights"][:, None, :, None])
                .permute(1, 0, 2, 3)
                .reshape(source_key_value_heads, -1)
                for value in normalized
            ],
            dim=1,
        )
        context_signatures = torch.cat(
            [
                (value["context"] * value["query_sqrt_weights"][:, None, :, None])
                .permute(1, 0, 2, 3)
                .reshape(source_query_heads, -1)
                for value in normalized
            ],
            dim=1,
        )
        contribution_signatures = torch.cat(
            [
                (
                    (
                        value["context"].square().sum(dim=-1).sqrt()
                        * value["query_sqrt_weights"][:, None, :]
                    )
                    .permute(1, 0, 2)
                    .reshape(source_query_heads, -1)
                    if value["contribution"] is None
                    else (
                        value["contribution"]
                        * value["query_sqrt_weights"][:, None, :, None]
                    )
                    .permute(1, 0, 2, 3)
                    .reshape(source_query_heads, -1)
                )
                for value in normalized
            ],
            dim=1,
        )
        cluster = build_gqa_activation_cluster_plan(
            query_signatures,
            key_signatures,
            value_signatures,
            context_signatures,
            contribution_signatures,
            source_query_heads=source_query_heads,
            source_key_value_heads=source_key_value_heads,
            target_query_heads=target_query_heads,
            target_key_value_heads=target_key_value_heads,
            source_head_dim=head_dim,
            target_head_dim=head_dim,
            ridge=cluster_ridge,
            max_iterations=_positive_int(
                cluster_max_iterations, "cluster_max_iterations"
            ),
            distance_chunk_size=_positive_int(
                cluster_distance_chunk_size, "cluster_distance_chunk_size"
            ),
        )
        if not cluster.report.full_source_coverage:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Activation clustering did not prove full source coverage."
            )

        source_q_importance = (
            query_signatures.square().mean(dim=1)
            + context_signatures.square().mean(dim=1)
            + contribution_signatures.square().mean(dim=1)
        )
        source_kv_importance = key_signatures.square().mean(
            dim=1
        ) + value_signatures.square().mean(dim=1)

        case_targets: list[dict[str, Any]] = []
        query_groups: tuple[tuple[int, ...], ...] | None = None
        query_weights: tuple[tuple[float, ...], ...] | None = None
        kv_groups: tuple[tuple[int, ...], ...] | None = None
        kv_weights: tuple[tuple[float, ...], ...] | None = None
        for value in normalized:
            repeated_key = value["key"].repeat_interleave(
                topology.source_query_heads_per_group, dim=1
            )
            parent_scores = torch.matmul(
                value["query"], repeated_key.transpose(-2, -1)
            ) / math.sqrt(head_dim)
            centered_parent_scores = _center_scores(parent_scores, value["mask"])
            target_scores, q_groups, q_weights = _weighted_group_rows(
                centered_parent_scores,
                cluster.query_head_assignment,
                target_query_heads,
                source_q_importance,
            )
            target_q_linear, q_groups_2, q_weights_2 = _weighted_group_rows(
                value["q_linear"],
                cluster.query_head_assignment,
                target_query_heads,
                source_q_importance,
            )
            target_k_linear, kv_groups_2, kv_weights_2 = _weighted_group_rows(
                value["k_linear"],
                cluster.group_assignment,
                target_key_value_heads,
                source_kv_importance,
            )
            target_key_anchor, _kv_groups_3, _kv_weights_3 = _weighted_group_rows(
                value["key"],
                cluster.group_assignment,
                target_key_value_heads,
                source_kv_importance,
            )
            target_value, _kv_groups_4, _kv_weights_4 = _weighted_group_rows(
                value["value"],
                cluster.group_assignment,
                target_key_value_heads,
                source_kv_importance,
            )
            factor_q, factor_k = _factor_score_barycenter(
                target_scores,
                target_key_anchor,
                target_query_heads=target_query_heads,
                target_key_value_heads=target_key_value_heads,
                head_dim=head_dim,
                query_sqrt_weights=value["query_sqrt_weights"],
            )
            q_linear_target = _projection_target_from_runtime(
                factor_q,
                target_q_linear,
                norm_weight=q_norm,
                epsilon=norm_epsilon,
                cosine=value["query_cosine"],
                sine=value["query_sine"],
            )
            k_linear_target = _projection_target_from_runtime(
                factor_k,
                target_k_linear,
                norm_weight=k_norm,
                epsilon=norm_epsilon,
                cosine=value["key_cosine"],
                sine=value["key_sine"],
            )
            if query_groups is None:
                query_groups, query_weights = q_groups, q_weights
                kv_groups, kv_weights = kv_groups_2, kv_weights_2
            elif (
                query_groups != q_groups
                or query_weights != q_weights
                or kv_groups != kv_groups_2
                or kv_weights != kv_weights_2
                or q_groups_2 != q_groups
                or q_weights_2 != q_weights
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    "Full-coverage barycentric assignments changed between cases."
                )
            case_targets.append(
                {
                    "target_scores": target_scores,
                    "q_linear": q_linear_target,
                    "k_linear": k_linear_target,
                    "value": target_value,
                }
            )

        query_fit_sqrt = torch.cat(
            [value["query_sqrt_weights"].reshape(-1, 1) for value in normalized],
            dim=0,
        )
        key_value_fit_sqrt = torch.cat(
            [value["key_value_sqrt_weights"].reshape(-1, 1) for value in normalized],
            dim=0,
        )
        query_input_rows = torch.cat(
            [
                value["target_q_input"].reshape(-1, target_hidden)
                for value in normalized
            ],
            dim=0,
        )
        key_value_input_rows = torch.cat(
            [
                value["target_kv_input"].reshape(-1, target_hidden)
                for value in normalized
            ],
            dim=0,
        )
        query_target_rows = torch.cat(
            [
                value["q_linear"]
                .permute(0, 2, 1, 3)
                .reshape(-1, target_query_heads * head_dim)
                for value in case_targets
            ],
            dim=0,
        )
        key_target_rows = torch.cat(
            [
                value["k_linear"]
                .permute(0, 2, 1, 3)
                .reshape(-1, target_key_value_heads * head_dim)
                for value in case_targets
            ],
            dim=0,
        )
        value_target_rows = torch.cat(
            [
                value["value"]
                .permute(0, 2, 1, 3)
                .reshape(-1, target_key_value_heads * head_dim)
                for value in case_targets
            ],
            dim=0,
        )
        query_weight, query_fit_report = _fit_linear(
            query_input_rows * query_fit_sqrt,
            query_target_rows * query_fit_sqrt,
            role="query",
            ridge=ridge,
            prior=normalized_priors["q_proj.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        key_weight, key_fit_report = _fit_linear(
            key_value_input_rows * key_value_fit_sqrt,
            key_target_rows * key_value_fit_sqrt,
            role="key",
            ridge=ridge,
            prior=normalized_priors["k_proj.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        value_weight, value_fit_report = _fit_linear(
            key_value_input_rows * key_value_fit_sqrt,
            value_target_rows * key_value_fit_sqrt,
            role="value",
            ridge=ridge,
            prior=normalized_priors["v_proj.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        if _producer_only:
            producer_fp64 = {
                "q_proj.weight": query_weight,
                "k_proj.weight": key_weight,
                "v_proj.weight": value_weight,
            }
            producer_compiled = {
                name: value.to(device="cpu", dtype=template_dtypes[name])
                .contiguous()
                .clone()
                for name, value in producer_fp64.items()
            }
            score_predictions = []
            score_targets = []
            materialized_q = producer_compiled["q_proj.weight"].to(torch.float64)
            materialized_k = producer_compiled["k_proj.weight"].to(torch.float64)
            for value, targets in zip(normalized, case_targets):
                batch, query_length, _hidden = map(int, value["target_q_input"].shape)
                key_length = int(value["target_kv_input"].shape[1])
                q_linear_child = (
                    (value["target_q_input"] @ materialized_q.T)
                    .reshape(
                        batch,
                        query_length,
                        target_query_heads,
                        head_dim,
                    )
                    .permute(0, 2, 1, 3)
                )
                k_linear_child = (
                    (value["target_kv_input"] @ materialized_k.T)
                    .reshape(
                        batch,
                        key_length,
                        target_key_value_heads,
                        head_dim,
                    )
                    .permute(0, 2, 1, 3)
                )
                q_child = _apply_rope(
                    _rms_norm(q_linear_child, q_norm, norm_epsilon),
                    value["query_cosine"],
                    value["query_sine"],
                )
                k_child = _apply_rope(
                    _rms_norm(k_linear_child, k_norm, norm_epsilon),
                    value["key_cosine"],
                    value["key_sine"],
                )
                repeated_k = k_child.repeat_interleave(
                    topology.target_query_heads_per_group, dim=1
                )
                scores = torch.matmul(
                    q_child, repeated_k.transpose(-2, -1)
                ) / math.sqrt(head_dim)
                score_predictions.append(
                    _center_scores(scores, value["mask"])
                    * value["query_sqrt_weights"][:, None, :, None]
                )
                score_targets.append(
                    targets["target_scores"]
                    * value["query_sqrt_weights"][:, None, :, None]
                )
            source_after = _state_sha256(source_values)
            producer_source_values = {
                "q_proj.weight": source_query_weight,
                "k_proj.weight": source_key_weight,
                "v_proj.weight": source_value_weight,
            }
            source_logical = sum(
                int(value.numel()) for value in producer_source_values.values()
            )
            target_logical = sum(
                int(value.numel()) for value in producer_compiled.values()
            )
            assert query_groups is not None
            assert query_weights is not None
            assert kv_groups is not None
            assert kv_weights is not None
            if (
                source_before != source_after
                or sorted(source for group in query_groups for source in group)
                != list(range(source_query_heads))
                or sorted(source for group in kv_groups for source in group)
                != list(range(source_key_value_heads))
                or target_logical >= source_logical
                or any(
                    int(target_templates[name].numel())
                    >= int(producer_source_values[name].numel())
                    for name in producer_compiled
                )
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    "The split attention producer failed its invariants."
                )
            producer_priors = {
                name: normalized_priors[name]
                for name in producer_compiled
                if normalized_priors[name] is not None
            }
            producer_template_spec = {
                name: template_spec[name] for name in producer_compiled
            }
            return Lfm2CompiledStructuralAttentionProducerFold(
                topology=topology,
                cluster_plan=cluster,
                query_weight=producer_compiled["q_proj.weight"],
                key_weight=producer_compiled["k_proj.weight"],
                value_weight=producer_compiled["v_proj.weight"],
                query_groups=query_groups,
                query_barycentric_weights=query_weights,
                key_value_groups=kv_groups,
                key_value_barycentric_weights=kv_weights,
                source_logical_parameters=source_logical,
                target_logical_parameters=target_logical,
                source_state_sha256_before=source_before,
                source_state_sha256_after=source_after,
                target_template_sha256=_template_sha256(producer_template_spec),
                residual_assignment_sha256=(residual_assignment_sha256),
                source_derived_prior_sha256=(
                    None if not producer_priors else _state_sha256(producer_priors)
                ),
                compiled_state_sha256=_state_sha256(producer_compiled),
                centered_score_relative_rmse=_relative_rmse(
                    torch.cat([value.reshape(-1) for value in score_predictions]),
                    torch.cat([value.reshape(-1) for value in score_targets]),
                ),
                query_row_weighting_mode=query_row_weighting_mode,
                key_value_row_weighting_mode=key_value_row_weighting_mode,
                query_row_sqrt_weight_bundle_sha256=(
                    query_row_sqrt_weight_bundle_sha256
                ),
                key_value_row_sqrt_weight_bundle_sha256=(
                    key_value_row_sqrt_weight_bundle_sha256
                ),
                prior_correction_reports=(
                    query_fit_report,
                    key_fit_report,
                    value_fit_report,
                ),
            )

        child_context_rows = []
        endpoint_rows = []
        score_predictions = []
        score_targets = []
        for value, targets in zip(normalized, case_targets):
            batch, query_length, _hidden = map(int, value["target_q_input"].shape)
            key_length = int(value["target_kv_input"].shape[1])
            q_linear_child = (
                (value["target_q_input"] @ query_weight.T)
                .reshape(batch, query_length, target_query_heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            k_linear_child = (
                (value["target_kv_input"] @ key_weight.T)
                .reshape(batch, key_length, target_key_value_heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            value_child = (
                (value["target_kv_input"] @ value_weight.T)
                .reshape(batch, key_length, target_key_value_heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            q_child = _apply_rope(
                _rms_norm(q_linear_child, q_norm, norm_epsilon),
                value["query_cosine"],
                value["query_sine"],
            )
            k_child = _apply_rope(
                _rms_norm(k_linear_child, k_norm, norm_epsilon),
                value["key_cosine"],
                value["key_sine"],
            )
            repeated_k = k_child.repeat_interleave(
                topology.target_query_heads_per_group, dim=1
            )
            scores = torch.matmul(q_child, repeated_k.transpose(-2, -1)) / math.sqrt(
                head_dim
            )
            score_predictions.append(
                _center_scores(scores, value["mask"])
                * value["query_sqrt_weights"][:, None, :, None]
            )
            score_targets.append(
                targets["target_scores"] * value["query_sqrt_weights"][:, None, :, None]
            )
            if value["mask"] is not None:
                scores = scores.masked_fill(~value["mask"].unsqueeze(1), -1e30)
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float64)
            repeated_v = value_child.repeat_interleave(
                topology.target_query_heads_per_group, dim=1
            )
            context = torch.matmul(probabilities, repeated_v)
            child_context_rows.append(
                context.permute(0, 2, 1, 3).reshape(
                    batch * query_length, target_query_heads * head_dim
                )
                * value["query_sqrt_weights"].reshape(-1, 1)
            )
            endpoint_rows.append(
                (value["parent_output"] - value["target_residual"]).reshape(
                    batch * query_length, target_output
                )
                * value["query_sqrt_weights"].reshape(-1, 1)
            )
        child_context_design = torch.cat(child_context_rows, dim=0)
        endpoint_target = torch.cat(endpoint_rows, dim=0)
        output_weight, output_fit_report = _fit_linear(
            child_context_design,
            endpoint_target,
            role="output",
            ridge=ridge,
            prior=normalized_priors["out_proj.weight"],
            max_prior_delta_ratio=max_prior_delta_ratio,
        )

        fp64_state = {
            "q_proj.weight": query_weight,
            "k_proj.weight": key_weight,
            "v_proj.weight": value_weight,
            "out_proj.weight": output_weight,
        }
        compiled = {
            name: fp64_state[name]
            .to(device="cpu", dtype=template_dtypes[name])
            .contiguous()
            .clone()
            for name in _STATE_KEYS
        }
        materialized_output = compiled["out_proj.weight"].to(torch.float64)
        endpoint_prediction = child_context_design @ materialized_output.T
        endpoint_error = _relative_rmse(endpoint_prediction, endpoint_target)
        score_error = _relative_rmse(
            torch.cat([value.reshape(-1) for value in score_predictions]),
            torch.cat([value.reshape(-1) for value in score_targets]),
        )

    source_after = _state_sha256(source_values)
    compiled_sha = _state_sha256(compiled)
    source_logical = sum(int(value.numel()) for value in source_values.values())
    target_logical = sum(math.prod(shape) for shape in template_shapes.values())
    per_projection_smaller = all(
        int(target_templates[name].numel()) < int(source_values[name].numel())
        for name in _STATE_KEYS
    )
    audit = Lfm2StructuralAttentionFoldAudit(
        source_unchanged=source_before == source_after,
        blank_target_values_read=False,
        full_source_query_coverage=(
            sorted(source for members in (query_groups or ()) for source in members)
            == list(range(source_query_heads))
        ),
        full_source_key_value_coverage=(
            sorted(source for members in (kv_groups or ()) for source in members)
            == list(range(source_key_value_heads))
        ),
        source_query_heads_dropped=0,
        source_key_value_heads_dropped=0,
        target_shapes_exact=all(
            tuple(compiled[name].shape) == template_shapes[name] for name in _STATE_KEYS
        ),
        every_projection_structurally_smaller=per_projection_smaller,
    )
    if not audit.all_hold or target_logical >= source_logical:
        raise Lfm2StructuralResidualAttentionFoldError(
            "The compiled attention state failed its structural invariants."
        )
    assert query_groups is not None
    assert query_weights is not None
    assert kv_groups is not None
    assert kv_weights is not None
    return Lfm2CompiledStructuralAttentionFold(
        topology=topology,
        cluster_plan=cluster,
        query_weight=compiled["q_proj.weight"],
        key_weight=compiled["k_proj.weight"],
        value_weight=compiled["v_proj.weight"],
        output_weight=compiled["out_proj.weight"],
        query_groups=query_groups,
        query_barycentric_weights=query_weights,
        key_value_groups=kv_groups,
        key_value_barycentric_weights=kv_weights,
        source_logical_parameters=source_logical,
        target_logical_parameters=target_logical,
        source_state_sha256_before=source_before,
        source_state_sha256_after=source_after,
        target_template_sha256=_template_sha256(template_spec),
        residual_assignment_sha256=residual_assignment_sha256,
        source_derived_prior_sha256=prior_sha,
        compiled_state_sha256=compiled_sha,
        centered_score_relative_rmse=score_error,
        residual_endpoint_relative_rmse=endpoint_error,
        query_row_weighting_mode=query_row_weighting_mode,
        key_value_row_weighting_mode=key_value_row_weighting_mode,
        query_row_sqrt_weight_bundle_sha256=(query_row_sqrt_weight_bundle_sha256),
        key_value_row_sqrt_weight_bundle_sha256=(
            key_value_row_sqrt_weight_bundle_sha256
        ),
        prior_correction_reports=(
            query_fit_report,
            key_fit_report,
            value_fit_report,
            output_fit_report,
        ),
        audit=audit,
    )


def compile_lfm2_structural_attention_producer_fold(
    **compiler_inputs: Any,
) -> Lfm2CompiledStructuralAttentionProducerFold:
    """Compile only Q/K/V, leaving O for actual-child-context fitting.

    ``compiler_inputs`` are the same keyword arguments accepted by
    :func:`compile_lfm2_structural_residual_attention_fold`.  The split path
    stops before internally simulating attention context or fitting O.
    """

    if "_producer_only" in compiler_inputs:
        raise Lfm2StructuralResidualAttentionFoldError(
            "_producer_only is compiler-owned."
        )
    result = compile_lfm2_structural_residual_attention_fold(
        **compiler_inputs,
        _producer_only=True,
    )
    if not isinstance(result, Lfm2CompiledStructuralAttentionProducerFold):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Attention producer compiler returned the wrong artifact."
        )
    return result


def compile_lfm2_structural_short_conv_producer_fold(
    **compiler_inputs: Any,
) -> Lfm2CompiledStructuralShortConvProducerFold:
    """Compile only B/C/x input projection and FIR, leaving output separate."""

    if "_producer_only" in compiler_inputs:
        raise Lfm2StructuralResidualAttentionFoldError(
            "_producer_only is compiler-owned."
        )
    result = compile_lfm2_structural_residual_short_conv_fold(
        **compiler_inputs,
        _producer_only=True,
    )
    if not isinstance(result, Lfm2CompiledStructuralShortConvProducerFold):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Short-conv producer compiler returned the wrong artifact."
        )
    return result


def _actual_attention_context_rows(
    value: Any,
    *,
    case: Lfm2CachedAttentionFoldCase,
    context_width: int,
    name: str,
) -> Any:
    """Normalize [B,H,Q,D], [B,Q,C], or [B*Q,C] actual child context."""

    raw_ndim = getattr(value, "ndim", None)
    if raw_ndim not in {2, 3, 4}:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} must have rank 2, 3, or 4."
        )
    context = _float_tensor(value, name, ndim=int(raw_ndim))
    parent_endpoint = _float_tensor(
        case.parent_post_attention_residual_target_space,
        f"{name}.parent_endpoint",
        ndim=3,
    )
    batch, query_length, _output = map(int, parent_endpoint.shape)
    if context.ndim == 4:
        if (
            int(context.shape[0]) != batch
            or int(context.shape[2]) != query_length
            or int(context.shape[1]) * int(context.shape[3]) != context_width
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{name} rank-4 context must be [batch, heads, query, dim]."
            )
        return context.permute(0, 2, 1, 3).reshape(batch * query_length, context_width)
    if context.ndim == 3:
        if tuple(context.shape) != (
            batch,
            query_length,
            context_width,
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{name} rank-3 context must be [batch, query, width]."
            )
        return context.reshape(batch * query_length, context_width)
    if tuple(context.shape) != (
        batch * query_length,
        context_width,
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} row-major context has the wrong shape."
        )
    return context


def compile_lfm2_structural_attention_consumer_fold(
    *,
    target_output_weight_template: Any,
    actual_child_contexts: Sequence[Any],
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    output_weight_prior: Any | None = None,
    ridge: float = 0.0,
    max_prior_delta_ratio: float | None = 1.0,
) -> Lfm2CompiledStructuralAttentionConsumerFold:
    """Fit O only from caller-supplied contexts produced by installed Q/K/V."""

    torch = _torch()
    if (
        not cases
        or len(actual_child_contexts) != len(cases)
        or not all(isinstance(value, Lfm2CachedAttentionFoldCase) for value in cases)
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Actual attention contexts must align one-to-one with cases."
        )
    ridge = _nonnegative_float(ridge, "ridge")
    if max_prior_delta_ratio is not None:
        max_prior_delta_ratio = _nonnegative_float(
            max_prior_delta_ratio, "max_prior_delta_ratio"
        )
    template_dtype, template_shape = _dense_template(
        target_output_weight_template,
        "target_output_weight_template",
    )
    target_output, context_width = template_shape
    prior = None
    if output_weight_prior is not None:
        prior = _float_tensor(output_weight_prior, "output_weight_prior", ndim=2)
        if tuple(prior.shape) != template_shape:
            raise Lfm2StructuralResidualAttentionFoldError(
                "output_weight_prior does not match the target O template."
            )
    context_rows = []
    endpoint_rows = []
    for case_index, (actual_context, case) in enumerate(
        zip(actual_child_contexts, cases)
    ):
        context_rows.append(
            _actual_attention_context_rows(
                actual_context,
                case=case,
                context_width=context_width,
                name=f"actual_child_contexts[{case_index}]",
            )
        )
        parent_endpoint = _float_tensor(
            case.parent_post_attention_residual_target_space,
            f"cases[{case_index}].parent_endpoint",
            ndim=3,
        )
        target_residual = _float_tensor(
            case.target_pre_attention_residual,
            f"cases[{case_index}].target_residual",
            ndim=3,
        )
        if (
            tuple(parent_endpoint.shape) != tuple(target_residual.shape)
            or int(parent_endpoint.shape[2]) != target_output
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"cases[{case_index}] endpoint topology is incompatible."
            )
        endpoint_rows.append(
            (parent_endpoint - target_residual).reshape(-1, target_output)
        )
    design = torch.cat(context_rows, dim=0)
    endpoint = torch.cat(endpoint_rows, dim=0)
    weight, fit = _fit_linear(
        design,
        endpoint,
        role="attention_output_actual_child_context",
        ridge=ridge,
        prior=prior,
        max_prior_delta_ratio=max_prior_delta_ratio,
    )
    compiled_weight = weight.to(device="cpu", dtype=template_dtype).contiguous().clone()
    compiled_state = {"out_proj.weight": compiled_weight}
    materialized_prediction = design @ compiled_weight.to(torch.float64).T
    return Lfm2CompiledStructuralAttentionConsumerFold(
        output_weight=compiled_weight,
        context_width=context_width,
        target_output_size=target_output,
        case_count=len(cases),
        target_template_sha256=_template_sha256(
            {"out_proj.weight": (template_shape, str(template_dtype))}
        ),
        source_derived_prior_sha256=(
            None if prior is None else _state_sha256({"out_proj.weight": prior})
        ),
        actual_child_context_sha256=_state_sha256(
            {f"context.{index:04d}": value for index, value in enumerate(context_rows)}
        ),
        endpoint_target_sha256=_tensor_sha256(endpoint),
        compiled_state_sha256=_state_sha256(compiled_state),
        residual_endpoint_relative_rmse=_relative_rmse(
            materialized_prediction, endpoint
        ),
        prior_correction_report=fit,
    )


def _normalize_attention_value_output_case(
    case: Lfm2CachedAttentionFoldCase,
    *,
    case_index: int,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    query_norm_weight: Any | None,
    key_norm_weight: Any | None,
    normalized_query_norm_weight: Any | None,
    normalized_key_norm_weight: Any | None,
    norm_epsilon: float,
) -> dict[str, Any]:
    """Reconstruct the fixed child score path for one coupled V/O case."""

    torch = _torch()
    topology = producer.topology
    prefix = f"cases[{case_index}]"
    rope_evidence_dtype = _rope_evidence_dtype(
        case.parent_query_linear,
        case.parent_query,
        case.parent_key_linear,
        case.parent_key,
        name=prefix,
    )
    target_q_input = _float_tensor(
        case.target_query_input,
        f"{prefix}.target_query_input",
        ndim=3,
    )
    target_kv_input = _float_tensor(
        case.target_key_value_input,
        f"{prefix}.target_key_value_input",
        ndim=3,
    )
    parent_q_linear = _float_tensor(
        case.parent_query_linear,
        f"{prefix}.parent_query_linear",
        ndim=4,
    )
    parent_k_linear = _float_tensor(
        case.parent_key_linear,
        f"{prefix}.parent_key_linear",
        ndim=4,
    )
    parent_q = _float_tensor(
        case.parent_query,
        f"{prefix}.parent_query",
        ndim=4,
    )
    parent_k = _float_tensor(
        case.parent_key,
        f"{prefix}.parent_key",
        ndim=4,
    )
    parent_endpoint = _float_tensor(
        case.parent_post_attention_residual_target_space,
        f"{prefix}.parent_post_attention_residual_target_space",
        ndim=3,
    )
    target_residual = _float_tensor(
        case.target_pre_attention_residual,
        f"{prefix}.target_pre_attention_residual",
        ndim=3,
    )
    batch, query_length, target_hidden = map(int, target_q_input.shape)
    key_length = int(target_kv_input.shape[1])
    if (
        target_hidden != topology.target_hidden_size
        or int(target_kv_input.shape[0]) != batch
        or int(target_kv_input.shape[2]) != topology.target_hidden_size
        or tuple(parent_q_linear.shape)
        != (
            batch,
            topology.source_query_heads,
            query_length,
            topology.head_dim,
        )
        or tuple(parent_q.shape) != tuple(parent_q_linear.shape)
        or tuple(parent_k_linear.shape)
        != (
            batch,
            topology.source_key_value_heads,
            key_length,
            topology.head_dim,
        )
        or tuple(parent_k.shape) != tuple(parent_k_linear.shape)
        or tuple(parent_endpoint.shape)
        != (batch, query_length, topology.target_output_size)
        or tuple(target_residual.shape) != tuple(parent_endpoint.shape)
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{prefix} tensor shapes do not match the fixed Q/K topology."
        )
    mask = (
        None
        if case.key_mask is None
        else _boolean_tensor(
            case.key_mask,
            f"{prefix}.key_mask",
            shape=(batch, query_length, key_length),
        )
    )
    supplied_rope = (
        case.query_cosine,
        case.query_sine,
        case.key_cosine,
        case.key_sine,
    )
    if any(value is not None for value in supplied_rope) and not all(
        value is not None for value in supplied_rope
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{prefix} must supply all four rotary tensors together."
        )
    rope_values: dict[str, Any | None] = {}
    for name, value, length in (
        ("query_cosine", case.query_cosine, query_length),
        ("query_sine", case.query_sine, query_length),
        ("key_cosine", case.key_cosine, key_length),
        ("key_sine", case.key_sine, key_length),
    ):
        normalized = (
            None if value is None else _float_tensor(value, f"{prefix}.{name}", ndim=3)
        )
        if normalized is not None and tuple(normalized.shape) != (
            batch,
            length,
            topology.head_dim,
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{prefix}.{name} has the wrong rotary shape."
            )
        rope_values[name] = normalized
    normalized_parent_q = _cached_lfm2_rms_norm(
        case.parent_query_linear,
        query_norm_weight,
        norm_epsilon,
    )
    normalized_parent_k = _cached_lfm2_rms_norm(
        case.parent_key_linear,
        key_norm_weight,
        norm_epsilon,
    )
    if all(value is None for value in supplied_rope):
        if normalized_query_norm_weight is None and torch.allclose(
            normalized_parent_q,
            parent_q,
            rtol=1e-7,
            atol=1e-9,
        ):
            query_cosine = query_sine = None
        else:
            query_cosine, query_sine = _infer_rope_from_cached_rows(
                normalized_parent_q,
                parent_q,
                f"{prefix}.query",
                evidence_dtype=rope_evidence_dtype,
            )
        if normalized_key_norm_weight is None and torch.allclose(
            normalized_parent_k,
            parent_k,
            rtol=1e-7,
            atol=1e-9,
        ):
            key_cosine = key_sine = None
        else:
            key_cosine, key_sine = _infer_rope_from_cached_rows(
                normalized_parent_k,
                parent_k,
                f"{prefix}.key",
                evidence_dtype=rope_evidence_dtype,
            )
        rope_values = {
            "query_cosine": query_cosine,
            "query_sine": query_sine,
            "key_cosine": key_cosine,
            "key_sine": key_sine,
        }
    else:
        _validate_rope_reconstruction(
            normalized_parent_q,
            parent_q,
            rope_values["query_cosine"],
            rope_values["query_sine"],
            f"{prefix}.query",
            evidence_dtype=rope_evidence_dtype,
        )
        _validate_rope_reconstruction(
            normalized_parent_k,
            parent_k,
            rope_values["key_cosine"],
            rope_values["key_sine"],
            f"{prefix}.key",
            evidence_dtype=rope_evidence_dtype,
        )
    _validate_shared_qk_rope(
        rope_values["query_cosine"],
        rope_values["query_sine"],
        rope_values["key_cosine"],
        rope_values["key_sine"],
        prefix,
        evidence_dtype=rope_evidence_dtype,
    )

    query_weight = producer.query_weight.to(torch.float64)
    key_weight = producer.key_weight.to(torch.float64)
    q_linear = (
        (target_q_input @ query_weight.T)
        .reshape(
            batch,
            query_length,
            topology.target_query_heads,
            topology.head_dim,
        )
        .permute(0, 2, 1, 3)
    )
    k_linear = (
        (target_kv_input @ key_weight.T)
        .reshape(
            batch,
            key_length,
            topology.target_key_value_heads,
            topology.head_dim,
        )
        .permute(0, 2, 1, 3)
    )
    query = _apply_rope(
        _rms_norm(
            q_linear,
            normalized_query_norm_weight,
            norm_epsilon,
        ),
        rope_values["query_cosine"],
        rope_values["query_sine"],
    )
    key = _apply_rope(
        _rms_norm(
            k_linear,
            normalized_key_norm_weight,
            norm_epsilon,
        ),
        rope_values["key_cosine"],
        rope_values["key_sine"],
    )
    repeated_key = key.repeat_interleave(
        topology.target_query_heads_per_group,
        dim=1,
    )
    scores = torch.matmul(
        query,
        repeated_key.transpose(-2, -1),
    ) / math.sqrt(topology.head_dim)
    candidate_scores = scores
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e30)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float64)
    weighted_inputs = torch.matmul(
        probabilities,
        target_kv_input.unsqueeze(1),
    )
    return {
        "batch": batch,
        "query_length": query_length,
        "scores": candidate_scores,
        "key_mask": mask,
        "probabilities": probabilities,
        "weighted_inputs": weighted_inputs,
        "endpoint": (parent_endpoint - target_residual).reshape(
            batch * query_length, topology.target_output_size
        ),
    }


def evaluate_lfm2_compiled_attention_producer_score_preservation(
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    *,
    query_norm_weight: Any | None = None,
    key_norm_weight: Any | None = None,
    norm_epsilon: float = 1e-6,
    top_k: int = 8,
    relative_margin_floor: float = 0.5,
    thresholds: Lfm2AttentionScorePreservationThresholds | None = None,
) -> Lfm2AttentionScorePreservationReport:
    """Evaluate one compiled Q/K path against cached parent score decisions.

    The parent reference is the compiler-authenticated barycenter of every
    source query head assigned to each target head.  The candidate is
    reconstructed analytically from the target-shaped Q/K weights and the
    held-out live-child inputs.  No model forward, parent execution, numeric
    weight equality, gradient, or optimizer is involved.
    """

    torch = _torch()
    if not isinstance(
        producer,
        Lfm2CompiledStructuralAttentionProducerFold,
    ):
        raise TypeError("producer must be Lfm2CompiledStructuralAttentionProducerFold.")
    selected_cases = tuple(cases)
    if not selected_cases or any(
        not isinstance(case, Lfm2CachedAttentionFoldCase) for case in selected_cases
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "cases must contain held-out Lfm2CachedAttentionFoldCase values."
        )
    topology = producer.topology
    q_norm = (
        None
        if query_norm_weight is None
        else _float_tensor(
            query_norm_weight,
            "query_norm_weight",
            ndim=1,
        )
    )
    k_norm = (
        None
        if key_norm_weight is None
        else _float_tensor(
            key_norm_weight,
            "key_norm_weight",
            ndim=1,
        )
    )
    if q_norm is not None and tuple(q_norm.shape) != (topology.head_dim,):
        raise Lfm2StructuralResidualAttentionFoldError(
            "query_norm_weight must have shape [head_dim]."
        )
    if k_norm is not None and tuple(k_norm.shape) != (topology.head_dim,):
        raise Lfm2StructuralResidualAttentionFoldError(
            "key_norm_weight must have shape [head_dim]."
        )
    if (
        len(producer.query_groups) != topology.target_query_heads
        or len(producer.query_barycentric_weights) != topology.target_query_heads
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Compiled Q/K query-head barycentric evidence is incomplete."
        )

    reference_scores = []
    candidate_scores = []
    key_masks = []
    with torch.no_grad():
        for case_index, case in enumerate(selected_cases):
            normalized = _normalize_attention_value_output_case(
                case,
                case_index=case_index,
                producer=producer,
                query_norm_weight=q_norm,
                key_norm_weight=k_norm,
                normalized_query_norm_weight=q_norm,
                normalized_key_norm_weight=k_norm,
                norm_epsilon=norm_epsilon,
            )
            parent_query = _float_tensor(
                case.parent_query,
                f"cases[{case_index}].parent_query",
                ndim=4,
            )
            parent_key = _float_tensor(
                case.parent_key,
                f"cases[{case_index}].parent_key",
                ndim=4,
            )
            repeated_parent_key = parent_key.repeat_interleave(
                topology.source_query_heads_per_group,
                dim=1,
            )
            source_scores = torch.matmul(
                parent_query,
                repeated_parent_key.transpose(-2, -1),
            ) / math.sqrt(topology.head_dim)
            grouped_scores = []
            for target_head, (members, weights) in enumerate(
                zip(
                    producer.query_groups,
                    producer.query_barycentric_weights,
                    strict=True,
                )
            ):
                if (
                    not members
                    or len(members) != len(weights)
                    or any(
                        isinstance(member, bool)
                        or not isinstance(member, int)
                        or member < 0
                        or member >= topology.source_query_heads
                        for member in members
                    )
                    or not all(
                        math.isfinite(float(weight)) and float(weight) >= 0.0
                        for weight in weights
                    )
                ):
                    raise Lfm2StructuralResidualAttentionFoldError(
                        "Compiled Q/K barycentric query group is invalid."
                    )
                weight_sum = math.fsum(float(weight) for weight in weights)
                if not math.isclose(
                    weight_sum,
                    1.0,
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                ):
                    raise Lfm2StructuralResidualAttentionFoldError(
                        "Compiled Q/K barycentric query weights do not sum to one."
                    )
                index = torch.tensor(members, dtype=torch.long)
                selected = source_scores.index_select(1, index)
                barycentric = torch.tensor(
                    weights,
                    dtype=torch.float64,
                ).reshape(1, len(members), 1, 1)
                grouped = (selected * barycentric).sum(
                    dim=1,
                    keepdim=False,
                )
                if not bool(torch.isfinite(grouped).all().item()):
                    raise Lfm2StructuralResidualAttentionFoldError(
                        f"Target query head {target_head} has non-finite "
                        "held-out reference scores."
                    )
                grouped_scores.append(grouped)
            reference_scores.append(torch.stack(grouped_scores, dim=1).contiguous())
            candidate_scores.append(
                normalized["scores"]
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
                .contiguous()
            )
            key_masks.append(normalized["key_mask"])

    return evaluate_lfm2_structural_attention_score_preservation(
        tuple(reference_scores),
        tuple(candidate_scores),
        key_masks=tuple(key_masks),
        top_k=top_k,
        relative_margin_floor=relative_margin_floor,
        thresholds=thresholds,
    )


def _attention_context_rows_from_value(
    normalized_cases: Sequence[Mapping[str, Any]],
    value_weight: Any,
    topology: Lfm2StructuralAttentionTopology,
) -> tuple[Any, ...]:
    """Evaluate V through fixed cached child probabilities without a model."""

    value_heads = value_weight.reshape(
        topology.target_key_value_heads,
        topology.head_dim,
        topology.target_hidden_size,
    )
    rows = []
    for case in normalized_cases:
        weighted_inputs = case["weighted_inputs"]
        head_contexts = []
        for query_head in range(topology.target_query_heads):
            group = query_head // topology.target_query_heads_per_group
            head_contexts.append(weighted_inputs[:, query_head] @ value_heads[group].T)
        context = _torch().stack(head_contexts, dim=1)
        rows.append(
            context.permute(0, 2, 1, 3).reshape(
                int(case["batch"]) * int(case["query_length"]),
                topology.target_query_heads * topology.head_dim,
            )
        )
    return tuple(rows)


def _fit_attention_value_from_context_target(
    normalized_cases: Sequence[Mapping[str, Any]],
    context_target: Any,
    *,
    topology: Lfm2StructuralAttentionTopology,
    value_weight_prior: Any,
    ridge: float,
    max_prior_delta_ratio: float | None,
    iteration: int,
) -> tuple[Any, tuple[Lfm2PriorCorrectionReport, ...]]:
    """Solve every GQA value group against a desired flattened context."""

    context_offset = 0
    context_heads = []
    for case in normalized_cases:
        row_count = int(case["batch"]) * int(case["query_length"])
        context_heads.append(
            context_target[context_offset : context_offset + row_count]
            .reshape(
                int(case["batch"]),
                int(case["query_length"]),
                topology.target_query_heads,
                topology.head_dim,
            )
            .permute(0, 2, 1, 3)
        )
        context_offset += row_count
    if context_offset != int(context_target.shape[0]):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Coupled attention context target row alignment failed."
        )

    groups = []
    reports = []
    heads_per_group = topology.target_query_heads_per_group
    for group in range(topology.target_key_value_heads):
        start = group * heads_per_group
        stop = start + heads_per_group
        design = _torch().cat(
            [
                case["weighted_inputs"][:, start:stop].reshape(
                    -1,
                    topology.target_hidden_size,
                )
                for case in normalized_cases
            ],
            dim=0,
        )
        target = _torch().cat(
            [
                heads[:, start:stop].reshape(
                    -1,
                    topology.head_dim,
                )
                for heads in context_heads
            ],
            dim=0,
        )
        prior = value_weight_prior[
            group * topology.head_dim : (group + 1) * topology.head_dim
        ]
        fitted, report = _fit_linear(
            design,
            target,
            role=f"attention_value_output_iteration_{iteration}_group_{group}",
            ridge=ridge,
            prior=prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        groups.append(fitted)
        reports.append(report)
    return _torch().cat(groups, dim=0), tuple(reports)


def _attention_context_heads(
    context_rows: Sequence[Any],
    normalized_cases: Sequence[Mapping[str, Any]],
    topology: Lfm2StructuralAttentionTopology,
) -> tuple[Any, ...]:
    return tuple(
        rows.reshape(
            int(case["batch"]),
            int(case["query_length"]),
            topology.target_query_heads,
            topology.head_dim,
        ).permute(0, 2, 1, 3)
        for rows, case in zip(context_rows, normalized_cases)
    )


def _fit_attention_value_context_transport(
    actual_context_heads: Sequence[Any],
    normalized_cases: Sequence[Mapping[str, Any]],
    context_target: Any,
    *,
    topology: Lfm2StructuralAttentionTopology,
    live_value_weight: Any,
    value_weight_prior: Any,
    ridge: float,
    context_ridge: float,
    max_prior_delta_ratio: float | None,
    iteration: int,
) -> tuple[Any, tuple[Any, ...], tuple[Lfm2PriorCorrectionReport, ...]]:
    """Fit exact head-local V transforms from live contexts.

    For fixed attention probabilities, ``V' = A V`` implies ``C' = C A.T``.
    This remains exact when a sampled trace omits earlier K/V design rows,
    because the caller-supplied live context already contains their effect.
    """

    torch = _torch()
    context_offset = 0
    target_heads = []
    for case in normalized_cases:
        row_count = int(case["batch"]) * int(case["query_length"])
        target_heads.append(
            context_target[context_offset : context_offset + row_count]
            .reshape(
                int(case["batch"]),
                int(case["query_length"]),
                topology.target_query_heads,
                topology.head_dim,
            )
            .permute(0, 2, 1, 3)
        )
        context_offset += row_count
    if context_offset != int(context_target.shape[0]):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Coupled attention transport target row alignment failed."
        )

    value_groups = []
    transforms = []
    reports = []
    heads_per_group = topology.target_query_heads_per_group
    identity = torch.eye(topology.head_dim, dtype=torch.float64)
    for group in range(topology.target_key_value_heads):
        value_slice = slice(
            group * topology.head_dim,
            (group + 1) * topology.head_dim,
        )
        live_group = live_value_weight[value_slice]
        prior_group = value_weight_prior[value_slice]
        transform_prior, _prior_report = _fit_linear(
            live_group.T,
            prior_group.T,
            role=f"attention_value_transport_prior_group_{group}",
            ridge=context_ridge,
            prior=identity,
            max_prior_delta_ratio=None,
        )
        start = group * heads_per_group
        stop = start + heads_per_group
        design = torch.cat(
            [
                heads[:, start:stop].reshape(
                    -1,
                    topology.head_dim,
                )
                for heads in actual_context_heads
            ],
            dim=0,
        )
        target = torch.cat(
            [
                heads[:, start:stop].reshape(
                    -1,
                    topology.head_dim,
                )
                for heads in target_heads
            ],
            dim=0,
        )
        transform, report = _fit_linear(
            design,
            target,
            role=(
                f"attention_value_context_transport_iteration_{iteration}_group_{group}"
            ),
            ridge=ridge,
            prior=transform_prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        transforms.append(transform)
        value_groups.append(transform @ live_group)
        reports.append(report)

    transported_heads = []
    for heads in actual_context_heads:
        case_heads = []
        for query_head in range(topology.target_query_heads):
            group = query_head // topology.target_query_heads_per_group
            case_heads.append(heads[:, query_head] @ transforms[group].T)
        transported_heads.append(torch.stack(case_heads, dim=1))
    transported_rows = tuple(
        heads.permute(0, 2, 1, 3).reshape(
            int(case["batch"]) * int(case["query_length"]),
            topology.target_query_heads * topology.head_dim,
        )
        for heads, case in zip(transported_heads, normalized_cases)
    )
    return torch.cat(value_groups, dim=0), transported_rows, tuple(reports)


def compile_lfm2_structural_attention_value_output_fold(
    *,
    producer: Lfm2CompiledStructuralAttentionProducerFold,
    target_value_weight_template: Any,
    target_output_weight_template: Any,
    actual_child_contexts: Sequence[Any],
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    live_value_weight: Any,
    value_weight_prior: Any,
    output_weight_prior: Any,
    query_norm_weight: Any | None = None,
    key_norm_weight: Any | None = None,
    norm_epsilon: float = 1e-6,
    ridge: float = 0.0,
    context_ridge: float = 1e-6,
    max_prior_delta_ratio: float | None = 1.0,
    iterations: int = 2,
    fit_tolerance: float = 1e-10,
    context_equivalence_tolerance: float = 5e-2,
) -> Lfm2CompiledStructuralAttentionValueOutputFold:
    """Fit V and O jointly while holding the installed score path fixed.

    The supplied contexts must come from the live child after only Q/K were
    installed.  They first validate the analytical fixed-score reconstruction
    and establish an O-only control.  Alternating closed-form V/O proposals are
    committed to the artifact only when their endpoint fit is non-increasing.
    """

    torch = _torch()
    if not isinstance(
        producer,
        Lfm2CompiledStructuralAttentionProducerFold,
    ):
        raise TypeError("producer must be Lfm2CompiledStructuralAttentionProducerFold.")
    if (
        not cases
        or len(actual_child_contexts) != len(cases)
        or not all(isinstance(value, Lfm2CachedAttentionFoldCase) for value in cases)
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Actual attention contexts must align one-to-one with cases."
        )
    iterations = _positive_int(iterations, "iterations")
    ridge = _nonnegative_float(ridge, "ridge")
    context_ridge = _nonnegative_float(
        context_ridge,
        "context_ridge",
    )
    fit_tolerance = _nonnegative_float(
        fit_tolerance,
        "fit_tolerance",
    )
    context_equivalence_tolerance = _nonnegative_float(
        context_equivalence_tolerance,
        "context_equivalence_tolerance",
    )
    norm_epsilon = _nonnegative_float(
        norm_epsilon,
        "norm_epsilon",
    )
    if norm_epsilon == 0.0:
        raise Lfm2StructuralResidualAttentionFoldError("norm_epsilon must be positive.")
    if max_prior_delta_ratio is not None:
        max_prior_delta_ratio = _nonnegative_float(
            max_prior_delta_ratio,
            "max_prior_delta_ratio",
        )

    value_dtype, value_shape = _dense_template(
        target_value_weight_template,
        "target_value_weight_template",
    )
    output_dtype, output_shape = _dense_template(
        target_output_weight_template,
        "target_output_weight_template",
    )
    topology = producer.topology
    expected_value_shape = (
        topology.target_key_value_heads * topology.head_dim,
        topology.target_hidden_size,
    )
    expected_output_shape = (
        topology.target_output_size,
        topology.target_query_heads * topology.head_dim,
    )
    if value_shape != expected_value_shape or output_shape != expected_output_shape:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Coupled V/O target templates do not match producer topology."
        )
    live_value = _float_tensor(
        live_value_weight,
        "live_value_weight",
        ndim=2,
    )
    value_prior = _float_tensor(
        value_weight_prior,
        "value_weight_prior",
        ndim=2,
    )
    output_prior = _float_tensor(
        output_weight_prior,
        "output_weight_prior",
        ndim=2,
    )
    if (
        tuple(live_value.shape) != value_shape
        or tuple(value_prior.shape) != value_shape
        or tuple(output_prior.shape) != output_shape
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Coupled V/O live state and priors must have target shapes."
        )
    q_norm = (
        None
        if query_norm_weight is None
        else _float_tensor(
            query_norm_weight,
            "query_norm_weight",
            ndim=1,
        )
    )
    k_norm = (
        None
        if key_norm_weight is None
        else _float_tensor(
            key_norm_weight,
            "key_norm_weight",
            ndim=1,
        )
    )
    if q_norm is not None and tuple(q_norm.shape) != (topology.head_dim,):
        raise Lfm2StructuralResidualAttentionFoldError(
            "query_norm_weight must have shape [head_dim]."
        )
    if k_norm is not None and tuple(k_norm.shape) != (topology.head_dim,):
        raise Lfm2StructuralResidualAttentionFoldError(
            "key_norm_weight must have shape [head_dim]."
        )

    with torch.no_grad():
        normalized_cases = tuple(
            _normalize_attention_value_output_case(
                case,
                case_index=case_index,
                producer=producer,
                query_norm_weight=query_norm_weight,
                key_norm_weight=key_norm_weight,
                normalized_query_norm_weight=q_norm,
                normalized_key_norm_weight=k_norm,
                norm_epsilon=norm_epsilon,
            )
            for case_index, case in enumerate(cases)
        )
        actual_context_rows = tuple(
            _actual_attention_context_rows(
                actual_context,
                case=case,
                context_width=(topology.target_query_heads * topology.head_dim),
                name=f"actual_child_contexts[{case_index}]",
            )
            for case_index, (actual_context, case) in enumerate(
                zip(actual_child_contexts, cases)
            )
        )
        reconstructed_rows = _attention_context_rows_from_value(
            normalized_cases,
            live_value,
            topology,
        )
        actual_context = torch.cat(actual_context_rows, dim=0)
        reconstructed_context = torch.cat(reconstructed_rows, dim=0)
        context_error = _relative_rmse(
            reconstructed_context,
            actual_context,
        )
        use_analytical_value_design = context_error <= context_equivalence_tolerance
        value_context_solver = (
            "complete-fixed-score-weighted-input"
            if use_analytical_value_design
            else "exact-live-context-head-local-value-transport"
        )
        actual_context_head_cases = _attention_context_heads(
            actual_context_rows,
            normalized_cases,
            topology,
        )
        endpoint_target = torch.cat(
            [value["endpoint"] for value in normalized_cases],
            dim=0,
        )

        baseline_output, baseline_report = _fit_linear(
            actual_context,
            endpoint_target,
            role="attention_value_output_actual_context_control",
            ridge=ridge,
            prior=output_prior,
            max_prior_delta_ratio=max_prior_delta_ratio,
        )
        accepted_value = live_value.to(dtype=value_dtype).to(torch.float64)
        accepted_output = baseline_output.to(dtype=output_dtype).to(torch.float64)
        accepted_context_rows = (
            _attention_context_rows_from_value(
                normalized_cases,
                accepted_value,
                topology,
            )
            if use_analytical_value_design
            else actual_context_rows
        )
        accepted_context = torch.cat(accepted_context_rows, dim=0)
        fit_history = [
            _relative_rmse(
                accepted_context @ accepted_output.T,
                endpoint_target,
            )
        ]
        correction_reports = [baseline_report]
        accepted_iterations = 0

        for iteration in range(1, iterations + 1):
            gram = accepted_output.T @ accepted_output + context_ridge * torch.eye(
                int(accepted_output.shape[1]),
                dtype=torch.float64,
            )
            right = endpoint_target @ accepted_output
            try:
                context_target = torch.linalg.solve(
                    gram,
                    right.T,
                ).T
            except RuntimeError:
                context_target = torch.linalg.lstsq(
                    gram,
                    right.T,
                    driver="gelsd",
                ).solution.T
            if not bool(torch.isfinite(context_target).all().item()):
                raise Lfm2StructuralResidualAttentionFoldError(
                    "Coupled V/O context target is non-finite."
                )
            if use_analytical_value_design:
                candidate_value_fp64, value_reports = (
                    _fit_attention_value_from_context_target(
                        normalized_cases,
                        context_target,
                        topology=topology,
                        value_weight_prior=value_prior,
                        ridge=ridge,
                        max_prior_delta_ratio=max_prior_delta_ratio,
                        iteration=iteration,
                    )
                )
                candidate_context_rows = None
            else:
                (
                    candidate_value_fp64,
                    candidate_context_rows,
                    value_reports,
                ) = _fit_attention_value_context_transport(
                    actual_context_head_cases,
                    normalized_cases,
                    context_target,
                    topology=topology,
                    live_value_weight=live_value,
                    value_weight_prior=value_prior,
                    ridge=ridge,
                    context_ridge=context_ridge,
                    max_prior_delta_ratio=max_prior_delta_ratio,
                    iteration=iteration,
                )
            candidate_value = candidate_value_fp64.to(dtype=value_dtype).to(
                torch.float64
            )
            if use_analytical_value_design:
                candidate_context_rows = _attention_context_rows_from_value(
                    normalized_cases,
                    candidate_value,
                    topology,
                )
            else:
                # Refit the materialized V transform so endpoint scoring tracks
                # the exact target-dtype tensor that will be installed.
                materialized_transforms = []
                for group in range(topology.target_key_value_heads):
                    value_slice = slice(
                        group * topology.head_dim,
                        (group + 1) * topology.head_dim,
                    )
                    transform, _report = _fit_linear(
                        live_value[value_slice].T,
                        candidate_value[value_slice].T,
                        role=(
                            "attention_value_materialized_transport_"
                            f"iteration_{iteration}_group_{group}"
                        ),
                        ridge=0.0,
                        prior=torch.eye(
                            topology.head_dim,
                            dtype=torch.float64,
                        ),
                        max_prior_delta_ratio=None,
                    )
                    materialized_transforms.append(transform)
                materialized_head_cases = []
                for heads in actual_context_head_cases:
                    case_heads = []
                    for query_head in range(topology.target_query_heads):
                        group = query_head // topology.target_query_heads_per_group
                        case_heads.append(
                            heads[:, query_head] @ materialized_transforms[group].T
                        )
                    materialized_head_cases.append(torch.stack(case_heads, dim=1))
                candidate_context_rows = tuple(
                    heads.permute(0, 2, 1, 3).reshape(
                        int(case["batch"]) * int(case["query_length"]),
                        topology.target_query_heads * topology.head_dim,
                    )
                    for heads, case in zip(
                        materialized_head_cases,
                        normalized_cases,
                    )
                )
            candidate_context = torch.cat(
                candidate_context_rows,
                dim=0,
            )
            candidate_output_fp64, output_report = _fit_linear(
                candidate_context,
                endpoint_target,
                role=(f"attention_value_output_iteration_{iteration}_output"),
                ridge=ridge,
                prior=output_prior,
                max_prior_delta_ratio=max_prior_delta_ratio,
            )
            candidate_output = candidate_output_fp64.to(dtype=output_dtype).to(
                torch.float64
            )
            candidate_error = _relative_rmse(
                candidate_context @ candidate_output.T,
                endpoint_target,
            )
            if candidate_error > fit_history[-1] + fit_tolerance:
                break
            accepted_value = candidate_value
            accepted_output = candidate_output
            accepted_context = candidate_context
            fit_history.append(candidate_error)
            correction_reports.extend(value_reports)
            correction_reports.append(output_report)
            accepted_iterations += 1

    compiled_value = (
        accepted_value.to(device="cpu", dtype=value_dtype).contiguous().clone()
    )
    compiled_output = (
        accepted_output.to(device="cpu", dtype=output_dtype).contiguous().clone()
    )
    compiled_state = {
        "v_proj.weight": compiled_value,
        "out_proj.weight": compiled_output,
    }
    template_spec = {
        "v_proj.weight": (value_shape, str(value_dtype)),
        "out_proj.weight": (output_shape, str(output_dtype)),
    }
    return Lfm2CompiledStructuralAttentionValueOutputFold(
        value_weight=compiled_value,
        output_weight=compiled_output,
        topology=topology,
        case_count=len(cases),
        iterations_requested=iterations,
        iterations_accepted=accepted_iterations,
        fit_relative_rmse_history=tuple(fit_history),
        target_template_sha256=_template_sha256(template_spec),
        source_derived_prior_sha256=_state_sha256(
            {
                "v_proj.weight": value_prior,
                "out_proj.weight": output_prior,
            }
        ),
        fixed_query_key_sha256=producer.score_state_sha256,
        live_value_sha256=_tensor_sha256(live_value),
        actual_child_context_sha256=_state_sha256(
            {
                f"context.{index:04d}": value
                for index, value in enumerate(actual_context_rows)
            }
        ),
        reconstructed_live_context_sha256=_state_sha256(
            {
                f"context.{index:04d}": value
                for index, value in enumerate(reconstructed_rows)
            }
        ),
        endpoint_target_sha256=_tensor_sha256(endpoint_target),
        compiled_state_sha256=_state_sha256(compiled_state),
        context_reconstruction_relative_rmse=context_error,
        residual_endpoint_relative_rmse=fit_history[-1],
        value_context_solver=value_context_solver,
        prior_correction_reports=tuple(correction_reports),
        fit_tolerance=fit_tolerance,
    )


def _actual_short_conv_context_rows(
    value: Any,
    *,
    case: Lfm2CachedShortConvFoldCase,
    context_width: int,
    endpoint_width: int,
    name: str,
) -> Any:
    raw_ndim = getattr(value, "ndim", None)
    if raw_ndim not in {2, 3}:
        raise Lfm2StructuralResidualAttentionFoldError(f"{name} must have rank 2 or 3.")
    context = _float_tensor(value, name, ndim=int(raw_ndim))
    endpoint = _float_tensor(
        case.parent_post_operator_residual_target_space,
        f"{name}.parent_endpoint",
        ndim=3,
    )
    batch, sequence, observed_hidden = map(int, endpoint.shape)
    if observed_hidden != endpoint_width:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} endpoint has the wrong target width."
        )
    expected = (batch * sequence, context_width)
    if context.ndim == 3:
        if tuple(context.shape) != (
            batch,
            sequence,
            context_width,
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{name} rank-3 context has the wrong shape."
            )
        return context.reshape(expected)
    if tuple(context.shape) != expected:
        raise Lfm2StructuralResidualAttentionFoldError(
            f"{name} row-major context has the wrong shape."
        )
    return context


def compile_lfm2_structural_short_conv_consumer_fold(
    *,
    target_out_proj_weight_template: Any,
    actual_child_contexts: Sequence[Any],
    cases: Sequence[Lfm2CachedShortConvFoldCase],
    target_out_proj_bias_template: Any | None = None,
    out_proj_weight_prior: Any | None = None,
    out_proj_bias_prior: Any | None = None,
    ridge: float = 0.0,
    max_prior_delta_ratio: float | None = 1.0,
) -> Lfm2CompiledStructuralShortConvConsumerFold:
    """Fit short-conv output only from caller-supplied actual child context."""

    torch = _torch()
    if (
        not cases
        or len(actual_child_contexts) != len(cases)
        or not all(isinstance(value, Lfm2CachedShortConvFoldCase) for value in cases)
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Actual short-conv contexts must align one-to-one with cases."
        )
    ridge = _nonnegative_float(ridge, "ridge")
    if max_prior_delta_ratio is not None:
        max_prior_delta_ratio = _nonnegative_float(
            max_prior_delta_ratio, "max_prior_delta_ratio"
        )
    weight_dtype, weight_shape = _dense_template(
        target_out_proj_weight_template,
        "target_out_proj_weight_template",
    )
    target_residual_width = int(weight_shape[0])
    target_context_width = int(weight_shape[1])
    fit_bias = target_out_proj_bias_template is not None
    bias_dtype = None
    bias_shape = None
    if fit_bias:
        bias_dtype, bias_shape = _dense_template(
            target_out_proj_bias_template,
            "target_out_proj_bias_template",
            ndim=1,
        )
        if bias_shape != (target_residual_width,):
            raise Lfm2StructuralResidualAttentionFoldError(
                "Short-conv output bias template has the wrong shape."
            )
    if (out_proj_bias_prior is not None) != fit_bias:
        raise Lfm2StructuralResidualAttentionFoldError(
            "Short-conv output bias prior/template topology differs."
        )
    weight_prior = None
    if out_proj_weight_prior is not None:
        weight_prior = _float_tensor(
            out_proj_weight_prior, "out_proj_weight_prior", ndim=2
        )
        if tuple(weight_prior.shape) != weight_shape:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Short-conv output weight prior has the wrong shape."
            )
    bias_prior = None
    if fit_bias:
        bias_prior = _float_tensor(out_proj_bias_prior, "out_proj_bias_prior", ndim=1)
        if tuple(bias_prior.shape) != bias_shape:
            raise Lfm2StructuralResidualAttentionFoldError(
                "Short-conv output bias prior has the wrong shape."
            )
    context_rows = []
    endpoint_rows = []
    for case_index, (actual_context, case) in enumerate(
        zip(actual_child_contexts, cases)
    ):
        context_rows.append(
            _actual_short_conv_context_rows(
                actual_context,
                case=case,
                context_width=target_context_width,
                endpoint_width=target_residual_width,
                name=f"actual_child_contexts[{case_index}]",
            )
        )
        parent_endpoint = _float_tensor(
            case.parent_post_operator_residual_target_space,
            f"cases[{case_index}].parent_endpoint",
            ndim=3,
        )
        target_residual = _float_tensor(
            case.target_pre_operator_residual,
            f"cases[{case_index}].target_residual",
            ndim=3,
        )
        if tuple(parent_endpoint.shape) != tuple(target_residual.shape):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"cases[{case_index}] endpoint topology is incompatible."
            )
        endpoint_rows.append(
            (parent_endpoint - target_residual).reshape(-1, target_residual_width)
        )
    design = torch.cat(context_rows, dim=0)
    endpoint = torch.cat(endpoint_rows, dim=0)
    weight, bias, fit = _fit_affine(
        design,
        endpoint,
        role="short_conv_output_actual_child_context",
        ridge=ridge,
        weight_prior=weight_prior,
        bias_prior=bias_prior,
        max_prior_delta_ratio=max_prior_delta_ratio,
        fit_bias=fit_bias,
    )
    compiled_weight = weight.to(device="cpu", dtype=weight_dtype).contiguous().clone()
    compiled_bias = (
        None
        if not fit_bias
        else bias.to(device="cpu", dtype=bias_dtype).contiguous().clone()
    )
    compiled_state = {"out_proj.weight": compiled_weight}
    if fit_bias:
        compiled_state["out_proj.bias"] = compiled_bias
    prediction = design @ compiled_weight.to(torch.float64).T
    if fit_bias:
        prediction = prediction + compiled_bias.to(torch.float64)
    prior_values = {} if weight_prior is None else {"out_proj.weight": weight_prior}
    if fit_bias:
        prior_values["out_proj.bias"] = bias_prior
    return Lfm2CompiledStructuralShortConvConsumerFold(
        out_proj_weight=compiled_weight,
        out_proj_bias=compiled_bias,
        target_hidden_size=target_residual_width,
        case_count=len(cases),
        target_template_sha256=_template_sha256(
            {
                name: (
                    tuple(value.shape),
                    str(value.dtype),
                )
                for name, value in {
                    "out_proj.weight": target_out_proj_weight_template,
                    **(
                        {}
                        if not fit_bias
                        else {"out_proj.bias": (target_out_proj_bias_template)}
                    ),
                }.items()
            }
        ),
        source_derived_prior_sha256=(
            None if not prior_values else _state_sha256(prior_values)
        ),
        actual_child_context_sha256=_state_sha256(
            {f"context.{index:04d}": value for index, value in enumerate(context_rows)}
        ),
        endpoint_target_sha256=_tensor_sha256(endpoint),
        compiled_state_sha256=_state_sha256(compiled_state),
        residual_endpoint_relative_rmse=_relative_rmse(prediction, endpoint),
        prior_correction_report=fit,
        target_context_width=(
            target_context_width
            if target_context_width != target_residual_width
            else None
        ),
    )


def install_lfm2_structural_attention_fold(
    target_attention: Any,
    compiled: Lfm2CompiledStructuralAttentionFold,
) -> Lfm2StructuralAttentionInstallReport:
    """Copy a compiled Q/K/V/O state into existing target Parameters in place."""

    torch = _torch()
    if not isinstance(compiled, Lfm2CompiledStructuralAttentionFold):
        raise TypeError("compiled must be Lfm2CompiledStructuralAttentionFold.")
    modules = {
        "q_proj.weight": getattr(
            getattr(target_attention, "q_proj", None), "weight", None
        ),
        "k_proj.weight": getattr(
            getattr(target_attention, "k_proj", None), "weight", None
        ),
        "v_proj.weight": getattr(
            getattr(target_attention, "v_proj", None), "weight", None
        ),
        "out_proj.weight": getattr(
            getattr(target_attention, "out_proj", None), "weight", None
        ),
    }
    expected = compiled.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in modules.values()):
        raise Lfm2StructuralResidualAttentionInstallError(
            "target_attention must expose q_proj/k_proj/v_proj/out_proj weights."
        )
    for name in _STATE_KEYS:
        if (
            tuple(modules[name].shape) != tuple(expected[name].shape)
            or modules[name].dtype != expected[name].dtype
        ):
            raise Lfm2StructuralResidualAttentionInstallError(
                f"Target {name} does not match the compiled shape/dtype."
            )
    identities = {name: id(value) for name, value in modules.items()}
    shapes = {name: tuple(value.shape) for name, value in modules.items()}
    gradients = {
        name: (
            bool(value.requires_grad),
            None if value.grad is None else id(value.grad),
            None if value.grad is None else _tensor_sha256(value.grad),
        )
        for name, value in modules.items()
    }
    backup = {name: value.detach().cpu().clone() for name, value in modules.items()}
    try:
        with torch.no_grad():
            for name in _STATE_KEYS:
                modules[name].copy_(
                    expected[name].to(
                        device=modules[name].device,
                        dtype=modules[name].dtype,
                    )
                )
        installed = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in modules.items()
        }
        report = Lfm2StructuralAttentionInstallReport(
            parameter_identities_preserved=all(
                id(modules[name]) == identities[name] for name in _STATE_KEYS
            ),
            parameter_shapes_preserved=all(
                tuple(modules[name].shape) == shapes[name] for name in _STATE_KEYS
            ),
            gradient_state_preserved=all(
                (
                    bool(modules[name].requires_grad),
                    None if modules[name].grad is None else id(modules[name].grad),
                    None
                    if modules[name].grad is None
                    else _tensor_sha256(modules[name].grad),
                )
                == gradients[name]
                for name in _STATE_KEYS
            ),
            target_logical_parameters=sum(
                int(value.numel()) for value in modules.values()
            ),
            compiled_state_sha256=compiled.compiled_state_sha256,
            installed_state_sha256=_state_sha256(installed),
        )
        if not report.accepted:
            raise Lfm2StructuralResidualAttentionInstallError(
                "Installed state failed identity, shape, gradient, or content checks."
            )
        return report
    except Exception:
        with torch.no_grad():
            for name in _STATE_KEYS:
                modules[name].copy_(
                    backup[name].to(
                        device=modules[name].device,
                        dtype=modules[name].dtype,
                    )
                )
        raise


def install_lfm2_structural_short_conv_fold(
    target_short_conv: Any,
    compiled: Lfm2CompiledStructuralShortConvFold,
) -> Lfm2StructuralAttentionInstallReport:
    """Atomically copy a compiled B/C/x, FIR, and output state in place."""

    torch = _torch()
    if not isinstance(compiled, Lfm2CompiledStructuralShortConvFold):
        raise TypeError("compiled must be Lfm2CompiledStructuralShortConvFold.")
    in_proj = getattr(target_short_conv, "in_proj", None)
    depthwise_conv = getattr(target_short_conv, "conv", None)
    out_proj = getattr(target_short_conv, "out_proj", None)
    available = {
        "in_proj.weight": getattr(in_proj, "weight", None),
        "in_proj.bias": getattr(in_proj, "bias", None),
        "conv.weight": getattr(depthwise_conv, "weight", None),
        "conv.bias": getattr(depthwise_conv, "bias", None),
        "out_proj.weight": getattr(out_proj, "weight", None),
        "out_proj.bias": getattr(out_proj, "bias", None),
    }
    expected = compiled.state_dict()
    expected_bias = "in_proj.bias" in expected
    observed_bias = all(
        isinstance(available[name], torch.Tensor)
        for name in ("in_proj.bias", "conv.bias", "out_proj.bias")
    )
    observed_partial_bias = any(
        isinstance(available[name], torch.Tensor)
        for name in ("in_proj.bias", "conv.bias", "out_proj.bias")
    )
    if observed_partial_bias != observed_bias or expected_bias != observed_bias:
        raise Lfm2StructuralResidualAttentionInstallError(
            "Target short-conv bias topology does not match the compiled state."
        )
    modules = {name: available[name] for name in expected}
    if any(not isinstance(value, torch.Tensor) for value in modules.values()):
        raise Lfm2StructuralResidualAttentionInstallError(
            "target_short_conv must expose canonical in_proj, depthwise conv, "
            "and out_proj Parameters."
        )
    for name in modules:
        if (
            tuple(modules[name].shape) != tuple(expected[name].shape)
            or modules[name].dtype != expected[name].dtype
        ):
            raise Lfm2StructuralResidualAttentionInstallError(
                f"Target {name} does not match the compiled shape/dtype."
            )
    identities = {name: id(value) for name, value in modules.items()}
    shapes = {name: tuple(value.shape) for name, value in modules.items()}
    gradients = {
        name: (
            bool(value.requires_grad),
            None if value.grad is None else id(value.grad),
            None if value.grad is None else _tensor_sha256(value.grad),
        )
        for name, value in modules.items()
    }
    backup = {name: value.detach().cpu().clone() for name, value in modules.items()}
    try:
        with torch.no_grad():
            for name in modules:
                modules[name].copy_(
                    expected[name].to(
                        device=modules[name].device,
                        dtype=modules[name].dtype,
                    )
                )
        installed = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in modules.items()
        }
        report = Lfm2StructuralAttentionInstallReport(
            parameter_identities_preserved=all(
                id(modules[name]) == identities[name] for name in modules
            ),
            parameter_shapes_preserved=all(
                tuple(modules[name].shape) == shapes[name] for name in modules
            ),
            gradient_state_preserved=all(
                (
                    bool(modules[name].requires_grad),
                    None if modules[name].grad is None else id(modules[name].grad),
                    None
                    if modules[name].grad is None
                    else _tensor_sha256(modules[name].grad),
                )
                == gradients[name]
                for name in modules
            ),
            target_logical_parameters=sum(
                int(value.numel()) for value in modules.values()
            ),
            compiled_state_sha256=compiled.compiled_state_sha256,
            installed_state_sha256=_state_sha256(installed),
        )
        if not report.accepted:
            raise Lfm2StructuralResidualAttentionInstallError(
                "Installed short-conv failed identity, shape, gradient, or "
                "content checks."
            )
        return report
    except Exception:
        with torch.no_grad():
            for name in modules:
                modules[name].copy_(
                    backup[name].to(
                        device=modules[name].device,
                        dtype=modules[name].dtype,
                    )
                )
        raise


Lfm2StructuralShortConvInstallReport = Lfm2StructuralAttentionInstallReport
Lfm2StructuralMlpInstallReport = Lfm2StructuralAttentionInstallReport


@dataclass(frozen=True, slots=True)
class Lfm2CompiledStructuralLayerFold:
    """One atomic target-shaped attention plus SwiGLU structural candidate."""

    attention: Lfm2CompiledStructuralAttentionFold
    mlp: Lfm2CompiledStructuralMlpFold
    compiled_state_sha256: str

    @property
    def source_logical_parameters(self) -> int:
        return (
            self.attention.source_logical_parameters
            + self.mlp.source_logical_parameters
        )

    @property
    def target_logical_parameters(self) -> int:
        return (
            self.attention.target_logical_parameters
            + self.mlp.target_logical_parameters
        )

    def state_dict(self) -> Mapping[str, Any]:
        values = {
            f"self_attn.{name}": value
            for name, value in self.attention.state_dict().items()
        }
        values.update(
            {
                f"feed_forward.{name}": value
                for name, value in self.mlp.state_dict().items()
            }
        )
        return MappingProxyType(
            {name: value.detach().cpu().clone() for name, value in values.items()}
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-lfm2-structural-layer-fold-v1",
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "removed_logical_parameters": (
                self.source_logical_parameters - self.target_logical_parameters
            ),
            "compiled_state_sha256": self.compiled_state_sha256,
            "attention": self.attention.as_dict(),
            "mlp": self.mlp.as_dict(),
            "atomic_qkvo_and_swiglu": True,
            "blank_target_values_read": False,
            "training_steps": 0,
        }


def compose_lfm2_structural_layer_fold(
    attention: Lfm2CompiledStructuralAttentionFold,
    mlp: Lfm2CompiledStructuralMlpFold,
) -> Lfm2CompiledStructuralLayerFold:
    """Bind independently compiled branches into one atomic layer candidate."""

    if not isinstance(attention, Lfm2CompiledStructuralAttentionFold):
        raise TypeError("attention must be Lfm2CompiledStructuralAttentionFold.")
    if not isinstance(mlp, Lfm2CompiledStructuralMlpFold):
        raise TypeError("mlp must be Lfm2CompiledStructuralMlpFold.")
    if (
        attention.topology.target_hidden_size != mlp.target_hidden_size
        or attention.topology.target_output_size != mlp.target_hidden_size
        or attention.residual_assignment_sha256 != mlp.residual_assignment_sha256
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "Attention and SwiGLU must share one target hidden width and the same "
            "global sparse residual-assignment digest."
        )
    values = {
        f"self_attn.{name}": value for name, value in attention.state_dict().items()
    }
    values.update(
        {f"feed_forward.{name}": value for name, value in mlp.state_dict().items()}
    )
    return Lfm2CompiledStructuralLayerFold(
        attention=attention,
        mlp=mlp,
        compiled_state_sha256=_state_sha256(values),
    )


def install_lfm2_structural_mlp_fold(
    target_mlp: Any,
    compiled: Lfm2CompiledStructuralMlpFold,
) -> Lfm2StructuralMlpInstallReport:
    """Copy a compiled gate/up/down state into existing target Parameters."""

    torch = _torch()
    if not isinstance(compiled, Lfm2CompiledStructuralMlpFold):
        raise TypeError("compiled must be Lfm2CompiledStructuralMlpFold.")
    modules = {
        "w1.weight": getattr(getattr(target_mlp, "w1", None), "weight", None),
        "w3.weight": getattr(getattr(target_mlp, "w3", None), "weight", None),
        "w2.weight": getattr(getattr(target_mlp, "w2", None), "weight", None),
    }
    expected = compiled.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in modules.values()):
        raise Lfm2StructuralResidualAttentionInstallError(
            "target_mlp must expose w1/w3/w2 weights."
        )
    for name in modules:
        if (
            tuple(modules[name].shape) != tuple(expected[name].shape)
            or modules[name].dtype != expected[name].dtype
        ):
            raise Lfm2StructuralResidualAttentionInstallError(
                f"Target {name} does not match the compiled shape/dtype."
            )
    identities = {name: id(value) for name, value in modules.items()}
    shapes = {name: tuple(value.shape) for name, value in modules.items()}
    gradients = {
        name: (
            bool(value.requires_grad),
            None if value.grad is None else id(value.grad),
            None if value.grad is None else _tensor_sha256(value.grad),
        )
        for name, value in modules.items()
    }
    backup = {name: value.detach().cpu().clone() for name, value in modules.items()}
    try:
        with torch.no_grad():
            for name in modules:
                modules[name].copy_(
                    expected[name].to(
                        device=modules[name].device,
                        dtype=modules[name].dtype,
                    )
                )
        installed = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in modules.items()
        }
        report = Lfm2StructuralMlpInstallReport(
            parameter_identities_preserved=all(
                id(modules[name]) == identities[name] for name in modules
            ),
            parameter_shapes_preserved=all(
                tuple(modules[name].shape) == shapes[name] for name in modules
            ),
            gradient_state_preserved=all(
                (
                    bool(modules[name].requires_grad),
                    None if modules[name].grad is None else id(modules[name].grad),
                    None
                    if modules[name].grad is None
                    else _tensor_sha256(modules[name].grad),
                )
                == gradients[name]
                for name in modules
            ),
            target_logical_parameters=sum(
                int(value.numel()) for value in modules.values()
            ),
            compiled_state_sha256=compiled.compiled_state_sha256,
            installed_state_sha256=_state_sha256(installed),
        )
        if not report.accepted:
            raise Lfm2StructuralResidualAttentionInstallError(
                "Installed SwiGLU failed identity, shape, gradient, or content checks."
            )
        return report
    except Exception:
        with torch.no_grad():
            for name in modules:
                modules[name].copy_(
                    backup[name].to(
                        device=modules[name].device,
                        dtype=modules[name].dtype,
                    )
                )
        raise


def install_lfm2_structural_layer_fold(
    target_layer: Any,
    compiled: Lfm2CompiledStructuralLayerFold,
) -> Lfm2StructuralAttentionInstallReport:
    """Atomically install target-shaped QKVO and SwiGLU into a prebuilt layer."""

    torch = _torch()
    if not isinstance(compiled, Lfm2CompiledStructuralLayerFold):
        raise TypeError("compiled must be Lfm2CompiledStructuralLayerFold.")
    attention = getattr(target_layer, "self_attn", None)
    mlp = getattr(target_layer, "feed_forward", None)
    modules = {
        "self_attn.q_proj.weight": getattr(
            getattr(attention, "q_proj", None), "weight", None
        ),
        "self_attn.k_proj.weight": getattr(
            getattr(attention, "k_proj", None), "weight", None
        ),
        "self_attn.v_proj.weight": getattr(
            getattr(attention, "v_proj", None), "weight", None
        ),
        "self_attn.out_proj.weight": getattr(
            getattr(attention, "out_proj", None), "weight", None
        ),
        "feed_forward.w1.weight": getattr(getattr(mlp, "w1", None), "weight", None),
        "feed_forward.w3.weight": getattr(getattr(mlp, "w3", None), "weight", None),
        "feed_forward.w2.weight": getattr(getattr(mlp, "w2", None), "weight", None),
    }
    expected = compiled.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in modules.values()):
        raise Lfm2StructuralResidualAttentionInstallError(
            "target_layer must expose standard self_attn QKVO and feed_forward "
            "w1/w3/w2 Parameters."
        )
    for name in modules:
        if (
            tuple(modules[name].shape) != tuple(expected[name].shape)
            or modules[name].dtype != expected[name].dtype
        ):
            raise Lfm2StructuralResidualAttentionInstallError(
                f"Target {name} does not match the compiled shape/dtype."
            )
    identities = {name: id(value) for name, value in modules.items()}
    shapes = {name: tuple(value.shape) for name, value in modules.items()}
    gradients = {
        name: (
            bool(value.requires_grad),
            None if value.grad is None else id(value.grad),
            None if value.grad is None else _tensor_sha256(value.grad),
        )
        for name, value in modules.items()
    }
    backup = {name: value.detach().cpu().clone() for name, value in modules.items()}
    try:
        with torch.no_grad():
            for name in modules:
                modules[name].copy_(
                    expected[name].to(
                        device=modules[name].device,
                        dtype=modules[name].dtype,
                    )
                )
        installed = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in modules.items()
        }
        report = Lfm2StructuralAttentionInstallReport(
            parameter_identities_preserved=all(
                id(modules[name]) == identities[name] for name in modules
            ),
            parameter_shapes_preserved=all(
                tuple(modules[name].shape) == shapes[name] for name in modules
            ),
            gradient_state_preserved=all(
                (
                    bool(modules[name].requires_grad),
                    None if modules[name].grad is None else id(modules[name].grad),
                    None
                    if modules[name].grad is None
                    else _tensor_sha256(modules[name].grad),
                )
                == gradients[name]
                for name in modules
            ),
            target_logical_parameters=sum(
                int(value.numel()) for value in modules.values()
            ),
            compiled_state_sha256=compiled.compiled_state_sha256,
            installed_state_sha256=_state_sha256(installed),
        )
        if not report.accepted:
            raise Lfm2StructuralResidualAttentionInstallError(
                "Atomic layer installation failed identity, shape, gradient, "
                "or content checks."
            )
        return report
    except Exception:
        with torch.no_grad():
            for name in modules:
                modules[name].copy_(
                    backup[name].to(
                        device=modules[name].device,
                        dtype=modules[name].dtype,
                    )
                )
        raise


@dataclass(frozen=True, slots=True)
class Lfm2StructuralSplitInstallReport:
    """Audit for one producer install, consumer commit, or rollback."""

    phase: str
    parameter_identities_preserved: bool
    parameter_shapes_preserved: bool
    parameter_dtypes_preserved: bool
    gradient_objects_preserved: bool
    parameter_values_verified: bool
    target_logical_parameters: int
    expected_state_sha256: str
    observed_state_sha256: str

    @property
    def accepted(self) -> bool:
        return (
            self.parameter_identities_preserved
            and self.parameter_shapes_preserved
            and self.parameter_dtypes_preserved
            and self.gradient_objects_preserved
            and self.parameter_values_verified
            and self.expected_state_sha256 == self.observed_state_sha256
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "accepted": self.accepted,
            "parameter_identities_preserved": (self.parameter_identities_preserved),
            "parameter_shapes_preserved": (self.parameter_shapes_preserved),
            "parameter_dtypes_preserved": (self.parameter_dtypes_preserved),
            "gradient_objects_preserved": (self.gradient_objects_preserved),
            "parameter_values_verified": (self.parameter_values_verified),
            "target_logical_parameters": self.target_logical_parameters,
            "expected_state_sha256": self.expected_state_sha256,
            "observed_state_sha256": self.observed_state_sha256,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }


def _resolve_split_parameter(target: Any, name: str) -> Any:
    torch = _torch()
    value = target
    for component in name.split("."):
        value = getattr(value, component, None)
        if value is None:
            break
    if not isinstance(value, torch.nn.Parameter):
        raise Lfm2StructuralResidualAttentionInstallError(
            f"Target does not expose Parameter {name!r}."
        )
    return value


def _split_parameter_state(
    parameters: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Any]:
    return {
        name: parameters[name].detach().cpu().contiguous().clone() for name in names
    }


class Lfm2StructuralSplitInstallTransaction:
    """Snapshot spanning a producer install and later consumer commit.

    The transaction owns exact pre-producer parameter and gradient snapshots.
    Leaving its context without a successful consumer commit rolls everything
    back.  It never executes the module.
    """

    __slots__ = (
        "_active",
        "_committed",
        "_consumer_names",
        "_gradient_objects",
        "_gradient_snapshots",
        "_kind",
        "_original_requires_grad",
        "_original_state",
        "_original_state_sha256",
        "_parameter_dtypes",
        "_parameter_ids",
        "_parameter_shapes",
        "_parameters",
        "_producer_expected",
        "_producer_names",
        "_rolled_back",
        "_target",
        "producer_report",
    )

    def __init__(
        self,
        *,
        kind: str,
        target: Any,
        producer_names: Sequence[str],
        consumer_names: Sequence[str],
        producer_expected: Mapping[str, Any],
    ) -> None:
        self._kind = kind
        self._target = target
        self._producer_names = tuple(producer_names)
        self._consumer_names = tuple(consumer_names)
        all_names = self._producer_names + self._consumer_names
        if (
            not all_names
            or len(set(all_names)) != len(all_names)
            or set(self._producer_names) != set(producer_expected)
        ):
            raise Lfm2StructuralResidualAttentionInstallError(
                "Split transaction parameter roles are invalid."
            )
        self._parameters = {
            name: _resolve_split_parameter(target, name) for name in all_names
        }
        self._parameter_ids = {
            name: id(value) for name, value in self._parameters.items()
        }
        self._parameter_shapes = {
            name: tuple(value.shape) for name, value in self._parameters.items()
        }
        self._parameter_dtypes = {
            name: value.dtype for name, value in self._parameters.items()
        }
        self._original_requires_grad = {
            name: bool(value.requires_grad) for name, value in self._parameters.items()
        }
        self._gradient_objects = {
            name: value.grad for name, value in self._parameters.items()
        }
        self._gradient_snapshots = {
            name: (
                None
                if value.grad is None
                else value.grad.detach().cpu().contiguous().clone()
            )
            for name, value in self._parameters.items()
        }
        self._original_state = _split_parameter_state(self._parameters, all_names)
        self._original_state_sha256 = _state_sha256(self._original_state)
        self._producer_expected = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in producer_expected.items()
        }
        for name, expected in self._producer_expected.items():
            parameter = self._parameters[name]
            if (
                tuple(expected.shape) != tuple(parameter.shape)
                or expected.dtype != parameter.dtype
            ):
                raise Lfm2StructuralResidualAttentionInstallError(
                    f"Compiled producer {name} does not match target."
                )
        self._active = True
        self._committed = False
        self._rolled_back = False
        self.producer_report = None

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def active(self) -> bool:
        return self._active

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def rolled_back(self) -> bool:
        return self._rolled_back

    @property
    def original_state_sha256(self) -> str:
        return self._original_state_sha256

    def _all_names(self) -> tuple[str, ...]:
        return self._producer_names + self._consumer_names

    def _static_state_holds(self) -> tuple[bool, bool, bool, bool]:
        identities = True
        shapes = True
        dtypes = True
        gradients = True
        for name, original in self._parameters.items():
            try:
                current = _resolve_split_parameter(self._target, name)
            except Lfm2StructuralResidualAttentionInstallError:
                identities = shapes = dtypes = gradients = False
                continue
            identities = identities and (
                current is original and id(current) == self._parameter_ids[name]
            )
            shapes = shapes and (tuple(current.shape) == self._parameter_shapes[name])
            dtypes = dtypes and (current.dtype == self._parameter_dtypes[name])
            expected_gradient = self._gradient_objects[name]
            gradients = gradients and (
                bool(current.requires_grad) == self._original_requires_grad[name]
                and current.grad is expected_gradient
            )
            if expected_gradient is not None:
                gradients = gradients and (
                    tuple(expected_gradient.shape)
                    == tuple(self._gradient_snapshots[name].shape)
                    and expected_gradient.dtype == self._gradient_snapshots[name].dtype
                    and _tensor_sha256(expected_gradient)
                    == _tensor_sha256(self._gradient_snapshots[name])
                )
        return identities, shapes, dtypes, gradients

    def _producer_state_holds(self) -> bool:
        observed = _split_parameter_state(self._parameters, self._producer_names)
        return _state_sha256(observed) == _state_sha256(self._producer_expected)

    def rollback(self) -> Lfm2StructuralSplitInstallReport:
        if not self._active:
            if self._rolled_back:
                observed = _split_parameter_state(self._parameters, self._all_names())
                static = self._static_state_holds()
                return Lfm2StructuralSplitInstallReport(
                    phase="rolled-back",
                    parameter_identities_preserved=static[0],
                    parameter_shapes_preserved=static[1],
                    parameter_dtypes_preserved=static[2],
                    gradient_objects_preserved=static[3],
                    parameter_values_verified=(
                        _state_sha256(observed) == self._original_state_sha256
                    ),
                    target_logical_parameters=sum(
                        int(value.numel()) for value in self._parameters.values()
                    ),
                    expected_state_sha256=self._original_state_sha256,
                    observed_state_sha256=_state_sha256(observed),
                )
            raise Lfm2StructuralResidualAttentionInstallError(
                "A committed split transaction cannot be rolled back."
            )
        for name, parameter in self._parameters.items():
            current = _resolve_split_parameter(self._target, name)
            if current is not parameter:
                raise Lfm2StructuralResidualAttentionInstallError(
                    "Cannot exactly roll back a replaced Parameter object."
                )
        with _torch().no_grad():
            for name, parameter in self._parameters.items():
                parameter.copy_(
                    self._original_state[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
                parameter.requires_grad_(self._original_requires_grad[name])
                original_gradient = self._gradient_objects[name]
                if original_gradient is None:
                    parameter.grad = None
                else:
                    original_gradient.copy_(
                        self._gradient_snapshots[name].to(
                            device=original_gradient.device,
                            dtype=original_gradient.dtype,
                        )
                    )
                    parameter.grad = original_gradient
        observed = _split_parameter_state(self._parameters, self._all_names())
        observed_sha = _state_sha256(observed)
        static = self._static_state_holds()
        report = Lfm2StructuralSplitInstallReport(
            phase="rolled-back",
            parameter_identities_preserved=static[0],
            parameter_shapes_preserved=static[1],
            parameter_dtypes_preserved=static[2],
            gradient_objects_preserved=static[3],
            parameter_values_verified=(observed_sha == self._original_state_sha256),
            target_logical_parameters=sum(
                int(value.numel()) for value in self._parameters.values()
            ),
            expected_state_sha256=self._original_state_sha256,
            observed_state_sha256=observed_sha,
        )
        if not report.accepted:
            raise Lfm2StructuralResidualAttentionInstallError(
                "Split transaction rollback was not exact."
            )
        self._active = False
        self._rolled_back = True
        return report

    def __enter__(self) -> "Lfm2StructuralSplitInstallTransaction":
        if not self._active:
            raise Lfm2StructuralResidualAttentionInstallError(
                "Split transaction is no longer active."
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._active:
            self.rollback()
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self._kind,
            "active": self._active,
            "committed": self._committed,
            "rolled_back": self._rolled_back,
            "producer_parameter_names": list(self._producer_names),
            "consumer_parameter_names": list(self._consumer_names),
            "original_state_sha256": self._original_state_sha256,
            "producer_report": (
                None if self.producer_report is None else self.producer_report.as_dict()
            ),
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }


def _install_split_producer(
    *,
    kind: str,
    target: Any,
    producer_expected: Mapping[str, Any],
    consumer_names: Sequence[str],
    compiled_state_sha256: str,
) -> Lfm2StructuralSplitInstallTransaction:
    transaction = Lfm2StructuralSplitInstallTransaction(
        kind=kind,
        target=target,
        producer_names=tuple(producer_expected),
        consumer_names=tuple(consumer_names),
        producer_expected=producer_expected,
    )
    try:
        with _torch().no_grad():
            for name, expected in producer_expected.items():
                parameter = transaction._parameters[name]
                parameter.copy_(
                    expected.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        observed = _split_parameter_state(
            transaction._parameters,
            transaction._producer_names,
        )
        observed_sha = _state_sha256(observed)
        static = transaction._static_state_holds()
        report = Lfm2StructuralSplitInstallReport(
            phase="producer-installed",
            parameter_identities_preserved=static[0],
            parameter_shapes_preserved=static[1],
            parameter_dtypes_preserved=static[2],
            gradient_objects_preserved=static[3],
            parameter_values_verified=(
                observed_sha == compiled_state_sha256
                and transaction._producer_state_holds()
            ),
            target_logical_parameters=sum(
                int(value.numel()) for value in observed.values()
            ),
            expected_state_sha256=compiled_state_sha256,
            observed_state_sha256=observed_sha,
        )
        if not report.accepted:
            raise Lfm2StructuralResidualAttentionInstallError(
                "Split producer installation failed its audit."
            )
        transaction.producer_report = report
        return transaction
    except Exception:
        if transaction.active:
            transaction.rollback()
        raise


def _install_split_consumer(
    *,
    kind: str,
    target: Any,
    consumer_expected: Mapping[str, Any],
    compiled_state_sha256: str,
    transaction: Lfm2StructuralSplitInstallTransaction,
) -> Lfm2StructuralSplitInstallReport:
    if not isinstance(transaction, Lfm2StructuralSplitInstallTransaction):
        raise TypeError("transaction must be Lfm2StructuralSplitInstallTransaction.")
    try:
        if (
            not transaction.active
            or transaction.kind != kind
            or transaction._target is not target
            or set(transaction._consumer_names) != set(consumer_expected)
            or not transaction._producer_state_holds()
        ):
            raise Lfm2StructuralResidualAttentionInstallError(
                "Consumer does not match the active producer transaction."
            )
        static_before = transaction._static_state_holds()
        if not all(static_before):
            raise Lfm2StructuralResidualAttentionInstallError(
                "Target topology/gradient state changed between split phases."
            )
        for name, expected in consumer_expected.items():
            parameter = transaction._parameters[name]
            if (
                tuple(expected.shape) != tuple(parameter.shape)
                or expected.dtype != parameter.dtype
            ):
                raise Lfm2StructuralResidualAttentionInstallError(
                    f"Compiled consumer {name} does not match target."
                )
        with _torch().no_grad():
            for name, expected in consumer_expected.items():
                parameter = transaction._parameters[name]
                parameter.copy_(
                    expected.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        observed = _split_parameter_state(
            transaction._parameters,
            transaction._consumer_names,
        )
        observed_sha = _state_sha256(observed)
        static_after = transaction._static_state_holds()
        report = Lfm2StructuralSplitInstallReport(
            phase="consumer-committed",
            parameter_identities_preserved=static_after[0],
            parameter_shapes_preserved=static_after[1],
            parameter_dtypes_preserved=static_after[2],
            gradient_objects_preserved=static_after[3],
            parameter_values_verified=(
                observed_sha == compiled_state_sha256
                and transaction._producer_state_holds()
            ),
            target_logical_parameters=sum(
                int(value.numel()) for value in observed.values()
            ),
            expected_state_sha256=compiled_state_sha256,
            observed_state_sha256=observed_sha,
        )
        if not report.accepted:
            raise Lfm2StructuralResidualAttentionInstallError(
                "Split consumer installation failed its audit."
            )
        transaction._active = False
        transaction._committed = True
        return report
    except Exception:
        if transaction.active:
            transaction.rollback()
        raise


def install_lfm2_structural_attention_producer_fold(
    target_attention: Any,
    compiled: Lfm2CompiledStructuralAttentionProducerFold,
) -> Lfm2StructuralSplitInstallTransaction:
    """Install Q/K/V and retain an exact Q/K/V/O rollback snapshot."""

    if not isinstance(compiled, Lfm2CompiledStructuralAttentionProducerFold):
        raise TypeError("compiled must be Lfm2CompiledStructuralAttentionProducerFold.")
    return _install_split_producer(
        kind="attention-qkv-o",
        target=target_attention,
        producer_expected=compiled.state_dict(),
        consumer_names=("out_proj.weight",),
        compiled_state_sha256=compiled.compiled_state_sha256,
    )


def install_lfm2_structural_attention_score_producer_fold(
    target_attention: Any,
    compiled: Lfm2CompiledStructuralAttentionProducerFold,
) -> Lfm2StructuralSplitInstallTransaction:
    """Install fixed Q/K while retaining V/O for a coupled consumer solve."""

    if not isinstance(
        compiled,
        Lfm2CompiledStructuralAttentionProducerFold,
    ):
        raise TypeError("compiled must be Lfm2CompiledStructuralAttentionProducerFold.")
    return _install_split_producer(
        kind="attention-qk-vo",
        target=target_attention,
        producer_expected=compiled.score_state_dict(),
        consumer_names=("v_proj.weight", "out_proj.weight"),
        compiled_state_sha256=compiled.score_state_sha256,
    )


def install_lfm2_structural_attention_consumer_fold(
    target_attention: Any,
    compiled: Lfm2CompiledStructuralAttentionConsumerFold,
    transaction: Lfm2StructuralSplitInstallTransaction,
) -> Lfm2StructuralSplitInstallReport:
    """Commit O against an active Q/K/V producer transaction."""

    if not isinstance(compiled, Lfm2CompiledStructuralAttentionConsumerFold):
        raise TypeError("compiled must be Lfm2CompiledStructuralAttentionConsumerFold.")
    return _install_split_consumer(
        kind="attention-qkv-o",
        target=target_attention,
        consumer_expected=compiled.state_dict(),
        compiled_state_sha256=compiled.compiled_state_sha256,
        transaction=transaction,
    )


def install_lfm2_structural_attention_value_output_fold(
    target_attention: Any,
    compiled: Lfm2CompiledStructuralAttentionValueOutputFold,
    transaction: Lfm2StructuralSplitInstallTransaction,
) -> Lfm2StructuralSplitInstallReport:
    """Commit coupled V/O while verifying installed Q/K are byte unchanged."""

    if not isinstance(
        compiled,
        Lfm2CompiledStructuralAttentionValueOutputFold,
    ):
        raise TypeError(
            "compiled must be Lfm2CompiledStructuralAttentionValueOutputFold."
        )
    if (
        not isinstance(
            transaction,
            Lfm2StructuralSplitInstallTransaction,
        )
        or transaction.kind != "attention-qk-vo"
        or not transaction._producer_state_holds()
        or _state_sha256(
            _split_parameter_state(
                transaction._parameters,
                transaction._producer_names,
            )
        )
        != compiled.fixed_query_key_sha256
    ):
        if (
            isinstance(
                transaction,
                Lfm2StructuralSplitInstallTransaction,
            )
            and transaction.active
        ):
            transaction.rollback()
        raise Lfm2StructuralResidualAttentionInstallError(
            "Coupled V/O consumer does not match the fixed Q/K transaction."
        )
    return _install_split_consumer(
        kind="attention-qk-vo",
        target=target_attention,
        consumer_expected=compiled.state_dict(),
        compiled_state_sha256=compiled.compiled_state_sha256,
        transaction=transaction,
    )


def install_lfm2_structural_short_conv_producer_fold(
    target_short_conv: Any,
    compiled: Lfm2CompiledStructuralShortConvProducerFold,
) -> Lfm2StructuralSplitInstallTransaction:
    """Install B/C/x plus FIR and retain an exact full-operator snapshot.

    Dense writers expose one consumer parameter.  Authenticated fixed-rank
    writers expose the two physical factors ``[R,C]`` and ``[H,R]``.  The
    producer stage does not mutate either representation, but the split
    transaction must snapshot the exact physical consumer topology so a later
    factor-aware consumer commit (or rollback) remains byte exact.
    """

    if not isinstance(compiled, Lfm2CompiledStructuralShortConvProducerFold):
        raise TypeError("compiled must be Lfm2CompiledStructuralShortConvProducerFold.")
    torch = _torch()
    writer = getattr(target_short_conv, "out_proj", None)
    if isinstance(getattr(writer, "weight", None), torch.nn.Parameter):
        consumer_names = ["out_proj.weight"]
        if compiled.bias:
            consumer_names.append("out_proj.bias")
    elif (
        type(writer) is torch.nn.Sequential
        and len(writer) == 2
        and all(type(module) is torch.nn.Linear for module in writer)
        and all(module.bias is None for module in writer)
        and not compiled.bias
    ):
        consumer_names = ["out_proj.0.weight", "out_proj.1.weight"]
    else:
        raise Lfm2StructuralResidualAttentionInstallError(
            "Short-conv split install found neither a dense writer nor an "
            "authenticated two-factor bias-free writer."
        )
    return _install_split_producer(
        kind="short-conv-producer-consumer",
        target=target_short_conv,
        producer_expected=compiled.state_dict(),
        consumer_names=tuple(consumer_names),
        compiled_state_sha256=compiled.compiled_state_sha256,
    )


def install_lfm2_structural_short_conv_consumer_fold(
    target_short_conv: Any,
    compiled: Lfm2CompiledStructuralShortConvConsumerFold,
    transaction: Lfm2StructuralSplitInstallTransaction,
) -> Lfm2StructuralSplitInstallReport:
    """Commit output projection against an active short-conv producer."""

    if not isinstance(compiled, Lfm2CompiledStructuralShortConvConsumerFold):
        raise TypeError("compiled must be Lfm2CompiledStructuralShortConvConsumerFold.")
    return _install_split_consumer(
        kind="short-conv-producer-consumer",
        target=target_short_conv,
        consumer_expected=compiled.state_dict(),
        compiled_state_sha256=compiled.compiled_state_sha256,
        transaction=transaction,
    )


def install_lfm2_structural_short_conv_factor_consumer_fold(
    target_short_conv: Any,
    *,
    encoder_weight: Any,
    decoder_weight: Any,
    transaction: Lfm2StructuralSplitInstallTransaction,
) -> Lfm2StructuralSplitInstallReport:
    """Commit one fixed-rank physical writer pair against its producer split.

    The caller owns the authenticated logical-action factorization artifact;
    this installation boundary independently authenticates the two physical
    parameter names, shapes, dtypes, identities, gradients, values, and active
    producer transaction.  No module is replaced and no forward is executed.
    """

    torch = _torch()
    writer = getattr(target_short_conv, "out_proj", None)
    if (
        type(writer) is not torch.nn.Sequential
        or len(writer) != 2
        or any(type(module) is not torch.nn.Linear for module in writer)
        or any(module.bias is not None for module in writer)
    ):
        raise Lfm2StructuralResidualAttentionInstallError(
            "Factor consumer install requires exactly two bias-free Linear "
            "writer modules."
        )
    expected = {
        "out_proj.0.weight": encoder_weight,
        "out_proj.1.weight": decoder_weight,
    }
    for name, value in expected.items():
        parameter = _resolve_split_parameter(target_short_conv, name)
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or tuple(value.shape) != tuple(parameter.shape)
            or value.dtype != parameter.dtype
            or not value.is_floating_point()
            or value.is_complex()
            or not bool(torch.isfinite(value.detach()).all().item())
        ):
            raise Lfm2StructuralResidualAttentionInstallError(
                f"Compiled factor consumer {name} does not match target."
            )
    return _install_split_consumer(
        kind="short-conv-producer-consumer",
        target=target_short_conv,
        consumer_expected=expected,
        compiled_state_sha256=_state_sha256(expected),
        transaction=transaction,
    )


def rollback_lfm2_structural_split_install(
    transaction: Lfm2StructuralSplitInstallTransaction,
) -> Lfm2StructuralSplitInstallReport:
    """Explicitly restore the exact pre-producer branch snapshot."""

    if not isinstance(transaction, Lfm2StructuralSplitInstallTransaction):
        raise TypeError("transaction must be Lfm2StructuralSplitInstallTransaction.")
    return transaction.rollback()


def cached_short_conv_case_from_lfm2_trace(
    tensors: Mapping[str, Any],
    *,
    phase: str,
    layer_index: int,
    target_input: Any | None = None,
    parent_post_operator_residual_target_space: Any | None = None,
    target_pre_operator_residual: Any | None = None,
    parent_contribution: Any | None = None,
    reset_state: bool | None = None,
) -> Lfm2CachedShortConvFoldCase:
    """Adapt one chronological sealed LFM2 B/C/x phase without a rerun."""

    if not isinstance(tensors, Mapping):
        raise Lfm2StructuralResidualAttentionFoldError(
            "tensors must be a verified trace mapping."
        )
    _positive_int(layer_index + 1, "layer_index + 1")
    if not isinstance(phase, str) or not phase:
        raise Lfm2StructuralResidualAttentionFoldError(
            "phase must be a non-empty string."
        )
    if reset_state is not None and not isinstance(reset_state, bool):
        raise Lfm2StructuralResidualAttentionFoldError(
            "reset_state must be boolean when supplied."
        )
    base = f"{phase}.layer.{layer_index:03d}."

    def required(suffix: str) -> Any:
        key = base + suffix
        if key not in tensors:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"Verified trace is missing {key!r}."
            )
        return tensors[key]

    contribution = parent_contribution
    if contribution is None:
        contribution = tensors.get(base + "short_conv.context")
    return Lfm2CachedShortConvFoldCase(
        target_input=(
            required("operator.input") if target_input is None else target_input
        ),
        parent_b=required("short_conv.b"),
        parent_c=required("short_conv.c"),
        parent_x=required("short_conv.x"),
        parent_post_operator_residual_target_space=(
            required("residual.post_operator")
            if parent_post_operator_residual_target_space is None
            else parent_post_operator_residual_target_space
        ),
        target_pre_operator_residual=(
            required("residual.pre")
            if target_pre_operator_residual is None
            else target_pre_operator_residual
        ),
        reset_state=(
            not phase.startswith("decode") if reset_state is None else reset_state
        ),
        parent_contribution=contribution,
    )


def cached_attention_case_from_lfm2_trace(
    tensors: Mapping[str, Any],
    *,
    phase: str,
    layer_index: int,
    decode_trace_mode: str = _CACHED_VERIFIED_TRACE_MODE,
    target_query_input: Any | None = None,
    target_key_value_input: Any | None = None,
    parent_post_attention_residual_target_space: Any | None = None,
    target_pre_attention_residual: Any | None = None,
) -> Lfm2CachedAttentionFoldCase:
    """Adapt one verified circuit-cache phase without running the parent.

    This direct adapter is valid when the phase's sampled K/V rows cover the
    same key span as Q.  Longer cached-decode histories should be assembled
    from their already-sealed phase rows before calling the tensor API above.
    """

    if not isinstance(tensors, Mapping):
        raise Lfm2StructuralResidualAttentionFoldError(
            "tensors must be a verified trace mapping."
        )
    if decode_trace_mode not in _SUPPORTED_ATTENTION_TRACE_MODES:
        raise Lfm2StructuralResidualAttentionFoldError(
            "decode_trace_mode must be 'cached_verified' or 'full_prefix_canonical'."
        )
    _positive_int(layer_index + 1, "layer_index + 1")
    base = f"{phase}.layer.{layer_index:03d}."

    def required(suffix: str) -> Any:
        key = base + suffix
        if key not in tensors:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"Verified trace is missing {key!r}."
            )
        return tensors[key]

    operator_input = required("operator.input")
    query_linear = required("attention.q_linear")
    key_linear = required("attention.k_linear")
    value = required("attention.v")
    query = required("attention.q")
    key = required("attention.k")
    context = required("attention.context")
    q_length = int(query.shape[2])
    k_length = int(key.shape[2])
    if q_length != k_length:
        raise Lfm2StructuralResidualAttentionFoldError(
            "This direct phase adapter requires aligned sampled Q/K rows; use the "
            "multi-phase sealed-history assembler for cached decode."
        )
    torch = _torch()
    batch = int(query.shape[0])
    # This direct adapter is legal only for aligned Q/K spans.  An incremental
    # decode span has length one, while a full-prefix span may have many rows;
    # the same triangular mask is correct for both and prevents full-prefix
    # decode rows from accidentally attending to future sampled positions.
    key_mask = torch.ones((batch, q_length, k_length), dtype=torch.bool).tril()
    return Lfm2CachedAttentionFoldCase(
        target_query_input=(
            operator_input if target_query_input is None else target_query_input
        ),
        target_key_value_input=(
            operator_input if target_key_value_input is None else target_key_value_input
        ),
        parent_query_linear=query_linear,
        parent_key_linear=key_linear,
        parent_value=value,
        parent_query=query,
        parent_key=key,
        parent_context=context,
        parent_post_attention_residual_target_space=(
            required("residual.post_operator")
            if parent_post_attention_residual_target_space is None
            else parent_post_attention_residual_target_space
        ),
        target_pre_attention_residual=(
            required("residual.pre")
            if target_pre_attention_residual is None
            else target_pre_attention_residual
        ),
        key_mask=key_mask,
    )


def cached_attention_cases_from_lfm2_trace_phases(
    tensors: Mapping[str, Any],
    *,
    phases: Sequence[str],
    layer_index: int,
    decode_trace_mode: str = _CACHED_VERIFIED_TRACE_MODE,
    target_tensors: Mapping[str, Any] | None = None,
    parent_post_attention_residual_target_space_by_phase: Mapping[str, Any]
    | None = None,
) -> tuple[Lfm2CachedAttentionFoldCase, ...]:
    """Assemble prefill/decode histories entirely from sealed cache tensors.

    The current K-linear/V/operator-input rows are accumulated across phases
    and checked against each sealed decode KV cache.  This supplies full key
    histories to the compiler without executing the parent again.
    """

    torch = _torch()
    if not isinstance(tensors, Mapping) or not phases:
        raise Lfm2StructuralResidualAttentionFoldError(
            "tensors and at least one ordered phase are required."
        )
    if decode_trace_mode not in _SUPPORTED_ATTENTION_TRACE_MODES:
        raise Lfm2StructuralResidualAttentionFoldError(
            "decode_trace_mode must be 'cached_verified' or 'full_prefix_canonical'."
        )
    if target_tensors is not None and not isinstance(target_tensors, Mapping):
        raise Lfm2StructuralResidualAttentionFoldError(
            "target_tensors must be a mapping when supplied."
        )
    if (
        parent_post_attention_residual_target_space_by_phase is not None
        and not isinstance(
            parent_post_attention_residual_target_space_by_phase, Mapping
        )
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "parent target-space endpoints must be a phase mapping."
        )
    _positive_int(layer_index + 1, "layer_index + 1")
    ordered_phases = tuple(phases)
    if len(set(ordered_phases)) != len(ordered_phases) or any(
        not isinstance(value, str) or not value for value in ordered_phases
    ):
        raise Lfm2StructuralResidualAttentionFoldError(
            "phases must be unique non-empty strings in execution order."
        )

    if decode_trace_mode == _FULL_PREFIX_CANONICAL_TRACE_MODE:
        result = []
        for phase in ordered_phases:
            base = f"{phase}.layer.{layer_index:03d}."

            def required(mapping: Mapping[str, Any], suffix: str) -> Any:
                key = base + suffix
                if key not in mapping:
                    raise Lfm2StructuralResidualAttentionFoldError(
                        f"Verified trace is missing {key!r}."
                    )
                return mapping[key]

            key_cache_name = base + "attention.key_cache"
            value_cache_name = base + "attention.value_cache"
            if key_cache_name in tensors or value_cache_name in tensors:
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{phase} full_prefix_canonical evidence must not contain "
                    "incremental key/value caches."
                )
            target_mapping = tensors if target_tensors is None else target_tensors
            target_input = required(target_mapping, "operator.input")
            parent_query_linear = required(tensors, "attention.q_linear")
            parent_key_linear = required(tensors, "attention.k_linear")
            parent_value = required(tensors, "attention.v")
            parent_query = required(tensors, "attention.q")
            parent_key = required(tensors, "attention.k")
            parent_context = required(tensors, "attention.context")
            if (
                not isinstance(target_input, torch.Tensor)
                or target_input.ndim != 3
                or any(
                    not isinstance(value, torch.Tensor) or value.ndim != 4
                    for value in (
                        parent_query_linear,
                        parent_key_linear,
                        parent_value,
                        parent_query,
                        parent_key,
                        parent_context,
                    )
                )
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{phase} full-prefix attention rows have invalid ranks."
                )
            query_length = int(parent_query.shape[2])
            key_length = int(parent_key.shape[2])
            batch = int(parent_query.shape[0])
            if (
                query_length != key_length
                or int(target_input.shape[0]) != batch
                or int(target_input.shape[1]) != query_length
                or int(parent_query_linear.shape[2]) != query_length
                or int(parent_context.shape[2]) != query_length
                or int(parent_key_linear.shape[2]) != key_length
                or int(parent_value.shape[2]) != key_length
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{phase} full-prefix parent/target rows do not represent "
                    "one aligned sampled causal span."
                )
            key_mask = torch.ones(
                (batch, query_length, key_length),
                dtype=torch.bool,
            ).tril()
            if parent_post_attention_residual_target_space_by_phase is None:
                parent_endpoint = required(tensors, "residual.post_operator")
            else:
                if phase not in parent_post_attention_residual_target_space_by_phase:
                    raise Lfm2StructuralResidualAttentionFoldError(
                        f"Target-space parent endpoint is missing phase {phase!r}."
                    )
                parent_endpoint = parent_post_attention_residual_target_space_by_phase[
                    phase
                ]
            result.append(
                Lfm2CachedAttentionFoldCase(
                    target_query_input=target_input,
                    target_key_value_input=target_input,
                    parent_query_linear=parent_query_linear,
                    parent_key_linear=parent_key_linear,
                    parent_value=parent_value,
                    parent_query=parent_query,
                    parent_key=parent_key,
                    parent_context=parent_context,
                    parent_post_attention_residual_target_space=parent_endpoint,
                    target_pre_attention_residual=required(
                        target_mapping,
                        "residual.pre",
                    ),
                    key_mask=key_mask,
                )
            )
        return tuple(result)

    source_key_linear_history = None
    source_key_history = None
    source_value_history = None
    target_key_input_history = None
    result = []
    for phase_index, phase in enumerate(ordered_phases):
        base = f"{phase}.layer.{layer_index:03d}."

        def required(mapping: Mapping[str, Any], suffix: str) -> Any:
            key = base + suffix
            if key not in mapping:
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"Verified trace is missing {key!r}."
                )
            return mapping[key]

        target_mapping = tensors if target_tensors is None else target_tensors
        source_query_linear_current = required(tensors, "attention.q_linear")
        source_query_current = required(tensors, "attention.q")
        source_key_linear_current = required(tensors, "attention.k_linear")
        source_key_current = required(tensors, "attention.k")
        source_value_current = required(tensors, "attention.v")
        target_input_current = required(target_mapping, "operator.input")
        for name, value, axis in (
            ("attention.k_linear", source_key_linear_current, 2),
            ("attention.k", source_key_current, 2),
            ("attention.v", source_value_current, 2),
            ("target operator.input", target_input_current, 1),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 4 - (
                1 if axis == 1 else 0
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{phase} {name} has an invalid cached rank."
                )
        current_query_length = int(source_query_current.shape[2])
        if (
            int(source_query_linear_current.shape[2]) != current_query_length
            or int(source_key_current.shape[2]) != current_query_length
            or int(source_key_linear_current.shape[2]) != current_query_length
            or int(source_value_current.shape[2]) != current_query_length
            or int(target_input_current.shape[1]) != current_query_length
        ):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{phase} cached-incremental parent/target phase rows are not "
                "position-aligned."
            )
        if phase_index == 0:
            source_key_linear_history = source_key_linear_current
            source_key_history = source_key_current
            source_value_history = source_value_current
            target_key_input_history = target_input_current
        else:
            source_key_linear_history = torch.cat(
                (source_key_linear_history, source_key_linear_current), dim=2
            )
            source_key_history = torch.cat(
                (source_key_history, source_key_current), dim=2
            )
            source_value_history = torch.cat(
                (source_value_history, source_value_current), dim=2
            )
            target_key_input_history = torch.cat(
                (target_key_input_history, target_input_current), dim=1
            )
        key_cache_name = base + "attention.key_cache"
        value_cache_name = base + "attention.value_cache"
        cached_key = tensors.get(key_cache_name)
        cached_value = tensors.get(value_cache_name)
        if (cached_key is None) != (cached_value is None):
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{phase} must expose both sealed key/value caches or neither."
            )
        if phase_index > 0 and cached_key is None:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{phase} cached_verified decode evidence is missing its "
                "authenticated key/value cache."
            )
        if cached_key is not None:
            if (
                not isinstance(cached_key, torch.Tensor)
                or not isinstance(cached_value, torch.Tensor)
                or cached_key.ndim != 4
                or cached_value.ndim != 4
                or int(cached_key.shape[2]) != int(cached_value.shape[2])
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{phase} sealed key/value caches have invalid rolling-window shapes."
                )
            cache_length = int(cached_key.shape[2])
            accumulated_length = int(source_key_history.shape[2])
            if cache_length <= 0 or cache_length > accumulated_length:
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{phase} sealed cache length exceeds accumulated K/V history."
                )
            # Long traces intentionally seal a bounded tail.  Crop every
            # aligned representation together before validating the exact
            # post-norm cache; otherwise K/V, pre-norm K, and the target KV
            # design rows refer to different causal positions.
            source_key_linear_history = source_key_linear_history[
                :, :, -cache_length:, :
            ].contiguous()
            source_key_history = source_key_history[
                :, :, -cache_length:, :
            ].contiguous()
            source_value_history = source_value_history[
                :, :, -cache_length:, :
            ].contiguous()
            target_key_input_history = target_key_input_history[
                :, -cache_length:, :
            ].contiguous()
        if key_cache_name in tensors:
            if not torch.allclose(source_key_history, cached_key, rtol=1e-5, atol=1e-7):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{phase} accumulated K rows do not reproduce the sealed key cache."
                )
            source_key_history = cached_key
        if value_cache_name in tensors:
            if not torch.allclose(
                source_value_history, cached_value, rtol=1e-5, atol=1e-7
            ):
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"{phase} accumulated V rows do not reproduce the sealed value cache."
                )
            source_value_history = cached_value

        parent_query = source_query_current
        query_length = int(parent_query.shape[2])
        key_length = int(source_key_history.shape[2])
        query_start = key_length - query_length
        if query_start < 0:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"{phase} query rows exceed the accumulated key history."
            )
        query_positions = torch.arange(query_start, key_length).reshape(1, -1, 1)
        key_positions = torch.arange(key_length).reshape(1, 1, -1)
        key_mask = (key_positions <= query_positions).expand(
            int(parent_query.shape[0]), -1, -1
        )
        if parent_post_attention_residual_target_space_by_phase is None:
            parent_endpoint = required(tensors, "residual.post_operator")
        else:
            if phase not in parent_post_attention_residual_target_space_by_phase:
                raise Lfm2StructuralResidualAttentionFoldError(
                    f"Target-space parent endpoint is missing phase {phase!r}."
                )
            parent_endpoint = parent_post_attention_residual_target_space_by_phase[
                phase
            ]
        result.append(
            Lfm2CachedAttentionFoldCase(
                target_query_input=target_input_current,
                target_key_value_input=target_key_input_history,
                parent_query_linear=source_query_linear_current,
                parent_key_linear=source_key_linear_history,
                parent_value=source_value_history,
                parent_query=parent_query,
                parent_key=source_key_history,
                parent_context=required(tensors, "attention.context"),
                parent_post_attention_residual_target_space=parent_endpoint,
                target_pre_attention_residual=required(target_mapping, "residual.pre"),
                key_mask=key_mask,
            )
        )
    return tuple(result)


def cached_mlp_case_from_lfm2_trace(
    tensors: Mapping[str, Any],
    *,
    phase: str,
    layer_index: int,
    target_input: Any | None = None,
    parent_post_mlp_residual_target_space: Any | None = None,
    target_pre_mlp_residual: Any | None = None,
) -> Lfm2CachedMlpFoldCase:
    """Adapt sealed LFM2 gate/up/post-SwiGLU/residual tensors without a rerun."""

    if not isinstance(tensors, Mapping):
        raise Lfm2StructuralResidualAttentionFoldError(
            "tensors must be a verified trace mapping."
        )
    _positive_int(layer_index + 1, "layer_index + 1")
    base = f"{phase}.layer.{layer_index:03d}."

    def required(suffix: str) -> Any:
        key = base + suffix
        if key not in tensors:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"Verified trace is missing {key!r}."
            )
        return tensors[key]

    def rows(value: Any, name: str) -> Any:
        torch = _torch()
        if not isinstance(value, torch.Tensor) or value.ndim not in {2, 3}:
            raise Lfm2StructuralResidualAttentionFoldError(
                f"Verified trace {name} must have rank 2 or [batch, sequence, width]."
            )
        return value.reshape(-1, int(value.shape[-1]))

    return Lfm2CachedMlpFoldCase(
        target_input=rows(
            required("mlp.input") if target_input is None else target_input,
            "mlp.input",
        ),
        parent_gate=rows(required("mlp.gate"), "mlp.gate"),
        parent_up=rows(required("mlp.up"), "mlp.up"),
        parent_post_swiglu=rows(required("mlp.post_swiglu"), "mlp.post_swiglu"),
        parent_post_mlp_residual_target_space=(
            rows(required("residual.post_layer"), "residual.post_layer")
            if parent_post_mlp_residual_target_space is None
            else rows(
                parent_post_mlp_residual_target_space,
                "parent_post_mlp_residual_target_space",
            )
        ),
        target_pre_mlp_residual=(
            rows(required("residual.post_operator"), "residual.post_operator")
            if target_pre_mlp_residual is None
            else rows(target_pre_mlp_residual, "target_pre_mlp_residual")
        ),
    )


__all__ = [
    "Lfm2AttentionScorePreservationReport",
    "Lfm2AttentionScorePreservationThresholds",
    "Lfm2CachedAttentionFoldCase",
    "Lfm2CachedMlpFoldCase",
    "Lfm2CachedNormalizedInputMlpFoldCase",
    "Lfm2CachedShortConvFoldCase",
    "Lfm2CompiledStructuralAttentionConsumerFold",
    "Lfm2CompiledStructuralAttentionFold",
    "Lfm2CompiledStructuralAttentionProducerFold",
    "Lfm2CompiledStructuralLayerFold",
    "Lfm2CompiledStructuralMlpFold",
    "Lfm2CompiledRealMedoidStructuralMlpFold",
    "Lfm2CompiledNormalizedInputRealMedoidStructuralMlpFold",
    "Lfm2PreparedStructuralMlpFoldBasis",
    "Lfm2CompiledStructuralShortConvConsumerFold",
    "Lfm2CompiledStructuralShortConvFold",
    "Lfm2CompiledStructuralShortConvProducerFold",
    "Lfm2PriorCorrectionReport",
    "Lfm2SparseResidualAssignment",
    "Lfm2StructuralAttentionFoldAudit",
    "Lfm2StructuralAttentionInstallReport",
    "Lfm2StructuralMlpInstallReport",
    "Lfm2StructuralSplitInstallReport",
    "Lfm2StructuralSplitInstallTransaction",
    "Lfm2StructuralShortConvInstallReport",
    "Lfm2StructuralAttentionTopology",
    "Lfm2StructuralResidualAttentionFoldError",
    "Lfm2StructuralResidualAttentionInstallError",
    "cached_attention_case_from_lfm2_trace",
    "cached_attention_cases_from_lfm2_trace_phases",
    "cached_mlp_case_from_lfm2_trace",
    "cached_short_conv_case_from_lfm2_trace",
    "compose_lfm2_structural_layer_fold",
    "compile_lfm2_structural_attention_consumer_fold",
    "compile_lfm2_structural_attention_producer_fold",
    "compile_lfm2_structural_residual_attention_fold",
    "compile_lfm2_normalized_input_real_medoid_structural_mlp_fold",
    "compile_lfm2_real_medoid_structural_residual_mlp_fold",
    "compile_lfm2_structural_residual_mlp_fold",
    "compile_lfm2_structural_residual_short_conv_fold",
    "compile_lfm2_structural_short_conv_consumer_fold",
    "compile_lfm2_structural_short_conv_producer_fold",
    "evaluate_lfm2_structural_attention_score_preservation",
    "evaluate_lfm2_compiled_attention_producer_score_preservation",
    "install_lfm2_structural_attention_consumer_fold",
    "install_lfm2_structural_attention_fold",
    "install_lfm2_structural_attention_producer_fold",
    "install_lfm2_structural_layer_fold",
    "install_lfm2_structural_mlp_fold",
    "install_lfm2_structural_short_conv_consumer_fold",
    "install_lfm2_structural_short_conv_fold",
    "install_lfm2_structural_short_conv_producer_fold",
    "lfm2_authenticated_tensor_sha256",
    "prepare_lfm2_structural_residual_mlp_fold_basis",
    "rollback_lfm2_structural_split_install",
]
