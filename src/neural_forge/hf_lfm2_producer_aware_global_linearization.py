"""Producer-aware residual-attention folding for authenticated LFM2 stacks.

This v2a linearization keeps the v1 residual output writers and adds one
virtual hidden input transport to each pre-norm operator and MLP branch.  A
transport is parameterized as ``A = U @ C`` and perturbs the frozen normalized
producer input by ``normalized_input @ A.T``.  It can be absorbed into ordinary
dense child weights without a runtime sidecar:

* attention: ``delta_Wq/Wk/Wv = Wq/Wk/Wv @ A`` as one atomic group;
* short convolution: ``delta_Win = Win @ A`` across all B/C/X rows;
* SwiGLU: ``delta_Wgate/Wup = Wgate/Wup @ A`` as one atomic group.

All derivatives are explicit continuous FP64 JVPs/VJPs.  This module performs
no model forward, autograd, backward, optimizer step, training, pruning,
quantization, parameter sharing, or numerical quality gating.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal

from neural_forge.dynamic_fold_executor import run_fold_only_callback
from neural_forge.hf_lfm2_cached_qk_score_residual_fold import (
    apply_lfm2_finite_radius_rope,
    apply_lfm2_finite_radius_rope_transpose,
)
from neural_forge.hf_lfm2_global_linearization import (
    Lfm2AttentionPrimalSnapshot,
    Lfm2GlobalLinearizationError,
    Lfm2LinearizationNormBounds,
    Lfm2MacrocellLinearization,
    Lfm2MlpPrimalSnapshot,
    Lfm2ResidualSegmentTangent,
    Lfm2ShortConvPrimalSnapshot,
    Lfm2WholeStackPrimalSnapshot,
    Lfm2WholeStackConditioningScales,
    _causal_fir,
    _causal_fir_vjp,
    _linear,
    _macrocell_views,
    _macrocell_norm_factors,
    _matrix_spectral_norm_upper,
    _masked_softmax_jvp,
    _masked_softmax_vjp,
    _plain_fp64,
    _prefixed_state,
    _repeat_gqa,
    _rmsnorm_linearization_norm_upper,
    _scatter_scaled_batch_cotangent,
    _short_conv_padding_active,
    _silu_derivative,
    _stack_local_state,
    _sum_gqa_repeats,
    _terminal_transfer_norm_upper,
    _validate_coefficient,
    _validate_output_basis,
    _whole_stack_tensor_state,
    _workspace_bytes,
    lfm2_continuous_fp64_rmsnorm_jvp,
    lfm2_continuous_fp64_rmsnorm_vjp,
)
from neural_forge.hf_lfm2_global_residual_closure import (
    Lfm2GlobalResidualBoundaryBundle,
    Lfm2GlobalResidualLinearizationKernel,
    Lfm2GlobalResidualWriterSite,
    Lfm2GlobalResidualWriterTopology,
    adjoint_lfm2_global_residual_writer_block,
    apply_lfm2_global_residual_writer_block,
    fingerprint_lfm2_global_residual_tensor,
)


LFM2_PRODUCER_AWARE_LINEARIZATION_SCHEMA = (
    "neural-forge-lfm2-producer-aware-global-linearization-v2a"
)


class Lfm2ProducerAwareLinearizationError(Lfm2GlobalLinearizationError):
    """Producer-aware geometry, authentication, or transaction is invalid."""


class Lfm2ProducerAwareAuthenticationError(Lfm2ProducerAwareLinearizationError):
    """Producer-aware immutable evidence changed after capture."""


_MAX_FINITE_NORM_EVIDENCE = sys.float_info.max
_LOG_MAX_FINITE_NORM_EVIDENCE = math.log(_MAX_FINITE_NORM_EVIDENCE)
_TARGET_DTYPE_REPRESENTABILITY_MASS_LEVELS = (
    ("mass_0p01_percent", 0.0001),
    ("mass_0p1_percent", 0.001),
    ("mass_1_percent", 0.01),
    ("mass_10_percent", 0.1),
    ("mass_50_percent", 0.5),
)
_TARGET_DTYPE_REPRESENTABILITY_LEVELS = frozenset(
    {
        "requested",
        "first_effectful",
        *(name for name, _fraction in _TARGET_DTYPE_REPRESENTABILITY_MASS_LEVELS),
    }
)
_TARGET_DTYPE_REPRESENTABILITY_CHUNK_COORDINATES = 1 << 20


@dataclass(frozen=True, slots=True)
class Lfm2ProducerAwareLinearizationNormEvidence(Lfm2LinearizationNormBounds):
    """Finite proxies plus advisory log-domain evidence for deep-stack bounds.

    ``Lfm2LinearizationNormBounds`` is retained as the compatibility surface.
    When ``finite_float_overflow_observed`` is true, those inherited float
    fields are max-float sentinels rather than certified upper bounds.  The
    log-domain fields authenticate the untruncated analytic evidence without
    giving it solver or quality-gate authority.
    """

    operator_log_squared_norm_evidence: float | None
    block_log_squared_norm_evidence: tuple[float | None, ...]
    finite_float_overflow_observed: bool


def _require_finite_nonnegative_norm_operand(value: float) -> float:
    """Validate one operand before finite-saturating diagnostic arithmetic."""

    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware norm evidence contains a non-finite or negative factor."
        )
    return numeric


@dataclass(frozen=True, slots=True)
class _FiniteNormArithmetic:
    value: float
    saturated: bool = False


def _finite_norm_arithmetic(
    value: float | _FiniteNormArithmetic,
) -> _FiniteNormArithmetic:
    if isinstance(value, _FiniteNormArithmetic):
        _require_finite_nonnegative_norm_operand(value.value)
        return value
    return _FiniteNormArithmetic(_require_finite_nonnegative_norm_operand(value))


def _finite_saturating_norm_product(
    left: float | _FiniteNormArithmetic,
    right: float | _FiniteNormArithmetic,
) -> _FiniteNormArithmetic:
    """Multiply finite norm factors, using max-float as an overflow sentinel."""

    left_item = _finite_norm_arithmetic(left)
    right_item = _finite_norm_arithmetic(right)
    product = left_item.value * right_item.value
    operation_saturated = not math.isfinite(product)
    return _FiniteNormArithmetic(
        (
            _MAX_FINITE_NORM_EVIDENCE
            if operation_saturated
            else product
        ),
        left_item.saturated or right_item.saturated or operation_saturated,
    )


def _finite_saturating_norm_square(
    value: float | _FiniteNormArithmetic,
) -> _FiniteNormArithmetic:
    """Square a finite norm factor without allowing Python ``**`` overflow."""

    item = _finite_norm_arithmetic(value)
    try:
        squared = item.value**2
    except OverflowError:
        return _FiniteNormArithmetic(_MAX_FINITE_NORM_EVIDENCE, True)
    operation_saturated = not math.isfinite(squared)
    return _FiniteNormArithmetic(
        _MAX_FINITE_NORM_EVIDENCE if operation_saturated else squared,
        item.saturated or operation_saturated,
    )


def _finite_saturating_norm_sum(
    left: float | _FiniteNormArithmetic,
    right: float | _FiniteNormArithmetic,
) -> _FiniteNormArithmetic:
    """Add finite nonnegative evidence, saturating only representational overflow."""

    left_item = _finite_norm_arithmetic(left)
    right_item = _finite_norm_arithmetic(right)
    total = left_item.value + right_item.value
    operation_saturated = not math.isfinite(total)
    return _FiniteNormArithmetic(
        _MAX_FINITE_NORM_EVIDENCE if operation_saturated else total,
        left_item.saturated or right_item.saturated or operation_saturated,
    )


def _finite_saturating_norm_fsum(
    values: Sequence[float | _FiniteNormArithmetic],
) -> _FiniteNormArithmetic:
    """Retain ``math.fsum`` precision while making an over-range sum finite."""

    checked = tuple(_finite_norm_arithmetic(value) for value in values)
    try:
        total = math.fsum(value.value for value in checked)
    except OverflowError:
        return _FiniteNormArithmetic(_MAX_FINITE_NORM_EVIDENCE, True)
    return _FiniteNormArithmetic(
        total,
        any(value.saturated for value in checked),
    )


def _finite_norm_log(value: float) -> float | None:
    """Return a natural-log norm amplitude, with ``None`` representing zero."""

    numeric = _require_finite_nonnegative_norm_operand(value)
    return None if numeric == 0.0 else math.log(numeric)


def _log_norm_product(
    log_left: float | None,
    right: float,
) -> float | None:
    """Multiply a log-domain amplitude by one finite nonnegative factor."""

    right_value = _require_finite_nonnegative_norm_operand(right)
    if log_left is None or right_value == 0.0:
        return None
    return log_left + math.log(right_value)


def _log_norm_square(log_value: float | None) -> float | None:
    return None if log_value is None else 2.0 * log_value


def _log_norm_sum(
    log_left: float | None,
    log_right: float | None,
) -> float | None:
    """Add nonnegative quantities represented by natural logarithms."""

    if log_left is None:
        return log_right
    if log_right is None:
        return log_left
    high = max(log_left, log_right)
    low = min(log_left, log_right)
    return high + math.log1p(math.exp(low - high))


def _log_norm_fsum(values: Sequence[float | None]) -> float | None:
    total = None
    for value in values:
        total = _log_norm_sum(total, value)
    return total


def lfm2_producer_aware_site_causal_orders(
    layer_index: int,
) -> tuple[int, int, int, int]:
    """Return op-in, op-out, MLP-in, and MLP-down timeline positions."""

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "layer_index must be a nonnegative integer."
        )
    start = 6 * layer_index
    return (start, start + 1, start + 3, start + 4)


def lfm2_producer_aware_boundary_causal_order(
    layer_index: int,
    boundary_kind: str,
) -> int:
    """Return the shared site/boundary timeline position for one boundary."""

    site_orders = lfm2_producer_aware_site_causal_orders(layer_index)
    kind = str(boundary_kind).strip().lower()
    if kind == "post_operator_residual":
        return site_orders[1] + 1
    if kind == "post_layer_residual":
        return site_orders[3] + 1
    if kind == "terminal_observable":
        return 6 * (layer_index + 1)
    raise Lfm2ProducerAwareLinearizationError(
        "boundary_kind is invalid for the producer-aware causal timeline."
    )


def build_lfm2_producer_aware_writer_topology(
    snapshot: Lfm2WholeStackPrimalSnapshot,
    *,
    head_dimension: int,
    site_ids: Sequence[str] | None = None,
    requested_basis_rank: int | None = None,
) -> Lfm2GlobalResidualWriterTopology:
    """Derive the canonical four-sites-per-layer topology from primals."""

    if not isinstance(snapshot, Lfm2WholeStackPrimalSnapshot) or not (
        snapshot.verify_authentication()
    ):
        raise Lfm2ProducerAwareAuthenticationError(
            "snapshot must be authenticated before topology construction."
        )
    if (
        isinstance(head_dimension, bool)
        or not isinstance(head_dimension, int)
        or head_dimension <= 0
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "head_dimension must be a positive integer."
        )
    supplied_ids = (
        None if site_ids is None else tuple(str(value).strip() for value in site_ids)
    )
    expected_site_count = 4 * snapshot.layer_count
    if supplied_ids is not None and (
        len(supplied_ids) != expected_site_count
        or any(not value for value in supplied_ids)
        or len(set(supplied_ids)) != expected_site_count
    ):
        raise Lfm2ProducerAwareLinearizationError(
            f"site_ids must contain {expected_site_count} unique nonempty "
            "canonical names."
        )
    sites = []
    for layer_index, macrocell in enumerate(snapshot.batches[0].macrocells):
        kinds = (
            (
                "attention_input_transport"
                if macrocell.operator_kind == "full_attention"
                else "short_conv_input_transport"
            ),
            (
                "attention_output"
                if macrocell.operator_kind == "full_attention"
                else "short_conv_output"
            ),
            "mlp_input_transport",
            "mlp_down",
        )
        widths = (
            snapshot.hidden_size,
            macrocell.input_widths[0],
            snapshot.hidden_size,
            macrocell.input_widths[1],
        )
        causal_orders = lfm2_producer_aware_site_causal_orders(layer_index)
        for local_index, (kind, width, causal_order) in enumerate(
            zip(kinds, widths, causal_orders, strict=True)
        ):
            global_index = 4 * layer_index + local_index
            site_id = (
                supplied_ids[global_index]
                if supplied_ids is not None
                else f"layer.{layer_index}.{kind}"
            )
            sites.append(
                Lfm2GlobalResidualWriterSite(
                    site_id=site_id,
                    layer_index=layer_index,
                    causal_order=causal_order,
                    writer_kind=kind,
                    input_width=width,
                )
            )
    return Lfm2GlobalResidualWriterTopology(
        hidden_size=snapshot.hidden_size,
        head_dimension=head_dimension,
        sites=tuple(sites),
        requested_basis_rank=requested_basis_rank,
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for producer-aware LFM2 folding."
        ) from exc
    return torch


@dataclass(frozen=True, slots=True)
class Lfm2ProducerAwareSegmentCotangent:
    """Transpose result for one input transport plus one output writer."""

    incoming_residual: Any
    input_transport_coefficients: Any
    output_writer_coefficients: Any


def _producer_transport_tangent(
    normalized_input: Any,
    coefficients: Any,
    output_basis: Any,
    *,
    name: str,
) -> Any:
    basis = _validate_output_basis(
        output_basis,
        int(normalized_input.shape[-1]),
        normalized_input.device,
    )
    block = _validate_coefficient(
        coefficients,
        rank=int(basis.shape[1]),
        input_width=int(normalized_input.shape[-1]),
        device=normalized_input.device,
        name=name,
    )
    return apply_lfm2_global_residual_writer_block(
        normalized_input,
        basis,
        block,
    )


def _producer_transport_vjp(
    normalized_input: Any,
    normalized_input_cotangent: Any,
    output_basis: Any,
) -> Any:
    return adjoint_lfm2_global_residual_writer_block(
        normalized_input,
        output_basis,
        normalized_input_cotangent,
    )


def lfm2_producer_aware_attention_segment_jvp(
    snapshot: Lfm2AttentionPrimalSnapshot | SimpleNamespace,
    incoming_residual_tangent: Any,
    input_transport_coefficients: Any,
    output_writer_coefficients: Any,
    output_basis: Any,
) -> Lfm2ResidualSegmentTangent:
    """Attention JVP with coupled Q/K/V input transport and O correction."""

    torch = _torch()
    p = snapshot.parameters
    incoming = _plain_fp64(
        incoming_residual_tangent,
        "attention_incoming_residual_tangent",
        shape=tuple(snapshot.residual.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    output_block = _validate_coefficient(
        output_writer_coefficients,
        rank=int(basis.shape[1]),
        input_width=p.writer_input_width,
        device=snapshot.residual.device,
        name="attention_output_writer_coefficients",
    )
    batch, sequence, _ = map(int, snapshot.residual.shape)
    with torch.no_grad():
        d_normalized = lfm2_continuous_fp64_rmsnorm_jvp(
            snapshot.residual,
            incoming,
            p.operator_norm_gain,
            p.operator_norm_epsilon,
        )
        d_normalized = d_normalized + _producer_transport_tangent(
            snapshot.normalized_input,
            input_transport_coefficients,
            basis,
            name="attention_input_transport_coefficients",
        )
        dq_linear = _linear(d_normalized, p.query_weight).reshape(
            batch,
            sequence,
            p.query_heads,
            p.head_dim,
        )
        dk_linear = _linear(d_normalized, p.key_weight).reshape(
            batch,
            sequence,
            p.key_value_heads,
            p.head_dim,
        )
        d_value = (
            _linear(d_normalized, p.value_weight)
            .reshape(batch, sequence, p.key_value_heads, p.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        dq_pre = (
            lfm2_continuous_fp64_rmsnorm_jvp(
                snapshot.query_linear,
                dq_linear,
                p.query_norm_gain,
                p.query_key_norm_epsilon,
            )
            .transpose(1, 2)
            .contiguous()
        )
        dk_pre = (
            lfm2_continuous_fp64_rmsnorm_jvp(
                snapshot.key_linear,
                dk_linear,
                p.key_norm_gain,
                p.query_key_norm_epsilon,
            )
            .transpose(1, 2)
            .contiguous()
        )
        dq = apply_lfm2_finite_radius_rope(dq_pre, p.rope_cosine, p.rope_sine)
        dk = apply_lfm2_finite_radius_rope(dk_pre, p.rope_cosine, p.rope_sine)
        repeats = p.query_heads // p.key_value_heads
        drepeated_key = _repeat_gqa(dk, repeats)
        drepeated_value = _repeat_gqa(d_value, repeats)
        dscores = (
            dq @ snapshot.repeated_key.transpose(-1, -2)
            + snapshot.query @ drepeated_key.transpose(-1, -2)
        ) * p.scaling
        mask = p.causal_mask.expand_as(snapshot.probabilities)
        dprobabilities = _masked_softmax_jvp(
            snapshot.probabilities,
            dscores,
            mask,
        )
        dcontext = (
            dprobabilities @ snapshot.repeated_value
            + snapshot.probabilities @ drepeated_value
        )
        dwriter_input = (
            dcontext.transpose(1, 2)
            .contiguous()
            .reshape(batch, sequence, p.writer_input_width)
        )
        ddelta = _linear(dwriter_input, p.output_weight)
        ddelta = ddelta + apply_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            output_block,
        )
        output = incoming + ddelta
    return Lfm2ResidualSegmentTangent(delta=ddelta, output=output)


def lfm2_producer_aware_attention_segment_vjp(
    snapshot: Lfm2AttentionPrimalSnapshot | SimpleNamespace,
    output_cotangent: Any,
    output_basis: Any,
) -> Lfm2ProducerAwareSegmentCotangent:
    """Exact transpose of producer-aware attention JVP."""

    torch = _torch()
    p = snapshot.parameters
    cotangent = _plain_fp64(
        output_cotangent,
        "attention_output_cotangent",
        shape=tuple(snapshot.output.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    batch, sequence, _ = map(int, snapshot.residual.shape)
    with torch.no_grad():
        output_bar = adjoint_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            cotangent,
        )
        writer_input_bar = cotangent @ p.output_weight
        context_bar = (
            writer_input_bar.reshape(batch, sequence, p.query_heads, p.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        probability_bar = context_bar @ snapshot.repeated_value.transpose(-1, -2)
        repeated_value_bar = snapshot.probabilities.transpose(-1, -2) @ context_bar
        mask = p.causal_mask.expand_as(snapshot.probabilities)
        score_bar = (
            _masked_softmax_vjp(
                snapshot.probabilities,
                probability_bar,
                mask,
            )
            * p.scaling
        )
        query_bar = score_bar @ snapshot.repeated_key
        repeated_key_bar = score_bar.transpose(-1, -2) @ snapshot.query
        repeats = p.query_heads // p.key_value_heads
        key_bar = _sum_gqa_repeats(
            repeated_key_bar,
            p.key_value_heads,
            repeats,
        )
        value_bar = _sum_gqa_repeats(
            repeated_value_bar,
            p.key_value_heads,
            repeats,
        )
        query_pre_bar = (
            apply_lfm2_finite_radius_rope_transpose(
                query_bar,
                p.rope_cosine,
                p.rope_sine,
            )
            .transpose(1, 2)
            .contiguous()
        )
        key_pre_bar = (
            apply_lfm2_finite_radius_rope_transpose(
                key_bar,
                p.rope_cosine,
                p.rope_sine,
            )
            .transpose(1, 2)
            .contiguous()
        )
        query_linear_bar = lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.query_linear,
            query_pre_bar,
            p.query_norm_gain,
            p.query_key_norm_epsilon,
        ).reshape(batch, sequence, -1)
        key_linear_bar = lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.key_linear,
            key_pre_bar,
            p.key_norm_gain,
            p.query_key_norm_epsilon,
        ).reshape(batch, sequence, -1)
        value_linear_bar = (
            value_bar.transpose(1, 2).contiguous().reshape(batch, sequence, -1)
        )
        normalized_bar = (
            query_linear_bar @ p.query_weight
            + key_linear_bar @ p.key_weight
            + value_linear_bar @ p.value_weight
        )
        transport_bar = _producer_transport_vjp(
            snapshot.normalized_input,
            normalized_bar,
            basis,
        )
        incoming_bar = cotangent + lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.residual,
            normalized_bar,
            p.operator_norm_gain,
            p.operator_norm_epsilon,
        )
    return Lfm2ProducerAwareSegmentCotangent(
        incoming_residual=incoming_bar,
        input_transport_coefficients=transport_bar,
        output_writer_coefficients=output_bar,
    )


def lfm2_producer_aware_short_conv_segment_jvp(
    snapshot: Lfm2ShortConvPrimalSnapshot | SimpleNamespace,
    incoming_residual_tangent: Any,
    input_transport_coefficients: Any,
    output_writer_coefficients: Any,
    output_basis: Any,
) -> Lfm2ResidualSegmentTangent:
    """Short-conv JVP with one transport shared by all B/C/X rows."""

    torch = _torch()
    p = snapshot.parameters
    incoming = _plain_fp64(
        incoming_residual_tangent,
        "short_conv_incoming_residual_tangent",
        shape=tuple(snapshot.residual.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    output_block = _validate_coefficient(
        output_writer_coefficients,
        rank=int(basis.shape[1]),
        input_width=p.writer_input_width,
        device=snapshot.residual.device,
        name="short_conv_output_writer_coefficients",
    )
    with torch.no_grad():
        d_normalized = lfm2_continuous_fp64_rmsnorm_jvp(
            snapshot.residual,
            incoming,
            p.operator_norm_gain,
            p.operator_norm_epsilon,
        )
        d_normalized = d_normalized + _producer_transport_tangent(
            snapshot.normalized_input,
            input_transport_coefficients,
            basis,
            name="short_conv_input_transport_coefficients",
        )
        d_masked = d_normalized
        if _short_conv_padding_active(p.padding_mask):
            d_masked = d_normalized * p.padding_mask.unsqueeze(-1)
        d_projected = _linear(d_masked, p.input_weight)
        db, dc, dx = d_projected.chunk(3, dim=-1)
        dbx = db * snapshot.x_values + snapshot.b_values * dx
        dfiltered = _causal_fir(dbx, p.stored_fir_kernel)
        dwriter_input = dc * snapshot.filtered + snapshot.c_values * dfiltered
        ddelta = _linear(dwriter_input, p.output_weight)
        ddelta = ddelta + apply_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            output_block,
        )
        output = incoming + ddelta
    return Lfm2ResidualSegmentTangent(delta=ddelta, output=output)


def lfm2_producer_aware_short_conv_segment_vjp(
    snapshot: Lfm2ShortConvPrimalSnapshot | SimpleNamespace,
    output_cotangent: Any,
    output_basis: Any,
) -> Lfm2ProducerAwareSegmentCotangent:
    """Exact transpose of producer-aware short-conv JVP."""

    torch = _torch()
    p = snapshot.parameters
    cotangent = _plain_fp64(
        output_cotangent,
        "short_conv_output_cotangent",
        shape=tuple(snapshot.output.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    with torch.no_grad():
        output_bar = adjoint_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            cotangent,
        )
        writer_input_bar = cotangent @ p.output_weight
        c_bar = writer_input_bar * snapshot.filtered
        filtered_bar = writer_input_bar * snapshot.c_values
        bx_bar = _causal_fir_vjp(filtered_bar, p.stored_fir_kernel)
        b_bar = bx_bar * snapshot.x_values
        x_bar = bx_bar * snapshot.b_values
        projected_bar = torch.cat((b_bar, c_bar, x_bar), dim=-1)
        masked_bar = projected_bar @ p.input_weight
        normalized_bar = masked_bar
        if _short_conv_padding_active(p.padding_mask):
            normalized_bar = masked_bar * p.padding_mask.unsqueeze(-1)
        transport_bar = _producer_transport_vjp(
            snapshot.normalized_input,
            normalized_bar,
            basis,
        )
        incoming_bar = cotangent + lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.residual,
            normalized_bar,
            p.operator_norm_gain,
            p.operator_norm_epsilon,
        )
    return Lfm2ProducerAwareSegmentCotangent(
        incoming_residual=incoming_bar,
        input_transport_coefficients=transport_bar,
        output_writer_coefficients=output_bar,
    )


def lfm2_producer_aware_mlp_segment_jvp(
    snapshot: Lfm2MlpPrimalSnapshot | SimpleNamespace,
    incoming_residual_tangent: Any,
    input_transport_coefficients: Any,
    output_writer_coefficients: Any,
    output_basis: Any,
) -> Lfm2ResidualSegmentTangent:
    """SwiGLU JVP with coupled gate/up input transport and W2 correction."""

    torch = _torch()
    p = snapshot.parameters
    incoming = _plain_fp64(
        incoming_residual_tangent,
        "mlp_incoming_residual_tangent",
        shape=tuple(snapshot.residual.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    output_block = _validate_coefficient(
        output_writer_coefficients,
        rank=int(basis.shape[1]),
        input_width=p.writer_input_width,
        device=snapshot.residual.device,
        name="mlp_down_writer_coefficients",
    )
    with torch.no_grad():
        d_normalized = lfm2_continuous_fp64_rmsnorm_jvp(
            snapshot.residual,
            incoming,
            p.norm_gain,
            p.norm_epsilon,
        )
        d_normalized = d_normalized + _producer_transport_tangent(
            snapshot.normalized_input,
            input_transport_coefficients,
            basis,
            name="mlp_input_transport_coefficients",
        )
        dgate = _linear(d_normalized, p.gate_weight)
        dup = _linear(d_normalized, p.up_weight)
        dactivation = _silu_derivative(snapshot.gate) * dgate
        dwriter_input = dactivation * snapshot.up + snapshot.activation * dup
        ddelta = _linear(dwriter_input, p.down_weight)
        ddelta = ddelta + apply_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            output_block,
        )
        output = incoming + ddelta
    return Lfm2ResidualSegmentTangent(delta=ddelta, output=output)


def lfm2_producer_aware_mlp_segment_vjp(
    snapshot: Lfm2MlpPrimalSnapshot | SimpleNamespace,
    output_cotangent: Any,
    output_basis: Any,
) -> Lfm2ProducerAwareSegmentCotangent:
    """Exact transpose of producer-aware SwiGLU JVP."""

    torch = _torch()
    p = snapshot.parameters
    cotangent = _plain_fp64(
        output_cotangent,
        "mlp_output_cotangent",
        shape=tuple(snapshot.output.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    with torch.no_grad():
        output_bar = adjoint_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            cotangent,
        )
        writer_input_bar = cotangent @ p.down_weight
        gate_bar = writer_input_bar * snapshot.up * _silu_derivative(snapshot.gate)
        up_bar = writer_input_bar * snapshot.activation
        normalized_bar = gate_bar @ p.gate_weight + up_bar @ p.up_weight
        transport_bar = _producer_transport_vjp(
            snapshot.normalized_input,
            normalized_bar,
            basis,
        )
        incoming_bar = cotangent + lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.residual,
            normalized_bar,
            p.norm_gain,
            p.norm_epsilon,
        )
    return Lfm2ProducerAwareSegmentCotangent(
        incoming_residual=incoming_bar,
        input_transport_coefficients=transport_bar,
        output_writer_coefficients=output_bar,
    )


def _producer_input_direct_norm_factors(
    macrocell: Lfm2MacrocellLinearization,
    output_basis: Any,
) -> tuple[float, float]:
    """Bound direct operator/MLP transport maps at frozen primals."""

    operator = macrocell.operator
    basis_norm = _matrix_spectral_norm_upper(output_basis)
    operator_transport_input = basis_norm * _matrix_spectral_norm_upper(
        operator.normalized_input.reshape(-1, operator.parameters.hidden_size)
    )
    if isinstance(operator, Lfm2AttentionPrimalSnapshot):
        p = operator.parameters
        repeats = p.query_heads // p.key_value_heads
        query_norm = _rmsnorm_linearization_norm_upper(
            operator.query_linear,
            p.query_norm_gain,
            p.query_key_norm_epsilon,
        )
        key_norm = _rmsnorm_linearization_norm_upper(
            operator.key_linear,
            p.key_norm_gain,
            p.query_key_norm_epsilon,
        )
        half = p.head_dim // 2
        rope_gain = float(
            (p.rope_cosine[..., :half].square() + p.rope_sine[..., :half].square())
            .sqrt()
            .amax()
            .item()
        )
        query_lipschitz = (
            _matrix_spectral_norm_upper(p.query_weight) * query_norm * rope_gain
        )
        repeated_key_lipschitz = (
            _matrix_spectral_norm_upper(p.key_weight)
            * key_norm
            * rope_gain
            * math.sqrt(repeats)
        )
        repeated_value_lipschitz = _matrix_spectral_norm_upper(
            p.value_weight
        ) * math.sqrt(repeats)
        score_lipschitz = p.scaling * (
            _matrix_spectral_norm_upper(operator.repeated_key) * query_lipschitz
            + _matrix_spectral_norm_upper(operator.query) * repeated_key_lipschitz
        )
        softmax_lipschitz = float(
            (2.0 * operator.probabilities * (1.0 - operator.probabilities))
            .amax()
            .item()
        )
        context_lipschitz = (
            _matrix_spectral_norm_upper(operator.repeated_value)
            * softmax_lipschitz
            * score_lipschitz
            + _matrix_spectral_norm_upper(operator.probabilities)
            * repeated_value_lipschitz
        )
        operator_branch = (
            _matrix_spectral_norm_upper(p.output_weight) * context_lipschitz
        )
    else:
        p = operator.parameters
        padding_gain = 1.0
        if p.padding_mask is not None:
            padding_gain = float(p.padding_mask.abs().amax().item())
        projected_lipschitz = padding_gain * _matrix_spectral_norm_upper(p.input_weight)
        fir_lipschitz = float(p.stored_fir_kernel.abs().sum(dim=-1).amax().item())
        writer_input_lipschitz = projected_lipschitz * (
            float(operator.filtered.abs().amax().item())
            + float(operator.c_values.abs().amax().item())
            * fir_lipschitz
            * (
                float(operator.x_values.abs().amax().item())
                + float(operator.b_values.abs().amax().item())
            )
        )
        operator_branch = (
            _matrix_spectral_norm_upper(p.output_weight) * writer_input_lipschitz
        )
    mlp = macrocell.mlp
    mp = mlp.parameters
    mlp_transport_input = basis_norm * _matrix_spectral_norm_upper(
        mlp.normalized_input.reshape(-1, mp.hidden_size)
    )
    producer_lipschitz = float(
        (mlp.up * _silu_derivative(mlp.gate)).abs().amax().item()
    ) * _matrix_spectral_norm_upper(mp.gate_weight) + float(
        mlp.activation.abs().amax().item()
    ) * _matrix_spectral_norm_upper(mp.up_weight)
    mlp_branch = _matrix_spectral_norm_upper(mp.down_weight) * producer_lipschitz
    result = (
        operator_transport_input * operator_branch,
        mlp_transport_input * mlp_branch,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in result):
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware direct transport norm evidence is non-finite."
        )
    return result


def lfm2_producer_aware_whole_stack_norm_evidence(
    snapshot: Lfm2WholeStackPrimalSnapshot,
    output_basis: Any,
) -> tuple[
    Lfm2WholeStackConditioningScales,
    Lfm2ProducerAwareLinearizationNormEvidence,
]:
    """Return separate local scales and authenticated downstream norm evidence.

    Conditioning values are diagnostic numerical scales, never gates.  A
    closure adapter must multiply both scale families by its authenticated
    maximum coordinate weight before using them in weighted normal units.
    Representable downstream bounds retain their finite certified fields;
    over-range bounds use explicit finite sentinels plus advisory log evidence.
    """

    if not isinstance(snapshot, Lfm2WholeStackPrimalSnapshot) or not (
        snapshot.verify_authentication()
    ):
        raise Lfm2ProducerAwareAuthenticationError(
            "snapshot must be authenticated before norm analysis."
        )
    basis = _validate_output_basis(
        output_basis,
        snapshot.hidden_size,
        snapshot.batches[0].macrocells[0].operator.residual.device,
    )
    layer_count = snapshot.layer_count
    conditioning_blocks = [_FiniteNormArithmetic(0.0)] * (4 * layer_count)
    certified_blocks = [_FiniteNormArithmetic(0.0)] * (4 * layer_count)
    certified_log_blocks: list[float | None] = [None] * (4 * layer_count)
    for batch in snapshot.batches:
        factors = []
        for macrocell in batch.macrocells:
            operator_transfer, mlp_transfer, operator_output, mlp_output = (
                _macrocell_norm_factors(macrocell, basis)
            )
            operator_transport, mlp_transport = _producer_input_direct_norm_factors(
                macrocell,
                basis,
            )
            factors.append(
                (
                    operator_transfer,
                    mlp_transfer,
                    operator_transport,
                    operator_output,
                    mlp_transport,
                    mlp_output,
                )
            )
        terminal_transfer = _terminal_transfer_norm_upper(batch.terminal)
        for layer_index, layer_factors in enumerate(factors):
            operator_transfer, mlp_transfer, *directs = layer_factors
            del operator_transfer
            site_offset = 4 * layer_index
            for local_index, direct in enumerate(directs):
                is_operator_site = local_index < 2
                direct_log = _finite_norm_log(direct)
                direct_square = _finite_saturating_norm_square(direct)
                conditioning = direct_square
                if is_operator_site:
                    conditioning = _finite_saturating_norm_product(
                        direct_square,
                        _finite_saturating_norm_sum(
                            1.0,
                            _finite_saturating_norm_square(mlp_transfer),
                        ),
                    )
                block_index = site_offset + local_index
                conditioning_blocks[block_index] = _finite_saturating_norm_sum(
                    conditioning_blocks[block_index],
                    conditioning,
                )
                amplitude = _FiniteNormArithmetic(direct)
                amplitude_log = direct_log
                energy = 0.0
                energy_log = None
                if is_operator_site:
                    energy = _finite_saturating_norm_sum(
                        energy,
                        _finite_saturating_norm_square(amplitude),
                    )
                    energy_log = _log_norm_sum(
                        energy_log,
                        _log_norm_square(amplitude_log),
                    )
                    amplitude = _finite_saturating_norm_product(
                        amplitude,
                        mlp_transfer,
                    )
                    amplitude_log = _log_norm_product(
                        amplitude_log,
                        mlp_transfer,
                    )
                energy = _finite_saturating_norm_sum(
                    energy,
                    _finite_saturating_norm_square(amplitude),
                )
                energy_log = _log_norm_sum(
                    energy_log,
                    _log_norm_square(amplitude_log),
                )
                for later_layer in range(layer_index + 1, layer_count):
                    later_operator, later_mlp, *_ = factors[later_layer]
                    amplitude = _finite_saturating_norm_product(
                        amplitude,
                        later_operator,
                    )
                    amplitude_log = _log_norm_product(
                        amplitude_log,
                        later_operator,
                    )
                    energy = _finite_saturating_norm_sum(
                        energy,
                        _finite_saturating_norm_square(amplitude),
                    )
                    energy_log = _log_norm_sum(
                        energy_log,
                        _log_norm_square(amplitude_log),
                    )
                    amplitude = _finite_saturating_norm_product(
                        amplitude,
                        later_mlp,
                    )
                    amplitude_log = _log_norm_product(
                        amplitude_log,
                        later_mlp,
                    )
                    energy = _finite_saturating_norm_sum(
                        energy,
                        _finite_saturating_norm_square(amplitude),
                    )
                    energy_log = _log_norm_sum(
                        energy_log,
                        _log_norm_square(amplitude_log),
                    )
                if snapshot.terminal_present:
                    terminal_amplitude = _finite_saturating_norm_product(
                        amplitude,
                        terminal_transfer,
                    )
                    terminal_amplitude_log = _log_norm_product(
                        amplitude_log,
                        terminal_transfer,
                    )
                    energy = _finite_saturating_norm_sum(
                        energy,
                        _finite_saturating_norm_square(terminal_amplitude),
                    )
                    energy_log = _log_norm_sum(
                        energy_log,
                        _log_norm_square(terminal_amplitude_log),
                    )
                certified_blocks[block_index] = _finite_saturating_norm_sum(
                    certified_blocks[block_index],
                    energy,
                )
                certified_log_blocks[block_index] = _log_norm_sum(
                    certified_log_blocks[block_index],
                    energy_log,
                )
    conditioning_arithmetic = tuple(conditioning_blocks)
    certified_arithmetic = tuple(certified_blocks)
    conditioning_tuple = tuple(value.value for value in conditioning_arithmetic)
    certified_tuple = tuple(value.value for value in certified_arithmetic)
    certified_log_tuple = tuple(certified_log_blocks)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (*conditioning_tuple, *certified_tuple)
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware downstream bound overflowed and cannot be clipped."
        )
    site_order = tuple(
        f"layer.{layer_index}.{kind}"
        for layer_index in range(layer_count)
        for kind in (
            "operator.input_transport",
            "operator.output_writer",
            "mlp.input_transport",
            "mlp.down_writer",
        )
    )
    boundary_order = tuple(
        kind
        for _ in range(layer_count)
        for kind in ("post_operator_residual", "post_layer_residual")
    ) + (("terminal_observable",) if snapshot.terminal_present else ())
    conditioning_total = _finite_saturating_norm_fsum(conditioning_arithmetic)
    conditioning = Lfm2WholeStackConditioningScales(
        operator_squared_scale=conditioning_total.value,
        block_squared_scales=conditioning_tuple,
        site_order=site_order,
        method="producer_aware_local_direct_energy_without_downstream_product",
    )
    operator_log_evidence = _log_norm_fsum(certified_log_tuple)
    certified_total = _finite_saturating_norm_fsum(certified_arithmetic)
    finite_float_overflow_observed = certified_total.saturated or any(
        value.saturated for value in certified_arithmetic
    )
    certified = Lfm2ProducerAwareLinearizationNormEvidence(
        operator_squared_norm_upper_bound=certified_total.value,
        block_squared_norm_upper_bounds=certified_tuple,
        site_order=site_order,
        boundary_order=boundary_order,
        operator_log_squared_norm_evidence=operator_log_evidence,
        block_log_squared_norm_evidence=certified_log_tuple,
        finite_float_overflow_observed=finite_float_overflow_observed,
    )
    if not snapshot.verify_authentication():
        raise Lfm2ProducerAwareAuthenticationError(
            "snapshot changed during producer-aware norm analysis."
        )
    return conditioning, certified


def _validate_v2_bundle_inputs(
    state: Mapping[str, Any],
    coefficient_blocks: Sequence[Any],
    output_basis: Any,
) -> tuple[tuple[Any, ...], int, int, int, bool]:
    layer_count = int(state["stack.layer_count"].item())
    batch_count = int(state["stack.batch_count"].item())
    row_count = int(state["stack.row_count"].item())
    hidden = int(state["stack.hidden_size"].item())
    terminal_present = bool(int(state["stack.terminal_present"].item()))
    coefficients = tuple(coefficient_blocks)
    if (
        layer_count <= 0
        or len(coefficients) != 4 * layer_count
        or output_basis.ndim != 2
        or int(output_basis.shape[0]) != hidden
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware coefficients, basis, or stack geometry is invalid."
        )
    return coefficients, batch_count, row_count, hidden, terminal_present


def _producer_aware_stack_jvp_bundle_tensor_only(
    state: Mapping[str, Any],
    coefficient_blocks: Sequence[Any],
    output_basis: Any,
    *,
    maximum_workspace_bytes: int,
) -> tuple[Any, ...]:
    """One four-sites-per-layer sweep over every variable-length batch."""

    torch = _torch()
    coefficients, batch_count, row_count, hidden, terminal_present = (
        _validate_v2_bundle_inputs(state, coefficient_blocks, output_basis)
    )
    device = output_basis.device
    effects = tuple(
        torch.zeros((row_count, hidden), dtype=torch.float64, device="cpu")
        for _ in range(
            2 * int(state["stack.layer_count"].item()) + int(terminal_present)
        )
    )
    incoming = [
        torch.zeros_like(state[f"batch.{index:04d}.layer.000.operator.residual"])
        for index in range(batch_count)
    ]
    with torch.no_grad():
        layer_count = int(state["stack.layer_count"].item())
        for layer_index in range(layer_count):
            layer_parameters = _prefixed_state(
                state,
                f"layer.{layer_index:03d}.",
                device=device,
            )
            site_offset = 4 * layer_index
            for batch_index in range(batch_count):
                local = _stack_local_state(
                    state,
                    batch_index=batch_index,
                    layer_index=layer_index,
                    layer_parameters=layer_parameters,
                    device=device,
                )
                staged_incoming = incoming[batch_index].to(device=device).contiguous()
                if (
                    _workspace_bytes(
                        local,
                        staged_incoming,
                        output_basis,
                        *coefficients[site_offset : site_offset + 4],
                    )
                    > maximum_workspace_bytes
                ):
                    raise Lfm2ProducerAwareLinearizationError(
                        "A producer-aware JVP batch exceeds maximum_workspace_bytes."
                    )
                operator, mlp = _macrocell_views(local)
                if int(local["operator.kind"].item()) == 0:
                    operator_tangent = lfm2_producer_aware_attention_segment_jvp(
                        operator,
                        staged_incoming,
                        coefficients[site_offset],
                        coefficients[site_offset + 1],
                        output_basis,
                    )
                else:
                    operator_tangent = lfm2_producer_aware_short_conv_segment_jvp(
                        operator,
                        staged_incoming,
                        coefficients[site_offset],
                        coefficients[site_offset + 1],
                        output_basis,
                    )
                mlp_tangent = lfm2_producer_aware_mlp_segment_jvp(
                    mlp,
                    operator_tangent.output,
                    coefficients[site_offset + 2],
                    coefficients[site_offset + 3],
                    output_basis,
                )
                selected = state[f"batch.{batch_index:04d}.selected_target_positions"]
                canonical = state[
                    f"batch.{batch_index:04d}.canonical_boundary_row_indices"
                ]
                effects[2 * layer_index].index_copy_(
                    0,
                    canonical,
                    operator_tangent.output.detach()
                    .to(device="cpu")
                    .reshape(-1, hidden)
                    .index_select(0, selected),
                )
                effects[2 * layer_index + 1].index_copy_(
                    0,
                    canonical,
                    mlp_tangent.output.detach()
                    .to(device="cpu")
                    .reshape(-1, hidden)
                    .index_select(0, selected),
                )
                incoming[batch_index] = mlp_tangent.output.detach().to(device="cpu")
        if terminal_present:
            gain = state["terminal.norm_gain"].to(device=device).contiguous()
            epsilon = float(state["terminal.norm_epsilon"].item())
            for batch_index in range(batch_count):
                residual = state[f"batch.{batch_index:04d}.terminal.residual"].to(
                    device=device
                )
                terminal_effect = lfm2_continuous_fp64_rmsnorm_jvp(
                    residual,
                    incoming[batch_index].to(device=device),
                    gain,
                    epsilon,
                )
                selected = state[f"batch.{batch_index:04d}.selected_target_positions"]
                canonical = state[
                    f"batch.{batch_index:04d}.canonical_boundary_row_indices"
                ]
                effects[-1].index_copy_(
                    0,
                    canonical,
                    terminal_effect.detach()
                    .to(device="cpu")
                    .reshape(-1, hidden)
                    .index_select(0, selected),
                )
    return tuple(value.contiguous() for value in effects)


def _producer_aware_stack_vjp_bundle_tensor_only(
    state: Mapping[str, Any],
    boundary_cotangents_cpu: Sequence[Any],
    output_basis: Any,
    *,
    boundary_row_scales_cpu: Sequence[Any] | None,
    maximum_workspace_bytes: int,
) -> tuple[Any, ...]:
    """One exact four-sites-per-layer reverse sweep with boundary injections."""

    torch = _torch()
    layer_count = int(state["stack.layer_count"].item())
    batch_count = int(state["stack.batch_count"].item())
    row_count = int(state["stack.row_count"].item())
    hidden = int(state["stack.hidden_size"].item())
    terminal_present = bool(int(state["stack.terminal_present"].item()))
    if (
        layer_count <= 0
        or output_basis.ndim != 2
        or int(output_basis.shape[0]) != hidden
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware basis or authenticated stack geometry is invalid."
        )
    cotangents = tuple(boundary_cotangents_cpu)
    boundary_count = 2 * layer_count + int(terminal_present)
    if len(cotangents) != boundary_count or any(
        tuple(value.shape) != (row_count, hidden) for value in cotangents
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware VJP cotangent bundle has incorrect geometry."
        )
    scales = None if boundary_row_scales_cpu is None else tuple(boundary_row_scales_cpu)
    if scales is not None and (
        len(scales) != boundary_count
        or any(tuple(value.shape) != (row_count,) for value in scales)
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware VJP row-scale bundle has incorrect geometry."
        )
    device = output_basis.device
    rank = int(output_basis.shape[1])
    coefficient_bars: list[Any] = []
    for layer_index in range(layer_count):
        parameters = _prefixed_state(
            state,
            f"layer.{layer_index:03d}.",
            device=device,
        )
        operator_width = int(
            (
                parameters["attention.output_weight"]
                if int(parameters["operator.kind"].item()) == 0
                else parameters["short_conv.output_weight"]
            ).shape[1]
        )
        mlp_width = int(parameters["mlp.down_weight"].shape[1])
        for width in (hidden, operator_width, hidden, mlp_width):
            coefficient_bars.append(
                torch.zeros(
                    (rank, width),
                    dtype=torch.float64,
                    device=device,
                )
            )
    downstream = [
        torch.zeros_like(
            state[f"batch.{index:04d}.layer.{layer_count - 1:03d}.mlp.output"]
        )
        for index in range(batch_count)
    ]
    with torch.no_grad():
        if terminal_present:
            gain = state["terminal.norm_gain"].to(device=device).contiguous()
            epsilon = float(state["terminal.norm_epsilon"].item())
            for batch_index in range(batch_count):
                selected = state[f"batch.{batch_index:04d}.selected_target_positions"]
                canonical = state[
                    f"batch.{batch_index:04d}.canonical_boundary_row_indices"
                ]
                residual = state[f"batch.{batch_index:04d}.terminal.residual"]
                terminal_bar = _scatter_scaled_batch_cotangent(
                    cotangents[-1],
                    None if scales is None else scales[-1],
                    selected,
                    canonical,
                    batch_shape=tuple(residual.shape),
                    device=device,
                )
                downstream[batch_index] = lfm2_continuous_fp64_rmsnorm_vjp(
                    residual.to(device=device),
                    terminal_bar,
                    gain,
                    epsilon,
                ).to(device="cpu")
        for layer_index in range(layer_count - 1, -1, -1):
            layer_parameters = _prefixed_state(
                state,
                f"layer.{layer_index:03d}.",
                device=device,
            )
            site_offset = 4 * layer_index
            for batch_index in range(batch_count):
                local = _stack_local_state(
                    state,
                    batch_index=batch_index,
                    layer_index=layer_index,
                    layer_parameters=layer_parameters,
                    device=device,
                )
                if (
                    _workspace_bytes(
                        local,
                        downstream[batch_index],
                        output_basis,
                    )
                    > maximum_workspace_bytes
                ):
                    raise Lfm2ProducerAwareLinearizationError(
                        "A producer-aware VJP batch exceeds maximum_workspace_bytes."
                    )
                operator, mlp = _macrocell_views(local)
                selected = state[f"batch.{batch_index:04d}.selected_target_positions"]
                canonical = state[
                    f"batch.{batch_index:04d}.canonical_boundary_row_indices"
                ]
                operator_bar = _scatter_scaled_batch_cotangent(
                    cotangents[2 * layer_index],
                    None if scales is None else scales[2 * layer_index],
                    selected,
                    canonical,
                    batch_shape=tuple(operator.output.shape),
                    device=device,
                )
                layer_bar = _scatter_scaled_batch_cotangent(
                    cotangents[2 * layer_index + 1],
                    None if scales is None else scales[2 * layer_index + 1],
                    selected,
                    canonical,
                    batch_shape=tuple(mlp.output.shape),
                    device=device,
                ) + downstream[batch_index].to(device=device)
                mlp_result = lfm2_producer_aware_mlp_segment_vjp(
                    mlp,
                    layer_bar,
                    output_basis,
                )
                combined_operator_bar = operator_bar + mlp_result.incoming_residual
                if int(local["operator.kind"].item()) == 0:
                    operator_result = lfm2_producer_aware_attention_segment_vjp(
                        operator,
                        combined_operator_bar,
                        output_basis,
                    )
                else:
                    operator_result = lfm2_producer_aware_short_conv_segment_vjp(
                        operator,
                        combined_operator_bar,
                        output_basis,
                    )
                coefficient_bars[site_offset].add_(
                    operator_result.input_transport_coefficients
                )
                coefficient_bars[site_offset + 1].add_(
                    operator_result.output_writer_coefficients
                )
                coefficient_bars[site_offset + 2].add_(
                    mlp_result.input_transport_coefficients
                )
                coefficient_bars[site_offset + 3].add_(
                    mlp_result.output_writer_coefficients
                )
                downstream[batch_index] = operator_result.incoming_residual.to(
                    device="cpu"
                )
    return tuple(value.contiguous() for value in coefficient_bars)


class Lfm2ProducerAwareWholeStackLinearizationKernel(
    Lfm2GlobalResidualLinearizationKernel
):
    """Authenticated dynamic-layer producer-aware bundle kernel."""

    __slots__ = ()

    def __init__(
        self,
        topology: Lfm2GlobalResidualWriterTopology,
        boundaries: Lfm2GlobalResidualBoundaryBundle,
        snapshot: Lfm2WholeStackPrimalSnapshot,
        *,
        linearization_receipt: Any,
    ) -> None:
        if not isinstance(snapshot, Lfm2WholeStackPrimalSnapshot) or not (
            snapshot.verify_authentication()
        ):
            raise Lfm2ProducerAwareAuthenticationError(
                "snapshot must be an authenticated whole-stack primal bank."
            )
        if (
            not topology.verify_authentication()
            or not boundaries.verify_authentication()
        ):
            raise Lfm2ProducerAwareAuthenticationError(
                "Producer-aware topology and boundaries must be authenticated."
            )
        layer_count = snapshot.layer_count
        expected_site_count = 4 * layer_count
        if (
            topology.hidden_size != snapshot.hidden_size
            or len(topology.sites) != expected_site_count
        ):
            raise Lfm2ProducerAwareLinearizationError(
                "Producer-aware topology must contain exactly four sites per layer."
            )
        hidden = snapshot.hidden_size
        for layer_index in range(layer_count):
            macrocell = snapshot.batches[0].macrocells[layer_index]
            expected_operator_transport = (
                "attention_input_transport"
                if macrocell.operator_kind == "full_attention"
                else "short_conv_input_transport"
            )
            expected_operator_output = (
                "attention_output"
                if macrocell.operator_kind == "full_attention"
                else "short_conv_output"
            )
            sites = topology.sites[4 * layer_index : 4 * layer_index + 4]
            if (
                tuple(site.layer_index for site in sites) != (layer_index,) * 4
                or tuple(site.writer_kind for site in sites)
                != (
                    expected_operator_transport,
                    expected_operator_output,
                    "mlp_input_transport",
                    "mlp_down",
                )
                or tuple(site.input_width for site in sites)
                != (
                    hidden,
                    macrocell.input_widths[0],
                    hidden,
                    macrocell.input_widths[1],
                )
            ):
                raise Lfm2ProducerAwareLinearizationError(
                    f"Producer-aware topology differs at layer {layer_index}."
                )
        expected_boundary_count = 2 * layer_count + int(snapshot.terminal_present)
        if len(boundaries.boundaries) != expected_boundary_count:
            raise Lfm2ProducerAwareLinearizationError(
                "Producer-aware boundary inventory is incorrect."
            )
        for layer_index in range(layer_count):
            pair = boundaries.boundaries[2 * layer_index : 2 * layer_index + 2]
            sites = topology.sites[4 * layer_index : 4 * layer_index + 4]
            if (
                tuple(value.layer_index for value in pair) != (layer_index, layer_index)
                or tuple(value.boundary_kind for value in pair)
                != ("post_operator_residual", "post_layer_residual")
                or not (
                    sites[0].causal_order
                    < sites[1].causal_order
                    < pair[0].causal_order
                    < sites[2].causal_order
                    < sites[3].causal_order
                    < pair[1].causal_order
                )
            ):
                raise Lfm2ProducerAwareLinearizationError(
                    f"Producer-aware causal timeline differs at layer {layer_index}."
                )
        if snapshot.terminal_present and (
            boundaries.boundaries[-1].boundary_kind != "terminal_observable"
            or boundaries.boundaries[-1].layer_index != layer_count - 1
            or boundaries.boundaries[-1].causal_order
            <= boundaries.boundaries[-2].causal_order
        ):
            raise Lfm2ProducerAwareLinearizationError(
                "Terminal observable must follow the final layer residual."
            )
        if any(
            boundary.defect.numel() // hidden != snapshot.row_count
            or int(boundary.defect.shape[-1]) != hidden
            for boundary in boundaries.boundaries
        ):
            raise Lfm2ProducerAwareLinearizationError(
                "Every boundary must use the canonical donor-row archive."
            )
        state = _whole_stack_tensor_state(snapshot)
        state_sha256 = {
            name: fingerprint_lfm2_global_residual_tensor(value)
            for name, value in state.items()
        }
        super().__init__(
            topology,
            boundaries,
            tensor_state=state,
            tensor_state_sha256=state_sha256,
            linearization_receipt=linearization_receipt,
            transpose_contract_body={
                "kernel": "lfm2_continuous_fp64_producer_aware_stack_v2a",
                "layers": layer_count,
                "sites": expected_site_count,
                "site_order_per_layer": [
                    "operator_input_transport",
                    "operator_output_writer",
                    "mlp_input_transport",
                    "mlp_down_writer",
                ],
                "input_transport": "A=output_basis@coefficient_block",
                "transport_effect": "normalized_input@A.T",
                "attention_absorption": ["Wq@A", "Wk@A", "Wv@A"],
                "short_conv_absorption": "Win@A_all_BCX_rows",
                "mlp_absorption": ["Wgate@A", "Wup@A"],
                "inter_layer_runtime_cast": "straight_through_identity",
                "token_bridge": "authenticated_gather_scatter",
                "bundle_fast_path": True,
                "mutable_bundle_cache": False,
            },
        )

    def _jvp_bundle_tensor_only(
        self,
        state: Mapping[str, Any],
        coefficient_blocks: tuple[Any, ...],
        output_basis: Any,
        *,
        maximum_workspace_bytes: int,
    ) -> tuple[Any, ...]:
        return _producer_aware_stack_jvp_bundle_tensor_only(
            state,
            coefficient_blocks,
            output_basis,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )

    def _vjp_bundle_tensor_only(
        self,
        state: Mapping[str, Any],
        boundary_cotangents_cpu: tuple[Any, ...],
        output_basis: Any,
        *,
        boundary_row_scales_cpu: tuple[Any, ...] | None,
        maximum_workspace_bytes: int,
    ) -> tuple[Any, ...]:
        return _producer_aware_stack_vjp_bundle_tensor_only(
            state,
            boundary_cotangents_cpu,
            output_basis,
            boundary_row_scales_cpu=boundary_row_scales_cpu,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )

    def _jvp_chunk_tensor_only(
        self,
        state: Mapping[str, Any],
        coefficient_blocks: tuple[Any, ...],
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> Any:
        bundle = _producer_aware_stack_jvp_bundle_tensor_only(
            state,
            coefficient_blocks,
            output_basis,
            maximum_workspace_bytes=2**63 - 1,
        )
        return bundle[boundary_index][row_start:row_stop].to(device=output_basis.device)

    def _vjp_chunk_tensor_only(
        self,
        state: Mapping[str, Any],
        boundary_cotangent: Any,
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> tuple[Any, ...]:
        torch = _torch()
        row_count = int(state["stack.row_count"].item())
        hidden = int(state["stack.hidden_size"].item())
        boundary_count = 2 * int(state["stack.layer_count"].item()) + int(
            state["stack.terminal_present"].item()
        )
        cotangents = [
            torch.zeros((row_count, hidden), dtype=torch.float64, device="cpu")
            for _ in range(boundary_count)
        ]
        cotangents[boundary_index][row_start:row_stop].copy_(
            boundary_cotangent.to(device="cpu")
        )
        return _producer_aware_stack_vjp_bundle_tensor_only(
            state,
            tuple(cotangents),
            output_basis,
            boundary_row_scales_cpu=None,
            maximum_workspace_bytes=2**63 - 1,
        )


_ATTENTION_SEED_ROLES = (
    "query_weight",
    "key_weight",
    "value_weight",
    "operator_output_weight",
    "mlp_gate_weight",
    "mlp_up_weight",
    "mlp_down_weight",
)
_SHORT_CONV_SEED_ROLES = (
    "operator_input_weight",
    "operator_output_weight",
    "mlp_gate_weight",
    "mlp_up_weight",
    "mlp_down_weight",
)


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware evidence must be canonical finite JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _seal_report(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "report_sha256": _canonical_sha256(detached)})


def _verify_report(report: Mapping[str, Any]) -> bool:
    if not isinstance(report, Mapping):
        return False
    try:
        body = dict(report)
        supplied = body.pop("report_sha256", None)
        return isinstance(supplied, str) and supplied == _canonical_sha256(body)
    except (TypeError, ValueError):
        return False


_LFM2_PRODUCER_AWARE_DENSE_COMPUTE_SCHEMA = (
    "neural-forge-lfm2-producer-aware-dense-absorption-compute-v1"
)


def _resolve_dense_absorption_compute_device(torch: Any, compute_device: Any) -> Any:
    """Resolve one canonical fold-only algebra device without fallback."""

    if compute_device == "cpu":
        return torch.device("cpu")
    if not isinstance(compute_device, str):
        raise Lfm2ProducerAwareLinearizationError(
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
        raise Lfm2ProducerAwareLinearizationError(
            "CUDA compute_device must be explicitly indexed as 'cuda:N'."
        )
    index = int(index_text)
    if not torch.cuda.is_available() or index >= int(torch.cuda.device_count()):
        raise Lfm2ProducerAwareLinearizationError(
            f"CUDA compute_device index {index} is unavailable."
        )
    return torch.device("cuda", index)


def _dense_absorption_compute_receipt(
    torch: Any,
    requested: str,
    resolved: Any,
) -> Mapping[str, Any]:
    cuda_identity = None
    if resolved.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved)
        cuda_identity = {
            "device_index": int(resolved.index),
            "device_name": str(properties.name),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "total_memory_bytes": int(properties.total_memory),
        }
    return _seal_report(
        {
            "schema": _LFM2_PRODUCER_AWARE_DENSE_COMPUTE_SCHEMA,
            "kind": "producer_aware_dense_absorption_compute_execution",
            "requested_device": requested,
            "resolved_device": str(resolved),
            "backend": resolved.type,
            "compute_dtype": "float64",
            "archive_devices": {
                "layer_seed_weights": "cpu",
                "output_basis": "cpu",
                "coefficient_blocks": "cpu",
            },
            "output_devices": {
                "transaction_seed_weights": "cpu",
                "dense_corrected_weights": "cpu",
            },
            "output_basis_staged_to_resolved_device": True,
            "coefficient_blocks_staged_to_resolved_device": True,
            "active_layer_seed_weights_staged_to_resolved_device": True,
            "active_layers_staged_one_at_a_time": True,
            "gpu_is_ephemeral_dense_algebra_workspace": resolved.type == "cuda",
            "system_memory_is_immutable_archive": True,
            "silent_device_fallback_performed": False,
            "cuda_identity": cuda_identity,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }
    )


def _verify_dense_absorption_compute_receipt(receipt: Any) -> bool:
    if not isinstance(receipt, Mapping) or not _verify_report(receipt):
        return False
    expected_archives = {
        "layer_seed_weights": "cpu",
        "output_basis": "cpu",
        "coefficient_blocks": "cpu",
    }
    expected_outputs = {
        "transaction_seed_weights": "cpu",
        "dense_corrected_weights": "cpu",
    }
    requested = receipt.get("requested_device")
    resolved = receipt.get("resolved_device")
    backend = receipt.get("backend")
    cuda_identity = receipt.get("cuda_identity")
    counters = (
        "autograd_grad_calls",
        "backward_calls",
        "optimizer_steps",
        "training_steps",
        "parameters_pruned",
        "parameters_quantized",
    )
    if (
        receipt.get("schema") != _LFM2_PRODUCER_AWARE_DENSE_COMPUTE_SCHEMA
        or receipt.get("kind") != "producer_aware_dense_absorption_compute_execution"
        or receipt.get("compute_dtype") != "float64"
        or receipt.get("archive_devices") != expected_archives
        or receipt.get("output_devices") != expected_outputs
        or receipt.get("output_basis_staged_to_resolved_device") is not True
        or receipt.get("coefficient_blocks_staged_to_resolved_device") is not True
        or receipt.get("active_layer_seed_weights_staged_to_resolved_device")
        is not True
        or receipt.get("active_layers_staged_one_at_a_time") is not True
        or receipt.get("system_memory_is_immutable_archive") is not True
        or receipt.get("silent_device_fallback_performed") is not False
        or any(receipt.get(counter) != 0 for counter in counters)
    ):
        return False
    if requested == "cpu":
        return (
            resolved == "cpu"
            and backend == "cpu"
            and cuda_identity is None
            and receipt.get("gpu_is_ephemeral_dense_algebra_workspace") is False
        )
    if not isinstance(requested, str):
        return False
    prefix, separator, index_text = requested.partition(":")
    if (
        prefix != "cuda"
        or separator != ":"
        or not index_text.isascii()
        or not index_text.isdecimal()
        or index_text != str(int(index_text))
        or resolved != requested
        or backend != "cuda"
        or receipt.get("gpu_is_ephemeral_dense_algebra_workspace") is not True
        or not isinstance(cuda_identity, Mapping)
    ):
        return False
    capability = cuda_identity.get("compute_capability")
    return (
        cuda_identity.get("device_index") == int(index_text)
        and isinstance(cuda_identity.get("device_name"), str)
        and bool(cuda_identity.get("device_name"))
        and isinstance(capability, list)
        and len(capability) == 2
        and all(isinstance(value, int) and value >= 0 for value in capability)
        and isinstance(cuda_identity.get("total_memory_bytes"), int)
        and cuda_identity.get("total_memory_bytes") > 0
    )


def _dense_weight(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise Lfm2ProducerAwareLinearizationError(
            f"{name} must be one finite nonempty dense real matrix."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _storage_identity(value: Any) -> tuple[str, int | None, int]:
    return (
        value.device.type,
        value.device.index,
        int(value.untyped_storage().data_ptr()),
    )


@dataclass(frozen=True, slots=True)
class Lfm2ProducerAwareLayerSeedWeights:
    """Immutable physical seed matrices and names for one child layer."""

    layer_index: int
    operator_kind: Literal["full_attention", "short_conv"]
    parameter_ids: Mapping[str, str]
    seed_weights: Mapping[str, Any] = field(repr=False, compare=False)
    authentication_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2ProducerAwareLinearizationError(
                "layer_index must be a nonnegative integer."
            )
        kind = str(self.operator_kind).strip().lower()
        if kind not in {"full_attention", "short_conv"}:
            raise Lfm2ProducerAwareLinearizationError(
                "operator_kind must be full_attention or short_conv."
            )
        roles = (
            _ATTENTION_SEED_ROLES
            if kind == "full_attention"
            else _SHORT_CONV_SEED_ROLES
        )
        ids = {
            str(role): str(value).strip() for role, value in self.parameter_ids.items()
        }
        sources = dict(self.seed_weights)
        if set(ids) != set(roles) or set(sources) != set(roles):
            raise Lfm2ProducerAwareLinearizationError(
                "Physical parameter ids and seed weights must exactly cover the "
                "operator and MLP absorption roles."
            )
        if any(not value for value in ids.values()) or len(set(ids.values())) != len(
            ids
        ):
            raise Lfm2ProducerAwareLinearizationError(
                "Physical parameter ids must be nonempty and independently addressable."
            )
        weights = {role: _dense_weight(sources[role], role) for role in roles}
        source_identities = [_storage_identity(sources[role]) for role in roles]
        if len(set(source_identities)) != len(source_identities):
            raise Lfm2ProducerAwareLinearizationError(
                "Seed matrices cannot share tensor storage."
            )
        hidden = int(weights["mlp_down_weight"].shape[0])
        intermediate = int(weights["mlp_down_weight"].shape[1])
        if tuple(weights["mlp_gate_weight"].shape) != (intermediate, hidden) or tuple(
            weights["mlp_up_weight"].shape
        ) != (intermediate, hidden):
            raise Lfm2ProducerAwareLinearizationError(
                "MLP gate/up/down seed geometry is inconsistent."
            )
        if kind == "full_attention":
            query_width = int(weights["query_weight"].shape[0])
            if (
                int(weights["query_weight"].shape[1]) != hidden
                or int(weights["key_weight"].shape[1]) != hidden
                or int(weights["value_weight"].shape[1]) != hidden
                or tuple(weights["key_weight"].shape)
                != tuple(weights["value_weight"].shape)
                or tuple(weights["operator_output_weight"].shape)
                != (hidden, query_width)
            ):
                raise Lfm2ProducerAwareLinearizationError(
                    "Attention Q/K/V/O seed geometry is inconsistent."
                )
        elif tuple(weights["operator_input_weight"].shape) != (
            3 * hidden,
            hidden,
        ) or tuple(weights["operator_output_weight"].shape) != (hidden, hidden):
            raise Lfm2ProducerAwareLinearizationError(
                "Short-conv input/output seed geometry is inconsistent."
            )
        object.__setattr__(self, "operator_kind", kind)
        object.__setattr__(self, "parameter_ids", MappingProxyType(ids))
        object.__setattr__(self, "seed_weights", MappingProxyType(weights))
        object.__setattr__(
            self,
            "authentication_sha256",
            self.recompute_authentication_sha256(),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.seed_weights["mlp_down_weight"].shape[0])

    @property
    def operator_output_width(self) -> int:
        return int(self.seed_weights["operator_output_weight"].shape[1])

    @property
    def mlp_output_width(self) -> int:
        return int(self.seed_weights["mlp_down_weight"].shape[1])

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": LFM2_PRODUCER_AWARE_LINEARIZATION_SCHEMA,
                "kind": "layer_seed_weights",
                "layer_index": self.layer_index,
                "operator_kind": self.operator_kind,
                "parameter_ids": dict(self.parameter_ids),
                "seed_sha256": {
                    role: fingerprint_lfm2_global_residual_tensor(value)
                    for role, value in sorted(self.seed_weights.items())
                },
            }
        )

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == self.recompute_authentication_sha256()


def capture_lfm2_producer_aware_layer_seed(
    layer_index: int,
    macrocell: Lfm2MacrocellLinearization,
    *,
    parameter_ids: Mapping[str, str],
) -> Lfm2ProducerAwareLayerSeedWeights:
    """Capture physical absorption seeds from one authenticated macrocell."""

    if not isinstance(macrocell, Lfm2MacrocellLinearization) or not (
        macrocell.verify_authentication()
    ):
        raise Lfm2ProducerAwareAuthenticationError(
            "macrocell must be an authenticated primal linearization."
        )
    p = macrocell.operator.parameters
    mp = macrocell.mlp.parameters
    if macrocell.operator_kind == "full_attention":
        weights = {
            "query_weight": p.query_weight,
            "key_weight": p.key_weight,
            "value_weight": p.value_weight,
            "operator_output_weight": p.output_weight,
            "mlp_gate_weight": mp.gate_weight,
            "mlp_up_weight": mp.up_weight,
            "mlp_down_weight": mp.down_weight,
        }
    else:
        weights = {
            "operator_input_weight": p.input_weight,
            "operator_output_weight": p.output_weight,
            "mlp_gate_weight": mp.gate_weight,
            "mlp_up_weight": mp.up_weight,
            "mlp_down_weight": mp.down_weight,
        }
    return Lfm2ProducerAwareLayerSeedWeights(
        layer_index=layer_index,
        operator_kind=macrocell.operator_kind,
        parameter_ids=parameter_ids,
        seed_weights=weights,
    )


@dataclass(frozen=True, slots=True)
class Lfm2ProducerAwareDenseTransaction:
    """Authenticated independently owned physical updates for one stack."""

    parameter_ids: tuple[str, ...]
    layer_indices: tuple[int, ...]
    roles: tuple[str, ...]
    seed_weights: tuple[Any, ...] = field(repr=False, compare=False)
    corrected_weights: tuple[Any, ...] = field(repr=False, compare=False)
    atomic_groups: tuple[tuple[str, ...], ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        parameter_ids = tuple(self.parameter_ids)
        layer_indices = tuple(self.layer_indices)
        roles = tuple(self.roles)
        seeds = tuple(
            _dense_weight(value, "transaction.seed") for value in self.seed_weights
        )
        corrected = tuple(
            _dense_weight(value, "transaction.corrected")
            for value in self.corrected_weights
        )
        count = len(parameter_ids)
        if (
            count == 0
            or len(set(parameter_ids)) != count
            or any(
                len(value) != count
                for value in (layer_indices, roles, seeds, corrected)
            )
            or any(
                tuple(seed.shape) != tuple(value.shape) or seed.dtype != value.dtype
                for seed, value in zip(seeds, corrected, strict=True)
            )
        ):
            raise Lfm2ProducerAwareLinearizationError(
                "Dense transaction inventory or geometry is invalid."
            )
        groups = tuple(tuple(group) for group in self.atomic_groups)
        flattened_groups = tuple(value for group in groups for value in group)
        if (
            set(flattened_groups) != set(parameter_ids)
            or len(flattened_groups) != count
        ):
            raise Lfm2ProducerAwareLinearizationError(
                "Atomic groups must partition every physical parameter exactly once."
            )
        object.__setattr__(self, "parameter_ids", parameter_ids)
        object.__setattr__(self, "layer_indices", layer_indices)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "seed_weights", seeds)
        object.__setattr__(self, "corrected_weights", corrected)
        object.__setattr__(self, "atomic_groups", groups)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))
        if not self.verify_authentication():
            raise Lfm2ProducerAwareAuthenticationError(
                "Dense transaction report does not authenticate its tensors."
            )

    def verify_authentication(self) -> bool:
        compute_receipt = self.report.get("compute_execution_receipt")
        representability = self.report.get("target_dtype_representability")
        expected_changed_parameter_ids = [
            parameter_id
            for parameter_id, seed, corrected in zip(
                self.parameter_ids,
                self.seed_weights,
                self.corrected_weights,
                strict=True,
            )
            if fingerprint_lfm2_global_residual_tensor(seed)
            != fingerprint_lfm2_global_residual_tensor(corrected)
        ]
        return (
            _verify_report(self.report)
            and _verify_dense_absorption_compute_receipt(compute_receipt)
            and _verify_report(representability)
            and self.report.get("compute_execution_receipt_sha256")
            == compute_receipt.get("report_sha256")
            and self.report.get("target_dtype_representability_sha256")
            == representability.get("report_sha256")
            and self.report.get("target_dtype_representability_level")
            == representability.get("selected_level")
            and self.report.get("target_dtype_representability_ray_multiplier")
            == representability.get("selected_ray_multiplier")
            and self.report.get("target_dtype_representability_frontier")
            == representability.get("frontier")
            and self.report.get("target_dtype_changed_parameter_ids")
            == expected_changed_parameter_ids
            and self.report.get("parameter_ids") == list(self.parameter_ids)
            and self.report.get("layer_indices") == list(self.layer_indices)
            and self.report.get("roles") == list(self.roles)
            and self.report.get("seed_sha256")
            == [
                fingerprint_lfm2_global_residual_tensor(value)
                for value in self.seed_weights
            ]
            and self.report.get("corrected_sha256")
            == [
                fingerprint_lfm2_global_residual_tensor(value)
                for value in self.corrected_weights
            ]
            and self.report.get("atomic_groups")
            == [list(group) for group in self.atomic_groups]
        )


def materialize_lfm2_producer_aware_stack_transaction(
    layer_seeds: Sequence[Lfm2ProducerAwareLayerSeedWeights],
    output_basis: Any,
    coefficient_blocks: Sequence[Any],
    *,
    compute_device: str = "cpu",
    required_effect_parameter_ids: Sequence[str] | None = None,
    target_dtype_representability_level: str = "first_effectful",
) -> Lfm2ProducerAwareDenseTransaction:
    """Absorb every transport/writer into representable dense stack weights.

    The closure is solved in continuous FP64 coordinates, while the child may be
    materialized in BF16 or FP32.  A nonzero analytic correction can therefore
    disappear when ``seed + delta`` is rounded to the target dtype.  This routine
    measures that target-dtype geometry.  ``requested`` preserves the requested
    unit-scale correction and permits an honestly reported target-dtype no-op;
    the legacy effect/mass levels apply one global positive multiplier to the
    complete analytic correction ray.  It never injects coordinate-wise
    ``nextafter`` noise: adjacent target values are used only to measure
    deterministic rounding breakpoints.
    """

    torch = _torch()
    resolved_compute_device = _resolve_dense_absorption_compute_device(
        torch,
        compute_device,
    )
    compute_receipt = _dense_absorption_compute_receipt(
        torch,
        compute_device,
        resolved_compute_device,
    )
    layers = tuple(layer_seeds)
    if (
        not layers
        or tuple(value.layer_index for value in layers) != tuple(range(len(layers)))
        or not all(value.verify_authentication() for value in layers)
    ):
        raise Lfm2ProducerAwareAuthenticationError(
            "layer_seeds must be a nonempty authenticated contiguous stack."
        )
    hidden = layers[0].hidden_size
    if any(value.hidden_size != hidden for value in layers):
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware layer seeds disagree on hidden width."
        )
    basis = _plain_fp64(output_basis, "output_basis", device=output_basis.device)
    if basis.ndim != 2 or tuple(basis.shape)[0] != hidden:
        raise Lfm2ProducerAwareLinearizationError(
            "output_basis must have shape [hidden, rank]."
        )
    coefficients = tuple(coefficient_blocks)
    expected_site_count = 4 * len(layers)
    if len(coefficients) != expected_site_count:
        raise Lfm2ProducerAwareLinearizationError(
            "Producer-aware materialization requires exactly four blocks per layer."
        )
    rank = int(basis.shape[1])
    basis_archive = (
        basis.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    )
    coefficient_archives = []
    for layer in layers:
        site_offset = 4 * layer.layer_index
        widths = (
            hidden,
            layer.operator_output_width,
            hidden,
            layer.mlp_output_width,
        )
        for local_index, width in enumerate(widths):
            coefficient_archives.append(
                _validate_coefficient(
                    coefficients[site_offset + local_index],
                    rank=rank,
                    input_width=width,
                    device=basis.device,
                    name=f"coefficient_blocks[{site_offset + local_index}]",
                )
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
                .clone()
            )
    coefficient_archives = tuple(coefficient_archives)
    basis_compute = basis_archive.to(
        device=resolved_compute_device,
        dtype=torch.float64,
    )
    parameter_ids: list[str] = []
    layer_indices: list[int] = []
    roles: list[str] = []
    seeds: list[Any] = []
    groups: list[tuple[str, ...]] = []

    for layer in layers:
        if layer.operator_kind == "full_attention":
            operator_roles = ("query_weight", "key_weight", "value_weight")
            groups.append(
                tuple(layer.parameter_ids[role] for role in operator_roles)
            )
        else:
            operator_roles = ("operator_input_weight",)
            groups.append((layer.parameter_ids["operator_input_weight"],))
        groups.append((layer.parameter_ids["operator_output_weight"],))
        groups.append(
            tuple(
                layer.parameter_ids[role]
                for role in ("mlp_gate_weight", "mlp_up_weight")
            )
        )
        groups.append((layer.parameter_ids["mlp_down_weight"],))
        for role in (
            *operator_roles,
            "operator_output_weight",
            "mlp_gate_weight",
            "mlp_up_weight",
            "mlp_down_weight",
        ):
            parameter_ids.append(layer.parameter_ids[role])
            layer_indices.append(layer.layer_index)
            roles.append(role)
            seeds.append(layer.seed_weights[role])
    if len(set(parameter_ids)) != len(parameter_ids):
        raise Lfm2ProducerAwareLinearizationError(
            "Physical parameter ids cannot be shared across layers or roles."
        )
    selected_level = str(target_dtype_representability_level).strip().lower()
    if selected_level not in _TARGET_DTYPE_REPRESENTABILITY_LEVELS:
        raise Lfm2ProducerAwareLinearizationError(
            "target_dtype_representability_level must be requested, "
            "first_effectful, or one of the deterministic coordinate-mass levels."
        )
    if required_effect_parameter_ids is None:
        required_ids = tuple(parameter_ids)
    else:
        required_ids = tuple(
            str(value).strip() for value in required_effect_parameter_ids
        )
        if (
            not required_ids
            or any(not value for value in required_ids)
            or len(set(required_ids)) != len(required_ids)
            or not set(required_ids).issubset(set(parameter_ids))
        ):
            raise Lfm2ProducerAwareLinearizationError(
                "required_effect_parameter_ids must be a nonempty unique subset "
                "of the dense transaction inventory."
            )
    required_id_set = set(required_ids)

    def dense_delta_stream():
        with torch.no_grad():
            for layer in layers:
                site_offset = 4 * layer.layer_index
                blocks = tuple(
                    value.to(device=resolved_compute_device, dtype=torch.float64)
                    for value in coefficient_archives[site_offset : site_offset + 4]
                )

                def right_absorption_delta(
                    role: str,
                    coefficients_for_role: Any,
                ) -> Any:
                    active_weight = layer.seed_weights[role].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    )
                    return (active_weight @ basis_compute) @ coefficients_for_role

                if layer.operator_kind == "full_attention":
                    for role in ("query_weight", "key_weight", "value_weight"):
                        yield (
                            layer,
                            role,
                            right_absorption_delta(role, blocks[0])
                            .detach()
                            .to(device="cpu", dtype=torch.float64)
                            .contiguous(),
                        )
                else:
                    role = "operator_input_weight"
                    yield (
                        layer,
                        role,
                        right_absorption_delta(role, blocks[0])
                        .detach()
                        .to(device="cpu", dtype=torch.float64)
                        .contiguous(),
                    )
                yield (
                    layer,
                    "operator_output_weight",
                    (basis_compute @ blocks[1])
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .contiguous(),
                )
                for role in ("mlp_gate_weight", "mlp_up_weight"):
                    yield (
                        layer,
                        role,
                        right_absorption_delta(role, blocks[2])
                        .detach()
                        .to(device="cpu", dtype=torch.float64)
                        .contiguous(),
                    )
                yield (
                    layer,
                    "mlp_down_weight",
                    (basis_compute @ blocks[3])
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .contiguous(),
                )

    requested_delta_records: list[dict[str, Any]] = []
    breakpoint_histogram: dict[int, int] = {}
    all_breakpoint_histogram: dict[int, int] = {}
    first_breakpoint: float | None = None
    first_breakpoint_witness: dict[str, Any] | None = None
    first_all_breakpoint: float | None = None
    first_all_breakpoint_witness: dict[str, Any] | None = None
    all_nonzero_coordinate_count = 0
    required_nonzero_coordinate_count = 0
    finite_required_breakpoint_count = 0
    unavailable_required_breakpoint_count = 0
    finite_all_breakpoint_count = 0
    unavailable_all_breakpoint_count = 0
    requested_changed_coordinate_count = 0
    requested_changed_required_coordinate_count = 0
    requested_changed_tensor_count = 0
    requested_changed_required_tensor_count = 0
    requested_norm = 0.0
    requested_norm_saturated = False

    with torch.no_grad():
        for expected_id, (layer, role, delta) in zip(
            parameter_ids,
            dense_delta_stream(),
            strict=True,
        ):
            parameter_id = layer.parameter_ids[role]
            seed = layer.seed_weights[role]
            if parameter_id != expected_id or tuple(delta.shape) != tuple(seed.shape):
                raise Lfm2ProducerAwareLinearizationError(
                    "Dense correction stream changed physical order or geometry."
                )
            if not bool(torch.isfinite(delta).all().item()):
                raise Lfm2ProducerAwareLinearizationError(
                    f"Analytic dense correction for {parameter_id} is non-finite."
                )
            nonzero_mask = delta != 0
            nonzero_count = int(torch.count_nonzero(nonzero_mask).item())
            all_nonzero_coordinate_count += nonzero_count
            required = parameter_id in required_id_set
            if required:
                required_nonzero_coordinate_count += nonzero_count
            tensor_norm = float(torch.linalg.vector_norm(delta).item())
            if math.isfinite(tensor_norm):
                requested_norm = math.hypot(requested_norm, tensor_norm)
                if not math.isfinite(requested_norm):
                    requested_norm = _MAX_FINITE_NORM_EVIDENCE
                    requested_norm_saturated = True
            else:
                requested_norm = _MAX_FINITE_NORM_EVIDENCE
                requested_norm_saturated = True

            seed_fp64 = seed.to(dtype=torch.float64)
            requested_cast = (seed_fp64 + delta).to(dtype=seed.dtype)
            requested_cast = torch.where(nonzero_mask, requested_cast, seed)
            requested_finite = bool(torch.isfinite(requested_cast).all().item())
            changed_count = int(torch.count_nonzero(requested_cast != seed).item())
            requested_changed_coordinate_count += changed_count
            requested_changed_tensor_count += int(changed_count > 0)
            if required:
                requested_changed_required_coordinate_count += changed_count
                requested_changed_required_tensor_count += int(changed_count > 0)

            requested_delta_records.append(
                {
                    "parameter_id": parameter_id,
                    "layer_index": layer.layer_index,
                    "role": role,
                    "target_dtype": str(seed.dtype).removeprefix("torch."),
                    "coordinate_count": int(seed.numel()),
                    "requested_analytic_nonzero_coordinate_count": nonzero_count,
                    "requested_analytic_delta_sha256": (
                        fingerprint_lfm2_global_residual_tensor(delta)
                    ),
                    "requested_analytic_delta_l2_norm": (
                        tensor_norm
                        if math.isfinite(tensor_norm)
                        else _MAX_FINITE_NORM_EVIDENCE
                    ),
                    "requested_target_dtype_changed_coordinate_count": changed_count,
                    "requested_target_dtype_finite": requested_finite,
                    "required_effect_scope": required,
                }
            )

            if nonzero_count == 0:
                continue
            seed_flat = seed.reshape(-1)
            delta_flat = delta.reshape(-1)
            for start in range(
                0,
                int(delta_flat.numel()),
                _TARGET_DTYPE_REPRESENTABILITY_CHUNK_COORDINATES,
            ):
                stop = min(
                    int(delta_flat.numel()),
                    start + _TARGET_DTYPE_REPRESENTABILITY_CHUNK_COORDINATES,
                )
                seed_chunk = seed_flat[start:stop]
                delta_chunk = delta_flat[start:stop]
                active = delta_chunk != 0
                if not bool(active.any().item()):
                    continue
                direction = torch.where(
                    delta_chunk > 0,
                    torch.full_like(seed_chunk, math.inf),
                    torch.full_like(seed_chunk, -math.inf),
                )
                adjacent = torch.nextafter(seed_chunk, direction)
                spacing = (adjacent.to(torch.float64) - seed_chunk.to(torch.float64)).abs()
                active_indices = torch.nonzero(active, as_tuple=False).reshape(-1)
                active_spacing = spacing[active_indices]
                active_delta = delta_chunk[active_indices].abs()
                breakpoints = (active_spacing / active_delta) * 0.5
                finite = torch.isfinite(breakpoints)
                finite_count = int(torch.count_nonzero(finite).item())
                unavailable_count = int(active_indices.numel()) - finite_count
                finite_all_breakpoint_count += finite_count
                unavailable_all_breakpoint_count += unavailable_count
                if required:
                    finite_required_breakpoint_count += finite_count
                    unavailable_required_breakpoint_count += unavailable_count
                if finite_count == 0:
                    continue
                finite_values = breakpoints[finite]
                finite_positions = active_indices[finite]
                histogram_exponents = torch.where(
                    finite_values > 0,
                    torch.ceil(torch.log2(finite_values)),
                    torch.full_like(finite_values, -1075.0),
                ).to(torch.int64)
                unique_exponents, exponent_counts = torch.unique(
                    histogram_exponents,
                    return_counts=True,
                )
                for exponent, count in zip(
                    unique_exponents.tolist(),
                    exponent_counts.tolist(),
                    strict=True,
                ):
                    exponent_value = int(exponent)
                    count_value = int(count)
                    all_breakpoint_histogram[exponent_value] = (
                        all_breakpoint_histogram.get(exponent_value, 0) + count_value
                    )
                    if required:
                        breakpoint_histogram[exponent_value] = (
                            breakpoint_histogram.get(exponent_value, 0) + count_value
                        )
                local_value, local_index = torch.min(finite_values, dim=0)
                candidate = float(local_value.item())
                coordinate = int(finite_positions[int(local_index.item())].item())
                adjacent_value = adjacent[coordinate]
                witness = {
                    "parameter_id": parameter_id,
                    "layer_index": layer.layer_index,
                    "role": role,
                    "flat_coordinate_index": start + coordinate,
                    "target_dtype": str(seed.dtype).removeprefix("torch."),
                    "seed_value": float(seed_chunk[coordinate].item()),
                    "directional_adjacent_value": float(adjacent_value.item()),
                    "directional_ulp": float(spacing[coordinate].item()),
                    "requested_analytic_delta_value": float(
                        delta_chunk[coordinate].item()
                    ),
                    "first_change_midpoint_breakpoint": candidate,
                    "required_effect_scope": required,
                }
                if first_all_breakpoint is None or candidate < first_all_breakpoint:
                    first_all_breakpoint = candidate
                    first_all_breakpoint_witness = witness
                if required and (
                    first_breakpoint is None or candidate < first_breakpoint
                ):
                    first_breakpoint = candidate
                    first_breakpoint_witness = witness

    effect_scope = (
        "required_live_parameters"
        if required_nonzero_coordinate_count > 0
        else "complete_dense_closure"
    )
    effect_nonzero_coordinate_count = (
        required_nonzero_coordinate_count
        if required_nonzero_coordinate_count > 0
        else all_nonzero_coordinate_count
    )
    requested_changed_effect_coordinate_count = (
        requested_changed_required_coordinate_count
        if required_nonzero_coordinate_count > 0
        else requested_changed_coordinate_count
    )
    effect_first_breakpoint = (
        first_breakpoint
        if required_nonzero_coordinate_count > 0
        else first_all_breakpoint
    )
    effect_first_breakpoint_witness = (
        first_breakpoint_witness
        if required_nonzero_coordinate_count > 0
        else first_all_breakpoint_witness
    )
    effect_breakpoint_histogram = (
        breakpoint_histogram
        if required_nonzero_coordinate_count > 0
        else all_breakpoint_histogram
    )

    def strictly_above(value: float) -> float | None:
        if not math.isfinite(value):
            return None
        candidate = math.nextafter(float(value), math.inf)
        return candidate if math.isfinite(candidate) else None

    first_effectful_multiplier: float | None
    if effect_nonzero_coordinate_count == 0:
        first_effectful_multiplier = 1.0
    elif requested_changed_effect_coordinate_count > 0:
        first_effectful_multiplier = 1.0
    elif effect_first_breakpoint is None:
        first_effectful_multiplier = None
    else:
        # The exact midpoint is a round-to-nearest tie, and an FP64 scalar one
        # ULP above the ratio can still collapse back to that midpoint during
        # the multiply/add.  One complete directional target-dtype ULP is the
        # smallest closed-form tie-independent witness: it does not depend on
        # the target mantissa's even/odd endpoint and remains on the same ray.
        first_effectful_multiplier = strictly_above(
            max(1.0, 2.0 * effect_first_breakpoint)
        )

    sorted_histogram = tuple(sorted(effect_breakpoint_histogram.items()))

    def mass_level_multiplier(target_count: int) -> float | None:
        cumulative = 0
        selected_exponent = None
        for exponent, count in sorted_histogram:
            cumulative += count
            if cumulative >= target_count:
                selected_exponent = exponent
                break
        if selected_exponent is None or selected_exponent >= 1024:
            return None
        raw = 0.0 if selected_exponent < -1074 else math.ldexp(1.0, selected_exponent)
        if requested_changed_effect_coordinate_count >= target_count:
            return 1.0
        # Histogram bins contain midpoint breakpoints.  Doubling the bin's
        # upper edge reaches one complete directional ULP for every coordinate
        # counted in the bin, so the advertised mass does not depend on a
        # round-to-even tie at the midpoint.
        return strictly_above(max(1.0, 2.0 * raw))

    level_specs: list[dict[str, Any]] = [
        {
            "level": "requested",
            "target_nonzero_coordinate_fraction": None,
            "target_nonzero_coordinate_count": None,
            "ray_multiplier": 1.0,
        },
        {
            "level": "first_effectful",
            "target_nonzero_coordinate_fraction": (
                None
                if effect_nonzero_coordinate_count == 0
                else 1.0 / effect_nonzero_coordinate_count
            ),
            "target_nonzero_coordinate_count": (
                0 if effect_nonzero_coordinate_count == 0 else 1
            ),
            "ray_multiplier": first_effectful_multiplier,
        },
    ]
    for level, fraction in _TARGET_DTYPE_REPRESENTABILITY_MASS_LEVELS:
        target_count = (
            0
            if effect_nonzero_coordinate_count == 0
            else max(1, math.ceil(fraction * effect_nonzero_coordinate_count))
        )
        level_specs.append(
            {
                "level": level,
                "target_nonzero_coordinate_fraction": fraction,
                "target_nonzero_coordinate_count": target_count,
                "ray_multiplier": (
                    1.0
                    if target_count == 0
                    else mass_level_multiplier(target_count)
                ),
            }
        )
    selected_spec = next(
        value for value in level_specs if value["level"] == selected_level
    )
    selected_multiplier = selected_spec["ray_multiplier"]
    if selected_multiplier is None:
        raise Lfm2ProducerAwareLinearizationError(
            f"Representability level {selected_level} has no finite target-dtype "
            "ray multiplier for the requested analytic correction."
        )

    finite_level_multipliers = tuple(
        sorted(
            {
                float(value["ray_multiplier"])
                for value in level_specs
                if value["ray_multiplier"] is not None
            }
        )
    )
    level_evaluations = {
        value: {
            "changed_coordinate_count": 0,
            "changed_tensor_count": 0,
            "changed_required_coordinate_count": 0,
            "changed_required_tensor_count": 0,
            "all_target_tensors_finite": True,
        }
        for value in finite_level_multipliers
    }
    corrected: list[Any] = []
    selected_parameter_diagnostics: list[dict[str, Any]] = []
    installed_norm = 0.0
    installed_norm_saturated = False
    installed_dot_terms: list[float] = []
    selected_continuous_norm = min(
        _MAX_FINITE_NORM_EVIDENCE,
        requested_norm * float(selected_multiplier),
    )
    selected_continuous_norm_saturated = (
        requested_norm_saturated
        or not math.isfinite(requested_norm * float(selected_multiplier))
    )

    with torch.no_grad():
        for record, expected_id, (layer, role, delta) in zip(
            requested_delta_records,
            parameter_ids,
            dense_delta_stream(),
            strict=True,
        ):
            parameter_id = layer.parameter_ids[role]
            seed = layer.seed_weights[role]
            if (
                parameter_id != expected_id
                or record["requested_analytic_delta_sha256"]
                != fingerprint_lfm2_global_residual_tensor(delta)
            ):
                raise Lfm2ProducerAwareAuthenticationError(
                    "Dense correction changed between representability passes."
                )
            nonzero_mask = delta != 0
            seed_fp64 = seed.to(dtype=torch.float64)
            selected_scaled_delta = delta * float(selected_multiplier)
            selected_continuous = seed_fp64 + selected_scaled_delta
            selected_corrected = selected_continuous.to(dtype=seed.dtype)
            selected_corrected = torch.where(
                nonzero_mask,
                selected_corrected,
                seed,
            )
            selected_finite = bool(torch.isfinite(selected_corrected).all().item())
            installed_delta = selected_corrected.to(torch.float64) - seed_fp64
            installed_tensor_norm = float(
                torch.linalg.vector_norm(installed_delta).item()
            )
            if math.isfinite(installed_tensor_norm):
                installed_norm = math.hypot(installed_norm, installed_tensor_norm)
                if not math.isfinite(installed_norm):
                    installed_norm = _MAX_FINITE_NORM_EVIDENCE
                    installed_norm_saturated = True
            else:
                installed_norm = _MAX_FINITE_NORM_EVIDENCE
                installed_norm_saturated = True
            installed_dot = float((delta * installed_delta).sum().item())
            if math.isfinite(installed_dot):
                installed_dot_terms.append(installed_dot)
            selected_changed_count = int(
                torch.count_nonzero(selected_corrected != seed).item()
            )
            per_tensor_cosine = None
            requested_tensor_norm = float(record["requested_analytic_delta_l2_norm"])
            if (
                requested_tensor_norm > 0.0
                and installed_tensor_norm > 0.0
                and math.isfinite(installed_dot)
            ):
                per_tensor_cosine = max(
                    -1.0,
                    min(
                        1.0,
                        installed_dot
                        / (requested_tensor_norm * installed_tensor_norm),
                    ),
                )
            selected_parameter_diagnostics.append(
                {
                    **record,
                    "continuous_projected_delta_sha256": (
                        fingerprint_lfm2_global_residual_tensor(
                            selected_scaled_delta
                        )
                    ),
                    "installed_target_dtype_delta_sha256": (
                        fingerprint_lfm2_global_residual_tensor(installed_delta)
                    ),
                    "installed_target_dtype_delta_l2_norm": (
                        installed_tensor_norm
                        if math.isfinite(installed_tensor_norm)
                        else _MAX_FINITE_NORM_EVIDENCE
                    ),
                    "installed_to_requested_analytic_direction_cosine": (
                        per_tensor_cosine
                    ),
                    "selected_target_dtype_changed_coordinate_count": (
                        selected_changed_count
                    ),
                    "selected_target_dtype_finite": selected_finite,
                }
            )
            corrected.append(selected_corrected.contiguous().clone())

            for multiplier, evaluation in level_evaluations.items():
                if multiplier == float(selected_multiplier):
                    candidate = selected_corrected
                else:
                    candidate = (seed_fp64 + delta * multiplier).to(dtype=seed.dtype)
                    candidate = torch.where(nonzero_mask, candidate, seed)
                finite = bool(torch.isfinite(candidate).all().item())
                changed = int(torch.count_nonzero(candidate != seed).item())
                evaluation["all_target_tensors_finite"] = (
                    evaluation["all_target_tensors_finite"] and finite
                )
                evaluation["changed_coordinate_count"] += changed
                evaluation["changed_tensor_count"] += int(changed > 0)
                if parameter_id in required_id_set:
                    evaluation["changed_required_coordinate_count"] += changed
                    evaluation["changed_required_tensor_count"] += int(changed > 0)

    selected_evaluation = level_evaluations[float(selected_multiplier)]
    selected_changed_effect_coordinate_count = int(
        selected_evaluation["changed_required_coordinate_count"]
        if required_nonzero_coordinate_count > 0
        else selected_evaluation["changed_coordinate_count"]
    )
    if not selected_evaluation["all_target_tensors_finite"]:
        raise Lfm2ProducerAwareLinearizationError(
            f"Representability level {selected_level} produces non-finite target "
            "dtype weights."
        )
    requested_directional_projection = selected_level == "requested"
    if (
        required_nonzero_coordinate_count > 0
        and selected_evaluation["changed_required_coordinate_count"] == 0
        and not requested_directional_projection
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "A nonzero required analytic dense correction cannot publish a "
            "target-dtype physical no-op; choose an effectful representability "
            "level or inspect the authenticated ULP geometry."
        )
    if (
        all_nonzero_coordinate_count > 0
        and selected_evaluation["changed_coordinate_count"] == 0
        and not requested_directional_projection
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "A nonzero analytic dense closure cannot publish a total target-dtype "
            "physical no-op; choose an effectful representability level or inspect "
            "the authenticated ULP geometry."
        )
    selected_target_count = selected_spec["target_nonzero_coordinate_count"]
    if (
        selected_level.startswith("mass_")
        and type(selected_target_count) is int
        and selected_changed_effect_coordinate_count < selected_target_count
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "The selected coordinate-mass representability level did not reach "
            "its authenticated target-dtype change mass."
        )
    if (
        all_nonzero_coordinate_count == 0
        and selected_evaluation["changed_coordinate_count"] != 0
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "An exactly zero analytic dense closure changed physical bytes."
        )
    if (
        required_nonzero_coordinate_count == 0
        and selected_evaluation["changed_required_coordinate_count"] != 0
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "An exactly zero required analytic correction changed physical bytes."
        )
    installed_dot_total = math.fsum(installed_dot_terms)
    installed_cosine = None
    if (
        requested_norm > 0.0
        and installed_norm > 0.0
        and math.isfinite(installed_dot_total)
        and not requested_norm_saturated
        and not installed_norm_saturated
    ):
        installed_cosine = max(
            -1.0,
            min(1.0, installed_dot_total / (requested_norm * installed_norm)),
        )
    installed_norm_ratio = None
    installed_projection_scale = None
    if (
        requested_norm > 0.0
        and math.isfinite(requested_norm)
        and not requested_norm_saturated
        and math.isfinite(installed_norm)
        and not installed_norm_saturated
    ):
        candidate_norm_ratio = installed_norm / requested_norm
        installed_norm_ratio = (
            candidate_norm_ratio if math.isfinite(candidate_norm_ratio) else None
        )
        if math.isfinite(installed_dot_total):
            installed_projection_scale = (
                installed_dot_total / requested_norm / requested_norm
            )
            if not math.isfinite(installed_projection_scale):
                installed_projection_scale = None

    evaluated_frontier = []
    for spec in level_specs:
        multiplier = spec["ray_multiplier"]
        evaluation = (
            None if multiplier is None else level_evaluations[float(multiplier)]
        )
        changed_required = (
            None
            if evaluation is None
            else int(evaluation["changed_required_coordinate_count"])
        )
        evaluated_frontier.append(
            {
                **spec,
                "selected_for_this_transaction": spec["level"] == selected_level,
                "available_finite_multiplier": multiplier is not None,
                "all_target_tensors_finite": (
                    None
                    if evaluation is None
                    else bool(evaluation["all_target_tensors_finite"])
                ),
                "changed_coordinate_count": (
                    None
                    if evaluation is None
                    else int(evaluation["changed_coordinate_count"])
                ),
                "changed_tensor_count": (
                    None
                    if evaluation is None
                    else int(evaluation["changed_tensor_count"])
                ),
                "changed_required_coordinate_count": changed_required,
                "changed_required_tensor_count": (
                    None
                    if evaluation is None
                    else int(evaluation["changed_required_tensor_count"])
                ),
                "changed_required_nonzero_coordinate_fraction": (
                    None
                    if changed_required is None
                    or required_nonzero_coordinate_count == 0
                    else changed_required / required_nonzero_coordinate_count
                ),
                "changed_effect_coordinate_count": (
                    None
                    if evaluation is None
                    else int(
                        evaluation["changed_required_coordinate_count"]
                        if required_nonzero_coordinate_count > 0
                        else evaluation["changed_coordinate_count"]
                    )
                ),
                "changed_effect_nonzero_coordinate_fraction": (
                    None
                    if evaluation is None or effect_nonzero_coordinate_count == 0
                    else (
                        int(
                            evaluation["changed_required_coordinate_count"]
                            if required_nonzero_coordinate_count > 0
                            else evaluation["changed_coordinate_count"]
                        )
                        / effect_nonzero_coordinate_count
                    )
                ),
                "behavioral_winner_claimed": False,
                "numeric_or_manual_quality_gate": False,
            }
        )

    if not all(value.verify_authentication() for value in layers):
        raise Lfm2ProducerAwareAuthenticationError(
            "Layer seed state changed during dense absorption."
        )
    representability_report = _seal_report(
        {
            "schema": LFM2_PRODUCER_AWARE_LINEARIZATION_SCHEMA,
            "kind": "target_dtype_global_ray_representability_projection",
            "policy": "deterministic-target-dtype-ulp-global-ray-v1",
            "selected_level": selected_level,
            "selected_ray_multiplier": float(selected_multiplier),
            "requested_analytic_ray_multiplier": 1.0,
            "requested_directional_projection_without_ulp_escalation": (
                requested_directional_projection
            ),
            "target_dtype_ulp_escalation_applied": (
                float(selected_multiplier) != 1.0
            ),
            "single_global_positive_multiplier_applied_to_every_dense_delta": True,
            "continuous_analytic_ray_preserved_exactly": True,
            "coordinate_wise_rounding_noise_injected": False,
            "nextafter_values_installed_as_weights": False,
            "nextafter_used_only_for_rounding_breakpoint_geometry": True,
            "required_effect_parameter_ids": list(required_ids),
            "required_effect_parameter_count": len(required_ids),
            "representability_effect_scope": effect_scope,
            "representability_effect_nonzero_coordinate_count": (
                effect_nonzero_coordinate_count
            ),
            "requested_analytic_nonzero_coordinate_count": (
                all_nonzero_coordinate_count
            ),
            "required_analytic_nonzero_coordinate_count": (
                required_nonzero_coordinate_count
            ),
            "required_analytic_dense_correction_is_exactly_zero": (
                required_nonzero_coordinate_count == 0
            ),
            "complete_analytic_dense_correction_is_exactly_zero": (
                all_nonzero_coordinate_count == 0
            ),
            "requested_target_dtype_changed_coordinate_count": (
                requested_changed_coordinate_count
            ),
            "requested_target_dtype_changed_tensor_count": (
                requested_changed_tensor_count
            ),
            "requested_target_dtype_changed_required_coordinate_count": (
                requested_changed_required_coordinate_count
            ),
            "requested_target_dtype_changed_required_tensor_count": (
                requested_changed_required_tensor_count
            ),
            "finite_required_breakpoint_count": finite_required_breakpoint_count,
            "unavailable_finite_required_breakpoint_count": (
                unavailable_required_breakpoint_count
            ),
            "finite_all_breakpoint_count": finite_all_breakpoint_count,
            "unavailable_finite_all_breakpoint_count": (
                unavailable_all_breakpoint_count
            ),
            "first_change_midpoint_breakpoint": effect_first_breakpoint,
            "first_change_breakpoint_witness": effect_first_breakpoint_witness,
            "first_effectful_multiplier_geometry": (
                "one-full-directional-target-dtype-ulp-on-smallest-midpoint-ray"
            ),
            "coordinate_mass_multiplier_geometry": (
                "one-full-directional-target-dtype-ulp-at-histogram-upper-edge"
            ),
            "breakpoint_power_of_two_histogram": [
                {"ceil_log2_breakpoint": exponent, "coordinate_count": count}
                for exponent, count in sorted_histogram
            ],
            "frontier": evaluated_frontier,
            "frontier_is_deterministic_coordinate_mass_evidence": True,
            "frontier_member_is_behavioral_winner": False,
            "decoded_behavior_may_compare_frontier_members": True,
            "requested_analytic_dense_delta_l2_norm": requested_norm,
            "requested_analytic_dense_delta_norm_saturated": (
                requested_norm_saturated
            ),
            "continuous_projected_dense_delta_l2_norm": (
                selected_continuous_norm
            ),
            "continuous_projected_dense_delta_norm_saturated": (
                selected_continuous_norm_saturated
            ),
            "installed_target_dtype_dense_delta_l2_norm": installed_norm,
            "installed_target_dtype_dense_delta_norm_saturated": (
                installed_norm_saturated
            ),
            "installed_to_requested_analytic_direction_cosine": installed_cosine,
            "installed_to_requested_analytic_delta_l2_ratio": (
                installed_norm_ratio
            ),
            "installed_to_requested_analytic_projection_scale": (
                installed_projection_scale
            ),
            "installed_projection_scale_is_functional_ray_diagnostic_only": True,
            "selected_changed_coordinate_count": int(
                selected_evaluation["changed_coordinate_count"]
            ),
            "selected_changed_tensor_count": int(
                selected_evaluation["changed_tensor_count"]
            ),
            "selected_changed_required_coordinate_count": int(
                selected_evaluation["changed_required_coordinate_count"]
            ),
            "selected_changed_required_tensor_count": int(
                selected_evaluation["changed_required_tensor_count"]
            ),
            "selected_target_tensors_finite": bool(
                selected_evaluation["all_target_tensors_finite"]
            ),
            "parameter_diagnostics": selected_parameter_diagnostics,
            "nonzero_required_closure_may_publish_seed_identical": (
                requested_directional_projection
            ),
            "nonzero_complete_closure_may_publish_seed_identical": (
                requested_directional_projection
            ),
            "exact_zero_required_closure_may_publish_seed_identical": True,
            "exact_zero_complete_closure_may_publish_seed_identical": True,
            "seed_identical_requested_projection_claims_realized_effect": False,
            "selection_is_target_dtype_execution_geometry_not_quality": True,
            "numeric_or_manual_quality_gate": False,
            "decoded_intelligence_gate": False,
            "dense_delta_algebra_pass_count": 2,
        }
    )
    report = _seal_report(
        {
            "schema": LFM2_PRODUCER_AWARE_LINEARIZATION_SCHEMA,
            "kind": "producer_aware_dense_stack_transaction",
            "layer_seed_authentication_sha256": [
                value.authentication_sha256 for value in layers
            ],
            "output_basis_sha256": fingerprint_lfm2_global_residual_tensor(basis),
            "coefficient_sha256": [
                fingerprint_lfm2_global_residual_tensor(value)
                for value in coefficient_archives
            ],
            "compute_execution_receipt_sha256": compute_receipt["report_sha256"],
            "compute_execution_receipt": dict(compute_receipt),
            "parameter_ids": parameter_ids,
            "layer_indices": layer_indices,
            "roles": roles,
            "seed_sha256": [
                fingerprint_lfm2_global_residual_tensor(value) for value in seeds
            ],
            "corrected_sha256": [
                fingerprint_lfm2_global_residual_tensor(value) for value in corrected
            ],
            "atomic_groups": [list(group) for group in groups],
            "target_dtype_representability_sha256": representability_report[
                "report_sha256"
            ],
            "target_dtype_representability": dict(representability_report),
            "target_dtype_representability_level": selected_level,
            "target_dtype_representability_ray_multiplier": float(
                selected_multiplier
            ),
            "target_dtype_representability_frontier": evaluated_frontier,
            "target_dtype_changed_parameter_ids": [
                parameter_id
                for parameter_id, seed, value in zip(
                    parameter_ids,
                    seeds,
                    corrected,
                    strict=True,
                )
                if fingerprint_lfm2_global_residual_tensor(seed)
                != fingerprint_lfm2_global_residual_tensor(value)
            ],
            "attention_absorption": "delta_Wrole=Wrole@output_basis@C",
            "short_conv_absorption": "delta_Win=Win@output_basis@C_all_BCX",
            "mlp_absorption": "delta_Wgate_up=Wgate_up@output_basis@C",
            "output_absorption": "delta_Wout=output_basis@C",
            "seed_tensors_mutated": False,
            "physical_parameter_sharing": False,
            "runtime_sidecar_required": False,
            "numerical_quality_gate": False,
            "dense_delta_algebra_pass_count": 2,
            "execution": {
                "backward_calls": 0,
                "autograd_grad_calls": 0,
                "optimizer_steps": 0,
                "training_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
            },
        }
    )
    return Lfm2ProducerAwareDenseTransaction(
        parameter_ids=tuple(parameter_ids),
        layer_indices=tuple(layer_indices),
        roles=tuple(roles),
        seed_weights=tuple(seeds),
        corrected_weights=tuple(corrected),
        atomic_groups=tuple(groups),
        report=report,
    )


@dataclass(frozen=True, slots=True)
class Lfm2ProducerAwareInstallationReceipt:
    """Authenticated successful all-or-nothing physical installation."""

    post_install_sha256: Mapping[str, str]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "post_install_sha256",
            MappingProxyType(dict(self.post_install_sha256)),
        )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))
        if not self.verify_authentication():
            raise Lfm2ProducerAwareAuthenticationError(
                "Installation receipt failed authentication."
            )

    def verify_authentication(self) -> bool:
        return _verify_report(self.report) and self.report.get(
            "post_install_sha256"
        ) == dict(self.post_install_sha256)


def _copy_live_parameter(parameter: Any, value: Any) -> None:
    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def install_lfm2_producer_aware_dense_transaction(
    transaction: Lfm2ProducerAwareDenseTransaction,
    live_parameters: Mapping[str, Any],
    *,
    deferred_parameter_ids: Sequence[str] = (),
) -> Lfm2ProducerAwareInstallationReceipt:
    """Install coupled roles atomically, optionally deferring singleton roles."""

    torch = _torch()
    if not isinstance(transaction, Lfm2ProducerAwareDenseTransaction) or not (
        transaction.verify_authentication()
    ):
        raise Lfm2ProducerAwareAuthenticationError(
            "transaction must be authenticated immutable evidence."
        )
    deferred_ids = tuple(str(value).strip() for value in deferred_parameter_ids)
    transaction_id_set = set(transaction.parameter_ids)
    if (
        any(not value for value in deferred_ids)
        or len(set(deferred_ids)) != len(deferred_ids)
        or not set(deferred_ids).issubset(transaction_id_set)
        or any(
            parameter_id in group and len(group) != 1
            for parameter_id in deferred_ids
            for group in transaction.atomic_groups
        )
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "Deferred ids must be unique singleton atomic groups in the transaction."
        )
    install_ids = tuple(
        value for value in transaction.parameter_ids if value not in set(deferred_ids)
    )
    if not isinstance(live_parameters, Mapping) or set(live_parameters) != set(
        install_ids
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "live_parameters must exactly cover nondeferred physical transaction ids."
        )
    parameters = dict(live_parameters)
    if any(
        not isinstance(value, torch.nn.Parameter)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        for value in parameters.values()
    ):
        raise Lfm2ProducerAwareLinearizationError(
            "Every live target must be an ordinary dense Parameter."
        )
    identities = [_storage_identity(value) for value in parameters.values()]
    if len(set(identities)) != len(identities):
        raise Lfm2ProducerAwareLinearizationError(
            "Live physical parameters cannot share tensor storage."
        )
    seed_by_id = dict(
        zip(transaction.parameter_ids, transaction.seed_weights, strict=True)
    )
    corrected_by_id = dict(
        zip(transaction.parameter_ids, transaction.corrected_weights, strict=True)
    )
    snapshots: dict[str, Any] = {}
    for parameter_id, parameter in parameters.items():
        seed = seed_by_id[parameter_id]
        if tuple(parameter.shape) != tuple(seed.shape) or parameter.dtype != seed.dtype:
            raise Lfm2ProducerAwareLinearizationError(
                f"Live parameter {parameter_id} differs from seed geometry or dtype."
            )
        if fingerprint_lfm2_global_residual_tensor(parameter) != (
            fingerprint_lfm2_global_residual_tensor(seed)
        ):
            raise Lfm2ProducerAwareAuthenticationError(
                f"Live parameter {parameter_id} is stale relative to its seed."
            )
        snapshots[parameter_id] = (
            parameter.detach().to(device="cpu").contiguous().clone()
        )
    installed: list[str] = []

    def transaction_callback() -> Mapping[str, str]:
        try:
            with torch.no_grad():
                for parameter_id in install_ids:
                    _copy_live_parameter(
                        parameters[parameter_id],
                        corrected_by_id[parameter_id],
                    )
                    installed.append(parameter_id)
            observed = {
                parameter_id: fingerprint_lfm2_global_residual_tensor(
                    parameters[parameter_id]
                )
                for parameter_id in install_ids
            }
            expected = {
                parameter_id: fingerprint_lfm2_global_residual_tensor(
                    corrected_by_id[parameter_id]
                )
                for parameter_id in install_ids
            }
            if observed != expected:
                raise Lfm2ProducerAwareAuthenticationError(
                    "Physical installation did not reproduce corrected bytes."
                )
            return MappingProxyType(observed)
        except Exception as exc:
            rollback_errors = []
            with torch.no_grad():
                for parameter_id in reversed(installed):
                    try:
                        _copy_live_parameter(
                            parameters[parameter_id],
                            snapshots[parameter_id],
                        )
                    except Exception as rollback_exc:  # pragma: no cover - hardware
                        rollback_errors.append((parameter_id, repr(rollback_exc)))
            if rollback_errors:
                raise Lfm2ProducerAwareLinearizationError(
                    f"Producer-aware install and rollback failed: {rollback_errors!r}."
                ) from exc
            raise Lfm2ProducerAwareLinearizationError(
                "Producer-aware install failed and every changed parameter was restored."
            ) from exc

    post_hashes, guard = run_fold_only_callback(
        transaction_callback,
        phase="lfm2-producer-aware-dense-stack-transaction",
        protected_models=(),
        freeze_parameters=False,
    )
    report = _seal_report(
        {
            "schema": LFM2_PRODUCER_AWARE_LINEARIZATION_SCHEMA,
            "kind": "producer_aware_dense_installation_receipt",
            "transaction_report_sha256": transaction.report["report_sha256"],
            "post_install_sha256": dict(post_hashes),
            "installed_parameter_ids": list(install_ids),
            "deferred_parameter_ids": list(deferred_ids),
            "all_nondeferred_physical_parameters_installed": True,
            "all_physical_parameters_installed": not deferred_ids,
            "all_coupled_groups_atomic": True,
            "rollback_available_for_every_parameter": True,
            "physical_parameter_sharing": False,
            "numerical_quality_gate": False,
            "fold_only_guard": guard.as_dict(),
        }
    )
    return Lfm2ProducerAwareInstallationReceipt(
        post_install_sha256=post_hashes,
        report=report,
    )


__all__ = [
    "LFM2_PRODUCER_AWARE_LINEARIZATION_SCHEMA",
    "Lfm2ProducerAwareAuthenticationError",
    "Lfm2ProducerAwareDenseTransaction",
    "Lfm2ProducerAwareInstallationReceipt",
    "Lfm2ProducerAwareLayerSeedWeights",
    "Lfm2ProducerAwareLinearizationError",
    "Lfm2ProducerAwareSegmentCotangent",
    "Lfm2ProducerAwareWholeStackLinearizationKernel",
    "build_lfm2_producer_aware_writer_topology",
    "capture_lfm2_producer_aware_layer_seed",
    "install_lfm2_producer_aware_dense_transaction",
    "lfm2_producer_aware_attention_segment_jvp",
    "lfm2_producer_aware_attention_segment_vjp",
    "lfm2_producer_aware_boundary_causal_order",
    "lfm2_producer_aware_mlp_segment_jvp",
    "lfm2_producer_aware_mlp_segment_vjp",
    "lfm2_producer_aware_short_conv_segment_jvp",
    "lfm2_producer_aware_short_conv_segment_vjp",
    "lfm2_producer_aware_site_causal_orders",
    "lfm2_producer_aware_whole_stack_norm_evidence",
    "materialize_lfm2_producer_aware_stack_transaction",
]
