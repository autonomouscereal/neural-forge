"""Algebra-aware intact-channel Q-DEIM for an LFM2 short convolution.

The nonlinear path of one LFM2 short-convolution channel is an inseparable
tuple::

    B_i(x) * X_i(x) -> depthwise_FIR_i -> C_i(x) * (...) -> O[:, i]

Consequently, dense basis changes on the internal channel axis do not commute
with either the products or the depthwise FIR.  This compiler never averages
those producers or kernels.  It selects complete real parent-channel tuples
with deterministic structural Q-DEIM and reconstructs the omitted channel
action only in the final dense writer.

For parent producer rows ``B, C, X``, FIR rows ``K``, and writer ``O``, the
structural metric is built in FP64 as::

    G_BX   = .5 * ((B B.T) * (X X.T) + (B X.T) * (X B.T))
    G_ctx  = (C C.T) * (K K.T) * G_BX
    G_full = (O.T O) * G_ctx

Q-DEIM/QRCP chooses complete channels ``S`` from the leading canonical
eigenbasis of ``G_full``.  The structural interpolation and dense writer are::

    R       = solve(G_ctx[S, S], G_ctx[S, :])
    O_prior = O @ R.T

Optional residual seams are absorbed into the producer and writer matrices in
the same closed-form materialization.  They are not runtime modules.  All
algebra is deterministic FP64 on CPU followed by one explicit FP32
materialization.  There is no training, backpropagation, optimizer, pruning,
quantization, sparsity, parameter sharing, sidecar, trace capture, or network
access in this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_mlp_qdeim import (
    _canonicalize_pod_signs as _canonicalize_qdeim_basis_signs,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_SHORT_CONV_CHANNEL_QDEIM_SCHEMA = (
    "neural-forge-lfm2-short-conv-intact-channel-qdeim-v1"
)
LFM2_SHORT_CONV_CHANNEL_QDEIM_RANK_DEFICIT_SCHEMA = (
    "neural-forge-lfm2-short-conv-intact-channel-qdeim-rank-deficit-v1"
)
LFM2_SHORT_CONV_CHANNEL_QDEIM_SOURCE_SCHEMA = (
    "neural-forge-lfm2-short-conv-intact-channel-source-v1"
)
LFM2_SHORT_CONV_CHANNEL_QDEIM_SEAM_SCHEMA = (
    "neural-forge-lfm2-short-conv-intact-channel-seam-v1"
)
LFM2_SHORT_CONV_CHANNEL_QDEIM_STATE_SCHEMA = (
    "neural-forge-lfm2-short-conv-intact-channel-state-v1"
)

_STATE_KEYS = ("conv.weight", "in_proj.weight", "out_proj.weight")
_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "candidate_model_forward_calls": 0,
        "model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "new_parent_trace_rows": 0,
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "producer_channels_dense_averaged": 0,
        "fir_channels_dense_averaged": 0,
        "runtime_bridge_modules": 0,
        "runtime_parameter_aliases_created": 0,
        "runtime_parameter_sharing_used": False,
        "parent_sidecars_required": 0,
        "network_access": False,
    }
)


class Lfm2ShortConvChannelQdeimError(ValueError):
    """An authenticated intact-channel Q-DEIM compilation failed closed."""


class Lfm2ShortConvChannelQdeimInsufficientRankError(Lfm2ShortConvChannelQdeimError):
    """The sealed structural metric cannot support the requested width."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = MappingProxyType(dict(report))
        status = str(report.get("status", "structural-rank-deficit"))
        observed = int(report.get("observed_structural_rank", 0))
        required = int(report.get("required_structural_rank", 0))
        super().__init__(
            f"Intact-channel Q-DEIM failed closed ({status}): structural rank "
            f"{observed} is below required rank {required}."
        )


@dataclass(frozen=True, slots=True)
class Lfm2CompiledShortConvChannelQdeim:
    """One dense bridge-free short convolution compiled from intact tuples."""

    selected_source_channels: tuple[int, ...]
    in_proj_weight: Any = field(repr=False, compare=False)
    conv_weight: Any = field(repr=False, compare=False)
    out_proj_weight: Any = field(repr=False, compare=False)
    structural_pod_basis: Any = field(repr=False, compare=False)
    interpolation_reconstruction: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def state_dict(self) -> Mapping[str, Any]:
        """Return independent detached copies of the three dense parameters."""

        return MappingProxyType(
            {
                "conv.weight": self.conv_weight.detach()
                .to(device="cpu")
                .contiguous()
                .clone(),
                "in_proj.weight": self.in_proj_weight.detach()
                .to(device="cpu")
                .contiguous()
                .clone(),
                "out_proj.weight": self.out_proj_weight.detach()
                .to(device="cpu")
                .contiguous()
                .clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable JSON-compatible copy of the authenticated report."""

        return _deep_thaw(self.report)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise RuntimeError(
            "Intact-channel LFM2 short-conv Q-DEIM requires PyTorch."
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


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2ShortConvChannelQdeimError(f"{label} must be one positive integer.")
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ShortConvChannelQdeimError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _materialized_fp32(value: Any, *, label: str, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or value.dtype != torch.float32
        or bool(getattr(value, "is_quantized", False))
    ):
        raise Lfm2ShortConvChannelQdeimError(
            f"{label} must be one dense materialized non-quantized FP32 "
            f"rank-{ndim} tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2ShortConvChannelQdeimError(f"{label} contains non-finite values.")
    return result


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": lfm2_authenticated_tensor_sha256(tensor),
    }


def _state_sha256(state: Mapping[str, Any]) -> str:
    if tuple(sorted(state)) != tuple(sorted(_STATE_KEYS)):
        raise Lfm2ShortConvChannelQdeimError(
            "Short-conv Q-DEIM state namespace is incomplete."
        )
    return canonical_sha256(
        {
            "schema": LFM2_SHORT_CONV_CHANNEL_QDEIM_STATE_SCHEMA,
            "tensors": [_tensor_record(name, state[name]) for name in sorted(state)],
        }
    )


def _storage_identity(tensor: Any) -> tuple[str, str, int, int]:
    return (
        str(tensor.device),
        str(tensor.dtype),
        int(tensor.untyped_storage().data_ptr()),
        int(tensor.untyped_storage().nbytes()),
    )


def _assert_no_aliases(tensors: Mapping[str, Any]) -> None:
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    for name, tensor in tensors.items():
        groups.setdefault(_storage_identity(tensor), []).append(name)
    aliases = [tuple(names) for names in groups.values() if len(names) > 1]
    if aliases:
        raise Lfm2ShortConvChannelQdeimError(
            f"Materialized Q-DEIM parameters share tensor storage: {aliases!r}."
        )


def _source_tensors(
    source_in_proj_weight: Any,
    source_conv_weight: Any,
    source_out_proj_weight: Any,
) -> tuple[Any, Any, Any, int, int, int]:
    in_proj = _materialized_fp32(
        source_in_proj_weight,
        label="source_in_proj_weight",
        ndim=2,
    )
    conv = _materialized_fp32(
        source_conv_weight,
        label="source_conv_weight",
        ndim=3,
    )
    out_proj = _materialized_fp32(
        source_out_proj_weight,
        label="source_out_proj_weight",
        ndim=2,
    )
    source_channels = int(conv.shape[0])
    source_hidden = int(in_proj.shape[1])
    kernel_size = int(conv.shape[2])
    if (
        int(conv.shape[1]) != 1
        or tuple(in_proj.shape) != (3 * source_channels, source_hidden)
        or tuple(out_proj.shape) != (source_hidden, source_channels)
    ):
        raise Lfm2ShortConvChannelQdeimError(
            "Source tensors do not form one bias-free LFM2 short convolution: "
            "expected in_proj[3P,H], depthwise conv[P,1,K], and out_proj[H,P]."
        )
    _assert_no_aliases(
        {
            "source.conv.weight": conv,
            "source.in_proj.weight": in_proj,
            "source.out_proj.weight": out_proj,
        }
    )
    return in_proj, conv, out_proj, source_channels, source_hidden, kernel_size


def lfm2_short_conv_channel_qdeim_source_state_sha256(
    source_in_proj_weight: Any,
    source_conv_weight: Any,
    source_out_proj_weight: Any,
) -> str:
    """Fingerprint the detached dense parent short-convolution state."""

    in_proj, conv, out_proj, channels, hidden, kernel = _source_tensors(
        source_in_proj_weight,
        source_conv_weight,
        source_out_proj_weight,
    )
    return canonical_sha256(
        {
            "schema": LFM2_SHORT_CONV_CHANNEL_QDEIM_SOURCE_SCHEMA,
            "topology": {
                "source_internal_channels": channels,
                "source_hidden_size": hidden,
                "kernel_size": kernel,
                "bias": False,
                "parameter_dtype": "float32",
            },
            "tensors": [
                _tensor_record("conv.weight", conv),
                _tensor_record("in_proj.weight", in_proj),
                _tensor_record("out_proj.weight", out_proj),
            ],
        }
    )


def _normalized_seams(
    input_decoder: Any | None,
    output_encoder: Any | None,
    *,
    source_hidden: int,
) -> tuple[Any | None, Any | None, int, str]:
    if (input_decoder is None) != (output_encoder is None):
        raise Lfm2ShortConvChannelQdeimError(
            "Standalone residual folding requires both input_decoder D[C,H] "
            "and output_encoder E[H,C], or neither for a parent-shell fold."
        )
    if input_decoder is None:
        return None, None, source_hidden, "parent-shell"
    decoder = _materialized_fp32(
        input_decoder,
        label="input_decoder",
        ndim=2,
    )
    encoder = _materialized_fp32(
        output_encoder,
        label="output_encoder",
        ndim=2,
    )
    child_hidden = int(decoder.shape[0])
    if (
        child_hidden <= 0
        or child_hidden > source_hidden
        or tuple(decoder.shape) != (child_hidden, source_hidden)
        or tuple(encoder.shape) != (source_hidden, child_hidden)
    ):
        raise Lfm2ShortConvChannelQdeimError(
            "Residual seams must be D[C,H] and E[H,C] with 0 < C <= H."
        )
    _assert_no_aliases({"input_decoder": decoder, "output_encoder": encoder})
    return decoder, encoder, child_hidden, "standalone-residual-seams-absorbed"


def lfm2_short_conv_channel_qdeim_seam_state_sha256(
    *,
    source_hidden_size: int,
    input_decoder: Any | None = None,
    output_encoder: Any | None = None,
) -> str:
    """Fingerprint either the parent-shell identity seam or a D/E seam pair."""

    source_hidden = _positive_int(
        source_hidden_size,
        label="source_hidden_size",
    )
    decoder, encoder, external_hidden, mode = _normalized_seams(
        input_decoder,
        output_encoder,
        source_hidden=source_hidden,
    )
    tensors: list[dict[str, Any]] = []
    if decoder is not None and encoder is not None:
        tensors = [
            _tensor_record("input_decoder", decoder),
            _tensor_record("output_encoder", encoder),
        ]
    return canonical_sha256(
        {
            "schema": LFM2_SHORT_CONV_CHANNEL_QDEIM_SEAM_SCHEMA,
            "mode": mode,
            "source_hidden_size": source_hidden,
            "external_hidden_size": external_hidden,
            "tensors": tensors,
        }
    )


def lfm2_short_conv_channel_qdeim_parameter_accounting(
    *,
    source_hidden_size: int,
    source_internal_channels: int,
    target_internal_channels: int,
    kernel_size: int,
    external_hidden_size: int | None = None,
) -> Mapping[str, Any]:
    """Return exact dense element counts; no storage or quantization shortcuts."""

    source_hidden = _positive_int(source_hidden_size, label="source_hidden_size")
    source_channels = _positive_int(
        source_internal_channels,
        label="source_internal_channels",
    )
    target_channels = _positive_int(
        target_internal_channels,
        label="target_internal_channels",
    )
    kernel = _positive_int(kernel_size, label="kernel_size")
    if target_channels > source_channels:
        raise Lfm2ShortConvChannelQdeimError(
            "target_internal_channels cannot exceed source_internal_channels."
        )
    external_hidden = (
        source_hidden
        if external_hidden_size is None
        else _positive_int(external_hidden_size, label="external_hidden_size")
    )
    if external_hidden > source_hidden:
        raise Lfm2ShortConvChannelQdeimError(
            "external_hidden_size cannot exceed source_hidden_size."
        )
    source_count = source_channels * (4 * source_hidden + kernel)
    candidate_count = target_channels * (4 * external_hidden + kernel)
    body = {
        "schema": "neural-forge-lfm2-short-conv-intact-channel-count-v1",
        "source_hidden_size": source_hidden,
        "external_hidden_size": external_hidden,
        "source_internal_channels": source_channels,
        "target_internal_channels": target_channels,
        "kernel_size": kernel,
        "source_dense_parameter_count": source_count,
        "candidate_dense_parameter_count": candidate_count,
        "dense_parameters_removed": source_count - candidate_count,
        "component_compression_ratio": source_count / candidate_count,
        "candidate_times_four": candidate_count * 4,
        "source_exactly_four_times_candidate": source_count == 4 * candidate_count,
        "parameter_sharing_used": False,
        "sparse_storage_used": False,
        "quantization_used": False,
    }
    return MappingProxyType({**body, "accounting_sha256": canonical_sha256(body)})


def _normalized_protected_channels(
    values: Sequence[int],
    *,
    source_channels: int,
    target_channels: int,
) -> tuple[int, ...]:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise Lfm2ShortConvChannelQdeimError(
            "protected_source_channels must contain integer channel indices."
        )
    try:
        parsed = tuple(int(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2ShortConvChannelQdeimError(
            "protected_source_channels must contain integer channel indices."
        ) from exc
    canonical = tuple(sorted(parsed))
    if (
        len(set(canonical)) != len(canonical)
        or len(canonical) > target_channels
        or any(index < 0 or index >= source_channels for index in canonical)
    ):
        raise Lfm2ShortConvChannelQdeimError(
            "Protected channels must be unique in-range indices and cannot exceed "
            "the requested compact width."
        )
    return canonical


def _rank_tolerance(
    values: Any,
    *,
    matrix_width: int,
    relative_tolerance: float | None,
) -> float:
    torch = _torch()
    maximum = float(values.abs().max().item()) if int(values.numel()) else 0.0
    if relative_tolerance is None:
        return matrix_width * torch.finfo(torch.float64).eps * maximum
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or not math.isfinite(float(relative_tolerance))
        or float(relative_tolerance) < 0.0
    ):
        raise Lfm2ShortConvChannelQdeimError(
            "structural_rank_relative_tolerance must be finite and nonnegative."
        )
    return float(relative_tolerance) * maximum


def _orthogonal_extension(matrix: Any, q_columns: Any, index: int) -> tuple[Any, float]:
    torch = _torch()
    vector = matrix[:, index].contiguous().clone()
    if int(q_columns.shape[1]):
        vector.sub_(q_columns @ (q_columns.T @ vector))
        vector.sub_(q_columns @ (q_columns.T @ vector))
    norm = float(torch.linalg.vector_norm(vector).item())
    return vector, norm


def _deterministic_protected_qrcp_indices(
    basis: Any,
    protected_channels: tuple[int, ...],
) -> tuple[tuple[int, ...], Any]:
    """Deterministic twice-reorthogonalized QRCP with forced intact pivots."""

    torch = _torch()
    source_channels, modes = map(int, basis.shape)
    matrix = basis.T.contiguous()
    q_columns = torch.empty(
        (modes, modes),
        dtype=torch.float64,
        device=matrix.device,
    )
    selected: list[int] = []
    residual_norms: list[float] = []
    selected_mask = torch.zeros(source_channels, dtype=torch.bool, device=matrix.device)
    epsilon = torch.finfo(torch.float64).eps

    def append(index: int) -> None:
        vector, norm = _orthogonal_extension(
            matrix,
            q_columns[:, : len(selected)],
            index,
        )
        if not math.isfinite(norm) or norm <= epsilon:
            raise Lfm2ShortConvChannelQdeimError(
                "A protected or Q-DEIM pivot is structurally dependent."
            )
        q_columns[:, len(selected)].copy_(vector / norm)
        selected_mask[index] = True
        selected.append(index)
        residual_norms.append(norm)

    for protected in protected_channels:
        append(protected)

    while len(selected) < modes:
        q_active = q_columns[:, : len(selected)]
        residual = matrix.contiguous().clone()
        if int(q_active.shape[1]):
            residual.sub_(q_active @ (q_active.T @ residual))
            residual.sub_(q_active @ (q_active.T @ residual))
        scores = residual.square().sum(dim=0).masked_fill(selected_mask, -1.0)
        maximum = float(scores.max().item())
        tie_tolerance = 128.0 * epsilon * max(1.0, abs(maximum))
        eligible = torch.nonzero(
            (~selected_mask) & (scores >= maximum - tie_tolerance),
            as_tuple=False,
        ).flatten()
        if not int(eligible.numel()):
            raise Lfm2ShortConvChannelQdeimError(
                "Deterministic Q-DEIM QRCP found no independent channel pivot."
            )
        append(int(eligible.min().item()))

    sampled = basis[list(selected), :].contiguous()
    singular = torch.linalg.svdvals(sampled)
    maximum = float(singular.max().item())
    tolerance = max(sampled.shape) * epsilon * maximum
    minimum = float(singular.min().item())
    if not math.isfinite(minimum) or minimum <= tolerance:
        raise Lfm2ShortConvChannelQdeimError(
            "Deterministic Q-DEIM pivots do not form a stable sampled basis."
        )
    return tuple(selected), torch.tensor(residual_norms, dtype=torch.float64)


def _rank_deficit_report(
    *,
    status: str,
    source_state_sha256: str,
    seam_state_sha256: str,
    target_channels: int,
    observed_rank: int,
    source_channels: int,
    protected_channels: tuple[int, ...],
    structural_rank_tolerance: float,
    g_full: Any,
    eigenvalues: Any,
    g_ctx: Any,
    selected_channels: Sequence[int] = (),
) -> dict[str, Any]:
    body = {
        "schema": LFM2_SHORT_CONV_CHANNEL_QDEIM_RANK_DEFICIT_SCHEMA,
        "status": status,
        "source_state_sha256": source_state_sha256,
        "seam_state_sha256": seam_state_sha256,
        "source_internal_channels": source_channels,
        "target_internal_channels": target_channels,
        "required_structural_rank": target_channels,
        "observed_structural_rank": observed_rank,
        "structural_rank_deficit": max(0, target_channels - observed_rank),
        "structural_rank_absolute_tolerance": structural_rank_tolerance,
        "protected_source_channels": list(protected_channels),
        "selected_source_channels": [int(index) for index in selected_channels],
        "g_full_fp64": _tensor_record("G_full", g_full),
        "g_ctx_fp64": _tensor_record("G_ctx", g_ctx),
        "structural_eigenvalues_fp64": _tensor_record(
            "G_full.eigenvalues",
            eigenvalues,
        ),
        "interpolation_matrix_materialized": False,
        "candidate_parameters_materialized": False,
        "random_mode_completion_permitted": False,
        "silent_pseudoinverse_permitted": False,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _channel_tuple_lineage(
    *,
    selected: tuple[int, ...],
    source_b: Any,
    source_c: Any,
    source_x: Any,
    source_conv: Any,
    candidate_in: Any,
    candidate_conv: Any,
) -> tuple[list[dict[str, Any]], str]:
    target_channels = len(selected)
    candidate_b, candidate_c, candidate_x = candidate_in.chunk(3, dim=0)
    rows: list[dict[str, Any]] = []
    for target_index, source_index in enumerate(selected):
        row = {
            "target_channel_index": target_index,
            "source_channel_index": source_index,
            "source_tuple": {
                "B": _tensor_record(
                    f"source.B[{source_index}]",
                    source_b[source_index].contiguous(),
                ),
                "C": _tensor_record(
                    f"source.C[{source_index}]",
                    source_c[source_index].contiguous(),
                ),
                "X": _tensor_record(
                    f"source.X[{source_index}]",
                    source_x[source_index].contiguous(),
                ),
                "FIR": _tensor_record(
                    f"source.FIR[{source_index}]",
                    source_conv[source_index].contiguous(),
                ),
            },
            "candidate_tuple": {
                "B": _tensor_record(
                    f"candidate.B[{target_index}]",
                    candidate_b[target_index].contiguous(),
                ),
                "C": _tensor_record(
                    f"candidate.C[{target_index}]",
                    candidate_c[target_index].contiguous(),
                ),
                "X": _tensor_record(
                    f"candidate.X[{target_index}]",
                    candidate_x[target_index].contiguous(),
                ),
                "FIR": _tensor_record(
                    f"candidate.FIR[{target_index}]",
                    candidate_conv[target_index].contiguous(),
                ),
            },
            "producer_channel_dense_average_used": False,
            "fir_channel_dense_average_used": False,
        }
        rows.append(row)
    if len(rows) != target_channels:  # pragma: no cover - loop invariant
        raise AssertionError("Channel tuple lineage is incomplete.")
    return rows, canonical_sha256(rows)


def compile_lfm2_short_conv_channel_qdeim(
    *,
    source_in_proj_weight: Any,
    source_conv_weight: Any,
    source_out_proj_weight: Any,
    expected_source_state_sha256: str,
    target_internal_channels: int,
    input_decoder: Any | None = None,
    output_encoder: Any | None = None,
    expected_seam_state_sha256: str | None = None,
    protected_source_channels: Sequence[int] = (),
    structural_rank_relative_tolerance: float | None = None,
) -> Lfm2CompiledShortConvChannelQdeim:
    """Compile one dense intact-channel structural Q-DEIM candidate.

    ``input_decoder`` is ``D[C,H]`` and maps child residual coordinates into
    parent coordinates.  ``output_encoder`` is ``E[H,C]`` and maps the parent
    branch output back into child coordinates.  When present, both maps are
    absorbed into the returned matrices; the returned candidate has no bridge
    or parent sidecar at runtime.
    """

    torch = _torch()
    expected_source = _digest(
        expected_source_state_sha256,
        label="expected_source_state_sha256",
    )
    (
        source_in,
        source_conv,
        source_out,
        source_channels,
        source_hidden,
        kernel_size,
    ) = _source_tensors(
        source_in_proj_weight,
        source_conv_weight,
        source_out_proj_weight,
    )
    actual_source = lfm2_short_conv_channel_qdeim_source_state_sha256(
        source_in,
        source_conv,
        source_out,
    )
    if actual_source != expected_source:
        raise Lfm2ShortConvChannelQdeimError(
            "Authenticated source short-convolution state changed."
        )
    target_channels = _positive_int(
        target_internal_channels,
        label="target_internal_channels",
    )
    if target_channels > source_channels:
        raise Lfm2ShortConvChannelQdeimError(
            "target_internal_channels cannot exceed source internal width."
        )
    protected = _normalized_protected_channels(
        protected_source_channels,
        source_channels=source_channels,
        target_channels=target_channels,
    )
    decoder, encoder, external_hidden, seam_mode = _normalized_seams(
        input_decoder,
        output_encoder,
        source_hidden=source_hidden,
    )
    actual_seam = lfm2_short_conv_channel_qdeim_seam_state_sha256(
        source_hidden_size=source_hidden,
        input_decoder=decoder,
        output_encoder=encoder,
    )
    if expected_seam_state_sha256 is None:
        if decoder is not None:
            raise Lfm2ShortConvChannelQdeimError(
                "Standalone residual seams require expected_seam_state_sha256."
            )
    elif (
        _digest(
            expected_seam_state_sha256,
            label="expected_seam_state_sha256",
        )
        != actual_seam
    ):
        raise Lfm2ShortConvChannelQdeimError(
            "Authenticated residual D/E seam state changed."
        )

    source_b_fp32, source_c_fp32, source_x_fp32 = source_in.chunk(3, dim=0)
    with torch.no_grad():
        b = source_b_fp32.to(dtype=torch.float64)
        c = source_c_fp32.to(dtype=torch.float64)
        x = source_x_fp32.to(dtype=torch.float64)
        kernel = source_conv[:, 0, :].to(dtype=torch.float64)
        writer = source_out.to(dtype=torch.float64)

        b_bt = (b @ b.T).contiguous()
        x_xt = (x @ x.T).contiguous()
        b_xt = (b @ x.T).contiguous()
        x_bt = (x @ b.T).contiguous()
        c_ct = (c @ c.T).contiguous()
        k_kt = (kernel @ kernel.T).contiguous()
        o_to = (writer.T @ writer).contiguous()
        g_bx = (0.5 * (b_bt * x_xt + b_xt * x_bt)).contiguous()
        g_bx = (0.5 * (g_bx + g_bx.T)).contiguous()
        g_ctx = (c_ct * k_kt * g_bx).contiguous()
        g_ctx = (0.5 * (g_ctx + g_ctx.T)).contiguous()
        g_full = (o_to * g_ctx).contiguous()
        g_full = (0.5 * (g_full + g_full.T)).contiguous()
        for name, value in (
            ("G_BX", g_bx),
            ("G_ctx", g_ctx),
            ("G_full", g_full),
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise Lfm2ShortConvChannelQdeimError(
                    f"Structural FP64 metric {name} contains non-finite values."
                )

        eigenvalues_ascending, eigenvectors_ascending = torch.linalg.eigh(g_full)
        eigenvalues = torch.flip(eigenvalues_ascending, dims=(0,)).contiguous()
        eigenvectors = torch.flip(eigenvectors_ascending, dims=(1,)).contiguous()
        rank_tolerance = _rank_tolerance(
            eigenvalues,
            matrix_width=source_channels,
            relative_tolerance=structural_rank_relative_tolerance,
        )
        observed_rank = int((eigenvalues > rank_tolerance).sum().item())
        if observed_rank < target_channels and target_channels < source_channels:
            raise Lfm2ShortConvChannelQdeimInsufficientRankError(
                _rank_deficit_report(
                    status="insufficient-separable-gram-rank",
                    source_state_sha256=actual_source,
                    seam_state_sha256=actual_seam,
                    target_channels=target_channels,
                    observed_rank=observed_rank,
                    source_channels=source_channels,
                    protected_channels=protected,
                    structural_rank_tolerance=rank_tolerance,
                    g_full=g_full,
                    eigenvalues=eigenvalues,
                    g_ctx=g_ctx,
                )
            )
        basis = _canonicalize_qdeim_basis_signs(
            eigenvectors[:, :target_channels].contiguous()
        )
        try:
            if target_channels == source_channels:
                selected = tuple(range(source_channels))
                pivot_norms = torch.ones(source_channels, dtype=torch.float64)
                selection_method = "full-width-canonical-identity"
            else:
                selected, pivot_norms = _deterministic_protected_qrcp_indices(
                    basis,
                    protected,
                )
                selection_method = "protected-deterministic-qdeim-qrcp"
        except Lfm2ShortConvChannelQdeimError as exc:
            report = _rank_deficit_report(
                status="protected-qdeim-pivot-rank-deficit",
                source_state_sha256=actual_source,
                seam_state_sha256=actual_seam,
                target_channels=target_channels,
                observed_rank=observed_rank,
                source_channels=source_channels,
                protected_channels=protected,
                structural_rank_tolerance=rank_tolerance,
                g_full=g_full,
                eigenvalues=eigenvalues,
                g_ctx=g_ctx,
            )
            report["pivot_failure"] = str(exc)
            result_body = dict(report)
            result_body.pop("result_sha256")
            report["result_sha256"] = canonical_sha256(result_body)
            raise Lfm2ShortConvChannelQdeimInsufficientRankError(report) from exc

        index = torch.tensor(selected, dtype=torch.long)
        sampled_ctx = g_ctx.index_select(0, index).index_select(1, index).contiguous()
        sampled_to_all = g_ctx.index_select(0, index).contiguous()
        sampled_singular = torch.linalg.svdvals(sampled_ctx)
        sampled_maximum = float(sampled_singular.max().item())
        sampled_tolerance = (
            max(sampled_ctx.shape) * torch.finfo(torch.float64).eps * sampled_maximum
        )
        interpolation_rank = int((sampled_singular > sampled_tolerance).sum().item())
        if interpolation_rank < target_channels and target_channels < source_channels:
            raise Lfm2ShortConvChannelQdeimInsufficientRankError(
                _rank_deficit_report(
                    status="selected-context-interpolation-rank-deficit",
                    source_state_sha256=actual_source,
                    seam_state_sha256=actual_seam,
                    target_channels=target_channels,
                    observed_rank=interpolation_rank,
                    source_channels=source_channels,
                    protected_channels=protected,
                    structural_rank_tolerance=sampled_tolerance,
                    g_full=g_full,
                    eigenvalues=eigenvalues,
                    g_ctx=g_ctx,
                    selected_channels=selected,
                )
            )
        if target_channels == source_channels:
            interpolation = torch.eye(source_channels, dtype=torch.float64)
            interpolation_formula = (
                "I (full-width exact form of solve(G_ctx[S,S],G_ctx[S,:]))"
            )
        else:
            interpolation = torch.linalg.solve(
                sampled_ctx,
                sampled_to_all,
            ).contiguous()
            interpolation_formula = "solve(G_ctx[S,S],G_ctx[S,:])"
        if not bool(torch.isfinite(interpolation).all().item()):
            raise Lfm2ShortConvChannelQdeimError(
                "Structural interpolation contains non-finite FP64 values."
            )

        selected_b = b.index_select(0, index).contiguous()
        selected_c = c.index_select(0, index).contiguous()
        selected_x = x.index_select(0, index).contiguous()
        selected_kernel = source_conv.to(dtype=torch.float64).index_select(
            0,
            index,
        )
        if decoder is not None:
            decoder_fp64_t = decoder.to(dtype=torch.float64).T.contiguous()
            selected_b = (selected_b @ decoder_fp64_t).contiguous()
            selected_c = (selected_c @ decoder_fp64_t).contiguous()
            selected_x = (selected_x @ decoder_fp64_t).contiguous()
        candidate_in_fp64 = torch.cat(
            (selected_b, selected_c, selected_x),
            dim=0,
        ).contiguous()
        writer_prior_fp64 = (writer @ interpolation.T).contiguous()
        if encoder is not None:
            writer_prior_fp64 = (
                encoder.to(dtype=torch.float64).T @ writer_prior_fp64
            ).contiguous()

        candidate_in = candidate_in_fp64.to(dtype=torch.float32).contiguous().clone()
        candidate_conv = selected_kernel.to(dtype=torch.float32).contiguous().clone()
        candidate_out = writer_prior_fp64.to(dtype=torch.float32).contiguous().clone()
        for name, value in (
            ("in_proj.weight", candidate_in),
            ("conv.weight", candidate_conv),
            ("out_proj.weight", candidate_out),
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise Lfm2ShortConvChannelQdeimError(
                    f"{name} overflowed during explicit FP32 materialization."
                )

    state = {
        "conv.weight": candidate_conv,
        "in_proj.weight": candidate_in,
        "out_proj.weight": candidate_out,
    }
    _assert_no_aliases(state)
    tuple_lineage, tuple_lineage_sha256 = _channel_tuple_lineage(
        selected=selected,
        source_b=source_b_fp32,
        source_c=source_c_fp32,
        source_x=source_x_fp32,
        source_conv=source_conv,
        candidate_in=candidate_in,
        candidate_conv=candidate_conv,
    )
    accounting = lfm2_short_conv_channel_qdeim_parameter_accounting(
        source_hidden_size=source_hidden,
        source_internal_channels=source_channels,
        target_internal_channels=target_channels,
        kernel_size=kernel_size,
        external_hidden_size=external_hidden,
    )
    fp64_lineage = {
        "formulae": {
            "G_BX": (".5*((B@B.T)*(X@X.T)+(B@X.T)*(X@B.T))"),
            "G_ctx": "(C@C.T)*(K@K.T)*G_BX",
            "G_full": "(O.T@O)*G_ctx",
            "interpolation": interpolation_formula,
            "writer_prior": "O@R.T",
            "input_seam_absorption": (
                "selected_B/C/X@D.T" if decoder is not None else "identity"
            ),
            "output_seam_absorption": (
                "E.T@O_prior" if encoder is not None else "identity"
            ),
        },
        "symmetry_closure": "0.5*(G+G.T) in FP64",
        "G_BX": _tensor_record("G_BX", g_bx),
        "G_ctx": _tensor_record("G_ctx", g_ctx),
        "G_full": _tensor_record("G_full", g_full),
        "structural_eigenvalues": _tensor_record(
            "G_full.eigenvalues",
            eigenvalues,
        ),
        "canonical_structural_basis": _tensor_record(
            "G_full.leading_canonical_eigenbasis",
            basis,
        ),
        "qdeim_pivot_residual_norms": _tensor_record(
            "qdeim.pivot_residual_norms",
            pivot_norms,
        ),
        "sampled_context_gram": _tensor_record(
            "G_ctx[S,S]",
            sampled_ctx,
        ),
        "sampled_to_all_context_gram": _tensor_record(
            "G_ctx[S,:]",
            sampled_to_all,
        ),
        "interpolation_reconstruction": _tensor_record("R", interpolation),
        "candidate_in_proj_before_fp32": _tensor_record(
            "candidate.in_proj.fp64",
            candidate_in_fp64,
        ),
        "candidate_out_proj_before_fp32": _tensor_record(
            "candidate.out_proj.fp64",
            writer_prior_fp64,
        ),
        "solve_device": "cpu",
        "solve_dtype": "float64",
        "materialization_dtype": "float32",
    }
    fp64_lineage_sha256 = canonical_sha256(fp64_lineage)
    result_body = {
        "schema": LFM2_SHORT_CONV_CHANNEL_QDEIM_SCHEMA,
        "status": "materialized-intact-channel-structural-qdeim",
        "method": "separable-cp-gram-protected-intact-channel-qdeim-qrcp",
        "source_state_sha256": actual_source,
        "seam_state_sha256": actual_seam,
        "seam_mode": seam_mode,
        "source_hidden_size": source_hidden,
        "external_hidden_size": external_hidden,
        "source_internal_channels": source_channels,
        "target_internal_channels": target_channels,
        "kernel_size": kernel_size,
        "selected_source_channels": list(selected),
        "protected_source_channels": list(protected),
        "selection_method": selection_method,
        "observed_structural_rank": observed_rank,
        "required_structural_rank": target_channels,
        "structural_rank_required_for_materialization": (
            target_channels < source_channels
        ),
        "structural_rank_absolute_tolerance": rank_tolerance,
        "selected_context_interpolation_rank": interpolation_rank,
        "selected_context_rank_required_for_materialization": (
            target_channels < source_channels
        ),
        "selected_context_interpolation_absolute_tolerance": sampled_tolerance,
        "channel_tuple_lineage": tuple_lineage,
        "channel_tuple_lineage_sha256": tuple_lineage_sha256,
        "intact_tuple_contract": {
            "B_C_X_rows_selected_from_one_identical_source_index": True,
            "FIR_row_selected_from_that_identical_source_index": True,
            "producer_or_FIR_channel_dense_averaging_permitted": False,
            "omitted_channel_action_reconstructed_only_in_dense_writer": True,
            "interpolation_formula": interpolation_formula,
        },
        "fp64_lineage": fp64_lineage,
        "fp64_lineage_sha256": fp64_lineage_sha256,
        "candidate_state": {
            name: _tensor_record(name, state[name]) for name in sorted(state)
        },
        "candidate_state_sha256": _state_sha256(state),
        "parameter_accounting": dict(accounting),
        "parameter_storage": {
            "stored_dense_parameter_count": sum(
                int(tensor.numel()) for tensor in state.values()
            ),
            "unique_dense_parameter_tensor_count": len(state),
            "storage_alias_group_count": 0,
            "runtime_bridge_parameter_count": 0,
            "parent_sidecar_parameter_count": 0,
        },
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    report = MappingProxyType(
        {**result_body, "result_sha256": canonical_sha256(result_body)}
    )
    return Lfm2CompiledShortConvChannelQdeim(
        selected_source_channels=selected,
        in_proj_weight=candidate_in,
        conv_weight=candidate_conv,
        out_proj_weight=candidate_out,
        structural_pod_basis=basis.detach().cpu().contiguous().clone(),
        interpolation_reconstruction=(
            interpolation.detach().cpu().contiguous().clone()
        ),
        report=report,
    )


def authenticate_lfm2_short_conv_channel_qdeim(
    candidate: Lfm2CompiledShortConvChannelQdeim,
) -> Mapping[str, Any]:
    """Authenticate materialized tensors and the sealed JSON receipt."""

    if not isinstance(candidate, Lfm2CompiledShortConvChannelQdeim):
        raise TypeError("candidate must be Lfm2CompiledShortConvChannelQdeim.")
    report = dict(candidate.report)
    if report.get("schema") != LFM2_SHORT_CONV_CHANNEL_QDEIM_SCHEMA:
        raise Lfm2ShortConvChannelQdeimError(
            "Short-conv Q-DEIM receipt schema is not canonical."
        )
    _assert_no_aliases(
        {
            "conv.weight": candidate.conv_weight,
            "in_proj.weight": candidate.in_proj_weight,
            "out_proj.weight": candidate.out_proj_weight,
        }
    )
    state = candidate.state_dict()
    _assert_no_aliases(state)
    if _state_sha256(state) != report.get("candidate_state_sha256"):
        raise Lfm2ShortConvChannelQdeimError(
            "Materialized short-conv Q-DEIM state changed."
        )
    if list(candidate.selected_source_channels) != report.get(
        "selected_source_channels"
    ):
        raise Lfm2ShortConvChannelQdeimError(
            "Selected intact-channel coordinates changed."
        )
    lineage = report.get("fp64_lineage")
    if not isinstance(lineage, Mapping):
        raise Lfm2ShortConvChannelQdeimError(
            "Sealed FP64 construction lineage is missing."
        )
    if (
        _tensor_record(
            "G_full.leading_canonical_eigenbasis",
            candidate.structural_pod_basis,
        )
        != lineage.get("canonical_structural_basis")
        or _tensor_record("R", candidate.interpolation_reconstruction)
        != lineage.get("interpolation_reconstruction")
        or canonical_sha256(lineage) != report.get("fp64_lineage_sha256")
    ):
        raise Lfm2ShortConvChannelQdeimError(
            "Materialized FP64 Q-DEIM lineage changed."
        )
    tuple_lineage = report.get("channel_tuple_lineage")
    if not isinstance(tuple_lineage, list) or canonical_sha256(
        tuple_lineage
    ) != report.get("channel_tuple_lineage_sha256"):
        raise Lfm2ShortConvChannelQdeimError(
            "Intact B/C/X/FIR tuple lineage authentication failed."
        )
    expected_result = report.pop("result_sha256", None)
    if canonical_sha256(report) != expected_result:
        raise Lfm2ShortConvChannelQdeimError(
            "Short-conv Q-DEIM result receipt authentication failed."
        )
    return candidate.report


__all__ = [
    "LFM2_SHORT_CONV_CHANNEL_QDEIM_RANK_DEFICIT_SCHEMA",
    "LFM2_SHORT_CONV_CHANNEL_QDEIM_SCHEMA",
    "Lfm2CompiledShortConvChannelQdeim",
    "Lfm2ShortConvChannelQdeimError",
    "Lfm2ShortConvChannelQdeimInsufficientRankError",
    "authenticate_lfm2_short_conv_channel_qdeim",
    "compile_lfm2_short_conv_channel_qdeim",
    "lfm2_short_conv_channel_qdeim_parameter_accounting",
    "lfm2_short_conv_channel_qdeim_seam_state_sha256",
    "lfm2_short_conv_channel_qdeim_source_state_sha256",
]
