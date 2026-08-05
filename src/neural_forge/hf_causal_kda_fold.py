"""Chimera-omni Kimi-Delta-Attention (KDA) residual-branch folding.

This is stage 0/1 of the KDA lane of
``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section 1.1.  The parent
is a delta-rule recurrence with channel-wise diagonal decay in the key basis
(``chimera-omni/src/chimera_omni/modeling/layers.py:88-195``); the child is the
same operator class at a smaller geometry.  Everything here is closed form:
one exact unroll, deterministic channel selection, condition-free ridges, an
exact SVD for the decay rank, and one endpoint-coupled writer solve.  There is
no optimizer, gradient, parent forward, runtime sidecar, or database anywhere
in this file.

The design invariant transplanted from the GQA kernel
(``hf_causal_gqa_attention_fold.py:536`` onward) is: install the selection
machinery first, replay the operator the child actually computes, then fit the
writer endpoint-coupled against the residual delta.  For KDA the role of
"installed probabilities" is played by the *installed recurrence replay* --
step 7 of the design's ordered recipe.

Implementation deviations from the design document, with reasons
----------------------------------------------------------------

1. **Evidence and parent parameters are two arguments, not one.**  The design
   sketches ``hf_chimera_circuit_evidence.py`` holding the capture record
   (section 5.1).  That module is a separate deliverable, so the capture
   dataclass lives here in the ``HFCausalLayerEvidence`` field style
   (``hf_causal_circuit_evidence.py:214-233``).  It carries *rows only*.  Conv
   taps, decay bias, decay log-scale, the head norm gain, and the writer are
   parent *weights*, not activations, so they arrive in a separate sealed
   ``ChimeraKDAParameters`` carry.  Fitting a recipe step that the design calls
   an "exact carry" out of activations instead of weights would have made the
   exact step approximate.

2. **The decay inverse chain is exact, not approximate.**  The design (recipe
   step 5) writes ``u = inverse_sigmoid(g*/g_min)`` as if it were a lossy
   inversion.  Because the parent caches the pre-scale logits, ``u`` equals
   ``exp(A_h) * z`` identically, and the residual of that identity is sealed as
   ``decay_inverse_chain_relative_rmse`` -- an evidence-consistency gate rather
   than a fitted quantity.  The clamp policy still exists and still counts
   saturated rows, because a *merged* head's target can leave the interior even
   when every parent row is inside it.

3. **The child decay log-scale carries the parent's, it is not re-estimated
   from the target RMS.**  "Pick child ``A_h`` to match the per-head scale of
   ``u``" is satisfied exactly by the barycentric mean of the parent
   ``A_h`` over the merged group: for a singleton group the child's linear
   target is then *literally* the parent's ``z``, so an identity-geometry fold
   is exact instead of exact-up-to-a-rescale.  Scale dispersion inside a merged
   group is measured and fails closed above policy.

4. **Conv taps are barycentric, not carried, when heads merge.**  Depthwise
   taps are channel-aligned, so a pure channel selection carries them exactly
   (design recipe step 3).  A merged head has several parent tap sets for one
   child channel; they are combined with the same barycentric weights as every
   other merged signal and ``merged_conv_tap_dispersion`` is sealed.  Claiming
   "exact carry" for a merged head would have been false.

5. **Balanced coordinates rank; a deterministic QRCP cuts.**  The design says
   to rank with ``build_balanced_krylov_basis`` and to select channels with the
   pivot discipline of the deployed ShortConv channel QDEIM.  The balanced
   report's ``canonical_pivot_indices`` are sign-canonicalization pivots and
   may repeat, so the selection is the greedy row-pivoted Gram-Schmidt of the
   balanced decoder (a QDEIM in the ShortConv sense), seeded in balanced-mode
   order.  The canonical pivots are sealed beside the selection so the two can
   be compared.  When the balanced build cannot support the requested rank the
   selection falls back to observability-weighted energy pivots with the reason
   sealed, rather than failing a fold that has a deterministic answer.

6. **No rotation anywhere.**  ``Diag(alpha)`` is diagonal in the key basis and
   ``HeadwiseRMSNorm`` carries a per-(head, channel) gain, so both axes are cut
   by channel-aligned selection only (design section 1.1, coordinates 2 and 3,
   and the Trade-offs entry that rejects the rotation).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

from .balanced_krylov import BalancedKrylovError, build_balanced_krylov_basis
from .dynamic_circuit_ir import canonical_sha256


SCHEMA = "neural-forge-chimera-kda-fold-v1"
EVIDENCE_SCHEMA = "neural-forge-chimera-kda-layer-evidence-v1"
UNROLL_SCHEMA = "neural-forge-chimera-kda-effective-weight-unroll-v1"
SHORT_CONV_TO_KDA_SCHEMA = "neural-forge-causal-short-conv-to-kda-fold-v1"
GQA_TO_KDA_SCHEMA = "neural-forge-causal-gqa-to-kda-fold-v1"
KDA_TO_GQA_SCHEMA = "neural-forge-causal-kda-to-gqa-fold-v1"

DESIGN_DOCUMENT = "docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md"


class HFCausalKDAFoldError(ValueError):
    """A KDA fold is unsupported or its evidence is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise HFCausalKDAFoldError("PyTorch is required for KDA folding.") from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalKDAFoldError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalKDAFoldError(f"{name} must be positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise HFCausalKDAFoldError(f"{name} must be positive and finite.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalKDAFoldError(f"{name} must be non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise HFCausalKDAFoldError(f"{name} must be finite and non-negative.")
    return result


def _negative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalKDAFoldError(f"{name} must be negative.") from exc
    if not math.isfinite(result) or result >= 0.0:
        raise HFCausalKDAFoldError(f"{name} must be negative and finite.")
    return result


def _tensor(value: Any, name: str, *, ndim: int | None = None) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise HFCausalKDAFoldError(f"{name} must be a tensor.")
    if ndim is not None and value.ndim != ndim:
        raise HFCausalKDAFoldError(
            f"{name} must be rank {ndim}, got rank {value.ndim}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise HFCausalKDAFoldError(f"{name} contains non-finite values.")
    return result


def _mask(value: Any, name: str, shape: tuple[int, int]) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise HFCausalKDAFoldError(f"{name} must have shape {shape}.")
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


def _relative_rmse(prediction: Any, target: Any) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _retained_energy(prediction: Any, target: Any) -> float:
    error = (prediction - target).square().sum()
    total = target.square().sum().clamp_min(1e-30)
    return float((1.0 - error / total).clamp(min=0.0, max=1.0).item())


def _masked_rows(value: Any, mask: Any) -> Any:
    trailing = tuple(int(size) for size in value.shape[2:])
    return value.reshape(int(mask.numel()), *trailing)[mask.reshape(-1)].contiguous()


def _ridge(source: Any, target: Any, ridge: float) -> Any:
    torch = _torch()
    if source.ndim != 2 or target.ndim != 2 or int(source.shape[0]) != int(
        target.shape[0]
    ):
        raise HFCausalKDAFoldError("Ridge evidence matrices are incompatible.")
    gram = source.T @ source
    return torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=torch.float64),
        source.T @ target,
    )


def _balanced_groups(total: int, count: int) -> tuple[tuple[int, ...], ...]:
    """Contiguous complete head groups, the GQA kernel's pattern (``:121``)."""

    total = _positive_int(total, "group source count")
    count = _positive_int(count, "group target count")
    if count > total:
        raise HFCausalKDAFoldError(
            "KDA head expansion is outside the contraction-only fold."
        )
    quotient, remainder = divmod(total, count)
    groups = []
    cursor = 0
    for index in range(count):
        width = quotient + (1 if index < remainder else 0)
        groups.append(tuple(range(cursor, cursor + width)))
        cursor += width
    return tuple(groups)


# ---------------------------------------------------------------------------
# Geometry, policy, parent carries, evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChimeraKDAGeometry:
    """One KDA operator geometry (``layers.py:91-117``, ``config.py:29-35``)."""

    hidden_size: int
    heads: int
    key_head_dim: int
    value_head_dim: int
    decay_rank: int
    conv_kernel_size: int
    gate_lower_bound: float = -5.0
    rms_norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "heads",
            "key_head_dim",
            "value_head_dim",
            "decay_rank",
            "conv_kernel_size",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "gate_lower_bound",
            _negative_float(self.gate_lower_bound, "gate_lower_bound"),
        )
        object.__setattr__(
            self,
            "rms_norm_epsilon",
            _positive_float(self.rms_norm_epsilon, "rms_norm_epsilon"),
        )
        if self.decay_rank > min(self.hidden_size, self.key_width):
            raise HFCausalKDAFoldError(
                "decay_rank cannot exceed the composed decay map dimensions."
            )

    @property
    def key_width(self) -> int:
        return self.heads * self.key_head_dim

    @property
    def value_width(self) -> int:
        return self.heads * self.value_head_dim

    @property
    def retention_floor(self) -> float:
        return math.exp(self.gate_lower_bound)

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }


@dataclass(frozen=True, slots=True)
class ChimeraKDAFoldPolicy:
    """Frozen policy constants for one KDA fold (design section 4)."""

    ridge: float = 1e-6
    unroll_reconstruction_tolerance: float = 1e-6
    conv_carry_tolerance: float = 1e-5
    decay_chain_tolerance: float = 1e-5
    saturation_epsilon: float = 1e-6
    maximum_saturated_row_fraction: float = 0.05
    maximum_decay_saturated_row_fraction: float = 0.35
    # Dispersion thresholds are meant to be calibrated on the selection
    # partition of a real parent (design recipe step 6).  These defaults admit
    # the spread a chimera *initialization* already carries -- ``decay_log_scale``
    # is drawn from ``log(U(1, 16))`` (``layers.py:111-113``), so two randomly
    # paired heads routinely sit a relative 0.7 apart -- and still refuse the
    # pathological merges.
    maximum_merged_beta_dispersion: float = 0.5
    maximum_merged_decay_dispersion: float = 1.0
    state_drift_growth_cap: float = 4.0
    state_drift_floor: float = 1e-3
    bf16_replay: bool = True

    def __post_init__(self) -> None:
        for name in (
            "ridge",
            "unroll_reconstruction_tolerance",
            "conv_carry_tolerance",
            "decay_chain_tolerance",
            "saturation_epsilon",
            "maximum_saturated_row_fraction",
            "maximum_decay_saturated_row_fraction",
            "maximum_merged_beta_dispersion",
            "maximum_merged_decay_dispersion",
            "state_drift_growth_cap",
            "state_drift_floor",
        ):
            object.__setattr__(
                self, name, _nonnegative_float(getattr(self, name), name)
            )
        if not isinstance(self.bf16_replay, bool):
            raise HFCausalKDAFoldError("bf16_replay must be boolean.")
        if not 0.0 < self.saturation_epsilon < 0.5:
            raise HFCausalKDAFoldError("saturation_epsilon must lie in (0, 0.5).")

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }


@dataclass(frozen=True, slots=True)
class ChimeraKDAParameters:
    """The parent weights a KDA fold carries rather than fits.

    Conv taps, the decay bias, the decay log-scale, the head-norm gain and the
    writer are weights; the evidence record holds rows only.  ``*_conv_weight``
    are ``[width, kernel_size]`` -- the depthwise ``[width, 1, K]`` layout of
    ``nn.Conv1d(groups=channels)`` with its singleton axis dropped.
    """

    query_conv_weight: Any
    query_conv_bias: Any
    key_conv_weight: Any
    key_conv_bias: Any
    value_conv_weight: Any
    value_conv_bias: Any
    decay_bias: Any
    decay_log_scale: Any
    output_norm_weight: Any
    output_weight: Any
    geometry: ChimeraKDAGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, ChimeraKDAGeometry):
            raise HFCausalKDAFoldError("geometry must be a ChimeraKDAGeometry.")
        shapes = {
            "query_conv_weight": (self.geometry.key_width, self.geometry.conv_kernel_size),
            "query_conv_bias": (self.geometry.key_width,),
            "key_conv_weight": (self.geometry.key_width, self.geometry.conv_kernel_size),
            "key_conv_bias": (self.geometry.key_width,),
            "value_conv_weight": (
                self.geometry.value_width,
                self.geometry.conv_kernel_size,
            ),
            "value_conv_bias": (self.geometry.value_width,),
            "decay_bias": (self.geometry.heads, self.geometry.key_head_dim),
            "decay_log_scale": (self.geometry.heads,),
            "output_norm_weight": (self.geometry.heads, self.geometry.value_head_dim),
            "output_weight": (self.geometry.hidden_size, self.geometry.value_width),
        }
        for name, shape in shapes.items():
            value = _tensor(getattr(self, name), name, ndim=len(shape)).clone()
            if tuple(int(size) for size in value.shape) != shape:
                raise HFCausalKDAFoldError(f"{name} must have shape {shape}.")
            object.__setattr__(self, name, value)

    def tensor_items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (descriptor.name, getattr(self, descriptor.name))
            for descriptor in fields(self)
            if descriptor.name != "geometry"
        )


@dataclass(frozen=True, slots=True)
class ChimeraKDALayerEvidence:
    """Complete cached circuit states for one chimera KDA branch.

    Field style mirrors ``HFCausalLayerEvidence``
    (``hf_causal_circuit_evidence.py:214-233``): one sealed capture, rank-3
    ``[batch, seq, .]`` rows unless the field is head-shaped, no live parent.
    """

    index: int
    block_input: Any
    attention_norm_output: Any
    pre_conv_query: Any
    pre_conv_key: Any
    pre_conv_value: Any
    query_rows: Any
    key_rows: Any
    value_rows: Any
    retention_rows: Any
    decay_logit_rows: Any
    beta_rows: Any
    recurrent_output_rows: Any
    gate_rows: Any
    attention_delta: Any
    block_output: Any
    state_snapshots: Any
    state_snapshot_positions: tuple[int, ...]
    geometry: ChimeraKDAGeometry
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or (
            self.index < 0
        ):
            raise HFCausalKDAFoldError(
                "layer evidence index must be a non-negative integer."
            )
        if not isinstance(self.geometry, ChimeraKDAGeometry):
            raise HFCausalKDAFoldError("geometry must be a ChimeraKDAGeometry.")
        geometry = self.geometry
        rank3 = {
            "block_input": geometry.hidden_size,
            "attention_norm_output": geometry.hidden_size,
            "pre_conv_query": geometry.key_width,
            "pre_conv_key": geometry.key_width,
            "pre_conv_value": geometry.value_width,
            "beta_rows": geometry.heads,
            "attention_delta": geometry.hidden_size,
            "block_output": geometry.hidden_size,
        }
        for name, width in rank3.items():
            value = _tensor(getattr(self, name), name, ndim=3).clone()
            if int(value.shape[2]) != width:
                raise HFCausalKDAFoldError(f"{name} must have width {width}.")
            object.__setattr__(self, name, value)
        rank4 = {
            "query_rows": geometry.key_head_dim,
            "key_rows": geometry.key_head_dim,
            "value_rows": geometry.value_head_dim,
            "retention_rows": geometry.key_head_dim,
            "decay_logit_rows": geometry.key_head_dim,
            "recurrent_output_rows": geometry.value_head_dim,
            "gate_rows": geometry.value_head_dim,
        }
        for name, width in rank4.items():
            value = _tensor(getattr(self, name), name, ndim=4).clone()
            if tuple(int(size) for size in value.shape[2:]) != (geometry.heads, width):
                raise HFCausalKDAFoldError(
                    f"{name} must have head shape ({geometry.heads}, {width})."
                )
            object.__setattr__(self, name, value)
        batch, sequence = int(self.block_input.shape[0]), int(self.block_input.shape[1])
        for name in (*rank3, *rank4):
            value = getattr(self, name)
            if tuple(int(size) for size in value.shape[:2]) != (batch, sequence):
                raise HFCausalKDAFoldError(f"{name} has the wrong sequence shape.")
        snapshots = _tensor(self.state_snapshots, "state_snapshots", ndim=5).clone()
        positions = tuple(int(value) for value in self.state_snapshot_positions)
        expected = (
            len(positions),
            batch,
            geometry.heads,
            geometry.key_head_dim,
            geometry.value_head_dim,
        )
        if tuple(int(size) for size in snapshots.shape) != expected:
            raise HFCausalKDAFoldError(f"state_snapshots must have shape {expected}.")
        if any(not 0 <= value < sequence for value in positions) or list(
            positions
        ) != sorted(set(positions)):
            raise HFCausalKDAFoldError(
                "state_snapshot_positions must be strictly increasing in range."
            )
        object.__setattr__(self, "state_snapshots", snapshots)
        object.__setattr__(self, "state_snapshot_positions", positions)
        object.__setattr__(self, "metadata", _report(dict(self.metadata)))

    @property
    def batch(self) -> int:
        return int(self.block_input.shape[0])

    @property
    def sequence(self) -> int:
        return int(self.block_input.shape[1])

    def tensor_items(self) -> tuple[tuple[str, Any], ...]:
        skipped = {"index", "geometry", "metadata", "state_snapshot_positions"}
        return tuple(
            (descriptor.name, getattr(self, descriptor.name))
            for descriptor in fields(self)
            if descriptor.name not in skipped
        )


# ---------------------------------------------------------------------------
# Parent capture (one forward; the fold itself performs zero)
# ---------------------------------------------------------------------------


def chimera_kda_geometry(module: Any, *, hidden_size: int | None = None) -> ChimeraKDAGeometry:
    """Read one chimera ``KimiDeltaAttention`` geometry off the live module."""

    try:
        heads = int(module.num_heads)
        key_head_dim = int(module.key_head_dim)
        value_head_dim = int(module.value_head_dim)
        decay_rank = int(module.decay_down.weight.shape[0])
        kernel_size = int(module.q_conv.kernel_size)
        lower_bound = float(module.gate_lower_bound)
        width = int(hidden_size or module.q_proj.weight.shape[1])
        epsilon = float(module.output_norm.eps)
    except AttributeError as exc:
        raise HFCausalKDAFoldError(
            "module does not expose the chimera KimiDeltaAttention layout."
        ) from exc
    return ChimeraKDAGeometry(
        hidden_size=width,
        heads=heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        decay_rank=decay_rank,
        conv_kernel_size=kernel_size,
        gate_lower_bound=lower_bound,
        rms_norm_epsilon=epsilon,
    )


def chimera_kda_parameters(module: Any, *, geometry: ChimeraKDAGeometry | None = None) -> ChimeraKDAParameters:
    """Seal the parent weights the recipe carries instead of fitting."""

    resolved = geometry or chimera_kda_geometry(module)
    return ChimeraKDAParameters(
        query_conv_weight=module.q_conv.conv.weight.reshape(resolved.key_width, -1),
        query_conv_bias=module.q_conv.conv.bias,
        key_conv_weight=module.k_conv.conv.weight.reshape(resolved.key_width, -1),
        key_conv_bias=module.k_conv.conv.bias,
        value_conv_weight=module.v_conv.conv.weight.reshape(resolved.value_width, -1),
        value_conv_bias=module.v_conv.conv.bias,
        decay_bias=module.decay_bias,
        decay_log_scale=module.decay_log_scale,
        output_norm_weight=module.output_norm.weight,
        output_weight=module.out.weight,
        geometry=resolved,
    )


def capture_chimera_kda_layer_evidence(
    module: Any,
    normalized_rows: Any,
    *,
    layer_index: int,
    block_input: Any | None = None,
    block_output: Any | None = None,
    state_snapshot_stride: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> ChimeraKDALayerEvidence:
    """Run the parent branch exactly once and seal every intermediate.

    The recurrence is stepped with the module's *own* ``delta_rule_update``,
    ``output_norm``, ``output_gate`` and ``out``, so the capture is a trace of
    the parent rather than a re-implementation of it.  The reconstruction is
    checked against ``module(normalized_rows)`` and the residual is sealed.
    """

    torch = _torch()
    geometry = chimera_kda_geometry(module)
    rows = normalized_rows.detach()
    if rows.ndim != 3 or int(rows.shape[2]) != geometry.hidden_size:
        raise HFCausalKDAFoldError(
            "normalized_rows must be [batch, sequence, hidden_size]."
        )
    batch, sequence = int(rows.shape[0]), int(rows.shape[1])
    with torch.no_grad():
        pre_conv_query = module.q_proj(rows)
        pre_conv_key = module.k_proj(rows)
        pre_conv_value = module.v_proj(rows)
        query, key, value, retention, beta = module._project_inputs(rows)
        decay_logits = module.decay_up(module.decay_down(rows)).view(
            batch, sequence, geometry.heads, geometry.key_head_dim
        ).float() + module.decay_bias.view(
            1, 1, geometry.heads, geometry.key_head_dim
        )
        state_dtype = (
            torch.float32
            if rows.dtype in {torch.float16, torch.bfloat16}
            else rows.dtype
        )
        state = torch.zeros(
            batch,
            geometry.heads,
            geometry.key_head_dim,
            geometry.value_head_dim,
            dtype=state_dtype,
        )
        outputs = []
        snapshots = []
        positions = []
        stride = int(state_snapshot_stride)
        for position in range(sequence):
            state = module.delta_rule_update(
                state,
                key[:, position].to(dtype=state_dtype),
                value[:, position].to(dtype=state_dtype),
                retention[:, position].to(dtype=state_dtype),
                beta[:, position].to(dtype=state_dtype),
            )
            outputs.append(
                torch.einsum(
                    "bhk,bhkv->bhv", query[:, position].to(dtype=state_dtype), state
                )
            )
            boundary = position == sequence - 1 or (
                stride > 0 and (position + 1) % stride == 0
            )
            if boundary:
                snapshots.append(state.clone())
                positions.append(position)
        recurrent = torch.stack(outputs, dim=1).to(dtype=rows.dtype)
        normed = module.output_norm(recurrent)
        gate = torch.sigmoid(module.output_gate(rows)).view(
            batch, sequence, geometry.heads, geometry.value_head_dim
        )
        gated = (gate * normed).reshape(batch, sequence, geometry.value_width)
        delta = module.out(gated)
        parent_delta = module(rows)
    reconstruction = _relative_rmse(_tensor(delta, "delta"), _tensor(parent_delta, "parent_delta"))
    if not snapshots:  # pragma: no cover - a zero-length sequence is rejected above
        raise HFCausalKDAFoldError("The capture produced no state snapshot.")
    resolved_input = rows if block_input is None else block_input
    resolved_output = (
        _tensor(resolved_input, "block_input") + _tensor(delta, "attention_delta")
        if block_output is None
        else block_output
    )
    sealed = {
        "schema": EVIDENCE_SCHEMA,
        "component": "chimera_kda_branch_capture",
        "parent_forward_calls": 1,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "capture_reconstruction_relative_rmse": reconstruction,
        "state_snapshot_stride": stride,
        **dict(metadata or {}),
    }
    return ChimeraKDALayerEvidence(
        index=int(layer_index),
        block_input=resolved_input,
        attention_norm_output=rows,
        pre_conv_query=pre_conv_query,
        pre_conv_key=pre_conv_key,
        pre_conv_value=pre_conv_value,
        query_rows=query,
        key_rows=key,
        value_rows=value,
        retention_rows=retention,
        decay_logit_rows=decay_logits,
        beta_rows=beta,
        recurrent_output_rows=recurrent,
        gate_rows=gate,
        attention_delta=delta,
        block_output=resolved_output,
        state_snapshots=torch.stack(snapshots, dim=0),
        state_snapshot_positions=tuple(positions),
        geometry=geometry,
        metadata=sealed,
    )


# ---------------------------------------------------------------------------
# The exact effective-weight unroll -- the evidence-consistency gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KDAWeightUnroll:
    """The exact ``o_t = sum_{s<=t} w_h(t,s) v_s`` identity for one branch."""

    effective_weights: Any
    reconstruction: Any
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _report(dict(self.diagnostics)))

    @property
    def relative_rmse(self) -> float:
        return float(self.diagnostics["reconstruction_relative_rmse"])


def kda_effective_weights(
    evidence: ChimeraKDALayerEvidence,
    *,
    row_mask: Any | None = None,
    tolerance: float = 1e-6,
    fail_closed: bool = True,
) -> KDAWeightUnroll:
    """Unroll the delta recurrence into signed per-(t, s) weights, exactly.

    ``w_h(t,s) = beta_s * q_t^T [prod_{r=s+1..t} (I - beta_r k_r k_r^T)
    Diag(alpha_r)] k_s`` (design section 1.1).  The product is accumulated
    forward over ``t`` on the whole source-position axis at once, which costs
    ``O(T^2 Dk)`` per head and needs **zero parent forwards**: every input is a
    cached row.  Reconstructing the cached ``o_t`` from the weights is the KDA
    evidence-consistency gate, the analogue of the ShortConv FIR exactness
    check.
    """

    torch = _torch()
    if not isinstance(evidence, ChimeraKDALayerEvidence):
        raise HFCausalKDAFoldError("evidence must be a ChimeraKDALayerEvidence.")
    tolerance = _nonnegative_float(tolerance, "tolerance")
    query = evidence.query_rows
    key = evidence.key_rows
    value = evidence.value_rows
    retention = evidence.retention_rows
    beta = evidence.beta_rows
    batch, sequence, heads, key_dim = key.shape
    mask = (
        torch.ones(batch, sequence, dtype=torch.bool)
        if row_mask is None
        else _mask(row_mask, "row_mask", (batch, sequence))
    )
    carry = torch.zeros(batch, heads, sequence, key_dim, dtype=torch.float64)
    weights = torch.zeros(batch, heads, sequence, sequence, dtype=torch.float64)
    for position in range(sequence):
        decayed = carry * retention[:, position].unsqueeze(2)
        current_key = key[:, position]
        projection = torch.einsum("bhsk,bhk->bhs", decayed, current_key)
        write = beta[:, position]
        decayed = decayed - write.unsqueeze(-1).unsqueeze(-1) * projection.unsqueeze(
            -1
        ) * current_key.unsqueeze(2)
        decayed[:, :, position] = write.unsqueeze(-1) * current_key
        carry = decayed
        weights[:, :, position, :] = torch.einsum(
            "bhsk,bhk->bhs", carry, query[:, position]
        )
    reconstruction = torch.einsum("bhts,bshv->bthv", weights, value)
    target = evidence.recurrent_output_rows
    selected_prediction = _masked_rows(reconstruction, mask)
    selected_target = _masked_rows(target, mask)
    relative = _relative_rmse(selected_prediction, selected_target)
    maximum = float((reconstruction - target).abs().max().item())
    causal = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    leakage = float(weights.masked_select(~causal.reshape(1, 1, sequence, sequence)).abs().max().item()) if sequence > 1 else 0.0
    diagnostics = {
        "schema": UNROLL_SCHEMA,
        "component": "kda_effective_weight_unroll",
        "layer_index": int(evidence.index),
        "batch": batch,
        "sequence": sequence,
        "heads": heads,
        "key_head_dim": key_dim,
        "value_head_dim": int(value.shape[3]),
        "evidence_rows": int(mask.sum().item()),
        "parent_forward_calls": 0,
        "reconstruction_relative_rmse": relative,
        "reconstruction_maximum_absolute_error": maximum,
        "acausal_weight_maximum_absolute_value": leakage,
        "tolerance": tolerance,
        "within_tolerance": bool(relative <= tolerance),
        "mean_absolute_weight": float(weights.abs().mean().item()),
        "row_mass_mean": float(weights.sum(dim=-1).mean().item()),
        "row_mass_standard_deviation": float(weights.sum(dim=-1).std().item()),
        "effective_weights_sha256": _tensor_sha256(weights),
    }
    if fail_closed and relative > tolerance:
        raise HFCausalKDAFoldError(
            "The KDA effective-weight unroll does not reconstruct the cached "
            f"recurrent output ({relative:.3e} > {tolerance:.3e}); the evidence "
            "is inconsistent and the fold fails closed."
        )
    return KDAWeightUnroll(
        effective_weights=weights,
        reconstruction=reconstruction,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Head assignment and channel selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KDAHeadAssignment:
    """Barycentric complete head contraction scored on gated-output energy."""

    source_head_groups: tuple[tuple[int, ...], ...]
    barycentric_weights: tuple[tuple[float, ...], ...]
    gated_output_energy: tuple[float, ...]

    def __post_init__(self) -> None:
        groups = tuple(tuple(int(item) for item in group) for group in self.source_head_groups)
        weights = tuple(tuple(float(item) for item in group) for group in self.barycentric_weights)
        if not groups or any(not group for group in groups):
            raise HFCausalKDAFoldError("source_head_groups cannot be empty.")
        if len(weights) != len(groups) or any(
            len(weight) != len(group) for weight, group in zip(weights, groups)
        ):
            raise HFCausalKDAFoldError("barycentric weights must match the groups.")
        for weight in weights:
            if abs(sum(weight) - 1.0) > 1e-9 or any(value < 0.0 for value in weight):
                raise HFCausalKDAFoldError(
                    "barycentric weights must be a convex combination."
                )
        object.__setattr__(self, "source_head_groups", groups)
        object.__setattr__(self, "barycentric_weights", weights)
        object.__setattr__(
            self,
            "gated_output_energy",
            tuple(float(value) for value in self.gated_output_energy),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_head_groups": [list(group) for group in self.source_head_groups],
            "barycentric_weights": [list(group) for group in self.barycentric_weights],
            "gated_output_energy": list(self.gated_output_energy),
        }


def _head_assignment(
    evidence: ChimeraKDALayerEvidence,
    target_heads: int,
    *,
    row_mask: Any,
) -> KDAHeadAssignment:
    torch = _torch()
    source_heads = evidence.geometry.heads
    groups = _balanced_groups(source_heads, target_heads)
    gated = evidence.gate_rows * evidence.recurrent_output_rows
    rows = _masked_rows(gated, row_mask)
    energy = rows.square().sum(dim=(0, 2))
    weights = []
    for group in groups:
        block = torch.tensor([float(energy[index].item()) for index in group], dtype=torch.float64)
        total = float(block.sum().item())
        if total <= 0.0:
            weights.append(tuple(1.0 / len(group) for _ in group))
        else:
            weights.append(tuple(float(value.item()) / total for value in block))
    return KDAHeadAssignment(
        source_head_groups=groups,
        barycentric_weights=tuple(weights),
        gated_output_energy=tuple(float(value.item()) for value in energy),
    )


def _merge_heads(value: Any, assignment: KDAHeadAssignment, *, axis: int = -2) -> Any:
    """Barycentric contraction over one explicit head axis."""

    torch = _torch()
    dimension = axis % value.ndim
    merged = []
    for group, weights in zip(
        assignment.source_head_groups, assignment.barycentric_weights
    ):
        block = None
        for index, weight in zip(group, weights):
            piece = value.select(dimension, index) * weight
            block = piece if block is None else block + piece
        merged.append(block)
    return torch.stack(merged, dim=dimension).contiguous()


def _merge_head_scalars(value: Any, assignment: KDAHeadAssignment) -> Any:
    """Barycentric contraction over a trailing scalar head axis."""

    return _merge_heads(value, assignment, axis=-1)


def _group_dispersion(
    value: Any, assignment: KDAHeadAssignment, *, axis: int = -2
) -> float:
    """Largest within-group relative spread of a head-indexed signal."""

    torch = _torch()
    dimension = axis % value.ndim
    worst = 0.0
    for group in assignment.source_head_groups:
        if len(group) == 1:
            continue
        stacked = torch.stack(
            [value.select(dimension, index) for index in group], dim=0
        )
        spread = stacked.std(dim=0, unbiased=False).mean()
        scale = stacked.abs().mean().clamp_min(1e-30)
        worst = max(worst, float((spread / scale).item()))
    return worst


def _channel_pivots(basis: Any, count: int) -> tuple[tuple[int, ...], bool]:
    """Greedy row-pivoted Gram-Schmidt over channels (a QDEIM on ``basis``).

    ``basis`` is ``[channels, modes]`` with modes in importance order.  Ties go
    to the lowest channel index, so the result is deterministic.  The boolean
    reports whether the pivot sequence stayed numerically independent for the
    full count; a rank-deficient tail is completed by residual energy.
    """

    torch = _torch()
    channels = int(basis.shape[0])
    count = _positive_int(count, "channel count")
    if count > channels:
        raise HFCausalKDAFoldError("Channel expansion is outside the fold.")
    residual = basis.clone()
    selected: list[int] = []
    independent = True
    for _ in range(count):
        norms = residual.square().sum(dim=1)
        for index in selected:
            norms[index] = -1.0
        index = int(torch.argmax(norms).item())
        if float(norms[index].item()) <= 1e-24:
            independent = False
            energy = basis.square().sum(dim=1)
            for taken in selected:
                energy[taken] = -1.0
            index = int(torch.argmax(energy).item())
            selected.append(index)
            continue
        selected.append(index)
        row = residual[index]
        residual = residual - torch.outer(residual @ row, row) / row.dot(row)
    return tuple(sorted(selected)), independent


@dataclass(frozen=True, slots=True)
class KDAChannelSelection:
    """Channel-aligned key/value cuts, ranked in balanced coordinates."""

    key_channels: tuple[tuple[int, ...], ...]
    value_channels: tuple[tuple[int, ...], ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("key_channels", "value_channels"):
            groups = tuple(
                tuple(int(item) for item in group) for group in getattr(self, name)
            )
            if not groups or any(not group for group in groups):
                raise HFCausalKDAFoldError(f"{name} cannot be empty.")
            if any(len(set(group)) != len(group) for group in groups):
                raise HFCausalKDAFoldError(f"{name} must be distinct per head.")
            object.__setattr__(self, name, groups)
        object.__setattr__(self, "report", _report(dict(self.report)))


def select_kda_channels(
    evidence: ChimeraKDALayerEvidence,
    parameters: ChimeraKDAParameters,
    geometry: ChimeraKDAGeometry,
    *,
    assignment: KDAHeadAssignment,
    row_mask: Any,
    policy: ChimeraKDAFoldPolicy | None = None,
) -> KDAChannelSelection:
    """Rank state channels in balanced coordinates, cut them channel-aligned.

    Key axis: the controllability sketch is the cached state-writing rows
    ``sqrt(beta_s) k_s`` plus the cached chunk-boundary state snapshots read in
    key coordinates; the observability sketch is the cached ``q_t`` rows
    weighted by the downstream gate/norm/writer sensitivity.  Value axis:
    channel-aligned selection ranked by observability through the ``W_out``
    columns.  Neither axis is rotated -- ``Diag(alpha)`` must stay diagonal and
    the head-norm gain must stay a vector (design section 1.1).
    """

    torch = _torch()
    policy = policy or ChimeraKDAFoldPolicy()
    source = evidence.geometry
    if geometry.key_head_dim > source.key_head_dim or geometry.value_head_dim > (
        source.value_head_dim
    ):
        raise HFCausalKDAFoldError(
            "KDA channel expansion is outside the contraction-only fold."
        )
    key_rows = _masked_rows(evidence.key_rows, row_mask)
    query_rows = _masked_rows(evidence.query_rows, row_mask)
    value_rows = _masked_rows(evidence.value_rows, row_mask)
    beta_rows = _masked_rows(evidence.beta_rows, row_mask)
    gate_rows = _masked_rows(evidence.gate_rows, row_mask)
    writer = parameters.output_weight
    key_channels: list[tuple[int, ...]] = []
    value_channels: list[tuple[int, ...]] = []
    diagnostics: list[dict[str, Any]] = []
    for target_head, group in enumerate(assignment.source_head_groups):
        weights = assignment.barycentric_weights[target_head]
        write_rows = torch.cat(
            [
                (beta_rows[:, index].clamp_min(0.0).sqrt().unsqueeze(-1))
                * key_rows[:, index]
                * math.sqrt(weight)
                for index, weight in zip(group, weights)
            ],
            dim=0,
        )
        snapshot_rows = torch.cat(
            [
                evidence.state_snapshots[:, :, index]
                .reshape(-1, source.key_head_dim, source.value_head_dim)
                .permute(0, 2, 1)
                .reshape(-1, source.key_head_dim)
                * math.sqrt(weight)
                for index, weight in zip(group, weights)
            ],
            dim=0,
        )
        head_sensitivity = []
        for index, weight in zip(group, weights):
            block = writer[
                :,
                index * source.value_head_dim : (index + 1) * source.value_head_dim,
            ]
            channel_gain = (
                parameters.output_norm_weight[index].abs()
                * block.square().sum(dim=0).sqrt()
            )
            scale = (gate_rows[:, index] * channel_gain.reshape(1, -1)).square().sum(
                dim=1
            ).sqrt()
            head_sensitivity.append(query_rows[:, index] * scale.unsqueeze(-1) * weight)
        observability = torch.cat(head_sensitivity, dim=0)
        selection_method = "balanced_krylov_pivot_qdeim"
        fallback_reason: str | None = None
        hankel: list[float] = []
        canonical: list[int] = []
        if geometry.key_head_dim == source.key_head_dim:
            selected = tuple(range(source.key_head_dim))
            independent = True
            selection_method = "identity"
        else:
            try:
                basis = build_balanced_krylov_basis(
                    write_rows,
                    geometry.key_head_dim,
                    forward_responses=snapshot_rows,
                    output_sensitivities=observability,
                    center=False,
                )
                decoder = torch.tensor(basis.decoder, dtype=torch.float64)
                hankel = [float(value) for value in basis.report.hankel_singular_values]
                canonical = [int(value) for value in basis.report.canonical_pivot_indices]
            except BalancedKrylovError as exc:
                selection_method = "observability_weighted_energy_qdeim"
                fallback_reason = str(exc)
                decoder = torch.cat([write_rows, observability], dim=0).T.contiguous()
            selected, independent = _channel_pivots(decoder, geometry.key_head_dim)
        key_channels.append(selected)
        value_scale = []
        for index, weight in zip(group, weights):
            block = writer[
                :,
                index * source.value_head_dim : (index + 1) * source.value_head_dim,
            ]
            value_scale.append(
                weight
                * parameters.output_norm_weight[index].abs()
                * block.square().sum(dim=0).sqrt()
                * gate_rows[:, index].abs().mean(dim=0)
            )
        column_scale = torch.stack(value_scale, dim=0).sum(dim=0)
        weighted_value = torch.cat(
            [value_rows[:, index] * math.sqrt(weight) for index, weight in zip(group, weights)],
            dim=0,
        ) * column_scale.reshape(1, -1)
        if geometry.value_head_dim == source.value_head_dim:
            value_selected = tuple(range(source.value_head_dim))
            value_independent = True
        else:
            value_selected, value_independent = _channel_pivots(
                weighted_value.T.contiguous(), geometry.value_head_dim
            )
        value_channels.append(value_selected)
        diagnostics.append(
            {
                "target_head": target_head,
                "source_heads": list(group),
                "key_channels": list(selected),
                "value_channels": list(value_selected),
                "key_selection_method": selection_method,
                "key_selection_fallback_reason": fallback_reason,
                "key_pivots_independent": bool(independent),
                "value_pivots_independent": bool(value_independent),
                "balanced_hankel_singular_values": hankel,
                "balanced_canonical_pivot_indices": canonical,
                "retained_key_write_energy_fraction": float(
                    (
                        write_rows[:, list(selected)].square().sum()
                        / write_rows.square().sum().clamp_min(1e-30)
                    ).item()
                ),
                "retained_value_observable_energy_fraction": float(
                    (
                        weighted_value[:, list(value_selected)].square().sum()
                        / weighted_value.square().sum().clamp_min(1e-30)
                    ).item()
                ),
            }
        )
    body = {
        "schema": SCHEMA,
        "component": "kda_channel_selection",
        "layer_index": int(evidence.index),
        "rotation_applied": False,
        "selection_is_channel_aligned": True,
        "source_key_head_dim": source.key_head_dim,
        "source_value_head_dim": source.value_head_dim,
        "target_key_head_dim": geometry.key_head_dim,
        "target_value_head_dim": geometry.value_head_dim,
        "evidence_rows": int(row_mask.sum().item()),
        "parent_forward_calls": 0,
        "per_head": diagnostics,
        "policy": policy.as_dict(),
    }
    return KDAChannelSelection(
        key_channels=tuple(key_channels),
        value_channels=tuple(value_channels),
        report=body,
    )


# ---------------------------------------------------------------------------
# The child operator
# ---------------------------------------------------------------------------


def _causal_depthwise_conv(rows: Any, weight: Any, bias: Any, kernel_size: int) -> Any:
    """Left-padded depthwise FIR + bias + SiLU (``layers.py:60-74``)."""

    torch = _torch()
    channels = int(rows.shape[2])
    padded = torch.nn.functional.pad(
        rows.transpose(1, 2), (kernel_size - 1, 0)
    )
    windows = padded.unfold(-1, kernel_size, 1)
    filtered = (windows * weight.reshape(1, channels, 1, kernel_size)).sum(dim=-1)
    activated = filtered.transpose(1, 2) + bias.reshape(1, 1, channels)
    return activated * torch.sigmoid(activated)


def _l2_normalize(value: Any) -> Any:
    return value / value.square().sum(dim=-1, keepdim=True).clamp_min(1e-30).sqrt()


def _logit(probability: Any, epsilon: float) -> tuple[Any, float]:
    torch = _torch()
    clamped = probability.clamp(min=epsilon, max=1.0 - epsilon)
    saturated = float(
        (probability != clamped).to(dtype=torch.float64).mean().item()
    )
    return torch.log(clamped) - torch.log1p(-clamped), saturated


@dataclass(frozen=True, slots=True)
class HFCausalKDAFold:
    """One compiled chimera KDA child branch plus its sealed receipt."""

    query_weight: Any
    key_weight: Any
    value_weight: Any
    query_conv_weight: Any
    query_conv_bias: Any
    key_conv_weight: Any
    key_conv_bias: Any
    value_conv_weight: Any
    value_conv_bias: Any
    decay_down_weight: Any
    decay_up_weight: Any
    decay_bias: Any
    decay_log_scale: Any
    write_strength_weight: Any
    output_gate_weight: Any
    output_norm_weight: Any
    output_weight: Any
    geometry: ChimeraKDAGeometry
    assignment: KDAHeadAssignment
    selection: KDAChannelSelection
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        geometry = self.geometry
        if not isinstance(geometry, ChimeraKDAGeometry):
            raise HFCausalKDAFoldError("geometry must be a ChimeraKDAGeometry.")
        shapes = {
            "query_weight": (geometry.key_width, geometry.hidden_size),
            "key_weight": (geometry.key_width, geometry.hidden_size),
            "value_weight": (geometry.value_width, geometry.hidden_size),
            "query_conv_weight": (geometry.key_width, geometry.conv_kernel_size),
            "query_conv_bias": (geometry.key_width,),
            "key_conv_weight": (geometry.key_width, geometry.conv_kernel_size),
            "key_conv_bias": (geometry.key_width,),
            "value_conv_weight": (geometry.value_width, geometry.conv_kernel_size),
            "value_conv_bias": (geometry.value_width,),
            "decay_down_weight": (geometry.decay_rank, geometry.hidden_size),
            "decay_up_weight": (geometry.key_width, geometry.decay_rank),
            "decay_bias": (geometry.heads, geometry.key_head_dim),
            "decay_log_scale": (geometry.heads,),
            "write_strength_weight": (geometry.heads, geometry.hidden_size),
            "output_gate_weight": (geometry.value_width, geometry.hidden_size),
            "output_norm_weight": (geometry.heads, geometry.value_head_dim),
        }
        for name, shape in shapes.items():
            value = _tensor(getattr(self, name), name, ndim=len(shape)).clone()
            if tuple(int(size) for size in value.shape) != shape:
                raise HFCausalKDAFoldError(f"{name} must have shape {shape}.")
            object.__setattr__(self, name, value)
        writer = _tensor(self.output_weight, "output_weight", ndim=2).clone()
        if int(writer.shape[1]) != geometry.value_width:
            raise HFCausalKDAFoldError("output_weight has the wrong input width.")
        object.__setattr__(self, "output_weight", writer)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def project(self, child_rows: Any) -> tuple[Any, Any, Any, Any, Any]:
        """Reproduce ``_project_inputs`` for the compiled child."""

        torch = _torch()
        geometry = self.geometry
        rows = _tensor(child_rows, "child_rows", ndim=3)
        if int(rows.shape[2]) != geometry.hidden_size:
            raise HFCausalKDAFoldError("Child replay width differs from the geometry.")
        batch, sequence = int(rows.shape[0]), int(rows.shape[1])

        def head_view(flat: Any, dim: int) -> Any:
            return flat.reshape(batch, sequence, geometry.heads, dim)

        query = head_view(
            _causal_depthwise_conv(
                rows @ self.query_weight.T,
                self.query_conv_weight,
                self.query_conv_bias,
                geometry.conv_kernel_size,
            ),
            geometry.key_head_dim,
        )
        key = head_view(
            _causal_depthwise_conv(
                rows @ self.key_weight.T,
                self.key_conv_weight,
                self.key_conv_bias,
                geometry.conv_kernel_size,
            ),
            geometry.key_head_dim,
        )
        value = head_view(
            _causal_depthwise_conv(
                rows @ self.value_weight.T,
                self.value_conv_weight,
                self.value_conv_bias,
                geometry.conv_kernel_size,
            ),
            geometry.value_head_dim,
        )
        query = _l2_normalize(query)
        key = _l2_normalize(key)
        logits = head_view(
            (rows @ self.decay_down_weight.T) @ self.decay_up_weight.T,
            geometry.key_head_dim,
        ) + self.decay_bias.reshape(1, 1, geometry.heads, geometry.key_head_dim)
        scale = self.decay_log_scale.exp().reshape(1, 1, geometry.heads, 1)
        retention = torch.exp(geometry.gate_lower_bound * torch.sigmoid(scale * logits))
        beta = torch.sigmoid(rows @ self.write_strength_weight.T)
        return query, key, value, retention, beta

    def recurrence(
        self,
        child_rows: Any,
        *,
        snapshot_positions: Sequence[int] = (),
    ) -> tuple[Any, Any]:
        """Run the installed child recurrence; return outputs and snapshots."""

        torch = _torch()
        geometry = self.geometry
        query, key, value, retention, beta = self.project(child_rows)
        batch, sequence = int(query.shape[0]), int(query.shape[1])
        state = torch.zeros(
            batch,
            geometry.heads,
            geometry.key_head_dim,
            geometry.value_head_dim,
            dtype=torch.float64,
        )
        wanted = tuple(int(value_) for value_ in snapshot_positions)
        outputs = []
        snapshots = []
        for position in range(sequence):
            decayed = retention[:, position].unsqueeze(-1) * state
            prediction = torch.einsum("bhk,bhkv->bhv", key[:, position], decayed)
            error = value[:, position] - prediction
            state = decayed + beta[:, position].unsqueeze(-1).unsqueeze(
                -1
            ) * key[:, position].unsqueeze(-1) * error.unsqueeze(-2)
            outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, position], state))
            if position in wanted:
                snapshots.append(state.clone())
        recurrent = torch.stack(outputs, dim=1)
        stacked = (
            torch.stack(snapshots, dim=0)
            if snapshots
            else torch.zeros(
                0,
                batch,
                geometry.heads,
                geometry.key_head_dim,
                geometry.value_head_dim,
                dtype=torch.float64,
            )
        )
        return recurrent, stacked

    def gated_context(self, child_rows: Any, recurrent: Any) -> Any:
        """Head-norm, gate, and flatten -- the writer's input rows."""

        torch = _torch()
        geometry = self.geometry
        rows = _tensor(child_rows, "child_rows", ndim=3)
        batch, sequence = int(rows.shape[0]), int(rows.shape[1])
        scale = (
            recurrent.square().mean(dim=-1, keepdim=True) + geometry.rms_norm_epsilon
        ).rsqrt()
        normed = recurrent * scale * self.output_norm_weight.reshape(
            1, 1, geometry.heads, geometry.value_head_dim
        )
        gate = torch.sigmoid(rows @ self.output_gate_weight.T).reshape(
            batch, sequence, geometry.heads, geometry.value_head_dim
        )
        return (gate * normed).reshape(batch, sequence, geometry.value_width)

    def replay(self, child_rows: Any) -> tuple[Any, Any, Any]:
        """Replay the whole installed branch in FP64."""

        recurrent, snapshots = self.recurrence(child_rows)
        context = self.gated_context(child_rows, recurrent)
        return (context @ self.output_weight.T).contiguous(), recurrent, snapshots

    def child_state_dict(self) -> dict[str, Any]:
        """Emit the compiled child in chimera ``KimiDeltaAttention`` names."""

        geometry = self.geometry
        return {
            "q_proj.weight": self.query_weight,
            "k_proj.weight": self.key_weight,
            "v_proj.weight": self.value_weight,
            "q_conv.conv.weight": self.query_conv_weight.reshape(
                geometry.key_width, 1, geometry.conv_kernel_size
            ),
            "q_conv.conv.bias": self.query_conv_bias,
            "k_conv.conv.weight": self.key_conv_weight.reshape(
                geometry.key_width, 1, geometry.conv_kernel_size
            ),
            "k_conv.conv.bias": self.key_conv_bias,
            "v_conv.conv.weight": self.value_conv_weight.reshape(
                geometry.value_width, 1, geometry.conv_kernel_size
            ),
            "v_conv.conv.bias": self.value_conv_bias,
            "decay_down.weight": self.decay_down_weight,
            "decay_up.weight": self.decay_up_weight,
            "decay_bias": self.decay_bias,
            "decay_log_scale": self.decay_log_scale,
            "write_strength.weight": self.write_strength_weight,
            "output_gate.weight": self.output_gate_weight,
            "output_norm.weight": self.output_norm_weight,
            "out.weight": self.output_weight,
        }


def replay_kda_branch(fold: HFCausalKDAFold, child_rows: Any, *, dtype: Any = None) -> Any:
    """Replay one compiled KDA branch in FP64 or with BF16 intermediates."""

    torch = _torch()
    if not isinstance(fold, HFCausalKDAFold):
        raise HFCausalKDAFoldError("fold must be an HFCausalKDAFold.")
    if dtype is None or dtype == torch.float64:
        delta, _recurrent, _snapshots = fold.replay(child_rows)
        return delta
    if dtype != torch.bfloat16:
        raise HFCausalKDAFoldError("Only FP64 and BF16 replay dtypes are supported.")
    return _replay_kda_bf16(fold, _tensor(child_rows, "child_rows", ndim=3))


def _replay_kda_bf16(fold: HFCausalKDAFold, child_rows: Any) -> Any:
    """Replay conv, normalize, recurrence, gate and writer all in BF16."""

    torch = _torch()
    geometry = fold.geometry
    rows = child_rows.to(dtype=torch.bfloat16)
    batch, sequence = int(rows.shape[0]), int(rows.shape[1])

    def project(weight: Any) -> Any:
        return rows @ weight.to(dtype=torch.bfloat16).transpose(0, 1)

    def conv(flat: Any, weight: Any, bias: Any, dim: int) -> Any:
        padded = torch.nn.functional.pad(
            flat.transpose(1, 2), (geometry.conv_kernel_size - 1, 0)
        )
        windows = padded.unfold(-1, geometry.conv_kernel_size, 1)
        taps = weight.to(dtype=torch.bfloat16).reshape(
            1, int(flat.shape[2]), 1, geometry.conv_kernel_size
        )
        filtered = (windows * taps).sum(dim=-1).transpose(1, 2)
        activated = filtered + bias.to(dtype=torch.bfloat16).reshape(1, 1, -1)
        activated = activated * torch.sigmoid(activated)
        return activated.reshape(batch, sequence, geometry.heads, dim)

    query = conv(
        project(fold.query_weight),
        fold.query_conv_weight,
        fold.query_conv_bias,
        geometry.key_head_dim,
    )
    key = conv(
        project(fold.key_weight),
        fold.key_conv_weight,
        fold.key_conv_bias,
        geometry.key_head_dim,
    )
    value = conv(
        project(fold.value_weight),
        fold.value_conv_weight,
        fold.value_conv_bias,
        geometry.value_head_dim,
    )
    query = query / query.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    key = key / key.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    logits = (
        project(fold.decay_down_weight) @ fold.decay_up_weight.to(
            dtype=torch.bfloat16
        ).transpose(0, 1)
    ).reshape(batch, sequence, geometry.heads, geometry.key_head_dim)
    logits = logits + fold.decay_bias.to(dtype=torch.bfloat16).reshape(
        1, 1, geometry.heads, geometry.key_head_dim
    )
    scale = fold.decay_log_scale.to(dtype=torch.bfloat16).exp().reshape(
        1, 1, geometry.heads, 1
    )
    retention = torch.exp(
        torch.tensor(geometry.gate_lower_bound, dtype=torch.bfloat16)
        * torch.sigmoid(scale * logits)
    )
    beta = torch.sigmoid(project(fold.write_strength_weight))
    state = torch.zeros(
        batch,
        geometry.heads,
        geometry.key_head_dim,
        geometry.value_head_dim,
        dtype=torch.bfloat16,
    )
    outputs = []
    for position in range(sequence):
        decayed = retention[:, position].unsqueeze(-1) * state
        prediction = torch.einsum("bhk,bhkv->bhv", key[:, position], decayed)
        error = value[:, position] - prediction
        state = decayed + beta[:, position].unsqueeze(-1).unsqueeze(-1) * key[
            :, position
        ].unsqueeze(-1) * error.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, position], state))
    recurrent = torch.stack(outputs, dim=1)
    norm_scale = (
        recurrent.float().square().mean(dim=-1, keepdim=True)
        + geometry.rms_norm_epsilon
    ).rsqrt().to(dtype=torch.bfloat16)
    normed = recurrent * norm_scale * fold.output_norm_weight.to(
        dtype=torch.bfloat16
    ).reshape(1, 1, geometry.heads, geometry.value_head_dim)
    gate = torch.sigmoid(project(fold.output_gate_weight)).reshape(
        batch, sequence, geometry.heads, geometry.value_head_dim
    )
    gated = (gate * normed).reshape(batch, sequence, geometry.value_width)
    delta = gated @ fold.output_weight.to(dtype=torch.bfloat16).transpose(0, 1)
    return delta.to(dtype=torch.float64)


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------


def compile_hf_causal_kda_fold(
    evidence: ChimeraKDALayerEvidence,
    parameters: ChimeraKDAParameters,
    geometry: ChimeraKDAGeometry,
    *,
    fit_token_mask: Any,
    validation_token_mask: Any,
    valid_token_mask: Any | None = None,
    child_normalized_rows: Any | None = None,
    endpoint_target: Any | None = None,
    policy: ChimeraKDAFoldPolicy | None = None,
    unroll: KDAWeightUnroll | None = None,
) -> HFCausalKDAFold:
    """Compile one chimera KDA branch into a smaller KDA branch, closed form.

    Recipe steps 1-9 of design section 1.1, in order: unroll gate, barycentric
    head contraction, channel selection, Q/K refit against pre-conv targets
    with the child's own conv/SiLU/L2 replayed, closed-form decay refit through
    the inverse chain, beta refit, installed-recurrence replay, gate and
    endpoint-coupled writer solve, then BF16 replay and diagnostics.

    ``child_normalized_rows`` may be narrower than the parent: every consumer
    fit is a ridge from child rows, which is what makes the cross-width chart
    lane target construction rather than a new solver (design section 2).
    """

    torch = _torch()
    if not isinstance(evidence, ChimeraKDALayerEvidence):
        raise HFCausalKDAFoldError("evidence must be a ChimeraKDALayerEvidence.")
    if not isinstance(parameters, ChimeraKDAParameters):
        raise HFCausalKDAFoldError("parameters must be a ChimeraKDAParameters.")
    if not isinstance(geometry, ChimeraKDAGeometry):
        raise HFCausalKDAFoldError("geometry must be a ChimeraKDAGeometry.")
    if parameters.geometry != evidence.geometry:
        raise HFCausalKDAFoldError(
            "The parent parameter carry and the evidence describe different "
            "geometries."
        )
    policy = policy or ChimeraKDAFoldPolicy()
    source = evidence.geometry
    if geometry.conv_kernel_size != source.conv_kernel_size:
        raise HFCausalKDAFoldError(
            "Conv-tap carry requires the child kernel size to equal the "
            "parent's; a different kernel is a kind translation, not a fold."
        )
    batch, sequence = evidence.batch, evidence.sequence
    child = (
        evidence.attention_norm_output
        if child_normalized_rows is None
        else _tensor(child_normalized_rows, "child_normalized_rows", ndim=3)
    )
    if tuple(int(size) for size in child.shape[:2]) != (batch, sequence):
        raise HFCausalKDAFoldError("child rows have the wrong sequence shape.")
    if int(child.shape[2]) != geometry.hidden_size:
        raise HFCausalKDAFoldError("child rows do not match the child hidden size.")
    delta_target = (
        evidence.attention_delta
        if endpoint_target is None
        else _tensor(endpoint_target, "endpoint_target", ndim=3)
    )
    if tuple(int(size) for size in delta_target.shape[:2]) != (batch, sequence):
        raise HFCausalKDAFoldError("endpoint target has the wrong sequence shape.")
    fit = _mask(fit_token_mask, "fit_token_mask", (batch, sequence))
    validation = _mask(validation_token_mask, "validation_token_mask", (batch, sequence))
    valid = (
        (fit | validation)
        if valid_token_mask is None
        else _mask(valid_token_mask, "valid_token_mask", (batch, sequence))
    )
    if (
        bool((fit & validation).any().item())
        or not bool(fit.any().item())
        or not bool(validation.any().item())
        or bool(((fit | validation) & ~valid).any().item())
    ):
        raise HFCausalKDAFoldError(
            "KDA fit/validation masks must be disjoint nonempty valid subsets."
        )

    # Step 1 -- the exact unroll, the evidence-consistency gate.
    unroll = unroll or kda_effective_weights(
        evidence,
        row_mask=valid,
        tolerance=policy.unroll_reconstruction_tolerance,
    )
    if not bool(unroll.diagnostics["within_tolerance"]):
        raise HFCausalKDAFoldError(
            "The supplied unroll is outside its own reconstruction tolerance."
        )

    # Evidence-consistency gate 2: the carried conv taps reproduce the cached
    # post-conv rows, and gate 3: the decay inverse chain closes.
    replayed_query = _l2_normalize(
        _causal_depthwise_conv(
            evidence.pre_conv_query,
            parameters.query_conv_weight,
            parameters.query_conv_bias,
            source.conv_kernel_size,
        ).reshape(batch, sequence, source.heads, source.key_head_dim)
    )
    replayed_value = _causal_depthwise_conv(
        evidence.pre_conv_value,
        parameters.value_conv_weight,
        parameters.value_conv_bias,
        source.conv_kernel_size,
    ).reshape(batch, sequence, source.heads, source.value_head_dim)
    conv_carry_error = max(
        _relative_rmse(_masked_rows(replayed_query, valid), _masked_rows(evidence.query_rows, valid)),
        _relative_rmse(_masked_rows(replayed_value, valid), _masked_rows(evidence.value_rows, valid)),
    )
    if conv_carry_error > policy.conv_carry_tolerance:
        raise HFCausalKDAFoldError(
            "The carried depthwise conv taps do not reproduce the cached q/v "
            f"rows ({conv_carry_error:.3e} > {policy.conv_carry_tolerance:.3e})."
        )
    # ``u = inverse_sigmoid(g*/g_min)`` is not inverted numerically: the parent
    # caches the pre-scale logits, so ``u = exp(A_h) * z`` identically.  The
    # chain is *checked* in probability space, where the comparison is
    # well-conditioned; inverting a saturated sigmoid to check it would measure
    # the conditioning of the check rather than the consistency of the cache.
    chain_target = evidence.decay_logit_rows * parameters.decay_log_scale.exp().reshape(
        1, 1, source.heads, 1
    )
    decay_probability = torch.log(
        evidence.retention_rows.clamp_min(1e-300)
    ) / source.gate_lower_bound
    decay_chain_error = _relative_rmse(
        _masked_rows(torch.sigmoid(chain_target), valid),
        _masked_rows(decay_probability, valid),
    )
    if decay_chain_error > policy.decay_chain_tolerance:
        raise HFCausalKDAFoldError(
            "The cached retention and decay logits disagree through the inverse "
            f"chain ({decay_chain_error:.3e} > {policy.decay_chain_tolerance:.3e})."
        )
    _clamped_decay, decay_saturated = _logit(
        decay_probability, policy.saturation_epsilon
    )
    if decay_saturated > policy.maximum_decay_saturated_row_fraction:
        raise HFCausalKDAFoldError(
            f"The parent decay sits at a numerical bound on {decay_saturated:.4f} "
            "of rows, above the sealed clamp policy "
            f"({policy.maximum_decay_saturated_row_fraction:.4f})."
        )

    # Step 2 -- barycentric complete head contraction on gated-output energy.
    assignment = _head_assignment(evidence, geometry.heads, row_mask=fit)

    # Step 3 -- channel selection, ranked balanced, cut channel-aligned.
    selection = select_kda_channels(
        evidence,
        parameters,
        geometry,
        assignment=assignment,
        row_mask=fit,
        policy=policy,
    )

    def gather_key(value: Any) -> Any:
        merged = _merge_heads(value, assignment)
        return torch.stack(
            [
                merged[..., head, list(channels)]
                for head, channels in enumerate(selection.key_channels)
            ],
            dim=-2,
        )

    def gather_value(value: Any) -> Any:
        merged = _merge_heads(value, assignment)
        return torch.stack(
            [
                merged[..., head, list(channels)]
                for head, channels in enumerate(selection.value_channels)
            ],
            dim=-2,
        )

    def flatten(value: Any) -> Any:
        return value.reshape(batch, sequence, -1)

    child_fit = _masked_rows(child, fit)

    def fit_rows(target: Any) -> Any:
        return _ridge(child_fit, _masked_rows(target, fit), policy.ridge).T.contiguous()

    # Step 4 -- Q/K/V refit against pre-conv targets; conv, SiLU and the L2
    # normalization are replayed by the child, never fitted through.
    pre_conv_query = flatten(
        gather_key(
            evidence.pre_conv_query.reshape(
                batch, sequence, source.heads, source.key_head_dim
            )
        )
    )
    pre_conv_key = flatten(
        gather_key(
            evidence.pre_conv_key.reshape(
                batch, sequence, source.heads, source.key_head_dim
            )
        )
    )
    pre_conv_value = flatten(
        gather_value(
            evidence.pre_conv_value.reshape(
                batch, sequence, source.heads, source.value_head_dim
            )
        )
    )
    query_weight = fit_rows(pre_conv_query)
    key_weight = fit_rows(pre_conv_key)
    value_weight = fit_rows(pre_conv_value)

    def carry_taps(weight: Any, bias: Any, head_dim: int, channels: tuple[tuple[int, ...], ...]) -> tuple[Any, Any, float]:
        taps = weight.reshape(source.heads, head_dim, source.conv_kernel_size)
        biases = bias.reshape(source.heads, head_dim)
        merged_taps = _merge_heads(taps.permute(2, 0, 1), assignment).permute(1, 2, 0)
        merged_bias = _merge_heads(biases.unsqueeze(0), assignment).squeeze(0)
        selected_taps = torch.stack(
            [merged_taps[head, list(group)] for head, group in enumerate(channels)],
            dim=0,
        )
        selected_bias = torch.stack(
            [merged_bias[head, list(group)] for head, group in enumerate(channels)],
            dim=0,
        )
        dispersion = _group_dispersion(taps, assignment, axis=0)
        return (
            selected_taps.reshape(-1, source.conv_kernel_size).contiguous(),
            selected_bias.reshape(-1).contiguous(),
            dispersion,
        )

    query_taps, query_tap_bias, query_tap_dispersion = carry_taps(
        parameters.query_conv_weight,
        parameters.query_conv_bias,
        source.key_head_dim,
        selection.key_channels,
    )
    key_taps, key_tap_bias, key_tap_dispersion = carry_taps(
        parameters.key_conv_weight,
        parameters.key_conv_bias,
        source.key_head_dim,
        selection.key_channels,
    )
    value_taps, value_tap_bias, value_tap_dispersion = carry_taps(
        parameters.value_conv_weight,
        parameters.value_conv_bias,
        source.value_head_dim,
        selection.value_channels,
    )

    # Step 5 -- decay refit, closed form through the inverse chain.
    decay_scale_dispersion = _group_dispersion(
        parameters.decay_log_scale.reshape(1, source.heads), assignment, axis=-1
    )
    if decay_scale_dispersion > policy.maximum_merged_decay_dispersion:
        raise HFCausalKDAFoldError(
            "Merged-head decay log-scale dispersion "
            f"({decay_scale_dispersion:.4f}) exceeds the selection-partition "
            f"threshold ({policy.maximum_merged_decay_dispersion:.4f})."
        )
    child_log_scale = _merge_head_scalars(
        parameters.decay_log_scale.reshape(1, source.heads), assignment
    ).reshape(geometry.heads)
    decay_target = gather_key(chain_target) / child_log_scale.exp().reshape(
        1, 1, geometry.heads, 1
    )
    child_decay_bias = torch.stack(
        [
            _merge_heads(parameters.decay_bias.unsqueeze(0), assignment)
            .squeeze(0)[head, list(channels)]
            for head, channels in enumerate(selection.key_channels)
        ],
        dim=0,
    )
    decay_linear_target = decay_target - child_decay_bias.reshape(
        1, 1, geometry.heads, geometry.key_head_dim
    )
    decay_map = _ridge(
        child_fit, _masked_rows(flatten(decay_linear_target), fit), policy.ridge
    ).T.contiguous()
    left, singular, right = torch.linalg.svd(decay_map, full_matrices=False)
    rank = min(geometry.decay_rank, int(singular.numel()))
    decay_up_weight = (left[:, :rank] * singular[:rank].reshape(1, -1)).contiguous()
    decay_down_weight = right[:rank, :].contiguous()
    truncated = decay_up_weight @ decay_down_weight
    decay_truncation_error = _relative_rmse(truncated, decay_map)
    decay_tail_energy = float(
        (
            singular[rank:].square().sum() / singular.square().sum().clamp_min(1e-30)
        ).item()
    )
    if rank < geometry.decay_rank:
        decay_up_weight = torch.cat(
            [
                decay_up_weight,
                torch.zeros(
                    geometry.key_width, geometry.decay_rank - rank, dtype=torch.float64
                ),
            ],
            dim=1,
        )
        decay_down_weight = torch.cat(
            [
                decay_down_weight,
                torch.zeros(
                    geometry.decay_rank - rank, geometry.hidden_size, dtype=torch.float64
                ),
            ],
            dim=0,
        )

    # Step 6 -- beta refit through the inverse sigmoid.
    beta_dispersion = _group_dispersion(evidence.beta_rows, assignment, axis=-1)
    if beta_dispersion > policy.maximum_merged_beta_dispersion:
        raise HFCausalKDAFoldError(
            f"Merged-head beta dispersion ({beta_dispersion:.4f}) exceeds the "
            f"selection-partition threshold "
            f"({policy.maximum_merged_beta_dispersion:.4f})."
        )
    beta_logits, beta_saturated = _logit(evidence.beta_rows, policy.saturation_epsilon)
    write_strength_weight = fit_rows(_merge_head_scalars(beta_logits, assignment))

    # Step 8a -- the gate, inverse-sigmoid ridge against the parent pattern.
    gate_logits, gate_saturated = _logit(evidence.gate_rows, policy.saturation_epsilon)
    output_gate_weight = fit_rows(flatten(gather_value(gate_logits)))
    output_norm_weight = torch.stack(
        [
            _merge_heads(parameters.output_norm_weight.unsqueeze(0), assignment)
            .squeeze(0)[head, list(channels)]
            for head, channels in enumerate(selection.value_channels)
        ],
        dim=0,
    ).contiguous()
    saturated_fraction = max(beta_saturated, gate_saturated)
    if saturated_fraction > policy.maximum_saturated_row_fraction:
        raise HFCausalKDAFoldError(
            f"Inverse-sigmoid targets saturated on {saturated_fraction:.4f} of "
            "rows, above the sealed clamp policy "
            f"({policy.maximum_saturated_row_fraction:.4f})."
        )

    # Step 7 -- installed-recurrence replay before the writer solve.
    staged = HFCausalKDAFold(
        query_weight=query_weight,
        key_weight=key_weight,
        value_weight=value_weight,
        query_conv_weight=query_taps,
        query_conv_bias=query_tap_bias,
        key_conv_weight=key_taps,
        key_conv_bias=key_tap_bias,
        value_conv_weight=value_taps,
        value_conv_bias=value_tap_bias,
        decay_down_weight=decay_down_weight,
        decay_up_weight=decay_up_weight,
        decay_bias=child_decay_bias,
        decay_log_scale=child_log_scale,
        write_strength_weight=write_strength_weight,
        output_gate_weight=output_gate_weight,
        output_norm_weight=output_norm_weight,
        output_weight=torch.zeros(
            int(delta_target.shape[2]), geometry.value_width, dtype=torch.float64
        ),
        geometry=geometry,
        assignment=assignment,
        selection=selection,
        report={"schema": SCHEMA, "component": "staged_before_writer_solve"},
    )
    recurrent, child_snapshots = staged.recurrence(
        child, snapshot_positions=evidence.state_snapshot_positions
    )
    context = staged.gated_context(child, recurrent)

    # Step 8b -- endpoint-coupled writer solve against the residual delta.
    output_weight = _ridge(
        _masked_rows(context, fit), _masked_rows(delta_target, fit), policy.ridge
    ).T.contiguous()
    fold = HFCausalKDAFold(
        query_weight=query_weight,
        key_weight=key_weight,
        value_weight=value_weight,
        query_conv_weight=query_taps,
        query_conv_bias=query_tap_bias,
        key_conv_weight=key_taps,
        key_conv_bias=key_tap_bias,
        value_conv_weight=value_taps,
        value_conv_bias=value_tap_bias,
        decay_down_weight=decay_down_weight,
        decay_up_weight=decay_up_weight,
        decay_bias=child_decay_bias,
        decay_log_scale=child_log_scale,
        write_strength_weight=write_strength_weight,
        output_gate_weight=output_gate_weight,
        output_norm_weight=output_norm_weight,
        output_weight=output_weight,
        geometry=geometry,
        assignment=assignment,
        selection=selection,
        report={"schema": SCHEMA, "component": "staged_before_receipt"},
    )
    predicted = context @ output_weight.T
    fit_prediction = _masked_rows(predicted, fit)
    fit_target = _masked_rows(delta_target, fit)
    validation_prediction = _masked_rows(predicted, validation)
    validation_target = _masked_rows(delta_target, validation)
    zero_baseline = _relative_rmse(
        torch.zeros_like(validation_target), validation_target
    )
    validation_error = _relative_rmse(validation_prediction, validation_target)

    # Step 9 -- BF16 replay of every intermediate, then the acceptance gates.
    predicted_bf16 = (
        _replay_kda_bf16(fold, child) if policy.bf16_replay else predicted
    )
    validation_error_bf16 = _relative_rmse(
        _masked_rows(predicted_bf16, validation), validation_target
    )
    if validation_error_bf16 >= zero_baseline:
        raise HFCausalKDAFoldError(
            "The BF16 validation endpoint is at or above the no-branch baseline "
            f"({validation_error_bf16:.6f} >= {zero_baseline:.6f})."
        )

    # Long-horizon state-drift gate on the selected sub-block of the parent
    # state; the growth policy constant lives in the sealed policy.
    merged_snapshots = _merge_heads(evidence.state_snapshots, assignment, axis=2)
    parent_snapshots = torch.stack(
        [
            merged_snapshots[:, :, head][:, :, list(selection.key_channels[head])][
                :, :, :, list(selection.value_channels[head])
            ]
            for head in range(geometry.heads)
        ],
        dim=2,
    )
    drift = [
        _relative_rmse(child_snapshots[index], parent_snapshots[index])
        for index in range(int(child_snapshots.shape[0]))
    ]
    # Superlinearity is measured against the *horizon*, not the snapshot index,
    # and only where the error is materially above the replay noise floor: a
    # drift ratio computed from a 1e-8 baseline measures float64 rounding, not
    # a state that is running away.
    horizons = [float(position + 1) for position in evidence.state_snapshot_positions]
    growth = 0.0
    applicable = len(drift) > 1 and max(drift) > policy.state_drift_floor and drift[0] > 0.0
    if applicable:
        growth = max(
            value * horizons[0] / (drift[0] * horizons[index])
            for index, value in enumerate(drift)
            if index > 0
        )
    if growth > policy.state_drift_growth_cap:
        raise HFCausalKDAFoldError(
            f"The child state drift grows superlinearly (growth {growth:.4f} > "
            f"cap {policy.state_drift_growth_cap:.4f}) across chunk boundaries."
        )
    body = {
        "schema": SCHEMA,
        "component": "chimera_kda_recurrent_branch_fold",
        "method": (
            "unroll_gate_barycentric_heads_balanced_ranked_channel_selection_"
            "inverse_sigmoid_ridges_installed_recurrence_endpoint_writer"
        ),
        "layer_index": int(evidence.index),
        "source_geometry": source.as_dict(),
        "target_geometry": geometry.as_dict(),
        "policy": policy.as_dict(),
        "assignment": assignment.as_dict(),
        "channel_selection_report_sha256": selection.report["report_sha256"],
        "unroll_report_sha256": unroll.diagnostics["report_sha256"],
        "unroll_reconstruction_relative_rmse": unroll.relative_rmse,
        "conv_tap_carry_relative_rmse": conv_carry_error,
        "decay_inverse_chain_relative_rmse": decay_chain_error,
        "all_source_heads_covered_exactly_once": True,
        "rotation_applied": False,
        "recurrence_installed_before_writer_fit": True,
        "complete_target_residual_endpoint_consumed": True,
        "fit_rows": int(fit.sum().item()),
        "validation_rows": int(validation.sum().item()),
        "fit_residual_relative_rmse": _relative_rmse(fit_prediction, fit_target),
        "fit_residual_retained_energy_fraction": _retained_energy(
            fit_prediction, fit_target
        ),
        "validation_residual_relative_rmse": validation_error,
        "validation_residual_retained_energy_fraction": _retained_energy(
            validation_prediction, validation_target
        ),
        "validation_no_branch_relative_rmse": zero_baseline,
        "validation_beats_no_branch_baseline": bool(validation_error < zero_baseline),
        "validation_residual_relative_rmse_bf16": validation_error_bf16,
        "bf16_versus_fp64_endpoint_relative_delta": _relative_rmse(
            _masked_rows(predicted_bf16, validation), validation_prediction
        ),
        "merged_beta_dispersion": beta_dispersion,
        "merged_decay_log_scale_dispersion": decay_scale_dispersion,
        "merged_query_conv_tap_dispersion": query_tap_dispersion,
        "merged_key_conv_tap_dispersion": key_tap_dispersion,
        "merged_value_conv_tap_dispersion": value_tap_dispersion,
        "inverse_sigmoid_saturated_row_fraction": saturated_fraction,
        "beta_saturated_row_fraction": beta_saturated,
        "gate_saturated_row_fraction": gate_saturated,
        "decay_probability_saturated_row_fraction": decay_saturated,
        "decay_rank": geometry.decay_rank,
        "decay_rank_truncation_relative_error": decay_truncation_error,
        "decay_rank_discarded_singular_energy_fraction": decay_tail_energy,
        "state_snapshot_positions": list(evidence.state_snapshot_positions),
        "state_snapshot_relative_rmse": drift,
        "state_drift_growth_ratio": growth,
        "state_drift_growth_cap": policy.state_drift_growth_cap,
        "state_drift_gate_applicable": bool(applicable),
        "recurrent_output_relative_rmse": _relative_rmse(
            _masked_rows(recurrent, validation),
            _masked_rows(
                gather_value(evidence.recurrent_output_rows), validation
            ),
        ),
        "ridge": policy.ridge,
        "numeric_weight_matching_objective": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
        "query_weight_sha256": _tensor_sha256(query_weight),
        "key_weight_sha256": _tensor_sha256(key_weight),
        "value_weight_sha256": _tensor_sha256(value_weight),
        "decay_up_weight_sha256": _tensor_sha256(decay_up_weight),
        "decay_down_weight_sha256": _tensor_sha256(decay_down_weight),
        "write_strength_weight_sha256": _tensor_sha256(write_strength_weight),
        "output_gate_weight_sha256": _tensor_sha256(output_gate_weight),
        "output_weight_sha256": _tensor_sha256(output_weight),
    }
    return HFCausalKDAFold(
        query_weight=query_weight,
        key_weight=key_weight,
        value_weight=value_weight,
        query_conv_weight=query_taps,
        query_conv_bias=query_tap_bias,
        key_conv_weight=key_taps,
        key_conv_bias=key_tap_bias,
        value_conv_weight=value_taps,
        value_conv_bias=value_tap_bias,
        decay_down_weight=decay_down_weight,
        decay_up_weight=decay_up_weight,
        decay_bias=child_decay_bias,
        decay_log_scale=child_log_scale,
        write_strength_weight=write_strength_weight,
        output_gate_weight=output_gate_weight,
        output_norm_weight=output_norm_weight,
        output_weight=output_weight,
        geometry=geometry,
        assignment=assignment,
        selection=selection,
        report=_report(body),
    )


# ---------------------------------------------------------------------------
# Stage-2 comparability hooks: the shortconv -> KDA pole construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstantKeyKDAPole:
    """The scalar IIR a constant-key KDA head collapses to (design section 3)."""

    pole: float
    write_strength: float
    exact: bool
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _report(dict(self.diagnostics)))


def constant_key_kda_pole(
    *,
    key: Any,
    retention: Any,
    write_strength: float,
    tolerance: float = 1e-9,
) -> ConstantKeyKDAPole:
    """``a = (1 - beta) * sum_i k_i^2 alpha_i`` for a constant-key KDA head.

    With ``W_k = 0`` the key is the constant unit vector
    ``L2Norm(SiLU(conv_bias))`` (the conv carries a bias, ``layers.py:64-70``)
    and, with the query on the same direction, the head's effective weight
    sequence is ``h_l = beta * k^T B^l k`` for
    ``B = (I - beta k k^T) Diag(alpha)``.

    That is a *single* pole exactly when ``alpha`` is constant across the
    channels the key occupies, because only then is ``k`` an eigenvector of
    ``Diag(alpha)``.  Otherwise ``k^T B k`` is the Rayleigh quotient of a
    multi-pole mixture and the design's formula is a first-order reading of it.
    ``exact`` says which case this is; the deviation is measured, not assumed
    away.
    """

    torch = _torch()
    key_vector = _tensor(key, "key", ndim=1)
    retention_vector = _tensor(retention, "retention", ndim=1)
    if int(key_vector.numel()) != int(retention_vector.numel()):
        raise HFCausalKDAFoldError("key and retention must have the same width.")
    beta = _nonnegative_float(write_strength, "write_strength")
    if beta > 1.0:
        raise HFCausalKDAFoldError("write_strength must lie in [0, 1].")
    tolerance = _nonnegative_float(tolerance, "tolerance")
    norm = float(key_vector.square().sum().sqrt().item())
    if abs(norm - 1.0) > 1e-6:
        raise HFCausalKDAFoldError(
            "The constant key must be L2-normalized, as the parent normalizes it."
        )
    mass = key_vector.square()
    pole = float(((1.0 - beta) * (mass * retention_vector).sum()).item())
    support = mass > tolerance
    supported = retention_vector[support]
    spread = (
        float((supported.max() - supported.min()).item())
        if int(supported.numel()) > 0
        else 0.0
    )
    exact = spread <= tolerance
    diagnostics = {
        "schema": SHORT_CONV_TO_KDA_SCHEMA,
        "component": "constant_key_kda_effective_pole",
        "pole": pole,
        "write_strength": beta,
        "single_pole_exact": bool(exact),
        "retention_spread_on_key_support": spread,
        "retention_spread_tolerance": tolerance,
        "key_support_channels": int(support.sum().item()),
        "torch_dtype": str(torch.float64).removeprefix("torch."),
    }
    return ConstantKeyKDAPole(
        pole=pole,
        write_strength=beta,
        exact=bool(exact),
        diagnostics=diagnostics,
    )


@dataclass(frozen=True, slots=True)
class LagPoleFit:
    """Per-head poles and mixing coefficients matching an L-tap FIR."""

    poles: tuple[float, ...]
    coefficients: tuple[float, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", _report(dict(self.report)))


def fir_taps_to_kda_poles(
    taps: Any,
    *,
    heads: int,
    gate_lower_bound: float = -5.0,
    condition_cap: float = 1e4,
    tail_lags: int = 32,
) -> LagPoleFit:
    """Match an ``L``-tap FIR at lags ``0..L-1`` with ``L`` KDA poles.

    One constant-key head is one pole, so a Vandermonde system
    ``V[l, h] = a_h^l`` reproduces the taps exactly at the matched lags with
    ``c = V^{-1} taps`` (closed form; conditioning reported).  Poles are laid
    out deterministically between the retention floor ``exp(g_min)`` and a
    value near one, so the *unmatched* tail ``sum_h c_h a_h^lag`` for
    ``lag >= L`` is minimized by poles near the floor -- and is measured, never
    assumed zero.  ``heads < L`` fails closed (design fail-closed condition 4).
    """

    torch = _torch()
    tap_vector = _tensor(taps, "taps", ndim=1)
    length = int(tap_vector.numel())
    heads = _positive_int(heads, "heads")
    condition_cap = _positive_float(condition_cap, "condition_cap")
    tail_lags = _positive_int(tail_lags, "tail_lags")
    floor = math.exp(_negative_float(gate_lower_bound, "gate_lower_bound"))
    if heads < length:
        raise HFCausalKDAFoldError(
            f"A {length}-tap FIR needs at least {length} KDA heads to carry one "
            f"pole per lag; the child has {heads}."
        )
    if length == 1:
        poles = torch.tensor([floor], dtype=torch.float64)
    else:
        ceiling = 0.9
        poles = torch.tensor(
            [
                floor * (ceiling / floor) ** (index / (length - 1))
                for index in range(length)
            ],
            dtype=torch.float64,
        )
    vandermonde = torch.stack(
        [poles**lag for lag in range(length)], dim=0
    ).contiguous()
    condition = float(torch.linalg.cond(vandermonde).item())
    if not math.isfinite(condition) or condition > condition_cap:
        raise HFCausalKDAFoldError(
            f"The pole Vandermonde system is too ill-conditioned "
            f"({condition:.3e} > {condition_cap:.3e}); the translation fails closed."
        )
    coefficients = torch.linalg.solve(vandermonde, tap_vector)
    matched = vandermonde @ coefficients
    tail = torch.stack(
        [(coefficients * poles**lag).sum() for lag in range(length, length + tail_lags)]
    )
    body = {
        "schema": SHORT_CONV_TO_KDA_SCHEMA,
        "component": "fir_lag_pole_vandermonde_fit",
        "fir_length": length,
        "child_heads": heads,
        "poles": [float(value.item()) for value in poles],
        "coefficients": [float(value.item()) for value in coefficients],
        "retention_floor": floor,
        "vandermonde_condition_number": condition,
        "condition_cap": condition_cap,
        "matched_lag_maximum_absolute_error": float(
            (matched - tap_vector).abs().max().item()
        ),
        "tail_lags_measured": tail_lags,
        "tail_energy_fraction": float(
            (
                tail.square().sum() / tap_vector.square().sum().clamp_min(1e-30)
            ).item()
        ),
        "tail_maximum_absolute_value": float(tail.abs().max().item()),
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
    }
    return LagPoleFit(
        poles=tuple(float(value.item()) for value in poles),
        coefficients=tuple(float(value.item()) for value in coefficients),
        report=body,
    )


def _unimplemented_translation(name: str, section: str, verdict: str) -> None:
    raise HFCausalKDAFoldError(
        f"{name} is designed but not implemented in this wave. "
        f"See {DESIGN_DOCUMENT} section {section} (verdict: {verdict}) and the "
        "stage ladder in section 5; this entry point fails closed rather than "
        "returning an unmeasured translation."
    )


def compile_short_conv_to_kda_translation(*_args: Any, **_kwargs: Any) -> Any:
    """Fail closed: stage-3 lane, see the design document.

    The closed-form primitives this translation is built from are implemented
    and tested here (``constant_key_kda_pole``, ``fir_taps_to_kda_poles``).
    What is missing is the LFM2 gate linearization hand-off and the routed
    remainder, which belong to the real-model stage.
    """

    _unimplemented_translation(
        "compile_short_conv_to_kda_translation", "3", "honest, exact-in-limit"
    )


def compile_gqa_to_kda_translation(*_args: Any, **_kwargs: Any) -> Any:
    """Fail closed: the balanced-truncation lag-profile lane (stage 4)."""

    _unimplemented_translation(
        "compile_gqa_to_kda_translation",
        "3",
        "measured-partial, the new balanced-truncation lane",
    )


def compile_kda_to_gqa_translation(*_args: Any, **_kwargs: Any) -> Any:
    """Fail closed: signed unrolled weights into synthesized GQA evidence.

    ``kda_effective_weights`` already produces the exact signed weight matrix
    this translation consumes; the missing half is the log-magnitude
    factorization into desired scores and the per-row mass-variance
    obstruction measurement.
    """

    _unimplemented_translation(
        "compile_kda_to_gqa_translation", "3", "measured-partial, primary lane"
    )


__all__ = [
    "DESIGN_DOCUMENT",
    "EVIDENCE_SCHEMA",
    "GQA_TO_KDA_SCHEMA",
    "KDA_TO_GQA_SCHEMA",
    "SCHEMA",
    "SHORT_CONV_TO_KDA_SCHEMA",
    "UNROLL_SCHEMA",
    "ChimeraKDAFoldPolicy",
    "ChimeraKDAGeometry",
    "ChimeraKDALayerEvidence",
    "ChimeraKDAParameters",
    "ConstantKeyKDAPole",
    "HFCausalKDAFold",
    "HFCausalKDAFoldError",
    "KDAChannelSelection",
    "KDAHeadAssignment",
    "KDAWeightUnroll",
    "LagPoleFit",
    "capture_chimera_kda_layer_evidence",
    "chimera_kda_geometry",
    "chimera_kda_parameters",
    "compile_gqa_to_kda_translation",
    "compile_hf_causal_kda_fold",
    "compile_kda_to_gqa_translation",
    "compile_short_conv_to_kda_translation",
    "constant_key_kda_pole",
    "fir_taps_to_kda_poles",
    "kda_effective_weights",
    "replay_kda_branch",
    "select_kda_channels",
]
