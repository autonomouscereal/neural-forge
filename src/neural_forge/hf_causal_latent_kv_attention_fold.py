"""Chimera-omni gated-MLA compressed-latent-KV residual-branch folding.

``GatedMultiHeadLatentAttention``
(``chimera-omni/src/chimera_omni/modeling/layers.py:231-306``) is NoPE softmax
attention whose keys and values are read out of one shared low-rank latent,
whose queries are optionally read out of a second one, and whose context is
gated channel-wise before a single writer.  This module folds it closed form,
following ``docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md`` section 1.

Every solve is a ridge, an SVD or an exact carry.  There is no optimizer,
gradient, backward pass, parent forward, runtime sidecar or database here.

Implementation deviations from the design document, with reasons
----------------------------------------------------------------

1. **The shared key block is materialized by an explicit broadcast rather than
   folded into ``kv_up``.**  ``mla_extra_head_dim`` channels come straight off
   ``kv_down`` and are the *same* row for every head
   (``layers.py:290-294``); writing that broadcast out makes the head-independent
   rank-``Dx`` score term measurable
   (``shared_key_channel_score_energy_fraction``) instead of hidden inside a
   per-head key.

2. **The latent bases are POD on the observed rows, not an SVD of the weight.**
   Truncating ``kv_up`` by weight SVD is exact for the *map* but says nothing
   about the rows the trunk actually produces.  The leading right singular
   vectors of the observed latent rows are the coordinates the evidence
   supports, and the discarded singular energy is reported exactly.  When the
   child latent width equals the parent's the basis is the identity, so a
   width-preserving fold is not perturbed by a rotation it does not need.

3. **The child's RMSNorms are replayed, not fitted through.**  ``q_norm`` and
   ``kv_norm`` gains are fitted diagonally against the parent's own normalized
   rows, and the *replayed* child normalization feeds the next solve.  This is
   the pre-RoPE discipline of ``hf_causal_gqa_attention_fold`` transplanted:
   fitting a projection through a normalization measurably hurts, because the
   unit-RMS direction is not a linear function of the cached state.

4. **The gate is fitted against the captured pre-activation.**  KDA needs an
   inverse-sigmoid target because no boundary emits its gate logits; here the
   ``gate`` linear's output *is* a captured module boundary, so the solve is a
   plain ridge and no clamp policy is required.  The saturated-row fraction of
   ``sigmoid(gate)`` is still reported, because saturation still bounds what a
   refit can recover.

5. **No score chart is taken or accepted.**  Unlike the CSA summary path, whose
   score is an inner product in the residual width, every MLA score is an inner
   product in head coordinates and every consumer solve reads child-width rows.
   A residual-width crossing therefore needs nothing here beyond a
   chart-projected endpoint target, which the caller supplies.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_closed_form_causal_circuit_fold import fit_rmsnorm_diagonal


SCHEMA = "neural-forge-chimera-latent-kv-attention-fold-v1"
EVIDENCE_SCHEMA = "neural-forge-chimera-latent-kv-attention-evidence-v1"

DESIGN_DOCUMENT = "docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md"


class HFCausalLatentKVAttentionFoldError(ValueError):
    """A gated-MLA fold is unsupported or its evidence is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise HFCausalLatentKVAttentionFoldError(
            "PyTorch is required for compressed-latent-KV folding."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalLatentKVAttentionFoldError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HFCausalLatentKVAttentionFoldError(
            f"{name} must be a non-negative integer."
        )
    return value


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalLatentKVAttentionFoldError(f"{name} must be positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise HFCausalLatentKVAttentionFoldError(
            f"{name} must be positive and finite."
        )
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalLatentKVAttentionFoldError(
            f"{name} must be non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise HFCausalLatentKVAttentionFoldError(
            f"{name} must be finite and non-negative."
        )
    return result


def _tensor(value: Any, name: str, *, ndim: int | None = None) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise HFCausalLatentKVAttentionFoldError(f"{name} must be a tensor.")
    if ndim is not None and value.ndim != ndim:
        raise HFCausalLatentKVAttentionFoldError(
            f"{name} must be rank {ndim}, got rank {value.ndim}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise HFCausalLatentKVAttentionFoldError(f"{name} contains non-finite values.")
    return result


def _mask(value: Any, name: str, shape: tuple[int, int]) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise HFCausalLatentKVAttentionFoldError(f"{name} must have shape {shape}.")
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


def _energy_fraction(part: Any, total: Any) -> float:
    return float(
        (part.square().sum() / total.square().sum().clamp_min(1e-30)).item()
    )


def _masked_rows(value: Any, mask: Any) -> Any:
    trailing = tuple(int(size) for size in value.shape[2:])
    return value.reshape(int(mask.numel()), *trailing)[mask.reshape(-1)].contiguous()


def _ridge(source: Any, target: Any, ridge: float) -> Any:
    torch = _torch()
    if source.ndim != 2 or target.ndim != 2 or int(source.shape[0]) != int(
        target.shape[0]
    ):
        raise HFCausalLatentKVAttentionFoldError(
            "Ridge evidence matrices are incompatible."
        )
    gram = source.T @ source
    return torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=torch.float64),
        source.T @ target,
    )


def _balanced_groups(total: int, count: int) -> tuple[tuple[int, ...], ...]:
    """Complete contiguous head assignment; every source head is used once."""

    total = _positive_int(total, "source head count")
    count = _positive_int(count, "target head count")
    if count > total:
        raise HFCausalLatentKVAttentionFoldError(
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


def _select_channels(values: tuple[Any, ...], target_dim: int) -> tuple[int, ...]:
    """Deterministic energy-ranked channel selection for a NoPE head axis."""

    torch = _torch()
    source_dim = int(values[0].shape[-1])
    target_dim = _positive_int(target_dim, "target_dim")
    if target_dim > source_dim:
        raise HFCausalLatentKVAttentionFoldError(
            "Head dimension expansion is outside the contraction-only fold."
        )
    energy = torch.zeros(source_dim, dtype=torch.float64)
    for value in values:
        energy = energy + value.square().sum(dim=tuple(range(value.ndim - 1)))
    order = torch.argsort(energy, descending=True, stable=True)[:target_dim]
    return tuple(sorted(int(value) for value in order.tolist()))


def _take(value: Any, selected: tuple[int, ...]) -> Any:
    torch = _torch()
    return value.index_select(
        -1, torch.tensor(selected, dtype=torch.long)
    ).contiguous()


# ---------------------------------------------------------------------------
# Geometry, policy, evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChimeraLatentKVGeometry:
    """One ``GatedMultiHeadLatentAttention`` geometry (``layers.py:234-271``)."""

    hidden_size: int
    heads: int
    key_head_dim: int
    value_head_dim: int
    extra_head_dim: int
    kv_latent_size: int
    query_latent_size: int = 0
    rms_norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "heads",
            "key_head_dim",
            "value_head_dim",
            "kv_latent_size",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in ("extra_head_dim", "query_latent_size"):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "rms_norm_epsilon",
            _positive_float(self.rms_norm_epsilon, "rms_norm_epsilon"),
        )

    @property
    def query_head_dim(self) -> int:
        return self.key_head_dim + self.extra_head_dim

    @property
    def query_width(self) -> int:
        return self.heads * self.query_head_dim

    @property
    def key_value_width(self) -> int:
        return self.heads * (self.key_head_dim + self.value_head_dim)

    @property
    def value_width(self) -> int:
        return self.heads * self.value_head_dim

    @property
    def compressed_width(self) -> int:
        return self.kv_latent_size + self.extra_head_dim

    @property
    def attention_scaling(self) -> float:
        return 1.0 / math.sqrt(float(self.query_head_dim))

    @property
    def compressed_query(self) -> bool:
        return self.query_latent_size > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }


@dataclass(frozen=True, slots=True)
class ChimeraLatentKVFoldPolicy:
    """Frozen policy constants for one gated-MLA fold (design section 5)."""

    ridge: float = 1e-6
    readout_identity_tolerance: float = 1e-5
    delta_reconstruction_tolerance: float = 1e-5
    minimum_attention_probability_overlap: float = 0.25
    gate_saturation_epsilon: float = 1e-3
    maximum_gate_saturated_row_fraction: float = 0.5
    bf16_replay: bool = True

    def __post_init__(self) -> None:
        for name in (
            "ridge",
            "readout_identity_tolerance",
            "delta_reconstruction_tolerance",
            "minimum_attention_probability_overlap",
            "gate_saturation_epsilon",
            "maximum_gate_saturated_row_fraction",
        ):
            object.__setattr__(
                self, name, _nonnegative_float(getattr(self, name), name)
            )
        if not 0.0 < self.gate_saturation_epsilon < 0.5:
            raise HFCausalLatentKVAttentionFoldError(
                "gate_saturation_epsilon must lie in (0, 0.5)."
            )
        if not isinstance(self.bf16_replay, bool):
            raise HFCausalLatentKVAttentionFoldError("bf16_replay must be boolean.")

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }


@dataclass(frozen=True, slots=True)
class ChimeraLatentKVParameters:
    """The parent weights a gated-MLA fold reads rather than fits.

    ``kv_up_weight`` and ``output_weight`` are needed by the two evidence
    gates: the first recomputes the captured key/value readout from the
    captured normalized latent, the second rebuilds the captured branch delta.
    """

    kv_up_weight: Any
    output_weight: Any
    geometry: ChimeraLatentKVGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, ChimeraLatentKVGeometry):
            raise HFCausalLatentKVAttentionFoldError(
                "geometry must be a ChimeraLatentKVGeometry."
            )
        shapes = {
            "kv_up_weight": (
                self.geometry.key_value_width,
                self.geometry.kv_latent_size,
            ),
            "output_weight": (self.geometry.hidden_size, self.geometry.value_width),
        }
        for name, shape in shapes.items():
            value = _tensor(getattr(self, name), name, ndim=2).clone()
            if tuple(int(size) for size in value.shape) != shape:
                raise HFCausalLatentKVAttentionFoldError(
                    f"{name} must have shape {shape}."
                )
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ChimeraLatentKVLayerEvidence:
    """Complete cached circuit states for one chimera gated-MLA branch.

    Field style mirrors ``HFCausalLayerEvidence``
    (``hf_causal_circuit_evidence.py:214-233``): one sealed capture, rank-3
    ``[batch, sequence, width]`` rows, no live parent.  ``query_latent_rows``
    and ``normalized_query_latent_rows`` are present exactly when the geometry
    declares a compressed query path.
    """

    index: int
    block_input: Any
    attention_norm_output: Any
    query_rows: Any
    compressed_kv_rows: Any
    normalized_latent_rows: Any
    kv_rows: Any
    gate_rows: Any
    attention_delta: Any
    block_output: Any
    geometry: ChimeraLatentKVGeometry
    metadata: Mapping[str, Any]
    query_latent_rows: Any | None = None
    normalized_query_latent_rows: Any | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or (
            self.index < 0
        ):
            raise HFCausalLatentKVAttentionFoldError(
                "layer evidence index must be a non-negative integer."
            )
        if not isinstance(self.geometry, ChimeraLatentKVGeometry):
            raise HFCausalLatentKVAttentionFoldError(
                "geometry must be a ChimeraLatentKVGeometry."
            )
        geometry = self.geometry
        widths = {
            "block_input": geometry.hidden_size,
            "attention_norm_output": geometry.hidden_size,
            "query_rows": geometry.query_width,
            "compressed_kv_rows": geometry.compressed_width,
            "normalized_latent_rows": geometry.kv_latent_size,
            "kv_rows": geometry.key_value_width,
            "gate_rows": geometry.value_width,
            "attention_delta": geometry.hidden_size,
            "block_output": geometry.hidden_size,
        }
        for name, width in widths.items():
            value = _tensor(getattr(self, name), name, ndim=3).clone()
            if int(value.shape[2]) != width:
                raise HFCausalLatentKVAttentionFoldError(
                    f"{name} must have width {width}."
                )
            object.__setattr__(self, name, value)
        batch = int(self.block_input.shape[0])
        sequence = int(self.block_input.shape[1])
        for name in widths:
            if tuple(int(size) for size in getattr(self, name).shape[:2]) != (
                batch,
                sequence,
            ):
                raise HFCausalLatentKVAttentionFoldError(
                    f"{name} has the wrong sequence shape."
                )
        latent_rows = {
            "query_latent_rows": geometry.query_latent_size,
            "normalized_query_latent_rows": geometry.query_latent_size,
        }
        for name, width in latent_rows.items():
            value = getattr(self, name)
            if not geometry.compressed_query:
                if value is not None:
                    raise HFCausalLatentKVAttentionFoldError(
                        f"{name} is present but the geometry declares a dense "
                        "query projection."
                    )
                continue
            if value is None:
                raise HFCausalLatentKVAttentionFoldError(
                    f"{name} is required by a compressed query path."
                )
            rows = _tensor(value, name, ndim=3).clone()
            if tuple(int(size) for size in rows.shape) != (batch, sequence, width):
                raise HFCausalLatentKVAttentionFoldError(
                    f"{name} must be [batch, sequence, {width}]."
                )
            object.__setattr__(self, name, rows)
        object.__setattr__(self, "metadata", _report(dict(self.metadata)))

    @property
    def batch(self) -> int:
        return int(self.block_input.shape[0])

    @property
    def sequence(self) -> int:
        return int(self.block_input.shape[1])

    def head_query(self) -> Any:
        """``[batch, heads, sequence, query_head_dim]`` parent queries."""

        geometry = self.geometry
        return (
            self.query_rows.reshape(
                self.batch, self.sequence, geometry.heads, geometry.query_head_dim
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def head_key_value(self) -> tuple[Any, Any]:
        """Split the readout into per-head base keys and values."""

        geometry = self.geometry
        blocks = self.kv_rows.reshape(
            self.batch,
            self.sequence,
            geometry.heads,
            geometry.key_head_dim + geometry.value_head_dim,
        ).permute(0, 2, 1, 3)
        key = blocks[..., : geometry.key_head_dim].contiguous()
        value = blocks[..., geometry.key_head_dim :].contiguous()
        return key, value

    def shared_key_rows(self) -> Any:
        """``[batch, sequence, extra_head_dim]`` shared key channels."""

        return self.compressed_kv_rows[
            ..., self.geometry.kv_latent_size :
        ].contiguous()

    def latent_rows(self) -> Any:
        """``[batch, sequence, kv_latent_size]`` pre-normalization latent."""

        return self.compressed_kv_rows[
            ..., : self.geometry.kv_latent_size
        ].contiguous()

    def head_gate(self) -> Any:
        geometry = self.geometry
        return (
            self.gate_rows.reshape(
                self.batch, self.sequence, geometry.heads, geometry.value_head_dim
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def tensor_items(self) -> tuple[tuple[str, Any], ...]:
        skipped = {"index", "geometry", "metadata"}
        return tuple(
            (descriptor.name, getattr(self, descriptor.name))
            for descriptor in fields(self)
            if descriptor.name not in skipped
            and getattr(self, descriptor.name) is not None
        )


# ---------------------------------------------------------------------------
# Masks, probabilities, the shared-key broadcast
# ---------------------------------------------------------------------------


def latent_attention_probabilities(
    query: Any,
    key: Any,
    *,
    attention_mask: Any,
    scaling: float,
) -> tuple[Any, Any]:
    """Scores and probabilities for one full-causal NoPE MLA replay.

    The parent masks with ``finfo.min`` (``layers.py:303``); both that and
    ``-inf`` send masked entries to exactly zero after the softmax for any
    finite unmasked score, so the replay is equivalent.
    """

    torch = _torch()
    batch, heads, sequence, _dim = query.shape
    scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
    key_mask = attention_mask.reshape(batch, 1, 1, sequence).to(torch.bool)
    causal = (
        torch.ones(sequence, sequence, dtype=torch.bool)
        .tril()
        .reshape(1, 1, sequence, sequence)
    )
    allowed = key_mask & causal
    no_key = ~allowed.any(dim=-1, keepdim=True)
    if bool(no_key.any().item()):
        diagonal = torch.eye(sequence, dtype=torch.bool).reshape(
            1, 1, sequence, sequence
        )
        allowed = allowed | (no_key & diagonal)
    probabilities = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
    query_valid = attention_mask.reshape(batch, 1, sequence, 1).to(torch.bool)
    return scores.contiguous(), (probabilities * query_valid).contiguous()


def broadcast_shared_key(shared_rows: Any, heads: int) -> Any:
    """Broadcast the shared key block to every head (``layers.py:291-294``)."""

    heads = _positive_int(heads, "heads")
    batch, sequence, extra = (
        int(shared_rows.shape[0]),
        int(shared_rows.shape[1]),
        int(shared_rows.shape[2]),
    )
    return (
        shared_rows.reshape(batch, 1, sequence, extra)
        .expand(batch, heads, sequence, extra)
        .contiguous()
    )


def assemble_latent_keys(base_key: Any, shared_rows: Any) -> Any:
    """``k = cat([k_base, k_extra])`` with the shared block on every head."""

    torch = _torch()
    if int(shared_rows.shape[-1]) == 0:
        return base_key.contiguous()
    shared = broadcast_shared_key(shared_rows, int(base_key.shape[1]))
    return torch.cat([base_key, shared], dim=-1).contiguous()


# ---------------------------------------------------------------------------
# Parent capture
# ---------------------------------------------------------------------------


def chimera_latent_kv_geometry(
    module: Any,
    *,
    hidden_size: int | None = None,
    rms_norm_epsilon: float | None = None,
) -> ChimeraLatentKVGeometry:
    """Read one chimera gated-MLA geometry off a live module."""

    try:
        width = int(hidden_size or module.kv_down.weight.shape[1])
        epsilon = rms_norm_epsilon
        if epsilon is None:
            epsilon = float(getattr(module.kv_norm, "eps", 1e-6))
        return ChimeraLatentKVGeometry(
            hidden_size=width,
            heads=int(module.num_heads),
            key_head_dim=int(module.key_head_dim),
            value_head_dim=int(module.value_head_dim),
            extra_head_dim=int(module.extra_head_dim),
            kv_latent_size=int(module.kv_latent_size),
            query_latent_size=int(module.query_latent_size),
            rms_norm_epsilon=epsilon,
        )
    except AttributeError as exc:
        raise HFCausalLatentKVAttentionFoldError(
            "module does not expose the chimera GatedMultiHeadLatentAttention "
            "layout."
        ) from exc


def chimera_latent_kv_parameters(
    module: Any,
    *,
    geometry: ChimeraLatentKVGeometry | None = None,
) -> ChimeraLatentKVParameters:
    """Carry the two parent weights the evidence gates need."""

    resolved = geometry or chimera_latent_kv_geometry(module)
    try:
        return ChimeraLatentKVParameters(
            kv_up_weight=module.kv_up.weight,
            output_weight=module.out.weight,
            geometry=resolved,
        )
    except AttributeError as exc:
        raise HFCausalLatentKVAttentionFoldError(
            "module does not expose kv_up and out."
        ) from exc


def capture_chimera_latent_kv_layer_evidence(
    module: Any,
    normalized_rows: Any,
    *,
    layer_index: int,
    block_input: Any | None = None,
    block_output: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ChimeraLatentKVLayerEvidence:
    """Run the parent branch exactly once and seal every module boundary."""

    torch = _torch()
    geometry = chimera_latent_kv_geometry(module)
    rows = normalized_rows.detach()
    if rows.ndim != 3 or int(rows.shape[2]) != geometry.hidden_size:
        raise HFCausalLatentKVAttentionFoldError(
            "normalized_rows must be [batch, sequence, hidden_size]."
        )
    with torch.no_grad():
        compressed = module.kv_down(rows)
        latent = compressed[..., : geometry.kv_latent_size]
        normalized_latent = module.kv_norm(latent)
        kv_rows = module.kv_up(normalized_latent)
        gate_rows = module.gate(rows)
        if geometry.compressed_query:
            query_latent = module.q_down(rows)
            normalized_query_latent = module.q_norm(query_latent)
            query_rows = module.q_up(normalized_query_latent)
        else:
            query_latent = None
            normalized_query_latent = None
            query_rows = module.q_proj(rows)
        delta = module(rows)
    resolved_input = rows if block_input is None else block_input
    sealed = {
        "schema": EVIDENCE_SCHEMA,
        "component": "chimera_latent_kv_branch_capture",
        "parent_forward_calls": 1,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        **dict(metadata or {}),
    }
    return ChimeraLatentKVLayerEvidence(
        index=int(layer_index),
        block_input=resolved_input,
        attention_norm_output=rows,
        query_rows=query_rows,
        compressed_kv_rows=compressed,
        normalized_latent_rows=normalized_latent,
        kv_rows=kv_rows,
        gate_rows=gate_rows,
        attention_delta=delta,
        block_output=(
            _tensor(resolved_input, "block_input") + _tensor(delta, "delta")
            if block_output is None
            else block_output
        ),
        geometry=geometry,
        metadata=sealed,
        query_latent_rows=query_latent,
        normalized_query_latent_rows=normalized_query_latent,
    )


# ---------------------------------------------------------------------------
# The evidence-consistency gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatentKVConsistency:
    """The two exact identities that bind one gated-MLA evidence record."""

    probabilities: Any
    scores: Any
    context: Any
    readout_relative_rmse: float
    delta_relative_rmse: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _report(dict(self.diagnostics)))


def latent_kv_consistency(
    evidence: ChimeraLatentKVLayerEvidence,
    parameters: ChimeraLatentKVParameters,
    *,
    attention_mask: Any | None = None,
    row_mask: Any | None = None,
    readout_tolerance: float = 1e-5,
    delta_tolerance: float = 1e-5,
    fail_closed: bool = True,
) -> LatentKVConsistency:
    """Rebuild the sealed branch write from the derived rows, and check it.

    Two identities, both exact in exact arithmetic and both measured here:

    1. ``kv_up(kv_norm(latent))`` recomputed from the *captured* normalized
       latent and the carried ``kv_up`` weight against the *captured*
       ``kv_up`` output.  This is the MLA absorption statement made concrete:
       everything the key side contributes to any score passes through those
       ``kv_latent_size`` coordinates.
    2. ``out(sigmoid(gate) * softmax(scores) v)`` rebuilt from the derived
       split, the derived shared-key broadcast and the derived probabilities
       against the *captured* branch delta.

    The parent computes in fp32 (``layers.py:301-306``), so the measured
    numbers are fp32 round-off rather than zero; the tolerances are the sealed
    policy's.
    """

    torch = _torch()
    if not isinstance(evidence, ChimeraLatentKVLayerEvidence):
        raise HFCausalLatentKVAttentionFoldError(
            "evidence must be a ChimeraLatentKVLayerEvidence."
        )
    if not isinstance(parameters, ChimeraLatentKVParameters):
        raise HFCausalLatentKVAttentionFoldError(
            "parameters must be a ChimeraLatentKVParameters."
        )
    if parameters.geometry.as_dict() != evidence.geometry.as_dict():
        raise HFCausalLatentKVAttentionFoldError(
            "The carried parameters and the evidence describe different "
            "geometries."
        )
    readout_tolerance = _nonnegative_float(readout_tolerance, "readout_tolerance")
    delta_tolerance = _nonnegative_float(delta_tolerance, "delta_tolerance")
    geometry = evidence.geometry
    batch, sequence = evidence.batch, evidence.sequence
    mask = (
        torch.ones(batch, sequence, dtype=torch.bool)
        if row_mask is None
        else _mask(row_mask, "row_mask", (batch, sequence))
    )
    keys_visible = (
        torch.ones(batch, sequence, dtype=torch.bool)
        if attention_mask is None
        else _mask(attention_mask, "attention_mask", (batch, sequence))
    )
    readout = evidence.normalized_latent_rows @ parameters.kv_up_weight.T
    readout_error = _relative_rmse(
        _masked_rows(readout, mask), _masked_rows(evidence.kv_rows, mask)
    )
    key_base, value = evidence.head_key_value()
    key = assemble_latent_keys(key_base, evidence.shared_key_rows())
    scores, probabilities = latent_attention_probabilities(
        evidence.head_query(),
        key,
        attention_mask=keys_visible,
        scaling=geometry.attention_scaling,
    )
    context = torch.matmul(probabilities, value)
    gated = torch.sigmoid(evidence.head_gate()) * context
    flat = gated.permute(0, 2, 1, 3).reshape(batch, sequence, geometry.value_width)
    delta = flat @ parameters.output_weight.T
    delta_error = _relative_rmse(
        _masked_rows(delta, mask), _masked_rows(evidence.attention_delta, mask)
    )
    if geometry.extra_head_dim:
        shared_scores = torch.matmul(
            evidence.head_query()[..., geometry.key_head_dim :],
            broadcast_shared_key(
                evidence.shared_key_rows(), geometry.heads
            ).transpose(-1, -2),
        ) * geometry.attention_scaling
        shared_fraction = _energy_fraction(shared_scores, scores)
    else:
        shared_fraction = 0.0
    diagnostics = {
        "schema": SCHEMA,
        "component": "latent_kv_evidence_consistency",
        "layer_index": int(evidence.index),
        "evidence_rows": int(mask.sum().item()),
        "parent_forward_calls": 0,
        "readout_identity_relative_rmse": readout_error,
        "readout_identity_tolerance": readout_tolerance,
        "delta_reconstruction_relative_rmse": delta_error,
        "delta_reconstruction_tolerance": delta_tolerance,
        "shared_key_channel_score_energy_fraction": shared_fraction,
        "shared_key_channels": geometry.extra_head_dim,
        "compressed_query_path": geometry.compressed_query,
        "kv_rows_sha256": _tensor_sha256(evidence.kv_rows),
        "attention_delta_sha256": _tensor_sha256(evidence.attention_delta),
    }
    if fail_closed and readout_error > readout_tolerance:
        raise HFCausalLatentKVAttentionFoldError(
            "The carried kv_up weight does not reproduce the captured "
            f"key/value readout ({readout_error:.3e} > "
            f"{readout_tolerance:.3e})."
        )
    if fail_closed and delta_error > delta_tolerance:
        raise HFCausalLatentKVAttentionFoldError(
            "The compressed-latent attention replay does not reconstruct the "
            f"sealed branch write ({delta_error:.3e} > {delta_tolerance:.3e})."
        )
    return LatentKVConsistency(
        probabilities=probabilities,
        scores=scores,
        context=context,
        readout_relative_rmse=readout_error,
        delta_relative_rmse=delta_error,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Latent bases
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatentBasis:
    """One orthonormal latent basis plus the energy its truncation discards."""

    basis: Any
    retained_energy_fraction: float
    identity: bool

    def project(self, rows: Any) -> Any:
        return (rows @ self.basis).contiguous()


def latent_pod_basis(rows: Any, target_size: int) -> LatentBasis:
    """Leading right singular vectors of the observed latent rows.

    A width-preserving fold takes the identity rather than a rotation it does
    not need, so a child that keeps the parent's latent width is not perturbed
    by an arbitrary change of coordinates.
    """

    torch = _torch()
    matrix = rows.reshape(-1, int(rows.shape[-1]))
    source = int(matrix.shape[1])
    target_size = _positive_int(target_size, "target latent size")
    if target_size > source:
        raise HFCausalLatentKVAttentionFoldError(
            "Latent expansion is outside the contraction-only fold."
        )
    if target_size == source:
        return LatentBasis(
            basis=torch.eye(source, dtype=torch.float64),
            retained_energy_fraction=1.0,
            identity=True,
        )
    _u, singular, vectors = torch.linalg.svd(matrix, full_matrices=False)
    energy = singular.square()
    retained = float(
        (energy[:target_size].sum() / energy.sum().clamp_min(1e-30)).item()
    )
    return LatentBasis(
        basis=vectors[:target_size].T.contiguous(),
        retained_energy_fraction=retained,
        identity=False,
    )


# ---------------------------------------------------------------------------
# The compiled child
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HFCausalLatentKVAttentionFold:
    """One compiled chimera gated-MLA child branch plus its sealed receipt."""

    kv_down_weight: Any
    kv_norm_weight: Any
    kv_up_weight: Any
    gate_weight: Any
    output_weight: Any
    geometry: ChimeraLatentKVGeometry
    report: Mapping[str, Any]
    query_down_weight: Any | None = None
    query_norm_weight: Any | None = None
    query_up_weight: Any | None = None
    query_weight: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, ChimeraLatentKVGeometry):
            raise HFCausalLatentKVAttentionFoldError(
                "geometry must be a ChimeraLatentKVGeometry."
            )
        geometry = self.geometry
        shapes = {
            "kv_down_weight": (geometry.compressed_width, geometry.hidden_size),
            "kv_up_weight": (geometry.key_value_width, geometry.kv_latent_size),
            "gate_weight": (geometry.value_width, geometry.hidden_size),
            "output_weight": (geometry.hidden_size, geometry.value_width),
        }
        for name, shape in shapes.items():
            value = _tensor(getattr(self, name), name, ndim=2).clone()
            if tuple(int(size) for size in value.shape) != shape:
                raise HFCausalLatentKVAttentionFoldError(
                    f"{name} must have shape {shape}."
                )
            object.__setattr__(self, name, value)
        gain = _tensor(self.kv_norm_weight, "kv_norm_weight", ndim=1).clone()
        if int(gain.numel()) != geometry.kv_latent_size:
            raise HFCausalLatentKVAttentionFoldError(
                "kv_norm_weight has the wrong latent width."
            )
        object.__setattr__(self, "kv_norm_weight", gain)
        if geometry.compressed_query:
            if self.query_weight is not None:
                raise HFCausalLatentKVAttentionFoldError(
                    "A compressed query path emits no dense q_proj."
                )
            query_shapes = {
                "query_down_weight": (
                    geometry.query_latent_size,
                    geometry.hidden_size,
                ),
                "query_up_weight": (
                    geometry.query_width,
                    geometry.query_latent_size,
                ),
            }
            for name, shape in query_shapes.items():
                value = getattr(self, name)
                if value is None:
                    raise HFCausalLatentKVAttentionFoldError(f"{name} is required.")
                matrix = _tensor(value, name, ndim=2).clone()
                if tuple(int(size) for size in matrix.shape) != shape:
                    raise HFCausalLatentKVAttentionFoldError(
                        f"{name} must have shape {shape}."
                    )
                object.__setattr__(self, name, matrix)
            query_gain = _tensor(
                self.query_norm_weight, "query_norm_weight", ndim=1
            ).clone()
            if int(query_gain.numel()) != geometry.query_latent_size:
                raise HFCausalLatentKVAttentionFoldError(
                    "query_norm_weight has the wrong latent width."
                )
            object.__setattr__(self, "query_norm_weight", query_gain)
        else:
            if any(
                getattr(self, name) is not None
                for name in (
                    "query_down_weight",
                    "query_norm_weight",
                    "query_up_weight",
                )
            ):
                raise HFCausalLatentKVAttentionFoldError(
                    "A dense query path emits no compressed query tensors."
                )
            matrix = _tensor(self.query_weight, "query_weight", ndim=2).clone()
            if tuple(int(size) for size in matrix.shape) != (
                geometry.query_width,
                geometry.hidden_size,
            ):
                raise HFCausalLatentKVAttentionFoldError(
                    "query_weight has the wrong shape."
                )
            object.__setattr__(self, "query_weight", matrix)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    # -- replay ------------------------------------------------------------

    def _normalize(self, rows: Any, gain: Any) -> Any:
        epsilon = self.geometry.rms_norm_epsilon
        return rows * (
            rows.square().mean(dim=-1, keepdim=True) + epsilon
        ).rsqrt() * gain.reshape(1, 1, -1)

    def project_query(self, child_rows: Any) -> Any:
        """``[batch, heads, sequence, query_head_dim]`` installed queries."""

        geometry = self.geometry
        if geometry.compressed_query:
            latent = child_rows @ self.query_down_weight.T
            flat = self._normalize(latent, self.query_norm_weight) @ (
                self.query_up_weight.T
            )
        else:
            flat = child_rows @ self.query_weight.T
        batch, sequence = int(child_rows.shape[0]), int(child_rows.shape[1])
        return (
            flat.reshape(batch, sequence, geometry.heads, geometry.query_head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def project_key_value(self, child_rows: Any) -> tuple[Any, Any, Any]:
        """Installed keys, values and the child's own normalized latent."""

        geometry = self.geometry
        compressed = child_rows @ self.kv_down_weight.T
        latent = compressed[..., : geometry.kv_latent_size]
        shared = compressed[..., geometry.kv_latent_size :]
        normalized = self._normalize(latent, self.kv_norm_weight)
        blocks = normalized @ self.kv_up_weight.T
        batch, sequence = int(child_rows.shape[0]), int(child_rows.shape[1])
        blocks = blocks.reshape(
            batch,
            sequence,
            geometry.heads,
            geometry.key_head_dim + geometry.value_head_dim,
        ).permute(0, 2, 1, 3)
        key = assemble_latent_keys(
            blocks[..., : geometry.key_head_dim].contiguous(), shared
        )
        value = blocks[..., geometry.key_head_dim :].contiguous()
        return key, value, normalized

    def replay(self, child_rows: Any, *, attention_mask: Any) -> tuple[Any, Any]:
        """Replay the whole branch in FP64; return the delta and probabilities."""

        torch = _torch()
        rows = _tensor(child_rows, "child_rows", ndim=3)
        geometry = self.geometry
        batch, sequence = int(rows.shape[0]), int(rows.shape[1])
        mask = _mask(attention_mask, "attention_mask", (batch, sequence))
        query = self.project_query(rows)
        key, value, _normalized = self.project_key_value(rows)
        _scores, probabilities = latent_attention_probabilities(
            query, key, attention_mask=mask, scaling=geometry.attention_scaling
        )
        context = torch.matmul(probabilities, value)
        gate = torch.sigmoid(rows @ self.gate_weight.T).reshape(
            batch, sequence, geometry.heads, geometry.value_head_dim
        ).permute(0, 2, 1, 3)
        flat = (gate * context).permute(0, 2, 1, 3).reshape(
            batch, sequence, geometry.value_width
        )
        return (flat @ self.output_weight.T).contiguous(), probabilities

    def child_state_dict(self) -> dict[str, Any]:
        """Emit the compiled child in chimera gated-MLA parameter names."""

        state = {
            "kv_down.weight": self.kv_down_weight,
            "kv_norm.weight": self.kv_norm_weight,
            "kv_up.weight": self.kv_up_weight,
            "gate.weight": self.gate_weight,
            "out.weight": self.output_weight,
        }
        if self.geometry.compressed_query:
            state["q_down.weight"] = self.query_down_weight
            state["q_norm.weight"] = self.query_norm_weight
            state["q_up.weight"] = self.query_up_weight
        else:
            state["q_proj.weight"] = self.query_weight
        return state


def _replay_latent_kv_bf16(
    fold: HFCausalLatentKVAttentionFold,
    child_rows: Any,
    *,
    attention_mask: Any,
) -> Any:
    """Replay every intermediate of the installed branch in BF16."""

    torch = _torch()
    geometry = fold.geometry
    rows = child_rows.to(dtype=torch.bfloat16)
    batch, sequence = int(rows.shape[0]), int(rows.shape[1])

    def cast(value: Any) -> Any:
        return value.to(dtype=torch.bfloat16)

    def normalize(values: Any, gain: Any) -> Any:
        epsilon = torch.tensor(geometry.rms_norm_epsilon, dtype=torch.bfloat16)
        scale = (values.square().mean(dim=-1, keepdim=True) + epsilon).rsqrt()
        return values * scale * cast(gain).reshape(1, 1, -1)

    if geometry.compressed_query:
        latent = rows @ cast(fold.query_down_weight).transpose(0, 1)
        query_flat = normalize(latent, fold.query_norm_weight) @ cast(
            fold.query_up_weight
        ).transpose(0, 1)
    else:
        query_flat = rows @ cast(fold.query_weight).transpose(0, 1)
    query = query_flat.reshape(
        batch, sequence, geometry.heads, geometry.query_head_dim
    ).permute(0, 2, 1, 3)
    compressed = rows @ cast(fold.kv_down_weight).transpose(0, 1)
    latent = compressed[..., : geometry.kv_latent_size]
    shared = compressed[..., geometry.kv_latent_size :]
    blocks = normalize(latent, fold.kv_norm_weight) @ cast(
        fold.kv_up_weight
    ).transpose(0, 1)
    blocks = blocks.reshape(
        batch,
        sequence,
        geometry.heads,
        geometry.key_head_dim + geometry.value_head_dim,
    ).permute(0, 2, 1, 3)
    key = blocks[..., : geometry.key_head_dim]
    value = blocks[..., geometry.key_head_dim :]
    if geometry.extra_head_dim:
        key = torch.cat(
            [
                key,
                shared.reshape(batch, 1, sequence, geometry.extra_head_dim).expand(
                    batch, geometry.heads, sequence, geometry.extra_head_dim
                ),
            ],
            dim=-1,
        )
    scores = torch.matmul(query, key.transpose(-1, -2)) * torch.tensor(
        geometry.attention_scaling, dtype=torch.bfloat16
    )
    causal = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    allowed = attention_mask.reshape(batch, 1, 1, sequence) & causal.reshape(
        1, 1, sequence, sequence
    )
    probabilities = torch.softmax(
        scores.masked_fill(~allowed, torch.finfo(torch.bfloat16).min), dim=-1
    )
    context = torch.matmul(probabilities, value)
    gate = torch.sigmoid(rows @ cast(fold.gate_weight).transpose(0, 1)).reshape(
        batch, sequence, geometry.heads, geometry.value_head_dim
    ).permute(0, 2, 1, 3)
    flat = (gate * context).permute(0, 2, 1, 3).reshape(
        batch, sequence, geometry.value_width
    )
    delta = flat @ cast(fold.output_weight).transpose(0, 1)
    return delta.to(dtype=torch.float64)


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------


def _merge_heads(value: Any, groups: tuple[tuple[int, ...], ...]) -> Any:
    torch = _torch()
    return torch.stack(
        [value[:, list(group)].mean(dim=1) for group in groups], dim=1
    ).contiguous()


def compile_hf_causal_latent_kv_attention_fold(
    evidence: ChimeraLatentKVLayerEvidence,
    parameters: ChimeraLatentKVParameters,
    geometry: ChimeraLatentKVGeometry,
    *,
    attention_mask: Any,
    fit_token_mask: Any,
    validation_token_mask: Any,
    valid_token_mask: Any | None = None,
    child_normalized_rows: Any | None = None,
    endpoint_target: Any | None = None,
    policy: ChimeraLatentKVFoldPolicy | None = None,
    consistency: LatentKVConsistency | None = None,
) -> HFCausalLatentKVAttentionFold:
    """Compile one chimera gated-MLA branch into a smaller one, closed form.

    Ordered recipe of design section 1.3: evidence gates, head assignment and
    channel selection, latent POD bases, consumer ridges with the child's own
    normalizations replayed, installed-probability replay, gate ridge and an
    endpoint-coupled writer, BF16 replay, no-branch floor.
    """

    torch = _torch()
    if not isinstance(evidence, ChimeraLatentKVLayerEvidence):
        raise HFCausalLatentKVAttentionFoldError(
            "evidence must be a ChimeraLatentKVLayerEvidence."
        )
    if not isinstance(geometry, ChimeraLatentKVGeometry):
        raise HFCausalLatentKVAttentionFoldError(
            "geometry must be a ChimeraLatentKVGeometry."
        )
    source = evidence.geometry
    policy = policy or ChimeraLatentKVFoldPolicy()
    if geometry.extra_head_dim != source.extra_head_dim:
        raise HFCausalLatentKVAttentionFoldError(
            "The shared key channel count is a contract constant, not a fold "
            f"coordinate; changing it is a kind translation (see "
            f"{DESIGN_DOCUMENT} section 1.1)."
        )
    if geometry.compressed_query != source.compressed_query:
        raise HFCausalLatentKVAttentionFoldError(
            "The compressed query path is a contract shape of the crossing: "
            "both sides must declare a query latent or neither may."
        )
    batch, sequence = evidence.batch, evidence.sequence
    child = (
        evidence.attention_norm_output
        if child_normalized_rows is None
        else _tensor(child_normalized_rows, "child_normalized_rows", ndim=3)
    )
    if tuple(int(size) for size in child.shape[:2]) != (batch, sequence):
        raise HFCausalLatentKVAttentionFoldError(
            "child rows have the wrong sequence shape."
        )
    if int(child.shape[2]) != geometry.hidden_size:
        raise HFCausalLatentKVAttentionFoldError(
            "child rows do not match the child hidden size."
        )
    delta_target = (
        evidence.attention_delta
        if endpoint_target is None
        else _tensor(endpoint_target, "endpoint_target", ndim=3)
    )
    if int(delta_target.shape[2]) != geometry.hidden_size:
        raise HFCausalLatentKVAttentionFoldError(
            "endpoint_target width differs from the child residual width."
        )
    mask = _mask(attention_mask, "attention_mask", (batch, sequence))
    fit = _mask(fit_token_mask, "fit_token_mask", (batch, sequence))
    validation = _mask(
        validation_token_mask, "validation_token_mask", (batch, sequence)
    )
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
        raise HFCausalLatentKVAttentionFoldError(
            "MLA fit/validation masks must be disjoint nonempty valid subsets."
        )

    # Step 1 -- the ridge-independent evidence gates.
    gates = consistency or latent_kv_consistency(
        evidence,
        parameters,
        attention_mask=mask,
        row_mask=valid,
        readout_tolerance=policy.readout_identity_tolerance,
        delta_tolerance=policy.delta_reconstruction_tolerance,
    )

    # Step 2 -- head assignment and channel selection on the merged targets.
    groups = _balanced_groups(source.heads, geometry.heads)
    key_base, value = evidence.head_key_value()
    query = evidence.head_query()
    query_base = query[..., : source.key_head_dim]
    query_shared = query[..., source.key_head_dim :]
    gate = evidence.head_gate()
    merged_query_base = _merge_heads(query_base, groups)
    merged_query_shared = _merge_heads(query_shared, groups)
    merged_key = _merge_heads(key_base, groups)
    merged_value = _merge_heads(value, groups)
    merged_gate = _merge_heads(gate, groups)
    merged_probabilities = _merge_heads(gates.probabilities, groups)
    key_channels = _select_channels(
        (merged_query_base, merged_key), geometry.key_head_dim
    )
    value_channels = _select_channels(
        (merged_value, merged_gate), geometry.value_head_dim
    )
    desired_query = torch.cat(
        [_take(merged_query_base, key_channels), merged_query_shared], dim=-1
    )
    desired_key = _take(merged_key, key_channels)
    desired_value = _take(merged_value, value_channels)
    desired_gate = _take(merged_gate, value_channels)

    def head_rows(value_: Any) -> Any:
        return value_.permute(0, 2, 1, 3).reshape(batch, sequence, -1).contiguous()

    child_fit = _masked_rows(child, fit)

    def ridge_from_child(target: Any) -> Any:
        return _ridge(child_fit, _masked_rows(target, fit), policy.ridge).T.contiguous()

    # Steps 3 and 4 -- the key/value latent path.
    latent_basis = latent_pod_basis(
        _masked_rows(evidence.latent_rows(), fit), geometry.kv_latent_size
    )
    compressed_target = torch.cat(
        [
            latent_basis.project(evidence.latent_rows()),
            evidence.shared_key_rows(),
        ],
        dim=-1,
    )
    kv_down_weight = ridge_from_child(compressed_target)
    child_compressed = child @ kv_down_weight.T
    child_latent = child_compressed[..., : geometry.kv_latent_size]
    normalized_target = latent_basis.project(evidence.normalized_latent_rows)
    kv_norm_fold = fit_rmsnorm_diagonal(
        _masked_rows(child_latent, fit),
        _masked_rows(normalized_target, fit),
        epsilon=geometry.rms_norm_epsilon,
        ridge=policy.ridge,
    )
    child_normalized_latent = child_latent * (
        child_latent.square().mean(dim=-1, keepdim=True)
        + geometry.rms_norm_epsilon
    ).rsqrt() * kv_norm_fold.weight.reshape(1, 1, -1)
    kv_up_weight = _ridge(
        _masked_rows(child_normalized_latent, fit),
        _masked_rows(
            head_rows(torch.cat([desired_key, desired_value], dim=-1)), fit
        ),
        policy.ridge,
    ).T.contiguous()

    # Step 5 -- the query path.
    query_norm_fold = None
    query_basis = None
    if geometry.compressed_query:
        query_basis = latent_pod_basis(
            _masked_rows(evidence.query_latent_rows, fit), geometry.query_latent_size
        )
        query_down_weight = ridge_from_child(
            query_basis.project(evidence.query_latent_rows)
        )
        child_query_latent = child @ query_down_weight.T
        query_norm_fold = fit_rmsnorm_diagonal(
            _masked_rows(child_query_latent, fit),
            _masked_rows(
                query_basis.project(evidence.normalized_query_latent_rows), fit
            ),
            epsilon=geometry.rms_norm_epsilon,
            ridge=policy.ridge,
        )
        child_normalized_query = child_query_latent * (
            child_query_latent.square().mean(dim=-1, keepdim=True)
            + geometry.rms_norm_epsilon
        ).rsqrt() * query_norm_fold.weight.reshape(1, 1, -1)
        query_up_weight = _ridge(
            _masked_rows(child_normalized_query, fit),
            _masked_rows(head_rows(desired_query), fit),
            policy.ridge,
        ).T.contiguous()
        query_weight = None
    else:
        query_down_weight = None
        query_up_weight = None
        query_weight = ridge_from_child(head_rows(desired_query))

    # Step 6 -- installed-probability replay.
    gate_weight = ridge_from_child(head_rows(desired_gate))
    staged = HFCausalLatentKVAttentionFold(
        kv_down_weight=kv_down_weight,
        kv_norm_weight=kv_norm_fold.weight,
        kv_up_weight=kv_up_weight,
        gate_weight=gate_weight,
        output_weight=torch.zeros(
            geometry.hidden_size, geometry.value_width, dtype=torch.float64
        ),
        geometry=geometry,
        report={"schema": SCHEMA, "component": "staged_before_receipt"},
        query_down_weight=query_down_weight,
        query_norm_weight=(
            None if query_norm_fold is None else query_norm_fold.weight
        ),
        query_up_weight=query_up_weight,
        query_weight=query_weight,
    )
    child_query = staged.project_query(child)
    child_key, child_value, _child_normalized = staged.project_key_value(child)
    _child_scores, child_probabilities = latent_attention_probabilities(
        child_query,
        child_key,
        attention_mask=mask,
        scaling=geometry.attention_scaling,
    )
    overlap = float(
        torch.minimum(child_probabilities, merged_probabilities)
        .sum(dim=-1)
        .permute(0, 2, 1)[valid]
        .mean()
        .item()
    )
    if overlap < policy.minimum_attention_probability_overlap:
        raise HFCausalLatentKVAttentionFoldError(
            f"The replayed compressed-latent probabilities keep only "
            f"{overlap:.4f} of the parent's attention mass, below the sealed "
            f"policy fraction ({policy.minimum_attention_probability_overlap:.4f})."
        )

    # Steps 7 and 8 -- the gate saturation report and the endpoint-coupled writer.
    child_gate = torch.sigmoid(
        (child @ gate_weight.T)
        .reshape(batch, sequence, geometry.heads, geometry.value_head_dim)
        .permute(0, 2, 1, 3)
    )
    saturated = (
        (child_gate < policy.gate_saturation_epsilon)
        | (child_gate > 1.0 - policy.gate_saturation_epsilon)
    )
    saturated_fraction = float(
        saturated.permute(0, 2, 1, 3)[valid].to(dtype=torch.float64).mean().item()
    )
    if saturated_fraction > policy.maximum_gate_saturated_row_fraction:
        raise HFCausalLatentKVAttentionFoldError(
            f"The installed output gate is saturated on {saturated_fraction:.4f} "
            "of the evidence, above the sealed policy fraction "
            f"({policy.maximum_gate_saturated_row_fraction:.4f})."
        )
    context = torch.matmul(child_probabilities, child_value)
    flat_context = (child_gate * context).permute(0, 2, 1, 3).reshape(
        batch, sequence, geometry.value_width
    )
    output_weight = _ridge(
        _masked_rows(flat_context, fit),
        _masked_rows(delta_target, fit),
        policy.ridge,
    ).T.contiguous()
    fold = HFCausalLatentKVAttentionFold(
        kv_down_weight=kv_down_weight,
        kv_norm_weight=kv_norm_fold.weight,
        kv_up_weight=kv_up_weight,
        gate_weight=gate_weight,
        output_weight=output_weight,
        geometry=geometry,
        report={"schema": SCHEMA, "component": "staged_before_receipt"},
        query_down_weight=query_down_weight,
        query_norm_weight=(
            None if query_norm_fold is None else query_norm_fold.weight
        ),
        query_up_weight=query_up_weight,
        query_weight=query_weight,
    )
    predicted, _probabilities = fold.replay(child, attention_mask=mask)

    # Step 9 -- BF16 replay, then the no-branch floor.
    predicted_bf16 = (
        _replay_latent_kv_bf16(fold, child, attention_mask=mask)
        if policy.bf16_replay
        else predicted
    )
    fit_target = _masked_rows(delta_target, fit)
    validation_target = _masked_rows(delta_target, validation)
    zero_baseline = _relative_rmse(
        torch.zeros_like(validation_target), validation_target
    )
    validation_error = _relative_rmse(
        _masked_rows(predicted, validation), validation_target
    )
    validation_error_bf16 = _relative_rmse(
        _masked_rows(predicted_bf16, validation), validation_target
    )
    if validation_error_bf16 >= zero_baseline:
        raise HFCausalLatentKVAttentionFoldError(
            "The BF16 validation endpoint is at or above the no-branch baseline "
            f"({validation_error_bf16:.6f} >= {zero_baseline:.6f})."
        )
    body = {
        "schema": SCHEMA,
        "component": "chimera_latent_kv_attention_branch_fold",
        "method": (
            "evidence_gated_barycentric_head_latent_pod_consumer_ridges_"
            "installed_probability_replay_endpoint_coupled_gated_writer"
        ),
        "layer_index": int(evidence.index),
        "source_geometry": source.as_dict(),
        "target_geometry": geometry.as_dict(),
        "policy": policy.as_dict(),
        "consistency_report_sha256": gates.diagnostics["report_sha256"],
        "readout_identity_relative_rmse": gates.readout_relative_rmse,
        "delta_reconstruction_relative_rmse": gates.delta_relative_rmse,
        "shared_key_channel_score_energy_fraction": gates.diagnostics[
            "shared_key_channel_score_energy_fraction"
        ],
        "shared_key_channels_are_head_independent": True,
        "source_head_groups": [list(group) for group in groups],
        "all_source_heads_covered_exactly_once": True,
        "selected_key_channels": list(key_channels),
        "selected_value_channels": list(value_channels),
        "kv_latent_basis": "identity" if latent_basis.identity else "observed_row_pod",
        "kv_latent_retained_energy_fraction": latent_basis.retained_energy_fraction,
        "query_latent_basis": (
            None
            if query_basis is None
            else ("identity" if query_basis.identity else "observed_row_pod")
        ),
        "query_latent_retained_energy_fraction": (
            None if query_basis is None else query_basis.retained_energy_fraction
        ),
        "compressed_query_path": geometry.compressed_query,
        "child_normalizations_replayed_not_fitted_through": True,
        "kv_norm_report_sha256": kv_norm_fold.report["report_sha256"],
        "kv_norm_gain_relative_rmse": kv_norm_fold.report["relative_rmse"],
        "query_norm_report_sha256": (
            None if query_norm_fold is None else query_norm_fold.report["report_sha256"]
        ),
        "query_norm_gain_relative_rmse": (
            None if query_norm_fold is None else query_norm_fold.report["relative_rmse"]
        ),
        "qk_installed_before_writer_fit": True,
        "target_softmax_probabilities_replayed": True,
        "attention_probability_overlap": overlap,
        "attention_probability_relative_rmse": _relative_rmse(
            child_probabilities.permute(0, 2, 1, 3)[valid],
            merged_probabilities.permute(0, 2, 1, 3)[valid],
        ),
        "gate_fitted_against_captured_pre_activation": True,
        "gate_saturated_row_fraction": saturated_fraction,
        "score_chart_used": False,
        "score_chart_not_required": (
            "every_mla_score_is_an_inner_product_in_head_coordinates"
        ),
        "fit_rows": int(fit.sum().item()),
        "validation_rows": int(validation.sum().item()),
        "fit_residual_relative_rmse": _relative_rmse(
            _masked_rows(predicted, fit), fit_target
        ),
        "fit_residual_retained_energy_fraction": _retained_energy(
            _masked_rows(predicted, fit), fit_target
        ),
        "validation_residual_relative_rmse": validation_error,
        "validation_residual_retained_energy_fraction": _retained_energy(
            _masked_rows(predicted, validation), validation_target
        ),
        "validation_no_branch_relative_rmse": zero_baseline,
        "validation_beats_no_branch_baseline": bool(
            validation_error < zero_baseline
        ),
        "validation_residual_relative_rmse_bf16": validation_error_bf16,
        "bf16_versus_fp64_endpoint_relative_delta": _relative_rmse(
            _masked_rows(predicted_bf16, validation),
            _masked_rows(predicted, validation),
        ),
        "ridge": policy.ridge,
        "numeric_weight_matching_objective": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
        "kv_down_weight_sha256": _tensor_sha256(kv_down_weight),
        "kv_up_weight_sha256": _tensor_sha256(kv_up_weight),
        "gate_weight_sha256": _tensor_sha256(gate_weight),
        "output_weight_sha256": _tensor_sha256(output_weight),
    }
    return HFCausalLatentKVAttentionFold(
        kv_down_weight=kv_down_weight,
        kv_norm_weight=kv_norm_fold.weight,
        kv_up_weight=kv_up_weight,
        gate_weight=gate_weight,
        output_weight=output_weight,
        geometry=geometry,
        report=_report(body),
        query_down_weight=query_down_weight,
        query_norm_weight=(
            None if query_norm_fold is None else query_norm_fold.weight
        ),
        query_up_weight=query_up_weight,
        query_weight=query_weight,
    )


__all__ = [
    "DESIGN_DOCUMENT",
    "EVIDENCE_SCHEMA",
    "SCHEMA",
    "ChimeraLatentKVFoldPolicy",
    "ChimeraLatentKVGeometry",
    "ChimeraLatentKVLayerEvidence",
    "ChimeraLatentKVParameters",
    "HFCausalLatentKVAttentionFold",
    "HFCausalLatentKVAttentionFoldError",
    "LatentBasis",
    "LatentKVConsistency",
    "assemble_latent_keys",
    "broadcast_shared_key",
    "capture_chimera_latent_kv_layer_evidence",
    "chimera_latent_kv_geometry",
    "chimera_latent_kv_parameters",
    "compile_hf_causal_latent_kv_attention_fold",
    "latent_attention_probabilities",
    "latent_kv_consistency",
    "latent_pod_basis",
]
