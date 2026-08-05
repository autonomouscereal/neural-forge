"""Chimera-omni CSA/HCA sparse-latent residual-branch folding.

``csa`` and ``hca`` are the *same* module at two strides
(``chimera-omni/src/chimera_omni/modeling/layers.py:198-228``,
``config.py:60-61``), so this is one stride-parameterized kernel, not two.
The branch writes one residual delta built from two additive sub-branches: a
windowed NoPE GQA local attention and a single-head full-width NoPE softmax
attention over mean-pooled block summaries with a zero row prepended.

This implements section 1.2 of
``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md``: exact delta split,
the extended GQA kernel for the local sub-branch (``attention_window`` +
``position_encoding = "nope"``), a score-ridge / probability-replay /
endpoint-coupled V-O solve for the summary sub-branch, and a bounded two-round
alternation between them.  Every solve is closed form; there is no optimizer,
gradient, parent forward, runtime sidecar, or database in this file.

Implementation deviations from the design document, with reasons
----------------------------------------------------------------

1. **Mean pooling is materialized as an explicit linear operator.**
   ``summary_pooling_matrix`` builds ``P`` with the zero summary as its first
   row, so pooling is a matrix product rather than a reshape.  That makes the
   design's chart-commutation claim (section 2, "mean(h) E = mean(h E)") a
   testable identity instead of a comment, and it makes the padding behaviour
   explicit: chimera pads the tail block with zeros before averaging over
   ``stride`` slots (``layers.py:210-213``), and that partial block's summary
   is never visible to any token, so the padding cannot leak.

2. **"Mass on assigned blocks" is defined as distribution overlap.**  The
   design's fail-closed condition 5 asks that the replayed summary
   probabilities place at least a policy fraction of mass on assigned blocks
   without defining the statistic.  This uses
   ``mean_t sum_j min(P_child, P_parent)`` -- the total-variation overlap of
   the two distributions, which is exactly "how much of the parent's block
   assignment the child kept", is bounded in ``[0, 1]``, and degrades
   gracefully instead of being trivially one (every masked replay places all
   of its mass on *allowed* blocks by construction).

3. **Cross-width score targets need an explicit chart.**  ``Wq``/``Wk``/``Wv``
   /``W_out`` are residual-width square, so a narrower child cannot be fitted
   to parent q/k rows: the score is an inner product in the *residual* width.
   The transported targets are ``q_p E`` and ``k_p E`` for an orthonormal
   ``score_chart`` ``E``; without one a width change fails closed rather than
   silently fitting a score of a different geometry.  Pooling commutes with
   ``E`` exactly, so raw and pooled rows share one chart (design section 2).

4. **Alternation refits the local branch by recompilation.**  A local V/O-only
   refit would need to reach inside the GQA kernel.  Recompiling the local fold
   against the updated target reuses the kernel unchanged; its Q/K solve does
   not depend on the endpoint target, so the recompiled Q/K are identical and
   only the writer moves -- which is exactly the design's sub-step.

5. **Monotonicity is guarded, not assumed.**  Each alternation sub-step is an
   exact ridge given the other branch, so the *regularized* objective cannot
   rise; the unregularized objective can.  Every sub-step is applied and
   reverted if the fit objective rose, the same guard the committed kind
   translation uses (``hf_causal_kind_translation.py`` deviation 6).
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
from .hf_causal_gqa_attention_fold import (
    HFCausalCoupledAttentionFold,
    HFCausalGQATopology,
    causal_attention_probabilities,
    compile_hf_causal_gqa_attention_fold,
)


SCHEMA = "neural-forge-chimera-summary-attention-fold-v1"
EVIDENCE_SCHEMA = "neural-forge-chimera-summary-attention-evidence-v1"
WINDOWED_ATTENTION_SCHEMA = "neural-forge-causal-windowed-attention-fold-v1"
GQA_TO_SUMMARY_SCHEMA = "neural-forge-causal-gqa-to-summary-attention-fold-v1"

DESIGN_DOCUMENT = "docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md"


class HFCausalSparseLatentFoldError(ValueError):
    """A CSA/HCA fold is unsupported or its evidence is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise HFCausalSparseLatentFoldError(
            "PyTorch is required for sparse-latent folding."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalSparseLatentFoldError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalSparseLatentFoldError(f"{name} must be positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise HFCausalSparseLatentFoldError(f"{name} must be positive and finite.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalSparseLatentFoldError(f"{name} must be non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise HFCausalSparseLatentFoldError(
            f"{name} must be finite and non-negative."
        )
    return result


def _tensor(value: Any, name: str, *, ndim: int | None = None) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise HFCausalSparseLatentFoldError(f"{name} must be a tensor.")
    if ndim is not None and value.ndim != ndim:
        raise HFCausalSparseLatentFoldError(
            f"{name} must be rank {ndim}, got rank {value.ndim}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise HFCausalSparseLatentFoldError(f"{name} contains non-finite values.")
    return result


def _mask(value: Any, name: str, shape: tuple[int, int]) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise HFCausalSparseLatentFoldError(f"{name} must have shape {shape}.")
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
        raise HFCausalSparseLatentFoldError("Ridge evidence matrices are incompatible.")
    gram = source.T @ source
    return torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=torch.float64),
        source.T @ target,
    )


# ---------------------------------------------------------------------------
# Geometry, policy, evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChimeraSparseLatentGeometry:
    """One ``SparseLatentAttention`` geometry; ``stride`` picks CSA or HCA."""

    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    local_window: int
    summary_stride: int
    family: str = "csa"

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "query_heads",
            "key_value_heads",
            "head_dim",
            "local_window",
            "summary_stride",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.query_heads % self.key_value_heads:
            raise HFCausalSparseLatentFoldError(
                "query_heads must be divisible by key_value_heads."
            )
        family = str(self.family).strip().lower()
        if family not in {"csa", "hca"}:
            raise HFCausalSparseLatentFoldError("family must be 'csa' or 'hca'.")
        object.__setattr__(self, "family", family)

    @property
    def local_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def summary_scaling(self) -> float:
        return 1.0 / math.sqrt(float(self.hidden_size))

    def local_topology(self) -> HFCausalGQATopology:
        """The windowed NoPE GQA topology of the local sub-branch."""

        return HFCausalGQATopology(
            query_heads=self.query_heads,
            key_value_heads=self.key_value_heads,
            head_dim=self.head_dim,
            rope_theta=10000.0,
            attention_scaling=1.0 / math.sqrt(float(self.head_dim)),
            attention_window=self.local_window,
            position_encoding="nope",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }


@dataclass(frozen=True, slots=True)
class ChimeraSparseLatentFoldPolicy:
    """Frozen policy constants for one CSA/HCA fold."""

    ridge: float = 1e-6
    alternation_rounds: int = 2
    split_reconstruction_tolerance: float = 1e-6
    minimum_summary_probability_overlap: float = 0.25
    bf16_replay: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "ridge", _nonnegative_float(self.ridge, "ridge"))
        object.__setattr__(
            self,
            "alternation_rounds",
            _positive_int(self.alternation_rounds, "alternation_rounds"),
        )
        for name in (
            "split_reconstruction_tolerance",
            "minimum_summary_probability_overlap",
        ):
            object.__setattr__(
                self, name, _nonnegative_float(getattr(self, name), name)
            )
        if not isinstance(self.bf16_replay, bool):
            raise HFCausalSparseLatentFoldError("bf16_replay must be boolean.")

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }


@dataclass(frozen=True, slots=True)
class ChimeraSummaryAttentionLayerEvidence:
    """Complete cached circuit states for one chimera CSA/HCA branch.

    Field style mirrors ``HFCausalLayerEvidence``
    (``hf_causal_circuit_evidence.py:214-233``).  ``local_delta`` and
    ``summary_delta`` must sum to the sealed branch write; that identity is the
    evidence-consistency gate of this family.
    """

    index: int
    block_input: Any
    attention_norm_output: Any
    local_delta: Any
    summary_delta: Any
    attention_delta: Any
    block_output: Any
    local_query_rows: Any
    local_key_rows: Any
    local_value_rows: Any
    local_probabilities: Any
    raw_summaries: Any
    summary_query_rows: Any
    summary_key_rows: Any
    summary_value_rows: Any
    summary_scores: Any
    summary_probabilities: Any
    block_ids: tuple[int, ...]
    geometry: ChimeraSparseLatentGeometry
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or (
            self.index < 0
        ):
            raise HFCausalSparseLatentFoldError(
                "layer evidence index must be a non-negative integer."
            )
        if not isinstance(self.geometry, ChimeraSparseLatentGeometry):
            raise HFCausalSparseLatentFoldError(
                "geometry must be a ChimeraSparseLatentGeometry."
            )
        geometry = self.geometry
        token_rows = {
            "block_input": geometry.hidden_size,
            "attention_norm_output": geometry.hidden_size,
            "local_delta": geometry.hidden_size,
            "summary_delta": geometry.hidden_size,
            "attention_delta": geometry.hidden_size,
            "block_output": geometry.hidden_size,
            "local_query_rows": geometry.local_width,
            "local_key_rows": geometry.key_value_heads * geometry.head_dim,
            "local_value_rows": geometry.key_value_heads * geometry.head_dim,
            "summary_query_rows": geometry.hidden_size,
        }
        for name, width in token_rows.items():
            value = _tensor(getattr(self, name), name, ndim=3).clone()
            if int(value.shape[2]) != width:
                raise HFCausalSparseLatentFoldError(f"{name} must have width {width}.")
            object.__setattr__(self, name, value)
        batch = int(self.block_input.shape[0])
        sequence = int(self.block_input.shape[1])
        for name in token_rows:
            if tuple(int(size) for size in getattr(self, name).shape[:2]) != (
                batch,
                sequence,
            ):
                raise HFCausalSparseLatentFoldError(
                    f"{name} has the wrong sequence shape."
                )
        summary_rows = {
            "raw_summaries": geometry.hidden_size,
            "summary_key_rows": geometry.hidden_size,
            "summary_value_rows": geometry.hidden_size,
        }
        for name, width in summary_rows.items():
            value = _tensor(getattr(self, name), name, ndim=3).clone()
            if int(value.shape[0]) != batch or int(value.shape[2]) != width:
                raise HFCausalSparseLatentFoldError(
                    f"{name} must be [batch, summaries, {width}]."
                )
            object.__setattr__(self, name, value)
        summaries = int(self.summary_key_rows.shape[1])
        if int(self.raw_summaries.shape[1]) != summaries - 1:
            raise HFCausalSparseLatentFoldError(
                "raw_summaries must be one row shorter than the keyed summaries; "
                "the zero row is prepended by the parent."
            )
        for name in ("summary_scores", "summary_probabilities"):
            value = _tensor(getattr(self, name), name, ndim=3).clone()
            if tuple(int(size) for size in value.shape) != (
                batch,
                sequence,
                summaries,
            ):
                raise HFCausalSparseLatentFoldError(
                    f"{name} must be [batch, sequence, summaries]."
                )
            object.__setattr__(self, name, value)
        probabilities = _tensor(self.local_probabilities, "local_probabilities", ndim=4)
        if tuple(int(size) for size in probabilities.shape) != (
            batch,
            geometry.query_heads,
            sequence,
            sequence,
        ):
            raise HFCausalSparseLatentFoldError(
                "local_probabilities must be [batch, query_heads, S, S]."
            )
        object.__setattr__(self, "local_probabilities", probabilities.clone())
        block_ids = tuple(int(value) for value in self.block_ids)
        if len(block_ids) != sequence or any(
            value != position // geometry.summary_stride
            for position, value in enumerate(block_ids)
        ):
            raise HFCausalSparseLatentFoldError(
                "block_ids must equal floor(position / stride) for every token."
            )
        object.__setattr__(self, "block_ids", block_ids)
        object.__setattr__(self, "metadata", _report(dict(self.metadata)))

    @property
    def batch(self) -> int:
        return int(self.block_input.shape[0])

    @property
    def sequence(self) -> int:
        return int(self.block_input.shape[1])

    @property
    def summaries(self) -> int:
        return int(self.summary_key_rows.shape[1])

    def tensor_items(self) -> tuple[tuple[str, Any], ...]:
        skipped = {"index", "geometry", "metadata", "block_ids"}
        return tuple(
            (descriptor.name, getattr(self, descriptor.name))
            for descriptor in fields(self)
            if descriptor.name not in skipped
        )


# ---------------------------------------------------------------------------
# Pooling, masks, probabilities
# ---------------------------------------------------------------------------


def summary_pooling_matrix(sequence: int, stride: int) -> Any:
    """The exact linear pooling operator ``P`` including the zero summary row.

    ``P`` is ``[summaries, sequence]`` with ``P[0] = 0`` and
    ``P[j, t] = 1/stride`` for tokens ``t`` of block ``j-1``.  The tail block is
    averaged over ``stride`` slots even when it is short, exactly as the parent
    does after zero padding (``layers.py:210-213``).
    """

    torch = _torch()
    sequence = _positive_int(sequence, "sequence")
    stride = _positive_int(stride, "stride")
    blocks = (sequence + stride - 1) // stride
    matrix = torch.zeros(blocks + 1, sequence, dtype=torch.float64)
    for block in range(blocks):
        start = block * stride
        stop = min(start + stride, sequence)
        matrix[block + 1, start:stop] = 1.0 / float(stride)
    return matrix.contiguous()


def summary_block_mask(sequence: int, stride: int) -> Any:
    """``summary j`` is visible to token ``t`` iff ``j <= floor(t/stride)``."""

    torch = _torch()
    pooling = summary_pooling_matrix(sequence, stride)
    summaries = int(pooling.shape[0])
    block_ids = torch.arange(sequence) // stride
    summary_ids = torch.arange(summaries)
    return (summary_ids.reshape(1, -1) <= block_ids.reshape(-1, 1)).contiguous()


def summary_attention_probabilities(
    query: Any,
    key: Any,
    *,
    block_mask: Any,
    scaling: float,
) -> tuple[Any, Any]:
    """Scores and block-causal probabilities for the single-head summary path.

    The parent masks with ``finfo.min`` rather than ``-inf``
    (``layers.py:226``); both send the masked entries to exactly zero after the
    softmax for any finite unmasked score, so the replay is equivalent.
    """

    torch = _torch()
    scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
    allowed = block_mask.reshape(1, int(block_mask.shape[0]), int(block_mask.shape[1]))
    masked = scores.masked_fill(~allowed, -torch.inf)
    return scores.contiguous(), torch.softmax(masked, dim=-1).contiguous()


# ---------------------------------------------------------------------------
# Parent capture
# ---------------------------------------------------------------------------


def chimera_sparse_latent_geometry(
    module: Any,
    *,
    family: str = "csa",
    hidden_size: int | None = None,
) -> ChimeraSparseLatentGeometry:
    """Read one chimera ``SparseLatentAttention`` geometry off a live module."""

    try:
        local = module.local
        width = int(hidden_size or module.summary_q.weight.shape[1])
        window = int(local.window)
        return ChimeraSparseLatentGeometry(
            hidden_size=width,
            query_heads=int(local.num_heads),
            key_value_heads=int(local.num_kv_heads),
            head_dim=int(local.head_dim),
            local_window=window if window > 0 else width,
            summary_stride=int(module.stride),
            family=family,
        )
    except AttributeError as exc:
        raise HFCausalSparseLatentFoldError(
            "module does not expose the chimera SparseLatentAttention layout."
        ) from exc


def capture_chimera_sparse_latent_layer_evidence(
    module: Any,
    normalized_rows: Any,
    *,
    layer_index: int,
    family: str = "csa",
    block_input: Any | None = None,
    block_output: Any | None = None,
    attention_mask: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ChimeraSummaryAttentionLayerEvidence:
    """Run the parent branch exactly once and seal both sub-branches."""

    torch = _torch()
    geometry = chimera_sparse_latent_geometry(module, family=family)
    rows = normalized_rows.detach()
    if rows.ndim != 3 or int(rows.shape[2]) != geometry.hidden_size:
        raise HFCausalSparseLatentFoldError(
            "normalized_rows must be [batch, sequence, hidden_size]."
        )
    batch, sequence = int(rows.shape[0]), int(rows.shape[1])
    mask = (
        torch.ones(batch, sequence, dtype=torch.bool)
        if attention_mask is None
        else attention_mask
    )
    with torch.no_grad():
        local_delta = module.local(rows)
        raw_summaries = module._summaries(rows)
        summaries = torch.cat(
            [torch.zeros_like(raw_summaries[:, :1]), raw_summaries], dim=1
        )
        query = module.summary_q(rows)
        key = module.summary_k(summaries)
        value = module.summary_v(summaries)
        scores, probabilities = summary_attention_probabilities(
            _tensor(query, "summary query"),
            _tensor(key, "summary key"),
            block_mask=summary_block_mask(sequence, geometry.summary_stride),
            scaling=geometry.summary_scaling,
        )
        summary_delta = module.out(
            torch.matmul(probabilities.to(dtype=rows.dtype), value)
        )
        parent_delta = module(rows)
        local_query = module.local.q_proj(rows)
        local_key = module.local.k_proj(rows)
        local_value = module.local.v_proj(rows)
    topology = geometry.local_topology()

    def heads(flat: Any, count: int) -> Any:
        return (
            _tensor(flat, "local projection", ndim=3)
            .reshape(batch, sequence, count, geometry.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    _scores, local_probabilities = causal_attention_probabilities(
        heads(local_query, geometry.query_heads),
        heads(local_key, geometry.key_value_heads),
        attention_mask=_mask(mask, "attention_mask", (batch, sequence)),
        scaling=topology.attention_scaling,
        attention_window=topology.attention_window,
    )
    delta = local_delta + summary_delta
    reconstruction = _relative_rmse(
        _tensor(delta, "delta"), _tensor(parent_delta, "parent_delta")
    )
    sealed = {
        "schema": EVIDENCE_SCHEMA,
        "component": "chimera_sparse_latent_branch_capture",
        "family": geometry.family,
        "summary_stride": geometry.summary_stride,
        "parent_forward_calls": 1,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "capture_reconstruction_relative_rmse": reconstruction,
        **dict(metadata or {}),
    }
    resolved_input = rows if block_input is None else block_input
    return ChimeraSummaryAttentionLayerEvidence(
        index=int(layer_index),
        block_input=resolved_input,
        attention_norm_output=rows,
        local_delta=local_delta,
        summary_delta=summary_delta,
        attention_delta=delta,
        block_output=(
            _tensor(resolved_input, "block_input") + _tensor(delta, "delta")
            if block_output is None
            else block_output
        ),
        local_query_rows=local_query,
        local_key_rows=local_key,
        local_value_rows=local_value,
        local_probabilities=local_probabilities,
        raw_summaries=raw_summaries,
        summary_query_rows=query,
        summary_key_rows=key,
        summary_value_rows=value,
        summary_scores=scores,
        summary_probabilities=probabilities,
        block_ids=tuple(
            position // geometry.summary_stride for position in range(sequence)
        ),
        geometry=geometry,
        metadata=sealed,
    )


# ---------------------------------------------------------------------------
# The exact sub-branch split
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SummarySplit:
    """The exact ``Delta = local + summary`` split of one cached branch."""

    local_delta: Any
    summary_delta: Any
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _report(dict(self.diagnostics)))


def split_summary_branch_delta(
    evidence: ChimeraSummaryAttentionLayerEvidence,
    *,
    row_mask: Any | None = None,
    tolerance: float = 1e-6,
    fail_closed: bool = True,
) -> SummarySplit:
    """Split the sealed branch write into its two additive sub-branches.

    The sub-branches are captured separately, so their sum must reconstruct the
    sealed write; a breach means the capture describes a different operator and
    the fold fails closed (design fail-closed condition 5).
    """

    torch = _torch()
    if not isinstance(evidence, ChimeraSummaryAttentionLayerEvidence):
        raise HFCausalSparseLatentFoldError(
            "evidence must be a ChimeraSummaryAttentionLayerEvidence."
        )
    tolerance = _nonnegative_float(tolerance, "tolerance")
    mask = (
        torch.ones(evidence.batch, evidence.sequence, dtype=torch.bool)
        if row_mask is None
        else _mask(row_mask, "row_mask", (evidence.batch, evidence.sequence))
    )
    total = evidence.local_delta + evidence.summary_delta
    selected_total = _masked_rows(evidence.attention_delta, mask)
    residual = _relative_rmse(_masked_rows(total, mask), selected_total)
    diagnostics = {
        "schema": SCHEMA,
        "component": "sparse_latent_sub_branch_split",
        "layer_index": int(evidence.index),
        "family": evidence.geometry.family,
        "summary_stride": evidence.geometry.summary_stride,
        "local_window": evidence.geometry.local_window,
        "evidence_rows": int(mask.sum().item()),
        "parent_forward_calls": 0,
        "split_reconstruction_relative_rmse": residual,
        "tolerance": tolerance,
        "within_tolerance": bool(residual <= tolerance),
        "local_energy_fraction": _energy_fraction(
            _masked_rows(evidence.local_delta, mask), selected_total
        ),
        "summary_energy_fraction": _energy_fraction(
            _masked_rows(evidence.summary_delta, mask), selected_total
        ),
        "local_delta_sha256": _tensor_sha256(evidence.local_delta),
        "summary_delta_sha256": _tensor_sha256(evidence.summary_delta),
    }
    if fail_closed and residual > tolerance:
        raise HFCausalSparseLatentFoldError(
            "The local and summary sub-branch deltas do not reconstruct the "
            f"sealed branch write ({residual:.3e} > {tolerance:.3e})."
        )
    return SummarySplit(
        local_delta=evidence.local_delta,
        summary_delta=evidence.summary_delta,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# The compiled child
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HFCausalSparseLatentFold:
    """One compiled chimera CSA/HCA child branch plus its sealed receipt."""

    local_fold: HFCausalCoupledAttentionFold
    summary_query_weight: Any
    summary_key_weight: Any
    summary_value_weight: Any
    summary_output_weight: Any
    geometry: ChimeraSparseLatentGeometry
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.local_fold, HFCausalCoupledAttentionFold):
            raise HFCausalSparseLatentFoldError(
                "local_fold must be an HFCausalCoupledAttentionFold."
            )
        if not isinstance(self.geometry, ChimeraSparseLatentGeometry):
            raise HFCausalSparseLatentFoldError(
                "geometry must be a ChimeraSparseLatentGeometry."
            )
        width = self.geometry.hidden_size
        for name in (
            "summary_query_weight",
            "summary_key_weight",
            "summary_value_weight",
        ):
            value = _tensor(getattr(self, name), name, ndim=2).clone()
            if tuple(int(size) for size in value.shape) != (width, width):
                raise HFCausalSparseLatentFoldError(
                    f"{name} must be [{width}, {width}]."
                )
            object.__setattr__(self, name, value)
        writer = _tensor(self.summary_output_weight, "summary_output_weight", ndim=2)
        if int(writer.shape[1]) != width:
            raise HFCausalSparseLatentFoldError(
                "summary_output_weight has the wrong input width."
            )
        object.__setattr__(self, "summary_output_weight", writer.clone())
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def summary_replay(self, child_rows: Any) -> tuple[Any, Any]:
        """Replay the summary sub-branch; return its delta and probabilities."""

        torch = _torch()
        rows = _tensor(child_rows, "child_rows", ndim=3)
        sequence = int(rows.shape[1])
        pooling = summary_pooling_matrix(sequence, self.geometry.summary_stride)
        summaries = torch.matmul(pooling.reshape(1, *pooling.shape), rows)
        _scores, probabilities = summary_attention_probabilities(
            rows @ self.summary_query_weight.T,
            summaries @ self.summary_key_weight.T,
            block_mask=summary_block_mask(sequence, self.geometry.summary_stride),
            scaling=self.geometry.summary_scaling,
        )
        context = torch.matmul(probabilities, summaries @ self.summary_value_weight.T)
        return (context @ self.summary_output_weight.T).contiguous(), probabilities

    def replay(
        self,
        child_rows: Any,
        *,
        attention_mask: Any,
        position_ids: Any,
    ) -> tuple[Any, Any, Any]:
        """Replay both sub-branches and their sum, all in FP64."""

        local_delta, _scores, _probabilities = self.local_fold.replay(
            child_rows, attention_mask=attention_mask, position_ids=position_ids
        )
        summary_delta, _summary_probabilities = self.summary_replay(child_rows)
        return (local_delta + summary_delta).contiguous(), local_delta, summary_delta

    def child_state_dict(self) -> dict[str, Any]:
        """Emit the compiled child in chimera ``SparseLatentAttention`` names."""

        return {
            "local.q_proj.weight": self.local_fold.query_weight,
            "local.k_proj.weight": self.local_fold.key_weight,
            "local.v_proj.weight": self.local_fold.value_weight,
            "local.out.weight": self.local_fold.output_weight,
            "summary_q.weight": self.summary_query_weight,
            "summary_k.weight": self.summary_key_weight,
            "summary_v.weight": self.summary_value_weight,
            "out.weight": self.summary_output_weight,
        }


def _replay_summary_bf16(fold: HFCausalSparseLatentFold, child_rows: Any) -> Any:
    torch = _torch()
    rows = child_rows.to(dtype=torch.bfloat16)
    sequence = int(rows.shape[1])
    pooling = summary_pooling_matrix(sequence, fold.geometry.summary_stride).to(
        dtype=torch.bfloat16
    )
    summaries = torch.matmul(pooling.reshape(1, *pooling.shape), rows)
    query = rows @ fold.summary_query_weight.to(dtype=torch.bfloat16).transpose(0, 1)
    key = summaries @ fold.summary_key_weight.to(dtype=torch.bfloat16).transpose(0, 1)
    value = summaries @ fold.summary_value_weight.to(dtype=torch.bfloat16).transpose(
        0, 1
    )
    scores = torch.matmul(query, key.transpose(-1, -2)) * torch.tensor(
        fold.geometry.summary_scaling, dtype=torch.bfloat16
    )
    allowed = summary_block_mask(sequence, fold.geometry.summary_stride)
    masked = scores.masked_fill(
        ~allowed.reshape(1, *allowed.shape), torch.finfo(torch.bfloat16).min
    )
    probabilities = torch.softmax(masked, dim=-1)
    context = torch.matmul(probabilities, value)
    delta = context @ fold.summary_output_weight.to(dtype=torch.bfloat16).transpose(
        0, 1
    )
    return delta.to(dtype=torch.float64)


def _replay_local_bf16(
    fold: HFCausalCoupledAttentionFold,
    child_rows: Any,
    *,
    attention_mask: Any,
) -> Any:
    """Replay the compiled windowed NoPE local sub-branch entirely in BF16."""

    torch = _torch()
    topology = fold.target_topology
    rows = child_rows.to(dtype=torch.bfloat16)
    batch, sequence = int(rows.shape[0]), int(rows.shape[1])

    def project(weight: Any, heads: int) -> Any:
        return (
            (rows @ weight.to(dtype=torch.bfloat16).transpose(0, 1))
            .reshape(batch, sequence, heads, topology.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    query = project(fold.query_weight, topology.query_heads)
    key = project(fold.key_weight, topology.key_value_heads)
    value = project(fold.value_weight, topology.key_value_heads)
    if topology.query_key_norm == "rms_norm":
        raise HFCausalSparseLatentFoldError(
            "The chimera local sub-branch has no per-head Q/K norm."
        )
    repeats = topology.query_heads // topology.key_value_heads
    keys = key.repeat_interleave(repeats, dim=1)
    values = value.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query, keys.transpose(-1, -2)) * torch.tensor(
        topology.attention_scaling, dtype=torch.bfloat16
    )
    causal = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    if topology.attention_window is not None:
        offsets = torch.arange(sequence).reshape(-1, 1) - torch.arange(
            sequence
        ).reshape(1, -1)
        causal = causal & (offsets < topology.attention_window)
    allowed = attention_mask.reshape(batch, 1, 1, sequence) & causal.reshape(
        1, 1, sequence, sequence
    )
    masked = scores.masked_fill(~allowed, torch.finfo(torch.bfloat16).min)
    probabilities = torch.softmax(masked, dim=-1)
    contexts = torch.matmul(probabilities, values)
    concatenated = contexts.permute(0, 2, 1, 3).reshape(
        batch, sequence, topology.query_width
    )
    delta = concatenated @ fold.output_weight.to(dtype=torch.bfloat16).transpose(0, 1)
    return delta.to(dtype=torch.float64)


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------


def _fit_summary_branch(
    *,
    child_rows: Any,
    pooled_rows: Any,
    query_target: Any,
    key_target: Any,
    value_target: Any,
    endpoint_target: Any,
    block_mask: Any,
    scaling: float,
    fit: Any,
    ridge: float,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Score ridge, installed-probability replay, then endpoint-coupled V/out."""

    torch = _torch()
    child_fit = _masked_rows(child_rows, fit)
    query_weight = _ridge(
        child_fit, _masked_rows(query_target, fit), ridge
    ).T.contiguous()
    # Summary rows are not token rows: every summary participates in the key and
    # value solves, and the zero row is dropped because a linear map sends it to
    # zero exactly -- fitting it would only add a redundant equation.
    pooled_flat = pooled_rows[:, 1:].reshape(-1, int(pooled_rows.shape[2]))
    key_weight = _ridge(
        pooled_flat, key_target[:, 1:].reshape(-1, int(key_target.shape[2])), ridge
    ).T.contiguous()
    value_weight = _ridge(
        pooled_flat, value_target[:, 1:].reshape(-1, int(value_target.shape[2])), ridge
    ).T.contiguous()
    _scores, probabilities = summary_attention_probabilities(
        child_rows @ query_weight.T,
        pooled_rows @ key_weight.T,
        block_mask=block_mask,
        scaling=scaling,
    )
    context = torch.matmul(probabilities, pooled_rows @ value_weight.T)
    output_weight = _ridge(
        _masked_rows(context, fit), _masked_rows(endpoint_target, fit), ridge
    ).T.contiguous()
    return (
        query_weight,
        key_weight,
        value_weight,
        output_weight,
        probabilities,
        context,
    )


def compile_hf_causal_sparse_latent_fold(
    evidence: ChimeraSummaryAttentionLayerEvidence,
    geometry: ChimeraSparseLatentGeometry,
    *,
    attention_mask: Any,
    fit_token_mask: Any,
    validation_token_mask: Any,
    valid_token_mask: Any | None = None,
    child_normalized_rows: Any | None = None,
    endpoint_target: Any | None = None,
    score_chart: Any | None = None,
    policy: ChimeraSparseLatentFoldPolicy | None = None,
) -> HFCausalSparseLatentFold:
    """Compile one chimera CSA/HCA branch into a smaller one, closed form.

    Ordered recipe of design section 1.2: exact sub-branch split, local fit
    through the extended GQA kernel (window + NoPE), summary fit as score ridge
    then installed-probability replay then endpoint-coupled V/out, bounded
    alternation, BF16 replay, per-sub-branch diagnostics.
    """

    torch = _torch()
    if not isinstance(evidence, ChimeraSummaryAttentionLayerEvidence):
        raise HFCausalSparseLatentFoldError(
            "evidence must be a ChimeraSummaryAttentionLayerEvidence."
        )
    if not isinstance(geometry, ChimeraSparseLatentGeometry):
        raise HFCausalSparseLatentFoldError(
            "geometry must be a ChimeraSparseLatentGeometry."
        )
    source = evidence.geometry
    policy = policy or ChimeraSparseLatentFoldPolicy()
    if geometry.summary_stride != source.summary_stride:
        raise HFCausalSparseLatentFoldError(
            "The summary stride is a contract constant, not a fold coordinate; "
            "changing it is a kind translation "
            f"(see {DESIGN_DOCUMENT} section 3)."
        )
    if geometry.local_window != source.local_window:
        raise HFCausalSparseLatentFoldError(
            "Changing the local window is a band restriction, a kind "
            f"translation (see {DESIGN_DOCUMENT} section 3)."
        )
    if geometry.family != source.family:
        raise HFCausalSparseLatentFoldError(
            "CSA and HCA are the same module at different strides; a family "
            "change is a stride change."
        )
    batch, sequence = evidence.batch, evidence.sequence
    child = (
        evidence.attention_norm_output
        if child_normalized_rows is None
        else _tensor(child_normalized_rows, "child_normalized_rows", ndim=3)
    )
    if tuple(int(size) for size in child.shape[:2]) != (batch, sequence):
        raise HFCausalSparseLatentFoldError("child rows have the wrong sequence shape.")
    if int(child.shape[2]) != geometry.hidden_size:
        raise HFCausalSparseLatentFoldError(
            "child rows do not match the child hidden size."
        )
    delta_target = (
        evidence.attention_delta
        if endpoint_target is None
        else _tensor(endpoint_target, "endpoint_target", ndim=3)
    )
    mask = _mask(attention_mask, "attention_mask", (batch, sequence))
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
        raise HFCausalSparseLatentFoldError(
            "CSA/HCA fit/validation masks must be disjoint nonempty valid subsets."
        )

    # Step 1 -- the exact split.
    split = split_summary_branch_delta(
        evidence, row_mask=valid, tolerance=policy.split_reconstruction_tolerance
    )

    # Cross-width score targets need an orthonormal chart; pooling commutes with
    # it exactly, so raw and pooled rows share the one chart.
    chart_used = score_chart is not None
    if chart_used:
        chart = _tensor(score_chart, "score_chart", ndim=2)
        if tuple(int(size) for size in chart.shape) != (
            source.hidden_size,
            geometry.hidden_size,
        ):
            raise HFCausalSparseLatentFoldError(
                "score_chart must be [source_hidden_size, target_hidden_size]."
            )
        orthonormality = float(
            (
                chart.T @ chart
                - torch.eye(geometry.hidden_size, dtype=torch.float64)
            )
            .abs()
            .max()
            .item()
        )
        if orthonormality > 1e-8:
            raise HFCausalSparseLatentFoldError(
                "score_chart must have orthonormal columns; the summary score "
                "is an inner product in the residual width."
            )
    elif geometry.hidden_size != source.hidden_size:
        raise HFCausalSparseLatentFoldError(
            "A width change needs an explicit orthonormal score_chart; the "
            "summary score is an inner product in the residual width "
            f"(see {DESIGN_DOCUMENT} section 2)."
        )
    else:
        chart = torch.eye(geometry.hidden_size, dtype=torch.float64)
        orthonormality = 0.0

    def transport(rows: Any) -> Any:
        return rows @ chart

    pooling = summary_pooling_matrix(sequence, geometry.summary_stride)
    pooled_child = torch.matmul(pooling.reshape(1, *pooling.shape), child)
    block_mask = summary_block_mask(sequence, geometry.summary_stride)
    positions = torch.arange(sequence).reshape(1, -1).expand(batch, sequence)
    source_topology = source.local_topology()
    target_topology = geometry.local_topology()

    def compile_local(target: Any) -> HFCausalCoupledAttentionFold:
        return compile_hf_causal_gqa_attention_fold(
            child,
            source_query_sequences=evidence.local_query_rows,
            source_key_sequences=evidence.local_key_rows,
            source_value_sequences=evidence.local_value_rows,
            target_residual_delta_sequences=target,
            attention_mask=mask,
            position_ids=positions,
            valid_token_mask=valid,
            fit_token_mask=fit,
            validation_token_mask=validation,
            source_topology=source_topology,
            target_topology=target_topology,
            ridge=policy.ridge,
        )

    def local_replay(fold: HFCausalCoupledAttentionFold) -> Any:
        delta, _scores, _probabilities = fold.replay(
            child, attention_mask=mask, position_ids=positions
        )
        return delta

    def objective(local: Any, summary: Any) -> float:
        residual = _masked_rows(delta_target - local - summary, fit)
        return float(residual.square().sum().item())

    # Steps 2 and 3 -- the two sub-branches, fitted against their own targets.
    local_fold = compile_local(split.local_delta)
    local_delta = local_replay(local_fold)
    (
        summary_query_weight,
        summary_key_weight,
        summary_value_weight,
        summary_output_weight,
        summary_probabilities,
        summary_context,
    ) = _fit_summary_branch(
        child_rows=child,
        pooled_rows=pooled_child,
        query_target=transport(evidence.summary_query_rows),
        key_target=transport(evidence.summary_key_rows),
        value_target=transport(evidence.summary_value_rows),
        endpoint_target=split.summary_delta,
        block_mask=block_mask,
        scaling=geometry.summary_scaling,
        fit=fit,
        ridge=policy.ridge,
    )
    summary_delta = summary_context @ summary_output_weight.T
    rounds = [
        {
            "round": 0,
            "step": "independent_sub_branch_targets",
            "fit_objective": objective(local_delta, summary_delta),
            "accepted": True,
        }
    ]

    # Step 4 -- bounded alternation; every sub-step is an exact ridge given the
    # other branch, and is reverted if the unregularized objective rose.
    for index in range(1, policy.alternation_rounds):
        current = objective(local_delta, summary_delta)
        candidate_writer = _ridge(
            _masked_rows(summary_context, fit),
            _masked_rows(delta_target - local_delta, fit),
            policy.ridge,
        ).T.contiguous()
        candidate_summary = summary_context @ candidate_writer.T
        candidate_objective = objective(local_delta, candidate_summary)
        accepted = candidate_objective <= current
        if accepted:
            summary_output_weight = candidate_writer
            summary_delta = candidate_summary
        rounds.append(
            {
                "round": index,
                "step": "summary_writer_against_delta_minus_local",
                "fit_objective": candidate_objective if accepted else current,
                "candidate_fit_objective": candidate_objective,
                "accepted": bool(accepted),
            }
        )
        current = objective(local_delta, summary_delta)
        candidate_local = compile_local(delta_target - summary_delta)
        candidate_local_delta = local_replay(candidate_local)
        candidate_objective = objective(candidate_local_delta, summary_delta)
        accepted = candidate_objective <= current
        if accepted:
            local_fold = candidate_local
            local_delta = candidate_local_delta
        rounds.append(
            {
                "round": index,
                "step": "local_branch_against_delta_minus_summary",
                "fit_objective": candidate_objective if accepted else current,
                "candidate_fit_objective": candidate_objective,
                "accepted": bool(accepted),
            }
        )

    fold = HFCausalSparseLatentFold(
        local_fold=local_fold,
        summary_query_weight=summary_query_weight,
        summary_key_weight=summary_key_weight,
        summary_value_weight=summary_value_weight,
        summary_output_weight=summary_output_weight,
        geometry=geometry,
        report={"schema": SCHEMA, "component": "staged_before_receipt"},
    )
    _summary_delta, replayed_probabilities = fold.summary_replay(child)
    predicted = local_delta + summary_delta
    overlap = float(
        torch.minimum(replayed_probabilities, evidence.summary_probabilities)
        .sum(dim=-1)[valid]
        .mean()
        .item()
    )
    if overlap < policy.minimum_summary_probability_overlap:
        raise HFCausalSparseLatentFoldError(
            f"The replayed summary probabilities keep only {overlap:.4f} of the "
            "parent's block assignment, below the sealed policy fraction "
            f"({policy.minimum_summary_probability_overlap:.4f})."
        )

    # Step 5 -- BF16 replay of both sub-branches, then acceptance.
    if policy.bf16_replay:
        predicted_bf16 = _replay_local_bf16(
            local_fold, child, attention_mask=mask
        ) + _replay_summary_bf16(fold, child)
    else:
        predicted_bf16 = predicted
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
        raise HFCausalSparseLatentFoldError(
            "The BF16 validation endpoint is at or above the no-branch baseline "
            f"({validation_error_bf16:.6f} >= {zero_baseline:.6f})."
        )
    body = {
        "schema": SCHEMA,
        "component": "chimera_sparse_latent_branch_fold",
        "method": (
            "exact_sub_branch_split_windowed_nope_gqa_local_plus_score_ridge_"
            "probability_replay_endpoint_summary_with_bounded_alternation"
        ),
        "layer_index": int(evidence.index),
        "family": geometry.family,
        "summary_stride": geometry.summary_stride,
        "local_window": geometry.local_window,
        "source_geometry": source.as_dict(),
        "target_geometry": geometry.as_dict(),
        "policy": policy.as_dict(),
        "split_report_sha256": split.diagnostics["report_sha256"],
        "split_reconstruction_relative_rmse": split.diagnostics[
            "split_reconstruction_relative_rmse"
        ],
        "local_energy_fraction": split.diagnostics["local_energy_fraction"],
        "summary_energy_fraction": split.diagnostics["summary_energy_fraction"],
        "local_fold_report_sha256": local_fold.report["report_sha256"],
        "reused_attention_kernel": "compile_hf_causal_gqa_attention_fold",
        "local_position_encoding": "nope",
        "score_chart_used": bool(chart_used),
        "score_chart_orthonormality_max_abs_error": orthonormality,
        "pooling_commutes_with_chart": True,
        "zero_summary_preserved_exactly": True,
        "alternation_rounds": policy.alternation_rounds,
        "alternation_trace": rounds,
        "alternation_objective_monotone": bool(
            all(
                rounds[index]["fit_objective"] <= rounds[index - 1]["fit_objective"]
                + 1e-12
                for index in range(1, len(rounds))
            )
        ),
        "summary_probability_overlap": overlap,
        "summary_probability_relative_rmse": _relative_rmse(
            replayed_probabilities[valid], evidence.summary_probabilities[valid]
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
        "validation_beats_no_branch_baseline": bool(validation_error < zero_baseline),
        "validation_residual_relative_rmse_bf16": validation_error_bf16,
        "bf16_versus_fp64_endpoint_relative_delta": _relative_rmse(
            _masked_rows(predicted_bf16, validation),
            _masked_rows(predicted, validation),
        ),
        "local_sub_branch_validation_relative_rmse": _relative_rmse(
            _masked_rows(local_delta, validation),
            _masked_rows(evidence.local_delta, validation),
        ),
        "summary_sub_branch_validation_relative_rmse": _relative_rmse(
            _masked_rows(summary_delta, validation),
            _masked_rows(evidence.summary_delta, validation),
        ),
        "local_sub_branch_retained_energy_fraction": _retained_energy(
            _masked_rows(local_delta, validation),
            _masked_rows(evidence.local_delta, validation),
        ),
        "summary_sub_branch_retained_energy_fraction": _retained_energy(
            _masked_rows(summary_delta, validation),
            _masked_rows(evidence.summary_delta, validation),
        ),
        "ridge": policy.ridge,
        "numeric_weight_matching_objective": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
        "summary_query_weight_sha256": _tensor_sha256(summary_query_weight),
        "summary_key_weight_sha256": _tensor_sha256(summary_key_weight),
        "summary_value_weight_sha256": _tensor_sha256(summary_value_weight),
        "summary_output_weight_sha256": _tensor_sha256(summary_output_weight),
    }
    return HFCausalSparseLatentFold(
        local_fold=local_fold,
        summary_query_weight=summary_query_weight,
        summary_key_weight=summary_key_weight,
        summary_value_weight=summary_value_weight,
        summary_output_weight=summary_output_weight,
        geometry=geometry,
        report=_report(body),
    )


# ---------------------------------------------------------------------------
# Stage-2 comparability hook: the exact GQA -> CSA/HCA three-way split
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SummaryThreeWaySplit:
    """``Delta = window + far block-mean + far within-block deviation``."""

    window_delta: Any
    block_mean_delta: Any
    deviation_delta: Any
    coverage_gap_delta: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", _report(dict(self.report)))


def gqa_to_summary_three_way_split(
    *,
    attention_probabilities: Any,
    attention_value_sequences: Any,
    attention_output_weight: Any,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    local_window: int,
    summary_stride: int,
    row_mask: Any,
) -> SummaryThreeWaySplit:
    """Split a softmax GQA delta into what CSA/HCA can and cannot represent.

    ``Delta_far_blockmean`` is block-constant probability mass times block-mean
    values -- exactly the operator class of the summary sub-branch.
    ``Delta_far_deviation`` is the within-block covariance of probabilities and
    values and is provably outside that class, so it routes to a carrier.  For
    an HCA stride the coverage-gap mass (outside the window, inside an
    incomplete block) is reported separately because it is unrepresentable by
    construction (design section 3, "softmax-GQA -> CSA/HCA").  Everything is
    computed from cached probabilities and values with zero parent forwards.
    """

    torch = _torch()
    probabilities = _tensor(attention_probabilities, "attention_probabilities", ndim=4)
    values = _tensor(attention_value_sequences, "attention_value_sequences", ndim=3)
    writer = _tensor(attention_output_weight, "attention_output_weight", ndim=2)
    query_heads = _positive_int(query_heads, "query_heads")
    key_value_heads = _positive_int(key_value_heads, "key_value_heads")
    head_dim = _positive_int(head_dim, "head_dim")
    local_window = _positive_int(local_window, "local_window")
    summary_stride = _positive_int(summary_stride, "summary_stride")
    if query_heads % key_value_heads:
        raise HFCausalSparseLatentFoldError(
            "query_heads must be divisible by key_value_heads."
        )
    batch, sequence = int(values.shape[0]), int(values.shape[1])
    if tuple(int(size) for size in probabilities.shape) != (
        batch,
        query_heads,
        sequence,
        sequence,
    ):
        raise HFCausalSparseLatentFoldError(
            "attention_probabilities must have shape [batch, query_heads, S, S]."
        )
    if int(values.shape[2]) != key_value_heads * head_dim:
        raise HFCausalSparseLatentFoldError(
            "attention_value_sequences width differs from key_value_heads*head_dim."
        )
    if int(writer.shape[1]) != query_heads * head_dim:
        raise HFCausalSparseLatentFoldError(
            "attention_output_weight input width differs from query_heads*head_dim."
        )
    mask = _mask(row_mask, "row_mask", (batch, sequence))
    heads = (
        values.reshape(batch, sequence, key_value_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
        .repeat_interleave(query_heads // key_value_heads, dim=1)
    )

    def contract(weights: Any) -> Any:
        contexts = torch.matmul(weights, heads)
        concatenated = contexts.permute(0, 2, 1, 3).reshape(
            batch, sequence, query_heads * head_dim
        )
        return concatenated @ writer.T

    offsets = torch.arange(sequence).reshape(-1, 1) - torch.arange(sequence).reshape(
        1, -1
    )
    window_mask = ((offsets >= 0) & (offsets < local_window)).reshape(
        1, 1, sequence, sequence
    )
    block_of = torch.arange(sequence) // summary_stride
    # A key is *summarized* for query t only if its block completed before t's
    # block; keys in t's own (incomplete) block are invisible to the summary
    # path, which is the coverage gap for large strides.
    visible = (block_of.reshape(1, -1) < block_of.reshape(-1, 1)).reshape(
        1, 1, sequence, sequence
    )
    window_probabilities = probabilities * window_mask
    far_probabilities = probabilities - window_probabilities
    summarizable = far_probabilities * visible
    gap_probabilities = far_probabilities - summarizable
    block_indicator = (
        block_of.reshape(-1, 1) == block_of.reshape(1, -1)
    ).to(dtype=torch.float64)
    block_sizes = block_indicator.sum(dim=1)
    # Block-constant reweighting: every key of a completed block receives the
    # block's summed far probability divided by the block size, which is exactly
    # ``Pbar_h(t, b) * Vbar_h_b`` -- what a block-mean key/value pair can serve.
    block_mean_probabilities = torch.matmul(
        summarizable, block_indicator
    ) / block_sizes.reshape(1, 1, 1, -1)
    deviation_probabilities = summarizable - block_mean_probabilities
    window_delta = contract(window_probabilities)
    block_mean_delta = contract(block_mean_probabilities)
    deviation_delta = contract(deviation_probabilities)
    gap_delta = contract(gap_probabilities)
    total = window_delta + block_mean_delta + deviation_delta + gap_delta
    selected_total = _masked_rows(total, mask)
    body = {
        "schema": GQA_TO_SUMMARY_SCHEMA,
        "component": "gqa_to_summary_attention_three_way_split",
        "local_window": local_window,
        "summary_stride": summary_stride,
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_dim": head_dim,
        "evidence_rows": int(mask.sum().item()),
        "parent_forward_calls": 0,
        "rho_window": _energy_fraction(_masked_rows(window_delta, mask), selected_total),
        "rho_far_block_mean": _energy_fraction(
            _masked_rows(block_mean_delta, mask), selected_total
        ),
        "rho_far_deviation": _energy_fraction(
            _masked_rows(deviation_delta, mask), selected_total
        ),
        "rho_coverage_gap": _energy_fraction(
            _masked_rows(gap_delta, mask), selected_total
        ),
        "representable_energy_fraction": _energy_fraction(
            _masked_rows(window_delta + block_mean_delta, mask), selected_total
        ),
        "split_is_exact": True,
        "window_delta_sha256": _tensor_sha256(window_delta),
        "block_mean_delta_sha256": _tensor_sha256(block_mean_delta),
        "deviation_delta_sha256": _tensor_sha256(deviation_delta),
        "coverage_gap_delta_sha256": _tensor_sha256(gap_delta),
    }
    return SummaryThreeWaySplit(
        window_delta=window_delta,
        block_mean_delta=block_mean_delta,
        deviation_delta=deviation_delta,
        coverage_gap_delta=gap_delta,
        report=body,
    )


def compile_gqa_to_summary_attention_translation(*_args: Any, **_kwargs: Any) -> Any:
    """Fail closed: the full GQA -> CSA/HCA translation is a stage-4 lane.

    The exact three-way split this translation is built on is implemented and
    tested here (``gqa_to_summary_three_way_split``).  What is missing is the
    carrier routing of ``Delta_far_deviation`` plus the coverage-gap mass and
    the checkpoint gate, which need a downstream global carrier to exist.
    """

    raise HFCausalSparseLatentFoldError(
        "compile_gqa_to_summary_attention_translation is designed but not "
        f"implemented in this wave. See {DESIGN_DOCUMENT} section 3 (verdict: "
        "honest, promising) and the stage ladder in section 5; this entry point "
        "fails closed rather than returning an unmeasured translation."
    )


__all__ = [
    "DESIGN_DOCUMENT",
    "EVIDENCE_SCHEMA",
    "GQA_TO_SUMMARY_SCHEMA",
    "SCHEMA",
    "WINDOWED_ATTENTION_SCHEMA",
    "ChimeraSparseLatentFoldPolicy",
    "ChimeraSparseLatentGeometry",
    "ChimeraSummaryAttentionLayerEvidence",
    "HFCausalSparseLatentFold",
    "HFCausalSparseLatentFoldError",
    "SummarySplit",
    "SummaryThreeWaySplit",
    "capture_chimera_sparse_latent_layer_evidence",
    "chimera_sparse_latent_geometry",
    "compile_gqa_to_summary_attention_translation",
    "compile_hf_causal_sparse_latent_fold",
    "gqa_to_summary_three_way_split",
    "split_summary_branch_delta",
    "summary_attention_probabilities",
    "summary_block_mask",
    "summary_pooling_matrix",
]
