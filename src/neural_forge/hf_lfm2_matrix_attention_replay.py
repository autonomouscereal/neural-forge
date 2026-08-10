"""Structured, cached-only LFM2 attention replay for matrix cells C06-C10.

The evaluator in this module operates on already authenticated tensors.  It
does not load a model, recapture a trace, flatten heads into regression rows,
or make an acceptance decision for the caller.  Its purpose is narrower: run
the real same-topology LFM2 attention equations on a candidate projection
bundle while preserving batch, head, query, and cache axes.

Every evaluation first replays a caller-supplied reference projection bundle.
That copy-exact calibration must reproduce the sealed phase tensors before a
candidate intervention is measured.  Consequently a missing prefill prefix,
a discontinuous KV cache, or an inaccurate reference projection fails closed
instead of being mistaken for candidate error.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import (
    _reconstruct_attention_causal_path,
    _rotate_half,
)
from neural_forge.hf_lfm2_cpcf_layer_compiler import _runtime_rms_norm
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2AttentionScorePreservationReport,
    Lfm2AttentionScorePreservationThresholds,
    evaluate_lfm2_structural_attention_score_preservation,
    lfm2_authenticated_tensor_sha256,
)


LFM2_MATRIX_ATTENTION_REPLAY_SCHEMA = (
    "neural-forge-lfm2-matrix-structured-attention-replay-v1"
)
Lfm2MatrixAttentionCell: TypeAlias = Literal["C06", "C07", "C08", "C09", "C10"]
Lfm2MatrixAttentionPhaseKind: TypeAlias = Literal["prefill", "decode"]
Lfm2MatrixProjection: TypeAlias = Callable[[Tensor], Tensor]

_SUPPORTED_CELLS = frozenset({"C06", "C07", "C08", "C09", "C10"})
_QUERY_CELLS = frozenset({"C06", "C10"})
_KEY_CELLS = frozenset({"C07", "C10"})
_VALUE_CELLS = frozenset({"C08", "C10"})
_CANDIDATE_O_CELLS = frozenset({"C09", "C10"})

# Copy calibration is an infrastructure proof, so callers may tighten these
# limits but may never relax them.  The defaults are intentionally the same as
# the hard ceilings: FP32/BF16 runtime-form replay needs modest tolerance,
# while an arbitrary caller value must not turn a wrong reference into a
# purported copy-exact result.
MAX_COPY_EXACT_ABSOLUTE_TOLERANCE = 2e-3
MAX_COPY_EXACT_RELATIVE_TOLERANCE = 2e-2


class Lfm2MatrixAttentionReplayError(ValueError):
    """A structured phase, candidate, or replay invariant is invalid."""


class Lfm2MatrixAttentionTopologyError(Lfm2MatrixAttentionReplayError):
    """The replay does not have legal same-topology LFM2 geometry."""


class Lfm2MatrixAttentionContinuityError(Lfm2MatrixAttentionReplayError):
    """The sealed prefill/decode KV trajectory is incomplete or discontinuous."""


class Lfm2MatrixAttentionProjectionError(Lfm2MatrixAttentionReplayError):
    """A projection callable violated its declared tensor contract."""


class Lfm2MatrixAttentionCopyExactError(Lfm2MatrixAttentionReplayError):
    """The reference projection bundle did not reproduce the sealed trace."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2MatrixAttentionReplayError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2MatrixAttentionReplayError(f"{name} must be a non-negative integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2MatrixAttentionReplayError(f"{name} must be positive and finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2MatrixAttentionReplayError(
            f"{name} must be positive and finite."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2MatrixAttentionReplayError(f"{name} must be positive and finite.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2MatrixAttentionReplayError(f"{name} must be non-negative and finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2MatrixAttentionReplayError(
            f"{name} must be non-negative and finite."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2MatrixAttentionReplayError(f"{name} must be non-negative and finite.")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2MatrixAttentionReplayError(f"{name} must be a non-empty string.")
    return value.strip()


def _finite_tensor(value: Any, name: str, *, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise Lfm2MatrixAttentionReplayError(
            f"{name} must be a non-empty dense real rank-{ndim} tensor."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise Lfm2MatrixAttentionReplayError(f"{name} contains a non-finite value.")
    return value


@dataclass(frozen=True, slots=True)
class Lfm2MatrixAttentionTopology:
    """The unchanged LFM2 attention geometry used by one matrix cell."""

    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    norm_epsilon: float

    def __post_init__(self) -> None:
        hidden = _positive_int(self.hidden_size, "hidden_size")
        query_heads = _positive_int(self.query_heads, "query_heads")
        key_value_heads = _positive_int(self.key_value_heads, "key_value_heads")
        head_dim = _positive_int(self.head_dim, "head_dim")
        epsilon = _positive_float(self.norm_epsilon, "norm_epsilon")
        if query_heads % key_value_heads:
            raise Lfm2MatrixAttentionTopologyError(
                "query_heads must be divisible by key_value_heads for GQA."
            )
        if hidden != query_heads * head_dim:
            raise Lfm2MatrixAttentionTopologyError(
                "Same-topology LFM2 replay requires hidden_size == "
                "query_heads * head_dim."
            )
        object.__setattr__(self, "norm_epsilon", epsilon)

    @property
    def groups(self) -> int:
        return self.query_heads // self.key_value_heads

    @property
    def key_value_width(self) -> int:
        return self.key_value_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class Lfm2MatrixAttentionProjectionBundle:
    """Pure projection callables and non-authoritative storage declarations.

    Each callable accepts ``[batch, queries, hidden]`` and returns a rank-three
    projection.  Q and O return ``hidden_size`` columns; K and V return
    ``key_value_heads * head_dim`` columns.  Counts are declarations for the
    complete executable bundle.  They are deliberately not inferred from a
    materialized dense view because a folded callable may execute factors.
    This evaluator seals but never verifies them; only an authenticated
    installed-state/alias manifest may confer physical compaction authority.
    """

    bundle_id: str
    topology: Lfm2MatrixAttentionTopology
    query_projection: Lfm2MatrixProjection = field(repr=False, compare=False)
    key_projection: Lfm2MatrixProjection = field(repr=False, compare=False)
    value_projection: Lfm2MatrixProjection = field(repr=False, compare=False)
    output_projection: Lfm2MatrixProjection = field(repr=False, compare=False)
    query_norm_weight: Tensor = field(repr=False, compare=False)
    key_norm_weight: Tensor = field(repr=False, compare=False)
    declared_logical_stored_parameter_count: int
    declared_physical_materialized_parameter_count: int
    declared_accounting_scope: str = "complete_projection_bundle"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _identifier(self.bundle_id, "bundle_id"))
        if not isinstance(self.topology, Lfm2MatrixAttentionTopology):
            raise TypeError("topology must be Lfm2MatrixAttentionTopology.")
        for name in (
            "query_projection",
            "key_projection",
            "value_projection",
            "output_projection",
        ):
            if not callable(getattr(self, name)):
                raise Lfm2MatrixAttentionProjectionError(f"{name} must be callable.")
        query_norm = _finite_tensor(
            self.query_norm_weight,
            "query_norm_weight",
            ndim=1,
        )
        key_norm = _finite_tensor(
            self.key_norm_weight,
            "key_norm_weight",
            ndim=1,
        )
        expected = (self.topology.head_dim,)
        if tuple(query_norm.shape) != expected or tuple(key_norm.shape) != expected:
            raise Lfm2MatrixAttentionTopologyError(
                "Q/K RMSNorm weights must each have shape [head_dim]."
            )
        _nonnegative_int(
            self.declared_logical_stored_parameter_count,
            "declared_logical_stored_parameter_count",
        )
        _nonnegative_int(
            self.declared_physical_materialized_parameter_count,
            "declared_physical_materialized_parameter_count",
        )
        object.__setattr__(
            self,
            "declared_accounting_scope",
            _identifier(self.declared_accounting_scope, "declared_accounting_scope"),
        )


@dataclass(frozen=True, slots=True)
class Lfm2MatrixAttentionPhase:
    """One sealed, structured parent attention phase.

    Tensor names intentionally mirror the trace-cache endpoints.  Q/K/V and
    contexts retain their head axes; no tensor is converted into flattened
    regression rows.
    """

    topology: Lfm2MatrixAttentionTopology
    phase_id: str
    phase_kind: Lfm2MatrixAttentionPhaseKind
    decode_step: int | None
    operator_input: Tensor = field(repr=False, compare=False)
    residual_pre: Tensor = field(repr=False, compare=False)
    query_linear: Tensor = field(repr=False, compare=False)
    key_linear: Tensor = field(repr=False, compare=False)
    query_normalized_pre_rope: Tensor = field(repr=False, compare=False)
    key_normalized_pre_rope: Tensor = field(repr=False, compare=False)
    rope_cos: Tensor = field(repr=False, compare=False)
    rope_sin: Tensor = field(repr=False, compare=False)
    query_post_rope: Tensor = field(repr=False, compare=False)
    key_post_rope_current: Tensor = field(repr=False, compare=False)
    value_current: Tensor = field(repr=False, compare=False)
    key_history: Tensor = field(repr=False, compare=False)
    value_history: Tensor = field(repr=False, compare=False)
    qk_scores: Tensor = field(repr=False, compare=False)
    additive_mask: Tensor = field(repr=False, compare=False)
    probabilities: Tensor = field(repr=False, compare=False)
    context: Tensor = field(repr=False, compare=False)
    context_from_probabilities: Tensor = field(repr=False, compare=False)
    output_write: Tensor = field(repr=False, compare=False)
    residual_post_operator: Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.topology, Lfm2MatrixAttentionTopology):
            raise TypeError("topology must be Lfm2MatrixAttentionTopology.")
        object.__setattr__(self, "phase_id", _identifier(self.phase_id, "phase_id"))
        if self.phase_kind not in {"prefill", "decode"}:
            raise Lfm2MatrixAttentionReplayError(
                "phase_kind must be 'prefill' or 'decode'."
            )
        if self.phase_kind == "prefill":
            if self.decode_step is not None:
                raise Lfm2MatrixAttentionReplayError(
                    "A prefill phase cannot declare decode_step."
                )
        elif (
            isinstance(self.decode_step, bool)
            or not isinstance(self.decode_step, int)
            or self.decode_step < 0
        ):
            raise Lfm2MatrixAttentionReplayError(
                "A decode phase requires a non-negative decode_step."
            )
        self._validate_tensors()

    def _validate_tensors(self) -> None:
        topology = self.topology
        operator_input = _finite_tensor(
            self.operator_input,
            f"{self.phase_id}.operator_input",
            ndim=3,
        )
        batch, queries, hidden = (int(size) for size in operator_input.shape)
        if hidden != topology.hidden_size:
            raise Lfm2MatrixAttentionTopologyError(
                f"{self.phase_id} operator input width differs from hidden_size."
            )
        history = int(self.key_history.shape[2]) if self.key_history.ndim == 4 else -1
        expected_shapes = {
            "residual_pre": (batch, queries, hidden),
            "query_linear": (batch, topology.query_heads, queries, topology.head_dim),
            "key_linear": (
                batch,
                topology.key_value_heads,
                queries,
                topology.head_dim,
            ),
            "query_normalized_pre_rope": (
                batch,
                topology.query_heads,
                queries,
                topology.head_dim,
            ),
            "key_normalized_pre_rope": (
                batch,
                topology.key_value_heads,
                queries,
                topology.head_dim,
            ),
            "rope_cos": (batch, queries, topology.head_dim),
            "rope_sin": (batch, queries, topology.head_dim),
            "query_post_rope": (
                batch,
                topology.query_heads,
                queries,
                topology.head_dim,
            ),
            "key_post_rope_current": (
                batch,
                topology.key_value_heads,
                queries,
                topology.head_dim,
            ),
            "value_current": (
                batch,
                topology.key_value_heads,
                queries,
                topology.head_dim,
            ),
            "key_history": (
                batch,
                topology.key_value_heads,
                history,
                topology.head_dim,
            ),
            "value_history": (
                batch,
                topology.key_value_heads,
                history,
                topology.head_dim,
            ),
            "qk_scores": (batch, topology.query_heads, queries, history),
            "probabilities": (batch, topology.query_heads, queries, history),
            "context": (
                batch,
                topology.query_heads,
                queries,
                topology.head_dim,
            ),
            "context_from_probabilities": (
                batch,
                topology.query_heads,
                queries,
                topology.head_dim,
            ),
            "output_write": (batch, queries, hidden),
            "residual_post_operator": (batch, queries, hidden),
        }
        tensors = {
            name: _finite_tensor(
                getattr(self, name),
                f"{self.phase_id}.{name}",
                ndim=len(shape),
            )
            for name, shape in expected_shapes.items()
        }
        for name, shape in expected_shapes.items():
            if tuple(tensors[name].shape) != shape:
                raise Lfm2MatrixAttentionTopologyError(
                    f"{self.phase_id}.{name} has shape {tuple(tensors[name].shape)}, "
                    f"expected {shape}."
                )
        mask = _finite_tensor(
            self.additive_mask,
            f"{self.phase_id}.additive_mask",
            ndim=4,
        )
        if (
            int(mask.shape[0]) not in {1, batch}
            or int(mask.shape[1]) not in {1, topology.query_heads}
            or tuple(mask.shape[-2:]) != (queries, history)
        ):
            raise Lfm2MatrixAttentionTopologyError(
                f"{self.phase_id}.additive_mask is not broadcastable to "
                "[batch, query_heads, queries, history]."
            )
        if bool((mask > 0).any().item()):
            raise Lfm2MatrixAttentionReplayError(
                f"{self.phase_id}.additive_mask contains a positive score."
            )
        if not bool((mask == 0).any(dim=-1).all().item()):
            raise Lfm2MatrixAttentionReplayError(
                f"{self.phase_id}.additive_mask has a query with no valid key."
            )
        devices = {
            str(tensor.device) for tensor in (operator_input, mask, *tensors.values())
        }
        if len(devices) != 1:
            raise Lfm2MatrixAttentionReplayError(
                f"{self.phase_id} tensors must reside on one device."
            )

    @property
    def batch_size(self) -> int:
        return int(self.operator_input.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.operator_input.shape[1])

    @property
    def history_count(self) -> int:
        return int(self.key_history.shape[2])


@dataclass(frozen=True, slots=True)
class Lfm2MatrixAttentionReplayTolerance:
    """Numerical policy used only for copy-exact infrastructure calibration."""

    absolute: float = 2e-3
    relative: float = 2e-2

    def __post_init__(self) -> None:
        absolute = _nonnegative_float(self.absolute, "absolute")
        relative = _nonnegative_float(self.relative, "relative")
        if absolute > MAX_COPY_EXACT_ABSOLUTE_TOLERANCE:
            raise Lfm2MatrixAttentionReplayError(
                "absolute exceeds the independently fixed copy-exact ceiling."
            )
        if relative > MAX_COPY_EXACT_RELATIVE_TOLERANCE:
            raise Lfm2MatrixAttentionReplayError(
                "relative exceeds the independently fixed copy-exact ceiling."
            )
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "relative", relative)


@dataclass(frozen=True, slots=True)
class Lfm2MatrixAttentionCopyExactCheck:
    """One authenticated endpoint check from reference replay calibration."""

    phase_id: str
    endpoint: str
    shape: tuple[int, ...]
    maximum_absolute_error: float
    aggregate_relative_l2: float
    passed: bool


@dataclass(frozen=True, slots=True)
class Lfm2StructuredTensorMetrics:
    """Error metrics that retain per-batch and, where applicable, head axes."""

    shape: tuple[int, ...]
    aggregate_relative_l2: float
    aggregate_cosine: float
    maximum_absolute_error: float
    root_mean_square_error: float
    per_batch_relative_l2: tuple[float, ...]
    per_batch_cosine: tuple[float, ...]
    per_head_relative_l2: tuple[float, ...] | None
    per_head_cosine: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class Lfm2MatrixAttentionAttribution:
    """Exact one-sided and joint context/O intervention paths."""

    qk_only_context: Tensor = field(repr=False, compare=False)
    value_only_context: Tensor = field(repr=False, compare=False)
    joint_context: Tensor = field(repr=False, compare=False)
    parent_o_candidate_context: Tensor = field(repr=False, compare=False)
    candidate_o_parent_context: Tensor = field(repr=False, compare=False)
    candidate_o_candidate_context: Tensor = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Lfm2MatrixAttentionPhaseReplay:
    """One structured candidate phase after applying cell isolation semantics."""

    phase_id: str
    query_linear: Tensor = field(repr=False, compare=False)
    key_linear: Tensor = field(repr=False, compare=False)
    query_normalized_pre_rope: Tensor = field(repr=False, compare=False)
    key_normalized_pre_rope: Tensor = field(repr=False, compare=False)
    query_post_rope: Tensor = field(repr=False, compare=False)
    key_post_rope_current: Tensor = field(repr=False, compare=False)
    value_current: Tensor = field(repr=False, compare=False)
    key_history: Tensor = field(repr=False, compare=False)
    value_history: Tensor = field(repr=False, compare=False)
    qk_scores: Tensor = field(repr=False, compare=False)
    probabilities: Tensor = field(repr=False, compare=False)
    context: Tensor = field(repr=False, compare=False)
    output_write: Tensor = field(repr=False, compare=False)
    residual_post_operator: Tensor = field(repr=False, compare=False)
    attribution: Lfm2MatrixAttentionAttribution = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Lfm2MatrixAttentionPhaseMetrics:
    """Structured drift from one sealed parent phase."""

    phase_id: str
    query_post_rope: Lfm2StructuredTensorMetrics
    key_post_rope_current: Lfm2StructuredTensorMetrics
    value_current: Lfm2StructuredTensorMetrics
    qk_scores: Lfm2StructuredTensorMetrics
    probabilities: Lfm2StructuredTensorMetrics
    context: Lfm2StructuredTensorMetrics
    output_write: Lfm2StructuredTensorMetrics
    residual_post_operator: Lfm2StructuredTensorMetrics
    key_cache: Lfm2StructuredTensorMetrics
    value_cache: Lfm2StructuredTensorMetrics
    key_cache_prefix_drift: Lfm2StructuredTensorMetrics | None
    value_cache_prefix_drift: Lfm2StructuredTensorMetrics | None


@dataclass(frozen=True, slots=True)
class Lfm2MatrixAttentionReplayEvaluation:
    """Copy-authenticated C06-C10 replay and diagnostic measurements."""

    schema: str
    cell: Lfm2MatrixAttentionCell
    reference_bundle_id: str
    candidate_bundle_id: str
    active_components: tuple[str, ...]
    phases: tuple[Lfm2MatrixAttentionPhaseReplay, ...] = field(
        repr=False,
        compare=False,
    )
    phase_metrics: tuple[Lfm2MatrixAttentionPhaseMetrics, ...]
    copy_exact_metrics: tuple[Lfm2MatrixAttentionPhaseMetrics, ...]
    copy_exact_tolerance: Lfm2MatrixAttentionReplayTolerance
    copy_exact_endpoint_checks: tuple[Lfm2MatrixAttentionCopyExactCheck, ...]
    score_preservation: Lfm2AttentionScorePreservationReport
    copy_exact_verified: bool
    declared_logical_stored_parameter_count: int
    declared_physical_materialized_parameter_count: int
    declared_accounting_scope: str
    parameter_accounting_authority: bool
    parameter_accounting_evidence: str
    replay_sha256: str


def _validate_trajectory(
    phases: Sequence[Lfm2MatrixAttentionPhase],
) -> tuple[Lfm2MatrixAttentionPhase, ...]:
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        raise TypeError("phases must be a sequence of structured phases.")
    result = tuple(phases)
    if not result or any(
        not isinstance(phase, Lfm2MatrixAttentionPhase) for phase in result
    ):
        raise Lfm2MatrixAttentionReplayError(
            "phases must contain at least one Lfm2MatrixAttentionPhase."
        )
    topology = result[0].topology
    if any(phase.topology != topology for phase in result):
        raise Lfm2MatrixAttentionTopologyError(
            "Every phase must declare the same attention topology."
        )
    if result[0].phase_kind != "prefill":
        raise Lfm2MatrixAttentionContinuityError(
            "The first structured phase must be the complete prefill."
        )
    if result[0].query_count != result[0].history_count:
        raise Lfm2MatrixAttentionContinuityError(
            "The prefill operator input does not cover its complete K/V history."
        )
    if not (
        torch.equal(result[0].key_post_rope_current, result[0].key_history)
        and torch.equal(result[0].value_current, result[0].value_history)
    ):
        raise Lfm2MatrixAttentionContinuityError(
            "The prefill current K/V rows are not its complete sealed histories."
        )
    seen_ids: set[str] = set()
    for index, phase in enumerate(result):
        if phase.phase_id in seen_ids:
            raise Lfm2MatrixAttentionContinuityError(
                f"Duplicate phase_id {phase.phase_id!r}."
            )
        seen_ids.add(phase.phase_id)
        if index == 0:
            continue
        previous = result[index - 1]
        expected_step = index - 1
        if phase.phase_kind != "decode" or phase.decode_step != expected_step:
            raise Lfm2MatrixAttentionContinuityError(
                "Decode phases must follow prefill with contiguous zero-based steps."
            )
        if phase.query_count != 1:
            raise Lfm2MatrixAttentionContinuityError(
                f"Decode phase {phase.phase_id!r} must contribute exactly one token."
            )
        if phase.batch_size != previous.batch_size:
            raise Lfm2MatrixAttentionContinuityError(
                f"Decode phase {phase.phase_id!r} changed batch size."
            )
        if phase.history_count != previous.history_count + 1:
            raise Lfm2MatrixAttentionContinuityError(
                f"Decode phase {phase.phase_id!r} did not append one KV row."
            )
        prefix = previous.history_count
        if not (
            torch.equal(phase.key_history[:, :, :prefix, :], previous.key_history)
            and torch.equal(
                phase.value_history[:, :, :prefix, :],
                previous.value_history,
            )
        ):
            raise Lfm2MatrixAttentionContinuityError(
                f"Decode phase {phase.phase_id!r} changed its sealed KV prefix."
            )
        if not (
            torch.equal(phase.key_history[:, :, -1:, :], phase.key_post_rope_current)
            and torch.equal(phase.value_history[:, :, -1:, :], phase.value_current)
        ):
            raise Lfm2MatrixAttentionContinuityError(
                f"Decode phase {phase.phase_id!r} current KV is not its history suffix."
            )
    return result


def _validate_bundle_for_phase(
    bundle: Lfm2MatrixAttentionProjectionBundle,
    phase: Lfm2MatrixAttentionPhase,
    *,
    role: str,
) -> None:
    if not isinstance(bundle, Lfm2MatrixAttentionProjectionBundle):
        raise TypeError(f"{role} must be Lfm2MatrixAttentionProjectionBundle.")
    if bundle.topology != phase.topology:
        raise Lfm2MatrixAttentionTopologyError(
            f"{role} topology differs from the sealed phase topology."
        )
    for name, value in (
        ("query_norm_weight", bundle.query_norm_weight),
        ("key_norm_weight", bundle.key_norm_weight),
    ):
        if value.device != phase.operator_input.device:
            raise Lfm2MatrixAttentionProjectionError(
                f"{role}.{name} is not on the replay device."
            )


def _project(
    projection: Lfm2MatrixProjection,
    value: Tensor,
    *,
    width: int,
    name: str,
) -> Tensor:
    try:
        result = projection(value)
    except Exception as exc:
        raise Lfm2MatrixAttentionProjectionError(f"{name} failed.") from exc
    if not isinstance(result, Tensor):
        raise Lfm2MatrixAttentionProjectionError(f"{name} did not return a tensor.")
    expected = (int(value.shape[0]), int(value.shape[1]), width)
    if (
        result.layout != torch.strided
        or result.device != value.device
        or result.dtype != value.dtype
        or tuple(result.shape) != expected
        or not result.is_floating_point()
        or result.is_complex()
        or not bool(torch.isfinite(result).all().item())
    ):
        raise Lfm2MatrixAttentionProjectionError(
            f"{name} must return finite {expected} on {value.device} in {value.dtype}."
        )
    return result.contiguous()


def _candidate_query(
    phase: Lfm2MatrixAttentionPhase,
    bundle: Lfm2MatrixAttentionProjectionBundle,
) -> tuple[Tensor, Tensor, Tensor]:
    topology = phase.topology
    linear_flat = _project(
        bundle.query_projection,
        phase.operator_input,
        width=topology.hidden_size,
        name=f"{bundle.bundle_id}.query_projection",
    )
    linear_sequence = linear_flat.view(
        phase.batch_size,
        phase.query_count,
        topology.query_heads,
        topology.head_dim,
    )
    normalized_sequence = _runtime_rms_norm(
        linear_sequence,
        bundle.query_norm_weight,
        topology.norm_epsilon,
    )
    normalized = normalized_sequence.transpose(1, 2).contiguous()
    rotated = (
        normalized * phase.rope_cos.unsqueeze(1)
        + _rotate_half(torch, normalized) * phase.rope_sin.unsqueeze(1)
    ).contiguous()
    return linear_sequence.transpose(1, 2).contiguous(), normalized, rotated


def _candidate_key(
    phase: Lfm2MatrixAttentionPhase,
    bundle: Lfm2MatrixAttentionProjectionBundle,
) -> tuple[Tensor, Tensor, Tensor]:
    topology = phase.topology
    linear_flat = _project(
        bundle.key_projection,
        phase.operator_input,
        width=topology.key_value_width,
        name=f"{bundle.bundle_id}.key_projection",
    )
    linear_sequence = linear_flat.view(
        phase.batch_size,
        phase.query_count,
        topology.key_value_heads,
        topology.head_dim,
    )
    normalized_sequence = _runtime_rms_norm(
        linear_sequence,
        bundle.key_norm_weight,
        topology.norm_epsilon,
    )
    normalized = normalized_sequence.transpose(1, 2).contiguous()
    rotated = (
        normalized * phase.rope_cos.unsqueeze(1)
        + _rotate_half(torch, normalized) * phase.rope_sin.unsqueeze(1)
    ).contiguous()
    return linear_sequence.transpose(1, 2).contiguous(), normalized, rotated


def _candidate_value(
    phase: Lfm2MatrixAttentionPhase,
    bundle: Lfm2MatrixAttentionProjectionBundle,
) -> Tensor:
    topology = phase.topology
    flat = _project(
        bundle.value_projection,
        phase.operator_input,
        width=topology.key_value_width,
        name=f"{bundle.bundle_id}.value_projection",
    )
    return (
        flat.view(
            phase.batch_size,
            phase.query_count,
            topology.key_value_heads,
            topology.head_dim,
        )
        .transpose(1, 2)
        .contiguous()
    )


def _project_output(
    context: Tensor,
    phase: Lfm2MatrixAttentionPhase,
    bundle: Lfm2MatrixAttentionProjectionBundle,
) -> Tensor:
    flat = context.transpose(1, 2).reshape(
        phase.batch_size,
        phase.query_count,
        phase.topology.hidden_size,
    )
    return _project(
        bundle.output_projection,
        flat,
        width=phase.topology.hidden_size,
        name=f"{bundle.bundle_id}.output_projection",
    )


def _context_from_probabilities(
    probabilities: Tensor,
    value_history: Tensor,
    groups: int,
) -> Tensor:
    repeated_value = value_history.repeat_interleave(groups, dim=1)
    return torch.matmul(probabilities, repeated_value).contiguous()


def _replay_one_phase(
    *,
    phase: Lfm2MatrixAttentionPhase,
    cell: Lfm2MatrixAttentionCell,
    reference: Lfm2MatrixAttentionProjectionBundle,
    candidate: Lfm2MatrixAttentionProjectionBundle,
    prior_key_history: Tensor | None,
    prior_value_history: Tensor | None,
) -> Lfm2MatrixAttentionPhaseReplay:
    topology = phase.topology
    if cell in _QUERY_CELLS:
        query_linear, query_normalized, query = _candidate_query(phase, candidate)
    else:
        query_linear = phase.query_linear
        query_normalized = phase.query_normalized_pre_rope
        query = phase.query_post_rope

    if cell in _KEY_CELLS:
        key_linear, key_normalized, current_key = _candidate_key(phase, candidate)
        key_history = (
            current_key
            if prior_key_history is None
            else torch.cat((prior_key_history, current_key), dim=2).contiguous()
        )
    else:
        key_linear = phase.key_linear
        key_normalized = phase.key_normalized_pre_rope
        current_key = phase.key_post_rope_current
        key_history = phase.key_history

    if cell in _VALUE_CELLS:
        current_value = _candidate_value(phase, candidate)
        value_history = (
            current_value
            if prior_value_history is None
            else torch.cat((prior_value_history, current_value), dim=2).contiguous()
        )
    else:
        current_value = phase.value_current
        value_history = phase.value_history

    if int(key_history.shape[2]) != phase.history_count or tuple(
        key_history.shape
    ) != tuple(value_history.shape):
        raise Lfm2MatrixAttentionContinuityError(
            f"Candidate phase {phase.phase_id!r} did not reproduce KV trajectory length."
        )

    if cell == "C09":
        scores = phase.qk_scores
        probabilities = phase.probabilities
        context = phase.context
    elif cell == "C08":
        scores = phase.qk_scores
        probabilities = phase.probabilities
        context = _context_from_probabilities(
            probabilities,
            value_history,
            topology.groups,
        )
    else:
        causal = _reconstruct_attention_causal_path(
            torch,
            query=query,
            key_history=key_history,
            value_history=value_history,
            additive_mask=phase.additive_mask,
            groups=topology.groups,
            sample_count=phase.query_count,
            head_dim=topology.head_dim,
        )
        scores = causal["qk_scores"].contiguous()
        probabilities = causal["probabilities"].contiguous()
        context = causal["reconstructed_context"].contiguous()

    qk_only_context = _context_from_probabilities(
        probabilities,
        phase.value_history,
        topology.groups,
    )
    value_only_context = _context_from_probabilities(
        phase.probabilities,
        value_history,
        topology.groups,
    )
    parent_o_candidate_context = _project_output(context, phase, reference)
    candidate_o_parent_context = _project_output(phase.context, phase, candidate)
    candidate_o_candidate_context = _project_output(context, phase, candidate)
    if cell in _CANDIDATE_O_CELLS:
        output_write = candidate_o_candidate_context
    else:
        output_write = parent_o_candidate_context
    residual_post_operator = (phase.residual_pre + output_write).contiguous()
    return Lfm2MatrixAttentionPhaseReplay(
        phase_id=phase.phase_id,
        query_linear=query_linear,
        key_linear=key_linear,
        query_normalized_pre_rope=query_normalized,
        key_normalized_pre_rope=key_normalized,
        query_post_rope=query,
        key_post_rope_current=current_key,
        value_current=current_value,
        key_history=key_history,
        value_history=value_history,
        qk_scores=scores,
        probabilities=probabilities,
        context=context,
        output_write=output_write,
        residual_post_operator=residual_post_operator,
        attribution=Lfm2MatrixAttentionAttribution(
            qk_only_context=qk_only_context,
            value_only_context=value_only_context,
            joint_context=context,
            parent_o_candidate_context=parent_o_candidate_context,
            candidate_o_parent_context=candidate_o_parent_context,
            candidate_o_candidate_context=candidate_o_candidate_context,
        ),
    )


def _trajectory_replay(
    *,
    phases: tuple[Lfm2MatrixAttentionPhase, ...],
    cell: Lfm2MatrixAttentionCell,
    reference: Lfm2MatrixAttentionProjectionBundle,
    candidate: Lfm2MatrixAttentionProjectionBundle,
) -> tuple[Lfm2MatrixAttentionPhaseReplay, ...]:
    replays = []
    prior_key = None
    prior_value = None
    for phase in phases:
        replay = _replay_one_phase(
            phase=phase,
            cell=cell,
            reference=reference,
            candidate=candidate,
            prior_key_history=prior_key,
            prior_value_history=prior_value,
        )
        replays.append(replay)
        prior_key = replay.key_history
        prior_value = replay.value_history
    return tuple(replays)


def _vector_metrics(reference: Tensor, candidate: Tensor) -> tuple[float, float]:
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    difference = cand - ref
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    cand_norm = float(torch.linalg.vector_norm(cand).item())
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    if ref_norm == 0.0:
        relative = 0.0 if difference_norm == 0.0 else math.inf
    else:
        relative = difference_norm / ref_norm
    if ref_norm == 0.0 and cand_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or cand_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(ref, cand).item()) / (ref_norm * cand_norm)
        cosine = max(-1.0, min(1.0, cosine))
    return relative, cosine


def _structured_metrics(
    reference: Tensor,
    candidate: Tensor,
    *,
    head_axis: int | None,
) -> Lfm2StructuredTensorMetrics:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise Lfm2MatrixAttentionTopologyError(
            "Metric reference and candidate shapes differ."
        )
    ref = reference.detach().to(device="cpu", dtype=torch.float64).contiguous()
    cand = candidate.detach().to(device="cpu", dtype=torch.float64).contiguous()
    relative, cosine = _vector_metrics(ref, cand)
    difference = cand - ref
    batch_values = tuple(
        _vector_metrics(ref[index], cand[index]) for index in range(int(ref.shape[0]))
    )
    head_values = None
    if head_axis is not None:
        if head_axis != 1 or ref.ndim < 2:
            raise Lfm2MatrixAttentionReplayError(
                "Structured head metrics currently require head_axis=1."
            )
        head_values = tuple(
            _vector_metrics(ref[:, index], cand[:, index])
            for index in range(int(ref.shape[1]))
        )
    return Lfm2StructuredTensorMetrics(
        shape=tuple(int(size) for size in ref.shape),
        aggregate_relative_l2=relative,
        aggregate_cosine=cosine,
        maximum_absolute_error=float(difference.abs().max().item()),
        root_mean_square_error=float(difference.square().mean().sqrt().item()),
        per_batch_relative_l2=tuple(value[0] for value in batch_values),
        per_batch_cosine=tuple(value[1] for value in batch_values),
        per_head_relative_l2=(
            None if head_values is None else tuple(value[0] for value in head_values)
        ),
        per_head_cosine=(
            None if head_values is None else tuple(value[1] for value in head_values)
        ),
    )


def _phase_metrics(
    phase: Lfm2MatrixAttentionPhase,
    replay: Lfm2MatrixAttentionPhaseReplay,
) -> Lfm2MatrixAttentionPhaseMetrics:
    prefix = phase.history_count - phase.query_count
    key_prefix = None
    value_prefix = None
    if prefix > 0:
        key_prefix = _structured_metrics(
            phase.key_history[:, :, :prefix, :],
            replay.key_history[:, :, :prefix, :],
            head_axis=1,
        )
        value_prefix = _structured_metrics(
            phase.value_history[:, :, :prefix, :],
            replay.value_history[:, :, :prefix, :],
            head_axis=1,
        )
    return Lfm2MatrixAttentionPhaseMetrics(
        phase_id=phase.phase_id,
        query_post_rope=_structured_metrics(
            phase.query_post_rope,
            replay.query_post_rope,
            head_axis=1,
        ),
        key_post_rope_current=_structured_metrics(
            phase.key_post_rope_current,
            replay.key_post_rope_current,
            head_axis=1,
        ),
        value_current=_structured_metrics(
            phase.value_current,
            replay.value_current,
            head_axis=1,
        ),
        qk_scores=_structured_metrics(
            phase.qk_scores,
            replay.qk_scores,
            head_axis=1,
        ),
        probabilities=_structured_metrics(
            phase.probabilities,
            replay.probabilities,
            head_axis=1,
        ),
        context=_structured_metrics(phase.context, replay.context, head_axis=1),
        output_write=_structured_metrics(
            phase.output_write,
            replay.output_write,
            head_axis=None,
        ),
        residual_post_operator=_structured_metrics(
            phase.residual_post_operator,
            replay.residual_post_operator,
            head_axis=None,
        ),
        key_cache=_structured_metrics(
            phase.key_history,
            replay.key_history,
            head_axis=1,
        ),
        value_cache=_structured_metrics(
            phase.value_history,
            replay.value_history,
            head_axis=1,
        ),
        key_cache_prefix_drift=key_prefix,
        value_cache_prefix_drift=value_prefix,
    )


def _assert_copy_exact(
    *,
    phases: tuple[Lfm2MatrixAttentionPhase, ...],
    replays: tuple[Lfm2MatrixAttentionPhaseReplay, ...],
    tolerance: Lfm2MatrixAttentionReplayTolerance,
) -> tuple[Lfm2MatrixAttentionCopyExactCheck, ...]:
    endpoint_pairs = (
        ("query_linear", "query_linear"),
        ("key_linear", "key_linear"),
        ("query_normalized_pre_rope", "query_normalized_pre_rope"),
        ("key_normalized_pre_rope", "key_normalized_pre_rope"),
        ("query_post_rope", "query_post_rope"),
        ("key_post_rope_current", "key_post_rope_current"),
        ("value_current", "value_current"),
        ("key_history", "key_history"),
        ("value_history", "value_history"),
        ("qk_scores", "qk_scores"),
        ("probabilities", "probabilities"),
        ("context", "context"),
        ("output_write", "output_write"),
        ("residual_post_operator", "residual_post_operator"),
    )
    checks: list[Lfm2MatrixAttentionCopyExactCheck] = []
    for phase, replay in zip(phases, replays, strict=True):
        for phase_name, replay_name in endpoint_pairs:
            expected = getattr(phase, phase_name).detach().to(torch.float64)
            observed = getattr(replay, replay_name).detach().to(torch.float64)
            passed = bool(
                torch.allclose(
                    observed,
                    expected,
                    rtol=tolerance.relative,
                    atol=tolerance.absolute,
                )
            )
            metrics = _structured_metrics(expected, observed, head_axis=None)
            checks.append(
                Lfm2MatrixAttentionCopyExactCheck(
                    phase_id=phase.phase_id,
                    endpoint=phase_name,
                    shape=tuple(int(size) for size in expected.shape),
                    maximum_absolute_error=metrics.maximum_absolute_error,
                    aggregate_relative_l2=metrics.aggregate_relative_l2,
                    passed=passed,
                )
            )
            if not passed:
                maximum = metrics.maximum_absolute_error
                raise Lfm2MatrixAttentionCopyExactError(
                    f"Reference copy replay changed {phase.phase_id}.{phase_name} "
                    f"(max abs {maximum:.9g})."
                )
        reconstructed = replay.context.detach().to(torch.float64)
        cached_reconstruction = phase.context_from_probabilities.detach().to(
            torch.float64
        )
        passed = bool(
            torch.allclose(
                reconstructed,
                cached_reconstruction,
                rtol=tolerance.relative,
                atol=tolerance.absolute,
            )
        )
        metrics = _structured_metrics(
            cached_reconstruction,
            reconstructed,
            head_axis=None,
        )
        checks.append(
            Lfm2MatrixAttentionCopyExactCheck(
                phase_id=phase.phase_id,
                endpoint="context_from_probabilities",
                shape=tuple(int(size) for size in reconstructed.shape),
                maximum_absolute_error=metrics.maximum_absolute_error,
                aggregate_relative_l2=metrics.aggregate_relative_l2,
                passed=passed,
            )
        )
        if not passed:
            maximum = metrics.maximum_absolute_error
            raise Lfm2MatrixAttentionCopyExactError(
                f"Reference copy replay changed "
                f"{phase.phase_id}.context_from_probabilities "
                f"(max abs {maximum:.9g})."
            )
    return tuple(checks)


def _expanded_key_mask(phase: Lfm2MatrixAttentionPhase) -> Tensor:
    mask = phase.additive_mask == 0
    if int(mask.shape[0]) == 1 and phase.batch_size != 1:
        mask = mask.expand(
            phase.batch_size,
            int(mask.shape[1]),
            phase.query_count,
            phase.history_count,
        )
    return mask.contiguous()


def _active_components(cell: Lfm2MatrixAttentionCell) -> tuple[str, ...]:
    return {
        "C06": ("query_projection", "query_norm"),
        "C07": ("key_projection", "key_norm", "key_cache"),
        "C08": ("value_projection", "value_cache"),
        "C09": ("output_projection",),
        "C10": (
            "query_projection",
            "query_norm",
            "key_projection",
            "key_norm",
            "value_projection",
            "output_projection",
            "key_cache",
            "value_cache",
        ),
    }[cell]


def _canonical_metric_value(value: Any) -> Any:
    """Encode legitimate infinite errors without permitting ambiguous NaNs."""

    if isinstance(value, float):
        if math.isnan(value):
            raise Lfm2MatrixAttentionReplayError(
                "Attention replay metrics cannot contain NaN."
            )
        if math.isinf(value):
            return {
                "nonfinite_float": (
                    "positive_infinity" if value > 0.0 else "negative_infinity"
                )
            }
        return value
    if isinstance(value, dict):
        return {str(key): _canonical_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_metric_value(item) for item in value]
    return value


def _replay_sha256(
    *,
    cell: Lfm2MatrixAttentionCell,
    reference: Lfm2MatrixAttentionProjectionBundle,
    candidate: Lfm2MatrixAttentionProjectionBundle,
    phases: tuple[Lfm2MatrixAttentionPhase, ...],
    copy_replays: tuple[Lfm2MatrixAttentionPhaseReplay, ...],
    candidate_replays: tuple[Lfm2MatrixAttentionPhaseReplay, ...],
    copy_metrics: tuple[Lfm2MatrixAttentionPhaseMetrics, ...],
    candidate_metrics: tuple[Lfm2MatrixAttentionPhaseMetrics, ...],
    copy_tolerance: Lfm2MatrixAttentionReplayTolerance,
    copy_checks: tuple[Lfm2MatrixAttentionCopyExactCheck, ...],
    score_report: Lfm2AttentionScorePreservationReport,
) -> str:
    """Seal all input authority and every tensor/metric returned by evaluation."""

    digest = hashlib.sha256()
    accounting_evidence = "caller_declared_unverified_by_evaluator"
    for value in (
        LFM2_MATRIX_ATTENTION_REPLAY_SCHEMA,
        cell,
        reference.bundle_id,
        candidate.bundle_id,
        str(candidate.declared_logical_stored_parameter_count),
        str(candidate.declared_physical_materialized_parameter_count),
        candidate.declared_accounting_scope,
        accounting_evidence,
        str(copy_tolerance.absolute),
        str(copy_tolerance.relative),
        score_report.report_sha256,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")

    topology = phases[0].topology
    for value in (
        topology.hidden_size,
        topology.query_heads,
        topology.key_value_heads,
        topology.head_dim,
        topology.norm_epsilon,
    ):
        digest.update(repr(value).encode("ascii"))
        digest.update(b"\0")

    phase_tensor_names = (
        "operator_input",
        "residual_pre",
        "query_linear",
        "key_linear",
        "query_normalized_pre_rope",
        "key_normalized_pre_rope",
        "rope_cos",
        "rope_sin",
        "query_post_rope",
        "key_post_rope_current",
        "value_current",
        "key_history",
        "value_history",
        "qk_scores",
        "additive_mask",
        "probabilities",
        "context",
        "context_from_probabilities",
        "output_write",
        "residual_post_operator",
    )
    replay_tensor_names = (
        "query_linear",
        "key_linear",
        "query_normalized_pre_rope",
        "key_normalized_pre_rope",
        "query_post_rope",
        "key_post_rope_current",
        "value_current",
        "key_history",
        "value_history",
        "qk_scores",
        "probabilities",
        "context",
        "output_write",
        "residual_post_operator",
    )
    attribution_tensor_names = (
        "qk_only_context",
        "value_only_context",
        "joint_context",
        "parent_o_candidate_context",
        "candidate_o_parent_context",
        "candidate_o_candidate_context",
    )
    for phase in phases:
        for value in (phase.phase_id, phase.phase_kind, repr(phase.decode_step)):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for name in phase_tensor_names:
            digest.update(name.encode("ascii"))
            digest.update(
                bytes.fromhex(lfm2_authenticated_tensor_sha256(getattr(phase, name)))
            )
    for lane, replays in (
        ("copy", copy_replays),
        ("candidate", candidate_replays),
    ):
        digest.update(lane.encode("ascii"))
        digest.update(b"\0")
        for replay in replays:
            digest.update(replay.phase_id.encode("utf-8"))
            digest.update(b"\0")
            for name in replay_tensor_names:
                digest.update(name.encode("ascii"))
                digest.update(
                    bytes.fromhex(
                        lfm2_authenticated_tensor_sha256(getattr(replay, name))
                    )
                )
            for name in attribution_tensor_names:
                digest.update(f"attribution.{name}".encode("ascii"))
                digest.update(
                    bytes.fromhex(
                        lfm2_authenticated_tensor_sha256(
                            getattr(replay.attribution, name)
                        )
                    )
                )
    digest.update(
        bytes.fromhex(
            canonical_sha256(
                _canonical_metric_value(
                    {
                        "copy_metrics": [asdict(value) for value in copy_metrics],
                        "candidate_metrics": [
                            asdict(value) for value in candidate_metrics
                        ],
                        "copy_checks": [asdict(value) for value in copy_checks],
                    }
                )
            )
        )
    )
    return digest.hexdigest()


def evaluate_lfm2_matrix_attention_replay(
    *,
    phases: Sequence[Lfm2MatrixAttentionPhase],
    reference: Lfm2MatrixAttentionProjectionBundle,
    candidate: Lfm2MatrixAttentionProjectionBundle,
    cell: Lfm2MatrixAttentionCell,
    tolerance: Lfm2MatrixAttentionReplayTolerance | None = None,
    score_thresholds: Lfm2AttentionScorePreservationThresholds | None = None,
    top_k: int = 8,
    relative_margin_floor: float = 0.5,
) -> Lfm2MatrixAttentionReplayEvaluation:
    """Replay one same-topology C06-C10 intervention over a sealed trajectory.

    ``reference`` is executable calibration authority, not a second trace.  It
    is replayed through C10 first and must reproduce all sealed parent primals.
    The candidate pass then changes only the component named by ``cell``.
    """

    if cell not in _SUPPORTED_CELLS:
        raise Lfm2MatrixAttentionReplayError(
            f"cell must be one of {sorted(_SUPPORTED_CELLS)}."
        )
    trajectory = _validate_trajectory(phases)
    _validate_bundle_for_phase(reference, trajectory[0], role="reference")
    _validate_bundle_for_phase(candidate, trajectory[0], role="candidate")
    policy = tolerance or Lfm2MatrixAttentionReplayTolerance()
    if not isinstance(policy, Lfm2MatrixAttentionReplayTolerance):
        raise TypeError("tolerance must be Lfm2MatrixAttentionReplayTolerance.")

    with torch.no_grad():
        copy_replays = _trajectory_replay(
            phases=trajectory,
            cell="C10",
            reference=reference,
            candidate=reference,
        )
        copy_checks = _assert_copy_exact(
            phases=trajectory,
            replays=copy_replays,
            tolerance=policy,
        )
        candidate_replays = _trajectory_replay(
            phases=trajectory,
            cell=cell,
            reference=reference,
            candidate=candidate,
        )

    copy_metrics = tuple(
        _phase_metrics(phase, replay)
        for phase, replay in zip(trajectory, copy_replays, strict=True)
    )
    candidate_metrics = tuple(
        _phase_metrics(phase, replay)
        for phase, replay in zip(trajectory, candidate_replays, strict=True)
    )
    score_report = evaluate_lfm2_structural_attention_score_preservation(
        tuple(phase.qk_scores for phase in trajectory),
        tuple(replay.qk_scores for replay in candidate_replays),
        key_masks=tuple(_expanded_key_mask(phase) for phase in trajectory),
        top_k=top_k,
        relative_margin_floor=relative_margin_floor,
        thresholds=score_thresholds,
    )
    replay_digest = _replay_sha256(
        cell=cell,
        reference=reference,
        candidate=candidate,
        phases=trajectory,
        copy_replays=copy_replays,
        candidate_replays=candidate_replays,
        copy_metrics=copy_metrics,
        candidate_metrics=candidate_metrics,
        copy_tolerance=policy,
        copy_checks=copy_checks,
        score_report=score_report,
    )
    return Lfm2MatrixAttentionReplayEvaluation(
        schema=LFM2_MATRIX_ATTENTION_REPLAY_SCHEMA,
        cell=cell,
        reference_bundle_id=reference.bundle_id,
        candidate_bundle_id=candidate.bundle_id,
        active_components=_active_components(cell),
        phases=candidate_replays,
        phase_metrics=candidate_metrics,
        copy_exact_metrics=copy_metrics,
        copy_exact_tolerance=policy,
        copy_exact_endpoint_checks=copy_checks,
        score_preservation=score_report,
        copy_exact_verified=True,
        declared_logical_stored_parameter_count=(
            candidate.declared_logical_stored_parameter_count
        ),
        declared_physical_materialized_parameter_count=(
            candidate.declared_physical_materialized_parameter_count
        ),
        declared_accounting_scope=candidate.declared_accounting_scope,
        parameter_accounting_authority=False,
        parameter_accounting_evidence="caller_declared_unverified_by_evaluator",
        replay_sha256=replay_digest,
    )


__all__ = [
    "LFM2_MATRIX_ATTENTION_REPLAY_SCHEMA",
    "MAX_COPY_EXACT_ABSOLUTE_TOLERANCE",
    "MAX_COPY_EXACT_RELATIVE_TOLERANCE",
    "Lfm2MatrixAttentionAttribution",
    "Lfm2MatrixAttentionCell",
    "Lfm2MatrixAttentionContinuityError",
    "Lfm2MatrixAttentionCopyExactCheck",
    "Lfm2MatrixAttentionCopyExactError",
    "Lfm2MatrixAttentionPhase",
    "Lfm2MatrixAttentionPhaseMetrics",
    "Lfm2MatrixAttentionPhaseReplay",
    "Lfm2MatrixAttentionProjectionBundle",
    "Lfm2MatrixAttentionProjectionError",
    "Lfm2MatrixAttentionReplayError",
    "Lfm2MatrixAttentionReplayEvaluation",
    "Lfm2MatrixAttentionReplayTolerance",
    "Lfm2MatrixAttentionTopology",
    "Lfm2MatrixAttentionTopologyError",
    "Lfm2StructuredTensorMetrics",
    "evaluate_lfm2_matrix_attention_replay",
]
