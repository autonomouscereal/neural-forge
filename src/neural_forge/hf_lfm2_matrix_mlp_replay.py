"""Structured, cached-only LFM2 SwiGLU replay for matrix cells C11-C13.

This module evaluates one isolated MLP projection intervention against sealed
parent tensors.  It deliberately does not load a model, capture traces, train,
publish a child, or decide whether an answer is intelligent.  The caller must
first provide a reference projection bundle that reproduces the sealed gate,
up, post-SwiGLU, down-write, and residual endpoints.  Only then is the chosen
candidate projection replayed through the real nonlinear SwiGLU path.

The result therefore replaces the old ``not-composable`` sentinel for C11-C13
with an actual component and residual-boundary measurement, while remaining a
diagnostic rather than physical-compaction or intelligence authority.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_MATRIX_MLP_REPLAY_SCHEMA = "neural-forge-lfm2-matrix-structured-mlp-replay-v1"
Lfm2MatrixMlpCell: TypeAlias = Literal["C11", "C12", "C13"]
Lfm2MatrixMlpProjection: TypeAlias = Callable[[Tensor], Tensor]

_SUPPORTED_CELLS = frozenset({"C11", "C12", "C13"})

# Reference replay is an infrastructure proof.  Callers may tighten these
# tolerances but cannot relax them to authenticate an incorrect implementation.
MAX_COPY_EXACT_ABSOLUTE_TOLERANCE = 2e-3
MAX_COPY_EXACT_RELATIVE_TOLERANCE = 2e-2


class Lfm2MatrixMlpReplayError(ValueError):
    """A structured phase, projection bundle, or replay invariant is invalid."""


class Lfm2MatrixMlpTopologyError(Lfm2MatrixMlpReplayError):
    """The supplied tensors do not have legal same-topology SwiGLU geometry."""


class Lfm2MatrixMlpProjectionError(Lfm2MatrixMlpReplayError):
    """A projection callable violated its declared tensor contract."""


class Lfm2MatrixMlpCopyExactError(Lfm2MatrixMlpReplayError):
    """The reference projection bundle did not reproduce the sealed trace."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2MatrixMlpReplayError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2MatrixMlpReplayError(f"{name} must be a non-negative integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2MatrixMlpReplayError(f"{name} must be non-negative and finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2MatrixMlpReplayError(
            f"{name} must be non-negative and finite."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2MatrixMlpReplayError(f"{name} must be non-negative and finite.")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2MatrixMlpReplayError(f"{name} must be a non-empty string.")
    return value.strip()


def _finite_rank_three(value: Any, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 3
        or any(int(size) <= 0 for size in value.shape)
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise Lfm2MatrixMlpReplayError(
            f"{name} must be a non-empty dense real rank-three tensor."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise Lfm2MatrixMlpReplayError(f"{name} contains a non-finite value.")
    return value


@dataclass(frozen=True, slots=True)
class Lfm2MatrixMlpTopology:
    """Same-topology hidden and intermediate dimensions for one MLP site."""

    hidden_size: int
    intermediate_size: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hidden_size",
            _positive_int(self.hidden_size, "hidden_size"),
        )
        object.__setattr__(
            self,
            "intermediate_size",
            _positive_int(self.intermediate_size, "intermediate_size"),
        )


@dataclass(frozen=True, slots=True)
class Lfm2MatrixMlpProjectionBundle:
    """Executable gate/up/down projections plus non-authoritative counts.

    Projection callables consume rank-three tensors.  Gate and up map hidden
    width to intermediate width; down maps intermediate width to hidden width.
    Declared counts are sealed for provenance but are not verified here.  Only
    an authenticated installed-state and alias manifest can prove physical
    parameter compaction.
    """

    bundle_id: str
    topology: Lfm2MatrixMlpTopology
    gate_projection: Lfm2MatrixMlpProjection = field(repr=False, compare=False)
    up_projection: Lfm2MatrixMlpProjection = field(repr=False, compare=False)
    down_projection: Lfm2MatrixMlpProjection = field(repr=False, compare=False)
    declared_logical_stored_parameter_count: int
    declared_physical_materialized_parameter_count: int
    declared_accounting_scope: str = "complete_mlp_projection_bundle"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _identifier(self.bundle_id, "bundle_id"))
        if not isinstance(self.topology, Lfm2MatrixMlpTopology):
            raise TypeError("topology must be Lfm2MatrixMlpTopology.")
        for name in ("gate_projection", "up_projection", "down_projection"):
            if not callable(getattr(self, name)):
                raise Lfm2MatrixMlpProjectionError(f"{name} must be callable.")
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
class Lfm2MatrixMlpPhase:
    """One sealed parent MLP phase with batch/token axes preserved."""

    topology: Lfm2MatrixMlpTopology
    phase_id: str
    mlp_input: Tensor = field(repr=False, compare=False)
    residual_post_operator: Tensor = field(repr=False, compare=False)
    gate: Tensor = field(repr=False, compare=False)
    up: Tensor = field(repr=False, compare=False)
    post_swiglu: Tensor = field(repr=False, compare=False)
    output_write: Tensor = field(repr=False, compare=False)
    residual_post_layer: Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.topology, Lfm2MatrixMlpTopology):
            raise TypeError("topology must be Lfm2MatrixMlpTopology.")
        object.__setattr__(self, "phase_id", _identifier(self.phase_id, "phase_id"))
        tensors = {
            name: _finite_rank_three(getattr(self, name), f"{self.phase_id}.{name}")
            for name in (
                "mlp_input",
                "residual_post_operator",
                "gate",
                "up",
                "post_swiglu",
                "output_write",
                "residual_post_layer",
            )
        }
        batch, tokens, hidden = (int(size) for size in tensors["mlp_input"].shape)
        if hidden != self.topology.hidden_size:
            raise Lfm2MatrixMlpTopologyError(
                f"{self.phase_id} MLP input width differs from hidden_size."
            )
        hidden_shape = (batch, tokens, self.topology.hidden_size)
        intermediate_shape = (batch, tokens, self.topology.intermediate_size)
        expected_shapes = {
            "mlp_input": hidden_shape,
            "residual_post_operator": hidden_shape,
            "gate": intermediate_shape,
            "up": intermediate_shape,
            "post_swiglu": intermediate_shape,
            "output_write": hidden_shape,
            "residual_post_layer": hidden_shape,
        }
        for name, expected in expected_shapes.items():
            if tuple(tensors[name].shape) != expected:
                raise Lfm2MatrixMlpTopologyError(
                    f"{self.phase_id}.{name} has shape {tuple(tensors[name].shape)}, "
                    f"expected {expected}."
                )
        devices = {str(tensor.device) for tensor in tensors.values()}
        if len(devices) != 1:
            raise Lfm2MatrixMlpReplayError(
                f"{self.phase_id} tensors must reside on one device."
            )
        dtypes = {tensor.dtype for tensor in tensors.values()}
        if len(dtypes) != 1:
            raise Lfm2MatrixMlpReplayError(
                f"{self.phase_id} tensors must use one runtime dtype."
            )

    @property
    def batch_size(self) -> int:
        return int(self.mlp_input.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.mlp_input.shape[1])


@dataclass(frozen=True, slots=True)
class Lfm2MatrixMlpReplayTolerance:
    """Numerical policy used only for copy-exact infrastructure calibration."""

    absolute: float = 2e-3
    relative: float = 2e-2

    def __post_init__(self) -> None:
        absolute = _nonnegative_float(self.absolute, "absolute")
        relative = _nonnegative_float(self.relative, "relative")
        if absolute > MAX_COPY_EXACT_ABSOLUTE_TOLERANCE:
            raise Lfm2MatrixMlpReplayError(
                "absolute exceeds the independently fixed copy-exact ceiling."
            )
        if relative > MAX_COPY_EXACT_RELATIVE_TOLERANCE:
            raise Lfm2MatrixMlpReplayError(
                "relative exceeds the independently fixed copy-exact ceiling."
            )
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "relative", relative)


@dataclass(frozen=True, slots=True)
class Lfm2MatrixMlpCopyExactCheck:
    """One endpoint check from reference replay calibration."""

    phase_id: str
    endpoint: str
    shape: tuple[int, ...]
    maximum_absolute_error: float
    aggregate_relative_l2: float
    passed: bool


@dataclass(frozen=True, slots=True)
class Lfm2MatrixMlpTensorMetrics:
    """Drift metrics retaining per-batch values and the original tensor shape."""

    shape: tuple[int, ...]
    aggregate_relative_l2: float
    aggregate_cosine: float
    maximum_absolute_error: float
    root_mean_square_error: float
    per_batch_relative_l2: tuple[float, ...]
    per_batch_cosine: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Lfm2MatrixMlpAttribution:
    """One-sided and joint nonlinear intervention paths."""

    candidate_gate_parent_up_post: Tensor = field(repr=False, compare=False)
    parent_gate_candidate_up_post: Tensor = field(repr=False, compare=False)
    candidate_gate_candidate_up_post: Tensor = field(repr=False, compare=False)
    parent_down_candidate_gate_post: Tensor = field(repr=False, compare=False)
    parent_down_candidate_up_post: Tensor = field(repr=False, compare=False)
    parent_down_joint_post: Tensor = field(repr=False, compare=False)
    candidate_down_parent_post: Tensor = field(repr=False, compare=False)
    candidate_down_joint_post: Tensor = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Lfm2MatrixMlpPhaseReplay:
    """One actual SwiGLU replay under C11, C12, or C13 isolation semantics."""

    phase_id: str
    gate: Tensor = field(repr=False, compare=False)
    up: Tensor = field(repr=False, compare=False)
    post_swiglu: Tensor = field(repr=False, compare=False)
    output_write: Tensor = field(repr=False, compare=False)
    residual_post_layer: Tensor = field(repr=False, compare=False)
    attribution: Lfm2MatrixMlpAttribution = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Lfm2MatrixMlpPhaseMetrics:
    """Structured candidate drift from one sealed parent phase."""

    phase_id: str
    gate: Lfm2MatrixMlpTensorMetrics
    up: Lfm2MatrixMlpTensorMetrics
    post_swiglu: Lfm2MatrixMlpTensorMetrics
    output_write: Lfm2MatrixMlpTensorMetrics
    residual_post_layer: Lfm2MatrixMlpTensorMetrics


@dataclass(frozen=True, slots=True)
class Lfm2MatrixMlpReplayEvaluation:
    """Copy-authenticated C11-C13 replay and diagnostic measurements."""

    schema: str
    cell: Lfm2MatrixMlpCell
    reference_bundle_id: str
    candidate_bundle_id: str
    active_components: tuple[str, ...]
    composition_path: tuple[str, ...]
    phases: tuple[Lfm2MatrixMlpPhaseReplay, ...] = field(repr=False, compare=False)
    phase_metrics: tuple[Lfm2MatrixMlpPhaseMetrics, ...]
    copy_exact_metrics: tuple[Lfm2MatrixMlpPhaseMetrics, ...]
    copy_exact_tolerance: Lfm2MatrixMlpReplayTolerance
    copy_exact_endpoint_checks: tuple[Lfm2MatrixMlpCopyExactCheck, ...]
    copy_exact_verified: bool
    mlp_branch_composable: bool
    residual_post_layer_composable: bool
    downstream_trajectory_composable: bool
    declared_logical_stored_parameter_count: int
    declared_physical_materialized_parameter_count: int
    declared_accounting_scope: str
    parameter_accounting_authority: bool
    parameter_accounting_evidence: str
    replay_sha256: str


def _validate_phases(
    phases: Sequence[Lfm2MatrixMlpPhase],
) -> tuple[Lfm2MatrixMlpPhase, ...]:
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        raise TypeError("phases must be a sequence of structured MLP phases.")
    result = tuple(phases)
    if not result or any(not isinstance(phase, Lfm2MatrixMlpPhase) for phase in result):
        raise Lfm2MatrixMlpReplayError(
            "phases must contain at least one Lfm2MatrixMlpPhase."
        )
    topology = result[0].topology
    topology.__post_init__()
    for phase in result:
        phase.__post_init__()
    if any(phase.topology != topology for phase in result):
        raise Lfm2MatrixMlpTopologyError("All phases must use one MLP topology.")
    phase_ids = tuple(phase.phase_id for phase in result)
    if len(set(phase_ids)) != len(phase_ids):
        raise Lfm2MatrixMlpReplayError("phase_id values must be unique.")
    return result


def _validate_bundle(
    bundle: Lfm2MatrixMlpProjectionBundle,
    phase: Lfm2MatrixMlpPhase,
    *,
    role: str,
) -> None:
    if not isinstance(bundle, Lfm2MatrixMlpProjectionBundle):
        raise TypeError(f"{role} must be Lfm2MatrixMlpProjectionBundle.")
    bundle.topology.__post_init__()
    bundle.__post_init__()
    if bundle.topology != phase.topology:
        raise Lfm2MatrixMlpTopologyError(
            f"{role} projection topology differs from the sealed phase topology."
        )


def _project(
    projection: Lfm2MatrixMlpProjection,
    value: Tensor,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> Tensor:
    try:
        # Candidate code receives an isolated value so an in-place projection
        # cannot corrupt authenticated phase authority.
        result = projection(value.detach().clone())
    except Exception as exc:
        raise Lfm2MatrixMlpProjectionError(f"{name} raised during replay.") from exc
    try:
        result = _finite_rank_three(result, name)
    except Lfm2MatrixMlpReplayError as exc:
        raise Lfm2MatrixMlpProjectionError(str(exc)) from exc
    if tuple(result.shape) != expected_shape:
        raise Lfm2MatrixMlpProjectionError(
            f"{name} returned shape {tuple(result.shape)}, expected {expected_shape}."
        )
    if result.device != value.device:
        raise Lfm2MatrixMlpProjectionError(
            f"{name} moved the projection to a different device."
        )
    if result.dtype != value.dtype:
        raise Lfm2MatrixMlpProjectionError(
            f"{name} changed the runtime dtype from {value.dtype} to {result.dtype}."
        )
    return result.detach().contiguous().clone()


def _swiglu(gate: Tensor, up: Tensor) -> Tensor:
    # This is the exact LFM2 dense MLP nonlinearity: silu(w1(x)) * w3(x).
    return F.silu(gate) * up


def _replay_one_phase(
    *,
    phase: Lfm2MatrixMlpPhase,
    cell: Lfm2MatrixMlpCell,
    reference: Lfm2MatrixMlpProjectionBundle,
    candidate: Lfm2MatrixMlpProjectionBundle,
) -> Lfm2MatrixMlpPhaseReplay:
    topology = phase.topology
    batch, tokens = phase.batch_size, phase.token_count
    intermediate_shape = (batch, tokens, topology.intermediate_size)
    hidden_shape = (batch, tokens, topology.hidden_size)

    parent_gate = _project(
        reference.gate_projection,
        phase.mlp_input,
        expected_shape=intermediate_shape,
        name=f"{phase.phase_id}.reference.gate_projection",
    )
    parent_up = _project(
        reference.up_projection,
        phase.mlp_input,
        expected_shape=intermediate_shape,
        name=f"{phase.phase_id}.reference.up_projection",
    )
    candidate_gate = _project(
        candidate.gate_projection,
        phase.mlp_input,
        expected_shape=intermediate_shape,
        name=f"{phase.phase_id}.candidate.gate_projection",
    )
    candidate_up = _project(
        candidate.up_projection,
        phase.mlp_input,
        expected_shape=intermediate_shape,
        name=f"{phase.phase_id}.candidate.up_projection",
    )

    parent_post = _swiglu(parent_gate, parent_up)
    candidate_gate_parent_up_post = _swiglu(candidate_gate, parent_up)
    parent_gate_candidate_up_post = _swiglu(parent_gate, candidate_up)
    candidate_joint_post = _swiglu(candidate_gate, candidate_up)

    parent_down_candidate_gate_post = _project(
        reference.down_projection,
        candidate_gate_parent_up_post,
        expected_shape=hidden_shape,
        name=f"{phase.phase_id}.reference.down_projection.candidate_gate_post",
    )
    parent_down_candidate_up_post = _project(
        reference.down_projection,
        parent_gate_candidate_up_post,
        expected_shape=hidden_shape,
        name=f"{phase.phase_id}.reference.down_projection.candidate_up_post",
    )
    parent_down_joint_post = _project(
        reference.down_projection,
        candidate_joint_post,
        expected_shape=hidden_shape,
        name=f"{phase.phase_id}.reference.down_projection.joint_post",
    )
    candidate_down_parent_post = _project(
        candidate.down_projection,
        parent_post,
        expected_shape=hidden_shape,
        name=f"{phase.phase_id}.candidate.down_projection.parent_post",
    )
    candidate_down_joint_post = _project(
        candidate.down_projection,
        candidate_joint_post,
        expected_shape=hidden_shape,
        name=f"{phase.phase_id}.candidate.down_projection.joint_post",
    )

    if cell == "C11":
        gate = candidate_gate
        up = parent_up
        post_swiglu = candidate_gate_parent_up_post
        output_write = parent_down_candidate_gate_post
    elif cell == "C12":
        gate = parent_gate
        up = candidate_up
        post_swiglu = parent_gate_candidate_up_post
        output_write = parent_down_candidate_up_post
    else:  # C13
        gate = parent_gate
        up = parent_up
        post_swiglu = parent_post
        output_write = candidate_down_parent_post

    return Lfm2MatrixMlpPhaseReplay(
        phase_id=phase.phase_id,
        gate=gate,
        up=up,
        post_swiglu=post_swiglu,
        output_write=output_write,
        residual_post_layer=phase.residual_post_operator + output_write,
        attribution=Lfm2MatrixMlpAttribution(
            candidate_gate_parent_up_post=candidate_gate_parent_up_post,
            parent_gate_candidate_up_post=parent_gate_candidate_up_post,
            candidate_gate_candidate_up_post=candidate_joint_post,
            parent_down_candidate_gate_post=parent_down_candidate_gate_post,
            parent_down_candidate_up_post=parent_down_candidate_up_post,
            parent_down_joint_post=parent_down_joint_post,
            candidate_down_parent_post=candidate_down_parent_post,
            candidate_down_joint_post=candidate_down_joint_post,
        ),
    )


def _trajectory_replay(
    *,
    phases: tuple[Lfm2MatrixMlpPhase, ...],
    cell: Lfm2MatrixMlpCell,
    reference: Lfm2MatrixMlpProjectionBundle,
    candidate: Lfm2MatrixMlpProjectionBundle,
) -> tuple[Lfm2MatrixMlpPhaseReplay, ...]:
    return tuple(
        _replay_one_phase(
            phase=phase,
            cell=cell,
            reference=reference,
            candidate=candidate,
        )
        for phase in phases
    )


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


def _tensor_metrics(reference: Tensor, candidate: Tensor) -> Lfm2MatrixMlpTensorMetrics:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise Lfm2MatrixMlpTopologyError(
            "Metric reference and candidate shapes differ."
        )
    ref = reference.detach().to(device="cpu", dtype=torch.float64).contiguous()
    cand = candidate.detach().to(device="cpu", dtype=torch.float64).contiguous()
    relative, cosine = _vector_metrics(ref, cand)
    difference = cand - ref
    batch_values = tuple(
        _vector_metrics(ref[index], cand[index]) for index in range(int(ref.shape[0]))
    )
    return Lfm2MatrixMlpTensorMetrics(
        shape=tuple(int(size) for size in ref.shape),
        aggregate_relative_l2=relative,
        aggregate_cosine=cosine,
        maximum_absolute_error=float(difference.abs().max().item()),
        root_mean_square_error=float(difference.square().mean().sqrt().item()),
        per_batch_relative_l2=tuple(value[0] for value in batch_values),
        per_batch_cosine=tuple(value[1] for value in batch_values),
    )


def _phase_metrics(
    phase: Lfm2MatrixMlpPhase,
    replay: Lfm2MatrixMlpPhaseReplay,
) -> Lfm2MatrixMlpPhaseMetrics:
    return Lfm2MatrixMlpPhaseMetrics(
        phase_id=phase.phase_id,
        gate=_tensor_metrics(phase.gate, replay.gate),
        up=_tensor_metrics(phase.up, replay.up),
        post_swiglu=_tensor_metrics(phase.post_swiglu, replay.post_swiglu),
        output_write=_tensor_metrics(phase.output_write, replay.output_write),
        residual_post_layer=_tensor_metrics(
            phase.residual_post_layer,
            replay.residual_post_layer,
        ),
    )


def _copy_check(
    *,
    phase_id: str,
    endpoint: str,
    reference: Tensor,
    candidate: Tensor,
    tolerance: Lfm2MatrixMlpReplayTolerance,
) -> Lfm2MatrixMlpCopyExactCheck:
    metrics = _tensor_metrics(reference, candidate)
    passed = bool(
        torch.allclose(
            candidate.detach().to(torch.float64),
            reference.detach().to(torch.float64),
            atol=tolerance.absolute,
            rtol=tolerance.relative,
        )
    )
    return Lfm2MatrixMlpCopyExactCheck(
        phase_id=phase_id,
        endpoint=endpoint,
        shape=metrics.shape,
        maximum_absolute_error=metrics.maximum_absolute_error,
        aggregate_relative_l2=metrics.aggregate_relative_l2,
        passed=passed,
    )


def _assert_copy_exact(
    *,
    phases: tuple[Lfm2MatrixMlpPhase, ...],
    replays: tuple[Lfm2MatrixMlpPhaseReplay, ...],
    tolerance: Lfm2MatrixMlpReplayTolerance,
) -> tuple[Lfm2MatrixMlpCopyExactCheck, ...]:
    checks = []
    for phase, replay in zip(phases, replays, strict=True):
        for endpoint in (
            "gate",
            "up",
            "post_swiglu",
            "output_write",
            "residual_post_layer",
        ):
            checks.append(
                _copy_check(
                    phase_id=phase.phase_id,
                    endpoint=endpoint,
                    reference=getattr(phase, endpoint),
                    candidate=getattr(replay, endpoint),
                    tolerance=tolerance,
                )
            )
    failed = tuple(check for check in checks if not check.passed)
    if failed:
        first = failed[0]
        raise Lfm2MatrixMlpCopyExactError(
            "Reference MLP replay failed copy-exact calibration at "
            f"{first.phase_id}.{first.endpoint}: max_abs="
            f"{first.maximum_absolute_error:.6g}, relative_l2="
            f"{first.aggregate_relative_l2:.6g}."
        )
    return tuple(checks)


def _active_components(cell: Lfm2MatrixMlpCell) -> tuple[str, ...]:
    return {
        "C11": ("gate_projection",),
        "C12": ("up_projection",),
        "C13": ("down_projection",),
    }[cell]


def _composition_path(cell: Lfm2MatrixMlpCell) -> tuple[str, ...]:
    return {
        "C11": (
            "candidate_gate_projection",
            "reference_up_projection",
            "swiglu",
            "reference_down_projection",
            "residual_add",
        ),
        "C12": (
            "reference_gate_projection",
            "candidate_up_projection",
            "swiglu",
            "reference_down_projection",
            "residual_add",
        ),
        "C13": (
            "reference_gate_projection",
            "reference_up_projection",
            "swiglu",
            "candidate_down_projection",
            "residual_add",
        ),
    }[cell]


def _canonical_metric_value(value: Any) -> Any:
    """Encode legitimate infinite errors while rejecting ambiguous NaNs."""

    if isinstance(value, float):
        if math.isnan(value):
            raise Lfm2MatrixMlpReplayError("MLP replay metrics cannot contain NaN.")
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
    cell: Lfm2MatrixMlpCell,
    reference: Lfm2MatrixMlpProjectionBundle,
    candidate: Lfm2MatrixMlpProjectionBundle,
    phases: tuple[Lfm2MatrixMlpPhase, ...],
    copy_replays: tuple[Lfm2MatrixMlpPhaseReplay, ...],
    candidate_replays: tuple[Lfm2MatrixMlpPhaseReplay, ...],
    copy_metrics: tuple[Lfm2MatrixMlpPhaseMetrics, ...],
    candidate_metrics: tuple[Lfm2MatrixMlpPhaseMetrics, ...],
    tolerance: Lfm2MatrixMlpReplayTolerance,
    copy_checks: tuple[Lfm2MatrixMlpCopyExactCheck, ...],
) -> str:
    """Seal input authority and every tensor/metric returned by evaluation."""

    digest = hashlib.sha256()
    accounting_evidence = "caller_declared_unverified_by_evaluator"
    for value in (
        LFM2_MATRIX_MLP_REPLAY_SCHEMA,
        cell,
        reference.bundle_id,
        candidate.bundle_id,
        str(candidate.declared_logical_stored_parameter_count),
        str(candidate.declared_physical_materialized_parameter_count),
        candidate.declared_accounting_scope,
        accounting_evidence,
        "parameter_accounting_authority=false",
        "mlp_branch_composable=true",
        "residual_post_layer_composable=true",
        "downstream_trajectory_composable=false",
        repr(_active_components(cell)),
        repr(_composition_path(cell)),
        str(tolerance.absolute),
        str(tolerance.relative),
        str(phases[0].topology.hidden_size),
        str(phases[0].topology.intermediate_size),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")

    phase_tensor_names = (
        "mlp_input",
        "residual_post_operator",
        "gate",
        "up",
        "post_swiglu",
        "output_write",
        "residual_post_layer",
    )
    replay_tensor_names = (
        "gate",
        "up",
        "post_swiglu",
        "output_write",
        "residual_post_layer",
    )
    attribution_tensor_names = (
        "candidate_gate_parent_up_post",
        "parent_gate_candidate_up_post",
        "candidate_gate_candidate_up_post",
        "parent_down_candidate_gate_post",
        "parent_down_candidate_up_post",
        "parent_down_joint_post",
        "candidate_down_parent_post",
        "candidate_down_joint_post",
    )
    for phase in phases:
        digest.update(phase.phase_id.encode("utf-8"))
        digest.update(b"\0")
        for name in phase_tensor_names:
            digest.update(name.encode("ascii"))
            digest.update(
                bytes.fromhex(lfm2_authenticated_tensor_sha256(getattr(phase, name)))
            )
    for lane, replays in (("copy", copy_replays), ("candidate", candidate_replays)):
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


def evaluate_lfm2_matrix_mlp_replay(
    *,
    phases: Sequence[Lfm2MatrixMlpPhase],
    reference: Lfm2MatrixMlpProjectionBundle,
    candidate: Lfm2MatrixMlpProjectionBundle,
    cell: Lfm2MatrixMlpCell,
    tolerance: Lfm2MatrixMlpReplayTolerance | None = None,
) -> Lfm2MatrixMlpReplayEvaluation:
    """Replay one same-topology C11-C13 intervention over sealed MLP phases."""

    if cell not in _SUPPORTED_CELLS:
        raise Lfm2MatrixMlpReplayError(
            f"cell must be one of {sorted(_SUPPORTED_CELLS)}."
        )
    structured_phases = _validate_phases(phases)
    _validate_bundle(reference, structured_phases[0], role="reference")
    _validate_bundle(candidate, structured_phases[0], role="candidate")
    supplied_policy = tolerance or Lfm2MatrixMlpReplayTolerance()
    if not isinstance(supplied_policy, Lfm2MatrixMlpReplayTolerance):
        raise TypeError("tolerance must be Lfm2MatrixMlpReplayTolerance.")
    # Reconstruct the policy so object.__setattr__ cannot bypass the fixed
    # ceilings of the frozen dataclass after construction.
    policy = Lfm2MatrixMlpReplayTolerance(
        absolute=supplied_policy.absolute,
        relative=supplied_policy.relative,
    )

    with torch.no_grad():
        copy_replays = _trajectory_replay(
            phases=structured_phases,
            cell="C13",
            reference=reference,
            candidate=reference,
        )
        copy_checks = _assert_copy_exact(
            phases=structured_phases,
            replays=copy_replays,
            tolerance=policy,
        )
        candidate_replays = _trajectory_replay(
            phases=structured_phases,
            cell=cell,
            reference=reference,
            candidate=candidate,
        )

    copy_metrics = tuple(
        _phase_metrics(phase, replay)
        for phase, replay in zip(structured_phases, copy_replays, strict=True)
    )
    candidate_metrics = tuple(
        _phase_metrics(phase, replay)
        for phase, replay in zip(structured_phases, candidate_replays, strict=True)
    )
    replay_digest = _replay_sha256(
        cell=cell,
        reference=reference,
        candidate=candidate,
        phases=structured_phases,
        copy_replays=copy_replays,
        candidate_replays=candidate_replays,
        copy_metrics=copy_metrics,
        candidate_metrics=candidate_metrics,
        tolerance=policy,
        copy_checks=copy_checks,
    )
    return Lfm2MatrixMlpReplayEvaluation(
        schema=LFM2_MATRIX_MLP_REPLAY_SCHEMA,
        cell=cell,
        reference_bundle_id=reference.bundle_id,
        candidate_bundle_id=candidate.bundle_id,
        active_components=_active_components(cell),
        composition_path=_composition_path(cell),
        phases=candidate_replays,
        phase_metrics=candidate_metrics,
        copy_exact_metrics=copy_metrics,
        copy_exact_tolerance=policy,
        copy_exact_endpoint_checks=copy_checks,
        copy_exact_verified=True,
        mlp_branch_composable=True,
        residual_post_layer_composable=True,
        downstream_trajectory_composable=False,
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
    "LFM2_MATRIX_MLP_REPLAY_SCHEMA",
    "MAX_COPY_EXACT_ABSOLUTE_TOLERANCE",
    "MAX_COPY_EXACT_RELATIVE_TOLERANCE",
    "Lfm2MatrixMlpAttribution",
    "Lfm2MatrixMlpCell",
    "Lfm2MatrixMlpCopyExactCheck",
    "Lfm2MatrixMlpCopyExactError",
    "Lfm2MatrixMlpPhase",
    "Lfm2MatrixMlpPhaseMetrics",
    "Lfm2MatrixMlpPhaseReplay",
    "Lfm2MatrixMlpProjection",
    "Lfm2MatrixMlpProjectionBundle",
    "Lfm2MatrixMlpProjectionError",
    "Lfm2MatrixMlpReplayError",
    "Lfm2MatrixMlpReplayEvaluation",
    "Lfm2MatrixMlpReplayTolerance",
    "Lfm2MatrixMlpTensorMetrics",
    "Lfm2MatrixMlpTopology",
    "Lfm2MatrixMlpTopologyError",
    "evaluate_lfm2_matrix_mlp_replay",
]
