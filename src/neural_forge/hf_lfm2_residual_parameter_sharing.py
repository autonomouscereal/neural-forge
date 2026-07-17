"""Residual, activation-aligned FFN parameter sharing for LFM2.

The ordinary LFM2 sharing primitive replaces two selected FFNs with one
averaged parameter bundle.  That is compact, but it can erase an entire
layer's distinct transformation.  This module instead keeps one FFN in each
activation-selected pair byte-for-byte and represents the other FFN as:

``shared preserved weight + deterministic truncated-SVD residual``.

The coupled neuron permutation from :class:`Lfm2ParameterSharingPlan` is
applied to both SwiGLU producer rows and the down-projection columns.  A
full-rank residual therefore preserves the approximated FFN's function (up to
floating-point roundoff), while a low rank gives a controllable physical
compression.  No optimizer, backward pass, gradient, or training step is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

import neural_forge.hf_lfm2_parameter_sharing as base_sharing
from neural_forge.hf_lfm2_parameter_sharing import (
    DEFAULT_CORRELATION_CHUNK_SIZE,
    DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    Lfm2ActivationEvidence,
    Lfm2FfnPairPlan,
    Lfm2ParameterSharingError,
    Lfm2ParameterSharingPlan,
    Lfm2ParameterSharingTopologyError,
    Lfm2ParameterSharingTransactionError,
)


PLAN_SCHEMA = "neural-forge-lfm2-residual-parameter-sharing-plan-v1"
REPORT_SCHEMA = "neural-forge-lfm2-residual-parameter-sharing-report-v1"
METHOD = "activation-aligned-shared-base-plus-truncated-svd-residual"
_ROLES = ("w1", "w3", "w2")
_ORIENTATIONS = ("earlier", "later")
_TRANSACTION_LOCK = threading.RLock()


class Lfm2ResidualParameterSharingError(Lfm2ParameterSharingError):
    """Base error for residual LFM2 parameter sharing."""


class Lfm2ResidualParameterSharingTopologyError(
    Lfm2ParameterSharingTopologyError,
    Lfm2ResidualParameterSharingError,
):
    """The requested rank, orientation, or LFM2 topology is unsupported."""


class Lfm2ResidualParameterSharingTransactionError(
    Lfm2ParameterSharingTransactionError,
    Lfm2ResidualParameterSharingError,
):
    """Planning, installation, accounting, or rollback failed closed."""


class Lfm2SharedBaseResidualLinear(torch.nn.Module):
    """Bias-free linear operator with a shared base and low-rank residual."""

    def __init__(
        self,
        shared_base: torch.nn.Parameter,
        residual_left: torch.Tensor,
        residual_right: torch.Tensor,
    ) -> None:
        super().__init__()
        if not isinstance(shared_base, torch.nn.Parameter) or shared_base.ndim != 2:
            raise Lfm2ResidualParameterSharingTopologyError(
                "shared_base must be a rank-2 Parameter."
            )
        if (
            not isinstance(residual_left, torch.Tensor)
            or not isinstance(residual_right, torch.Tensor)
            or residual_left.ndim != 2
            or residual_right.ndim != 2
            or residual_left.shape[0] != shared_base.shape[0]
            or residual_right.shape[1] != shared_base.shape[1]
            or residual_left.shape[1] != residual_right.shape[0]
            or residual_left.shape[1] <= 0
        ):
            raise Lfm2ResidualParameterSharingTopologyError(
                "Residual factors do not match the shared base matrix."
            )
        if (
            residual_left.device != shared_base.device
            or residual_right.device != shared_base.device
            or residual_left.dtype != shared_base.dtype
            or residual_right.dtype != shared_base.dtype
            or not bool(torch.isfinite(residual_left).all())
            or not bool(torch.isfinite(residual_right).all())
        ):
            raise Lfm2ResidualParameterSharingTopologyError(
                "Residual factors must be finite and match base device/dtype."
            )
        self.in_features = int(shared_base.shape[1])
        self.out_features = int(shared_base.shape[0])
        self.residual_rank = int(residual_left.shape[1])
        self.weight = shared_base
        self.register_parameter("bias", None)
        requires_grad = bool(shared_base.requires_grad)
        self.residual_left = torch.nn.Parameter(
            residual_left.detach().clone(),
            requires_grad=requires_grad,
        )
        self.residual_right = torch.nn.Parameter(
            residual_right.detach().clone(),
            requires_grad=requires_grad,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight)
        residual = F.linear(F.linear(inputs, self.residual_right), self.residual_left)
        return base + residual

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.residual_rank}, bias=False"
        )


@dataclass(frozen=True, slots=True)
class Lfm2ResidualRolePlan:
    """Low-rank residual factors and honest Frobenius diagnostics for one role."""

    role: str
    matrix_shape: tuple[int, int]
    rank: int
    residual_left: Any
    residual_right: Any
    residual_left_sha256: str
    residual_right_sha256: str
    residual_frobenius_energy: float
    retained_singular_energy: float
    truncation_error_energy: float
    relative_frobenius_error: float

    @property
    def residual_parameters(self) -> int:
        return self.rank * (self.matrix_shape[0] + self.matrix_shape[1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "matrix_shape": list(self.matrix_shape),
            "rank": self.rank,
            "residual_parameters": self.residual_parameters,
            "residual_frobenius_energy": self.residual_frobenius_energy,
            "retained_singular_energy": self.retained_singular_energy,
            "truncation_error_energy": self.truncation_error_energy,
            "relative_frobenius_error": self.relative_frobenius_error,
            "residual_left_sha256": self.residual_left_sha256,
            "residual_right_sha256": self.residual_right_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ResidualFfnPairPlan:
    """One selected pair, its preserved side, and three residual projections."""

    sharing_pair_plan: Lfm2FfnPairPlan
    preserve: Literal["earlier", "later"]
    preserved_layer: int
    approximated_layer: int
    rank: int
    role_plans: tuple[Lfm2ResidualRolePlan, ...]
    parameters_saved: int

    @property
    def layers(self) -> tuple[int, int]:
        return self.sharing_pair_plan.layers

    def role_plan(self, role: str) -> Lfm2ResidualRolePlan:
        for value in self.role_plans:
            if value.role == role:
                return value
        raise Lfm2ResidualParameterSharingTransactionError(
            f"Residual pair plan is missing role {role!r}."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layers": list(self.layers),
            "preserve": self.preserve,
            "preserved_layer": self.preserved_layer,
            "approximated_layer": self.approximated_layer,
            "rank": self.rank,
            "parameters_saved": self.parameters_saved,
            "permutation": [
                int(value)
                for value in self.sharing_pair_plan.permutation.detach()
                .to(device="cpu")
                .tolist()
            ],
            "roles": [value.as_dict() for value in self.role_plans],
        }


@dataclass(frozen=True, slots=True)
class Lfm2ResidualParameterSharingPlan:
    """Sealed residual plan built from the standard activation-selected plan."""

    schema: str
    sharing_plan: Lfm2ParameterSharingPlan
    pair_plans: tuple[Lfm2ResidualFfnPairPlan, ...]
    rank: int
    orientations: tuple[Literal["earlier", "later"], ...]
    logical_parameters_before: int
    unique_parameters_before: int
    projected_logical_parameters_after: int
    projected_unique_parameters_after: int
    projected_parameters_saved: int
    plan_sha256: str
    training_steps: int = 0
    gradients_used: bool = False

    @property
    def selected_pairs(self) -> tuple[tuple[int, int], ...]:
        return self.sharing_plan.selected_pairs

    @property
    def parent_identity_sha256(self) -> str:
        return self.sharing_plan.parent_identity_sha256

    @property
    def evidence_sha256(self) -> str:
        return self.sharing_plan.evidence_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method": METHOD,
            "parent_identity_sha256": self.parent_identity_sha256,
            "evidence_sha256": self.evidence_sha256,
            "selected_pairs": [list(value) for value in self.selected_pairs],
            "rank": self.rank,
            "orientations": list(self.orientations),
            "pairs": [value.as_dict() for value in self.pair_plans],
            "logical_parameters_before": self.logical_parameters_before,
            "unique_parameters_before": self.unique_parameters_before,
            "projected_logical_parameters_after": (
                self.projected_logical_parameters_after
            ),
            "projected_unique_parameters_after": (
                self.projected_unique_parameters_after
            ),
            "projected_parameters_saved": self.projected_parameters_saved,
            "preserves_execution_depth": True,
            "preserves_attention_and_recurrent_operators": True,
            "training_steps": self.training_steps,
            "gradients_used": self.gradients_used,
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ResidualParameterSharingReport:
    """Authenticated physical counts and residual diagnostics after install."""

    schema: str
    parent_identity_before: str
    model_identity_after: str
    evidence_sha256: str
    selected_pairs: tuple[tuple[int, int], ...]
    preserved_layers: tuple[int, ...]
    approximated_layers: tuple[int, ...]
    rank: int
    pair_plans: tuple[Lfm2ResidualFfnPairPlan, ...]
    logical_parameters_before: int
    logical_parameters_after: int
    unique_parameters_before: int
    unique_parameters_after: int
    parameters_saved: int
    shared_base_aliases: int
    topology_preserved: bool
    attention_and_recurrent_operators_preserved: bool
    buffers_preserved: bool
    only_approximated_ffn_projections_replaced: bool
    training_steps: int = 0
    gradients_used: bool = False

    @property
    def compression_fraction(self) -> float:
        if self.unique_parameters_before <= 0:
            return 0.0
        return self.parameters_saved / self.unique_parameters_before

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method": METHOD,
            "parent_identity_before": self.parent_identity_before,
            "model_identity_after": self.model_identity_after,
            "evidence_sha256": self.evidence_sha256,
            "selected_pairs": [list(value) for value in self.selected_pairs],
            "preserved_layers": list(self.preserved_layers),
            "approximated_layers": list(self.approximated_layers),
            "rank": self.rank,
            "pairs": [value.as_dict() for value in self.pair_plans],
            "logical_parameters_before": self.logical_parameters_before,
            "logical_parameters_after": self.logical_parameters_after,
            "unique_parameters_before": self.unique_parameters_before,
            "unique_parameters_after": self.unique_parameters_after,
            "parameters_saved": self.parameters_saved,
            "compression_fraction": self.compression_fraction,
            "shared_base_aliases": self.shared_base_aliases,
            "topology_preserved": self.topology_preserved,
            "execution_depth_preserved": True,
            "attention_and_recurrent_operators_preserved": (
                self.attention_and_recurrent_operators_preserved
            ),
            "buffers_preserved": self.buffers_preserved,
            "only_approximated_ffn_projections_replaced": (
                self.only_approximated_ffn_projections_replaced
            ),
            "training_steps": self.training_steps,
            "gradients_used": self.gradients_used,
        }


@dataclass(frozen=True, slots=True)
class _ModuleReplacement:
    owner: Any
    role: str
    original_module: Any
    installed_module: Lfm2SharedBaseResidualLinear
    shared_base: Any


class InstalledLfm2ResidualParameterSharing:
    """Live residual-sharing installation with exact-object rollback."""

    def __init__(
        self,
        *,
        model: Any,
        report: Lfm2ResidualParameterSharingReport,
        replacements: tuple[_ModuleReplacement, ...],
        buffer_snapshot: Mapping[str, tuple[Any, Any]],
    ) -> None:
        self.model = model
        self.report = report
        self._replacements = replacements
        self._buffer_snapshot = dict(buffer_snapshot)
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def rollback(self) -> bool:
        """Restore exact original Linear modules and their Parameter objects."""

        with _TRANSACTION_LOCK, torch.no_grad():
            if not self._active:
                return False
            for value in self._replacements:
                if getattr(value.owner, value.role) is not value.installed_module:
                    raise Lfm2ResidualParameterSharingTransactionError(
                        "A residual FFN module changed after installation; refusing "
                        "to clobber external state during rollback."
                    )
            for value in self._replacements:
                setattr(value.owner, value.role, value.original_module)
            self._active = False
            if any(
                getattr(value.owner, value.role) is not value.original_module
                for value in self._replacements
            ):
                raise Lfm2ResidualParameterSharingTransactionError(
                    "Exact residual module rollback verification failed."
                )
            base_sharing._verify_buffers_unchanged(  # noqa: SLF001
                torch,
                self.model,
                self._buffer_snapshot,
            )
            if (
                base_sharing._parent_identity(self.model)  # noqa: SLF001
                != self.report.parent_identity_before
            ):
                raise Lfm2ResidualParameterSharingTransactionError(
                    "Rollback restored selected modules but not the exact parent."
                )
            return True

    def __enter__(self) -> InstalledLfm2ResidualParameterSharing:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.rollback()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2ResidualParameterSharingTransactionError(
            "Residual plan metadata is not canonical JSON."
        ) from exc


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-residual-factor-v1\0")
    digest.update(str(tensor.dtype).replace("torch.", "").encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameter_counts(model: Any) -> tuple[int, int]:
    named = tuple(model.named_parameters(remove_duplicate=False))
    logical = sum(int(parameter.numel()) for _, parameter in named)
    unique = {id(parameter): parameter for _, parameter in named}
    return logical, sum(int(parameter.numel()) for parameter in unique.values())


def lfm2_residual_parameter_sharing_savings(
    hidden_size: int,
    intermediate_size: int,
    pair_count: int,
    rank: int,
) -> int:
    """Return exact unique-parameter savings for residual FFN sharing.

    Per pair this removes three ``[I, H]``-equivalent matrices and installs
    six rank factors, for ``3*H*I - 3*(H+I)*rank`` physical parameters.
    """

    for value, name, allow_zero in (
        (hidden_size, "hidden_size", False),
        (intermediate_size, "intermediate_size", False),
        (pair_count, "pair_count", True),
        (rank, "rank", False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise Lfm2ResidualParameterSharingTopologyError(
                f"{name} must be a {qualifier} integer."
            )
    if rank > min(hidden_size, intermediate_size):
        raise Lfm2ResidualParameterSharingTopologyError(
            "rank cannot exceed the full matrix rank."
        )
    per_pair = 3 * hidden_size * intermediate_size
    per_pair -= 3 * (hidden_size + intermediate_size) * rank
    return pair_count * per_pair


def _normalize_orientations(
    preserve: str | Sequence[str],
    pair_count: int,
) -> tuple[Literal["earlier", "later"], ...]:
    if isinstance(preserve, str):
        values = (preserve,) * pair_count
    elif isinstance(preserve, Sequence):
        values = tuple(preserve)
    else:
        values = ()
    if len(values) != pair_count or any(value not in _ORIENTATIONS for value in values):
        raise Lfm2ResidualParameterSharingTopologyError(
            "preserve must be 'earlier', 'later', or one orientation per pair."
        )
    return tuple(values)  # type: ignore[return-value]


def _inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(permutation)
    result[permutation] = torch.arange(
        int(permutation.numel()),
        dtype=permutation.dtype,
        device=permutation.device,
    )
    return result


def _aligned_target_and_base(
    layers: Sequence[Any],
    pair_plan: Lfm2FfnPairPlan,
    preserve: Literal["earlier", "later"],
    role: str,
) -> tuple[torch.nn.Parameter, torch.Tensor, int, int]:
    earlier, later = pair_plan.layers
    permutation = pair_plan.permutation.detach().to(dtype=torch.long)
    if preserve == "earlier":
        preserved, approximated = earlier, later
        alignment = permutation
    else:
        preserved, approximated = later, earlier
        alignment = _inverse_permutation(permutation)
    base = getattr(layers[preserved].feed_forward, role).weight
    target_parameter = getattr(layers[approximated].feed_forward, role).weight
    alignment = alignment.to(device=target_parameter.device)
    dimension = 1 if role == "w2" else 0
    aligned_target = target_parameter.detach().index_select(dimension, alignment)
    if aligned_target.shape != base.shape:
        raise Lfm2ResidualParameterSharingTopologyError(
            f"Aligned {role} target does not match the preserved base."
        )
    return base, aligned_target, preserved, approximated


def _truncated_svd_role_plan(
    role: str,
    base: torch.nn.Parameter,
    aligned_target: torch.Tensor,
    rank: int,
) -> Lfm2ResidualRolePlan:
    compute_dtype = torch.float64 if base.dtype == torch.float64 else torch.float32
    delta = aligned_target.detach().to(dtype=compute_dtype) - base.detach().to(
        dtype=compute_dtype
    )
    if not bool(torch.isfinite(delta).all()):
        raise Lfm2ResidualParameterSharingTransactionError(
            f"{role} residual contains NaN or infinity."
        )
    try:
        left_vectors, singular_values, right_vectors = torch.linalg.svd(
            delta,
            full_matrices=False,
        )
    except Exception as exc:
        raise Lfm2ResidualParameterSharingTransactionError(
            f"Deterministic SVD failed for {role}."
        ) from exc
    if not all(
        bool(torch.isfinite(value).all())
        for value in (left_vectors, singular_values, right_vectors)
    ):
        raise Lfm2ResidualParameterSharingTransactionError(
            f"Deterministic SVD produced non-finite {role} factors."
        )
    if rank > int(singular_values.numel()):
        raise Lfm2ResidualParameterSharingTopologyError(
            f"rank exceeds the full rank of {role}."
        )

    left_vectors = left_vectors[:, :rank].clone()
    singular_values = singular_values[:rank].clone()
    right_vectors = right_vectors[:rank, :].clone()
    # SVD signs are algebraically arbitrary.  Canonicalize each component so
    # repeated planning on the same backend produces byte-stable factors.
    maxima = left_vectors.abs().argmax(dim=0)
    columns = torch.arange(rank, device=left_vectors.device)
    pivots = left_vectors[maxima, columns]
    signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
    left_vectors.mul_(signs.unsqueeze(0))
    right_vectors.mul_(signs.unsqueeze(1))

    roots = singular_values.clamp_min(0).sqrt()
    left = left_vectors * roots.unsqueeze(0)
    right = roots.unsqueeze(1) * right_vectors
    left = left.to(device=base.device, dtype=base.dtype).detach().contiguous()
    right = right.to(device=base.device, dtype=base.dtype).detach().contiguous()
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise Lfm2ResidualParameterSharingTransactionError(
            f"Converted {role} residual factors are non-finite."
        )

    runtime_residual = left.to(dtype=compute_dtype) @ right.to(dtype=compute_dtype)
    energy = float(delta.square().sum().item())
    retained = float(singular_values.square().sum().item())
    error_energy = float((delta - runtime_residual).square().sum().item())
    if not all(math.isfinite(value) and value >= 0 for value in (energy, retained, error_energy)):
        raise Lfm2ResidualParameterSharingTransactionError(
            f"{role} residual energy diagnostics are invalid."
        )
    relative_error = math.sqrt(error_energy / energy) if energy > 0 else 0.0
    return Lfm2ResidualRolePlan(
        role=role,
        matrix_shape=(int(base.shape[0]), int(base.shape[1])),
        rank=rank,
        residual_left=left,
        residual_right=right,
        residual_left_sha256=_tensor_sha256(left),
        residual_right_sha256=_tensor_sha256(right),
        residual_frobenius_energy=energy,
        retained_singular_energy=retained,
        truncation_error_energy=error_energy,
        relative_frobenius_error=relative_error,
    )


def _plan_body(plan: Lfm2ResidualParameterSharingPlan) -> dict[str, Any]:
    return {
        key: value
        for key, value in plan.as_dict().items()
        if key != "plan_sha256"
    }


def _seal_plan(plan: Lfm2ResidualParameterSharingPlan) -> Lfm2ResidualParameterSharingPlan:
    digest = hashlib.sha256(_canonical_json(_plan_body(plan))).hexdigest()
    return Lfm2ResidualParameterSharingPlan(
        schema=plan.schema,
        sharing_plan=plan.sharing_plan,
        pair_plans=plan.pair_plans,
        rank=plan.rank,
        orientations=plan.orientations,
        logical_parameters_before=plan.logical_parameters_before,
        unique_parameters_before=plan.unique_parameters_before,
        projected_logical_parameters_after=plan.projected_logical_parameters_after,
        projected_unique_parameters_after=plan.projected_unique_parameters_after,
        projected_parameters_saved=plan.projected_parameters_saved,
        plan_sha256=digest,
    )


def _validate_plan_seal(plan: Lfm2ResidualParameterSharingPlan) -> None:
    for pair in plan.pair_plans:
        if tuple(value.role for value in pair.role_plans) != _ROLES:
            raise Lfm2ResidualParameterSharingTransactionError(
                "Residual pair plan does not contain exactly w1/w3/w2."
            )
        for role in pair.role_plans:
            if (
                _tensor_sha256(role.residual_left) != role.residual_left_sha256
                or _tensor_sha256(role.residual_right) != role.residual_right_sha256
            ):
                raise Lfm2ResidualParameterSharingTransactionError(
                    "Residual factors changed after planning."
                )
    expected = hashlib.sha256(_canonical_json(_plan_body(plan))).hexdigest()
    if plan.plan_sha256 != expected:
        raise Lfm2ResidualParameterSharingTransactionError(
            "Residual plan metadata changed after sealing."
        )


def plan_lfm2_residual_parameter_sharing(
    model: Any,
    evidence: Lfm2ActivationEvidence,
    *,
    pair_count: int,
    rank: int,
    preserve: str | Sequence[str] = "earlier",
    layer_pairs: Sequence[Sequence[int]] | None = None,
    pair_score_weights: Sequence[float] = (0.5, 0.5),
    exact_assignment_limit: int = DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    chunk_size: int = DEFAULT_CORRELATION_CHUNK_SIZE,
    variance_floor: float = 0.0,
) -> Lfm2ResidualParameterSharingPlan:
    """Select pairs with the standard planner and stage deterministic residuals."""

    with _TRANSACTION_LOCK, torch.no_grad():
        sharing_plan = base_sharing.plan_lfm2_activation_aligned_parameter_sharing(
            model,
            evidence,
            pair_count=pair_count,
            layer_pairs=layer_pairs,
            pair_score_weights=pair_score_weights,
            merge_weights=(1.0, 1.0),
            exact_assignment_limit=exact_assignment_limit,
            chunk_size=chunk_size,
            variance_floor=variance_floor,
        )
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
            or rank > min(sharing_plan.hidden_size, sharing_plan.intermediate_size)
        ):
            raise Lfm2ResidualParameterSharingTopologyError(
                "rank must be positive and cannot exceed the projection rank."
            )
        orientations = _normalize_orientations(preserve, pair_count)
        layers, _ = base_sharing._validate_evidence(model, evidence)  # noqa: SLF001
        pair_plans: list[Lfm2ResidualFfnPairPlan] = []
        for sharing_pair, orientation in zip(
            sharing_plan.pair_plans,
            orientations,
        ):
            role_plans: list[Lfm2ResidualRolePlan] = []
            preserved_layer: int | None = None
            approximated_layer: int | None = None
            for role in _ROLES:
                base, target, preserved, approximated = _aligned_target_and_base(
                    layers,
                    sharing_pair,
                    orientation,
                    role,
                )
                if preserved_layer is None:
                    preserved_layer = preserved
                    approximated_layer = approximated
                elif (
                    preserved_layer != preserved
                    or approximated_layer != approximated
                ):
                    raise Lfm2ResidualParameterSharingTransactionError(
                        "Coupled FFN roles disagree on pair orientation."
                    )
                role_plans.append(
                    _truncated_svd_role_plan(role, base, target, rank)
                )
            if preserved_layer is None or approximated_layer is None:
                raise Lfm2ResidualParameterSharingTransactionError(
                    "Residual pair planning produced no coupled FFN roles."
                )
            parameters_saved = lfm2_residual_parameter_sharing_savings(
                sharing_plan.hidden_size,
                sharing_plan.intermediate_size,
                1,
                rank,
            )
            pair_plans.append(
                Lfm2ResidualFfnPairPlan(
                    sharing_pair_plan=sharing_pair,
                    preserve=orientation,
                    preserved_layer=preserved_layer,
                    approximated_layer=approximated_layer,
                    rank=rank,
                    role_plans=tuple(role_plans),
                    parameters_saved=parameters_saved,
                )
            )
        logical_before, unique_before = _parameter_counts(model)
        residual_parameters = sum(
            role.residual_parameters
            for pair in pair_plans
            for role in pair.role_plans
        )
        total_savings = sum(pair.parameters_saved for pair in pair_plans)
        plan = Lfm2ResidualParameterSharingPlan(
            schema=PLAN_SCHEMA,
            sharing_plan=sharing_plan,
            pair_plans=tuple(pair_plans),
            rank=rank,
            orientations=orientations,
            logical_parameters_before=logical_before,
            unique_parameters_before=unique_before,
            projected_logical_parameters_after=logical_before + residual_parameters,
            projected_unique_parameters_after=unique_before - total_savings,
            projected_parameters_saved=total_savings,
            plan_sha256="",
        )
        if plan.projected_unique_parameters_after <= 0:
            raise Lfm2ResidualParameterSharingTransactionError(
                "Projected unique parameter count is invalid."
            )
        return _seal_plan(plan)


def _build_residual_linear(
    shared_base: torch.nn.Parameter,
    role_plan: Lfm2ResidualRolePlan,
) -> Lfm2SharedBaseResidualLinear:
    return Lfm2SharedBaseResidualLinear(
        shared_base,
        role_plan.residual_left,
        role_plan.residual_right,
    )


def _replace_projection(owner: Any, role: str, module: Any) -> None:
    setattr(owner, role, module)


def _verify_parameter_namespace(
    model: Any,
    parameters_before: Mapping[str, Any],
    replaced_parameter_ids: set[int],
    installed_modules: Sequence[Lfm2SharedBaseResidualLinear],
) -> None:
    parameters_after = dict(model.named_parameters(remove_duplicate=False))
    if not set(parameters_before).issubset(parameters_after):
        raise Lfm2ResidualParameterSharingTransactionError(
            "Residual installation removed an unexpected Parameter name."
        )
    for name, parameter in parameters_before.items():
        if id(parameter) not in replaced_parameter_ids and parameters_after[name] is not parameter:
            raise Lfm2ResidualParameterSharingTransactionError(
                f"Residual installation changed unselected Parameter {name!r}."
            )
    installed_factor_ids = {
        id(value)
        for module in installed_modules
        for value in (module.residual_left, module.residual_right)
    }
    new_names = set(parameters_after) - set(parameters_before)
    new_parameter_ids = {id(parameters_after[name]) for name in new_names}
    if (
        len(new_names) != 2 * len(installed_modules)
        or new_parameter_ids != installed_factor_ids
        or any(
            not name.endswith((".residual_left", ".residual_right"))
            for name in new_names
        )
    ):
        raise Lfm2ResidualParameterSharingTransactionError(
            "Residual installation added an unexpected Parameter."
        )


def install_lfm2_residual_parameter_sharing_plan(
    model: Any,
    evidence: Lfm2ActivationEvidence,
    plan: Lfm2ResidualParameterSharingPlan,
) -> InstalledLfm2ResidualParameterSharing:
    """Atomically install a sealed residual-sharing plan."""

    with _TRANSACTION_LOCK, torch.no_grad():
        layers, contracts = base_sharing._validate_evidence(  # noqa: SLF001
            model,
            evidence,
        )
        if not isinstance(plan, Lfm2ResidualParameterSharingPlan):
            raise Lfm2ResidualParameterSharingTransactionError(
                "A sealed LFM2 residual-sharing plan is required."
            )
        if plan.schema != PLAN_SCHEMA:
            raise Lfm2ResidualParameterSharingTransactionError(
                "Residual plan schema is invalid."
            )
        _validate_plan_seal(plan)
        sharing_plan = plan.sharing_plan
        if (
            not isinstance(sharing_plan, Lfm2ParameterSharingPlan)
            or plan.parent_identity_sha256 != evidence.parent_identity_sha256
            or plan.evidence_sha256 != evidence.evidence_sha256
            or len(plan.pair_plans) != sharing_plan.requested_pair_count
            or tuple(pair.layers for pair in plan.pair_plans)
            != sharing_plan.selected_pairs
            or plan.orientations != tuple(pair.preserve for pair in plan.pair_plans)
            or sharing_plan.layer_count != len(layers)
            or sharing_plan.hidden_size != contracts[0].hidden_size
            or sharing_plan.intermediate_size
            != contracts[0].ffn_intermediate_size
        ):
            raise Lfm2ResidualParameterSharingTransactionError(
                "Residual plan is stale or does not match model/evidence."
            )
        used_layers = [
            index
            for pair in plan.selected_pairs
            for index in pair
        ]
        if len(set(used_layers)) != 2 * len(plan.selected_pairs):
            raise Lfm2ResidualParameterSharingTopologyError(
                "Residual pair plan contains overlapping layers."
            )
        logical_before, unique_before = _parameter_counts(model)
        if (
            logical_before != plan.logical_parameters_before
            or unique_before != plan.unique_parameters_before
        ):
            raise Lfm2ResidualParameterSharingTransactionError(
                "Model Parameter counts changed after residual planning."
            )

        parameters_before = dict(model.named_parameters(remove_duplicate=False))
        buffers_before = base_sharing._buffer_snapshot(torch, model)  # noqa: SLF001
        structure_before = base_sharing._structure_snapshot(layers)  # noqa: SLF001
        replacements: list[_ModuleReplacement] = []
        replaced_parameter_ids: set[int] = set()
        installed_modules: list[Lfm2SharedBaseResidualLinear] = []
        for pair in plan.pair_plans:
            preserved_ffn = layers[pair.preserved_layer].feed_forward
            approximated_ffn = layers[pair.approximated_layer].feed_forward
            for role in _ROLES:
                preserved_module = getattr(preserved_ffn, role)
                original_module = getattr(approximated_ffn, role)
                original_weight = original_module.weight
                role_plan = pair.role_plan(role)
                if (
                    not isinstance(original_module, torch.nn.Linear)
                    or original_module.bias is not None
                    or role_plan.rank != plan.rank
                    or tuple(role_plan.matrix_shape)
                    != tuple(preserved_module.weight.shape)
                ):
                    raise Lfm2ResidualParameterSharingTopologyError(
                        "Selected approximated projection is not an independent "
                        "bias-free Linear matching its residual plan."
                    )
                installed = _build_residual_linear(
                    preserved_module.weight,
                    role_plan,
                )
                replacements.append(
                    _ModuleReplacement(
                        owner=approximated_ffn,
                        role=role,
                        original_module=original_module,
                        installed_module=installed,
                        shared_base=preserved_module.weight,
                    )
                )
                installed_modules.append(installed)
                replaced_parameter_ids.add(id(original_weight))

        try:
            for value in replacements:
                _replace_projection(
                    value.owner,
                    value.role,
                    value.installed_module,
                )
            post_layers, _ = base_sharing._inspect_model(  # noqa: SLF001
                model,
                reject_aliasing=False,
            )
            if (
                base_sharing._structure_snapshot(post_layers)  # noqa: SLF001
                != structure_before
            ):
                raise Lfm2ResidualParameterSharingTransactionError(
                    "Decoder depth, FFN shell, or attention/recurrent operator changed."
                )
            for value in replacements:
                if (
                    getattr(value.owner, value.role) is not value.installed_module
                    or value.installed_module.weight is not value.shared_base
                ):
                    raise Lfm2ResidualParameterSharingTransactionError(
                        "Installed residual module does not share its preserved base."
                    )
            _verify_parameter_namespace(
                model,
                parameters_before,
                replaced_parameter_ids,
                installed_modules,
            )
            base_sharing._verify_buffers_unchanged(  # noqa: SLF001
                torch,
                model,
                buffers_before,
            )
            logical_after, unique_after = _parameter_counts(model)
            if (
                logical_after != plan.projected_logical_parameters_after
                or unique_after != plan.projected_unique_parameters_after
                or unique_before - unique_after != plan.projected_parameters_saved
            ):
                raise Lfm2ResidualParameterSharingTransactionError(
                    "Residual physical/logical Parameter accounting failed."
                )
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise Lfm2ResidualParameterSharingTransactionError(
                    "A gradient appeared during zero-training residual installation."
                )
            report = Lfm2ResidualParameterSharingReport(
                schema=REPORT_SCHEMA,
                parent_identity_before=evidence.parent_identity_sha256,
                model_identity_after=base_sharing._parent_identity(  # noqa: SLF001
                    model
                ),
                evidence_sha256=evidence.evidence_sha256,
                selected_pairs=plan.selected_pairs,
                preserved_layers=tuple(
                    pair.preserved_layer for pair in plan.pair_plans
                ),
                approximated_layers=tuple(
                    pair.approximated_layer for pair in plan.pair_plans
                ),
                rank=plan.rank,
                pair_plans=plan.pair_plans,
                logical_parameters_before=logical_before,
                logical_parameters_after=logical_after,
                unique_parameters_before=unique_before,
                unique_parameters_after=unique_after,
                parameters_saved=unique_before - unique_after,
                shared_base_aliases=3 * len(plan.pair_plans),
                topology_preserved=True,
                attention_and_recurrent_operators_preserved=True,
                buffers_preserved=True,
                only_approximated_ffn_projections_replaced=True,
            )
            return InstalledLfm2ResidualParameterSharing(
                model=model,
                report=report,
                replacements=tuple(replacements),
                buffer_snapshot=buffers_before,
            )
        except Exception as exc:
            for value in replacements:
                setattr(value.owner, value.role, value.original_module)
            try:
                base_sharing._verify_buffers_unchanged(  # noqa: SLF001
                    torch,
                    model,
                    buffers_before,
                )
                restored = base_sharing._parent_identity(model)  # noqa: SLF001
            except Exception as rollback_exc:
                raise Lfm2ResidualParameterSharingTransactionError(
                    "Residual installation failed and rollback could not be "
                    "authenticated."
                ) from rollback_exc
            if restored != evidence.parent_identity_sha256:
                raise Lfm2ResidualParameterSharingTransactionError(
                    "Residual installation failed and rollback did not restore "
                    "the parent."
                ) from exc
            raise


def install_lfm2_residual_parameter_sharing(
    model: Any,
    evidence: Lfm2ActivationEvidence,
    *,
    pair_count: int,
    rank: int,
    preserve: str | Sequence[str] = "earlier",
    layer_pairs: Sequence[Sequence[int]] | None = None,
    pair_score_weights: Sequence[float] = (0.5, 0.5),
    exact_assignment_limit: int = DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    chunk_size: int = DEFAULT_CORRELATION_CHUNK_SIZE,
    variance_floor: float = 0.0,
) -> InstalledLfm2ResidualParameterSharing:
    """Plan and atomically install residual activation-aligned sharing."""

    plan = plan_lfm2_residual_parameter_sharing(
        model,
        evidence,
        pair_count=pair_count,
        rank=rank,
        preserve=preserve,
        layer_pairs=layer_pairs,
        pair_score_weights=pair_score_weights,
        exact_assignment_limit=exact_assignment_limit,
        chunk_size=chunk_size,
        variance_floor=variance_floor,
    )
    return install_lfm2_residual_parameter_sharing_plan(model, evidence, plan)


build_lfm2_residual_parameter_sharing_plan = (
    plan_lfm2_residual_parameter_sharing
)


__all__ = [
    "InstalledLfm2ResidualParameterSharing",
    "Lfm2ResidualFfnPairPlan",
    "Lfm2ResidualParameterSharingError",
    "Lfm2ResidualParameterSharingPlan",
    "Lfm2ResidualParameterSharingReport",
    "Lfm2ResidualParameterSharingTopologyError",
    "Lfm2ResidualParameterSharingTransactionError",
    "Lfm2ResidualRolePlan",
    "Lfm2SharedBaseResidualLinear",
    "METHOD",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "build_lfm2_residual_parameter_sharing_plan",
    "install_lfm2_residual_parameter_sharing",
    "install_lfm2_residual_parameter_sharing_plan",
    "lfm2_residual_parameter_sharing_savings",
    "plan_lfm2_residual_parameter_sharing",
]
