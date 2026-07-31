"""Catalog-planned launch frontiers for zero-training LFM2 CPCF-v5 folds.

This module is deliberately upstream of model loading and compilation.  It
turns one authenticated, inspected LFM2 graph and one sealed 4,096-row token
bridge into independent exact dense target plans.  Candidate rank is selected
by the registered catalog's executable supported-Pareto policy; callers cannot
force a rank through this surface.

Two lifecycle states are distinct:

``capture_quarantine``
    Produces non-promotable candidates for evidence capture.  It requires no
    candidate-output or manual-review evidence.

``reviewed_rerun``
    Replays an already reviewed candidate.  It is the only lifecycle in which
    later behavioral promotion may be attempted, and it requires explicit
    plan-bound state, output, and manual-review approvals.

Planning never executes a parent, target, GPU, compiler, or remote command.  It
does not train, prune, quantize, pack, sparsify, or factorize parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import ModelCircuitGraph, canonical_sha256
from .dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from .dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCandidate,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
)
from .dynamic_target_synthesis import (
    BudgetedDynamicTargetPlan,
    plan_budgeted_dynamic_target,
)
from .hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
    StrictTokenizerBridgeArtifact,
)


SCHEMA = "neural-forge-lfm2-cpcf-v5-frontier-v1"
STRICT_TARGET_VOCABULARY_ROWS = 4_096
DEFAULT_COMPRESSION_RATIOS = (4, 10, 25, 50, 100, 250)
V5_LANE_INTERNAL_GPU_COUNT = 2
_AUTOMATIC_SELECTION_POLICY = "supported_pareto_balanced_executable_v1"


class Lfm2CpcfV5FrontierError(ValueError):
    """The CPCF-v5 launch frontier is unauthenticated or inconsistent."""


class Lfm2CpcfV5FrontierLifecycle(str, Enum):
    """Evidence and promotion lifecycle for one planned frontier."""

    CAPTURE_QUARANTINE = "capture_quarantine"
    REVIEWED_RERUN = "reviewed_rerun"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CpcfV5FrontierError(f"{name} must be nonempty text.")
    return value.strip()


def _digest(value: Any, name: str) -> str:
    result = _text(value, name).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise Lfm2CpcfV5FrontierError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfV5FrontierError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfV5FrontierError(
            f"{name} must be a non-negative integer."
        )
    return value


def _compression_ratio(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise Lfm2CpcfV5FrontierError(
            "compression ratios must be finite numbers greater than one."
        )
    try:
        ratio = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise Lfm2CpcfV5FrontierError(
            "compression ratios must be finite numbers greater than one."
        ) from exc
    if ratio <= 1:
        raise Lfm2CpcfV5FrontierError(
            "compression ratios must be greater than one."
        )
    return ratio


def _ratio_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _ratio_slug(value: Fraction) -> str:
    if value.denominator == 1:
        return f"{value.numerator}x"
    return f"{value.numerator}-over-{value.denominator}x"


def _typed_mapping(
    values: Mapping[str, Any] | None,
    *,
    value_type: type[Any],
    name: str,
) -> Mapping[str, Any]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise Lfm2CpcfV5FrontierError(f"{name} must be a mapping.")
    result: dict[str, Any] = {}
    for raw_key, value in values.items():
        key = _digest(raw_key, f"{name} key")
        if type(value) is not value_type:
            raise Lfm2CpcfV5FrontierError(
                f"{name}[{key!r}] must be {value_type.__name__}."
            )
        if key in result:
            raise Lfm2CpcfV5FrontierError(f"{name} contains a duplicate key.")
        result[key] = value
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5ExactCompression:
    """Requested and realized ratios represented without floating-point loss."""

    requested_numerator: int
    requested_denominator: int
    parent_unique_parameters: int
    target_unique_parameters: int
    realized_numerator: int
    realized_denominator: int

    def __post_init__(self) -> None:
        for name in (
            "requested_numerator",
            "requested_denominator",
            "parent_unique_parameters",
            "target_unique_parameters",
            "realized_numerator",
            "realized_denominator",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        requested = Fraction(
            self.requested_numerator,
            self.requested_denominator,
        )
        realized = Fraction(
            self.parent_unique_parameters,
            self.target_unique_parameters,
        )
        if requested <= 1:
            raise Lfm2CpcfV5FrontierError(
                "requested compression ratio must be greater than one."
            )
        if self.target_unique_parameters >= self.parent_unique_parameters:
            raise Lfm2CpcfV5FrontierError(
                "target_unique_parameters must be genuinely smaller."
            )
        if (
            requested.numerator != self.requested_numerator
            or requested.denominator != self.requested_denominator
            or realized.numerator != self.realized_numerator
            or realized.denominator != self.realized_denominator
        ):
            raise Lfm2CpcfV5FrontierError(
                "Compression ratios must be exact reduced fractions derived "
                "from the target plan."
            )
        if realized < requested:
            raise Lfm2CpcfV5FrontierError(
                "The selected at-or-below target does not achieve its requested "
                "compression ratio."
            )

    @classmethod
    def from_plan(
        cls,
        *,
        requested_ratio: Fraction,
        parent_unique_parameters: int,
        target_unique_parameters: int,
    ) -> Lfm2CpcfV5ExactCompression:
        requested = _compression_ratio(requested_ratio)
        realized = Fraction(
            _positive_int(
                parent_unique_parameters,
                "parent_unique_parameters",
            ),
            _positive_int(
                target_unique_parameters,
                "target_unique_parameters",
            ),
        )
        return cls(
            requested_numerator=requested.numerator,
            requested_denominator=requested.denominator,
            parent_unique_parameters=parent_unique_parameters,
            target_unique_parameters=target_unique_parameters,
            realized_numerator=realized.numerator,
            realized_denominator=realized.denominator,
        )

    @property
    def requested_ratio(self) -> Fraction:
        return Fraction(
            self.requested_numerator,
            self.requested_denominator,
        )

    @property
    def realized_ratio(self) -> Fraction:
        return Fraction(
            self.realized_numerator,
            self.realized_denominator,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_ratio": {
                "numerator": self.requested_numerator,
                "denominator": self.requested_denominator,
                "text": _ratio_text(self.requested_ratio),
            },
            "parent_unique_parameters": self.parent_unique_parameters,
            "target_unique_parameters": self.target_unique_parameters,
            "realized_ratio": {
                "numerator": self.realized_numerator,
                "denominator": self.realized_denominator,
                "float_diagnostic": float(self.realized_ratio),
            },
            "exact_plan_derived_arithmetic": True,
            "ratio_admission_threshold_used": False,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5ApprovedReviewBindings:
    """Independent approvals required before a rerun may seek promotion."""

    target_plan_fingerprint: str
    approved_candidate_state_sha256: str
    approved_candidate_outputs_sha256: str
    approved_manual_review_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "target_plan_fingerprint",
            "approved_candidate_state_sha256",
            "approved_candidate_outputs_sha256",
            "approved_manual_review_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "approved_review_bindings",
            "target_plan_fingerprint": self.target_plan_fingerprint,
            "approved_candidate_state_sha256": (
                self.approved_candidate_state_sha256
            ),
            "approved_candidate_outputs_sha256": (
                self.approved_candidate_outputs_sha256
            ),
            "approved_manual_review_sha256": (
                self.approved_manual_review_sha256
            ),
            "approved_bindings_are_behavioral_evidence": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5ResumeReference:
    """One immutable resume admission bound to exactly one target plan."""

    target_plan_fingerprint: str
    resume_admission_sha256: str
    resume_artifact_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "target_plan_fingerprint",
            "resume_admission_sha256",
            "resume_artifact_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "resume_reference",
            "target_plan_fingerprint": self.target_plan_fingerprint,
            "resume_admission_sha256": self.resume_admission_sha256,
            "resume_artifact_sha256": self.resume_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5LaunchIdentity:
    """Disjoint namespaces derived solely from an authenticated plan."""

    target_plan_fingerprint: str
    candidate_namespace_sha256: str
    resume_namespace_sha256: str
    quarantine_namespace_sha256: str

    def __post_init__(self) -> None:
        plan = _digest(
            self.target_plan_fingerprint,
            "target_plan_fingerprint",
        )
        object.__setattr__(self, "target_plan_fingerprint", plan)
        expected = {
            "candidate_namespace_sha256": canonical_sha256(
                {
                    "schema": SCHEMA,
                    "namespace": "candidate",
                    "target_plan_fingerprint": plan,
                }
            ),
            "resume_namespace_sha256": canonical_sha256(
                {
                    "schema": SCHEMA,
                    "namespace": "resume",
                    "target_plan_fingerprint": plan,
                }
            ),
            "quarantine_namespace_sha256": canonical_sha256(
                {
                    "schema": SCHEMA,
                    "namespace": "quarantine",
                    "target_plan_fingerprint": plan,
                }
            ),
        }
        for name, expected_digest in expected.items():
            observed = _digest(getattr(self, name), name)
            object.__setattr__(self, name, observed)
            if observed != expected_digest:
                raise Lfm2CpcfV5FrontierError(
                    f"{name} is not derived from the exact target plan."
                )
        if len(set(expected.values())) != len(expected):
            raise Lfm2CpcfV5FrontierError(
                "Candidate, resume, and quarantine namespaces must be disjoint."
            )

    @classmethod
    def from_plan_fingerprint(
        cls,
        target_plan_fingerprint: str,
    ) -> Lfm2CpcfV5LaunchIdentity:
        plan = _digest(
            target_plan_fingerprint,
            "target_plan_fingerprint",
        )

        def namespace(label: str) -> str:
            return canonical_sha256(
                {
                    "schema": SCHEMA,
                    "namespace": label,
                    "target_plan_fingerprint": plan,
                }
            )

        return cls(
            target_plan_fingerprint=plan,
            candidate_namespace_sha256=namespace("candidate"),
            resume_namespace_sha256=namespace("resume"),
            quarantine_namespace_sha256=namespace("quarantine"),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "launch_identity",
            "target_plan_fingerprint": self.target_plan_fingerprint,
            "candidate_namespace_sha256": self.candidate_namespace_sha256,
            "resume_namespace_sha256": self.resume_namespace_sha256,
            "quarantine_namespace_sha256": (
                self.quarantine_namespace_sha256
            ),
            "all_namespaces_plan_fingerprint_bound": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5LifecycleContract:
    """Promotion authority for one plan-bound launch lane."""

    lifecycle: Lfm2CpcfV5FrontierLifecycle
    target_plan_fingerprint: str
    approved_review_bindings: Lfm2CpcfV5ApprovedReviewBindings | None = None

    def __post_init__(self) -> None:
        if type(self.lifecycle) is not Lfm2CpcfV5FrontierLifecycle:
            raise Lfm2CpcfV5FrontierError(
                "lifecycle must be Lfm2CpcfV5FrontierLifecycle."
            )
        plan = _digest(
            self.target_plan_fingerprint,
            "target_plan_fingerprint",
        )
        object.__setattr__(self, "target_plan_fingerprint", plan)
        bindings = self.approved_review_bindings
        if self.lifecycle is Lfm2CpcfV5FrontierLifecycle.CAPTURE_QUARANTINE:
            if bindings is not None:
                raise Lfm2CpcfV5FrontierError(
                    "capture_quarantine requires no reviewed bindings and "
                    "cannot be promotable."
                )
        elif (
            type(bindings) is not Lfm2CpcfV5ApprovedReviewBindings
            or bindings.target_plan_fingerprint != plan
        ):
            raise Lfm2CpcfV5FrontierError(
                "reviewed_rerun requires state, output, and manual-review "
                "approvals bound to the exact target plan."
            )

    @property
    def promotion_permitted_after_behavior_gate(self) -> bool:
        return self.lifecycle is Lfm2CpcfV5FrontierLifecycle.REVIEWED_RERUN

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "lifecycle_contract",
            "lifecycle": self.lifecycle.value,
            "target_plan_fingerprint": self.target_plan_fingerprint,
            "approved_review_bindings": (
                None
                if self.approved_review_bindings is None
                else self.approved_review_bindings.as_dict()
            ),
            "promotion_permitted_after_behavior_gate": (
                self.promotion_permitted_after_behavior_gate
            ),
            "behavioral_acceptance_still_required": True,
            "capture_quarantine_is_non_promotable": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5LaunchLane:
    """One exact dense target and its non-executing launch contract."""

    request_index: int
    target_plan: BudgetedDynamicTargetPlan
    selected_candidate: DenseTargetTopologyCandidate
    exact_compression: Lfm2CpcfV5ExactCompression
    identity: Lfm2CpcfV5LaunchIdentity
    lifecycle_contract: Lfm2CpcfV5LifecycleContract
    resume_reference: Lfm2CpcfV5ResumeReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_index",
            _nonnegative_int(self.request_index, "request_index"),
        )
        if type(self.target_plan) is not BudgetedDynamicTargetPlan:
            raise Lfm2CpcfV5FrontierError(
                "target_plan must be BudgetedDynamicTargetPlan."
            )
        if type(self.selected_candidate) is not DenseTargetTopologyCandidate:
            raise Lfm2CpcfV5FrontierError(
                "selected_candidate must be DenseTargetTopologyCandidate."
            )
        if type(self.exact_compression) is not Lfm2CpcfV5ExactCompression:
            raise Lfm2CpcfV5FrontierError(
                "exact_compression must be Lfm2CpcfV5ExactCompression."
            )
        if type(self.identity) is not Lfm2CpcfV5LaunchIdentity:
            raise Lfm2CpcfV5FrontierError(
                "identity must be Lfm2CpcfV5LaunchIdentity."
            )
        if type(self.lifecycle_contract) is not Lfm2CpcfV5LifecycleContract:
            raise Lfm2CpcfV5FrontierError(
                "lifecycle_contract must be Lfm2CpcfV5LifecycleContract."
            )
        rank = self.target_plan.target_certificate.selected_candidate_rank
        if rank >= len(self.target_plan.frontier.candidates):
            raise Lfm2CpcfV5FrontierError(
                "Selected target rank is outside the authenticated frontier."
            )
        selected = self.target_plan.frontier.candidates[rank]
        configuration = dict(selected.configuration)
        forbidden = (
            selected.quantized,
            selected.pruned,
            selected.sparse_storage,
            selected.packed,
            selected.factorized,
            selected.codebook_backed,
            selected.non_native_weight_sharing,
        )
        if (
            selected != self.selected_candidate
            or self.target_plan.target_certificate.selection_policy
            != _AUTOMATIC_SELECTION_POLICY
            or not selected.compiler_executable
            or selected.missing_compiler_capabilities
            or not selected.dense_parameters
            or any(forbidden)
            or self.target_plan.target_request.exact_unique_parameters
            != selected.unique_parameters
            or self.exact_compression.parent_unique_parameters
            != selected.source_unique_parameters
            or self.exact_compression.target_unique_parameters
            != selected.unique_parameters
            or configuration.get("vocab_size")
            != STRICT_TARGET_VOCABULARY_ROWS
            or configuration.get(
                "_neural_forge_tokenizer_bridge_required"
            )
            is not True
            or self.identity.target_plan_fingerprint
            != self.target_plan.fingerprint
            or self.lifecycle_contract.target_plan_fingerprint
            != self.target_plan.fingerprint
        ):
            raise Lfm2CpcfV5FrontierError(
                "Launch lane is not the automatic exact dense executable "
                "catalog plan it claims to represent."
            )
        if self.resume_reference is not None:
            authenticate_lfm2_cpcf_v5_resume(
                self,
                self.resume_reference,
            )

    @property
    def target_plan_fingerprint(self) -> str:
        return self.target_plan.fingerprint

    @property
    def selected_candidate_rank(self) -> int:
        return self.target_plan.target_certificate.selected_candidate_rank

    @property
    def promotion_permitted_after_behavior_gate(self) -> bool:
        return (
            self.lifecycle_contract.promotion_permitted_after_behavior_gate
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "launch_lane",
            "request_index": self.request_index,
            "target_plan": self.target_plan.as_dict(),
            "target_plan_fingerprint": self.target_plan_fingerprint,
            "selected_candidate_rank": self.selected_candidate_rank,
            "selected_candidate_fingerprint": (
                self.selected_candidate.fingerprint
            ),
            "selected_candidate_configuration": dict(
                self.selected_candidate.configuration
            ),
            "exact_compression": self.exact_compression.as_dict(),
            "identity": self.identity.as_dict(),
            "lifecycle_contract": self.lifecycle_contract.as_dict(),
            "resume_reference": (
                None
                if self.resume_reference is None
                else self.resume_reference.as_dict()
            ),
            "planning_only": True,
            "remote_execution_performed": False,
            "training_steps": 0,
            "backward_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5LaunchWave:
    """A bounded group that an external supervisor may launch concurrently."""

    wave_index: int
    target_plan_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wave_index",
            _nonnegative_int(self.wave_index, "wave_index"),
        )
        plans = tuple(
            _digest(value, "target plan fingerprint")
            for value in self.target_plan_fingerprints
        )
        if not plans or len(set(plans)) != len(plans):
            raise Lfm2CpcfV5FrontierError(
                "A launch wave needs unique target plan fingerprints."
            )
        object.__setattr__(self, "target_plan_fingerprints", plans)

    def as_dict(self) -> dict[str, Any]:
        return {
            "wave_index": self.wave_index,
            "target_plan_fingerprints": list(
                self.target_plan_fingerprints
            ),
            "launch_is_delegated_to_external_supervisor": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5FrontierSchedule:
    """Typed, non-executing multi-ratio CPCF-v5 launch schedule."""

    parent_graph_fingerprint: str
    bridge_artifact_sha256: str
    bridge_profile_sha256: str
    donor_tokenizer_semantics_sha256: str
    target_tokenizer_semantics_sha256: str
    target_vocabulary_rows: int
    lifecycle: Lfm2CpcfV5FrontierLifecycle
    maximum_parallel_lanes: int
    lanes: tuple[Lfm2CpcfV5LaunchLane, ...]
    waves: tuple[Lfm2CpcfV5LaunchWave, ...]

    def __post_init__(self) -> None:
        for name in (
            "parent_graph_fingerprint",
            "bridge_artifact_sha256",
            "bridge_profile_sha256",
            "donor_tokenizer_semantics_sha256",
            "target_tokenizer_semantics_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        if self.target_vocabulary_rows != STRICT_TARGET_VOCABULARY_ROWS:
            raise Lfm2CpcfV5FrontierError(
                "CPCF-v5 frontier requires exactly 4,096 target rows."
            )
        if type(self.lifecycle) is not Lfm2CpcfV5FrontierLifecycle:
            raise Lfm2CpcfV5FrontierError(
                "lifecycle must be Lfm2CpcfV5FrontierLifecycle."
            )
        parallel = _positive_int(
            self.maximum_parallel_lanes,
            "maximum_parallel_lanes",
        )
        object.__setattr__(self, "maximum_parallel_lanes", parallel)
        lanes = tuple(self.lanes)
        if not lanes or any(
            type(lane) is not Lfm2CpcfV5LaunchLane for lane in lanes
        ):
            raise Lfm2CpcfV5FrontierError(
                "lanes must contain typed launch lanes."
            )
        if tuple(lane.request_index for lane in lanes) != tuple(
            range(len(lanes))
        ):
            raise Lfm2CpcfV5FrontierError(
                "Launch lanes must preserve request order."
            )
        plans = tuple(lane.target_plan_fingerprint for lane in lanes)
        targets = tuple(
            lane.exact_compression.target_unique_parameters for lane in lanes
        )
        ratios = tuple(
            lane.exact_compression.requested_ratio for lane in lanes
        )
        if (
            len(set(plans)) != len(plans)
            or len(set(targets)) != len(targets)
            or len(set(ratios)) != len(ratios)
        ):
            raise Lfm2CpcfV5FrontierError(
                "Every requested width must have a distinct exact plan, target, "
                "and namespace."
            )
        for lane in lanes:
            configuration = dict(lane.selected_candidate.configuration)
            if (
                lane.target_plan.frontier.parent_graph_fingerprint
                != self.parent_graph_fingerprint
                or lane.lifecycle_contract.lifecycle is not self.lifecycle
                or configuration.get(TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY)
                != self.bridge_artifact_sha256
                or configuration.get(TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY)
                != self.bridge_profile_sha256
                or configuration.get(PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY)
                != self.donor_tokenizer_semantics_sha256
                or configuration.get(CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY)
                != self.target_tokenizer_semantics_sha256
            ):
                raise Lfm2CpcfV5FrontierError(
                    "Launch lane differs from the parent, bridge, or lifecycle "
                    "bound by its schedule."
                )
        waves = tuple(self.waves)
        if not waves or tuple(wave.wave_index for wave in waves) != tuple(
            range(len(waves))
        ):
            raise Lfm2CpcfV5FrontierError(
                "Launch waves must be nonempty, ordered, and zero-based."
            )
        flattened = tuple(
            plan
            for wave in waves
            for plan in wave.target_plan_fingerprints
        )
        if flattened != plans or any(
            len(wave.target_plan_fingerprints) > parallel for wave in waves
        ):
            raise Lfm2CpcfV5FrontierError(
                "Launch waves must cover every lane once in order within the "
                "declared parallelism limit."
            )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def lane_for_plan_fingerprint(
        self,
        target_plan_fingerprint: str,
    ) -> Lfm2CpcfV5LaunchLane:
        requested = _digest(
            target_plan_fingerprint,
            "target_plan_fingerprint",
        )
        matches = tuple(
            lane
            for lane in self.lanes
            if lane.target_plan_fingerprint == requested
        )
        if len(matches) != 1:
            raise Lfm2CpcfV5FrontierError(
                "Schedule does not contain exactly one requested target plan."
            )
        return matches[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "frontier_schedule",
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "bridge_profile_sha256": self.bridge_profile_sha256,
            "donor_tokenizer_semantics_sha256": (
                self.donor_tokenizer_semantics_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                self.target_tokenizer_semantics_sha256
            ),
            "target_vocabulary_rows": self.target_vocabulary_rows,
            "lifecycle": self.lifecycle.value,
            "maximum_parallel_lanes": self.maximum_parallel_lanes,
            "lane_internal_gpu_count": V5_LANE_INTERNAL_GPU_COUNT,
            "lane_internal_gpu_parallelism_is_separate_from_cross_lane_parallelism": (
                True
            ),
            "cross_lane_parallelism_requires_disjoint_gpu_pairs": True,
            "gpu_device_assignment_performed": False,
            "lanes": [lane.as_dict() for lane in self.lanes],
            "waves": [wave.as_dict() for wave in self.waves],
            "candidate_resume_and_quarantine_isolated_per_plan": True,
            "automatic_candidate_selection": True,
            "caller_forced_candidate_rank": False,
            "planning_only": True,
            "remote_execution_performed": False,
            "parent_or_child_model_loaded": False,
            "training_steps": 0,
            "backward_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "sparse_or_packed_parameters": 0,
        }


def authenticate_lfm2_cpcf_v5_resume(
    lane: Lfm2CpcfV5LaunchLane,
    reference: Lfm2CpcfV5ResumeReference,
) -> Lfm2CpcfV5ResumeReference:
    """Reject a resume admission from any other target width."""

    if type(lane) is not Lfm2CpcfV5LaunchLane:
        raise Lfm2CpcfV5FrontierError(
            "lane must be Lfm2CpcfV5LaunchLane."
        )
    if type(reference) is not Lfm2CpcfV5ResumeReference:
        raise Lfm2CpcfV5FrontierError(
            "reference must be Lfm2CpcfV5ResumeReference."
        )
    if reference.target_plan_fingerprint != lane.target_plan_fingerprint:
        raise Lfm2CpcfV5FrontierError(
            "Cross-width resume is forbidden: the admission target plan does "
            "not match this launch lane."
        )
    return reference


def _normalize_ratios(
    values: Sequence[str | int | float],
) -> tuple[Fraction, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise Lfm2CpcfV5FrontierError(
            "compression_ratios must be a sequence."
        )
    ratios = tuple(_compression_ratio(value) for value in values)
    if not ratios:
        raise Lfm2CpcfV5FrontierError(
            "compression_ratios cannot be empty."
        )
    if len(set(ratios)) != len(ratios):
        raise Lfm2CpcfV5FrontierError(
            "Duplicate compression ratios would alias one launch width."
        )
    return ratios


def _waves(
    lanes: Sequence[Lfm2CpcfV5LaunchLane],
    maximum_parallel_lanes: int,
) -> tuple[Lfm2CpcfV5LaunchWave, ...]:
    parallel = _positive_int(
        maximum_parallel_lanes,
        "maximum_parallel_lanes",
    )
    plans = tuple(lane.target_plan_fingerprint for lane in lanes)
    return tuple(
        Lfm2CpcfV5LaunchWave(
            wave_index=index,
            target_plan_fingerprints=plans[
                index * parallel : (index + 1) * parallel
            ],
        )
        for index in range((len(plans) + parallel - 1) // parallel)
    )


def plan_lfm2_cpcf_v5_frontier(
    *,
    parent_graph: ModelCircuitGraph,
    expected_parent_graph_fingerprint: str,
    tokenizer_bridge: StrictTokenizerBridgeArtifact,
    expected_bridge_artifact_sha256: str,
    compression_ratios: Sequence[str | int | float] = (
        DEFAULT_COMPRESSION_RATIOS
    ),
    lifecycle: Lfm2CpcfV5FrontierLifecycle = (
        Lfm2CpcfV5FrontierLifecycle.CAPTURE_QUARANTINE
    ),
    approved_review_bindings: (
        Mapping[str, Lfm2CpcfV5ApprovedReviewBindings] | None
    ) = None,
    resume_references: Mapping[str, Lfm2CpcfV5ResumeReference] | None = None,
    child_name_prefix: str = "lfm2-cpcf-v5",
    blank_seed: int = 0,
    maximum_candidates_per_ratio: int = 8,
    maximum_parallel_lanes: int = 1,
    minimum_hidden_size: int = 32,
) -> Lfm2CpcfV5FrontierSchedule:
    """Plan independent exact targets without loading or executing any model."""

    if type(parent_graph) is not ModelCircuitGraph:
        raise Lfm2CpcfV5FrontierError(
            "parent_graph must be an inspected ModelCircuitGraph."
        )
    expected_parent = _digest(
        expected_parent_graph_fingerprint,
        "expected_parent_graph_fingerprint",
    )
    if (
        parent_graph.fingerprint != expected_parent
        or parent_graph.family != "lfm2"
        or tuple(
            sorted({interface.modality for interface in parent_graph.interfaces})
        )
        != ("text",)
    ):
        raise Lfm2CpcfV5FrontierError(
            "Parent graph differs from the authenticated LFM2 text graph."
        )
    if type(tokenizer_bridge) is not StrictTokenizerBridgeArtifact:
        raise Lfm2CpcfV5FrontierError(
            "tokenizer_bridge must be StrictTokenizerBridgeArtifact."
        )
    expected_bridge = _digest(
        expected_bridge_artifact_sha256,
        "expected_bridge_artifact_sha256",
    )
    try:
        tokenizer_bridge.verify_files()
    except ValueError as exc:
        raise Lfm2CpcfV5FrontierError(
            "Strict tokenizer bridge failed file reauthentication."
        ) from exc
    profile = tokenizer_bridge.transport_profile
    if (
        tokenizer_bridge.artifact_sha256 != expected_bridge
        or profile.target_row_capacity != STRICT_TARGET_VOCABULARY_ROWS
        or profile.target_vocab_size != STRICT_TARGET_VOCABULARY_ROWS
        or parent_graph.tokenizer_fingerprint
        != tokenizer_bridge.donor_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfV5FrontierError(
            "Strict 4,096-row bridge differs from the authenticated parent or "
            "independent artifact pin."
        )
    if type(lifecycle) is not Lfm2CpcfV5FrontierLifecycle:
        raise Lfm2CpcfV5FrontierError(
            "lifecycle must be Lfm2CpcfV5FrontierLifecycle."
        )
    ratios = _normalize_ratios(compression_ratios)
    prefix = _text(child_name_prefix, "child_name_prefix")
    seed = _nonnegative_int(blank_seed, "blank_seed")
    maximum = _positive_int(
        maximum_candidates_per_ratio,
        "maximum_candidates_per_ratio",
    )
    parallel = _positive_int(
        maximum_parallel_lanes,
        "maximum_parallel_lanes",
    )
    minimum_hidden = _positive_int(
        minimum_hidden_size,
        "minimum_hidden_size",
    )
    reviews = _typed_mapping(
        approved_review_bindings,
        value_type=Lfm2CpcfV5ApprovedReviewBindings,
        name="approved_review_bindings",
    )
    resumes = _typed_mapping(
        resume_references,
        value_type=Lfm2CpcfV5ResumeReference,
        name="resume_references",
    )
    if (
        lifecycle
        is Lfm2CpcfV5FrontierLifecycle.CAPTURE_QUARANTINE
        and reviews
    ):
        raise Lfm2CpcfV5FrontierError(
            "capture_quarantine accepts no reviewed promotion bindings."
        )

    catalogs = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                minimum_hidden_size=minimum_hidden,
                tokenizer_bridge_artifact=tokenizer_bridge,
            ),
        )
    )
    raw: list[
        tuple[
            Fraction,
            BudgetedDynamicTargetPlan,
            DenseTargetTopologyCandidate,
        ]
    ] = []
    for ratio in ratios:
        plan = plan_budgeted_dynamic_target(
            parent_graph,
            DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                compression_ratio=_ratio_text(ratio),
                target_vocabulary_size=STRICT_TARGET_VOCABULARY_ROWS,
                match_policy="at_or_below",
                maximum_candidates=maximum,
            ),
            child_name=f"{prefix}-{_ratio_slug(ratio)}",
            # This frontier authenticates exactly one automatic selection
            # policy (``_AUTOMATIC_SELECTION_POLICY``).  Pin the balanced
            # supported-pareto selector explicitly so a later default change
            # in ``plan_budgeted_dynamic_target`` cannot silently retarget
            # the sealed v5 lanes.
            prefer_component_risk=False,
            tokenizer_fingerprint=(
                tokenizer_bridge.target_tokenizer_semantics_sha256
            ),
            adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
            blank_seed=seed,
            catalogs=catalogs,
        )
        rank = plan.target_certificate.selected_candidate_rank
        candidate = plan.frontier.candidates[rank]
        raw.append((ratio, plan, candidate))

    plan_fingerprints = {plan.fingerprint for _ratio, plan, _candidate in raw}
    if lifecycle is Lfm2CpcfV5FrontierLifecycle.REVIEWED_RERUN:
        if set(reviews) != plan_fingerprints:
            raise Lfm2CpcfV5FrontierError(
                "reviewed_rerun requires exactly one approved binding for "
                "every target plan and no cross-width extras."
            )
    elif reviews:
        raise Lfm2CpcfV5FrontierError(
            "Only reviewed_rerun accepts reviewed promotion bindings."
        )
    if not set(resumes).issubset(plan_fingerprints):
        raise Lfm2CpcfV5FrontierError(
            "resume_references contains a target plan outside this frontier."
        )
    for key, reference in resumes.items():
        if reference.target_plan_fingerprint != key:
            raise Lfm2CpcfV5FrontierError(
                "A resume mapping key differs from its typed plan binding."
            )

    lanes = tuple(
        Lfm2CpcfV5LaunchLane(
            request_index=index,
            target_plan=plan,
            selected_candidate=candidate,
            exact_compression=Lfm2CpcfV5ExactCompression.from_plan(
                requested_ratio=ratio,
                parent_unique_parameters=(
                    parent_graph.unique_parameter_count
                ),
                target_unique_parameters=candidate.unique_parameters,
            ),
            identity=Lfm2CpcfV5LaunchIdentity.from_plan_fingerprint(
                plan.fingerprint
            ),
            lifecycle_contract=Lfm2CpcfV5LifecycleContract(
                lifecycle=lifecycle,
                target_plan_fingerprint=plan.fingerprint,
                approved_review_bindings=reviews.get(plan.fingerprint),
            ),
            resume_reference=resumes.get(plan.fingerprint),
        )
        for index, (ratio, plan, candidate) in enumerate(raw)
    )
    return Lfm2CpcfV5FrontierSchedule(
        parent_graph_fingerprint=parent_graph.fingerprint,
        bridge_artifact_sha256=tokenizer_bridge.artifact_sha256,
        bridge_profile_sha256=tokenizer_bridge.profile_fingerprint,
        donor_tokenizer_semantics_sha256=(
            tokenizer_bridge.donor_tokenizer_semantics_sha256
        ),
        target_tokenizer_semantics_sha256=(
            tokenizer_bridge.target_tokenizer_semantics_sha256
        ),
        target_vocabulary_rows=profile.target_row_capacity,
        lifecycle=lifecycle,
        maximum_parallel_lanes=parallel,
        lanes=lanes,
        waves=_waves(lanes, parallel),
    )


def authorize_lfm2_cpcf_v5_reviewed_rerun(
    capture_schedule: Lfm2CpcfV5FrontierSchedule,
    approved_review_bindings: Mapping[
        str,
        Lfm2CpcfV5ApprovedReviewBindings,
    ],
    *,
    resume_references: Mapping[str, Lfm2CpcfV5ResumeReference] | None = None,
) -> Lfm2CpcfV5FrontierSchedule:
    """Authorize the same exact plans after independent evidence review."""

    if type(capture_schedule) is not Lfm2CpcfV5FrontierSchedule:
        raise Lfm2CpcfV5FrontierError(
            "capture_schedule must be Lfm2CpcfV5FrontierSchedule."
        )
    if (
        capture_schedule.lifecycle
        is not Lfm2CpcfV5FrontierLifecycle.CAPTURE_QUARANTINE
    ):
        raise Lfm2CpcfV5FrontierError(
            "Only a capture_quarantine schedule may become a reviewed_rerun."
        )
    reviews = _typed_mapping(
        approved_review_bindings,
        value_type=Lfm2CpcfV5ApprovedReviewBindings,
        name="approved_review_bindings",
    )
    plans = {lane.target_plan_fingerprint for lane in capture_schedule.lanes}
    if set(reviews) != plans or any(
        binding.target_plan_fingerprint != key
        for key, binding in reviews.items()
    ):
        raise Lfm2CpcfV5FrontierError(
            "Reviewed authorization requires exact plan-bound state, output, "
            "and manual-review bindings for every lane."
        )
    if resume_references is None:
        resumes: Mapping[str, Lfm2CpcfV5ResumeReference] = MappingProxyType(
            {
                lane.target_plan_fingerprint: lane.resume_reference
                for lane in capture_schedule.lanes
                if lane.resume_reference is not None
            }
        )
    else:
        resumes = _typed_mapping(
            resume_references,
            value_type=Lfm2CpcfV5ResumeReference,
            name="resume_references",
        )
    if not set(resumes).issubset(plans) or any(
        reference.target_plan_fingerprint != key
        for key, reference in resumes.items()
    ):
        raise Lfm2CpcfV5FrontierError(
            "Reviewed resume references must remain within their exact target "
            "plan widths."
        )
    lanes = tuple(
        replace(
            lane,
            lifecycle_contract=Lfm2CpcfV5LifecycleContract(
                lifecycle=Lfm2CpcfV5FrontierLifecycle.REVIEWED_RERUN,
                target_plan_fingerprint=lane.target_plan_fingerprint,
                approved_review_bindings=reviews[
                    lane.target_plan_fingerprint
                ],
            ),
            resume_reference=resumes.get(lane.target_plan_fingerprint),
        )
        for lane in capture_schedule.lanes
    )
    return Lfm2CpcfV5FrontierSchedule(
        parent_graph_fingerprint=(
            capture_schedule.parent_graph_fingerprint
        ),
        bridge_artifact_sha256=capture_schedule.bridge_artifact_sha256,
        bridge_profile_sha256=capture_schedule.bridge_profile_sha256,
        donor_tokenizer_semantics_sha256=(
            capture_schedule.donor_tokenizer_semantics_sha256
        ),
        target_tokenizer_semantics_sha256=(
            capture_schedule.target_tokenizer_semantics_sha256
        ),
        target_vocabulary_rows=capture_schedule.target_vocabulary_rows,
        lifecycle=Lfm2CpcfV5FrontierLifecycle.REVIEWED_RERUN,
        maximum_parallel_lanes=capture_schedule.maximum_parallel_lanes,
        lanes=lanes,
        waves=capture_schedule.waves,
    )


__all__ = [
    "DEFAULT_COMPRESSION_RATIOS",
    "Lfm2CpcfV5ApprovedReviewBindings",
    "Lfm2CpcfV5ExactCompression",
    "Lfm2CpcfV5FrontierError",
    "Lfm2CpcfV5FrontierLifecycle",
    "Lfm2CpcfV5FrontierSchedule",
    "Lfm2CpcfV5LaunchIdentity",
    "Lfm2CpcfV5LaunchLane",
    "Lfm2CpcfV5LaunchWave",
    "Lfm2CpcfV5LifecycleContract",
    "Lfm2CpcfV5ResumeReference",
    "SCHEMA",
    "STRICT_TARGET_VOCABULARY_ROWS",
    "V5_LANE_INTERNAL_GPU_COUNT",
    "authenticate_lfm2_cpcf_v5_resume",
    "authorize_lfm2_cpcf_v5_reviewed_rerun",
    "plan_lfm2_cpcf_v5_frontier",
]
