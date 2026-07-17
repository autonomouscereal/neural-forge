"""Global causal allocation for large, zero-training LFM2 FFN folds.

This lane deliberately separates three things that earlier experiments mixed:

* a physical proposal is an exact variable-width FFN payload with stable
  medoids, source-to-target assignments, and consumer coefficients;
* causal evidence is measured through the complete residual/operator path and
  scores final-margin, EOS, repetition, task-anchor, residual, and attention
  damage; and
* allocation chooses fourteen layer widths globally under one exact physical
  parameter budget.

Only the already transactional variable-width FFN installer is authorized in
this first slice.  Attention-head and short-convolution candidates may be
recorded for diagnostics, but planning never treats them as installable.
Consequently a 50% request fails closed when FFN contraction alone cannot meet
it.  No fallback silently substitutes a closed local-importance, independent
factorization, shared-basis, uniform-width, layer-sharing, or exact-token lane.

Parent generation is outside this module.  Evidence binds one immutable parent
trace/output envelope.  Candidate measurements and all staged promotion
evaluations must use ``full_prefix_canonical`` child traces.  The runner
instantiates a real child, installs the real physical plan, evaluates it under
the repository's fold-only training guard, and commits or byte-exactly rolls
back the existing variable-width transaction.
"""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_fold_executor import (
    fingerprint_model_runtime_state,
    fingerprint_model_state,
    run_fold_only_callback,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    FULL_PREFIX_CANONICAL_TRACE_MODE,
)
from neural_forge.hf_lfm2_variable_width_fold import (
    LFM2_VARIABLE_WIDTH_LAYER_COUNT,
    Lfm2VariableWidthEvidence,
    Lfm2VariableWidthFixedPlanBundle,
    install_lfm2_variable_width_fold_plan,
    plan_lfm2_variable_width_fold_for_widths,
    verify_lfm2_variable_width_evidence,
    verify_lfm2_variable_width_fixed_plan_bundle,
)


EVIDENCE_SCHEMA = "neural-forge-lfm2-global-causal-evidence-v1"
PROPOSAL_SCHEMA = "neural-forge-lfm2-global-causal-ffn-proposal-v1"
CANDIDATE_SCHEMA = "neural-forge-lfm2-global-causal-candidate-v1"
PLAN_SCHEMA = "neural-forge-lfm2-global-causal-plan-v1"
STAGE_SCHEMA = "neural-forge-lfm2-global-causal-stage-v1"
DECISION_SCHEMA = "neural-forge-lfm2-global-causal-decision-v1"
RUN_SCHEMA = "neural-forge-lfm2-global-causal-run-v1"
METHOD_ID = "lfm2_global_causal_variable_width_ffn"
PHYSICAL_INSTALLER_ID = "lfm2_variable_width_medoid_transaction_v1"
TARGET_FRACTIONS = (0.35, 0.50)
INSTALLABLE_SITE_KIND = "ffn_width"
SITE_KINDS = ("ffn_width", "attention_head", "short_conv_channel")


class Lfm2GlobalCausalError(RuntimeError):
    """Base failure for global causal evidence, planning, or execution."""


class Lfm2GlobalCausalEvidenceError(Lfm2GlobalCausalError):
    """A causal option or sealed evidence envelope is invalid or stale."""


class Lfm2GlobalCausalInfeasible(Lfm2GlobalCausalError):
    """No authorized physical selection reaches a requested exact budget."""


class Lfm2GlobalCausalExecutionError(Lfm2GlobalCausalError):
    """A physical transaction or sequential promotion gate failed closed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2GlobalCausalEvidenceError(
            "Global causal provenance must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2GlobalCausalEvidenceError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2GlobalCausalEvidenceError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2GlobalCausalEvidenceError(f"{name} must be a non-negative integer.")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2GlobalCausalEvidenceError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2GlobalCausalEvidenceError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2GlobalCausalEvidenceError(f"{name} must be finite and non-negative.")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2GlobalCausalEvidenceError(f"{name} must be a nonempty string.")
    return value.strip()


def _unique_parameter_count(model: Any) -> int:
    try:
        parameters = tuple(model.parameters())
    except Exception as exc:
        raise Lfm2GlobalCausalEvidenceError(
            "Could not enumerate the LFM2 parent parameters."
        ) from exc
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


@dataclass(frozen=True, slots=True)
class Lfm2CausalDamageWeights:
    """Bound weighting policy for normalized downstream damage channels."""

    final_logit_margin: float = 0.25
    eos: float = 0.15
    repetition: float = 0.15
    task_anchor: float = 0.25
    residual_path: float = 0.10
    attention_path: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "final_logit_margin",
            "eos",
            "repetition",
            "task_anchor",
            "residual_path",
            "attention_path",
        ):
            value = _finite_nonnegative(getattr(self, name), name)
            if value <= 0.0:
                raise Lfm2GlobalCausalEvidenceError(
                    "Every causal damage channel must have positive weight."
                )
            object.__setattr__(self, name, value)
        if not math.isclose(
            sum(self.as_dict().values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Lfm2GlobalCausalEvidenceError(
                "Causal damage weights must sum to exactly one."
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "final_logit_margin": self.final_logit_margin,
            "eos": self.eos,
            "repetition": self.repetition,
            "task_anchor": self.task_anchor,
            "residual_path": self.residual_path,
            "attention_path": self.attention_path,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CausalDamage:
    """Worst parent-relative damage observed after a physical ablation."""

    final_logit_margin: float
    eos: float
    repetition: float
    task_anchor: float
    residual_path: float
    attention_path: float

    def __post_init__(self) -> None:
        for name in (
            "final_logit_margin",
            "eos",
            "repetition",
            "task_anchor",
            "residual_path",
            "attention_path",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "final_logit_margin": self.final_logit_margin,
            "eos": self.eos,
            "repetition": self.repetition,
            "task_anchor": self.task_anchor,
            "residual_path": self.residual_path,
            "attention_path": self.attention_path,
        }

    def score(self, weights: Lfm2CausalDamageWeights) -> float:
        if not isinstance(weights, Lfm2CausalDamageWeights):
            raise Lfm2GlobalCausalEvidenceError(
                "weights must be Lfm2CausalDamageWeights."
            )
        return sum(
            self.as_dict()[name] * weight for name, weight in weights.as_dict().items()
        )


@dataclass(frozen=True, slots=True)
class Lfm2CausalFfnProposal:
    """One real isolated FFN-width proposal prepared for causal measurement."""

    schema: str
    option_id: str
    source_evidence_sha256: str
    layer_index: int
    source_width: int
    target_width: int
    parameters_saved: int
    medoid_source_neurons: tuple[int, ...]
    layer_binding_sha256: str
    isolated_plan_sha256: str
    proposal_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "option_id": self.option_id,
            "source_evidence_sha256": self.source_evidence_sha256,
            "layer_index": self.layer_index,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "parameters_saved": self.parameters_saved,
            "medoid_source_neurons": list(self.medoid_source_neurons),
            "layer_binding_sha256": self.layer_binding_sha256,
            "isolated_plan_sha256": self.isolated_plan_sha256,
            "proposal_sha256": self.proposal_sha256,
            "physical_installer_id": PHYSICAL_INSTALLER_ID,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


def _proposal_body(proposal: Lfm2CausalFfnProposal) -> dict[str, Any]:
    result = proposal.as_dict()
    result.pop("proposal_sha256", None)
    return result


def prepare_lfm2_causal_ffn_proposal(
    parent: Any,
    variable_width_evidence: Lfm2VariableWidthEvidence,
    *,
    layer_index: int,
    target_width: int,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
) -> Lfm2CausalFfnProposal:
    """Prepare one isolated, physically installable proposal without a forward."""

    proposal, _bundle = prepare_lfm2_causal_ffn_proposal_bundle(
        parent,
        variable_width_evidence,
        layer_index=layer_index,
        target_width=target_width,
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
    )
    return proposal


def prepare_lfm2_causal_ffn_proposal_bundle(
    parent: Any,
    variable_width_evidence: Lfm2VariableWidthEvidence,
    *,
    layer_index: int,
    target_width: int,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
) -> tuple[Lfm2CausalFfnProposal, Lfm2VariableWidthFixedPlanBundle]:
    """Prepare one proposal and the exact physical bundle it fingerprints.

    Candidate-evidence producers must evaluate the same tensor payload whose
    medoids and consumer coefficients are later authenticated by the global
    allocator.  Returning the already-built fixed-width bundle avoids a second
    clustering pass and removes any opportunity for measurement/installation
    drift.
    """

    verify_lfm2_variable_width_evidence(
        variable_width_evidence,
        parent=parent,
    )
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < LFM2_VARIABLE_WIDTH_LAYER_COUNT
    ):
        raise Lfm2GlobalCausalEvidenceError(
            "layer_index must be in the fourteen-layer LFM2 range."
        )
    source_width = variable_width_evidence.source_intermediate_width
    if (
        isinstance(target_width, bool)
        or not isinstance(target_width, int)
        or not 0 < target_width < source_width
    ):
        raise Lfm2GlobalCausalEvidenceError(
            "target_width must be a positive strict FFN contraction."
        )
    widths = [source_width] * LFM2_VARIABLE_WIDTH_LAYER_COUNT
    widths[layer_index] = target_width
    bundle = plan_lfm2_variable_width_fold_for_widths(
        parent,
        variable_width_evidence,
        widths,
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
    )
    layer_plan = bundle.plan.layer_plans[layer_index]
    structural = {
        "schema": PROPOSAL_SCHEMA,
        "source_evidence_sha256": variable_width_evidence.evidence_sha256,
        "layer_index": layer_index,
        "source_width": source_width,
        "target_width": target_width,
        "parameters_saved": (
            (source_width - target_width) * 3 * variable_width_evidence.hidden_size
        ),
        "medoid_source_neurons": list(layer_plan.medoid_source_neurons),
        "layer_binding_sha256": bundle.layer_binding_sha256[layer_index],
        "isolated_plan_sha256": bundle.plan.plan_sha256,
    }
    option_id = (
        f"ffn-l{layer_index:02d}-w{target_width:04d}-{_json_sha256(structural)[:16]}"
    )
    proposal = Lfm2CausalFfnProposal(
        schema=PROPOSAL_SCHEMA,
        option_id=option_id,
        source_evidence_sha256=variable_width_evidence.evidence_sha256,
        layer_index=layer_index,
        source_width=source_width,
        target_width=target_width,
        parameters_saved=structural["parameters_saved"],
        medoid_source_neurons=layer_plan.medoid_source_neurons,
        layer_binding_sha256=bundle.layer_binding_sha256[layer_index],
        isolated_plan_sha256=bundle.plan.plan_sha256,
        proposal_sha256="",
    )
    sealed = replace(
        proposal,
        proposal_sha256=_json_sha256(_proposal_body(proposal)),
    )
    return sealed, bundle


@dataclass(frozen=True, slots=True)
class Lfm2CausalRemovalCandidate:
    """Causal damage measured for one structural proposal."""

    schema: str
    option_id: str
    site_kind: Literal[
        "ffn_width",
        "attention_head",
        "short_conv_channel",
    ]
    layer_index: int
    source_units: int
    target_units: int
    parameters_saved: int
    damage: Lfm2CausalDamage
    measured_child_trace_sha256: str
    output_event_sha256: str
    validation_case_ids_sha256: str
    measurement_case_count: int
    proposal_sha256: str | None = None
    layer_binding_sha256: str | None = None
    retained_source_units: tuple[int, ...] = ()
    child_trace_mode: str = FULL_PREFIX_CANONICAL_TRACE_MODE

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(self, "option_id", _text(self.option_id, "option_id"))
        if self.site_kind not in SITE_KINDS:
            raise Lfm2GlobalCausalEvidenceError(
                f"site_kind must be one of {SITE_KINDS}."
            )
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < LFM2_VARIABLE_WIDTH_LAYER_COUNT
        ):
            raise Lfm2GlobalCausalEvidenceError(
                "layer_index must be in the fourteen-layer LFM2 range."
            )
        source = _positive_int(self.source_units, "source_units")
        target = _positive_int(self.target_units, "target_units")
        if target >= source:
            raise Lfm2GlobalCausalEvidenceError(
                "A causal removal candidate must contract its source units."
            )
        object.__setattr__(
            self,
            "parameters_saved",
            _positive_int(self.parameters_saved, "parameters_saved"),
        )
        if not isinstance(self.damage, Lfm2CausalDamage):
            raise Lfm2GlobalCausalEvidenceError("damage must be Lfm2CausalDamage.")
        for name in (
            "measured_child_trace_sha256",
            "output_event_sha256",
            "validation_case_ids_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "measurement_case_count",
            _positive_int(self.measurement_case_count, "measurement_case_count"),
        )
        if self.child_trace_mode != FULL_PREFIX_CANONICAL_TRACE_MODE:
            raise Lfm2GlobalCausalEvidenceError(
                "Causal candidates must use full_prefix_canonical child traces."
            )
        retained = tuple(self.retained_source_units)
        if (
            len(retained) != target
            or len(set(retained)) != target
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < source
                for value in retained
            )
        ):
            raise Lfm2GlobalCausalEvidenceError(
                "retained_source_units must bind every target unit uniquely."
            )
        object.__setattr__(self, "retained_source_units", retained)
        if self.site_kind == INSTALLABLE_SITE_KIND:
            if self.schema != CANDIDATE_SCHEMA:
                raise Lfm2GlobalCausalEvidenceError(
                    "FFN candidates must use the canonical candidate schema."
                )
            if self.proposal_sha256 is None or self.layer_binding_sha256 is None:
                raise Lfm2GlobalCausalEvidenceError(
                    "FFN candidates need physical proposal and layer bindings."
                )
            object.__setattr__(
                self,
                "proposal_sha256",
                _digest(self.proposal_sha256, "proposal_sha256"),
            )
            object.__setattr__(
                self,
                "layer_binding_sha256",
                _digest(
                    self.layer_binding_sha256,
                    "layer_binding_sha256",
                ),
            )
        elif self.proposal_sha256 is not None or self.layer_binding_sha256 is not None:
            raise Lfm2GlobalCausalEvidenceError(
                "Head/conv candidates cannot claim an FFN physical binding."
            )

    @classmethod
    def from_ffn_proposal(
        cls,
        proposal: Lfm2CausalFfnProposal,
        *,
        damage: Lfm2CausalDamage,
        measured_child_trace_sha256: str,
        output_event_sha256: str,
        validation_case_ids_sha256: str,
        measurement_case_count: int,
    ) -> "Lfm2CausalRemovalCandidate":
        if (
            not isinstance(proposal, Lfm2CausalFfnProposal)
            or proposal.schema != PROPOSAL_SCHEMA
            or _json_sha256(_proposal_body(proposal)) != proposal.proposal_sha256
        ):
            raise Lfm2GlobalCausalEvidenceError(
                "The physical FFN proposal seal is invalid."
            )
        return cls(
            schema=CANDIDATE_SCHEMA,
            option_id=proposal.option_id,
            site_kind=INSTALLABLE_SITE_KIND,
            layer_index=proposal.layer_index,
            source_units=proposal.source_width,
            target_units=proposal.target_width,
            parameters_saved=proposal.parameters_saved,
            damage=damage,
            measured_child_trace_sha256=measured_child_trace_sha256,
            output_event_sha256=output_event_sha256,
            validation_case_ids_sha256=validation_case_ids_sha256,
            measurement_case_count=measurement_case_count,
            proposal_sha256=proposal.proposal_sha256,
            layer_binding_sha256=proposal.layer_binding_sha256,
            retained_source_units=proposal.medoid_source_neurons,
        )

    @property
    def physically_installable(self) -> bool:
        return self.site_kind == INSTALLABLE_SITE_KIND

    def score(self, weights: Lfm2CausalDamageWeights) -> float:
        return self.damage.score(weights)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "option_id": self.option_id,
            "site_kind": self.site_kind,
            "layer_index": self.layer_index,
            "source_units": self.source_units,
            "target_units": self.target_units,
            "parameters_saved": self.parameters_saved,
            "damage": self.damage.as_dict(),
            "measured_child_trace_sha256": (self.measured_child_trace_sha256),
            "output_event_sha256": self.output_event_sha256,
            "validation_case_ids_sha256": (self.validation_case_ids_sha256),
            "measurement_case_count": self.measurement_case_count,
            "proposal_sha256": self.proposal_sha256,
            "layer_binding_sha256": self.layer_binding_sha256,
            "retained_source_units": list(self.retained_source_units),
            "child_trace_mode": self.child_trace_mode,
            "physically_installable": self.physically_installable,
            "physical_installer_id": (
                PHYSICAL_INSTALLER_ID if self.physically_installable else None
            ),
            "parent_forward_calls_during_measurement": 0,
            "parameter_gradients_used": False,
            "exact_token_match_objective_used": False,
        }


@dataclass(frozen=True, slots=True)
class Lfm2GlobalCausalEvidence:
    """One parent-bound causal frontier plus its physical FFN source evidence."""

    schema: str
    parent_identity_sha256: str
    parent_unique_parameters: int
    parent_trace_cache_key: str
    parent_corpus_sha256: str
    parent_output_envelope_sha256: str
    task_anchor_manifest_sha256: str
    variable_width_evidence: Lfm2VariableWidthEvidence
    candidates: tuple[Lfm2CausalRemovalCandidate, ...]
    damage_weights: Lfm2CausalDamageWeights
    cluster_ridge: float
    cluster_max_iterations: int
    cluster_distance_chunk_size: int
    parent_cache_build_count: int
    activation_jvp_calls: int
    activation_jvp_manifest_sha256: str | None
    evidence_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_identity_sha256": self.parent_identity_sha256,
            "parent_unique_parameters": self.parent_unique_parameters,
            "parent_trace_cache_key": self.parent_trace_cache_key,
            "parent_corpus_sha256": self.parent_corpus_sha256,
            "parent_output_envelope_sha256": (self.parent_output_envelope_sha256),
            "task_anchor_manifest_sha256": self.task_anchor_manifest_sha256,
            "variable_width_evidence": self.variable_width_evidence.as_dict(),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "damage_weights": self.damage_weights.as_dict(),
            "cluster": {
                "ridge": self.cluster_ridge,
                "max_iterations": self.cluster_max_iterations,
                "distance_chunk_size": self.cluster_distance_chunk_size,
            },
            "parent_cache_build_count": self.parent_cache_build_count,
            "parent_cache_built_at_most_once": (self.parent_cache_build_count <= 1),
            "parent_response_generation_calls": 0,
            "parent_forward_calls_during_candidate_measurement": 0,
            "activation_jvp_calls": self.activation_jvp_calls,
            "activation_jvp_manifest_sha256": (self.activation_jvp_manifest_sha256),
            "activation_jvp_scope": (
                "residual_states_only" if self.activation_jvp_calls else "unused"
            ),
            "parameter_jvp_or_gradient_calls": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "candidate_child_trace_mode": FULL_PREFIX_CANONICAL_TRACE_MODE,
            "exact_token_match_objective_used": False,
            "evidence_sha256": self.evidence_sha256,
        }


def _evidence_body(evidence: Lfm2GlobalCausalEvidence) -> dict[str, Any]:
    result = evidence.as_dict()
    result.pop("evidence_sha256", None)
    return result


def _cluster_values(
    *,
    cluster_ridge: Any,
    cluster_max_iterations: Any,
    cluster_distance_chunk_size: Any,
) -> tuple[float, int, int]:
    ridge = _finite_nonnegative(cluster_ridge, "cluster_ridge")
    iterations = _positive_int(
        cluster_max_iterations,
        "cluster_max_iterations",
    )
    chunk = _positive_int(
        cluster_distance_chunk_size,
        "cluster_distance_chunk_size",
    )
    return ridge, iterations, chunk


def seal_lfm2_global_causal_evidence(
    parent: Any,
    variable_width_evidence: Lfm2VariableWidthEvidence,
    candidates: Sequence[Lfm2CausalRemovalCandidate],
    *,
    parent_output_envelope_sha256: str,
    task_anchor_manifest_sha256: str,
    damage_weights: Lfm2CausalDamageWeights | None = None,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    parent_cache_build_count: int = 0,
    activation_jvp_calls: int = 0,
    activation_jvp_manifest_sha256: str | None = None,
) -> Lfm2GlobalCausalEvidence:
    """Seal and physically reauthenticate every installable causal option."""

    verify_lfm2_variable_width_evidence(
        variable_width_evidence,
        parent=parent,
    )
    selected_weights = (
        Lfm2CausalDamageWeights() if damage_weights is None else damage_weights
    )
    if not isinstance(selected_weights, Lfm2CausalDamageWeights):
        raise Lfm2GlobalCausalEvidenceError(
            "damage_weights must be Lfm2CausalDamageWeights or None."
        )
    ridge, iterations, chunk = _cluster_values(
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
    )
    ordered = tuple(sorted(candidates, key=lambda value: value.option_id))
    if (
        not ordered
        or any(
            not isinstance(candidate, Lfm2CausalRemovalCandidate)
            for candidate in ordered
        )
        or len({candidate.option_id for candidate in ordered}) != len(ordered)
    ):
        raise Lfm2GlobalCausalEvidenceError(
            "Candidates must be nonempty, typed, and option-ID unique."
        )
    seen_layer_widths: set[tuple[int, int]] = set()
    for candidate in ordered:
        if not candidate.physically_installable:
            continue
        key = (candidate.layer_index, candidate.target_units)
        if key in seen_layer_widths:
            raise Lfm2GlobalCausalEvidenceError(
                "Only one measured physical FFN option may bind a layer/width."
            )
        seen_layer_widths.add(key)
        expected_saved = (
            (candidate.source_units - candidate.target_units)
            * 3
            * variable_width_evidence.hidden_size
        )
        if (
            candidate.source_units != variable_width_evidence.source_intermediate_width
            or candidate.parameters_saved != expected_saved
        ):
            raise Lfm2GlobalCausalEvidenceError(
                "An FFN option's physical parameter accounting is invalid."
            )
        proposal = prepare_lfm2_causal_ffn_proposal(
            parent,
            variable_width_evidence,
            layer_index=candidate.layer_index,
            target_width=candidate.target_units,
            cluster_ridge=ridge,
            cluster_max_iterations=iterations,
            cluster_distance_chunk_size=chunk,
        )
        if (
            candidate.option_id != proposal.option_id
            or candidate.proposal_sha256 != proposal.proposal_sha256
            or candidate.layer_binding_sha256 != proposal.layer_binding_sha256
            or candidate.retained_source_units != proposal.medoid_source_neurons
        ):
            raise Lfm2GlobalCausalEvidenceError(
                f"Causal option {candidate.option_id!r} does not bind the "
                "physical proposal that was measured."
            )
    cache_builds = _nonnegative_int(
        parent_cache_build_count,
        "parent_cache_build_count",
    )
    if cache_builds > 1:
        raise Lfm2GlobalCausalEvidenceError(
            "Parent evidence may be cached once, never regenerated per option."
        )
    jvp_calls = _nonnegative_int(activation_jvp_calls, "activation_jvp_calls")
    if (jvp_calls == 0) != (activation_jvp_manifest_sha256 is None):
        raise Lfm2GlobalCausalEvidenceError(
            "A residual-state JVP manifest is required exactly when JVPs were used."
        )
    jvp_manifest = (
        None
        if activation_jvp_manifest_sha256 is None
        else _digest(
            activation_jvp_manifest_sha256,
            "activation_jvp_manifest_sha256",
        )
    )
    evidence = Lfm2GlobalCausalEvidence(
        schema=EVIDENCE_SCHEMA,
        parent_identity_sha256=variable_width_evidence.parent_identity_sha256,
        parent_unique_parameters=_unique_parameter_count(parent),
        parent_trace_cache_key=variable_width_evidence.cache_key,
        parent_corpus_sha256=variable_width_evidence.heldout_corpus_sha256,
        parent_output_envelope_sha256=_digest(
            parent_output_envelope_sha256,
            "parent_output_envelope_sha256",
        ),
        task_anchor_manifest_sha256=_digest(
            task_anchor_manifest_sha256,
            "task_anchor_manifest_sha256",
        ),
        variable_width_evidence=variable_width_evidence,
        candidates=ordered,
        damage_weights=selected_weights,
        cluster_ridge=ridge,
        cluster_max_iterations=iterations,
        cluster_distance_chunk_size=chunk,
        parent_cache_build_count=cache_builds,
        activation_jvp_calls=jvp_calls,
        activation_jvp_manifest_sha256=jvp_manifest,
        evidence_sha256="",
    )
    return replace(
        evidence,
        evidence_sha256=_json_sha256(_evidence_body(evidence)),
    )


def verify_lfm2_global_causal_evidence(
    evidence: Lfm2GlobalCausalEvidence,
    *,
    parent: Any | None = None,
) -> Lfm2GlobalCausalEvidence:
    if (
        not isinstance(evidence, Lfm2GlobalCausalEvidence)
        or evidence.schema != EVIDENCE_SCHEMA
        or _json_sha256(_evidence_body(evidence)) != evidence.evidence_sha256
        or evidence.parent_trace_cache_key != evidence.variable_width_evidence.cache_key
        or evidence.parent_corpus_sha256
        != evidence.variable_width_evidence.heldout_corpus_sha256
        or evidence.parent_identity_sha256
        != evidence.variable_width_evidence.parent_identity_sha256
        or len({value.option_id for value in evidence.candidates})
        != len(evidence.candidates)
    ):
        raise Lfm2GlobalCausalEvidenceError(
            "The global causal evidence seal or binding is invalid."
        )
    verify_lfm2_variable_width_evidence(
        evidence.variable_width_evidence,
        parent=parent,
    )
    if parent is not None and _unique_parameter_count(parent) != (
        evidence.parent_unique_parameters
    ):
        raise Lfm2GlobalCausalEvidenceError(
            "The live parent parameter count differs from causal evidence."
        )
    return evidence


@dataclass(frozen=True, slots=True)
class Lfm2GlobalCausalBudget:
    requested_fraction: float
    structural_quantum: int
    removed_ffn_neurons: int
    parameters_saved: int
    target_unique_parameters: int
    realized_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_fraction": self.requested_fraction,
            "structural_quantum": self.structural_quantum,
            "removed_ffn_neurons": self.removed_ffn_neurons,
            "parameters_saved": self.parameters_saved,
            "target_unique_parameters": self.target_unique_parameters,
            "realized_fraction": self.realized_fraction,
            "budget_met_exactly_in_physical_parameter_units": True,
        }


def lfm2_global_causal_budget(
    fraction: float,
    *,
    parent_unique_parameters: int,
    hidden_size: int,
) -> Lfm2GlobalCausalBudget:
    try:
        selected = float(fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2GlobalCausalInfeasible(
            "Target fraction must be 0.35 or 0.50."
        ) from exc
    if selected not in TARGET_FRACTIONS:
        raise Lfm2GlobalCausalInfeasible(
            "This large-target lane is fixed to 0.35 and/or 0.50."
        )
    parent_parameters = _positive_int(
        parent_unique_parameters,
        "parent_unique_parameters",
    )
    hidden = _positive_int(hidden_size, "hidden_size")
    quantum = 3 * hidden
    removed = int(math.floor(parent_parameters * selected / quantum + 0.5))
    saved = removed * quantum
    return Lfm2GlobalCausalBudget(
        requested_fraction=selected,
        structural_quantum=quantum,
        removed_ffn_neurons=removed,
        parameters_saved=saved,
        target_unique_parameters=parent_parameters - saved,
        realized_fraction=saved / parent_parameters,
    )


@dataclass(frozen=True, slots=True)
class _AllocationState:
    score: float
    widths: tuple[int, ...]
    option_ids: tuple[str, ...]


def _state_key(state: _AllocationState) -> tuple[Any, ...]:
    return (state.score, state.option_ids, state.widths)


def _allocation_states(
    evidence: Lfm2GlobalCausalEvidence,
    *,
    maximum_parameters_saved: int,
    lower_widths: Sequence[int] | None = None,
    upper_widths: Sequence[int] | None = None,
    max_states: int,
) -> dict[int, _AllocationState]:
    source = evidence.variable_width_evidence.source_intermediate_width
    lower = (
        (1,) * LFM2_VARIABLE_WIDTH_LAYER_COUNT
        if lower_widths is None
        else tuple(lower_widths)
    )
    upper = (
        (source,) * LFM2_VARIABLE_WIDTH_LAYER_COUNT
        if upper_widths is None
        else tuple(upper_widths)
    )
    if (
        len(lower) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
        or len(upper) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
        or any(
            not 0 < minimum <= maximum <= source
            for minimum, maximum in zip(lower, upper, strict=True)
        )
    ):
        raise Lfm2GlobalCausalInfeasible("Allocation width bounds are invalid.")
    by_layer: dict[int, list[Lfm2CausalRemovalCandidate]] = {
        index: [] for index in range(LFM2_VARIABLE_WIDTH_LAYER_COUNT)
    }
    for candidate in evidence.candidates:
        if candidate.physically_installable:
            by_layer[candidate.layer_index].append(candidate)
    states = {0: _AllocationState(0.0, (), ())}
    for layer_index in range(LFM2_VARIABLE_WIDTH_LAYER_COUNT):
        choices: list[tuple[int, int, float, str | None]] = []
        if lower[layer_index] <= source <= upper[layer_index]:
            choices.append((0, source, 0.0, None))
        for candidate in by_layer[layer_index]:
            if lower[layer_index] <= candidate.target_units <= upper[layer_index]:
                choices.append(
                    (
                        candidate.parameters_saved,
                        candidate.target_units,
                        candidate.score(evidence.damage_weights),
                        candidate.option_id,
                    )
                )
        if not choices:
            raise Lfm2GlobalCausalInfeasible(
                f"Layer {layer_index} has no authorized width in its bounds."
            )
        choices.sort(
            key=lambda value: (
                value[0],
                value[2],
                "" if value[3] is None else value[3],
            )
        )
        updated: dict[int, _AllocationState] = {}
        for saved_before, state in states.items():
            for saved, width, score, option_id in choices:
                total = saved_before + saved
                if total > maximum_parameters_saved:
                    continue
                candidate_state = _AllocationState(
                    score=state.score + score,
                    widths=state.widths + (width,),
                    option_ids=(
                        state.option_ids
                        if option_id is None
                        else state.option_ids + (option_id,)
                    ),
                )
                current = updated.get(total)
                if current is None or _state_key(candidate_state) < _state_key(current):
                    updated[total] = candidate_state
        if not updated:
            raise Lfm2GlobalCausalInfeasible(
                f"No exact physical state remains after layer {layer_index}."
            )
        if len(updated) > max_states:
            raise Lfm2GlobalCausalInfeasible(
                "Exact global allocation exceeded its immutable state ceiling; "
                "refusing an approximate selection."
            )
        states = updated
    return states


@dataclass(frozen=True, slots=True)
class Lfm2GlobalCausalStage:
    schema: str
    stage_index: int
    cumulative_parameters_saved: int
    cumulative_fraction: float
    target_widths: tuple[int, ...]
    option_ids: tuple[str, ...]
    cumulative_causal_score: float
    physical_bundle: Lfm2VariableWidthFixedPlanBundle = field(repr=False)
    stage_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage_index": self.stage_index,
            "cumulative_parameters_saved": self.cumulative_parameters_saved,
            "cumulative_fraction": self.cumulative_fraction,
            "target_widths": list(self.target_widths),
            "option_ids": list(self.option_ids),
            "cumulative_causal_score": self.cumulative_causal_score,
            "physical_bundle_sha256": self.physical_bundle.bundle_sha256,
            "physical_plan_sha256": self.physical_bundle.plan.plan_sha256,
            "physical_evidence_sha256": (
                self.physical_bundle.fixed_evidence.evidence_sha256
            ),
            "layer_binding_sha256": list(self.physical_bundle.layer_binding_sha256),
            "child_trace_mode_required": FULL_PREFIX_CANONICAL_TRACE_MODE,
            "sequential_global_evaluation_required": True,
            "stage_sha256": self.stage_sha256,
        }


def _stage_body(stage: Lfm2GlobalCausalStage) -> dict[str, Any]:
    result = stage.as_dict()
    result.pop("stage_sha256", None)
    return result


@dataclass(frozen=True, slots=True)
class Lfm2GlobalCausalPlan:
    schema: str
    method_id: str
    evidence_sha256: str
    budget: Lfm2GlobalCausalBudget
    stages: tuple[Lfm2GlobalCausalStage, ...]
    selected_option_ids: tuple[str, ...]
    unsupported_option_ids: tuple[str, ...]
    maximum_stage_fraction: float
    max_states: int
    plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "evidence_sha256": self.evidence_sha256,
            "budget": self.budget.as_dict(),
            "stages": [stage.as_dict() for stage in self.stages],
            "selected_option_ids": list(self.selected_option_ids),
            "unsupported_option_ids": list(self.unsupported_option_ids),
            "maximum_stage_fraction": self.maximum_stage_fraction,
            "max_states": self.max_states,
            "plan_sha256": self.plan_sha256,
            "physical_installer_id": PHYSICAL_INSTALLER_ID,
            "head_or_conv_installer_claimed": False,
            "exact_final_budget": True,
            "parent_generation_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


def _plan_body(plan: Lfm2GlobalCausalPlan) -> dict[str, Any]:
    result = plan.as_dict()
    result.pop("plan_sha256", None)
    return result


def _candidate_by_id(
    evidence: Lfm2GlobalCausalEvidence,
) -> dict[str, Lfm2CausalRemovalCandidate]:
    return {candidate.option_id: candidate for candidate in evidence.candidates}


def _build_stage(
    parent: Any,
    evidence: Lfm2GlobalCausalEvidence,
    state: _AllocationState,
    *,
    stage_index: int,
) -> Lfm2GlobalCausalStage:
    bundle = plan_lfm2_variable_width_fold_for_widths(
        parent,
        evidence.variable_width_evidence,
        state.widths,
        cluster_ridge=evidence.cluster_ridge,
        cluster_max_iterations=evidence.cluster_max_iterations,
        cluster_distance_chunk_size=evidence.cluster_distance_chunk_size,
    )
    candidates = _candidate_by_id(evidence)
    for option_id in state.option_ids:
        candidate = candidates[option_id]
        if (
            bundle.layer_binding_sha256[candidate.layer_index]
            != candidate.layer_binding_sha256
            or bundle.plan.layer_plans[candidate.layer_index].medoid_source_neurons
            != candidate.retained_source_units
        ):
            raise Lfm2GlobalCausalEvidenceError(
                f"Selected option {option_id!r} changed physical medoids or "
                "consumer binding in the combined plan."
            )
    saved = bundle.plan.parameters_saved
    if saved != sum(
        (evidence.variable_width_evidence.source_intermediate_width - width)
        * 3
        * evidence.variable_width_evidence.hidden_size
        for width in state.widths
    ):
        raise Lfm2GlobalCausalEvidenceError(
            "The combined physical plan disagrees with global accounting."
        )
    stage = Lfm2GlobalCausalStage(
        schema=STAGE_SCHEMA,
        stage_index=stage_index,
        cumulative_parameters_saved=saved,
        cumulative_fraction=saved / evidence.parent_unique_parameters,
        target_widths=state.widths,
        option_ids=tuple(sorted(state.option_ids)),
        cumulative_causal_score=state.score,
        physical_bundle=bundle,
        stage_sha256="",
    )
    return replace(
        stage,
        stage_sha256=_json_sha256(_stage_body(stage)),
    )


def plan_lfm2_global_causal_target(
    parent: Any,
    evidence: Lfm2GlobalCausalEvidence,
    *,
    target_fraction: float,
    maximum_stage_fraction: float = 0.10,
    max_states: int = 100_000,
) -> Lfm2GlobalCausalPlan:
    """Choose and physically bind a staged exact 35% or 50% FFN plan."""

    verify_lfm2_global_causal_evidence(evidence, parent=parent)
    stage_fraction = _finite_nonnegative(
        maximum_stage_fraction,
        "maximum_stage_fraction",
    )
    if stage_fraction <= 0.0:
        raise Lfm2GlobalCausalInfeasible("maximum_stage_fraction must be positive.")
    state_limit = _positive_int(max_states, "max_states")
    budget = lfm2_global_causal_budget(
        target_fraction,
        parent_unique_parameters=evidence.parent_unique_parameters,
        hidden_size=evidence.variable_width_evidence.hidden_size,
    )
    states = _allocation_states(
        evidence,
        maximum_parameters_saved=budget.parameters_saved,
        max_states=state_limit,
    )
    final_state = states.get(budget.parameters_saved)
    if final_state is None:
        maximum_authorized = max(states)
        unsupported = tuple(
            candidate.option_id
            for candidate in evidence.candidates
            if not candidate.physically_installable
        )
        raise Lfm2GlobalCausalInfeasible(
            "No authorized physical FFN selection reaches the exact "
            f"{target_fraction:.0%} structural budget. Maximum represented "
            f"exact saving is {maximum_authorized}; unsupported head/conv "
            f"options ({len(unsupported)}) were not counted."
        )

    final_widths = final_state.widths
    source_width = evidence.variable_width_evidence.source_intermediate_width
    previous_widths = (source_width,) * LFM2_VARIABLE_WIDTH_LAYER_COUNT
    previous_saved = 0
    stage_limit = max(
        budget.structural_quantum,
        int(
            math.floor(
                evidence.parent_unique_parameters
                * stage_fraction
                / budget.structural_quantum
            )
        )
        * budget.structural_quantum,
    )
    stage_states: list[_AllocationState] = []
    while previous_saved + stage_limit < budget.parameters_saved:
        desired = previous_saved + stage_limit
        bounded = _allocation_states(
            evidence,
            maximum_parameters_saved=desired,
            lower_widths=final_widths,
            upper_widths=previous_widths,
            max_states=state_limit,
        )
        reachable = [saved for saved in bounded if previous_saved < saved <= desired]
        if not reachable:
            break
        next_saved = max(reachable)
        next_state = bounded[next_saved]
        if next_state.widths == previous_widths:
            break
        stage_states.append(next_state)
        previous_widths = next_state.widths
        previous_saved = next_saved
    if not stage_states or stage_states[-1].widths != final_widths:
        stage_states.append(final_state)

    stages = tuple(
        _build_stage(parent, evidence, state, stage_index=index)
        for index, state in enumerate(stage_states)
    )
    if (
        stages[-1].cumulative_parameters_saved != budget.parameters_saved
        or stages[-1].target_widths != final_widths
        or any(
            left.cumulative_parameters_saved >= right.cumulative_parameters_saved
            for left, right in zip(stages, stages[1:])
        )
        or any(
            right_width > left_width
            for left, right in zip(stages, stages[1:])
            for left_width, right_width in zip(
                left.target_widths,
                right.target_widths,
                strict=True,
            )
        )
    ):
        raise Lfm2GlobalCausalInfeasible(
            "Sequential global stages are not monotonic and exact."
        )
    plan = Lfm2GlobalCausalPlan(
        schema=PLAN_SCHEMA,
        method_id=METHOD_ID,
        evidence_sha256=evidence.evidence_sha256,
        budget=budget,
        stages=stages,
        selected_option_ids=tuple(sorted(final_state.option_ids)),
        unsupported_option_ids=tuple(
            candidate.option_id
            for candidate in evidence.candidates
            if not candidate.physically_installable
        ),
        maximum_stage_fraction=stage_fraction,
        max_states=state_limit,
        plan_sha256="",
    )
    return replace(plan, plan_sha256=_json_sha256(_plan_body(plan)))


def verify_lfm2_global_causal_plan(
    plan: Lfm2GlobalCausalPlan,
    evidence: Lfm2GlobalCausalEvidence,
    *,
    parent: Any | None = None,
) -> Lfm2GlobalCausalPlan:
    verify_lfm2_global_causal_evidence(evidence, parent=parent)
    if (
        not isinstance(plan, Lfm2GlobalCausalPlan)
        or plan.schema != PLAN_SCHEMA
        or plan.method_id != METHOD_ID
        or plan.evidence_sha256 != evidence.evidence_sha256
        or not plan.stages
        or _json_sha256(_plan_body(plan)) != plan.plan_sha256
        or plan.stages[-1].cumulative_parameters_saved != plan.budget.parameters_saved
        or plan.stages[-1].option_ids != plan.selected_option_ids
    ):
        raise Lfm2GlobalCausalEvidenceError(
            "The global causal plan structure or seal is invalid."
        )
    for index, stage in enumerate(plan.stages):
        if (
            stage.schema != STAGE_SCHEMA
            or stage.stage_index != index
            or _json_sha256(_stage_body(stage)) != stage.stage_sha256
            or stage.target_widths != stage.physical_bundle.plan.target_widths
            or stage.cumulative_parameters_saved
            != stage.physical_bundle.plan.parameters_saved
        ):
            raise Lfm2GlobalCausalEvidenceError(
                "A global causal stage changed after planning."
            )
        verify_lfm2_variable_width_fixed_plan_bundle(
            stage.physical_bundle,
            parent=parent,
        )
    return plan


@dataclass(frozen=True, slots=True)
class Lfm2GlobalCausalEvaluationContext:
    """A child-only evaluation context; the live parent is intentionally absent."""

    child: Any
    evidence_sha256: str
    parent_trace_cache_key: str
    parent_output_envelope_sha256: str
    task_anchor_manifest_sha256: str
    stage: Lfm2GlobalCausalStage
    transaction_report: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Lfm2GlobalCausalStageDecision:
    """Binding whole-output/trajectory decision for one cumulative child."""

    accepted: bool
    reason: str
    child_trace_cache_key: str
    child_output_events_sha256: str
    trajectory_gate_passed: bool
    intelligence_gate_passed: bool
    coherent_response_gate_passed: bool
    metrics: tuple[tuple[str, float | int | bool | str], ...] = ()
    child_trace_mode: str = FULL_PREFIX_CANONICAL_TRACE_MODE

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise Lfm2GlobalCausalExecutionError(
                "Stage decision accepted must be boolean."
            )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        for name in (
            "child_trace_cache_key",
            "child_output_events_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.child_trace_mode != FULL_PREFIX_CANONICAL_TRACE_MODE:
            raise Lfm2GlobalCausalExecutionError(
                "Child promotion must use full_prefix_canonical traces."
            )
        for name in (
            "trajectory_gate_passed",
            "intelligence_gate_passed",
            "coherent_response_gate_passed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise Lfm2GlobalCausalExecutionError(f"{name} must be boolean.")
        if self.accepted and not all(
            (
                self.trajectory_gate_passed,
                self.intelligence_gate_passed,
                self.coherent_response_gate_passed,
            )
        ):
            raise Lfm2GlobalCausalExecutionError(
                "An accepted stage must pass trajectory, intelligence, and "
                "coherent-response gates."
            )
        normalized = []
        for name, value in self.metrics:
            key = _text(name, "metric name")
            if not isinstance(value, (str, int, float, bool)):
                raise Lfm2GlobalCausalExecutionError(
                    f"Metric {key!r} is not a JSON scalar."
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise Lfm2GlobalCausalExecutionError(f"Metric {key!r} is not finite.")
            normalized.append((key, value))
        normalized.sort(key=lambda value: value[0])
        if len({name for name, _value in normalized}) != len(normalized):
            raise Lfm2GlobalCausalExecutionError("Stage metric names must be unique.")
        object.__setattr__(self, "metrics", tuple(normalized))

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "accepted": self.accepted,
            "reason": self.reason,
            "child_trace_cache_key": self.child_trace_cache_key,
            "child_output_events_sha256": self.child_output_events_sha256,
            "trajectory_gate_passed": self.trajectory_gate_passed,
            "intelligence_gate_passed": self.intelligence_gate_passed,
            "coherent_response_gate_passed": (self.coherent_response_gate_passed),
            "metrics": dict(self.metrics),
            "child_trace_mode": self.child_trace_mode,
            "exact_token_match_is_diagnostic_only": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2GlobalCausalStageAttempt:
    stage_sha256: str
    decision: Lfm2GlobalCausalStageDecision
    fold_report: Mapping[str, Any]
    transaction_committed: bool
    rollback_exact: bool | None
    parent_forward_calls: int
    guard_reports: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_sha256": self.stage_sha256,
            "decision": self.decision.as_dict(),
            "decision_sha256": self.decision.fingerprint,
            "fold_report": dict(self.fold_report),
            "transaction_committed": self.transaction_committed,
            "rollback_exact": self.rollback_exact,
            "parent_forward_calls": self.parent_forward_calls,
            "guard_reports": [dict(value) for value in self.guard_reports],
        }


@dataclass(frozen=True, slots=True)
class Lfm2GlobalCausalRun:
    plan_sha256: str
    evidence_sha256: str
    status: Literal["accepted", "rejected"]
    attempts: tuple[Lfm2GlobalCausalStageAttempt, ...]
    final_stage_sha256: str | None
    committed_child: Any = field(default=None, repr=False, compare=False)

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "evidence_sha256": self.evidence_sha256,
            "status": self.status,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "final_stage_sha256": self.final_stage_sha256,
            "real_child_instantiated": bool(self.attempts),
            "real_physical_installer_used": bool(self.attempts),
            "checkpoint_published": False,
            "parent_forward_calls": sum(
                attempt.parent_forward_calls for attempt in self.attempts
            ),
            "new_parent_generation_calls": 0,
            "optimizer_constructions": sum(
                int(report["optimizer_constructions"])
                for attempt in self.attempts
                for report in attempt.guard_reports
            ),
            "optimizer_steps": sum(
                int(report["optimizer_steps"])
                for attempt in self.attempts
                for report in attempt.guard_reports
            ),
            "backward_calls": sum(
                int(report["backward_calls"])
                for attempt in self.attempts
                for report in attempt.guard_reports
            ),
            "autograd_grad_calls": sum(
                int(report["autograd_grad_calls"])
                for attempt in self.attempts
                for report in attempt.guard_reports
            ),
            "exact_token_match_objective_used": False,
        }


def _call_evaluator(
    evaluator: Any,
    context: Lfm2GlobalCausalEvaluationContext,
) -> Lfm2GlobalCausalStageDecision:
    callback = getattr(evaluator, "evaluate", None)
    if not callable(callback):
        if callable(evaluator):
            callback = evaluator
        else:
            raise Lfm2GlobalCausalExecutionError(
                "evaluator must be callable or expose evaluate()."
            )
    value = callback(context)
    if not isinstance(value, Lfm2GlobalCausalStageDecision):
        raise Lfm2GlobalCausalExecutionError(
            "Evaluator must return Lfm2GlobalCausalStageDecision."
        )
    return value


def audit_lfm2_parent_forward_calls(
    parent: Any,
    callback: Callable[[], Any],
    *,
    phase: str = "lfm2-parent-forward-audit",
    event_sink: list[Mapping[str, Any]] | None = None,
    max_recorded_events: int = 64,
) -> tuple[Any, int]:
    """Run a callback while auditing exact live-parent module identities.

    A module-local hook is unsafe here: physical installers legitimately
    ``deepcopy`` parent module shells, and PyTorch copies their hook registries
    as well.  A subsequent child forward would then invoke the copied closure
    and look like a parent call.  The global pre-hook below never mutates the
    parent and counts only modules whose object identity belongs to the exact
    live parent graph.
    """

    if not callable(callback):
        raise TypeError("callback must be callable.")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("phase must be a nonempty string.")
    if (
        not isinstance(max_recorded_events, int)
        or isinstance(max_recorded_events, bool)
        or max_recorded_events < 1
    ):
        raise ValueError("max_recorded_events must be a positive integer.")
    if event_sink is not None and not hasattr(event_sink, "append"):
        raise TypeError("event_sink must support append() or be None.")

    try:
        named_modules = tuple(parent.named_modules(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported torch releases
        named_modules = tuple(parent.named_modules())
    exact_modules: dict[int, tuple[Any, list[str]]] = {}
    for raw_path, module in named_modules:
        path = raw_path or "<root>"
        existing = exact_modules.get(id(module))
        if existing is None:
            exact_modules[id(module)] = (module, [path])
        elif path not in existing[1]:
            existing[1].append(path)

    calls = 0

    def observe(module: Any, _inputs: Any) -> None:
        nonlocal calls
        exact = exact_modules.get(id(module))
        if exact is None or module is not exact[0]:
            return
        calls += 1
        if event_sink is None or len(event_sink) >= max_recorded_events:
            return
        frames = traceback.extract_stack(limit=32)[:-1]
        event_sink.append(
            {
                "schema": "neural-forge-lfm2-parent-forward-audit-event-v1",
                "phase": phase,
                "call_index": calls,
                "module_paths": tuple(exact[1]),
                "module_type": (
                    f"{type(module).__module__}.{type(module).__qualname__}"
                ),
                "call_stack": tuple(
                    {
                        "file": frame.filename,
                        "line": int(frame.lineno),
                        "function": frame.name,
                    }
                    for frame in frames
                ),
            }
        )

    from torch.nn.modules.module import register_module_forward_pre_hook

    handle = register_module_forward_pre_hook(observe)
    try:
        decision = callback()
    finally:
        handle.remove()
    return decision, calls


def _evaluate_with_parent_forward_audit(
    parent: Any,
    callback: Callable[[], Lfm2GlobalCausalStageDecision],
) -> tuple[Lfm2GlobalCausalStageDecision, int]:
    decision, calls = audit_lfm2_parent_forward_calls(parent, callback)
    return decision, calls


def execute_lfm2_global_causal_plan(
    parent: Any,
    evidence: Lfm2GlobalCausalEvidence,
    plan: Lfm2GlobalCausalPlan,
    *,
    child_factory: Callable[[], Any],
    evaluator: Any,
    residual_corrector_factory: (
        Callable[[Lfm2GlobalCausalStage, Any], Callable[..., Any] | None] | None
    ) = None,
) -> Lfm2GlobalCausalRun:
    """Instantiate, evaluate, commit/rollback real cumulative child proposals."""

    verify_lfm2_global_causal_plan(plan, evidence, parent=parent)
    if not callable(child_factory):
        raise Lfm2GlobalCausalExecutionError("child_factory must be callable.")
    if residual_corrector_factory is not None and not callable(
        residual_corrector_factory
    ):
        raise Lfm2GlobalCausalExecutionError(
            "residual_corrector_factory must be callable or None."
        )
    attempts = []
    committed_child = None
    for stage_index, stage in enumerate(plan.stages):
        guard_reports = []
        child, factory_guard = run_fold_only_callback(
            child_factory,
            phase=f"lfm2-global-causal-stage-{stage_index}-child-factory",
            protected_models=(parent,),
        )
        guard_reports.append(factory_guard.as_dict())
        if child is parent:
            raise Lfm2GlobalCausalExecutionError(
                "The physical child must be distinct from the parent."
            )
        blank_state = fingerprint_model_state(child)
        blank_runtime = fingerprint_model_runtime_state(child)
        transaction = None
        try:

            def install() -> Any:
                corrector = (
                    None
                    if residual_corrector_factory is None
                    else residual_corrector_factory(stage, child)
                )
                return install_lfm2_variable_width_fold_plan(
                    parent,
                    child,
                    stage.physical_bundle.fixed_evidence,
                    stage.physical_bundle.plan,
                    residual_corrector=corrector,
                )

            transaction, install_guard = run_fold_only_callback(
                install,
                phase=f"lfm2-global-causal-stage-{stage_index}-install",
                protected_models=(parent,),
            )
            guard_reports.append(install_guard.as_dict())
            fold_report = transaction.report.as_dict()
            context = Lfm2GlobalCausalEvaluationContext(
                child=child,
                evidence_sha256=evidence.evidence_sha256,
                parent_trace_cache_key=evidence.parent_trace_cache_key,
                parent_output_envelope_sha256=(evidence.parent_output_envelope_sha256),
                task_anchor_manifest_sha256=(evidence.task_anchor_manifest_sha256),
                stage=stage,
                transaction_report=MappingProxyType(fold_report),
            )

            def evaluate() -> tuple[Lfm2GlobalCausalStageDecision, int]:
                return _evaluate_with_parent_forward_audit(
                    parent,
                    lambda: _call_evaluator(evaluator, context),
                )

            evaluation, evaluation_guard = run_fold_only_callback(
                evaluate,
                phase=f"lfm2-global-causal-stage-{stage_index}-evaluation",
                protected_models=(parent,),
            )
            guard_reports.append(evaluation_guard.as_dict())
            decision, parent_forward_calls = evaluation
            if parent_forward_calls:
                raise Lfm2GlobalCausalExecutionError(
                    "Sequential child evaluation called the live parent instead "
                    "of using its sealed cache."
                )
            final_stage = stage_index == len(plan.stages) - 1
            if decision.accepted and final_stage:
                transaction.commit()
                committed_child = child
                attempts.append(
                    Lfm2GlobalCausalStageAttempt(
                        stage_sha256=stage.stage_sha256,
                        decision=decision,
                        fold_report=MappingProxyType(fold_report),
                        transaction_committed=True,
                        rollback_exact=None,
                        parent_forward_calls=0,
                        guard_reports=tuple(
                            MappingProxyType(dict(value)) for value in guard_reports
                        ),
                    )
                )
                break

            rollback_result = transaction.rollback()
            rollback_exact = bool(
                rollback_result
                and fingerprint_model_state(child) == blank_state
                and fingerprint_model_runtime_state(child) == blank_runtime
            )
            if not rollback_exact:
                raise Lfm2GlobalCausalExecutionError(
                    "A staged physical proposal did not roll back exactly."
                )
            attempts.append(
                Lfm2GlobalCausalStageAttempt(
                    stage_sha256=stage.stage_sha256,
                    decision=decision,
                    fold_report=MappingProxyType(fold_report),
                    transaction_committed=False,
                    rollback_exact=True,
                    parent_forward_calls=0,
                    guard_reports=tuple(
                        MappingProxyType(dict(value)) for value in guard_reports
                    ),
                )
            )
            if not decision.accepted:
                break
        except BaseException:
            if transaction is not None and transaction.active:
                transaction.rollback()
            if (
                fingerprint_model_state(child) != blank_state
                or fingerprint_model_runtime_state(child) != blank_runtime
            ):
                raise Lfm2GlobalCausalExecutionError(
                    "Execution failed and the real child did not restore exactly."
                )
            raise

    accepted = (
        committed_child is not None
        and len(attempts) == len(plan.stages)
        and all(attempt.decision.accepted for attempt in attempts)
    )
    return Lfm2GlobalCausalRun(
        plan_sha256=plan.plan_sha256,
        evidence_sha256=evidence.evidence_sha256,
        status="accepted" if accepted else "rejected",
        attempts=tuple(attempts),
        final_stage_sha256=(plan.stages[-1].stage_sha256 if accepted else None),
        committed_child=committed_child,
    )


__all__ = [
    "CANDIDATE_SCHEMA",
    "DECISION_SCHEMA",
    "EVIDENCE_SCHEMA",
    "INSTALLABLE_SITE_KIND",
    "Lfm2CausalDamage",
    "Lfm2CausalDamageWeights",
    "Lfm2CausalFfnProposal",
    "Lfm2CausalRemovalCandidate",
    "Lfm2GlobalCausalBudget",
    "Lfm2GlobalCausalError",
    "Lfm2GlobalCausalEvaluationContext",
    "Lfm2GlobalCausalEvidence",
    "Lfm2GlobalCausalEvidenceError",
    "Lfm2GlobalCausalExecutionError",
    "Lfm2GlobalCausalInfeasible",
    "Lfm2GlobalCausalPlan",
    "Lfm2GlobalCausalRun",
    "Lfm2GlobalCausalStage",
    "Lfm2GlobalCausalStageAttempt",
    "Lfm2GlobalCausalStageDecision",
    "METHOD_ID",
    "PHYSICAL_INSTALLER_ID",
    "PLAN_SCHEMA",
    "PROPOSAL_SCHEMA",
    "RUN_SCHEMA",
    "SITE_KINDS",
    "STAGE_SCHEMA",
    "TARGET_FRACTIONS",
    "execute_lfm2_global_causal_plan",
    "lfm2_global_causal_budget",
    "plan_lfm2_global_causal_target",
    "prepare_lfm2_causal_ffn_proposal",
    "seal_lfm2_global_causal_evidence",
    "verify_lfm2_global_causal_evidence",
    "verify_lfm2_global_causal_plan",
]
