"""Top-K cumulative residual/attention allocation for zero-training LFM2 folds.

The first global causal allocator deliberately keeps one additive optimum as a
control.  This module adds the missing interaction experiment without changing
that control:

* retain multiple exact-budget structural allocations;
* bind every allocation to an exact physical variable-width bundle;
* install and evaluate each complete multi-layer child, not isolated layers;
* rank coherent whole-response and parent-demonstrated intelligence retention;
* audit against live-parent forwards and every training operation; and
* roll every search child back byte-for-byte.

This module is intentionally not imported by the live cached-causal producer.
It is a post-frontier consumer so an in-flight evidence contract remains
immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_fold_executor import (
    fingerprint_model_runtime_state,
    fingerprint_model_state,
    run_fold_only_callback,
)
from neural_forge.hf_lfm2_global_causal_allocator import (
    Lfm2CausalDamageWeights,
    Lfm2CausalRemovalCandidate,
    Lfm2GlobalCausalEvidence,
    Lfm2GlobalCausalInfeasible,
    Lfm2GlobalCausalPlan,
    Lfm2GlobalCausalStage,
    audit_lfm2_parent_forward_calls,
    lfm2_global_causal_budget,
    verify_lfm2_global_causal_evidence,
    verify_lfm2_global_causal_plan,
)
from neural_forge.hf_lfm2_canonical_proposal_pack import (
    Lfm2CanonicalProposalPack,
)
from neural_forge import (
    hf_lfm2_global_causal_allocator as global_allocator_module,
    hf_lfm2_variable_width_fold as variable_width_fold_module,
)
from neural_forge.hf_lfm2_variable_width_fold import (
    LFM2_VARIABLE_WIDTH_LAYER_COUNT,
    Lfm2LayerWidthCurve,
    Lfm2VariableWidthFoldPlan,
    Lfm2VariableWidthFixedPlanBundle,
    Lfm2VariableWidthLayerPlan,
    allocate_lfm2_variable_widths,
    fingerprint_lfm2_variable_width_layer_plan,
    install_lfm2_variable_width_fold_plan,
    verify_lfm2_variable_width_fixed_plan_bundle,
)


ALTERNATIVE_SCHEMA = "neural-forge-lfm2-exact-allocation-alternative-v1"
PLAN_SCHEMA = "neural-forge-lfm2-interaction-beam-plan-v1"
MATERIALIZATION_SCHEMA = "neural-forge-lfm2-interaction-materialization-v1"
BOUND_ALTERNATIVE_SCHEMA = "neural-forge-lfm2-bound-allocation-alternative-v1"
BOUND_PLAN_SCHEMA = "neural-forge-lfm2-bound-interaction-beam-plan-v1"
QUALITY_SCHEMA = "neural-forge-lfm2-cumulative-interaction-quality-v1"
ATTEMPT_SCHEMA = "neural-forge-lfm2-cumulative-interaction-attempt-v1"
RUN_SCHEMA = "neural-forge-lfm2-interaction-beam-run-v1"
METHOD_ID = "lfm2_topk_cumulative_residual_attention_beam_v1"


class Lfm2InteractionBeamError(RuntimeError):
    """The top-K interaction plan, binding, or evaluation is invalid."""


class Lfm2InteractionBeamInfeasible(Lfm2InteractionBeamError):
    """No exact physical allocation reaches the requested target."""


class Lfm2InteractionBeamExecutionError(Lfm2InteractionBeamError):
    """A cumulative child was unsafe or did not roll back exactly."""


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
        raise Lfm2InteractionBeamError(
            "Interaction-beam provenance must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2InteractionBeamError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2InteractionBeamError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2InteractionBeamError(
            f"{name} must be a nonnegative integer."
        )
    return value


def _finite(value: Any, name: str) -> float:
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2InteractionBeamError(f"{name} must be finite.") from exc
    if not math.isfinite(selected):
        raise Lfm2InteractionBeamError(f"{name} must be finite.")
    return selected


def _unit_interval(value: Any, name: str) -> float:
    selected = _finite(value, name)
    if not 0.0 <= selected <= 1.0:
        raise Lfm2InteractionBeamError(f"{name} must be in [0, 1].")
    return selected


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2InteractionBeamError(f"{name} must be nonempty text.")
    return value.strip()


def _unique_parameter_count(model: Any) -> int:
    try:
        parameters = tuple(model.parameters(remove_duplicate=True))
    except TypeError:  # pragma: no cover - older supported PyTorch
        parameters = tuple(model.parameters())
    unique: dict[int, Any] = {}
    for parameter in parameters:
        unique.setdefault(id(parameter), parameter)
    return sum(int(parameter.numel()) for parameter in unique.values())


@dataclass(frozen=True, slots=True)
class Lfm2ExactAllocationAlternative:
    """One exact-budget multi-layer allocation retained by the additive beam."""

    schema: str
    isolated_rank: int
    evidence_sha256: str
    target_widths: tuple[int, ...]
    option_ids: tuple[str, ...]
    proposal_sha256: tuple[str, ...]
    parameters_saved: int
    target_unique_parameters: int
    isolated_causal_score: float
    alternative_sha256: str

    def __post_init__(self) -> None:
        if self.schema != ALTERNATIVE_SCHEMA:
            raise Lfm2InteractionBeamError(
                "Exact allocation alternative schema is invalid."
            )
        object.__setattr__(
            self,
            "isolated_rank",
            _nonnegative_int(self.isolated_rank, "isolated_rank"),
        )
        object.__setattr__(
            self,
            "evidence_sha256",
            _digest(self.evidence_sha256, "evidence_sha256"),
        )
        widths = tuple(
            _positive_int(value, "target_width")
            for value in self.target_widths
        )
        if len(widths) != LFM2_VARIABLE_WIDTH_LAYER_COUNT:
            raise Lfm2InteractionBeamError(
                "An allocation must contain fourteen target widths."
            )
        object.__setattr__(self, "target_widths", widths)
        option_ids = tuple(
            _text(value, "option_id") for value in self.option_ids
        )
        if (
            option_ids != tuple(sorted(option_ids))
            or len(set(option_ids)) != len(option_ids)
        ):
            raise Lfm2InteractionBeamError(
                "Allocation option IDs must be unique and sorted."
            )
        object.__setattr__(self, "option_ids", option_ids)
        proposals = tuple(
            _digest(value, "proposal_sha256")
            for value in self.proposal_sha256
        )
        if len(proposals) != len(option_ids):
            raise Lfm2InteractionBeamError(
                "Every selected option needs one proposal seal."
            )
        object.__setattr__(self, "proposal_sha256", proposals)
        object.__setattr__(
            self,
            "parameters_saved",
            _positive_int(self.parameters_saved, "parameters_saved"),
        )
        object.__setattr__(
            self,
            "target_unique_parameters",
            _positive_int(
                self.target_unique_parameters,
                "target_unique_parameters",
            ),
        )
        score = _finite(self.isolated_causal_score, "isolated_causal_score")
        if score < 0.0:
            raise Lfm2InteractionBeamError(
                "isolated_causal_score must be nonnegative."
            )
        object.__setattr__(self, "isolated_causal_score", score)
        object.__setattr__(
            self,
            "alternative_sha256",
            _digest(self.alternative_sha256, "alternative_sha256"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "isolated_rank": self.isolated_rank,
            "evidence_sha256": self.evidence_sha256,
            "target_widths": list(self.target_widths),
            "option_ids": list(self.option_ids),
            "proposal_sha256": list(self.proposal_sha256),
            "parameters_saved": self.parameters_saved,
            "target_unique_parameters": self.target_unique_parameters,
            "isolated_causal_score": self.isolated_causal_score,
            "alternative_sha256": self.alternative_sha256,
        }


def _alternative_body(
    alternative: Lfm2ExactAllocationAlternative,
) -> dict[str, Any]:
    result = alternative.as_dict()
    result.pop("alternative_sha256", None)
    return result


@dataclass(frozen=True, slots=True)
class Lfm2InteractionBeamPlan:
    """A sealed additive control plus alternative exact-budget allocations."""

    schema: str
    method_id: str
    evidence_sha256: str
    parent_unique_parameters: int
    requested_fraction: float
    target_parameters_saved: int
    target_unique_parameters: int
    realized_fraction: float
    top_k_requested: int
    per_budget_beam: int
    maximum_states: int
    minimum_layer_difference: int
    alternatives: tuple[Lfm2ExactAllocationAlternative, ...]
    unsupported_option_ids: tuple[str, ...]
    plan_sha256: str

    @property
    def control(self) -> Lfm2ExactAllocationAlternative:
        return self.alternatives[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "evidence_sha256": self.evidence_sha256,
            "parent_unique_parameters": self.parent_unique_parameters,
            "requested_fraction": self.requested_fraction,
            "target_parameters_saved": self.target_parameters_saved,
            "target_unique_parameters": self.target_unique_parameters,
            "realized_fraction": self.realized_fraction,
            "top_k_requested": self.top_k_requested,
            "per_budget_beam": self.per_budget_beam,
            "maximum_states": self.maximum_states,
            "minimum_layer_difference": self.minimum_layer_difference,
            "alternatives": [
                alternative.as_dict() for alternative in self.alternatives
            ],
            "unsupported_option_ids": list(self.unsupported_option_ids),
            "plan_sha256": self.plan_sha256,
            "control_is_single_best_additive": True,
            "cumulative_interaction_evaluation_required": True,
            "exact_token_match_objective_used": False,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


def _plan_body(plan: Lfm2InteractionBeamPlan) -> dict[str, Any]:
    result = plan.as_dict()
    result.pop("plan_sha256", None)
    return result


@dataclass(frozen=True, slots=True)
class _BeamState:
    score: float
    widths: tuple[int, ...]
    option_ids: tuple[str, ...]
    proposal_sha256: tuple[str, ...]


def _beam_state_key(state: _BeamState) -> tuple[Any, ...]:
    return (
        state.score,
        state.option_ids,
        state.widths,
        state.proposal_sha256,
    )


def _resolve_target_saving(
    evidence: Lfm2GlobalCausalEvidence,
    *,
    target_fraction: float | None,
    target_parameters_saved: int | None,
    target_unique_parameters: int | None,
) -> tuple[float, int, int]:
    supplied = sum(
        value is not None
        for value in (
            target_fraction,
            target_parameters_saved,
            target_unique_parameters,
        )
    )
    if supplied != 1:
        raise Lfm2InteractionBeamError(
            "Specify exactly one target fraction, saving, or target size."
        )
    parent_parameters = evidence.parent_unique_parameters
    if target_parameters_saved is not None:
        saved = _positive_int(
            target_parameters_saved,
            "target_parameters_saved",
        )
        fraction = saved / parent_parameters
    elif target_unique_parameters is not None:
        target = _positive_int(
            target_unique_parameters,
            "target_unique_parameters",
        )
        if target >= parent_parameters:
            raise Lfm2InteractionBeamError(
                "target_unique_parameters must contract the parent."
            )
        saved = parent_parameters - target
        fraction = saved / parent_parameters
    else:
        fraction = _unit_interval(target_fraction, "target_fraction")
        if fraction <= 0.0 or fraction >= 1.0:
            raise Lfm2InteractionBeamError(
                "target_fraction must be a strict contraction in (0, 1)."
            )
        if fraction in (0.35, 0.50):
            saved = lfm2_global_causal_budget(
                fraction,
                parent_unique_parameters=parent_parameters,
                hidden_size=evidence.variable_width_evidence.hidden_size,
            ).parameters_saved
        else:
            saved = int(math.floor(parent_parameters * fraction + 0.5))
    if saved >= parent_parameters:
        raise Lfm2InteractionBeamError(
            "The requested saving removes the complete parent."
        )
    return fraction, saved, parent_parameters - saved


def _candidate_choices(
    evidence: Lfm2GlobalCausalEvidence,
) -> dict[int, tuple[Lfm2CausalRemovalCandidate, ...]]:
    by_layer: dict[int, list[Lfm2CausalRemovalCandidate]] = {
        index: [] for index in range(LFM2_VARIABLE_WIDTH_LAYER_COUNT)
    }
    for candidate in evidence.candidates:
        if candidate.physically_installable:
            by_layer[candidate.layer_index].append(candidate)
    return {
        index: tuple(
            sorted(
                values,
                key=lambda candidate: (
                    candidate.parameters_saved,
                    candidate.score(evidence.damage_weights),
                    candidate.option_id,
                ),
            )
        )
        for index, values in by_layer.items()
    }


def enumerate_lfm2_interaction_beam_plan(
    evidence: Lfm2GlobalCausalEvidence,
    *,
    target_fraction: float | None = None,
    target_parameters_saved: int | None = None,
    target_unique_parameters: int | None = None,
    top_k: int = 8,
    per_budget_beam: int = 32,
    maximum_states: int = 200_000,
    minimum_layer_difference: int = 1,
) -> Lfm2InteractionBeamPlan:
    """Retain deterministic top-K exact FFN allocations for interaction tests.

    The first alternative is exactly the minimum additive causal-score control.
    Later alternatives have the same physical parameter saving but differ in
    one or more layer widths.  Fractions above the FFN-only physical ceiling
    fail closed; future topology/attention candidates must enter through a
    separately authenticated physical catalog rather than imaginary savings.
    """

    verify_lfm2_global_causal_evidence(evidence)
    if (
        target_fraction is None
        and target_parameters_saved is None
        and target_unique_parameters is None
    ):
        target_fraction = 0.35
    requested_top_k = _positive_int(top_k, "top_k")
    beam_width = _positive_int(per_budget_beam, "per_budget_beam")
    if beam_width < requested_top_k:
        raise Lfm2InteractionBeamError(
            "per_budget_beam must be at least top_k."
        )
    state_limit = _positive_int(maximum_states, "maximum_states")
    layer_difference = _nonnegative_int(
        minimum_layer_difference,
        "minimum_layer_difference",
    )
    if layer_difference > LFM2_VARIABLE_WIDTH_LAYER_COUNT:
        raise Lfm2InteractionBeamError(
            "minimum_layer_difference exceeds the layer count."
        )
    fraction, target_saved, target_parameters = _resolve_target_saving(
        evidence,
        target_fraction=target_fraction,
        target_parameters_saved=target_parameters_saved,
        target_unique_parameters=target_unique_parameters,
    )
    source_width = evidence.variable_width_evidence.source_intermediate_width
    choices_by_layer = _candidate_choices(evidence)
    states: dict[int, list[_BeamState]] = {
        0: [
            _BeamState(
                score=0.0,
                widths=(),
                option_ids=(),
                proposal_sha256=(),
            )
        ]
    }
    for layer_index in range(LFM2_VARIABLE_WIDTH_LAYER_COUNT):
        choices: list[
            tuple[int, int, float, str | None, str | None]
        ] = [(0, source_width, 0.0, None, None)]
        for candidate in choices_by_layer[layer_index]:
            choices.append(
                (
                    candidate.parameters_saved,
                    candidate.target_units,
                    candidate.score(evidence.damage_weights),
                    candidate.option_id,
                    candidate.proposal_sha256,
                )
            )
        updated: dict[int, list[_BeamState]] = {}
        for saved_before, bucket in states.items():
            for state in bucket:
                for saved, width, score, option_id, proposal_sha in choices:
                    cumulative = saved_before + saved
                    if cumulative > target_saved:
                        continue
                    option_ids = (
                        state.option_ids
                        if option_id is None
                        else state.option_ids + (option_id,)
                    )
                    proposal_hashes = (
                        state.proposal_sha256
                        if proposal_sha is None
                        else state.proposal_sha256 + (proposal_sha,)
                    )
                    candidate_state = _BeamState(
                        score=state.score + score,
                        widths=state.widths + (width,),
                        option_ids=option_ids,
                        proposal_sha256=proposal_hashes,
                    )
                    destination = updated.setdefault(cumulative, [])
                    destination.append(candidate_state)
        for saved, bucket in tuple(updated.items()):
            unique = {}
            for value in sorted(bucket, key=_beam_state_key):
                unique.setdefault((value.widths, value.option_ids), value)
            updated[saved] = sorted(
                unique.values(),
                key=_beam_state_key,
            )[:beam_width]
        retained_states = sum(len(bucket) for bucket in updated.values())
        if retained_states > state_limit:
            raise Lfm2InteractionBeamInfeasible(
                "Top-K exact allocation exceeded the immutable state ceiling."
            )
        if not updated:
            raise Lfm2InteractionBeamInfeasible(
                f"No physical state remains after layer {layer_index}."
            )
        states = updated
    exact = states.get(target_saved, [])
    if not exact:
        maximum_represented = max(states)
        raise Lfm2InteractionBeamInfeasible(
            "No exact FFN allocation reaches the requested saving "
            f"{target_saved}; maximum represented saving is "
            f"{maximum_represented}. Attention/depth/conv/vocabulary savings "
            "were not counted without physical payloads."
        )
    selected_states: list[_BeamState] = []
    for state in sorted(exact, key=_beam_state_key):
        if any(
            sum(
                left != right
                for left, right in zip(
                    state.widths,
                    prior.widths,
                    strict=True,
                )
            )
            < layer_difference
            for prior in selected_states
        ):
            continue
        selected_states.append(state)
        if len(selected_states) == requested_top_k:
            break
    if not selected_states:
        raise Lfm2InteractionBeamInfeasible(
            "The diversity gate removed every exact allocation."
        )
    alternatives = []
    for rank, state in enumerate(selected_states):
        alternative = Lfm2ExactAllocationAlternative(
            schema=ALTERNATIVE_SCHEMA,
            isolated_rank=rank,
            evidence_sha256=evidence.evidence_sha256,
            target_widths=state.widths,
            option_ids=tuple(sorted(state.option_ids)),
            proposal_sha256=tuple(
                proposal
                for _option, proposal in sorted(
                    zip(
                        state.option_ids,
                        state.proposal_sha256,
                        strict=True,
                    )
                )
            ),
            parameters_saved=target_saved,
            target_unique_parameters=target_parameters,
            isolated_causal_score=state.score,
            alternative_sha256="0" * 64,
        )
        alternatives.append(
            replace(
                alternative,
                alternative_sha256=_json_sha256(
                    _alternative_body(alternative)
                ),
            )
        )
    plan = Lfm2InteractionBeamPlan(
        schema=PLAN_SCHEMA,
        method_id=METHOD_ID,
        evidence_sha256=evidence.evidence_sha256,
        parent_unique_parameters=evidence.parent_unique_parameters,
        requested_fraction=fraction,
        target_parameters_saved=target_saved,
        target_unique_parameters=target_parameters,
        realized_fraction=target_saved / evidence.parent_unique_parameters,
        top_k_requested=requested_top_k,
        per_budget_beam=beam_width,
        maximum_states=state_limit,
        minimum_layer_difference=layer_difference,
        alternatives=tuple(alternatives),
        unsupported_option_ids=tuple(
            sorted(
                candidate.option_id
                for candidate in evidence.candidates
                if not candidate.physically_installable
            )
        ),
        plan_sha256="0" * 64,
    )
    sealed = replace(plan, plan_sha256=_json_sha256(_plan_body(plan)))
    return verify_lfm2_interaction_beam_plan(sealed, evidence)


def verify_lfm2_interaction_beam_plan(
    plan: Lfm2InteractionBeamPlan,
    evidence: Lfm2GlobalCausalEvidence,
) -> Lfm2InteractionBeamPlan:
    """Reauthenticate exact accounting and every selected proposal binding."""

    verify_lfm2_global_causal_evidence(evidence)
    if (
        not isinstance(plan, Lfm2InteractionBeamPlan)
        or plan.schema != PLAN_SCHEMA
        or plan.method_id != METHOD_ID
        or plan.evidence_sha256 != evidence.evidence_sha256
        or plan.parent_unique_parameters != evidence.parent_unique_parameters
        or not plan.alternatives
        or _json_sha256(_plan_body(plan)) != plan.plan_sha256
    ):
        raise Lfm2InteractionBeamError(
            "The interaction-beam plan seal or evidence binding is invalid."
        )
    candidates = {
        candidate.option_id: candidate
        for candidate in evidence.candidates
        if candidate.physically_installable
    }
    source_width = evidence.variable_width_evidence.source_intermediate_width
    seen = set()
    for expected_rank, alternative in enumerate(plan.alternatives):
        if (
            alternative.isolated_rank != expected_rank
            or alternative.evidence_sha256 != evidence.evidence_sha256
            or alternative.parameters_saved != plan.target_parameters_saved
            or alternative.target_unique_parameters
            != plan.target_unique_parameters
            or _json_sha256(_alternative_body(alternative))
            != alternative.alternative_sha256
            or alternative.alternative_sha256 in seen
        ):
            raise Lfm2InteractionBeamError(
                "An exact allocation alternative seal is invalid."
            )
        seen.add(alternative.alternative_sha256)
        selected = []
        widths = [source_width] * LFM2_VARIABLE_WIDTH_LAYER_COUNT
        saved = 0
        score = 0.0
        proposal_by_option = dict(
            zip(
                alternative.option_ids,
                alternative.proposal_sha256,
                strict=True,
            )
        )
        for option_id in alternative.option_ids:
            candidate = candidates.get(option_id)
            if candidate is None:
                raise Lfm2InteractionBeamError(
                    f"Alternative references unknown option {option_id!r}."
                )
            if widths[candidate.layer_index] != source_width:
                raise Lfm2InteractionBeamError(
                    "An alternative selects multiple widths for one layer."
                )
            if proposal_by_option[option_id] != candidate.proposal_sha256:
                raise Lfm2InteractionBeamError(
                    "An alternative proposal seal differs from the evidence."
                )
            widths[candidate.layer_index] = candidate.target_units
            saved += candidate.parameters_saved
            score += candidate.score(evidence.damage_weights)
            selected.append(option_id)
        if (
            tuple(widths) != alternative.target_widths
            or saved != alternative.parameters_saved
            or tuple(sorted(selected)) != alternative.option_ids
            or not math.isclose(
                score,
                alternative.isolated_causal_score,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise Lfm2InteractionBeamError(
                "An alternative disagrees with its exact physical candidates."
            )
    if (
        plan.target_unique_parameters
        != plan.parent_unique_parameters - plan.target_parameters_saved
        or not math.isclose(
            plan.realized_fraction,
            plan.target_parameters_saved / plan.parent_unique_parameters,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or tuple(sorted(plan.unsupported_option_ids))
        != plan.unsupported_option_ids
    ):
        raise Lfm2InteractionBeamError(
            "The interaction-beam target accounting is invalid."
        )
    return plan


@dataclass(frozen=True, slots=True)
class Lfm2InteractionMaterialization:
    """One exact bundle plus its canonical proposal-pack provenance."""

    schema: str
    bundle: Lfm2VariableWidthFixedPlanBundle = field(repr=False)
    canonical_option_ids: tuple[str, ...] = ()
    canonical_pack_manifest_sha256: tuple[str, ...] = ()
    canonical_pack_tensor_sha256: tuple[str, ...] = ()
    clustering_calls: int = 0

    def __post_init__(self) -> None:
        if self.schema != MATERIALIZATION_SCHEMA:
            raise Lfm2InteractionBeamError(
                "Interaction materialization schema is invalid."
            )
        option_ids = tuple(
            _text(value, "canonical_option_id")
            for value in self.canonical_option_ids
        )
        if (
            option_ids != tuple(sorted(option_ids))
            or len(set(option_ids)) != len(option_ids)
        ):
            raise Lfm2InteractionBeamError(
                "Canonical option IDs must be unique and sorted."
            )
        object.__setattr__(self, "canonical_option_ids", option_ids)
        manifests = tuple(
            _digest(value, "canonical_pack_manifest_sha256")
            for value in self.canonical_pack_manifest_sha256
        )
        tensors = tuple(
            _digest(value, "canonical_pack_tensor_sha256")
            for value in self.canonical_pack_tensor_sha256
        )
        if len(manifests) != len(option_ids) or len(tensors) != len(option_ids):
            raise Lfm2InteractionBeamError(
                "Every canonical option needs manifest and tensor seals."
            )
        object.__setattr__(
            self,
            "canonical_pack_manifest_sha256",
            manifests,
        )
        object.__setattr__(
            self,
            "canonical_pack_tensor_sha256",
            tensors,
        )
        object.__setattr__(
            self,
            "clustering_calls",
            _nonnegative_int(self.clustering_calls, "clustering_calls"),
        )


def materialize_lfm2_canonical_multi_pack_allocation(
    parent: Any,
    evidence: Lfm2GlobalCausalEvidence,
    alternative: Lfm2ExactAllocationAlternative,
    packs: Mapping[str, Lfm2CanonicalProposalPack],
) -> Lfm2InteractionMaterialization:
    """Combine exact changed-layer packs without any new clustering.

    Each canonical pack contains one isolated changed layer.  This function
    takes the changed layer from every selected option, clones unchanged layers
    from the authenticated parent, and seals the complete multi-layer bundle.
    It never selects medoids, fits assignments, aggregates consumers, or calls
    the clustering implementation.
    """

    verify_lfm2_global_causal_evidence(evidence, parent=parent)
    if (
        not isinstance(alternative, Lfm2ExactAllocationAlternative)
        or alternative.evidence_sha256 != evidence.evidence_sha256
    ):
        raise Lfm2InteractionBeamError(
            "Canonical materialization received the wrong alternative."
        )
    if not isinstance(packs, Mapping):
        raise Lfm2InteractionBeamError(
            "packs must map exact option IDs to canonical packs."
        )
    supplied_ids = tuple(sorted(packs))
    if supplied_ids != alternative.option_ids:
        raise Lfm2InteractionBeamError(
            "Canonical packs do not exactly cover the selected options."
        )
    candidate_by_id = {
        candidate.option_id: candidate
        for candidate in evidence.candidates
        if candidate.physically_installable
    }
    canonical_by_layer: dict[int, Lfm2VariableWidthLayerPlan] = {}
    canonical_requires_grad: (
        tuple[tuple[bool, bool, bool], ...] | None
    ) = None
    manifest_sha = []
    tensor_sha = []
    for option_id in alternative.option_ids:
        pack = packs[option_id]
        if not isinstance(pack, Lfm2CanonicalProposalPack):
            raise Lfm2InteractionBeamError(
                f"Canonical option {option_id!r} has the wrong pack type."
            )
        verify_lfm2_variable_width_fixed_plan_bundle(
            pack.bundle,
            parent=parent,
        )
        candidate = candidate_by_id.get(option_id)
        if candidate is None:
            raise Lfm2InteractionBeamError(
                f"Canonical option {option_id!r} is absent from evidence."
            )
        layer_index = candidate.layer_index
        layer_plan = pack.bundle.plan.layer_plans[layer_index]
        contracted = tuple(
            value
            for value in pack.bundle.plan.layer_plans
            if value.target_width != value.source_width
        )
        pack_requires_grad = tuple(
            tuple(bool(flag) for flag in value.requires_grad)
            for value in pack.bundle.plan.layer_plans
        )
        if (
            pack.proposal.option_id != option_id
            or pack.proposal.proposal_sha256
            != candidate.proposal_sha256
            or pack.proposal.layer_binding_sha256
            != candidate.layer_binding_sha256
            or pack.proposal.medoid_source_neurons
            != candidate.retained_source_units
            or len(contracted) != 1
            or contracted[0] is not layer_plan
            or layer_plan.target_width != candidate.target_units
            or layer_index in canonical_by_layer
        ):
            raise Lfm2InteractionBeamError(
                f"Canonical pack {option_id!r} changed its proposal payload."
            )
        if canonical_requires_grad is None:
            canonical_requires_grad = pack_requires_grad
        elif canonical_requires_grad != pack_requires_grad:
            raise Lfm2InteractionBeamError(
                "Canonical packs disagree on source requires-grad metadata."
            )
        canonical_by_layer[layer_index] = layer_plan
        manifest_sha.append(
            _digest(
                pack.manifest["manifest_sha256"],
                "canonical_pack_manifest_sha256",
            )
        )
        tensor_sha.append(
            _digest(
                pack.manifest["tensor_file_sha256"],
                "canonical_pack_tensor_sha256",
            )
        )

    if canonical_requires_grad is None:
        raise Lfm2InteractionBeamError(
            "Canonical allocation did not provide source metadata."
        )
    variable_evidence = evidence.variable_width_evidence
    widths = alternative.target_widths
    source_width = variable_evidence.source_intermediate_width
    fixed_curves = []
    for layer_index, (curve, target_width) in enumerate(
        zip(variable_evidence.curves, widths, strict=True)
    ):
        by_width = {point.width: point for point in curve.points}
        if target_width not in by_width:
            raise Lfm2InteractionBeamError(
                f"Layer {layer_index} lacks target width {target_width}."
            )
        source_point = by_width[source_width]
        points = (
            (source_point,)
            if target_width == source_width
            else (source_point, by_width[target_width])
        )
        fixed_curves.append(
            Lfm2LayerWidthCurve(
                layer_index=layer_index,
                points=points,
            )
        )
    widths_sha = variable_width_fold_module._json_sha256(
        {
            "source_evidence_sha256": variable_evidence.evidence_sha256,
            "target_widths": list(widths),
        }
    )
    fixed = replace(
        variable_evidence,
        curves=tuple(fixed_curves),
        evidence_builder_id=(
            f"{variable_evidence.evidence_builder_id}:fixed-widths:"
            f"{widths_sha}"
        ),
        evidence_sha256="",
    )
    fixed = replace(
        fixed,
        evidence_sha256=variable_width_fold_module._json_sha256(
            variable_width_fold_module._evidence_body(fixed)
        ),
    )
    torch = variable_width_fold_module._torch()
    ridge = variable_width_fold_module._finite_nonnegative(
        evidence.cluster_ridge,
        "cluster_ridge",
    )
    iterations = variable_width_fold_module._positive_int(
        evidence.cluster_max_iterations,
        "cluster_max_iterations",
    )
    chunk_size = variable_width_fold_module._positive_int(
        evidence.cluster_distance_chunk_size,
        "cluster_distance_chunk_size",
    )
    with variable_width_fold_module._TRANSACTION_LOCK, torch.no_grad():
        (
            layers,
            hidden,
            validated_source_width,
            kinds,
        ) = variable_width_fold_module._validate_live_evidence(
            parent,
            fixed,
        )
        parent_before = variable_width_fold_module._parent_identity(parent)
        parameters_saved = sum(
            (validated_source_width - width) * 3 * hidden
            for width in widths
        )
        maximum_step = max(
            1,
            max(validated_source_width - width for width in widths),
        )
        allocation = allocate_lfm2_variable_widths(
            fixed,
            target_parameters_saved=parameters_saved,
            minimum_width=min(widths),
            maximum_width_step=maximum_step,
            whole_trajectory_weight=1.0,
            residual_weight=1.0,
        )
        if (
            allocation.target_widths != widths
            or parameters_saved != alternative.parameters_saved
        ):
            raise Lfm2InteractionBeamError(
                "Canonical multi-pack allocation lost its exact budget."
            )
        layer_plans = []
        for layer_index, (layer, target_width) in enumerate(
            zip(layers, widths, strict=True)
        ):
            gate_source = layer.feed_forward.w1.weight
            up_source = layer.feed_forward.w3.weight
            down_source = layer.feed_forward.w2.weight
            live_requires_grad = (
                bool(gate_source.requires_grad),
                bool(up_source.requires_grad),
                bool(down_source.requires_grad),
            )
            expected_requires_grad = canonical_requires_grad[layer_index]
            if live_requires_grad not in (
                expected_requires_grad,
                (False, False, False),
            ):
                raise Lfm2InteractionBeamError(
                    "Live parent requires-grad metadata differs from every "
                    "canonical source pack."
                )
            canonical = canonical_by_layer.get(layer_index)
            if canonical is not None:
                if (
                    target_width == validated_source_width
                    or canonical.source_width != validated_source_width
                    or canonical.target_width != target_width
                    or canonical.requires_grad != expected_requires_grad
                ):
                    raise Lfm2InteractionBeamError(
                        "Canonical layer metadata differs from the parent."
                    )
                layer_plan = Lfm2VariableWidthLayerPlan(
                    layer_index=layer_index,
                    source_width=validated_source_width,
                    target_width=target_width,
                    medoid_source_neurons=tuple(
                        canonical.medoid_source_neurons
                    ),
                    source_to_target=tuple(canonical.source_to_target),
                    consumer_coefficients=tuple(
                        canonical.consumer_coefficients
                    ),
                    gate_weight=variable_width_fold_module._clone_cpu(
                        canonical.gate_weight,
                        f"canonical.multi.layer.{layer_index}.gate",
                    ),
                    up_weight=variable_width_fold_module._clone_cpu(
                        canonical.up_weight,
                        f"canonical.multi.layer.{layer_index}.up",
                    ),
                    down_weight=variable_width_fold_module._clone_cpu(
                        canonical.down_weight,
                        f"canonical.multi.layer.{layer_index}.down",
                    ),
                    requires_grad=expected_requires_grad,
                )
            else:
                if target_width != validated_source_width:
                    raise Lfm2InteractionBeamError(
                        "A contracted layer is missing its canonical pack."
                    )
                layer_plan = Lfm2VariableWidthLayerPlan(
                    layer_index=layer_index,
                    source_width=validated_source_width,
                    target_width=validated_source_width,
                    medoid_source_neurons=tuple(
                        range(validated_source_width)
                    ),
                    source_to_target=tuple(range(validated_source_width)),
                    consumer_coefficients=(1.0,) * validated_source_width,
                    gate_weight=variable_width_fold_module._clone_cpu(
                        gate_source,
                        f"canonical.multi.layer.{layer_index}.gate",
                    ),
                    up_weight=variable_width_fold_module._clone_cpu(
                        up_source,
                        f"canonical.multi.layer.{layer_index}.up",
                    ),
                    down_weight=variable_width_fold_module._clone_cpu(
                        down_source,
                        f"canonical.multi.layer.{layer_index}.down",
                    ),
                    requires_grad=expected_requires_grad,
                )
            layer_plans.append(layer_plan)
        source_parameters = variable_width_fold_module._unique_parameter_count(
            parent
        )
        plan = Lfm2VariableWidthFoldPlan(
            schema=variable_width_fold_module.PLAN_SCHEMA,
            method_id=variable_width_fold_module.METHOD_ID,
            parent_identity_sha256=(
                variable_evidence.parent_identity_sha256
            ),
            evidence_sha256=fixed.evidence_sha256,
            allocation=allocation,
            layer_plans=tuple(layer_plans),
            source_unique_parameters=source_parameters,
            target_unique_parameters=source_parameters - parameters_saved,
            parameters_saved=parameters_saved,
            hidden_size=hidden,
            source_intermediate_width=validated_source_width,
            layer_kinds=kinds,
            cluster_ridge=ridge,
            cluster_max_iterations=iterations,
            cluster_distance_chunk_size=chunk_size,
            plan_sha256="",
        )
        plan = replace(
            plan,
            plan_sha256=variable_width_fold_module._json_sha256(
                variable_width_fold_module._plan_body(plan)
            ),
        )
        variable_width_fold_module._validate_plan_seal(plan)
        layer_bindings = tuple(
            fingerprint_lfm2_variable_width_layer_plan(layer_plan)
            for layer_plan in plan.layer_plans
        )
        bundle = Lfm2VariableWidthFixedPlanBundle(
            schema=variable_width_fold_module.FIXED_WIDTH_BUNDLE_SCHEMA,
            source_evidence_sha256=variable_evidence.evidence_sha256,
            fixed_evidence=fixed,
            plan=plan,
            layer_binding_sha256=layer_bindings,
            bundle_sha256="",
        )
        bundle = replace(
            bundle,
            bundle_sha256=variable_width_fold_module._json_sha256(
                variable_width_fold_module._fixed_bundle_body(bundle)
            ),
        )
        verify_lfm2_variable_width_fixed_plan_bundle(
            bundle,
            parent=parent,
        )
        if (
            bundle.plan.target_unique_parameters
            != alternative.target_unique_parameters
            or variable_width_fold_module._parent_identity(parent)
            != parent_before
        ):
            raise Lfm2InteractionBeamError(
                "Canonical multi-pack materialization changed accounting "
                "or mutated the parent."
            )
    return Lfm2InteractionMaterialization(
        schema=MATERIALIZATION_SCHEMA,
        bundle=bundle,
        canonical_option_ids=alternative.option_ids,
        canonical_pack_manifest_sha256=tuple(manifest_sha),
        canonical_pack_tensor_sha256=tuple(tensor_sha),
        clustering_calls=0,
    )


def seal_lfm2_global_causal_evidence_from_canonical_packs(
    parent: Any,
    variable_width_evidence: Any,
    candidates: Sequence[Lfm2CausalRemovalCandidate],
    packs: Mapping[str, Lfm2CanonicalProposalPack],
    *,
    parent_output_envelope_sha256: str,
    task_anchor_manifest_sha256: str,
    damage_weights: Lfm2CausalDamageWeights | None = None,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    parent_cache_build_count: int = 0,
) -> Lfm2GlobalCausalEvidence:
    """Seal measured evidence against canonical packs, never reclustering."""

    variable_width_fold_module.verify_lfm2_variable_width_evidence(
        variable_width_evidence,
        parent=parent,
    )
    selected_weights = (
        Lfm2CausalDamageWeights() if damage_weights is None else damage_weights
    )
    if not isinstance(selected_weights, Lfm2CausalDamageWeights):
        raise Lfm2InteractionBeamError(
            "damage_weights must be Lfm2CausalDamageWeights or None."
        )
    ridge, iterations, chunk = global_allocator_module._cluster_values(
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
        or len({candidate.option_id for candidate in ordered})
        != len(ordered)
    ):
        raise Lfm2InteractionBeamError(
            "Canonical candidates must be nonempty, typed, and unique."
        )
    physical_ids = tuple(
        candidate.option_id
        for candidate in ordered
        if candidate.physically_installable
    )
    if tuple(sorted(packs)) != physical_ids:
        raise Lfm2InteractionBeamError(
            "Canonical packs do not exactly cover measured physical options."
        )
    seen_coordinates = set()
    for candidate in ordered:
        if not candidate.physically_installable:
            continue
        coordinate = (candidate.layer_index, candidate.target_units)
        if coordinate in seen_coordinates:
            raise Lfm2InteractionBeamError(
                "Canonical evidence has duplicate layer/width options."
            )
        seen_coordinates.add(coordinate)
        expected_saved = (
            (candidate.source_units - candidate.target_units)
            * 3
            * variable_width_evidence.hidden_size
        )
        pack = packs[candidate.option_id]
        verify_lfm2_variable_width_fixed_plan_bundle(
            pack.bundle,
            parent=parent,
        )
        if (
            candidate.source_units
            != variable_width_evidence.source_intermediate_width
            or candidate.parameters_saved != expected_saved
            or pack.proposal.option_id != candidate.option_id
            or pack.proposal.proposal_sha256
            != candidate.proposal_sha256
            or pack.proposal.layer_binding_sha256
            != candidate.layer_binding_sha256
            or pack.proposal.medoid_source_neurons
            != candidate.retained_source_units
            or pack.proposal.source_evidence_sha256
            != variable_width_evidence.evidence_sha256
        ):
            raise Lfm2InteractionBeamError(
                f"Measured option {candidate.option_id!r} differs from its "
                "canonical physical pack."
            )
    cache_builds = _nonnegative_int(
        parent_cache_build_count,
        "parent_cache_build_count",
    )
    if cache_builds > 1:
        raise Lfm2InteractionBeamError(
            "Parent evidence may be cached once, never rebuilt per option."
        )
    evidence = Lfm2GlobalCausalEvidence(
        schema=global_allocator_module.EVIDENCE_SCHEMA,
        parent_identity_sha256=(
            variable_width_evidence.parent_identity_sha256
        ),
        parent_unique_parameters=(
            global_allocator_module._unique_parameter_count(parent)
        ),
        parent_trace_cache_key=variable_width_evidence.cache_key,
        parent_corpus_sha256=(
            variable_width_evidence.heldout_corpus_sha256
        ),
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
        activation_jvp_calls=0,
        activation_jvp_manifest_sha256=None,
        evidence_sha256="",
    )
    sealed = replace(
        evidence,
        evidence_sha256=global_allocator_module._json_sha256(
            global_allocator_module._evidence_body(evidence)
        ),
    )
    return verify_lfm2_global_causal_evidence(sealed, parent=parent)


def _canonical_stage_from_state(
    parent: Any,
    evidence: Lfm2GlobalCausalEvidence,
    state: Any,
    packs: Mapping[str, Lfm2CanonicalProposalPack],
    *,
    stage_index: int,
) -> Lfm2GlobalCausalStage:
    candidate_by_id = {
        candidate.option_id: candidate
        for candidate in evidence.candidates
        if candidate.physically_installable
    }
    option_ids = tuple(sorted(state.option_ids))
    proposal_hashes = tuple(
        candidate_by_id[option_id].proposal_sha256
        for option_id in option_ids
    )
    alternative = Lfm2ExactAllocationAlternative(
        schema=ALTERNATIVE_SCHEMA,
        isolated_rank=stage_index,
        evidence_sha256=evidence.evidence_sha256,
        target_widths=tuple(state.widths),
        option_ids=option_ids,
        proposal_sha256=proposal_hashes,
        parameters_saved=sum(
            candidate_by_id[option_id].parameters_saved
            for option_id in option_ids
        ),
        target_unique_parameters=(
            evidence.parent_unique_parameters
            - sum(
                candidate_by_id[option_id].parameters_saved
                for option_id in option_ids
            )
        ),
        isolated_causal_score=state.score,
        alternative_sha256="0" * 64,
    )
    alternative = replace(
        alternative,
        alternative_sha256=_json_sha256(
            _alternative_body(alternative)
        ),
    )
    materialization = materialize_lfm2_canonical_multi_pack_allocation(
        parent,
        evidence,
        alternative,
        {option_id: packs[option_id] for option_id in option_ids},
    )
    bundle = materialization.bundle
    stage = Lfm2GlobalCausalStage(
        schema=global_allocator_module.STAGE_SCHEMA,
        stage_index=stage_index,
        cumulative_parameters_saved=bundle.plan.parameters_saved,
        cumulative_fraction=(
            bundle.plan.parameters_saved
            / evidence.parent_unique_parameters
        ),
        target_widths=tuple(state.widths),
        option_ids=option_ids,
        cumulative_causal_score=state.score,
        physical_bundle=bundle,
        stage_sha256="",
    )
    return replace(
        stage,
        stage_sha256=global_allocator_module._json_sha256(
            global_allocator_module._stage_body(stage)
        ),
    )


def plan_lfm2_global_causal_target_from_canonical_packs(
    parent: Any,
    evidence: Lfm2GlobalCausalEvidence,
    packs: Mapping[str, Lfm2CanonicalProposalPack],
    *,
    target_fraction: float,
    maximum_stage_fraction: float = 0.10,
    max_states: int = 100_000,
) -> Lfm2GlobalCausalPlan:
    """Build exact staged causal plans from packs with zero reclustering."""

    verify_lfm2_global_causal_evidence(evidence, parent=parent)
    stage_fraction = global_allocator_module._finite_nonnegative(
        maximum_stage_fraction,
        "maximum_stage_fraction",
    )
    if stage_fraction <= 0.0:
        raise Lfm2InteractionBeamInfeasible(
            "maximum_stage_fraction must be positive."
        )
    state_limit = _positive_int(max_states, "max_states")
    budget = lfm2_global_causal_budget(
        target_fraction,
        parent_unique_parameters=evidence.parent_unique_parameters,
        hidden_size=evidence.variable_width_evidence.hidden_size,
    )
    states = global_allocator_module._allocation_states(
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
    final_widths = tuple(final_state.widths)
    source_width = evidence.variable_width_evidence.source_intermediate_width
    previous_widths = (
        source_width,
    ) * LFM2_VARIABLE_WIDTH_LAYER_COUNT
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
    stage_states = []
    while previous_saved + stage_limit < budget.parameters_saved:
        desired = previous_saved + stage_limit
        bounded = global_allocator_module._allocation_states(
            evidence,
            maximum_parameters_saved=desired,
            lower_widths=final_widths,
            upper_widths=previous_widths,
            max_states=state_limit,
        )
        reachable = [
            saved
            for saved in bounded
            if previous_saved < saved <= desired
        ]
        if not reachable:
            break
        next_saved = max(reachable)
        next_state = bounded[next_saved]
        if next_state.widths == previous_widths:
            break
        stage_states.append(next_state)
        previous_widths = tuple(next_state.widths)
        previous_saved = next_saved
    if not stage_states or tuple(stage_states[-1].widths) != final_widths:
        stage_states.append(final_state)
    stages = tuple(
        _canonical_stage_from_state(
            parent,
            evidence,
            state,
            packs,
            stage_index=index,
        )
        for index, state in enumerate(stage_states)
    )
    if (
        stages[-1].cumulative_parameters_saved
        != budget.parameters_saved
        or stages[-1].target_widths != final_widths
        or any(
            left.cumulative_parameters_saved
            >= right.cumulative_parameters_saved
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
        raise Lfm2InteractionBeamInfeasible(
            "Canonical sequential stages are not monotonic and exact."
        )
    plan = Lfm2GlobalCausalPlan(
        schema=global_allocator_module.PLAN_SCHEMA,
        method_id=global_allocator_module.METHOD_ID,
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
    sealed = replace(
        plan,
        plan_sha256=global_allocator_module._json_sha256(
            global_allocator_module._plan_body(plan)
        ),
    )
    return verify_lfm2_global_causal_plan(
        sealed,
        evidence,
        parent=parent,
    )


def build_and_seal_lfm2_cached_causal_frontier_from_canonical_packs(
    parent: Any,
    variable_width_evidence: Any,
    candidates: Sequence[Lfm2CausalRemovalCandidate],
    split: Any,
    widths_by_layer: Sequence[Sequence[int]],
    measurement_reports: Sequence[Mapping[str, Any]],
    *,
    canonical_packs: Mapping[str, Lfm2CanonicalProposalPack],
    parent_output_envelope_sha256: str,
    parent_output_file_sha256: str,
    task_anchor_manifest_sha256: str,
    task_anchor_manifest: Mapping[str, Any],
    tokenizer_fingerprint: str,
    damage_weights: Lfm2CausalDamageWeights | None = None,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    maximum_stage_fraction: float = 0.10,
    max_states: int = 100_000,
) -> tuple[Lfm2GlobalCausalEvidence, Lfm2GlobalCausalPlan, dict[str, Any]]:
    """Canonical-pack replacement for the reclustering merge finalizer."""

    from neural_forge.hf_lfm2_cached_causal_evidence import (
        seal_lfm2_cached_causal_frontier_document,
    )

    evidence = seal_lfm2_global_causal_evidence_from_canonical_packs(
        parent,
        variable_width_evidence,
        candidates,
        canonical_packs,
        parent_output_envelope_sha256=parent_output_envelope_sha256,
        task_anchor_manifest_sha256=task_anchor_manifest_sha256,
        damage_weights=damage_weights,
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
        parent_cache_build_count=0,
    )
    exact = plan_lfm2_global_causal_target_from_canonical_packs(
        parent,
        evidence,
        canonical_packs,
        target_fraction=0.35,
        maximum_stage_fraction=maximum_stage_fraction,
        max_states=max_states,
    )
    try:
        plan_lfm2_global_causal_target_from_canonical_packs(
            parent,
            evidence,
            canonical_packs,
            target_fraction=0.50,
            maximum_stage_fraction=maximum_stage_fraction,
            max_states=max_states,
        )
    except Lfm2GlobalCausalInfeasible as exc:
        fifty_reason = str(exc)
    else:
        raise Lfm2InteractionBeamError(
            "Canonical FFN frontier unexpectedly authorized 50%."
        )
    document = seal_lfm2_cached_causal_frontier_document(
        evidence,
        split,
        widths_by_layer,
        measurement_reports,
        exact,
        parent_output_file_sha256=parent_output_file_sha256,
        tokenizer_fingerprint=tokenizer_fingerprint,
        task_anchor_manifest=task_anchor_manifest,
        fifty_percent_infeasible_reason=fifty_reason,
    )
    return evidence, exact, document


@dataclass(frozen=True, slots=True)
class Lfm2BoundExactAllocation:
    """An allocation bound to one complete installable physical bundle."""

    schema: str
    alternative: Lfm2ExactAllocationAlternative
    physical_bundle: Lfm2VariableWidthFixedPlanBundle = field(repr=False)
    canonical_payload_required: bool
    canonical_option_ids: tuple[str, ...]
    canonical_pack_manifest_sha256: tuple[str, ...]
    canonical_pack_tensor_sha256: tuple[str, ...]
    binding_guard_sha256: str
    bound_alternative_sha256: str

    def __post_init__(self) -> None:
        if self.schema != BOUND_ALTERNATIVE_SCHEMA:
            raise Lfm2InteractionBeamError(
                "Bound exact allocation schema is invalid."
            )
        if not isinstance(
            self.alternative,
            Lfm2ExactAllocationAlternative,
        ):
            raise Lfm2InteractionBeamError(
                "Bound allocation lost its structural alternative."
            )
        if not isinstance(self.canonical_payload_required, bool):
            raise Lfm2InteractionBeamError(
                "canonical_payload_required must be boolean."
            )
        object.__setattr__(
            self,
            "canonical_option_ids",
            tuple(
                _text(value, "canonical_option_id")
                for value in self.canonical_option_ids
            ),
        )
        object.__setattr__(
            self,
            "canonical_pack_manifest_sha256",
            tuple(
                _digest(value, "canonical_pack_manifest_sha256")
                for value in self.canonical_pack_manifest_sha256
            ),
        )
        object.__setattr__(
            self,
            "canonical_pack_tensor_sha256",
            tuple(
                _digest(value, "canonical_pack_tensor_sha256")
                for value in self.canonical_pack_tensor_sha256
            ),
        )
        object.__setattr__(
            self,
            "binding_guard_sha256",
            _digest(self.binding_guard_sha256, "binding_guard_sha256"),
        )
        object.__setattr__(
            self,
            "bound_alternative_sha256",
            _digest(
                self.bound_alternative_sha256,
                "bound_alternative_sha256",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "alternative": self.alternative.as_dict(),
            "physical_bundle_sha256": self.physical_bundle.bundle_sha256,
            "physical_plan_sha256": self.physical_bundle.plan.plan_sha256,
            "physical_fixed_evidence_sha256": (
                self.physical_bundle.fixed_evidence.evidence_sha256
            ),
            "physical_layer_binding_sha256": list(
                self.physical_bundle.layer_binding_sha256
            ),
            "canonical_payload_required": self.canonical_payload_required,
            "canonical_option_ids": list(self.canonical_option_ids),
            "canonical_pack_manifest_sha256": list(
                self.canonical_pack_manifest_sha256
            ),
            "canonical_pack_tensor_sha256": list(
                self.canonical_pack_tensor_sha256
            ),
            "binding_guard_sha256": self.binding_guard_sha256,
            "bound_alternative_sha256": self.bound_alternative_sha256,
            "clustering_calls": 0,
            "parent_forward_calls": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


def _bound_alternative_body(
    alternative: Lfm2BoundExactAllocation,
) -> dict[str, Any]:
    result = alternative.as_dict()
    result.pop("bound_alternative_sha256", None)
    return result


@dataclass(frozen=True, slots=True)
class Lfm2BoundInteractionBeamPlan:
    """A structural beam whose every alternative has an authenticated bundle."""

    schema: str
    structural_plan: Lfm2InteractionBeamPlan
    alternatives: tuple[Lfm2BoundExactAllocation, ...]
    canonical_payload_required: bool
    bound_plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "structural_plan": self.structural_plan.as_dict(),
            "alternatives": [
                alternative.as_dict() for alternative in self.alternatives
            ],
            "canonical_payload_required": self.canonical_payload_required,
            "bound_plan_sha256": self.bound_plan_sha256,
            "cumulative_physical_children_required": True,
            "exact_token_match_objective_used": False,
            "parent_forward_calls": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


def _bound_plan_body(plan: Lfm2BoundInteractionBeamPlan) -> dict[str, Any]:
    result = plan.as_dict()
    result.pop("bound_plan_sha256", None)
    return result


def _guarded_parent_silent(
    parent: Any,
    callback: Callable[[], Any],
    *,
    phase: str,
) -> tuple[Any, Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    (result, parent_calls), guard = run_fold_only_callback(
        lambda: audit_lfm2_parent_forward_calls(
            parent,
            callback,
            phase=phase,
            event_sink=events,
        ),
        phase=phase,
        protected_models=(parent,),
    )
    if parent_calls:
        raise Lfm2InteractionBeamExecutionError(
            "A cumulative interaction operation called the live parent; "
            f"phase={phase!r}, calls={parent_calls}, "
            f"first_event={events[:1]!r}."
        )
    report = guard.as_dict()
    for name in (
        "optimizer_constructions",
        "optimizer_steps",
        "backward_calls",
        "autograd_grad_calls",
    ):
        if int(report[name]) != 0:
            raise Lfm2InteractionBeamExecutionError(
                f"Fold-only phase {phase!r} recorded forbidden {name}."
            )
    return result, MappingProxyType(dict(report))


def bind_lfm2_interaction_beam_plan(
    parent: Any,
    evidence: Lfm2GlobalCausalEvidence,
    plan: Lfm2InteractionBeamPlan,
    *,
    materialize: Callable[
        [Lfm2ExactAllocationAlternative],
        Lfm2InteractionMaterialization,
    ],
    require_canonical_payload: bool = True,
) -> Lfm2BoundInteractionBeamPlan:
    """Bind each exact allocation to a complete physical multi-layer payload."""

    verify_lfm2_global_causal_evidence(evidence, parent=parent)
    verify_lfm2_interaction_beam_plan(plan, evidence)
    if not callable(materialize):
        raise Lfm2InteractionBeamError("materialize must be callable.")
    if not isinstance(require_canonical_payload, bool):
        raise Lfm2InteractionBeamError(
            "require_canonical_payload must be boolean."
        )
    candidate_by_id = {
        candidate.option_id: candidate
        for candidate in evidence.candidates
        if candidate.physically_installable
    }
    bound = []
    for alternative in plan.alternatives:
        phase = (
            "lfm2-interaction-beam-bind-"
            f"{alternative.isolated_rank:03d}"
        )
        materialization, guard_report = _guarded_parent_silent(
            parent,
            lambda alternative=alternative: materialize(alternative),
            phase=phase,
        )
        if not isinstance(
            materialization,
            Lfm2InteractionMaterialization,
        ):
            raise Lfm2InteractionBeamError(
                "materialize must return Lfm2InteractionMaterialization."
            )
        bundle = verify_lfm2_variable_width_fixed_plan_bundle(
            materialization.bundle,
            parent=parent,
        )
        if (
            tuple(bundle.plan.allocation.target_widths)
            != alternative.target_widths
            or bundle.plan.parameters_saved != alternative.parameters_saved
            or bundle.plan.target_unique_parameters
            != alternative.target_unique_parameters
            or materialization.clustering_calls != 0
        ):
            raise Lfm2InteractionBeamError(
                "Materialized bundle differs from the exact allocation or "
                "performed forbidden reclustering."
            )
        for option_id in alternative.option_ids:
            candidate = candidate_by_id[option_id]
            layer_plan = bundle.plan.layer_plans[candidate.layer_index]
            if (
                bundle.layer_binding_sha256[candidate.layer_index]
                != candidate.layer_binding_sha256
                or layer_plan.medoid_source_neurons
                != candidate.retained_source_units
                or layer_plan.target_width != candidate.target_units
            ):
                raise Lfm2InteractionBeamError(
                    f"Combined payload changed canonical option {option_id!r}."
                )
        canonical_ids = materialization.canonical_option_ids
        if require_canonical_payload:
            if (
                canonical_ids != alternative.option_ids
                or len(materialization.canonical_pack_manifest_sha256)
                != len(alternative.option_ids)
                or len(materialization.canonical_pack_tensor_sha256)
                != len(alternative.option_ids)
            ):
                raise Lfm2InteractionBeamError(
                    "A production interaction beam requires every selected "
                    "canonical proposal pack."
                )
        elif canonical_ids and canonical_ids != alternative.option_ids:
            raise Lfm2InteractionBeamError(
                "Optional canonical proposal IDs differ from the allocation."
            )
        guard_sha = _json_sha256(dict(guard_report))
        value = Lfm2BoundExactAllocation(
            schema=BOUND_ALTERNATIVE_SCHEMA,
            alternative=alternative,
            physical_bundle=bundle,
            canonical_payload_required=require_canonical_payload,
            canonical_option_ids=canonical_ids,
            canonical_pack_manifest_sha256=(
                materialization.canonical_pack_manifest_sha256
            ),
            canonical_pack_tensor_sha256=(
                materialization.canonical_pack_tensor_sha256
            ),
            binding_guard_sha256=guard_sha,
            bound_alternative_sha256="0" * 64,
        )
        bound.append(
            replace(
                value,
                bound_alternative_sha256=_json_sha256(
                    _bound_alternative_body(value)
                ),
            )
        )
    result = Lfm2BoundInteractionBeamPlan(
        schema=BOUND_PLAN_SCHEMA,
        structural_plan=plan,
        alternatives=tuple(bound),
        canonical_payload_required=require_canonical_payload,
        bound_plan_sha256="0" * 64,
    )
    sealed = replace(
        result,
        bound_plan_sha256=_json_sha256(_bound_plan_body(result)),
    )
    return verify_lfm2_bound_interaction_beam_plan(
        sealed,
        evidence,
        parent=parent,
    )


def verify_lfm2_bound_interaction_beam_plan(
    plan: Lfm2BoundInteractionBeamPlan,
    evidence: Lfm2GlobalCausalEvidence,
    *,
    parent: Any | None = None,
) -> Lfm2BoundInteractionBeamPlan:
    if not isinstance(plan, Lfm2BoundInteractionBeamPlan):
        raise Lfm2InteractionBeamError(
            "The bound interaction plan type is invalid."
        )
    verify_lfm2_interaction_beam_plan(plan.structural_plan, evidence)
    if (
        plan.schema != BOUND_PLAN_SCHEMA
        or len(plan.alternatives)
        != len(plan.structural_plan.alternatives)
        or not isinstance(plan.canonical_payload_required, bool)
        or _json_sha256(_bound_plan_body(plan)) != plan.bound_plan_sha256
    ):
        raise Lfm2InteractionBeamError(
            "The bound interaction plan seal is invalid."
        )
    candidates = {
        candidate.option_id: candidate
        for candidate in evidence.candidates
        if candidate.physically_installable
    }
    for structural, bound in zip(
        plan.structural_plan.alternatives,
        plan.alternatives,
        strict=True,
    ):
        verify_lfm2_variable_width_fixed_plan_bundle(
            bound.physical_bundle,
            parent=parent,
        )
        if (
            bound.alternative != structural
            or bound.canonical_payload_required
            != plan.canonical_payload_required
            or _json_sha256(_bound_alternative_body(bound))
            != bound.bound_alternative_sha256
            or tuple(bound.physical_bundle.plan.allocation.target_widths)
            != structural.target_widths
        ):
            raise Lfm2InteractionBeamError(
                "A bound alternative differs from its structural plan."
            )
        if plan.canonical_payload_required and (
            bound.canonical_option_ids != structural.option_ids
            or len(bound.canonical_pack_manifest_sha256)
            != len(structural.option_ids)
            or len(bound.canonical_pack_tensor_sha256)
            != len(structural.option_ids)
        ):
            raise Lfm2InteractionBeamError(
                "A bound alternative lost canonical pack provenance."
            )
        for option_id in structural.option_ids:
            candidate = candidates[option_id]
            if (
                bound.physical_bundle.layer_binding_sha256[
                    candidate.layer_index
                ]
                != candidate.layer_binding_sha256
            ):
                raise Lfm2InteractionBeamError(
                    "A bound alternative changed a candidate payload."
                )
    return plan


@dataclass(frozen=True, slots=True)
class Lfm2InteractionScoreWeights:
    """Whole-response retention weights; exact token equality is absent."""

    intelligence_retention: float = 0.30
    coherent_response: float = 0.25
    trajectory_retention: float = 0.15
    residual_path_retention: float = 0.10
    attention_path_retention: float = 0.10
    termination_quality: float = 0.05
    repetition_quality: float = 0.05

    def __post_init__(self) -> None:
        for name in self.as_dict():
            value = _unit_interval(getattr(self, name), name)
            if value <= 0.0:
                raise Lfm2InteractionBeamError(
                    "Every interaction score weight must be positive."
                )
            object.__setattr__(self, name, value)
        if not math.isclose(
            sum(self.as_dict().values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Lfm2InteractionBeamError(
                "Interaction score weights must sum to one."
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "intelligence_retention": self.intelligence_retention,
            "coherent_response": self.coherent_response,
            "trajectory_retention": self.trajectory_retention,
            "residual_path_retention": self.residual_path_retention,
            "attention_path_retention": self.attention_path_retention,
            "termination_quality": self.termination_quality,
            "repetition_quality": self.repetition_quality,
        }


@dataclass(frozen=True, slots=True)
class Lfm2InteractionAcceptanceThresholds:
    """User-facing success boundary for a cumulative folded child."""

    minimum_intelligence_retention: float = 0.90
    minimum_coherent_response: float = 0.90
    minimum_trajectory_retention: float = 0.0
    minimum_residual_path_retention: float = 0.0
    minimum_attention_path_retention: float = 0.0

    def __post_init__(self) -> None:
        for name in self.as_dict():
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), name),
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "minimum_intelligence_retention": (
                self.minimum_intelligence_retention
            ),
            "minimum_coherent_response": self.minimum_coherent_response,
            "minimum_trajectory_retention": (
                self.minimum_trajectory_retention
            ),
            "minimum_residual_path_retention": (
                self.minimum_residual_path_retention
            ),
            "minimum_attention_path_retention": (
                self.minimum_attention_path_retention
            ),
        }


@dataclass(frozen=True, slots=True)
class Lfm2CumulativeInteractionQuality:
    """Cached-evidence quality for one complete multi-layer folded child."""

    schema: str
    child_trace_cache_key: str
    child_output_events_sha256: str
    intelligence_retention: float
    coherent_response: float
    trajectory_retention: float
    residual_path_retention: float
    attention_path_retention: float
    termination_quality: float
    repetition_quality: float
    evaluator_id: str
    metrics: tuple[tuple[str, float | int | bool | str], ...] = ()
    exact_token_match_diagnostic: float | None = None
    quality_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != QUALITY_SCHEMA:
            raise Lfm2InteractionBeamError(
                "Cumulative interaction quality schema is invalid."
            )
        for name in (
            "child_trace_cache_key",
            "child_output_events_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        for name in (
            "intelligence_retention",
            "coherent_response",
            "trajectory_retention",
            "residual_path_retention",
            "attention_path_retention",
            "termination_quality",
            "repetition_quality",
        ):
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "evaluator_id",
            _text(self.evaluator_id, "evaluator_id"),
        )
        normalized = []
        for name, value in self.metrics:
            key = _text(name, "metric_name")
            if not isinstance(value, (str, int, float, bool)):
                raise Lfm2InteractionBeamError(
                    f"Metric {key!r} is not a JSON scalar."
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise Lfm2InteractionBeamError(
                    f"Metric {key!r} is not finite."
                )
            normalized.append((key, value))
        normalized.sort(key=lambda item: item[0])
        if len({name for name, _value in normalized}) != len(normalized):
            raise Lfm2InteractionBeamError(
                "Cumulative metric names must be unique."
            )
        object.__setattr__(self, "metrics", tuple(normalized))
        if self.exact_token_match_diagnostic is not None:
            object.__setattr__(
                self,
                "exact_token_match_diagnostic",
                _unit_interval(
                    self.exact_token_match_diagnostic,
                    "exact_token_match_diagnostic",
                ),
            )
        if self.quality_sha256:
            object.__setattr__(
                self,
                "quality_sha256",
                _digest(self.quality_sha256, "quality_sha256"),
            )

    def weighted_retention(
        self,
        weights: Lfm2InteractionScoreWeights,
    ) -> float:
        if not isinstance(weights, Lfm2InteractionScoreWeights):
            raise Lfm2InteractionBeamError(
                "weights must be Lfm2InteractionScoreWeights."
            )
        values = {
            "intelligence_retention": self.intelligence_retention,
            "coherent_response": self.coherent_response,
            "trajectory_retention": self.trajectory_retention,
            "residual_path_retention": self.residual_path_retention,
            "attention_path_retention": self.attention_path_retention,
            "termination_quality": self.termination_quality,
            "repetition_quality": self.repetition_quality,
        }
        return sum(
            values[name] * weight
            for name, weight in weights.as_dict().items()
        )

    def meets(
        self,
        thresholds: Lfm2InteractionAcceptanceThresholds,
    ) -> bool:
        if not isinstance(
            thresholds,
            Lfm2InteractionAcceptanceThresholds,
        ):
            raise Lfm2InteractionBeamError(
                "thresholds must be Lfm2InteractionAcceptanceThresholds."
            )
        return (
            self.intelligence_retention
            >= thresholds.minimum_intelligence_retention
            and self.coherent_response
            >= thresholds.minimum_coherent_response
            and self.trajectory_retention
            >= thresholds.minimum_trajectory_retention
            and self.residual_path_retention
            >= thresholds.minimum_residual_path_retention
            and self.attention_path_retention
            >= thresholds.minimum_attention_path_retention
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "child_trace_cache_key": self.child_trace_cache_key,
            "child_output_events_sha256": self.child_output_events_sha256,
            "intelligence_retention": self.intelligence_retention,
            "coherent_response": self.coherent_response,
            "trajectory_retention": self.trajectory_retention,
            "residual_path_retention": self.residual_path_retention,
            "attention_path_retention": self.attention_path_retention,
            "termination_quality": self.termination_quality,
            "repetition_quality": self.repetition_quality,
            "evaluator_id": self.evaluator_id,
            "metrics": dict(self.metrics),
            "exact_token_match_diagnostic": self.exact_token_match_diagnostic,
            "exact_token_match_is_diagnostic_only": True,
            "quality_sha256": self.quality_sha256,
        }


def _quality_body(
    quality: Lfm2CumulativeInteractionQuality,
) -> dict[str, Any]:
    result = quality.as_dict()
    result.pop("quality_sha256", None)
    return result


def seal_lfm2_cumulative_interaction_quality(
    *,
    child_trace_cache_key: str,
    child_output_events_sha256: str,
    intelligence_retention: float,
    coherent_response: float,
    trajectory_retention: float,
    residual_path_retention: float,
    attention_path_retention: float,
    termination_quality: float,
    repetition_quality: float,
    evaluator_id: str,
    metrics: Sequence[
        tuple[str, float | int | bool | str]
    ] = (),
    exact_token_match_diagnostic: float | None = None,
) -> Lfm2CumulativeInteractionQuality:
    value = Lfm2CumulativeInteractionQuality(
        schema=QUALITY_SCHEMA,
        child_trace_cache_key=child_trace_cache_key,
        child_output_events_sha256=child_output_events_sha256,
        intelligence_retention=intelligence_retention,
        coherent_response=coherent_response,
        trajectory_retention=trajectory_retention,
        residual_path_retention=residual_path_retention,
        attention_path_retention=attention_path_retention,
        termination_quality=termination_quality,
        repetition_quality=repetition_quality,
        evaluator_id=evaluator_id,
        metrics=tuple(metrics),
        exact_token_match_diagnostic=exact_token_match_diagnostic,
        quality_sha256="",
    )
    return replace(
        value,
        quality_sha256=_json_sha256(_quality_body(value)),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CumulativeInteractionContext:
    """Everything a cache-only whole-response evaluator may consume."""

    child: Any = field(repr=False, compare=False)
    alternative: Lfm2BoundExactAllocation
    attempt_index: int
    evidence_sha256: str
    parent_trace_cache_key: str
    parent_output_envelope_sha256: str
    task_anchor_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2CumulativeInteractionAttempt:
    schema: str
    bound_alternative_sha256: str
    isolated_rank: int
    quality: Lfm2CumulativeInteractionQuality
    weighted_retention: float
    meets_acceptance_gate: bool
    rollback_exact: bool
    blank_child_state_sha256: str
    blank_child_runtime_sha256: str
    installed_child_state_sha256: str
    installed_child_runtime_sha256: str
    guard_reports: tuple[Mapping[str, Any], ...]
    attempt_sha256: str

    def __post_init__(self) -> None:
        if self.schema != ATTEMPT_SCHEMA:
            raise Lfm2InteractionBeamError(
                "Cumulative interaction attempt schema is invalid."
            )
        object.__setattr__(
            self,
            "bound_alternative_sha256",
            _digest(
                self.bound_alternative_sha256,
                "bound_alternative_sha256",
            ),
        )
        object.__setattr__(
            self,
            "isolated_rank",
            _nonnegative_int(self.isolated_rank, "isolated_rank"),
        )
        if not isinstance(
            self.quality,
            Lfm2CumulativeInteractionQuality,
        ):
            raise Lfm2InteractionBeamError(
                "Attempt quality has the wrong type."
            )
        retention = _unit_interval(
            self.weighted_retention,
            "weighted_retention",
        )
        object.__setattr__(self, "weighted_retention", retention)
        for name in ("meets_acceptance_gate", "rollback_exact"):
            if not isinstance(getattr(self, name), bool):
                raise Lfm2InteractionBeamError(
                    f"{name} must be boolean."
                )
        for name in (
            "blank_child_state_sha256",
            "blank_child_runtime_sha256",
            "installed_child_state_sha256",
            "installed_child_runtime_sha256",
            "attempt_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        reports = tuple(
            MappingProxyType(dict(report))
            for report in self.guard_reports
        )
        if not reports:
            raise Lfm2InteractionBeamError(
                "An attempt requires fold-only guard reports."
            )
        object.__setattr__(self, "guard_reports", reports)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "bound_alternative_sha256": (
                self.bound_alternative_sha256
            ),
            "isolated_rank": self.isolated_rank,
            "quality": self.quality.as_dict(),
            "weighted_retention": self.weighted_retention,
            "meets_acceptance_gate": self.meets_acceptance_gate,
            "rollback_exact": self.rollback_exact,
            "blank_child_state_sha256": self.blank_child_state_sha256,
            "blank_child_runtime_sha256": (
                self.blank_child_runtime_sha256
            ),
            "installed_child_state_sha256": (
                self.installed_child_state_sha256
            ),
            "installed_child_runtime_sha256": (
                self.installed_child_runtime_sha256
            ),
            "guard_reports": [
                dict(report) for report in self.guard_reports
            ],
            "attempt_sha256": self.attempt_sha256,
            "transaction_committed": False,
            "parent_forward_calls": 0,
        }


def _attempt_body(
    attempt: Lfm2CumulativeInteractionAttempt,
) -> dict[str, Any]:
    result = attempt.as_dict()
    result.pop("attempt_sha256", None)
    return result


@dataclass(frozen=True, slots=True)
class Lfm2InteractionBeamRun:
    """All top-K cumulative measurements, ranked after exact rollback."""

    schema: str
    bound_plan_sha256: str
    evidence_sha256: str
    weights: Lfm2InteractionScoreWeights
    thresholds: Lfm2InteractionAcceptanceThresholds
    attempts: tuple[Lfm2CumulativeInteractionAttempt, ...]
    interaction_ranking: tuple[str, ...]
    best_bound_alternative_sha256: str | None
    status: Literal["accepted_candidate_found", "no_candidate_met_gate"]
    run_sha256: str

    def as_dict(self) -> dict[str, Any]:
        guard_reports = [
            report
            for attempt in self.attempts
            for report in attempt.guard_reports
        ]
        return {
            "schema": self.schema,
            "bound_plan_sha256": self.bound_plan_sha256,
            "evidence_sha256": self.evidence_sha256,
            "weights": self.weights.as_dict(),
            "thresholds": self.thresholds.as_dict(),
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "interaction_ranking": list(self.interaction_ranking),
            "best_bound_alternative_sha256": (
                self.best_bound_alternative_sha256
            ),
            "status": self.status,
            "run_sha256": self.run_sha256,
            "all_search_children_rolled_back": all(
                attempt.rollback_exact for attempt in self.attempts
            ),
            "parent_forward_calls": 0,
            "parent_cache_build_calls": 0,
            "parent_generation_calls": 0,
            "optimizer_constructions": sum(
                int(report["optimizer_constructions"])
                for report in guard_reports
            ),
            "optimizer_steps": sum(
                int(report["optimizer_steps"])
                for report in guard_reports
            ),
            "backward_calls": sum(
                int(report["backward_calls"])
                for report in guard_reports
            ),
            "autograd_grad_calls": sum(
                int(report["autograd_grad_calls"])
                for report in guard_reports
            ),
            "exact_token_match_objective_used": False,
            "checkpoint_published": False,
        }


def _run_body(run: Lfm2InteractionBeamRun) -> dict[str, Any]:
    result = run.as_dict()
    result.pop("run_sha256", None)
    return result


def _installed_lfm2_widths(child: Any) -> tuple[int, ...]:
    try:
        widths = tuple(
            int(layer.feed_forward.w1.out_features)
            for layer in child.model.layers
        )
    except (AttributeError, TypeError) as exc:
        raise Lfm2InteractionBeamExecutionError(
            "The installed child is not a fourteen-layer LFM2 model."
        ) from exc
    if len(widths) != LFM2_VARIABLE_WIDTH_LAYER_COUNT:
        raise Lfm2InteractionBeamExecutionError(
            "The installed child lost the fourteen-layer topology."
        )
    return widths


def evaluate_lfm2_interaction_beam(
    parent: Any,
    evidence: Lfm2GlobalCausalEvidence,
    plan: Lfm2BoundInteractionBeamPlan,
    *,
    child_factory: Callable[
        [Lfm2BoundExactAllocation],
        Any,
    ],
    evaluator: Callable[
        [Lfm2CumulativeInteractionContext],
        Lfm2CumulativeInteractionQuality,
    ],
    weights: Lfm2InteractionScoreWeights | None = None,
    thresholds: Lfm2InteractionAcceptanceThresholds | None = None,
) -> Lfm2InteractionBeamRun:
    """Measure complete top-K children from caches and roll all of them back."""

    verify_lfm2_global_causal_evidence(evidence, parent=parent)
    verify_lfm2_bound_interaction_beam_plan(
        plan,
        evidence,
        parent=parent,
    )
    if not callable(child_factory) or not callable(evaluator):
        raise Lfm2InteractionBeamExecutionError(
            "child_factory and evaluator must be callable."
        )
    selected_weights = (
        Lfm2InteractionScoreWeights() if weights is None else weights
    )
    selected_thresholds = (
        Lfm2InteractionAcceptanceThresholds()
        if thresholds is None
        else thresholds
    )
    if not isinstance(selected_weights, Lfm2InteractionScoreWeights):
        raise Lfm2InteractionBeamExecutionError(
            "weights must be Lfm2InteractionScoreWeights or None."
        )
    if not isinstance(
        selected_thresholds,
        Lfm2InteractionAcceptanceThresholds,
    ):
        raise Lfm2InteractionBeamExecutionError(
            "thresholds must be Lfm2InteractionAcceptanceThresholds or None."
        )
    attempts = []
    for attempt_index, alternative in enumerate(plan.alternatives):
        phase_root = f"lfm2-interaction-beam-{attempt_index:03d}"
        child, factory_guard = _guarded_parent_silent(
            parent,
            lambda alternative=alternative: child_factory(alternative),
            phase=f"{phase_root}-child-factory",
        )
        if child is parent:
            raise Lfm2InteractionBeamExecutionError(
                "A cumulative search child must be distinct from the parent."
            )
        blank_state = fingerprint_model_state(child)
        blank_runtime = fingerprint_model_runtime_state(child)
        transaction = None
        guard_reports: list[Mapping[str, Any]] = [factory_guard]
        try:
            transaction, install_guard = _guarded_parent_silent(
                parent,
                lambda alternative=alternative, child=child: (
                    install_lfm2_variable_width_fold_plan(
                        parent,
                        child,
                        alternative.physical_bundle.fixed_evidence,
                        alternative.physical_bundle.plan,
                    )
                ),
                phase=f"{phase_root}-install",
            )
            guard_reports.append(install_guard)
            if (
                _installed_lfm2_widths(child)
                != alternative.alternative.target_widths
                or _unique_parameter_count(child)
                != alternative.alternative.target_unique_parameters
            ):
                raise Lfm2InteractionBeamExecutionError(
                    "Installed cumulative child differs from its exact target."
                )
            installed_state = fingerprint_model_state(child)
            installed_runtime = fingerprint_model_runtime_state(child)
            context = Lfm2CumulativeInteractionContext(
                child=child,
                alternative=alternative,
                attempt_index=attempt_index,
                evidence_sha256=evidence.evidence_sha256,
                parent_trace_cache_key=evidence.parent_trace_cache_key,
                parent_output_envelope_sha256=(
                    evidence.parent_output_envelope_sha256
                ),
                task_anchor_manifest_sha256=(
                    evidence.task_anchor_manifest_sha256
                ),
            )
            quality, evaluation_guard = _guarded_parent_silent(
                parent,
                lambda context=context: evaluator(context),
                phase=f"{phase_root}-cached-evaluation",
            )
            guard_reports.append(evaluation_guard)
            if (
                not isinstance(
                    quality,
                    Lfm2CumulativeInteractionQuality,
                )
                or not quality.quality_sha256
                or _json_sha256(_quality_body(quality))
                != quality.quality_sha256
            ):
                raise Lfm2InteractionBeamExecutionError(
                    "Evaluator returned unsealed cumulative quality."
                )
            rollback_result, rollback_guard = _guarded_parent_silent(
                parent,
                transaction.rollback,
                phase=f"{phase_root}-rollback",
            )
            guard_reports.append(rollback_guard)
            transaction = None
            rollback_exact = bool(
                rollback_result
                and fingerprint_model_state(child) == blank_state
                and fingerprint_model_runtime_state(child) == blank_runtime
            )
            if not rollback_exact:
                raise Lfm2InteractionBeamExecutionError(
                    "A cumulative interaction child did not roll back exactly."
                )
            attempt = Lfm2CumulativeInteractionAttempt(
                schema=ATTEMPT_SCHEMA,
                bound_alternative_sha256=(
                    alternative.bound_alternative_sha256
                ),
                isolated_rank=alternative.alternative.isolated_rank,
                quality=quality,
                weighted_retention=quality.weighted_retention(
                    selected_weights
                ),
                meets_acceptance_gate=quality.meets(
                    selected_thresholds
                ),
                rollback_exact=True,
                blank_child_state_sha256=blank_state,
                blank_child_runtime_sha256=blank_runtime,
                installed_child_state_sha256=installed_state,
                installed_child_runtime_sha256=installed_runtime,
                guard_reports=tuple(guard_reports),
                attempt_sha256="0" * 64,
            )
            attempts.append(
                replace(
                    attempt,
                    attempt_sha256=_json_sha256(_attempt_body(attempt)),
                )
            )
        except BaseException:
            if transaction is not None:
                active = bool(getattr(transaction, "active", True))
                if active:
                    try:
                        _guarded_parent_silent(
                            parent,
                            transaction.rollback,
                            phase=f"{phase_root}-exception-rollback",
                        )
                    except BaseException:
                        pass
            if (
                fingerprint_model_state(child) != blank_state
                or fingerprint_model_runtime_state(child) != blank_runtime
            ):
                raise Lfm2InteractionBeamExecutionError(
                    "Interaction evaluation failed and the child did not "
                    "restore exactly."
                )
            raise
    ranked_attempts = sorted(
        attempts,
        key=lambda attempt: (
            not attempt.meets_acceptance_gate,
            -attempt.weighted_retention,
            plan.alternatives[
                attempt.isolated_rank
            ].alternative.isolated_causal_score,
            attempt.bound_alternative_sha256,
        ),
    )
    accepted = [
        attempt
        for attempt in ranked_attempts
        if attempt.meets_acceptance_gate
    ]
    best = (
        None if not accepted else accepted[0].bound_alternative_sha256
    )
    run = Lfm2InteractionBeamRun(
        schema=RUN_SCHEMA,
        bound_plan_sha256=plan.bound_plan_sha256,
        evidence_sha256=evidence.evidence_sha256,
        weights=selected_weights,
        thresholds=selected_thresholds,
        attempts=tuple(attempts),
        interaction_ranking=tuple(
            attempt.bound_alternative_sha256
            for attempt in ranked_attempts
        ),
        best_bound_alternative_sha256=best,
        status=(
            "accepted_candidate_found"
            if best is not None
            else "no_candidate_met_gate"
        ),
        run_sha256="0" * 64,
    )
    sealed = replace(run, run_sha256=_json_sha256(_run_body(run)))
    return verify_lfm2_interaction_beam_run(
        sealed,
        plan,
        evidence,
    )


def verify_lfm2_interaction_beam_run(
    run: Lfm2InteractionBeamRun,
    plan: Lfm2BoundInteractionBeamPlan,
    evidence: Lfm2GlobalCausalEvidence,
) -> Lfm2InteractionBeamRun:
    """Authenticate all cumulative measurements, ranks, and zero-op guards."""

    verify_lfm2_bound_interaction_beam_plan(plan, evidence)
    if (
        not isinstance(run, Lfm2InteractionBeamRun)
        or run.schema != RUN_SCHEMA
        or run.bound_plan_sha256 != plan.bound_plan_sha256
        or run.evidence_sha256 != evidence.evidence_sha256
        or len(run.attempts) != len(plan.alternatives)
        or _json_sha256(_run_body(run)) != run.run_sha256
    ):
        raise Lfm2InteractionBeamError(
            "The interaction-beam run seal or binding is invalid."
        )
    bound_by_sha = {
        alternative.bound_alternative_sha256: alternative
        for alternative in plan.alternatives
    }
    seen = set()
    for expected_rank, attempt in enumerate(run.attempts):
        alternative = bound_by_sha.get(
            attempt.bound_alternative_sha256
        )
        if (
            alternative is None
            or attempt.isolated_rank != expected_rank
            or alternative.alternative.isolated_rank != expected_rank
            or attempt.bound_alternative_sha256 in seen
            or _json_sha256(_attempt_body(attempt))
            != attempt.attempt_sha256
            or _json_sha256(_quality_body(attempt.quality))
            != attempt.quality.quality_sha256
            or not attempt.rollback_exact
            or attempt.weighted_retention
            != attempt.quality.weighted_retention(run.weights)
            or attempt.meets_acceptance_gate
            != attempt.quality.meets(run.thresholds)
        ):
            raise Lfm2InteractionBeamError(
                "A cumulative interaction attempt is invalid."
            )
        seen.add(attempt.bound_alternative_sha256)
        for report in attempt.guard_reports:
            for name in (
                "optimizer_constructions",
                "optimizer_steps",
                "backward_calls",
                "autograd_grad_calls",
            ):
                if int(report[name]) != 0:
                    raise Lfm2InteractionBeamError(
                        f"Interaction run recorded forbidden {name}."
                    )
    expected_ranking = tuple(
        attempt.bound_alternative_sha256
        for attempt in sorted(
            run.attempts,
            key=lambda attempt: (
                not attempt.meets_acceptance_gate,
                -attempt.weighted_retention,
                plan.alternatives[
                    attempt.isolated_rank
                ].alternative.isolated_causal_score,
                attempt.bound_alternative_sha256,
            ),
        )
    )
    accepted = [
        alternative_sha
        for alternative_sha in expected_ranking
        if next(
            attempt
            for attempt in run.attempts
            if attempt.bound_alternative_sha256 == alternative_sha
        ).meets_acceptance_gate
    ]
    expected_best = None if not accepted else accepted[0]
    expected_status = (
        "accepted_candidate_found"
        if expected_best is not None
        else "no_candidate_met_gate"
    )
    if (
        run.interaction_ranking != expected_ranking
        or run.best_bound_alternative_sha256 != expected_best
        or run.status != expected_status
    ):
        raise Lfm2InteractionBeamError(
            "The cumulative interaction ranking is invalid."
        )
    return run


__all__ = [
    "ALTERNATIVE_SCHEMA",
    "ATTEMPT_SCHEMA",
    "BOUND_ALTERNATIVE_SCHEMA",
    "BOUND_PLAN_SCHEMA",
    "Lfm2BoundExactAllocation",
    "Lfm2BoundInteractionBeamPlan",
    "Lfm2CumulativeInteractionContext",
    "Lfm2CumulativeInteractionQuality",
    "Lfm2CumulativeInteractionAttempt",
    "Lfm2ExactAllocationAlternative",
    "Lfm2InteractionAcceptanceThresholds",
    "Lfm2InteractionBeamError",
    "Lfm2InteractionBeamExecutionError",
    "Lfm2InteractionBeamInfeasible",
    "Lfm2InteractionBeamPlan",
    "Lfm2InteractionBeamRun",
    "Lfm2InteractionMaterialization",
    "Lfm2InteractionScoreWeights",
    "MATERIALIZATION_SCHEMA",
    "METHOD_ID",
    "PLAN_SCHEMA",
    "QUALITY_SCHEMA",
    "RUN_SCHEMA",
    "bind_lfm2_interaction_beam_plan",
    "build_and_seal_lfm2_cached_causal_frontier_from_canonical_packs",
    "enumerate_lfm2_interaction_beam_plan",
    "evaluate_lfm2_interaction_beam",
    "materialize_lfm2_canonical_multi_pack_allocation",
    "plan_lfm2_global_causal_target_from_canonical_packs",
    "seal_lfm2_global_causal_evidence_from_canonical_packs",
    "seal_lfm2_cumulative_interaction_quality",
    "verify_lfm2_bound_interaction_beam_plan",
    "verify_lfm2_interaction_beam_plan",
    "verify_lfm2_interaction_beam_run",
]
