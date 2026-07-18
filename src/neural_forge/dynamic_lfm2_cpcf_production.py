"""Exact, non-overlapping production registration for bridged LFM2 CPCF.

The generic production catalog intentionally knows nothing about individual
fold implementations.  This module binds one already-planned exact LFM2 blank,
one verified tokenizer bridge, and one sealed CPCF evidence resource into a
single-attempt capability.  The exact-size range prevents the CPCF lane from
silently taking over unrelated LFM2 targets that happen to share its method
identifier.
"""

from __future__ import annotations

from typing import Any

from .dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from .dynamic_lfm2_cpcf_adapter import (
    Lfm2CpcfFoldStackAdapter,
    Lfm2CpcfStackResources,
)
from .dynamic_lfm2_cpcf_reviewed_gate import (
    Lfm2CpcfReviewedPromotionEvidence,
    authenticate_lfm2_cpcf_reviewed_promotion_gate,
)
from .dynamic_production_catalog import (
    CertificateScoreTerm,
    ProductionFoldBundle,
    ProductionFoldCandidateSpec,
    ProductionFoldCapability,
)
from .dynamic_target_synthesis import BudgetedDynamicTargetPlan
from .hf_lfm2_dense_budget_model_fold import DENSE_BUDGET_MODEL_METHOD
from .hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    strict_tokenizer_bridge_configuration,
)


SCHEMA = "neural-forge-lfm2-cpcf-production-bundle-v1"
_BRIDGE_REQUIRED_KEY = "_neural_forge_tokenizer_bridge_required"
_DENSE_CATALOG_KEY = "_neural_forge_dense_topology_catalog"


class Lfm2CpcfProductionBundleError(RuntimeError):
    """The exact target, bridge, and CPCF resources are not one sealed lane."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CpcfProductionBundleError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfProductionBundleError(f"{name} must be a positive integer.")
    return value


def _planned_cache_key(resources: Lfm2CpcfStackResources) -> str:
    cache_keys = {
        phase.parent_evidence.cache_key for phase in resources.phases
    }
    if len(cache_keys) != 1:
        raise Lfm2CpcfProductionBundleError(
            "CPCF resources do not bind exactly one parent trace cache."
        )
    return _digest(cache_keys.pop(), "CPCF parent cache key")


def _verify_planned_cpcf_topology(
    *,
    target_plan: BudgetedDynamicTargetPlan,
    resources: Lfm2CpcfStackResources,
    bridge: StrictTokenizerBridgeArtifact,
) -> None:
    planned = target_plan.target_certificate.planned_graph
    depth = _positive_int(planned.layer_count, "planned target depth")
    expected_spans = tuple((index,) for index in range(depth))
    configuration = dict(planned.configuration)
    target_rows = _positive_int(
        bridge.transport_profile.target_row_capacity,
        "tokenizer bridge target row capacity",
    )
    phase_layer_counts = {
        len(phase.parent_evidence.layers) for phase in resources.phases
    }
    try:
        tied_rows = int(resources.source_tied_table.shape[0])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise Lfm2CpcfProductionBundleError(
            "CPCF source token table does not expose a target-row dimension."
        ) from exc
    if (
        planned.ordered_source_spans != expected_spans
        or phase_layer_counts != {depth}
    ):
        raise Lfm2CpcfProductionBundleError(
            "Planned target depth/source spans and CPCF parent phase layers "
            "must describe one equal-depth, one-source-layer-per-target-layer "
            "fold."
        )
    if (
        configuration.get("vocab_size") != target_rows
        or resources.trace_adapter.target_row_capacity != target_rows
        or tied_rows != target_rows
    ):
        raise Lfm2CpcfProductionBundleError(
            "Planned vocabulary, trace bridge, and folded source token table "
            "do not share one exact target-row capacity."
        )


def _verify_target_lane(
    *,
    target_plan: BudgetedDynamicTargetPlan,
    target_adapter: Lfm2TargetSynthesisAdapter,
    resources: Lfm2CpcfStackResources,
) -> tuple[
    int,
    str,
    StrictTokenizerBridgeArtifact,
    Lfm2CpcfReviewedPromotionEvidence,
]:
    if type(target_plan) is not BudgetedDynamicTargetPlan:
        raise Lfm2CpcfProductionBundleError(
            "target_plan must be an authenticated BudgetedDynamicTargetPlan."
        )
    if type(target_adapter) is not Lfm2TargetSynthesisAdapter:
        raise Lfm2CpcfProductionBundleError(
            "target_adapter must have the exact LFM2 synthesis-adapter type."
        )
    if type(resources) is not Lfm2CpcfStackResources:
        raise Lfm2CpcfProductionBundleError(
            "resources must have the exact CPCF stack-resource type."
        )
    request = target_plan.target_request
    exact_size = _positive_int(
        request.exact_unique_parameters,
        "exact target parameter count",
    )
    if (
        target_plan.target_certificate.parent_graph_fingerprint
        != resources.parent_graph_fingerprint
    ):
        raise Lfm2CpcfProductionBundleError(
            "Target plan and CPCF parent evidence name different inspected "
            "parent graph fingerprints."
        )
    if (
        request.target_family != "lfm2"
        or request.target_modalities != ("text",)
        or request.adapter_id
        not in {None, Lfm2TargetSynthesisAdapter.adapter_id}
        or request.fold_method_id not in {None, DENSE_BUDGET_MODEL_METHOD}
    ):
        raise Lfm2CpcfProductionBundleError(
            "Target plan is not an LFM2 text blank for the registered CPCF "
            "dense residual-attention method."
        )
    configuration = dict(request.configuration)
    if (
        configuration.get(_BRIDGE_REQUIRED_KEY) is not True
        or not isinstance(configuration.get(_DENSE_CATALOG_KEY), str)
        or not str(configuration[_DENSE_CATALOG_KEY]).strip()
    ):
        raise Lfm2CpcfProductionBundleError(
            "Target plan is not an exact bridge-required dense topology."
        )
    bridge = target_adapter.tokenizer_bridge_artifact
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise Lfm2CpcfProductionBundleError(
            "Target adapter does not carry a strict tokenizer bridge."
        )
    try:
        bridge.verify_files()
        trace = resources.trace_adapter.verify()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfProductionBundleError(
            "Target or trace tokenizer bridge failed authentication."
        ) from exc
    expected_bridge_configuration = strict_tokenizer_bridge_configuration(bridge)
    if any(
        configuration.get(name) != value
        for name, value in expected_bridge_configuration.items()
    ):
        raise Lfm2CpcfProductionBundleError(
            "Planned target configuration differs from its tokenizer bridge."
        )
    if (
        request.tokenizer_fingerprint
        != bridge.target_tokenizer_semantics_sha256
        or trace.bridge_artifact_sha256 != bridge.artifact_sha256
        or trace.transport_profile_fingerprint != bridge.profile_fingerprint
        or trace.target_row_capacity
        != bridge.transport_profile.target_row_capacity
    ):
        raise Lfm2CpcfProductionBundleError(
            "Target plan, trace cache, and tokenizer bridge identities differ."
        )
    cache_key = _planned_cache_key(resources)
    if target_adapter.evidence_fingerprint != cache_key:
        raise Lfm2CpcfProductionBundleError(
            "Target adapter evidence fingerprint differs from the CPCF parent "
            "trace cache."
        )
    try:
        reviewed_gate = authenticate_lfm2_cpcf_reviewed_promotion_gate(
            resources.final_evaluator
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfProductionBundleError(
            "CPCF resources do not carry an authenticated reviewed-promotion "
            "gate."
        ) from exc
    reviewed = reviewed_gate.evidence
    certificate = target_plan.target_certificate
    candidate = target_plan.frontier.candidates[
        certificate.selected_candidate_rank
    ]
    requested_budget = target_plan.frontier.request.as_dict()
    if (
        reviewed.blank_target_parent_graph_fingerprint
        != certificate.parent_graph_fingerprint
        or reviewed.exact_target_unique_parameters != exact_size
        or dict(reviewed.target_requested_budget) != requested_budget
        or reviewed.target_frontier_fingerprint
        != target_plan.frontier.fingerprint
        or reviewed.target_certificate_sha256 != certificate.fingerprint
        or reviewed.selected_target_candidate_fingerprint
        != candidate.fingerprint
        or reviewed.selected_target_candidate_rank
        != certificate.selected_candidate_rank
    ):
        raise Lfm2CpcfProductionBundleError(
            "Reviewed CPCF run receipt does not describe the exact parent "
            "graph, budget, frontier, certificate, selected candidate/rank, "
            "and parameter count in this target-plan preflight."
        )
    _verify_planned_cpcf_topology(
        target_plan=target_plan,
        resources=resources,
        bridge=bridge,
    )
    return exact_size, cache_key, bridge, reviewed


def build_lfm2_cpcf_exact_production_bundle(
    *,
    target_plan: BudgetedDynamicTargetPlan,
    target_adapter: Lfm2TargetSynthesisAdapter,
    resources: Lfm2CpcfStackResources,
    capability_id: str | None = None,
    portfolio_id: str | None = None,
    candidate_id: str = "bridged-cpcf",
    priority: int = 300,
) -> ProductionFoldBundle:
    """Bind one exact bridged blank and CPCF cache into one production lane."""

    exact_size, cache_key, bridge, reviewed = _verify_target_lane(
        target_plan=target_plan,
        target_adapter=target_adapter,
        resources=resources,
    )
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise Lfm2CpcfProductionBundleError("priority must be an integer.")
    request = target_plan.target_request
    capability = ProductionFoldCapability(
        capability_id=(
            capability_id or f"lfm2-text-cpcf-exact-{exact_size}"
        ),
        source_families=("lfm2",),
        target_family="lfm2",
        target_modalities=("text",),
        minimum_target_unique_parameters=exact_size,
        maximum_target_unique_parameters=exact_size,
    )
    candidate = ProductionFoldCandidateSpec(
        candidate_id=candidate_id,
        method_id=DENSE_BUDGET_MODEL_METHOD,
        evidence_fingerprint=cache_key,
        target_adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
        priority=priority,
        blank_seed=request.blank_seed,
        configuration=request.configuration,
        metadata=(
            ("registration_schema", SCHEMA),
            ("target_plan_fingerprint", target_plan.fingerprint),
            (
                "reviewed_target_plan_binding_sha256",
                reviewed.target_plan_binding_sha256,
            ),
            (
                "reviewed_cpcf_run_receipt_sha256",
                reviewed.cpcf_run_receipt_sha256,
            ),
            (
                "reviewed_evidence_fingerprint",
                reviewed.full_reviewed_evidence_fingerprint,
            ),
            (
                "reviewed_target_frontier_fingerprint",
                reviewed.target_frontier_fingerprint,
            ),
            (
                "reviewed_target_certificate_sha256",
                reviewed.target_certificate_sha256,
            ),
            (
                "reviewed_selected_target_candidate_fingerprint",
                reviewed.selected_target_candidate_fingerprint,
            ),
            ("cpcf_resource_fingerprint", resources.resource_fingerprint),
            ("tokenizer_bridge_artifact_sha256", bridge.artifact_sha256),
            (
                "target_tokenizer_semantics_sha256",
                bridge.target_tokenizer_semantics_sha256,
            ),
        ),
    )
    return ProductionFoldBundle(
        capability=capability,
        target_adapters=(target_adapter,),
        stack_adapters=(Lfm2CpcfFoldStackAdapter(resources),),
        candidate_specs=(candidate,),
        maximum_attempts=1,
        portfolio_id=(
            portfolio_id
            or f"neural_forge.production.lfm2-cpcf-exact-{exact_size}"
        ),
        version=1,
        priority=priority,
        score_terms=(
            CertificateScoreTerm(
                "candidate_outputs_replayed_exactly",
                weight=1.0,
            ),
            CertificateScoreTerm(
                "inference_contract_satisfied",
                weight=1.0,
            ),
        ),
    )


__all__ = [
    "Lfm2CpcfProductionBundleError",
    "SCHEMA",
    "build_lfm2_cpcf_exact_production_bundle",
]
