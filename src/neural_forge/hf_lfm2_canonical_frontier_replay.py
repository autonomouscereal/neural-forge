"""Authenticate an LFM2 causal frontier from its exact canonical packs.

The original cached-frontier replay predates canonical proposal packs.  It
reconstructs each proposal and cumulative stage by clustering again, which is
not a valid authentication path when the measured cross-device proposal is
already sealed.  This module instead rehydrates the exact changed-layer packs,
rebuilds only cache-derived source evidence, and proves that the resulting
evidence and 35% plan retain the frontier's original seals.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge import (
    hf_lfm2_cached_causal_evidence as cached_causal_module,
    hf_lfm2_variable_width_fold as variable_width_fold_module,
)
from neural_forge.dynamic_fold_executor import run_fold_only_callback
from neural_forge.hf_lfm2_canonical_proposal_pack import (
    Lfm2CanonicalProposalPack,
)
from neural_forge.hf_lfm2_global_causal_allocator import (
    Lfm2CausalDamageWeights,
    Lfm2GlobalCausalEvidence,
    Lfm2GlobalCausalInfeasible,
    Lfm2GlobalCausalPlan,
    audit_lfm2_parent_forward_calls,
)
from neural_forge.hf_lfm2_interaction_beam_allocator import (
    plan_lfm2_global_causal_target_from_canonical_packs,
    seal_lfm2_global_causal_evidence_from_canonical_packs,
)


FINALIZER_SCHEMA = "neural-forge-lfm2-canonical-merge-finalizer-v1"
REPLAY_SCHEMA = "neural-forge-lfm2-canonical-frontier-replay-v1"


class Lfm2CanonicalFrontierReplayError(RuntimeError):
    """Canonical causal-frontier replay failed its provenance contract."""


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
        raise Lfm2CanonicalFrontierReplayError(
            "Canonical replay provenance must be finite JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CanonicalFrontierReplayError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _guard_is_zero(report: Mapping[str, Any]) -> bool:
    return all(
        int(report.get(name, -1)) == 0
        for name in (
            "optimizer_constructions",
            "optimizer_steps",
            "backward_calls",
            "autograd_grad_calls",
        )
    )


def _canonical_pack_binding(
    packs: Mapping[str, Lfm2CanonicalProposalPack],
) -> dict[str, dict[str, str]]:
    result = {}
    for option_id, pack in sorted(packs.items()):
        if (
            not isinstance(option_id, str)
            or not option_id
            or not isinstance(pack, Lfm2CanonicalProposalPack)
            or pack.proposal.option_id != option_id
        ):
            raise Lfm2CanonicalFrontierReplayError(
                "Canonical replay packs are malformed or keyed incorrectly."
            )
        result[option_id] = {
            "manifest_sha256": _digest(
                pack.manifest.get("manifest_sha256"),
                f"{option_id}.manifest_sha256",
            ),
            "tensor_file_sha256": _digest(
                pack.manifest.get("tensor_file_sha256"),
                f"{option_id}.tensor_file_sha256",
            ),
            "proposal_sha256": _digest(
                pack.proposal.proposal_sha256,
                f"{option_id}.proposal_sha256",
            ),
            "layer_binding_sha256": _digest(
                pack.proposal.layer_binding_sha256,
                f"{option_id}.layer_binding_sha256",
            ),
        }
    if not result:
        raise Lfm2CanonicalFrontierReplayError(
            "Canonical replay requires at least one exact proposal pack."
        )
    return result


def _verify_finalizer_report(
    report: Mapping[str, Any],
    document: Mapping[str, Any],
    pack_binding: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Any]:
    if not isinstance(report, Mapping):
        raise Lfm2CanonicalFrontierReplayError(
            "Canonical finalizer report must be a mapping."
        )
    body = dict(report)
    seal = body.pop("finalizer_report_sha256", None)
    if (
        body.get("schema") != FINALIZER_SCHEMA
        or _digest(seal, "finalizer_report_sha256") != _json_sha256(body)
    ):
        raise Lfm2CanonicalFrontierReplayError(
            "Canonical finalizer report seal is invalid."
        )
    for name in (
        "run_contract_sha256",
        "frontier_sha256",
        "evidence_sha256",
        "plan_sha256",
        "finalizer_implementation_sha256",
        "canonical_allocator_implementation_sha256",
        "canonical_pack_binding_sha256",
        "guard_report_sha256",
    ):
        _digest(body.get(name), name)
    expected_ids = tuple(
        sorted(str(candidate.get("option_id")) for candidate in document["candidates"])
    )
    if (
        tuple(sorted(pack_binding)) != expected_ids
        or body["frontier_sha256"] != document["frontier_sha256"]
        or body["evidence_sha256"]
        != document["global_evidence"]["evidence_sha256"]
        or body["plan_sha256"] != document["targets"]["0.35"]["plan_sha256"]
        or body["canonical_pack_count"] != len(pack_binding)
        or body["canonical_pack_binding_sha256"]
        != _json_sha256(pack_binding)
        or body.get("exact_token_match_objective_used") is not False
        or any(
            body.get(name) != 0
            for name in (
                "clustering_calls",
                "parent_forward_calls",
                "parent_cache_build_calls",
                "parent_generation_calls",
                "optimizer_steps",
                "backward_calls",
            )
        )
    ):
        raise Lfm2CanonicalFrontierReplayError(
            "Canonical finalizer report does not bind this frontier and pack set."
        )
    return report


@dataclass(frozen=True, slots=True)
class Lfm2CanonicalFrontierReplay:
    """Authenticated tensors, exact plan, and a zero-operation replay receipt."""

    evidence: Lfm2GlobalCausalEvidence = field(repr=False)
    plan: Lfm2GlobalCausalPlan = field(repr=False)
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))


def replay_lfm2_cached_causal_frontier_from_canonical_packs(
    parent: Any,
    parent_cache: Any,
    parent_rows: Any,
    document: Mapping[str, Any],
    canonical_packs: Mapping[str, Lfm2CanonicalProposalPack],
    finalizer_report: Mapping[str, Any],
    *,
    maximum_stage_fraction: float = 0.10,
    max_states: int = 100_000,
) -> Lfm2CanonicalFrontierReplay:
    """Replay a sealed frontier without proposal or stage reclustering."""

    cached_causal_module.verify_lfm2_cached_causal_frontier_document(document)
    pack_binding = _canonical_pack_binding(canonical_packs)
    _verify_finalizer_report(finalizer_report, document, pack_binding)
    binding = document["parent_binding"]
    if (
        parent_cache.cache_key != binding["parent_trace_cache_key"]
        or parent_cache.corpus_fingerprint != binding["parent_corpus_sha256"]
        or parent_rows.trace_rows_sha256 != binding["parent_trace_rows_sha256"]
    ):
        raise Lfm2CanonicalFrontierReplayError(
            "Live parent cache differs from the frontier binding."
        )
    clustering_calls = 0
    original_clustering = (
        variable_width_fold_module.build_gated_mlp_activation_cluster_plan
    )

    def forbidden_clustering(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal clustering_calls
        clustering_calls += 1
        raise Lfm2CanonicalFrontierReplayError(
            "Canonical frontier replay is forbidden from reclustering."
        )

    def rebuild() -> tuple[Lfm2GlobalCausalEvidence, Lfm2GlobalCausalPlan]:
        variable_width_fold_module.build_gated_mlp_activation_cluster_plan = (
            forbidden_clustering
        )
        try:
            split = cached_causal_module._split_from_dict(document["split"])
            variable = (
                cached_causal_module.build_lfm2_causal_physical_source_evidence(
                    parent,
                    parent_rows,
                    split,
                    document["widths_by_layer"],
                    cache_key=parent_cache.cache_key,
                    heldout_corpus_sha256=parent_cache.corpus_fingerprint,
                )
            )
            if (
                variable.evidence_sha256
                != document["variable_width_evidence_sha256"]
            ):
                raise Lfm2CanonicalFrontierReplayError(
                    "Rebuilt physical source evidence differs from the frontier."
                )
            candidates = tuple(
                cached_causal_module._candidate_from_dict(value)
                for value in document["candidates"]
            )
            global_record = document["global_evidence"]
            weights = global_record["damage_weights"]
            cluster = global_record["cluster"]
            evidence = seal_lfm2_global_causal_evidence_from_canonical_packs(
                parent,
                variable,
                candidates,
                canonical_packs,
                parent_output_envelope_sha256=(
                    binding["parent_output_envelope_sha256"]
                ),
                task_anchor_manifest_sha256=(
                    binding["task_anchor_manifest_sha256"]
                ),
                damage_weights=Lfm2CausalDamageWeights(
                    final_logit_margin=weights["final_logit_margin"],
                    eos=weights["eos"],
                    repetition=weights["repetition"],
                    task_anchor=weights["task_anchor"],
                    residual_path=weights["residual_path"],
                    attention_path=weights["attention_path"],
                ),
                cluster_ridge=cluster["ridge"],
                cluster_max_iterations=cluster["max_iterations"],
                cluster_distance_chunk_size=cluster["distance_chunk_size"],
                parent_cache_build_count=0,
            )
            if evidence.evidence_sha256 != global_record["evidence_sha256"]:
                raise Lfm2CanonicalFrontierReplayError(
                    "Canonical replay evidence differs from the frontier seal."
                )
            plan = plan_lfm2_global_causal_target_from_canonical_packs(
                parent,
                evidence,
                canonical_packs,
                target_fraction=0.35,
                maximum_stage_fraction=maximum_stage_fraction,
                max_states=max_states,
            )
            if plan.plan_sha256 != document["targets"]["0.35"]["plan_sha256"]:
                raise Lfm2CanonicalFrontierReplayError(
                    "Canonical replay plan differs from the frontier seal."
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
            except Lfm2GlobalCausalInfeasible:
                pass
            else:  # pragma: no cover - current physical FFN ceiling
                raise Lfm2CanonicalFrontierReplayError(
                    "Frontier claims 50% infeasibility but canonical replay "
                    "found a physical plan."
                )
            return evidence, plan
        finally:
            variable_width_fold_module.build_gated_mlp_activation_cluster_plan = (
                original_clustering
            )

    parent_events: list[Mapping[str, Any]] = []
    ((result, parent_calls), guard) = run_fold_only_callback(
        lambda: audit_lfm2_parent_forward_calls(
            parent,
            rebuild,
            phase="lfm2-canonical-frontier-replay",
            event_sink=parent_events,
        ),
        phase="lfm2-canonical-frontier-replay",
        protected_models=(parent,),
    )
    if parent_calls or clustering_calls or not _guard_is_zero(guard.as_dict()):
        raise Lfm2CanonicalFrontierReplayError(
            "Canonical replay used a forbidden parent forward, clustering, "
            "or training operation."
        )
    evidence, plan = result
    receipt_body = {
        "schema": REPLAY_SCHEMA,
        "run_contract_sha256": finalizer_report["run_contract_sha256"],
        "finalizer_report_sha256": finalizer_report[
            "finalizer_report_sha256"
        ],
        "frontier_sha256": document["frontier_sha256"],
        "evidence_sha256": evidence.evidence_sha256,
        "plan_sha256": plan.plan_sha256,
        "canonical_pack_binding_sha256": _json_sha256(pack_binding),
        "canonical_pack_count": len(pack_binding),
        "guard_report_sha256": _json_sha256(guard.as_dict()),
        "clustering_calls": 0,
        "parent_forward_calls": 0,
        "parent_cache_build_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "exact_token_match_objective_used": False,
    }
    receipt = {
        **receipt_body,
        "replay_receipt_sha256": _json_sha256(receipt_body),
    }
    return Lfm2CanonicalFrontierReplay(
        evidence=evidence,
        plan=plan,
        receipt=receipt,
    )


__all__ = [
    "FINALIZER_SCHEMA",
    "REPLAY_SCHEMA",
    "Lfm2CanonicalFrontierReplay",
    "Lfm2CanonicalFrontierReplayError",
    "replay_lfm2_cached_causal_frontier_from_canonical_packs",
]
