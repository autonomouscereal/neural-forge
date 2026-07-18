"""Catalog-planned compiler-owned blank targets for LFM2 CPCF folding.

The CPCF backend must never infer a hand-written child shape from a parent
checkpoint.  This module binds the existing family-neutral parameter frontier,
LFM2 topology catalog, target registry, independent circuit inspector, and
compiler-owned zeroing path into one reusable operation.  It accepts either an
exact unique-parameter budget or a compression ratio and returns a genuinely
smaller dense blank whose parameter count and topology were certified before
any folding component is installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    BudgetedDynamicTargetPlan,
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesisResult,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_tokenizer_bridge import StrictTokenizerBridgeArtifact


LFM2_CPCF_BLANK_TARGET_SCHEMA = "neural-forge-lfm2-cpcf-blank-target-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2CpcfBlankTargetError(ValueError):
    """The requested CPCF blank cannot be planned or authenticated."""


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
        raise Lfm2CpcfBlankTargetError(
            "Blank-target metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfBlankTargetError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CpcfBlankTargetError(f"{name} must be nonempty text.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfBlankTargetError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True, slots=True)
class Lfm2CpcfBlankTarget:
    """One exact catalog plan and its independently authenticated zero blank."""

    plan: BudgetedDynamicTargetPlan
    synthesis: DynamicTargetSynthesisResult
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, BudgetedDynamicTargetPlan):
            raise TypeError("plan must be BudgetedDynamicTargetPlan.")
        if not isinstance(self.synthesis, DynamicTargetSynthesisResult):
            raise TypeError("synthesis must be DynamicTargetSynthesisResult.")
        if self.synthesis.request != self.plan.target_request:
            raise Lfm2CpcfBlankTargetError(
                "Synthesized target differs from its catalog plan."
            )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def model(self) -> Any:
        return self.synthesis.model

    @property
    def unique_parameters(self) -> int:
        return self.synthesis.resource_profile.unique_parameters

    @property
    def realized_compression_ratio(self) -> float:
        return float(self.report["realized_compression_ratio"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def plan_and_synthesize_lfm2_cpcf_blank(
    parent_model: Any,
    *,
    parent_name: str,
    parent_tokenizer_fingerprint: str,
    evidence_fingerprint: str,
    child_name: str,
    compression_ratio: str | int | float | None = None,
    target_unique_parameters: int | None = None,
    target_vocabulary_size: int | None = None,
    child_tokenizer_fingerprint: str | None = None,
    tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None,
    minimum_hidden_size: int = 32,
    maximum_candidates: int = 8,
    selected_candidate_rank: int | None = None,
    blank_seed: int = 0,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Lfm2CpcfBlankTarget:
    """Plan and materialize one exact dense zero blank through the registry."""

    name = _text(parent_name, "parent_name")
    child = _text(child_name, "child_name")
    parent_tokenizer = _digest(
        parent_tokenizer_fingerprint,
        "parent_tokenizer_fingerprint",
    )
    evidence = _digest(evidence_fingerprint, "evidence_fingerprint")
    supplied = (
        compression_ratio is not None,
        target_unique_parameters is not None,
    )
    if sum(supplied) != 1:
        raise Lfm2CpcfBlankTargetError(
            "Supply exactly one compression_ratio or target_unique_parameters."
        )
    if target_unique_parameters is not None:
        target_unique_parameters = _positive_int(
            target_unique_parameters,
            "target_unique_parameters",
        )
    if target_vocabulary_size is not None:
        target_vocabulary_size = _positive_int(
            target_vocabulary_size,
            "target_vocabulary_size",
        )
    minimum_hidden_size = _positive_int(
        minimum_hidden_size,
        "minimum_hidden_size",
    )
    maximum_candidates = _positive_int(
        maximum_candidates,
        "maximum_candidates",
    )
    selected_tokenizer = (
        parent_tokenizer
        if child_tokenizer_fingerprint is None
        else _digest(
            child_tokenizer_fingerprint,
            "child_tokenizer_fingerprint",
        )
    )
    if tokenizer_bridge_artifact is not None:
        if not isinstance(
            tokenizer_bridge_artifact,
            StrictTokenizerBridgeArtifact,
        ):
            raise Lfm2CpcfBlankTargetError(
                "tokenizer_bridge_artifact must be a strict bridge artifact."
            )
        tokenizer_bridge_artifact.verify_files()
        if (
            target_vocabulary_size
            != tokenizer_bridge_artifact.transport_profile.target_row_capacity
        ):
            raise Lfm2CpcfBlankTargetError(
                "Target vocabulary size differs from the strict bridge."
            )

    parent_state_before = fingerprint_model_state(parent_model)
    parent_graph = inspect_model_circuits(
        parent_model,
        model_name=name,
        tokenizer_fingerprint=parent_tokenizer,
    )
    if parent_graph.family != "lfm2":
        raise Lfm2CpcfBlankTargetError(
            "The selected parent is not an inspected LFM2 model."
        )
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 32),
        evidence_fingerprint=evidence,
        device=device,
        dtype=dtype,
        tokenizer_bridge_artifact=tokenizer_bridge_artifact,
    )
    adapter_registry = DynamicTargetAdapterRegistry((adapter,))
    topology_registry = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                minimum_hidden_size=minimum_hidden_size,
                tokenizer_bridge_artifact=tokenizer_bridge_artifact,
            ),
        )
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=adapter_registry,
        stack_adapters=DynamicFoldStackAdapterRegistry(),
        topology_catalogs=topology_registry,
    )
    budget = DenseParameterBudgetRequest(
        target_family="lfm2",
        target_modalities=("text",),
        target_unique_parameters=target_unique_parameters,
        compression_ratio=compression_ratio,
        target_vocabulary_size=target_vocabulary_size,
        match_policy=(
            "exact" if target_unique_parameters is not None else "at_or_below"
        ),
        maximum_candidates=maximum_candidates,
    )
    plan = system.plan_budgeted_target(
        parent_graph=parent_graph,
        budget_request=budget,
        child_name=child,
        selected_candidate_rank=selected_candidate_rank,
        tokenizer_fingerprint=selected_tokenizer,
        adapter_id=adapter.adapter_id,
        blank_seed=blank_seed,
    )
    synthesis = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=adapter_registry,
    ).synthesize_planned(
        plan=plan,
        parent_models={name: parent_model},
        parent_tokenizer_fingerprints={name: parent_tokenizer},
    )
    parent_state_after = fingerprint_model_state(parent_model)
    if parent_state_before != parent_state_after:
        raise Lfm2CpcfBlankTargetError(
            "Parent state changed while planning the blank target."
        )
    if not (
        synthesis.blank_provenance.compiler_created_blank
        and synthesis.blank_provenance.all_parameters_zero
        and synthesis.blank_provenance.dense_materialized_parameters
        and synthesis.topology_provenance_verified
    ):
        raise Lfm2CpcfBlankTargetError(
            "Target synthesis did not return an authenticated dense zero blank."
        )
    source_parameters = int(parent_graph.unique_parameter_count)
    target_parameters = int(synthesis.resource_profile.unique_parameters)
    if (
        target_unique_parameters is not None
        and target_parameters != target_unique_parameters
    ):
        raise Lfm2CpcfBlankTargetError(
            "Catalog target does not exactly match the requested parameter count."
        )
    if target_parameters >= source_parameters:
        raise Lfm2CpcfBlankTargetError(
            "Catalog target does not compact the parent parameter count."
        )
    candidate = plan.frontier.candidates[
        plan.target_certificate.selected_candidate_rank
    ]
    body = {
        "schema": LFM2_CPCF_BLANK_TARGET_SCHEMA,
        "parent_graph_fingerprint": parent_graph.fingerprint,
        "parent_state_fingerprint": parent_state_before,
        "parent_unique_parameters": source_parameters,
        "target_unique_parameters": target_parameters,
        "realized_compression_ratio": source_parameters / target_parameters,
        "requested_budget": budget.as_dict(),
        "frontier_fingerprint": plan.frontier.fingerprint,
        "target_certificate_sha256": plan.target_certificate.fingerprint,
        "selected_candidate_fingerprint": candidate.fingerprint,
        "selected_candidate_rank": (plan.target_certificate.selected_candidate_rank),
        "target_topology_fingerprint": synthesis.topology_fingerprint,
        "blank_state_fingerprint": synthesis.blank_state_fingerprint,
        "compiler_created_blank": True,
        "all_parameters_zero": True,
        "dense_materialized_parameters": True,
        "topology_provenance_verified": True,
        "parent_state_preserved": True,
        "quantization_used": False,
        "pruning_used": False,
        "training_used": False,
        "weights_installed": False,
    }
    return Lfm2CpcfBlankTarget(
        plan=plan,
        synthesis=synthesis,
        report={**body, "report_sha256": _sha(body)},
    )


__all__ = [
    "LFM2_CPCF_BLANK_TARGET_SCHEMA",
    "Lfm2CpcfBlankTarget",
    "Lfm2CpcfBlankTargetError",
    "plan_and_synthesize_lfm2_cpcf_blank",
]
