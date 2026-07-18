"""Catalog integration for the cached predictive circuit fold (CPCF).

The CPCF compiler is intentionally a model-level fold.  This adapter exposes
it through the same target-catalog, evidence, transaction, behavior-gate, and
fresh-reload contracts as every other production dynamic-fold lane.

Large parent-side phase tensors are authenticated while they are materialized
and rehashed at both sides of the compiler call.  A compile-scoped Torch
dispatch firewall additionally rejects every attempted write to the evidence,
phase inputs, tied interface, parent checkpoint, and live blank.  The compiler
receives only a disposable staging copy of the blank; an authenticated
``copy_`` installation is the sole mutation path into the live child.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .dynamic_behavior_gate import (
    BehaviorGateBinding,
    DynamicBehaviorGateError,
    resolve_behavior_gate_binding,
)
from .dynamic_circuit_ir import JsonScalar, ModelCircuitGraph, canonical_sha256
from .dynamic_fold_compiler import DynamicFoldSystemStack
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvaluationContext,
    FoldEvidence,
    FoldExecutionContext,
    FoldOperatorRegistry,
    FoldTransaction,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from .dynamic_fold_planner import DynamicFoldPlan, FoldAction
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
    encode_lfm2_layer_types,
    encode_lfm2_source_spans,
    validate_lfm2_dense_budgeted_child,
)
from .dynamic_lfm2_cpcf_reviewed_gate import (
    ORDERED_PHASE_BINDING_SCHEMA,
    REVIEWED_EVIDENCE_SCHEMA,
    Lfm2CpcfReviewedPromotionEvidence,
    Lfm2CpcfReviewedPromotionError,
    Lfm2CpcfReviewedPromotionGate,
    authenticate_lfm2_cpcf_reviewed_promotion_gate,
)
from .dynamic_target_synthesis import (
    DynamicTargetSynthesisError,
    DynamicTargetSynthesisResult,
    verify_compiler_authenticated_target,
)
from .hf_lfm2_cpcf_model_compiler import Lfm2CpcfTargetPhase
from .hf_lfm2_cpcf_model_fold import (
    LFM2_CPCF_MODEL_FOLD_SCHEMA,
    Lfm2CpcfModelFoldPolicy,
    compile_lfm2_cpcf_model_fold,
)
from .hf_lfm2_dense_budget_model_fold import DENSE_BUDGET_MODEL_METHOD
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceAdapter,
)

if TYPE_CHECKING:
    from .dynamic_lfm2_cpcf_resources import (
        Lfm2CpcfCachePhaseCoverageBinding,
    )


LFM2_CPCF_STACK_ADAPTER_ID = "neural_forge.lfm2.cpcf_catalog"
LFM2_CPCF_CATALOG_SCHEMA = "neural-forge-lfm2-cpcf-catalog-stack-v1"
LFM2_CPCF_BLANK_REPORT_SCHEMA = "neural-forge-lfm2-cpcf-catalog-blank-v1"
LFM2_CPCF_PARENT_TENSOR_SEAL_SCHEMA = (
    "neural-forge-lfm2-cpcf-parent-tensor-seal-v1"
)
LFM2_CPCF_PARAMETER_PATH_WRITE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-cpcf-parameter-path-write-receipt-v1"
)
LFM2_CPCF_STAGED_INSTALL_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-cpcf-staged-install-receipt-v1"
)
LFM2_CPCF_COMPILE_WRITE_GUARD_SCHEMA = (
    "neural-forge-lfm2-cpcf-compile-write-guard-v1"
)
LFM2_CPCF_CATALOG_COMPILE_REPORT_SCHEMA = (
    "neural-forge-lfm2-cpcf-catalog-compile-report-v1"
)
LFM2_CPCF_RESOURCE_SCHEMA = "neural-forge-lfm2-cpcf-catalog-resource-v1"
LFM2_CPCF_EXTERNAL_GATE_SCHEMA = (
    "neural-forge-lfm2-cpcf-external-intelligence-gate-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2CpcfCatalogError(RuntimeError):
    """The CPCF catalog stack is incomplete or unauthenticated."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for LFM2 CPCF folding.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfCatalogError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CpcfCatalogError(f"{name} must be nonempty text.")
    return value.strip()


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
        raise Lfm2CpcfCatalogError(
            "CPCF catalog metadata must be finite canonical JSON."
        ) from exc


def _tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfCatalogError(
            f"{name} must be a materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


def _phase_body(phase: Lfm2CpcfTargetPhase) -> dict[str, Any]:
    return {
        "schema": "neural-forge-lfm2-cpcf-target-phase-v1",
        "batch_index": phase.batch_index,
        "record_id": phase.record_id,
        "split": phase.split,
        "phase": phase.phase,
        "donor_input_sha256": phase.donor_input_sha256,
        "target_alignment_sha256": phase.target_alignment_sha256,
        "input_ids_sha256": _tensor_sha256(
            phase.input_ids,
            "phase.input_ids",
        ),
        "attention_mask_sha256": _tensor_sha256(
            phase.attention_mask,
            "phase.attention_mask",
        ),
        "position_ids_sha256": _tensor_sha256(
            phase.position_ids,
            "phase.position_ids",
        ),
        "batch_size": phase.batch_size,
        "full_prefix_tokens": phase.full_prefix_tokens,
        "sampled_suffix_tokens": phase.sampled_suffix_tokens,
        "parent_evidence_sha256": phase.parent_evidence.evidence_sha256,
        "numeric_token_id_equality_required": False,
        "actual_child_full_prefix_required": True,
    }


def _phase_metadata(phase: Lfm2CpcfTargetPhase) -> dict[str, Any]:
    body = _phase_body(phase)
    observed_sha = canonical_sha256(body)
    if observed_sha != phase.phase_sha256:
        raise Lfm2CpcfCatalogError(
            f"CPCF target phase {phase.identity!r} failed its metadata seal."
        )
    return {
        **phase.as_dict(),
        "phase_sha256": observed_sha,
    }


def _parent_tensor_path(
    phase: Lfm2CpcfTargetPhase,
    *,
    layer_index: int | None,
    field_name: str,
) -> str:
    prefix = (
        f"phase[{phase.batch_index}:{phase.phase}]"
        if layer_index is None
        else (
            f"phase[{phase.batch_index}:{phase.phase}]"
            f".layer[{layer_index}]"
        )
    )
    return f"{prefix}.{field_name}"


def _parent_tensor_bindings(
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> tuple[tuple[str, Any], ...]:
    bindings: list[tuple[str, Any]] = []
    for phase in phases:
        evidence = phase.parent_evidence
        for layer in evidence.layers:
            bindings.extend(
                (
                    _parent_tensor_path(
                        phase,
                        layer_index=layer.layer_index,
                        field_name=str(name),
                    ),
                    tensor,
                )
                for name, tensor in sorted(layer.tensors.items())
            )
        bindings.extend(
            (
                _parent_tensor_path(
                    phase,
                    layer_index=None,
                    field_name=str(name),
                ),
                tensor,
            )
            for name, tensor in sorted(evidence.final_norm_tensors.items())
        )
    paths = tuple(path for path, _tensor in bindings)
    if len(set(paths)) != len(paths):
        raise Lfm2CpcfCatalogError(
            "CPCF parent evidence exposes duplicate tensor paths."
        )
    return tuple(bindings)


def _authenticate_parent_phase_tensors(
    phase: Lfm2CpcfTargetPhase,
) -> dict[str, str]:
    """Recompute the canonical installability receipts for one phase.

    ``as_dict`` is intentionally used here because the installability
    evidence classes own the canonical tensor-SHA implementation.  This keeps
    the catalog seal byte-for-byte aligned with the cache authenticator.
    """

    evidence = phase.parent_evidence
    document = evidence.as_dict()
    layer_documents = document.get("layers")
    if (
        document.get("cache_key") != evidence.cache_key
        or document.get("batch_index") != evidence.batch_index
        or document.get("input_sha256") != evidence.input_sha256
        or document.get("phase") != evidence.phase
        or not isinstance(layer_documents, list)
        or len(layer_documents) != len(evidence.layers)
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF parent phase document differs from its typed evidence."
        )
    authenticated: dict[str, str] = {}
    layer_sha: list[str] = []
    for layer, layer_document in zip(
        evidence.layers,
        layer_documents,
        strict=True,
    ):
        inventory = layer_document.get("field_inventory")
        if (
            layer_document.get("layer_index") != layer.layer_index
            or layer_document.get("operator_kind") != layer.operator_kind
            or layer_document.get("source_width") != layer.source_width
            or not isinstance(inventory, Mapping)
            or set(inventory) != set(layer.tensors)
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF parent layer inventory differs from its tensor bank."
            )
        body = {
            "schema": layer_document.get("schema"),
            "cache_key": evidence.cache_key,
            "batch_index": evidence.batch_index,
            "phase": evidence.phase,
            "layer_index": layer.layer_index,
            "operator_kind": layer.operator_kind,
            "source_width": layer.source_width,
            "field_inventory": dict(inventory),
            "functional_identities_verified": True,
        }
        observed_layer_sha = canonical_sha256(body)
        if observed_layer_sha != layer.evidence_sha256:
            raise Lfm2CpcfCatalogError(
                "CPCF parent layer tensor authentication changed."
            )
        layer_sha.append(observed_layer_sha)
        for name, row in inventory.items():
            if not isinstance(row, Mapping):
                raise Lfm2CpcfCatalogError(
                    "CPCF parent tensor inventory row is invalid."
                )
            authenticated[
                _parent_tensor_path(
                    phase,
                    layer_index=layer.layer_index,
                    field_name=str(name),
                )
            ] = _digest(
                row.get("sha256"),
                "parent tensor inventory sha256",
            )
    final_inventory = document.get("final_norm_field_inventory")
    if (
        not isinstance(final_inventory, Mapping)
        or set(final_inventory) != set(evidence.final_norm_tensors)
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF final-normalization inventory differs from its tensor bank."
        )
    for name, row in final_inventory.items():
        if not isinstance(row, Mapping):
            raise Lfm2CpcfCatalogError(
                "CPCF final-normalization inventory row is invalid."
            )
        authenticated[
            _parent_tensor_path(
                phase,
                layer_index=None,
                field_name=str(name),
            )
        ] = _digest(
            row.get("sha256"),
            "final-normalization tensor inventory sha256",
        )
    phase_body = {
        "schema": document.get("schema"),
        "cache_key": evidence.cache_key,
        "batch_index": evidence.batch_index,
        "input_sha256": evidence.input_sha256,
        "phase": evidence.phase,
        "layer_evidence_sha256": layer_sha,
        "final_norm_field_inventory": dict(final_inventory),
        "parent_side_compiler_inputs_ready": True,
    }
    if canonical_sha256(phase_body) != evidence.evidence_sha256:
        raise Lfm2CpcfCatalogError(
            "CPCF parent phase tensor authentication changed."
        )
    return authenticated


def _authenticate_parent_tensors(
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> Mapping[str, str]:
    authenticated: dict[str, str] = {}
    for phase in phases:
        phase_tensors = _authenticate_parent_phase_tensors(phase)
        overlap = set(authenticated).intersection(phase_tensors)
        if overlap:
            raise Lfm2CpcfCatalogError(
                "CPCF parent tensor authentication paths overlap."
            )
        authenticated.update(phase_tensors)
    if set(authenticated) != {
        path for path, _tensor in _parent_tensor_bindings(phases)
    }:
        raise Lfm2CpcfCatalogError(
            "CPCF parent tensor authentication is incomplete."
        )
    return MappingProxyType(authenticated)


@dataclass(frozen=True, slots=True)
class _ParentTensorMutationSeal:
    path: str
    object_id: int
    storage_object_id: int
    storage_data_ptr: int
    storage_nbytes: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    version: int
    content_sha256: str


def _parent_tensor_mutation_seal(
    path: str,
    tensor: Any,
    *,
    content_sha256: str,
) -> _ParentTensorMutationSeal:
    torch = _torch()
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout != torch.strided
        or tensor.device.type == "meta"
    ):
        raise Lfm2CpcfCatalogError(
            f"CPCF parent evidence tensor {path!r} is not materialized dense "
            "storage."
        )
    storage = tensor.untyped_storage()
    version = getattr(tensor, "_version", None)
    if isinstance(version, bool) or not isinstance(version, int):
        raise Lfm2CpcfCatalogError(
            f"CPCF parent evidence tensor {path!r} exposes no mutation "
            "version counter."
        )
    return _ParentTensorMutationSeal(
        path=path,
        object_id=id(tensor),
        storage_object_id=id(storage),
        storage_data_ptr=int(storage.data_ptr()),
        storage_nbytes=int(storage.nbytes()),
        storage_offset=int(tensor.storage_offset()),
        shape=tuple(int(size) for size in tensor.shape),
        stride=tuple(int(value) for value in tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        version=version,
        content_sha256=_digest(
            content_sha256,
            f"parent tensor {path} content_sha256",
        ),
    )


def _blank_report(
    target: DynamicTargetSynthesisResult,
    selected: Lfm2DenseBudgetedSelectedTarget,
) -> Mapping[str, Any]:
    try:
        topology = verify_compiler_authenticated_target(target)
    except DynamicTargetSynthesisError as exc:
        raise Lfm2CpcfCatalogError(
            "CPCF target lacks compiler-authenticated blank provenance."
        ) from exc
    provenance = target.blank_provenance
    if (
        provenance is None
        or not provenance.compiler_created_blank
        or not provenance.all_parameters_zero
        or not provenance.dense_materialized_parameters
        or not target.topology_provenance_verified
        or fingerprint_model_state(target.model)
        != target.blank_state_fingerprint
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF requires the live compiler-created dense zero target."
        )
    body = {
        "schema": LFM2_CPCF_BLANK_REPORT_SCHEMA,
        "compiler_created_blank": True,
        "all_parameters_zero": True,
        "dense_materialized_parameters": True,
        "topology_provenance_verified": True,
        "weights_installed": False,
        "training_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "target_unique_parameters": (
            selected.exact_target_unique_parameters
        ),
        "target_graph_fingerprint": target.graph.fingerprint,
        "target_topology_fingerprint": topology,
        "target_blank_state_fingerprint": target.blank_state_fingerprint,
        "target_blank_provenance_digest": target.blank_provenance_digest,
        "source_graph_fingerprint": selected.source_graph_fingerprint,
        "source_state_fingerprint": selected.source_state_fingerprint,
    }
    return MappingProxyType(
        {**body, "report_sha256": canonical_sha256(body)}
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfExternalBehaviorGateContract:
    """Proof that promotion is independent, free-running, and semantic.

    This is deliberately a *reviewed promotion-rerun* contract.  A candidate's
    manual packet digest cannot exist before the first fold.  That first live
    run must therefore remain quarantined and use the packet-capture workflow;
    it cannot instantiate this production stack or publish.  Only a
    deterministic rerun bound to the authenticated reviewed packet can pass
    stack construction and reach the publication gates.
    """

    independent_holdout_manifest_sha256: str
    independent_parent_envelope_sha256: str
    parent_relative_manual_packet_sha256: str
    independent_holdout_case_count: int = 64
    candidate_free_generation: bool = True
    parent_relative_manual_scoring: bool = True
    forced_decode_only: bool = False
    exact_text_match_required: bool = False
    contract_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "independent_holdout_manifest_sha256",
            "independent_parent_envelope_sha256",
            "parent_relative_manual_packet_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        if (
            isinstance(self.independent_holdout_case_count, bool)
            or not isinstance(self.independent_holdout_case_count, int)
            or self.independent_holdout_case_count < 64
            or self.candidate_free_generation is not True
            or self.parent_relative_manual_scoring is not True
            or self.forced_decode_only is not False
            or self.exact_text_match_required is not False
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF promotion requires at least 64 independent free-"
                "generation cases plus parent-relative semantic manual review."
            )
        object.__setattr__(
            self,
            "contract_sha256",
            canonical_sha256(self.body()),
        )

    def body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_EXTERNAL_GATE_SCHEMA,
            "lifecycle": "reviewed-promotion-rerun",
            "first_fold_quarantine_required": True,
            "first_fold_publication_allowed": False,
            "independent_holdout_manifest_sha256": (
                self.independent_holdout_manifest_sha256
            ),
            "independent_parent_envelope_sha256": (
                self.independent_parent_envelope_sha256
            ),
            "parent_relative_manual_packet_sha256": (
                self.parent_relative_manual_packet_sha256
            ),
            "independent_holdout_case_count": (
                self.independent_holdout_case_count
            ),
            "candidate_free_generation": self.candidate_free_generation,
            "parent_relative_manual_scoring": (
                self.parent_relative_manual_scoring
            ),
            "forced_decode_only": self.forced_decode_only,
            "exact_text_match_required": self.exact_text_match_required,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.body(), "contract_sha256": self.contract_sha256}


def _strict_behavior_binding(
    value: Any,
    contract: Lfm2CpcfExternalBehaviorGateContract,
    *,
    name: str,
) -> BehaviorGateBinding:
    if type(value) is not Lfm2CpcfReviewedPromotionGate:
        raise Lfm2CpcfCatalogError(
            f"{name} must be the exact typed CPCF reviewed-promotion gate."
        )
    try:
        authenticate_lfm2_cpcf_reviewed_promotion_gate(value)
    except Lfm2CpcfReviewedPromotionError as exc:
        raise Lfm2CpcfCatalogError(
            f"{name} is not a constructor-authenticated immutable reviewed "
            "promotion gate."
        ) from exc
    evidence = getattr(value, "evidence", None)
    if not isinstance(evidence, Lfm2CpcfReviewedPromotionEvidence):
        raise Lfm2CpcfCatalogError(
            f"{name} has no typed reviewed-promotion evidence."
        )
    reviewed = dict(evidence.reviewed_evidence)
    if (
        reviewed.get("case_count")
        != contract.independent_holdout_case_count
        or reviewed.get("manifest_sha256")
        != contract.independent_holdout_manifest_sha256
        or reviewed.get("parent_envelope_sha256")
        != contract.independent_parent_envelope_sha256
        or reviewed.get("parent_relative_manual_packet_sha256")
        != contract.parent_relative_manual_packet_sha256
        or reviewed.get("parent_relative_manual_report_passed") is not True
        or reviewed.get("automatic_gate_ready_for_manual_completion")
        is not True
    ):
        raise Lfm2CpcfCatalogError(
            f"{name} reviewed evidence differs from the external promotion "
            "contract."
        )
    try:
        binding = resolve_behavior_gate_binding(value)
    except DynamicBehaviorGateError as exc:
        raise Lfm2CpcfCatalogError(
            f"{name} must expose a canonical behavior-gate manifest."
        ) from exc
    components = binding.manifest.get("components")
    reviewed_components = (
        []
        if not isinstance(components, list)
        else [
            component
            for component in components
            if isinstance(component, Mapping)
            and component.get("schema") == REVIEWED_EVIDENCE_SCHEMA
        ]
    )
    if (
        binding.baseline_fingerprint
        != evidence.full_reviewed_evidence_fingerprint
        or len(reviewed_components) != 1
        or reviewed_components[0].get("reviewed_evidence_fingerprint")
        != evidence.reviewed_evidence_fingerprint
        or reviewed_components[0].get("candidate_state_fingerprint")
        != evidence.candidate_state_fingerprint
        or reviewed_components[0].get("candidate_outputs_sha256")
        != evidence.candidate_outputs_sha256
    ):
        raise Lfm2CpcfCatalogError(
            f"{name} behavior manifest is not bound to the exact reviewed "
            "candidate state, outputs, and evidence chain."
        )
    return binding


def _authenticate_cpcf_resource_coverage(
    *,
    coverage_binding: Lfm2CpcfCachePhaseCoverageBinding,
    phases: Sequence[Lfm2CpcfTargetPhase],
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    parent_graph_fingerprint: str,
    policy: Lfm2CpcfModelFoldPolicy,
    reviewed: Lfm2CpcfReviewedPromotionEvidence,
) -> dict[str, Any]:
    """Cross-bind one exact cache/trace/phase root to reviewed compilation."""

    # Local import avoids a module-load cycle: the production resource factory
    # imports this adapter in order to construct Lfm2CpcfStackResources.
    from .dynamic_lfm2_cpcf_resources import (
        Lfm2CpcfCachePhaseCoverageBinding,
    )

    if type(coverage_binding) is not Lfm2CpcfCachePhaseCoverageBinding:
        raise Lfm2CpcfCatalogError(
            "coverage_binding must be the exact typed CPCF cache-phase "
            "coverage binding."
        )
    if (
        canonical_sha256(coverage_binding.body())
        != coverage_binding.binding_fingerprint
        or any(
            canonical_sha256(record.body()) != record.record_fingerprint
            for record in coverage_binding.records
        )
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF cache-phase coverage binding failed its canonical seal."
        )

    ordered_phases = tuple(phases)
    phase_sha256 = tuple(
        _phase_metadata(phase)["phase_sha256"] for phase in ordered_phases
    )
    parent_evidence_sha256 = tuple(
        _digest(
            phase.parent_evidence.evidence_sha256,
            "phase parent_evidence.evidence_sha256",
        )
        for phase in ordered_phases
    )
    phase_identities = tuple(phase.identity for phase in ordered_phases)
    phase_cache_keys = {
        _digest(
            phase.parent_evidence.cache_key,
            "phase parent_evidence.cache_key",
        )
        for phase in ordered_phases
    }
    if len(phase_cache_keys) != 1:
        raise Lfm2CpcfCatalogError(
            "CPCF stack resources must bind one parent cache."
        )
    cache_key = phase_cache_keys.pop()
    coverage_phase_sha256 = tuple(
        value
        for record in coverage_binding.records
        for value in record.target_phase_sha256
    )
    coverage_parent_evidence_sha256 = tuple(
        value
        for record in coverage_binding.records
        for value in record.parent_evidence_sha256
    )
    coverage_phase_identities = tuple(
        (record.ordinal, phase_name)
        for record in coverage_binding.records
        for phase_name in record.phase_names
    )
    trace_records = tuple(trace_adapter.records)
    if (
        coverage_binding.cache_key != cache_key
        or coverage_binding.cache_key != reviewed.parent_cache_key
        or coverage_binding.cache_parent_fingerprint
        != reviewed.cache_live_parent_fingerprint
        or coverage_binding.parent_graph_fingerprint
        != parent_graph_fingerprint
        or coverage_binding.parent_graph_fingerprint
        != reviewed.blank_target_parent_graph_fingerprint
        or coverage_binding.trace_adapter_sha256
        != trace_adapter.evidence_sha256
        or coverage_binding.trace_adapter_sha256
        != reviewed.trace_adapter_sha256
        or coverage_binding.trajectory_corpus_sha256
        != reviewed.trajectory_corpus_sha256
        or coverage_binding.cache_batch_count != trace_adapter.record_count
        or tuple(record.record_id for record in coverage_binding.records)
        != tuple(record.record_id for record in trace_records)
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF coverage cache, parent identity, graph, trace adapter, or "
            "trajectory corpus differs from the stack and reviewed evidence."
        )
    if (
        coverage_binding.phase_count != len(phase_sha256)
        or coverage_phase_identities != phase_identities
        or coverage_phase_sha256 != phase_sha256
        or coverage_parent_evidence_sha256 != parent_evidence_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF coverage does not bind the exact ordered resource phase "
            "identities, target seals, and parent-evidence seals."
        )

    policy_sha256 = canonical_sha256(policy.as_dict())
    reviewed_phase_sha256 = tuple(
        reviewed.compiled_model_fold_phase_sha256
    )
    phase_binding_sha256 = canonical_sha256(
        {
            "schema": ORDERED_PHASE_BINDING_SCHEMA,
            "phase_count": len(phase_sha256),
            "ordered_phase_sha256": list(phase_sha256),
        }
    )
    if (
        reviewed.compiled_model_fold_phase_count != len(phase_sha256)
        or reviewed_phase_sha256 != phase_sha256
        or reviewed.compiled_model_fold_phase_binding_sha256
        != phase_binding_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF resource phases differ from the reviewed compiler phase "
            "sequence or binding seal."
        )
    if (
        reviewed.compiled_model_fold_policy_sha256 != policy_sha256
        or canonical_sha256(dict(reviewed.compiled_model_fold_policy))
        != policy_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF resource policy differs from the reviewed compiler policy "
            "seal."
        )

    return {
        "coverage_binding_fingerprint": (
            coverage_binding.binding_fingerprint
        ),
        "coverage_cache_manifest_sha256": (
            coverage_binding.cache_manifest_sha256
        ),
        "coverage_cache_parent_fingerprint": (
            coverage_binding.cache_parent_fingerprint
        ),
        "coverage_cache_corpus_fingerprint": (
            coverage_binding.cache_corpus_fingerprint
        ),
        "coverage_trajectory_corpus_sha256": (
            coverage_binding.trajectory_corpus_sha256
        ),
        "coverage_phase_count": coverage_binding.phase_count,
        "reviewed_compiled_model_fold_result_sha256": _digest(
            reviewed.compiled_model_fold_result_sha256,
            "reviewed compiled model-fold result SHA-256",
        ),
        "reviewed_compiled_model_fold_policy_sha256": _digest(
            reviewed.compiled_model_fold_policy_sha256,
            "reviewed compiled model-fold policy SHA-256",
        ),
        "reviewed_compiled_model_fold_phase_binding_sha256": _digest(
            reviewed.compiled_model_fold_phase_binding_sha256,
            "reviewed compiled model-fold phase binding SHA-256",
        ),
        "reviewed_compiled_blank_target_report_sha256": _digest(
            reviewed.compiled_blank_target_report_sha256,
            "reviewed compiled blank-target report SHA-256",
        ),
        "reviewed_compiled_parameter_receipt_sha256": _digest(
            reviewed.compiled_parameter_receipt_sha256,
            "reviewed compiled parameter receipt SHA-256",
        ),
    }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfStackResources:
    """Reusable, already-materialized CPCF evidence and publication callbacks."""

    phases: tuple[Lfm2CpcfTargetPhase, ...]
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter
    coverage_binding: Lfm2CpcfCachePhaseCoverageBinding
    parent_graph_fingerprint: str
    source_tied_table: Any
    policy: Lfm2CpcfModelFoldPolicy
    external_behavior_contract: Lfm2CpcfExternalBehaviorGateContract
    expected_full_reviewed_evidence_fingerprint: str
    final_evaluator: Lfm2CpcfReviewedPromotionGate
    model_saver: Callable[[Any, Path], None]
    model_loader: Callable[[Path], Any]
    fresh_reload_evaluator: Lfm2CpcfReviewedPromotionGate
    saver_id: str
    loader_id: str
    snapshot_config: DynamicFoldSnapshotConfig | None = None
    resource_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        phases = tuple(self.phases)
        if not phases or any(
            not isinstance(value, Lfm2CpcfTargetPhase) for value in phases
        ):
            raise Lfm2CpcfCatalogError(
                "phases must contain typed CPCF target phases."
            )
        object.__setattr__(self, "phases", phases)
        if not isinstance(
            self.trace_adapter,
            Lfm2StrictTokenizerBridgeTraceAdapter,
        ):
            raise Lfm2CpcfCatalogError(
                "trace_adapter must be a strict tokenizer bridge adapter."
            )
        self.trace_adapter.verify()
        object.__setattr__(
            self,
            "parent_graph_fingerprint",
            _digest(
                self.parent_graph_fingerprint,
                "parent_graph_fingerprint",
            ),
        )
        if not isinstance(self.policy, Lfm2CpcfModelFoldPolicy):
            raise Lfm2CpcfCatalogError(
                "policy must be Lfm2CpcfModelFoldPolicy."
            )
        if not isinstance(
            self.external_behavior_contract,
            Lfm2CpcfExternalBehaviorGateContract,
        ):
            raise Lfm2CpcfCatalogError(
                "external_behavior_contract has the wrong type."
            )
        object.__setattr__(
            self,
            "expected_full_reviewed_evidence_fingerprint",
            _digest(
                self.expected_full_reviewed_evidence_fingerprint,
                "expected_full_reviewed_evidence_fingerprint",
            ),
        )
        for name in ("model_saver", "model_loader"):
            if not callable(getattr(self, name)):
                raise Lfm2CpcfCatalogError(
                    f"CPCF stack resource {name} must be callable."
                )
        if (
            type(self.final_evaluator) is not Lfm2CpcfReviewedPromotionGate
            or self.fresh_reload_evaluator is not self.final_evaluator
            or not callable(getattr(self.final_evaluator, "evaluate", None))
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF final and fresh-reload evaluation must use one exact "
                "typed reviewed-promotion gate."
            )
        final_binding = _strict_behavior_binding(
            self.final_evaluator,
            self.external_behavior_contract,
            name="final_evaluator",
        )
        reload_binding = _strict_behavior_binding(
            self.fresh_reload_evaluator,
            self.external_behavior_contract,
            name="fresh_reload_evaluator",
        )
        if (
            final_binding.manifest_fingerprint
            != reload_binding.manifest_fingerprint
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF final and fresh-reload behavior manifests differ."
        )
        reviewed = self.final_evaluator.evidence
        if (
            reviewed.full_reviewed_evidence_fingerprint
            != self.expected_full_reviewed_evidence_fingerprint
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF reviewed promotion gate differs from the independently "
                "pinned full reviewed-evidence fingerprint."
            )
        if (
            reviewed.bridge_artifact_sha256
            != self.trace_adapter.bridge_artifact_sha256
            or reviewed.bridge_profile_sha256
            != self.trace_adapter.transport_profile_fingerprint
            or reviewed.target_tokenizer_semantics_sha256
            != self.trace_adapter.target_tokenizer_semantics_sha256
            or reviewed.candidate_tokenizer_fingerprint
            != self.trace_adapter.target_tokenizer_package_fingerprint
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF reviewed promotion evidence is not bound to the exact "
                "strict tokenizer bridge artifact, transport profile, and "
                "target tokenizer package/semantics used by the trace "
                "adapter."
            )
        coverage_fields = _authenticate_cpcf_resource_coverage(
            coverage_binding=self.coverage_binding,
            phases=phases,
            trace_adapter=self.trace_adapter,
            parent_graph_fingerprint=self.parent_graph_fingerprint,
            policy=self.policy,
            reviewed=reviewed,
        )
        object.__setattr__(self, "saver_id", _text(self.saver_id, "saver_id"))
        object.__setattr__(self, "loader_id", _text(self.loader_id, "loader_id"))
        if self.snapshot_config is not None and not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise Lfm2CpcfCatalogError(
                "snapshot_config must be DynamicFoldSnapshotConfig or None."
            )
        table = self.source_tied_table
        torch = _torch()
        if (
            not isinstance(table, torch.Tensor)
            or table.layout != torch.strided
            or table.device.type == "meta"
            or table.ndim != 2
            or table.numel() == 0
            or not torch.is_floating_point(table)
            or table.requires_grad
            or table.grad_fn is not None
            or not bool(torch.isfinite(table.detach()).all().item())
        ):
            raise Lfm2CpcfCatalogError(
                "source_tied_table must be a finite detached dense rank-two "
                "floating tensor."
            )
        phase_cache_keys = {
            phase.parent_evidence.cache_key for phase in phases
        }
        if len(phase_cache_keys) != 1:
            raise Lfm2CpcfCatalogError(
                "CPCF stack resources must bind one parent cache."
            )
        resource_body = {
            "schema": LFM2_CPCF_RESOURCE_SCHEMA,
            "cache_key": _digest(
                phase_cache_keys.pop(),
                "resource cache_key",
            ),
            "ordered_phase_sha256": [
                _phase_metadata(phase)["phase_sha256"]
                for phase in phases
            ],
            "ordered_parent_evidence_sha256": [
                _digest(
                    phase.parent_evidence.evidence_sha256,
                    "phase parent_evidence.evidence_sha256",
                )
                for phase in phases
            ],
            "trace_adapter_sha256": _digest(
                self.trace_adapter.evidence_sha256,
                "trace_adapter.evidence_sha256",
            ),
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "bridge_artifact_sha256": _digest(
                self.trace_adapter.bridge_artifact_sha256,
                "trace_adapter.bridge_artifact_sha256",
            ),
            "transport_profile_sha256": _digest(
                self.trace_adapter.transport_profile_fingerprint,
                "trace_adapter.transport_profile_fingerprint",
            ),
            "source_tied_table_sha256": _tensor_sha256(
                table,
                "source_tied_table",
            ),
            "policy_sha256": canonical_sha256(self.policy.as_dict()),
            **coverage_fields,
            "external_behavior_contract_sha256": (
                self.external_behavior_contract.contract_sha256
            ),
            "behavior_gate_manifest_fingerprint": (
                final_binding.manifest_fingerprint
            ),
            "reviewed_bridge_artifact_sha256": (
                reviewed.bridge_artifact_sha256
            ),
            "reviewed_bridge_profile_sha256": (
                reviewed.bridge_profile_sha256
            ),
            "reviewed_target_tokenizer_semantics_sha256": (
                reviewed.target_tokenizer_semantics_sha256
            ),
            "reviewed_candidate_tokenizer_fingerprint": (
                reviewed.candidate_tokenizer_fingerprint
            ),
            "expected_full_reviewed_evidence_fingerprint": (
                self.expected_full_reviewed_evidence_fingerprint
            ),
            "saver_id": self.saver_id,
            "loader_id": self.loader_id,
            "snapshot_config": (
                None
                if self.snapshot_config is None
                else self.snapshot_config.as_dict()
            ),
            "target_blank_fingerprint_bound": False,
        }
        object.__setattr__(
            self,
            "resource_fingerprint",
            canonical_sha256(resource_body),
        )


class Lfm2CpcfSealedEvidence:
    """Executor-safe seal over large parent evidence and target-bound metadata."""

    __slots__ = (
        "_phases",
        "_trace_adapter",
        "_source_tied_table",
        "_blank_target_report",
        "_policy",
        "_cache_key",
        "_phase_sha256",
        "_parent_evidence_sha256",
        "_parent_tensor_seals",
        "_parent_tensor_content_sha256",
        "_parent_tensor_content_aggregate_sha256",
        "_trace_adapter_sha256",
        "_bridge_artifact_sha256",
        "_source_tied_table_sha256",
        "_blank_report_sha256",
        "_policy_sha256",
        "_bundle_sha256",
    )

    def __init__(
        self,
        *,
        phases: Sequence[Lfm2CpcfTargetPhase],
        trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
        source_tied_table: Any,
        blank_target_report: Mapping[str, Any],
        policy: Lfm2CpcfModelFoldPolicy,
    ) -> None:
        ordered = tuple(phases)
        cache_keys = {
            phase.parent_evidence.cache_key for phase in ordered
        }
        if len(cache_keys) != 1:
            raise Lfm2CpcfCatalogError(
                "All CPCF phases must share one parent cache key."
            )
        self._phases = ordered
        self._trace_adapter = trace_adapter.verify()
        self._source_tied_table = source_tied_table
        self._blank_target_report = MappingProxyType(
            dict(blank_target_report)
        )
        self._policy = policy
        self._cache_key = _digest(cache_keys.pop(), "phase cache_key")
        self._phase_sha256 = tuple(
            _phase_metadata(phase)["phase_sha256"] for phase in ordered
        )
        self._parent_evidence_sha256 = tuple(
            _digest(
                phase.parent_evidence.evidence_sha256,
                "phase parent_evidence.evidence_sha256",
            )
            for phase in ordered
        )
        parent_content = _authenticate_parent_tensors(ordered)
        self._parent_tensor_content_sha256 = parent_content
        self._parent_tensor_seals = tuple(
            _parent_tensor_mutation_seal(
                path,
                tensor,
                content_sha256=parent_content[path],
            )
            for path, tensor in _parent_tensor_bindings(ordered)
        )
        self._parent_tensor_content_aggregate_sha256 = canonical_sha256(
            [
                {
                    "path": path,
                    "content_sha256": parent_content[path],
                }
                for path in sorted(parent_content)
            ]
        )
        self._trace_adapter_sha256 = _digest(
            trace_adapter.evidence_sha256,
            "trace_adapter.evidence_sha256",
        )
        self._bridge_artifact_sha256 = _digest(
            trace_adapter.bridge_artifact_sha256,
            "trace_adapter.bridge_artifact_sha256",
        )
        self._source_tied_table_sha256 = _tensor_sha256(
            source_tied_table,
            "source_tied_table",
        )
        self._blank_report_sha256 = _digest(
            blank_target_report.get("report_sha256"),
            "blank_target_report.report_sha256",
        )
        self._policy_sha256 = canonical_sha256(policy.as_dict())
        self._bundle_sha256 = canonical_sha256(self._body())
        self.verify()

    @property
    def cache_key(self) -> str:
        return self._cache_key

    @property
    def bundle_sha256(self) -> str:
        return self._bundle_sha256

    @property
    def phases(self) -> tuple[Lfm2CpcfTargetPhase, ...]:
        return self._phases

    @property
    def trace_adapter(self) -> Lfm2StrictTokenizerBridgeTraceAdapter:
        return self._trace_adapter

    @property
    def source_tied_table(self) -> Any:
        return self._source_tied_table

    @property
    def blank_target_report(self) -> Mapping[str, Any]:
        return self._blank_target_report

    @property
    def policy(self) -> Lfm2CpcfModelFoldPolicy:
        return self._policy

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_CATALOG_SCHEMA,
            "cache_key": self._cache_key,
            "ordered_phase_sha256": list(self._phase_sha256),
            "ordered_parent_evidence_sha256": list(
                self._parent_evidence_sha256
            ),
            "parent_tensor_content_aggregate_sha256": (
                self._parent_tensor_content_aggregate_sha256
            ),
            "parent_tensor_seal_schema": (
                LFM2_CPCF_PARENT_TENSOR_SEAL_SCHEMA
            ),
            "trace_adapter_sha256": self._trace_adapter_sha256,
            "bridge_artifact_sha256": self._bridge_artifact_sha256,
            "source_tied_table_sha256": self._source_tied_table_sha256,
            "blank_target_report_sha256": self._blank_report_sha256,
            "policy_sha256": self._policy_sha256,
            "parent_phase_tensor_rehash_required": True,
            "parent_tensor_storage_object_version_sealed": True,
        }

    def verify(self) -> Lfm2CpcfSealedEvidence:
        self._trace_adapter.verify()
        if (
            self._trace_adapter.evidence_sha256
            != self._trace_adapter_sha256
            or self._trace_adapter.bridge_artifact_sha256
            != self._bridge_artifact_sha256
        ):
            raise Lfm2CpcfCatalogError(
                "Tokenizer bridge trace-adapter identity changed."
            )
        identities = tuple(phase.identity for phase in self._phases)
        if len(set(identities)) != len(identities):
            raise Lfm2CpcfCatalogError("CPCF target phase identities repeat.")
        observed_ordinals = {identity[0] for identity in identities}
        expected_ordinals = {
            int(record.ordinal) for record in self._trace_adapter.records
        }
        if observed_ordinals != expected_ordinals:
            raise Lfm2CpcfCatalogError(
                "CPCF phases do not cover every tokenizer-adapter record."
            )
        for phase in self._phases:
            record = self._trace_adapter.records[phase.batch_index]
            if (
                phase.parent_evidence.cache_key != self._cache_key
                or phase.donor_input_sha256
                != record.donor_cache_input_sha256
                or phase.target_alignment_sha256
                != record.alignment_sha256
                or phase.record_id != record.record_id
                or phase.split != record.split
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF phase differs from its strict tokenizer record."
                )
        observed_phase_sha = tuple(
            _phase_metadata(phase)["phase_sha256"]
            for phase in self._phases
        )
        if observed_phase_sha != self._phase_sha256:
            raise Lfm2CpcfCatalogError("CPCF ordered phase seal changed.")
        if tuple(
            phase.parent_evidence.evidence_sha256
            for phase in self._phases
        ) != self._parent_evidence_sha256:
            raise Lfm2CpcfCatalogError(
                "CPCF parent evidence identity changed."
            )
        self._verify_parent_tensor_mutation_seals()
        if _tensor_sha256(
            self._source_tied_table,
            "source_tied_table",
        ) != self._source_tied_table_sha256:
            raise Lfm2CpcfCatalogError("CPCF source tied table changed.")
        blank = dict(self._blank_target_report)
        blank_sha = blank.pop("report_sha256", None)
        if (
            blank_sha != self._blank_report_sha256
            or canonical_sha256(blank) != self._blank_report_sha256
        ):
            raise Lfm2CpcfCatalogError("CPCF blank-target seal changed.")
        if canonical_sha256(self._policy.as_dict()) != self._policy_sha256:
            raise Lfm2CpcfCatalogError("CPCF model-fold policy changed.")
        if canonical_sha256(self._body()) != self._bundle_sha256:
            raise Lfm2CpcfCatalogError("CPCF evidence bundle seal changed.")
        return self

    def _verify_parent_tensor_mutation_seals(self) -> None:
        current = dict(_parent_tensor_bindings(self._phases))
        if set(current) != {
            seal.path for seal in self._parent_tensor_seals
        }:
            raise Lfm2CpcfCatalogError(
                "CPCF parent evidence tensor paths changed."
            )
        for seal in self._parent_tensor_seals:
            observed = _parent_tensor_mutation_seal(
                seal.path,
                current[seal.path],
                content_sha256=seal.content_sha256,
            )
            if observed != seal:
                raise Lfm2CpcfCatalogError(
                    "CPCF parent evidence tensor storage/object/version seal "
                    f"changed at {seal.path!r}."
                )

    def verify_parent_tensor_integrity(
        self,
    ) -> Lfm2CpcfSealedEvidence:
        """Rehash all parent evidence at the last pre-compile boundary."""

        self.verify()
        observed = _authenticate_parent_tensors(self._phases)
        observed_aggregate = canonical_sha256(
            [
                {
                    "path": path,
                    "content_sha256": observed[path],
                }
                for path in sorted(observed)
            ]
        )
        if (
            dict(observed) != dict(self._parent_tensor_content_sha256)
            or observed_aggregate
            != self._parent_tensor_content_aggregate_sha256
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF parent evidence failed authenticated tensor integrity "
                "revalidation."
            )
        # Catch ordinary in-place writes that race with the canonical scan.
        self._verify_parent_tensor_mutation_seals()
        return self


class Lfm2CpcfEvidenceProvider:
    """Provide sealed CPCF evidence without regenerating parent traces."""

    def __init__(self, evidence: Lfm2CpcfSealedEvidence) -> None:
        if not isinstance(evidence, Lfm2CpcfSealedEvidence):
            raise Lfm2CpcfCatalogError(
                "evidence must be Lfm2CpcfSealedEvidence."
            )
        self.evidence = evidence.verify()
        self.prepare_count = 0
        self.refresh_count = 0
        self.restore_count = 0

    @property
    def base_fingerprint(self) -> str:
        return self.evidence.cache_key

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> FoldEvidence:
        del context
        if action.method_id != DENSE_BUDGET_MODEL_METHOD:
            raise Lfm2CpcfCatalogError(
                "CPCF evidence received an unsupported fold method."
            )
        self.evidence.verify()
        self.prepare_count += 1
        return FoldEvidence(
            self.evidence.bundle_sha256,
            payload=self.evidence,
            channels=action.evidence_requirements,
            parent_forward_count=0,
            child_forward_count=0,
            root_fingerprint=self.evidence.cache_key,
        )

    def refresh(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        transaction_report: Mapping[str, Any],
    ) -> None:
        del action, context, transaction_report
        self.evidence.verify()
        self.refresh_count += 1

    def restore(
        self,
        context: FoldExecutionContext,
        state_fingerprint: str,
        action_ids: Sequence[str],
    ) -> None:
        del action_ids
        if context.child_state_fingerprint != _digest(
            state_fingerprint,
            "state_fingerprint",
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF evidence restore observed an unrestored child."
            )
        self.evidence.verify()
        self.restore_count += 1


@dataclass(frozen=True, slots=True)
class _CpcfModelSnapshot:
    parameter_bindings: tuple[tuple[str, Any, bool], ...]
    parameter_storage_identity: Mapping[str, tuple[Any, ...]]
    buffer_bindings: tuple[tuple[str, Any], ...]
    buffer_storage_identity: Mapping[str, tuple[Any, ...]]
    tensor_values: Mapping[int, Any]
    module_modes: tuple[tuple[Any, bool], ...]
    state_fingerprint: str
    runtime_fingerprint: str
    portable_topology_fingerprint: str


def _named_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(
            model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        )
    except TypeError:  # pragma: no cover - old Torch compatibility
        return tuple(model.named_parameters(recurse=True))


def _named_buffers(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(
            model.named_buffers(
                recurse=True,
                remove_duplicate=False,
            )
        )
    except TypeError:  # pragma: no cover - old Torch compatibility
        return tuple(model.named_buffers(recurse=True))


def _tensor_storage_identity(tensor: Any) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        int(getattr(storage, "_cdata", 0)),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        int(tensor.numel()),
        int(tensor.element_size()),
        tuple(int(size) for size in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _parameter_storage_identity(parameter: Any) -> tuple[Any, ...]:
    return _tensor_storage_identity(parameter)


def _portable_model_topology_fingerprint(model: Any) -> str:
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    object_ordinals: dict[int, int] = {}
    storage_ordinals: dict[tuple[Any, ...], int] = {}
    rows: list[dict[str, Any]] = []
    for kind, values in (("parameter", parameters), ("buffer", buffers)):
        for name, tensor in values:
            object_ordinal = object_ordinals.setdefault(
                id(tensor),
                len(object_ordinals),
            )
            storage = tensor.untyped_storage()
            storage_key = (
                str(tensor.device),
                int(getattr(storage, "_cdata", 0)),
            )
            storage_ordinal = storage_ordinals.setdefault(
                storage_key,
                len(storage_ordinals),
            )
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "object_alias_ordinal": object_ordinal,
                    "storage_alias_ordinal": storage_ordinal,
                    "storage_offset": int(tensor.storage_offset()),
                    "shape": [int(size) for size in tensor.shape],
                    "stride": [int(value) for value in tensor.stride()],
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "requires_grad": bool(tensor.requires_grad),
                }
            )
    module_rows = [
        {
            "type": (
                f"{type(module).__module__}.{type(module).__qualname__}"
            ),
            "training": bool(module.training),
        }
        for module in model.modules()
    ]
    return canonical_sha256(
        {
            "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
            "tensors": rows,
            "modules": module_rows,
        }
    )


@dataclass(frozen=True, slots=True)
class _CompileProtectedTensorSeal:
    path: str
    tensor: Any
    object_id: int
    storage_identity: tuple[Any, ...]
    version: int


def _compile_protected_tensor_bindings(
    *,
    phases: Sequence[Lfm2CpcfTargetPhase],
    source_tied_table: Any,
    parent_model: Any,
    live_child_model: Any,
    staging_child_model: Any,
) -> tuple[tuple[str, Any], ...]:
    bindings: list[tuple[str, Any]] = list(_parent_tensor_bindings(phases))
    for phase in phases:
        phase_prefix = (
            f"phase[{phase.batch_index}:{phase.phase}]"
        )
        bindings.extend(
            (
                (f"{phase_prefix}.input_ids", phase.input_ids),
                (f"{phase_prefix}.attention_mask", phase.attention_mask),
                (f"{phase_prefix}.position_ids", phase.position_ids),
            )
        )
    bindings.append(("source_tied_table", source_tied_table))
    bindings.extend(
        (f"parent.parameter.{name}", parameter)
        for name, parameter in _named_parameters(parent_model)
    )
    bindings.extend(
        (f"parent.buffer.{name}", buffer)
        for name, buffer in _named_buffers(parent_model)
    )
    # The compiler receives only the staging child.  Sealing the live blank as
    # well turns accidental closure/global access into a fail-closed event.
    bindings.extend(
        (f"live_child.parameter.{name}", parameter)
        for name, parameter in _named_parameters(live_child_model)
    )
    bindings.extend(
        (f"live_child.buffer.{name}", buffer)
        for name, buffer in _named_buffers(live_child_model)
    )
    bindings.extend(
        (f"staging_child.buffer.{name}", buffer)
        for name, buffer in _named_buffers(staging_child_model)
    )
    paths = tuple(path for path, _tensor in bindings)
    if len(set(paths)) != len(paths):
        raise Lfm2CpcfCatalogError(
            "CPCF compile write-guard paths are not unique."
        )
    return tuple(bindings)


def _iter_tensors(value: Any) -> tuple[Any, ...]:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Mapping):
        tensors: list[Any] = []
        for item in value.values():
            tensors.extend(_iter_tensors(item))
        return tuple(tensors)
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_iter_tensors(item))
        return tuple(tensors)
    return ()


class _CompileSealedStorageWriteGuard:
    """Torch-dispatch firewall over every read-only compiler storage."""

    def __init__(
        self,
        bindings: Sequence[tuple[str, Any]],
        *,
        writable_bindings: Sequence[tuple[str, Any]],
    ) -> None:
        torch = _torch()
        protected_seals: list[_CompileProtectedTensorSeal] = []
        writable_seals: list[_CompileProtectedTensorSeal] = []
        for items, destination in (
            (tuple(bindings), protected_seals),
            (tuple(writable_bindings), writable_seals),
        ):
            for path, tensor in items:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.layout != torch.strided
                    or tensor.device.type == "meta"
                ):
                    raise Lfm2CpcfCatalogError(
                        f"CPCF compile-protected tensor {path!r} is not "
                        "materialized dense storage."
                    )
                version = getattr(tensor, "_version", None)
                if isinstance(version, bool) or not isinstance(version, int):
                    raise Lfm2CpcfCatalogError(
                        f"CPCF compile-protected tensor {path!r} has no "
                        "mutation version."
                    )
                destination.append(
                    _CompileProtectedTensorSeal(
                        path=path,
                        tensor=tensor,
                        object_id=id(tensor),
                        storage_identity=_tensor_storage_identity(tensor),
                        version=version,
                    )
                )
        if len({seal.path for seal in protected_seals}) != len(
            protected_seals
        ) or len({seal.path for seal in writable_seals}) != len(
            writable_seals
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF compile guard tensor paths are not unique."
            )
        protected_storage_roots = {
            (seal.storage_identity[0], seal.storage_identity[-1])
            for seal in protected_seals
        }
        writable_storage_roots = {
            (seal.storage_identity[0], seal.storage_identity[-1])
            for seal in writable_seals
        }
        if (
            not writable_seals
            or protected_storage_roots.intersection(writable_storage_roots)
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF staging writable storage is empty or aliases sealed "
                "read-only compiler storage."
            )
        self._seals = tuple(protected_seals)
        self._writable_seals = tuple(writable_seals)
        self._violations: list[dict[str, Any]] = []
        self._writable_events: list[dict[str, Any]] = []
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
        except ImportError as exc:  # pragma: no cover - version boundary
            raise Lfm2CpcfCatalogError(
                "PyTorch TorchDispatchMode is required for CPCF compile "
                "write protection."
            ) from exc
        guard = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(
                self,
                func: Any,
                types: Any,
                args: tuple[Any, ...] = (),
                kwargs: Mapping[str, Any] | None = None,
            ) -> Any:
                del types
                resolved_kwargs = dict(kwargs or {})
                written = guard._written_tensors(
                    func,
                    args,
                    resolved_kwargs,
                )
                protected = guard._protected_paths(written)
                if protected:
                    operation = str(func)
                    guard._violations.append(
                        {
                            "operation": operation,
                            "protected_paths": sorted(protected),
                        }
                    )
                    raise Lfm2CpcfCatalogError(
                        "CPCF compiler attempted a write to sealed storage "
                        f"via {operation}: {sorted(protected)!r}."
                    )
                writable = guard._protected_paths(
                    written,
                    seals=guard._writable_seals,
                )
                result = func(*args, **resolved_kwargs)
                if writable:
                    guard._writable_events.append(
                        {
                            "operation": str(func),
                            "writable_paths": sorted(writable),
                        }
                    )
                return result

        self.mode = _Mode()

    @staticmethod
    def _written_tensors(
        func: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        schema = getattr(func, "_schema", None)
        arguments = getattr(schema, "arguments", ())
        positional_index = 0
        written: list[Any] = []
        for argument in arguments:
            value: Any
            if not bool(argument.kwarg_only) and positional_index < len(args):
                value = args[positional_index]
                positional_index += 1
            elif argument.name in kwargs:
                value = kwargs[argument.name]
            else:
                if not bool(argument.kwarg_only):
                    positional_index += 1
                continue
            alias_info = getattr(argument, "alias_info", None)
            if alias_info is not None and bool(alias_info.is_write):
                written.extend(_iter_tensors(value))
        return tuple(written)

    def _protected_paths(
        self,
        tensors: Sequence[Any],
        *,
        seals: Sequence[_CompileProtectedTensorSeal] | None = None,
    ) -> set[str]:
        protected: set[str] = set()
        for tensor in tensors:
            identity = _tensor_storage_identity(tensor)
            storage_cdata = identity[0]
            data_ptr = identity[1]
            nbytes = identity[2]
            device = identity[-1]
            for seal in self._seals if seals is None else seals:
                sealed_identity = seal.storage_identity
                sealed_cdata = sealed_identity[0]
                sealed_ptr = sealed_identity[1]
                sealed_nbytes = sealed_identity[2]
                sealed_device = sealed_identity[-1]
                same_storage = (
                    storage_cdata != 0
                    and storage_cdata == sealed_cdata
                    and device == sealed_device
                )
                overlapping_range = (
                    nbytes > 0
                    and sealed_nbytes > 0
                    and device == sealed_device
                    and data_ptr < sealed_ptr + sealed_nbytes
                    and sealed_ptr < data_ptr + nbytes
                )
                if same_storage or overlapping_range:
                    protected.add(seal.path)
        return protected

    def assert_clean(
        self,
        current_bindings: Sequence[tuple[str, Any]],
        current_writable_bindings: Sequence[tuple[str, Any]],
        *,
        require_complete_writes: bool,
    ) -> Mapping[str, Any]:
        if self._violations:
            first = self._violations[0]
            raise Lfm2CpcfCatalogError(
                "CPCF compiler attempted a sealed-storage write even if "
                "the compiler swallowed the dispatch error: "
                f"{first['operation']} -> {first['protected_paths']!r}."
            )
        current_items = tuple(current_bindings)
        current = dict(current_items)
        writable_items = tuple(current_writable_bindings)
        current_writable = dict(writable_items)
        if len(current) != len(current_items) or set(current) != {
            seal.path for seal in self._seals
        }:
            raise Lfm2CpcfCatalogError(
                "CPCF compile-protected tensor bindings changed."
            )
        if (
            len(current_writable) != len(writable_items)
            or set(current_writable)
            != {seal.path for seal in self._writable_seals}
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF staging writable tensor bindings changed."
            )
        for seal in self._seals:
            tensor = current[seal.path]
            version = getattr(tensor, "_version", None)
            if (
                tensor is not seal.tensor
                or id(tensor) != seal.object_id
                or _tensor_storage_identity(tensor)
                != seal.storage_identity
                or version != seal.version
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF compile-protected tensor object, storage, or "
                    f"version changed at {seal.path!r}."
                )
        for seal in self._writable_seals:
            tensor = current_writable[seal.path]
            if (
                tensor is not seal.tensor
                or id(tensor) != seal.object_id
                or _tensor_storage_identity(tensor)
                != seal.storage_identity
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF staging writable tensor object or storage changed "
                    f"at {seal.path!r}."
                )
        written_paths = {
            path
            for event in self._writable_events
            for path in event["writable_paths"]
        }
        missing_storage_paths = []
        writable_groups: dict[tuple[Any, ...], list[str]] = {}
        for seal in self._writable_seals:
            storage_root = (
                seal.storage_identity[0],
                seal.storage_identity[-1],
            )
            writable_groups.setdefault(storage_root, []).append(seal.path)
        for paths in writable_groups.values():
            if not set(paths).intersection(written_paths):
                missing_storage_paths.append(sorted(paths))
        if missing_storage_paths and require_complete_writes:
            raise Lfm2CpcfCatalogError(
                "CPCF compiler did not write every unique staging target "
                f"parameter storage: {missing_storage_paths!r}."
            )
        ordered_events = sorted(
            (
                {
                    "operation": str(event["operation"]),
                    "writable_paths": list(event["writable_paths"]),
                }
                for event in self._writable_events
            ),
            key=lambda event: (
                event["operation"],
                tuple(event["writable_paths"]),
            ),
        )
        body = {
            "schema": LFM2_CPCF_COMPILE_WRITE_GUARD_SCHEMA,
            "protected_binding_count": len(self._seals),
            "protected_unique_storage_count": len(
                {seal.storage_identity[:3] for seal in self._seals}
            ),
            "attempted_write_count": 0,
            "all_protected_bindings_unchanged": True,
            "compiler_write_attempts_rejected": True,
            "writable_parameter_path_count": len(
                self._writable_seals
            ),
            "writable_unique_storage_count": len(writable_groups),
            "authenticated_writable_parameter_paths": sorted(
                written_paths
            ),
            "authenticated_writable_operation_count": len(
                self._writable_events
            ),
            "authenticated_writable_event_aggregate_sha256": (
                canonical_sha256(ordered_events)
            ),
            "all_unique_writable_parameter_storages_written": (
                not missing_storage_paths
            ),
            "writable_parameter_objects_and_storages_preserved": True,
        }
        return MappingProxyType(
            {**body, "report_sha256": canonical_sha256(body)}
        )


def _capture_model(model: Any) -> _CpcfModelSnapshot:
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    if any(parameter.grad is not None for _name, parameter in parameters):
        raise Lfm2CpcfCatalogError(
            "CPCF requires a zero-gradient target before folding."
        )
    values: dict[int, Any] = {}
    for _name, tensor in (*parameters, *buffers):
        values.setdefault(
            id(tensor),
            tensor.detach().to(device="cpu").contiguous().clone(),
        )
    return _CpcfModelSnapshot(
        parameter_bindings=tuple(
            (name, parameter, bool(parameter.requires_grad))
            for name, parameter in parameters
        ),
        parameter_storage_identity=MappingProxyType(
            {
                name: _parameter_storage_identity(parameter)
                for name, parameter in parameters
            }
        ),
        buffer_bindings=buffers,
        buffer_storage_identity=MappingProxyType(
            {
                name: _tensor_storage_identity(buffer)
                for name, buffer in buffers
            }
        ),
        tensor_values=MappingProxyType(values),
        module_modes=tuple(
            (module, bool(module.training)) for module in model.modules()
        ),
        state_fingerprint=fingerprint_model_state(model),
        runtime_fingerprint=fingerprint_model_runtime_state(model),
        portable_topology_fingerprint=(
            _portable_model_topology_fingerprint(model)
        ),
    )


def _restore_model(model: Any, snapshot: _CpcfModelSnapshot) -> None:
    torch = _torch()
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    if tuple((name, id(value)) for name, value in parameters) != tuple(
        (name, id(value))
        for name, value, _requires_grad in snapshot.parameter_bindings
    ) or tuple((name, id(value)) for name, value in buffers) != tuple(
        (name, id(value)) for name, value in snapshot.buffer_bindings
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF target tensor topology changed before rollback."
        )
    if {
        name: _parameter_storage_identity(parameter)
        for name, parameter in parameters
    } != dict(snapshot.parameter_storage_identity):
        raise Lfm2CpcfCatalogError(
            "CPCF target parameter storage changed before rollback."
        )
    if {
        name: _tensor_storage_identity(buffer)
        for name, buffer in buffers
    } != dict(snapshot.buffer_storage_identity):
        raise Lfm2CpcfCatalogError(
            "CPCF target buffer storage changed before rollback."
        )
    with torch.no_grad():
        restored: set[int] = set()
        for _name, tensor in (*parameters, *buffers):
            if id(tensor) in restored:
                continue
            source = snapshot.tensor_values[id(tensor)]
            if (
                tuple(tensor.shape) != tuple(source.shape)
                or tensor.dtype != source.dtype
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF target tensor shape or dtype changed before rollback."
                )
            tensor.copy_(source.to(device=tensor.device, dtype=tensor.dtype))
            restored.add(id(tensor))
        for _name, parameter, requires_grad in snapshot.parameter_bindings:
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
    for module, training in snapshot.module_modes:
        module.training = training
    if (
        fingerprint_model_state(model) != snapshot.state_fingerprint
        or fingerprint_model_runtime_state(model)
        != snapshot.runtime_fingerprint
        or _portable_model_topology_fingerprint(model)
        != snapshot.portable_topology_fingerprint
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF rollback did not restore the exact target."
        )


def _assert_model_matches_snapshot_topology(
    model: Any,
    snapshot: _CpcfModelSnapshot,
    *,
    name: str,
) -> None:
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    if tuple((path, id(value)) for path, value in parameters) != tuple(
        (path, id(value))
        for path, value, _requires_grad in snapshot.parameter_bindings
    ):
        raise Lfm2CpcfCatalogError(
            f"CPCF {name} Parameter objects or paths changed."
        )
    if tuple((path, id(value)) for path, value in buffers) != tuple(
        (path, id(value)) for path, value in snapshot.buffer_bindings
    ):
        raise Lfm2CpcfCatalogError(
            f"CPCF {name} buffer objects or paths changed."
        )
    if {
        path: _parameter_storage_identity(parameter)
        for path, parameter in parameters
    } != dict(snapshot.parameter_storage_identity):
        raise Lfm2CpcfCatalogError(
            f"CPCF {name} Parameter storage topology changed."
        )
    if {
        path: _tensor_storage_identity(buffer)
        for path, buffer in buffers
    } != dict(snapshot.buffer_storage_identity):
        raise Lfm2CpcfCatalogError(
            f"CPCF {name} buffer storage topology changed."
        )
    if (
        fingerprint_model_runtime_state(model) != snapshot.runtime_fingerprint
        or _portable_model_topology_fingerprint(model)
        != snapshot.portable_topology_fingerprint
    ):
        raise Lfm2CpcfCatalogError(
            f"CPCF {name} runtime topology changed."
        )


def _model_storage_roots(model: Any) -> set[tuple[str, int]]:
    roots: set[tuple[str, int]] = set()
    for _name, tensor in (
        *_named_parameters(model),
        *_named_buffers(model),
    ):
        storage = tensor.untyped_storage()
        roots.add(
            (
                str(tensor.device),
                int(getattr(storage, "_cdata", 0)),
            )
        )
    return roots


def _make_disposable_staging_blank(
    live_model: Any,
    live_snapshot: _CpcfModelSnapshot,
) -> tuple[Any, _CpcfModelSnapshot]:
    try:
        staging_model = deepcopy(live_model)
    except BaseException as exc:
        raise Lfm2CpcfCatalogError(
            "CPCF could not create a disposable staging blank."
        ) from exc
    staging_snapshot = _capture_model(staging_model)
    if (
        staging_snapshot.state_fingerprint
        != live_snapshot.state_fingerprint
        or staging_snapshot.portable_topology_fingerprint
        != live_snapshot.portable_topology_fingerprint
        or _model_storage_roots(staging_model).intersection(
            _model_storage_roots(live_model)
        )
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF staging blank is not an exact storage-disjoint copy of "
            "the authenticated live blank."
        )
    return staging_model, staging_snapshot


def _named_install_tensors(
    model: Any,
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (f"parameter.{name}", parameter)
        for name, parameter in _named_parameters(model)
    ) + tuple(
        (f"buffer.{name}", buffer)
        for name, buffer in _named_buffers(model)
    )


def _install_staged_candidate(
    *,
    live_model: Any,
    live_snapshot: _CpcfModelSnapshot,
    staging_model: Any,
    staging_snapshot: _CpcfModelSnapshot,
    staging_candidate_state_fingerprint: str,
    staging_parameter_receipt: Mapping[str, Any],
    authenticated_compiler_write_paths: Sequence[str],
    expected_unique_parameters: int,
) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    torch = _torch()
    candidate = _digest(
        staging_candidate_state_fingerprint,
        "staging_candidate_state_fingerprint",
    )
    _assert_model_matches_snapshot_topology(
        live_model,
        live_snapshot,
        name="live blank before staged installation",
    )
    _assert_model_matches_snapshot_topology(
        staging_model,
        staging_snapshot,
        name="staging candidate",
    )
    if (
        fingerprint_model_state(live_model)
        != live_snapshot.state_fingerprint
        or fingerprint_model_state(staging_model) != candidate
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF live blank or staging candidate changed before "
            "authenticated installation."
        )
    live_values = dict(_named_install_tensors(live_model))
    staging_values = dict(_named_install_tensors(staging_model))
    if (
        len(live_values) != len(_named_install_tensors(live_model))
        or len(staging_values) != len(_named_install_tensors(staging_model))
        or set(live_values) != set(staging_values)
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF staging/live install tensor paths differ."
        )
    with torch.no_grad():
        copied_objects: set[int] = set()
        for path in sorted(live_values):
            destination = live_values[path]
            source = staging_values[path]
            if (
                tuple(destination.shape) != tuple(source.shape)
                or destination.dtype != source.dtype
                or destination.layout != source.layout
            ):
                raise Lfm2CpcfCatalogError(
                    f"CPCF staged install tensor {path!r} differs in shape, "
                    "dtype, or layout."
                )
            if id(destination) in copied_objects:
                continue
            destination.copy_(
                source.detach().to(
                    device=destination.device,
                    dtype=destination.dtype,
                )
            )
            copied_objects.add(id(destination))
    installed_candidate = fingerprint_model_state(live_model)
    if installed_candidate != candidate:
        raise Lfm2CpcfCatalogError(
            "CPCF controlled staged copy did not reproduce the exact "
            "authenticated candidate state."
        )
    _assert_model_matches_snapshot_topology(
        live_model,
        live_snapshot,
        name="live candidate after staged installation",
    )
    live_parameter_receipt = _parameter_path_write_receipt(
        live_model,
        live_snapshot,
        expected_unique_parameters=expected_unique_parameters,
        candidate_state_fingerprint=installed_candidate,
        authenticated_compiler_write_paths=(
            authenticated_compiler_write_paths
        ),
    )
    if (
        staging_parameter_receipt.get(
            "resulting_parameter_path_aggregate_sha256"
        )
        != live_parameter_receipt.get(
            "resulting_parameter_path_aggregate_sha256"
        )
        or staging_parameter_receipt.get("target_unique_parameters")
        != live_parameter_receipt.get("target_unique_parameters")
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF live installation differs from its authenticated staging "
            "parameter receipt."
        )
    body = {
        "schema": LFM2_CPCF_STAGED_INSTALL_RECEIPT_SCHEMA,
        "live_blank_state_fingerprint": live_snapshot.state_fingerprint,
        "staging_blank_state_fingerprint": staging_snapshot.state_fingerprint,
        "staging_candidate_state_fingerprint": candidate,
        "installed_candidate_state_fingerprint": installed_candidate,
        "staging_parameter_receipt_sha256": staging_parameter_receipt[
            "receipt_sha256"
        ],
        "live_parameter_receipt_sha256": live_parameter_receipt[
            "receipt_sha256"
        ],
        "copied_tensor_object_count": len(copied_objects),
        "live_parameter_and_buffer_objects_preserved": True,
        "live_parameter_and_buffer_storages_preserved": True,
        "staging_and_live_candidate_states_equal": True,
        "controlled_copy_only": True,
    }
    install_receipt = MappingProxyType(
        {**body, "receipt_sha256": canonical_sha256(body)}
    )
    return installed_candidate, live_parameter_receipt, install_receipt


def _parameter_path_write_receipt(
    model: Any,
    snapshot: _CpcfModelSnapshot,
    *,
    expected_unique_parameters: int,
    candidate_state_fingerprint: str,
    authenticated_compiler_write_paths: Sequence[str],
) -> Mapping[str, Any]:
    """Prove every blank storage was authentically written by the compiler."""

    torch = _torch()
    candidate = _digest(
        candidate_state_fingerprint,
        "candidate_state_fingerprint",
    )
    if fingerprint_model_state(model) != candidate:
        raise Lfm2CpcfCatalogError(
            "CPCF candidate fingerprint changed during path receipt."
        )
    current = _named_parameters(model)
    before_bindings = snapshot.parameter_bindings
    if tuple((name, id(value)) for name, value in current) != tuple(
        (name, id(value))
        for name, value, _requires_grad in before_bindings
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF target parameter paths or objects changed during "
            "compilation."
        )
    current_storage = {
        name: _parameter_storage_identity(parameter)
        for name, parameter in current
    }
    if current_storage != dict(snapshot.parameter_storage_identity):
        raise Lfm2CpcfCatalogError(
            "CPCF target parameter storage topology changed during "
            "compilation."
        )
    before_by_path = {
        name: snapshot.tensor_values[id(parameter)]
        for name, parameter, _requires_grad in before_bindings
    }
    current_by_path = dict(current)
    if len(current_by_path) != len(current):
        raise Lfm2CpcfCatalogError(
            "CPCF target exposes duplicate parameter path names."
        )
    authenticated_paths = {
        _text(path, "authenticated_compiler_write_path")
        for path in authenticated_compiler_write_paths
    }
    if authenticated_paths != set(current_by_path):
        raise Lfm2CpcfCatalogError(
            "CPCF authenticated compiler write paths do not cover the exact "
            "target parameter topology."
        )
    storage_groups: dict[tuple[Any, ...], list[str]] = {}
    for path, identity in snapshot.parameter_storage_identity.items():
        storage_groups.setdefault(identity, []).append(path)
    group_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    unique_numel = 0
    changed_storage_count = 0
    zero_valued_folded_storage_count = 0
    for identity, raw_paths in sorted(
        storage_groups.items(),
        key=lambda item: sorted(item[1]),
    ):
        paths = sorted(raw_paths)
        representative = current_by_path[paths[0]]
        unique_numel += int(representative.numel())
        before_hashes = {
            _tensor_sha256(
                before_by_path[path],
                f"compiler blank parameter {path}",
            )
            for path in paths
        }
        after_hashes = {
            _tensor_sha256(
                current_by_path[path],
                f"compiled parameter {path}",
            )
            for path in paths
        }
        if len(before_hashes) != 1 or len(after_hashes) != 1:
            raise Lfm2CpcfCatalogError(
                "CPCF tied parameter aliases disagree on before/after bytes."
            )
        before_sha = before_hashes.pop()
        after_sha = after_hashes.pop()
        before_values = before_by_path[paths[0]]
        group_written = bool(set(paths).intersection(authenticated_paths))
        if (
            int(torch.count_nonzero(before_values).item()) != 0
            or not group_written
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF did not authentically write every unique target "
                "parameter storage from the compiler-zero blank."
            )
        changed = before_sha != after_sha
        changed_storage_count += int(changed)
        zero_valued = (
            int(torch.count_nonzero(representative.detach()).item()) == 0
        )
        zero_valued_folded_storage_count += int(zero_valued)
        alias_group_sha = canonical_sha256(
            {
                "paths": paths,
                "shape": [int(size) for size in representative.shape],
                "dtype": str(representative.dtype).removeprefix("torch."),
                "numel": int(representative.numel()),
            }
        )
        group_rows.append(
            {
                "alias_group_sha256": alias_group_sha,
                "paths": paths,
                "tied_alias": len(paths) > 1,
                "shape": [int(size) for size in representative.shape],
                "dtype": str(representative.dtype).removeprefix("torch."),
                "numel": int(representative.numel()),
                "compiler_zero_sha256": before_sha,
                "result_sha256": after_sha,
                "changed_from_compiler_zero": changed,
                "authenticated_compiler_write": True,
                "zero_valued_folded_result": zero_valued,
                "overwritten_from_compiler_blank": True,
            }
        )
        path_rows.extend(
            {
                "parameter_path": path,
                "alias_group_sha256": alias_group_sha,
                "compiler_zero_sha256": before_sha,
                "result_sha256": after_sha,
                "changed_from_compiler_zero": changed,
                "authenticated_compiler_write": True,
                "overwritten_from_compiler_blank": True,
            }
            for path in paths
        )
    if unique_numel != expected_unique_parameters:
        raise Lfm2CpcfCatalogError(
            "CPCF independent parameter-path inventory differs from the "
            "exact target budget."
        )
    path_rows.sort(key=lambda row: str(row["parameter_path"]))
    group_rows.sort(key=lambda row: str(row["alias_group_sha256"]))
    result_aggregate = canonical_sha256(
        [
            {
                "parameter_path": row["parameter_path"],
                "alias_group_sha256": row["alias_group_sha256"],
                "result_sha256": row["result_sha256"],
            }
            for row in path_rows
        ]
    )
    body = {
        "schema": LFM2_CPCF_PARAMETER_PATH_WRITE_RECEIPT_SCHEMA,
        "compiler_blank_state_fingerprint": snapshot.state_fingerprint,
        "candidate_state_fingerprint": candidate,
        "named_parameter_path_count": len(path_rows),
        "unique_parameter_storage_count": len(group_rows),
        "target_unique_parameters": unique_numel,
        "parameter_paths": path_rows,
        "alias_groups": group_rows,
        "resulting_parameter_path_aggregate_sha256": result_aggregate,
        "all_named_parameter_paths_accounted": True,
        "authenticated_compiler_write_paths": sorted(authenticated_paths),
        "changed_unique_parameter_storage_count": changed_storage_count,
        "zero_valued_folded_storage_count": (
            zero_valued_folded_storage_count
        ),
        "all_unique_parameter_storages_changed_from_compiler_zero": (
            changed_storage_count == len(group_rows)
        ),
        "all_unique_parameter_storages_authentically_written": True,
        "resulting_path_hashes_verified": True,
        "tied_aliases_explicitly_accounted": True,
    }
    return MappingProxyType(
        {**body, "receipt_sha256": canonical_sha256(body)}
    )


def _verify_source_tied_table(
    parent_model: Any,
    table: Any,
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
) -> None:
    torch = _torch()
    get_embedding = getattr(parent_model, "get_input_embeddings", None)
    embedding = get_embedding() if callable(get_embedding) else None
    weight = getattr(embedding, "weight", None)
    mapping = tuple(int(value) for value in trace_adapter.mapping)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or tuple(table.shape)
        != (len(mapping), int(weight.shape[1]))
        or any(value < 0 or value >= int(weight.shape[0]) for value in mapping)
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF tied interface table shape or bridge rows differ from the "
            "live parent embedding."
        )
    with torch.no_grad():
        for start in range(0, len(mapping), 1024):
            stop = min(len(mapping), start + 1024)
            indices = torch.tensor(
                mapping[start:stop],
                device=weight.device,
                dtype=torch.long,
            )
            expected = weight.detach().index_select(0, indices)
            observed = table[start:stop].detach().to(
                device=weight.device,
                dtype=weight.dtype,
            )
            if not torch.equal(expected, observed):
                raise Lfm2CpcfCatalogError(
                    "CPCF source tied table is not the bridge-selected live "
                    "parent embedding."
                )


def _authenticated_compiler_write_paths(
    report: Mapping[str, Any],
) -> tuple[str, ...]:
    guard = report.get("compile_sealed_storage_write_guard")
    paths = (
        None
        if not isinstance(guard, Mapping)
        else guard.get("authenticated_writable_parameter_paths")
    )
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not path for path in paths)
        or paths != sorted(set(paths))
        or guard.get("all_unique_writable_parameter_storages_written")
        is not True
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF report lacks its authenticated staging parameter write "
            "set."
        )
    return tuple(paths)


def _authenticate_compiled_model_fold_report(
    report: Mapping[str, Any],
    *,
    expected_result_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Independently authenticate the compiler-owned immutable report."""

    if not isinstance(report, Mapping):
        raise Lfm2CpcfCatalogError(
            "CPCF compiler returned no model-fold report mapping."
        )
    compiler_report = deepcopy(dict(report))
    result_sha256 = _digest(
        compiler_report.pop("result_sha256", None),
        "compiled_model_fold.result_sha256",
    )
    if (
        expected_result_sha256 is not None
        and _digest(
            expected_result_sha256,
            "compiled_model_fold expected result_sha256",
        )
        != result_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF compiler result object and serialized report seals differ."
        )
    if canonical_sha256(compiler_report) != result_sha256:
        raise Lfm2CpcfCatalogError(
            "CPCF compiled model-fold result_sha256 seal is invalid."
        )
    if (
        compiler_report.get("schema") != LFM2_CPCF_MODEL_FOLD_SCHEMA
        or compiler_report.get("all_layers_compiled_in_order") is not True
        or compiler_report.get("actual_folded_upstream_replay_used")
        is not True
        or compiler_report.get("source_state_preserved") is not True
        or compiler_report.get("target_parameter_count_preserved") is not True
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF compiled report lacks complete model-fold invariants."
        )
    return MappingProxyType(
        {**compiler_report, "result_sha256": result_sha256}
    )


def _seal_catalog_compile_report(
    body: Mapping[str, Any],
    *,
    compiled_model_fold_result_sha256: str,
) -> Mapping[str, Any]:
    """Seal adapter/install metadata without modifying compiler provenance."""

    normalized = deepcopy(dict(body))
    if "report_sha256" in normalized:
        raise Lfm2CpcfCatalogError(
            "CPCF catalog compile-report body must be unsealed."
        )
    normalized["schema"] = LFM2_CPCF_CATALOG_COMPILE_REPORT_SCHEMA
    normalized["compiled_model_fold_result_sha256"] = _digest(
        compiled_model_fold_result_sha256,
        "compiled_model_fold_result_sha256",
    )
    return MappingProxyType(
        {**normalized, "report_sha256": canonical_sha256(normalized)}
    )


def _authenticate_catalog_compile_report(
    report: Mapping[str, Any],
    *,
    compiled_model_fold_result_sha256: str,
    expected_report_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Authenticate the separately sealed catalog/install wrapper."""

    if not isinstance(report, Mapping):
        raise Lfm2CpcfCatalogError(
            "CPCF catalog compile report must be a mapping."
        )
    catalog_report = deepcopy(dict(report))
    report_sha256 = _digest(
        catalog_report.pop("report_sha256", None),
        "catalog_compile_report.report_sha256",
    )
    compiler_sha256 = _digest(
        catalog_report.get("compiled_model_fold_result_sha256"),
        "catalog_compile_report.compiled_model_fold_result_sha256",
    )
    if (
        expected_report_sha256 is not None
        and report_sha256
        != _digest(
            expected_report_sha256,
            "expected catalog_compile_report.report_sha256",
        )
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF catalog compile report no longer matches its bound seal."
        )
    if (
        catalog_report.get("schema")
        != LFM2_CPCF_CATALOG_COMPILE_REPORT_SCHEMA
        or compiler_sha256
        != _digest(
            compiled_model_fold_result_sha256,
            "compiled_model_fold_result_sha256",
        )
        or canonical_sha256(catalog_report) != report_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF catalog compile-report seal or compiler binding is invalid."
        )
    return MappingProxyType(
        {**catalog_report, "report_sha256": report_sha256}
    )


class Lfm2CpcfModelTransaction:
    """Exact transaction around one complete CPCF child installation."""

    method_id = DENSE_BUDGET_MODEL_METHOD

    def __init__(
        self,
        context: FoldExecutionContext,
        snapshot: _CpcfModelSnapshot,
    ) -> None:
        self.context = context
        self._snapshot = snapshot
        self.incumbent_state_fingerprint = snapshot.state_fingerprint
        self.candidate_state_fingerprint = snapshot.state_fingerprint
        self._compiled_model_fold_report_canonical = _canonical_json(
            {}
        )
        self._catalog_compile_report_canonical = _canonical_json(
            {
                "schema": LFM2_CPCF_CATALOG_SCHEMA,
                "prepared": False,
            }
        )
        self._catalog_compile_report_sha256: str | None = None
        self.active = True

    @property
    def compiled_model_fold_report(self) -> Mapping[str, Any]:
        """Return a detached view of the immutable compiler report bytes."""

        return MappingProxyType(
            json.loads(
                self._compiled_model_fold_report_canonical.decode("ascii")
            )
        )

    @property
    def report(self) -> Mapping[str, Any]:
        """Return a detached view of immutable catalog/install report bytes."""

        return MappingProxyType(
            json.loads(
                self._catalog_compile_report_canonical.decode("ascii")
            )
        )

    def bind_candidate(
        self,
        *,
        compiled_model_fold_report: Mapping[str, Any],
        catalog_compile_report: Mapping[str, Any],
        candidate_state_fingerprint: str,
    ) -> None:
        if not self.active:
            raise Lfm2CpcfCatalogError(
                "Cannot bind an inactive CPCF transaction."
            )
        candidate = _digest(
            candidate_state_fingerprint,
            "candidate_state_fingerprint",
        )
        if candidate == self.incumbent_state_fingerprint:
            raise Lfm2CpcfCatalogError(
                "CPCF candidate is identical to the compiler-zero target."
            )
        if (
            fingerprint_model_runtime_state(self.context.child_model)
            != self._snapshot.runtime_fingerprint
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF changed target runtime topology."
            )
        compiler_report = _authenticate_compiled_model_fold_report(
            compiled_model_fold_report
        )
        catalog_report = _authenticate_catalog_compile_report(
            catalog_compile_report,
            compiled_model_fold_result_sha256=compiler_report[
                "result_sha256"
            ],
        )
        receipt = catalog_report.get("parameter_path_write_receipt")
        authenticated_paths = _authenticated_compiler_write_paths(
            catalog_report
        )
        observed_receipt = _parameter_path_write_receipt(
            self.context.child_model,
            self._snapshot,
            expected_unique_parameters=(
                self.context.child_graph.unique_parameter_count
            ),
            candidate_state_fingerprint=candidate,
            authenticated_compiler_write_paths=authenticated_paths,
        )
        if (
            not isinstance(receipt, Mapping)
            or dict(receipt) != dict(observed_receipt)
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF report lacks the exact independent parameter-path "
                "write receipt."
            )
        self._compiled_model_fold_report_canonical = _canonical_json(
            dict(compiler_report)
        )
        self._catalog_compile_report_canonical = _canonical_json(
            dict(catalog_report)
        )
        self._catalog_compile_report_sha256 = catalog_report[
            "report_sha256"
        ]
        self.candidate_state_fingerprint = candidate

    def verify_parameter_path_write_receipt(self) -> Mapping[str, Any]:
        receipt = self.report.get("parameter_path_write_receipt")
        authenticated_paths = _authenticated_compiler_write_paths(
            self.report
        )
        observed = _parameter_path_write_receipt(
            self.context.child_model,
            self._snapshot,
            expected_unique_parameters=(
                self.context.child_graph.unique_parameter_count
            ),
            candidate_state_fingerprint=self.candidate_state_fingerprint,
            authenticated_compiler_write_paths=authenticated_paths,
        )
        if (
            not isinstance(receipt, Mapping)
            or dict(receipt) != dict(observed)
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF live target no longer matches its parameter-path "
                "write receipt."
            )
        return observed

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise Lfm2CpcfCatalogError("CPCF transaction is inactive.")
        current = self.context.child_state_fingerprint
        if (
            current != self.candidate_state_fingerprint
            or fingerprint_model_runtime_state(self.context.child_model)
            != self._snapshot.runtime_fingerprint
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF candidate changed before commit."
            )
        self.verify_parameter_path_write_receipt()
        compiler_report = _authenticate_compiled_model_fold_report(
            self.compiled_model_fold_report
        )
        catalog_report = _authenticate_catalog_compile_report(
            self.report,
            compiled_model_fold_result_sha256=compiler_report[
                "result_sha256"
            ],
            expected_report_sha256=self._catalog_compile_report_sha256,
        )
        if (
            catalog_report.get("complete_target_parameter_overwrite")
            is not True
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF catalog compile report lacks complete installation "
                "invariants."
            )
        self.active = False
        return {
            "schema": LFM2_CPCF_CATALOG_SCHEMA,
            "compiled_model_fold": deepcopy(dict(compiler_report)),
            "catalog_compile_report": deepcopy(dict(catalog_report)),
            "action": "commit",
            "accepted": True,
            "exact": True,
            "candidate_state_fingerprint": (
                self.candidate_state_fingerprint
            ),
            "final_state_fingerprint": current,
            "parameter_topology_preserved": True,
            "zero_gradient": True,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise Lfm2CpcfCatalogError("CPCF transaction is inactive.")
        _restore_model(self.context.child_model, self._snapshot)
        final = self.context.child_state_fingerprint
        self.active = False
        return {
            "schema": LFM2_CPCF_CATALOG_SCHEMA,
            "compiled_model_fold": deepcopy(
                dict(self.compiled_model_fold_report)
            ),
            "catalog_compile_report": deepcopy(dict(self.report)),
            "action": "rollback",
            "accepted": False,
            "exact": True,
            "rollback_exact": True,
            "candidate_state_fingerprint": (
                self.candidate_state_fingerprint
            ),
            "final_state_fingerprint": final,
            "parameter_topology_preserved": True,
            "zero_gradient": True,
        }


class Lfm2CpcfModelOperator:
    """One full-source/full-target CPCF operator selected by the catalog."""

    method_id = DENSE_BUDGET_MODEL_METHOD

    def __init__(
        self,
        selected_target: Lfm2DenseBudgetedSelectedTarget,
        evidence: Lfm2CpcfSealedEvidence,
    ) -> None:
        if not isinstance(
            selected_target,
            Lfm2DenseBudgetedSelectedTarget,
        ):
            raise Lfm2CpcfCatalogError(
                "selected_target must be Lfm2DenseBudgetedSelectedTarget."
            )
        if not isinstance(evidence, Lfm2CpcfSealedEvidence):
            raise Lfm2CpcfCatalogError(
                "evidence must be Lfm2CpcfSealedEvidence."
            )
        self.selected_target = selected_target
        self.evidence = evidence.verify()
        self.resource_fingerprint = evidence.bundle_sha256

    def _validate_action(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> tuple[str, Any, ModelCircuitGraph]:
        selected = self.selected_target
        if (
            action.method_id != self.method_id
            or len(action.sources) != 1
            or action.compression_delta <= 0
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF requires one positive-compaction whole-model action."
            )
        source = action.sources[0]
        if source.parent_name != selected.source_parent:
            raise Lfm2CpcfCatalogError(
                "CPCF action source differs from target selection."
            )
        parent_graph = context.parent_graphs[source.parent_name]
        if (
            parent_graph.fingerprint != selected.source_graph_fingerprint
            or parent_graph.tokenizer_fingerprint
            != selected.parent_tokenizer_semantics_sha256
            or set(source.node_ids)
            != {node.node_id for node in parent_graph.nodes}
            or set(action.target_node_ids)
            != {node.node_id for node in context.child_graph.nodes}
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF action must own the complete authenticated parent and "
                "target circuits."
            )
        expected_paths = {
            name for name, _parameter in _named_parameters(context.child_model)
        }
        if set(action.target_parameter_paths) != expected_paths:
            raise Lfm2CpcfCatalogError(
                "CPCF action does not own every target parameter path."
            )
        configuration = dict(action.configuration)
        expected_configuration = {
            "catalog_id": selected.catalog_id,
            "source_depth": len(parent_graph.layer_indices),
            "target_depth": selected.target_depth,
            "source_spans": encode_lfm2_source_spans(
                selected.ordered_source_spans
            ),
            "target_layer_types": encode_lfm2_layer_types(
                selected.layer_types
            ),
            "target_hidden_size": selected.hidden_size,
            "target_intermediate_size": selected.intermediate_size,
            "target_query_heads": selected.query_heads,
            "target_key_value_heads": selected.key_value_heads,
            "target_head_dim": selected.head_dim,
            "tokenizer_bridge_required": True,
            "tokenizer_bridge_artifact_sha256": (
                selected.tokenizer_bridge_artifact_sha256
            ),
            "tokenizer_transport_profile_sha256": (
                selected.tokenizer_transport_profile_sha256
            ),
            "parent_tokenizer_semantics_sha256": (
                selected.parent_tokenizer_semantics_sha256
            ),
            "child_tokenizer_semantics_sha256": (
                selected.child_tokenizer_semantics_sha256
            ),
        }
        if any(
            configuration.get(name) != value
            for name, value in expected_configuration.items()
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF action topology or tokenizer bridge differs from the "
                "selected target."
            )
        source_depth = len(parent_graph.layer_indices)
        if (
            selected.target_depth != source_depth
            or selected.ordered_source_spans
            != tuple((index,) for index in range(source_depth))
        ):
            raise Lfm2CpcfCatalogError(
                "Current CPCF model compilation is width/GQA/FFN compaction "
                "at equal depth; depth composition remains fail-closed."
            )
        parent_model = context.parent_models[source.parent_name]
        if (
            fingerprint_model_state(parent_model)
            != selected.source_state_fingerprint
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF source checkpoint changed after target selection."
            )
        validate_lfm2_dense_budgeted_child(
            selected,
            context.child_model,
            context.child_graph,
        )
        return source.parent_name, parent_model, parent_graph

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> FoldTransaction:
        source_name, parent_model, _parent_graph = self._validate_action(
            action,
            context,
        )
        if (
            not isinstance(evidence.payload, Lfm2CpcfSealedEvidence)
            or evidence.payload is not self.evidence
            or evidence.fingerprint != self.evidence.bundle_sha256
            or evidence.root_fingerprint != self.evidence.cache_key
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF fold evidence lost its exact catalog bundle binding."
            )
        sealed = self.evidence.verify_parent_tensor_integrity()
        if (
            context.child_state_fingerprint
            != sealed.blank_target_report[
                "target_blank_state_fingerprint"
            ]
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF target no longer matches its compiler blank receipt."
            )
        _verify_source_tied_table(
            parent_model,
            sealed.source_tied_table,
            sealed.trace_adapter,
        )
        parent_before = fingerprint_model_state(parent_model)
        parent_runtime_before = fingerprint_model_runtime_state(parent_model)
        parent_module_modes = tuple(
            (module, bool(module.training))
            for module in parent_model.modules()
        )
        live_snapshot = _capture_model(context.child_model)
        transaction = Lfm2CpcfModelTransaction(context, live_snapshot)
        try:
            staging_model, staging_snapshot = (
                _make_disposable_staging_blank(
                    context.child_model,
                    live_snapshot,
                )
            )
            validate_lfm2_dense_budgeted_child(
                self.selected_target,
                staging_model,
                context.child_graph,
            )
            guard = _CompileSealedStorageWriteGuard(
                _compile_protected_tensor_bindings(
                    phases=sealed.phases,
                    source_tied_table=sealed.source_tied_table,
                    parent_model=parent_model,
                    live_child_model=context.child_model,
                    staging_child_model=staging_model,
                ),
                writable_bindings=_named_parameters(staging_model),
            )
            compile_error: BaseException | None = None
            result: Any = None
            serialized_compiler_report: dict[str, Any] | None = None
            try:
                with guard.mode:
                    result = compile_lfm2_cpcf_model_fold(
                        parent_model,
                        staging_model,
                        phases=sealed.phases,
                        source_tied_table=sealed.source_tied_table,
                        blank_target_report=sealed.blank_target_report,
                        planned_target_unique_parameters=(
                            self.selected_target.exact_target_unique_parameters
                        ),
                        policy=sealed.policy,
                    )
                    # Result serialization remains compiler-controlled code and
                    # therefore stays inside the same dispatch firewall.
                    serialized_compiler_report = dict(result.as_dict())
            except BaseException as exc:
                compile_error = exc
            try:
                current_guard_bindings = (
                    _compile_protected_tensor_bindings(
                        phases=sealed.phases,
                        source_tied_table=sealed.source_tied_table,
                        parent_model=parent_model,
                        live_child_model=context.child_model,
                        staging_child_model=staging_model,
                    )
                )
                guard_report = guard.assert_clean(
                    current_guard_bindings,
                    _named_parameters(staging_model),
                    require_complete_writes=(
                        compile_error is None
                    ),
                )
            except BaseException as guard_exc:
                if compile_error is not None:
                    raise guard_exc from compile_error
                raise
            if compile_error is not None:
                raise compile_error
            if serialized_compiler_report is None:
                raise Lfm2CpcfCatalogError(
                    "CPCF compiler returned no authenticated report."
                )
            compiler_report = _authenticate_compiled_model_fold_report(
                serialized_compiler_report,
                expected_result_sha256=getattr(
                    result,
                    "result_sha256",
                    None,
                ),
            )
            _assert_model_matches_snapshot_topology(
                staging_model,
                staging_snapshot,
                name="staging candidate",
            )
            validate_lfm2_dense_budgeted_child(
                self.selected_target,
                staging_model,
                context.child_graph,
            )
            staging_candidate = fingerprint_model_state(staging_model)
            staging_write_receipt = _parameter_path_write_receipt(
                staging_model,
                staging_snapshot,
                expected_unique_parameters=(
                    self.selected_target.exact_target_unique_parameters
                ),
                candidate_state_fingerprint=staging_candidate,
                authenticated_compiler_write_paths=(
                    guard_report[
                        "authenticated_writable_parameter_paths"
                    ]
                ),
            )
            # The dispatch firewall blocks writes before execution.  These
            # canonical post-hashes independently catch non-dispatch storage
            # replacement and preserve the original pre/post content proof.
            sealed.verify_parent_tensor_integrity()
            if fingerprint_model_state(parent_model) != parent_before:
                raise Lfm2CpcfCatalogError(
                    "CPCF detected a non-dispatch/raw-storage parent state "
                    "mutation. The parent resource is fatally invalidated; "
                    "restoration without duplicating the checkpoint cannot "
                    "be authenticated."
                )
            if (
                fingerprint_model_runtime_state(parent_model)
                != parent_runtime_before
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF model fold changed parent runtime topology."
                )
            if (
                fingerprint_model_runtime_state(context.child_model)
                != live_snapshot.runtime_fingerprint
                or fingerprint_model_state(context.child_model)
                != live_snapshot.state_fingerprint
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF compiler reached or changed the live blank before "
                    "authenticated staged installation."
                )
            (
                candidate,
                write_receipt,
                staged_install_receipt,
            ) = _install_staged_candidate(
                live_model=context.child_model,
                live_snapshot=live_snapshot,
                staging_model=staging_model,
                staging_snapshot=staging_snapshot,
                staging_candidate_state_fingerprint=staging_candidate,
                staging_parameter_receipt=staging_write_receipt,
                authenticated_compiler_write_paths=(
                    guard_report[
                        "authenticated_writable_parameter_paths"
                    ]
                ),
                expected_unique_parameters=(
                    self.selected_target.exact_target_unique_parameters
                ),
            )
            catalog_compile_report = _seal_catalog_compile_report(
                {
                    "method_id": self.method_id,
                    "source_parent": source_name,
                    "catalog_id": self.selected_target.catalog_id,
                    "evidence_bundle_sha256": sealed.bundle_sha256,
                    "parent_cache_key": sealed.cache_key,
                    "tokenizer_bridge_artifact_sha256": (
                        sealed.trace_adapter.bridge_artifact_sha256
                    ),
                    "parameter_path_write_receipt": dict(write_receipt),
                    "parameter_path_write_receipt_sha256": (
                        write_receipt["receipt_sha256"]
                    ),
                    "staging_parameter_path_write_receipt": dict(
                        staging_write_receipt
                    ),
                    "staging_parameter_path_write_receipt_sha256": (
                        staging_write_receipt["receipt_sha256"]
                    ),
                    "staged_install_receipt": dict(
                        staged_install_receipt
                    ),
                    "staged_install_receipt_sha256": (
                        staged_install_receipt["receipt_sha256"]
                    ),
                    "compile_sealed_storage_write_guard": dict(
                        guard_report
                    ),
                    "compile_sealed_storage_write_guard_sha256": (
                        guard_report["report_sha256"]
                    ),
                    "complete_target_parameter_overwrite": (
                        write_receipt[
                            "all_unique_parameter_storages_authentically_"
                            "written"
                        ]
                        is True
                        and write_receipt[
                            "resulting_path_hashes_verified"
                        ]
                        is True
                    ),
                    "seed_erasure_by_construction": True,
                    "parameter_topology_preserved": True,
                    "compiler_received_disposable_staging_blank": True,
                    "compiler_received_live_child": False,
                    "live_install_controlled_copy_only": True,
                    "parent_immutable": True,
                    "parent_traces_regenerated": False,
                    "quantization_used": False,
                    "pruning_used": False,
                    "training_used": False,
                    "zero_gradient": True,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                },
                compiled_model_fold_result_sha256=compiler_report[
                    "result_sha256"
                ],
            )
            transaction.bind_candidate(
                compiled_model_fold_report=compiler_report,
                catalog_compile_report=catalog_compile_report,
                candidate_state_fingerprint=candidate,
            )
            return transaction
        except BaseException as exc:
            for module, training in parent_module_modes:
                module.training = training
            try:
                if (
                    fingerprint_model_runtime_state(parent_model)
                    != parent_runtime_before
                ):
                    raise Lfm2CpcfCatalogError(
                        "CPCF failure changed non-mode parent runtime state; "
                        "the parent resource is fatally invalidated."
                    )
            except BaseException as restoration_exc:
                if transaction.active:
                    transaction.rollback()
                raise restoration_exc from exc
            if transaction.active:
                transaction.rollback()
            raise


class Lfm2CpcfFoldOnlyInitializer:
    """Prove that one non-mutating action owns the complete zero target."""

    def __init__(
        self,
        cache_key: str,
        selected_target: Lfm2DenseBudgetedSelectedTarget,
        blank_state_fingerprint: str,
    ) -> None:
        self.cache_key = _digest(cache_key, "cache_key")
        self.selected_target = selected_target
        self.blank_state_fingerprint = _digest(
            blank_state_fingerprint,
            "blank_state_fingerprint",
        )

    def initialize(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
    ) -> Mapping[str, Any]:
        before = context.child_state_fingerprint
        selected = self.selected_target
        if (
            before != self.blank_state_fingerprint
            or plan.child_state_fingerprint != self.blank_state_fingerprint
            or plan.evidence_fingerprint != self.cache_key
            or plan.exact_target_unique_parameters
            != selected.exact_target_unique_parameters
            or len(plan.actions) != 1
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF plan budget, evidence, action count, or blank changed."
            )
        action = plan.actions[0]
        if (
            action.method_id != DENSE_BUDGET_MODEL_METHOD
            or not action.signature.zero_gradient
            or action.compression_delta <= 0
            or set(action.target_node_ids)
            != {node.node_id for node in context.child_graph.nodes}
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF target must compile as one positive-compaction, "
                "zero-gradient, complete-model action."
            )
        if (
            len(context.parent_graphs) != 1
            or dict(plan.parent_state_fingerprints)
            != {
                selected.source_parent: selected.source_state_fingerprint
            }
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF execution parent differs from target selection."
            )
        parent_graph = context.parent_graphs[selected.source_parent]
        if (
            parent_graph.fingerprint != selected.source_graph_fingerprint
            or len(parent_graph.layer_indices) != selected.target_depth
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF source graph differs from its selected equal-depth lane."
            )
        validate_lfm2_dense_budgeted_child(
            selected,
            context.child_model,
            context.child_graph,
        )
        if context.child_state_fingerprint != before:
            raise Lfm2CpcfCatalogError("CPCF initializer mutated the target.")
        return {
            "kind": "lfm2_cpcf_fold_only",
            "child_state_fingerprint": before,
            "complete_target_coverage": True,
            "parent_cache_fingerprint": self.cache_key,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


class Lfm2CpcfAdmissionEvaluator:
    """Local structural admission; external intelligence remains mandatory."""

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        transaction = context.transaction
        if not isinstance(transaction, Lfm2CpcfModelTransaction):
            return FoldDecision.reject(
                "CPCF action lacks its typed whole-model transaction."
            )
        report = dict(transaction.report)
        operation_audit = report.get("operation_audit")
        receipt = report.get("parameter_receipt")
        try:
            path_receipt = transaction.verify_parameter_path_write_receipt()
        except Lfm2CpcfCatalogError:
            path_receipt = None
        audit_zero = isinstance(operation_audit, Mapping) and all(
            operation_audit.get(name) == 0
            for name in (
                "parent_model_forward_calls",
                "target_model_forward_calls",
                "generation_calls",
                "parent_cache_generation_calls",
                "training_steps",
                "backward_calls",
                "optimizer_steps",
                "parameters_pruned",
                "parameters_quantized",
                "fallback_paths",
            )
        )
        receipt_valid = (
            isinstance(receipt, Mapping)
            and receipt.get("parameter_count_compaction_only") is True
            and receipt.get("checkpoint_byte_ratio_used") is False
            and receipt.get("quantization_used") is False
            and receipt.get("pruning_used") is False
            and receipt.get("target_unique_parameters")
            == context.execution_context.child_graph.unique_parameter_count
        )
        path_receipt_valid = (
            isinstance(path_receipt, Mapping)
            and path_receipt.get(
                "all_named_parameter_paths_accounted"
            )
            is True
            and path_receipt.get(
                "all_unique_parameter_storages_authentically_written"
            )
            is True
            and path_receipt.get("resulting_path_hashes_verified") is True
            and path_receipt.get("tied_aliases_explicitly_accounted") is True
            and path_receipt.get("receipt_sha256")
            == report.get("parameter_path_write_receipt_sha256")
        )
        metrics: dict[str, JsonScalar] = {
            "local_gate_only": True,
            "model_level_action": True,
            "all_layers_compiled_in_order": bool(
                report.get("all_layers_compiled_in_order", False)
            ),
            "actual_folded_upstream_replay_used": bool(
                report.get("actual_folded_upstream_replay_used", False)
            ),
            "complete_target_parameter_overwrite": bool(
                report.get("complete_target_parameter_overwrite", False)
            ),
            "seed_erasure_by_construction": bool(
                report.get("seed_erasure_by_construction", False)
            ),
            "source_state_preserved": bool(
                report.get("source_state_preserved", False)
            ),
            "target_parameter_count_preserved": bool(
                report.get("target_parameter_count_preserved", False)
            ),
            "operation_audit_zero": audit_zero,
            "parameter_receipt_valid": receipt_valid,
            "parameter_path_write_receipt_valid": path_receipt_valid,
            "quantization_used": bool(
                report.get("quantization_used", True)
            ),
            "pruning_used": bool(report.get("pruning_used", True)),
            "training_used": bool(report.get("training_used", True)),
            "zero_gradient": bool(report.get("zero_gradient", False)),
        }
        required = (
            metrics["all_layers_compiled_in_order"],
            metrics["actual_folded_upstream_replay_used"],
            metrics["complete_target_parameter_overwrite"],
            metrics["seed_erasure_by_construction"],
            metrics["source_state_preserved"],
            metrics["target_parameter_count_preserved"],
            metrics["operation_audit_zero"],
            metrics["parameter_receipt_valid"],
            metrics["parameter_path_write_receipt_valid"],
            not metrics["quantization_used"],
            not metrics["pruning_used"],
            not metrics["training_used"],
            metrics["zero_gradient"],
        )
        if not all(required):
            return FoldDecision.reject(
                "CPCF model transaction failed a fold-only structural "
                "invariant.",
                metrics,
            )
        return FoldDecision.accept(
            "CPCF filled the compiler blank by sealed parent-derived circuit "
            "folding; the independent external intelligence gate remains "
            "mandatory.",
            metrics,
        )


class Lfm2CpcfFoldStackAdapter:
    """Higher-priority LFM2 catalog lane for bridged CPCF evidence."""

    adapter_id = LFM2_CPCF_STACK_ADAPTER_ID
    version = 1
    priority = 200
    target_families = ("lfm2",)
    target_modalities = ("text",)
    method_ids = (DENSE_BUDGET_MODEL_METHOD,)

    def __init__(self, resources: Lfm2CpcfStackResources) -> None:
        if not isinstance(resources, Lfm2CpcfStackResources):
            raise Lfm2CpcfCatalogError(
                "resources must be Lfm2CpcfStackResources."
            )
        self.resources = resources
        self.resource_fingerprint = _digest(
            resources.resource_fingerprint,
            "resources.resource_fingerprint",
        )

    def _selected_target(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> Lfm2DenseBudgetedSelectedTarget:
        if (
            not isinstance(target, DynamicTargetSynthesisResult)
            or target.adapter_id != Lfm2TargetSynthesisAdapter.adapter_id
            or target.adapter_version != Lfm2TargetSynthesisAdapter.version
            or not isinstance(
                target.adapter_payload,
                Lfm2DenseBudgetedSelectedTarget,
            )
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF stack requires a dense LFM2 catalog target."
            )
        return target.adapter_payload

    def _validate_target(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> tuple[
        Lfm2DenseBudgetedSelectedTarget,
        Mapping[str, Any],
        BehaviorGateBinding,
    ]:
        selected = self._selected_target(target)
        resources = self.resources
        binding = _strict_behavior_binding(
            resources.final_evaluator,
            resources.external_behavior_contract,
            name="final_evaluator",
        )
        trace = resources.trace_adapter.verify()
        reviewed = resources.final_evaluator.evidence
        phase_cache_keys = {
            phase.parent_evidence.cache_key for phase in resources.phases
        }
        if (
            target.resource_profile.family != "lfm2"
            or target.resource_profile.modalities != ("text",)
            or target.graph.family != "lfm2"
            or target.graph.model_name != target.request.child_name
            or resources.parent_graph_fingerprint
            != selected.source_graph_fingerprint
            or reviewed.blank_target_parent_graph_fingerprint
            != selected.source_graph_fingerprint
            or reviewed.live_parent_state_fingerprint
            != selected.source_state_fingerprint
            or phase_cache_keys != {reviewed.parent_cache_key}
            or reviewed.trace_adapter_sha256 != trace.evidence_sha256
            or not selected.tokenizer_bridge_required
            or trace.bridge_artifact_sha256
            != selected.tokenizer_bridge_artifact_sha256
            or trace.transport_profile_fingerprint
            != selected.tokenizer_transport_profile_sha256
            or reviewed.bridge_artifact_sha256
            != selected.tokenizer_bridge_artifact_sha256
            or reviewed.bridge_profile_sha256
            != selected.tokenizer_transport_profile_sha256
            or reviewed.target_tokenizer_semantics_sha256
            != selected.child_tokenizer_semantics_sha256
            or trace.target_tokenizer_semantics_sha256
            != selected.child_tokenizer_semantics_sha256
            or reviewed.candidate_tokenizer_fingerprint
            != trace.target_tokenizer_package_fingerprint
            or trace.target_row_capacity != selected.vocabulary_size
            or target.request.exact_unique_parameters
            != selected.exact_target_unique_parameters
            or target.resource_profile.unique_parameters
            != selected.exact_target_unique_parameters
            or target.graph.unique_parameter_count
            != selected.exact_target_unique_parameters
            or reviewed.exact_target_unique_parameters
            != selected.exact_target_unique_parameters
            or reviewed.target_topology_fingerprint
            != target.topology_fingerprint
            or reviewed.blank_target_state_fingerprint
            != target.blank_state_fingerprint
            or dict(target.parent_graph_fingerprints)
            != {
                selected.source_parent: selected.source_graph_fingerprint
            }
        ):
            raise Lfm2CpcfCatalogError(
                "Synthesized target differs from CPCF topology, exact budget, "
                "parent, or tokenizer bridge evidence."
            )
        source_depth = sum(
            len(span) for span in selected.ordered_source_spans
        )
        if (
            selected.target_depth != source_depth
            or selected.ordered_source_spans
            != tuple((index,) for index in range(source_depth))
            or any(
                len(phase.parent_evidence.layers) != source_depth
                for phase in resources.phases
            )
            or int(resources.source_tied_table.shape[0])
            != selected.vocabulary_size
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF evidence is not complete for the selected equal-depth "
                "dense target."
            )
        validate_lfm2_dense_budgeted_child(
            selected,
            target.model,
            target.graph,
        )
        report = _blank_report(target, selected)
        return selected, report, binding

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        try:
            self._validate_target(target)
        except Exception:
            # Capability discovery must be fail-closed and must never claim a
            # high-priority CPCF route that build() would reject.  The exact
            # validator is read-only: it reauthenticates the reviewed gate and
            # trace, checks every target/resource binding, validates the live
            # native topology, and verifies compiler-blank provenance.
            return False
        return True

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        selected, blank_report, binding = self._validate_target(target)
        evidence = Lfm2CpcfSealedEvidence(
            phases=self.resources.phases,
            trace_adapter=self.resources.trace_adapter,
            source_tied_table=self.resources.source_tied_table,
            blank_target_report=blank_report,
            policy=self.resources.policy,
        )
        return DynamicFoldSystemStack(
            operators=FoldOperatorRegistry(
                (Lfm2CpcfModelOperator(selected, evidence),)
            ),
            evidence_provider=Lfm2CpcfEvidenceProvider(evidence),
            evaluator=Lfm2CpcfAdmissionEvaluator(),
            final_evaluator=self.resources.final_evaluator,
            model_saver=self.resources.model_saver,
            model_loader=self.resources.model_loader,
            fresh_reload_evaluator=(
                self.resources.fresh_reload_evaluator
            ),
            saver_id=self.resources.saver_id,
            loader_id=self.resources.loader_id,
            behavior_gate_id=binding.gate_id,
            behavior_gate_version=binding.gate_version,
            behavior_baseline_fingerprint=(
                binding.baseline_fingerprint
            ),
            behavior_gate_manifest_fingerprint=(
                binding.manifest_fingerprint
            ),
            behavior_gate_manifest=binding.as_dict(),
            initializer=Lfm2CpcfFoldOnlyInitializer(
                evidence.cache_key,
                selected,
                target.blank_state_fingerprint,
            ),
            snapshot_config=self.resources.snapshot_config,
            evidence_fingerprint=evidence.cache_key,
            publication_recorder=None,
        )


__all__ = [
    "LFM2_CPCF_CATALOG_SCHEMA",
    "LFM2_CPCF_EXTERNAL_GATE_SCHEMA",
    "LFM2_CPCF_PARAMETER_PATH_WRITE_RECEIPT_SCHEMA",
    "LFM2_CPCF_PARENT_TENSOR_SEAL_SCHEMA",
    "LFM2_CPCF_RESOURCE_SCHEMA",
    "LFM2_CPCF_STACK_ADAPTER_ID",
    "Lfm2CpcfAdmissionEvaluator",
    "Lfm2CpcfCatalogError",
    "Lfm2CpcfEvidenceProvider",
    "Lfm2CpcfExternalBehaviorGateContract",
    "Lfm2CpcfFoldOnlyInitializer",
    "Lfm2CpcfFoldStackAdapter",
    "Lfm2CpcfModelOperator",
    "Lfm2CpcfModelTransaction",
    "Lfm2CpcfSealedEvidence",
    "Lfm2CpcfStackResources",
]
