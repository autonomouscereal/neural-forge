from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.dynamic_counted_child_state import (  # noqa: E402
    ContinuousChildStateAllocation,
    build_counted_child_state_plan,
)
from neural_forge.dynamic_fold_phase_cache import (  # noqa: E402
    DynamicFoldPhaseCache,
    FoldPhaseKey,
)
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (  # noqa: E402
    LFM2_BALANCED_BASIS_PHASE,
    LFM2_BALANCED_BASIS_PHASE_ABI,
    LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
    LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA,
    AuthenticatedLfm2BalancedBasisPhase,
    PhaseAnchorAuthority,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    SELECTIVE_BATCH_PAYLOAD_SCHEMA,
)
from neural_forge import hf_lfm2_exact4x_causal_diagnostic as causal  # noqa: E402
from neural_forge import hf_lfm2_exact4x_first_divergence as divergence  # noqa: E402
from neural_forge.hf_lfm2_moving_gauge_boundary_phase_cache import (  # noqa: E402
    AuthenticatedLfm2MovingGaugeBoundaryPhase,
    Lfm2MovingGaugeBoundaryExtractionProgress,
    Lfm2MovingGaugeBoundaryExtractionSources,
    Lfm2MovingGaugeBoundaryImportRequest,
    Lfm2MovingGaugeBoundaryLayerBinding,
    Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
    Lfm2MovingGaugeBoundaryPhaseAuthority,
    fingerprint_lfm2_moving_gauge_basis_split,
    lfm2_moving_gauge_boundary_implementation_sha256,
    load_existing_lfm2_moving_gauge_boundary_phase,
    load_or_extract_lfm2_moving_gauge_boundary_phase,
)
from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (  # noqa: E402
    fingerprint_lfm2_moving_gauge_tensor,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _BridgeRecord:
    ordinal: int
    record_id: str
    split: str
    alignment_sha256: str
    donor_cache_input_sha256: str
    target_cache_input_sha256: str


class _Adapter:
    def __init__(self, records: tuple[_BridgeRecord, ...], provenance) -> None:
        self.records = records
        self.evidence_sha256 = provenance["bridge_trace_adapter"]
        self.bridge_artifact_sha256 = provenance["bridge_artifact"]
        self.transport_profile_fingerprint = provenance["bridge_profile"]
        self.verify_calls = 0

    def verify(self):
        self.verify_calls += 1
        return self


@dataclass(frozen=True)
class _Reference:
    index: int
    input_sha256: str
    record_sha256: str
    ordered_tensor_names: tuple[str, ...]
    _inventory: MappingProxyType

    @property
    def record(self):
        return {"tensors": dict(self._inventory)}


class _Payload:
    def __init__(
        self,
        index: int,
        names: tuple[str, ...],
        tensors,
        *,
        cache_key: str,
        session_sha256: str,
        reference: _Reference,
    ) -> None:
        self.tensors = MappingProxyType({name: tensors[name].clone() for name in names})
        self.batch_index = index
        self.input_sha256 = reference.input_sha256
        self.ordered_tensor_names = names
        self.session_sha256 = session_sha256
        inventory = reference.record["tensors"]
        self.logical_bytes = sum(int(inventory[name]["size_bytes"]) for name in names)
        body = {
            "schema": SELECTIVE_BATCH_PAYLOAD_SCHEMA,
            "session_sha256": session_sha256,
            "cache_key": cache_key,
            "batch_index": index,
            "input_sha256": reference.input_sha256,
            "record_sha256": reference.record_sha256,
            "ordered_tensor_names": list(names),
            "ordered_tensor_descriptors": [dict(inventory[name]) for name in names],
            "logical_bytes": self.logical_bytes,
        }
        self.selection_sha256 = canonical_sha256(body)
        self.released = False


class _Session:
    def __init__(self, cache_key, evidence_sha256, references, tensors) -> None:
        self.cache_key = cache_key
        self.evidence_sha256 = evidence_sha256
        self.batch_references = references
        self._tensors = tensors
        self.load_calls = 0
        self.verify_calls = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def load_selected_tensors(self, index, names):
        self.load_calls += 1
        return _Payload(
            index,
            names,
            self._tensors[index],
            cache_key=self.cache_key,
            session_sha256=self.evidence_sha256,
            reference=self.batch_references[index],
        )

    @staticmethod
    def release_selected_tensors(payload):
        payload.released = True

    def verify(self):
        self.verify_calls += 1
        return self


class _RejectingSession:
    def __init__(self, inner) -> None:
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __enter__(self):
        self.inner.__enter__()
        return self

    def __exit__(self, *args):
        return self.inner.__exit__(*args)

    def verify(self):
        raise RuntimeError("synthetic selective-session verification failure")


class _PoisoningSession:
    def __init__(self, inner) -> None:
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __enter__(self):
        self.inner.__enter__()
        return self

    def __exit__(self, *args):
        return self.inner.__exit__(*args)

    def load_selected_tensors(self, index, names):
        payload = self.inner.load_selected_tensors(index, names)
        tensors = dict(payload.tensors)
        first = names[0]
        tensors[first] = tensors[first] + 123
        payload.tensors = MappingProxyType(tensors)
        return payload


class _Cache:
    def __init__(self, key: str, manifest: dict[str, Any], completion: str) -> None:
        self.cache_key = key
        self.manifest = manifest
        self._completion = MappingProxyType({"completion_sha256": completion})


def _descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "size_bytes": tensor.numel() * tensor.element_size(),
        "sha256": fingerprint_lfm2_moving_gauge_tensor(tensor),
    }


def _provenance() -> dict[str, str]:
    names = (
        "basis_completion",
        "basis_map",
        "basis_phase_key",
        "bridge_artifact",
        "bridge_profile",
        "bridge_trace_adapter",
        "child_completion",
        "child_identity",
        "child_manifest",
        "child_package",
        "child_selective_session",
        "child_state_dict",
        "child_topology",
        "mapped_calibration_file",
        "parent_completion",
        "parent_identity",
        "parent_manifest",
        "parent_selective_session",
        "parent_topology",
        "portable_runtime_receipt",
        "r12_run_receipt",
        "route_package",
        "route_runtime",
        "source_spans",
        "v3_ledger",
        "v3_request",
    )
    return {name: _sha(name) for name in names}


def _implementation_identity() -> dict[str, Any]:
    file_names = (
        "balanced_basis_phase",
        "canonical_lfm2",
        "carrier_child_package",
        "carrier_relocation_package",
        "core",
        "direct_local_loader",
        "dynamic_blank_execution",
        "routing_runtime",
        "runner",
        "teacher_forced_readout",
        "trace_cache",
        "transformers_lfm2_runtime",
    )
    body = {
        "schema": divergence.LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA,
        "files": {
            name: {"path": f"{name}.py", "sha256": _sha(name)} for name in file_names
        },
        "packages": {
            "python": "test",
            "safetensors": "test",
            "torch": "test",
            "transformers": "test",
        },
        "numeric_contract": {"runtime_device": "cpu", "runtime_dtype": "float32"},
        "execution_contract": {
            "network_allowed": False,
            "parent_model_loaded": False,
            "parent_forward_calls": 0,
            "training_steps": 0,
        },
    }
    return {**body, "implementation_sha256": canonical_sha256(body)}


def _v3_documents(
    provenance: dict[str, str],
    *,
    parent_parameter_count: int = 100,
    dense_parameter_count: int = 20,
    continuous_parameter_count: int = 5,
    child_parameter_count: int = 25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    budget_body = {
        "schema": divergence.LFM2_EXACT4X_BUDGET_OBSERVATION_SCHEMA,
        "parent_parameter_elements": parent_parameter_count,
        "physical_dense_parameter_elements": dense_parameter_count,
        "continuous_route_parameter_elements": continuous_parameter_count,
        "counted_total_parameter_elements": child_parameter_count,
        "target_total_parameter_elements": child_parameter_count,
        "parameter_compaction_ratio": (parent_parameter_count / child_parameter_count),
        "parameter_count_authority": {"caller_parameter_count_assertion_used": False},
        "at_least_fourfold_parameter_compaction": True,
        "one_complete_child_observed": True,
        "partial_component_variant": False,
        "pruning_credit": False,
        "quantization_credit": False,
        "parameter_sharing_credit": False,
    }
    budget = {
        **budget_body,
        "budget_observation_sha256": canonical_sha256(budget_body),
    }
    logit_contract = divergence.build_lfm2_exact4x_parent_logit_domain_contract(
        child_package_sha256=provenance["child_package"],
        route_package_sha256=provenance["route_package"],
        semantic_source_tokenizer_sha256=_sha("semantic-source-tokenizer"),
        semantic_source_lexicon_package_sha256=_sha("semantic-source-lexicon"),
        packaged_bridge_reference_sha256=_sha("bridge-reference-file"),
        bridge_reference_schema="test-bridge-reference-v1",
        bridge_artifact_sha256=provenance["bridge_artifact"],
        bridge_profile_sha256=provenance["bridge_profile"],
        bridge_target_tokenizer_semantics_sha256=_sha("target-tokenizer"),
        bridge_target_tokenizer_package_fingerprint=_sha("target-package"),
        trace_cache_key=_sha("parent-cache-key"),
        trace_selective_session_sha256=provenance["parent_selective_session"],
        trace_logit_descriptor_inventory_sha256=_sha("trace-logit-descriptors"),
        trace_batch_count=2,
        trace_phase_count=2,
        defined_semantic_row_count=8,
        maximum_defined_token_id=7,
        route_donor_row_count=8,
        parent_physical_row_capacity=8,
    )
    v3_request = divergence.build_lfm2_exact4x_first_divergence_request(
        child_package_sha256=provenance["child_package"],
        child_state_sha256=provenance["child_identity"],
        route_package_sha256=provenance["route_package"],
        route_runtime_sha256=provenance["route_runtime"],
        portable_runtime_receipt_sha256=provenance["portable_runtime_receipt"],
        trace_cache_key=_sha("parent-cache-key"),
        trace_selective_session_sha256=provenance["parent_selective_session"],
        basis_phase_key_sha256=provenance["basis_phase_key"],
        basis_completion_sha256=provenance["basis_completion"],
        basis_map_sha256=provenance["basis_map"],
        budget_observation=budget,
        layer_count=2,
        focus_layer_indices=(0,),
        runtime_device="cpu",
        runtime_dtype="float32",
        maximum_selected_parent_bytes=1_000_000,
        implementation_identity=_implementation_identity(),
        parent_logit_domain_contract=logit_contract,
    )
    ledger_body = {
        "schema": causal.LFM2_EXACT4X_FIRST_DIVERGENCE_LEDGER_SCHEMA,
        "status": "complete-observation",
        "request": v3_request,
        "ordered_batch_audit_sha256s": [_sha("batch-0"), _sha("batch-1")],
        "batch_count": 2,
        "split_summaries": {
            "fit": {"batch_count": 1},
            "heldout": {"batch_count": 1},
        },
        "first_discrete_c_out_parent_winner_mismatch": None,
        "component_first_divergence_is_descriptive_not_thresholded": True,
        "numeric_metrics_are_diagnostic_only": True,
        "numeric_acceptance_decision": None,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "safety_observations_are_separate_not_failures": True,
        "generation_cap_saturation_is_censored_not_failure": True,
        "free_generation_performed": False,
        "operation_audit": {},
    }
    v3_ledger = {**ledger_body, "ledger_sha256": canonical_sha256(ledger_body)}
    return v3_request, v3_ledger


def _request(
    provenance: dict[str, str],
    *,
    parent_parameter_count: int = 100,
    child_parameter_count: int = 25,
    basis_split: dict[int, str] | None = None,
) -> Lfm2MovingGaugeBoundaryImportRequest:
    basis_split = basis_split or {0: "fit", 1: "heldout"}
    return Lfm2MovingGaugeBoundaryImportRequest(
        parent_cache_key=_sha("parent-cache-key"),
        child_cache_key=_sha("child-cache-key"),
        expected_batch_count=2,
        expected_aligned_row_count=6,
        parent_width=3,
        child_width=2,
        parent_parameter_count=parent_parameter_count,
        child_parameter_count=child_parameter_count,
        parent_storage_dtype="bfloat16",
        child_storage_dtype="float32",
        layer_bindings=(
            Lfm2MovingGaugeBoundaryLayerBinding(0, 0, 0, "short_conv"),
            Lfm2MovingGaugeBoundaryLayerBinding(1, 1, 1, "full_attention"),
        ),
        focused_writer_layer_indices=(0,),
        basis_split_sha256=fingerprint_lfm2_moving_gauge_basis_split(
            basis_split,
            basis_phase_key_sha256=provenance["basis_phase_key"],
            basis_completion_sha256=provenance["basis_completion"],
            basis_map_sha256=provenance["basis_map"],
        ),
        implementation_sha256=lfm2_moving_gauge_boundary_implementation_sha256(),
        provenance_sha256s=provenance,
        checkpoint_hmac_key_id="moving-gauge-test",
        shard_batch_count=1,
    )


def _source_tensors(side: str, width: int, dtype: torch.dtype):
    all_batches = []
    fields = {
        0: (
            "mlp.down",
            "operator.output",
            "residual.post_layer",
            "residual.post_operator",
            "residual.pre",
        ),
        1: (
            "residual.post_layer",
            "residual.post_operator",
            "residual.pre",
        ),
    }
    for batch in range(2):
        tensors = {}
        for phase_index, (phase, rows) in enumerate(
            (("prefill", 2), ("decode.000", 1))
        ):
            for layer, layer_fields in fields.items():
                for field_index, field_name in enumerate(layer_fields):
                    offset = (
                        (0 if side == "parent" else 10_000)
                        + batch * 1_000
                        + phase_index * 200
                        + layer * 50
                        + field_index * 7
                    )
                    value = torch.arange(rows * width, dtype=torch.float32).reshape(
                        1, rows, width
                    )
                    tensors[f"{phase}.layer.{layer:03d}.{field_name}"] = (
                        value + offset
                    ).to(dtype=dtype)
        all_batches.append(tensors)
    return tuple(all_batches)


_BASIS_SECRET = b"balanced-basis-boundary-test-secret"


def _counted_plan(
    provenance: dict[str, str],
    *,
    input_elements: int = 2,
    output_elements: int = 3,
):
    artifact = provenance["route_package"]
    return build_counted_child_state_plan(
        parent_unique_parameters=100,
        target_total_parameter_elements=25,
        continuous_allocations=(
            ContinuousChildStateAllocation(
                state_id="semantic-carrier-route.c_in",
                parameter_elements=input_elements,
                tensor_sha256=_sha(f"route-c-in-{input_elements}"),
                artifact_sha256=artifact,
            ),
            ContinuousChildStateAllocation(
                state_id="semantic-carrier-route.c_out",
                parameter_elements=output_elements,
                tensor_sha256=_sha(f"route-c-out-{output_elements}"),
                artifact_sha256=artifact,
            ),
        ),
    )


def _authenticated_basis_phase(provenance, records, counted_plan, phase_cache):
    authority = PhaseAnchorAuthority(
        key_id="balanced-basis-test",
        secret=_BASIS_SECRET,
    )
    inventory_body = {
        "schema": "test-lfm2-weight-complete-inventory-v1",
        "source_width": 3,
        "parameter_alias_groups": [
            {"canonical_path": "first.weight", "numel": 60},
            {"canonical_path": "second.weight", "numel": 40},
        ],
    }
    inventory = {
        **inventory_body,
        "inventory_sha256": canonical_sha256(inventory_body),
    }
    external = {
        "carrier_route_artifact": provenance["route_package"],
        "counted_child_state_plan": counted_plan.fingerprint,
    }
    phase_key = FoldPhaseKey(
        phase_name=LFM2_BALANCED_BASIS_PHASE,
        dependency_sha256s={
            "parent_inventory": inventory["inventory_sha256"],
            **{f"external.{name}": value for name, value in external.items()},
        },
        numeric_contract={"calculation_dtype": "torch.float64"},
        abi={
            "schema": LFM2_BALANCED_BASIS_PHASE_ABI,
            "receipt_schema": LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
            "source_anchor_schema": LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA,
            "source_anchor_key_id": authority.key_id,
            "source_width": 3,
            "target_width": 2,
        },
    )
    provenance["basis_phase_key"] = phase_key.phase_key_sha256
    provenance["basis_map"] = _sha("authenticated-basis-map")
    basis_report_body = {
        "cache": {
            "record_horizons": [
                {
                    "batch_index": index,
                    "record_id": record.donor_cache_input_sha256,
                    "split": ("fit", "heldout")[index],
                }
                for index, record in enumerate(records)
            ]
        }
    }
    basis_report = {
        **basis_report_body,
        "result_sha256": canonical_sha256(basis_report_body),
    }
    nested_body = {
        "hashes": {"basis_sha256": provenance["basis_map"]},
    }
    nested_report = {
        **nested_body,
        "run_sha256": canonical_sha256(nested_body),
    }
    descriptors = [
        {
            "name": "basis",
            "dtype": "torch.float64",
            "shape": [3, 2],
            "numel": 6,
            "size_bytes": 48,
            "sha256": provenance["basis_map"],
        }
    ]
    tensor_payload = b"b" * 48
    tensor_file = {
        "filename": "basis.safetensors",
        "format": "safetensors",
        "metadata": {"contract": "test"},
        "size_bytes": 48,
        "sha256": hashlib.sha256(tensor_payload).hexdigest(),
    }
    bindings = {
        "schema": "test-balanced-basis-bindings-v1",
        "inventory": inventory,
        "inventory_sha256": inventory["inventory_sha256"],
        "selected_target": {
            "hidden_size": 2,
            "exact_target_unique_parameters": (
                counted_plan.dense_model_parameter_elements
            ),
        },
        "external_dependency_sha256s": external,
    }
    receipt_body = {
        "schema": LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
        "phase_key": phase_key.as_dict(),
        "phase_key_sha256": phase_key.phase_key_sha256,
        "bindings": bindings,
        "basis_report": basis_report,
        "balanced_basis_report": nested_report,
        "tensor_file": tensor_file,
        "tensor_descriptors": descriptors,
        "execution": {"training_steps": 0},
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": canonical_sha256(receipt_body),
    }
    anchor_body = {
        "schema": LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA,
        "algorithm": "hmac-sha256",
        "key_id": authority.key_id,
        "phase_name": LFM2_BALANCED_BASIS_PHASE,
        "phase_key_sha256": phase_key.phase_key_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "tensor_file_sha256": tensor_file["sha256"],
        "tensor_inventory_sha256": canonical_sha256(descriptors),
        "basis_result_sha256": basis_report["result_sha256"],
        "balanced_run_sha256": nested_report["run_sha256"],
        "basis_map_sha256": provenance["basis_map"],
    }
    anchor = {**anchor_body, "hmac_sha256": authority.sign(anchor_body)}
    completion_result = {
        "schema": LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
        "receipt_sha256": receipt["receipt_sha256"],
        "tensor_file_sha256": tensor_file["sha256"],
        "tensor_inventory_sha256": canonical_sha256(descriptors),
        "basis_result_sha256": basis_report["result_sha256"],
        "balanced_run_sha256": nested_report["run_sha256"],
        "basis_map_sha256": provenance["basis_map"],
    }
    with phase_cache.begin(phase_key) as transaction:
        (transaction.payload_path / "basis.safetensors").write_bytes(tensor_payload)
        (transaction.payload_path / "receipt.json").write_bytes(
            canonical_json_bytes(receipt) + b"\n"
        )
        (transaction.payload_path / "source-anchor.json").write_bytes(
            canonical_json_bytes(anchor) + b"\n"
        )
        entry = transaction.complete(result=completion_result)
    provenance["basis_completion"] = entry.completion_sha256
    phase = AuthenticatedLfm2BalancedBasisPhase(
        phase_key=phase_key,
        completion_sha256=entry.completion_sha256,
        receipt=MappingProxyType(receipt),
        source_anchor=MappingProxyType(anchor),
        basis_result=SimpleNamespace(
            source_width=3,
            target_width=2,
            map_sha256=provenance["basis_map"],
            report=MappingProxyType(nested_report),
        ),
        report=MappingProxyType(basis_report),
    )
    return phase, authority


def _case(tmp_path):
    provenance = _provenance()
    records = tuple(
        _BridgeRecord(
            ordinal=index,
            record_id=f"record-{index}",
            split=("selection", "validation")[index],
            alignment_sha256=_sha(f"alignment-{index}"),
            donor_cache_input_sha256=_sha(f"parent-input-{index}"),
            target_cache_input_sha256=_sha(f"child-input-{index}"),
        )
        for index in range(2)
    )
    adapter = _Adapter(records, provenance)
    counted_plan = _counted_plan(provenance)
    basis_phase_cache = DynamicFoldPhaseCache(tmp_path / "basis-phase-cache")
    basis_phase, basis_authority = _authenticated_basis_phase(
        provenance,
        records,
        counted_plan,
        basis_phase_cache,
    )
    v3_request, v3_ledger = _v3_documents(provenance)
    provenance["v3_request"] = v3_request["request_sha256"]
    provenance["v3_ledger"] = v3_ledger["ledger_sha256"]
    request = _request(provenance)
    parent_tensors = _source_tensors("parent", 3, torch.bfloat16)
    child_tensors = _source_tensors("child", 2, torch.float32)

    def references(side, tensors):
        return tuple(
            _Reference(
                index=index,
                input_sha256=(
                    records[index].donor_cache_input_sha256
                    if side == "parent"
                    else records[index].target_cache_input_sha256
                ),
                record_sha256=_sha(f"{side}-record-{index}"),
                ordered_tensor_names=tuple(sorted(batch)),
                _inventory=MappingProxyType(
                    {name: _descriptor(value) for name, value in batch.items()}
                ),
            )
            for index, batch in enumerate(tensors)
        )

    parent_refs = references("parent", parent_tensors)
    child_refs = references("child", child_tensors)
    parent_manifest = {
        "manifest_sha256": provenance["parent_manifest"],
        "parent": {
            "identity_sha256": provenance["parent_identity"],
            "state_dict_sha256": _sha("parent-state"),
        },
        "corpus": {
            "batch_count": 2,
            "ordered_batch_input_sha256": [
                value.donor_cache_input_sha256 for value in records
            ],
        },
        "layer_topology": {
            "layer_topology_sha256": provenance["parent_topology"],
            "layers": [
                {"contract": {"kind": "short_conv", "hidden_size": 3}},
                {"contract": {"kind": "full_attention", "hidden_size": 3}},
            ],
        },
    }
    child_manifest = {
        "manifest_sha256": provenance["child_manifest"],
        "parent": {
            "identity_sha256": provenance["child_identity"],
            "state_dict_sha256": provenance["child_state_dict"],
        },
        "corpus": {
            "batch_count": 2,
            "ordered_batch_input_sha256": [
                value.target_cache_input_sha256 for value in records
            ],
        },
        "layer_topology": {
            "layer_topology_sha256": provenance["child_topology"],
            "layers": [
                {"contract": {"kind": "short_conv", "hidden_size": 2}},
                {"contract": {"kind": "full_attention", "hidden_size": 2}},
            ],
        },
    }
    parent_cache = _Cache(
        request.parent_cache_key,
        parent_manifest,
        provenance["parent_completion"],
    )
    child_cache = _Cache(
        request.child_cache_key,
        child_manifest,
        provenance["child_completion"],
    )
    calls = {"provider": 0, "parent_session": 0, "child_session": 0}

    def parent_factory():
        calls["parent_session"] += 1
        return _Session(
            request.parent_cache_key,
            provenance["parent_selective_session"],
            parent_refs,
            parent_tensors,
        )

    def child_factory():
        calls["child_session"] += 1
        return _Session(
            request.child_cache_key,
            provenance["child_selective_session"],
            child_refs,
            child_tensors,
        )

    def provider():
        calls["provider"] += 1
        return Lfm2MovingGaugeBoundaryExtractionSources(
            parent_cache=parent_cache,
            child_cache=child_cache,
            trace_adapter=adapter,
            basis_phase=basis_phase,
            basis_phase_authority=basis_authority,
            basis_phase_cache=basis_phase_cache,
            counted_child_state_plan=counted_plan,
            v3_first_divergence_request=v3_request,
            v3_first_divergence_ledger=v3_ledger,
            parent_session_factory=parent_factory,
            child_session_factory=child_factory,
        )

    return request, provider, calls


def test_zero_forward_boundary_phase_resumes_and_builds_evidence(tmp_path) -> None:
    request, provider, calls = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "phase-cache")
    checkpoint_root = tmp_path / "checkpoints"
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"a" * 32,
    )
    partial = load_or_extract_lfm2_moving_gauge_boundary_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
        maximum_new_shards=1,
    )
    assert isinstance(partial, Lfm2MovingGaugeBoundaryExtractionProgress)
    assert partial.completed_shards == 1
    assert partial.total_shards == 4
    work_root = checkpoint_root / request.phase_key.phase_key_sha256
    # An uncheckpointed shard-looking file is never admitted to the chain.
    (work_root / "shards" / "orphan.safetensors").write_bytes(b"orphan")

    completed = load_or_extract_lfm2_moving_gauge_boundary_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
    )
    assert isinstance(completed, AuthenticatedLfm2MovingGaugeBoundaryPhase)
    assert completed.receipt["execution"] == {
        "parent_model_objects_constructed": 0,
        "child_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "child_trace_recaptures": 0,
        "model_mutations": 0,
        "backward_calls": 0,
        "autograd_grad_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    assert completed.receipt["tensor_bytes_authenticated_lazily_on_field_access"]
    assert calls == {"provider": 2, "parent_session": 2, "child_session": 1}
    assert not (completed.entry.payload_path / "shards" / "orphan.safetensors").exists()

    fit = completed.boundary_pair(0, "pre", split="fit")
    assert fit.parent_rows.shape == (3, 3)
    assert fit.child_rows.shape == (3, 2)
    assert fit.parent_rows.dtype == torch.bfloat16
    assert fit.child_rows.dtype == torch.float32
    assert fit.record_ids == ("record-0",) * 3
    assert fit.weights.dtype == torch.float64
    assert float(fit.weights.sum()) == pytest.approx(1.0)
    row = completed.row_index["rows"][0]
    assert row["bridge_split"] == "selection"
    assert row["basis_split"] == "fit"
    assert row["parent_input_sha256"] != row["child_input_sha256"]

    writers = completed.focused_writer_outputs(0, split="fit")
    assert torch.equal(
        writers.child_combined_writer_output,
        writers.child_operator_output + writers.child_mlp_down,
    )
    input_evidence = completed.build_boundary_evidence(
        layer_index=0,
        endpoint="pre",
        boundary_id="layer.000.residual.pre",
        interface_role="input",
        interface_locked=True,
        base_encoder=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64
        ),
    )
    terminal_evidence = completed.build_boundary_evidence(
        layer_index=1,
        endpoint="post_layer",
        boundary_id="stack.terminal.residual",
        interface_role="terminal",
        interface_locked=True,
        base_encoder=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64
        ),
    )
    assert input_evidence.verify_authentication()
    assert terminal_evidence.verify_authentication()
    assert input_evidence.fit_parent_rows.dtype == torch.float64
    assert input_evidence.fit_child_rows.dtype == torch.float64
    assert input_evidence.row_alignment_sha256 == completed.row_alignment_sha256
    with pytest.raises(TypeError):
        completed.row_index["rows"][0]["basis_split"] = "heldout"
    with pytest.raises(TypeError):
        completed.receipt["ordered_checkpoints"][0]["row_start"] = 999
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="role, layer, endpoint",
    ):
        completed.build_boundary_evidence(
            layer_index=0,
            endpoint="pre",
            boundary_id="stack.terminal.residual",
            interface_role="terminal",
            interface_locked=True,
            base_encoder=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64
            ),
        )

    def forbidden_provider():
        raise AssertionError("phase hit touched its source provider")

    hit = load_or_extract_lfm2_moving_gauge_boundary_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=tmp_path / "unused-checkpoints",
        source_provider=forbidden_provider,
    )
    assert isinstance(hit, AuthenticatedLfm2MovingGaugeBoundaryPhase)
    assert hit.receipt_sha256 == completed.receipt_sha256


def test_complete_phase_rejects_wrong_hmac_authority(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "phase-cache")
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"b" * 32,
    )
    completed = load_or_extract_lfm2_moving_gauge_boundary_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=tmp_path / "checkpoints",
        source_provider=provider,
    )
    assert isinstance(completed, AuthenticatedLfm2MovingGaugeBoundaryPhase)
    wrong = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"c" * 32,
    )
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="HMAC|anchor|checkpoint",
    ):
        load_existing_lfm2_moving_gauge_boundary_phase(
            cache,
            request,
            authority=wrong,
        )


def test_resumable_prefix_rejects_checkpoint_hmac_tamper(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "phase-cache")
    checkpoint_root = tmp_path / "checkpoints"
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"d" * 32,
    )
    partial = load_or_extract_lfm2_moving_gauge_boundary_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
        maximum_new_shards=1,
    )
    assert isinstance(partial, Lfm2MovingGaugeBoundaryExtractionProgress)
    checkpoint = (
        checkpoint_root
        / request.phase_key.phase_key_sha256
        / "checkpoints"
        / "checkpoint-0000.json"
    )
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    value["checkpoint_hmac_sha256"] = "0" * 64
    checkpoint.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="HMAC",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=checkpoint_root,
            source_provider=provider,
        )


def test_request_phase_key_binds_route_topology_and_storage_contract() -> None:
    provenance = _provenance()
    first = _request(provenance)
    changed = dict(provenance)
    changed["route_runtime"] = _sha("different-route-runtime")
    second = _request(changed)
    assert first.request_sha256 != second.request_sha256
    assert first.phase_key.phase_key_sha256 != second.phase_key.phase_key_sha256
    assert first.as_dict()["numeric_token_id_equality_required"] is False
    assert first.as_dict()["basis_split_is_chart_authority"] is True


def test_request_rejects_alternate_basis_split_under_same_phase_key(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    sources = provider()
    report = dict(sources.basis_phase.report)
    cache_report = dict(report["cache"])
    cache_report["record_horizons"] = [
        {**dict(row), "split": ("heldout", "fit")[index]}
        for index, row in enumerate(cache_report["record_horizons"])
    ]
    report["cache"] = cache_report
    alternate = replace(
        sources,
        basis_phase=replace(
            sources.basis_phase,
            report=MappingProxyType(report),
        ),
    )
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"e" * 32,
    )
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="authority",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            DynamicFoldPhaseCache(tmp_path / "phase-cache"),
            request,
            authority=authority,
            checkpoint_root=tmp_path / "checkpoints",
            source_provider=lambda: alternate,
        )


def test_resealed_alternate_basis_report_and_matching_request_fail_hmac(
    tmp_path,
) -> None:
    request, provider, _calls = _case(tmp_path)
    sources = provider()
    report = dict(sources.basis_phase.report)
    cache_report = dict(report["cache"])
    cache_report["record_horizons"] = [
        {**dict(row), "split": ("heldout", "fit")[index]}
        for index, row in enumerate(cache_report["record_horizons"])
    ]
    report_body = {"cache": cache_report}
    report = {**report_body, "result_sha256": canonical_sha256(report_body)}
    receipt = dict(sources.basis_phase.receipt)
    receipt["basis_report"] = report
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(receipt_body)
    anchor = dict(sources.basis_phase.source_anchor)
    anchor["receipt_sha256"] = receipt["receipt_sha256"]
    anchor["basis_result_sha256"] = report["result_sha256"]
    # The attacker can recompute every public SHA but not the source HMAC.
    alternate_sources = replace(
        sources,
        basis_phase=replace(
            sources.basis_phase,
            receipt=MappingProxyType(receipt),
            source_anchor=MappingProxyType(anchor),
            report=MappingProxyType(report),
        ),
    )
    alternate_split = {0: "heldout", 1: "fit"}
    alternate_request = replace(
        request,
        basis_split_sha256=fingerprint_lfm2_moving_gauge_basis_split(
            alternate_split,
            basis_phase_key_sha256=request.provenance_sha256s["basis_phase_key"],
            basis_completion_sha256=request.provenance_sha256s["basis_completion"],
            basis_map_sha256=request.provenance_sha256s["basis_map"],
        ),
        request_sha256="",
    )
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"e2" * 16,
    )
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="authority",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            DynamicFoldPhaseCache(tmp_path / "phase-cache"),
            alternate_request,
            authority=authority,
            checkpoint_root=tmp_path / "checkpoints",
            source_provider=lambda: alternate_sources,
        )


def test_unverified_session_never_admits_a_resumable_checkpoint(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    valid_sources = provider()
    rejecting_sources = replace(
        valid_sources,
        parent_session_factory=lambda: _RejectingSession(
            valid_sources.parent_session_factory()
        ),
    )
    cache = DynamicFoldPhaseCache(tmp_path / "phase-cache")
    checkpoint_root = tmp_path / "checkpoints"
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"f" * 32,
    )
    with pytest.raises(RuntimeError, match="verification failure"):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=checkpoint_root,
            source_provider=lambda: rejecting_sources,
            maximum_new_shards=1,
        )
    work_root = checkpoint_root / request.phase_key.phase_key_sha256
    assert not (work_root / "checkpoints" / "checkpoint-0000.json").exists()

    completed = load_or_extract_lfm2_moving_gauge_boundary_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
    )
    assert isinstance(completed, AuthenticatedLfm2MovingGaugeBoundaryPhase)
    assert completed.verify_authentication()


def test_parameter_counts_must_match_authenticated_v3_budget(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    forged = replace(
        request,
        parent_parameter_count=1,
        child_parameter_count=1,
        request_sha256="",
    )
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=forged.checkpoint_hmac_key_id,
        secret=b"g" * 32,
    )
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="budget authority",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            DynamicFoldPhaseCache(tmp_path / "phase-cache"),
            forged,
            authority=authority,
            checkpoint_root=tmp_path / "checkpoints",
            source_provider=provider,
        )


def test_self_resealed_fake_v3_counts_cannot_override_hmac_plan(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    sources = provider()
    provenance = dict(request.provenance_sha256s)
    forged_v3_request, forged_v3_ledger = _v3_documents(
        provenance,
        parent_parameter_count=4,
        dense_parameter_count=1,
        continuous_parameter_count=0,
        child_parameter_count=1,
    )
    provenance["v3_request"] = forged_v3_request["request_sha256"]
    provenance["v3_ledger"] = forged_v3_ledger["ledger_sha256"]
    forged_request = _request(
        provenance,
        parent_parameter_count=4,
        child_parameter_count=1,
    )
    forged_sources = replace(
        sources,
        v3_first_divergence_request=forged_v3_request,
        v3_first_divergence_ledger=forged_v3_ledger,
    )
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=forged_request.checkpoint_hmac_key_id,
        secret=b"g2" * 16,
    )
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="budget authority",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            DynamicFoldPhaseCache(tmp_path / "phase-cache"),
            forged_request,
            authority=authority,
            checkpoint_root=tmp_path / "checkpoints",
            source_provider=lambda: forged_sources,
        )


def test_changed_counted_plan_cannot_override_hmac_basis_dependency(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    sources = provider()
    changed_plan = _counted_plan(
        dict(request.provenance_sha256s),
        input_elements=1,
        output_elements=4,
    )
    changed_sources = replace(
        sources,
        counted_child_state_plan=changed_plan,
    )
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"g3" * 16,
    )
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="authority",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            DynamicFoldPhaseCache(tmp_path / "phase-cache"),
            request,
            authority=authority,
            checkpoint_root=tmp_path / "checkpoints",
            source_provider=lambda: changed_sources,
        )


def test_resealed_fake_basis_completion_fails_phase_cache_reauthentication(
    tmp_path,
) -> None:
    request, provider, _calls = _case(tmp_path)
    sources = provider()
    provenance = dict(request.provenance_sha256s)
    provenance["basis_completion"] = _sha("forged-basis-completion")
    forged_v3_request, forged_v3_ledger = _v3_documents(provenance)
    provenance["v3_request"] = forged_v3_request["request_sha256"]
    provenance["v3_ledger"] = forged_v3_ledger["ledger_sha256"]
    forged_request = _request(provenance)
    forged_sources = replace(
        sources,
        basis_phase=replace(
            sources.basis_phase,
            completion_sha256=provenance["basis_completion"],
        ),
        v3_first_divergence_request=forged_v3_request,
        v3_first_divergence_ledger=forged_v3_ledger,
    )
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=forged_request.checkpoint_hmac_key_id,
        secret=b"g4" * 16,
    )
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="authority",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            DynamicFoldPhaseCache(tmp_path / "boundary-phase-cache"),
            forged_request,
            authority=authority,
            checkpoint_root=tmp_path / "checkpoints",
            source_provider=lambda: forged_sources,
        )


def test_tampered_request_is_rejected_before_checkpoint_io(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    object.__setattr__(request, "implementation_sha256", "0" * 64)
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"h" * 32,
    )
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="request authentication",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            DynamicFoldPhaseCache(tmp_path / "phase-cache"),
            request,
            authority=authority,
            checkpoint_root=tmp_path / "checkpoints",
            source_provider=provider,
        )
    assert not (tmp_path / "checkpoints").exists()


def test_selected_tensors_must_match_sealed_reference_descriptors(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    sources = provider()
    poisoned = replace(
        sources,
        parent_session_factory=lambda: _PoisoningSession(
            sources.parent_session_factory()
        ),
    )
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"i" * 32,
    )
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="sealed descriptor",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            DynamicFoldPhaseCache(tmp_path / "phase-cache"),
            request,
            authority=authority,
            checkpoint_root=tmp_path / "checkpoints",
            source_provider=lambda: poisoned,
            maximum_new_shards=1,
        )


@pytest.mark.parametrize("variant", ("false-claims", "extra", "descriptor-extra"))
def test_resealed_checkpoint_overclaims_are_rejected(tmp_path, variant: str) -> None:
    request, provider, _calls = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "phase-cache")
    checkpoint_root = tmp_path / "checkpoints"
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"j" * 32,
    )
    partial = load_or_extract_lfm2_moving_gauge_boundary_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
        maximum_new_shards=1,
    )
    assert isinstance(partial, Lfm2MovingGaugeBoundaryExtractionProgress)
    checkpoint = (
        checkpoint_root
        / request.phase_key.phase_key_sha256
        / "checkpoints"
        / "checkpoint-0000.json"
    )
    body = json.loads(checkpoint.read_text(encoding="utf-8"))
    body.pop("checkpoint_hmac_sha256")
    body.pop("checkpoint_sha256")
    if variant == "false-claims":
        body["source_cache_key"] = "0" * 64
        body["source_selective_session_sha256"] = "0" * 64
        body["source_storage_dtype_preserved"] = False
        body["unselected_tensor_payloads_loaded"] = True
    elif variant == "extra":
        body["unexpected_overclaim"] = True
    else:
        descriptor = next(iter(body["shard"]["tensor_descriptors"].values()))
        descriptor["unexpected_overclaim"] = True
        body["shard"]["tensor_inventory_sha256"] = canonical_sha256(
            body["shard"]["tensor_descriptors"]
        )
    sealed = {**body, "checkpoint_sha256": canonical_sha256(body)}
    value = {
        **sealed,
        "checkpoint_hmac_sha256": authority.sign_checkpoint(sealed),
    }
    checkpoint.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(Lfm2MovingGaugeBoundaryPhaseAuthenticationError):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=checkpoint_root,
            source_provider=provider,
        )


def test_resealed_source_record_must_match_aligned_checkpoint_rows(tmp_path) -> None:
    request, provider, _calls = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "phase-cache")
    checkpoint_root = tmp_path / "checkpoints"
    authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"k" * 32,
    )
    partial = load_or_extract_lfm2_moving_gauge_boundary_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
        maximum_new_shards=1,
    )
    assert isinstance(partial, Lfm2MovingGaugeBoundaryExtractionProgress)
    checkpoint = (
        checkpoint_root
        / request.phase_key.phase_key_sha256
        / "checkpoints"
        / "checkpoint-0000.json"
    )
    body = json.loads(checkpoint.read_text(encoding="utf-8"))
    body.pop("checkpoint_hmac_sha256")
    body.pop("checkpoint_sha256")
    body["source_records"][0]["input_sha256"] = _sha("self-resealed-input-lie")
    sealed = {**body, "checkpoint_sha256": canonical_sha256(body)}
    resealed = {
        **sealed,
        "checkpoint_hmac_sha256": authority.sign_checkpoint(sealed),
    }
    checkpoint.write_bytes(canonical_json_bytes(resealed) + b"\n")
    with pytest.raises(
        Lfm2MovingGaugeBoundaryPhaseAuthenticationError,
        match="source record",
    ):
        load_or_extract_lfm2_moving_gauge_boundary_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=checkpoint_root,
            source_provider=provider,
        )
