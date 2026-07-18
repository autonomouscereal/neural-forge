from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import neural_forge.hf_lfm2_cpcf_resume_artifact as resume_module
from neural_forge.dynamic_circuit_ir import (
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    COMPLETED_LAYERS_SCHEMA,
    Lfm2CpcfArtifactReference,
    Lfm2CpcfBoundaryBindings,
    Lfm2CpcfCheckpointBindings,
    Lfm2CpcfFailureBindings,
    Lfm2CpcfResumeArtifactConflictError,
    Lfm2CpcfResumeArtifactIntegrityError,
    Lfm2CpcfResumeArtifactStore,
    build_lfm2_cpcf_completed_layers_document,
)

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _boundary_bindings(*, phase_id: str = "fit") -> Lfm2CpcfBoundaryBindings:
    return Lfm2CpcfBoundaryBindings(
        parent_sha256=_digest("parent"),
        cache_sha256=_digest("cache"),
        evidence_sha256=_digest("evidence"),
        phase_id=phase_id,
        trace_sha256=_digest("trace"),
        bridge_sha256=_digest("bridge"),
        tokenizer_sha256=_digest("tokenizer"),
        source_table_sha256=_digest("source-table"),
        blank_sha256=_digest("blank"),
        target_sha256=_digest("target"),
        boundary_policy_sha256=_digest("boundary-policy"),
        solver_schema_sha256=_digest("solver-schema"),
    )


def _prefix_model(*, layer_count: int = 4):
    result = {
        "model.embed.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "model.rotary.inv_freq": torch.tensor([1.0, 0.5], dtype=torch.float32),
    }
    for index in range(layer_count):
        result[f"model.layers.{index}.weight"] = torch.eye(3) * (index + 1)
    return result


def _target_topology(prefix_model, *, layer_count: int = 4):
    tensor_metadata = {}
    for name, tensor in sorted(prefix_model.items()):
        is_buffer = name == "model.rotary.inv_freq"
        tensor_metadata[name] = {
            "role": "buffer" if is_buffer else "parameter",
            "persistent": False if is_buffer else True,
            "shape": [int(size) for size in tensor.shape],
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "stride": [int(size) for size in tensor.stride()],
            "storage_offset": int(tensor.storage_offset()),
            "original_device_type": tensor.device.type,
            "original_device_index": tensor.device.index,
        }
    groups = [[name] for name in sorted(prefix_model)]
    snapshot_metadata = {
        "schema": "neural-forge-model-tensor-snapshot-metadata-v1",
        "tensor_metadata": tensor_metadata,
        "object_alias_groups": groups,
        "storage_alias_groups": groups,
        "all_named_parameters_and_buffers": True,
        "nonpersistent_buffers_included": True,
    }
    return resume_module.build_lfm2_cpcf_target_topology_snapshot(
        prefix_model=prefix_model,
        layer_count=layer_count,
        snapshot_metadata=snapshot_metadata,
    )


def _incoming_bank():
    return {
        "fit.prefill.residual": torch.ones(1, 2, 3),
        "selection.decode.residual": torch.zeros(1, 1, 3),
    }


def _incoming_metadata(incoming, *, layer_index: int):
    return resume_module.build_lfm2_cpcf_incoming_state_bank_metadata(
        boundary="pre_layer",
        layer_index=layer_index,
        ordered_phases=[
            {
                "tensor_name": "fit.prefill.residual",
                "identity": [0, "prefill"],
                "phase_sha256": _digest("phase-prefill"),
            },
            {
                "tensor_name": "selection.decode.residual",
                "identity": [1, "decode.000"],
                "phase_sha256": _digest("phase-decode"),
            },
        ],
        incoming_state_bank=incoming,
    )


def _layer_receipts(*, count: int, final_bank_sha256: str):
    receipts = []
    previous = _digest("seed-bank")
    for index in range(count):
        next_bank = (
            final_bank_sha256 if index == count - 1 else _digest(f"bank-{index + 1}")
        )
        body = {
            "schema": "neural-forge-lfm2-cpcf-layer-compiler-v1",
            "layer_index": index,
            "operator_kind": "short_conv" if index % 2 == 0 else "full_attention",
            "source_width": 256,
            "target_width": 128,
            "input_bank_sha256": previous,
            "next_bank_sha256": next_bank,
            "boundary_charts_sha256": _digest(f"charts-{index}"),
            "preinstall_replay_sha256": _digest(f"preinstall-{index}"),
            "operator_replay_sha256": _digest(f"operator-replay-{index}"),
            "final_replay_sha256": _digest(f"final-replay-{index}"),
            "operator_artifact": {"schema": "operator-v1"},
            "mlp_artifact": {"schema": "mlp-v1"},
            "operator_install_report": {"installed": True},
            "mlp_install_report": {"installed": True},
            "split_metrics": {"fit": {}, "validation": {}},
            "split_phase_counts": {"fit": 1, "validation": 1},
            "solver_evidence_policy": {"fit_only": True},
            "actual_child_state_policy": {"actual": True},
            "audit": {"source_unchanged": True},
            "execution": {
                "closed_form_compilers": True,
                "parent_model_forward_calls": 0,
                "target_model_forward_calls": 0,
                "target_layer_forward_calls": 0,
                "generation_calls": 0,
                "cache_generation_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "training_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "fallback_paths": 0,
            },
        }
        receipts.append({**body, "result_sha256": canonical_sha256(body)})
        previous = next_bank
    return receipts


def _checkpoint_bindings(
    boundary,
    *,
    next_layer_index: int,
    receipts,
    installed_parameter_paths,
    topology,
    incoming_metadata,
) -> Lfm2CpcfCheckpointBindings:
    result = resume_module.build_lfm2_cpcf_checkpoint_bindings(
        boundary=boundary.reference,
        completed_layer_receipts=receipts,
        installed_parameter_paths=installed_parameter_paths,
        target_topology_snapshot=topology,
        incoming_state_bank_metadata=incoming_metadata,
        full_policy_sha256=_digest("full-policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
    )
    assert result.next_layer_index == next_layer_index
    return result


def _layer_failure_report(
    *,
    layer_index: int = 1,
    operator_kind: str = "short_conv",
    stage: str = "operator_compile",
    input_bank_sha256: str | None = None,
    nested_report=None,
):
    target_state = _digest("target-state")
    target_inventory = _digest("target-inventory")
    source_state = _digest("source-state")
    source_inventory = _digest("source-inventory")
    evidence = _digest("evidence-tree")
    input_tensor = _digest("input-tensor")
    charts = _digest("charts")
    charts_payload = _digest("charts-payload")
    proof_body = {
        "target_state_sha256_before": target_state,
        "target_state_sha256_after": target_state,
        "target_full_inventory_sha256_before": target_inventory,
        "target_full_inventory_sha256_after": target_inventory,
        "target_tensor_bytes_restored": True,
        "target_tensor_identities_preserved": True,
        "target_tensor_aliases_preserved": True,
        "target_tensor_shapes_preserved": True,
        "target_tensor_dtypes_preserved": True,
        "target_all_named_parameters_and_buffers_restored": True,
        "target_parameter_and_buffer_registries_restored": True,
        "target_nonpersistent_buffer_membership_restored": True,
        "target_parameter_gradient_identity_and_bytes_restored": True,
        "target_parameter_gradient_storage_aliases_restored": True,
        "target_module_modes_restored": True,
        "source_state_sha256_before": source_state,
        "source_state_sha256_after": source_state,
        "source_full_inventory_sha256_before": source_inventory,
        "source_full_inventory_sha256_after": source_inventory,
        "source_state_preserved": True,
        "source_full_inventory_preserved": True,
        "evidence_sha256_before": evidence,
        "evidence_sha256_after": evidence,
        "evidence_preserved": True,
        "input_tensor_sha256_before": input_tensor,
        "input_tensor_sha256_after": input_tensor,
        "input_tensors_preserved": True,
        "boundary_charts_sha256_before": charts,
        "boundary_charts_sha256_after": charts,
        "boundary_charts_payload_sha256_before": charts_payload,
        "boundary_charts_payload_sha256_after": charts_payload,
        "boundary_charts_preserved": True,
        "exact_layer_rollback_proven": True,
    }
    proof = {**proof_body, "proof_sha256": canonical_sha256(proof_body)}
    if nested_report is None:
        nested_sha = None
        nested_bytes_sha = None
        nested_preserved = False
    else:
        nested_sha = nested_report["report_sha256"]
        nested_bytes_sha = hashlib.sha256(
            canonical_json_bytes(nested_report)
        ).hexdigest()
        nested_preserved = True
    body = {
        "schema": "neural-forge-lfm2-cpcf-layer-failure-v3",
        "stage": stage,
        "layer_index": layer_index,
        "operator_kind": operator_kind,
        "input_bank_sha256": (
            _digest("input-bank") if input_bank_sha256 is None else input_bank_sha256
        ),
        "input_tensor_sha256": input_tensor,
        "boundary_charts_sha256": charts,
        "boundary_charts_payload_sha256": charts_payload,
        "cause": {"module": "tests", "type": "Failure", "message": "failed"},
        "nested_short_conv_validation_report": nested_report,
        "nested_short_conv_validation_report_sha256": nested_sha,
        "nested_short_conv_validation_report_canonical_bytes_sha256": (
            nested_bytes_sha
        ),
        "nested_short_conv_validation_report_preserved_byte_for_byte": (
            nested_preserved
        ),
        "rollback_proof": proof,
        "layer_result_issued": False,
        "promotion_claimed": False,
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def _failure_bindings(
    *,
    report=None,
    boundary=None,
    checkpoint=None,
) -> Lfm2CpcfFailureBindings:
    report = _layer_failure_report() if report is None else report
    proof = report["rollback_proof"]
    return Lfm2CpcfFailureBindings(
        invocation_sha256=_digest("invocation"),
        layer_index=report["layer_index"],
        operator_id=report["operator_kind"],
        stage_id=report["stage"],
        guard_proof_sha256=report["boundary_charts_payload_sha256"],
        parent_proof_sha256=proof["source_full_inventory_sha256_before"],
        rollback_proof_sha256=proof["proof_sha256"],
        boundary=None if boundary is None else boundary.reference,
        checkpoint=None if checkpoint is None else checkpoint.reference,
    )


def _prefix_snapshot_receipt():
    tensor_name = "model.layers.0.weight"
    tensor_inventory = {
        tensor_name: {
            "shape": [3, 3],
            "dtype": "float32",
            "tensor_sha256": _digest("failed-prefix-tensor"),
            "role": "parameter",
            "persistent": True,
            "original_stride": [3, 1],
            "original_storage_offset": 0,
            "original_device_type": "cuda",
            "original_device_index": 0,
        }
    }
    body = {
        "state_fingerprint": _digest("failed-prefix-state"),
        "device": "cpu",
        "contiguous": True,
        "tensor_count": 1,
        "tensor_inventory": tensor_inventory,
        "object_alias_groups": [[tensor_name]],
        "storage_alias_groups": [[tensor_name]],
        "all_named_parameters_and_buffers": True,
        "nonpersistent_buffers_included": True,
        "snapshot_role": ("failed_transaction_prefix_before_whole_model_rollback"),
    }
    return {**body, "snapshot_sha256": canonical_sha256(body)}


def _model_failure_report(
    *,
    completed_layer_receipts,
    current_incoming_bank_sha256: str,
    active_layer_index: int,
    active_operator_kind: str = "short_conv",
):
    source_state = _digest("model-source-state")
    source_inventory = _digest("model-source-inventory")
    target_state = _digest("model-target-state")
    target_inventory = _digest("model-target-inventory")
    proof_body = {
        "source_state_fingerprint_before": source_state,
        "source_state_fingerprint_after": source_state,
        "source_full_inventory_sha256_before": source_inventory,
        "source_full_inventory_sha256_after": source_inventory,
        "source_state_preserved": True,
        "source_full_inventory_preserved": True,
        "target_state_fingerprint_before": target_state,
        "target_state_fingerprint_after": target_state,
        "target_full_inventory_sha256_before": target_inventory,
        "target_full_inventory_sha256_after": target_inventory,
        "target_tensor_bytes_restored": True,
        "target_tensor_identities_preserved": True,
        "target_tensor_aliases_preserved": True,
        "target_all_named_parameters_and_buffers_restored": True,
        "target_parameter_and_buffer_registries_restored": True,
        "target_nonpersistent_buffers_restored": True,
        "target_parameter_gradient_identity_and_bytes_restored": True,
        "target_parameter_gradient_storage_aliases_restored": True,
        "target_module_modes_restored": True,
        "exact_whole_model_rollback_proven": True,
    }
    proof = {**proof_body, "proof_sha256": canonical_sha256(proof_body)}
    nested = _layer_failure_report(
        layer_index=active_layer_index,
        operator_kind=active_operator_kind,
        input_bank_sha256=current_incoming_bank_sha256,
    )
    nested_bytes_sha = hashlib.sha256(canonical_json_bytes(nested)).hexdigest()
    snapshot = _prefix_snapshot_receipt()
    body = {
        "schema": "neural-forge-lfm2-cpcf-model-failure-v3",
        "stage": "layer_compile_install",
        "active_layer_index": active_layer_index,
        "active_operator_kind": active_operator_kind,
        "cause": {"module": "tests", "type": "Failure", "message": "failed"},
        "boundary_bundle_sha256": _digest("boundary-bundle"),
        "current_incoming_bank_sha256": current_incoming_bank_sha256,
        "current_incoming_layer_index": active_layer_index,
        "completed_layer_count": len(completed_layer_receipts),
        "completed_layer_receipts": completed_layer_receipts,
        "failed_prefix_target_cpu_tensor_snapshot": snapshot,
        "nested_layer_failure_report": nested,
        "nested_layer_failure_report_sha256": nested["report_sha256"],
        "nested_layer_failure_report_canonical_bytes_sha256": (nested_bytes_sha),
        "nested_layer_failure_report_preserved_byte_for_byte": True,
        "rollback_proof": proof,
        "compiled_model_fold_issued": False,
        "parameter_compaction_success_receipt_issued": False,
        "promotion_claimed": False,
        "promotable_candidate": False,
        "failure_evidence_only": True,
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def _model_failure_bindings(
    *,
    report,
    boundary=None,
    checkpoint=None,
) -> Lfm2CpcfFailureBindings:
    proof = report["rollback_proof"]
    snapshot = report["failed_prefix_target_cpu_tensor_snapshot"]
    return Lfm2CpcfFailureBindings(
        invocation_sha256=_digest("model-invocation"),
        layer_index=report["active_layer_index"],
        operator_id=report["active_operator_kind"],
        stage_id=report["stage"],
        guard_proof_sha256=snapshot["snapshot_sha256"],
        parent_proof_sha256=proof["source_state_fingerprint_before"],
        rollback_proof_sha256=proof["proof_sha256"],
        boundary=None if boundary is None else boundary.reference,
        checkpoint=None if checkpoint is None else checkpoint.reference,
    )


def _publish_boundary(
    store: Lfm2CpcfResumeArtifactStore,
    *,
    reports: dict[str, object] | None = None,
    tensor_offset: float = 0.0,
):
    return store.publish_boundary(
        bindings=_boundary_bindings(),
        reports=(
            {
                "schema": "boundary-report-v1",
                "rows": 12,
                "split": "fit",
            }
            if reports is None
            else reports
        ),
        tensors={
            "design": torch.arange(12, dtype=torch.float32).reshape(4, 3)
            + tensor_offset,
            "endpoint": torch.tensor([0.25, -0.5], dtype=torch.bfloat16),
        },
    )


def _checkpoint_inputs(
    boundary,
    *,
    next_layer_index: int = 2,
    incoming_state_bank=None,
):
    prefix = _prefix_model()
    topology = _target_topology(prefix)
    incoming = (
        _incoming_bank()
        if incoming_state_bank is None
        else {name: value.clone() for name, value in incoming_state_bank.items()}
    )
    incoming_metadata = _incoming_metadata(
        incoming,
        layer_index=next_layer_index,
    )
    receipts = _layer_receipts(
        count=next_layer_index,
        final_bank_sha256=incoming_metadata["state_bank_sha256"],
    )
    installed_paths = resume_module.build_lfm2_cpcf_installed_parameter_paths(
        target_topology_snapshot=topology,
        next_layer_index=next_layer_index,
    )
    bindings = _checkpoint_bindings(
        boundary,
        next_layer_index=next_layer_index,
        receipts=receipts,
        installed_parameter_paths=installed_paths,
        topology=topology,
        incoming_metadata=incoming_metadata,
    )
    completed = build_lfm2_cpcf_completed_layers_document(
        bindings=bindings,
        completed_layer_receipts=receipts,
        installed_parameter_paths=installed_paths,
        target_topology_snapshot=topology,
        incoming_state_bank=incoming,
        incoming_state_bank_metadata=incoming_metadata,
    )
    return {
        "bindings": bindings,
        "prefix_model": prefix,
        "incoming_state_bank": incoming,
        "completed_layers": completed,
    }


def _publish_checkpoint(store, boundary, *, next_layer_index: int = 2):
    return store.publish_checkpoint(
        **_checkpoint_inputs(boundary, next_layer_index=next_layer_index)
    )


def _rewrite_manifest(path: Path, mutate) -> None:
    manifest = json.loads(path.read_text(encoding="ascii"))
    mutate(manifest)
    body = dict(manifest)
    body.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(body)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")


def test_boundary_round_trip_has_exact_non_promotable_layout(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    result = _publish_boundary(store)

    assert (
        result.path
        == (tmp_path / "resume" / "boundaries" / result.artifact_key).resolve()
    )
    assert {path.name for path in result.path.iterdir()} == {
        "manifest.json",
        "reports.json",
        "tensors.safetensors",
    }
    assert result.binding_key == _boundary_bindings().binding_key
    assert result.artifact_key != result.binding_key
    assert result.manifest["status"] == "non-promotable-intermediate"
    assert result.manifest["promotion_authorized"] is False
    assert result.manifest["accepted_artifact"] is False
    assert result.manifest["behavior_capability_proven"] is False
    assert result.manifest["training_performed"] is False
    assert result.manifest["optimizer_steps"] == 0
    assert result.reports["rows"] == 12
    assert torch.equal(
        result.tensors["design"],
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
    )
    reopened = store.open_boundary(
        result.reference,
        expected_bindings=_boundary_bindings(),
    )
    assert reopened.certificate_sha256 == result.certificate_sha256


def test_binding_key_discovers_distinct_content_addresses(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    first = _publish_boundary(store)
    reused = _publish_boundary(store)
    changed = _publish_boundary(
        store,
        reports={
            "schema": "boundary-report-v1",
            "rows": 99,
            "split": "fit",
        },
    )

    assert reused.artifact_key == first.artifact_key
    assert reused.reused_existing is True
    assert changed.artifact_key != first.artifact_key
    assert changed.binding_key == first.binding_key
    assert store.discover_boundary_references(
        first.binding_key,
        expected_bindings=_boundary_bindings(),
    ) == tuple(
        sorted(
            (first.reference, changed.reference),
            key=lambda reference: reference.artifact_key,
        )
    )


def test_boundary_serialization_is_deterministic_across_cold_stores(
    tmp_path: Path,
):
    first = _publish_boundary(Lfm2CpcfResumeArtifactStore(tmp_path / "one"))
    second = _publish_boundary(Lfm2CpcfResumeArtifactStore(tmp_path / "two"))

    assert first.artifact_key == second.artifact_key
    assert first.certificate_sha256 == second.certificate_sha256
    assert (first.path / "tensors.safetensors").read_bytes() == (
        second.path / "tensors.safetensors"
    ).read_bytes()


def test_self_consistent_boundary_replacement_fails_content_key_and_pin(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    result = _publish_boundary(store)
    report_path = result.path / "reports.json"
    replacement = (
        canonical_json_bytes(
            {
                "schema": "boundary-report-v1",
                "rows": 999,
                "split": "fit",
            }
        )
        + b"\n"
    )
    report_path.write_bytes(replacement)

    def mutate(manifest):
        for record in manifest["files"]:
            if record["path"] == "reports.json":
                record["size_bytes"] = len(replacement)
                record["sha256"] = hashlib.sha256(replacement).hexdigest()

    _rewrite_manifest(result.path / "manifest.json", mutate)
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="content key|pinned",
    ):
        store.open_boundary(result.reference)


@pytest.mark.parametrize("extra_name", ["extra.json", "nested"])
def test_boundary_rejects_any_extra_file_or_directory(
    tmp_path: Path,
    extra_name: str,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    result = _publish_boundary(store)
    extra = result.path / extra_name
    if extra_name == "nested":
        extra.mkdir()
    else:
        extra.write_text("{}\n", encoding="ascii")

    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="extra files",
    ):
        store.open_boundary(result.reference)


def test_boundary_rejects_manifest_and_tensor_tampering(tmp_path: Path):
    first_store = Lfm2CpcfResumeArtifactStore(tmp_path / "manifest-store")
    manifest_result = _publish_boundary(first_store)
    manifest_path = manifest_result.path / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="canonical JSON",
    ):
        first_store.open_boundary(manifest_result.reference)

    tensor_store = Lfm2CpcfResumeArtifactStore(tmp_path / "tensor-store")
    tensor_result = _publish_boundary(tensor_store)
    tensor_path = tensor_result.path / "tensors.safetensors"
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="manifest record",
    ):
        tensor_store.open_boundary(tensor_result.reference)


def test_rejects_traversal_and_artifact_symlink(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="SHA-256",
    ):
        Lfm2CpcfArtifactReference(
            artifact_kind="boundary",
            artifact_key="../outside",
            manifest_sha256=_digest("manifest"),
        )

    result = _publish_boundary(store)
    alias_key = _digest("alias")
    alias = result.path.parent / alias_key
    try:
        alias.symlink_to(result.path, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("Directory symlinks are unavailable on this platform.")
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="symlink or junction",
    ):
        store.open_boundary(
            Lfm2CpcfArtifactReference(
                artifact_kind="boundary",
                artifact_key=alias_key,
                manifest_sha256=result.reference.manifest_sha256,
            )
        )


def test_checkpoint_binds_boundary_prefix_bank_and_progress(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    checkpoint = _publish_checkpoint(store, boundary)

    assert {path.name for path in checkpoint.path.iterdir()} == {
        "manifest.json",
        "prefix-model.safetensors",
        "incoming-state-bank.safetensors",
        "completed-layers.json",
    }
    bindings = checkpoint.manifest["bindings"]
    assert bindings["boundary"] == boundary.reference.as_dict()
    assert bindings["next_layer_index"] == 2
    assert checkpoint.completed_layers["schema"] == COMPLETED_LAYERS_SCHEMA
    assert checkpoint.completed_layers["completed_layer_indices"] == (0, 1)
    assert checkpoint.completed_layers["incoming_state_layer_index"] == 2
    reopened = store.open_checkpoint(checkpoint.reference)
    assert torch.equal(reopened.prefix_model["model.layers.0.weight"], torch.eye(3))
    assert reopened.certificate_sha256 == checkpoint.certificate_sha256


def test_public_incoming_bank_flattener_borrows_tensors_and_reproduces_typed_seal():
    states = {
        (0, "prefill"): torch.ones(1, 2, 3),
        (1, "decode.000"): torch.zeros(1, 1, 3),
    }
    phases = (
        SimpleNamespace(
            identity=(0, "prefill"),
            phase_sha256=_digest("phase-prefill"),
        ),
        SimpleNamespace(
            identity=(1, "decode.000"),
            phase_sha256=_digest("phase-decode"),
        ),
    )
    seed_metadata = resume_module.build_lfm2_cpcf_incoming_state_bank_metadata(
        boundary="pre_layer",
        layer_index=2,
        ordered_phases=[
            {
                "tensor_name": "first",
                "identity": [0, "prefill"],
                "phase_sha256": phases[0].phase_sha256,
            },
            {
                "tensor_name": "second",
                "identity": [1, "decode.000"],
                "phase_sha256": phases[1].phase_sha256,
            },
        ],
        incoming_state_bank={
            "first": states[(0, "prefill")],
            "second": states[(1, "decode.000")],
        },
    )
    bank = SimpleNamespace(
        boundary="pre_layer",
        layer_index=2,
        phases=phases,
        states=states,
        state_bank_sha256=seed_metadata["state_bank_sha256"],
    )

    flattened, metadata = resume_module.flatten_lfm2_cpcf_incoming_state_bank(bank)

    assert metadata["state_bank_sha256"] == bank.state_bank_sha256
    assert len(flattened) == 2
    assert (
        next(iter(flattened.values())).data_ptr() == states[(0, "prefill")].data_ptr()
    )


def test_public_incoming_bank_flattener_accepts_failure_snapshot_accessors():
    states = {
        (0, "prefill"): torch.ones(1, 2, 3),
        (1, "decode.000"): torch.zeros(1, 1, 3),
    }
    identities = tuple(states)
    references = tuple(
        {
            "identity": [identity[0], identity[1]],
            "phase_sha256": _digest(f"snapshot-phase-{ordinal}"),
            "parent_evidence_reference": {
                "cache_key": _digest(f"external-parent-cache-{ordinal}")
            },
        }
        for ordinal, identity in enumerate(identities)
    )
    seed_metadata = resume_module.build_lfm2_cpcf_incoming_state_bank_metadata(
        boundary="pre_layer",
        layer_index=2,
        ordered_phases=[
            {
                "tensor_name": f"state-{ordinal}",
                "identity": [identity[0], identity[1]],
                "phase_sha256": references[ordinal]["phase_sha256"],
            }
            for ordinal, identity in enumerate(identities)
        ],
        incoming_state_bank={
            f"state-{ordinal}": states[identity]
            for ordinal, identity in enumerate(identities)
        },
    )

    class FailureSnapshot:
        boundary = "pre_layer"
        layer_index = 2
        phase_identities = identities
        phase_references = references
        state_bank_sha256 = seed_metadata["state_bank_sha256"]
        verify_calls = 0

        def verify(self):
            self.verify_calls += 1

        def state(self, identity):
            return states[identity].clone()

    snapshot = FailureSnapshot()
    flattened, metadata = resume_module.flatten_lfm2_cpcf_incoming_state_bank(snapshot)

    assert snapshot.verify_calls == 1
    assert metadata["state_bank_sha256"] == snapshot.state_bank_sha256
    assert len(flattened) == len(states)
    assert all(
        "parent_evidence" not in str(record) for record in metadata["ordered_states"]
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document.update(completed_layer_indices=[0, 2]),
        lambda document: document.update(next_layer_index=3),
        lambda document: document.update(incoming_state_layer_index=3),
        lambda document: document.update(fold_seals_sha256=_digest("wrong-fold-seals")),
        lambda document: document.update(
            install_receipt_sha256=_digest("wrong-install")
        ),
        lambda document: document.update(
            incoming_state_bank_sha256=_digest("wrong-state")
        ),
        lambda document: document.update(completed_layer_receipts=[]),
        lambda document: document["completed_layer_receipts"][0].update(
            next_bank_sha256=_digest("wrong-next-bank")
        ),
        lambda document: document.update(
            installed_parameter_paths=["model.layers.99.weight"]
        ),
        lambda document: document["incoming_state_bank_metadata"].update(layer_index=3),
    ],
)
def test_checkpoint_rejects_contradictory_progress(
    tmp_path: Path,
    mutate,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    kwargs = _checkpoint_inputs(boundary)
    mutate(kwargs["completed_layers"])
    with pytest.raises(Lfm2CpcfResumeArtifactIntegrityError):
        store.publish_checkpoint(**kwargs)


def test_checkpoint_rejects_prefix_state_that_does_not_match_topology(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    kwargs = _checkpoint_inputs(boundary)
    kwargs["prefix_model"] = {"unrelated.weight": torch.ones(1)}

    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="target topology",
    ):
        store.publish_checkpoint(**kwargs)


def test_checkpoint_rejects_missing_boundary_reference(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    elsewhere = Lfm2CpcfResumeArtifactStore(tmp_path / "elsewhere")
    missing_boundary = _publish_boundary(elsewhere)
    kwargs = _checkpoint_inputs(missing_boundary, next_layer_index=0)
    with pytest.raises(Lfm2CpcfResumeArtifactIntegrityError):
        store.publish_checkpoint(**kwargs)


def test_checkpoint_content_change_changes_key_and_identical_reuses(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    kwargs = _checkpoint_inputs(boundary)
    first = store.publish_checkpoint(**kwargs)
    reused = store.publish_checkpoint(**kwargs)
    changed_incoming = {
        name: value.clone() for name, value in kwargs["incoming_state_bank"].items()
    }
    changed_incoming["fit.prefill.residual"].zero_()
    changed = store.publish_checkpoint(
        **_checkpoint_inputs(
            boundary,
            incoming_state_bank=changed_incoming,
        )
    )

    assert reused.artifact_key == first.artifact_key
    assert reused.reused_existing is True
    assert changed.artifact_key != first.artifact_key


def test_checkpoint_key_rejects_distinct_valid_safetensors_bytes(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    checkpoint = _publish_checkpoint(store, boundary)
    tensor_path = checkpoint.path / "prefix-model.safetensors"
    payload = tensor_path.read_bytes()
    header_size = int.from_bytes(payload[:8], byteorder="little", signed=False)
    header = payload[8 : 8 + header_size]
    replacement = (
        (header_size + 8).to_bytes(8, byteorder="little", signed=False)
        + header
        + b" " * 8
        + payload[8 + header_size :]
    )
    from safetensors.torch import load

    assert set(load(replacement)) == set(checkpoint.prefix_model)
    tensor_path.write_bytes(replacement)

    manifest_path = checkpoint.path / "manifest.json"

    def mutate(manifest):
        for record in manifest["files"]:
            if record["path"] == tensor_path.name:
                record["size_bytes"] = len(replacement)
                record["sha256"] = hashlib.sha256(replacement).hexdigest()

    _rewrite_manifest(manifest_path, mutate)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    replacement_reference = Lfm2CpcfArtifactReference(
        artifact_kind="checkpoint",
        artifact_key=checkpoint.artifact_key,
        manifest_sha256=manifest["manifest_sha256"],
    )
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="content key",
    ):
        store.open_checkpoint(replacement_reference)


def test_failure_requires_existing_authenticated_lineage(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    missing = Lfm2CpcfArtifactReference(
        artifact_kind="boundary",
        artifact_key=_digest("missing"),
        manifest_sha256=_digest("missing-manifest"),
    )
    bindings = Lfm2CpcfFailureBindings(
        invocation_sha256=_digest("invocation"),
        layer_index=0,
        operator_id="short-conv",
        stage_id="fit",
        guard_proof_sha256=_digest("guard"),
        parent_proof_sha256=_digest("parent"),
        rollback_proof_sha256=_digest("rollback"),
        boundary=missing,
    )
    with pytest.raises(Lfm2CpcfResumeArtifactIntegrityError):
        store.publish_failure(bindings=bindings, report={"failed": True})


def test_failure_rejects_disagreeing_boundary_and_checkpoint(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    first = _publish_boundary(store)
    second = store.publish_boundary(
        bindings=_boundary_bindings(phase_id="selection"),
        reports={"schema": "report-v1"},
        tensors={"x": torch.ones(1)},
    )
    checkpoint = _publish_checkpoint(store, first)
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="lineage disagree",
    ):
        store.publish_failure(
            bindings=_failure_bindings(
                boundary=second,
                checkpoint=checkpoint,
            ),
            report={"failed": True},
        )


def test_failure_is_one_self_hashed_no_success_document(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    report = _layer_failure_report()
    failure = store.publish_failure(
        bindings=_failure_bindings(boundary=boundary, report=report),
        report=report,
    )

    assert {path.name for path in failure.path.iterdir()} == {"diagnostic.json"}
    assert failure.diagnostic["success_claimed"] is False
    assert failure.diagnostic["compaction_success_proven"] is False
    assert failure.diagnostic["behavior_success_proven"] is False
    assert failure.diagnostic["training_performed"] is False
    assert failure.diagnostic["optimizer_steps"] == 0
    reopened = store.open_failure(failure.artifact_key)
    assert reopened.certificate_sha256 == failure.certificate_sha256


def test_model_failure_round_trip_preserves_exact_prefix_and_nested_proofs(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    checkpoint_inputs = _checkpoint_inputs(boundary)
    checkpoint = store.publish_checkpoint(**checkpoint_inputs)
    completed = checkpoint_inputs["completed_layers"]
    current_runtime_bank_sha = completed["incoming_state_bank_metadata"][
        "state_bank_sha256"
    ]
    report = _model_failure_report(
        completed_layer_receipts=completed["completed_layer_receipts"],
        current_incoming_bank_sha256=current_runtime_bank_sha,
        active_layer_index=completed["next_layer_index"],
    )
    failure = store.publish_failure(
        bindings=_model_failure_bindings(
            report=report,
            checkpoint=checkpoint,
        ),
        report=report,
    )

    reopened = store.open_failure(failure.artifact_key)
    reopened_report = reopened.diagnostic["report"]
    assert reopened_report["schema"] == ("neural-forge-lfm2-cpcf-model-failure-v3")
    assert reopened_report["completed_layer_count"] == 2
    assert (
        reopened_report["nested_layer_failure_report"]["input_bank_sha256"]
        == current_runtime_bank_sha
    )


def test_model_failure_accepts_progress_checkpoint_publication_stage(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    checkpoint_inputs = _checkpoint_inputs(boundary)
    checkpoint = store.publish_checkpoint(**checkpoint_inputs)
    completed = checkpoint_inputs["completed_layers"]
    current_bank_sha = completed["incoming_state_bank_metadata"]["state_bank_sha256"]
    report = _model_failure_report(
        completed_layer_receipts=completed["completed_layer_receipts"],
        current_incoming_bank_sha256=current_bank_sha,
        active_layer_index=completed["next_layer_index"],
    )
    report["stage"] = "progress_checkpoint_publish"
    report["active_layer_index"] = None
    report["active_operator_kind"] = None
    report["nested_layer_failure_report"] = None
    report["nested_layer_failure_report_sha256"] = None
    report["nested_layer_failure_report_canonical_bytes_sha256"] = None
    report["nested_layer_failure_report_preserved_byte_for_byte"] = False
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = canonical_sha256(body)

    diagnostic = store.publish_failure(
        bindings=_model_failure_bindings(
            report=report,
            checkpoint=checkpoint,
        ),
        report=report,
    )

    assert diagnostic.diagnostic["report"]["stage"] == ("progress_checkpoint_publish")
    assert diagnostic.diagnostic["success_claimed"] is False


def test_model_failure_rejects_nested_layer_from_another_incoming_bank(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    checkpoint_inputs = _checkpoint_inputs(boundary)
    checkpoint = store.publish_checkpoint(**checkpoint_inputs)
    completed = checkpoint_inputs["completed_layers"]
    current_runtime_bank_sha = completed["incoming_state_bank_metadata"][
        "state_bank_sha256"
    ]
    report = _model_failure_report(
        completed_layer_receipts=completed["completed_layer_receipts"],
        current_incoming_bank_sha256=current_runtime_bank_sha,
        active_layer_index=completed["next_layer_index"],
    )
    nested = report["nested_layer_failure_report"]
    nested["input_bank_sha256"] = _digest("unrelated-incoming-bank")
    nested_body = dict(nested)
    nested_body.pop("report_sha256")
    nested["report_sha256"] = canonical_sha256(nested_body)
    report["nested_layer_failure_report_sha256"] = nested["report_sha256"]
    report["nested_layer_failure_report_canonical_bytes_sha256"] = hashlib.sha256(
        canonical_json_bytes(nested)
    ).hexdigest()
    report_body = dict(report)
    report_body.pop("report_sha256")
    report["report_sha256"] = canonical_sha256(report_body)

    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="Nested layer failure differs",
    ):
        store.publish_failure(
            bindings=_model_failure_bindings(
                report=report,
                checkpoint=checkpoint,
            ),
            report=report,
        )


def test_failure_rejects_nested_success_or_promotion_claim(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    nested_body = {
        "schema": "short-conv-diagnostic-v1",
        "success_claimed": True,
        "promotion_authorized": False,
    }
    nested = {
        **nested_body,
        "report_sha256": canonical_sha256(nested_body),
    }
    report = _layer_failure_report(nested_report=nested)

    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="success or promotion claim",
    ):
        store.publish_failure(
            bindings=_failure_bindings(boundary=boundary, report=report),
            report=report,
        )


def test_failure_identical_reuses_and_tamper_fails_closed(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    report = _layer_failure_report()
    bindings = _failure_bindings(boundary=boundary, report=report)
    first = store.publish_failure(bindings=bindings, report=report)
    reused = store.publish_failure(bindings=bindings, report=report)
    assert reused.reused_existing is True

    diagnostic_path = first.path / "diagnostic.json"
    diagnostic_path.write_bytes(diagnostic_path.read_bytes() + b" ")
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="canonical JSON",
    ):
        store.open_failure(first.artifact_key)


def test_racing_invalid_boundary_is_left_untouched_and_stage_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    marker = b"racing-owner"
    captured = {}

    def race(_source: Path, published: Path) -> None:
        captured["destination"] = published
        published.mkdir()
        (published / "owner.bin").write_bytes(marker)
        raise FileExistsError("race")

    monkeypatch.setattr(resume_module, "_atomic_rename_noreplace", race)
    with pytest.raises(
        Lfm2CpcfResumeArtifactConflictError,
        match="left untouched",
    ):
        _publish_boundary(store)

    destination = captured["destination"]
    assert (destination / "owner.bin").read_bytes() == marker
    assert not list(destination.parent.glob(".cpcf-resume-*.staging"))


def test_real_simultaneous_publishers_reuse_all_artifact_kinds(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    with ThreadPoolExecutor(max_workers=4) as executor:
        boundaries = list(
            executor.map(lambda _index: _publish_boundary(store), range(4))
        )
    assert len({item.artifact_key for item in boundaries}) == 1
    boundary = boundaries[0]

    with ThreadPoolExecutor(max_workers=4) as executor:
        checkpoints = list(
            executor.map(
                lambda _index: _publish_checkpoint(store, boundary),
                range(4),
            )
        )
    assert len({item.artifact_key for item in checkpoints}) == 1
    checkpoint = checkpoints[0]
    report = _layer_failure_report()

    with ThreadPoolExecutor(max_workers=4) as executor:
        failures = list(
            executor.map(
                lambda _index: store.publish_failure(
                    bindings=_failure_bindings(
                        report=report,
                        boundary=boundary,
                        checkpoint=checkpoint,
                    ),
                    report=report,
                ),
                range(4),
            )
        )
    assert len({item.artifact_key for item in failures}) == 1


def test_failed_stage_write_removes_only_owned_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    collection = tmp_path / "resume" / "boundaries"
    unrelated = collection / ".cpcf-resume-unrelated.staging"
    collection.mkdir(parents=True, exist_ok=True)
    unrelated.mkdir()
    (unrelated / "owner.txt").write_text("someone else", encoding="ascii")

    def fail(*_args, **_kwargs):
        raise resume_module.Lfm2CpcfResumeArtifactPublicationError("injected")

    monkeypatch.setattr(resume_module, "_save_safetensors", fail)
    with pytest.raises(
        resume_module.Lfm2CpcfResumeArtifactPublicationError,
        match="injected",
    ):
        _publish_boundary(store)

    assert (unrelated / "owner.txt").read_text(encoding="ascii") == "someone else"
    assert list(collection.glob(".cpcf-resume-*.staging")) == [unrelated]


def test_explicit_stale_cleanup_removes_only_signed_dead_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    collection = store._collection("boundaries")
    dead_stage, _identity, _token = store._stage(collection)
    foreign = collection / ".cpcf-resume-foreign.staging"
    foreign.mkdir()
    (foreign / ".owner.json").write_text("{}\n", encoding="ascii")

    monkeypatch.setattr(
        resume_module,
        "_process_identity",
        lambda _pid: ("dead", None),
    )
    result = store.cleanup_stale_stages(
        minimum_age_seconds=0,
        now_unix_ns=time.time_ns() + 1,
    )

    assert not dead_stage.exists()
    assert foreign.exists()
    assert len(result["removed"]) == 1
    assert len(result["preserved"]) == 1


def test_stale_cleanup_preserves_signed_live_stage(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    collection = store._collection("boundaries")
    stage, _identity, _token = store._stage(collection)

    result = store.cleanup_stale_stages(
        minimum_age_seconds=0,
        now_unix_ns=time.time_ns() + 1,
    )

    assert stage.exists()
    assert result["removed"] == ()
    assert len(result["preserved"]) == 1


def test_stale_cleanup_preserves_live_owner_with_unavailable_birth_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    collection = store._collection("boundaries")
    monkeypatch.setattr(
        resume_module,
        "_process_identity",
        lambda _pid: ("unknown", None),
    )
    stage, _identity, _token = store._stage(collection)
    monkeypatch.setattr(
        resume_module,
        "_process_identity",
        lambda _pid: ("live", "now-available"),
    )

    result = store.cleanup_stale_stages(
        minimum_age_seconds=0,
        now_unix_ns=time.time_ns() + 1,
    )

    assert stage.exists()
    assert result["removed_count"] == 0
    assert result["preserved_count"] == 1


def test_stale_cleanup_report_memory_is_explicitly_bounded(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    collection = store._collection("boundaries")
    for index in range(5):
        stage = collection / f".cpcf-resume-foreign-{index}.staging"
        stage.mkdir()
        (stage / ".owner.json").write_text("{}\n", encoding="ascii")

    result = store.cleanup_stale_stages(
        minimum_age_seconds=0,
        now_unix_ns=time.time_ns() + 1,
        maximum_reported_entries=2,
    )

    assert result["preserved_count"] == 5
    assert len(result["preserved"]) == 2
    assert result["report_truncated"] is True
    assert all(path.exists() for path in collection.glob("*.staging"))


def test_second_authentication_detects_mutation_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    result = _publish_boundary(store)
    original = resume_module._authenticate_manifest
    calls = 0

    def authenticate(root: Path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            reports = root / "reports.json"
            reports.write_bytes(reports.read_bytes() + b" ")
        return original(root, **kwargs)

    monkeypatch.setattr(resume_module, "_authenticate_manifest", authenticate)
    with pytest.raises(Lfm2CpcfResumeArtifactIntegrityError):
        store.open_boundary(result.reference)


def test_descriptor_identity_detects_swap_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    result = _publish_boundary(store)
    reports = result.path / "reports.json"
    backup = tmp_path / "reports.backup"
    replacement = tmp_path / "reports.replacement"
    replacement.write_bytes(reports.read_bytes())
    real_open = os.open
    triggered = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal triggered
        if Path(path) == reports and not triggered:
            triggered = True
            os.replace(reports, backup)
            os.replace(replacement, reports)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(resume_module.os, "open", racing_open)
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="changed identity",
    ):
        store.open_boundary(result.reference)


def test_store_rejects_symlink_root(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("Directory symlinks are unavailable on this platform.")
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="real directory",
    ):
        Lfm2CpcfResumeArtifactStore(alias)


def test_failure_binding_requires_resume_reference():
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="reference",
    ):
        Lfm2CpcfFailureBindings(
            invocation_sha256=_digest("invocation"),
            layer_index=0,
            operator_id="short-conv",
            stage_id="fit",
            guard_proof_sha256=_digest("guard"),
            parent_proof_sha256=_digest("parent"),
            rollback_proof_sha256=_digest("rollback"),
        )


def test_tensor_inputs_must_be_finite_and_identifiers_canonical(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="not finite",
    ):
        store.publish_boundary(
            bindings=_boundary_bindings(),
            reports={"schema": "report-v1"},
            tensors={"bad": torch.tensor([float("nan")])},
        )
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="stable identifier",
    ):
        _boundary_bindings(phase_id=" fit ")


def test_verified_json_is_frozen_and_tensor_views_are_disposable(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    first_view = boundary.tensors
    first_view["design"].zero_()
    second_view = boundary.tensors
    assert torch.count_nonzero(second_view["design"]).item() > 0
    assert not hasattr(boundary, "_tensors")
    private_view = boundary._tensor_archive.load()
    private_view["design"].zero_()
    assert torch.count_nonzero(boundary.tensors["design"]).item() > 0
    with pytest.raises(TypeError):
        boundary.manifest["status"] = "accepted"

    checkpoint = _publish_checkpoint(store, boundary)
    prefix_view = checkpoint.prefix_model
    prefix_view["model.layers.0.weight"].zero_()
    assert not hasattr(checkpoint, "_prefix_model")
    private_prefix = checkpoint._prefix_model_archive.load()
    private_prefix["model.layers.0.weight"].zero_()
    reauthenticated = store.final_reauthenticate_checkpoint(checkpoint)
    assert torch.equal(
        reauthenticated.prefix_model["model.layers.0.weight"],
        torch.eye(3),
    )


def test_exact_no_success_flag_types_are_required(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)

    def mutate(manifest):
        manifest["optimizer_steps"] = False

    _rewrite_manifest(boundary.path / "manifest.json", mutate)
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="non-promotable",
    ):
        store.open_boundary(boundary.reference)


def test_artifact_files_are_not_hardlinked_on_current_platform(
    tmp_path: Path,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    result = _publish_boundary(store)
    report = result.path / "reports.json"
    alias = tmp_path / "report-hardlink.json"
    try:
        os.link(report, alias)
    except (NotImplementedError, OSError):
        pytest.skip("Hardlinks are unavailable on this filesystem.")
    assert report.lstat().st_nlink > 1
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="unaliased regular file",
    ):
        store.open_boundary(result.reference)


def test_wrong_reference_kind_is_rejected(tmp_path: Path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    boundary = _publish_boundary(store)
    wrong = Lfm2CpcfArtifactReference(
        artifact_kind="checkpoint",
        artifact_key=boundary.artifact_key,
        manifest_sha256=boundary.reference.manifest_sha256,
    )
    with pytest.raises(
        Lfm2CpcfResumeArtifactIntegrityError,
        match="pinned boundary",
    ):
        store.open_boundary(wrong)
