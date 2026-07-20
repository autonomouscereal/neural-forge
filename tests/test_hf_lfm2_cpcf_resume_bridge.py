from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import inspect
import os
import textwrap
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

import neural_forge.hf_lfm2_cpcf_model_compiler as model_compiler
import neural_forge.hf_lfm2_cpcf_model_fold as model_fold
import neural_forge.hf_lfm2_cpcf_resume_bridge as bridge
import neural_forge.hf_lfm2_cpcf_v5_model_compiler as v5_compiler
import neural_forge.hf_lfm2_cpcf_v5_resume as v5_resume
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_closed_form_causal_circuit_fold import (
    PredictiveBoundaryMap,
)
from neural_forge.hf_cpcf_boundary_chart import CpcfBoundaryChart
from neural_forge.hf_lfm2_cpcf_boundary_bundle import (
    LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA,
    LFM2_CPCF_LAYER_CHARTS_SCHEMA,
    Lfm2CpcfBoundaryBundle,
    Lfm2CpcfLayerBoundaryCharts,
)
from neural_forge.hf_lfm2_cpcf_installability import (
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    Lfm2CpcfBoundaryBindings,
    Lfm2CpcfResumeArtifactStore,
    build_lfm2_cpcf_target_topology_snapshot,
)
from neural_forge.hf_tied_causal_interface_fold import (
    TiedCausalInterfaceFold,
)


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sealed(body):
    return {**body, "report_sha256": canonical_sha256(body)}


def _mapping(seed: int) -> PredictiveBoundaryMap:
    generator = torch.Generator().manual_seed(seed)
    encoder = torch.randn((3, 2), generator=generator, dtype=torch.float64)
    decoder = torch.randn((2, 3), generator=generator, dtype=torch.float64)
    predictors = {
        "future_output_coordinates": torch.randn(
            (2, 2),
            generator=generator,
            dtype=torch.float64,
        ),
        "next_action": torch.randn(
            (2, 1),
            generator=generator,
            dtype=torch.float64,
        ),
    }
    body = {
        "schema": "neural-forge-closed-form-causal-circuit-fold-v1",
        "component": "predictive_boundary_map",
        "source_width": 3,
        "target_width": 2,
        "encoder_sha256": bridge._predictive_tensor_sha256(encoder),
        "decoder_sha256": bridge._predictive_tensor_sha256(decoder),
    }
    return PredictiveBoundaryMap(
        encoder=encoder,
        decoder=decoder,
        feature_predictors=predictors,
        report=_sealed(body),
    )


def _chart(layer_index: int, suffix: str, seed: int) -> CpcfBoundaryChart:
    mapping = _mapping(seed)
    name = f"layer.{layer_index:03d}.residual.{suffix}"
    body = {
        "schema": "neural-forge-cpcf-boundary-chart-v1",
        "name": name,
        "source_width": 3,
        "target_width": 2,
        "predictive_mapping": mapping.as_dict(),
    }
    return CpcfBoundaryChart(
        name=name,
        mapping=mapping,
        report=_sealed(body),
    )


def _bundle() -> Lfm2CpcfBoundaryBundle:
    target_weight = torch.tensor(
        [
            [0.2, -0.3],
            [0.7, 0.1],
            [-0.5, 0.4],
            [0.6, -0.2],
        ],
        dtype=torch.float64,
    )
    producer_encoder = torch.tensor(
        [
            [0.7, -0.2],
            [0.1, 0.8],
            [-0.4, 0.3],
        ],
        dtype=torch.float64,
    )
    base_target_sha = bridge._interface_tensor_sha256(target_weight)
    base_encoder_sha = bridge._interface_tensor_sha256(producer_encoder)
    base_body = {
        "schema": "neural-forge-tied-causal-interface-v1",
        "target_weight_sha256": base_target_sha,
        "producer_encoder_sha256": base_encoder_sha,
        "source_table_sha256": _digest("source-table"),
    }
    base_report = _sealed(base_body)
    base_sha = base_report["report_sha256"]
    anchor_body = {
        "schema": "neural-forge-tied-causal-interface-anchor-v1",
        "target_weight_sha256": bridge._runtime_tensor_sha256(target_weight),
        "producer_encoder_sha256": bridge._runtime_tensor_sha256(producer_encoder),
        "base_interface_report_sha256": base_sha,
        "base_interface_lineage": {
            "report": base_report,
            "report_sha256": base_sha,
            "target_weight_sha256": base_target_sha,
            "producer_encoder_sha256": base_encoder_sha,
            "source_table_sha256": base_report["source_table_sha256"],
            "tensor_identity_verified": True,
        },
    }
    anchor_report = _sealed(anchor_body)
    interface = TiedCausalInterfaceFold(
        target_weight=target_weight,
        producer_encoder=producer_encoder,
        report=anchor_report,
    )
    post_operator = _chart(0, "post_operator", 101)
    post_layer = _chart(0, "post_layer", 102)
    layer_body = {
        "schema": LFM2_CPCF_LAYER_CHARTS_SCHEMA,
        "layer_index": 0,
        "operator_kind": "short_conv",
        "source_width": 3,
        "target_width": 2,
        "post_operator_report_sha256": post_operator.report["report_sha256"],
        "post_layer_report_sha256": post_layer.report["report_sha256"],
        "independent_boundary_coordinates": True,
    }
    layer = Lfm2CpcfLayerBoundaryCharts(
        layer_index=0,
        operator_kind="short_conv",
        post_operator=post_operator,
        post_layer=post_layer,
        layer_charts_sha256=canonical_sha256(layer_body),
    )
    split_phase = {split: _digest(f"phase-{split}") for split in bridge._SPLITS}
    split_row = {split: _digest(f"row-{split}") for split in bridge._SPLITS}
    anchor_sha = anchor_report["report_sha256"]
    bundle_body = {
        "schema": LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA,
        "source_width": 3,
        "target_width": 2,
        "vocabulary_size": 4,
        "layer_count": 1,
        "layer_operator_kinds": ["short_conv"],
        "interface_report_sha256": anchor_sha,
        "base_interface_report_sha256": base_sha,
        "causal_anchor_report_sha256": anchor_sha,
        "interface_lineage": {
            "base_interface_report": base_report,
            "causal_anchor_report": anchor_report,
            "base_interface_report_sha256": base_sha,
            "causal_anchor_report_sha256": anchor_sha,
            "active_interface_report_sha256": anchor_sha,
        },
        "layer_charts": [layer.as_dict()],
        "split_phase_sha256": split_phase,
        "split_row_sha256": split_row,
    }
    bundle_sha = canonical_sha256(bundle_body)
    return Lfm2CpcfBoundaryBundle(
        interface=interface,
        layers=(layer,),
        split_phase_sha256=split_phase,
        split_row_sha256=split_row,
        report={**bundle_body, "bundle_sha256": bundle_sha},
        bundle_sha256=bundle_sha,
    )


def _bundle_with_depth(depth: int) -> Lfm2CpcfBoundaryBundle:
    if depth < 1:
        raise ValueError("depth must be positive")
    base = _bundle()
    layers = [base.layers[0]]
    for layer_index in range(1, depth):
        post_operator = _chart(
            layer_index,
            "post_operator",
            101 + layer_index * 2,
        )
        post_layer = _chart(
            layer_index,
            "post_layer",
            102 + layer_index * 2,
        )
        layer_body = {
            "schema": LFM2_CPCF_LAYER_CHARTS_SCHEMA,
            "layer_index": layer_index,
            "operator_kind": "short_conv",
            "source_width": 3,
            "target_width": 2,
            "post_operator_report_sha256": (post_operator.report["report_sha256"]),
            "post_layer_report_sha256": post_layer.report["report_sha256"],
            "independent_boundary_coordinates": True,
        }
        layers.append(
            Lfm2CpcfLayerBoundaryCharts(
                layer_index=layer_index,
                operator_kind="short_conv",
                post_operator=post_operator,
                post_layer=post_layer,
                layer_charts_sha256=canonical_sha256(layer_body),
            )
        )
    body = dict(base.report)
    body.pop("bundle_sha256")
    body["layer_count"] = depth
    body["layer_operator_kinds"] = ["short_conv"] * depth
    body["layer_charts"] = [layer.as_dict() for layer in layers]
    bundle_sha = canonical_sha256(body)
    return Lfm2CpcfBoundaryBundle(
        interface=base.interface,
        layers=tuple(layers),
        split_phase_sha256=base.split_phase_sha256,
        split_row_sha256=base.split_row_sha256,
        report={**body, "bundle_sha256": bundle_sha},
        bundle_sha256=bundle_sha,
    )


def _bindings(flattened) -> Lfm2CpcfBoundaryBindings:
    return Lfm2CpcfBoundaryBindings(
        parent_sha256=_digest("parent"),
        cache_sha256=_digest("cache"),
        evidence_sha256=_digest("evidence"),
        phase_id="all-splits",
        trace_sha256=_digest("trace"),
        bridge_sha256=flattened.bridge_sha256,
        tokenizer_sha256=_digest("tokenizer"),
        source_table_sha256=_digest("source-table-binding"),
        blank_sha256=_digest("blank"),
        target_sha256=_digest("target"),
        boundary_policy_sha256=_digest("boundary-policy"),
        solver_schema_sha256=_digest("solver-schema"),
    )


def _phase_and_bank():
    donor_sha = _digest("donor-input")
    layer = Lfm2CpcfInstallableLayerEvidence(
        layer_index=0,
        operator_kind="short_conv",
        source_width=3,
        tensors={"operator.input": torch.ones(1, 2, 3)},
        evidence_sha256=_digest("layer-evidence"),
    )
    parent = Lfm2CpcfInstallablePhaseEvidence(
        cache_key=_digest("cache-key"),
        batch_index=0,
        input_sha256=donor_sha,
        phase="prefill",
        layers=(layer,),
        final_norm_tensors={"output.final_norm_output": torch.ones(1, 2, 3)},
        evidence_sha256=_digest("phase-evidence"),
    )
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids)
    positions = torch.tensor([[0, 1]], dtype=torch.long)
    body = {
        "schema": model_compiler.LFM2_CPCF_TARGET_PHASE_SCHEMA,
        "batch_index": 0,
        "record_id": "record-0",
        "split": "fit",
        "phase": "prefill",
        "donor_input_sha256": donor_sha,
        "target_alignment_sha256": _digest("alignment"),
        "input_ids_sha256": model_compiler._tensor_sha256(ids),
        "attention_mask_sha256": model_compiler._tensor_sha256(mask),
        "position_ids_sha256": model_compiler._tensor_sha256(positions),
        "batch_size": 1,
        "full_prefix_tokens": 2,
        "sampled_suffix_tokens": 1,
        "parent_evidence_sha256": parent.evidence_sha256,
        "numeric_token_id_equality_required": False,
        "actual_child_full_prefix_required": True,
    }
    phase = model_compiler.Lfm2CpcfTargetPhase(
        batch_index=0,
        record_id="record-0",
        split="fit",
        phase="prefill",
        donor_input_sha256=donor_sha,
        target_alignment_sha256=_digest("alignment"),
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        sampled_suffix_tokens=1,
        parent_evidence=parent,
        phase_sha256=model_compiler._sha(body),
    )
    state = torch.tensor(
        [[[0.2, -0.1], [0.5, 0.7]]],
        dtype=torch.float32,
    )
    bank = model_compiler._phase_state_bank(
        "embedding_output",
        0,
        (phase,),
        {phase.identity: state},
    )
    return phase, bank


class _Blank(torch.nn.Module):
    def __init__(
        self,
        *,
        tied: bool = True,
        dtype=torch.float32,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(4, 2, dtype=dtype)
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False, dtype=dtype) for _index in range(depth)]
        )
        self.lm_head = torch.nn.Linear(2, 4, bias=False, dtype=dtype)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.register_buffer(
            "rotary_inv_freq",
            torch.tensor([1.0, 0.5], dtype=dtype),
            persistent=False,
        )
        self.config = SimpleNamespace(model_type="lfm2")

    def zero_parameters(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()

    def get_input_embeddings(self):
        return self.model.embed_tokens


def _snapshot_parts(model):
    values, inventory = bridge._target_inventory(model)
    tensors = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in values.items()
    }
    metadata = {
        "schema": "neural-forge-model-tensor-snapshot-metadata-v1",
        "all_named_parameters_and_buffers": True,
        "nonpersistent_buffers_included": True,
        **inventory,
    }
    return tensors, metadata


def _model_failure(bundle, bank):
    prefix_model = _Blank()
    prefix_model.zero_parameters()
    with torch.no_grad():
        prefix_model.model.embed_tokens.weight.copy_(
            torch.tensor(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                    [0.7, 0.8],
                ]
            )
        )
    prefix, metadata = _snapshot_parts(prefix_model)
    snapshot_receipt = model_fold._cpu_tensor_snapshot_receipt_from_parts(
        prefix,
        metadata,
    )
    source_state = _digest("source-state")
    source_inventory = _digest("source-inventory")
    target_state = _digest("target-state")
    target_inventory = _digest("target-inventory")
    rollback_body = {
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
    rollback = {
        **rollback_body,
        "proof_sha256": canonical_sha256(rollback_body),
    }
    report_body = {
        "schema": model_fold.LFM2_CPCF_MODEL_FAILURE_SCHEMA,
        "stage": "layer_compile_install",
        "active_layer_index": 0,
        "active_operator_kind": "short_conv",
        "cause": {"module": "tests", "type": "Rejected", "message": "gate failed"},
        "boundary_bundle_sha256": bundle.bundle_sha256,
        "current_incoming_bank_sha256": bank.state_bank_sha256,
        "current_incoming_layer_index": 0,
        "completed_layer_count": 0,
        "completed_layer_receipts": [],
        "failed_prefix_target_cpu_tensor_snapshot": dict(snapshot_receipt),
        "nested_layer_failure_report": None,
        "nested_layer_failure_report_sha256": None,
        "nested_layer_failure_report_canonical_bytes_sha256": None,
        "nested_layer_failure_report_preserved_byte_for_byte": False,
        "rollback_proof": rollback,
        "compiled_model_fold_issued": False,
        "parameter_compaction_success_receipt_issued": False,
        "promotion_claimed": False,
        "promotable_candidate": False,
        "failure_evidence_only": True,
    }
    report = {**report_body, "report_sha256": canonical_sha256(report_body)}
    return model_fold.Lfm2CpcfModelFoldFailure(
        "gate failed",
        stage="layer_compile_install",
        boundary_bundle=bundle,
        current_incoming_bank=bank,
        completed_layer_receipts=(),
        prefix_target_cpu_tensor_snapshot=prefix,
        prefix_target_snapshot_metadata=metadata,
        rollback_proof=rollback,
        report=report,
    )


def _progress_layer_receipt(
    *,
    layer_index: int,
    input_bank_sha256: str,
    next_bank_sha256: str,
    charts_sha256: str,
):
    body = {
        "schema": "neural-forge-lfm2-cpcf-layer-compiler-v1",
        "layer_index": layer_index,
        "operator_kind": "short_conv",
        "source_width": 3,
        "target_width": 2,
        "input_bank_sha256": input_bank_sha256,
        "next_bank_sha256": next_bank_sha256,
        "boundary_charts_sha256": charts_sha256,
        "preinstall_replay_sha256": _digest(f"preinstall-{layer_index}"),
        "operator_replay_sha256": _digest(f"operator-{layer_index}"),
        "final_replay_sha256": _digest(f"final-{layer_index}"),
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
    return model_fold.Lfm2CpcfCompletedLayerReceipt(
        {**body, "result_sha256": canonical_sha256(body)}
    )


def _progress_event(*, next_layer_index: int, prefix_offset: float = 0.0):
    bundle = _bundle_with_depth(2)
    phase, seed_bank = _phase_and_bank()
    banks = [seed_bank]
    for layer_index in range(2):
        state = seed_bank.state(phase).clone() + float(layer_index + 1)
        banks.append(
            model_compiler._phase_state_bank(
                f"post_layer_{layer_index}",
                layer_index + 1,
                (phase,),
                {phase.identity: state},
            )
        )
    receipts = tuple(
        _progress_layer_receipt(
            layer_index=layer_index,
            input_bank_sha256=banks[layer_index].state_bank_sha256,
            next_bank_sha256=banks[layer_index + 1].state_bank_sha256,
            charts_sha256=bundle.layers[layer_index].layer_charts_sha256,
        )
        for layer_index in range(next_layer_index)
    )
    target = _Blank(depth=2)
    target.zero_parameters()
    with torch.no_grad():
        target.model.embed_tokens.weight.fill_(0.25 + prefix_offset)
        for layer_index in range(next_layer_index):
            target.model.layers[layer_index].weight.fill_(layer_index + 1.0)
    event = model_fold.Lfm2CpcfModelFoldProgressEvent.capture(
        boundary_bundle=bundle,
        incoming_state_bank=banks[next_layer_index],
        completed_layer_results=receipts,
        target_prefix_snapshot=model_fold._snapshot_model(target),
        full_policy_sha256=_digest("full-policy"),
    )
    return event, bundle, phase, banks[next_layer_index]


def test_boundary_bundle_round_trip_publishes_every_tensor_and_report(tmp_path):
    bundle = _bundle()
    flattened = bridge.flatten_lfm2_cpcf_boundary_bundle(bundle)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    published = bridge.publish_lfm2_cpcf_boundary_bundle(
        store,
        bundle,
        bindings=_bindings(flattened),
    )
    reconstructed = bridge.open_lfm2_cpcf_boundary_bundle(
        store,
        published.reference,
        expected_bindings=_bindings(flattened),
    )

    assert reconstructed.bundle_sha256 == bundle.bundle_sha256
    assert reconstructed.as_dict() == bundle.as_dict()
    second = bridge.flatten_lfm2_cpcf_boundary_bundle(reconstructed)
    assert second.reports == flattened.reports
    assert set(second.tensors) == set(flattened.tensors)
    assert len(second.tensors) == 10
    for name in second.tensors:
        assert torch.equal(second.tensors[name], flattened.tensors[name])


@pytest.mark.parametrize("mutation", ["bytes", "missing", "extra", "report"])
def test_boundary_reconstruction_rejects_tensor_and_report_tamper(mutation):
    flattened = bridge.flatten_lfm2_cpcf_boundary_bundle(_bundle())
    reports = copy.deepcopy(bridge._deep_thaw(flattened.reports))
    tensors = {name: value.clone() for name, value in flattened.tensors.items()}
    if mutation == "bytes":
        tensors["layer.00000000.post_layer.predictor.00000000"][0, 0] += 1.0
    elif mutation == "missing":
        tensors.pop("layer.00000000.post_layer.decoder")
    elif mutation == "extra":
        tensors["unreported"] = torch.ones(1, dtype=torch.float64)
    else:
        reports["layers"][0]["post_layer"]["mapping_report"]["source_width"] = 99
    with pytest.raises(bridge.Lfm2CpcfResumeBridgeError):
        bridge.reconstruct_lfm2_cpcf_boundary_bundle(
            reports=reports,
            tensors=tensors,
        )


def test_boundary_reconstruction_rejects_resealed_numeric_report_lie():
    flattened = bridge.flatten_lfm2_cpcf_boundary_bundle(_bundle())
    reports = copy.deepcopy(bridge._deep_thaw(flattened.reports))
    mapping = reports["layers"][0]["post_operator"]["mapping_report"]
    mapping["encoder_sha256"] = _digest("forged-encoder")
    mapping_body = dict(mapping)
    mapping_body.pop("report_sha256")
    mapping["report_sha256"] = canonical_sha256(mapping_body)
    reports["layers"][0]["post_operator"]["report"]["predictive_mapping"] = mapping
    chart = reports["layers"][0]["post_operator"]["report"]
    chart_body = dict(chart)
    chart_body.pop("report_sha256")
    chart["report_sha256"] = canonical_sha256(chart_body)
    bridge_body = dict(reports)
    bridge_body.pop("bridge_sha256")
    reports["bridge_sha256"] = canonical_sha256(bridge_body)
    with pytest.raises(bridge.Lfm2CpcfResumeBridgeError, match="differs"):
        bridge.reconstruct_lfm2_cpcf_boundary_bundle(
            reports=reports,
            tensors=flattened.tensors,
        )


def test_progress_checkpoint_publishes_seed_then_reuses_one_pinned_boundary(
    tmp_path,
):
    seed_event, bundle, phase, seed_bank = _progress_event(next_layer_index=0)
    bindings = _bindings(bridge.flatten_lfm2_cpcf_boundary_bundle(bundle))
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    seed = bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
        store,
        seed_event,
        boundary_bindings=bindings,
        full_policy_sha256=_digest("full-policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
        global_installed_parameter_paths=(
            "model.embed_tokens.weight",
            "lm_head.weight",
        ),
    )

    assert seed.checkpoint.completed_layers["next_layer_index"] == 0
    assert seed.event_report_sha256 == seed_event.report_sha256
    assert seed.full_policy_sha256 == seed_event.full_policy_sha256
    assert seed.compiler_fingerprint_sha256 == _digest("compiler")
    assert seed.checkpoint.manifest["accepted_artifact"] is False
    assert seed.checkpoint.manifest["promotion_authorized"] is False
    assert set(seed.checkpoint.incoming_state_bank) == {
        "state.00000000.00000000.prefill"
    }
    assert all(
        "parent" not in name and "evidence" not in name
        for name in seed.checkpoint.incoming_state_bank
    )
    reconstructed_seed = bridge.reconstruct_lfm2_cpcf_phase_state_bank(
        store,
        seed.checkpoint,
        phases=(phase,),
    )
    assert reconstructed_seed.state_bank_sha256 == seed_bank.state_bank_sha256

    duplicate = bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
        store,
        seed_event,
        boundary_bindings=bindings,
        full_policy_sha256=_digest("full-policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
        global_installed_parameter_paths=(
            "model.embed_tokens.weight",
            "lm_head.weight",
        ),
        pinned_boundary=seed.boundary,
        previous_checkpoint=seed.checkpoint,
    )
    assert duplicate.boundary.reference == seed.boundary.reference
    assert duplicate.checkpoint.reference == seed.checkpoint.reference

    layer_event, _bundle_again, _phase_again, layer_bank = _progress_event(
        next_layer_index=1
    )
    layer = bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
        store,
        layer_event,
        boundary_bindings=bindings,
        full_policy_sha256=_digest("full-policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
        global_installed_parameter_paths=(
            "model.embed_tokens.weight",
            "lm_head.weight",
        ),
        pinned_boundary=seed.boundary.reference,
        previous_checkpoint=seed.checkpoint.reference,
    )
    assert layer.boundary.reference == seed.boundary.reference
    assert layer.checkpoint.reference != seed.checkpoint.reference
    assert layer.checkpoint.completed_layers["next_layer_index"] == 1
    assert list(layer.checkpoint.completed_layers["installed_parameter_paths"]) == [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.weight",
    ]
    reconstructed_layer = bridge.reconstruct_lfm2_cpcf_phase_state_bank(
        store,
        layer.checkpoint,
        phases=(phase,),
    )
    assert reconstructed_layer.state_bank_sha256 == layer_bank.state_bank_sha256
    assert len(list((tmp_path / "resume" / "boundaries").iterdir())) == 1


def test_progress_checkpoint_rejects_policy_compiler_and_receipt_lineage_drift(
    tmp_path,
):
    seed_event, bundle, _phase, _bank = _progress_event(next_layer_index=0)
    bindings = _bindings(bridge.flatten_lfm2_cpcf_boundary_bundle(bundle))
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    with pytest.raises(
        bridge.Lfm2CpcfResumeBridgeError,
        match="policy",
    ):
        bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
            store,
            seed_event,
            boundary_bindings=bindings,
            full_policy_sha256=_digest("wrong-policy"),
            compiler_fingerprint_sha256=_digest("compiler"),
        )
    seed = bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
        store,
        seed_event,
        boundary_bindings=bindings,
        full_policy_sha256=_digest("full-policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
    )
    divergent_seed, _same_bundle, _same_phase, _same_bank = _progress_event(
        next_layer_index=0,
        prefix_offset=1.0,
    )
    with pytest.raises(
        bridge.Lfm2CpcfResumeBridgeError,
        match="Duplicate progress index",
    ):
        bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
            store,
            divergent_seed,
            boundary_bindings=bindings,
            full_policy_sha256=_digest("full-policy"),
            compiler_fingerprint_sha256=_digest("compiler"),
            pinned_boundary=seed.boundary,
            previous_checkpoint=seed.checkpoint,
        )
    with pytest.raises(
        bridge.Lfm2CpcfResumeBridgeError,
        match="global install inventory",
    ):
        bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
            store,
            seed_event,
            boundary_bindings=bindings,
            full_policy_sha256=_digest("full-policy"),
            compiler_fingerprint_sha256=_digest("compiler"),
            global_installed_parameter_paths=("model.embed_tokens.weight",),
            pinned_boundary=seed.boundary,
            previous_checkpoint=seed.checkpoint,
        )
    layer_event, _bundle_again, _phase_again, _layer_bank = _progress_event(
        next_layer_index=1
    )
    with pytest.raises(
        bridge.Lfm2CpcfResumeBridgeError,
        match="compiler",
    ):
        bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
            store,
            layer_event,
            boundary_bindings=bindings,
            full_policy_sha256=_digest("full-policy"),
            compiler_fingerprint_sha256=_digest("changed-compiler"),
            pinned_boundary=seed.boundary,
            previous_checkpoint=seed.checkpoint,
        )

    forged_receipts = list(layer_event.completed_layer_receipts)
    forged = copy.deepcopy(forged_receipts[0])
    forged["input_bank_sha256"] = _digest("unrelated-seed-bank")
    body = dict(forged)
    body.pop("result_sha256")
    forged["result_sha256"] = canonical_sha256(body)
    object.__setattr__(
        layer_event,
        "_completed_layer_receipt_bytes",
        (model_fold._canonical_json(forged),),
    )
    with pytest.raises(
        bridge.Lfm2CpcfResumeBridgeError,
        match="verification",
    ):
        bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
            store,
            layer_event,
            boundary_bindings=bindings,
            full_policy_sha256=_digest("full-policy"),
            compiler_fingerprint_sha256=_digest("compiler"),
            pinned_boundary=seed.boundary,
            previous_checkpoint=seed.checkpoint,
        )


def test_progress_checkpoint_rejects_bundle_tamper_and_final_nonresumable(
    tmp_path,
):
    layer_event, _layer_bundle, _phase, _bank = _progress_event(next_layer_index=1)
    wrong_bindings = _bindings(bridge.flatten_lfm2_cpcf_boundary_bundle(_bundle()))
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    with pytest.raises(
        bridge.Lfm2CpcfResumeBridgeError,
        match="boundary bindings",
    ):
        bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
            store,
            layer_event,
            boundary_bindings=wrong_bindings,
            full_policy_sha256=_digest("full-policy"),
            compiler_fingerprint_sha256=_digest("compiler"),
        )

    final_event, final_bundle, _final_phase, _final_bank = _progress_event(
        next_layer_index=2
    )
    final_bindings = _bindings(bridge.flatten_lfm2_cpcf_boundary_bundle(final_bundle))
    with pytest.raises(
        bridge.Lfm2CpcfResumeBridgeError,
        match="not a resumable prefix",
    ):
        bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
            store,
            final_event,
            boundary_bindings=final_bindings,
            full_policy_sha256=_digest("full-policy"),
            compiler_fingerprint_sha256=_digest("compiler"),
        )


def test_progress_checkpoint_reauthenticates_event_and_pinned_store_bytes(
    tmp_path,
):
    event, bundle, _phase, _bank = _progress_event(next_layer_index=0)
    bindings = _bindings(bridge.flatten_lfm2_cpcf_boundary_bundle(bundle))
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    published = bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
        store,
        event,
        boundary_bindings=bindings,
        full_policy_sha256=_digest("full-policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
    )
    reports = published.boundary.path / "reports.json"
    reports.write_bytes(reports.read_bytes() + b" ")
    with pytest.raises(bridge.Lfm2CpcfResumeArtifactIntegrityError):
        bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
            store,
            event,
            boundary_bindings=bindings,
            full_policy_sha256=_digest("full-policy"),
            compiler_fingerprint_sha256=_digest("compiler"),
            pinned_boundary=published.boundary,
            previous_checkpoint=published.checkpoint,
        )

    fresh_event, fresh_bundle, _fresh_phase, _fresh_bank = _progress_event(
        next_layer_index=0
    )
    fresh_store = Lfm2CpcfResumeArtifactStore(tmp_path / "fresh-resume")
    report = bytearray(fresh_event._report_bytes)
    report[-2] = ord("0") if report[-2] != ord("0") else ord("1")
    object.__setattr__(fresh_event, "_report_bytes", bytes(report))
    with pytest.raises(
        bridge.Lfm2CpcfResumeBridgeError,
        match="verification",
    ):
        bridge.publish_lfm2_cpcf_model_fold_progress_checkpoint(
            fresh_store,
            fresh_event,
            boundary_bindings=_bindings(
                bridge.flatten_lfm2_cpcf_boundary_bundle(fresh_bundle)
            ),
            full_policy_sha256=_digest("full-policy"),
            compiler_fingerprint_sha256=_digest("compiler"),
        )


def test_failure_context_publication_bank_reconstruction_and_prefix_install(tmp_path):
    bundle = _bundle()
    phase, bank = _phase_and_bank()
    failure = _model_failure(bundle, bank)
    flattened = bridge.flatten_lfm2_cpcf_boundary_bundle(bundle)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    publications = bridge.publish_lfm2_cpcf_model_fold_failure_context(
        store,
        failure,
        boundary_bindings=_bindings(flattened),
        full_policy_sha256=_digest("full-policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
        invocation_sha256=_digest("invocation"),
        guard_proof_sha256=failure.report["failed_prefix_target_cpu_tensor_snapshot"][
            "snapshot_sha256"
        ],
        parent_proof_sha256=failure.rollback_proof["source_state_fingerprint_before"],
        rollback_proof_sha256=failure.rollback_proof["proof_sha256"],
        global_installed_parameter_paths=(
            "model.embed_tokens.weight",
            "lm_head.weight",
        ),
    )

    incoming_names = set(publications.checkpoint.incoming_state_bank)
    assert len(incoming_names) == 1
    assert all(
        "parent" not in name and "evidence" not in name for name in incoming_names
    )
    assert publications.boundary.manifest["accepted_artifact"] is False
    assert publications.checkpoint.manifest["promotion_authorized"] is False
    assert publications.failure.diagnostic["success_claimed"] is False

    reconstructed = bridge.reconstruct_lfm2_cpcf_phase_state_bank(
        store,
        publications.checkpoint,
        phases=(phase,),
    )
    assert reconstructed.state_bank_sha256 == bank.state_bank_sha256
    assert torch.equal(reconstructed.state(phase), bank.state(phase))

    blank = _Blank()
    blank.zero_parameters()
    blank_sha = fingerprint_model_state(blank)
    install = bridge.install_lfm2_cpcf_prefix_checkpoint(
        store,
        publications.checkpoint,
        blank,
        expected_blank_state_sha256=blank_sha,
    )
    prefix = publications.checkpoint.prefix_model
    installed, inventory = bridge._target_inventory(blank)
    assert install["exact_prefix_tensor_bytes_installed"] is True
    assert install["training_steps"] == 0
    assert ["lm_head.weight", "model.embed_tokens.weight"] in inventory[
        "object_alias_groups"
    ]
    for name, value in installed.items():
        assert torch.equal(value.detach().cpu().contiguous(), prefix[name])


def test_failure_publication_rejects_caller_binding_mismatch(tmp_path):
    bundle = _bundle()
    _phase, bank = _phase_and_bank()
    failure = _model_failure(bundle, bank)
    flattened = bridge.flatten_lfm2_cpcf_boundary_bundle(bundle)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    with pytest.raises(
        bridge.Lfm2CpcfResumeBridgeError,
        match="(?i)caller-supplied",
    ):
        bridge.publish_lfm2_cpcf_model_fold_failure_context(
            store,
            failure,
            boundary_bindings=_bindings(flattened),
            full_policy_sha256=_digest("full-policy"),
            compiler_fingerprint_sha256=_digest("compiler"),
            invocation_sha256=_digest("invocation"),
            guard_proof_sha256=_digest("wrong-guard"),
            parent_proof_sha256=failure.rollback_proof[
                "source_state_fingerprint_before"
            ],
            rollback_proof_sha256=failure.rollback_proof["proof_sha256"],
        )


def test_phase_reconstruction_reauthenticates_phase_tensor_bytes(tmp_path):
    bundle = _bundle()
    phase, bank = _phase_and_bank()
    failure = _model_failure(bundle, bank)
    flattened = bridge.flatten_lfm2_cpcf_boundary_bundle(bundle)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    pubs = bridge.publish_lfm2_cpcf_model_fold_failure_context(
        store,
        failure,
        boundary_bindings=_bindings(flattened),
        full_policy_sha256=_digest("full-policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
        invocation_sha256=_digest("invocation"),
        guard_proof_sha256=failure.report["failed_prefix_target_cpu_tensor_snapshot"][
            "snapshot_sha256"
        ],
        parent_proof_sha256=failure.rollback_proof["source_state_fingerprint_before"],
        rollback_proof_sha256=failure.rollback_proof["proof_sha256"],
    )
    phase.input_ids[0, 0] += 1
    with pytest.raises(bridge.Lfm2CpcfResumeBridgeError, match="changed"):
        bridge.reconstruct_lfm2_cpcf_phase_state_bank(
            store,
            pubs.checkpoint,
            phases=(phase,),
        )


@pytest.mark.parametrize("variant", ["untied", "dtype", "extra"])
def test_prefix_install_rejects_topology_drift_without_mutation(tmp_path, variant):
    bundle = _bundle()
    _phase, bank = _phase_and_bank()
    failure = _model_failure(bundle, bank)
    flattened = bridge.flatten_lfm2_cpcf_boundary_bundle(bundle)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "resume")
    pubs = bridge.publish_lfm2_cpcf_model_fold_failure_context(
        store,
        failure,
        boundary_bindings=_bindings(flattened),
        full_policy_sha256=_digest("full-policy"),
        compiler_fingerprint_sha256=_digest("compiler"),
        invocation_sha256=_digest("invocation"),
        guard_proof_sha256=failure.report["failed_prefix_target_cpu_tensor_snapshot"][
            "snapshot_sha256"
        ],
        parent_proof_sha256=failure.rollback_proof["source_state_fingerprint_before"],
        rollback_proof_sha256=failure.rollback_proof["proof_sha256"],
    )
    if variant == "untied":
        blank = _Blank(tied=False)
    elif variant == "dtype":
        blank = _Blank(dtype=torch.float64)
    else:
        blank = _Blank()
        blank.extra = torch.nn.Parameter(torch.zeros(1))
    blank.zero_parameters()
    before = fingerprint_model_state(blank)
    with pytest.raises(bridge.Lfm2CpcfResumeBridgeError, match="inventory"):
        bridge.install_lfm2_cpcf_prefix_checkpoint(
            store,
            pubs.checkpoint,
            blank,
            expected_blank_state_sha256=before,
        )
    assert fingerprint_model_state(blank) == before


def _v5_context(blank, *, source_label: str = "source"):
    prefix, metadata = _snapshot_parts(blank)
    topology = build_lfm2_cpcf_target_topology_snapshot(
        prefix_model=prefix,
        layer_count=2,
        snapshot_metadata=metadata,
    )
    context = v5_resume.build_lfm2_cpcf_v5_resume_context(
        source_model_state_sha256=_digest(f"{source_label}-state"),
        source_model_inventory_sha256=_digest(f"{source_label}-inventory"),
        cache_sha256=_digest("v5-cache"),
        selective_catalog_sha256=_digest("v5-catalog"),
        selective_source_sha256=_digest("v5-selective-source"),
        selective_provider_sha256=_digest("v5-provider"),
        trace_sha256=_digest("v5-trace"),
        tokenizer_sha256=_digest("v5-tokenizer"),
        token_bridge_sha256=_digest("v5-token-bridge"),
        source_tied_table_sha256=_digest("v5-source-table"),
        blank_target_report_sha256=_digest("v5-blank-report"),
        blank_target_state_sha256=fingerprint_model_state(blank),
        target_topology_sha256=topology["topology_sha256"],
        planned_target_unique_parameters=sum(
            parameter.numel() for parameter in blank.parameters()
        ),
        runtime_policy_sha256=_digest("v5-runtime-policy"),
        physical_rss_policy_sha256=(
            v5_resume.build_lfm2_cpcf_v5_physical_rss_policy_sha256(
                full_phase_tree_logical_bytes=2_000_000,
                minimum_reduction_fraction=0.10,
                maximum_increase_bytes=1_000_000,
                maximum_process_rss_bytes=2_001_000,
            )
        ),
        compiler_implementation_sha256=_digest("v5-compiler"),
        resume_implementation_sha256=_digest("v5-resume"),
        selector_contract_sha256=_digest("v5-selector"),
        numerical_contract_sha256=_digest("v5-numerical"),
        gpu_contract_sha256=_digest("v5-gpu"),
        bootstrap_authentication_sha256=_digest("v5-bootstrap"),
        final_norm_authentication_sha256=_digest("v5-final-norm"),
    )
    return context


def _v5_boundary_compiler(bundle, phase, *, completed: int):
    observations = {(0, "output.final_norm_output"): _digest("v5-final-observation")}
    for layer_index in range(completed):
        for boundary_name in (
            "residual.post_operator",
            "residual.post_layer",
        ):
            observations[(0, f"layer.{layer_index:03d}.{boundary_name}")] = _digest(
                f"v5-observation-{layer_index}-{boundary_name}"
            )
    lineage = bundle.report["interface_lineage"]
    return SimpleNamespace(
        phases=(phase,),
        source_width=3,
        target_width=2,
        layer_kinds=tuple(layer.operator_kind for layer in bundle.layers),
        split_phase_sha256=dict(bundle.split_phase_sha256),
        split_row_sha256=dict(bundle.split_row_sha256),
        split_rows={"selection": 0, "fit": 2, "validation": 0},
        interface=bundle.interface,
        base_interface=SimpleNamespace(report=lineage["base_interface_report"]),
        anchor_fold=SimpleNamespace(
            interface=bundle.interface,
            report=lineage["causal_anchor_report"],
        ),
        anchor_split_reports={
            split: {"schema": "test-v5-anchor-split-v1", "split": split}
            for split in bridge._SPLITS
        },
        fit_future=torch.tensor(
            [[0.1, 0.2], [0.3, 0.4]],
            dtype=torch.float64,
        ),
        validation_future=torch.tensor(
            [[0.5, 0.6], [0.7, 0.8]],
            dtype=torch.float64,
        ),
        fit_weights=torch.tensor([0.5, 0.5], dtype=torch.float64),
        validation_weights=torch.tensor(
            [0.25, 0.75],
            dtype=torch.float64,
        ),
        fit_weight_summary=[{"phase_index": 0, "weight": 1.0}],
        validation_weight_summary=[{"phase_index": 0, "weight": 1.0}],
        layer_results=list(bundle.layers[:completed]),
        _observation_sha=observations,
    )


def _fresh_v5_boundary_compiler(bundle, phase):
    return _v5_boundary_compiler(bundle, phase, completed=0)


class _MockedV5RssGuard:
    def __init__(self, next_layer_index: int) -> None:
        self.current = [1_000]
        self.lifetime_hwm = [1_000]
        self.patchers = (
            mock.patch.object(
                v5_compiler,
                "_process_rss_bytes",
                side_effect=lambda: self.current[0],
            ),
            mock.patch.object(
                v5_compiler,
                "_process_peak_rss_bytes",
                side_effect=lambda: self.lifetime_hwm[0],
            ),
        )
        for patcher in self.patchers:
            patcher.start()
        self.guard = v5_compiler._PhysicalRssGuard(
            maximum_increase_bytes=1_000_000,
            maximum_process_rss_bytes=2_001_000,
            full_phase_tree_logical_bytes=2_000_000,
            minimum_reduction_fraction=0.10,
            sample_interval_seconds=1.0,
        )
        self.label = (
            "prefix_000000_persisted_before_residency"
            if next_layer_index == 0
            else (
                f"layer_{next_layer_index - 1:03d}_compiled_"
                "checkpoint_persisted_before_release"
            )
        )
        self.expectation = self.guard.next_checkpoint_expectation(self.label)

    def issue(
        self,
        *,
        passed: bool,
        peak_increase_bytes: int,
    ):
        self.current[0] = 1_000 + peak_increase_bytes
        self.lifetime_hwm[0] = max(
            self.lifetime_hwm[0],
            self.current[0],
        )
        if passed:
            return self.guard.checkpoint(self.label)
        try:
            self.guard.checkpoint(self.label)
        except v5_compiler.Lfm2CpcfV5PhysicalRssError as exc:
            return exc.report
        raise AssertionError("Mocked RSS failure unexpectedly passed.")

    def close(self) -> None:
        self.guard.abort()
        for patcher in reversed(self.patchers):
            patcher.stop()


_V5_TEST_RSS_GUARDS = {}


@pytest.fixture(autouse=True)
def _cleanup_mocked_v5_rss_guards():
    yield
    while _V5_TEST_RSS_GUARDS:
        _unused, harness = _V5_TEST_RSS_GUARDS.popitem()
        harness.close()


def _v5_rss_expectation(next_layer_index: int):
    harness = _MockedV5RssGuard(next_layer_index)
    expectation = harness.expectation
    _V5_TEST_RSS_GUARDS[expectation["expectation_sha256"]] = harness
    return expectation


def _v5_rss_receipt(
    expectation,
    *,
    passed: bool,
    peak_increase_bytes: int = 0,
):
    harness = _V5_TEST_RSS_GUARDS.pop(expectation["expectation_sha256"])
    try:
        return harness.issue(
            passed=passed,
            peak_increase_bytes=peak_increase_bytes,
        )
    finally:
        harness.close()


def _v5_progress(next_layer_index: int):
    bundle = _bundle_with_depth(2)
    phase, _unused_bank = _phase_and_bank()
    blank = _Blank(depth=2)
    blank.zero_parameters()
    context = _v5_context(blank)
    compiler = _v5_boundary_compiler(
        bundle,
        phase,
        completed=next_layer_index,
    )
    frontier = v5_resume.Lfm2CpcfV5BoundaryFrontier.capture(
        compiler,
        context=context,
        next_layer_index=next_layer_index,
        expected_layer_count=2,
    )
    target = _Blank(depth=2)
    target.zero_parameters()
    with torch.no_grad():
        target.model.embed_tokens.weight.copy_(bundle.interface.target_weight)
        for layer_index in range(next_layer_index):
            target.model.layers[layer_index].weight.fill_(float(layer_index + 1))
    banks = [model_compiler.seed_lfm2_cpcf_phase_states(target, (phase,))]
    for layer_index in range(next_layer_index):
        state = banks[-1].state(phase).clone() + float(layer_index + 1)
        banks.append(
            model_compiler._phase_state_bank(
                f"post_layer_{layer_index}",
                layer_index + 1,
                (phase,),
                {phase.identity: state},
            )
        )
    receipts = tuple(
        _progress_layer_receipt(
            layer_index=layer_index,
            input_bank_sha256=banks[layer_index].state_bank_sha256,
            next_bank_sha256=banks[layer_index + 1].state_bank_sha256,
            charts_sha256=bundle.layers[layer_index].layer_charts_sha256,
        )
        for layer_index in range(next_layer_index)
    )
    event = v5_resume.Lfm2CpcfV5ProgressEvent.capture(
        frontier=frontier,
        incoming_state_bank=banks[-1],
        completed_layer_receipts=receipts,
        target_model=target,
    )
    return {
        "blank": blank,
        "bundle": bundle,
        "phase": phase,
        "context": context,
        "frontier": frontier,
        "target": target,
        "incoming": banks[-1],
        "receipts": receipts,
        "event": event,
    }


@pytest.mark.parametrize("next_layer_index", [0, 1, 2])
def test_v5_authenticated_frontier_restores_exact_prefix_without_replay(
    tmp_path,
    next_layer_index,
):
    progress = _v5_progress(next_layer_index)
    frontier_tensor_names = set(progress["frontier"].tensors)
    assert "interface.target_weight" in frontier_tensor_names
    assert all(
        "fit_future" not in name
        and "validation_future" not in name
        and "phase_weight" not in name
        for name in frontier_tensor_names
    )
    store = Lfm2CpcfResumeArtifactStore(tmp_path / f"v5-resume-{next_layer_index}")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    expectation = _v5_rss_expectation(next_layer_index)
    pending = coordinator.publish_pending(
        progress["event"],
        rss_checkpoint_expectation=expectation,
    )
    admission = coordinator.admit(
        pending,
        rss_checkpoint_receipt=_v5_rss_receipt(
            expectation,
            passed=True,
        ),
    )
    state = coordinator.open_admitted(admission["admission_sha256"])
    resumed_target = _Blank(depth=2)
    resumed_target.zero_parameters()
    fresh_compiler = _fresh_v5_boundary_compiler(
        progress["bundle"],
        progress["phase"],
    )

    restored = coordinator.restore(
        state,
        target_model=resumed_target,
        phases=(progress["phase"],),
        boundary_compiler=fresh_compiler,
        expected_context=progress["context"],
        expected_blank_state_sha256=fingerprint_model_state(resumed_target),
    )

    assert fingerprint_model_state(resumed_target) == fingerprint_model_state(
        progress["target"]
    )
    assert (
        restored.state_bank.state_bank_sha256 == progress["incoming"].state_bank_sha256
    )
    assert len(restored.completed_layer_receipts) == next_layer_index
    assert [layer.layer_index for layer in fresh_compiler.layer_results] == list(
        range(next_layer_index)
    )
    assert [layer.layer_charts_sha256 for layer in fresh_compiler.layer_results] == [
        receipt.report["boundary_charts_sha256"] for receipt in progress["receipts"]
    ]
    assert restored.frontier.frontier_sha256 == progress["frontier"].frontier_sha256
    assert restored.prefix_install_report["training_steps"] == 0


def test_v5_progress_rejects_receipt_frontier_chart_drift():
    progress = _v5_progress(1)
    original = progress["receipts"][0]
    drifted = _progress_layer_receipt(
        layer_index=original.layer_index,
        input_bank_sha256=original.input_bank_sha256,
        next_bank_sha256=original.next_bank_sha256,
        charts_sha256=_digest("different-upstream-transported-chart"),
    )

    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="receipts and incoming child state are not contiguous",
    ):
        v5_resume.Lfm2CpcfV5ProgressEvent.capture(
            frontier=progress["frontier"],
            incoming_state_bank=progress["incoming"],
            completed_layer_receipts=(drifted,),
            target_model=progress["target"],
        )


def test_v5_resume_rejects_live_context_drift_and_store_tamper(tmp_path):
    progress = _v5_progress(1)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    expectation = _v5_rss_expectation(1)
    pending = coordinator.publish_pending(
        progress["event"],
        rss_checkpoint_expectation=expectation,
    )
    admission = coordinator.admit(
        pending,
        rss_checkpoint_receipt=_v5_rss_receipt(
            expectation,
            passed=True,
        ),
    )
    state = coordinator.open_admitted(admission["admission_sha256"])
    blank = _Blank(depth=2)
    blank.zero_parameters()
    before = fingerprint_model_state(blank)
    with pytest.raises(v5_resume.Lfm2CpcfV5ResumeError, match="context"):
        coordinator.restore(
            state,
            target_model=blank,
            phases=(progress["phase"],),
            boundary_compiler=_fresh_v5_boundary_compiler(
                progress["bundle"],
                progress["phase"],
            ),
            expected_context=_v5_context(
                progress["blank"],
                source_label="changed-source",
            ),
            expected_blank_state_sha256=before,
        )
    assert fingerprint_model_state(blank) == before

    frontier_path = (
        store.root
        / "boundaries"
        / pending.frontier_reference.artifact_key
        / "reports.json"
    )
    frontier_path.write_bytes(frontier_path.read_bytes() + b" ")
    with pytest.raises(bridge.Lfm2CpcfResumeArtifactIntegrityError):
        coordinator.open_admitted(admission["admission_sha256"])


def test_v5_rss_rejection_retains_pending_but_blocks_normal_admission(
    tmp_path,
):
    progress = _v5_progress(1)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    expectation = _v5_rss_expectation(1)
    pending = coordinator.publish_pending(
        progress["event"],
        rss_checkpoint_expectation=expectation,
    )
    rejection = coordinator.record_rss_rejection(
        pending,
        rss_violation_report=(
            rejection_receipt := _v5_rss_receipt(
                expectation,
                passed=False,
                peak_increase_bytes=1_500_000,
            )
        ),
    )

    store.final_reauthenticate_boundary(store.open_boundary(pending.frontier_reference))
    store.final_reauthenticate_checkpoint(
        store.open_checkpoint(pending.checkpoint.reference)
    )
    assert rejection["promotable"] is False
    assert rejection["next_layer_index"] == 1
    with pytest.raises(v5_resume.Lfm2CpcfV5ResumeError):
        coordinator.open_admitted(rejection["rejection_sha256"])
    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="outcome is invalid",
    ):
        coordinator.admit(
            pending,
            rss_checkpoint_receipt=rejection_receipt,
        )


def test_v5_coordinator_has_one_definition_per_security_method() -> None:
    source = textwrap.dedent(inspect.getsource(v5_resume.Lfm2CpcfV5ResumeCoordinator))
    class_node = ast.parse(source).body[0]
    names = [
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    for name in (
        "_document_path",
        "_write_once",
        "_open_document",
        "publish_pending",
        "admit",
        "record_rss_rejection",
        "open_admitted",
    ):
        assert names.count(name) == 1


def test_v5_document_publication_crash_never_poisons_final_key(
    tmp_path,
    monkeypatch,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    key = _digest("crash-before-rename")
    path = coordinator._document_path("v5-progress", key)

    def fail_before_install(_source, _destination):
        raise OSError("simulated publication crash")

    monkeypatch.setattr(
        v5_resume.resume_artifact,
        "_atomic_rename_noreplace",
        fail_before_install,
    )
    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="before installing",
    ):
        coordinator._write_once(path, {"payload": "complete-only"})

    assert not os.path.lexists(path)
    assert list(path.parent.glob(".v5-document-*.tmp")) == []


def test_v5_partial_temporary_write_is_removed_without_final_key(
    tmp_path,
    monkeypatch,
):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    key = _digest("partial-temporary-write")
    path = coordinator._document_path("v5-progress", key)
    real_write = os.write
    calls = [0]

    def partial_then_crash(descriptor, payload):
        if calls[0] == 0:
            calls[0] += 1
            chunk = payload[: max(1, len(payload) // 2)]
            return real_write(descriptor, chunk)
        raise OSError("simulated short-write crash")

    monkeypatch.setattr(v5_resume.os, "write", partial_then_crash)
    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="atomically publish",
    ):
        coordinator._write_once(
            path,
            {"payload": "must-never-appear-truncated"},
        )

    assert not os.path.lexists(path)
    assert list(path.parent.glob(".v5-document-*.tmp")) == []


def test_v5_document_key_and_symlink_confusion_fail_closed(tmp_path):
    progress = _v5_progress(0)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    expectation = _v5_rss_expectation(0)
    pending = coordinator.publish_pending(
        progress["event"],
        rss_checkpoint_expectation=expectation,
    )
    real_path = coordinator._document_path(
        "v5-progress",
        pending.event_report_sha256,
    )
    copied_key = _digest("copied-as-another-key")
    copied_path = coordinator._document_path("v5-progress", copied_key)
    copied_path.write_bytes(real_path.read_bytes())

    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="schema, seal, or requested content key",
    ):
        coordinator._open_document(
            copied_path,
            kind="progress",
            expected_key=copied_key,
        )
    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="path stem",
    ):
        coordinator._open_document(
            real_path,
            kind="progress",
            expected_key=copied_key,
        )

    link_key = _digest("symlink-as-another-key")
    link_path = coordinator._document_path("v5-progress", link_key)
    try:
        link_path.symlink_to(real_path)
    except OSError:
        pytest.skip("This Windows account cannot create a test symlink.")
    with pytest.raises(
        bridge.Lfm2CpcfResumeArtifactIntegrityError,
        match="regular file",
    ):
        coordinator._open_document(
            link_path,
            kind="progress",
            expected_key=link_key,
        )


def test_v5_swapped_collection_symlink_fails_identity_gate(tmp_path):
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    collection = coordinator._collection("v5-terminals")
    moved = collection.with_name("real-v5-terminals")
    collection.rename(moved)
    try:
        try:
            collection.symlink_to(moved, target_is_directory=True)
        except OSError:
            pytest.skip("This Windows account cannot create a test symlink.")
        with pytest.raises(
            bridge.Lfm2CpcfResumeArtifactIntegrityError,
            match="real directory",
        ):
            coordinator._collection("v5-terminals")
    finally:
        if collection.is_symlink():
            collection.unlink()
        if moved.exists() and not collection.exists():
            moved.rename(collection)


def test_v5_pending_capability_progress_and_typed_rss_fail_closed(
    tmp_path,
):
    progress = _v5_progress(1)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    expectation = _v5_rss_expectation(1)
    pending = coordinator.publish_pending(
        progress["event"],
        rss_checkpoint_expectation=expectation,
    )
    valid_receipt = _v5_rss_receipt(expectation, passed=True)

    with pytest.raises(v5_resume.Lfm2CpcfV5ResumeError):
        coordinator.open_admitted(pending.pending_sha256)
    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="not issued by this coordinator",
    ):
        coordinator.admit(
            replace(pending, capability_token="0" * 64),
            rss_checkpoint_receipt=valid_receipt,
        )
    restarted = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="not issued by this coordinator",
    ):
        restarted.admit(
            pending,
            rss_checkpoint_receipt=valid_receipt,
        )
    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="exact typed",
    ):
        coordinator.admit(
            pending,
            rss_checkpoint_receipt=dict(valid_receipt),
        )

    assert not hasattr(
        v5_resume,
        "_issue_lfm2_cpcf_v5_rss_checkpoint_receipt",
    )
    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="exact live",
    ):
        v5_resume.Lfm2CpcfV5PhysicalRssCheckpointReceipt(
            dict(valid_receipt),
            _guard=object(),
            _sequence_capability=object(),
        )

    event_path = coordinator._document_path(
        "v5-progress",
        pending.event_report_sha256,
    )
    event_path.write_bytes(event_path.read_bytes() + b" ")
    with pytest.raises(
        bridge.Lfm2CpcfResumeArtifactIntegrityError,
    ):
        coordinator.admit(
            pending,
            rss_checkpoint_receipt=valid_receipt,
        )


def test_v5_frontier_rejects_untyped_or_looser_rss_policy(tmp_path):
    progress = _v5_progress(1)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    expectation = _v5_rss_expectation(1)
    looser_body = dict(expectation)
    looser_body.pop("expectation_sha256")
    looser_body["maximum_increase_bytes"] = 1_100_000
    looser = {
        **looser_body,
        "expectation_sha256": canonical_sha256(looser_body),
    }

    with pytest.raises(
        v5_resume.Lfm2CpcfV5ResumeError,
        match="exact one-shot expectation",
    ):
        coordinator.publish_pending(
            progress["event"],
            rss_checkpoint_expectation=looser,
        )


def test_v5_admit_reject_race_has_exactly_one_terminal_outcome(tmp_path):
    progress = _v5_progress(1)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    expectation = _v5_rss_expectation(1)
    pending = coordinator.publish_pending(
        progress["event"],
        rss_checkpoint_expectation=expectation,
    )
    admission_receipt = _v5_rss_receipt(expectation, passed=True)
    barrier = threading.Barrier(2)
    successes = []
    failures = []

    def run_admission():
        try:
            barrier.wait()
            successes.append(
                (
                    "admission",
                    coordinator.admit(
                        pending,
                        rss_checkpoint_receipt=admission_receipt,
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - race evidence
            failures.append(exc)

    def run_rejection():
        try:
            barrier.wait()
            successes.append(
                (
                    "rejection",
                    coordinator.record_rss_rejection(
                        pending,
                        rss_violation_report=admission_receipt,
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - race evidence
            failures.append(exc)

    threads = [
        threading.Thread(target=run_admission),
        threading.Thread(target=run_rejection),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], v5_resume.Lfm2CpcfV5ResumeError)
    terminal_keys = coordinator._terminal_pending_keys()
    assert terminal_keys == (pending.pending_sha256,)
    kind, outcome = successes[0]
    if kind == "admission":
        state = coordinator.open_admitted(outcome["admission_sha256"])
        assert state.frontier.frontier_sha256 == pending.frontier_sha256
    else:
        with pytest.raises(v5_resume.Lfm2CpcfV5ResumeError):
            coordinator.open_admitted(outcome["rejection_sha256"])


def test_v5_real_rss_guard_failure_publishes_typed_terminal_rejection(
    tmp_path,
    monkeypatch,
):
    current = [1_000]
    monkeypatch.setattr(
        v5_compiler,
        "_process_rss_bytes",
        lambda: current[0],
    )
    monkeypatch.setattr(
        v5_compiler,
        "_process_peak_rss_bytes",
        lambda: current[0],
    )
    guard = v5_compiler._PhysicalRssGuard(
        maximum_increase_bytes=1_000_000,
        maximum_process_rss_bytes=2_001_000,
        full_phase_tree_logical_bytes=2_000_000,
        minimum_reduction_fraction=0.10,
        sample_interval_seconds=1.0,
    )
    progress = _v5_progress(1)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = v5_resume.Lfm2CpcfV5ResumeCoordinator(store)
    label = "layer_000_compiled_checkpoint_persisted_before_release"
    pending = coordinator.publish_pending(
        progress["event"],
        rss_checkpoint_expectation=guard.next_checkpoint_expectation(label),
    )
    current[0] = 1_001_001
    try:
        with pytest.raises(
            v5_compiler.Lfm2CpcfV5PhysicalRssError,
            match="physical RSS increase",
        ) as captured:
            guard.checkpoint(label)
        assert type(captured.value.report) is (
            v5_resume.Lfm2CpcfV5PhysicalRssCheckpointReceipt
        )
        rejection = coordinator.record_rss_rejection(
            pending,
            rss_violation_report=captured.value.report,
        )
    finally:
        guard.abort()

    assert rejection["promotable"] is False
    assert coordinator._terminal_pending_keys() == (pending.pending_sha256,)
