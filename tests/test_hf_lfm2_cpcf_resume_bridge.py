from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

import neural_forge.hf_lfm2_cpcf_model_compiler as model_compiler
import neural_forge.hf_lfm2_cpcf_model_fold as model_fold
import neural_forge.hf_lfm2_cpcf_resume_bridge as bridge
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
