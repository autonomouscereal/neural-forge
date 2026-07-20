from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_model_fold as model_fold  # noqa: E402
import neural_forge.hf_lfm2_cpcf_layer_compiler as layer_compiler  # noqa: E402
import neural_forge.hf_lfm2_cpcf_model_compiler as model_compiler  # noqa: E402
import neural_forge.hf_closed_form_causal_circuit_fold as causal_fold  # noqa: E402
import neural_forge.hf_tied_causal_interface_fold as tied_fold  # noqa: E402
from neural_forge.hf_closed_form_causal_circuit_fold import (  # noqa: E402
    PredictiveBoundaryMap,
)
from neural_forge.hf_cpcf_boundary_chart import CpcfBoundaryChart  # noqa: E402
from neural_forge.hf_lfm2_cpcf_boundary_bundle import (  # noqa: E402
    LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA,
    LFM2_CPCF_LAYER_CHARTS_SCHEMA,
    Lfm2CpcfBoundaryBundle,
    Lfm2CpcfLayerBoundaryCharts,
)
from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (  # noqa: E402
    Lfm2CpcfCompletedLayerReceipt,
    Lfm2CpcfModelFoldError,
    Lfm2CpcfModelFoldPolicy,
    Lfm2CpcfModelFoldResumeState,
    audit_lfm2_cpcf_parameter_compaction,
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (  # noqa: E402
    Lfm2CpcfShortConvV4SolverPolicy,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceFold,
)


class _TiedModel(torch.nn.Module):
    def __init__(self, rows: int, width: int) -> None:
        super().__init__()
        self.table = torch.nn.Embedding(rows, width)
        self.lm_head = torch.nn.Linear(width, rows, bias=False)
        self.lm_head.weight = self.table.weight
        self.config = SimpleNamespace(model_type="lfm2")


class _TransactionalModel(torch.nn.Module):
    def __init__(self, width: int, value: float, *, depth: int = 1) -> None:
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.full((width,), value))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _index in range(depth)]
        )
        self.model.embedding_norm = SimpleNamespace(variance_epsilon=1e-5)
        self.config = SimpleNamespace(
            model_type="lfm2",
            hidden_size=width,
        )


def _real_phase_bank():
    donor_sha = "1" * 64
    layer_evidence = Lfm2CpcfInstallableLayerEvidence(
        layer_index=0,
        operator_kind="short_conv",
        source_width=2,
        tensors={"operator.input": torch.ones(1, 2, 2)},
        evidence_sha256="2" * 64,
    )
    parent_evidence = Lfm2CpcfInstallablePhaseEvidence(
        cache_key="3" * 64,
        batch_index=0,
        input_sha256=donor_sha,
        phase="prefill",
        layers=(layer_evidence,),
        final_norm_tensors={"output.final_norm_output": torch.ones(1, 2, 2)},
        evidence_sha256="4" * 64,
    )
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids)
    positions = torch.tensor([[0, 1]], dtype=torch.long)
    phase_body = {
        "schema": model_compiler.LFM2_CPCF_TARGET_PHASE_SCHEMA,
        "batch_index": 0,
        "record_id": "record-0",
        "split": "fit",
        "phase": "prefill",
        "donor_input_sha256": donor_sha,
        "target_alignment_sha256": "5" * 64,
        "input_ids_sha256": model_compiler._tensor_sha256(ids),
        "attention_mask_sha256": model_compiler._tensor_sha256(mask),
        "position_ids_sha256": model_compiler._tensor_sha256(positions),
        "batch_size": 1,
        "full_prefix_tokens": 2,
        "sampled_suffix_tokens": 1,
        "parent_evidence_sha256": parent_evidence.evidence_sha256,
        "numeric_token_id_equality_required": False,
        "actual_child_full_prefix_required": True,
    }
    phase = model_compiler.Lfm2CpcfTargetPhase(
        batch_index=0,
        record_id="record-0",
        split="fit",
        phase="prefill",
        donor_input_sha256=donor_sha,
        target_alignment_sha256="5" * 64,
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        sampled_suffix_tokens=1,
        parent_evidence=parent_evidence,
        phase_sha256=model_compiler._sha(phase_body),
    )
    bank = model_compiler._phase_state_bank(
        "embedding_output",
        0,
        (phase,),
        {phase.identity: torch.zeros(1, 2, 1)},
    )
    return phase, bank


def _resume_digest(label: str) -> str:
    return model_fold._sha({"label": label})


def _resume_sealed(body):
    return {**body, "report_sha256": model_fold._sha(body)}


def _resume_phases():
    phases = []
    for batch_index, split in enumerate(("selection", "fit", "validation")):
        donor_sha = _resume_digest(f"donor-{split}")
        layers = tuple(
            Lfm2CpcfInstallableLayerEvidence(
                layer_index=layer_index,
                operator_kind="short_conv",
                source_width=3,
                tensors={
                    "residual.pre": torch.full(
                        (1, 2, 3),
                        float(batch_index + layer_index + 1),
                    )
                },
                evidence_sha256=_resume_digest(f"layer-{split}-{layer_index}"),
            )
            for layer_index in range(2)
        )
        parent = Lfm2CpcfInstallablePhaseEvidence(
            cache_key=_resume_digest("resume-cache"),
            batch_index=batch_index,
            input_sha256=donor_sha,
            phase="prefill",
            layers=layers,
            final_norm_tensors={
                "output.final_norm_output": torch.full(
                    (1, 2, 3),
                    float(batch_index + 4),
                )
            },
            evidence_sha256=_resume_digest(f"phase-evidence-{split}"),
        )
        ids = torch.tensor([[1, 2]], dtype=torch.long)
        mask = torch.ones_like(ids)
        positions = torch.tensor([[0, 1]], dtype=torch.long)
        alignment_sha = _resume_digest(f"alignment-{split}")
        phase_body = {
            "schema": model_compiler.LFM2_CPCF_TARGET_PHASE_SCHEMA,
            "batch_index": batch_index,
            "record_id": f"record-{split}",
            "split": split,
            "phase": "prefill",
            "donor_input_sha256": donor_sha,
            "target_alignment_sha256": alignment_sha,
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
        phases.append(
            model_compiler.Lfm2CpcfTargetPhase(
                batch_index=batch_index,
                record_id=f"record-{split}",
                split=split,
                phase="prefill",
                donor_input_sha256=donor_sha,
                target_alignment_sha256=alignment_sha,
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                sampled_suffix_tokens=1,
                parent_evidence=parent,
                phase_sha256=model_compiler._sha(phase_body),
            )
        )
    return tuple(phases)


def _resume_mapping(seed: int) -> PredictiveBoundaryMap:
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
        "encoder_sha256": causal_fold._tensor_sha256(encoder),
        "decoder_sha256": causal_fold._tensor_sha256(decoder),
    }
    return PredictiveBoundaryMap(
        encoder=encoder,
        decoder=decoder,
        feature_predictors=predictors,
        report=_resume_sealed(body),
    )


def _resume_chart(
    layer_index: int,
    suffix: str,
    seed: int,
) -> CpcfBoundaryChart:
    mapping = _resume_mapping(seed)
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
        report=_resume_sealed(body),
    )


def _resume_bundle(
    phases,
    *,
    wrong_phase_lineage: bool = False,
) -> Lfm2CpcfBoundaryBundle:
    target_weight = torch.tensor(
        [[0.2, -0.3], [0.7, 0.1], [-0.5, 0.4]],
        dtype=torch.float64,
    )
    producer_encoder = torch.tensor(
        [[0.7, -0.2], [0.1, 0.8], [-0.4, 0.3]],
        dtype=torch.float64,
    )
    target_sha = tied_fold._tensor_sha256(target_weight)
    encoder_sha = tied_fold._tensor_sha256(producer_encoder)
    base_body = {
        "schema": "neural-forge-tied-causal-interface-v1",
        "target_weight_sha256": target_sha,
        "producer_encoder_sha256": encoder_sha,
        "source_table_sha256": _resume_digest("source-table"),
    }
    base_report = _resume_sealed(base_body)
    base_sha = base_report["report_sha256"]
    anchor_body = {
        "schema": "neural-forge-tied-causal-interface-anchor-v1",
        "target_weight_sha256": target_sha,
        "producer_encoder_sha256": encoder_sha,
        "base_interface_report_sha256": base_sha,
        "base_interface_lineage": {
            "report": base_report,
            "report_sha256": base_sha,
            "target_weight_sha256": target_sha,
            "producer_encoder_sha256": encoder_sha,
            "source_table_sha256": base_report["source_table_sha256"],
            "tensor_identity_verified": True,
        },
    }
    anchor_report = _resume_sealed(anchor_body)
    interface = TiedCausalInterfaceFold(
        target_weight=target_weight,
        producer_encoder=producer_encoder,
        report=anchor_report,
    )
    layers = []
    for layer_index in range(2):
        post_operator = _resume_chart(
            layer_index,
            "post_operator",
            100 + layer_index * 2,
        )
        post_layer = _resume_chart(
            layer_index,
            "post_layer",
            101 + layer_index * 2,
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
                layer_charts_sha256=model_fold._sha(layer_body),
            )
        )
    split_phase, split_row = model_fold._phase_lineage_hashes(phases)
    if wrong_phase_lineage:
        split_phase["fit"] = _resume_digest("wrong-fit-lineage")
    anchor_sha = anchor_report["report_sha256"]
    bundle_body = {
        "schema": LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA,
        "source_width": 3,
        "target_width": 2,
        "vocabulary_size": 3,
        "layer_count": 2,
        "layer_operator_kinds": ["short_conv", "short_conv"],
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
        "layer_charts": [layer.as_dict() for layer in layers],
        "split_phase_sha256": split_phase,
        "split_row_sha256": split_row,
    }
    bundle_sha = model_fold._sha(bundle_body)
    return Lfm2CpcfBoundaryBundle(
        interface=interface,
        layers=tuple(layers),
        split_phase_sha256=split_phase,
        split_row_sha256=split_row,
        report={**bundle_body, "bundle_sha256": bundle_sha},
        bundle_sha256=bundle_sha,
    )


def _resume_bank(
    phases,
    *,
    layer_index: int,
    state_value: float,
):
    boundary = (
        "input_embeddings"
        if layer_index == 0
        else f"layer.{layer_index - 1:03d}.post_layer"
    )
    return model_compiler._phase_state_bank(
        boundary,
        layer_index,
        phases,
        {
            phase.identity: torch.full(
                (phase.batch_size, phase.full_prefix_tokens, 2),
                state_value,
            )
            for phase in phases
        },
    )


def _resume_layer_receipt(
    *,
    layer_index: int,
    charts: Lfm2CpcfLayerBoundaryCharts,
    input_bank_sha256: str,
    next_bank_sha256: str,
    phases,
) -> dict:
    split_counts = {
        split: sum(phase.split == split for phase in phases)
        for split in ("selection", "fit", "validation")
    }
    body = {
        "schema": layer_compiler.LFM2_CPCF_LAYER_COMPILER_SCHEMA,
        "layer_index": layer_index,
        "operator_kind": charts.operator_kind,
        "source_width": charts.source_width,
        "target_width": charts.target_width,
        "input_bank_sha256": input_bank_sha256,
        "next_bank_sha256": next_bank_sha256,
        "boundary_charts_sha256": charts.layer_charts_sha256,
        "preinstall_replay_sha256": _resume_digest(f"preinstall-{layer_index}"),
        "operator_replay_sha256": _resume_digest(f"operator-{layer_index}"),
        "final_replay_sha256": _resume_digest(f"final-{layer_index}"),
        "operator_artifact": {"kind": "synthetic"},
        "mlp_artifact": {"kind": "synthetic"},
        "operator_install_report": {"kind": "synthetic"},
        "mlp_install_report": {"kind": "synthetic"},
        "split_metrics": {},
        "split_phase_counts": split_counts,
        "solver_evidence_policy": {"synthetic": True},
        "actual_child_state_policy": {"synthetic": True},
        "audit": {"synthetic": True},
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
    return {**body, "result_sha256": model_fold._sha(body)}


def _synthetic_compiled_layer(receipt, next_bank):
    result = object.__new__(layer_compiler.Lfm2CpcfCompiledLayerResult)
    object.__setattr__(result, "layer_index", receipt["layer_index"])
    object.__setattr__(result, "operator_kind", receipt["operator_kind"])
    object.__setattr__(
        result,
        "input_bank_sha256",
        receipt["input_bank_sha256"],
    )
    object.__setattr__(
        result,
        "final_replay",
        SimpleNamespace(post_layer_bank=next_bank),
    )
    object.__setattr__(result, "report", receipt)
    object.__setattr__(
        result,
        "result_sha256",
        receipt["result_sha256"],
    )
    return result


def _resume_prefix_install_report(target, *, next_layer_index: int):
    topology_sha, tensor_count = model_fold._target_topology_sha256(target)
    body = {
        "schema": model_fold.LFM2_CPCF_PREFIX_INSTALL_REPORT_SCHEMA,
        "checkpoint_reference": {
            "artifact_kind": "checkpoint",
            "artifact_key": _resume_digest("checkpoint-key"),
            "manifest_sha256": _resume_digest("checkpoint-manifest"),
        },
        "target_topology_sha256": topology_sha,
        "blank_state_sha256_before": _resume_digest("blank-state"),
        "installed_state_sha256_after": model_fold.fingerprint_model_state(target),
        "tensor_count": tensor_count,
        "next_layer_index": next_layer_index,
        "exact_inventory_verified": True,
        "exact_object_aliases_verified": True,
        "exact_storage_aliases_verified": True,
        "exact_shape_dtype_stride_offset_device_verified": True,
        "tensor_identities_preserved": True,
        "exact_prefix_tensor_bytes_installed": True,
        "loaders_used": 0,
        "parent_evidence_tensors_copied": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "promotion_claimed": False,
    }
    return {**body, "report_sha256": model_fold._sha(body)}


def _blank_report(target_unique_parameters: int):
    body = {
        "compiler_created_blank": True,
        "all_parameters_zero": True,
        "dense_materialized_parameters": True,
        "topology_provenance_verified": True,
        "weights_installed": False,
        "training_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "target_unique_parameters": target_unique_parameters,
    }
    return {**body, "report_sha256": model_fold._sha(body)}


def _valid_resume_case():
    phases = _resume_phases()
    bundle = _resume_bundle(phases)
    initial_bank = _resume_bank(
        phases,
        layer_index=0,
        state_value=0.0,
    )
    incoming_bank = _resume_bank(
        phases,
        layer_index=1,
        state_value=1.0,
    )
    target = _TransactionalModel(2, 1.0, depth=2)
    policy = Lfm2CpcfModelFoldPolicy()
    receipt = _resume_layer_receipt(
        layer_index=0,
        charts=bundle.layers[0],
        input_bank_sha256=initial_bank.state_bank_sha256,
        next_bank_sha256=incoming_bank.state_bank_sha256,
        phases=phases,
    )
    resume = Lfm2CpcfModelFoldResumeState(
        prefix_install_report=_resume_prefix_install_report(
            target,
            next_layer_index=1,
        ),
        full_policy_sha256=model_fold._sha(policy.as_dict()),
        boundary_bundle=bundle,
        incoming_state_bank=incoming_bank,
        completed_layer_receipts=(receipt,),
    )
    return SimpleNamespace(
        phases=phases,
        bundle=bundle,
        initial_bank=initial_bank,
        incoming_bank=incoming_bank,
        target=target,
        policy=policy,
        receipt=receipt,
        resume=resume,
    )


def _patch_progress_fold_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    phases,
    bundle,
    banks,
):
    layer_calls = []

    monkeypatch.setattr(
        model_fold,
        "fit_lfm2_cpcf_boundary_bundle",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_tied_interface",
        lambda *_args, **_kwargs: {
            "report_sha256": _resume_digest("interface-install")
        },
    )
    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_neutral_norms",
        lambda *_args, **_kwargs: {
            "report_sha256": _resume_digest("neutral-norm-install")
        },
    )
    monkeypatch.setattr(
        model_fold,
        "_preserved_qk_norm_weights",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        model_fold,
        "seed_lfm2_cpcf_phase_states",
        lambda *_args, **_kwargs: banks[0],
    )

    def _compile_layer(_source, target, **kwargs):
        index = kwargs["layer_index"]
        layer_calls.append(index)
        with torch.no_grad():
            target.scalar.add_(index + 1)
        receipt = _resume_layer_receipt(
            layer_index=index,
            charts=bundle.layers[index],
            input_bank_sha256=banks[index].state_bank_sha256,
            next_bank_sha256=banks[index + 1].state_bank_sha256,
            phases=phases,
        )
        return _synthetic_compiled_layer(receipt, banks[index + 1])

    monkeypatch.setattr(
        model_fold,
        "compile_and_install_lfm2_cpcf_layer",
        _compile_layer,
    )
    monkeypatch.setattr(
        model_fold,
        "_fit_and_install_final_norm",
        lambda *_args, **_kwargs: {"report_sha256": _resume_digest("final-norm")},
    )
    return layer_calls


def _sealed_layer_failure(
    *,
    input_bank_sha256: str,
    boundary_charts_sha256: str,
) -> layer_compiler.Lfm2CpcfLayerCompilationFailure:
    target_sha = "a" * 64
    source_sha = "b" * 64
    evidence_sha = "c" * 64
    input_tensor_sha = "d" * 64
    charts_payload_sha = "e" * 64
    proof_body = {
        "target_state_sha256_before": target_sha,
        "target_state_sha256_after": target_sha,
        "target_full_inventory_sha256_before": target_sha,
        "target_full_inventory_sha256_after": target_sha,
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
        "source_state_sha256_before": source_sha,
        "source_state_sha256_after": source_sha,
        "source_full_inventory_sha256_before": source_sha,
        "source_full_inventory_sha256_after": source_sha,
        "source_state_preserved": True,
        "source_full_inventory_preserved": True,
        "evidence_sha256_before": evidence_sha,
        "evidence_sha256_after": evidence_sha,
        "evidence_preserved": True,
        "input_tensor_sha256_before": input_tensor_sha,
        "input_tensor_sha256_after": input_tensor_sha,
        "input_tensors_preserved": True,
        "boundary_charts_sha256_before": boundary_charts_sha256,
        "boundary_charts_sha256_after": boundary_charts_sha256,
        "boundary_charts_payload_sha256_before": charts_payload_sha,
        "boundary_charts_payload_sha256_after": charts_payload_sha,
        "boundary_charts_preserved": True,
        "exact_layer_rollback_proven": True,
    }
    proof = {
        **proof_body,
        "proof_sha256": layer_compiler._sha(proof_body),
    }
    report_body = {
        "schema": layer_compiler.LFM2_CPCF_LAYER_FAILURE_SCHEMA,
        "stage": "operator_compile",
        "layer_index": 0,
        "operator_kind": "short_conv",
        "input_bank_sha256": input_bank_sha256,
        "input_tensor_sha256": input_tensor_sha,
        "boundary_charts_sha256": boundary_charts_sha256,
        "boundary_charts_payload_sha256": charts_payload_sha,
        "cause": {
            "module": "synthetic",
            "type": "Rejected",
            "message": "heldout gate rejected",
        },
        "nested_short_conv_validation_report": None,
        "nested_short_conv_validation_report_sha256": None,
        "nested_short_conv_validation_report_canonical_bytes_sha256": None,
        "nested_short_conv_validation_report_preserved_byte_for_byte": False,
        "rollback_proof": proof,
        "layer_result_issued": False,
        "promotion_claimed": False,
    }
    report = {
        **report_body,
        "report_sha256": layer_compiler._sha(report_body),
    }
    return layer_compiler.Lfm2CpcfLayerCompilationFailure(
        "heldout gate rejected",
        stage="operator_compile",
        layer_index=0,
        operator_kind="short_conv",
        input_bank_sha256=input_bank_sha256,
        input_tensor_sha256=input_tensor_sha,
        boundary_charts_sha256=boundary_charts_sha256,
        boundary_charts_payload_sha256=charts_payload_sha,
        rollback_proof=proof,
        report=report,
    )


def test_independent_parameter_inventory_counts_native_tie_once() -> None:
    model = _TiedModel(17, 8)
    inventory = independent_parameter_inventory(model)
    assert inventory["unique_parameters"] == 17 * 8
    assert inventory["raw_named_parameter_numel"] == 2 * 17 * 8
    assert inventory["alias_group_count"] == 1
    assert inventory["independent_methods_agree"] is True


def test_model_snapshot_covers_ties_and_nonpersistent_buffers() -> None:
    model = _TiedModel(4, 3)
    model.register_buffer(
        "rotary_inv_freq",
        torch.tensor([2.0]),
        persistent=False,
    )
    original_parameter = model.table.weight
    original_buffer = model.rotary_inv_freq
    snapshot = model_fold._snapshot_model(model)
    metadata = model_fold._snapshot_metadata_plain(snapshot)

    assert ["lm_head.weight", "table.weight"] in metadata["object_alias_groups"]
    assert metadata["tensor_metadata"]["rotary_inv_freq"]["persistent"] is False
    with torch.no_grad():
        model.table.weight.fill_(9.0)
        model.rotary_inv_freq.fill_(11.0)
    model_fold._restore_model(model, snapshot)

    assert model.table.weight is original_parameter
    assert model.lm_head.weight is original_parameter
    assert model.rotary_inv_freq is original_buffer
    assert torch.equal(
        model.table.weight,
        snapshot.tensor_values["table.weight"],
    )
    assert torch.equal(model.rotary_inv_freq, torch.tensor([2.0]))


def test_model_snapshot_restores_gradient_storage_and_persistence_set() -> None:
    model = _TiedModel(4, 3)
    model.register_buffer(
        "rotary_inv_freq",
        torch.tensor([2.0]),
        persistent=False,
    )
    gradient = torch.arange(
        model.table.weight.numel(),
        dtype=model.table.weight.dtype,
    ).reshape_as(model.table.weight)
    model.table.weight.grad = gradient
    gradient_pointer = int(gradient.untyped_storage().data_ptr())
    snapshot = model_fold._snapshot_model(model)

    gradient.data = gradient.clone()
    model._non_persistent_buffers_set.clear()
    assert int(gradient.untyped_storage().data_ptr()) != gradient_pointer

    model_fold._restore_model(model, snapshot)

    assert model.table.weight.grad is gradient
    assert int(gradient.untyped_storage().data_ptr()) == gradient_pointer
    assert "rotary_inv_freq" in model._non_persistent_buffers_set
    assert (
        model_fold._sha(
            model_fold._model_inventory(
                model,
                include_tensor_bytes=True,
            )
        )
        == snapshot.full_inventory_sha256
    )


def test_success_guard_rejects_and_restores_tie_break_and_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _TiedModel(8, 3)
    target = _TiedModel(4, 3)
    original = target.table.weight

    def _break_topology(_source, live_target, **_kwargs):
        live_target.lm_head.weight = torch.nn.Parameter(
            live_target.lm_head.weight.detach().clone()
        )
        live_target.table.weight = torch.nn.Parameter(
            live_target.table.weight.detach().clone()
        )
        return object()

    monkeypatch.setattr(
        model_fold,
        "_compile_lfm2_cpcf_model_fold_mutating",
        _break_topology,
    )
    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldFailure,
        match="identities, aliases",
    ):
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            target,
            phases=(),
            source_tied_table=None,
            blank_target_report={},
            planned_target_unique_parameters=12,
        )

    assert target.table.weight is original
    assert target.lm_head.weight is original


def test_boundary_fit_input_and_source_table_mutation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase, _bank = _real_phase_bank()
    source = _TransactionalModel(2, 1.0)
    target = _TransactionalModel(1, 0.0)
    source_table = torch.ones(3, 2)

    def _mutate_boundary_inputs(_source, _target, **_kwargs):
        phase.parent_evidence.layers[0].tensors["operator.input"].fill_(9.0)
        source_table.fill_(7.0)
        raise RuntimeError("boundary fit rejected")

    monkeypatch.setattr(
        model_fold,
        "_compile_lfm2_cpcf_model_fold_mutating",
        _mutate_boundary_inputs,
    )
    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldInstallError,
        match="cached evidence",
    ):
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            target,
            phases=(phase,),
            source_tied_table=source_table,
            blank_target_report={},
            planned_target_unique_parameters=1,
        )

    assert torch.equal(target.scalar, torch.zeros_like(target.scalar))


def test_keyboard_interrupt_restores_target_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _TransactionalModel(2, 1.0)
    target = _TransactionalModel(1, 0.0)
    original = target.scalar

    def _interrupt(_source, live_target, **_kwargs):
        live_target.scalar = torch.nn.Parameter(torch.ones(1))
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        model_fold,
        "_compile_lfm2_cpcf_model_fold_mutating",
        _interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            target,
            phases=(),
            source_tied_table=None,
            blank_target_report={},
            planned_target_unique_parameters=1,
        )

    assert target.scalar is original
    assert torch.equal(target.scalar, torch.zeros_like(target.scalar))


def test_parameter_receipt_allows_any_fold_but_marks_promotion() -> None:
    source = _TiedModel(128, 16)
    target = _TiedModel(8, 8)
    target_count = 8 * 8
    diagnostic = audit_lfm2_cpcf_parameter_compaction(
        source,
        target,
        planned_target_unique_parameters=target_count,
        promotion_compression_ratio=100.0,
    )
    assert diagnostic["realized_compression_ratio"] == 32.0
    assert diagnostic["promotion_ratio_met"] is False
    assert diagnostic["valid_lower_ratio_diagnostic_fold"] is True
    assert diagnostic["quantization_used"] is False


def test_parameter_receipt_rejects_noncompact_or_wrong_plan() -> None:
    source = _TiedModel(16, 8)
    target = _TiedModel(16, 8)
    with pytest.raises(Lfm2CpcfModelFoldError, match="does not compact"):
        audit_lfm2_cpcf_parameter_compaction(
            source,
            target,
            planned_target_unique_parameters=16 * 8,
        )
    smaller = _TiedModel(8, 8)
    with pytest.raises(Lfm2CpcfModelFoldError, match="catalog plan"):
        audit_lfm2_cpcf_parameter_compaction(
            source,
            smaller,
            planned_target_unique_parameters=63,
        )


def test_model_fold_policy_keeps_generic_budget_separate_from_promotion() -> None:
    generic = Lfm2CpcfModelFoldPolicy()
    v3_document = {
        "ridge": 1e-8,
        "state_reconstruction_weight": 0.05,
        "minimum_predictive_energy_fraction": 0.90,
        "minimum_validation_block_retention": 0.0,
        "enforce_validation_block_floor": True,
        "short_conv_gate_policy": {
            "maximum_fit_endpoint_relative_rmse": 1.0,
            "maximum_validation_endpoint_relative_rmse": 1.0,
            "minimum_fit_endpoint_cosine_similarity": 0.0,
            "minimum_validation_endpoint_cosine_similarity": 0.0,
            "enforce": True,
        },
        "promotion_compression_ratio": 1.0,
        "require_promotion_ratio": False,
    }
    production = Lfm2CpcfModelFoldPolicy(
        promotion_compression_ratio=100.0,
        require_promotion_ratio=True,
    )
    assert generic.promotion_compression_ratio == 1.0
    assert generic.require_promotion_ratio is False
    assert production.promotion_compression_ratio == 100.0
    assert production.require_promotion_ratio is True
    assert production.resolved_short_conv_policy().enforce is True
    assert generic.as_dict() == v3_document
    assert "short_conv_v4_solver_policy" not in generic.as_dict()

    solver_policy = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-8, 1e-4),
        variable_projection_passes=2,
        target_dtype="bfloat16",
    )
    v4 = Lfm2CpcfModelFoldPolicy(
        short_conv_v4_solver_policy=solver_policy,
    )
    assert v4.short_conv_v4_solver_policy is solver_policy
    assert v4.as_dict()["short_conv_v4_solver_policy"] == solver_policy.as_dict()
    assert (
        v4.as_dict()["short_conv_v4_solver_policy"]["authentication_sha256"]
        == solver_policy.authentication_sha256
    )
    assert solver_policy.verify_authentication() is True
    with pytest.raises(TypeError, match="short_conv_v4_solver_policy"):
        Lfm2CpcfModelFoldPolicy(short_conv_v4_solver_policy=object())

    attention_policy = (
        layer_compiler.Lfm2CpcfAttentionSequentialTransportPolicy(
            enabled=True
        )
    )
    attention = Lfm2CpcfModelFoldPolicy(
        attention_sequential_transport_policy=attention_policy,
    )
    assert (
        attention.resolved_attention_sequential_transport_policy(
            "full_attention"
        )
        is attention_policy
    )
    assert (
        attention.resolved_attention_sequential_transport_policy(
            "short_conv"
        )
        is None
    )
    assert attention.as_dict()[
        "attention_sequential_transport_policy"
    ] == attention_policy.as_dict()
    disabled_attention = Lfm2CpcfModelFoldPolicy(
        attention_sequential_transport_policy=(
            layer_compiler.Lfm2CpcfAttentionSequentialTransportPolicy(
                enabled=False
            )
        ),
    )
    assert disabled_attention.as_dict() == generic.as_dict()
    assert (
        disabled_attention.resolved_attention_sequential_transport_policy(
            "full_attention"
        )
        is None
    )
    with pytest.raises(TypeError, match="attention_sequential_transport_policy"):
        Lfm2CpcfModelFoldPolicy(
            attention_sequential_transport_policy=object(),
        )
    with pytest.raises(
        Lfm2CpcfModelFoldError,
        match="canonical operator kind",
    ):
        attention.resolved_attention_sequential_transport_policy("unknown")


def test_model_fold_passes_one_authenticated_v4_policy_to_each_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase, _bank = _real_phase_bank()
    source = _TransactionalModel(2, 1.0)
    target = _TransactionalModel(1, 0.0)
    solver_policy = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-8, 1e-4),
        variable_projection_passes=2,
        target_dtype="bfloat16",
    )
    policy = Lfm2CpcfModelFoldPolicy(
        short_conv_v4_solver_policy=solver_policy,
        attention_sequential_transport_policy=(
            layer_compiler.Lfm2CpcfAttentionSequentialTransportPolicy(
                enabled=True
            )
        ),
    )
    charts = SimpleNamespace(operator_kind="short_conv")
    boundary = SimpleNamespace(
        interface=object(),
        layers=(charts,),
        bundle_sha256="2" * 64,
    )
    state_bank = SimpleNamespace(layer_index=0)
    observed = {}

    class _ReachedLayer(RuntimeError):
        pass

    def _layer_spy(*_args, **kwargs):
        observed.update(kwargs)
        raise _ReachedLayer("layer dispatch reached")

    monkeypatch.setattr(
        model_fold,
        "independent_parameter_inventory",
        lambda model: {
            "unique_parameters": 2 if model is source else 1,
        },
    )
    monkeypatch.setattr(
        model_fold,
        "_blank_receipt",
        lambda *_args, **_kwargs: {"report_sha256": "3" * 64},
    )
    monkeypatch.setattr(
        model_fold,
        "_assert_target_is_zero_blank",
        lambda _model: None,
    )
    monkeypatch.setattr(
        model_fold,
        "fingerprint_model_state",
        lambda _model: "4" * 64,
    )
    monkeypatch.setattr(
        model_fold,
        "audit_lfm2_cpcf_parameter_compaction",
        lambda *_args, **_kwargs: {"promotion_ratio_met": True},
    )
    monkeypatch.setattr(
        model_fold,
        "fit_lfm2_cpcf_boundary_bundle",
        lambda *_args, **_kwargs: boundary,
    )
    monkeypatch.setattr(
        model_fold,
        "_boundary_bundle_inventory_sha256",
        lambda _bundle: "5" * 64,
    )
    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_tied_interface",
        lambda *_args, **_kwargs: {"report_sha256": "6" * 64},
    )
    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_neutral_norms",
        lambda *_args, **_kwargs: {"report_sha256": "7" * 64},
    )
    monkeypatch.setattr(
        model_fold,
        "_preserved_qk_norm_weights",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        model_fold,
        "seed_lfm2_cpcf_phase_states",
        lambda *_args, **_kwargs: state_bank,
    )
    monkeypatch.setattr(
        model_fold,
        "compile_and_install_lfm2_cpcf_layer",
        _layer_spy,
    )

    with pytest.raises(_ReachedLayer, match="layer dispatch reached"):
        model_fold._compile_lfm2_cpcf_model_fold_mutating(
            source,
            target,
            phases=(phase,),
            source_tied_table=torch.ones(3, 2),
            blank_target_report={},
            planned_target_unique_parameters=1,
            policy=policy,
            progress=model_fold._ModelFoldProgress(),
        )

    assert observed["short_conv_v4_solver_policy"] is solver_policy
    assert observed["short_conv_gate_policy"] == policy.resolved_short_conv_policy()
    assert observed["attention_sequential_transport_policy"] is None
    assert observed["layer_index"] == 0


def test_whole_model_failure_restores_blank_gradient_and_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _TransactionalModel(2, 1.0)
    target = _TransactionalModel(1, 0.0)
    target.train()
    original_gradient = torch.tensor([7.0])
    target.scalar.grad = original_gradient
    target.register_buffer(
        "rotary_inv_freq",
        torch.tensor([2.0]),
        persistent=False,
    )

    def _mutate_then_fail(_source, model, **_kwargs):
        with torch.no_grad():
            model.scalar.fill_(9.0)
            model.rotary_inv_freq.fill_(11.0)
        model.eval()
        model.scalar.requires_grad_(False)
        model.scalar.grad = None
        raise RuntimeError("late layer failure")

    monkeypatch.setattr(
        model_fold,
        "_compile_lfm2_cpcf_model_fold_mutating",
        _mutate_then_fail,
    )
    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldFailure,
        match="late layer failure",
    ) as caught:
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            target,
            phases=(),
            source_tied_table=None,
            blank_target_report={},
            planned_target_unique_parameters=1,
        )
    failure = caught.value
    assert isinstance(failure.__cause__, RuntimeError)
    assert failure.stage == "validate_request"
    assert failure.boundary_bundle is None
    assert failure.current_incoming_bank is None
    assert failure.completed_layer_receipts == ()
    assert torch.equal(
        failure.prefix_target_cpu_tensor_snapshot["scalar"],
        torch.tensor([9.0]),
    )
    assert torch.equal(
        failure.prefix_target_cpu_tensor_snapshot["rotary_inv_freq"],
        torch.tensor([11.0]),
    )
    assert (
        failure.prefix_target_snapshot_metadata["tensor_metadata"]["rotary_inv_freq"][
            "persistent"
        ]
        is False
    )
    assert (
        failure.report["failed_prefix_target_cpu_tensor_snapshot"][
            "nonpersistent_buffers_included"
        ]
        is True
    )
    assert failure.rollback_proof["exact_whole_model_rollback_proven"] is True
    assert failure.report["compiled_model_fold_issued"] is False
    assert failure.report["parameter_compaction_success_receipt_issued"] is False
    assert failure.report["promotion_claimed"] is False
    assert failure.report["promotable_candidate"] is False
    assert torch.equal(target.scalar, torch.zeros_like(target.scalar))
    assert target.scalar.requires_grad is True
    assert target.scalar.grad is original_gradient
    assert torch.equal(target.scalar.grad, torch.tensor([7.0]))
    assert target.training is True
    assert torch.equal(target.rotary_inv_freq, torch.tensor([2.0]))
    mutable_snapshot = failure.prefix_target_cpu_tensor_snapshot
    mutable_snapshot["scalar"].fill_(123.0)
    assert torch.equal(
        failure.prefix_target_cpu_tensor_snapshot["scalar"],
        torch.tensor([9.0]),
    )
    mutable_report = failure.report
    mutable_report["rollback_proof"]["exact_whole_model_rollback_proven"] = False
    assert failure.report["rollback_proof"]["exact_whole_model_rollback_proven"] is True
    forged_report = failure.report
    forged_report["promotable_candidate"] = True
    forged_body = dict(forged_report)
    forged_body.pop("report_sha256")
    forged_report["report_sha256"] = model_fold._sha(forged_body)
    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldError,
        match="claims success",
    ):
        model_fold.Lfm2CpcfModelFoldFailure(
            "forged success",
            stage=failure.stage,
            boundary_bundle=failure.boundary_bundle,
            current_incoming_bank=failure.current_incoming_bank,
            completed_layer_receipts=(failure.completed_layer_receipts),
            prefix_target_cpu_tensor_snapshot=(
                failure.prefix_target_cpu_tensor_snapshot
            ),
            prefix_target_snapshot_metadata=(failure.prefix_target_snapshot_metadata),
            rollback_proof=failure.rollback_proof,
            report=forged_report,
            layer_failure_report=failure.layer_failure_report,
            layer_failure_report_bytes=(failure.layer_failure_report_bytes),
        )


def test_failure_incoming_bank_owns_only_child_states_and_parent_references() -> None:
    phase, bank = _real_phase_bank()
    snapshot = model_fold._snapshot_incoming_bank_evidence(bank)

    assert isinstance(
        snapshot,
        model_fold.Lfm2CpcfFailureIncomingBankSnapshot,
    )
    assert snapshot.state_bank_sha256 == bank.state_bank_sha256
    assert snapshot.phase_identities == (phase.identity,)
    assert snapshot.as_dict()["parent_evidence_tensors_copied"] is False
    reference = snapshot.phase_references[0]
    assert reference["parent_evidence_reference"]["cache_key"] == (
        phase.parent_evidence.cache_key
    )
    assert "layers" not in reference["parent_evidence_reference"]
    child_state = snapshot.state(phase.identity)
    assert child_state.data_ptr() != bank.states[phase.identity].data_ptr()
    bank.states[phase.identity].fill_(19.0)
    phase.parent_evidence.layers[0].tensors["operator.input"].fill_(77.0)
    assert torch.equal(
        snapshot.state(phase.identity),
        torch.zeros_like(child_state),
    )
    assert (
        snapshot.phase_references[0]["parent_evidence_reference"]["evidence_sha256"]
        == phase.parent_evidence.evidence_sha256
    )


def test_prefix_snapshot_failure_cannot_skip_whole_model_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _TransactionalModel(2, 1.0)
    target = _TransactionalModel(1, 0.0)
    target.train()
    original_gradient = torch.tensor([7.0])
    target.scalar.grad = original_gradient
    target.register_buffer(
        "rotary_inv_freq",
        torch.tensor([2.0]),
        persistent=False,
    )

    def _mutate_then_fail(
        _source,
        live_target,
        **_kwargs,
    ):
        with torch.no_grad():
            live_target.scalar.fill_(9.0)
            live_target.rotary_inv_freq.fill_(11.0)
        live_target.eval()
        live_target.scalar.requires_grad_(False)
        live_target.scalar.grad = None
        raise RuntimeError("fold rejected")

    monkeypatch.setattr(
        model_fold,
        "_compile_lfm2_cpcf_model_fold_mutating",
        _mutate_then_fail,
    )
    original_snapshot = model_fold._snapshot_model
    snapshot_calls = 0

    def _fail_second_snapshot(model):
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 2:
            raise MemoryError("diagnostic snapshot allocation failed")
        return original_snapshot(model)

    monkeypatch.setattr(
        model_fold,
        "_snapshot_model",
        _fail_second_snapshot,
    )
    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldInstallError,
        match="diagnostic snapshot",
    ) as caught:
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            target,
            phases=(),
            source_tied_table=None,
            blank_target_report={},
            planned_target_unique_parameters=1,
        )

    assert isinstance(caught.value.__cause__, MemoryError)
    assert torch.equal(target.scalar, torch.zeros_like(target.scalar))
    assert target.scalar.requires_grad is True
    assert target.scalar.grad is original_gradient
    assert torch.equal(target.scalar.grad, torch.tensor([7.0]))
    assert target.training is True
    assert torch.equal(target.rotary_inv_freq, torch.tensor([2.0]))


def test_whole_model_proof_rejects_source_gradient_or_mode_mutation() -> None:
    source = _TransactionalModel(2, 1.0)
    target = _TransactionalModel(1, 0.0)
    source_snapshot = model_fold._snapshot_model_auth(source)
    snapshot = model_fold._snapshot_model(target)
    source.eval()
    source.scalar.grad = torch.ones_like(source.scalar)

    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldInstallError,
        match="gradients",
    ):
        model_fold._prove_exact_whole_model_rollback(
            source,
            target,
            source_snapshot_before=source_snapshot,
            target_snapshot_before=snapshot,
        )


def test_production_failure_clone_rejects_synthetic_namespace() -> None:
    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldError,
        match="typed bundle",
    ):
        model_fold._clone_boundary_bundle_evidence(
            SimpleNamespace(bundle_sha256="1" * 64)
        )


def test_authenticated_resume_matches_cold_result_and_target_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = _resume_phases()
    bundle = _resume_bundle(phases)
    banks = (
        _resume_bank(phases, layer_index=0, state_value=0.0),
        _resume_bank(phases, layer_index=1, state_value=1.0),
        _resume_bank(phases, layer_index=2, state_value=2.0),
    )
    source = _TransactionalModel(3, 4.0, depth=2)
    cold_target = _TransactionalModel(2, 0.0, depth=2)
    policy = Lfm2CpcfModelFoldPolicy()
    source_table = torch.ones(3, 3)
    blank_report = _blank_report(2)
    fit_calls = []
    layer_calls = []
    prefix_scalar = {}

    def _fit_bundle(*_args, **_kwargs):
        fit_calls.append(True)
        return bundle

    def _compile_layer(_source, target, **kwargs):
        index = kwargs["layer_index"]
        layer_calls.append(index)
        assert kwargs["input_bank"].state_bank_sha256 == (
            banks[index].state_bank_sha256
        )
        with torch.no_grad():
            target.scalar.add_(index + 1)
        if index == 0:
            prefix_scalar["value"] = target.scalar.detach().clone()
        receipt = _resume_layer_receipt(
            layer_index=index,
            charts=bundle.layers[index],
            input_bank_sha256=banks[index].state_bank_sha256,
            next_bank_sha256=banks[index + 1].state_bank_sha256,
            phases=phases,
        )
        return _synthetic_compiled_layer(receipt, banks[index + 1])

    monkeypatch.setattr(
        model_fold,
        "fit_lfm2_cpcf_boundary_bundle",
        _fit_bundle,
    )
    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_tied_interface",
        lambda *_args, **_kwargs: {
            "report_sha256": _resume_digest("interface-install")
        },
    )
    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_neutral_norms",
        lambda *_args, **_kwargs: {
            "report_sha256": _resume_digest("neutral-norm-install")
        },
    )
    monkeypatch.setattr(
        model_fold,
        "_preserved_qk_norm_weights",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        model_fold,
        "seed_lfm2_cpcf_phase_states",
        lambda *_args, **_kwargs: banks[0],
    )
    monkeypatch.setattr(
        model_fold,
        "compile_and_install_lfm2_cpcf_layer",
        _compile_layer,
    )
    monkeypatch.setattr(
        model_fold,
        "_fit_and_install_final_norm",
        lambda *_args, **_kwargs: {"report_sha256": _resume_digest("final-norm")},
    )

    cold = model_fold.compile_lfm2_cpcf_model_fold(
        source,
        cold_target,
        phases=phases,
        source_tied_table=source_table,
        blank_target_report=blank_report,
        planned_target_unique_parameters=2,
        policy=policy,
    )
    cold_state_sha = model_fold.fingerprint_model_state(cold_target)
    observed_events = []
    observed_target = _TransactionalModel(2, 0.0, depth=2)
    observed = model_fold.compile_lfm2_cpcf_model_fold(
        source,
        observed_target,
        phases=phases,
        source_tied_table=source_table,
        blank_target_report=blank_report,
        planned_target_unique_parameters=2,
        policy=policy,
        progress_observer=observed_events.append,
    )
    resumed_target = _TransactionalModel(2, 0.0, depth=2)
    with torch.no_grad():
        resumed_target.scalar.copy_(prefix_scalar["value"])
    resume = Lfm2CpcfModelFoldResumeState(
        prefix_install_report=_resume_prefix_install_report(
            resumed_target,
            next_layer_index=1,
        ),
        full_policy_sha256=model_fold._sha(policy.as_dict()),
        boundary_bundle=bundle,
        incoming_state_bank=banks[1],
        completed_layer_receipts=(cold.layer_results[0].as_dict(),),
    )
    resumed_events = []
    resumed = model_fold.compile_lfm2_cpcf_model_fold(
        source,
        resumed_target,
        phases=phases,
        source_tied_table=source_table,
        blank_target_report=blank_report,
        planned_target_unique_parameters=2,
        policy=policy,
        resume_state=resume,
        progress_observer=resumed_events.append,
    )

    assert fit_calls == [True, True]
    assert layer_calls == [0, 1, 0, 1, 1]
    assert [event.next_layer_index for event in observed_events] == [0, 1, 2]
    assert [event.next_layer_index for event in resumed_events] == [2]
    assert all(
        event.stage == "progress_checkpoint_publish"
        and event.report["parent_evidence_tensors_copied"] == 0
        and event.report["live_target_model_exposed"] is False
        for event in (*observed_events, *resumed_events)
    )
    assert [
        int(event.target_prefix_cpu_tensor_snapshot["scalar"][0])
        for event in observed_events
    ] == [0, 1, 3]
    observed_events[1].target_prefix_cpu_tensor_snapshot["scalar"].fill_(88.0)
    observed_events[1].verify()
    assert isinstance(
        resumed.layer_results[0],
        Lfm2CpcfCompletedLayerReceipt,
    )
    assert model_fold._canonical_json(observed.as_dict()) == (
        model_fold._canonical_json(cold.as_dict())
    )
    assert model_fold._canonical_json(resumed.as_dict()) == (
        model_fold._canonical_json(cold.as_dict())
    )
    assert resumed.result_sha256 == cold.result_sha256
    assert resumed.final_state_bank.state_bank_sha256 == (
        cold.final_state_bank.state_bank_sha256
    )
    assert model_fold.fingerprint_model_state(resumed_target) == cold_state_sha
    assert model_fold.fingerprint_model_state(observed_target) == cold_state_sha
    assert torch.equal(resumed_target.scalar, cold_target.scalar)


def test_resume_rejects_wrong_policy() -> None:
    case = _valid_resume_case()
    wrong_policy = Lfm2CpcfModelFoldPolicy(ridge=1e-5)

    with pytest.raises(Lfm2CpcfModelFoldError, match="policy SHA-256"):
        model_fold._authenticate_model_fold_resume_state(
            case.resume,
            target_model=case.target,
            phases=case.phases,
            policy=wrong_policy,
        )


def test_resume_rejects_wrong_boundary_bundle_lineage() -> None:
    case = _valid_resume_case()
    wrong_bundle = _resume_bundle(
        case.phases,
        wrong_phase_lineage=True,
    )
    forged = Lfm2CpcfModelFoldResumeState(
        prefix_install_report=case.resume.prefix_install_report,
        full_policy_sha256=case.resume.full_policy_sha256,
        boundary_bundle=wrong_bundle,
        incoming_state_bank=case.incoming_bank,
        completed_layer_receipts=(case.receipt,),
    )

    with pytest.raises(Lfm2CpcfModelFoldError, match="phase lineage"):
        model_fold._authenticate_model_fold_resume_state(
            forged,
            target_model=case.target,
            phases=case.phases,
            policy=case.policy,
        )


def test_resume_rejects_wrong_incoming_bank_chain() -> None:
    case = _valid_resume_case()
    wrong_bank = _resume_bank(
        case.phases,
        layer_index=1,
        state_value=9.0,
    )
    forged = Lfm2CpcfModelFoldResumeState(
        prefix_install_report=case.resume.prefix_install_report,
        full_policy_sha256=case.resume.full_policy_sha256,
        boundary_bundle=case.bundle,
        incoming_state_bank=wrong_bank,
        completed_layer_receipts=(case.receipt,),
    )
    authenticated = model_fold._authenticate_model_fold_resume_state(
        forged,
        target_model=case.target,
        phases=case.phases,
        policy=case.policy,
    )

    with pytest.raises(Lfm2CpcfModelFoldError, match="does not end"):
        model_fold._validate_resume_receipt_chain(
            authenticated,
            initial_state_bank=case.initial_bank,
            phases=case.phases,
        )


def test_resume_rejects_live_prefix_byte_mismatch() -> None:
    case = _valid_resume_case()
    with torch.no_grad():
        case.target.scalar.add_(1.0)

    with pytest.raises(Lfm2CpcfModelFoldError, match="installed prefix bytes"):
        model_fold._authenticate_model_fold_resume_state(
            case.resume,
            target_model=case.target,
            phases=case.phases,
            policy=case.policy,
        )


def test_resume_rejects_wrong_completed_receipt_chain() -> None:
    case = _valid_resume_case()
    wrong_receipt = _resume_layer_receipt(
        layer_index=0,
        charts=case.bundle.layers[0],
        input_bank_sha256=_resume_digest("wrong-input-bank"),
        next_bank_sha256=case.incoming_bank.state_bank_sha256,
        phases=case.phases,
    )
    forged = Lfm2CpcfModelFoldResumeState(
        prefix_install_report=case.resume.prefix_install_report,
        full_policy_sha256=case.resume.full_policy_sha256,
        boundary_bundle=case.bundle,
        incoming_state_bank=case.incoming_bank,
        completed_layer_receipts=(wrong_receipt,),
    )
    authenticated = model_fold._authenticate_model_fold_resume_state(
        forged,
        target_model=case.target,
        phases=case.phases,
        policy=case.policy,
    )

    with pytest.raises(Lfm2CpcfModelFoldError, match="causal state-bank"):
        model_fold._validate_resume_receipt_chain(
            authenticated,
            initial_state_bank=case.initial_bank,
            phases=case.phases,
        )


def test_resume_rejects_checkpoint_after_all_layers() -> None:
    case = _valid_resume_case()
    final_bank = _resume_bank(
        case.phases,
        layer_index=2,
        state_value=2.0,
    )
    second_receipt = _resume_layer_receipt(
        layer_index=1,
        charts=case.bundle.layers[1],
        input_bank_sha256=case.incoming_bank.state_bank_sha256,
        next_bank_sha256=final_bank.state_bank_sha256,
        phases=case.phases,
    )
    target = _TransactionalModel(2, 3.0, depth=2)

    with pytest.raises(Lfm2CpcfModelFoldError, match="no unfinished"):
        Lfm2CpcfModelFoldResumeState(
            prefix_install_report=_resume_prefix_install_report(
                target,
                next_layer_index=2,
            ),
            full_policy_sha256=case.resume.full_policy_sha256,
            boundary_bundle=case.bundle,
            incoming_state_bank=final_bank,
            completed_layer_receipts=(case.receipt, second_receipt),
        )


def test_resumed_failure_restores_prefix_and_keeps_prior_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_resume_case()
    source = _TransactionalModel(3, 4.0, depth=2)
    prefix_scalar = case.target.scalar.detach().clone()
    prefix_sha = model_fold.fingerprint_model_state(case.target)

    monkeypatch.setattr(
        model_fold,
        "fit_lfm2_cpcf_boundary_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "resume must not refit the boundary bundle"
        ),
    )
    monkeypatch.setattr(
        model_fold,
        "seed_lfm2_cpcf_phase_states",
        lambda *_args, **_kwargs: case.initial_bank,
    )
    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_tied_interface",
        lambda *_args, **_kwargs: {
            "report_sha256": _resume_digest("interface-install")
        },
    )
    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_neutral_norms",
        lambda *_args, **_kwargs: {
            "report_sha256": _resume_digest("neutral-norm-install")
        },
    )
    monkeypatch.setattr(
        model_fold,
        "_preserved_qk_norm_weights",
        lambda *_args, **_kwargs: {},
    )

    def _fail_layer(_source, target, **_kwargs):
        with torch.no_grad():
            target.scalar.fill_(99.0)
        raise RuntimeError("synthetic resumed layer failure")

    monkeypatch.setattr(
        model_fold,
        "compile_and_install_lfm2_cpcf_layer",
        _fail_layer,
    )

    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldFailure,
        match="synthetic resumed layer failure",
    ) as caught:
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            case.target,
            phases=case.phases,
            source_tied_table=torch.ones(3, 3),
            blank_target_report=_blank_report(2),
            planned_target_unique_parameters=2,
            policy=case.policy,
            resume_state=case.resume,
        )

    failure = caught.value
    assert model_fold.fingerprint_model_state(case.target) == prefix_sha
    assert torch.equal(case.target.scalar, prefix_scalar)
    assert failure.report["completed_layer_count"] == 1
    assert len(failure.completed_layer_receipts) == 1
    assert (
        failure.completed_layer_receipts[0]["result_sha256"]
        == (case.receipt["result_sha256"])
    )
    assert failure.current_incoming_bank.state_bank_sha256 == (
        case.incoming_bank.state_bank_sha256
    )


def test_progress_observer_exception_is_structured_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = _resume_phases()
    bundle = _resume_bundle(phases)
    banks = (
        _resume_bank(phases, layer_index=0, state_value=0.0),
        _resume_bank(phases, layer_index=1, state_value=1.0),
        _resume_bank(phases, layer_index=2, state_value=2.0),
    )
    layer_calls = _patch_progress_fold_harness(
        monkeypatch,
        phases=phases,
        bundle=bundle,
        banks=banks,
    )
    source = _TransactionalModel(3, 4.0, depth=2)
    target = _TransactionalModel(2, 0.0, depth=2)
    blank_sha = model_fold.fingerprint_model_state(target)
    events = []

    def _observer(event):
        event.verify()
        events.append(event)
        if event.next_layer_index == 1:
            raise RuntimeError("checkpoint publisher unavailable")

    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldFailure,
        match="checkpoint publisher unavailable",
    ) as caught:
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            target,
            phases=phases,
            source_tied_table=torch.ones(3, 3),
            blank_target_report=_blank_report(2),
            planned_target_unique_parameters=2,
            progress_observer=_observer,
        )

    failure = caught.value
    assert [event.next_layer_index for event in events] == [0, 1]
    assert layer_calls == [0]
    assert failure.stage == "progress_checkpoint_publish"
    assert failure.report["active_layer_index"] is None
    assert failure.report["active_operator_kind"] is None
    assert failure.report["completed_layer_count"] == 1
    assert (
        failure.completed_layer_receipts[0]["result_sha256"]
        == (events[-1].completed_layer_receipts[0]["result_sha256"])
    )
    assert model_fold.fingerprint_model_state(target) == blank_sha
    assert torch.equal(target.scalar, torch.zeros_like(target.scalar))


def test_progress_observer_cancellation_rolls_back_then_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = _resume_phases()
    bundle = _resume_bundle(phases)
    banks = (
        _resume_bank(phases, layer_index=0, state_value=0.0),
        _resume_bank(phases, layer_index=1, state_value=1.0),
        _resume_bank(phases, layer_index=2, state_value=2.0),
    )
    layer_calls = _patch_progress_fold_harness(
        monkeypatch,
        phases=phases,
        bundle=bundle,
        banks=banks,
    )
    source = _TransactionalModel(3, 4.0, depth=2)
    target = _TransactionalModel(2, 0.0, depth=2)
    blank_sha = model_fold.fingerprint_model_state(target)

    def _cancel(event):
        assert event.next_layer_index == 0
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            target,
            phases=phases,
            source_tied_table=torch.ones(3, 3),
            blank_target_report=_blank_report(2),
            planned_target_unique_parameters=2,
            progress_observer=_cancel,
        )

    assert layer_calls == []
    assert model_fold.fingerprint_model_state(target) == blank_sha
    assert torch.equal(target.scalar, torch.zeros_like(target.scalar))


def test_progress_event_tampering_and_observer_target_mutation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = _resume_phases()
    bundle = _resume_bundle(phases)
    banks = (
        _resume_bank(phases, layer_index=0, state_value=0.0),
        _resume_bank(phases, layer_index=1, state_value=1.0),
        _resume_bank(phases, layer_index=2, state_value=2.0),
    )
    _patch_progress_fold_harness(
        monkeypatch,
        phases=phases,
        bundle=bundle,
        banks=banks,
    )
    source = _TransactionalModel(3, 4.0, depth=2)
    target = _TransactionalModel(2, 0.0, depth=2)
    blank_sha = model_fold.fingerprint_model_state(target)

    def _tamper(event):
        event._target_prefix_cpu_tensor_snapshot["scalar"].fill_(12.0)
        with torch.no_grad():
            target.scalar.fill_(13.0)

    with pytest.raises(
        model_fold.Lfm2CpcfModelFoldFailure,
        match="prefix evidence or seal",
    ) as caught:
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            target,
            phases=phases,
            source_tied_table=torch.ones(3, 3),
            blank_target_report=_blank_report(2),
            planned_target_unique_parameters=2,
            progress_observer=_tamper,
        )

    assert caught.value.stage == "progress_checkpoint_publish"
    assert model_fold.fingerprint_model_state(target) == blank_sha
    assert torch.equal(target.scalar, torch.zeros_like(target.scalar))
