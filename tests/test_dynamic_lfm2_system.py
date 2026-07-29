from __future__ import annotations

import json
from dataclasses import replace

import pytest

from neural_forge.dynamic_behavior_gate import (
    BehaviorGateBinding,
    expose_behavior_gate_binding,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_artifact import DynamicFoldArtifactError
from neural_forge.dynamic_fold_compiler import DynamicFoldCompilerError
from neural_forge.dynamic_fold_executor import FoldDecision, fingerprint_model_state
from neural_forge.dynamic_fold_registry import ExperimentLedger
from neural_forge.dynamic_lfm2_adapter import (
    DynamicLfm2AdapterError,
    Lfm2PreparedEvidence,
    Lfm2TargetSynthesisAdapter,
)
from neural_forge.dynamic_lfm2_system import (
    Lfm2DynamicFoldStackAdapter,
    Lfm2DynamicStackResources,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_lfm2_closed_loop_fold import (
    Lfm2ClosedLoopFoldConfig,
    Lfm2ClosedLoopFoldEvidence,
    Lfm2MLPFoldEvidence,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _model(seed: int, intermediate_size: int):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv"],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    return transformers.Lfm2ForCausalLM(config)


def _evidence(seed: int = 10) -> Lfm2ClosedLoopFoldEvidence:
    generator = torch.Generator().manual_seed(seed)
    selection_rows = 24
    fit_rows = 32
    gate = torch.randn(
        selection_rows,
        6,
        generator=generator,
        dtype=torch.float64,
    )
    up = torch.randn(
        selection_rows,
        6,
        generator=generator,
        dtype=torch.float64,
    )
    post = torch.nn.functional.silu(gate) * up
    contribution = torch.randn(
        selection_rows,
        6,
        8,
        generator=generator,
        dtype=torch.float64,
    )
    child_post = torch.randn(
        fit_rows,
        5,
        generator=generator,
        dtype=torch.float64,
    )
    desired_down = torch.randn(
        8,
        5,
        generator=generator,
        dtype=torch.float64,
    )
    residual = torch.randn(
        fit_rows,
        8,
        generator=generator,
        dtype=torch.float64,
    )
    return Lfm2ClosedLoopFoldEvidence(
        mlp=Lfm2MLPFoldEvidence(
            parent_gate_rows=gate,
            parent_up_rows=up,
            parent_post_swiglu_rows=post,
            parent_contribution_rows=contribution,
            child_post_swiglu_rows=child_post,
            parent_branch_output_rows=(residual + child_post @ desired_down.T),
            child_pre_branch_residual_rows=residual,
        )
    )


class _Accept:
    def evaluate(self, context):
        with torch.no_grad():
            context.execution_context.child_model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return FoldDecision.accept("whole-output LFM catalog test gate")


class _MutatingAccept:
    def evaluate(self, context):
        with torch.no_grad():
            context.execution_context.child_model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
            next(context.execution_context.child_model.parameters()).add_(1.0)
        return FoldDecision.accept("must be rejected before catalog closure")


def _saver(model, destination):
    torch.save(model.state_dict(), destination / "model.pt")
    (destination / "config.json").write_text(
        json.dumps(model.config.to_dict()),
        encoding="utf-8",
    )


def _loader(source):
    payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config = transformers.Lfm2Config(**payload)
    if "block_ff_dim" in payload:
        config.block_ff_dim = payload["block_ff_dim"]
    model = transformers.Lfm2ForCausalLM(config)
    model.load_state_dict(
        torch.load(
            source / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    return model


def _catalog(
    parent,
    tokenizer,
    ledger,
    final_evaluator=None,
    fresh_reload_evaluator=None,
):
    cache_payload = {"cached_parent_rows": True}
    cache_fingerprint = canonical_sha256(cache_payload)
    evidence = _evidence()

    def evidence_builder(_request):
        return Lfm2PreparedEvidence(
            evidence=evidence,
            child_forward_count=0,
        )

    final_gate = _Accept() if final_evaluator is None else final_evaluator
    if fresh_reload_evaluator is None:

        def fresh_gate(model):
            with torch.no_grad():
                model(
                    input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                    use_cache=False,
                )
            return FoldDecision.accept(
                f"fresh local reload accepted: {model.config.model_type}"
            )

    else:
        fresh_gate = fresh_reload_evaluator
    manifest_body = {
        "schema": "test-behavior-gate-manifest-v1",
        "gate_id": "test.lfm2.whole_output",
        "gate_version": 1,
        "baseline_fingerprint": cache_fingerprint,
    }
    manifest_fingerprint = canonical_sha256(manifest_body)
    behavior_binding = BehaviorGateBinding(
        gate_id="test.lfm2.whole_output",
        gate_version=1,
        baseline_fingerprint=cache_fingerprint,
        manifest_fingerprint=manifest_fingerprint,
        manifest={
            **manifest_body,
            "manifest_fingerprint": manifest_fingerprint,
        },
    )
    expose_behavior_gate_binding(final_gate, behavior_binding)
    expose_behavior_gate_binding(fresh_gate, behavior_binding)

    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=cache_fingerprint,
        ledger=ledger,
    )
    stack_adapter = Lfm2DynamicFoldStackAdapter(
        Lfm2DynamicStackResources(
            parent_cache_fingerprint=cache_fingerprint,
            parent_cache_payload=cache_payload,
            evidence_builder=evidence_builder,
            final_evaluator=final_gate,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=fresh_gate,
            saver_id="test-local-lfm-saver-v1",
            loader_id="test-local-lfm-loader-v1",
            ledger=ledger,
            fold_config=Lfm2ClosedLoopFoldConfig(
                endpoint_ridge=0.0,
                max_endpoint_relative_rmse=1e-9,
                min_endpoint_cosine_similarity=0.999999999,
            ),
        )
    )
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    return target_adapter, stack_adapter, cache_fingerprint, graph


def test_automatic_lfm_catalog_synthesizes_folds_and_reloads(tmp_path) -> None:
    parent = _model(1, 6)
    tokenizer = canonical_sha256({"tokenizer": "local-lfm"})
    ledger = ExperimentLedger()
    target_adapter, stack_adapter, evidence_fingerprint, graph = _catalog(
        parent,
        tokenizer,
        ledger,
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
        ledger=ledger,
    )

    result = system.run(
        parent_models={"parent": parent},
        target_request=DynamicTargetRequest(
            child_name="automatic-lfm-child",
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=graph.unique_parameter_count - 24,
            tokenizer_fingerprint=tokenizer,
            blank_seed=2,
        ),
        evidence_fingerprint=evidence_fingerprint,
        destination=tmp_path / "published-lfm",
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert result.complete
    assert result.target.adapter_id == target_adapter.adapter_id
    assert result.target.adapter_report["hub_loader_used"] is False
    assert any(
        action.method_id == "lfm2_closed_loop_layer"
        for action in result.published.compilation.plan.actions
    )
    assert result.published.artifact.fresh_reload_decision.accepted
    assert ledger.decision(result.target.adapter_payload.signature).allowed is False
    closure = ledger.records[-1]
    assert closure.reason_class == "whole_output_behavior_and_fresh_reload_accept"
    assert dict(closure.metrics)["fresh_reload_behavior_accepted"] is True
    assert dict(closure.metrics)["artifact_certificate_sha256"] == (
        result.published.artifact.certificate_sha256
    )
    reloaded = _loader(tmp_path / "published-lfm" / "child")
    assert reloaded.config.intermediate_size == 5
    assert fingerprint_model_state(reloaded) == (
        result.published.artifact.child_state_fingerprint
    )


def test_lfm_catalog_stack_rejects_tampered_target_payload() -> None:
    parent = _model(3, 6)
    tokenizer = canonical_sha256({"tokenizer": "local-lfm"})
    ledger = ExperimentLedger()
    target_adapter, stack_adapter, evidence_fingerprint, graph = _catalog(
        parent,
        tokenizer,
        ledger,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((target_adapter,)),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="automatic-lfm-child",
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=graph.unique_parameter_count - 24,
            tokenizer_fingerprint=tokenizer,
            blank_seed=4,
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    forged = replace(
        target,
        topology_fingerprint=canonical_sha256({"forged": True}),
    )

    with pytest.raises(
        DynamicLfm2AdapterError,
        match="topology fingerprint",
    ):
        stack_adapter.build(forged)

    assert evidence_fingerprint == (
        target.adapter_payload.signature.evidence_fingerprint
    )


def test_lfm_catalog_rejects_run_evidence_different_from_cache(
    tmp_path,
) -> None:
    parent = _model(5, 6)
    tokenizer = canonical_sha256({"tokenizer": "local-lfm"})
    ledger = ExperimentLedger()
    target_adapter, stack_adapter, _evidence_fingerprint, graph = _catalog(
        parent,
        tokenizer,
        ledger,
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
        ledger=ledger,
    )

    with pytest.raises(
        DynamicFoldCompilerError,
        match="cached evidence fingerprint",
    ):
        system.run(
            parent_models={"parent": parent},
            target_request=DynamicTargetRequest(
                child_name="automatic-lfm-child",
                target_family="lfm2",
                target_modalities=("text",),
                exact_unique_parameters=graph.unique_parameter_count - 24,
                tokenizer_fingerprint=tokenizer,
                blank_seed=6,
            ),
            evidence_fingerprint=canonical_sha256({"different_cache": True}),
            destination=tmp_path / "must-not-publish",
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    assert not (tmp_path / "must-not-publish").exists()


def test_lfm_catalog_does_not_close_target_for_mutating_final_gate(
    tmp_path,
) -> None:
    parent = _model(7, 6)
    tokenizer = canonical_sha256({"tokenizer": "local-lfm"})
    ledger = ExperimentLedger()
    target_adapter, stack_adapter, evidence_fingerprint, graph = _catalog(
        parent,
        tokenizer,
        ledger,
        final_evaluator=_MutatingAccept(),
    )
    request = DynamicTargetRequest(
        child_name="automatic-lfm-child",
        target_family="lfm2",
        target_modalities=("text",),
        exact_unique_parameters=graph.unique_parameter_count - 24,
        tokenizer_fingerprint=tokenizer,
        blank_seed=8,
    )
    selected = (
        DynamicTargetSynthesizer(
            inspectors=default_inspector_registry(),
            adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        )
        .synthesize(
            request=request,
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )
        .adapter_payload
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
        ledger=ledger,
    )

    with pytest.raises(
        DynamicLfm2AdapterError,
        match="final evaluator mutated",
    ):
        system.run(
            parent_models={"parent": parent},
            target_request=request,
            evidence_fingerprint=evidence_fingerprint,
            destination=tmp_path / "must-not-publish",
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    assert ledger.decision(selected.signature).allowed
    assert not any(record.signature == selected.signature for record in ledger.records)
    assert not any(record.status == "accepted" for record in ledger.records)
    assert not (tmp_path / "must-not-publish").exists()


def test_lfm_catalog_does_not_accept_before_fresh_reload_passes(tmp_path) -> None:
    parent = _model(9, 6)
    tokenizer = canonical_sha256({"tokenizer": "local-lfm"})
    ledger = ExperimentLedger()
    target_adapter, stack_adapter, evidence_fingerprint, graph = _catalog(
        parent,
        tokenizer,
        ledger,
        fresh_reload_evaluator=lambda _model: FoldDecision.reject(
            "fresh standalone behavior failed"
        ),
    )
    request = DynamicTargetRequest(
        child_name="automatic-lfm-child",
        target_family="lfm2",
        target_modalities=("text",),
        exact_unique_parameters=graph.unique_parameter_count - 24,
        tokenizer_fingerprint=tokenizer,
        blank_seed=10,
    )
    selected = (
        DynamicTargetSynthesizer(
            inspectors=default_inspector_registry(),
            adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        )
        .synthesize(
            request=request,
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )
        .adapter_payload
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
        ledger=ledger,
    )

    with pytest.raises(
        DynamicFoldArtifactError,
        match="failed its behavioral gate",
    ):
        system.run(
            parent_models={"parent": parent},
            target_request=request,
            evidence_fingerprint=evidence_fingerprint,
            destination=tmp_path / "must-not-publish",
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    assert ledger.decision(selected.signature).allowed
    assert not any(record.signature == selected.signature for record in ledger.records)
    midpoint = len(ledger.records) // 2
    assert midpoint > 0
    assert all(record.status == "accepted" for record in ledger.records[:midpoint])
    assert all(
        record.status == "rejected"
        and record.reason_class == "fresh_reload_behavior_gate_reject"
        for record in ledger.records[midpoint:]
    )
    assert [record.signature for record in ledger.records[:midpoint]] == [
        record.signature for record in ledger.records[midpoint:]
    ]
    assert not ledger.decision(ledger.records[-1].signature).allowed
    assert not (tmp_path / "must-not-publish").exists()

    with pytest.raises(
        DynamicFoldCompilerError,
        match="previously closed exact attempt",
    ):
        system.run(
            parent_models={"parent": parent},
            target_request=request,
            evidence_fingerprint=evidence_fingerprint,
            destination=tmp_path / "retry-must-not-publish",
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )
    assert not (tmp_path / "retry-must-not-publish").exists()
