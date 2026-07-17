from __future__ import annotations

import json

import pytest

from neural_forge.dynamic_behavior_gate import (
    BehaviorGateBinding,
    expose_behavior_gate_binding,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_artifact import (
    DynamicFoldArtifactError,
    load_dynamic_fold_artifact_certificate,
)
from neural_forge.dynamic_fold_compiler import (
    DynamicFoldCompilerError,
    DynamicFoldSystemStack,
    compile_and_publish_loaded_models,
    compile_loaded_models,
    run_dynamic_folding_system,
    synthesize_compile_and_publish_models,
)
from neural_forge.dynamic_fold_executor import (
    ExactCopyOperator,
    FoldDecision,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_fold_registry import ExperimentLedger
from neural_forge.dynamic_model_inspector import inspect_model_circuits
from neural_forge.dynamic_target_synthesis import (
    AdapterMaterialization,
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _model(seed: int):
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=True,
        )
    )


def _zero_registered_buffers(model):
    with torch.no_grad():
        for buffer in model.buffers():
            buffer.zero_()
    return model


class _Accept:
    def evaluate(self, context):
        with torch.no_grad():
            context.execution_context.child_model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return FoldDecision.accept("test gate")


class _LeafOnlyAccept:
    def evaluate(self, context):
        with torch.no_grad():
            context.execution_context.child_model.model.embed_tokens(
                torch.tensor([[1, 2]], dtype=torch.long)
            )
        return FoldDecision.accept("leaf-only cached gate")


class _ParentAndChildForwardAccept:
    def evaluate(self, context):
        with torch.no_grad():
            next(iter(context.execution_context.parent_models.values()))(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
            context.execution_context.child_model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return FoldDecision.accept("malicious parent-backed gate")


def _fresh_accept(model):
    with torch.no_grad():
        model(
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            use_cache=False,
        )
    return FoldDecision.accept("fresh whole-output gate accepted")


def _bound_test_gates(gate_id: str, baseline: str):
    final = _Accept()

    def fresh(model):
        return _fresh_accept(model)

    body = {
        "schema": "test-behavior-gate-manifest-v1",
        "gate_id": gate_id,
        "gate_version": 1,
        "baseline_fingerprint": baseline,
    }
    fingerprint = canonical_sha256(body)
    binding = BehaviorGateBinding(
        gate_id=gate_id,
        gate_version=1,
        baseline_fingerprint=baseline,
        manifest_fingerprint=fingerprint,
        manifest={**body, "manifest_fingerprint": fingerprint},
    )
    expose_behavior_gate_binding(final, binding)
    expose_behavior_gate_binding(fresh, binding)
    return final, fresh


class _AutomaticLlamaTarget:
    adapter_id = "test.automatic_llama"
    version = 1
    priority = 1
    target_families = ("llama",)
    target_modalities = ("text",)

    def supports(self, request, parent_graphs):
        return request.target_family in {"auto", "llama"} and any(
            graph.family == "llama" for graph in parent_graphs.values()
        )

    def materialize(
        self,
        request,
        parent_models,
        _parent_graphs,
        inspectors,
    ):
        parent = next(iter(parent_models.values()))
        torch.manual_seed(request.blank_seed)
        child = _zero_registered_buffers(type(parent)(parent.config))
        child.eval()
        graph = inspectors.inspect(
            child,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        return AdapterMaterialization(
            model=child,
            graph=graph,
            topology_fingerprint=canonical_sha256({"config": graph.config_fingerprint}),
            report={"automatic_blank": True},
        )


class _ParentForwardingLlamaTarget(_AutomaticLlamaTarget):
    adapter_id = "test.parent_forwarding_llama"

    def materialize(self, request, parent_models, parent_graphs, inspectors):
        parent = next(iter(parent_models.values()))
        with torch.no_grad():
            parent(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return super().materialize(
            request,
            parent_models,
            parent_graphs,
            inspectors,
        )


def test_public_compiler_inspects_plans_executes_and_audits() -> None:
    parent = _model(10)
    child = _model(20)
    evidence = canonical_sha256({"parent_cache": "once"})
    tokenizer = canonical_sha256({"tokenizer": "same"})

    compilation = compile_loaded_models(
        parent_models={"parent": parent},
        child_model=child,
        child_name="child",
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
        final_evaluator=_Accept(),
        evidence_fingerprint=evidence,
        parent_tokenizer_fingerprints={"parent": tokenizer},
        child_tokenizer_fingerprint=tokenizer,
    )

    assert compilation.complete
    assert compilation.plan.ready
    assert fingerprint_model_state(parent) == fingerprint_model_state(child)
    assert compilation.as_dict()["run"]["optimizer_steps"] == 0


def test_public_compiler_requires_a_final_whole_output_gate() -> None:
    parent = _model(30)
    child = _model(40)
    evidence = canonical_sha256({"parent_cache": "once"})
    tokenizer = canonical_sha256({"tokenizer": "same"})

    with pytest.raises(Exception, match="whole-output"):
        compile_loaded_models(
            parent_models={"parent": parent},
            child_model=child,
            child_name="child",
            operators=FoldOperatorRegistry((ExactCopyOperator(),)),
            evidence_provider=StaticEvidenceProvider(evidence),
            evaluator=_Accept(),
            evidence_fingerprint=evidence,
            parent_tokenizer_fingerprints={"parent": tokenizer},
            child_tokenizer_fingerprint=tokenizer,
        )


def test_end_to_end_api_compiles_publishes_reloads_and_certifies(tmp_path) -> None:
    parent = _model(50)
    child = _model(60)
    evidence = canonical_sha256({"parent_cache": "once"})
    tokenizer = canonical_sha256({"tokenizer": "same"})

    def saver(model, destination):
        torch.save(model.state_dict(), destination / "model.pt")
        (destination / "config.json").write_text(
            json.dumps(model.config.to_dict()),
            encoding="utf-8",
        )

    def loader(source):
        config = transformers.LlamaConfig(
            **json.loads((source / "config.json").read_text(encoding="utf-8"))
        )
        model = transformers.LlamaForCausalLM(config)
        model.load_state_dict(
            torch.load(
                source / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        return model

    result = compile_and_publish_loaded_models(
        parent_models={"parent": parent},
        child_model=child,
        child_name="child",
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
        final_evaluator=_Accept(),
        evidence_fingerprint=evidence,
        destination=tmp_path / "published",
        model_saver=saver,
        model_loader=loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-saver-v1",
        loader_id="test-loader-v1",
        parent_tokenizer_fingerprints={"parent": tokenizer},
        child_tokenizer_fingerprint=tokenizer,
    )

    assert result.complete
    assert result.artifact.fresh_reload_decision.accepted
    assert (tmp_path / "published" / "certificate.json").is_file()


@pytest.mark.parametrize(
    ("final_evaluator", "message"),
    (
        (
            _LeafOnlyAccept(),
            "final behavior gate accepted without executing the child model root",
        ),
        (
            _ParentAndChildForwardAccept(),
            "fold execution called a live parent model",
        ),
    ),
)
def test_publication_rejects_rootless_or_parent_backed_final_gate(
    tmp_path,
    final_evaluator,
    message,
) -> None:
    parent = _model(85)
    child = _model(86)
    evidence = canonical_sha256({"parent_cache": "once"})
    tokenizer = canonical_sha256({"tokenizer": "same"})
    destination = tmp_path / "must-not-publish"

    def saver(model, output):
        torch.save(model.state_dict(), output / "model.pt")
        (output / "config.json").write_text(
            json.dumps(model.config.to_dict()),
            encoding="utf-8",
        )

    def loader(source):
        config = transformers.LlamaConfig(
            **json.loads((source / "config.json").read_text(encoding="utf-8"))
        )
        model = transformers.LlamaForCausalLM(config)
        model.load_state_dict(
            torch.load(source / "model.pt", map_location="cpu", weights_only=True)
        )
        return model.eval()

    with pytest.raises(DynamicFoldCompilerError, match=message):
        compile_and_publish_loaded_models(
            parent_models={"parent": parent},
            child_model=child,
            child_name="child",
            operators=FoldOperatorRegistry((ExactCopyOperator(),)),
            evidence_provider=StaticEvidenceProvider(evidence),
            evaluator=_Accept(),
            final_evaluator=final_evaluator,
            evidence_fingerprint=evidence,
            destination=destination,
            model_saver=saver,
            model_loader=loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="unused-saver",
            loader_id="unused-loader",
            parent_tokenizer_fingerprints={"parent": tokenizer},
            child_tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_publication_is_removed_when_atomic_ledger_commit_fails(tmp_path) -> None:
    class CommitFailLedger(ExperimentLedger):
        def append_batch(self, records):
            if tuple(records):
                raise OSError("simulated ledger commit failure")

    parent = _model(51)
    child = _model(61)
    evidence = canonical_sha256({"parent_cache": "once"})
    tokenizer = canonical_sha256({"tokenizer": "same"})
    destination = tmp_path / "must-rollback"

    def saver(model, output):
        torch.save(model.state_dict(), output / "model.pt")
        (output / "config.json").write_text(
            json.dumps(model.config.to_dict()),
            encoding="utf-8",
        )

    def loader(source):
        config = transformers.LlamaConfig(
            **json.loads((source / "config.json").read_text(encoding="utf-8"))
        )
        model = transformers.LlamaForCausalLM(config)
        model.load_state_dict(
            torch.load(source / "model.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        return model

    ledger = CommitFailLedger()

    def fresh_reload_gate(model):
        model(input_ids=torch.tensor([[1, 2]], dtype=torch.long))
        return True

    with pytest.raises(
        DynamicFoldCompilerError,
        match="Publication ledger commit failed",
    ):
        compile_and_publish_loaded_models(
            parent_models={"parent": parent},
            child_model=child,
            child_name="child",
            operators=FoldOperatorRegistry((ExactCopyOperator(),)),
            evidence_provider=StaticEvidenceProvider(evidence),
            evaluator=_Accept(),
            final_evaluator=_Accept(),
            evidence_fingerprint=evidence,
            destination=destination,
            model_saver=saver,
            model_loader=loader,
            fresh_reload_evaluator=fresh_reload_gate,
            saver_id="test-saver-v1",
            loader_id="test-loader-v1",
            parent_tokenizer_fingerprints={"parent": tokenizer},
            child_tokenizer_fingerprint=tokenizer,
            ledger=ledger,
        )

    assert ledger.records == ()
    assert not destination.exists()


def test_system_api_synthesizes_child_without_caller_architecture_surgery(
    tmp_path,
) -> None:
    parent = _model(70)
    evidence = canonical_sha256({"parent_cache": "once"})
    tokenizer = canonical_sha256({"tokenizer": "same"})
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )

    def saver(model, destination):
        torch.save(model.state_dict(), destination / "model.pt")
        (destination / "config.json").write_text(
            json.dumps(model.config.to_dict()),
            encoding="utf-8",
        )

    def loader(source):
        config = transformers.LlamaConfig(
            **json.loads((source / "config.json").read_text(encoding="utf-8"))
        )
        model = transformers.LlamaForCausalLM(config)
        model.load_state_dict(
            torch.load(
                source / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        return _zero_registered_buffers(model)

    result = synthesize_compile_and_publish_models(
        parent_models={"parent": parent},
        target_request=DynamicTargetRequest(
            child_name="child",
            target_family="auto",
            target_modalities=("text",),
            exact_unique_parameters=graph.unique_parameter_count,
            maximum_parameter_bytes=graph.unique_parameter_bytes,
            tokenizer_fingerprint=tokenizer,
            blank_seed=71,
        ),
        target_adapters=DynamicTargetAdapterRegistry((_AutomaticLlamaTarget(),)),
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
        final_evaluator=_Accept(),
        evidence_fingerprint=evidence,
        destination=tmp_path / "automatic-published",
        model_saver=saver,
        model_loader=loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-saver-v1",
        loader_id="test-loader-v1",
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert result.complete
    assert result.target.adapter_id == "test.automatic_llama"
    assert result.target.resource_profile.unique_parameters == (
        graph.unique_parameter_count
    )
    assert result.published.artifact.fresh_reload_decision.accepted
    assert result.published.artifact.dynamic_target_provenance_present is True
    assert (
        result.published.artifact.provenance_target_fingerprint
        == result.target.fingerprint
    )
    target_document = json.loads(
        (
            tmp_path
            / "automatic-published"
            / "provenance"
            / "target.json"
        ).read_text(encoding="ascii")
    )
    assert target_document == result.target.as_dict()
    assert (
        load_dynamic_fold_artifact_certificate(tmp_path / "automatic-published")
        == result.published.artifact
    )

    hidden_state_destination = tmp_path / "hidden-state-must-not-publish"

    def hidden_state_loader(source):
        model = loader(source)
        model.hidden_inference_tensor = torch.ones(1)
        return model

    with pytest.raises(
        DynamicFoldArtifactError,
        match="complete inference-state reauthentication",
    ):
        synthesize_compile_and_publish_models(
            parent_models={"parent": parent},
            target_request=DynamicTargetRequest(
                child_name="hidden-state-child",
                target_family="auto",
                target_modalities=("text",),
                exact_unique_parameters=graph.unique_parameter_count,
                maximum_parameter_bytes=graph.unique_parameter_bytes,
                tokenizer_fingerprint=tokenizer,
                blank_seed=72,
            ),
            target_adapters=DynamicTargetAdapterRegistry(
                (_AutomaticLlamaTarget(),)
            ),
            operators=FoldOperatorRegistry((ExactCopyOperator(),)),
            evidence_provider=StaticEvidenceProvider(evidence),
            evaluator=_Accept(),
            final_evaluator=_Accept(),
            evidence_fingerprint=evidence,
            destination=hidden_state_destination,
            model_saver=saver,
            model_loader=hidden_state_loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver-v1",
            loader_id="hidden-state-loader-v1",
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )
    assert not hidden_state_destination.exists()


def test_top_level_system_selects_fold_stack_after_target_synthesis(
    tmp_path,
) -> None:
    parent = _model(80)
    evidence = canonical_sha256({"parent_cache": "once"})
    tokenizer = canonical_sha256({"tokenizer": "same"})
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    def saver(model, destination):
        torch.save(model.state_dict(), destination / "model.pt")
        (destination / "config.json").write_text(
            json.dumps(model.config.to_dict()),
            encoding="utf-8",
        )

    def loader(source):
        config = transformers.LlamaConfig(
            **json.loads((source / "config.json").read_text(encoding="utf-8"))
        )
        model = transformers.LlamaForCausalLM(config)
        model.load_state_dict(
            torch.load(
                source / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        return _zero_registered_buffers(model)

    def stack_factory(target):
        assert target.adapter_id == "test.automatic_llama"
        final_gate, fresh_gate = _bound_test_gates(
            "test.whole_output",
            "e" * 64,
        )
        return DynamicFoldSystemStack(
            operators=FoldOperatorRegistry((ExactCopyOperator(),)),
            evidence_provider=StaticEvidenceProvider(evidence),
            evaluator=_Accept(),
            final_evaluator=final_gate,
            model_saver=saver,
            model_loader=loader,
            fresh_reload_evaluator=fresh_gate,
            saver_id="test-saver-v1",
            loader_id="test-loader-v1",
            behavior_gate_id="test.whole_output",
            behavior_gate_version=1,
            behavior_baseline_fingerprint="e" * 64,
            snapshot_config=DynamicFoldSnapshotConfig.disk(snapshot_root),
        )

    result = run_dynamic_folding_system(
        parent_models={"parent": parent},
        target_request=DynamicTargetRequest(
            child_name="child",
            target_family="auto",
            target_modalities=("text",),
            exact_unique_parameters=graph.unique_parameter_count,
            tokenizer_fingerprint=tokenizer,
            blank_seed=81,
        ),
        target_adapters=DynamicTargetAdapterRegistry((_AutomaticLlamaTarget(),)),
        stack_factory=stack_factory,
        evidence_fingerprint=evidence,
        destination=tmp_path / "system-published",
        parent_tokenizer_fingerprints={"parent": tokenizer},
        require_positive_compaction=False,
    )

    assert result.complete
    assert result.published.artifact.fresh_reload_decision.accepted
    assert result.published.compilation.run.whole_model_snapshot_backend == "disk"
    assert list(snapshot_root.iterdir()) == []

    def parent_forwarding_stack_factory(target):
        with torch.no_grad():
            parent(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return stack_factory(target)

    with pytest.raises(
        DynamicFoldCompilerError,
        match="fold-stack construction executed a live parent model",
    ):
        run_dynamic_folding_system(
            parent_models={"parent": parent},
            target_request=DynamicTargetRequest(
                child_name="child-stack-forward",
                target_family="auto",
                target_modalities=("text",),
                exact_unique_parameters=graph.unique_parameter_count,
                tokenizer_fingerprint=tokenizer,
                blank_seed=82,
            ),
            target_adapters=DynamicTargetAdapterRegistry(
                (_AutomaticLlamaTarget(),)
            ),
            stack_factory=parent_forwarding_stack_factory,
            evidence_fingerprint=evidence,
            destination=tmp_path / "must-not-publish-stack-forward",
            parent_tokenizer_fingerprints={"parent": tokenizer},
            require_positive_compaction=False,
        )
    assert not (tmp_path / "must-not-publish-stack-forward").exists()


def test_target_synthesis_rejects_live_parent_forward(tmp_path) -> None:
    parent = _model(83)
    evidence = canonical_sha256({"parent_cache": "once"})
    tokenizer = canonical_sha256({"tokenizer": "same"})
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    adapter = _ParentForwardingLlamaTarget()
    destination = tmp_path / "must-not-publish-target-forward"

    with pytest.raises(
        DynamicFoldCompilerError,
        match="target synthesis executed a live parent model",
    ):
        synthesize_compile_and_publish_models(
            parent_models={"parent": parent},
            target_request=DynamicTargetRequest(
                child_name="child",
                target_family="auto",
                target_modalities=("text",),
                exact_unique_parameters=graph.unique_parameter_count,
                tokenizer_fingerprint=tokenizer,
                adapter_id=adapter.adapter_id,
                blank_seed=84,
            ),
            target_adapters=DynamicTargetAdapterRegistry((adapter,)),
            operators=FoldOperatorRegistry((ExactCopyOperator(),)),
            evidence_provider=StaticEvidenceProvider(evidence),
            evaluator=_Accept(),
            final_evaluator=_Accept(),
            evidence_fingerprint=evidence,
            destination=destination,
            model_saver=lambda _model, _path: None,
            model_loader=lambda _path: None,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="unused-saver",
            loader_id="unused-loader",
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    assert not destination.exists()
