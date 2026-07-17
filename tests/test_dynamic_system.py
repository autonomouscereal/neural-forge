import json

import pytest

from neural_forge.dynamic_behavior_gate import (
    BehaviorGateBinding,
    expose_behavior_gate_binding,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_compiler import (
    DynamicFoldCompilerError,
    DynamicFoldSystemStack,
)
from neural_forge.dynamic_fold_executor import (
    DynamicFoldExecutionError,
    ExactCopyOperator,
    FoldDecision,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
    DynamicFoldingSystemError,
)
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


class _Accept:
    def evaluate(self, context):
        with torch.no_grad():
            context.execution_context.child_model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return FoldDecision.accept("whole-output test gate")


def _bound_test_gates(gate_id: str, baseline: str):
    final = _Accept()

    def fresh(model):
        with torch.no_grad():
            model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return True

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


class _TargetAdapter:
    adapter_id = "test.catalog.target"
    version = 1
    priority = 10
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
        child = type(parent)(parent.config)
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
            report={"catalog_materialized": True},
        )


class _StackAdapter:
    adapter_id = "test.catalog.stack"
    version = 1
    priority = 10
    target_families = ("llama",)
    target_modalities = ("text",)
    method_ids = ("copy_exact",)

    def __init__(self, saver, loader):
        self._saver = saver
        self._loader = loader

    def supports(self, target):
        return target.adapter_id == _TargetAdapter.adapter_id

    def build(self, _target):
        evidence = canonical_sha256({"cached_parent_evidence": "once"})
        final_gate, fresh_gate = _bound_test_gates(
            "test.catalog.whole_output",
            "e" * 64,
        )
        return DynamicFoldSystemStack(
            operators=FoldOperatorRegistry((ExactCopyOperator(),)),
            evidence_provider=StaticEvidenceProvider(evidence),
            evaluator=_Accept(),
            final_evaluator=final_gate,
            model_saver=self._saver,
            model_loader=self._loader,
            fresh_reload_evaluator=fresh_gate,
            saver_id="test-catalog-saver-v1",
            loader_id="test-catalog-loader-v1",
            behavior_gate_id="test.catalog.whole_output",
            behavior_gate_version=1,
            behavior_baseline_fingerprint="e" * 64,
        )


def _saver(model, destination):
    torch.save(model.state_dict(), destination / "model.pt")
    (destination / "config.json").write_text(
        json.dumps(model.config.to_dict()),
        encoding="utf-8",
    )


def _loader(source):
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


def test_persistent_catalog_runs_without_per_request_stack_dispatch(
    tmp_path,
) -> None:
    parent = _model(100)
    tokenizer = canonical_sha256({"tokenizer": "shared"})
    evidence = canonical_sha256({"cached_parent_evidence": "once"})
    parameter_count = sum(
        parameter.numel()
        for parameter in {id(value): value for value in parent.parameters()}.values()
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((_TargetAdapter(),)),
        stack_adapters=DynamicFoldStackAdapterRegistry(
            (_StackAdapter(_saver, _loader),)
        ),
        require_positive_compaction=False,
    )

    result = system.run(
        parent_models={"parent": parent},
        target_request=DynamicTargetRequest(
            child_name="catalog-child",
            target_family="auto",
            target_modalities=("text",),
            exact_unique_parameters=parameter_count,
            tokenizer_fingerprint=tokenizer,
            blank_seed=101,
        ),
        evidence_fingerprint=evidence,
        destination=tmp_path / "published",
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert result.complete
    assert result.target.adapter_id == _TargetAdapter.adapter_id
    assert result.published.artifact.fresh_reload_decision.accepted


def test_catalog_fails_closed_without_a_compatible_stack(tmp_path) -> None:
    parent = _model(200)
    tokenizer = canonical_sha256({"tokenizer": "shared"})
    parameter_count = sum(
        parameter.numel()
        for parameter in {id(value): value for value in parent.parameters()}.values()
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((_TargetAdapter(),)),
        stack_adapters=DynamicFoldStackAdapterRegistry(),
        require_positive_compaction=False,
    )

    with pytest.raises(DynamicFoldingSystemError, match="No registered"):
        system.run(
            parent_models={"parent": parent},
            target_request=DynamicTargetRequest(
                child_name="catalog-child",
                target_family="auto",
                target_modalities=("text",),
                exact_unique_parameters=parameter_count,
                tokenizer_fingerprint=tokenizer,
                blank_seed=201,
            ),
            evidence_fingerprint=canonical_sha256({"cached_parent_evidence": "once"}),
            destination=tmp_path / "unpublished",
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )


def test_production_catalog_rejects_noncompressive_target(tmp_path) -> None:
    parent = _model(300)
    tokenizer = canonical_sha256({"tokenizer": "shared"})
    parameter_count = sum(
        parameter.numel()
        for parameter in {id(value): value for value in parent.parameters()}.values()
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((_TargetAdapter(),)),
        stack_adapters=DynamicFoldStackAdapterRegistry(
            (_StackAdapter(_saver, _loader),)
        ),
    )

    with pytest.raises(
        DynamicFoldCompilerError,
        match="positive learned-parameter compaction",
    ):
        system.run(
            parent_models={"parent": parent},
            target_request=DynamicTargetRequest(
                child_name="same-size-control",
                target_family="llama",
                target_modalities=("text",),
                exact_unique_parameters=parameter_count,
                tokenizer_fingerprint=tokenizer,
                blank_seed=301,
            ),
            evidence_fingerprint=canonical_sha256({"cached_parent_evidence": "once"}),
            destination=tmp_path / "must-not-publish",
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    assert not (tmp_path / "must-not-publish").exists()


def test_automatic_system_rejects_aliased_parent_mapping(tmp_path) -> None:
    parent = _model(400)
    tokenizer = canonical_sha256({"tokenizer": "shared"})
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((_TargetAdapter(),)),
        stack_adapters=DynamicFoldStackAdapterRegistry(
            (_StackAdapter(_saver, _loader),)
        ),
    )

    with pytest.raises(
        DynamicFoldCompilerError,
        match="must not alias the same model object",
    ):
        system.run(
            parent_models={"alias_a": parent, "alias_b": parent},
            target_request=DynamicTargetRequest(
                child_name="aliased-control",
                target_family="llama",
                target_modalities=("text",),
                exact_unique_parameters=1,
                tokenizer_fingerprint=tokenizer,
                blank_seed=401,
            ),
            evidence_fingerprint=canonical_sha256({"cached_parent_evidence": "once"}),
            destination=tmp_path / "must-not-publish",
            parent_tokenizer_fingerprints={
                "alias_a": tokenizer,
                "alias_b": tokenizer,
            },
        )

    assert not (tmp_path / "must-not-publish").exists()


def test_target_materializer_cannot_mutate_parent_or_hide_training_attempt(
    tmp_path,
) -> None:
    class _MaliciousTarget(_TargetAdapter):
        adapter_id = "test.catalog.malicious-target"

        def materialize(
            self,
            request,
            parent_models,
            parent_graphs,
            inspectors,
        ):
            parent = next(iter(parent_models.values()))
            try:
                torch.optim.SGD(parent.parameters(), lr=0.1)
            except DynamicFoldExecutionError:
                pass
            with torch.no_grad():
                next(parent.parameters()).add_(1.0)
            return super().materialize(
                request,
                parent_models,
                parent_graphs,
                inspectors,
            )

    parent = _model(500)
    tokenizer = canonical_sha256({"tokenizer": "shared"})
    parameter_count = sum(
        parameter.numel()
        for parameter in {id(value): value for value in parent.parameters()}.values()
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((_MaliciousTarget(),)),
        stack_adapters=DynamicFoldStackAdapterRegistry(
            (_StackAdapter(_saver, _loader),)
        ),
        require_positive_compaction=False,
    )

    with pytest.raises(
        DynamicFoldExecutionError,
        match="mutated a protected model",
    ):
        system.run(
            parent_models={"parent": parent},
            target_request=DynamicTargetRequest(
                child_name="malicious-control",
                target_family="llama",
                target_modalities=("text",),
                exact_unique_parameters=parameter_count,
                tokenizer_fingerprint=tokenizer,
                blank_seed=501,
            ),
            evidence_fingerprint=canonical_sha256({"cached_parent_evidence": "once"}),
            destination=tmp_path / "must-not-publish",
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    assert not (tmp_path / "must-not-publish").exists()
