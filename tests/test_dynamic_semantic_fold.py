from __future__ import annotations

import json

import pytest

from neural_forge.dynamic_behavior_gate import (
    BehaviorGateBinding,
    expose_behavior_gate_binding,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import FoldDecision
from neural_forge.dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from neural_forge.dynamic_lfm2_semantic_system import (
    LFM2_BOUND_SEMANTIC_WHOLE_MODEL_STACK_ADAPTER_ID,
    Lfm2BoundSemanticWholeModelFoldStackAdapter,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import inspect_model_circuits
from neural_forge.dynamic_model_inspector import default_inspector_registry
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
    synthesize_dense_parameter_frontier,
)
from neural_forge.dynamic_semantic_fold import (
    DynamicSemanticFoldError,
    SemanticComponentFoldReceipt,
    SemanticWholeModelCompileResult,
    SemanticWholeModelFoldEvidence,
    SemanticWholeModelFoldResources,
    SemanticWholeModelTargetBinding,
    ResolvingSemanticWholeModelFoldStackAdapter,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)
from neural_forge.hf_lfm2_dense_budget_model_fold import (
    DENSE_BUDGET_MODEL_METHOD,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _parent(seed: int = 1):
    torch.manual_seed(seed)
    return transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=["conv", "full_attention"],
            conv_L_cache=3,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
    )


class _SealedCache:
    def __init__(self, cache_key: str) -> None:
        self.cache_key = cache_key
        self.verify_count = 0

    def verify(self):
        self.verify_count += 1
        return self


class _CompleteSemanticCompiler:
    compiler_id = "test.semantic.complete-lfm2"
    version = 1
    resource_fingerprint = canonical_sha256(
        {"test_semantic_compiler": "complete-lfm2-v1"}
    )

    def compile(self, request):
        child = request.context.child_model
        paths = tuple(
            name
            for name, _parameter in child.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        )
        with torch.no_grad():
            for index, (_name, parameter) in enumerate(
                child.named_parameters(recurse=True, remove_duplicate=False)
            ):
                parameter.fill_((index + 1) / 10_000.0)
        token_paths = tuple(
            path
            for path in paths
            if "embed_tokens" in path or "lm_head" in path
        )
        body_paths = tuple(path for path in paths if path not in token_paths)
        return SemanticWholeModelCompileResult(
            covered_target_parameter_paths=paths,
            component_receipts=(
                SemanticComponentFoldReceipt(
                    component_id="token-interface",
                    covered_parameter_paths=token_paths,
                    receipt={
                        "method": "cached-token-interface-fold",
                        "numeric_gate": False,
                    },
                ),
                SemanticComponentFoldReceipt(
                    component_id="residual-attention-body",
                    covered_parameter_paths=body_paths,
                    receipt={
                        "method": "cached-residual-attention-fold",
                        "numeric_gate": False,
                    },
                ),
            ),
            whole_model_receipt={
                "status": "one-complete-child-installed",
                "component_count": 2,
            },
            operation_audit={
                "training_steps": 0,
                "optimizer_constructions": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "autograd_grad_calls": 0,
                "pruning_operations": 0,
                "quantization_operations": 0,
                "parent_forward_calls": 0,
                "parent_trace_recaptures": 0,
                "cached_parent_evidence_only": True,
                "complete_model_compaction": True,
                "component_diagnostics_have_acceptance_authority": False,
                "numeric_diagnostics_have_acceptance_authority": False,
            },
        )


class _StaticSemanticResourceResolver:
    resolver_id = "test.semantic.configure-once-resolver"
    version = 1
    priority = 500
    target_families = ("lfm2",)
    target_modalities = ("text",)
    method_ids = (DENSE_BUDGET_MODEL_METHOD,)

    def __init__(self, resources):
        self._resources = resources
        self.resolve_count = 0
        self.resource_fingerprint = canonical_sha256(
            {
                "resolver": self.resolver_id,
                "resources": resources.resource_fingerprint,
            }
        )

    def supports(self, target):
        return self._resources.binding.matches(target)

    def resolve(self, target):
        if not self.supports(target):
            raise DynamicSemanticFoldError("Resolver target is unsupported.")
        self.resolve_count += 1
        return self._resources


class _AcceptWholeOutput:
    def evaluate(self, context):
        with torch.no_grad():
            context.execution_context.child_model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return FoldDecision.accept("decoded whole-output manual-review test gate")


def _behavior_gates(baseline_fingerprint: str):
    final = _AcceptWholeOutput()

    def fresh(model):
        with torch.no_grad():
            model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return FoldDecision.accept("fresh decoded whole-output review accepted")

    body = {
        "schema": "test-semantic-behavior-gate-v1",
        "gate_id": "test.semantic.whole-output",
        "gate_version": 1,
        "baseline_fingerprint": baseline_fingerprint,
    }
    manifest_fingerprint = canonical_sha256(body)
    binding = BehaviorGateBinding(
        gate_id=body["gate_id"],
        gate_version=body["gate_version"],
        baseline_fingerprint=baseline_fingerprint,
        manifest_fingerprint=manifest_fingerprint,
        manifest={**body, "manifest_fingerprint": manifest_fingerprint},
    )
    expose_behavior_gate_binding(final, binding)
    expose_behavior_gate_binding(fresh, binding)
    return final, fresh


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
        torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _semantic_catalog(parent):
    tokenizer = canonical_sha256({"tokenizer": "semantic-lfm2-test"})
    evidence_root = canonical_sha256({"cache": "sealed-parent-evidence"})
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    catalogs = DenseTargetTopologyCatalogRegistry(
        (Lfm2DenseTargetTopologyCatalog(),)
    )
    frontier = synthesize_dense_parameter_frontier(
        parent_graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            target_vocabulary_size=128,
            maximum_candidates=8,
        ),
        catalogs=catalogs,
    )
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 1),
        evidence_fingerprint=evidence_root,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    target_plan = plan_budgeted_dynamic_target(
        parent_graph,
        frontier.request,
        child_name="semantic-four-x-child",
        selected_candidate_rank=1,
        tokenizer_fingerprint=tokenizer,
        adapter_id=target_adapter.adapter_id,
        fold_method_id=DENSE_BUDGET_MODEL_METHOD,
        blank_seed=2,
        catalogs=catalogs,
    )
    binding = SemanticWholeModelTargetBinding(
        binding_id="test-lfm2-semantic-four-x",
        target_family="lfm2",
        target_modalities=("text",),
        target_adapter_id=target_adapter.adapter_id,
        target_adapter_version=target_adapter.version,
        method_id=DENSE_BUDGET_MODEL_METHOD,
        exact_target_unique_parameters=(
            target_plan.target_request.exact_unique_parameters
        ),
        parent_graph_fingerprints=(("parent", parent_graph.fingerprint),),
        child_tokenizer_fingerprint=tokenizer,
        target_configuration_sha256=canonical_sha256(
            dict(target_plan.target_request.configuration)
        ),
    )
    evidence = SemanticWholeModelFoldEvidence(
        root_fingerprint=evidence_root,
        source=_SealedCache(evidence_root),
    )
    final, fresh = _behavior_gates(evidence_root)
    resources = SemanticWholeModelFoldResources(
        binding=binding,
        evidence=evidence,
        compiler=_CompleteSemanticCompiler(),
        final_evaluator=final,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=fresh,
        saver_id="test-semantic-lfm2-saver-v1",
        loader_id="test-semantic-lfm2-loader-v1",
        stack_adapter_id=LFM2_BOUND_SEMANTIC_WHOLE_MODEL_STACK_ADAPTER_ID,
    )
    stack_adapter = Lfm2BoundSemanticWholeModelFoldStackAdapter(
        resources,
        require_tokenizer_bridge=False,
    )
    return (
        tokenizer,
        evidence_root,
        target_adapter,
        target_plan,
        stack_adapter,
        parent_graph,
    )


def test_automatic_system_executes_one_complete_semantic_four_x_action(
    tmp_path,
) -> None:
    parent = _parent()
    (
        tokenizer,
        evidence_root,
        target_adapter,
        target_plan,
        stack_adapter,
        parent_graph,
    ) = _semantic_catalog(parent)
    resolver = _StaticSemanticResourceResolver(stack_adapter.resources)
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry(
            (
                ResolvingSemanticWholeModelFoldStackAdapter(
                    resolver
                ),
            )
        ),
        topology_catalogs=DenseTargetTopologyCatalogRegistry(
            (Lfm2DenseTargetTopologyCatalog(),)
        ),
    )

    result = system.run(
        parent_models={"parent": parent},
        target_request=target_plan.target_request,
        evidence_fingerprint=evidence_root,
        destination=tmp_path / "published-semantic-child",
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    compilation = result.published.compilation
    assert result.complete
    assert len(compilation.plan.actions) == 1
    assert compilation.plan.actions[0].method_id == DENSE_BUDGET_MODEL_METHOD
    assert len(compilation.run.attempts) == 1
    attempt = compilation.run.attempts[0]
    assert attempt.accepted
    transaction = attempt.transaction_report
    semantic = transaction["semantic_compile_result"]
    assert len(semantic["component_receipts"]) == 2
    assert set(semantic["covered_target_parameter_paths"]) == set(
        compilation.child_graph.parameter_paths
    )
    assert transaction["complete_model_transaction"] is True
    assert (
        transaction["component_receipts_are_diagnostics_inside_transaction"]
        is True
    )
    assert compilation.run.observed_parent_module_forward_calls == 0
    assert compilation.run.optimizer_steps == 0
    assert compilation.run.backward_calls == 0
    assert compilation.run.pruning_operations == 0
    assert compilation.run.quantization_operations == 0
    assert parent_graph.unique_parameter_count / (
        result.target.resource_profile.unique_parameters
    ) >= 4.0
    assert result.published.artifact.fresh_reload_decision.accepted
    assert resolver.resolve_count == 1


def test_lfm2_current_semantic_registration_rejects_unbridged_target() -> None:
    parent = _parent(seed=3)
    (
        tokenizer,
        _evidence_root,
        target_adapter,
        target_plan,
        stack_adapter,
        _parent_graph,
    ) = _semantic_catalog(parent)
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((target_adapter,))
    ).synthesize(
        request=target_plan.target_request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    with pytest.raises(
        DynamicSemanticFoldError,
        match=(
            "different family, target adapter, method, or stack identity|"
            "do(?:es)? not support"
        ),
    ):
        # Re-create the same resource registration with the current exact-4x
        # backend's bridge requirement rather than the explicit test override.
        current = Lfm2BoundSemanticWholeModelFoldStackAdapter(
            stack_adapter.resources,
            require_tokenizer_bridge=True,
        )
        current.build(target)


def test_component_receipts_cannot_claim_an_incomplete_whole_model() -> None:
    with pytest.raises(
        DynamicSemanticFoldError,
        match="do not compose",
    ):
        SemanticWholeModelCompileResult(
            covered_target_parameter_paths=("a", "b"),
            component_receipts=(
                SemanticComponentFoldReceipt(
                    component_id="attention-only-diagnostic",
                    covered_parameter_paths=("a",),
                    receipt={"diagnostic": True},
                ),
            ),
            whole_model_receipt={"status": "incomplete"},
            operation_audit={
                **{name: 0 for name in (
                    "training_steps",
                    "optimizer_constructions",
                    "optimizer_steps",
                    "backward_calls",
                    "autograd_grad_calls",
                    "pruning_operations",
                    "quantization_operations",
                    "parent_forward_calls",
                    "parent_trace_recaptures",
                )},
                "cached_parent_evidence_only": True,
                "complete_model_compaction": True,
                "component_diagnostics_have_acceptance_authority": False,
                "numeric_diagnostics_have_acceptance_authority": False,
            },
        )
