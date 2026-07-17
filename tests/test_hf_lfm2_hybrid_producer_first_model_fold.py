from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_compiler import DynamicFoldingCompiler
from neural_forge.dynamic_fold_executor import FoldDecision, fingerprint_model_state
from neural_forge.dynamic_lfm2_adapter import (
    HYBRID_MODEL_METHOD,
    DynamicLfm2AdapterError,
    Lfm2PreparedHybridProducerEvidence,
    Lfm2TargetSynthesisAdapter,
    build_lfm2_dynamic_compiler_bridge,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import default_inspector_registry
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_lfm2_canonical import inspect_lfm2_layer
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_closed_loop_fold import Lfm2ClosedLoopFoldConfig
from neural_forge.hf_lfm2_hybrid_producer_first_model_fold import (
    Lfm2HybridProducerFirstFoldError,
    install_lfm2_hybrid_producer_first_model_fold,
)
from neural_forge.hf_lfm2_model_fold import (
    Lfm2ModelFoldConfig,
    Lfm2ModelFoldEvidenceError,
    Lfm2ModelFoldSpan,
    fingerprint_lfm2_model_feature_state,
)
from neural_forge.hf_lfm2_producer_first_model_fold import (
    Lfm2ProducerFirstCapture,
    Lfm2ProducerFirstEvidence,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _model(
    seed: int,
    *,
    layer_types: tuple[str, ...],
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=intermediate_size,
        num_hidden_layers=len(layer_types),
        num_attention_heads=query_heads,
        num_key_value_heads=key_value_heads,
        head_dim=2,
        layer_types=list(layer_types),
        conv_L_cache=3,
        conv_bias=True,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    return transformers.Lfm2ForCausalLM(config).eval()


class _Rows:
    def __init__(
        self,
        *,
        layer_kinds: tuple[str, ...],
        layer_fields: dict[int, dict[str, torch.Tensor]],
        cache_key: str,
        trace_rows_sha256: str,
    ) -> None:
        self.paths = tuple(f"row-{index}" for index in range(12))
        self.path_splits = (
            *("selection" for _ in range(4)),
            *("fit" for _ in range(4)),
            *("validation" for _ in range(4)),
        )
        self.layer_kinds = layer_kinds
        self.cache_key = cache_key
        self.corpus_fingerprint = "b" * 64
        self.trace_rows_sha256 = trace_rows_sha256
        self.layer_fields = MappingProxyType(
            {
                index: MappingProxyType(dict(values))
                for index, values in layer_fields.items()
            }
        )

    def field(self, layer_index: int, name: str) -> torch.Tensor:
        return self.layer_fields[layer_index][name]


def _parent_rows(parent) -> _Rows:
    generator = torch.Generator().manual_seed(103)
    row_count = 12
    hidden = 8
    source_neurons = 6
    residual_pre = torch.zeros(row_count, hidden, dtype=torch.float64)
    fields: dict[int, dict[str, torch.Tensor]] = {}
    for layer_index, layer in enumerate(parent.model.layers):
        contract = inspect_lfm2_layer(layer)
        post_operator = residual_pre + torch.randn(
            row_count,
            hidden,
            generator=generator,
            dtype=torch.float64,
        )
        residual_post = post_operator + torch.randn(
            row_count,
            hidden,
            generator=generator,
            dtype=torch.float64,
        )
        gate = torch.randn(
            row_count,
            source_neurons,
            generator=generator,
            dtype=torch.float64,
        )
        up = torch.randn(
            row_count,
            source_neurons,
            generator=generator,
            dtype=torch.float64,
        )
        values = {
            "residual.pre": residual_pre,
            "residual.post_operator": post_operator,
            "residual.post_layer": residual_post,
            "mlp.gate": gate,
            "mlp.up": up,
            "mlp.post_swiglu": torch.nn.functional.silu(gate) * up,
        }
        if contract.kind == "full_attention":
            values.update(
                {
                    "attention.q": torch.randn(
                        row_count,
                        4,
                        2,
                        generator=generator,
                        dtype=torch.float64,
                    ),
                    "attention.k": torch.randn(
                        row_count,
                        2,
                        2,
                        generator=generator,
                        dtype=torch.float64,
                    ),
                    "attention.v": torch.randn(
                        row_count,
                        2,
                        2,
                        generator=generator,
                        dtype=torch.float64,
                    ),
                    "attention.context": torch.randn(
                        row_count,
                        4,
                        2,
                        generator=generator,
                        dtype=torch.float64,
                    ),
                }
            )
        fields[layer_index] = values
        residual_pre = residual_post
    return _Rows(
        layer_kinds=tuple(
            inspect_lfm2_layer(layer).kind for layer in parent.model.layers
        ),
        layer_fields=fields,
        cache_key="a" * 64,
        trace_rows_sha256="c" * 64,
    )


def _child_rows(child) -> _Rows:
    generator = torch.Generator().manual_seed(104)
    row_count = 12
    hidden = 8
    target_neurons = 4
    fields: dict[int, dict[str, torch.Tensor]] = {}
    for layer_index, layer in enumerate(child.model.layers):
        contract = inspect_lfm2_layer(layer)
        values = {
            "residual.pre": torch.randn(
                row_count,
                hidden,
                generator=generator,
                dtype=torch.float64,
            ),
            "residual.post_operator": torch.randn(
                row_count,
                hidden,
                generator=generator,
                dtype=torch.float64,
            ),
            "mlp.post_swiglu": torch.randn(
                row_count,
                target_neurons,
                generator=generator,
                dtype=torch.float64,
            ),
        }
        if contract.kind == "full_attention":
            values["attention.context"] = torch.randn(
                row_count,
                2,
                2,
                generator=generator,
                dtype=torch.float64,
            )
        fields[layer_index] = values
    return _Rows(
        layer_kinds=tuple(
            inspect_lfm2_layer(layer).kind for layer in child.model.layers
        ),
        layer_fields=fields,
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
    )


def _state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _assert_state_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _unique_parameters(model) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += int(parameter.numel())
    return total


def _fixture(
    *,
    failing_capture: bool = False,
    misaligned_capture: bool = False,
):
    parent = _model(
        100,
        layer_types=("conv", "full_attention", "conv"),
        intermediate_size=6,
        query_heads=4,
        key_value_heads=2,
    )
    child = _model(
        101,
        layer_types=("full_attention", "conv"),
        intermediate_size=4,
        query_heads=2,
        key_value_heads=1,
    )
    parent_rows = _parent_rows(parent)
    captured_rows = _child_rows(child)
    if misaligned_capture:
        captured_rows.corpus_fingerprint = "f" * 64
    blank_state = fingerprint_lfm2_model_feature_state(child)
    captured_states: list[str] = []

    def capture(model):
        state = fingerprint_lfm2_model_feature_state(model)
        captured_states.append(state)
        assert state != blank_state
        assert getattr(
            model,
            "_neural_forge_producer_plan_fingerprint",
            None,
        )
        if failing_capture:
            raise Lfm2HybridProducerFirstFoldError("synthetic capture failure")
        return Lfm2ProducerFirstCapture(
            rows=captured_rows,
            cache_key=captured_rows.cache_key,
            child_forward_count=1,
        )

    evidence = Lfm2ProducerFirstEvidence(
        parent_rows=parent_rows,
        selection_indices=(0, 1, 2, 3),
        fit_indices=(4, 5, 6, 7),
        validation_indices=(8, 9, 10, 11),
        capture_child=capture,
        parent_cache_fingerprint=parent_rows.cache_key,
        parent_identity_sha256=(
            fingerprint_lfm2_parent(parent).identity_sha256
        ),
        capture_builder_id="test.hybrid-producer-first.v1",
    )
    spans = (
        Lfm2ModelFoldSpan((0, 1), 0),
        Lfm2ModelFoldSpan((2,), 1),
    )
    config = Lfm2ModelFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=1e-8,
            max_endpoint_relative_rmse=1e9,
            min_endpoint_cosine_similarity=-1.0,
            max_endpoint_prior_delta_ratio=1e12,
        ),
        max_span_correction_anchors=4,
        max_span_endpoint_prior_delta_ratio=1e12,
    )
    return parent, child, spans, evidence, config, captured_states


def test_hybrid_depth_fold_captures_only_complete_producer_and_rolls_back_blank():
    parent, child, spans, evidence, config, captured_states = _fixture()
    parent_before = _state(parent)
    blank_before = _state(child)
    blank_fingerprint = fingerprint_lfm2_model_feature_state(child)

    transaction = install_lfm2_hybrid_producer_first_model_fold(
        parent,
        child,
        spans,
        evidence,
        config=config,
    )

    assert len(captured_states) == 1
    assert captured_states[0] == transaction.plan.producer_state_sha256
    assert captured_states[0] != blank_fingerprint
    assert transaction.plan.bindings[0].anchor_source_layer_index == 1
    assert transaction.plan.bindings[1].anchor_source_layer_index == 2
    assert transaction.plan.model_plan.parameters_removed > 0
    assert transaction.plan.child_forward_count == 1
    assert transaction.plan.as_dict()["blank_features_used"] is False
    _assert_state_equal(parent_before, _state(parent))

    report = transaction.rollback()
    assert report["rollback_exact"] is True
    assert report["restored_state_sha256"] == blank_fingerprint
    _assert_state_equal(blank_before, _state(child))
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


def test_hybrid_capture_failure_restores_compiler_zero_child_exactly():
    parent, child, spans, evidence, config, captured_states = _fixture(
        failing_capture=True
    )
    blank_before = _state(child)

    with pytest.raises(
        Lfm2HybridProducerFirstFoldError,
        match="synthetic capture failure",
    ):
        install_lfm2_hybrid_producer_first_model_fold(
            parent,
            child,
            spans,
            evidence,
            config=config,
        )

    assert len(captured_states) == 1
    _assert_state_equal(blank_before, _state(child))
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


def test_hybrid_late_evidence_failure_restores_compiler_zero_child_exactly():
    parent, child, spans, evidence, config, captured_states = _fixture(
        misaligned_capture=True
    )
    blank_before = _state(child)

    with pytest.raises(
        Lfm2ModelFoldEvidenceError,
        match="not exactly aligned",
    ):
        install_lfm2_hybrid_producer_first_model_fold(
            parent,
            child,
            spans,
            evidence,
            config=config,
        )

    assert len(captured_states) == 1
    _assert_state_equal(blank_before, _state(child))
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


class _TopologyFinalGate:
    def __init__(self, *, reject: bool) -> None:
        self.reject = reject
        self.calls = 0

    def evaluate(self, context):
        self.calls += 1
        assert len(context.execution_context.parent_graphs["parent"].layer_indices) == 3
        assert len(context.execution_context.child_graph.layer_indices) == 2
        if self.reject:
            return FoldDecision.reject("synthetic mapped trajectory rejection")
        return FoldDecision.accept("test final gate")


@pytest.mark.parametrize(
    "reject_final",
    (False, True),
    ids=("commit", "final-reject-exact-rollback"),
)
def test_dynamic_compiler_runs_hybrid_parent_only_provider_to_one_commit(
    reject_final,
):
    parent = _model(
        110,
        layer_types=("conv", "full_attention", "conv"),
        intermediate_size=6,
        query_heads=4,
        key_value_heads=2,
    )
    tokenizer_fingerprint = canonical_sha256(
        {"tokenizer": "hybrid-producer-first-integration"}
    )
    cache_fingerprint = "a" * 64
    exact_target = _unique_parameters(
        _model(
            111,
            layer_types=("full_attention", "conv"),
            intermediate_size=4,
            query_heads=2,
            key_value_heads=1,
        )
    )
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(2, 1, 4),
        evidence_fingerprint=cache_fingerprint,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((target_adapter,)),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="hybrid-child",
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=exact_target,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=target_adapter.adapter_id,
            fold_method_id=HYBRID_MODEL_METHOD,
            blank_seed=17,
            configuration=(
                ("source_parent", "parent"),
                ("target_depth", 2),
                ("target_layer_types", "full_attention,conv"),
                ("ordered_source_spans", "0,1;2"),
            ),
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer_fingerprint},
    )
    child = target.model
    blank = fingerprint_model_state(child)
    blank_state = _state(child)
    blank_parameter_ids = tuple(id(parameter) for parameter in child.parameters())
    parent_rows = _parent_rows(parent)
    captured_rows = _child_rows(child)
    captured_states: list[str] = []

    def capture(model):
        state = fingerprint_lfm2_model_feature_state(model)
        captured_states.append(state)
        assert fingerprint_model_state(model) != blank
        return Lfm2ProducerFirstCapture(
            rows=captured_rows,
            cache_key=captured_rows.cache_key,
            child_forward_count=1,
        )

    evidence = Lfm2ProducerFirstEvidence(
        parent_rows=parent_rows,
        selection_indices=(0, 1, 2, 3),
        fit_indices=(4, 5, 6, 7),
        validation_indices=(8, 9, 10, 11),
        capture_child=capture,
        parent_cache_fingerprint=cache_fingerprint,
        parent_identity_sha256=(
            fingerprint_lfm2_parent(parent).identity_sha256
        ),
        capture_builder_id="test.dynamic-hybrid-provider.v1",
    )
    builder_calls = 0

    def model_builder(request):
        nonlocal builder_calls
        builder_calls += 1
        assert request.child_state_fingerprint == blank
        assert fingerprint_model_state(request.child_model) == blank
        return Lfm2PreparedHybridProducerEvidence(evidence=evidence)

    model_config = Lfm2ModelFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=1e-8,
            max_endpoint_relative_rmse=1e9,
            min_endpoint_cosine_similarity=-1.0,
            max_endpoint_prior_delta_ratio=1e12,
        ),
        max_span_correction_anchors=4,
        max_span_endpoint_prior_delta_ratio=1e12,
    )
    bridge = build_lfm2_dynamic_compiler_bridge(
        parent_cache_fingerprint=cache_fingerprint,
        parent_cache_payload={"sealed": "parent-cache"},
        evidence_builder=lambda _request: None,
        model_evidence_builder=model_builder,
        model_fold_config=model_config,
        selected_target=target.adapter_payload,
        final_evaluator=(
            final_gate := _TopologyFinalGate(reject=reject_final)
        ),
        blank_state_fingerprint=blank,
        parent_cache_integrity_check=lambda _payload: cache_fingerprint,
    )
    compilation = DynamicFoldingCompiler(
        operators=bridge.operators,
    ).compile(
        parent_models={"parent": parent},
        child_model=child,
        child_name="hybrid-child",
        evidence_provider=bridge.evidence_provider,
        evaluator=bridge.evaluator,
        final_evaluator=bridge.final_evaluator,
        evidence_fingerprint=cache_fingerprint,
        parent_tokenizer_fingerprints={
            "parent": tokenizer_fingerprint,
        },
        child_tokenizer_fingerprint=tokenizer_fingerprint,
        exact_target_unique_parameters=exact_target,
        initializer=bridge.initializer,
    )

    assert builder_calls == 1
    assert len(captured_states) == 1
    assert bridge.evidence_provider.build_count == 1
    assert final_gate.calls == 1
    assert compilation.run.observed_parent_module_forward_calls == 0
    assert compilation.run.observed_parent_root_forward_calls == 0
    if reject_final:
        assert compilation.complete is False
        assert compilation.run.status == "final_gate_rejected"
        assert compilation.run.accepted_action_ids == ()
        assert compilation.run.completed_target_parameter_paths == ()
        assert fingerprint_model_state(child) == blank
        _assert_state_equal(blank_state, _state(child))
        assert tuple(id(parameter) for parameter in child.parameters()) == (
            blank_parameter_ids
        )
        assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")
        setattr(
            child,
            "_neural_forge_producer_plan_fingerprint",
            "synthetic-unauthenticated-marker",
        )
        with pytest.raises(
            DynamicLfm2AdapterError,
            match="not authenticated by its parent-only cache",
        ):
            bridge.evidence_provider.restore(
                SimpleNamespace(
                    child_model=child,
                    child_state_fingerprint=blank,
                ),
                blank,
                ("unknown-provisional-action",),
            )
        assert hasattr(child, "_neural_forge_producer_plan_fingerprint")
        delattr(child, "_neural_forge_producer_plan_fingerprint")
    else:
        assert compilation.complete is True
        assert fingerprint_model_state(child) != blank
