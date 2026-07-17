from __future__ import annotations

import copy
from types import MappingProxyType, SimpleNamespace

import pytest

from experiments.run_lfm2_dynamic_catalog import (
    CachedProductionEvidenceBuilder,
    ProductionParentCachePayload,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_fold_planner import plan_dynamic_fold
from neural_forge.dynamic_fold_registry import default_method_registry
from neural_forge.dynamic_lfm2_adapter import (
    PRODUCER_FIRST_MODEL_METHOD,
    Lfm2ModelEvidenceBuildRequest,
    Lfm2PreparedHybridProducerEvidence,
    Lfm2ProducerFirstEvidenceBuildRequest,
    Lfm2SelectedTarget,
    Lfm2TargetSynthesisAdapter,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    DynamicModelInspectionError,
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCaptureConfig,
    fingerprint_lfm2_parent,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tokenizer_fingerprint() -> str:
    return canonical_sha256({"tokenizer": "producer-first-catalog-test"})


def _model(seed: int, *, intermediate_size: int):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=intermediate_size,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv", "full_attention", "conv"],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    return transformers.Lfm2ForCausalLM(config).eval()


def _unique_parameters(model) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += int(parameter.numel())
    return total


def _synthesize(parent):
    tokenizer = _tokenizer_fingerprint()
    exact_count = _unique_parameters(_model(99, intermediate_size=5))
    evidence = canonical_sha256({"cache": "producer-first"})
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(4, 2, 5),
        evidence_fingerprint=evidence,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="producer-first-child",
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=exact_count,
            tokenizer_fingerprint=tokenizer,
            adapter_id=adapter.adapter_id,
            fold_method_id=PRODUCER_FIRST_MODEL_METHOD,
            blank_seed=7,
            configuration=(("source_parent", "parent"),),
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    return target


def test_full_depth_marker_round_trips_and_plans_one_model_action() -> None:
    parent = _model(1, intermediate_size=6)
    target = _synthesize(parent)
    selected = target.adapter_payload
    assert isinstance(selected, Lfm2SelectedTarget)
    metadata = dict(target.graph.metadata)
    assert metadata["fold_method_id"] == PRODUCER_FIRST_MODEL_METHOD
    assert "ordered_source_spans" not in metadata
    assert target.resource_profile.layer_count == 3

    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    plan = plan_dynamic_fold(
        (parent_graph,),
        target.graph,
        child_state_fingerprint=target.blank_state_fingerprint,
        evidence_fingerprint=selected.signature.evidence_fingerprint,
        parent_state_fingerprints={"parent": fingerprint_model_state(parent)},
        exact_target_unique_parameters=target.resource_profile.unique_parameters,
    )
    assert plan.ready
    assert len(plan.actions) == 1
    assert plan.actions[0].method_id == PRODUCER_FIRST_MODEL_METHOD
    assert set(plan.actions[0].target_node_ids) == {
        node.node_id for node in target.graph.nodes
    }
    assert (
        default_method_registry()
        .get(PRODUCER_FIRST_MODEL_METHOD)
        .standalone
        is True
    )

    reloaded_config = transformers.Lfm2Config.from_dict(
        target.model.config.to_dict()
    )
    reloaded = transformers.Lfm2ForCausalLM(reloaded_config).eval()
    reloaded_graph = inspect_model_circuits(
        reloaded,
        model_name="producer-first-reloaded",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    assert (
        dict(reloaded_graph.metadata)["fold_method_id"]
        == PRODUCER_FIRST_MODEL_METHOD
    )


def test_full_depth_marker_rejects_depth_span_tampering() -> None:
    child = _model(2, intermediate_size=5)
    child.config._neural_forge_fold_method_id = PRODUCER_FIRST_MODEL_METHOD
    child.config._neural_forge_ordered_source_spans = "0;1;2"
    child.config._neural_forge_target_layer_types = (
        "conv,full_attention,conv"
    )

    with pytest.raises(
        DynamicModelInspectionError,
        match="must not include depth spans",
    ):
        inspect_model_circuits(
            child,
            model_name="tampered-producer-first",
            tokenizer_fingerprint=_tokenizer_fingerprint(),
        )


def test_full_depth_config_builder_does_not_mutate_parent() -> None:
    parent = _model(3, intermediate_size=6)
    before = copy.deepcopy(parent.config.to_dict())
    _synthesize(parent)
    assert parent.config.to_dict() == before


def test_catalog_evidence_uses_verified_cache_parent_identity(tmp_path) -> None:
    parent = _model(4, intermediate_size=6)
    child = _model(5, intermediate_size=5)
    parent_identity = fingerprint_lfm2_parent(parent).identity_sha256

    class Cache:
        cache_key = "a" * 64
        corpus_fingerprint = "b" * 64
        parent_fingerprint = parent_identity

        def __init__(self):
            self.verified = False

        def __len__(self):
            return 0

        def verify(self):
            self.verified = True
            return self

    class Rows:
        paths = ("selection", "fit", "validation")
        path_splits = ("selection", "fit", "validation")
        layer_kinds = ("short_conv", "full_attention", "short_conv")
        cache_key = Cache.cache_key
        corpus_fingerprint = Cache.corpus_fingerprint
        trace_rows_sha256 = "c" * 64
        layer_fields = MappingProxyType(
            {
                index: MappingProxyType(
                    {"mlp.post_swiglu": torch.zeros(3, 6)}
                )
                for index in range(3)
            }
        )

        def field(self, layer_index, name):
            return self.layer_fields[layer_index][name]

    cache = Cache()
    payload = ProductionParentCachePayload(
        cache=cache,
        rows=Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    builder = CachedProductionEvidenceBuilder(
        cache_root=tmp_path / "child-cache",
        capture_config=Lfm2CircuitTraceCaptureConfig(),
    )
    prepared = builder.build_producer_first(
        Lfm2ProducerFirstEvidenceBuildRequest(
            action=None,
            child_model=child,
            child_graph=None,
            blank_state_fingerprint=fingerprint_model_state(child),
            selected_target=None,
            parent_cache_fingerprint=cache.cache_key,
            parent_cache_payload=payload,
        )
    )

    assert cache.verified is True
    assert prepared.evidence.parent_cache_fingerprint == cache.cache_key
    assert prepared.evidence.parent_identity_sha256 == parent_identity
    assert prepared.evidence.identity()["parent_cache_key"] == cache.cache_key
    assert (
        prepared.evidence.identity()["parent_identity_sha256"]
        == parent_identity
    )


def test_hybrid_catalog_builder_only_binds_capture_from_compiler_zero(
    tmp_path,
    monkeypatch,
) -> None:
    parent = _model(40, intermediate_size=6)
    child = _model(41, intermediate_size=5)
    parent_identity = fingerprint_lfm2_parent(parent).identity_sha256

    class Cache:
        cache_key = "1" * 64
        corpus_fingerprint = "2" * 64
        parent_fingerprint = parent_identity

        def __len__(self):
            return 0

        def verify(self):
            return self

    class Rows:
        paths = ("selection", "fit", "validation")
        path_splits = ("selection", "fit", "validation")
        layer_kinds = ("short_conv", "full_attention", "short_conv")
        cache_key = Cache.cache_key
        corpus_fingerprint = Cache.corpus_fingerprint
        trace_rows_sha256 = "3" * 64
        layer_fields = MappingProxyType(
            {
                index: MappingProxyType(
                    {"mlp.post_swiglu": torch.zeros(3, 6)}
                )
                for index in range(3)
            }
        )

        def field(self, layer_index, name):
            return self.layer_fields[layer_index][name]

    payload = ProductionParentCachePayload(
        cache=Cache(),
        rows=Rows(),
        calibration_batches=(),
        calibration_splits=(),
        partition=SimpleNamespace(
            selection_indices=(0,),
            fit_indices=(1,),
            validation_indices=(2,),
        ),
    )
    builder = CachedProductionEvidenceBuilder(
        cache_root=tmp_path / "hybrid-child-cache",
        capture_config=Lfm2CircuitTraceCaptureConfig(),
    )
    capture_calls = 0

    def forbidden_capture(*_args, **_kwargs):
        nonlocal capture_calls
        capture_calls += 1
        raise AssertionError("blank child trace capture is forbidden")

    monkeypatch.setattr(
        "experiments.run_lfm2_dynamic_catalog.residual.build_child_trace_cache",
        forbidden_capture,
    )
    blank = fingerprint_model_state(child)
    prepared = builder.build_model(
        Lfm2ModelEvidenceBuildRequest(
            action=None,
            child_model=child,
            child_graph=None,
            child_state_fingerprint=blank,
            selected_target=SimpleNamespace(
                ordered_source_spans=((0,), (1,), (2,)),
                target_layer_types=(
                    "conv",
                    "full_attention",
                    "conv",
                ),
            ),
            parent_cache_fingerprint=Cache.cache_key,
            parent_cache_payload=payload,
        )
    )

    assert isinstance(prepared, Lfm2PreparedHybridProducerEvidence)
    assert prepared.child_forward_count == 0
    assert capture_calls == 0
    assert fingerprint_model_state(child) == blank
    assert builder.records[-1]["kind"] == "hybrid_parent_only_evidence"
    assert builder.records[-1]["blank_child_features_used"] is False
