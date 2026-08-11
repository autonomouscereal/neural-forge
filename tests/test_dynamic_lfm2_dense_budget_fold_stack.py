from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from neural_forge.dynamic_behavior_gate import (
    BehaviorGateBinding,
    expose_behavior_gate_binding,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_child_replay_phase_cache import (
    DynamicChildReplayPhaseCacheContext,
)
from neural_forge.dynamic_fold_compiler import DynamicFoldingCompiler
from neural_forge.dynamic_fold_executor import (
    FoldDecision,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import (
    EXTERNAL_RESOURCE_BINDINGS_CONFIGURATION_KEY,
    external_resource_bindings_sha256,
)
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache
from neural_forge.dynamic_fold_registry import ExperimentLedger
from neural_forge.dynamic_lfm2_adapter import (
    DynamicLfm2AdapterError,
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
from neural_forge.dynamic_parameter_frontier import (
    CAP_HYBRID_OPERATOR_COMPOSITION,
    CAP_ORDERED_DEPTH_COMPOSITION,
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    LFM2_CURRENT_COMPILER_CAPABILITIES,
    Lfm2DenseTargetTopologyCatalog,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (
    Lfm2BalancedBasisPhaseCacheContext,
    PhaseAnchorAuthority,
)
from neural_forge.hf_lfm2_dense_budget_model_fold import (
    DENSE_BUDGET_MODEL_METHOD,
    LFM2_SPARSE_FEATURE_FOLD_BUNDLE_RESOURCE_NAME,
    Lfm2DenseCacheVerificationLease,
    Lfm2DenseBudgetFoldConfig,
    Lfm2DenseBudgetFoldError,
    _build_residual_basis,
    _initial_structural_fold_or_authenticated_resume,
)
from neural_forge.hf_lfm2_sparse_feature_fold_bundle import (
    AXIS_MLP_POST_SWIGLU,
    AXIS_SHORT_CONV_CONTEXT,
    admit_lfm2_sparse_feature_axis,
    build_lfm2_sparse_feature_fold_bundle,
)
from neural_forge.hf_lfm2_dynamic_closed_loop_adapter import (
    Lfm2DynamicClosedLoopAdapterConfig,
)
from neural_forge.hf_lfm2_macro_cell_attractor_gate import (
    Lfm2MacroCellAttractorGateConfig,
)
from neural_forge.dynamic_closed_loop_fold import ClosedLoopCompilerConfig
from neural_forge.hf_tokenizer_bridge import (
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    seal_strict_tokenizer_bridge_artifact,
)
from neural_forge.matrix_technique_body_routes import (
    build_t5_body_route_source_witness,
    fit_body_component_routes,
)


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")


def _tokenizer_fingerprint() -> str:
    return canonical_sha256(
        {"kind": "dense-budget-fold-test-tokenizer", "vocabulary_size": 41}
    )


class _BridgeTokenizer:
    def __init__(self, surfaces: list[str]) -> None:
        self.surfaces = list(surfaces)
        self.all_special_ids: list[int] = []
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None

    def __len__(self) -> int:
        return len(self.surfaces)

    def get_vocab(self) -> dict[str, int]:
        return {
            f"raw-{index}-{surface}": index
            for index, surface in enumerate(self.surfaces)
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"raw-{token_id}-{self.surfaces[token_id]}"

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.surfaces.index(text)]


def _strict_bridge(tmp_path, *, target_vocab_size: int = 17):
    donor = _BridgeTokenizer([f"<{index}>" for index in range(41)])
    target = _BridgeTokenizer(donor.surfaces[:target_vocab_size])
    root = tmp_path / "strict-target-tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=41,
        target_row_capacity=target_vocab_size,
    )


def _model(*, hidden: int, intermediate: int, heads: int, seed: int):
    config = transformers.Lfm2Config(
        vocab_size=41,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _batch(prompt, forced):
    input_ids = torch.tensor([list(prompt)], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "forced_decode_ids": torch.tensor([list(forced)], dtype=torch.long),
    }


def _target_unique_parameters() -> int:
    vocabulary = 41
    hidden = 4
    intermediate = 6
    query_heads = 1
    key_value_heads = 1
    head_dim = 4
    fir = 3
    token = vocabulary * hidden
    body = hidden
    body += 2 * hidden + 4 * hidden * hidden + fir * hidden
    body += 3 * hidden * intermediate
    body += (
        2 * hidden
        + 2
        * hidden
        * head_dim
        * (query_heads + key_value_heads)
        + 2 * head_dim
    )
    body += 3 * hidden * intermediate
    return token + body


def _authenticated_dense_request(
    parent,
    *,
    child_name: str,
    exact_unique_parameters: int,
    blank_seed: int,
    allow_depth_composition: bool = False,
):
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    extra_capabilities = (
        (
            CAP_ORDERED_DEPTH_COMPOSITION,
            CAP_HYBRID_OPERATOR_COMPOSITION,
        )
        if allow_depth_composition
        else ()
    )
    plan = plan_budgeted_dynamic_target(
        parent_graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            target_unique_parameters=exact_unique_parameters,
            match_policy="exact",
            maximum_candidates=64,
        ),
        child_name=child_name,
        tokenizer_fingerprint=tokenizer,
        adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
        blank_seed=blank_seed,
        catalogs=DenseTargetTopologyCatalogRegistry(
            (
                Lfm2DenseTargetTopologyCatalog(
                    minimum_hidden_size=4,
                    hidden_size_multiple=4,
                    minimum_head_dim=4,
                    compiler_capabilities=(
                        *LFM2_CURRENT_COMPILER_CAPABILITIES,
                        *extra_capabilities,
                    ),
                ),
            )
        ),
    )
    assert plan.target_request.component_allocation_receipt is not None
    return plan.target_request


class _Accept:
    def evaluate(self, _context):
        return FoldDecision.accept("dense test whole-output acceptance")


class _WholeOutputAccept:
    def evaluate(self, context):
        with torch.no_grad():
            context.execution_context.child_model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return FoldDecision.accept("dense test child-root acceptance")


class _ProductionParentCachePayload:
    def __init__(self, cache, *, cache_key: str | None = None) -> None:
        self.cache = cache
        self._cache_key = (
            cache.cache_key if cache_key is None else cache_key
        )
        self.verify_calls = 0

    @property
    def cache_key(self) -> str:
        return self._cache_key

    def verify(self) -> None:
        self.verify_calls += 1
        self.cache.verify()


def _behavior_binding(cache_key: str) -> BehaviorGateBinding:
    body = {
        "schema": "test-behavior-gate-manifest-v1",
        "gate_id": "test.lfm2.dense_budget",
        "gate_version": 1,
        "baseline_fingerprint": cache_key,
    }
    fingerprint = canonical_sha256(body)
    return BehaviorGateBinding(
        gate_id="test.lfm2.dense_budget",
        gate_version=1,
        baseline_fingerprint=cache_key,
        manifest_fingerprint=fingerprint,
        manifest={**body, "manifest_fingerprint": fingerprint},
    )


def _saver(model, destination):
    torch.save(model.state_dict(), destination / "model.pt")
    (destination / "config.json").write_text(
        json.dumps(model.config.to_dict()),
        encoding="utf-8",
    )


def _loader(source):
    payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config = transformers.Lfm2Config(**payload)
    # Lfm2Config currently consumes this serialized topology alias but resets
    # it to None during direct ``**payload`` construction.  Restore the sealed
    # value before model creation so the compiler-authenticated config
    # fingerprint survives a local artifact round trip.
    config.block_ff_dim = payload["block_ff_dim"]
    model = transformers.Lfm2ForCausalLM(config)
    model.load_state_dict(
        torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    # Published artifacts must prove a frozen fresh reload; ``eval()`` alone
    # leaves ``requires_grad`` set on a freshly constructed model.
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def sealed_parent(tmp_path_factory):
    parent = _model(hidden=8, intermediate=12, heads=2, seed=2101)
    batches = [
        _batch((1, 2, 3, 4), (5,)),
        _batch((6, 7, 8, 9), (10,)),
        _batch((11, 12, 13, 14), (15,)),
        _batch((16, 17, 18, 19), (20,)),
    ]
    capture_config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=2,
        max_decode_steps=2,
        max_capture_bytes=64 * 1024 * 1024,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("dense-budget-parent-cache"),
        parent,
        batches,
        config=capture_config,
        input_device="cpu",
    )
    return parent, cache


def _synthesize_dense_target(
    parent,
    cache,
    *,
    child_name: str,
    blank_seed: int,
):
    tokenizer = _tokenizer_fingerprint()
    request = _authenticated_dense_request(
        parent,
        child_name=child_name,
        exact_unique_parameters=_target_unique_parameters(),
        blank_seed=blank_seed,
    )
    return DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (
                Lfm2TargetSynthesisAdapter(
                    terminal_topology=Lfm2Topology(1, 1, 6),
                    evidence_fingerprint=cache.cache_key,
                    device="cpu",
                    dtype=torch.float32,
                ),
            )
        ),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )


def _sparse_axis_paths(layer_index: int, axis_kind: str) -> dict[str, str]:
    if axis_kind == AXIS_MLP_POST_SWIGLU:
        prefix = f"model.layers.{layer_index}.feed_forward"
        return {
            "w1": f"{prefix}.w1.weight",
            "w3": f"{prefix}.w3.weight",
            "w2": f"{prefix}.w2.weight",
        }
    prefix = f"model.layers.{layer_index}.conv"
    packed = f"{prefix}.in_proj.weight"
    return {
        "in_proj_b": packed,
        "in_proj_c": packed,
        "in_proj_x": packed,
        "fir_kernel": f"{prefix}.conv.weight",
        "out_proj": f"{prefix}.out_proj.weight",
    }


def _short_conv_slices(width: int) -> dict[str, tuple[int, int]]:
    return {
        "in_proj_b": (0, width),
        "in_proj_c": (width, 2 * width),
        "in_proj_x": (2 * width, 3 * width),
    }


def _sparse_feature_bundle(
    parent,
    cache,
    target,
    *,
    source_graph=None,
    target_graph=None,
    parent_state_fingerprint: str | None = None,
    evidence_cache_sha256: str | None = None,
):
    tokenizer = _tokenizer_fingerprint()
    source_graph = (
        inspect_model_circuits(
            parent,
            model_name="parent",
            tokenizer_fingerprint=tokenizer,
        )
        if source_graph is None
        else source_graph
    )
    target_graph = target.graph if target_graph is None else target_graph
    parent_state = (
        fingerprint_model_state(parent)
        if parent_state_fingerprint is None
        else parent_state_fingerprint
    )
    cache_sha = (
        cache.cache_key if evidence_cache_sha256 is None else evidence_cache_sha256
    )
    axes = []
    axis_specs = (
        (0, AXIS_MLP_POST_SWIGLU),
        (0, AXIS_SHORT_CONV_CONTEXT),
        (1, AXIS_MLP_POST_SWIGLU),
    )
    for ordinal, (layer_index, axis_kind) in enumerate(axis_specs):
        source_layer = parent.model.layers[layer_index]
        target_layer = target.model.model.layers[layer_index]
        if axis_kind == AXIS_MLP_POST_SWIGLU:
            component = "C13"
            endpoint = "mlp.post_swiglu"
            graph_site = f"model.layers.{layer_index}.feed_forward"
            parent_weight = source_layer.feed_forward.w2.weight
            target_width = int(target_layer.feed_forward.w2.weight.shape[1])
        else:
            component = "C05"
            endpoint = "short_conv.context"
            graph_site = f"model.layers.{layer_index}.conv"
            parent_weight = source_layer.conv.out_proj.weight
            target_width = int(target_layer.conv.out_proj.weight.shape[1])
        parent_weight = (
            parent_weight.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        source_width = int(parent_weight.shape[1])
        generator = torch.Generator().manual_seed(31_000 + ordinal)
        component_rows = torch.randn(
            (2 * source_width + 3, source_width),
            generator=generator,
            dtype=torch.float64,
        )
        heldout_rows = torch.randn(
            (source_width + 2, source_width),
            generator=generator,
            dtype=torch.float64,
        )
        authority_site = f"L{layer_index:02d}"
        fold = fit_body_component_routes(
            parent_weight=parent_weight,
            component_rows=component_rows,
            heldout_component_rows=heldout_rows,
            component=component,
            level="X4",
            site=authority_site,
            kept_width=target_width,
            active_k=min(2, target_width),
            channel_selection="explicit",
            kept_channel_ids=tuple(range(target_width)),
            install_coefficients="write",
        )
        paths = _sparse_axis_paths(layer_index, axis_kind)
        selection_sha = canonical_sha256(
            {
                "schema": "test-lfm2-sparse-feature-selection-v1",
                "layer_index": layer_index,
                "axis_kind": axis_kind,
                "kept_channel_ids": list(range(target_width)),
            }
        )
        writer_role = "w2" if axis_kind == AXIS_MLP_POST_SWIGLU else "out_proj"
        witness = build_t5_body_route_source_witness(
            fold,
            parent_weight=parent_weight,
            component_rows=component_rows,
            heldout_component_rows=heldout_rows,
            parameter_path=paths[writer_role],
            graph_fingerprint=source_graph.fingerprint,
            parent_state_fingerprint=parent_state,
            component_id=fold.component_id,
            layer_index=layer_index,
            site=authority_site,
            role="writer",
            evidence_cache_sha256=cache_sha,
            evidence_selection_sha256=selection_sha,
            evidence_endpoint=endpoint,
            coefficient_set="write",
        )
        source_artifacts = {
            "parent_state": witness.parent_state_fingerprint,
            "component_evidence_cache": witness.evidence_cache_sha256,
            "component_evidence_selection": (witness.evidence_selection_sha256),
            "body_route_record": witness.body_route_record_sha256,
            "body_route_source_witness": witness.witness_sha256,
        }
        axis_kwargs = {
            "source_graph": source_graph,
            "target_graph": target_graph,
            "source_graph_sha256": source_graph.fingerprint,
            "target_graph_sha256": target_graph.fingerprint,
            "layer_index": layer_index,
            "site": graph_site,
            "axis_kind": axis_kind,
            "source_parameter_paths": paths,
            "target_parameter_paths": paths,
            "body_route_fold": fold,
            "source_witness": witness,
            "coefficient_set": "write",
            "source_artifact_sha256s": source_artifacts,
        }
        if axis_kind == AXIS_SHORT_CONV_CONTEXT:
            axis_kwargs.update(
                source_parameter_row_slices=_short_conv_slices(source_width),
                target_parameter_row_slices=_short_conv_slices(target_width),
            )
        axes.append(admit_lfm2_sparse_feature_axis(**axis_kwargs))
    return build_lfm2_sparse_feature_fold_bundle(
        source_graph=source_graph,
        target_graph=target_graph,
        source_graph_sha256=source_graph.fingerprint,
        target_graph_sha256=target_graph.fingerprint,
        axes=axes,
        source_artifact_sha256s={
            "parent_trace_cache": cache_sha,
            "parent_state": parent_state,
        },
    )


def _dense_stack_resources(cache, *, sparse_feature_fold_bundle=None):
    final_gate = _Accept()

    def fresh_gate(_model):
        return FoldDecision.accept("unused fresh sparse-bundle gate")

    binding = _behavior_binding(cache.cache_key)
    expose_behavior_gate_binding(final_gate, binding)
    expose_behavior_gate_binding(fresh_gate, binding)
    return Lfm2DynamicStackResources(
        parent_cache_fingerprint=cache.cache_key,
        parent_cache_payload=cache,
        evidence_builder=lambda _request: None,
        final_evaluator=final_gate,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=fresh_gate,
        saver_id="test-sparse-feature-bundle-saver-v1",
        loader_id="test-sparse-feature-bundle-loader-v1",
        ledger=ExperimentLedger(),
        dense_fold_config=Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="weight_complete",
            run_closed_loop=False,
        ),
        sparse_feature_fold_bundle=sparse_feature_fold_bundle,
    )


def test_dynamic_lfm2_system_sparse_feature_bundle_import_is_lazy() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys\n"
            "import neural_forge.dynamic_lfm2_system\n"
            "assert 'neural_forge.hf_lfm2_sparse_feature_fold_bundle' "
            "not in sys.modules\n"
            "assert 'neural_forge.matrix_technique_body_routes' "
            "not in sys.modules\n",
        ),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_dense_stack_sparse_feature_bundle_is_typed_and_legacy_empty(
    sealed_parent,
) -> None:
    parent, cache = sealed_parent
    target = _synthesize_dense_target(
        parent,
        cache,
        child_name="dense-sparse-bundle-typing",
        blank_seed=22_001,
    )

    with pytest.raises(
        DynamicLfm2AdapterError,
        match="sparse_feature_fold_bundle has the wrong type",
    ):
        _dense_stack_resources(
            cache,
            sparse_feature_fold_bundle=object(),
        )

    stack = Lfm2DynamicFoldStackAdapter(_dense_stack_resources(cache)).build(target)
    operator = stack.operators.get(DENSE_BUDGET_MODEL_METHOD)
    assert stack.resource_fingerprints == ()
    assert operator.sparse_feature_fold_bundle is None
    assert operator.resource_fingerprint is None


def test_dense_stack_threads_sparse_feature_bundle_into_action_identity(
    sealed_parent,
) -> None:
    parent, cache = sealed_parent
    parent_state = fingerprint_model_state(parent)
    target = _synthesize_dense_target(
        parent,
        cache,
        child_name="dense-sparse-bundle-action",
        blank_seed=22_002,
    )
    bundle = _sparse_feature_bundle(parent, cache, target)
    adapter = Lfm2DynamicFoldStackAdapter(
        _dense_stack_resources(
            cache,
            sparse_feature_fold_bundle=bundle,
        )
    )

    assert adapter.supports(target)
    stack = adapter.build(target)
    operator = stack.operators.get(DENSE_BUDGET_MODEL_METHOD)
    expected_resources = {
        LFM2_SPARSE_FEATURE_FOLD_BUNDLE_RESOURCE_NAME: bundle.bundle_sha256
    }
    expected_aggregate = external_resource_bindings_sha256(expected_resources)
    assert operator.sparse_feature_fold_bundle is bundle
    assert stack.resource_fingerprints == tuple(sorted(expected_resources.items()))
    assert operator.resource_fingerprint == expected_aggregate

    compilation = DynamicFoldingCompiler(
        operators=stack.operators,
        ledger=ExperimentLedger(),
    ).compile(
        parent_models={"parent": parent},
        child_model=target.model,
        child_name=target.graph.model_name,
        evidence_provider=stack.evidence_provider,
        evaluator=stack.evaluator,
        final_evaluator=stack.final_evaluator,
        evidence_fingerprint=cache.cache_key,
        parent_tokenizer_fingerprints={"parent": _tokenizer_fingerprint()},
        child_tokenizer_fingerprint=_tokenizer_fingerprint(),
        exact_target_unique_parameters=_target_unique_parameters(),
        initializer=stack.initializer,
        resource_fingerprints=dict(stack.resource_fingerprints),
    )

    assert compilation.complete
    assert len(compilation.plan.actions) == 1
    action = compilation.plan.actions[0]
    assert (
        dict(action.configuration)[EXTERNAL_RESOURCE_BINDINGS_CONFIGURATION_KEY]
        == expected_aggregate
    )
    assert action.signature.configuration == action.configuration
    assert fingerprint_model_state(parent) == parent_state


@pytest.mark.parametrize(
    ("foreign_binding", "message"),
    (
        ("source_graph", "source graph differs"),
        ("target_graph", "target graph differs"),
        ("parent_state", "parent state differs"),
        ("evidence_cache", "evidence cache differs"),
    ),
)
def test_dense_stack_rejects_foreign_sparse_feature_bundle_bindings(
    sealed_parent,
    foreign_binding,
    message,
) -> None:
    parent, cache = sealed_parent
    target = _synthesize_dense_target(
        parent,
        cache,
        child_name=f"dense-sparse-bundle-foreign-{foreign_binding}",
        blank_seed=22_100 + len(foreign_binding),
    )
    source_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    bundle_kwargs = {}
    if foreign_binding == "source_graph":
        bundle_kwargs["source_graph"] = replace(
            source_graph,
            model_name="foreign-parent",
        )
    elif foreign_binding == "target_graph":
        bundle_kwargs["target_graph"] = replace(
            target.graph,
            model_name="foreign-child",
        )
    elif foreign_binding == "parent_state":
        bundle_kwargs["parent_state_fingerprint"] = "f" * 64
    else:
        bundle_kwargs["evidence_cache_sha256"] = "e" * 64
    bundle = _sparse_feature_bundle(
        parent,
        cache,
        target,
        **bundle_kwargs,
    )
    adapter = Lfm2DynamicFoldStackAdapter(
        _dense_stack_resources(
            cache,
            sparse_feature_fold_bundle=bundle,
        )
    )

    assert not adapter.supports(target)
    with pytest.raises(DynamicLfm2AdapterError, match=message):
        adapter.build(target)


def test_dense_stack_reauthenticates_sparse_feature_bundle_after_resource_init(
    sealed_parent,
) -> None:
    parent, cache = sealed_parent
    target = _synthesize_dense_target(
        parent,
        cache,
        child_name="dense-sparse-bundle-tamper",
        blank_seed=22_003,
    )
    bundle = _sparse_feature_bundle(parent, cache, target)
    resources = _dense_stack_resources(
        cache,
        sparse_feature_fold_bundle=bundle,
    )
    object.__setattr__(bundle, "bundle_sha256", "f" * 64)
    adapter = Lfm2DynamicFoldStackAdapter(resources)

    assert not adapter.supports(target)
    with pytest.raises(
        DynamicLfm2AdapterError,
        match="bundle failed authentication",
    ):
        adapter.build(target)


def test_non_dense_lfm2_target_refuses_unused_sparse_feature_bundle(
    sealed_parent,
) -> None:
    parent, cache = sealed_parent
    dense_target = _synthesize_dense_target(
        parent,
        cache,
        child_name="dense-sparse-bundle-for-legacy-refusal",
        blank_seed=22_004,
    )
    bundle = _sparse_feature_bundle(parent, cache, dense_target)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    legacy_target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (
                Lfm2TargetSynthesisAdapter(
                    terminal_topology=Lfm2Topology(2, 2, 11),
                    evidence_fingerprint=cache.cache_key,
                    device="cpu",
                    dtype=torch.float32,
                ),
            )
        ),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="legacy-lfm2-sparse-bundle-refusal",
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=(parent_graph.unique_parameter_count - 2 * 3 * 8),
            tokenizer_fingerprint=tokenizer,
            blank_seed=22_005,
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    adapter = Lfm2DynamicFoldStackAdapter(
        _dense_stack_resources(
            cache,
            sparse_feature_fold_bundle=bundle,
        )
    )

    assert not adapter.supports(legacy_target)
    with pytest.raises(
        DynamicLfm2AdapterError,
        match="only consumed by dense budget",
    ):
        adapter.build(legacy_target)


def test_authenticated_resume_skips_structural_fold_while_fresh_executes_it() -> None:
    calls = []
    events = []

    class _Progress:
        def emit(self, stage, **metadata):
            events.append((stage, dict(metadata)))

    def structural_fold():
        calls.append("structural-fold")
        return {"schema": "fresh-structural-fold-report"}

    fresh = _initial_structural_fold_or_authenticated_resume(
        restart_resume=None,
        structural_fold=structural_fold,
        fold_progress=_Progress(),
    )
    assert fresh == {"schema": "fresh-structural-fold-report"}
    assert calls == ["structural-fold"]

    resume = SimpleNamespace(
        reference=SimpleNamespace(
            capsule_key="capsule-key",
            manifest_sha256="a" * 64,
        ),
        child_state_sha256="b" * 64,
        child_topology_sha256="c" * 64,
    )
    resumed = _initial_structural_fold_or_authenticated_resume(
        restart_resume=resume,
        structural_fold=structural_fold,
        fold_progress=_Progress(),
    )
    assert calls == ["structural-fold"]
    assert resumed["structural_parent_to_blank_fold_executed"] is False
    assert resumed["capsule_key"] == "capsule-key"
    assert resumed["parent_forward_calls"] == 0
    assert events[-1][0] == "dense_structural_fold_skipped_restart_resume"


@pytest.fixture(scope="module")
def production_budget_parent(tmp_path_factory):
    config = transformers.Lfm2Config(
        vocab_size=41,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2300)
        parent = transformers.Lfm2ForCausalLM(config)
    parent.eval()
    batches = [
        _batch((1, 2, 3, 4), (5,)),
        _batch((6, 7, 8, 9), (10,)),
        _batch((11, 12, 13, 14), (15,)),
        _batch((16, 17, 18, 19), (20,)),
    ]
    cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("production-dense-budget-parent-cache"),
        parent,
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=1,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    return parent, cache


@pytest.fixture(scope="module")
def production_ten_x_parent(tmp_path_factory):
    layer_types = ["conv", "full_attention"] * 7
    config = transformers.Lfm2Config(
        vocab_size=41,
        hidden_size=128,
        intermediate_size=320,
        num_hidden_layers=len(layer_types),
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        layer_types=layer_types,
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2400)
        parent = transformers.Lfm2ForCausalLM(config)
    parent.eval()
    batches = [
        _batch((1, 2, 3, 4), (5,)),
        _batch((6, 7, 8, 9), (10,)),
        _batch((11, 12, 13, 14), (15,)),
        _batch((16, 17, 18, 19), (20,)),
    ]
    cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("production-dense-budget-10x-parent-cache"),
        parent,
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=1,
            max_decode_steps=1,
            max_capture_bytes=128 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    return parent, cache


def _production_extreme_ratio_parent(
    tmp_path_factory,
    *,
    ratio: int,
    intermediate_size: int,
    seed: int,
):
    config = transformers.Lfm2Config(
        vocab_size=41,
        hidden_size=224,
        intermediate_size=intermediate_size,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=7,
        head_dim=16,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        parent = transformers.Lfm2ForCausalLM(config)
    parent.eval()
    batches = [
        _batch((1, 2, 3, 4), (5,)),
        _batch((6, 7, 8, 9), (10,)),
        _batch((11, 12, 13, 14), (15,)),
        _batch((16, 17, 18, 19), (20,)),
    ]
    cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp(
            f"production-dense-budget-{ratio}x-parent-cache"
        ),
        parent,
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=1,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    return parent, cache


@pytest.fixture(scope="module")
def production_hundred_x_parent(tmp_path_factory):
    """Small full-depth parent just above the honest dense 100x floor."""

    return _production_extreme_ratio_parent(
        tmp_path_factory,
        ratio=100,
        intermediate_size=687,
        seed=2500,
    )


@pytest.fixture(scope="module")
def production_two_hundred_fifty_x_parent(tmp_path_factory):
    """Smallest integer FFN width clearing the same dense 250x floor."""

    return _production_extreme_ratio_parent(
        tmp_path_factory,
        ratio=250,
        intermediate_size=2_121,
        seed=2750,
    )


@pytest.fixture(scope="module")
def production_thousand_x_parent(tmp_path_factory):
    """Exact 1000x parent over the same full-depth 12,848-param child."""

    return _production_extreme_ratio_parent(
        tmp_path_factory,
        ratio=1_000,
        intermediate_size=9_290,
        seed=3500,
    )


def test_dense_budget_vocab_margin_requires_circuit_balanced_basis(
    tmp_path,
) -> None:
    defaults = Lfm2DenseBudgetFoldConfig()
    assert defaults.child_replay_phase_cache_context is None
    assert defaults.macro_cell_attractor_gate_config is None
    assert defaults.streaming_vocabulary_margin_v2 is False
    assert defaults.streaming_vocabulary_margin_pair_chunk_size == 256
    assert defaults.vocabulary_margin_candidate_gate_enabled is False
    assert defaults.vocabulary_margin_candidate_replay_mode == (
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )
    assert (
        defaults
        .vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
        == pytest.approx(0.9)
    )
    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="requires residual_basis_mode='circuit_balanced'",
    ):
        Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="weight_complete",
            vocabulary_margin_observability=True,
        )
    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="requires vocabulary_margin_observability=True",
    ):
        Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="circuit_balanced",
            vocabulary_margin_candidate_gate_enabled=True,
        )
    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="requires vocabulary_margin_observability=True",
    ):
        Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="circuit_balanced",
            streaming_vocabulary_margin_v2=True,
        )
    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="requires run_closed_loop=True",
    ):
        Lfm2DenseBudgetFoldConfig(
            run_closed_loop=False,
            macro_cell_attractor_gate_config=(
                Lfm2MacroCellAttractorGateConfig()
            ),
        )
    context = DynamicChildReplayPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "child-replay-phase-test-cache")
    )
    assert Lfm2DenseBudgetFoldConfig(
        child_replay_phase_cache_context=context
    ).child_replay_phase_cache_context is context
    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="child_replay_phase_cache_context has the wrong type",
    ):
        Lfm2DenseBudgetFoldConfig(
            child_replay_phase_cache_context=object()
        )


def test_dense_budget_balanced_basis_consumes_cached_vocab_margin(
    sealed_parent,
) -> None:
    parent, cache = sealed_parent
    parent_state = fingerprint_model_state(parent)
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    basis = _build_residual_basis(
        parent,
        graph,
        cache,
        4,
        Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="circuit_balanced",
            run_closed_loop=False,
            vocabulary_margin_observability=True,
            vocabulary_margin_top_k=3,
            vocabulary_margin_floor_fraction=0.05,
        ),
    )

    report = basis.as_dict()
    margin = report["vocabulary_margin_observability"]
    assert margin["enabled"] is True
    assert margin["full_parent_logits_read_from_sealed_cache"] is True
    assert margin["parent_forward_calls"] == 0
    assert margin["trace_regeneration_calls"] == 0
    assert margin["training_steps"] == 0
    projection_gate = basis.evaluate_vocabulary_margin_projection_gate(
        basis.basis,
        minimum_retained_energy_fraction=0.0,
        minimum_sign_agreement_fraction=0.0,
        minimum_top_competitor_nonnegative_fraction=0.0,
        minimum_all_top_k_nonnegative_fraction=0.0,
    )
    assert projection_gate["gate_passed"] is True
    assert fingerprint_model_state(parent) == parent_state


def _compile_vocabulary_margin_gated_child(
    sealed_parent,
    *,
    minimum_fraction: float,
    child_name: str,
    parent_cache_payload=None,
    parent_cache_integrity_check=None,
    dense_fold_config=None,
    replay_mode: str = CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    stack_sink=None,
    residual_basis_phase_cache_context=None,
    dense_fold_progress_sink=None,
):
    parent, cache = sealed_parent
    tokenizer = _tokenizer_fingerprint()
    exact = _target_unique_parameters()
    request = _authenticated_dense_request(
        parent,
        child_name=child_name,
        exact_unique_parameters=exact,
        blank_seed=2501,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (
                Lfm2TargetSynthesisAdapter(
                    terminal_topology=Lfm2Topology(1, 1, 6),
                    evidence_fingerprint=cache.cache_key,
                    device="cpu",
                    dtype=torch.float32,
                ),
            )
        ),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    blank_state = fingerprint_model_state(target.model)
    final_gate = _Accept()

    def fresh_gate(_model):
        return FoldDecision.accept("unused fresh gate")

    binding = _behavior_binding(cache.cache_key)
    expose_behavior_gate_binding(final_gate, binding)
    expose_behavior_gate_binding(fresh_gate, binding)
    stack = Lfm2DynamicFoldStackAdapter(
        Lfm2DynamicStackResources(
            parent_cache_fingerprint=cache.cache_key,
            parent_cache_payload=(
                cache
                if parent_cache_payload is None
                else parent_cache_payload
            ),
            evidence_builder=lambda _request: None,
            final_evaluator=final_gate,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=fresh_gate,
            saver_id="test-vocabulary-margin-gate-saver-v1",
            loader_id="test-vocabulary-margin-gate-loader-v1",
            ledger=ExperimentLedger(),
            parent_cache_integrity_check=parent_cache_integrity_check,
            residual_basis_phase_cache_context=(
                residual_basis_phase_cache_context
            ),
            dense_fold_progress_sink=dense_fold_progress_sink,
            dense_fold_config=(
                dense_fold_config
                if dense_fold_config is not None
                else Lfm2DenseBudgetFoldConfig(
                    residual_basis_mode="circuit_balanced",
                    run_closed_loop=False,
                    circuit_split_seed=(
                        "dense-budget-vocabulary-margin-candidate-gate-v1"
                    ),
                    vocabulary_margin_observability=True,
                    vocabulary_margin_top_k=3,
                    vocabulary_margin_candidate_gate_enabled=True,
                    vocabulary_margin_candidate_replay_mode=replay_mode,
                    vocabulary_margin_candidate_minimum_sign_agreement_fraction=(
                        minimum_fraction
                    ),
                    vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction=(
                        minimum_fraction
                    ),
                    vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction=(
                        minimum_fraction
                    ),
                    vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction=(
                        minimum_fraction
                    ),
                )
            ),
        )
    ).build(target)
    if stack_sink is not None:
        stack_sink.append(stack)
    compilation = DynamicFoldingCompiler(
        operators=stack.operators,
        ledger=ExperimentLedger(),
    ).compile(
        parent_models={"parent": parent},
        child_model=target.model,
        child_name=child_name,
        evidence_provider=stack.evidence_provider,
        evaluator=stack.evaluator,
        final_evaluator=stack.final_evaluator,
        evidence_fingerprint=cache.cache_key,
        parent_tokenizer_fingerprints={"parent": tokenizer},
        child_tokenizer_fingerprint=tokenizer,
        exact_target_unique_parameters=exact,
        initializer=stack.initializer,
    )
    return compilation, target, blank_state


def test_dynamic_stack_forwards_typed_balanced_basis_phase_context(
    sealed_parent,
    tmp_path,
) -> None:
    context = Lfm2BalancedBasisPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "balanced-basis"),
        anchor_authority=PhaseAnchorAuthority(
            key_id="dynamic-stack-test-v1",
            secret=b"x" * 32,
        ),
    )
    stacks = []
    progress_events = []

    compilation, _target, _blank_state = (
        _compile_vocabulary_margin_gated_child(
            sealed_parent,
            minimum_fraction=0.0,
            child_name="dense-balanced-basis-context",
            stack_sink=stacks,
            residual_basis_phase_cache_context=context,
            dense_fold_progress_sink=(
                lambda stage, metadata: progress_events.append(
                    (stage, dict(metadata))
                )
            ),
        )
    )

    assert compilation.complete
    operator = stacks[0].operators.get(DENSE_BUDGET_MODEL_METHOD)
    assert operator.residual_basis_phase_cache_context is context
    assert operator.progress_sink is not None
    assert progress_events[0][0] == "dense_structural_token_transport_started"
    assert progress_events[-1][0] == "dense_structural_fold_completed"


def test_dynamic_stack_rejects_untyped_balanced_basis_phase_context() -> None:
    with pytest.raises(
        DynamicLfm2AdapterError,
        match="residual_basis_phase_cache_context has the wrong type",
    ):
        Lfm2DynamicStackResources(
            parent_cache_fingerprint="a" * 64,
            parent_cache_payload=object(),
            evidence_builder=lambda _request: None,
            final_evaluator=_Accept(),
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=lambda _model: True,
            saver_id="invalid-context-saver-v1",
            loader_id="invalid-context-loader-v1",
            ledger=ExperimentLedger(),
            residual_basis_phase_cache_context=object(),
        )


def test_dense_budget_stack_unwraps_verified_production_cache_payload(
    sealed_parent,
) -> None:
    _parent, cache = sealed_parent
    payload = _ProductionParentCachePayload(cache)

    compilation, _target, _blank_state = (
        _compile_vocabulary_margin_gated_child(
            sealed_parent,
            minimum_fraction=0.0,
            child_name="dense-production-cache-payload",
            parent_cache_payload=payload,
            replay_mode=CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        )
    )

    assert compilation.complete
    assert payload.verify_calls == 1


def test_dense_budget_stack_reuses_verified_cache_through_identity_lease(
    sealed_parent,
    monkeypatch,
) -> None:
    _parent, cache = sealed_parent
    payload = _ProductionParentCachePayload(cache)
    identity_checks = []
    stacks = []

    def cheap_identity_check(candidate) -> str:
        assert candidate is payload
        assert candidate.cache is cache
        identity_checks.append(candidate.cache_key)
        return candidate.cache_key

    def forbidden_full_verify(_cache):
        raise AssertionError("verification lease unexpectedly rescanned cache")

    monkeypatch.setattr(
        Lfm2CircuitTraceCache,
        "verify",
        forbidden_full_verify,
    )
    compilation, target, _blank_state = _compile_vocabulary_margin_gated_child(
        sealed_parent,
        minimum_fraction=0.0,
        child_name="dense-production-cache-verification-lease",
        parent_cache_payload=payload,
        parent_cache_integrity_check=cheap_identity_check,
        dense_fold_config=Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="weight_complete",
            run_closed_loop=False,
        ),
        stack_sink=stacks,
    )

    assert compilation.complete
    assert payload.verify_calls == 0
    assert identity_checks
    assert set(identity_checks) == {cache.cache_key}
    provider = stacks[0].evidence_provider
    state_fingerprint = fingerprint_model_state(target.model)
    provider.restore(
        SimpleNamespace(child_state_fingerprint=state_fingerprint),
        state_fingerprint,
        (),
    )
    assert provider.prepare_count >= 1
    assert provider.refresh_count == 1
    assert provider.restore_count == 1


def test_dense_budget_cache_verification_lease_accepts_authenticated_session(
    sealed_parent,
    monkeypatch,
) -> None:
    _parent, cache = sealed_parent

    class _IdentitySession:
        def __init__(self) -> None:
            self.cache_key = cache.cache_key
            self.verify_calls = 0

        def verify(self):
            self.verify_calls += 1
            return self

    def forbidden_full_verify(_cache):
        raise AssertionError("authenticated session unexpectedly rescanned cache")

    monkeypatch.setattr(
        Lfm2CircuitTraceCache,
        "verify",
        forbidden_full_verify,
    )
    session = _IdentitySession()
    lease = Lfm2DenseCacheVerificationLease(
        cache,
        cache.cache_key,
        authenticated_session=session,
    )

    assert lease.verify(cache) is cache
    assert session.verify_calls == 2
    session.cache_key = "f" * 64
    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="identity differs",
    ):
        lease.verify(cache)


def test_dense_budget_stack_rejects_payload_without_typed_trace_cache(
    sealed_parent,
) -> None:
    _parent, cache = sealed_parent
    payload = _ProductionParentCachePayload(
        object(),
        cache_key=cache.cache_key,
    )

    with pytest.raises(
        DynamicLfm2AdapterError,
        match="verified production payload containing one",
    ):
        _compile_vocabulary_margin_gated_child(
            sealed_parent,
            minimum_fraction=0.0,
            child_name="dense-invalid-production-cache-payload",
            parent_cache_payload=payload,
        )
    assert payload.verify_calls == 0


def test_dense_budget_candidate_gate_replays_once_and_commits(
    sealed_parent,
) -> None:
    parent, _cache = sealed_parent
    parent_state = fingerprint_model_state(parent)
    compilation, target, blank_state = (
        _compile_vocabulary_margin_gated_child(
            sealed_parent,
            minimum_fraction=0.0,
            child_name="dense-vocabulary-gate-pass",
        )
    )

    assert compilation.complete
    assert fingerprint_model_state(target.model) != blank_state
    assert fingerprint_model_state(parent) == parent_state
    assert compilation.run.observed_parent_module_forward_calls == 0
    assert compilation.run.observed_parent_root_forward_calls == 0
    attempt = compilation.run.attempts[0]
    assert attempt.accepted
    audit = attempt.transaction_report[
        "vocabulary_margin_candidate_gate"
    ]
    assert audit["enabled"] is True
    assert audit["gate_passed"] is True
    assert audit["candidate_replay_capture_calls"] == 1
    assert audit["candidate_replay_mode"] == (
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )
    assert audit["single_pass_causal_authenticated"] is True
    assert audit["heldout_record_count"] == 1
    assert audit["actual_child_forward_calls"] == 1
    expected_canonical_prefix_forwards = audit["evaluation"]["phase_count"]
    assert audit["canonical_prefix_child_forward_calls"] == (
        expected_canonical_prefix_forwards
    )
    assert audit["saved_child_forward_calls"] == (
        expected_canonical_prefix_forwards
        - audit["actual_child_forward_calls"]
    )
    assert audit["parent_forward_calls"] == 0
    assert audit["trace_regeneration_calls"] == 0
    assert audit["training_steps"] == 0
    assert audit["evaluation"]["candidate_forward_calls"] == 0


def test_dense_budget_candidate_metric_shortfall_is_retained_for_review(
    sealed_parent,
) -> None:
    parent, _cache = sealed_parent
    parent_state = fingerprint_model_state(parent)
    compilation, target, blank_state = (
        _compile_vocabulary_margin_gated_child(
            sealed_parent,
            minimum_fraction=1.0,
            child_name="dense-vocabulary-gate-reject",
            replay_mode=CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        )
    )

    assert compilation.complete is True
    assert fingerprint_model_state(target.model) != blank_state
    assert fingerprint_model_state(parent) == parent_state
    assert compilation.run.observed_parent_module_forward_calls == 0
    assert compilation.run.observed_parent_root_forward_calls == 0
    attempt = compilation.run.attempts[0]
    assert attempt.accepted is True
    assert attempt.rollback_exact is None
    audit = attempt.transaction_report[
        "vocabulary_margin_candidate_gate"
    ]
    assert audit["enabled"] is True
    assert audit["gate_passed"] is False
    assert audit["candidate_replay_capture_calls"] == 1
    assert audit["actual_child_forward_calls"] > 0
    assert audit["parent_forward_calls"] == 0
    assert audit["training_steps"] == 0
    assert attempt.transaction_report["action"] == "commit"
    assert attempt.transaction_report["accepted"] is True
    decision_metrics = dict(attempt.decision.metrics)
    assert decision_metrics[
        "vocabulary_margin_candidate_is_decision_authority"
    ] is False
    evaluation = audit["evaluation"]
    for name, value in evaluation["metrics"].items():
        if name in evaluation["minimums"]:
            assert decision_metrics[
                f"vocabulary_margin_candidate_{name}"
            ] == pytest.approx(value)
            assert decision_metrics[
                f"vocabulary_margin_candidate_minimum_{name}"
            ] == pytest.approx(evaluation["minimums"][name])
            assert decision_metrics[
                f"vocabulary_margin_candidate_check_{name}"
            ] is evaluation["checks"][name]
    assert decision_metrics[
        "vocabulary_margin_candidate_heldout_record_count"
    ] == audit["heldout_record_count"]
    assert decision_metrics[
        "vocabulary_margin_candidate_actual_child_forward_calls"
    ] == audit["actual_child_forward_calls"]
    assert decision_metrics[
        "vocabulary_margin_candidate_parent_forward_calls"
    ] == 0
    assert decision_metrics[
        "vocabulary_margin_candidate_training_steps"
    ] == 0
    assert decision_metrics[
        "vocabulary_margin_candidate_audit_sha256"
    ] == audit["audit_sha256"]


def test_dense_budget_stack_fills_blank_with_parent_structural_fold(
    sealed_parent,
) -> None:
    parent, cache = sealed_parent
    parent_state = fingerprint_model_state(parent)
    tokenizer = _tokenizer_fingerprint()
    exact = _target_unique_parameters()
    request = _authenticated_dense_request(
        parent,
        child_name="dense-child",
        exact_unique_parameters=exact,
        blank_seed=2102,
    )
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 6),
        evidence_fingerprint=cache.cache_key,
        device="cpu",
        dtype=torch.float32,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((target_adapter,)),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    blank_state = fingerprint_model_state(target.model)
    assert all(
        int(torch.count_nonzero(parameter).item()) == 0
        for parameter in target.model.parameters()
    )
    final_gate = _Accept()

    def fresh_gate(_model):
        return FoldDecision.accept("fresh dense reload accepted")

    binding = _behavior_binding(cache.cache_key)
    expose_behavior_gate_binding(final_gate, binding)
    expose_behavior_gate_binding(fresh_gate, binding)
    stack = Lfm2DynamicFoldStackAdapter(
        Lfm2DynamicStackResources(
            parent_cache_fingerprint=cache.cache_key,
            parent_cache_payload=cache,
            evidence_builder=lambda _request: None,
            final_evaluator=final_gate,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=fresh_gate,
            saver_id="test-dense-saver-v1",
            loader_id="test-dense-loader-v1",
            ledger=ExperimentLedger(),
            dense_fold_config=Lfm2DenseBudgetFoldConfig(
                residual_basis_mode="weight_complete",
                run_closed_loop=True,
                closed_loop_adapter_config=(
                    Lfm2DynamicClosedLoopAdapterConfig(
                        ridge=1e-5,
                        norm_ridge=1e-5,
                        max_prior_delta_ratio=4.0,
                        cluster_max_iterations=4,
                        cluster_distance_chunk_size=16,
                        minimum_layer_relative_rmse_improvement=-1_000_000.0,
                        minimum_final_norm_relative_rmse_improvement=(
                            -1_000_000.0
                        ),
                        minimum_output_kl_improvement=-1_000_000.0,
                        minimum_output_known_sequence_nll_improvement=(
                            -1_000_000.0
                        ),
                        output_validation_tolerance=1_000_000.0,
                    )
                ),
                closed_loop_compiler_config=ClosedLoopCompilerConfig(
                    validation_fraction=0.25,
                    split_seed="dense-budget-stack-test-v1",
                ),
            ),
        )
    ).build(target)
    compilation = DynamicFoldingCompiler(
        operators=stack.operators,
        ledger=ExperimentLedger(),
    ).compile(
        parent_models={"parent": parent},
        child_model=target.model,
        child_name="dense-child",
        evidence_provider=stack.evidence_provider,
        evaluator=stack.evaluator,
        final_evaluator=stack.final_evaluator,
        evidence_fingerprint=cache.cache_key,
        parent_tokenizer_fingerprints={"parent": tokenizer},
        child_tokenizer_fingerprint=tokenizer,
        exact_target_unique_parameters=exact,
        initializer=stack.initializer,
    )

    assert compilation.complete
    assert compilation.run.status == "complete"
    assert compilation.plan.actions[0].method_id == DENSE_BUDGET_MODEL_METHOD
    assert fingerprint_model_state(target.model) != blank_state
    assert fingerprint_model_state(parent) == parent_state
    assert target.graph.unique_parameter_count == exact
    assert sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in target.model.parameters()
    ) > 0
    assert compilation.run.optimizer_constructions == 0
    assert compilation.run.optimizer_steps == 0
    assert compilation.run.backward_calls == 0
    assert compilation.run.autograd_grad_calls == 0
    assert compilation.run.observed_parent_module_forward_calls == 0
    assert compilation.run.observed_parent_root_forward_calls == 0
    attempt = compilation.run.attempts[0]
    assert attempt.accepted
    assert attempt.method_id == DENSE_BUDGET_MODEL_METHOD
    transaction_report = dict(attempt.transaction_report)
    assert transaction_report["closed_loop_enabled"] is True
    assert transaction_report["complete_target_parameter_overwrite"] is True
    assert transaction_report["residual_basis_mode"] == "weight_complete"
    assert transaction_report["quantization_used"] is False
    assert transaction_report["pruning_used"] is False
    assert transaction_report["training_used"] is False
    assert transaction_report["initial_structural_fold"][
        "complete_target_parameter_overwrite"
    ]
    assert transaction_report["closed_loop"]["audit"]["training_steps"] == 0
    assert transaction_report["closed_loop"]["audit"]["backward_calls"] == 0
    assert transaction_report["closed_loop"]["audit"]["optimizer_steps"] == 0


def test_strict_tokenizer_bridge_materializes_and_folds_reduced_vocab_rows(
    sealed_parent,
    tmp_path,
) -> None:
    parent, cache = sealed_parent
    artifact = _strict_bridge(tmp_path)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=(
            artifact.donor_tokenizer_semantics_sha256
        ),
    )
    catalogs = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                minimum_hidden_size=4,
                hidden_size_multiple=4,
                minimum_head_dim=4,
                tokenizer_bridge_artifact=artifact,
            ),
        )
    )
    plan = plan_budgeted_dynamic_target(
        parent_graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=2,
            target_vocabulary_size=17,
            maximum_candidates=64,
        ),
        child_name="strict-bridge-dense-child",
        adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
        catalogs=catalogs,
    )
    selected_candidate = plan.frontier.candidates[
        plan.target_certificate.selected_candidate_rank
    ]
    assert selected_candidate.compiler_executable
    assert dict(selected_candidate.configuration)[
        TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY
    ] == artifact.artifact_sha256

    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 4),
        evidence_fingerprint=cache.cache_key,
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge_artifact=artifact,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((target_adapter,)),
    ).synthesize(
        request=plan.target_request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={
            "parent": artifact.donor_tokenizer_semantics_sha256
        },
    )
    selected = target.adapter_payload
    assert selected.tokenizer_bridge_required
    assert selected.tokenizer_bridge_artifact_sha256 == (
        artifact.artifact_sha256
    )
    assert selected.tokenizer_transport_profile_sha256 == (
        artifact.profile_fingerprint
    )
    assert target.graph.tokenizer_fingerprint == (
        artifact.target_tokenizer_semantics_sha256
    )
    assert target.model.config.vocab_size == 17
    assert target.model.get_input_embeddings().weight.shape[0] == 17
    assert all(
        int(torch.count_nonzero(parameter).item()) == 0
        for parameter in target.model.parameters()
    )

    def fresh_gate(_model):
        return FoldDecision.accept("unused bridge fresh gate")

    final_gate = _Accept()
    binding = _behavior_binding(cache.cache_key)
    expose_behavior_gate_binding(final_gate, binding)
    expose_behavior_gate_binding(fresh_gate, binding)
    base_resources = {
        "parent_cache_fingerprint": cache.cache_key,
        "parent_cache_payload": cache,
        "evidence_builder": lambda _request: None,
        "final_evaluator": final_gate,
        "model_saver": _saver,
        "model_loader": _loader,
        "fresh_reload_evaluator": fresh_gate,
        "saver_id": "strict-bridge-dense-saver-v1",
        "loader_id": "strict-bridge-dense-loader-v1",
        "ledger": ExperimentLedger(),
    }
    with pytest.raises(
        DynamicLfm2AdapterError,
        match="requires its explicit strict tokenizer bridge artifact",
    ):
        Lfm2DynamicFoldStackAdapter(
            Lfm2DynamicStackResources(
                **base_resources,
                dense_fold_config=Lfm2DenseBudgetFoldConfig(
                    run_closed_loop=False
                ),
            )
        ).build(target)
    # A bridged closed-loop target without its sealed target-token trace
    # adapter is now refused by the stack adapter while it validates the
    # target, ahead of the equivalent refusal inside the dense fold itself.
    with pytest.raises(
        DynamicLfm2AdapterError,
        match="requires its sealed target-token trace adapter",
    ):
        Lfm2DynamicFoldStackAdapter(
            Lfm2DynamicStackResources(
                **base_resources,
                dense_fold_config=Lfm2DenseBudgetFoldConfig(),
                tokenizer_bridge_artifact=artifact,
            )
        ).build(target)

    fold_config = Lfm2DenseBudgetFoldConfig(run_closed_loop=False)
    stack = Lfm2DynamicFoldStackAdapter(
        Lfm2DynamicStackResources(
            **base_resources,
            dense_fold_config=fold_config,
            tokenizer_bridge_artifact=artifact,
        )
    ).build(target)
    compilation = DynamicFoldingCompiler(
        operators=stack.operators,
        ledger=ExperimentLedger(),
    ).compile(
        parent_models={"parent": parent},
        child_model=target.model,
        child_name=plan.target_request.child_name,
        evidence_provider=stack.evidence_provider,
        evaluator=stack.evaluator,
        final_evaluator=stack.final_evaluator,
        evidence_fingerprint=cache.cache_key,
        parent_tokenizer_fingerprints={
            "parent": artifact.donor_tokenizer_semantics_sha256
        },
        child_tokenizer_fingerprint=(
            artifact.target_tokenizer_semantics_sha256
        ),
        exact_target_unique_parameters=(
            selected.exact_target_unique_parameters
        ),
        initializer=stack.initializer,
    )

    assert compilation.complete
    basis = _build_residual_basis(
        parent,
        parent_graph,
        cache,
        selected.hidden_size,
        fold_config,
    )
    indices = torch.tensor(
        artifact.transport_profile.target_to_donor_row_indices,
        dtype=torch.long,
    )
    expected_embedding = basis.fold_consumer_weight(
        parent.get_input_embeddings()
        .weight.detach()
        .to(dtype=torch.float64)
        .index_select(0, indices)
    )
    torch.testing.assert_close(
        target.model.get_input_embeddings().weight.detach(),
        expected_embedding.to(dtype=torch.float32),
    )
    assert target.model.get_input_embeddings().weight is (
        target.model.get_output_embeddings().weight
    )
    attempt = compilation.run.attempts[0]
    report = dict(attempt.transaction_report)
    vocabulary_report = report["initial_structural_fold"][
        "strict_vocabulary_operator_transport"
    ]
    assert report["tokenizer_bridge_required"] is True
    assert report["tokenizer_bridge_artifact_sha256"] == (
        artifact.artifact_sha256
    )
    assert report["initial_structural_fold"][
        "tokenizer_bridge_used"
    ] is True
    assert vocabulary_report["bridge_artifact_sha256"] == (
        artifact.artifact_sha256
    )
    assert vocabulary_report["target_row_capacity"] == 17
    assert compilation.run.optimizer_steps == 0
    assert compilation.run.backward_calls == 0


def test_dense_budget_stack_fails_closed_on_depth_composition(
    sealed_parent,
) -> None:
    parent, cache = sealed_parent
    parent_state = fingerprint_model_state(parent)
    tokenizer = _tokenizer_fingerprint()
    exact = 41 * 4 + 4 + 2 * 4 + 2 * 4 * 4 * (1 + 1) + 2 * 4
    exact += 3 * 4 * 6
    request = _authenticated_dense_request(
        parent,
        child_name="dense-depth-child",
        exact_unique_parameters=exact,
        blank_seed=2202,
        allow_depth_composition=True,
    )
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 6),
        evidence_fingerprint=cache.cache_key,
        device="cpu",
        dtype=torch.float32,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((target_adapter,)),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    blank_state = fingerprint_model_state(target.model)
    final_gate = _Accept()

    def fresh_gate(_model):
        return FoldDecision.accept("unused depth-changing reload gate")

    binding = _behavior_binding(cache.cache_key)
    expose_behavior_gate_binding(final_gate, binding)
    expose_behavior_gate_binding(fresh_gate, binding)
    adapter = Lfm2DynamicFoldStackAdapter(
        Lfm2DynamicStackResources(
            parent_cache_fingerprint=cache.cache_key,
            parent_cache_payload=cache,
            evidence_builder=lambda _request: None,
            final_evaluator=final_gate,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=fresh_gate,
            saver_id="test-depth-saver-v1",
            loader_id="test-depth-loader-v1",
            ledger=ExperimentLedger(),
        )
    )
    with pytest.raises(
        DynamicLfm2AdapterError,
        match="Ordered source-span composition is required",
    ):
        adapter.build(target)
    assert fingerprint_model_state(target.model) == blank_state
    assert fingerprint_model_state(parent) == parent_state


@pytest.mark.parametrize(
    ("parent_fixture", "compression_ratio"),
    (
        ("production_budget_parent", 2),
        ("production_budget_parent", 4),
        ("production_ten_x_parent", 10),
        ("production_hundred_x_parent", 100),
        ("production_two_hundred_fifty_x_parent", 250),
        ("production_thousand_x_parent", 1_000),
    ),
)
def test_automatic_supported_ratio_selector_folds_and_publishes_dense_target(
    request,
    tmp_path,
    parent_fixture,
    compression_ratio,
) -> None:
    parent, cache = request.getfixturevalue(parent_fixture)
    parent_state = fingerprint_model_state(parent)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 32),
        evidence_fingerprint=cache.cache_key,
        device="cpu",
        dtype=torch.float32,
    )
    final_gate = _WholeOutputAccept()

    def fresh_gate(model):
        with torch.no_grad():
            model(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        return FoldDecision.accept("automatic target fresh gate")

    binding = _behavior_binding(cache.cache_key)
    expose_behavior_gate_binding(final_gate, binding)
    expose_behavior_gate_binding(fresh_gate, binding)
    stack_adapter = Lfm2DynamicFoldStackAdapter(
        Lfm2DynamicStackResources(
            parent_cache_fingerprint=cache.cache_key,
            parent_cache_payload=cache,
            evidence_builder=lambda _request: None,
            final_evaluator=final_gate,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=fresh_gate,
            saver_id="automatic-dense-saver-v1",
            loader_id="automatic-dense-loader-v1",
            ledger=ExperimentLedger(),
            dense_fold_config=Lfm2DenseBudgetFoldConfig(
                run_closed_loop=False,
            ),
        )
    )
    target_registry = DynamicTargetAdapterRegistry((target_adapter,))
    system = AutomaticDynamicFoldingSystem(
        target_adapters=target_registry,
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
    )
    plan = system.plan_budgeted_target(
        parent_graph=parent_graph,
        budget_request=DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=compression_ratio,
            maximum_candidates=64,
        ),
        child_name=f"automatic-dense-{compression_ratio}x",
        tokenizer_fingerprint=tokenizer,
        adapter_id=target_adapter.adapter_id,
        blank_seed=2300 + compression_ratio,
    )
    rank = plan.target_certificate.selected_candidate_rank
    candidate = plan.frontier.candidates[rank]
    assert plan.target_certificate.selection_policy == (
        "architecture_discovered_component_risk_executable_v1"
    )
    receipt = plan.target_certificate.component_allocation_receipt
    assert receipt is not None
    assert plan.target_request.component_allocation_receipt == receipt
    assert candidate.compiler_executable
    assert candidate.missing_compiler_capabilities == ()
    assert len(candidate.ordered_source_spans) == len(parent_graph.layer_indices)
    assert candidate.ordered_source_spans == tuple(
        (index,) for index in range(len(parent_graph.layer_indices))
    )
    result = system.run(
        parent_models={"parent": parent},
        target_request=plan.target_request,
        evidence_fingerprint=cache.cache_key,
        destination=(
            tmp_path / f"automatic-dense-{compression_ratio}x-published"
        ),
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    assert result.complete
    assert result.target.request.component_allocation_receipt == receipt
    assert (
        dict(result.target.adapter_report)["component_allocation_receipt_sha256"]
        == receipt.fingerprint
    )
    assert (
        dict(result.target.adapter_report)[
            "component_allocation_verified_by_target_adapter"
        ]
        is True
    )
    assert result.target.blank_provenance.compiler_created_blank
    assert result.target.blank_provenance.all_parameters_zero
    assert result.target.graph.unique_parameter_count == candidate.unique_parameters
    assert result.target.graph.unique_parameter_count <= (
        parent_graph.unique_parameter_count // compression_ratio
    )
    assert candidate.compression_ratio >= compression_ratio
    extreme_parent_parameters = {
        100: 1_285_568,
        250: 3_212_864,
        1_000: 12_848_000,
    }
    if compression_ratio in extreme_parent_parameters:
        expected_parent_parameters = extreme_parent_parameters[
            compression_ratio
        ]
        assert (
            parent_graph.unique_parameter_count
            == expected_parent_parameters
        )
        assert candidate.unique_parameters == 12_848
        assert candidate.compression_ratio == pytest.approx(
            expected_parent_parameters / 12_848
        )
        hundred_x_configuration = dict(candidate.configuration)
        assert hundred_x_configuration["hidden_size"] == 32
        assert hundred_x_configuration["intermediate_size"] == 32
        assert hundred_x_configuration["num_hidden_layers"] == 2
        assert hundred_x_configuration["num_attention_heads"] == 1
        assert hundred_x_configuration["num_key_value_heads"] == 1
        assert hundred_x_configuration["head_dim"] == 8
    assert fingerprint_model_state(parent) == parent_state
    assert any(
        action.method_id == DENSE_BUDGET_MODEL_METHOD
        for action in result.published.compilation.plan.actions
    )
    assert result.published.compilation.run.optimizer_constructions == 0
    assert result.published.compilation.run.optimizer_steps == 0
    assert result.published.compilation.run.backward_calls == 0
    assert result.published.compilation.run.autograd_grad_calls == 0
    attempt = result.published.compilation.run.attempts[0]
    assert attempt.accepted
    report = dict(attempt.transaction_report)
    assert report["complete_target_parameter_overwrite"] is True
    assert report["closed_loop_enabled"] is False
    assert report["quantization_used"] is False
    assert report["pruning_used"] is False
    assert report["training_used"] is False
    assert result.published.artifact.fresh_reload_decision.accepted
    reloaded = _loader(
        tmp_path
        / f"automatic-dense-{compression_ratio}x-published"
        / "child"
    )
    assert fingerprint_model_state(reloaded) == (
        result.published.artifact.child_state_fingerprint
    )


def test_tiny_two_layer_10x_budget_fails_closed_at_discrete_catalog_boundary(
    production_budget_parent,
) -> None:
    """This tiny-budget refusal is not a general claim about 10x LFM2."""

    parent, cache = production_budget_parent
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    assert parent_graph.unique_parameter_count == 80_992
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 32),
        evidence_fingerprint=cache.cache_key,
        device="cpu",
        dtype=torch.float32,
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry(),
    )

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="No supported-Pareto dense topology is executable",
    ):
        system.plan_budgeted_target(
            parent_graph=parent_graph,
            budget_request=DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                compression_ratio=10,
                maximum_candidates=64,
            ),
            child_name="automatic-dense-10x",
            tokenizer_fingerprint=tokenizer,
            adapter_id=adapter.adapter_id,
            blank_seed=2310,
        )
