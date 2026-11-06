from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import MappingProxyType, MethodType, SimpleNamespace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    LFM2_CURRENT_COMPILER_CAPABILITIES,
    Lfm2DenseTargetTopologyCatalog,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_sparse_feature_fold_bundle import (
    AXIS_MLP_POST_SWIGLU,
    AXIS_SHORT_CONV_CONTEXT,
)
import neural_forge.hf_lfm2_t5_sparse_feature_fold_builder as t5_builder
from neural_forge.hf_lfm2_t5_sparse_feature_fold_builder import (
    Lfm2AuthenticatedT5SparseFeatureFoldBuilder,
    Lfm2T5SparseFeatureFoldBuilderError,
    Lfm2T5SparseFeatureFoldSolveConfig,
    build_lfm2_t5_evidence_split_resolution,
    build_lfm2_t5_sparse_feature_evidence_authority,
)


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")


_LAYER_TYPES = (
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
)
_SHORT_CONV_LAYERS = (0, 1, 3, 5, 7, 9, 11, 13)
_SPLIT_SCHEMA = "neural-forge-production-lfm2-sealed-cache-evidence-splits-v1"


def _model(*, hidden: int, intermediate: int, heads: int, seed: int):
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=14,
        num_attention_heads=heads,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=list(_LAYER_TYPES),
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


def _batch(index: int):
    input_ids = torch.tensor([[1 + index, 2 + index, 3 + index]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "forced_decode_ids": torch.tensor([[8 + index]], dtype=torch.long),
    }


def _deterministic_split(cache):
    inputs = tuple(cache.manifest["corpus"]["ordered_batch_input_sha256"])
    ranked = sorted(range(len(inputs)), key=lambda index: (inputs[index], index))
    splits = ["fit"] * len(inputs)
    splits[ranked[0]] = "selection"
    splits[ranked[-1]] = "validation"
    ordered = tuple(splits)
    split_sha256 = canonical_sha256(
        {
            "schema": _SPLIT_SCHEMA,
            "parent_cache_key": cache.cache_key,
            "corpus_fingerprint": cache.corpus_fingerprint,
            "ordered_batch_input_sha256": list(inputs),
            "ordered_splits": list(ordered),
            "assignment": "content_digest_rank_extrema_v1",
        }
    )
    return ordered, split_sha256


def _synthesize_target(parent, cache, source_graph, tokenizer_sha256: str):
    target_shape = _model(hidden=4, intermediate=6, heads=1, seed=5002)
    exact_unique_parameters = sum(
        int(parameter.numel()) for parameter in target_shape.parameters()
    )
    request = plan_budgeted_dynamic_target(
        source_graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            target_unique_parameters=exact_unique_parameters,
            match_policy="exact",
            maximum_candidates=256,
        ),
        child_name="t5-full-axis-child",
        tokenizer_fingerprint=tokenizer_sha256,
        adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
        blank_seed=5003,
        catalogs=DenseTargetTopologyCatalogRegistry(
            (
                Lfm2DenseTargetTopologyCatalog(
                    minimum_hidden_size=4,
                    hidden_size_multiple=4,
                    minimum_head_dim=4,
                    compiler_capabilities=LFM2_CURRENT_COMPILER_CAPABILITIES,
                ),
            )
        ),
    ).target_request
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
        parent_tokenizer_fingerprints={"parent": tokenizer_sha256},
    )


@pytest.fixture(scope="module")
def authenticated_case(tmp_path_factory):
    parent = _model(hidden=8, intermediate=12, heads=2, seed=5001)
    cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("t5-full-axis-cache"),
        parent,
        [_batch(index) for index in range(3)],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    tokenizer_sha256 = canonical_sha256(
        {"kind": "t5-full-axis-test-tokenizer", "vocabulary_size": 31}
    )
    source_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_sha256,
    )
    target = _synthesize_target(parent, cache, source_graph, tokenizer_sha256)
    ordered_splits, evidence_split_sha256 = _deterministic_split(cache)
    with cache.selective_authenticated_session() as session:
        split_resolution = build_lfm2_t5_evidence_split_resolution(
            cache=cache,
            ordered_splits=ordered_splits,
            evidence_split_source="sealed_cache_deterministic",
            evidence_split_sha256=evidence_split_sha256,
            maximum_selection_rows=1,
            maximum_fit_rows=1,
            maximum_validation_rows=1,
        )
        authority = build_lfm2_t5_sparse_feature_evidence_authority(
            cache=cache,
            selective_session=session,
            split_resolution=split_resolution,
        )
        yield SimpleNamespace(
            parent=parent,
            cache=cache,
            source_graph=source_graph,
            target=target,
            session=session,
            ordered_splits=ordered_splits,
            evidence_split_sha256=evidence_split_sha256,
            split_resolution=split_resolution,
            authority=authority,
        )


def _builder(case) -> Lfm2AuthenticatedT5SparseFeatureFoldBuilder:
    return Lfm2AuthenticatedT5SparseFeatureFoldBuilder(
        parent_model=case.parent,
        source_graph=case.source_graph,
        evidence_authority=case.authority,
        solve_config=Lfm2T5SparseFeatureFoldSolveConfig(
            active_k=1,
            channels_per_chunk=4,
        ),
    )


@pytest.mark.parametrize("tamper", ("legacy_schema", "missing_receipt"))
def test_t5_authority_independently_refuses_legacy_or_receiptless_cache_handle(
    authenticated_case,
    tamper,
) -> None:
    case = authenticated_case
    manifest = case.cache.manifest
    if tamper == "legacy_schema":
        manifest["implementation"]["schema"] = (
            "neural-forge-lfm2-circuit-implementation-v1"
        )
    else:
        manifest["implementation"].pop("runtime_semantics_receipt")
    forged_cache = replace(
        case.cache,
        _manifest=MappingProxyType(manifest),
    )

    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="canonical runtime receipt",
    ):
        build_lfm2_t5_sparse_feature_evidence_authority(
            cache=forged_cache,
            selective_session=case.session,
            split_resolution=case.split_resolution,
        )


def test_fresh_import_rejects_preimport_native_torch_and_attention_tamper() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        r"""
        import importlib
        import sys
        import tempfile

        import torch
        import transformers
        from transformers.models.lfm2 import modeling_lfm2

        from neural_forge.hf_lfm2_circuit_trace_cache import (
            FULL_PREFIX_CANONICAL_TRACE_MODE,
            Lfm2CircuitTraceCaptureConfig,
            build_lfm2_circuit_trace_cache,
        )

        builder_name = "neural_forge.hf_lfm2_t5_sparse_feature_fold_builder"
        assert builder_name not in sys.modules
        layer_types = (
            "conv", "conv", "full_attention", "conv", "full_attention",
            "conv", "full_attention", "conv", "full_attention", "conv",
            "full_attention", "conv", "full_attention", "conv",
        )
        config = transformers.Lfm2Config(
            vocab_size=31,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=14,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            layer_types=list(layer_types),
            conv_L_cache=3,
            conv_bias=False,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
        config._attn_implementation = "eager"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5001)
            parent = transformers.Lfm2ForCausalLM(config)
        parent.eval()

        def batch(index):
            input_ids = torch.tensor(
                [[1 + index, 2 + index, 3 + index]],
                dtype=torch.long,
            )
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "forced_decode_ids": torch.tensor(
                    [[8 + index]],
                    dtype=torch.long,
                ),
            }

        with tempfile.TemporaryDirectory() as root:
            build_lfm2_circuit_trace_cache(
                root,
                parent,
                [batch(index) for index in range(3)],
                config=Lfm2CircuitTraceCaptureConfig(
                    sample_last_tokens=2,
                    max_decode_steps=1,
                    max_capture_bytes=64 * 1024 * 1024,
                    decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
                ),
                input_device="cpu",
            )
        assert builder_name not in sys.modules

        def expect_rejected(label, mutate, restore):
            mutate()
            assert builder_name not in sys.modules
            try:
                importlib.import_module(builder_name)
            except Exception as exc:
                message = str(exc).lower()
                assert any(
                    marker in message
                    for marker in (
                        "independent",
                        "canonical",
                        "factory-owned",
                        "source",
                    )
                ), message
                print(f"REJECTED {label}: {type(exc).__qualname__}")
            else:
                raise AssertionError(f"pre-import tamper was accepted: {label}")
            finally:
                restore()
            assert builder_name not in sys.modules

        from neural_forge import hf_lfm2_dynamic_blank_materializer as dynamic_blank

        dynamic_mlp_class = dynamic_blank._dynamic_mlp_class()

        def poisoned_dynamic_call(_self, hidden_states, *_args, **_kwargs):
            return torch.zeros_like(hidden_states)

        expect_rejected(
            "dynamic_mlp_call_path",
            lambda: setattr(dynamic_mlp_class, "__call__", poisoned_dynamic_call),
            lambda: delattr(dynamic_mlp_class, "__call__"),
        )

        native_mlp_forward = modeling_lfm2.Lfm2MLP.forward

        def poisoned_mlp_forward(_self, hidden_states):
            return torch.zeros_like(hidden_states)

        expect_rejected(
            "native_mlp_forward",
            lambda: setattr(modeling_lfm2.Lfm2MLP, "forward", poisoned_mlp_forward),
            lambda: setattr(modeling_lfm2.Lfm2MLP, "forward", native_mlp_forward),
        )

        native_rotate_half = modeling_lfm2.rotate_half

        def poisoned_rotate_half(value):
            return torch.zeros_like(value)

        expect_rejected(
            "native_rotate_half",
            lambda: setattr(modeling_lfm2, "rotate_half", poisoned_rotate_half),
            lambda: setattr(modeling_lfm2, "rotate_half", native_rotate_half),
        )

        native_repeat_kv = modeling_lfm2.repeat_kv

        def poisoned_repeat_kv(value, *_args, **_kwargs):
            return torch.zeros_like(value)

        expect_rejected(
            "native_repeat_kv",
            lambda: setattr(modeling_lfm2, "repeat_kv", poisoned_repeat_kv),
            lambda: setattr(modeling_lfm2, "repeat_kv", native_repeat_kv),
        )

        native_call_impl = torch.nn.Module._call_impl

        def poisoned_call_impl(_self, *_args, **_kwargs):
            return torch.zeros(1)

        expect_rejected(
            "torch_module_call_impl",
            lambda: setattr(torch.nn.Module, "_call_impl", poisoned_call_impl),
            lambda: setattr(torch.nn.Module, "_call_impl", native_call_impl),
        )

        native_mlp_class = modeling_lfm2.Lfm2MLP

        class ForeignMLP(native_mlp_class):
            def __call__(self, hidden_states, *_args, **_kwargs):
                return torch.zeros_like(hidden_states)

        expect_rejected(
            "native_mlp_class_shell",
            lambda: setattr(modeling_lfm2, "Lfm2MLP", ForeignMLP),
            lambda: setattr(modeling_lfm2, "Lfm2MLP", native_mlp_class),
        )

        native_linear_class = torch.nn.Linear

        class ForeignLinear(native_linear_class):
            def __call__(self, value, *_args, **_kwargs):
                return torch.zeros_like(value)

        expect_rejected(
            "torch_linear_class_shell",
            lambda: setattr(torch.nn, "Linear", ForeignLinear),
            lambda: setattr(torch.nn, "Linear", native_linear_class),
        )

        native_attention_registry = modeling_lfm2.ALL_ATTENTION_FUNCTIONS

        class ForeignAttentionRegistry:
            def get_interface(self, name, fallback):
                return native_attention_registry.get_interface(name, fallback)

        expect_rejected(
            "foreign_attention_registry",
            lambda: setattr(
                modeling_lfm2,
                "ALL_ATTENTION_FUNCTIONS",
                ForeignAttentionRegistry(),
            ),
            lambda: setattr(
                modeling_lfm2,
                "ALL_ATTENTION_FUNCTIONS",
                native_attention_registry,
            ),
        )

        def poisoned_attention(*_args, **_kwargs):
            return torch.zeros(1)

        expect_rejected(
            "attention_eager_registry",
            lambda: modeling_lfm2.ALL_ATTENTION_FUNCTIONS.register(
                "eager",
                poisoned_attention,
            ),
            lambda: None,
        )
        """
    )
    environment = dict(os.environ)
    path_entries = [str(repo_root / "src"), str(repo_root)]
    existing = environment.get("PYTHONPATH")
    if existing:
        path_entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(path_entries)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "REJECTED dynamic_mlp_call_path" in completed.stdout
    assert "REJECTED native_mlp_forward" in completed.stdout
    assert "REJECTED native_rotate_half" in completed.stdout
    assert "REJECTED native_repeat_kv" in completed.stdout
    assert "REJECTED torch_module_call_impl" in completed.stdout
    assert "REJECTED native_mlp_class_shell" in completed.stdout
    assert "REJECTED torch_linear_class_shell" in completed.stdout
    assert "REJECTED foreign_attention_registry" in completed.stdout
    assert "REJECTED attention_eager_registry" in completed.stdout


def test_real_cache_builds_deterministic_bounded_full_axis_bundle(
    authenticated_case, monkeypatch
) -> None:
    case = authenticated_case
    parent_state = fingerprint_model_state(case.parent)
    target_state = fingerprint_model_state(case.target.model)
    parent_forward_calls = 0

    def forbidden_parent_forward(*_args, **_kwargs):
        nonlocal parent_forward_calls
        parent_forward_calls += 1
        raise AssertionError("T5 construction executed the parent")

    monkeypatch.setattr(case.parent, "forward", forbidden_parent_forward)
    payloads = []
    original_load = t5_builder.load_lfm2_circuit_trace_selected_rows

    def observed_load(*args, **kwargs):
        payload = original_load(*args, **kwargs)
        payloads.append(payload)
        return payload

    monkeypatch.setattr(
        t5_builder,
        "load_lfm2_circuit_trace_selected_rows",
        observed_load,
    )
    builder = _builder(case)
    first = builder(case.target.adapter_payload, case.target)
    second = builder(case.target.adapter_payload, case.target)

    expected_axes = tuple(
        (layer_index, axis_kind)
        for layer_index in range(14)
        for axis_kind in (
            AXIS_MLP_POST_SWIGLU,
            *((AXIS_SHORT_CONV_CONTEXT,) if layer_index in _SHORT_CONV_LAYERS else ()),
        )
    )
    assert first.bundle_sha256 == second.bundle_sha256
    assert tuple((axis.layer_index, axis.axis_kind) for axis in first.axes) == (
        expected_axes
    )
    assert len(first.axes) == 22
    assert all(axis.coefficient_set == "write" for axis in first.axes)
    assert case.authority.selection_indices
    assert len(case.authority.fit_indices) == 1
    assert len(case.authority.validation_indices) == 1
    selection_batches = {
        index for index, split in enumerate(case.ordered_splits) if split == "selection"
    }
    assert payloads
    assert all(payload.batch_index not in selection_batches for payload in payloads)
    assert all(payload.maximum_materialized_source_rows == 1 for payload in payloads)
    assert all(
        payload.logical_bytes < payload.source_logical_bytes for payload in payloads
    )
    assert all(payload.released for payload in payloads)
    for payload in payloads:
        with pytest.raises(Exception, match="released"):
            _ = payload.tensors
    artifacts = first.provenance["source_artifact_sha256s"]
    assert artifacts["component_evidence_split_resolution"] == (
        case.split_resolution.binding_sha256
    )
    assert artifacts["parent_trace_path_plan"] == case.authority.path_plan_sha256
    assert artifacts["target_blank_state"] == case.target.blank_state_fingerprint
    assert artifacts["target_topology"] == case.target.topology_fingerprint
    assert artifacts["parent_trace_layer_topology"] == (
        case.cache.layer_topology_fingerprint
    )
    assert artifacts["target_live_layer_topology"]
    assert artifacts["source_runtime_semantics"]
    assert artifacts["target_runtime_semantics"]
    assert artifacts["canonical_runtime_receipt"] == (
        case.authority.canonical_runtime_receipt_sha256
    )
    assert fingerprint_model_state(case.parent) == parent_state
    assert fingerprint_model_state(case.target.model) == target_state
    assert parent_forward_calls == 0


def test_solver_cannot_mutate_owned_fit_or_heldout_rows(
    authenticated_case,
    monkeypatch,
) -> None:
    case = authenticated_case
    parent_state = fingerprint_model_state(case.parent)
    target_state = fingerprint_model_state(case.target.model)
    original_fit = t5_builder.fit_body_component_routes

    def mutating_fit(*args, **kwargs):
        kwargs["component_rows"].add_(1.0)
        kwargs["heldout_component_rows"].sub_(1.0)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(t5_builder, "fit_body_component_routes", mutating_fit)
    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="solver_(fit|heldout)_rows changed",
    ):
        _builder(case)(case.target.adapter_payload, case.target)
    assert fingerprint_model_state(case.parent) == parent_state
    assert fingerprint_model_state(case.target.model) == target_state


def test_witness_consumer_cannot_mutate_owned_fit_or_heldout_rows(
    authenticated_case,
    monkeypatch,
) -> None:
    case = authenticated_case
    original_witness = t5_builder.build_t5_body_route_source_witness

    def mutating_witness(*args, **kwargs):
        witness = original_witness(*args, **kwargs)
        kwargs["component_rows"].add_(1.0)
        kwargs["heldout_component_rows"].sub_(1.0)
        return witness

    monkeypatch.setattr(
        t5_builder,
        "build_t5_body_route_source_witness",
        mutating_witness,
    )
    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="witness_(fit|heldout)_rows changed",
    ):
        _builder(case)(case.target.adapter_payload, case.target)


def test_admission_consumer_cannot_mutate_witness_row_clones(
    authenticated_case,
    monkeypatch,
) -> None:
    case = authenticated_case
    original_witness = t5_builder.build_t5_body_route_source_witness
    original_admit = t5_builder.admit_lfm2_sparse_feature_axis
    captured = []

    def observed_witness(*args, **kwargs):
        captured.append((kwargs["component_rows"], kwargs["heldout_component_rows"]))
        return original_witness(*args, **kwargs)

    def mutating_admit(*args, **kwargs):
        fit_rows, heldout_rows = captured[-1]
        fit_rows.add_(1.0)
        heldout_rows.sub_(1.0)
        return original_admit(*args, **kwargs)

    monkeypatch.setattr(
        t5_builder,
        "build_t5_body_route_source_witness",
        observed_witness,
    )
    monkeypatch.setattr(
        t5_builder,
        "admit_lfm2_sparse_feature_axis",
        mutating_admit,
    )
    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="witness_(fit|heldout)_rows changed",
    ):
        _builder(case)(case.target.adapter_payload, case.target)


def test_cross_axis_writer_aba_is_rejected_even_when_state_is_restored(
    authenticated_case,
    monkeypatch,
) -> None:
    case = authenticated_case
    later_writer = case.parent.model.layers[13].feed_forward.w2.weight
    original_writer = later_writer.detach().clone()
    parent_state = fingerprint_model_state(case.parent)
    original_fit = t5_builder.fit_body_component_routes
    calls = 0

    def aba_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        with torch.no_grad():
            if calls == 1:
                later_writer.add_(1.0)
            elif calls == 2:
                later_writer.copy_(original_writer)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(t5_builder, "fit_body_component_routes", aba_fit)
    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="changed around the all-axis solve",
    ):
        _builder(case)(case.target.adapter_payload, case.target)
    assert calls == 22
    assert fingerprint_model_state(case.parent) == parent_state


def test_selected_payload_is_released_when_public_release_rejects(
    authenticated_case,
    monkeypatch,
) -> None:
    case = authenticated_case
    payloads = []
    original_load = t5_builder.load_lfm2_circuit_trace_selected_rows

    def observed_load(*args, **kwargs):
        payload = original_load(*args, **kwargs)
        payloads.append(payload)
        return payload

    def rejected_release(*_args, **_kwargs):
        raise RuntimeError("session identity changed before public release")

    monkeypatch.setattr(
        t5_builder,
        "load_lfm2_circuit_trace_selected_rows",
        observed_load,
    )
    monkeypatch.setattr(
        t5_builder,
        "release_lfm2_circuit_trace_selected_rows",
        rejected_release,
    )
    with pytest.raises(RuntimeError, match="session identity changed"):
        _builder(case)(case.target.adapter_payload, case.target)
    assert payloads and all(payload.released for payload in payloads)
    for payload in payloads:
        with pytest.raises(Exception, match="released"):
            _ = payload.tensors


def test_builder_binding_rejects_post_construction_config_mutation(
    authenticated_case,
) -> None:
    builder = _builder(authenticated_case)
    object.__setattr__(builder.solve_config, "active_k", 2)
    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="binding changed",
    ):
        builder.verify()


@pytest.mark.parametrize(
    ("side", "module_path", "method_name"),
    (
        ("source", (0, "feed_forward"), "forward"),
        ("source", (0, "ffn_norm"), "forward"),
        ("target", (0, "feed_forward"), "forward"),
        ("target", (0, "conv"), "slow_forward"),
    ),
)
def test_internal_runtime_instance_callable_overrides_fail_closed(
    authenticated_case,
    side,
    module_path,
    method_name,
) -> None:
    case = authenticated_case
    model = case.parent if side == "source" else case.target.model
    layer_index, attribute = module_path
    module = getattr(model.model.layers[layer_index], attribute)
    parent_state = fingerprint_model_state(case.parent)
    target_state = fingerprint_model_state(case.target.model)

    def replacement(_self, value, *_args, **_kwargs):
        return torch.zeros_like(value)

    assert method_name not in vars(module)
    if side == "source":

        def callback():
            return _builder(case)
    else:
        builder = _builder(case)

        def callback():
            return builder(case.target.adapter_payload, case.target)

    setattr(module, method_name, MethodType(replacement, module))
    try:
        with pytest.raises(
            Lfm2T5SparseFeatureFoldBuilderError,
            match="instance callable override",
        ):
            callback()
    finally:
        delattr(module, method_name)
    assert fingerprint_model_state(case.parent) == parent_state
    assert fingerprint_model_state(case.target.model) == target_state


def test_runtime_class_method_and_internal_hook_tamper_fail_closed(
    authenticated_case,
    monkeypatch,
) -> None:
    case = authenticated_case
    conv_class = type(case.parent.model.layers[0].conv)

    def replacement(_self, value, *_args, **_kwargs):
        return torch.zeros_like(value)

    with monkeypatch.context() as scoped:
        scoped.setattr(conv_class, "slow_forward", replacement)
        with pytest.raises(
            Lfm2T5SparseFeatureFoldBuilderError,
            match=(
                "runtime was replaced|method .* was replaced|canonical source receipt"
            ),
        ):
            _builder(case)

    builder = _builder(case)
    target_norm = case.target.model.model.layers[0].ffn_norm
    handle = target_norm.register_forward_hook(
        lambda _module, _inputs, output: torch.zeros_like(output)
    )
    try:
        with pytest.raises(
            Lfm2T5SparseFeatureFoldBuilderError,
            match="forward hooks",
        ):
            builder(case.target.adapter_payload, case.target)
    finally:
        handle.remove()


@pytest.mark.parametrize("side", ("source", "target"))
def test_same_state_foreign_feed_forward_shell_fails_topology(
    authenticated_case,
    monkeypatch,
    side,
) -> None:
    case = authenticated_case
    model = case.parent if side == "source" else case.target.model
    layer = model.model.layers[0]
    original = layer.feed_forward

    class ForeignFeedForward(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.w1 = value.w1
            self.w3 = value.w3
            self.w2 = value.w2

        def forward(self, value):
            return torch.zeros_like(value)

    parent_state = fingerprint_model_state(case.parent)
    target_state = fingerprint_model_state(case.target.model)
    builder = _builder(case) if side == "target" else None
    monkeypatch.setattr(layer, "feed_forward", ForeignFeedForward(original))
    assert fingerprint_model_state(case.parent) == parent_state
    assert fingerprint_model_state(case.target.model) == target_state
    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="canonical topology inspection|authenticated graph or layer topology",
    ):
        if side == "source":
            _builder(case)
        else:
            assert builder is not None
            builder(case.target.adapter_payload, case.target)


def test_split_and_authority_fail_closed_on_noncanonical_authority(
    authenticated_case,
) -> None:
    case = authenticated_case
    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="only admits the cache-derived deterministic split",
    ):
        build_lfm2_t5_evidence_split_resolution(
            cache=case.cache,
            ordered_splits=case.ordered_splits,
            evidence_split_source="supplied_matching_calibration",
            evidence_split_sha256=case.evidence_split_sha256,
        )
    relabeled = tuple(reversed(case.ordered_splits))
    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="preimage",
    ):
        build_lfm2_t5_evidence_split_resolution(
            cache=case.cache,
            ordered_splits=relabeled,
            evidence_split_source="sealed_cache_deterministic",
            evidence_split_sha256=case.evidence_split_sha256,
        )
    with pytest.raises(
        Lfm2T5SparseFeatureFoldBuilderError,
        match="base-only sealed-cache authority",
    ):
        build_lfm2_t5_evidence_split_resolution(
            cache=case.cache,
            ordered_splits=case.ordered_splits,
            evidence_split_source="sealed_cache_deterministic",
            evidence_split_sha256=case.evidence_split_sha256,
            on_policy_evidence_identity_sha256="a" * 64,
            on_policy_extension_cache_keys=("b" * 64,),
        )
