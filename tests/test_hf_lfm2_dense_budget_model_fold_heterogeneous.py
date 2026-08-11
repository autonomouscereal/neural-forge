from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_closed_loop_fold import (  # noqa: E402
    ClosedLoopCompilerConfig,
    closed_loop_compiler_restart_policy,
)
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_state,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
from neural_forge.dynamic_lfm2_adapter import (  # noqa: E402
    Lfm2DenseBudgetedSelectedTarget,
)
from neural_forge.hf_feature_fold import build_feature_fold_map  # noqa: E402
from neural_forge.hf_lfm2_dense_budget_model_fold import (  # noqa: E402
    LFM2_SPARSE_FEATURE_FOLD_BUNDLE_RESOURCE_NAME,
    Lfm2DenseBudgetFoldConfig,
    Lfm2DenseBudgetFoldError,
    Lfm2DenseBudgetModelOperator,
    _Lfm2DenseFoldProgress,
    _factorize_short_conv_dense_writer_action,
    _fold_dense_blank_from_parent,
    _seal_short_conv_output_action_evidence,
)
from neural_forge.hf_lfm2_sparse_feature_fold_bundle import (  # noqa: E402
    AXIS_MLP_POST_SWIGLU,
    AXIS_SHORT_CONV_CONTEXT,
    Lfm2SparseFeatureFoldBundle,
    admit_lfm2_sparse_feature_axis,
    build_lfm2_sparse_feature_fold_bundle,
)
from neural_forge import hf_lfm2_dense_budget_model_fold as dense_fold_module  # noqa: E402
from neural_forge.hf_lfm2_dynamic_blank_materializer import (  # noqa: E402
    extract_lfm2_dynamic_physical_layout_from_config,
    materialize_lfm2_dynamic_blank_child,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCache,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (  # noqa: E402
    Lfm2HeterogeneousLayerPhysicalLayout,
    build_lfm2_heterogeneous_physical_layout,
)
from neural_forge.matrix_technique_body_routes import (  # noqa: E402
    build_t5_body_route_source_witness,
    fit_body_component_routes,
)


def _config(
    *,
    hidden: int,
    intermediate: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    layer_types: tuple[str, ...],
):
    config = transformers.Lfm2Config(
        vocab_size=23,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=len(layer_types),
        num_attention_heads=query_heads,
        num_key_value_heads=key_value_heads,
        head_dim=head_dim,
        layer_types=list(layer_types),
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "sdpa"
    return config


def _model(config, *, seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _zero(model) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()


def _residual_map(parent, target_width: int):
    return build_feature_fold_map(
        parent.get_input_embeddings().weight.detach(),
        target_width,
        source_width=int(parent.config.hidden_size),
        chunk_size=16,
        sketch_size=8,
        balance_strength=0.25,
        allow_sign_flip=False,
    )


_SPARSE_CACHE_KEY = "b" * 64


def _sparse_axis_paths(layer_index: int, axis_kind: str) -> tuple[str, dict[str, str]]:
    if axis_kind == AXIS_MLP_POST_SWIGLU:
        site = f"model.layers.{layer_index}.feed_forward"
        return site, {
            "w1": f"{site}.w1.weight",
            "w3": f"{site}.w3.weight",
            "w2": f"{site}.w2.weight",
        }
    site = f"model.layers.{layer_index}.conv"
    packed = f"{site}.in_proj.weight"
    return site, {
        "in_proj_b": packed,
        "in_proj_c": packed,
        "in_proj_x": packed,
        "fir_kernel": f"{site}.conv.weight",
        "out_proj": f"{site}.out_proj.weight",
    }


def _sparse_short_slices(width: int) -> dict[str, tuple[int, int]]:
    return {
        "in_proj_b": (0, width),
        "in_proj_c": (width, 2 * width),
        "in_proj_x": (2 * width, 3 * width),
    }


def _sparse_source_artifacts(witness) -> dict[str, str]:
    return {
        "parent_state": witness.parent_state_fingerprint,
        "component_evidence_cache": witness.evidence_cache_sha256,
        "component_evidence_selection": witness.evidence_selection_sha256,
        "body_route_record": witness.body_route_record_sha256,
        "body_route_source_witness": witness.witness_sha256,
    }


def _sparse_bundle_fixture(
    parent,
    child,
    *,
    parent_state_fingerprint: str | None = None,
    evidence_cache_sha256: str = _SPARSE_CACHE_KEY,
    route_seed_offset: int = 0,
):
    source_graph = inspect_model_circuits(parent, model_name="sparse-parent")
    target_graph = inspect_model_circuits(child, model_name="sparse-child")
    parent_state = (
        fingerprint_model_state(parent)
        if parent_state_fingerprint is None
        else parent_state_fingerprint
    )
    axes = []
    for layer_index, (source_layer, target_layer) in enumerate(
        zip(parent.model.layers, child.model.layers, strict=True)
    ):
        specifications = [
            (
                AXIS_MLP_POST_SWIGLU,
                "C13",
                source_layer.feed_forward.w2.weight,
                int(target_layer.feed_forward.w1.weight.shape[0]),
                "w2",
                "mlp.post_swiglu",
            )
        ]
        if not source_layer.is_attention_layer:
            specifications.append(
                (
                    AXIS_SHORT_CONV_CONTEXT,
                    "C05",
                    source_layer.conv.out_proj.weight,
                    int(target_layer.conv.in_proj.weight.shape[0]) // 3,
                    "out_proj",
                    "short_conv.context",
                )
            )
        for (
            axis_kind,
            component,
            live_writer,
            target_width,
            writer_role,
            evidence_endpoint,
        ) in specifications:
            source_width = int(live_writer.shape[1])
            generator = torch.Generator().manual_seed(
                51_000 + route_seed_offset + layer_index * 10 + len(axes)
            )
            rows = torch.randn(
                (max(48, source_width * 3), source_width),
                generator=generator,
                dtype=torch.float64,
            )
            normalized_writer = live_writer.detach().to(
                device="cpu", dtype=torch.float64
            )
            fold = fit_body_component_routes(
                parent_weight=normalized_writer,
                component_rows=rows,
                component=component,
                level="X4",
                site=f"L{layer_index:02d}",
                kept_width=target_width,
                active_k=min(2, target_width),
                channel_selection="explicit",
                kept_channel_ids=tuple(range(target_width)),
                install_coefficients="write",
            )
            site, paths = _sparse_axis_paths(layer_index, axis_kind)
            selection_sha256 = canonical_sha256(
                {
                    "axis_kind": axis_kind,
                    "layer_index": layer_index,
                    "kept_channel_ids": list(range(target_width)),
                    "route_seed_offset": route_seed_offset,
                }
            )
            witness = build_t5_body_route_source_witness(
                fold,
                parent_weight=normalized_writer,
                component_rows=rows,
                parameter_path=paths[writer_role],
                graph_fingerprint=source_graph.fingerprint,
                parent_state_fingerprint=parent_state,
                component_id=fold.component_id,
                layer_index=layer_index,
                site=f"L{layer_index:02d}",
                role="writer",
                evidence_cache_sha256=evidence_cache_sha256,
                evidence_selection_sha256=selection_sha256,
                evidence_endpoint=evidence_endpoint,
                coefficient_set="write",
            )
            kwargs = {
                "source_graph": source_graph,
                "target_graph": target_graph,
                "source_graph_sha256": source_graph.fingerprint,
                "target_graph_sha256": target_graph.fingerprint,
                "layer_index": layer_index,
                "site": site,
                "axis_kind": axis_kind,
                "source_parameter_paths": paths,
                "target_parameter_paths": paths,
                "body_route_fold": fold,
                "source_witness": witness,
                "coefficient_set": "write",
                "source_artifact_sha256s": _sparse_source_artifacts(witness),
            }
            if axis_kind == AXIS_SHORT_CONV_CONTEXT:
                kwargs.update(
                    source_parameter_row_slices=_sparse_short_slices(source_width),
                    target_parameter_row_slices=_sparse_short_slices(target_width),
                )
            axes.append(admit_lfm2_sparse_feature_axis(**kwargs))
    bundle = build_lfm2_sparse_feature_fold_bundle(
        source_graph=source_graph,
        target_graph=target_graph,
        source_graph_sha256=source_graph.fingerprint,
        target_graph_sha256=target_graph.fingerprint,
        axes=axes,
    )
    return bundle, source_graph, target_graph


def _blank_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_state_exact(model, expected: dict[str, torch.Tensor]) -> None:
    assert tuple(model.state_dict()) == tuple(expected)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def _unsafe_bundle_axes(bundle, axes) -> Lfm2SparseFeatureFoldBundle:
    """Build an intentionally unverified object for materializer tamper tests."""

    result = object.__new__(Lfm2SparseFeatureFoldBundle)
    for name, value in (
        ("source_graph", bundle.source_graph),
        ("target_graph", bundle.target_graph),
        ("source_graph_sha256", bundle.source_graph_sha256),
        ("target_graph_sha256", bundle.target_graph_sha256),
        ("axes", tuple(axes)),
        ("attention_sites", bundle.attention_sites),
        ("provenance", bundle.provenance),
        ("report", bundle.report),
        ("bundle_sha256", bundle.bundle_sha256),
    ):
        object.__setattr__(result, name, value)
    return result


def _heterogeneous_child(*, factorized_writer: bool = False):
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    layers = (
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=0,
            mlp_intermediate_size=7,
            internal_width=10,
            writer_rank=3 if factorized_writer else None,
            selection_evidence_sha256="0" * 64,
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=1,
            mlp_intermediate_size=8,
            query_heads=2,
            key_value_heads=1,
            head_dim=4,
            selection_evidence_sha256="1" * 64,
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=2,
            mlp_intermediate_size=9,
            query_heads=4,
            key_value_heads=2,
            head_dim=2,
            selection_evidence_sha256="2" * 64,
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=3,
            mlp_intermediate_size=10,
            internal_width=6,
            selection_evidence_sha256="3" * 64,
        ),
    )
    descriptor = build_lfm2_heterogeneous_physical_layout(
        hidden_size=8,
        vocab_size=23,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        layers=layers,
        source_topology_sha256="a" * 64,
        allocation_evidence_sha256="b" * 64,
        token_bridge_evidence_sha256="c" * 64,
    )
    child = _model(
        _config(
            hidden=8,
            intermediate=8,
            query_heads=2,
            key_value_heads=1,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4102,
    )
    materialize_lfm2_dynamic_blank_child(
        child,
        child.config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    _zero(child)
    return child


def _sparse_parent(*, seed: int):
    return _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=("conv", "full_attention", "full_attention", "conv"),
        ),
        seed=seed,
    )


def test_legacy_dense_import_does_not_register_t5_body_routes() -> None:
    script = """
import json
from neural_forge.matrix_technique_registry import registered_pairs
before = registered_pairs()
import neural_forge.hf_lfm2_dense_budget_model_fold  # noqa: F401
after = registered_pairs()
print(json.dumps({"before": before, "after": after}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["after"] == payload["before"]
    assert all(pair[0] != "T5" for pair in payload["after"])


def test_moving_gauge_resume_preserves_policy_and_marks_terminal_override(
    tmp_path: Path,
) -> None:
    capsule_root = tmp_path / "capsules"
    configured = ClosedLoopCompilerConfig(
        allow_terminal_only_restart_resume=True,
    )
    config = Lfm2DenseBudgetFoldConfig(
        body_reclosure_mode="moving_gauge_v3",
        closed_loop_compiler_config=configured,
        restart_capsule_directory=capsule_root,
        resume_from_restart_capsule=capsule_root,
    )

    compiler_config = config._closed_loop_compiler_config_for_execution()

    assert config.body_reclosure_mode == "moving_gauge_v3"
    assert compiler_config.fold_body_reclosure is True
    assert compiler_config.fold_output_interface is True
    assert compiler_config.allow_terminal_only_restart_resume is True
    assert closed_loop_compiler_restart_policy(
        compiler_config,
        (0, 1),
    ) == closed_loop_compiler_restart_policy(
        replace(
            compiler_config,
            allow_terminal_only_restart_resume=False,
        ),
        (0, 1),
    )


def test_moving_gauge_exact_resume_preserves_reclosure_policy(
    tmp_path: Path,
) -> None:
    capsule_root = tmp_path / "capsules"
    config = Lfm2DenseBudgetFoldConfig(
        body_reclosure_mode="moving_gauge_v3",
        restart_capsule_directory=capsule_root,
        resume_from_restart_capsule=capsule_root,
    )

    compiler_config = config._closed_loop_compiler_config_for_execution()

    assert compiler_config.fold_body_reclosure is True
    assert compiler_config.allow_terminal_only_restart_resume is False


def test_nonresume_moving_gauge_compiler_policy_is_unchanged() -> None:
    configured = ClosedLoopCompilerConfig(
        validation_fraction=0.4,
        split_seed="nonresume-policy",
        selected_paths=("text",),
        selected_layers=(1,),
        fold_output_interface=False,
        merge_adjacent_layer_captures=True,
    )
    config = Lfm2DenseBudgetFoldConfig(
        body_reclosure_mode="moving_gauge_v3",
        closed_loop_compiler_config=configured,
    )

    compiler_config = config._closed_loop_compiler_config_for_execution()

    assert compiler_config == replace(
        configured,
        fold_body_reclosure=True,
    )
    assert compiler_config.allow_terminal_only_restart_resume is False


def test_fold_uses_each_materialized_layer_physical_geometry() -> None:
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4101,
    )
    child = _heterogeneous_child()
    parent_before = fingerprint_model_state(parent)

    report = _fold_dense_blank_from_parent(
        parent,
        child,
        _residual_map(parent, 8),
    )

    assert report["complete_target_parameter_overwrite"] is True
    assert report["training_used"] is False
    assert report["parent_forward_calls"] == 0
    assert report["sparse_feature_fold_bundle_used"] is False
    assert report["sparse_feature_fold_bundle"] is None
    assert report["dynamic_body_feature_map_solve_count"] == 6
    assert fingerprint_model_state(parent) == parent_before
    assert [layer["swiglu"]["target_neurons"] for layer in report["layers"]] == [
        7,
        8,
        9,
        10,
    ]
    first_conv = report["layers"][0]["operator"]
    last_conv = report["layers"][3]["operator"]
    assert first_conv["source_internal_channels"] == 16
    assert first_conv["target_internal_channels"] == 10
    assert first_conv["internal_feature_fold"]["full_source_coverage"] is True
    assert last_conv["target_internal_channels"] == 6
    first_attention = report["layers"][1]["operator"]
    second_attention = report["layers"][2]["operator"]
    assert first_attention["target_physical_geometry"] == {
        "query_heads": 2,
        "key_value_heads": 1,
        "head_dim": 4,
    }
    assert second_attention["target_physical_geometry"] == {
        "query_heads": 4,
        "key_value_heads": 2,
        "head_dim": 2,
    }
    assert first_attention["query"]["target_heads"] == 2
    assert first_attention["query"]["target_head_dim"] == 4
    assert second_attention["query"]["target_heads"] == 4
    assert second_attention["query"]["target_head_dim"] == 2
    assert tuple(child.model.layers[0].conv.in_proj.weight.shape) == (30, 8)
    assert tuple(child.model.layers[3].conv.out_proj.weight.shape) == (8, 6)
    assert tuple(child.model.layers[2].self_attn.q_proj.weight.shape) == (8, 8)
    assert all(
        int(torch.count_nonzero(parameter).item()) > 0
        for parameter in child.parameters()
    )
    with torch.no_grad():
        logits = child(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            use_cache=False,
        ).logits
    assert tuple(logits.shape) == (1, 3, 23)
    assert bool(torch.isfinite(logits).all())


def test_authenticated_sparse_bundle_materializes_exact_six_axis_formulas(
    tmp_path: Path,
) -> None:
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4111,
    )
    child = _heterogeneous_child()
    mapping = _residual_map(parent, 8)
    bundle, source_graph, target_graph = _sparse_bundle_fixture(parent, child)
    axes = {(axis.layer_index, axis.axis_kind): axis for axis in bundle.axes}

    report = _fold_dense_blank_from_parent(
        parent,
        child,
        mapping,
        sparse_feature_fold_bundle=bundle,
        source_graph=source_graph,
        target_graph=target_graph,
        trace_cache_key=_SPARSE_CACHE_KEY,
    )

    assert len(bundle.axes) == 6
    assert report["dynamic_body_feature_map_solve_count"] == 0
    sparse_report = report["sparse_feature_fold_bundle"]
    assert sparse_report["resource_name"] == (
        LFM2_SPARSE_FEATURE_FOLD_BUNDLE_RESOURCE_NAME
    )
    assert sparse_report["bundle_sha256"] == bundle.bundle_sha256
    assert sparse_report["admitted_axis_sha256s"] == (
        sparse_report["consumed_axis_sha256s"]
    )
    assert sparse_report["admitted_axis_count"] == 6
    assert sparse_report["consumed_axis_count"] == 6
    assert sparse_report["all_admissions_consumed"] is True
    assert sparse_report["live_source_values_reauthenticated"] is True
    assert sparse_report["runtime_parameter_count"] == 0
    assert sparse_report["runtime_buffer_count"] == 0
    assert sparse_report["runtime_sidecar_count"] == 0

    for layer_index, (source_layer, target_layer) in enumerate(
        zip(parent.model.layers, child.model.layers, strict=True)
    ):
        feature_map = axes[
            (layer_index, AXIS_MLP_POST_SWIGLU)
        ].feature_map
        expected_w1 = feature_map.fold_producer_weight(
            mapping.fold_consumer_weight(
                source_layer.feed_forward.w1.weight.detach().to(torch.float64)
            )
        )
        expected_w3 = feature_map.fold_producer_weight(
            mapping.fold_consumer_weight(
                source_layer.feed_forward.w3.weight.detach().to(torch.float64)
            )
        )
        expected_w2 = mapping.fold_producer_weight(
            feature_map.fold_consumer_weight(
                source_layer.feed_forward.w2.weight.detach().to(torch.float64)
            )
        )
        for actual, expected in (
            (target_layer.feed_forward.w1.weight, expected_w1),
            (target_layer.feed_forward.w3.weight, expected_w3),
            (target_layer.feed_forward.w2.weight, expected_w2),
        ):
            torch.testing.assert_close(
                actual,
                expected.to(dtype=actual.dtype),
                rtol=0,
                atol=0,
            )
        if source_layer.is_attention_layer:
            continue
        conv_map = axes[
            (layer_index, AXIS_SHORT_CONV_CONTEXT)
        ].feature_map
        source_blocks = source_layer.conv.in_proj.weight.chunk(3, dim=0)
        expected_in = torch.cat(
            tuple(
                conv_map.fold_producer_weight(
                    mapping.fold_consumer_weight(block.detach().to(torch.float64))
                )
                for block in source_blocks
            ),
            dim=0,
        )
        expected_out = mapping.fold_producer_weight(
            conv_map.fold_consumer_weight(
                source_layer.conv.out_proj.weight.detach().to(torch.float64)
            )
        )
        expected_fir = source_layer.conv.conv.weight.detach().index_select(
            0,
            conv_map.kept_source_ids,
        )
        for actual, expected in (
            (target_layer.conv.in_proj.weight, expected_in),
            (target_layer.conv.out_proj.weight, expected_out),
            (target_layer.conv.conv.weight, expected_fir),
        ):
            torch.testing.assert_close(
                actual,
                expected.to(dtype=actual.dtype),
                rtol=0,
                atol=0,
            )

    assert not hasattr(child, "sparse_feature_fold_bundle")
    assert all("sparse_feature" not in name for name in child.state_dict())
    descriptor = extract_lfm2_dynamic_physical_layout_from_config(child.config)
    source_receipt = materialize_lfm2_dynamic_blank_child(
        child,
        child.config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.inference_mode():
        expected_logits = child(inputs, use_cache=False).logits
    checkpoint = tmp_path / "sparse-bundle-folded-child"
    child.save_pretrained(checkpoint, safe_serialization=True)
    assert all("sparse" not in path.name for path in checkpoint.iterdir())
    config_payload = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    fresh_config = transformers.Lfm2Config(**config_payload)
    recovered = extract_lfm2_dynamic_physical_layout_from_config(fresh_config)
    fresh = transformers.Lfm2ForCausalLM(fresh_config)
    fresh_receipt = materialize_lfm2_dynamic_blank_child(
        fresh,
        fresh_config,
        recovered,
        expected_descriptor_sha256=recovered.descriptor_sha256,
    )
    state = safetensors_torch.load_file(str(checkpoint / "model.safetensors"))
    loaded = fresh.load_state_dict(state, strict=True)
    with torch.inference_mode():
        actual_logits = fresh(inputs, use_cache=False).logits
    assert loaded.missing_keys == []
    assert loaded.unexpected_keys == []
    assert fresh_receipt == source_receipt
    assert recovered == descriptor
    assert sum(parameter.numel() for parameter in fresh.parameters()) == (
        descriptor.accounting.unique_parameter_elements
    )
    assert tuple(fresh.state_dict()) == tuple(child.state_dict())
    for name, expected in child.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    assert not hasattr(fresh, "sparse_feature_fold_bundle")


@pytest.mark.parametrize("identity", ("parent_state", "trace_cache"))
def test_sparse_bundle_wrong_state_or_cache_fails_before_any_target_write(
    identity: str,
) -> None:
    parent = _sparse_parent(seed=4121)
    child = _heterogeneous_child()
    fixture_kwargs = (
        {"parent_state_fingerprint": "c" * 64}
        if identity == "parent_state"
        else {"evidence_cache_sha256": "d" * 64}
    )
    bundle, source_graph, target_graph = _sparse_bundle_fixture(
        parent,
        child,
        **fixture_kwargs,
    )
    before = _blank_state(child)

    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="live parent state|sealed trace cache",
    ):
        _fold_dense_blank_from_parent(
            parent,
            child,
            _residual_map(parent, 8),
            sparse_feature_fold_bundle=bundle,
            source_graph=source_graph,
            target_graph=target_graph,
            trace_cache_key=_SPARSE_CACHE_KEY,
        )

    _assert_state_exact(child, before)


@pytest.mark.parametrize("coverage", ("missing", "extra"))
def test_sparse_bundle_coverage_fails_before_any_target_write(coverage: str) -> None:
    parent = _sparse_parent(seed=4131)
    child = _heterogeneous_child()
    bundle, source_graph, target_graph = _sparse_bundle_fixture(parent, child)
    if coverage == "missing":
        candidate = build_lfm2_sparse_feature_fold_bundle(
            source_graph=source_graph,
            target_graph=target_graph,
            source_graph_sha256=source_graph.fingerprint,
            target_graph_sha256=target_graph.fingerprint,
            axes=bundle.axes[:-1],
        )
    else:
        candidate = _unsafe_bundle_axes(bundle, (*bundle.axes, bundle.axes[0]))
    before = _blank_state(child)

    with pytest.raises(Lfm2DenseBudgetFoldError):
        _fold_dense_blank_from_parent(
            parent,
            child,
            _residual_map(parent, 8),
            sparse_feature_fold_bundle=candidate,
            source_graph=source_graph,
            target_graph=target_graph,
            trace_cache_key=_SPARSE_CACHE_KEY,
        )

    _assert_state_exact(child, before)


@pytest.mark.parametrize("payload", ("map", "fold", "witness"))
def test_sparse_bundle_tamper_fails_before_any_target_write(payload: str) -> None:
    parent = _sparse_parent(seed=4141)
    child = _heterogeneous_child()
    bundle, source_graph, target_graph = _sparse_bundle_fixture(parent, child)
    axis = bundle.axes[0]
    if payload == "map":
        axis.feature_map.kept_source_ids[0] = axis.feature_map.kept_source_ids[1]
    elif payload == "fold":
        assert int(axis.body_route_fold.write_coefficients.numel()) > 0
        axis.body_route_fold.write_coefficients[0] += 0.5
    else:
        axis.source_witness.parent_weight[0, 0] += 0.5
    before = _blank_state(child)

    with pytest.raises(Lfm2DenseBudgetFoldError, match="failed authentication"):
        _fold_dense_blank_from_parent(
            parent,
            child,
            _residual_map(parent, 8),
            sparse_feature_fold_bundle=bundle,
            source_graph=source_graph,
            target_graph=target_graph,
            trace_cache_key=_SPARSE_CACHE_KEY,
        )

    _assert_state_exact(child, before)


def test_sparse_bundle_live_writer_mutation_is_reauthenticated_before_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _sparse_parent(seed=4151)
    child = _heterogeneous_child()
    bundle, source_graph, target_graph = _sparse_bundle_fixture(parent, child)
    sealed_state = fingerprint_model_state(parent)
    with torch.no_grad():
        parent.model.layers[0].feed_forward.w2.weight[0, 0] += 1.0
    monkeypatch.setattr(
        dense_fold_module,
        "fingerprint_model_state",
        lambda _model: sealed_state,
    )
    before = _blank_state(child)

    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="live route/value authentication",
    ):
        _fold_dense_blank_from_parent(
            parent,
            child,
            _residual_map(parent, 8),
            sparse_feature_fold_bundle=bundle,
            source_graph=source_graph,
            target_graph=target_graph,
            trace_cache_key=_SPARSE_CACHE_KEY,
        )

    _assert_state_exact(child, before)


def test_sparse_bundle_verify_axis_swap_fails_before_any_target_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _sparse_parent(seed=4156)
    child = _heterogeneous_child()
    bundle_a, source_graph, target_graph = _sparse_bundle_fixture(parent, child)
    bundle_b, source_graph_b, target_graph_b = _sparse_bundle_fixture(
        parent,
        child,
        route_seed_offset=1_000,
    )
    assert source_graph_b.fingerprint == source_graph.fingerprint
    assert target_graph_b.fingerprint == target_graph.fingerprint
    assert bundle_b.bundle_sha256 != bundle_a.bundle_sha256
    assert tuple(axis.admission_sha256 for axis in bundle_b.axes) != tuple(
        axis.admission_sha256 for axis in bundle_a.axes
    )
    original_verify = Lfm2SparseFeatureFoldBundle.verify
    verify_calls = 0

    def _verify_then_swap(self):
        nonlocal verify_calls
        verified = original_verify(self)
        if self is bundle_a:
            verify_calls += 1
            object.__setattr__(self, "axes", bundle_b.axes)
        return verified

    monkeypatch.setattr(Lfm2SparseFeatureFoldBundle, "verify", _verify_then_swap)
    before = _blank_state(child)

    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="sealed admission order|pre-write closure",
    ):
        _fold_dense_blank_from_parent(
            parent,
            child,
            _residual_map(parent, 8),
            sparse_feature_fold_bundle=bundle_a,
            source_graph=source_graph,
            target_graph=target_graph,
            trace_cache_key=_SPARSE_CACHE_KEY,
        )

    assert verify_calls == 1
    _assert_state_exact(child, before)


def test_sparse_bundle_rejects_factorized_short_conv_before_write() -> None:
    parent = _sparse_parent(seed=4161)
    dense_child = _heterogeneous_child()
    bundle, source_graph, target_graph = _sparse_bundle_fixture(parent, dense_child)
    factorized_child = _heterogeneous_child(factorized_writer=True)
    before = _blank_state(factorized_child)

    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="requires dense source and target",
    ):
        _fold_dense_blank_from_parent(
            parent,
            factorized_child,
            _residual_map(parent, 8),
            sparse_feature_fold_bundle=bundle,
            source_graph=source_graph,
            target_graph=target_graph,
            trace_cache_key=_SPARSE_CACHE_KEY,
        )

    _assert_state_exact(factorized_child, before)


def test_sparse_bundle_restart_capsule_configuration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = _sparse_parent(seed=4171)
    child = _heterogeneous_child()
    bundle, _source_graph, _target_graph = _sparse_bundle_fixture(parent, child)
    selected_target = object.__new__(Lfm2DenseBudgetedSelectedTarget)
    trace_cache = object.__new__(Lfm2CircuitTraceCache)
    monkeypatch.setattr(
        dense_fold_module,
        "_verify_dense_trace_cache",
        lambda cache, _lease: cache,
    )
    capsule_root = tmp_path / "capsules"
    config = Lfm2DenseBudgetFoldConfig(
        restart_capsule_directory=capsule_root,
        resume_from_restart_capsule=capsule_root,
    )

    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="cannot use restart capsules",
    ):
        Lfm2DenseBudgetModelOperator(
            selected_target,
            trace_cache,
            config=config,
            sparse_feature_fold_bundle=bundle,
        )


def test_structural_fold_progress_is_hash_chained_and_semantics_preserving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4151,
    )
    baseline_child = _heterogeneous_child()
    observed_child = _heterogeneous_child()
    mapping = _residual_map(parent, 8)
    real_release = dense_fold_module._release_cpu_allocator_retention
    monkeypatch.setattr(
        dense_fold_module,
        "_release_cpu_allocator_retention",
        lambda **_metadata: {},
    )
    baseline_report = _fold_dense_blank_from_parent(parent, baseline_child, mapping)
    monkeypatch.setattr(
        dense_fold_module,
        "_release_cpu_allocator_retention",
        real_release,
    )
    events: list[tuple[str, dict[str, object]]] = []

    def sink(stage, metadata) -> None:
        events.append((stage, dict(metadata)))

    progress = _Lfm2DenseFoldProgress(
        sink,
        action_sha256="d" * 64,
        catalog_id="heterogeneous-progress-test",
        parent_cache_key="e" * 64,
        target_hidden_size=8,
    )
    observed_report = _fold_dense_blank_from_parent(
        parent,
        observed_child,
        mapping,
        progress=progress,
    )

    assert baseline_report == observed_report
    assert tuple(baseline_child.state_dict()) == tuple(observed_child.state_dict())
    for name, expected in baseline_child.state_dict().items():
        torch.testing.assert_close(
            observed_child.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )
    expected_stages = [
        "dense_structural_token_transport_started",
        "dense_structural_token_transport_completed",
    ]
    for _layer_index in range(4):
        expected_stages.extend(
            (
                "dense_structural_layer_started",
                "dense_structural_layer_completed",
            )
        )
    expected_stages.append("dense_structural_fold_completed")
    assert [stage for stage, _metadata in events] == expected_stages
    previous = "0" * 64
    for sequence, (stage, metadata) in enumerate(events):
        assert metadata["fold_progress_sequence"] == sequence
        assert metadata["fold_progress_previous_event_sha256"] == previous
        body = {
            "schema": metadata["fold_progress_schema"],
            "binding_sha256": metadata["fold_progress_binding_sha256"],
            "sequence": metadata["fold_progress_sequence"],
            "previous_event_sha256": metadata[
                "fold_progress_previous_event_sha256"
            ],
            "stage": stage,
            **{
                key: value
                for key, value in metadata.items()
                if not key.startswith("fold_progress_")
            },
        }
        assert metadata["fold_progress_event_sha256"] == canonical_sha256(body)
        previous = str(metadata["fold_progress_event_sha256"])
    completed_layers = [
        metadata
        for stage, metadata in events
        if stage == "dense_structural_layer_completed"
    ]
    assert [metadata["layer_index"] for metadata in completed_layers] == [0, 1, 2, 3]
    assert all(
        metadata["cpu_allocator_trim_attempted"]
        == metadata["cpu_allocator_trim_supported"]
        for metadata in completed_layers
    )
    assert observed_report["per_layer_cpu_fp64_temporary_release"] is True


@pytest.mark.parametrize(
    "reserved_name",
    (
        "schema",
        "sequence",
        "stage",
        "process_rss_bytes",
        "fold_progress_event_sha256",
    ),
)
def test_structural_fold_progress_rejects_authenticated_field_shadowing(
    reserved_name: str,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    progress = _Lfm2DenseFoldProgress(
        lambda stage, metadata: events.append((stage, dict(metadata))),
        action_sha256="a" * 64,
        catalog_id="metadata-shadow-test",
        parent_cache_key="b" * 64,
        target_hidden_size=8,
    )

    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="shadows authenticated fields",
    ):
        progress.emit("reserved-field-test", **{reserved_name: "forged"})

    assert events == []
    progress.emit("valid-event", evidence="accepted")
    assert events[0][1]["fold_progress_sequence"] == 0


def test_allocator_cleanup_failures_are_observable_but_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_gc() -> int:
        raise RuntimeError("synthetic gc failure")

    def fail_library_load(_name) -> object:
        raise OSError("synthetic allocator lookup failure")

    monkeypatch.setattr(dense_fold_module.gc, "collect", fail_gc)
    monkeypatch.setattr(dense_fold_module.ctypes, "CDLL", fail_library_load)

    report = dense_fold_module._release_cpu_allocator_retention(
        caller_declared_release_plan_units=3,
    )

    assert report["caller_declared_release_plan_units"] == 3
    assert report["python_gc_attempted"] is True
    assert report["python_gc_succeeded"] is False
    assert report["python_gc_collected_objects"] is None
    assert report["python_gc_error_type"] == "RuntimeError"
    assert report["cpu_allocator_trim_supported"] is False
    assert report["cpu_allocator_trim_attempted"] is False
    assert report["cpu_allocator_trim_error_type"] == "OSError"


def test_same_geometry_dense_child_remains_supported() -> None:
    layer_types = ("conv", "full_attention")
    parent = _model(
        _config(
            hidden=8,
            intermediate=12,
            query_heads=2,
            key_value_heads=1,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4201,
    )
    child = _model(
        _config(
            hidden=4,
            intermediate=6,
            query_heads=1,
            key_value_heads=1,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4202,
    )
    _zero(child)

    report = _fold_dense_blank_from_parent(
        parent,
        child,
        _residual_map(parent, 4),
    )

    assert report["complete_target_parameter_overwrite"] is True
    conv = report["layers"][0]["operator"]
    assert conv["source_internal_channels"] == 8
    assert conv["target_internal_channels"] == 4
    attention = report["layers"][1]["operator"]
    assert attention["target_physical_geometry"] == {
        "query_heads": 1,
        "key_value_heads": 1,
        "head_dim": 4,
    }


def test_factorized_short_conv_writer_installs_low_rank_of_dense_folded_action() -> (
    None
):
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4301,
    )
    dense_child = _heterogeneous_child()
    factor_child = _heterogeneous_child(factorized_writer=True)
    mapping = _residual_map(parent, 8)
    parent_before = fingerprint_model_state(parent)

    dense_report = _fold_dense_blank_from_parent(parent, dense_child, mapping)
    factor_report = _fold_dense_blank_from_parent(parent, factor_child, mapping)

    dense_writer = dense_child.model.layers[0].conv.out_proj.weight.detach().float()
    factors = factor_child.model.layers[0].conv.out_proj
    composed = (factors[1].weight @ factors[0].weight).detach().float()
    actual_relative_error = float(
        torch.linalg.vector_norm(dense_writer - composed)
        / torch.linalg.vector_norm(dense_writer)
    )
    operator = factor_report["layers"][0]["operator"]
    factorization = operator["factorization"]

    assert dense_report["factor_writer_layer_count"] == 0
    assert factor_report["factor_writer_layer_count"] == 1
    assert factor_report["factor_writer_cached_output_action_evidence_count"] == 0
    assert operator["kind"] == (
        "factorized_residual_internal_short_conv_transport"
    )
    assert operator["target_writer_topology"] == "factorized"
    assert operator["target_writer_rank"] == 3
    assert factorization["method"] == (
        "deterministic_structural_cpu_fp64_exact_svd_truncation"
    )
    assert factorization["optimization_used"] is False
    assert factorization["training_used"] is False
    assert factorization["parent_forward_calls"] == 0
    assert factorization["relative_frobenius_error"] == pytest.approx(
        actual_relative_error,
        abs=2e-6,
    )
    assert torch.linalg.matrix_rank(composed).item() <= 3
    assert torch.linalg.vector_norm(dense_writer - composed) < (
        torch.linalg.vector_norm(dense_writer)
    )
    assert sum(parameter.numel() for parameter in dense_child.parameters()) - sum(
        parameter.numel() for parameter in factor_child.parameters()
    ) == 8 * 10 - 3 * (8 + 10)
    assert factor_report["complete_target_parameter_overwrite"] is True
    assert fingerprint_model_state(parent) == parent_before
    assert all(parameter.grad is None for parameter in factor_child.parameters())


def test_factorized_short_conv_writer_fresh_checkpoint_reload_is_exact(
    tmp_path: Path,
) -> None:
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4401,
    )
    child = _heterogeneous_child(factorized_writer=True)
    report = _fold_dense_blank_from_parent(
        parent,
        child,
        _residual_map(parent, 8),
    )
    descriptor = extract_lfm2_dynamic_physical_layout_from_config(child.config)
    source_receipt = materialize_lfm2_dynamic_blank_child(
        child,
        child.config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.inference_mode():
        expected_logits = child(inputs, use_cache=False).logits

    checkpoint = tmp_path / "factor-folded-child"
    child.save_pretrained(checkpoint, safe_serialization=True)
    config_payload = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    fresh_config = transformers.Lfm2Config(**config_payload)
    recovered = extract_lfm2_dynamic_physical_layout_from_config(fresh_config)
    fresh = transformers.Lfm2ForCausalLM(fresh_config)
    fresh_receipt = materialize_lfm2_dynamic_blank_child(
        fresh,
        fresh_config,
        recovered,
        expected_descriptor_sha256=recovered.descriptor_sha256,
    )
    state = safetensors_torch.load_file(str(checkpoint / "model.safetensors"))
    loaded = fresh.load_state_dict(state, strict=True)
    with torch.inference_mode():
        actual_logits = fresh(inputs, use_cache=False).logits

    assert report["complete_target_parameter_overwrite"] is True
    assert descriptor.layers[0].short_conv_writer_rank == 3
    assert sum(parameter.numel() for parameter in fresh.parameters()) == (
        descriptor.accounting.unique_parameter_elements
    )
    assert loaded.missing_keys == []
    assert loaded.unexpected_keys == []
    assert recovered == descriptor
    assert fresh_receipt == source_receipt
    assert tuple(fresh.state_dict()) == tuple(child.state_dict())
    for name, expected in child.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)


def test_factorized_short_conv_writer_tamper_fails_before_any_target_write() -> None:
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4501,
    )
    child = _heterogeneous_child(factorized_writer=True)
    child.model.layers[0].conv.writer_rank = 4

    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="factor writer disagrees",
    ):
        _fold_dense_blank_from_parent(
            parent,
            child,
            _residual_map(parent, 8),
        )

    assert all(
        int(torch.count_nonzero(parameter).item()) == 0
        for parameter in child.parameters()
    )


def test_factorized_short_conv_writer_preserves_target_dtype() -> None:
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4601,
    )
    child = _heterogeneous_child(factorized_writer=True).to(dtype=torch.bfloat16)
    _zero(child)

    report = _fold_dense_blank_from_parent(
        parent,
        child,
        _residual_map(parent, 8),
    )

    factors = child.model.layers[0].conv.out_proj
    assert factors[0].weight.dtype == torch.bfloat16
    assert factors[1].weight.dtype == torch.bfloat16
    assert all(parameter.dtype == torch.bfloat16 for parameter in child.parameters())
    assert all(parameter.grad is None for parameter in child.parameters())
    assert report["training_used"] is False
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0


def test_zero_dense_writer_action_installs_two_explicit_zero_factors() -> None:
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4701,
    )
    with torch.no_grad():
        parent.model.layers[0].conv.out_proj.weight.zero_()
    child = _heterogeneous_child(factorized_writer=True)

    report = _fold_dense_blank_from_parent(
        parent,
        child,
        _residual_map(parent, 8),
    )

    factors = child.model.layers[0].conv.out_proj
    factorization = report["layers"][0]["operator"]["factorization"]
    assert factorization["method"] == "deterministic_structural_zero_dense_action"
    assert factorization["relative_frobenius_error"] == 0.0
    assert torch.count_nonzero(factors[0].weight).item() == 0
    assert torch.count_nonzero(factors[1].weight).item() == 0
    assert report["complete_target_parameter_overwrite"] is True
    assert report["target_nonzero_parameter_elements"] > 0


def test_cached_output_action_factorization_is_authenticated_and_tamper_evident() -> (
    None
):
    dense = torch.arange(1, 49, dtype=torch.float64).reshape(6, 8)
    covariance = torch.diag(
        torch.tensor([12.0, 9.0, 4.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    )
    evidence = _seal_short_conv_output_action_evidence(
        layer_index=3,
        target_hidden_size=6,
        sampled_output_row_count=32,
        parent_cache_key="a" * 64,
        selective_session_sha256="b" * 64,
        selected_payload_sha256s=("c" * 64,),
        covariance=covariance,
    )

    encoder, decoder, report = _factorize_short_conv_dense_writer_action(
        dense,
        2,
        output_action_evidence=evidence,
    )
    repeated_encoder, repeated_decoder, repeated_report = (
        _factorize_short_conv_dense_writer_action(
            dense,
            2,
            output_action_evidence=evidence,
        )
    )

    assert report["method"] == (
        "authenticated_cached_output_action_eigenspace_projection"
    )
    assert report["cached_output_action_evidence_used"] is True
    assert report["cached_output_action_evidence"]["evidence_sha256"] == (
        evidence.evidence_sha256
    )
    assert torch.equal(encoder, repeated_encoder)
    assert torch.equal(decoder, repeated_decoder)
    assert report["factorization_sha256"] == repeated_report["factorization_sha256"]

    evidence.covariance[0, 0] += 1.0
    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="changed after sealing",
    ):
        _factorize_short_conv_dense_writer_action(
            dense,
            2,
            output_action_evidence=evidence,
        )
