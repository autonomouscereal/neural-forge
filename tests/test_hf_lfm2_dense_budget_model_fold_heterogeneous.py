from __future__ import annotations

import json
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
from neural_forge.hf_feature_fold import build_feature_fold_map  # noqa: E402
from neural_forge.hf_lfm2_dense_budget_model_fold import (  # noqa: E402
    Lfm2DenseBudgetFoldConfig,
    Lfm2DenseBudgetFoldError,
    _Lfm2DenseFoldProgress,
    _factorize_short_conv_dense_writer_action,
    _fold_dense_blank_from_parent,
    _seal_short_conv_output_action_evidence,
)
from neural_forge import hf_lfm2_dense_budget_model_fold as dense_fold_module  # noqa: E402
from neural_forge.hf_lfm2_dynamic_blank_materializer import (  # noqa: E402
    extract_lfm2_dynamic_physical_layout_from_config,
    materialize_lfm2_dynamic_blank_child,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (  # noqa: E402
    Lfm2HeterogeneousLayerPhysicalLayout,
    build_lfm2_heterogeneous_physical_layout,
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
