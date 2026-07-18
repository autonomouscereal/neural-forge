from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_layer_compiler as layer_compiler  # noqa: E402
import neural_forge.hf_lfm2_cpcf_v5_model_compiler as v5  # noqa: E402


def test_v5_has_distinct_compiler_resume_and_artifact_fingerprints() -> None:
    implementation = v5.lfm2_cpcf_v5_compiler_implementation()

    assert implementation["schema"] == (
        v5.LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA
    )
    assert implementation["model_fold_schema"] == (
        v5.LFM2_CPCF_V5_MODEL_FOLD_SCHEMA
    )
    assert implementation["resume_schema"] == v5.LFM2_CPCF_V5_RESUME_SCHEMA
    assert implementation["artifact_schema"] == (
        v5.LFM2_CPCF_V5_ARTIFACT_SCHEMA
    )
    assert implementation["legacy_v4_default_unchanged"] is True
    assert implementation["resume_implementation_status"] == (
        "fail_closed_not_yet_integrated"
    )
    assert implementation[
        "short_conv_v5_numerical_policy_integration_status"
    ] == (
        "endpoint_first_preinstall_selector_integrated_"
        "candidate_generation_pending"
    )
    assert any(
        record["filename"] == "hf_lfm2_cpcf_short_conv_v5.py"
        for record in implementation["ordered_source_files"]
    )
    short_conv_v5_source = next(
        record
        for record in implementation["ordered_source_files"]
        if record["filename"] == "hf_lfm2_cpcf_short_conv_v5.py"
    )
    assert implementation[
        "short_conv_v5_numerical_policy_source_sha256"
    ] == short_conv_v5_source["sha256"]
    assert len(
        implementation[
            "short_conv_v5_numerical_policy_source_sha256"
        ]
    ) == 64
    assert len(implementation["ordered_source_files"]) >= 8
    assert all(
        len(record["sha256"]) == 64
        for record in implementation["ordered_source_files"]
    )
    assert len(implementation["implementation_sha256"]) == 64


def test_v5_resume_fails_closed_before_touching_models() -> None:
    with pytest.raises(
        v5.Lfm2CpcfV5ModelCompilerError,
        match="resume is not yet integrated",
    ):
        v5.compile_lfm2_cpcf_v5_selective_model_fold(
            None,
            None,
            trace_adapter=None,
            evidence_provider=None,
            source_tied_table=None,
            blank_target_report={},
            planned_target_unique_parameters=1,
            runtime_policy=None,
            resume_state=object(),
        )


def test_layer_compiler_exposes_opt_in_candidate_replay_without_new_default():
    signature = inspect.signature(
        layer_compiler.compile_and_install_lfm2_cpcf_layer
    )
    parameter = signature.parameters["candidate_replay_evaluator"]
    selector_parameter = signature.parameters["candidate_selector"]

    assert parameter.default is None
    assert selector_parameter.default is None
    assert (
        signature.parameters["short_conv_v4_solver_policy"].default
        is None
    )


def test_v5_endpoint_selector_request_fails_before_touching_models() -> None:
    policy = v5.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=(
            v5.legacy_model_fold.Lfm2CpcfModelFoldPolicy()
        ),
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=1,
        require_short_conv_v5_endpoint_first_selector=True,
    )
    with pytest.raises(
        v5.Lfm2CpcfV5ModelCompilerError,
        match="needs Lfm2CpcfV5EndpointFirstCandidateSelector",
    ):
        v5.compile_lfm2_cpcf_v5_selective_model_fold(
            None,
            None,
            trace_adapter=None,
            evidence_provider=None,
            source_tied_table=None,
            blank_target_report={},
            planned_target_unique_parameters=1,
            runtime_policy=policy,
        )


def test_physical_rss_receipt_uses_sampled_process_memory(
    monkeypatch,
) -> None:
    current = [10_000]
    peak = [10_000]
    monkeypatch.setattr(v5, "_process_rss_bytes", lambda: current[0])
    monkeypatch.setattr(v5, "_process_peak_rss_bytes", lambda: peak[0])
    guard = v5._PhysicalRssGuard(
        maximum_increase_bytes=8_000,
        full_phase_tree_logical_bytes=10_000,
        minimum_reduction_fraction=0.20,
        sample_interval_seconds=1.0,
    )
    current[0] = 15_000
    peak[0] = 15_000
    guard.checkpoint("resident_pair")
    current[0] = 11_000
    report = guard.close()

    assert report["start_rss_bytes"] == 10_000
    assert report["current_rss_bytes"] == 11_000
    assert report["peak_rss_bytes"] == 15_000
    assert report["peak_increase_bytes"] == 5_000
    assert report["maximum_increase_bytes"] == 8_000
    assert report["physical_rss_gate_passed"] is True
    assert report["sampler_stopped"] is True
    assert len(report["report_sha256"]) == 64


def test_physical_rss_gate_fails_on_full_tree_sized_growth(
    monkeypatch,
) -> None:
    current = [1_000]
    monkeypatch.setattr(v5, "_process_rss_bytes", lambda: current[0])
    monkeypatch.setattr(v5, "_process_peak_rss_bytes", lambda: current[0])
    guard = v5._PhysicalRssGuard(
        maximum_increase_bytes=800,
        full_phase_tree_logical_bytes=1_000,
        minimum_reduction_fraction=0.20,
        sample_interval_seconds=1.0,
    )
    current[0] = 1_801
    try:
        with pytest.raises(
            v5.Lfm2CpcfV5ModelCompilerError,
            match="physical RSS increase",
        ):
            guard.checkpoint("mapped_pages_resident")
    finally:
        guard.abort()


def test_physical_rss_gate_allows_zero_growth_budget(
    monkeypatch,
) -> None:
    monkeypatch.setattr(v5, "_process_rss_bytes", lambda: 4_000)
    monkeypatch.setattr(v5, "_process_peak_rss_bytes", lambda: 4_000)
    guard = v5._PhysicalRssGuard(
        maximum_increase_bytes=0,
        full_phase_tree_logical_bytes=1_000,
        minimum_reduction_fraction=1.0,
        sample_interval_seconds=1.0,
    )

    report = guard.close()

    assert report["maximum_increase_bytes"] == 0
    assert report["peak_increase_bytes"] == 0
    assert report["physical_rss_gate_passed"] is True


def test_real_toy_cache_compiles_every_layer_through_selective_v5(
    tmp_path,
    monkeypatch,
) -> None:
    pytest.importorskip("safetensors")
    transformers = pytest.importorskip("transformers")
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        Lfm2CircuitTraceCaptureConfig,
        build_lfm2_circuit_trace_cache,
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_cpcf_candidate_replay import (
        CpuReferenceLfm2CpcfCandidateReplayEvaluator,
    )
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        Lfm2CpcfModelFoldPolicy,
        _sha,
        independent_parameter_inventory,
    )
    from neural_forge.hf_lfm2_cpcf_selective_evidence import (
        build_lfm2_cpcf_selective_evidence_provider,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
        Lfm2CpcfShortConvGatePolicy,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        Lfm2CpcfShortConvV4SolverPolicy,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
        LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FOLD_SCHEMA,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
        LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
    )

    def _model(
        *,
        hidden_size: int,
        intermediate_size: int,
        attention_heads: int,
        seed: int,
        zero: bool,
    ):
        config = transformers.Lfm2Config(
            vocab_size=31,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=2,
            num_attention_heads=attention_heads,
            num_key_value_heads=1,
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
            model = transformers.Lfm2ForCausalLM(config).eval()
        if zero:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
        return model

    source = _model(
        hidden_size=8,
        intermediate_size=12,
        attention_heads=2,
        seed=1201,
        zero=False,
    )
    target = _model(
        hidden_size=4,
        intermediate_size=8,
        attention_heads=1,
        seed=1202,
        zero=True,
    )
    prompts = tuple(
        torch.tensor([[start, start + 1, start + 2, start + 3]])
        for start in range(1, 6)
    )
    forced = tuple(
        torch.tensor([[token]]) for token in range(9, 14)
    )
    batches = tuple(
        {
            "input_ids": prompt,
            "attention_mask": torch.ones_like(prompt),
            "forced_decode_ids": suffix,
        }
        for prompt, suffix in zip(prompts, forced, strict=True)
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "cache",
        source,
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=4,
            max_decode_steps=1,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(cache.path, verify=False)
    )
    catalog = provider.catalog
    prefill_by_batch = {
        reference.batch_index: reference
        for reference in catalog.phase_references
        if reference.phase == "prefill"
    }
    records = tuple(
        SimpleNamespace(
            ordinal=index,
            record_id=f"v5-e2e-{index}",
            split=split,
            donor_cache_input_sha256=(
                prefill_by_batch[index].input_sha256
            ),
            alignment_sha256=v5._sha({"alignment": index}),
            target_input_ids=prompts[index],
            attention_mask=torch.ones_like(prompts[index]),
            target_forced_decode_ids=forced[index],
            position_ids=None,
        )
        for index, split in enumerate(
            ("selection", "fit", "fit", "fit", "validation")
        )
    )

    class _TraceAdapter:
        def __init__(self, values) -> None:
            self.records = tuple(values)

        def verify(self):
            return self

    target_parameters = int(
        independent_parameter_inventory(target)["unique_parameters"]
    )
    assert target_parameters == 484
    blank_body = {
        "compiler_created_blank": True,
        "all_parameters_zero": True,
        "dense_materialized_parameters": True,
        "topology_provenance_verified": True,
        "weights_installed": False,
        "training_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "target_unique_parameters": target_parameters,
    }
    blank_report = {
        **blank_body,
        "report_sha256": _sha(blank_body),
    }
    layer_tree_bytes = sum(
        reference.logical_bytes
        for reference in catalog.dynamic_catalog.references
    )
    full_tree_bytes = (
        layer_tree_bytes + catalog.final_norm_logical_bytes
    )
    resident_budget = max(
        layer_tree_bytes,
        catalog.final_norm_logical_bytes,
    )
    gate_policy = Lfm2CpcfShortConvGatePolicy(
        maximum_fit_endpoint_relative_rmse=1e6,
        maximum_validation_endpoint_relative_rmse=1e6,
        minimum_fit_endpoint_cosine_similarity=-1.0,
        minimum_validation_endpoint_cosine_similarity=-1.0,
        enforce=True,
    )
    solver_policy = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-6,),
        variable_projection_passes=2,
        target_dtype="float32",
    )
    model_policy = Lfm2CpcfModelFoldPolicy(
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.0,
        minimum_validation_block_retention=0.0,
        short_conv_gate_policy=gate_policy,
        short_conv_v4_solver_policy=solver_policy,
    )

    class _LastCandidateSelector:
        def select(self, request):
            selected_id = request["candidate_ids"][-1]
            body = {
                "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
                "kind": "test_authenticated_last_candidate_override",
                "candidate_count": len(request["candidate_ids"]),
                "selection_reports_consumed": len(
                    request["candidate_ids"]
                ),
                "fit_reports_consumed": 0,
                "validation_reports_consumed": 0,
                "validation_accessible": False,
                "selected_candidate_id": selected_id,
                "selected_candidate_state_sha256": request[
                    "candidate_state_sha256"
                ][selected_id],
                "execution": {
                    "training_steps": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                },
            }
            return {**body, "report_sha256": _sha(body)}

    # The process-RSS gate itself is covered above. This integration test
    # freezes the probe so it tests the complete evidence/runtime wiring
    # without depending on unrelated pytest process allocations.
    monkeypatch.setattr(v5, "_process_rss_bytes", lambda: 10_000_000)
    monkeypatch.setattr(v5, "_process_peak_rss_bytes", lambda: 10_000_000)

    result = v5.compile_lfm2_cpcf_v5_selective_model_fold(
        source,
        target,
        trace_adapter=_TraceAdapter(records),
        evidence_provider=provider,
        source_tied_table=source.get_input_embeddings().weight,
        blank_target_report=blank_report,
        planned_target_unique_parameters=target_parameters,
        runtime_policy=v5.Lfm2CpcfV5RuntimePolicy(
            model_fold_policy=model_policy,
            maximum_resident_evidence_bytes=resident_budget,
            full_phase_tree_logical_bytes=full_tree_bytes,
            minimum_physical_rss_reduction_fraction=0.10,
            rss_sample_interval_seconds=1.0,
        ),
        candidate_replay_evaluator=(
            CpuReferenceLfm2CpcfCandidateReplayEvaluator()
        ),
        candidate_selector=_LastCandidateSelector(),
    )

    assert result.report["schema"] == v5.LFM2_CPCF_V5_MODEL_FOLD_SCHEMA
    assert len(result.layer_results) == 2
    assert result.report["ordered_layer_result_sha256"] == [
        receipt.result_sha256 for receipt in result.layer_results
    ]
    assert result.report[
        "authenticated_full_phase_tree_logical_bytes"
    ] == full_tree_bytes
    assert result.report[
        "physical_rss_baseline_bound_to_authenticated_catalog"
    ] is True
    residency = result.report["residency_receipt"]
    assert residency["peak_occupied_layers"] == 2
    assert residency["peak_resident_bytes"] == layer_tree_bytes
    assert residency["requested_layer_indices"] == [0, 1]
    assert residency["released_layer_indices"] == [0, 1]
    assert result.report["full_all_layer_phase_tree_materialized"] is False
    assert result.report["tensor_free_phase_shells_used"] is True
    assert all(
        not layer.tensors
        for phase in result.final_state_bank.phases
        for layer in phase.parent_evidence.layers
    )
    short_conv = result.layer_results[0].report["operator_artifact"]
    assert short_conv["schema"] == (
        LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FOLD_SCHEMA
    )
    assert short_conv["legacy_v4_report_claimed"] is False
    assert short_conv["selection_changed_from_v4"] is True
    assert short_conv["candidate_replay_backend"]["kind"] == (
        "injected_cpu_reference"
    )
    assert result.report["candidate_replay_backend_injected"] is True
    assert result.report["preinstall_candidate_selector_provided"] is True
    assert result.report["preinstall_candidate_selector_active"] is True
    assert result.report[
        "preinstall_candidate_selector_layer_indices"
    ] == [0]
    assert result.report["short_conv_v5_numerical_policy_integrated"] is False
    assert result.report[
        "short_conv_v5_numerical_candidate_generation_integrated"
    ] is False
    assert result.report[
        "short_conv_v5_endpoint_first_selector_integrated"
    ] is False
    assert result.report[
        "short_conv_v5_post_install_rescoring_performed"
    ] is False
    assert result.report["physical_rss_receipt"][
        "physical_rss_gate_passed"
    ] is True
    assert result.report["operation_audit"] == {
        "parent_model_forward_calls": 0,
        "target_model_forward_calls": 0,
        "generation_calls": 0,
        "parent_cache_generation_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "fallback_paths": 0,
    }
