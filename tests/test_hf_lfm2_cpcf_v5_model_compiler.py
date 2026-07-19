from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_layer_compiler as layer_compiler  # noqa: E402
import neural_forge.hf_lfm2_cpcf_v5_model_compiler as v5  # noqa: E402
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (  # noqa: E402
    Lfm2CpcfV5FirBasisPolicy,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (  # noqa: E402
    Lfm2CpcfShortConvV5NumericalFrontierAdapter,
)
from neural_forge.hf_lfm2_cpcf_v5_admission_proof import (  # noqa: E402
    authenticate_lfm2_cpcf_v5_admission_chain_proof,
)


def test_v5_has_distinct_compiler_resume_and_artifact_fingerprints() -> None:
    implementation = v5.lfm2_cpcf_v5_compiler_implementation()

    assert implementation["schema"] == (v5.LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA)
    assert implementation["model_fold_schema"] == (v5.LFM2_CPCF_V5_MODEL_FOLD_SCHEMA)
    assert implementation["resume_schema"] == v5.LFM2_CPCF_V5_RESUME_SCHEMA
    assert implementation["artifact_schema"] == (v5.LFM2_CPCF_V5_ARTIFACT_SCHEMA)
    assert implementation["legacy_v4_default_unchanged"] is True
    assert implementation["resume_implementation_status"] == (
        "authenticated_partial_boundary_frontier_integrated"
    )
    assert implementation["short_conv_v5_numerical_policy_integration_status"] == (
        "real_v5_numerical_frontier_integrated_with_per_layer_provenance_gate"
    )
    assert any(
        record["filename"] == "hf_lfm2_cpcf_short_conv_v5.py"
        for record in implementation["ordered_source_files"]
    )
    assert any(
        record["filename"] == "hf_lfm2_cpcf_short_conv_v5_numerical_adapter.py"
        for record in implementation["ordered_source_files"]
    )
    short_conv_v5_source = next(
        record
        for record in implementation["ordered_source_files"]
        if record["filename"] == "hf_lfm2_cpcf_short_conv_v5.py"
    )
    assert (
        implementation["short_conv_v5_numerical_policy_source_sha256"]
        == short_conv_v5_source["sha256"]
    )
    assert len(implementation["short_conv_v5_numerical_policy_source_sha256"]) == 64
    assert len(implementation["ordered_source_files"]) >= 8
    assert all(
        len(record["sha256"]) == 64 for record in implementation["ordered_source_files"]
    )
    assert len(implementation["implementation_sha256"]) == 64


def test_v5_untyped_resume_fails_closed_before_touching_models() -> None:
    with pytest.raises(
        v5.Lfm2CpcfV5ModelCompilerError,
        match="admitted typed v5 resume state",
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
    signature = inspect.signature(layer_compiler.compile_and_install_lfm2_cpcf_layer)
    parameter = signature.parameters["candidate_replay_evaluator"]
    selector_parameter = signature.parameters["candidate_selector"]
    frontier_parameter = signature.parameters["numerical_frontier_adapter"]

    assert parameter.default is None
    assert selector_parameter.default is None
    assert frontier_parameter.default is None
    assert signature.parameters["short_conv_v4_solver_policy"].default is None


def test_v5_endpoint_selector_request_fails_before_touching_models() -> None:
    policy = v5.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=(v5.legacy_model_fold.Lfm2CpcfModelFoldPolicy()),
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


def test_required_real_v5_adapter_rejects_generic_adapter_before_models() -> None:
    class _GenericAdapter:
        def generate(self, _request):
            raise AssertionError("must not execute")

    policy = v5.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=(v5.legacy_model_fold.Lfm2CpcfModelFoldPolicy()),
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=1,
        require_short_conv_v5_real_numerical_frontier_adapter=True,
    )

    with pytest.raises(
        v5.Lfm2CpcfV5ModelCompilerError,
        match="requires the concrete authenticated",
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
            numerical_frontier_adapter=_GenericAdapter(),
        )


def test_concrete_real_v5_layer_provenance_is_policy_bound() -> None:
    adapter = Lfm2CpcfShortConvV5NumericalFrontierAdapter(
        basis_policy=Lfm2CpcfV5FirBasisPolicy(
            maximum_projected_fir_offdiagonal_relative_frobenius=0.0,
            maximum_per_lag_commutator_relative_frobenius=1.0,
        )
    )
    contract = v5._real_v5_numerical_adapter_contract(adapter)
    assert contract is not None
    zero_execution = {
        "parent_model_forward_calls": 0,
        "target_model_forward_calls": 0,
        "generation_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    provenance = {
        "implementation": contract["implementation"],
        "real_v5_numerical_algorithm": True,
        "dynamic_source_target_topology": True,
        "basis_policy_sha256": contract["basis_policy_sha256"],
        "ridge_policy_sha256": contract["ridge_policy_sha256"],
        "basis_actual_commutator_gate": 1.0,
        "basis_actual_commutator_residual": 0.75,
        "basis_actual_commutator_gate_passed": True,
        "basis_full_source_numeric_participation": True,
        "source_channels_dropped": 0,
        "source_channels_pruned": 0,
        "source_channels_quantized": 0,
        "source_fir_resized_before_basis": True,
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "selection_performed_by_adapter": False,
        "validation_performed_by_adapter": False,
        "model_forward_calls": 0,
        "training": False,
        "pruning": False,
        "quantization": False,
        "execution": zero_execution,
    }
    frontier_body = {
        "kind": "authenticated_fit_only_numerical_frontier",
        "adapter_id": adapter.adapter_id,
        "full_source_numeric_participation": True,
        "source_channels_dropped": 0,
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "execution": zero_execution,
        "adapter_provenance": provenance,
        "adapter_provenance_sha256": v5._sha(provenance),
    }
    frontier = {
        **frontier_body,
        "report_sha256": v5._sha(frontier_body),
    }
    artifact_body = {
        "numerical_frontier_adapter_active": True,
        "fit_only_numerical_generation": True,
        "legacy_v4_numerical_generation_claimed": False,
        "numerical_frontier": frontier,
    }
    artifact = {
        **artifact_body,
        "report_sha256": v5._sha(artifact_body),
    }
    receipt_body = {
        "layer_index": 4,
        "operator_kind": "short_conv",
        "operator_artifact": artifact,
    }
    receipt = SimpleNamespace(
        layer_index=4,
        report={
            **receipt_body,
            "result_sha256": v5._sha(receipt_body),
        },
    )

    proof = v5._prove_real_v5_numerical_layer(
        receipt,
        adapter_contract=contract,
    )
    assert proof is not None
    assert proof["layer_index"] == 4
    assert proof["projected_fir_offdiagonal_gate"] == 0.0
    assert proof["per_lag_commutator_gate"] == 1.0

    provenance["implementation"] = "tests.generic_adapter"
    assert (
        v5._prove_real_v5_numerical_layer(
            receipt,
            adapter_contract=contract,
        )
        is None
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
        maximum_process_rss_bytes=18_000,
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
    assert report["relative_maximum_increase_bytes"] == 8_000
    assert report["maximum_process_rss_bytes"] == 18_000
    assert report["absolute_headroom_bytes"] == 8_000
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
        maximum_process_rss_bytes=1_800,
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
        maximum_process_rss_bytes=4_000,
        full_phase_tree_logical_bytes=1_000,
        minimum_reduction_fraction=1.0,
        sample_interval_seconds=1.0,
    )

    report = guard.close()

    assert report["maximum_increase_bytes"] == 0
    assert report["peak_increase_bytes"] == 0
    assert report["physical_rss_gate_passed"] is True


def test_physical_rss_lifetime_delta_uses_lifetime_hwm_baseline(
    monkeypatch,
) -> None:
    current = [5_000]
    lifetime_hwm = [30_000]
    monkeypatch.setattr(v5, "_process_rss_bytes", lambda: current[0])
    monkeypatch.setattr(
        v5,
        "_process_peak_rss_bytes",
        lambda: lifetime_hwm[0],
    )
    guard = v5._PhysicalRssGuard(
        maximum_increase_bytes=2_000,
        maximum_process_rss_bytes=32_000,
        full_phase_tree_logical_bytes=10_000,
        minimum_reduction_fraction=0.20,
        sample_interval_seconds=1.0,
    )
    lifetime_hwm[0] = 31_000

    checkpoint = guard.checkpoint("one-more-lifetime-peak-byte-range")
    report = guard.close()

    assert checkpoint["start_rss_bytes"] == 5_000
    assert checkpoint["start_lifetime_peak_bytes"] == 30_000
    assert checkpoint["current_lifetime_peak_bytes"] == 31_000
    assert checkpoint["peak_current_rss_increase_bytes"] == 0
    assert checkpoint["lifetime_peak_increase_bytes"] == 1_000
    assert checkpoint["peak_increase_bytes"] == 1_000
    assert report["peak_increase_bytes"] == 1_000


def test_physical_rss_guard_rederives_absolute_headroom_at_actual_start(
    monkeypatch,
) -> None:
    current = [1_400]
    lifetime_hwm = [1_400]
    monkeypatch.setattr(v5, "_process_rss_bytes", lambda: current[0])
    monkeypatch.setattr(
        v5,
        "_process_peak_rss_bytes",
        lambda: lifetime_hwm[0],
    )
    # A runner estimate taken before a 400-byte allocation would have seen
    # 1_000 bytes and incorrectly inferred 500 bytes of headroom.  The guard
    # must derive its authority from the actual 1_400-byte start instead.
    guard = v5._PhysicalRssGuard(
        maximum_increase_bytes=500,
        maximum_process_rss_bytes=1_500,
        full_phase_tree_logical_bytes=1_000,
        minimum_reduction_fraction=0.50,
        sample_interval_seconds=1.0,
    )
    assert guard.start_rss_bytes == 1_400
    assert guard.relative_maximum_increase_bytes == 500
    assert guard.absolute_headroom_bytes == 100
    assert guard.maximum_increase_bytes == 100

    current[0] = 1_501
    lifetime_hwm[0] = 1_501
    try:
        with pytest.raises(
            v5.Lfm2CpcfV5PhysicalRssError,
            match="process cap 1500",
        ) as captured:
            guard.checkpoint("post-allocation-growth")
    finally:
        guard.abort()
    assert captured.value.report["maximum_process_rss_bytes"] == 1_500
    assert captured.value.report["absolute_headroom_bytes"] == 100
    assert captured.value.report["maximum_increase_bytes"] == 100


def test_runtime_contract_binds_instance_and_callable_object_state() -> None:
    class StatefulSelector:
        def __init__(self, threshold):
            self.threshold = threshold
            self.tags = {"fit", "selection"}

        def select(self, request):
            return request

    first = StatefulSelector(0.25)
    second = StatefulSelector(0.50)
    first_sha = v5._optional_runtime_contract_sha256(
        first,
        kind="candidate_selector",
    )
    second_sha = v5._optional_runtime_contract_sha256(
        second,
        kind="candidate_selector",
    )
    assert first_sha != second_sha
    first.threshold = 0.75
    assert (
        v5._optional_runtime_contract_sha256(
            first,
            kind="candidate_selector",
        )
        != first_sha
    )

    class RuntimeRecord:
        def __init__(self, name):
            self.name = name

        def as_dict(self):
            return {"name": self.name}

    class StatefulCallable:
        def __init__(self, scale):
            self.scale = scale

        def __call__(self, *args, **kwargs):
            return args, kwargs

    class Scheduler:
        def __init__(self, scale):
            self.policy = RuntimeRecord("policy")
            self.replicas = (RuntimeRecord("replica"),)
            worker = StatefulCallable(scale)
            self.gpu_worker = worker
            self.cpu_parity_worker = worker
            self.device_receipt_probe = worker
            self.installed_state_probe = worker
            self.cpu_installed_state_probe = worker
            self.host_buffer_factory = worker

        def execute(self, items):
            return items

    class Evaluator:
        def __init__(self, scale):
            self.scheduler = Scheduler(scale)

        def evaluate(self, items):
            return items

    evaluator_a = Evaluator(1.0)
    evaluator_b = Evaluator(2.0)
    assert v5._optional_runtime_contract_sha256(
        evaluator_a,
        kind="candidate_replay",
    ) != v5._optional_runtime_contract_sha256(
        evaluator_b,
        kind="candidate_replay",
    )

    class NumericalAdapter:
        def __init__(self, ridge):
            self.ridge = ridge

        def generate(self, request):
            return request

    assert v5._optional_runtime_contract_sha256(
        NumericalAdapter(1e-8),
        kind="numerical_frontier",
    ) != v5._optional_runtime_contract_sha256(
        NumericalAdapter(1e-4),
        kind="numerical_frontier",
    )


def test_runtime_contract_binds_reachable_global_callable(
    monkeypatch,
) -> None:
    selector = v5.Lfm2CpcfV5EndpointFirstCandidateSelector()
    before = v5._optional_runtime_contract_sha256(
        selector,
        kind="candidate_selector",
    )

    def replacement(_reports):
        return {"changed": True}

    monkeypatch.setattr(
        v5,
        "rescore_sealed_v4_short_conv_selection_frontier",
        replacement,
    )
    after = v5._optional_runtime_contract_sha256(
        selector,
        kind="candidate_selector",
    )

    assert after != before
    assert selector.select({"candidate_selection_reports": {}}) == {"changed": True}


def test_runtime_contract_binds_transitive_global_from_real_replay_class(
    monkeypatch,
) -> None:
    import neural_forge.hf_lfm2_cpcf_candidate_replay as replay_module

    evaluator = replay_module.CpuReferenceLfm2CpcfCandidateReplayEvaluator()
    before = v5._optional_runtime_contract_sha256(
        evaluator,
        kind="candidate_replay",
    )

    monkeypatch.setattr(
        replay_module,
        "_sha",
        lambda _value: "0" * 64,
    )

    assert (
        v5._optional_runtime_contract_sha256(
            evaluator,
            kind="candidate_replay",
        )
        != before
    )


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
    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfResumeArtifactStore,
    )
    from neural_forge.hf_lfm2_cpcf_v5_resume import (
        Lfm2CpcfV5ResumeCoordinator,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
        Lfm2CpcfShortConvGatePolicy,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        Lfm2CpcfShortConvV4SolverPolicy,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
        LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FOLD_SCHEMA,
        Lfm2CpcfShortConvNumericalFrontier,
        Lfm2CpcfShortConvNumericalRawCandidate,
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
    forced = tuple(torch.tensor([[token]]) for token in range(9, 14))
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
            donor_cache_input_sha256=(prefill_by_batch[index].input_sha256),
            alignment_sha256=v5._sha({"alignment": index}),
            target_input_ids=prompts[index],
            attention_mask=torch.ones_like(prompts[index]),
            target_forced_decode_ids=forced[index],
            position_ids=None,
        )
        for index, split in enumerate(("selection", "fit", "fit", "fit", "validation"))
    )

    class _TraceAdapter:
        def __init__(self, values) -> None:
            self.records = tuple(values)
            self.evidence_sha256 = v5._sha(
                {
                    "kind": "test-trace",
                    "record_ids": [record.record_id for record in self.records],
                }
            )
            self.target_tokenizer_package_fingerprint = v5._sha(
                {"kind": "test-target-tokenizer-package"}
            )
            self.bridge_artifact_sha256 = v5._sha({"kind": "test-token-bridge"})

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
        reference.logical_bytes for reference in catalog.dynamic_catalog.references
    )
    full_tree_bytes = layer_tree_bytes + catalog.final_norm_logical_bytes
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

    class _ToyNumericalFrontierAdapter:
        def generate(self, request):
            source_hidden = int(request.report["source_hidden_size"])
            target_hidden = int(request.report["target_hidden_size"])
            encoder = torch.zeros(
                (source_hidden, target_hidden),
                dtype=torch.float64,
            )
            for source_index in range(source_hidden):
                encoder[
                    source_index,
                    source_index % target_hidden,
                ] = 1.0

            def value(role, amount):
                template = request.target_templates[role]
                return (
                    None
                    if template is None
                    else torch.full_like(
                        template,
                        amount,
                        dtype=torch.float64,
                    )
                )

            candidates = tuple(
                Lfm2CpcfShortConvNumericalRawCandidate(
                    request_sha256=request.request_sha256,
                    candidate_id=candidate_id,
                    candidate_kind="e2e_toy_alternate",
                    in_proj_weight=value("in_proj_weight", amount),
                    in_proj_bias=value("in_proj_bias", amount),
                    fir_kernel=value("fir_kernel", amount),
                    fir_bias=value("fir_bias", amount),
                    writer_weight=value("writer_weight", amount),
                    writer_bias=value("writer_bias", amount),
                    method_provenance={
                        "method": "e2e_constant_frontier",
                        "constant": amount,
                    },
                )
                for candidate_id, amount in (
                    ("e2e-alternate-a", 0.03125),
                    ("e2e-alternate-z", -0.0625),
                )
            )
            return Lfm2CpcfShortConvNumericalFrontier(
                adapter_id="tests.e2e_toy_frontier",
                request_sha256=request.request_sha256,
                context_encoder=encoder,
                raw_candidates=candidates,
                fit_case_authentication_sha256=tuple(
                    request.report["fit_case_authentication_sha256"]
                ),
                adapter_provenance={
                    "implementation": "tests.e2e_toy_frontier.v1",
                    "real_v5_numerical_algorithm": False,
                },
            )

    class _LastCandidateSelector:
        def select(self, request):
            selected_id = next(
                candidate_id
                for candidate_id in request["candidate_ids"]
                if candidate_id != request["default_v4_selected_candidate_id"]
            )
            body = {
                "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
                "kind": "test_authenticated_last_candidate_override",
                "candidate_count": len(request["candidate_ids"]),
                "selection_reports_consumed": len(request["candidate_ids"]),
                "fit_reports_consumed": 0,
                "validation_reports_consumed": 0,
                "validation_accessible": False,
                "selected_candidate_id": selected_id,
                "selected_candidate_state_sha256": request["candidate_state_sha256"][
                    selected_id
                ],
                "execution": {
                    "training_steps": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                },
            }
            return {**body, "report_sha256": _sha(body)}

    numerical_adapter = _ToyNumericalFrontierAdapter()
    trace_adapter = _TraceAdapter(records)
    candidate_selector = _LastCandidateSelector()
    replay_evaluator = CpuReferenceLfm2CpcfCandidateReplayEvaluator()
    runtime_policy = v5.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=model_policy,
        maximum_resident_evidence_bytes=resident_budget,
        full_phase_tree_logical_bytes=full_tree_bytes,
        minimum_physical_rss_reduction_fraction=0.10,
        rss_sample_interval_seconds=1.0,
    )
    resume_coordinator = Lfm2CpcfV5ResumeCoordinator(
        Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    )

    # The process-RSS gate itself is covered above. This integration test
    # freezes the probe so it tests the complete evidence/runtime wiring
    # without depending on unrelated pytest process allocations.
    monkeypatch.setattr(v5, "_process_rss_bytes", lambda: 10_000_000)
    monkeypatch.setattr(v5, "_process_peak_rss_bytes", lambda: 10_000_000)

    result = v5.compile_lfm2_cpcf_v5_selective_model_fold(
        source,
        target,
        trace_adapter=trace_adapter,
        evidence_provider=provider,
        source_tied_table=source.get_input_embeddings().weight,
        blank_target_report=blank_report,
        planned_target_unique_parameters=target_parameters,
        runtime_policy=runtime_policy,
        candidate_replay_evaluator=replay_evaluator,
        candidate_selector=candidate_selector,
        numerical_frontier_adapter=numerical_adapter,
        resume_coordinator=resume_coordinator,
    )

    assert result.report["schema"] == v5.LFM2_CPCF_V5_MODEL_FOLD_SCHEMA
    assert len(result.layer_results) == 2
    assert result.report["ordered_layer_result_sha256"] == [
        receipt.result_sha256 for receipt in result.layer_results
    ]
    assert (
        result.report["authenticated_full_phase_tree_logical_bytes"] == full_tree_bytes
    )
    assert result.report["physical_rss_baseline_bound_to_authenticated_catalog"] is True
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
        LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FOLD_SCHEMA
    )
    assert short_conv["legacy_v4_report_claimed"] is False
    assert short_conv["selection_changed_from_v4"] is True
    assert short_conv["candidate_replay_backend"]["kind"] == ("injected_cpu_reference")
    assert result.report["candidate_replay_backend_injected"] is True
    assert result.report["preinstall_candidate_selector_provided"] is True
    assert result.report["preinstall_candidate_selector_active"] is True
    assert result.report["preinstall_candidate_selector_layer_indices"] == [0]
    assert result.report["short_conv_v5_numerical_policy_integrated"] is False
    assert (
        result.report["short_conv_v5_numerical_candidate_generation_integrated"]
        is False
    )
    assert result.report["short_conv_numerical_frontier_adapter_provided"] is True
    assert result.report["short_conv_numerical_frontier_adapter_active"] is True
    assert result.report["short_conv_numerical_frontier_adapter_layer_indices"] == [0]
    assert result.report["short_conv_v5_real_numerical_adapter_required"] is False
    assert (
        result.report["short_conv_v5_real_numerical_adapter_concrete_instance_provided"]
        is False
    )
    assert result.report["short_conv_v5_real_numerical_provenance_layer_indices"] == []
    assert result.report["short_conv_v5_real_numerical_layer_proofs"] == []
    assert result.report["short_conv_v5_endpoint_first_selector_integrated"] is False
    assert result.report["short_conv_v5_post_install_rescoring_performed"] is False
    selected_amount = {
        "e2e-alternate-a": 0.03125,
        "e2e-alternate-z": -0.0625,
    }[short_conv["selected_candidate_id"]]
    installed_conv = target.model.layers[0].conv
    assert torch.equal(
        installed_conv.in_proj.weight,
        torch.full_like(installed_conv.in_proj.weight, selected_amount),
    )
    assert torch.equal(
        installed_conv.conv.weight,
        torch.full_like(installed_conv.conv.weight, selected_amount),
    )
    assert torch.equal(
        installed_conv.out_proj.weight,
        torch.full_like(installed_conv.out_proj.weight, selected_amount),
    )
    assert result.report["physical_rss_receipt"]["physical_rss_gate_passed"] is True
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

    cold_target_sha256 = v5.legacy_model_fold.fingerprint_model_state(target)
    assert [
        record["next_layer_index"]
        for record in result.report["resume_admitted_prefixes"]
    ] == [0, 1, 2]
    assert result.report["resume_input_admission_chain_proof"] is None
    assert result.report["resume_input_admission_chain_proof_sha256"] is None
    for record in result.report["resume_admitted_prefixes"]:
        proof = authenticate_lfm2_cpcf_v5_admission_chain_proof(
            record["admission_chain_proof"]
        )
        assert proof["proof_sha256"] == (
            record["admission_chain_proof_sha256"]
        )
        assert proof["admission_document"]["admission_sha256"] == (
            record["admission_sha256"]
        )
    intermediate_admission = result.report["resume_admitted_prefixes"][1][
        "admission_sha256"
    ]
    resume_state = resume_coordinator.open_admitted(intermediate_admission)
    resumed_target = _model(
        hidden_size=4,
        intermediate_size=8,
        attention_heads=1,
        seed=1202,
        zero=True,
    )
    resumed_provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(cache.path, verify=False)
    )
    resumed = v5.compile_lfm2_cpcf_v5_selective_model_fold(
        source,
        resumed_target,
        trace_adapter=_TraceAdapter(records),
        evidence_provider=resumed_provider,
        source_tied_table=source.get_input_embeddings().weight,
        blank_target_report=blank_report,
        planned_target_unique_parameters=target_parameters,
        runtime_policy=runtime_policy,
        candidate_replay_evaluator=(CpuReferenceLfm2CpcfCandidateReplayEvaluator()),
        candidate_selector=_LastCandidateSelector(),
        numerical_frontier_adapter=_ToyNumericalFrontierAdapter(),
        resume_state=resume_state,
        resume_coordinator=resume_coordinator,
    )
    assert resumed.report["resume_used"] is True
    assert resumed.report["resume_start_layer_index"] == 1
    assert resumed.report["resume_completed_prefix_layer_indices"] == [0]
    assert resumed.report["resume_executed_layer_indices"] == [1]
    assert resumed.report["resume_completed_layers_replayed"] is False
    assert resumed.report["resume_actual_upstream_child_state_restored"] is True
    input_proof = authenticate_lfm2_cpcf_v5_admission_chain_proof(
        resumed.report["resume_input_admission_chain_proof"]
    )
    assert input_proof["proof_sha256"] == (
        resumed.report["resume_input_admission_chain_proof_sha256"]
    )
    assert input_proof["admission_document"]["admission_sha256"] == (
        resumed.report["resume_admission_sha256"]
    )
    assert (
        v5.legacy_model_fold.fingerprint_model_state(resumed_target)
        == cold_target_sha256
    )
    assert resumed.boundary_bundle.bundle_sha256 == (
        result.boundary_bundle.bundle_sha256
    )
    assert resumed.layer_results[0].result_sha256 == (
        result.layer_results[0].result_sha256
    )
    assert resumed.layer_results[1].input_bank_sha256 == (
        result.layer_results[1].input_bank_sha256
    )
    assert resumed.layer_results[1].next_bank_sha256 == (
        result.layer_results[1].next_bank_sha256
    )
    assert resumed.final_state_bank.state_bank_sha256 == (
        result.final_state_bank.state_bank_sha256
    )
    assert (
        resumed.final_norm_report["report_sha256"]
        == (result.final_norm_report["report_sha256"])
    )

    class _MutatingSelector:
        def __init__(self):
            self.calls = 0
            self.delegate = _LastCandidateSelector()

        def select(self, request):
            self.calls += 1
            return self.delegate.select(request)

    mutating_target = _model(
        hidden_size=4,
        intermediate_size=8,
        attention_heads=1,
        seed=1203,
        zero=True,
    )
    mutating_provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(cache.path, verify=False)
    )
    mutating_coordinator = Lfm2CpcfV5ResumeCoordinator(
        Lfm2CpcfResumeArtifactStore(tmp_path / "v5-mutating-runtime-resume")
    )
    with pytest.raises(
        v5.Lfm2CpcfV5ModelCompilerError,
        match="runtime state changed before resumable prefix publication",
    ):
        v5.compile_lfm2_cpcf_v5_selective_model_fold(
            source,
            mutating_target,
            trace_adapter=_TraceAdapter(records),
            evidence_provider=mutating_provider,
            source_tied_table=source.get_input_embeddings().weight,
            blank_target_report=blank_report,
            planned_target_unique_parameters=target_parameters,
            runtime_policy=runtime_policy,
            candidate_replay_evaluator=(CpuReferenceLfm2CpcfCandidateReplayEvaluator()),
            candidate_selector=_MutatingSelector(),
            numerical_frontier_adapter=_ToyNumericalFrontierAdapter(),
            resume_coordinator=mutating_coordinator,
        )
    assert len(mutating_coordinator._terminal_pending_keys()) == 1
