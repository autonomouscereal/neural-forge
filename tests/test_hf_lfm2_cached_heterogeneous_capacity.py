from __future__ import annotations

import hashlib
import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from test_dynamic_lfm2_heterogeneous_budgeted_materialization import (
    _parent,
)
from test_hf_lfm2_dynamic_closed_loop_adapter import _batch

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache
from neural_forge.dynamic_behavior_gate import (
    observe_model_execution,
)
from neural_forge.dynamic_model_inspector import (
    inspect_model_circuits,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_parameter_frontier import (
    Lfm2HeterogeneousLayoutSelectionInput,
)
from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (
    ATTENTION_GEOMETRY_ENABLED_POLICY,
    ATTENTION_GEOMETRY_RISK_METHOD,
    AUTOMATIC_MLP_OPTION_POLICY,
    CAPACITY_EVIDENCE_SCHEMA,
    CAPACITY_OPTION_BANK_SCHEMA,
    MAX_DYNAMIC_SHORT_CONV_WRITER_RANKS,
    SHORT_CONV_WRITER_ACTION_RISK_METHOD,
    Lfm2CachedHeterogeneousCapacityOptionBank,
    Lfm2CachedHeterogeneousCapacityError,
    _ShortConvWriterActionAccumulator,
    _cached_mlp_nonlinear_statistics,
    build_lfm2_cached_heterogeneous_capacity_option_bank,
    derive_lfm2_automatic_mlp_width_policy,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    CACHED_VERIFIED_TRACE_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceSelectedBatchPayload,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (
    build_lfm2_short_conv_layer_option,
    build_lfm2_heterogeneous_target_pins,
)
from neural_forge.lfm2_native_exact4x_target import (
    lfm2_identity_tokenizer_evidence_sha256,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_writer_capacity_contract_uses_new_schema_generation() -> None:
    assert CAPACITY_EVIDENCE_SCHEMA.endswith("-v4")
    assert CAPACITY_OPTION_BANK_SCHEMA.endswith("-v4")
    assert SHORT_CONV_WRITER_ACTION_RISK_METHOD.endswith("-v2")


def _bf16_runtime_closed_mlp_evidence() -> dict[str, torch.Tensor]:
    gate = torch.tensor([[0.1, 0.1]], dtype=torch.bfloat16)
    up = torch.tensor([[3.0, 3.0]], dtype=torch.bfloat16)
    post_swiglu = torch.nn.functional.silu(gate) * up
    post_operator = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
    mlp_down = torch.tensor(
        [[0.00390625, 0.00390625]],
        dtype=torch.bfloat16,
    )
    post_layer = post_operator + mlp_down
    zeros = torch.zeros_like(post_operator)
    return {
        "operator_input": zeros,
        "mlp_input": zeros,
        "gate": gate,
        "up": up,
        "post_swiglu": post_swiglu,
        "mlp_down": mlp_down,
        "post_operator": post_operator,
        "post_layer": post_layer,
        "final_post_layer": post_layer,
    }


def test_cached_mlp_closure_authenticates_bf16_before_float32_statistics() -> None:
    evidence = _bf16_runtime_closed_mlp_evidence()

    # Both identities are exact runtime-dtype captures.  Reinterpreting their
    # independently rounded operands in float32 reproduces the former false
    # negatives and proves this fixture exercises both repaired checks.
    assert torch.equal(
        evidence["post_operator"] + evidence["mlp_down"],
        evidence["post_layer"],
    )
    assert not torch.allclose(
        evidence["post_operator"].float() + evidence["mlp_down"].float(),
        evidence["post_layer"].float(),
        rtol=2e-3,
        atol=2e-5,
    )
    assert torch.equal(
        torch.nn.functional.silu(evidence["gate"]) * evidence["up"],
        evidence["post_swiglu"],
    )
    widened_gate = evidence["gate"].float()
    assert not torch.allclose(
        widened_gate * torch.sigmoid(widened_gate) * evidence["up"].float(),
        evidence["post_swiglu"].float(),
        rtol=2e-3,
        atol=2e-5,
    )

    statistics, row_count = _cached_mlp_nonlinear_statistics(**evidence)

    assert row_count == 1
    assert set(statistics) == {
        "post",
        "positive",
        "negative",
        "tangent",
        "downstream",
    }
    assert all(value.dtype == torch.float64 for value in statistics.values())
    assert all(torch.isfinite(value).all() for value in statistics.values())


@pytest.mark.parametrize(
    ("endpoint", "message"),
    (
        ("post_layer", "Cached MLP writer does not close"),
        ("post_swiglu", "Cached post-SwiGLU evidence does not close"),
    ),
)
def test_cached_mlp_runtime_closure_rejects_corruption(
    endpoint: str,
    message: str,
) -> None:
    evidence = _bf16_runtime_closed_mlp_evidence()
    evidence[endpoint] = evidence[endpoint] + torch.tensor(
        0.125,
        dtype=torch.bfloat16,
    )

    with pytest.raises(Lfm2CachedHeterogeneousCapacityError, match=message):
        _cached_mlp_nonlinear_statistics(**evidence)


def test_cached_capacity_bank_selects_high_capacity_by_trace_pressure(
    tmp_path,
    monkeypatch,
) -> None:
    parent = _parent()
    tokenizer_semantics = _digest("cached-capacity-tokenizer")
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_semantics,
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "parent-cache",
        parent,
        [
            _batch((1, 2, 3, 4), (5,)),
            _batch((6, 7, 8, 9), (10,)),
        ],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    legacy_batches = tuple(cache.iter_batches())
    legacy_session = cache.selective_authenticated_session()
    original_selected_load = (
        trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession.load_selected_tensors
    )

    def load_from_legacy_full_batch(session, index, names):
        batch = legacy_batches[index]
        tensors = {name: batch.tensors[name] for name in names}
        return Lfm2CircuitTraceSelectedBatchPayload(
            batch_index=index,
            input_sha256=batch.input_sha256,
            ordered_tensor_names=tuple(names),
            logical_bytes=sum(
                int(value.numel() * value.element_size()) for value in tensors.values()
            ),
            session_sha256=session.evidence_sha256,
            selection_sha256="0" * 64,
            tensors=tensors,
        )

    with observe_model_execution(parent) as parent_execution:
        try:
            with monkeypatch.context() as legacy_patch:
                legacy_patch.setattr(
                    trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession,
                    "load_selected_tensors",
                    load_from_legacy_full_batch,
                )
                first = build_lfm2_cached_heterogeneous_capacity_option_bank(
                    graph,
                    cache,
                    selective_session=legacy_session,
                    short_conv_internal_widths=(12, 14),
                    attention_head_options=((2, 1, 4), (4, 2, 8)),
                    mlp_intermediate_sizes=(18, 20),
                )
                assert legacy_session.closed is False
        finally:
            legacy_session.close()

        sessions = []
        selections = []
        original_session_factory = cache.selective_authenticated_session

        def tracked_session_factory(_cache):
            session = original_session_factory()
            sessions.append(session)
            return session

        def tracked_selected_load(session, index, names):
            payload = original_selected_load(session, index, names)
            selections.append((index, tuple(names), payload))
            return payload

        def forbidden_full_load(*_args, **_kwargs):
            raise AssertionError(
                "selective capacity discovery entered a full cache load path"
            )

        with monkeypatch.context() as selective_patch:
            selective_patch.setattr(
                trace_cache.Lfm2CircuitTraceCache,
                "selective_authenticated_session",
                tracked_session_factory,
            )
            for name in ("verify", "iter_batches", "load_batch"):
                selective_patch.setattr(
                    trace_cache.Lfm2CircuitTraceCache,
                    name,
                    forbidden_full_load,
                )
            selective_patch.setattr(
                trace_cache,
                "_load_verified_batch",
                forbidden_full_load,
            )
            selective_patch.setattr(
                trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession,
                "load_selected_tensors",
                tracked_selected_load,
            )
            second = build_lfm2_cached_heterogeneous_capacity_option_bank(
                graph,
                cache,
                short_conv_internal_widths=(12, 14),
                attention_head_options=((2, 1, 4), (4, 2, 8)),
                mlp_intermediate_sizes=(18, 20),
            )

    assert parent_execution["module_forward_calls"] == 0
    assert parent_execution["root_forward_calls"] == 0
    assert first == second
    assert first.as_dict() == second.as_dict()
    assert first.evidence_sha256 == second.evidence_sha256
    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert len(selections) == cache.batch_count * len(first.layer_evidence)
    for selection_index, (batch_index, names, payload) in enumerate(selections):
        layer_index = selection_index % len(first.layer_evidence)
        phases = tuple(
            sorted(
                legacy_batches[batch_index].record["capture"]["phase_diagnostics"],
                key=lambda phase: (
                    0 if phase in {"prefill", "cached_prefill"} else 1,
                    0
                    if phase in {"prefill", "cached_prefill"}
                    else int(phase.removeprefix("decode.")),
                ),
            )
        )
        expected_names = tuple(
            sorted(
                {
                    f"{phase}.layer.{layer_index:03d}.{endpoint}"
                    for phase in phases
                    for endpoint in (
                        "residual.pre",
                        "operator.input",
                        "residual.post_operator",
                        "mlp.input",
                        "mlp.gate",
                        "mlp.up",
                        "mlp.post_swiglu",
                        "mlp.down",
                        "residual.post_layer",
                    )
                }
                | {
                    f"{phase}.layer.{len(first.layer_evidence) - 1:03d}."
                    "residual.post_layer"
                    for phase in phases
                }
            )
        )
        assert names == expected_names
        assert names == tuple(sorted(set(names)))
        assert payload.released is True
    assert first.source_graph_fingerprint == graph.fingerprint
    assert first.parent_cache_key == cache.cache_key
    assert len(first.layer_evidence) == 4
    assert len(first.layer_options) == 16
    assert first.as_dict()["selection_contract"] == {
        "cached_parent_traces_only": True,
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "global_exact_budget_solver_selects_layout": True,
        "hardcoded_layer_indices": False,
        "fixed_566_567_production_default": False,
        "mlp_width_risk_depends_on_cached_nonlinear_defect": True,
        "attention_topology_risk_depends_on_cached_causal_geometry": False,
        "factor_writer_risk_depends_on_cached_output_action": False,
        "factor_writer_candidate_site_count": 0,
        "maximum_factor_writer_sites": 1,
        "factor_writer_site_hardcoded": False,
        "dense_physical_weights": True,
        "parameter_sharing": False,
        "quantization": False,
    }

    pins = build_lfm2_heterogeneous_target_pins(
        hidden_size=16,
        vocab_size=64,
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
        target_vocab_evidence_sha256=tokenizer_semantics,
        runtime_evidence_sha256=_digest("cached-capacity-runtime"),
        token_bridge_evidence_sha256=(
            lfm2_identity_tokenizer_evidence_sha256(
                tokenizer_semantics_sha256=tokenizer_semantics,
                vocabulary_size=64,
            )
        ),
    )
    selection = Lfm2HeterogeneousLayoutSelectionInput(
        target_pins=pins,
        layer_options=first.layer_options,
    ).select(
        graph,
        # All-low is 8,040. This exact count adds one high-attention
        # allocation (+1,160) and one high-MLP allocation (+96).
        exact_target_unique_parameters=9_296,
    )
    descriptor = selection.descriptor
    attention_evidence = tuple(
        value for value in first.layer_evidence if value.layer_type == "full_attention"
    )
    expected_high_attention = max(
        attention_evidence,
        key=lambda value: (
            value.operator_delta_energy_fraction,
            -value.layer_index,
        ),
    ).layer_index
    expected_high_mlp = max(
        first.layer_evidence,
        key=lambda value: (
            value.mlp_capacity_risks[0].predicted_defect
            - value.mlp_capacity_risks[-1].predicted_defect,
            -value.layer_index,
        ),
    ).layer_index

    assert descriptor.accounting.unique_parameter_elements == 9_296
    assert [
        layer.layer_index
        for layer in descriptor.layers
        if layer.layer_type == "full_attention" and layer.query_heads == 4
    ] == [expected_high_attention]
    assert [
        layer.layer_index
        for layer in descriptor.layers
        if layer.mlp_intermediate_size == 20
    ] == [expected_high_mlp]

    automatic = build_lfm2_cached_heterogeneous_capacity_option_bank(
        graph,
        cache,
        short_conv_internal_widths=(12, 14),
        attention_head_options=((2, 1, 4), (4, 2, 4)),
        target_hidden_size=16,
        target_vocab_size=64,
        exact_target_unique_parameters=8_040,
        conv_kernel_size=3,
    )
    assert automatic.mlp_option_policy == AUTOMATIC_MLP_OPTION_POLICY
    assert automatic.exact_budget_lattice_sha256 != "0" * 64
    assert all(
        len(evidence.mlp_capacity_risks) > 2
        and evidence.mlp_capacity_risks[0].predicted_defect
        > evidence.mlp_capacity_risks[-1].predicted_defect
        for evidence in automatic.layer_evidence
    )
    assert any(
        high == low + 8
        for evidence in automatic.layer_evidence
        if evidence.layer_type == "full_attention"
        for low in (value.intermediate_size for value in evidence.mlp_capacity_risks)
        for high in (value.intermediate_size for value in evidence.mlp_capacity_risks)
    )
    automatic_selection = Lfm2HeterogeneousLayoutSelectionInput(
        target_pins=pins,
        layer_options=automatic.layer_options,
    ).select(graph, exact_target_unique_parameters=8_040)
    assert automatic_selection.descriptor.accounting.unique_parameter_elements == 8_040
    assert (
        automatic_selection.descriptor.accounting.unique_parameter_elements
        == automatic_selection.descriptor.accounting.logical_parameter_elements
    )
    assert all(
        option.short_conv_writer_rank is None
        and option.mlp_intermediate_size
        < automatic.layer_evidence[layer_index].source_mlp_intermediate_size
        for layer_index, evidence in enumerate(automatic.layer_evidence)
        for option in automatic.layer_options
        if option.source_layer_sha256 == evidence.source_layer_sha256
    )


def test_cached_attention_geometry_risk_streams_causal_qk_kv_context(
    tmp_path,
) -> None:
    parent = _parent()
    parent.config._attn_implementation = "eager"
    tokenizer_semantics = _digest("cached-attention-geometry-tokenizer")
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_semantics,
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "parent-cache",
        parent,
        [
            _batch((1, 2, 3, 4), (5, 6)),
            _batch((7, 8, 9, 10), (11, 12)),
        ],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=2,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=CACHED_VERIFIED_TRACE_MODE,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )
    capture_contract = cache.manifest["capture"]
    assert "decode_trace_mode" not in capture_contract
    assert capture_contract["capture_causal_circuit_evidence"]
    assert capture_contract["attention_complete_kv_history"]
    assert capture_contract["attention_exact_causal_scores_and_probabilities"]

    with observe_model_execution(parent) as parent_execution:
        bank = build_lfm2_cached_heterogeneous_capacity_option_bank(
            graph,
            cache,
            short_conv_internal_widths=(12, 14),
            attention_head_options=((2, 1, 8), (4, 2, 8)),
            mlp_intermediate_sizes=(18, 20),
            attention_geometry_risk_enabled=True,
            attention_geometry_score_query_chunk_size=1,
        )

    assert parent_execution["module_forward_calls"] == 0
    assert parent_execution["root_forward_calls"] == 0
    assert bank.attention_geometry_policy == ATTENTION_GEOMETRY_ENABLED_POLICY
    assert bank.as_dict()["selection_contract"][
        "attention_topology_risk_depends_on_cached_causal_geometry"
    ]
    attention_evidence = tuple(
        evidence
        for evidence in bank.layer_evidence
        if evidence.layer_type == "full_attention"
    )
    assert attention_evidence
    for evidence in attention_evidence:
        assert tuple(
            report.geometry for report in evidence.attention_geometry_risks
        ) == ((2, 1, 8), (4, 2, 8))
        low, full = evidence.attention_geometry_risks
        assert low.fold_method_id == ATTENTION_GEOMETRY_RISK_METHOD
        assert low.causal_query_row_count > 0
        assert low.cached_key_value_row_count >= low.causal_query_row_count
        assert low.predicted_defect > 0.0
        assert full.predicted_defect == 0.0
        assert full.causal_score_tail_fraction == 0.0
        assert full.context_tail_fraction == 0.0
        assert full.key_value_memory_tail_fraction == 0.0

        options = tuple(
            option
            for option in bank.layer_options
            if option.source_layer_sha256 == evidence.source_layer_sha256
            and option.mlp_intermediate_size == 18
        )
        risk = {
            (option.query_heads, option.key_value_heads, option.head_dim): (
                option.structural_risk
            )
            for option in options
        }
        assert risk[(2, 1, 8)] > risk[(4, 2, 8)]

    with pytest.raises(
        Lfm2CachedHeterogeneousCapacityError,
        match="preserve head_dim",
    ):
        build_lfm2_cached_heterogeneous_capacity_option_bank(
            graph,
            cache,
            short_conv_internal_widths=(12, 14),
            attention_head_options=((2, 1, 4), (4, 2, 8)),
            mlp_intermediate_sizes=(18, 20),
            attention_geometry_risk_enabled=True,
        )


def test_production_exact4x_width_policy_is_broad_and_operator_exchangeable() -> None:
    signature = inspect.signature(build_lfm2_cached_heterogeneous_capacity_option_bank)
    assert signature.parameters["mlp_intermediate_sizes"].default is None
    layer_types = (
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
    policy = derive_lfm2_automatic_mlp_width_policy(
        layer_types=layer_types,
        source_widths=(2_560,) * len(layer_types),
        short_conv_internal_widths=(704, 1_024),
        attention_head_options=((12, 6, 64), (16, 8, 64)),
        target_hidden_size=784,
        target_vocab_size=4_096,
        exact_target_unique_parameters=57_420_144,
        conv_kernel_size=3,
    )
    assert policy.policy_id == AUTOMATIC_MLP_OPTION_POLICY
    assert policy.exact_witness_search_candidate_count > 1
    assert policy.exact_witness_search_retained_count == 1
    assert all(len(widths) > 20 for widths in policy.widths_by_layer)
    assert all(set(widths) != {566, 567} for widths in policy.widths_by_layer)
    # At H784 the larger attention geometry costs exactly 256 MLP neurons.
    # The bank explicitly contains both sides of that physical capacity trade.
    assert all(
        any(width + 256 in widths for width in widths)
        for layer_type, widths in zip(
            layer_types,
            policy.widths_by_layer,
            strict=True,
        )
        if layer_type == "full_attention"
    )
    for invalid_attention in (
        ((12, 5, 64), (16, 8, 64)),
        ((12, 6, 63), (16, 8, 64)),
        ((12, 6, 64), (16, 0, 64)),
    ):
        with pytest.raises(Lfm2CachedHeterogeneousCapacityError):
            derive_lfm2_automatic_mlp_width_policy(
                layer_types=layer_types,
                source_widths=(2_560,) * len(layer_types),
                short_conv_internal_widths=(704, 1_024),
                attention_head_options=invalid_attention,
                target_hidden_size=784,
                target_vocab_size=4_096,
                exact_target_unique_parameters=57_420_144,
                conv_kernel_size=3,
            )
    with pytest.raises(Lfm2CachedHeterogeneousCapacityError):
        derive_lfm2_automatic_mlp_width_policy(
            layer_types=layer_types,
            source_widths=(2_560,) * len(layer_types),
            short_conv_internal_widths=(0, 1_024),
            attention_head_options=((12, 6, 64), (16, 8, 64)),
            target_hidden_size=784,
            target_vocab_size=4_096,
            exact_target_unique_parameters=57_420_144,
            conv_kernel_size=3,
        )


def test_cached_short_conv_writer_action_risk_is_rank_monotone() -> None:
    accumulator = _ShortConvWriterActionAccumulator(layer_index=3)
    output = torch.diag(torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float32))
    accumulator.add_phase(output, phase="prefill")

    risks = accumulator.risks((1, 2, 3))

    assert tuple(value.writer_rank for value in risks) == (1, 2, 3)
    assert all(
        value.fold_method_id == SHORT_CONV_WRITER_ACTION_RISK_METHOD for value in risks
    )
    assert risks[0].predicted_defect > risks[1].predicted_defect
    assert risks[1].predicted_defect > risks[2].predicted_defect > 0.0
    assert all(value.sampled_output_row_count == 4 for value in risks)
    assert all(value.sampling_uncertainty_fraction == 0.0 for value in risks)


def test_cached_short_conv_writer_covariance_is_streamed_and_bounded() -> None:
    accumulator = _ShortConvWriterActionAccumulator(layer_index=5)
    generator = torch.Generator().manual_seed(7)
    for phase_index in range(16):
        accumulator.add_phase(
            torch.randn(128, 32, generator=generator, dtype=torch.bfloat16),
            phase=f"cached-{phase_index}",
        )

    assert accumulator.sampled_output_row_count == 2_048
    assert tuple(accumulator.feature_gram.shape) == (32, 32)
    assert not hasattr(accumulator, "rows")
    risks = accumulator.risks((4, 16))
    assert all(value.sampled_output_row_count == 2_048 for value in risks)
    assert all(value.sampling_uncertainty_fraction == 0.0 for value in risks)


def test_undersampled_writer_rank_cannot_claim_zero_defect() -> None:
    accumulator = _ShortConvWriterActionAccumulator(layer_index=7)
    output = torch.zeros(2, 8, dtype=torch.float32)
    output[0, 0] = 1.0
    output[1, 1] = 1.0
    accumulator.add_phase(output, phase="undersampled")

    low, high = accumulator.risks((2, 4))

    assert low.output_action_tail_fraction == 0.0
    assert high.output_action_tail_fraction == 0.0
    assert low.sampling_uncertainty_fraction > high.sampling_uncertainty_fraction > 0.0
    assert low.predicted_defect > high.predicted_defect > 0.0
    assert low.observed_effective_rank == pytest.approx(2.0)


def test_cached_option_bank_generates_output_action_aware_factor_writers(
    tmp_path,
) -> None:
    parent = _parent()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=_digest("writer-risk-tokenizer"),
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "writer-risk-cache",
        parent,
        [_batch((1, 2, 3, 4), (5,))],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )

    bank = build_lfm2_cached_heterogeneous_capacity_option_bank(
        graph,
        cache,
        short_conv_internal_widths=(12, 14),
        short_conv_writer_ranks=(2, 4),
        attention_head_options=((2, 1, 4), (4, 2, 8)),
        mlp_intermediate_sizes=(18, 20),
        target_hidden_size=16,
    )

    assert bank.as_dict()["selection_contract"][
        "factor_writer_risk_depends_on_cached_output_action"
    ]
    assert (
        bank.as_dict()["selection_contract"]["factor_writer_candidate_site_count"] == 1
    )
    conv_evidence = tuple(
        value for value in bank.layer_evidence if value.layer_type == "conv"
    )
    assert conv_evidence
    for evidence in conv_evidence:
        assert tuple(
            value.writer_rank for value in evidence.short_conv_writer_risks
        ) == (2, 4)
        low, high = evidence.short_conv_writer_risks
        assert low.predicted_defect >= high.predicted_defect
        options = tuple(
            value
            for value in bank.layer_options
            if value.source_layer_sha256 == evidence.source_layer_sha256
            and value.mlp_intermediate_size == 18
            and value.short_conv_internal_width == 12
        )
        observed_ranks = {value.short_conv_writer_rank for value in options}
        if any(value.short_conv_writer_rank is not None for value in options):
            assert observed_ranks == {None, 2, 4}
        else:
            assert observed_ranks == {None}
        option_risk = {
            value.short_conv_writer_rank: value.structural_risk for value in options
        }
        if 2 in option_risk:
            assert option_risk[None] <= option_risk[4] <= option_risk[2]

    factor_option = next(
        value
        for value in bank.layer_options
        if value.short_conv_writer_rank is not None
    )
    other_evidence = next(
        value
        for value in conv_evidence
        if value.source_layer_sha256 != factor_option.source_layer_sha256
    )
    forged_second_site = build_lfm2_short_conv_layer_option(
        source_layer_sha256=other_evidence.source_layer_sha256,
        mlp_intermediate_size=factor_option.mlp_intermediate_size,
        internal_width=int(factor_option.short_conv_internal_width),
        writer_rank=int(factor_option.short_conv_writer_rank),
        selection_evidence_sha256=_digest("forged-second-factor-site"),
        structural_risk=factor_option.structural_risk,
    )
    forged_values = {
        "source_graph_fingerprint": bank.source_graph_fingerprint,
        "parent_cache_key": bank.parent_cache_key,
        "mlp_option_policy": bank.mlp_option_policy,
        "attention_geometry_policy": bank.attention_geometry_policy,
        "exact_budget_lattice_sha256": bank.exact_budget_lattice_sha256,
        "layer_evidence": bank.layer_evidence,
        "layer_options": (*bank.layer_options, forged_second_site),
    }
    provisional = object.__new__(Lfm2CachedHeterogeneousCapacityOptionBank)
    for name, value in forged_values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    forged_sha256 = canonical_sha256(provisional._body())
    with pytest.raises(
        Lfm2CachedHeterogeneousCapacityError,
        match="only one dynamically selected source layer",
    ):
        Lfm2CachedHeterogeneousCapacityOptionBank(
            **forged_values,
            evidence_sha256=forged_sha256,
        )

    unsupported_rank = build_lfm2_short_conv_layer_option(
        source_layer_sha256=factor_option.source_layer_sha256,
        mlp_intermediate_size=factor_option.mlp_intermediate_size,
        internal_width=int(factor_option.short_conv_internal_width),
        writer_rank=3,
        selection_evidence_sha256=_digest("unsupported-writer-risk"),
        structural_risk=factor_option.structural_risk,
    )
    unsupported_values = {
        **forged_values,
        "layer_options": (*bank.layer_options, unsupported_rank),
    }
    provisional = object.__new__(Lfm2CachedHeterogeneousCapacityOptionBank)
    for name, value in unsupported_values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    unsupported_sha256 = canonical_sha256(provisional._body())
    with pytest.raises(
        Lfm2CachedHeterogeneousCapacityError,
        match="lacks matching cached output-action risk evidence",
    ):
        Lfm2CachedHeterogeneousCapacityOptionBank(
            **unsupported_values,
            evidence_sha256=unsupported_sha256,
        )


def test_exact4x_writer_ranks_are_discovered_from_integer_lattice() -> None:
    layer_types = (
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
    with pytest.raises(
        Lfm2CachedHeterogeneousCapacityError,
        match="no dense compact",
    ):
        derive_lfm2_automatic_mlp_width_policy(
            layer_types=layer_types,
            source_widths=(2_560,) * len(layer_types),
            short_conv_internal_widths=(704, 1_024),
            attention_head_options=((12, 6, 64), (16, 8, 64)),
            target_hidden_size=816,
            target_vocab_size=4_096,
            exact_target_unique_parameters=53_552_368,
            conv_kernel_size=3,
        )

    policy = derive_lfm2_automatic_mlp_width_policy(
        layer_types=layer_types,
        source_widths=(2_560,) * len(layer_types),
        short_conv_internal_widths=(704, 1_024),
        attention_head_options=((12, 6, 64), (16, 8, 64)),
        target_hidden_size=816,
        target_vocab_size=4_096,
        exact_target_unique_parameters=53_552_368,
        conv_kernel_size=3,
        discover_exact_short_conv_writer_ranks=True,
    )

    assert policy.exact_witness_factor_writer_count == 1
    assert policy.exact_witness_search_candidate_count > (
        policy.exact_witness_search_retained_count
    )
    assert policy.exact_witness_search_retained_count <= 1 + len(
        policy.short_conv_writer_ranks
    )
    assert (
        1
        <= len(policy.short_conv_writer_ranks)
        <= (MAX_DYNAMIC_SHORT_CONV_WRITER_RANKS)
    )
    assert policy.short_conv_writer_ranks == tuple(
        sorted(set(policy.short_conv_writer_ranks))
    )
    assert all(len(widths) >= 3 for widths in policy.widths_by_layer)
    for rank in policy.short_conv_writer_ranks:
        assert any(
            rank * (816 + internal) < 816 * internal
            for internal in (704, 1_024)
            if rank < min(816, internal)
        )
