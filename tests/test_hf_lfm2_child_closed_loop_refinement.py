from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_child_closed_loop_refinement as refinement_module  # noqa: E402
from neural_forge.hf_lfm2_child_closed_loop_refinement import (  # noqa: E402
    Lfm2NormMaterializer,
    build_lfm2_final_norm_pairs,
    build_lfm2_mlp_cases_from_candidate_post_operator,
    build_lfm2_operator_producer_cases,
    compute_lfm2_candidate_post_layer,
    compute_lfm2_candidate_post_operator,
    extract_lfm2_actual_child_contexts,
    pair_lfm2_parent_child_replay,
    restore_lfm2_refinement_layer,
    snapshot_lfm2_refinement_layer,
    split_lfm2_replay_batches,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CACHED_VERIFIED_TRACE_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
    capture_lfm2_child_replay,
)


def test_legacy_cached_mode_requires_numerical_not_top1_verification() -> None:
    capture = {
        "decode_trace_mode": None,
        "all_cached_logits_verified": True,
        "all_cached_top1_verified": False,
        "cached_full_prefix_verifications": [
            {
                "numerically_close": True,
                "top1_agreement": False,
            }
        ],
    }
    batch = SimpleNamespace(record={"capture": capture})

    assert (
        refinement_module._verified_parent_decode_trace_mode(batch)
        == CACHED_VERIFIED_TRACE_MODE
    )

    capture["cached_full_prefix_verifications"][0][
        "numerically_close"
    ] = False
    with pytest.raises(
        refinement_module.Lfm2ChildClosedLoopRefinementError,
        match="does not authenticate",
    ):
        refinement_module._verified_parent_decode_trace_mode(batch)


def _model(*, seed: int):
    config = transformers.Lfm2Config(
        vocab_size=37,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config)


def _batch(prompt, forced=(9,)):
    input_ids = torch.tensor([list(prompt)], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "forced_decode_ids": torch.tensor([list(forced)], dtype=torch.long),
    }


@pytest.fixture(scope="module")
def traced_pair(tmp_path_factory):
    parent = _model(seed=801)
    child = _model(seed=802)
    batches = [
        _batch((1, 2, 3, 4)),
        _batch((5, 6, 7, 8), forced=(10,)),
    ]
    config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=2,
        max_decode_steps=2,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )
    parent_cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("closed-loop-parent"),
        parent,
        batches,
        config=config,
        input_device="cpu",
    )
    child_replay = capture_lfm2_child_replay(
        child,
        batches,
        selected_layer_indices=[1],
        config=config,
        input_device="cpu",
    )
    with torch.no_grad():
        attention = child.model.layers[1].self_attn
        attention.q_proj.weight.add_(0.125)
        attention.k_proj.weight.sub_(0.075)
        attention.v_proj.weight.add_(0.05)
    producer_replay = capture_lfm2_child_replay(
        child,
        child_replay.iter_batches(),
        selected_layer_indices=[1],
        config=config,
        input_device="cpu",
    )
    assert producer_replay.replay_sha256 != child_replay.replay_sha256

    def forbidden_forward(self, *_args, **_kwargs):
        raise AssertionError("forward-free orchestration executed a model")

    parent.forward = MethodType(forbidden_forward, parent)
    child.forward = MethodType(forbidden_forward, child)
    return parent, child, parent_cache, child_replay, producer_replay


def _identity_coordinate_map(value):
    return value.clone()


def _offset_norm_fitter(offset, requests):
    def fit(request):
        requests.append(request)
        return Lfm2NormMaterializer(
            role=request.role,
            materialize_callback=lambda value: value + offset,
            report={
                "offset": offset,
                "fit_case_count": len(request.fit_inputs),
                "validation_case_count": len(request.validation_inputs),
            },
        )

    return fit


def _origin_tensor(paired, origin, layer_index, suffix, *, child):
    batch = paired.load_batch(origin.batch_index)
    tensors = batch.child.tensors if child else batch.parent.tensors
    return tensors[f"{origin.phase}.layer.{layer_index:03d}.{suffix}"]


def test_operator_cases_use_live_child_inputs_and_hold_out_whole_batches(
    traced_pair,
) -> None:
    _parent, _child, parent_cache, child_replay, _producer_replay = traced_pair
    paired = pair_lfm2_parent_child_replay(parent_cache, child_replay)
    default_split = split_lfm2_replay_batches(paired)
    split = split_lfm2_replay_batches(
        paired,
        validation_batch_indices=[1],
    )
    requests = []
    cases = build_lfm2_operator_producer_cases(
        split,
        layer_index=1,
        operator_kind="full_attention",
        coordinate_map_callback=_identity_coordinate_map,
        norm_fit_callback=_offset_norm_fitter(100.0, requests),
    )

    assert default_split.fit_batch_indices == (0, 1)
    assert default_split.validation_batch_indices == ()
    assert split.fit_batch_indices == (0,)
    assert split.validation_batch_indices == (1,)
    assert {origin.batch_index for origin in cases.fit_origins} == {0}
    assert {origin.batch_index for origin in cases.validation_origins} == {1}
    assert len(cases.fit_cases) == len(cases.validation_cases) == 2
    assert len(requests) == 1
    request = requests[0]
    assert len(request.fit_inputs) == 2
    assert len(request.validation_inputs) == 2
    # The authenticated parent capture is full-prefix canonical.  Its decode
    # phase is therefore an independent sampled causal span, not a new token
    # to append to the prefill span.
    assert cases.fit_origins[1].phase == "decode.000"
    assert cases.fit_cases[1].parent_key.shape[2] == 1
    assert cases.fit_cases[1].target_key_value_input.shape[1] == 1

    first_case = cases.fit_cases[0]
    first_origin = cases.fit_origins[0]
    child_residual = _origin_tensor(
        paired,
        first_origin,
        1,
        "residual.pre",
        child=True,
    )
    parent_residual = _origin_tensor(
        paired,
        first_origin,
        1,
        "residual.pre",
        child=False,
    )
    parent_endpoint = _origin_tensor(
        paired,
        first_origin,
        1,
        "residual.post_operator",
        child=False,
    )
    torch.testing.assert_close(
        first_case.target_pre_attention_residual,
        child_residual,
    )
    torch.testing.assert_close(
        first_case.target_query_input,
        child_residual + 100.0,
    )
    torch.testing.assert_close(
        first_case.parent_post_attention_residual_target_space,
        parent_endpoint,
    )
    assert not torch.equal(
        first_case.target_pre_attention_residual,
        parent_residual,
    )
    assert not torch.equal(
        first_case.target_query_input,
        _origin_tensor(
            paired,
            first_origin,
            1,
            "operator.input",
            child=False,
        ),
    )


def test_context_candidate_mlp_and_final_norm_stay_live_child_aligned(
    traced_pair,
) -> None:
    _parent, child, parent_cache, child_replay, producer_replay = traced_pair
    initial_paired = pair_lfm2_parent_child_replay(
        parent_cache,
        child_replay,
    )
    initial_split = split_lfm2_replay_batches(
        initial_paired,
        validation_batch_indices=[1],
    )
    operator_cases = build_lfm2_operator_producer_cases(
        initial_split,
        layer_index=1,
        operator_kind="full_attention",
        coordinate_map_callback=_identity_coordinate_map,
        norm_fit_callback=_offset_norm_fitter(10.0, []),
    )
    paired = pair_lfm2_parent_child_replay(parent_cache, producer_replay)
    split = split_lfm2_replay_batches(
        paired,
        validation_batch_indices=[1],
    )
    contexts = extract_lfm2_actual_child_contexts(
        operator_cases,
        paired,
    )

    assert [origin.prefix_length for origin in contexts.fit_origins] == [4, 5]
    assert [origin.sample_count for origin in contexts.fit_origins] == [2, 1]
    for context, origin in zip(
        contexts.fit_contexts,
        contexts.fit_origins,
    ):
        expected = _origin_tensor(
            paired,
            origin,
            1,
            "attention.context",
            child=True,
        )
        assert torch.equal(context, expected)
        assert origin.prefix_length > origin.sample_count

    zero_output = torch.zeros_like(child.model.layers[1].self_attn.out_proj.weight)
    candidate_operator = compute_lfm2_candidate_post_operator(
        operator_cases,
        contexts,
        output_weight=zero_output,
    )
    for value, case, current in zip(
        candidate_operator.fit_values,
        operator_cases.fit_cases,
        contexts.fit_current_post_operator,
    ):
        torch.testing.assert_close(
            value,
            case.target_pre_attention_residual,
            check_dtype=False,
        )
        assert not torch.allclose(value.to(current.dtype), current)
    assert candidate_operator.comparison.candidate_fit.finite is True
    assert candidate_operator.comparison.current_fit.finite is True
    assert candidate_operator.comparison.candidate_validation is not None

    mlp_requests = []
    mlp_cases = build_lfm2_mlp_cases_from_candidate_post_operator(
        split,
        operator_cases,
        candidate_operator,
        coordinate_map_callback=_identity_coordinate_map,
        norm_fit_callback=_offset_norm_fitter(20.0, mlp_requests),
    )
    assert len(mlp_requests) == 1
    assert len(mlp_requests[0].fit_inputs) == 2
    assert len(mlp_requests[0].validation_inputs) == 2
    first_mlp = mlp_cases.fit_cases[0]
    first_candidate = candidate_operator.fit_values[0].reshape(-1, 8)
    torch.testing.assert_close(
        first_mlp.target_pre_mlp_residual,
        first_candidate,
        check_dtype=False,
    )
    torch.testing.assert_close(
        first_mlp.target_input,
        first_candidate + 20.0,
        check_dtype=False,
    )
    current_post_operator = contexts.fit_current_post_operator[0].reshape(-1, 8)
    assert not torch.allclose(
        first_mlp.target_pre_mlp_residual.to(current_post_operator.dtype),
        current_post_operator,
    )

    target_mlp = child.model.layers[1].feed_forward
    candidate_layer = compute_lfm2_candidate_post_layer(
        mlp_cases,
        gate_weight=target_mlp.w1.weight,
        up_weight=target_mlp.w3.weight,
        down_weight=target_mlp.w2.weight,
    )
    assert candidate_layer.comparison.candidate_fit.finite is True
    assert candidate_layer.comparison.current_fit.finite is True
    assert candidate_layer.comparison.candidate_validation is not None

    final_pairs = build_lfm2_final_norm_pairs(
        split,
        final_layer_index=1,
        coordinate_map_callback=_identity_coordinate_map,
        parent_final_norm_callback=lambda value: value * 3.0,
    )
    assert len(final_pairs.fit_inputs) == 2
    assert len(final_pairs.validation_inputs) == 2
    first_origin = final_pairs.fit_origins[0]
    expected_child = _origin_tensor(
        paired,
        first_origin,
        1,
        "residual.post_layer",
        child=True,
    ).reshape(-1, 8)
    expected_parent = _origin_tensor(
        paired,
        first_origin,
        1,
        "residual.post_layer",
        child=False,
    ).reshape(-1, 8)
    assert torch.equal(final_pairs.fit_inputs[0], expected_child)
    torch.testing.assert_close(
        final_pairs.fit_targets[0],
        expected_parent * 3.0,
    )
    assert not torch.equal(
        final_pairs.fit_inputs[0],
        expected_parent,
    )


def test_layer_snapshot_restores_exact_bytes_and_runtime_contract() -> None:
    model = _model(seed=850)
    layer = model.model.layers[1]
    with torch.no_grad():
        for index, parameter in enumerate(layer.parameters()):
            parameter.grad = torch.full_like(
                parameter,
                float(index + 1),
            )
    state_before = {
        name: value.detach().cpu().clone() for name, value in layer.state_dict().items()
    }
    parameter_contract = {
        name: (
            id(parameter),
            tuple(parameter.shape),
            parameter.dtype,
            id(parameter.grad),
            parameter.grad.detach().cpu().clone(),
        )
        for name, parameter in layer.named_parameters()
    }
    snapshot = snapshot_lfm2_refinement_layer(layer)

    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.add_(1.25)
            parameter.grad.mul_(3.0)
    report = restore_lfm2_refinement_layer(layer, snapshot)

    assert report.exact is True
    assert report.parameter_identities_preserved is True
    assert report.parameter_shapes_preserved is True
    assert report.parameter_dtypes_preserved is True
    assert report.gradient_state_preserved is True
    for name, value in layer.state_dict().items():
        assert torch.equal(value.detach().cpu(), state_before[name])
    for name, parameter in layer.named_parameters():
        expected = parameter_contract[name]
        assert id(parameter) == expected[0]
        assert tuple(parameter.shape) == expected[1]
        assert parameter.dtype == expected[2]
        assert id(parameter.grad) == expected[3]
        assert torch.equal(parameter.grad.detach().cpu(), expected[4])
