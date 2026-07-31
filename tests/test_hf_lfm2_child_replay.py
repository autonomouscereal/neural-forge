from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2ChildEvidenceProfile,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCaptureError,
    capture_lfm2_child_replay,
)


def _model(*, seed: int = 701):
    config = transformers.Lfm2Config(
        vocab_size=31,
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


def _batch(*, prompt=(1, 2, 3, 4), forced=(5, 6)):
    input_ids = torch.tensor([list(prompt)], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "forced_decode_ids": torch.tensor([list(forced)], dtype=torch.long),
    }


def _canonical_config(*, sample_last_tokens: int = 2):
    return Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=sample_last_tokens,
        max_decode_steps=4,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )


def _module_modes(model):
    return {name: bool(module.training) for name, module in model.named_modules()}


def _instrument_forwards(model):
    calls: list[dict[str, object]] = []

    def observed(_module, _args, kwargs):
        input_ids = kwargs["input_ids"]
        calls.append(
            {
                "input_length": int(input_ids.shape[1]),
                "use_cache": kwargs.get("use_cache"),
                "has_past": kwargs.get("past_key_values") is not None,
                "training": bool(model.training),
                "grad_enabled": torch.is_grad_enabled(),
                "attention_mask": kwargs["attention_mask"].detach().cpu().clone(),
                "position_ids": (
                    None
                    if kwargs.get("position_ids") is None
                    else kwargs["position_ids"].detach().cpu().clone()
                ),
            }
        )

    model.register_forward_pre_hook(observed, with_kwargs=True)
    return calls


def _instrument_trace_forwards(monkeypatch):
    calls: list[dict[str, object]] = []
    original = trace_cache._forward

    def observed(model, **kwargs):
        input_ids = kwargs["input_ids"]
        calls.append(
            {
                "input_length": int(input_ids.shape[1]),
                "use_cache": kwargs["use_cache"],
                "has_past": kwargs["past_key_values"] is not None,
                "training": bool(model.training),
                "grad_enabled": torch.is_grad_enabled(),
                "attention_mask": kwargs["attention_mask"].detach().cpu().clone(),
                "position_ids": (
                    None
                    if kwargs["position_ids"] is None
                    else kwargs["position_ids"].detach().cpu().clone()
                ),
            }
        )
        return original(model, **kwargs)

    monkeypatch.setattr(trace_cache, "_forward", observed)
    return calls


def _captured_layer_indices(tensors):
    result: set[int] = set()
    for name in tensors:
        if ".layer." in name:
            result.add(int(name.split(".layer.", 1)[1].split(".", 1)[0]))
    return result


def test_child_replay_is_exact_selected_memory_only_and_deterministic(
    monkeypatch,
) -> None:
    parent = _model(seed=710)
    child = _model(seed=711)
    parent_calls = _instrument_forwards(parent)
    child_calls = _instrument_forwards(child)
    child.train()
    child.model.layers[0].eval()
    modes_before = _module_modes(child)
    source = _batch()

    def forbidden_disk_path(*_args, **_kwargs):
        raise AssertionError("in-memory child replay touched cache persistence")

    monkeypatch.setattr(
        trace_cache,
        "_write_canonical_json",
        forbidden_disk_path,
    )
    monkeypatch.setattr(trace_cache, "_publish_batch", forbidden_disk_path)

    first = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[1],
        config=_canonical_config(),
        input_device="cpu",
    )
    replayed = capture_lfm2_child_replay(
        child,
        first.iter_batches(),
        selected_layer_indices=[1],
        config=_canonical_config(),
        input_device="cpu",
    )

    assert "tokenizer" not in inspect.signature(capture_lfm2_child_replay).parameters
    assert "parent_model" not in inspect.signature(capture_lfm2_child_replay).parameters
    assert parent_calls == []
    assert [call["input_length"] for call in child_calls] == [
        4,
        5,
        6,
        4,
        5,
        6,
    ]
    assert all(call["use_cache"] is False for call in child_calls)
    assert all(call["has_past"] is False for call in child_calls)
    assert all(call["training"] is False for call in child_calls)
    assert all(call["grad_enabled"] is False for call in child_calls)
    assert _module_modes(child) == modes_before

    assert first.replay_sha256 == replayed.replay_sha256
    assert first.selected_layer_indices == (1,)
    assert len(first) == 1
    captured = first.load_batch(0)
    assert captured.phase_order == (
        "prefill",
        "decode.000",
        "decode.001",
    )
    assert torch.equal(captured.input_ids, source["input_ids"])
    assert torch.equal(captured.attention_mask, source["attention_mask"])
    assert torch.equal(
        captured.forced_decode_ids,
        source["forced_decode_ids"],
    )
    assert _captured_layer_indices(captured.tensors) == {1}
    assert set(captured.capture["phase_diagnostics"]) == set(captured.phase_order)
    assert [
        tuple(captured.capture["phase_diagnostics"][phase])
        for phase in captured.phase_order
    ] == [("layer.001",), ("layer.001",), ("layer.001",)]
    assert captured.record["phase_rows"] == [
        {
            "forced_decode_step": None,
            "phase": "prefill",
            "prefix_length": 4,
            "sample_count": 2,
            "sample_start": 2,
        },
        {
            "forced_decode_step": 0,
            "phase": "decode.000",
            "prefix_length": 5,
            "sample_count": 1,
            "sample_start": 4,
        },
        {
            "forced_decode_step": 1,
            "phase": "decode.001",
            "prefix_length": 6,
            "sample_count": 1,
            "sample_start": 5,
        },
    ]
    metadata = first.metadata
    assert metadata["forward_counts"] == {"child": 3, "parent": 0}
    assert metadata["runtime"] == {
        "tokenizer_accessed": False,
        "parent_model_supplied": False,
        "parent_forward_count": 0,
        "disk_artifacts_written": 0,
        "intermediate_cache": "memory_only",
        "child_state_preserved": True,
    }


def test_point_specific_profile_filters_only_after_full_layer_validation() -> None:
    child = _model(seed=712)
    profile = Lfm2ChildEvidenceProfile(
        profile_role="layer.post_consumer",
        layer_suffixes=(
            (
                0,
                (
                    "residual.post_layer",
                    "residual.post_operator",
                    "residual.pre",
                    "short_conv.context",
                ),
            ),
            (
                1,
                (
                    "attention.context",
                    "residual.post_layer",
                    "residual.post_operator",
                    "residual.pre",
                ),
            ),
        ),
    )
    replay = capture_lfm2_child_replay(
        child,
        [_batch()],
        selected_layer_indices=[0, 1],
        config=_canonical_config(),
        input_device="cpu",
        child_evidence_profile=profile,
    )

    assert replay.child_evidence_profile == profile
    batch = replay.load_batch(0)
    assert batch.child_evidence_profile == profile
    assert batch.record["child_evidence_profile_sha256"] == profile.profile_sha256
    assert replay.metadata["selection"]["child_evidence_profile"] == (
        profile.as_dict()
    )
    expected_inputs = {
        "input.input_ids",
        "input.attention_mask",
        "input.forced_decode_ids",
    }
    assert expected_inputs <= set(batch.tensors)
    for phase in batch.phase_order:
        expected_phase_names = {
            f"{phase}.layer.{layer_index:03d}.{suffix}"
            for layer_index, suffixes in profile.layer_suffixes
            for suffix in suffixes
        }
        observed_phase_names = {
            name
            for name in batch.tensors
            if name.startswith(f"{phase}.")
        }
        assert observed_phase_names == expected_phase_names
        for layer_index in profile.selected_layer_indices:
            diagnostic = batch.capture["phase_diagnostics"][phase][
                f"layer.{layer_index:03d}"
            ]
            assert diagnostic["post_swiglu_verified"] is True
            assert diagnostic["operator_residual_verified"] is True
            assert diagnostic["layer_residual_verified"] is True
            assert diagnostic[
                "full_layer_algebra_validated_before_filter"
            ] is True
    assert not any(name.endswith(".logits") for name in batch.tensors)
    assert not any(name.endswith(".output.lm_head_input") for name in batch.tensors)
    assert not any(name.endswith(".forced_token_ids") for name in batch.tensors)
    assert batch.capture["retained_capture_bytes"] == batch.capture["capture_bytes"]
    assert batch.capture["transient_validation_peak_bytes"] > 0
    assert batch.capture["estimated_peak_capture_bytes"] == (
        batch.capture["retained_capture_bytes"]
        + batch.capture["transient_validation_peak_bytes"]
    )


def test_point_specific_profile_and_batch_tampering_fail_closed() -> None:
    profile = Lfm2ChildEvidenceProfile(
        profile_role="layer.pre_producer",
        layer_suffixes=(
            (1, ("residual.post_layer", "residual.pre")),
        ),
    )
    body = profile.as_dict()
    body["layers"][0]["retained_suffixes"].append("operator.output")
    with pytest.raises(ValueError, match="inventory|seal|SHA-256|changed"):
        Lfm2ChildEvidenceProfile.from_dict(body)

    child = _model(seed=713)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="differs from selected_layer_indices",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0],
            config=_canonical_config(),
            input_device="cpu",
            child_evidence_profile=profile,
        )


def test_output_profile_retains_logits_and_lm_head_but_not_forced_duplicates() -> None:
    child = _model(seed=714)
    profile = Lfm2ChildEvidenceProfile(
        profile_role="output_interface.pre_output_interface",
        layer_suffixes=(
            (1, ("residual.post_layer", "residual.pre")),
        ),
        retain_logits=True,
        retain_lm_head_input=True,
    )
    batch = capture_lfm2_child_replay(
        child,
        [_batch()],
        selected_layer_indices=[1],
        config=_canonical_config(),
        input_device="cpu",
        child_evidence_profile=profile,
    ).load_batch(0)

    for phase in batch.phase_order:
        assert f"{phase}.logits" in batch.tensors
        assert f"{phase}.output.lm_head_input" in batch.tensors
        assert f"{phase}.forced_token_ids" not in batch.tensors


def test_point_profile_rejects_unprofiled_causal_output_globals() -> None:
    child = _model(seed=715)
    profile = Lfm2ChildEvidenceProfile(
        profile_role="layer.pre_producer",
        layer_suffixes=((1, ("residual.post_layer", "residual.pre")),),
    )
    causal_config = Lfm2CircuitTraceCaptureConfig(
        capture_causal_circuit_evidence=True,
    )
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="do not yet authenticate causal-circuit output-boundary globals",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[1],
            config=causal_config,
            input_device="cpu",
            child_evidence_profile=profile,
        )


def test_child_replay_attention_context_uses_full_prefix_not_sampled_tail() -> None:
    child = _model(seed=720)
    calls = _instrument_forwards(child)
    config = _canonical_config(sample_last_tokens=2)
    full = capture_lfm2_child_replay(
        child,
        [_batch(prompt=(1, 2, 3, 4, 5, 6), forced=(7,))],
        selected_layer_indices=[1],
        config=config,
        input_device="cpu",
    )
    tail_only = capture_lfm2_child_replay(
        child,
        [_batch(prompt=(4, 5, 6), forced=(7,))],
        selected_layer_indices=[1],
        config=config,
        input_device="cpu",
    )

    assert [call["input_length"] for call in calls] == [6, 7, 3, 4]
    assert all(call["use_cache"] is False for call in calls)
    full_context = full.load_batch(0).tensors["prefill.layer.001.attention.context"]
    tail_context = tail_only.load_batch(0).tensors[
        "prefill.layer.001.attention.context"
    ]
    assert full_context.shape == tail_context.shape == (1, 2, 2, 4)
    assert not torch.allclose(full_context, tail_context)
    assert full.load_batch(0).record["phase_rows"][0] == {
        "forced_decode_step": None,
        "phase": "prefill",
        "prefix_length": 6,
        "sample_count": 2,
        "sample_start": 4,
    }


def test_selected_downstream_layer_observes_live_upstream_child_state() -> None:
    child = _model(seed=730)
    config = _canonical_config()
    layer_zero_calls = 0

    def count_layer_zero(_module, _args, _output):
        nonlocal layer_zero_calls
        layer_zero_calls += 1

    handle = child.model.layers[0].register_forward_hook(count_layer_zero)
    try:
        before = capture_lfm2_child_replay(
            child,
            [_batch(forced=(7,))],
            selected_layer_indices=[1],
            config=config,
            input_device="cpu",
        )
        with torch.no_grad():
            child.model.layers[0].feed_forward.w2.weight.add_(0.75)
        after = capture_lfm2_child_replay(
            child,
            [_batch(forced=(7,))],
            selected_layer_indices=[1],
            config=config,
            input_device="cpu",
        )
    finally:
        handle.remove()

    assert layer_zero_calls == 4
    before_residual = before.load_batch(0).tensors["prefill.layer.001.residual.pre"]
    after_residual = after.load_batch(0).tensors["prefill.layer.001.residual.pre"]
    assert not torch.allclose(before_residual, after_residual)
    assert _captured_layer_indices(before.load_batch(0).tensors) == {1}
    assert _captured_layer_indices(after.load_batch(0).tensors) == {1}


def test_single_pass_replay_matches_every_canonical_phase_endpoint(
    monkeypatch,
) -> None:
    child = _model(seed=740)
    calls = _instrument_trace_forwards(monkeypatch)
    config = _canonical_config(sample_last_tokens=3)
    source = _batch(prompt=(1, 2, 3, 4, 5), forced=(6, 7, 8))
    source["position_ids"] = torch.tensor([[10, 11, 12, 13, 14]])

    canonical = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
    )
    single_pass = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
        replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        causal_equivalence_reference=canonical,
    )

    assert [call["input_length"] for call in calls] == [5, 6, 7, 8, 8]
    assert all(call["use_cache"] is False for call in calls)
    assert all(call["has_past"] is False for call in calls)
    trajectory_call = calls[-1]
    assert torch.equal(
        trajectory_call["attention_mask"],
        torch.ones((1, 8), dtype=torch.long),
    )
    assert torch.equal(
        trajectory_call["position_ids"],
        torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]),
    )

    expected = canonical.load_batch(0)
    actual = single_pass.load_batch(0)
    assert actual.phase_order == expected.phase_order == (
        "prefill",
        "decode.000",
        "decode.001",
        "decode.002",
    )
    assert set(actual.tensors) == set(expected.tensors)
    for name in sorted(expected.tensors):
        left = expected.tensors[name]
        right = actual.tensors[name]
        assert left.shape == right.shape, name
        assert left.dtype == right.dtype, name
        if left.is_floating_point():
            torch.testing.assert_close(
                right,
                left,
                rtol=config.logit_rtol,
                atol=config.logit_atol,
                msg=name,
            )
        else:
            assert torch.equal(right, left), name
        if name.endswith(".logits"):
            assert torch.equal(
                torch.argmax(right, dim=-1),
                torch.argmax(left, dim=-1),
            ), name

    names = set(actual.tensors)
    required_suffixes = {
        "residual.pre",
        "operator.input",
        "operator.output",
        "residual.post_operator",
        "mlp.input",
        "mlp.gate",
        "mlp.up",
        "mlp.post_swiglu",
        "mlp.down",
        "residual.post_layer",
    }
    for phase in actual.phase_order:
        for layer_index in (0, 1):
            prefix = f"{phase}.layer.{layer_index:03d}."
            assert {prefix + suffix for suffix in required_suffixes} <= names
        assert {
            f"{phase}.layer.000.short_conv.b",
            f"{phase}.layer.000.short_conv.c",
            f"{phase}.layer.000.short_conv.x",
            f"{phase}.layer.000.short_conv.bx",
            f"{phase}.layer.000.short_conv.context",
            f"{phase}.layer.000.short_conv.o",
            f"{phase}.layer.001.attention.q_linear",
            f"{phase}.layer.001.attention.q",
            f"{phase}.layer.001.attention.k_linear",
            f"{phase}.layer.001.attention.k",
            f"{phase}.layer.001.attention.v",
            f"{phase}.layer.001.attention.context",
            f"{phase}.layer.001.attention.o",
            f"{phase}.logits",
            f"{phase}.output.lm_head_input",
        } <= names
    assert "input.attention_mask" in names
    assert "input.position_ids" in names
    assert not any(name.startswith("trajectory.") for name in names)

    assert canonical.metadata["forward_counts"] == {"child": 4, "parent": 0}
    assert single_pass.metadata["forward_counts"] == {"child": 1, "parent": 0}
    assert single_pass.metadata["replay_benchmark"] == {
        "batch_count": 1,
        "phase_count": 4,
        "actual_child_forwards": 1,
        "canonical_prefix_child_forwards": 4,
        "saved_child_forwards": 3,
    }
    verification = single_pass.metadata["causal_equivalence"]
    assert verification["reference_supplied"] is True
    assert verification["tolerance_equivalent"] is True
    assert verification["compared_tensors"] == len(actual.tensors)
    assert verification["compared_logits"] == 4
    assert verification["logit_error_model"] == (
        trace_cache.CHILD_REPLAY_LOGIT_ERROR_MODEL
    )
    assert verification["non_logit_tolerance_unchanged"] is False
    assert verification["non_logit_error_model"] == (
        trace_cache.CHILD_REPLAY_NON_LOGIT_ERROR_MODEL
    )
    numeric_policy = single_pass.metadata["numeric_execution_policy"]
    assert verification["numeric_policy_sha256"] == (
        numeric_policy["numeric_policy_sha256"]
    )
    assert verification["non_logit_envelope_sha256"] == (
        numeric_policy["non_logit_envelope_sha256"]
    )
    assert verification[
        "shape_dtype_device_hash_and_finiteness_authenticated"
    ] is True
    assert actual.capture["child_replay_capture_mode"] == (
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )
    assert actual.capture["child_forward_count"] == 1
    assert actual.capture["canonical_prefix_forward_count"] == 4


def test_single_pass_replay_uses_one_child_forward_for_each_batch(
    monkeypatch,
) -> None:
    child = _model(seed=750)
    calls = _instrument_trace_forwards(monkeypatch)
    replay = capture_lfm2_child_replay(
        child,
        [
            _batch(prompt=(1, 2, 3), forced=(4, 5, 6)),
            _batch(prompt=(7, 8, 9, 10), forced=(11,)),
        ],
        selected_layer_indices=[0, 1],
        config=_canonical_config(sample_last_tokens=2),
        input_device="cpu",
        replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    )

    assert [call["input_length"] for call in calls] == [6, 5]
    assert replay.metadata["forward_counts"] == {"child": 2, "parent": 0}
    assert replay.metadata["replay_benchmark"] == {
        "batch_count": 2,
        "phase_count": 6,
        "actual_child_forwards": 2,
        "canonical_prefix_child_forwards": 6,
        "saved_child_forwards": 4,
    }
    assert replay.metadata["causal_equivalence"] == {
        "reference_supplied": False,
        "tolerance_equivalent": None,
    }


def test_single_pass_keeps_legacy_full_profile_and_rejects_point_filter() -> None:
    child = _model(seed=751)
    profile = Lfm2ChildEvidenceProfile(
        profile_role="layer.pre_producer",
        layer_suffixes=((1, ("residual.post_layer", "residual.pre")),),
    )
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="currently require prefix_by_prefix_canonical",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[1],
            config=_canonical_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
            child_evidence_profile=profile,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda child: setattr(
                child.config,
                "_attn_implementation",
                "flash_attention_2",
            ),
            "certified only for eager or SDPA",
        ),
        (
            lambda child: setattr(child.config, "is_causal", False),
            "config.is_causal=True",
        ),
    ],
)
def test_single_pass_replay_fails_closed_before_forward_for_unsupported_runtime(
    mutate,
    message,
    monkeypatch,
) -> None:
    child = _model(seed=760)
    calls = _instrument_trace_forwards(monkeypatch)
    mutate(child)

    with pytest.raises(Lfm2CircuitTraceCaptureError, match=message):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
    assert calls == []


def test_single_pass_replay_rejects_noncontiguous_positions_before_forward(
    monkeypatch,
) -> None:
    child = _model(seed=770)
    calls = _instrument_trace_forwards(monkeypatch)
    source = _batch()
    source["position_ids"] = torch.tensor([[0, 1, 3, 4]], dtype=torch.long)

    with pytest.raises(Lfm2CircuitTraceCaptureError, match="contiguous position_ids"):
        capture_lfm2_child_replay(
            child,
            [source],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
    assert calls == []


def test_single_pass_reference_check_fails_closed_on_future_dependent_logits(
    monkeypatch,
) -> None:
    child = _model(seed=780)
    source = _batch(prompt=(1, 2, 3, 4), forced=(5, 6))
    config = _canonical_config(sample_last_tokens=2)
    canonical = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
    )
    original_output_logits = trace_cache._output_logits

    def future_dependent(torch_module, output):
        logits = original_output_logits(torch_module, output).clone()
        logits[:, :, 0] += logits[:, -1:, 1]
        return logits

    monkeypatch.setattr(trace_cache, "_output_logits", future_dependent)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="differs from the canonical prefix trace",
    ):
        capture_lfm2_child_replay(
            child,
            [source],
            selected_layer_indices=[0, 1],
            config=config,
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
            causal_equivalence_reference=canonical,
        )


def test_single_pass_rejects_future_dependent_lm_head_override_before_forward(
    monkeypatch,
) -> None:
    child = _model(seed=781)
    calls = _instrument_trace_forwards(monkeypatch)

    def future_dependent(self, hidden_states):
        logits = torch.nn.functional.linear(hidden_states, self.weight, None)
        logits[..., 0] += 1e-4 * hidden_states[:, -1:, 0]
        return logits

    child.lm_head.forward = MethodType(future_dependent, child.lm_head)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="exact bias-free torch.nn.Linear LM head",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
    assert calls == []


def test_single_pass_rejects_class_level_lm_head_forward_spoof_before_forward(
    monkeypatch,
) -> None:
    child = _model(seed=7811)
    calls = _instrument_trace_forwards(monkeypatch)
    original_forward = torch.nn.Linear.forward

    def future_dependent(self, hidden_states):
        logits = original_forward(self, hidden_states)
        if self is child.lm_head:
            logits[..., 0] += 1e-4 * hidden_states[:, -1:, 0]
        return logits

    future_dependent.__module__ = original_forward.__module__
    future_dependent.__qualname__ = original_forward.__qualname__
    monkeypatch.setattr(torch.nn.Linear, "forward", future_dependent)

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="authenticated canonical torch.nn.Linear.forward implementation",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
    assert calls == []


def test_single_pass_rejects_reduced_float32_matmul_before_forward(
    monkeypatch,
) -> None:
    child = _model(seed=782)
    calls = _instrument_trace_forwards(monkeypatch)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high")

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="requires float32 matmul precision 'highest'",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
    assert calls == []


def test_single_pass_rejects_replaced_functional_linear_before_forward(
    monkeypatch,
) -> None:
    child = _model(seed=783)
    calls = _instrument_trace_forwards(monkeypatch)
    native_linear = torch.nn.functional.linear

    def replaced_linear(*args, **kwargs):
        return native_linear(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "linear", replaced_linear)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="exact bias-free torch.nn.Linear LM head",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
    assert calls == []


def test_ieee_fp32_runtime_rejects_tf32_and_accepts_ieee_gpu_modes() -> None:
    matmul = SimpleNamespace(fp32_precision="ieee", allow_tf32=True)
    backends = SimpleNamespace(
        fp32_precision="ieee",
        cuda=SimpleNamespace(matmul=matmul),
    )
    cuda = SimpleNamespace(get_device_capability=lambda _device: (8, 0))
    fake_torch = SimpleNamespace(
        get_float32_matmul_precision=lambda: "highest",
        backends=backends,
        cuda=cuda,
    )
    device = SimpleNamespace(type="cuda")

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="rejects TF32-capable CUDA matmul",
    ):
        trace_cache._ieee_fp32_matmul_runtime_binding(fake_torch, device)

    matmul.allow_tf32 = False
    cuda.get_device_capability = lambda _device: (8, 6)
    binding = trace_cache._ieee_fp32_matmul_runtime_binding(fake_torch, device)
    assert binding["cuda_compute_capability"] == [8, 6]
    assert binding["tf32_hardware_supported"] is True
    assert binding["legacy_cuda_allow_tf32"] is False
    assert binding["ieee_fp32_products_proven"] is True

    matmul.allow_tf32 = True
    cuda.get_device_capability = lambda _device: (6, 1)
    binding = trace_cache._ieee_fp32_matmul_runtime_binding(fake_torch, device)
    assert binding["cuda_compute_capability"] == [6, 1]
    assert binding["tf32_hardware_supported"] is False
    assert binding["ieee_fp32_products_proven"] is True


def test_lm_head_error_model_accepts_only_cancellation_safe_fp32_drift() -> None:
    hidden_size = 816
    weight = torch.ones((2, hidden_size), dtype=torch.float32)
    weight[0, 1::2] = -1.0
    weight[1].zero_()
    weight[1, 0] = 1.0
    head_input = torch.ones((1, 1, hidden_size), dtype=torch.float32)
    reference_logits = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    candidate_logits = torch.tensor([[[2.67029e-5, 1.0]]], dtype=torch.float32)

    assert not torch.allclose(
        reference_logits.to(torch.float64),
        candidate_logits.to(torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    result = trace_cache._lm_head_logit_forward_error_check(
        torch,
        name="prefill.logits",
        reference_logits=reference_logits,
        candidate_logits=candidate_logits,
        reference_head_input=head_input,
        candidate_head_input=head_input,
        lm_head_weight=weight,
        rtol=1e-5,
        atol=1e-5,
    )

    assert result["strict_allclose"] is False
    assert result["operation_envelope_used"] is True
    assert result["operation_envelope_value_count"] == 1
    assert 0.0 < result["maximum_operation_envelope_ratio"] < 1.0
    assert result["reduction_width"] == hidden_size
    assert result["unit_roundoff"] == 2.0**-24


def test_lm_head_error_model_rejects_nondecision_changing_excess() -> None:
    hidden_size = 816
    weight = torch.ones((2, hidden_size), dtype=torch.float32)
    weight[0, 1::2] = -1.0
    weight[1].zero_()
    weight[1, 0] = 1.0
    head_input = torch.ones((1, 1, hidden_size), dtype=torch.float32)
    reference_logits = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    candidate_logits = torch.tensor([[[0.2, 1.0]]], dtype=torch.float32)

    assert torch.equal(
        torch.argmax(reference_logits, dim=-1),
        torch.argmax(candidate_logits, dim=-1),
    )
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="beyond its IEEE LM-head dot-product error envelope",
    ):
        trace_cache._lm_head_logit_forward_error_check(
            torch,
            name="prefill.logits",
            reference_logits=reference_logits,
            candidate_logits=candidate_logits,
            reference_head_input=head_input,
            candidate_head_input=head_input,
            lm_head_weight=weight,
            rtol=1e-5,
            atol=1e-5,
        )


class _RecordedEnvelopeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_size = 816
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_size, 1, bias=False)
                for _index in range(14)
            ]
        )
        self.lm_head = torch.nn.Linear(hidden_size, 2, bias=False)


def _a0011_non_logit_envelope():
    numeric_policy = trace_cache.compile_lfm2_child_replay_numeric_policy(
        _RecordedEnvelopeModel(),
        input_device="cpu",
    )
    return numeric_policy["non_logit_envelope_policy"]


def test_recorded_fp32_non_logit_envelope_accepts_a0011_kernel_drift_only() -> None:
    envelope = _a0011_non_logit_envelope()
    reference = torch.ones((1, 1, 816), dtype=torch.float32)
    legal_runtime = reference.clone()
    legal_runtime[..., 0] += 1.60933e-05

    result = trace_cache._cross_kernel_non_logit_forward_error_check(
        torch,
        name="prefill.output.lm_head_input",
        reference_tensor=reference,
        candidate_tensor=legal_runtime,
        envelope_policy=envelope,
        rtol=float(envelope["strict_rtol"]),
        atol=float(envelope["strict_atol"]),
    )

    assert result["strict_allclose"] is False
    assert result["operation_envelope_used"] is True
    assert result["operation_envelope_value_count"] == 1
    assert result["operation_class"] == (
        "complete-residual-stack-plus-final-rmsnorm"
    )
    assert result["path_depth"] == 29
    assert result["reduction_width"] == 816
    assert result["maximum_abs_error"] == pytest.approx(
        1.60933e-05,
        abs=1e-10,
    )
    assert 0.0 < result["maximum_operation_envelope_ratio"] < 1.0
    assert len(result["receipt_sha256"]) == 64

    materially_wrong = reference.clone()
    materially_wrong[..., 0] += 0.1
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="beyond its recorded IEEE cross-kernel operation envelope",
    ):
        trace_cache._cross_kernel_non_logit_forward_error_check(
            torch,
            name="prefill.output.lm_head_input",
            reference_tensor=reference,
            candidate_tensor=materially_wrong,
            envelope_policy=envelope,
            rtol=float(envelope["strict_rtol"]),
            atol=float(envelope["strict_atol"]),
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("shape", "changed shape"),
        ("dtype", "changed dtype"),
        ("device", "changed recorded CPU device/layout"),
        ("hash", "runtime hash binding failed"),
        ("nonfinite", "contains NaN or infinity"),
    ),
)
def test_recorded_non_logit_envelope_rejects_invalid_evidence(
    failure: str,
    message: str,
) -> None:
    envelope = _a0011_non_logit_envelope()
    reference = torch.ones((1, 1, 816), dtype=torch.float32)
    candidate = reference.clone()
    descriptor = trace_cache._tensor_descriptor(torch, candidate)
    if failure == "shape":
        candidate = candidate[..., :-1]
    elif failure == "dtype":
        candidate = candidate.to(torch.float64)
    elif failure == "device":
        candidate = torch.empty((1, 1, 816), device="meta")
    elif failure == "hash":
        descriptor = {**descriptor, "sha256": "0" * 64}
    else:
        candidate[..., 0] = float("nan")

    with pytest.raises(Lfm2CircuitTraceCaptureError, match=message):
        trace_cache._cross_kernel_non_logit_forward_error_check(
            torch,
            name="prefill.output.lm_head_input",
            reference_tensor=reference,
            candidate_tensor=candidate,
            envelope_policy=envelope,
            candidate_descriptor=descriptor,
            rtol=float(envelope["strict_rtol"]),
            atol=float(envelope["strict_atol"]),
        )


def test_single_pass_exact_top1_remains_hard_after_numeric_envelope(
    monkeypatch,
) -> None:
    child = _model(seed=785)
    source = _batch(prompt=(1, 2, 3, 4), forced=(5, 6))
    config = _canonical_config(sample_last_tokens=2)
    canonical = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
    )
    original_output_logits = trace_cache._output_logits

    def top1_dependent(torch_module, output):
        logits = original_output_logits(torch_module, output).clone()
        winner = torch.argmax(logits, dim=-1, keepdim=True)
        replacement = winner.remainder(int(logits.shape[-1]) - 1) + 1
        winning_values = logits.gather(-1, winner)
        logits.scatter_(-1, replacement, winning_values + 1.0)
        return logits

    def accept_numeric_envelope(*_args, **_kwargs):
        return {
            "strict_allclose": False,
            "operation_envelope_used": True,
            "operation_envelope_value_count": 1,
            "maximum_operation_envelope_ratio": 0.5,
            "maximum_operation_envelope_bound": 2.0,
            "unit_roundoff": 2.0**-24,
            "reduction_width": 8,
        }

    monkeypatch.setattr(trace_cache, "_output_logits", top1_dependent)
    monkeypatch.setattr(
        trace_cache,
        "_lm_head_logit_forward_error_check",
        accept_numeric_envelope,
    )
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="choose a different top-1 token",
    ):
        capture_lfm2_child_replay(
            child,
            [source],
            selected_layer_indices=[0, 1],
            config=config,
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
            causal_equivalence_reference=canonical,
        )


def test_single_pass_replay_matches_sdpa_prefix_oracle_within_tolerance() -> None:
    child = _model(seed=790)
    child.config._attn_implementation = "sdpa"
    source = _batch(prompt=(1, 2, 3, 4, 5), forced=(6, 7))
    config = _canonical_config(sample_last_tokens=3)
    canonical = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
    )
    single_pass = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
        replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        causal_equivalence_reference=canonical,
    )

    assert single_pass.metadata["causal_equivalence"][
        "tolerance_equivalent"
    ] is True
    assert single_pass.metadata["forward_counts"] == {"child": 1, "parent": 0}


def test_single_pass_replay_rejects_instance_forward_override() -> None:
    child = _model(seed=800)
    original = child.forward

    def wrapped(self, *args, **kwargs):
        return original(*args, **kwargs)

    child.forward = MethodType(wrapped, child)
    with pytest.raises(Lfm2CircuitTraceCaptureError, match="without instance"):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )


class _PretendAcceleratorTensor:
    """A CPU tensor that reports an accelerator device to the staging path.

    The staging branch only ever asks a source for ``layout``, ``device``,
    ``detach()`` and ``contiguous()`` before handing the contiguous result to
    ``copy_``, so this is enough to exercise the real pinned-buffer, pooling,
    single-synchronization, and deferred-validation logic on a CPU-only box.
    """

    def __init__(self, value) -> None:
        self._value = value
        self.layout = value.layout
        self.device = SimpleNamespace(type="cuda")

    def numel(self):
        return self._value.numel()

    def detach(self):
        return _PretendAcceleratorTensor(self._value.detach())

    def contiguous(self):
        return self._value.contiguous()


class _PretendAcceleratorTorch:
    """Delegate to real torch while faking the CUDA-only staging surface."""

    def __init__(self, real, *, synchronized: list[object], pinned: list[int]) -> None:
        self._real = real
        self._synchronized = synchronized
        self._pinned = pinned
        self.cuda = SimpleNamespace(
            synchronize=lambda device=None: synchronized.append(device)
        )

    def __getattr__(self, name):
        return getattr(self._real, name)

    def is_tensor(self, value):
        return isinstance(value, _PretendAcceleratorTensor) or self._real.is_tensor(
            value
        )

    def empty(self, *args, pin_memory: bool = False, **kwargs):
        if pin_memory:
            self._pinned.append(int(args[0]))
        return self._real.empty(*args, **kwargs)


def _pretend_accelerator_stage():
    synchronized: list[object] = []
    pinned: list[int] = []
    fake = _PretendAcceleratorTorch(torch, synchronized=synchronized, pinned=pinned)
    return trace_cache._HostTransferStage(fake), synchronized, pinned


def test_staged_accelerator_copies_land_on_one_synchronization() -> None:
    stage, synchronized, pinned = _pretend_accelerator_stage()
    first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
    third = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    handles = [
        stage.stage(_PretendAcceleratorTensor(value), f"staged {index}")
        for index, value in enumerate((first, second, third))
    ]
    assert all(handle.tensor is None for handle in handles)
    assert synchronized == []

    stage.resolve()

    assert len(synchronized) == 1
    assert pinned == [6, 6, 4]
    assert [handle.name for handle in handles] == [
        "staged 0",
        "staged 1",
        "staged 2",
    ]
    for handle, value in zip(handles, (first, second, third)):
        assert torch.equal(handle.tensor, value)
        assert handle.tensor.is_contiguous()
        assert handle.pinned is None
    assert stage.counters == {
        "device_synchronizations": 1,
        "staged_host_transfers": 3,
        "direct_host_transfers": 0,
    }

    # The pinned pool is reused, so a second phase allocates nothing new.
    stage.stage(_PretendAcceleratorTensor(first), "staged 3")
    stage.resolve()
    assert pinned == [6, 6, 4]
    assert len(synchronized) == 2


def test_staged_accelerator_copies_defer_the_same_finiteness_failure() -> None:
    stage, synchronized, _pinned = _pretend_accelerator_stage()
    clean = torch.ones((2, 2), dtype=torch.float32)
    poisoned = torch.ones((2, 2), dtype=torch.float32)
    poisoned[1, 1] = float("nan")

    stage.stage(_PretendAcceleratorTensor(clean), "clean endpoint")
    stage.stage(_PretendAcceleratorTensor(poisoned), "poisoned endpoint")
    assert synchronized == []

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="poisoned endpoint contains NaN or infinity",
    ):
        stage.resolve()
    assert len(synchronized) == 1


def test_parallel_sealed_replay_is_byte_identical_to_serial_hashing(
    monkeypatch,
) -> None:
    child = _model(seed=915)
    source = _batch()
    config = _canonical_config()

    parallel = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
    )
    parallel_batch = parallel.load_batch(0)

    # An independent serial recomputation of the sealed inventory must match
    # the pipeline-sealed one byte for byte.
    expected_inventory = {
        name: trace_cache._tensor_descriptor(torch, value)
        for name, value in sorted(parallel_batch.tensors.items())
    }
    assert trace_cache._canonical_json(
        parallel_batch.record["tensors"]
    ) == trace_cache._canonical_json(expected_inventory)

    def serial_tensor_sha256(torch_module, values):
        return tuple(
            trace_cache._tensor_sha256(torch_module, value) for value in values
        )

    monkeypatch.setattr(trace_cache, "_parallel_tensor_sha256", serial_tensor_sha256)
    monkeypatch.setattr(trace_cache, "_SEAL_PIPELINE_ENABLED", False)
    serial = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
    )

    assert serial.replay_sha256 == parallel.replay_sha256
    assert trace_cache._canonical_json(
        serial.metadata
    ) == trace_cache._canonical_json(parallel.metadata)
    assert trace_cache._canonical_json(
        serial.load_batch(0).record
    ) == trace_cache._canonical_json(parallel_batch.record)
    assert serial.load_batch(0).record["replay_batch_sha256"] == (
        parallel_batch.record["replay_batch_sha256"]
    )

    # Wall-clock accounting is reported beside the replay and never sealed.
    assert parallel.capture_timing["seal_pipeline_enabled"] is True
    assert serial.capture_timing["seal_pipeline_enabled"] is False
    assert parallel.capture_timing["sealed_in_metadata"] is False
    assert parallel.capture_timing["batch_count"] == 1
    assert parallel.capture_timing["sealed_tensor_count"] == len(
        parallel_batch.tensors
    )
    assert parallel.capture_timing["device_synchronizations"] == 0
    assert parallel.capture_timing["staged_host_transfers"] == 0
    assert parallel.capture_timing["direct_host_transfers"] > 0
    assert parallel.capture_timing["replay_seconds"] >= 0.0
    assert "capture_timing" not in parallel.metadata
    assert "replay_seconds" not in trace_cache._canonical_json(
        parallel.metadata
    ).decode("ascii")


def test_deferred_finiteness_fails_closed_on_an_injected_nan_before_sealing(
    monkeypatch,
) -> None:
    child = _model(seed=916)
    original_output_tensor = trace_cache._output_tensor
    sealed: list[int] = []
    original_inventory = trace_cache._ordered_tensor_descriptors

    def poisoned_output_tensor(torch_module, output, endpoint_name):
        value = original_output_tensor(torch_module, output, endpoint_name)
        if endpoint_name != "MLP gate":
            return value
        poisoned = value.clone()
        # The last element always survives trailing-token sampling.
        poisoned.reshape(-1)[-1] = float("nan")
        return poisoned

    def counted_inventory(torch_module, tensors):
        sealed.append(len(tensors))
        return original_inventory(torch_module, tensors)

    monkeypatch.setattr(trace_cache, "_output_tensor", poisoned_output_tensor)
    monkeypatch.setattr(trace_cache, "_ordered_tensor_descriptors", counted_inventory)

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="contains NaN or infinity",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
        )
    assert sealed == []


def test_single_pass_replay_rejects_uncertified_low_precision_before_forward(
    monkeypatch,
) -> None:
    child = _model(seed=810).to(dtype=torch.bfloat16)
    calls = _instrument_trace_forwards(monkeypatch)

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="not certified low-precision",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
    assert calls == []


def _uneven_shard_corpus():
    """Batches of deliberately different lengths so assignment can matter."""

    return [
        _batch(prompt=(1, 2, 3, 4), forced=(5, 6)),
        _batch(prompt=(7, 8, 9), forced=(10, 11, 12)),
        _batch(prompt=(2, 4, 6, 8, 10), forced=(3,)),
        _batch(prompt=(1, 1, 1), forced=(2, 2)),
        _batch(prompt=(5, 5, 5, 5, 5, 5), forced=(1, 2, 3)),
    ]


def _assert_replays_are_byte_identical(reference, candidate) -> None:
    assert candidate.replay_sha256 == reference.replay_sha256
    assert trace_cache._canonical_json(
        candidate.metadata
    ) == trace_cache._canonical_json(reference.metadata)
    assert len(candidate) == len(reference)
    for index in range(len(reference)):
        expected = reference.load_batch(index)
        observed = candidate.load_batch(index)
        assert observed.index == expected.index
        assert observed.input_sha256 == expected.input_sha256
        assert trace_cache._canonical_json(
            observed.record
        ) == trace_cache._canonical_json(expected.record)
        assert set(observed.tensors) == set(expected.tensors)
        for name, value in expected.tensors.items():
            assert observed.tensors[name].dtype == value.dtype
            assert torch.equal(observed.tensors[name], value)


def test_two_cpu_shard_workers_seal_the_same_bytes_as_one_device() -> None:
    child = _model(seed=920)
    corpus = _uneven_shard_corpus()
    config = _canonical_config()
    modes_before = _module_modes(child)

    single = capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
    )
    sharded = capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
        replay_devices=("cpu", "cpu"),
    )

    _assert_replays_are_byte_identical(single, sharded)
    assert _module_modes(child) == modes_before
    # The device only ever reaches the envelope, never a batch record.
    assert "device" not in trace_cache._canonical_json(
        single.load_batch(0).record
    ).decode("ascii")

    shard_timing = sharded.capture_timing["replay_sharding"]
    assert shard_timing["worker_count"] == 2
    assert shard_timing["devices"] == ["cpu", "cpu"]
    assert shard_timing["device_type"] == "cpu"
    assert shard_timing["device_class"] is None
    assert shard_timing["assignment"] == "round_robin"
    assert shard_timing["batch_assignment"] == [0, 1, 0, 1, 0]
    assert shard_timing["materialized_replica_count"] == 1
    assert len(shard_timing["resident_child_bytes_per_worker"]) == 2
    assert shard_timing["resident_child_bytes_total"] == sum(
        shard_timing["resident_child_bytes_per_worker"]
    )
    assert shard_timing["replay_wall_seconds"] >= 0.0
    assert "replay_sharding" not in single.capture_timing
    # Sharding is wall-clock accounting only and never enters the seal.
    assert "replay_sharding" not in trace_cache._canonical_json(
        sharded.metadata
    ).decode("ascii")


def test_shard_workers_stage_their_host_copies_independently() -> None:
    child = _model(seed=921)
    corpus = _uneven_shard_corpus()
    config = _canonical_config()

    single = capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
    )
    sharded = capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
        replay_devices=("cpu", "cpu"),
    )

    workers = sharded.capture_timing["replay_sharding"]["per_worker"]
    assert len(workers) == 2
    # A shared staging pool would land every host copy on one worker's
    # counters.  Each worker owns its own _HostTransferStage, so the counts
    # split by assignment and still add up to the single-device total.
    assert all(worker["direct_host_transfers"] > 0 for worker in workers)
    assert sum(worker["direct_host_transfers"] for worker in workers) == (
        single.capture_timing["direct_host_transfers"]
    )
    assert sharded.capture_timing["direct_host_transfers"] == (
        single.capture_timing["direct_host_transfers"]
    )
    assert sharded.capture_timing["sealed_tensor_count"] == (
        single.capture_timing["sealed_tensor_count"]
    )
    assert [entry["batch_index"] for entry in sharded.capture_timing["per_batch"]] == (
        list(range(len(corpus)))
    )
    assert sharded.capture_timing["peak_capture_bytes"] == (
        single.capture_timing["peak_capture_bytes"]
    )
    assert all(
        worker["peak_capture_bytes"] <= sharded.capture_timing["peak_capture_bytes"]
        for worker in workers
    )
    assert all(worker["resident_child_bytes"] > 0 for worker in workers)


def test_shard_assignment_strategies_and_worker_counts_all_seal_alike() -> None:
    child = _model(seed=922)
    corpus = _uneven_shard_corpus()
    config = _canonical_config()

    single = capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
    )
    variants = {
        "one_worker": ("cpu",),
        "three_workers": ("cpu", "cpu", "cpu"),
        "token_cost": trace_cache.Lfm2ReplayShardingConfig(
            devices=("cpu", "cpu", "cpu"),
            assignment="token_cost",
        ),
    }
    assignments = {}
    for name, devices in variants.items():
        candidate = capture_lfm2_child_replay(
            child,
            corpus,
            selected_layer_indices=[0, 1],
            config=config,
            input_device="cpu",
            replay_devices=devices,
        )
        _assert_replays_are_byte_identical(single, candidate)
        assignments[name] = candidate.capture_timing["replay_sharding"][
            "batch_assignment"
        ]

    assert assignments["one_worker"] == [0] * len(corpus)
    assert assignments["three_workers"] == [0, 1, 2, 0, 1]
    # The cost-balanced plan really does move work around, and the sealed
    # bytes above are identical anyway.
    assert assignments["token_cost"] != assignments["three_workers"]


def test_sharded_replay_matches_single_device_under_an_evidence_profile() -> None:
    child = _model(seed=923)
    profile = Lfm2ChildEvidenceProfile(
        profile_role="layer.post_consumer",
        layer_suffixes=(
            (1, ("residual.post_layer", "residual.pre")),
        ),
    )
    corpus = _uneven_shard_corpus()

    single = capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[1],
        config=_canonical_config(),
        input_device="cpu",
        child_evidence_profile=profile,
    )
    sharded = capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[1],
        config=_canonical_config(),
        input_device="cpu",
        child_evidence_profile=profile,
        replay_devices=("cpu", "cpu"),
    )

    _assert_replays_are_byte_identical(single, sharded)
    assert sharded.load_batch(0).child_evidence_profile == profile


def test_sharded_single_pass_replay_matches_the_single_device_capture() -> None:
    child = _model(seed=924)
    corpus = _uneven_shard_corpus()
    config = _canonical_config()

    single = capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
        replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    )
    sharded = capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[0, 1],
        config=config,
        input_device="cpu",
        replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        replay_devices=("cpu", "cpu"),
    )

    _assert_replays_are_byte_identical(single, sharded)
    # The single-pass causal certificate is sealed into every batch record,
    # so an equal record already proves the replica certified the same one.
    assert sharded.load_batch(0).capture["causal_equivalence_contract"] == (
        single.load_batch(0).capture["causal_equivalence_contract"]
    )


def test_shard_replicas_are_fingerprint_checked_before_and_after_replay(
    monkeypatch,
) -> None:
    child = _model(seed=925)
    corpus = _uneven_shard_corpus()
    original = trace_cache.fingerprint_lfm2_parent
    observed: list[int] = []

    def counted(model):
        observed.append(id(model))
        return original(model)

    monkeypatch.setattr(trace_cache, "fingerprint_lfm2_parent", counted)
    capture_lfm2_child_replay(
        child,
        corpus,
        selected_layer_indices=[0, 1],
        config=_canonical_config(),
        input_device="cpu",
        replay_devices=("cpu", "cpu"),
    )

    # Primary before, primary for the replica-set baseline, the replica once
    # before replay, the replica once after replay, and the primary after.
    assert len(observed) == 5
    assert observed.count(id(child)) == 3
    replica_ids = {value for value in observed if value != id(child)}
    assert len(replica_ids) == 1


def test_a_replica_that_drifts_during_replay_fails_closed(monkeypatch) -> None:
    child = _model(seed=926)
    corpus = _uneven_shard_corpus()
    original = trace_cache._replay_child_batch

    def drifting(torch_module, model, *args, **kwargs):
        result = original(torch_module, model, *args, **kwargs)
        if model is not child:
            with torch.no_grad():
                model.model.layers[0].feed_forward.w1.weight.add_(1.0)
        return result

    monkeypatch.setattr(trace_cache, "_replay_child_batch", drifting)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="exact parameter bytes",
    ):
        capture_lfm2_child_replay(
            child,
            corpus,
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_devices=("cpu", "cpu"),
        )


def test_shard_request_rejects_a_device_that_is_not_the_live_child_device() -> None:
    child = _model(seed=927)

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="live child's own device",
    ):
        capture_lfm2_child_replay(
            child,
            _uneven_shard_corpus(),
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_devices=("cuda:0", "cuda:1"),
        )


def test_shard_request_rejects_an_already_hooked_child() -> None:
    child = _model(seed=930)
    _instrument_forwards(child)

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="requires an unhooked child",
    ):
        capture_lfm2_child_replay(
            child,
            _uneven_shard_corpus(),
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_devices=("cpu", "cpu"),
        )


def test_shard_request_rejects_more_workers_than_batches() -> None:
    child = _model(seed=928)

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="every worker must receive at least one batch",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_devices=("cpu", "cpu"),
        )


def test_a_failing_shard_batch_reports_the_lowest_failing_index() -> None:
    child = _model(seed=929)
    corpus = _uneven_shard_corpus()
    poisoned = [
        dict(corpus[0]),
        {**corpus[1], "input_ids": torch.tensor([[1, 2, 999]], dtype=torch.long)},
        {**corpus[2], "input_ids": torch.tensor([[1, 998, 3]], dtype=torch.long)},
        dict(corpus[3]),
        dict(corpus[4]),
    ]
    for entry in (poisoned[1], poisoned[2]):
        entry["attention_mask"] = torch.ones_like(entry["input_ids"])

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="Batch 1 prompt contains out-of-vocabulary tokens",
    ):
        capture_lfm2_child_replay(
            child,
            poisoned,
            selected_layer_indices=[0, 1],
            config=_canonical_config(),
            input_device="cpu",
            replay_devices=("cpu", "cpu"),
        )
