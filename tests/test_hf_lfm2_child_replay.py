from __future__ import annotations

import inspect
from types import MethodType

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
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
