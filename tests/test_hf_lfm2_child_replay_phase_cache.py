from __future__ import annotations

# Optional dependency probes intentionally precede the Torch-backed imports.
# ruff: noqa: E402

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache
from neural_forge.dynamic_child_replay_phase_cache import (
    DynamicChildReplayPhaseCacheContext,
)
from neural_forge.dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseCacheIntegrityError,
)
from neural_forge.hf_lfm2_child_replay_phase_cache import (
    LFM2_CHILD_REPLAY_PHASE,
    describe_lfm2_child_replay_phase,
    load_or_capture_lfm2_child_replay_phase,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2ChildEvidenceProfile,
    Lfm2CircuitTraceCaptureConfig,
)


def _model(*, seed: int = 701):
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


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


def test_lfm2_child_replay_phase_cold_then_portable_hot_hit(
    tmp_path: Path,
) -> None:
    context = DynamicChildReplayPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "replays")
    )
    batches = (_batch(), _batch(prompt=(7, 8, 9), forced=(10,)))
    cold = load_or_capture_lfm2_child_replay_phase(
        context,
        _model(seed=710),
        batches,
        selected_layer_indices=(0, 1),
        config=_canonical_config(),
        input_device="cpu",
    )
    assert cold.cache_hit is False
    assert cold.producer_invoked is True
    assert cold.replay.capture_timing

    def forbidden_capture():
        raise AssertionError("hot child replay phase invoked its producer")

    hot = load_or_capture_lfm2_child_replay_phase(
        DynamicChildReplayPhaseCacheContext(
            cache=DynamicFoldPhaseCache(tmp_path / "replays")
        ),
        _model(seed=710),
        batches,
        selected_layer_indices=(0, 1),
        config=_canonical_config(),
        input_device="cpu",
        replay_devices=("cpu", "cpu"),
        capture_provider=forbidden_capture,
    )
    assert hot.cache_hit is True
    assert hot.producer_invoked is False
    assert hot.replay.capture_timing == {}
    assert hot.entry.completion_sha256 == cold.entry.completion_sha256
    _assert_replays_are_byte_identical(cold.replay, hot.replay)


def test_lfm2_child_replay_phase_key_binds_semantics_not_object_identity() -> None:
    batches = (_batch(),)
    config = _canonical_config()

    def describe(child, values=batches, selected=(0,), selected_config=config, profile=None):
        return describe_lfm2_child_replay_phase(
            child,
            values,
            selected_layer_indices=selected,
            config=selected_config,
            input_device="cpu",
            child_evidence_profile=profile,
        )

    baseline = describe(_model(seed=720))
    assert describe(_model(seed=720)).phase_key.as_dict() == (
        baseline.phase_key.as_dict()
    )

    changed_child = _model(seed=720)
    with torch.no_grad():
        next(changed_child.parameters()).view(-1)[0].add_(1.0)
    assert describe(changed_child).phase_key.phase_key_sha256 != (
        baseline.phase_key.phase_key_sha256
    )
    assert describe(
        _model(seed=720),
        values=(_batch(forced=(5, 7)),),
    ).phase_key.phase_key_sha256 != baseline.phase_key.phase_key_sha256
    assert describe(
        _model(seed=720),
        selected=(1,),
    ).phase_key.phase_key_sha256 != baseline.phase_key.phase_key_sha256
    assert describe(
        _model(seed=720),
        selected_config=_canonical_config(sample_last_tokens=1),
    ).phase_key.phase_key_sha256 != baseline.phase_key.phase_key_sha256
    profile = Lfm2ChildEvidenceProfile(
        profile_role="phase-cache-test",
        layer_suffixes=((0, ("residual.pre",)),),
    )
    assert describe(
        _model(seed=720),
        profile=profile,
    ).phase_key.phase_key_sha256 != baseline.phase_key.phase_key_sha256


def test_lfm2_child_replay_phase_failure_does_not_publish_and_tamper_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replays"
    context = DynamicChildReplayPhaseCacheContext(
        cache=DynamicFoldPhaseCache(root)
    )
    child = _model(seed=730)
    batches = (_batch(),)
    request = describe_lfm2_child_replay_phase(
        child,
        batches,
        selected_layer_indices=(0,),
        config=_canonical_config(),
        input_device="cpu",
    )

    def failed_capture():
        raise RuntimeError("synthetic replay failure")

    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        load_or_capture_lfm2_child_replay_phase(
            context,
            child,
            batches,
            selected_layer_indices=(0,),
            config=_canonical_config(),
            input_device="cpu",
            capture_provider=failed_capture,
        )
    assert not context.cache.entry_path(request.phase_key).exists()

    published = load_or_capture_lfm2_child_replay_phase(
        context,
        child,
        batches,
        selected_layer_indices=(0,),
        config=_canonical_config(),
        input_device="cpu",
    )
    replay_path = published.entry.payload_path / "replay.json"
    replay_path.write_bytes(replay_path.read_bytes() + b" ")
    with pytest.raises(FoldPhaseCacheIntegrityError, match="payload differs"):
        load_or_capture_lfm2_child_replay_phase(
            context,
            child,
            batches,
            selected_layer_indices=(0,),
            config=_canonical_config(),
            input_device="cpu",
            capture_provider=lambda: (_ for _ in ()).throw(
                AssertionError("tampered hit invoked producer")
            ),
        )


def test_lfm2_child_replay_phase_api_has_no_parent_or_tokenizer() -> None:
    parameters = inspect.signature(
        load_or_capture_lfm2_child_replay_phase
    ).parameters
    assert "parent_model" not in parameters
    assert "tokenizer" not in parameters
    assert LFM2_CHILD_REPLAY_PHASE == "lfm2-child-replay-v1"
