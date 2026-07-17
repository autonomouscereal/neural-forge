from __future__ import annotations

from types import MethodType

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
from experiments import run_lfm2_residual_attention_fold as residual  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    TRACE_FILENAME,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCaptureError,
    Lfm2CircuitTraceCorruptionError,
    build_lfm2_circuit_trace_cache,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)


def _parent(*, seed: int = 101):
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


def _batch(*, forced=(5, 6)):
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "forced_decode_ids": torch.tensor([list(forced)], dtype=torch.long),
    }


def _module_modes(model):
    return {name: bool(module.training) for name, module in model.named_modules()}


def _instrument_forwards(model):
    calls: list[dict[str, object]] = []
    original = model.forward

    def observed(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        calls.append(
            {
                "use_cache": kwargs.get("use_cache"),
                "has_past": kwargs.get("past_key_values") is not None,
                "input_length": int(input_ids.shape[1]),
                "grad_enabled": torch.is_grad_enabled(),
                "training": bool(self.training),
            }
        )
        return original(*args, **kwargs)

    model.forward = MethodType(observed, model)
    return calls


def test_real_lfm2_cache_captures_bcx_attention_mlp_and_forced_decode(
    tmp_path,
) -> None:
    model = _parent()
    assert not hasattr(model.config, "head_dim")
    calls = _instrument_forwards(model)
    model.train()
    model.model.layers[0].eval()
    modes_before = _module_modes(model)
    config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=3,
        max_decode_steps=4,
        logit_rtol=1e-5,
        logit_atol=1e-6,
    )

    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch()],
        config=config,
        input_device="cpu",
    )

    assert _module_modes(model) == modes_before
    assert [call["use_cache"] for call in calls] == [
        False,
        True,
        True,
        False,
        True,
        False,
    ]
    assert [call["has_past"] for call in calls] == [
        False,
        False,
        True,
        False,
        True,
        False,
    ]
    assert [call["input_length"] for call in calls] == [4, 4, 1, 5, 1, 6]
    assert all(call["grad_enabled"] is False for call in calls)
    assert all(call["training"] is False for call in calls)

    manifest = cache.manifest
    assert cache.path.name == cache.cache_key
    assert len(cache.parent_fingerprint) == 64
    assert len(cache.corpus_fingerprint) == 64
    assert len(cache.implementation_fingerprint) == 64
    assert len(cache.layer_topology_fingerprint) == 64
    assert manifest["capture"]["gradients_used"] is False
    assert manifest["capture"]["training_steps"] == 0
    assert manifest["capture"]["prefill_use_cache"] is False
    assert manifest["capture"]["forced_decode_use_cache"] is True
    assert manifest["execution"]["attention_backend"] == "eager"
    assert manifest["execution"]["input_device_type"] == "cpu"
    assert manifest["execution"]["fast_short_conv_selected"] is False
    assert isinstance(manifest["execution"]["fast_short_conv_available"], bool)
    topology = manifest["layer_topology"]["layers"]
    assert [entry["contract"]["kind"] for entry in topology] == [
        "short_conv",
        "full_attention",
    ]
    assert topology[0]["contract"]["operator"]["input_projection_chunk_order"] == [
        "b",
        "c",
        "x",
    ]
    assert topology[0]["contract"]["operator"]["not_attention_qkv"] is True
    assert topology[1]["contract"]["operator"]["head_dim"] == 4

    batch = cache.load_batch(0)
    tensors = batch.tensors
    assert torch.equal(batch.input_ids, _batch()["input_ids"])
    assert torch.equal(batch.forced_decode_ids, _batch()["forced_decode_ids"])

    assert tensors["prefill.layer.000.short_conv.b"].shape == (1, 3, 8)
    assert tensors["prefill.layer.000.short_conv.c"].shape == (1, 3, 8)
    assert tensors["prefill.layer.000.short_conv.x"].shape == (1, 3, 8)
    torch.testing.assert_close(
        tensors["prefill.layer.000.short_conv.bx"],
        tensors["prefill.layer.000.short_conv.b"]
        * tensors["prefill.layer.000.short_conv.x"],
    )
    assert tensors["prefill.layer.000.mlp.gate"].shape == (1, 3, 12)
    assert tensors["prefill.layer.000.mlp.up"].shape == (1, 3, 12)
    assert tensors["prefill.layer.000.mlp.post_swiglu"].shape == (
        1,
        3,
        12,
    )
    assert tensors["prefill.layer.000.mlp.down"].shape == (1, 3, 8)

    assert tensors["prefill.layer.001.attention.q"].shape == (1, 2, 3, 4)
    assert tensors["prefill.layer.001.attention.k"].shape == (1, 1, 3, 4)
    assert tensors["prefill.layer.001.attention.v"].shape == (1, 1, 3, 4)
    assert tensors["prefill.layer.001.attention.context"].shape == (
        1,
        2,
        3,
        4,
    )
    assert tensors["prefill.layer.001.attention.o"].shape == (1, 3, 8)
    assert tensors["decode.000.layer.001.attention.q"].shape == (1, 2, 1, 4)
    assert tensors["cached_prefill.layer.001.attention.key_cache"].shape == (1, 1, 3, 4)
    assert tensors["cached_prefill.layer.000.short_conv.conv_state"].shape == (1, 8, 3)
    assert tensors["cached_prefill.layer.000.short_conv.state_flags"].tolist() == [
        1,
        0,
        1,
    ]
    assert tensors["cached_prefill.layer.000.short_conv.kernel_size"].tolist() == [3]
    torch.testing.assert_close(
        tensors["decode.001.layer.001.attention.k"],
        tensors["decode.001.layer.001.attention.key_cache"][:, :, -1:, :],
    )
    torch.testing.assert_close(
        tensors["decode.001.layer.001.attention.v"],
        tensors["decode.001.layer.001.attention.value_cache"][:, :, -1:, :],
    )

    assert not any(name.startswith("prefill.layer.000.attention.") for name in tensors)
    assert not any(
        name.startswith("decode.000.layer.000.attention.") for name in tensors
    )
    assert not any(name.startswith("prefill.layer.001.short_conv.") for name in tensors)
    assert tensors["prefill.logits"].shape == (1, 3, 31)
    for step in range(2):
        prefix = f"decode.{step:03d}"
        torch.testing.assert_close(
            tensors[f"{prefix}.logits"],
            tensors[f"{prefix}.full_prefix_logits"],
            rtol=config.logit_rtol,
            atol=config.logit_atol,
        )

    record = batch.record
    verifications = record["capture"]["cached_full_prefix_verifications"]
    assert [item["phase"] for item in verifications] == [
        "cached_prefill",
        "decode.000",
        "decode.001",
    ]
    assert all(item["numerically_close"] is True for item in verifications)
    assert all(item["top1_agreement"] is True for item in verifications)
    assert record["capture"]["all_cached_logits_verified"] is True
    assert record["capture"]["all_cached_top1_verified"] is True
    for value in tensors.values():
        assert value.device.type == "cpu"
        assert value.requires_grad is False
        assert value.grad_fn is None


def test_full_prefix_canonical_mode_never_accepts_cached_logits_and_keeps_paths(
    tmp_path,
) -> None:
    model = _parent(seed=151)
    batch = _batch()
    base = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=3,
        max_decode_steps=4,
    )
    cached = build_lfm2_circuit_trace_cache(
        tmp_path / "cached",
        model,
        [batch],
        config=base,
        input_device="cpu",
    )
    calls = _instrument_forwards(model)
    canonical = build_lfm2_circuit_trace_cache(
        tmp_path / "canonical",
        model,
        [batch],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=3,
            max_decode_steps=4,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )

    assert [call["use_cache"] for call in calls] == [False, False, False]
    assert [call["has_past"] for call in calls] == [False, False, False]
    assert [call["input_length"] for call in calls] == [4, 5, 6]
    capture = canonical.manifest["capture"]
    assert capture["decode_trace_mode"] == FULL_PREFIX_CANONICAL_TRACE_MODE
    assert capture["canonical_trace_source"] == "use_cache_false_full_prefix"
    assert capture["cached_prefill_use_cache"] is False
    assert capture["forced_decode_use_cache"] is False
    assert capture["cached_logits_accepted"] is False

    canonical_batch = canonical.load_batch(0)
    tensors = canonical_batch.tensors
    assert "prefill.logits" in tensors
    assert "decode.000.logits" in tensors
    assert "decode.001.logits" in tensors
    assert not any(name.startswith("cached_prefill.") for name in tensors)
    assert not any(name.endswith(".full_prefix_logits") for name in tensors)
    record = canonical_batch.record["capture"]
    assert record["decode_trace_mode"] == FULL_PREFIX_CANONICAL_TRACE_MODE
    assert record["cached_logits_accepted"] is False
    assert record["cached_forward_count"] == 0
    assert record["cached_full_prefix_verifications"] == []
    assert record["all_cached_logits_verified"] is None
    assert record["all_cached_top1_verified"] is None

    cached_rows = residual.extract_trace_rows(cached)
    canonical_rows = residual.extract_trace_rows(canonical)
    assert canonical_rows.paths == cached_rows.paths
    assert canonical_rows.path_splits == cached_rows.path_splits
    assert canonical_rows.layer_kinds == cached_rows.layer_kinds
    for layer_index in range(len(canonical_rows.layer_kinds)):
        assert set(canonical_rows.layer_fields[layer_index]) == set(
            cached_rows.layer_fields[layer_index]
        )
        for field in canonical_rows.layer_fields[layer_index]:
            assert (
                canonical_rows.field(layer_index, field).shape
                == cached_rows.field(layer_index, field).shape
            )


def test_cache_reuses_without_parent_forward_and_identity_changes_split_paths(
    tmp_path,
) -> None:
    model = _parent(seed=202)
    calls = _instrument_forwards(model)
    config = Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2)
    first = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(7,))],
        config=config,
    )
    first_call_count = len(calls)
    first_trace = (
        first.path / "batches" / "batch-000000" / TRACE_FILENAME
    ).read_bytes()

    reused = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(7,))],
        config=config,
    )

    assert reused.path == first.path
    assert len(calls) == first_call_count
    assert (
        first.path / "batches" / "batch-000000" / TRACE_FILENAME
    ).read_bytes() == first_trace
    assert open_lfm2_circuit_trace_cache(first.path).cache_key == first.cache_key

    loaded = first.load_batch(0)
    original = loaded.tensors["prefill.layer.000.short_conv.b"].clone()
    loaded.tensors["prefill.layer.000.short_conv.b"].zero_()
    reloaded = first.load_batch(0)
    assert torch.equal(reloaded.tensors["prefill.layer.000.short_conv.b"], original)

    changed_corpus = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(8,))],
        config=config,
    )
    assert changed_corpus.path != first.path
    assert changed_corpus.corpus_fingerprint != first.corpus_fingerprint

    parent_before = fingerprint_lfm2_parent(model)
    with torch.no_grad():
        model.model.embed_tokens.weight[0, 0].add_(0.25)
    parent_after = fingerprint_lfm2_parent(model)
    assert parent_after != parent_before
    changed_parent = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(7,))],
        config=config,
    )
    assert changed_parent.path not in {first.path, changed_corpus.path}
    assert changed_parent.parent_fingerprint != first.parent_fingerprint


@pytest.mark.parametrize(
    "audited_collector_sha256",
    (
        "02fc790bc0d5f447bb39f3484a1c318ff28850b4e60843d9a21daac54691af31",
        "93da3605e25482b092ee2a23ee7d45cadf26b1a86dcd56db8b2046aee7450eaa",
    ),
)
def test_audited_collector_revision_reuses_sealed_cache_without_forward(
    tmp_path,
    monkeypatch,
    audited_collector_sha256,
) -> None:
    model = _parent(seed=203)
    calls = _instrument_forwards(model)
    current = trace_cache.implementation_identity()

    def identity_with_collector(collector_sha256):
        body = {
            key: value
            for key, value in current.items()
            if key != "implementation_sha256"
        }
        body["collector_source_sha256"] = collector_sha256
        return {
            **body,
            "implementation_sha256": trace_cache._json_sha256(body),
        }

    legacy = identity_with_collector(audited_collector_sha256)
    monkeypatch.setattr(trace_cache, "implementation_identity", lambda: legacy)
    config = Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2)
    first = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(7,))],
        config=config,
    )
    first_call_count = len(calls)

    revised = identity_with_collector("f" * 64)
    monkeypatch.setattr(trace_cache, "implementation_identity", lambda: revised)
    reused = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(7,))],
        config=config,
    )

    assert reused.path == first.path
    assert len(calls) == first_call_count
    assert reused.implementation_fingerprint == legacy["implementation_sha256"]


def test_bfloat16_runtime_algebra_is_checked_in_runtime_dtype(tmp_path) -> None:
    model = _parent(seed=250).to(dtype=torch.bfloat16)
    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(7,))],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            logit_rtol=0.5,
            logit_atol=1.0,
            require_top1_agreement=False,
        ),
        input_device="cpu",
    )
    tensors = cache.load_batch(0).tensors
    for layer_index in range(2):
        prefix = f"prefill.layer.{layer_index:03d}"
        torch.testing.assert_close(
            torch.nn.functional.silu(tensors[f"{prefix}.mlp.gate"])
            * tensors[f"{prefix}.mlp.up"],
            tensors[f"{prefix}.mlp.post_swiglu"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            tensors[f"{prefix}.residual.pre"] + tensors[f"{prefix}.operator.output"],
            tensors[f"{prefix}.residual.post_operator"],
            rtol=0.0,
            atol=0.0,
        )


def test_parent_fingerprint_binds_runtime_buffers_not_loader_path() -> None:
    model = _parent(seed=275)
    before = fingerprint_lfm2_parent(model)
    model.config._name_or_path = "C:/a/different/local/checkpoint/path"
    renamed = fingerprint_lfm2_parent(model)
    assert renamed == before

    buffers = tuple(
        (name, value)
        for name, value in model.named_buffers()
        if value.is_floating_point() and value.numel() > 0
    )
    assert buffers
    name, runtime_buffer = buffers[0]
    assert name
    with torch.no_grad():
        runtime_buffer.reshape(-1)[0].add_(0.25)
    changed = fingerprint_lfm2_parent(model)
    assert changed.state_dict_sha256 == before.state_dict_sha256
    assert changed.runtime_buffer_sha256 != before.runtime_buffer_sha256
    assert changed.identity_sha256 != before.identity_sha256


def test_cache_corruption_fails_closed_and_is_not_recomputed(tmp_path) -> None:
    model = _parent(seed=303)
    calls = _instrument_forwards(model)
    config = Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2)
    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(9,))],
        config=config,
    )
    call_count = len(calls)
    trace_path = cache.path / "batches" / "batch-000000" / TRACE_FILENAME
    corrupted = bytearray(trace_path.read_bytes())
    corrupted[-1] ^= 0x01
    trace_path.write_bytes(corrupted)

    with pytest.raises(
        Lfm2CircuitTraceCorruptionError,
        match="trace file hash changed",
    ):
        open_lfm2_circuit_trace_cache(cache.path)
    with pytest.raises(Lfm2CircuitTraceCorruptionError):
        build_lfm2_circuit_trace_cache(
            tmp_path,
            model,
            [_batch(forced=(9,))],
            config=config,
        )
    assert len(calls) == call_count
    assert trace_path.read_bytes() == corrupted


def test_mislabeled_short_conv_and_reserved_inputs_fail_before_forward(
    tmp_path,
) -> None:
    model = _parent(seed=404)
    calls = _instrument_forwards(model)
    model.model.layers[0].is_attention_layer = True
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="canonical topology validation",
    ):
        build_lfm2_circuit_trace_cache(
            tmp_path / "mislabeled",
            model,
            [_batch(forced=(5,))],
        )
    assert calls == []

    clean = _parent(seed=405)
    clean_calls = _instrument_forwards(clean)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="unsupported or reserved fields",
    ):
        build_lfm2_circuit_trace_cache(
            tmp_path / "reserved",
            clean,
            [{**_batch(forced=(5,)), "past_key_values": torch.tensor(0)}],
        )
    assert clean_calls == []

    padded = _parent(seed=406)
    padded_calls = _instrument_forwards(padded)
    padded_batch = _batch(forced=(5,))
    padded_batch["attention_mask"] = torch.tensor([[1, 1, 1, 0]])
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="must be unpadded",
    ):
        build_lfm2_circuit_trace_cache(
            tmp_path / "padded",
            padded,
            [padded_batch],
        )
    assert padded_calls == []

    multi = _parent(seed=407)
    multi_calls = _instrument_forwards(multi)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="exactly one unpadded sequence",
    ):
        build_lfm2_circuit_trace_cache(
            tmp_path / "multi",
            multi,
            [
                {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
                    "attention_mask": torch.ones((2, 3), dtype=torch.long),
                    "forced_decode_ids": torch.tensor([[7], [8]], dtype=torch.long),
                }
            ],
        )
    assert multi_calls == []
