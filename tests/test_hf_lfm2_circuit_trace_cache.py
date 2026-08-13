from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from types import MethodType

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
from experiments import run_lfm2_residual_attention_fold as residual  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    AUTHENTICATED_SESSION_SCHEMA,
    BATCH_RECORD_FILENAME,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    TRACE_FILENAME,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCaptureError,
    Lfm2CircuitTraceCorruptionError,
    build_lfm2_circuit_trace_cache,
    extract_lfm2_legacy_cache_recapture_inputs,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallabilityAdapter,
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


def _legacy_v1_implementation() -> dict[str, object]:
    current = trace_cache.implementation_identity()
    body = {
        "schema": "neural-forge-lfm2-circuit-implementation-v1",
        "collector_source_sha256": current["collector_source_sha256"],
        "canonical_contract_source_sha256": current["canonical_contract_source_sha256"],
        "transformers_lfm2_source_sha256": current["transformers_lfm2_source_sha256"],
        "torch_version": current["torch_version"],
        "transformers_version": current["transformers_version"],
        "safetensors_version": current["safetensors_version"],
    }
    return {
        **body,
        "implementation_sha256": trace_cache._json_sha256(body),
    }


def _republish_as_legacy_v1_cache(
    root: Path,
    model,
    batches,
    config: Lfm2CircuitTraceCaptureConfig,
    source,
) -> Path:
    prepared = trace_cache._prepare_batches(torch, batches, config)
    _layers, _contracts, topology = trace_cache._layer_topology(model)
    device = torch.device("cpu")
    manifest = trace_cache._manifest_for(
        fingerprint_lfm2_parent(model),
        trace_cache._corpus_identity(prepared),
        _legacy_v1_implementation(),
        topology,
        config,
        device,
        trace_cache._execution_identity(model, device),
    )
    cache_path = root / manifest["cache_key"]
    batches_path = cache_path / "batches"
    batches_path.mkdir(parents=True)
    trace_cache._write_canonical_json(
        cache_path / trace_cache.MANIFEST_FILENAME,
        manifest,
    )
    entries = []
    for prepared_batch in prepared:
        source_batch = source.load_batch(prepared_batch.index)
        entries.append(
            trace_cache._publish_batch(
                torch,
                batches_path / f"batch-{prepared_batch.index:06d}",
                manifest,
                prepared_batch,
                source_batch.tensors,
                source_batch.record["capture"],
            )
        )
    trace_cache._write_canonical_json(
        cache_path / trace_cache.COMPLETION_FILENAME,
        trace_cache._completion_for(manifest, entries),
    )
    return cache_path


def _reseal_manifest_implementation(cache_path: Path, mutate) -> None:
    manifest_path = cache_path / trace_cache.MANIFEST_FILENAME
    manifest = trace_cache._read_canonical_json(manifest_path)
    implementation = manifest["implementation"]
    mutate(implementation)
    receipt = implementation.get("runtime_semantics_receipt")
    if isinstance(receipt, dict):
        implementation["runtime_semantics_receipt_sha256"] = trace_cache._json_sha256(
            receipt
        )
    implementation_body = {
        name: value
        for name, value in implementation.items()
        if name != "implementation_sha256"
    }
    implementation["implementation_sha256"] = trace_cache._json_sha256(
        implementation_body
    )
    manifest_body = {
        name: value for name, value in manifest.items() if name != "manifest_sha256"
    }
    manifest["manifest_sha256"] = trace_cache._json_sha256(manifest_body)
    manifest_path.write_bytes(trace_cache._canonical_json(manifest) + b"\n")


def _module_modes(model):
    return {name: bool(module.training) for name, module in model.named_modules()}


def _instrument_forwards(monkeypatch, model):
    calls: list[dict[str, object]] = []
    original = trace_cache._forward

    def observed(actual_model, *args, **kwargs):
        if actual_model is model:
            input_ids = kwargs.get("input_ids")
            calls.append(
                {
                    "use_cache": kwargs.get("use_cache"),
                    "has_past": kwargs.get("past_key_values") is not None,
                    "input_length": int(input_ids.shape[1]),
                    "grad_enabled": torch.is_grad_enabled(),
                    "training": bool(actual_model.training),
                }
            )
        return original(actual_model, *args, **kwargs)

    monkeypatch.setattr(trace_cache, "_forward", observed)
    return calls


def test_real_sealed_cache_streaming_vocabulary_margin_v2_has_zero_new_parent_forwards(
    tmp_path,
) -> None:
    from neural_forge.hf_lfm2_balanced_circuit_basis import (
        build_lfm2_circuit_balanced_basis,
    )

    model = _parent(seed=607)
    batches = [
        {
            "input_ids": torch.tensor(
                [[1, 2, 3, 4 + index]],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "forced_decode_ids": torch.tensor(
                [[8 + index, 12 + index]],
                dtype=torch.long,
            ),
        }
        for index in range(4)
    ]
    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=3,
            max_decode_steps=2,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    parent_forwards_after_publication = 0

    def forbidden_forward(*_args, **_kwargs):
        nonlocal parent_forwards_after_publication
        parent_forwards_after_publication += 1
        raise AssertionError("streaming margin adapter executed the parent")

    model.forward = forbidden_forward
    with cache.selective_authenticated_session() as session:
        result = build_lfm2_circuit_balanced_basis(
            model,
            cache,
            4,
            selective_session=session,
            validation_fraction=0.25,
            split_seed="real-sealed-cache-streaming-margin-v2",
            max_rows_per_block=8,
            vocabulary_margin_observability=True,
            vocabulary_margin_top_k=3,
            streaming_vocabulary_margin_v2=True,
            streaming_vocabulary_margin_pair_chunk_size=5,
        )
        margin = result.as_dict()["vocabulary_margin_observability"]
        adapter = margin["adapter"]
        authentication = adapter["cache_authentication"]
        assert authentication["cache_key"] == cache.cache_key
        assert authentication["manifest_sha256"] == cache.manifest["manifest_sha256"]
        assert (
            authentication["completion_sha256"]
            == session.as_dict()["completion_sha256"]
        )
        assert authentication["selective_session_sha256"] == (session.evidence_sha256)
        assert (
            adapter["inventory"]["complete_record_phase_inventory_authenticated"]
            is True
        )
        assert adapter["fit"]["all_declared_rows_consumed"] is True
        assert adapter["fit"]["selection_pass"]["row_cap_or_subsample_applied"] is False
        assert adapter["retention"]["numeric_evidence_bound"] == (
            "H*H+H-plus-one-phase"
        )
        assert margin["heldout_projection_gate"]["parent_forward_calls"] == 0
        assert result.as_dict()["model_forward_calls"] == 0
    assert parent_forwards_after_publication == 0


def test_real_selective_preallocation_is_bit_exact_with_legacy_rows(tmp_path) -> None:
    model = _parent(seed=608)
    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(5, 6)), _batch(forced=(7, 8))],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=3,
            max_decode_steps=2,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    splits = ("selection", "fit")
    legacy = residual.extract_trace_rows(cache, batch_splits=splits)

    with cache.selective_authenticated_session() as session:
        preallocated = residual.extract_trace_rows(
            cache,
            batch_splits=splits,
            selective_session=session,
        )

    assert preallocated.paths == legacy.paths
    assert preallocated.path_splits == legacy.path_splits
    assert preallocated.layer_kinds == legacy.layer_kinds
    assert preallocated.trace_rows_sha256 == legacy.trace_rows_sha256
    for layer_index, fields in legacy.layer_fields.items():
        assert set(preallocated.layer_fields[layer_index]) == set(fields)
        for field, expected in fields.items():
            torch.testing.assert_close(
                preallocated.field(layer_index, field),
                expected,
                rtol=0.0,
                atol=0.0,
            )


def test_real_lfm2_cache_captures_bcx_attention_mlp_and_forced_decode(
    tmp_path,
    monkeypatch,
) -> None:
    model = _parent()
    assert not hasattr(model.config, "head_dim")
    calls = _instrument_forwards(monkeypatch, model)
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


def test_opt_in_causal_circuit_cache_captures_installable_operators(
    tmp_path,
) -> None:
    model = _parent(seed=109)
    config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=3,
        max_decode_steps=4,
        capture_causal_circuit_evidence=True,
    )

    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch()],
        config=config,
        input_device="cpu",
    )

    capture = cache.manifest["capture"]
    assert capture["capture_causal_circuit_evidence"] is True
    assert capture["attention_pre_rope_qk"] is True
    assert capture["attention_exact_causal_scores_and_probabilities"] is True
    assert capture["short_conv_exact_fir_state_and_response"] is True
    assert capture["final_normalization_boundary"] is True

    batch = cache.load_batch(0)
    tensors = batch.tensors
    assert tensors["prefill.layer.001.attention.q_normalized_pre_rope"].shape == (
        1,
        2,
        4,
        4,
    )
    assert tensors["prefill.layer.001.attention.k_normalized_pre_rope"].shape == (
        1,
        1,
        4,
        4,
    )
    assert tensors["prefill.layer.001.attention.rope_cos"].shape == (1, 4, 4)
    assert tensors["prefill.layer.001.attention.rope_sin"].shape == (1, 4, 4)
    assert tensors["prefill.layer.001.attention.key_history"].shape == (
        1,
        1,
        4,
        4,
    )
    assert tensors["prefill.layer.001.attention.qk_scores"].shape == (
        1,
        2,
        3,
        4,
    )
    assert tensors["prefill.layer.001.attention.probabilities"].shape == (
        1,
        2,
        3,
        4,
    )
    assert (
        tensors["prefill.layer.001.attention.qk_scores"].dtype
        == tensors["prefill.layer.001.attention.q"].dtype
    )
    assert (
        tensors["prefill.layer.001.attention.additive_mask"].dtype
        == tensors["prefill.layer.001.attention.q"].dtype
    )
    assert (
        tensors["prefill.layer.001.attention.probabilities"].dtype
        == tensors["prefill.layer.001.attention.q"].dtype
    )
    assert (
        tensors["prefill.layer.001.attention.context_from_probabilities"].dtype
        == tensors["prefill.layer.001.attention.context"].dtype
    )
    torch.testing.assert_close(
        tensors["prefill.layer.001.attention.probabilities"].sum(dim=-1),
        torch.ones(
            (1, 2, 3),
            dtype=tensors["prefill.layer.001.attention.probabilities"].dtype,
        ),
    )
    torch.testing.assert_close(
        tensors["prefill.layer.001.attention.context_from_probabilities"],
        tensors["prefill.layer.001.attention.context"].to(torch.float32),
        rtol=2e-2,
        atol=2e-3,
    )
    assert tensors["decode.000.layer.001.attention.key_history"].shape == (
        1,
        1,
        5,
        4,
    )
    assert tensors["decode.001.layer.001.attention.key_history"].shape == (
        1,
        1,
        6,
        4,
    )

    assert tensors["prefill.layer.000.short_conv.bx_history"].shape == (
        1,
        4,
        8,
    )
    assert tensors["prefill.layer.000.short_conv.pre_bx_state"].shape == (
        1,
        8,
        3,
    )
    assert tensors["prefill.layer.000.short_conv.fir_kernel"].shape == (8, 3)
    assert tensors["prefill.layer.000.short_conv.fir_bias"].shape == (8,)
    assert tensors["prefill.layer.000.short_conv.fir_has_bias"].tolist() == [0]
    assert tensors["decode.000.layer.000.short_conv.pre_bx_state"].shape == (
        1,
        8,
        3,
    )
    torch.testing.assert_close(
        tensors["prefill.layer.000.short_conv.context_from_fir"][:, -3:, :],
        tensors["prefill.layer.000.short_conv.context"].to(torch.float32),
        rtol=2e-2,
        atol=2e-3,
    )

    assert tensors["prefill.output.final_norm_input"].shape == (1, 3, 8)
    assert tensors["prefill.output.final_norm_output"].shape == (1, 3, 8)
    torch.testing.assert_close(
        tensors["prefill.output.final_norm_output"],
        tensors["prefill.output.lm_head_input"],
        rtol=0.0,
        atol=0.0,
    )
    assert tensors["decode.001.output.final_norm_input"].shape == (1, 1, 8)

    diagnostics = batch.record["capture"]["phase_diagnostics"]["prefill"]
    assert (
        diagnostics["layer.001"]["attention_context_reconstructed_from_probabilities"]
        is True
    )
    assert (
        diagnostics["layer.001"]["attention_context_authenticated_on_execution_device"]
        is True
    )
    assert (
        diagnostics["layer.001"][
            "attention_evidence_serialized_after_device_authentication"
        ]
        is True
    )
    assert (
        diagnostics["layer.001"]["attention_reconstruction_execution_device_type"]
        == "cpu"
    )
    assert diagnostics["layer.000"]["short_conv_context_reconstructed_from_fir"] is True
    assert diagnostics["output"]["final_norm_boundary_captured"] is True


def test_opt_in_attention_reconstruction_uses_runtime_bfloat16_path(
    tmp_path,
) -> None:
    model = _parent(seed=113).to(dtype=torch.bfloat16)
    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(5,))],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )

    tensors = cache.load_batch(0).tensors
    prefix = "prefill.layer.001.attention"
    assert tensors[f"{prefix}.qk_scores"].dtype == torch.bfloat16
    assert tensors[f"{prefix}.additive_mask"].dtype == torch.bfloat16
    assert tensors[f"{prefix}.probabilities"].dtype == torch.bfloat16
    assert tensors[f"{prefix}.context_from_probabilities"].dtype == torch.bfloat16
    torch.testing.assert_close(
        tensors[f"{prefix}.context_from_probabilities"],
        tensors[f"{prefix}.context"],
        rtol=2e-2,
        atol=2e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_device_local_bfloat16_attention_avoids_cpu_false_negative() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 support is required")
    query_heads = 16
    key_value_heads = 8
    sequence = 256
    head_dim = 64
    groups = query_heads // key_value_heads
    generator = torch.Generator(device="cpu").manual_seed(7)
    query = torch.randn(
        (1, query_heads, sequence, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    key = torch.randn(
        (1, key_value_heads, sequence, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    value = torch.randn(
        (1, key_value_heads, sequence, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    additive_mask = torch.triu(
        torch.full(
            (1, 1, sequence, sequence),
            float("-inf"),
            dtype=torch.float32,
        ),
        diagonal=1,
    ).to(torch.bfloat16)

    cpu_replay = trace_cache._reconstruct_attention_causal_path(
        torch,
        query=query,
        key_history=key,
        value_history=value,
        additive_mask=additive_mask,
        groups=groups,
        sample_count=sequence,
        head_dim=head_dim,
    )["reconstructed_context"]
    gpu_inputs = {
        "query": query.to("cuda"),
        "key_history": key.to("cuda"),
        "value_history": value.to("cuda"),
        "additive_mask": additive_mask.to("cuda"),
    }
    gpu_replay = trace_cache._reconstruct_attention_causal_path(
        torch,
        **gpu_inputs,
        groups=groups,
        sample_count=sequence,
        head_dim=head_dim,
    )["reconstructed_context"]
    gpu_repeat = trace_cache._reconstruct_attention_causal_path(
        torch,
        **gpu_inputs,
        groups=groups,
        sample_count=sequence,
        head_dim=head_dim,
    )["reconstructed_context"]

    assert torch.equal(gpu_replay, gpu_repeat)
    assert not torch.allclose(
        cpu_replay,
        gpu_replay.cpu(),
        rtol=2e-2,
        atol=2e-3,
    )
    assert (
        int(
            (
                ~torch.isclose(
                    cpu_replay,
                    gpu_replay.cpu(),
                    rtol=2e-2,
                    atol=2e-3,
                )
            )
            .sum()
            .item()
        )
        >= 1
    )


def test_causal_circuit_evidence_rejects_noncanonical_trace_mode() -> None:
    with pytest.raises(ValueError, match="cached_verified"):
        Lfm2CircuitTraceCaptureConfig(
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
            capture_causal_circuit_evidence=True,
        )


def test_full_prefix_canonical_mode_never_accepts_cached_logits_and_keeps_paths(
    tmp_path,
    monkeypatch,
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
    calls = _instrument_forwards(monkeypatch, model)
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
    monkeypatch,
) -> None:
    model = _parent(seed=202)
    calls = _instrument_forwards(monkeypatch, model)
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


def test_v1_cache_is_migration_only_and_cannot_bridge_v2_reuse(
    tmp_path,
    monkeypatch,
) -> None:
    model = _parent(seed=1202)
    batches = (_batch(forced=(7,)), _batch(forced=(8,)))
    config = Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2)
    source = build_lfm2_circuit_trace_cache(
        tmp_path / "source-v2",
        model,
        batches,
        config=config,
        input_device="cpu",
    )
    mixed_root = tmp_path / "mixed"
    legacy_path = _republish_as_legacy_v1_cache(
        mixed_root,
        model,
        batches,
        config,
        source,
    )
    manifest = trace_cache._read_canonical_json(
        legacy_path / trace_cache.MANIFEST_FILENAME
    )
    completion = trace_cache._read_canonical_json(
        legacy_path / trace_cache.COMPLETION_FILENAME
    )

    for verify in (True, False):
        with pytest.raises(
            Lfm2CircuitTraceCorruptionError,
            match="implementation identity",
        ):
            open_lfm2_circuit_trace_cache(legacy_path, verify=verify)

    extracted = extract_lfm2_legacy_cache_recapture_inputs(
        legacy_path,
        expected_cache_key=manifest["cache_key"],
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_completion_sha256=completion["completion_sha256"],
        expected_implementation_sha256=manifest["implementation"][
            "implementation_sha256"
        ],
    )
    assert extracted.cache_key == legacy_path.name
    assert len(extracted.batches) == len(batches)
    for actual, expected in zip(extracted.batches, batches):
        torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
        torch.testing.assert_close(
            actual["attention_mask"],
            expected["attention_mask"],
        )
        torch.testing.assert_close(
            actual["forced_decode_ids"],
            expected["forced_decode_ids"],
        )
    extracted.batches[0]["input_ids"].zero_()
    repeated = extract_lfm2_legacy_cache_recapture_inputs(
        legacy_path,
        expected_cache_key=manifest["cache_key"],
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_completion_sha256=completion["completion_sha256"],
        expected_implementation_sha256=manifest["implementation"][
            "implementation_sha256"
        ],
    )
    assert torch.equal(repeated.batches[0]["input_ids"], batches[0]["input_ids"])
    with pytest.raises(
        Lfm2CircuitTraceCorruptionError,
        match="pinned cache key",
    ):
        extract_lfm2_legacy_cache_recapture_inputs(
            legacy_path,
            expected_cache_key="0" * 64,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_completion_sha256=completion["completion_sha256"],
            expected_implementation_sha256=manifest["implementation"][
                "implementation_sha256"
            ],
        )

    calls = _instrument_forwards(monkeypatch, model)
    current = build_lfm2_circuit_trace_cache(
        mixed_root,
        model,
        batches,
        config=config,
        input_device="cpu",
    )
    assert current.path != legacy_path
    assert current.manifest["implementation"]["schema"] == (
        "neural-forge-lfm2-circuit-implementation-v2"
    )
    assert calls


def test_v2_cache_binds_full_capture_runtime_receipt_and_verifier_source(
    tmp_path,
) -> None:
    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        _parent(seed=1203),
        [_batch(forced=(7,))],
        config=Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2),
        input_device="cpu",
    )
    implementation = cache.manifest["implementation"]
    receipt = implementation["runtime_semantics_receipt"]
    assert implementation["schema"] == ("neural-forge-lfm2-circuit-implementation-v2")
    assert receipt["schema"] == trace_cache.LFM2_RUNTIME_SEMANTICS_RECEIPT_SCHEMA
    assert implementation["runtime_semantics_receipt_schema"] == receipt["schema"]
    assert implementation["runtime_semantics_receipt_sha256"] == (
        trace_cache._json_sha256(receipt)
    )
    assert implementation["runtime_semantics_receipt_source_sha256"] == (
        trace_cache.lfm2_runtime_semantics_receipt_source_sha256()
    )
    assert open_lfm2_circuit_trace_cache(cache.path, verify=False).cache_key == (
        cache.cache_key
    )
    assert open_lfm2_circuit_trace_cache(cache.path, verify=True).cache_key == (
        cache.cache_key
    )


def test_v2_cache_rejects_self_resealed_runtime_receipt_tamper(tmp_path) -> None:
    source = build_lfm2_circuit_trace_cache(
        tmp_path / "source",
        _parent(seed=1204),
        [_batch(forced=(7,))],
        config=Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2),
        input_device="cpu",
    )

    def delete_receipt(implementation):
        implementation.pop("runtime_semantics_receipt")

    def change_receipt(implementation):
        implementation["runtime_semantics_receipt"]["tampered"] = True

    def inject_implementation_field(implementation):
        implementation["unrecognized_runtime_authority"] = "injected"

    for name, mutate in (
        ("delete", delete_receipt),
        ("change", change_receipt),
        ("inject", inject_implementation_field),
    ):
        candidate = tmp_path / name / source.cache_key
        candidate.parent.mkdir()
        shutil.copytree(source.path, candidate)
        _reseal_manifest_implementation(candidate, mutate)
        with pytest.raises(Lfm2CircuitTraceCorruptionError):
            open_lfm2_circuit_trace_cache(candidate, verify=False)


def test_capture_receipt_source_identity_moves_key_and_old_identity_refuses(
    tmp_path,
    monkeypatch,
) -> None:
    model = _parent(seed=1205)
    batch = _batch(forced=(7,))
    config = Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2)
    canonical = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [batch],
        config=config,
        input_device="cpu",
    )
    canonical_implementation = canonical.manifest["implementation"]
    with monkeypatch.context() as context:
        context.setattr(
            trace_cache,
            "lfm2_runtime_semantics_receipt_source_sha256",
            lambda: "0" * 64,
        )
        changed = build_lfm2_circuit_trace_cache(
            tmp_path,
            model,
            [batch],
            config=config,
            input_device="cpu",
        )
    changed_implementation = changed.manifest["implementation"]
    assert changed.cache_key != canonical.cache_key
    assert (
        changed_implementation["runtime_semantics_receipt"]
        == (canonical_implementation["runtime_semantics_receipt"])
    )
    assert (
        changed_implementation["runtime_semantics_receipt_sha256"]
        == (canonical_implementation["runtime_semantics_receipt_sha256"])
    )
    assert changed_implementation["runtime_semantics_receipt_source_sha256"] == (
        "0" * 64
    )
    with pytest.raises(
        Lfm2CircuitTraceCorruptionError,
        match="implementation differs",
    ):
        open_lfm2_circuit_trace_cache(changed.path, verify=False)


def test_capture_rejects_root_instance_call_override_before_forward(
    tmp_path,
    monkeypatch,
) -> None:
    model = _parent(seed=1206)
    forward_calls = 0

    def poisoned_root_forward(_self, *_args, **_kwargs):
        return torch.zeros(1)

    def forbidden_cache_forward(*_args, **_kwargs):
        nonlocal forward_calls
        forward_calls += 1
        raise AssertionError("cache capture reached the parent")

    model.forward = MethodType(poisoned_root_forward, model)
    monkeypatch.setattr(trace_cache, "_forward", forbidden_cache_forward)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="before cache capture",
    ):
        build_lfm2_circuit_trace_cache(
            tmp_path,
            model,
            [_batch(forced=(7,))],
            input_device="cpu",
        )
    assert forward_calls == 0


def test_capture_rejects_foreign_same_name_module_model_class_before_forward(
    tmp_path,
    monkeypatch,
) -> None:
    model = _parent(seed=1207)
    native_class = type(model.model.layers[0].feed_forward)

    def poisoned_forward(_self, hidden_states):
        return torch.zeros_like(hidden_states)

    foreign_class = type(
        native_class.__name__,
        (native_class,),
        {
            "__module__": native_class.__module__,
            "forward": poisoned_forward,
        },
    )
    foreign = foreign_class(model.config)
    foreign.load_state_dict(model.model.layers[0].feed_forward.state_dict())
    model.model.layers[0].feed_forward = foreign
    forward_calls = 0

    def forbidden_cache_forward(*_args, **_kwargs):
        nonlocal forward_calls
        forward_calls += 1
        raise AssertionError("cache capture reached the parent")

    monkeypatch.setattr(trace_cache, "_forward", forbidden_cache_forward)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="before cache capture",
    ):
        build_lfm2_circuit_trace_cache(
            tmp_path,
            model,
            [_batch(forced=(7,))],
            input_device="cpu",
        )
    assert forward_calls == 0


@pytest.mark.parametrize(
    ("boundary", "expected_label"),
    (
        ("after_capture", "after batch 0"),
        ("between_batches", "before batch 1"),
        ("before_publication", "before cache publication"),
    ),
)
def test_capture_reauthenticates_model_at_every_batch_and_publication_boundary(
    tmp_path,
    monkeypatch,
    boundary,
    expected_label,
) -> None:
    model = _parent(seed=1208)
    batches = [_batch(forced=(7,)), _batch(forced=(8,))]
    mutated = False

    def mutate_model_once() -> None:
        nonlocal mutated
        if mutated:
            return
        mutated = True
        norm = model.model.layers[0].ffn_norm

        def poisoned_norm_forward(_self, hidden_states):
            return torch.zeros_like(hidden_states)

        norm.forward = MethodType(poisoned_norm_forward, norm)

    if boundary == "after_capture":
        original_capture = trace_cache._capture_batch

        def capture_then_mutate(*args, **kwargs):
            result = original_capture(*args, **kwargs)
            mutate_model_once()
            return result

        monkeypatch.setattr(trace_cache, "_capture_batch", capture_then_mutate)
    elif boundary == "between_batches":
        original_publish = trace_cache._publish_batch

        def publish_then_mutate(*args, **kwargs):
            result = original_publish(*args, **kwargs)
            batch = args[3]
            if batch.index == 0:
                mutate_model_once()
            return result

        monkeypatch.setattr(trace_cache, "_publish_batch", publish_then_mutate)
    else:
        original_write = trace_cache._write_canonical_json

        def write_then_mutate(path, value):
            result = original_write(path, value)
            if path.name == trace_cache.COMPLETION_FILENAME:
                mutate_model_once()
            return result

        monkeypatch.setattr(trace_cache, "_write_canonical_json", write_then_mutate)

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match=expected_label,
    ):
        build_lfm2_circuit_trace_cache(
            tmp_path,
            model,
            batches,
            config=Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2),
            input_device="cpu",
        )
    assert mutated is True
    assert not any(
        path.name.startswith(".lfm2-circuit-stage-") for path in tmp_path.iterdir()
    )


def test_fresh_cache_capture_rejects_poisoned_silu_before_any_forward() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        r"""
        import tempfile
        from pathlib import Path

        import torch
        import transformers

        import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache
        import neural_forge.hf_lfm2_runtime_semantics_receipt as runtime_receipt

        assert not runtime_receipt._CANONICAL_RECEIPT
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
        parent = transformers.Lfm2ForCausalLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "forced_decode_ids": torch.tensor([[5]], dtype=torch.long),
        }
        forward_calls = 0

        def poisoned_silu(value, *_args, **_kwargs):
            return torch.zeros_like(value)

        def forbidden_forward(*_args, **_kwargs):
            global forward_calls
            forward_calls += 1
            raise AssertionError("poisoned cache reached parent inference")

        torch.nn.functional.silu = poisoned_silu
        trace_cache._forward = forbidden_forward
        with tempfile.TemporaryDirectory() as root:
            try:
                trace_cache.build_lfm2_circuit_trace_cache(
                    root,
                    parent,
                    [batch],
                    input_device="cpu",
                )
            except trace_cache.Lfm2CircuitTraceCaptureError as exc:
                assert "source authentication" in str(exc).lower(), str(exc)
                print("REJECTED poisoned_silu_before_capture")
            else:
                raise AssertionError("poisoned capture runtime was accepted")
            assert not tuple(Path(root).iterdir())
        assert forward_calls == 0
        """
    )
    environment = dict(os.environ)
    path_entries = [str(repo_root / "src"), str(repo_root)]
    existing = environment.get("PYTHONPATH")
    if existing:
        path_entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(path_entries)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "REJECTED poisoned_silu_before_capture" in completed.stdout


def _sealed_complete_causal_cache_implementation() -> dict[str, str]:
    return {
        "schema": "neural-forge-lfm2-circuit-implementation-v1",
        "collector_source_sha256": (
            "2c3fc8a337da325564369976f056337de197e3c573203a250cfd5fe626daf349"
        ),
        "canonical_contract_source_sha256": (
            "5956f90939ab57dcabb32377d795b00eda0d948666c63b5688259b5680aa738a"
        ),
        "transformers_lfm2_source_sha256": (
            "2ec8be8af6b65482f54d29e52da9765b863ce37b65846423dfbfda72f944f150"
        ),
        "torch_version": "2.11.0+cu128",
        "transformers_version": "5.12.1",
        "safetensors_version": "0.8.0",
        "implementation_sha256": (
            "00700dc26c52f54e8abec5eb7f213d5123a9f55466b286cf3caaabe0a2ac98d5"
        ),
    }


def _sealed_compiler_cache_implementation() -> dict[str, str]:
    return {
        "schema": "neural-forge-lfm2-circuit-implementation-v1",
        "collector_source_sha256": (
            "71e99ee9b1c22ae893b76df90a3de93341e1b894ea7c39606f98d62e18aa13bf"
        ),
        "canonical_contract_source_sha256": (
            "5956f90939ab57dcabb32377d795b00eda0d948666c63b5688259b5680aa738a"
        ),
        "transformers_lfm2_source_sha256": (
            "2ec8be8af6b65482f54d29e52da9765b863ce37b65846423dfbfda72f944f150"
        ),
        "torch_version": "2.11.0+cu128",
        "transformers_version": "5.12.1",
        "safetensors_version": "0.8.0",
        "implementation_sha256": (
            "eb3bddeb3414c80b252b37dfa14b991e9066894f2534138dc9c118c9e711bff4"
        ),
    }


def _rehashed_implementation(
    value: dict[str, str],
    **changes: str,
) -> dict[str, str]:
    body = {key: item for key, item in value.items() if key != "implementation_sha256"}
    body.update(changes)
    return {
        **body,
        "implementation_sha256": trace_cache._json_sha256(body),
    }


def _sealed_sharded_reader_cache_implementation() -> dict[str, str]:
    identity = _rehashed_implementation(
        _sealed_complete_causal_cache_implementation(),
        collector_source_sha256=(
            "16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945"
        ),
        canonical_contract_source_sha256=(
            "789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d"
        ),
    )
    assert identity["implementation_sha256"] == (
        "732ca9ce7bd135d1e14d99fdc55e3aa9b1b40ad68c3e5a9d6c1dd2842b35c227"
    )
    return identity


def _runtime_reader_implementation(torch_version: str) -> dict[str, str]:
    return _rehashed_implementation(
        _sealed_complete_causal_cache_implementation(),
        collector_source_sha256=trace_cache._source_file_sha256(
            Path(trace_cache.__file__).resolve(),
            "collector",
        ),
        torch_version=torch_version,
    )


def _current_heterogeneous_reader_implementation() -> dict[str, str]:
    return _rehashed_implementation(
        _runtime_reader_implementation("2.11.0+cu128"),
        canonical_contract_source_sha256=(
            "789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d"
        ),
    )


@pytest.mark.parametrize(
    "stored_implementation",
    (
        _sealed_sharded_reader_cache_implementation,
        _sealed_complete_causal_cache_implementation,
    ),
    ids=("base-extension-sharded-reader", "ahb-complete-causal-a0003"),
)
def test_receipt_required_reader_refuses_legacy_implementation_compatibility(
    stored_implementation,
) -> None:
    stored = stored_implementation()
    current = trace_cache.implementation_identity()

    assert stored["schema"] == "neural-forge-lfm2-circuit-implementation-v1"
    assert current["schema"] == "neural-forge-lfm2-circuit-implementation-v2"
    assert not trace_cache._compatible_implementation_identity(stored, current)


def test_exact_implementation_identity_remains_compatible() -> None:
    current = trace_cache.implementation_identity()

    assert trace_cache._compatible_implementation_identity(current, current)


@pytest.mark.parametrize("torch_version", ("2.11.0+cu128", "2.6.0+cu118"))
def test_sealed_complete_causal_cache_rejects_unlisted_current_readers(
    torch_version,
) -> None:
    stored = _sealed_complete_causal_cache_implementation()
    stored_body = dict(stored)
    stored_sha256 = stored_body.pop("implementation_sha256")
    assert trace_cache._json_sha256(stored_body) == stored_sha256
    current = _runtime_reader_implementation(torch_version)

    assert not trace_cache._compatible_implementation_identity(stored, current)


def test_sealed_compiler_cache_rejects_current_collector_with_native_canonical() -> (
    None
):
    stored = _sealed_compiler_cache_implementation()
    stored_body = dict(stored)
    stored_sha256 = stored_body.pop("implementation_sha256")
    assert trace_cache._json_sha256(stored_body) == stored_sha256
    current = trace_cache.implementation_identity()

    assert not trace_cache._compatible_implementation_identity(stored, current)


def test_sealed_sharded_cache_cannot_bypass_capture_receipt_requirement() -> None:
    stored = _sealed_sharded_reader_cache_implementation()
    current = trace_cache.implementation_identity()

    assert not trace_cache._compatible_implementation_identity(stored, current)


def test_authentication_progress_reader_matches_exact_a0006_remote_identity() -> None:
    historical = _rehashed_implementation(
        _sealed_complete_causal_cache_implementation(),
        collector_source_sha256=(
            "23e47f4cc03091a4b70968ad0849294999e5d97f522462eb41663369f2bfc791"
        ),
        canonical_contract_source_sha256=(
            "789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d"
        ),
    )
    assert historical == {
        "schema": "neural-forge-lfm2-circuit-implementation-v1",
        "collector_source_sha256": (
            "23e47f4cc03091a4b70968ad0849294999e5d97f522462eb41663369f2bfc791"
        ),
        "canonical_contract_source_sha256": (
            "789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d"
        ),
        "transformers_lfm2_source_sha256": (
            "2ec8be8af6b65482f54d29e52da9765b863ce37b65846423dfbfda72f944f150"
        ),
        "torch_version": "2.11.0+cu128",
        "transformers_version": "5.12.1",
        "safetensors_version": "0.8.0",
        "implementation_sha256": (
            "d8af5f9df4a50462faeca9b3b5d41c45d8c6b81afcf604e61106e74de16f4f41"
        ),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("collector_source_sha256", "a" * 64),
        (
            "canonical_contract_source_sha256",
            "5956f90939ab57dcabb32377d795b00eda0d948666c63b5688259b5680aa738a",
        ),
        ("transformers_lfm2_source_sha256", "b" * 64),
        ("torch_version", "2.11.1+cu128"),
        ("transformers_version", "5.12.2"),
        ("safetensors_version", "0.8.1"),
    ),
)
@pytest.mark.parametrize(
    "stored_implementation",
    (
        _sealed_sharded_reader_cache_implementation,
        _sealed_complete_causal_cache_implementation,
    ),
    ids=("base-extension-sharded-reader", "ahb-complete-causal-a0003"),
)
def test_authentication_progress_reader_rejects_current_identity_neighbors(
    stored_implementation,
    field,
    value,
) -> None:
    stored = stored_implementation()
    current = _rehashed_implementation(
        _current_heterogeneous_reader_implementation(),
        **{field: value},
    )

    assert not trace_cache._compatible_implementation_identity(stored, current)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("collector_source_sha256", "a" * 64),
        ("canonical_contract_source_sha256", "b" * 64),
        ("torch_version", "2.11.1+cu128"),
    ),
)
@pytest.mark.parametrize(
    "stored_implementation",
    (
        _sealed_sharded_reader_cache_implementation,
        _sealed_complete_causal_cache_implementation,
    ),
    ids=("base-extension-sharded-reader", "ahb-complete-causal-a0003"),
)
def test_authentication_progress_reader_rejects_stored_cache_neighbors(
    stored_implementation,
    field,
    value,
) -> None:
    stored = _rehashed_implementation(
        stored_implementation(),
        **{field: value},
    )
    current = _current_heterogeneous_reader_implementation()

    assert not trace_cache._compatible_implementation_identity(stored, current)


@pytest.mark.parametrize(
    "stored_implementation",
    (
        _sealed_sharded_reader_cache_implementation,
        _sealed_complete_causal_cache_implementation,
    ),
    ids=("base-extension-sharded-reader", "ahb-complete-causal-a0003"),
)
def test_authentication_progress_reader_rejects_current_implementation_tamper(
    stored_implementation,
) -> None:
    current = _current_heterogeneous_reader_implementation()
    current["implementation_sha256"] = "c" * 64

    assert not trace_cache._compatible_implementation_identity(
        stored_implementation(),
        current,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("torch_version", "2.6.1+cu118"),
        ("canonical_contract_source_sha256", "a" * 64),
        ("transformers_lfm2_source_sha256", "b" * 64),
        ("transformers_version", "5.12.2"),
        ("safetensors_version", "0.8.1"),
    ),
)
@pytest.mark.parametrize("torch_version", ("2.11.0+cu128", "2.6.0+cu118"))
def test_explicit_reader_pair_rejects_unaudited_runtime_changes(
    torch_version,
    field,
    value,
) -> None:
    stored = _sealed_complete_causal_cache_implementation()
    current = _rehashed_implementation(
        _runtime_reader_implementation(torch_version),
        **{field: value},
    )

    assert not trace_cache._compatible_implementation_identity(stored, current)


@pytest.mark.parametrize("torch_version", ("2.11.0+cu128", "2.6.0+cu118"))
@pytest.mark.parametrize("collector_sha256", ("a" * 64, "b" * 64))
def test_explicit_reader_pair_rejects_unrelated_current_collectors(
    torch_version,
    collector_sha256,
) -> None:
    stored = _sealed_complete_causal_cache_implementation()
    current = _rehashed_implementation(
        _runtime_reader_implementation(torch_version),
        collector_source_sha256=collector_sha256,
    )

    assert not trace_cache._compatible_implementation_identity(stored, current)


@pytest.mark.parametrize("torch_version", ("2.11.0+cu128", "2.6.0+cu118"))
def test_explicit_reader_pair_rejects_current_implementation_tamper(
    torch_version,
) -> None:
    stored = _sealed_complete_causal_cache_implementation()
    current = _runtime_reader_implementation(torch_version)
    current["implementation_sha256"] = "c" * 64

    assert not trace_cache._compatible_implementation_identity(stored, current)


def test_explicit_reader_pair_rejects_a_different_stored_implementation() -> None:
    stored = _rehashed_implementation(
        _sealed_complete_causal_cache_implementation(),
        torch_version="2.11.1+cu128",
    )
    current = _runtime_reader_implementation("2.6.0+cu118")

    assert not trace_cache._compatible_implementation_identity(stored, current)


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


def test_cached_top1_numeric_tie_is_verified_without_claiming_exact_agreement():
    cached = torch.tensor([[[13.3750, 13.3750, 9.0]]])
    full = torch.tensor([[[13.2500, 13.3125, 9.0]]])
    config = Lfm2CircuitTraceCaptureConfig(
        logit_rtol=0.0,
        logit_atol=0.25,
        require_top1_agreement=True,
        allow_top1_disagreement_within_numeric_error_envelope=True,
    )

    report = trace_cache._compare_logits(
        torch,
        cached,
        full,
        config=config,
        name="bf16 near tie",
    )

    assert report["numerically_close"] is True
    assert report["top1_agreement"] is False
    assert report["top1_decision_verified"] is True
    assert report["top1_disagreement_count"] == 1
    assert report["top1_disagreement_within_numeric_error_envelope"] is True
    assert report["first_top1_disagreement"] == {
        "row_index": 0,
        "cached_token_id": 0,
        "full_prefix_token_id": 1,
        "cached_winner_margin": 0.0,
        "full_prefix_winner_margin": 0.0625,
        "cached_preference_gap": 0.0,
        "full_prefix_preference_gap": 0.0625,
        "row_max_abs_error": 0.125,
        "winner_pair_error_envelope": 0.1875,
        "within_numeric_error_envelope": True,
    }

    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="within_error_envelope=True",
    ):
        trace_cache._compare_logits(
            torch,
            cached,
            full,
            config=Lfm2CircuitTraceCaptureConfig(
                logit_rtol=0.0,
                logit_atol=0.25,
                require_top1_agreement=True,
                allow_top1_disagreement_within_numeric_error_envelope=False,
            ),
            name="strict exact winner",
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


def test_cache_corruption_fails_closed_and_is_not_recomputed(
    tmp_path,
    monkeypatch,
) -> None:
    model = _parent(seed=303)
    calls = _instrument_forwards(monkeypatch, model)
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


def test_load_batch_reuses_the_verified_hash_under_an_open_fd_guard(
    tmp_path,
    monkeypatch,
) -> None:
    model = _parent(seed=606)
    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(9,))],
        config=Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2),
        input_device="cpu",
    )
    opened = open_lfm2_circuit_trace_cache(cache.path, verify=False)
    assert opened.verified_batch_guard_count == 0

    first = opened.load_batch(0)
    assert opened.verified_batch_guard_count == 1

    hashed: list[str] = []
    original_file_sha256 = trace_cache._file_sha256

    def counted_file_sha256(path):
        hashed.append(Path(path).name)
        return original_file_sha256(path)

    monkeypatch.setattr(trace_cache, "_file_sha256", counted_file_sha256)
    second = opened.load_batch(0)

    name = "prefill.layer.000.short_conv.b"
    assert torch.equal(first.tensors[name], second.tensors[name])
    # Reuse skips redundant hashing, never the payload read: each load still
    # returns freshly cloned storage the caller may mutate independently.
    assert first.tensors[name] is not second.tensors[name]
    assert second.record == first.record
    assert TRACE_FILENAME not in hashed
    assert BATCH_RECORD_FILENAME in hashed

    trace_path = cache.path / "batches" / "batch-000000" / TRACE_FILENAME
    corrupted = bytearray(trace_path.read_bytes())
    corrupted[-1] ^= 0x01
    trace_path.write_bytes(corrupted)
    with pytest.raises(
        Lfm2CircuitTraceCorruptionError,
        match="trace identity changed",
    ):
        opened.load_batch(0)

    opened.release_verified_batch_guards()
    assert opened.verified_batch_guard_count == 0
    with pytest.raises(
        Lfm2CircuitTraceCorruptionError,
        match="trace file hash changed",
    ):
        opened.load_batch(0)


def test_authenticated_session_verifies_each_payload_once_in_canonical_order(
    tmp_path,
    monkeypatch,
) -> None:
    model = _parent(seed=333)
    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(9,)), _batch(forced=(10,))],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=2,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )
    opened = open_lfm2_circuit_trace_cache(cache.path, verify=False)
    payload_loads: list[int] = []
    original_load = trace_cache._load_verified_batch

    def counted_load(cache_path, manifest, entry):
        payload_loads.append(int(entry["batch_index"]))
        return original_load(cache_path, manifest, entry)

    monkeypatch.setattr(
        trace_cache,
        "_load_verified_batch",
        counted_load,
    )
    adapter = Lfm2CpcfInstallabilityAdapter(opened)

    # Session construction authenticates metadata/layout only. It never
    # performs the old throwaway full-payload verification pass.
    assert payload_loads == []
    phases = tuple(adapter.iter_compiler_inputs())
    evidence = dict(adapter.cache_authentication_evidence() or {})

    assert payload_loads == [0, 1]
    assert [phase.batch_index for phase in phases] == [0, 0, 1, 1]
    assert evidence["schema"] == AUTHENTICATED_SESSION_SCHEMA
    assert evidence["cache_key"] == cache.cache_key
    assert evidence["batch_count"] == 2
    assert [batch["batch_index"] for batch in evidence["ordered_verified_batches"]] == [
        0,
        1,
    ]
    body = dict(evidence)
    claimed = body.pop("session_sha256")
    assert claimed == trace_cache._json_sha256(body)
    assert (
        evidence["authentication_contract"]["duplicate_session_payload_loads_permitted"]
        is False
    )

    # Repeating the same authenticated consumption produces the exact same
    # canonical receipt and still performs one payload load per batch.
    repeated = open_lfm2_circuit_trace_cache(
        cache.path,
        verify=False,
    ).authenticated_session()
    tuple(repeated.iter_batches())
    assert repeated.as_dict() == evidence
    assert payload_loads == [0, 1, 0, 1]

    # The established conservative APIs remain available and unchanged.
    assert opened.load_batch(0).input_sha256 == cache.load_batch(0).input_sha256
    assert opened.verify() is opened


def test_authenticated_session_fails_closed_on_tamper_and_noncanonical_read(
    tmp_path,
) -> None:
    model = _parent(seed=334)
    cache = build_lfm2_circuit_trace_cache(
        tmp_path,
        model,
        [_batch(forced=(9,)), _batch(forced=(10,))],
        config=Lfm2CircuitTraceCaptureConfig(sample_last_tokens=2),
        input_device="cpu",
    )
    opened = open_lfm2_circuit_trace_cache(cache.path, verify=False)
    ordering_session = opened.authenticated_session()
    with pytest.raises(
        trace_cache.Lfm2CircuitTraceCacheError,
        match="canonical order",
    ):
        ordering_session.load_batch(1)
    assert ordering_session.verified_batch_count == 0

    tamper_session = opened.authenticated_session()
    trace_path = cache.path / "batches" / "batch-000000" / TRACE_FILENAME
    corrupted = bytearray(trace_path.read_bytes())
    corrupted[-1] ^= 0x01
    trace_path.write_bytes(corrupted)
    with pytest.raises(
        Lfm2CircuitTraceCorruptionError,
        match="trace file hash changed",
    ):
        tamper_session.load_batch(0)
    assert tamper_session.verified_batch_count == 0
    assert tamper_session.complete is False


def test_mislabeled_short_conv_and_reserved_inputs_fail_before_forward(
    tmp_path,
    monkeypatch,
) -> None:
    model = _parent(seed=404)
    calls = _instrument_forwards(monkeypatch, model)
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
    clean_calls = _instrument_forwards(monkeypatch, clean)
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
    padded_calls = _instrument_forwards(monkeypatch, padded)
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
    multi_calls = _instrument_forwards(monkeypatch, multi)
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
