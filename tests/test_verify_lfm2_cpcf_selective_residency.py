from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

from experiments import (  # noqa: E402
    verify_lfm2_cpcf_selective_residency as verifier,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)


def _parent() -> Any:
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
        torch.manual_seed(20260718)
        return transformers.Lfm2ForCausalLM(config)


def _cache(root: Path) -> Path:
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[5]], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[1, 4, 3, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[6]], dtype=torch.long),
        },
    ]
    return build_lfm2_circuit_trace_cache(
        root,
        _parent(),
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    ).path


def test_real_verifier_contract_walks_and_releases_every_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_index = 0

    def memory() -> dict[str, int]:
        nonlocal sample_index
        sample_index += 1
        return {
            "rss_bytes": sample_index * 1024,
            "peak_rss_bytes": sample_index * 1024,
            "virtual_bytes": sample_index * 2048,
        }

    monkeypatch.setattr(verifier, "_linux_process_memory", memory)
    receipt = verifier.run(
        Namespace(
            trace_cache_directory=_cache(tmp_path / "cache"),
            output=tmp_path / "unused.json",
            maximum_logical_resident_gib=0.0,
            maximum_process_rss_gib=1.0,
        )
    )

    assert receipt["status"] == "passed"
    assert receipt["layer_count"] == 2
    assert [row["layer_index"] for row in receipt["layer_receipts"]] == [
        0,
        1,
    ]
    assert receipt["residency_snapshot"]["requested_layer_indices"] == [
        0,
        1,
    ]
    assert receipt["residency_snapshot"]["promoted_layer_indices"] == [
        0,
        1,
    ]
    assert receipt["residency_snapshot"]["released_layer_indices"] == [
        0,
        1,
    ]
    assert receipt["residency_snapshot"]["peak_occupied_layers"] == 2
    assert receipt["physical_rss_gate"]["passed"] is True
    assert receipt["physical_rss_gate"]["gate_uses_process_high_water"] is True
    assert receipt["parent_model_loaded"] is False
    assert receipt["parent_forward_calls"] == 0
    assert receipt["training_steps"] == 0
    body = dict(receipt)
    supplied = body.pop("receipt_sha256")
    assert supplied == verifier._sha(body)
