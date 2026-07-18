from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
    open_lfm2_circuit_trace_cache,
)


def _tiny_parent() -> Any:
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
        torch.manual_seed(1927)
        return transformers.Lfm2ForCausalLM(config)


def _sealed_cache(tmp_path: Path):
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
        tmp_path,
        _tiny_parent(),
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )


def test_selective_safetensors_mappings_close_after_each_selection(
    tmp_path,
    monkeypatch,
) -> None:
    cache = _sealed_cache(tmp_path)
    real_safe_open, save_file = trace_cache._safetensors()
    calls = {"entered": 0, "exited": 0, "active": 0}

    class TrackedSafeOpen:
        def __init__(self, *args, **kwargs) -> None:
            self._context = real_safe_open(*args, **kwargs)
            self._handle = None

        def __enter__(self):
            self._handle = self._context.__enter__()
            calls["entered"] += 1
            calls["active"] += 1
            return self._handle

        def __exit__(self, exc_type, exc, traceback):
            try:
                return self._context.__exit__(exc_type, exc, traceback)
            finally:
                calls["active"] -= 1
                calls["exited"] += 1

    monkeypatch.setattr(
        trace_cache,
        "_safetensors",
        lambda: (TrackedSafeOpen, save_file),
    )
    session = open_lfm2_circuit_trace_cache(
        cache.path,
        verify=False,
    ).selective_authenticated_session()
    try:
        assert calls == {"entered": 2, "exited": 2, "active": 0}
        names = session.batch_references[0].ordered_tensor_names[:2]
        payload = session.load_selected_tensors(0, names)
        assert calls == {"entered": 3, "exited": 3, "active": 0}
        assert payload.released is False
        session.release_selected_tensors(payload)
        assert payload.released is True
        evidence = session.as_dict()["authentication_contract"]
        assert evidence["safetensors_mapping_held_open"] is False
        assert evidence["safetensors_mapping_opened_per_selection_and_closed"] is True
    finally:
        session.close()
    assert calls["active"] == 0
    assert calls["entered"] == calls["exited"]
