from __future__ import annotations

import copy
import gc
import os
import weakref
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
import neural_forge.hf_lfm2_circuit_trace_row_selection as row_selection  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCacheError,
    Lfm2CircuitTraceCorruptionError,
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


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_safetensor_flat_row_copy_is_exact_in_source_dtype(
    tmp_path,
    dtype,
) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = torch.arange(2 * 5 * 8, dtype=torch.float32).reshape(2, 5, 8)
    source = (source / 7).to(dtype=dtype)
    path = tmp_path / f"rows-{str(dtype).removeprefix('torch.')}.safetensors"
    save_file({"rows": source}, str(path))
    indices = (0, 2, 4, 5, 6, 9)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        selected, maximum_source_rows, slice_count = (
            row_selection._copy_selected_safetensor_rows(
                torch,
                handle.get_slice("rows"),
                shape=source.shape,
                indices=indices,
                expected_dtype=str(dtype).removeprefix("torch."),
                name="rows",
            )
        )
    expected = source.reshape(-1, source.shape[-1]).index_select(
        0,
        torch.tensor(indices, dtype=torch.long),
    )
    assert torch.equal(selected, expected)
    assert selected.dtype == dtype
    assert selected.device.type == "cpu"
    assert selected.is_contiguous()
    assert selected.requires_grad is False
    assert selected.grad_fn is None
    assert maximum_source_rows == 2
    assert slice_count == 5


def _first_multiline_residual_reference(session):
    reference = session.batch_references[0]
    inventory = reference.record["tensors"]
    name = next(
        candidate
        for candidate in reference.ordered_tensor_names
        if ".residual.pre" in candidate
        and len(inventory[candidate]["shape"]) >= 2
        and inventory[candidate]["numel"] // inventory[candidate]["shape"][-1] >= 2
    )
    descriptor = inventory[name]
    row_count = descriptor["numel"] // descriptor["shape"][-1]
    return name, descriptor, row_count


def test_authenticated_row_selection_is_exact_owned_and_deterministic(
    tmp_path,
) -> None:
    cache = _sealed_cache(tmp_path)
    session = open_lfm2_circuit_trace_cache(
        cache.path,
        verify=False,
    ).selective_authenticated_session()
    try:
        evidence_before = session.as_dict()
        name, descriptor, row_count = _first_multiline_residual_reference(session)
        indices = tuple(sorted({0, row_count // 2, row_count - 1}))
        full = session.load_selected_tensors(0, (name,))
        expected = full.tensors[name].reshape(-1, descriptor["shape"][-1]).index_select(
            0,
            torch.tensor(indices, dtype=torch.long),
        )
        session.release_selected_tensors(full)

        payload = row_selection.load_lfm2_circuit_trace_selected_rows(
            session,
            0,
            {name: indices},
        )
        selected = payload.tensors[name]
        assert torch.equal(selected, expected)
        assert selected.dtype == expected.dtype
        assert tuple(selected.shape) == (len(indices), descriptor["shape"][-1])
        assert payload.row_indices == {name: indices}
        assert payload.logical_bytes == selected.numel() * selected.element_size()
        assert payload.source_logical_bytes == descriptor["size_bytes"]
        assert payload.logical_bytes <= payload.source_logical_bytes
        assert payload.maximum_materialized_source_rows <= len(indices)
        assert payload.selected_row_count == len(indices)
        assert payload.source_row_count == row_count
        assert payload.source_tensor_count == 1
        assert payload.safetensors_get_slice_calls == 1
        assert payload.coalesced_slice_count >= 1
        assert payload.whole_tensor_cpu_clone_calls == 0
        assert payload.unselected_rows_loaded is False
        selection_sha256 = payload.selection_sha256
        tensor_ref = weakref.ref(selected)
        del selected
        row_selection.release_lfm2_circuit_trace_selected_rows(session, payload)
        row_selection.release_lfm2_circuit_trace_selected_rows(session, payload)
        assert payload.released is True
        with pytest.raises(Lfm2CircuitTraceCacheError, match="was released"):
            _ = payload.tensors
        del payload
        gc.collect()
        assert tensor_ref() is None

        repeated = row_selection.load_lfm2_circuit_trace_selected_rows(
            session,
            0,
            {name: indices},
        )
        assert repeated.selection_sha256 == selection_sha256
        row_selection.release_lfm2_circuit_trace_selected_rows(session, repeated)
        assert session.as_dict() == evidence_before
    finally:
        session.close()


def test_authenticated_row_selection_uses_only_slices_and_closes_mapping(
    tmp_path,
    monkeypatch,
) -> None:
    cache = _sealed_cache(tmp_path)
    real_safe_open, save_file = trace_cache._safetensors()
    calls = {
        "entered": 0,
        "exited": 0,
        "active": 0,
        "get_slice": 0,
        "get_tensor": 0,
    }

    class TrackedHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def get_slice(self, name):
            calls["get_slice"] += 1
            return self._handle.get_slice(name)

        def get_tensor(self, _name):
            calls["get_tensor"] += 1
            raise AssertionError("row selection materialized a whole tensor")

    class TrackedSafeOpen:
        def __init__(self, *args, **kwargs) -> None:
            self._context = real_safe_open(*args, **kwargs)

        def __enter__(self):
            calls["entered"] += 1
            calls["active"] += 1
            return TrackedHandle(self._context.__enter__())

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
        name, _descriptor, row_count = _first_multiline_residual_reference(session)
        payload = row_selection.load_lfm2_circuit_trace_selected_rows(
            session,
            0,
            {name: (0, row_count - 1)},
        )
        assert calls["active"] == 0
        assert calls["get_slice"] == 1
        assert calls["get_tensor"] == 0
        assert payload.tensors[name].shape[0] == 2
        row_selection.release_lfm2_circuit_trace_selected_rows(session, payload)
    finally:
        session.close()
    assert calls["active"] == 0
    assert calls["entered"] == calls["exited"]
    assert all(state.trace_guard.closed for state in session._batch_states)
    session.close()
    with pytest.raises(Lfm2CircuitTraceCacheError, match="closed"):
        row_selection.load_lfm2_circuit_trace_selected_rows(
            session,
            0,
            {name: (0,)},
        )


@pytest.mark.parametrize(
    "indices",
    ((), (-1,), (0, 0), (1, 0), (False,), (0.5,)),
)
def test_authenticated_row_selection_rejects_invalid_plans_before_mapping(
    tmp_path,
    monkeypatch,
    indices,
) -> None:
    cache = _sealed_cache(tmp_path)
    session = open_lfm2_circuit_trace_cache(
        cache.path,
        verify=False,
    ).selective_authenticated_session()
    name, _descriptor, row_count = _first_multiline_residual_reference(session)
    mappings = 0
    real_safe_open, save_file = trace_cache._safetensors()

    def tracked_safe_open(*args, **kwargs):
        nonlocal mappings
        mappings += 1
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(
        trace_cache,
        "_safetensors",
        lambda: (tracked_safe_open, save_file),
    )
    try:
        with pytest.raises(Lfm2CircuitTraceCacheError):
            row_selection.load_lfm2_circuit_trace_selected_rows(
                session,
                0,
                {name: indices},
            )
        with pytest.raises(Lfm2CircuitTraceCacheError):
            row_selection.load_lfm2_circuit_trace_selected_rows(
                session,
                0,
                {name: (row_count,)},
            )
        with pytest.raises(TypeError):
            row_selection.load_lfm2_circuit_trace_selected_rows(session, 0, {})
        assert mappings == 0
    finally:
        session.close()


def test_authenticated_row_selection_fails_closed_after_file_identity_tamper(
    tmp_path,
) -> None:
    cache = _sealed_cache(tmp_path)
    session = open_lfm2_circuit_trace_cache(
        cache.path,
        verify=False,
    ).selective_authenticated_session()
    name, _descriptor, _row_count = _first_multiline_residual_reference(session)
    trace_path = session._batch_states[0].trace_path
    current = trace_path.stat()
    os.utime(
        trace_path,
        ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
    )
    with pytest.raises(Lfm2CircuitTraceCorruptionError, match="identity changed"):
        row_selection.load_lfm2_circuit_trace_selected_rows(
            session,
            0,
            {name: (0,)},
        )
    with pytest.raises(Lfm2CircuitTraceCorruptionError):
        session.close()
    assert session.closed is True
    assert all(state.trace_guard.closed for state in session._batch_states)


def test_authenticated_row_selection_rejects_in_memory_descriptor_tamper(
    tmp_path,
    monkeypatch,
) -> None:
    cache = _sealed_cache(tmp_path)
    session = open_lfm2_circuit_trace_cache(
        cache.path,
        verify=False,
    ).selective_authenticated_session()
    name, _descriptor, _row_count = _first_multiline_residual_reference(session)
    session._batch_states[0].record["tensors"][name]["sha256"] = "0" * 64
    monkeypatch.setattr(
        trace_cache,
        "_safetensors",
        lambda: (_ for _ in ()).throw(
            AssertionError("descriptor tamper reached safetensors mapping")
        ),
    )
    with pytest.raises(
        Lfm2CircuitTraceCorruptionError,
        match="self-hash does not match",
    ):
        row_selection.load_lfm2_circuit_trace_selected_rows(
            session,
            0,
            {name: (0,)},
        )
    session.close()
    assert all(state.trace_guard.closed for state in session._batch_states)


def test_authenticated_row_selection_rejects_resealed_record_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    cache = _sealed_cache(tmp_path)
    session = open_lfm2_circuit_trace_cache(
        cache.path,
        verify=False,
    ).selective_authenticated_session()
    name, _descriptor, _row_count = _first_multiline_residual_reference(session)
    state = session._batch_states[0]
    record = copy.deepcopy(dict(state.record))
    record["tensors"][name]["sha256"] = "0" * 64
    record.pop("record_sha256")
    record["record_sha256"] = trace_cache._json_sha256(record)
    state.record = MappingProxyType(record)
    monkeypatch.setattr(
        trace_cache,
        "_safetensors",
        lambda: (_ for _ in ()).throw(
            AssertionError("resealed descriptor reached safetensors mapping")
        ),
    )
    with pytest.raises(
        Lfm2CircuitTraceCorruptionError,
        match="record seal changed",
    ):
        row_selection.load_lfm2_circuit_trace_selected_rows(
            session,
            0,
            {name: (0,)},
        )
    session.close()
    assert all(state.trace_guard.closed for state in session._batch_states)
