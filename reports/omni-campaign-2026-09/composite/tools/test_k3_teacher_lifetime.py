"""CPU synthetic lifetime regressions, not real-model/GPU memory evidence."""
import inspect
import sys
from pathlib import Path
import weakref

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_k3_dense_pass import (kdp, work, fixture, corpus_files, base_argv, run_pass)  # noqa: F401


@pytest.mark.parametrize("copy_prefix", [False, True])
def test_teacher_reference_record_keeps_required_fit_tensors(copy_prefix):
    prefix = torch.randn(5, 7, dtype=torch.float64)
    persisted = prefix.clone() if copy_prefix else prefix
    taps = {"latent_in": torch.randn(5, 3), "y_lat": torch.randn(5, 3),
            "router_topk_idx": torch.zeros(5, 2, dtype=torch.long), "router_topk_w": torch.ones(5, 2)}
    required = {name: value for name, value in taps.items() if name != "latent_in"}
    unused_ref = weakref.ref(taps["latent_in"])
    record = kdp.teacher_temporary_lifetime(prefix, persisted, taps)
    assert unused_ref() is None
    assert set(taps) == set(required) and all(taps[k] is required[k] for k in required)
    assert torch.equal(prefix, persisted) and persisted.dtype == torch.float64
    assert record["released_references"]["teacher_prefix_local_alias"]["logical_bytes"] == 5 * 7 * 8
    assert record["released_references"]["unused_teacher_latent_in"]["logical_bytes"] == 5 * 3 * 4
    assert record["prefix_storage_retained_by_residual_pair"] == (not copy_prefix)
    assert record["cuda_allocator_bytes_freed"] is None


def test_dense_teacher_has_no_latent_to_remove():
    prefix = torch.ones(2, 3)
    record = kdp.teacher_temporary_lifetime(prefix, prefix, {})
    assert set(record["released_references"]) == {"teacher_prefix_local_alias"}


def test_actual_run_layer_releases_unused_teacher_references_before_folded_attention(
        tmp_path, fixture, corpus_files, monkeypatch):
    actual_ffn, actual_attention = kdp.ffn_half, kdp.attention_half
    pending, checked = {}, []

    def ffn(runner, *args, **kwargs):
        result = actual_ffn(runner, *args, **kwargs)
        if runner.w.is_moe and kwargs.get("dense") is None:
            pending[runner.w.layer_idx] = weakref.ref(result[1]["latent_in"])
        return result

    def attention(runner, *args, **kwargs):
        layer = runner.w.layer_idx
        if layer in pending:
            assert pending.pop(layer)() is None, "teacher latent remained alive across folded attention"
            caller = inspect.currentframe().f_back
            try:
                assert caller.f_code.co_name == "run_layer"
                assert "t_prefix_new" not in caller.f_locals, "teacher GPU prefix alias remained in run_layer"
                assert "latent_in" not in caller.f_locals["t_taps"]
            finally:
                del caller
            checked.append(layer)
        return actual_attention(runner, *args, **kwargs)

    monkeypatch.setattr(kdp, "ffn_half", ffn)
    monkeypatch.setattr(kdp, "attention_half", attention)
    receipt = run_pass(base_argv(fixture["dir"], corpus_files, tmp_path / "fold", tmp_path / "receipts"))
    assert checked == [1, 2]
    assert all("teacher_temporary_lifetime" in row for row in receipt["layers"].values())
    assert receipt["layers"]["2"]["teacher_temporary_lifetime"]["cuda_allocator_bytes_freed"] is None
