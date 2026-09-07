"""All-row K3 scoring regressions; CPU fixtures are not model-quality evidence."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_dense_pass as kdp
import k3_common as common


def legacy_scores(pred, target):
    pred, target = pred.float(), target.float()
    cos = F.cosine_similarity(pred, target, dim=-1).flatten()
    return {"relL2": float((pred - target).norm() / max(float(target.norm()), 1e-30)),
            "cos": float(cos.mean()),
            "cos_p05": float(torch.quantile(cos, cos.new_tensor(0.05)))}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("chunk", [1, 7, 128])
def test_all_metrics_match_original_population_not_chunk_averages(dtype, chunk):
    rng = torch.Generator().manual_seed(3107)
    pred = torch.randn(83, 29, generator=rng).to(dtype)
    target = torch.randn(83, 29, generator=rng).to(dtype)
    pred[:5] = 0
    target[5:10] = 0
    actual = common.write_scores(pred, target, chunk_rows=chunk)
    expected = legacy_scores(pred, target)
    assert actual == pytest.approx(expected, rel=2e-6, abs=2e-7)
    # Exact global p05, including the unequal tail; not a merge of chunk p05s.
    assert actual["cos_p05"] == expected["cos_p05"]


def test_noncontiguous_leading_axes_and_vector_keep_original_row_order():
    rng = torch.Generator().manual_seed(8)
    pred = torch.randn(4, 7, 13, generator=rng).transpose(0, 1)
    target = torch.randn(4, 7, 13, generator=rng).transpose(0, 1)
    assert not pred.is_contiguous()
    assert common.write_scores(pred, target, chunk_rows=3) == pytest.approx(
        legacy_scores(pred, target), rel=2e-6, abs=2e-7)
    assert common.write_scores(pred[0, 0], target[0, 0]) == pytest.approx(
        legacy_scores(pred[0, 0], target[0, 0]), rel=2e-6, abs=2e-7)


def test_exact_cosines_only_cpu_scalars_and_bounded_scratch(monkeypatch):
    original = F.cosine_similarity
    calls = []

    def bounded(pred, target, *args, **kwargs):
        assert pred.shape[0] <= 11
        assert pred.dtype == target.dtype == torch.float32
        calls.append(pred.shape[0])
        return original(pred, target, *args, **kwargs)

    monkeypatch.setattr(F, "cosine_similarity", bounded)
    pred = torch.randn(103, 53, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn_like(pred)
    scores = common.WriteScoreAccumulator(11)
    scores.update(pred[:37], target[:37])
    scores.update(pred[37:], target[37:])
    assert sum(calls) == scores.rows == 103
    assert sum(t.numel() for t in scores.cosines) == 103
    assert all(t.device.type == "cpu" and t.ndim == 1 and not t.requires_grad for t in scores.cosines)
    assert all(math.isfinite(x) for x in scores.result().values())


@pytest.mark.parametrize("scale", [0.0, 1e-25, 1e25])
def test_norm_sums_preserve_definition_without_fp32_square_overflow(scale):
    pred = torch.full((17, 13), scale, dtype=torch.float32)
    target = torch.full_like(pred, scale / 2)
    actual = common.write_scores(pred, target, chunk_rows=3)
    expected = float((pred - target).double().norm()) / max(float(target.double().norm()), 1e-30)
    assert actual["relL2"] == pytest.approx(expected, rel=1e-12)


def test_nonfinite_scores_are_not_silently_repaired():
    pred, target = torch.ones(5, 3), torch.ones(5, 3)
    pred[4, 1] = float("nan")
    actual = common.write_scores(pred, target, chunk_rows=2)
    assert all(math.isnan(value) for value in actual.values())


@pytest.mark.parametrize("chunk", [0, -1, True, 2.5])
def test_invalid_chunk_is_refused(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        common.WriteScoreAccumulator(chunk)


def test_empty_or_misaligned_population_is_refused():
    with pytest.raises(ValueError, match="at least one row"):
        common.write_scores(torch.empty(0, 3), torch.empty(0, 3))
    with pytest.raises(ValueError, match="equal shapes"):
        common.write_scores(torch.ones(5, 3), torch.ones(1, 3))


def fake_fold(dtype):
    torch.manual_seed(91)
    fold = object.__new__(kdp.DualCoresetFold)
    fold.chunk, fold.device = 7, torch.device("cpu")
    fold.latent = torch.randn(83, 11).to(dtype)
    fold.y_lat = torch.randn(83, 11)
    fold.beta, fold.linear_beta = 1.0, 0.0
    fold.cfg = {"rms_norm_eps": 1e-6}
    fold.w = SimpleNamespace(body={
        "block_sparse_moe.routed_expert_norm.weight": torch.randn(11).to(dtype),
        "block_sparse_moe.routed_expert_up_proj.weight": torch.randn(19, 11).to(dtype)})
    fold.fit = torch.arange(83) % 3 != 0
    fold.held = ~fold.fit
    fold.timings = {}
    bank = SimpleNamespace(psi=lambda x, _beta, _linear: x[:, :9].contiguous())
    w2p = torch.randn(11, 9)
    fold._last_prior = w2p * 0.25
    return fold, bank, w2p


def legacy_map(fold, bank, w2p, rows, runtime_dtype=False):
    predictions = []
    for start in range(0, rows.numel(), fold.chunk):
        psi = bank.psi(fold.latent[rows[start:start + fold.chunk]], fold.beta, fold.linear_beta)
        pred = F.linear(psi, w2p.to(psi.dtype)).float() if runtime_dtype else F.linear(psi.float(), w2p.float())
        predictions.append(pred)
    pred, target = torch.cat(predictions), fold.y_lat[rows]
    return {"y_lat": legacy_scores(pred, target),
            "routed_write": legacy_scores(fold._routed_write(pred), fold._routed_write(target))}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("runtime_dtype", [False, True])
def test_prediction_and_routed_projection_never_see_whole_population(monkeypatch, dtype, runtime_dtype):
    fold, bank, w2p = fake_fold(dtype)
    rows = torch.arange(83)
    expected = legacy_map(fold, bank, w2p, rows, runtime_dtype)
    original_linear, original_cos = F.linear, F.cosine_similarity
    observed = []

    def bounded_linear(x, *args, **kwargs):
        assert x.shape[0] <= fold.chunk
        observed.append(x.shape[0])
        return original_linear(x, *args, **kwargs)

    def bounded_cos(x, *args, **kwargs):
        assert x.shape[0] <= fold.chunk
        return original_cos(x, *args, **kwargs)

    monkeypatch.setattr(F, "linear", bounded_linear)
    monkeypatch.setattr(F, "cosine_similarity", bounded_cos)
    actual = fold._score_map(bank, w2p, rows, runtime_dtype)
    assert observed
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=2e-6, abs=2e-7)


def test_every_held_fit_runtime_zero_prior_metric_survives():
    fold, bank, w2p = fake_fold(torch.float32)
    result = fold.score(bank, w2p)
    held, fit = torch.nonzero(fold.held).flatten(), torch.nonzero(fold.fit).flatten()
    for rows, suffix, runtime in ((held, "heldout", False), (held, "heldout_runtime_dtype", True),
                                  (fit, "fit", False)):
        expected = legacy_map(fold, bank, w2p, rows, runtime)
        for key in expected:
            actual = {k: v for k, v in result[f"{key}_{suffix}"].items() if k != "dtype"}
            assert actual == pytest.approx(expected[key], rel=2e-6, abs=2e-7)
    for name, writer in (("zero", torch.zeros_like(w2p)), ("prior", fold._last_prior)):
        expected = legacy_map(fold, bank, writer, held)
        for key in expected:
            assert result["baselines"][name][f"{key}_heldout"] == pytest.approx(expected[key], rel=2e-6, abs=2e-7)
    assert result["rows"] == {"held": 28, "fit": 55}
    assert result["scoring"]["rows_sampled"] is False
    assert {"score_seconds", "score_extras_seconds"} <= fold.timings.keys()


def test_scoring_oom_cannot_leave_a_serialized_or_accepted_fold(tmp_path, monkeypatch):
    """The real run_rung ordering must never turn an unfinished score into a child."""
    reached = []

    def refit(channels):
        reached.append("refit")
        return torch.ones(3, 4), SimpleNamespace(), {}

    def score(bank, writer):
        reached.append("score")
        raise torch.cuda.OutOfMemoryError("injected scoring allocation failure")

    fold = SimpleNamespace(
        n_experts=8, intermediate=16,
        stage1=lambda _k: (torch.arange(4), {"k_pool_selected": 4}),
        stage2=lambda channels, _k: (channels, {"skipped": "stage1 only"}),
        refit=refit, score=score)
    monkeypatch.setattr(kdp.c14, "save_tensors", lambda *_a, **_kw: pytest.fail("serialized an unscored fold"))
    with pytest.raises(torch.cuda.OutOfMemoryError, match="injected scoring"):
        kdp.c14.run_rung(fold, 4, 1, str(tmp_path), 1, {}, 3)
    assert reached == ["refit", "score"]
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optional bounded-scratch CUDA check")
def test_cuda_scratch_is_smaller_than_full_population():
    pred = torch.randn(4096, 512, device="cuda")
    target = torch.randn_like(pred)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    expected = legacy_scores(pred, target)
    torch.cuda.synchronize()
    full_scratch = torch.cuda.max_memory_allocated() - before
    torch.cuda.reset_peak_memory_stats()
    actual = common.write_scores(pred, target, chunk_rows=64)
    torch.cuda.synchronize()
    chunk_scratch = torch.cuda.max_memory_allocated() - before
    assert chunk_scratch < full_scratch / 4
    assert actual == pytest.approx(expected, rel=2e-6, abs=2e-7)
