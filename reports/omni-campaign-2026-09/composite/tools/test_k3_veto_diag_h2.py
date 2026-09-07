"""CPU checks that H2 distinguishes fit arithmetic from the installed runtime."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_veto_diag_h2 as h2


def fixture(dtype: torch.dtype, chunk: int):
    gen = torch.Generator().manual_seed(17)
    n, hidden, latent, shared_width = 8, 16, 8, 12

    def rand(*shape, scale=0.1):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype)

    body = {
        "block_sparse_moe.routed_expert_norm.weight": torch.ones(latent, dtype=dtype),
        "block_sparse_moe.routed_expert_up_proj.weight": rand(hidden, latent),
        "block_sparse_moe.shared_experts.gate_proj.weight": rand(shared_width, hidden),
        "block_sparse_moe.shared_experts.up_proj.weight": rand(shared_width, hidden),
        "block_sparse_moe.shared_experts.down_proj.weight": rand(hidden, shared_width),
    }
    weights = SimpleNamespace(body=body, is_moe=True)
    x, carried = rand(n, hidden, scale=1), rand(n, hidden, scale=100)
    y = torch.randn(n, latent, generator=gen)
    cfg = {"rms_norm_eps": 1e-6, "activation_situ_beta": 1.0, "activation_situ_linear_beta": None}
    routed, shared = h2.moe_write_split(weights, x, y, cfg, chunk)
    targets = h2.HeldTargets(weights, cfg, torch.arange(n), carried, x, y, routed, shared, chunk)
    return weights, cfg, x, carried, y, targets


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("chunk", [3, 8])
def test_teacher_targets_keep_runtime_dtype_and_addition_order(dtype, chunk):
    weights, cfg, x, carried, y, targets = fixture(dtype, chunk)
    actual_moe = h2.kdp.moe_write(weights, x, y, cfg, chunk)
    assert targets.moe_out.dtype == dtype
    assert targets.prefix_out.dtype == dtype
    assert torch.equal(targets.moe_out, actual_moe)
    assert torch.equal(targets.prefix_out, carried + actual_moe)


@pytest.mark.parametrize("kind", ["identity", "perturbed", "zero"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("chunk", [3, 8])
def test_child_write_matches_actual_ffn_half(dtype, chunk, kind):
    weights, cfg, x, carried, y, targets = fixture(dtype, chunk)
    pred = {"identity": y, "perturbed": y + 0.1, "zero": torch.zeros_like(y)}[kind]
    actual_prefix, _ = h2.kdp.ffn_half(
        SimpleNamespace(w=weights), carried, x, cfg, device=torch.device("cpu"),
        token_chunk=chunk, latent=y, y_lat=pred,
    )
    actual_moe = h2.kdp.moe_write(weights, x, pred, cfg, chunk)
    got = targets.child_write(pred)
    assert got["moe_out"].dtype == dtype
    assert got["prefix_out"].dtype == dtype
    assert torch.equal(got["moe_out"], actual_moe)
    assert torch.equal(got["prefix_out"], actual_prefix)
    assert targets.score_child(pred)["prefix_out_heldout"] == h2.compare(actual_prefix, targets.prefix_out)


@pytest.mark.parametrize("held_rows", [[0, 1, 2, 3, 4, 5, 6, 7], [0, 3, 4, 7]])
def test_fp32_fit_and_runtime_map_scores_are_separate(held_rows):
    weights, cfg, x, carried, y, targets = fixture(torch.bfloat16, 3)
    gen = torch.Generator().manual_seed(33)
    latent = torch.randn(8, 8, generator=gen).bfloat16()
    w1, w3 = (torch.randn(11, 8, generator=gen).bfloat16() for _ in range(2))
    w2p = torch.randn(8, 11, generator=gen)
    bank = SimpleNamespace(psi=lambda z, beta, linear_beta: h2.situ(F.linear(z, w1), F.linear(z, w3), beta, linear_beta))
    fold = SimpleNamespace(device=torch.device("cpu"), latent=latent, chunk=5, beta=1.0, linear_beta=None)
    rows = torch.tensor(held_rows)
    targets = h2.HeldTargets(weights, cfg, rows, carried, x, y, targets.routed_write, targets.shared, 3)
    dense = {"w1_s": w1, "w3_s": w3, "w2p": w2p}
    runtime_pred = h2.kdp.dense_y_lat(dense, latent[rows], cfg, 3)
    fit_pred = h2.predict_y(bank, fold, w2p, rows)
    assert not torch.equal(fit_pred, runtime_pred), "fixture must distinguish the two map arithmetic paths"
    assert torch.equal(h2.predict_y(bank, fold, w2p, rows, runtime_dtype=True, token_chunk=3), runtime_pred)
    scores = h2.score_maps(bank, fold, w2p, rows, targets)
    assert scores["heldout"]["prediction_mode"] == "fp32_fit_map"
    assert scores["heldout_runtime_dtype"]["prediction_mode"] == "runtime_dtype_map"
    assert scores["heldout_runtime_dtype"]["map_dtype"] == "bfloat16"
    assert scores["heldout"]["write_dtype"] == "bfloat16"
    assert scores["heldout"]["y_lat_heldout"] == h2.compare(fit_pred, y[rows])
    assert scores["heldout_runtime_dtype"]["y_lat_heldout"] == h2.compare(runtime_pred, y[rows])
    actual_prefix, _ = h2.kdp.ffn_half(
        SimpleNamespace(w=weights), carried[rows], x[rows], cfg, device=torch.device("cpu"),
        token_chunk=3, dense=dense, latent=latent[rows],
    )
    assert scores["heldout_runtime_dtype"]["prefix_out_heldout"] == h2.compare(actual_prefix, targets.prefix_out)


def test_tiny_diagnostic_receipt_wires_both_map_modes(tmp_path):
    from test_k3_dense_pass import build_synth, write_ids

    model_dir = tmp_path / "model"
    build_synth(model_dir, layers=2)
    vocab = h2.K3Index(str(model_dir)).text_config["vocab_size"]
    ids = write_ids(tmp_path / "ids.safetensors", rows=4, seq=16, vocab=vocab, seed=11,
                    split=[0, 0, 0, 1], real_lengths=[16, 14, 16, 12], role="h2-test")
    receipt_dir = tmp_path / "receipts"
    args = h2.build_parser().parse_args([
        "--src-dir", str(model_dir), "--input-ids", str(ids), "--rung", "64",
        "--device", "cpu", "--experts-device", "cpu", "--scratch", str(tmp_path / "scratch"),
        "--receipt-dir", str(receipt_dir), "--l00-receipt", "", "--holdout-blocks", "6",
        "--chunk", "8", "--token-chunk", "7", "--max-gram-gb", "0",
    ])
    receipt = h2.run(args)
    written = json.loads((receipt_dir / h2.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert written == receipt
    assert receipt["schema"] == "omni-campaign.k3_veto_diag_h2.v2"
    assert "not installed-map parity" in receipt["scoring_contract"]["heldout"]
    fit = receipt["fit"]
    assert fit["prior_scored"]["prediction_mode"] == "fp32_fit_map"
    assert fit["prior_scored_runtime_dtype"]["prediction_mode"] == "runtime_dtype_map"
    for ladder in (fit["ladder_zero_prior"], fit["ladder_law_prior"]):
        assert len(ladder) == len(h2.c14.LAMBDA_LADDER)
        for row in ladder:
            assert row["heldout"]["map_dtype"] == "float32"
            assert row["heldout_runtime_dtype"]["map_dtype"] == "bfloat16"
            assert row["heldout_runtime_dtype"]["write_dtype"] == "bfloat16"
            assert row["heldout"]["projection_chunk"] == 8
            assert row["heldout_runtime_dtype"]["projection_chunk"] == 7
    summary = receipt["h2_summary"]
    best_runtime = fit["zero_prior_summary"]["best_heldout_runtime_dtype"]
    assert summary["best_zero_prior_prefix_out_relL2_held_runtime_dtype"] == best_runtime["prefix_out_heldout"]["relL2"]
    assert "not a runtime acceptance decision" in fit["zero_prior_summary"]["selection_metric"]


def test_runtime_map_uses_token_chunks_not_fit_chunks():
    _, _, _, _, _, targets = fixture(torch.bfloat16, 3)
    latent = torch.arange(64).reshape(8, 8).to(torch.bfloat16) / 64
    calls = []

    def psi(z, beta, linear_beta):
        calls.append(len(z))
        return z

    bank = SimpleNamespace(psi=psi)
    fold = SimpleNamespace(device=torch.device("cpu"), latent=latent, chunk=5, beta=1.0, linear_beta=None)
    scores = h2.score_maps(bank, fold, torch.eye(8), torch.arange(8), targets)
    assert calls == [5, 3, 3, 3, 2], "fit uses fold.chunk; installed-map scoring uses token_chunk"
    assert scores["heldout"]["projection_chunk"] == 5
    assert scores["heldout_runtime_dtype"]["projection_chunk"] == 3
