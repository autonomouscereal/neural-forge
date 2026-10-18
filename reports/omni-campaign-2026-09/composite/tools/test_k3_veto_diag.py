"""Tests for ``k3_veto_diag.py`` on the lane's synthetic K3 (CPU, fp32).

  test_uniform_full_width_capture_then_diag_is_exact_and_law_admits
      capture + K=full diag on the uniform-router fixture: the cache is written and
      digested, THE LAW admits (update_scale < 1e-3), the free solve reproduces
      y_lat on the held rows, the eigen-floored solve lifts nothing and equals the
      free solve, the spectrum is recorded, nothing was installed.
  test_sparse_router_diag_records_the_law_and_the_measurements
      top-2-of-8 routing at a small rung, from the capture's cache: the law's
      verdict is derived from the ladder it records, every candidate is scored on
      the held rows and on the fit split, the fit-split SSE from the Gram identity
      equals a direct computation, zero scores relL2 1.0, and `summarise` collects
      both configurations.

Server:
    cd $OC/composite/tools && CUDA_VISIBLE_DEVICES="" \\
        /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \\
        -p no:cacheprovider test_k3_veto_diag.py -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import k3_dense_pass as kdp  # noqa: E402
import k3_veto_diag as kvd  # noqa: E402
from test_k3_dense_pass import build_synth, write_ids  # noqa: E402

sys.path.insert(0, kdp.LANE_DIR)
import c14_union_coreset as c14  # noqa: E402
import k3_common  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402


@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("k3_veto_diag")


@pytest.fixture(scope="module")
def sparse_fixture(work: Path) -> dict:
    out = work / "synthetic-k3"
    synth = build_synth(out)
    return {"dir": str(out), "index": k3_common.K3Index(str(out)), "synth": synth}


@pytest.fixture(scope="module")
def uniform_fixture(work: Path) -> dict:
    out = work / "synthetic-k3-uniform"
    synth = build_synth(out, top_k=8)
    index = k3_common.K3Index(str(out))
    for layer in range(index.num_layers):
        if not index.is_moe_layer(layer):
            continue
        prefix = index.layer_prefix_for(layer)
        shard = index.weight_map[f"{prefix}block_sparse_moe.gate.weight"]
        path = os.path.join(str(out), shard)
        tensors = load_file(path)
        tensors[f"{prefix}block_sparse_moe.gate.weight"] = torch.zeros_like(
            tensors[f"{prefix}block_sparse_moe.gate.weight"])
        tensors[f"{prefix}block_sparse_moe.gate.e_score_correction_bias"] = torch.zeros_like(
            tensors[f"{prefix}block_sparse_moe.gate.e_score_correction_bias"])
        save_file(tensors, path)
    return {"dir": str(out), "index": index, "synth": synth}


@pytest.fixture(scope="module")
def ids(work: Path, sparse_fixture: dict) -> Path:
    vocab = sparse_fixture["index"].text_config["vocab_size"]
    return write_ids(work / "ids.safetensors", rows=6, seq=16, vocab=vocab, seed=11,
                     split=[0, 0, 0, 0, 1, 1], real_lengths=[16, 14, 16, 12, 16, 13], role="evidence")


def capture_argv(model_dir: str, ids_path: Path, root: Path, **extra) -> list[str]:
    argv = ["capture", "--src-dir", model_dir, "--input-ids", str(ids_path),
            "--scratch-out", str(root / "pass_out"), "--cache-dir", str(root / "cache"),
            "--receipt-dir", str(root / "receipts"), "--device", "cpu", "--experts-device", "cpu",
            "--compute-dtype", "fp32", "--layers", "0-1", "--row-chunk", "1", "--token-chunk", "7",
            "--chunk", "8", "--check-blocks", "6", "--max-gram-gb", "0", "--max-device-gb", "0",
            "--no-layer0-receipt", "--digest-receipt", str(root / "no-such-digest.json")]
    for key, value in extra.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    return argv


def read_receipt(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_uniform_full_width_capture_then_diag_is_exact_and_law_admits(work: Path, uniform_fixture: dict, ids: Path):
    cfg = uniform_fixture["index"].text_config
    full = int(cfg["num_experts"]) * int(cfg["moe_intermediate_size"])
    root = work / "uniform"
    assert kvd.main(capture_argv(uniform_fixture["dir"], ids, root, then_rung=full, then_k_pool_factor=1,
                                 save_gram_dir=root / "gram")) == 0
    # the cache: every tensor, digested, sized like the corpus
    manifest = read_receipt(root / "cache" / kvd.CACHE_MANIFEST)
    cache, _ = kvd.load_cache(str(root / "cache"))
    tokens = 6 * 16
    assert cache["latent_in"].shape[0] == tokens and cache["y_lat"].shape == (tokens, cfg["routed_expert_hidden_size"])
    assert cache["fit_mask"].dtype == torch.bool and int(cache["fit_mask"].sum()) == 16 + 14 + 16 + 12
    assert int(cache["held_mask"].sum()) == 16 + 13
    assert manifest["layer"] == 1 and manifest["layer0"]["stream_divergence"]["held"]["relL2"] == 0.0
    # nothing installed: no dense/L01 artefacts under the scratch out
    assert not (root / "pass_out" / "dense" / "L01" / "ffn.safetensors").exists()
    assert not list((root / "pass_out" / "dense" / "L01").glob("*formE*")) if (root / "pass_out" / "dense" / "L01").exists() else True
    receipt = read_receipt(root / "receipts" / f"k3_veto_diag_K{full}_kp1.json")
    assert receipt["mode"] == "capture+diag" and receipt["hypothesis"].startswith("H3")
    assert receipt["gram"]["K"] == full
    law = receipt["law"]
    assert law["law_refused"] is False and law["law_message"] is None
    assert law["update_scale"] < 1e-3 and law["veto_binding"] is False, law
    assert law["returned_is_an_install"] is False
    assert len(law["ladder"]) == len(c14.LAMBDA_LADDER)
    cands = receipt["candidates"]
    assert cands["free"]["heldout"]["y_lat_heldout"]["relL2"] < 1e-4, cands["free"]
    assert cands["free"]["fit_relL2_from_gram_identity"] < 1e-4, cands["free"]
    assert cands["zero"]["heldout"]["y_lat_heldout"]["relL2"] == pytest.approx(1.0, abs=1e-6)
    # the prior IS the dense answer here (m_c W2 = W2 / E), so it scores like the free solve
    assert cands["prior"]["heldout"]["y_lat_heldout"]["relL2"] < 1e-3, cands["prior"]
    # the eigen-floored solve at qf_solve's floor lifted nothing and equals the free solve
    floored = receipt["eigen_solves"]["eigen_floor_1e-12"]
    assert floored["eigenvalues_lifted_to_the_floor"] == 0
    assert abs(cands["eigen_floor_1e-12"]["norm_F"] - cands["free"]["norm_F"]) < 1e-5 * max(cands["free"]["norm_F"], 1e-12)
    assert cands["eigen_floor_1e-12"]["heldout"]["y_lat_heldout"]["relL2"] < 1e-4
    spectrum = receipt["gram"]["spectrum"]
    assert spectrum["n"] == full and spectrum["eigenvalue_max"] > 0
    assert set(spectrum["numerical_rank_at_rel"]) == {f"{r:g}" for r in kvd.SPECTRUM_THRESHOLDS}
    assert receipt["gram"]["saved"] and os.path.exists(receipt["gram"]["saved"]["path"])
    saved = load_file(receipt["gram"]["saved"]["path"])
    assert saved["gram_all"].shape == (full, full) and saved["gram_all"].dtype == torch.float64


def test_sparse_router_diag_records_the_law_and_the_measurements(work: Path, sparse_fixture: dict, ids: Path):
    root = work / "sparse"
    assert kvd.main(capture_argv(sparse_fixture["dir"], ids, root)) == 0
    assert (root / "receipts" / "k3_veto_diag_capture.json").exists()
    # diag from the cache, in-process, so the objects can be checked directly
    index = sparse_fixture["index"]
    cache, manifest = kvd.load_cache(str(root / "cache"))
    fit_mask, held_mask = cache.pop("fit_mask"), cache.pop("held_mask")
    stager = kdp.ShardStager(sparse_fixture["dir"], None, enabled=False, window=1, prefetch=0,
                             delete_after_consume=False, run_id="t")
    device = torch.device("cpu")
    weights, _, _ = kdp.read_layer(index, manifest["layer"], stager, device, device, torch.float32)
    cfg = index.text_config
    receipt, sink = kvd.run_config(weights, dict(cache), cfg, fit_mask, held_mask, rung=48, k_pool_factor=1,
                                   device=device, name="K48_kp1", chunk=8, check_blocks=6,
                                   max_gram_gb=0.0, max_device_gb=0.0)
    # the law's verdict follows from the ladder it recorded, and never raised
    scales = [row["update_scale"] for row in receipt["law"]["ladder"]]
    assert receipt["law"]["law_refused"] == (min(scales) > 1.0)
    assert receipt["law"]["free_solve_update_scale"] == scales[0]
    assert receipt["law"]["min_update_scale_on_ladder"] == min(scales)
    assert receipt["law"]["returned_candidate"] in sink.candidates
    expected = {"free", "prior", "zero", "pinv_floor_1e-12"} | {f"ridge_rel_{r:g}" for r in c14.LAMBDA_LADDER} \
        | {f"eigen_floor_{f:g}" for f in kvd.EIGEN_FLOORS}
    assert set(sink.candidates) == expected and set(receipt["candidates"]) == expected
    for name, rec in receipt["candidates"].items():
        assert "heldout" in rec and 0.0 <= rec["heldout"]["y_lat_heldout"]["relL2"]
        assert rec["fit_relL2_from_gram_identity"] >= 0.0
    assert receipt["candidates"]["zero"]["fit_relL2_from_gram_identity"] == pytest.approx(1.0, abs=1e-6)
    assert receipt["candidates"]["zero"]["heldout"]["y_lat_heldout"]["relL2"] == pytest.approx(1.0, abs=1e-6)
    # the fit-split SSE from the Gram identity equals the direct residual on the fit rows
    fold = kdp.DualCoresetFold(weights, dict(cache), cfg, device, fit_mask, held_mask, check_blocks=6, chunk=8,
                               max_gram_gb=0.0, max_device_gb=0.0)
    bank = c14.ExpertBank(weights, sink.record["channels"], device, dtype=fold.latent.dtype)
    rows = torch.nonzero(fold.fit).flatten()
    psi = bank.psi(fold.latent[rows], fold.beta, fold.linear_beta).double()
    for name in ("free", "ridge_rel_100", "eigen_floor_1e-12", "prior"):
        w = sink.candidates[name].double()
        direct = float((psi @ w.T - fold.y_lat[rows].double()).pow(2).sum())
        via_gram = receipt["candidates"][name]["fit_sse_from_gram_identity"]
        assert abs(direct - via_gram) <= 1e-6 * max(direct, 1e-12), (name, direct, via_gram)
    # the held-out score of the free solve equals CoresetFold.score on the same map
    ref = fold.score(bank, sink.candidates["free"])
    got = receipt["candidates"]["free"]["heldout"]
    assert abs(ref["y_lat_heldout"]["relL2"] - got["y_lat_heldout"]["relL2"]) < 1e-6
    assert abs(ref["routed_write_heldout"]["relL2"] - got["routed_write_heldout"]["relL2"]) < 1e-6
    # the spectrum is consistent with the Gram
    spectrum = receipt["gram"]["spectrum"]
    assert spectrum["n"] == receipt["gram"]["K"] == int(bank.w1.shape[0])
    assert spectrum["eigenvalue_max"] >= spectrum["percentiles"]["99"] >= spectrum["eigenvalue_min"]
    # the receipt file and the summary
    path = kvd.finish_receipt({**kvd.base_receipt(kvd.build_parser().parse_args(
        ["diag", "--cache-dir", str(root / "cache"), "--receipt-dir", str(root / "receipts"), "--rung", "48"]),
        "diag"), **receipt}, str(root / "receipts"), "K48_kp1")
    assert os.path.exists(path)
    assert kvd.main(["summarise", "--receipt-dir", str(root / "receipts"), "--verdict", "inconclusive",
                     "--note", "test"]) == 0
    summary = read_receipt(root / "receipts" / "k3_veto_diag_receipt.json")
    names = {row["name"] for row in summary["configurations"]}
    assert names == {"K48_kp1"} and summary["verdict"] == "inconclusive"
    row = summary["configurations"][0]
    assert row["law_refused"] == receipt["law"]["law_refused"]
    assert row["heldout_relL2"]["zero"] == pytest.approx(1.0, abs=1e-6)
