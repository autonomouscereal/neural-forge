"""Tests for ``k3_dense_pass.py`` -- the dual-stream K3 pass, on the lane's synthetic K3.

CPU, fp32, tiny fixtures built by ``k3_synth.build`` (3 layers: L0 KDA + dense
MLP, L1 KDA + MoE (non-snapshot at block size 2), L2 MLA + MoE (snapshot)).
The run-book row's six tests are here by name, plus the mechanics they rest on:

  1 test_full_width_uniform_router_fold_is_exact_and_streams_agree
  2 test_t11c_two_stream_resume_reproduces_a_single_pass  (+ the kill-and-resume case)
  3 test_staged_deletion_touches_only_files_the_run_created
  4 test_share_rate_refusal_only_when_source_on_share
  5 test_layer0_verbatim_sha_matches_import_receipt (+ the REAL K3 layer 0 when the Lexar copy is present)
  6 test_chunked_apply_attn_res_equals_unchunked

and the review findings, one test each:
  test_stager_refuses_by_name_instead_of_hanging          (window deadlock / kept shards / stall watchdog)
  test_digest_receipt_status_partial_is_not_all_verified  (incremental digest receipt)
  test_non_finite_fit_input_is_a_named_refusal            (section 3b, NaN never reaches the Gram)
  test_missing_layer0_receipt_is_refused_unless_flagged   (run-book test 5 is not skipped silently)
  test_window_above_layer0_rebuilds_the_dense_prefix_and_stamps_the_rung
        (`--layers 1-1`; snapshot rung + installed-fold stamp; receipt carry by mtime / rung / marker)
  t11c additionally checks that a resume over done markers keeps the fold metrics.

v1.2.0 -- the measured acceptance rule that replaced THE LAW's norm-ratio veto:
  test_measured_solver_admits_a_fit_the_law_refuses_and_pins_c14_arithmetic
        (a ~1e-3-scaled prior: THE LAW refuses every rung, the measured solver installs the
        answer and records the law's verdict; a ZERO prior no longer refuses; with a full-scale
        prior the two solvers agree rung for rung -- the drift pin)
  test_evaluate_acceptance_on_the_l01_numbers   (G1-G3 on the recorded K3 L01 numbers: the
        rel=1 rung passes, the prior and the zero map fail G1, the free solve fails G2,
        an exact fold passes on the floor, a prefix worse than the zero child fails G3)
  test_a_no_signal_map_is_refused_and_nothing_is_installed   (end to end: a solver that
        returns the zero map -> named refusal with the numbers in the partial receipt, no
        ffn.safetensors, no done marker, the c14 solver restored)
  the uniform-router exact fold additionally asserts verdict accepted + law would_refuse False.

Server:
    cd $OC/composite/tools && CUDA_VISIBLE_DEVICES="" \\
        /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \\
        -p no:cacheprovider test_k3_dense_pass.py -q
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 ... -k gpu   (the one GPU cell)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import k3_dense_pass as kdp  # noqa: E402

sys.path.insert(0, kdp.LANE_DIR)
import c14_union_coreset as c14  # noqa: E402
import k3_common  # noqa: E402
import k3_layer  # noqa: E402
import k3_synth  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

REAL_K3 = "/mnt/lexar/models/moonshotai/Kimi-K3"
REAL_L0_RECEIPT = os.environ.get(
    "K3_LAYER0_IMPORT_RECEIPT", "/home/cereal/omni-campaign/composite/receipts/k3_layer0_import_receipt.json")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("k3_dense_pass")


def build_synth(out_dir: Path, *, top_k: int = 2, layers: int = 3) -> dict:
    argv = ["--out-dir", str(out_dir), "--layers", str(layers), "--hidden", "64", "--experts", "8",
            "--top-k", str(top_k), "--attn-res-block-size", "2", "--latent", "32", "--moe-intermediate", "64"]
    return k3_synth.build(k3_synth.build_parser().parse_args(argv))


@pytest.fixture(scope="module")
def fixture(work: Path) -> dict:
    out = work / "synthetic-k3"
    synth = build_synth(out)
    return {"dir": str(out), "index": k3_common.K3Index(str(out)), "synth": synth}


@pytest.fixture(scope="module")
def uniform_fixture(work: Path) -> dict:
    """top_k = num_experts and a ZERO router: every expert open on every token with
    weight 1/E.  That is the one regime where a dense union of every channel is
    algebraically the routed teacher, so a full-width fold must be EXACT -- any
    drift is a mechanics bug, not approximation error."""
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


def write_ids(path: Path, *, rows: int, seq: int, vocab: int, seed: int, split: list[int],
              real_lengths: list[int], role: str) -> Path:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (rows, seq), generator=generator, dtype=torch.long)
    split_t = torch.tensor(split, dtype=torch.int8)
    lengths = torch.tensor(real_lengths, dtype=torch.long)
    real = torch.arange(seq).unsqueeze(0) < lengths.unsqueeze(1)
    fit = real & (split_t == 0).unsqueeze(1)
    held = real & (split_t == 1).unsqueeze(1)
    save_file({"input_ids": ids, "split": split_t, "real_lengths": lengths, "fit_mask": fit, "held_mask": held},
              str(path), metadata={"role": role, "schema": "test-k3-corpus-ids"})
    return path


@pytest.fixture(scope="module")
def corpus_files(work: Path, fixture: dict) -> dict:
    vocab = fixture["index"].text_config["vocab_size"]
    evidence = write_ids(work / "ids.safetensors", rows=4, seq=16, vocab=vocab, seed=11,
                         split=[0, 0, 0, 1], real_lengths=[16, 14, 16, 12], role="evidence")
    review = write_ids(work / "review.safetensors", rows=2, seq=8, vocab=vocab, seed=12,
                       split=[2, 2], real_lengths=[8, 5], role="review")
    other = write_ids(work / "ids-other.safetensors", rows=4, seq=16, vocab=vocab, seed=99,
                      split=[0, 0, 0, 1], real_lengths=[16, 14, 16, 12], role="evidence")
    return {"evidence": evidence, "review": review, "other": other}


def base_argv(model_dir: str, corpus: dict, out: Path, receipts: Path, **extra) -> list[str]:
    argv = ["--src-dir", model_dir, "--input-ids", str(corpus["evidence"]),
            "--review-input-ids", str(corpus["review"]),
            "--out", str(out), "--receipt-dir", str(receipts),
            "--device", "cpu", "--experts-device", "cpu", "--kda-backend", "torch",
            "--compute-dtype", "fp32", "--rung", "64", "--k-pool-factor", "1",
            "--snapshot-every", "1", "--row-chunk", "1", "--token-chunk", "7", "--chunk", "8",
            "--holdout-blocks", "6", "--max-gram-gb", "0", "--max-device-gb", "0",
            "--layers", "0-2", "--digest-receipt", str(out / "no-such-digest.json"),
            # synthetic fixtures have no import receipt: say so out loud (the default path is a refusal)
            "--no-layer0-receipt",
            # v1.2.0: the random 8-expert toy carries NO signal at a partial rung (held y_lat relL2 0.926
            # vs the prior's 0.925 at K=64), so the measured rule refuses it -- correctly.  The mechanics
            # tests (streaming, resume, staging) therefore run with the verdict RECORDED, not enforced;
            # the gate tests pass acceptance="gate" explicitly (argparse keeps the last occurrence)
            "--acceptance", "record"]
    for key, value in extra.items():
        flag = "--" + key.replace("_", "-")
        if (key == "no_layer0_receipt" and value is False) or key == "layer0_import_receipt":
            # an explicit receipt (or an explicit "no flag") means the sha-match is exercised
            if "--no-layer0-receipt" in argv:
                argv.remove("--no-layer0-receipt")
            if key == "no_layer0_receipt":
                continue
        if value is True:
            argv.append(flag)
        elif value is not None:
            argv += [flag, str(value)]
    return argv


def make_run(argv: list[str]) -> "kdp.DualStreamRun":
    """A DualStreamRun built the way main() builds it, without running the pass."""
    args = kdp.build_parser().parse_args(argv)
    args.model_dir = args.model_dir or args.src_dir
    policy = kdp.resolve_share_policy(args.src_dir, no_share_read=None, allow_share_read=False,
                                      share_rate_receipt=args.share_rate_receipt)
    index = k3_common.K3Index(args.src_dir)
    corpus = kdp.Corpus(args.input_ids, list(args.review_input_ids))
    stager = kdp.ShardStager(args.src_dir, None, enabled=False, window=1, prefetch=0,
                             delete_after_consume=False, run_id="t")
    return kdp.DualStreamRun(args, index, corpus, stager, policy)


def run_pass(argv: list[str]) -> dict:
    assert kdp.main(argv) == 0
    receipt_dir = argv[argv.index("--receipt-dir") + 1]
    with open(os.path.join(receipt_dir, "k3_dense_pass_receipt.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / max(float(b.float().norm()), 1e-30))


def sidecar_sha(receipt_dir: Path) -> str:
    """The full receipt's sha256 lives beside it (a file cannot carry its own digest)."""
    return (receipt_dir / kdp.RECEIPT_SHA_SIDECAR).read_text(encoding="utf-8").split()[0]


# ------------------------------------------------------------------ mechanics
def test_layer_halves_match_streamed_layer(fixture: dict, corpus_files: dict):
    """attention_half + ffn_half (row-chunked, token-chunked, experts on the host) ==
    `StreamedLayer.forward` -- teacher AND dense swap-in -- on every fixture layer."""
    index = fixture["index"]
    cfg = index.text_config
    corpus = kdp.Corpus(str(corpus_files["evidence"]), [])
    stager = kdp.ShardStager(fixture["dir"], None, enabled=False, window=1, prefetch=0,
                             delete_after_consume=False, run_id="t")
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(5)
    hidden = index.hidden_size
    seg = corpus.segments[0]
    checked = []
    for layer in range(index.num_layers):
        weights, raw_body, timing = kdp.read_layer(index, layer, stager, device, device, torch.float32)
        assert timing["bytes"] > 0 and raw_body
        runner = k3_layer.StreamedLayer(weights, cfg, kda_backend="torch")
        n_blocks = k3_common.blocks_before(layer, cfg["attn_res_block_size"])
        prefix = torch.randn(corpus.tokens, hidden, generator=generator) * 0.05
        blocks = torch.randn(corpus.tokens, n_blocks, hidden, generator=generator) * 0.05
        pair = kdp.Pair(prefix.clone(), blocks.clone())
        with torch.no_grad():
            ref_prefix, ref_blocks, ref_taps = runner.forward(
                prefix.view(seg.rows, seg.T, hidden).clone(), blocks.clone(), collect=True)
            carried, ffn_in, blocks_after = kdp.attention_half(
                runner, pair, corpus, device, row_chunk=1, token_chunk=7)
            new_prefix, taps = kdp.ffn_half(runner, carried, ffn_in, cfg, device=device, token_chunk=7)
        assert torch.equal(blocks_after, ref_blocks), f"layer {layer}: block stacks differ"
        score = rel(new_prefix, ref_prefix.reshape(-1, hidden))
        assert score < 1e-5, f"layer {layer}: prefix relL2 {score}"
        entry = {"layer": layer, "prefix_relL2": score, "moe": weights.is_moe}
        if weights.is_moe:
            assert rel(taps["y_lat"], ref_taps["y_lat"]) < 1e-5
            assert torch.equal(taps["router_topk_idx"].long().sort(-1).values,
                               ref_taps["router_topk_idx"].long().sort(-1).values)
            # the dense swap-in: a random form-E child through both paths
            k = 24
            latent = int(cfg["routed_expert_hidden_size"])
            dense = {"w1_s": torch.randn(k, latent, generator=generator) * 0.05,
                     "w3_s": torch.randn(k, latent, generator=generator) * 0.05,
                     "w2p": torch.randn(latent, k, generator=generator) * 0.05}
            with torch.no_grad():
                ref_out, _ = k3_layer.moe_forward(weights, ffn_in.unsqueeze(0), cfg, collect=False, dense=dense)
                ours, dense_taps = kdp.ffn_half(runner, torch.zeros_like(carried), ffn_in, cfg, device=device,
                                                token_chunk=7, dense=dense)
            entry["dense_relL2"] = rel(ours, ref_out.reshape(-1, hidden))
            assert entry["dense_relL2"] < 1e-5, entry
            assert dense_taps["dense_channels"] == k
        checked.append(entry)
    assert any(e["moe"] for e in checked) and any(not e["moe"] for e in checked)


def test_chunked_apply_attn_res_equals_unchunked():
    """run-book test 6, on a small T, fp32 and bf16, through the tool's own chunk size."""
    generator = torch.Generator().manual_seed(3)
    for dtype in (torch.float32, torch.bfloat16):
        prefix = (torch.randn(37, 64, generator=generator) * 0.1).to(dtype)
        blocks = (torch.randn(37, 3, 64, generator=generator) * 0.1).to(dtype)
        proj = (torch.randn(1, 64, generator=generator) * 0.1).to(dtype)
        norm = (torch.randn(64, generator=generator) * 0.1 + 1.0).to(dtype)
        reference = k3_common.apply_attn_res(prefix, blocks, proj, norm, 1e-5, token_chunk_size=None)
        for chunk in (1, 7, 8, 36, 37, 1024):
            chunked = k3_common.apply_attn_res(prefix, blocks, proj, norm, 1e-5, token_chunk_size=chunk)
            assert chunked.dtype == reference.dtype
            assert torch.equal(chunked, reference), (dtype, chunk)


def test_share_rate_refusal_only_when_source_on_share(tmp_path: Path):
    """run-book test 4: the share_rate receipt is consulted ONLY for a source on the share."""
    share = tmp_path / "share"
    on_share = share / "models" / "K3"
    on_share.mkdir(parents=True)
    local = tmp_path / "lexar" / "K3"
    local.mkdir(parents=True)
    receipt = tmp_path / "share_rate.json"

    # local source: no receipt needed, --no-share-read is the default
    policy = kdp.resolve_share_policy(str(local), no_share_read=None, allow_share_read=False,
                                      share_rate_receipt=str(receipt), share_root=str(share))
    assert policy["no_share_read"] is True and policy["source_on_share"] is False
    assert policy["share_rate"] is None
    # share source, receipt missing: refused
    with pytest.raises(SystemExit, match="missing"):
        kdp.resolve_share_policy(str(on_share), no_share_read=None, allow_share_read=False,
                                 share_rate_receipt=str(receipt), share_root=str(share))
    # share source, dd rate below 1 MB/s: refused
    receipt.write_text(json.dumps({"MB_per_s": 0.4}), encoding="utf-8")
    with pytest.raises(SystemExit, match="MB/s"):
        kdp.resolve_share_policy(str(on_share), no_share_read=None, allow_share_read=False,
                                 share_rate_receipt=str(receipt), share_root=str(share))
    # share source with a measured rate: allowed, recorded
    receipt.write_text(json.dumps({"MB_per_s": 25.5}), encoding="utf-8")
    policy = kdp.resolve_share_policy(str(on_share), no_share_read=None, allow_share_read=False,
                                      share_rate_receipt=str(receipt), share_root=str(share))
    assert policy["source_on_share"] and policy["no_share_read"] is False
    assert policy["share_rate"]["MB_per_s"] == 25.5
    # share source + --no-share-read is a contradiction
    with pytest.raises(SystemExit, match="no-share-read"):
        kdp.resolve_share_policy(str(on_share), no_share_read=True, allow_share_read=False,
                                 share_rate_receipt=str(receipt), share_root=str(share))
    # --allow-share-read overrides the rate check but records it
    receipt.write_text(json.dumps({"MB_per_s": 0.4}), encoding="utf-8")
    policy = kdp.resolve_share_policy(str(on_share), no_share_read=None, allow_share_read=True,
                                      share_rate_receipt=str(receipt), share_root=str(share))
    assert "overridden" in policy["note"]


# ------------------------------------------------------------------ the pass
def test_full_width_uniform_router_fold_is_exact_and_streams_agree(work: Path, uniform_fixture: dict,
                                                                     corpus_files: dict):
    """run-book test 1: with --rung = the full union width the folded stream equals the teacher."""
    cfg = uniform_fixture["index"].text_config
    full = int(cfg["num_experts"]) * int(cfg["moe_intermediate_size"])
    out, receipts = work / "exact-out", work / "exact-receipts"
    body = work / "exact-body"
    argv = base_argv(uniform_fixture["dir"], corpus_files, out, receipts, rung=full, body_dir=body,
                     acceptance="gate")
    receipt = run_pass(argv)
    assert receipt["layers_folded"] == [1, 2], receipt["layers_folded"]
    index = uniform_fixture["index"]
    for layer in ("1", "2"):
        entry = receipt["layers"][layer]
        assert entry["width_installed"] == full, entry["width_installed"]
        assert entry["fit_relL2_heldout"] < 1e-4, entry["fit_relL2_heldout"]
        assert entry["update_scale"] < 1e-3 and entry["veto_binding"] is False, entry["fold"]["ridge"]
        # v1.2.0: the measured acceptance rule, recorded and (here) passed
        acc = entry["acceptance"]
        assert acc["verdict"] == "accepted" and acc["enforced"] is True and acc["failing"] == [], acc
        assert set(acc["gates"]) == {"G1_y_lat_heldout", "G1_routed_write_heldout", "G2_held_over_fit",
                                     "G3_prefix_vs_zero_child"}
        assert acc["constants"] == kdp.ACCEPTANCE_CONSTANTS and acc["calibrated_on"]["date_utc"]
        assert entry["law_verdict"]["would_refuse"] is False and entry["law_verdict"]["message"] is None
        ridge = entry["fold"]["ridge"]
        assert len(ridge["ladder"]) == 9 and all("check_relL2" in r and "fit_relL2" in r for r in ridge["ladder"])
        assert ridge["lambda_relative"] == min(ridge["ladder"], key=lambda r: r["check_sse"])["lambda_relative"]
        held = entry["fold"]["heldout"]
        assert held["baselines"]["zero"]["y_lat_heldout"]["relL2"] == pytest.approx(1.0)
        assert held["baselines"]["prior"]["y_lat_heldout"]["relL2"] < 1e-3   # uniform router: the prior IS the answer
        assert held["y_lat_fit"]["relL2"] < 1e-4 and held["y_lat_heldout_runtime_dtype"]["dtype"] == "float32"
        assert entry["zero_child_prefix"]["held"]["relL2"] > 1e-3 and entry["zero_child_prefix"]["held"]["rows"] > 0
        marker = json.load(open(os.path.join(os.path.dirname(entry["ffn"]["path"]), "done.json")))
        assert marker["fold_metrics"]["acceptance"]["verdict"] == "accepted"
        assert marker["fold_metrics"]["law_verdict"]["would_refuse"] is False
        assert entry["stream_divergence"]["held"]["relL2"] < 1e-4, entry["stream_divergence"]
        assert entry["stream_divergence"]["real"]["relL2"] < 1e-4, entry["stream_divergence"]
        # the block stack is scored too (the prefix alone reads ~0 on snapshot layers)
        blocks = entry["stream_divergence"]["held"]["block_stack"]
        assert blocks is not None and blocks["blocks"] >= 1 and blocks["relL2"] < 1e-4, entry["stream_divergence"]
        assert entry["fold_metrics_source"] == "fit"
        assert [c["tensor"] for c in entry["finite_checks"]] == [
            "folded latent_in (fit input)", "teacher y_lat (fit target)",
            "teacher prefix after the layer", "folded prefix after the layer"]
        # --body-dir: the layer body verbatim -- attention + norms + residual maps, nothing MoE
        body_rec = entry["body"]
        assert body_rec["byte_exact"] and os.path.exists(body_rec["path"])
        body_tensors = load_file(body_rec["path"])
        assert any(k.startswith("self_attn.") for k in body_tensors) and "input_layernorm.weight" in body_tensors
        assert not any(k.startswith(("block_sparse_moe.", "mlp.")) or ".experts." in k for k in body_tensors)
        prefix = index.layer_prefix_for(int(layer))
        for short, tensor in body_tensors.items():
            source = load_file(os.path.join(uniform_fixture["dir"], index.weight_map[f"{prefix}{short}"]))
            assert torch.equal(tensor, source[f"{prefix}{short}"]), short
        assert entry["router"]["experts_touched"] == int(cfg["num_experts"])
        ffn = entry["ffn"]
        assert os.path.exists(ffn["path"]) and ffn["header_parameters"] == ffn["parameters"]["child_moe_total"]
        assert os.path.exists(os.path.join(os.path.dirname(ffn["path"]), "done.json"))
        got = load_file(ffn["path"])
        assert got["w1_s"].shape[0] == full and "block_sparse_moe.shared_experts.down_proj.weight" in got
        assert entry["held_rows"]["rows"] == receipt["corpus"]["held_tokens"]
        assert entry["dequant_seconds"] >= 0 and entry["pivot_seconds"] >= 0 and entry["peak_rss_bytes"]
    model_level = receipt["body_model_level"]
    assert model_level["byte_exact"] and set(model_level["names"]) == set(index.model_level_keys())
    assert "mlp.gate_proj.weight" not in load_file(receipt["layers"]["0"]["body"]["path"])
    assert receipt["installed_ffn_sha256"] == {"1": receipt["layers"]["1"]["ffn"]["sha256"],
                                               "2": receipt["layers"]["2"]["ffn"]["sha256"]}
    assert receipt["layers"]["0"]["mlp"]["import_receipt_comparison"]["skipped_by_flag"] == "--no-layer0-receipt"
    assert receipt["layers"]["0"]["folded"]["cloned_from_teacher"] is True
    assert receipt["layers"]["0"]["mlp"]["header_parameters"] > 0
    eos = receipt["end_of_stack"]
    assert eos["held"]["top1_agreement"] >= 0.99 and eos["held"]["kl_parent_child_mean"] < 1e-6, eos["held"]
    # G4: the first window with an end_of_stack number writes the calibration beside the receipts
    wa = receipt["window_acceptance"]
    assert wa["verdict"] == "calibrated" and os.path.exists(wa["calibration_written"]), wa
    assert wa["calibration"]["held_top1_agreement"] == eos["held"]["top1_agreement"]
    assert abs(eos["held"]["teacher_nll"] - eos["held"]["folded_nll"]) < 1e-4, eos["held"]
    assert eos["review"]["files"], eos["review"]
    review = load_file(next(iter(eos["review"]["files"].values()))["path"])
    for key in ("teacher_next_token_logprob", "folded_next_token_logprob", "teacher_topk_ids",
                "kl_teacher_folded", "row_index", "position"):
        assert key in review, list(review)
    assert int(review["position"].numel()) == (8 - 1) + (5 - 1)
    assert float(review["kl_teacher_folded"].max()) < 1e-6
    assert receipt["final_prefix_sha256"]["teacher"] and receipt["determinism_sha256"]
    assert receipt["share_policy"]["no_share_read"] is True and receipt["share_policy"]["source_on_share"] is False
    crosscheck = receipt["layers"]["1"].get("dequant_crosscheck")
    if crosscheck is not None and "error" not in crosscheck:
        assert crosscheck["bit_exact"] is True, crosscheck


def test_t11c_two_stream_resume_reproduces_a_single_pass(work: Path, fixture: dict, corpus_files: dict):
    """run-book test 2: a pass cut in two and resumed from the two-pair snapshot lands on the
    same timing-free receipt core as the straight pass; a killed pass resumed over done
    markers reuses its folds and lands there too."""
    straight_out, straight_rc = work / "straight-out", work / "straight-receipts"
    straight = run_pass(base_argv(fixture["dir"], corpus_files, straight_out, straight_rc))
    assert straight["layers_folded"] == [1, 2]
    assert straight["start"] == {"layer": 0, "from": "embedding"}

    seg_out, seg_rc = work / "segmented-out", work / "segmented-receipts"
    first = run_pass(base_argv(fixture["dir"], corpus_files, seg_out, seg_rc, layers="0-1"))
    first_sha = sidecar_sha(seg_rc)
    assert first["end_of_window_snapshot"]["layer"] == 2
    assert sorted(int(k) for k in first["layers"]) == [0, 1]
    for stream in ("teacher", "folded"):
        path = first["end_of_window_snapshot"]["files"][stream]["path"]
        assert os.path.exists(path)
        assert k3_common.read_metadata(path)["input_ids_sha256"] == first["corpus"]["corpus_sha256"]
    second = run_pass(base_argv(fixture["dir"], corpus_files, seg_out, seg_rc, layers="2-2", resume=True))
    assert second["start"]["from"] == "dual_snapshot" and second["start"]["layer"] == 2
    assert second["resumed_from_receipt"]["carried"] is True
    assert second["resumed_from_receipt"]["layers_carried_forward"] == [0, 1]
    assert sorted(int(k) for k in second["layers"]) == [0, 1, 2]
    assert second["determinism_sha256"] == straight["determinism_sha256"], "the segmented pass drifted"
    assert second["final_prefix_sha256"] == straight["final_prefix_sha256"]
    for layer in ("1", "2"):
        assert second["layers"][layer]["ffn"]["tensor_sha256"] == straight["layers"][layer]["ffn"]["tensor_sha256"]
        assert second["layers"][layer]["fit_skipped"] is None

    # the full receipt cannot carry its own sha: it is in the sidecar and the (newer) partial
    assert "receipt_sha256" not in second
    assert sidecar_sha(seg_rc) == k3_common.sha256_file(str(seg_rc / kdp.RECEIPT_NAME))
    partial = json.load(open(seg_rc / kdp.PARTIAL_RECEIPT_NAME, encoding="utf-8"))
    assert partial["receipt_sha256"] == sidecar_sha(seg_rc)
    assert second["resumed_from_receipt"]["previous_receipt_sha256"] == first_sha
    assert second["resumed_from_receipt"]["path"].endswith(kdp.PARTIAL_RECEIPT_NAME)

    # killed after the last fold, resumed over the done markers: nothing is refitted and the
    # reused layer keeps its fold metrics (they come back from the done marker)
    os.remove(seg_rc / "k3_dense_pass_receipt.json")
    third = run_pass(base_argv(fixture["dir"], corpus_files, seg_out, seg_rc, layers="0-2", resume=True))
    assert third["start"]["from"] == "dual_snapshot" and third["start"]["layer"] == 2
    assert third["start"]["installed_layers"] == [1]
    assert third["layers"]["2"]["fit_skipped"] is not None, third["layers"]["2"].get("fit_skipped")
    assert third["layers"]["2"]["ffn"]["reused"] is True
    assert third["layers"]["2"]["fold_metrics_source"] == "done_marker"
    for key in kdp.FOLD_METRIC_KEYS:
        assert key in third["layers"]["2"], key
    for key in ("veto_binding", "update_scale", "fit_relL2_heldout", "width_installed"):
        assert third["layers"]["2"][key] == straight["layers"]["2"][key], key
    # the recorded verdict and the law's verdict come back from the marker too
    assert third["layers"]["2"]["acceptance"]["verdict"] == straight["layers"]["2"]["acceptance"]["verdict"]
    assert third["layers"]["2"]["law_verdict"] == straight["layers"]["2"]["law_verdict"]
    # G4 on the resumed window gates against the calibration the previous window wrote
    assert second["window_acceptance"]["verdict"] == "calibrated"
    assert third["window_acceptance"]["verdict"] == "accepted", third["window_acceptance"]
    # the ridge record minus its measured-memory block (peaks are measurements, not results)
    ridge = {name: {k: v for k, v in r["layers"]["2"]["fold"]["ridge"].items() if k != "memory"}
             for name, r in (("third", third), ("straight", straight))}
    assert ridge["third"] == ridge["straight"]
    assert third["layers"]["2"]["ffn"]["parameters"] == straight["layers"]["2"]["ffn"]["parameters"]
    assert third["layers"]["2"]["ffn"]["header_parameters"] == straight["layers"]["2"]["ffn"]["header_parameters"]
    assert third["determinism_sha256"] == straight["determinism_sha256"], "resume over done markers drifted"
    assert third["end_of_stack"]["held"] == straight["end_of_stack"]["held"]
    # the layer-2 done marker itself carries the metrics
    marker = json.load(open(os.path.join(os.path.dirname(third["layers"]["2"]["ffn"]["path"]), "done.json")))
    assert marker["fold_metrics"]["veto_binding"] == straight["layers"]["2"]["veto_binding"]

    # a snapshot stamped with another corpus is not a resume point
    other = dict(corpus_files, evidence=corpus_files["other"])
    with pytest.raises(SystemExit, match="no dual snapshot"):
        kdp.main(base_argv(fixture["dir"], other, seg_out, work / "foreign-receipts", layers="2-2", resume=True))


def test_staged_deletion_touches_only_files_the_run_created(work: Path, fixture: dict, corpus_files: dict):
    """run-book test 3: with --delete-staged-after-consume the stager removes only the
    shards THIS run copied, only after both streams consumed the layer; a decoy and an
    unmanaged file under a shard's name are neither used nor touched."""
    index = fixture["index"]
    stage = work / "stage"
    stage.mkdir()
    decoy = stage / "decoy.txt"
    decoy.write_bytes(b"not a shard, not yours")
    foreign_shard = index.shards_for(index.layer_keys(2))[0]
    foreign = stage / foreign_shard
    foreign.write_bytes(b"garbage that must be neither read nor deleted")
    out, receipts = work / "staged-out", work / "staged-receipts"
    receipt = run_pass(base_argv(fixture["dir"], corpus_files, out, receipts, stage_dir=stage, force_stage=True,
                                 delete_staged_after_consume=True, stage_window=1, prefetch=1))
    staging = receipt["staging"]
    assert staging["enabled"] is True
    events = staging["events"]
    staged = {e["shard"] for e in events if e["action"] == "staged"}
    deleted = {e["shard"] for e in events if e["action"] == "deleted_after_both_streams_consumed"}
    run_id = staging["run_id"]
    assert all(e["run_id"] == run_id for e in events if e["action"].startswith(("staged", "deleted")))
    assert deleted <= staged, (deleted, staged)
    expected = {s for layer in (0, 1) for s in index.shards_for(index.layer_keys(layer))}
    assert deleted == expected, (deleted, expected)
    assert foreign_shard not in staged and foreign_shard not in deleted
    assert any(e["action"] == "unmanaged_file_left_alone" and e["shard"] == foreign_shard for e in events)
    assert decoy.read_bytes() == b"not a shard, not yours"
    assert foreign.read_bytes() == b"garbage that must be neither read nor deleted"
    # layer 2 was read from the source, not from the garbage
    shard_reads = receipt["layers"]["2"]["read"]["shards"]
    assert all(not s["staged"] for s in shard_reads), shard_reads
    assert receipt["layers_folded"] == [1, 2]
    # every deletion happened after the layer that consumed the shard finished
    for layer in ("0", "1"):
        released = receipt["layers"][layer]["staged_release"]
        assert any(r["action"] == "deleted_after_both_streams_consumed" for r in released), released
    manifest = json.load(open(stage / kdp.STAGING_MANIFEST, encoding="utf-8"))
    assert not (set(manifest["staged"]) & deleted)
    # model-level shards (embedding, head) were read from the source, never staged, and
    # nothing this run staged is still holding a window slot
    head_shard = index.weight_map[f"{index.model_prefix}embed_tokens.weight"]
    assert head_shard not in staged and head_shard not in manifest["staged"]
    assert any(e["action"] == "model_level_read_from_source" and e["shard"] == head_shard for e in events)
    assert staging["window_residents"] == [] and staging["still_staged_by_this_run"] == []


def test_stager_refuses_by_name_instead_of_hanging(work: Path, fixture: dict):
    """The window can only be released by the main thread; a wait that needs a release the
    waiter itself owes is a deadlock and is refused with the residents named.  With
    deletion off, a consumed shard leaves the window but stays on disk."""
    index = fixture["index"]
    shard_a = index.shards_for(index.layer_keys(0))[0]
    shard_b = index.shards_for(index.layer_keys(1))[0]
    stage = work / "stage-deadlock"
    stager = kdp.ShardStager(fixture["dir"], str(stage), enabled=True, window=1, prefetch=0,
                             delete_after_consume=True, run_id="dl")
    try:
        assert stager.path_for(shard_a) == str(stage / shard_a)
        with pytest.raises(SystemExit, match="stager deadlock") as caught:
            stager.path_for(shard_b)
        assert shard_a in str(caught.value) and shard_b in str(caught.value)
        # releasing what was consumed lets the next shard through
        stager.release([shard_a], still_needed=set())
        assert stager.path_for(shard_b) == str(stage / shard_b)
        assert not (stage / shard_a).exists() and (stage / shard_b).exists()
    finally:
        stager.close()

    stage_keep = work / "stage-keep"
    keeper = kdp.ShardStager(fixture["dir"], str(stage_keep), enabled=True, window=1, prefetch=0,
                             delete_after_consume=False, run_id="kp")
    try:
        keeper.path_for(shard_a)
        released = keeper.release([shard_a], still_needed=set())
        assert released[0]["action"] == "kept_no_delete_flag"
        assert keeper.owned[shard_a]["state"] == "consumed_kept" and (stage_keep / shard_a).exists()
        assert keeper.path_for(shard_b) == str(stage_keep / shard_b)  # no window slot held by the kept shard
        assert keeper.path_for(shard_a) == str(stage_keep / shard_a)  # still readable
        rec = keeper.record()
        assert rec["window_residents"] == [shard_b] and sorted(rec["still_staged_by_this_run"]) == sorted([shard_a, shard_b])
    finally:
        keeper.close()

    # the progress watchdog: a copy in flight that has not moved for stall_seconds is refused
    stalled = kdp.ShardStager(fixture["dir"], str(work / "stage-stall"), enabled=True, window=1, prefetch=0,
                              delete_after_consume=True, run_id="st", stall_seconds=1.0)
    stalled.owned["fake-shard"] = {"run_id": "st", "state": "copying"}
    stalled._copy_bytes = 12345
    stalled._copy_last_progress = kdp.time.time() - 5.0
    with pytest.raises(SystemExit, match="stager stalled"):
        stalled._check_stalled("fake-shard")
    stalled._copy_last_progress = kdp.time.time()
    stalled._check_stalled("fake-shard")  # progressing: no refusal


def test_digest_receipt_status_partial_is_not_all_verified(tmp_path: Path):
    """The Lexar digest receipt is written incrementally: 44 of 96 shards VERIFIED is
    'partial', never 'all_verified'; a mismatch or a missing file is never all_verified."""
    path = tmp_path / "digest.json"
    files = [{"file": f"model-{i:05d}-of-000096.safetensors", "status": "VERIFIED"} for i in range(1, 45)]
    path.write_text(json.dumps({"files": files, "verified": 44, "mismatch": 0, "missing": 0}), encoding="utf-8")
    got = kdp.digest_receipt_status(json.loads(path.read_text()), str(path), expected_files=96)
    assert got["files_verified"] == 44 and got["files_expected"] == 96
    assert got["partial"] is True and got["all_verified"] is False
    files = [{"file": f"model-{i:05d}-of-000096.safetensors", "status": "VERIFIED"} for i in range(1, 97)]
    path.write_text(json.dumps({"files": files, "verified": 96, "mismatch": 0, "missing": 0}), encoding="utf-8")
    got = kdp.digest_receipt_status(json.loads(path.read_text()), str(path), expected_files=96)
    assert got["partial"] is False and got["all_verified"] is True
    path.write_text(json.dumps({"files": files, "verified": 96, "mismatch": 1, "missing": 0}), encoding="utf-8")
    assert kdp.digest_receipt_status(json.loads(path.read_text()), str(path), expected_files=96)["all_verified"] is False
    # no top-level counters: derived from the file statuses
    files[3]["status"] = "MISSING"
    path.write_text(json.dumps({"files": files}), encoding="utf-8")
    got = kdp.digest_receipt_status(json.loads(path.read_text()), str(path), expected_files=96)
    assert got["missing"] == 1 and got["all_verified"] is False


def test_non_finite_fit_input_is_a_named_refusal(work: Path, fixture: dict, corpus_files: dict, monkeypatch):
    """Section 3b: a NaN in the teacher's y_lat never reaches the fp64 Gram -- the pass stops
    by name and the partial receipt records the refusal."""
    real = kdp.routed_y_lat

    def poisoned(weights, x, latent, cfg, device):
        y_lat, taps, seconds = real(weights, x, latent, cfg, device)
        y_lat[0, 0] = float("nan")
        return y_lat, taps, seconds

    monkeypatch.setattr(kdp, "routed_y_lat", poisoned)
    out, receipts = work / "nan-out", work / "nan-receipts"
    with pytest.raises(SystemExit, match="non-finite") as caught:
        kdp.main(base_argv(fixture["dir"], corpus_files, out, receipts))
    assert "teacher y_lat" in str(caught.value) and "layer 1" in str(caught.value)
    partial = json.load(open(receipts / kdp.PARTIAL_RECEIPT_NAME, encoding="utf-8"))
    assert partial["refusal"]["layer"] == 1 and "non-finite" in partial["refusal"]["message"]
    assert "1" not in partial["layers"] and not os.path.exists(out / "dense" / "L01" / "ffn.safetensors")


def test_missing_layer0_receipt_is_refused_unless_flagged(work: Path, fixture: dict, corpus_files: dict):
    """Run-book test 5 is not skipped silently: the default receipt path missing is a refusal
    naming --no-layer0-receipt; the flag records the skip."""
    missing = work / "no-such-layer0-receipt.json"
    with pytest.raises(SystemExit, match="no-layer0-receipt"):
        kdp.main(base_argv(fixture["dir"], corpus_files, work / "l0-missing-out", work / "l0-missing-receipts",
                           layers="0-0", layer0_import_receipt=missing, no_layer0_receipt=False))
    assert kdp.build_parser().parse_args(["--input-ids", "x"]).layer0_import_receipt == kdp.DEFAULT_LAYER0_IMPORT_RECEIPT


def test_window_above_layer0_rebuilds_the_dense_prefix_and_stamps_the_rung(work: Path, fixture: dict,
                                                                         corpus_files: dict):
    """The S2P command is `--layers 1-1` with no prior window: every layer below is dense, so
    the pair is rebuilt from the embedding through layer 0 instead of refusing.  The
    end-of-window snapshot carries the rung and the installed fold; a resume at another
    rung is refused, a resume at the same rung proceeds; the receipt carried on resume is
    the newer of full / partial by mtime."""
    out, receipts = work / "l11-out", work / "l11-receipts"
    receipt = run_pass(base_argv(fixture["dir"], corpus_files, out, receipts, layers="1-1"))
    receipt_sha = sidecar_sha(receipts)
    assert receipt_sha == k3_common.sha256_file(str(receipts / kdp.RECEIPT_NAME))
    assert receipt["start"]["from"] == "embedding" and receipt["start"]["auto_prefix_layers"] == [0]
    assert receipt["start"]["requested_first_layer"] == 1
    assert sorted(int(k) for k in receipt["layers"]) == [0, 1] and receipt["layers_folded"] == [1]
    assert receipt["layers"]["0"]["mlp"]["header_parameters"] > 0
    snap = receipt["end_of_window_snapshot"]
    assert snap["layer"] == 2 and snap["rung"] == 64 and snap["installed_layers"] == [1]
    for stream in ("teacher", "folded"):
        stamp = k3_common.read_metadata(snap["files"][stream]["path"])
        assert stamp["rung"] == "64" and stamp["tool_version"] == kdp.TOOL_VERSION
        assert json.loads(stamp["installed_ffn_sha256"]) == {"1": receipt["layers"]["1"]["ffn"]["sha256"]}

    # another rung: the snapshot (folds fitted at 64) is not a resume point, by name
    with pytest.raises(SystemExit, match="rung"):
        kdp.main(base_argv(fixture["dir"], corpus_files, out, work / "l11-other-rung", layers="2-2",
                           resume=True, rung=32))
    # the carried receipt is refused at another rung, too
    run = make_run(base_argv(fixture["dir"], corpus_files, out, receipts, layers="2-2", resume=True, rung=32))
    assert run.receipt["resumed_from_receipt"]["carried"] is False
    assert "rung" in run.receipt["resumed_from_receipt"]["reason"]
    # the same rung carries the NEWER file (the partial, written last) with the full receipt's sha
    run = make_run(base_argv(fixture["dir"], corpus_files, out, receipts, layers="2-2", resume=True))
    carried = run.receipt["resumed_from_receipt"]
    assert carried["carried"] is True and carried["path"].endswith(kdp.PARTIAL_RECEIPT_NAME)
    assert carried["previous_receipt_sha256"] == receipt_sha and carried["layers_dropped"] == {}
    assert carried["layers_carried_forward"] == [0, 1]
    full = receipts / kdp.RECEIPT_NAME
    later = os.path.getmtime(receipts / kdp.PARTIAL_RECEIPT_NAME) + 10
    os.utime(full, (later, later))
    run = make_run(base_argv(fixture["dir"], corpus_files, out, receipts, layers="2-2", resume=True))
    carried = run.receipt["resumed_from_receipt"]
    assert carried["path"] == str(full) and carried["previous_receipt_sha256"] == receipt_sha  # from the sidecar
    # a fold replaced on disk drops its receipt entry and un-stamps the snapshot
    ffn = Path(receipt["layers"]["1"]["ffn"]["path"])
    tensors = load_file(str(ffn))
    tensors["w2p"] = tensors["w2p"] * 2
    save_file(tensors, str(ffn))
    with pytest.raises(SystemExit, match="hashes differently"):
        make_run(base_argv(fixture["dir"], corpus_files, out, receipts, layers="2-2", resume=True))
    os.remove(ffn.parent / "done.json")
    run = make_run(base_argv(fixture["dir"], corpus_files, out, receipts, layers="2-2", resume=True))
    assert run.receipt["resumed_from_receipt"]["layers_dropped"] == {"1": run.receipt["resumed_from_receipt"]["layers_dropped"]["1"]}
    assert run.receipt["resumed_from_receipt"]["layers_carried_forward"] == [0]
    assert run.dual_snapshot_available(2) is False
    assert run.receipt["snapshots_refused"][0]["layer"] == 2 and "done marker" in run.receipt["snapshots_refused"][0]["reason"]
    # and with the fold gone, `--layers 2-2 --resume` has no resume point: refused by name
    with pytest.raises(SystemExit, match="no dual snapshot"):
        kdp.main(base_argv(fixture["dir"], corpus_files, out, work / "l11-damaged-receipts", layers="2-2",
                           resume=True))


def import_style_receipt(model_dir: str, index: k3_common.K3Index, layer: int, path: Path,
                         tamper: bool = False) -> Path:
    prefix = index.layer_prefix_for(layer)
    names = [f"{prefix}mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")]
    consumed = []
    for name in names:
        shard = os.path.join(model_dir, index.weight_map[name])
        rec = kdp.read_raw_tensor_records(shard, [name])[name]
        sha = rec["sha256_import_lane"]
        if tamper and name.endswith("up_proj.weight"):
            sha = "0" * 64
        consumed.append({"name": name, "dtype": rec["dtype"], "shape": rec["shape"], "sha256": sha})
    payload = {"schema": "test.k3_layer0_import_receipt", "layer_index": layer,
               "columns": [{"import_receipt": {"consumed": consumed}}]}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def test_layer0_verbatim_sha_matches_import_receipt(work: Path, fixture: dict, corpus_files: dict):
    """run-book test 5, on the fixture: the verbatim dense MLP sha-matches a receipt in the
    import lane's own format (dtype + shape + bytes), and a tampered receipt is refused."""
    index = fixture["index"]
    good = import_style_receipt(fixture["dir"], index, 0, work / "l0_receipt.json")
    out, receipts = work / "l0-out", work / "l0-receipts"
    receipt = run_pass(base_argv(fixture["dir"], corpus_files, out, receipts, layers="0-0",
                                 layer0_import_receipt=good))
    mlp = receipt["layers"]["0"]["mlp"]
    comparison = mlp["import_receipt_comparison"]
    assert comparison["compared"] and comparison["all_match"] and comparison["tensors_compared"] == 3, comparison
    assert all(t["byte_exact"] for t in mlp["tensors"].values())
    written = load_file(mlp["path"])
    for short in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
        source = load_file(os.path.join(fixture["dir"], index.weight_map[f"{index.layer_prefix_for(0)}{short}"]))
        assert torch.equal(written[short], source[f"{index.layer_prefix_for(0)}{short}"])
    assert receipt["end_of_window_snapshot"]["layer"] == 1
    bad = import_style_receipt(fixture["dir"], index, 0, work / "l0_receipt_bad.json", tamper=True)
    with pytest.raises(SystemExit, match="sha-match"):
        kdp.main(base_argv(fixture["dir"], corpus_files, work / "l0-bad-out", work / "l0-bad-receipts",
                           layers="0-0", layer0_import_receipt=bad))


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_K3, "model.safetensors.index.json"))
                    or not os.path.exists(REAL_L0_RECEIPT),
                    reason="the real K3 Lexar copy or k3_layer0_import_receipt.json is not here")
def test_real_k3_layer0_matches_k3_layer0_import_receipt():
    """run-book test 5 on the REAL checkpoint: the three dense-MLP tensors of layer 0, read
    off the Lexar copy, hash exactly as the import lane recorded them."""
    index = k3_common.K3Index(REAL_K3)
    with open(REAL_L0_RECEIPT, "r", encoding="utf-8") as handle:
        got = json.load(handle)
    assert int(got["layer_index"]) == 0
    consumed = {c["name"]: c for c in got["columns"][0]["import_receipt"]["consumed"]}
    prefix = index.layer_prefix_for(0)
    names = [f"{prefix}mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")]
    for name in names:
        shard = os.path.join(REAL_K3, index.weight_map[name])
        rec = kdp.read_raw_tensor_records(shard, [name])[name]
        assert rec["sha256_import_lane"] == consumed[name]["sha256"], name
        assert rec["shape"] == consumed[name]["shape"] and rec["dtype"] == consumed[name]["dtype"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the GPU cell needs a CUDA device")
def test_gpu_cell_dual_pass_cuda_fla(work: Path, fixture: dict, corpus_files: dict):
    """The one GPU cell: the dual pass on cuda:0 with the fla KDA backend at the parent's
    bf16, residual pairs on the host, experts on the host -- the real run's placement."""
    out, receipts = work / "gpu-out", work / "gpu-receipts"
    argv = base_argv(fixture["dir"], corpus_files, out, receipts)
    for flag, value in (("--device", "cuda:0"), ("--kda-backend", "fla"), ("--compute-dtype", "auto")):
        argv[argv.index(flag) + 1] = value
    argv += ["--max-device-gb", "-1"]
    receipt = run_pass(argv)
    assert receipt["device"] == "cuda:0" and receipt["kda_backend"] == "fla"
    assert receipt["experts_device"] == "cpu" and receipt["resid_device"] == "cpu"
    assert receipt["layers_folded"] == [1, 2]
    for layer in ("1", "2"):
        entry = receipt["layers"][layer]
        assert entry["layer_bytes_on_device"]["experts_packed"] == 0, entry["layer_bytes_on_device"]
        assert entry["layer_bytes_on_device"]["body"] > 0
        div = entry["stream_divergence"]["held"]["relL2"]
        assert div == div and div < 1.0, div
        assert entry["fold"]["ridge"]["device_memory"]["resident_total_bytes"] > 0
    assert receipt["end_of_stack"]["held"]["positions"] > 0


def test_auto_compute_dtype_casts_k3s_mixed_f32_rows(work: Path, fixture: dict, corpus_files: dict):
    """The shipped Kimi-K3 stores the KDA conv kernels / o_norm / A_log / dt_bias in
    F32 beside BF16 projections.  ``--compute-dtype auto`` must land the layer at
    the parent's bf16 (what ``transformers`` does at ``torch_dtype=bfloat16``) or
    ``F.conv1d`` dies on the bf16 activation against the F32 kernel -- the
    2026-09-06 GPU6 L0-1 launch.  The verbatim (raw) copies keep the on-disk dtype."""
    src = fixture["index"]
    kda = next(layer for layer in range(src.num_layers) if src.is_kda_layer(layer))
    mixed_dir = work / "synthetic-k3-mixed"
    if mixed_dir.exists():
        shutil.rmtree(mixed_dir)
    shutil.copytree(fixture["dir"], mixed_dir)
    index = k3_common.K3Index(str(mixed_dir))
    cfg = index.text_config
    prefix = index.layer_prefix_for(kda)
    f32_keys = [f"{prefix}self_attn.{name}" for name in
                ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "o_norm.weight")]
    shard = index.weight_map[f32_keys[0]]
    assert all(index.weight_map[key] == shard for key in f32_keys)
    path = os.path.join(str(mixed_dir), shard)
    tensors = load_file(path)
    for key in f32_keys:
        tensors[key] = tensors[key].to(torch.float32)
    save_file(tensors, path)
    index = k3_common.K3Index(str(mixed_dir))
    device = torch.device("cpu")
    corpus = kdp.Corpus(str(corpus_files["evidence"]), [])
    n_blocks = k3_common.blocks_before(kda, cfg["attn_res_block_size"])
    generator = torch.Generator().manual_seed(11)
    x_bf16 = (torch.randn(corpus.tokens, index.hidden_size, generator=generator) * 0.05).to(torch.bfloat16)

    # negative control: the uncast harness load reproduces the real-checkpoint failure
    raw, _ = k3_layer.load_layer(index, kda, "cpu", "cpu", log_reads=False)
    assert raw.attn["q_conv1d.weight"].dtype == torch.float32 and raw.attn["q_proj.weight"].dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="scalar type"):
        with torch.no_grad():
            k3_layer.StreamedLayer(raw, cfg, kda_backend="torch").attention(
                x_bf16[: corpus.segments[0].T].view(1, -1, index.hidden_size), state=None)

    stager = kdp.ShardStager(str(mixed_dir), None, enabled=False, window=1, prefetch=0,
                             delete_after_consume=False, run_id="t")
    weights, raw_body, timing = kdp.read_layer(index, kda, stager, device, device, None, keep_attention=True)
    assert timing["compute_dtype"] == "torch.bfloat16", timing
    assert timing["compute_dtype_source"].startswith("parent projection dtype"), timing
    floating = {f"{name}.{key}": tensor.dtype
                for name, group in (("attn", weights.attn), ("res", weights.res), ("body", weights.body))
                for key, tensor in group.items() if tensor.is_floating_point()}
    assert set(floating.values()) == {torch.bfloat16}, floating
    assert weights.attn["A_log"].dtype == torch.bfloat16  # F32 on disk in every fixture and in the real K3
    # the raw copy that feeds the verbatim writes and the sha match is untouched
    assert raw_body["self_attn.q_conv1d.weight"].dtype == torch.float32
    assert raw_body["self_attn.o_norm.weight"].dtype == torch.float32
    assert torch.equal(raw_body["self_attn.q_conv1d.weight"], tensors[f"{prefix}self_attn.q_conv1d.weight"])

    runner = k3_layer.StreamedLayer(weights, cfg, kda_backend="torch")
    pair = kdp.Pair(x_bf16.clone(),
                    (torch.randn(corpus.tokens, n_blocks, index.hidden_size, generator=generator) * 0.05)
                    .to(torch.bfloat16))
    with torch.no_grad():
        carried, ffn_in, _ = kdp.attention_half(runner, pair, corpus, device, row_chunk=1, token_chunk=7)
    assert carried.dtype == torch.bfloat16 and ffn_in.dtype == torch.bfloat16
    assert torch.isfinite(carried).all() and torch.isfinite(ffn_in).all()

    # an explicit --compute-dtype still wins
    fp32, _, timing32 = kdp.read_layer(index, kda, stager, device, device, torch.float32)
    assert timing32["compute_dtype"] == "torch.float32" and timing32["compute_dtype_source"] == "--compute-dtype"
    assert fp32.attn["q_proj.weight"].dtype == torch.float32


# ------------------------------------------------------------------ v1.2.0: the measured acceptance rule
def _gram_system(a_inner, a_check, y_inner, y_check):
    gi = a_inner.double().T @ a_inner.double()
    ci = a_inner.double().T @ y_inner.double()
    gc_ = a_check.double().T @ a_check.double()
    cc = a_check.double().T @ y_check.double()
    return gi, ci, gc_, cc, float(y_check.double().pow(2).sum())


def _solve(fn, system, prior, **kw):
    gi, ci, gc_, cc, cysq = system
    return fn(gi.clone(), ci.clone(), gc_.clone(), cc.clone(), cysq, prior.clone(), **kw)


def test_measured_solver_admits_a_fit_the_law_refuses_and_pins_c14_arithmetic():
    """The routed-union regime in miniature: the answer W_true is exactly recoverable, but the
    prior is 1e-3 of it (the parent column x a mean router mass).  THE LAW refuses every rung
    by construction; the measured solver installs the answer, records the law's verdict, and
    scores < 0.05 relL2 on rows it never saw.  A zero prior -- the honest prior for a
    from-scratch dense replacement -- no longer refuses either.  With a full-scale prior the
    two solvers agree rung for rung and on the installed map (the drift pin on the copied
    ladder arithmetic)."""
    g = torch.Generator().manual_seed(7)
    n, k, out = 600, 24, 6
    a = torch.randn(n, k, generator=g)
    w_true = torch.randn(out, k, generator=g)
    y = a @ w_true.T + 0.01 * torch.randn(n, out, generator=g)
    a_held = torch.randn(300, k, generator=g)
    y_held = a_held @ w_true.T
    system = _gram_system(a[:400], a[400:], y[:400], y[400:])
    fit_y_sq = float(y.double().pow(2).sum())

    # (1) a 1e-3-scaled prior: THE LAW refuses, the measured rule measures
    tiny = 1e-3 * w_true
    with pytest.raises(SystemExit, match="THE LAW refuses"):
        _solve(c14.law_abiding_solve_gram, system, tiny)
    stash: dict = {}
    solved, rec = _solve(kdp.measured_solve_gram, system, tiny, fit_y_sq=fit_y_sq, stash=stash)
    assert rel(solved.float(), w_true) < 0.02, rel(solved.float(), w_true)
    assert rel(a_held @ solved.float().T, y_held) < 0.05
    law = rec["law_verdict"]
    assert law["would_refuse"] is True and law["message"].startswith("no lambda on the ladder brings update_scale")
    assert law["status"].startswith("RECORDED") and law["lambdas_admitted"] == [] and rec["lambdas_vetoed"] == 9
    assert rec["free_solve_update_scale"] > 100 and rec["lambda_law_would_have_picked"] is None
    assert len(rec["ladder"]) == 9 and [r["lambda_relative"] for r in rec["ladder"]] == list(c14.LAMBDA_LADDER)
    for row in rec["ladder"]:
        assert all(key in row for key in ("check_sse", "check_relL2", "update_scale", "solution_norm", "fit_sse", "fit_relL2"))
    assert rec["lambda_relative"] == min(rec["ladder"], key=lambda r: r["check_sse"])["lambda_relative"]
    assert rec["free_fit_relL2"] is not None and rec["free_fit_relL2"] < 0.05
    assert torch.equal(stash["prior"], tiny) and "no norm veto" in rec["lambda_selected_on"]
    # the fit relL2 from the Gram identity IS the direct residual on the fit rows
    direct = rel(a @ solved.float().T, y)
    assert rec["chosen_fit_relL2_gram"] == pytest.approx(direct, abs=2e-3)

    # (2) a zero prior: THE LAW cannot even form its ratio; the measured solver still solves
    zsolved, zrec = _solve(kdp.measured_solve_gram, system, torch.zeros_like(w_true), fit_y_sq=fit_y_sq)
    assert rel(zsolved.float(), w_true) < 0.02
    assert zrec["update_scale"] is None and zrec["law_verdict"]["reference_prior_norm_zero"] is True
    assert zrec["law_verdict"]["would_refuse"] is True and "zero prior" in zrec["law_verdict"]["message"]
    assert zrec["veto_binding"] is True and zrec["lambdas_vetoed"] == 9

    # (3) the pin: a full-scale prior THE LAW admits -> both solvers pick the same rung and map
    full_prior = w_true + 0.05 * torch.randn(out, k, generator=g)
    law_w, law_rec = _solve(c14.law_abiding_solve_gram, system, full_prior)
    meas_w, meas_rec = _solve(kdp.measured_solve_gram, system, full_prior)
    assert law_rec["lambda_relative"] == meas_rec["lambda_relative"]
    assert torch.allclose(law_w, meas_w), rel(meas_w, law_w)
    assert meas_rec["law_verdict"]["would_refuse"] is False and meas_rec["lambdas_vetoed"] == law_rec["lambdas_vetoed"]
    for lr, mr in zip(law_rec["ladder"], meas_rec["ladder"]):
        assert lr["lambda_relative"] == mr["lambda_relative"]
        assert lr["check_sse"] == pytest.approx(mr["check_sse"], rel=1e-9)
        assert lr["update_scale"] == pytest.approx(mr["update_scale"], rel=1e-9)
    assert law_rec["free_solve_update_scale"] == pytest.approx(meas_rec["free_solve_update_scale"], rel=1e-9)


def _heldout(y, rw, fit, *, y_rt=None, rw_rt=None, prior_y=0.9775, prior_rw=0.9353, zero=1.0):
    block = lambda v: {"relL2": v, "cos": 0.0, "cos_p05": 0.0}  # noqa: E731
    return {
        "y_lat_heldout": block(y), "routed_write_heldout": block(rw),
        "y_lat_heldout_runtime_dtype": {**block(y if y_rt is None else y_rt), "dtype": "bfloat16"},
        "routed_write_heldout_runtime_dtype": {**block(rw if rw_rt is None else rw_rt), "dtype": "bfloat16"},
        "y_lat_fit": block(fit),
        "baselines": {"zero": {"y_lat_heldout": block(zero), "routed_write_heldout": block(zero)},
                      "prior": {"y_lat_heldout": block(prior_y), "routed_write_heldout": block(prior_rw)}},
    }


def test_evaluate_acceptance_on_the_l01_numbers():
    """The rule on the recorded K3 L01 numbers (H1 receipt): the check-split rung rel=1 passes
    every gate; the prior, the zero map and the free solve are refused by the gate that
    catches them; an exact fold passes on the floor; a prefix no better than the zero child
    fails G3.  Every gate carries value, threshold and pass."""
    stream, zero_child = {"relL2": 0.4319}, {"relL2": 0.5107}
    good = kdp.evaluate_acceptance(_heldout(0.6437, 0.7099, 0.5527), stream, zero_child)
    assert good["verdict"] == "accepted" and good["failing"] == [] and good["enforced"] is True
    g = good["gates"]
    assert g["G1_y_lat_heldout"]["threshold"] == pytest.approx(0.9 * 0.9775)
    assert g["G1_routed_write_heldout"]["threshold"] == pytest.approx(0.9 * 0.9353)
    assert g["G2_held_over_fit"]["value"] == pytest.approx(0.6437 / 0.5527) and g["G2_held_over_fit"]["threshold"] == 1.5
    assert g["G3_prefix_vs_zero_child"]["threshold"] == pytest.approx(0.95 * 0.5107)
    for gate in g.values():
        assert set(gate) >= {"value", "threshold", "pass", "note"}

    # the prior itself (relL2 0.9775 / 0.9353): refused by G1 on both scores
    prior = kdp.evaluate_acceptance(_heldout(0.9775, 0.9353, 0.97), stream, zero_child)
    assert prior["verdict"] == "refused" and prior["failing"] == ["G1_y_lat_heldout", "G1_routed_write_heldout"]
    # the zero map: refused by G1
    zero = kdp.evaluate_acceptance(_heldout(1.0, 1.0, 1.0), {"relL2": 0.5107}, zero_child)
    assert "G1_y_lat_heldout" in zero["failing"] and "G3_prefix_vs_zero_child" in zero["failing"]
    # the rel=1e-6 free solve (held 0.7503 / 0.8457, fit 0.3781): G1 on routed_write AND G2
    free = kdp.evaluate_acceptance(_heldout(0.7503, 0.8457, 0.3781), stream, zero_child)
    assert free["verdict"] == "refused"
    assert free["failing"] == ["G1_routed_write_heldout", "G2_held_over_fit"], free["failing"]
    assert free["gates"]["G2_held_over_fit"]["value"] == pytest.approx(0.7503 / 0.3781)
    msg = kdp.acceptance_message(1, 17408, free)
    assert "G2_held_over_fit 1.98" in msg and "threshold 1.5000" in msg and "nothing installed" in msg
    # a map that passes in fp32 and fails in the runtime dtype is a refusal (G1 scores the worst of the two)
    cast = kdp.evaluate_acceptance(_heldout(0.6437, 0.7099, 0.5527, y_rt=0.95), stream, zero_child)
    assert cast["failing"] == ["G1_y_lat_heldout"] and cast["gates"]["G1_y_lat_heldout"]["value"] == 0.95
    # G3: a folded prefix no better than the zero child's
    worse = kdp.evaluate_acceptance(_heldout(0.6437, 0.7099, 0.5527), {"relL2": 0.50}, zero_child)
    assert worse["failing"] == ["G3_prefix_vs_zero_child"]
    # an exact fold: ratios are noise over noise, the floor carries it
    exact = kdp.evaluate_acceptance(_heldout(2e-7, 3e-7, 1e-8, prior_y=1e-7, prior_rw=1e-7), {"relL2": 1e-7}, {"relL2": 0.2})
    assert exact["verdict"] == "accepted" and exact["gates"]["G2_held_over_fit"]["exact_fold"] is True
    assert exact["gates"]["G1_y_lat_heldout"]["exact_fold"] is True   # the prior IS the answer on a uniform router
    # but the floor is a floor: 2e-3 against a prior at 1e-7 is a refusal
    assert kdp.evaluate_acceptance(_heldout(2e-3, 2e-3, 1.5e-3, prior_y=1e-7, prior_rw=1e-7), {"relL2": 1e-7},
                                   {"relL2": 0.2})["failing"] == ["G1_y_lat_heldout", "G1_routed_write_heldout"]
    # a missing number never passes by default
    missing = kdp.evaluate_acceptance({"y_lat_heldout": {"relL2": 0.5}}, None, None)
    assert missing["verdict"] == "refused" and len(missing["failing"]) == 4
    # record mode: the verdict is computed, not enforced, and says so
    recorded = kdp.evaluate_acceptance(_heldout(1.0, 1.0, 1.0), stream, zero_child, mode="record")
    assert recorded["verdict"] == "refused" and recorded["enforced"] is False and recorded["mode"] == "record"


def test_a_no_signal_map_is_refused_and_nothing_is_installed(work: Path, fixture: dict, corpus_files: dict,
                                                             monkeypatch):
    """End to end on the sparse fixture: a solver that hands back the zero map (held relL2 1.0)
    is refused by name by G1; the partial receipt carries the gate, its value and threshold and
    the held-out numbers; dense/L01 has no ffn.safetensors and no done marker; the c14 solver
    is restored after the swap."""
    real = kdp.measured_solve_gram
    original_c14 = c14.law_abiding_solve_gram

    def no_signal(*args, **kw):
        solved, record = real(*args, **kw)
        return torch.zeros_like(solved), record

    monkeypatch.setattr(kdp, "measured_solve_gram", no_signal)
    out, receipts = work / "nosignal-out", work / "nosignal-receipts"
    with pytest.raises(SystemExit, match="refused by the measured acceptance rule") as caught:
        kdp.main(base_argv(fixture["dir"], corpus_files, out, receipts, acceptance="gate"))
    message = str(caught.value)
    assert "layer 1 K=64" in message and "G1_y_lat_heldout 1.0000 vs threshold" in message
    assert getattr(caught.value, "detail", {}).get("done_marker_written") is False
    assert c14.law_abiding_solve_gram is original_c14
    partial = json.load(open(receipts / kdp.PARTIAL_RECEIPT_NAME, encoding="utf-8"))
    refusal = partial["refusal"]
    assert refusal["layer"] == 1 and "G1_y_lat_heldout" in refusal["gate"]
    assert refusal["gates"]["G1_y_lat_heldout"]["value"] == pytest.approx(1.0)
    assert refusal["gates"]["G1_y_lat_heldout"]["threshold"] < 0.91
    assert refusal["heldout"]["baselines"]["zero"]["y_lat_heldout"]["relL2"] == pytest.approx(1.0)
    assert refusal["ridge"]["law_verdict"]["status"].startswith("RECORDED")
    assert refusal["zero_child_prefix_held"]["relL2"] > 0 and refusal["stream_divergence_held"]["relL2"] > 0
    assert "1" not in partial["layers"]
    layer_dir = out / "dense" / "L01"
    assert not os.path.exists(layer_dir / "ffn.safetensors") and not os.path.exists(layer_dir / "done.json")
    assert refusal["removed"] == [str(layer_dir / "ffn.safetensors")]
    # and a resume finds nothing to reuse at layer 1 (no marker), so the refusal is not sticky
    run = make_run(base_argv(fixture["dir"], corpus_files, out, receipts, resume=True))
    assert run.read_done_marker(1) is None


def test_the_sparse_toy_is_a_no_signal_fold_and_the_rule_says_so(work: Path, fixture: dict, corpus_files: dict):
    """The mechanics fixture (8 random experts, top-2, K=64 of 512) carries no signal the dense
    child can learn: under --acceptance gate layer 1 is refused by G1 with the numbers; under
    --acceptance record the same verdict is recorded (enforced False) and the pass completes,
    which is what the streaming / resume tests rely on.  Neither mode hides the number."""
    out, receipts = work / "nosig-gate-out", work / "nosig-gate-receipts"
    with pytest.raises(SystemExit, match="G1_y_lat_heldout") as caught:
        kdp.main(base_argv(fixture["dir"], corpus_files, out, receipts, acceptance="gate"))
    detail = caught.value.detail
    assert detail["gates"]["G1_y_lat_heldout"]["value"] > detail["gates"]["G1_y_lat_heldout"]["threshold"]
    assert not os.path.exists(out / "dense" / "L01" / "done.json")
    recorded = run_pass(base_argv(fixture["dir"], corpus_files, work / "nosig-rec-out", work / "nosig-rec-receipts"))
    entry = recorded["layers"]["1"]
    assert entry["acceptance"]["mode"] == "record" and entry["acceptance"]["enforced"] is False
    assert entry["acceptance"]["verdict"] == "refused" and "G1_y_lat_heldout" in entry["acceptance"]["failing"]
    assert entry["acceptance"]["gates"]["G1_y_lat_heldout"]["value"] == pytest.approx(
        detail["gates"]["G1_y_lat_heldout"]["value"], rel=1e-6)
    assert recorded["arguments"]["acceptance"] == "record"
    marker = json.load(open(os.path.join(os.path.dirname(entry["ffn"]["path"]), "done.json")))
    assert marker["fold_metrics"]["acceptance"]["enforced"] is False
