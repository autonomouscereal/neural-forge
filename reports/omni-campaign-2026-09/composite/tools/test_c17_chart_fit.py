"""Tests for c17_chart_fit.py (run-book row C17, tests (1)-(4) plus the mechanics they
rest on, plus one test per review finding of 2026-09-06).  CPU only, synthetic rows,
tiny widths; no model weights.

Run on the server:

    cd $OC/composite/tools && /home/cereal/model-surgery-lab/venv/bin/python \
        -m pytest -o addopts="" -p no:cacheprovider test_c17_chart_fit.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c17_chart_fit as c17  # noqa: E402

D_K3 = 12
D_TRUNK = 8
SEQ = 64
ROWS = 2
REAL = 60  # real tokens per row on both sides
CORPUS_SHA = c17.sha256_text("synthetic k3_dense_pass corpus")  # what k3_dense_pass stamps into held_rows metadata
IDS_SHA = c17.sha256_text("synthetic k3_input_ids int64 bytes")  # sha256 over the int64 ids bytes (both writers)


def orthonormal(rows: int, cols: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(rows, cols, generator=generator, dtype=torch.float64))
    return q[:, :cols]


def sealed(boundary: dict) -> dict:
    """Seal a boundary map exactly as k3_corpus_build.write_json does."""

    body = {k: v for k, v in boundary.items() if k != c17.BOUNDARY_SEAL_FIELD}
    body[c17.BOUNDARY_SEAL_FIELD] = c17.estate_canonical_sha256(body)
    return body


def write_k3_pass_receipt(k3_dir: Path, *, corpus_sha: str = CORPUS_SHA, ids_sha: str = IDS_SHA,
                          name: str = "k3_dense_pass_receipt.json") -> Path:
    """A minimal k3_dense_pass receipt beside the resid/ dir (the tool's first search location)."""

    receipt = {"schema": c17.K3_PASS_RECEIPT_SCHEMA, "tool": "k3_dense_pass",
               "corpus": {"corpus_sha256": corpus_sha, "tokens": ROWS * SEQ,
                          "segments": [{"path": "k3_input_ids.safetensors", "role": "evidence", "rows": ROWS,
                                        "sequence_length": SEQ, "input_ids_sha256": ids_sha}]}}
    path = k3_dir.parent / name
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# (1) alignment
# --------------------------------------------------------------------------------------


def test_alignment_92_64_covers_all_93_k3_layers_exactly_once():
    table = c17.alignment_table(92, 64, "trunk0")
    summary = c17.alignment_summary(table, 92, "trunk0")
    covered = [l for row in table for l in row["k3_layers"]]
    assert sorted(covered) == list(range(93))
    assert len(covered) == 93 == len(set(covered))
    assert summary["covered_exactly_once"] is True
    assert summary["trunk_layers_with_two_moe_layers"] == 28
    assert summary["trunk_layers_with_one_moe_layer"] == 36
    assert summary["trunk_layers_with_other_counts"] == 0
    assert table[0]["k3_layers"] == [0, 1]
    assert table[0]["moe_layers"] == [1]
    assert table[63]["moe_layers"][-1] == 92
    # contiguous, monotone intervals
    for a, b in zip(table, table[1:]):
        assert a["moe_stop"] == b["moe_start"]


def test_alignment_skip_policy_leaves_dense_layer_out_and_bad_policy_refused():
    table = c17.alignment_table(92, 64, "skip")
    covered = [l for row in table for l in row["k3_layers"]]
    assert sorted(covered) == list(range(1, 93))
    assert c17.alignment_summary(table, 92, "skip")["covered_exactly_once"] is True
    with pytest.raises(SystemExit):
        c17.alignment_table(92, 64, "trunk1")
    with pytest.raises(SystemExit):
        c17.parse_alignment("64:92")
    assert c17.parse_alignment("92:64") == (92, 64)


# --------------------------------------------------------------------------------------
# (2) identity recovery
# --------------------------------------------------------------------------------------


def test_identity_recovery_ridge_cos_across_widths():
    generator = torch.Generator().manual_seed(7)
    n = 600
    x = torch.randn(n, D_K3, generator=generator, dtype=torch.float64)
    p = orthonormal(D_K3, D_TRUNK, 3)  # [12, 8] orthonormal columns
    y = x @ p
    split = c17.split_by_trunk_token(torch.arange(n), 0.25, 0)
    fitted = c17.fit_layer_charts(x, y, split["fit_mask"], split["held_mask"], arms=("ridge", "procrustes", "svd"))
    ridge = fitted["ridge"]
    assert ridge["held"]["forward"]["cos"] >= 1.0 - 1.0e-6
    assert ridge["held"]["forward"]["rel_l2"] <= 1.0e-4
    assert (ridge["C_in"] - p).abs().max() < 1.0e-4
    # Procrustes in the SVD chart: the chart spans x's top-8 PCs, not P's column space, so
    # only orthonormality is exact here; forward cos is recorded, not gated.
    c_in = fitted["procrustes"]["C_in"]
    assert (c_in.transpose(0, 1) @ c_in - torch.eye(D_TRUNK, dtype=torch.float64)).abs().max() < 1.0e-9
    assert fitted["svd"]["notes"]["rank"] == D_TRUNK


def test_identity_recovery_equal_width_ridge_and_procrustes():
    generator = torch.Generator().manual_seed(11)
    n = 500
    x = torch.randn(n, D_TRUNK, generator=generator, dtype=torch.float64)
    r = orthonormal(D_TRUNK, D_TRUNK, 5)
    y = x @ r
    split = c17.split_by_trunk_token(torch.arange(n), 0.2, 1)
    fitted = c17.fit_layer_charts(x, y, split["fit_mask"], split["held_mask"])
    for arm in ("ridge", "procrustes"):
        held = fitted[arm]["held"]
        assert held["forward"]["cos"] >= 1.0 - 1.0e-6, (arm, held["forward"])
        assert held["backward"]["cos"] >= 1.0 - 1.0e-6, (arm, held["backward"])
        assert held["roundtrip"]["cos"] >= 1.0 - 1.0e-6, (arm, held["roundtrip"])
    assert (fitted["procrustes"]["C_in"] - r).abs().max() < 1.0e-9
    # the SVD control round-trips exactly at full rank
    assert fitted["svd"]["held"]["roundtrip"]["cos"] >= 1.0 - 1.0e-9


# --------------------------------------------------------------------------------------
# finding 3: the svd control and Procrustes arms are centred (means folded into the biases)
# --------------------------------------------------------------------------------------


def test_svd_and_procrustes_arms_are_centred_so_a_shared_offset_survives():
    generator = torch.Generator().manual_seed(31)
    n = 600
    z = torch.randn(n, D_TRUNK, generator=generator, dtype=torch.float64)  # 8-dim latent
    lift = orthonormal(D_K3, D_TRUNK, 41)  # [12, 8]: the latent's span in K3 space
    basis, _ = torch.linalg.qr(torch.cat((lift, torch.randn(D_K3, D_K3 - D_TRUNK, generator=generator, dtype=torch.float64)), dim=1))
    offset_x = 30.0 * basis[:, D_TRUNK]  # a 30-sigma fixed offset OUTSIDE the latent span
    x = z @ lift.transpose(0, 1) + 0.01 * torch.randn(n, D_K3, generator=generator, dtype=torch.float64) + offset_x
    r = orthonormal(D_TRUNK, D_TRUNK, 43)
    offset_y = torch.full((D_TRUNK,), 25.0, dtype=torch.float64)
    y = z @ r + offset_y  # orthogonal image of the latent plus its own fixed offset
    split = c17.split_by_trunk_token(torch.arange(n), 0.2, 0)
    fitted = c17.fit_layer_charts(x, y, split["fit_mask"], split["held_mask"])
    svd, pro = fitted["svd"], fitted["procrustes"]
    assert svd["notes"]["centred"] is True and pro["notes"]["centred"] is True
    # svd control: the round-trip keeps the offset (C_out_bias = mean_x, C_in_bias = -mean_x C_in)
    mean_x = x[split["fit_mask"]].mean(dim=0)
    assert (svd["C_out_bias"] - mean_x).abs().max() < 1.0e-9
    assert (svd["C_in_bias"] + mean_x @ svd["C_in"]).abs().max() < 1.0e-9
    assert svd["held"]["roundtrip"]["cos"] >= 0.999 and svd["held"]["roundtrip"]["rel_l2"] <= 0.01
    # the uncentred chart (zero biases, the pre-fix behaviour) drops the offset: cos collapses
    x_held = x[split["held_mask"]]
    uncentred = c17.linalg.write_scores(x_held @ svd["C_in"] @ svd["C_out"], x_held)
    assert uncentred["cos"] < 0.5, uncentred
    # Procrustes on centred rows recovers the orthogonal map through both offsets
    assert pro["held"]["forward"]["cos"] >= 0.999 and pro["held"]["forward"]["rel_l2"] <= 0.01
    assert pro["held"]["backward"]["cos"] >= 0.999
    assert (pro["C_in_bias"] - (y[split["fit_mask"]].mean(dim=0) - mean_x @ pro["C_in"])).abs().max() < 1.0e-9
    uncentred_pro = c17.linalg.write_scores(x_held @ pro["C_in"], y[split["held_mask"]])
    assert uncentred_pro["cos"] < 0.5, uncentred_pro


# --------------------------------------------------------------------------------------
# (3) disjoint held-out split
# --------------------------------------------------------------------------------------


def test_split_by_trunk_token_is_disjoint_even_when_tokens_repeat():
    tokens = torch.tensor([5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 10, 11])  # pooled over two K3 layers
    split = c17.split_by_trunk_token(tokens, 0.25, 0)
    fit, held = split["fit_mask"], split["held_mask"]
    assert not bool((fit & held).any())
    assert bool((fit | held).all())
    fit_tokens = set(tokens[fit].tolist())
    held_tokens = set(tokens[held].tolist())
    assert fit_tokens.isdisjoint(held_tokens)
    assert split["unique_trunk_tokens"] == 7
    assert split["held_trunk_tokens"] == 2
    counts = {int(t): int((tokens == t).sum()) for t in held_tokens}
    assert int(held.sum()) == sum(counts.values())  # every copy of a held token is held
    with pytest.raises(SystemExit):  # a split that would hold every token is refused
        c17.split_by_trunk_token(torch.tensor([1, 2]), 0.99, 0)


# --------------------------------------------------------------------------------------
# synthetic fixture: 6:4 alignment, 7 K3 layers, 4 trunk layers, boundary map, taps
# --------------------------------------------------------------------------------------


def write_fixture(root: Path, *, k3_rows_fn=None, trunk_rows_fn=None, positions_2d: bool = False,
                  pairs_per_row: int = REAL, bad_pair: bool = False, k3_metadata_sha: str = CORPUS_SHA,
                  boundary_ids_sha: str = IDS_SHA, pass_receipt: bool = True, break_seal: bool = False) -> dict:
    generator = torch.Generator().manual_seed(123)
    root.mkdir(parents=True, exist_ok=True)
    k3_dir, trunk_dir = root / "resid", root / "taps"
    rows = []
    trunk_flat, k3_flat = [], []
    for r in range(ROWS):
        pairs = []
        for p in range(pairs_per_row):
            q, k = p, min(p + r, SEQ - 1)  # row 1's K3 side is shifted by one token
            pairs.append([q, k])
            trunk_flat.append(r * SEQ + q)
            k3_flat.append(r * SEQ + k)
        if bad_pair and r == 0:
            pairs.append([SEQ + 3, 0])
        rows.append({"file": "k3_input_ids", "row_index": r, "source": "evidence", "split": "fit",
                     "qwen_tokens_real": REAL, "k3_tokens_real": REAL, "aligned_pairs": len(pairs),
                     "separator_pairs": 0, "aligned_fraction": 1.0, "pairs": pairs})
    rows.append({"file": "k3_review_input_ids", "row_index": 0, "source": "bank", "split": "review",
                 "qwen_tokens_real": 10, "k3_tokens_real": 10, "aligned_pairs": 1, "separator_pairs": 0,
                 "aligned_fraction": 0.1, "pairs": [[0, 0]]})
    boundary = sealed({"schema": c17.BOUNDARY_SCHEMA, "sequence_length": SEQ, "review_sequence_length": None,
                       "aggregate": {"all": {"rows": 3}}, "rows": rows,
                       "evidence_manifest_file_sha256": "0" * 64, "evidence_manifest_sha256": "1" * 64,
                       "ids_content_sha256": {"k3_input_ids": {"int64": boundary_ids_sha, "fp32": "2" * 64}}})
    if break_seal:
        boundary["rows"][0]["aligned_fraction"] = 0.5  # edited after sealing
    (root / "boundary_map.json").write_text(json.dumps(boundary, indent=2, ensure_ascii=False), encoding="utf-8")
    trunk_positions = torch.tensor(sorted(set(trunk_flat)), dtype=torch.int64)
    k3_positions = torch.tensor(sorted(set(k3_flat)), dtype=torch.int64)
    # one latent row per K3 token; K3 layers see it plus their own noise, trunk layers see
    # it through a per-layer chart P_k plus noise -- so a real chart exists to recover
    base = torch.randn(k3_positions.numel(), D_K3, generator=generator)
    k3_of_trunk = {t: k for t, k in zip(trunk_flat, k3_flat)}
    k3_row_of = {int(p): i for i, p in enumerate(k3_positions.tolist())}
    trunk_source = torch.tensor([k3_row_of[k3_of_trunk[int(t)]] for t in trunk_positions.tolist()])
    trunk_rows = {}
    for k in range(4):
        p_k = orthonormal(D_K3, D_TRUNK, 100 + k).to(torch.float32)
        y = base[trunk_source] @ p_k + 0.05 * torch.randn(trunk_positions.numel(), D_TRUNK, generator=generator)
        if trunk_rows_fn:
            y = trunk_rows_fn(k, y)
        trunk_rows[k] = y
        payload = {"ffn_in": y}
        if positions_2d:
            payload["positions"] = torch.stack((trunk_positions // SEQ, trunk_positions % SEQ), dim=1)
        else:
            payload["held_positions"] = trunk_positions
        (trunk_dir / f"L{k:02d}").mkdir(parents=True)
        save_file(payload, str(trunk_dir / f"L{k:02d}" / "held_rows.safetensors"))
    k3_rows = {}
    for l in range(7):
        x = base + 0.05 * torch.randn(k3_positions.numel(), D_K3, generator=generator)
        if k3_rows_fn:
            x = k3_rows_fn(l, x)
        k3_rows[l] = x
        (k3_dir / f"L{l:02d}").mkdir(parents=True)
        save_file({"held_positions": k3_positions, "folded_moe_x": x.to(torch.bfloat16).to(torch.float32),
                   "teacher_moe_x": x}, str(k3_dir / f"L{l:02d}" / "held_rows.safetensors"),
                  metadata={"layer": str(l), "tool": "k3_dense_pass", "input_ids_sha256": k3_metadata_sha})
    pass_receipt_path = write_k3_pass_receipt(k3_dir) if pass_receipt else None
    return {"k3_dir": k3_dir, "trunk_dir": trunk_dir, "boundary": root / "boundary_map.json",
            "trunk_rows": trunk_rows, "k3_rows": k3_rows, "pairs_per_row": pairs_per_row,
            "pass_receipt": pass_receipt_path}


def run_args(fixture: dict, out: Path, *extra: str):
    argv = ["--k3-resid", str(fixture["k3_dir"]), "--trunk-taps", str(fixture["trunk_dir"]),
            "--boundary-map", str(fixture["boundary"]), "--out", str(out), "--alignment", "6:4",
            "--dense-layer0", "trunk0", "--no-gram-eig", *extra]
    if c17.under_root_disk(out):  # a tmp dir under /home must not trip the root-disk refusal
        argv.append("--allow-root-disk")
    return c17.build_parser().parse_args(argv)


def test_pipeline_end_to_end_small(tmp_path: Path):
    fixture = write_fixture(tmp_path)
    out = tmp_path / "charts"
    body = c17.run(run_args(fixture, out, "--min-paired-rows", "8", "--k3-stream", "teacher", "--chart-dtype", "float64"))
    assert body["layers_fitted"] == [0, 1, 2, 3]
    assert body["layers_underdetermined"] == []
    pairs = ROWS * fixture["pairs_per_row"]
    # trunk 0 pools K3 layers 0 and 1; trunk 1 pools 2 and 3; trunk 2 gets 4; trunk 3 pools 5 and 6
    assert body["paired_rows_per_trunk_layer"] == {"0": 2 * pairs, "1": 2 * pairs, "2": pairs, "3": 2 * pairs}
    assert body["alignment"]["summary"]["covered_exactly_once"] is True
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    # finding 2: the receipt goes out under the tool's own schema, not mediafold's
    assert receipt["schema"] == c17.RECEIPT_SCHEMA == "chimera_omni.composite.c17_chart_fit.receipt.v1"
    assert receipt["schema"] != c17.receipts.RECEIPT_SCHEMA
    assert receipt["mediafold_receipt_schema"] == c17.receipts.RECEIPT_SCHEMA
    assert c17.sha256_file(out / "receipt.json") == body["receipt_sha256"]
    # finding 5: the K3 rows are tied to the boundary map's ids through the pass receipt
    identity = receipt["corpus_identity"]
    assert identity["verified"] is True and all(c["match"] for c in identity["checks"])
    assert [c["check"] for c in identity["checks"]] == ["boundary_map_sha256 seal recomputed",
                                                        "k3_pass_receipt evidence ids == boundary_map ids"]
    assert identity["k3_pass_receipt"]["path"] == str(fixture["pass_receipt"])
    assert receipt["boundary_map"]["boundary_map_seal_verified"] is True
    for record in receipt["layers"]:
        assert record["status"] == "fitted"
        assert record["split"]["disjoint"] is True
        assert record["split"]["fit_rows"] + record["split"]["held_rows"] == record["paired_rows"]
        assert record["unique_trunk_tokens"] == pairs
        assert record["gate"]["rows"] == pairs and record["gate"]["underdetermined"] is False
        assert record["gram"]["k3_fit"]["finite"] and record["gram"]["trunk_fit"]["finite"]
        assert record["gram"]["k3_fit"]["dead_columns"] == 0
        assert all(entry["matches_pass_receipt"] is True for entry in record["k3_rows"])
        assert record["ridge_regime"]["forward"] == "overdetermined" and record["ridge_regime"]["backward"] == "overdetermined"
        for arm in ("ridge", "svd", "procrustes"):
            held = record["arms"][arm]["held"]
            assert held is not None and held["forward"]["rows"] == record["split"]["held_rows"]
            assert record["arms"][arm]["held_summary_source"] == "held"
            assert -1.0 - 1e-12 <= held["roundtrip"]["cos"] <= 1.0 + 1e-12
            written = record["files"][arm]
            path = Path(written["path"])
            assert path.is_file() and c17.sha256_file(path) == written["file_sha256"]
            state = load_file(str(path))
            assert list(state["C_in"].shape) == [D_K3, D_TRUNK]
            assert list(state["C_out"].shape) == [D_TRUNK, D_K3]
            assert state["C_in"].dtype == torch.float64
            assert written["parameters"] == 2 * D_K3 * D_TRUNK + D_K3 + D_TRUNK
        assert record["arms"]["svd"]["notes"]["centred"] is True
        assert c17.sha256_file(out / f"L{record['trunk_layer']:02d}" / "layer.json") == record["layer_json_sha256"]
        # the chart exists by construction (y = x P_k + 5% noise): the ridge recovers it
        assert record["arms"]["ridge"]["held"]["forward"]["cos"] >= 0.95, record["arms"]["ridge"]["held_summary"]
        assert record["arms"]["ridge"]["held"]["forward"]["rel_l2"] <= 0.25
    # trunk 0's pool holds the DENSE layer 0 and MoE layer 1 (--dense-layer0 trunk0)
    assert receipt["layers"][0]["k3_layers"] == [0, 1]
    record = receipt["layers"][2]
    assert record["k3_layers"] == [4]
    assert record["arms"]["ridge"]["notes"]["in"]["lambda_ladder"] is not None
    assert (out / "alignment.json").is_file()
    assert receipt["projected_bytes"]["layers"] == 4 and receipt["control_charts_written"] is True


def test_pipeline_teacher_vs_folded_stream_and_layer_subset(tmp_path: Path):
    fixture = write_fixture(tmp_path)
    folded = c17.run(run_args(fixture, tmp_path / "a", "--min-paired-rows", "8", "--layers", "2"))
    teacher = c17.run(run_args(fixture, tmp_path / "b", "--min-paired-rows", "8", "--layers", "2", "--k3-stream", "teacher"))
    assert folded["layers_fitted"] == [2] and teacher["layers_fitted"] == [2]
    assert folded["k3_rows_key"] == "folded_moe_x" and teacher["k3_rows_key"] == "teacher_moe_x"
    a = folded["layers"][0]["arms"]["ridge"]["held_summary"]["forward"]["cos"]
    b = teacher["layers"][0]["arms"]["ridge"]["held_summary"]["forward"]["cos"]
    assert a != b  # bf16-rounded folded rows are a different input than the fp32 teacher rows
    assert not (tmp_path / "a" / "L00").exists()


def test_positions_2d_layout_accepted(tmp_path: Path):
    fixture = write_fixture(tmp_path, positions_2d=True)
    body = c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "8", "--layers", "1"))
    assert body["layers"][0]["trunk_taps"]["position_key"] == "positions"
    assert body["layers"][0]["paired_rows"] == 2 * ROWS * REAL


# --------------------------------------------------------------------------------------
# finding 5: corpus identity (K3 pass receipt <-> boundary map <-> taps manifest)
# --------------------------------------------------------------------------------------


def test_corpus_identity_refusals_and_the_unverified_escape_hatch(tmp_path: Path, monkeypatch):
    # the pass's default receipt dir may hold a REAL receipt on the server: point the search elsewhere
    monkeypatch.setattr(c17, "DEFAULT_K3_PASS_RECEIPT_DIR", str(tmp_path / "no-such-receipt-dir"))
    # held_rows metadata stamped by another corpus build
    fixture = write_fixture(tmp_path / "meta", k3_metadata_sha="3" * 64)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "meta" / "charts", "--min-paired-rows", "8", "--layers", "2"))
    assert "input_ids_sha256" in str(caught.value) and "corpus_sha256" in str(caught.value) and "refusing" in str(caught.value)
    # the pass receipt's evidence ids are not the boundary map's ids
    fixture = write_fixture(tmp_path / "ids", boundary_ids_sha="4" * 64)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "ids" / "charts", "--min-paired-rows", "8", "--layers", "2"))
    assert "corpus identity mismatch" in str(caught.value) and "boundary_map.ids_content_sha256" in str(caught.value)
    # the boundary map was edited after k3_corpus_build sealed it
    fixture = write_fixture(tmp_path / "seal", break_seal=True)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "seal" / "charts", "--min-paired-rows", "8", "--layers", "2"))
    assert "boundary_map_sha256" in str(caught.value) and "recomputed" in str(caught.value)
    # no pass receipt anywhere: refused, unless the run is recorded as unverified
    fixture = write_fixture(tmp_path / "none", pass_receipt=False)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "none" / "charts", "--min-paired-rows", "8", "--layers", "2"))
    assert "--k3-pass-receipt" in str(caught.value) and "--allow-unverified-k3-corpus" in str(caught.value)
    body = c17.run(run_args(fixture, tmp_path / "none" / "charts", "--min-paired-rows", "8", "--layers", "2",
                            "--allow-unverified-k3-corpus"))
    assert body["corpus_identity"]["verified"] is False and body["corpus_identity"]["k3_pass_receipt"] is None
    assert any("NOT tied" in note for note in body["corpus_identity"]["skipped"])
    assert body["layers"][0]["k3_rows"][0]["matches_pass_receipt"] is None
    # an explicit receipt path is honoured; a missing one is a refusal, not a silent fallback
    fixture = write_fixture(tmp_path / "explicit")
    moved = tmp_path / "explicit" / "elsewhere.json"
    moved.write_bytes(fixture["pass_receipt"].read_bytes())
    fixture["pass_receipt"].unlink()
    body = c17.run(run_args(fixture, tmp_path / "explicit" / "charts", "--min-paired-rows", "8", "--layers", "2",
                            "--k3-pass-receipt", str(moved)))
    assert body["corpus_identity"]["k3_pass_receipt"]["path"] == str(moved)
    with pytest.raises(SystemExit):
        c17.run(run_args(fixture, tmp_path / "explicit" / "charts2", "--min-paired-rows", "8", "--layers", "2",
                         "--k3-pass-receipt", str(tmp_path / "explicit" / "absent.json")))
    # a bare-string ids_content_sha256 (older writer) is read the same way as the {int64, fp32} form
    assert c17.boundary_ids_int64_sha256({"file": "k3_input_ids", "ids_content_sha256": {"k3_input_ids": "ab"}}) == "ab"
    assert c17.boundary_ids_int64_sha256({"file": "k3_input_ids", "ids_content_sha256": {"k3_input_ids": {"int64": "cd"}}}) == "cd"
    assert c17.boundary_ids_int64_sha256({"file": "k3_input_ids", "ids_content_sha256": {}}) is None


# --------------------------------------------------------------------------------------
# the runner's tap layout: ffn_input_rows.safetensors (prenorm, no positions) + taps_receipt.json
# --------------------------------------------------------------------------------------


def runner_tap_oracle(manifest_rows: list[dict], tapped: list[int], *, stride: int, micro_batch: int) -> list[tuple[int, int]]:
    """LITERAL replica of run_omni_frozen_review's evidence tap selection (evidence_sequences ->
    StreamedChimera.run batching -> the `select` closure -> `x[mask]` row-major per batch),
    returning (row_index, position) in dump order.  Kept independent of c17's replay."""

    ordered = sorted(manifest_rows, key=lambda r: int(r["batch_index"]))
    row_index_of = {int(r["batch_index"]): i for i, r in enumerate(ordered)}
    chosen = [r for r in ordered if int(r["batch_index"]) in set(tapped)]
    sequences = [[int(x) for x in r["ids"]] for r in chosen]
    batch_of = [int(r["batch_index"]) for r in chosen]
    lengths = [len(s) for s in sequences]
    order = sorted(range(len(sequences)), key=lambda i: -lengths[i])
    batches = [order[i: i + micro_batch] for i in range(0, len(order), micro_batch)]
    counter = {"n": 0}
    taken: list[tuple[int, int]] = []
    for batch in batches:
        width = max(lengths[i] for i in batch)
        valid = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, i in enumerate(batch):
            valid[row, : lengths[i]] = True
        mask = torch.zeros_like(valid)
        flat = torch.nonzero(valid.reshape(-1)).flatten()
        for k in flat.tolist():
            if counter["n"] % stride == 0:
                mask.reshape(-1)[k] = True
            counter["n"] += 1
        for row, i in enumerate(batch):  # x[mask] is row-major
            for p in range(width):
                if bool(mask[row, p]):
                    taken.append((row_index_of[batch_of[i]], p))
    return taken


def write_runner_fixture(root: Path, *, stride: int = 3, micro_batch: int = 2, lengths=(56, 64),
                         batch_indices=(2, 5), pairs_per_row: int = 50, break_rows=False,
                         break_manifest_sha: bool = False, break_tap_sha: bool = False,
                         boundary_manifest_sha: str | None = None) -> dict:
    """``break_rows``: "both" (or True) truncates layer 1's file AND records the short count in
    the taps receipt; "file" truncates only the file and keeps the receipt honest."""

    generator = torch.Generator().manual_seed(321)
    root.mkdir(parents=True, exist_ok=True)
    k3_dir, trunk_dir = root / "resid", root / "taps"
    manifest_rows = [{"batch_index": b, "split": "held", "ids": [7] * n} for b, n in zip(batch_indices, lengths)]
    manifest = {"rows": manifest_rows, "sequence_length": SEQ, "tokenizer_sha256": "synthetic"}
    manifest_path = root / "evidence.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = c17.sha256_file(manifest_path)
    rows, trunk_flat, k3_flat = [], [], []
    for r in range(len(manifest_rows)):  # row_index = rank in sorted batch_index order
        pairs = [[p, min(p + r, SEQ - 1)] for p in range(pairs_per_row)]
        trunk_flat += [r * SEQ + q for q, _ in pairs]
        k3_flat += [r * SEQ + k for _, k in pairs]
        rows.append({"file": "k3_input_ids", "row_index": r, "source": "evidence", "split": "held",
                     "qwen_tokens_real": lengths[r], "k3_tokens_real": lengths[r], "aligned_pairs": len(pairs),
                     "separator_pairs": 0, "aligned_fraction": 1.0, "pairs": pairs})
    boundary = sealed({"schema": c17.BOUNDARY_SCHEMA, "sequence_length": SEQ, "rows": rows,
                       "evidence_manifest_file_sha256": boundary_manifest_sha or manifest_sha,
                       "evidence_manifest_sha256": "synthetic", "evidence_manifest_path": str(manifest_path),
                       "ids_content_sha256": {"k3_input_ids": {"int64": IDS_SHA, "fp32": "2" * 64}}})
    (root / "boundary_map.json").write_text(json.dumps(boundary), encoding="utf-8")
    tapped = runner_tap_oracle(manifest_rows, list(batch_indices), stride=stride, micro_batch=micro_batch)
    tapped_flat = [r * SEQ + p for r, p in tapped]
    k3_positions = torch.tensor(sorted(set(k3_flat)), dtype=torch.int64)
    k3_row_of = {int(p): i for i, p in enumerate(k3_positions.tolist())}
    k3_of_trunk = {t: k for t, k in zip(trunk_flat, k3_flat)}
    base = torch.randn(k3_positions.numel(), D_K3, generator=generator)
    layers_meta = []
    truncate = {True: "both", False: None}.get(break_rows, break_rows)
    for k in range(4):
        p_k = orthonormal(D_K3, D_TRUNK, 200 + k).to(torch.float32)
        y = torch.randn(len(tapped_flat), D_TRUNK, generator=generator)
        for i, t in enumerate(tapped_flat):
            if t in k3_of_trunk:
                y[i] = base[k3_row_of[k3_of_trunk[t]]] @ p_k + 0.05 * torch.randn(D_TRUNK, generator=generator)
        recorded_rows = int(y.shape[0])
        if truncate and k == 1:
            y = y[:-1]
            if truncate == "both":
                recorded_rows = int(y.shape[0])
        (trunk_dir / f"L{k:02d}").mkdir(parents=True)
        path = trunk_dir / f"L{k:02d}" / "ffn_input_rows.safetensors"
        save_file({"prenorm": y.to(torch.bfloat16)}, str(path), metadata={"schema": c17.TAP_SCHEMA, "layer": str(k)})
        file_sha = "5" * 64 if (break_tap_sha and k == 1) else c17.sha256_file(path)
        layers_meta.append({"layer": k, "rows": recorded_rows, "hidden": D_TRUNK, "dtype": "torch.bfloat16",
                            "path": str(path), "sha256": file_sha})
    receipt = {"schema": c17.TAP_SCHEMA, "rows_from": "evidence manifest rows",
               "convention": "prenorm = input of blocks.{i}.norm2 (post-attention residual); normed = its output",
               "evidence": {"manifest": str(manifest_path), "manifest_sha256": "0" * 64 if break_manifest_sha else manifest_sha,
                            "rows": list(batch_indices), "splits": ["held"], "held_only": True},
               "stride": stride, "tap_max_rows": sum(lengths) // stride, "layers": layers_meta}
    (trunk_dir / "taps_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for l in range(7):
        x = base + 0.05 * torch.randn(k3_positions.numel(), D_K3, generator=generator)
        (k3_dir / f"L{l:02d}").mkdir(parents=True)
        save_file({"held_positions": k3_positions, "folded_moe_x": x, "teacher_moe_x": x.clone()},
                  str(k3_dir / f"L{l:02d}" / "held_rows.safetensors"),
                  metadata={"layer": str(l), "tool": "k3_dense_pass", "input_ids_sha256": CORPUS_SHA})
    write_k3_pass_receipt(k3_dir)
    expected_paired = len(set(tapped_flat) & set(trunk_flat))
    return {"k3_dir": k3_dir, "trunk_dir": trunk_dir, "boundary": root / "boundary_map.json", "manifest": manifest_path,
            "manifest_rows": manifest_rows, "tapped": tapped, "expected_paired_per_k3_layer": expected_paired,
            "stride": stride, "micro_batch": micro_batch, "manifest_sha256": manifest_sha}


def test_replay_tap_selection_matches_the_runner_rule():
    manifest_rows = [{"batch_index": b, "split": "held", "ids": [1] * n}
                     for b, n in [(9, 20), (3, 33), (11, 33), (0, 5), (4, 17), (12, 1)]]
    for tapped, stride, mb in [([0, 3, 4, 9, 11, 12], 3, 2), ([3, 11], 1, 8), ([0, 4, 9], 5, 1), ([12], 2, 8)]:
        oracle = runner_tap_oracle(manifest_rows, tapped, stride=stride, micro_batch=mb)
        got, notes = c17.replay_tap_selection(manifest_rows, tapped, stride=stride, micro_batch=mb, sequence_length=40)
        assert got.tolist() == [r * 40 + p for r, p in oracle], (tapped, stride, mb)
        assert notes["selected_rows"] == len(oracle) and notes["valid_tokens"] == sum(len(manifest_rows[i]["ids"]) for i in range(6) if manifest_rows[i]["batch_index"] in tapped)
        assert int(torch.unique(got).numel()) == got.numel()
    # row_index is the rank among ALL manifest rows, not the batch_index itself
    got, _ = c17.replay_tap_selection(manifest_rows, [12], stride=1, micro_batch=8, sequence_length=40)
    assert got.tolist() == [5 * 40 + 0]
    with pytest.raises(SystemExit):  # unsorted tap list is not the runner's order
        c17.replay_tap_selection(manifest_rows, [9, 3], stride=1, micro_batch=8, sequence_length=40)
    with pytest.raises(SystemExit):  # unknown batch_index
        c17.replay_tap_selection(manifest_rows, [99], stride=1, micro_batch=8, sequence_length=40)


def test_pipeline_runner_tap_layout_end_to_end(tmp_path: Path):
    fixture = write_runner_fixture(tmp_path)
    out = tmp_path / "charts"
    body = c17.run(run_args(fixture, out, "--min-paired-rows", "8", "--tap-micro-batch", str(fixture["micro_batch"])))
    assert body["layers_fitted"] == [0, 1, 2, 3]
    assert body["trunk_positions"]["source"].startswith("replayed")
    assert body["trunk_positions"]["evidence_manifest"]["sha256_verified_against_taps_receipt"] is True
    assert body["trunk_positions"]["replay"]["stride"] == fixture["stride"]
    assert body["trunk_positions"]["replay"]["selected_rows"] == len(fixture["tapped"])
    # finding 5: manifest <-> boundary map <-> taps is closed and recorded
    checks = {c["check"]: c for c in body["corpus_identity"]["checks"]}
    assert checks["taps evidence manifest == boundary_map evidence manifest"]["match"] is True
    assert checks["taps evidence manifest == boundary_map evidence manifest"]["left"]["sha256"] == fixture["manifest_sha256"]
    assert checks["k3_pass_receipt evidence ids == boundary_map ids"]["match"] is True
    assert body["corpus_identity"]["skipped"] == []
    n = fixture["expected_paired_per_k3_layer"]
    assert n >= 16  # the fixture pairs a meaningful share of the tapped rows
    assert body["paired_rows_per_trunk_layer"] == {"0": 2 * n, "1": 2 * n, "2": n, "3": 2 * n}
    for record in body["layers"]:
        assert record["trunk_taps"]["rows_key"] == "prenorm"
        assert record["trunk_taps"]["position_key"] == "replayed:taps_receipt"
        assert record["trunk_taps"]["source_dtype"] == "torch.bfloat16"
        assert record["trunk_taps"]["sha256_matches_taps_receipt"] is True  # finding 9
        assert record["unique_trunk_tokens"] == n
        assert record["split"]["disjoint"] is True
        # the chart exists by construction on the paired rows; unpaired tapped rows never enter the fit
        assert record["arms"]["ridge"]["held"]["forward"]["cos"] >= 0.95, record["arms"]["ridge"]["held_summary"]
    # an explicit --evidence-manifest pointing at the same file is accepted; a moved copy with the same bytes too
    copy = tmp_path / "copy.json"
    copy.write_bytes(fixture["manifest"].read_bytes())
    again = c17.run(run_args(fixture, tmp_path / "b", "--min-paired-rows", "8", "--layers", "2",
                             "--tap-micro-batch", str(fixture["micro_batch"]), "--evidence-manifest", str(copy)))
    assert again["paired_rows_per_trunk_layer"] == {"2": n}


def test_runner_tap_layout_refusals(tmp_path: Path):
    # finding 1: file AND receipt short by one row -> the replay refuses on the receipt's row count
    fixture = write_runner_fixture(tmp_path / "both", break_rows="both")
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "both" / "charts", "--min-paired-rows", "8", "--layers", "1",
                         "--tap-micro-batch", str(fixture["micro_batch"])))
    assert "replayed selection yields" in str(caught.value) and "refusing" in str(caught.value)
    # only the file is short (receipt honest) -> load_rows_file refuses on the file's row count
    fixture = write_runner_fixture(tmp_path / "file", break_rows="file")
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "file" / "charts", "--min-paired-rows", "8", "--layers", "1",
                         "--tap-micro-batch", str(fixture["micro_batch"])))
    assert "replayed tap selection" in str(caught.value) and "refusing" in str(caught.value)
    fixture = write_runner_fixture(tmp_path / "sha", break_manifest_sha=True)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "sha" / "charts", "--min-paired-rows", "8", "--layers", "1"))
    assert "sha256" in str(caught.value) and "refusing" in str(caught.value)
    # finding 5: the taps' manifest is not the boundary map's manifest
    fixture = write_runner_fixture(tmp_path / "bmap", boundary_manifest_sha="1" * 64)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "bmap" / "charts", "--min-paired-rows", "8", "--layers", "1"))
    assert "corpus identity mismatch" in str(caught.value) and "taps evidence manifest" in str(caught.value)
    # a wrong micro-batch on unequal lengths changes the dump order; the count still matches, so the
    # receipt must carry the micro-batch used (it does) -- here the equal-count case is exercised
    fixture = write_runner_fixture(tmp_path / "mb", lengths=(64, 64))
    body = c17.run(run_args(fixture, tmp_path / "mb" / "charts", "--min-paired-rows", "8", "--layers", "1"))
    assert body["trunk_positions"]["replay"]["micro_batch"] == c17.RUNNER_MICRO_BATCH == 8


def test_tap_file_sha256_mismatch_is_refused(tmp_path: Path):
    """finding 9: the runner's layers[].sha256 must match the file actually loaded."""

    fixture = write_runner_fixture(tmp_path, break_tap_sha=True)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "8", "--layers", "1",
                         "--tap-micro-batch", str(fixture["micro_batch"])))
    assert "layers[1].sha256" in str(caught.value) and "refusing" in str(caught.value)
    body = c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "8", "--layers", "0",
                            "--tap-micro-batch", str(fixture["micro_batch"])))
    assert body["layers"][0]["trunk_taps"]["sha256_matches_taps_receipt"] is True
    # nothing recorded -> nothing compared, recorded as None (the held_rows layout has no taps receipt)
    plain = write_fixture(tmp_path / "plain")
    body = c17.run(run_args(plain, tmp_path / "plain" / "charts", "--min-paired-rows", "8", "--layers", "1"))
    assert body["layers"][0]["trunk_taps"]["sha256_matches_taps_receipt"] is None


# --------------------------------------------------------------------------------------
# (4) the refusal -- finding 8: gated on unique trunk tokens / per-K3-layer pairs, not the pool
# --------------------------------------------------------------------------------------


def test_refuses_fewer_than_4096_paired_rows_without_the_flag(tmp_path: Path):
    fixture = write_fixture(tmp_path)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "charts"))
    message = str(caught.value)
    assert "4096" in message and "--allow-underdetermined" in message and "trunk layer 0" in message
    assert not (tmp_path / "charts" / "L00" / "ridge.safetensors").exists()
    assert c17.DEFAULT_MIN_PAIRED_ROWS == 4096


def test_gate_runs_on_unique_trunk_tokens_not_pooled_rows(tmp_path: Path):
    fixture = write_fixture(tmp_path)
    pairs = ROWS * REAL  # 120 distinct trunk tokens per K3 layer; trunk 0 pools two K3 layers -> 240 rows
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "200", "--layers", "0"))
    message = str(caught.value)
    assert "unique_trunk_tokens" in message and f"pooled {2 * pairs} paired rows" in message and "trunk layer 0" in message
    body = c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", str(pairs), "--layers", "0"))
    gate = body["layers"][0]["gate"]
    assert gate["rows"] == pairs and gate["pooled_paired_rows"] == 2 * pairs and gate["underdetermined"] is False
    assert body["gate_rows_per_trunk_layer"] == {"0": pairs}
    gate = c17.gate_quantity(50, [{"paired_rows": 40}, {"paired_rows": 90}], 130, 45)
    assert gate["rows"] == 40 and gate["quantity"] == "min_paired_rows_per_k3_layer" and gate["underdetermined"] is True
    gate = c17.gate_quantity(30, [{"paired_rows": 40}], 40, 45)
    assert gate["rows"] == 30 and gate["quantity"] == "unique_trunk_tokens"


def test_allow_underdetermined_fits_and_flags(tmp_path: Path):
    fixture = write_fixture(tmp_path)
    body = c17.run(run_args(fixture, tmp_path / "charts", "--allow-underdetermined", "--layers", "0,2"))
    assert body["layers_fitted"] == [0, 2]
    assert body["layers_underdetermined"] == [0, 2]
    for record in body["layers"]:
        assert record["underdetermined"] is True
        assert all(entry["underdetermined"] for entry in record["k3_rows"])
    assert (tmp_path / "charts" / "L02" / "ridge.safetensors").is_file()


def test_non_finite_rows_are_a_named_refusal(tmp_path: Path):
    def poison(l: int, x: torch.Tensor) -> torch.Tensor:
        if l == 4:
            x[3, 0] = float("nan")
        return x

    fixture = write_fixture(tmp_path, k3_rows_fn=poison)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "8", "--layers", "2"))
    assert "non-finite" in str(caught.value)


def test_pair_outside_sequence_length_refused(tmp_path: Path):
    fixture = write_fixture(tmp_path, bad_pair=True)
    with pytest.raises(SystemExit) as caught:
        c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "8", "--layers", "1"))
    assert "outside sequence lengths" in str(caught.value)


def test_dead_trunk_column_is_counted_not_fatal(tmp_path: Path):
    def kill(k: int, y: torch.Tensor) -> torch.Tensor:
        if k == 1:
            y[:, 5] = 0.0
        return y

    fixture = write_fixture(tmp_path, trunk_rows_fn=kill)
    body = c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "8", "--layers", "1"))
    gram = body["layers"][0]["gram"]["trunk_fit"]
    assert gram["dead_columns"] == 1 and gram["dead_column_index_head"] == [5]
    assert body["layers"][0]["arms"]["ridge"]["held"]["forward"]["cos"] is not None


def test_gram_diagnostics_eigen_and_refusal():
    generator = torch.Generator().manual_seed(2)
    rows = torch.randn(50, 6, generator=generator, dtype=torch.float64)
    diag = c17.gram_diagnostics(rows, "x", eig=True)
    assert diag["gram_rank_rel1e-12"] == 6 and diag["gram_eig_min"] > 0
    rows[0, 0] = float("inf")
    with pytest.raises(SystemExit):
        c17.gram_diagnostics(rows, "x")


def test_missing_rows_key_and_duplicate_positions_refused(tmp_path: Path):
    path = tmp_path / "held_rows.safetensors"
    save_file({"held_positions": torch.tensor([1, 1]), "ffn_in": torch.zeros(2, 3)}, str(path))
    with pytest.raises(SystemExit):
        c17.load_rows_file(str(path), ("ffn_in",), SEQ)
    save_file({"held_positions": torch.tensor([1, 2]), "other": torch.zeros(2, 3)}, str(path))
    with pytest.raises(SystemExit):
        c17.load_rows_file(str(path), ("ffn_in",), SEQ)
    with pytest.raises(SystemExit):
        c17.load_rows_file(str(tmp_path / "absent.safetensors"), ("ffn_in",), SEQ)


def test_ladder_subsample_and_shared_svd_chart_match_the_full_paths():
    generator = torch.Generator().manual_seed(21)
    n = 800
    x = torch.randn(n, D_K3, generator=generator, dtype=torch.float64)
    y = x @ orthonormal(D_K3, D_TRUNK, 9) + 0.01 * torch.randn(n, D_TRUNK, generator=generator, dtype=torch.float64)
    split = c17.split_by_trunk_token(torch.arange(n), 0.2, 0)
    sub = c17.fit_layer_charts(x, y, split["fit_mask"], split["held_mask"], ladder_rows=128)
    full = c17.fit_layer_charts(x, y, split["fit_mask"], split["held_mask"], ladder_rows=0)
    assert sub["ridge"]["notes"]["in"]["ladder_rows"] == 128
    assert full["ridge"]["notes"]["in"]["ladder_rows"] == int(split["fit_mask"].sum())
    # both refit on every fit row: the chart of record differs only through the chosen rung
    assert sub["ridge"]["held"]["forward"]["cos"] >= 0.99 and full["ridge"]["held"]["forward"]["cos"] >= 0.99
    # the procrustes arm reuses the svd arm's chart: identical to computing it alone
    alone = c17.fit_layer_charts(x, y, split["fit_mask"], split["held_mask"], arms=("procrustes",))
    assert (alone["procrustes"]["C_in"] - full["procrustes"]["C_in"]).abs().max() < 1.0e-12


# --------------------------------------------------------------------------------------
# finding 4: the ladder walks every fit row by default; both regimes are recorded
# --------------------------------------------------------------------------------------


def test_ladder_default_is_every_fit_row_and_design_regimes_are_recorded(tmp_path: Path):
    args = c17.build_parser().parse_args(["--k3-resid", "a", "--trunk-taps", "b", "--boundary-map", "c", "--out", "d"])
    assert args.ladder_rows == 0
    generator = torch.Generator().manual_seed(5)
    n = 800
    x = torch.randn(n, D_K3, generator=generator, dtype=torch.float64)
    y = x @ orthonormal(D_K3, D_TRUNK, 9)
    split = c17.split_by_trunk_token(torch.arange(n), 0.2, 0)
    fit_rows = int(split["fit_mask"].sum())
    full = c17.fit_layer_charts(x, y, split["fit_mask"], split["held_mask"], arms=("ridge",), ladder_rows=0)
    fwd, bwd = full["ridge"]["notes"]["in"], full["ridge"]["notes"]["out"]
    assert fwd["design_width"] == D_K3 + 1 and bwd["design_width"] == D_TRUNK + 1
    assert fwd["ladder_rows"] == fit_rows and fwd["ladder_inner_fit_rows"] == fit_rows - round(0.2 * fit_rows)
    assert fwd["ladder_regime"] == "overdetermined" and fwd["refit_regime"] == "overdetermined"
    assert full["ridge"]["notes"]["regime"] == {"forward": "overdetermined", "backward": "overdetermined",
                                                "ladder_forward": "overdetermined", "ladder_backward": "overdetermined"}
    # a 12-row ladder subsample leaves 10 inner-fit rows against 13 design columns: interpolation regime
    sub = c17.fit_layer_charts(x, y, split["fit_mask"], split["held_mask"], arms=("ridge",), ladder_rows=12)
    assert sub["ridge"]["notes"]["in"]["ladder_inner_fit_rows"] == 10
    assert sub["ridge"]["notes"]["in"]["ladder_regime"] == "underdetermined"
    assert sub["ridge"]["notes"]["in"]["refit_regime"] == "overdetermined"  # the refit still sees every fit row
    # fewer fit rows than design columns: the refit itself is ridge-only-solvable and says so
    # (12 rows, holdout 2 -> 10 fit rows: under the 13-column forward design, over the 9-column backward one)
    tiny = c17.split_by_trunk_token(torch.arange(12), 0.2, 0)
    small = c17.fit_layer_charts(x[:12], y[:12], tiny["fit_mask"], tiny["held_mask"], arms=("ridge",))
    assert small["ridge"]["notes"]["in"]["refit_fit_rows"] == 10
    assert small["ridge"]["notes"]["in"]["refit_regime"] == "underdetermined"
    assert small["ridge"]["notes"]["out"]["refit_regime"] == "overdetermined"
    # no ladder: no ladder regime, the refit regime still recorded
    plain = c17.fit_layer_charts(x, y, split["fit_mask"], split["held_mask"], arms=("ridge",), ladder=False)
    assert plain["ridge"]["notes"]["in"]["ladder_regime"] is None and plain["ridge"]["notes"]["in"]["refit_regime"] == "overdetermined"
    # the receipt lists the layers whose refit / ladder ran under-determined per direction
    fixture = write_fixture(tmp_path)
    body = c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "8", "--layers", "1"))
    assert body["layers"][0]["ridge_regime"]["forward"] == "overdetermined"
    assert body["layers_ridge_refit_underdetermined"] == {"forward": [], "backward": []}
    assert body["layers_ridge_ladder_underdetermined"] == {"forward": [], "backward": []}
    body = c17.run(run_args(fixture, tmp_path / "charts2", "--min-paired-rows", "8", "--layers", "1", "--ladder-rows", "12"))
    assert body["layers_ridge_ladder_underdetermined"] == {"forward": [1], "backward": []}


# --------------------------------------------------------------------------------------
# finding 7: no holdout -> held_summary is None, never the fit scores relabelled
# --------------------------------------------------------------------------------------


def test_held_summary_is_none_without_a_holdout(tmp_path: Path):
    generator = torch.Generator().manual_seed(8)
    n = 200
    x = torch.randn(n, D_K3, generator=generator, dtype=torch.float64)
    y = x @ orthonormal(D_K3, D_TRUNK, 9)
    fit_mask = torch.ones(n, dtype=torch.bool)
    fitted = c17.fit_layer_charts(x, y, fit_mask, ~fit_mask, arms=("ridge",))
    assert fitted["ridge"]["held"] is None
    assert c17.held_summary(fitted["ridge"]) is None
    assert fitted["ridge"]["fit"]["forward"]["rows"] == n
    fixture = write_fixture(tmp_path)
    body = c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "8", "--layers", "2", "--holdout-fraction", "0"))
    record = body["layers"][0]
    assert record["split"]["held_rows"] == 0
    for arm in ("ridge", "svd", "procrustes"):
        assert record["arms"][arm]["held"] is None
        assert record["arms"][arm]["held_summary"] is None and record["arms"][arm]["held_summary_source"] is None
        assert record["arms"][arm]["fit"]["forward"]["rows"] == record["paired_rows"]
    assert all(value is None for key, value in body["ridge_held"].items() if key != "source")


# --------------------------------------------------------------------------------------
# finding 6: root-disk refusal, projected footprint, optional control charts
# --------------------------------------------------------------------------------------


def test_root_disk_out_is_refused_and_the_footprint_is_projected(tmp_path: Path):
    assert c17.under_root_disk("/home/cereal/omni-campaign/composite/charts") is True
    assert c17.under_root_disk("/home") is True
    assert c17.under_root_disk("/mnt/lexar/omni-campaign/composite/charts") is False
    assert c17.under_root_disk("/homeless/charts") is False
    fixture = write_fixture(tmp_path)
    never = "/home/__c17_test_never_created__/charts"
    args = c17.build_parser().parse_args(["--k3-resid", str(fixture["k3_dir"]), "--trunk-taps", str(fixture["trunk_dir"]),
                                          "--boundary-map", str(fixture["boundary"]), "--out", never, "--alignment", "6:4",
                                          "--min-paired-rows", "8", "--layers", "1"])
    with pytest.raises(SystemExit) as caught:
        c17.run(args)
    assert "--allow-root-disk" in str(caught.value) and "/mnt/lexar/omni-campaign" in str(caught.value)
    assert not Path(never).exists()
    # the projection is printed before the first write and recorded; control charts optional
    body = c17.run(run_args(fixture, tmp_path / "charts", "--min-paired-rows", "8", "--layers", "1", "--skip-control-charts"))
    projected = body["projected_bytes"]
    per_arm = (2 * D_K3 * D_TRUNK + D_K3 + D_TRUNK) * 4 + 8 + c17.SAFETENSORS_HEADER_ALLOWANCE
    assert projected["bytes_per_arm"] == per_arm and projected["arms_written"] == ["ridge"]
    assert projected["bytes_per_layer"] == per_arm and projected["bytes_total"] == per_arm and projected["layers"] == 1
    assert body["control_charts_written"] is False
    record = body["layers"][0]
    assert record["files"]["ridge"] is not None and record["files"]["svd"] is None and record["files"]["procrustes"] is None
    assert per_arm - c17.SAFETENSORS_HEADER_ALLOWANCE <= record["files"]["ridge"]["bytes"] <= per_arm
    assert not (tmp_path / "charts" / "L01" / "svd_chart.safetensors").exists()
    for arm in ("svd", "procrustes"):  # fitted and scored, just not written
        assert record["arms"][arm]["written"] is False and record["arms"][arm]["held"]["forward"]["cos"] is not None
    assert record["arms"]["ridge"]["written"] is True
    full = c17.run(run_args(fixture, tmp_path / "charts_full", "--min-paired-rows", "8", "--layers", "1"))
    assert full["projected_bytes"]["bytes_per_layer"] == 3 * per_arm and full["projected_bytes"]["arms_written"] == list(c17.ARMS)


# --------------------------------------------------------------------------------------
# finding 10: resume reuses sha-verified layers; the receipt merges every L*/layer.json
# --------------------------------------------------------------------------------------


def test_resume_reuses_verified_layers_and_merges_the_receipt(tmp_path: Path):
    fixture = write_fixture(tmp_path)
    out = tmp_path / "charts"
    common = ("--min-paired-rows", "8")
    first = c17.run(run_args(fixture, out, *common, "--layers", "0,1"))
    assert first["layers_fitted"] == [0, 1] and first["layers_reused"] == [] and first["previous_receipt_sha256"] is None
    second = c17.run(run_args(fixture, out, *common, "--layers", "2"))
    assert second["layers_fitted"] == [2] and second["layers_reused"] == [0, 1] and second["layers_present"] == [0, 1, 2]
    assert second["previous_receipt_sha256"] == first["receipt_sha256"]
    assert sorted(second["paired_rows_per_trunk_layer"]) == ["0", "1", "2"]
    statuses = {r["trunk_layer"]: r["status"] for r in second["layers"]}
    assert statuses == {0: "reused", 1: "reused", 2: "fitted"}
    assert second["ridge_held"]["forward_cos"]["layers"] == 3
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["layers_present"] == [0, 1, 2] and receipt["layers"][0]["reuse"]["layer_json_sha256"]
    # a requested layer that verifies is reused, not refitted; --refit forces it
    third = c17.run(run_args(fixture, out, *common, "--layers", "2"))
    assert third["layers_fitted"] == [] and third["layers_reused"] == [0, 1, 2]
    assert third["layers"][2]["layer_json_sha256"] == second["layers"][2]["layer_json_sha256"]
    forced = c17.run(run_args(fixture, out, *common, "--layers", "2", "--refit"))
    assert forced["layers_fitted"] == [2] and forced["layers"][2]["refit_reason"] == "--refit"
    # a changed fit signature refits the requested layer and says why
    reseeded = c17.run(run_args(fixture, out, *common, "--layers", "1", "--seed", "5"))
    assert reseeded["layers_fitted"] == [1]
    assert any("fit signature" in reason for reason in reseeded["layers"][1]["refit_reason"])
    # a corrupted arm file: refitted when requested, otherwise the layer is marked invalid and excluded
    (out / "L01" / "ridge.safetensors").write_bytes(b"\x00" * 64)
    (out / "L00" / "svd_chart.safetensors").write_bytes(b"\x00" * 64)
    repaired = c17.run(run_args(fixture, out, *common, "--layers", "1"))
    assert repaired["layers_fitted"] == [1]
    assert any("sha256 differs" in reason for reason in repaired["layers"][1]["refit_reason"])
    assert repaired["layers_invalid"] == [0] and repaired["layers_present"] == [1, 2]
    invalid = [r for r in repaired["layers"] if r["trunk_layer"] == 0][0]
    assert invalid["status"] == "invalid" and any("svd" in reason for reason in invalid["reasons"])
    assert "0" not in repaired["paired_rows_per_trunk_layer"]
    # changed input rows invalidate the reuse
    (fixture["k3_dir"] / "L04" / "held_rows.safetensors").write_bytes((fixture["k3_dir"] / "L05" / "held_rows.safetensors").read_bytes())
    changed = c17.run(run_args(fixture, out, *common, "--layers", "2"))
    assert changed["layers_fitted"] == [2] and any("K3 layer 4" in reason for reason in changed["layers"][2]["refit_reason"])
