"""Tests for ``qf_stream27.py`` -- the streamed C2 engine.  CPU only, tiny fixture.

Everything here is a claim about MECHANICS on ``qf_arch.tiny_text_config``
(random-init, both layer kinds, fp32 on the CPU with the torch kernels forced,
exactly as ``qf_selftest.py`` runs the resident lane).  The four tests the run
book names for this tool are here as ``test_gram_accumulation_matches_resident_
on_one_site``, ``test_parity_receipt_deterministic``, ``test_tap_index_64_
distinct_grams_from_2_buffers`` and ``test_streamed_build_reproduces_resident_
build``; the 4B reproduction is the same last test at the real checkpoint and
is run on the server by hand (see the report), not here.

    cd $OC/composite/tools && CUDA_VISIBLE_DEVICES="" \\
        /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \\
        -p no:cacheprovider test_qf_stream27.py -q
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qf_stream27 as qs  # noqa: E402

qf_arch = qs.qf_arch
qf_corpus = qs.qf_corpus
qf_build = qs.qf_build
qf_export = qs.qf_export
qf_solve = qs.qf_solve
import qf_harness  # noqa: E402
import qf_selftest  # noqa: E402


def quiet(*_a, **_k) -> None:
    return None


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("qf_stream27")


@pytest.fixture(scope="module")
def tiny(work: Path):
    """The lane's own tiny parent + sealed evidence, via ``qf_selftest.tiny_parent``."""

    qf_selftest.WORK = work
    parent, manifest = qf_selftest.tiny_parent("parent")
    return {"parent": parent, "manifest": manifest, "evidence": work / "parent-evidence.json"}


@pytest.fixture(scope="module")
def tiny64(work: Path):
    """A 64-layer tiny parent: 48 linear + 16 full layers, TWO shared buffers."""

    directory = work / "parent64"
    if directory.exists():
        shutil.rmtree(directory)
    config = qf_arch.tiny_text_config(layers_count=64)
    directory.mkdir(parents=True)
    model = qf_arch.seeded_tiny_model(config)
    model.save_pretrained(str(directory), safe_serialization=True)
    manifest = qf_corpus.synthetic_manifest(
        vocab_size=int(config.vocab_size), sequence_length=32, batches=6, held_every=3,
        max_id=256, out=work / "parent64-evidence.json",
    )
    return {"parent": directory, "manifest": manifest}


def resident_site_gram(parent: Path, manifest, site: int):
    """The resident lane's own capture at one site: fp64 rows, then ``F^T F``."""

    harness = qf_harness.Harness(parent, "cpu", torch.float32, manifest)
    layer = harness.layers()[site]
    name = f"L{site:03d}"
    captured = harness.capture({name: layer.mlp}, outputs={name: layer.mlp})[name]
    gate = layer.mlp.gate_proj.weight.detach().to("cpu", torch.float64)
    up = layer.mlp.up_proj.weight.detach().to("cpu", torch.float64)
    phi = qf_solve.swiglu(captured["fit"], gate, up)
    return {
        "G": phi.T @ phi,
        "B": phi.T @ captured["out_fit"],
        "YY": float(captured["out_fit"].pow(2).sum()),
        "rows": int(captured["fit"].shape[0]),
        "phi": phi,
        "X": captured["fit"],
        "Y": captured["out_fit"],
        "harness": harness,
    }


# --------------------------------------------------------------------- tests
def test_full_sequence_preflight_preserves_pristine_trunk_and_parity(tiny):
    harness = qs.StreamedHarness(tiny["parent"], "cpu", torch.float32, tiny["manifest"])
    ids = harness.batches[0][1]
    before = harness.engine.final_hidden(ids).clone()
    report = qs.full_sequence_preflight(harness)
    assert report["finite_hidden_and_last_logits"] is True
    assert report["input_shape"] == list(ids.shape)
    assert report["final_hidden_shape"] == list(before.shape)
    assert report["cuda_peak_allocated_bytes"] is None
    assert report["host_rss_after_bytes"] > 0
    assert len(report["input_ids_sha256"]) == 64
    assert harness.residual.boundary == 0 and not harness.engine.layers.folded
    assert torch.equal(before, harness.engine.final_hidden(ids))
    assert qs.streamed_parity(harness)["deterministic"] is True
    harness.tap.remove()


def test_full_sequence_preflight_refuses_nonfinite_and_folded_state(tiny, monkeypatch):
    harness = qs.StreamedHarness(tiny["parent"], "cpu", torch.float32, tiny["manifest"])
    harness.residual.boundary = 1
    with pytest.raises(SystemExit, match="pristine"):
        qs.full_sequence_preflight(harness)
    harness.residual.boundary = 0
    ids = harness.batches[0][1]
    monkeypatch.setattr(harness.engine, "final_hidden", lambda _: torch.full(
        (*ids.shape, harness.engine.text_config.hidden_size), float("nan")))
    with pytest.raises(SystemExit, match="invalid final hidden"):
        qs.full_sequence_preflight(harness)
    harness.tap.remove()


def test_check_split_matches_fit_check_rows(tiny):
    """The per-batch check flag is the row mask ``qf_corpus.fit_check_rows`` builds."""

    manifest = tiny["manifest"]
    length = int(manifest["sequence_length"])
    mask = qf_corpus.fit_check_rows(manifest)
    fit_position = 0
    for row in sorted(manifest["rows"], key=lambda r: int(r["batch_index"])):
        if row["split"] != "fit":
            continue
        block = mask[fit_position * length:(fit_position + 1) * length]
        assert bool(block.all()) == qs.check_flag(fit_position)
        assert bool(block.any()) == qs.check_flag(fit_position)
        fit_position += 1


def test_split_gram_solve_matches_row_solve():
    """``law_abiding_solve_split_gram`` picks the same rung as the row form, same weight."""

    torch.manual_seed(20260902)
    n, k, h = 600, 40, 24
    A = torch.randn(n, k, dtype=torch.float64)
    A[:, :5] *= 30.0  # a badly scaled design, so the ladder has something to choose
    W_true = torch.randn(h, k, dtype=torch.float64)
    Y = A @ W_true.T + 0.3 * torch.randn(n, h, dtype=torch.float64)
    prior = W_true + 0.5 * torch.randn(h, k, dtype=torch.float64)
    check = torch.zeros(n, dtype=torch.bool)
    check[200:400] = True
    row_w, row_guard = qf_solve.law_abiding_solve_split(A, Y, check, prior)
    inner = ~check
    gram_w, gram_guard = qs.law_abiding_solve_split_gram(
        A[inner].T @ A[inner], A[inner].T @ Y[inner],
        A[check].T @ A[check], A[check].T @ Y[check], float(Y[check].pow(2).sum()),
        prior,
    )
    assert gram_guard["lambda_relative"] == row_guard["lambda_relative"]
    assert gram_guard["lambda_law_would_have_picked"] == row_guard["lambda_law_would_have_picked"]
    assert torch.allclose(gram_w, row_w, rtol=1e-9, atol=1e-9)
    for a, b in zip(gram_guard["ladder"], row_guard["ladder"]):
        assert abs(a["check_sse"] - b["check_sse"]) <= 1e-8 * max(1.0, b["check_sse"]), (a, b)
        assert abs(a["update_scale"] - b["update_scale"]) <= 1e-9


def test_gram_accumulation_matches_resident_on_one_site(tiny):
    """Streamed per-batch Grams at L001 == the resident row-form Gram, to fp64."""

    site = 1
    resident = resident_site_gram(tiny["parent"], tiny["manifest"], site)
    harness = qs.StreamedHarness(tiny["parent"], "cpu", torch.float32, tiny["manifest"])
    grams = harness.capture_group_grams([site])[site]
    G, B, YY, rows = grams.fit_grams()
    assert rows == resident["rows"]
    assert torch.allclose(G, resident["G"], rtol=1e-11, atol=1e-11 * float(resident["G"].abs().max()))
    assert torch.allclose(B, resident["B"], rtol=1e-11, atol=1e-11 * float(resident["B"].abs().max()))
    assert abs(YY - resident["YY"]) <= 1e-11 * resident["YY"]
    # The stash is the resident's captured rows exactly (same bits, bf16/fp32 lossless).
    X = torch.cat([x for split, x, _ in grams.stash if split != "held"]).to(torch.float64)
    Y = torch.cat([y for split, _, y in grams.stash if split != "held"]).to(torch.float64)
    assert torch.equal(X, resident["X"])
    assert torch.equal(Y, resident["Y"])
    # And the inner/check partition is the lane's own mask.
    mask = qf_corpus.fit_check_rows(tiny["manifest"])
    phi = resident["phi"]
    assert torch.allclose(grams.G["check"], phi[mask].T @ phi[mask], rtol=1e-11, atol=1e-9)
    assert torch.allclose(grams.G["inner"], phi[~mask].T @ phi[~mask], rtol=1e-11, atol=1e-9)
    assert grams.rows["check"] == int(mask.sum())
    # The rows form hands qf_mlp.narrow_mlp_site the resident's own blocks.
    rows = grams.captured_rows()
    assert torch.equal(rows["fit"], resident["X"])
    assert torch.equal(rows["local_fit"], resident["Y"])
    assert set(rows) == {"fit", "hld", "local_fit", "local_hld"}
    # And the per-batch accumulation is the one-GEMM Gram to fp64 summation order.
    distance = grams.accumulation_vs_rows()
    assert distance["rows_agree"]
    assert distance["G_max_abs_diff_rel"] < 1e-11, distance
    assert distance["B_max_abs_diff_rel"] < 1e-11, distance
    assert distance["YY_rel_diff"] < 1e-11, distance
    print(f"\nL001 accumulation vs rows: G {distance['G_max_abs_diff_rel']:.3e} B {distance['B_max_abs_diff_rel']:.3e}")
    harness.tap.remove()


def test_tap_index_64_distinct_grams_from_2_buffers(tiny64):
    """64 sites through TWO shared buffers give 64 distinct Grams, each the resident's."""

    harness = qs.StreamedHarness(tiny64["parent"], "cpu", torch.float32, tiny64["manifest"])
    assert sorted(harness.engine.layers.buffers_by_type.keys()) == ["full_attention", "linear_attention"]
    assert len(harness.engine.layers.mlp_modules()) == 2
    sites = list(range(64))
    grams = harness.capture_group_grams(sites)
    assert sorted(grams) == sites
    fit = {i: grams[i].fit_grams()[0] for i in sites}
    distinct = sum(
        1 for a in sites for b in sites if a < b and not torch.equal(fit[a], fit[b])
    )
    assert distinct == 64 * 63 // 2, f"only {distinct} of {64 * 63 // 2} pairs differ"

    resident = qf_harness.Harness(tiny64["parent"], "cpu", torch.float32, tiny64["manifest"])
    layers = resident.layers()
    targets = {f"L{i:03d}": layers[i].mlp for i in sites}
    captured = resident.capture(targets, outputs=targets)
    worst = 0.0
    for i in sites:
        mlp = layers[i].mlp
        gate = mlp.gate_proj.weight.detach().to("cpu", torch.float64)
        up = mlp.up_proj.weight.detach().to("cpu", torch.float64)
        phi = qf_solve.swiglu(captured[f"L{i:03d}"]["fit"], gate, up)
        G_res = phi.T @ phi
        scale = float(G_res.abs().max())
        worst = max(worst, float((fit[i] - G_res).abs().max()) / scale)
        assert torch.allclose(fit[i], G_res, rtol=1e-11, atol=1e-11 * scale), i
    print(f"\n64 sites / 2 buffers: worst Gram relative max-abs difference {worst:.3e}")
    harness.tap.remove()


def test_parity_receipt_deterministic(tiny):
    """Two fresh engines produce the same parity receipt, and it agrees with the resident."""

    reports = []
    for _ in range(2):
        harness = qs.StreamedHarness(tiny["parent"], "cpu", torch.float32, tiny["manifest"])
        reports.append(qs.streamed_parity(harness, resident_check=True))
        harness.tap.remove()
    first, second = reports
    assert first == second, (first, second)
    assert first["deterministic"] is True
    assert first["final_hidden_bit_identical_under_tap"] is True
    assert first["tap_fired_for"] == [0, 1, 2]
    assert first["hook_placement_worst"] < 1e-5, first["hook_placement_relL2"]
    resident = first["resident"]
    assert resident["logits_bit_identical"] is True, resident
    assert resident["all_mlp_inputs_bit_identical"] is True, resident
    # The receipt seals canonically (no NaN, JSON-able).
    qf_arch.canonical_sha256(first)


def test_refusals_and_delegation(tiny):
    """No ``--streamed`` -> the lane's own parser/builder; other axes are refused."""

    base = ["--parent", str(tiny["parent"]), "--out", str(tiny["parent"].parent / "never"),
            "--evidence", str(tiny["evidence"]), "--tag", "X", "--device", "cpu",
            "--dtype", "float32"]
    args = qs.parse_args(base + ["--mlp-keep", "64"])
    assert args.streamed is False and args.resident_check is False
    args = qs.parse_args(base + ["--streamed", "--mlp-keep", "64", "--resident-check"])
    assert args.streamed is True and args.resident_check is True and args.mlp_keep == 64
    assert args.solve_form == "rows" and args.src_dir == ""
    qs.refuse_unsupported(args)
    args = qs.parse_args(base + ["--streamed", "--mlp-keep", "64", "--solve-form", "gram",
                                 "--src-dir=" + str(tiny["parent"])])
    assert args.solve_form == "gram" and args.src_dir == str(tiny["parent"])
    qs.refuse_unsupported(args)
    assert qs.read_dir_of(args) == str(tiny["parent"])

    def refused(extra: list[str]) -> str:
        try:
            qs.refuse_unsupported(qs.parse_args(base + ["--streamed"] + extra))
        except SystemExit as error:
            return str(error)
        return ""

    assert "--mlp-keep" in refused([])
    assert "--excise" in refused(["--mlp-keep", "64", "--excise", "0"])
    assert "--vocab-keep" in refused(["--vocab-keep", "384"])
    assert "--vocab-keep" in refused(["--mlp-keep", "64", "--vocab-keep", "384"])
    assert "reader-metric" not in refused(["--mlp-keep", "64"])
    assert "evidence" in refused(["--mlp-keep", "64", "--mlp-select", "reader-metric"])
    assert "--drop-mtp" in refused(["--mlp-keep", "64", "--carry-mtp"])
    assert "--solve-form" in refused(["--mlp-keep", "64", "--solve-form", "rowz"])
    share = qs.parse_args(["--parent", "/mnt/ai_data/models/x", "--out", "o", "--evidence", "e",
                           "--tag", "t", "--streamed", "--mlp-keep", "64"])
    with pytest.raises(SystemExit, match="share"):
        qs.refuse_unsupported(share)
    # A local --parent whose READ source is the share is the same refusal.
    staged = qs.parse_args(base + ["--streamed", "--mlp-keep", "64", "--src-dir", "/mnt/ai_data/models/x"])
    with pytest.raises(SystemExit, match="share"):
        qs.refuse_unsupported(staged)
    allowed = qs.parse_args(base + ["--streamed", "--mlp-keep", "64", "--src-dir", "/mnt/ai_data/models/x",
                                    "--allow-share-read"])
    qs.refuse_unsupported(allowed)


@pytest.fixture(scope="module")
def both_children(tiny, work: Path):
    """The resident build and the streamed build of the SAME child, side by side."""

    common = [
        "--parent", str(tiny["parent"]), "--evidence", str(tiny["evidence"]),
        "--tag", "C2K64", "--parent-code", "TINY-128", "--device", "cpu",
        "--dtype", "float32", "--mlp-keep", "64", "--mlp-group", "3",
    ]
    resident_out = work / "child-resident"
    streamed_out = work / "child-streamed"
    for out in (resident_out, streamed_out):
        if out.exists():
            shutil.rmtree(out)
    resident = qf_build.build(qf_build.parse_args(common + ["--out", str(resident_out)]))
    streamed = qs.build_streamed(
        qs.parse_args(common + ["--out", str(streamed_out), "--streamed", "--resident-check"])
    )
    return {"resident": resident, "streamed": streamed, "resident_out": resident_out,
            "streamed_out": streamed_out}


def test_streamed_build_reproduces_resident_build(both_children):
    """The run book's test (1) at fixture scale: same child, byte for byte."""

    resident, streamed = both_children["resident"], both_children["streamed"]
    r_axis, s_axis = resident["axes"][0], streamed["axes"][0]
    assert s_axis["schedule"]["groups"] == r_axis["schedule"]["groups"]
    assert s_axis["schedule"]["corpus_passes"] == r_axis["schedule"]["corpus_passes"]
    assert s_axis["config_standard"] and r_axis["config_standard"]
    for r_site, s_site in zip(r_axis["per_site"], s_axis["per_site"]):
        assert s_site["site"] == r_site["site"]
        assert s_site["selected_channels"] == r_site["selected_channels"], r_site["site"]
        assert s_site["solve"]["lambda_relative"] == r_site["solve"]["lambda_relative"], r_site["site"]
        for key in ("fit", "held"):
            assert abs(s_site["scores"][key]["relL2"] - r_site["scores"][key]["relL2"]) < 1e-6
        assert abs(s_site["identity_precondition_relL2"] - r_site["identity_precondition_relL2"]) < 1e-6
    r_digest = resident["export"]["tensor_sha256"]
    s_digest = streamed["export"]["tensor_sha256"]
    assert sorted(r_digest) == sorted(s_digest)
    differing = sorted(name for name in r_digest if r_digest[name] != s_digest[name])
    if differing:
        # Name the difference before failing: fp64 summation order is the only
        # thing that may differ, and it shows as a last-ulp flip in a writer.
        from safetensors import safe_open

        worst = 0.0
        with safe_open(str(both_children["resident_out"] / "model.safetensors"), "pt") as a, \
                safe_open(str(both_children["streamed_out"] / "model.safetensors"), "pt") as b:
            for name in differing:
                worst = max(worst, float((a.get_tensor(name).double() - b.get_tensor(name).double()).abs().max()))
        pytest.fail(f"{len(differing)} tensors differ (worst max-abs {worst:.3e}): {differing[:8]}")
    assert streamed["export"]["child_tensors_sha256"] == resident["export"]["child_tensors_sha256"]
    assert streamed["storage"] == resident["storage"]
    assert streamed["verification"]["parameters_counted_live"] == resident["verification"]["parameters_counted_live"]
    assert streamed["preconditions"]["parity"]["resident"]["logits_bit_identical"] is True
    assert streamed["engine"]["gram_passes"] == s_axis["schedule"]["corpus_passes"] + 2  # + parity's two
    reread = json.loads((both_children["streamed_out"] / qs.RECEIPT_NAME).read_text(encoding="ascii"))
    assert reread["receipt_sha256"] == streamed["receipt_sha256"]
    assert reread["builder"].startswith("qf_stream27.py")


def test_folded_buffers_reproduce_the_exported_child(tiny, both_children):
    """After the fold, the streamed engine's own forward IS the child's forward."""

    harness = qs.StreamedHarness(tiny["parent"], "cpu", torch.float32, tiny["manifest"])
    report = qs.mlp_axis_streamed(harness, 64, group_size=3, log=quiet)
    assert report["config_standard"]
    assert sorted(harness.engine.layers.folded) == list(range(8))
    ids = harness.batches[0][1]
    streamed_logits = harness.engine.logits(ids)
    # Every layer now materialises with the narrow buffer, strict load included.
    for index in range(8):
        layer = harness.engine.layers[index]
        assert int(layer.mlp.intermediate_size) == 64
        assert layer.mlp.gate_proj.weight.shape == (64, 128)
    child = qf_harness.Harness(both_children["streamed_out"], "cpu", torch.float32, tiny["manifest"])
    child_logits = child.logits(ids)
    assert torch.equal(streamed_logits, child_logits)
    harness.tap.remove()


def test_gram_solve_form_matches_rows_form_to_fp64_order(tiny, both_children, work: Path):
    """``--solve-form gram`` picks the same channels and rungs; writers agree to fp64 order."""

    out = work / "child-streamed-gram"
    if out.exists():
        shutil.rmtree(out)
    gram = qs.build_streamed(qs.parse_args([
        "--parent", str(tiny["parent"]), "--evidence", str(tiny["evidence"]),
        "--tag", "C2K64", "--parent-code", "TINY-128", "--device", "cpu",
        "--dtype", "float32", "--mlp-keep", "64", "--mlp-group", "3",
        "--out", str(out), "--streamed", "--solve-form", "gram", "--no-parity",
    ]))
    rows = both_children["streamed"]
    assert gram["axes"][0]["solve_form"] == "gram"
    assert rows["axes"][0]["solve_form"] == "rows"
    for g_site, r_site in zip(gram["axes"][0]["per_site"], rows["axes"][0]["per_site"]):
        assert g_site["selected_channels"] == r_site["selected_channels"], r_site["site"]
        assert g_site["solve"]["lambda_relative"] == r_site["solve"]["lambda_relative"], r_site["site"]
        assert abs(g_site["scores"]["fit"]["relL2"] - r_site["scores"]["fit"]["relL2"]) < 1e-6
    from safetensors import safe_open

    worst = 0.0
    with safe_open(str(both_children["streamed_out"] / "model.safetensors"), "pt") as a, \
            safe_open(str(out / "model.safetensors"), "pt") as b:
        for name in a.keys():
            x, y = a.get_tensor(name).double(), b.get_tensor(name).double()
            worst = max(worst, float((x - y).abs().max() / max(float(x.abs().max()), 1e-30)))
    print(f"\ngram vs rows child: worst relative max-abs tensor difference {worst:.3e}")
    assert worst < 1e-4, worst


def test_src_dir_reads_a_staged_copy(tiny, both_children, work: Path):
    """``--src-dir`` reads the shards from a local copy and yields the same child bytes."""

    staged = work / "parent-staged"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(tiny["parent"], staged)
    out = work / "child-streamed-srcdir"
    if out.exists():
        shutil.rmtree(out)
    built = qs.build_streamed(qs.parse_args([
        "--parent", str(tiny["parent"]), "--src-dir", str(staged),
        "--evidence", str(tiny["evidence"]),
        "--tag", "C2K64", "--parent-code", "TINY-128", "--device", "cpu",
        "--dtype", "float32", "--mlp-keep", "64", "--mlp-group", "3",
        "--out", str(out), "--streamed", "--no-parity",
    ]))
    assert built["preconditions"]["engine"]["weights_read_from"] == str(staged)
    identity = built["preconditions"]["source_identity"]
    assert identity["mode"] == "byte-identical-staged-checkpoint"
    assert identity["weights"] == qs.checkpoint_weight_manifest(tiny["parent"])
    assert built["export"]["child_tensors_sha256"] == both_children["resident"]["export"]["child_tensors_sha256"]
    # A staged copy with a different config is refused before anything streams.
    bad = work / "parent-staged-bad"
    if bad.exists():
        shutil.rmtree(bad)
    shutil.copytree(tiny["parent"], bad)
    body = json.loads((bad / "config.json").read_text(encoding="utf-8"))
    body.get("text_config", body)["rms_norm_eps"] = 1e-5
    (bad / "config.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(SystemExit, match="different config"):
        qs.StreamedHarness(tiny["parent"], "cpu", torch.float32, tiny["manifest"], read_dir=bad)


def test_src_dir_rejects_same_config_different_weights_before_engine(tiny, tmp_path, monkeypatch):
    staged = tmp_path / "substituted"
    shutil.copytree(tiny["parent"], staged)
    from safetensors.torch import load_file, save_file

    weights = load_file(str(staged / "model.safetensors"))
    name = sorted(weights)[0]
    weights[name] = weights[name].clone()
    weights[name].view(-1)[0] += 1
    save_file(weights, str(staged / "model.safetensors"))
    monkeypatch.setattr(qs, "StreamedTrunk", lambda *_a, **_k: pytest.fail("constructed unverified engine"))
    with pytest.raises(SystemExit, match="different checkpoint weights/index"):
        qs.StreamedHarness(tiny["parent"], "cpu", torch.float32, tiny["manifest"], read_dir=staged)


def test_source_recheck_rejects_both_copies_changed_together(tiny, tmp_path):
    parent, staged = tmp_path / "parent", tmp_path / "staged"
    shutil.copytree(tiny["parent"], parent)
    shutil.copytree(parent, staged)
    identity = qs.bind_checkpoint_source(parent, staged)
    for directory in (parent, staged):
        with (directory / "model.safetensors").open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(SystemExit, match="changed after the initial provenance check"):
        qs.recheck_checkpoint_source(identity)


def test_weight_manifest_covers_index_and_all_shards(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}))
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "b.safetensors").write_bytes(b"b")
    first = qs.checkpoint_weight_manifest(tmp_path)
    assert set(first["files"]) == {index.name, "a.safetensors", "b.safetensors"}
    (tmp_path / "b.safetensors").write_bytes(b"c")
    assert qs.checkpoint_weight_manifest(tmp_path)["sha256"] != first["sha256"]
    (tmp_path / "b.safetensors").unlink()
    with pytest.raises(SystemExit, match="missing or escaping"):
        qs.checkpoint_weight_manifest(tmp_path)


def test_weight_manifest_rejects_unbound_single_file_beside_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "a.safetensors"}})
    )
    (tmp_path / "a.safetensors").write_bytes(b"indexed weights")
    (tmp_path / "model.safetensors").write_bytes(b"different HF-preferred weights")
    with pytest.raises(SystemExit, match="ambiguous single-file and indexed"):
        qs.checkpoint_weight_manifest(tmp_path)


def test_weight_manifest_accepts_index_explicitly_naming_the_single_file(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "model.safetensors"}})
    )
    (tmp_path / "model.safetensors").write_bytes(b"same engine source")
    manifest = qs.checkpoint_weight_manifest(tmp_path)
    assert set(manifest["files"]) == {"model.safetensors", "model.safetensors.index.json"}


def test_weight_manifest_rejects_symlink_escape(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside weights")
    try:
        (root / "model.safetensors").symlink_to(outside)
    except OSError:
        pytest.skip("host does not permit creating symlinks")
    with pytest.raises(SystemExit, match="missing or escaping"):
        qs.checkpoint_weight_manifest(root)


@pytest.mark.parametrize("name", ["../escaped.safetensors", "/escaped.safetensors", "..\\escaped.safetensors"])
def test_weight_manifest_rejects_index_path_escape(tmp_path, name):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": name}}))
    with pytest.raises(SystemExit, match="unsafe checkpoint file path"):
        qs.checkpoint_weight_manifest(tmp_path)
