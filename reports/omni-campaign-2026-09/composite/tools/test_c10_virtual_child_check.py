"""Tests for ``c10_virtual_child_check.py`` -- CPU, fp32, the synthetic Chimera + form-E K3 of
``test_c10_union_install.py`` (with RoPE on, gqa_rope_dim 8).

  test_tampered_partial_is_refused   (runs first; the next test's resume consumes the partial)
      a partial whose bytes no longer hash like its record, or whose Qwen rows moved, is refused.
  test_virtual_child_equals_the_written_child
      c10 --stop-after-layer <last> -> partial; the check passes on it; then c10 --resume writes the
      child; the virtual child's logits (layer-major streamed engine) equal ChimeraOmniModel's logits
      on the written child (load_chimera_checkpoint) bit for bit, and the virtual accounting equals
      oc_seal on the written child.

  test_sweep_variants_and_reference_in_one_pass   (1.2.0, 2026-10-11: sweep variants + --reference + --evidence)

Server:
    cd $OC/composite/tools && CHIMERA_SRC=$OC/import/rq27b/tree_ef1ffb9/src CUDA_VISIBLE_DEVICES="" \\
        /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" -p no:cacheprovider \\
        test_c10_virtual_child_check.py -q
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

import c10_union_install as c10  # noqa: E402
import c10_virtual_child_check as check  # noqa: E402
import test_c10_union_install as base  # noqa: E402  (fixture builders only; its tests are not collected here)
from safetensors.torch import load_file, save_file  # noqa: E402
from chimera_omni.config import ChimeraConfig  # noqa: E402
from chimera_omni.importing.safetensors_io import load_chimera_checkpoint, read_checkpoint_config, read_safetensors  # noqa: E402

K3_CHANNELS = 10
PROMPTS = [[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], [100, 200, 300, 400, 500], [1500, 7, 7, 7, 1999, 64, 65]]


@pytest.fixture(scope="module")
def stopped(tmp_path_factory) -> dict:
    work = tmp_path_factory.mktemp("c10vc")
    trunks = base.build_trunks(work / "trunks", rope_dim=8)
    k3_dir = work / "k3"
    base.make_k3_dense(k3_dir)
    charts = work / "charts"
    base.make_charts(charts, bias_scale=0.05)
    out = work / "child"
    base.run(trunks, k3_dir, charts, out, K3_CHANNELS, ["--stop-after-layer", str(base.LAYERS - 1), "--chart-bias-ceiling", "0"],
             expect=c10.EXIT_STOPPED)
    prompts = work / "prompt_ids.json"
    prompts.write_text(json.dumps(PROMPTS), encoding="utf-8")
    return {"work": work, "trunks": trunks, "k3": k3_dir, "charts": charts, "out": out,
            "partial": Path(f"{out}.partial"), "prompts": prompts}


def check_argv(stopped: dict, partial: Path, out: Path) -> list[str]:
    return ["--trunk", str(stopped["trunks"]["trunk"]), "--partial", str(partial), "--k3-channels", str(K3_CHANNELS),
            "--parent-parameter-count", str(10 ** 9), "--media-units-parameters", "1000",
            "--prompt-ids", str(stopped["prompts"]), "--device", "cpu", "--out", str(out)]


# runs FIRST: the next test's --resume consumes the partial
def test_tampered_partial_is_refused(stopped: dict) -> None:
    work = stopped["work"]
    # bytes that no longer hash like the record
    bad = work / "bad_bytes.partial"
    shutil.copytree(stopped["partial"], bad)
    got = load_file(str(bad / "L01.safetensors"))
    got["down"] = got["down"].clone()
    got["down"][0, 0] += 1.0
    save_file(got, str(bad / "L01.safetensors"))
    with pytest.raises(SystemExit) as refusal:
        check.main(check_argv(stopped, bad, work / "bad_bytes.json"))
    assert "sha256" in str(refusal.value)
    # Qwen rows moved, record re-sealed to match (only the verbatim check can see it)
    moved = work / "moved_rows.partial"
    shutil.copytree(stopped["partial"], moved)
    got = load_file(str(moved / "L02.safetensors"))
    up = got["up"].clone()
    up[0] = up[0] * 1.5
    save_file({"up": up.contiguous(), "down": got["down"]}, str(moved / "L02.safetensors"))
    record = json.loads((moved / "L02.json").read_text(encoding="utf-8"))
    record["installed"]["up_sha256"] = c10.sha256_tensor_bytes(up)
    record["partial"]["tensors_sha256"] = c10.sha256_file(str(moved / "L02.safetensors"))
    (moved / "L02.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit) as refusal:
        check.main(check_argv(stopped, moved, work / "moved_rows.json"))
    assert "verbatim" in str(refusal.value)


def test_virtual_child_equals_the_written_child(stopped: dict) -> None:
    receipt_path = stopped["work"] / "virtual_check.json"
    assert check.main(check_argv(stopped, stopped["partial"], receipt_path)) == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["structural_load"]["strict_load_equivalent"] is True
    assert receipt["accounting"]["equals_c10_arithmetic"] is True
    assert all(r["qwen_gate_rows_verbatim"] and r["qwen_up_rows_verbatim"] for r in receipt["identity_rows"])
    assert receipt["comparison"]["all_logits_finite"] is True
    assert receipt["verdict"].startswith("loads")

    # the virtual child's logits, straight from the engine
    trunk_dir = Path(stopped["trunks"]["trunk"])
    doc = read_checkpoint_config(trunk_dir)
    trunk_cfg = ChimeraConfig.from_manifest(doc["config"])
    views = read_safetensors(trunk_dir, verify=True)
    layers, _rows, width = check.check_partial(stopped["partial"], views, base.LAYERS, base.TRUNK_WIDTH, base.HIDDEN,
                                               torch.float32)
    child_cfg = check.child_config(trunk_cfg, width, K3_CHANNELS, base.TRUNK_WIDTH)
    _lt, virtual, _facts = check.streamed_logits(trunk_cfg, child_cfg, views, layers, {}, PROMPTS,
                                                 torch.device("cpu"), torch.float32)

    # now let c10 write the child from the same partial and load it the normal way
    argv = ["--trunk", str(stopped["trunks"]["trunk"]), "--parent-trunk", str(stopped["trunks"]["parent"]),
            "--k3-dense", str(stopped["k3"]), "--charts", str(stopped["charts"]),
            "--evidence", str(stopped["trunks"]["evidence"]), "--k3-channels", str(K3_CHANNELS), "--device", "cpu",
            "--group-size", "2", "--chunk", "64", "--gram-block", "16", "--out", str(stopped["out"]),
            "--k3-config-from-flags", "--chart-bias-ceiling", "0", "--resume"]
    assert c10.main(argv) == 0
    written_total, _ = c10.header_total(stopped["out"])
    assert written_total == receipt["accounting"]["child_header_parameters"]
    model, cfg, _ = load_chimera_checkpoint(stopped["out"], device="cpu", dtype=torch.float32)
    model.eval()
    band = int(cfg.ledger_band_begin())
    with torch.no_grad():
        for ids, mine in zip(PROMPTS, virtual):
            ref = model(torch.tensor([ids]), ledger_band_logit_mask=True)["logits"][0].float()
            assert torch.equal(ref[:, :band], mine[:, :band]), float((ref[:, :band] - mine[:, :band]).abs().max())


def test_prefix_partial_needs_the_flag_and_is_labelled_hybrid(tmp_path) -> None:
    """A prefix partial (c10 --stop-after-layer 1) is refused unless --pad-missing-with-trunk; with it the
    receipt says HYBRID, and a trunk-padded layer computes the trunk's own FFN."""
    trunks = base.build_trunks(tmp_path / "trunks", rope_dim=8)
    k3_dir = tmp_path / "k3"
    base.make_k3_dense(k3_dir)
    charts = tmp_path / "charts"
    base.make_charts(charts, bias_scale=0.05)
    out = tmp_path / "child"
    base.run(trunks, k3_dir, charts, out, K3_CHANNELS, ["--stop-after-layer", "1", "--chart-bias-ceiling", "0"],
             expect=c10.EXIT_STOPPED)
    prompts = tmp_path / "ids.json"
    prompts.write_text(json.dumps(PROMPTS), encoding="utf-8")
    # an overflow symlink for a layer not yet written (dangling) must not count as a persisted layer
    import os
    os.symlink(str(tmp_path / "overflow" / "L03.safetensors"), f"{out}.partial/L03.safetensors")
    argv = ["--trunk", str(trunks["trunk"]), "--partial", f"{out}.partial", "--k3-channels", str(K3_CHANNELS),
            "--prompt-ids", str(prompts), "--device", "cpu", "--out", str(tmp_path / "r.json")]
    with pytest.raises(SystemExit) as refusal:
        check.main(argv)
    assert "pad-missing-with-trunk" in str(refusal.value)
    assert check.main(argv + ["--pad-missing-with-trunk"]) == 0
    receipt = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert receipt["union_layers"] == [0, 1] and receipt["trunk_padded_layers"] == [2, 3]
    assert receipt["child_kind"].startswith("HYBRID") and receipt["verdict"].startswith("HYBRID")
    assert receipt["accounting"]["equals_c10_arithmetic"] is True
    # a trunk-padded FFN is the trunk's FFN
    views = read_safetensors(Path(trunks["trunk"]), verify=True)
    cfg = ChimeraConfig.from_manifest(read_checkpoint_config(Path(trunks["trunk"]))["config"])
    width = base.TRUNK_WIDTH + K3_CHANNELS
    child_cfg = check.child_config(cfg, width, K3_CHANNELS, base.TRUNK_WIDTH)
    padded = check.trunk_padded_layer(views, 3, base.TRUNK_WIDTH, base.HIDDEN, width, torch.float32)
    x = torch.randn(1, 9, base.HIDDEN)
    valid = torch.ones(1, 9, dtype=torch.bool)
    with torch.no_grad():
        a = check.run_block(check.build_block(cfg, 3, views, torch.device("cpu"), torch.float32, None), x, valid)
        b = check.run_block(check.build_block(child_cfg, 3, views, torch.device("cpu"), torch.float32, padded), x, valid)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_sweep_variants_and_reference_in_one_pass(tmp_path) -> None:
    """1.2.0: c10 --sweep-targets variants (down-only partials + a shared up) load as virtual children;
    the one-pass evaluation equals the single-child streamed path per variant (same top-1 counts, same
    KL), trunk/additive:0 IS the trunk, and the --reference parent's held NLL equals a direct forward of
    the parent checkpoint.  A shared up edited after the sweep is refused."""
    trunks = base.build_trunks(tmp_path / "trunks", rope_dim=8)
    k3_dir = tmp_path / "k3"
    base.make_k3_dense(k3_dir)
    charts = tmp_path / "charts"
    base.make_charts(charts, bias_scale=0.05)
    sweep = tmp_path / "sweep"
    names = ("legacy", "a0", "a05")
    base.run_sweep(trunks, {"dir": k3_dir, "charts": charts}, sweep,
                   "legacy=parent/additive:1;a0=trunk/additive:0;a05=trunk/additive:0.5")
    prompts = tmp_path / "ids.json"
    prompts.write_text(json.dumps(PROMPTS), encoding="utf-8")
    out = tmp_path / "multi.json"
    argv = ["--trunk", str(trunks["trunk"]), "--k3-channels", str(K3_CHANNELS),
            "--parent-parameter-count", str(10 ** 9), "--reference", f"parent={trunks['parent']}",
            "--evidence", str(trunks["evidence"]), "--prompt-ids", str(prompts), "--device", "cpu",
            "--position-chunk", "16", "--out", str(out)]
    for name in names:
        argv += ["--variant", f"{name}={sweep / 'variants' / name}"]
    assert check.main(argv) == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["tool_version"] == check.TOOL_VERSION == "1.2.0"
    for name in names:
        v = receipt["variants"][name]
        assert v["structural_load"]["strict_load_equivalent"] is True
        assert v["accounting"]["equals_c10_arithmetic"] is True
        assert all(r["shared_up"] and r["qwen_gate_rows_verbatim"] and r["qwen_up_rows_verbatim"]
                   for r in v["identity_rows"])
        assert v["open_loop"] is True
    ev = receipt["eval"]["models"]
    assert set(ev) == {"trunk", "parent", *names}
    held, _rec = check.held_sequences(str(trunks["evidence"]), None)
    assert receipt["eval"]["sequences"]["held"] == len(held) > 0
    # trunk/additive:0 is the trunk
    assert ev["a0"]["prompts"]["top1_agreement_vs_trunk"] == 1.0 and ev["a0"]["held"]["top1_agreement_vs_trunk"] == 1.0
    assert max(r["held_residual_relL2_vs_trunk"] for r in ev["a0"]["per_layer"]) < 1e-4
    assert abs(ev["a0"]["held"]["nll_mean"] - ev["trunk"]["held"]["nll_mean"]) < 1e-4
    # per variant == the single-child streamed path
    trunk_dir = Path(trunks["trunk"])
    trunk_cfg = ChimeraConfig.from_manifest(read_checkpoint_config(trunk_dir)["config"])
    views = read_safetensors(trunk_dir, verify=True)
    band = int(trunk_cfg.ledger_band_begin())
    for name in names:
        layers, _rows, width = check.check_partial(sweep / "variants" / name, views, base.LAYERS, base.TRUNK_WIDTH,
                                                   base.HIDDEN, torch.float32)
        child_cfg = check.child_config(trunk_cfg, width, K3_CHANNELS, base.TRUNK_WIDTH)
        lt, lc, facts = check.streamed_logits(trunk_cfg, child_cfg, views, layers, {}, PROMPTS, torch.device("cpu"),
                                              torch.float32)
        cmp = check.compare(lt, lc, band)
        assert cmp["top1_agree"] == ev[name]["prompts"]["top1_agree_vs_trunk"]
        assert abs(cmp["kl_trunk_child_mean"] - ev[name]["prompts"]["kl_trunk_child_mean"]) < 1e-5
    # the reference is the parent checkpoint
    model, cfg, _ = load_chimera_checkpoint(trunks["parent"], device="cpu", dtype=torch.float32)
    model.eval()
    nll, count = 0.0, 0
    with torch.no_grad():
        for ids in held:
            logits = model(torch.tensor([ids]), ledger_band_logit_mask=True)["logits"][0].float()[:, :band]
            lp = torch.log_softmax(logits, -1)
            nxt = torch.tensor(ids[1:])
            keep = nxt < band
            picked = lp[:-1].gather(1, nxt.clamp_max(band - 1).unsqueeze(1)).squeeze(1)
            nll += float(-picked[keep].sum())
            count += int(keep.sum())
    assert abs(ev["parent"]["held"]["nll_mean"] - nll / count) < 1e-4
    assert ev["parent"]["held"]["top1_agreement_vs_reference"] == 1.0
    # a shared up edited after the sweep is refused (the cache keys on size + mtime as well)
    up_path = sweep / "shared" / "L01.up.safetensors"
    got = load_file(str(up_path))
    up = got["up"].clone()
    up[-1, 0] += 1.0
    save_file({"up": up.contiguous()}, str(up_path))
    with pytest.raises(SystemExit) as refusal:
        check.main(argv)
    assert "sha256" in str(refusal.value)
