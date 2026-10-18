"""Tests for d2_register_check.py (OMNI1D2's checks between the register DRY RUN and --confirm).  CPU only; server.

    OC=/home/cereal/omni-campaign; EF=$OC/import/rq27b/tree_ef1ffb9
    CUDA_VISIBLE_DEVICES= CHIMERA_OMNI_SRC=$EF OC_LEDGER_DIR=$OC/ledger python -m pytest -q test_d2_register_check.py

A tiny Chimera checkpoint written by the REAL importer (test_p1_qwen35_reader's fixture: 3 GDN + 1 gated-GQA layer,
FFN 24 = 8 trunk + 16 K3), adapters from the trainer's own AdapterBank at nonzero B and jittered gains, a recipe built
from the expected pieces, and the REAL ledger (register dry run -> --confirm -> delta-compose) and P1 reader.  The
oracle is the trainer's forward: `site_forward_hook` on the base weights must equal the composed weights' plain matmul
(test 1), and the reader's composition of a recipe the site check passes must equal that oracle (test 2), so "site
check OK" means "the registered model is the trained one".  The NEGATIVE CONTROL is a recipe with the gate / up row
halves of `blocks.0.ffn.up.weight` swapped: the ledger registers it (dry run rc 0, the log check passes too), its
composed weights are not the trained ones, and only the site check refuses it.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_SRC = os.environ.get("CHIMERA_OMNI_SRC", "")
if _SRC:
    sys.path.insert(0, str(Path(_SRC) / "src"))
LEDGER_DIR = Path(os.environ.get("OC_LEDGER_DIR", "") or HERE.parents[1] / "ledger")
sys.path.insert(0, str(LEDGER_DIR))

torch = pytest.importorskip("torch")
F = pytest.importorskip("test_p1_qwen35_reader")  # needs chimera_omni (CHIMERA_OMNI_SRC)
from safetensors.torch import save_file  # noqa: E402

import d2_register_check as R  # noqa: E402
import oc_checkpoint  # noqa: E402
import streamed_kd_trainer as T  # noqa: E402

D2_ID = "OC02-D-Q38-27BU-TINYD2"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qwen_effect(spec, bound, w):
    """The ORACLE: the dense qwen3_5 weight whose plain matmul is the trainer's hooked Linear (site_forward_hook)."""

    w = w.float().clone()
    b = {k: v.detach().float() for k, v in bound.items()}
    if spec.kind == "full":
        w += spec.scaling * (b["b"] @ b["a"])
    elif spec.kind == "ffn_in":
        w[: spec.trunk] += spec.scaling * (b["b"] @ b["a"])
        if "gain" in b:
            w[spec.trunk:] *= b["gain"].view(-1, 1)
    else:
        w[:, : spec.trunk] += spec.scaling * (b["b"] @ b["a"])
        if "k3_a" in b:
            w[:, spec.trunk:] += spec.k3_scaling * (b["k3_b"] @ b["k3_a"])
    return w


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("d2reg")
    cfg = F.chimera_cfg()
    model, _ = F.import_qwen35_state(F.qwen_state(), cfg, dtype=torch.bfloat16)
    F.write_chimera(root / "omni1", {n: p.detach().clone() for n, p in model.named_parameters()}, cfg)
    parent = root / "parent"
    parent.mkdir()
    (parent / "config.json").write_text(json.dumps(F.hf_config()), encoding="utf-8")
    argv = ["/x/streamed_kd_trainer.py", "train", "--engine", "streamed", "--p1-chimera", str(root / "omni1"),
            "--p1-parent-config", str(parent), "--data-config", "/x/d.json", "--seq-len", "8", "--out", "/x/o", "--safety-dir", "/x/s",
            "--lora-rank", "4", "--lora-alpha", "8", "--k3-lora-rank", "2", "--k3-lora-alpha", "2", "--trunk-ffn-width", str(F.TRUNK)]
    ns = T.parser().parse_args(argv[1:])
    source = T.P1Source(root / "omni1", parent, None)
    config = T.text_config_from_source(source, torch.bfloat16, "sdpa")
    plan = T.build_site_plan(config, T.site_plan_from_args(ns))
    layer_types = [str(t) for t in config.layer_types[: int(config.num_hidden_layers)]]
    bank = T.AdapterBank(plan, layer_types, "cpu", 7, b_init_std=0.05, gain_jitter=0.2)
    run = root / "run-long"
    run.mkdir()
    T.save_adapters(run / "adapters.safetensors", bank, {"step": 1})
    (run / "training-receipt.json").write_text(json.dumps({"argv": argv}), encoding="utf-8")
    trained = root / "d2.trained"
    trained.write_text(json.dumps({"long": {"adapters": str(run / "adapters.safetensors"), "sha256": sha(run / "adapters.safetensors"),
                                            "receipt_sha256": sha(run / "training-receipt.json")}}), encoding="utf-8")
    # the ledger: a parent, its pre-D clone (the D2 base stand-in)
    registry, ledger = root / "registry", root / "ledger.jsonl"

    def oc(*args: str) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = oc_checkpoint.main([*args, "--registry", str(registry), "--ledger", str(ledger)])
        return rc, buf.getvalue()

    assert oc("register", "--checkpoint-id", "PARENT-TINYOMNI", "--stage", "parent", "--directory", str(root / "omni1"),
              "--parent-directory", str(root / "omni1"), "--parent-provenance-sha256", "a" * 64, "--recipe-command", "tiny",
              "--confirm")[0] == 0
    assert oc("clone", "--source", "PARENT-TINYOMNI", "--stage-about-to-run", "D", "--allow-unreviewed-source", "--confirm")[0] == 0
    base_id = "PARENT-TINYOMNI.clone-pre-D1"
    base = oc_checkpoint.current(registry, base_id)
    (root / "base_record.json").write_text(json.dumps(base), encoding="utf-8")
    return SimpleNamespace(root=root, parent=parent, source=source, plan=plan, layer_types=layer_types, bank=bank, trained=trained,
                           adapters=T.load_adapters(run / "adapters.safetensors"), oc=oc, base_id=base_id,
                           base=base["checkpoint"], base_record=root / "base_record.json")


def write_recipe(fx, name: str, mutate=None) -> Path:
    """A recipe from the expected pieces (one entry per piece, fp32 tensors) -- what d2_delta_recipe.py must write."""

    pieces = R.expected_pieces(fx.source.reader, fx.plan, fx.layer_types, fx.adapters)
    tensors, entries = {}, []
    for k, p in enumerate(pieces):
        region = [list(r) for r in p["region"]]
        if p["op"] == "lora":
            tensors[f"d2.{k}.a"], tensors[f"d2.{k}.b"] = p["a"].contiguous().clone(), p["b"].contiguous().clone()
            entries.append({"op": "lora", "target": p["target"], "a": f"d2.{k}.a", "b": f"d2.{k}.b", "alpha": p["alpha"], "slice": region})
        else:
            tensors[f"d2.{k}.gain"] = p["tensor"].contiguous().clone()
            entries.append({"op": "scale", "target": p["target"], "tensor": f"d2.{k}.gain", "dim": 0, "slice": region})
    if mutate:
        mutate(entries, tensors)
    d = fx.root / name
    d.mkdir()
    save_file(tensors, str(d / "delta.safetensors"))
    (d / "DELTA.json").write_text(json.dumps({"schema": oc_checkpoint.DELTA_RECIPE_SCHEMA, "base_checkpoint_id": fx.base_id,
                                              "base_directory_sha256": fx.base["directory_sha256"], "entries": entries}), encoding="utf-8")
    return d


def register_args(fx, checkpoint_id: str, delta: Path) -> list[str]:
    """The queue's ARGS (d2_launch.sh register_d2), on the tiny world."""

    return ["register", "--checkpoint-id", checkpoint_id, "--stage", "distill", "--wave", "oc02", "--tag", "TINYD2",
            "--parent-code", "Q38-27BU", "--directory", str(delta), "--delta-base", fx.base_id, "--delta-base-sha256",
            fx.base["directory_sha256"], "--parent-directory", str(fx.root / "omni1"), "--parameter-count", "auto",
            "--parent-parameter-count", "auto", "--recipe", str(delta / "DELTA.json"), "--recipe-command", "tiny-d2", "--notes", "test"]


def swap_gate_up_layer0(entries, tensors):
    """Same shapes, wrong halves: layer 0's gate LoRA lands on up_proj's trunk rows and vice versa."""

    up = [e for e in entries if e["op"] == "lora" and e["target"] == "blocks.0.ffn.up.weight"]
    assert len(up) == 2
    up[0]["slice"], up[1]["slice"] = up[1]["slice"], up[0]["slice"]


def test_the_oracle_is_the_trainers_forward(fx):
    reader = fx.source.reader
    gen = torch.Generator().manual_seed(3)
    checked = 0
    for layer, layer_type in enumerate(fx.layer_types):
        for spec in fx.plan[layer_type]:
            w = reader.get(f"{R.QWEN_LAYER_PREFIX}{layer}.{spec.site}.weight").float()
            module = torch.nn.Linear(spec.in_features, spec.out_features, bias=False)
            module.weight.data = w.clone()
            module._kd_spec, module._kd_bound = spec, fx.bank.layer_sites[layer][spec.site]
            x = torch.randn(5, spec.in_features, generator=gen)
            with torch.no_grad():
                hooked = T.site_forward_hook(module, (x,), module(x))
                plain = x @ qwen_effect(spec, module._kd_bound, w).T
            assert torch.allclose(hooked, plain, rtol=1e-5, atol=1e-6), (layer, spec.site)
            checked += 1
    assert checked == 4 * 3 + 3 * 2 + 1 * 2  # every layer's FFN + 2 GDN or 2 GQA sites


def test_a_correct_recipe_passes_both_checks_and_composes_to_the_trained_model(fx):
    delta = write_recipe(fx, "delta-ok")
    rc, out = fx.oc(*register_args(fx, D2_ID, delta))
    assert rc == 0 and "DRY RUN" in out
    log = fx.root / "dryrun-ok.log"
    log.write_text(out, encoding="utf-8")
    base_params = int(fx.base["parameter_count"])
    added = R.delta_tensor_elements(delta)
    assert added > 0 and re.search(rf"parameters\s+{base_params + added:,}\s", out)  # composed = base + adapters, NOT base
    assert R.main(["log", "--log", str(log), "--checkpoint-id", D2_ID, "--base-id", fx.base_id, "--base-sha256",
                   fx.base["directory_sha256"], "--base-record", str(fx.base_record), "--delta-dir", str(delta)]) == 0
    assert R.main(["sites", "--delta-dir", str(delta), "--trained", str(fx.trained), "--chimera", str(fx.root / "omni1")]) == 0
    # then the ledger registers it, and the reader's composition IS the trained model
    assert fx.oc(*register_args(fx, D2_ID, delta), "--confirm")[0] == 0
    plan_path = fx.root / "plan-ok.json"
    assert fx.oc("delta-compose", "--checkpoint-id", D2_ID, "--deep", "--plan-out", str(plan_path))[0] == 0
    p1 = T.load_peer_module("p1_qwen35_reader")
    composed = p1.Qwen35LayoutReader(fx.base["directory"], plan=json.loads(plan_path.read_text(encoding="utf-8")))
    for layer, layer_type in enumerate(fx.layer_types):
        for spec in fx.plan[layer_type]:
            name = f"{R.QWEN_LAYER_PREFIX}{layer}.{spec.site}.weight"
            want = qwen_effect(spec, fx.bank.layer_sites[layer][spec.site], fx.source.reader.get(name)).to(torch.bfloat16).float()
            assert torch.allclose(composed.get(name).float(), want, rtol=2 ** -7, atol=1e-6), name  # one bf16 rounding apart
    composed.close()


def test_negative_control_a_swapped_half_registers_in_the_ledger_and_only_the_site_check_refuses_it(fx, capsys):
    delta = write_recipe(fx, "delta-swapped", swap_gate_up_layer0)
    rc, out = fx.oc(*register_args(fx, D2_ID + "X", delta))
    assert rc == 0, out  # the ledger cannot see it: same shapes, real targets, no orphans
    log = fx.root / "dryrun-swapped.log"
    log.write_text(out, encoding="utf-8")
    assert R.main(["log", "--log", str(log), "--checkpoint-id", D2_ID + "X", "--base-id", fx.base_id, "--base-sha256",
                   fx.base["directory_sha256"], "--base-record", str(fx.base_record), "--delta-dir", str(delta)]) == 0
    capsys.readouterr()
    assert R.main(["sites", "--delta-dir", str(delta), "--trained", str(fx.trained), "--chimera", str(fx.root / "omni1")]) == 3
    assert "is not the trained" in capsys.readouterr().out
    # and the registered model would not be the trained one
    assert fx.oc(*register_args(fx, D2_ID + "X", delta), "--confirm")[0] == 0
    plan_path = fx.root / "plan-swapped.json"
    assert fx.oc("delta-compose", "--checkpoint-id", D2_ID + "X", "--deep", "--plan-out", str(plan_path))[0] == 0
    p1 = T.load_peer_module("p1_qwen35_reader")
    composed = p1.Qwen35LayoutReader(fx.base["directory"], plan=json.loads(plan_path.read_text(encoding="utf-8")))
    spec = next(s for s in fx.plan[fx.layer_types[0]] if s.site == "mlp.gate_proj")
    name = f"{R.QWEN_LAYER_PREFIX}0.mlp.gate_proj.weight"
    want = qwen_effect(spec, fx.bank.layer_sites[0][spec.site], fx.source.reader.get(name)).to(torch.bfloat16).float()
    assert not torch.allclose(composed.get(name).float(), want, rtol=2 ** -7, atol=1e-6)
    composed.close()


def test_site_check_refusals(fx):
    reader = fx.source.reader
    pieces = R.expected_pieces(reader, fx.plan, fx.layer_types, fx.adapters)
    shapes = {n: list(reader.shards.spec(n)["shape"]) for n in reader.shards.names()}

    def recipe_of(edit=None):
        tensors, entries = {}, []
        for k, p in enumerate(pieces):
            region = [list(r) for r in p["region"]]
            if p["op"] == "lora":
                tensors[f"{k}a"], tensors[f"{k}b"] = p["a"].clone(), p["b"].clone()
                entries.append({"op": "lora", "target": p["target"], "a": f"{k}a", "b": f"{k}b", "alpha": p["alpha"], "slice": region})
            else:
                tensors[f"{k}g"] = p["tensor"].clone()
                entries.append({"op": "scale", "target": p["target"], "tensor": f"{k}g", "dim": 0, "slice": region})
        if edit:
            edit(entries, tensors)
        return R.sites_check(pieces, {"entries": entries}, tensors, shapes)

    problems, counts = recipe_of()
    assert problems == [] and counts["missing"] == 0 and counts["scale"] == 4 and counts["layers"] == 4
    assert reader.specs[f"{R.QWEN_LAYER_PREFIX}0.linear_attn.in_proj_qkv.weight"]["kind"] == "cat"
    qkv0 = [p for p in pieces if p["layer"] == 0 and p["site"] == "linear_attn.in_proj_qkv"]
    assert [p["target"].split(".")[-2] for p in qkv0] == ["q_proj", "k_proj", "v_proj"]  # B split by the map's row parts
    assert sum(int(p["b"].shape[0]) for p in qkv0) == int(fx.adapters["layers.0.linear_attn.in_proj_qkv.b"].shape[0])

    def rounded(entries, tensors):
        tensors["0b"] = tensors["0b"].to(torch.bfloat16)
    assert any("stored as" in p for p in recipe_of(rounded)[0])

    def dropped_gain(entries, tensors):
        entries[:] = [e for e in entries if e["op"] != "scale" or e["target"] != "blocks.1.ffn.up.weight"]
    assert any(p.startswith("no entry for layer 1 mlp.up_proj gain") for p in recipe_of(dropped_gain)[0])

    def scaling(entries, tensors):
        next(e for e in entries if e["op"] == "lora")["alpha"] = 4.0
    assert any("not the trained scaling" in p for p in recipe_of(scaling)[0])

    def doubled(entries, tensors):
        entries.append(dict(entries[0]))
    assert any("repeats entry 0" in p for p in recipe_of(doubled)[0])

    def shifted(entries, tensors):  # a layer-3 entry written onto block 2 (the "blocks.t" mistake), same shapes
        e = next(e for e in entries if e["target"] == "blocks.3.ffn.down.weight")
        e["target"] = "blocks.2.ffn.down.weight"
    problems = recipe_of(shifted)[0]
    assert any("repeats" in p or "is not the trained" in p for p in problems) and any("no entry for layer 3" in p for p in problems)


def test_log_check_reads_the_dry_run_strictly():
    recipe = {"base_checkpoint_id": "B", "base_directory_sha256": "c" * 64,
              "entries": [{"op": "lora"}, {"op": "lora"}, {"op": "scale"}]}
    log = "\n".join(["=== REGISTER D2 ===", "  stage              distill", f"  directory_sha256   {'d' * 64}",
                     "  parameters         1,100  [derived-from-safetensors-headers]",
                     f"  delta base         B pinned at {'c' * 16} (matches its seal)", "  delta chain        R -> B -> D2",
                     "  delta ops          {'replace': 0, 'lora': 2, 'scale': 1} on 2 target tensor(s); replaced 0, added 100",
                     "  status             candidate -- x", "", "DRY RUN: nothing was written.  Re-run with --confirm."])
    kw = dict(checkpoint_id="D2", base_id="B", base_sha256="c" * 64, recipe=recipe)
    assert R.log_check(log, base_parameters=1000, added=100, **kw) == []
    assert R.log_check(log, base_parameters=1100, added=0, **kw)  # the "composed = base" rule refuses this correct plan
    assert R.log_check(log.replace("DRY RUN: nothing was written.", "written"), base_parameters=1000, added=100, **kw)
    assert R.log_check(log, base_parameters=1000, added=100, **{**kw, "recipe": {**recipe, "entries": recipe["entries"] + [{"op": "replace"}]}})
    assert R.log_check(log, base_parameters=1000, added=100, **{**kw, "base_sha256": "e" * 64})
