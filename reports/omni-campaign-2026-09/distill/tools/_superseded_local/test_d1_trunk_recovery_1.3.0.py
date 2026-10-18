"""Tests for ``d1_trunk_recovery.py`` -- CPU, a synthetic 2-layer Chimera (kda, gated_gqa with RoPE) parent of
FFN width 48 and a c10-style union child (trunk 24 rows of the parent + 10 K3 channels + 2 zero-pad rows).

  test_analytic_gradient_equals_autograd          d1 loss + gradient (residual and delta sites) vs autograd, fp64
  test_host_reduced_shards_equal_the_full_batch   uneven shards summed on the host == the full-batch gradient;
                                                  a 2-replica host-staged trainer == the 1-replica trainer; the
                                                  threaded step with lazy (callable) micro-batches == sequential
  test_threaded_shards_run_concurrently           1.2.4: with parallel on, two devices' shards overlap in time
                                                  (a sleeping stand-in for the loss: wall < 1.5x one device)
  test_two_layer_end_to_end                       both layers trained: D1 loss decreases on fit and held tokens,
                                                  decomposition check bit-exact, K3 / pad / attention untouched,
                                                  base + delta (apply_delta) re-run from scratch == the tool's
                                                  final child stream bit for bit, head KL finite, resume reloads
                                                  the rungs and reproduces the delta bytes
  test_right_padded_rows                          `lengths`: padding invalid everywhere, re-run equality kept
  test_charted_child_maps_text_layers_and_checks_identity
                                                  OMNI1 layout (charted media block between text layers): identity
                                                  verified + skipped, text->block map, delta on child block indices,
                                                  same training as the plain child (skips on a tree without charted)
  test_lr_zero_reproduces_the_base                lr 0: stored tensors == the base slices, stream == original child
  test_single_layer_and_stop_after                --layers 1 --stop-after-layer 1 trains one layer only
  test_two_replica_run_matches_one_replica        --devices cpu,cpu (host-reduced DP) vs cpu
  test_block_build_matches_the_fold_engine        build_block/run_block == c10_virtual_child_check's on the child
  test_refusals                                   --out overlaps, foreign --out, ids outside the text rows, no
                                                  union metadata, held rows >= rows
  test_delta_out_guard                            1.2.4: --delta-out inside a model dir / around one / under
                                                  /mnt/ai_data / foreign non-empty / a file is refused before
                                                  anything is written; a fresh one gets a marker and is reusable
  test_dp_bench_cpu_smoke                         d1_dp_bench: 1.2.3 path vs current path on cpu,cpu, parity
  test_deterministic_safetensors_writer           rung/delta bytes depend only on content (order-insensitive)
  test_ids_loaders_and_lr_schedule                .u32 + --seq-len, FILE::KEY, warmup/cosine endpoints
  test_prefix_import_reproduces_the_full_run      1.3.0: L00 from the canonical dir (d1_canonical_rungs sync) + L01
                                                  trained == both trained here: L01 rung, delta bytes, stream; pin /
                                                  layer-count / flag / tampered-rung refusals
  test_boundary_resume_is_bit_identical           1.3.0: the L00 boundary == the recomputed layer-1 inputs (P, C, B)
                                                  bit for bit; resume after a lost L01 runs no layer-0 forward and
                                                  ends on the uninterrupted delta bytes; corrupt / foreign cache ignored
  test_boundary_due_rule                          1.3.0: when a boundary is written (end of prefix, every K new layers,
                                                  never after the last layer)
  test_memory_log_and_rungs_not_in_ram            1.3.0: per-layer memory record + freed buffers; rs.trained holds file
                                                  refs only; a changed rung file is refused at delta time
  test_successor_contract_and_prereg_on_the_real_dataset
                                                  1.3.0: r4's data pins (manifest a488e75b, ids 74f79199, 917/62, the
                                                  successor prereg) verify through the unchanged d1_source_contract.py

Server (CPU only; CHIMERA_SRC = the ef1ffb9 tree the S5 child was written with):
    cd $OC/distill/tools && EF=$OC/import/rq27b/tree_ef1ffb9 && CHIMERA_SRC=$EF/src CHIMERA_OMNI_SRC=$EF \\
        PYTHONPATH=$EF/src CUDA_VISIBLE_DEVICES="" /home/cereal/model-surgery-lab/venv/bin/python -m pytest \\
        -o addopts="" -p no:cacheprovider test_d1_trunk_recovery.py -q
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d1_trunk_recovery as d1  # noqa: E402

d1.bind_chimera()
from chimera_omni.config import ChimeraConfig  # noqa: E402
from chimera_omni.importing.safetensors_io import read_safetensors, write_safetensors  # noqa: E402
from chimera_omni.modeling import ChimeraOmniModel  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

HIDDEN = 32
PARENT_W = 48
Q = 24
K3 = 10
PAD = 2
W = Q + K3 + PAD
LAYERS = 2
VOCAB = 2048
ROWS = 6
SEQ = 32


def chimera_config(width: int, meta: dict[str, str] | None = None) -> ChimeraConfig:
    cfg = ChimeraConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=width, num_layers=LAYERS, num_heads=4,
        num_kv_heads=4, max_sequence_length=128, block_pattern=("kda", "gated_gqa"), residual_mode="none",
        attention_head_dim=8, kda_head_dim=8, kda_value_head_dim=8, kda_decay_rank=8, native_feature_dim=32,
        gqa_rope_dim=8, gqa_rope_theta=1.0e4, activation_kind="silu",
    )
    if meta:
        cfg = dataclasses.replace(cfg, metadata={**dict(cfg.metadata), **meta})
    cfg.validate()
    return cfg


def write_model(state: dict[str, torch.Tensor], cfg: ChimeraConfig, directory: Path) -> None:
    write_safetensors(state, directory, config=cfg.to_manifest(), ledger=cfg.ledger().to_manifest(),
                      source={"import_source": "d1 unit test"}, chimera_commit="d" * 40)


def build_fixture(root: Path, *, meta: bool = True) -> dict[str, Path]:
    torch.manual_seed(5)
    pcfg = chimera_config(PARENT_W)
    parent = ChimeraOmniModel(pcfg).float()
    pstate = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    write_model(pstate, pcfg, root / "parent")
    ccfg = chimera_config(W, {"c10_trunk_intermediate_size": str(Q), "c10_k3_channels": str(K3)} if meta else None)
    cstate = {k: v.clone() for k, v in pstate.items()}
    keep = torch.arange(0, PARENT_W, 2)[:Q]
    g = torch.Generator().manual_seed(9)
    for k in range(LAYERS):
        up = pstate[f"blocks.{k}.ffn.up.weight"]
        down = pstate[f"blocks.{k}.ffn.down.weight"]
        gate_q, up_q = up[:PARENT_W][keep], up[PARENT_W:][keep]
        k3_gate = torch.randn(K3, HIDDEN, generator=g) * 0.2
        k3_up = torch.randn(K3, HIDDEN, generator=g) * 0.2
        k3_down = torch.randn(HIDDEN, K3, generator=g) * 0.05
        zr = torch.zeros(PAD, HIDDEN)
        cstate[f"blocks.{k}.ffn.up.weight"] = torch.cat([gate_q, k3_gate, zr, up_q, k3_up, zr]).contiguous()
        cstate[f"blocks.{k}.ffn.down.weight"] = torch.cat([down[:, keep] * 1.7, k3_down, torch.zeros(HIDDEN, PAD)], 1).contiguous()
    child = ChimeraOmniModel(ccfg).float()
    child.load_state_dict(cstate, strict=True)
    write_model(cstate, ccfg, root / "child")
    text_rows = min(int(ccfg.ledger_band_begin()), int(pcfg.ledger_band_begin()))
    ids = torch.randint(0, text_rows, (ROWS, SEQ), generator=torch.Generator().manual_seed(3))
    save_file({"input_ids": ids, "other": torch.zeros(3)}, str(root / "ids.safetensors"))
    return {"parent": root / "parent", "child": root / "child", "ids": root / "ids.safetensors"}


def charted_child(fx: dict[str, Path], root: Path) -> Path:
    """The union child with one charted (ACE-style) media block inserted after text layer 0 -- the OMNI1 layout.
    Needs a tree that knows the ``charted`` kind (branch claude/omni-composite lineage, e.g. tree_11adb32)."""
    charted = pytest.importorskip("chimera_omni.charted", reason="this chimera tree has no charted block kind")
    from chimera_omni.importing.safetensors_io import read_checkpoint_config

    ccfg = ChimeraConfig.from_manifest(read_checkpoint_config(fx["child"])["config"])
    spec = charted.ChartedBlockSpec(layer_index=1, source="ace", component="ace.dit.block12.mlp", type_ids=(4,), width=24,
                                    heads=2, kv_heads=1, head_dim=8, mlp_width=20, rope_theta=1.0e6, sliding_window=3,
                                    time_freq_dim=8, source_depth_fraction=0.5)
    spec.validate()
    cfg = dataclasses.replace(ccfg, num_layers=LAYERS + 1, block_pattern=("kda", "charted", "gated_gqa"),
                              charted_blocks=(spec,))
    cfg.validate()
    torch.manual_seed(13)
    model = ChimeraOmniModel(cfg).float()
    base = {n: v.torch() for n, v in read_safetensors(fx["child"]).items()}
    state = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("blocks.1."):
            state[name] = tensor.detach().clone()               # the charted block's own (random) state
        elif name.startswith("blocks.2."):
            state[name] = base["blocks.1." + name[len("blocks.2."):]]
        else:
            state[name] = base[name]
    model.load_state_dict(state, strict=True)
    write_model(state, cfg, root / "child_charted")
    return root / "child_charted"


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> dict[str, Path]:
    return build_fixture(tmp_path_factory.mktemp("d1fx"))


def argv(fx: dict[str, Path], out: Path, *extra: str) -> list[str]:
    return ["--child", str(fx["child"]), "--parent", str(fx["parent"]), "--ids", str(fx["ids"]), "--held-rows", "1",
            "--devices", "cpu", "--dtype", "float32", "--row-batch", "2", "--tokens-per-step", "64", "--micro-tokens", "32",
            "--epochs", "25", "--lr", "3e-3", "--warmup-steps", "3", "--clip", "0", "--min-free-gib", "0", "--log-every", "50",
            "--out", str(out), *extra]


# ------------------------------------------------------------------ the objective
def reference_loss(xm, y, gate, up, down, norm, eps, residual):
    x = xm
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm
    f = (F.silu(n @ gate.T) * (n @ up.T)) @ down.T
    pred, target = (x + f, y) if residual else (f, y - x)
    rel = (pred - target).norm(dim=-1) / target.norm(dim=-1)
    cos = (pred * target).sum(-1) / (pred.norm(dim=-1) * target.norm(dim=-1))
    return (rel + 1 - cos).sum()


@pytest.mark.parametrize("residual", [True, False])
def test_analytic_gradient_equals_autograd(residual: bool) -> None:
    g = torch.Generator().manual_seed(1)
    T, h, q = 40, 16, 12
    xm = torch.randn(T, h, generator=g, dtype=torch.float64) * torch.linspace(0.5, 4.0, T, dtype=torch.float64).unsqueeze(1)
    y = xm + 0.3 * torch.randn(T, h, generator=g, dtype=torch.float64)
    p = {"gate": torch.randn(q, h, generator=g, dtype=torch.float64) * 0.4,
         "up": torch.randn(q, h, generator=g, dtype=torch.float64) * 0.4,
         "down": torch.randn(h, q, generator=g, dtype=torch.float64) * 0.4,
         "norm": 1.0 + 0.1 * torch.randn(h, generator=g, dtype=torch.float64)}
    scale = 1.0 / 97.0
    loss, grads, stats = d1.d1_loss_and_grads(xm, y, p, 1e-6, scale, residual=residual)
    leaves = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    ref = reference_loss(xm, y, leaves["gate"], leaves["up"], leaves["down"], leaves["norm"], 1e-6, residual) * scale
    ref.backward()
    assert torch.allclose(loss, ref.detach(), rtol=1e-13, atol=0)
    for k in d1.PARAM_NAMES:
        assert torch.allclose(grads[k], leaves[k].grad, rtol=1e-10, atol=1e-14), k
    assert stats["tokens"] == T


def test_host_reduced_shards_equal_the_full_batch() -> None:
    g = torch.Generator().manual_seed(2)
    T, h, q = 50, 16, 12
    xm = torch.randn(T, h, generator=g, dtype=torch.float64) * 2
    y = xm + 0.5 * torch.randn(T, h, generator=g, dtype=torch.float64)
    p = {"gate": torch.randn(q, h, generator=g, dtype=torch.float64) * 0.3,
         "up": torch.randn(q, h, generator=g, dtype=torch.float64) * 0.3,
         "down": torch.randn(h, q, generator=g, dtype=torch.float64) * 0.3,
         "norm": torch.ones(h, dtype=torch.float64)}
    _, full, _ = d1.d1_loss_and_grads(xm, y, p, 1e-6, 1.0 / T)
    total = None
    for lo, hi in ((0, 7), (7, 31), (31, 50)):
        _, part, _ = d1.d1_loss_and_grads(xm[lo:hi], y[lo:hi], p, 1e-6, 1.0 / T)
        total = part if total is None else {k: total[k] + part[k] for k in total}
    for k in d1.PARAM_NAMES:
        assert torch.allclose(total[k], full[k], rtol=1e-12, atol=1e-15), k
    # trainer: 2 host-staged replicas == 1 replica (fp64 master)
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, clip=1.0, norm_eps=1e-6, residual=True, mm_dtype=None,
              master_dtype=torch.float64)
    one = d1.TrunkTrainer(p, ["cpu"], **kw)
    two = d1.TrunkTrainer(p, ["cpu", "cpu"], **kw)
    for step in range(4):
        idx = torch.randperm(T, generator=torch.Generator().manual_seed(step))[:40]
        x_s, y_s = xm[idx], y[idx]
        h1, l1, _ = one.shard_grads([[(x_s, y_s)]], 40)
        h2, l2, _ = two.shard_grads([[(x_s[:13], y_s[:13])], [(x_s[13:25], y_s[13:25]), (x_s[25:], y_s[25:])]], 40)
        assert abs(l1 - l2) < 1e-12
        one.apply(h1, 1e-2)
        two.apply(h2, 1e-2)
    for k in d1.PARAM_NAMES:
        assert torch.allclose(one.master()[k], two.master()[k], rtol=1e-11, atol=1e-14), k
        assert torch.equal(two.replicas[0][k], two.replicas[1][k]), f"replica {k} not host-staged"
    # 1.2.4: the threaded step (forced on CPU) with lazy micro-batches == the sequential step
    seq = d1.TrunkTrainer(p, ["cpu", "cpu"], parallel=False, **kw)
    thr = d1.TrunkTrainer(p, ["cpu", "cpu"], parallel=True, **kw)
    assert d1.TrunkTrainer(p, ["cpu", "cpu"], **kw).parallel is False      # auto: CPU replicas stay sequential
    for step in range(3):
        idx = torch.randperm(T, generator=torch.Generator().manual_seed(10 + step))[:40]
        x_s, y_s = xm[idx], y[idx]
        hs, ls, ss = seq.shard_grads([[(x_s[:13], y_s[:13])], [(x_s[13:25], y_s[13:25]), (x_s[25:], y_s[25:])]], 40)
        ht, lt, st = thr.shard_grads([[lambda: (x_s[:13], y_s[:13])],
                                      [lambda: (x_s[13:25], y_s[13:25]), lambda: (x_s[25:], y_s[25:])]], 40)
        assert abs(ls - lt) < 1e-13 and abs(ss["rel_mean"] - st["rel_mean"]) < 1e-13
        for k in d1.PARAM_NAMES:
            assert torch.allclose(hs[k], ht[k], rtol=1e-13, atol=1e-16), k
        seq.apply(hs, 1e-2)
        thr.apply(ht, 1e-2)
    for k in d1.PARAM_NAMES:
        assert torch.allclose(seq.master()[k], thr.master()[k], rtol=1e-12, atol=1e-15), k


def test_threaded_shards_run_concurrently(monkeypatch) -> None:
    import threading
    import time

    real = d1.d1_loss_and_grads
    seen: set[int] = set()

    def slow(*a, **k):
        seen.add(threading.get_ident())
        time.sleep(0.25)
        return real(*a, **k)

    monkeypatch.setattr(d1, "d1_loss_and_grads", slow)
    h, q = 8, 6
    p = {"gate": torch.randn(q, h), "up": torch.randn(q, h), "down": torch.randn(h, q), "norm": torch.ones(h)}
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, clip=1.0, norm_eps=1e-6, residual=True, mm_dtype=None,
              master_dtype=torch.float32)
    x = torch.randn(8, h)
    shards = [[(x[:2], x[:2] + 1), (x[2:4], x[2:4] + 1)], [(x[4:6], x[4:6] + 1), (x[6:], x[6:] + 1)]]
    t0 = time.perf_counter()
    d1.TrunkTrainer(p, ["cpu", "cpu"], parallel=False, **kw).shard_grads(shards, 8)
    t_seq = time.perf_counter() - t0
    seen.clear()
    t0 = time.perf_counter()
    d1.TrunkTrainer(p, ["cpu", "cpu"], parallel=True, **kw).shard_grads(shards, 8)
    t_thr = time.perf_counter() - t0
    assert t_seq >= 0.95 and len(seen) == 2, (t_seq, len(seen))
    assert t_thr < 0.75, f"threaded step took {t_thr:.3f}s (sequential {t_seq:.3f}s): the devices did not overlap"


# ------------------------------------------------------------------ end to end
def rerun_from_scratch(fx: dict[str, Path], delta_dir: Path | None, ids_path: Path | None = None) -> torch.Tensor:
    """embed -> every child block built from base (+ delta via apply_delta) -> run_block; fp32 CPU."""
    ck = d1.open_checkpoint(fx["child"])
    got = load_file(str(ids_path or fx["ids"]))
    ids = got["input_ids"]
    lengths = got.get("lengths", torch.full((ids.shape[0],), ids.shape[1]))
    valid = torch.arange(ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    emb = read_safetensors(fx["child"], ["embed.weight"])["embed.weight"].torch()
    x = F.embedding(torch.where(valid, ids, torch.zeros_like(ids)), emb)
    x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
    manifest = values = None
    if delta_dir is not None:
        manifest = json.loads((delta_dir / "D1_delta.json").read_text())
        values = load_file(str(delta_dir / "D1_delta.safetensors"))
    for k in range(LAYERS):
        state, _ = d1.read_block_state(ck, k)
        if manifest is not None:
            full = {f"blocks.{k}.{n}": t for n, t in state.items()}
            for entry in manifest["entries"]:
                if entry["base_tensor"] in full:
                    full[entry["base_tensor"]] = d1.apply_delta(full[entry["base_tensor"]], entry, values[entry["key"]])
            state = {n[len(f"blocks.{k}."):]: t for n, t in full.items()}
        blk = d1.build_block(ck, k, state, "cpu", torch.float32)
        with torch.no_grad():   # the tool's row batches (--row-batch 2): identical shapes, identical bits
            x = torch.cat([d1.run_block(blk, x[r:r + 2], valid[r:r + 2]) for r in range(0, ROWS, 2)])
    return x.reshape(-1, HIDDEN)


def test_two_layer_end_to_end(fx: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "run"
    rs = d1.run(argv(fx, out))
    rec = json.loads((out / "d1_receipt.json").read_text())
    assert rec["train_layers"] == [0, 1]
    assert (out / "d1_receipt.json.sha256").read_text().strip() == d1.sha256_file(out / "d1_receipt.json")
    for layer in rec["layers"]:
        assert layer["trained"] is True
        assert layer["decomposition_check"]["bit_exact"] is True
        r = layer["record"]
        for split in ("held", "fit_sample"):
            b, a = r["before"][split], r["after"][split]
            assert b["trunk_only_residual"]["finite"] and a["trunk_only_residual"]["finite"]
            assert a["trunk_only_residual"]["d1_loss"] < b["trunk_only_residual"]["d1_loss"], (layer["text_layer"], split)
            assert b["k3_term_energy_rel_trunk_ffn"] > 0 and a["k3_term_energy_rel_trunk_ffn"] > 0
            assert b["full_layer_with_k3_residual"]["relL2"] != b["trunk_only_residual"]["relL2"]   # K3 is outside the loss
        assert r["loss_curve"][-1]["loss"] < r["loss_curve"][0]["loss"]
    kl = rec["head_kl"]
    assert kl["d1"]["logits_finite"] and kl["original"]["logits_finite"]
    assert kl["vocab_rows"][1] <= VOCAB and kl["positions"] == SEQ

    # the delta touches the trunk slices + norm2 only; K3, pad and everything else stay the base bytes
    delta_dir = out / "delta"
    manifest = json.loads((delta_dir / "D1_delta.json").read_text())
    assert manifest["schema"] == d1.DELTA_SCHEMA and manifest["trained_text_layers"] == [0, 1]
    assert manifest["parameters"] == LAYERS * (3 * HIDDEN * Q + HIDDEN)
    assert all(meta.get("sha256") for meta in manifest["base"]["files"].values())
    values = load_file(str(delta_dir / "D1_delta.safetensors"))
    base = read_safetensors(fx["child"])
    for entry in manifest["entries"]:
        assert d1.tensor_bytes_sha256(values[entry["key"]]) == entry["sha256"]
        assert base[entry["base_tensor"]].sha256() == entry["base_tensor_sha256"]
    for k in range(LAYERS):
        up_b = base[f"blocks.{k}.ffn.up.weight"].torch()
        down_b = base[f"blocks.{k}.ffn.down.weight"].torch()
        up_n, down_n = up_b.clone(), down_b.clone()
        for entry in manifest["entries"]:
            if entry["base_tensor"] == f"blocks.{k}.ffn.up.weight":
                up_n = d1.apply_delta(up_n, entry, values[entry["key"]])
            if entry["base_tensor"] == f"blocks.{k}.ffn.down.weight":
                down_n = d1.apply_delta(down_n, entry, values[entry["key"]])
        assert torch.equal(up_n[Q:W], up_b[Q:W]) and torch.equal(up_n[W + Q:], up_b[W + Q:])      # K3 + pad rows
        assert torch.equal(down_n[:, Q:], down_b[:, Q:])                                          # K3 + pad columns
        assert not torch.equal(up_n[:Q], up_b[:Q]) and not torch.equal(down_n[:, :Q], down_b[:, :Q])
        assert torch.equal(d1.trained_tensors(rs, k)["gate"], up_n[:Q])
    # base + delta re-run from scratch == the tool's own final child stream (the prefix advance kept the K3 term)
    assert torch.equal(rerun_from_scratch(fx, delta_dir), rs.C)
    assert not torch.equal(rerun_from_scratch(fx, None), rs.C)

    # resume: rungs reloaded (no training), identical delta bytes
    rs2 = d1.run(argv(fx, out, "--resume"))
    rec2 = json.loads((out / "d1_receipt.json").read_text())
    assert all(layer.get("resumed_from") for layer in rec2["layers"])
    assert rec2["delta"]["file_sha256"] == rec["delta"]["file_sha256"]
    assert torch.equal(rs2.C, rs.C)


def test_right_padded_rows(fx: dict[str, Path], tmp_path: Path) -> None:
    """`lengths` [rows]: positions >= length are invalid -- zero in the streams, out of training / diagnostics / KL;
    base + delta re-run with the same validity == the tool's stream; padding ids may be anything."""
    got = load_file(str(fx["ids"]))
    lengths = torch.tensor([SEQ, 20, SEQ, 25, SEQ, 17])
    ids = got["input_ids"].clone()
    ids[1, 20:] = 10 ** 6                       # padding outside the vocab is never embedded
    padded = tmp_path / "padded.safetensors"
    save_file({"input_ids": ids, "lengths": lengths}, str(padded))
    out = tmp_path / "pad"
    a = argv(fx, out)
    a[a.index("--ids") + 1] = str(padded)
    rs = d1.run(a)
    rec = json.loads((out / "d1_receipt.json").read_text())
    assert rec["rows"]["valid_tokens"] == int(lengths.sum())
    assert rec["rows"]["fit_valid_tokens"] == int(lengths[:5].sum()) and rec["rows"]["held_valid_tokens"] == 17
    assert rec["layers"][0]["record"]["fit_tokens"] == int(lengths[:5].sum())
    assert rec["layers"][0]["record"]["before"]["held"]["trunk_only_residual"]["tokens"] == 17
    assert rec["head_kl"]["positions"] == 17
    invalid = (torch.arange(SEQ).unsqueeze(0) >= lengths.unsqueeze(1)).reshape(-1)
    assert bool((rs.C[invalid] == 0).all()) and bool((rs.P[invalid] == 0).all())
    fixed = tmp_path / "padded_ok.safetensors"
    ids[1, 20:] = 0
    save_file({"input_ids": ids, "lengths": lengths}, str(fixed))
    assert torch.equal(rerun_from_scratch(fx, out / "delta", fixed), rs.C)


def test_charted_child_maps_text_layers_and_checks_identity(fx: dict[str, Path], tmp_path: Path) -> None:
    """OMNI1 layout: a charted block between text layers is verified identity on text rows and skipped; text
    layer i maps to the child's i-th non-charted block; the delta names the child's block indices; base + delta
    re-run through all three child blocks == the tool's stream; the plain child's training is reproduced."""
    child = charted_child(fx, tmp_path)
    cfx = {**fx, "child": child}
    rs = d1.run(argv(cfx, tmp_path / "charted"))
    rec = json.loads((tmp_path / "charted" / "d1_receipt.json").read_text())
    assert rec["child_charted_blocks"] == [1]
    assert rec["charted_checks"] == [{"block": 1, "identity_on_text_rows": True, "checked_tokens": SEQ}]
    assert [(r["text_layer"], r["child_block"], r["parent_block"]) for r in rec["layers"]] == [(0, 0, 0), (1, 2, 1)]
    manifest = json.loads((tmp_path / "charted" / "delta" / "D1_delta.json").read_text())
    assert sorted({e["base_tensor"].split(".")[1] for e in manifest["entries"]}) == ["0", "2"]
    plain = d1.run(argv(fx, tmp_path / "plain"))
    for i in range(LAYERS):
        for name in d1.PARAM_NAMES:
            assert torch.equal(d1.trained_tensors(rs, i)[name], d1.trained_tensors(plain, i)[name]), (i, name)
    assert torch.equal(rs.C, plain.C)
    ck = d1.open_checkpoint(child)
    values = load_file(str(tmp_path / "charted" / "delta" / "D1_delta.safetensors"))
    ids = load_file(str(fx["ids"]))["input_ids"]
    x = F.embedding(ids, read_safetensors(child, ["embed.weight"])["embed.weight"].torch())
    valid = torch.ones(ids.shape, dtype=torch.bool)
    for k in range(LAYERS + 1):
        state, _ = d1.read_block_state(ck, k)
        full = {f"blocks.{k}.{n}": t for n, t in state.items()}
        for entry in manifest["entries"]:
            if entry["base_tensor"] in full:
                full[entry["base_tensor"]] = d1.apply_delta(full[entry["base_tensor"]], entry, values[entry["key"]])
        blk = d1.build_block(ck, k, {n[len(f"blocks.{k}."):]: t for n, t in full.items()}, "cpu", torch.float32)
        with torch.no_grad():
            x = torch.cat([d1.run_block(blk, x[r:r + 2], valid[r:r + 2]) for r in range(0, ROWS, 2)])
    assert torch.equal(x.reshape(-1, HIDDEN), rs.C)


def test_lr_zero_reproduces_the_base(fx: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "lr0"
    rs = d1.run(argv(fx, out, "--lr", "0", "--epochs", "1"))
    base = read_safetensors(fx["child"])
    for k in range(LAYERS):
        up_b = base[f"blocks.{k}.ffn.up.weight"].torch()
        assert torch.equal(d1.trained_tensors(rs, k)["gate"], up_b[:Q])
        assert torch.equal(d1.trained_tensors(rs, k)["up"], up_b[W:W + Q])
        assert torch.equal(d1.trained_tensors(rs, k)["down"], base[f"blocks.{k}.ffn.down.weight"].torch()[:, :Q])
    assert torch.equal(rs.C, rerun_from_scratch(fx, None))
    kl = json.loads((out / "d1_receipt.json").read_text())["head_kl"]
    assert kl["d1"]["kl_mean"] == pytest.approx(kl["original"]["kl_mean"], rel=1e-6, abs=1e-9)


def test_single_layer_and_stop_after(fx: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "one"
    d1.run(argv(fx, out, "--layers", "1", "--stop-after-layer", "1"))
    rec = json.loads((out / "d1_receipt.json").read_text())
    assert [layer["trained"] for layer in rec["layers"]] == [False, True]
    assert json.loads((out / "delta" / "D1_delta.json").read_text())["trained_text_layers"] == [1]
    out2 = tmp_path / "zero"
    d1.run(argv(fx, out2, "--layers", "1", "--stop-after-layer", "0", "--base-digest", "headers"))
    rec2 = json.loads((out2 / "d1_receipt.json").read_text())
    assert len(rec2["layers"]) == 1 and rec2["delta"] is None and "head_kl" not in rec2


def test_two_replica_run_matches_one_replica(fx: dict[str, Path], tmp_path: Path) -> None:
    a = d1.run(argv(fx, tmp_path / "dp1", "--master-dtype", "float64", "--epochs", "4"))
    b = d1.run(argv(fx, tmp_path / "dp2", "--master-dtype", "float64", "--epochs", "4", "--devices", "cpu,cpu"))
    for k in range(LAYERS):
        for name in d1.PARAM_NAMES:
            assert torch.allclose(d1.trained_tensors(a, k)[name], d1.trained_tensors(b, k)[name], rtol=1e-6, atol=1e-7)


def test_block_build_matches_the_fold_engine(fx: dict[str, Path]) -> None:
    fold = HERE.parent.parent / "composite" / "tools"
    sys.path.insert(0, str(fold))
    try:
        import c10_virtual_child_check as vcheck
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"c10_virtual_child_check not importable here: {error!r}")
    ck = d1.open_checkpoint(fx["child"])
    views = read_safetensors(fx["child"])
    x = torch.randn(2, SEQ, HIDDEN, generator=torch.Generator().manual_seed(4))
    valid = torch.ones(2, SEQ, dtype=torch.bool)
    for k in range(LAYERS):
        state, _ = d1.read_block_state(ck, k)
        mine = d1.run_block(d1.build_block(ck, k, state, "cpu", torch.float32), x, valid)
        theirs = vcheck.run_block(vcheck.build_block(ck.cfg, k, views, torch.device("cpu"), torch.float32, None), x, valid)
        assert torch.equal(mine, theirs), k


def test_refusals(fx: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="overlaps"):
        d1.run(argv(fx, fx["child"] / "inside"))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "x.txt").write_text("not ours")
    with pytest.raises(SystemExit, match="did not create"):
        d1.run(argv(fx, foreign))
    bad_ids = tmp_path / "bad_ids.safetensors"
    save_file({"input_ids": torch.full((3, SEQ), 10 ** 6, dtype=torch.long)}, str(bad_ids))
    a = argv(fx, tmp_path / "bad")
    a[a.index("--ids") + 1] = str(bad_ids)
    with pytest.raises(SystemExit, match="outside the text rows"):
        d1.run(a)
    with pytest.raises(SystemExit, match="leaves no fit rows"):
        d1.run(argv(fx, tmp_path / "held", "--held-rows", str(ROWS)))
    plain = build_fixture(tmp_path / "plainfx", meta=False)
    with pytest.raises(SystemExit, match="trunk-width"):
        d1.run(argv(plain, tmp_path / "plain"))
    with pytest.raises(SystemExit, match="union layout"):
        d1.run(argv(plain, tmp_path / "plain2", "--trunk-width", str(Q), "--k3-channels", str(W)))


def test_delta_out_guard(fx: dict[str, Path], tmp_path: Path) -> None:
    cases = [(fx["child"], "overlaps"),                         # the base checkpoint itself
             (fx["parent"] / "sub", "overlaps"),                # inside a model directory
             (fx["child"].parent, "overlaps")]                  # a directory around a model directory
    foreign = tmp_path / "registered"
    foreign.mkdir()
    (foreign / "config.json").write_text("{}")
    cases.append((foreign, "did not create"))
    afile = tmp_path / "afile"
    afile.write_text("x")
    cases.append((afile, "not a directory"))
    for n, (target, match) in enumerate(cases):
        out = tmp_path / f"refused{n}"
        before = sorted(p.name for p in target.iterdir()) if target.is_dir() else None
        with pytest.raises(SystemExit, match=match):
            d1.run(argv(fx, out, "--layers", "1", "--stop-after-layer", "1", "--delta-out", str(target)))
        assert not out.exists(), "a refused --delta-out must refuse before --out is created"
        if before is not None:
            assert sorted(p.name for p in target.iterdir()) == before, "a refused --delta-out was written to"
    with pytest.raises(SystemExit, match="/mnt/ai_data"):
        d1.guard_delta_out(Path("/mnt/ai_data/omni-campaign/d1-delta"), tmp_path / "o", [fx["child"]])
    assert d1.guard_delta_out(tmp_path / "o" / "delta2", tmp_path / "o", [fx["child"]]) is False   # inside --out
    # a fresh --delta-out: marker + delta there, reusable by a second run into the same directory
    dx = tmp_path / "dx"
    d1.run(argv(fx, tmp_path / "ok1", "--layers", "1", "--stop-after-layer", "1", "--delta-out", str(dx)))
    assert (dx / d1.MARKER).is_file() and (dx / "D1_delta.safetensors").is_file() and (dx / "D1_delta.json").is_file()
    assert not (tmp_path / "ok1" / "delta").exists()
    first = (dx / "D1_delta.safetensors").read_bytes()
    d1.run(argv(fx, tmp_path / "ok2", "--layers", "1", "--stop-after-layer", "1", "--delta-out", str(dx)))
    assert (dx / "D1_delta.safetensors").read_bytes() == first


def test_dp_bench_cpu_smoke(tmp_path: Path) -> None:
    import d1_dp_bench

    out = tmp_path / "bench.json"
    res = d1_dp_bench.run(["--devices", "cpu,cpu", "--hidden", "32", "--trunk", "24", "--tokens-per-step", "96",
                           "--micro-tokens", "32", "--buffer-dtype", "float32", "--threaded", "on", "--warmup", "1",
                           "--steps", "2", "--out", str(out)])
    doc = json.loads(out.read_text())
    assert doc["tool"]["version"] == d1_dp_bench.TOOL_VERSION and doc["d1_tool"]["version"] == d1.TOOL_VERSION
    assert res["threaded"] is True and res["parity"]["steps"] == 3
    assert res["parity"]["max_abs_diff"] <= 1e-6 and res["parity"]["loss_abs_diff"] <= 1e-6
    assert len(res["seconds_per_step"]["current"]) == 2


def test_deterministic_safetensors_writer(tmp_path: Path) -> None:
    g = torch.Generator().manual_seed(6)
    tensors = {f"t{i}": torch.randn(3, 5, generator=g).to(dt) for i, dt in
               enumerate((torch.float32, torch.bfloat16, torch.float32, torch.float16, torch.float32, torch.bfloat16))}
    tensors["i"] = torch.arange(7, dtype=torch.int64)
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    d1.save_safetensors_deterministic(tensors, a, {"x": "1", "tool": "t"})
    d1.save_safetensors_deterministic(dict(reversed(list(tensors.items()))), b, {"tool": "t", "x": "1"})
    assert a.read_bytes() == b.read_bytes()
    got = load_file(str(a))
    assert set(got) == set(tensors) and all(torch.equal(got[k], tensors[k]) for k in tensors)
    assert read_safetensors(a)["t1"].sha256() == read_safetensors(b)["t1"].sha256()


def test_ids_loaders_and_lr_schedule(tmp_path: Path) -> None:
    import numpy as np

    raw = np.arange(4 * 8, dtype=np.uint32)
    raw.tofile(str(tmp_path / "ids.u32"))
    ids, lengths, rec = d1.load_ids(str(tmp_path / "ids.u32"), 8, 3)
    assert tuple(ids.shape) == (3, 8) and int(ids[2, 7]) == 23 and rec["rows_in_file"] == 4
    assert lengths.tolist() == [8, 8, 8]
    with pytest.raises(SystemExit):
        d1.load_ids(str(tmp_path / "ids.u32"), None, None)
    save_file({"a": torch.zeros(2, 4, dtype=torch.long), "b": torch.ones(2, 4, dtype=torch.int32)}, str(tmp_path / "two.safetensors"))
    with pytest.raises(SystemExit, match="name the ids tensor"):
        d1.load_ids(str(tmp_path / "two.safetensors"), None, None)
    ids, _lengths, rec = d1.load_ids(str(tmp_path / "two.safetensors") + "::b", None, None)
    assert ids.dtype == torch.int64 and int(ids.sum()) == 8 and rec["key"] == "b"
    assert d1.lr_at(0, 100, 1.0, 10, 0.1) == pytest.approx(0.1)
    assert d1.lr_at(9, 100, 1.0, 10, 0.1) == pytest.approx(1.0)
    assert d1.lr_at(10, 100, 1.0, 10, 0.1) == pytest.approx(1.0)
    assert d1.lr_at(100, 100, 1.0, 10, 0.1) == pytest.approx(0.1)
    assert d1.parse_layers("0-2,5", 8) == [0, 1, 2, 5]
    with pytest.raises(SystemExit):
        d1.parse_layers("9", 8)


# ------------------------------------------------------------------ 1.3.0: canonical prefix, boundary cache, memory
SUCCESSOR = Path("/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/d1-source-successor-v2")


def receipt(out: Path) -> dict:
    return json.loads((out / "d1_receipt.json").read_text())


def canonical_from(run_out: Path, canon: Path) -> str:
    """The canonical rung dir built from a finished run with the production sync tool; its SHA256SUMS pin."""
    import d1_canonical_rungs as cr

    return cr.sync(canon, run_out / "d1_receipt.json", run_out / "rungs", "test")["sha256sums_sha256"]


def test_prefix_import_reproduces_the_full_run(fx: dict[str, Path], tmp_path: Path) -> None:
    """--prefix-dir: L00 imported from the canonical dir (never retrained), L01 trained on top == the run that
    trained both layers itself: same L01 rung bytes, same delta bytes, same final stream.  Refusals: a wrong
    SHA256SUMS pin, more layers than listed, flags without --prefix-dir, a tampered rung."""
    import shutil

    a = d1.run(argv(fx, tmp_path / "a"))
    ra = receipt(tmp_path / "a")
    canon = tmp_path / "canon"
    pin = canonical_from(tmp_path / "a", canon)
    out = tmp_path / "b"
    b = d1.run(argv(fx, out, "--prefix-dir", str(canon), "--prefix-sums-sha256", pin, "--prefix-layers", "1"))
    rb = receipt(out)
    assert rb["layers"][0]["imported_from_prefix"]["sha256sums_sha256"] == pin and "rung" not in rb["layers"][0]
    assert rb["layers"][1]["rung"]["sha256"] == ra["layers"][1]["rung"]["sha256"]
    assert not (out / "rungs" / "L00.safetensors").exists()
    assert rb["delta"]["file_sha256"] == ra["delta"]["file_sha256"]
    assert torch.equal(b.C, a.C) and rb["prefix"]["layers"] == 1
    with pytest.raises(SystemExit, match="does not hash to --prefix-sums"):
        d1.run(argv(fx, tmp_path / "w1", "--prefix-dir", str(canon), "--prefix-sums-sha256", "0" * 64, "--prefix-layers", "1"))
    with pytest.raises(SystemExit, match="does not list"):
        d1.run(argv(fx, tmp_path / "w2", "--prefix-dir", str(canon), "--prefix-sums-sha256", pin, "--prefix-layers", "3"))
    with pytest.raises(SystemExit, match="need --prefix-dir"):
        d1.run(argv(fx, tmp_path / "w3", "--prefix-layers", "1"))
    bad = tmp_path / "canon_bad"
    shutil.copytree(canon, bad)
    target = bad / "L00.safetensors"
    os.chmod(target, 0o644)
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 1
    target.write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="does not hash"):
        d1.run(argv(fx, tmp_path / "w4", "--prefix-dir", str(bad), "--prefix-sums-sha256", pin, "--prefix-layers", "1"))


def test_boundary_resume_is_bit_identical(fx: dict[str, Path], tmp_path: Path, monkeypatch) -> None:
    """The boundary after L00 (P, C, B) == the layer-1 inputs a fresh run recomputes, bit for bit; a run that
    lost L01 resumes from it without one layer-0 forward and ends on the uninterrupted run's delta bytes and
    streams.  A corrupted or foreign cache is ignored and the run recomputes to the same result."""
    ref = d1.run(argv(fx, tmp_path / "ref"))
    ref_delta = receipt(tmp_path / "ref")["delta"]["file_sha256"]
    l0 = d1.run(argv(fx, tmp_path / "l0", "--stop-after-layer", "0"))    # recomputed layer-1 inputs
    bd, out = tmp_path / "bd", tmp_path / "run"
    first = d1.run(argv(fx, out, "--boundary-dir", str(bd)))
    latest = json.loads((bd / "LATEST.json").read_text())
    assert latest["layer"] == 0 and sorted(p.name for p in bd.iterdir()) == ["L00", "LATEST.json"]
    doc = json.loads((bd / "L00" / "boundary.json").read_text())
    n_tok = ROWS * SEQ
    assert doc["files"]["P"]["bytes"] == doc["files"]["C"]["bytes"] == n_tok * HIDDEN * 4
    assert doc["files"]["B"]["bytes"] == len(first.kl_rows) * SEQ * HIDDEN * 4
    print(f"BOUNDARY SIZE tiny: {doc['bytes']} bytes {({k: v['bytes'] for k, v in doc['files'].items()})}")
    assert receipt(out)["boundary"]["history"][0]["written"] is True
    for suffix in (".safetensors", ".json"):          # the crash: L01's rung was never committed
        (out / "rungs" / f"L01{suffix}").unlink()

    seen: dict = {}
    real_load = d1.BoundaryCache.load

    def spy_load(self, rs, expected):
        res = real_load(self, rs, expected)
        seen.update(res=res, P=rs.P.clone(), C=rs.C.clone(), B=None if rs.B is None else rs.B.clone())
        return res

    built: list = []
    real_build = d1.blocks_on_devices

    def spy_build(ck, k, state, devices, dtype):
        built.append((str(ck.root), k))
        return real_build(ck, k, state, devices, dtype)

    monkeypatch.setattr(d1.BoundaryCache, "load", spy_load)
    monkeypatch.setattr(d1, "blocks_on_devices", spy_build)
    rs = d1.run(argv(fx, out, "--boundary-dir", str(bd), "--resume"))
    assert seen["res"]["layer"] == 0
    assert torch.equal(seen["P"], l0.P) and torch.equal(seen["C"], l0.C) and torch.equal(seen["B"], l0.B)
    assert (str(fx["parent"].absolute()), 0) not in built and (str(fx["child"].absolute()), 0) not in built
    assert (str(fx["parent"].absolute()), 1) in built
    rec = receipt(out)
    assert rec["layers"][0]["from_boundary"] and "rung" in rec["layers"][1] and rec["boundary"]["resumed_from"]["layer"] == 0
    assert rec["delta"]["file_sha256"] == ref_delta
    assert torch.equal(rs.C, ref.C) and torch.equal(rs.P, ref.P)

    monkeypatch.setattr(d1.BoundaryCache, "load", real_load)
    monkeypatch.setattr(d1, "blocks_on_devices", real_build)
    cfile = bd / "L00" / "C.bin"
    raw = bytearray(cfile.read_bytes())
    raw[5] ^= 0x40
    cfile.write_bytes(bytes(raw))
    rs2 = d1.run(argv(fx, out, "--boundary-dir", str(bd), "--resume"))
    rec2 = receipt(out)
    assert rec2["boundary"]["resumed_from"] is None and rec2["delta"]["file_sha256"] == ref_delta
    assert torch.equal(rs2.C, ref.C)
    bd2 = tmp_path / "bd2"                               # a foreign identity (other lr) is never loaded
    d1.run(argv(fx, tmp_path / "other", "--boundary-dir", str(bd2), "--lr", "1e-3"))
    out3 = tmp_path / "run3"
    d1.run(argv(fx, out3, "--boundary-dir", str(bd2), "--resume"))
    assert receipt(out3)["boundary"]["resumed_from"] is None
    assert receipt(out3)["delta"]["file_sha256"] == ref_delta


def test_boundary_due_rule() -> None:
    due = d1.boundary_due
    assert not due(63, 63, True, False, 40, 1)                  # never after the last layer
    assert due(10, 63, False, False, -10 ** 6, 1)               # end of the imported prefix
    assert not due(9, 63, False, True, -10 ** 6, 1)             # mid-prefix: the next layer replays too
    assert due(11, 63, True, False, 10, 1) and due(12, 63, True, False, 11, 1)
    assert not due(12, 63, True, False, 11, 3) and due(14, 63, True, False, 11, 3)
    assert due(5, 63, True, False, 4, 0)                        # every <= 0 means every layer


def test_memory_log_and_rungs_not_in_ram(fx: dict[str, Path], tmp_path: Path) -> None:
    rs = d1.run(argv(fx, tmp_path / "m"))
    rec = receipt(tmp_path / "m")
    for layer in rec["layers"]:
        mem = layer["memory"]
        assert mem["after_free"]["rss"] > 0 and mem["after_free"]["host_mem_available"] > 0
        assert mem["freed"]["files_dropped"] >= 1 and mem["freed"]["malloc_trim_released"] is not None
    assert rec["host_memory_at_start"]["rss"] > 0 and rec["host_memory_at_end"]["rss"] > 0
    print(f"MEMORY L01 after free: {d1.memory_line(rec['layers'][1]['memory']['after_free'])}")
    assert all(set(v) == {"block", "base_shas", "file", "file_sha256"} for v in rs.trained.values())
    assert d1.PROGRESS["phase"] == "done"
    path = Path(rs.trained[0]["file"])
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="changed during the run"):
        d1.trained_tensors(rs, 0)


@pytest.mark.skipif(not (SUCCESSOR / "manifest.json").is_file(), reason="the successor dataset lives on the AI server")
def test_successor_contract_and_prereg_on_the_real_dataset() -> None:
    """The production data pins, unchanged from r4: d1_source_contract.py is Codex's file byte for byte, the
    successor manifest a488e75b admits the ids 74f79199 (917 fit / 62 held rows), the prereg verifies."""
    assert d1.sha256_file(HERE / "d1_source_contract.py") == "0f2092d06f3232e62dd2657a7985b9b3c69da9c0cd07231fbf2924e98d479a86"

    def args_for(held: str):
        return d1.build_parser().parse_args([
            "--child", "x", "--parent", "y", "--out", "z", "--ids", str(SUCCESSOR / "d1_source_ids.safetensors"),
            "--seq-len", "4096", "--lengths-key", "lengths", "--held-rows", held,
            "--preregistration", str(SUCCESSOR / "PREREGISTRATION_D1_SOURCE_SUCCESSOR_2026-10-13.md"),
            "--source-manifest", str(SUCCESSOR / "manifest.json"),
            "--source-manifest-sha256", "a488e75b46a667ebe91c23a08e5ce50a07a49be8208520f9af2d71bc060bbf88",
            "--source-validator-sha256", d1.sha256_file(HERE / "d1_source_contract.py")])

    pre = d1.preregistration_record(args_for("62"))
    src = d1.source_contract_record(args_for("62"))
    assert pre["path"].endswith("PREREGISTRATION_D1_SOURCE_SUCCESSOR_2026-10-13.md")
    assert src["dataset"]["sha256"] == "74f79199297891af208bfcb340a99d6bc846a2337ce08a9599f83bc11f2892ad"
    c = src["actual_counts"]
    assert (c["fit_rows"], c["held_rows"], c["fit_real_tokens"], c["diagnostic_real_tokens"]) == (917, 62, 3752131, 250280)
    print(f"PREREG {pre}")
    with pytest.raises(SystemExit, match="source contract refused"):
        d1.source_contract_record(args_for("61"))
