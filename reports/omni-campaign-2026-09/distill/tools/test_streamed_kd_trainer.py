"""Tests for streamed_kd_trainer.py (T(ii)).  CPU only; run with CUDA hidden.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<tools dir> python -m pytest -q test_streamed_kd_trainer.py

The engine tests build a tiny random Qwen3.5 VL checkpoint (text: 4 layers, 3 GDN
+ 1 gated-GQA, the real 3:1 pattern; a 1-block vision tower so the file has the
real `model.language_model.*` / `model.visual.*` layout) with the stock
`transformers` classes and compare the streamed / resident engines against a
stock full-load autograd of the same file.  The objective is checked against an
independent per-position float64 implementation written with loops.  Teacher
caches are written by the REAL `qtext_teacher_cache.py build-text` (imported
from this directory), so the reader is tested against the producer, not against
a copy of its format.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import streamed_kd_trainer as T  # noqa: E402

T.select_linear_attention_kernels("torch")

TOOL = HERE / "streamed_kd_trainer.py"
TEXT_ROWS = 290  # 10 "band" rows above the text rows of the 300-row tiny vocab


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def tiny_text_config(vocab: int = 300, inter: int = 48, layers: int = 4):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    types = (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)
    return Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=32, intermediate_size=inter, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_conv_kernel_dim=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=types, max_position_embeddings=512, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )


def save_tiny_model(directory: Path, seed: int) -> Path:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    torch.manual_seed(seed)
    config = Qwen3_5Config(
        text_config=tiny_text_config().to_dict(),
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 32, "patch_size": 4, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "num_position_embeddings": 16, "in_channels": 3},
        tie_word_embeddings=False,
    )
    model = Qwen3_5ForConditionalGeneration(config).to(torch.float32)
    with torch.no_grad():  # non-trivial norms and weights big enough to matter
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                p.copy_(torch.randn_like(p) * 0.1)
            elif p.dim() >= 2:
                p.copy_(torch.randn_like(p) * 0.08)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory), safe_serialization=True)
    index = directory / "model.safetensors.index.json"
    if not index.is_file():  # a small save is one file; the streamed engines read through an index
        from safetensors import safe_open

        with safe_open(str(directory / "model.safetensors"), framework="pt") as handle:
            names = list(handle.keys())
        total = (directory / "model.safetensors").stat().st_size
        index.write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": {n: "model.safetensors" for n in names}}))
    return directory


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory) -> Path:
    return save_tiny_model(tmp_path_factory.mktemp("tiny") / "student", seed=11)


@pytest.fixture(scope="module")
def teacher_dir(tmp_path_factory) -> Path:
    return save_tiny_model(tmp_path_factory.mktemp("tinyt") / "teacher", seed=23)


def write_corpus(path: Path, rows: int = 6, length: int = 20, seed: int = 7) -> Path:
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    motif = torch.randint(0, 280, (rows, 8), generator=g)
    motif[motif == 7] = 8  # the EOS id appears only where it is put
    ids = motif.repeat(1, (length + 7) // 8)[:, :length].contiguous()
    ids[:, 9] = 7  # an EOS id in every row
    lengths = torch.full((rows,), length, dtype=torch.int64)
    lengths[1] = length - 3  # one right-padded row: its tail is never a target
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"input_ids": ids, "lengths": lengths}, str(path))
    return path


@pytest.fixture(scope="module")
def qtext_cache(tmp_path_factory, teacher_dir) -> Path:
    """A sealed Q38-style cache written by the real qtext_teacher_cache.py."""

    work = tmp_path_factory.mktemp("cache")
    corpus = write_corpus(work / "corpus.safetensors")
    q = T.load_peer_module("qtext_teacher_cache")
    rc = q.main(["build-text", "--model-dir", str(teacher_dir), "--teacher", "tiny-t", "--corpus", str(corpus),
                 "--device", "cpu", "--dtype", "float32", "--linear-attn-kernel", "torch", "--top-k", "8",
                 "--vocab-rows", str(TEXT_ROWS), "--micro-batch", "2", "--rows-per-shard", "4",
                 "--head-chunk-rows", "64", "--out-root", str(work / "root"), "--teacher-digest", "full"])
    assert rc == 0
    sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    return work / "root" / "tiny-t" / sha


def write_xtok(path: Path, qwen_file: str, pairs: dict[int, list[int]], top_k: int = 4, seed: int = 0,
               corpus_sha256: str | None = None) -> Path:
    """An xtok target with the keys and metadata k3_teacher_cache.py xtok writes
    (plus, when given, the `qwen_corpus_sha256` provenance key the trainer reads)."""

    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    q_row, q_pos = [], []
    for row, positions in sorted(pairs.items()):
        q_row += [row] * len(positions)
        q_pos += positions
    n = len(q_pos)
    ids = torch.randint(0, TEXT_ROWS, (n, top_k), generator=g, dtype=torch.int32)
    ids[::3, 1] = -1  # some K3 tokens without an identical-byte Qwen token
    lp = (torch.log_softmax(torch.randn(n, top_k, generator=g), -1) + math.log(0.7)).to(torch.bfloat16)
    tensors = {"q_file": torch.zeros(n, dtype=torch.int32), "q_row": torch.tensor(q_row, dtype=torch.int32),
               "q_pos": torch.tensor(q_pos, dtype=torch.int32), "k3_pos": torch.tensor(q_pos, dtype=torch.int32),
               "qwen_ids": ids, "k3_logprobs": lp}
    metadata = {"tool": "test", "qwen_files": json.dumps([qwen_file])}
    if corpus_sha256 is not None:
        metadata["qwen_corpus_sha256"] = json.dumps({qwen_file: corpus_sha256})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    return path


def corpus_sha(qtext_cache: Path) -> str:
    return json.loads((qtext_cache / "manifest.json").read_text())["corpus"]["sha256"]


def write_xtok_receipt_chain(xtok: Path, qwen_file: str, ids_sha: str) -> Path:
    """The two files k3_teacher_cache.py xtok leaves beside / behind its target:
    xtok_receipt.json (target + boundary map shas, the keys cmd_xtok writes) and
    a boundary map carrying `ids_files_sha256` (the composite/corpus format)."""

    bm = xtok.parent / "boundary_map.json"
    bm.write_text(json.dumps({"schema": "neural-forge-k3-boundary-map-v2", "ids_files_sha256": {qwen_file: ids_sha}, "rows": []}))
    receipt = {"schema": "test.xtok", "boundary_map": {"path": str(bm), "sha256": hashlib.sha256(bm.read_bytes()).hexdigest()},
               "target": {"path": str(xtok), "sha256": hashlib.sha256(xtok.read_bytes()).hexdigest()}, "qwen_files": [qwen_file]}
    (xtok.parent / "xtok_receipt.json").write_text(json.dumps(receipt))
    return bm


def objective_cfg(text_rows: int, **kw) -> T.ObjectiveConfig:
    base = dict(head_chunk_positions=5, text_vocab_rows=text_rows, eos_ids=(7,))
    base.update(kw)
    return T.ObjectiveConfig(**base)


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------


def reference_objective(hidden, head, targets, cfg, denominator):
    """Independent loop implementation in float64."""

    H = hidden.shape[-1]
    logq_all = torch.log_softmax(hidden.reshape(-1, H) @ head[: cfg.text_vocab_rows].T, dim=-1)
    total = hidden.new_zeros(())
    for p in range(logq_all.shape[0]):
        lq = logq_all[p]
        nid = int(targets["next_ids"][p])
        if nid >= 0:
            total = total + (targets["ce_w"][p].double() + targets["eos_w"][p].double()) * (-lq[nid])
        for key_ids, key_lp, key_w in (("q_ids", "q_lp", "q_w"), ("k_ids", "k_lp", "k_w")):
            if key_w not in targets or float(targets[key_w][p]) <= 0:
                continue
            ids = [int(i) for i in targets[key_ids][p]]
            lps = [float(x) for x in targets[key_lp][p]]
            in_set = {i for i in ids if i >= 0}
            kl = hidden.new_zeros(())
            mass = 0.0
            for i, lp in zip(ids, lps):
                if i < 0:
                    continue
                pk = math.exp(lp)
                mass += pk
                kl = kl + pk * (lp - lq[i])
            p_tail = max(0.0, 1.0 - mass)
            rest = [v for v in range(cfg.text_vocab_rows) if v not in in_set]
            q_tail = torch.logsumexp(lq[rest], dim=0)
            if p_tail > 0:
                kl = kl + p_tail * (math.log(p_tail) - q_tail)
            total = total + float(targets[key_w][p]) * kl
        if "ul_ids" in targets and cfg.w_ul > 0:
            for c in targets["ul_ids"][p]:
                c = int(c)
                if c >= 0:
                    total = total + cfg.w_ul * (-torch.log(torch.clamp(1 - lq[c].exp(), min=1e-6)))
    return total / denominator


def test_objective_matches_independent_reference():
    torch.manual_seed(1)
    batch, length, H, V, text_rows = 2, 40, 12, 60, 50  # >= 2 motif periods: unlikelihood has candidates
    cfg = objective_cfg(text_rows, w_eos=0.3, w_ul=0.2)
    ids, targets = T.synthetic_parity_batch(batch, length, text_rows, 6, 8, 3, cfg.eos_ids, cfg)
    hidden = torch.randn(batch, length, H, dtype=torch.float64, requires_grad=True)
    head = torch.randn(V, H, dtype=torch.float64)
    denom = float(batch * (length - 1))
    ref = reference_objective(hidden, head, targets, cfg, denom)
    ref.backward()
    parts, grad = T.chunked_objective(hidden, head[:text_rows], targets, cfg, denominator=denom)
    assert abs(parts["total"] - float(ref.detach())) < 1e-10 * max(1.0, abs(float(ref.detach())))
    assert torch.allclose(grad, hidden.grad, atol=1e-11, rtol=1e-9)
    for key in ("q38", "k3", "ce", "eos", "ul"):
        assert parts[key] != 0.0, key  # every term exercised


def test_band_rows_never_enter_the_loss():
    torch.manual_seed(2)
    cfg = objective_cfg(40)
    ids, targets = T.synthetic_parity_batch(1, 6, 40, 6, 8, 0, cfg.eos_ids, cfg)
    hidden = torch.randn(1, 6, 8, dtype=torch.float64)
    head = torch.randn(55, 8, dtype=torch.float64)
    with pytest.raises(T.TrainerError):
        T.chunked_objective(hidden, head, targets, cfg, denominator=5.0)
    a, _ = T.chunked_objective(hidden, head[:40], targets, cfg, denominator=5.0)
    head2 = head.clone()
    head2[40:] += 100.0
    b, _ = T.chunked_objective(hidden, head2[:40], targets, cfg, denominator=5.0)
    assert a["total"] == b["total"]


def test_bucket_kl_and_jsd_identities():
    torch.manual_seed(4)
    logq = torch.log_softmax(torch.randn(3, 9, dtype=torch.float64), -1)
    logp = torch.log_softmax(torch.randn(3, 9, dtype=torch.float64), -1)
    ids = torch.arange(9).expand(3, 9).clone()
    full = (logp.exp() * (logp - logq)).sum(-1)
    assert torch.allclose(T.bucket_forward_kl(logq, ids, logp), full, atol=1e-12)
    logits = torch.randn(3, 9, dtype=torch.float64, requires_grad=True)  # K = V: no tail, gradient finite
    T.bucket_forward_kl(torch.log_softmax(logits, -1), ids, logp).sum().backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert torch.allclose(T.bucket_jsd(logq, ids, logq, 0.5), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert bool((T.bucket_jsd(logq, ids, logp, 0.5) > 0).all())


def test_k3_unmatched_mass_goes_to_the_tail():
    logq = torch.log_softmax(torch.randn(1, 10, dtype=torch.float64), -1)
    ids = torch.tensor([[2, -1, 5]])
    lp = torch.log(torch.tensor([[0.5, 0.3, 0.1]], dtype=torch.float64))
    kl = T.bucket_forward_kl(logq, ids, lp)
    rest = [v for v in range(10) if v not in (2, 5)]
    q_tail = torch.logsumexp(logq[0, rest], 0)
    expected = 0.5 * (math.log(0.5) - logq[0, 2]) + 0.1 * (math.log(0.1) - logq[0, 5]) + 0.4 * (math.log(0.4) - q_tail)
    assert abs(float(kl[0]) - float(expected)) < 1e-12


def test_fp32_core_wrapper_is_identity_in_fp32_and_upcasts_half():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    wrapped = mq.torch_chunk_gated_delta_rule
    original = wrapped._kd_wrapped
    torch.manual_seed(8)
    q, k, v = (torch.randn(1, 70, 2, 8) for _ in range(3))
    g = -torch.rand(1, 70, 2)
    beta = torch.rand(1, 70, 2)
    a, _ = wrapped(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    b, _ = original(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert torch.equal(a, b)  # fp32: untouched, so the fp32 parity arm is unaffected
    half = [t.to(torch.float16) for t in (q, k, v)]
    c, _ = wrapped(*half, g=g.to(torch.float16), beta=beta.to(torch.float16), use_qk_l2norm_in_kernel=True)
    d, _ = original(*(t.float() for t in half), g=g.to(torch.float16).float(), beta=beta.to(torch.float16).float(), use_qk_l2norm_in_kernel=True)
    assert c.dtype == torch.float16 and torch.equal(c, d.to(torch.float16))


def test_unlikelihood_candidates():
    row = torch.tensor([[1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 3]])
    cand = T.unlikelihood_candidates(row, 3, 4)
    assert sorted(int(c) for c in cand[0, 7] if c >= 0) == [3, 4]
    # at t=10 (predicting 3): followers of (1,2) are 3,4,5 -> gold 3 excluded
    assert sorted(int(c) for c in cand[0, 10] if c >= 0) == [4, 5]
    assert (cand[0, :4] < 0).all()


# --------------------------------------------------------------------------
# site plan and hooks
# --------------------------------------------------------------------------


def test_site_plan_counts_at_omni1_geometry():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=248_320, hidden_size=5120, intermediate_size=16_640, num_hidden_layers=64,
        num_attention_heads=24, num_key_value_heads=4, head_dim=256, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=48,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16,
    )
    plan = T.build_site_plan(cfg, T.SitePlanConfig())
    totals: dict[str, int] = {}
    for layer_type in cfg.layer_types:
        for spec in plan[layer_type]:
            for part, shape in spec.part_shapes().items():
                key = "gain" if part == "gain" else ("k3" if part.startswith("k3") else ("ffn" if spec.site.startswith("mlp") else "attn"))
                totals[key] = totals.get(key, 0) + int(np.prod(shape))
    assert totals["ffn"] == 116_391_936  # design section 4 D2, verbatim
    assert totals["k3"] == 35_651_584
    assert totals["gain"] == 786_432
    assert totals["attn"] <= 123_731_968  # the design's upper bound
    assert totals["attn"] == 99_614_720
    with pytest.raises(T.TrainerError):
        T.build_site_plan(cfg, T.SitePlanConfig(gqa_sites=("q_proj",)))
    with pytest.raises(T.TrainerError):
        T.build_site_plan(cfg, T.SitePlanConfig(gdn_sites=("in_proj_a",)))


def test_hooks_match_explicit_algebra_and_the_fold_rule():
    torch.manual_seed(5)
    hidden, inter, trunk = 6, 10, 4
    x = torch.randn(3, hidden, dtype=torch.float64)
    xi = torch.randn(3, inter, dtype=torch.float64)
    specs = {
        "up": T.SiteSpec("up", "ffn_in", hidden, inter, 2, 4.0, trunk=trunk, gain=True),
        "down": T.SiteSpec("down", "ffn_out", inter, hidden, 2, 4.0, trunk=trunk, k3_rank=3, k3_alpha=1.5),
        "full": T.SiteSpec("full", "full", hidden, hidden, 2, 2.0),
    }
    layer = torch.nn.Module()
    for name, spec in specs.items():
        setattr(layer, name, torch.nn.Linear(spec.in_features, spec.out_features, bias=False).double())
    T.install_site_hooks(layer, list(specs.values()))
    bound = {name: {k: torch.randn(s, dtype=torch.float64) for k, s in spec.part_shapes().items()} for name, spec in specs.items()}
    for name in specs:
        getattr(layer, name)._kd_bound = bound[name]
    up = layer.up(x)
    exp_up = x @ layer.up.weight.T
    exp_up[:, :trunk] += 2.0 * (x @ bound["up"]["a"].T @ bound["up"]["b"].T)
    exp_up[:, trunk:] *= bound["up"]["gain"]
    assert torch.allclose(up, exp_up)
    down = layer.down(xi)
    D = layer.down.weight
    exp_down = xi @ D.T + 2.0 * (xi[:, :trunk] @ bound["down"]["a"].T @ bound["down"]["b"].T) \
        + 0.5 * (xi[:, trunk:] @ bound["down"]["k3_a"].T @ bound["down"]["k3_b"].T)
    assert torch.allclose(down, exp_down)
    full = layer.full(x)
    assert torch.allclose(full, x @ layer.full.weight.T + 1.0 * (x @ bound["full"]["a"].T @ bound["full"]["b"].T))
    # the D4 fold rule stated in adapters_manifest.json reproduces the hooked FFN exactly
    g = bound["up"]["gain"]
    act = torch.nn.functional.silu(xi)  # any channel activation
    merged_down = D.clone()
    merged_down[:, :trunk] += 2.0 * bound["down"]["b"] @ bound["down"]["a"]
    merged_down[:, trunk:] += 0.5 * bound["down"]["k3_b"] @ bound["down"]["k3_a"]
    merged_down[:, trunk:] *= g
    gained = act.clone()
    gained[:, trunk:] *= g
    assert torch.allclose(layer.down(gained), act @ merged_down.T)


def test_bank_init_is_deterministic_and_gain_starts_at_one(tiny_dir):
    src = T.SafetensorsSource(tiny_dir)
    cfg = T.text_config_from_source(src, torch.float32, "sdpa")
    plan = T.build_site_plan(cfg, T.SitePlanConfig(rank=4, k3_rank=2, trunk_ffn_width=32))
    types = list(cfg.layer_types)
    a = T.AdapterBank(plan, types, "cpu", 9)
    b = T.AdapterBank(plan, types, "cpu", 9)
    for name in a.names:
        assert torch.equal(a.tensors[name], b.tensors[name])
        if name.endswith(".gain"):
            assert bool((a.tensors[name] == 1.0).all())
        if name.endswith(".b") or name.endswith("k3_b"):
            assert bool((a.tensors[name] == 0).all())
    frozen = T.AdapterBank(plan, types, "cpu", 9, train_gain=False)
    assert not any(n.endswith(".gain") for n in frozen.trainable_names())


# --------------------------------------------------------------------------
# engines vs the stock full load
# --------------------------------------------------------------------------


def _parity(tiny_dir, mode, layer_devices, forced=()):
    src = T.SafetensorsSource(tiny_dir)
    cfg = T.text_config_from_source(src, torch.float32, "sdpa")
    obj = objective_cfg(TEXT_ROWS, w_eos=0.2, w_ul=0.1, head_chunk_positions=7)
    plan = T.build_site_plan(cfg, T.SitePlanConfig(rank=4, k3_rank=2, trunk_ffn_width=32))
    types = list(cfg.layer_types)
    ids, targets = T.synthetic_parity_batch(2, 24, TEXT_ROWS, 6, 8, 5, obj.eos_ids, obj)
    denom = float(2 * 23)
    ref_bank = T.AdapterBank(plan, types, "cpu", 3, b_init_std=0.05, gain_jitter=0.1)
    _, text, head = T.load_reference(tiny_dir, torch.float32, "cpu", "sdpa")
    ref_parts = T.reference_forward_backward(text, head, ref_bank, ids, targets, obj, denominator=denom)
    bank = T.AdapterBank(plan, types, "cpu", 3, b_init_std=0.05, gain_jitter=0.1)
    engine = T.LayerwiseEngine(src, cfg, mode=mode, dtype=torch.float32, layer_devices=layer_devices,
                               head_device=layer_devices[-1], bank=bank, text_vocab_rows=TEXT_ROWS, force_host_hops=forced)
    parts = engine.forward_backward(ids, targets, obj, denominator=denom)
    # a second micro-batch through the same engine must not see stale boundary grads
    bank.zero_grad()
    parts = engine.forward_backward(ids, targets, obj, denominator=denom)
    engine.close()
    rows, per_site = T.compare_grads(T.collect_grads(ref_bank), T.collect_grads(bank))
    return ref_parts, parts, per_site


def test_streamed_engine_matches_stock_autograd(tiny_dir):
    ref_parts, parts, per_site = _parity(tiny_dir, "streamed", ["cpu"])
    assert abs(parts["total"] - ref_parts["total"]) <= 1e-6 * abs(ref_parts["total"])
    assert len(per_site) >= 10
    for site, entry in per_site.items():
        assert entry["min_ref_norm"] > 0, site
        assert entry["max_rel_l2"] < 1e-5, (site, entry)


def test_resident_engine_matches_stock_autograd(tiny_dir):
    ref_parts, parts, per_site = _parity(tiny_dir, "resident", ["cpu"], forced=(2,))
    assert abs(parts["total"] - ref_parts["total"]) <= 1e-6 * abs(ref_parts["total"])
    for site, entry in per_site.items():
        assert entry["max_rel_l2"] < 1e-5, (site, entry)


def test_engine_refuses_a_source_with_missing_layer_tensors(tiny_dir):
    src = T.SafetensorsSource(tiny_dir)

    class Missing:
        text_prefix = src.text_prefix

        def get(self, n):
            return src.get(n)

        def has(self, n):
            return src.has(n)

        def names_under(self, prefix):
            return [n for n in src.names_under(prefix) if not n.endswith("layers.2.mlp.up_proj.weight")]

        def config_dict(self):
            return src.config_dict()

        def identity(self):
            return {"kind": "test"}

    cfg = T.text_config_from_source(src, torch.float32, "sdpa")
    with pytest.raises(T.TrainerError):
        T.LayerwiseEngine(Missing(), cfg, mode="streamed", dtype=torch.float32, layer_devices=["cpu"], head_device="cpu", bank=None, text_vocab_rows=TEXT_ROWS)


def test_source_check_command_on_a_safetensors_source(tiny_dir, tmp_path):
    rc = T.main(["source-check", "--model", str(tiny_dir), "--kernels", "torch", "--dtype", "float32", "--trunk-ffn-width", "32",
                 "--lora-rank", "4", "--k3-lora-rank", "2", "--text-vocab-rows", str(TEXT_ROWS), "--seq-len", "8",
                 "--receipt", str(tmp_path / "sc.json")])
    assert rc == 0
    receipt = json.loads((tmp_path / "sc.json").read_text())
    assert receipt["pass"] and receipt["probe_layers"] == [0, 3] and receipt["names_checked_layers"] == 4
    assert receipt["probe_grad_nonzero"] > 0


def test_build_source_refuses_ambiguous_choices(tiny_dir):
    with pytest.raises(T.TrainerError):
        T.build_source(SimpleNamespace(model=str(tiny_dir), p1_chimera="x", source_plugin=None))
    with pytest.raises(T.TrainerError):
        T.build_source(SimpleNamespace(model=None, p1_chimera="x", p1_parent_config=None, source_plugin=None))


# --------------------------------------------------------------------------
# data: the Q-text cache reader against its producer, xtok, sampler, batches
# --------------------------------------------------------------------------


def test_qtext_reader_matches_the_producers_own_reader(qtext_cache):
    q = T.load_peer_module("qtext_teacher_cache")
    cache = T.QTextCache(qtext_cache)
    assert cache.rows == 6 and cache.top_k == 8 and cache.seq_len == 20
    for shard in cache.manifest["shard_read_order"]:
        produced = q.read_shard(qtext_cache / shard)
        for row in produced["record"]["row_table"]:
            got = cache.row(int(row["row"]))
            o, c = int(row["offset"]), int(row["count"])
            pos = produced["pos"][o : o + c]
            assert torch.equal(got["topk_ids"][pos], produced["topk_ids"][o : o + c].long())
            assert torch.equal(got["topk_logprobs"][pos], produced["topk_logprobs"][o : o + c].float())
            assert int(got["stored"].sum()) == c == row["length"]
            assert bool((got["topk_ids"][~got["stored"]] == -1).all())
    assert int(cache.row(1)["stored"].sum()) == 17  # the right-padded row


def test_qtext_reader_refuses_a_changed_corpus(qtext_cache, tmp_path):
    cache = T.QTextCache(qtext_cache)
    moved = write_corpus(tmp_path / "other.safetensors", seed=99)
    with pytest.raises(T.TrainerError):
        T.QTextCache(qtext_cache, corpus_path=moved)
    assert cache.rows == 6


def test_mixture_sampler_deterministic_resumable_and_fractions():
    a = T.DataSource("text", "text", SimpleNamespace(seq_len=4), row_ids=list(range(7)), token_budget=1e9)
    b = T.DataSource("onp", "onpolicy", SimpleNamespace(seq_len=4), row_ids=[100, 101, 102, 103, 104], fraction=0.5, start_after_tokens=40)
    s1 = T.MixtureSampler([a, b], 3)
    first = s1.next_rows(10, tokens_seen=0)
    assert all(src == 0 for src, _ in first)  # on-policy inactive before its start
    assert sorted(r for _, r in first[:7]) == list(range(7))  # a full permutation per epoch
    state = s1.state()
    later = s1.next_rows(200, tokens_seen=100)
    share = sum(1 for src, _ in later if src == 1) / len(later)
    assert 0.35 < share < 0.65
    assert all(r >= 100 for src, r in later if src == 1)  # row ids, not local indices
    s2 = T.MixtureSampler([a, b], 3)
    s2.load_state(json.loads(json.dumps(state)))
    assert s2.next_rows(200, tokens_seen=100) == later


def test_build_batch_weights_with_xtok(qtext_cache, tmp_path):
    cache = T.QTextCache(qtext_cache)
    xtok = write_xtok(tmp_path / "xtok_target.safetensors", "corpus", {2: [0, 5, 11], 4: [3]})
    k3 = T.XtokTargets(xtok, "corpus")
    assert k3.pairs == 4 and sorted(k3.by_row) == [2, 4]
    src = [T.DataSource("t", "text", cache, row_ids=sorted(cache.index)), T.DataSource("k", "k3", cache, xtok=k3, row_ids=[2, 4])]
    cfg = objective_cfg(TEXT_ROWS, lambda_k3=0.3, k3_q38_anchor=0.5, w_q38=1.0, w_ce=0.1, w_eos=0.1)
    ids, tg = T.build_batch(src, [(0, 1), (1, 2)], cfg)
    q_w = tg["q_w"].view(2, 20)
    k_w = tg["k_w"].view(2, 20)
    assert torch.allclose(q_w[0, :17], torch.ones(17)) and bool((q_w[0, 17:] == 0).all())  # padded row
    assert torch.allclose(q_w[1], torch.full((20,), 0.15))
    expect = torch.zeros(20)
    expect[[0, 5, 11]] = 0.15
    assert torch.allclose(k_w[1], expect) and bool((k_w[0] == 0).all())
    ce_w = tg["ce_w"].view(2, 20)
    assert bool((ce_w[0, 16:] == 0).all()) and bool((ce_w[0, :16] > 0).all())  # t+1 < length
    assert tg["next_ids"].view(2, 20)[1, :19].tolist() == ids[1, 1:].tolist()
    eos_w = tg["eos_w"].view(2, 20)
    assert float(eos_w[1, 8]) == pytest.approx(0.1) and int((eos_w > 0).sum()) == 2


def test_xtok_is_joined_only_to_its_own_corpus(qtext_cache, tmp_path):
    """Review finding (1.2.0): the K3 join was by file stem only."""

    cache = T.QTextCache(qtext_cache)
    good, other = corpus_sha(qtext_cache), "0" * 64
    pairs = {2: [0, 5, 11], 4: [3]}
    # 1. the metadata key
    ok = T.XtokTargets(write_xtok(tmp_path / "m" / "x.safetensors", "corpus", pairs, corpus_sha256=good), "corpus")
    got = ok.verify_against(cache)
    assert got["method"] == "metadata" and got["corpus_sha256"] == good and got["rows_in_cache"] == 2
    wrong = T.XtokTargets(write_xtok(tmp_path / "w" / "x.safetensors", "corpus", pairs, corpus_sha256=other), "corpus")
    with pytest.raises(T.TrainerError, match="Qwen corpus"):
        wrong.verify_against(cache)
    # 2. the receipt chain k3_teacher_cache.py xtok already writes
    chained = write_xtok(tmp_path / "c" / "xtok_target.safetensors", "corpus", pairs)
    bm = write_xtok_receipt_chain(chained, "corpus", good)
    got = T.XtokTargets(chained, "corpus").verify_against(cache)
    assert got["method"] == "receipt_chain" and got["boundary_map"]["path"] == str(bm)
    write_xtok_receipt_chain(chained, "corpus", other)  # the map pairs a different Qwen ids file
    with pytest.raises(T.TrainerError, match="boundary map pairs"):
        T.XtokTargets(chained, "corpus").verify_against(cache)
    write_xtok_receipt_chain(chained, "corpus", good)
    bm.write_text(bm.read_text() + " ")  # the map changed after the xtok build
    with pytest.raises(T.TrainerError, match="digests to"):
        T.XtokTargets(chained, "corpus").verify_against(cache)
    write_xtok_receipt_chain(chained, "corpus", good)
    write_xtok(chained, "corpus", pairs, seed=5)  # a rebuilt target under the old receipt
    with pytest.raises(T.TrainerError, match="not the receipt"):
        T.XtokTargets(chained, "corpus").verify_against(cache)
    # 3. no provenance: refused unless the data config accepts it
    bare = T.XtokTargets(write_xtok(tmp_path / "b" / "x.safetensors", "corpus", pairs), "corpus")
    with pytest.raises(T.TrainerError, match="no proof"):
        bare.verify_against(cache)
    assert bare.verify_against(cache, unverified_ok=True)["method"].startswith("unverified")
    # structural, whatever the provenance: a pair must predict a real token of its row
    for bad_pairs in ({0: [19]}, {1: [16]}, {0: [-1]}):
        bad = T.XtokTargets(write_xtok(tmp_path / "s" / "x.safetensors", "corpus", bad_pairs, corpus_sha256=good), "corpus")
        with pytest.raises(T.TrainerError, match="outside the row"):
            bad.verify_against(cache, unverified_ok=True)
    # and through the data config, with the provenance in the data identity
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"sources": [{"name": "k", "kind": "k3", "qtext_cache": str(qtext_cache), "xtok": str(tmp_path / "w" / "x.safetensors")}]}))
    with pytest.raises(T.TrainerError):
        T.load_data_config(data)
    data.write_text(json.dumps({"sources": [{"name": "k", "kind": "k3", "qtext_cache": str(qtext_cache), "xtok": str(tmp_path / "m" / "x.safetensors")}]}))
    _, identity = T.load_data_config(data)
    assert identity["sources"][0]["xtok_provenance"]["method"] == "metadata"


def test_loss_scaler_skips_and_backs_off():
    p = torch.zeros(3, requires_grad=True)
    p.grad = torch.tensor([1.0, float("inf"), 0.0])
    s = T.DynamicLossScaler(1024.0, growth_interval=2)
    assert not s.unscale_and_check([p]) and s.scale == 512.0 and s.skipped_steps == 1
    p.grad = torch.ones(3) * 512
    assert s.unscale_and_check([p]) and torch.equal(p.grad, torch.ones(3))
    p.grad = torch.ones(3)
    s.unscale_and_check([p])
    assert s.scale == 1024.0 and s.growths == 1


# --------------------------------------------------------------------------
# the train command: exact resume and data parallel == single process
# --------------------------------------------------------------------------


def _data_config(tmp_path: Path, qtext_cache: Path, seed: int = 0) -> Path:
    xtok = write_xtok(tmp_path / "xtok_target.safetensors", "corpus", {0: [1, 4, 9], 3: [2, 6, 12], 5: [0, 10]},
                      seed=seed, corpus_sha256=corpus_sha(qtext_cache))
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"sources": [
        {"name": "q38", "kind": "text", "qtext_cache": str(qtext_cache), "tokens": 1e6},
        {"name": "k3rows", "kind": "k3", "qtext_cache": str(qtext_cache), "xtok": str(xtok), "tokens": 5e5},
    ]}))
    return path


def _train_args(tiny_dir, data, out, safety, *extra):
    return ["train", "--model", str(tiny_dir), "--devices", "cpu", "--dtype", "float32", "--kernels", "torch",
            "--data-config", str(data), "--seq-len", "20", "--micro-batch", "2", "--total-steps", "4",
            "--text-vocab-rows", str(TEXT_ROWS), "--trunk-ffn-width", "32", "--lora-rank", "4", "--k3-lora-rank", "2",
            "--lr", "5e-3", "--gain-lr", "5e-3", "--warmup-steps", "1", "--eos-ids", "7", "--head-chunk-positions", "16",
            "--out", str(out), "--safety-dir", str(safety), "--safety-every-steps", "1", "--log-every", "1",
            "--no-preregistration", WAIVER, *extra]


WAIVER = "unit test: tiny synthetic model, no Stage D claim"


def test_exact_resume_is_bitwise(tmp_path, tiny_dir, qtext_cache):
    from safetensors.torch import load_file

    data = _data_config(tmp_path, qtext_cache)
    assert T.main(_train_args(tiny_dir, data, tmp_path / "full", tmp_path / "s_full")) == 0
    assert T.main(_train_args(tiny_dir, data, tmp_path / "half", tmp_path / "s_half", "--max-steps-this-run", "2")) == 0
    kept = sorted(p.name for p in (tmp_path / "s_half").glob("ckpt-*"))
    assert kept == ["ckpt-00000001", "ckpt-00000002"]
    assert T.main(_train_args(tiny_dir, data, tmp_path / "half", tmp_path / "s_half", "--resume", "auto")) == 0
    assert sorted(p.name for p in (tmp_path / "s_half").glob("ckpt-*")) == ["ckpt-00000003", "ckpt-00000004"]  # keep 2
    a = load_file(str(tmp_path / "full" / "adapters.safetensors"))
    b = load_file(str(tmp_path / "half" / "adapters.safetensors"))
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    opt_a = torch.load(tmp_path / "s_full" / "ckpt-00000004" / "optimizer.pt", weights_only=False)
    opt_b = torch.load(tmp_path / "s_half" / "ckpt-00000004" / "optimizer.pt", weights_only=False)
    for idx, st in opt_a["state"].items():
        assert torch.equal(st["exp_avg"], opt_b["state"][idx]["exp_avg"])
        assert torch.equal(st["exp_avg_sq"], opt_b["state"][idx]["exp_avg_sq"])
    receipt = json.loads((tmp_path / "full" / "training-receipt.json").read_text())
    assert receipt["applied_steps"] == 4
    assert all(math.isfinite(r["loss"]) for r in receipt["steps"])
    assert any(r["k3"] > 0 for r in receipt["steps"])  # the K3 term ran in training
    gains = [k for k in a if k.endswith(".gain")]
    assert gains and any(not torch.equal(a[k], torch.ones_like(a[k])) for k in gains)  # the gain trains
    # a changed config must not silently resume
    assert T.main(_train_args(tiny_dir, data, tmp_path / "half", tmp_path / "s_half", "--resume", "auto", "--lr", "1e-3")) == 2


def test_a_weight_moving_run_needs_the_pin_or_a_named_waiver(tmp_path, tiny_dir, qtext_cache, capsys):
    """Review finding (1.3.1): --preregistration was optional, so a D2 line without it trained with nothing pinned."""

    data = _data_config(tmp_path, qtext_cache)

    def without_waiver(args):
        i = args.index("--no-preregistration")
        return args[:i] + args[i + 2:]

    # refused before anything loads: a --model that does not exist is never opened
    bare = without_waiver(_train_args(tmp_path / "no-such-model", data, tmp_path / "r", tmp_path / "sr"))
    capsys.readouterr()
    assert T.main(bare) == 2
    assert "must name --preregistration" in capsys.readouterr().err
    assert not (tmp_path / "r" / "training-receipt.json").exists()
    # a gain-only run moves weights too
    assert T.main(without_waiver(_train_args(tmp_path / "no-such-model", data, tmp_path / "r", tmp_path / "sr", "--lr", "0"))) == 2
    # an empty reason and both flags at once are refused
    assert T.main(_train_args(tmp_path / "no-such-model", data, tmp_path / "r", tmp_path / "sr", "--no-preregistration", " ")) == 2
    assert T.main(_train_args(tmp_path / "no-such-model", data, tmp_path / "r", tmp_path / "sr", "--preregistration", str(tmp_path))) == 2
    assert "exclusive" in capsys.readouterr().err
    # the waiver is receipted with its reason
    assert T.main(_train_args(tiny_dir, data, tmp_path / "w", tmp_path / "sw", "--total-steps", "1")) == 0
    rec = json.loads((tmp_path / "w" / "training-receipt.json").read_text())["preregistration"]
    assert rec == {"pinned": False, "waived": True, "reason": WAIVER}
    # a rate cell that moves nothing (lr = 0, gain-lr = 0: the gpu0 lane's P2) runs unpinned and says so
    rate = without_waiver(_train_args(tiny_dir, data, tmp_path / "z", tmp_path / "sz", "--total-steps", "1", "--lr", "0", "--gain-lr", "0"))
    assert T.main(rate) == 0
    rec = json.loads((tmp_path / "z" / "training-receipt.json").read_text())["preregistration"]
    assert rec["pinned"] is False and rec["waived"] is False and "no weight moves" in rec["reason"]
    # the gain frozen: --gain-lr alone moves nothing
    ns = T.parser().parse_args(without_waiver(_train_args(tiny_dir, data, tmp_path / "f", tmp_path / "sf", "--lr", "0", "--freeze-k3-gain")))
    assert not T.run_moves_weights(ns)
    ns = T.parser().parse_args(without_waiver(_train_args(tiny_dir, data, tmp_path / "f", tmp_path / "sf", "--lr", "0")))
    assert T.run_moves_weights(ns)
    # the waiver text is not part of the run config: rewording it does not block a resume
    a = T.parser().parse_args(_train_args(tiny_dir, data, tmp_path / "f", tmp_path / "sf"))
    b = T.parser().parse_args(_train_args(tiny_dir, data, tmp_path / "f", tmp_path / "sf", "--no-preregistration", "other words"))
    assert T.run_config_digest(a) == T.run_config_digest(b)


def test_data_parallel_equals_single_process(tmp_path, tiny_dir, qtext_cache):
    from safetensors.torch import load_file

    data = _data_config(tmp_path, qtext_cache)
    single = _train_args(tiny_dir, data, tmp_path / "one", tmp_path / "s1", "--dump-step-grads", str(tmp_path / "g_one.safetensors"))
    single[single.index("--total-steps") + 1] = "2"
    assert T.main(single) == 0
    dp = list(single)
    dp[dp.index("--micro-batch") + 1] = "1"
    dp[dp.index("--out") + 1] = str(tmp_path / "two")
    dp[dp.index("--safety-dir") + 1] = str(tmp_path / "s2")
    dp[dp.index("--devices") + 1] = "cpu,cpu"
    dp[dp.index("--dump-step-grads") + 1] = str(tmp_path / "g_two.safetensors")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", str(TOOL), *dp],
                            env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    ga = load_file(str(tmp_path / "g_one.safetensors"))
    gb = load_file(str(tmp_path / "g_two.safetensors"))
    assert set(ga) == set(gb) and len(ga) > 0
    grad_rel = max(float((ga[k] - gb[k]).norm() / (ga[k].norm() + 1e-30)) for k in ga)
    print(f"DP vs single, reduced step-1 gradient: max rel-L2 {grad_rel:.3e}")
    assert grad_rel < 1e-5, grad_rel
    a = load_file(str(tmp_path / "one" / "adapters.safetensors"))
    b = load_file(str(tmp_path / "two" / "adapters.safetensors"))
    #: the two runs sum the same gradients in a different order (gloo over two
    #: halves vs one process); fp32 rounding of ~1e-7 is amplified by Adam's
    #: per-element normalisation, so the bound is on rel-L2 per tensor, with
    #: the element-wise max reported beside it
    rel_l2 = max(float((a[k] - b[k]).norm() / (a[k].norm() + 1e-30)) for k in a)
    worst = max(float((a[k] - b[k]).abs().max() / (a[k].abs().max() + 1e-30)) for k in a)
    print(f"DP vs single: max rel-L2 {rel_l2:.3e}, max element rel {worst:.3e}")
    assert rel_l2 < 1e-5 and worst < 1e-3, (rel_l2, worst)
    one = json.loads((tmp_path / "one" / "training-receipt.json").read_text())
    two = json.loads((tmp_path / "two" / "training-receipt.json").read_text())
    for r1, r2 in zip(one["steps"], two["steps"]):
        assert abs(r1["loss"] - r2["loss"]) <= 1e-5 * abs(r1["loss"])


# --------------------------------------------------------------------------
# 1.3.0 review fixes: resume identity and the skip guard
# --------------------------------------------------------------------------


def test_resume_refuses_changed_data_or_base(tmp_path, tiny_dir, qtext_cache, capsys):
    """Review finding (1.2.0): resume compared only a digest of the CLI strings."""

    import shutil

    from safetensors.torch import load_file

    base = tmp_path / "base"
    shutil.copytree(tiny_dir, base)
    data = _data_config(tmp_path, qtext_cache)
    safety = tmp_path / "s"
    assert T.main(_train_args(base, data, tmp_path / "o", safety, "--max-steps-this-run", "2")) == 0
    xtok = tmp_path / "xtok_target.safetensors"
    xtok_bytes, data_text = xtok.read_bytes(), data.read_text()  # restored byte-exact (safetensors metadata order varies)
    # the teacher data rebuilt under the same path, data.json untouched: the data identity moved
    _data_config(tmp_path, qtext_cache, seed=9)
    assert data.read_text() == data_text and xtok.read_bytes() != xtok_bytes
    capsys.readouterr()
    assert T.main(_train_args(base, data, tmp_path / "o", safety, "--resume", "auto")) == 2
    assert "different data identity" in capsys.readouterr().err
    xtok.write_bytes(xtok_bytes)
    # an edited data.json at the same path is a different run config (counted by content)
    edited = json.loads(data_text)
    edited["sources"][1]["tokens"] = 4e5
    data.write_text(json.dumps(edited))
    assert T.main(_train_args(base, data, tmp_path / "o", safety, "--resume", "auto")) == 2
    assert "different run config" in capsys.readouterr().err
    data.write_text(data_text)
    # the base weights rewritten in place: the source identity moved
    weights = base / "model.safetensors"
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert T.main(_train_args(base, data, tmp_path / "o", safety, "--resume", "auto")) == 2
    assert "different source identity" in capsys.readouterr().err
    # ... continued only on the receipted override, and the change is carried in every later receipt
    assert T.main(_train_args(base, data, tmp_path / "o", safety, "--resume", "auto", "--allow-identity-change")) == 0
    receipt = json.loads((tmp_path / "o" / "training-receipt.json").read_text())
    assert receipt["identity_override"] and "source" in receipt["identity_override"]
    assert "files" in receipt["identity_override"]["source"]["differing_keys"]
    assert [s["start_step"] for s in receipt["segments"]] == [0, 2] and receipt["segments"][1]["identity_override"]
    manifest = json.loads((tmp_path / "o" / "adapters_manifest.json").read_text())
    assert len(manifest["segments"]) == 2 and manifest["data_identity_sha256"] == receipt["data_identity_sha256"]
    # an explicit --resume path is digest-checked like --resume auto
    ckpt = safety / "ckpt-00000004"
    state = load_file(str(ckpt / "adapters.safetensors"))
    from safetensors.torch import save_file

    save_file({k: v + 1 for k, v in state.items()}, str(ckpt / "adapters.safetensors"))
    capsys.readouterr()
    assert T.main(_train_args(base, data, tmp_path / "o2", safety, "--resume", str(ckpt), "--allow-identity-change")) == 2
    assert "does not match its COMPLETE.json digest" in capsys.readouterr().err


def _always_nonfinite(monkeypatch, pattern=None):
    """clip_grad_norm_ reports a non-finite norm (all steps, or per `pattern`)."""

    real = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def fake(parameters, max_norm, *a, **kw):
        n = calls["n"]
        calls["n"] += 1
        norm = real(parameters, max_norm, *a, **kw)
        if pattern is None or pattern[n % len(pattern)]:
            return torch.tensor(float("inf"))
        return norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake)


def test_persistent_skips_halt_with_a_checkpoint_and_exit_nonzero(tmp_path, tiny_dir, qtext_cache, monkeypatch):
    """Review finding (1.2.0): a run that skipped every step exited 0 as a success.
    1.3.4: a skipped step is a retried WINDOW (the step counter does not advance), so the halt is at step 0."""

    data = _data_config(tmp_path, qtext_cache)
    args = _train_args(tiny_dir, data, tmp_path / "o", tmp_path / "s", "--max-consecutive-skips", "3")
    args[args.index("--total-steps") + 1] = "6"
    _always_nonfinite(monkeypatch)
    assert T.main(args) == T.EXIT_HALTED
    receipt = json.loads((tmp_path / "o" / "training-receipt.json").read_text())
    assert receipt["verdict"] == "halted_nonfinite" and receipt["step"] == 0 and receipt["applied_steps"] == 0
    assert receipt["skipped_steps"] == 3 and all(r["rewound"] for r in receipt["steps"])
    assert "3 consecutive" in receipt["halt_reason"] and receipt["exit_code"] == T.EXIT_HALTED
    assert not (tmp_path / "o" / "adapters.safetensors").exists()  # nothing exported from a halted run
    state = json.loads((tmp_path / "s" / "ckpt-00000000" / "trainer_state.json").read_text())
    assert state["skip_guard"]["consecutive_skips"] == 3 and state["halt_reason"]
    assert state["sampler"]["global_rows"] == 0  # the rewound rows were never consumed
    # after a ruling the run resumes from the halt checkpoint (guard settings are not part of the config)
    monkeypatch.undo()
    assert T.main(args[:-2] + ["--max-consecutive-skips", "8", "--resume", "auto"]) == 0
    receipt = json.loads((tmp_path / "o" / "training-receipt.json").read_text())
    assert receipt["verdict"] == "completed" and receipt["applied_steps"] == 6 and receipt["step"] == 6


def test_skip_fraction_and_no_applied_step_exit_nonzero(tmp_path, tiny_dir, qtext_cache, monkeypatch):
    data = _data_config(tmp_path, qtext_cache)
    # every second step skipped: never 2 in a row, but 2 of every 4 > 0.4
    args = _train_args(tiny_dir, data, tmp_path / "f", tmp_path / "sf", "--skip-window", "4", "--max-skip-fraction", "0.4")
    _always_nonfinite(monkeypatch, pattern=[True, False])
    assert T.main(args) == T.EXIT_HALTED
    receipt = json.loads((tmp_path / "f" / "training-receipt.json").read_text())
    # 1.3.4: four attempts, two applied steps (each skipped window was retried, not lost)
    assert receipt["verdict"] == "halted_nonfinite" and receipt["step"] == 2 and receipt["applied_steps"] == 2
    assert receipt["steps_this_run"] == 4 and "2 of the last 4" in receipt["halt_reason"]
    monkeypatch.undo()
    # guards off, every step skipped: still not a success, and not an endless retry (HARD_RETRY_LIMIT)
    monkeypatch.setattr(T, "HARD_RETRY_LIMIT", 5)
    _always_nonfinite(monkeypatch)
    args = _train_args(tiny_dir, data, tmp_path / "n", tmp_path / "sn", "--max-consecutive-skips", "0", "--skip-window", "0")
    assert T.main(args) == T.EXIT_HALTED
    receipt = json.loads((tmp_path / "n" / "training-receipt.json").read_text())
    assert receipt["verdict"] == "halted_nonfinite" and receipt["step"] == 0 and receipt["skipped_steps"] == 5
    assert "HARD_RETRY_LIMIT" in receipt["halt_reason"]


# --------------------------------------------------------------------------
# 1.3.3: checkpoint granularity, head chunk sizing, the memory model, memory-probe
# --------------------------------------------------------------------------


def _parity_chunked(tiny_dir, mode, layer_devices, *, ffn, recompute, forced=()):
    src = T.SafetensorsSource(tiny_dir)
    cfg = T.text_config_from_source(src, torch.float32, "sdpa")
    obj = objective_cfg(TEXT_ROWS, w_eos=0.2, w_ul=0.1, head_chunk_positions=7)
    plan = T.build_site_plan(cfg, T.SitePlanConfig(rank=4, k3_rank=2, trunk_ffn_width=32))
    types = list(cfg.layer_types)
    ids, targets = T.synthetic_parity_batch(3, 24, TEXT_ROWS, 6, 8, 5, obj.eos_ids, obj)
    denom = float(3 * 23)
    ref_bank = T.AdapterBank(plan, types, "cpu", 3, b_init_std=0.05, gain_jitter=0.1)
    _, text, head = T.load_reference(tiny_dir, torch.float32, "cpu", "sdpa")
    ref_parts = T.reference_forward_backward(text, head, ref_bank, ids, targets, obj, denominator=denom)
    bank = T.AdapterBank(plan, types, "cpu", 3, b_init_std=0.05, gain_jitter=0.1)
    engine = T.LayerwiseEngine(src, cfg, mode=mode, dtype=torch.float32, layer_devices=layer_devices,
                               head_device=layer_devices[-1], bank=bank, text_vocab_rows=TEXT_ROWS, force_host_hops=forced,
                               ffn_chunk_tokens=ffn, recompute_tokens=recompute)
    parts = engine.forward_backward(ids, targets, obj, denominator=denom)
    bank.zero_grad()  # a second micro-batch must not see stale slice / group grads
    parts = engine.forward_backward(ids, targets, obj, denominator=denom)
    engine.close()
    rows, per_site = T.compare_grads(T.collect_grads(ref_bank), T.collect_grads(bank))
    return ref_parts, parts, per_site, engine


@pytest.mark.parametrize("ffn,recompute", [(5, 24), (7, 0), (0, 48), (100, 100)])
def test_chunked_streamed_engine_matches_stock_autograd(tiny_dir, ffn, recompute):
    """Row groups of one row (recompute 24 = one 24-token row) and FFN slices of 5 tokens cross every
    boundary; every adapter gradient must still be the stock full-load autograd one."""

    ref_parts, parts, per_site, engine = _parity_chunked(tiny_dir, "streamed", ["cpu"], ffn=ffn, recompute=recompute)
    assert engine.split and all(v["pass"] for v in engine.split_contract.values())
    assert abs(parts["total"] - ref_parts["total"]) <= 1e-6 * abs(ref_parts["total"])
    assert len(per_site) >= 10
    for site, entry in per_site.items():
        assert entry["min_ref_norm"] > 0, site
        assert entry["max_rel_l2"] < 1e-5, (site, entry)


def test_chunked_resident_engine_with_a_host_hop_matches_stock_autograd(tiny_dir):
    ref_parts, parts, per_site, _ = _parity_chunked(tiny_dir, "resident", ["cpu"], ffn=5, recompute=24, forced=(2,))
    assert abs(parts["total"] - ref_parts["total"]) <= 1e-6 * abs(ref_parts["total"])
    for site, entry in per_site.items():
        assert entry["max_rel_l2"] < 1e-5, (site, entry)


def test_split_halves_are_the_stock_layer_bit_for_bit(tiny_dir):
    src = T.SafetensorsSource(tiny_dir)
    cfg = T.text_config_from_source(src, torch.float32, "sdpa")
    engine = T.LayerwiseEngine(src, cfg, mode="streamed", dtype=torch.float32, layer_devices=["cpu"], head_device="cpu",
                               bank=None, text_vocab_rows=TEXT_ROWS, ffn_chunk_tokens=4, recompute_tokens=0)
    assert set(engine.split_contract) == {"linear_attention", "full_attention"}
    for entry in engine.split_contract.values():
        assert entry["split_vs_stock_max_abs"] == 0.0 and entry["pass"]
    # and on the REAL weights: the halves of every layer equal the stock call bitwise on CPU fp32
    g = torch.Generator().manual_seed(4)
    x = torch.randn(2, 9, cfg.hidden_size, generator=g)
    for index in range(cfg.num_hidden_layers):
        layer = engine.layer(index)
        ctx = engine._context(2, 9)
        with torch.no_grad():
            stock = engine._run(layer, index, x, ctx)
            split = engine._ffn_half(layer, engine._mixer_half(layer, index, x, ctx))
        assert torch.equal(stock, split), index
    engine.close()


def test_split_contract_refuses_a_layer_whose_halves_differ(tiny_dir, monkeypatch):
    src = T.SafetensorsSource(tiny_dir)
    cfg = T.text_config_from_source(src, torch.float32, "sdpa")
    original = T.LayerwiseEngine._ffn_half
    monkeypatch.setattr(T.LayerwiseEngine, "_ffn_half", staticmethod(lambda layer, h: original(layer, h) * (1.0 + 1e-3)))
    with pytest.raises(T.TrainerError, match="changed the decoder layer"):
        T.LayerwiseEngine(src, cfg, mode="streamed", dtype=torch.float32, layer_devices=["cpu"], head_device="cpu",
                          bank=None, text_vocab_rows=TEXT_ROWS, ffn_chunk_tokens=4)


def test_chunked_forward_equals_the_whole_layer_forward(tiny_dir):
    src = T.SafetensorsSource(tiny_dir)
    cfg = T.text_config_from_source(src, torch.float32, "sdpa")
    ids = torch.randint(0, TEXT_ROWS, (3, 17), generator=torch.Generator().manual_seed(2))
    out = {}
    for knobs in ((0, 0), (5, 17), (1, 1)):
        engine = T.LayerwiseEngine(src, cfg, mode="streamed", dtype=torch.float32, layer_devices=["cpu"], head_device="cpu",
                                   bank=None, text_vocab_rows=TEXT_ROWS, ffn_chunk_tokens=knobs[0], recompute_tokens=knobs[1])
        out[knobs] = engine.forward_hidden(ids, store_boundaries=False)
        engine.close()
    ref = out[(0, 0)]
    for knobs, value in out.items():
        assert torch.allclose(value, ref, rtol=1e-5, atol=1e-6), knobs


def test_row_groups_and_token_slices_cover_every_token_once():
    for batch, length, recompute, ffn in [(8, 4096, 4096, 4096), (1, 32768, 4096, 4096), (4, 4096, 16384, 1000), (3, 7, 0, 0), (5, 3, 4, 2)]:
        groups = T.row_groups(batch, length, recompute)
        covered = [r for g in groups for r in range(g.start, g.stop)]
        assert covered == list(range(batch)), (batch, length, recompute)
        for g in groups:
            rows = g.stop - g.start
            slices = T.token_slices(length, rows, ffn)
            assert [p for s in slices for p in range(s.start, s.stop)] == list(range(length))
            if ffn:
                assert all((s.stop - s.start) * rows <= max(ffn, rows) for s in slices)
    assert len(T.row_groups(8, 4096, 4096)) == 8 and len(T.row_groups(1, 32768, 4096)) == 1
    assert len(T.token_slices(32768, 1, 4096)) == 8


def test_head_chunk_is_sized_from_the_text_rows():
    assert T.head_chunk_positions_for(248_320) == 256  # 256 MiB per fp32 [n, 248,320]: 1.3.2 used 1,024 (0.95 GiB per copy)
    assert T.head_chunk_positions_for(248_320, budget_mib=1024) == 1024
    assert T.head_chunk_positions_for(248_320, explicit=99) == 99
    assert T.head_chunk_positions_for(10_000_000) == 16  # floor
    assert T.head_chunk_positions_for(TEXT_ROWS) % 64 == 0


def omni1_text_config(layers: int = 64):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    types = (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)
    return Qwen3_5TextConfig(
        vocab_size=248_320, hidden_size=5120, intermediate_size=16_640, num_hidden_layers=layers, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_num_key_heads=16, linear_num_value_heads=48, layer_types=types, tie_word_embeddings=False,
    )


def test_memory_model_reproduces_the_p2_oom_and_clears_it():
    """S0 P2 (2026-10-13): both cells OOM on a 23.56 GiB 3090 at 32,768 tokens per micro-batch in the
    whole-layer recompute.  The model must put the 1.3.2 path over the card and the 1.3.3 4k cells under it."""

    cfg = omni1_text_config()
    common = dict(dtype_bytes=2, text_rows=248_320, trunk_ffn_width=4352, trainable_parameters=252_444_672, kernel="fla",
                  head_positions=256)
    card = 23.56 * 2**30
    old_a = T.memory_model(cfg, batch=8, length=4096, recompute_tokens=0, ffn_chunk_tokens=0, **{**common, "head_positions": 1024})
    old_b = T.memory_model(cfg, batch=1, length=32768, recompute_tokens=0, ffn_chunk_tokens=0, **{**common, "head_positions": 1024})
    assert old_a["peak_bytes"] > card and old_b["peak_bytes"] > card
    assert old_a["peak_phase"].startswith("backward")
    new_a = T.memory_model(cfg, batch=8, length=4096, recompute_tokens=4096, ffn_chunk_tokens=4096, **common)
    new_mb4 = T.memory_model(cfg, batch=4, length=4096, recompute_tokens=4096, ffn_chunk_tokens=4096, **common)
    assert new_a["peak_bytes"] < 0.9 * card and new_mb4["peak_bytes"] < 0.9 * card
    assert new_a["group_tokens"] == 4096 and new_a["ffn_slice_tokens"] == 4096
    # the static set is the head + two layer buffers + adapters, grads, AdamW moments
    static = new_a["static_bytes"]
    assert static["head"] == 248_320 * 5120 * 2 and static["adamw_moments_fp32"] == 252_444_672 * 8
    # activation terms are linear in the tokens a phase sees
    s1 = T.memory_model(cfg, batch=1, length=32768, recompute_tokens=4096, ffn_chunk_tokens=2048, **common)
    s2 = T.memory_model(cfg, batch=1, length=32768, recompute_tokens=4096, ffn_chunk_tokens=4096, **common)
    d = s2["phases_bytes"]["backward.ffn.full_attention"] - s1["phases_bytes"]["backward.ffn.full_attention"]
    assert abs(d - 2048 * s1["per_token_bytes"]["ffn_backward"]) <= 2


def test_synthetic_source_serves_the_stock_names(tiny_dir):
    src = T.SyntheticSource(tiny_dir, intermediate_size=40, layer_types=["linear_attention", "full_attention"], seed=1)
    cfg = T.text_config_from_source(src, torch.float32, "sdpa")
    assert cfg.intermediate_size == 40 and list(cfg.layer_types) == ["linear_attention", "full_attention"]
    assert src.has("model.layers.1.self_attn.q_proj.weight") and not src.has("model.layers.0.self_attn.q_proj.weight")
    a = src.get("model.layers.0.mlp.up_proj.weight")
    assert tuple(a.shape) == (40, cfg.hidden_size) and torch.equal(a, src.get("model.layers.0.mlp.up_proj.weight"))
    assert src.identity()["kind"] == "synthetic"


def test_memory_probe_runs_on_cpu_on_real_and_synthetic_sources(tiny_dir, tmp_path):
    base = ["--kernels", "torch", "--dtype", "float32", "--device", "cpu", "--trunk-ffn-width", "32", "--lora-rank", "4",
            "--k3-lora-rank", "2", "--text-vocab-rows", str(TEXT_ROWS), "--seq-len", "20", "--batch", "2", "--eos-ids", "7",
            "--ffn-chunk-tokens", "6", "--recompute-tokens", "20"]
    assert T.main(["memory-probe", "--model", str(tiny_dir), *base, "--receipt", str(tmp_path / "a.json")]) == 0
    a = json.loads((tmp_path / "a.json").read_text())
    assert a["oom"] is None and len(a["step_seconds"]) == 2 and a["grads_finite"]
    assert a["predicted"]["group_tokens"] == 20 and a["predicted"]["ffn_slice_tokens"] == 6
    assert a["measured"]["split_contract"]["linear_attention"]["pass"]
    assert T.main(["memory-probe", "--synthetic-config", str(tiny_dir), "--synthetic-ffn", "48", *base,
                   "--receipt", str(tmp_path / "b.json")]) == 0
    b = json.loads((tmp_path / "b.json").read_text())
    assert b["source_identity"]["kind"] == "synthetic" and b["geometry"]["layer_types"] == ["linear_attention", "full_attention"]


def test_train_receipt_carries_the_memory_model_and_granularity_is_not_config(tmp_path, tiny_dir, qtext_cache):
    data = _data_config(tmp_path, qtext_cache)
    assert T.main(_train_args(tiny_dir, data, tmp_path / "a", tmp_path / "sa", "--ffn-chunk-tokens", "7",
                              "--recompute-tokens", "20", "--max-steps-this-run", "2")) == 0
    receipt = json.loads((tmp_path / "a" / "training-receipt.json").read_text())
    assert receipt["memory_engine"]["ffn_chunk_tokens"] == 7 and receipt["memory_model"]["ffn_slice_tokens"] == 7
    assert receipt["memory_model"]["group_tokens"] == 20
    # a resume may change the memory granularity (prereg section 6: splits of the same sums)
    assert T.main(_train_args(tiny_dir, data, tmp_path / "a", tmp_path / "sa", "--ffn-chunk-tokens", "3",
                              "--recompute-tokens", "0", "--resume", "auto")) == 0
    whole = json.loads((tmp_path / "a" / "training-receipt.json").read_text())
    assert whole["applied_steps"] == 2 and whole["step"] == 4


def test_sdpa_gqa_repeat_is_the_same_arithmetic(tiny_dir, monkeypatch):
    """--sdpa-gqa repeat only changes which SDPA kernel may run (K/V repeated first): the adapter gradients
    stay the stock autograd ones.  The patch is process-global, so it is undone after the test."""

    from transformers.integrations import sdpa_attention as sdpa

    monkeypatch.setattr(sdpa, "use_gqa_in_sdpa", sdpa.use_gqa_in_sdpa)
    assert T.select_sdpa_gqa("repeat") == "repeat"
    assert sdpa.use_gqa_in_sdpa(None, None) is False
    ref_parts, parts, per_site, _ = _parity_chunked(tiny_dir, "streamed", ["cpu"], ffn=5, recompute=24)
    assert abs(parts["total"] - ref_parts["total"]) <= 1e-6 * abs(ref_parts["total"])
    for site, entry in per_site.items():
        assert entry["max_rel_l2"] < 1e-5, (site, entry)
    with pytest.raises(T.TrainerError):
        T.select_sdpa_gqa("never")


def test_memory_check_re_predicts_probe_receipts(tiny_dir, tmp_path, capsys):
    base = ["--kernels", "torch", "--dtype", "float32", "--device", "cpu", "--trunk-ffn-width", "32", "--lora-rank", "4",
            "--k3-lora-rank", "2", "--text-vocab-rows", str(TEXT_ROWS), "--seq-len", "20", "--batch", "2", "--eos-ids", "7",
            "--ffn-chunk-tokens", "6", "--recompute-tokens", "20"]
    assert T.main(["memory-probe", "--synthetic-config", str(tiny_dir), "--synthetic-ffn", "48", *base,
                   "--receipt", str(tmp_path / "p.json")]) == 0
    receipt = json.loads((tmp_path / "p.json").read_text())
    # a CPU probe measures no device phase; give it two measured phases to re-predict
    receipt["measured"]["phases_peak_allocated_gib"] = {"head": 0.01, "backward.ffn.full_attention": 0.02}
    (tmp_path / "p.json").write_text(json.dumps(receipt))
    assert T.main(["memory-check", "--receipts", str(tmp_path / "p.json"), "--receipt", str(tmp_path / "c.json")]) == 0
    out = json.loads((tmp_path / "c.json").read_text())
    assert {r["phase"] for r in out["rows"]} == {"head", "backward.ffn.full_attention"}
    assert all(r["predicted_gib"] is not None and r["ratio"] for r in out["rows"]) and out["worst"] is not None
    assert "MEMORY-CHECK 2 phases" in capsys.readouterr().out


# --------------------------------------------------------------------------
# 1.3.4 (card D2-T1): assistant_only, exclude_rows, gt / all masks, per-source w_ul, doc-boundary UL, mixed widths,
# raw shard verification and the bf16 mass interval (D2-T5), the non-finite rewind, deferred sources
# --------------------------------------------------------------------------


def build_cache_for(work: Path, teacher_dir: Path, tensors: dict, tag: str) -> Path:
    """A sealed tiny Q38 cache over a corpus written from `tensors` by the real qtext_teacher_cache.py."""

    from safetensors.torch import save_file

    work.mkdir(parents=True, exist_ok=True)
    corpus = work / f"{tag}.safetensors"
    save_file(tensors, str(corpus))
    q = T.load_peer_module("qtext_teacher_cache")
    rc = q.main(["build-text", "--model-dir", str(teacher_dir), "--teacher", "tiny-t", "--corpus", str(corpus),
                 "--device", "cpu", "--dtype", "float32", "--linear-attn-kernel", "torch", "--top-k", "8",
                 "--vocab-rows", str(TEXT_ROWS), "--micro-batch", "2", "--rows-per-shard", "4",
                 "--head-chunk-rows", "64", "--out-root", str(work / "root"), "--teacher-digest", "full"])
    assert rc == 0
    return work / "root" / "tiny-t" / hashlib.sha256(corpus.read_bytes()).hexdigest()


def corpus_tensors(tmp: Path, rows: int = 6, length: int = 20, seed: int = 7) -> dict:
    from safetensors.torch import load_file

    return load_file(str(write_corpus(tmp / f"src-{rows}-{length}-{seed}.safetensors", rows=rows, length=length, seed=seed)))


def mask_file(path: Path, kind: str, corpus_sha256: str, spans: list[list[int]]) -> Path:
    path.write_text(json.dumps({"schema": T.MASK_SCHEMA, "kind": kind, "corpora": {corpus_sha256: {"spans": spans}}}))
    return path


def exclusion_file(path: Path, corpus_sha256: str, rows: list[int]) -> Path:
    path.write_text(json.dumps({"schema": T.EXCLUDE_SCHEMA, "corpora": {corpus_sha256: {"rows": rows}}}))
    return path


def text_config(tmp_path: Path, cache: Path, **extra) -> Path:
    tag = hashlib.sha256(json.dumps([str(cache), extra], sort_keys=True).encode()).hexdigest()[:8]
    path = tmp_path / f"data-{tag}.json"
    path.write_text(json.dumps({"sources": [{"name": "q38", "kind": "text", "qtext_cache": str(cache), **extra}]}))
    return path


def test_assistant_only_with_an_all_ones_mask_is_bitwise_the_legacy_path(tmp_path, tiny_dir, teacher_dir, qtext_cache):
    """The plan's equivalence test: every position supervised == the 1.3.3 objective, bit for bit (targets AND a
    trained adapter).  The masked corpus carries `assistant_mask` (which qtext_teacher_cache ignores), so its cache
    stores exactly the legacy positions."""

    from safetensors.torch import load_file

    t = corpus_tensors(tmp_path)
    owned = torch.arange(20).unsqueeze(0) < t["lengths"].unsqueeze(1)  # every real token is "assistant"
    masked_cache = build_cache_for(tmp_path / "m", teacher_dir, {**t, "assistant_mask": owned}, "masked")
    legacy = T.QTextCache(qtext_cache)
    masked = T.QTextCache(masked_cache)
    assert masked.corpus_target_mask is not None and masked.target_mask_key == "assistant_mask"
    cfg = objective_cfg(TEXT_ROWS)
    a = [T.DataSource("t", "text", legacy, row_ids=sorted(legacy.index))]
    b = [T.DataSource("t", "text", masked, row_ids=sorted(masked.index), supervision="assistant_only")]
    ida, ta = T.build_batch(a, [(0, 1), (0, 3)], cfg)
    idb, tb = T.build_batch(b, [(0, 1), (0, 3)], cfg)
    assert torch.equal(ida, idb) and set(ta) == set(tb)
    for k in ta:
        assert torch.equal(ta[k], tb[k]), k
    data_a = text_config(tmp_path, qtext_cache)
    data_b = text_config(tmp_path, masked_cache, supervision="assistant_only")
    for data, out in ((data_a, "la"), (data_b, "lb")):
        args = _train_args(tiny_dir, data, tmp_path / out, tmp_path / f"s{out}")
        args[args.index("--total-steps") + 1] = "2"
        assert T.main(args) == 0
    wa = load_file(str(tmp_path / "la" / "adapters.safetensors"))
    wb = load_file(str(tmp_path / "lb" / "adapters.safetensors"))
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k
    ra = json.loads((tmp_path / "la" / "training-receipt.json").read_text())
    rb = json.loads((tmp_path / "lb" / "training-receipt.json").read_text())
    assert rb["data_identity"]["sources"][0]["supervision"] == "assistant_only"
    assert [s["loss"] for s in ra["steps"]] == [s["loss"] for s in rb["steps"]]


def test_assistant_only_trains_only_on_assistant_targets(tmp_path, teacher_dir):
    t = corpus_tensors(tmp_path)
    target = torch.zeros(6, 20, dtype=torch.bool)
    target[:, 12:16] = True  # tokens 12..15 are the assistant's
    cache = T.QTextCache(build_cache_for(tmp_path / "tm", teacher_dir, {**t, "target_mask": target}, "tm"))
    src = [T.DataSource("tool", "text", cache, row_ids=sorted(cache.index), supervision="assistant_only", w_ul=0.0)]
    cfg = objective_cfg(TEXT_ROWS, w_ul=0.1)
    ids, tg = T.build_batch(src, [(0, 0), (0, 2)], cfg)
    on = torch.zeros(2, 20, dtype=torch.bool)
    on[:, 11:15] = True  # positions 11..14 predict tokens 12..15
    assert torch.equal(tg["q_w"].view(2, 20) > 0, on)  # the cache stored only these (target_mask shifted)
    assert torch.equal(tg["ce_w"].view(2, 20) > 0, on)
    assert bool((tg["next_ids"].view(2, 20)[~on] == -1).all())
    assert "ul_ids" not in tg or bool((tg["ul_ids"] == -1).all())  # w_ul 0 on the tool lane
    # a corpus without any assistant mask cannot be assistant_only
    plain = build_cache_for(tmp_path / "nm", teacher_dir, t, "nomask")
    with pytest.raises(T.TrainerError, match="assistant_only needs"):
        T.load_data_config(text_config(tmp_path, plain, supervision="assistant_only"))


def test_all_mask_everywhere_gives_zero_gradient_and_the_unmasked_control_does_not(tmp_path, tiny_dir, qtext_cache):
    from safetensors.torch import load_file

    sha = corpus_sha(qtext_cache)
    everything = mask_file(tmp_path / "all.json", "all", sha, [[r, 0, 20] for r in range(6)])
    data = text_config(tmp_path, qtext_cache, all_mask=str(everything))
    cfg = objective_cfg(TEXT_ROWS)
    sources, identity = T.load_data_config(data)
    ids, tg = T.build_batch(sources, [(0, 0), (0, 1)], cfg)
    assert float(tg["q_w"].sum()) == 0 and float(tg["ce_w"].sum()) == 0 and float(tg["eos_w"].sum()) == 0
    assert bool((tg["ul_ids"] == -1).all()) and identity["sources"][0]["all_mask"]["positions"] == 120
    for name, config in (("masked", data), ("control", text_config(tmp_path, qtext_cache))):
        args = _train_args(tiny_dir, config, tmp_path / name, tmp_path / f"s{name}", "--dump-step-grads", str(tmp_path / f"g-{name}.safetensors"))
        args[args.index("--total-steps") + 1] = "1"
        assert T.main(args) == 0
    masked = load_file(str(tmp_path / "g-masked.safetensors"))
    control = load_file(str(tmp_path / "g-control.safetensors"))
    assert masked and all(float(v.abs().max()) == 0.0 for v in masked.values())
    assert any(float(v.abs().max()) > 0.0 for v in control.values())  # negative control: the same run unmasked moves
    receipt = json.loads((tmp_path / "masked" / "training-receipt.json").read_text())
    assert receipt["steps"][0]["positions"] == {"q38_positions": 0, "k3_positions": 0, "ce_positions": 0, "eos_positions": 0, "ul_candidates": 0}
    assert receipt["steps"][0]["loss"] == 0.0
    control_receipt = json.loads((tmp_path / "control" / "training-receipt.json").read_text())
    assert control_receipt["steps"][0]["positions"]["q38_positions"] > 0 and control_receipt["steps"][0]["positions"]["ce_positions"] > 0


def test_gt_mask_turns_off_ce_eos_ul_and_keeps_kd(tmp_path, qtext_cache):
    sha = corpus_sha(qtext_cache)
    gt = mask_file(tmp_path / "gt.json", "gt", sha, [[0, 5, 12]])
    sources, _ = T.load_data_config(text_config(tmp_path, qtext_cache, gt_mask=str(gt)))
    cfg = objective_cfg(TEXT_ROWS)
    plain, _ = T.load_data_config(text_config(tmp_path, qtext_cache))
    _, tg = T.build_batch(sources, [(0, 0)], cfg)
    _, tp = T.build_batch(plain, [(0, 0)], cfg)
    assert torch.equal(tg["q_w"], tp["q_w"])  # KL kept
    off = torch.zeros(20, dtype=torch.bool)
    off[5:12] = True
    assert bool((tg["ce_w"][off] == 0).all()) and torch.equal(tg["ce_w"][~off], tp["ce_w"][~off])
    assert float(tp["eos_w"][8]) > 0 and float(tg["eos_w"][8]) == 0  # the EOS at token 9 is predicted by position 8
    assert bool((tg["ul_ids"].view(20, -1)[off] == -1).all())
    # a gt-mask and an all-mask on the same position: the all-mask wins (KD off too)
    both = mask_file(tmp_path / "all.json", "all", sha, [[0, 8, 10]])
    sources, _ = T.load_data_config(text_config(tmp_path, qtext_cache, gt_mask=str(gt), all_mask=str(both)))
    _, tb = T.build_batch(sources, [(0, 0)], cfg)
    assert float(tb["q_w"][8]) == 0 and float(tb["q_w"][7]) > 0 and float(tb["ce_w"][7]) == 0
    # a mask of the wrong kind, or with spans outside the corpus, is refused
    with pytest.raises(T.TrainerError, match="is a 'gt' mask"):
        T.load_data_config(text_config(tmp_path, qtext_cache, all_mask=str(gt)))
    bad = mask_file(tmp_path / "bad.json", "gt", sha, [[0, 5, 99]])
    with pytest.raises(T.TrainerError, match="outside corpus"):
        T.load_data_config(text_config(tmp_path, qtext_cache, gt_mask=str(bad)))


def test_exclude_rows_are_never_drawn_and_hold_across_resume(tmp_path, tiny_dir, qtext_cache, capsys):
    from safetensors.torch import load_file

    sha = corpus_sha(qtext_cache)
    excl = exclusion_file(tmp_path / "excl.json", sha, [2, 4])
    data = text_config(tmp_path, qtext_cache, exclude_rows=str(excl))
    sources, identity = T.load_data_config(data)
    assert sources[0].row_ids == [0, 1, 3, 5]
    assert identity["sources"][0]["exclude_rows"]["rows_removed_from_this_source"] == 2
    sampler = T.MixtureSampler(sources, 5)
    first = sampler.next_rows(10, 0)
    state = json.loads(json.dumps(sampler.state()))
    resumed = T.MixtureSampler(T.load_data_config(data)[0], 5)
    resumed.load_state(state)
    later = resumed.next_rows(30, 0)
    assert not ({r for _, r in first + later} & {2, 4})
    # exact resume through the trainer with the exclusion in force
    full = _train_args(tiny_dir, data, tmp_path / "full", tmp_path / "sf")
    half = _train_args(tiny_dir, data, tmp_path / "half", tmp_path / "sh", "--max-steps-this-run", "2")
    assert T.main(full) == 0 and T.main(half) == 0
    assert T.main(_train_args(tiny_dir, data, tmp_path / "half", tmp_path / "sh", "--resume", "auto")) == 0
    a, b = load_file(str(tmp_path / "full" / "adapters.safetensors")), load_file(str(tmp_path / "half" / "adapters.safetensors"))
    assert all(torch.equal(a[k], b[k]) for k in a)
    # an exclusion file changed under the same path is new data: the resume is refused
    exclusion_file(excl, sha, [2])
    capsys.readouterr()
    assert T.main(_train_args(tiny_dir, data, tmp_path / "half", tmp_path / "sh", "--resume", "auto")) == 2
    assert "different data identity" in capsys.readouterr().err


def test_masked_objective_matches_the_independent_reference_and_stock_autograd(tiny_dir, qtext_cache, tmp_path):
    """CPU grad-parity arm for the 1.3.4 targets: (1) chunked_objective with masked weights and a per-position UL
    weight == the float64 loop reference; (2) the streamed engine's adapter gradients on a REAL masked batch from
    build_batch == a stock full-load autograd of the same batch."""

    torch.manual_seed(3)
    cfg = objective_cfg(50, w_eos=0.3, w_ul=0.2)
    ids, targets = T.synthetic_parity_batch(2, 40, 50, 6, 8, 4, cfg.eos_ids, cfg)
    off = torch.zeros(80, dtype=torch.bool)
    off[10:30] = True
    targets["q_w"] = torch.where(off, 0.0, targets["q_w"])
    targets["ce_w"] = torch.where(off, 0.0, targets["ce_w"])
    targets["ul_w"] = torch.where(torch.arange(80) < 40, 0.5, 0.05).float()
    hidden = torch.randn(2, 40, 12, dtype=torch.float64, requires_grad=True)
    head = torch.randn(60, 12, dtype=torch.float64)

    def reference(h):
        base = reference_objective(h, head, {k: v for k, v in targets.items() if k not in ("ul_ids", "ul_w")}, cfg, 78.0)
        logq = torch.log_softmax(h.reshape(-1, 12) @ head[:50].T, dim=-1)
        ul = h.new_zeros(())
        for p in range(80):
            for c in targets["ul_ids"][p]:
                if int(c) >= 0:
                    ul = ul + float(targets["ul_w"][p]) * (-torch.log(torch.clamp(1 - logq[p, int(c)].exp(), min=1e-6)))
        return base + ul / 78.0

    ref = reference(hidden)
    ref.backward()
    parts, grad = T.chunked_objective(hidden, head[:50], targets, cfg, denominator=78.0)
    assert abs(parts["total"] - float(ref.detach())) < 1e-10 * max(1.0, abs(float(ref.detach())))
    assert torch.allclose(grad, hidden.grad, atol=1e-11, rtol=1e-9) and parts["ul"] != 0.0
    # (2) engine vs stock autograd
    sha = corpus_sha(qtext_cache)
    gt = mask_file(tmp_path / "gt.json", "gt", sha, [[0, 3, 15], [1, 0, 6]])
    allm = mask_file(tmp_path / "all.json", "all", sha, [[0, 10, 14]])
    sources, _ = T.load_data_config(text_config(tmp_path, qtext_cache, gt_mask=str(gt), all_mask=str(allm), w_ul=0.05))
    obj = objective_cfg(TEXT_ROWS, w_eos=0.2, w_ul=0.1, head_chunk_positions=7)
    ids, targets = T.build_batch(sources, [(0, 0), (0, 1)], obj)
    assert "ul_w" in targets and float(targets["q_w"].view(2, 20)[0, 10:14].sum()) == 0
    src = T.SafetensorsSource(tiny_dir)
    config = T.text_config_from_source(src, torch.float32, "sdpa")
    plan = T.build_site_plan(config, T.SitePlanConfig(rank=4, k3_rank=2, trunk_ffn_width=32))
    types = list(config.layer_types)
    ref_bank = T.AdapterBank(plan, types, "cpu", 3, b_init_std=0.05, gain_jitter=0.1)
    _, text, head_w = T.load_reference(tiny_dir, torch.float32, "cpu", "sdpa")
    ref_parts = T.reference_forward_backward(text, head_w, ref_bank, ids, targets, obj, denominator=38.0)
    bank = T.AdapterBank(plan, types, "cpu", 3, b_init_std=0.05, gain_jitter=0.1)
    engine = T.LayerwiseEngine(src, config, mode="streamed", dtype=torch.float32, layer_devices=["cpu"], head_device="cpu",
                               bank=bank, text_vocab_rows=TEXT_ROWS)
    parts = engine.forward_backward(ids, targets, obj, denominator=38.0)
    engine.close()
    _, per_site = T.compare_grads(T.collect_grads(ref_bank), T.collect_grads(bank))
    assert abs(parts["total"] - ref_parts["total"]) <= 1e-6 * abs(ref_parts["total"])
    assert len(per_site) >= 10
    for site, entry in per_site.items():
        assert entry["max_rel_l2"] < 1e-5, (site, entry)


def test_ul_scan_restarts_at_document_boundaries():
    row = torch.tensor([[1, 2, 3, 9, 1, 2, 4, 1, 2, 5]])
    legacy = T.unlikelihood_candidates(row, 3, 4)
    assert sorted(int(c) for c in legacy[0, 8] if c >= 0) == [3, 4]  # "1 2 ?" seen twice before
    segs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])  # a new document starts at index 4
    split = T.unlikelihood_candidates(row, 3, 4, segs)
    assert sorted(int(c) for c in split[0, 8] if c >= 0) == [4]  # only the same document's "1 2 4"
    assert bool((split[0, 3] == -1).all())  # the last position of a document predicts the next one's token
    one = T.unlikelihood_candidates(row, 3, 4, torch.zeros_like(row))
    assert torch.equal(one, legacy)  # one segment == 1.3.3


def test_rows_of_different_widths_share_a_batch_padded_and_never_targeted(tmp_path, tiny_dir, teacher_dir, qtext_cache, capsys):
    narrow = T.QTextCache(build_cache_for(tmp_path / "n", teacher_dir, corpus_tensors(tmp_path, rows=4, length=12, seed=3), "narrow"))
    wide = T.QTextCache(qtext_cache)
    sources = [T.DataSource("w", "text", wide, row_ids=sorted(wide.index)), T.DataSource("n", "text", narrow, row_ids=sorted(narrow.index))]
    cfg = objective_cfg(TEXT_ROWS)
    with pytest.raises(T.TrainerError, match="need a pad id"):
        T.build_batch(sources, [(0, 0), (1, 0)], cfg)
    ids, tg = T.build_batch(sources, [(0, 0), (1, 0)], cfg, pad_id=7)
    assert ids.shape == (2, 20) and bool((ids[1, 12:] == 7).all())
    assert float(tg["q_w"].view(2, 20)[1, 12:].sum()) == 0 and float(tg["ce_w"].view(2, 20)[1, 11:].sum()) == 0
    ids1, tg1 = T.build_batch(sources, [(1, 0)], cfg)  # alone: its own width, no padding
    assert ids1.shape == (1, 12) and torch.equal(tg1["q_w"], tg["q_w"].view(2, 20)[1, :12])
    data = tmp_path / "mixed.json"
    data.write_text(json.dumps({"sources": [{"name": "w", "kind": "text", "qtext_cache": str(qtext_cache)},
                                            {"name": "n", "kind": "text", "qtext_cache": str(narrow.dir), "budget_unit": "real_tokens", "tokens": 200}]}))
    capsys.readouterr()
    assert T.main(_train_args(tiny_dir, data, tmp_path / "o0", tmp_path / "s0")) == 2  # the 1.3.3 rule without --max-width
    assert "cache seq_len 12" in capsys.readouterr().err
    assert T.main(_train_args(tiny_dir, data, tmp_path / "o", tmp_path / "s", "--max-width", "20", "--pad-id", "7")) == 0
    receipt = json.loads((tmp_path / "o" / "training-receipt.json").read_text())
    assert receipt["widest_source"] == 20 and set(receipt["rows_by_source_applied"]) <= {"w", "n"}
    assert T.main(_train_args(tiny_dir, data, tmp_path / "o2", tmp_path / "s2", "--max-width", "20", "--pad-id", "295")) == 2
    assert "--pad-id inside the text rows" in capsys.readouterr().err


def test_real_tokens_budgets_spend_real_lengths():
    cache = SimpleNamespace(seq_len=10, index={0: ("s", 0, 0, 3), 1: ("s", 0, 0, 7)})
    s = T.DataSource("short", "text", cache, row_ids=[0, 1], token_budget=100, budget_unit="real_tokens")
    sampler = T.MixtureSampler([s], 1)
    picks = sampler.next_rows(4, 0)
    assert sampler.state()["consumed_tokens"]["short"] == sum({0: 3, 1: 7}[r] for _, r in picks) == 20


def test_a_real_id_outside_the_text_rows_is_refused(tmp_path, tiny_dir, teacher_dir, capsys):
    t = corpus_tensors(tmp_path)
    t["input_ids"][2, 5] = 295  # a band row of the 300-row tiny vocab: never a text target
    cache = build_cache_for(tmp_path / "band", teacher_dir, t, "band")
    capsys.readouterr()
    assert T.main(_train_args(tiny_dir, text_config(tmp_path, cache), tmp_path / "o", tmp_path / "s")) == 2
    assert "outside the text rows" in capsys.readouterr().err


def _rewrite_shard_logprobs(cache: Path, edit) -> None:
    """Edit shard-00000's topk_logprobs.bf16 in place and re-seal shard.json + the manifest entry (a test-only
    forgery: every sha matches, so only the content check can refuse it)."""

    shard = cache / "shard-00000"
    record = json.loads((shard / "shard.json").read_text())
    lp = np.fromfile(shard / "topk_logprobs.bf16", dtype="<i2").reshape(record["positions"], record["top_k"])
    values = torch.from_numpy(lp.copy()).view(torch.bfloat16).float()
    edit(values)
    values.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2").tofile(shard / "topk_logprobs.bf16")
    record["files"]["topk_logprobs.bf16"]["sha256"] = hashlib.sha256((shard / "topk_logprobs.bf16").read_bytes()).hexdigest()
    (shard / "shard.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    manifest = json.loads((cache / "manifest.json").read_text())
    manifest["shards"][0]["shard_json_sha256"] = hashlib.sha256((shard / "shard.json").read_bytes()).hexdigest()
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def test_bf16_mass_above_one_is_used_as_stored_and_beyond_the_interval_is_refused(tmp_path, qtext_cache):
    """D2-T5.  Codex measured bf16 top-64 masses up to 1.0033815: such a row is accepted, its stored logprobs reach
    the objective unchanged (no clamp, no rescale) and only the tail bucket is clamped at 0.  A row that no bf16
    rounding of a distribution's top-k can explain is refused at load (negative control), as is a flipped byte."""

    import shutil

    # (1) the objective on a row whose stored mass is above 1
    stored = torch.tensor([[math.log(0.5), math.log(0.5) + 0.004, math.log(1e-4)]]).to(torch.bfloat16).float()
    assert float(stored.exp().sum()) > 1.0 and float(T.bf16_min_mass(stored)[0]) <= 1.0 + T.BF16_MASS_SLACK
    logq = torch.log_softmax(torch.randn(1, 30, dtype=torch.float64), -1)
    ids = torch.tensor([[3, 7, 11]])
    p, lq, p_tail, log_q_tail, valid = T.bucket_terms(logq, ids, stored.double())
    assert torch.equal(p, stored.double().exp()) and float(p_tail[0]) == 0.0  # as stored; tail clamped at 0
    kl = T.bucket_forward_kl(logq, ids, stored.double())
    manual = (stored.double().exp() * (stored.double() - logq.gather(1, ids))).sum()
    assert abs(float(kl[0]) - float(manual)) < 1e-12 and math.isfinite(float(kl[0]))
    # (2) a sealed cache with one position pushed just above 1 is accepted and read back exactly
    ok = tmp_path / "ok"
    shutil.copytree(qtext_cache, ok)

    def nudge(values):
        values[0, 0] = math.log(0.5)
        values[0, 1] = math.log(0.5) + 0.004
        values[0, 2:] = torch.clamp(values[0, 2:], max=math.log(1e-5))

    _rewrite_shard_logprobs(ok, nudge)
    cache = T.QTextCache(ok, verify_corpus=False)
    reading = cache.verify_raw()
    assert reading["stored_mass_max"] > 1.0 and reading["positions_with_stored_mass_above_1"] >= 1
    record = json.loads((ok / "shard-00000" / "shard.json").read_text())
    first = record["row_table"][0]
    pos0 = int(np.fromfile(ok / "shard-00000" / "pos.u32", dtype="<u4")[int(first["offset"])])
    raw = np.fromfile(ok / "shard-00000" / "topk_logprobs.bf16", dtype="<i2").reshape(-1, 8)
    got = cache.row(int(first["row"]))["topk_logprobs"][pos0]
    assert torch.equal(got, torch.from_numpy(raw[int(first["offset"])].copy()).view(torch.bfloat16).float())
    # (3) negative control: a position whose mass is 1.2 is refused, although every sha matches
    bad = tmp_path / "bad"
    shutil.copytree(qtext_cache, bad)

    def corrupt(values):
        values[0, 0] = math.log(0.6)
        values[0, 1] = math.log(0.6)

    _rewrite_shard_logprobs(bad, corrupt)
    with pytest.raises(T.TrainerError, match="not a distribution's top-k"):
        T.QTextCache(bad, verify_corpus=False).verify_raw()
    # (4) a byte changed without re-sealing is refused by the raw shard verification
    flipped = tmp_path / "flip"
    shutil.copytree(qtext_cache, flipped)
    path = flipped / "shard-00000" / "topk_logprobs.bf16"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(T.TrainerError, match="raw shard verification failed"):
        T.QTextCache(flipped, verify_corpus=False).verify_raw()


def test_a_nonfinite_window_is_retried_and_loses_no_rows(tmp_path, tiny_dir, qtext_cache, monkeypatch):
    """1.3.3 advanced the step counter on a skipped step, so its rows were never trained.  1.3.4 retries the window:
    a run whose first attempt is non-finite ends bit-identical to a run without it."""

    from safetensors.torch import load_file

    data = _data_config(tmp_path, qtext_cache)
    clean = _train_args(tiny_dir, data, tmp_path / "clean", tmp_path / "sc")
    clean[clean.index("--total-steps") + 1] = "3"
    assert T.main(clean) == 0
    _always_nonfinite(monkeypatch, pattern=[True, False, False, False])
    hit = _train_args(tiny_dir, data, tmp_path / "hit", tmp_path / "sh")
    hit[hit.index("--total-steps") + 1] = "3"
    assert T.main(hit) == 0
    monkeypatch.undo()
    receipt = json.loads((tmp_path / "hit" / "training-receipt.json").read_text())
    assert receipt["steps_this_run"] == 4 and receipt["applied_steps"] == 3 and receipt["steps"][0]["rewound"]
    assert receipt["steps"][0]["rows_by_source"] == receipt["steps"][1]["rows_by_source"]
    a, b = load_file(str(tmp_path / "clean" / "adapters.safetensors")), load_file(str(tmp_path / "hit" / "adapters.safetensors"))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_a_deferred_source_waits_for_its_seal_then_joins_exactly(tmp_path, tiny_dir, qtext_cache):
    """K3 rows enter D2's last rung under the hard gate "K3 cache sealed before the first K3 row": a deferred source
    is not loaded before its threshold, the run checkpoints and exits EXIT_DEFERRED while the marker is absent, and
    resumes into exactly the run that had the marker from the start."""

    from safetensors.torch import load_file

    xtok = write_xtok(tmp_path / "xtok_target.safetensors", "corpus", {0: [1, 4, 9], 3: [2, 6, 12], 5: [0, 10]},
                      corpus_sha256=corpus_sha(qtext_cache))
    marker = tmp_path / "k3.sealed"
    data = tmp_path / "deferred.json"
    data.write_text(json.dumps({"sources": [
        {"name": "q38", "kind": "text", "qtext_cache": str(qtext_cache), "tokens": 1e3},
        {"name": "k3rows", "kind": "k3", "qtext_cache": str(qtext_cache), "xtok": str(xtok), "tokens": 5e5,
         "defer_until_tokens": 80, "sealed_marker": str(marker)}]}))
    sources, identity = T.load_data_config(data)
    assert not sources[1].loaded and identity["sources"][1]["deferred"]["loaded"] is False
    assert T.main(_train_args(tiny_dir, data, tmp_path / "w", tmp_path / "sw")) == T.EXIT_DEFERRED
    receipt = json.loads((tmp_path / "w" / "training-receipt.json").read_text())
    assert receipt["verdict"] == "waiting_deferred_source" and receipt["step"] == 2 and receipt["tokens_seen"] == 80
    assert "k3rows" not in receipt["rows_by_source_applied"] and "does not exist yet" in receipt["deferred_wait"]
    marker.write_text("sealed\n")
    assert T.main(_train_args(tiny_dir, data, tmp_path / "w", tmp_path / "sw", "--resume", "auto")) == 0
    resumed = json.loads((tmp_path / "w" / "training-receipt.json").read_text())
    assert resumed["verdict"] == "completed" and resumed["step"] == 4
    assert resumed["segments"][-1]["deferred_loaded"][0]["name"] == "k3rows"
    assert any(r["k3"] > 0 for r in resumed["steps"])  # the K3 term ran after the seal
    assert T.main(_train_args(tiny_dir, data, tmp_path / "s", tmp_path / "ss")) == 0  # marker present: loads in the loop
    a, b = load_file(str(tmp_path / "w" / "adapters.safetensors")), load_file(str(tmp_path / "s" / "adapters.safetensors"))
    assert all(torch.equal(a[k], b[k]) for k in a)
    # a resume whose deferred source cannot be re-loaded (the seal withdrawn) is refused, not silently re-planned
    marker.unlink()
    assert T.main(_train_args(tiny_dir, data, tmp_path / "w", tmp_path / "sw", "--resume", "auto")) == 2


def test_data_check_loads_like_train_and_plans_rows(tmp_path, qtext_cache, capsys):
    sha = corpus_sha(qtext_cache)
    excl = exclusion_file(tmp_path / "excl.json", sha, [2])
    data = tmp_path / "dc.json"
    data.write_text(json.dumps({"sources": [
        {"name": "q38", "kind": "text", "qtext_cache": str(qtext_cache), "tokens": 200, "exclude_rows": str(excl)},
        {"name": "later", "kind": "text", "qtext_cache": str(qtext_cache), "tokens": 100, "budget_unit": "real_tokens",
         "defer_until_tokens": 1e6, "sealed_marker": str(tmp_path / "nothing")}]}))
    assert T.main(["data-check", "--data-config", str(data), "--seq-len", "20", "--text-vocab-rows", str(TEXT_ROWS),
                   "--rows-per-step", "4", "--receipt", str(tmp_path / "dc_receipt.json")]) == 0
    out = capsys.readouterr().out
    assert "DATA-CHECK OK sources 2 (loaded 1) planned rows 10 suggested_total_steps 3" in out
    receipt = json.loads((tmp_path / "dc_receipt.json").read_text())
    q = receipt["plan"][0]
    assert q["rows"] == 5 and q["excluded"] == 1 and q["raw_verify"]["shards_verified"] == 2 and q["raw_verify"]["stored_mass_max"] <= 1.0 + T.BF16_MASS_SLACK
    assert receipt["plan"][1]["loaded"] is False
    # a subset, and a name that is not in the config
    assert T.main(["data-check", "--data-config", str(data), "--seq-len", "20", "--text-vocab-rows", str(TEXT_ROWS), "--sources", "q38"]) == 0
    assert T.main(["data-check", "--data-config", str(data), "--seq-len", "20", "--sources", "nope"]) == 2


def test_count_real_tokens_counts_the_drawn_rows(tmp_path, tiny_dir, qtext_cache):
    data = text_config(tmp_path, qtext_cache)
    args = _train_args(tiny_dir, data, tmp_path / "o", tmp_path / "s", "--count-real-tokens")
    assert T.main(args) == 0
    receipt = json.loads((tmp_path / "o" / "training-receipt.json").read_text())
    assert receipt["tokens_seen"] == sum(r["real_tokens"] for r in receipt["steps"])
    assert receipt["tokens_seen"] < 4 * 2 * 20 or all(r["real_tokens"] == 40 for r in receipt["steps"])  # row 1 has 17 real tokens
    nominal = json.loads((tmp_path / "o" / "training-receipt.json").read_text())["tokens_per_step"]
    assert nominal == 40
