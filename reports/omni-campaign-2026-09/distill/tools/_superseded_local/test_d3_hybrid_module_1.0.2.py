"""Tests for d3_hybrid_module.py (E1, the D3 hybrid).  CPU only; run with CUDA hidden.

    OC=/home/cereal/omni-campaign; EF=$OC/import/rq27b/tree_ef1ffb9
    CUDA_VISIBLE_DEVICES= CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src:$OC/distill/tools \\
        /home/cereal/model-surgery-lab/venv/bin/python -m pytest -q test_d3_hybrid_module.py

The central fixture is a tiny OMNI-shaped Chimera child: four text layers imported from a random
qwen3_5 state by the REAL `chimera_omni.importing.qwen35.import_qwen35_state`, three real
`ChartedSourceBlock`s (ACE-like, H3-like, HY3D-like) inserted after text layer 1, a composite
manifest with placements / trunk map / interface bands.  The hybrid is checked against
`ChimeraOmniModel.forward` on that child, the streamed D3b gradients against one autograd graph,
and every sub-stage's loss against going down.  The P1 reader (>= 1.0.5) and the trainer are the
real peer tools from this directory.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_SRC = os.environ.get("CHIMERA_OMNI_SRC", "")
if _SRC:
    sys.path.insert(0, str(Path(_SRC) / "src"))

import d3_hybrid_module as D  # noqa: E402

T = D.trainer_module()
T.select_linear_attention_kernels("torch")


# --------------------------------------------------------------------------
# pure pieces
# --------------------------------------------------------------------------


def test_tool_version_and_p1_gate(monkeypatch) -> None:
    assert D.TOOL_VERSION.count(".") == 2 and tuple(map(int, D.TOOL_VERSION.split("."))) >= (1, 0, 2)
    p1 = D.p1_module()
    assert hasattr(p1.Qwen35LayoutReader, "extra")
    monkeypatch.setattr(p1, "TOOL_VERSION", "p1_qwen35_reader 1.0.4")
    with pytest.raises(D.D3Error, match="no charted-block support"):
        D.p1_module()


def test_lazy_row_adam_is_dense_adam_when_every_row_steps() -> None:
    g = torch.Generator().manual_seed(0)
    table = torch.randn(6, 5, generator=g)
    dense = torch.nn.Parameter(table.clone())
    opt = torch.optim.Adam([dense], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
    lazy = D.LazyRowAdam(table.clone(), lr=0.05)
    for _ in range(7):
        grad = torch.randn(6, 5, generator=g)
        dense.grad = grad.clone()
        opt.step()
        lazy.step(torch.arange(6), grad)
    assert torch.allclose(lazy.table, dense.detach(), rtol=0, atol=2e-6)


def test_lazy_row_adam_touches_only_named_rows_with_own_step_count() -> None:
    table = torch.zeros(4, 3)
    lazy = D.LazyRowAdam(table, lr=0.1)
    lazy.step(torch.tensor([1, 3]), torch.ones(2, 3))
    lazy.step(torch.tensor([3]), torch.ones(1, 3))
    assert torch.equal(table[0], torch.zeros(3)) and torch.equal(table[2], torch.zeros(3))
    assert lazy.steps.tolist() == [0, 1, 0, 2]
    # first Adam step of a row moves it by exactly lr (bias-corrected m/sqrt(v) = 1)
    assert torch.allclose(table[1], torch.full((3,), -0.1), atol=1e-6)
    with pytest.raises(D.D3Error, match="unique"):
        lazy.step(torch.tensor([0, 0]), torch.ones(2, 3))
    with pytest.raises(D.D3Error, match="non-finite"):
        lazy.step(torch.tensor([0]), torch.full((1, 3), float("nan")))


def test_unique_row_grads_sums_repeats() -> None:
    rows, grads = D.unique_row_grads(torch.tensor([2, 0, 2]), torch.tensor([[1.0], [2.0], [3.0]]))
    assert rows.tolist() == [0, 2] and grads.flatten().tolist() == [2.0, 4.0]


def test_slice_delta_starts_as_identity_and_only_its_ranges_get_gradient() -> None:
    base = torch.randn(6, 4)
    rows = D.SliceDelta("w", (6, 4), [(1, 2), (4, 6)], 0, "cpu")
    assert torch.equal(rows.full(base), base)
    rows.full(base).sum().backward()
    assert rows.delta.grad.shape == (3, 4) and torch.equal(rows.delta.grad, torch.ones(3, 4))
    cols = D.SliceDelta("w", (6, 4), [(2, 4)], 1, "cpu")
    with torch.no_grad():
        cols.delta.fill_(1.0)
    out = cols.full(base)
    assert torch.equal(out[:, :2], base[:, :2]) and torch.equal(out[:, 2:], base[:, 2:] + 1)
    with pytest.raises(D.D3Error, match="outside"):
        D.SliceDelta("w", (6, 4), [(5, 7)], 0, "cpu")


def test_xd_chunks_split_seal_and_refuse(tmp_path) -> None:
    identity = {"base": "tiny", "lora_below_d": "0" * 64}
    writer = D.XdChunkWriter(tmp_path / "xd", identity=identity, max_chunk_bytes=3 * 4 * 8 * 2)  # three [4, 8] bf16
    samples = {f"s{i}": torch.randn(4, 8) for i in range(7)}
    for key, value in samples.items():
        writer.add(key, value, {"tokens": 4})
    index = writer.close()
    assert len(index["chunks"]) == 3 and index["samples"] == 7
    reader = D.XdChunkReader(tmp_path / "xd", expect_identity=identity)
    for key, value in samples.items():
        assert torch.equal(reader.get(key), value.to(torch.bfloat16))
    with pytest.raises(D.D3Error, match="identity"):
        D.XdChunkReader(tmp_path / "xd", expect_identity={**identity, "base": "other"})
    chunk = tmp_path / "xd" / index["chunks"][0]["file"]
    chunk.write_bytes(chunk.read_bytes()[:-1] + b"\x00")
    with pytest.raises(D.D3Error, match="does not hash"):
        D.XdChunkReader(tmp_path / "xd")
    with pytest.raises(D.D3Error, match="not empty"):
        D.XdChunkWriter(tmp_path / "xd", identity=identity)


def test_teacher_x0_recovers_the_clean_latent_under_each_convention() -> None:
    g = torch.Generator().manual_seed(3)
    x0, noise = torch.randn(5, 4, generator=g), torch.randn(5, 4, generator=g)
    sigma = torch.rand(5, generator=g)
    x_t = (1 - sigma[:, None]) * x0 + sigma[:, None] * noise
    # comfy CONST parents (ACE, QIMG): raw = noise - x0 and x0 = x_t - sigma raw  (SIGN -1)
    assert torch.allclose(D.teacher_x0(x_t, noise - x0, sigma, -1.0), x0, atol=1e-6)
    # HY3D upstream / H3 negated outputs: raw = x0 - noise  (SIGN +1)
    assert torch.allclose(D.teacher_x0(x_t, x0 - noise, sigma, +1.0), x0, atol=1e-6)


def test_flow_x0_loss_reads_only_each_types_emission_band() -> None:
    bands = {4: {"emission": (0, 2)}, 7: {"emission": (2, 5)}}
    types = torch.tensor([[4, 4, 7]])
    target = torch.zeros(1, 3, 6)
    pred = torch.zeros(1, 3, 6)
    pred[..., 5] = 100.0              # outside every band: ignored
    pred[0, 2, 2:5] = 1.0             # the HY3D row is off by 1 on its band
    loss = D.flow_x0_loss(pred, target, types, torch.ones(1, 3), bands)
    assert math.isclose(float(loss), 0.5, rel_tol=1e-6)   # type-balanced: (0 + 1) / 2
    with pytest.raises(D.D3Error, match="no supervised"):
        D.flow_x0_loss(pred, target, types, torch.zeros(1, 3), bands)
    raw = D.flow_x0_loss(pred, target, types, torch.ones(1, 3), bands, param="raw", x_t=torch.zeros(1, 3, 6),
                         sigma=torch.full((1, 3), 0.5), signs={4: -1.0, 7: 1.0})
    assert math.isclose(float(raw), 0.5 * 0.25, rel_tol=1e-6)


def test_band_teacher_logprobs_is_the_band_softmax() -> None:
    g = torch.Generator().manual_seed(5)
    full = torch.randn(3, 20, generator=g)
    band = slice(8, 20)
    lp_full = torch.log_softmax(full, -1)
    top, ids = full[:, band].topk(4, dim=-1)
    got = D.band_teacher_logprobs(top - torch.logsumexp(full, -1, keepdim=True), torch.logsumexp(full, -1),
                                  torch.logsumexp(full[:, band], -1))
    want = torch.log_softmax(full[:, band], -1).gather(1, ids)
    assert torch.allclose(got, want, atol=1e-6)
    assert lp_full.shape == (3, 20)


# --------------------------------------------------------------------------
# the tiny OMNI-shaped child
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    built = D.build_tiny_child(tmp_path_factory.mktemp("d3") / "child")
    lora = T.SitePlanConfig(rank=4, alpha=4.0, k3_rank=0, trunk_ffn_width=D.TINY.TRUNK, gain=False)
    hybrid = D.HybridD3(built["dir"], built["parent_config"],
                        D.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch", lora=lora, lora_seed=3,
                                        pin_boundaries=False))
    return built, hybrid


def test_layout_is_read_from_the_block_schedule(tiny) -> None:
    built, hybrid = tiny
    assert hybrid.reader.text_blocks == [0, 1, 5, 6]
    assert [p.block for p in hybrid.layout] == [2, 3, 4]
    assert {p.after_text_layer for p in hybrid.layout} == {1} and {p.depth for p in hybrid.layout} == {2}
    assert [p.container for p in hybrid.layout] == ["tiny_ace", "tiny_h3", "tiny_hy3d"]


def test_manifest_disagreement_is_refused(tiny, tmp_path) -> None:
    built, hybrid = tiny
    manifest = json.loads((built["dir"] / D.MANIFEST_FILENAME).read_text())
    manifest["blocks"]["placements"][0]["pairing"]["insert_after"] = 0
    with pytest.raises(D.D3Error, match="insert_after"):
        D.unit_layout(hybrid.reader, hybrid.cfg, manifest)
    manifest = json.loads((built["dir"] / D.MANIFEST_FILENAME).read_text())
    manifest["blocks"]["trunk_layer_map"]["2"] = 4
    with pytest.raises(D.D3Error, match="trunk_layer_map"):
        D.unit_layout(hybrid.reader, hybrid.cfg, manifest)


@pytest.mark.parametrize("kind", ["text", "ace", "h3", "hy3d", "ocr"])
def test_hybrid_matches_the_chimera_forward(tiny, kind) -> None:
    built, hybrid = tiny
    seg = D.tiny_segment(hybrid.cfg, kind, seed=17, suffix=3)
    ref = D.chimera_reference(built["model"], seg)
    out = hybrid.forward(seg)
    assert D.rel_l2(out["normed"], ref["hidden"]) < 1e-5
    positions = torch.arange(int(seg["ids"].shape[1]))
    assert D.rel_l2(hybrid.text_logits(out["normed"], positions), ref["logits"][0, :, : hybrid.base_vocab]) < 1e-5
    if ref["emission"] is not None:
        assert D.rel_l2(out["emission"], ref["emission"]) < 1e-5


@pytest.mark.parametrize("kind", ["text", "ace", "h3", "hy3d", "ocr"])
def test_every_unit_writes_exact_zeros_off_its_own_rows(tiny, kind) -> None:
    built, hybrid = tiny
    seg = D.tiny_segment(hybrid.cfg, kind, seed=23, suffix=2)
    out = hybrid.forward(seg)
    context = out["context"]
    for placement in hybrid.layout:
        write = out["writes"][placement.container]
        owned = hybrid.units[placement.container].owned_rows(context["row_types"], context["valid"])
        if owned is None:
            assert torch.equal(write, torch.zeros_like(write))
            continue
        assert torch.equal(write.masked_fill(owned.unsqueeze(-1), 0.0), torch.zeros_like(write))
        assert float(write[owned].norm()) > 0
        assert kind in placement.container


def test_x_d_capture_stops_before_the_units(tiny) -> None:
    built, hybrid = tiny
    seg = D.tiny_segment(hybrid.cfg, "ace", seed=5)
    out = hybrid.forward(seg)
    context = hybrid.prepare(seg)
    x_d = hybrid.run_layers(context["x"], 0, 2, context, units_at_stop=False)
    assert torch.equal(x_d, out["pre_units"][1])
    with_units = hybrid.run_layers(context["x"], 0, 2, context)
    assert not torch.equal(with_units, x_d)


def test_zero_deltas_leave_bridge_and_head_bit_identical(tiny) -> None:
    built, hybrid = tiny
    seg = D.tiny_segment(hybrid.cfg, "hy3d", seed=29)
    before = hybrid.forward(seg)
    hybrid.add_emission_deltas([7])
    hybrid.add_bridge_deltas([7])
    try:
        after = hybrid.forward(seg)
        assert torch.equal(before["normed"], after["normed"]) and torch.equal(before["emission"], after["emission"])
    finally:
        hybrid.head_deltas.clear()
        hybrid.bridge_deltas.clear()


def test_d3b_streamed_gradients_equal_one_autograd_graph(tiny) -> None:
    built, hybrid = tiny
    report = D.d3b_grad_parity(hybrid)
    assert report["pass"], report
    assert report["zero_reference_grads"] == 0 and report["tensors"] > 10
    assert math.isclose(report["loss_streamed"], report["loss_reference"], rel_tol=1e-6)


def test_every_sub_stage_trains(tiny) -> None:
    built, hybrid = tiny
    report = D.synthetic_training(hybrid, steps=30)
    for stage in ("d3a", "d3b", "d3c", "d3d"):
        rep = report[stage]
        assert math.isfinite(rep["last"]) and rep["last"] < 0.7 * rep["first"], (stage, rep)
    assert report["d3c"]["head_rows_updated"] == 30 * 12


def test_band_trainer_moves_only_code_rows_it_saw(tiny) -> None:
    built, hybrid = tiny
    band = D.BandTrainer(hybrid, lr=1e-2, codes=12)
    try:
        before = band.embed_master.clone()
        ids = torch.tensor([[1, 2, 3, hybrid.music_band_begin + 4, hybrid.music_band_begin + 9]])
        top_ids = torch.tensor([[4, 1], [9, 0], [2, 3]])
        lp = torch.log(torch.tensor([[0.5, 0.2], [0.4, 0.1], [0.3, 0.3]]))
        band.step({"ids": ids}, 3, top_ids, lp)
        changed = (band.embed_master != before).any(dim=1).nonzero().flatten().tolist()
        offset = band.offset
        assert changed == [offset + 4, offset + 9]
        assert band.embed_opt.steps.sum() == 2 and band.head_opt.steps.tolist() == [1] * 12
    finally:
        hybrid.band_embed_override = None


def test_band_trainer_grads_do_not_accumulate_across_steps(tiny) -> None:
    """1.0.1: on CPU fp32 `head_rows()` WAS the master, so step n trained on g1 + ... + gn."""
    built, hybrid = tiny
    band = D.BandTrainer(hybrid, lr=0.0, codes=12)
    seen: list[torch.Tensor] = []
    step = band.head_opt.step
    band.head_opt.step = lambda rows, grads: (seen.append(grads.clone()), step(rows, grads))[1]
    try:
        assert band.head_rows() is not band.head_master
        ids = torch.tensor([[1, 2, 3, hybrid.music_band_begin + 4, hybrid.music_band_begin + 9]])
        top_ids = torch.tensor([[4, 1], [9, 0], [2, 3]])
        lp = torch.log(torch.tensor([[0.5, 0.2], [0.4, 0.1], [0.3, 0.3]]))
        band.step({"ids": ids}, 3, top_ids, lp)
        band.step({"ids": ids}, 3, top_ids, lp)
        assert float(seen[0].norm()) > 0
        assert torch.equal(seen[0], seen[1])
        assert band.head_master.grad is None
    finally:
        hybrid.band_embed_override = None


def test_with_text_suffix_appends_text_after_the_media() -> None:
    seg = {"ids": torch.tensor([[1, 2, 3, 9, 9]]), "prompt_tokens": 3, "media_rows": 2, "indices": torch.tensor([[3, 4]]),
           "source_positions": {"h3": torch.ones(1, 5, 3)}}
    out = D.with_text_suffix(seg, 4)
    assert out["ids"].tolist() == [[1, 2, 3, 9, 9, 1, 2, 3, 1]] and out["suffix_tokens"] == 4
    assert out["indices"].tolist() == [[3, 4]] and out["media_rows"] == 2
    assert out["source_positions"]["h3"].shape == (1, 9, 3) and float(out["source_positions"]["h3"][0, 5:].abs().sum()) == 0.0
    assert D.with_text_suffix(seg, 0) is seg


def test_grad_group_norms_refuses_a_dead_path() -> None:
    live, dead, none = torch.zeros(3, requires_grad=True), torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True)
    live.grad, dead.grad = torch.tensor([3.0, 4.0, 0.0]), torch.zeros(2)
    norms, problems = D.grad_group_norms({"unit": [live, dead], "bridge": [dead], "lora": [none]})
    assert norms == {"unit": 5.0, "bridge": 0.0, "lora": None}
    assert problems == ["bridge: gradient norm exactly 0 (dead path)", "lora: no gradient"]


@pytest.mark.parametrize("stage,kind,unit", [("d3b", "ace", "tiny_ace"), ("d3b", "h3", "tiny_h3"), ("d3d", "hy3d", "tiny_hy3d"),
                                             ("d3b", "hy3d", "tiny_hy3d"), ("d3d", "ocr", "tiny_hy3d")])
def test_rate_cell_runs_at_lr_zero_and_changes_nothing(tiny, stage, kind, unit) -> None:
    built, hybrid = tiny
    placement = next(p for p in hybrid.layout if p.container == unit)
    seg = D.tiny_segment(hybrid.cfg, kind, prompt=6, seed=31)   # [prompt | media]: no text after, as S5 segments
    snapshot = {k: v.clone() for k, v in hybrid.units[unit].state_dict().items()}
    cell, refusals = D.rate_cell(hybrid, placement, seg, stage=stage, steps=2)
    if kind == "ocr":
        # the HY3D unit on a segment with no HY3D rows writes nothing: its path is dead and that must refuse
        assert any("unit: " in r for r in refusals) and any("bridge_delta." in r for r in refusals), refusals
        assert not hybrid.head_deltas and not hybrid.bridge_deltas
        return
    assert refusals == [], refusals
    assert cell["params_unchanged_at_lr0"] and cell["trainable_parameters"] > 0 and cell["tok_per_s"] > 0
    assert all(math.isfinite(v) for v in cell["losses"])
    assert all(torch.equal(v, hybrid.units[unit].state_dict()[k]) for k, v in snapshot.items())
    assert not hybrid.head_deltas and not hybrid.bridge_deltas
    # every trainable group got a non-zero gradient on every step (1.0.1's D3d cell had exact zeros)
    assert len(cell["grad_norms"]) == 2
    for norms in cell["grad_norms"]:
        assert "unit" in norms and "lora" in norms and all(v is not None and v > 0 for v in norms.values()), norms
    if stage == "d3d":
        assert {k for k in cell["grad_norms"][0] if k.startswith("bridge_delta.")} == {"bridge_delta.feature_proj.weight",
                                                                                        "bridge_delta.type_embed.weight"}
        media_end = int(seg["ids"].shape[1])
        assert cell["answer_positions"][0] == media_end and cell["answer_positions"][1] == cell["tokens"] - 2
        assert cell["tokens"] == media_end + D.D3D_RATE_SUFFIX


def test_trim_segment_keeps_the_first_rows_of_each_type() -> None:
    seg = {"ids": torch.tensor([[1, 2, 9, 9, 9, 9, 9]]), "prompt_tokens": 2, "media_rows": 5,
           "features": torch.arange(5.0).view(1, 5, 1), "types": torch.tensor([[3, 3, 2, 2, 2]]),
           "positions": torch.zeros(1, 5, 4), "sigma": torch.arange(5.0).view(1, 5), "indices": torch.arange(2, 7).view(1, 5)}
    out = D.trim_segment(seg, 1, 9)
    assert out["types"].tolist() == [[3, 2]] and out["features"].flatten().tolist() == [0.0, 2.0]
    assert out["ids"].tolist() == [[1, 2, 9, 9]] and out["indices"].tolist() == [[2, 3]] and out["trimmed_from"] == 5
    assert D.trim_segment(seg, 0, 9) is seg
