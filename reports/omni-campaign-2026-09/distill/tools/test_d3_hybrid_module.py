"""Tests for d3_hybrid_module.py (E1, the D3 hybrid; 1.1.0).  CPU only; CUDA is hidden by this file.

    OC=/home/cereal/omni-campaign; EF=$OC/import/rq27b/tree_ef1ffb9
    cd $OC/distill/tools && CUDA_VISIBLE_DEVICES= CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src:$OC/distill/tools \\
        nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -q -p no:cacheprovider test_d3_hybrid_module.py

The central fixture is a tiny OMNI-shaped Chimera child: four text layers imported from a random
qwen3_5 state by the REAL `chimera_omni.importing.qwen35.import_qwen35_state`, three real
`ChartedSourceBlock`s (ACE-like, H3-like, HY3D-like) inserted after text layer 1, a composite
manifest with placements / trunk map / interface bands.  The hybrid is checked against
`ChimeraOmniModel.forward` on that child, the streamed D3b gradients against one autograd graph,
and every sub-stage's loss against going down.  The P1 reader (>= 1.0.5) and the trainer are the
real peer tools from this directory.

1.1.0 adds (second half of this file):
  * reference arms against Codex's own code, copied read-only into codex_ref/ (SOURCES.json pins
    the branch HEAD 81f54988 and every file's sha): the velocity loss (random tensors and 3 GENUINE
    H3 replay batches in fixtures/d3_codex_h3_replay, PROVENANCE.json), the D3a refit loss, the D3d
    feature-KD streamed backward, the speech-in targets and prompt-only read scope, the D4
    optimiser union and replay selection;
  * parity arms against 1.0.2 (_superseded/ or _superseded_local/d3_hybrid_module_1.0.2.py) on the
    paths 1.1.0 did not change: forward, x0-endpoint D3b gradients, D3a without the write term,
    D3c steps, D3d steps, the x0 rate cell -- bit for bit;
  * D3b guided velocity moving exactly the intended parameters; D3c reserved rows; D3a declared
    write target; D3d alignment checks; D3e lanes; native state across stages; exact stage resume;
    receipt joins; D4 guards.
"""

from __future__ import annotations

import os

# hide CUDA by mechanism before torch loads (card X-T1), unless a GPU run is explicitly allowed
if os.environ.get("OC_TEST_ALLOW_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

if os.environ.get("OC_TEST_ALLOW_GPU") != "1":
    assert not torch.cuda.is_available(), "CUDA must be hidden for the d3_hybrid_module CPU suite"

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
    assert D.TOOL_VERSION.count(".") == 2 and tuple(map(int, D.TOOL_VERSION.split("."))) >= (1, 1, 0)
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


# ==========================================================================
# 1.1.0
# ==========================================================================

import copy  # noqa: E402
import importlib.util  # noqa: E402

CODEX_REF = HERE / "codex_ref"
H3_FIXTURE = HERE / "fixtures" / "d3_codex_h3_replay" / "codex-h3-native-modality-v3" / "native-h3-v1"
#: OMNI1's interface bands (composite_manifest of OC02-F-Q38-27BU-OMNI1.clone-pre-D1, read 2026-10-14)
OMNI1_BANDS = {4: {"feature": (0, 384), "emission": (0, 128)}, 2: {"feature": (384, 480), "emission": (128, 224)},
               3: {"feature": (480, 512), "emission": (224, 256)}, 1: {"feature": (512, 576), "emission": (256, 320)},
               7: {"feature": (576, 640), "emission": (320, 384)}, 6: {"feature": (640, 2176), "emission": (384, 385)}}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def codex():
    """Codex's modules, read-only copies; every file must still hash like SOURCES.json says."""
    sources = json.loads((CODEX_REF / "SOURCES.json").read_text(encoding="utf-8"))
    assert sources["head"].startswith("81f54988") and sources["edited"] is False
    for name, pin in sources["files"].items():
        assert _sha(CODEX_REF / name) == pin["sha256"], name
    out = {"R": _load("codex_ref_training", CODEX_REF / "training.py"),
           "C": _load("codex_ref_media_cache", CODEX_REF / "media_cache.py"),
           "A": _load("codex_ref_train_audio", CODEX_REF / "train_audio.py"),
           "F": _load("codex_ref_train_refit", CODEX_REF / "train_refit.py"),
           "V": _load("codex_ref_train_vision", CODEX_REF / "train_vision.py")}
    sys.path.insert(0, str(HERE))
    import codex_ref.consolidation as consolidation  # relative import of .corpus
    out["K"] = consolidation
    return out


@pytest.fixture(scope="module")
def v102():
    for d in ("_superseded", "_superseded_local"):
        path = HERE / d / "d3_hybrid_module_1.0.2.py"
        if path.is_file():
            assert _sha(path) == "540fca2d9d976da336c8d24ae5369c5a303ff37367cae54a9bcfc762a7e95197"
            module = _load("d3_hybrid_module_v102", path)
            # 1.0.2 finds its peers (streamed_kd_trainer.py) beside its own file; the archived copy
            # sits in a sub-directory, so point it at this directory's peers -- the SAME trainer module
            # object the 1.1.0 side uses (both arms then share one engine implementation)
            module.HERE = HERE
            assert module.trainer_module() is T
            return module
    raise AssertionError("d3_hybrid_module_1.0.2.py (sha 540fca2d...) missing from _superseded/ and _superseded_local/")


def _fresh(built, seed: int = 3, module=None):
    module = module or D
    lora = T.SitePlanConfig(rank=4, alpha=4.0, k3_rank=0, trunk_ffn_width=D.TINY.TRUNK, gain=False)
    return module.HybridD3(built["dir"], built["parent_config"],
                           module.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch", lora=lora, lora_seed=seed,
                                               pin_boundaries=False))


@pytest.fixture(scope="module")
def child(tmp_path_factory):
    return D.build_tiny_child(tmp_path_factory.mktemp("d3c") / "child")


@pytest.fixture(scope="module")
def child_qvl(tmp_path_factory):
    return D.build_tiny_child(tmp_path_factory.mktemp("d3q") / "child", with_qvl=True)


# -- velocity objective ------------------------------------------------------


def _random_flow(seed: int, *, m: int = 7, f: int = 48):
    g = torch.Generator().manual_seed(seed)
    types = torch.tensor([[4, 4, 2, 2, 3, 7, 7][:m]])
    bands = {t: {"feature": D.TINY_BANDS[t][0], "emission": D.TINY_BANDS[t][1]} for t in (4, 2, 3, 7)}
    return {"pred": torch.randn(1, m, f, generator=g), "target": torch.randn(1, m, f, generator=g),
            "x_t": torch.randn(1, m, f, generator=g), "sigma": torch.rand(1, m, generator=g) * 0.9 + 0.05,
            "mask": torch.tensor([[1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.0][:m]]), "types": types, "bands": bands,
            "channels": torch.rand(1, m, f, generator=g) > 0.3}


@pytest.mark.parametrize("param,target_kind,layout,use_channels", [
    ("x0", "x0", "feature", False), ("x0", "parent_raw", "feature", False), ("x0", "canonical_velocity", "feature", True),
    ("raw", "canonical_velocity", "emission", False), ("raw", "parent_raw", "emission", True),
    ("canonical_velocity", "x0", "emission", False)])
def test_flow_velocity_loss_equals_codex(codex, param, target_kind, layout, use_channels) -> None:
    r = _random_flow(5)
    signs = {4: -1.0, 2: 1.0, 3: 1.0, 7: 1.0}
    kwargs = dict(x_t=r["x_t"], sigma=r["sigma"], signs=signs, channel_mask=r["channels"] if use_channels else None)
    if layout == "feature":
        kwargs["noisy_feature_bands"] = D.TINY_NOISY_FEATURE_BANDS
    ours = D.flow_velocity_loss(r["pred"], r["target"], r["types"], r["mask"], r["bands"], prediction_kind=param,
                                target_kind=target_kind, x_t_layout=layout, **kwargs)
    theirs = codex["R"].flow_velocity_loss(r["pred"], r["target"], r["types"], r["mask"], r["bands"], param=param,
                                           target_kind=target_kind, x_t_layout=layout, **kwargs)
    assert math.isfinite(float(ours)) and abs(float(ours) - float(theirs)) <= 1e-6 * abs(float(theirs))


def test_x0_head_velocity_is_the_sigma_weighted_endpoint_error() -> None:
    r = _random_flow(6)
    x0 = r["target"]
    # an x0 target and an x0 head: v_pred - v_true = (x0 - x0_hat) / sigma on the emission band
    loss = D.flow_velocity_loss(r["pred"], x0, r["types"], torch.ones(1, 7), r["bands"], prediction_kind="x0", target_kind="x0",
                                x_t=r["x_t"], sigma=r["sigma"], noisy_feature_bands=D.TINY_NOISY_FEATURE_BANDS)
    per_type = []
    for t in (2, 3, 4, 7):
        lo, hi = r["bands"][t]["emission"]
        rows = r["types"][0] == t
        err = ((r["pred"][0, rows, lo:hi] - x0[0, rows, lo:hi]) / r["sigma"][0, rows][:, None]).square().mean(-1)
        per_type.append(err.mean())
    assert torch.allclose(loss, torch.stack(per_type).mean(), rtol=1e-6)


def test_velocity_refusals() -> None:
    r = _random_flow(7)
    with pytest.raises(D.D3Error, match="noisy-channel"):
        D.flow_velocity_loss(r["pred"], r["target"], r["types"], r["mask"], r["bands"], x_t=r["x_t"], sigma=r["sigma"],
                             target_kind="x0", noisy_feature_bands={4: (6, 12)})
    with pytest.raises(D.D3Error, match="emission-wide"):
        D.flow_velocity_loss(r["pred"], r["target"], r["types"], r["mask"], r["bands"], x_t=r["x_t"], sigma=r["sigma"],
                             target_kind="x0", noisy_feature_bands={**D.TINY_NOISY_FEATURE_BANDS, 4: (5, 12)})
    with pytest.raises(D.D3Error, match="sign"):
        D.flow_velocity_loss(r["pred"], r["target"], r["types"], r["mask"], r["bands"], prediction_kind="canonical_velocity",
                             target_kind="parent_raw", signs={4: 0.5})
    bad = r["sigma"].clone()
    bad[0, 0] = 0.0
    with pytest.raises(D.D3Error, match="sigma"):
        D.flow_velocity_loss(r["pred"], r["target"], r["types"], r["mask"], r["bands"], x_t=r["x_t"], sigma=bad, target_kind="x0",
                             noisy_feature_bands=D.TINY_NOISY_FEATURE_BANDS)
    with pytest.raises(D.D3Error, match="detached"):
        D.flow_velocity_loss(r["pred"], r["target"].requires_grad_(), r["types"], r["mask"], r["bands"],
                             prediction_kind="canonical_velocity", target_kind="canonical_velocity")


def test_audited_noisy_bands_pick_the_matching_table() -> None:
    assert D.audited_noisy_bands(OMNI1_BANDS) == D.OMNI1_NOISY_FEATURE_BANDS
    tiny = {t: {"feature": v[0], "emission": v[1]} for t, v in D.TINY_BANDS.items()}
    assert D.audited_noisy_bands(tiny) == D.TINY_NOISY_FEATURE_BANDS
    moved = copy.deepcopy(OMNI1_BANDS)
    moved[4]["emission"] = (0, 127)
    with pytest.raises(D.D3Error, match="no audited"):
        D.audited_noisy_bands(moved)
    # the audited json Codex shipped (artifacts/chimera_recovery/2026-10-13/native_noisy_feature_bands.json) is this table
    shipped = {"1": [512, 576], "2": [384, 480], "3": [480, 512], "4": [256, 384], "7": [576, 640]}
    assert {int(k): tuple(v) for k, v in shipped.items()} == D.OMNI1_NOISY_FEATURE_BANDS


def test_tiny_flow_fixture_is_a_genuine_noising(child) -> None:
    for kind in ("ace", "h3", "hy3d"):
        seg = D.tiny_flow_batch(child["cfg"], kind, rows=6, seed=3)
        for r in range(6):
            t = int(seg["types"][0, r])
            elo, ehi = D.TINY_BANDS[t][1]
            flo, fhi = D.TINY_NOISY_FEATURE_BANDS[t]
            s = float(seg["sigma"][0, r])
            x_t = seg["features"][0, r, flo:fhi]
            assert torch.allclose(x_t, (1 - s) * seg["x0"][0, r, elo:ehi] + s * seg["eps"][0, r, elo:ehi], atol=1e-6)
            # the parent-raw teacher recovers x0 with its sign: x0 = x_t + SIGN sigma raw
            assert torch.allclose(D.teacher_x0(x_t, seg["raw"][0, r, elo:ehi], torch.tensor(s), seg["signs"][t]),
                                  seg["x0"][0, r, elo:ehi], atol=1e-6)


def test_velocity_parity_on_three_genuine_h3_replay_batches(codex) -> None:
    """Codex's sealed joint-AV guided batches: compact inflate, receipt binding and the loss, ours vs
    Codex's own adapter, on the receipt's raw head and on this engine's x0 head (rel <= 1e-6)."""
    from safetensors.torch import load_file

    prov = json.loads((H3_FIXTURE.parents[1] / "PROVENANCE.json").read_text(encoding="utf-8"))
    names = sorted(p.name for p in H3_FIXTURE.glob("*.safetensors"))
    assert len(names) == 3
    for name in names:
        tensors = H3_FIXTURE / name
        receipt = json.loads(tensors.with_suffix(".json").read_text(encoding="utf-8"))
        assert _sha(tensors) == receipt["cache_sha256"] == prov["files"][f"codex-h3-native-modality-v3/native-h3-v1/{name}"]
        stored = load_file(str(tensors))
        ours = D.inflate_native_batch(stored, receipt["storage"], width=6144)
        theirs = codex["C"].inflate_native_batch(stored, receipt["storage"])
        assert set(ours) == set(theirs) and all(torch.equal(ours[k], theirs[k]) for k in ours)
        g = torch.Generator().manual_seed(int(receipt["cache_sha256"][:8], 16))
        pred = torch.randn(ours["native_media_targets"].shape, generator=g)
        loss_raw, info = D.bind_native_flow_batch(receipt, ours, OMNI1_BANDS, route="joint_av", flow_param="raw",
                                                 signs={int(k): v for k, v in receipt["parent_signs"].items()})
        ref_raw = codex["R"].ReadyObjectiveAdapter(receipt).bind("joint_av", theirs, OMNI1_BANDS,
                                                                 parent_signs={int(k): v for k, v in receipt["parent_signs"].items()})
        a, b = float(loss_raw(pred)), float(ref_raw(pred))
        assert math.isfinite(a) and abs(a - b) <= 1e-6 * abs(b), (name, a, b)
        loss_x0, info0 = D.bind_native_flow_batch(receipt, ours, OMNI1_BANDS, route="joint_av", flow_param="x0")
        ref_x0 = codex["R"].flow_velocity_loss(pred, theirs["native_media_targets"], theirs["native_media_type_ids"],
                                               theirs["native_media_loss_mask"], OMNI1_BANDS, param="x0", target_kind=receipt["target_kind"],
                                               x_t=theirs["native_media_features"], sigma=theirs["native_media_noise_levels"],
                                               noisy_feature_bands={int(k): tuple(v) for k, v in receipt["noisy_feature_bands"].items()},
                                               channel_mask=theirs["native_media_channel_loss_mask"])
        a, b = float(loss_x0(pred)), float(ref_x0)
        assert math.isfinite(a) and abs(a - b) <= 1e-6 * abs(b), (name, a, b)
        assert info["receipt_prediction_kind"] == "raw" and info0["engine_prediction_kind"] == "x0"
        assert info["identity"]["tensors_sha256"] == receipt["cache_sha256"]


def test_bind_native_flow_batch_refuses_a_moved_allocation() -> None:
    from safetensors.torch import load_file

    tensors = sorted(H3_FIXTURE.glob("*.safetensors"))[0]
    receipt = json.loads(tensors.with_suffix(".json").read_text(encoding="utf-8"))
    batch = D.inflate_native_batch(load_file(str(tensors)), receipt["storage"], width=6144)
    bad = copy.deepcopy(receipt)
    bad["noisy_feature_bands"]["3"] = [481, 513]
    with pytest.raises(D.D3Error, match="digest"):
        D.bind_native_flow_batch(bad, batch, OMNI1_BANDS, route="joint_av")
    bad = copy.deepcopy(receipt)
    bad["no_clean_target_in_inputs"] = False
    with pytest.raises(D.D3Error, match="separation"):
        D.bind_native_flow_batch(bad, batch, OMNI1_BANDS, route="joint_av")
    # a parent-raw HEAD needs the receipt's signs even when the target is canonical velocity
    assert receipt["target_kind"] == "canonical_velocity"
    bad = copy.deepcopy(receipt)
    del bad["parent_signs"]
    with pytest.raises(D.D3Error, match="parent_signs"):
        D.bind_native_flow_batch(bad, batch, OMNI1_BANDS, route="joint_av", flow_param="raw")
    D.bind_native_flow_batch(bad, batch, OMNI1_BANDS, route="joint_av", flow_param="x0")  # the x0 head needs none
    with pytest.raises(D.D3Error, match="differ"):
        D.bind_native_flow_batch(receipt, batch, OMNI1_BANDS, route="joint_av", flow_param="raw", signs={2: 1.0, 3: -1.0})


# -- D3b: the guided-velocity step moves exactly the intended parameters ------------


def test_d3b_guided_velocity_grad_parity(tiny) -> None:
    built, hybrid = tiny
    report = D.d3b_grad_parity(hybrid, objective="guided_velocity")
    assert report["pass"], report
    assert report["zero_reference_grads"] == 0 and report["tensors"] > 10


def test_d3b_guided_velocity_updates_unit_emission_and_upper_lora_only(child) -> None:
    hybrid = _fresh(child)
    context, x_d, depth, loss_fn = D._tiny_d3b_setup(hybrid, seed=12, objective="guided_velocity")
    ace = hybrid.layout[0].container
    others = {p.container: {k: v.clone() for k, v in hybrid.units[p.container].state_dict().items()}
              for p in hybrid.layout if p.container != ace}
    bridge = {k: v.clone() for k, v in hybrid.bridge.state_dict().items()}
    head = {k: v.clone() for k, v in hybrid.media_head.state_dict().items()}
    unit_before = {k: v.clone() for k, v in hybrid.units[ace].state_dict().items()}
    bank_before = {k: v.detach().clone() for k, v in hybrid.bank.tensors.items()}
    deltas = hybrid.add_emission_deltas([4])
    hybrid.freeze_lora_outside(range(depth, hybrid.num_layers))
    trainable = hybrid.unit_parameters([ace]) + [d.delta for d in deltas] + hybrid.lora_parameters(range(depth, hybrid.num_layers))
    opt = torch.optim.AdamW(trainable, lr=1e-2, weight_decay=0.0)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        D.d3b_forward_backward(hybrid, context, x_d, depth, loss_fn)
        opt.step()
    # the unit moved (except its source-frozen coefficients)
    assert any(not torch.equal(unit_before[k], v) for k, v in hybrid.units[ace].state_dict().items())
    # emission delta rows: only ACE's out band rows and type_affine row 4, and they moved
    out_delta, ta_delta = deltas
    assert out_delta.ranges == [tuple(hybrid.bands[4]["emission"])] and ta_delta.ranges == [(4, 5)]
    assert float(out_delta.delta.abs().sum()) > 0 and float(ta_delta.delta.abs().sum()) > 0
    # LoRA: layers >= d moved (B factors leave zero), layers < d bit-unchanged
    moved_upper = [n for n in hybrid.bank.names if int(n.split(".")[1]) >= depth and not torch.equal(bank_before[n], hybrid.bank.tensors[n])]
    assert moved_upper and all(torch.equal(bank_before[n], hybrid.bank.tensors[n]) for n in hybrid.bank.names if int(n.split(".")[1]) < depth)
    # every other unit, the bridge and the frozen head weights bit-unchanged
    for name, state in others.items():
        assert all(torch.equal(state[k], v) for k, v in hybrid.units[name].state_dict().items()), name
    assert all(torch.equal(bridge[k], v) for k, v in hybrid.bridge.state_dict().items())
    assert all(torch.equal(head[k], v) for k, v in hybrid.media_head.state_dict().items())
    hybrid.head_deltas.clear()


def test_a_second_delta_on_one_interface_parameter_is_refused(child) -> None:
    hybrid = _fresh(child)
    hybrid.add_emission_deltas([4])
    with pytest.raises(D.D3Error, match="one delta per interface parameter"):
        hybrid.add_emission_deltas([7])
    hybrid.add_bridge_deltas([6])
    with pytest.raises(D.D3Error, match="one delta per interface parameter"):
        hybrid.add_bridge_deltas([3])
    with pytest.raises(D.D3Error, match="no interface band"):
        _fresh(child).add_bridge_deltas([99])


@pytest.mark.parametrize("objective", ["guided_velocity", "x0_endpoint"])
def test_rate_cell_objectives(tiny, objective) -> None:
    built, hybrid = tiny
    placement = next(p for p in hybrid.layout if p.container == "tiny_ace")
    seg = D.tiny_segment(hybrid.cfg, "ace", prompt=6, seed=31)
    cell, refusals = D.rate_cell(hybrid, placement, seg, stage="d3b", steps=2, flow_objective=objective)
    assert refusals == [] and cell["flow_objective"] == objective and cell["params_unchanged_at_lr0"]
    assert "random rate fixture" in cell["teacher_targets"]


# -- parity with 1.0.2 on the unchanged paths ----------------------------------------


def test_v102_forward_is_bit_identical(child, v102) -> None:
    new, old = _fresh(child), _fresh(child, module=v102)
    for kind in ("text", "ace", "h3", "hy3d", "ocr"):
        seg = D.tiny_segment(child["cfg"], kind, seed=41, suffix=2)
        a, b = new.forward(seg), old.forward(seg)
        assert torch.equal(a["normed"], b["normed"]), kind
        if "emission" in b:
            assert torch.equal(a["emission"], b["emission"]), kind
        assert set(a["writes"]) == set(b["writes"]) and all(torch.equal(a["writes"][k], b["writes"][k]) for k in a["writes"])
    # with (non-zero) interface deltas too
    for h in (new, old):
        for d in h.add_emission_deltas([4, 7]) + h.add_bridge_deltas([4, 7]):
            with torch.no_grad():
                d.delta.copy_(torch.linspace(-0.05, 0.05, d.delta.numel()).view_as(d.delta))
    seg = D.tiny_segment(child["cfg"], "hy3d", seed=43)
    a, b = new.forward(seg), old.forward(seg)
    assert torch.equal(a["normed"], b["normed"]) and torch.equal(a["emission"], b["emission"])


def test_v102_x0_endpoint_d3b_gradients_are_bit_identical(child, v102) -> None:
    new, old = _fresh(child), _fresh(child, module=v102)
    grads = []
    for module, h in ((D, new), (v102, old)):
        ctx, x_d, depth, loss_fn = module._tiny_d3b_setup(h, seed=2)
        deltas = h.add_emission_deltas([4])
        h.freeze_lora_outside(range(depth, h.num_layers))
        ts = h.unit_parameters([h.layout[0].container]) + [d.delta for d in deltas] + h.lora_parameters(range(depth, h.num_layers))
        loss = module.d3b_forward_backward(h, ctx, x_d, depth, loss_fn)
        grads.append((loss, [t.grad.clone() for t in ts]))
    assert grads[0][0] == grads[1][0]
    assert len(grads[0][1]) == len(grads[1][1]) and all(torch.equal(a, b) for a, b in zip(grads[0][1], grads[1][1]))


def test_v102_d3a_d3c_d3d_paths_are_bit_identical(child, v102) -> None:
    new, old = _fresh(child), _fresh(child, module=v102)
    # D3a without a write term
    g = torch.Generator().manual_seed(4)
    host = torch.randn(9, new.hidden, generator=g)
    unit_n, unit_o = new.units["tiny_hy3d"], old.units["tiny_hy3d"]
    p_in = torch.randn(9, unit_n.chart_in.out_features, generator=g)
    p_out = p_in + 0.1 * torch.randn(9, unit_n.chart_in.out_features, generator=g)
    for u in (unit_n, unit_o):
        u.requires_grad_(True)
    ln, _ = D.d3a_loss(unit_n, host, p_in, parent_out=p_out, levels=torch.zeros(9), row_types=torch.full((9,), 7))
    lo, _ = v102.d3a_loss(unit_o, host, p_in, parent_out=p_out, levels=torch.zeros(9), row_types=torch.full((9,), 7))
    ln.backward()
    lo.backward()
    assert torch.equal(ln, lo)
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(unit_n.parameters(), unit_o.parameters()) if a.grad is not None)
    for u in (unit_n, unit_o):
        u.zero_grad(set_to_none=True)
        u.requires_grad_(False)
    # D3c: two band steps
    ids = torch.tensor([[1, 2, 3, new.music_band_begin + 4, new.music_band_begin + 9]])
    top_ids = torch.tensor([[4, 1], [9, 0], [2, 3]])
    lp = torch.log(torch.tensor([[0.5, 0.2], [0.4, 0.1], [0.3, 0.3]]))
    bn, bo = D.BandTrainer(new, lr=1e-2, codes=12), v102.BandTrainer(old, lr=1e-2, codes=12)
    for _ in range(2):
        assert bn.step({"ids": ids}, 3, top_ids, lp) == bo.step({"ids": ids}, 3, top_ids, lp)
    assert torch.equal(bn.head_master, bo.head_master) and torch.equal(bn.embed_master, bo.embed_master)
    new.band_embed_override = old.band_embed_override = None
    # D3d: bridge deltas + LoRA, answer KD
    seg = D.tiny_segment(child["cfg"], "ocr", prompt=6, rows=5, seed=8, suffix=4)
    length = int(seg["ids"].shape[1])
    answer = torch.arange(length - 4, length - 1)
    g = torch.Generator().manual_seed(44)
    q_lp, q_ids = torch.log_softmax(torch.randn(answer.numel(), new.base_vocab, generator=g) * 3.0, -1).topk(5, dim=-1)
    targets = D.answer_kd_targets(length, answer, torch.randint(0, D.TINY.VOCAB, (answer.numel(),), generator=g), q_ids, q_lp)
    out = []
    for module, h in ((D, new), (v102, old)):
        ds = h.add_bridge_deltas([6])
        for name in h.bank.names:
            h.bank.tensors[name].requires_grad_(True)
        loss, _, _ = module.full_depth_forward_backward(h, seg, module.d3d_head_loss(h, targets, denominator=3.0))
        out.append((loss, [d.delta.grad.clone() for d in ds] + [h.bank.tensors[n].grad.clone() for n in h.bank.names
                                                              if h.bank.tensors[n].grad is not None]))
    assert out[0][0] == out[1][0] and len(out[0][1]) == len(out[1][1])
    assert all(torch.equal(a, b) for a, b in zip(out[0][1], out[1][1]))


def test_v102_rate_cell_x0_is_bit_identical(child, v102) -> None:
    new, old = _fresh(child), _fresh(child, module=v102)
    seg = D.tiny_segment(child["cfg"], "ace", prompt=6, seed=31)
    pn = next(p for p in new.layout if p.container == "tiny_ace")
    po = next(p for p in old.layout if p.container == "tiny_ace")
    cn, rn = D.rate_cell(new, pn, seg, stage="d3b", steps=2, flow_objective="x0_endpoint")
    co, ro = v102.rate_cell(old, po, seg, stage="d3b", steps=2)
    assert rn == ro == [] and cn["losses"] == co["losses"] and cn["grad_norms"] == co["grad_norms"]


# -- D3c: 64,000 written of 65,536 allocated ----------------------------------------


def test_d3c_reserved_rows_are_outside_the_normaliser_and_frozen(child) -> None:
    hybrid = _fresh(child)
    codes = 12
    assert hybrid.music_rows == 16 and hybrid.music_rows - codes == 4
    band = D.BandTrainer(hybrid, lr=5e-2, codes=codes)
    ids = torch.tensor([[1, 2, 3, hybrid.music_band_begin + 4, hybrid.music_band_begin + 9, hybrid.music_band_begin + 11]])
    top_ids = torch.tensor([[4, 1], [9, 0], [11, 3], [2, 5]])
    lp = torch.log(torch.tensor([[0.5, 0.2], [0.4, 0.1], [0.3, 0.3], [0.6, 0.2]]))
    contract = hybrid.music_contract(codes)
    for _ in range(3):
        band.step({"ids": ids}, 3, top_ids, lp)
    assert band._frozen_digests() == band.frozen
    tables = band.music_tables()
    assert tables["music.embed"].shape == (16, hybrid.hidden) and tables["music.head"].shape == (16, hybrid.hidden)
    assert D.tensor_sha256(tables["music.head"][codes:]) == contract["reserved_head_sha256"]
    assert D.tensor_sha256(tables["music.embed"][codes:]) == contract["reserved_embed_sha256"]
    assert band.head_opt.steps.tolist() == [3] * codes
    # the normaliser: band logits are over the written rows only (the head master holds 12 rows)
    assert band.head_master.shape[0] == codes
    hybrid.band_embed_override = None


@pytest.mark.parametrize("case", ["reserved_id", "ledger_id", "teacher_code", "repeat", "nonfinite"])
def test_d3c_refuses_rows_outside_the_written_codes(child, case) -> None:
    hybrid = _fresh(child)
    band = D.BandTrainer(hybrid, lr=1e-2, codes=12)
    ids = torch.tensor([[1, 2, 3, hybrid.music_band_begin + 4, hybrid.music_band_begin + 9]])
    top_ids = torch.tensor([[4, 1], [9, 0], [2, 3]])
    lp = torch.log(torch.tensor([[0.5, 0.2], [0.4, 0.1], [0.3, 0.3]]))
    if case == "reserved_id":
        ids[0, 4] = hybrid.music_band_begin + 13
    elif case == "ledger_id":
        ids[0, 3] = hybrid.music_band_begin - 2
    elif case == "teacher_code":
        top_ids[1, 0] = 12
    elif case == "repeat":
        top_ids[2] = torch.tensor([3, 3])
    else:
        lp[0, 0] = float("nan")
    with pytest.raises(D.D3Error):
        band.step({"ids": ids}, 3, top_ids, lp)
    hybrid.band_embed_override = None


# -- D3a: declared host-write target ----------------------------------------------


def test_d3a_chart_out_is_owned_only_with_a_declared_target(child) -> None:
    hybrid = _fresh(child)
    unit = hybrid.units["tiny_hy3d"]
    without = D.d3a_parameters(unit, write_target_declared=False)
    with_ = D.d3a_parameters(unit, write_target_declared=True)
    assert not any(k.startswith("chart_out.") for k in without) and "chart_out.weight" in with_
    assert set(with_) - set(without) == {"chart_out.weight"}
    # 1.0.2's census owned chart_out, but its loss never reached it
    unit.requires_grad_(True)
    g = torch.Generator().manual_seed(2)
    host = torch.randn(6, hybrid.hidden, generator=g)
    p_in = torch.randn(6, unit.chart_in.out_features, generator=g)
    p_out = p_in + 0.1
    loss, _ = D.d3a_loss(unit, host, p_in, parent_out=p_out, row_types=torch.full((6,), 7), levels=torch.zeros(6))
    loss.backward()
    assert unit.chart_out.weight.grad is None
    unit.zero_grad(set_to_none=True)
    target = torch.randn(6, hybrid.hidden, generator=g)
    loss, parts = D.d3a_loss(unit, host, p_in, parent_out=p_out, row_types=torch.full((6,), 7), levels=torch.zeros(6),
                             host_write_target=target)
    loss.backward()
    assert unit.chart_out.weight.grad is not None and float(unit.chart_out.weight.grad.norm()) > 0 and "chart_out_rel_mse" in parts
    unit.zero_grad(set_to_none=True)
    unit.requires_grad_(False)


def test_d3a_complete_loss_equals_codex(child, codex) -> None:
    hybrid = _fresh(child)
    unit = hybrid.units["tiny_hy3d"]
    assert float(unit.delta_scale) == 1.0
    g = torch.Generator().manual_seed(8)
    host = torch.randn(7, hybrid.hidden, generator=g)
    p_in = torch.randn(7, unit.chart_in.out_features, generator=g)
    p_out = p_in + 0.2 * torch.randn(7, unit.chart_in.out_features, generator=g)
    target = torch.randn(7, hybrid.hidden, generator=g)
    types = torch.tensor([7, 7, 0, 7, 7, 7, 0])
    weights = {"chart_in": 1.0, "block": 0.5, "chart_out": 2.0}
    ours, parts = D.d3a_loss(unit, host, p_in, parent_out=p_out, levels=torch.zeros(7), row_types=types, w_in=1.0, w_block=0.5,
                             host_write_target=target, w_write=2.0)
    theirs, tparts = codex["F"].complete_refit_loss(D, unit, host, p_in, p_out, target, levels=torch.zeros(7), row_types=types,
                                                    positions=None, weights=weights, unit_weight=1.0)
    assert abs(float(ours) - float(theirs)) <= 1e-6 * abs(float(theirs))
    assert abs(parts["chart_out_rel_mse"] - tparts["chart_out_rel_mse"]) <= 1e-6 * tparts["chart_out_rel_mse"]


def test_d3a_declaration_rules() -> None:
    good = {"objective": "teacher_delta_host_write", "weights": {"chart_in": 1.0, "block": 1.0, "chart_out": 1.0},
            "d2_child_sha256": "a" * 64, "host_cache_sha256": "b" * 64, "host_write_target_sha256": "c" * 64,
            "teacher_cache_sha256": "d" * 64, "host_write_target_independent": True, "inferred_from_old_cache": False}
    assert D.check_d3a_declaration(good) == good and D.check_d3a_declaration(None) is None
    for key, value in (("objective", "x"), ("d2_child_sha256", "short"), ("inferred_from_old_cache", True),
                       ("host_write_target_independent", False), ("weights", {"chart_in": 1.0, "block": 1.0})):
        with pytest.raises(D.D3Error):
            D.check_d3a_declaration({**good, key: value})


# -- D3d: QVL feature KD and alignment ------------------------------------------------


def _d3d_setup(hybrid, seed: int = 8):
    seg = D.tiny_segment(hybrid.cfg, "ocr", prompt=6, rows=5, seed=seed, suffix=4)
    length = int(seg["ids"].shape[1])
    answer = torch.arange(length - 4, length - 1)
    g = torch.Generator().manual_seed(seed + 40)
    q_lp, q_ids = torch.log_softmax(torch.randn(answer.numel(), hybrid.base_vocab, generator=g) * 3.0, -1).topk(5, dim=-1)
    targets = D.answer_kd_targets(length, answer, seg["ids"][0, answer + 1], q_ids, q_lp)
    qvl = next(p for p in hybrid.layout if 6 in p.type_ids)
    teacher = torch.randn(1, 5, int(qvl.spec.width), generator=g) * 0.5
    return seg, targets, answer, qvl, teacher


def _d3d_trainables(hybrid, qvl):
    ds = hybrid.add_bridge_deltas([6])
    with torch.no_grad():
        for d in ds:
            d.delta.copy_(torch.linspace(-0.02, 0.02, d.delta.numel()).view_as(d.delta))
        for n in hybrid.bank.names:
            if n.endswith(".b"):
                hybrid.bank.tensors[n].copy_(torch.full_like(hybrid.bank.tensors[n], 0.01))
    return hybrid.unit_parameters([qvl.container]) + [d.delta for d in ds] + hybrid.lora_parameters()


def test_d3d_feature_kd_streamed_equals_one_autograd_graph(child_qvl) -> None:
    hybrid = _fresh(child_qvl)
    seg, targets, answer, qvl, teacher = _d3d_setup(hybrid)
    tensors = _d3d_trainables(hybrid, qvl)
    feature = D.UnitFeatureKD(qvl.container, teacher)
    loss, _, _ = D.full_depth_forward_backward(hybrid, seg, D.d3d_head_loss(hybrid, targets, denominator=float(answer.numel())),
                                               feature=feature)
    streamed = [t.grad.clone() for t in tensors]
    # reference: one graph over ALL layers, text KD on the head + 0.1 x feature MSE on the captured block
    # output.  It runs on a RESIDENT twin (same checkpoint, same LoRA seed, same deltas / bank values): the
    # streamed engine keeps one weight buffer per layer type, so one autograd graph across layers that
    # share a type would see the buffer reloaded under it (D3b's reference spans only d..L-1, one layer
    # per type on the tiny child, so it never hit this).
    twin = D.HybridD3(child_qvl["dir"], child_qvl["parent_config"],
                      D.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch", mode="resident", lora_seed=3,
                                      lora=T.SitePlanConfig(rank=4, alpha=4.0, k3_rank=0, trunk_ffn_width=D.TINY.TRUNK, gain=False),
                                      pin_boundaries=False))
    seg2, targets2, answer2, qvl2, teacher2 = _d3d_setup(twin)
    assert torch.equal(seg2["ids"], seg["ids"]) and torch.equal(teacher2, teacher) and qvl2.container == qvl.container
    twin_tensors = _d3d_trainables(twin, qvl2)
    assert len(twin_tensors) == len(tensors) and all(torch.equal(a, b) for a, b in zip(twin_tensors, tensors))
    context = twin.prepare(seg2, media_grad=True)
    captured: list[torch.Tensor] = []
    with D.capture_block_output(twin.units[qvl.container], captured), torch.enable_grad():
        h = context["x"]
        contexts = twin.contexts(h)
        for i in range(twin.num_layers):
            h = twin.engine._run(twin.engine.layer(i), i, h, contexts)
            if twin.units_after(i):
                h = twin.apply_units(h, i, context)
        normed = twin.final_norm(h)
        text = D.d3d_head_loss(twin, targets2, denominator=float(answer2.numel()))(normed, context)
        feat = 0.1 * (captured[0].float() - teacher2).square().mean()
        (text + feat).backward()
    reference = [t.grad.clone() for t in twin_tensors]
    assert len(captured) == 1 and feature.captures == 1
    assert abs(loss - float(text + feat)) <= 1e-5 * abs(float(text + feat))
    worst = max(float((a - b).norm() / b.norm().clamp_min(1e-30)) for a, b in zip(streamed, reference))
    assert worst <= 1e-5 and all(float(b.norm()) > 0 for b in reference)
    for model, ts in ((hybrid, tensors), (twin, twin_tensors)):
        for t in ts:
            t.grad = None
            t.requires_grad_(False)
        model.bridge_deltas.clear()


def test_d3d_feature_kd_equals_codex(child_qvl, codex) -> None:
    grads = []
    for which in ("ours", "codex"):
        hybrid = _fresh(child_qvl)
        seg, targets, answer, qvl, teacher = _d3d_setup(hybrid, seed=9)
        ds = hybrid.add_bridge_deltas([6])
        tensors = hybrid.unit_parameters([qvl.container]) + [d.delta for d in ds] + hybrid.lora_parameters()
        text = D.d3d_head_loss(hybrid, targets, denominator=float(answer.numel()))
        if which == "ours":
            loss, _, _ = D.full_depth_forward_backward(hybrid, seg, text, feature=D.UnitFeatureKD(qvl.container, teacher))
        else:
            loss = codex["V"].feature_forward_backward(D, hybrid, seg, text, qvl.container, teacher)["loss"]
        grads.append((loss, [t.grad.clone() if t.grad is not None else torch.zeros_like(t) for t in tensors]))
    assert abs(grads[0][0] - grads[1][0]) <= 1e-6 * abs(grads[1][0])
    assert all(torch.allclose(a, b, rtol=1e-6, atol=1e-9) for a, b in zip(grads[0][1], grads[1][1]))


def test_d3d_trains_the_qvl_unit_with_feature_kd(child_qvl) -> None:
    hybrid = _fresh(child_qvl)
    report = D.synthetic_training(hybrid, steps=25)
    assert report["d3d"]["feature_kd"] and report["d3d"]["last"] < 0.7 * report["d3d"]["first"]


def test_check_vision_alignment() -> None:
    bands = {6: {"feature": (32, 48), "emission": (16, 17)}}
    n = 4
    patches = torch.randn(n, 16)
    features = torch.zeros(1, n, 48)
    features[0, :, 32:48] = patches
    positions = torch.rand(1, n, 4)
    seg = {"types": torch.full((1, n), 6), "features": features, "positions": positions}
    teacher = {"block_in": torch.randn(n, 8), "block_out": torch.randn(n, 8), "grid_thw": torch.tensor([[1, 2, 2]]),
               "cu_seqlens": torch.tensor([0, n], dtype=torch.int32), "positions": positions[0].clone(), "patch_flat": patches.clone()}
    assert D.check_vision_alignment(seg, teacher, bands) == {"rows": n, "pixels_checked": True, "feature_width": 8}
    bad = dict(teacher, patch_flat=patches + 1e-3)
    with pytest.raises(D.D3Error, match="pixel"):
        D.check_vision_alignment(seg, bad, bands)
    with pytest.raises(D.D3Error, match="positions"):
        D.check_vision_alignment(seg, dict(teacher, positions=torch.rand(n, 4)), bands)
    with pytest.raises(D.D3Error, match="block_out"):
        D.check_vision_alignment(seg, dict(teacher, block_out=torch.randn(n + 1, 8)), bands)
    leak = dict(seg, features=features.clone())
    leak["features"][0, 0, 3] = 1.0
    with pytest.raises(D.D3Error, match="outside the C11"):
        D.check_vision_alignment(leak, teacher, bands)
    no_pixels = {k: v for k, v in teacher.items() if k != "patch_flat"}
    assert D.check_vision_alignment(seg, no_pixels, bands)["pixels_checked"] is False


def test_check_answer_targets() -> None:
    ids = torch.tensor([5, 6, 7, 8, 9, 3])
    mask = torch.tensor([False, False, False, True, True, True])
    positions = torch.tensor([2, 3, 4])
    q_ids = torch.randint(0, 40, (3, 4))
    q_lp = torch.log_softmax(torch.randn(3, 4), -1)
    targets = D.answer_kd_targets(6, positions, ids[positions + 1], q_ids, q_lp)
    assert D.check_answer_targets(targets, ids, mask, 40) == 3
    with pytest.raises(D.D3Error, match="positions before assistant"):
        D.check_answer_targets(targets, ids, torch.tensor([False, False, True, True, True, True]), 40)
    bad = dict(targets, q_ids=targets["q_ids"].clone())
    bad["q_ids"][3, 0] = 40
    with pytest.raises(D.D3Error, match="native text rows"):
        D.check_answer_targets(bad, ids, mask, 40)
    wrong = dict(targets, next_ids=targets["next_ids"].clone())
    wrong["next_ids"][2] = 1
    with pytest.raises(D.D3Error, match="own assistant tokens"):
        D.check_answer_targets(wrong, ids, mask, 40)


# -- D3e: speech lanes ---------------------------------------------------------------


def _speech_in_segment(hybrid, *, audio_rows: int = 4, prompt_text: int = 5, answer: int = 4, eos: int = 7, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    placeholder = hybrid.media_placeholder
    text = torch.randint(8, D.TINY.VOCAB, (prompt_text,), generator=g)
    body = torch.randint(8, D.TINY.VOCAB, (answer - 1,), generator=g)
    ids = torch.cat([text[:2], torch.full((audio_rows,), placeholder), text[2:], body, torch.tensor([eos])]).view(1, -1)
    prompt_length = 2 + audio_rows + prompt_text - 2
    feats = torch.zeros(1, audio_rows, D.TINY.FEATURE)
    lo, hi = D.TINY_BANDS[3][0]
    feats[0, :, lo:hi] = torch.randn(audio_rows, hi - lo, generator=g)
    seg = {"ids": ids, "features": feats, "types": torch.full((1, audio_rows), 3), "positions": torch.rand(1, audio_rows, 4, generator=g),
           "sigma": torch.zeros(1, audio_rows), "indices": torch.arange(2, 2 + audio_rows).view(1, -1), "media_rows": audio_rows,
           "source_positions": {"h3": torch.zeros(1, ids.shape[1], 3)}}
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, prompt_length:] = True
    return seg, mask, prompt_length, eos


def test_speech_in_targets_equal_codex(child, codex) -> None:
    hybrid = _fresh(child)
    seg, mask, prompt_length, eos = _speech_in_segment(hybrid)
    ours, n = D.speech_in_targets(hybrid, seg, mask, prompt_length, eos)
    theirs, m = codex["A"].assistant_ce_targets(hybrid, seg, mask, prompt_length, eos)
    assert n == m == 4 and set(ours) == set(theirs) and all(torch.equal(ours[k], theirs[k]) for k in ours)
    D.check_speech_in(hybrid, seg, prompt_length)
    codex["A"].validate_clean_observation(hybrid, seg, prompt_length)
    bad = mask.clone()
    bad[0, -1] = False
    with pytest.raises(D.D3Error, match="EOS"):
        D.speech_in_targets(hybrid, seg, bad, prompt_length, eos)
    noisy = dict(seg, sigma=torch.full((1, 4), 0.3))
    with pytest.raises(D.D3Error, match="clean"):
        D.check_speech_in(hybrid, noisy, prompt_length)


def test_prompt_only_scope_equals_codex_and_blocks_answer_leakage(child, codex) -> None:
    hybrid = _fresh(child)
    seg, mask, prompt_length, eos = _speech_in_segment(hybrid, seed=3)
    h3 = D.h3_speech_unit(hybrid)
    with D.prompt_only_unit(hybrid, h3.container, prompt_length):
        ours = hybrid.forward(seg)
    with codex["A"].prompt_only_h3(hybrid, h3.container, prompt_length):
        theirs = hybrid.forward(seg)
    assert torch.equal(ours["normed"], theirs["normed"]) and "unit_write" not in hybrid.__dict__
    # changing the answer tokens must not change the H3 write on the audio rows
    other = dict(seg, ids=seg["ids"].clone())
    other["ids"][0, prompt_length:-1] = (other["ids"][0, prompt_length:-1] + 3) % D.TINY.VOCAB
    with D.prompt_only_unit(hybrid, h3.container, prompt_length):
        again = hybrid.forward(other)
    assert torch.equal(ours["writes"][h3.container][:, 2:6], again["writes"][h3.container][:, 2:6])
    # without the scope the answer leaks into the audio rows' write
    free_a, free_b = hybrid.forward(seg), hybrid.forward(other)
    assert not torch.equal(free_a["writes"][h3.container][:, 2:6], free_b["writes"][h3.container][:, 2:6])


def test_speech_in_lane_step_trains_unit_bridge_and_lora(child) -> None:
    hybrid = _fresh(child)
    seg, mask, prompt_length, eos = _speech_in_segment(hybrid, seed=5)
    h3 = D.h3_speech_unit(hybrid)
    ds = hybrid.add_bridge_deltas([3])
    groups = {"unit": hybrid.unit_parameters([h3.container]), "bridge": [d.delta for d in ds], "lora": hybrid.lora_parameters()}
    out = D.speech_lane_step(hybrid, "speech_in_ce", seg, prompt_length=prompt_length, target_mask=mask, completion_eos_id=eos)
    norms, problems = D.grad_group_norms(groups)
    assert problems == [] and out["answer_positions"] == 4 and math.isfinite(out["loss"])
    for ts in groups.values():
        for t in ts:
            t.grad = None
            t.requires_grad_(False)
    hybrid.bridge_deltas.clear()


def _speech_out_segment(hybrid, *, frames: int = 3, prompt_length: int = 5, sigma: float = 0.3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = 2 * frames
    ids = torch.cat([torch.randint(8, D.TINY.VOCAB, (prompt_length,), generator=g), torch.full((n,), hybrid.media_placeholder)]).view(1, -1)
    flo, fhi = hybrid.noisy_feature_bands()[3]
    elo, ehi = hybrid.bands[3]["emission"]
    x0 = torch.randn(n, ehi - elo, generator=g)
    eps = torch.randn(n, ehi - elo, generator=g)
    features, target = torch.zeros(1, n, D.TINY.FEATURE), torch.zeros(1, n, D.TINY.FEATURE)
    features[0, :, flo:fhi] = (1 - sigma) * x0 + sigma * eps
    target[0, :, elo:ehi] = x0
    chart = D.type3_chart(prompt_length, frames, plan="audio40hz")
    source = torch.zeros(1, prompt_length + n, 3)
    source[0, prompt_length:] = chart
    seg = {"ids": ids, "features": features, "types": torch.full((1, n), 3), "positions": torch.rand(1, n, 4, generator=g),
           "sigma": torch.full((1, n), sigma), "indices": torch.arange(prompt_length, prompt_length + n).view(1, -1), "media_rows": n,
           "source_positions": {"h3": source}}
    return seg, target, torch.ones(1, n), prompt_length


def test_speech_out_loss_equals_codex_and_trains(child, codex) -> None:
    hybrid = _fresh(child)
    for sigma in D.SPEECH_SIGMAS:
        seg, target, weights, p = _speech_out_segment(hybrid, sigma=sigma, seed=int(sigma * 100))
        D.check_speech_out(hybrid, seg, target, weights, p)
        pred = torch.randn(target.shape, generator=torch.Generator().manual_seed(1))
        ours = D.speech_out_flow_loss(hybrid, seg, target, weights)
        hybrid.emission = lambda normed, ctx: pred  # the loss reads the head output through model.emission
        try:
            a = float(ours(torch.zeros(1), {"types": seg["types"]}))
        finally:
            del hybrid.__dict__["emission"]
        b = float(codex["R"].flow_velocity_loss(pred, target, seg["types"], weights, hybrid.bands, param="x0", target_kind="x0",
                                                x_t=seg["features"], sigma=seg["sigma"],
                                                noisy_feature_bands={3: hybrid.noisy_feature_bands()[3]}))
        assert abs(a - b) <= 1e-6 * abs(b), (sigma, a, b)
    seg, target, weights, p = _speech_out_segment(hybrid, sigma=0.7, seed=9)
    h3 = D.h3_speech_unit(hybrid)
    ds = hybrid.add_bridge_deltas([3])
    es = hybrid.add_emission_deltas([3])
    groups = {"unit": hybrid.unit_parameters([h3.container]), "bridge": [d.delta for d in ds], "emission": [d.delta for d in es],
              "lora": hybrid.lora_parameters()}
    out = D.speech_lane_step(hybrid, "speech_out_flow", seg, prompt_length=p, target=target, weights=weights)
    norms, problems = D.grad_group_norms(groups)
    assert problems == [] and math.isfinite(out["loss"]) and out["sigma"] == [0.7]
    for ts in groups.values():
        for t in ts:
            t.grad = None
            t.requires_grad_(False)
    hybrid.bridge_deltas.clear()
    hybrid.head_deltas.clear()


def test_speech_out_refuses_clean_targets_in_inputs(child) -> None:
    hybrid = _fresh(child)
    seg, target, weights, p = _speech_out_segment(hybrid)
    leak = dict(seg, features=seg["features"].clone())
    lo, hi = hybrid.bands[3]["feature"]
    leak["features"][0, :, lo] = 1.0   # a column of the type-3 band outside the noisy sub-band
    with pytest.raises(D.D3Error, match="outside the type-3"):
        D.check_speech_out(hybrid, leak, target, weights, p)
    with pytest.raises(D.D3Error, match="sigma"):
        D.check_speech_out(hybrid, dict(seg, sigma=torch.zeros_like(seg["sigma"])), target, weights, p)


def test_type3_chart_and_replay_schedule() -> None:
    chart = D.type3_chart(10, 3, plan="audio40hz")
    assert chart.tolist() == [[10, 0, 0], [11, 0, 0], [12, 0, 0], [10, 0, 1], [11, 0, 1], [12, 0, 1]]
    assert torch.equal(D.type3_chart(10, 3, plan="zero"), torch.zeros(6, 3))
    with pytest.raises(D.D3Error):
        D.type3_chart(10, 3, plan="other")
    speech = [f"s{i}" for i in range(40)]
    replay = [f"r{i}" for i in range(20)]
    plan = D.h3_replay_schedule(speech, replay)
    ids = [step["id"] for step in plan["order"]]
    assert plan["replay"] == 10 and plan["replay_deficit"] == 0 and len(ids) == len(set(ids)) == 50
    assert abs(plan["fraction_realised"] - 0.2) < 1e-9
    assert [s["id"] for s in plan["order"] if s["kind"] == "speech"] == speech
    short = D.h3_replay_schedule(speech, replay[:4])
    assert short["replay"] == 4 and short["replay_deficit"] == 6 and len({s["id"] for s in short["order"]}) == 44
    with pytest.raises(D.D3Error, match="both"):
        D.h3_replay_schedule(speech, ["s1"])


def test_stage_order_and_aliases() -> None:
    assert D.D3_STAGE_ORDER.index("D3d") < D.D3_STAGE_ORDER.index("D3e_in") < D.D3_STAGE_ORDER.index("D3e_out")
    assert D.canonical_stage("D3d_audio_ce") == "D3e_in" and D.canonical_stage("D3d_speech_flow") == "D3e_out"
    assert D.predecessor("D3e_in") == "D3d" and D.predecessor("D3a") is None
    assert D.stage_order(["D3f.image_edit"])[-1] == "D3f.image_edit"
    with pytest.raises(D.D3Error):
        D.stage_order(["D3z"])


# -- native state across stages + exact stage resume ---------------------------------


def _train_some(hybrid, steps: int = 2, seed: int = 4):
    context, x_d, depth, loss_fn = D._tiny_d3b_setup(hybrid, seed=seed, objective="guided_velocity")
    deltas = hybrid.add_emission_deltas([4])
    hybrid.freeze_lora_outside(range(depth, hybrid.num_layers))
    ts = hybrid.unit_parameters(["tiny_ace"]) + [d.delta for d in deltas] + hybrid.lora_parameters(range(depth, hybrid.num_layers))
    opt = torch.optim.AdamW(ts, lr=1e-2, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        D.d3b_forward_backward(hybrid, context, x_d, depth, loss_fn)
        opt.step()
    for t in ts:
        t.grad = None
        t.requires_grad_(False)
    return ts


def test_native_state_carries_units_interfaces_and_lora(child, tmp_path) -> None:
    first = _fresh(child)
    _train_some(first)
    seg = D.tiny_segment(child["cfg"], "ace", seed=50)
    want = first.forward(seg)
    D.export_native_state(first, tmp_path / "ace", stage="D3b.ace", step=2, total_steps=2, trained_units=["tiny_ace"])
    second = _fresh(child, seed=11)   # another LoRA seed: the import must replace the bank
    info = D.import_native_state(second, tmp_path / "ace", stage="D3b.qimg")
    got = second.forward(seg)
    assert info["units"] == ["tiny_ace"] and info["interfaces"] == ["emission.out.weight", "emission.type_affine.weight"]
    assert torch.equal(want["normed"], got["normed"]) and torch.equal(want["emission"], got["emission"])
    # cumulative: the next stage adds its own emission rows; the earlier ones stay
    es = second.add_emission_deltas([7])
    with torch.no_grad():
        for d in es:
            d.delta.add_(0.03)
    seg7 = D.tiny_segment(child["cfg"], "hy3d", seed=51)
    want7, want4 = second.forward(seg7), second.forward(seg)
    D.export_native_state(second, tmp_path / "qimg", stage="D3b.qimg", step=1, total_steps=1)
    third = _fresh(child)
    D.import_native_state(third, tmp_path / "qimg", stage="D3b.hy3d")
    got7, got4 = third.forward(seg7), third.forward(seg)
    assert torch.equal(want7["emission"], got7["emission"]) and torch.equal(want4["emission"], got4["emission"])
    manifest = json.loads((tmp_path / "qimg" / "manifest.json").read_text())
    ranges = manifest["descriptors"]["emission.out.weight"]["ranges"]
    assert ranges == sorted([list(D.TINY_BANDS[4][1]), list(D.TINY_BANDS[7][1])])
    assert "unit.tiny_ace.chart_in.weight" in manifest["descriptors"]   # inherited units travel on


def test_native_state_refusals(child, tmp_path) -> None:
    first = _fresh(child)
    _train_some(first)
    with pytest.raises(D.D3Error, match="whole declared budget"):
        D.export_native_state(first, tmp_path / "x", stage="D3b.ace", step=1, total_steps=2, trained_units=["tiny_ace"])
    D.export_native_state(first, tmp_path / "ace", stage="D3b.ace", step=2, total_steps=2, trained_units=["tiny_ace"])
    with pytest.raises(D.D3Error, match="inherits from"):
        D.import_native_state(_fresh(child), tmp_path / "ace", stage="D3c")
    busy = _fresh(child)
    busy.add_emission_deltas([4])
    with pytest.raises(D.D3Error, match="fresh hybrid"):
        D.import_native_state(busy, tmp_path / "ace", stage="D3b.qimg")
    data = (tmp_path / "ace" / "native_state.safetensors").read_bytes()
    (tmp_path / "ace" / "native_state.safetensors").write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(D.D3Error, match="hash"):
        D.import_native_state(_fresh(child), tmp_path / "ace", stage="D3b.qimg")


def test_native_state_music_tables_keep_reserved_rows(child, tmp_path) -> None:
    order = D.D3_STAGE_ORDER
    h = _fresh(child)
    band = D.BandTrainer(h, lr=5e-2, codes=12)
    ids = torch.tensor([[1, 2, 3, h.music_band_begin + 4, h.music_band_begin + 9]])
    band.step({"ids": ids}, 3, torch.tensor([[4, 1], [9, 0], [2, 3]]), torch.log(torch.tensor([[0.5, 0.2], [0.4, 0.1], [0.3, 0.3]])))
    D.export_native_state(h, tmp_path / "d3c", stage="D3c", step=1, total_steps=1, band=band, order=order)
    nxt = _fresh(child)
    info = D.import_native_state(nxt, tmp_path / "d3c", stage="D3d", order=order)
    assert info["music"]
    o = nxt.music_offset
    assert torch.equal(nxt.band_embed_override[o:o + 12], band.embed_master[o:o + 12])
    assert torch.equal(nxt.inherited_band_head[:12], band.head_master)
    assert D.tensor_sha256(nxt.inherited_band_head[12:]) == nxt.music_contract(12)["reserved_head_sha256"]
    # the inherited tables (and the D3c-trained LoRA bank) reproduce the producer's forward
    out = nxt.forward({"ids": ids})
    ref = h.forward({"ids": ids})
    assert torch.equal(out["normed"], ref["normed"])
    D.export_native_state(nxt, tmp_path / "d3d", stage="D3d", step=1, total_steps=1, order=order)
    m = json.loads((tmp_path / "d3d" / "manifest.json").read_text())
    assert m["music_contract"]["written_rows"] == 12 and m["descriptors"]["music.head"]["kind"] == "music"
    h.band_embed_override = None


def test_stage_checkpoint_resume_is_exact_and_pins_total_steps(child, tmp_path) -> None:
    def build():
        h = _fresh(child)
        context, x_d, depth, loss_fn = D._tiny_d3b_setup(h, seed=6, objective="guided_velocity")
        deltas = h.add_emission_deltas([4])
        h.freeze_lora_outside(range(depth, h.num_layers))
        h.unit_parameters(["tiny_ace"])
        named = {f"unit.{k}": v for k, v in h.units["tiny_ace"].named_parameters() if v.requires_grad}
        named.update({f"emission.{d.name}": d.delta for d in deltas})
        named.update({f"lora.{n}": h.bank.tensors[n] for n in h.bank.names if int(n.split(".")[1]) >= depth})
        opt = torch.optim.AdamW(list(named.values()), lr=1e-2, weight_decay=0.0)
        return h, (context, x_d, depth, loss_fn), named, opt

    def step(h, setup, opt):
        opt.zero_grad(set_to_none=True)
        D.d3b_forward_backward(h, setup[0], setup[1], setup[2], setup[3])
        opt.step()

    identity = {"stage": "D3b.ace", "base": "tiny", "objective": "guided_velocity"}
    h, setup, named, opt = build()
    step(h, setup, opt)
    D.save_stage_checkpoint(tmp_path / "ck", identity=identity, total_steps=4, step=1, sampler_counter=1, trainables=named,
                            optimizers={"adamw": opt})
    for _ in range(2):
        step(h, setup, opt)
    h2, setup2, named2, opt2 = build()
    got = D.load_stage_checkpoint(tmp_path / "ck", identity=identity, total_steps=4, trainables=named2, optimizers={"adamw": opt2})
    assert got == {"step": 1, "sampler_counter": 1}
    for _ in range(2):
        step(h2, setup2, opt2)
    assert all(torch.equal(named[k], named2[k]) for k in named)
    h3, _, named3, opt3 = build()
    with pytest.raises(D.D3Error, match="total_steps"):
        D.load_stage_checkpoint(tmp_path / "ck", identity=identity, total_steps=5, trainables=named3, optimizers={"adamw": opt3})
    with pytest.raises(D.D3Error, match="identity"):
        D.load_stage_checkpoint(tmp_path / "ck", identity={**identity, "base": "other"}, total_steps=4, trainables=named3,
                                optimizers={"adamw": opt3})
    # an old-style checkpoint without total_steps is refused
    state = torch.load(tmp_path / "ck" / "state.pt", weights_only=False)
    del state["total_steps"]
    (tmp_path / "old").mkdir()
    torch.save(state, tmp_path / "old" / "state.pt")
    receipt = json.loads((tmp_path / "ck" / "receipt.json").read_text())
    receipt.pop("total_steps")
    receipt["state_sha256"] = _sha(tmp_path / "old" / "state.pt")
    (tmp_path / "old" / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(D.D3Error, match="no declared total_steps"):
        D.load_stage_checkpoint(tmp_path / "old", identity=identity, total_steps=4, trainables=named3, optimizers={"adamw": opt3})
    for hh in (h, h2, h3):
        hh.head_deltas.clear()


def test_lazy_row_adam_state_survives_a_stage_checkpoint(tmp_path) -> None:
    table = torch.randn(6, 3)
    a = D.LazyRowAdam(table.clone(), lr=0.1)
    a.step(torch.tensor([1, 4]), torch.ones(2, 3))
    D.save_stage_checkpoint(tmp_path / "ck", identity={"s": 1}, total_steps=3, step=1, sampler_counter=1, trainables={},
                            lazy={"head": a}, tables={"head": a.table})
    b = D.LazyRowAdam(table.clone(), lr=0.1)
    D.load_stage_checkpoint(tmp_path / "ck", identity={"s": 1}, total_steps=3, trainables={}, lazy={"head": b}, tables={"head": b.table})
    for opt in (a, b):
        opt.step(torch.tensor([4, 5]), torch.full((2, 3), 0.5))
    assert torch.equal(a.table, b.table) and torch.equal(a.steps, b.steps)


_PINS = {"corpus_sha256": "a" * 64, "teacher_manifest_sha256": "b" * 64, "target_shard_sha256": "c" * 64, "sealed_target": True}


def _pair_row(i: int, pair: int, *, lane: str = "h3", group: str | None = None) -> dict:
    binding = {"input_sha256": f"{2 * pair + 1:064x}", "target_sha256": f"{2 * pair + 2:064x}", "conditioning_sha256": "d" * 64}
    return {"id": f"r{i}", "recovery_group_id": group or f"g{i % 5}", "split": "train", "held": False, "lane": lane, **_PINS,
            **binding, "pair_id": D.canonical_sha256(binding)}


def test_d4_select_replay_equals_codex(codex) -> None:
    rows = [_pair_row(i, i) for i in range(32)]
    ours = D.d4_select_replay("h3", rows, quota=300, unit="pairs")
    theirs = codex["K"].select_replay("h3", rows)
    assert ours["selected"] == theirs["selected_ids"] and ours["amount"] == theirs["selected_amount"] == 32
    assert ours["deficit"] == theirs["deficit"] == 268
    text = [{"id": f"t{i}", "recovery_group_id": "g", "split": "train", "held": False, "lane": "short_text", "real_tokens": 400_000, **_PINS}
            for i in range(10)]
    ours = D.d4_select_replay("short_text", text, quota=2_500_000, unit="real_tokens")
    theirs = codex["K"].select_replay("short_text", text)
    assert ours["selected"] == theirs["selected_ids"] and ours["overshoot"] == theirs["overshoot"] == 300_000
    # the rows both refuse: an unsealed target, a missing source pin, a pair_id not bound to its shas,
    # and a target replayed as its own input
    unsealed = [dict(rows[0], sealed_target=False)]
    unpinned = [{k: v for k, v in rows[0].items() if k != "target_shard_sha256"}]
    unbound = [dict(rows[0], pair_id="p0")]
    same = {"input_sha256": "9" * 64, "target_sha256": "9" * 64, "conditioning_sha256": "d" * 64}
    selfie = [dict(rows[0], **same, pair_id=D.canonical_sha256(same))]
    for match, case in (("sealed", unsealed), ("source pin", unpinned), ("digest", unbound), ("target cannot", selfie)):
        with pytest.raises(D.D3Error, match=match):
            D.d4_select_replay("h3", case, quota=300, unit="pairs")
        with pytest.raises(codex["K"].Error):
            codex["K"].select_replay("h3", case)


# -- D4 driver prep ------------------------------------------------------------------


def test_cache_identity_reads_authentic_receipts() -> None:
    receipt = json.loads(sorted(H3_FIXTURE.glob("*.json"))[0].read_text(encoding="utf-8"))
    ident = D.cache_identity(receipt, item={"tensors": {"sha256": receipt["cache_sha256"]},
                                            "recovery_group_id": receipt["source"]["recovery_group_id"]})
    assert ident["tensors_sha256"] == receipt["cache_sha256"] and ident["recovery_group_id"] == receipt["source"]["recovery_group_id"]
    assert ident["source"]["source_manifest_sha256"] == receipt["source"]["manifest_sha256"]
    with pytest.raises(D.D3Error, match="contradictory"):
        D.cache_identity({**receipt, "tensors_sha256": "0" * 64})
    with pytest.raises(D.D3Error, match="tensors sha"):
        D.cache_identity(receipt, item={"tensors": {"sha256": "1" * 64}})
    with pytest.raises(D.D3Error, match="nested source"):
        D.cache_identity({k: v for k, v in receipt.items() if k != "source"})
    claude = {"schema": "omni-campaign.teacher_cache_shard.v2", "file": "shard-00000.safetensors", "file_sha256": "e" * 64,
              "item_id": "x", "item_sha256": "f" * 64, "corpus_sha256": "a" * 64, "config_sha256": "b" * 64, "teacher": "qvl32_vit_l13"}
    got = D.cache_identity(claude, manifest={"shards": [{"name": "shard-00000.safetensors", "sha256": "e" * 64}]})
    assert got["tensors_sha256"] == "e" * 64
    with pytest.raises(D.D3Error, match="sealed cache manifest"):
        D.cache_identity(claude, manifest={"shards": [{"name": "shard-00000.safetensors", "sha256": "9" * 64}]})
    speech = {"schema": "chimera_recovery.speech_flow_cache.v1", "tensors_sha256": "c" * 64, "recovery_group_id": "g", "split": "train",
              "held": False, "source_row_sha256": "d" * 64, "id": "speech:sigma0"}
    assert D.cache_identity(speech)["tensors_sha256"] == "c" * 64
    with pytest.raises(D.D3Error, match="unknown"):
        D.cache_identity({"schema": "something"})


def test_sigma_grid_of() -> None:
    assert D.sigma_grid_of({"sigma_grid": [0.05, 0.3, 0.7, 1.0]}) == (0.05, 0.3, 0.7, 1.0)
    assert D.sigma_grid_of({"sigmas": [0.5]}) == (0.5,)
    assert D.sigma_grid_of({"sigma_grid": [0.5], "sigmas": [0.5]}) == (0.5,)
    for bad in ({}, {"sigma_grid": [0.5], "sigmas": [0.4]}, {"sigma_grid": []}, {"sigma_grid": [1.5]}):
        with pytest.raises(D.D3Error):
            D.sigma_grid_of(bad)


def _union_groups():
    from decimal import Decimal
    rates = {"D2.lora": "1e-4", "D2.k3_lora": "1e-4", "D2.gain": "1e-3", "D3.unit": "2e-5", "D3.emission": "2e-5", "D3.bridge": "2e-5",
             "D3.music_embed": "1e-3", "D3.music_head": "1e-3", "D3.shared_lora": "1e-4"}
    return [{"family": f, "previous_base_lr": r, "lr": str(Decimal(r) * Decimal("0.3")),
             "parameters": [{"parameter_id": f"{f}.p{i}", "numel": 10 + i} for i in range(2)]} for f, r in rates.items()]


def test_d4_optimizer_union_equals_codex(codex) -> None:
    groups = _union_groups()
    ours, theirs = D.d4_check_optimizer_union(groups), codex["K"].optimizer_union(groups)
    assert ours["parameters"] == theirs["parameters"] and ours["numel"] == theirs["numel"] and ours["families"] == theirs["families"]
    bad = copy.deepcopy(groups)
    bad[0]["lr"] = "3.1e-5"
    dup = copy.deepcopy(groups)
    dup[1]["parameters"][0]["parameter_id"] = dup[0]["parameters"][0]["parameter_id"]
    missing = groups[1:]
    for case in (bad, dup, missing):
        with pytest.raises(D.D3Error):
            D.d4_check_optimizer_union(case)
        with pytest.raises(codex["K"].Error):
            codex["K"].optimizer_union(case)


def test_d4_base_replay_and_chain_guards(codex) -> None:
    D.d4_check_base(["OC02-F-Q38-27BU-OMNI1.clone-pre-D1", "OC02-D-Q38-27BU-OMNI1D1", "OMNI1D1.clone-pre-D2"])
    with pytest.raises(D.D3Error, match="twice"):
        D.d4_check_base(["OC02-F-Q38-27BU-OMNI1.clone-pre-D1", "OC02-D-Q38-27BU-OMNI1D2"])
    rows = [_pair_row(i, i % 7) for i in range(32)]
    pick = D.d4_select_replay("h3", rows, quota=300, unit="pairs", excluded_groups={"g4"})
    assert pick["amount"] == len(pick["selected"]) and pick["deficit"] == 300 - pick["amount"] and pick["duplicated"] == 0
    assert len({rows[int(i[1:])]["pair_id"] for i in pick["selected"]}) == pick["amount"]
    text = [{"id": f"t{i}", "recovery_group_id": "g", "split": "train", "held": False, "lane": "short_text", "real_tokens": 1000,
             **_PINS} for i in range(10)]
    got = D.d4_select_replay("short_text", text, quota=4500, unit="real_tokens", excluded_ids={"t0"})
    assert got["selected"] == ["t1", "t2", "t3", "t4", "t5"] and got["overshoot"] == 500
    with pytest.raises(D.D3Error, match="twice"):
        D.d4_select_replay("short_text", text + text[:1], quota=10**9, unit="real_tokens")
    # stage chain: speech-in after vision, before speech-out; one shared r16 lineage
    sha = lambda i: f"{i:064x}"  # noqa: E731
    order = ("D2",) + D.D3_STAGE_ORDER
    states, bank = [], sha(1000)
    for i, role in enumerate(order):
        state = {"stage_role": role, "state_sha256": sha(i + 1), "parent_state_sha256": sha(i) if i else None}
        if role.startswith("D3"):
            new_bank = bank if role == "D3a" else sha(2000 + i)
            state.update({"shared_lora_semantics": "unchanged" if role == "D3a" else "full_replacement",
                          "shared_lora_state_sha256": new_bank, "shared_lora_parent_sha256": bank})
            bank = new_bank
        states.append(state)
    out = D.d4_check_stage_chain(states, initial_shared_bank_sha256=sha(1000))
    assert out["stages"] == 10 and out["shared_r16_state_sha256"] == bank
    # Codex's chain checker agrees once its stage names are used
    codex_names = {"D3e_in": "D3d_audio_ce", "D3e_out": "D3d_speech_flow"}
    codex_states = [{**s, "stage_role": codex_names.get(s["stage_role"], s["stage_role"]),
                     "semantics": "full_replacement"} for s in states]
    assert codex["K"].sequential_states(codex_states, initial_shared_bank_sha256=sha(1000))["shared_r16_state_sha256"] == bank
    swapped = states[:8] + [states[9], states[8]]
    with pytest.raises(D.D3Error, match="order"):
        D.d4_check_stage_chain(swapped, initial_shared_bank_sha256=sha(1000))
    summed = copy.deepcopy(states)
    summed[5]["shared_lora_semantics"] = "add"
    with pytest.raises(D.D3Error, match="summed"):
        D.d4_check_stage_chain(summed, initial_shared_bank_sha256=sha(1000))


# -- 1.1.1: the Codex ReadyObjectiveAdapter guards in bind_native_flow_batch -----------------


def _h3_fixture():
    from safetensors.torch import load_file

    tensors = sorted(H3_FIXTURE.glob("*.safetensors"))[0]
    receipt = json.loads(tensors.with_suffix(".json").read_text(encoding="utf-8"))
    return receipt, D.inflate_native_batch(load_file(str(tensors)), receipt["storage"], width=6144)


def _codex_refuses(codex, receipt, batch, route="joint_av", signs=None) -> bool:
    try:
        codex["R"].ReadyObjectiveAdapter(receipt).bind(route, batch, OMNI1_BANDS, parent_signs=signs)
    except codex["R"].ReadinessError:
        return True
    return False


def test_bind_refuses_what_codex_refuses(codex) -> None:
    receipt, batch = _h3_fixture()
    signs = {int(k): v for k, v in receipt["parent_signs"].items()}
    # accepted by both on the authentic batch
    D.bind_native_flow_batch(receipt, batch, OMNI1_BANDS, route="joint_av", flow_param="raw", signs=signs)
    assert not _codex_refuses(codex, receipt, batch, signs=signs)
    # the receipt alone never sets a sign (1.1.0 took the receipt signs when the caller gave none)
    with pytest.raises(D.D3Error, match="caller audited signs"):
        D.bind_native_flow_batch(receipt, batch, OMNI1_BANDS, route="joint_av", flow_param="raw")
    assert _codex_refuses(codex, receipt, batch, signs=None)
    # flow_signs-style superset is fine when it agrees on the receipt types
    D.bind_native_flow_batch(receipt, batch, OMNI1_BANDS, route="joint_av", flow_param="raw", signs={**signs, 4: -1.0, 7: 1.0})
    with pytest.raises(D.D3Error, match="route"):
        D.bind_native_flow_batch(receipt, batch, OMNI1_BANDS, route="music_generate")
    assert _codex_refuses(codex, receipt, batch, route="music_generate")
    with pytest.raises(D.D3Error, match="unsupported"):
        D.bind_native_flow_batch(receipt, batch, OMNI1_BANDS, route="text_instruction")
    cases = []
    bad = copy.deepcopy(receipt)
    bad["av_alignment"]["same_origin"] = False
    cases.append(("one pinned source window", bad, batch))
    bad = copy.deepcopy(receipt)
    del bad["corpus_sha256"]
    cases.append(("corpus_sha256", bad, batch))
    bad = copy.deepcopy(receipt)
    bad["prediction_kind"] = "unknown"
    cases.append(("head parametrisation", bad, batch))
    pad = dict(batch, native_media_valid_mask=batch["native_media_valid_mask"].clone())
    pad["native_media_valid_mask"][0, 0] = False
    cases.append(("padding", receipt, pad))
    nosrc = {k: v for k, v in batch.items() if k != "charted_source_positions"}
    cases.append(("source positions", receipt, nosrc))
    alias = dict(batch, native_media_targets=batch["native_media_features"])
    cases.append(("aliases", receipt, alias))
    for match, r, b in cases:
        with pytest.raises(D.D3Error, match=match):
            D.bind_native_flow_batch(r, b, OMNI1_BANDS, route="joint_av", flow_param="x0")
        assert _codex_refuses(codex, r, b, signs=signs), match


def _ace_case(codex, *, seed: int = 77):
    g = torch.Generator().manual_seed(seed)
    m, length = 4, 12
    codes = torch.randint(0, D.ACE_CODES, (3,), generator=g)
    ids = torch.randint(0, 1000, (1, length), generator=g)
    ids[0, 1:4] = D.OMNI1_MUSIC_BEGIN + codes
    features = torch.zeros(1, m, 6144)
    features[0, :, :128] = torch.randn(m, 128, generator=g)
    features[0, :, 128:256] = 1.0
    sigma = torch.full((1, m), 0.5)
    x0 = torch.randn(m, 128, generator=g)
    features[0, :, 256:384] = 0.5 * x0 + 0.5 * torch.randn(m, 128, generator=g)
    target = torch.zeros(1, m, 6144)
    target[0, :, :128] = x0
    types = torch.full((1, m), 4)
    batch = {"input_ids": ids, "native_media_features": features, "native_media_targets": target, "native_media_type_ids": types,
             "native_media_indices": torch.arange(6, 6 + m).view(1, m), "native_media_valid_mask": torch.ones(1, m, dtype=torch.bool),
             "native_media_positions": torch.rand(1, m, 4, generator=g), "native_media_loss_mask": torch.ones(1, m),
             "native_media_noise_levels": sigma, "ace_code_indices": torch.arange(1, 4).view(1, 3)}
    th = codex["R"].tensor_sha256
    receipt = {"schema": "chimera_recovery.native_flow_cache.v1", "sealed": True, "all_finite": True, "all_structural_ok": True,
               "guided_teacher": True, "no_clean_target_in_inputs": True, "cache_sha256": "a" * 64, "corpus_sha256": "b" * 64,
               "teacher_sha256": "c" * 64, "recovery_amendment_sha256": "d" * 64, "target_kind": "x0", "prediction_kind": "x0",
               "input_layout": "feature", "target_layout": "emission", "route": "music_generate", "noisy_feature_bands": {"4": [256, 384]},
               "source": {"manifest_sha256": "e" * 64, "shard_sha256": "f" * 64, "sidecar_sha256": "1" * 64, "recovery_group_id": "g",
                          "item_id": "x"},
               "conditioning_contract": {"ace_source_latents": "authentic_code_detokenizer", "ace_chunk_masks": "ones_with_right_padding_zero",
                                         "regime": "teacher_audio_code_conditioned_generation", "full_clean_latent_in_inputs": False,
                                         "source_band_sha256": th(features[..., :128][types == 4]),
                                         "chunk_band_sha256": th(features[..., 128:256][types == 4]),
                                         "codes_sha256": th(codes.to(torch.int32)), "conditioning_manifest_sha256": "2" * 64,
                                         "ace_step15_sha256": "3" * 64, "model_base_sha256": "4" * 64}}
    receipt["noisy_feature_bands_sha256"] = codex["R"].digest(receipt["noisy_feature_bands"])
    return receipt, batch


def test_bind_ace_authentic_conditioning_equals_codex(codex) -> None:
    receipt, batch = _ace_case(codex)
    loss, info = D.bind_native_flow_batch(receipt, batch, OMNI1_BANDS, route="music", flow_param="x0")
    ref = codex["R"].ReadyObjectiveAdapter(receipt).bind("music_generate", batch, OMNI1_BANDS)
    pred = torch.randn(batch["native_media_targets"].shape, generator=torch.Generator().manual_seed(5))
    a, b = float(loss(pred)), float(ref(pred))
    assert info["route"] == "music_generate" and math.isfinite(a) and abs(a - b) <= 1e-6 * abs(b)
    old = copy.deepcopy(receipt)
    old["conditioning_contract"]["full_clean_latent_in_inputs"] = True
    moved = dict(batch, input_ids=batch["input_ids"].clone())
    moved["input_ids"][0, 2] += 1
    leak = dict(batch, native_media_features=batch["native_media_features"].clone())
    leak["native_media_features"][0, :, :128] = batch["native_media_targets"][0, :, :128]
    for match, r, bt in (("authentic code conditioning", old, batch), ("code ids", receipt, moved), ("band digest", receipt, leak)):
        with pytest.raises(D.D3Error, match=match):
            D.bind_native_flow_batch(r, bt, OMNI1_BANDS, route="music_generate")
        assert _codex_refuses(codex, r, bt, route="music_generate"), match
