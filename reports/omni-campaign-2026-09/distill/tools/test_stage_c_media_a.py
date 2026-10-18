"""Pure tests for stage_c_media_a.py 1.0.2 (lane media-a, Claude, 2026-10-12/13). Server, CPU only:

    cd /home/cereal/omni-campaign/distill/tools
    export CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9
    export PYTHONPATH=$CHIMERA_OMNI_SRC/src OMP_NUM_THREADS=8
    nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" -p no:cacheprovider -q test_stage_c_media_a.py

Covers everything that does not need ComfyUI: H3/QIMG geometry against the node formulas, the H3 audio
time-shift against an independent restatement and a finite difference, the x0/velocity algebra with a
perfect synthetic teacher (flat and native audio conventions), sigma plans, seeds and noise identity, CFG,
block-row selection, frame resampling and cover crop, the item reader and contamination screen, shard
atomicity / resume / corruption / non-finite refusal, plan pinning, teacher identity, sealing + verify,
plan arithmetic, and the registry manifest builder on a fixture registry. The comfy-backed paths are covered
by itest_stage_c_media_a.py (inside the r5 image, on a P40).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = Path(__file__).resolve().parent
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load(HERE / "stage_c_media_a.py", "stage_c_media_a")
SIG = M.runtime_signature({"device_name": "cpu-test", "torch": "t"}, {"vae_dtype": "float32"})


# ----------------------------------------------------------------------------- geometry


def _node_temporal(length):
    # independent restatement of comfy_extras/nodes_minimax_h3.py (r5 image)
    n = max(5, length)
    while n % 17 != 5:
        n += 1
    latent_t = 2 if n <= 5 else ((n - 5) // 17) * 5 + 2
    return n, latent_t, round(n / 24 * 40)


@pytest.mark.parametrize("length", [5, 22, 39, 48, 56, 124, 362])
def test_h3_temporal_shape_matches_node(length):
    assert M.h3_temporal_shape(length) == _node_temporal(length)


def test_h3_latent_shapes_design_canvas():
    geo = M.h3_latent_shapes(512, 320, 48)
    assert geo["frame_count"] == 56
    assert geo["video"] == (1, 24, 17, 20, 32)
    assert geo["audio"] == (1, 32, 2, 93)
    assert geo["video_tokens"] == 17 * 10 * 16 == 2720
    assert geo["av_tokens"] == 2720 + 186
    with pytest.raises(SystemExit):
        M.h3_latent_shapes(500, 320, 48)


def test_qimg_latent_shape():
    geo = M.qimg_latent_shape(512, 512)
    assert geo["latent"] == (1, 16, 1, 64, 64)
    assert geo["image_tokens"] == 1024


# ----------------------------------------------------------------------------- schedules


def test_time_shift_sigma_restated_and_slope_is_derivative():
    for s in (0.05, 0.2, 0.5, 0.8, 0.97):
        base = s / (12 - 11 * s)  # invert sigma = 12 b / (1 + 11 b)
        expect = 3 * base / (1 + 2 * base)
        assert math.isclose(M.time_shift_sigma(s, 12.0, 3.0), expect, rel_tol=1e-12)
        h = 1e-6
        fd = (M.time_shift_sigma(s + h, 12.0, 3.0) - M.time_shift_sigma(s - h, 12.0, 3.0)) / (2 * h)
        assert math.isclose(M.time_shift_slope(s, 12.0, 3.0), fd, rel_tol=1e-5)
    assert math.isclose(M.time_shift_sigma(1.0, 12.0, 3.0), 1.0)


def test_time_shift_matches_mediafold_copy_when_present():
    path = OC / "media" / "mediafold" / "c16_h3.py"
    if not path.is_file():
        pytest.skip("mediafold not on this box")
    sys.path.insert(0, str(OC / "media"))
    from mediafold import c16_h3

    for s in (0.1, 0.5, 0.9):
        assert math.isclose(M.time_shift_sigma(s, 12, 3), c16_h3.time_shift_sigma(s, 12, 3), rel_tol=1e-12)
        assert math.isclose(M.time_shift_slope(s, 12, 3), c16_h3.time_shift_slope(s, 12, 3), rel_tol=1e-12)
    layout = c16_h3.packed_layout(7, 9, (24, 3, 4, 6))
    assert M.h3_packed_kinds(7, 9, 3 * 2 * 3) == [int(v) for v in layout["token_kind"].tolist()]


def test_sigma_plan_stratified():
    rows = M.sigma_plan(512, per_item=4, mode="stratified-shifted", shift=12.0, seed=7)
    assert len(rows) == 2048
    assert rows == M.sigma_plan(512, per_item=4, mode="stratified-shifted", shift=12.0, seed=7)
    assert rows != M.sigma_plan(512, per_item=4, mode="stratified-shifted", shift=12.0, seed=8)
    for item in range(0, 512, 37):
        mine = [r for r in rows if r["item"] == item]
        us = [r["u"] for r in mine]
        assert us == sorted(us) and all(0.02 <= u <= 0.98 for u in us)
        for j, r in enumerate(mine):
            assert 0.02 + 0.24 * j <= r["u"] <= 0.02 + 0.24 * (j + 1)
            assert math.isclose(r["sigma"], 12 * r["u"] / (1 + 11 * r["u"]))
    assert len({r["seed"] for r in rows}) == 2048


def test_sigma_plan_fixed_and_cycle_and_errors():
    rows = M.sigma_plan(3, per_item=2, mode="fixed", shift=1.0, seed=1, sigmas=[0.3, 0.7])
    assert [(r["item"], r["sigma"]) for r in rows] == [(0, 0.3), (0, 0.7), (1, 0.3), (1, 0.7), (2, 0.3), (2, 0.7)]
    rows = M.sigma_plan(5, per_item=1, mode="cycle", shift=1.0, seed=1, sigmas=[0.2, 0.5, 0.9])
    assert [r["sigma"] for r in rows] == [0.2, 0.5, 0.9, 0.2, 0.5]
    with pytest.raises(SystemExit):
        M.sigma_plan(3, per_item=2, mode="fixed", shift=1.0, seed=1, sigmas=[0.3])
    with pytest.raises(SystemExit):
        M.sigma_plan(3, per_item=2, mode="cycle", shift=1.0, seed=1, sigmas=[0.3])
    with pytest.raises(SystemExit):
        M.sigma_plan(3, per_item=1, mode="fixed", shift=1.0, seed=1, sigmas=[1.0])


def test_seeds_and_noise_identity():
    assert M.target_seed(5, 3, 2) == 5 * 1_000_003 + 3 * 64 + 2
    with pytest.raises(ValueError):
        M.target_seed(5, 3, 64)
    a = M.make_noise([(1, 24, 2, 4, 4), (1, 32, 2, 8)], 123)
    b = M.make_noise([(1, 24, 2, 4, 4), (1, 32, 2, 8)], 123)
    c = M.make_noise([(1, 24, 2, 4, 4), (1, 32, 2, 8)], 124)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert M.noise_sha(a) == M.noise_sha(b) != M.noise_sha(c)
    # stream order: the video draw comes first from the same generator
    g = torch.Generator().manual_seed(123)
    assert torch.equal(a[0], torch.randn((1, 24, 2, 4, 4), generator=g))
    assert torch.equal(a[1], torch.randn((1, 32, 2, 8), generator=g))


# ----------------------------------------------------------------------------- x0 / velocity algebra


def test_perfect_teacher_recovers_x0_and_velocity_h3():
    """A perfect H3 teacher returns (flat convention) video v = n - x0 and audio slope * (n_a - x0_a)."""

    g = torch.Generator().manual_seed(0)
    x0v, x0a = torch.randn(1, 24, 3, 4, 4, generator=g), torch.randn(1, 32, 2, 9, generator=g)
    nv, na = torch.randn(x0v.shape, generator=g), torch.randn(x0a.shape, generator=g)
    for sigma in (0.2, 0.55, 0.97):
        sa = M.time_shift_sigma(sigma, 12, 3)
        slope = M.time_shift_slope(sigma, 12, 3)
        xv, xa = M.flow_interp(x0v, nv, sigma), M.flow_interp(x0a, na, sa)
        ret_v, ret_a = nv - x0v, slope * (na - x0a)  # what comfy's MiniMaxH3Model returns
        den_v, den_a = xv - sigma * ret_v, xa - sigma * ret_a  # comfy CONST calculate_denoised on the flat ODE
        out = M.h3_native_targets(x_video=xv, x_audio=xa, den_video=den_v, den_audio=den_a, sigma_v=sigma)
        assert torch.allclose(out["v_video"], nv - x0v, atol=2e-5)
        assert torch.allclose(out["x0hat_video"], x0v, atol=2e-5)
        assert torch.allclose(out["v_audio"], ret_a, atol=2e-5)
        assert torch.allclose(out["v_audio_native"], na - x0a, atol=5e-5)
        assert torch.allclose(out["x0hat_audio"], x0a, atol=5e-5)
        assert math.isclose(out["sigma_audio"], sa) and math.isclose(out["slope_audio"], slope)


def test_velocity_from_denoised_and_cfg():
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    den = x - 0.4 * v
    assert torch.allclose(M.velocity_from_denoised(x, den, 0.4), v, atol=1e-6)
    c, u = torch.randn(4), torch.randn(4)
    assert M.cfg_combine(c, u, 1.0) is c
    assert M.cfg_combine(c, None, 4.0) is c
    assert torch.allclose(M.cfg_combine(c, u, 4.0), u + 4.0 * (c - u))
    # linear in the convention: CFG on denoised == CFG on velocity
    x = torch.randn(4)
    vc, vu = M.velocity_from_denoised(x, c, 0.3), M.velocity_from_denoised(x, u, 0.3)
    assert torch.allclose(M.velocity_from_denoised(x, M.cfg_combine(c, u, 2.5), 0.3), M.cfg_combine(vc, vu, 2.5), atol=1e-5)


# ----------------------------------------------------------------------------- block rows


def test_block_row_selection():
    kinds = M.h3_packed_kinds(40, 93, 2720)
    assert len(kinds) == 40 + 186 + 2720
    rows = M.block_row_selection(kinds, 256, seed=9)
    assert rows == M.block_row_selection(kinds, 256, seed=9)
    assert rows != M.block_row_selection(kinds, 256, seed=10)
    assert len(rows) == 256 == len(set(rows)) and rows == sorted(rows)
    picked = [kinds[r] for r in rows]
    assert M.KIND_TEXT not in picked
    assert picked.count(M.KIND_AUDIO) == round(256 * 186 / (186 + 2720))
    with_text = M.block_row_selection(kinds, 256, seed=9, include_text=True)
    assert M.KIND_TEXT in [kinds[r] for r in with_text]
    small = M.block_row_selection(M.h3_packed_kinds(3, 1, 2), 100, seed=1)
    assert small == [3, 4, 5, 6]
    assert M.block_row_selection(M.h3_packed_kinds(3, 0, 0), 5, seed=1) == []


# ----------------------------------------------------------------------------- media helpers


def test_pick_frames_resample():
    src = [k / 30 for k in range(90)]
    tgt = M.frame_times_for_target(48, 24)
    picks = M.pick_frames(src, tgt)
    assert len(picks) == 48 and picks[0] == 0
    assert all(src[p] <= t + 1e-6 and (p + 1 == len(src) or src[p + 1] > t + 1e-6) for p, t in zip(picks, tgt))
    slow = M.pick_frames([0, 0.5, 1.0], [0, 0.25, 0.5, 0.75])
    assert slow == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        M.pick_frames([0.1, 0.2], [0.0])


def test_cover_crop_and_resize():
    assert M.cover_crop_box(1920, 1080, 512, 320) == (569, 320, 28, 0)
    assert M.cover_crop_box(320, 512, 512, 320) == (512, 820, 0, 250)
    frames = torch.rand(3, 90, 160, 3)
    out = M.resize_cover(frames, 64, 32)
    assert tuple(out.shape) == (3, 32, 64, 3) and float(out.min()) >= 0 and float(out.max()) <= 1


# ----------------------------------------------------------------------------- items, bank, shards


def _write_items(tmp_path, rows, name="items.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_read_items_and_errors(tmp_path):
    good = _write_items(tmp_path, [{"id": "a", "prompt": "a cat", "video": "/x.mp4"}, {"id": "b", "prompt": "a dog", "video": "/y.mp4"}])
    assert [r["id"] for r in M.read_items(good, media_key="video")] == ["a", "b"]
    for bad in ([{"id": "a", "prompt": "x", "video": "/x"}, {"id": "a", "prompt": "y", "video": "/y"}],
                [{"id": "a", "prompt": " ", "video": "/x"}],
                [{"id": "a", "prompt": "x"}],
                [{"prompt": "x", "video": "/x"}]):
        with pytest.raises(SystemExit):
            M.read_items(_write_items(tmp_path, bad, "bad.jsonl"), media_key="video")
    assert len(M.read_items(_write_items(tmp_path, [{"id": "a", "prompt": "x"}], "t.jsonl"), media_key="")) == 1


def test_bank_contamination(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"cases": [{"prompt": "Describe the  red marble rolling across the table", "id": "x"}]}))
    strings = M.bank_strings([bank])
    assert M.contaminated("describe the red marble rolling across the table", strings)
    assert M.contaminated("Please: describe the red marble rolling across the table, slowly", strings)
    assert not M.contaminated("a blue bird", strings)


def test_shard_store_resume_corruption_and_nonfinite(tmp_path):
    store = M.ShardStore(tmp_path / "c", SIG)
    assert store.done(0) is None
    rec = store.write(0, {"v": torch.randn(2, 3), "i": torch.arange(2)}, {"targets": [1, 2]})
    assert rec["finite"] and rec["finite_checked"] == ["v"]
    assert store.done(0)["sha256"] == rec["sha256"]
    with pytest.raises(SystemExit):
        store.write(1, {"v": torch.tensor([1.0, float("nan")])}, {})
    assert not store.path(1).exists() and not list((tmp_path / "c").glob("*.tmp"))
    store.path(0).write_bytes(store.path(0).read_bytes()[:-1] + b"\x00")
    with pytest.raises(SystemExit):
        store.done(0)


def test_claim_cache_pins_plan(tmp_path):
    d = tmp_path / "t" / "abc"
    first = M.claim_cache(d, {"a": 1})
    assert M.claim_cache(d, {"a": 1})["plan_sha256"] == first["plan_sha256"]
    with pytest.raises(SystemExit):
        M.claim_cache(d, {"a": 2})
    with pytest.raises(SystemExit):
        M.cache_dir(tmp_path, "Bad Name", "abc")


def test_teacher_identity_and_seal_verify(tmp_path):
    teacher = tmp_path / "teacher.bin"
    teacher.write_bytes(b"weights" * 100)
    before = M.teacher_identity([teacher], "full")
    assert M.identity_unchanged(before, M.teacher_identity([teacher], "full"))
    store = M.ShardStore(tmp_path / "cache", SIG)
    store.write(0, {"v": torch.ones(3)}, {})
    store.write(1, {"v": torch.zeros(3)}, {})
    M.seal_manifest(tmp_path / "cache", {"teacher_integrity": {"unchanged": True}})
    result = M.verify_cache(tmp_path / "cache")
    assert result["ok"] and result["shards"] == 2 and not result["bad_shards"]
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert manifest["read_order"] == ["shard-00000.safetensors", "shard-00001.safetensors"]
    assert manifest["tool"]["version"] == M.TOOL_VERSION
    store.path(1).write_bytes(store.path(1).read_bytes() + b" ")
    assert M.verify_cache(tmp_path / "cache")["bad_shards"] == ["shard-00001.safetensors"]
    teacher.write_bytes(b"weights" * 99 + b"weightz")
    assert not M.identity_unchanged(before, M.teacher_identity([teacher], "full"))


def test_seal_ignores_empty_markers_and_refuses_empty(tmp_path):
    d = tmp_path / "cache"
    store = M.ShardStore(d, SIG)
    store.write_empty(0, [{"index": 0, "id": "a", "dropped": "ValueError: x"}])
    with pytest.raises(SystemExit):
        M.seal_manifest(d, {"teacher_integrity": {"unchanged": True}})
    store.write(1, {"v": torch.ones(2)}, {})
    M.seal_manifest(d, {"teacher_integrity": {"unchanged": True}})
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["read_order"] == ["shard-00001.safetensors"]
    assert manifest["empty_shard_markers"] == ["shard-00000.empty.json"]
    assert M.verify_cache(d)["ok"]
    assert [p.name for p in M.shard_sidecars(d)] == ["shard-00001.json"]


def test_decode_failures_drop_but_device_failures_raise():
    dropped = []
    meta = {"id": "x"}

    def bad_decode():
        raise ValueError("clip too short")

    assert M.decode_or_drop(bad_decode, meta, dropped) is None
    assert dropped == [meta] and meta["dropped"].startswith("ValueError")
    assert M.decode_or_drop(lambda: 5, {"id": "y"}, dropped) == 5

    def oom():
        raise RuntimeError("CUDA error: out of memory")

    with pytest.raises(RuntimeError):
        M.decode_or_drop(oom, {"id": "z"}, dropped)
    assert len(dropped) == 1
    assert M.check_drop_fraction(1, 4, 0.5)["fraction"] == 0.25
    with pytest.raises(SystemExit):
        M.check_drop_fraction(3, 4, 0.5)
    with pytest.raises(SystemExit):
        M.check_drop_fraction(0, 0, 0.5)


def test_plan_bytes_h3_design_numbers():
    out = M.plan_bytes(teacher="h3", items=512, per_item=4, cfg=1.0, width=512, height=320, length=48, block_rows=256)
    floats = 24 * 17 * 20 * 32 + 32 * 2 * 93
    assert out["floats_per_latent"] == floats == 267072
    assert out["targets"] == 2048 and out["forwards"] == 2048
    per_target = 2 * floats * 4 + 2 * 256 * 5376 * 2
    assert out["bytes_per_target"] == per_target
    assert out["bytes_total"] == per_target * 2048 + 512 * floats * 4
    assert out["gib_total"] < 60
    cfg = M.plan_bytes(teacher="qimg", items=2048, per_item=1, cfg=4.0, width=512, height=512)
    assert cfg["forwards"] == 4096 and cfg["bytes_per_target"] == 4 * 65536 * 4


# ----------------------------------------------------------------------------- registry manifest builder


def test_manifest_builder_on_fixture_registry(tmp_path, monkeypatch):
    src = os.environ.get("CHIMERA_OMNI_SRC")
    if not src or not (Path(src) / "src" / "chimera_omni" / "data" / "curated_shards.py").is_file():
        pytest.skip("CHIMERA_OMNI_SRC with chimera_omni.data.curated_shards not set")
    media = tmp_path / "media"
    media.mkdir()
    rows = []
    for k in range(6):
        clip = media / f"clip{k}.mp4"
        if k != 4:
            clip.write_bytes(b"fake%d" % k)
        rows.append({"modality": "video", "record_id": f"r{k}",
                     "input_json": {"messages": [{"role": "user", "content": f"user: A short clip number {k} of a river"}]},
                     "target_json": {"response": f"video artifact {k}"},
                     "media_refs": [{"path": str(clip), "kind": "video", "sha256": "x"}]})
    rows.append({"modality": "image", "record_id": "img", "instruction": "an image", "response": "r",
                 "media_refs": [{"path": str(media / "clip0.mp4"), "kind": "image", "sha256": "x"}]})
    rows.append({"modality": "video", "record_id": "bank", "instruction": "Describe the red marble rolling across the wooden table",
                 "response": "r", "media_refs": [{"path": str(media / "clip1.mp4"), "kind": "video", "sha256": "y"}]})
    shard = tmp_path / "rows.jsonl"
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows))
    shards_jsonl = tmp_path / "shards.jsonl"
    shards_jsonl.write_text(json.dumps({"path": str(shard)}) + "\n")
    entry_pointer = tmp_path / "entry.json"
    entry_pointer.write_text(json.dumps({"shards_jsonl": str(shards_jsonl)}))
    master = tmp_path / "master.json"
    master.write_text(json.dumps({"entries": [{"pointer": str(entry_pointer), "dataset_id": "fixture", "training_status": "ready_untrained"}]}))
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(["Describe the red marble rolling across the wooden table"]))
    out = tmp_path / "items.jsonl"
    rc = M.main(["manifest", "--route", "video", "--pointer", str(master), "--out", str(out), "--count", "0",
                 "--held-every", "2", "--exclude-bank", str(bank)])
    assert rc == 0
    items = [json.loads(line) for line in out.read_text().splitlines()]
    receipt = json.loads(out.with_suffix(".receipt.json").read_text())
    # clip4 missing on disk; the bank row is carved out AND its clip (clip1) is banned for every row; the image row
    # is another route
    assert receipt["counts"]["missing_file"] == 1
    assert receipt["counts"]["excluded_bank"] == 1
    assert receipt["counts"]["excluded_bank_media"] == 1
    assert len(items) == 4 and {i["video"] for i in items} == {str(media / f"clip{k}.mp4") for k in (0, 2, 3, 5)}
    assert all(i["prompt"].startswith("A short clip number") for i in items)
    assert [i["split"] for i in items].count("held") == 2
    assert receipt["deviations"][0]["name"] == "prompt_not_a_caption"
    assert M.read_items(out, media_key="video")


def test_cli_plan_and_version(capsys):
    assert M.main(["plan", "--teacher", "h3", "--items-count", "512", "--block-rows", "256", "--seconds-per-forward", "3.0"]) == 0
    out = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert out["targets"] == 2048 and out["hours_at_measured_rate"] == round(2048 * 3.0 / 3600, 2)
    assert M.TOOL_VERSION == "1.0.2"


# ----------------------------------------------------------------------------- 1.0.2: runtime signature on resume


def test_runtime_signature_pins_resume_and_seal(tmp_path):
    d = tmp_path / "cache"
    p40 = {"device_requested": "cuda:0", "device_name": "Tesla P40", "capability": "sm_61", "torch": "2.6.0+cu124",
           "comfy_version": "0.30.0", "should_use_bf16": False, "should_use_fp16": False}
    a3090 = dict(p40, device_name="NVIDIA GeForce RTX 3090", capability="sm_86", should_use_bf16=True)
    sig_p40 = M.runtime_signature(p40, {"inference_dtype": "torch.float32", "manual_cast_dtype": "None"})
    sig_3090 = M.runtime_signature(a3090, {"inference_dtype": "torch.bfloat16", "manual_cast_dtype": "torch.bfloat16"})
    assert sig_p40["tool_version"] == M.TOOL_VERSION and sig_p40["tool_sha256"] == M.sha256_file(HERE / "stage_c_media_a.py")
    store = M.ShardStore(d, sig_3090)
    store.write(0, {"v": torch.ones(2)}, {})
    assert store.check_resume() == 1
    assert M.ShardStore(d, M.runtime_signature(a3090)).check_resume(runtime_only=True) == 1  # pre-load check ignores dtypes
    with pytest.raises(SystemExit, match="different runtime"):
        M.ShardStore(d, M.runtime_signature(p40)).check_resume(runtime_only=True)  # refused before any teacher loads
    with pytest.raises(SystemExit, match="different runtime"):
        M.ShardStore(d, sig_p40).check_resume()
    same_card_other_dtype = M.runtime_signature(a3090, {"inference_dtype": "torch.float32", "manual_cast_dtype": "None"})
    with pytest.raises(SystemExit, match="inference_dtype"):
        M.ShardStore(d, same_card_other_dtype).check_resume()
    other_tool = dict(sig_3090, tool_version="1.0.1")
    with pytest.raises(SystemExit, match="tool_version"):
        M.ShardStore(d, other_tool).check_resume()
    # a shard written without a signature (stage_c_media_a <= 1.0.1) is never resumed into
    legacy = tmp_path / "legacy"
    M.ShardStore(legacy, sig_3090).write(0, {"v": torch.ones(2)}, {})
    side = legacy / "shard-00000.json"
    meta = json.loads(side.read_text())
    meta.pop("runtime_signature")
    side.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="< 1.0.2"):
        M.ShardStore(legacy, sig_3090).check_resume()
    with pytest.raises(SystemExit, match="runtime signatures"):
        M.seal_manifest(legacy, {"teacher_integrity": {"unchanged": True}})
    # a directory that somehow holds two runtimes is never sealed
    M.ShardStore(d, sig_p40).write(1, {"v": torch.ones(2)}, {})
    with pytest.raises(SystemExit, match="2 different runtime signatures"):
        M.seal_manifest(d, {"teacher_integrity": {"unchanged": True}})
    (d / "shard-00001.json").unlink()
    (d / "shard-00001.safetensors").unlink()
    M.seal_manifest(d, {"teacher_integrity": {"unchanged": True}, "runtime": {"device_name": "sealing process"}})
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["runtime_signature"] == sig_3090
    assert M.verify_cache(d)["ok"]


# ----------------------------------------------------------------------------- 1.0.2: drops across resumes


def test_environment_errors_are_never_drops():
    import errno

    for exc in (FileNotFoundError(errno.ENOENT, "gone"), OSError(errno.EIO, "I/O error"), OSError(errno.ESTALE, "stale"),
                PermissionError(errno.EACCES, "denied"), MemoryError(), TimeoutError()):
        assert M.is_environment_error(exc), exc
        dropped = []

        def boom(exc=exc):
            raise exc

        with pytest.raises(type(exc)):
            M.decode_or_drop(boom, {"id": "x"}, dropped)
        assert dropped == []
    for exc in (ValueError("clip too short"), OSError("image file is truncated (3 bytes not processed)"), RuntimeError("bad frame")):
        assert not M.is_environment_error(exc), exc
    pil = pytest.importorskip("PIL.Image")
    assert not M.is_environment_error(pil.UnidentifiedImageError("cannot identify image file"))


def test_aggregate_drops_and_marker_supersession(tmp_path):
    d = tmp_path / "cache"
    store = M.ShardStore(d, SIG)
    store.write(0, {"v": torch.ones(1)}, {"items": [{"index": 0}], "dropped": [{"index": 1, "id": "b"}]})
    store.write_empty(1, [{"index": 2, "id": "c"}, {"index": 3, "id": "d"}])
    with pytest.raises(SystemExit, match="2 planned items have no shard record"):
        M.aggregate_drops(d, 6)
    store.write(2, {"v": torch.ones(2)}, {"items": [{"index": 4}, {"index": 5}], "dropped": []})
    agg = M.aggregate_drops(d, 6)
    assert agg["kept"] == 3 and [m["index"] for m in agg["dropped"]] == [1, 2, 3]
    # the retried shard 1 now decodes: its data shard supersedes (deletes) the all-dropped marker
    store.write(1, {"v": torch.ones(1)}, {"items": [{"index": 2}], "dropped": [{"index": 3, "id": "d"}]})
    assert not store.empty_path(1).exists()
    agg = M.aggregate_drops(d, 6)
    assert agg["kept"] == 4 and [m["index"] for m in agg["dropped"]] == [1, 3]
    with pytest.raises(SystemExit, match="outside the plan"):
        M.aggregate_drops(d, 5)


def _png(path, seed):
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (24, 40, 3), dtype=np.uint8)).save(path)


class _FakeVAE:
    """Stands in for comfy's VAE only; decoding (real PIL), sharding, drop accounting and sealing are the shipped code."""

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def encode(self, pixels):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("CUDA error: out of memory")
        h, w = pixels.shape[1] // 8, pixels.shape[2] // 8
        return pixels[:, ::8, ::8, 0].reshape(1, 1, 1, h, w).expand(1, 16, 1, h, w).clone()


def test_qimg_encode_resume_counts_every_drop_and_refuses_another_runtime(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    media = tmp_path / "media"
    media.mkdir()
    rows = []
    for k in range(8):
        path = media / f"img{k}.png"
        if k in (1, 2, 3):
            path.write_bytes(b"not an image %d" % k)  # real PIL UnidentifiedImageError: a decode verdict
        else:
            _png(path, k)
        rows.append({"id": f"i{k}", "prompt": f"picture {k}", "image": str(path)})
    items = _write_items(tmp_path, rows)
    vae_file = tmp_path / "vae.safetensors"
    vae_file.write_bytes(b"teacher" * 50)
    runtime = {"device_requested": "cpu", "device_name": "cpu-test", "torch": torch.__version__, "comfy_version": "test"}
    state = {"vae": _FakeVAE(fail_on_call=2), "loads": 0}

    def fake_load_vae(path):
        state["loads"] += 1
        return state["vae"], {"path": str(path), "vae_dtype": "torch.float32", "device": "cpu"}

    monkeypatch.setattr(M, "common_runtime", lambda args: dict(runtime))
    monkeypatch.setattr(M, "load_vae", fake_load_vae)
    argv = ["qimg-encode", "--items", str(items), "--cache-root", str(tmp_path / "root"), "--width", "64", "--height", "64",
            "--items-per-shard", "2", "--vae", str(vae_file), "--teacher-digest", "full"]
    # run 1: shard 0 keeps i0 and drops i1, shard 1 drops i2 + i3 (marker), the 2nd encode (i4, shard 2) dies on a device fault
    with pytest.raises(RuntimeError, match="out of memory"):
        M.main(argv)
    d = next((tmp_path / "root").glob("qimg-vae/*"))
    assert [p.name for p in M.shard_sidecars(d)] == ["shard-00000.json"] and (d / "shard-00001.empty.json").is_file()
    # 1.0.1 judged the rerun on ITS drops only (2 of 8 = 25%); the cache really misses 3 of 8 = 37.5%
    state["vae"] = _FakeVAE()
    with pytest.raises(SystemExit, match="3 of 8 items dropped"):
        M.main(argv + ["--max-drop-fraction", "0.3"])
    assert not (d / "manifest.json").exists()
    assert M.main(argv + ["--max-drop-fraction", "0.5"]) == 0
    manifest = json.loads((d / "manifest.json").read_text())
    assert [m["id"] for m in manifest["dropped"]] == ["i1", "i2", "i3"] and manifest["kept"] == 5
    assert manifest["drop_check"]["dropped"] == 3 and manifest["drop_check"]["total"] == 8
    assert all(m["dropped"].startswith("UnidentifiedImageError") for m in manifest["dropped"])
    assert manifest["runtime_signature"]["device_name"] == "cpu-test"
    assert manifest["runtime_signature"]["component"] == {"vae_device": "cpu", "vae_dtype": "torch.float32"}
    assert M.verify_cache(d)["ok"]
    # a resume on another card is refused BEFORE the teacher loads
    loads = state["loads"]
    monkeypatch.setattr(M, "common_runtime", lambda args: dict(runtime, device_name="Tesla P40"))
    with pytest.raises(SystemExit, match="different runtime"):
        M.main(argv)
    assert state["loads"] == loads


def test_qimg_encode_missing_media_is_fatal_not_a_drop(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    good = tmp_path / "ok.png"
    _png(good, 1)
    items = _write_items(tmp_path, [{"id": "a", "prompt": "p", "image": str(good)},
                                    {"id": "b", "prompt": "q", "image": str(tmp_path / "unmounted" / "b.png")}])
    vae_file = tmp_path / "vae.bin"
    vae_file.write_bytes(b"w")
    monkeypatch.setattr(M, "common_runtime", lambda args: {"device_name": "cpu-test"})
    monkeypatch.setattr(M, "load_vae", lambda path: (_FakeVAE(), {"vae_dtype": "torch.float32", "device": "cpu"}))
    with pytest.raises(FileNotFoundError):
        M.main(["qimg-encode", "--items", str(items), "--cache-root", str(tmp_path / "root"), "--width", "64", "--height", "64",
                "--items-per-shard", "1", "--vae", str(vae_file), "--teacher-digest", "stat"])
    d = next((tmp_path / "root").glob("qimg-vae/*"))
    assert [p.name for p in M.shard_sidecars(d)] == ["shard-00000.json"]
    assert not list(d.glob("*.empty.json")) and not (d / "manifest.json").exists()
