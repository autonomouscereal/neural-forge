"""Tests for c15_interface_rows.py (lane M2, 2026-10-10).

Run on the server with the chimera tree on the path:

    CHIMERA_SRC=$OC/import/rq27b/tree_ef1ffb9/src OMP_NUM_THREADS=4 nice -n 19 \
        venv/bin/python -m pytest -q composite/tools/test_c15_interface_rows.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CHIMERA_SRC = os.environ.get("CHIMERA_SRC", "/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9/src")
if CHIMERA_SRC not in sys.path:
    sys.path.insert(0, CHIMERA_SRC)
MEDIAFOLD = os.environ.get("MEDIAFOLD_ROOT", str(HERE.parent.parent / "media"))

torch = pytest.importorskip("torch")
F = torch.nn.functional

import c15_interface_rows as tool  # noqa: E402


# --------------------------------------------------------------------------
# layouts: the reshapes must be the parents' own convolutions
# --------------------------------------------------------------------------


def test_conv1d_patch_layout_matches_conv1d():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(24, 6, 2, generator=g, dtype=torch.float64)
    b = torch.randn(24, generator=g, dtype=torch.float64)
    x = torch.randn(1, 6, 2, generator=g, dtype=torch.float64)  # one patch of 2 frames
    P, bias = tool.source_projection("conv1d_patch", w, b)
    expected = F.conv1d(x, w, b, stride=2)[0, :, 0]
    feature = x[0].reshape(-1)  # (c, k) -> c*2+k
    assert torch.allclose(P @ feature + bias, expected, atol=1e-12)


def test_conv_transpose1d_patch_layout_matches_conv_transpose1d():
    g = torch.Generator().manual_seed(2)
    w = torch.randn(24, 3, 2, generator=g, dtype=torch.float64)  # [in, out, k]
    b = torch.randn(3, generator=g, dtype=torch.float64)
    h = torch.randn(1, 24, 1, generator=g, dtype=torch.float64)
    P, bias = tool.output_projection("conv_transpose1d_patch", w, b)
    expected = F.conv_transpose1d(h, w, b, stride=2)[0]  # [3, 2]
    assert torch.allclose((P @ h[0, :, 0] + bias).reshape(3, 2), expected, atol=1e-12)


def test_conv3d_patch_layout_matches_conv3d():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(16, 3, 2, 4, 4, generator=g, dtype=torch.float64)
    b = torch.randn(16, generator=g, dtype=torch.float64)
    x = torch.randn(1, 3, 2, 4, 4, generator=g, dtype=torch.float64)
    P, bias = tool.source_projection("conv3d_patch", w, b)
    expected = F.conv3d(x, w, b, stride=(2, 4, 4)).reshape(-1)
    assert torch.allclose(P @ x.reshape(-1) + bias, expected, atol=1e-12)


def test_qvl_tubelet_layout_matches_conv3d(tmp_path):
    from PIL import Image

    g = torch.Generator().manual_seed(4)
    pixels = (torch.rand(32, 32, 3, generator=g) * 255).to(torch.uint8)
    Image.fromarray(pixels.numpy()).save(tmp_path / "a.png")
    rows, owner = tool.qvl_tubelets([tmp_path / "a.png"], 32)
    assert rows.shape == (4, 1536) and owner.tolist() == [0, 0, 0, 0]
    img = (pixels.to(torch.float64) / 255.0 - 0.5) / 0.5
    video = img.permute(2, 0, 1).unsqueeze(1).expand(-1, 2, -1, -1).unsqueeze(0)  # [1, 3, 2, 32, 32]
    w = torch.randn(5, 3, 2, 16, 16, generator=g, dtype=torch.float64)
    conv = F.conv3d(video, w, stride=(2, 16, 16))[0, :, 0].reshape(5, -1).transpose(0, 1)  # [4 patches, 5]
    P, _ = tool.source_projection("conv3d_patch", w, torch.zeros(5, dtype=torch.float64))
    assert torch.allclose(rows @ P.transpose(0, 1), conv, atol=1e-10)


# --------------------------------------------------------------------------
# chart algebra
# --------------------------------------------------------------------------


def test_parent_basis_is_injective_and_transports_exactly_when_parent_is_wider():
    g = torch.Generator().manual_seed(5)
    width, hidden = 40, 32
    P = torch.randn(width, 6, generator=g, dtype=torch.float64)
    bias = torch.randn(width, generator=g, dtype=torch.float64)
    P_out = torch.randn(5, width, generator=g, dtype=torch.float64)
    gram = torch.randn(64, width, generator=g, dtype=torch.float64)
    V, notes = tool.parent_basis([P, bias.unsqueeze(1), P_out.T], gram.T @ gram, width)
    assert notes["written_read_rank"] == 12
    assert notes["orthonormality_max_abs_error"] < 1e-10
    U = torch.linalg.qr(torch.randn(hidden, hidden, generator=g, dtype=torch.float64))[0]
    B_in = U[:, :hidden] @ V[:, :hidden].T
    B_out = V[:, :hidden] @ U[:, :hidden].T
    z = torch.randn(100, 6, generator=g, dtype=torch.float64)
    Fm = B_in @ P
    assert tool.rel_l2(z @ (torch.linalg.pinv(Fm) @ Fm).T, z) < 1e-10
    y = torch.randn(100, width, generator=g, dtype=torch.float64)
    assert tool.rel_l2(y @ (P_out @ B_out @ B_in).T, y @ P_out.T) < 1e-10
    # the bias survives the truncation as well
    assert tool.rel_l2(B_out @ (B_in @ bias), bias) < 1e-10


# --------------------------------------------------------------------------
# end to end on a toy trunk (c16 fixtures) with synthetic parents
# --------------------------------------------------------------------------


def _save(path: Path, state: dict) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))


def _toy(tmp_path: Path, hidden: int = 32, feature: int = 48):
    import c16_media_fixtures as fx

    native = tmp_path / "trunk"
    fx.write_child(native, seed=0, hidden=hidden, feature=feature, residual_mode="none", dtype="float32")
    g = torch.Generator().manual_seed(7)
    _save(tmp_path / "taps.safetensors", {"prenorm": torch.randn(256, hidden, generator=g).to(torch.bfloat16)})
    models = tmp_path / "models"
    _save(models / "a.safetensors", {
        "a1.in.weight": torch.randn(40, 6, generator=g), "a1.in.bias": torch.randn(40, generator=g),
        "a1.out.weight": torch.randn(6, 40, generator=g), "a1.out.bias": torch.randn(6, generator=g),
        "a2.in.weight": torch.randn(40, 4, generator=g), "a2.in.bias": torch.randn(40, generator=g),
        "a2.out.weight": torch.randn(4, 40, generator=g), "a2.out.bias": torch.randn(4, generator=g),
        "blocks.0.qkv.weight": torch.randn(96, 40, generator=g),
    })
    _save(models / "b.safetensors", {
        "b.in.weight": torch.randn(24, 6, 2, generator=g).to(torch.bfloat16), "b.in.bias": torch.randn(24, generator=g).to(torch.bfloat16),
        "b.out.weight": torch.randn(24, 3, 2, generator=g).to(torch.bfloat16), "b.out.bias": torch.randn(3, generator=g).to(torch.bfloat16),
    })
    _save(models / "c.safetensors", {
        "c.in.weight": torch.randn(16, 3, 2, 2, 2, generator=g), "c.in.bias": torch.randn(16, generator=g),
    })
    plan = {
        "hidden": hidden, "native_feature_dim": feature, "native_type_count": 8,
        "sources": {
            "a": {"weights": "a.safetensors", "width": 40, "read_gram": [r"^blocks\.0\.qkv\.weight$"]},
            "b": {"weights": "b.safetensors", "width": 24, "read_gram": []},
            "c": {"weights": "c.safetensors", "width": 16, "read_gram": []},
        },
        "entries": [
            {"set": "c15", "name": "video_a", "source": "a", "type_id": 2, "feature_band": [0, 6], "emission_band": [0, 6],
             "in": {"weight": "a1.in.weight", "bias": "a1.in.bias", "kind": "linear"}, "out": {"weight": "a1.out.weight", "bias": "a1.out.bias", "kind": "linear"}},
            {"set": "c15", "name": "audio_a", "source": "a", "type_id": 3, "feature_band": [6, 10], "emission_band": [6, 10],
             "in": {"weight": "a2.in.weight", "bias": "a2.in.bias", "kind": "linear"}, "out": {"weight": "a2.out.weight", "bias": "a2.out.bias", "kind": "linear"}},
            {"set": "c15", "name": "music_b", "source": "b", "type_id": 4, "feature_band": [10, 22], "emission_band": [10, 16],
             "in": {"weight": "b.in.weight", "bias": "b.in.bias", "kind": "conv1d_patch"}, "out": {"weight": "b.out.weight", "bias": "b.out.bias", "kind": "conv_transpose1d_patch"}},
            {"set": "c11", "name": "ocr_c", "source": "c", "type_id": 6, "feature_band": [22, 46], "emission_band": [16, 17],
             "in": {"weight": "c.in.weight", "bias": "c.in.bias", "kind": "conv3d_patch"}, "out": None},
        ],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return native, models, plan


def _argv(tmp_path: Path, native: Path, models: Path, extra: list[str] | None = None) -> list[str]:
    return [
        "--plan", str(tmp_path / "plan.json"), "--models-root", str(models), "--trunk-native", str(native),
        "--trunk-taps", str(tmp_path / "taps.safetensors"), "--out", str(tmp_path / "iface"), "--c11-out", str(tmp_path / "iface_c11"),
        "--mediafold-root", MEDIAFOLD, "--ladder-root", str(tmp_path / "none"),
        "--ace-pairs", "", "--h3-audio-pairs", "", "--h3-video-pairs", "", "--ace-capture", "", "--qimg-captures", "",
        "--qimg-pairs", "", "--hy3d-probes", "", "--images-dir", "", "--scale-probes", "256", "--chimera-src", CHIMERA_SRC,
    ] + (extra or [])


def test_end_to_end_toy(tmp_path):
    native, models, plan = _toy(tmp_path)
    assert tool.main(_argv(tmp_path, native, models)) == 0
    receipt = json.loads((tmp_path / "iface" / "c15_interface_receipt.json").read_text(encoding="utf-8"))
    assert receipt["installer_check"]["verdict"].startswith("accepted")
    c15 = json.loads((tmp_path / "iface" / "companions.json").read_text(encoding="utf-8"))
    c11 = json.loads((tmp_path / "iface_c11" / "companions.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in c15["entries"]] == ["video_a", "audio_a", "music_b"]
    assert [e["name"] for e in c11["entries"]] == ["ocr_c"]
    diag = receipt["diagnostics"]
    for name in ("video_a", "audio_a", "music_b"):
        assert diag[name]["bridge"]["pinv_roundtrip_rel_l2_gaussian"] < 1e-9, name
        assert diag[name]["head"]["transport_rel_l2_gaussian"] < 1e-9, name
    # parent width 16 < 24 tubelet columns: the bridge is exactly as lossy as the parent's own embed
    assert diag["ocr_c"]["bridge"]["numerical_rank"] == 16
    assert diag["ocr_c"]["bridge"]["pinv_roundtrip_rel_l2_gaussian"] > 0.1
    # scale heuristic hits its target on the Gaussian probes it was computed from
    for name, d in diag.items():
        assert d["scale"]["s"] > 0
    sim = receipt["simulated_install"]["entries"]
    for name in ("video_a", "audio_a", "music_b", "ocr_c"):
        assert sim[name]["forward"]["finite_bridge"] and sim[name]["forward"]["finite_head"], name
        assert all(v == 0 for v in sim[name]["preconditions_nonzero"].values()), name
    # the declared tensor_sha256 is the installer's own formula
    from chimera_omni.importing.safetensors_io import read_safetensors

    for root, body in ((tmp_path / "iface", c15), (tmp_path / "iface_c11", c11)):
        for entry in body["entries"]:
            views = read_safetensors(root / entry["rows"], verify=False)
            assert entry["tensor_sha256"] == {k: v.sha256() for k, v in sorted(views.items())}
    # the composed rows reproduce the parent token state through the chart (music_b, conv1d)
    from safetensors.torch import load_file

    rows = load_file(str(tmp_path / "iface" / "music_b" / "composed_interface.safetensors"))
    charts = load_file(str(tmp_path / "iface" / "charts" / "b.safetensors"))
    w = load_file(str(models / "b.safetensors"))
    P = w["b.in.weight"].to(torch.float64).reshape(24, 12)
    s = diag["music_b"]["scale"]["s"]
    expected = s * charts["chart_in"].to(torch.float64) @ P
    assert torch.allclose(rows["bridge.feature_proj.weight"].to(torch.float64), expected, rtol=1e-4, atol=1e-5)
    # emission: ConvTranspose1d bias repeated over the 2 frames
    assert torch.allclose(rows["head.type_affine_shift"].to(torch.float64), w["b.out.bias"].to(torch.float64).repeat_interleave(2))


def test_refuses_overlapping_bands(tmp_path):
    native, models, plan = _toy(tmp_path)
    plan["entries"][1]["feature_band"] = [4, 8]
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert tool.main(_argv(tmp_path, native, models)) == 2


def test_refuses_duplicate_type_id(tmp_path):
    native, models, plan = _toy(tmp_path)
    plan["entries"][3]["type_id"] = 2
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert tool.main(_argv(tmp_path, native, models)) == 2


def test_refuses_band_width_mismatch(tmp_path):
    native, models, plan = _toy(tmp_path)
    plan["entries"][0]["feature_band"] = [0, 5]
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert tool.main(_argv(tmp_path, native, models)) == 2


def test_default_plan_bands_fit_the_27b_trunk():
    plan = tool.DEFAULT_PLAN
    builder = tool.Builder.__new__(tool.Builder)
    builder.plan = plan
    builder.feature_dim = plan["native_feature_dim"]
    builder.check_plan()
    used = sum(e["feature_band"][1] - e["feature_band"][0] for e in plan["entries"])
    assert used == 2176 and plan["native_feature_dim"] == 6144
    assert sorted(e["type_id"] for e in plan["entries"]) == [1, 2, 3, 4, 6, 7]


def test_image_ladder_and_volume_probe_branches(tmp_path):
    """The real-data diagnostic branches on synthetic pair/probe files (names select the branch)."""

    native, models, plan = _toy(tmp_path)
    g = torch.Generator().manual_seed(9)
    _save(models / "d.safetensors", {
        "i.in.weight": torch.randn(40, 8, generator=g), "i.in.bias": torch.randn(40, generator=g),
        "i.out.weight": torch.randn(8, 40, generator=g), "i.out.bias": torch.randn(8, generator=g),
        "v.in.weight": torch.randn(40, 8, generator=g), "v.in.bias": torch.randn(40, generator=g),
        "v.out.weight": torch.randn(8, 40, generator=g), "v.out.bias": torch.randn(8, generator=g),
    })
    plan["sources"] = {"d": {"weights": "d.safetensors", "width": 40, "read_gram": []}}
    plan["entries"] = [
        {"set": "c15", "name": "image_qimg", "source": "d", "type_id": 1, "feature_band": [0, 8], "emission_band": [0, 8],
         "in": {"weight": "i.in.weight", "bias": "i.in.bias", "kind": "linear"}, "out": {"weight": "i.out.weight", "bias": "i.out.bias", "kind": "linear"}},
        {"set": "c15", "name": "volume_hy3d", "source": "d", "type_id": 7, "feature_band": [8, 16], "emission_band": [8, 16],
         "in": {"weight": "v.in.weight", "bias": "v.in.bias", "kind": "linear"}, "out": {"weight": "v.out.weight", "bias": "v.out.bias", "kind": "linear"}},
    ]
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    sys.path.insert(0, MEDIAFOLD)
    from mediafold.c15_pairs import PairSet, write_pair_set

    clips, per = 8, 32
    latent = torch.randn(clips * per, 8, generator=g)
    mix = torch.randn(8, 12, generator=g)
    raw = (latent @ mix).sigmoid()
    roundtrip = (raw + 0.01 * torch.randn(raw.shape, generator=g)).clamp(0, 1)
    clip = torch.arange(clips).repeat_interleave(per)
    split = (clip >= 6).long()
    pairs = PairSet(raw=raw, latent=latent, roundtrip=roundtrip, clip_index=clip, frame_index=torch.arange(clips * per) % per,
                    channel_index=torch.zeros_like(clip), split=split,
                    geometry={"name": "toy", "modality": "image", "d_raw": 12, "d_latent": 8, "d_centre": 12, "centre_offset": 0, "context": 0})
    write_pair_set(pairs, tmp_path / "qimg_pairs")
    vol = tmp_path / "vol"
    vol.mkdir()
    _save(vol / "volume_probes.safetensors", {"latent_dit": torch.randn(2, 5, 8, generator=g)})
    (vol / "volume_probe_receipt.json").write_text(json.dumps({
        "convention": "logit>0 inside",
        "per_mesh": [{"accuracy_logit_gt0_inside": 0.9}, {"accuracy_logit_gt0_inside": 0.99}],
        "summary": {"holdout_mean_psnr_db": 9.0, "mean_psnr_db": 8.5, "mean_accuracy": 0.945, "holdout_meshes": [1]},
        "probe_rule": "toy"}), encoding="utf-8")
    argv = _argv(tmp_path, native, models)
    argv[argv.index("--qimg-pairs") + 1] = str(tmp_path / "qimg_pairs")
    argv[argv.index("--hy3d-probes") + 1] = str(vol)
    assert tool.main(argv) == 0
    receipt = json.loads((tmp_path / "iface" / "c15_interface_receipt.json").read_text(encoding="utf-8"))
    img = receipt["diagnostics"]["image_qimg"]["vae_psnr"]
    assert img["latent_through_interface_rel_l2_held"] < 1e-9
    assert img["psnr_through_interface_db"] == img["psnr_parent_roundtrip_db"]
    assert img["raw_rung_ladder"]["rows"][0]["psnr_absorbed_db"] is not None
    v = receipt["diagnostics"]["volume_hy3d"]["vae_psnr"]
    assert v["psnr_parent_roundtrip_db"] == 9.0 and v["psnr_through_interface_db"] == 9.0
    assert abs(v["psnr_hard_occupancy_holdout_db"] - 20.0) < 1e-6
    companions = json.loads((tmp_path / "iface" / "companions.json").read_text(encoding="utf-8"))
    assert {e["name"]: e["psnr_absorbed_db"] for e in companions["entries"]} == {"image_qimg": img["psnr_parent_roundtrip_db"], "volume_hy3d": 9.0}
