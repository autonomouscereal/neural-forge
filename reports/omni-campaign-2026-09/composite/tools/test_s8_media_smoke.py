"""Tests for s8_media_smoke.py / s8_media_decode.py pure helpers (CPU, no checkpoint).

Run on the server:
  CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src $PY -m pytest -q test_s8_media_smoke.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_t", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s8 = load("s8_media_smoke")
dec = load("s8_media_decode")
s7p = load("s7_pairs_parent")


def test_ace_emission_layout_inverts_the_feature_packing():
    lat = torch.randn(100, 64)
    # pack exactly like s7_pairs_parent.ace_token_features does (c*2 + k over the 2 frames of a token)
    rows = s7p.ace_token_features(torch.zeros(1, 100, 0), lat.unsqueeze(0))[0]
    assert rows.shape == (50, 128)
    assert torch.equal(s8.ace_emission_to_latent(rows), lat)


def test_qimg_unpack_inverts_pack():
    x = torch.randn(1, 16, 1, 64, 64)
    rows = s7p.qimg_pack(x)[0]
    assert torch.equal(s8.qimg_unpack(rows, 1, 64, 64), x)


def test_h3_video_and_audio_unpack_invert_pack():
    v = torch.randn(1, 24, 4, 4, 6)
    rows = s7p.h3_patchify_video(v)
    assert torch.equal(s8.h3_unpatchify_video(rows, 4, 2, 3), v)
    a = torch.randn(1, 32, 2, 22)
    assert torch.equal(s8.h3_unpack_audio(s7p.h3_pack_audio(a)), a)


def test_h3_source_positions_match_comfy_layout_by_hand():
    pos = s8.h3_source_positions(text_len=3, latent_t=4, latent_h=4, latent_w=4, audio_t=2)
    assert pos.shape == (3 + 4 + 4 * 4, 3)
    assert pos[:3, 0].tolist() == [0.0, 1.0, 2.0] and float(pos[:3, 1:].abs().sum()) == 0.0
    # audio: t = 3 + [0, 1] for each stereo channel, w pinned to the frame grid extremes (0 and 16), h = 0
    assert pos[3:7, 0].tolist() == [3.0, 4.0, 3.0, 4.0]
    assert pos[3:5, 2].tolist() == [0.0, 0.0] and pos[5:7, 2].tolist() == [16.0, 16.0]
    # video: 2x2 patch grid of a 4x4 latent -> (h, w) in {0, 16}; t = 3 + exclusive cumsum of 5/3 * (1, 4, 4, 4)
    video = pos[7:].view(4, 4, 3)
    expected_t = [3.0, 3.0 + 5 / 3, 3.0 + 5 * 5 / 3, 3.0 + 9 * 5 / 3]
    assert torch.allclose(video[:, 0, 0], torch.tensor(expected_t, dtype=torch.float64))
    assert video[0, :, 1:].tolist() == [[0.0, 0.0], [0.0, 16.0], [16.0, 0.0], [16.0, 16.0]]


def test_solve_clean_shared_noise_is_exact():
    z, n = torch.randn(1, 16, 1, 8, 8), torch.randn(1, 16, 1, 8, 8)
    sig = [0.15, 0.35, 0.65, 0.9]
    xs = torch.stack([(1 - s) * z + s * n for s in sig])
    zz, nn, resid = s8.solve_clean_shared_noise(xs, sig)
    assert torch.allclose(zz, z, atol=1e-5) and torch.allclose(nn, n, atol=1e-5) and resid < 1e-6


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_ideal_raw_reconstructs_clean_and_metrics(sign):
    clean, noise, sigma = torch.randn(1, 64, 50), torch.randn(1, 64, 50), 0.4
    x_t = (1 - sigma) * clean + sigma * noise
    ideal = (clean - x_t) / (sign * sigma)
    assert torch.allclose(s8.x0_from_raw(x_t, ideal, sigma, sign), clean, atol=1e-5)
    m = s8.latent_metrics(x_t, clean, ideal, sigma, sign, parent_raw=ideal)
    assert m["rel_l2_x0_vs_clean"] < 1e-5 and abs(m["cos_raw_vs_ideal"] - 1) < 1e-9 and abs(m["oracle_scalar_gain"] - 1) < 1e-6
    assert abs(m["cos_raw_vs_parent"] - 1) < 1e-9
    zero = s8.latent_metrics(x_t, clean, torch.zeros_like(x_t), sigma, sign)
    assert abs(zero["rel_l2_x0_vs_clean"] - zero["rel_l2_xt_vs_clean_v0_baseline"]) < 1e-12 and zero["raw_nonzero"] is False


def test_best_sign_finds_the_flow_convention():
    clean, noise, sigma = torch.randn(4, 64), torch.randn(4, 64), 0.5
    x_t = (1 - sigma) * clean + sigma * noise
    assert s8.best_sign(x_t, clean, noise - clean, sigma)["better"] == -1.0   # velocity n - z  -> x0 = x - s v
    assert s8.best_sign(x_t, clean, clean - noise, sigma)["better"] == 1.0    # raw = z - n (comfy negates) -> x0 = x + s raw


def test_sign_table_is_the_documented_one():
    assert s8.SIGN == {"ace": -1.0, "qimg": -1.0, "hy3d": 1.0, "h3_video": 1.0, "h3_audio": 1.0}


def test_decode_metrics():
    a = torch.rand(8, 8, 3)
    assert dec.psnr(a, a) == float("inf")
    assert abs(dec.psnr(torch.zeros(4), torch.full((4,), 0.1)) - 20.0) < 1e-5
    x = torch.randn(2, 4800)
    assert abs(dec.snr_db(x * 0.0, x) - 0.0) < 1e-9
    assert abs(dec.stft_cos(x, x) - 1.0) < 1e-9
    g = torch.zeros(4, 4, 4, dtype=torch.bool)
    h = g.clone()
    g[0], h[0, :2] = True, True
    assert abs(dec.iou(g, h) - 0.5) < 1e-12 and dec.iou(g & False, h & False) is None
    assert dec.frames_strip(torch.rand(3, 3, 5, 7)).shape == (5, 21, 3)


def test_verdict_vs_baseline():
    m = {"x_t": {"psnr_vs_clean": 10.0}, "x0_omni_final": {"psnr_vs_clean": 9.0}, "x0_parent": {"psnr_vs_clean": 20.0}}
    v = dec.verdict_vs_baseline(m, "psnr_vs_clean")
    assert v["x0_omni_final"]["beats_noisy_input"] is False and v["x0_parent"]["beats_noisy_input"] is True
