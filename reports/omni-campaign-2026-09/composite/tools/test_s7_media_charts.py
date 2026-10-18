"""Tests for the lane-M1 media-chart tools (s7_pairs_parent / s7_interface_latent / s7_chart_fit).

Run on the server with the chimera tree bound:
    CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9 PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src \
      OMP_NUM_THREADS=4 nice -n 19 venv/bin/python -m pytest -q composite/tools/test_s7_media_charts.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_t", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_t"] = module
    spec.loader.exec_module(module)
    return module


pp = _load("s7_pairs_parent")


def test_qimg_pack_matches_comfy_process_img():
    x = torch.randn(1, 16, 1, 8, 6)
    got = pp.qimg_pack(x)
    # comfy process_img literal
    b, c, t, h, w = x.shape
    ref = x.view(b, c, t, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(b, t * (h // 2) * (w // 2), c * 4)
    assert torch.equal(got, ref)
    # img_in on the packed rows == the per-patch linear the parent applies
    W = torch.randn(7, 64)
    assert torch.allclose(got[0] @ W.T, ref[0] @ W.T)


def test_h3_patchify_roundtrip_and_audio_order():
    v = torch.randn(1, 24, 3, 4, 6)
    rows = pp.h3_patchify_video(v)
    assert rows.shape == (3 * 2 * 3, 96)
    # unpatchify (comfy literal) restores the latent
    t, h, w = 3, 2, 3
    back = torch.einsum("nthwcrpq->nctrhpwq", rows.reshape(1, t, h, w, 24, 1, 2, 2)).reshape(1, 24, 3, 4, 6)
    assert torch.equal(back, v)
    a = torch.randn(1, 32, 2, 5)
    packed = pp.h3_pack_audio(a)
    assert packed.shape == (10, 32)
    assert torch.equal(packed[0], a[0, :, 0, 0]) and torch.equal(packed[5], a[0, :, 1, 0]) and torch.equal(packed[6], a[0, :, 1, 1])


def test_ace_features_reproduce_conv1d_proj_in():
    ctx = torch.randn(2, 10, 128, dtype=torch.float64)
    hid = torch.randn(2, 10, 64, dtype=torch.float64)
    conv = torch.nn.Conv1d(192, 16, kernel_size=2, stride=2).double()
    ref = conv(torch.cat((ctx, hid), -1).transpose(1, 2)).transpose(1, 2)
    feats = pp.ace_token_features(ctx, hid)
    got = feats @ conv.weight.reshape(16, -1).T + conv.bias
    assert torch.allclose(got, ref, atol=1e-10)


def test_qwen_vl_patchify_matches_conv3d_patch_embed():
    pixels = torch.randn(3, 64, 96, dtype=torch.float64)
    flat, grid, yx = pp.qwen_vl_patchify(pixels)
    assert flat.shape == (4 * 6, 1536) and grid.tolist() == [[1, 4, 6]]
    conv = torch.nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(2, 16, 16)).double()
    # transformers Qwen3VLVisionPatchEmbed: view(-1, C, T, P, P) -> conv3d
    ref = conv(flat.view(-1, 3, 2, 16, 16)).view(-1, 8)
    got = flat @ conv.weight.reshape(8, -1).T + conv.bias
    assert torch.allclose(got, ref, atol=1e-9)
    # row r is patch (y, x): its pixels are the image block at that grid cell
    r = 7
    y, x = yx[r].tolist()
    block = pixels[:, y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16]
    assert torch.equal(flat[r].view(3, 2, 16, 16)[:, 0], block)
    pos = pp.image_positions(yx, 4, 6)
    assert pos[:, 1].max() == 1.0 and pos[:, 2].max() == 1.0 and pos[:, 0].abs().max() == 0


def test_interface_projections_match_parent_layers():
    il = _load("s7_interface_latent")
    from safetensors.torch import save_file
    tmp = Path(os.environ.get("PYTEST_TMP", "/tmp")) / "m1_iface_test.safetensors"
    convt = torch.nn.ConvTranspose1d(12, 5, kernel_size=2, stride=2).double()
    conv = torch.nn.Conv1d(6, 12, kernel_size=2, stride=2).double()
    save_file({"o.w": convt.weight.float().contiguous(), "o.b": convt.bias.float().contiguous(),
               "i.w": conv.weight.float().contiguous(), "i.b": conv.bias.float().contiguous()}, str(tmp))
    P_src, b_src = il.source_projection(str(tmp), ("i.w", "i.b", "conv1d_k2"))
    P_out, b_out = il.output_projection(str(tmp), ("o.w", "o.b", "convt1d_k2"), 12)
    convt = convt.float().double(); conv = conv.float().double()
    hidden = torch.randn(1, 12, 3, dtype=torch.float64)  # [B, W, tokens]
    ref = convt(hidden)  # [1, 5, 6]: token t -> frames 2t, 2t+1
    got = hidden[0].T @ P_out.T + b_out  # [tokens, 10] in (c*2 + k) order
    assert torch.allclose(got.reshape(3, 5, 2), ref[0].reshape(5, 3, 2).permute(1, 0, 2), atol=1e-6)
    frames = torch.randn(1, 6, 4, dtype=torch.float64)
    ref_in = conv(frames)[0].T
    feats = frames[0].T.reshape(2, 2, 6).permute(0, 2, 1).reshape(2, 12)
    assert torch.allclose(feats @ P_src.T + b_src, ref_in, atol=1e-6)


def test_pca_directions_orthonormal_and_ordered():
    il = _load("s7_interface_latent")
    g = torch.Generator().manual_seed(0)
    rows = torch.randn(400, 6, generator=g, dtype=torch.float64) * torch.tensor([5.0, 3, 2, 1, 0.5, 0.1], dtype=torch.float64)
    d, notes = il.pca_directions(rows, 4)
    assert d.shape == (4, 6)
    assert torch.allclose(d @ d.T, torch.eye(4, dtype=torch.float64), atol=1e-10)
    assert abs(d[0, 0]) > 0.95 and notes["retained_energy"] > 0.98


def test_chart_fit_writes_c16_loadable_container(tmp_path):
    src = os.environ.get("CHIMERA_OMNI_SRC")
    if not src:
        pytest.skip("CHIMERA_OMNI_SRC not set")
    if str(Path(src) / "src") not in sys.path:
        sys.path.insert(0, str(Path(src) / "src"))
    fixtures = _load("c16_media_fixtures")
    c16 = _load("c16_install_media")
    cf = _load("s7_chart_fit")
    hidden = 32
    root = tmp_path / "unit"
    fixtures.write_block_container(root, "qvl32_vit_Lmid", hidden=hidden, with_charts=False)
    unit = root / "qvl32_vit_Lmid" if (root / "qvl32_vit_Lmid").is_dir() else root
    body = json.loads((unit / "charted_block.json").read_text())
    width = int(body["spec"]["width"])
    g = torch.Generator().manual_seed(3)
    n = 600
    host = torch.randn(n, hidden, generator=g)
    parent = host @ torch.randn(hidden, width, generator=g) * 0.3 + 0.01 * torch.randn(n, width, generator=g)
    group = torch.arange(n) // 50
    held = group % 4 == 3
    from safetensors.torch import save_file
    rows_dir = tmp_path / "tap" / "hy3d_dit_L10"
    rows_dir.mkdir(parents=True)
    save_file({"host_rows": host, "parent_rows": parent, "held": held, "group": group, "type_ids": torch.full((n,), 7),
               "forward": group.clone(), "token": torch.arange(n), "sigma": torch.full((n,), 0.5),
               "bridge_rows_sample": torch.randn(80, hidden, generator=g), "bridge_rows_index": torch.arange(80)}, str(rows_dir / "paired_rows.safetensors"))
    code = cf.main(["--unit", "hy3d_dit_L10", "--unit-dir", str(unit), "--tap-rows", str(tmp_path / "tap"), "--hidden", str(hidden),
                    "--chimera-src", src, "--type-ids", "6"])
    assert code == 0
    cfg = fixtures.child_config(hidden=hidden)
    container = c16.load_container(unit, cfg)
    assert container["chart_source"] == "container"
    assert container["tensor_sha256_check"]["charts.safetensors"]["verified"] is True
    assert tuple(container["charts"]["chart_in.weight"].shape) == (width, hidden)
    assert tuple(container["charts"]["chart_out.weight"].shape) == (hidden, width)
    receipt = json.loads((unit / "chart_receipt.json").read_text())
    held_fwd = receipt["diagnostics"]["ridge_split_held"]["forward"]
    assert held_fwd["cos"] > 0.95  # a near-linear synthetic pair is recovered
    assert (unit / "charted_block.json.pre-m1").is_file()
    assert json.loads((unit / "charted_block.json").read_text())["spec"]["type_ids"] == [6]
    # a stale digest is refused by c16 (provenance travels with the file)
    body = json.loads((unit / "charted_block.json").read_text())
    body["tensor_sha256"]["charts.safetensors"]["chart_in.bias"] = "0" * 64
    (unit / "charted_block.json").write_text(json.dumps(body))
    with pytest.raises(c16.Refusal):
        c16.load_container(unit, cfg)


def test_chart_fit_refuses_shared_groups(tmp_path):
    src = os.environ.get("CHIMERA_OMNI_SRC")
    if not src:
        pytest.skip("CHIMERA_OMNI_SRC not set")
    if str(Path(src) / "src") not in sys.path:
        sys.path.insert(0, str(Path(src) / "src"))
    fixtures = _load("c16_media_fixtures")
    cf = _load("s7_chart_fit")
    root = tmp_path / "unit"
    fixtures.write_block_container(root, "qvl32_vit_Lmid", hidden=16, with_charts=False)
    unit = root / "qvl32_vit_Lmid" if (root / "qvl32_vit_Lmid").is_dir() else root
    width = int(json.loads((unit / "charted_block.json").read_text())["spec"]["width"])
    n = 200
    from safetensors.torch import save_file
    d = tmp_path / "tap" / "hy3d_dit_L10"
    d.mkdir(parents=True)
    held = torch.zeros(n, dtype=torch.bool)
    held[::7] = True
    save_file({"host_rows": torch.randn(n, 16), "parent_rows": torch.randn(n, width), "held": held, "group": torch.zeros(n, dtype=torch.long),
               "type_ids": torch.full((n,), 7)}, str(d / "paired_rows.safetensors"))
    with pytest.raises(SystemExit):
        cf.main(["--unit", "hy3d_dit_L10", "--unit-dir", str(unit), "--tap-rows", str(tmp_path / "tap"), "--hidden", "16", "--chimera-src", src])


def test_direct_composition_equals_compose_interface_square_case():
    """s7_interface_latent writes compose_interface's formulas directly; check them on a case compose accepts."""
    media = Path(os.environ.get("OC", "/home/cereal/omni-campaign")) / "media"
    if not (media / "mediafold").is_dir():
        pytest.skip("mediafold not on this box")
    sys.path.insert(0, str(media))
    from mediafold.c15_fit import AbsorptionMaps, compose_interface
    from mediafold.linalg import LinearMap
    g = torch.Generator().manual_seed(1)
    d, W, H = 6, 10, 12
    P_src = torch.randn(W, d, generator=g, dtype=torch.float64)
    P_out = torch.randn(d, W, generator=g, dtype=torch.float64)
    B_in = torch.randn(H, W, generator=g, dtype=torch.float64)
    B_out = torch.randn(W, H, generator=g, dtype=torch.float64)
    out_bias = torch.randn(d, generator=g, dtype=torch.float64)
    maps = AbsorptionMaps(encoder=LinearMap(torch.eye(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)),
                          decoder=LinearMap(torch.eye(d, dtype=torch.float64), out_bias))
    c = compose_interface(maps, hidden=H, source_projection=P_src, output_projection=P_out, chart_in=B_in, chart_out=B_out)
    assert torch.allclose(c.feature_proj_weight, B_in @ P_src)
    assert torch.allclose(c.emission_out_weight, P_out @ B_out)
    assert torch.allclose(c.emission_shift, out_bias)


def test_host_rows_interface_selection(tmp_path):
    hr = _load("s7_host_rows")
    a = tmp_path / "a"
    c = tmp_path / "c"
    a.mkdir()
    c.mkdir()
    (a / "companions.json").write_text(json.dumps({"entries": [
        {"name": "video_h3", "type_id": 2, "feature_band": [0, 4], "rows": "x"}, {"name": "audio_h3", "type_id": 3, "feature_band": [4, 6], "rows": "y"},
        {"name": "image_qimg", "type_id": 1, "feature_band": [6, 8], "rows": "z"}]}))
    (c / "companions.json").write_text(json.dumps({"entries": [{"name": "ocr_qvl32", "type_id": 6, "feature_band": [8, 9], "rows": "w", "unit": "qvl32_vit_L13"}]}))
    got = hr.interface_entries([a, c], "h3_dit_L25_curve6", {2, 3})
    assert sorted(e["name"] for e in got) == ["audio_h3", "video_h3"] and {e["_selected_by"] for e in got} == {"type_id"}
    got = hr.interface_entries([a, c], "qvl32_vit_L13", {6})
    assert [e["name"] for e in got] == ["ocr_qvl32"] and got[0]["_selected_by"] == "unit"
    with pytest.raises(SystemExit):
        hr.interface_entries([a, c], "hy3d_dit_L10", {7})


# ---------------------------------------------------------------- PREP-B: the HY3D time-token row


def test_hy3d_segment_keeps_row0_as_the_first_time_row():
    g = torch.Generator().manual_seed(5)
    n, W = 6, 8
    block_input = torch.randn(n + 1, W, generator=g)
    latent = torch.randn(n, 64, generator=g)
    pos = torch.rand(n, 4, generator=g)
    seg = pp.hy3d_segment(block_input, latent, pos)
    assert seg["features"].shape == (n + 1, 64) and bool(seg["features"][0].eq(0).all())
    assert torch.equal(seg["features"][1:], latent)
    assert seg["widths"].tolist() == [0] + [64] * n
    assert torch.equal(seg["parent"], block_input)  # row 0 = the time token, kept and first
    assert seg["positions"][0].tolist() == list(pp.TIME_ROW_POSITION) and torch.equal(seg["positions"][1:], pos)
    assert seg["time_row"].tolist() == [True] + [False] * n
    old = pp.hy3d_segment(block_input, latent, pos, time_row=False)  # the 1.0.0 layout
    assert torch.equal(old["parent"], block_input[1:]) and not bool(old["time_row"].any())
    with pytest.raises(SystemExit):
        pp.hy3d_segment(block_input[1:], latent, pos)


def test_pair_writer_records_and_polices_time_rows(tmp_path):
    g = torch.Generator().manual_seed(6)
    n = 5
    seg = pp.hy3d_segment(torch.randn(n + 1, 8, generator=g), torch.randn(n, 64, generator=g), torch.rand(n, 4, generator=g))
    kw = dict(prompt="p", parent_rows=seg["parent"], type_ids=torch.full((n + 1,), 7), sigma=torch.full((n + 1,), 0.5),
              positions=seg["positions"], group=0, held=False)
    w = pp.PairWriter("hy3d_dit_L10", tmp_path, feature_width=64, parent_width=8, parent_dtype=torch.float16)
    with pytest.raises(SystemExit):  # a "time row" that is not first
        w.add_forward(features=seg["features"], meta={"_widths": seg["widths"].clone()}, time_row=seg["time_row"].roll(1), **kw)
    content = seg["features"].clone()
    content[0, 3] = 1.0  # a time row with feature content
    w2 = pp.PairWriter("hy3d_dit_L10", tmp_path, feature_width=64, parent_width=8, parent_dtype=torch.float16)
    with pytest.raises(SystemExit):
        w2.add_forward(features=content, meta={"_widths": seg["widths"].clone()}, time_row=seg["time_row"], **kw)
    w3 = pp.PairWriter("hy3d_dit_L10", tmp_path, feature_width=64, parent_width=8, parent_dtype=torch.float16)
    w3.add_forward(features=seg["features"], meta={"_widths": seg["widths"].clone()}, time_row=seg["time_row"], **kw)
    receipt = w3.write({})
    assert receipt["rows_time"] == 1 and receipt["forwards"][0]["time_rows"] == 1
    from safetensors import safe_open
    with safe_open(str(tmp_path / "hy3d_dit_L10" / "parent_capture.safetensors"), "pt") as h:
        assert h.get_tensor("time_row").tolist() == [True] + [False] * n
        assert h.get_tensor("feature_width").tolist() == [0] + [64] * n


def _time_pairs(forwards: int = 3, n: int = 4):
    rows = forwards * (n + 1)
    forward = torch.arange(forwards).repeat_interleave(n + 1)
    token = torch.arange(n + 1).repeat(forwards)
    time_row = token == 0
    features = torch.randn(rows, 64)
    features[time_row] = 0
    widths = torch.where(time_row, 0, 64)
    positions = torch.rand(rows, 4)
    positions[time_row] = -1.0
    sigma = torch.full((rows,), 0.5)
    return dict(time_row=time_row, forward=forward, token=token, type_ids=torch.full((rows,), 7), features=features,
                widths=widths, positions=positions, sigma=sigma)


def test_host_rows_time_row_contract_is_structural():
    hr = _load("s7_host_rows")
    good = _time_pairs()
    report = hr.check_time_rows(**good, owned_types=[7], required=True)
    assert report["time_rows"] == 3 and report["forwards_with_time_row"] == 3
    # a unit that requires the row refuses pairs without it
    none = dict(good, time_row=torch.zeros_like(good["time_row"]))
    with pytest.raises(SystemExit):
        hr.check_time_rows(**none, owned_types=[7], required=True)
    assert hr.check_time_rows(**none, required=False)["time_rows"] == 0
    for breach in ("not_first", "position", "sigma", "type", "content"):
        bad = {k: v.clone() for k, v in good.items()}
        if breach == "not_first":
            bad["time_row"] = bad["token"] == 1
        elif breach == "position":
            bad["positions"][0] = 0.0
        elif breach == "sigma":
            bad["sigma"][0] = 0.9
        elif breach == "type":
            bad["type_ids"][0] = 6
        else:
            bad["features"][0, 0] = 1.0
        with pytest.raises(SystemExit):
            hr.check_time_rows(**bad, owned_types=[7], required=True)


def test_host_rows_reads_the_requirement_from_the_unit(tmp_path):
    hr = _load("s7_host_rows")
    unit = tmp_path / "hy3d" / "hy3d_dit_L10"
    unit.mkdir(parents=True)
    (unit / "charted_block.json").write_text(json.dumps({"spec": {"type_ids": [7]}, "runtime_requirements": {"hy3d_time_token_row": {"layout": "x"}}}))
    req = hr.unit_time_row_requirement("hy3d_dit_L10", tmp_path)
    assert req is not None and req["type_ids"] == [7]
    assert hr.unit_time_row_requirement("ace_dit_L12", tmp_path) is None  # no unit json there
    (unit / "charted_block.json").write_text(json.dumps({"spec": {"type_ids": [7]}}))
    assert hr.unit_time_row_requirement("hy3d_dit_L10", tmp_path) is None


def test_replicate_rows_weights_only_the_extra_rows():
    cf = _load("s7_chart_fit")
    base = torch.tensor([True, True, False, True])
    extra = torch.tensor([False, True, False, False])
    assert cf.replicate_rows(base, extra, 1).tolist() == [0, 1, 3]
    assert cf.replicate_rows(base, extra, 4).tolist() == [0, 1, 3, 1, 1, 1]


def _time_unit(tmp_path, hidden, *, requirement=True):
    fixtures = _load("c16_media_fixtures")
    root = tmp_path / "unit"
    fixtures.write_block_container(root, "qvl32_vit_Lmid", hidden=hidden, with_charts=False)
    unit = root / "qvl32_vit_Lmid"
    body = json.loads((unit / "charted_block.json").read_text())
    if requirement:
        body["runtime_requirements"] = {"hy3d_time_token_row": {"layout": "[1 time row | latent rows]"}}
    (unit / "charted_block.json").write_text(json.dumps(body))
    return unit, int(body["spec"]["width"])


def _time_rows_file(tmp_path, hidden, width, *, with_time=True, seed=4):
    g = torch.Generator().manual_seed(seed)
    forwards, n = 8, 60
    rows = forwards * (n + 1)
    token = torch.arange(n + 1).repeat(forwards)
    time_row = (token == 0) if with_time else torch.zeros(rows, dtype=torch.bool)
    group = torch.arange(forwards).repeat_interleave(n + 1)
    held = group % 4 == 3
    host = torch.randn(rows, hidden, generator=g)
    host[:, -1] = 0
    t = token == 0
    host[t] = torch.randn(int(t.sum()), hidden, generator=g) * 0.05
    host[t, -1] = 8.0  # the trunk time row lives in a direction no content row uses
    M = torch.randn(hidden, width, generator=g) * 0.3
    parent = host @ M + 0.01 * torch.randn(rows, width, generator=g)
    parent[t] = 5.0 + 0.01 * torch.randn(int(t.sum()), width, generator=g)  # its own target
    d = tmp_path / "tap" / "hy3d_dit_L10"
    d.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({"host_rows": host, "parent_rows": parent, "held": held, "group": group, "type_ids": torch.full((rows,), 7),
               "forward": group.clone(), "token": token, "sigma": torch.full((rows,), 0.5), "time_row": time_row,
               "bridge_rows_sample": torch.randn(80, hidden, generator=g), "bridge_rows_index": torch.arange(80)},
              str(d / "paired_rows.safetensors"))
    return tmp_path / "tap"


def _bind_tree():
    src = os.environ.get("CHIMERA_OMNI_SRC")
    if not src:
        pytest.skip("CHIMERA_OMNI_SRC not set")
    if str(Path(src) / "src") not in sys.path:
        sys.path.insert(0, str(Path(src) / "src"))
    return src


def test_chart_fit_refuses_rows_without_the_required_time_row(tmp_path):
    src = _bind_tree()
    cf = _load("s7_chart_fit")
    hidden = 24
    unit, width = _time_unit(tmp_path, hidden)
    tap = _time_rows_file(tmp_path, hidden, width, with_time=False)
    with pytest.raises(SystemExit):
        cf.main(["--unit", "hy3d_dit_L10", "--unit-dir", str(unit), "--tap-rows", str(tap), "--hidden", str(hidden), "--chimera-src", src, "--arms", "ridge"])
    # and the converse: time rows into a unit that declares no time-row requirement
    unit2, width2 = _time_unit(tmp_path / "b", hidden, requirement=False)
    tap2 = _time_rows_file(tmp_path / "b", hidden, width2, with_time=True)
    with pytest.raises(SystemExit):
        cf.main(["--unit", "hy3d_dit_L10", "--unit-dir", str(unit2), "--tap-rows", str(tap2), "--hidden", str(hidden), "--chimera-src", src, "--arms", "ridge"])


def test_chart_fit_time_rows_are_fitted_and_weighted(tmp_path):
    src = _bind_tree()
    cf = _load("s7_chart_fit")
    c16 = _load("c16_install_media")
    fixtures = _load("c16_media_fixtures")
    hidden = 24
    results = {}
    for weight in (1, 64):
        base = tmp_path / f"w{weight}"
        unit, width = _time_unit(base, hidden)
        tap = _time_rows_file(base, hidden, width)
        code = cf.main(["--unit", "hy3d_dit_L10", "--unit-dir", str(unit), "--tap-rows", str(tap), "--hidden", str(hidden),
                        "--chimera-src", src, "--arms", "ridge", "--time-row-weight", str(weight)])
        assert code == 0
        receipt = json.loads((unit / "chart_receipt.json").read_text())
        block = json.loads((unit / "charted_block.json").read_text())["charts"]
        assert block["time_rows"]["weight"] == weight and block["time_rows"]["rows"] == 8 and block["time_rows"]["rows_held"] == 2
        assert block["arms"] == ["ridge"]
        assert receipt["diagnostics"]["time_rows"]["split_ridge_held_forward"]["time"]["rows"] == 2
        assert c16.load_container(unit, fixtures.child_config(hidden=hidden))["chart_source"] == "container"
        results[weight] = receipt["diagnostics"]["time_rows"]["split_ridge_held_forward"]["time"]["rel_l2"]
    assert results[64] < results[1]  # weighting the 1-per-forward time rows moves their held fit


def test_chart_fit_default_time_row_weight_and_weight_choice_record(tmp_path):
    """1.2.0: the default weight is the PREP-B sweep's choice, and the sweep receipt is recorded (never a gate)."""

    src = _bind_tree()
    cf = _load("s7_chart_fit")
    assert cf.DEFAULT_TIME_ROW_WEIGHT == 4096
    hidden = 24
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"schema": "omni-campaign.prepb_time_row_weight_sweep.v1", "rule": "test rule",
                                 "sweep": [{"weight": 1}, {"weight": 4096}], "chosen_weight": 4096}))
    unit, width = _time_unit(tmp_path, hidden)
    tap = _time_rows_file(tmp_path, hidden, width)
    code = cf.main(["--unit", "hy3d_dit_L10", "--unit-dir", str(unit), "--tap-rows", str(tap), "--hidden", str(hidden),
                    "--chimera-src", src, "--arms", "ridge", "--weight-choice", str(sweep), "--lane", "PREP-B test"])
    assert code == 0
    block = json.loads((unit / "charted_block.json").read_text())
    wc = block["charts"]["time_rows"]["weight_choice"]
    assert block["charts"]["time_rows"]["weight"] == 4096
    assert wc["present"] and wc["chosen_weight"] == 4096 and wc["weight_used_equals_choice"] is True
    assert wc["sha256"] == cf.sha256_file(sweep) and wc["weights_swept"] == [1, 4096]
    assert block["charts"]["producer"]["lane"] == "PREP-B test"
    assert "x weight 4096" in block["s7_unit"]["charts_status"]
    assert "parent_pairs_sha256" in block["charts"]["pairs"] and "paired_rows_json_sha256" in block["charts"]["pairs"]
    # an explicit other weight is recorded as not the choice -- and is still written (a record, not a gate)
    unit2, width2 = _time_unit(tmp_path / "b", hidden)
    tap2 = _time_rows_file(tmp_path / "b", hidden, width2)
    assert cf.main(["--unit", "hy3d_dit_L10", "--unit-dir", str(unit2), "--tap-rows", str(tap2), "--hidden", str(hidden),
                    "--chimera-src", src, "--arms", "ridge", "--weight-choice", str(sweep), "--time-row-weight", "1"]) == 0
    wc2 = json.loads((unit2 / "chart_receipt.json").read_text())["diagnostics"]["time_rows"]["weight_choice"]
    assert wc2["weight_used"] == 1 and wc2["weight_used_equals_choice"] is False
    # a missing sweep receipt is recorded, not refused
    unit3, width3 = _time_unit(tmp_path / "c", hidden)
    tap3 = _time_rows_file(tmp_path / "c", hidden, width3)
    assert cf.main(["--unit", "hy3d_dit_L10", "--unit-dir", str(unit3), "--tap-rows", str(tap3), "--hidden", str(hidden),
                    "--chimera-src", src, "--arms", "ridge", "--weight-choice", str(tmp_path / "nope.json")]) == 0
    assert json.loads((unit3 / "charted_block.json").read_text())["charts"]["time_rows"]["weight_choice"]["present"] is False


# ---------------------------------------------------------------- S7-on-S5: s7_host_rows 1.3.0 --stream-blocks


def _tiny_host_rows_inputs(tmp_path, *, hidden=32, feature=48, layers=8):
    """A tiny residual-free Chimera checkpoint + tokenizer + one interface entry + a parent pair set (type 4, ACE)."""

    _bind_tree()
    fixtures = _load("c16_media_fixtures")
    model_dir = tmp_path / "model"
    fixtures.write_child(model_dir, seed=0, dtype="float32", hidden=hidden, feature=feature, music_band=24,
                         residual_mode="none", intermediate=40, num_layers=layers)
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate("alpha beta gamma delta eps zeta eta theta iota kappa".split())}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))
    band = (0, 8)
    iface = tmp_path / "iface"
    fixtures.write_interface_set(iface, [{"name": "music_ace", "source": "ace", "type_id": 4, "feature_band": list(band)}],
                                 hidden=hidden, feature=feature, seed=1)
    body = json.loads((iface / "companions.json").read_text())
    for entry in body["entries"]:
        entry["unit"] = "ace_dit_L12"
    (iface / "companions.json").write_text(json.dumps(body))
    (iface / "c11").mkdir()
    (iface / "c11" / "companions.json").write_text(json.dumps({"entries": []}))
    g = torch.Generator().manual_seed(9)
    prompts = ["alpha beta", "gamma delta eps", "alpha beta", "zeta eta theta iota", "kappa", "gamma delta eps"]
    rows_per = [5, 3, 5, 6, 4, 3]
    feats, types, fwd, token, widths, pos, sigma, parent, held, group, forwards = [], [], [], [], [], [], [], [], [], [], []
    for i, (p, n) in enumerate(zip(prompts, rows_per)):
        feats.append(torch.randn(n, 8, generator=g))
        types.append(torch.full((n,), 4))
        fwd.append(torch.full((n,), i))
        token.append(torch.arange(n))
        widths.append(torch.full((n,), 8))
        pos.append(torch.rand(n, 4, generator=g))
        sigma.append(torch.full((n,), 0.25 * (i % 3)))
        parent.append(torch.randn(n, 12, generator=g))
        held.append(torch.full((n,), i % 3 == 2))
        group.append(torch.full((n,), i))
        forwards.append({"index": i, "rows": n, "prompt": p})
    pairs = tmp_path / "pairs" / "ace_dit_L12"
    pairs.mkdir(parents=True)
    from safetensors.torch import save_file
    save_file({"features": torch.cat(feats), "type_ids": torch.cat(types), "forward": torch.cat(fwd), "token": torch.cat(token),
               "feature_width": torch.cat(widths), "positions": torch.cat(pos), "sigma": torch.cat(sigma),
               "parent_rows": torch.cat(parent), "held": torch.cat(held), "group": torch.cat(group)},
              str(pairs / "parent_capture.safetensors"))
    (pairs / "parent_capture.json").write_text(json.dumps({"forwards": forwards}))
    units_root = tmp_path / "units"
    units_root.mkdir()
    return {"model": model_dir, "tokenizer": tok_path, "interface": iface, "pairs": tmp_path / "pairs", "units_root": units_root}


def test_host_rows_streamed_blocks_equal_resident(tmp_path):
    """1.3.0: the layer-major streamed forward gives bit-identical host rows to the resident (1.2.0) path."""

    src = _bind_tree()
    hr = _load("s7_host_rows")
    assert hr.TOOL_VERSION == "1.3.0"
    inp = _tiny_host_rows_inputs(tmp_path)
    base = ["--unit", "ace_dit_L12", "--model", str(inp["model"]), "--tokenizer", str(inp["tokenizer"]),
            "--interface", str(inp["interface"]), "--pairs", str(inp["pairs"]), "--insert-after", "5",
            "--chimera-src", src, "--device", "cpu", "--dtype", "float32", "--max-batch-tokens", "16",
            "--bridge-sample", "7", "--units-root", str(inp["units_root"])]
    assert hr.main(base + ["--out", str(tmp_path / "resident")]) == 0
    assert hr.main(base + ["--out", str(tmp_path / "streamed"), "--stream-blocks"]) == 0
    from safetensors import safe_open
    got = {}
    for arm in ("resident", "streamed"):
        with safe_open(str(tmp_path / arm / "ace_dit_L12" / "paired_rows.safetensors"), "pt") as h:
            got[arm] = {k: h.get_tensor(k) for k in h.keys()}
    assert set(got["resident"]) == set(got["streamed"])
    for key in got["resident"]:
        assert torch.equal(got["resident"][key], got["streamed"][key]), key
    assert got["resident"]["host_rows"].shape == (26, 32) and bool(torch.isfinite(got["resident"]["host_rows"]).all())
    assert float(got["resident"]["host_rows"].norm()) > 0
    rec = {arm: json.loads((tmp_path / arm / "ace_dit_L12" / "paired_rows.json").read_text()) for arm in ("resident", "streamed")}
    assert rec["resident"]["batches"] == rec["streamed"]["batches"] > 1
    assert rec["resident"]["trunk"]["block_residency"].startswith("resident")
    assert rec["streamed"]["trunk"]["block_residency"].startswith("streamed")
    assert rec["resident"]["trunk"]["block_loads"] == rec["streamed"]["trunk"]["block_loads"] == 6
    assert rec["resident"]["diagnostics"]["causal_prompt_check"] == rec["streamed"]["diagnostics"]["causal_prompt_check"]
    assert rec["resident"]["diagnostics"]["content_share_pre_norm_median_per_batch"] == rec["streamed"]["diagnostics"]["content_share_pre_norm_median_per_batch"]
    assert rec["streamed"]["tool"]["version"] == "1.3.0"


def test_chart_fit_trunk_label_records_the_host_trunk(tmp_path):
    """1.3.0: --trunk-label names the trunk the host rows came from (default S3 text unchanged)."""

    src = _bind_tree()
    cf = _load("s7_chart_fit")
    hidden = 24
    for label in ("S3", "S5"):
        unit, width = _time_unit(tmp_path / label, hidden)
        tap = _time_rows_file(tmp_path / label, hidden, width)
        argv = ["--unit", "hy3d_dit_L10", "--unit-dir", str(unit), "--tap-rows", str(tap), "--hidden", str(hidden),
                "--chimera-src", src, "--arms", "ridge", "--weight-choice", ""]
        if label == "S5":
            argv += ["--trunk-label", "S5"]
        assert cf.main(argv) == 0
        block = json.loads((unit / "charted_block.json").read_text())
        if label == "S3":
            assert block["charts"]["valid_only_with"].startswith("the latent interface rows these host rows were captured under (s7/interface_latent) and the S3 trunk prefix 0..31")
            assert "fitted on S3-trunk host rows" in block["s7_unit"]["charts_status"]
            assert "trunk_label" not in block["charts"]
        else:
            assert "the S5 trunk prefix 0..31" in block["charts"]["valid_only_with"] and "S3" not in block["charts"]["valid_only_with"]
            assert "fitted on S5-trunk host rows" in block["s7_unit"]["charts_status"]
            assert block["charts"]["trunk_label"] == "S5"


def test_dryrun_passes_c15_c11_c18_and_child_geometry(tmp_path):
    """s7_dryrun 1.1.0: interface rows, C11 rows and the band table install into the tiny child with the given geometry."""

    _bind_tree()
    fixtures = _load("c16_media_fixtures")
    dr = _load("s7_dryrun")
    assert dr.TOOL_VERSION == "1.1.0"
    hidden, feature, band = 32, 60, 40
    fixtures.write_interface_set(tmp_path / "c15", [{"name": "music_ace", "source": "ACE", "type_id": 4, "feature_band": [0, 20]}],
                                 hidden=hidden, feature=feature, seed=1)
    fixtures.write_interface_set(tmp_path / "c11", [{"name": "ocr", "source": "QVL32", "type_id": 6, "feature_band": [20, 40]}],
                                 hidden=hidden, feature=feature, seed=2)
    fixtures.write_band_table(tmp_path / "c18" / "band.safetensors", width=band, hidden=hidden, seed=3)
    fixtures.write_block_container(tmp_path / "blocks", "ace_L12", hidden=hidden, seed=4)
    unit = tmp_path / "blocks" / "ace_L12"
    code = dr.main(["--units", str(unit), "--scratch", str(tmp_path / "s"), "--hidden", str(hidden), "--run-name", "T",
                    "--build-child", "--feature", str(feature), "--music-band", str(band), "--layers", "8", "--intermediate", "40",
                    "--c15", str(tmp_path / "c15"), "--c11", str(tmp_path / "c11"), "--c18", str(tmp_path / "c18" / "band.safetensors")])
    assert code == 0
    assert not (tmp_path / "s" / "tap" / "ace_L12").exists()  # the unit ships charts: no synthetic tap rows
    manifest = json.loads((tmp_path / "s" / "out_T" / "composite_manifest.json").read_text())
    text = json.dumps(manifest)
    assert "music_ace" in text and "ocr" in text and "music_codes" in text
