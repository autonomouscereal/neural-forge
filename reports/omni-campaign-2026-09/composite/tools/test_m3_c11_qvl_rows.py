"""Tests for m3_c11_qvl_rows.py (lane M3, Claude, 2026-10-10). Server, CPU only:

    cd /home/cereal/omni-campaign/composite/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 PYTHONPATH=$CHIMERA_OMNI_SRC/src \
    OMP_NUM_THREADS=8 nice -n 19 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \
        -p no:cacheprovider -q test_m3_c11_qvl_rows.py

Synthetic captures whose merger IS a shared per-patch linear map: the fit must recover it
(held rel_l2 ~ 0), the written rows must sum over each 2x2 group to the merged token (through
the alignment for arm qvl32), the alignment must recover a planted linear map between two
toy vocabularies on their shared strings, and the written set must load through
c16_install_media.load_interface_set with the declared tensor_sha256 verified.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

HERE = Path(__file__).resolve().parent
H, OUT, FEAT = 8, 16, 2304


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rows = _load(HERE / "m3_c11_qvl_rows.py", "m3_c11_qvl_rows")
c16 = _load(HERE / "c16_install_media.py", "c16_install_media_for_c11")


def _world(tmp: Path, *, images: int = 15, vocab: int = 300):
    g = torch.Generator().manual_seed(0)
    P = torch.randn(H, 3, 2, 16, 16, generator=g) * 0.05
    b = torch.randn(H, generator=g) * 0.1
    M = torch.randn(OUT, H, generator=g)
    c = torch.randn(OUT, generator=g)
    parent = tmp / "parent.safetensors"
    qemb = torch.randn(vocab, OUT, generator=g)
    save_file({"visual.patch_embed.proj.weight": P, "visual.patch_embed.proj.bias": b,
               "model.embed_tokens.weight": qemb}, str(parent))
    cap = tmp / "cap"
    cap.mkdir()
    records = []
    for i in range(images):
        x = torch.randn(64, 1536, generator=g)
        p = x @ P.reshape(H, -1).T + b
        merged = p.view(-1, 4, H).sum(1) @ M.T + c
        save_file({"pixels": x, "patch_embed": p, "block13_in": p.clone(), "merger_in": p.clone(), "merged": merged,
                   "grid_thw": torch.tensor([1, 8, 8])}, str(cap / f"img{i:04d}.safetensors"))
        records.append({"index": i, "file": f"img{i:04d}.safetensors"})
    (cap / "capture_receipt.json").write_text(json.dumps({"images": records, "parent": str(parent), "config": {}}))
    # trunk: embedding = A_true @ qvl embedding + a on the shared strings
    A_true = torch.randn(OUT, OUT, generator=g) * 0.3
    a_true = torch.randn(OUT, generator=g)
    tokens = [f"t{i}" for i in range(vocab)]
    (tmp / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}))
    trunk_ids = {t: vocab - 1 - i for i, t in enumerate(tokens[: vocab - 20])}  # 20 strings missing in the trunk
    (tmp / "tokenizer.json").write_text(json.dumps({"model": {"vocab": trunk_ids}}))
    temb = torch.zeros(vocab, OUT)
    for t, tid in trunk_ids.items():
        temb[tid] = qemb[int(t[1:])] @ A_true.T + a_true
    native = tmp / "native"
    native.mkdir()
    save_file({"embed.weight": temb, "native_media_bridge.type_embed.weight": torch.randn(8, OUT, generator=g)},
              str(native / "model-00001-of-00001.safetensors"))
    (native / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "embed.weight": "model-00001-of-00001.safetensors",
        "native_media_bridge.type_embed.weight": "model-00001-of-00001.safetensors"}}))
    return parent, cap, native, (M, c, A_true, a_true)


class _Cfg:
    hidden_size = OUT
    native_feature_dim = FEAT
    native_type_count = 8


@pytest.mark.parametrize("arm", ["q38u", "qvl32"])
def test_rows_recover_the_planted_maps_and_load(tmp_path, arm) -> None:
    parent, cap, native, (M, c, A_true, a_true) = _world(tmp_path)
    out = tmp_path / f"c11_{arm}"
    argv = ["--captures", str(cap), "--parent", str(parent), "--prefix", "visual.", "--arm", arm,
            "--trunk-native", str(native), "--hidden", str(OUT), "--native-feature-dim", str(FEAT), "--out", str(out)]
    if arm == "qvl32":
        argv += ["--align-vocab", str(tmp_path / "vocab.json"), "--trunk-tokenizer", str(tmp_path / "tokenizer.json")]
    assert rows.main(argv) == 0
    report = json.loads((out / "c11_rows_receipt.json").read_text())
    assert report["patch_embed"]["exactness_rel_l2_vs_capture"] < 1e-6
    assert report["merger_fit"]["held"]["rel_l2"] < 1e-4
    assert report["group_sum_consistency_rel_l2"] < 1e-5
    if arm == "qvl32":
        assert report["alignment"]["matched_token_strings"] == 280
        assert report["alignment"]["held"]["rel_l2"] < 1e-4
    entries = c16.load_interface_set(out, "c11", _Cfg())
    assert len(entries) == 1 and entries[0]["type_id"] == 6
    assert entries[0]["feature_band"] == [640, 2176] and entries[0]["emission_band"] == [384, 385]
    assert entries[0]["tensor_sha256_check"] == {"declared": 4, "verified": True}
    state = {k: v.torch() for k, v in c16.read_single(entries[0]["rows_path"]).items()}
    assert tuple(state["bridge.feature_proj.weight"].shape) == (OUT, 1536)
    assert float(state["head.out.weight"].abs().max()) == 0.0
    # the planted end-to-end linear map: (A) M P, recovered
    P = c16.read_single(parent)["visual.patch_embed.proj.weight"].torch().double().reshape(H, -1)
    want = M.double() @ P
    if arm == "qvl32":
        want = A_true.double() @ want
    got = state["bridge.feature_proj.weight"].double()
    assert float((got - want).norm() / want.norm()) < 1e-3


def test_arm_qvl32_refuses_without_alignment_inputs(tmp_path) -> None:
    parent, cap, native, _ = _world(tmp_path, images=5)
    with pytest.raises(SystemExit):
        rows.main(["--captures", str(cap), "--parent", str(parent), "--prefix", "visual.", "--arm", "qvl32",
                   "--trunk-native", str(native), "--hidden", str(OUT), "--out", str(tmp_path / "o")])
