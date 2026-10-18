"""Tests for c18_music_band.py (lane M2, 2026-10-10).

    CHIMERA_SRC=$OC/import/rq27b/tree_ef1ffb9/src OMP_NUM_THREADS=4 nice -n 19 \
        venv/bin/python -m pytest -q composite/tools/test_c18_music_band.py
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

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")

import c18_music_band as tool  # noqa: E402

TEXT = [
    "the quick brown fox jumps over the lazy dog " * 4,
    "music codes and tokens and text and bytes " * 4,
    "a slow piano ballad in c minor at seventy two bpm " * 4,
    "def fold(x): return [v for v in x if v > 0] " * 4,
]


def _bpe(vocab_size: int, corpus: list[str], added: list[str], path: Path) -> int:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|endoftext|>"], show_progress=False)
    tok.train_from_iterator(corpus, trainer)
    if added:
        tok.add_special_tokens(added)
    path.mkdir(parents=True, exist_ok=True)
    tok.save(str(path / "tokenizer.json"))
    size = tok.get_vocab_size(with_added_tokens=True)
    (path / "config.json").write_text(json.dumps({"vocab_size": size}), encoding="utf-8")
    return size


def _setup(tmp_path: Path, *, exact: bool = True, music_band: int = 24, codes: int = 20):
    import c16_media_fixtures as fx
    from safetensors.torch import save_file

    hidden, ace_hidden = 32, 40
    native = tmp_path / "trunk"
    fx.write_child(native, seed=0, hidden=hidden, feature=48, base_vocab=320, music_band=music_band, residual_mode="none", dtype="float32")
    trunk_tok = tmp_path / "trunk_tok"
    _bpe(318, TEXT, [], trunk_tok)
    ace_dir = tmp_path / "ace" / "component"
    code_names = [f"<|audio_code_{i}|>" for i in range(codes + 4)]  # 4 named-but-excluded codes
    ace_size = _bpe(300, TEXT[::-1] + ["extra words for a different merge order"], code_names, ace_dir / "tokenizer")
    code_begin = ace_size - len(code_names)
    # trunk tables from the toy child; ACE table built so the anchors are EXACTLY linear in the trunk rows
    from chimera_omni.importing.bridge import token_alignment
    from chimera_omni.importing.tokenizer import ChimeraTokenizer
    from safetensors import safe_open

    index = json.loads((native / "model.safetensors.index.json").read_text())
    with safe_open(str(native / index["weight_map"]["embed.weight"]), framework="pt") as h:
        embed = h.get_tensor("embed.weight").to(torch.float64)
    t = ChimeraTokenizer.from_qwen(trunk_tok, base_vocab_size=320)
    a = ChimeraTokenizer.from_qwen(ace_dir / "tokenizer", base_vocab_size=ace_size)
    pairs, _ = token_alignment(t, a, ["hello"], full_vocab=True)
    g = torch.Generator().manual_seed(3)
    M = torch.randn(ace_hidden, hidden, generator=g, dtype=torch.float64)
    ace = torch.randn(ace_size, ace_hidden, generator=g, dtype=torch.float64) * (0.0 if exact else 1.0)
    special = set(a.base.special_ids)
    for tid, pieces in pairs.items():
        if len(pieces) == 1 and pieces[0] not in special and pieces[0] < code_begin:
            ace[pieces[0]] = M @ embed[tid] + (0.0 if exact else 0.3 * torch.randn(ace_hidden, generator=g, dtype=torch.float64))
    ace[code_begin:] = torch.randn(len(code_names), ace_hidden, generator=g, dtype=torch.float64)
    save_file({"embed_tokens.weight": ace.to(torch.float32).contiguous()}, str(ace_dir / "model.safetensors"))
    return native, trunk_tok, ace_dir, code_begin, codes, M, embed


def _argv(tmp_path, native, trunk_tok, ace_dir, code_begin, codes, extra=None):
    return [
        "--trunk-native", str(native), "--trunk-tokenizer-dir", str(trunk_tok), "--trunk-base-vocab", "320",
        "--ace-component", str(ace_dir), "--code-begin", str(code_begin), "--code-count", str(codes),
        "--out", str(tmp_path / "out" / "band.safetensors"), "--retrieval-queries", "16", "--chimera-src", CHIMERA_SRC,
        "--min-anchor-factor", "2",
    ] + (extra or [])


def test_band_table_end_to_end_exact_linear(tmp_path):
    native, trunk_tok, ace_dir, code_begin, codes, M, embed = _setup(tmp_path, exact=True)
    assert tool.main(_argv(tmp_path, native, trunk_tok, ace_dir, code_begin, codes)) == 0
    sidecar = json.loads((tmp_path / "out" / "band.safetensors.json").read_text())
    al = sidecar["alignment"]
    assert al["candidate_tokens"] == 320 and al["scope"] == "full_vocab"
    assert al["aligned_tokens"] + al["unmapped_tokens"] == 320
    # trunk rows are an exact linear function of the ACE anchor rows: the embed map recovers them
    assert sidecar["diagnostics"]["embed.weight"]["held"]["rel_l2"] < 1e-3
    assert abs(sidecar["diagnostics"]["embed.weight"]["calibrated"]["gamma"] - 1.0) < 1e-2  # an exact fit needs no calibration
    from safetensors.torch import load_file

    table = load_file(str(tmp_path / "out" / "band.safetensors"))
    assert table["embed.weight"].shape == (24, 32) and table["lm_head.weight"].shape == (24, 32)
    assert bool(table["embed.weight"][codes:].eq(0).all()) and bool(table["lm_head.weight"][codes:].eq(0).all())
    assert bool(table["embed.weight"][:codes].ne(0).any())
    # the band rows are the map applied to the ACE code rows
    maps = load_file(str(tmp_path / "out" / "band_maps.safetensors"))
    ace = load_file(str(ace_dir / "model.safetensors"))["embed_tokens.weight"].to(torch.float64)
    w = maps["embed.weight.map.weight"].to(torch.float64)
    b = maps["embed.weight.map.bias"].to(torch.float64)
    expected = ace[code_begin : code_begin + codes] @ w + b
    assert torch.allclose(table["embed.weight"][:codes].to(torch.float64), expected, rtol=1e-4, atol=1e-4)
    assert sidecar["installer_check"]["tensor_sha256_check"]["verified"] is True


def test_installer_accepts_the_table_on_the_toy_child(tmp_path):
    """c16 install_band itself, on the toy child (writes a toy child; the real one is never written here)."""

    native, trunk_tok, ace_dir, code_begin, codes, M, embed = _setup(tmp_path, exact=False)
    assert tool.main(_argv(tmp_path, native, trunk_tok, ace_dir, code_begin, codes)) == 0
    import c16_install_media as c16

    out = tmp_path / "child_out"
    oc = HERE.parent.parent
    rc = c16.main([
        "--chimera-src", str(Path(CHIMERA_SRC).parent), "--mediafold-root", str(oc / "media"), "--ledger-root", str(oc / "ledger"),
        "--child", str(native), "--c18", str(tmp_path / "out" / "band.safetensors"), "--out", str(out), "--min-free-gib", "0",
    ])
    assert rc == 0
    manifest = json.loads((out / "composite_manifest.json").read_text())
    assert manifest["band"]["rows_written"] == 24
    assert manifest["band"]["table_tensor_sha256_check"]["verified"] is True


def test_refuses_band_width_smaller_than_code_count(tmp_path):
    native, trunk_tok, ace_dir, code_begin, codes, M, embed = _setup(tmp_path, music_band=24, codes=20)
    assert tool.main(_argv(tmp_path, native, trunk_tok, ace_dir, code_begin, 30)) == 2


def test_refuses_nonzero_band_rows(tmp_path):
    native, trunk_tok, ace_dir, code_begin, codes, M, embed = _setup(tmp_path)
    from safetensors import safe_open
    from safetensors.torch import save_file

    index = json.loads((native / "model.safetensors.index.json").read_text())
    shard = native / index["weight_map"]["embed.weight"]
    with safe_open(str(shard), framework="pt") as h:
        state = {k: h.get_tensor(k) for k in h.keys()}
        meta = h.metadata()
    state["embed.weight"][-1] = 1.0
    save_file(state, str(shard), metadata=meta)
    assert tool.main(_argv(tmp_path, native, trunk_tok, ace_dir, code_begin, codes)) == 2


def test_deviation_calibration_restores_the_spread(tmp_path):
    native, trunk_tok, ace_dir, code_begin, codes, M, embed = _setup(tmp_path, exact=False)
    assert tool.main(_argv(tmp_path, native, trunk_tok, ace_dir, code_begin, codes)) == 0
    side = json.loads((tmp_path / "out" / "band.safetensors.json").read_text())
    for key in ("embed.weight", "lm_head.weight"):
        d = side["diagnostics"][key]
        assert d["calibrated"]["gamma"] >= 1.0  # ridge only ever shrinks
        assert d["ridge"]["held"]["rel_l2"] > 0
    assert tool.main(_argv(tmp_path, native, trunk_tok, ace_dir, code_begin, codes, ["--deviation-calibration", "none", "--out", str(tmp_path / "raw" / "band.safetensors")])) == 0
    raw = json.loads((tmp_path / "raw" / "band.safetensors.json").read_text())
    assert raw["diagnostics"]["embed.weight"]["calibrated"]["gamma"] == 1.0
