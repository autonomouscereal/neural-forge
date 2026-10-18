"""Tests for ``k3_cache_parity.py`` on the lane's synthetic K3 (CPU): two builds of k3_teacher_cache hashed tensor
by tensor, and the negative control -- a deliberate 1-ulp change of one prefix element must be reported DIFFER.

  test_tensor_sha256_is_raw_bytes_and_sees_one_ulp
  test_frozen_124_and_v130_digests_are_bit_identical   (full pass: 6 layers, every block slot, every head tensor)
  test_one_ulp_injection_is_caught_from_that_layer_on  (the negative control; exit 3)
  test_missing_tensors_are_not_identical

Server:
  cd $OC/distill/tools && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OC/distill/tools \\
      $PY -m pytest -o addopts="" -p no:cacheprovider test_k3_cache_parity.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import k3_cache_parity as kcp  # noqa: E402
from test_k3_teacher_cache import ids_files, run_args, synth, work  # noqa: E402,F401 -- the 1.2.4 suite's fixtures

V124 = str(HERE / "k3_teacher_cache.py")
V130 = str(HERE / "k3_teacher_cache_v130.py")


def digest(tool: str, synth, ids_files, out: Path, inject=None, *extra) -> dict:
    argv = run_args(synth, ids_files, out, "--group-tokens", "20", "--compute-dtype", "bf16", *extra)
    return kcp.digest_run(tool, None, argv, inject_ulp=inject, echo=False)


def write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload))
    return str(path)


def test_tensor_sha256_is_raw_bytes_and_sees_one_ulp():
    t = torch.randn(300, 7).to(torch.bfloat16)
    a = kcp.tensor_sha256(t)
    assert a == kcp.tensor_sha256(t.clone()) == kcp.tensor_sha256(torch.cat([t[:100], t[100:]]))
    s = t.clone()
    kcp.one_ulp(s)
    assert kcp.tensor_sha256(s) != a and float((s.float() - t.float()).abs().max()) > 0
    z = torch.zeros(4, 2)
    n = z.clone()
    n[0, 0] = -0.0
    assert kcp.tensor_sha256(z) != kcp.tensor_sha256(n)  # +0.0 vs -0.0: bits, not values
    stack = torch.randn(50, 3, 9)
    assert kcp.tensor_sha256(stack[:, 1]) == kcp.tensor_sha256(stack[:, 1].contiguous())


def test_frozen_124_and_v130_digests_are_bit_identical(synth, ids_files, work, capsys):
    a = digest(V124, synth, ids_files, work / "kcp-124")
    b = digest(V130, synth, ids_files, work / "kcp-130")
    assert a["subject"]["version"] == "1.2.4" and b["subject"]["version"] == "1.3.0"
    res = kcp.compare(a, b)
    n_layers = len(a["digests"]["layers"])
    assert n_layers == 6 and a["digests"]["head"]
    assert res["bit_identical"] and res["tensors"] == len(kcp.flatten(a["digests"])) > n_layers
    rc = kcp.main(["compare", write(work / "a.json", a), write(work / "b.json", b), "--quiet"])
    assert rc == 0 and "BIT-IDENTICAL" in capsys.readouterr().out


def test_one_ulp_injection_is_caught_from_that_layer_on(synth, ids_files, work, capsys):
    a = digest(V124, synth, ids_files, work / "kcp-neg-124")
    b = digest(V130, synth, ids_files, work / "kcp-neg-130", 2)
    res = kcp.compare(a, b)
    assert not res["bit_identical"]
    assert "L02 prefix" in res["differ"] and not any(d.startswith(("L00", "L01")) for d in res["differ"])
    assert any(d.startswith("head ") for d in res["differ"])  # it reaches the cache
    rc = kcp.main(["compare", write(work / "na.json", a), write(work / "nb.json", b), "--quiet"])
    assert rc == 3 and "DIFFER" in capsys.readouterr().out


def test_missing_tensors_are_not_identical(synth, ids_files, work):
    a = digest(V130, synth, ids_files, work / "kcp-miss-a")
    b = json.loads(json.dumps(a))
    b["digests"]["layers"].pop("5")
    res = kcp.compare(a, b)
    assert not res["bit_identical"] and "L05 prefix" in res["only_a"] and not res["only_b"]
    assert all(d in res["only_a"] for d in res["differ"])  # a tensor hashed on one side only is never EQUAL
