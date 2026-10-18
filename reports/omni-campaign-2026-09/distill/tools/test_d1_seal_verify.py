"""Tests for ``d1_seal_verify.py`` -- CPU / filesystem + numpy, no torch, no GPU.

A synthetic hardened run (4 text layers: L00-L01 imported from a canonical dir, L02-L03 trained "here") is built
with the trainer's own formats (receipt + sidecar, rung records, the deterministic safetensors layout, the
D1_delta manifest, a base dir whose sealed digest is oc_seal's), then synced with d1_canonical_rungs 1.1.0 the way
the seal queue does it.

  test_clean_run_verifies              every check passes; the receipt is written with its sidecar
  test_refusals                        failed / unfinalized status, launch mismatch, unsynced canonical, a rung
                                       whose input_chain is off, a flipped payload byte, a non-finite element
                                       (hashes made consistent), a rung file whose tensors are not its record's
                                       entries, a base digest that is not the sealed one
  test_resumed_attempt_shapes          (1.1.0, d1seal review) the receipt of an attempt that RESUMED after a restart:
                                       L00-L01 boundary-resumed over the canonical prefix (``rung`` + ``from_boundary``,
                                       no import record), L02 boundary-resumed own rung, L03 replay-resumed
                                       (``resumed_from``, no sha) -> sync + verify pass (1.0.0 / sync 1.1.0 refused);
                                       a replay record pointing outside the run, a boundary prefix rung with a foreign
                                       sha and a replayed rung whose chain is off are refused
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
HERE = Path(__file__).resolve().parent
os.environ.setdefault("OC_LEDGER_ROOT", str(HERE.parents[1] / "ledger") if (HERE.parents[1] / "ledger").is_dir()
                      else "/home/cereal/omni-campaign/ledger")
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ["OC_LEDGER_ROOT"])

import d1_canonical_rungs as cr  # noqa: E402
import d1_seal_verify as sv  # noqa: E402
from oc_seal import directory_digest  # noqa: E402

N, P, H, Q = 4, 2, 8, 4          # layers, prefix, hidden, trunk width
PARTS = (("ffn.up.weight", "gate_trunk", [Q, H], [[0, Q], [0, H]]), ("ffn.up.weight", "up_trunk", [Q, H], [[6, 6 + Q], [0, H]]),
         ("ffn.down.weight", "down_trunk", [H, Q], [[0, H], [0, Q]]), ("norm2.weight", "whole", [H], [[0, H]]))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_st(path: Path, tensors: dict[str, np.ndarray], meta: dict[str, str]) -> None:
    """d1_trunk_recovery.save_safetensors_deterministic's layout for BF16 (uint16 bit arrays)."""
    names = sorted(tensors)
    header = {"__metadata__": dict(sorted(meta.items()))}
    off = 0
    for n in names:
        nb = tensors[n].size * 2
        header[n] = {"dtype": "BF16", "shape": list(tensors[n].shape), "data_offsets": [off, off + nb]}
        off += nb
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    with open(path, "wb") as fh:
        fh.write(len(text).to_bytes(8, "little"))
        fh.write(text)
        for n in names:
            fh.write(tensors[n].astype("<u2").tobytes())


def bf16(shape, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def layer_tensors(i: int, tag: str) -> tuple[dict[str, np.ndarray], list[dict]]:
    tensors, entries = {}, []
    for k, (base, part, shape, sl) in enumerate(PARTS):
        key = f"blocks.{i}.{base}::{part}"
        t = bf16(shape, 1000 * i + k + (7 if tag == "hard" else 0))
        tensors[key] = t
        entries.append({"key": key, "base_tensor": f"blocks.{i}.{base}", "part": part, "slice": sl, "shape": shape,
                        "dtype": "bfloat16", "sha256": sha(t.astype("<u2").tobytes()), "base_tensor_sha256": "b" * 64})
    return tensors, entries


def build(tmp: Path) -> dict:
    base = tmp / "clone"
    base.mkdir()
    for name, body in (("config.json", b"{}"), ("model.safetensors.index.json", b'{"weight_map": {}}'),
                       ("composite_manifest.json", b"{}")):
        (base / name).write_bytes(body)
    for s in (1, 2):
        write_st(base / f"model-0000{s}-of-00002.safetensors", {f"t{s}": bf16([4], s)}, {})
    sealed, _ = directory_digest(base)
    canon, run, delta = tmp / "canon", tmp / "run", tmp / "delta"
    canon.mkdir()
    (run / "trainer" / "rungs").mkdir(parents=True)
    delta.mkdir()
    # canonical prefix L00..L01 (a predecessor's rungs)
    state = {}
    for i in range(P):
        t, e = layer_tensors(i, "old")
        write_st(canon / f"L{i:02d}.safetensors", t, {"schema": "omni-campaign.d1-rung.v1"})
        doc = {"schema": "omni-campaign.d1-rung.v1", "text_layer": i, "child_block": i,
               "file_sha256": cr.sha256_file(canon / f"L{i:02d}.safetensors"), "entries": e, "input_chain": "codex"}
        (canon / f"L{i:02d}.json").write_text(json.dumps(doc))
        state[i] = {"safetensors": doc["file_sha256"], "json": cr.sha256_file(canon / f"L{i:02d}.json")}
    sums_sha = cr.write_sums(canon, state)
    identity = {"tool": ["d1_trunk_recovery", "1.3.1"], "child": {"path": str(base)}}
    chain = sv.sha256_json(identity)
    layers, all_entries, tensors = [], [], {}
    for i in range(N):
        if i < P:
            t, e = layer_tensors(i, "old")
            s = state[i]["safetensors"]
            layers.append({"text_layer": i, "child_block": i, "parent_block": i, "trained": True,
                           "resumed_from": str(canon / f"L{i:02d}.safetensors"),
                           "imported_from_prefix": {"dir": str(canon), "sha256sums_sha256": sums_sha, "json_sha256": state[i]["json"]}})
        else:
            t, e = layer_tensors(i, "hard")
            rp = run / "trainer" / "rungs" / f"L{i:02d}.safetensors"
            write_st(rp, t, {"schema": "omni-campaign.d1-rung.v1"})
            s = cr.sha256_file(rp)
            (rp.with_suffix(".json")).write_text(json.dumps({"schema": "omni-campaign.d1-rung.v1", "text_layer": i, "child_block": i,
                                                             "file": str(rp), "file_sha256": s, "entries": e, "input_chain": chain}))
            layers.append({"text_layer": i, "child_block": i, "parent_block": i, "trained": True, "rung": {"file": str(rp), "sha256": s}})
        tensors.update(t)
        all_entries.extend({**x, "text_layer": i} for x in e)
        chain = sv.sha256_json([chain, i, s])
    dfile = delta / "D1_delta.safetensors"
    write_st(dfile, tensors, {"schema": "omni-campaign.d1-delta.v1", "tool": "d1_trunk_recovery 1.3.1"})
    base_doc = {"path": str(base), "config_sha256": cr.sha256_file(base / "config.json"),
                "index_sha256": cr.sha256_file(base / "model.safetensors.index.json"), "mode": "full",
                "files": {p.name: {"bytes": p.stat().st_size, "mtime_ns": 1, "sha256": cr.sha256_file(p)} for p in base.glob("*.safetensors")}}
    params = sum(int(np.prod(e["shape"])) for e in all_entries)
    manifest = {"schema": "omni-campaign.d1-delta.v1", "tool": {"name": "d1_trunk_recovery", "version": "1.3.1", "sha256": "c" * 64},
                "base": base_doc, "trained_text_layers": list(range(N)), "parameters": params, "file": str(dfile),
                "file_sha256": cr.sha256_file(dfile), "bytes": dfile.stat().st_size, "entries": all_entries,
                "reader": "d1_trunk_recovery.apply_delta (slice overwrite of the base tensor)"}
    write_json_sha(delta / "D1_delta.json", manifest)
    receipt = {"schema": "omni-campaign.d1-trunk-recovery.v1", "tool": {"name": "d1_trunk_recovery", "version": "1.3.1", "sha256": "c" * 64},
               "identity": identity, "run_identity_sha256": sv.sha256_json(identity), "child_text_blocks": N,
               "stop_after_layer": N - 1, "train_layers": list(range(N)), "layers": layers,
               "prefix": {"dir": str(canon), "layers": P, "sha256sums_sha256": sums_sha, "rung_sha256": [state[i]["safetensors"] for i in range(P)]},
               "verdict": f"{N} layer(s) trained, streams finite through text layer {N - 1}; delta written",
               "delta": {"dir": str(delta), "file": str(dfile), "file_sha256": manifest["file_sha256"], "bytes": manifest["bytes"],
                         "parameters": params, "manifest_sha256": cr.sha256_file(delta / "D1_delta.json")}}
    write_json_sha(run / "trainer" / "d1_receipt.json", receipt)
    (run / "launch.json").write_text(json.dumps({"launch_id": "L1", "unit": "claude-d1-hardened.service"}))
    (run / "runtime-status.json").write_text(json.dumps({"completed": True, "state": "succeeded", "exit_code": 0, "alert": None,
                                                         "launch_id": "L1", "systemd": {"service_result": "success"}}))
    return {"base": base, "sealed": sealed, "canon": canon, "run": run, "delta": delta}


def write_json_sha(path: Path, doc: dict) -> None:
    raw = json.dumps(doc, indent=1, sort_keys=True).encode()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_text(sha(raw) + "\n")


def args(w: dict, out: Path) -> list[str]:
    return ["--run-dir", str(w["run"]), "--canonical", str(w["canon"]), "--delta-dir", str(w["delta"]),
            "--base-dir", str(w["base"]), "--base-sealed-sha256", w["sealed"], "--expect-layers", str(N), "--out", str(out)]


def synced(w: dict) -> dict:
    cr.sync(w["canon"], w["run"] / "trainer" / "d1_receipt.json", w["run"] / "trainer" / "rungs", "hardened")
    return w


def test_clean_run_verifies(tmp_path: Path) -> None:
    w = synced(build(tmp_path))
    out = tmp_path / "verify.json"
    assert sv.main(args(w, out)) == 0
    doc = json.loads(out.read_text())
    assert doc["verified"] is True and doc["receipt"]["prefix_layers"] == P and doc["receipt"]["new_rungs"] == N - P
    assert doc["canonical"]["layers"] == N and doc["delta_payload"]["nonfinite_elements"] == 0
    assert doc["delta_payload"]["tensors"] == 4 * N and doc["base"]["rederived_from_run_shard_hashes"] is True
    assert (out.with_name(out.name + ".sha256")).read_text().split()[0] == sv.sha256_file(out)


def refused(w: dict, out: Path, match: str) -> None:
    assert sv.main(args(w, out)) == 3
    doc = json.loads(out.read_text())
    assert doc["verified"] is False and match in doc["error"], doc["error"]


def test_refusals(tmp_path: Path) -> None:
    def fresh(name: str) -> dict:
        d = tmp_path / name
        d.mkdir()
        return synced(build(d))

    w = fresh("status")
    st = json.loads((w["run"] / "runtime-status.json").read_text())
    (w["run"] / "runtime-status.json").write_text(json.dumps({**st, "state": "failed", "exit_code": 1}))
    refused(w, tmp_path / "a.json", "completed also means failure")
    (w["run"] / "runtime-status.json").write_text(json.dumps({**st, "systemd": {}}))
    refused(w, tmp_path / "b.json", "SERVICE_RESULT")
    (w["run"] / "runtime-status.json").write_text(json.dumps({**st, "launch_id": "L0"}))
    refused(w, tmp_path / "c.json", "launch.json")

    (tmp_path / "unsynced").mkdir()
    w = build(tmp_path / "unsynced")   # canonical stops at L01: the seal must sync first
    refused(w, tmp_path / "d.json", "expected 4")

    w = fresh("chain")
    rj = w["run"] / "trainer" / "rungs" / "L03.json"
    doc = json.loads(rj.read_text())
    rj.write_text(json.dumps({**doc, "input_chain": "0" * 64}))
    refused(w, tmp_path / "e.json", "input_chain")

    w = fresh("byte")
    f = w["delta"] / "D1_delta.safetensors"
    raw = bytearray(f.read_bytes())
    raw[-3] ^= 0x01
    os.chmod(f, 0o644)
    f.write_bytes(bytes(raw))
    refused(w, tmp_path / "f.json", "payload hash differs")

    w = fresh("inf")   # an inf element: check_payload alone, with every hash made consistent around it
    f = w["delta"] / "D1_delta.safetensors"
    raw = bytearray(f.read_bytes())
    raw[-2:] = (0x7F80).to_bytes(2, "little")
    f.write_bytes(bytes(raw))
    man = json.loads((w["delta"] / "D1_delta.json").read_text())
    hashes = sv.span_hashes(f)
    for e in man["entries"]:
        e["sha256"] = hashes[e["key"]]
    man["file_sha256"] = sha(bytes(raw))
    with pytest.raises(sv.Refusal, match="non-finite"):
        sv.check_payload(w["delta"], man)
    man["entries"][0]["sha256"] = "0" * 64
    with pytest.raises(sv.Refusal, match="payload hash"):
        sv.check_payload(w["delta"], man)

    w = fresh("rungbytes")   # a canonical rung whose tensor bytes no longer match its record (file sha re-pinned)
    jp = w["canon"] / "L00.json"
    d = json.loads(jp.read_text())
    d["entries"][0]["sha256"] = "1" * 64
    os.chmod(jp, 0o644)
    jp.write_text(json.dumps(d))
    st = cr.canonical_state(w["canon"])
    st[0]["json"] = cr.sha256_file(jp)
    cr.write_sums(w["canon"], st)
    rec_path = w["run"] / "trainer" / "d1_receipt.json"
    rec = json.loads(rec_path.read_text())
    rec["layers"][0]["imported_from_prefix"]["json_sha256"] = st[0]["json"]   # the run "imported" the edited record
    write_json_sha(rec_path, rec)
    refused(w, tmp_path / "g.json", "tensor payload hashes differ")

    w = fresh("base")
    w["sealed"] = "a" * 64
    refused(w, tmp_path / "h.json", "re-derived base digest")


def resumed(w: dict, edit=None) -> dict:
    """Rewrite the run's receipt into the shape d1_trunk_recovery 1.3.1 writes for a RESUMED attempt."""
    rec_path = w["run"] / "trainer" / "d1_receipt.json"
    rec = json.loads(rec_path.read_text())
    st = cr.canonical_state(w["canon"])
    for i, layer in enumerate(rec["layers"]):
        base = {k: layer[k] for k in ("text_layer", "child_block", "parent_block")}
        if i < P:            # boundary resume over the canonical prefix
            rec["layers"][i] = {**base, "from_boundary": "/b/L02", "trained": True,
                                "rung": {"file": str(w["canon"] / f"L{i:02d}.safetensors"), "sha256": st[i]["safetensors"]}}
        elif i == P:         # boundary resume, own rung
            rec["layers"][i] = {**base, "from_boundary": "/b/L02", "trained": True, "rung": layer["rung"]}
        else:                # replayed from the run's own rungs dir (load_rung): no sha in the record
            rec["layers"][i] = {**base, "trained": True, "resumed_from": layer["rung"]["file"], "record": {}}
    if edit:
        edit(rec)
    write_json_sha(rec_path, rec)
    return w


def test_resumed_attempt_shapes(tmp_path: Path) -> None:
    def fresh(name: str, edit=None) -> dict:
        d = tmp_path / name
        d.mkdir()
        return resumed(build(d), edit)

    w = synced(fresh("ok"))
    assert sorted(cr.canonical_state(w["canon"])) == list(range(N))
    out = tmp_path / "ok.json"
    assert sv.main(args(w, out)) == 0, json.loads(out.read_text()).get("error")
    doc = json.loads(out.read_text())
    assert doc["verified"] is True and doc["receipt"]["prefix_layers"] == P and doc["receipt"]["new_rungs"] == N - P

    def elsewhere(rec: dict) -> None:
        rec["layers"][N - 1]["resumed_from"] = "/elsewhere/L03.safetensors"
    w = fresh("elsewhere", elsewhere)
    with pytest.raises(SystemExit, match="resumed from /elsewhere"):
        synced(w)
    refused(w, tmp_path / "e.json", "resumed from /elsewhere")

    def foreign(rec: dict) -> None:
        rec["layers"][0]["rung"]["sha256"] = "9" * 64
    w = fresh("foreign", foreign)
    with pytest.raises(SystemExit, match="canonical .* != source receipt"):
        synced(w)
    refused(w, tmp_path / "f.json", "boundary rung sha")

    w = synced(fresh("chain"))   # a replayed rung whose input_chain is not the recomputed chain
    rj = w["run"] / "trainer" / "rungs" / f"L{N - 1:02d}.json"
    rj.write_text(json.dumps({**json.loads(rj.read_text()), "input_chain": "0" * 64}))
    refused(w, tmp_path / "c.json", "input_chain")

