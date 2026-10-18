#!/usr/bin/env python3
"""chimera_reassemble.py -- write ONE Chimera Omni checkpoint from the trained parts.

The full-parameter KD trains the text weights in the qwen3_5 layout (`streamed_full_kd_trainer` / `omni_full_kd_trainer`
export: a HostWeightStore safetensors directory) and the media modules in Chimera names (`media_full_kd_trainer` /
`omni_full_kd_trainer` export: `media_modules.safetensors`, with `music.head` / `music.embed` for the C18 code rows).
The runtime loads a Chimera child, so this tool maps everything back:

  * every qwen3_5 text tensor goes back through the P1 reader's OWN inverse, `p1_qwen35_reader.importer_forward` (the
    importer's literal rewrite: split of q/k/v and conv, the q/gate un-interleave, norms stored as 1 + w, the gated
    norm broadcast) into its Chimera part(s); embed / lm_head rows [0, base) only;
  * a norm stored as 1 + w is kept in the base dtype when that is exact, else written in fp32 (bf16 cannot hold
    1 + w exactly for small w; the reader serves fp32 norms exactly);
  * media tensors replace their Chimera names; music.head / music.embed replace the written code rows of lm_head /
    embed in the C18 band;
  * every other tensor is the base child's value (the D1 delta plan composed, exactly as training read it).

The output has the base child's shard grouping, an index, the base's non-tensor files, and REASSEMBLY.json.  `verify`
re-opens the output with the P1 reader and checks every text tensor against the trained value (bit-exact, or exact in
fp32 for widened norms), every media tensor, the music rows, and that untouched tensors equal the base.

  build   --base CHILD [--plan P1_PLAN] --text TEXT_DIR [--media media_modules.safetensors] --out OUT
  verify  --out OUT --base CHILD [--plan ...] --text TEXT_DIR [--media ...]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

TOOL_NAME = "chimera_reassemble"
TOOL_VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent


def _p1() -> Any:
    name = "p1_qwen35_reader"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "p1_qwen35_reader.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class SafeDir:
    """Tensors of a safetensors directory by name, read on demand."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(self.dir / self.weight_map[name]), framework="pt", device="cpu") as h:
            return h.get_tensor(name)


def open_base(args: argparse.Namespace) -> Any:
    return _p1().open_reader(str(args.base), plan_path=str(args.plan) if args.plan else None)


def owners(reader: Any) -> dict[str, list[tuple[str, int, Any]]]:
    """Chimera name -> [(qwen name, part index, rows)] for every qwen tensor of the map."""

    out: dict[str, list[tuple[str, int, Any]]] = {}
    for q, spec in reader.specs.items():
        for k, (cname, rows) in enumerate(spec["parts"]):
            out.setdefault(cname, []).append((q, k, rows))
    return out


def chimera_parts(reader: Any, q: str, value: torch.Tensor) -> list[torch.Tensor]:
    p1 = _p1()
    spec = dict(reader.specs[q])
    kind = spec["kind"]
    if kind == "cat":
        spec["part_rows"] = [reader.shards.spec(c)["shape"][0] for c, _ in spec["parts"]]
    if kind == "row0":
        spec["broadcast_rows"] = reader.shards.spec(spec["parts"][0][0])["shape"][0]
    return p1.importer_forward(q, kind, value, spec)


def build(args: argparse.Namespace) -> int:
    from safetensors.torch import load_file, save_file

    started = time.perf_counter()
    reader = open_base(args)
    text = SafeDir(Path(args.text))
    missing = sorted(set(reader.specs) - set(text.weight_map))
    if missing:
        raise SystemExit(f"the text directory lacks {len(missing)} mapped tensors (first {missing[:3]})")
    media = load_file(str(args.media)) if args.media else {}
    music = {k: media.pop(k) for k in ("music.head", "music.embed") if k in media}
    unknown = sorted(k for k in media if not reader.shards.has(k))
    if unknown:
        raise SystemExit(f"media tensors not in the base child: {unknown[:4]}")
    own = owners(reader)
    out = Path(args.out)
    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    music_begin = None
    # the C18 band start, read from the composite config the same way d3_hybrid_module does (ChimeraConfig ledger)
    try:
        from chimera_omni.config import ChimeraConfig

        cfg = ChimeraConfig.from_manifest(reader.manifest["config"])
        music_begin = int(cfg.ledger().range("music_codes").begin)
    except Exception as exc:  # noqa: BLE001 - only needed when music rows are given
        if music:
            raise SystemExit(f"music rows given but the C18 band start is unreadable: {exc}")
    by_file: dict[str, list[str]] = {}
    for cname in reader.shards.names():
        by_file.setdefault(reader.shards.spec(cname)["file"], []).append(cname)
    record: dict[str, Any] = {"widened_norms": [], "text_tensors": 0, "media_tensors": 0, "music_rows": None, "files": []}
    parts_cache: dict[str, list[torch.Tensor]] = {}
    for filename, cnames in sorted(by_file.items()):
        tensors: dict[str, torch.Tensor] = {}
        for cname in sorted(cnames):
            base = reader._raw(cname)
            value = base.clone()
            for q, k, rows in own.get(cname, []):
                if q not in parts_cache:
                    parts_cache[q] = chimera_parts(reader, q, text.get(q))
                part = parts_cache[q][k]
                if rows:
                    value[rows[0]:rows[1]] = part.to(value.dtype)
                elif reader.specs[q]["kind"] == "one_plus":
                    exact = part.float()
                    cast = exact.to(value.dtype)
                    if bool((cast.float() == exact).all()):
                        value = cast
                    else:
                        value = exact
                        record["widened_norms"].append(cname)
                else:
                    value = part.to(value.dtype).reshape(value.shape)
                record["text_tensors"] += 1
            if cname in media:
                if tuple(media[cname].shape) != tuple(value.shape):
                    raise SystemExit(f"{cname}: media shape {tuple(media[cname].shape)} != base {tuple(value.shape)}")
                value = media[cname].to(value.dtype)
                record["media_tensors"] += 1
            if music and cname in ("embed.weight", "lm_head.weight"):
                rows = music["music.embed" if cname == "embed.weight" else "music.head"]
                value[music_begin:music_begin + rows.shape[0]] = rows.to(value.dtype)
                record["music_rows"] = [music_begin, music_begin + int(rows.shape[0])]
            tensors[cname] = value.contiguous()
        save_file(tensors, str(tmp / filename), metadata={"format": "pt", "tool": f"{TOOL_NAME} {TOOL_VERSION}"})
        record["files"].append({"file": filename, "tensors": len(tensors), "sha256": sha256_file(tmp / filename)})
        parts_cache.clear()
        log(f"{filename}: {len(tensors)} tensors")
    weight_map = {c: reader.shards.spec(c)["file"] for c in reader.shards.names()}
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"tool": TOOL_NAME}, "weight_map": weight_map}, indent=1),
                                                      encoding="utf-8")
    for f in Path(args.base).iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, tmp / f.name)
    receipt = {"schema": "neural-forge-omni-reassembly-v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
               "base": str(args.base), "plan": str(args.plan) if args.plan else None, "text": str(args.text),
               "media": str(args.media) if args.media else None,
               "media_sha256": sha256_file(Path(args.media)) if args.media else None, **record,
               "seconds": round(time.perf_counter() - started, 1)}
    (tmp / "REASSEMBLY.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    tmp.rename(out)
    log(f"REASSEMBLED {out}: {record['text_tensors']} text parts, {record['media_tensors']} media tensors, music rows "
        f"{record['music_rows']}, {len(record['widened_norms'])} norms widened to fp32")
    return 0


def verify(args: argparse.Namespace) -> int:
    from safetensors.torch import load_file

    p1 = _p1()
    base = open_base(args)
    built = p1.open_reader(str(args.out), plan_path=None)
    text = SafeDir(Path(args.text))
    media = load_file(str(args.media)) if args.media else {}
    music = {k: media.pop(k) for k in ("music.head", "music.embed") if k in media}
    receipt = json.loads((Path(args.out) / "REASSEMBLY.json").read_text(encoding="utf-8"))
    bad_text, bad_media, bad_other = [], [], []
    widened = set(receipt.get("widened_norms", []))
    norm_gap = 0.0
    for q in base.specs:
        want = text.get(q)
        if base.specs[q]["kind"] == "one_plus":
            # the invariant is the STORED value: 1 + w, exactly as the importer writes it, in the dtype the build chose
            cname = base.specs[q]["parts"][0][0]
            stored = built.shards.view(cname)
            expected = (want.float() + 1.0).to(stored.dtype)
            if not torch.equal(stored, expected):
                bad_text.append(q)
            norm_gap = max(norm_gap, float((built.get(q).float() - want.float()).abs().max()))
            continue
        got = built.get(q).to(torch.float32)
        if not torch.equal(got, want.to(torch.float32)):
            bad_text.append(q)
    touched = {c for c, _ in ((c, None) for spec in base.specs.values() for c, _ in spec["parts"])} | set(media)
    for c, v in media.items():
        if not torch.equal(built.shards.view(c).float(), v.to(built.shards.view(c).dtype).float()):
            bad_media.append(c)
    mb = receipt.get("music_rows")
    if music and mb:
        for c, key in (("embed.weight", "music.embed"), ("lm_head.weight", "music.head")):
            if not torch.equal(built.shards.view(c)[mb[0]:mb[1]].float(), music[key].to(built.shards.view(c).dtype).float()):
                bad_media.append(f"{c}[music rows]")
    for c in base.shards.names():
        if c in touched:
            continue
        if not torch.equal(built.shards.view(c), base._raw(c)):
            bad_other.append(c)
    result = {"text_mismatch": bad_text[:10], "text_checked": len(base.specs), "norm_served_max_gap": norm_gap,
              "media_mismatch": bad_media[:10],
              "untouched_mismatch": bad_other[:10], "pass": not (bad_text or bad_media or bad_other)}
    print(json.dumps(result, indent=1), flush=True)
    return 0 if result["pass"] else 2


def selftest(args: argparse.Namespace) -> int:
    """Tiny Chimera child: perturb every text tensor and the media modules, reassemble, verify, and check the P1 reader
    of the result serves the perturbed values."""

    import tempfile
    from types import SimpleNamespace

    d3 = _load_peer("d3_hybrid_module")
    fk = _load_peer("streamed_full_kd_trainer")
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        built = d3.build_tiny_child(Path(tmp) / "child")
        src = fk.skt.P1Source(Path(built["dir"]), Path(built["parent_config"]), None)
        store = fk.HostWeightStore(src, dtype=torch.float32)
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for name, t in store.tensors.items():
                t.add_(torch.randn(t.shape, generator=g) * 0.01)
        store.save(Path(tmp) / "text")
        media = {}
        for name in src.reader.extras():
            if name.startswith(("blocks.",)) and ".charted." in name or name.startswith("native_media_head."):
                v = src.reader.extra(name)
                if v.is_floating_point():
                    media[name] = (v.float() + torch.randn(v.shape, generator=g) * 0.01).to(v.dtype)
        save_file({k: v.contiguous() for k, v in media.items()}, str(Path(tmp) / "media.safetensors"))
        ns = SimpleNamespace(base=built["dir"], plan=None, text=str(Path(tmp) / "text"), media=str(Path(tmp) / "media.safetensors"),
                             out=str(Path(tmp) / "out"))
        rc = build(ns)
        rc2 = verify(ns)
        print(json.dumps({"selftest": "PASS" if rc == 0 and rc2 == 0 else "FAIL", "media_tensors": len(media)}), flush=True)
        return 0 if rc == 0 and rc2 == 0 else 2


def _load_peer(name: str) -> Any:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--base", required=True)
        p.add_argument("--plan", default=None)
        p.add_argument("--text", required=True)
        p.add_argument("--media", default=None)
        p.add_argument("--out", required=True)
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.command == "selftest":
        return selftest(args)
    return build(args) if args.command == "build" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
