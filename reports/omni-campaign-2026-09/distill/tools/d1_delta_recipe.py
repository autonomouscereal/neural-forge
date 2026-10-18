#!/usr/bin/env python3
"""d1_delta_recipe.py -- D1's delta -> the ledger's delta recipe, so `oc_checkpoint.py register --delta-base` can take it.

Trainer-prep lane, 2026-10-13 (E4).  Two formats meet here and neither tool knows the other:

  * d1_trunk_recovery.py (1.2.4) writes ``D1_delta.safetensors`` + ``D1_delta.json`` (schema
    ``omni-campaign.d1-delta.v1``): tensor keys ``<base tensor>::<part>``, each entry's slice as one
    ``[begin, end]`` pair PER DIMENSION, plus the D1 base it was trained on (path, config / index sha256);
  * oc_checkpoint.py (1.1.0) registers a delta only from ``DELTA.json`` (schema
    ``neural-forge-omni-delta-recipe-v1``): ops replace / lora / scale on the BASE's tensor names, slices as
    ``[dim, start, stop]``, the base named by checkpoint id AND pinned by its sealed ``directory_sha256``.

This tool writes the second from the first, into the delta directory, and writes nothing else.  Every D1 entry
becomes one ``replace`` (D1 overwrites trunk slices; it adds no parameters).  Refused (exit 2) before anything
is written: a D1 manifest whose ``.sha256`` sidecar or file sha does not match; a delta tensor missing, or of
another shape / dtype / byte sha than its entry says; a D1 base directory that is not the ledger base's
directory; a D1 base whose config / index sha256 differ from the ledger base's files now; a ledger base record
without a sealed directory digest.  The ledger then re-checks every op against the base's headers
(``register``) and re-hashes the base (``delta-compose --deep``); the P1 reader composes from that plan.

Usage (the D1 launcher, after the D1 run and before registration):
  $L show --checkpoint-id OC02-F-Q38-27BU-OMNI1.clone-pre-D1 > base_record.json
  $PY d1_delta_recipe.py --delta-dir $DELTA --base-record base_record.json
  $PY $L register --checkpoint-id OC02-D-Q38-27BU-OMNI1D1 --stage distill --directory $DELTA \\
      --delta-base OC02-F-Q38-27BU-OMNI1.clone-pre-D1 ...            # dry run first, then --confirm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

TOOL_NAME = "d1_delta_recipe"
TOOL_VERSION = "1.0.0"  # 2026-10-13: first version (trainer-prep lane, E4)
D1_DELTA_SCHEMA = "omni-campaign.d1-delta.v1"       # d1_trunk_recovery.DELTA_SCHEMA
RECIPE_SCHEMA = "neural-forge-omni-delta-recipe-v1"  # oc_checkpoint.DELTA_RECIPE_SCHEMA
RECIPE_NAME = "DELTA.json"                           # oc_checkpoint.DELTA_RECIPE_NAME


class Refusal(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


def log(message: str) -> None:
    print(f"[{TOOL_NAME}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes_sha256(tensor: Any) -> str:
    """d1_trunk_recovery.tensor_bytes_sha256: the tensor's own bytes in its own dtype."""
    import torch

    t = tensor.detach().contiguous().cpu()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def recipe_slice(pairs: list[list[int]]) -> list[list[int]]:
    """D1's per-dimension [begin, end] pairs -> the ledger's [dim, start, stop] entries (every dim, explicit)."""
    out = []
    for dim, pair in enumerate(pairs):
        if len(pair) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in pair):
            raise Refusal(f"slice pair {pair!r} on dim {dim} is not two integers")
        begin, end = pair
        if not 0 <= begin < end:
            raise Refusal(f"slice [{begin}, {end}) on dim {dim} is empty or negative")
        out.append([dim, int(begin), int(end)])
    return out


def load_d1_delta(delta_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the D1 manifest, {key: tensor}) after every integrity check d1's own writer makes possible."""
    from safetensors.torch import load_file

    manifest_path = delta_dir / "D1_delta.json"
    if not manifest_path.is_file():
        raise Refusal(f"{delta_dir} has no D1_delta.json (not a d1_trunk_recovery delta directory)")
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    manifest_sha = sha256_file(manifest_path)
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] != manifest_sha:
        raise Refusal(f"{manifest_path} does not match its .sha256 sidecar (edited after d1 wrote it?)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != D1_DELTA_SCHEMA:
        raise Refusal(f"{manifest_path}: schema {manifest.get('schema')!r} != {D1_DELTA_SCHEMA}")
    path = delta_dir / Path(str(manifest["file"])).name
    if not path.is_file():
        raise Refusal(f"{path} is missing")
    if sha256_file(path) != manifest["file_sha256"]:
        raise Refusal(f"{path} does not digest to the manifest's file_sha256 {str(manifest['file_sha256'])[:16]}")
    tensors = load_file(str(path))
    keys = [e["key"] for e in manifest["entries"]]
    if len(set(keys)) != len(keys):
        raise Refusal("the D1 manifest names one tensor twice")
    if set(keys) != set(tensors):
        raise Refusal(f"D1 manifest entries and {path.name} disagree: only in manifest "
                      f"{sorted(set(keys) - set(tensors))[:3]}, only in file {sorted(set(tensors) - set(keys))[:3]}")
    for entry in manifest["entries"]:
        t = tensors[entry["key"]]
        if list(t.shape) != list(entry["shape"]) or str(t.dtype).replace("torch.", "") != entry["dtype"]:
            raise Refusal(f"{entry['key']}: {list(t.shape)} {t.dtype} in the file, the manifest says "
                          f"{entry['shape']} {entry['dtype']}")
        if tensor_bytes_sha256(t) != entry["sha256"]:
            raise Refusal(f"{entry['key']}: bytes differ from the manifest's sha256")
    return manifest, tensors


def check_base(manifest: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """The D1 base must BE the ledger base: same directory, same config / index bytes now."""
    checkpoint = record.get("checkpoint") or {}
    directory, sealed = checkpoint.get("directory"), checkpoint.get("directory_sha256")
    if not directory or not isinstance(sealed, str) or len(sealed) != 64:
        raise Refusal("the base record has no checkpoint.directory / sealed checkpoint.directory_sha256")
    base = manifest.get("base") or {}
    trained_on = base.get("path")
    if trained_on is None or Path(trained_on).resolve() != Path(directory).resolve():
        raise Refusal(f"D1 was trained on {trained_on}, but the ledger base {record.get('checkpoint_id')} lives at "
                      f"{directory}; a delta registered on another base composes onto bytes it was not trained on")
    for name, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        live = sha256_file(Path(directory) / name)
        if base.get(key) != live:
            raise Refusal(f"{directory}/{name} digests {live[:16]} now; D1 recorded {str(base.get(key))[:16]}")
    return {"checkpoint_id": record.get("checkpoint_id"), "directory": directory, "directory_sha256": sealed,
            "d1_base_digest_mode": base.get("mode")}


def build_recipe(manifest: dict[str, Any], base: dict[str, Any], manifest_sha: str) -> dict[str, Any]:
    entries = [{"op": "replace", "target": e["base_tensor"], "tensor": e["key"], "slice": recipe_slice(e["slice"]),
                "text_layer": e.get("text_layer"), "part": e.get("part")}
               for e in sorted(manifest["entries"], key=lambda e: e["key"])]
    if not entries:
        raise Refusal("the D1 delta trained no layer")
    return {"schema": RECIPE_SCHEMA, "base_checkpoint_id": base["checkpoint_id"],
            "base_directory_sha256": base["directory_sha256"], "entries": entries,
            "source": {"tool": TOOL_NAME, "version": TOOL_VERSION, "d1_manifest_sha256": manifest_sha,
                       "d1_file_sha256": manifest["file_sha256"], "d1_tool": manifest.get("tool"),
                       "trained_text_layers": manifest.get("trained_text_layers"),
                       "parameters_replaced": manifest.get("parameters")}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    ap.add_argument("--delta-dir", required=True, help="d1_trunk_recovery's --delta-out (holds D1_delta.json)")
    ap.add_argument("--base-record", required=True, help="`oc_checkpoint.py show --checkpoint-id <base>` output (JSON)")
    ap.add_argument("--base-checkpoint-id", default=None, help="must equal the record's checkpoint_id when given")
    ap.add_argument("--force", action="store_true", help="replace an existing DELTA.json")
    args = ap.parse_args(argv)
    try:
        delta_dir = Path(args.delta_dir).absolute()
        record = json.loads(Path(args.base_record).read_text(encoding="utf-8"))
        if args.base_checkpoint_id and record.get("checkpoint_id") != args.base_checkpoint_id:
            raise Refusal(f"--base-record is {record.get('checkpoint_id')!r}, not {args.base_checkpoint_id!r}")
        manifest, _ = load_d1_delta(delta_dir)
        base = check_base(manifest, record)
        recipe = build_recipe(manifest, base, sha256_file(delta_dir / "D1_delta.json"))
        out = delta_dir / RECIPE_NAME
        if out.exists() and not args.force:
            raise Refusal(f"{out} exists (pass --force to replace it)")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(recipe, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, out)
    except Refusal as refusal:
        print(f"[{TOOL_NAME}] REFUSED: {refusal.message}", file=sys.stderr, flush=True)
        return 2
    log(f"{out}: {len(recipe['entries'])} replace entries on {base['checkpoint_id']} "
        f"(sealed {base['directory_sha256'][:16]}); sha256 {sha256_file(out)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
