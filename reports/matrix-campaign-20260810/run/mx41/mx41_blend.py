#!/usr/bin/env python3
"""COMPLEMENT-MERGE: convex blends of two sibling children at identical storage.

`INTELLIGENCE_RETENTION_PLAN_2026-08-14.md` #7.  `ARCH.V19X13` (10/15, holds the
coding-dedup list and the arithmetic chain) and `ARCH.V19X13G01` (11/15, gains
four lanes and loses dedup on every base it has ever been run on) differ in
**exactly 1,024 numbers**: the `model.embedding_norm` diagonal that
`mx41_termgain.py` overwrites in place.  Every other tensor is byte-identical.

That makes any convex blend of the two diagonals a legal closed-form child at
byte-identical storage -- no solve, no capture, no added parameter, one CPU pass
-- and it makes the question sharp: **is the terminal-gain case swap
coordinate-separable, or is it one indivisible trade?**

Two arms, as the plan specifies:

* ``--alpha`` : d = (1-a) * d_source + a * d_other, for a in {0.25, 0.5, 0.75}.
  If the union of the two children's held cases exists anywhere on the segment,
  an interior point finds it.  If interior points read *below both endpoints*,
  the segment is a ridge and the trade is not separable by mixing.
* ``--partition`` : take the top-|delta| coordinates from the other child and the
  rest from the source.  This asks the separability question directly: if the
  LIST-to-SET regression is carried by a handful of coordinates, a partition that
  excludes them keeps the gains and drops the loss.

**The refusals are the point.**  The two sources must agree on every tensor
except the diagonal, must agree on parameter count, and must agree on config.  A
"blend" of two children that differ somewhere else is not a blend of a diagonal,
and its reading would be attributed to the wrong thing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PARENT_PARAMS = 229_693_184
DIAG = "model.embedding_norm.weight"
SCHEMA = "neural-forge-matrix-wave41-blend-build-receipt-v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="the child whose bytes are kept")
    ap.add_argument("--other", type=Path, required=True, help="the sibling whose diagonal is mixed in")
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--alpha", type=float, help="convex weight on --other's diagonal")
    ap.add_argument("--partition", type=int, help="take this many top-|delta| coordinates from --other")
    ap.add_argument(
        "--restore",
        type=Path,
        help=(
            "SURGICAL arm: take --other's diagonal WHOLE and restore --source's "
            "values only on the coordinates named in this `mx41_coordsearch.py` "
            "artifact.  The inverse of --partition in both directions: the base "
            "is the sibling rather than the source, and the coordinate set is "
            "chosen by closed-form margin attribution on the lane being "
            "protected rather than by the size of the diagonal's own delta"
        ),
    )
    ap.add_argument(
        "--restore-k",
        type=int,
        help="which `restore_sets` entry of the --restore artifact to use",
    )
    ap.add_argument("--source-cell-receipt", type=Path)
    ap.add_argument("--other-cell-receipt", type=Path)
    args = ap.parse_args()
    arms = [args.alpha is not None, args.partition is not None, args.restore is not None]
    if sum(arms) != 1:
        raise SystemExit("REFUSED: exactly one of --alpha, --partition or --restore")
    if (args.restore is None) != (args.restore_k is None):
        raise SystemExit("REFUSED: --restore and --restore-k travel together")

    a_tensors = load_file(str(args.source / "model.safetensors"))
    b_tensors = load_file(str(args.other / "model.safetensors"))

    if sorted(a_tensors) != sorted(b_tensors):
        raise SystemExit("REFUSED: the two children do not carry the same tensor set")
    differing = [
        k for k in sorted(a_tensors)
        if a_tensors[k].shape != b_tensors[k].shape
        or not torch.equal(a_tensors[k], b_tensors[k])
    ]
    if differing != [DIAG]:
        raise SystemExit(
            f"REFUSED: the two children differ in {differing[:6]} (n={len(differing)}), "
            f"not in {DIAG} alone.  A blend of these is not a blend of a diagonal."
        )
    a_cfg = json.loads((args.source / "config.json").read_text())
    b_cfg = json.loads((args.other / "config.json").read_text())
    if a_cfg != b_cfg:
        raise SystemExit("REFUSED: the two children do not share a config")

    da = a_tensors[DIAG].to(torch.float64)
    db = b_tensors[DIAG].to(torch.float64)
    delta = (db - da).abs()

    if args.alpha is not None:
        blended = (1.0 - args.alpha) * da + args.alpha * db
        arm = f"convex alpha={args.alpha}"
        detail = {"alpha": args.alpha}
    elif args.restore is not None:
        art = json.loads(args.restore.read_text(encoding="utf-8"))
        key = str(int(args.restore_k))
        sets = art.get("restore_sets") or {}
        if key not in sets:
            raise SystemExit(
                f"REFUSED: {args.restore} has no restore set of size {key}; it "
                f"carries {sorted(sets)}"
            )
        coords = [int(i) for i in sets[key]]
        blended = db.clone()
        blended[torch.tensor(coords, dtype=torch.long)] = da[
            torch.tensor(coords, dtype=torch.long)
        ]
        arm = (
            f"surgical restore, {len(coords)} coordinates of the source returned "
            f"into the sibling's diagonal"
        )
        detail = {
            "restore_k": len(coords),
            "coordinates": sorted(coords),
            "artifact": str(args.restore),
            "protected_lanes": art.get("protected_lanes"),
            "gain_lanes": art.get("gain_lanes"),
            "reconstruction_gate_worst_relative": art.get(
                "reconstruction_gate_worst_relative"
            ),
            "selection": (
                "coordinates whose A->B move costs the protected lane's own margin "
                "and does not pay any gain lane's, ranked by the net"
            ),
        }
    else:
        k = int(args.partition)
        idx = torch.topk(delta, k).indices
        blended = da.clone()
        blended[idx] = db[idx]
        arm = f"per-entry partition, top-{k} |delta| coordinates from the sibling"
        detail = {
            "partition_k": k,
            "coordinates": sorted(int(i) for i in idx.tolist()),
            "delta_at_partition": round(float(delta[idx].min()), 8),
            "delta_max": round(float(delta.max()), 8),
        }

    outdir = args.child
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    for name in (
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
        "vocab.json", "merges.txt",
    ):
        if (args.source / name).is_file():
            shutil.copyfile(args.source / name, outdir / name)

    out_tensors = dict(a_tensors)
    out_tensors[DIAG] = blended.to(a_tensors[DIAG].dtype).clone()
    save_file(out_tensors, str(outdir / "model.safetensors"), metadata={"format": "pt"})

    live = int(sum(int(t.numel()) for t in out_tensors.values()))
    # `mx*-build-receipt.json`, not `mx41-`: a blend's sources are SEALED
    # children from earlier waves and their receipts are named for the wave that
    # built them.  Looking for this wave's name found nothing, defaulted every
    # carried field to None, and the sealer then refused on `excised` being
    # None -- which is the right refusal for the wrong reason, and cost two
    # reviews.  Every other script in this tree globs; this one now does too.
    source_receipts = sorted(args.source.glob("mx*-build-receipt.json"))
    if not source_receipts:
        raise SystemExit(
            f"REFUSED: {args.source} carries no mx*-build-receipt.json, so the "
            "blend cannot carry its source's axes and would seal with incomplete "
            "provenance."
        )
    source_build = json.loads(source_receipts[-1].read_text(encoding="utf-8"))
    if source_build and live != int(source_build["child_parameters"]):
        raise SystemExit(
            f"the blend changed the parameter count: {live} against the source's "
            f"{source_build['child_parameters']}.  It overwrites values."
        )

    receipt = {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "mx41_blend.py",
        "tag": str(args.tag),
        "arm": "COMPLEMENT-MERGE sibling diagonal blend at identical storage: " + arm,
        "blend": detail,
        "source_model": str(args.source),
        "other_model": str(args.other),
        "source_cell_receipt": str(args.source_cell_receipt or ""),
        "other_cell_receipt": str(args.other_cell_receipt or ""),
        "source_tensors_sha256": hashlib.sha256(
            (args.source / "model.safetensors").read_bytes()).hexdigest(),
        "other_tensors_sha256": hashlib.sha256(
            (args.other / "model.safetensors").read_bytes()).hexdigest(),
        "differing_tensors_between_sources": differing,
        "coordinates_changed_from_source": int((blended != da).sum()),
        "child_model": str(outdir),
        "child_tensors_sha256": hashlib.sha256(
            (outdir / "model.safetensors").read_bytes()).hexdigest(),
        "child_parameters": live,
        "parent_parameters": PARENT_PARAMS,
        "net_stored_ratio": live / PARENT_PARAMS,
        "parameters_added": 0,
        "excised": [int(x) for x in (source_build.get("excised") or [])],
        "residual_basis": source_build.get("residual_basis"),
        "axes": source_build.get("axes", []),
        "note": (
            "A convex combination of two sealed children's terminal diagonals.  No "
            "capture, no solve, no gradient, no added parameter: one CPU pass over "
            "1,024 numbers, at byte-identical storage to both endpoints."
        ),
    }
    (outdir / "mx41-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k != "blend"}, indent=1))
    print(f"blend {args.tag}: {receipt['coordinates_changed_from_source']} of 1024 coordinates moved, "
          f"{live} parameters, {live / PARENT_PARAMS:.6f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
