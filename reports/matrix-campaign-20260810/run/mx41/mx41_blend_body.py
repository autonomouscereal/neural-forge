#!/usr/bin/env python3
"""BODY-BLEND: convex blends of two or three sealed sibling BODIES.

Wave 29's `mx41_blend.py` blends two children that differ in **exactly 1,024
numbers** -- the terminal `model.embedding_norm` diagonal -- and refuses
anything else, because a blend of two children that differ somewhere else is a
blend of something the receipt cannot name.  That refusal is correct and this
file does not weaken it; it answers a different question about a different pair
of tensors, and it states the reason a straight convex blend of two `ARCH.CL*`
siblings is NOT the thing anyone means by one.

**The channel-identity problem, which is the whole design.**  `mx41_build.py`'s
`intermediate_size` fold is a SELECTION, not a projection: `pivoted_columns`
picks `keep_ff` real channels of the parent's 2,560, `w1` and `w3` keep those
parent rows VERBATIM, and only `w2` is re-solved on the surviving columns.  Two
siblings that disagree about which channels to keep -- and wave 31 measured
`ARCH.CL1792MV` agreeing with `ARCH.CL1792`'s plain selection on 1,557 to 1,686
of 1,792 channels per site -- therefore store *the same parent rows in a
different order*, plus a few hundred rows the other does not have at all.  Row
`i` of one child's `w1` is parent channel `a`; row `i` of the other's is parent
channel `b`.  A convex blend of row `i` with row `i` is a blend of two unrelated
channels, and its reading would be attributed to "the blend" when it is really a
reading of a scrambled basis.

So the builder recovers each sibling's selection first, by matching each child
`w1`/`w3` row pair against the parent's own rows.  The match is EXACT -- these
are copied bf16 bytes, not approximations -- and a row that matches no parent
row, or matches more than one, is a refusal rather than a nearest neighbour.

Two arms, and the second one exists to be measured rather than assumed:

* ``--align channel`` -- the aligned blend.  The child keeps the SOURCE's
  selection and the SOURCE's row order, so `w1` and `w3` are byte-identical to
  the source's and the shape, the config and the parameter count cannot move.
  `w2` is blended COLUMN-WISE over the aligned intersection: column `i` of the
  child is the convex combination of every sibling's own solved coefficient for
  *the same parent channel*.  A channel the sibling did not keep contributes
  nothing and its weight is renormalised onto the siblings that did, because a
  column scaled by `1-a` is not a convex combination of anything.  Every tensor
  outside the feed-forward interior -- attention, conv, norms, the tied table --
  carries no permutation ambiguity and is blended straight.
* ``--align none`` -- the naive blend, every tensor combined entry by entry
  including `w1` and `w3`.  This is the arm the paragraph above says is
  meaningless.  It is built and reviewed anyway, once, because "meaningless" is
  a prediction and the campaign's standard is that a prediction gets a number.

``--census`` writes the alignment census and builds nothing, so the cost of
finding out whether two bodies are alignable at all is one CPU pass and no
checkpoint.

**The refusals.**  Every sibling must carry the same tensor set at the same
shapes, the same config, and the same live parameter count; the source must
carry a build receipt, so the blend inherits the axes it is a blend of; the
weights must be non-negative and sum to at most one; and ``--align channel``
requires the parent, because without it there is no channel identity to align
on.
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
SCHEMA = "neural-forge-matrix-wave41-body-blend-build-receipt-v1"
#: The three tensors the `intermediate_size` fold owns.  `w1`/`w3` are verbatim
#: parent rows under the selection's own order; `w2` is the only solved one.
W1 = "feed_forward.w1.weight"
W3 = "feed_forward.w3.weight"
W2 = "feed_forward.w2.weight"


def _ff_sites(tensors: dict[str, torch.Tensor]) -> list[str]:
    """Every layer prefix that carries a feed-forward triple, in layer order."""

    prefixes = sorted(
        {k[: -len(W1)] for k in tensors if k.endswith(W1)},
        key=lambda p: int(p.split(".")[2]),
    )
    return prefixes


def _row_key(w1: torch.Tensor, w3: torch.Tensor, i: int) -> bytes:
    """The exact bytes of one channel's (w1, w3) row pair.

    Widened to fp32 first, which is EXACT from bf16 and injective, because numpy
    has no bfloat16 and a comparison that silently rounded would turn "verbatim"
    into "close enough" -- which is the one thing this recovery may not do.
    """

    return (
        w1[i].to(torch.float32).contiguous().numpy().tobytes()
        + w3[i].to(torch.float32).contiguous().numpy().tobytes()
    )


def parent_channel_index(parent: dict[str, torch.Tensor], prefix: str) -> dict[bytes, int]:
    """Map each parent channel's own (w1 row, w3 row) bytes to its index.

    The pair rather than either alone: a duplicated `w1` row is possible in
    principle and a duplicated `(w1, w3)` pair is what would actually make two
    channels indistinguishable to the fold.  Duplicates are refused rather than
    resolved by position, because resolving them by position is guessing.
    """

    w1 = parent[prefix + W1]
    w3 = parent[prefix + W3]
    index: dict[bytes, int] = {}
    for j in range(int(w1.shape[0])):
        key = _row_key(w1, w3, j)
        if key in index:
            raise SystemExit(
                f"REFUSED: {prefix} parent channels {index[key]} and {j} are "
                "byte-identical in both w1 and w3, so a recovered selection "
                "cannot name which one a child kept"
            )
        index[key] = j
    return index


def recover_selection(
    child: dict[str, torch.Tensor], parent_index: dict[bytes, int], prefix: str, label: str
) -> list[int]:
    """The parent channel each of the child's rows IS, in the child's own order."""

    w1 = child[prefix + W1]
    w3 = child[prefix + W3]
    sel: list[int] = []
    for i in range(int(w1.shape[0])):
        got = parent_index.get(_row_key(w1, w3, i))
        if got is None:
            raise SystemExit(
                f"REFUSED: {label} {prefix} row {i} matches no parent channel.  "
                "This body's w1/w3 rows are not verbatim parent rows, so its "
                "selection cannot be recovered and it cannot be aligned."
            )
        sel.append(got)
    if len(set(sel)) != len(sel):
        raise SystemExit(f"REFUSED: {label} {prefix} recovered a selection with repeats")
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="the body whose selection, row order and unmatched columns are kept")
    ap.add_argument("--other", type=Path, action="append", default=[],
                    help="a sibling body to mix in; repeatable, paired with --weight")
    ap.add_argument("--weight", type=float, action="append", default=[],
                    help="the convex weight on the matching --other; the source takes the rest")
    ap.add_argument("--parent", type=Path,
                    help="the parent checkpoint, for channel identity; required by --align channel")
    ap.add_argument("--child", type=Path)
    ap.add_argument("--tag")
    ap.add_argument("--align", choices=("channel", "none"), default="channel")
    ap.add_argument("--census", action="store_true",
                    help="write the alignment census and build nothing")
    ap.add_argument("--out-census", type=Path)
    args = ap.parse_args()

    if len(args.other) != len(args.weight):
        raise SystemExit("REFUSED: --other and --weight travel together, one weight each")
    if not args.other:
        raise SystemExit("REFUSED: a blend needs at least one --other")
    if any(w < 0.0 for w in args.weight):
        raise SystemExit("REFUSED: a convex weight cannot be negative")
    total_other = sum(args.weight)
    if total_other > 1.0 + 1e-12:
        raise SystemExit(
            f"REFUSED: the --weight list sums to {total_other}, which leaves the "
            "source a negative weight.  This builder writes convex combinations."
        )
    if not args.census and (args.child is None or args.tag is None):
        raise SystemExit("REFUSED: a build needs --child and --tag")
    if args.align == "channel" and args.parent is None:
        raise SystemExit(
            "REFUSED: --align channel recovers each sibling's channel selection "
            "against the parent's own rows, so --parent is not optional"
        )

    source_w = 1.0 - total_other
    src = load_file(str(args.source / "model.safetensors"))
    others = [load_file(str(p / "model.safetensors")) for p in args.other]
    labels = [p.name for p in args.other]

    for label, oth in zip(labels, others):
        if sorted(src) != sorted(oth):
            raise SystemExit(f"REFUSED: {label} does not carry the source's tensor set")
        bad = [k for k in sorted(src) if src[k].shape != oth[k].shape]
        if bad:
            raise SystemExit(f"REFUSED: {label} disagrees on the shape of {bad[:6]}")
    src_cfg = json.loads((args.source / "config.json").read_text())
    for label, path in zip(labels, args.other):
        if json.loads((path / "config.json").read_text()) != src_cfg:
            raise SystemExit(f"REFUSED: {label} does not share the source's config")

    sites = _ff_sites(src)
    census: list[dict[str, object]] = []
    selections: dict[str, dict[str, list[int]]] = {}
    if args.align == "channel":
        parent = load_file(str(args.parent / "model.safetensors"))
        for prefix in sites:
            pidx = parent_channel_index(parent, prefix)
            sel_s = recover_selection(src, pidx, prefix, args.source.name)
            row = {
                "site": prefix,
                "keep": len(sel_s),
                "parent_channels": len(pidx),
                "siblings": {},
            }
            selections[prefix] = {args.source.name: sel_s}
            for label, oth in zip(labels, others):
                sel_o = recover_selection(oth, pidx, prefix, label)
                selections[prefix][label] = sel_o
                shared = set(sel_s) & set(sel_o)
                same_position = sum(1 for a, b in zip(sel_s, sel_o) if a == b)
                row["siblings"][label] = {
                    "shared_channels": len(shared),
                    "source_only": len(sel_s) - len(shared),
                    "same_channel_at_same_row": same_position,
                }
            census.append(row)
        del parent

    # The census is the artifact that says whether the naive arm is even
    # answering the question, so it is written whether or not a child follows.
    summary = {
        "schema": "neural-forge-matrix-wave41-body-blend-alignment-census-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(args.source),
        "others": [str(p) for p in args.other],
        "weights": {"source": source_w, **{l: w for l, w in zip(labels, args.weight)}},
        "align": args.align,
        "sites": census,
    }
    if census:
        for label in labels:
            shared = [int(r["siblings"][label]["shared_channels"]) for r in census]
            pos = [int(r["siblings"][label]["same_channel_at_same_row"]) for r in census]
            summary.setdefault("totals", {})[label] = {
                "shared_channels_min": min(shared),
                "shared_channels_max": max(shared),
                "shared_channels_total": sum(shared),
                "same_channel_at_same_row_min": min(pos),
                "same_channel_at_same_row_max": max(pos),
                "same_channel_at_same_row_total": sum(pos),
                "rows_total": sum(int(r["keep"]) for r in census),
            }
    if args.out_census is not None:
        args.out_census.write_text(json.dumps(summary, indent=1))
    if args.census:
        print(json.dumps(summary.get("totals", {}), indent=1))
        for r in census:
            per = "  ".join(
                f"{l}: {r['siblings'][l]['shared_channels']} shared, "
                f"{r['siblings'][l]['same_channel_at_same_row']} in place"
                for l in labels
            )
            print(f"  {r['site']:<20s} keep {r['keep']}   {per}")
        return 0

    # ------------------------------------------------------------------ build
    out: dict[str, torch.Tensor] = {}
    ff_owned = {prefix + name for prefix in sites for name in (W1, W3, W2)}
    moved = 0
    w2_columns_blended = 0
    w2_columns_source_only = 0
    for key in sorted(src):
        a = src[key].to(torch.float64)
        if args.align == "channel" and key in ff_owned:
            if key.endswith(W1) or key.endswith(W3):
                # Verbatim parent rows under the SOURCE's order.  Mixing them
                # would mix channel identities; they are kept, and the receipt
                # says so rather than leaving a reader to infer it.
                out[key] = src[key].clone()
                continue
            prefix = key[: -len(W2)]
            sel_s = selections[prefix][args.source.name]
            width = len(sel_s)
            # `mass[i]` is the convex weight this column's blend actually has to
            # divide by: the source's, plus every sibling that KEPT the same
            # parent channel.  Renormalising by it is the whole reason a channel
            # a sibling dropped costs nothing -- a column multiplied by 1-a is
            # not a convex combination, it is a shrunken one.
            mass = torch.full((width,), source_w, dtype=torch.float64)
            pairs = []
            for label, oth, w in zip(labels, others, args.weight):
                position = {c: j for j, c in enumerate(selections[prefix][label])}
                here = [(i, position[c]) for i, c in enumerate(sel_s) if c in position]
                idx_s = torch.tensor([i for i, _ in here], dtype=torch.long)
                idx_o = torch.tensor([j for _, j in here], dtype=torch.long)
                mass[idx_s] += w
                pairs.append((oth, w, idx_s, idx_o))
            blended = a * (source_w / mass)[None, :]
            for oth, w, idx_s, idx_o in pairs:
                other_w2 = oth[prefix + W2].to(torch.float64)
                blended[:, idx_s] += other_w2[:, idx_o] * (w / mass[idx_s])[None, :]
            untouched = int((mass == source_w).sum())
            w2_columns_source_only += untouched
            w2_columns_blended += width - untouched
            out[key] = blended.to(src[key].dtype).clone()
        else:
            acc = a * source_w
            for oth, w in zip(others, args.weight):
                acc = acc + oth[key].to(torch.float64) * w
            out[key] = acc.to(src[key].dtype).clone()
        moved += int((out[key] != src[key]).sum())

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
    save_file(out, str(outdir / "model.safetensors"), metadata={"format": "pt"})

    live = int(sum(int(t.numel()) for t in out.values()))
    # `mx*-build-receipt.json`, not `mx41-`: a blend's sources are SEALED
    # children from earlier waves and their receipts are named for the wave that
    # built them.  Wave 29 lost two reviews to a glob that named its own wave.
    source_receipts = sorted(args.source.glob("mx*-build-receipt.json"))
    if not source_receipts:
        raise SystemExit(
            f"REFUSED: {args.source} carries no mx*-build-receipt.json, so the "
            "blend cannot carry its source's axes and would seal with incomplete "
            "provenance."
        )
    source_build = json.loads(source_receipts[-1].read_text(encoding="utf-8"))
    if live != int(source_build["child_parameters"]):
        raise SystemExit(
            f"the blend changed the parameter count: {live} against the source's "
            f"{source_build['child_parameters']}.  It overwrites values."
        )

    receipt = {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "mx41_blend_body.py",
        "tag": str(args.tag),
        "arm": (
            "BODY-BLEND convex combination of sealed sibling bodies at identical "
            f"storage, alignment={args.align}"
        ),
        "blend": {
            "align": args.align,
            "weights": summary["weights"],
            "w2_columns_blended": w2_columns_blended,
            "w2_columns_source_only": w2_columns_source_only,
            "w1_w3_kept_from_source": args.align == "channel",
            "alignment_census": summary.get("totals", {}),
        },
        "source_model": str(args.source),
        "other_models": [str(p) for p in args.other],
        "source_tensors_sha256": hashlib.sha256(
            (args.source / "model.safetensors").read_bytes()).hexdigest(),
        "other_tensors_sha256": [
            hashlib.sha256((p / "model.safetensors").read_bytes()).hexdigest()
            for p in args.other
        ],
        "coordinates_changed_from_source": moved,
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
            "A convex combination of sealed sibling BODIES.  No capture, no "
            "solve, no gradient, no added parameter: one CPU pass, at "
            "byte-identical storage to every endpoint.  Under align=channel the "
            "w1/w3 rows are the source's verbatim parent rows and only the "
            "solved w2 columns are mixed, on the recovered channel intersection."
        ),
    }
    (outdir / "mx41-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k != "blend"}, indent=1))
    print(
        f"blend {args.tag}: align={args.align} source_weight={source_w:.4f} "
        f"{moved} coordinates moved, {w2_columns_blended} w2 columns blended, "
        f"{w2_columns_source_only} source-only, {live} parameters, "
        f"{live / PARENT_PARAMS:.6f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
