#!/usr/bin/env python3
"""Where does a vocabulary keep-set stop covering the corpus's own segmentation?

This file exists because RETOK-BODY died on its first measurement and handed
back a better question.

`INTELLIGENCE_RETENTION_PLAN_2026-08-14.md` #2 costs one new capture to solve the
CL896 body against ``ARCH.V19``'s retokenized stream, on the theory that the
composed child reads a token sequence its body was never solved for -- "sequence
shift is untested".  Wave 30 built the stream and tested it: the V19 tokenizer's
segmentation of all twenty-four evidence batches is **identical to the parent's
own, at every one of 4,181 positions**, differing only in what the ids are
called.  A capture through the V19 interface is therefore the parent's capture
with the rows renamed, the re-solve is the sealed solve, and there is nothing at
K = 19,456 for the plan's #2 to find.

That is not a null about vocabulary folds.  It is a measurement of ONE K, and it
converts the plan's question into a sharper one that the whole ladder can answer
for the price of a tokenizer pass per rung:

    at which K does the keep-set stop being segmentation-complete on this corpus?

The vocabulary axis is this campaign's champion and it carries this campaign's
sharpest phase transition: 9/15 at K = 9,216 and 4/15 at K = 8,192, five cases
across 1,024 rows and 0.0046 ratio points.  Field synthesis #26 (Gemma 3 270M's
rare-token-capacity rationale) predicts vocabulary folds fail first on rare-token
segmentation, invisible to aggregate metrics.  If the segmentation break and the
review cliff land at the same rung, the campaign's sharpest transition has a
mechanism instead of a location -- and every future vocabulary cell can be priced
before it is built, by a tokenizer pass that needs no GPU at all.

The census reports, per rung: the corpus re-segmented by that child's tokenizer,
how many positions it takes against the parent's own segmentation of the same
text, how many batches segment identically, and how many positions differ once
the child's ids are injected back into the parent's table.  Segmentation is
compared IN ONE ID SPACE -- child ids mapped through the kept-row order -- because
comparing them in two id spaces compares the renaming and reports it as a shift,
which is the error that made the first reading of this look like a 34% sequence
change.

**The 0.66x round trip is real and is not a vocabulary effect.**  Decoding the
sealed 6,335-position trace and re-encoding it with the PARENT's own tokenizer
gives 4,181 positions.  The sealed ids are not the canonical tokenization of
their own text, so every arm here -- including the parent control -- runs on the
re-encode.  It is reported rather than hidden because it bounds what a
retokenized capture could ever have measured.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
TRACE = Path(
    "/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810"
    "/evidence/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff"
)
M = Path("/mnt/ai_data/neural-forge-matrix")
SCHEMA = "neural-forge-matrix-wave32-retokenization-census-v1"

#: The sealed vocabulary ladder: cell -> (retained child directory, sealed review).
#: Every rung whose child survived retention; the ones that did not are named in
#: the report as absent rather than silently skipped.
LADDER: tuple[tuple[str, str, int], ...] = (
    ("V32", "wave24/children/MX24-V32", 11),
    ("V22", "wave26/children/MX26-V22", 13),
    ("V20", "wave25/children/MX25-V20", 13),
    ("V19", "wave27/children/MX27-V19", 14),
    ("V18", "wave26/children/MX26-V18", 13),
    ("V17", "wave27/children/MX27-V17", 11),
    ("V14", "wave25/children/MX25-V14", 9),
    ("V12", "wave25/children/MX25-V12", 11),
    ("V10", "wave25/children/MX25-V10", 10),
    ("V9", "wave26/children/MX26-V9", 9),
    ("V8", "wave24/children/MX24-V8", 4),
)


def injection(child_dir: Path) -> list[int]:
    """The child's kept rows as parent ids, resolved by token STRING, never by id.

    ``mx32_retok.injection`` in one function, duplicated for the same reason that
    file duplicates ``split_by_batch``: this script is run against deployments
    whose sibling may not be importable, and a census that silently imports a
    different wave's mapping is a census of that wave.
    """

    parent = json.loads((PARENT / "tokenizer.json").read_text(encoding="utf-8"))
    child = json.loads((child_dir / "tokenizer.json").read_text(encoding="utf-8"))
    p_vocab = parent["model"]["vocab"]
    c_vocab = child["model"]["vocab"]
    order = [0] * len(c_vocab)
    for token, cid in c_vocab.items():
        if token not in p_vocab:
            raise SystemExit(
                f"{child_dir.name}: kept token {token!r} is not in the parent's "
                f"table; this child is not a row subset of its parent"
            )
        order[int(cid)] = int(p_vocab[token])
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    parent_tok = AutoTokenizer.from_pretrained(str(PARENT))

    texts: list[str] = []
    original_positions = 0
    for directory in sorted((TRACE / "batches").iterdir()):
        with safe_open(str(directory / "trace.safetensors"), "pt") as handle:
            prompt = handle.get_tensor("input.input_ids")
            forced = handle.get_tensor("input.forced_decode_ids")
        ids = torch.cat([prompt, forced], dim=1)[0]
        original_positions += int(ids.shape[0])
        texts.append(parent_tok.decode(ids, skip_special_tokens=False))

    base = [parent_tok(t, add_special_tokens=False).input_ids for t in texts]
    base_positions = sum(len(x) for x in base)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace": TRACE.name,
        "batches": len(texts),
        "sealed_trace_positions": original_positions,
        "parent_reencode_positions": base_positions,
        "round_trip_ratio": round(base_positions / original_positions, 6),
        "round_trip_note": (
            "the sealed ids are not the canonical tokenization of their own text: "
            "decoding and re-encoding with the PARENT's tokenizer loses 34% of the "
            "positions.  Every rung below, and the parent control, run on the "
            "re-encode, so this is a property of the corpus and not of any fold"
        ),
        "rungs": [],
    }

    print(
        f"{'cell':>5} {'K':>6} {'positions':>9} {'vs parent':>9} "
        f"{'batches identical':>17} {'positions differing':>19} {'review':>6}"
    )
    for cell, rel, review in LADDER:
        directory = M / rel
        if not directory.is_dir():
            report["rungs"].append({"cell": cell, "absent": str(directory)})
            print(f"{cell:>5} ABSENT {directory}")
            continue
        child_tok = AutoTokenizer.from_pretrained(str(directory))
        order = injection(directory)
        segmented = [child_tok(t, add_special_tokens=False).input_ids for t in texts]
        # One id space.  The child's ids mean nothing against the parent's until
        # they are put back through the kept-row order.
        lifted = [[order[i] for i in seq] for seq in segmented]
        positions = sum(len(x) for x in segmented)
        identical_batches = sum(1 for a, b in zip(lifted, base) if a == b)
        differing = sum(
            sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
            for a, b in zip(lifted, base)
        )
        row = {
            "cell": cell,
            "keep_rows": int(child_tok.vocab_size),
            "child": str(directory),
            "positions": positions,
            "positions_vs_parent": round(positions / base_positions, 6),
            "batches_segmenting_identically": identical_batches,
            "batches": len(texts),
            "positions_differing": differing,
            "positions_differing_fraction": round(differing / max(base_positions, 1), 6),
            "segmentation_complete": differing == 0,
            "sealed_review": review,
        }
        report["rungs"].append(row)
        print(
            f"{cell:>5} {row['keep_rows']:>6} {positions:>9} "
            f"{row['positions_vs_parent']:>9.4f} {identical_batches:>17} "
            f"{differing:>19} {review:>6}",
            flush=True,
        )

    priced = [r for r in report["rungs"] if r.get("segmentation_complete") is not None]
    complete = [r for r in priced if r["segmentation_complete"]]
    broken = [r for r in priced if not r["segmentation_complete"]]
    report["verdict"] = {
        "segmentation_complete_rungs": [r["cell"] for r in complete],
        "segmentation_broken_rungs": [r["cell"] for r in broken],
        "deepest_complete_keep_rows": (
            min((r["keep_rows"] for r in complete), default=None)
        ),
        "shallowest_broken_keep_rows": (
            max((r["keep_rows"] for r in broken), default=None)
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print()
    print(json.dumps(report["verdict"], indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
