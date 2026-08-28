#!/usr/bin/env python3
"""The other half of the vocabulary question: can the child still SAY it?

``mx34_retokcensus.py`` answers the input-side question and answers it flatly:
every rung of the sealed vocabulary ladder, down to ``ARCH.V8`` at 8,190 kept
rows, segments the evidence corpus **identically to the parent**.  Not one
position of 4,181 differs at any K.  So the plan's #2 -- re-solve a body against
the child's own retokenized stream -- has nothing to re-solve against, at every K
the campaign has ever built, and the input distribution is not what breaks a
vocabulary child.

The output side is a different table and nobody has ever looked at it.  A token
that is not in the keep-set **cannot be emitted at all**: the fold deletes its
row from a tied table, so it is absent from the unembedding as well as the
embedding, and every probability mass that would have gone to it goes somewhere
else.  The campaign's sharpest phase transition is on this axis -- 9/15 at
K = 9,216 and 4/15 at K = 8,192, five cases over 1,024 rows -- and field
synthesis #26 predicts exactly this failure mode, on rare tokens, invisible to
aggregate metrics.

So this script asks the emission-side question against the only text that
matters for it: **the parent's own sealed answers to the review's own prompts.**
For each rung it reports how many of the tokens the parent actually emitted
survive the fold, how many distinct types are lost, and which lanes lose the
most.  If the review cliff and the emission-coverage cliff land on the same
rung, the campaign's sharpest transition stops being a location and becomes a
mechanism -- and a mechanism computable from a tokenizer and a dossier, with no
GPU and no build.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
M = Path("/mnt/ai_data/neural-forge-matrix")
SCHEMA = "neural-forge-matrix-wave34-emission-coverage-v1"

#: ``(cell, path under /mnt/ai_data/neural-forge-matrix, sealed review or None)``.
#:
#: Wave 31 adds the two rungs ABOVE the sealed ladder's top and marks their
#: review ``None``, because they are measured here BEFORE they are graded: wave
#: 30 located the review cliff at the 0.81 token-coverage crossing and the only
#: honest way to use that as a predictor is to read the coverage of a rung whose
#: review does not exist yet.  A ``None`` prints as ``-`` and is written to the
#: artifact as null; it is never silently filled in from a neighbour.
LADDER: tuple[tuple[str, str, int | None], ...] = (
    ("V26", "wave34/children/MX34-V26CL1792", None),
    ("V24", "wave34/children/MX34-V24CL1792", None),
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


def kept_parent_ids(child_dir: Path) -> set[int]:
    """The parent ids this child's table still carries, resolved by token STRING."""

    parent = json.loads((PARENT / "tokenizer.json").read_text(encoding="utf-8"))
    child = json.loads((child_dir / "tokenizer.json").read_text(encoding="utf-8"))
    p_vocab = parent["model"]["vocab"]
    out: set[int] = set()
    for token in child["model"]["vocab"]:
        if token not in p_vocab:
            raise SystemExit(
                f"{child_dir.name}: kept token {token!r} is not in the parent's table"
            )
        out.add(int(p_vocab[token]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dossier",
        type=Path,
        required=True,
        help="any sealed frozen-review dossier; only its `parent_responses` are read",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    parent_tok = AutoTokenizer.from_pretrained(str(PARENT))
    dossier = json.loads(args.dossier.read_text(encoding="utf-8"))
    lanes = [
        {"id": r["id"], "text": r["response_verbatim"]}
        for r in dossier["parent_responses"]
    ]
    for lane in lanes:
        lane["ids"] = parent_tok(lane["text"], add_special_tokens=False).input_ids

    total_tokens = sum(len(lane["ids"]) for lane in lanes)
    all_types = {i for lane in lanes for i in lane["ids"]}

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dossier": str(args.dossier),
        "dossier_sha256": dossier.get("dossier_sha256"),
        "lanes": len(lanes),
        "parent_emitted_tokens": total_tokens,
        "parent_emitted_types": len(all_types),
        "rungs": [],
    }

    print(
        f"{'cell':>5} {'K':>6} {'tok kept':>9} {'tok lost':>8} "
        f"{'type kept':>9} {'types lost':>10} {'worst lane':>28} {'review':>6}"
    )
    for cell, rel, review in LADDER:
        directory = M / rel
        if not directory.is_dir():
            report["rungs"].append({"cell": cell, "absent": str(directory)})
            continue
        kept = kept_parent_ids(directory)
        per_lane = []
        for lane in lanes:
            lost = sum(1 for i in lane["ids"] if i not in kept)
            per_lane.append(
                {
                    "lane": lane["id"],
                    "tokens": len(lane["ids"]),
                    "tokens_lost": lost,
                    "fraction_lost": round(lost / max(len(lane["ids"]), 1), 6),
                }
            )
        tokens_lost = sum(p["tokens_lost"] for p in per_lane)
        types_lost = sorted(i for i in all_types if i not in kept)
        worst = max(per_lane, key=lambda p: p["fraction_lost"])
        row = {
            "cell": cell,
            "keep_rows": len(kept),
            "tokens_kept": total_tokens - tokens_lost,
            "tokens_lost": tokens_lost,
            "token_coverage": round((total_tokens - tokens_lost) / max(total_tokens, 1), 6),
            "types_lost": len(types_lost),
            "type_coverage": round(
                (len(all_types) - len(types_lost)) / max(len(all_types), 1), 6
            ),
            "lost_type_examples": [
                {"id": i, "piece": parent_tok.decode([i])} for i in types_lost[:12]
            ],
            "worst_lane": worst["lane"],
            "worst_lane_fraction_lost": worst["fraction_lost"],
            "per_lane": per_lane,
            "sealed_review": review,
        }
        report["rungs"].append(row)
        print(
            f"{cell:>5} {row['keep_rows']:>6} {row['token_coverage']:>9.5f} "
            f"{tokens_lost:>8} {row['type_coverage']:>9.5f} {len(types_lost):>10} "
            f"{worst['lane'][:28]:>28} {('-' if review is None else str(review)):>6}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
