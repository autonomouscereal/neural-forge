#!/usr/bin/env python3
"""What the vocabulary axis's hard floor actually IS, and which way a corpus moves it.

Wave 31's `mx33_vocab.py` refused K = 6,144, 5,120 and 4,096 with

    the kept set loses sealed-corpus occurrences on one of the two streams;
    the closure is wrong or K is below the observed vocabulary.

and `NOTES-MX-AG` §6 recorded a floor "between 6,144 and 8,192 rows on this
corpus -- a property of the evidence, not of the model".  Wave 32 was sent to
move that floor by capturing a LARGER census corpus.  **This file exists because
that instruction has a sign error in it, and the sign is measurable in seconds.**

The gate demands ZERO lost occurrences over the census corpus.  The kept set that
satisfies it is exactly

    mandatory (509 added/special + 256 ByteLevel + their merge closure)
    UNION the transitive merge closure of every token the corpus OBSERVES

so the floor is the size of that set, and it is **monotone increasing in the
corpus**: a broader corpus observes more distinct tokens, whose closures are
larger, and the floor goes UP.  Shrinking the floor requires a NARROWER corpus,
which is the opposite of what a broader census is for.

So this probe reports three things and refuses to editorialise about any of them:

* ``--floor`` -- the exact floor on the sealed corpus, replacing "between 6,144
  and 8,192" with a number, and the decomposition that produces it.
* ``--extend`` -- the same number after the census is widened by a second
  corpus, so the direction is measured rather than argued.
* ``--decompose K`` -- at a K below the floor, exactly what is lost: how many
  observed tokens fall out, how many occurrences they carry on each stream, and
  whether the merge closure over the surviving kept set is still STRUCTURALLY
  intact.  That last one is the question the refusal's own message conflates
  with the first: "the closure is wrong" and "K is below the observed
  vocabulary" are two different conditions and only the first is a defect.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_vocab_module():
    """Import `mx33_vocab` from beside this file, whatever the cwd is."""

    spec = importlib.util.spec_from_file_location("mx33_vocab", HERE / "mx33_vocab.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def closure_of(vocab, token_ids: set[int]) -> set[int]:
    """Every id needed to still PRODUCE every id in `token_ids`."""

    tokens: set[str] = set()
    for tid in token_ids:
        token = vocab.token_of_id.get(tid)
        if token is None:
            continue
        tokens.add(token)
        vocab.ancestors(token, tokens)
    out = {vocab.id_of_token[t] for t in tokens if t in vocab.id_of_token}
    out |= {tid for tid in token_ids if tid in vocab.added_ids}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--extend", type=Path, action="append", default=[],
                    help="a second corpus of raw text lines to widen the census with")
    ap.add_argument("--decompose", type=int, action="append", default=[],
                    help="report what a K BELOW the floor actually loses")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    mv = _load_vocab_module()
    from transformers import AutoTokenizer

    vocab = mv.Vocabulary(mv.PARENT / "tokenizer.json")
    parent_tok = AutoTokenizer.from_pretrained(str(mv.PARENT))
    corpus, corpus_meta = mv.sealed_corpus_ids()
    texts = [parent_tok.decode(ids, skip_special_tokens=False) for ids in corpus]
    donor = mv.census(corpus)
    canonical = mv.canonical_census(texts, parent_tok)
    counts = dict(donor)
    for tid, n in canonical.items():
        counts[tid] = counts.get(tid, 0) + n

    mand = mv.mandatory_ids(vocab)
    observed = set(counts)
    floor_set = mand | closure_of(vocab, observed)
    report: dict[str, object] = {
        "schema": "neural-forge-matrix-wave33-vocabulary-floor-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sealed_corpus": corpus_meta,
        "mandatory_rows": len(mand),
        "observed_distinct_tokens": len(observed),
        "observed_donor_only": len(set(donor) - set(canonical)),
        "observed_canonical_only": len(set(canonical) - set(donor)),
        "closure_added_over_observed": len(closure_of(vocab, observed) - observed),
        "floor_rows": len(floor_set),
        "floor_statement": (
            "The smallest K at which `mx33_vocab.py`'s zero-loss gate can pass on "
            "the sealed corpus.  Wave 31 bracketed it between 6,144 and 8,192 by "
            "building; this is the number itself, and it is a property of the "
            "census corpus alone."
        ),
        "monotonicity": (
            "The floor is monotone INCREASING in the census corpus: more observed "
            "tokens means larger closures means a larger mandatory kept set.  A "
            "broader census corpus raises this number and cannot lower it."
        ),
        "merge_closure_at_floor": mv.merge_closure_intact(vocab, floor_set),
    }

    for extra in args.extend:
        lines = [
            line for line in extra.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        widened = dict(counts)
        for tid, n in mv.canonical_census(lines, parent_tok).items():
            widened[tid] = widened.get(tid, 0) + n
        wobserved = set(widened)
        wfloor = mand | closure_of(vocab, wobserved)
        report.setdefault("widened", []).append({
            "corpus": str(extra),
            "lines": len(lines),
            "observed_distinct_tokens": len(wobserved),
            "new_distinct_tokens_over_sealed": len(wobserved - observed),
            "floor_rows": len(wfloor),
            "floor_moved_by": len(wfloor) - len(floor_set),
        })

    for k in args.decompose:
        order, chosen = mv.choose_kept(vocab, counts, int(k))
        kept = set(order)
        lost = observed - kept
        report.setdefault("below_floor", []).append({
            "k": k,
            "kept_rows": len(kept),
            "observed_tokens_lost": len(lost),
            "donor_occurrences_lost": sum(donor.get(t, 0) for t in lost),
            "canonical_occurrences_lost": sum(canonical.get(t, 0) for t in lost),
            "donor_occurrence_coverage": (
                1.0 - sum(donor.get(t, 0) for t in lost) / max(1, sum(donor.values()))
            ),
            "canonical_occurrence_coverage": (
                1.0 - sum(canonical.get(t, 0) for t in lost) / max(1, sum(canonical.values()))
            ),
            "chooser_report": chosen,
            "merge_closure": mv.merge_closure_intact(vocab, kept),
        })

    print(json.dumps(report, indent=1))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
