#!/usr/bin/env python3
"""Wave 25: the cross term, measured -- ORTHO-PAIR's rho census (S3, pre-read P1).

Twenty-three waves have described composition damage with one sentence of
arithmetic -- "13/15 and 9/15 compose to 5/15, which is additive interference
exactly as L3 predicts" -- and not one of them has measured the quantity that
sentence is about.  L3's identity is

    ||e_A + e_B||^2_M = ||e_A||^2_M + ||e_B||^2_M + 2 <e_A, e_B>_M

and the cross term has never been computed, because "e_A" was never given a
representation.  Scores are not a vector space; you cannot inner-product 13/15
with 9/15.

### The representation

The one surface every child in this campaign shares with the parent, whatever it
did to the residual width, the block count or the token table, is **the next-token
distribution at a text prefix**.  So:

* take the 24 captured batches, decoded to text, and cut each at ``--cuts``
  character offsets -- a prefix is a STRING, so it is the same object for a model
  with a different tokenizer, which is exactly the property the vocabulary axis
  destroys for every other comparison;
* hand each prefix to each model **through that model's own tokenizer**, and read
  the last position's logits;
* restrict every model's logits to the parent ids all models in the triple can
  emit (for a vocabulary child, its own kept rows, carried back through the
  injection `mx27_retok.injection` recomputes and asserts), and log-softmax over
  exactly that restricted set;
* the error field of child X is ``e_X = logp_X - logp_parent``, one vector per
  prefix.

That is a real vector in a real inner-product space, it is defined identically
for a width child, an excision, a head merge and a vocabulary coreset, and it is
the first object in this campaign that L3's identity can actually be evaluated on.

### What is reported, and what would falsify what

For each sealed triple (A, B, AB):

* ``rho`` -- the cosine of the two solo error fields.  S3 pre-registered rho > 0
  for the compositions that lose more than their components and rho ~ 0 for the
  living pair;
* the **additivity residual** ``||e_AB - (e_A + e_B)|| / ||e_AB||`` and the R^2 of
  the additive prediction.  This is the sharper measurement and S3 did not ask
  for it: rho only prices the cross term ASSUMING superposition holds.  If the
  residual is large, damage composes NONLINEARLY and the arithmetic every wave
  note has been writing is wrong in a way no rho can repair -- which is itself
  the first sealed mechanism evidence on L3, for one census's price;
* both metrics, always: the identity metric on the restricted log-probabilities
  and the parent's own next-token probability weighting.  A census that is a
  function of one metric choice is a metric choice, not a census.

Nothing here builds, installs or gates anything (L1).

Usage::

    mx27_rho.py --triple 'V16X13:MX16-X13A,MX25-V8,MX25-V8X13' ... --out P1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import mx27_retok

PARENT = mx27_retok.PARENT
SCHEMA = "neural-forge-matrix-wave27-cross-error-census-v1"


def log(message: str) -> None:
    print(f"[mx27rho] {message}", flush=True)


def prefixes(cuts: int) -> list[dict[str, Any]]:
    """The comparison surface: ``cuts`` character prefixes of each corpus row.

    Offsets are a fixed fraction of each row's length, so the surface is a
    function of the corpus and the one integer -- no model, no tokenizer and no
    review prompt is consulted in choosing it.
    """

    from transformers import AutoTokenizer

    parent_tok = AutoTokenizer.from_pretrained(str(PARENT))
    out: list[dict[str, Any]] = []
    for index, ids in mx27_retok.parent_batches():
        text = parent_tok.decode(ids, skip_special_tokens=False)
        for k in range(1, cuts + 1):
            offset = max(1, round(k * len(text) / (cuts + 1)))
            out.append({"batch_index": index, "cut": k, "text": text[:offset]})
    return out


class Reader:
    """One model, its tokenizer, and its map into parent id space."""

    def __init__(self, name: str, directory: Path, device: str) -> None:
        from transformers import AutoTokenizer, Lfm2ForCausalLM

        self.name = name
        self.directory = directory
        config = json.loads((directory / "config.json").read_text())
        self.vocab = int(config["vocab_size"])
        parent_vocab = int(json.loads((PARENT / "config.json").read_text())["vocab_size"])
        if self.vocab == parent_vocab:
            self.to_parent = list(range(self.vocab))
            self.tokenizer = AutoTokenizer.from_pretrained(str(PARENT))
        else:
            self.to_parent = mx27_retok.injection(directory)
            self.tokenizer = AutoTokenizer.from_pretrained(str(directory))
        self.model = (
            Lfm2ForCausalLM.from_pretrained(str(directory), dtype=torch.float32)
            .to(device)
            .eval()
        )
        self.device = device

    @torch.no_grad()
    def logits(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        """Last-position logits per prefix, in PARENT id space (fp64, CPU)."""

        out = torch.full(
            (len(rows), 65536), float("-inf"), dtype=torch.float64
        )
        index = torch.tensor(self.to_parent, dtype=torch.long)
        for position, row in enumerate(rows):
            ids = self.tokenizer(row["text"], add_special_tokens=False).input_ids
            if not ids:
                raise SystemExit(f"prefix {position} tokenizes to nothing")
            tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(input_ids=tensor, use_cache=False).logits
            out[position].index_copy_(
                0, index, logits[0, -1].to("cpu", torch.float64)
            )
        return out

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def restricted_logprobs(
    raw: torch.Tensor, keep: torch.Tensor
) -> torch.Tensor:
    """Log-softmax over exactly the shared rows -- never over the full table.

    A child that cannot emit a token must not be charged for the parent's mass on
    it, and the parent must not be credited for mass the comparison excludes; the
    only comparison that is fair in both directions is the one both models are
    renormalised onto.
    """

    sub = raw.index_select(1, keep)
    return sub - torch.logsumexp(sub, dim=1, keepdim=True)


def census(
    label: str,
    parent_lp: torch.Tensor,
    fields: dict[str, torch.Tensor],
    weights: torch.Tensor,
) -> dict[str, Any]:
    """L3's identity, evaluated, in both metrics."""

    e_a = fields["A"] - parent_lp
    e_b = fields["B"] - parent_lp
    e_ab = fields["AB"] - parent_lp

    def report(w: torch.Tensor | None) -> dict[str, Any]:
        def ip(x: torch.Tensor, y: torch.Tensor) -> float:
            return float((x * y * (1.0 if w is None else w)).sum())

        na, nb, nab = ip(e_a, e_a) ** 0.5, ip(e_b, e_b) ** 0.5, ip(e_ab, e_ab) ** 0.5
        cross = ip(e_a, e_b)
        predicted = e_a + e_b
        residual = e_ab - predicted
        denom = ip(e_ab, e_ab)
        return {
            "norm_A": na,
            "norm_B": nb,
            "norm_AB_observed": nab,
            "norm_AB_predicted_additive": ip(predicted, predicted) ** 0.5,
            "cross_term": cross,
            "rho": cross / (na * nb) if na > 0 and nb > 0 else float("nan"),
            "additive_prediction_relative_residual": (
                ip(residual, residual) ** 0.5 / nab if nab > 0 else float("nan")
            ),
            "additive_prediction_r2": (
                1.0 - ip(residual, residual) / denom if denom > 0 else float("nan")
            ),
            "cosine_observed_vs_additive": (
                ip(e_ab, predicted)
                / (nab * ip(predicted, predicted) ** 0.5)
                if nab > 0 and ip(predicted, predicted) > 0
                else float("nan")
            ),
        }

    return {
        "pair": label,
        "prefixes": int(e_a.shape[0]),
        "shared_rows": int(e_a.shape[1]),
        "identity_metric": report(None),
        "parent_probability_weighted_metric": report(weights),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--triple",
        action="append",
        required=True,
        help="LABEL:<A dir>,<B dir>,<AB dir> -- absolute paths or names under --root",
    )
    ap.add_argument("--root", type=Path, default=Path("/mnt/ai_data/neural-forge-matrix"))
    ap.add_argument("--cuts", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = prefixes(int(args.cuts))
    log(f"comparison surface: {len(rows)} text prefixes from 24 corpus rows")

    triples: list[tuple[str, list[Path]]] = []
    for spec in args.triple:
        label, _, paths = spec.partition(":")
        dirs = [Path(p) for p in paths.split(",")]
        if len(dirs) != 3:
            raise SystemExit(f"--triple {spec!r} does not name exactly three models")
        for d in dirs:
            if not (d / "config.json").is_file():
                raise SystemExit(f"{d} is not a model directory")
        triples.append((label, dirs))

    parent_reader = Reader("PARENT", PARENT, args.device)
    parent_raw = parent_reader.logits(rows)
    parent_reader.close()
    log("parent read")

    cache: dict[str, torch.Tensor] = {}
    results: list[dict[str, Any]] = []
    for label, dirs in triples:
        keep_sets: list[set[int]] = []
        for d in dirs:
            reader_key = str(d)
            if reader_key not in cache:
                reader = Reader(reader_key, d, args.device)
                cache[reader_key] = reader.logits(rows)
                keep_sets.append(set(reader.to_parent))
                reader.close()
                log(f"  read {d.name}")
            else:
                config = json.loads((d / "config.json").read_text())
                if int(config["vocab_size"]) == 65536:
                    keep_sets.append(set(range(65536)))
                else:
                    keep_sets.append(set(mx27_retok.injection(d)))
        shared = sorted(set.intersection(*keep_sets))
        keep = torch.tensor(shared, dtype=torch.long)
        parent_lp = restricted_logprobs(parent_raw, keep)
        weights = parent_lp.exp()
        fields = {
            role: restricted_logprobs(cache[str(d)], keep)
            for role, d in zip(("A", "B", "AB"), dirs)
        }
        row = census(label, parent_lp, fields, weights)
        row["models"] = {
            role: str(d) for role, d in zip(("A", "B", "AB"), dirs)
        }
        results.append(row)
        m = row["identity_metric"]
        log(
            f"{label}: shared rows {len(shared)}  rho {m['rho']:+.4f}  "
            f"||e_A|| {m['norm_A']:.1f} ||e_B|| {m['norm_B']:.1f} "
            f"||e_AB|| {m['norm_AB_observed']:.1f} (additive prediction "
            f"{m['norm_AB_predicted_additive']:.1f}, relative residual "
            f"{m['additive_prediction_relative_residual']:.4f}, R2 "
            f"{m['additive_prediction_r2']:+.4f})"
        )

    body = {
        "schema": SCHEMA,
        "pre_read": "P1",
        "purpose": (
            "L3's cross term, evaluated on the one surface every child shares "
            "with the parent whatever it folded: the next-token distribution at "
            "a text prefix.  Orders; gates nothing (L1)."
        ),
        "surface": {
            "prefixes": len(rows),
            "cuts_per_corpus_row": int(args.cuts),
            "rule": "character offsets k*len(text)/(cuts+1); a prefix is a STRING, "
            "so it is the same object for models with different tokenizers",
            "corpus": str(mx27_retok.TRACE.name),
        },
        "metrics": {
            "identity": "unweighted inner product on restricted log-probabilities",
            "parent_probability_weighted": "the same, weighted by the parent's own "
            "restricted next-token probability",
        },
        "results": results,
    }
    print(json.dumps(body, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(body, indent=1))
        log(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
