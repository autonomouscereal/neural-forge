#!/usr/bin/env python3
"""Wave 41: the vocabulary axis has never varied WHICH rows, only HOW MANY.

`mx43_vocab.py`'s `choose_kept` ranks candidates by ``(-census_count,
parent_token_id)`` and its own docstring says what that means: "with a
67,584-position census and a 16,384-row budget, the tie-break decides most of
the table."  Every sealed receipt agrees -- ``observed_tokens_admitted`` is
4,633 at every K ever built, so at the campaign's floor rungs roughly 4,600 of
the kept rows are chosen by the census and the remaining thousands are chosen by
**parent id order**, which is the BPE merge order and therefore a proxy for the
training corpus's frequency.

Sixteen waves have moved K.  Not one has moved the tie-break.  The two rungs
that bracket the floor -- ``ARCH.V10CL896X09`` alive at 1/15 and
``ARCH.V9CL896X09`` dead at 0/15 -- differ by exactly 1,024 rows and those rows
are one contiguous blind tie-break block.  This file re-orders the discretionary
fill at FIXED K, on the SAME body, holding everything else the deletion arm
holds, so that a case delta is attributable to the SELECTION and to nothing else.

### What is held fixed, and each of these is asserted rather than intended

* **The mandatory set.**  All added/special tokens, all 256 ByteLevel alphabet
  tokens and the transitive merge closure of both, taken from
  `mx43_vocab.mandatory_ids` -- the same function the gather uses, imported
  rather than reimplemented, because a set this load-bearing with two
  implementations is a contract that decays the first time one of them is
  edited.  Coverage therefore cannot move between arms and the below-floor gate
  cannot fire differently on one.
* **K.**  The arm lands on exactly the number of rows the sealed control landed
  on, or it refuses.  An arm at a different row count is a different rung.
* **The parameter count**, stated on the command line and asserted before
  anything is written.  Wave 40's tie-builder rule, carried to the arm that
  needs it for the same reason: a comparison whose two children do not have the
  same count is two readings of different things.
* **The weights.**  Every kept row is the source table's own bytes, gathered by
  index, asserted with ``torch.equal`` and not a tolerance.  ONLY the selection
  moves.
* **The tokenizer construction.**  `mx43_vocab.remapped_tokenizer` writes it,
  the same function that writes the deletion twin's, so a difference between the
  two children cannot be a difference between two tokenizer writers.
* **The merge closure.**  Structural, never releasable, exactly as in the
  gather.  A re-ranked kept set that cannot be PRODUCED is a broken tokenizer
  wearing a selection receipt.

### The arms, and why they are these three

The slate's Arms A and B rank the discretionary fill by a SECOND census over
held-out text "sealed by hash, disjoint from the frozen 15 AND the 30-prompt
extension bank".  **No corpus this estate holds passes that test**, and the
audit is in `run/mx43/README.md` §2 with each corpus's own words quoted: the
paraphrase triplets are restatements OF the frozen cases, the causal-support
pack forbids selection in its own ``purpose`` field, the extension banks are the
rider, the sealed corpus IS the evidence census that supplies the primary key,
and wikitext/lambada are the benchmark column.  Selecting a vocabulary on any of
them contaminates something the campaign publishes.

So wave 41 builds the three arms that need no corpus at all.  This is a change
of ARM, registered with its cause before anything was built, and not a waiver of
the class:

``P`` -- ``(-census, -parent_id)``.  **The sealed tie-break run backwards.**  The
one ordering the estate has never varied, inverted.  Where the sealed fill takes
the earliest-merged (most frequent in training) unobserved tokens, P takes the
latest-merged.  This is the maximal perturbation of the discretionary fill that
still respects the census, the closure and K.

``R`` -- ``(-census, blake2b(seed, parent_id))``.  A seeded permutation, seed in
the receipt, reproducible from the receipt alone.  P is one extreme and the
sealed order is the other; R is neither, so the three together bracket the space
of orderings rather than sampling two points of it.

``C`` -- ``(-census, -leverage, parent_id)``, leverage ``s_r = e_r' M e_r`` with
``M = E' E`` the tied-unembedding Gram, in fp64.  The slate's Arm C verbatim.
**It is built with the pre-registered prediction that it FAILS**: `NOTES-MX-M`
§2.3 recorded wave 9's row-norm ordering reaching 2.34% coverage at M=4,096, and
a leverage key is that key's second cousin.  Predicted AND built.  A
predict-and-skip would close the class on a prediction, which is the thing this
campaign refuses.

### The identity gate, which is the whole reason a null here is readable

``--arm SEALED`` runs this file's own ranker with the sealed key and asserts the
result is `mx43_vocab.choose_kept`'s output **element for element**.  It runs on
every build, not only on the self-test, so the ranker that produced the arm is
demonstrated to be the ranker that reproduces the control.  Without it a null is
indistinguishable from a re-ranker that quietly did nothing.

### The top-up, stated rather than hidden

The greedy fill admits a candidate only if its whole ancestor closure fits in the
remaining budget.  Under the sealed order the budget closes exactly; under a
different order it may close one or two rows short, because a different set of
candidates arrives with deep closures.  The arm then tops up, in its OWN key
order, with candidates whose closure is ALREADY entirely kept -- cost exactly one
row each -- until K is met.  That is the natural completion of the same greedy
rule and not a second rule, and ``rows_added_by_topup`` is published so a reader
can see how much of the arm's tail is the top-up rather than the key.

Usage::

    mx43_vocabrowset.py --mode divergence --k 9216 --source <dir> --arm P --arm R --arm C
    mx43_vocabrowset.py --mode build --arm P --k 9216 --tag RP9CL896X13 \\
        --expect-parameters 93504256 --source <dir> --child <out dir> --below-floor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import mx43_vocab as V

SCHEMA = "neural-forge-matrix-wave43-rowset-vocabulary-child-v1"
ARMS = ("SEALED", "P", "R", "C")
#: Published in the receipt.  The permutation arm is reproducible from the
#: receipt alone, which is what makes it a control rather than a coin flip.
RANDOM_SEED = 41


def log(message: str) -> None:
    print(f"[mx43rowset] {message}", flush=True)


# ------------------------------------------------------------------ the keys
def leverage_scores(source: Path) -> dict[int, float]:
    """``s_r = e_r' M e_r`` with ``M = E' E``, the tied-unembedding Gram, fp64.

    This is the slate's Arm C key.  ``lm_head`` is tied to
    ``model.embed_tokens.weight`` on this parent, so the Gram of the table IS
    the reader's own metric and no second tensor is involved.
    """

    import torch
    from safetensors.torch import load_file

    table = load_file(str(source / "model.safetensors"))["model.embed_tokens.weight"]
    e = table.to(torch.float64)
    gram = e.T @ e
    scores = ((e @ gram) * e).sum(dim=1)
    return {int(i): float(scores[i]) for i in range(int(e.shape[0]))}


def arm_key(
    arm: str, counts: dict[int, int], leverage: dict[int, float] | None
) -> Callable[[int], tuple]:
    if arm == "SEALED":
        return lambda tid: (-counts.get(tid, 0), tid)
    if arm == "P":
        return lambda tid: (-counts.get(tid, 0), -tid)
    if arm == "R":
        def key(tid: int) -> tuple:
            digest = hashlib.blake2b(
                f"{RANDOM_SEED}:{tid}".encode("ascii"), digest_size=8
            ).digest()
            return (-counts.get(tid, 0), int.from_bytes(digest, "big"), tid)
        return key
    if arm == "C":
        assert leverage is not None
        return lambda tid: (-counts.get(tid, 0), -leverage.get(tid, 0.0), tid)
    raise SystemExit(f"unknown arm {arm!r}; expected one of {ARMS}")


def choose_kept_ranked(
    vocab: V.Vocabulary, counts: dict[int, int], k: int, key: Callable[[int], tuple]
) -> tuple[list[int], dict[str, Any]]:
    """`mx43_vocab.choose_kept` with the discretionary key lifted out.

    The mandatory set, the closure rule, the budget and the skip-if-it-does-not-
    fit branch are the gather's.  The ONLY thing this function changes is the
    order the discretionary candidates are considered in, which is the variable
    the whole cell exists to move.
    """

    kept: set[int] = set(V.mandatory_ids(vocab))
    mandatory_size = len(kept)
    if mandatory_size > k:
        raise SystemExit(
            f"the mandatory set (specials + byte alphabet + their closure) is "
            f"{mandatory_size} rows and K is {k}."
        )
    ranking = sorted((tid for tid in vocab.token_of_id if tid not in kept), key=key)
    observed_admitted = 0
    skipped_for_budget = 0
    for tid in ranking:
        if len(kept) >= k:
            break
        if tid in kept:
            continue
        token = vocab.token_of_id[tid]
        block: set[str] = {token}
        vocab.ancestors(token, block)
        addition = {
            vocab.id_of_token[t]
            for t in block
            if t in vocab.id_of_token and vocab.id_of_token[t] not in kept
        }
        if len(kept) + len(addition) > k:
            skipped_for_budget += 1
            continue
        kept |= addition
        if counts.get(tid, 0) > 0:
            observed_admitted += 1
    # THE TOP-UP, and it is in the receipt.  A re-ordered fill can close the
    # budget one or two rows short because a different set of candidates arrives
    # carrying deep closures.  Cost-one candidates -- those whose whole closure
    # is already kept -- are admitted in the arm's OWN key order until K is met.
    topup = 0
    if len(kept) < k:
        for tid in ranking:
            if len(kept) >= k:
                break
            if tid in kept:
                continue
            token = vocab.token_of_id[tid]
            block = {token}
            vocab.ancestors(token, block)
            addition = {
                vocab.id_of_token[t]
                for t in block
                if t in vocab.id_of_token and vocab.id_of_token[t] not in kept
            }
            if len(addition) != 1:
                continue
            kept |= addition
            topup += 1
    order = sorted(kept)
    report = {
        "k": k,
        "kept_rows": len(order),
        "mandatory_rows": mandatory_size,
        "mandatory_rule": (
            "all 509 added/special tokens, all 256 ByteLevel alphabet tokens, "
            "and the transitive merge-tree closure of both"
        ),
        "observed_tokens_admitted_by_ranking": observed_admitted,
        "tokens_skipped_because_their_closure_did_not_fit": skipped_for_budget,
        "rows_added_by_topup": topup,
        "dead_parent_rows_dropped": len(vocab.dead_ids),
    }
    return order, report


def divergence(
    sealed_order: list[int], order: list[int], key: Callable[[int], tuple],
    sealed_key: Callable[[int], tuple], vocab: V.Vocabulary
) -> dict[str, Any]:
    """How far this arm's kept set and this arm's ORDER are from the sealed ones.

    A null on an arm whose kept set is the sealed one is not a null about
    selection; it is the sealed child re-read.  Both numbers are published so a
    reader can tell which kind of null they are looking at.
    """

    a, b = set(sealed_order), set(order)
    ids = [tid for tid in vocab.token_of_id]
    ra = {tid: i for i, tid in enumerate(sorted(ids, key=sealed_key))}
    rb = {tid: i for i, tid in enumerate(sorted(ids, key=key))}
    n = len(ids)
    # Spearman footrule, normalised: mean |rank difference| over the whole
    # candidate list, divided by its maximum.  Zero means the arm did nothing.
    footrule = sum(abs(ra[t] - rb[t]) for t in ids) / (n * n / 2.0)
    return {
        "kept_rows_sealed": len(a),
        "kept_rows_arm": len(b),
        "rows_only_in_sealed": len(a - b),
        "rows_only_in_arm": len(b - a),
        "symmetric_difference": len(a ^ b),
        "jaccard": len(a & b) / len(a | b) if a | b else 1.0,
        "normalised_footrule_vs_parent_id_order": footrule,
        "claim": (
            "symmetric_difference is how many rows this arm actually moved at "
            "fixed K.  If it is ZERO the arm IS the sealed child and a null "
            "says nothing about selection; the footrule is how far the arm's "
            "whole candidate ORDER is from the sealed one, which is nonzero "
            "even when the kept set is not"
        ),
    }


# ---------------------------------------------------------------- the write
def build(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    source = Path(args.source)
    outdir = Path(args.child)
    started = time.monotonic()
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    tokenizer_file = source / "tokenizer.json"
    if not tokenizer_file.exists():
        tokenizer_file = V.PARENT / "tokenizer.json"
    vocab = V.Vocabulary(tokenizer_file)
    if int(source_config["vocab_size"]) != 65536:
        raise SystemExit(
            f"--source declares vocab_size {source_config['vocab_size']}; this "
            "builder re-ranks the PARENT vocabulary once, not a folded one twice."
        )
    source_axes: list[dict[str, Any]] = []
    source_excised: list[int] = []
    source_basis: dict[str, Any] | None = None
    source_receipts = sorted(source.glob("mx*-build-receipt.json"))
    if source_receipts:
        source_build = json.loads(source_receipts[-1].read_text(encoding="utf-8"))
        source_axes = list(source_build.get("axes", []))
        source_excised = [int(x) for x in source_build.get("excised", [])]
        source_basis = source_build.get("residual_basis")
        log(
            f"carried {len(source_axes)} axes from {source_receipts[-1].name} "
            f"(excised {source_excised}, residual basis "
            f"{'yes' if source_basis else 'none'})"
        )
    elif source.resolve() != V.PARENT.resolve():
        raise SystemExit(
            f"{source} carries no mx*-build-receipt.json, so its axes cannot be "
            "carried and a composed child would seal with incomplete provenance."
        )

    parent_tok = AutoTokenizer.from_pretrained(str(V.PARENT))
    corpus, corpus_meta = V.sealed_corpus_ids()
    corpus_texts = [parent_tok.decode(ids, skip_special_tokens=False) for ids in corpus]
    donor_counts = V.census(corpus)
    canonical_counts = V.canonical_census(corpus_texts, parent_tok)
    counts = dict(donor_counts)
    for tid, n in canonical_counts.items():
        counts[tid] = counts.get(tid, 0) + n

    k = int(args.k)
    arm = str(args.arm)
    leverage = leverage_scores(source) if arm == "C" else None
    if arm == "C":
        # Named so the log says which key produced the ordering, and so a reader
        # of the log can see the leverage arm was BUILT rather than predicted.
        log(f"arm C leverage computed over {len(leverage)} rows from the tied Gram")
    key = arm_key(arm, counts, leverage)
    sealed_key = arm_key("SEALED", counts, None)

    # THE CONTROL IS THE SEALED FUNCTION ITSELF, called here, not a restatement
    # of it.  `mx43_vocab.choose_kept` is what every sealed vocabulary child in
    # the estate was built by.
    sealed_order, sealed_report = V.choose_kept(vocab, counts, k)
    identity_order, _ = choose_kept_ranked(vocab, counts, k, sealed_key)
    if identity_order != sealed_order:
        raise SystemExit(
            "IDENTITY GATE FAILED: this file's ranker with the SEALED key does "
            "not reproduce mx43_vocab.choose_kept's kept set.  Every arm below "
            "would then be measuring this file against itself instead of "
            "against the sealed control."
        )
    log(f"identity gate: the sealed key reproduces choose_kept exactly ({len(sealed_order)} rows)")

    order, report = choose_kept_ranked(vocab, counts, k, key)
    keptset = set(order)
    if len(order) != len(sealed_order):
        raise SystemExit(
            f"arm {arm} landed on {len(order)} rows and the sealed control on "
            f"{len(sealed_order)}.  The arms are compared at FIXED K; an arm at "
            "a different row count is a different rung and the comparison would "
            "not be controlled."
        )
    mandatory = set(V.mandatory_ids(vocab))
    if mandatory - keptset:
        raise SystemExit(
            f"arm {arm} dropped {len(mandatory - keptset)} MANDATORY rows.  The "
            "mandatory set is held exactly as sealed so coverage cannot move "
            "between arms; an arm that drops one publishes a selection result "
            "that is really a coverage result."
        )
    div = divergence(sealed_order, order, key, sealed_key, vocab)
    log(
        f"arm {arm}: moved {div['symmetric_difference']} of {len(order)} rows "
        f"(jaccard {div['jaccard']:.6f}, footrule {div['normalised_footrule_vs_parent_id_order']:.6f})"
    )

    cov = V.coverage(donor_counts, keptset)
    cov_canonical = V.coverage(canonical_counts, keptset)
    closure_state = V.merge_closure_intact(vocab, keptset)
    if closure_state["intact"] is not True:
        raise SystemExit(
            f"{closure_state['unproducible_kept_tokens']} kept tokens cannot be "
            "produced from the restricted merge list, e.g. ids "
            f"{closure_state['first_unproducible']}.  The closure is structural "
            "and there is no flag for it on a rowset arm any more than on a "
            "gather."
        )
    lost = int(cov["occurrences_lost"]) + int(cov_canonical["occurrences_lost"])
    if lost and not args.below_floor:
        raise SystemExit(
            f"K={k} is BELOW THE CENSUS FLOOR on arm {arm}: the kept set loses "
            f"{cov['occurrences_lost']} donor-mapped and "
            f"{cov_canonical['occurrences_lost']} canonical sealed-corpus "
            "occurrences.  Re-run with --below-floor to take it deliberately."
        )
    if lost:
        log(
            f"BELOW FLOOR, deliberately: {cov['occurrences_lost']} donor-mapped "
            f"and {cov_canonical['occurrences_lost']} canonical occurrences "
            f"dropped; coverage {cov['occurrence_coverage']:.6f} donor / "
            f"{cov_canonical['occurrence_coverage']:.6f} canonical"
        )

    outdir.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(source / "model.safetensors"))
    table = tensors["model.embed_tokens.weight"]
    if int(table.shape[0]) != 65536:
        raise SystemExit(f"source table has {table.shape[0]} rows, expected 65,536")
    index = torch.tensor(order, dtype=torch.long)
    gathered = table.index_select(0, index).clone()
    for probe in (0, len(order) // 3, len(order) // 2, len(order) - 1):
        if not torch.equal(gathered[probe], table[order[probe]]):
            raise SystemExit(f"row {probe} is not the source's bytes")
    if not torch.equal(gathered, table[index]):
        raise SystemExit("the gathered table is not a row subset of the source's")
    tensors["model.embed_tokens.weight"] = gathered

    live = int(sum(int(t.numel()) for t in tensors.values()))
    expected_parameters = int(args.expect_parameters)
    if int(live) != int(expected_parameters):
        raise SystemExit(
            f"REFUSED before writing: arm {arm} lands on {live:,} parameters and "
            f"the deletion twin it is read against has {expected_parameters:,}.  "
            "A rowset arm exists only as a CONTROLLED comparison; a count that "
            "misses the twin's is a second reading of a different thing."
        )
    save_file(tensors, str(outdir / "model.safetensors"), metadata={"format": "pt"})

    body, new_of_old = V.remapped_tokenizer(vocab, order)
    (outdir / "tokenizer.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    child_config = dict(source_config)
    child_config["vocab_size"] = len(order)
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        old = source_config.get(field)
        if old is not None:
            if int(old) not in new_of_old:
                raise SystemExit(f"config {field}={old} was not kept")
            child_config[field] = new_of_old[int(old)]
    (outdir / "config.json").write_text(json.dumps(child_config, indent=2), encoding="utf-8")

    for name in V.SIDECARS:
        src = source / name
        if not src.exists():
            src = V.PARENT / name
        if not src.exists():
            continue
        if name == "generation_config.json":
            gen = json.loads(src.read_text(encoding="utf-8"))
            for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
                if field in gen and gen[field] is not None:
                    values = gen[field]
                    if isinstance(values, list):
                        gen[field] = [new_of_old[int(v)] for v in values]
                    else:
                        gen[field] = new_of_old[int(values)]
            (outdir / name).write_text(json.dumps(gen, indent=2), encoding="utf-8")
        else:
            (outdir / name).write_bytes(src.read_bytes())

    child_tok = AutoTokenizer.from_pretrained(str(outdir))
    review_texts = V.review_pack_texts()

    def _agreement(texts: list[str]) -> dict[str, Any]:
        lossless = exact = positions = agreed = dropped = 0
        for text in texts:
            got = child_tok(text, add_special_tokens=False).input_ids
            if child_tok.decode(got, skip_special_tokens=False) == text:
                lossless += 1
            expected: list[int] = []
            for t in parent_tok(text, add_special_tokens=False).input_ids:
                image = new_of_old.get(int(t))
                if image is None:
                    dropped += 1
                    expected.append(-1)
                else:
                    expected.append(image)
            if list(got) == expected:
                exact += 1
            positions += len(expected)
            agreed += sum(
                1 for i in range(min(len(got), len(expected))) if got[i] == expected[i]
            )
        return {
            "rows": len(texts),
            "rows_lossless_on_text": lossless,
            "rows_segmented_exactly_as_parent": exact,
            "parent_positions": positions,
            "parent_positions_whose_token_was_dropped": dropped,
            "positions_agreeing": agreed,
            "position_agreement": agreed / positions if positions else 0.0,
        }

    corpus_agreement = _agreement(corpus_texts)
    review_agreement = _agreement(review_texts)
    if (corpus_agreement["rows_lossless_on_text"] != len(corpus_texts)
            or review_agreement["rows_lossless_on_text"] != len(review_texts)):
        raise SystemExit(
            "the child tokenizer is not lossless on text; the byte alphabet or "
            "the merge closure is broken."
        )
    log(
        f"lossless on text: {corpus_agreement['rows_lossless_on_text']}/"
        f"{len(corpus_texts)} corpus rows, "
        f"{review_agreement['rows_lossless_on_text']}/{len(review_texts)} review "
        f"prompts; position agreement "
        f"{corpus_agreement['position_agreement']:.4f} corpus / "
        f"{review_agreement['position_agreement']:.4f} review"
    )

    def _tpb(texts: list[str]) -> dict[str, float]:
        pb = sum(len(parent_tok(t, add_special_tokens=False).input_ids) for t in texts)
        cb = sum(len(child_tok(t, add_special_tokens=False).input_ids) for t in texts)
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        return {
            "utf8_bytes": nbytes,
            "parent_tokens": pb,
            "child_tokens": cb,
            "parent_tokens_per_byte": pb / nbytes if nbytes else 0.0,
            "child_tokens_per_byte": cb / nbytes if nbytes else 0.0,
            "sequence_length_ratio": (cb / pb) if pb else 0.0,
        }

    tokens_per_byte = {
        "sealed_corpus": _tpb(corpus_texts),
        "frozen_review_prompts": _tpb(review_texts),
    }
    review_counts = V.census(
        [parent_tok(t, add_special_tokens=False).input_ids for t in review_texts]
    )
    receipt = {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "mx43_vocabrowset.py",
        "tag": str(args.tag),
        "arm": f"vocab-row-coreset, discretionary fill RE-RANKED (arm {arm}, zero solve)",
        "rowset": {
            "arm": arm,
            "key_rule": {
                "SEALED": "(-census_count, parent_token_id) -- mx43_vocab.choose_kept's own key",
                "P": "(-census_count, -parent_token_id) -- the sealed tie-break RUN BACKWARDS",
                "R": f"(-census_count, blake2b(seed={RANDOM_SEED}, parent_token_id), parent_token_id)",
                "C": "(-census_count, -leverage, parent_token_id), leverage s_r = e_r' (E' E) e_r in fp64",
            }[arm],
            "random_seed": RANDOM_SEED if arm == "R" else None,
            "identity_gate": (
                "this file's ranker with the SEALED key reproduces "
                "mx43_vocab.choose_kept's kept set element for element; asserted "
                "on every build, not only on the self-test"
            ),
            "divergence_vs_sealed": div,
            "arms_not_built_and_why": (
                "the slate's Arms A and B rank by a SECOND census over held-out "
                "text disjoint from the frozen 15 and the extension bank.  No "
                "corpus in this estate satisfies that: the paraphrase triplets "
                "are restatements of the frozen cases, the causal-support pack "
                "forbids selection in its own purpose field, the extension banks "
                "are the rider, the sealed corpus IS the primary key's census, "
                "and wikitext/lambada are the benchmark column.  Registered as a "
                "substitution with its cause BEFORE the build, not waived"
            ),
            "prediction_for_arm_C": (
                "PRE-REGISTERED TO FAIL.  NOTES-MX-M section 2.3: wave 9's "
                "row-norm ordering reached 2.34 per cent coverage at M=4,096, "
                "and a leverage key is that key's second cousin.  Predicted AND "
                "built; a predict-and-skip would close the class on a prediction"
            ),
        },
        "adjudication": {
            "item": "A2",
            "status": "RESOLVED 2026-08-13 by Cyril: a vocabulary row coreset "
            "with a tokenizer remap COUNTS AS FOLDING and is frontier-eligible",
            "scale_caveat_recorded_at_his_instruction": (
                "the token interface is 29.2% of THIS parent and ~1% of a 70B "
                "model by the same arithmetic (docs/PHASE1 section 4.1); these "
                "rows are this model's phase-1 numbers, not a transferable "
                "strategy"
            ),
        },
        "source_model": str(source),
        "source_config_vocab_size": int(source_config["vocab_size"]),
        "source_tensors_sha256": V._sha256(source / "model.safetensors"),
        "source_cell_receipt": str(args.source_cell_receipt or ""),
        "child_model": str(outdir),
        "child_tensors_sha256": V._sha256(outdir / "model.safetensors"),
        "child_directory_sha256": V._directory_sha256(outdir),
        "child_parameters": live,
        "expected_parameters": expected_parameters,
        "parent_parameters": V.PARENT_PARAMS,
        "net_stored_ratio": live / V.PARENT_PARAMS,
        "excised": source_excised,
        "residual_basis": source_basis,
        "axes": source_axes
        + [
            {
                "axis": "vocab_size",
                "kind": "token-interface",
                "sites": ["model.embed_tokens.weight (tied to lm_head)"],
                "config_fields": [
                    "vocab_size",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                ],
                "selection": (
                    f"the {len(order)} rows kept by the mandatory set (all "
                    "added/special tokens, all 256 ByteLevel alphabet tokens and "
                    "the transitive merge closure of both, taken from "
                    "mx43_vocab.mandatory_ids), then the DISCRETIONARY fill in "
                    f"arm {arm}'s order until the budget is exactly met.  The "
                    "census, the closure rule, the budget and the "
                    "skip-if-it-does-not-fit branch are the sealed gather's; the "
                    "ONLY variable is the order the discretionary candidates are "
                    f"considered in.  This arm moved {div['symmetric_difference']} "
                    f"rows of {len(order)} against the sealed control at the same K"
                ),
                "solve": (
                    "NONE.  The rows are gathered verbatim; the identity gate is "
                    "torch.equal against the source table's own bytes.  The "
                    "tokenizer is written by mx43_vocab.remapped_tokenizer, the "
                    "same function that writes the deletion twin's, so a "
                    "difference between the two children cannot be a difference "
                    "between two tokenizer writers"
                ),
                "evidence": (
                    "the sealed corpus, for a token OCCURRENCE census only; no "
                    "activation, no capture, no forward pass.  Arm C additionally "
                    "reads the source checkpoint's own embedding table to form "
                    "the tied Gram, which is not evidence and not a fit"
                ),
                "parameters_removed": (65536 - len(order))
                * int(source_config["hidden_size"]),
                "heldout": (
                    "not applicable: nothing is fitted.  What IS measured out of "
                    "sample is the segmentation shift -- position agreement "
                    f"{corpus_agreement['position_agreement']:.4f} on the sealed "
                    f"corpus and {review_agreement['position_agreement']:.4f} on "
                    "the frozen review prompts"
                ),
            }
        ],
        "vocabulary": {
            **report,
            "sealed_control_report": sealed_report,
            "sealed_corpus": corpus_meta,
            "sealed_corpus_coverage_donor_mapped": cov,
            "sealed_corpus_coverage_canonical": cov_canonical,
            "below_census_floor": bool(lost),
            "merge_closure": closure_state,
            "lossless_on_text_and_agreement": {
                "sealed_corpus": corpus_agreement,
                "frozen_review_prompts": review_agreement,
            },
            "frozen_review_prompt_coverage_DIAGNOSTIC_ONLY": V.coverage(
                review_counts, keptset
            ),
            "tokens_per_byte": tokens_per_byte,
        },
        "identity_gate": {
            "kind": "bytes",
            "assertion": "torch.equal(gathered_table, source_table[kept_ids])",
            "passed": True,
        },
        "evidence": {
            "trace_cache_key": str(V.TRACE.name),
            "fit_rows": 0,
            "heldout_rows": 0,
            "parent_model_forward_calls": 0,
            "split_authority": "not consumed: a rowset arm solves nothing",
            "conditioned": False,
        },
        "solve": {
            "gradients": 0,
            "optimizer_steps": 0,
            "closed_form": "none -- a rowset arm is a gather under a different order",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (outdir / "mx43-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    # NOTES-MX-AO section 9: parse every artifact in the command that writes it.
    reread = json.loads((outdir / "mx43-build-receipt.json").read_text(encoding="utf-8"))
    if reread["schema"] != SCHEMA or int(reread["child_parameters"]) != live:
        raise SystemExit("the build receipt did not survive the write")
    log(
        f"{args.tag}: {live:,} parameters, {live / V.PARENT_PARAMS:.10f}x, arm "
        f"{arm}, {div['symmetric_difference']} rows moved vs the sealed control"
    )
    return 0


def divergence_mode(args: argparse.Namespace) -> int:
    """Publish, per arm, how many rows it would move.  Gates nothing; costs no GPU.

    An arm whose symmetric difference against the sealed control is ZERO is the
    sealed child, and building it would spend a frozen review re-reading a
    sealed result.  This pre-read says which arms are real before any of them is
    built, and it publishes for the arms that are not built too.
    """

    from transformers import AutoTokenizer

    source = Path(args.source)
    tokenizer_file = source / "tokenizer.json"
    if not tokenizer_file.exists():
        tokenizer_file = V.PARENT / "tokenizer.json"
    vocab = V.Vocabulary(tokenizer_file)
    parent_tok = AutoTokenizer.from_pretrained(str(V.PARENT))
    corpus, _ = V.sealed_corpus_ids()
    corpus_texts = [parent_tok.decode(ids, skip_special_tokens=False) for ids in corpus]
    counts = dict(V.census(corpus))
    for tid, n in V.canonical_census(corpus_texts, parent_tok).items():
        counts[tid] = counts.get(tid, 0) + n
    sealed_key = arm_key("SEALED", counts, None)
    leverage = leverage_scores(source) if "C" in args.arm else None
    report: dict[str, Any] = {
        "schema": "neural-forge-matrix-wave43-rowset-divergence-preread-v1",
        "purpose": (
            "how many rows each candidate re-ranking actually moves at fixed K.  "
            "Publishes and GATES NOTHING; it exists so a wave does not spend a "
            "frozen review on an arm whose kept set is the sealed child's"
        ),
        "source_model": str(source),
        "rungs": [],
    }
    for k in args.k:
        sealed_order, _ = V.choose_kept(vocab, counts, int(k))
        row: dict[str, Any] = {"k": int(k), "sealed_kept_rows": len(sealed_order), "arms": {}}
        for arm in args.arm:
            key = arm_key(arm, counts, leverage)
            order, rep = choose_kept_ranked(vocab, counts, int(k), key)
            row["arms"][arm] = {
                **divergence(sealed_order, order, key, sealed_key, vocab),
                "rows_added_by_topup": rep["rows_added_by_topup"],
                "observed_tokens_admitted_by_ranking": rep[
                    "observed_tokens_admitted_by_ranking"
                ],
            }
            d = row["arms"][arm]
            log(
                f"K={k} arm {arm}: moved {d['symmetric_difference']} rows "
                f"(jaccard {d['jaccard']:.6f}, footrule "
                f"{d['normalised_footrule_vs_parent_id_order']:.6f}, topup "
                f"{rep['rows_added_by_topup']})"
            )
        report["rungs"].append(row)
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        reread = json.loads(Path(args.out).read_text(encoding="utf-8"))
        if reread["schema"] != report["schema"] or len(reread["rungs"]) != len(args.k):
            raise SystemExit("the divergence pre-read did not survive the write")
        log(f"divergence -> {args.out} (re-parsed OK)")
    else:
        print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["divergence", "build"])
    ap.add_argument("--arm", action="append", choices=list(ARMS))
    ap.add_argument("--k", type=int, action="append")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--source", type=Path, default=V.PARENT)
    ap.add_argument("--source-cell-receipt", type=Path)
    ap.add_argument("--child", type=Path)
    ap.add_argument("--tag", default="R")
    ap.add_argument(
        "--expect-parameters",
        type=int,
        help="the deletion twin's parameter count.  The builder REFUSES before "
        "writing if this arm does not land on it to the byte",
    )
    ap.add_argument(
        "--below-floor",
        action="store_true",
        help="build at a K that drops sealed-corpus token occurrences.  The "
        "merge-closure assertion still fires and is still fatal",
    )
    args = ap.parse_args()
    if not args.k:
        raise SystemExit("--k is required")
    if not args.arm:
        raise SystemExit("--arm is required")
    if args.mode == "divergence":
        return divergence_mode(args)
    if args.child is None:
        raise SystemExit("--child is required for --mode build")
    if len(args.k) != 1 or len(args.arm) != 1:
        raise SystemExit("--mode build takes exactly one --k and one --arm")
    if args.expect_parameters is None:
        raise SystemExit(
            "--expect-parameters is required: a rowset arm exists only as a "
            "controlled comparison against a deletion twin, and the twin's count "
            "is asserted before anything is written"
        )
    args.k = args.k[0]
    args.arm = args.arm[0]
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
