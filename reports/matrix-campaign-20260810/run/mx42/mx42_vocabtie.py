#!/usr/bin/env python3
"""Wave 40: C1, vocabulary-axis row TYING -- the estate's worst census kill, built.

`docs/UNBUILT_INVENTORY_2026-08-28.md` §2 ranks this first of nine classes that
were closed without a child.  The kill is at `wave14/NOTES-MX-R.md:65-74`: a
nearest-neighbour cosine over a 4,096-row sample of the post-12,288 tail, median
**0.2977**, and the class was closed on it.  Two things are wrong with that.

1. **The comparison was tie against VERBATIM.**  The mechanism that actually won
   this campaign, and that sets its floor, is **deletion**.  A tied row at cosine
   0.30 is a poor stand-in for its members; a deleted row is a worse one.  The
   comparison that decides the class -- tie against delete, at one budget -- was
   never run.
2. **The statistic is a reconstruction statistic**, on the axis this estate has
   sealed five times as non-predictive (`wave14/NOTES-MX-R.md:264-269`: fourteen
   coresets inside the alive fidelity band composing to 0/15).

So the 0.2977 does not get to be the verdict.  It gets to be the **instrument**:
it is exactly the nearest-neighbour assignment this file uses, so the kill
statistic makes its own prediction and is then read against a child.

### WHAT WAS MEASURED BEFORE THIS FILE WAS WRITTEN, AND WHAT IT FORCED

C1 as specified is a **many-to-one** vocabulary: the tail tokens keep their
merges and their positions, and several of them address one shared row.  That
form was tried first, on this parent, on this stack, and **it is not
constructible as a stock child.**  A `tokenizer.json` whose `model.vocab` maps
seven distinct tokens onto one id is refused at load:

    Exception: Error while attempting to unpickle Tokenizer:
    Token `Ã¹` out of vocabulary at line 1 column 2145524

**and the token it names is different on different runs** -- five loads of the
same file named `Ã¹` three times, `Ġexpressive` once and `Ã¹` at three different
column offsets.  That is the Rust `HashMap` id->token inverse being built in a
nondeterministic order, surfacing as a nondeterministic refusal.  Had it loaded,
the same nondeterminism would have made `decode` a non-function: the string a
tied id renders as would depend on the process that rendered it, and a frozen
review of a child whose decode is not a function is not a frozen review.

**That is a measurement and it is this wave's first result.**  It is recorded in
the receipt under `many_to_one_probe` and it is why the arm below is the
INJECTIVE realisation of C1 rather than the many-to-one one.

### THE ARM THAT IS BUILT, AND WHY IT IS THE BETTER EXPERIMENT ANYWAY

At a budget of R rows:

* the **deletion twin** keeps the top R ids by the census+closure ranking,
  verbatim, and drops every other token -- which then splits into kept pieces;
* this **tie arm** keeps the same R ids, chosen by the same function on the same
  census, and writes the same tokenizer -- and then spends the **c lowest-ranked
  non-mandatory rows** of that budget as shared rows, each carrying the
  closed-form weighted centroid of itself and every dropped token that is nearer
  to it than to any other tie row.

Everything else is controlled to the byte: the parameter count, the tokenizer,
the merge closure, the segmentation, the coverage, the tokens-per-byte and the
body underneath.  **The only difference between this child and its deletion twin
is the values in c rows of one tensor.**  That is a cleaner reading of "does
tying beat deleting" than the many-to-one form would have given, because the
many-to-one form changes the tokenizer at the same time and a difference could
then be either.

The head is still verbatim: `R - c` rows are `torch.equal` to the source's own
bytes, asserted and not toleranced, exactly as the deletion arm's whole table is.
The mandatory set -- all 509 added/special tokens, all 256 ByteLevel alphabet
rows and the transitive closure of both -- is EXCLUDED from the tie rows by
construction, because a byte-alphabet row carrying the mean of thirty dropped
tokens would break the only reason an arbitrary byte can still be encoded.

### THE SHARED ROW IS CLOSED FORM AND THERE IS NO SOLVE

For tie row j with its own parent row e_j and assigned tail rows {e_t}:

    row_j = (w_j e_j + SUM_t w_t e_t) / (w_j + SUM_t w_t),   w = 1 + census count

the weighted centroid, which is the minimiser of the weighted sum of squared
distances to its members and is the closed form the critique specified.  The
weight is `1 + count` rather than `count` because the tail is the part of the
table the census did not see: on a 67,584-position census almost every tail row
has count zero, so `count` alone would make the centroid undefined and `1 +
count` reduces to the plain mean exactly where the evidence is silent and tilts
toward the observed rows exactly where it is not.  Nothing is fitted, no
activation is read, no forward pass is made.

Usage::

    mx42_vocabtie.py --mode build --k 9216 --tie-rows 512 \\
        --source <model dir> --child <out dir> --tag T9CL896X09 \\
        --expect-parameters 93504256 [--below-floor]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

#: EVERY shared part of this arm is imported from the deletion arm rather than
#: reimplemented.  `duplicated-contract-decay`: a contract copied away from its
#: source goes stale silently, and the ONE claim this child makes about its twin
#: is that their tokenizers are the same object.  Importing `choose_kept`,
#: `remapped_tokenizer`, `merge_closure_intact` and the census makes that claim
#: true by construction instead of by inspection -- and it means wave 39's five
#: asserted vocabulary-gather guards cover this arm too, on the shared path.
import mx42_vocab as V

SCHEMA = "neural-forge-matrix-wave42-tied-vocabulary-child-v1"
PARENT = V.PARENT
PARENT_PARAMS = V.PARENT_PARAMS
SIDECARS = V.SIDECARS

#: What the many-to-one form did on this stack, before this file existed.  It is
#: carried in every receipt this module writes because it is the reason the arm
#: has the shape it has.
MANY_TO_ONE_PROBE = {
    "what_was_tried": (
        "C1 as literally specified: a tokenizer.json whose model.vocab maps "
        "several distinct tail tokens onto one row id, so the tail keeps its "
        "merges and its positions and addresses a shared row"
    ),
    "result": "REFUSED AT LOAD by tokenizers 0.22.2 / transformers 5.12.1",
    "error": (
        "Exception: Error while attempting to unpickle Tokenizer: Token `...` "
        "out of vocabulary -- raised from AutoTokenizer.from_pretrained through "
        "copy.deepcopy(tokenizer_object).__setstate__"
    ),
    "and_the_refusal_itself_is_nondeterministic": (
        "five loads of the identical file named three different tokens at four "
        "different column offsets, which is the Rust HashMap id->token inverse "
        "being built in a nondeterministic order.  Had it loaded, decode of a "
        "tied id would have been a non-function across processes"
    ),
    "consequence": (
        "the many-to-one form of C1 is not a stock-loadable child on this stack "
        "at all, so it cannot be frozen-reviewed and cannot be sealed.  The "
        "INJECTIVE form below is what is built, and it is the better controlled "
        "comparison anyway: it changes c rows of one tensor and nothing else"
    ),
}


def log(message: str) -> None:
    print(f"[mx42tie] {message}", flush=True)


def tie_partition(
    vocab: V.Vocabulary, order: list[int], counts: dict[int, int], tie_rows: int
) -> tuple[list[int], list[int]]:
    """Split the kept budget into a VERBATIM head and `tie_rows` shared rows.

    The tie rows are the LOWEST-RANKED kept ids that are not mandatory, by the
    same `(-census count, parent id)` ranking `choose_kept` selects with -- so
    the rows spent on the tail are the rows the census cared least about, and
    the frequent head is untouched.  The mandatory set is excluded outright: a
    ByteLevel alphabet row carrying the centroid of thirty dropped tokens would
    break the only reason an arbitrary byte can still be encoded on a parent
    with `byte_fallback: false` and `unk_token: null`.
    """

    mandatory = V.mandatory_ids(vocab)
    eligible = [tid for tid in order if tid not in mandatory]
    if tie_rows > len(eligible):
        raise SystemExit(
            f"--tie-rows {tie_rows} exceeds the {len(eligible)} non-mandatory "
            f"rows in a budget of {len(order)}; the mandatory set is "
            f"{len(mandatory)} rows and may not be spent"
        )
    ranked = sorted(eligible, key=lambda tid: (-counts.get(tid, 0), tid))
    tie = sorted(ranked[len(ranked) - tie_rows:])
    tieset = set(tie)
    head = [tid for tid in order if tid not in tieset]
    return head, tie


def assign_tail(table: Any, tail_ids: list[int], tie_ids: list[int], torch: Any):
    """Nearest tie row for every dropped id, by cosine -- the wave-14 statistic.

    This is deliberately the SAME measurement the census used to kill the class:
    `wave14/NOTES-MX-R.md:65-74` reports a median nearest-neighbour cosine of
    0.2977 over the post-12,288 tail.  Using it as the assignment rule means the
    kill statistic supplies this child's structure and is then read against a
    behavioural verdict rather than standing in for one.

    Chunked because the full matrix is |tail| x |tie| and the tail is 56,320
    rows.  `torch.max` returns the FIRST maximal index, so the tie-break is the
    lowest tie-row index and the assignment is deterministic.
    """

    tie_index = torch.tensor(tie_ids, dtype=torch.long)
    tie_rows = table.index_select(0, tie_index).to(torch.float32)
    tie_unit = tie_rows / tie_rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    assignment: list[int] = []
    cosines: list[float] = []
    chunk = 4096
    for start in range(0, len(tail_ids), chunk):
        block = tail_ids[start : start + chunk]
        rows = table.index_select(0, torch.tensor(block, dtype=torch.long)).to(torch.float32)
        unit = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = unit @ tie_unit.T
        best, where = torch.max(sim, dim=1)
        assignment.extend(int(x) for x in where)
        cosines.extend(float(x) for x in best)
    return assignment, cosines


def build(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import load_file, save_file

    source = Path(args.source)
    outdir = Path(args.child)
    started = time.monotonic()
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    tokenizer_file = source / "tokenizer.json"
    if not tokenizer_file.exists():
        tokenizer_file = PARENT / "tokenizer.json"
    vocab = V.Vocabulary(tokenizer_file)

    # The source child's own axes, carried VERBATIM out of its own build receipt,
    # exactly as the deletion arm carries them.  A composed child that restates
    # its parent's provenance in a driver is a child whose provenance is that
    # driver's opinion.
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
    elif source.resolve() != PARENT.resolve():
        raise SystemExit(
            f"{source} carries no mx*-build-receipt.json, so its axes cannot be "
            "carried and a composed child would seal with incomplete provenance."
        )
    if int(source_config["vocab_size"]) != 65536:
        raise SystemExit(
            f"--source declares vocab_size {source_config['vocab_size']}; this "
            "builder ties the PARENT vocabulary once, not a folded one twice."
        )

    from transformers import AutoTokenizer

    parent_tok = AutoTokenizer.from_pretrained(str(PARENT))
    corpus, corpus_meta = V.sealed_corpus_ids()
    corpus_texts = [parent_tok.decode(ids, skip_special_tokens=False) for ids in corpus]
    donor_counts = V.census(corpus)
    canonical_counts = V.canonical_census(corpus_texts, parent_tok)
    counts = dict(donor_counts)
    for tid, n in canonical_counts.items():
        counts[tid] = counts.get(tid, 0) + n

    # THE SELECTION IS THE DELETION ARM'S, called rather than copied.
    order, report = V.choose_kept(vocab, counts, int(args.k))
    keptset = set(order)
    log(f"kept {len(order)} rows of 65,536 ({report['mandatory_rows']} mandatory)")
    cov = V.coverage(donor_counts, keptset)
    cov_canonical = V.coverage(canonical_counts, keptset)
    log(
        f"sealed-corpus coverage: donor-mapped {cov['occurrence_coverage']:.6f} "
        f"({cov['occurrences_lost']} lost), canonical "
        f"{cov_canonical['occurrence_coverage']:.6f} "
        f"({cov_canonical['occurrences_lost']} lost)"
    )
    closure_state = V.merge_closure_intact(vocab, keptset)
    if not closure_state["intact"]:
        raise SystemExit(
            f"{closure_state['unproducible_kept_tokens']} kept tokens cannot be "
            "produced from the restricted merge list.  The closure is broken; "
            "this is a defect and there is no flag for it."
        )
    lost = int(cov["occurrences_lost"]) + int(cov_canonical["occurrences_lost"])
    # THE COVERAGE GATE IS THE TWIN'S, UNCHANGED AND UNRELAXED.  A tie child at
    # budget R has the deletion twin's tokenizer exactly, so it loses exactly the
    # occurrences the twin loses -- the tie buys row CONTENT, not coverage, and a
    # builder that quietly waived this gate would be publishing a different
    # child's coverage.
    if lost and not args.below_floor:
        raise SystemExit(
            f"K={args.k} is BELOW THE CENSUS FLOOR: the kept set loses "
            f"{cov['occurrences_lost']} donor-mapped and "
            f"{cov_canonical['occurrences_lost']} canonical sealed-corpus "
            "occurrences.  This is the deletion twin's gate, unchanged: tying "
            "buys row content and not coverage.  Re-run with --below-floor to "
            "take it deliberately."
        )
    if lost:
        log(
            f"BELOW FLOOR, deliberately and identically to the twin: "
            f"{cov['occurrences_lost']} donor-mapped and "
            f"{cov_canonical['occurrences_lost']} canonical occurrences dropped"
        )

    head_ids, tie_ids = tie_partition(vocab, order, counts, int(args.tie_rows))
    tail_ids = sorted(tid for tid in vocab.token_of_id if tid not in keptset)
    log(
        f"budget {len(order)} = {len(head_ids)} verbatim head + {len(tie_ids)} "
        f"tie rows; {len(tail_ids)} dropped ids to place "
        f"({len(vocab.dead_ids)} dead parent rows carry no token and are not "
        "placed, because they can never be produced)"
    )

    outdir.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(source / "model.safetensors"))
    table = tensors["model.embed_tokens.weight"]
    if int(table.shape[0]) != 65536:
        raise SystemExit(f"source table has {table.shape[0]} rows, expected 65,536")

    assignment, cosines = assign_tail(table, tail_ids, tie_ids, torch)
    covered = len(assignment)
    # THE TIE'S WHOLE CLAIM is that every dropped id has an image.  A tie that
    # leaves ids unmapped is deletion wearing a tie's receipt.
    if covered != len(tail_ids):
        raise SystemExit(
            f"{covered} of {len(tail_ids)} dropped ids were placed.  A tie in "
            "which some of the tail has no image is deletion with a different "
            "receipt, and this builder will not write one."
        )
    members: list[list[int]] = [[] for _ in tie_ids]
    for tid, slot in zip(tail_ids, assignment):
        members[slot].append(tid)
    empty = [j for j, m in enumerate(members) if not m]
    if empty:
        raise SystemExit(
            f"{len(empty)} of {len(tie_ids)} tie rows received no member "
            f"(first: {empty[:8]}).  An empty tie row is a row spent on nothing; "
            "lower --tie-rows."
        )

    index = torch.tensor(order, dtype=torch.long)
    gathered = table.index_select(0, index).clone()
    position_of = {tid: i for i, tid in enumerate(order)}

    # THE HEAD IS VERBATIM AND IT IS ASSERTED, not toleranced -- exactly as the
    # deletion arm asserts its whole table.
    head_index = torch.tensor(head_ids, dtype=torch.long)
    head_rows = gathered.index_select(
        0, torch.tensor([position_of[t] for t in head_ids], dtype=torch.long)
    )
    if not torch.equal(head_rows, table[head_index]):
        raise SystemExit(
            "the verbatim head is not the source's own bytes; the gather or the "
            "partition is wrong."
        )

    # THE SHARED ROW.  Closed form, fp64 accumulation, cast back to the table's
    # own dtype.  No gradient, no optimiser, no forward call.
    dtype = gathered.dtype
    weights_used: list[float] = []
    sizes: list[int] = []
    for j, tid in enumerate(tie_ids):
        rows = [tid] + members[j]
        w = torch.tensor(
            [1.0 + float(counts.get(t, 0)) for t in rows], dtype=torch.float64
        )
        block = table.index_select(0, torch.tensor(rows, dtype=torch.long)).to(torch.float64)
        centroid = (w.unsqueeze(1) * block).sum(dim=0) / w.sum()
        gathered[position_of[tid]] = centroid.to(dtype)
        weights_used.append(float(w.sum()))
        sizes.append(len(members[j]))

    # EXACTLY c ROWS MOVE.  Every other row in this child is the source's row at
    # the same token, bit for bit, so a behavioural difference against the twin
    # is attributable to the tie and to nothing else.
    same = torch.eq(gathered, table.index_select(0, index)).all(dim=1)
    moved_rows = int((~same).sum())
    if moved_rows != int(args.tie_rows):
        raise SystemExit(
            f"{moved_rows} rows differ from the source, expected exactly "
            f"{args.tie_rows}.  The control this comparison rests on is that "
            "NOTHING but the tie rows moved."
        )
    tensors["model.embed_tokens.weight"] = gathered
    save_file(tensors, str(outdir / "model.safetensors"), metadata={"format": "pt"})

    # THE TOKENIZER IS THE DELETION ARM'S, written by the deletion arm's own
    # function on the same kept set.
    body, new_of_old = V.remapped_tokenizer(vocab, order)
    (outdir / "tokenizer.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # ...and the merge list is re-derived INDEPENDENTLY here and compared, so the
    # claim "this child's tokenizer is the twin's tokenizer" is checked by two
    # implementations agreeing rather than by one being trusted.
    kept_tokens = {vocab.token_of_id[t] for t in keptset if t in vocab.token_of_id}
    expected_merges = sum(
        1
        for left, right in vocab.merges
        if left in kept_tokens and right in kept_tokens and (left + right) in kept_tokens
    )
    merges_kept = len(body["model"]["merges"])
    if merges_kept != expected_merges:
        raise SystemExit(
            f"the written merge list has {merges_kept} entries and the closure "
            f"rule says {expected_merges}.  This child's tokenizer must be the "
            "deletion twin's tokenizer or the comparison is not controlled."
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

    for name in SIDECARS:
        src = source / name
        if not src.exists():
            src = PARENT / name
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

    # ------------------------------------------------------------------ proof
    child_tok = AutoTokenizer.from_pretrained(str(outdir))
    child_tok_again = AutoTokenizer.from_pretrained(str(outdir))
    probe_ids = list(range(len(order)))
    decoded_probe = child_tok.convert_ids_to_tokens(probe_ids)
    decoded_probe_again = child_tok_again.convert_ids_to_tokens(probe_ids)
    # THIS CHILD'S DECODE IS A FUNCTION, AND THAT IS NOT FREE.  The many-to-one
    # form of this same class does not load at all on this stack and named a
    # DIFFERENT token in its refusal on every attempt (see MANY_TO_ONE_PROBE).
    # The injective form is asserted to be injective rather than assumed to be:
    # two independently constructed tokenizers, two independently built id->token
    # inverses, one answer.
    if decoded_probe != decoded_probe_again:
        first = next(
            i for i, (a, b) in enumerate(zip(decoded_probe, decoded_probe_again)) if a != b
        )
        raise SystemExit(
            f"the child's id->token map is not a function: id {first} reads "
            f"{decoded_probe[first]!r} on one load and "
            f"{decoded_probe_again[first]!r} on another.  A frozen review of a "
            "child whose decode depends on the process is not a frozen review."
        )
    review_texts = V.review_pack_texts()

    def _agreement(texts: list[str]) -> dict[str, Any]:
        lossless = 0
        exact = 0
        positions = 0
        agreed = 0
        dropped = 0
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
    if corpus_agreement["rows_lossless_on_text"] != len(corpus_texts) or review_agreement[
        "rows_lossless_on_text"
    ] != len(review_texts):
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

    live = int(sum(int(t.numel()) for t in tensors.values()))
    expected_parameters = int(args.expect_parameters)
    # THE COUNT IS THE CONTROL.  This child exists to be read against a deletion
    # twin at the same ratio, and the caller states that twin's count on the
    # command line before the build runs.  A count that drifts turns a controlled
    # comparison into two unrelated readings.
    if int(live) != int(expected_parameters):
        raise SystemExit(
            f"this child has {live:,} parameters and the deletion twin it is "
            f"compared against has {expected_parameters:,}.  A tie arm that does "
            "not land on its twin's count to the byte is not a controlled "
            "comparison."
        )

    import statistics

    receipt = {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "mx42_vocabtie.py",
        "tag": str(args.tag),
        "arm": "vocab-row-TYING, injective realisation (zero-solve, closed form)",
        "repairs_census_kill": {
            "item": "C1 -- vocabulary-axis row tying",
            "killed_at": "wave 14, on a nearest-neighbour cosine of 0.2977 over "
            "a 4,096-row sample of the post-12,288 tail "
            "(wave14/NOTES-MX-R.md:65-74)",
            "why_the_kill_does_not_hold": (
                "the comparison was tie against VERBATIM.  The mechanism that "
                "shipped and set this campaign's floor is DELETION, and a "
                "deleted row is a worse stand-in for its members than a "
                "0.30-cosine tied one.  The comparison that decides the class "
                "was never run"
            ),
            "the_kill_statistic_is_now_the_instrument": (
                "the same nearest-neighbour cosine is this child's assignment "
                "rule, so the number that closed the class supplies the "
                "structure and is then read against a behavioural verdict"
            ),
        },
        "many_to_one_probe": MANY_TO_ONE_PROBE,
        "source_model": str(source),
        "source_config_vocab_size": int(source_config["vocab_size"]),
        "source_tensors_sha256": V._sha256(source / "model.safetensors"),
        "source_cell_receipt": str(args.source_cell_receipt or ""),
        "child_model": str(outdir),
        "child_tensors_sha256": V._sha256(outdir / "model.safetensors"),
        "child_tokenizer_sha256": V._sha256(outdir / "tokenizer.json"),
        "child_directory_sha256": V._directory_sha256(outdir),
        "child_parameters": live,
        "parent_parameters": PARENT_PARAMS,
        "net_stored_ratio": live / PARENT_PARAMS,
        "expected_parameters_asserted": expected_parameters,
        "excised": source_excised,
        "residual_basis": source_basis,
        "axes": source_axes
        + [
            {
                "axis": "vocab_size",
                "kind": "token-interface (TIED, not deleted)",
                "sites": ["model.embed_tokens.weight (tied to lm_head)"],
                "config_fields": [
                    "vocab_size",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                ],
                "selection": (
                    f"the deletion arm's own kept set at K={args.k}, chosen by "
                    "mx42_vocab.choose_kept on the same two-stream census -- "
                    "imported and called, not reimplemented.  Of those "
                    f"{len(order)} rows, the {len(head_ids)} highest-ranked are "
                    f"kept VERBATIM and the {len(tie_ids)} lowest-ranked "
                    "non-mandatory rows are spent as SHARED rows.  The mandatory "
                    "set (509 added/special tokens, 256 ByteLevel alphabet rows "
                    "and the transitive closure of both) is excluded from the "
                    "tie rows by construction"
                ),
                "solve": (
                    "NONE, and the head is still torch.equal to the source's own "
                    "bytes.  Each tie row is the closed-form weighted centroid "
                    "of itself and the dropped tokens nearest to it in cosine, "
                    "weight 1 + census count, accumulated in fp64 and cast back. "
                    "Exactly "
                    f"{moved_rows} of {len(order)} rows differ from the source, "
                    "asserted"
                ),
                "evidence": (
                    "the sealed corpus, for a token OCCURRENCE census only, and "
                    "the source table's own rows for the cosine assignment; no "
                    "activation, no capture, no forward pass"
                ),
                "parameters_removed": (65536 - len(order))
                * int(source_config["hidden_size"]),
                "heldout": (
                    "not applicable: nothing is fitted.  The tokenizer is the "
                    "deletion twin's, so segmentation, coverage and "
                    "tokens-per-byte are controlled to the byte and the only "
                    "difference between the two children is the values in "
                    f"{moved_rows} rows"
                ),
            }
        ],
        "tying": {
            "k": int(args.k),
            "tie_rows": int(args.tie_rows),
            "verbatim_head_rows": len(head_ids),
            "rows_that_differ_from_the_source": moved_rows,
            "dropped_ids_placed": covered,
            "dropped_ids_total": len(tail_ids),
            "dead_parent_rows_not_placed": len(vocab.dead_ids),
            "assignment_rule": (
                "nearest tie row by cosine on the SOURCE table's own rows; "
                "torch.max returns the first maximal index so the tie-break is "
                "the lowest tie-row index and the assignment is deterministic"
            ),
            "assignment_cosine": {
                "median": statistics.median(cosines),
                "mean": sum(cosines) / len(cosines),
                "min": min(cosines),
                "max": max(cosines),
                "wave14_census_median_for_comparison": 0.2977,
                "note": (
                    "wave 14 measured a median nearest-neighbour cosine of "
                    "0.2977 over a 4,096-row sample of the post-12,288 tail "
                    "against the KEPT set, and closed the class on it.  The "
                    "number here is against the TIE ROWS only, which is a "
                    "smaller and lower-ranked target set, so it is expected to "
                    "be lower, not higher.  It is published so the kill "
                    "statistic and this child's structure can be read together"
                ),
            },
            "cluster_sizes": {
                "min": min(sizes),
                "median": statistics.median(sizes),
                "mean": sum(sizes) / len(sizes),
                "max": max(sizes),
                "empty": 0,
            },
            "shared_row_form": (
                "weighted centroid, w = 1 + census count, fp64 accumulation, "
                "cast back to the table dtype.  The minimiser of the weighted "
                "sum of squared distances to the members, which is the closed "
                "form FOLD_CLASS_CRITIQUE C1 specified"
            ),
            "merge_list_entries": merges_kept,
            "merge_list_independently_rederived": expected_merges,
        },
        "vocabulary": {
            **report,
            "sealed_corpus": corpus_meta,
            "sealed_corpus_coverage_donor_mapped": cov,
            "sealed_corpus_coverage_canonical": cov_canonical,
            "below_census_floor": bool(lost),
            "merge_closure": closure_state,
            "lossless_on_text_and_agreement": {
                "sealed_corpus": corpus_agreement,
                "frozen_review_prompts": review_agreement,
                "claim": "decode(child_encode(t)) == t for every t, and the "
                "id->token map is a FUNCTION -- asserted across two "
                "independently loaded tokenizers, because the many-to-one form "
                "of this class is not",
            },
            "tokens_per_byte": tokens_per_byte,
        },
        "identity_gate": {
            "kind": "bytes-on-the-head",
            "assertion": "torch.equal(head_rows, source_table[head_ids]) over "
            f"the {len(head_ids)} verbatim rows, and exactly {moved_rows} rows "
            "differ from the source in the whole table",
            "passed": True,
        },
        "evidence": {
            "trace_cache_key": str(V.TRACE.name),
            "fit_rows": 0,
            "heldout_rows": 0,
            "parent_model_forward_calls": 0,
            "split_authority": "not consumed: this arm solves nothing",
            "conditioned": False,
        },
        "solve": {
            "gradients": 0,
            "optimizer_steps": 0,
            "closed_form": "weighted centroid per tie row",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (outdir / "mx42-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    # NOTES-MX-AO section 9: parse every artifact in the command that writes it.
    reread = json.loads((outdir / "mx42-build-receipt.json").read_text(encoding="utf-8"))
    if reread["schema"] != SCHEMA or reread["child_parameters"] != live:
        raise SystemExit("the build receipt did not survive its own write")
    log(
        f"{args.tag}: {live:,} parameters, {live / PARENT_PARAMS:.6f}x, "
        f"{len(head_ids)} verbatim + {len(tie_ids)} tied rows carrying "
        f"{covered} dropped ids, median assignment cosine "
        f"{statistics.median(cosines):.4f}, receipt re-parsed OK"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["build"])
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument(
        "--tie-rows",
        type=int,
        required=True,
        help="how many of the K rows are spent as SHARED rows.  They are taken "
        "from the lowest-ranked non-mandatory end of the kept set",
    )
    ap.add_argument("--source", type=Path, default=PARENT)
    ap.add_argument("--source-cell-receipt", type=Path)
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--tag", default="T")
    ap.add_argument(
        "--expect-parameters",
        type=int,
        required=True,
        help="the DELETION TWIN's parameter count.  This child must land on it "
        "to the byte or the comparison is not controlled, and the builder "
        "refuses rather than writing an uncontrolled child",
    )
    ap.add_argument(
        "--below-floor",
        action="store_true",
        help="build at a K that drops sealed-corpus token occurrences.  This is "
        "the deletion twin's gate, unchanged: tying buys row content, not "
        "coverage",
    )
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
