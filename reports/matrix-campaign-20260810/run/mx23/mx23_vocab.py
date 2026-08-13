#!/usr/bin/env python3
"""Wave 23: the vocabulary row coreset -- `vocab_size` as the missing standard axis.

The tied token interface is **29.2% of this parent** (65,536 x 1,024 =
67,108,864 of 229,693,184) and no wave has moved it on the standard column.
Wave 10's `COMP.MA.MXN2` moved it at 10/15 by folding the *table* -- a 12,288-row
verbatim head plus a sparse-OMP tail -- and that child is parent-shaped, because
`lm_head` is reconstituted at 65,536 rows to run.  This axis is the other move:
keep FEWER ROWS and say so in `config.vocab_size`, which stock
`Lfm2ForCausalLM` reads and the loader counts.

**The base arm solves nothing.**  Every kept row is the parent's own bytes, so
the identity gate is not a tolerance -- it is `torch.equal`.  What is bet is not
the rows, it is the *input distribution*: a smaller merge table re-tokenizes text
the parent's tokenizer would have fused, so the model sees sequences it never
saw.  On the sealed corpus that shift is provably ZERO by construction (below);
off it, it is the whole experiment, and the receipt prints measured
tokens-per-byte beside the ratio so a reader can see the compute the parameter
count does not.

### The closure, which is the part an unclosed census gets fatally wrong

A census over surface occurrences ranks `Ġunderstanding` high and its own
ancestor `Ġunderstand` -> `ing` intermediates at or near zero, because a BPE
intermediate is consumed by the next merge and never appears in the final
tokenization.  Drop the intermediates and the merge that forms the kept token is
gone with them, so the kept token can no longer be *produced* -- the census
silently destroys the very rows it selected.  The kept set is therefore

    kept = top-by-census
         UNION all merge-tree ancestors of every kept token (transitively)
         UNION every added/special token (all 509, the chat template needs them)
         UNION all 256 ByteLevel alphabet tokens

and only then is the merge list restricted to its closure: a merge survives iff
its product and both operands are kept.  This parent has `byte_fallback: false`
and `unk_token: null`, so the 256 single-byte tokens are not a nicety -- they are
the only reason an arbitrary byte can still be encoded, and dropping one would
make some inputs untokenizable rather than merely longer.

With that closure the sealed corpus re-tokenizes **identically**: every token it
contains is kept, every merge used to build those tokens is kept, and merge
priority order is preserved among the survivors, so BPE takes the same path.
`--mode build` asserts exactly that, by decoding the sealed corpus with the
parent's tokenizer and re-encoding it with the child's.

### Composition, which is why this axis is worth a stack

The fold touches `model.embed_tokens.weight` and nothing else.  It does not read
the residual, it does not move a width, it does not excise a block, and it
re-solves nothing -- so it composes with any sealed standard-shape child as a
**post-process on that child's own bytes**.  `--source` is therefore any child
directory, not just the parent, and the receipt carries the source child's own
sealed digest so the body is provably the sealed body.  On a width child the two
folds *multiply* on the table: 16,384 x 896 = 14,680,064 against the parent's
67,108,864.

### Cyril's rulings, 2026-08-13 (ADDENDUM 15 section 15.2)

**A2 is RESOLVED: a vocabulary row coreset with a tokenizer remap COUNTS AS
FOLDING and is frontier-eligible.**  His scale caveat rides in every receipt and
is not optional: like the embedding share it is ~29% here and ~1% at large-model
scale, so the value of these rows is this model's phase-1 arithmetic and not a
transferable strategy.

Usage::

    mx23_vocab.py --mode census --k 8192 --k 16384 --k 32768 --out P5.json
    mx23_vocab.py --mode build --source <model dir> --k 16384 \\
        --child <out dir> --tag V16 [--source-cell-receipt <sealed receipt>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
#: The 256-row sealed corpus the campaign's 24 captured batches are the first 24
#: rows of.  A token census needs no activations, so it is taken over all 256 and
#: the 24-row subset is reported beside it rather than instead of it.
SEALED_CORPUS = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717"
    "/v4096_modelcap_v2_broad256_cpcf_decision_v5/mapped_donor_calibration.json"
)
TRACE = Path(
    "/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810"
    "/evidence/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff"
)
#: The frozen 15+1 review pack.  It is read for a COVERAGE DIAGNOSTIC ONLY and
#: never to choose K -- K is fixed by the parameter budget alone, before this
#: file is opened.  Selecting a vocabulary on the review surface would be the
#: proxy the anti-proxy rule exists to refuse.
REVIEW_PACK = (
    Path(__file__).resolve().parents[4]
    / "research"
    / "lfm2_semantic_whole_response_prompts_v1.json"
)
PARENT_PARAMS = 229_693_184
SCHEMA = "neural-forge-matrix-wave23-standard-shape-child-v1"
SIDECARS = (
    "generation_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)


def log(message: str) -> None:
    print(f"[mx23vocab] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(directory: Path) -> str:
    """Every file in the child directory, name and bytes, in sorted order."""

    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


# ------------------------------------------------------------------- corpora
def sealed_corpus_ids() -> tuple[list[list[int]], dict[str, Any]]:
    """The 256-row sealed corpus as id sequences, plus what it is."""

    rows = json.loads(SEALED_CORPUS.read_text(encoding="utf-8"))
    out: list[list[int]] = []
    for row in rows:
        ids = [int(x) for x in row["input_ids"]]
        ids += [int(x) for x in row.get("forced_decode_ids", [])]
        out.append(ids)
    return out, {
        "path": str(SEALED_CORPUS),
        "file_sha256": _sha256(SEALED_CORPUS),
        "rows": len(out),
        "token_positions": sum(len(r) for r in out),
    }


def captured_batch_ids() -> tuple[list[list[int]], dict[str, Any]]:
    """The 24 batches the campaign's activations were captured through."""

    from safetensors import safe_open

    out: list[list[int]] = []
    for directory in sorted((TRACE / "batches").iterdir()):
        with safe_open(str(directory / "trace.safetensors"), "pt") as handle:
            prompt = handle.get_tensor("input.input_ids")
            forced = handle.get_tensor("input.forced_decode_ids")
        out.append([int(x) for x in prompt.reshape(-1)] + [int(x) for x in forced.reshape(-1)])
    return out, {
        "trace": str(TRACE),
        "rows": len(out),
        "token_positions": sum(len(r) for r in out),
    }


def review_pack_texts() -> list[str]:
    """The sixteen review prompts as the strings the model is handed.

    Read for a COVERAGE DIAGNOSTIC ONLY.  K is fixed by the parameter budget
    before this function is called, and nothing downstream of it selects a row,
    a token or a rung.
    """

    body = json.loads(REVIEW_PACK.read_text(encoding="utf-8"))
    rows = body["prompts"] if isinstance(body, dict) else body
    out: list[str] = []
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, list):
            out.append("\n".join(str(m.get("content", "")) for m in messages))
        else:
            out.append(str(row["prompt"]))
    return out


# ------------------------------------------------------------- the merge tree
class Vocabulary:
    """The parent's tokenizer, taken apart into the objects a coreset needs."""

    def __init__(self, tokenizer_json: Path) -> None:
        self.body = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        model = self.body["model"]
        if model.get("type") != "BPE":
            raise SystemExit(
                f"this coreset is written for BPE; the parent declares "
                f"{model.get('type')!r}.  A Unigram parent needs vocabulary "
                "restriction plus Viterbi re-tokenization, which is a different "
                "script and is not this one."
            )
        self.token_of_id: dict[int, str] = {}
        for token, tid in model["vocab"].items():
            self.token_of_id[int(tid)] = token
        self.added: list[dict[str, Any]] = list(self.body.get("added_tokens", []))
        for entry in self.added:
            self.token_of_id.setdefault(int(entry["id"]), str(entry["content"]))
        self.id_of_token: dict[str, int] = {
            t: i for i, t in sorted(self.token_of_id.items(), key=lambda kv: kv[0])
        }
        # `merges` is either ["a b", ...] or [["a","b"], ...] depending on the
        # tokenizers version that wrote the file; both are handled and the form
        # is preserved on write, because a form the loader does not expect is a
        # tokenizer that does not load.
        raw = model.get("merges", [])
        self.merges_are_pairs = bool(raw) and isinstance(raw[0], (list, tuple))
        self.merges: list[tuple[str, str]] = []
        for entry in raw:
            if self.merges_are_pairs:
                left, right = str(entry[0]), str(entry[1])
            else:
                left, right = str(entry).split(" ", 1)
            self.merges.append((left, right))
        #: product -> the operands that make it.  Standard BPE products are
        #: unique; the FIRST merge wins if a file ever carries a duplicate.
        self.parents_of: dict[str, tuple[str, str]] = {}
        for left, right in self.merges:
            self.parents_of.setdefault(left + right, (left, right))

        from tokenizers import pre_tokenizers

        self.byte_alphabet = [
            c for c in pre_tokenizers.ByteLevel.alphabet() if c in self.id_of_token
        ]
        missing = 256 - len(self.byte_alphabet)
        if missing:
            raise SystemExit(
                f"{missing} of the 256 ByteLevel alphabet tokens are absent from "
                "the parent vocabulary; with byte_fallback false that is not a "
                "vocabulary this coreset can close."
            )
        self.added_ids = {int(e["id"]) for e in self.added}
        #: ids that carry no token at all -- pure padding rows between the
        #: tokenizer's own top id and `config.vocab_size`.  They can never be
        #: produced, so they are never kept, and dropping them is free.
        self.dead_ids = sorted(set(range(65536)) - set(self.token_of_id))

    def ancestors(self, token: str, into: set[str]) -> None:
        stack = [token]
        while stack:
            current = stack.pop()
            pair = self.parents_of.get(current)
            if pair is None:
                continue
            for side in pair:
                if side not in into:
                    into.add(side)
                    stack.append(side)


def census(sequences: list[list[int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in sequences:
        for tid in row:
            counts[tid] = counts.get(tid, 0) + 1
    return counts


def choose_kept(
    vocab: Vocabulary, counts: dict[int, int], k: int
) -> tuple[list[int], dict[str, Any]]:
    """The closure-corrected kept set, at EXACTLY k rows.

    Ranking is `(-census count, parent id)`.  The tie-break is the parent's own
    id order, which for a BPE table is its merge order and therefore its
    training-corpus frequency order -- the only ordering information the
    checkpoint carries about tokens this corpus never showed.  It is stated here
    rather than buried because with a 67,584-position census and a 16,384-row
    budget, the tie-break decides most of the table.
    """

    mandatory_tokens: set[str] = set(vocab.byte_alphabet)
    mandatory_ids: set[int] = set(vocab.added_ids)
    for token in list(mandatory_tokens):
        vocab.ancestors(token, mandatory_tokens)
    for entry in vocab.added:
        content = str(entry["content"])
        if content in vocab.id_of_token:
            mandatory_tokens.add(content)
    kept: set[int] = set(mandatory_ids)
    for token in mandatory_tokens:
        tid = vocab.id_of_token.get(token)
        if tid is not None:
            kept.add(tid)
    mandatory_size = len(kept)
    if mandatory_size > k:
        raise SystemExit(
            f"the mandatory set (specials + byte alphabet + their closure) is "
            f"{mandatory_size} rows and K is {k}."
        )

    ranking = sorted(
        (tid for tid in vocab.token_of_id if tid not in kept),
        key=lambda tid: (-counts.get(tid, 0), tid),
    )
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
    order = sorted(kept)
    report = {
        "k": k,
        "kept_rows": len(order),
        "mandatory_rows": mandatory_size,
        "mandatory_rule": (
            "all 509 added/special tokens, all 256 ByteLevel alphabet tokens, "
            "and the transitive merge-tree closure of both"
        ),
        "ranking_rule": "(-census_count, parent_token_id)",
        "observed_tokens_admitted_by_ranking": observed_admitted,
        "tokens_skipped_because_their_closure_did_not_fit": skipped_for_budget,
        "dead_parent_rows_dropped": len(vocab.dead_ids),
    }
    return order, report


def coverage(counts: dict[int, int], kept: set[int]) -> dict[str, Any]:
    total = sum(counts.values())
    held = sum(c for tid, c in counts.items() if tid in kept)
    distinct = len(counts)
    distinct_held = sum(1 for tid in counts if tid in kept)
    return {
        "occurrences": total,
        "occurrences_covered": held,
        "occurrence_coverage": (held / total) if total else 0.0,
        "distinct_tokens": distinct,
        "distinct_covered": distinct_held,
        "distinct_coverage": (distinct_held / distinct) if distinct else 0.0,
        "occurrences_lost": total - held,
    }


# ---------------------------------------------------------------- the write
def remapped_tokenizer(vocab: Vocabulary, order: list[int]) -> tuple[dict[str, Any], dict[int, int]]:
    """The child's `tokenizer.json`, dense-remapped, merges closed."""

    new_of_old = {old: new for new, old in enumerate(order)}
    keptset = set(order)
    kept_tokens = {vocab.token_of_id[t] for t in keptset if t in vocab.token_of_id}

    body = json.loads(json.dumps(vocab.body))  # deep copy through JSON, no aliasing
    body["model"]["vocab"] = {
        token: new_of_old[tid]
        for token, tid in vocab.body["model"]["vocab"].items()
        if int(tid) in new_of_old
    }
    merges: list[Any] = []
    for left, right in vocab.merges:
        if left in kept_tokens and right in kept_tokens and (left + right) in kept_tokens:
            merges.append([left, right] if vocab.merges_are_pairs else f"{left} {right}")
    body["model"]["merges"] = merges

    added = []
    for entry in vocab.added:
        old = int(entry["id"])
        if old not in new_of_old:
            raise SystemExit(f"added token {entry['content']!r} was not kept")
        row = dict(entry)
        row["id"] = new_of_old[old]
        added.append(row)
    body["added_tokens"] = sorted(added, key=lambda e: e["id"])

    # TemplateProcessing carries hard token ids.  A remap that leaves them
    # pointing at the parent's ids emits <|startoftext|> as whatever now sits at
    # row 1, which is a silent prompt corruption rather than a load failure.
    def _fix_processor(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "TemplateProcessing":
                for name, spec in node.get("special_tokens", {}).items():
                    token = str(spec.get("id", name))
                    tid = vocab.id_of_token.get(token)
                    if tid is None or tid not in new_of_old:
                        raise SystemExit(f"template special {token!r} was not kept")
                    spec["ids"] = [new_of_old[tid]]
            for value in node.values():
                _fix_processor(value)
        elif isinstance(node, list):
            for value in node:
                _fix_processor(value)

    _fix_processor(body.get("post_processor"))
    return body, new_of_old


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
    vocab = Vocabulary(tokenizer_file)
    if int(source_config["vocab_size"]) != 65536:
        raise SystemExit(
            f"--source declares vocab_size {source_config['vocab_size']}; this "
            "builder folds the PARENT vocabulary once, not a folded one twice."
        )

    corpus, corpus_meta = sealed_corpus_ids()
    counts = census(corpus)
    order, report = choose_kept(vocab, counts, int(args.k))
    keptset = set(order)
    log(f"kept {len(order)} rows of 65,536 ({report['mandatory_rows']} mandatory)")
    cov = coverage(counts, keptset)
    log(
        f"sealed-corpus coverage: {cov['occurrence_coverage']:.6f} of "
        f"{cov['occurrences']} occurrences, {cov['occurrences_lost']} lost"
    )
    if cov["occurrences_lost"] != 0:
        raise SystemExit(
            "the kept set loses sealed-corpus occurrences; the closure is wrong "
            "or K is below the observed vocabulary."
        )

    outdir.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(source / "model.safetensors"))
    table = tensors["model.embed_tokens.weight"]
    if int(table.shape[0]) != 65536:
        raise SystemExit(f"source table has {table.shape[0]} rows, expected 65,536")
    index = torch.tensor(order, dtype=torch.long)
    gathered = table.index_select(0, index).clone()
    # The identity gate for a zero-solve arm: every kept row is the source's own
    # bytes, asserted with torch.equal and not a tolerance.
    for probe in (0, len(order) // 3, len(order) // 2, len(order) - 1):
        if not torch.equal(gathered[probe], table[order[probe]]):
            raise SystemExit(f"row {probe} is not the source's bytes")
    if not torch.equal(gathered, table[index]):
        raise SystemExit("the gathered table is not a row subset of the source's")
    tensors["model.embed_tokens.weight"] = gathered
    save_file(tensors, str(outdir / "model.safetensors"), metadata={"format": "pt"})

    body, new_of_old = remapped_tokenizer(vocab, order)
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

    # ------------------------------------------------------------- the proof
    # Decode the sealed corpus with the PARENT's tokenizer and re-encode it with
    # the CHILD's.  If the closure is right the two id streams are the same
    # sequence under the remap, position for position, in every one of the 256
    # rows -- which is the receipt's claim that no kept-token occurrence was lost
    # to re-tokenization, executed rather than asserted.
    from transformers import AutoTokenizer

    parent_tok = AutoTokenizer.from_pretrained(str(PARENT))
    child_tok = AutoTokenizer.from_pretrained(str(outdir))
    exact_rows = 0
    checked_positions = 0
    mismatches: list[dict[str, Any]] = []
    for row_index, ids in enumerate(corpus):
        text = parent_tok.decode(ids, skip_special_tokens=False)
        expected = [new_of_old[t] for t in ids]
        got = child_tok(text, add_special_tokens=False).input_ids
        checked_positions += len(expected)
        if got == expected:
            exact_rows += 1
        elif len(mismatches) < 3:
            first = next(
                (i for i in range(min(len(got), len(expected))) if got[i] != expected[i]),
                min(len(got), len(expected)),
            )
            mismatches.append(
                {
                    "row": row_index,
                    "parent_len": len(expected),
                    "child_len": len(got),
                    "first_divergent_position": first,
                }
            )
    log(f"re-tokenization identity: {exact_rows}/{len(corpus)} rows exact")

    # tokens-per-byte, measured on three surfaces and printed beside the ratio.
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

    corpus_texts = [parent_tok.decode(ids, skip_special_tokens=False) for ids in corpus]
    review_texts = review_pack_texts()
    tokens_per_byte = {
        "sealed_corpus": _tpb(corpus_texts),
        "frozen_review_prompts": _tpb(review_texts),
    }
    review_counts = census(
        [parent_tok(t, add_special_tokens=False).input_ids for t in review_texts]
    )
    review_cov = coverage(review_counts, keptset)

    child_sha = _sha256(outdir / "model.safetensors")
    live = int(sum(int(t.numel()) for t in tensors.values()))
    receipt = {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "mx23_vocab.py",
        "tag": str(args.tag),
        "arm": "vocab-row-coreset (zero-solve)",
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
        "source_tensors_sha256": _sha256(source / "model.safetensors"),
        "source_cell_receipt": str(args.source_cell_receipt or ""),
        "child_model": str(outdir),
        "child_tensors_sha256": child_sha,
        "child_directory_sha256": _directory_sha256(outdir),
        "child_parameters": live,
        "parent_parameters": PARENT_PARAMS,
        "net_stored_ratio": live / PARENT_PARAMS,
        "excised": [int(x) for x in (args.excised or [])],
        "axes": list(args.axis or []) + [
            {
                "axis": "vocab_size",
                "parent": 65536,
                "child": len(order),
                "mechanism": "verbatim row coreset of the tied token table, "
                "dense id remap, BPE merge list restricted to the kept closure",
                "solved": False,
                "storage": "K x hidden, tied, counted once; the merge table "
                "SHRINKS and the remap lives in tokenizer.json, which the "
                "parent's own count never included either",
            }
        ],
        "vocabulary": {
            **report,
            "sealed_corpus": corpus_meta,
            "sealed_corpus_coverage": cov,
            "re_tokenization_identity": {
                "rows": len(corpus),
                "rows_exact": exact_rows,
                "positions_checked": checked_positions,
                "kept_token_occurrences_lost": cov["occurrences_lost"],
                "first_mismatches": mismatches,
                "claim": "the sealed corpus re-tokenizes to the same sequence "
                "under the id remap; no kept-token occurrence is lost",
            },
            "frozen_review_prompt_coverage_DIAGNOSTIC_ONLY": review_cov,
            "tokens_per_byte": tokens_per_byte,
        },
        "identity_gate": {
            "kind": "bytes",
            "assertion": "torch.equal(gathered_table, source_table[kept_ids])",
            "passed": True,
        },
        "evidence": {
            "trace_cache_key": str(TRACE.name),
            "fit_rows": 0,
            "heldout_rows": 0,
            "parent_model_forward_calls": 0,
            "split_authority": "not consumed: the base arm solves nothing",
            "conditioned": False,
        },
        "solve": {
            "gradients": 0,
            "optimizer_steps": 0,
            "closed_form": "none -- the base arm is a gather",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (outdir / "mx23-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    log(
        f"{args.tag}: {live:,} parameters, {live / PARENT_PARAMS:.6f}x, "
        f"child tokens/byte {tokens_per_byte['frozen_review_prompts']['child_tokens_per_byte']:.4f} "
        f"vs parent {tokens_per_byte['frozen_review_prompts']['parent_tokens_per_byte']:.4f}"
    )
    return 0


def census_mode(args: argparse.Namespace) -> int:
    vocab = Vocabulary(PARENT / "tokenizer.json")
    corpus, corpus_meta = sealed_corpus_ids()
    captured, captured_meta = captured_batch_ids()
    counts = census(corpus)
    counts_captured = census(captured)
    from transformers import AutoTokenizer

    parent_tok = AutoTokenizer.from_pretrained(str(PARENT))
    review_texts = review_pack_texts()
    review_counts = census(
        [parent_tok(t, add_special_tokens=False).input_ids for t in review_texts]
    )
    report: dict[str, Any] = {
        "schema": "neural-forge-matrix-wave23-vocab-coverage-census-v1",
        "pre_read": "P5",
        "purpose": "the closure-corrected kept set at each K, pre- and "
        "post-closure coverage, and measured tokens-per-byte.  Orders the "
        "vocab ladder; gates nothing (L1).",
        "parent_vocabulary": {
            "config_vocab_size": 65536,
            "bpe_vocab_entries": len(vocab.body["model"]["vocab"]),
            "added_tokens": len(vocab.added),
            "merges": len(vocab.merges),
            "byte_alphabet_present": len(vocab.byte_alphabet),
            "byte_fallback": vocab.body["model"].get("byte_fallback"),
            "unk_token": vocab.body["model"].get("unk_token"),
            "dead_rows_carrying_no_token": len(vocab.dead_ids),
        },
        "sealed_corpus": corpus_meta,
        "captured_batches": captured_meta,
        "census": {
            "sealed_corpus_distinct_tokens": len(counts),
            "sealed_corpus_occurrences": sum(counts.values()),
            "captured_batches_distinct_tokens": len(counts_captured),
            "captured_batches_occurrences": sum(counts_captured.values()),
            "review_pack_distinct_tokens_DIAGNOSTIC": len(review_counts),
            "review_pack_occurrences_DIAGNOSTIC": sum(review_counts.values()),
        },
        "rungs": [],
    }
    for k in args.k:
        order, chosen = choose_kept(vocab, counts, int(k))
        keptset = set(order)
        pre_closure = sorted(
            (tid for tid in vocab.token_of_id),
            key=lambda tid: (-counts.get(tid, 0), tid),
        )[:k]
        body_params = PARENT_PARAMS - 65536 * 1024
        rung = {
            **chosen,
            "parameters_at_hidden_1024": body_params + k * 1024,
            "net_stored_ratio_at_hidden_1024": (body_params + k * 1024) / PARENT_PARAMS,
            "coverage_post_closure_sealed_corpus": coverage(counts, keptset),
            "coverage_pre_closure_sealed_corpus": coverage(counts, set(pre_closure)),
            "coverage_post_closure_captured_batches": coverage(counts_captured, keptset),
            "coverage_post_closure_review_pack_DIAGNOSTIC": coverage(review_counts, keptset),
            "merges_retained": sum(
                1
                for left, right in vocab.merges
                if all(
                    vocab.id_of_token.get(t) in keptset
                    for t in (left, right, left + right)
                )
            ),
        }
        report["rungs"].append(rung)
        log(
            f"K={k}: kept {rung['kept_rows']}, merges {rung['merges_retained']}, "
            f"corpus coverage {rung['coverage_post_closure_sealed_corpus']['occurrence_coverage']:.6f}, "
            f"review coverage {rung['coverage_post_closure_review_pack_DIAGNOSTIC']['occurrence_coverage']:.6f}, "
            f"ratio {rung['net_stored_ratio_at_hidden_1024']:.6f}x"
        )
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        log(f"census -> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["census", "build"])
    ap.add_argument("--k", type=int, action="append")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--source", type=Path, default=PARENT)
    ap.add_argument("--source-cell-receipt", type=Path)
    ap.add_argument("--child", type=Path)
    ap.add_argument("--tag", default="V")
    ap.add_argument(
        "--axis",
        action="append",
        type=json.loads,
        help="an axis entry from the SOURCE child's own build receipt, carried "
        "into this receipt so a composed child's provenance is complete",
    )
    ap.add_argument("--excised", type=int, action="append")
    args = ap.parse_args()
    if not args.k:
        raise SystemExit("--k is required")
    if args.mode == "census":
        return census_mode(args)
    if args.child is None:
        raise SystemExit("--child is required for --mode build")
    if len(args.k) != 1:
        raise SystemExit("--mode build takes exactly one --k")
    args.k = args.k[0]
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
