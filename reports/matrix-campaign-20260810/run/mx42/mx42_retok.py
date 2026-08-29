#!/usr/bin/env python3
"""Wave 25: the campaign's evidence, re-tokenized into a vocabulary child's own stream.

Wave 23 built five vocabulary children and every one of them was solved against
NOTHING, which is why the axis is clean -- and it is also why the axis has never
been *corrected*.  ``ARCH.V16X13`` reads 5/15 and its failure signature is new:
not the loop-to-the-cap of a dying width child but a word substituted for itself
("Earth's Earth-like Earth"), which is what a model does when the token sequence
under it is no longer the one it was trained on.  That is a statement about the
INPUT DISTRIBUTION, and no solve in this campaign has ever been run on it,
because every capture in the campaign runs on the parent's own id stream.

This file makes the other stream available.  For a child whose ``vocab_size``
differs from the parent's:

* the 24 captured batches are decoded to TEXT with the parent's tokenizer (the
  same decode `mx42_vocab.py` takes its canonical census over);
* the text is re-encoded with the CHILD's tokenizer, giving ``child_ids``;
* ``child_ids`` are injected back into the parent's id space through the child's
  own kept-row map, giving ``parent_ids`` -- the SAME segmentation, named in the
  parent's table.

Feeding ``parent_ids`` to the parent and ``child_ids`` to the child puts the two
models on position-aligned streams over identical text: position *n* in both is
the same prefix, so a target taken from the parent at position *n* is a target
for the child at position *n*.  That is the whole reason this file exists, and
it is the precondition every solve in `mx42_termgain.py` rests on.

**The injection is a gate, not a convenience.**  It is recomputed here from the
two ``tokenizer.json`` files by token STRING (never by id), and the child's
embedding rows are asserted equal, by ``torch.equal``, to the parent's rows at
the injected ids.  A child that fails that assertion is not a row subset of its
parent and nothing downstream of it means what it says.

The fit/held-out split is the campaign's own, unchanged: it is a rule about
BATCH INDEX and re-tokenizing a batch does not move which batch it is.

Usage::

    mx42_retok.py --child <vocab child dir> --out <artifact.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
TRACE = Path(
    "/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810"
    "/evidence/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff"
)
SPLIT_MANIFEST = Path(
    "/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810"
    "/evidence/evidence-manifest.json"
)
SCHEMA = "neural-forge-matrix-wave42-retokenized-evidence-v1"


def log(message: str) -> None:
    print(f"[mx42retok] {message}", flush=True)


def split_by_batch() -> dict[int, str]:
    """The campaign's sealed FIT/held-out split, re-sealed before it is used.

    `mx42_build.split_by_batch` verbatim.  It is duplicated rather than imported
    because this file must run without a GPU and without torch's model stack,
    and the seal check is four lines.
    """

    body = json.loads(SPLIT_MANIFEST.read_text())
    declared = body.get("manifest_sha256")
    recomputed = hashlib.sha256(
        json.dumps(
            {k: v for k, v in body.items() if k != "manifest_sha256"},
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    if declared != recomputed:
        raise SystemExit(f"split manifest seal is broken: {declared} != {recomputed}")
    return {int(row["batch_index"]): str(row["split"]) for row in body["split"]["rows"]}


def parent_batches() -> list[tuple[int, list[int]]]:
    """The 24 captured batches as flat parent-id sequences, in batch order."""

    from safetensors import safe_open

    out: list[tuple[int, list[int]]] = []
    for directory in sorted((TRACE / "batches").iterdir()):
        index = int(directory.name.split("-")[1])
        with safe_open(str(directory / "trace.safetensors"), "pt") as handle:
            prompt = handle.get_tensor("input.input_ids")
            forced = handle.get_tensor("input.forced_decode_ids")
        ids = [int(x) for x in prompt.reshape(-1)] + [int(x) for x in forced.reshape(-1)]
        out.append((index, ids))
    if len(out) != 24:
        raise SystemExit(f"expected 24 captured batches, found {len(out)}")
    return out


def injection(child_dir: Path) -> list[int]:
    """``child id -> parent id``, recomputed from the two vocabularies by STRING.

    A vocabulary child's ids are a DENSE REMAP of a subset of the parent's, and
    the remap is not recorded anywhere as a list -- it is implicit in the two
    ``tokenizer.json`` files.  Recovering it by token string is the only reading
    that cannot be wrong for the interesting reason: an id-order assumption
    ("the kept ids were sorted, so child id i is the i-th kept id") happens to be
    true of `mx42_vocab.py` today and would silently mis-map any child built by
    anything else.
    """

    def table(path: Path) -> dict[str, int]:
        body = json.loads(path.read_text(encoding="utf-8"))
        out = {str(t): int(i) for t, i in body["model"]["vocab"].items()}
        for entry in body.get("added_tokens", []):
            out.setdefault(str(entry["content"]), int(entry["id"]))
        return out

    child_table = table(child_dir / "tokenizer.json")
    parent_table = table(PARENT / "tokenizer.json")
    size = int(json.loads((child_dir / "config.json").read_text())["vocab_size"])
    order = [-1] * size
    for token, cid in child_table.items():
        if cid >= size:
            raise SystemExit(f"child token {token!r} has id {cid} outside its own table")
        pid = parent_table.get(token)
        if pid is None:
            raise SystemExit(
                f"child token {token!r} is absent from the parent vocabulary; this "
                "child is not a row subset of its parent and no injection exists."
            )
        order[cid] = pid
    if -1 in order:
        raise SystemExit(
            f"{order.count(-1)} of {size} child rows carry no token; the child's "
            "table is larger than its own vocabulary and the injection is undefined."
        )
    if len(set(order)) != size:
        raise SystemExit("the injection is not one-to-one")
    return order


def verify_rows_are_the_parents(child_dir: Path, order: list[int]) -> dict[str, Any]:
    """``torch.equal`` on the tied table: every kept row is the parent's bytes.

    Wave 23's build asserted this at build time.  It is asserted AGAIN here,
    from the files on disk, because every solve downstream of this file targets
    the parent's logits through the child's own rows, and if the rows are not the
    parent's then the target is a target for a different model.
    """

    import torch
    from safetensors.torch import load_file

    child = load_file(str(child_dir / "model.safetensors"))["model.embed_tokens.weight"]
    parent = load_file(str(PARENT / "model.safetensors"))["model.embed_tokens.weight"]
    index = torch.tensor(order, dtype=torch.long)
    ok = bool(torch.equal(child, parent.index_select(0, index)))
    if not ok:
        raise SystemExit(
            "the child's token table is NOT the parent's rows at the injected ids; "
            "either the injection or the child is wrong, and both are fatal here."
        )
    return {
        "assertion": "torch.equal(child_embed, parent_embed[injection])",
        "passed": True,
        "child_rows": int(child.shape[0]),
        "hidden": int(child.shape[1]),
    }


def retokenize(child_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The 24 batches on the child's own segmentation, in both id spaces."""

    from transformers import AutoTokenizer

    parent_tok = AutoTokenizer.from_pretrained(str(PARENT))
    child_tok = AutoTokenizer.from_pretrained(str(child_dir))
    order = injection(child_dir)
    identity = verify_rows_are_the_parents(child_dir, order)
    split = split_by_batch()

    rows: list[dict[str, Any]] = []
    for index, ids in parent_batches():
        text = parent_tok.decode(ids, skip_special_tokens=False)
        child_ids = list(child_tok(text, add_special_tokens=False).input_ids)
        # Losslessness is the child tokenizer's own invariant (all 256 ByteLevel
        # rows kept, no unk).  It is re-checked per batch because a stream that
        # does not decode back to its text is not the corpus.
        if child_tok.decode(child_ids, skip_special_tokens=False) != text:
            raise SystemExit(f"batch {index}: the child tokenizer is not lossless on it")
        rows.append(
            {
                "batch_index": index,
                "split": split[index],
                "text": text,
                "parent_ids_original": ids,
                "child_ids": child_ids,
                "parent_ids": [order[i] for i in child_ids],
            }
        )
    bytes_total = sum(len(r["text"].encode("utf-8")) for r in rows)
    parent_positions = sum(len(r["parent_ids_original"]) for r in rows)
    child_positions = sum(len(r["child_ids"]) for r in rows)
    meta = {
        "schema": SCHEMA,
        "child_model": str(child_dir),
        "parent_model": str(PARENT),
        "trace": str(TRACE.name),
        "identity_gate": identity,
        "batches": len(rows),
        "utf8_bytes": bytes_total,
        "parent_stream_positions": parent_positions,
        "child_stream_positions": child_positions,
        "sequence_length_ratio": child_positions / parent_positions,
        "fit_positions": sum(len(r["child_ids"]) for r in rows if r["split"] == "fit"),
        "heldout_positions": sum(
            len(r["child_ids"]) for r in rows if r["split"] != "fit"
        ),
        "note": (
            "the parent runs on `parent_ids` and the child on `child_ids`; the two "
            "are the SAME segmentation of the same text named in the two tables, so "
            "position n is the same prefix in both streams and a target taken from "
            "the parent at n is a target for the child at n"
        ),
    }
    return rows, meta


def stream_artifact(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """The retokenized stream in the form a SOLVER can consume, and its digest.

    This is wave 30's half of RETOK-BODY and it is the whole reason the plan's
    #2 sat unexecuted for two waves.  Wave 29's ``mx29_retok.py`` computed
    ``parent_ids`` correctly and then wrote only the *metadata* to its artifact:
    the positions were counted, the identity gate passed, the sequence-length
    ratio was reported -- and the ids themselves were dropped on the floor when
    the process exited.  Nothing downstream could be solved against a stream
    that was never written down, so the "one new capture" the plan costs was
    not runnable at any price.

    The stream is written with its own ``stream_sha256`` over the ids alone, so
    a build receipt that names this evidence names bytes and not a filename.
    ``parent_ids`` is what a capture feeds the PARENT: same segmentation as the
    child's, named in the parent's table.  Feeding it to the parent is exactly
    equivalent to feeding ``child_ids`` to a parent-body-plus-child-table hybrid
    -- the identity gate above proves the two tables agree row for row on the
    injected ids -- which is why this wave captures through the parent itself
    and materializes no hybrid model.
    """

    stream = [
        {
            "batch_index": r["batch_index"],
            "split": r["split"],
            "parent_ids": r["parent_ids"],
            "child_ids": r["child_ids"],
            "parent_ids_original": r["parent_ids_original"],
        }
        for r in rows
    ]
    digest = hashlib.sha256(
        json.dumps(stream, allow_nan=False, ensure_ascii=True,
                   separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()
    body = dict(meta)
    body["stream_sha256"] = digest
    body["stream"] = stream
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--meta-only",
        action="store_true",
        help=(
            "write wave 29's metadata-only artifact.  Kept so the wave-29 read "
            "can be reproduced exactly; it is NOT solvable evidence and "
            "`mx42_build.py --evidence` refuses it"
        ),
    )
    args = ap.parse_args()
    rows, meta = retokenize(args.child)
    log(
        f"{meta['batches']} batches: {meta['parent_stream_positions']} parent "
        f"positions -> {meta['child_stream_positions']} child positions "
        f"({meta['sequence_length_ratio']:.4f}x), fit {meta['fit_positions']} / "
        f"held-out {meta['heldout_positions']}"
    )
    body = meta if args.meta_only else stream_artifact(rows, meta)
    if not args.meta_only:
        log(f"stream sha256 {body['stream_sha256']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(body, indent=1))
        log(f"-> {args.out}")
    else:
        print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
