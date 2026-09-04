"""The sealed evidence manifest every solve in this lane is fitted against.

LFM2's ``corpus()`` read 24 directories of ``trace.safetensors`` off a path
hardcoded three levels up the file, and its fit/held split came from a second
hardcoded manifest.  Both are gone.  Here the evidence is ONE json file that
names its tokenizer, its sequence length and every row's split, and that is
RE-SEALED on read -- because an evidence artifact that only names its own digest
is not evidence, and this file decides what every ridge in this lane is solved
against.

Schema ``neural-forge-omni-evidence-v1``::

    {"schema": "neural-forge-omni-evidence-v1",
     "tokenizer_sha256": "...",           # the tokenizer.json the ids came from
     "sequence_length": 264,              # every row is exactly this long
     "source": "free text: where the text came from",
     "rows": [{"batch_index": 0, "split": "fit", "ids": [...]}, ...],
     "manifest_sha256": "..."}            # over the body minus this field
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from qf_arch import canonical_json_bytes

SCHEMA = "neural-forge-omni-evidence-v1"

#: The internal fit/check split the lambda ladder is scored on: every third fit
#: batch from the third.  Carried verbatim from mx45 (``INTERNAL_CHECK_STRIDE``
#: 3, ``OFFSET`` 2) so a rung chosen here means what it meant there.
INTERNAL_CHECK_STRIDE = 3
INTERNAL_CHECK_OFFSET = 2


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _body_digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes({k: v for k, v in body.items() if k != "manifest_sha256"})
    ).hexdigest()


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read and RE-SEAL the manifest.  A broken seal is a refusal, not a warning."""

    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {body.get('schema')!r}, expected {SCHEMA!r}")
    declared = body.get("manifest_sha256")
    recomputed = _body_digest(body)
    if declared != recomputed:
        raise SystemExit(f"evidence manifest seal is broken: {declared} != {recomputed}")
    lengths = {len(row["ids"]) for row in body["rows"]}
    if lengths != {int(body["sequence_length"])}:
        raise SystemExit(
            f"{path}: rows are {sorted(lengths)} tokens long, manifest declares "
            f"{body['sequence_length']}"
        )
    splits = {row["split"] for row in body["rows"]}
    if not splits <= {"fit", "held"}:
        raise SystemExit(f"{path}: unknown splits {sorted(splits - {'fit', 'held'})}")
    if "fit" not in splits:
        raise SystemExit(f"{path}: no fit rows -- nothing to solve on")
    return body


def write_manifest(
    rows: list[dict[str, Any]],
    *,
    tokenizer_sha256: str,
    source: str,
    out: str | Path,
) -> dict[str, Any]:
    """Seal a manifest and parse it back before returning it."""

    lengths = {len(row["ids"]) for row in rows}
    if len(lengths) != 1:
        raise SystemExit(f"rows are ragged: {sorted(lengths)}")
    body = {
        "schema": SCHEMA,
        "tokenizer_sha256": tokenizer_sha256,
        "sequence_length": lengths.pop(),
        "source": source,
        "rows": [
            {"batch_index": int(row["batch_index"]), "split": str(row["split"]),
             "ids": [int(i) for i in row["ids"]]}
            for row in sorted(rows, key=lambda r: int(r["batch_index"]))
        ],
    }
    body["manifest_sha256"] = _body_digest(body)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written through a sibling temporary and RENAMED into place.  A plain
    # ``write_text`` is a truncate-then-write: a timeout, a kill or a full disk
    # firing between the two leaves a half-written file where the sealed
    # evidence was, and the good copy is already gone.  ``os.replace`` is atomic
    # within a filesystem, so the file at ``out`` is either the previous
    # manifest or this one, never a fragment of either.
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(body, indent=2), encoding="utf-8")
    os.replace(temporary, out)
    return read_manifest(out)


def corpus(manifest: dict[str, Any]) -> list[tuple[int, torch.Tensor]]:
    """``[(batch_index, ids[1, L])]`` in batch-index order."""

    return [
        (int(row["batch_index"]), torch.tensor([row["ids"]], dtype=torch.long))
        for row in sorted(manifest["rows"], key=lambda r: int(r["batch_index"]))
    ]


def split_by_batch(manifest: dict[str, Any]) -> dict[int, str]:
    return {int(row["batch_index"]): str(row["split"]) for row in manifest["rows"]}


def fit_check_rows(manifest: dict[str, Any]) -> torch.Tensor:
    """Boolean mask over the CONCATENATED fit rows: the internal check split.

    Rows, not batches, because that is what the solves index.  Every fit batch is
    ``sequence_length`` rows long and the mask marks every third fit batch from
    the third, whole.
    """

    length = int(manifest["sequence_length"])
    fit_batches = [
        int(row["batch_index"])
        for row in sorted(manifest["rows"], key=lambda r: int(r["batch_index"]))
        if row["split"] == "fit"
    ]
    mask = torch.zeros(len(fit_batches) * length, dtype=torch.bool)
    for position in range(len(fit_batches)):
        if position >= INTERNAL_CHECK_OFFSET and (position - INTERNAL_CHECK_OFFSET) % INTERNAL_CHECK_STRIDE == 0:
            mask[position * length : (position + 1) * length] = True
    if bool(mask.all()) or not bool(mask.any()):
        raise SystemExit(
            f"internal check split is degenerate: {int(mask.sum())} of {mask.numel()} rows"
        )
    return mask


def evidence_receipt(manifest: dict[str, Any], manifest_path: str | Path) -> dict[str, Any]:
    """What every receipt this lane writes says about the stream it was fitted on."""

    rows = manifest["rows"]
    fit = [row for row in rows if row["split"] == "fit"]
    held = [row for row in rows if row["split"] == "held"]
    length = int(manifest["sequence_length"])
    return {
        "schema": manifest["schema"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "source": manifest["source"],
        "sequence_length": length,
        "batches": len(rows),
        "fit_batches": len(fit),
        "held_batches": len(held),
        "fit_rows": len(fit) * length,
        "heldout_rows": len(held) * length,
        "internal_check_rows": int(fit_check_rows(manifest).sum()),
    }


def synthetic_manifest(
    *,
    vocab_size: int,
    sequence_length: int,
    batches: int,
    held_every: int,
    out: str | Path,
    seed: int = 20260901,
    reserved: int = 8,
    max_id: int | None = None,
    tokenizer_sha256: str | None = None,
) -> dict[str, Any]:
    """A deterministic id stream for the tiny-model self-tests.

    Not a corpus and it does not pretend to be one: it exercises the mechanics
    (row counts, splits, conditioning, the solves' conditioning number) on a
    random-init model where there is no text to be faithful to.  Ids avoid the
    first ``reserved`` rows so a vocabulary fold's mandatory-token logic is not
    accidentally satisfied by the evidence.

    ``tokenizer_sha256`` names the tokenizer.json the fixture actually carries,
    so a synthetic manifest passes the same consumption-time tokenizer check
    (:func:`qf_build.assert_evidence_tokenizer`) a real one does; left unset it
    falls back to a self-describing ``synthetic:`` label that no parent on disk
    can match.

    ``max_id`` bounds the stream below the table's width.  The C1 self-test sets
    it to the byte alphabet so the stream is spellable by ANY coreset, because
    the alternative -- a stream containing tokens the child deleted -- is not a
    tolerance question, it is a re-tokenization, and that is a different unit.
    """

    ceiling = int(max_id or vocab_size)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for index in range(batches):
        ids = torch.randint(
            reserved, ceiling, (sequence_length,), generator=generator
        ).tolist()
        rows.append(
            {
                "batch_index": index,
                "split": "held" if index % held_every == held_every - 1 else "fit",
                "ids": ids,
            }
        )
    return write_manifest(
        rows,
        tokenizer_sha256=(
            tokenizer_sha256
            or "synthetic:" + hashlib.sha256(f"{vocab_size}:{seed}".encode()).hexdigest()
        ),
        source=f"qf_corpus.synthetic_manifest seed={seed} vocab={vocab_size}",
        out=out,
    )
