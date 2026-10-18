#!/usr/bin/env python3
"""The Chimera Omni campaign ledger: append-only, canonical, single-writer.

Compatible in spirit -- and in bytes where it can be -- with the LFM2 matrix
ledger that carried 1,056 rows to sha256 `d2027831...`:

* a row is `canonical_json_bytes(line) + b"\\n"`, appended in BINARY mode.
  Text mode on a Windows workstation once appended CRLF to the mirror of an
  append-only file, so two copies of the same rows diverged byte for byte
  while `read_ledger` -- which splits on either -- saw nothing;
* rows are never rewritten.  A correction is a NEW row for a NEW record
  instance, exactly as a re-reviewed LFM2 cell is a new `-a6review` instance;
* a merge is a byte concatenation of shard lines, never a re-serialisation.
  It is idempotent: a row whose `(instance, receipt_sha)` is already in the
  authority is skipped, and a shard row that reuses an authority instance
  under a different receipt is REFUSED.  `--apply` OPENS THE AUTHORITY IN
  'ab' AND WRITES ONLY THE NEW LINES -- it never rewrites the file, because
  a rewrite truncates to zero first and a kill in that window would lose
  every prior row.  After `--apply` the file is re-read and its first
  `len(before)` bytes must still equal the bytes that were there -- a check
  on the disk, not on a string this function just built.

What is new for this campaign, and why:

1. **Two row schemas.**  The ported LFM2 sealers keep emitting
   `neural-forge-lfm2-matrix-cell-ledger-v1` rows for ARCH children and
   behaviour verdicts; the checkpoint registry emits
   `neural-forge-omni-checkpoint-ledger-v1` rows for register / clone / review
   / status / delete events.  Both live in one authority file because the
   campaign has one timeline, and both carry `campaign` = "omni-2026-09".  The
   LFM2 rows of the ancestor campaign never carry that column: that is how a
   reader tells the two campaigns apart without a second file.
2. **A byte check in `verify`.**  Every line must equal
   `canonical_json_bytes(json.loads(line))`.  A hand-edit that reformats a row
   is then a verify failure rather than an invisible fork between the server
   authority and the repo mirror -- the "repo-ledger-mirror copy-back" defect
   recurred twice in the LFM2 campaign (waves 25 and 32).
3. **A writer lock** around every append and every `--apply`, because this
   campaign has a live CLI appending rows and not only a once-per-wave merge.

Usage::

    oc_ledger.py append  --ledger L --row row.json
    oc_ledger.py merge   --authority L --shard S [--shard S...] --report R [--apply]
    oc_ledger.py verify  --ledger L [--registry DIR]
    oc_ledger.py digest  --ledger L
    oc_ledger.py rows    --ledger L [--checkpoint-id ID] [--schema S] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from oc_seal import (
    CAMPAIGN,
    OcSealError,
    WriterLock,
    canonical_json_bytes,
    file_sha256,
    iter_json_lines,
    read_sealed,
)

#: The ancestor campaign's row schema, emitted by the ported ARCH/behaviour
#: sealers.  Passed through verbatim; this tool never writes one.
CELL_ROW_SCHEMA = "neural-forge-lfm2-matrix-cell-ledger-v1"

#: This campaign's own row schema, written by `oc_checkpoint.py`.
CHECKPOINT_ROW_SCHEMA = "neural-forge-omni-checkpoint-ledger-v1"

ACCEPTED_ROW_SCHEMAS = (CELL_ROW_SCHEMA, CHECKPOINT_ROW_SCHEMA)

#: Every column a checkpoint row carries, in the order a reader wants them.
#: The list is asserted at write time: a row with an unknown column or a
#: missing one is refused, so a column added mid-campaign is a deliberate
#: schema bump and not a shape that drifts row by row.
CHECKPOINT_ROW_COLUMNS = (
    "schema",
    "campaign",
    "checkpoint_id",
    "record_instance_id",
    "record_seq",
    "event",
    "stage",
    "parent_checkpoint_id",
    "clone_source_id",
    "parent_code",
    "tag",
    "wave",
    "directory",
    "directory_sha256",
    "parameter_count",
    "parameter_count_source",
    "parent_parameter_count",
    "parent_parameter_count_source",
    "net_ratio",
    "quantization",
    "recipe_sha256",
    "status",
    "status_reason",
    "sealed_useful_or_partial",
    "parent_useful_or_partial",
    "sealed_verdict",
    "canonical_signal",
    "column_a",
    "column_b",
    "bank2_sealed_useful_or_partial",
    "bank2_canonical_signal",
    "media_out_artifacts",
    "media_out_scored",
    "package_path",
    "package_sha256",
    "dossier_sha256",
    "verdict_receipt_sha",
    "notes",
    "sealed_at",
    "receipt_sha",
)


class OcLedgerError(RuntimeError):
    """A ledger operation refused."""


def validate_row(
    row: Mapping[str, Any], *, where: str, require_campaign: bool = False
) -> dict[str, Any]:
    """Fail closed on anything that is not a row of an accepted schema.

    `require_campaign` is set on the WRITE paths (append, and shard ingestion in
    `merge`) and left off on the read paths.  R9 says every row of this campaign
    carries `campaign = "omni-2026-09"`, but that was only ever enforced on
    checkpoint rows: a ported ARCH/behaviour sealer emitting a cell row for THIS
    campaign without the column would have been accepted and then been
    indistinguishable from one of the ancestor campaign's 1,056 rows.  Reading
    stays permissive because the ancestor rows legitimately have no `campaign`
    and this tool must still parse them.
    """

    if not isinstance(row, Mapping):
        raise OcLedgerError(f"{where}: row is not an object")
    schema = row.get("schema")
    if schema not in ACCEPTED_ROW_SCHEMAS:
        raise OcLedgerError(
            f"{where}: schema {schema!r} is not one of {ACCEPTED_ROW_SCHEMAS}"
        )
    # Both schemas: `verify` and `merge` index these two by name.  A row without
    # them used to come out of either command as a KeyError traceback instead of
    # the `REFUSED: <why>` exit 3 the CLI contract promises.
    for column in ("receipt_sha", "sealed_at"):
        if column not in row:
            raise OcLedgerError(
                f"{where}: row has no {column!r}; every row of either schema "
                "carries one, and the ledger is indexed by it"
            )
        if not isinstance(row[column], str) or not row[column]:
            raise OcLedgerError(
                f"{where}: {column} is {row[column]!r}, not a non-empty string"
            )
    if require_campaign and row.get("campaign") != CAMPAIGN:
        raise OcLedgerError(
            f"{where}: campaign {row.get('campaign')!r} is not {CAMPAIGN!r}; a row "
            "written into this campaign's authority says which campaign it "
            "belongs to, or it cannot be told from an ancestor row"
        )
    if schema == CHECKPOINT_ROW_SCHEMA:
        missing = [c for c in CHECKPOINT_ROW_COLUMNS if c not in row]
        extra = [c for c in row if c not in CHECKPOINT_ROW_COLUMNS]
        if missing or extra:
            raise OcLedgerError(
                f"{where}: checkpoint row columns drifted -- missing {missing}, "
                f"unknown {extra}"
            )
        if row.get("campaign") != CAMPAIGN:
            raise OcLedgerError(
                f"{where}: campaign {row.get('campaign')!r} is not {CAMPAIGN!r}"
            )
    return dict(row)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read every row, refusing non-canonical bytes and unknown schemas."""

    rows: list[dict[str, Any]] = []
    for number, raw in iter_json_lines(path):
        where = f"{path}:{number}"
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OcLedgerError(f"{where} is not JSON") from exc
        validate_row(row, where=where)
        if canonical_json_bytes(row) != raw:
            raise OcLedgerError(
                f"{where} is not canonical JSON bytes -- the row was edited or "
                "re-serialised after it was appended"
            )
        rows.append(row)
    return rows


def append_rows(
    ledger: str | Path, rows: Iterable[Mapping[str, Any]], *, lock_timeout_s: float = 60.0
) -> int:
    """Append validated rows in binary mode under the writer lock."""

    payload = b""
    count = 0
    for row in rows:
        validate_row(row, where="append", require_campaign=True)
        payload += canonical_json_bytes(dict(row)) + b"\n"
        count += 1
    if not payload:
        return 0
    destination = Path(ledger).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with WriterLock(destination.with_suffix(destination.suffix + ".lock"),
                    timeout_s=lock_timeout_s):
        before = destination.read_bytes() if destination.is_file() else b""
        if before and not before.endswith(b"\n"):
            raise OcLedgerError(
                f"{destination}: the last row has no trailing newline; appending "
                "would splice two rows onto one line.  Inspect the tail by hand."
            )
        with destination.open("ab") as handle:
            handle.write(payload)
        after = destination.read_bytes()
        if not after.startswith(before) or not after.endswith(payload):
            raise OcLedgerError(
                f"{destination}: the append did not land as a suffix of the "
                "previous bytes; the ledger is damaged and must be inspected"
            )
    return count


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_append(args: argparse.Namespace) -> int:
    rows = []
    for path in args.row:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(body if isinstance(body, list) else [body])
    count = append_rows(args.ledger, rows)
    print(f"appended {count} row(s) to {args.ledger}")
    print(f"ledger sha256 {file_sha256(args.ledger)}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge shards into the authority exactly once, and only once.

    Same contract as `mx45_merge_ledger.py`, generalised to the two accepted
    schemas: rows are grouped by their instance id (`cell_instance_id` for
    ancestor rows, `record_instance_id` for checkpoint rows), deduplicated by
    `receipt_sha`, and a group that still has more than one distinct receipt is
    kept only if exactly one of them is named by a sealed verdict's
    `addendum6.reviewed_from_receipt_sha`.  Otherwise the merge REFUSES; it
    does not guess which measurement is the real one.

    TWO THINGS THE ANCESTOR DID NOT DO, both of them defects it got away with
    because it had no `verify`:

    * **dedup against the AUTHORITY, not only within the shard set.**  Running
      the same merge twice used to append every row a second time, and since
      rows are never rewritten the duplicate was permanent -- and this
      campaign's `verify` reports a duplicate `(instance, receipt_sha)` as a
      PROBLEM and exits 1, so a re-run poisoned the file for good.  A row whose
      `(instance, receipt_sha)` is already in the authority is SKIPPED and
      counted in the report; a merge is therefore idempotent.
    * **a real prefix check.**  `result = before + payload` then
      `result.startswith(before)` is a tautology and proved nothing.  What is
      actually checked now: the authority ends on a row boundary (otherwise the
      append splices two rows onto one line), a shard row may not reuse an
      instance already in the authority under a DIFFERENT receipt (a correction
      is a new record instance, never a second receipt for the same one), and
      after `--apply` the bytes on disk are re-read and their first
      `len(before)` bytes must still equal `before`.
    """

    authority = Path(args.authority).expanduser().absolute()
    before = authority.read_bytes() if authority.is_file() else b""
    before_rows = [line for line in before.split(b"\n") if line.strip()]
    if before and not before.endswith(b"\n"):
        raise OcLedgerError(
            f"{authority}: the last row has no trailing newline; a merge would "
            "splice two rows onto one line.  Inspect the tail by hand."
        )
    # (instance, receipt_sha) -> row number, so a re-run of the same merge is a
    # no-op instead of a permanent duplicate.
    already: dict[tuple[str, str], int] = {}
    for index, row in enumerate(read_rows(authority), start=1):
        instance = str(row.get("record_instance_id") or row.get("cell_instance_id"))
        already[(instance, str(row["receipt_sha"]))] = index

    reviewed_from: set[str] = set()
    for path in args.verdict_receipt:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = (body.get("addendum6") or {}).get("reviewed_from_receipt_sha")
        if digest:
            reviewed_from.add(str(digest))

    groups: dict[str, dict[str, tuple[str, bytes]]] = {}
    for shard in args.shard:
        for number, raw in iter_json_lines(shard):
            row = json.loads(raw.decode("utf-8"))
            validate_row(row, where=f"{shard}:{number}", require_campaign=True)
            if canonical_json_bytes(row) != raw:
                raise OcLedgerError(f"{shard}:{number} is not canonical JSON bytes")
            instance = str(
                row.get("record_instance_id") or row.get("cell_instance_id")
            )
            groups.setdefault(instance, {})[str(row["receipt_sha"])] = (
                str(row["sealed_at"]),
                raw,
            )

    kept: list[tuple[str, bytes]] = []
    superseded: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for instance, by_digest in sorted(groups.items()):
        # already in the authority, byte for byte: the merge has run before.
        for digest in sorted(d for d in by_digest if (instance, d) in already):
            skipped.append(
                {
                    "instance": instance,
                    "receipt_sha": digest,
                    "authority_row": str(already[(instance, digest)]),
                }
            )
            by_digest.pop(digest)
        if not by_digest:
            continue
        authority_digests = sorted(
            d for (i, d) in already if i == instance
        )
        if authority_digests:
            raise OcLedgerError(
                f"REFUSED: {instance} is already in the authority under receipt(s) "
                f"{[d[:12] for d in authority_digests]} and the shard carries "
                f"{[d[:12] for d in sorted(by_digest)]}.  Rows are never rewritten, "
                "so a correction is a NEW record instance -- never a second "
                "receipt for the same one."
            )
        if len(by_digest) == 1:
            kept.append(next(iter(by_digest.values())))
            continue
        pointed = [d for d in by_digest if d in reviewed_from]
        if len(pointed) != 1:
            raise OcLedgerError(
                f"REFUSED: {instance} has {len(by_digest)} distinct receipts "
                f"{sorted(d[:12] for d in by_digest)} and {len(pointed)} of them "
                "are named by a sealed verdict; a merge may not guess which "
                "measurement is the real one."
            )
        kept.append(by_digest[pointed[0]])
        superseded.extend(
            {"instance": instance, "receipt_sha": d}
            for d in sorted(by_digest)
            if d != pointed[0]
        )

    kept.sort(key=lambda pair: (pair[0], pair[1]))
    payload = b"".join(raw + b"\n" for _, raw in kept)

    report = {
        "schema": "neural-forge-omni-ledger-merge-v1",
        "campaign": CAMPAIGN,
        "authority": str(authority),
        "shards": [str(s) for s in args.shard],
        "rows_before": len(before_rows),
        "rows_appended": len(kept),
        "rows_after": len(before_rows) + len(kept),
        "rows_skipped_already_present": skipped,
        "superseded_rows_dropped": superseded,
        "appended_instances": sorted(
            instance for instance, by_digest in groups.items() if by_digest
        ),
        "applied": bool(args.apply),
    }
    if args.apply and payload:
        with WriterLock(
            authority.with_suffix(authority.suffix + ".lock"), timeout_s=60.0
        ):
            live = authority.read_bytes() if authority.is_file() else b""
            if live != before:
                raise OcLedgerError(
                    "refused: the authority changed between the plan and the "
                    "apply; re-run the dry run"
                )
            # APPEND, never rewrite.  `write_bytes` opens 'wb', which truncates
            # the authority to zero before the first byte lands: a kill or an
            # ENOSPC in that window loses every prior row, and the prefix checks
            # below would then only be able to REPORT the loss.  Opening 'ab'
            # makes the prefix invariant structural instead of asserted.
            with authority.open("ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            after = authority.read_bytes()
        if after[: len(before)] != before:
            raise OcLedgerError(
                "the authority's existing BYTES are no longer a prefix of the "
                "file after the write; the ledger is damaged and must be "
                "inspected"
            )
        after_rows = [line for line in after.split(b"\n") if line.strip()]
        if after_rows[: len(before_rows)] != before_rows:
            raise OcLedgerError("the prefix changed after the write; ledger damaged")
        report["rows_after_verified"] = len(after_rows)
        report["ledger_sha256"] = file_sha256(authority)
    elif args.apply:
        # every shard row was already in the authority: the merge has run
        # before.  Nothing is written, so the file stays byte-identical.
        report["rows_after_verified"] = len(before_rows)
        report["ledger_sha256"] = (
            file_sha256(authority) if authority.is_file() else None
        )
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    reread = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if reread["rows_after"] != report["rows_after"]:
        raise OcLedgerError(f"{args.report} does not read back as written")
    print(json.dumps(report, indent=1))
    if not args.apply:
        print("\nDRY RUN: the authority was not written.  Re-run with --apply.")
    elif not payload:
        print(
            f"\nNOTHING TO DO: all {len(skipped)} shard row(s) are already in the "
            "authority; the file was not touched."
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-read every row; resolve checkpoint rows to their sealed records."""

    rows = read_rows(args.ledger)
    resolved = verified = failed = unresolved = 0
    problems: list[str] = []

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows, start=1):
        instance = str(row.get("record_instance_id") or row.get("cell_instance_id"))
        key = (instance, str(row["receipt_sha"]))
        if key in seen:
            problems.append(
                f"row {index}: duplicate ({instance}, {key[1][:12]}) also at row "
                f"{seen[key]}"
            )
        seen[key] = index

    registry = Path(args.registry).expanduser().absolute() if args.registry else None
    for index, row in enumerate(rows, start=1):
        if row["schema"] != CHECKPOINT_ROW_SCHEMA or registry is None:
            unresolved += 1
            continue
        record = registry / str(row["checkpoint_id"]) / f"{row['record_instance_id']}.json"
        if not record.is_file():
            unresolved += 1
            problems.append(f"row {index}: no record at {record}")
            continue
        resolved += 1
        try:
            body = read_sealed(record)
        except OcSealError as exc:
            failed += 1
            problems.append(f"row {index}: {exc}")
            continue
        if body["receipt_sha"] != row["receipt_sha"]:
            failed += 1
            problems.append(
                f"row {index}: receipt_sha {row['receipt_sha'][:12]} != record "
                f"{body['receipt_sha'][:12]}"
            )
            continue
        verified += 1

    print(f"rows            {len(rows)}")
    print(f"schemas         {sorted({r['schema'] for r in rows})}")
    print(f"resolve/verify  {resolved}/{verified}")
    print(f"fail            {failed}")
    print(f"do-not-resolve  {unresolved}")
    print(f"ledger sha256   {file_sha256(args.ledger) if Path(args.ledger).is_file() else '-'}")
    for line in problems:
        print(f"  PROBLEM {line}")
    return 1 if (failed or problems) else 0


def cmd_digest(args: argparse.Namespace) -> int:
    path = Path(args.ledger)
    rows = read_rows(path)
    print(json.dumps(
        {
            "ledger": str(path),
            "rows": len(rows),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": file_sha256(path) if path.is_file() else None,
        },
        indent=1,
    ))
    return 0


def cmd_rows(args: argparse.Namespace) -> int:
    rows = read_rows(args.ledger)
    if args.checkpoint_id:
        rows = [r for r in rows if r.get("checkpoint_id") == args.checkpoint_id]
    if args.schema:
        rows = [r for r in rows if r["schema"] == args.schema]
    if args.limit:
        rows = rows[-args.limit:]
    for row in rows:
        print(canonical_json_bytes(row).decode("ascii"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("append", help="append validated rows (single writer)")
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--row", type=Path, action="append", required=True)
    p.set_defaults(func=cmd_append)

    p = sub.add_parser("merge", help="merge shards into the authority once")
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--shard", type=Path, action="append", required=True)
    p.add_argument("--verdict-receipt", type=Path, action="append", default=[])
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("verify", help="re-read and re-verify every row")
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--registry", type=Path, default=None)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("digest", help="row count and sha256 of the ledger file")
    p.add_argument("--ledger", type=Path, required=True)
    p.set_defaults(func=cmd_digest)

    p = sub.add_parser("rows", help="print rows, canonically")
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--checkpoint-id", default=None)
    p.add_argument("--schema", default=None, choices=list(ACCEPTED_ROW_SCHEMAS))
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_rows)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OcLedgerError, OcSealError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
