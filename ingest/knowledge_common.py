"""ingest.knowledge_common -- helpers shared by the knowledge ingesters (directives, rulings, documents,
datasets, foldhist).  Raw psycopg2 through omni_lab_db; every value is a bound parameter and every
identifier goes through psycopg2.sql.Identifier.

upsert() is the one write primitive: INSERT ... ON CONFLICT (natural key) DO UPDATE ... WHERE the row
actually differs, so a re-run of an unchanged source writes nothing and reports every row 'unchanged'.
That is how the ingesters prove they are idempotent (second run: inserted=0 updated=0).

Inventory snapshots (directives.json, docs.json, foldhist.json) live in inventory/<date>/ in this repo;
OMNI_LAB_INVENTORY or --inventory points elsewhere.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from . import common as C
from .common import db

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INVENTORY = REPO / "inventory" / "2026-10-21"
DATE_RE = re.compile(r"(20\d\d-[01]\d-[0-3]\d)")


def inventory_dir(args) -> Path:
    """--inventory, else $OMNI_LAB_INVENTORY, else the snapshot shipped in this repo."""
    given = getattr(args, "inventory", None) or os.environ.get("OMNI_LAB_INVENTORY", "")
    return Path(given) if given else DEFAULT_INVENTORY


def sha_text(text: str, n: int = 64) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def first_date(text: str | None) -> date | None:
    """First YYYY-MM-DD in a string that is a real calendar date."""
    for m in DATE_RE.finditer(text or ""):
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            continue
    return None


def utc_midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


class Tally:
    """Per-table inserted/updated/unchanged counters, mirrored into the Batch."""

    def __init__(self, batch: C.Batch):
        self.batch = batch
        self.c: Counter = Counter()

    def add(self, table: str, state: str) -> None:
        self.c[(table, state)] += 1
        if state in ("inserted", "updated"):
            self.batch.wrote()

    def lines(self) -> list[str]:
        tables = sorted({t for t, _ in self.c})
        return [f"{t}: inserted={self.c[(t, 'inserted')]} updated={self.c[(t, 'updated')]} "
                f"unchanged={self.c[(t, 'unchanged')]}" for t in tables]


def _adapt(value: Any) -> Any:
    from psycopg2.extras import Json
    if isinstance(value, dict):                     # lists are SQL arrays unless named in jsonb_cols
        return Json(value, dumps=db.canonical_json)
    return value


def upsert(conn, table: str, key_cols: tuple[str, ...], values: dict[str, Any],
           jsonb_cols: tuple[str, ...] = (), id_col: str = "id",
           ignore_in_compare: tuple[str, ...] = ("ingest_batch_id",)) -> tuple[Any, str]:
    """Insert or refresh one row by its natural key.  Returns (id, 'inserted'|'updated'|'unchanged').

    jsonb_cols are always sent as JSON (an empty list stays a JSON array, not a SQL array)."""
    from psycopg2 import sql
    from psycopg2.extras import Json
    cols = list(values)
    params = [Json(values[c], dumps=db.canonical_json) if c in jsonb_cols else _adapt(values[c]) for c in cols]
    upd = [c for c in cols if c not in key_cols]
    cmp_cols = [c for c in upd if c not in ignore_in_compare]
    ident = sql.Identifier
    q = sql.SQL("INSERT INTO {t} ({cols}) VALUES ({vals}) ON CONFLICT ({keys}) DO UPDATE SET {sets} "
                "WHERE ({old}) IS DISTINCT FROM ({new}) RETURNING {id}, (xmax = 0)").format(
        t=ident(table),
        cols=sql.SQL(", ").join(map(ident, cols)),
        vals=sql.SQL(", ").join(sql.Placeholder() * len(cols)),
        keys=sql.SQL(", ").join(map(ident, key_cols)),
        sets=sql.SQL(", ").join(sql.SQL("{c} = EXCLUDED.{c}").format(c=ident(c)) for c in upd),
        old=sql.SQL(", ").join(sql.SQL("{t}.{c}").format(t=ident(table), c=ident(c)) for c in cmp_cols),
        new=sql.SQL(", ").join(sql.SQL("EXCLUDED.{c}").format(c=ident(c)) for c in cmp_cols),
        id=ident(id_col))
    with conn.cursor() as cur:
        cur.execute(q, params)
        row = cur.fetchone()
        if row is not None:
            return row[0], ("inserted" if row[1] else "updated")
        where = sql.SQL(" AND ").join(sql.SQL("{c} = %s").format(c=ident(c)) for c in key_cols)
        cur.execute(sql.SQL("SELECT {id} FROM {t} WHERE ").format(id=ident(id_col), t=ident(table)) + where,
                    [values[c] for c in key_cols])
        return cur.fetchone()[0], "unchanged"


def standalone_main(module, add_args=None) -> int:
    """`python -m ingest.<name> [--inventory DIR]`: one batch through common.run_source, then the tally."""
    ap = argparse.ArgumentParser(prog=f"python -m ingest.{module.SOURCE}", description=module.__doc__.split("\n")[0])
    ap.add_argument("--inventory", help="inventory snapshot directory (default: inventory/2026-10-21 in this repo)")
    if add_args:
        add_args(ap)
    args = ap.parse_args()
    batch = C.run_source(module, args)
    print(f"[{module.SOURCE}] batch {batch.id}: seen={batch.seen} written={batch.written} skipped={batch.skipped}")
    for line in getattr(batch, "tally_lines", []):
        print("  " + line)
    for note in batch.notes[:40]:
        print("  note: " + note)
    return 0


def finish(batch: C.Batch, tally: Tally) -> None:
    """Expose the tally to the caller (standalone_main prints it; omni_lab.py ingest may too)."""
    batch.tally_lines = tally.lines()


def main_guard(module) -> None:
    sys.exit(standalone_main(module))
