#!/usr/bin/env python3
"""omni_lab -- control CLI for the Omni Lab database (no torch; psycopg2 only).

    omni_lab.py db migrate [--dry-run]     apply pending migrations/NNNN_*.sql as omni_lab
    omni_lab.py db check                   schema version, migration file parity, row counts, last batch per source
    omni_lab.py last [N]                   newest N rows of v_results (one line per eval run)
    omni_lab.py sql "SELECT ..."           read-only query as omni_lab_ro

Ingest, run, compare, grade and log-run subcommands are added by later steps of the build; they
use the same omni_lab_db helpers.  Connection settings: see omni_lab_db.py docstring and README.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omni_lab_db as db  # noqa: E402


def print_table(headers: list[str], rows: list[tuple], max_width: int = 60) -> None:
    def cell(v) -> str:
        s = "-" if v is None else str(v)
        s = s.replace("\n", " ")
        return s if len(s) <= max_width else s[: max_width - 1] + "~"
    cells = [[cell(v) for v in r] for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))


def cmd_db(args) -> int:
    conn = db.connect("rw", app=f"db-{args.action}")
    try:
        if args.action == "migrate":
            applied = db.migrate(conn, dry_run=args.dry_run)
            verb = "would apply" if args.dry_run else "applied"
            print(f"{verb}: {[f'{v:04d}_{n}' for v, n in applied] or 'nothing pending'}")
            print(f"schema version: {db.schema_version(conn)}")
            return 0
        # check
        files = db.migration_files()
        done = db.applied_migrations(conn)
        bad = 0
        print(f"schema version: {db.schema_version(conn)}  (files on disk: {len(files)})")
        for version, name, _path, sha in files:
            if version not in done:
                state = "PENDING"
                bad += 1
            elif done[version][1] != sha:
                state = "SHA-MISMATCH"
                bad += 1
            else:
                state = "ok"
            print(f"  {version:04d}_{name:<22} {sha[:16]}  {state}")
        print()
        print_table(["table", "rows"], db.table_counts(conn))
        with conn.cursor() as cur:
            cur.execute("SELECT source, max(started_at), count(*), "
                        "count(*) FILTER (WHERE status <> 'ok') FROM ingest_batch GROUP BY source ORDER BY source")
            rows = cur.fetchall()
        if rows:
            print()
            print_table(["source", "last batch", "batches", "not ok"], rows)
        conn.rollback()
        return 1 if bad else 0
    finally:
        conn.close()


def cmd_last(args) -> int:
    conn = db.connect("ro", app="last")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT eval_run_id, test, model_key, suite, engine, devices, correct, partial, miss, \"none\", "
                        "pass, fail, ungraded, n_results, n_cases, started_at::date FROM v_results "
                        "ORDER BY started_at DESC NULLS LAST, eval_run_id DESC LIMIT %s", (args.n,))
            rows = cur.fetchall()
        print_table(["run", "test", "model", "suite", "engine", "devices", "C", "P", "M", "N", "pass", "fail",
                     "ungr", "n", "of", "date"], rows, max_width=40)
        return 0
    finally:
        conn.close()


def cmd_sql(args) -> int:
    conn = db.connect("ro", app="sql")
    try:
        with conn.cursor() as cur:
            cur.execute(args.query)
            if cur.description is None:
                print("(no result set)")
                return 0
            headers = [d[0] for d in cur.description]
            rows = cur.fetchmany(args.limit)
        print_table(headers, rows, max_width=args.width)
        if len(rows) == args.limit:
            print(f"(first {args.limit} rows)")
        return 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="omni_lab.py", description="Omni Lab database CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("db", help="migrate / check the schema")
    p.add_argument("action", choices=("migrate", "check"))
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_db)
    p = sub.add_parser("last", help="newest rows of v_results")
    p.add_argument("n", nargs="?", type=int, default=20)
    p.set_defaults(func=cmd_last)
    p = sub.add_parser("sql", help="read-only SQL as omni_lab_ro")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--width", type=int, default=60)
    p.set_defaults(func=cmd_sql)
    args = ap.parse_args(argv)
    import psycopg2
    try:
        return args.func(args)
    except db.LabError as exc:
        print(f"[refused] {exc}", file=sys.stderr)
        return 2
    except psycopg2.Error as exc:
        print(f"[db error] {type(exc).__name__}: {str(exc).strip()}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
