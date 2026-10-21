#!/usr/bin/env python3
"""omni_lab -- control CLI for the Omni Lab database (no torch; psycopg2 only).

    omni_lab.py db migrate [--dry-run]     apply pending migrations/NNNN_*.sql as omni_lab
    omni_lab.py db check                   schema version, migration file parity, row counts, last batch per source
    omni_lab.py last [N]                   newest N rows of v_results (one line per eval run)
    omni_lab.py sql "SELECT ..."           read-only query as omni_lab_ro
    omni_lab.py ingest <source>... | models-runs | evals [--keep-going]
                                           run ingest/<source>.py (one ingest_batch each); idempotent
    omni_lab.py run --model M --suite S [--gpus 6,4,0 | --cpu] [--cases ...] [--cap N] [--save-test T]
                                           claim cards -> GPU worker -> results to console + database
    omni_lab.py run --test T --model M      re-run a saved test (suite + engine + gen + cards + case filter)
    omni_lab.py compare [RUN...] [--suite S] [--model M]...   runs x cases pivot ('-' = never ran that case)
    omni_lab.py list runs|suites|models|tests|cases <suite>
    omni_lab.py add-suite NAME --file F     a bank/task file -> a new suite
    omni_lab.py add-cases SUITE --file F    new cases -> next suite version (older runs show '-' for them)
    omni_lab.py grade RUN CASE GRADE [--why ...] | grade RUN --file grades.json   reader grades
    omni_lab.py log-run start|end|metric ... rows for training / fold / cache jobs
    omni_lab.py arch-check --report R.json [--model KEY]
                                           chimera_omni.conformance report -> tensor census + arch_check rows

run/compare/list/add-*/grade/log-run live in omni_lab_run.py; the GPU side is omni_lab_worker.py.
Connection settings: see omni_lab_db.py docstring and README.md.
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


def cmd_ingest(args) -> int:
    """Run ingesters (one ingest_batch each, in the given/group order) and print what each wrote."""
    import time
    import ingest
    names = ingest.resolve(args.sources)
    worst = 0
    for name in names:
        module = ingest.load(name)
        t0 = time.time()
        try:
            batch = ingest.common.run_source(module, args)
        except Exception as exc:  # the batch row is already marked failed by run_source
            print(f"[{name}] FAILED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}", file=sys.stderr)
            worst = 1
            if not args.keep_going:
                return 1
            continue
        print(f"[{name}] batch {batch.id}: seen={batch.seen} written={batch.written} skipped={batch.skipped} "
              f"files={len(set(batch.file_shas))} ({time.time() - t0:.1f}s)")
        for line in getattr(batch, "tally_lines", []):
            print("  " + line)
        for note in batch.notes[: args.notes]:
            print("  note: " + note)
    return worst


def cmd_arch_check(args) -> int:
    """Load conformance reports (ingest/arch_check.py) and print each model's requirement statuses."""
    import ingest
    batch = ingest.common.run_source(ingest.load("arch_check"), args)
    print(f"[arch_check] batch {batch.id}: seen={batch.seen} written={batch.written} skipped={batch.skipped}")
    for line in getattr(batch, "tally_lines", []):
        print("  " + line)
    for note in batch.notes[:20]:
        print("  note: " + note)
    return 0


def cmd_log_eval(args) -> int:
    """Load proof eval files (ingest/proof_eval.py); prints the eval run ids for `compare`."""
    import ingest
    batch = ingest.common.run_source(ingest.load("proof_eval"), args)
    print(f"[proof_eval] batch {batch.id}: seen={batch.seen} written={batch.written} skipped={batch.skipped}")
    for line in getattr(batch, "tally_lines", []):
        print("  " + line)
    return 0


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
    p = sub.add_parser("ingest", help="run ingesters: names under ingest/ or a group (models-runs)")
    p.add_argument("sources", nargs="+")
    p.add_argument("--keep-going", action="store_true", help="continue with the next source after a failure")
    p.add_argument("--notes", type=int, default=10, help="print up to N batch notes per source")
    p.add_argument("--inventory", help="inventory snapshot directory (knowledge ingesters)")
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("arch-check", help="load chimera_omni.conformance reports into the conformance tables")
    p.add_argument("--report", action="append", required=True, help="conformance report JSON (repeatable)")
    p.add_argument("--model", help="existing model_key / ledger id the (single) report describes")
    p.set_defaults(func=cmd_arch_check)
    p = sub.add_parser("log-eval", help="load a proof's omni-lab-proof-eval-v1 file(s): suite, eval run, results, grades")
    p.add_argument("--file", action="append", required=True, help="eval JSON (repeatable)")
    p.set_defaults(func=cmd_log_eval)
    import omni_lab_run
    omni_lab_run.add_parsers(sub)
    args = ap.parse_args(argv)
    import psycopg2
    try:
        return args.func(args)
    except db.LabError as exc:
        print(f"[refused] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[interrupted]", file=sys.stderr)
        return 130
    except psycopg2.Error as exc:
        print(f"[db error] {type(exc).__name__}: {str(exc).strip()}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
