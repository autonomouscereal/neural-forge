#!/usr/bin/env python3
"""bootstrap_db -- one-time creation of the omni_lab roles and database on agent-memory-postgres.

Run from the workstation (the vault lives there):
    C:/Python314/python.exe bootstrap/bootstrap_db.py plan        # read-only: what exists, what would be created
    C:/Python314/python.exe bootstrap/bootstrap_db.py apply       # create missing roles + database, then migrate

What `apply` does, in order, each step idempotent:
  1. for each missing role (omni_lab, omni_lab_ro): generate a password in-process
     (secrets.token_urlsafe(32)), store it in the vault FIRST (credman.py set <key>), read it back and
     compare in memory, then CREATE ROLE ... PASSWORD '<SCRAM-SHA-256 verifier>' (plaintext never sent).
     An existing role is left alone (no rotation here).
  2. CREATE DATABASE omni_lab OWNER omni_lab; REVOKE ALL ON DATABASE FROM PUBLIC;
     GRANT CONNECT TO omni_lab_ro.
  3. connect as omni_lab and apply migrations/ (omni_lab_db.migrate).
Admin connection: profile 'admin' (OMNI_LAB_ADMIN_PG_*; vault key agent_memory_pg).  Nothing in the
agent_memory or chimera_corpus databases is touched.  Rollback: DROP DATABASE omni_lab; DROP ROLE
omni_lab_ro; DROP ROLE omni_lab; credman.py rm omni_lab_pg / omni_lab_ro_pg.
Passwords are never printed.
"""
from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import omni_lab_db as db  # noqa: E402

CREDMAN = os.environ.get("OMNI_LAB_CREDMAN", "") or "C:/Users/cereal/.claude/skills/server-manager/credman.py"
DB_NAME = "omni_lab"
# role, vault key, extra role attributes.  omni_lab gets CREATEDB only so the test suite can create and
# drop its own throwaway omni_lab_test_* databases without the superuser credential.
ROLES = (("omni_lab", "omni_lab_pg", "CREATEDB"), ("omni_lab_ro", "omni_lab_ro_pg", ""))


def vault_get(key: str) -> str:
    out = subprocess.run([sys.executable, CREDMAN, "get", key], capture_output=True, text=True, timeout=30)
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return lines[-1] if out.returncode == 0 and lines else ""


def vault_set(key: str, value: str) -> None:
    out = subprocess.run([sys.executable, CREDMAN, "set", key, value], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise db.LabError(f"credman set {key} failed (exit {out.returncode})")
    if vault_get(key) != value:
        raise db.LabError(f"credman get {key} did not return the value just stored")


def state(cur) -> dict:
    cur.execute("SELECT rolname, rolcanlogin, rolcreatedb, rolsuper FROM pg_roles WHERE rolname = ANY(%s)",
                ([r for r, _k, _a in ROLES],))
    roles = {r: {"login": l, "createdb": c, "super": s} for r, l, c, s in cur.fetchall()}
    cur.execute("SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname = %s", (DB_NAME,))
    row = cur.fetchone()
    return {"roles": roles, "database_owner": row[0] if row else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("plan", "apply"))
    args = ap.parse_args()
    from psycopg2 import sql

    admin = db.connect("admin", dbname="postgres", app="bootstrap", autocommit=True)
    try:
        with admin.cursor() as cur:
            before = state(cur)
            print(f"[plan] roles present: {sorted(before['roles'])}; database {DB_NAME} owner: {before['database_owner']}")
            for role, key, _attrs in ROLES:
                print(f"[plan] role {role}: {'exists (left alone)' if role in before['roles'] else 'CREATE; vault key ' + key}")
            print(f"[plan] database {DB_NAME}: {'exists' if before['database_owner'] else 'CREATE OWNER omni_lab'}")
            if args.action == "plan":
                return 0
            for role, key, attrs in ROLES:
                if role in before["roles"]:
                    continue
                password = secrets.token_urlsafe(32)
                vault_set(key, password)                       # vault first, then the role
                verifier = db.scram_sha256_verifier(password)
                del password
                stmt = sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEROLE NOINHERIT CONNECTION LIMIT 20 "
                               + attrs + " PASSWORD {}").format(sql.Identifier(role), sql.Literal(verifier))
                cur.execute(stmt)
                print(f"[ok] role {role} created; password stored in vault key {key} (SCRAM verifier sent)")
            if not before["database_owner"]:
                cur.execute(sql.SQL("CREATE DATABASE {} OWNER {} ENCODING 'UTF8' TEMPLATE template0")
                            .format(sql.Identifier(DB_NAME), sql.Identifier("omni_lab")))
                print(f"[ok] database {DB_NAME} created")
            cur.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(DB_NAME)))
            cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(DB_NAME),
                                                                              sql.Identifier("omni_lab_ro")))
            after = state(cur)
            print(f"[ok] roles: {after['roles']}; database owner: {after['database_owner']}")
    finally:
        admin.close()

    conn = db.connect("rw", app="bootstrap-migrate")
    try:
        applied = db.migrate(conn)
        print(f"[ok] migrations applied: {[f'{v:04d}_{n}' for v, n in applied] or 'none pending'}; "
              f"schema version {db.schema_version(conn)}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except db.LabError as exc:
        print(f"[refused] {exc}", file=sys.stderr)
        sys.exit(2)
