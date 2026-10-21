#!/usr/bin/env python3
"""omni_lab_db -- shared database layer for the Omni Lab (omni_lab PostgreSQL database).

Raw psycopg2, every statement parameterised, one connection per caller, the caller owns
commit/close.  No ORM and no object-to-row mapping: each helper below is one explicit SQL
statement for one table.  Used by the migration CLI (omni_lab.py), the ingesters and the
eval runner.

Credentials (never in code or in the repo):
  profile  env prefix            default user   vault key
  rw       OMNI_LAB_PG_          omni_lab       omni_lab_pg
  ro       OMNI_LAB_RO_PG_       omni_lab_ro    omni_lab_ro_pg
  admin    OMNI_LAB_ADMIN_PG_    agent_memory   agent_memory_pg   (bootstrap and tests only)
Each prefix accepts HOST, PORT, DB, USER and one of PASSWORD, PASSWORD_FILE, PASSWORD_COMMAND
(same resolution order as the memory backend; "{python}" as the first word of a command is the
running interpreter).  Missing ro/admin HOST/PORT fall back to the rw values.  The first existing of
$OMNI_LAB_ENV_FILE, a gitignored .env beside this file (workstation: helper commands only) and
~/.config/omni-lab/env (AI server: mode 600, generated from the vault like chimera-intake/env) is
loaded; real environment variables always win.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

TOOL = "omni_lab"
VERSION = "0.1.0"
HERE = Path(__file__).resolve().parent
MIGRATIONS_DIR = HERE / "migrations"
MIGRATION_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
MIGRATION_LOCK_KEY = 0x6F6D6E696C6162  # 'omnilab' -- pg_advisory_xact_lock key for migrate()
ENV_PREFIXES = ("OMNI_LAB_PG_", "OMNI_LAB_RO_PG_", "OMNI_LAB_ADMIN_PG_")
PROFILES = {
    "rw":    {"prefix": "OMNI_LAB_PG_",       "user": "omni_lab",     "db": "omni_lab"},
    "ro":    {"prefix": "OMNI_LAB_RO_PG_",    "user": "omni_lab_ro",  "db": "omni_lab"},
    "admin": {"prefix": "OMNI_LAB_ADMIN_PG_", "user": "agent_memory", "db": "postgres"},
}


class LabError(RuntimeError):
    """A refusal or configuration error with a message meant for the operator."""


# ---------------------------------------------------------------------------
# environment / credentials
# ---------------------------------------------------------------------------

def load_local_env(path: Path) -> None:
    """KEY=VALUE lines with an OMNI_LAB_* prefix; real environment wins."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.startswith(ENV_PREFIXES):
            os.environ.setdefault(key, value)


def _first_env_file() -> Path | None:
    """OMNI_LAB_ENV_FILE, else .env beside this file (workstation), else ~/.config/omni-lab/env (AI server)."""
    for candidate in (os.environ.get("OMNI_LAB_ENV_FILE", ""), HERE / ".env", Path.home() / ".config/omni-lab/env"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


_ENV_FILE = _first_env_file()
if _ENV_FILE is not None:
    load_local_env(_ENV_FILE)


def read_password_file(path_text: str) -> str:
    if not path_text:
        return ""
    try:
        return Path(path_text).expanduser().read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def read_password_command(command: str) -> str:
    """Run a credential helper and return the last non-empty stdout line ("" on any failure)."""
    if not command:
        return ""
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return ""
    if not argv:
        return ""
    if argv[0] == "{python}":
        argv[0] = sys.executable
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _env(prefix: str, key: str, fallback_prefix: str | None = None, default: str = "") -> str:
    value = os.environ.get(prefix + key, "")
    if not value and fallback_prefix:
        value = os.environ.get(fallback_prefix + key, "")
    return value or default


def db_params(profile: str = "rw", dbname: str | None = None) -> dict[str, Any]:
    """Connection parameters for a profile.  Raises LabError if no password resolves."""
    if profile not in PROFILES:
        raise LabError(f"unknown profile {profile!r}")
    spec = PROFILES[profile]
    prefix = spec["prefix"]
    fallback = None if profile == "rw" else PROFILES["rw"]["prefix"]
    password = (os.environ.get(prefix + "PASSWORD", "")
                or read_password_file(os.environ.get(prefix + "PASSWORD_FILE", ""))
                or read_password_command(os.environ.get(prefix + "PASSWORD_COMMAND", "")))
    if not password:
        raise LabError(f"no password for profile {profile}: set {prefix}PASSWORD, {prefix}PASSWORD_FILE "
                       f"or {prefix}PASSWORD_COMMAND (vault key: see omni_lab_db.py docstring)")
    return {
        "host": _env(prefix, "HOST", fallback, "127.0.0.1"),
        "port": int(_env(prefix, "PORT", fallback, "25490")),
        "dbname": dbname or os.environ.get(prefix + "DB", "") or spec["db"],
        "user": os.environ.get(prefix + "USER", "") or spec["user"],
        "password": password,
    }


def connect(profile: str = "rw", dbname: str | None = None, app: str = "", autocommit: bool = False,
            statement_timeout_ms: int = 30000, lock_timeout_ms: int = 5000):
    """Open one psycopg2 connection.  'ro' sessions are additionally read-only at the session level
    (the omni_lab_ro role has SELECT only, which is the real guarantee)."""
    import psycopg2  # local import: the GPU worker venv never imports this module

    params = db_params(profile, dbname)
    options = f"-c statement_timeout={int(statement_timeout_ms)} -c lock_timeout={int(lock_timeout_ms)}"
    if profile == "ro":
        options += " -c default_transaction_read_only=on"
    conn = psycopg2.connect(host=params["host"], port=params["port"], dbname=params["dbname"],
                            user=params["user"], password=params["password"], connect_timeout=5,
                            options=options, application_name=f"{TOOL}/{app or profile}-{VERSION}")
    conn.autocommit = autocommit
    return conn


# ---------------------------------------------------------------------------
# hashing / provenance
# ---------------------------------------------------------------------------

def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def canonical_json(obj: Any) -> str:
    """Stable text form used for every content hash (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def this_host() -> str:
    """'ai' on the AI server, 'workstation' on Windows, else the hostname."""
    override = os.environ.get("OMNI_LAB_HOST_NAME", "")
    if override:
        return override
    if os.name == "nt":
        return "workstation"
    name = socket.gethostname()
    return "ai" if name == "cereal-X10DRG" else name


def provenance(path: str | Path, host: str | None = None) -> tuple[str, str, str]:
    """(src_path, src_sha256, src_host) for a source file."""
    p = Path(path)
    return (p.as_posix(), file_sha256(p), host or this_host())


def case_content_sha256(prompt: Any, tools: Any = None, media: Any = None) -> str:
    """Identity of a test case's question: an edited prompt/tool list/media ref is a different case."""
    return sha256_json({"prompt": prompt, "tools": tools, "media": media})


def scram_sha256_verifier(password: str, iterations: int = 4096, salt: bytes | None = None) -> str:
    """PostgreSQL SCRAM-SHA-256 verifier (RFC 5802/7677), computed client-side so CREATE/ALTER ROLE never
    sends the plaintext password to the server (or to its statement log).  ASCII passwords only: the
    generated ones are token_urlsafe, for which SASLprep is the identity."""
    import base64
    import hmac
    if not password.isascii():
        raise LabError("scram_sha256_verifier: non-ASCII password would need SASLprep")
    salt = salt if salt is not None else os.urandom(16)
    salted = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    stored_key = hashlib.sha256(client_key).digest()
    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
    b64 = lambda b: base64.b64encode(b).decode("ascii")  # noqa: E731
    return f"SCRAM-SHA-256${iterations}:{b64(salt)}${b64(stored_key)}:{b64(server_key)}"


def _j(value: Any) -> str | None:
    """Python value -> JSON text for a %s::jsonb parameter (None stays NULL)."""
    return None if value is None else canonical_json(value)


# ---------------------------------------------------------------------------
# migrations
# ---------------------------------------------------------------------------

def migration_files(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, str, Path, str]]:
    """[(version, name, path, sha256)] sorted; refuses gaps and duplicate versions."""
    found = []
    for p in sorted(directory.glob("*.sql")):
        m = MIGRATION_RE.match(p.name)
        if not m:
            raise LabError(f"migration file name not NNNN_name.sql: {p.name}")
        found.append((int(m.group(1)), m.group(2), p, file_sha256(p)))
    versions = [v for v, *_ in found]
    if len(set(versions)) != len(versions):
        raise LabError(f"duplicate migration versions in {directory}")
    if versions and versions != list(range(1, len(versions) + 1)):
        raise LabError(f"migration versions must run 1..N without gaps, found {versions}")
    return found


SCHEMA_MIGRATION_DDL = """
CREATE TABLE IF NOT EXISTS schema_migration (
    version     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    applied_by  TEXT NOT NULL DEFAULT current_user,
    tool        TEXT
)"""


def applied_migrations(conn) -> dict[int, tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.schema_migration') IS NOT NULL")
        if not cur.fetchone()[0]:
            return {}
        cur.execute("SELECT version, name, sha256 FROM schema_migration ORDER BY version")
        return {v: (n, s) for v, n, s in cur.fetchall()}


def migrate(conn, directory: Path = MIGRATIONS_DIR, dry_run: bool = False) -> list[tuple[int, str]]:
    """Apply pending migrations, each in its own transaction, serialised by an advisory lock.

    Refuses when an applied migration's file is missing or its sha256 changed: migrations are
    forward-only, a change is a new numbered file."""
    files = migration_files(directory)
    if conn.autocommit:
        raise LabError("migrate() needs a transactional connection (autocommit=False)")
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_KEY,))
        cur.execute(SCHEMA_MIGRATION_DDL)
    conn.commit()
    done = applied_migrations(conn)
    on_disk = {v: (n, s) for v, n, _p, s in files}
    for version, (name, sha) in done.items():
        if version not in on_disk:
            raise LabError(f"applied migration {version:04d}_{name} has no file in {directory}")
        if on_disk[version][1] != sha:
            raise LabError(f"migration {version:04d}_{name} changed after it was applied "
                           f"(db {sha[:12]} != file {on_disk[version][1][:12]}); add a new migration instead")
    applied: list[tuple[int, str]] = []
    for version, name, path, sha in files:
        if version in done:
            continue
        if dry_run:
            applied.append((version, name))
            continue
        sql_text = path.read_text(encoding="utf-8")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_KEY,))
                cur.execute("SELECT 1 FROM schema_migration WHERE version = %s", (version,))
                if cur.fetchone():          # applied concurrently by another process
                    conn.rollback()
                    continue
                cur.execute(sql_text)
                cur.execute("INSERT INTO schema_migration (version, name, sha256, tool) VALUES (%s, %s, %s, %s)",
                            (version, name, sha, f"{TOOL}-{VERSION}"))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        applied.append((version, name))
    return applied


def schema_version(conn) -> int:
    done = applied_migrations(conn)
    return max(done) if done else 0


def table_counts(conn) -> list[tuple[str, int]]:
    """Exact row count per base table in public (small database; count(*) is fine)."""
    from psycopg2 import sql
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name")
        names = [r[0] for r in cur.fetchall()]
        out = []
        for name in names:
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(name)))
            out.append((name, cur.fetchone()[0]))
    return out


# ---------------------------------------------------------------------------
# ingest batches
# ---------------------------------------------------------------------------

def start_batch(conn, source: str, parser: str | None = None, parser_sha256: str | None = None,
                host: str | None = None, argv: list[str] | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ingest_batch (source, parser, parser_sha256, host, argv) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb) RETURNING id",
                    (source, parser, parser_sha256, host or this_host(), _j(argv)))
        return cur.fetchone()[0]


def end_batch(conn, batch_id: int, status: str, rows_seen: int = 0, rows_written: int = 0,
              rows_skipped: int = 0, error: str | None = None, file_shas: list[str] | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE ingest_batch SET ended_at = clock_timestamp(), status = %s, rows_seen = %s, rows_written = %s, "
                    "rows_skipped = %s, error = %s, file_shas = %s WHERE id = %s",
                    (status, rows_seen, rows_written, rows_skipped, error, sorted(set(file_shas or [])), batch_id))
        if cur.rowcount != 1:
            raise LabError(f"ingest_batch {batch_id} not found")


def source_sha_already_ingested(conn, source: str, src_sha256: str) -> bool:
    """True when a successful batch of `source` already consumed this file sha (ingesters skip unless --force)."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM ingest_batch WHERE source = %s AND status = 'ok' AND file_shas @> ARRAY[%s]::text[] "
                    "LIMIT 1", (source, src_sha256))
        return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# hosts / gpus / env / tools
# ---------------------------------------------------------------------------

def ensure_host(conn, name: str, detail: dict | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO host (name, detail) VALUES (%s, %s::jsonb) "
                    "ON CONFLICT (name) DO UPDATE SET detail = host.detail || EXCLUDED.detail RETURNING id",
                    (name, _j(detail or {})))
        return cur.fetchone()[0]


def upsert_gpu(conn, host_id: int, gpu_index: int, uuid: str, name: str, sm: str | None = None,
               vram_mib: int | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO gpu (host_id, gpu_index, uuid, name, sm, vram_mib) VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (host_id, uuid) DO UPDATE SET gpu_index = EXCLUDED.gpu_index, name = EXCLUDED.name, "
                    "sm = COALESCE(EXCLUDED.sm, gpu.sm), vram_mib = COALESCE(EXCLUDED.vram_mib, gpu.vram_mib), "
                    "last_seen = now() RETURNING id",
                    (host_id, gpu_index, uuid, name, sm, vram_mib))
        return cur.fetchone()[0]


def gpu_id_by_index(conn, host_id: int, gpu_index: int) -> int | None:
    """Backfill-only lookup (status lines carry the index); live runs must use gpu_id_by_uuid."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM gpu WHERE host_id = %s AND gpu_index = %s ORDER BY last_seen DESC LIMIT 1",
                    (host_id, gpu_index))
        row = cur.fetchone()
        return row[0] if row else None


def gpu_id_by_uuid(conn, host_id: int, uuid: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM gpu WHERE host_id = %s AND uuid = %s", (host_id, uuid))
        row = cur.fetchone()
        return row[0] if row else None


ENV_FIELDS = ("python", "torch", "cuda", "driver", "transformers", "fla", "triton", "llama_cpp_build", "code_commit")


def ensure_env(conn, python: str | None = None, torch: str | None = None, cuda: str | None = None,
               driver: str | None = None, transformers: str | None = None, fla: str | None = None,
               triton: str | None = None, llama_cpp_build: str | None = None, code_commit: str | None = None,
               env_vars: dict | None = None, packages: dict | None = None) -> int:
    values = (python, torch, cuda, driver, transformers, fla, triton, llama_cpp_build, code_commit)
    env_sha = sha256_json({**dict(zip(ENV_FIELDS, values)), "env_vars": env_vars or {}, "packages": packages or {}})
    with conn.cursor() as cur:
        # SELECT first: an upsert would hold the row lock for the rest of a long ingest transaction
        cur.execute("SELECT id FROM env WHERE env_sha256 = %s", (env_sha,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute("INSERT INTO env (env_sha256, python, torch, cuda, driver, transformers, fla, triton, "
                    "llama_cpp_build, code_commit, env_vars, packages) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb) "
                    "ON CONFLICT (env_sha256) DO UPDATE SET env_sha256 = EXCLUDED.env_sha256 RETURNING id",
                    (env_sha, *values, _j(env_vars or {}), _j(packages or {})))
        return cur.fetchone()[0]


def ensure_tool_version(conn, tool: str, version: str, sha256: str, path: str | None = None,
                        pinned: bool = False, superseded: bool = False) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO tool (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
                    "RETURNING id", (tool,))
        tool_id = cur.fetchone()[0]
        cur.execute("INSERT INTO tool_version (tool_id, version, sha256, path, pinned, superseded) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (tool_id, sha256) DO UPDATE SET version = EXCLUDED.version, "
                    "path = COALESCE(EXCLUDED.path, tool_version.path), pinned = EXCLUDED.pinned, "
                    "superseded = EXCLUDED.superseded RETURNING id",
                    (tool_id, version, sha256, path, pinned, superseded))
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def upsert_model(conn, model_key: str, project: str, role: str, layout: str, family: str | None = None,
                 stage: str | None = None, path: str | None = None, host_id: int | None = None,
                 ledger_checkpoint_id: str | None = None, directory_sha256: str | None = None,
                 parameter_count: int | None = None, size_bytes: int | None = None, dtype: str | None = None,
                 config: dict | None = None, sealed_verdict: str | None = None, status: str = "present",
                 created_at: Any = None, notes: str | None = None,
                 prov: tuple[str, str, str] | None = None, batch_id: int | None = None) -> int:
    """Insert or refresh a model by model_key; NULL arguments never erase known values."""
    src_path, src_sha, src_host = prov or (None, None, None)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO model (model_key, project, role, layout, family, stage, path, host_id, ledger_checkpoint_id, "
            "directory_sha256, parameter_count, size_bytes, dtype, config, sealed_verdict, status, created_at, notes, "
            "src_path, src_sha256, src_host, ingest_batch_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (model_key) DO UPDATE SET "
            "project = EXCLUDED.project, role = EXCLUDED.role, layout = EXCLUDED.layout, "
            "family = COALESCE(EXCLUDED.family, model.family), stage = COALESCE(EXCLUDED.stage, model.stage), "
            "path = COALESCE(EXCLUDED.path, model.path), host_id = COALESCE(EXCLUDED.host_id, model.host_id), "
            "ledger_checkpoint_id = COALESCE(EXCLUDED.ledger_checkpoint_id, model.ledger_checkpoint_id), "
            "directory_sha256 = COALESCE(EXCLUDED.directory_sha256, model.directory_sha256), "
            "parameter_count = COALESCE(EXCLUDED.parameter_count, model.parameter_count), "
            "size_bytes = COALESCE(EXCLUDED.size_bytes, model.size_bytes), dtype = COALESCE(EXCLUDED.dtype, model.dtype), "
            "config = model.config || EXCLUDED.config, "
            "sealed_verdict = COALESCE(EXCLUDED.sealed_verdict, model.sealed_verdict), status = EXCLUDED.status, "
            "created_at = COALESCE(EXCLUDED.created_at, model.created_at), notes = COALESCE(EXCLUDED.notes, model.notes), "
            "src_path = COALESCE(EXCLUDED.src_path, model.src_path), src_sha256 = COALESCE(EXCLUDED.src_sha256, model.src_sha256), "
            "src_host = COALESCE(EXCLUDED.src_host, model.src_host), "
            "ingest_batch_id = COALESCE(EXCLUDED.ingest_batch_id, model.ingest_batch_id) "
            "RETURNING id",
            (model_key, project, role, layout, family, stage, path, host_id, ledger_checkpoint_id, directory_sha256,
             parameter_count, size_bytes, dtype, _j(config or {}), sealed_verdict, status, created_at, notes,
             src_path, src_sha, src_host, batch_id))
        return cur.fetchone()[0]


def model_id(conn, model_key: str) -> int | None:
    """Resolve by model_key or by ledger checkpoint id."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM model WHERE model_key = %s OR ledger_checkpoint_id = %s "
                    "ORDER BY (model_key = %s) DESC LIMIT 1", (model_key, model_key, model_key))
        row = cur.fetchone()
        return row[0] if row else None


def link_lineage(conn, child_id: int, parent_id: int, relation: str, ordinal: int = 0,
                 detail: dict | None = None, prov: tuple[str, str, str] | None = None,
                 batch_id: int | None = None) -> None:
    src_path, src_sha, src_host = prov or (None, None, None)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO model_lineage (child_id, parent_id, relation, ordinal, detail, "
                    "src_path, src_sha256, src_host, ingest_batch_id) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
                    "ON CONFLICT (child_id, parent_id, relation) DO UPDATE SET ordinal = EXCLUDED.ordinal, "
                    "detail = EXCLUDED.detail",
                    (child_id, parent_id, relation, ordinal, _j(detail or {}), src_path, src_sha, src_host, batch_id))


# ---------------------------------------------------------------------------
# runs / devices / metrics
# ---------------------------------------------------------------------------

def upsert_run(conn, run_key: str, project: str, kind: str, name: str, status: str,
               model_id_: int | None = None, tool_version_id: int | None = None, host_id: int | None = None,
               env_id: int | None = None, started_at: Any = None, ended_at: Any = None,
               wall_seconds: float | None = None, exit_code: int | None = None, config: dict | None = None,
               receipt_path: str | None = None, receipt_sha256: str | None = None, log_path: str | None = None,
               run_dir: str | None = None, parent_run_id: int | None = None, owner: str | None = None,
               notes: str | None = None, prov: tuple[str, str, str] | None = None,
               batch_id: int | None = None) -> int:
    """Insert or refresh a run by run_key.  config_sha256 is always recomputed from config."""
    cfg = config or {}
    src_path, src_sha, src_host = prov or (None, None, None)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO run (run_key, project, kind, name, model_id, tool_version_id, host_id, env_id, status, "
            "started_at, ended_at, wall_seconds, exit_code, config, config_sha256, receipt_path, receipt_sha256, "
            "log_path, run_dir, parent_run_id, owner, notes, src_path, src_sha256, src_host, ingest_batch_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s, %s, %s, %s) "
            "ON CONFLICT (run_key) DO UPDATE SET "
            "name = EXCLUDED.name, status = EXCLUDED.status, "
            "model_id = COALESCE(EXCLUDED.model_id, run.model_id), "
            "tool_version_id = COALESCE(EXCLUDED.tool_version_id, run.tool_version_id), "
            "host_id = COALESCE(EXCLUDED.host_id, run.host_id), env_id = COALESCE(EXCLUDED.env_id, run.env_id), "
            "started_at = COALESCE(EXCLUDED.started_at, run.started_at), "
            "ended_at = COALESCE(EXCLUDED.ended_at, run.ended_at), "
            "wall_seconds = COALESCE(EXCLUDED.wall_seconds, run.wall_seconds), "
            "exit_code = COALESCE(EXCLUDED.exit_code, run.exit_code), "
            "config = EXCLUDED.config, config_sha256 = EXCLUDED.config_sha256, "
            "receipt_path = COALESCE(EXCLUDED.receipt_path, run.receipt_path), "
            "receipt_sha256 = COALESCE(EXCLUDED.receipt_sha256, run.receipt_sha256), "
            "log_path = COALESCE(EXCLUDED.log_path, run.log_path), run_dir = COALESCE(EXCLUDED.run_dir, run.run_dir), "
            "parent_run_id = COALESCE(EXCLUDED.parent_run_id, run.parent_run_id), "
            "owner = COALESCE(EXCLUDED.owner, run.owner), notes = COALESCE(EXCLUDED.notes, run.notes), "
            "src_path = COALESCE(EXCLUDED.src_path, run.src_path), src_sha256 = COALESCE(EXCLUDED.src_sha256, run.src_sha256), "
            "src_host = COALESCE(EXCLUDED.src_host, run.src_host), "
            "ingest_batch_id = COALESCE(EXCLUDED.ingest_batch_id, run.ingest_batch_id) "
            "RETURNING id",
            (run_key, project, kind, name, model_id_, tool_version_id, host_id, env_id, status, started_at, ended_at,
             wall_seconds, exit_code, _j(cfg), sha256_json(cfg), receipt_path, receipt_sha256, log_path, run_dir,
             parent_run_id, owner, notes, src_path, src_sha, src_host, batch_id))
        return cur.fetchone()[0]


def finish_run(conn, run_id_: int, status: str, exit_code: int | None = None, receipt_path: str | None = None,
               receipt_sha256: str | None = None, log_path: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE run SET status = %s, ended_at = COALESCE(ended_at, now()), "
                    "exit_code = COALESCE(%s, exit_code), receipt_path = COALESCE(%s, receipt_path), "
                    "receipt_sha256 = COALESCE(%s, receipt_sha256), log_path = COALESCE(%s, log_path), "
                    "wall_seconds = COALESCE(wall_seconds, EXTRACT(EPOCH FROM (COALESCE(ended_at, now()) - started_at))) "
                    "WHERE id = %s",
                    (status, exit_code, receipt_path, receipt_sha256, log_path, run_id_))
        if cur.rowcount != 1:
            raise LabError(f"run {run_id_} not found")


def add_run_device(conn, run_id_: int, gpu_id: int, role: str = "compute", kernel_path: str | None = None,
                   kernels: dict | None = None, layers: tuple[int, int] | None = None,
                   mem_mib_peak: int | None = None, detail: dict | None = None) -> None:
    """layers=(lo, hi) is stored as the half-open int4range [lo, hi)."""
    with conn.cursor() as cur:
        cur.execute("INSERT INTO run_device (run_id, gpu_id, role, kernel_path, kernels, layers, mem_mib_peak, detail) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb, CASE WHEN %s::int IS NULL THEN NULL ELSE int4range(%s, %s) END, "
                    "%s, %s::jsonb) "
                    "ON CONFLICT (run_id, gpu_id, role) DO UPDATE SET kernel_path = EXCLUDED.kernel_path, "
                    "kernels = EXCLUDED.kernels, layers = EXCLUDED.layers, "
                    "mem_mib_peak = COALESCE(EXCLUDED.mem_mib_peak, run_device.mem_mib_peak), detail = EXCLUDED.detail",
                    (run_id_, gpu_id, role, kernel_path, _j(kernels or {}),
                     layers[0] if layers else None, layers[0] if layers else None, layers[1] if layers else None,
                     mem_mib_peak, _j(detail or {})))


def link_run_model(conn, run_id_: int, model_id_: int, role: str) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO run_model (run_id, model_id, role) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (run_id_, model_id_, role))


def put_metric(conn, run_id_: int, name: str, value: float, step: int = 0, denominator: float | None = None,
               ts: Any = None, extra: dict | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO metric (run_id, name, step, value, denominator, ts, extra) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (run_id, name, step) DO UPDATE SET value = EXCLUDED.value, "
                    "denominator = EXCLUDED.denominator, ts = COALESCE(EXCLUDED.ts, metric.ts), extra = EXCLUDED.extra",
                    (run_id_, name, step, float(value), denominator, ts, _j(extra)))


def put_metrics(conn, run_id_: int, rows: Iterable[tuple[str, int, float]]) -> int:
    """Bulk (name, step, value) rows for loss curves."""
    from psycopg2.extras import execute_values
    data = [(run_id_, n, int(s), float(v)) for n, s, v in rows]
    if not data:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, "INSERT INTO metric (run_id, name, step, value) VALUES %s "
                            "ON CONFLICT (run_id, name, step) DO UPDATE SET value = EXCLUDED.value", data)
    return len(data)


# ---------------------------------------------------------------------------
# suites / cases / graders / tests
# ---------------------------------------------------------------------------

def ensure_suite(conn, name: str, version: str, status: str = "sealed", modality_in: list[str] | None = None,
                 modality_out: list[str] | None = None, default_grader: str | None = None,
                 bank_sha256: str | None = None, description: str | None = None,
                 prov: tuple[str, str, str] | None = None, batch_id: int | None = None) -> int:
    src_path, src_sha, src_host = prov or (None, None, None)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO test_suite (name, version, status, modality_in, modality_out, default_grader, "
                    "bank_sha256, description, src_path, src_sha256, src_host, ingest_batch_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (name, version) DO UPDATE SET status = EXCLUDED.status, "
                    "default_grader = COALESCE(EXCLUDED.default_grader, test_suite.default_grader), "
                    "description = COALESCE(EXCLUDED.description, test_suite.description) RETURNING id",
                    (name, version, status, modality_in or ["text"], modality_out or ["text"], default_grader,
                     bank_sha256, description, src_path, src_sha, src_host, batch_id))
        return cur.fetchone()[0]


def suite_id(conn, name: str, version: str | None = None) -> int | None:
    """Exact version, or the newest (by created_at, then id) when version is None."""
    with conn.cursor() as cur:
        if version is None:
            cur.execute("SELECT id FROM test_suite WHERE name = %s ORDER BY created_at DESC, id DESC LIMIT 1", (name,))
        else:
            cur.execute("SELECT id FROM test_suite WHERE name = %s AND version = %s", (name, version))
        row = cur.fetchone()
        return row[0] if row else None


def ensure_case(conn, case_id: str, prompt: Any, tools: Any = None, media: Any = None, reference: Any = None,
                category: str | None = None, lane: str | None = None, tags: list[str] | None = None,
                modality_in: list[str] | None = None, modality_out: list[str] | None = None,
                grader: str | None = None, prov: tuple[str, str, str] | None = None,
                batch_id: int | None = None) -> int:
    """Global case keyed by (case_id, content_sha256).  The same question in two suites is one row."""
    content_sha = case_content_sha256(prompt, tools, media)
    src_path, src_sha, src_host = prov or (None, None, None)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO test_case (case_id, content_sha256, category, lane, tags, modality_in, modality_out, "
                    "prompt, tools, media, reference, grader, src_path, src_sha256, src_host, ingest_batch_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (case_id, content_sha256) DO UPDATE SET "
                    "category = COALESCE(EXCLUDED.category, test_case.category), "
                    "lane = COALESCE(EXCLUDED.lane, test_case.lane), "
                    "reference = COALESCE(EXCLUDED.reference, test_case.reference), "
                    "grader = COALESCE(EXCLUDED.grader, test_case.grader) RETURNING id",
                    (case_id, content_sha, category, lane, tags or [], modality_in or ["text"], modality_out or ["text"],
                     _j(prompt), _j(tools), _j(media), _j(reference), grader, src_path, src_sha, src_host, batch_id))
        return cur.fetchone()[0]


def add_suite_member(conn, suite_id_: int, case_pk: int, ordinal: int) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO suite_member (suite_id, case_pk, ordinal) VALUES (%s, %s, %s) "
                    "ON CONFLICT (suite_id, case_pk) DO UPDATE SET ordinal = EXCLUDED.ordinal",
                    (suite_id_, case_pk, ordinal))


def suite_cases(conn, suite_id_: int) -> list[tuple[int, str, Any]]:
    """[(case_pk, case_id, prompt)] in bank order."""
    with conn.cursor() as cur:
        cur.execute("SELECT tc.id, tc.case_id, tc.prompt FROM suite_member sm JOIN test_case tc ON tc.id = sm.case_pk "
                    "WHERE sm.suite_id = %s ORDER BY sm.ordinal", (suite_id_,))
        return cur.fetchall()


def ensure_grader(conn, name: str, kind: str, scale: str, description: str | None = None,
                  tool_version_id: int | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO grader (name, kind, scale, description, tool_version_id) VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (name) DO UPDATE SET scale = EXCLUDED.scale, "
                    "description = COALESCE(EXCLUDED.description, grader.description), "
                    "tool_version_id = COALESCE(EXCLUDED.tool_version_id, grader.tool_version_id) RETURNING id",
                    (name, kind, scale, description, tool_version_id))
        return cur.fetchone()[0]


def upsert_test(conn, name: str, suite_name: str, engine: str, gen: dict | None = None,
                suite_version: str | None = None, devices: list[int] | None = None,
                case_filter: list[str] | None = None, notes: str | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO test (name, suite_name, suite_version, engine, gen, devices, case_filter, notes) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s) "
                    "ON CONFLICT (name) DO UPDATE SET suite_name = EXCLUDED.suite_name, "
                    "suite_version = EXCLUDED.suite_version, engine = EXCLUDED.engine, gen = EXCLUDED.gen, "
                    "devices = EXCLUDED.devices, case_filter = EXCLUDED.case_filter, "
                    "notes = COALESCE(EXCLUDED.notes, test.notes) RETURNING id",
                    (name, suite_name, suite_version, engine, _j(gen or {}), devices, case_filter, notes))
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# eval runs / results / grades / artifacts
# ---------------------------------------------------------------------------

def upsert_eval_run(conn, run_id_: int, model_id_: int, suite_id_: int, engine: str, gen: dict | None = None,
                    test_id: int | None = None, tokenizer_sha256: str | None = None,
                    chat_template_sha256: str | None = None, bank_file_sha256: str | None = None,
                    thinking_requested: str | None = None, thinking_took_effect: bool | None = None,
                    case_filter: list[str] | None = None, label: str | None = None,
                    baseline_run_id: int | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO eval_run (run_id, model_id, suite_id, test_id, engine, gen, tokenizer_sha256, "
                    "chat_template_sha256, bank_file_sha256, thinking_requested, thinking_took_effect, case_filter, "
                    "label, baseline_run_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (run_id) DO UPDATE SET model_id = EXCLUDED.model_id, suite_id = EXCLUDED.suite_id, "
                    "test_id = COALESCE(EXCLUDED.test_id, eval_run.test_id), engine = EXCLUDED.engine, gen = EXCLUDED.gen, "
                    "tokenizer_sha256 = COALESCE(EXCLUDED.tokenizer_sha256, eval_run.tokenizer_sha256), "
                    "chat_template_sha256 = COALESCE(EXCLUDED.chat_template_sha256, eval_run.chat_template_sha256), "
                    "bank_file_sha256 = COALESCE(EXCLUDED.bank_file_sha256, eval_run.bank_file_sha256), "
                    "thinking_requested = COALESCE(EXCLUDED.thinking_requested, eval_run.thinking_requested), "
                    "thinking_took_effect = COALESCE(EXCLUDED.thinking_took_effect, eval_run.thinking_took_effect), "
                    "case_filter = EXCLUDED.case_filter, label = COALESCE(EXCLUDED.label, eval_run.label), "
                    "baseline_run_id = COALESCE(EXCLUDED.baseline_run_id, eval_run.baseline_run_id)",
                    (run_id_, model_id_, suite_id_, test_id, engine, _j(gen or {}), tokenizer_sha256,
                     chat_template_sha256, bank_file_sha256, thinking_requested, thinking_took_effect, case_filter,
                     label, baseline_run_id))


def upsert_eval_result(conn, eval_run_id: int, case_pk: int, output_text: str | None = None,
                       output_tokens: int | None = None, termination: str | None = None, form: str | None = None,
                       artifact_flags: list[str] | None = None, finite: bool | None = None,
                       latency_ms: int | None = None, tok_s: float | None = None, raw: Any = None,
                       prov: tuple[str, str, str] | None = None, batch_id: int | None = None) -> int:
    src_path, src_sha, src_host = prov or (None, None, None)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO eval_result (eval_run_id, case_pk, output_text, output_tokens, termination, form, "
                    "artifact_flags, finite, latency_ms, tok_s, raw, src_path, src_sha256, src_host, ingest_batch_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
                    "ON CONFLICT (eval_run_id, case_pk) DO UPDATE SET output_text = EXCLUDED.output_text, "
                    "output_tokens = EXCLUDED.output_tokens, termination = EXCLUDED.termination, form = EXCLUDED.form, "
                    "artifact_flags = EXCLUDED.artifact_flags, finite = EXCLUDED.finite, latency_ms = EXCLUDED.latency_ms, "
                    "tok_s = EXCLUDED.tok_s, raw = EXCLUDED.raw RETURNING id",
                    (eval_run_id, case_pk, output_text, output_tokens, termination, form, artifact_flags or [], finite,
                     latency_ms, tok_s, _j(raw), src_path, src_sha, src_host, batch_id))
        return cur.fetchone()[0]


def put_grade(conn, eval_result_id: int, grader_id: int, grade: str, score: float | None = None,
              why: str | None = None, detail: Any = None, prov: tuple[str, str, str] | None = None,
              batch_id: int | None = None) -> None:
    """One grade per (result, grader); regrading replaces it and bumps graded_at."""
    src_path, src_sha, src_host = prov or (None, None, None)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO eval_grade (eval_result_id, grader_id, grade, score, why, detail, "
                    "src_path, src_sha256, src_host, ingest_batch_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
                    "ON CONFLICT (eval_result_id, grader_id) DO UPDATE SET grade = EXCLUDED.grade, score = EXCLUDED.score, "
                    "why = EXCLUDED.why, detail = EXCLUDED.detail, graded_at = now()",
                    (eval_result_id, grader_id, grade, score, why, _j(detail), src_path, src_sha, src_host, batch_id))


def add_eval_artifact(conn, eval_result_id: int, kind: str, path: str, sha256: str | None = None,
                      nbytes: int | None = None, duration_s: float | None = None, width: int | None = None,
                      height: int | None = None, sample_rate: int | None = None,
                      metrics: dict | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO eval_artifact (eval_result_id, kind, path, sha256, bytes, duration_s, width, height, "
                    "sample_rate, metrics) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (eval_result_id, path) DO UPDATE SET sha256 = EXCLUDED.sha256, bytes = EXCLUDED.bytes, "
                    "metrics = EXCLUDED.metrics RETURNING id",
                    (eval_result_id, kind, path, sha256, nbytes, duration_s, width, height, sample_rate,
                     _j(metrics or {})))
        return cur.fetchone()[0]
