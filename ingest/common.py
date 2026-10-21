"""ingest.common -- shared helpers for the Omni Lab ingesters (raw psycopg2 via omni_lab_db; no ORM).

Every ingester is a module ingest/<source>.py with

    SOURCE = "<name>"                   # ingest_batch.source
    def run(conn, batch, args) -> None  # reads its source files, upserts rows, counts into batch

and is driven by `omni_lab.py ingest <source>` through run_source() below: one ingest_batch row per
invocation, committed first so a failure is still on record; all rows of one source go in ONE
transaction (all-or-nothing); every upsert is keyed on a natural key, so a re-run rewrites the same
rows instead of adding new ones.  Ingesters only READ their source files.

Source roots default to the AI-server layout and can be pointed elsewhere (tests use fixture trees):
    OMNI_LAB_OC      $OC  (/home/cereal/omni-campaign)
    OMNI_LAB_LEXAR   $L   (/mnt/lexar/omni-campaign)
    OMNI_LAB_MODELS  /home/cereal/models
    OMNI_LAB_NF_LAB  /home/cereal/model-surgery-lab
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import omni_lab_db as db  # noqa: E402

OC = os.environ.get("OMNI_LAB_OC", "/home/cereal/omni-campaign")
LX = os.environ.get("OMNI_LAB_LEXAR", "/mnt/lexar/omni-campaign")
MODELS = os.environ.get("OMNI_LAB_MODELS", "/home/cereal/models")
NF_LAB = os.environ.get("OMNI_LAB_NF_LAB", "/home/cereal/model-surgery-lab")
Q = OC + "/distill/queues"
SERVER_HOST = "ai"          # every source path these ingesters read lives on the AI server

# Status files whose runs are built by a specialised ingester (trainer_runs); status_runs skips them.
TRAINER_OWNED_STATUS = (
    re.compile(r"/distill/fullkd/[^/]+/queue\.status$"),
    re.compile(r"/distill/media_m1/[^/]+/queue\.status$"),
    re.compile(r"/distill/queues/d2\.status$"),
)


def roots_from_env() -> None:
    """Re-read the root overrides (tests set the env after import)."""
    global OC, LX, MODELS, NF_LAB, Q
    OC = os.environ.get("OMNI_LAB_OC", "/home/cereal/omni-campaign")
    LX = os.environ.get("OMNI_LAB_LEXAR", "/mnt/lexar/omni-campaign")
    MODELS = os.environ.get("OMNI_LAB_MODELS", "/home/cereal/models")
    NF_LAB = os.environ.get("OMNI_LAB_NF_LAB", "/home/cereal/model-surgery-lab")
    Q = OC + "/distill/queues"


# ---------------------------------------------------------------------------
# batch bookkeeping
# ---------------------------------------------------------------------------

class Batch:
    """Counters + file shas for one ingest_batch row."""

    def __init__(self, batch_id: int, source: str):
        self.id = batch_id
        self.source = source
        self.seen = 0
        self.written = 0
        self.skipped = 0
        self.file_shas: list[str] = []
        self.notes: list[str] = []

    def saw(self, n: int = 1) -> None:
        self.seen += n

    def wrote(self, n: int = 1) -> None:
        self.written += n

    def skip(self, why: str | None = None) -> None:
        self.skipped += 1
        if why and len(self.notes) < 200:
            self.notes.append(why)

    def prov(self, path: str | Path, host: str = SERVER_HOST) -> tuple[str, str, str]:
        """(src_path, src_sha256, src_host) for a source file; records the sha on the batch."""
        p = Path(path)
        sha = db.file_sha256(p)
        self.file_shas.append(sha)
        return (p.as_posix(), sha, host)


def run_source(module, args, conn_factory=None) -> Batch:
    """Run one ingester module inside its own batch.  Returns the finished Batch (status ok) or raises."""
    conn = conn_factory() if conn_factory else db.connect("rw", app=f"ingest-{module.SOURCE}",
                                                           statement_timeout_ms=600000)
    try:
        parser_path = Path(module.__file__)
        batch_id = db.start_batch(conn, module.SOURCE, parser=f"ingest/{parser_path.name}",
                                  parser_sha256=db.file_sha256(parser_path), host=db.this_host(),
                                  argv=list(sys.argv))
        conn.commit()
        batch = Batch(batch_id, module.SOURCE)
        try:
            module.run(conn, batch, args)
            note = "; ".join(batch.notes[:20]) if batch.notes else None
            db.end_batch(conn, batch_id, "ok", batch.seen, batch.written, batch.skipped, error=note,
                         file_shas=batch.file_shas)
            conn.commit()
        except BaseException as exc:
            conn.rollback()
            db.end_batch(conn, batch_id, "failed", batch.seen, 0, batch.skipped,
                         error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
            conn.commit()
            raise
        return batch
    finally:
        if conn_factory is None:
            conn.close()


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------

def load_json(path: str | Path, limit: int = 64 << 20) -> Any:
    """Parsed JSON or None (missing, too big, unparsable)."""
    try:
        p = Path(path)
        if p.stat().st_size > limit:
            return None
        return json.loads(p.read_bytes().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return None


def read_text(path: str | Path, limit: int = 32 << 20) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def iso_mtime(path: str | Path) -> str | None:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        return None


TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d))\s*(.*)$")


def parse_ts(value: Any) -> datetime | None:
    """ISO-8601 text ('...Z', '+00:00', fractional seconds) -> aware datetime; anything else -> None."""
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def seconds_between(a: datetime | None, b: datetime | None) -> float | None:
    return (b - a).total_seconds() if a and b else None


def sha16_path_key(path: str, host: str = SERVER_HOST) -> str:
    """model_key for an artifact with no ledger id: 'path:' + sha256('<host>:<path>')[:16].
    (A content digest of 1.8 TB of weights is not computed at ingest; the path key is stable per location.)"""
    return "path:" + hashlib.sha256(f"{host}:{path}".encode("utf-8")).hexdigest()[:16]


ST_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
                  "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U32": 4, "U16": 2, "U64": 8}


def safetensors_header(path: str | Path) -> dict | None:
    """The JSON header of one .safetensors file (reads 8 + n bytes, never the tensors)."""
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n > 200 << 20:
                return None
            return json.loads(fh.read(n))
    except (OSError, ValueError, struct.error):
        return None


def header_census(files: Iterable[str | Path]) -> dict:
    """Parameter count, tensor count, dtype histogram and top-level name prefixes over safetensors headers."""
    params = 0
    n_tensors = 0
    dtypes: dict[str, int] = {}
    top: dict[str, int] = {}
    read = 0
    for f in files:
        h = safetensors_header(f)
        if h is None:
            continue
        read += 1
        for name, spec in h.items():
            if name == "__metadata__" or not isinstance(spec, dict):
                continue
            count = 1
            for dim in spec.get("shape", []):
                count *= int(dim)
            params += count
            n_tensors += 1
            dt = str(spec.get("dtype"))
            dtypes[dt] = dtypes.get(dt, 0) + count
            head = name.split(".")[0]
            top[head] = top.get(head, 0) + 1
    return {"files_read": read, "params": params, "n_tensors": n_tensors, "dtype_params": dtypes,
            "top_prefixes": top}


def main_dtype(dtype_params: dict) -> str | None:
    return max(dtype_params, key=dtype_params.get) if dtype_params else None


# ---------------------------------------------------------------------------
# lookups shared by several ingesters
# ---------------------------------------------------------------------------

_HOST_CACHE: dict[int, dict[str, int]] = {}


def host_id(conn, name: str = SERVER_HOST) -> int:
    """host.id; a plain SELECT when the row exists, so a long ingest transaction never holds the host row
    lock (an upsert would block every concurrent ingester until this one commits)."""
    cache = _HOST_CACHE.setdefault(id(conn), {})
    if name not in cache:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM host WHERE name = %s", (name,))
            row = cur.fetchone()
        cache[name] = row[0] if row else db.ensure_host(conn, name, {"address": "192.168.50.222"} if name == "ai" else {})
    return cache[name]


def gpu_for(conn, index: int | None = None, uuid: str | None = None, host: str = SERVER_HOST) -> int | None:
    """gpu.id by UUID (exact) or by index (backfill: index -> card as last seen by the hardware ingester)."""
    hid = host_id(conn, host)
    if uuid:
        return db.gpu_id_by_uuid(conn, hid, uuid)
    if index is not None:
        return db.gpu_id_by_index(conn, hid, int(index))
    return None


def tool_version_for(conn, name: str, version: str | None = None, sha_prefix: str | None = None,
                     path: str | None = None) -> int | None:
    """tool_version.id by exact sha prefix (preferred) or by (tool name, version).  When nothing in the
    pins matches but a full sha is known, a tool_version row is created (pinned=false).  When only a
    short sha prefix is known and `path` names the tool file, the file and its _superseded* copies are
    hashed (read-only) and the one whose sha256 starts with the prefix is recorded."""
    if sha_prefix and len(sha_prefix) < 64 and path:
        found = _tool_file_with_prefix(path, sha_prefix.lower())
        if found:
            existing = _tool_version_lookup(conn, name, version or found[2], found[1], found[0])
            return existing or db.ensure_tool_version(conn, name, version or found[2] or "unknown", found[1], path=found[0])
    return _tool_version_lookup(conn, name, version, sha_prefix, path)


def _tool_file_with_prefix(path: str, prefix: str) -> tuple[str, str, str | None] | None:
    """(path, sha256, version-from-filename) of the tool file or a _superseded copy whose sha starts with prefix."""
    p = Path(path)
    candidates = [p] + sorted(p.parent.glob(f"_superseded*/{p.stem}*{p.suffix}"))
    for c in candidates:
        try:
            sha = db.file_sha256(c)
        except OSError:
            continue
        if sha.startswith(prefix):
            m = re.search(r"_(\d+\.\d+\.\d+[a-z]?)" + re.escape(p.suffix) + "$", c.name)
            return (c.as_posix(), sha, m.group(1) if m else None)
    return None


def _tool_version_lookup(conn, name: str, version: str | None, sha_prefix: str | None,
                         path: str | None = None) -> int | None:
    with conn.cursor() as cur:
        if sha_prefix:
            cur.execute("SELECT tv.id, tv.version FROM tool_version tv JOIN tool t ON t.id = tv.tool_id "
                        "WHERE t.name = %s AND tv.sha256 LIKE %s ORDER BY tv.pinned DESC, tv.id LIMIT 1",
                        (name, sha_prefix.lower() + "%"))
            row = cur.fetchone()
            if row:
                if version and row[1] == "unknown":   # the caller's receipt names the version this sha ran as
                    cur.execute("UPDATE tool_version SET version = %s, path = COALESCE(path, %s) WHERE id = %s",
                                (version, path, row[0]))
                return row[0]
        if version:
            cur.execute("SELECT tv.id FROM tool_version tv JOIN tool t ON t.id = tv.tool_id "
                        "WHERE t.name = %s AND tv.version = %s ORDER BY tv.pinned DESC, tv.id DESC LIMIT 1",
                        (name, version))
            row = cur.fetchone()
            if row:
                return row[0]
    if sha_prefix and len(sha_prefix) == 64:
        return db.ensure_tool_version(conn, name, version or "unknown", sha_prefix.lower(), path=path)
    return None


def lane_id(conn, key: str | None) -> int | None:
    if not key:
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM lane WHERE key = %s", (key,))
        row = cur.fetchone()
        return row[0] if row else None


def set_run_lane(conn, run_id: int, key: str | None) -> None:
    lid = lane_id(conn, key)
    if lid is not None:
        with conn.cursor() as cur:
            cur.execute("UPDATE run SET lane_id = %s WHERE id = %s", (lid, run_id))


def model_id_by_path(conn, path: str | None) -> int | None:
    """The model row whose canonical path is `path` (trailing slash tolerant); ledger rows win."""
    if not path:
        return None
    p = path.rstrip("/")
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM model WHERE rtrim(path, '/') = %s "
                    "ORDER BY (ledger_checkpoint_id IS NOT NULL) DESC, id LIMIT 1", (p,))
        row = cur.fetchone()
        return row[0] if row else None


def put_run_event(conn, run_id: int | None, ts: datetime | None, state: str, line: str,
                  prov: tuple[str, str, str], batch_id: int) -> bool:
    """One parsed status line; (src_path, line) is the natural key.  Re-ingest re-points run_id."""
    with conn.cursor() as cur:
        cur.execute("INSERT INTO run_event (run_id, ts, state, line, src_path, src_sha256, src_host, ingest_batch_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (src_path, line) DO UPDATE SET run_id = COALESCE(EXCLUDED.run_id, run_event.run_id), "
                    "state = EXCLUDED.state, src_sha256 = EXCLUDED.src_sha256, ingest_batch_id = EXCLUDED.ingest_batch_id",
                    (run_id, ts, state, line, prov[0], prov[1], prov[2], batch_id))
        return cur.rowcount == 1


def link_run_dataset(conn, run_id: int, dataset_version_id: int, role: str, detail: dict | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO run_dataset (run_id, dataset_version_id, role, detail) VALUES (%s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (run_id, dataset_version_id, role) DO UPDATE SET detail = EXCLUDED.detail",
                    (run_id, dataset_version_id, role, db.canonical_json(detail or {})))


STEP_LINE_RE = re.compile(r"^STEP (\{.*\})\s*$")
STEP_SCALARS = ("loss", "ce", "k3", "q38", "eos", "ul", "grad_norm", "lr", "tok_per_s", "tokens", "seconds",
                "real_tokens", "peak_vram_mib", "host_rss_gb", "clipped", "clipped_tensors", "bind_seconds")


def step_metrics(entry: dict) -> list[tuple[str, int, float]]:
    """(name, step, value) rows from one trainer STEP record / history entry.  Media steps carry a
    'route'; their loss is also stored as 'loss/<route>' so per-route curves can be pulled directly."""
    step = entry.get("step")
    if not isinstance(step, int):
        return []
    out = []
    for key in STEP_SCALARS:
        v = entry.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out.append((key, step, float(v)))
    route = entry.get("route")
    if isinstance(route, str) and isinstance(entry.get("loss"), (int, float)):
        out.append((f"loss/{route}", step, float(entry["loss"])))
    return out


def log_step_records(path: str | Path) -> list[dict]:
    """Every 'STEP {json}' line of a trainer log, in file order."""
    out = []
    for line in read_text(path).splitlines():
        m = STEP_LINE_RE.match(line.strip())
        if not m:
            continue
        try:
            rec = json.loads(m.group(1))
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out
