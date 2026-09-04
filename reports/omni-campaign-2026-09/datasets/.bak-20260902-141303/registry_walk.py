#!/usr/bin/env python3
"""Bounded checksum walk + Qwen-token census over the curated master registry.

The plan has asked for this walk since June and nobody has run it, for one
reason: the registry lives on the CIFS share at 5-25 MB/s, and every obvious way
of doing it (``find``, ``du``, ``sha256sum -r``, a recursive ``rglob``) walks
directories the share cannot afford and blocks forever on a stalled mount with
nothing written down.  So this tool never discovers a path.  It only follows the
ones the registry itself names:

    MASTER_SHARD_INDEX_POINTER.json
      -> entries[i].pointer          (631 SHARD_INDEX_POINTER.json files)
         -> pointer.shard_index      (shard_index.json)
         -> pointer.shards_jsonl     (shards.jsonl)
            -> shards[j].path        (the jsonl_shards/*.jsonl payloads)

and at every level it compares the sha256 it computed against the sha256 the
registry already published for that file:

    entry.shard_index_sha256     == sha256(shard_index.json)
    pointer.shard_index_sha256   == sha256(shard_index.json)   (both are checked;
                                                                a disagreement
                                                                between the two
                                                                is itself a
                                                                finding)
    pointer.shards_jsonl_sha256  == sha256(shards.jsonl)
    shards[j].sha256             == sha256(<shard file>)
    shards[j].byte_count         == bytes actually read
    shards[j].row_count          == JSONL lines actually parsed

Three properties are load-bearing:

* **Every read is timeout-wrapped.**  ``bounded()`` arms ``ITIMER_REAL`` around
  each 4 MiB ``read()`` and a second budget bounds the whole file, so a hung
  mount costs one entry and a recorded ``"status": "timeout"`` instead of a dead
  job.  A CIFS read wedged in uninterruptible sleep will not take the signal
  until the kernel returns from the syscall; the receipt says so in
  ``timeouts_enforced`` rather than pretending otherwise.
* **It is resumable, append-only.**  One canonical JSON line per finished entry
  is written and fsynced to the progress file.  A re-run reads that file, skips
  the entries already in it, and appends.  The report is a pure function of the
  progress log: ``--mode finalize`` rebuilds it at any time, which is how a
  partial answer is read out of a job that is still running.
* **The shard bytes are read exactly once.**  sha256 and the token census are
  computed in one streaming pass, 4 MiB at a time, because a second pass over
  the share would double the only expensive thing here.

The census tokenizes with the Qwen3.5-4B tokenizer (``tokenizer.json`` through
the ``tokenizers`` Rust binding -- no torch, no transformers import) and reports
per canonical route: rows, characters, instruction/response/total tokens, the
>=4k/>=8k/>=16k/>=32k token row counts (the measurement behind the campaign's
claim that no >=32k-token row exists anywhere), and the longest row by tokens
with its shard and line number.

Both an uncapped count and the loader's view are recorded.
``curated_shards.normalize_row`` compacts every field to 8,192 characters before
a trainer ever sees it; a census that silently inherited that cap would confirm
"no long rows" by construction.  ``tokens_total`` is uncapped;
``tokens_total_loader_view`` is what the shipped loader would hand a trainer.

Route canonicalisation is COPIED from ``chimera_omni.data.curated_shards`` and
then checked against it at runtime (``--curated-shards``): the copy's table is
compared to the source's via ``ast.literal_eval`` and any divergence lands in
``problems`` as ``route_alias_drift``.  A contract copied away from its source
goes stale silently; this one is asserted equal on every run.

Modes
-----
``--mode plan``      read the 631 pointers + shards.jsonl only (no payload
                     bytes) and write declared byte/row totals.  This is the
                     denominator for the completion projection and the first
                     measurement of the registry's on-disk size, which the R1
                     digest lists as unmeasured.
``--mode walk``      the full checksum walk + census.  Runs without a plan file,
                     with a null projection.
``--mode finalize``  rebuild the report JSON from an existing progress file.
``--mode self-test`` build a synthetic registry on local disk (with a corrupted
                     shard, a truncated shard and a missing shard), walk it, and
                     assert every verdict, the resume path and the finalizer.

Stdlib plus ``tokenizers``; canonical JSON and sealing come from the ledger
lane's ``oc_seal`` so this lane cannot fork the campaign's digest convention.
No ORM, no SQLAlchemy, no Pydantic, no torch.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA_REPORT = "neural-forge-omni-registry-walk-v1"
SCHEMA_ENTRY = "neural-forge-omni-registry-walk-entry-v1"
SCHEMA_PLAN = "neural-forge-omni-registry-plan-v1"
SCHEMA_PLAN_ENTRY = "neural-forge-omni-registry-plan-entry-v1"

MASTER_POINTER_DEFAULT = "/mnt/ai_data/omnicoder_2026/datasets/MASTER_SHARD_INDEX_POINTER.json"
TOKENIZER_DEFAULT = "/home/cereal/models/Qwen3.5-4B"
#: The import lane's synced branch checkout: repo root is ``.../src``, so the
#: package sits one ``src`` deeper.  Verified present on the server 2026-09-02.
CURATED_SHARDS_DEFAULT = (
    "/home/cereal/omni-campaign/chimera-omni-import/src/src/chimera_omni/data/curated_shards.py"
)

#: Read block.  4 MiB is one alarm window per ~0.2-0.8 s of share at the rates
#: this campaign has measured, which is the granularity a stall is worth
#: catching at.
CHUNK_BYTES = 4 << 20

#: The loader's compaction limit (``curated_shards._compact``), reproduced so
#: the census can report what a trainer would really receive.
LOADER_COMPACT_LIMIT = 8192

#: Token thresholds the campaign argues about; the last column is the ">=32k
#: evidence corpus does not exist here" claim.
TOKEN_BUCKETS = (4096, 8192, 16384, 32768)

# --- copied verbatim from chimera_omni.data.curated_shards, checked at runtime -
ROUTE_ALIASES: dict[str, str] = {
    "OCR": "ocr",
    "TTS": "tts",
    "speech/sound": "sound",
    "speech_sound": "sound",
    "image_editing": "image",
    "image_text_contract": "image",
    "ocr_text_contract": "ocr",
    "video_media_routing": "video",
    "multimodal_output": "multimodal",
    "multimodal_prompt_answer": "multimodal",
    "media_agent": "tool",
    "agentic": "tool",
    "agentic_planning": "tool",
    "agentic_procedure": "tool",
    "agentic_tooling": "tool",
    "agentic_workflow": "tool",
    "tool_call": "tool",
    "tool_calling": "tool",
    "tool_plan": "tool",
    "workflow": "tool",
    "workflow_plan": "tool",
    "browser_trace": "tool",
    "terminal_trace": "tool",
    "terminal_task": "tool",
    "code_generation": "code",
    "code_text": "code",
    "code_tool_agentic": "code",
    "math_reasoning": "math",
    "reasoning_math": "math",
    "quant_reasoning": "math",
    "quantitative_reasoning": "math",
    "analytical_reasoning": "reasoning",
    "domain_reasoning": "reasoning",
    "multi_step_reasoning": "reasoning",
    "qa_reasoning": "reasoning",
    "reasoning_grid": "reasoning",
    "scientific_reasoning": "reasoning",
    "workflow_reasoning": "reasoning",
    "chemistry_code_tool": "code",
    "finance_code_tool": "code",
    "healthcare_code_tool": "code",
    "science_code_tool": "code",
}
# -----------------------------------------------------------------------------


class BoundedReadTimeout(TimeoutError):
    """A single read did not return inside its window."""


class WalkError(RuntimeError):
    """Refusals that are the operator's problem, not a datum about the share."""


_HAVE_ITIMER = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


def timeouts_enforced() -> bool:
    """True when ``bounded()`` can actually interrupt a blocked read here.

    Windows has no ``SIGALRM`` and a signal only lands on the main thread; the
    self-test runs on the workstation, so this is recorded in the receipt rather
    than assumed.
    """

    return _HAVE_ITIMER and threading.current_thread() is threading.main_thread()


@contextlib.contextmanager
def bounded(seconds: float, what: str):
    """Arm a one-shot real-time alarm around a blocking read."""

    if seconds <= 0 or not timeouts_enforced():
        yield
        return

    def _fire(signum: int, frame: Any) -> None:  # pragma: no cover - timing path
        raise BoundedReadTimeout("%s: no return in %.1fs" % (what, seconds))

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# bounded reads
# --------------------------------------------------------------------------- #


def _read_status(
    status: str, path: str, started: float, total: int, lines: int, note: str
) -> dict[str, Any]:
    elapsed = time.time() - started
    out: dict[str, Any] = {
        "status": status,
        "path": path,
        "bytes_read": int(total),
        "lines_read": int(lines),
        "read_s": round(elapsed, 3),
        "sha256": "",
    }
    if elapsed > 0 and total > 0:
        out["mib_s"] = round(total / (1 << 20) / elapsed, 3)
    if note:
        out["note"] = note
    return out


def read_file_bounded(
    path: str,
    *,
    chunk_timeout_s: float,
    file_timeout_s: float,
    on_line: Callable[[bytes, int], None] | None = None,
    keep_bytes: bool = False,
) -> dict[str, Any]:
    """Read one named file once: sha256, byte count, optional per-line callback.

    Never lists a directory and never follows anything the caller did not name.
    Returns a status dict instead of raising: a missing, corrupt or hung shard is
    a RESULT of this walk, not a crash of it.
    """

    started = time.time()
    digest = hashlib.sha256()
    total = 0
    lines = 0
    buffer = b""
    kept: list[bytes] = []
    try:
        with bounded(chunk_timeout_s, "open " + path):
            handle = open(path, "rb")
    except BoundedReadTimeout as exc:
        return _read_status("timeout", path, started, total, lines, str(exc))
    except FileNotFoundError:
        return _read_status("missing", path, started, total, lines, "file does not exist")
    except OSError as exc:
        return _read_status("error", path, started, total, lines, "%s: %s" % (type(exc).__name__, exc))

    try:
        while True:
            if file_timeout_s > 0 and (time.time() - started) > file_timeout_s:
                return _read_status(
                    "timeout",
                    path,
                    started,
                    total,
                    lines,
                    "file budget %.0fs exhausted after %d bytes" % (file_timeout_s, total),
                )
            try:
                with bounded(chunk_timeout_s, "read " + path):
                    block = handle.read(CHUNK_BYTES)
            except BoundedReadTimeout as exc:
                return _read_status("timeout", path, started, total, lines, str(exc))
            except OSError as exc:
                return _read_status(
                    "error", path, started, total, lines, "%s: %s" % (type(exc).__name__, exc)
                )
            if not block:
                break
            digest.update(block)
            total += len(block)
            if keep_bytes:
                kept.append(block)
            if on_line is not None:
                buffer += block
                if b"\n" in buffer:
                    parts = buffer.split(b"\n")
                    buffer = parts.pop()
                    for part in parts:
                        lines += 1
                        on_line(part, lines)
        if on_line is not None and buffer:
            lines += 1
            on_line(buffer, lines)
    finally:
        handle.close()

    status = _read_status("ok", path, started, total, lines, "")
    status["sha256"] = digest.hexdigest()
    if keep_bytes:
        status["_bytes"] = b"".join(kept)
    return status


def read_json_bounded(
    path: str, *, chunk_timeout_s: float, file_timeout_s: float
) -> tuple[Any, dict[str, Any]]:
    """A bounded read of one named JSON file, returning ``(value_or_None, status)``.

    Metadata files are small (the master pointer is the biggest at 1.6 MB), so
    the bytes are kept from the digest pass and parsed in memory -- the share is
    never read twice for the same file.
    """

    status = read_file_bounded(
        path, chunk_timeout_s=chunk_timeout_s, file_timeout_s=file_timeout_s, keep_bytes=True
    )
    raw = status.pop("_bytes", b"")
    if status["status"] != "ok":
        return None, status
    try:
        value = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        status = dict(status)
        status["status"] = "unparsable"
        status["note"] = "%s: %s" % (type(exc).__name__, exc)
        return None, status
    return value, status


# --------------------------------------------------------------------------- #
# canonical bytes / sealing (from the ledger lane, never re-derived here)
# --------------------------------------------------------------------------- #


def _load_oc_seal(explicit: str | None) -> Any:
    """Import the ledger lane's sealer, or refuse.

    The alternative -- a private copy of ``canonical_json_bytes`` -- is exactly
    the drift the campaign's digest convention cannot afford.
    """

    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve().parent
    candidates.append(here.parent / "ledger")
    candidates.append(Path("/home/cereal/omni-campaign/ledger"))
    for candidate in candidates:
        if (candidate / "oc_seal.py").is_file():
            sys.path.insert(0, str(candidate))
            import oc_seal  # type: ignore

            return oc_seal
    raise WalkError(
        "oc_seal.py not found next to this lane (looked in: %s); pass --ledger-dir"
        % ", ".join(str(c) for c in candidates)
    )


# --------------------------------------------------------------------------- #
# route canonicalisation (copy + runtime parity check against the source)
# --------------------------------------------------------------------------- #


def canonical_route(value: object) -> str:
    route = str(value or "").strip()
    if not route:
        return "unknown"
    if route in ROUTE_ALIASES:
        return ROUTE_ALIASES[route]
    lowered = route.lower().replace("-", "_").replace(" ", "_")
    return ROUTE_ALIASES.get(lowered, lowered)


def check_route_alias_parity(source_path: str) -> dict[str, Any]:
    """Compare our copied ROUTE_ALIASES to the loader's, without importing it.

    ``chimera_omni.data.curated_shards`` is stdlib-only today, but importing it
    drags in the package ``__init__``; the table is extracted with ``ast``
    instead so this check cannot be broken by an unrelated import.
    """

    out: dict[str, Any] = {"source": source_path, "checked": False, "match": None}
    path = Path(source_path)
    if not path.is_file():
        out["note"] = "curated_shards.py not present at this path"
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except (OSError, SyntaxError) as exc:
        out["note"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    found: dict[str, str] | None = None
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "ROUTE_ALIASES":
                try:
                    found = dict(ast.literal_eval(value))
                except ValueError:
                    found = None
    out["checked"] = True
    out["source_sha256"] = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    if found is None:
        out["match"] = False
        out["note"] = "ROUTE_ALIASES not literal-evaluable in the source"
        return out
    out["match"] = found == ROUTE_ALIASES
    if not out["match"]:
        out["only_in_source"] = sorted(set(found) - set(ROUTE_ALIASES))
        out["only_in_copy"] = sorted(set(ROUTE_ALIASES) - set(found))
        out["value_differences"] = sorted(
            key for key in set(found) & set(ROUTE_ALIASES) if found[key] != ROUTE_ALIASES[key]
        )
    return out


def row_raw_route(row: Mapping[str, Any]) -> str:
    """The loader's route extraction, field for field."""

    for key in ("modality", "route", "task", "domain", "category", "declared_target_modality"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("metadata", "meta", "input_json", "target_json"):
        value = row.get(key)
        if isinstance(value, dict):
            for inner in ("modality", "route", "task", "domain", "category", "output_modality"):
                item = value.get(inner)
                if isinstance(item, str) and item.strip():
                    return item.strip()
    modalities = row.get("modalities")
    if isinstance(modalities, list) and modalities:
        return str(modalities[0])
    return "unknown"


def _jsonish(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _normalize_text(value: str) -> str:
    """``curated_shards._compact`` WITHOUT its 8,192-character truncation.

    The truncation is applied separately (``tokens_*_loader_view``) so the census
    can report both the row that exists and the row a trainer would see.
    """

    return " ".join(str(value).replace("\x00", " ").split())


def _messages_text(messages: Any, prefer_roles: tuple[str, ...]) -> str:
    if not isinstance(messages, list):
        return ""
    preferred: list[str] = []
    fallback: list[str] = []
    roles = {role.lower() for role in prefer_roles}
    for item in messages:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = str(item.get("role", "")).lower()
        if role in roles:
            preferred.append(content)
        else:
            fallback.append(content)
    if preferred:
        return "\n".join(preferred)
    return "\n".join(fallback[:2])


def _extract_from_payload(value: Any, prefer: tuple[str, ...]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        found = _messages_text(value.get("messages"), ("user", "assistant", "model", "system"))
        if found:
            return found
        for key in prefer:
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item
        return _jsonish(value)
    if isinstance(value, list):
        return _messages_text(value, ("user", "assistant", "model", "system")) or _jsonish(value)
    return ""


def extract_instruction(row: Mapping[str, Any]) -> str:
    for key in ("instruction", "prompt", "question", "query", "task_prompt", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_text(value)
    for key in ("input_json", "input"):
        found = _extract_from_payload(
            row.get(key), ("prompt", "instruction", "question", "query", "content", "text")
        )
        if found:
            return _normalize_text(found)
    found = _messages_text(row.get("messages"), ("user", "system"))
    if found:
        return _normalize_text(found)
    return _normalize_text(_jsonish(row.get("input_json") or row.get("messages") or ""))


def extract_response(row: Mapping[str, Any]) -> str:
    for key in ("response", "answer", "output", "completion", "target"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_text(value)
    for key in ("target_json", "target", "output_json"):
        found = _extract_from_payload(
            row.get(key),
            ("response", "answer", "output", "completion", "content", "transcript", "text"),
        )
        if found:
            return _normalize_text(found)
    found = _messages_text(row.get("messages"), ("assistant", "model"))
    if found:
        return _normalize_text(found)
    return _normalize_text(_jsonish(row.get("target_json") or ""))


def row_is_trainable(row: Mapping[str, Any]) -> bool:
    """``normalize_row``'s two policy refusals, verbatim."""

    if row.get("training_allowed") is False:
        return False
    return "do_not_train" not in str(row.get("use_policy", "")).lower()


# --------------------------------------------------------------------------- #
# token census
# --------------------------------------------------------------------------- #


class QwenCounter:
    """Batched token counting through ``tokenizers``; no torch, no transformers."""

    def __init__(self, tokenizer_dir: str, *, batch: int = 256) -> None:
        from tokenizers import Tokenizer  # local import: absent on a bare python3

        path = Path(tokenizer_dir)
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.is_file():
            raise WalkError("tokenizer.json not found at %s" % path)
        self.tokenizer_file = str(path)
        self.tokenizer_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.tokenizer = Tokenizer.from_file(str(path))
        # A census must not inherit a model_max_length truncation from the file.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.batch = int(batch)
        self.vocab_size = int(self.tokenizer.get_vocab_size())

    def count_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        out: list[int] = []
        for start in range(0, len(texts), self.batch):
            chunk = texts[start : start + self.batch]
            for encoded in self.tokenizer.encode_batch(chunk, add_special_tokens=False):
                out.append(len(encoded.ids))
        return out


def _new_route_bucket() -> dict[str, Any]:
    bucket: dict[str, Any] = {
        "rows": 0,
        "chars_instruction": 0,
        "chars_response": 0,
        "tokens_instruction": 0,
        "tokens_response": 0,
        "tokens_total": 0,
        "tokens_total_loader_view": 0,
        "rows_truncated_by_loader": 0,
        "max_tokens": 0,
        "max_tokens_row": None,
    }
    for threshold in TOKEN_BUCKETS:
        bucket["rows_ge_%d" % threshold] = 0
    return bucket


class Census:
    """Per-route accumulator; one instance per entry, merged by the finalizer."""

    def __init__(self) -> None:
        self.by_route: dict[str, dict[str, Any]] = {}
        self.by_raw_route: dict[str, int] = {}
        self.rows_total = 0
        self.rows_skipped_policy = 0
        self.rows_without_text = 0
        self.rows_unparsable = 0

    def add(
        self,
        *,
        route: str,
        raw_route: str,
        chars_instruction: int,
        chars_response: int,
        tokens_instruction: int,
        tokens_response: int,
        tokens_loader_view: int,
        truncated: bool,
        where: Mapping[str, Any],
    ) -> None:
        bucket = self.by_route.setdefault(route, _new_route_bucket())
        total = tokens_instruction + tokens_response
        bucket["rows"] += 1
        bucket["chars_instruction"] += chars_instruction
        bucket["chars_response"] += chars_response
        bucket["tokens_instruction"] += tokens_instruction
        bucket["tokens_response"] += tokens_response
        bucket["tokens_total"] += total
        bucket["tokens_total_loader_view"] += tokens_loader_view
        if truncated:
            bucket["rows_truncated_by_loader"] += 1
        for threshold in TOKEN_BUCKETS:
            if total >= threshold:
                bucket["rows_ge_%d" % threshold] += 1
        if total > bucket["max_tokens"]:
            bucket["max_tokens"] = total
            bucket["max_tokens_row"] = dict(where)
        self.by_raw_route[raw_route] = self.by_raw_route.get(raw_route, 0) + 1
        self.rows_total += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_total": self.rows_total,
            "rows_skipped_policy": self.rows_skipped_policy,
            "rows_without_text": self.rows_without_text,
            "rows_unparsable": self.rows_unparsable,
            "by_route": {route: self.by_route[route] for route in sorted(self.by_route)},
            "by_raw_route": {route: self.by_raw_route[route] for route in sorted(self.by_raw_route)},
        }


def merge_census(into: dict[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Sum two census dicts.  Used by the finalizer over the progress log."""

    for key in ("rows_total", "rows_skipped_policy", "rows_without_text", "rows_unparsable"):
        into[key] = int(into.get(key, 0)) + int(other.get(key, 0))
    dst_routes = into.setdefault("by_route", {})
    for route, bucket in (other.get("by_route") or {}).items():
        target = dst_routes.setdefault(route, _new_route_bucket())
        for key, value in bucket.items():
            if key == "max_tokens":
                if int(value) > int(target["max_tokens"]):
                    target["max_tokens"] = int(value)
                    target["max_tokens_row"] = bucket.get("max_tokens_row")
            elif key == "max_tokens_row":
                continue
            else:
                target[key] = int(target.get(key, 0)) + int(value)
    dst_raw = into.setdefault("by_raw_route", {})
    for route, count in (other.get("by_raw_route") or {}).items():
        dst_raw[route] = int(dst_raw.get(route, 0)) + int(count)
    return into


# --------------------------------------------------------------------------- #
# progress log
# --------------------------------------------------------------------------- #


class ProgressLog:
    """Append-only JSONL, one canonical line per finished entry, fsynced."""

    def __init__(self, path: str, canonical_json_bytes: Callable[[Any], bytes]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._canon = canonical_json_bytes

    def done_indexes(self) -> dict[int, str]:
        """Entry index -> dataset_id for every record already written."""

        done: dict[int, str] = {}
        if not self.path.is_file():
            return done
        with self.path.open("rb") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    # A torn last line from a killed process: ignore it; the
                    # entry is simply re-walked.
                    continue
                if isinstance(record, dict) and "entry_index" in record:
                    done[int(record["entry_index"])] = str(record.get("dataset_id", ""))
        return done

    def records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.path.is_file():
            return out
        with self.path.open("rb") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                if isinstance(record, dict):
                    out.append(record)
        return out

    def append(self, record: Mapping[str, Any]) -> None:
        line = self._canon(record) + b"\n"
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def write_json_atomic(path: str, payload: Mapping[str, Any], canonical: Callable[[Any], bytes]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_bytes(canonical(payload))
    os.replace(partial, target)
    return target


# --------------------------------------------------------------------------- #
# the walk
# --------------------------------------------------------------------------- #


def _sha_verdict(expected: Any, measured: str) -> str:
    expected_text = str(expected or "")
    if not expected_text:
        return "no_expected_value"
    if not measured:
        return "not_measured"
    return "match" if expected_text.lower() == measured.lower() else "MISMATCH"


def walk_entry(
    entry: Mapping[str, Any],
    entry_index: int,
    *,
    counter: QwenCounter | None,
    chunk_timeout_s: float,
    file_timeout_s: float,
    hash_shards: bool,
) -> dict[str, Any]:
    """Follow one registry entry to its shard bytes.  Never raises on share I/O."""

    started = time.time()
    problems: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "schema": SCHEMA_ENTRY,
        "entry_index": int(entry_index),
        "dataset_id": str(entry.get("dataset_id", "")),
        "label": str(entry.get("label", "")),
        "training_status": str(entry.get("training_status", "")),
        "declared": {
            "pointer": str(entry.get("pointer", "")),
            "shard_count": entry.get("shard_count"),
            "total_rows": entry.get("total_rows"),
            "shard_index_sha256": str(entry.get("shard_index_sha256", "")),
            "pointer_sha256": str(entry.get("pointer_sha256", "")),
            "modality_counts": dict(entry.get("modality_counts") or {}),
        },
        "started_at": now_utc(),
    }

    pointer_path = str(entry.get("pointer", ""))
    if not pointer_path:
        problems.append({"kind": "entry_without_pointer", "detail": "entry has no pointer field"})
        record["problems"] = problems
        record["wall_s"] = round(time.time() - started, 3)
        record["finished_at"] = now_utc()
        return record

    pointer, pointer_status = read_json_bounded(
        pointer_path, chunk_timeout_s=chunk_timeout_s, file_timeout_s=file_timeout_s
    )
    if entry.get("pointer_sha256"):
        pointer_status["expected_sha256"] = str(entry.get("pointer_sha256"))
        pointer_status["verdict"] = _sha_verdict(entry.get("pointer_sha256"), pointer_status.get("sha256", ""))
        if pointer_status["verdict"] == "MISMATCH":
            problems.append({"kind": "pointer_sha_mismatch", "detail": pointer_path})
    record["pointer_file"] = pointer_status
    if pointer is None:
        problems.append({"kind": "pointer_unreadable", "detail": pointer_status.get("note", pointer_status["status"])})
        record["problems"] = problems
        record["wall_s"] = round(time.time() - started, 3)
        record["finished_at"] = now_utc()
        return record

    # shard_index.json -- the sha the master entry publishes.
    shard_index_path = str(pointer.get("shard_index", ""))
    if shard_index_path:
        status = read_file_bounded(
            shard_index_path, chunk_timeout_s=chunk_timeout_s, file_timeout_s=file_timeout_s
        )
        status["expected_sha256_entry"] = str(entry.get("shard_index_sha256", ""))
        status["expected_sha256_pointer"] = str(pointer.get("shard_index_sha256", ""))
        status["verdict_entry"] = _sha_verdict(entry.get("shard_index_sha256"), status.get("sha256", ""))
        status["verdict_pointer"] = _sha_verdict(pointer.get("shard_index_sha256"), status.get("sha256", ""))
        status["entry_pointer_sha_agree"] = str(entry.get("shard_index_sha256", "")).lower() == str(
            pointer.get("shard_index_sha256", "")
        ).lower()
        record["shard_index"] = status
        if status["verdict_entry"] == "MISMATCH" or status["verdict_pointer"] == "MISMATCH":
            problems.append({"kind": "shard_index_sha_mismatch", "detail": shard_index_path})
        if not status["entry_pointer_sha_agree"]:
            problems.append({"kind": "shard_index_sha_disagreement", "detail": shard_index_path})
        if status["status"] != "ok":
            problems.append({"kind": "shard_index_unreadable", "detail": status.get("note", status["status"])})
    else:
        problems.append({"kind": "pointer_without_shard_index", "detail": pointer_path})

    # shards.jsonl -- the list of payload shards, and its own published sha.
    shards_jsonl_path = str(pointer.get("shards_jsonl", ""))
    shard_rows: list[dict[str, Any]] = []
    parse_errors = 0

    def _collect(line: bytes, _number: int) -> None:
        nonlocal parse_errors
        text = line.strip()
        if not text:
            return
        try:
            value = json.loads(text.decode("utf-8", errors="replace"))
        except ValueError:
            parse_errors += 1
            return
        if isinstance(value, dict):
            shard_rows.append(value)

    if shards_jsonl_path:
        status = read_file_bounded(
            shards_jsonl_path,
            chunk_timeout_s=chunk_timeout_s,
            file_timeout_s=file_timeout_s,
            on_line=_collect,
        )
        status["expected_sha256"] = str(pointer.get("shards_jsonl_sha256", ""))
        status["verdict"] = _sha_verdict(pointer.get("shards_jsonl_sha256"), status.get("sha256", ""))
        status["parse_errors"] = parse_errors
        status["shard_rows"] = len(shard_rows)
        record["shards_jsonl"] = status
        if status["verdict"] == "MISMATCH":
            problems.append({"kind": "shards_jsonl_sha_mismatch", "detail": shards_jsonl_path})
        if status["status"] != "ok":
            problems.append({"kind": "shards_jsonl_unreadable", "detail": status.get("note", status["status"])})
    else:
        problems.append({"kind": "pointer_without_shards_jsonl", "detail": pointer_path})

    declared_shard_count = entry.get("shard_count")
    if declared_shard_count is not None and len(shard_rows) != int(declared_shard_count):
        problems.append(
            {
                "kind": "shard_count_disagreement",
                "detail": "entry says %s, shards.jsonl lists %d" % (declared_shard_count, len(shard_rows)),
            }
        )

    census = Census()
    shard_results: list[dict[str, Any]] = []
    declared_bytes = 0
    read_bytes = 0
    dataset_id = str(entry.get("dataset_id", ""))

    for shard in shard_rows:
        shard_path = str(shard.get("path", ""))
        declared_bytes += int(shard.get("byte_count") or 0)
        result: dict[str, Any] = {
            "shard_id": shard.get("shard_id"),
            "filename": str(shard.get("filename", "")),
            "path": shard_path,
            "declared_bytes": int(shard.get("byte_count") or 0),
            "declared_rows": int(shard.get("row_count") or 0),
            "expected_sha256": str(shard.get("sha256", "")),
        }
        if not hash_shards:
            result["status"] = "skipped"
            shard_results.append(result)
            continue

        pending_texts: list[str] = []
        pending_meta: list[dict[str, Any]] = []

        def _row(line: bytes, number: int) -> None:
            # Always installed, so `lines_read` verifies the declared row_count
            # even when the census is off; the parse only happens with a counter.
            if counter is None:
                return
            text = line.strip()
            if not text:
                return
            try:
                row = json.loads(text.decode("utf-8", errors="replace"))
            except ValueError:
                census.rows_unparsable += 1
                return
            if not isinstance(row, dict):
                census.rows_unparsable += 1
                return
            if not row_is_trainable(row):
                census.rows_skipped_policy += 1
                return
            instruction = extract_instruction(row)
            response = extract_response(row)
            if not instruction and not response:
                census.rows_without_text += 1
                return
            pending_texts.append(instruction)
            pending_texts.append(response)
            pending_meta.append(
                {
                    "raw_route": row_raw_route(row),
                    "chars_instruction": len(instruction),
                    "chars_response": len(response),
                    "line_number": number,
                    "record_id": str(row.get("record_id") or row.get("source_id") or ""),
                }
            )

        status = read_file_bounded(
            shard_path,
            chunk_timeout_s=chunk_timeout_s,
            file_timeout_s=file_timeout_s,
            on_line=_row,
        )
        read_bytes += int(status.get("bytes_read") or 0)
        result.update(
            {
                "status": status["status"],
                "sha256": status.get("sha256", ""),
                "bytes_read": status.get("bytes_read", 0),
                "rows_read": status.get("lines_read", 0),
                "read_s": status.get("read_s", 0.0),
                "mib_s": status.get("mib_s"),
                "verdict": _sha_verdict(shard.get("sha256"), status.get("sha256", "")),
            }
        )
        if status.get("note"):
            result["note"] = status["note"]
        if result["verdict"] == "MISMATCH":
            problems.append({"kind": "shard_sha_mismatch", "detail": shard_path})
        if status["status"] != "ok":
            problems.append({"kind": "shard_unreadable", "detail": "%s: %s" % (shard_path, status["status"])})
        if status["status"] == "ok" and result["declared_bytes"] and result["declared_bytes"] != status["bytes_read"]:
            problems.append(
                {
                    "kind": "shard_byte_count_disagreement",
                    "detail": "%s: declared %d, read %d"
                    % (shard_path, result["declared_bytes"], status["bytes_read"]),
                }
            )

        if counter is not None and pending_meta:
            counts = counter.count_many(pending_texts)
            long_texts: list[str] = []
            long_index: list[int] = []
            for position, meta in enumerate(pending_meta):
                instruction_tokens = counts[2 * position]
                response_tokens = counts[2 * position + 1]
                truncated = (
                    meta["chars_instruction"] > LOADER_COMPACT_LIMIT
                    or meta["chars_response"] > LOADER_COMPACT_LIMIT
                )
                if truncated:
                    long_index.append(position)
                    long_texts.append(pending_texts[2 * position][:LOADER_COMPACT_LIMIT])
                    long_texts.append(pending_texts[2 * position + 1][:LOADER_COMPACT_LIMIT])
                meta["tokens_instruction"] = instruction_tokens
                meta["tokens_response"] = response_tokens
                meta["truncated"] = truncated
            loader_counts = counter.count_many(long_texts) if long_texts else []
            for slot, position in enumerate(long_index):
                pending_meta[position]["tokens_loader_view"] = (
                    loader_counts[2 * slot] + loader_counts[2 * slot + 1]
                )
            for meta in pending_meta:
                total_tokens = meta["tokens_instruction"] + meta["tokens_response"]
                census.add(
                    route=canonical_route(meta["raw_route"]),
                    raw_route=str(meta["raw_route"]),
                    chars_instruction=meta["chars_instruction"],
                    chars_response=meta["chars_response"],
                    tokens_instruction=meta["tokens_instruction"],
                    tokens_response=meta["tokens_response"],
                    tokens_loader_view=int(meta.get("tokens_loader_view", total_tokens)),
                    truncated=bool(meta["truncated"]),
                    where={
                        "dataset_id": dataset_id,
                        "shard": result["filename"],
                        "line_number": meta["line_number"],
                        "record_id": meta["record_id"],
                    },
                )
        result["rows_censused"] = len(pending_meta)
        if (
            status["status"] == "ok"
            and result["declared_rows"]
            and status.get("lines_read", 0) != result["declared_rows"]
        ):
            problems.append(
                {
                    "kind": "shard_row_count_disagreement",
                    "detail": "%s: declared %d, read %d"
                    % (shard_path, result["declared_rows"], status.get("lines_read", 0)),
                }
            )
        shard_results.append(result)

    record["shards"] = shard_results
    record["declared_bytes"] = declared_bytes
    record["bytes_read"] = read_bytes
    record["census"] = census.as_dict()
    record["problems"] = problems
    record["wall_s"] = round(time.time() - started, 3)
    record["finished_at"] = now_utc()
    return record


def plan_entry(
    entry: Mapping[str, Any],
    entry_index: int,
    *,
    chunk_timeout_s: float,
    file_timeout_s: float,
) -> dict[str, Any]:
    """Metadata-only pass: how many bytes the payload walk will have to read."""

    started = time.time()
    record: dict[str, Any] = {
        "schema": SCHEMA_PLAN_ENTRY,
        "entry_index": int(entry_index),
        "dataset_id": str(entry.get("dataset_id", "")),
        "training_status": str(entry.get("training_status", "")),
        "declared_shard_count": entry.get("shard_count"),
        "declared_total_rows": entry.get("total_rows"),
        "problems": [],
    }
    pointer_path = str(entry.get("pointer", ""))
    pointer, status = read_json_bounded(
        pointer_path, chunk_timeout_s=chunk_timeout_s, file_timeout_s=file_timeout_s
    )
    record["pointer_status"] = status["status"]
    record["pointer_read_s"] = status["read_s"]
    if pointer is None:
        record["problems"].append({"kind": "pointer_unreadable", "detail": pointer_path})
        record["wall_s"] = round(time.time() - started, 3)
        return record
    shard_rows: list[dict[str, Any]] = []

    def _collect(line: bytes, _number: int) -> None:
        text = line.strip()
        if not text:
            return
        try:
            value = json.loads(text.decode("utf-8", errors="replace"))
        except ValueError:
            return
        if isinstance(value, dict):
            shard_rows.append(value)

    shards_jsonl_path = str(pointer.get("shards_jsonl", ""))
    if shards_jsonl_path:
        js = read_file_bounded(
            shards_jsonl_path,
            chunk_timeout_s=chunk_timeout_s,
            file_timeout_s=file_timeout_s,
            on_line=_collect,
        )
        record["shards_jsonl_status"] = js["status"]
        if js["status"] != "ok":
            record["problems"].append({"kind": "shards_jsonl_unreadable", "detail": shards_jsonl_path})
    else:
        record["problems"].append({"kind": "pointer_without_shards_jsonl", "detail": pointer_path})

    record["shards_listed"] = len(shard_rows)
    record["payload_bytes"] = sum(int(row.get("byte_count") or 0) for row in shard_rows)
    record["payload_rows"] = sum(int(row.get("row_count") or 0) for row in shard_rows)
    record["wall_s"] = round(time.time() - started, 3)
    return record


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #


def build_report(
    records: list[dict[str, Any]],
    *,
    context: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
    complete: bool,
) -> dict[str, Any]:
    """The report is a pure function of the progress log plus run context."""

    entries_done = len(records)
    census: dict[str, Any] = {}
    problems: list[dict[str, Any]] = []
    shards_ok = shards_mismatch = shards_unreadable = shards_total = 0
    shard_index_ok = shard_index_mismatch = 0
    shards_jsonl_ok = shards_jsonl_mismatch = 0
    declared_bytes = bytes_read = 0
    read_seconds = 0.0
    wall_seconds = 0.0
    rows_declared = 0
    per_entry: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda item: int(item.get("entry_index", 0))):
        merge_census(census, record.get("census") or {})
        for problem in record.get("problems") or []:
            problems.append(
                {
                    "entry_index": record.get("entry_index"),
                    "dataset_id": record.get("dataset_id"),
                    "kind": problem.get("kind"),
                    "detail": problem.get("detail"),
                }
            )
        index = record.get("shard_index") or {}
        if index:
            if index.get("verdict_entry") == "match" and index.get("verdict_pointer") == "match":
                shard_index_ok += 1
            elif "MISMATCH" in (index.get("verdict_entry"), index.get("verdict_pointer")):
                shard_index_mismatch += 1
        jsonl = record.get("shards_jsonl") or {}
        if jsonl:
            if jsonl.get("verdict") == "match":
                shards_jsonl_ok += 1
            elif jsonl.get("verdict") == "MISMATCH":
                shards_jsonl_mismatch += 1
        entry_ok = entry_bad = 0
        for shard in record.get("shards") or []:
            shards_total += 1
            if shard.get("status") == "ok" and shard.get("verdict") == "match":
                shards_ok += 1
                entry_ok += 1
            elif shard.get("verdict") == "MISMATCH":
                shards_mismatch += 1
                entry_bad += 1
            elif shard.get("status") not in ("ok", "skipped"):
                shards_unreadable += 1
                entry_bad += 1
            read_seconds += float(shard.get("read_s") or 0.0)
        declared_bytes += int(record.get("declared_bytes") or 0)
        bytes_read += int(record.get("bytes_read") or 0)
        wall_seconds += float(record.get("wall_s") or 0.0)
        rows_declared += int((record.get("declared") or {}).get("total_rows") or 0)
        per_entry.append(
            {
                "entry_index": record.get("entry_index"),
                "dataset_id": record.get("dataset_id"),
                "training_status": record.get("training_status"),
                "shards_ok": entry_ok,
                "shards_bad": entry_bad,
                "rows_censused": int((record.get("census") or {}).get("rows_total") or 0),
                "bytes_read": int(record.get("bytes_read") or 0),
                "wall_s": record.get("wall_s"),
                "problem_kinds": sorted({str(p.get("kind")) for p in (record.get("problems") or [])}),
            }
        )

    rate = round(bytes_read / (1 << 20) / read_seconds, 3) if read_seconds > 0 and bytes_read else None
    projection: dict[str, Any] = {
        "measured_mib_s": rate,
        "entries_done": entries_done,
        "bytes_read": bytes_read,
        "read_seconds": round(read_seconds, 2),
        "wall_seconds": round(wall_seconds, 2),
    }
    total_entries = int(context.get("entry_count") or 0)
    plan_bytes = int((plan or {}).get("payload_bytes_total") or 0)
    if plan_bytes and rate:
        remaining = max(plan_bytes - bytes_read, 0)
        projection["plan_payload_bytes_total"] = plan_bytes
        projection["projected_remaining_s"] = round(remaining / (1 << 20) / rate, 1)
        projection["projected_total_s"] = round(plan_bytes / (1 << 20) / rate, 1)
        projection["projected_total_h"] = round(plan_bytes / (1 << 20) / rate / 3600.0, 2)
        projection["basis"] = "plan payload byte total / measured share read rate"
    elif entries_done and total_entries and wall_seconds > 0:
        per_entry_s = wall_seconds / entries_done
        projection["projected_total_s"] = round(per_entry_s * total_entries, 1)
        projection["projected_total_h"] = round(per_entry_s * total_entries / 3600.0, 2)
        projection["basis"] = "mean wall seconds per finished entry x entry_count (no plan file)"
    else:
        projection["basis"] = "insufficient data"

    return {
        "schema": SCHEMA_REPORT,
        "campaign": "omni-2026-09",
        "lane": "datasets",
        "complete": bool(complete),
        "generated_at": now_utc(),
        "context": dict(context),
        "totals": {
            "entries_in_registry": total_entries,
            "entries_walked": entries_done,
            "shards_seen": shards_total,
            "shards_sha_match": shards_ok,
            "shards_sha_mismatch": shards_mismatch,
            "shards_unreadable": shards_unreadable,
            "shard_index_match": shard_index_ok,
            "shard_index_mismatch": shard_index_mismatch,
            "shards_jsonl_match": shards_jsonl_ok,
            "shards_jsonl_mismatch": shards_jsonl_mismatch,
            "declared_payload_bytes_walked": declared_bytes,
            "bytes_read": bytes_read,
            "rows_declared_by_entries": rows_declared,
        },
        "projection": projection,
        "census": census,
        "problems": problems,
        "entries": per_entry,
    }


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #


def run_walk(args: argparse.Namespace, oc_seal: Any) -> int:
    master, master_status = read_json_bounded(
        args.pointer, chunk_timeout_s=args.chunk_timeout_s, file_timeout_s=args.file_timeout_s
    )
    if master is None:
        raise WalkError("master pointer unreadable: %s" % master_status.get("note", master_status["status"]))
    entries = list(master.get("entries") or [])

    counter: QwenCounter | None = None
    tokenizer_info: dict[str, Any] = {"enabled": False}
    if not args.no_tokens:
        counter = QwenCounter(args.tokenizer, batch=args.token_batch)
        tokenizer_info = {
            "enabled": True,
            "tokenizer_file": counter.tokenizer_file,
            "tokenizer_sha256": counter.tokenizer_sha256,
            "vocab_size": counter.vocab_size,
            "add_special_tokens": False,
            "loader_compact_limit": LOADER_COMPACT_LIMIT,
        }

    plan_payload: dict[str, Any] | None = None
    if args.plan and Path(args.plan).is_file():
        plan_payload = json.loads(Path(args.plan).read_text(encoding="utf-8"))

    context = {
        "mode": args.mode,
        "pointer": args.pointer,
        "pointer_sha256": master_status.get("sha256", ""),
        "pointer_bytes": master_status.get("bytes_read", 0),
        "pointer_read_s": master_status.get("read_s"),
        "schema_version": str(master.get("schema_version", "")),
        "registry_updated_at": str(master.get("updated_at", "")),
        "entry_count": int(master.get("entry_count") or len(entries)),
        "total_rows": int(master.get("total_rows") or 0),
        "total_shards": int(master.get("total_shards") or 0),
        "rows_by_state": dict(master.get("rows_by_state") or {}),
        "last_append_tag": str(master.get("last_append_tag", "")),
        "master_shard_collection_sha256": str(master.get("master_shard_collection_sha256", "")),
        "tokenizer": tokenizer_info,
        "route_alias_parity": check_route_alias_parity(args.curated_shards),
        "timeouts_enforced": timeouts_enforced(),
        "chunk_timeout_s": args.chunk_timeout_s,
        "file_timeout_s": args.file_timeout_s,
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "python": sys.version.split()[0],
        "started_at": now_utc(),
        "progress_file": str(Path(args.progress).resolve()),
    }
    if context["route_alias_parity"].get("match") is False:
        # Recorded, not fatal: the census is still a measurement, but the reader
        # must know the copy and the loader disagree.
        print("WARNING route_alias_drift: copy differs from %s" % args.curated_shards, flush=True)

    log = ProgressLog(args.progress, oc_seal.canonical_json_bytes)
    done = log.done_indexes()
    if done and not args.resume:
        raise WalkError(
            "%s already holds %d finished entries; pass --resume to continue or point --progress elsewhere"
            % (args.progress, len(done))
        )
    print(
        "entries=%d already_done=%d tokens=%s timeouts_enforced=%s"
        % (len(entries), len(done), not args.no_tokens, timeouts_enforced()),
        flush=True,
    )

    started = time.time()
    walked = 0
    for index, entry in enumerate(entries):
        if index < args.start_index:
            continue
        if args.max_entries and walked >= args.max_entries:
            break
        if index in done:
            continue
        record = walk_entry(
            entry,
            index,
            counter=counter,
            chunk_timeout_s=args.chunk_timeout_s,
            file_timeout_s=args.file_timeout_s,
            hash_shards=not args.no_shard_hashes,
        )
        log.append(record)
        walked += 1
        print(
            "[%d/%d] %s shards=%d bytes=%d rows=%d problems=%d %.1fs"
            % (
                index + 1,
                len(entries),
                record.get("dataset_id", ""),
                len(record.get("shards") or []),
                int(record.get("bytes_read") or 0),
                int((record.get("census") or {}).get("rows_total") or 0),
                len(record.get("problems") or []),
                float(record.get("wall_s") or 0.0),
            ),
            flush=True,
        )
        if args.checkpoint_every and walked % args.checkpoint_every == 0:
            report = build_report(log.records(), context=context, plan=plan_payload, complete=False)
            write_json_atomic(args.out, report, oc_seal.canonical_json_bytes)

    context["finished_at"] = now_utc()
    context["wall_s"] = round(time.time() - started, 2)
    records = log.records()
    complete = len(records) >= len(entries)
    report = build_report(records, context=context, plan=plan_payload, complete=complete)
    write_json_atomic(args.out, report, oc_seal.canonical_json_bytes)
    print(
        "WROTE %s complete=%s entries_walked=%d shards_match=%d mismatch=%d unreadable=%d"
        % (
            args.out,
            complete,
            report["totals"]["entries_walked"],
            report["totals"]["shards_sha_match"],
            report["totals"]["shards_sha_mismatch"],
            report["totals"]["shards_unreadable"],
        ),
        flush=True,
    )
    return 0


def run_plan(args: argparse.Namespace, oc_seal: Any) -> int:
    master, master_status = read_json_bounded(
        args.pointer, chunk_timeout_s=args.chunk_timeout_s, file_timeout_s=args.file_timeout_s
    )
    if master is None:
        raise WalkError("master pointer unreadable: %s" % master_status.get("note", master_status["status"]))
    entries = list(master.get("entries") or [])
    log = ProgressLog(args.progress, oc_seal.canonical_json_bytes)
    done = log.done_indexes()
    started = time.time()
    for index, entry in enumerate(entries):
        if index in done:
            continue
        record = plan_entry(
            entry, index, chunk_timeout_s=args.chunk_timeout_s, file_timeout_s=args.file_timeout_s
        )
        log.append(record)
        if (index + 1) % 25 == 0:
            print("[plan %d/%d] %.1fs" % (index + 1, len(entries), time.time() - started), flush=True)
    records = log.records()
    payload_bytes = sum(int(item.get("payload_bytes") or 0) for item in records)
    payload_rows = sum(int(item.get("payload_rows") or 0) for item in records)
    shards_listed = sum(int(item.get("shards_listed") or 0) for item in records)
    problems = [
        {"entry_index": item.get("entry_index"), "dataset_id": item.get("dataset_id"), **problem}
        for item in records
        for problem in (item.get("problems") or [])
    ]
    plan = {
        "schema": SCHEMA_PLAN,
        "campaign": "omni-2026-09",
        "generated_at": now_utc(),
        "pointer": args.pointer,
        "pointer_sha256": master_status.get("sha256", ""),
        "registry_updated_at": str(master.get("updated_at", "")),
        "entry_count": int(master.get("entry_count") or len(entries)),
        "entries_planned": len(records),
        "declared_total_rows": int(master.get("total_rows") or 0),
        "declared_total_shards": int(master.get("total_shards") or 0),
        "shards_listed": shards_listed,
        "payload_bytes_total": payload_bytes,
        "payload_gib_total": round(payload_bytes / (1 << 30), 3),
        "payload_rows_total": payload_rows,
        "metadata_wall_s": round(time.time() - started, 2),
        "problems": problems,
    }
    write_json_atomic(args.out, plan, oc_seal.canonical_json_bytes)
    print(
        "PLAN %s entries=%d shards=%d payload=%.3f GiB rows=%d in %.1fs"
        % (
            args.out,
            plan["entries_planned"],
            shards_listed,
            plan["payload_gib_total"],
            payload_rows,
            plan["metadata_wall_s"],
        ),
        flush=True,
    )
    return 0


def run_finalize(args: argparse.Namespace, oc_seal: Any) -> int:
    log = ProgressLog(args.progress, oc_seal.canonical_json_bytes)
    records = log.records()
    if not records:
        raise WalkError("no progress records at %s" % args.progress)
    plan_payload = None
    if args.plan and Path(args.plan).is_file():
        plan_payload = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    context: dict[str, Any] = {
        "mode": "finalize",
        "pointer": args.pointer,
        "progress_file": str(Path(args.progress).resolve()),
        "generated_from": "progress log only",
        "entry_count": args.entry_count,
    }
    report = build_report(records, context=context, plan=plan_payload, complete=False)
    write_json_atomic(args.out, report, oc_seal.canonical_json_bytes)
    print("WROTE %s from %d progress records" % (args.out, len(records)), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #


def _fixture(root: Path) -> tuple[str, dict[str, Any]]:
    """A synthetic registry with one clean entry and one damaged entry."""

    import random

    root.mkdir(parents=True, exist_ok=True)
    entries = []
    truth: dict[str, Any] = {}
    rng = random.Random(20260902)
    routes = ["text", "code", "OCR", "agentic_workflow", "long_context"]

    for dataset_index in range(2):
        dataset_id = "synth_ds_%d" % dataset_index
        base = root / "shards" / dataset_id
        (base / "jsonl_shards").mkdir(parents=True, exist_ok=True)
        shard_rows = []
        for shard_index in range(2):
            name = "%s-%06d.jsonl" % (dataset_id, shard_index)
            path = base / "jsonl_shards" / name
            lines = []
            for row_index in range(12):
                route = routes[(shard_index * 12 + row_index) % len(routes)]
                row = {
                    "modality": route,
                    "instruction": "please " + ("token " * rng.randint(1, 40)).strip(),
                    "response": "sure " + ("word " * rng.randint(1, 60)).strip(),
                    "record_id": "%s-%d-%d" % (dataset_id, shard_index, row_index),
                }
                if row_index == 3:
                    row["training_allowed"] = False
                if row_index == 4:
                    row["use_policy"] = "do_not_train"
                if row_index == 5:
                    row["instruction"] = "x " * (LOADER_COMPACT_LIMIT // 2 + 200)
                lines.append(json.dumps(row, ensure_ascii=False))
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            path.write_bytes(payload)
            shard_rows.append(
                {
                    "shard_id": shard_index,
                    "filename": name,
                    "path": str(path),
                    "byte_count": len(payload),
                    "row_count": len(lines),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "modality_counts": {"text": len(lines)},
                }
            )
        shard_index_path = base / "shard_index.json"
        shard_index_path.write_text(
            json.dumps({"dataset_id": dataset_id, "shards": shard_rows}, sort_keys=True), encoding="utf-8"
        )
        shards_jsonl = base / "shards.jsonl"
        shards_jsonl.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in shard_rows) + "\n", encoding="utf-8"
        )
        pointer_path = base / "SHARD_INDEX_POINTER.json"
        pointer_body = {
            "dataset_id": dataset_id,
            "schema_version": "omnicoder.training_shard_pointer.v1",
            "shard_count": len(shard_rows),
            "shard_index": str(shard_index_path),
            "shard_index_sha256": hashlib.sha256(shard_index_path.read_bytes()).hexdigest(),
            "shards_jsonl": str(shards_jsonl),
            "shards_jsonl_sha256": hashlib.sha256(shards_jsonl.read_bytes()).hexdigest(),
            "total_rows": sum(int(row["row_count"]) for row in shard_rows),
        }
        pointer_path.write_text(json.dumps(pointer_body, sort_keys=True), encoding="utf-8")
        entries.append(
            {
                "dataset_id": dataset_id,
                "label": dataset_id,
                "pointer": str(pointer_path),
                "shard_count": len(shard_rows),
                "shard_index_sha256": pointer_body["shard_index_sha256"],
                "total_rows": pointer_body["total_rows"],
                "training_status": "ready_untrained_candidate",
                "modality_counts": {"text": pointer_body["total_rows"]},
            }
        )
        truth[dataset_id] = {"shards": shard_rows, "pointer": str(pointer_path)}

    # Damage dataset 1: shard 0 rewritten after hashing (sha mismatch + byte
    # count disagreement), shard 1 deleted (unreadable).
    damaged = truth["synth_ds_1"]["shards"]
    Path(damaged[0]["path"]).write_bytes(b'{"modality":"text","instruction":"a","response":"b"}\n')
    Path(damaged[1]["path"]).unlink()

    # A third entry whose pointer does not exist at all.
    entries.append(
        {
            "dataset_id": "synth_ds_missing",
            "label": "missing",
            "pointer": str(root / "shards" / "nope" / "SHARD_INDEX_POINTER.json"),
            "shard_count": 1,
            "shard_index_sha256": "0" * 64,
            "total_rows": 1,
            "training_status": "ready_untrained_candidate",
        }
    )

    master = {
        "schema_version": "omnicoder.master_shard_collection.v1",
        "entry_count": len(entries),
        "total_rows": sum(int(entry.get("total_rows") or 0) for entry in entries),
        "total_shards": 5,
        "updated_at": "2026-09-02T00:00:00Z",
        "rows_by_state": {"ready_untrained_candidate": 1},
        "entries": entries,
    }
    master_path = root / "MASTER_SHARD_INDEX_POINTER.json"
    master_path.write_text(json.dumps(master, sort_keys=True), encoding="utf-8")
    return str(master_path), truth


def run_self_test(args: argparse.Namespace, oc_seal: Any) -> int:
    import shutil
    import tempfile

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print("PASS %s" % name, flush=True)
        else:
            failed += 1
            print("FAIL %s %s" % (name, detail), flush=True)

    work = Path(tempfile.mkdtemp(prefix="registry-walk-selftest-"))
    try:
        pointer, _truth = _fixture(work / "registry")
        progress = work / "progress.jsonl"
        out = work / "report.json"

        namespace = argparse.Namespace(
            mode="walk",
            pointer=pointer,
            out=str(out),
            progress=str(progress),
            plan="",
            tokenizer=args.tokenizer,
            curated_shards=args.curated_shards,
            no_tokens=args.no_tokens,
            token_batch=64,
            chunk_timeout_s=30.0,
            file_timeout_s=60.0,
            start_index=0,
            max_entries=0,
            checkpoint_every=1,
            no_shard_hashes=False,
            resume=True,
            entry_count=0,
        )
        run_walk(namespace, oc_seal)
        report = json.loads(out.read_text(encoding="utf-8"))
        totals = report["totals"]

        check("report is canonical JSON", oc_seal.canonical_json_bytes(report) == out.read_bytes())
        check("entries_walked==3", totals["entries_walked"] == 3, str(totals))
        check("shards_seen==4", totals["shards_seen"] == 4, str(totals))
        check("shards_sha_match==2", totals["shards_sha_match"] == 2, str(totals))
        check("shards_sha_mismatch==1", totals["shards_sha_mismatch"] == 1, str(totals))
        check("shards_unreadable==1", totals["shards_unreadable"] == 1, str(totals))
        kinds = {problem["kind"] for problem in report["problems"]}
        check("shard_sha_mismatch reported", "shard_sha_mismatch" in kinds, str(kinds))
        check("shard_unreadable reported", "shard_unreadable" in kinds, str(kinds))
        check("byte_count disagreement reported", "shard_byte_count_disagreement" in kinds, str(kinds))
        check("row_count disagreement reported", "shard_row_count_disagreement" in kinds, str(kinds))
        check("pointer_unreadable reported", "pointer_unreadable" in kinds, str(kinds))
        check("clean entry has no problems", report["entries"][0]["problem_kinds"] == [], str(report["entries"][0]))
        check(
            "shard_index sha verified",
            totals["shard_index_match"] >= 2 and totals["shard_index_mismatch"] == 0,
            str(totals),
        )
        check("shards_jsonl sha verified", totals["shards_jsonl_match"] >= 2, str(totals))

        if not args.no_tokens:
            census = report["census"]
            # 2 clean shards x 10 trainable rows + the 1 row the damaged shard
            # was overwritten with; 4 rows refused by training_allowed/use_policy.
            check("census rows counted", census["rows_total"] == 21, str(census["rows_total"]))
            check(
                "training_allowed=False skipped",
                census["rows_skipped_policy"] == 4,
                str(census["rows_skipped_policy"]),
            )
            check("ocr alias applied", "ocr" in census["by_route"], str(sorted(census["by_route"])))
            check("agentic_workflow -> tool", "tool" in census["by_route"], str(sorted(census["by_route"])))
            truncated = sum(bucket["rows_truncated_by_loader"] for bucket in census["by_route"].values())
            check("loader truncation counted", truncated >= 1, str(truncated))
            uncapped = sum(bucket["tokens_total"] for bucket in census["by_route"].values())
            capped = sum(bucket["tokens_total_loader_view"] for bucket in census["by_route"].values())
            check("uncapped tokens > loader view", uncapped > capped, "%d vs %d" % (uncapped, capped))

        # Resume: a second run must add nothing and must refuse without --resume.
        before = progress.read_bytes()
        run_walk(namespace, oc_seal)
        check("resume adds no rows", progress.read_bytes() == before)
        namespace.resume = False
        try:
            run_walk(namespace, oc_seal)
            check("refuses a populated progress file without --resume", False, "no refusal")
        except WalkError:
            check("refuses a populated progress file without --resume", True)
        namespace.resume = True

        # Finalize reproduces the same totals from the log alone.
        final_out = work / "final.json"
        finalize_namespace = argparse.Namespace(
            mode="finalize",
            pointer=pointer,
            out=str(final_out),
            progress=str(progress),
            plan="",
            entry_count=3,
        )
        run_finalize(finalize_namespace, oc_seal)
        finalized = json.loads(final_out.read_text(encoding="utf-8"))
        check("finalize reproduces totals", finalized["totals"] == report["totals"], "")
        check("finalize reproduces census", finalized["census"] == report["census"], "")

        # Plan mode.
        plan_out = work / "plan.json"
        plan_namespace = argparse.Namespace(
            mode="plan",
            pointer=pointer,
            out=str(plan_out),
            progress=str(work / "plan-progress.jsonl"),
            chunk_timeout_s=30.0,
            file_timeout_s=60.0,
        )
        run_plan(plan_namespace, oc_seal)
        plan = json.loads(plan_out.read_text(encoding="utf-8"))
        check("plan lists 4 shards", plan["shards_listed"] == 4, str(plan["shards_listed"]))
        check("plan payload bytes > 0", plan["payload_bytes_total"] > 0, str(plan["payload_bytes_total"]))

        # Bounded reads: a fifo nobody writes to must time out, not hang.
        if timeouts_enforced():
            fifo = work / "hang.fifo"
            try:
                os.mkfifo(fifo)
                status = read_file_bounded(str(fifo), chunk_timeout_s=2.0, file_timeout_s=4.0)
                check("bounded read times out on a stalled reader", status["status"] == "timeout", str(status))
            except (AttributeError, OSError) as exc:
                check("bounded read times out on a stalled reader", False, "fifo unavailable: %s" % exc)
        else:
            print("SKIP bounded read timeout (no SIGALRM on this host)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\n%d passed, %d failed" % (passed, failed), flush=True)
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("walk", "plan", "finalize", "self-test"), default="walk")
    parser.add_argument("--pointer", default=MASTER_POINTER_DEFAULT)
    parser.add_argument("--out", required=True, help="report/plan JSON (written atomically)")
    parser.add_argument("--progress", default="", help="append-only JSONL progress log")
    parser.add_argument("--plan", default="", help="plan JSON used as the projection denominator")
    parser.add_argument("--tokenizer", default=TOKENIZER_DEFAULT)
    parser.add_argument("--curated-shards", default=CURATED_SHARDS_DEFAULT)
    parser.add_argument("--ledger-dir", default="", help="directory holding oc_seal.py")
    parser.add_argument("--token-batch", type=int, default=256)
    parser.add_argument("--chunk-timeout-s", type=float, default=180.0)
    parser.add_argument("--file-timeout-s", type=float, default=1800.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-entries", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--entry-count", type=int, default=631)
    parser.add_argument("--no-tokens", action="store_true", help="checksums only, no census")
    parser.add_argument("--no-shard-hashes", action="store_true", help="metadata only (debug)")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    oc_seal = _load_oc_seal(args.ledger_dir or None)
    if not args.progress and args.mode in ("walk", "plan", "finalize"):
        args.progress = str(Path(args.out).with_suffix(".progress.jsonl"))
    try:
        if args.mode == "walk":
            return run_walk(args, oc_seal)
        if args.mode == "plan":
            return run_plan(args, oc_seal)
        if args.mode == "finalize":
            return run_finalize(args, oc_seal)
        return run_self_test(args, oc_seal)
    except WalkError as exc:
        print("REFUSED %s" % exc, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
