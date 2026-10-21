"""ingest.directives -- owner directives (Cyril's words, verbatim, dated, sourced, tagged) -> directive.

Source: inventory/<date>/directives.json, a list of {date_utc, agent, source, quote, tags[, note]}.  Every
quote was verified as an exact substring of its source when the inventory was built; this ingester does not
paraphrase or trim it.  Rows are keyed by where the words were said plus a hash of the quote, because one
prompt often carries several directives:

    ev:<agent_memory_events.id>:<sha8(quote)>        prompt filed by the memory hook (Codex, Claude Code, dot)
    cc:<session8>:<line>:<sha8(quote)>               Claude Code transcript line (queued mid-turn prompt)
    doc:<sha8(path)>:<line>:<sha8(quote)>            quote preserved only in a doc / memory file

verbatim is re-anchored to the source on every run: the inventory stored some quotes with whitespace
collapsed (24 of 128 event quotes on 2026-10-21), so the ingester reads the source text (agent_memory_events
.content read-only, the transcript JSONL line, or the doc file), finds the quote with whitespace-flexible
matching and stores the exact source span.  A quote that cannot be found in a readable source is refused
(skipped, noted on the batch); a source that cannot be read here (agent_memory credential missing, file
on another host) keeps the inventory quote and says so in summary.  The key is computed from the
inventory quote, so re-anchoring updates the row instead of adding a new one.

Columns: channel = codex | claude-code | dot | transcript | doc.  said_at_verified is false for dot rows
(sync time, not prompt time) and for doc rows (date read from the document text, not from the event).
src_path is the ORIGINAL source reference (event id, transcript path:line, doc path:line), not a file
this ingester read; src_sha256 is the sha256 of the inventory snapshot that carried the quote.
Re-runnable: an unchanged inventory writes nothing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import common as C
from . import knowledge_common as K
from .common import db

SOURCE = "directives"

EV_RE = re.compile(r"^agent_memory_events\.id=(\d+) \(event_type=(\w+), agent=([\w.-]+)")
CC_RE = re.compile(r"^(?P<path>.+/(?P<sess>[0-9a-f]{8})-[0-9a-f-]+\.jsonl):(?P<line>\d+)")
DOC_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+)$")
AGENT_CHANNEL = {"codex": "codex", "claudecode": "claude-code", "dot": "dot"}


def classify(item: dict) -> dict:
    """Key, channel and event id for one inventory row.  Raises LabError on an unrecognised source."""
    src = item["source"]
    qsha = K.sha_text(item["quote"], 8)
    m = EV_RE.match(src)
    if m:
        agent = m.group(3).lower()
        channel = AGENT_CHANNEL.get(agent)
        if channel is None:
            raise db.LabError(f"unknown agent {m.group(3)!r} in {src[:80]}")
        return {"key": f"ev:{m.group(1)}:{qsha}", "channel": channel, "event_id": int(m.group(1)),
                "origin": src, "doc_path": None, "doc_line": None}
    m = CC_RE.match(src)
    if m and "queued mid-turn prompt" in src:
        return {"key": f"cc:{m.group('sess')}:{m.group('line')}:{qsha}", "channel": "transcript", "event_id": None,
                "origin": src, "doc_path": None, "doc_line": None}
    m = DOC_RE.match(src)
    if m:
        return {"key": f"doc:{K.sha_text(m.group('path'), 8)}:{m.group('line')}:{qsha}", "channel": "doc",
                "event_id": None, "origin": src, "doc_path": m.group("path"), "doc_line": int(m.group("line"))}
    raise db.LabError(f"unrecognised directive source: {src[:120]}")


def date_from_doc(path: str, line: int) -> tuple[object, str] | None:
    """The first date on the cited line, else on the nearest line above it (a dated heading), else in the
    file name.  None when the file is unreadable here (doc rows then skip; run on the workstation)."""
    p = Path(path)
    text = C.read_text(p)
    if not text:
        return None
    lines = text.splitlines()
    for i in range(min(line, len(lines)) - 1, -1, -1):
        d = K.first_date(lines[i])
        if d:
            return d, f"{p.name}:{i + 1}"
    d = K.first_date(p.name)
    return (d, f"file name {p.name}") if d else None


def anchor(quote: str, text: str | None) -> str | None:
    """The exact span of `text` that `quote` stands for: identical, or equal up to whitespace runs."""
    if not text:
        return None
    if quote in text:
        return quote
    m = re.search(r"\s+".join(re.escape(tok) for tok in quote.split()), text)
    return text[m.start():m.end()] if m else None


def json_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in json_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in json_strings(v)]
    return []


class SourceTexts:
    """Read-only access to the texts the quotes came from (None = cannot be read on this host)."""

    def __init__(self, batch: C.Batch, event_ids: list[int]):
        self.events: dict[int, str] | None = None
        try:
            mem = db.connect("admin", dbname="agent_memory", app="ingest-directives-readonly")
        except Exception as exc:                       # no credential / no route: keep inventory quotes
            batch.notes.append(f"agent_memory not readable ({type(exc).__name__}); event quotes not re-anchored")
            return
        try:
            mem.set_session(readonly=True)
            with mem.cursor() as cur:
                cur.execute("SELECT id, content FROM agent_memory_events WHERE id = ANY(%s)", (event_ids,))
                self.events = {i: c for i, c in cur.fetchall()}
            mem.rollback()
        finally:
            mem.close()

    def event(self, event_id: int) -> str | None:
        return None if self.events is None else self.events.get(event_id, "")

    @staticmethod
    def transcript_line(path: str, line: int) -> list[str] | None:
        text = C.read_text(path, limit=1 << 31)
        if not text:
            return None
        lines = text.splitlines()
        if not 0 < line <= len(lines):
            return []
        try:
            return json_strings(json.loads(lines[line - 1]))
        except ValueError:
            return [lines[line - 1]]

    @staticmethod
    def doc(path: str) -> str | None:
        return C.read_text(path) or None


def source_span(src: SourceTexts, info: dict, quote: str) -> tuple[str | None, str]:
    """(exact span, state) with state 'exact' | 'reanchored' | 'not-found' | 'unreadable'."""
    if info["event_id"] is not None:
        texts = src.event(info["event_id"])
        texts = None if texts is None else [texts]
    elif info["channel"] == "transcript":
        m = CC_RE.match(info["origin"])
        texts = src.transcript_line(m.group("path"), int(m.group("line")))
    else:
        doc = src.doc(info["doc_path"])
        texts = None if doc is None else [doc]
    if texts is None:
        return None, "unreadable"
    for t in texts:
        span = anchor(quote, t)
        if span is not None:
            return span, ("exact" if span == quote else "reanchored")
    return None, "not-found"


def run(conn, batch: C.Batch, args) -> None:
    inv = K.inventory_dir(args) / "directives.json"
    prov = batch.prov(inv, host=db.this_host())
    items = json.loads(Path(inv).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise db.LabError(f"{inv}: expected a JSON list of directives")
    tally = K.Tally(batch)
    seen_keys: set[str] = set()
    event_ids = sorted({int(m.group(1)) for it in items if (m := EV_RE.match(it.get("source") or ""))})
    src = SourceTexts(batch, event_ids)
    anchor_states: dict[str, int] = {}
    for item in items:
        batch.saw()
        quote = (item.get("quote") or "").strip()
        if not quote:
            batch.skip(f"empty quote from {item.get('source', '?')[:60]}")
            continue
        info = classify(item)
        if info["key"] in seen_keys:
            batch.skip(f"duplicate directive {info['key']}")
            continue
        seen_keys.add(info["key"])
        said_at = C.parse_ts(item.get("date_utc"))
        verified = info["channel"] not in ("dot", "doc")
        summary = item.get("note")
        if said_at is None:
            found = date_from_doc(info["doc_path"], info["doc_line"]) if info["doc_path"] else None
            if not found:
                batch.skip(f"{info['key']}: no date in inventory and source not readable on {db.this_host()}")
                continue
            said_at = K.utc_midnight(found[0])
            verified = False
            summary = ((summary + "; ") if summary else "") + f"date {found[0]} read from {found[1]} (day only)"
        span, how = source_span(src, info, quote)
        anchor_states[how] = anchor_states.get(how, 0) + 1
        if how == "not-found":
            batch.skip(f"{info['key']}: quote not found in its source {info['origin'][:80]}")
            continue
        if how == "unreadable":
            span = quote
            summary = ((summary + "; ") if summary else "") + f"source not re-read on {db.this_host()}; inventory quote"
        _id, state = K.upsert(conn, "directive", ("directive_key",), {
            "directive_key": info["key"], "said_at": said_at, "said_at_verified": verified, "author": "cyril",
            "channel": info["channel"], "source_event_id": info["event_id"], "verbatim": span,
            "topics": sorted(set(item.get("tags") or [])), "summary": summary,
            "src_path": info["origin"], "src_sha256": prov[1], "src_host": prov[2], "ingest_batch_id": batch.id})
        tally.add("directive", state)
    K.finish(batch, tally)
    batch.tally_lines.append("source check: " + " ".join(f"{k}={v}" for k, v in sorted(anchor_states.items())))


if __name__ == "__main__":
    K.main_guard(sys.modules[__name__])
