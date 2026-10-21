"""ingest.arch_requirements -- Cyril's standing architecture requirements -> arch_requirement.

Source: inventory/<dir>/arch_requirements.json (schema omni_lab.arch_requirements.v1), a list of
{id, title, text, check, refs}.  Every ref names the words a requirement rests on: either
{event: <agent_memory_events.id>, fragment} or {key_prefix: 'cc:<session8>:<line>:', fragment}.
Each ref is resolved to the `directive` row(s) whose verbatim contains the fragment (case- and
whitespace-insensitive), so the requirement cites Cyril's own dated words, never a paraphrase.

An unresolved ref refuses the whole batch: load the directives first (`ingest directives`), or fix the
fragment.  Rows are keyed by requirement id; an unchanged inventory writes nothing.  The requirement
text lives here and only here: chimera_omni.conformance reports checks by id and arch_check enforces
the ids through its foreign key.

    omni_lab.py ingest arch_requirements --inventory inventory/2026-10-21-arch-v2
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import common as C
from . import knowledge_common as K
from .common import db

SOURCE = "arch_requirements"
SCHEMA = "omni_lab.arch_requirements.v1"
DEFAULT_DIR = K.REPO / "inventory" / "2026-10-21-arch-v2"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def resolve_ref(conn, ref: dict) -> list[tuple[int, int | None]]:
    """(directive.id, source_event_id) rows whose verbatim contains the ref's fragment."""
    fragment = _norm(ref.get("fragment", ""))
    if not fragment:
        raise db.LabError(f"ref without a fragment: {ref}")
    with conn.cursor() as cur:
        if "event" in ref:
            cur.execute("SELECT id, source_event_id, verbatim FROM directive WHERE source_event_id = %s ORDER BY id",
                        (int(ref["event"]),))
        elif "key_prefix" in ref:
            cur.execute("SELECT id, source_event_id, verbatim FROM directive WHERE directive_key LIKE %s ORDER BY id",
                        (str(ref["key_prefix"]).replace("%", r"\%").replace("_", r"\_") + "%",))
        else:
            raise db.LabError(f"ref needs 'event' or 'key_prefix': {ref}")
        rows = cur.fetchall()
    return [(rid, ev) for rid, ev, verbatim in rows if fragment in _norm(verbatim)]


def run(conn, batch: C.Batch, args) -> None:
    given = getattr(args, "inventory", None)
    inv = (Path(given) if given else DEFAULT_DIR) / "arch_requirements.json"
    prov = batch.prov(inv, host=db.this_host())
    doc = json.loads(Path(inv).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise db.LabError(f"{inv}: expected schema {SCHEMA}, got {doc.get('schema')!r}")
    tally = K.Tally(batch)
    unresolved: list[str] = []
    rows: list[dict] = []
    for req in doc["requirements"]:
        batch.saw()
        rid = str(req["id"])
        if not re.fullmatch(r"R\d+", rid):
            raise db.LabError(f"requirement id {rid!r} is not R<n>")
        directive_ids: list[int] = []
        event_ids: list[int] = []
        resolved_refs = []
        for ref in req["refs"]:
            hits = resolve_ref(conn, ref)
            if not hits:
                unresolved.append(f"{rid}: {ref}")
                continue
            resolved_refs.append({**ref, "directive_ids": [h[0] for h in hits]})
            for did, ev in hits:
                if did not in directive_ids:
                    directive_ids.append(did)
                if ev is not None and ev not in event_ids:
                    event_ids.append(int(ev))
        rows.append({"id": rid, "text": req["text"], "directive_ids": sorted(directive_ids),
                     "source_event_ids": sorted(event_ids),
                     "detail": {"title": req.get("title"), "check": req.get("check"), "refs": resolved_refs,
                                "inventory": prov[0], "inventory_sha256": prov[1], "spec": doc.get("spec")}})
    if unresolved:
        raise db.LabError("unresolved directive refs (load `ingest directives` first or fix the fragment): "
                          + "; ".join(unresolved))
    for row in rows:
        _id, state = K.upsert(conn, "arch_requirement", ("id",), row, jsonb_cols=("detail",), id_col="id",
                              ignore_in_compare=())
        tally.add("arch_requirement", state)
    K.finish(batch, tally)


if __name__ == "__main__":
    K.main_guard(sys.modules[__name__])
