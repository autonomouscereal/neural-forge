"""ingest.foldhist -- folding experiment history (inventory/<date>/foldhist.json, schema foldhist-v1)
-> technique, experiment, experiment_eval.

One experiment per inventory entry (key = entry id: 'NF-10', 'MX-4X', 'OC01-F-Q35-4B-X03', 'X3-08B', ...).
Text fields are kept as written; only two things are parsed:
    ratio_body  = the number after 'net ' when the entry states a net child/parent ratio <= 1
                  (foldhist notes: 'net' = child params / parent params); everything else stays NULL
                  and the ratio text is in params.ratio_text
    date        = the entry date when it STARTS with YYYY-MM-DD; the raw text is in detail.date_text
evidence_level: the schema allows one level, the inventory sometimes names two ('memory+report'); the
WEAKEST is stored (receipt > report > codex > memory) and the raw text is kept in detail.evidence_text.

Model links: ledger ids found in the entry (OCnn-..., PARENT-...) are kept in detail.model_refs and
resolved against model rows at ingest time; rows that do not exist yet stay NULL and are filled by a
re-run after the ledger/model ingesters.  experiment_eval links every eval_run of the resolved child
(role 'child') and parent (role 'parent').  Re-runnable.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

from . import common as C
from . import knowledge_common as K
from .common import db

SOURCE = "foldhist"

EVIDENCE_ORDER = ("receipt", "report", "codex", "memory")      # strongest first
LEDGER_ID_RE = re.compile(r"\b((?:OC\d\d-[A-Z]-[A-Z0-9]+(?:-[A-Z0-9]+)+)|(?:PARENT-[A-Z0-9]+(?:-[A-Z0-9]+)*))\b")
NET_RE = re.compile(r"^net\s+([01]?\.\d+)\b")
# Composite-chain entries whose text names no ledger id: the stage they built, as registered in
# $OC/ledger.jsonl (register events: C2K4352K3U is stage S5, OMNI1 is the S7 composite).
STAGE_CHILD = {"S5-C2K4352K3U": "OC02-F-Q38-27BU-C2K4352K3U", "S7-OMNI1": "OC02-F-Q38-27BU-OMNI1"}
PROJECT_OF_PROGRAM = (("neural-forge", "neural-forge"), ("matrix campaign", "neural-forge"),
                      ("density program", "neural-forge"), ("audit", "neural-forge"),
                      ("chimera omni media", "media"), ("chimera omni", "chimera-omni"))


def weakest_evidence(text: str | None) -> str:
    parts = [p.strip() for p in re.split(r"[+/,]", text or "") if p.strip()]
    known = [p for p in parts if p in EVIDENCE_ORDER]
    if not known:
        return "unverified"
    return max(known, key=EVIDENCE_ORDER.index)


def project_of(program: str | None) -> str:
    prog = (program or "").lower()
    for prefix, project in PROJECT_OF_PROGRAM:
        if prog.startswith(prefix):
            return project
    return "neural-forge"


def kind_of(entry: dict) -> str:
    prog = (entry.get("program") or "").lower()
    if prog.startswith("audit"):
        return "other"
    return "fold"


def ratio_body(ratio: str | None) -> float | None:
    m = NET_RE.match((ratio or "").strip())
    if not m:
        return None
    v = float(m.group(1))
    return v if 0 < v <= 1 else None


def entry_date(text: str | None) -> date | None:
    m = re.match(r"^(20\d\d-[01]\d-[0-3]\d)\b", (text or "").strip())
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(1))
    except ValueError:
        return None


def model_refs(entry: dict) -> dict:
    """{'child': id|None, 'parent': id|None, 'all': [...]} ledger ids named in the entry."""
    found: list[str] = []
    for field in ("id", "parent", "results", "technique"):
        for m in LEDGER_ID_RE.finditer(entry.get(field) or ""):
            if m.group(1) not in found:
                found.append(m.group(1))
    child = entry["id"] if LEDGER_ID_RE.fullmatch(entry["id"]) and not entry["id"].startswith("PARENT-") else None
    if child is None:
        child = STAGE_CHILD.get(entry["id"])
    if child is None:
        child = next((f for f in LEDGER_ID_RE.findall(entry.get("results") or "") if not f.startswith("PARENT-")), None)
    parent = next((f for f in LEDGER_ID_RE.findall(entry.get("parent") or "") if f.startswith("PARENT-")), None)
    return {"child": child, "parent": parent, "all": found}


def run(conn, batch: C.Batch, args) -> None:
    inv = K.inventory_dir(args) / "foldhist.json"
    prov = batch.prov(inv, host=db.this_host())
    doc = json.loads(Path(inv).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != "foldhist-v1" or not isinstance(doc.get("entries"), list):
        raise db.LabError(f"{inv}: expected schema foldhist-v1 with an 'entries' list")
    tally = K.Tally(batch)
    linked = 0
    for e in doc["entries"]:
        batch.saw()
        if not e.get("id") or not e.get("technique"):
            batch.skip(f"entry without id/technique: {str(e)[:80]}")
            continue
        family = "fold" if kind_of(e) == "fold" else "other"
        tech_id, state = K.upsert(conn, "technique", ("name",), {
            "name": e["technique"], "family": family, "description": f"first seen in foldhist {e['id']} ({e.get('program')})"},
            ignore_in_compare=("description",))
        tally.add("technique", state)
        refs = model_refs(e)
        child_id = db.model_id(conn, refs["child"]) if refs["child"] else None
        parent_id = db.model_id(conn, refs["parent"]) if refs["parent"] else None
        exp_id, state = K.upsert(conn, "experiment", ("key",), {
            "project": project_of(e.get("program")), "key": e["id"], "kind": kind_of(e), "technique_id": tech_id,
            "parent_id": parent_id, "child_id": child_id, "ratio_body": ratio_body(e.get("ratio")),
            "params": {"ratio_text": e.get("ratio"), "objective": e.get("objective"), "parent_text": e.get("parent")},
            "behaviour_legacy": e.get("behaviour"), "verdict": e.get("verdict"),
            "evidence_level": weakest_evidence(e.get("evidence")), "results_path": e.get("results"),
            "date": entry_date(e.get("date")),
            "detail": {"program": e.get("program"), "date_text": e.get("date"), "evidence_text": e.get("evidence"),
                       "model_refs": refs},
            "src_path": prov[0], "src_sha256": prov[1], "src_host": prov[2], "ingest_batch_id": batch.id},
            jsonb_cols=("params", "detail", "budget"))
        tally.add("experiment", state)
        for role, mid in (("child", child_id), ("parent", parent_id)):
            if mid is None:
                continue
            with conn.cursor() as cur:
                cur.execute("INSERT INTO experiment_eval (experiment_id, eval_run_id, role) "
                            "SELECT %s, er.run_id, %s FROM eval_run er WHERE er.model_id = %s "
                            "ON CONFLICT (experiment_id, eval_run_id) DO NOTHING", (exp_id, role, mid))
                linked += cur.rowcount
    if linked:
        tally.c[("experiment_eval", "inserted")] += linked
        batch.wrote(linked)
    K.finish(batch, tally)


if __name__ == "__main__":
    K.main_guard(sys.modules[__name__])
