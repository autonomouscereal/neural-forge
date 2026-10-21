"""ingest.codex_tasksets -- Codex's frozen recovery-eval task sets -> test_suite, test_case, suite_member.

Source (WORKSTATION, read-only): C:/Users/cereal/Documents/ai training/proof_outputs/chimera_recovery_eval_*/
tasks.jsonl + manifest.json, written by Codex's preparers (experiments/chimera_recovery/evaluation.py
prepare-bfcl / prepare-tool-seeds and text_eval_seed.py; protocol: EVALUATION_PROTOCOL.md, 2026-10-13).

  chimera_recovery_eval_bfcl_oct13_v3            -> codex_recovery_bfcl@oct13_v3            (62)
  chimera_recovery_eval_tools_executed_oct13_v2  -> codex_recovery_tools_executed@oct13_v2  (168)
  chimera_recovery_eval_frozen_oct13_v1          -> codex_recovery_frozen@oct13_v1          (230 = bfcl + tools)
  chimera_recovery_eval_text_oct13_v1            -> codex_recovery_text@oct13_v1            (44, from text_eval_seed)
  chimera_recovery_eval_combined_oct13_v1        -> codex_recovery_combined@oct13_v1        (274 = frozen + text)

Cases are global, so the combined and frozen suites link the same test_case rows as their parts.  The
prompt is input.messages, tools input.tools, media input.media (an empty list is stored as NULL); the
private oracle goes to test_case.reference and is never part of the prompt.  Root override:
OMNI_LAB_PROOF_OUTPUTS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import common as C
from . import evalkit as K
from .common import db

SOURCE = "codex_tasksets"
SETS = (("chimera_recovery_eval_bfcl_oct13_v3", "codex_recovery_bfcl", "oct13_v3"),
        ("chimera_recovery_eval_tools_executed_oct13_v2", "codex_recovery_tools_executed", "oct13_v2"),
        ("chimera_recovery_eval_frozen_oct13_v1", "codex_recovery_frozen", "oct13_v1"),
        ("chimera_recovery_eval_text_oct13_v1", "codex_recovery_text", "oct13_v1"),
        ("chimera_recovery_eval_combined_oct13_v1", "codex_recovery_combined", "oct13_v1"))


def proof_outputs() -> str:
    return os.environ.get("OMNI_LAB_PROOF_OUTPUTS", "C:/Users/cereal/Documents/ai training/proof_outputs")


def run(conn, batch: C.Batch, args) -> None:
    host = db.this_host()
    K.grader_id(conn, "chimera_recovery.evaluate")
    for dirname, suite_name, version in SETS:
        batch.saw()
        d = Path(proof_outputs(), dirname)
        tasks_path = d / "tasks.jsonl"
        if not tasks_path.is_file():
            batch.skip(f"missing {tasks_path}")
            continue
        prov = batch.prov(tasks_path, host=host)
        manifest = C.load_json(d / "manifest.json") or {}
        if (d / "manifest.json").is_file():
            batch.prov(d / "manifest.json", host=host)
        rows = [json.loads(line) for line in C.read_text(tasks_path).splitlines() if line.strip()]
        routes = sorted({r.get("route") for r in rows if r.get("route")})
        modality_in = sorted({"text"} | {m.get("kind") or m.get("type") for r in rows
                                         for m in (r.get("input") or {}).get("media") or [] if isinstance(m, dict)} - {None})
        suite = db.ensure_suite(conn, suite_name, version, status="sealed", modality_in=modality_in,
                                default_grader="chimera_recovery.evaluate", bank_sha256=prov[1],
                                description=f"Codex recovery eval tasks ({dirname}); routes {', '.join(routes)}; "
                                            f"manifest_sha256 {manifest.get('manifest_sha256')}; "
                                            f"grader_sha256 {manifest.get('grader_sha256')}",
                                prov=prov, batch_id=batch.id)
        pks = []
        for r in rows:
            inp = r.get("input") or {}
            media = inp.get("media") or None
            reference = {"oracle": r.get("oracle"), "evaluation_scope": r.get("evaluation_scope"),
                         "environment": r.get("environment"), "group_id": r.get("group_id"), "split": r.get("split"),
                         "task_schema": r.get("schema")}
            pks.append(db.ensure_case(conn, r["id"], inp.get("messages") or [], tools=inp.get("tools") or None,
                                      media=media, reference={k: v for k, v in reference.items() if v is not None},
                                      category=r.get("route"), lane=r.get("route"),
                                      tags=["codex_recovery", r.get("split") or "unsplit"],
                                      grader="chimera_recovery.evaluate", prov=prov, batch_id=batch.id))
            batch.wrote()
        K.set_membership(conn, suite, pks)
