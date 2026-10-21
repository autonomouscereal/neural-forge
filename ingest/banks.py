"""ingest.banks -- the prompt/test banks of the omni campaign -> test_suite, test_case, suite_member.

  $OC/calibration/banks/lfm2_semantic_whole_response_prompts_v1.json -> bank_v1@1        (16, reader-graded)
  $OC/calibration/banks/matrix_campaign_probe_prompts_v1.json        -> probe@1          (8 raw continuations)
  $OC/calibration/banks/matrix_campaign_smoke_prompts_v1.json        -> smoke@1          (3 chat prompts)
  $OC/calibration/banks/omni_toolcall_bank_v1.json                   -> toolcall_v1@1    (185, machine-graded)
  $OC/calibration/banks/omni_review_bank_v2_draft.json               -> bank_v2_draft@draft-<sha12> (40, DRAFT)
  $OC/import/rq27b/tree_c5e6052/scripts/import_receipt_qwen35.py     -> rq_prompt_pack@c5e6052 (8 raw prompts;
        'R-Q 210' counts teacher-forced token positions over these 8 prompts, not 210 prompts)

Read-only on every source; the R-Q pack is read with ast (the module is never imported).  Bank v2's
'<<HAYSTACK>>' cases keep the generator spec (with its pinned haystack sha256) in test_case.media, so the
case identity changes when the haystack would.
"""
from __future__ import annotations

import ast
from pathlib import Path

from . import common as C
from . import evalkit as K
from .common import db

SOURCE = "banks"
RQ_TREE = "tree_c5e6052"


def banks_dir() -> str:
    return C.OC + "/calibration/banks"


def rq_script() -> str:
    return f"{C.OC}/import/rq27b/{RQ_TREE}/scripts/import_receipt_qwen35.py"


def read_prompt_pack(path: str) -> list[str]:
    """PROMPT_PACK tuple from import_receipt_qwen35.py by AST (no import, no execution)."""
    tree = ast.parse(C.read_text(path))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id == "PROMPT_PACK" and node.value is not None:
            value = ast.literal_eval(node.value)
            if isinstance(value, (tuple, list)) and all(isinstance(v, str) for v in value):
                return list(value)
    raise db.LabError(f"{path}: no literal PROMPT_PACK tuple of strings")


def ingest_bank_v2_draft(conn, batch: C.Batch, path: str) -> None:
    prov = batch.prov(path)
    doc, rows = K.bank_rows(path)
    assets = {a["asset_id"]: a for a in doc.get("assets") or [] if isinstance(a, dict) and a.get("asset_id")}
    adir = str(Path(path).parent / "assets")
    mods_in, mods_out = set(), set()
    pks = []
    for row in rows:
        m_in = [m for m in str(row.get("modality_in") or "text").split("+") if m]
        m_out = [m for m in str(row.get("modality_out") or "text").split("+") if m]
        mods_in.update(m_in)
        mods_out.update(m_out)
        media = []
        for asset_id in sorted({part.get("asset_id") for msg in row["messages"] if isinstance(msg.get("content"), list)
                                for part in msg["content"] if isinstance(part, dict) and part.get("asset_id")}):
            a = assets.get(asset_id)
            media.append({"asset_id": asset_id, "kind": next((p.get("type") for msg in row["messages"]
                                                             if isinstance(msg.get("content"), list)
                                                             for p in msg["content"]
                                                             if isinstance(p, dict) and p.get("asset_id") == asset_id), None),
                          "path": f"{adir}/{a['path']}" if a and a.get("path") else None,
                          "sha256": a.get("sha256") if a else None,
                          "exists": bool(a) and Path(adir, a.get("path") or "").is_file()})
        if row.get("haystack_spec"):
            media.append({"kind": "haystack", "spec": row["haystack_spec"]})
        reference = {k: row.get(k) for k in ("context", "rubric", "ask", "difficulty", "payload_anchor",
                                             "prompt_tokens_target", "step", "runnable") if row.get(k) is not None}
        pk = db.ensure_case(conn, row["id"], row["messages"], media=media or None, reference=reference,
                            category=row.get("category"), lane=row.get("lane"),
                            tags=["bank_v2_draft", f"step-{row.get('step')}"] + (["runnable"] if row.get("runnable", True)
                                                                                else ["spec-only"]),
                            modality_in=m_in, modality_out=m_out, prov=prov, batch_id=batch.id)
        pks.append(pk)
        batch.wrote()
    order = ["text", "image", "video", "audio", "3d"]
    suite = db.ensure_suite(conn, "bank_v2_draft", f"draft-{(doc.get('bank_sha256') or prov[1])[:12]}", status="draft",
                            modality_in=sorted(mods_in, key=lambda m: order.index(m) if m in order else 99),
                            modality_out=sorted(mods_out, key=lambda m: order.index(m) if m in order else 99),
                            default_grader=None, bank_sha256=doc.get("bank_sha256") or prov[1],
                            description="omni review bank v2 DRAFT, unsealed, pending Cyril's review; rubrics are "
                                        f"diagnostics. counts {db.canonical_json(doc.get('counts'))}",
                            prov=prov, batch_id=batch.id)
    K.set_membership(conn, suite, pks)


def ingest_rq(conn, batch: C.Batch, path: str) -> None:
    prov = batch.prov(path)
    texts = read_prompt_pack(path)
    suite = db.ensure_suite(conn, "rq_prompt_pack", RQ_TREE.split("_", 1)[1], status="sealed",
                            modality_in=["text"], modality_out=["text"], bank_sha256=prov[1],
                            description="R-Q import-parity prompt pack (PROMPT_PACK in import_receipt_qwen35.py). "
                                        "Teacher-forced; 'R-Q 210' = 210 token positions over these 8 prompts.",
                            prov=prov, batch_id=batch.id)
    pks = []
    for n, text in enumerate(texts):
        pks.append(db.ensure_case(conn, f"rq-{n:02d}", {"raw_text": text}, category="import-parity",
                                  tags=["rq_prompt_pack", "raw-continuation", "teacher-forced"],
                                  prov=prov, batch_id=batch.id))
        batch.wrote()
    K.set_membership(conn, suite, pks)


def run(conn, batch: C.Batch, args) -> None:
    b = banks_dir()
    for path, fn in ((f"{b}/lfm2_semantic_whole_response_prompts_v1.json", lambda p: K.ensure_bank_v1(conn, batch, p)),
                     (f"{b}/matrix_campaign_probe_prompts_v1.json",
                      lambda p: K.ensure_raw_prompt_suite(conn, batch, p, "probe", "probe", chat=False,
                                                          description="8-prompt free-run probe, raw continuation, "
                                                                      "64 greedy tokens (matrix campaign stage 5).")),
                     (f"{b}/matrix_campaign_smoke_prompts_v1.json",
                      lambda p: K.ensure_raw_prompt_suite(conn, batch, p, "smoke", "smoke", chat=True,
                                                          description="3-prompt decode smoke, chat-formatted, "
                                                                      "128-token cap (matrix campaign stage 6).")),
                     (f"{b}/omni_toolcall_bank_v1.json", lambda p: K.ensure_toolcall_v1(conn, batch, p)),
                     (f"{b}/omni_review_bank_v2_draft.json", lambda p: ingest_bank_v2_draft(conn, batch, p)),
                     (rq_script(), lambda p: ingest_rq(conn, batch, p))):
        batch.saw()
        if not Path(path).is_file():
            batch.skip(f"missing {path}")
            continue
        fn(path)
