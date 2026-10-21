"""ingest.toolcall_bankruns -- omni_toolcall_bank_v1 runs through llama-server -> eval runs + machine grades.

Source: $OC/distill/toolcall/bank-runs/<label>.grade.json (schema omni-distill-toolcall-v1/bank-grade) with
its <label>.responses.jsonl, plus the serve receipts ($OC/distill/toolcall/receipts*/serve-*.json) whose
'url' matches the responses' 'server' field (card UUID, llama-server command, build) and the export check
naming the GGUF file and its sha256.

One run per grade file: 'eval:toolcall-bank:<label>:<responses sha12>' on suite toolcall_v1, engine 'gguf'.
Every response gets grader toolcall_selfgen.grade_case: 'fail' with the recorded problems when its id is in
the grade file's failures list, otherwise 'pass' (the grade file lists failures only; the ingester checks
that passes + failures equal the file's own totals and refuses the run when they do not).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from . import common as C
from . import evalkit as K
from .common import db

SOURCE = "toolcall_bankruns"


def grade_files() -> list[str]:
    return sorted(glob.glob(f"{C.OC}/distill/toolcall/bank-runs/*.grade.json"))


def serve_receipts() -> dict[str, list[tuple[str, dict]]]:
    """{url: [(path, receipt), ...]} over every serve receipt (later generations reuse the same ports)."""
    out: dict[str, list[tuple[str, dict]]] = {}
    for p in sorted(glob.glob(f"{C.OC}/distill/toolcall/receipts*/serve-*.json")):
        r = C.load_json(p)
        if isinstance(r, dict) and r.get("url"):
            out.setdefault(r["url"], []).append((p, r))
    return out


def serve_for(servers: dict, url: str, graded) -> tuple[str, dict] | None:
    """The serve receipt for `url` that started last before the grade time (None when none did)."""
    best = None
    for p, r in servers.get(url, []):
        start = C.parse_ts(r.get("started_utc"))
        if start is None or (graded is not None and start > graded):
            continue
        if best is None or start > C.parse_ts(best[1].get("started_utc")):
            best = (p, r)
    return best


def export_checks() -> dict[str, tuple[str, dict]]:
    """{gguf path: (receipt path, q4 block)} from export_check_*.json."""
    out = {}
    for p in sorted(glob.glob(f"{C.OC}/distill/toolcall/receipts*/export_check_*.json")):
        r = C.load_json(p)
        q4 = (r or {}).get("q4") if isinstance(r, dict) else None
        if isinstance(q4, dict) and q4.get("path"):
            out[q4["path"]] = (p, {**q4, "parent_dir": r.get("parent_dir"), "llama_cpp_commit": r.get("llama_cpp_commit"),
                                   "checked_utc": r.get("checked_utc")})
    return out


def ingest_grade_file(conn, batch: C.Batch, gpath: str, servers: dict, exports: dict) -> None:
    g = C.load_json(gpath)
    if not isinstance(g, dict) or g.get("schema") != "omni-distill-toolcall-v1/bank-grade":
        batch.skip(f"{gpath}: not a bank-grade file")
        return
    gprov = batch.prov(gpath)
    rpath = K.at((g.get("responses") or {}).get("path")) or ""
    if not os.path.isfile(rpath):           # moved with its grade file
        rpath = gpath[:-len(".grade.json")] + ".responses.jsonl"
    if not os.path.isfile(rpath):
        batch.skip(f"{gpath}: responses file {rpath} missing")
        return
    rprov = batch.prov(rpath)
    if (g.get("responses") or {}).get("sha256") not in (None, rprov[1]):
        batch.skip(f"{gpath}: responses sha differs from the grade file's; graded against the recorded file anyway")
    rows = [json.loads(line) for line in C.read_text(rpath).splitlines() if line.strip()]
    failures = {f["id"]: f for f in g.get("failures") or [] if isinstance(f, dict) and f.get("id")}
    if len(rows) != g.get("cases") or len(rows) - len(failures) != g.get("passed"):
        raise db.LabError(f"{gpath}: {len(rows)} responses / {len(failures)} failures do not reproduce "
                          f"cases={g.get('cases')} passed={g.get('passed')}")
    bank_path = K.at((g.get("bank") or {}).get("path")) or f"{C.OC}/calibration/banks/omni_toolcall_bank_v1.json"
    suite, _ids = K.ensure_toolcall_v1(conn, batch, bank_path)
    members = K.suite_members(conn, suite)
    bank_doc, bank_rows = K.bank_rows(bank_path)
    by_id = {r["id"]: r for r in bank_rows}

    used = sorted({r.get("server") for r in rows if r.get("server")})
    serves = [sv for sv in (serve_for(servers, u, C.parse_ts(g.get("graded_utc"))) for u in used) if sv]
    for sp, _ in serves:
        batch.prov(sp)
    gguf = next((s[1].get("gguf") for s in serves if s[1].get("gguf")), None)
    exp = exports.get(gguf or "")
    if exp:
        batch.prov(exp[0])
    label = Path(gpath).name[:-len(".grade.json")]
    parent_rec = K.ledger_by_directory().get(((exp or ("", {}))[1].get("parent_dir") or "").rstrip("/"))
    model = K.ensure_model_at(conn, gguf or rpath, role="export", layout="gguf", family="Qwen3.8-27B-U",
                              stage="parent-gguf",
                              config={"gguf_sha256": (exp or ("", {}))[1].get("sha256"),
                                      "quant": (exp or ("", {}))[1].get("quant"),
                                      "bytes": (exp or ("", {}))[1].get("bytes"),
                                      "llama_cpp_commit": (exp or ("", {}))[1].get("llama_cpp_commit")},
                              notes=f"GGUF the toolcall bank run '{label}' served", lineage=[
                                  (parent_rec["checkpoint_id"], "export", {"via": "selfgen_sampler export check",
                                                                           "receipt": (exp or ("",))[0]})
                              ] if parent_rec else [], prov=gprov, batch_id=batch.id)
    first = serves[0][1] if serves else {}
    tool = g.get("tool") or {}
    tool_v = C.tool_version_for(conn, tool.get("tool") or "toolcall_selfgen", tool.get("version"), tool.get("sha256"),
                                path=tool.get("path"))
    env_id = K.env_for(conn, python=tool.get("python"),
                       llama_cpp_build=((first.get("props") or {}).get("build_info")),
                       env_vars={"CUDA_VISIBLE_DEVICES": ",".join(s[1].get("env", {}).get("CUDA_VISIBLE_DEVICES", "")
                                                                  for s in serves) or None,
                                 "CUDA_DEVICE_ORDER": (first.get("env") or {}).get("CUDA_DEVICE_ORDER")},
                       packages={"llama_cpp_export_commit": (exp or ("", {}))[1].get("llama_cpp_commit")})
    starts = [C.parse_ts(s[1].get("started_utc")) for s in serves if s[1].get("started_utc")]
    started = min(starts) if starts else None
    ended = C.parse_ts(g.get("graded_utc"))
    rid = K.eval_run(conn, batch, run_key=f"eval:toolcall-bank:{label}:{rprov[1][:12]}",
                     name=f"toolcall bank {label}", project="chimera-omni", model=model, suite=suite, engine="gguf",
                     gen={"decoding": "greedy (llama-server)", "temperature": 0, "render": bank_doc.get("render")},
                     config={"label": g.get("label"), "servers": used, "gguf": gguf,
                             "llama_server_command": first.get("command"), "n_ctx": (first.get("props") or {}).get("n_ctx"),
                             "slots": (first.get("props") or {}).get("total_slots"),
                             "grade_receipt_sha256": g.get("receipt_sha256"), "responses": rpath},
                     started=started, ended=ended, tool_version_id=tool_v, env_id=env_id, receipt=rprov,
                     run_dir=str(Path(rpath).parent), label=g.get("label"),
                     bank_file_sha256=(g.get("bank") or {}).get("sha256"), thinking_requested="off")
    K.clear_devices(conn, rid)
    for s_path, s in serves:
        card = s.get("card_before") or {}
        gid = C.gpu_for(conn, uuid=card.get("uuid")) if card.get("uuid") else None
        if gid is None:
            batch.skip(f"{label}: no gpu row for {card.get('uuid')}")
            continue
        db.add_run_device(conn, rid, gid, role="server", kernel_path="llama.cpp-cuda",
                          kernels={"flash_attn": (s.get("log_analysis") or {}).get("flash_attn_state"),
                                   "command": s.get("command")},
                          detail={"url": s.get("url"), "receipt": s_path, "card_before": card})
    grader = K.grader_id(conn, "toolcall_selfgen.grade_case")
    for r in rows:
        batch.saw()
        case = by_id.get(r["id"])
        if case is None:
            batch.skip(f"{label}: response id {r['id']} not in the bank")
            continue
        pk = K.resolve_case(conn, batch, members, r["id"], case["messages"], tools=case.get("tools"))
        res = db.upsert_eval_result(conn, rid, pk, output_text=r.get("response"), output_tokens=r.get("tokens"),
                                    termination=K.termination_of(r.get("end")),
                                    latency_ms=int(r["wall_s"] * 1000) if isinstance(r.get("wall_s"), (int, float)) else None,
                                    raw={"lane": r.get("lane"), "server": r.get("server"), "end": r.get("end")},
                                    prov=rprov, batch_id=batch.id)
        fail = failures.get(r["id"])
        db.put_grade(conn, res, grader, "fail" if fail else "pass", score=0.0 if fail else 1.0,
                     why="; ".join(fail.get("problems") or []) if fail else None,
                     detail={"calls": fail.get("calls")} if fail else None, prov=gprov, batch_id=batch.id)
        batch.wrote()
    for lane, tally in (g.get("lanes") or {}).items():
        if isinstance(tally, dict) and isinstance(tally.get("passed"), int):
            db.put_metric(conn, rid, f"toolcall/{lane}", float(tally["passed"]), denominator=float(tally.get("cases") or 0))
    db.put_metric(conn, rid, "toolcall/passed", float(g["passed"]), denominator=float(g["cases"]))


def run(conn, batch: C.Batch, args) -> None:
    K.ensure_gpus(conn)
    servers, exports = serve_receipts(), export_checks()
    for gpath in grade_files():
        batch.saw()
        ingest_grade_file(conn, batch, gpath, servers, exports)
