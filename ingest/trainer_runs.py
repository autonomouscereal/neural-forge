"""ingest.trainer_runs -- stage-D training runs -> run (+segments/attempts), metric curves, run_device, models.

Sources (all read-only):
  $L/distill/fullkd/<run>/queue.status, receipt-*.json, train-NNN.log
      parent run 'distill:fullkd:<run>' + one child run per segment ('...:seg<N>'); per-step metrics from
      the 'STEP {json}' lines of each segment's log (the parent carries the merged curve: a step replayed
      by a later segment takes the later value) + receipt summaries; the student directory becomes a
      model (role student) with a 'distill' lineage edge to the --p1-chimera base.
  $L/distill/media_m1/<run>/queue.status, receipt-{probe,train}-NNN.json, {probe,train}-NNN.log
      one run per attempt (attempt i of a kind <-> log <kind>-<i:03d>.log, by order; the receipt named on
      the exit line is checked against that ordinal); history -> curves incl. loss/<route>; evals ->
      eval/<route> at their step.  out/export becomes a model (role media-unit, stage M1).
  $L/distill/media_m1/selftest*.json, $L/distill/m2/selftest*.json -> smoke runs (ended_at = file mtime).
  $OC/distill/queues/d2.status + the train log it names -> one run per 'train short start'.
  $L/distill/d1/run-*/ (launch.json, runtime-status.json, trainer/d1_receipt.json) -> one run per D1 run dir.
"""
from __future__ import annotations

import glob
import os
import re

from . import common as C
from .common import db

SOURCE = "trainer_runs"

FKD_QSTART = re.compile(r"^queue start: GPU(\d+), data (\S+) \(([0-9a-f]+)\), tokens (\d+), micro (\d+), lr (\S+?)"
                        r"(?:, extra \[([^\]]*)\])?, trainer ([0-9a-f]+)")
FKD_SEG_START = re.compile(r"^segment (\d+) start, log (\S+)")
FKD_SEG_EXIT = re.compile(r"^segment (\d+) exit (\d+):\s*(.*)$")
M1_START = re.compile(r"^start (probe|train): data (\S+) \(([0-9a-f]+)\), trainer ([0-9a-f]+), steps (\d+), lr (\S+)")
M1_EXIT = re.compile(r"^(probe|train) exit (\d+):\s*(.*)$")
RECEIPT_IN_TEXT = re.compile(r"(/\S+/receipt-[\w.-]+\.json)")
UUID_RX = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def exit_status(rc: int, text: str) -> str:
    if rc == 0:
        return "pass"
    if "REFUSED" in text:
        return "refused"
    if rc in (4,) or "WAITING" in text:
        return "stopped"
    if rc == 143:
        return "stopped"
    return "fail"


def status_lines(path: str) -> list[tuple]:
    out = []
    for raw in C.read_text(path).splitlines():
        m = C.TS_RE.match(raw.strip())
        if m:
            out.append((C.parse_ts(m.group(1)), m.group(2), raw.rstrip()))
    return out


def _model_for_dir(conn, batch, path: str, project: str, role: str, layout: str, stage: str, family: str | None,
                   notes: str, base_id: int | None, prov) -> int | None:
    if not os.path.isdir(path):
        return None
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    census = C.header_census(shards) if 0 < len(shards) <= 120 else {}
    size = sum(os.path.getsize(s) for s in shards)
    cfg = {"census": {k: census[k] for k in ("params", "n_tensors", "dtype_params", "top_prefixes")}} if census else {}
    for mname in ("student_manifest.json", "export_manifest.json", "manifest.json"):
        mp = os.path.join(path, mname)
        doc = C.load_json(mp) if os.path.isfile(mp) else None
        if isinstance(doc, dict):
            cfg["manifest"] = {k: v for k, v in doc.items()
                               if isinstance(v, (str, int, float, bool)) and (not isinstance(v, str) or len(v) < 600)}
            cfg["manifest_file"] = mname
            break
    existing = C.model_id_by_path(conn, path)
    key = None
    if existing is not None:
        with conn.cursor() as cur:
            cur.execute("SELECT model_key FROM model WHERE id = %s", (existing,))
            key = cur.fetchone()[0]
    mid = db.upsert_model(conn, key or C.sha16_path_key(path), project, role, layout, family=family, stage=stage,
                          path=path, host_id=C.host_id(conn), parameter_count=census.get("params") or None,
                          size_bytes=size or None, dtype=C.main_dtype(census.get("dtype_params") or {}) if census else None,
                          config=cfg, status="present", notes=notes, prov=prov, batch_id=batch.id)
    if base_id and base_id != mid:
        db.link_lineage(conn, mid, base_id, "distill", detail={"from": "trainer argv base"}, prov=prov, batch_id=batch.id)
    return mid


def _argv_value(argv: list, flag: str) -> str | None:
    if isinstance(argv, list) and flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return str(argv[i + 1])
    return None


def _merge_curve(conn, run_id: int, records: list[dict]) -> int:
    rows = {}
    for rec in records:
        for name, step, value in C.step_metrics(rec):
            rows[(name, step)] = value
    return db.put_metrics(conn, run_id, [(n, s, v) for (n, s), v in rows.items()])


# ---------------------------------------------------------------------------
# full-parameter KD
# ---------------------------------------------------------------------------

def ingest_fullkd(conn, batch, host_id) -> None:
    for run_dir in sorted(glob.glob(C.LX + "/distill/fullkd/*/")):
        run_dir = run_dir.rstrip("/")
        name = os.path.basename(run_dir)
        qs = os.path.join(run_dir, "queue.status")
        receipts = sorted(glob.glob(os.path.join(run_dir, "receipt-*.json")))
        if not os.path.isfile(qs) and not receipts:
            continue
        batch.saw()
        docs = {p: C.load_json(p) for p in receipts}
        final_path = max((p for p in receipts if isinstance(docs[p], dict)),
                         key=lambda p: (docs[p].get("step") or 0, p), default=None)
        final = docs.get(final_path) if final_path else None
        lines = status_lines(qs) if os.path.isfile(qs) else []
        qprov = batch.prov(qs) if lines else None
        rprov = batch.prov(final_path) if final_path else None
        argv = (final or {}).get("argv") or []
        base_dir = _argv_value(argv, "--p1-chimera")
        base_id = C.model_id_by_path(conn, base_dir)
        tool_vid = None
        tool_file = argv[0] if argv and str(argv[0]).endswith(".py") else None
        if final:
            # the trainer sha16 on the LAST queue start is the one that wrote the final receipt
            qsha = ([m.group(8) for _t, text, _r in lines for m in [FKD_QSTART.match(text)] if m] or [None])[-1]
            tool_vid = (C.tool_version_for(conn, str(final.get("tool")), final.get("tool_version"), qsha, tool_file)
                        if qsha else None) or C.tool_version_for(conn, str(final.get("tool")), final.get("tool_version"))
        env_id = db.ensure_env(conn, torch=final.get("torch"),
                               env_vars={"PYTORCH_CUDA_ALLOC_CONF": final.get("cuda_alloc_conf")}) if final and final.get("torch") else None
        last_text = lines[-1][1] if lines else ""
        status = ("pass" if last_text.startswith("DONE") or (final or {}).get("verdict") == "completed"
                  else "stopped" if "STOPPED" in last_text or "STOP file" in last_text else "unknown")
        started = lines[0][0] if lines else None
        ended = lines[-1][0] if lines and status != "unknown" else None
        student_dir = os.path.join(run_dir, "out", "student")
        student_id = _model_for_dir(conn, batch, student_dir, "chimera-omni", "student", "qwen3_5", f"fullkd-{name}",
                                    "Qwen3.8-27B-U", f"full-parameter KD student of fullkd {name} (text tensors only); "
                                    "not registered in the ledger", base_id, rprov or qprov) if final else None
        config = {}
        if final:
            config = {k: final.get(k) for k in ("command", "device", "dtype", "torch", "kernels", "optimizer", "objective",
                                                "tokens_per_step", "cuda_alloc_conf", "preregistration", "tied_head",
                                                "config_digest", "data_identity_sha256", "base_identity_sha256",
                                                "identity_override", "memory_engine", "parameters_trainable")}
            config["argv"] = argv
            config["data_config"] = _argv_value(argv, "--data-config")
            config["base_dir"] = base_dir
            config["tool"] = {"name": final.get("tool"), "version": final.get("tool_version")}
            config["final_receipt"] = final_path
            for ov in sorted(glob.glob(os.path.join(run_dir, "IDENTITY_OVERRIDE_*.json"))):
                config.setdefault("identity_override_files", []).append(ov)
        parent_key = f"distill:fullkd:{name}"
        parent_id = db.upsert_run(
            conn, parent_key, "chimera-omni", "distill" if not name.startswith("smoke") else "smoke", f"fullkd {name}",
            status, model_id_=student_id, tool_version_id=tool_vid, host_id=host_id, env_id=env_id,
            started_at=started, ended_at=ended, wall_seconds=C.seconds_between(started, ended),
            exit_code=(final or {}).get("exit_code"), config={k: v for k, v in config.items() if v is not None},
            receipt_path=final_path, receipt_sha256=rprov[1] if rprov else None, log_path=qs if lines else None,
            run_dir=run_dir, owner="claude",
            notes="full-parameter KD; parent of its segment runs; curve = merged STEP lines of every segment log",
            prov=qprov or rprov, batch_id=batch.id)
        C.set_run_lane(conn, parent_id, "L-TRAIN")
        if base_id:
            db.link_run_model(conn, parent_id, base_id, "base")
        if student_id:
            db.link_run_model(conn, parent_id, student_id, "output")
        if final:
            for mname, key in (("tokens_seen", "tokens_seen"), ("tok_s", "tokens_per_second_this_run"),
                               ("steps", "step"), ("parameters_trainable", "parameters_trainable"),
                               ("wall_seconds_last_segment", "wall_seconds")):
                v = final.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    db.put_metric(conn, parent_id, mname, float(v), extra={"receipt": final_path})
        batch.wrote()

        # segments
        qstart = None
        segs: dict[int, dict] = {}
        for ts, text, raw in lines:
            m = FKD_QSTART.match(text)
            if m:
                qstart = {"gpu": int(m.group(1)), "data": m.group(2), "data_sha16": m.group(3), "tokens": int(m.group(4)),
                          "micro": int(m.group(5)), "lr": m.group(6), "extra": (m.group(7) or "").split() or None,
                          "trainer_sha16": m.group(8), "line": text}
            m = FKD_SEG_START.match(text)
            if m:
                segs[int(m.group(1))] = {"start": ts, "log": m.group(2), "q": qstart, "lines": [(ts, text, raw)]}
                continue
            m = FKD_SEG_EXIT.match(text)
            if m and int(m.group(1)) in segs:
                s = segs[int(m.group(1))]
                s.update({"end": ts, "rc": int(m.group(2)), "exit_text": m.group(3)})
                s["lines"].append((ts, text, raw))
                continue
            open_seg = max((n for n, s in segs.items() if "end" not in s), default=None)
            target = segs[open_seg]["lines"] if open_seg is not None else None
            if target is not None:
                target.append((ts, text, raw))
            else:
                C.put_run_event(conn, parent_id, ts, "info", raw, qprov, batch.id)
        merged: list[dict] = []
        for n in sorted(segs):
            s = segs[n]
            batch.saw()
            rec_path = None
            m = RECEIPT_IN_TEXT.search(s.get("exit_text") or "")
            if m:
                rec_path = m.group(1)
            rec = docs.get(rec_path) if rec_path else None
            seg_status = exit_status(s["rc"], s.get("exit_text") or "") if "rc" in s else "unknown"
            q = s.get("q") or {}
            seg_tool = C.tool_version_for(conn, str((final or {}).get("tool") or "streamed_full_kd_trainer"),
                                          (rec or {}).get("tool_version"), q.get("trainer_sha16"),
                                          tool_file) if q.get("trainer_sha16") else None
            records = C.log_step_records(s["log"]) if os.path.isfile(s["log"]) else []
            merged.extend(records)
            seg_id = db.upsert_run(
                conn, f"{parent_key}:seg{n}", "chimera-omni", "distill", f"fullkd {name} seg{n}", seg_status,
                model_id_=student_id if seg_status == "pass" else None, tool_version_id=seg_tool, host_id=host_id,
                env_id=env_id, started_at=s["start"], ended_at=s.get("end"),
                wall_seconds=(rec or {}).get("wall_seconds") or C.seconds_between(s["start"], s.get("end")),
                exit_code=s.get("rc"), config={"queue": q, "segment": n, "exit_text": s.get("exit_text"),
                                               "receipt_verdict": (rec or {}).get("verdict"),
                                               "steps_in_log": len(records)},
                receipt_path=rec_path if rec else None, log_path=s["log"], run_dir=run_dir, parent_run_id=parent_id,
                owner="claude", prov=qprov, batch_id=batch.id)
            C.set_run_lane(conn, seg_id, "L-TRAIN")
            _merge_curve(conn, seg_id, records)
            if q.get("gpu") is not None:
                gid = C.gpu_for(conn, index=q["gpu"])
                if gid:
                    seg_kernels = (rec or final or {}).get("kernels") or {}
                    db.add_run_device(conn, seg_id, gid, "compute", kernel_path=seg_kernels.get("chunk_gated_delta_rule"),
                                      kernels=seg_kernels,
                                      detail={"mapping": "index-from-status", "device": (rec or final or {}).get("device"),
                                              "kernels_from": "segment receipt" if rec else "final receipt"})
                    fin_kernels = (final or {}).get("kernels") or {}
                    db.add_run_device(conn, parent_id, gid, "compute", kernel_path=fin_kernels.get("chunk_gated_delta_rule"),
                                      kernels=fin_kernels,
                                      detail={"mapping": "index-from-status", "device": (final or {}).get("device")})
            if rec:
                for mname, key in (("tokens_seen", "tokens_seen"), ("tok_s", "tokens_per_second_this_run"),
                                   ("steps", "step"), ("steps_this_run", "steps_this_run")):
                    v = rec.get(key)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        db.put_metric(conn, seg_id, mname, float(v), extra={"receipt": rec_path})
            for ts, text, raw in s["lines"]:
                C.put_run_event(conn, seg_id, ts, "info", raw, qprov, batch.id)
            batch.wrote()
        _merge_curve(conn, parent_id, merged)
        # receipts no segment names (e.g. smoke runs without a queue): one run each
        named = {RECEIPT_IN_TEXT.search(s.get("exit_text") or "").group(1)
                 for s in segs.values() if RECEIPT_IN_TEXT.search(s.get("exit_text") or "")}
        for p in receipts:
            if p in named or not isinstance(docs[p], dict):
                continue
            batch.saw()
            d = docs[p]
            rp = batch.prov(p)
            rid = db.upsert_run(conn, f"{parent_key}:{os.path.basename(p)}", "chimera-omni",
                                "smoke" if name.startswith("smoke") else "distill", f"fullkd {name} {os.path.basename(p)}",
                                "pass" if d.get("exit_code") == 0 else "fail" if d.get("exit_code") else "unknown",
                                host_id=host_id, env_id=env_id, ended_at=C.parse_ts(C.iso_mtime(p)),
                                wall_seconds=d.get("wall_seconds"), exit_code=d.get("exit_code"),
                                config={k: d.get(k) for k in ("verdict", "device", "dtype", "kernels", "step", "total_steps",
                                                              "tool", "tool_version", "argv")},
                                receipt_path=p, receipt_sha256=rp[1], run_dir=run_dir, parent_run_id=parent_id,
                                notes="ended_at = receipt file mtime", prov=rp, batch_id=batch.id)
            _merge_curve(conn, rid, [x for x in d.get("steps") or [] if isinstance(x, dict)])
            for mname, key in (("tokens_seen", "tokens_seen"), ("tok_s", "tokens_per_second_this_run")):
                v = d.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    db.put_metric(conn, rid, mname, float(v))
            batch.wrote()


# ---------------------------------------------------------------------------
# media M1
# ---------------------------------------------------------------------------

def ingest_media_m1(conn, batch, host_id) -> None:
    for run_dir in sorted(glob.glob(C.LX + "/distill/media_m1/*/")):
        run_dir = run_dir.rstrip("/")
        qs = os.path.join(run_dir, "queue.status")
        if not os.path.isfile(qs):
            continue
        name = os.path.basename(run_dir)
        batch.saw()
        qprov = batch.prov(qs)
        lines = status_lines(qs)
        parent_key = f"distill:media_m1:{name}"
        attempts, current = [], None
        ordinal = {"probe": 0, "train": 0}
        for ts, text, raw in lines:
            m = M1_START.match(text)
            if m:
                kind = m.group(1)
                current = {"kind": kind, "ordinal": ordinal[kind], "start": ts, "data": m.group(2),
                           "data_sha16": m.group(3), "trainer_sha16": m.group(4), "steps": int(m.group(5)),
                           "lr": m.group(6), "lines": [raw]}
                ordinal[kind] += 1
                attempts.append(current)
                continue
            m = M1_EXIT.match(text)
            if m and current and current["kind"] == m.group(1) and "rc" not in current:
                current.update({"end": ts, "rc": int(m.group(2)), "exit_text": m.group(3)})
                current["lines"].append(raw)
                continue
            if current:
                current["lines"].append(raw)
        export_dir = os.path.join(run_dir, "out", "export")
        final = None
        for a in attempts:
            rm = RECEIPT_IN_TEXT.search(a.get("exit_text") or "")
            a["receipt"] = rm.group(1) if rm else None
            if a["kind"] == "train" and a.get("rc") == 0 and a["receipt"]:
                final = a
        final_doc = C.load_json(final["receipt"]) if final and os.path.isfile(final["receipt"]) else None
        argv = (final_doc or {}).get("argv") or []
        base_id = C.model_id_by_path(conn, _argv_value(argv, "--omni1"))
        rprov = batch.prov(final["receipt"]) if final_doc else None
        export_id = _model_for_dir(conn, batch, export_dir, "media", "media-unit", "safetensors-raw", "M1", "OMNI1-media",
                                   f"media M1 export of {name} (trained media blocks + heads; trunk frozen); "
                                   "not registered in the ledger", base_id, rprov or qprov) if final_doc else None
        ftool = (final_doc or {}).get("tool") if isinstance((final_doc or {}).get("tool"), dict) else {}
        tool_vid = C.tool_version_for(conn, ftool.get("name") or "media_full_kd_trainer", ftool.get("version"),
                                      ftool.get("sha256"), ftool.get("path")) if ftool else None
        env_id = db.ensure_env(conn, torch=final_doc.get("torch")) if final_doc and final_doc.get("torch") else None
        last = attempts[-1] if attempts else None
        status = ("pass" if final and last is final else
                  "unknown" if last and "rc" not in last else "fail" if last else "unknown")
        started = attempts[0]["start"] if attempts else None
        ended = (last or {}).get("end")
        parent_id = db.upsert_run(
            conn, parent_key, "media", "distill", f"media M1 {name}", status, model_id_=export_id, tool_version_id=tool_vid,
            host_id=host_id, env_id=env_id, started_at=started, ended_at=ended,
            config={"argv": argv, "card": (final_doc or {}).get("card"), "device": (final_doc or {}).get("device"),
                    "dtype": (final_doc or {}).get("dtype"), "depth": (final_doc or {}).get("depth"),
                    "trunk": (final_doc or {}).get("trunk"), "data": (final_doc or {}).get("data"),
                    "tool": ftool, "trainable_parameters": (final_doc or {}).get("trainable_parameters"),
                    "final_receipt": final["receipt"] if final else None},
            receipt_path=final["receipt"] if final else None, receipt_sha256=rprov[1] if rprov else None, log_path=qs,
            run_dir=run_dir, owner="claude", notes="media M1 full-KD (media blocks + heads); parent of its attempts",
            prov=qprov, batch_id=batch.id)
        C.set_run_lane(conn, parent_id, "L-TRAIN")
        if base_id:
            db.link_run_model(conn, parent_id, base_id, "base")
        if export_id:
            db.link_run_model(conn, parent_id, export_id, "output")
        batch.wrote()
        for a in attempts:
            batch.saw()
            log = os.path.join(run_dir, f"{a['kind']}-{a['ordinal']:03d}.log")
            doc = C.load_json(a["receipt"]) if a.get("receipt") and os.path.isfile(a["receipt"]) else None
            mapping_ok = (a.get("receipt") is None or
                          os.path.basename(a["receipt"]) == f"receipt-{a['kind']}-{a['ordinal']:03d}.json")
            st = exit_status(a["rc"], a.get("exit_text") or "") if "rc" in a else "unknown"
            aid = db.upsert_run(
                conn, f"{parent_key}:{a['kind']}-{a['ordinal']:03d}", "media",
                "probe" if a["kind"] == "probe" else "distill", f"media M1 {name} {a['kind']}-{a['ordinal']:03d}", st,
                model_id_=export_id if a is final else None,
                tool_version_id=C.tool_version_for(conn, "media_full_kd_trainer", None, a["trainer_sha16"],
                                                   ftool.get("path") or C.OC + "/distill/tools/media_full_kd_trainer.py"),
                host_id=host_id, env_id=env_id if doc else None, started_at=a["start"], ended_at=a.get("end"),
                wall_seconds=(doc or {}).get("seconds") or C.seconds_between(a["start"], a.get("end")), exit_code=a.get("rc"),
                config={"data": a["data"], "data_sha16": a["data_sha16"], "trainer_sha16": a["trainer_sha16"],
                        "steps": a["steps"], "lr": a["lr"], "exit_text": a.get("exit_text"),
                        "log_mapping": "ordinal" + ("" if mapping_ok else " (MISMATCH with receipt name)"),
                        "card": (doc or {}).get("card"), "dtype": (doc or {}).get("dtype"),
                        "device": (doc or {}).get("device"), "argv": (doc or {}).get("argv")},
                receipt_path=a.get("receipt") if doc else None, log_path=log if os.path.isfile(log) else None,
                run_dir=run_dir, parent_run_id=parent_id, owner="claude", prov=qprov, batch_id=batch.id)
            C.set_run_lane(conn, aid, "L-TRAIN")
            history = [h for h in (doc or {}).get("history") or [] if isinstance(h, dict)]
            _merge_curve(conn, aid, history or (C.log_step_records(log) if os.path.isfile(log) else []))
            for ev in (doc or {}).get("evals") or []:
                if not isinstance(ev, dict) or not isinstance(ev.get("step"), int):
                    continue
                for route, r in (ev.get("routes") or {}).items():
                    if isinstance(r, dict) and isinstance(r.get("mean"), (int, float)):
                        db.put_metric(conn, aid, f"eval/{route}", float(r["mean"]), step=ev["step"],
                                      extra={"items": r.get("items"), "tag": ev.get("tag")})
            for mname, key in (("trainable_parameters", "trainable_parameters"), ("steps_done", "steps_done"),
                               ("seconds", "seconds")):
                v = (doc or {}).get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    db.put_metric(conn, aid, mname, float(v))
            for raw in a["lines"]:
                m = C.TS_RE.match(raw.strip())
                C.put_run_event(conn, aid, C.parse_ts(m.group(1)) if m else None, "info", raw, qprov, batch.id)
            batch.wrote()


def ingest_selftests(conn, batch, host_id) -> None:
    for p in sorted(glob.glob(C.LX + "/distill/media_m1/selftest*.json") + glob.glob(C.LX + "/distill/m2/selftest*.json")):
        doc = C.load_json(p)
        if not isinstance(doc, dict):
            continue
        batch.saw()
        prov = batch.prov(p)
        lane = "m2" if "/m2/" in p else "media_m1"
        tool = doc.get("tool") if isinstance(doc.get("tool"), dict) else {"name": doc.get("tool")}
        verdict = str(doc.get("verdict") or "")
        rid = db.upsert_run(conn, f"smoke:{lane}:{os.path.basename(p)}", "media", "smoke", f"{lane} {os.path.basename(p)}",
                            "pass" if verdict.upper() == "PASS" else "fail" if verdict else "unknown",
                            tool_version_id=C.tool_version_for(conn, str(tool.get("name")), tool.get("version"),
                                                               tool.get("sha256"), tool.get("path")) if tool.get("name") else None,
                            host_id=host_id, ended_at=C.parse_ts(C.iso_mtime(p)),
                            config={k: doc.get(k) for k in ("schema", "tool", "verdict", "torch", "refusals", "music_parity",
                                                            "joint_training", "media_step_parity") if k in doc},
                            receipt_path=p, receipt_sha256=prov[1], notes="ended_at = file mtime", prov=prov,
                            batch_id=batch.id)
        C.set_run_lane(conn, rid, "L-TRAIN")
        batch.wrote()


# ---------------------------------------------------------------------------
# D2 short train
# ---------------------------------------------------------------------------

def ingest_d2(conn, batch, host_id) -> None:
    qs = C.Q + "/d2.status"
    if not os.path.isfile(qs):
        return
    qprov = batch.prov(qs)
    groups, cur = [], None
    for ts, text, raw in status_lines(qs):
        if " queue start " in f" {text} " or cur is None:
            cur = {"lines": [], "start": ts}
            groups.append(cur)
        cur["lines"].append((ts, text, raw))
    for g in groups:
        batch.saw()
        text_all = "\n".join(t for _ts, t, _r in g["lines"])
        train = next(((ts, t) for ts, t, _r in g["lines"] if "train short start" in t), None)
        log = re.search(r"log=(\S+)", train[1]).group(1) if train and re.search(r"log=(\S+)", train[1]) else None
        base = re.search(r"\bbase=(\S+)", text_all)
        trainer = re.search(r"\btrainer=([0-9a-f]+)", text_all)
        data = re.search(r"\bdata (\S+) sha ([0-9a-f]+) max_width (\d+) rows (\d+)", text_all)
        micro = re.search(r"micro (\d+) acc (\d+)(?: world (\d+))?", text_all)
        ver = re.search(r"\b(d2_launch)-(\d+\.\d+\.\d+)", text_all)
        uuids = UUID_RX.findall(text_all)
        records = C.log_step_records(log) if log and os.path.isfile(log) else []
        tail = C.read_text(log)[-4000:] if log and os.path.isfile(log) else ""
        terminal = re.search(r"train short (?:exit|end|done)\D*(\d+)?", text_all)
        if terminal:
            status = "pass" if (terminal.group(1) or "0") == "0" else "fail"
            ended = g["lines"][-1][0]
            note = None
        elif "got signal: 15" in tail or "SIGTERM" in tail:
            status, ended = "stopped", C.parse_ts(C.iso_mtime(log))
            note = "no exit line in d2.status; the train log ends in a SIGTERM (signal 15) traceback; ended_at = log mtime"
        elif "Traceback" in tail:
            status, ended = "fail", C.parse_ts(C.iso_mtime(log))
            note = "no exit line in d2.status; the train log ends in a traceback; ended_at = log mtime"
        elif "refused" in text_all.lower() and not train:
            status, ended, note = "refused", g["lines"][-1][0], None
        else:
            status, ended, note = "unknown", None, "no exit line in d2.status (process state UNVERIFIED)"
        base_id = db.model_id(conn, base.group(1)) if base else None
        started = train[0] if train else g["start"]
        rid = db.upsert_run(
            conn, f"distill:d2:{'train_short' if train else 'queue'}:{started.strftime('%Y%m%dT%H%M%SZ')}", "chimera-omni",
            "distill", f"D2 {'short train' if train else 'queue'} {started:%Y-%m-%d %H:%M}Z", status,
            tool_version_id=C.tool_version_for(conn, "d2_launch", ver.group(2)) if ver else None, host_id=host_id,
            started_at=started, ended_at=ended, wall_seconds=C.seconds_between(started, ended),
            config={"base": base.group(1) if base else None, "trainer_sha16": trainer.group(1) if trainer else None,
                    "data": data.group(1) if data else None, "data_sha16": data.group(2) if data else None,
                    "max_width": int(data.group(3)) if data else None, "rows": int(data.group(4)) if data else None,
                    "micro": int(micro.group(1)) if micro else None, "accumulation": int(micro.group(2)) if micro else None,
                    "world": int(micro.group(3)) if micro and micro.group(3) else None,
                    "launcher": f"{ver.group(1)}-{ver.group(2)}" if ver else None, "steps_in_log": len(records),
                    "cuda_visible_devices": uuids or None},
            log_path=log, run_dir=C.LX + "/distill/d2", owner="claude", notes=note, prov=qprov, batch_id=batch.id)
        C.set_run_lane(conn, rid, "L-TRAIN")
        if base_id:
            db.link_run_model(conn, rid, base_id, "base")
        for u in dict.fromkeys(uuids):
            gid = C.gpu_for(conn, uuid=u)
            if gid:
                db.add_run_device(conn, rid, gid, "compute", detail={"mapping": "uuid-from-status"})
        _merge_curve(conn, rid, records)
        for ts, text, raw in g["lines"]:
            C.put_run_event(conn, rid, ts, "info", raw, qprov, batch.id)
        batch.wrote()


# ---------------------------------------------------------------------------
# D1 run directories
# ---------------------------------------------------------------------------

def ingest_d1(conn, batch, host_id) -> None:
    for run_dir in sorted(glob.glob(C.LX + "/distill/d1/run-*/")):
        run_dir = run_dir.rstrip("/")
        name = os.path.basename(run_dir)
        st_path = os.path.join(run_dir, "runtime-status.json")
        st = C.load_json(st_path) if os.path.isfile(st_path) else None
        if not isinstance(st, dict):
            continue
        batch.saw()
        prov = batch.prov(st_path)
        launch = C.load_json(os.path.join(run_dir, "launch.json")) or {}
        receipt_path = next((p for p in (os.path.join(run_dir, "trainer", "d1_receipt.json"),) if os.path.isfile(p)), None)
        rec = C.load_json(receipt_path) if receipt_path else None
        rec = rec if isinstance(rec, dict) else None
        owner = "claude" if "claude" in name else "codex" if "01a10f78" in name or "native" in name else None
        if st.get("state") == "succeeded" or (st.get("completed") and st.get("exit_code") == 0):
            status = "pass"
        elif st.get("completed") and st.get("exit_code"):
            status = "refused" if "refus" in str(st.get("error") or "") else "fail"
        else:
            status = "unknown"
        argv = st.get("argv") or launch.get("trainer_argv") or (rec or {}).get("argv") or []
        base_id = C.model_id_by_path(conn, _argv_value(argv, "--child"))
        delta_dir = ((rec or {}).get("delta") or {}).get("dir")
        out_id = C.model_id_by_path(conn, delta_dir) if delta_dir else None
        tool = (rec or {}).get("tool") if isinstance((rec or {}).get("tool"), dict) else {}
        tool_vid = C.tool_version_for(conn, tool.get("name"), tool.get("version"), tool.get("sha256")) if tool.get("name") else None
        env_id = db.ensure_env(conn, torch=rec.get("torch"), code_commit=((rec.get("chimera") or {}).get("commit")),
                               env_vars={"CUDA_VISIBLE_DEVICES": launch.get("cuda_visible_devices")}) if rec and rec.get("torch") else None
        started = C.parse_ts(st.get("started_at") or launch.get("launched_at_utc"))
        ended = C.parse_ts(st.get("finished_at"))
        rid = db.upsert_run(
            conn, f"distill:d1:{name}", "chimera-omni", "distill", f"D1 {name}", status, model_id_=out_id,
            tool_version_id=tool_vid, host_id=host_id, env_id=env_id, started_at=started, ended_at=ended,
            wall_seconds=(rec or {}).get("seconds") or C.seconds_between(started, ended), exit_code=st.get("exit_code"),
            config={"argv": argv, "launcher": launch.get("launcher"), "unit": launch.get("unit"),
                    "memory_max": launch.get("memory_max"), "cuda_visible_devices": launch.get("cuda_visible_devices"),
                    "cards": (rec or {}).get("cards"), "rows": (rec or {}).get("rows"),
                    "train_layers": len((rec or {}).get("train_layers") or []) or None,
                    "preregistration": ((rec or {}).get("preregistration") or {}).get("path"),
                    "error": st.get("error"), "state": st.get("state"), "completed": st.get("completed"),
                    "tool": tool or st.get("tool"), "verdict": (rec or {}).get("verdict")},
            receipt_path=receipt_path, log_path=os.path.join(run_dir, "runtime-log.txt"), run_dir=run_dir, owner=owner,
            notes=None if status != "unknown" else "runtime-status.json has completed=false and no exit (process state UNVERIFIED)",
            prov=prov, batch_id=batch.id)
        C.set_run_lane(conn, rid, "L-D1")
        if base_id:
            db.link_run_model(conn, rid, base_id, "base")
        if out_id:
            db.link_run_model(conn, rid, out_id, "output")
        for u in UUID_RX.findall(str(launch.get("cuda_visible_devices") or "")):
            gid = C.gpu_for(conn, uuid=u)
            if gid:
                db.add_run_device(conn, rid, gid, "compute", detail={"mapping": "uuid-from-launch"})
        hk = (((rec or {}).get("head_kl") or {}).get("d1") or {})
        for mname, v in (("head_kl_mean", hk.get("kl_mean")), ("head_top1_agreement", hk.get("top1_agreement")),
                         ("seconds", (rec or {}).get("seconds")), ("peak_rss_gb", (rec or {}).get("peak_rss_gb")),
                         ("fit_rows", ((rec or {}).get("rows") or {}).get("fit")),
                         ("held_rows", ((rec or {}).get("rows") or {}).get("held")),
                         ("valid_tokens", ((rec or {}).get("rows") or {}).get("valid_tokens"))):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                db.put_metric(conn, rid, mname, float(v), extra={"receipt": receipt_path})
        batch.wrote()


def register_receipt_tools(conn) -> None:
    """Tool versions named (with full sha) in any stage-D receipt, incl. nested trainer tools, so runs whose
    status lines carry only a sha16 resolve to a version."""
    paths = (glob.glob(C.LX + "/distill/media_m1/*.json") + glob.glob(C.LX + "/distill/media_m1/*/receipt-*.json")
             + glob.glob(C.LX + "/distill/m2/*.json"))
    for p in sorted(paths):
        doc = C.load_json(p)
        tool = doc.get("tool") if isinstance(doc, dict) else None
        if not isinstance(tool, dict):
            continue
        for t in [tool] + [v for v in tool.values() if isinstance(v, dict)]:
            if t.get("name") or t.get("path"):
                name = t.get("name") or os.path.splitext(os.path.basename(t["path"]))[0]
                if isinstance(t.get("sha256"), str) and len(t["sha256"]) == 64:
                    C.tool_version_for(conn, name, t.get("version"), t["sha256"], t.get("path"))


def run(conn, batch: C.Batch, args) -> None:
    host_id = C.host_id(conn)
    register_receipt_tools(conn)
    ingest_fullkd(conn, batch, host_id)
    ingest_selftests(conn, batch, host_id)
    ingest_media_m1(conn, batch, host_id)
    ingest_d2(conn, batch, host_id)
    ingest_d1(conn, batch, host_id)
