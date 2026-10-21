"""ingest.lab_runs -- run directories written by `omni_lab.py run` (spec.json + receipt.json + results.jsonl).

`omni_lab.py run` calls ingest_dir() itself the moment its worker exits; this source re-reads every run
directory under $OMNI_LAB_RUNS (default $L/scratch/omni-lab-runs) so a run made while the database was down,
or interrupted before its ingest, lands later with the same rows (run_key and (eval_run, case) are the keys).

Writes per run directory
  * run (kind eval, key = spec.run_key) + eval_run (suite, engine, gen, tokenizer / chat-template sha, thinking
    requested vs took effect, case filter, test, baseline), env (runtime versions, driver, CUDA env), tool_version
    of the worker (sha of its code), run_device per card BY UUID with role / kernel path / layer span / peak MiB.
  * eval_result per case (termination in the lab vocabulary, form box, artifact flags, latency, decode tok/s, raw
    receipt fields) and eval_grade per machine grade the worker recorded.  An oracle that said 'ungraded' writes
    no grade row (the cell reads 'ungraded', never a guess).
  * metric (step 0): load_seconds, wall_seconds, decode_tok_s_mean, n_results, n_unsupported.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from . import common as C
from . import evalkit as K
from .common import db

SOURCE = "lab_runs"
WORKER_TOOL = "omni_lab_worker"
TERMINATION = {"complete": "eos", "censored-at-cap": "cap", "runtime-error": "error", "out-of-memory": "error",
               "unsupported-modality": "unsupported-modality", "other": "other", "error": "error",
               "timeout": "timeout"}
STATUS = {"pass": "pass", "fail": "fail", "load-failed": "fail", "loading": "fail", "running": "fail",
          "stopped": "stopped", "refused": "refused"}


def runs_root() -> str:
    return os.environ.get("OMNI_LAB_RUNS", f"{C.LX}/scratch/omni-lab-runs")


def run_dirs() -> list[str]:
    return sorted(str(Path(p).parent) for p in glob.glob(f"{runs_root()}/*/spec.json"))


def read_results(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                       # a torn last line from a killed worker
    return out


def flags_of(row: dict, term: str | None) -> list[str]:
    flags = K.machine_flags(row.get("output_text"), row.get("output_tokens"), row.get("loop_detected"), term,
                            row.get("form_box"))
    if row.get("termination") == "out-of-memory":
        flags.append("oom")
    if term in ("unsupported-modality", "other") and row.get("unsupported_reason"):
        flags.append("engine-unsupported")
        flags = [f for f in flags if f != "empty"]
    if row.get("grader_errors"):
        flags.append("grader-error")
    return sorted(set(flags))


def ingest_dir(conn, batch: C.Batch, run_dir: str, exit_code: int | None = None) -> int | None:
    """One run directory -> rows.  Returns the run id, or None when the directory has no usable spec."""
    d = Path(run_dir)
    spec = C.load_json(d / "spec.json")
    if not isinstance(spec, dict) or spec.get("schema") != "omni-lab-run-spec-v1":
        batch.skip(f"{run_dir}: no omni-lab spec.json")
        return None
    host = spec.get("host") or db.this_host()
    sprov = batch.prov(d / "spec.json", host)
    receipt = C.load_json(d / "receipt.json") if (d / "receipt.json").is_file() else None
    rprov = batch.prov(d / "receipt.json", host) if receipt is not None else sprov
    receipt = receipt or {}
    report = receipt.get("report") or {}
    if exit_code is None and (d / "worker.exit").is_file():
        try:
            exit_code = int((d / "worker.exit").read_text().strip())
        except ValueError:
            exit_code = None

    hid = db.ensure_host(conn, host) if host != C.SERVER_HOST else C.host_id(conn)
    m = spec["model"]
    model = db.model_id(conn, m["model_key"])
    if model is None:
        model = db.upsert_model(conn, m["model_key"], m.get("project") or "chimera-omni", m.get("role") or "other",
                                m.get("layout") or "unknown", family=m.get("family"), stage=m.get("stage"),
                                path=m.get("path"), host_id=hid, notes="created by omni_lab run", prov=sprov,
                                batch_id=batch.id)
    suite = db.suite_id(conn, spec["suite"]["name"], spec["suite"]["version"])
    if suite is None:
        batch.skip(f"{run_dir}: suite {spec['suite']['name']}@{spec['suite']['version']} not in this database")
        return None
    test_id = None
    if spec.get("test_name"):
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM test WHERE name = %s", (spec["test_name"],))
            row = cur.fetchone()
            test_id = row[0] if row else None

    rt = receipt.get("runtime") or {}
    shas = receipt.get("code_sha256") or {}
    tool_v = db.ensure_tool_version(conn, WORKER_TOOL, receipt.get("version") or spec.get("worker_version") or "unknown",
                                    db.sha256_json(shas) if shas else db.sha256_json(spec.get("code_sha256") or {}),
                                    path=f"{spec.get('lab_dir')}/omni_lab_worker.py")
    env_vars = {k: v for k, v in (receipt.get("env_vars") or {}).items() if v is not None}
    env_id = db.ensure_env(conn, python=rt.get("python"), torch=rt.get("torch"), cuda=rt.get("cuda"),
                           driver=spec["device"].get("driver"), transformers=rt.get("transformers"), fla=rt.get("fla"),
                           triton=rt.get("triton"), code_commit=spec.get("code_commit"), env_vars=env_vars,
                           packages={"accelerate": rt.get("accelerate"), "safetensors": rt.get("safetensors"),
                                     "tokenizers": rt.get("tokenizers"),
                                     **{f"sha:{k}": (v or {}).get("sha256") for k, v in (receipt.get("tool_sha256") or {}).items()}})
    status = STATUS.get(receipt.get("status"), "unknown") if receipt else "fail"
    started = C.parse_ts(receipt.get("started_utc")) or C.parse_ts(spec.get("created_utc"))
    ended = C.parse_ts(receipt.get("ended_utc"))
    config = dict(spec.get("config") or {})
    config.update({"kernel_path": report.get("kernel_path"), "kernels": report.get("kernels"),
                   "placement": report.get("placement"), "load": report.get("load"),
                   "stops": report.get("stops"), "eos_id": report.get("eos_id"),
                   "ledger_band_mask_effective": report.get("ledger_band_mask"),
                   "tool_sha256": receipt.get("tool_sha256"), "worker_code_sha256": shas,
                   "load_seconds": receipt.get("load_seconds"), "worker_error": receipt.get("error"),
                   "runtime": rt})
    rid = db.upsert_run(conn, spec["run_key"], spec.get("project") or "chimera-omni", "eval", spec["name"], status,
                        model_id_=model, tool_version_id=tool_v, host_id=hid, env_id=env_id, started_at=started,
                        ended_at=ended, wall_seconds=receipt.get("wall_seconds"), exit_code=exit_code, config=config,
                        receipt_path=rprov[0], receipt_sha256=rprov[1], log_path=str(d / "worker.log"), run_dir=str(d),
                        owner=spec.get("owner"), notes=spec.get("notes"), prov=rprov, batch_id=batch.id)
    tok = report.get("tokenizer") or {}
    thinking = report.get("thinking") or {}
    db.upsert_eval_run(conn, rid, model, suite, spec["engine"], gen=spec["gen"], test_id=test_id,
                       tokenizer_sha256=tok.get("tokenizer_json_sha256") or tok.get("tokenizer_sha256"),
                       chat_template_sha256=tok.get("chat_template_sha256"),
                       bank_file_sha256=spec["suite"].get("bank_sha256"),
                       thinking_requested=spec["gen"].get("thinking"), thinking_took_effect=thinking.get("took_effect"),
                       case_filter=spec.get("case_filter"), label=spec.get("label"),
                       baseline_run_id=spec.get("baseline_run_id"))
    for role, key in (("baseline", spec.get("baseline_model_key")),):
        bm = db.model_id(conn, key) if key else None
        if bm is not None and bm != model:
            db.link_run_model(conn, rid, bm, role)

    # placement: cuda:i -> spec card i (UUID from nvidia-smi at claim time) -> gpu row
    K.clear_devices(conn, rid)
    cards = spec["device"].get("cards") or []
    placed = {p.get("cuda"): p for p in (report.get("placement") or {}).get("devices") or [] if isinstance(p, dict)}
    torch_cards = {c.get("cuda"): c for c in rt.get("cards") or []}
    peaks = receipt.get("peak_allocated_mib") or []
    for i, card in enumerate(cards):
        gid = db.gpu_id_by_uuid(conn, hid, card["uuid"])
        if gid is None:
            gid = db.upsert_gpu(conn, hid, card["index"], card["uuid"], card["name"], card.get("sm"), card.get("vram_mib"))
        p = placed.get(i, {})
        layers = p.get("layers") or []
        span = (layers[0], layers[-1] + 1) if layers and layers == list(range(layers[0], layers[-1] + 1)) else None
        tc = torch_cards.get(i, {})
        db.add_run_device(conn, rid, gid, role=p.get("role") or "compute",
                          kernel_path=p.get("kernel_path") or report.get("kernel_path"),
                          kernels=report.get("kernels") if isinstance(report.get("kernels"), dict) else {},
                          layers=span, mem_mib_peak=peaks[i] if i < len(peaks) else None,
                          detail={"cuda": i, "index_at_claim": card["index"], "mapping": "uuid-at-claim",
                                  "torch_uuid": tc.get("uuid"),
                                  "torch_uuid_matches": (tc.get("uuid") == card["uuid"]) if tc.get("uuid") else None,
                                  "used_mib_at_claim": card.get("used_mib"), "layers": layers,
                                  "other_modules": p.get("other_modules")})

    rows = read_results(d / "results.jsonl")
    rsprov = batch.prov(d / "results.jsonl", host) if (d / "results.jsonl").is_file() else rprov
    n_unsupported = 0
    tps = []
    for row in rows:
        batch.saw()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM test_case WHERE case_id = %s AND content_sha256 = %s",
                        (row["case_id"], row["content_sha256"]))
            hit = cur.fetchone()
        if hit is None:
            batch.skip(f"{run_dir}: case {row['case_id']} ({row['content_sha256'][:12]}) not in this database")
            continue
        term = TERMINATION.get(row.get("termination"), "other") if row.get("termination") else None
        if row.get("runtime_error") and term not in ("unsupported-modality",):
            term = "error"
        n_unsupported += term == "unsupported-modality"
        wall = row.get("wall_time_seconds")
        if isinstance(row.get("decode_tok_per_s"), (int, float)):
            tps.append(row["decode_tok_per_s"])
        res = db.upsert_eval_result(
            conn, rid, hit[0], output_text=row.get("output_text"), output_tokens=row.get("output_tokens"),
            termination=term, form=row.get("form_box"), artifact_flags=flags_of(row, term),
            latency_ms=int(wall * 1000) if isinstance(wall, (int, float)) else None, tok_s=row.get("decode_tok_per_s"),
            raw={k: row.get(k) for k in ("prompt_tokens", "termination", "runtime_error", "response_sha256",
                                         "prefill_seconds", "generated_ids_match_runner_sha", "unsupported_reason",
                                         "server", "form", "grader_errors") if row.get(k) is not None},
            prov=rsprov, batch_id=batch.id)
        batch.wrote()
        for g in row.get("grades") or []:
            if not g.get("grade"):
                continue                       # the oracle said ungraded: no row, the reason stays in raw
            db.put_grade(conn, res, K.grader_id(conn, g["grader"]), g["grade"], score=g.get("score"), why=g.get("why"),
                         detail=g.get("detail"), prov=rsprov, batch_id=batch.id)
            batch.wrote()
    for name, value in (("load_seconds", receipt.get("load_seconds")), ("wall_seconds", receipt.get("wall_seconds")),
                        ("decode_tok_s_mean", sum(tps) / len(tps) if tps else None), ("n_results", len(rows)),
                        ("n_unsupported", n_unsupported)):
        if isinstance(value, (int, float)):
            db.put_metric(conn, rid, name, value, denominator=len(spec["cases"]) if name.startswith("n_") else None)
    return rid


def run(conn, batch: C.Batch, args) -> None:
    for d in run_dirs():
        batch.saw()
        ingest_dir(conn, batch, d)
