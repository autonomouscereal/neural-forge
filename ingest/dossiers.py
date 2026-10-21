"""ingest.dossiers -- Chimera-native review dossiers (run_omni_frozen_review) -> eval runs + results.

Default sources: $OC/composite/{s0,s3,s8,d1}/review/oc-dossier-*.json (schema
neural-forge-omni-parent-calibration-dossier-v1): S0 parent import, S3 C2K4352 native, S5 C2K4352K3U
teacher-forced proxy, S8 OMNI1, D1 NOTHINK + TEMPLATE.  OMNI_LAB_DOSSIERS (os.pathsep-separated) overrides.

Each dossier becomes up to three eval runs, all on the same model and devices:
  'eval:dossier:<label>:<file sha12>'        bank_v1 (child_responses, 16 cases)
  'eval:dossier-smoke:<label>:<file sha12>'  smoke   (decode_smoke.records)
  'eval:dossier-probe:<label>:<file sha12>'  probe   (emit_zero_floor.probes: logit/terminal stats, no text)
Model under test: the ledger checkpoint whose review record names this dossier_sha256, else the ledger
checkpoint of a sibling dossier on the same model_directory, else the ledger directory, else a
'path:<sha16>' row (the S0 parent import, linked 'export' to its HF parent).  Sidecars read when present:
<dossier>.finite_guard.json (devices, per-card peak memory, per-case finiteness) and review-*.json in the
same directory whose dossier_file_sha256 is this file (machine upper-bound columns -> metric, step 0).
No grades are written here: the bank is reader-graded (student_tests attaches the reader's grades).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from . import common as C
from . import evalkit as K
from .common import db

SOURCE = "dossiers"
SCHEMA = "neural-forge-omni-parent-calibration-dossier-v1"
STAGES = ("s0", "s3", "s8", "d1")


def dossier_paths() -> list[str]:
    override = os.environ.get("OMNI_LAB_DOSSIERS", "")
    if override:
        return [p for p in override.split(os.pathsep) if p]
    out = []
    for stage in STAGES:
        out += [p for p in sorted(glob.glob(f"{C.OC}/composite/{stage}/review/oc-dossier-*.json"))
                if not p.endswith(".finite_guard.json")]
    return out


def ledger_reviews_by_dossier() -> dict[str, str]:
    """{dossier_sha256: checkpoint_id} from the ledger's review records."""
    out = {}
    for line in C.read_text(C.OC + "/ledger.jsonl").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("event") == "review" and rec.get("dossier_sha256"):
            out[rec["dossier_sha256"]] = rec["checkpoint_id"]
    return out


def kernel_label(kernels: dict | None) -> str | None:
    path = (kernels or {}).get("kernel_path") or ""
    if path.startswith("chimera-reference"):
        return "chimera-reference-fp32-kda"
    return path.split(" ", 1)[0] or None


def resolve_model(conn, batch: C.Batch, d: dict, reviews: dict[str, str], by_dir: dict[str, str],
                  prov: tuple[str, str, str]) -> tuple[int, str]:
    """(model id, how it was resolved)."""
    mdir = ((d.get("provenance") or {}).get("model_directory") or "").rstrip("/")
    cid = reviews.get(d.get("dossier_sha256") or "")
    how = "ledger review record names this dossier_sha256"
    if cid is None and mdir in by_dir:
        cid, how = by_dir[mdir], "sibling dossier on the same model_directory is named by a ledger review"
    if cid is not None:
        mid = K.ensure_ledger_model(conn, cid)
        if mid is not None:
            return mid, how
    ck = (d.get("export") or {}).get("checkpoint") or {}
    source_dir = ((ck.get("source") or {}).get("model_dir") or "").rstrip("/")
    lineage = []
    src_rec = K.ledger_by_directory().get(source_dir) if source_dir else None
    if src_rec is not None:
        lineage.append((src_rec["checkpoint_id"], "export",
                        {"via": "chimera_omni import", "import_source": (ck.get("source") or {}).get("import_source"),
                         "import_receipt": (d.get("provenance") or {}).get("import_receipt")}))
    is_s0 = "/composite/s0/" in mdir
    mid = K.ensure_model_at(conn, mdir, role="intermediate" if is_s0 else "other", layout="chimera.checkpoint.v1",
                            family="Qwen3.8-27B-U" if d.get("parent_code") == "Q38-27BU" else None,
                            stage="S0" if is_s0 else None,
                            config={"directory_digest_oc_calib": (d.get("provenance") or {}).get("directory_sha256"),
                                    "declared_class": (d.get("export") or {}).get("declared_class"),
                                    "parameters_counted_live": (d.get("export") or {}).get("parameters_counted_live")},
                            notes=f"model under test of dossier {d.get('label')}", lineage=lineage, prov=prov,
                            batch_id=batch.id)
    return mid, "ledger directory or path key"


def review_metrics(path: str, file_sha: str) -> tuple[dict[str, tuple[float, float | None]], list[str]]:
    """Upper-bound columns from review-*.json files that name this dossier file sha."""
    metrics: dict[str, tuple[float, float | None]] = {}
    used = []
    for rp in sorted(glob.glob(str(Path(path).parent / "review-*.json"))):
        r = C.load_json(rp)
        if not isinstance(r, dict) or r.get("dossier_file_sha256") != file_sha:
            continue
        used.append(rp)
        b1 = r.get("bank1") or {}
        total = b1.get("lanes_total")
        for name, value in (("column_a", (r.get("columns") or {}).get("column_a")),
                            ("column_b", (r.get("columns") or {}).get("column_b")),
                            ("canonical_signal", b1.get("canonical_signal")),
                            ("observed_miss", b1.get("observed_miss")),
                            ("canonical_artifact_miss", b1.get("canonical_artifact_miss")),
                            ("canonical_content_miss", b1.get("canonical_content_miss")),
                            ("partial", b1.get("partial")), ("useful", b1.get("useful"))):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"upper_bound/{name}"] = (float(value), float(total) if isinstance(total, (int, float)) else None)
    return metrics, used


def ingest_dossier(conn, batch: C.Batch, path: str, reviews: dict[str, str], by_dir: dict[str, str]) -> None:
    prov = batch.prov(path)
    d = C.load_json(path)
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        batch.skip(f"{path}: not a {SCHEMA} file")
        return
    label = d.get("label") or Path(path).stem
    gen = d.get("generation") or {}
    exp = d.get("export") or {}
    place = exp.get("placement") or {}
    cuda = place.get("cuda") or {}
    guard_path = path + ".finite_guard.json"
    guard = C.load_json(guard_path) if os.path.isfile(guard_path) else None
    if guard is not None:
        batch.prov(guard_path)
    cvd = (guard or {}).get("CUDA_VISIBLE_DEVICES") or cuda.get("CUDA_VISIBLE_DEVICES")
    order = cuda.get("CUDA_DEVICE_ORDER")

    model, how = resolve_model(conn, batch, d, reviews, by_dir, prov)
    mdir = ((d.get("provenance") or {}).get("model_directory") or "").rstrip("/")
    runner = d.get("runner") or {}
    tool_v = C.tool_version_for(conn, runner.get("tool") or "run_omni_frozen_review", runner.get("version"),
                                runner.get("tool_sha256"))
    env_id = K.env_for(conn, torch=exp.get("torch_version"), transformers=exp.get("transformers_version"),
                       code_commit=(exp.get("chimera_tree") or {}).get("commit"),
                       env_vars={"CUDA_VISIBLE_DEVICES": cvd, "CUDA_DEVICE_ORDER": order})
    kernels = exp.get("linear_attention_kernels") or {}
    thinking = gen.get("thinking") or {}
    started, ended = K.started_from(d.get("generated_at"), d.get("wall_time_seconds"))
    base_config = {"runtime": gen.get("runtime"), "mode": d.get("mode") or gen.get("mode"), "dtype": gen.get("dtype"),
                   "decode_path": gen.get("decode_path"), "devices": gen.get("devices"), "split": gen.get("split"),
                   "compute_architecture": gen.get("compute_architecture"), "cuda_visible_devices": cvd,
                   "cuda_device_order": order, "kernel_path": kernels.get("kernel_path"),
                   "chimera_commit": (exp.get("chimera_tree") or {}).get("commit"),
                   "checkpoint_chimera_commit": (exp.get("checkpoint") or {}).get("chimera_commit"),
                   "model_directory": mdir, "model_resolution": how,
                   "directory_digest_oc_calib": (d.get("provenance") or {}).get("directory_sha256"),
                   "tokenizer_directory": gen.get("tokenizer_directory"), "dossier_sha256": d.get("dossier_sha256"),
                   "lane_digest": d.get("lane_digest"), "label": label, "stage": d.get("stage"),
                   "load_wall_time_seconds": exp.get("load_wall_time_seconds"),
                   "peak_cuda_memory_allocated_bytes": exp.get("peak_cuda_memory_allocated_bytes"),
                   "finite_guard": guard_path if guard is not None else None}
    gen_common = {"decoding": gen.get("decoding"), "temperature": 0, "eos_token_id": gen.get("eos_token_id"),
                  "extra_stop_token_ids": gen.get("extra_stop_token_ids"),
                  "ledger_band_logit_mask": gen.get("ledger_band_logit_mask"),
                  "ledger_band_begin": gen.get("ledger_band_begin"), "dtype": gen.get("dtype"),
                  "split": gen.get("split"), "mode": d.get("mode"), "micro_batch": gen.get("micro_batch"),
                  "add_generation_prompt": gen.get("add_generation_prompt")}
    common_kw = dict(project="chimera-omni", model=model, tool_version_id=tool_v, env_id=env_id, receipt=prov,
                     started=started, ended=ended, wall_seconds=d.get("wall_time_seconds"), label=label,
                     tokenizer_sha256=gen.get("tokenizer_directory_sha256"),
                     thinking_requested=thinking.get("requested"), thinking_took_effect=thinking.get("took_effect"),
                     run_dir=str(Path(path).parent))
    key = f"{label}:{prov[1][:12]}"
    finite_by_case: dict[str, bool] = {}
    for seg in (guard or {}).get("segments") or []:
        cid = seg.get("case_id")
        if cid:
            finite_by_case[cid] = finite_by_case.get(cid, True) and bool(seg.get("finite"))

    def place_devices(run_id: int) -> None:
        K.clear_devices(conn, run_id)
        spans = K.layer_spans(place.get("block_device"))
        peaks = {p.get("device"): p for p in (guard or {}).get("cuda_peaks") or []}
        mapped = K.visible_gpus(conn, cvd, order)
        for gid, detail in mapped:
            dev = f"cuda:{detail['cuda']}"
            if gid is None:
                batch.skip(f"{label}: no gpu row for visible device {detail['visible']}")
                continue
            peak = peaks.get(dev, {}).get("max_memory_allocated_bytes")
            db.add_run_device(conn, run_id, gid, role=f"split-{detail['cuda']}" if len(mapped) > 1 else "compute",
                              kernel_path=kernel_label(kernels), kernels=kernels, layers=spans.get(dev),
                              mem_mib_peak=int(peak / 1048576) if isinstance(peak, (int, float)) else None,
                              detail={**detail, "device": dev, "head_device": place.get("head_device") == dev})

    # --- bank_v1 (child_responses) -------------------------------------------------------------
    bank = d.get("bank") or {}
    bank_path = bank.get("path")
    bank_abs = bank_path if bank_path and os.path.isabs(bank_path) else f"{C.OC}/{bank_path}"
    if bank_path and os.path.isfile(bank_abs) and (d.get("child_responses") or []):
        suite, _ids = K.ensure_bank_v1(conn, batch, bank_abs)
        members = K.suite_members(conn, suite)
        if db.file_sha256(bank_abs) != bank.get("file_sha256"):
            batch.skip(f"{label}: bank file sha differs from the dossier's; cases matched by prompt content")
        rid = K.eval_run(conn, batch, run_key=f"eval:dossier:{key}", name=f"oc-dossier {label} bank_v1",
                         suite=suite, engine="chimera", gen={"cap": gen.get("maximum_new_tokens"), **gen_common},
                         config={**base_config, "suite": "bank_v1"}, bank_file_sha256=bank.get("file_sha256"),
                         **common_kw)
        place_devices(rid)
        for r in d.get("child_responses") or []:
            batch.saw()
            pk = K.resolve_case(conn, batch, members, r["id"], r.get("prompt_messages"), category=r.get("category"))
            term = K.termination_of(r.get("termination"), r.get("runtime_error"))
            text = r.get("response_verbatim")
            wall = r.get("wall_time_seconds")
            raw = {k: r.get(k) for k in ("prompt_tokens", "response_sha256", "generated_token_ids_sha256",
                                         "logits_to_keep_used", "runtime_error", "termination", "teacher_forced",
                                         "wall_time_seconds_is") if r.get(k) is not None}
            db.upsert_eval_result(conn, rid, pk, output_text=text, output_tokens=r.get("generated_tokens"),
                                  termination=term,
                                  form="teacher-forced-proxy" if r.get("termination") == "teacher-forced-proxy" else None,
                                  artifact_flags=[] if term == "other" else K.machine_flags(
                                      text, r.get("generated_tokens"), r.get("loop_detected"), term),
                                  finite=finite_by_case.get(r["id"]),
                                  latency_ms=int(wall * 1000) if isinstance(wall, (int, float)) else None,
                                  raw=raw, prov=prov, batch_id=batch.id)
            batch.wrote()
        metrics, used = review_metrics(path, prov[1])
        for rp in used:
            batch.prov(rp)
        for name, (value, denom) in metrics.items():
            db.put_metric(conn, rid, name, value, denominator=denom,
                          extra={"source": used, "is_upper_bound": True, "note": "machine upper bound, not a reading"})
        pre = d.get("decode_path_preflight") or {}
        if isinstance(pre.get("agree"), bool):
            db.put_metric(conn, rid, "decode_path_agree", 1.0 if pre["agree"] else 0.0, extra={"steps": pre.get("steps")})

    # --- smoke (decode_smoke.records) ----------------------------------------------------------
    smoke = d.get("decode_smoke") or {}
    spath = smoke.get("prompts_file")
    spath_abs = spath if spath and os.path.isabs(spath) else f"{C.OC}/{spath}"
    if smoke.get("records") and spath and os.path.isfile(spath_abs):
        suite, _ids = K.ensure_raw_prompt_suite(conn, batch, spath_abs, "smoke", "smoke", chat=True,
                                                description="3-prompt decode smoke, chat-formatted, 128-token cap "
                                                            "(matrix campaign stage 6).")
        members = K.suite_members(conn, suite)
        rid = K.eval_run(conn, batch, run_key=f"eval:dossier-smoke:{key}", name=f"oc-dossier {label} smoke",
                         suite=suite, engine="chimera", gen={"cap": smoke.get("maximum_new_tokens"), **gen_common},
                         config={**base_config, "suite": "smoke", "smoke_verdict": smoke.get("verdict")},
                         bank_file_sha256=smoke.get("prompts_file_sha256"), **common_kw)
        place_devices(rid)
        for r in smoke["records"]:
            batch.saw()
            pk = K.resolve_case(conn, batch, members, r["id"], r.get("prompt_messages"), category=r.get("category"))
            term = K.termination_of(r.get("termination"), r.get("runtime_error"))
            wall = r.get("wall_time_seconds")
            db.upsert_eval_result(conn, rid, pk, output_text=r.get("response_verbatim"),
                                  output_tokens=r.get("generated_tokens"), termination=term,
                                  artifact_flags=K.machine_flags(r.get("response_verbatim"), r.get("generated_tokens"),
                                                                 r.get("loop_detected"), term),
                                  latency_ms=int(wall * 1000) if isinstance(wall, (int, float)) else None,
                                  raw={k: r.get(k) for k in ("prompt_tokens", "response_sha256", "terminated",
                                                             "termination", "runtime_error") if r.get(k) is not None},
                                  prov=prov, batch_id=batch.id)
            batch.wrote()

    # --- probe (emit_zero_floor.probes; no text) -----------------------------------------------
    ezf = d.get("emit_zero_floor") or {}
    probe = d.get("probe_prompts") or {}
    ppath = probe.get("prompts_file")
    ppath_abs = ppath if ppath and os.path.isabs(ppath) else f"{C.OC}/{ppath}"
    if ezf.get("probes") and ppath and os.path.isfile(ppath_abs):
        suite, ids = K.ensure_raw_prompt_suite(conn, batch, ppath_abs, "probe", "probe", chat=False,
                                               description="8-prompt free-run probe, raw continuation, 64 greedy "
                                                           "tokens (matrix campaign stage 5).")
        rid = K.eval_run(conn, batch, run_key=f"eval:dossier-probe:{key}", name=f"oc-dossier {label} emit-zero-floor",
                         suite=suite, engine="chimera", gen={"cap": 0, **gen_common},
                         config={**base_config, "suite": "probe", "measure": "emit-zero-floor (terminal state + "
                                                                              "logit stats per probe, no decode)"},
                         bank_file_sha256=probe.get("prompts_file_sha256"), **common_kw)
        place_devices(rid)
        for p in ezf["probes"]:
            batch.saw()
            pk = ids.get(f"probe-{p.get('probe_index')}")
            if pk is None:
                batch.skip(f"{label}: probe index {p.get('probe_index')} not in the probe bank")
                continue
            db.upsert_eval_result(conn, rid, pk, finite=bool(p.get("logit_finite")) and bool(p.get("terminal_finite")),
                                  raw=p, prov=prov, batch_id=batch.id)
            batch.wrote()
        probes_n = ezf.get("probe_count")
        for name in ("min_logit_std", "min_terminal_l2", "distinct_argmax_tokens"):
            if isinstance(ezf.get(name), (int, float)):
                denom = float(probes_n) if name == "distinct_argmax_tokens" and isinstance(probes_n, int) else None
                db.put_metric(conn, rid, f"emit_zero_floor/{name}", float(ezf[name]), denominator=denom)
        if isinstance(ezf.get("passed"), bool):
            db.put_metric(conn, rid, "emit_zero_floor/passed", 1.0 if ezf["passed"] else 0.0)


def run(conn, batch: C.Batch, args) -> None:
    K.ensure_gpus(conn)
    reviews = ledger_reviews_by_dossier()
    paths = dossier_paths()
    docs = {p: C.load_json(p) for p in paths}
    by_dir = {}
    for p, d in docs.items():
        if isinstance(d, dict) and d.get("dossier_sha256") in reviews:
            mdir = ((d.get("provenance") or {}).get("model_directory") or "").rstrip("/")
            by_dir.setdefault(mdir, reviews[d["dossier_sha256"]])
    for p in paths:
        batch.saw()
        ingest_dossier(conn, batch, p, reviews, by_dir)
