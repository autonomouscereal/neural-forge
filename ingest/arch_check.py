"""ingest.arch_check -- chimera_omni.conformance reports -> model_tensor_group + arch_check (+ a probe run).

The report is produced on the model side (it needs torch for configs, or only safetensors headers for
checkpoints):

    python -m chimera_omni.conformance --profile co2-proxy --dynamic --out report.json
    python -m chimera_omni.conformance --checkpoint <chimera checkpoint dir> --out report.json

and loaded here without torch:

    omni_lab.py arch-check --report report.json [--model MODEL_KEY]

Per report, in one transaction:
  * the model: --model names an existing row (a checkpoint already in `model`); otherwise a checkpoint
    report is matched by path, and a profile/config report gets its own design-stage row
    `arch:<profile>@<config sha12>` (layout chimera.config.v1, no weights, path NULL).
  * model_tensor_group: the census replaces the model's previous census exactly (stale prefixes go).
  * arch_check: one row per requirement in arch_requirement.  A requirement the report does not cover
    is written as 'unknown' so the gap shows in v_arch_conformance instead of a blank.
  * run (kind probe) with the report as receipt, the parameter count and the probe numbers as metrics.
A report naming a requirement id that is not in arch_requirement is refused (seed it first).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import common as C
from . import knowledge_common as K
from .common import db

SOURCE = "arch_check"
REPORT_SCHEMA = "chimera_omni.conformance.v1"
PROJECT = "chimera-omni"


def _requirements(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM arch_requirement ORDER BY length(id), id")
        return [row[0] for row in cur.fetchall()]


def _model_for(conn, report: dict, model_key: str | None, prov, batch: C.Batch) -> tuple[int, str]:
    source = report.get("source", {})
    if model_key:
        mid = db.model_id(conn, model_key)
        if mid is None:
            raise db.LabError(f"--model {model_key!r} is not a model row")
        return mid, model_key
    if source.get("kind") == "checkpoint":
        mid = C.model_id_by_path(conn, source.get("path"))
        if mid is None:
            raise db.LabError(f"no model row has path {source.get('path')!r}; pass --model")
        with conn.cursor() as cur:
            cur.execute("SELECT model_key FROM model WHERE id = %s", (mid,))
            return mid, cur.fetchone()[0]
    cfg = report["config"]
    meta = cfg.get("metadata", {})
    profile = source.get("profile") or meta.get("profile") or "config"
    key = f"arch:{profile}@{report['config_sha256'][:12]}"
    mid = db.upsert_model(
        conn, key, PROJECT, "other", "chimera.config.v1",
        family=meta.get("architecture") or "chimera-omni", stage=profile, path=None,
        parameter_count=int(report["parameter_count"]), dtype=None,
        config={"config_sha256": report["config_sha256"], "manifest": cfg, "source": source},
        status="present", notes="design-stage config: census from a meta-device build of the config; no weights exist",
        prov=prov, batch_id=batch.id)
    return mid, key


def load_report(conn, batch: C.Batch, tally: K.Tally, path: Path, model_key: str | None) -> dict:
    prov = batch.prov(path, host=db.this_host())
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA:
        raise db.LabError(f"{path}: expected schema {REPORT_SCHEMA}, got {report.get('schema')!r}")
    known = _requirements(conn)
    if not known:
        raise db.LabError("arch_requirement is empty: run `ingest arch_requirements` first")
    covered = {c["requirement_id"]: c for c in report["checks"]}
    unknown_ids = sorted(set(covered) - set(known))
    if unknown_ids:
        raise db.LabError(f"{path}: requirement ids not in arch_requirement: {unknown_ids}")
    report.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    mid, key = _model_for(conn, report, model_key, prov, batch)

    keep: list[tuple[str, int]] = []
    for group in report["census"]:
        batch.saw()
        values = {"model_id": mid, "prefix": group["prefix"], "layer_lo": int(group["layer_lo"]),
                  "layer_hi": int(group["layer_hi"]), "n_tensors": int(group["n_tensors"]),
                  "params": int(group["params"]), "dtype": group.get("dtype") or None,
                  "modality_scope": group["modality_scope"], "source_component": group.get("source_component"),
                  "detail": group.get("detail") or {}, "src_path": prov[0], "src_sha256": prov[1],
                  "src_host": prov[2], "ingest_batch_id": batch.id}
        _id, state = K.upsert(conn, "model_tensor_group", ("model_id", "prefix", "layer_lo"), values,
                              jsonb_cols=("detail",), id_col="model_id",
                              ignore_in_compare=("ingest_batch_id", "src_path", "src_sha256", "src_host"))
        tally.add("model_tensor_group", state)
        keep.append((group["prefix"], int(group["layer_lo"])))
    with conn.cursor() as cur:
        cur.execute("DELETE FROM model_tensor_group g WHERE g.model_id = %s AND NOT EXISTS "
                    "(SELECT 1 FROM unnest(%s::text[], %s::int[]) AS k(prefix, layer_lo) "
                    " WHERE k.prefix = g.prefix AND k.layer_lo = g.layer_lo)",
                    (mid, [p for p, _ in keep], [lo for _, lo in keep]))
        if cur.rowcount:
            batch.wrote(cur.rowcount)
            batch.notes.append(f"{key}: removed {cur.rowcount} stale census groups")

    for rid in known:
        batch.saw()
        check = covered.get(rid) or {"status": "unknown",
                                     "evidence": f"not evaluated by {report.get('tool', {}).get('module', 'the report')}"}
        _id, state = K.upsert(conn, "arch_check", ("model_id", "requirement_id"), {
            "model_id": mid, "requirement_id": rid, "status": check["status"], "evidence": check["evidence"],
            "checked_at": report.get("created_at"), "src_path": prov[0], "src_sha256": prov[1], "src_host": prov[2],
            "ingest_batch_id": batch.id}, id_col="model_id",
            ignore_in_compare=("ingest_batch_id", "checked_at", "src_path", "src_sha256", "src_host"))
        tally.add("arch_check", state)

    statuses = [covered.get(rid, {}).get("status", "unknown") for rid in known]
    dynamic = report.get("dynamic") or {}
    run_id = db.upsert_run(
        conn, f"probe:arch-conformance:{report['config_sha256'][:12]}:{prov[1][:12]}", PROJECT, "probe",
        f"arch conformance {key}", "fail" if "fail" in statuses else "pass", model_id_=mid,
        started_at=report.get("created_at"), ended_at=report.get("created_at"),
        config={"tool": report.get("tool"), "source": report.get("source"), "dynamic": bool(dynamic),
                "config_sha256": report["config_sha256"]},
        receipt_path=prov[0], receipt_sha256=prov[1], owner="claude",
        notes="checks: " + " ".join(f"{rid}={s}" for rid, s in zip(known, statuses)),
        prov=prov, batch_id=batch.id)
    db.link_run_model(conn, run_id, mid, "output")
    db.put_metric(conn, run_id, "parameter_count", float(report["parameter_count"]))
    for status in ("pass", "partial", "fail", "unknown"):
        db.put_metric(conn, run_id, f"requirements/{status}", float(statuses.count(status)), denominator=len(known))
    coverage = dynamic.get("gradient_coverage")
    if coverage:
        db.put_metric(conn, run_id, "probe/modality_specific_tensors", float(len(coverage["modality_specific"])),
                      denominator=coverage["parameters"])
    invariance = dynamic.get("type_id_invariance")
    if invariance:
        db.put_metric(conn, run_id, "probe/type_relabel_max_abs_diff", float(invariance["max_abs_diff"]))
    batch.wrote()
    return {"model_key": key, "model_id": mid, "run_id": run_id, "statuses": dict(zip(known, statuses))}


def run(conn, batch: C.Batch, args) -> None:
    reports = [Path(p) for p in (getattr(args, "report", None) or [])]
    if not reports:
        raise db.LabError("arch_check needs --report <conformance json> (one or more)")
    model_key = getattr(args, "model", None)
    if model_key and len(reports) != 1:
        raise db.LabError("--model applies to exactly one --report")
    tally = K.Tally(batch)
    batch.loaded = []
    for path in reports:
        batch.loaded.append(load_report(conn, batch, tally, path, model_key))
    K.finish(batch, tally)
    for item in batch.loaded:
        batch.tally_lines.append(f"{item['model_key']} (model {item['model_id']}, run {item['run_id']}): "
                                 + " ".join(f"{k}={v}" for k, v in item["statuses"].items()))


if __name__ == "__main__":
    K.main_guard(sys.modules[__name__])
