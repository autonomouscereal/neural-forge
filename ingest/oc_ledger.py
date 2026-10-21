"""ingest.oc_ledger -- $OC/ledger.jsonl + $OC/registry/ -> model, ledger_event, model_lineage, run (+metric).

The ledger stays the checkpoint authority; this mirrors it.  Per checkpoint id: one model row
(model_key = checkpoint_id), one ledger_event per line (record = the registry JSON when the matching
registry file exists, else the ledger line), lineage from parent_checkpoint_id, and one run per sealed
record:
  register  -> run kind fold | distill | build (clone)   'ledger:<id>:<seq>:register'
  review    -> run kind review, metrics column_a, column_b, canonical_signal,
               sealed_useful_or_partial, parent_useful_or_partial (step 0)
OMNI1's composite_manifest.json names its 7 lineage sources; each becomes a 'source' lineage edge
(Qwen3.8-27B-Uncensored -> PARENT-Q38-27BU; the others -> 'source:<name>' rows, layout 'external').
"""
from __future__ import annotations

import json
import os
import re

from . import common as C
from .common import db

SOURCE = "oc_ledger"

FAMILY = {"Q35-4B": "Qwen3.5-4B", "Q35-0.8B": "Qwen3.5-0.8B", "Q38-27BU": "Qwen3.8-27B-U"}
# readable stage labels for the OC02 chain (the ledger 'stage' column is only parent|fold|clone|distill)
STAGE_BY_SUFFIX = [
    (re.compile(r"-C2K4352$"), "S3"),
    (re.compile(r"-C2K4352K3U$"), "S5"),
    (re.compile(r"-OMNI1$"), "OMNI1"),
    (re.compile(r"-OMNI1\.clone-pre-D1$"), "OMNI1.clone-pre-D1"),
    (re.compile(r"-OMNI1D1$"), "D1"),
    (re.compile(r"-OMNI1D1\.clone-pre-D2$"), "D1.clone-pre-D2"),
]
SOURCE_NAME_TO_CHECKPOINT = {"Qwen3.8-27B-Uncensored": "PARENT-Q38-27BU"}
SOURCE_FAMILY = {"Kimi K3": "Kimi-K3", "ACE-Step": "ACE-Step", "MiniMax H3": "MiniMax-H3", "Qwen-Image": "Qwen-Image",
                 "Hunyuan3D 2.1": "Hunyuan3D-2.1", "Qwen3-VL-32B": "Qwen3-VL-32B"}
SOURCE_PROJECT = {"Kimi K3": "k3"}


def ledger_path() -> str:
    return C.OC + "/ledger.jsonl"


def registry_file(cid: str, seq: int, event: str) -> str:
    return f"{C.OC}/registry/{cid}/{cid}-{seq:04d}-{event}.json"


def project_for(cid: str) -> str:
    return "neural-forge" if cid.startswith("OC01-") or cid in ("PARENT-Q35-4B", "PARENT-Q35-0.8B") else "chimera-omni"


def stage_label(cid: str, ledger_stage: str | None) -> str | None:
    if ledger_stage == "parent":
        return "parent"
    for rx, label in STAGE_BY_SUFFIX:
        if rx.search(cid):
            return label
    if cid.startswith("OC01-F-"):
        return "OC01-" + cid.rsplit("-", 1)[-1]
    return ledger_stage


def role_for(cid: str, rec: dict) -> str:
    stage = rec.get("stage")
    directory = rec.get("directory") or ""
    if stage == "parent":
        return "parent"
    if stage == "clone":
        return "clone"
    if stage == "distill":
        return "delta" if "/deltas/" in directory else "student"
    if cid.endswith("-OMNI1"):
        return "composite"
    return "fold-child"


def layout_of(directory: str | None) -> str:
    """'chimera.checkpoint.v1' | 'hf' | 'gguf' | 'safetensors-raw' | 'unknown' from what is on disk."""
    if not directory or not os.path.isdir(directory):
        return "unknown"
    cfg = C.load_json(os.path.join(directory, "config.json"), 8 << 20)
    if isinstance(cfg, dict):
        if "chimera_commit" in cfg or isinstance(cfg.get("config"), dict) and "block_pattern" in cfg["config"]:
            return "chimera.checkpoint.v1"
        if cfg.get("architectures") or cfg.get("model_type"):
            return "hf"
    try:
        names = os.listdir(directory)
    except OSError:
        return "unknown"
    if any(n.endswith(".gguf") for n in names):
        return "gguf"
    return "safetensors-raw" if any(n.endswith(".safetensors") for n in names) else "unknown"


def run(conn, batch: C.Batch, args) -> None:
    path = ledger_path()
    lprov = batch.prov(path)
    host_id = C.host_id(conn)
    records = []
    for n, line in enumerate(C.read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise db.LabError(f"{path}:{n}: not JSON ({exc})") from None

    # pass 1: one model row per checkpoint, from its newest record (the ledger is append-only)
    newest: dict[str, dict] = {}
    first_sealed: dict[str, str] = {}
    verdict: dict[str, str] = {}
    for rec in records:
        cid = rec["checkpoint_id"]
        newest[cid] = rec
        first_sealed.setdefault(cid, rec.get("sealed_at"))
        if rec.get("event") == "review" and rec.get("sealed_verdict"):
            verdict[cid] = rec["sealed_verdict"]
    model_ids: dict[str, int] = {}
    for cid, rec in newest.items():
        batch.saw()
        directory = rec.get("directory")
        present = bool(directory) and os.path.isdir(directory)
        register = next((r for r in records if r["checkpoint_id"] == cid and r.get("event") in ("register", "clone")), rec)
        config = {"ledger_status": rec.get("status"), "ledger_status_reason": rec.get("status_reason"),
                  "ledger_stage": rec.get("stage"), "parent_code": rec.get("parent_code"),
                  "parameter_count_source": rec.get("parameter_count_source"),
                  "parent_parameter_count": rec.get("parent_parameter_count"), "quantization": rec.get("quantization"),
                  "recipe_sha256": rec.get("recipe_sha256"), "net_ratio": rec.get("net_ratio"),
                  "tag": rec.get("tag"), "wave": rec.get("wave")}
        model_ids[cid] = db.upsert_model(
            conn, cid, project_for(cid), role_for(cid, rec), layout_of(directory),
            family=FAMILY.get(rec.get("parent_code") or ""), stage=stage_label(cid, rec.get("stage")),
            path=directory, host_id=host_id, ledger_checkpoint_id=cid,
            directory_sha256=rec.get("directory_sha256"), parameter_count=rec.get("parameter_count"),
            dtype=None, config={k: v for k, v in config.items() if v is not None},
            sealed_verdict=verdict.get(cid), status="present" if present else "missing",
            created_at=C.parse_ts(first_sealed.get(cid)), notes=register.get("notes"), prov=lprov, batch_id=batch.id)
        batch.wrote()

    # pass 2: lineage from parent_checkpoint_id
    for cid, rec in newest.items():
        parent = rec.get("parent_checkpoint_id")
        if not parent:
            continue
        if parent not in model_ids:
            batch.skip(f"{cid}: parent {parent} not in ledger")
            continue
        relation = {"clone": "clone", "distill": "delta-of" if role_for(cid, rec) == "delta" else "distill"}.get(
            rec.get("stage"), "fold")
        db.link_lineage(conn, model_ids[cid], model_ids[parent], relation,
                        detail={"from": "ledger parent_checkpoint_id", "clone_source_id": rec.get("clone_source_id")},
                        prov=lprov, batch_id=batch.id)
        batch.wrote()

    # pass 3: ledger_event + one run per record
    for rec in records:
        cid, seq, event = rec["checkpoint_id"], int(rec["record_seq"]), rec.get("event") or "unknown"
        batch.saw()
        reg = registry_file(cid, seq, event)
        if os.path.isfile(reg):
            full = C.load_json(reg)
            eprov = batch.prov(reg)
        else:
            full, eprov = None, lprov
        record = full if isinstance(full, dict) else rec
        with conn.cursor() as cur:
            cur.execute("INSERT INTO ledger_event (model_id, checkpoint_id, record_seq, event, stage, sealed_at, "
                        "sealed_verdict, column_a, column_b, net_ratio, record, src_path, src_sha256, src_host, "
                        "ingest_batch_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
                        "ON CONFLICT (checkpoint_id, record_seq) DO UPDATE SET model_id = EXCLUDED.model_id, "
                        "event = EXCLUDED.event, stage = EXCLUDED.stage, sealed_at = EXCLUDED.sealed_at, "
                        "sealed_verdict = EXCLUDED.sealed_verdict, column_a = EXCLUDED.column_a, "
                        "column_b = EXCLUDED.column_b, net_ratio = EXCLUDED.net_ratio, record = EXCLUDED.record, "
                        "src_path = EXCLUDED.src_path, src_sha256 = EXCLUDED.src_sha256, "
                        "ingest_batch_id = EXCLUDED.ingest_batch_id",
                        (model_ids[cid], cid, seq, event, rec.get("stage"), C.parse_ts(rec.get("sealed_at")),
                         rec.get("sealed_verdict"), rec.get("column_a"), rec.get("column_b"), rec.get("net_ratio"),
                         db.canonical_json(record), *eprov, batch.id))
        batch.wrote()
        kind = "review" if event == "review" else {"fold": "fold", "distill": "distill", "clone": "build",
                                                    "parent": "build"}.get(rec.get("stage") or "", "other")
        sealed = C.parse_ts(rec.get("sealed_at"))
        config = {k: rec.get(k) for k in ("recipe_sha256", "receipt_sha", "verdict_receipt_sha", "dossier_sha256",
                                          "package_path", "package_sha256", "quantization", "tag", "wave",
                                          "parameter_count", "parent_parameter_count") if rec.get(k) is not None}
        run_id = db.upsert_run(conn, f"ledger:{cid}:{seq:04d}:{event}", project_for(cid), kind,
                               f"{event} {cid}", "pass", model_id_=model_ids[cid], host_id=host_id,
                               started_at=sealed, ended_at=sealed, config=config, receipt_path=eprov[0],
                               receipt_sha256=eprov[1], notes=(rec.get("notes") or "")[:4000] or None,
                               prov=eprov, batch_id=batch.id)
        parent = rec.get("parent_checkpoint_id")
        if parent in model_ids:
            db.link_run_model(conn, run_id, model_ids[parent], "base")
        if event == "review":
            for name in ("column_a", "column_b", "canonical_signal", "sealed_useful_or_partial",
                         "parent_useful_or_partial", "net_ratio"):
                v = rec.get(name)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    db.put_metric(conn, run_id, name, float(v), ts=sealed,
                                  extra={"verdict": rec.get("sealed_verdict")} if name == "column_a" else None)
        batch.wrote()

    # pass 4: composite lineage sources (OMNI1 composite_manifest.json)
    for cid, rec in newest.items():
        directory = rec.get("directory")
        manifest = os.path.join(directory, "composite_manifest.json") if directory else None
        if not manifest or not os.path.isfile(manifest):
            continue
        doc = C.load_json(manifest)
        names = ((doc or {}).get("sources_all") or {}).get("names") or []
        if not names:
            continue
        mprov = batch.prov(manifest)
        for ordinal, name in enumerate(names):
            batch.saw()
            src_cid = SOURCE_NAME_TO_CHECKPOINT.get(name)
            if src_cid and src_cid in model_ids:
                src_id = model_ids[src_cid]
            else:
                slug = re.sub(r"[^A-Za-z0-9.]+", "-", name).strip("-")
                src_id = db.upsert_model(conn, f"source:{slug}", SOURCE_PROJECT.get(name, "media" if name != "Kimi K3" else "k3"),
                                         "parent", "external", family=SOURCE_FAMILY.get(name, name), stage="source",
                                         status="unknown", notes=f"lineage source named by {cid} composite_manifest.json; "
                                                                 "weights not registered in the ledger",
                                         prov=mprov, batch_id=batch.id)
            db.link_lineage(conn, model_ids[cid], src_id, "source", ordinal=ordinal,
                            detail={"from": "composite_manifest.sources_all.names"}, prov=mprov, batch_id=batch.id)
            batch.wrote()
