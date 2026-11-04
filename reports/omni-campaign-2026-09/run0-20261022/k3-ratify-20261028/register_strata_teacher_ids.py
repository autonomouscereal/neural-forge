#!/usr/bin/env python3
"""register_strata_teacher_ids 1.0.0: the two Strata teacher identities in omni_lab (lead call, 2026-10-28).

One transaction, raw parameterized SQL through omni_lab_db (psycopg2). Idempotent: upsert_model keys on
model_key, lineage and run links use ON CONFLICT. --dry-run does everything and rolls back.

  flash-next-strata-iq3s  jimmy31chen RVN Qwen3.8-Flash-Next abliterated IQ3_S (gguf sha cb5958e5...).
                          Strata 2026-10-21T18:21:56Z .. 2026-10-27T16:58:09Z; offline scorer from 2026-10-18.
  strata-q4kxl-abl-v2     huihui-ai Huihui-Qwen3.8-Flash-Next-abliterated-v2 unsloth UD-Q4_K_XL-V2 (4 shards,
                          SHA256SUMS sha 5e80bc35...). Strata since 2026-10-27T17:04:33Z (11:04:33 MDT).

The shard digests come from SHA256SUMS and are accepted only when sha256.check says OK for every shard
(par_range_dl.py's receipt; the 104 GiB are not re-hashed here). The IQ3_S digest comes from its full
re-hash receipt gguf.sha256.ok.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/cereal/omni-campaign/lab")
import omni_lab_db as db  # noqa: E402

VERSION = "1.0.0"
Q4_DIR = Path("/home/cereal/models/gguf/Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL-V2")
IQ3_GGUF = Path("/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata/RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf")
IQ3_SHA_OK = Path("/home/cereal/omni-campaign/distill/toolcall/flash-gen-2026-10-18/gguf.sha256.ok")
IQ3_SHA = "cb5958e5b0e97804fa2973f7ce36943a1b0d735b1b83fc1e52b8c3f3b97f87e6"
RULING = "strata-teacher-ids-2026-10-28"
JOURNAL = ("journalctl -u strata-llama: IQ3_S first 'strata_llama-1.0.0 start: gguf sha cb5958e5b0e97804' "
           "2026-10-21T18:21:56Z, last IQ3_S unit stop 2026-10-27T16:58:09Z; 'strata_llama-1.2.0 start: q4kxl "
           "SHA256SUMS 5e80bc356a7ba365' 2026-10-27T16:58:10Z, unit Started (healthy) 2026-10-27T17:04:33Z; "
           "xcache build 'strata_llama-1.3.0' (same Q4 weights) from 2026-10-27T21:51:32Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q4_shards() -> list[dict]:
    sums = {}
    for line in (Q4_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name] = digest
    ok = set(l.split(":")[0] for l in (Q4_DIR / "sha256.check").read_text().splitlines() if l.endswith(": OK"))
    if len(sums) != 4 or set(sums) != ok:
        raise SystemExit(f"refuse: SHA256SUMS {sorted(sums)} vs sha256.check OK {sorted(ok)}")
    out = []
    for name in sorted(sums):
        path = Q4_DIR / name
        out.append({"name": name, "sha256": sums[name], "bytes": path.stat().st_size})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if IQ3_SHA_OK.read_text().strip() != IQ3_SHA:
        raise SystemExit(f"refuse: {IQ3_SHA_OK} does not hold {IQ3_SHA}")
    shards = q4_shards()
    sums_sha = sha256_file(Q4_DIR / "SHA256SUMS")
    q4_bytes = sum(s["bytes"] for s in shards)
    me = Path(__file__).resolve()

    q4_cfg = {
        "teacher_id": "strata-q4kxl-abl-v2",
        "served_as": "strata (llama-server behind strata-gate, 127.0.0.1:18401 / gate :18400)",
        "served_from": "2026-10-27T17:04:33Z",
        "served_until": None,
        "served_note": "2026-10-27 11:04:33 MDT; builds: strata_llama 1.2.0 (-ncmoe 14) 16:58:10Z, then "
                       "1.3.0 llamacpp-xcache (same weights, expert VRAM cache) from 21:51:32Z",
        "hf_repo": "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF",
        "variant": "unsloth-UD-Q4_K_XL-V2",
        "quant": "UD-Q4_K_XL",
        "shards": shards,
        "sha256sums_file_sha256": sums_sha,
        "sha_receipts": [str(Q4_DIR / "SHA256SUMS"), str(Q4_DIR / "sha256.check")],
        "judge_identity": "strata-q4kxl-abl-v2+mmproj-f16-766911a6 (media_judge_spec 1.3.0)",
        "kd_products": ["run0-intake-staging/strata-kd-gen/v1 (top-64 probs, normalised logprobs, no lse)"],
        "not_the_same_as": "flash-next-strata-iq3s (different checkpoint: gguf_ident compare DIFFERENT; base, "
                           "imatrix and qwen4exp.attention.compress_ratios differ; chat template sha 1479547f vs 924d3fcc)",
        "kd_weight_note": "own teacher id; whether it shares run 0's 'strata' KD weight with IQ3_S caches is a "
                          "run-0 teacher-weight decision, not settled by this registration",
        "evidence": JOURNAL,
        "registered_by": f"register_strata_teacher_ids {VERSION}",
        "ruling": RULING,
    }
    iq3_cfg = {
        "teacher_id": "flash-next-strata-iq3s",
        "model_id_string": "jimmy31chen/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata (IQ3_S, qwen4exp A3B)",
        "served_as": "strata (llama-server behind strata-gate)",
        "served_from": "2026-10-21T18:21:56Z",
        "served_until": "2026-10-27T16:58:09Z",
        "served_note": "first strata_llama 1.0.0 start to the last IQ3_S unit stop; Strata was down 16:58:09Z-"
                       "17:04:33Z during the swap. Used earlier as an offline teacher (teacher-score qtext caches "
                       "teacher_cache/qtext/flash-next-strata-iq3s, flash-gen-2026-10-18) from 2026-10-18. It stays "
                       "on disk as the Strata rollback (STRATA_MODEL=iq3s).",
        "gguf": str(IQ3_GGUF),
        "gguf_sha256": IQ3_SHA,
        "gguf_bytes": IQ3_GGUF.stat().st_size,
        "sha_receipts": [str(IQ3_SHA_OK)],
        "known_mislabel": "gen_worker_text.py 1.4.0 still stamps this id on rows made by the Q4 after "
                          "2026-10-27T17:04:33Z; rows dated after that are strata-q4kxl-abl-v2",
        "evidence": JOURNAL,
        "registered_by": f"register_strata_teacher_ids {VERSION}",
        "ruling": RULING,
    }

    conn = db.connect("rw", app=f"register_strata_teacher_ids {VERSION}")
    try:
        prov = db.provenance(me)
        batch = db.start_batch(conn, "strata_teacher_ids", parser=str(me), parser_sha256=prov[1],
                               argv=sys.argv[1:])
        host_id = db.ensure_host(conn, "ai")
        q4_id = db.upsert_model(
            conn, "strata-q4kxl-abl-v2", "chimera-omni", "teacher", "gguf", family="Qwen3.8-Flash-Next",
            stage="strata", path=str(Q4_DIR), host_id=host_id, size_bytes=q4_bytes, dtype="UD-Q4_K_XL",
            config=q4_cfg, status="present", created_at="2026-10-27T16:57:00Z",
            notes="Strata teacher since 2026-10-27T17:04:33Z (11:04:33 MDT): huihui-ai Huihui-Qwen3.8-Flash-Next-"
                  "abliterated-v2 UD-Q4_K_XL, 4 shards, SHA256SUMS sha 5e80bc35. Lead registration 2026-10-28.",
            prov=prov, batch_id=batch)
        iq3_id = db.upsert_model(
            conn, "flash-next-strata-iq3s", "chimera-omni", "teacher", "gguf", family="Qwen3.8-Flash-Next",
            stage="strata", path=str(IQ3_GGUF), host_id=host_id, size_bytes=IQ3_GGUF.stat().st_size,
            dtype="IQ3_S", config=iq3_cfg, status="present", created_at="2026-10-18T00:00:00Z",
            notes="Strata teacher 2026-10-21T18:21:56Z .. 2026-10-27T16:58:09Z: RVN Qwen3.8-Flash-Next "
                  "abliterated IQ3_S (gguf sha cb5958e5). Superseded by strata-q4kxl-abl-v2. Lead registration 2026-10-28.",
            prov=prov, batch_id=batch)
        written = 2
        with conn.cursor() as cur:
            cur.execute("SELECT id, model_key FROM model WHERE path = %s AND id <> %s ORDER BY id", (str(IQ3_GGUF), iq3_id))
            artifacts = cur.fetchall()
        for art_id, art_key in artifacts:
            db.link_lineage(conn, iq3_id, art_id, "source", detail={"what": "teacher id -> scanned GGUF artifact row",
                                                                   "artifact_key": art_key}, prov=prov, batch_id=batch)
            written += 1
        links = []
        for key, mid in (("strata-q4kxl-abl-v2", q4_id), ("flash-next-strata-iq3s", iq3_id)):
            with conn.cursor() as cur:
                cur.execute("SELECT id, kind FROM run WHERE config::text LIKE %s ORDER BY id", (f"%{key}%",))
                for run_id, kind in cur.fetchall():
                    role = "judge" if kind in ("review", "build", "eval") and key == "strata-q4kxl-abl-v2" else "teacher"
                    db.link_run_model(conn, run_id, mid, role)
                    links.append((key, run_id, kind, role))
        written += len(links)
        db.end_batch(conn, batch, "ok", rows_seen=written, rows_written=written,
                     file_shas=[prov[1], sums_sha, IQ3_SHA])
        with conn.cursor() as cur:
            cur.execute("SELECT id, model_key, role, layout, family, stage, dtype, size_bytes, status, "
                        "config->>'served_from', config->>'served_until' FROM model WHERE id IN (%s, %s) ORDER BY id",
                        (iq3_id, q4_id))
            rows = cur.fetchall()
        report = {"batch": batch, "models": rows, "lineage_to_artifacts": artifacts,
                  "run_links": {k: sum(1 for l in links if l[0] == k) for k in ("strata-q4kxl-abl-v2", "flash-next-strata-iq3s")},
                  "run_link_sample": links[:8], "dry_run": args.dry_run}
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
        print(json.dumps(report, default=str, indent=1))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
