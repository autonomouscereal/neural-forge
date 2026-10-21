"""ingest.p11_reads -- P11 long-context needle reads (p11_needle_engine read) -> eval runs on suite p11_needles.

Sources: $OC/distill/tps/**/p11_*_summary.json (schema omni-campaign.p11.read.v1/summary).  Today: the
v1.0.0 read (distill/tps/p11_omni1_summary.json, 2026-10-13) and the v1.1.x re-read with content/digit/strict
readouts (distill/tps/p11-v110/p11_omni1_v110_summary.json, 2026-10-16), both on OMNI1 at 32k/131k/262k.

Cases are the summary row ids ('L32768-d25', '...-control'); the prompt is the generator spec (engine,
needles file + sha256, length, depth) because the haystack is generated, not stored.  Teacher-forced: no
output text.  Grader p11_needle_engine.passkey_exact_teacher_forced on needle rows (controls stay ungraded);
mean_nll_digits, first_digit_rank, readouts and tok_s go to eval_result.raw / tok_s.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

from . import common as C
from . import evalkit as K
from .common import db

SOURCE = "p11_reads"
SCHEMA = "omni-campaign.p11.read.v1/summary"


def summary_files() -> list[str]:
    return sorted(set(glob.glob(f"{C.OC}/distill/tps/p11*_summary.json") +
                      glob.glob(f"{C.OC}/distill/tps/*/p11*_summary.json")))


def arg(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv or []):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def run(conn, batch: C.Batch, args) -> None:
    K.ensure_gpus(conn)
    grader = K.grader_id(conn, "p11_needle_engine.passkey_exact_teacher_forced")
    for path in summary_files():
        batch.saw()
        s = C.load_json(path)
        if not isinstance(s, dict) or s.get("schema") != SCHEMA or not s.get("rows"):
            batch.skip(f"{path}: not a P11 read summary")
            continue
        prov = batch.prov(path)
        ident = s.get("identity") or {}
        argv = s.get("argv") or []
        needles_sha = s.get("needles_sha256") or ""
        pks = []
        rows = []
        for r in s["rows"]:
            m = re.match(r"^L(\d+)-d(\d+)(-control)?$", r["id"])
            prompt = {"generator": "p11_needle_engine read", "needles": s.get("needles"), "needles_sha256": needles_sha,
                      "length": r.get("length"), "depth_pct": int(m.group(2)) if m else None,
                      "needle": bool(r.get("needle"))}
            pk = db.ensure_case(conn, r["id"], prompt, category="long-context-needle", lane=str(r.get("length")),
                                tags=["p11_needles", "teacher-forced"] + ([] if r.get("needle") else ["control"]),
                                reference={"expect_hit": bool(r.get("needle"))}, prov=prov, batch_id=batch.id)
            pks.append(pk)
            rows.append((r, pk))
        suite = db.ensure_suite(conn, "p11_needles", f"v1-{needles_sha[:12]}", status="sealed",
                                default_grader="p11_needle_engine.passkey_exact_teacher_forced",
                                bank_sha256=needles_sha,
                                description="P11 long-context passkey needles (teacher-forced read) at the lengths "
                                            "the run asked for; controls carry no needle",
                                prov=prov, batch_id=batch.id)
        # a later read may cover more lengths: membership is the union of every read, ordered by case id
        union = sorted(set(pks) | set(K.suite_members(conn, suite).values()))
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM test_case WHERE id = ANY(%s) ORDER BY case_id, id", (union,))
            K.set_membership(conn, suite, [row[0] for row in cur.fetchall()])
        model = K.ensure_model_at(conn, ident.get("model") or arg(argv, "--model") or "", role="composite",
                                  layout="chimera.checkpoint.v1", stage="OMNI1", prov=prov, batch_id=batch.id)
        tool_v = C.tool_version_for(conn, ident.get("tool") or "p11_needle_engine", ident.get("tool_version"),
                                    ident.get("tool_sha256"), path=argv[0] if argv else None)
        place = s.get("placement") or {}
        tree = ident.get("chimera_tree") or {}
        env_id = K.env_for(conn, code_commit=tree.get("commit"),
                           env_vars={"CUDA_VISIBLE_DEVICES": place.get("CUDA_VISIBLE_DEVICES"),
                                     "CUDA_DEVICE_ORDER": place.get("CUDA_DEVICE_ORDER")})
        label = arg(argv, "--label") or Path(path).stem
        kda = arg(argv, "--kda-kernel")
        rid = K.eval_run(conn, batch, run_key=f"eval:p11:{label}:{prov[1][:12]}", name=f"P11 needle read {label}",
                         project="chimera-omni", model=model, suite=suite, engine="chimera",
                         gen={"mode": "teacher-forced", "dtype": arg(argv, "--dtype"), "chunk": arg(argv, "--chunk"),
                              "lengths": arg(argv, "--lengths"), "attn_budget_mib": arg(argv, "--attn-budget-mib")},
                         config={"dtype": arg(argv, "--dtype"), "kda_kernel": kda, "argv": argv,
                                 "exit": s.get("exit"), "identity": ident, "needles": s.get("needles")},
                         ended=C.parse_ts(s.get("written_utc")), tool_version_id=tool_v, env_id=env_id, receipt=prov,
                         run_dir=str(Path(path).parent), label=label)
        K.clear_devices(conn, rid)
        for gid, detail in K.visible_gpus(conn, place.get("CUDA_VISIBLE_DEVICES"), place.get("CUDA_DEVICE_ORDER")):
            if gid is not None:
                db.add_run_device(conn, rid, gid, role="compute", kernel_path=f"kda-{kda}" if kda else None,
                                  kernels={"kda_kernel": kda}, detail={**detail, "inventory": place.get("inventory_at_start")})
        for r, pk in rows:
            batch.saw()
            res = db.upsert_eval_result(conn, rid, pk, tok_s=r.get("tok_s"),
                                        raw={k: v for k, v in r.items() if k not in ("id", "length", "tok_s")},
                                        prov=prov, batch_id=batch.id)
            if r.get("needle") and isinstance(r.get("passkey_exact_teacher_forced"), bool):
                ok = r["passkey_exact_teacher_forced"]
                db.put_grade(conn, res, grader, "pass" if ok else "fail", score=1.0 if ok else 0.0,
                             why=f"first_digit_rank {r.get('first_digit_rank')}, mean_nll_digits "
                                 f"{r.get('mean_nll_digits'):.3f}" if isinstance(r.get("mean_nll_digits"), float) else None,
                             detail={"readouts": r.get("readouts")}, prov=prov, batch_id=batch.id)
            batch.wrote()
