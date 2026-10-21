"""ingest.evalkit -- shared pieces of the eval ingesters (suites, cases, models under test, eval runs, grades).

Used by banks, codex_tasksets, dossiers, student_tests, toolcall_bankruns, s8_media and p11_reads.  Raw
psycopg2 through omni_lab_db; every write is a natural-key upsert, so a re-run rewrites the same rows.

Conventions the eval ingesters share
  * A case is global, keyed (case_id, content_sha256 of prompt+tools+media).  A run's result is attached
    to the case whose recorded prompt hashes the same; a recorded prompt that differs from the bank
    becomes its own case row (counted as 'prompt-mismatch' on the batch), never silently re-pointed.
  * A suite is (name, version); ensure_bank_suite() rebuilds membership from the bank file, so any
    ingester that needs a suite can create it from the same file the run used.
  * Model under test: the ledger checkpoint whose directory is the run's model directory, else an
    existing model row with that path, else a new 'path:<sha16>' row (common.sha16_path_key, the same
    key model_scan uses), so ingest order does not matter.
  * Grades: machine graders write what the source file recorded; reader grades ('human:claude') come
    only from grade files a reader wrote.  An empty / not-run cell is the absence of a row.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

from . import common as C
from .common import db

DEFAULT_OC = "/home/cereal/omni-campaign"
DEFAULT_LX = "/mnt/lexar/omni-campaign"


def at(path: str | None) -> str | None:
    """A path recorded inside a source file, read under OMNI_LAB_OC / OMNI_LAB_LEXAR when those point
    somewhere else (fixture trees in tests; a moved campaign root) and the file exists there.  With the
    default roots, or when no re-rooted copy exists, the path is returned as recorded."""
    if not path:
        return path
    for default, root in ((DEFAULT_OC, C.OC), (DEFAULT_LX, C.LX)):
        if root != default and (path == default or path.startswith(default + "/")):
            moved = root + path[len(default):]
            if os.path.exists(moved):
                return moved
    return path


# ---------------------------------------------------------------------------
# graders
# ---------------------------------------------------------------------------

GRADERS: dict[str, tuple[str, str, str]] = {
    "human:claude": ("human", "correct/partial/miss/none",
                     "Claude as reader, artifact-tolerant: loops/echo/cap-rides/missing EOS/empty count nothing; "
                     "only content misses count (student-tests-20261021 grades.json)"),
    "human:cyril": ("human", "correct/partial/miss/none", "Cyril's own reading"),
    "toolcall_selfgen.grade_case": ("machine", "pass/fail",
                                    "omni_toolcall_bank_v1 grading: first assistant turn, every expected call matched "
                                    "by name and argument (distill/tools/toolcall_selfgen.py grade_case)"),
    "student_test.code_check": ("machine", "0-1", "student_test.py run_code_check: fraction of CODE_ASSERTS passed"),
    "p11_needle_engine.passkey_exact_teacher_forced": (
        "machine", "pass/fail", "P11 needle read: passkey exact under teacher forcing (needle rows only; controls "
                                "carry no grade)"),
    "s8_media_decode.beats_noisy_input": (
        "machine", "pass/fail", "S8 media decode: x0_omni_final beats the noisy input on the decode's own metric "
                                "(owner policy 2026-10-06: a diagnostic, not a gate)"),
    "chimera_recovery.evaluate": ("machine", "pass/fail",
                                  "Codex recovery-eval oracle grader (vendored chimera_recovery/evaluation.py, "
                                  "sha 6a84626d...); no results ingested yet"),
}

# reader grades as written in grades.json -> (grade, score); 'no content' is an artifact, it scores nothing
READER_GRADE = {"correct": ("correct", 1.0), "partially correct": ("partial", 0.5), "partial": ("partial", 0.5),
                "content miss": ("miss", 0.0), "miss": ("miss", 0.0), "no content": ("none", None)}

_GRADER_CACHE: dict[int, dict[str, int]] = {}


def grader_id(conn, name: str) -> int:
    cache = _GRADER_CACHE.setdefault(id(conn), {})
    if name not in cache:
        kind, scale, desc = GRADERS[name]
        cache[name] = db.ensure_grader(conn, name, kind, scale, desc)
    return cache[name]


# ---------------------------------------------------------------------------
# termination / artifact flags (only from fields the source recorded)
# ---------------------------------------------------------------------------

TERMINATION = {"complete": "eos", "eos": "eos", "stop": "stop", "censored-at-cap": "cap", "cap": "cap",
               "length": "cap", "error": "error", "timeout": "timeout", "teacher-forced-proxy": "other"}


def termination_of(recorded: str | None, runtime_error: Any = None) -> str | None:
    if runtime_error:
        return "error"
    if recorded is None:
        return None
    return TERMINATION.get(str(recorded), "other")


def machine_flags(response: str | None, generated_tokens: int | None, loop_detected: Any,
                  termination: str | None, form_box: str | None = None) -> list[str]:
    """Artifact flags derived from recorded fields only: loop_detected, an empty response, a cap ride,
    and oc_calib.classify_form's box when the source carries one."""
    flags = []
    if loop_detected is True:
        flags.append("loop")
    if (response is not None and response.strip() == "") or generated_tokens == 0:
        flags.append("empty")
    if termination == "cap":
        flags.append("cap-ride")
    if form_box and form_box.upper() not in ("CONTENT",):
        flags.append(form_box.lower())
    return sorted(set(flags))


# ---------------------------------------------------------------------------
# suites / cases
# ---------------------------------------------------------------------------

def bank_rows(path: str | Path) -> tuple[dict, list[dict]]:
    """(top-level JSON, list of case dicts) for the bank layouts in $OC/calibration/banks."""
    doc = C.load_json(path)
    if not isinstance(doc, dict):
        raise db.LabError(f"{path}: not a JSON bank")
    rows = doc.get("prompts") if isinstance(doc.get("prompts"), list) else doc.get("cases")
    if not isinstance(rows, list):
        raise db.LabError(f"{path}: no 'prompts' or 'cases' list")
    return doc, rows


def suite_members(conn, suite_pk: int) -> dict[tuple[str, str], int]:
    """{(case_id, content_sha256): case_pk} for one suite."""
    with conn.cursor() as cur:
        cur.execute("SELECT tc.case_id, tc.content_sha256, tc.id FROM suite_member sm JOIN test_case tc "
                    "ON tc.id = sm.case_pk WHERE sm.suite_id = %s", (suite_pk,))
        return {(r[0], r[1]): r[2] for r in cur.fetchall()}


def set_membership(conn, suite_pk: int, case_pks: list[int]) -> None:
    """Make the suite's members exactly case_pks, in this order (sealed suites never change in practice;
    a re-run of the same bank file is a no-op)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM suite_member WHERE suite_id = %s AND NOT (case_pk = ANY(%s))", (suite_pk, case_pks))
        # move existing ordinals out of the way so UNIQUE (suite_id, ordinal) cannot collide mid-update
        cur.execute("UPDATE suite_member SET ordinal = -1 - ordinal WHERE suite_id = %s AND ordinal >= 0", (suite_pk,))
    for i, pk in enumerate(case_pks):
        db.add_suite_member(conn, suite_pk, pk, i)


def ensure_bank_v1(conn, batch: C.Batch, path: str) -> tuple[int, dict[str, int]]:
    """The frozen 16-case whole-response bank (lfm2_semantic_whole_response_prompts_v1.json) -> suite
    bank_v1@1.  Human-graded by design (no machine grader).  Returns (suite_id, {case_id: case_pk})."""
    prov = batch.prov(path)
    doc, rows = bank_rows(path)
    suite = db.ensure_suite(conn, "bank_v1", "1", status="sealed", default_grader=None, bank_sha256=prov[1],
                            description="frozen whole-response review bank (16 cases incl. the charged-not-scored "
                                        "safety lane); graded by a reader, artifact-tolerant. "
                                        f"schema {doc.get('schema')}",
                            prov=prov, batch_id=batch.id)
    ids = {}
    pks = []
    for row in rows:
        pk = db.ensure_case(conn, row["id"], row["messages"], reference={"context": row.get("context")},
                            category=row.get("category"), tags=["bank_v1"], prov=prov, batch_id=batch.id)
        ids[row["id"]] = pk
        pks.append(pk)
        batch.wrote()
    set_membership(conn, suite, pks)
    return suite, ids


def ensure_raw_prompt_suite(conn, batch: C.Batch, path: str, name: str, id_prefix: str, chat: bool,
                            description: str) -> tuple[int, dict[str, int]]:
    """matrix_campaign_{probe,smoke}_prompts_v1.json: a list of strings.  Case ids are '<prefix>-N' (1-based,
    the ids the dossiers use).  Smoke prompts are chat turns; probe prompts are raw continuations."""
    prov = batch.prov(path)
    doc, rows = bank_rows(path)
    suite = db.ensure_suite(conn, name, "1", status="sealed", bank_sha256=prov[1],
                            description=f"{description} prompt_set_sha256 {doc.get('prompt_set_sha256')}",
                            prov=prov, batch_id=batch.id)
    ids, pks = {}, []
    for n, text in enumerate(rows, 1):
        prompt = [{"role": "user", "content": text}] if chat else {"raw_text": text}
        cid = f"{id_prefix}-{n}"
        pk = db.ensure_case(conn, cid, prompt, category=doc.get("kind"),
                            tags=[name] + ([] if chat else ["raw-continuation"]), prov=prov, batch_id=batch.id)
        ids[cid] = pk
        pks.append(pk)
        batch.wrote()
    set_membership(conn, suite, pks)
    return suite, ids


def ensure_toolcall_v1(conn, batch: C.Batch, path: str) -> tuple[int, dict[str, int]]:
    prov = batch.prov(path)
    doc, rows = bank_rows(path)
    suite = db.ensure_suite(conn, "toolcall_v1", "1", status="sealed", default_grader="toolcall_selfgen.grade_case",
                            bank_sha256=prov[1],
                            description=f"omni_toolcall_bank_v1: 185 cases over 8 lanes; render: {doc.get('render')}; "
                                        f"heldout functions {len(doc.get('heldout_functions') or [])}",
                            prov=prov, batch_id=batch.id)
    grader_id(conn, "toolcall_selfgen.grade_case")
    ids, pks = {}, []
    for row in rows:
        pk = db.ensure_case(conn, row["id"], row["messages"], tools=row.get("tools"),
                            reference={"expected": row.get("expected")}, category="toolcall", lane=row.get("lane"),
                            tags=["toolcall_v1", row.get("lane") or ""], grader="toolcall_selfgen.grade_case",
                            prov=prov, batch_id=batch.id)
        ids[row["id"]] = pk
        pks.append(pk)
        batch.wrote()
    set_membership(conn, suite, pks)
    return suite, ids


def resolve_case(conn, batch: C.Batch, members: dict[tuple[str, str], int], case_id: str, prompt: Any,
                 tools: Any = None, media: Any = None, **case_kw) -> int:
    """The suite member whose content matches the recorded prompt; otherwise a separate case row."""
    sha = db.case_content_sha256(prompt, tools, media)
    pk = members.get((case_id, sha))
    if pk is not None:
        return pk
    batch.skip(f"prompt-mismatch {case_id}: recorded prompt differs from the bank row; kept as its own case")
    return db.ensure_case(conn, case_id, prompt, tools=tools, media=media, **case_kw)


# ---------------------------------------------------------------------------
# models under test
# ---------------------------------------------------------------------------

def ledger_by_directory() -> dict[str, dict]:
    """{directory: newest ledger record} from $OC/ledger.jsonl (read-only; the ledger is the authority)."""
    out: dict[str, dict] = {}
    for line in C.read_text(C.OC + "/ledger.jsonl").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("directory"):
            out[rec["directory"].rstrip("/")] = rec
    return out


def ledger_record(checkpoint_id: str) -> dict | None:
    found = None
    for line in C.read_text(C.OC + "/ledger.jsonl").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("checkpoint_id") == checkpoint_id:
            found = rec
    return found


def ensure_ledger_model(conn, checkpoint_id: str) -> int | None:
    """Model row for a ledger checkpoint id; when the oc_ledger ingester has not run yet, a minimal row
    with the same key and oc_ledger's own role/stage/layout rules (it refreshes the row when it runs)."""
    mid = db.model_id(conn, checkpoint_id)
    if mid is not None:
        return mid
    rec = ledger_record(checkpoint_id)
    if rec is None:
        return None
    from . import oc_ledger as L   # same rules as the ledger mirror, so the two writers agree
    directory = rec.get("directory")
    return db.upsert_model(conn, checkpoint_id, L.project_for(checkpoint_id), L.role_for(checkpoint_id, rec),
                           L.layout_of(directory), family=L.FAMILY.get(rec.get("parent_code") or ""),
                           stage=L.stage_label(checkpoint_id, rec.get("stage")), path=directory,
                           host_id=C.host_id(conn), ledger_checkpoint_id=checkpoint_id,
                           directory_sha256=rec.get("directory_sha256"), parameter_count=rec.get("parameter_count"),
                           status="present" if directory and os.path.isdir(directory) else "missing",
                           notes="minimal row written by an eval ingester; oc_ledger refreshes it")


def ensure_model_at(conn, path: str, *, role: str, layout: str, project: str = "chimera-omni",
                    family: str | None = None, stage: str | None = None, config: dict | None = None,
                    notes: str | None = None, status: str | None = None, lineage: list[tuple[str, str, dict]] = (),
                    prov: tuple[str, str, str] | None = None, batch_id: int | None = None) -> int:
    """Model row for the artifact at `path` on the AI server: ledger checkpoint first, then any row already
    at that path, else a new 'path:<sha16>' row.  `lineage` = [(parent ledger id, relation, detail)]."""
    p = path.rstrip("/")
    rec = ledger_by_directory().get(p)
    if rec is not None:
        mid = ensure_ledger_model(conn, rec["checkpoint_id"])
        if mid is not None:
            return mid
    mid = C.model_id_by_path(conn, p)
    if mid is None:
        if status is None:
            status = "present" if os.path.exists(p) else "missing"
        mid = db.upsert_model(conn, C.sha16_path_key(p), project, role, layout, family=family, stage=stage, path=p,
                              host_id=C.host_id(conn), config=config or {}, status=status, notes=notes,
                              prov=prov, batch_id=batch_id)
    for parent_cid, relation, detail in lineage:
        parent = ensure_ledger_model(conn, parent_cid)
        if parent is not None and parent != mid:
            db.link_lineage(conn, mid, parent, relation, detail=detail, prov=prov, batch_id=batch_id)
    return mid


# ---------------------------------------------------------------------------
# devices / env
# ---------------------------------------------------------------------------

def seed_host(conn, name: str = C.SERVER_HOST) -> None:
    """Prime common.host_id's cache with a plain SELECT when the host row exists, so an eval ingester never
    takes the host row lock (ensure_host's upsert) that a concurrent long ingest may be holding."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM host WHERE name = %s", (name,))
        row = cur.fetchone()
    if row:
        C._HOST_CACHE.setdefault(id(conn), {})[name] = row[0]


def ensure_gpus(conn) -> None:
    """The AI server's cards (by UUID) when the hardware ingester has not run yet; same parser and upsert."""
    seed_host(conn)
    hid = C.host_id(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM gpu WHERE host_id = %s", (hid,))
        if cur.fetchone()[0]:
            return
    if db.this_host() != C.SERVER_HOST or not shutil.which("nvidia-smi"):
        return
    from . import hardware as H
    out = subprocess.run(["nvidia-smi", f"--query-gpu={H.QUERY}", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60, check=True).stdout
    for card in H.parse_nvidia_smi(out):
        db.upsert_gpu(conn, hid, card["index"], card["uuid"], card["name"], card["sm"], card["vram_mib"])


def visible_gpus(conn, cuda_visible_devices: str | None, device_order: str | None) -> list[tuple[int | None, dict]]:
    """[(gpu.id or None, mapping detail)] in cuda:N order for a recorded CUDA_VISIBLE_DEVICES.
    Indexes map through the CURRENT nvidia-smi index (valid under CUDA_DEVICE_ORDER=PCI_BUS_ID, which every
    campaign launcher sets); UUIDs map exactly."""
    out = []
    if not cuda_visible_devices:
        return out
    for n, token in enumerate(t.strip() for t in str(cuda_visible_devices).split(",")):
        if not token:
            continue
        if token.startswith("GPU-"):
            out.append((C.gpu_for(conn, uuid=token), {"cuda": n, "visible": token, "mapping": "uuid"}))
        elif token.isdigit():
            exact = device_order == "PCI_BUS_ID"
            out.append((C.gpu_for(conn, index=int(token)),
                        {"cuda": n, "visible": token,
                         "mapping": "index-via-current-nvidia-smi" + ("" if exact else " (device order not PCI_BUS_ID: uncertain)")}))
    return out


def clear_devices(conn, run_id: int) -> None:
    """Drop a run's placement rows before re-adding them, so a re-ingest reflects the source exactly."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM run_device WHERE run_id = %s", (run_id,))


def layer_spans(block_device: list | None) -> dict[str, tuple[int, int]]:
    """{'cuda:N': (first, last+1)} from a per-block device list, when each device holds one contiguous span."""
    spans: dict[str, list[int]] = {}
    for i, dev in enumerate(block_device or []):
        spans.setdefault(str(dev), []).append(i)
    out = {}
    for dev, idx in spans.items():
        if idx == list(range(idx[0], idx[-1] + 1)):
            out[dev] = (idx[0], idx[-1] + 1)
    return out


def env_for(conn, *, torch: str | None = None, transformers: str | None = None, python: str | None = None,
            fla: str | None = None, code_commit: str | None = None, llama_cpp_build: str | None = None,
            env_vars: dict | None = None, packages: dict | None = None) -> int:
    return db.ensure_env(conn, python=python, torch=torch, transformers=transformers, fla=fla,
                         llama_cpp_build=llama_cpp_build, code_commit=code_commit,
                         env_vars={k: v for k, v in (env_vars or {}).items() if v is not None},
                         packages=packages or {})


# ---------------------------------------------------------------------------
# eval runs / results
# ---------------------------------------------------------------------------

def started_from(ended_iso: str | None, wall_seconds: float | None):
    end = C.parse_ts(ended_iso)
    if end is None or not isinstance(wall_seconds, (int, float)):
        return None, end
    return end - timedelta(seconds=float(wall_seconds)), end


def eval_run(conn, batch: C.Batch, *, run_key: str, name: str, project: str, model: int, suite: int, engine: str,
             gen: dict, config: dict, status: str = "pass", started=None, ended=None, wall_seconds=None,
             tool_version_id=None, env_id=None, receipt: tuple[str, str, str] | None = None, run_dir=None,
             owner=None, notes=None, label=None, tokenizer_sha256=None, chat_template_sha256=None,
             bank_file_sha256=None, thinking_requested=None, thinking_took_effect=None, case_filter=None) -> int:
    """One run(kind='eval') + its eval_run row.  `receipt` = provenance of the file the run is read from."""
    rid = db.upsert_run(conn, run_key, project, "eval", name, status, model_id_=model, tool_version_id=tool_version_id,
                        host_id=C.host_id(conn), env_id=env_id, started_at=started, ended_at=ended,
                        wall_seconds=wall_seconds, config={"engine": engine, **config},
                        receipt_path=receipt[0] if receipt else None, receipt_sha256=receipt[1] if receipt else None,
                        run_dir=run_dir, owner=owner, notes=notes, prov=receipt, batch_id=batch.id)
    db.upsert_eval_run(conn, rid, model, suite, engine, gen=gen, tokenizer_sha256=tokenizer_sha256,
                       chat_template_sha256=chat_template_sha256, bank_file_sha256=bank_file_sha256,
                       thinking_requested=thinking_requested, thinking_took_effect=thinking_took_effect,
                       case_filter=case_filter, label=label)
    batch.wrote()
    return rid


def run_id_by_receipt_sha(conn, sha: str, run_key_prefix: str) -> int | None:
    """The eval run ingested from the file with this sha256 (joins a later grade file to an earlier run)."""
    with conn.cursor() as cur:
        cur.execute("SELECT r.id FROM run r JOIN eval_run er ON er.run_id = r.id "
                    "WHERE r.receipt_sha256 = %s AND r.run_key LIKE %s ORDER BY r.id LIMIT 1", (sha, run_key_prefix + "%"))
        row = cur.fetchone()
        return row[0] if row else None


def result_id(conn, eval_run_id: int, case_id: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT res.id FROM eval_result res JOIN test_case tc ON tc.id = res.case_pk "
                    "WHERE res.eval_run_id = %s AND tc.case_id = %s ORDER BY res.id LIMIT 1", (eval_run_id, case_id))
        row = cur.fetchone()
        return row[0] if row else None


def put_reader_grade(conn, batch: C.Batch, eval_result_id: int, grader: str, entry: dict,
                     prov: tuple[str, str, str]) -> bool:
    """A reader's {grade, why, artifact_flags} -> eval_grade.  'not run' writes nothing (blank cell)."""
    text = str(entry.get("grade") or "").strip().lower()
    if text not in READER_GRADE:
        batch.skip(f"reader grade {text!r} has no mapping; not written")
        return False
    grade, score = READER_GRADE[text]
    db.put_grade(conn, eval_result_id, grader_id(conn, grader), grade, score=score, why=entry.get("why"),
                 detail={"grade_as_written": entry.get("grade"), "artifact_flags": entry.get("artifact_flags") or []},
                 prov=prov, batch_id=batch.id)
    batch.wrote()
    return True
