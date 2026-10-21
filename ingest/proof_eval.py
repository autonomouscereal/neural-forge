"""ingest.proof_eval -- a behaviour-graded proof's own eval file -> suite, eval run, results, grades, artifacts.

For evals that do not go through the GPU worker (a proof script that trains and grades its own model, e.g.
chimera_omni.proof.v2_proxy_mixed): the proof writes one `omni-lab-proof-eval-v1` JSON per evaluated model and

    omni_lab.py log-eval --file eval.json [--file eval2.json]

loads each in one transaction. `compare` then pivots these runs like any runner run.

Per file:
  * grader rows the file declares (machine graders of the proof), test_case rows (global, keyed by
    (case_id, content sha)), and the suite.  A suite that already exists must have exactly the file's cases:
    suites are immutable, so a changed case list is refused (bump suite.version instead).
  * the evaluated model (upsert by model_key; NULL fields never erase), with an optional lineage parent.
  * run (kind eval unless the file says otherwise, parent_run_id = the training run) + env + eval_run.
  * eval_result + eval_grade per case, eval_artifact per decoded image/audio/video/mesh output.
  * metric rows (step 0 summaries, with denominators) on the eval run.
  * optional experiment row (kind arch/...) keyed by experiment.key, linked to this eval run with a role.
Re-loading the same file rewrites the same keys (run_key, (eval_run, case), (result, grader)).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import common as C
from .common import db

SOURCE = "proof_eval"
SCHEMA = "omni-lab-proof-eval-v1"
TERMINATIONS = {"eos", "cap", "stop", "loop", "nan", "error", "timeout", "unsupported-modality", "other"}
ARTIFACT_KINDS = {"image", "audio", "video", "mesh", "latent", "text", "other"}


def _ts(value):
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _require(doc: dict, path: str) -> None:
    if doc.get("schema") != SCHEMA:
        raise db.LabError(f"{path}: schema {doc.get('schema')!r} is not {SCHEMA}")
    for key in ("suite", "cases", "model", "run", "results"):
        if key not in doc:
            raise db.LabError(f"{path}: missing {key!r}")
    ids = [c["id"] for c in doc["cases"]]
    if len(ids) != len(set(ids)):
        raise db.LabError(f"{path}: duplicate case ids")
    unknown = {r["case_id"] for r in doc["results"]} - set(ids)
    if unknown:
        raise db.LabError(f"{path}: results for cases the file does not define: {sorted(unknown)[:5]}")


def _cases(conn, batch: C.Batch, doc: dict, prov) -> dict[str, int]:
    pks: dict[str, int] = {}
    for case in doc["cases"]:
        reference = {k: case[k] for k in ("expected", "rubric") if case.get(k) is not None}
        pks[case["id"]] = db.ensure_case(
            conn, case["id"], case["prompt"], media=case.get("media"), reference=reference or None,
            category=case.get("category"), lane=case.get("lane"), tags=case.get("tags"),
            modality_in=case.get("modality_in"), modality_out=case.get("modality_out"), grader=case.get("grader"),
            prov=prov, batch_id=batch.id)
    return pks


def _suite(conn, batch: C.Batch, doc: dict, pks: dict[str, int], prov) -> int:
    spec = doc["suite"]
    ordered = [pks[c["id"]] for c in doc["cases"]]
    sid = db.suite_id(conn, spec["name"], str(spec["version"]))
    if sid is not None:
        have = {row[0] for row in db.suite_cases(conn, sid)}
        if have != set(ordered):
            raise db.LabError(f"suite {spec['name']}@{spec['version']} exists with other cases; suites are immutable "
                              "(give the file a new suite version)")
        return sid
    sid = db.ensure_suite(conn, spec["name"], str(spec["version"]), status="sealed",
                          modality_in=spec.get("modality_in") or sorted({m for c in doc["cases"]
                                                                        for m in c.get("modality_in") or ["text"]}),
                          modality_out=spec.get("modality_out") or sorted({m for c in doc["cases"]
                                                                          for m in c.get("modality_out") or ["text"]}),
                          default_grader=spec.get("default_grader"), bank_sha256=prov[1],
                          description=spec.get("description"), prov=prov, batch_id=batch.id)
    for ordinal, pk in enumerate(ordered):
        db.add_suite_member(conn, sid, pk, ordinal)
    return sid


def _model(conn, batch: C.Batch, doc: dict, hid: int, prov) -> int:
    m = doc["model"]
    mid = db.upsert_model(conn, m["model_key"], m.get("project") or "chimera-omni", m.get("role") or "other",
                          m.get("layout") or "unknown", family=m.get("family"), stage=m.get("stage"),
                          path=m.get("path"), host_id=hid, parameter_count=m.get("parameter_count"),
                          size_bytes=m.get("size_bytes"), dtype=m.get("dtype"), config=m.get("config"),
                          notes=m.get("notes"), prov=prov, batch_id=batch.id)
    if m.get("parent"):
        parent = db.model_id(conn, m["parent"])
        if parent is None:
            raise db.LabError(f"lineage parent {m['parent']!r} is not a model row")
        db.link_lineage(conn, mid, parent, m.get("relation") or "source", prov=prov, batch_id=batch.id)
    return mid


def _experiment(conn, batch: C.Batch, spec: dict, run_id: int, prov) -> None:
    technique = None
    if spec.get("technique"):
        with conn.cursor() as cur:
            cur.execute("INSERT INTO technique (name, family, description) VALUES (%s, %s, %s) "
                        "ON CONFLICT (name) DO UPDATE SET family = COALESCE(EXCLUDED.family, technique.family) "
                        "RETURNING id", (spec["technique"], spec.get("family"), spec.get("technique_description")))
            technique = cur.fetchone()[0]
    parent = db.model_id(conn, spec["parent"]) if spec.get("parent") else None
    child = db.model_id(conn, spec["child"]) if spec.get("child") else None
    src_path, src_sha, src_host = prov
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO experiment (project, key, kind, technique_id, parent_id, child_id, run_id, params, budget, "
            "verdict, evidence_level, results_path, date, detail, src_path, src_sha256, src_host, ingest_batch_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
            "ON CONFLICT (key) DO UPDATE SET technique_id = COALESCE(EXCLUDED.technique_id, experiment.technique_id), "
            "parent_id = COALESCE(EXCLUDED.parent_id, experiment.parent_id), "
            "child_id = COALESCE(EXCLUDED.child_id, experiment.child_id), "
            "run_id = COALESCE(EXCLUDED.run_id, experiment.run_id), params = experiment.params || EXCLUDED.params, "
            "budget = experiment.budget || EXCLUDED.budget, verdict = COALESCE(EXCLUDED.verdict, experiment.verdict), "
            "results_path = COALESCE(EXCLUDED.results_path, experiment.results_path), "
            "detail = experiment.detail || EXCLUDED.detail RETURNING id",
            (spec.get("project") or "chimera-omni", spec["key"], spec.get("kind") or "arch", technique, parent, child,
             spec.get("run_id"), db._j(spec.get("params") or {}), db._j(spec.get("budget") or {}), spec.get("verdict"),
             spec.get("evidence_level") or "receipt", spec.get("results_path"), spec.get("date"),
             db._j(spec.get("detail") or {}), src_path, src_sha, src_host, batch.id))
        exp_id = cur.fetchone()[0]
        cur.execute("INSERT INTO experiment_eval (experiment_id, eval_run_id, role) VALUES (%s, %s, %s) "
                    "ON CONFLICT (experiment_id, eval_run_id) DO UPDATE SET role = EXCLUDED.role",
                    (exp_id, run_id, spec.get("role") or "other"))


def load_file(conn, batch: C.Batch, path: str) -> int:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(doc, path)
    host = doc["run"].get("host") or db.this_host()
    prov = batch.prov(path, host)
    hid = db.ensure_host(conn, host) if host != C.SERVER_HOST else C.host_id(conn)
    for name, g in (doc.get("graders") or {}).items():
        db.ensure_grader(conn, name, g.get("kind") or "machine", g.get("scale") or "pass/fail", g.get("description"))
    pks = _cases(conn, batch, doc, prov)
    sid = _suite(conn, batch, doc, pks, prov)
    mid = _model(conn, batch, doc, hid, prov)

    run = doc["run"]
    env = doc.get("env") or {}
    env_id = db.ensure_env(conn, python=env.get("python"), torch=env.get("torch"), cuda=env.get("cuda"),
                           code_commit=env.get("code_commit"), packages=env.get("packages"))
    receipt = run.get("receipt")
    receipt_sha = db.file_sha256(receipt) if receipt and Path(receipt).is_file() else run.get("receipt_sha256")
    rid = db.upsert_run(conn, run["run_key"], run.get("project") or "chimera-omni", run.get("kind") or "eval",
                        run["name"], run.get("status") or "pass", model_id_=mid, host_id=hid, env_id=env_id,
                        started_at=_ts(run.get("started_utc")), ended_at=_ts(run.get("ended_utc")),
                        wall_seconds=run.get("wall_seconds"), exit_code=run.get("exit_code"),
                        config=run.get("config") or {}, receipt_path=run.get("receipt_path") or receipt,
                        receipt_sha256=receipt_sha, log_path=run.get("log_path"), run_dir=run.get("run_dir"),
                        parent_run_id=run.get("parent_run_id"), owner=run.get("owner") or "claude",
                        notes=run.get("notes"), prov=prov, batch_id=batch.id)
    db.upsert_eval_run(conn, rid, mid, sid, run.get("engine") or "proof", gen=run.get("gen") or {},
                       bank_file_sha256=prov[1], label=run.get("label"),
                       baseline_run_id=run.get("baseline_run_id"))

    graders = {}
    for res in doc["results"]:
        batch.saw()
        term = res.get("termination")
        if term is not None and term not in TERMINATIONS:
            raise db.LabError(f"{path}: case {res['case_id']}: termination {term!r} is not in the lab vocabulary")
        eid = db.upsert_eval_result(conn, rid, pks[res["case_id"]], output_text=res.get("output_text"),
                                    output_tokens=res.get("output_tokens"), termination=term,
                                    artifact_flags=res.get("artifact_flags"), finite=res.get("finite"),
                                    latency_ms=res.get("latency_ms"), raw=res.get("raw"), prov=prov,
                                    batch_id=batch.id)
        batch.wrote()
        grade = res.get("grade")
        if grade:
            name = grade["grader"]
            if name not in graders:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM grader WHERE name = %s", (name,))
                    row = cur.fetchone()
                if row is None:
                    raise db.LabError(f"{path}: grader {name!r} is neither declared in the file nor in the database")
                graders[name] = row[0]
            db.put_grade(conn, eid, graders[name], grade["grade"], score=grade.get("score"), why=grade.get("why"),
                         detail=grade.get("detail"), prov=prov, batch_id=batch.id)
            batch.wrote()
        for art in res.get("artifacts") or []:
            if art.get("kind") not in ARTIFACT_KINDS:
                raise db.LabError(f"{path}: artifact kind {art.get('kind')!r}")
            db.add_eval_artifact(conn, eid, art["kind"], art["path"], sha256=art.get("sha256"), nbytes=art.get("bytes"),
                                 duration_s=art.get("duration_s"), width=art.get("width"), height=art.get("height"),
                                 sample_rate=art.get("sample_rate"), metrics=art.get("metrics"))
            batch.wrote()
    for item in doc.get("metrics") or []:
        db.put_metric(conn, rid, item["name"], float(item["value"]), denominator=item.get("denominator"))
    if doc.get("experiment"):
        _experiment(conn, batch, doc["experiment"], rid, prov)
    return rid


def run(conn, batch: C.Batch, args) -> None:
    ids = [load_file(conn, batch, path) for path in args.file]
    batch.tally_lines = [f"eval run ids: {', '.join(str(i) for i in ids)}"]
