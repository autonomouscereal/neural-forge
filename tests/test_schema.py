#!/usr/bin/env python3
"""Schema + helper tests against a throwaway database.

    C:/Python314/python.exe -m unittest discover -s tests -v            (workstation)
    /home/cereal/.venvs/chimera-intake/bin/python -m unittest discover -s tests -v   (server, env file sourced)

DB tests create omni_lab_test_<hex> as the omni_lab role (CREATEDB), apply every migration, round-trip
rows through omni_lab_db helpers and the views, then DROP the database.  They never touch omni_lab
itself.  Pure-Python tests run without a database.  OMNI_LAB_TEST_KEEP=1 keeps the database for
inspection.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import omni_lab_db as db  # noqa: E402

TEST_DB_RE = re.compile(r"^omni_lab_test_[0-9a-f]{8}$")


def _have_rw() -> bool:
    try:
        db.db_params("rw")
        return True
    except db.LabError:
        return False


class PureTests(unittest.TestCase):
    def test_canonical_json_is_order_independent(self):
        self.assertEqual(db.sha256_json({"a": 1, "b": [1, 2]}), db.sha256_json({"b": [1, 2], "a": 1}))
        self.assertNotEqual(db.sha256_json({"a": 1}), db.sha256_json({"a": 2}))

    def test_case_identity_includes_tools_and_media(self):
        msgs = [{"role": "user", "content": "hi"}]
        base = db.case_content_sha256(msgs)
        self.assertEqual(base, db.case_content_sha256(msgs, None, None))
        self.assertNotEqual(base, db.case_content_sha256(msgs, [{"name": "f"}]))
        self.assertNotEqual(base, db.case_content_sha256(msgs, None, [{"kind": "image", "path": "x.png"}]))

    def test_scram_verifier_format_and_determinism(self):
        salt = bytes(range(16))
        v1 = db.scram_sha256_verifier("pencil-not-a-secret", salt=salt)
        v2 = db.scram_sha256_verifier("pencil-not-a-secret", salt=salt)
        self.assertEqual(v1, v2)
        self.assertRegex(v1, r"^SCRAM-SHA-256\$4096:[A-Za-z0-9+/=]{24}\$[A-Za-z0-9+/=]{44}:[A-Za-z0-9+/=]{44}$")
        self.assertNotEqual(v1, db.scram_sha256_verifier("pencil-not-a-secret"))  # random salt
        with self.assertRaises(db.LabError):
            db.scram_sha256_verifier("p\u00e4ss")

    def test_migration_files_contiguous(self):
        files = db.migration_files()
        self.assertEqual([v for v, *_ in files], list(range(1, len(files) + 1)))
        self.assertGreaterEqual(len(files), 5)

    def test_migration_files_refuse_gap_and_bad_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "0001_a.sql").write_text("SELECT 1;")
            Path(tmp, "0003_c.sql").write_text("SELECT 1;")
            with self.assertRaises(db.LabError):
                db.migration_files(Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "1_a.sql").write_text("SELECT 1;")
            with self.assertRaises(db.LabError):
                db.migration_files(Path(tmp))

    def test_password_command_takes_last_line(self):
        cmd = '{python} -c "print(\'banner\'); print(\'value-xyz\')"'
        self.assertEqual(db.read_password_command(cmd), "value-xyz")
        self.assertEqual(db.read_password_command('{python} -c "import sys; sys.exit(3)"'), "")
        self.assertEqual(db.read_password_command(""), "")


@unittest.skipUnless(_have_rw(), "no omni_lab credential resolvable (OMNI_LAB_PG_PASSWORD[_COMMAND|_FILE])")
class DbTests(unittest.TestCase):
    dbname: str

    @classmethod
    def setUpClass(cls):
        from psycopg2 import sql
        cls.dbname = f"omni_lab_test_{secrets.token_hex(4)}"
        assert TEST_DB_RE.match(cls.dbname)
        admin = db.connect("rw", app="test-create", autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(sql.Identifier(cls.dbname)))
        finally:
            admin.close()
        cls.conn = db.connect("rw", dbname=cls.dbname, app="test")
        cls.applied = db.migrate(cls.conn)

    @classmethod
    def tearDownClass(cls):
        from psycopg2 import sql
        cls.conn.close()
        if os.environ.get("OMNI_LAB_TEST_KEEP") == "1":
            print(f"\n[kept] {cls.dbname}")
            return
        assert TEST_DB_RE.match(cls.dbname), cls.dbname   # never drop anything else
        admin = db.connect("rw", app="test-drop", autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(cls.dbname)))
        finally:
            admin.close()

    def setUp(self):
        self.c = self.conn
        self.c.rollback()

    def tearDown(self):
        self.c.rollback()

    def q(self, query, params=None):
        with self.c.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall() if cur.description else None

    # -- migrations ------------------------------------------------------------------------------
    def test_00_migrations_applied_once(self):
        self.assertEqual([v for v, _ in self.applied], list(range(1, len(db.migration_files()) + 1)))
        self.assertEqual(db.migrate(self.c), [])
        self.assertEqual(db.schema_version(self.c), len(db.migration_files()))

    def test_01_migration_drift_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for _v, _n, path, _s in db.migration_files():
                shutil.copy(path, tmp)
            first = sorted(Path(tmp).glob("0001_*.sql"))[0]
            first.write_text(first.read_text() + "\n-- edited\n")
            with self.assertRaises(db.LabError):
                db.migrate(self.c, directory=Path(tmp))
        self.c.rollback()

    def test_02_expected_objects_exist(self):
        tables = {r[0] for r in self.q("SELECT table_name FROM information_schema.tables "
                                        "WHERE table_schema='public' AND table_type='BASE TABLE'")}
        for t in ("schema_migration", "ingest_batch", "project", "host", "gpu", "env", "tool", "tool_version",
                  "model", "model_lineage", "run", "run_device", "run_model", "metric", "run_event", "test_suite",
                  "test_case", "suite_member", "grader", "test", "eval_run", "eval_result", "eval_grade",
                  "eval_artifact", "technique", "experiment", "experiment_eval", "lane", "ledger_event",
                  "dataset", "dataset_version", "run_dataset", "directive", "ruling", "document",
                  "model_tensor_group", "arch_requirement", "arch_check"):
            self.assertIn(t, tables)
        views = {r[0] for r in self.q("SELECT table_name FROM information_schema.views WHERE table_schema='public'")}
        self.assertEqual(views, {"v_effective_grade", "v_eval_matrix", "v_results", "v_run_summary", "v_latest_models",
                                 "v_model_modality_coverage", "v_arch_conformance", "v_directives_by_topic"})

    # -- the full eval round trip ----------------------------------------------------------------
    def _fixture_eval(self, tag: str):
        c = self.c
        host = db.ensure_host(c, f"ai-{tag}", {"ip": "192.168.50.222"})
        g4 = db.upsert_gpu(c, host, 4, f"GPU-{tag}-4", "NVIDIA GeForce RTX 3090", "8.6", 24576)
        env_a = db.ensure_env(c, python="3.12.3", torch="2.11.0+cu128", transformers="5.12.1", fla="0.4",
                              env_vars={"CUDA_VISIBLE_DEVICES": "4"})
        tv = db.ensure_tool_version(c, "student_test", "1.0", "ab" * 32, path="/x/student_test.py", pinned=True)
        parent = db.upsert_model(c, f"PARENT-{tag}", "chimera-omni", "parent", "hf", family="Qwen3.8-27B-U", stage="S0")
        child = db.upsert_model(c, f"CHILD-{tag}", "chimera-omni", "student", "qwen3_5", stage="fullkd-run1",
                                config={"has_vision_config": True})
        db.link_lineage(c, child, parent, "distill")
        suite1 = db.ensure_suite(c, f"bank-{tag}", "1", default_grader="oc_calib.classify_form")
        cases = []
        for i in range(3):
            pk = db.ensure_case(c, f"{tag}-c{i}", [{"role": "user", "content": f"q{i}"}], category="reasoning",
                                reference={"expected": str(i)})
            db.add_suite_member(c, suite1, pk, i)
            cases.append(pk)
        machine = db.ensure_grader(c, "oc_calib.classify_form", "machine", "correct/partial/miss/none")
        human = db.ensure_grader(c, "human:claude", "human", "correct/partial/miss/none")
        test_id = db.upsert_test(c, f"bank-nothink-{tag}", f"bank-{tag}", "qwen35-student",
                                 gen={"cap": 512, "thinking": "off", "temperature": 0}, devices=[4])
        runs = {}
        for mkey, mid in (("parent", parent), ("child", child)):
            rid = db.upsert_run(c, f"eval:{tag}:{mkey}", "chimera-omni", "eval", f"{mkey} bank", "pass",
                                model_id_=mid, tool_version_id=tv, host_id=host, env_id=env_a,
                                started_at="2026-10-21T12:00:00Z", config={"engine": "qwen35-student", "dtype": "bf16"})
            db.add_run_device(c, rid, g4, kernel_path="fla-triton-sm86", kernels={"chunk_gated_delta_rule": "fla"},
                              layers=(0, 64), mem_mib_peak=20000)
            db.upsert_eval_run(c, rid, mid, suite1, "qwen35-student", gen={"cap": 512}, test_id=test_id,
                               thinking_requested="off", thinking_took_effect=True)
            runs[mkey] = rid
        # parent: correct, correct, partial (machine); child: machine says correct on c0 but human says miss
        grades = {"parent": ["correct", "correct", "partial"], "child": ["correct", "none", "miss"]}
        for mkey, rid in runs.items():
            for pk, g in zip(cases, grades[mkey]):
                res = db.upsert_eval_result(c, rid, pk, output_text=f"{mkey} answer", termination="eos",
                                            latency_ms=1200, output_tokens=40)
                db.put_grade(c, res, machine, g, score={"correct": 1, "partial": 0.5}.get(g, 0))
        res_child_c0 = self.q("SELECT id FROM eval_result WHERE eval_run_id=%s AND case_pk=%s", (runs["child"], cases[0]))[0][0]
        db.put_grade(c, res_child_c0, human, "miss", score=0, why="echoes the prompt")
        c.commit()
        return dict(host=host, g4=g4, env=env_a, parent=parent, child=child, suite1=suite1, cases=cases,
                    runs=runs, machine=machine, human=human, test=test_id)

    def test_10_eval_round_trip_and_views(self):
        f = self._fixture_eval("rt")
        rows = self.q("SELECT model_key, test, suite, engine, devices, correct, partial, miss, \"none\", ungraded, "
                      "n_results, n_cases, coverage, thinking_took_effect FROM v_results WHERE eval_run_id = ANY(%s) "
                      "ORDER BY model_key", ([f["runs"]["parent"], f["runs"]["child"]],))
        self.assertEqual(rows[0][:4], ("CHILD-rt", "bank-nothink-rt", "bank-rt@1", "qwen35-student"))
        self.assertEqual(rows[0][4], "4:NVIDIA GeForce RTX 3090/fla-triton-sm86")
        # child: c0 human 'miss' overrides machine 'correct' -> 0 correct, 0 partial, 2 miss, 1 none
        self.assertEqual(rows[0][5:13], (0, 0, 2, 1, 0, 3, 3, 1))
        self.assertEqual(rows[1][5:13], (2, 1, 0, 0, 0, 3, 3, 1))
        self.assertIs(rows[0][13], True)
        eff = self.q("SELECT grade, human_grade, machine_grade FROM v_eval_matrix WHERE eval_run_id=%s AND case_id='rt-c0'",
                     (f["runs"]["child"],))
        self.assertEqual(eff, [("miss", "miss", "correct")])
        summ = self.q("SELECT devices, dtype, engine, n_results, config_sha256 FROM v_run_summary WHERE id=%s",
                      (f["runs"]["parent"],))[0]
        self.assertEqual(summ[:4], ("4:NVIDIA GeForce RTX 3090/fla-triton-sm86", "bf16", "qwen35-student", 3))
        self.assertEqual(summ[4], db.sha256_json({"engine": "qwen35-student", "dtype": "bf16"}))
        lat = self.q("SELECT parents, n_children FROM v_latest_models WHERE model_key IN ('CHILD-rt','PARENT-rt') "
                     "ORDER BY model_key")
        self.assertEqual(lat, [("PARENT-rt (distill)", 0), (None, 1)])
        layers = self.q("SELECT lower(layers), upper(layers), kernels->>'chunk_gated_delta_rule' FROM run_device "
                        "WHERE run_id=%s", (f["runs"]["parent"],))
        self.assertEqual(layers, [(0, 64, "fla")])

    def test_11_new_cases_leave_old_runs_blank(self):
        f = self._fixture_eval("bl")
        c = self.c
        suite2 = db.ensure_suite(c, "bank-bl", "2")
        for pk, i in zip(f["cases"], range(3)):
            db.add_suite_member(c, suite2, pk, i)          # carried forward: same case rows
        new_pk = db.ensure_case(c, "bl-c3", [{"role": "user", "content": "q3 new"}])
        db.add_suite_member(c, suite2, new_pk, 3)
        edited_pk = db.ensure_case(c, "bl-c1", [{"role": "user", "content": "q1 EDITED"}])
        self.assertNotIn(edited_pk, f["cases"])            # same case_id, new prompt -> new row
        rid = db.upsert_run(c, "eval:bl:child-v2", "chimera-omni", "eval", "child bank v2", "pass",
                            model_id_=f["child"], config={"engine": "qwen35-student"})
        db.upsert_eval_run(c, rid, f["child"], suite2, "qwen35-student")
        for pk in f["cases"] + [new_pk]:
            r = db.upsert_eval_result(c, rid, pk, output_text="x", termination="eos")
            db.put_grade(c, r, f["machine"], "correct", 1)
        c.commit()
        self.assertEqual(db.suite_id(c, "bank-bl"), suite2)
        self.assertEqual(db.suite_id(c, "bank-bl", "1"), f["suite1"])
        # pivot across all versions of the suite: union of member cases x runs; old runs have no c3 row
        pivot = self.q(
            "WITH cases AS (SELECT DISTINCT tc.id, tc.case_id FROM test_suite ts JOIN suite_member sm ON sm.suite_id=ts.id "
            "               JOIN test_case tc ON tc.id=sm.case_pk WHERE ts.name='bank-bl'), "
            "runs AS (SELECT run_id FROM eval_run er JOIN test_suite ts ON ts.id=er.suite_id WHERE ts.name='bank-bl') "
            "SELECT cases.case_id, runs.run_id, m.grade FROM cases CROSS JOIN runs "
            "LEFT JOIN v_eval_matrix m ON m.case_pk=cases.id AND m.eval_run_id=runs.run_id "
            "ORDER BY cases.case_id, runs.run_id")
        blanks = [(cid, r) for cid, r, g in pivot if g is None]
        self.assertEqual(sorted(blanks), sorted([("bl-c3", f["runs"]["parent"]), ("bl-c3", f["runs"]["child"])]))
        self.assertEqual(len(pivot), 4 * 3)
        cov = self.q("SELECT n_cases, n_results, coverage FROM v_results WHERE eval_run_id=%s", (rid,))
        self.assertEqual(cov, [(4, 4, 1)])

    def test_12_modality_coverage_and_artifacts(self):
        c = self.c
        m = db.upsert_model(c, "OMNI-mod", "chimera-omni", "composite", "chimera.checkpoint.v1", stage="OMNI1")
        s = db.ensure_suite(c, "media-probe", "1", modality_in=["text", "image", "audio"], modality_out=["text", "image"])
        img = db.ensure_case(c, "mp-img", [{"role": "user", "content": "describe"}],
                             media=[{"kind": "image", "path": "/x/cat.png"}], modality_in=["text", "image"])
        aud = db.ensure_case(c, "mp-aud", [{"role": "user", "content": "transcribe"}],
                             media=[{"kind": "audio", "path": "/x/a.wav"}], modality_in=["text", "audio"])
        gen = db.ensure_case(c, "mp-gen", [{"role": "user", "content": "draw a cat"}], modality_out=["image"])
        for i, pk in enumerate((img, aud, gen)):
            db.add_suite_member(c, s, pk, i)
        g = db.ensure_grader(c, "human:cyril", "human", "pass/fail")
        rid = db.upsert_run(c, "eval:mod:omni", "chimera-omni", "eval", "omni media probe", "pass", model_id_=m)
        db.upsert_eval_run(c, rid, m, s, "chimera")
        r_img = db.upsert_eval_result(c, rid, img, output_text="a cat", termination="eos")
        db.put_grade(c, r_img, g, "pass", 1)
        db.upsert_eval_result(c, rid, aud, termination="unsupported-modality")
        r_gen = db.upsert_eval_result(c, rid, gen, termination="eos")
        art = db.add_eval_artifact(c, r_gen, "image", "/x/out.png", sha256="cd" * 32, nbytes=1234, width=512, height=512)
        self.assertEqual(art, db.add_eval_artifact(c, r_gen, "image", "/x/out.png", sha256="cd" * 32))
        c.commit()
        cov = {(d, mod): (n, p, u) for d, mod, n, p, u in self.q(
            "SELECT direction, modality, n_results, n_pass, n_unsupported FROM v_model_modality_coverage "
            "WHERE model_key='OMNI-mod'")}
        self.assertEqual(cov[("in", "audio")], (1, 0, 1))
        self.assertEqual(cov[("in", "image")], (1, 1, 0))
        self.assertEqual(cov[("in", "text")], (3, 1, 1))
        self.assertEqual(cov[("out", "image")], (1, 0, 0))

    def test_13_constraints_reject_bad_values(self):
        import psycopg2
        c = self.c
        for bad in ("INSERT INTO run (run_key, project, kind, name, status) VALUES ('k1','chimera-omni','eval','x','done')",
                    "INSERT INTO run (run_key, project, kind, name, status) VALUES ('k2','nope','eval','x','pass')",
                    "INSERT INTO model (model_key, project, role, layout) VALUES ('m','chimera-omni','wizard','hf')",
                    "INSERT INTO model_tensor_group (model_id, prefix, n_tensors, modality_scope) VALUES (1,'p',1,'smell')"):
            with self.assertRaises(psycopg2.Error, msg=bad):
                with c.cursor() as cur:
                    cur.execute(bad)
            c.rollback()

    def test_14_upserts_idempotent_and_null_safe(self):
        c = self.c
        a = db.upsert_model(c, "IDEM", "neural-forge", "fold-child", "hf", family="LFM2.5-230M", parameter_count=230,
                            config={"a": 1})
        b = db.upsert_model(c, "IDEM", "neural-forge", "fold-child", "hf", config={"b": 2})
        self.assertEqual(a, b)
        self.assertEqual(self.q("SELECT family, parameter_count, config FROM model WHERE id=%s", (a,)),
                         [("LFM2.5-230M", 230, {"a": 1, "b": 2})])
        db.upsert_model(c, "IDEM", "neural-forge", "fold-child", "hf", ledger_checkpoint_id="NF-IDEM-01")
        self.assertEqual(db.model_id(c, "NF-IDEM-01"), a)
        e1 = db.ensure_env(c, torch="2.11", env_vars={"X": "1"})
        self.assertEqual(e1, db.ensure_env(c, torch="2.11", env_vars={"X": "1"}))
        self.assertNotEqual(e1, db.ensure_env(c, torch="2.11", env_vars={"X": "2"}))
        r1 = db.upsert_run(c, "distill:idem", "chimera-omni", "distill", "kd", "running", config={"lr": 1e-5})
        db.put_metrics(c, r1, [("loss", s, 3.0 - s / 1000) for s in range(1, 612)])
        db.put_metric(c, r1, "column_a", 8, denominator=15)
        db.finish_run(c, r1, "pass", exit_code=0)
        r2 = db.upsert_run(c, "distill:idem", "chimera-omni", "distill", "kd", "pass", config={"lr": 1e-5})
        self.assertEqual(r1, r2)
        c.commit()
        self.assertEqual(self.q("SELECT count(*), max(step) FROM metric WHERE run_id=%s AND name='loss'", (r1,)), [(611, 611)])
        self.assertEqual(self.q("SELECT last_loss, n_metrics FROM v_run_summary WHERE id=%s", (r1,)),
                         [(3.0 - 611 / 1000, 612)])
        self.assertEqual(self.q("SELECT value, denominator FROM metric WHERE run_id=%s AND name='column_a'", (r1,)),
                         [(8.0, 15.0)])

    def test_15_ingest_batch_and_provenance(self):
        c = self.c
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write('{"x": 1}')
            path = fh.name
        try:
            prov = db.provenance(path)
            self.assertEqual(prov[1], db.file_sha256(path))
            bid = db.start_batch(c, "unit_test_source", parser="tests/test_schema.py", argv=["--root", "/x"])
            db.upsert_model(c, "PROV", "lab", "other", "hf", prov=prov, batch_id=bid)
            self.assertFalse(db.source_sha_already_ingested(c, "unit_test_source", prov[1]))
            db.end_batch(c, bid, "ok", rows_seen=1, rows_written=1, file_shas=[prov[1]])
            c.commit()
            self.assertTrue(db.source_sha_already_ingested(c, "unit_test_source", prov[1]))
            self.assertFalse(db.source_sha_already_ingested(c, "other_source", prov[1]))
            self.assertEqual(self.q("SELECT src_sha256, ingest_batch_id FROM model WHERE model_key='PROV'"),
                             [(prov[1], bid)])
        finally:
            os.unlink(path)

    def test_16_knowledge_arch_experiment_tables(self):
        c = self.c
        m = db.upsert_model(c, "ARCH-M", "chimera-omni", "composite", "chimera.checkpoint.v1", stage="OMNI1",
                            config={"p1_channel_partition": {"k3": [4352, 16640], "trunk": [0, 4352]}})
        with c.cursor() as cur:
            cur.execute("INSERT INTO directive (directive_key, said_at, channel, verbatim, topics, source_event_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                        ("ev:1", "2026-10-21T12:00:00Z", "claude-code", "every tensor shares all modalities",
                         ["architecture", "modality"], 1))
            did = cur.fetchone()[0]
            cur.execute("INSERT INTO arch_requirement (id, text, directive_ids) VALUES (%s, %s, %s)",
                        ("R5", "no modality towers or modality routing", [did]))
            cur.execute("INSERT INTO model_tensor_group (model_id, prefix, layer_lo, layer_hi, n_tensors, modality_scope, "
                        "source_component) VALUES (%s,%s,%s,%s,%s,%s,%s), (%s,%s,%s,%s,%s,%s,%s)",
                        (m, "model.layers.*.mlp", 0, 63, 192, "shared", "qwen3.8 trunk",
                         m, "native_media_bridge", -1, -1, 10, "image", "media bridge"))
            cur.execute("INSERT INTO arch_check (model_id, requirement_id, status, evidence) VALUES (%s,%s,%s,%s)",
                        (m, "R5", "fail", "native_media_bridge is image-only"))
            cur.execute("INSERT INTO technique (name, family) VALUES ('C2 MLP coreset','fold') RETURNING id")
            tid = cur.fetchone()[0]
            cur.execute("INSERT INTO experiment (project, key, kind, technique_id, child_id, params, behaviour_legacy, "
                        "evidence_level) VALUES ('neural-forge','MX-T','fold',%s,%s,%s::jsonb,'8/15','report') RETURNING id",
                        (tid, m, db.canonical_json({"ratio": 4})))
            cur.execute("INSERT INTO ruling (ruling_key, decided_at, who, authority, decision, directive_id) "
                        "VALUES ('R-T', now(), 'cyril', 'owner', 'review K3, inspiration only', %s)", (did,))
            cur.execute("INSERT INTO lane (project, key, owner) VALUES ('k3','L-K3','claude') RETURNING id")
            lane = cur.fetchone()[0]
            cur.execute("INSERT INTO dataset (project, key, kind) VALUES ('k3','teacher_cache/k3/bf16','teacher-cache') "
                        "RETURNING id")
            ds = cur.fetchone()[0]
            cur.execute("INSERT INTO dataset_version (dataset_id, version, sealed) VALUES (%s,'abc',false) RETURNING id", (ds,))
            dv = cur.fetchone()[0]
            cur.execute("INSERT INTO ledger_event (model_id, checkpoint_id, record_seq, event, record) "
                        "VALUES (%s,'ARCH-M',1,'register','{}'::jsonb)", (m,))
            cur.execute("INSERT INTO document (sha256, path, host, doc_type, arm) VALUES ('ff', '/x/STATE.md', 'ai', 'state', 'arch')")
        rid = db.upsert_run(c, "distill:k3:x", "k3", "distill", "k3 pass", "running")
        with c.cursor() as cur:
            cur.execute("UPDATE run SET lane_id=%s WHERE id=%s", (lane, rid))
            cur.execute("INSERT INTO run_dataset (run_id, dataset_version_id, role) VALUES (%s,%s,'teacher')", (rid, dv))
        c.commit()
        self.assertEqual(self.q("SELECT status, n_modality_specific_groups FROM v_arch_conformance "
                                "WHERE model_key='ARCH-M' AND requirement='R5'"), [("fail", 1)])
        self.assertEqual(self.q("SELECT topic FROM v_directives_by_topic WHERE directive_key='ev:1' ORDER BY topic"),
                         [("architecture",), ("modality",)])
        self.assertEqual(self.q("SELECT lane FROM v_run_summary WHERE id=%s", (rid,)), [("L-K3",)])


@unittest.skipUnless(_have_rw(), "no omni_lab credential")
class ReadOnlyRoleTests(unittest.TestCase):
    def test_ro_role_cannot_write_production_db(self):
        import psycopg2
        try:
            conn = db.connect("ro", app="test-ro")
        except db.LabError as exc:
            self.skipTest(str(exc))
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM project")
                self.assertGreaterEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT count(*) FROM v_results")
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute("SET default_transaction_read_only = off")  # session flag off: the GRANTs must still hold
                with self.assertRaises(psycopg2.Error):
                    cur.execute("INSERT INTO project (key, name) VALUES ('ro-test', 'x')")
            conn.rollback()
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
