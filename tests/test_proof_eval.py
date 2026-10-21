#!/usr/bin/env python3
"""Tests for ingest/proof_eval.py (omni_lab.py log-eval).

    C:/Python314/python.exe -m unittest tests.test_proof_eval -v

A throwaway omni_lab_test_<hex> database is created and dropped. Two fixture files share one suite (the
untyped model and its typed control); the test checks the suite is shared, results/grades/artifacts land,
the experiment links both eval runs, a reload is idempotent, and a changed case list under the same suite
version is refused.
"""
from __future__ import annotations

import copy
import json
import re
import secrets
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import omni_lab_db as db  # noqa: E402
from ingest import common as C  # noqa: E402
from ingest import proof_eval  # noqa: E402

TEST_DB_RE = re.compile(r"^omni_lab_test_[0-9a-f]{8}$")

DOC = {
    "schema": "omni-lab-proof-eval-v1",
    "suite": {"name": "fixture_proof", "version": "1", "description": "fixture"},
    "graders": {"fixture.exact": {"kind": "machine", "scale": "pass/fail", "description": "exact answer"}},
    "cases": [
        {"id": "fx-img-1", "category": "image_class", "prompt": [{"role": "user", "content": "<image> what is it?"}],
         "media": [{"kind": "image", "index": 1}], "expected": "zebra", "grader": "fixture.exact",
         "modality_in": ["image", "text"], "modality_out": ["text"]},
        {"id": "fx-gen-1", "category": "image_gen", "prompt": [{"role": "user", "content": "draw zebra"}],
         "expected": "zebra", "grader": "fixture.exact", "modality_in": ["text"], "modality_out": ["image"]},
    ],
    "model": {"model_key": "fixture:untyped", "layout": "chimera.proof.v1", "role": "other", "parameter_count": 10},
    "run": {"run_key": "eval:fixture:untyped", "name": "fixture untyped", "engine": "proof", "host": "fixture-host",
            "started_utc": "2026-10-21T20:00:00Z", "ended_utc": "2026-10-21T20:01:00Z"},
    "env": {"python": "3.11", "torch": "2.13", "code_commit": "abc"},
    "results": [
        {"case_id": "fx-img-1", "output_text": "zebra", "termination": "eos",
         "grade": {"grader": "fixture.exact", "grade": "pass", "score": 1.0}},
        {"case_id": "fx-gen-1", "output_text": "giraffe", "termination": "other",
         "grade": {"grader": "fixture.exact", "grade": "fail", "score": 0.0, "why": "classifier said giraffe"},
         "artifacts": [{"kind": "image", "path": "/x/gen.png", "sha256": "0" * 64, "bytes": 10, "width": 64,
                        "height": 64}]},
    ],
    "metrics": [{"name": "image_class", "value": 1.0, "denominator": 1}],
    "experiment": {"key": "FX-ARCH-1", "kind": "arch", "role": "child", "verdict": "fixture", "date": "2026-10-21"},
}


def _have_rw() -> bool:
    try:
        db.db_params("rw")
        return True
    except db.LabError:
        return False


@unittest.skipUnless(_have_rw(), "no omni_lab credential resolvable (OMNI_LAB_PG_PASSWORD[_COMMAND|_FILE])")
class ProofEvalDbTests(unittest.TestCase):
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
        cls.conn = db.connect("rw", dbname=cls.dbname, app="test-proof-eval", statement_timeout_ms=600000)
        db.migrate(cls.conn)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        from psycopg2 import sql
        cls.conn.close()
        cls.tmp.cleanup()
        assert TEST_DB_RE.match(cls.dbname), cls.dbname
        admin = db.connect("rw", app="test-drop", autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(cls.dbname)))
        finally:
            admin.close()

    def setUp(self):
        self.conn.rollback()

    def q(self, query, params=None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def write(self, name: str, doc: dict) -> str:
        path = Path(self.tmp.name, name)
        path.write_text(json.dumps(doc), encoding="utf-8")
        return str(path)

    def load(self, *paths: str):
        return C.run_source(proof_eval, SimpleNamespace(file=list(paths)), conn_factory=lambda: self.conn)

    def test_10_two_models_share_one_suite_and_one_experiment(self):
        typed = copy.deepcopy(DOC)
        typed["model"]["model_key"] = "fixture:typed"
        typed["run"]["run_key"] = "eval:fixture:typed"
        typed["experiment"]["role"] = "baseline"
        typed["results"][0]["output_text"] = "pizza"
        typed["results"][0]["grade"] = {"grader": "fixture.exact", "grade": "fail", "score": 0.0}
        self.load(self.write("untyped.json", DOC), self.write("typed.json", typed))
        suites = self.q("SELECT id FROM test_suite WHERE name = 'fixture_proof'")
        self.assertEqual(len(suites), 1)
        members = self.q("SELECT count(*) FROM suite_member WHERE suite_id = %s", (suites[0][0],))
        self.assertEqual(members[0][0], 2)
        grades = dict(self.q(
            "SELECT m.model_key || ':' || tc.case_id, g.grade FROM eval_grade g JOIN eval_result r ON r.id = g.eval_result_id "
            "JOIN eval_run er ON er.run_id = r.eval_run_id JOIN model m ON m.id = er.model_id "
            "JOIN test_case tc ON tc.id = r.case_pk"))
        self.assertEqual(grades, {"fixture:untyped:fx-img-1": "pass", "fixture:untyped:fx-gen-1": "fail",
                                  "fixture:typed:fx-img-1": "fail", "fixture:typed:fx-gen-1": "fail"})
        self.assertEqual(self.q("SELECT count(*) FROM eval_artifact")[0][0], 2)
        roles = sorted(r[0] for r in self.q("SELECT role FROM experiment_eval ee JOIN experiment e ON e.id = ee.experiment_id "
                                            "WHERE e.key = 'FX-ARCH-1'"))
        self.assertEqual(roles, ["baseline", "child"])
        metric = self.q("SELECT value, denominator FROM metric m JOIN run r ON r.id = m.run_id "
                        "WHERE r.run_key = 'eval:fixture:untyped' AND m.name = 'image_class'")
        self.assertEqual(metric, [(1.0, 1.0)])

    def test_20_reload_is_idempotent(self):
        path = self.write("again.json", DOC)
        before = self.q("SELECT count(*) FROM eval_result")[0][0]
        self.load(path)
        self.assertEqual(self.q("SELECT count(*) FROM eval_result")[0][0], before)
        self.assertEqual(self.q("SELECT count(*) FROM run WHERE run_key = 'eval:fixture:untyped'")[0][0], 1)

    def test_30_changed_cases_under_the_same_version_are_refused(self):
        changed = copy.deepcopy(DOC)
        changed["cases"] = changed["cases"][:1]
        changed["results"] = changed["results"][:1]
        changed["run"]["run_key"] = "eval:fixture:changed"
        with self.assertRaises(db.LabError):
            self.load(self.write("changed.json", changed))
        self.conn.rollback()

    def test_40_results_for_unknown_cases_are_refused(self):
        bad = copy.deepcopy(DOC)
        bad["results"].append({"case_id": "nope", "output_text": "x"})
        with self.assertRaises(db.LabError):
            self.load(self.write("bad.json", bad))
        self.conn.rollback()


if __name__ == "__main__":
    unittest.main()
