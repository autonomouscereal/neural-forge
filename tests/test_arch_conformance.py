#!/usr/bin/env python3
"""Tests for the architecture-conformance ingesters: arch_requirements and arch_check.

    C:/Python314/python.exe -m unittest tests.test_arch_conformance -v

A throwaway omni_lab_test_<hex> database is created and dropped (as test_ingest_knowledge does). The
directives come from the real inventory snapshots (2026-10-21 and 2026-10-21-arch-v2); the conformance
report is a small fixture in the chimera_omni.conformance.v1 shape, so the test needs no torch.
"""
from __future__ import annotations

import copy
import json
import re
import secrets
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import omni_lab_db as db  # noqa: E402
from ingest import arch_check, arch_requirements, directives  # noqa: E402
from ingest import common as C  # noqa: E402

TEST_DB_RE = re.compile(r"^omni_lab_test_[0-9a-f]{8}$")
INV_BASE = ROOT / "inventory" / "2026-10-21"
INV_V2 = ROOT / "inventory" / "2026-10-21-arch-v2"

REPORT = {
    "schema": "chimera_omni.conformance.v1",
    "created_at": "2026-10-21T19:00:00+00:00",
    "tool": {"module": "chimera_omni.conformance", "sha256": "0" * 64},
    "source": {"kind": "profile", "profile": "co2-fixture"},
    "config_sha256": "ab" * 32,
    "config": {"schema": "chimera_omni.config.v1", "metadata": {"architecture": "chimera-omni-v2", "profile": "co2-fixture"}},
    "parameter_count": 1000,
    "census": [
        {"prefix": "blocks.*.mixer[kda]", "layer_lo": 0, "layer_hi": 2, "n_tensors": 3, "params": 600, "dtype": "config:float32",
         "modality_scope": "shared", "source_component": "trunk:kda", "detail": {"role": "trunk", "layers": [0, 1, 2]}},
        {"prefix": "embed.weight[text]", "layer_lo": -1, "layer_hi": -1, "n_tensors": 1, "params": 300, "dtype": "config:float32",
         "modality_scope": "shared", "source_component": "token_io", "detail": {"role": "token_io"}},
        {"prefix": "native_media_bridge.type_embed[type 1]", "layer_lo": -1, "layer_hi": -1, "n_tensors": 1, "params": 100,
         "dtype": "config:float32", "modality_scope": "image", "source_component": "type_table",
         "detail": {"role": "type_indexed_row", "type_id": 1}},
    ],
    "dynamic": {"gradient_coverage": {"parameters": 5, "modality_specific": {"x[row 1]": ["image"]}},
                "type_id_invariance": {"max_abs_diff": 0.5, "pass": False}},
    "checks": [
        {"requirement_id": "R2", "status": "pass", "evidence": "dense"},
        {"requirement_id": "R3", "status": "fail", "evidence": "type-indexed tables"},
        {"requirement_id": "R5", "status": "fail", "evidence": "relabelling changed the output"},
    ],
}


def _have_rw() -> bool:
    try:
        db.db_params("rw")
        return True
    except db.LabError:
        return False


class PureTests(unittest.TestCase):
    def test_inventory_requirements_are_well_formed(self):
        doc = json.loads((INV_V2 / "arch_requirements.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], arch_requirements.SCHEMA)
        ids = [r["id"] for r in doc["requirements"]]
        self.assertEqual(ids, [f"R{i}" for i in range(1, len(ids) + 1)])
        for req in doc["requirements"]:
            self.assertTrue(req["refs"], req["id"])
            for ref in req["refs"]:
                self.assertTrue(("event" in ref) != ("key_prefix" in ref), ref)
                self.assertTrue(ref["fragment"].strip())

    def test_v2_directive_quotes_carry_their_source(self):
        items = json.loads((INV_V2 / "directives.json").read_text(encoding="utf-8"))
        self.assertTrue(items)
        for item in items:
            self.assertTrue(item["source"].startswith("agent_memory_events.id=1230100 "))
            self.assertEqual(directives.classify(item)["channel"], "claude-code")


@unittest.skipUnless(_have_rw(), "no omni_lab credential resolvable (OMNI_LAB_PG_PASSWORD[_COMMAND|_FILE])")
class ConformanceDbTests(unittest.TestCase):
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
        cls.conn = db.connect("rw", dbname=cls.dbname, app="test-arch-conformance", statement_timeout_ms=600000)
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

    def run_source(self, module, **kw):
        return C.run_source(module, SimpleNamespace(**kw), conn_factory=lambda: self.conn)

    def write_report(self, name: str, report: dict) -> Path:
        path = Path(self.tmp.name, name)
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_10_requirements_cite_owner_directives(self):
        self.run_source(directives, inventory=str(INV_BASE))
        self.run_source(directives, inventory=str(INV_V2))
        batch = self.run_source(arch_requirements, inventory=str(INV_V2))
        doc = json.loads((INV_V2 / "arch_requirements.json").read_text(encoding="utf-8"))
        rows = self.q("SELECT id, directive_ids, source_event_ids FROM arch_requirement ORDER BY length(id), id")
        self.assertEqual([r[0] for r in rows], [r["id"] for r in doc["requirements"]])
        for rid, directive_ids, event_ids in rows:
            self.assertTrue(directive_ids, rid)
        r3 = dict((r[0], r[2]) for r in rows)["R3"]
        self.assertIn(1230100, r3)          # the 2026-10-21 "[Cyril quote withheld]"
        self.assertIn(1228464, r3)          # "[Cyril quote withheld]"
        again = self.run_source(arch_requirements, inventory=str(INV_V2))
        self.assertEqual(again.written, 0)
        self.assertEqual(batch.seen, len(doc["requirements"]))

    def test_20_unresolved_fragment_refuses_the_batch(self):
        bad = Path(self.tmp.name, "bad-inventory")
        bad.mkdir()
        doc = json.loads((INV_V2 / "arch_requirements.json").read_text(encoding="utf-8"))
        doc["requirements"][0]["refs"].append({"event": 1230100, "fragment": "words Cyril never said"})
        (bad / "arch_requirements.json").write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(db.LabError):
            self.run_source(arch_requirements, inventory=str(bad))
        self.conn.rollback()
        status = self.q("SELECT status FROM ingest_batch WHERE source = 'arch_requirements' ORDER BY id DESC LIMIT 1")
        self.assertEqual(status[0][0], "failed")

    def test_30_report_writes_census_checks_and_a_probe_run(self):
        path = self.write_report("fixture.json", REPORT)
        self.run_source(arch_check, report=[str(path)], model=None)
        key = "arch:co2-fixture@" + "ab" * 6
        (mid, layout, status, path_col) = self.q("SELECT id, layout, status, path FROM model WHERE model_key = %s", (key,))[0]
        self.assertEqual((layout, status, path_col), ("chimera.config.v1", "present", None))
        scopes = dict(self.q("SELECT prefix, modality_scope FROM model_tensor_group WHERE model_id = %s", (mid,)))
        self.assertEqual(scopes["native_media_bridge.type_embed[type 1]"], "image")
        checks = dict(self.q("SELECT requirement_id, status FROM arch_check WHERE model_id = %s", (mid,)))
        n_req = self.q("SELECT count(*) FROM arch_requirement")[0][0]
        self.assertEqual(len(checks), n_req)                       # uncovered requirements are 'unknown', not blank
        self.assertEqual((checks["R2"], checks["R3"], checks["R5"], checks["R1"]), ("pass", "fail", "fail", "unknown"))
        view = self.q("SELECT n_modality_specific_groups FROM v_arch_conformance WHERE model_key = %s AND requirement = 'R3'",
                      (key,))
        self.assertEqual(view[0][0], 1)
        run = self.q("SELECT kind, status FROM run WHERE model_id = %s", (mid,))
        self.assertEqual(run, [("probe", "fail")])
        metrics = dict(self.q("SELECT m.name, m.value FROM metric m JOIN run r ON r.id = m.run_id WHERE r.model_id = %s",
                              (mid,)))
        self.assertEqual(metrics["probe/type_relabel_max_abs_diff"], 0.5)
        self.assertEqual(metrics["requirements/fail"], 2.0)

        again = self.run_source(arch_check, report=[str(path)], model=None)
        self.assertTrue(any("unchanged=" in line and "inserted=0 updated=0" in line for line in again.tally_lines))

        smaller = copy.deepcopy(REPORT)
        smaller["census"] = smaller["census"][:2]
        self.run_source(arch_check, report=[str(self.write_report("smaller.json", smaller))], model=None)
        prefixes = {r[0] for r in self.q("SELECT prefix FROM model_tensor_group WHERE model_id = %s", (mid,))}
        self.assertNotIn("native_media_bridge.type_embed[type 1]", prefixes)   # the census is replaced, not merged

    def test_40_unknown_requirement_or_model_is_refused(self):
        bogus = copy.deepcopy(REPORT)
        bogus["checks"].append({"requirement_id": "R99", "status": "pass", "evidence": "x"})
        with self.assertRaises(db.LabError):
            self.run_source(arch_check, report=[str(self.write_report("bogus.json", bogus))], model=None)
        self.conn.rollback()
        with self.assertRaises(db.LabError):
            self.run_source(arch_check, report=[str(self.write_report("ok.json", REPORT))], model="no-such-model")
        self.conn.rollback()


if __name__ == "__main__":
    unittest.main()
