#!/usr/bin/env python3
"""Tests for migration 0007 and ingest/foldrec.py (the Neural Forge folding record, schema foldrec-v1).

    C:/Python314/python.exe -m unittest tests.test_ingest_foldrec -v

Pure tests run the parsers and the plan builder on synthetic shards in a temp directory.  DB tests create a
throwaway omni_lab_test_<hex> database (as test_schema does), load the real foldhist inventory first (so the
43 summary experiments exist to link to), then run foldrec through ingest.common.run_source, check rows and
views against the fixture, run it again to prove it writes nothing, and change a shard to prove surplus rows
are removed.  The event-sourced directive is checked against the real agent_memory event 506681 when that
database is readable from here; otherwise the test expects the 'unreadable' skip.
"""
from __future__ import annotations

import copy
import json
import re
import secrets
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import omni_lab_db as db  # noqa: E402
from ingest import common as C  # noqa: E402
from ingest import foldhist, foldrec  # noqa: E402

TEST_DB_RE = re.compile(r"^omni_lab_test_[0-9a-f]{8}$")
INV = ROOT / "inventory" / "2026-10-21"
EV_QUOTE = "install the new clip model as our default internal clip model over the old clip model"


def _src(path, lines="1-2", quote="q"):
    return {"path": path, "lines": lines, "commit": None, "ev": None, "quote": quote}


def _arm(key, **kw):
    rec = {"key": key, "aliases": [], "title": f"title {key}", "date": "2026-07-11", "date_text": "2026-07-11",
           "program": "matrix-campaign", "wave": "MX wave 1", "owner": "codex",
           "parent": {"name": "LFM2.5-230M", "params": "229,693,184"},
           "child": {"ref": None, "params": None, "exists_now": "unknown"},
           "components": [], "axes": 1, "ratio_body": None, "ratio_net": None, "ratio_text": None,
           "objective": {"fit": "ridge", "target": "activation", "k": None, "horizon": None, "text": "ridge on acts"},
           "training_used": "none", "training_text": None,
           "behaviour": {"suite": None, "scorer": "none", "score": None, "useful": None, "partial": None, "miss": None,
                         "total": None, "column_b": None, "parent_score": None, "text": None},
           "diagnostics": [], "verdict_class": "fails", "verdict_text": "as written", "cyril": None,
           "sources": [_src("/x/NOTES.md", "1-9", f"quote for {key}")], "evidence_level": "notes"}
    rec.update(kw)
    return rec


def fixture_shards(tmp: Path, doc_path: Path) -> dict:
    """Two shards: 'shard-a' and 'shard-b'.  Returns {name: doc} (also written to tmp/<name>.json)."""
    comp_t1 = {"kind": "mlp_width", "layers": "0-13", "technique": "T1 ridge", "ratio": Decimal("0.25"),
               "ratio_text": "keep 1/4", "rank_or_k": "k=256", "notes": None}
    a = {"schema": "foldrec-v1", "shard": "shard-a", "extracted_at": "2026-10-22T13:00:00+00:00",
         "sources_read": [{"path": "/x/NOTES.md", "lines": "all", "note": ""},
                          {"path": "/x/NOTES-2.md", "lines": "1-50", "note": "first half"}],
         "sources_not_read": [{"path": "/x/BIG.log", "reason": "binary"}],
         "experiments": [
             _arm("MX-T.C1", aliases=["NF-02", "c1-short"], components=[comp_t1], ratio_body=Decimal("0.30"),
                  ratio_text="0.30 body", verdict_class="works",
                  behaviour={"suite": "frozen-15", "scorer": "cyril", "score": "12/15", "useful": 12, "partial": 2,
                             "miss": 1, "total": 15, "column_b": "B1", "parent_score": "15/15", "text": "ok"},
                  diagnostics=[{"name": "tf_top1", "value": "0.912", "horizon": "8", "denominator": "512",
                                "split": "held", "note": None},
                               {"name": "kl", "value": Decimal("0.0310"), "horizon": None, "denominator": None,
                                "split": "held", "note": None},
                               {"name": "other", "value": "56/64", "horizon": None, "denominator": None,
                                "split": None, "note": "rollout"}],
                  cyril="that's actually good (ev 123)"),
             _arm("MX-T.C1#wave2", aliases=["MX-T.C1"], wave="MX wave 2", verdict_class="partial",
                  components=[{**comp_t1, "technique": "t1   RIDGE", "ratio": Decimal("0.5")}]),
             _arm("X3-08B", program="chimera-omni", verdict_class="partial", evidence_level="receipt",
                  components=[{"kind": "experts_to_dense", "layers": "all", "technique": "C2 coreset", "ratio": None,
                               "ratio_text": "8 of 128", "rank_or_k": None, "notes": None}]),
             _arm("ARM-S1", aliases=["MX-4X"], program="nf-lfm2", verdict_class="closed_path",
                  child={"ref": "OC01-F-Q35-4B-X03", "params": "1.0B", "exists_now": "deleted"},
                  sources=[_src("/runs/fold1/receipt.json", "all", "receipt")]),
             _arm("ARM-S2", aliases=["MX-4X"], components=[{"kind": "warp_drive", "layers": "1", "technique": None,
                                                            "ratio": None, "ratio_text": None, "rank_or_k": None,
                                                            "notes": None}]),
             {"title": "no key here"},
         ],
         "findings": [{"key": "F1", "statement": "ridge alone never holds behaviour at 4x", "status": "proven",
                       "scope": "LFM2 230M", "experiment_keys": ["MX-T.C1", "NF-02", "NOPE-1"],
                       "sources": [_src("/x/NOTES.md", "3-4", "never holds")]},
                      {"key": "F2", "statement": "same words", "status": "open", "scope": None,
                       "experiment_keys": ["ARM-S1"], "sources": []}],
         "techniques": [{"name": "T1 ridge", "family": "linear fit", "description": "dense ridge",
                         "code": "src/neural_forge/fit.py:_fit_linear @abc1234", "first_date": "2026-07-01",
                         "status_text": "superseded by T5"}],
         "directives": [
             {"date": "2026-08-10", "quote": "grade behaviour first, never token match", "topic": "grading",
              "source": {"ev": None, "path": doc_path.as_posix(), "line": 2}, "speaker_check": "quoted in doc"},
             {"date": "2026-08-10", "quote": "words that are nowhere", "topic": "grading",
              "source": {"ev": None, "path": doc_path.as_posix(), "line": 2}, "speaker_check": "x"},
             {"date": "2026-08-10", "quote": "no source given", "topic": "other",
              "source": {"ev": None, "path": None, "line": None}, "speaker_check": "x"},
             {"date": "2026-08-10", "quote": "already said this", "topic": "process",
              "source": {"ev": None, "path": doc_path.as_posix(), "line": 3}, "speaker_check": "x"},
             {"date": None, "quote": EV_QUOTE, "topic": "other", "source": {"ev": 506681, "path": None, "line": None},
              "speaker_check": "user_prompt event"},
         ]}
    b = {"schema": "foldrec-v1", "shard": "shard-b", "extracted_at": "2026-10-22T14:00:00Z",
         "sources_read": [{"path": "/x/NOTES-2.md", "lines": "51-100", "note": ""}],
         "sources_not_read": [{"path": "/x/NOTES-2.md", "reason": "rest skipped"}],
         "experiments": [
             _arm("MX-T.C1", ratio_body=Decimal("0.25"), verdict_class="partial", title="title MX-T.C1",
                  components=[comp_t1],
                  behaviour={"suite": "frozen-15", "scorer": "claude", "score": "9/15", "useful": 9, "partial": 3,
                             "miss": 3, "total": 15, "column_b": None, "parent_score": None, "text": None},
                  sources=[_src("/x/NOTES.md", "1-9", "quote for MX-T.C1"), _src("/x/OTHER.md", "7", "other")]),
         ],
         "findings": [{"key": "F1", "statement": "a different statement under the same key", "status": "supported",
                       "scope": None, "experiment_keys": [], "sources": []},
                      {"key": "F2", "statement": "same words", "status": "open", "scope": "s2",
                       "experiment_keys": ["MX-T.C1"], "sources": []}],
         "techniques": [{"name": "t1 ridge", "family": "ridge", "description": None, "code": None,
                         "first_date": "2026-07-03", "status_text": "superseded by T5"}],
         "directives": [
             {"date": "2026-08-10", "quote": "grade behaviour first, never token match", "topic": "fold-objective",
              "source": {"ev": None, "path": doc_path.as_posix(), "line": 2}, "speaker_check": "doc again"},
             {"date": "2026-08-10", "quote": "behaviour first", "topic": "grading",
              "source": {"ev": None, "path": doc_path.as_posix(), "line": 2}, "speaker_check": "fragment"},
             {"date": "2026-08-10", "quote": "never token match ... already said this", "topic": "grading",
              "source": {"ev": None, "path": doc_path.as_posix(), "line": 2}, "speaker_check": "elided"},
             {"date": "2026-08-10", "quote": "keep the negatives", "topic": "process",
              "source": {"ev": 506681, "path": doc_path.as_posix(), "line": 4}, "speaker_check": "ev + doc"},
         ]}
    docs = {"shard-a": a, "shard-b": b}
    write_shards(tmp, docs)
    return docs


def write_shards(tmp: Path, docs: dict) -> None:
    """JSON with every Decimal written as its exact digits ('0.0310' stays 0.0310 in the file)."""
    def mark(v):
        if isinstance(v, Decimal):
            return f"__DEC__{v}"
        if isinstance(v, dict):
            return {k: mark(x) for k, x in v.items()}
        if isinstance(v, list):
            return [mark(x) for x in v]
        return v
    for name, doc in docs.items():
        text = re.sub(r'"__DEC__([-0-9.Ee+]+)"', r"\1", json.dumps(mark(doc), ensure_ascii=False, indent=1))
        with open(tmp / f"{name}.json", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)


class PureTests(unittest.TestCase):
    def test_value_parsers(self):
        self.assertEqual(foldrec.as_number("1,024"), Decimal("1024"))
        self.assertEqual(foldrec.as_number(Decimal("0.250")), Decimal("0.250"))
        self.assertIsNone(foldrec.as_number("4x"))
        self.assertIsNone(foldrec.as_number("about 0.25"))
        self.assertIsNone(foldrec.as_number(True))
        self.assertEqual(foldrec.parse_value("31.2%"), (31.2, "%"))
        self.assertEqual(foldrec.parse_value("56/64"), (0.875, "fraction"))
        self.assertEqual(foldrec.parse_value("-12.5 dB"), (-12.5, "dB"))
        self.assertEqual(foldrec.parse_value("4x"), (4.0, "x"))
        self.assertEqual(foldrec.parse_value("0.9 then 0.8"), (None, None))
        self.assertEqual(foldrec.parse_score("12/15"), (Decimal(12), Decimal(15)))
        self.assertEqual(foldrec.parse_score("56/64->64/64"), (None, None))
        self.assertEqual(foldrec.as_text(Decimal("0.0310")), "0.0310")       # digits kept as written
        self.assertEqual(foldrec.ev_id("ev:506681"), 506681)
        self.assertEqual(foldrec.ev_id("agent_memory_events.id=12 (x)"), 12)
        self.assertIsNone(foldrec.ev_id("none"))
        self.assertEqual(foldrec.jsonable({"a": [Decimal("2"), Decimal("0.5")]}), {"a": [2, 0.5]})

    def test_plan_merges_by_key_only_and_keeps_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc_path = Path(tmp, "doc.md")
            doc_path.write_text("x\n", encoding="utf-8")
            fixture_shards(Path(tmp), doc_path)
            plan = foldrec.build_plan(Path(tmp))
        self.assertEqual(plan.errors, [])
        arms = {a["key"]: a for a in plan.arms}
        self.assertEqual(sorted(arms), ["ARM-S1", "ARM-S2", "MX-T.C1", "MX-T.C1#wave2", "X3-08B"])
        c1 = arms["MX-T.C1"]
        self.assertEqual(c1["shards"], ["shard-a", "shard-b"])
        self.assertIsNone(c1["ratio_body"])                         # 0.30 vs 0.25: never picked
        self.assertIsNone(c1["verdict_class"])                      # works vs partial
        self.assertEqual([v["value"] for v in c1["conflicts"]["ratio_body"]], [0.3, 0.25])
        self.assertEqual([v["shard"] for v in c1["conflicts"]["verdict_class"]], ["shard-a", "shard-b"])
        self.assertNotIn("title", c1["conflicts"])                  # identical text is not a conflict
        self.assertEqual(len(c1["components"]), 1)                  # identical component from both shards
        self.assertEqual(c1["components"][0]["shards"], ["shard-a", "shard-b"])
        self.assertEqual([b["score"] for b in c1["behaviour"]], ["12/15", "9/15"])
        self.assertEqual(len(c1["sources"]), 2)
        self.assertEqual(c1["sources"][0]["shards"], ["shard-a", "shard-b"])
        self.assertEqual(c1["aliases"], ["NF-02", "c1-short"])
        self.assertEqual(arms["MX-T.C1#wave2"]["aliases"], ["MX-T.C1"])   # an alias never merges
        self.assertEqual(arms["MX-T.C1#wave2"]["behaviour"], [])         # 'nothing measured' is not a row
        self.assertIsNone(arms["ARM-S1"]["program"])
        self.assertEqual(arms["ARM-S2"]["components"][0]["kind"], "other")
        self.assertTrue(any("program='nf-lfm2'" in p for p in plan.problems))
        self.assertTrue(any("warp_drive" in p for p in plan.problems))
        self.assertEqual(plan.skipped, ["shard-a#5: experiment without a key"])
        self.assertEqual(plan.alias_only, {"MX-4X": ["ARM-S1", "ARM-S2"]})
        findings = {f["key"]: f for f in plan.findings}
        self.assertEqual(sorted(findings), ["F2", "shard-a:F1", "shard-b:F1"])
        self.assertEqual(findings["F2"]["shards"], ["shard-a", "shard-b"])
        self.assertEqual(findings["F2"]["experiment_keys"], ["ARM-S1", "MX-T.C1"])
        self.assertEqual(findings["F2"]["scope"], "s2")
        t1 = plan.techniques[foldrec.norm_name("T1 ridge")]
        self.assertEqual(len(t1["entries"]), 2)
        self.assertEqual(sorted(t1["arms"]), ["MX-T.C1", "MX-T.C1#wave2"])
        self.assertEqual(t1["cols"]["first_date_text"], "2026-07-01")
        self.assertIn("first_date_text", t1["cols"]["_conflicting"])
        links = foldrec.alias_links(plan, {"NF-02": (1, "foldhist"), "MX-4X": (2, "foldhist"),
                                           **{a["key"]: (0, "foldrec") for a in plan.arms}})
        self.assertEqual([(f, t, r) for f, t, r, _d in links],
                         [("MX-4X", "ARM-S1", "summarizes"), ("MX-4X", "ARM-S2", "summarizes"),
                          ("MX-T.C1#wave2", "MX-T.C1", "rerun_of"), ("NF-02", "MX-T.C1", "same_as")])

    def test_anchor_quote_and_rerun_rule(self):
        src = "x a  b c d e\nwe keep going now"
        self.assertEqual(foldrec.anchor_quote("a b c", src), "a  b c")
        self.assertEqual(foldrec.anchor_quote("a b ... d e", src), "a  b ... d e")
        self.assertEqual(foldrec.anchor_quote("a b … keep going", src), "a  b ... keep going")
        self.assertIsNone(foldrec.anchor_quote("d e ... a b", src))          # fragments must be in order
        self.assertEqual(foldrec.anchor_quote("keep going...", src), "keep going")
        self.assertIsNone(foldrec.anchor_quote("...", src))
        self.assertIsNone(foldrec.anchor_quote("a b", None))
        plan = foldrec.Plan()
        plan.arms = [{"key": "A", "aliases": []}, {"key": "A@w3", "aliases": ["A"]}, {"key": "A#x", "aliases": ["A"]},
                     {"key": "AB", "aliases": ["A"]}]
        self.assertEqual([(f, t, r) for f, t, r, _d in foldrec.alias_links(plan, {})],
                         [("A#x", "A", "rerun_of"), ("A@w3", "A", "rerun_of"), ("AB", "A", "same_as")])

    def test_plan_refuses_bad_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad.json").write_text("{not json", encoding="utf-8")
            Path(tmp, "wrong.json").write_text(json.dumps({"schema": "foldhist-v1", "shard": "wrong"}), encoding="utf-8")
            Path(tmp, "one.json").write_text(json.dumps({"schema": "foldrec-v1", "shard": "dup"}), encoding="utf-8")
            Path(tmp, "two.json").write_text(json.dumps({"schema": "foldrec-v1", "shard": "dup"}), encoding="utf-8")
            plan = foldrec.build_plan(Path(tmp))
        self.assertEqual(len(plan.errors), 3, plan.errors)
        self.assertTrue(any("bad.json" in e and "JSON" in e for e in plan.errors))
        self.assertTrue(any("wrong.json" in e for e in plan.errors))
        self.assertTrue(any("two.json" in e and "already used" in e for e in plan.errors))
        self.assertEqual([s.name for s in plan.shards], ["dup"])
        self.assertTrue(any("one.json" in p for p in plan.problems))     # file name != shard name


def _have_rw() -> bool:
    try:
        db.db_params("rw")
        return True
    except db.LabError:
        return False


def _agent_memory_readable() -> bool:
    try:
        conn = db.connect("admin", dbname="agent_memory", app="test-foldrec-probe")
    except Exception:
        return False
    conn.close()
    return True


@unittest.skipUnless(_have_rw(), "no omni_lab credential resolvable (OMNI_LAB_PG_PASSWORD[_COMMAND|_FILE])")
class FoldrecDbTests(unittest.TestCase):
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
        cls.conn = db.connect("rw", dbname=cls.dbname, app="test-ingest-foldrec", statement_timeout_ms=600000)
        db.migrate(cls.conn)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name, "shards")
        cls.dir.mkdir()
        cls.doc_path = Path(cls.tmp.name, "MEMORY_NOTE.md")
        cls.doc_path.write_text("# note 2026-08-10\nCyril: \"grade behaviour first, never token match\"\n"
                                "and \"already said this\" too\nCyril also: \"keep the negatives\"\n", encoding="utf-8")
        cls.docs = fixture_shards(cls.dir, cls.doc_path)
        cls.ev_readable = _agent_memory_readable()

    @classmethod
    def tearDownClass(cls):
        import os
        from psycopg2 import sql
        cls.conn.close()
        cls.tmp.cleanup()
        if os.environ.get("OMNI_LAB_TEST_KEEP") == "1":
            print(f"\n[kept] {cls.dbname}")
            return
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

    def ingest(self, module, inventory):
        return C.run_source(module, SimpleNamespace(inventory=str(inventory)), conn_factory=lambda: self.conn)

    @staticmethod
    def tally(batch) -> dict:
        out = {}
        for line in batch.tally_lines:
            table, _sep, rest = line.partition(": ")
            if "inserted=" in rest:
                out[table] = dict(kv.split("=") for kv in rest.split())
        return out

    def test_00_schema_objects_and_grants(self):
        tables = {r[0] for r in self.q("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")}
        for t in ("fold_shard", "fold_coverage", "fold_arm", "fold_component", "fold_behaviour", "fold_diagnostic",
                  "fold_source", "experiment_alias", "experiment_link", "fold_finding", "fold_finding_experiment",
                  "fold_finding_source", "v_fold_arms", "v_technique_scoreboard", "v_findings", "v_fold_coverage"):
            self.assertIn(t, tables)
            if self.q("SELECT 1 FROM pg_roles WHERE rolname='omni_lab_ro'"):
                self.assertTrue(self.q("SELECT has_table_privilege('omni_lab_ro', %s, 'SELECT')", (t,))[0][0], t)
        cols = {r[0] for r in self.q("SELECT column_name FROM information_schema.columns WHERE table_name='technique'")}
        self.assertTrue({"code_location", "first_date", "first_date_text", "status_text", "detail"} <= cols)

    def test_01_constraints(self):
        import psycopg2
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO experiment (project, key, kind, evidence_level) "
                        "VALUES ('neural-forge', 'EVID-NOTES', 'fold', 'notes') RETURNING id")
            eid = cur.fetchone()[0]
        self.conn.rollback()
        self.assertIsNotNone(eid)
        for bad in ("INSERT INTO experiment (project, key, kind, evidence_level) VALUES ('neural-forge','E2','fold','rumour')",
                    "INSERT INTO fold_shard (shard, schema) VALUES ('s', 'x'); INSERT INTO fold_coverage (shard, ordinal, "
                    "state, path) VALUES ('s', 0, 'skimmed', '/p')"):
            with self.assertRaises(psycopg2.Error, msg=bad):
                with self.conn.cursor() as cur:
                    cur.execute(bad)
            self.conn.rollback()

    def test_10_dry_run_writes_nothing(self):
        before = self.q("SELECT (SELECT count(*) FROM ingest_batch), (SELECT count(*) FROM experiment)")[0]
        rep = foldrec.dry_run(SimpleNamespace(inventory=str(self.dir)), conn=self.conn)
        self.assertEqual(rep["errors"], [])
        self.assertEqual(rep["arms"], 5)
        self.assertEqual(rep["records_seen"], 7 + 4 + 2 + 9)
        self.assertEqual(rep["shards"]["shard-a"]["experiments"], 6)
        self.assertEqual(self.q("SELECT (SELECT count(*) FROM ingest_batch), (SELECT count(*) FROM experiment)")[0], before)

    def test_20_ingest_links_and_views(self):
        self.ingest(foldhist, INV)
        x3_before = self.q("SELECT technique_id, verdict, src_path, ingest_batch_id, params FROM experiment "
                           "WHERE key='X3-08B'")[0]
        n_hist = self.q("SELECT count(*) FROM experiment")[0][0]
        child = db.upsert_model(self.conn, "OC01-F-Q35-4B-X03", "chimera-omni", "fold-child", "hf",
                                ledger_checkpoint_id="OC01-F-Q35-4B-X03")
        rid = db.upsert_run(self.conn, "fold:test:receipt", "neural-forge", "fold", "fold1", "pass",
                            receipt_path="/runs/fold1/receipt.json", run_dir="/runs/fold1")
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO directive (directive_key, said_at, channel, verbatim) "
                        "VALUES ('ev:1:pre', now(), 'codex', 'I have already said this before')")
        self.conn.commit()
        b = self.ingest(foldrec, self.dir)
        t = self.tally(b)
        self.assertEqual(self.q("SELECT count(*) FROM experiment")[0][0], n_hist + 4)   # X3-08B attaches
        self.assertEqual(self.q("SELECT count(*) FROM fold_arm")[0][0], 5)
        # an existing (foldhist) experiment keeps its own columns; the arm detail hangs off it
        self.assertEqual(self.q("SELECT technique_id, verdict, src_path, ingest_batch_id, params FROM experiment "
                                "WHERE key='X3-08B'")[0], x3_before)
        self.assertEqual(self.q("SELECT fa.program, fa.evidence_level FROM fold_arm fa JOIN experiment e "
                                "ON e.id=fa.experiment_id WHERE e.key='X3-08B'"), [("chimera-omni", "receipt")])
        # disagreeing numbers stay NULL with both variants and their shards
        c1 = self.q("SELECT ratio_body, verdict_class, conflicts, shards, run_ids FROM fold_arm WHERE arm_key='MX-T.C1'")[0]
        self.assertEqual(c1[:2], (None, None))
        self.assertEqual(sorted(c1[2]), ["ratio_body", "verdict_class"])
        self.assertEqual([v["shard"] for v in c1[2]["ratio_body"]], ["shard-a", "shard-b"])
        self.assertEqual(c1[3], ["shard-a", "shard-b"])
        diag = self.q("SELECT name, value_text, value_num, value_unit, horizon FROM fold_diagnostic fd JOIN experiment e "
                      "ON e.id=fd.experiment_id WHERE e.key='MX-T.C1' ORDER BY ordinal")
        self.assertEqual(diag, [("tf_top1", "0.912", 0.912, None, "8"), ("kl", "0.0310", 0.031, None, None),
                                ("other", "56/64", 0.875, "fraction", None)])
        beh = self.q("SELECT score, score_value, score_denominator, scorer, shards FROM fold_behaviour fb "
                     "JOIN experiment e ON e.id=fb.experiment_id WHERE e.key='MX-T.C1' ORDER BY ordinal")
        self.assertEqual([(s, int(v), int(d), sc, sh) for s, v, d, sc, sh in beh],
                         [("12/15", 12, 15, "cyril", ["shard-a"]), ("9/15", 9, 15, "claude", ["shard-b"])])
        # model + run links on a foldrec-owned experiment
        s1 = self.q("SELECT e.child_id, e.run_id, fa.run_ids, fa.program, e.project FROM experiment e "
                    "JOIN fold_arm fa ON fa.experiment_id=e.id WHERE e.key='ARM-S1'")[0]
        self.assertEqual(s1, (child, rid, [rid], None, "neural-forge"))
        # aliases and links
        links = set(self.q("SELECT f.key, t.key, l.relation FROM experiment_link l JOIN experiment f ON f.id=l.from_id "
                           "JOIN experiment t ON t.id=l.to_id"))
        self.assertEqual(links, {("NF-02", "MX-T.C1", "same_as"), ("MX-4X", "ARM-S1", "summarizes"),
                                 ("MX-4X", "ARM-S2", "summarizes"), ("MX-T.C1#wave2", "MX-T.C1", "rerun_of")})
        self.assertEqual(self.q("SELECT count(*) FROM experiment_alias WHERE alias='MX-4X'")[0][0], 2)
        # techniques: name match is case/space-insensitive; foldrec owns the 0007 columns
        tech = self.q("SELECT id, code_location, first_date::text, status_text, family FROM technique WHERE name='T1 ridge'")
        self.assertEqual(len(tech), 1)
        self.assertEqual(tech[0][1:], ("src/neural_forge/fit.py:_fit_linear @abc1234", "2026-07-01", "superseded by T5",
                                       "fold"))
        self.assertEqual(self.q("SELECT DISTINCT fc.technique_id FROM fold_component fc JOIN experiment e ON "
                                "e.id=fc.experiment_id WHERE e.key IN ('MX-T.C1','MX-T.C1#wave2')"), [(tech[0][0],)])
        # findings: same key + different statements -> namespaced; keys resolve through aliases
        fe = dict(self.q("SELECT fe.experiment_key, e.key FROM fold_finding_experiment fe JOIN fold_finding f "
                         "ON f.id=fe.finding_id LEFT JOIN experiment e ON e.id=fe.experiment_id "
                         "WHERE f.finding_key='shard-a:F1'"))
        self.assertEqual(fe, {"MX-T.C1": "MX-T.C1", "NF-02": "MX-T.C1", "NOPE-1": None})
        self.assertEqual(self.q("SELECT finding_key FROM fold_finding ORDER BY 1"), [("F2",), ("shard-a:F1",), ("shard-b:F1",)])
        # directives: verified + new only
        rows = dict(self.q("SELECT verbatim, channel FROM directive WHERE ingest_batch_id=%s", (b.id,)))
        expect = {"grade behaviour first, never token match": "doc", "never token match ... already said this": "doc",
                  "keep the negatives": "doc"}
        if self.ev_readable:
            expect[EV_QUOTE] = "codex"
        self.assertEqual(rows, expect)
        said = self.q("SELECT said_at::date::text, said_at_verified, src_path, topics, summary FROM directive "
                      "WHERE verbatim=%s", ("grade behaviour first, never token match",))
        self.assertEqual(said, [("2026-08-10", False, f"{self.doc_path.as_posix()}:2", ["fold-objective", "grading"],
                                 "foldrec shards shard-a, shard-b; speaker_check: quoted in doc | doc again; "
                                 "date 2026-08-10 from shard date (day only)")])
        fallback = self.q("SELECT summary FROM directive WHERE verbatim='keep the negatives'")[0][0]
        self.assertIn("[event 506681: " + ("not-found" if self.ev_readable else "unreadable")
                      + "; verified in the cited file instead]", fallback)
        notes = " ".join(b.notes)
        for expected in ("already-present: as ev:1:pre", "not-found", "no-source", "same-key-merged",
                         "contained: in doc:"):
            self.assertIn(expected, notes)
        # coverage
        self.assertEqual(self.q("SELECT count(*) FROM fold_coverage")[0][0], 3 + 2)
        cov = dict((p, (r, nr, n, c)) for p, r, nr, n, c in self.q(
            "SELECT path, shards_read, shards_not_read, arms_citing, cited_not_declared_read FROM v_fold_coverage"))
        self.assertEqual(cov["/x/NOTES-2.md"], ("shard-a, shard-b", "shard-b", 0, False))
        self.assertEqual(cov["/x/OTHER.md"], (None, None, 1, True))
        # views
        arm = self.q("SELECT components, techniques, behaviour_score, scorer, n_behaviour, diagnostics, aliases, "
                     "linked_from, has_conflicts FROM v_fold_arms WHERE key='MX-T.C1'")[0]
        self.assertEqual(arm[0], "mlp_width[0-13] T1 ridge r=0.25 k=k=256")
        self.assertEqual(arm[1:5], ("T1 ridge", "12/15", "cyril", 2))
        self.assertEqual(arm[5], "tf_top1=0.912@8 (held); kl=0.0310 (held); other=56/64")
        self.assertEqual(sorted(arm[6].split(", ")), ["NF-02", "c1-short"])     # collation-independent
        self.assertEqual(arm[7], "MX-T.C1#wave2 rerun_of, NF-02 same_as")
        self.assertIs(arm[8], True)
        sb = self.q("SELECT arms, arms_with_behaviour, best_behaviour, best_behaviour_text, tested_at_4x, arms_at_4x, "
                    "best_behaviour_at_4x, min_ratio, status FROM v_technique_scoreboard "
                    "WHERE technique='T1 ridge' AND kind='mlp_width'")[0]
        self.assertEqual(sb, (2, 1, Decimal("0.8000"), "12/15 (MX-T.C1)", True, 1, Decimal("0.8000"), Decimal("0.25"),
                              "superseded by T5"))
        self.assertEqual(self.q("SELECT n_experiments, n_unresolved, first_source FROM v_findings WHERE key='shard-a:F1'"),
                         [(3, 1, "/x/NOTES.md:3-4")])
        self.assertEqual(t["fold_arm"]["inserted"], "5")

        # a re-run of the unchanged shards writes nothing
        again = self.ingest(foldrec, self.dir)
        for table, counts in self.tally(again).items():
            self.assertEqual((counts["inserted"], counts["updated"]), ("0", "0"), table)
        self.assertEqual(again.written, 0)

        # a shard edit: surplus child rows and stale aliases/links are removed, the arm row stays
        docs = copy.deepcopy(self.docs)
        docs["shard-b"]["experiments"][0]["sources"] = [docs["shard-b"]["experiments"][0]["sources"][0]]
        docs["shard-a"]["experiments"][0]["aliases"] = ["c1-short"]
        write_shards(self.dir, docs)
        try:
            third = self.ingest(foldrec, self.dir)
        finally:
            write_shards(self.dir, self.docs)
        self.assertEqual(self.q("SELECT count(*) FROM fold_source fs JOIN experiment e ON e.id=fs.experiment_id "
                                "WHERE e.key='MX-T.C1'")[0][0], 1)
        self.assertEqual(self.q("SELECT count(*) FROM experiment_alias WHERE alias='NF-02'")[0][0], 0)
        self.assertEqual(self.q("SELECT count(*) FROM experiment_link l JOIN experiment f ON f.id=l.from_id "
                                "WHERE f.key='NF-02'")[0][0], 0)
        self.assertTrue(any(line.startswith("deleted ") for line in third.tally_lines), third.tally_lines)

    def test_30_invalid_shard_refuses_and_marks_batch_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.json").write_text(json.dumps({"schema": "foldrec-v0", "shard": "x"}), encoding="utf-8")
            with self.assertRaises(db.LabError):
                self.ingest(foldrec, tmp)
        self.assertEqual(self.q("SELECT status FROM ingest_batch WHERE source='foldrec' ORDER BY id DESC LIMIT 1"),
                         [("failed",)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
