#!/usr/bin/env python3
"""Tests for the knowledge ingesters: directives, rulings, documents, corpora, foldhist.

    C:/Python314/python.exe -m unittest tests.test_ingest_knowledge -v

Pure tests check the parsers on fixtures.  DB tests create a throwaway omni_lab_test_<hex> database (as
test_schema does), run each ingester through ingest.common.run_source, check rows against the source, and
run it a second time to prove it is idempotent (no row inserted or updated).  directives/documents/foldhist
read the real inventory snapshot in inventory/2026-10-21; rulings/corpora read fixture trees under a temp
$OMNI_LAB_OC / $OMNI_LAB_LEXAR, so the tests never touch the AI server's files.
"""
from __future__ import annotations

import json
import os
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
from ingest import corpora, directives, documents, foldhist, rulings  # noqa: E402

TEST_DB_RE = re.compile(r"^omni_lab_test_[0-9a-f]{8}$")
INV = ROOT / "inventory" / "2026-10-21"

RULINGS_FIXTURE = """# Stage D rulings and decisions, 2026-10-12

## Cyril's words (direct, interactive, 2026-10-12)

1. "pick whichever option gives the best result and take what you need"
2. "we can run the second pass we're fine we're fine on all the paperwork"

Prior rulings that still hold:
- 2026-08-28: the H3 US-licence restriction "doesn't matter, we're not shipping anything".

## What these rulings settle

| Item | Decision | Basis |
|---|---|---|
| K3 budget (D-R5-6) | **2.0M tokens** in four passes. | best model |

## Later rulings (dated, appended)

- **2026-10-13T14:26:34Z, ruling `cyril-h3-thermal-withdrawn-2026-10-13T14:26:34Z`.** Cyril to Codex, verbatim: "ignore the old heat check.... drop it from your notes then proceed". Recorded in `$OC/distill/queues/gpu6_h3_thermal_ruling.txt`.
- **2026-10-14, ruling `k3-b2-rebaseline-2026-10-14`** (Claude, lane gpu6, under the standing GO; not Cyril's words). The K3 parity reference changes.
  - continuation line that belongs to the bullet

### Cyril's words and approved realignment, 2026-10-17 (direct)

- **`cyril-full-distill-no-lora-2026-10-17`.** His words: "training on the small copy is pointless ... I don't want patch training I want full training no patches".
  - Every stage trains all weights.
- **`k3-b2-bf16-numerics-provisional-2026-10-17`.** Approved by Cyril 15:27Z ("yes keep all of the cards busy"). The text is in `$Q/k3_b2_numerics_ruling.txt`.
- **Plan of record:** `reports/omni-campaign-2026-09/REALIGNMENT_2026-10-17.md`.
"""


class PureTests(unittest.TestCase):
    def test_directive_keys_and_channels(self):
        ev = directives.classify({"source": "agent_memory_events.id=506681 (event_type=user_prompt, agent=Codex, "
                                            "session=x)", "quote": "fold 99% kimi k3"})
        self.assertEqual((ev["channel"], ev["event_id"]), ("codex", 506681))
        self.assertTrue(ev["key"].startswith("ev:506681:"))
        ev2 = directives.classify({"source": "agent_memory_events.id=506681 (event_type=user_prompt, agent=Codex, "
                                             "session=x)", "quote": "other words"})
        self.assertNotEqual(ev["key"], ev2["key"])            # two directives in one prompt stay two rows
        cc = directives.classify({"source": "C:/x/7d20c05f-b18d-4e59-b0b3-81366f801452.jsonl:1006 (queued mid-turn "
                                            "prompt; NOT in agent_memory_events)", "quote": "q"})
        self.assertEqual(cc["channel"], "transcript")
        self.assertTrue(cc["key"].startswith("cc:7d20c05f:1006:"))
        doc = directives.classify({"source": "C:/x/memory/a.md:11", "quote": "q"})
        self.assertEqual((doc["channel"], doc["doc_line"]), ("doc", 11))
        dot = directives.classify({"source": "agent_memory_events.id=1 (event_type=user_prompt, agent=dot, "
                                             "session=None)", "quote": "q"})
        self.assertEqual(dot["channel"], "dot")
        with self.assertRaises(db.LabError):
            directives.classify({"source": "somewhere else", "quote": "q"})

    def test_directive_anchor_restores_source_whitespace(self):
        src = "first line\n\nso keep   going: make it dense\nno moe at all"
        self.assertEqual(directives.anchor("make it dense", src), "make it dense")
        self.assertEqual(directives.anchor("keep going: make it dense", src), "keep   going: make it dense")
        self.assertEqual(directives.anchor("dense no moe at all", src), "dense\nno moe at all")
        self.assertIsNone(directives.anchor("towers", src))
        self.assertIsNone(directives.anchor("x", None))
        self.assertEqual(directives.json_strings({"a": ["x", {"b": "y"}], "n": 3}), ["x", "y"])

    def test_directive_date_from_doc(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "notes.md")
            p.write_text("# heading 2026-10-12\n\nline\n1. \"words\"\n", encoding="utf-8")
            d, where = directives.date_from_doc(p.as_posix(), 4)
            self.assertEqual(d.isoformat(), "2026-10-12")
            self.assertEqual(where, "notes.md:1")
            self.assertIsNone(directives.date_from_doc(Path(tmp, "missing.md").as_posix(), 1))

    def test_rulings_parser(self):
        recs = {r["key"]: r for r in rulings.parse_rulings_md(RULINGS_FIXTURE)}
        self.assertEqual(recs["OWNER-WORDS-2026-10-12-1"]["authority"], "owner")
        prior = [k for k in recs if k.startswith("OWNER-PRIOR-2026-08-28")]
        self.assertEqual(len(prior), 1)
        self.assertEqual(recs[prior[0]]["verbatim"], "doesn't matter, we're not shipping anything")
        self.assertEqual(recs["STAGE-D-2026-10-12/k3-budget"]["authority"], "agent")
        self.assertIn("2.0M tokens", recs["STAGE-D-2026-10-12/k3-budget"]["decision"])
        th = recs["cyril-h3-thermal-withdrawn-2026-10-13T14:26:34Z"]
        self.assertEqual((th["who"], th["authority"]), ("cyril", "owner"))
        self.assertEqual(th["decided_at"].isoformat(), "2026-10-13T14:26:34+00:00")
        self.assertEqual(th["refs"], ["$OC/distill/queues/gpu6_h3_thermal_ruling.txt"])
        lead = recs["k3-b2-rebaseline-2026-10-14"]
        self.assertEqual((lead["who"], lead["authority"], lead["verbatim"]), ("claude-lead", "lead", None))
        self.assertIn("continuation line", lead["decision"])
        own = recs["cyril-full-distill-no-lora-2026-10-17"]
        self.assertEqual(own["authority"], "owner")
        self.assertEqual(own["decided_at"].date().isoformat(), "2026-10-17")
        appr = recs["k3-b2-bf16-numerics-provisional-2026-10-17"]
        self.assertEqual((appr["who"], appr["authority"], appr["verbatim"]),
                         ("claude-lead", "owner", "yes keep all of the cards busy"))
        self.assertNotIn("plan-of-record", " ".join(recs))
        self.assertEqual(len(recs), 8)

    def test_rulings_directive_match(self):
        index = [(7, rulings.norm("Training on the small copy is pointless ... I don't want patch training"))]
        self.assertEqual(rulings.match_directive(index, "training on the small copy is pointless \u2026 x"), 7)
        self.assertIsNone(rulings.match_directive(index, "yes keep all of the cards busy"))
        self.assertIsNone(rulings.match_directive(index, None))

    def test_foldhist_helpers(self):
        self.assertEqual(foldhist.weakest_evidence("memory+report"), "memory")
        self.assertEqual(foldhist.weakest_evidence("receipt+report"), "report")
        self.assertEqual(foldhist.weakest_evidence("receipt"), "receipt")
        self.assertEqual(foldhist.weakest_evidence(None), "unverified")
        self.assertEqual(foldhist.ratio_body("net 0.951053"), 0.951053)
        self.assertEqual(foldhist.ratio_body("net 0.522729 (14,522,158,320); 0.25 per MLP"), 0.522729)
        self.assertIsNone(foldhist.ratio_body("4.000220x counted, 3.469006x body"))
        self.assertEqual(str(foldhist.entry_date("2026-09-07 build (Codex), registered 2026-10-10")), "2026-09-07")
        self.assertIsNone(foldhist.entry_date("2026-07 (A9/A11); re-scored 2026-08-10"))
        refs = foldhist.model_refs({"id": "OC01-F-Q35-4B-X03", "parent": "PARENT-Q35-4B (4,659,865,088)",
                                    "results": "x", "technique": "t"})
        self.assertEqual((refs["child"], refs["parent"]), ("OC01-F-Q35-4B-X03", "PARENT-Q35-4B"))
        refs = foldhist.model_refs({"id": "S3-C2K4352", "parent": "PARENT-Q38-27BU Qwen3.8",
                                    "results": "SCORECARD.md (OC02-F-Q38-27BU-C2K4352)", "technique": "t"})
        self.assertEqual((refs["child"], refs["parent"]), ("OC02-F-Q38-27BU-C2K4352", "PARENT-Q38-27BU"))

    def test_documents_mapping(self):
        self.assertEqual(documents.host_of("local:reviews"), "workstation")
        self.assertEqual(documents.host_of("server:oc"), "ai")
        self.assertEqual(documents.status_of({"status": "superseded-or-backup",
                                              "path": "/x/distill/queues/backups/a.md"}), "backup")
        self.assertEqual(documents.status_of({"status": "superseded-or-backup",
                                              "path": "/x/tools/_superseded/TOOL_PINS.local-before-x.json"}), "backup")
        self.assertEqual(documents.status_of({"status": "superseded-or-backup", "path": "/x/old/PLAN.md"}), "superseded")
        with self.assertRaises(db.LabError):
            documents.status_of({"status": "weird", "path": "x"})

    def test_corpora_helpers(self):
        doc = {"rows": 1948, "rows_by_route": {"code": 3}, "big": list(range(5000)), "set": "x" * 9000}
        self.assertEqual(corpora.count_of(doc), 1948)
        self.assertEqual(corpora.splits_of(doc), {"rows_by_route": {"code": 3}})
        c = corpora.compact(doc)
        self.assertEqual(c["big"]["_len"], 5000)
        self.assertEqual(len(c["set"]), corpora.SMALL)
        self.assertEqual(corpora.count_of({"vision_in": {"rows": 12}}), 12)
        self.assertIsNone(corpora.count_of({"rows": True}))

    def test_inventory_snapshot_matches_sums(self):
        sums = dict(reversed(line.split("  ", 1)) for line in (INV / "SHA256SUMS").read_text().splitlines() if line)
        for name, sha in sums.items():
            self.assertEqual(db.file_sha256(INV / name), sha, name)


def _have_rw() -> bool:
    try:
        db.db_params("rw")
        return True
    except db.LabError:
        return False


@unittest.skipUnless(_have_rw(), "no omni_lab credential resolvable (OMNI_LAB_PG_PASSWORD[_COMMAND|_FILE])")
class IngestDbTests(unittest.TestCase):
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
        cls.conn = db.connect("rw", dbname=cls.dbname, app="test-ingest-knowledge", statement_timeout_ms=600000)
        db.migrate(cls.conn)
        cls.tmp = tempfile.TemporaryDirectory()
        cls._old_env = {k: os.environ.get(k) for k in ("OMNI_LAB_OC", "OMNI_LAB_LEXAR")}
        os.environ["OMNI_LAB_OC"] = str(Path(cls.tmp.name, "oc"))
        os.environ["OMNI_LAB_LEXAR"] = str(Path(cls.tmp.name, "lx"))
        C.roots_from_env()
        cls._fixture_tree()

    @classmethod
    def tearDownClass(cls):
        from psycopg2 import sql
        cls.conn.close()
        for k, v in cls._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        C.roots_from_env()
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

    @classmethod
    def _fixture_tree(cls):
        oc, lx = Path(os.environ["OMNI_LAB_OC"]), Path(os.environ["OMNI_LAB_LEXAR"])
        q = oc / "distill" / "queues"
        q.mkdir(parents=True)
        (oc / "distill" / "STAGE_D_RULINGS_2026-10-12.md").write_text(RULINGS_FIXTURE, encoding="utf-8")
        (q / "gpu6_h3_thermal_ruling.txt").write_text("cyril-h3-thermal-withdrawn-2026-10-13T14:26:34Z\nquote: x\n")
        (oc / "rulings").mkdir()
        (oc / "rulings" / "OWNER-RULING-2026-10-12-STAGE-D-GO.json").write_text(json.dumps({
            "ruling_id": "OWNER-RULING-2026-10-12-STAGE-D-GO", "kind": "cyril_ruling", "sealed_at": "2026-10-13T00:05:05Z",
            "source": "Cyril, interactive", "sealer": "oc_seal", "attachments": [{"path": "/x", "sha256": "ab"}],
            "provenance": {"user_turn_timestamps_utc": ["2026-10-12T23:55:00.105Z"]},
            "verbatim": ["pick whichever option gives the best result and take what you need",
                         "we can run the second pass we're fine we're fine on all the paperwork"]}))
        corpus = oc / "distill" / "corpus"
        (corpus / "text").mkdir(parents=True)
        (corpus / "text" / "manifest.json").write_text(json.dumps({
            "schema": "x/text", "set": "(a) text", "rows": 1168, "rows_by_route": {"code": 1000, "math": 168},
            "identity": {"built_at_utc": "2026-10-13T01:00:00Z"}, "manifest_sha256": "f" * 64}))
        (corpus / "text" / "rows.bin").write_bytes(b"\0" * 100)
        (corpus / "verify_receipt.json").write_text(json.dumps({
            "ok": True, "receipt_sha256": "9" * 64,
            "manifests": {(corpus / "text" / "manifest.json").as_posix(): {"seal": "s", "recomputed_equal": True}}}))
        hold = lx / "corpus" / "holdouts"
        (hold / "registry_v1").mkdir(parents=True)
        (hold / "HOLDOUT_REGISTRY_v1.json").write_text(json.dumps({
            "version": 1, "created_utc": "2026-10-14T18:51:00Z", "dir": (hold / "registry_v1").as_posix(),
            "files": {"texts": {"rows": 37205, "sha256": "a"}, "media": {"rows": 10, "sha256": "b"}},
            "registry_sha256": "c" * 64}))
        ext = lx / "corpus" / "external" / "librispeech" / "LibriSpeech" / "test-clean" / "84"
        ext.mkdir(parents=True)
        (ext / "a.flac").write_bytes(b"x" * 10)
        (ext / "b.flac").write_bytes(b"y" * 20)
        dv = lx / "corpus" / "derived" / "hy3d_voxel64"
        for n, sealed in ((1, True), (2, False)):
            vd = dv / f"v{n}"
            (vd / "volumes").mkdir(parents=True)
            (vd / "volumes" / "a.npz").write_bytes(b"z" * 50)
            (vd / "manifest.json").write_text(json.dumps({
                "schema": "omni-campaign.raw3d.hy3d-voxel-set.v1", "version": f"v{n}", "items": 18 + n,
                "splits": {"rows_by_split": {"train": 17 + n, "val": 1}}, "built_at_utc": "2026-10-22T12:00:00Z"}))
            if sealed:
                import hashlib as _h
                (vd / "verify_receipt.json").write_text(json.dumps({
                    "ok": True, "manifest_sha256": _h.sha256((vd / "manifest.json").read_bytes()).hexdigest()}))
                (vd / "sealed").write_text("x")
        (dv / ".v3.building").mkdir()
        fk = lx / "distill" / "fullkd"
        fk.mkdir(parents=True)
        (fk / "fullkd_data_v1.json").write_text(json.dumps({"schema": "s", "sources": ["a", "b", "c"]}))

    def setUp(self):
        self.conn.rollback()

    def q(self, query, params=None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def ingest(self, module, **kw):
        return C.run_source(module, SimpleNamespace(inventory=str(INV), **kw), conn_factory=lambda: self.conn)

    def tally(self, batch) -> dict:
        out = {}
        for line in batch.tally_lines:
            table, rest = line.split(": ", 1)
            if "inserted=" not in rest:              # e.g. 'source check: exact=.. reanchored=..'
                continue
            out[table] = dict(kv.split("=") for kv in rest.split())
        return out

    def assert_idempotent(self, module):
        again = self.ingest(module)
        for table, counts in self.tally(again).items():
            self.assertEqual((counts["inserted"], counts["updated"]), ("0", "0"), f"{module.SOURCE}/{table}")
        self.assertEqual(again.written, 0)

    def test_10_directives_and_rulings(self):
        b = self.ingest(directives)
        items = json.loads((INV / "directives.json").read_text(encoding="utf-8"))
        n = self.q("SELECT count(*), count(*) FILTER (WHERE NOT said_at_verified), count(DISTINCT channel) FROM directive")[0]
        self.assertEqual(n[0] + b.skipped, len(items))
        self.assertGreaterEqual(n[0], len(items) - 3)                 # only undated doc rows may skip
        # every stored quote is the inventory quote, or its exact source span (equal up to whitespace runs)
        ws = lambda t: " ".join(t.split())  # noqa: E731
        by_src = {}
        for it in items:
            by_src.setdefault(it["source"], set()).add(ws(it["quote"]))
        for src, verbatim in self.q("SELECT src_path, verbatim FROM directive"):
            self.assertIn(ws(verbatim), by_src[src])
        self.assertTrue(any(line.startswith("source check: ") for line in b.tally_lines))
        ev = self.q("SELECT source_event_id, channel FROM directive WHERE directive_key LIKE 'ev:506681:%'")
        self.assertTrue(ev and all(r == (506681, "codex") for r in ev))
        self.assert_idempotent(directives)

        b = self.ingest(rulings)
        rows = dict(self.q("SELECT ruling_key, authority FROM ruling"))
        self.assertEqual(rows["OWNER-RULING-2026-10-12-STAGE-D-GO"], "owner")
        self.assertNotIn("OWNER-WORDS-2026-10-12-1", rows)            # same words as the sealed JSON ruling
        self.assertEqual(rows["k3-b2-rebaseline-2026-10-14"], "lead")
        self.assertEqual(rows["STAGE-D-2026-10-12/k3-budget"], "agent")
        att = self.q("SELECT attachments FROM ruling WHERE ruling_key = 'cyril-h3-thermal-withdrawn-2026-10-13T14:26:34Z'")[0][0]
        self.assertEqual([Path(a["path"]).name for a in att], ["gpu6_h3_thermal_ruling.txt"])
        linked = self.q("SELECT r.ruling_key, d.channel FROM ruling r JOIN directive d ON d.id = r.directive_id "
                        "WHERE r.ruling_key = 'cyril-full-distill-no-lora-2026-10-17'")
        self.assertEqual(len(linked), 1, "the 10-17 owner words should match the inventory directive")
        self.assert_idempotent(rulings)

    def test_20_documents(self):
        b = self.ingest(documents)
        inv = json.loads((INV / "docs.json").read_text(encoding="utf-8"))["docs"]
        self.assertEqual(b.seen, len(inv))
        (n, top, hosts) = self.q("SELECT count(*), count(rank), count(DISTINCT host) FROM document")[0]
        self.assertEqual((n, top, hosts), (len(inv), 30, 2))
        first = self.q("SELECT path, doc_type, arm, rank FROM document WHERE rank = 1")[0]
        top1 = next(d for d in inv if d.get("top30_rank") == 1)
        self.assertEqual(first, (top1["path"], top1["type"], top1["arm"], 1))
        self.assert_idempotent(documents)

    def test_30_foldhist(self):
        b = self.ingest(foldhist)
        inv = json.loads((INV / "foldhist.json").read_text(encoding="utf-8"))["entries"]
        self.assertEqual(self.q("SELECT count(*) FROM experiment")[0][0], len(inv))
        self.assertEqual(b.skipped, 0)
        x3 = self.q("SELECT kind, evidence_level, date, params->>'ratio_text', behaviour_legacy FROM experiment "
                    "WHERE key = 'X3-08B'")[0]
        e = next(x for x in inv if x["id"] == "X3-08B")
        self.assertEqual(x3, ("fold", "receipt", foldhist.entry_date(e["date"]), e["ratio"], e["behaviour"]))
        # model links fill on a re-run once the model rows exist
        self.assertIsNone(self.q("SELECT child_id FROM experiment WHERE key='OC01-F-Q35-4B-X03'")[0][0])
        mid = db.upsert_model(self.conn, "OC01-F-Q35-4B-X03", "chimera-omni", "fold-child", "hf",
                              ledger_checkpoint_id="OC01-F-Q35-4B-X03")
        self.conn.commit()
        self.ingest(foldhist)
        self.assertEqual(self.q("SELECT child_id FROM experiment WHERE key='OC01-F-Q35-4B-X03'")[0][0], mid)
        self.assert_idempotent(foldhist)

    def test_40_corpora(self):
        old = corpora.INTAKE_ENV
        corpora.INTAKE_ENV = str(Path(self.tmp.name, "no-such-env"))
        try:
            b = self.ingest(corpora)
            self.assertTrue(any("chimera_corpus registry skipped" in n for n in b.notes))
            rows = {k: (v, ic, s, sealed, sz) for k, v, ic, s, sealed, sz in self.q(
                "SELECT d.key, dv.version, dv.item_count, dv.splits, dv.sealed, dv.size_bytes FROM dataset d "
                "JOIN dataset_version dv ON dv.dataset_id = d.id")}
            self.assertEqual(rows["corpus/distill/text"][:4], ("f" * 16, 1168, {"rows_by_route": {"code": 1000, "math": 168}}, True))
            self.assertGreater(rows["corpus/distill/text"][4], 100)
            self.assertEqual(rows["corpus/distill"][0], "9" * 16)
            self.assertEqual(rows["corpus/holdout_registry"][:4], ("v1", 37205, {"rows_by_file": {"texts": 37205, "media": 10}}, True))
            ext = rows["corpus/external/librispeech"]
            self.assertEqual((ext[1], ext[2], ext[4]), (2, {"files_by_split": {"test-clean": 2}}, 30))
            self.assertEqual(rows["training_config/fullkd_data"][:2], ("v1", 3))
            dvs = {v: (ic, sp, se) for k, v, ic, sp, se in self.q(
                "SELECT d.key, dv.version, dv.item_count, dv.splits, dv.sealed FROM dataset d "
                "JOIN dataset_version dv ON dv.dataset_id = d.id WHERE d.key = 'corpus/derived/hy3d_voxel64'")}
            self.assertEqual(dvs, {"v1": (19, {"splits": {"rows_by_split": {"train": 18, "val": 1}}}, True),
                                   "v2": (20, {"splits": {"rows_by_split": {"train": 19, "val": 1}}}, False)})
            self.assert_idempotent(corpora)
        finally:
            corpora.INTAKE_ENV = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
