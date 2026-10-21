#!/usr/bin/env python3
"""Tests for the models/runs ingesters (hardware, tool_pins, lanes, oc_ledger, model_scan, teacher_caches,
trainer_runs, status_runs, fold_receipts).

Pure tests check the line grammars against real status-line shapes.  The DB test builds a small fixture
tree that mirrors the server layout ($OC, $L, models), points the ingesters at it through
OMNI_LAB_OC / OMNI_LAB_LEXAR / OMNI_LAB_MODELS / OMNI_LAB_NF_LAB, runs every ingester into a
throwaway omni_lab_test_<hex> database TWICE, and checks values and that the second pass changes no
row counts (idempotent).  It never touches the real omni_lab database or the real source files.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import omni_lab_db as db  # noqa: E402
from ingest import common as C  # noqa: E402
from ingest import (fold_receipts, hardware, lanes, model_scan, oc_ledger, status_runs, teacher_caches,  # noqa: E402
                    tool_pins, trainer_runs)

TEST_DB_RE = re.compile(r"^omni_lab_test_[0-9a-f]{8}$")


def _have_rw() -> bool:
    try:
        db.db_params("rw")
        return True
    except db.LabError:
        return False


class GrammarTests(unittest.TestCase):
    def test_line_state_and_job(self):
        cases = [
            ("job=j6-vl gpu=6 state=started runner=\"j6_vl.sh 1.1.0\"", "start", "j6-vl"),
            ("job=j6-vl-first gpu=6 state=failed rc=1", "fail", "j6-vl-first"),
            ("lane=p2p11 job=p2-v133 gpu=0 state=refused reason=\"D1 container\"", "refused", "p2-v133"),
            ("lane=p2p11 job=p11-v110 gpu=4 state=end rc=0", "end", "p11-v110"),
            ("queue START queue_gpu6.sh 1.0.0 k3tool=k3_teacher_cache 1.2.1", "start", "_queue"),
            ("queue END queue_gpu6.sh 1.5.0", "end", "_queue"),
            ("b2 WAIT S8 pid 917640 + GPU6 < 2000 MiB", "wait", "b2"),
            ("d2clean gpu2 ARMED d2clean_recache 1.0.1 pid 2029331 class p40 kernel torch", "start", "d2clean/gpu2"),
            ("d2clean gpu1 DONE: 12/12 clean caches sealed", "pass", "d2clean/gpu1"),
            ("d1_seal step=A0-wait state=waiting D1 STATE running: unit active", "wait", "A0-wait"),
            ("d3/tests start log=/home/cereal/omni-campaign/distill/d3/logs/tests.log", "start", "d3/tests"),
            ("DONE: /mnt/lexar/omni-campaign/research/fold/x3-08b/SUMMARY.md", "pass", "_queue"),
        ]
        for text, state, job in cases:
            with self.subTest(text=text):
                self.assertEqual(status_runs.line_state(text), state)
                self.assertEqual(status_runs.line_job(text), job)

    def test_pairing_superseded_open_and_orphan_terminal(self):
        evs = [{"job": "a", "state": "start", "text": "a1"}, {"job": "a", "state": "start", "text": "a2"},
               {"job": "a", "state": "pass", "text": "a3"}, {"job": "b", "state": "start", "text": "b1"},
               {"job": "c", "state": "refused", "text": "c1"}, {"job": "b", "state": "wait", "text": "b2"}]
        runs = status_runs.pair_runs(evs)
        got = sorted((r["job"], r["status"], len(r["events"])) for r in runs)
        self.assertEqual(got, [("a", "pass", 2), ("a", "superseded", 1), ("b", "unknown", 2), ("c", "refused", 1)])

    def test_nvidia_smi_parse(self):
        text = ("0, GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561, NVIDIA GeForce RTX 3090, 8.6, 24576 MiB, 580.178.04\n"
                "6, GPU-f5a09535-fc85-a027-50f7-734b344747dd, Quadro RTX 8000, 7.5, 46080 MiB, 580.178.04\n")
        cards = hardware.parse_nvidia_smi(text)
        self.assertEqual([(c["index"], c["sm"], c["vram_mib"]) for c in cards], [(0, "8.6", 24576), (6, "7.5", 46080)])

    def test_step_metrics_and_exit_status(self):
        rows = C.step_metrics({"step": 7, "loss": 1.5, "lr": 1e-5, "route": "joint_av", "applied": True})
        self.assertIn(("loss", 7, 1.5), rows)
        self.assertIn(("loss/joint_av", 7, 1.5), rows)
        self.assertNotIn("applied", [r[0] for r in rows])
        self.assertEqual(trainer_runs.exit_status(0, ""), "pass")
        self.assertEqual(trainer_runs.exit_status(2, "REFUSED: x"), "refused")
        self.assertEqual(trainer_runs.exit_status(143, ""), "stopped")
        self.assertEqual(trainer_runs.exit_status(1, "Traceback"), "fail")

    def test_tool_file_prefix_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            cur = Path(tmp, "trainer.py")
            cur.write_text("TOOL_VERSION = '1.0.1'")
            old = Path(tmp, "_superseded", "trainer_1.0.0.py")
            old.parent.mkdir()
            old.write_text("TOOL_VERSION = '1.0.0'")
            old_sha = db.file_sha256(old)
            found = C._tool_file_with_prefix(cur.as_posix(), old_sha[:16])
            self.assertEqual((found[1], found[2]), (old_sha, "1.0.0"))
            self.assertEqual(C._tool_file_with_prefix(cur.as_posix(), db.file_sha256(cur)[:16])[2], None)
            self.assertIsNone(C._tool_file_with_prefix(cur.as_posix(), "0" * 16))

    def test_classify(self):
        self.assertEqual(model_scan.classify("/mnt/lexar/omni-campaign/distill/fullkd/run1/out/student")[:3],
                         ("student", "chimera-omni", "fullkd-run1"))
        self.assertEqual(model_scan.classify("/home/cereal/omni-campaign/k3-run-oct09-claude/x")[1], "k3")


# ---------------------------------------------------------------------------- fixture tree

def _safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = C.ST_DTYPE_BYTES[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


UUID0 = "GPU-00000000-0000-0000-0000-000000000000"
UUID4 = "GPU-44444444-4444-4444-4444-444444444444"
UUID6 = "GPU-66666666-6666-6666-6666-666666666666"


def build_fixture(base: Path) -> dict:
    oc, lx, models = base / "oc", base / "lexar", base / "models"
    parent_dir = models / "Qwen3.8-27B-Uncensored"
    _safetensors(parent_dir / "model.safetensors", {"model.embed.weight": ("BF16", [4, 8])})
    _w(parent_dir / "config.json", json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
                                               "text_config": {"num_hidden_layers": 4, "layer_types": ["linear_attention"] * 3 + ["full_attention"]},
                                               "vision_config": {}}))
    child_dir = lx / "omni1" / "OC02-F-Q38-27BU-OMNI1"
    _safetensors(child_dir / "model-00001-of-00001.safetensors", {"blocks.0.mlp.w": ("BF16", [2, 8])})
    _w(child_dir / "config.json", json.dumps({"chimera_commit": "ef1ffb9", "config": {"architecture": "dense_decoder_native_omnimodal_encoder_free",
                                                                                    "block_pattern": ["kda", "gated_gqa", "charted"]}}))
    _w(child_dir / "composite_manifest.json", json.dumps({"sources_all": {"names": ["Qwen3.8-27B-Uncensored", "Kimi K3", "ACE-Step"]}}))
    clone_dir = lx / "clones" / "OC02-F-Q38-27BU-OMNI1.clone-pre-D1"
    _safetensors(clone_dir / "model.safetensors", {"blocks.0.mlp.w": ("BF16", [2, 8])})
    ledger = [
        {"checkpoint_id": "PARENT-Q38-27BU", "record_seq": 1, "event": "register", "stage": "parent", "parent_code": "Q38-27BU",
         "directory": str(parent_dir).replace("\\", "/"), "parameter_count": 32, "sealed_at": "2026-09-02T15:31:18.337Z",
         "status": "candidate"},
        {"checkpoint_id": "OC02-F-Q38-27BU-OMNI1", "record_seq": 1, "event": "register", "stage": "fold",
         "parent_checkpoint_id": "PARENT-Q38-27BU", "parent_code": "Q38-27BU", "directory": str(child_dir).replace("\\", "/"),
         "parameter_count": 16, "sealed_at": "2026-10-13T05:01:45.445Z", "notes": "OMNI1 composite", "status": "candidate"},
        {"checkpoint_id": "OC02-F-Q38-27BU-OMNI1", "record_seq": 2, "event": "review", "stage": "fold",
         "parent_checkpoint_id": "PARENT-Q38-27BU", "parent_code": "Q38-27BU", "directory": str(child_dir).replace("\\", "/"),
         "column_a": 5, "column_b": 0, "canonical_signal": 5, "sealed_verdict": "alive-degraded",
         "sealed_at": "2026-10-13T08:55:34.869Z", "status": "partially_capable"},
        {"checkpoint_id": "OC02-F-Q38-27BU-OMNI1.clone-pre-D1", "record_seq": 1, "event": "clone", "stage": "clone",
         "parent_checkpoint_id": "OC02-F-Q38-27BU-OMNI1", "clone_source_id": "OC02-F-Q38-27BU-OMNI1", "parent_code": "Q38-27BU",
         "directory": str(clone_dir).replace("\\", "/"), "sealed_at": "2026-10-13T20:02:58Z", "status": "cloned"},
    ]
    _w(oc / "ledger.jsonl", "".join(json.dumps(r) + "\n" for r in ledger))
    _w(oc / "registry" / "OC02-F-Q38-27BU-OMNI1" / "OC02-F-Q38-27BU-OMNI1-0002-review.json",
       json.dumps({**ledger[2], "review": {"full": True}}))
    trainer_sha = "ad3d18a9a12cd148" + "0" * 48
    _w(oc / "distill" / "tools" / "TOOL_PINS.json", json.dumps({
        "schema": "omni-campaign.tool-pins.v1", "root": str(oc / "distill").replace("\\", "/"),
        "tools": {"streamed_full_kd_trainer.py": {"name": "streamed_full_kd_trainer", "path": "tools/streamed_full_kd_trainer.py",
                                                  "sha256": trainer_sha, "version": "1.0.1", "mtime_utc": "2026-10-18T17:00:00Z",
                                                  "superseded": [{"path": "tools/_superseded/x.py", "sha256": "c94b8aba8c58f3cf" + "1" * 48,
                                                                  "version": "1.0.0"}]},
                  "k3_teacher_cache.py": {"name": "k3_teacher_cache", "path": "tools/k3_teacher_cache.py",
                                          "sha256": "434ced6d7e8e22e4" + "2" * 48, "version": "1.3.0"}}}))
    _w(oc / "distill" / "queues" / "LANES.json", json.dumps({"schema": "omni-campaign.lanes.v1", "lanes": {
        "L-K3": {"lane": "L-K3", "owner": "claude", "status": "ended", "lock_paths": ["/q/gpu6.lock"]},
        "L-TRAIN": {"lane": "L-TRAIN", "owner": "claude", "status": "running"},
        "L-D1": {"lane": "L-D1", "owner": "claude", "status": "ended"}}}))
    _w(oc / "distill" / "queues" / "gpu6.status",
       "2026-10-13T08:07:07Z queue START queue_gpu6.sh 1.0.0 k3tool=k3_teacher_cache 1.2.1 sha256=7899023cb8eb411c start_at=b2\n"
       "2026-10-13T08:07:07Z b2 WAIT S8 pid 917640 + GPU6 < 2000 MiB\n"
       "2026-10-13T09:00:00Z b2 START k3_teacher_cache 1.3.0 on GPU6\n"
       "2026-10-13T10:00:00Z b2 FAIL parity\n"
       "2026-10-19T08:21:43Z queue END queue_gpu6.sh 1.5.0\n")
    _w(oc / "distill" / "queues" / "LANES.go", "not a status file\n")
    # full-KD run1: two segments, a refusal then a completion
    fk = lx / "distill" / "fullkd" / "run1"
    _w(fk / "queue.status",
       "2026-10-18T04:05:51Z queue start: GPU4, data /x/fullkd_data_v3.json (65849268624ff79b), tokens 20000000, micro 8, lr 1e-5, trainer ad3d18a9a12cd148\n"
       "2026-10-18T04:05:51Z gpu4.lock held\n"
       f"2026-10-18T04:05:51Z segment 1 start, log {(fk / 'train-001.log').as_posix()}\n"
       "2026-10-18T17:12:23Z segment 1 exit 143: \n"
       f"2026-10-19T08:16:35Z segment 2 start, log {(fk / 'train-002.log').as_posix()}\n"
       f"2026-10-19T21:16:24Z segment 2 exit 0: TRAIN COMPLETED at step 3; receipt {(fk / 'receipt-002.json').as_posix()} sha256 9e6c\n"
       "2026-10-19T21:16:24Z DONE: student in /x/out/student\n")
    step = lambda s, loss: "STEP " + json.dumps({"step": s, "loss": loss, "lr": 1e-5, "tok_per_s": 200.0, "applied": True})  # noqa: E731
    _w(fk / "train-001.log", "[02:00:00] host weights\n" + step(1, 0.9) + "\n" + step(2, 0.8) + "\n")
    _w(fk / "train-002.log", step(2, 0.7) + "\n" + step(3, 0.6) + "\nTRAIN COMPLETED\n")
    _w(fk / "receipt-002.json", json.dumps({
        "schema": "neural-forge-omni-full-kd-trainer-v1", "tool": "streamed_full_kd_trainer", "tool_version": "1.0.1",
        "verdict": "completed", "exit_code": 0, "step": 3, "tokens_seen": 98304, "tokens_per_second_this_run": 218.43,
        "wall_seconds": 45004.8, "device": "cuda:0", "dtype": "bfloat16", "torch": "2.11.0+cu128",
        "kernels": {"chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk", "mode": "auto"},
        "argv": ["train", "--p1-chimera", str(clone_dir).replace("\\", "/"), "--data-config", "/x/fullkd_data_v3.json"],
        "optimizer": {"kind": "adafactor-no-momentum", "lr": 1e-5}}))
    _safetensors(fk / "out" / "student" / "model.safetensors", {"model.language_model.w": ("BF16", [4, 4]), "lm_head.weight": ("BF16", [4, 4])})
    _w(fk / "out" / "student" / "config.json", json.dumps({"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text"}))
    # D2 short train stopped by SIGTERM
    d2log = lx / "distill" / "d2" / "logs" / "train_short.log"
    _w(oc / "distill" / "queues" / "d2.status",
       "2026-10-17T14:47:55Z d2_launch-1.3.2 queue start pid=3380135 base=OC02-F-Q38-27BU-OMNI1.clone-pre-D1 trainer=ac45af12995c4997\n"
       f"2026-10-17T14:57:37Z d2_launch-1.3.2 cards 0 4 taken (CUDA_VISIBLE_DEVICES={UUID0},{UUID4}), MemAvailable 234 GiB\n"
       f"2026-10-17T14:57:38Z d2_launch-1.3.2 train short start log={d2log.as_posix()}\n")
    _w(d2log, step(1, 0.95) + "\nTraceback\nSignalException: Process 3392885 got signal: 15\n")
    # teacher cache: one sealed qtext version
    tc = lx / "teacher_cache" / "qtext" / "q38-27bu" / ("7b73" + "0" * 60)
    _w(tc / "manifest.json", json.dumps({
        "schema": "omni-distill-teacher-cache-manifest-v1", "sealed_utc": "2026-10-13T12:24:08Z", "bytes_total": 399556872,
        "positions_total": 1008982, "forward_tokens_total": 1145085, "seal_wall_seconds_this_session": 10170.17,
        "tool": {"name": "qtext_teacher_cache", "version": "1.0.5", "sha256": "f2465a11b7a2242f" + "3" * 48},
        "sessions": [{"after": {"model_dir": str(parent_dir).replace("\\", "/")}}], "shards": [{"i": 0}, {"i": 1}],
        "started_utc": "2026-10-13T09:35:00Z"}))
    # a fold receipt
    _w(oc / "k3-run-oct09-claude" / "receipts" / "k3_dense_pass_receipt.json", json.dumps({
        "schema": "omni-campaign.k3_dense_pass.v1", "tool": "k3_dense_pass", "tool_version": "1.0.0",
        "verdict": "sealed", "started_utc": "2026-10-09T01:00:00Z", "finished_utc": "2026-10-09T05:00:00Z",
        "layers": {str(i): {"x": "y" * 100} for i in range(60)}}))
    # a content-addressed cache entry receipt: must NOT become a run
    _w(base / "nf" / "artifacts" / "neural_forge_phase_cache_v1" / "lfm2-x" / "receipt.json", json.dumps({"verdict": "ok"}))
    return {"oc": oc, "lx": lx, "models": models, "nf": base / "nf", "child_dir": child_dir, "clone_dir": clone_dir}


@unittest.skipUnless(os.name == "posix", "the ingesters read the AI-server (POSIX) layout; run this class on the server")
@unittest.skipUnless(_have_rw(), "no omni_lab credential resolvable (OMNI_LAB_PG_PASSWORD[_COMMAND|_FILE])")
class IngestDbTests(unittest.TestCase):
    SOURCES = (tool_pins, lanes, oc_ledger, model_scan, teacher_caches, trainer_runs, status_runs, fold_receipts)

    @classmethod
    def setUpClass(cls):
        from psycopg2 import sql
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fx = build_fixture(Path(cls.tmp.name))
        cls.saved_env = {k: os.environ.get(k) for k in ("OMNI_LAB_OC", "OMNI_LAB_LEXAR", "OMNI_LAB_MODELS", "OMNI_LAB_NF_LAB")}
        os.environ.update({"OMNI_LAB_OC": cls.fx["oc"].as_posix(), "OMNI_LAB_LEXAR": cls.fx["lx"].as_posix(),
                           "OMNI_LAB_MODELS": cls.fx["models"].as_posix(), "OMNI_LAB_NF_LAB": cls.fx["nf"].as_posix()})
        C.roots_from_env()
        cls.dbname = f"omni_lab_test_{secrets.token_hex(4)}"
        assert TEST_DB_RE.match(cls.dbname)
        admin = db.connect("rw", app="test-create", autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(sql.Identifier(cls.dbname)))
        finally:
            admin.close()
        cls.conn = db.connect("rw", dbname=cls.dbname, app="test-ingest")
        db.migrate(cls.conn)
        hid = C.host_id(cls.conn)
        for idx, uuid, name, sm in ((0, UUID0, "NVIDIA GeForce RTX 3090", "8.6"), (4, UUID4, "NVIDIA GeForce RTX 3090", "8.6"),
                                    (6, UUID6, "Quadro RTX 8000", "7.5")):
            db.upsert_gpu(cls.conn, hid, idx, uuid, name, sm, 24576)
        cls.conn.commit()
        cls.batches1 = [C.run_source(m, None, conn_factory=lambda: cls.conn) for m in cls.SOURCES]
        cls.counts1 = dict(db.table_counts(cls.conn))
        cls.batches2 = [C.run_source(m, None, conn_factory=lambda: cls.conn) for m in cls.SOURCES]
        cls.counts2 = dict(db.table_counts(cls.conn))

    @classmethod
    def tearDownClass(cls):
        from psycopg2 import sql
        cls.conn.close()
        for k, v in cls.saved_env.items():
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

    def q(self, query, params=None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def test_00_idempotent_second_pass(self):
        volatile = {"ingest_batch"}
        diff = {t: (self.counts1[t], self.counts2[t]) for t in self.counts1
                if t not in volatile and self.counts1[t] != self.counts2[t]}
        self.assertEqual(diff, {})
        self.assertEqual(self.counts2["ingest_batch"], 2 * len(self.SOURCES))
        self.assertEqual(self.q("SELECT count(*) FROM ingest_batch WHERE status <> 'ok'")[0][0], 0)

    def test_01_ledger_models_lineage_events(self):
        rows = dict(self.q("SELECT model_key, role FROM model WHERE ledger_checkpoint_id IS NOT NULL"))
        self.assertEqual(rows, {"PARENT-Q38-27BU": "parent", "OC02-F-Q38-27BU-OMNI1": "composite",
                                "OC02-F-Q38-27BU-OMNI1.clone-pre-D1": "clone"})
        self.assertEqual(self.q("SELECT stage, sealed_verdict, layout FROM model WHERE model_key = 'OC02-F-Q38-27BU-OMNI1'")[0],
                         ("OMNI1", "alive-degraded", "chimera.checkpoint.v1"))
        rel = self.q("SELECT p.model_key, l.relation, l.ordinal FROM model_lineage l JOIN model c ON c.id = l.child_id "
                     "JOIN model p ON p.id = l.parent_id WHERE c.model_key = 'OC02-F-Q38-27BU-OMNI1' ORDER BY l.relation, l.ordinal")
        self.assertEqual(rel, [("PARENT-Q38-27BU", "fold", 0), ("PARENT-Q38-27BU", "source", 0),
                               ("source:Kimi-K3", "source", 1), ("source:ACE-Step", "source", 2)])
        ev = self.q("SELECT record_seq, event, column_a, src_path LIKE '%%registry%%', record ? 'review' FROM ledger_event "
                    "WHERE checkpoint_id = 'OC02-F-Q38-27BU-OMNI1' ORDER BY record_seq")
        self.assertEqual(ev, [(1, "register", None, False, False), (2, "review", 5, True, True)])
        m = dict(self.q("SELECT mt.name, mt.value FROM metric mt JOIN run r ON r.id = mt.run_id "
                        "WHERE r.run_key = 'ledger:OC02-F-Q38-27BU-OMNI1:0002:review'"))
        self.assertEqual((m["column_a"], m["column_b"], m["canonical_signal"]), (5.0, 0.0, 5.0))

    def test_02_scan_keeps_ledger_rows_and_adds_census(self):
        r = self.q("SELECT model_key, size_bytes > 0, config->'scan' IS NOT NULL FROM model WHERE path = %s",
                   (self.fx["child_dir"].as_posix(),))
        self.assertEqual(r, [("OC02-F-Q38-27BU-OMNI1", True, True)])
        self.assertEqual(self.q("SELECT count(*) FROM model WHERE path LIKE %s", ("%teacher_cache%",))[0][0], 0)

    def test_03_fullkd_parent_segments_curve_student(self):
        p = self.q("SELECT r.id, r.status, r.kind, tv.version, m.role, m.layout, m.stage FROM run r "
                   "LEFT JOIN tool_version tv ON tv.id = r.tool_version_id LEFT JOIN model m ON m.id = r.model_id "
                   "WHERE r.run_key = 'distill:fullkd:run1'")[0]
        self.assertEqual(p[1:], ("pass", "distill", "1.0.1", "student", "qwen3_5", "fullkd-run1"))
        segs = self.q("SELECT run_key, status, exit_code FROM run WHERE parent_run_id = %s ORDER BY run_key", (p[0],))
        self.assertEqual(segs, [("distill:fullkd:run1:seg1", "stopped", 143), ("distill:fullkd:run1:seg2", "pass", 0)])
        curve = self.q("SELECT step, value FROM metric WHERE run_id = %s AND name = 'loss' ORDER BY step", (p[0],))
        self.assertEqual(curve, [(1, 0.9), (2, 0.7), (3, 0.6)])          # step 2 replayed: the later segment wins
        dev = self.q("SELECT g.gpu_index, rd.kernel_path, rd.detail->>'mapping' FROM run_device rd JOIN gpu g ON g.id = rd.gpu_id "
                     "WHERE rd.run_id = %s", (p[0],))
        self.assertEqual(dev, [(4, "fla.ops.gated_delta_rule.chunk", "index-from-status")])
        lin = self.q("SELECT p.model_key, l.relation FROM model_lineage l JOIN model c ON c.id = l.child_id "
                     "JOIN model p ON p.id = l.parent_id WHERE c.stage = 'fullkd-run1'")
        self.assertEqual(lin, [("OC02-F-Q38-27BU-OMNI1.clone-pre-D1", "distill")])
        self.assertEqual(self.q("SELECT value FROM metric WHERE run_id = %s AND name = 'tokens_seen'", (p[0],))[0][0], 98304.0)

    def test_04_d2_stopped_with_uuid_devices(self):
        r = self.q("SELECT r.id, r.status, r.notes LIKE '%%SIGTERM%%' FROM run r WHERE r.run_key LIKE 'distill:d2:train_short:%%'")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][1:], ("stopped", True))
        idx = [x[0] for x in self.q("SELECT g.gpu_index FROM run_device rd JOIN gpu g ON g.id = rd.gpu_id WHERE rd.run_id = %s "
                                    "ORDER BY 1", (r[0][0],))]
        self.assertEqual(idx, [0, 4])
        base = self.q("SELECT m.model_key FROM run_model rm JOIN model m ON m.id = rm.model_id WHERE rm.run_id = %s AND rm.role = 'base'",
                      (r[0][0],))
        self.assertEqual(base, [("OC02-F-Q38-27BU-OMNI1.clone-pre-D1",)])

    def test_05_status_runs_pairing_and_events(self):
        runs = self.q("SELECT config->>'job', status FROM run WHERE run_key LIKE 'status:$OC/distill/queues/gpu6.status:%%' "
                      "ORDER BY started_at, run_key")
        self.assertEqual(runs, [("_queue", "pass"), ("b2", "fail")])
        self.assertEqual(self.q("SELECT count(*) FROM run_event WHERE src_path LIKE '%%gpu6.status'")[0][0], 5)
        self.assertEqual(self.q("SELECT count(*) FROM run WHERE run_key LIKE 'status:%%fullkd%%' OR run_key LIKE 'status:%%d2.status%%'")[0][0], 0)
        lane = self.q("SELECT l.key FROM run r JOIN lane l ON l.id = r.lane_id WHERE r.run_key LIKE 'status:%%gpu6.status:b2:%%'")
        self.assertEqual(lane, [("L-K3",)])

    def test_06_teacher_cache_dataset_and_run(self):
        dv = self.q("SELECT d.key, dv.sealed, dv.item_count, dv.size_bytes, dv.splits->>'shards' FROM dataset_version dv "
                    "JOIN dataset d ON d.id = dv.dataset_id")
        self.assertEqual(dv, [("teacher_cache/qtext/q38-27bu", True, 1008982, 399556872, "2")])
        r = self.q("SELECT r.status, r.wall_seconds, m.model_key, rd.role FROM run r JOIN run_dataset rd ON rd.run_id = r.id "
                   "LEFT JOIN model m ON m.id = r.model_id WHERE r.kind = 'cache' AND r.run_key LIKE 'cache:%%'")
        self.assertEqual(r, [("pass", 10170.17, "PARENT-Q38-27BU", "output")])

    def test_07_tool_pins_and_fold_receipt(self):
        tv = self.q("SELECT version, pinned, superseded FROM tool_version tv JOIN tool t ON t.id = tv.tool_id "
                    "WHERE t.name = 'streamed_full_kd_trainer' ORDER BY version")
        self.assertEqual(tv, [("1.0.0", False, True), ("1.0.1", True, False)])
        r = self.q("SELECT kind, project, status, wall_seconds, config->'layers' ? '_omitted_bytes' FROM run "
                   "WHERE run_key LIKE 'receipt:%%k3_dense_pass_receipt.json'")
        self.assertEqual(r, [("fold", "k3", "pass", 14400.0, True)])
        self.assertEqual(self.q("SELECT count(*) FROM run WHERE run_key LIKE '%%phase_cache%%'")[0][0], 0)

    def test_08_deleted_rows_survive_rescan(self):
        """A cleanup-recorded deletion stays 'deleted' through oc_ledger + model_scan re-runs (0006 guard)."""
        clone = str(self.fx["clone_dir"]).replace("\\", "/")
        rows = self.q("SELECT id, model_key FROM model WHERE path = %s", (clone,))
        self.assertTrue(rows, clone)
        # the clone row is ledger-owned (oc_ledger upsert) and its directory exists, so model_scan revisits it
        # through the existing-row UPDATE; both writers must leave the recorded deletion alone
        ids = [r[0] for r in rows]
        with self.conn.cursor() as cur:
            cur.execute("UPDATE model SET status = 'deleted', deleted_at = now(), delete_reason = 'test' "
                        "WHERE id = ANY(%s)", (ids,))
        self.conn.commit()
        try:
            for m in (oc_ledger, model_scan):
                C.run_source(m, None, conn_factory=lambda: self.conn)
            self.assertEqual(self.q("SELECT DISTINCT status FROM model WHERE id = ANY(%s)", (ids,)), [("deleted",)])
        finally:
            with self.conn.cursor() as cur:
                cur.execute("UPDATE model SET status = 'present', deleted_at = NULL, delete_reason = NULL "
                            "WHERE id = ANY(%s)", (ids,))
            self.conn.commit()

if __name__ == "__main__":
    unittest.main()
