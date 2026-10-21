#!/usr/bin/env python3
"""Eval ingesters (banks, dossiers, toolcall_bankruns, student_tests, p11_reads) against a fixture tree.

    C:/Python314/python.exe -m unittest tests.test_ingest_evals -v                       (workstation)
    /home/cereal/.venvs/chimera-intake/bin/python -m unittest tests.test_ingest_evals -v  (server)

Builds a miniature $OC / $L tree in a temp directory with the real file layouts (paths inside the files
are the real server paths; evalkit.at re-roots them), ingests it into a throwaway omni_lab_test_<hex>
database, and checks: the parent / OMNI1 / student pivot with blank cells, reader grades landing on the
runs they were written for, ledger resolution through dossier_sha256, machine grades, idempotent re-runs.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import omni_lab_db as db  # noqa: E402

TEST_DB_RE = re.compile(r"^omni_lab_test_[0-9a-f]{8}$")
OC = "/home/cereal/omni-campaign"
LX = "/mnt/lexar/omni-campaign"


def _have_rw() -> bool:
    try:
        db.db_params("rw")
        return True
    except db.LabError:
        return False


def _w(root: Path, rel: str, obj) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, str):
        p.write_text(obj, encoding="utf-8")
    else:
        p.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return p


def msg(text):
    return [{"role": "user", "content": text}]


def build_tree(tmp: Path) -> dict:
    oc, lx = tmp / "oc", tmp / "lx"
    bank = {"schema": "neural-forge-whole-response-review-prompts-v1", "prompts": [
        {"id": "a-case", "category": "reasoning", "messages": msg("2+2?"), "context": "4"},
        {"id": "b-case", "category": "coding", "messages": msg("write f"), "context": "code"},
        {"id": "c-case", "category": "writing", "messages": msg("an email"), "context": "email"}]}
    bank_p = _w(oc, "calibration/banks/lfm2_semantic_whole_response_prompts_v1.json", bank)
    _w(oc, "calibration/banks/matrix_campaign_probe_prompts_v1.json",
       {"kind": "free-run-probe", "prompt_set_sha256": "p", "prompts": ["The log reads:", "Three things:"]})
    _w(oc, "calibration/banks/matrix_campaign_smoke_prompts_v1.json",
       {"kind": "decode-smoke", "prompt_set_sha256": "s", "prompts": ["Explain a breaker."]})
    tools = [{"type": "function", "function": {"name": "geo", "parameters": {"type": "object"}}}]
    _w(oc, "calibration/banks/omni_toolcall_bank_v1.json", {"schema": "omni-distill-toolcall-v1/bank", "render": "r", "cases": [
        {"id": "tcb-1", "lane": "exec_single", "messages": msg("distance?"), "tools": tools, "expected": {"calls": []}},
        {"id": "tcb-2", "lane": "no_tool", "messages": msg("hello"), "tools": tools, "expected": {"calls": []}}]})
    bank_sha = db.file_sha256(bank_p)
    ledger = [
        {"checkpoint_id": "PARENT-Q38-27BU", "record_seq": 1, "event": "register", "stage": "parent",
         "directory": "/home/cereal/models/Qwen3.8-27B-Uncensored", "parameter_count": 27, "parent_code": "Q38-27BU"},
        {"checkpoint_id": "OC02-F-Q38-27BU-OMNI1", "record_seq": 1, "event": "register", "stage": "fold",
         "directory": LX + "/omni1/OC02-F-Q38-27BU-OMNI1", "parent_checkpoint_id": "PARENT-Q38-27BU", "parent_code": "Q38-27BU"},
        {"checkpoint_id": "OC02-F-Q38-27BU-OMNI1", "record_seq": 2, "event": "review", "stage": "fold",
         "directory": LX + "/omni1/OC02-F-Q38-27BU-OMNI1", "dossier_sha256": "omni1-dossier-sha", "sealed_verdict": "alive-degraded",
         "parent_code": "Q38-27BU"},
        {"checkpoint_id": "OC02-F-Q38-27BU-OMNI1.clone-pre-D1", "record_seq": 1, "event": "clone", "stage": "clone",
         "directory": LX + "/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1", "parent_checkpoint_id": "OC02-F-Q38-27BU-OMNI1"}]
    _w(oc, "ledger.jsonl", "\n".join(json.dumps(r) for r in ledger) + "\n")

    def dossier(label, stage, mdir, dsha, responses, cvd, source=None):
        return {"schema": "neural-forge-omni-parent-calibration-dossier-v1", "label": label, "stage": stage,
                "mode": "free-run", "dossier_sha256": dsha, "generated_at": "2026-10-10T02:00:00Z",
                "wall_time_seconds": 100.0, "parent_code": "Q38-27BU",
                "bank": {"path": "calibration/banks/lfm2_semantic_whole_response_prompts_v1.json", "file_sha256": bank_sha},
                "provenance": {"model_directory": mdir, "directory_sha256": "d" * 64},
                "runner": {"tool": "run_omni_frozen_review", "version": "1.2.0", "tool_sha256": "f" * 64},
                "generation": {"maximum_new_tokens": 1024, "dtype": "bfloat16", "devices": ["cuda:0", "cuda:1"],
                               "split": [2, 2], "eos_token_id": 9, "thinking": {"requested": "off", "took_effect": True},
                               "tokenizer_directory_sha256": "t" * 64, "runtime": "chimera-omni-native"},
                "export": {"torch_version": "2.11.0+cu128", "transformers_version": "5.12.1",
                           "chimera_tree": {"commit": "ef1ffb9c"}, "checkpoint": {"source": source or {}},
                           "linear_attention_kernels": {"kernel_path": "chimera-reference (per-position fp32 KDA)"},
                           "placement": {"block_device": ["cuda:0", "cuda:0", "cuda:1", "cuda:1"],
                                         "cuda": {"CUDA_VISIBLE_DEVICES": cvd, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}}},
                "child_responses": responses,
                "decode_smoke": {"maximum_new_tokens": 128, "prompts_file": "calibration/banks/matrix_campaign_smoke_prompts_v1.json",
                                 "records": [{"id": "smoke-1", "prompt_messages": msg("Explain a breaker."),
                                              "response_verbatim": "It trips.", "generated_tokens": 3,
                                              "termination": "complete", "loop_detected": False}]},
                "probe_prompts": {"prompts_file": "calibration/banks/matrix_campaign_probe_prompts_v1.json"},
                "emit_zero_floor": {"passed": True, "probe_count": 2, "distinct_argmax_tokens": 2, "probes": [
                    {"probe_index": 1, "logit_finite": True, "terminal_finite": True},
                    {"probe_index": 2, "logit_finite": True, "terminal_finite": True}]}}

    def resp(cid, text, term="complete", n=5, loop=False):
        p = next(x for x in bank["prompts"] if x["id"] == cid)
        return {"id": cid, "category": p["category"], "prompt_messages": p["messages"], "response_verbatim": text,
                "generated_tokens": n, "termination": term, "loop_detected": loop, "wall_time_seconds": 2.0}

    parent_d = _w(oc, "composite/s0/review/oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json", dossier(
        "PARENT-Q38-27BU-NATIVE-NOTHINK", "parent", OC + "/composite/s0/q38-import", "parent-dossier-sha",
        [resp("a-case", "4"), resp("b-case", "def f(): pass"), resp("c-case", "Dear team,")], "1,2",
        source={"model_dir": "/home/cereal/models/Qwen3.8-27B-Uncensored", "import_source": "qwen3_5"}))
    omni_d = _w(oc, "composite/s8/review/oc-dossier-OC02-OMNI1-NATIVE-NOTHINK.json", dossier(
        "OC02-OMNI1-NATIVE-NOTHINK", "fold", LX + "/omni1/OC02-F-Q38-27BU-OMNI1", "omni1-dossier-sha",
        [resp("a-case", "", n=0), resp("b-case", "loop loop", term="censored-at-cap", n=1024, loop=True),
         resp("c-case", "Dear")], "6,4"))
    _w(oc, "composite/s8/review/review-OC02-OMNI1.json",
       {"dossier_file_sha256": db.file_sha256(omni_d), "columns": {"column_a": 1, "column_b": 0},
        "bank1": {"lanes_total": 3, "canonical_signal": 1, "observed_miss": 2}})
    # toolcall bank run on the parent GGUF
    serve = {"url": "http://127.0.0.1:18321", "started_utc": "2026-10-14T15:12:43Z", "gguf": "/x/parent.Q4_K_M.gguf",
             "card_before": {"index": 1, "uuid": "GPU-p40-1"}, "env": {"CUDA_VISIBLE_DEVICES": "GPU-p40-1",
                                                                     "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
             "props": {"build_info": "b0-unknown"}, "command": ["llama-server"]}
    _w(oc, "distill/toolcall/receipts/serve-toolcall-gpu1.json", serve)
    _w(oc, "distill/toolcall/receipts/export_check_parent.json",
       {"q4": {"path": "/x/parent.Q4_K_M.gguf", "sha256": "q" * 64, "quant": "Q4_K_M"},
        "parent_dir": "/home/cereal/models/Qwen3.8-27B-Uncensored", "llama_cpp_commit": "b968"})
    rows = [{"id": "tcb-1", "lane": "exec_single", "response": "<tool_call>geo</tool_call>", "end": "eos", "tokens": 9,
             "wall_s": 1.5, "server": "http://127.0.0.1:18321"},
            {"id": "tcb-2", "lane": "no_tool", "response": "<tool_call>geo</tool_call>", "end": "eos", "tokens": 9,
             "wall_s": 1.5, "server": "http://127.0.0.1:18321"}]
    resp_p = _w(oc, "distill/toolcall/bank-runs/parent-q4km.responses.jsonl", "\n".join(json.dumps(r) for r in rows) + "\n")
    _w(oc, "distill/toolcall/bank-runs/parent-q4km.grade.json", {
        "schema": "omni-distill-toolcall-v1/bank-grade", "tool": {"tool": "toolcall_selfgen", "version": "1.0.2", "sha256": "c" * 64},
        "bank": {"path": OC + "/calibration/banks/omni_toolcall_bank_v1.json"},
        "responses": {"path": OC + "/distill/toolcall/bank-runs/parent-q4km.responses.jsonl"},
        "label": "parent q4km", "lanes": {"exec_single": {"passed": 1, "cases": 1}, "no_tool": {"passed": 0, "cases": 1}},
        "passed": 1, "cases": 2, "failures": [{"id": "tcb-2", "problems": ["called a tool on a no_tool case"]}],
        "graded_utc": "2026-10-14T15:23:30Z"})
    # student test run: 2 bank cases + 1 new prompt + the tool case
    sdir = "scratch/student-tests-20261021"
    student = LX + "/distill/fullkd/run1/out/student"
    _w(lx, "distill/fullkd/run1/out/student/config.json", {"layer_types": ["linear_attention", "full_attention"] * 2})
    _w(lx, "distill/fullkd/run1/out/student/student_manifest.json",
       {"base_identity": {"chimera_dir": LX + "/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1"}})
    recs = [dict(resp("a-case", "4"), form={"form_box": "CONTENT"}),
            dict(resp("b-case", "def f(): return 1"), form={"form_box": "CONTENT"}),
            {"id": "new-q", "category": "new-factual", "prompt_messages": msg("symbol for tungsten?"),
             "response_verbatim": "Tg", "generated_tokens": 2, "termination": "complete", "loop_detected": False,
             "form": {"form_box": "ECHO"}, "expected": "W"},
            {"id": "tcb-1", "category": "toolcall", "prompt_messages": msg("distance?"), "tools": tools,
             "response_verbatim": "<tool_call>geo</tool_call>", "generated_tokens": 9, "termination": "complete",
             "tool_grade": {"passed": True, "problems": []}}]
    argv = [lx.as_posix() + "/" + sdir + "/student_test.py", "--student", student,
            "--bank", OC + "/calibration/banks/lfm2_semantic_whole_response_prompts_v1.json",
            "--tool-bank", OC + "/calibration/banks/omni_toolcall_bank_v1.json",
            "--omni1-dossier", OC + "/composite/s8/review/oc-dossier-OC02-OMNI1-NATIVE-NOTHINK.json",
            "--parent-dossier", OC + "/composite/s0/review/oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json",
            "--parent-tool-responses", OC + "/distill/toolcall/bank-runs/parent-q4km.responses.jsonl"]
    results = _w(lx, f"{sdir}/results.json", {
        "schema": "chimera-omni-student-tests-20261021-v1", "tool": "student_test", "tool_version": "1.0.0",
        "tool_sha256": "a" * 64, "started_utc": "2026-10-21T13:29:24Z", "ended_utc": "2026-10-21T13:41:04Z",
        "student": student, "argv": argv,
        "generation": {"maximum_new_tokens": 1024, "dtype": "bfloat16", "thinking": {"requested": "off", "took_effect": True},
                       "torch_version": "2.11.0+cu128", "transformers_version": "5.12.1", "bank_file_sha256": bank_sha},
        "load": {"manifest": {"step": 611}, "kernels": {"attn_implementation": "sdpa"},
                 "placement": {"visible": "6,4", "plan": {"model.layers.0": 1, "model.layers.1": 0, "model.layers.2": 1,
                                                          "model.layers.3": 0, "lm_head": 0},
                               "devices": [{"cuda": 0, "capability": "sm75"}, {"cuda": 1, "capability": "sm86"}]}},
        "peak_allocated_gib": [10.0, 12.0], "records": recs})
    _w(lx, f"{sdir}/run_student_tests.sh", "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,4\n")
    _w(lx, f"{sdir}/grades.json", {
        "results_json_sha256": db.file_sha256(results), "cases": {
            "a-case": {"student": {"grade": "correct", "why": "4"}, "parent": {"grade": "correct", "why": "4"},
                       "omni1": {"grade": "no content", "why": "empty", "artifact_flags": ["EMPTY"]}},
            "b-case": {"student": {"grade": "partially correct", "why": "close"}, "parent": {"grade": "correct", "why": "ok"},
                       "omni1": {"grade": "content miss", "why": "loop", "artifact_flags": ["LOOP"]}},
            "new-q": {"student": {"grade": "content miss", "why": "Tg, not W"}},
            "tcb-1": {"student": {"grade": "correct", "why": "exact call"}, "parent": {"grade": "correct", "why": "same"},
                      "omni1": {"grade": "not run", "why": "no OMNI1 tool-bank result"}}}})
    # P11 read
    _w(oc, "distill/tps/p11_omni1_summary.json", {
        "schema": "omni-campaign.p11.read.v1/summary", "argv": ["p11_needle_engine.py", "read", "--kda-kernel", "fla",
                                                                  "--label", "p11_omni1", "--dtype", "bfloat16"],
        "identity": {"model": LX + "/omni1/OC02-F-Q38-27BU-OMNI1", "tool": "p11_needle_engine", "tool_version": "1.0.0",
                     "tool_sha256": "e" * 64},
        "needles": OC + "/distill/p11/needles_v1", "needles_sha256": "n" * 64, "placement": {"CUDA_VISIBLE_DEVICES": "4"},
        "rows": [{"id": "L32768-d25", "length": 32768, "needle": True, "passkey_exact_teacher_forced": False,
                  "mean_nll_digits": 1.6, "first_digit_rank": 9, "tok_s": 120.0},
                 {"id": "L32768-d25-control", "length": 32768, "needle": False, "passkey_exact_teacher_forced": False,
                  "tok_s": 120.0}], "written_utc": "2026-10-13T13:03:31Z"})
    return {"oc": oc, "lx": lx, "parent_d": parent_d, "omni_d": omni_d, "resp_p": resp_p}


@unittest.skipUnless(_have_rw(), "no omni_lab credential resolvable (OMNI_LAB_PG_PASSWORD[_COMMAND|_FILE])")
class EvalIngestTests(unittest.TestCase):
    SOURCES = ("banks", "dossiers", "toolcall_bankruns", "student_tests", "p11_reads")

    @classmethod
    def setUpClass(cls):
        from psycopg2 import sql
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tree = build_tree(Path(cls.tmp.name))
        cls._env = {k: os.environ.get(k) for k in ("OMNI_LAB_OC", "OMNI_LAB_LEXAR", "OMNI_LAB_STUDENT_TESTS",
                                                   "OMNI_LAB_DOSSIERS")}
        os.environ["OMNI_LAB_OC"] = cls.tree["oc"].as_posix()
        os.environ["OMNI_LAB_LEXAR"] = cls.tree["lx"].as_posix()
        os.environ.pop("OMNI_LAB_STUDENT_TESTS", None)
        os.environ.pop("OMNI_LAB_DOSSIERS", None)
        import ingest
        from ingest import common as C
        C.roots_from_env()
        cls.ingest, cls.C = ingest, C
        cls.dbname = f"omni_lab_test_{secrets.token_hex(4)}"
        assert TEST_DB_RE.match(cls.dbname)
        admin = db.connect("rw", app="test-create", autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(sql.Identifier(cls.dbname)))
        finally:
            admin.close()
        cls.conn = db.connect("rw", dbname=cls.dbname, app="test-evals")
        db.migrate(cls.conn)
        hid = db.ensure_host(cls.conn, "ai")
        for i, uuid, name in ((1, "GPU-p40-1", "Tesla P40"), (2, "GPU-p40-2", "Tesla P40"),
                              (4, "GPU-3090-4", "NVIDIA GeForce RTX 3090"), (6, "GPU-rtx8000-6", "Quadro RTX 8000")):
            db.upsert_gpu(cls.conn, hid, i, uuid, name)
        cls.conn.commit()
        cls.batches = cls._ingest_all()

    @classmethod
    def _ingest_all(cls):
        out = {}
        for name in cls.SOURCES:
            out[name] = cls.C.run_source(cls.ingest.load(name), None, conn_factory=lambda: cls.conn)
        return out

    @classmethod
    def tearDownClass(cls):
        from psycopg2 import sql
        cls.conn.close()
        for k, v in cls._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls.C.roots_from_env()
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

    def test_00_batches_ok_and_suites(self):
        rows = dict(self.q("SELECT source, count(*) FILTER (WHERE status = 'ok') FROM ingest_batch GROUP BY source"))
        for s in self.SOURCES:
            self.assertGreaterEqual(rows.get(s, 0), 1, s)
        suites = {(n, v): c for n, v, c in self.q(
            "SELECT ts.name, ts.version, count(sm.case_pk) FROM test_suite ts LEFT JOIN suite_member sm "
            "ON sm.suite_id = ts.id GROUP BY ts.name, ts.version")}
        self.assertEqual(suites[("bank_v1", "1")], 3)
        self.assertEqual(suites[("toolcall_v1", "1")], 2)
        self.assertEqual(suites[("student_tests_20261021", "1")], 4)
        self.assertEqual(suites[("student_new_20261021", "1")], 1)
        # bank cases are shared rows, not copies
        self.assertEqual(self.q("SELECT count(*) FROM test_case WHERE case_id = 'a-case'")[0][0], 1)

    def test_01_models_resolve_through_the_ledger(self):
        by_label = dict(self.q("SELECT er.label, m.model_key FROM eval_run er JOIN model m ON m.id = er.model_id"))
        self.assertEqual(by_label["OC02-OMNI1-NATIVE-NOTHINK"], "OC02-F-Q38-27BU-OMNI1")
        self.assertTrue(by_label["PARENT-Q38-27BU-NATIVE-NOTHINK"].startswith("path:"))
        lineage = self.q("SELECT c.stage, p.model_key, ml.relation FROM model_lineage ml JOIN model c ON c.id = ml.child_id "
                         "JOIN model p ON p.id = ml.parent_id ORDER BY 1, 2")
        self.assertIn(("S0", "PARENT-Q38-27BU", "export"), lineage)
        self.assertIn(("fullkd-run1", "OC02-F-Q38-27BU-OMNI1.clone-pre-D1", "distill"), lineage)

    def test_02_pivot_parent_omni1_student_with_blanks(self):
        rows = self.q("SELECT case_id, label, grade FROM v_eval_matrix vm JOIN eval_run er ON er.run_id = vm.eval_run_id "
                      "WHERE vm.suite IN ('bank_v1', 'student_tests_20261021', 'toolcall_v1')")
        cell = {(c, lab): g for c, lab, g in rows}
        self.assertEqual(cell[("a-case", "PARENT-Q38-27BU-NATIVE-NOTHINK")], "correct")
        self.assertEqual(cell[("a-case", "OC02-OMNI1-NATIVE-NOTHINK")], "none")
        self.assertEqual(cell[("b-case", "OC02-OMNI1-NATIVE-NOTHINK")], "miss")
        self.assertEqual(cell[("b-case", "student-tests-20261021")], "partial")
        self.assertEqual(cell[("tcb-1", "student-tests-20261021")], "correct")       # reader wins over machine pass
        self.assertEqual(cell[("tcb-1", "parent q4km")], "correct")
        self.assertEqual(cell[("tcb-2", "parent q4km")], "fail")
        self.assertIsNone(cell[("c-case", "OC02-OMNI1-NATIVE-NOTHINK")])             # result exists, nobody graded it
        self.assertNotIn(("c-case", "student-tests-20261021"), cell)                 # blank: the student never ran it
        self.assertNotIn(("new-q", "PARENT-Q38-27BU-NATIVE-NOTHINK"), cell)          # blank: newer case, older run
        self.assertNotIn(("tcb-1", "OC02-OMNI1-NATIVE-NOTHINK"), cell)              # 'not run' writes nothing

    def test_03_machine_fields_and_devices(self):
        flags = dict(self.q("SELECT tc.case_id, res.artifact_flags FROM eval_result res JOIN test_case tc ON tc.id = res.case_pk "
                            "JOIN eval_run er ON er.run_id = res.eval_run_id WHERE er.label = 'OC02-OMNI1-NATIVE-NOTHINK'"))
        self.assertEqual(flags["a-case"], ["empty"])
        self.assertEqual(flags["b-case"], ["cap-ride", "loop"])
        devices = self.q("SELECT g.gpu_index, rd.role, lower(rd.layers) FROM run_device rd JOIN gpu g ON g.id = rd.gpu_id "
                         "JOIN eval_run er ON er.run_id = rd.run_id WHERE er.label = 'OC02-OMNI1-NATIVE-NOTHINK' "
                         "AND er.suite_id = (SELECT id FROM test_suite WHERE name = 'bank_v1') ORDER BY 1")
        self.assertEqual(devices, [(4, "split-1", 2), (6, "split-0", 0)])
        student_dev = self.q("SELECT g.gpu_index, rd.role FROM run_device rd JOIN gpu g ON g.id = rd.gpu_id "
                             "JOIN eval_run er ON er.run_id = rd.run_id WHERE er.engine = 'qwen35-student' ORDER BY 1")
        self.assertEqual(student_dev, [(4, "linear-attn"), (6, "full-attn")])
        ub = dict(self.q("SELECT mt.name, mt.value FROM metric mt JOIN eval_run er ON er.run_id = mt.run_id "
                         "WHERE er.label = 'OC02-OMNI1-NATIVE-NOTHINK' AND mt.name LIKE 'upper_bound/%'"))
        self.assertEqual(ub["upper_bound/column_a"], 1.0)
        p11 = self.q("SELECT tc.case_id, vm.grade FROM v_eval_matrix vm JOIN test_case tc ON tc.id = vm.case_pk "
                     "WHERE vm.suite = 'p11_needles' ORDER BY 1")
        self.assertEqual(p11, [("L32768-d25", "fail"), ("L32768-d25-control", None)])

    def test_04_rerun_is_idempotent(self):
        tables = ("test_case", "suite_member", "run", "eval_run", "eval_result", "eval_grade", "metric", "run_device",
                  "model", "model_lineage", "env", "tool_version", "grader")
        before = {t: self.q(f"SELECT count(*) FROM {t}")[0][0] for t in tables}
        self._ingest_all()
        after = {t: self.q(f"SELECT count(*) FROM {t}")[0][0] for t in tables}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
