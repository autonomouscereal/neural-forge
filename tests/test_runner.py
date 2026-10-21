#!/usr/bin/env python3
"""Runner tests: omni_lab_run (run / compare / list / add-suite / add-cases / grade / log-run), the worker,
the graders and ingest.lab_runs.

Pure tests need nothing.  The DB tests create omni_lab_test_<hex> (as test_schema does), point the CLI at
it through OMNI_LAB_PG_DB / OMNI_LAB_RO_PG_DB, and drive the REAL worker as a subprocess with the 'echo'
mechanics bed (no torch, no GPU), so spec -> worker -> graders -> results.jsonl -> ingest -> compare is the
production path end to end.  Graders that need a campaign file (toolcall_selfgen, student_test) are checked
when the file exists on this host (the AI server has both; the workstation has toolcall_selfgen under
neural-forge/reports/omni-campaign-2026-09).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import secrets
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import omni_lab_db as db  # noqa: E402

TEST_DB_RE = re.compile(r"^omni_lab_test_[0-9a-f]{8}$")
WORKSTATION_OC = Path("C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09")
SERVER_OC = Path("/home/cereal/omni-campaign")
CAMPAIGN_OC = SERVER_OC if SERVER_OC.is_dir() else WORKSTATION_OC
STUDENT_TEST = Path(os.environ.get("OMNI_LAB_STUDENT_TEST",
                                   "/mnt/lexar/omni-campaign/scratch/student-tests-20261021/student_test.py"))


def _have_rw() -> bool:
    try:
        db.db_params("rw")
        return True
    except db.LabError:
        return False


def cli(*argv: str) -> tuple[int, str]:
    import omni_lab
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = omni_lab.main(list(argv))
    return code, buf.getvalue()


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
TOOL_CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
CODE = "```python\ndef add(a, b):\n    return a + b\n```"


def bank_v1_style(path: Path) -> Path:
    path.write_text(json.dumps({"schema": "test-bank", "prompts": [
        {"id": "t-chat", "category": "conversation", "prompt": "Say hello."},
        {"id": "t-code", "category": "coding", "prompt": "Write add(a, b).", "asserts": ["add(2, 3) == 5", "add(-1, 1) == 0"],
         "echo_response": CODE},
        {"id": "t-tool", "category": "toolcall", "messages": [{"role": "user", "content": "Weather in Paris?"}],
         "tools": TOOLS, "expected": {"calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
         "echo_response": TOOL_CALL},
        {"id": "t-image", "category": "vision", "messages": [{"role": "user", "content": [
            {"type": "image", "path": "card.png"}, {"type": "text", "text": "What is shown?"}]}]},
        {"id": "t-raw", "category": "probe", "raw_text": "The capital of France is"},
    ]}), encoding="utf-8")
    return path


class PureTests(unittest.TestCase):
    def test_parse_case_file_layouts(self):
        import omni_lab_run as R
        with tempfile.TemporaryDirectory() as tmp:
            cases = R.parse_case_file(bank_v1_style(Path(tmp, "b.json")), "x")
            by = {c["case_id"]: c for c in cases}
            self.assertEqual(by["t-chat"]["prompt"], [{"role": "user", "content": "Say hello."}])
            self.assertEqual(by["t-code"]["grader"], "student_test.code_check")
            self.assertEqual(by["t-tool"]["grader"], "toolcall_selfgen.grade_case")
            self.assertEqual(by["t-raw"]["prompt"], {"raw_text": "The capital of France is"})
            self.assertIsNone(by["t-chat"]["grader"])
            Path(tmp, "s.json").write_text(json.dumps(["one", "two"]), encoding="utf-8")
            self.assertEqual([c["case_id"] for c in R.parse_case_file(Path(tmp, "s.json"), "pfx")], ["pfx-1", "pfx-2"])
            Path(tmp, "t.jsonl").write_text("\n".join(json.dumps(
                {"id": f"task-{i}", "input": {"messages": [{"role": "user", "content": f"q{i}"}],
                                              "media": [{"kind": "image", "path": "a.png"}]},
                 "oracle": {"kind": "arithmetic", "expected": "4"}}) for i in range(2)), encoding="utf-8")
            t = R.parse_case_file(Path(tmp, "t.jsonl"), "x")
            self.assertEqual(t[0]["grader"], "chimera_recovery.evaluate")
            self.assertEqual(t[0]["modality_in"], ["image", "text"])
            Path(tmp, "d.json").write_text(json.dumps(["a", {"id": "x-1", "prompt": "b"}]), encoding="utf-8")
            with self.assertRaises(db.LabError):
                R.parse_case_file(Path(tmp, "d.json"), "x")

    def test_modality_gate_and_graders_applicable(self):
        from adapters import common as A
        from graders import registry as G
        img = {"case_id": "i", "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]}]}
        self.assertEqual(A.non_text_inputs(img), ["image"])
        self.assertEqual(A.non_text_inputs({"case_id": "a", "prompt": [{"role": "user", "content": "hi"}],
                                            "modality_in": ["text", "audio"]}), ["audio"])
        self.assertEqual(A.non_text_inputs({"case_id": "t", "prompt": [{"role": "user", "content": "hi"}]}), [])
        self.assertEqual(G.applicable({"grader": "toolcall_selfgen.grade_case", "reference": {"asserts": ["f(1)"]}}),
                         ["toolcall_selfgen.grade_case", "student_test.code_check"])
        self.assertEqual(G.applicable({"reference": {"oracle": {"kind": "arithmetic"}}}), ["chimera_recovery.evaluate"])
        self.assertEqual(G.applicable({"reference": {"context": "x"}}), [])

    def test_student_device_plans(self):
        from adapters import qwen35 as Q
        types = ["linear_attention"] * 3 + ["full_attention"]
        types = types * 4                              # 16 layers: 12 linear, 4 full
        sentinel = object()
        self.assertIs(Q.make_plan(3, sentinel), sentinel)  # 3 cards: student_test's own plan, unchanged
        one = Q.make_plan(1, None)(types)
        self.assertEqual(set(one.values()), {0})
        two = Q.make_plan(2, None)(types)
        self.assertTrue(all(two[f"model.layers.{i}"] == (0 if t == "full_attention" else 1) for i, t in enumerate(types)))
        four = Q.make_plan(4, None)(types)
        lin = [four[f"model.layers.{i}"] for i, t in enumerate(types) if t == "linear_attention"]
        self.assertEqual(lin, sorted(lin))             # contiguous runs
        self.assertEqual(set(lin), {1, 2, 3})
        self.assertEqual(four["lm_head"], 0)

    def test_chimera_command_line(self):
        from adapters import chimera as CH
        spec = {"model_path": "/m/OMNI1", "label": "L", "dtype": "bfloat16", "tokenizer_dir": None,
                "gen": {"cap": 64, "thinking": "off", "extra_eos_tokens": ["<|im_end|>"], "ledger_band_mask": True},
                "device": {"kind": "cuda", "split": "46,24"},
                "paths": {"chimera_runner": "/oc/run_omni_frozen_review.py", "probe_prompts": "/oc/probe.json",
                          "calibration_dir": "/oc/calibration", "chimera_src": "/oc/tree"}}
        cmd = CH.build_command(spec, Path("/r"), Path("/r/bank.json"), "ab" * 32, ["c1", "c2"])
        joined = " ".join(cmd)
        for piece in ("--model /m/OMNI1", "--bank-sha256 " + "ab" * 32, "--lanes c1,c2", "--split 46,24",
                      "--maximum-new-tokens 64", "--thinking off", "--ledger-band-logit-mask", "--skip-smoke",
                      "--chimera-src /oc/tree", "--text-only"):
            self.assertIn(piece, joined)

    def test_card_rules(self):
        import omni_lab_run as R
        smi = {0: {"index": 0, "name": "3090", "uuid": "GPU-a", "used_mib": 24},
               5: {"index": 5, "name": "P40", "uuid": "GPU-b", "used_mib": 10},
               1: {"index": 1, "name": "P40", "uuid": "GPU-c", "used_mib": 15889}}
        self.assertEqual([c["uuid"] for c in R.check_cards(smi, [0], allow_gpu5=False, busy_limit=2000)], ["GPU-a"])
        for idx, kw in (([5], {}), ([1], {}), ([3], {}), ([0, 0], {})):
            with self.assertRaises(db.LabError):
                R.check_cards(smi, idx, allow_gpu5=False, busy_limit=2000)
        self.assertEqual(len(R.check_cards(smi, [5], allow_gpu5=True, busy_limit=2000)), 1)

    def test_metric_args(self):
        import omni_lab_run as R
        self.assertEqual(R._metrics(["loss=1.5@10", "tok_s=3"]), [("loss", 10, 1.5), ("tok_s", 0, 3.0)])
        with self.assertRaises(db.LabError):
            R._metrics(["loss:1"])


@unittest.skipUnless(_have_rw(), "no omni_lab rw credentials in this environment")
class RunnerDbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from psycopg2 import sql
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dbname = f"omni_lab_test_{secrets.token_hex(4)}"
        assert TEST_DB_RE.match(cls.dbname)
        admin = db.connect("rw", app="test-create", autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(sql.Identifier(cls.dbname)))
        finally:
            admin.close()
        cls._env = {k: os.environ.get(k) for k in ("OMNI_LAB_PG_DB", "OMNI_LAB_RO_PG_DB", "OMNI_LAB_RUNS", "OMNI_LAB_OC",
                                                   "OMNI_LAB_GPU_PYTHON")}
        os.environ["OMNI_LAB_PG_DB"] = cls.dbname
        os.environ["OMNI_LAB_RO_PG_DB"] = cls.dbname
        os.environ["OMNI_LAB_RUNS"] = str(Path(cls.tmp.name, "runs"))
        os.environ["OMNI_LAB_OC"] = str(CAMPAIGN_OC)
        from ingest import common as C
        C.roots_from_env()
        cls.conn = db.connect("rw", app="test-runner")
        db.migrate(cls.conn)
        with cls.conn.cursor() as cur:   # the read-only role reads this throwaway DB like the real one
            cur.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO omni_lab_ro")
        hid = db.ensure_host(cls.conn, db.this_host())
        cls.model_dir = Path(cls.tmp.name, "model")
        cls.model_dir.mkdir()
        db.upsert_model(cls.conn, "TEST-ECHO-MODEL", "lab", "other", "hf", path=str(cls.model_dir), host_id=hid)
        db.upsert_model(cls.conn, "TEST-ECHO-PARENT", "lab", "parent", "hf", path=None, host_id=hid)
        cls.conn.commit()
        cls.bank = bank_v1_style(Path(cls.tmp.name, "bank.json"))

    @classmethod
    def tearDownClass(cls):
        from psycopg2 import sql
        cls.conn.close()
        for k, v in cls._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        from ingest import common as C
        C.roots_from_env()
        cls.tmp.cleanup()
        if os.environ.get("OMNI_LAB_TEST_KEEP") == "1":
            print(f"\n[kept] {cls.dbname}")
            return
        assert TEST_DB_RE.match(cls.dbname), cls.dbname
        admin = db.connect("rw", dbname="omni_lab", app="test-drop", autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(cls.dbname)))
        finally:
            admin.close()

    def q(self, query, params=None):
        self.conn.rollback()
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def run_echo(self, model: str, suite: str, *extra: str) -> tuple[int, str, int]:
        code, out = cli("run", "--model", model, "--suite", suite, "--engine", "echo", "--cpu",
                        "--gpu-python", sys.executable, "--baseline", "none", *extra)
        m = re.search(r"run id\s+(\d+)", out)
        self.assertIsNotNone(m, out)
        return code, out, int(m.group(1))

    def test_10_full_path(self):
        code, out = cli("add-suite", "tsuite", "--file", str(self.bank))
        self.assertEqual(code, 0, out)
        self.assertIn("tsuite@1: 5 cases", out)
        code, out = cli("add-suite", "tsuite", "--file", str(self.bank), "--version", "1")
        self.assertNotEqual(code, 0)                    # immutable: same name@version refused

        code, out, rid = self.run_echo("TEST-ECHO-MODEL", "tsuite", "--save-test", "echo-all", "--cap", "64")
        self.assertEqual(code, 0, out)
        rows = dict((r[0], r[1:]) for r in self.q(
            "SELECT tc.case_id, res.termination, res.output_text, eg.grade, res.artifact_flags FROM eval_result res "
            "JOIN test_case tc ON tc.id = res.case_pk JOIN v_effective_grade eg ON eg.eval_result_id = res.id "
            "WHERE res.eval_run_id = %s", (rid,)))
        self.assertEqual(set(rows), {"t-chat", "t-code", "t-tool", "t-image", "t-raw"})
        self.assertEqual(rows["t-image"][0], "unsupported-modality")     # recorded, never skipped
        self.assertIn("engine-unsupported", rows["t-image"][3])
        self.assertEqual(rows["t-chat"][0], "eos")
        self.assertIsNone(rows["t-chat"][2])                             # bank-style: reader-graded, no machine grade
        if (CAMPAIGN_OC / "distill/tools/toolcall_selfgen.py").is_file():
            self.assertEqual(rows["t-tool"][2], "pass")
        if STUDENT_TEST.is_file():
            self.assertEqual(rows["t-code"][2], "pass")
        st = self.q("SELECT r.status, r.kind, er.engine, t.name, r.config->>'device_kind', r.config_sha256 IS NOT NULL, "
                    "e.python IS NOT NULL OR e.id IS NULL FROM run r JOIN eval_run er ON er.run_id = r.id "
                    "LEFT JOIN test t ON t.id = er.test_id LEFT JOIN env e ON e.id = r.env_id WHERE r.id = %s", (rid,))[0]
        self.assertEqual(st[:5], ("pass", "eval", "echo", "echo-all", "cpu"))
        self.assertTrue(st[5])
        self.assertIn("results 5/5", out)                                # console summary line

        # the saved test re-runs by name; a second model gives the comparison a second column
        code, out, rid2 = self.run_echo("TEST-ECHO-PARENT", "tsuite", "--test", "echo-all")
        self.assertEqual(code, 0, out)
        type(self).rids = (rid, rid2)

    def test_15_back_to_back_runs_get_their_own_rows(self):
        # same model, suite and cases inside one second: two run keys, two run directories (was a FileExistsError)
        a = self.run_echo("TEST-ECHO-MODEL", "tsuite@1", "--cases", "t-chat")
        b = self.run_echo("TEST-ECHO-MODEL", "tsuite@1", "--cases", "t-chat")
        self.assertEqual((a[0], b[0]), (0, 0), a[1] + b[1])
        self.assertNotEqual(a[2], b[2])
        dirs = self.q("SELECT run_dir FROM run WHERE id = ANY(%s)", ([a[2], b[2]],))
        self.assertEqual(len({d[0] for d in dirs}), 2)

    def test_20_add_cases_blank_for_older_runs(self):
        extra = Path(self.tmp.name, "more.json")
        extra.write_text(json.dumps({"prompts": [{"id": "t-new", "category": "new", "prompt": "Name a prime."}]}),
                         encoding="utf-8")
        code, out = cli("add-cases", "tsuite", "--file", str(extra))
        self.assertEqual(code, 0, out)
        self.assertIn("tsuite@2: 6 cases", out)
        code, out, rid3 = self.run_echo("TEST-ECHO-MODEL", "tsuite@2", "--cases", "t-new,t-chat")
        self.assertEqual(code, 0, out)
        t0 = time.monotonic()
        code, out = cli("compare", "--suite", "tsuite")
        elapsed = time.monotonic() - t0
        self.assertEqual(code, 0, out)
        self.assertLess(elapsed, 2.0, out)
        line = next(ln for ln in out.splitlines() if ln.startswith("t-new"))
        cells = line.split()[2:]
        self.assertGreaterEqual(len(cells), 3, out)
        self.assertEqual(cells[:-1], ["-"] * (len(cells) - 1), line)   # every older run never had t-new
        self.assertNotEqual(cells[-1], "-", line)                        # the newest run (last column) did
        type(self).rid3 = rid3

    def test_30_human_grade_wins(self):
        rid = self.rids[0]
        code, out = cli("grade", str(rid), "t-chat", "correct", "--why", "greets", "--grader", "human:claude")
        self.assertEqual(code, 0, out)
        code, out = cli("grade", str(rid), "t-missing", "correct")
        self.assertNotEqual(code, 0)                                     # no result: nothing to grade
        code, out = cli("compare", str(self.rids[0]), str(self.rids[1]))
        line = next(ln for ln in out.splitlines() if ln.startswith("t-chat"))
        self.assertEqual(line.split()[2:], ["C*", "?"], line)

    def test_40_reingest_is_idempotent(self):
        from ingest import common as C
        from ingest import lab_runs
        before = self.q("SELECT (SELECT count(*) FROM eval_result), (SELECT count(*) FROM eval_grade), "
                        "(SELECT count(*) FROM run), (SELECT count(*) FROM metric)")[0]
        batch = C.run_source(lab_runs, None, conn_factory=lambda: self.conn)
        self.assertGreaterEqual(batch.seen, 3)
        after = self.q("SELECT (SELECT count(*) FROM eval_result), (SELECT count(*) FROM eval_grade), "
                       "(SELECT count(*) FROM run), (SELECT count(*) FROM metric)")[0]
        self.assertEqual(before, after)

    def test_50_log_run(self):
        code, out = cli("log-run", "start", "--kind", "distill", "--name", "test job", "--config", '{"lr": 1e-5}',
                        "--model", "TEST-ECHO-MODEL")
        self.assertEqual(code, 0, out)
        rid = int(out.strip().splitlines()[-1])
        code, out = cli("log-run", "metric", str(rid), "--metric", "loss=2.5@1", "--metric", "loss=2.0@2")
        self.assertEqual(code, 0, out)
        code, out = cli("log-run", "end", str(rid), "--status", "pass", "--exit-code", "0", "--metric", "tokens=1000")
        self.assertEqual(code, 0, out)
        row = self.q("SELECT status, exit_code, ended_at IS NOT NULL, (config->>'lr')::float FROM run WHERE id = %s",
                     (rid,))[0]
        self.assertEqual(row, ("pass", 0, True, 1e-05))
        self.assertEqual(self.q("SELECT count(*) FROM metric WHERE run_id = %s", (rid,))[0][0], 3)

    def test_60_list(self):
        for what in (("suites",), ("tests",), ("cases", "tsuite"), ("runs",), ("models", "TEST")):
            code, out = cli("list", *what)
            self.assertEqual(code, 0, out)
        self.assertIn("echo-all", cli("list", "tests")[1])


if __name__ == "__main__":
    unittest.main()
