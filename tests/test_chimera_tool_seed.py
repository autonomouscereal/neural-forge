"""Actual deterministic execution and split-isolation checks for operator seeds."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.chimera_recovery import tool_seed


ROOT = Path(__file__).resolve().parents[1]


class ToolSeedTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(prefix="seed-test-", dir=ROOT)

    def test_arithmetic_records_exact_rational_and_actual_zero_division(self):
        with self.temporary() as root:
            toolbox = tool_seed.Toolbox(root)
            self.assertEqual(toolbox.execute("local_arithmetic", {"operation": "divide", "a": 7, "b": 3})["value"], "7/3")
            failed = toolbox.execute("local_arithmetic", {"operation": "divide", "a": 7, "b": 0})
            self.assertFalse(failed["ok"])
            self.assertEqual(failed["error"]["type"], "ZeroDivisionError")

    def test_text_transforms_and_hex_errors_are_executed(self):
        with self.temporary() as root:
            toolbox = tool_seed.Toolbox(root)
            self.assertEqual(toolbox.execute("local_text", {"operation": "reverse", "text": "雪abc"})["value"], "cba雪")
            self.assertFalse(toolbox.execute("local_text", {"operation": "decode_hex", "text": "not-hex"})["ok"])
            self.assertEqual(toolbox.execute("local_text", {"operation": "decode_hex", "text": "6869"})["value"], "hi")

    def test_file_missing_write_append_and_read_have_real_state(self):
        with self.temporary() as root:
            toolbox = tool_seed.Toolbox(root)
            self.assertFalse(toolbox.execute("local_file", {"operation": "read", "name": "case.txt"})["ok"])
            written = toolbox.execute("local_file", {"operation": "write", "name": "case.txt", "text": "snow 雪"})
            self.assertEqual(written["value"]["characters"], 6)
            self.assertTrue((Path(root) / "case.txt").exists())
            toolbox.execute("local_file", {"operation": "append", "name": "case.txt", "text": " added"})
            self.assertEqual(toolbox.execute("local_file", {"operation": "read", "name": "case.txt"})["value"], "snow 雪 added")

    def test_schema_rejects_unknown_arguments_and_path_traversal_before_execution(self):
        with self.temporary() as root:
            toolbox = tool_seed.Toolbox(root)
            for arguments in ({"operation": "write", "name": "../escape.txt", "text": "bad"},
                              {"operation": "read", "name": "case.txt", "extra": "invalid"}):
                with self.assertRaises(Exception):
                    toolbox.execute("local_file", arguments)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_retry_episode_preserves_failure_then_later_success_and_call_ids(self):
        with self.temporary() as root:
            row, receipt = tool_seed.build_episode("retry_division", "tool_retry", "train", 0, 9, tool_seed.Toolbox(root))
        self.assertEqual([call["result"]["ok"] for call in receipt["calls"]], [False, True])
        self.assertEqual([item["status"] for item in row["quality"]["outcomes"]], ["failure", "success"])
        ids = [call["call_id"] for call in receipt["calls"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [m["tool_call_id"] for m in row["messages"] if m["role"] == "tool"])

    def test_parallel_batch_dispatch_and_reverse_delivery_keep_valid_pairing(self):
        with self.temporary() as root:
            row, receipt = tool_seed.build_episode("parallel_arithmetic", "tool_parallel", "validation", 0, 13, tool_seed.Toolbox(root))
        self.assertEqual(row["quality"]["parallel_groups"], 1)
        self.assertTrue(all(call["concurrent_dispatch"] for call in receipt["calls"]))
        calls = next(m["tool_calls"] for m in row["messages"] if m.get("tool_calls"))
        self.assertEqual([call["call_id"] for call in receipt["calls"]], [calls[1]["id"], calls[0]["id"]])

    def test_multiturn_context_and_assistant_mask_preserve_results_as_inputs(self):
        with self.temporary() as root:
            row, _ = tool_seed.build_episode("multiturn_text", "tool_multiturn", "validation", 0, 16, tool_seed.Toolbox(root))
        self.assertEqual(sum(m["role"] == "user" for m in row["messages"]), 2)
        for index, message in enumerate(row["messages"]):
            self.assertEqual(row["supervision"]["message_mask"][index], int(message["role"] == "assistant"))
        self.assertTrue(row["quality"]["schema_complete"])

    def test_generation_bounds_and_existing_directory_fail_closed(self):
        with self.temporary() as root:
            with self.assertRaises(ValueError):
                tool_seed.generate(Path(root) / "invalid", 127)
            with self.assertRaises(FileExistsError):
                tool_seed.generate(Path(root), 128)

    def test_generated_tranche_has_all_facets_actual_receipts_and_isolated_families(self):
        with self.temporary() as root:
            out = Path(root) / "corpus"
            report = tool_seed.generate(out, 128)
            rows = [json.loads(line) for line in (out / "ready.jsonl").read_text(encoding="utf-8").splitlines()]
            receipts = [json.loads(line) for line in (out / "execution_receipts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["episodes"], 128)
            self.assertEqual(len(report["tool_route_counts"]), 6)
            self.assertEqual(set(report["split_counts"]), {"train", "validation", "test"})
            self.assertGreater(report["observed_failures"], 0)
            self.assertGreater(report["parallel_groups"], 0)
            self.assertGreater(report["multi_user_turn_episodes"], 0)
            self.assertTrue(all(row["provenance"]["actual_tool_execution"] and row["quality"]["train_ready"] for row in rows))
            self.assertTrue(all(tool_seed.sha(receipt["calls"]) == row["provenance"]["execution_receipt_sha256"] for row, receipt in zip(rows, receipts)))
            self.assertEqual(report["isolation"]["cross_split_conflicts"], {})
            self.assertFalse(any(path.is_dir() for path in out.iterdir()))

    def test_split_audit_rejects_shared_source_case_or_parameters(self):
        with self.temporary() as root:
            row, _ = tool_seed.build_episode("call_add", "tool_call", "train", 0, 3, tool_seed.Toolbox(root))
        cloned = json.loads(json.dumps(row))
        cloned["provenance"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "cross_split"):
            tool_seed.audit_splits([row, cloned])


if __name__ == "__main__":
    unittest.main()
