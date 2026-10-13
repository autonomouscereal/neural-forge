"""Integrity and supervision checks against bounded synthetic trace fixtures."""
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import zlib

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/memory_harvest.py"
spec = importlib.util.spec_from_file_location("memory_harvest", SCRIPT)
harvest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harvest)


def event(number, kind, payload, session="fixture-session"):
    text = json.dumps(payload, ensure_ascii=False)
    raw = text.encode()
    compressed = zlib.compress(raw)
    return {"id": number, "event_uid": f"fixture-event-{number}", "archive_event_id": number,
            "archive_event_uid": f"fixture-event-{number}", "archive_id": number + 100,
            "created_at": datetime(2026, 10, 13, tzinfo=timezone.utc), "space": "fixture-project",
            "session_id": session, "agent": "Fixture", "source": "fixture_hook", "event_type": kind,
            "metadata": {}, "payload": compressed, "compression": "zlib", "payload_format": "json",
            "payload_sha256": hashlib.sha256(raw).hexdigest(), "raw_chars": len(text),
            "compressed_bytes": len(compressed), "redacted": True}


def call(number="recorded-1", value="input"):
    return {"id": number, "type": "function", "function": {"name": "fixture_tool", "arguments": {"value": value}}}


TOOLS = [{"type": "function", "function": {"name": "fixture_tool", "parameters": {
    "type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}}}]


def captured_messages(value="input"):
    return [{"role": "user", "content": "Read the supplied value and report the result."},
            {"role": "assistant", "content": "", "tool_calls": [call(value=value)]},
            {"role": "tool", "tool_call_id": "recorded-1", "content": '{"exit_code":0,"output":"read"}'},
            {"role": "assistant", "content": "The read completed."}]


class MemoryHarvestTests(unittest.TestCase):
    def test_archive_verifies_uncompressed_utf8_hash_and_character_length(self):
        payload = {"prompt": "snow 雪", "token_count": 7}
        self.assertEqual(harvest.archive_payload(event(1, "user_prompt", payload)), payload)

    def test_archive_rejects_mismatched_digest_before_using_body(self):
        sample = event(1, "user_prompt", {"prompt": "value"})
        sample["payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            harvest.archive_payload(sample)

    def test_archive_rejects_decompression_bomb_and_trailing_stream(self):
        sample = event(1, "user_prompt", {"prompt": "x" * 10000})
        with self.assertRaisesRegex(ValueError, "limit"):
            harvest.archive_payload(sample, max_bytes=100)
        sample = event(1, "user_prompt", {"prompt": "value"})
        sample["payload"] += zlib.compress(b"second stream")
        sample["compressed_bytes"] = len(sample["payload"])
        with self.assertRaisesRegex(ValueError, "limit"):
            harvest.archive_payload(sample)

    def test_redaction_preserves_benign_token_counts_and_call_ids(self):
        token = "sk-" + "x" * 32
        clean = harvest.redact({"api_key": token, "token_count": 4, "tool_call_id": "recorded-1",
                                "note": "Authorization: Bearer " + token})
        self.assertEqual(clean["api_key"], harvest.REDACTED)
        self.assertEqual(clean["token_count"], 4)
        self.assertEqual(clean["tool_call_id"], "recorded-1")
        self.assertNotIn(token, harvest.canonical(clean))

    def test_secret_in_assistant_arguments_is_excluded_not_trained_as_placeholder(self):
        sample = event(1, "tool_call", {"messages": captured_messages(), "tools": TOOLS})
        payload = harvest.archive_payload(sample)
        payload["messages"][1]["tool_calls"][0]["function"]["arguments"] = {"password": "fixture"}
        rows, report = harvest.extract_rows([event(1, "tool_call", payload)])
        self.assertEqual(rows, [])
        self.assertEqual(report["rejected"]["redacted_assistant_call"], 1)

    def test_archive_event_pairing_must_match_ids_and_uid(self):
        sample = event(1, "tool_call", {"messages": captured_messages()})
        sample["archive_event_uid"] = "different-event"
        rows, report = harvest.extract_rows([sample])
        self.assertEqual(rows, [])
        self.assertEqual(report["rejected"]["archive_event_identity_mismatch"], 1)

    def test_hook_needs_real_call_id_and_result(self):
        payload = {"tool_name": "fixture_tool", "tool_input": {"value": "one"}, "tool_response": {"exit_code": 0}}
        rows, report = harvest.extract_rows([event(1, "tool_call", payload)])
        self.assertEqual(rows, [])
        self.assertEqual(report["rejected"]["missing_recorded_call_id"], 1)

    def test_exact_turn_prompt_and_assistant_only_mask(self):
        events = [event(1, "user_prompt", {"prompt": "Read a value", "turn_id": "turn-1"}),
                  event(2, "tool_call", {"tool_name": "fixture_tool", "tool_use_id": "recorded-1",
                        "tool_input": {"value": "one"}, "tool_response": {"exit_code": 0}, "turn_id": "turn-1"})]
        rows, _ = harvest.extract_rows(events)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["supervision"]["message_mask"], [0, 1, 0])
        self.assertEqual(rows[0]["supervised_assistant_messages"], [1])
        self.assertEqual(rows[0]["messages"][1]["tool_calls"][0]["id"], "recorded-1")
        self.assertEqual(rows[0]["quality"]["context_linkage"], "exact_turn")
        self.assertFalse(rows[0]["quality"]["schema_complete"])
        self.assertFalse(rows[0]["quality"]["train_ready"])

    def test_adjacent_prompt_without_exact_turn_does_not_fabricate_context(self):
        events = [event(1, "user_prompt", {"prompt": "Adjacent is not an attestation"}),
                  event(2, "tool_call", {"tool_name": "fixture_tool", "tool_use_id": "recorded-1",
                        "tool_input": {"value": "one"}, "tool_response": {"exit_code": 0}})]
        rows, _ = harvest.extract_rows(events)
        self.assertFalse(rows[0]["quality"]["context_complete"])
        self.assertEqual(rows[0]["messages"][0]["role"], "assistant")

    def test_ambiguous_prompt_same_turn_is_rejected(self):
        events = [event(1, "user_prompt", {"prompt": "First", "turn_id": "same"}),
                  event(2, "user_prompt", {"prompt": "Second", "turn_id": "same"}),
                  event(3, "tool_call", {"tool_name": "fixture_tool", "tool_use_id": "c",
                        "tool_input": {"value": "one"}, "tool_response": {"exit_code": 0}, "turn_id": "same"})]
        rows, report = harvest.extract_rows(events)
        self.assertEqual(rows, [])
        self.assertEqual(report["rejected"]["ambiguous_prompt_for_turn"], 1)

    def test_parallel_results_can_arrive_in_reverse_order_without_losing_ids(self):
        messages = [{"role": "user", "content": "Read both values"},
                    {"role": "assistant", "tool_calls": [call("first"), call("second")]},
                    {"role": "tool", "tool_call_id": "second", "content": "second-result"},
                    {"role": "tool", "tool_call_id": "first", "content": "first-result"}]
        rows, _ = harvest.extract_rows([event(1, "tool_call", {"messages": messages})])
        self.assertEqual(rows[0]["quality"]["parallel_groups"], 1)
        self.assertEqual([m["tool_call_id"] for m in rows[0]["messages"] if m["role"] == "tool"], ["second", "first"])

    def test_error_retry_sequence_is_preserved_and_success_requires_observed_code(self):
        messages = captured_messages()[:3]
        messages[2]["content"] = '{"exit_code":1,"error":"missing dependency"}'
        messages += [{"role": "assistant", "tool_calls": [call("retry", "fixed") ]},
                     {"role": "tool", "tool_call_id": "retry", "content": '{"exit_code":0}'}]
        rows, _ = harvest.extract_rows([event(1, "tool_call", {"messages": messages})])
        self.assertEqual([o["status"] for o in rows[0]["quality"]["outcomes"]], ["failure", "success"])
        self.assertEqual(rows[0]["quality"]["failure_then_same_tool_success"], 1)
        self.assertEqual(harvest.result_status("apparently successful prose"), "unknown")

    def test_orphan_duplicate_missing_and_interrupted_results_fail_closed(self):
        fixtures = [
            [{"role": "tool", "tool_call_id": "no-call", "content": "result"}],
            captured_messages()[:3] + [captured_messages()[2]],
            captured_messages()[:2],
            captured_messages()[:2] + [{"role": "user", "content": "interrupt"}],
        ]
        for messages in fixtures:
            with self.subTest(messages=len(messages)):
                with self.assertRaises(ValueError):
                    harvest.pairing_and_schema(harvest.normalize_messages(messages), [])

    def test_captured_schema_validates_argument_type_and_required_properties(self):
        rows, _ = harvest.extract_rows([event(1, "tool_call", {"messages": captured_messages(), "tools": TOOLS})])
        try:
            import jsonschema
        except ImportError:
            self.assertFalse(rows[0]["quality"]["train_ready"])
        else:
            self.assertTrue(rows[0]["quality"]["format_ready"])
            self.assertFalse(rows[0]["quality"]["train_ready"])
            messages = captured_messages()
            messages[1]["tool_calls"][0]["function"]["arguments"] = {"value": 7}
            bad, _ = harvest.extract_rows([event(2, "tool_call", {"messages": messages, "tools": TOOLS})])
            self.assertFalse(bad[0]["quality"]["schema_complete"])

    def test_external_schema_reference_is_never_fetched(self):
        tools = [{"type": "function", "function": {"name": "fixture_tool", "parameters": {"$ref": "https://invalid.example/schema"}}}]
        rows, _ = harvest.extract_rows([event(1, "tool_call", {"messages": captured_messages(), "tools": tools})])
        self.assertFalse(rows[0]["quality"]["schema_complete"])

    def test_text_repair_summary_requires_same_turn_prompt_and_has_no_tool_schema_gate(self):
        events = [event(1, "user_prompt", {"prompt": "Describe the repair and verification", "turn_id": "text-turn"}),
                  event(2, "session_stop", {"last_assistant_message": "The parser now rejects unmatched results; verified with two tests.", "turn_id": "text-turn"})]
        rows, _ = harvest.extract_rows(events)
        self.assertEqual(rows[0]["route"], "text_reasoning")
        self.assertTrue(rows[0]["quality"]["prompt_linked"])
        self.assertFalse(rows[0]["quality"]["context_complete"])
        self.assertFalse(rows[0]["quality"]["train_ready"])
        self.assertEqual(rows[0]["supervision"]["message_mask"], [0, 1])

    def test_old_original_timestamp_is_rejected_even_if_ingested_in_2026(self):
        sample = event(1, "session_stop", {"timestamp": "2024-12-31T23:59:59Z", "last_assistant_message": "outside range"})
        rows, report = harvest.extract_rows([sample])
        self.assertEqual(rows, [])
        self.assertEqual(report["rejected"]["source_date_outside_2025_2026"], 1)

    def test_exact_eval_prompt_overlap_is_excluded(self):
        sample = event(1, "tool_call", {"messages": captured_messages()})
        excluded = {harvest.normalize_text(captured_messages()[0]["content"])}
        rows, report = harvest.extract_rows([sample], excluded_prompts=excluded)
        self.assertEqual(rows, [])
        self.assertEqual(report["rejected"]["frozen_eval_prompt_overlap"], 1)

    def test_session_and_duplicate_connected_component_stays_in_one_split(self):
        rows, _ = harvest.extract_rows([event(1, "tool_call", {"messages": captured_messages()}, "session-A"),
                                        event(2, "tool_call", {"messages": captured_messages()}, "session-B"),
                                        event(3, "tool_call", {"messages": captured_messages("different")}, "session-B")])
        grouped, report = harvest.group_and_split(rows)
        self.assertEqual(len({r["provenance"]["group_id"] for r in grouped}), 1)
        self.assertEqual(len({r["provenance"]["split"] for r in grouped}), 1)
        self.assertEqual(report["exact_duplicate_rows"], 1)
        self.assertEqual(grouped[1]["quality"]["dedup_exclusion"], "exact_duplicate")

    def test_near_duplicate_different_sessions_stay_in_one_split(self):
        text = " ".join(f"word{i}" for i in range(120))
        rows, _ = harvest.extract_rows([event(1, "session_stop", {"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "Response"}]}, "A"),
                                        event(2, "session_stop", {"messages": [{"role": "user", "content": text + " extra"}, {"role": "assistant", "content": "Response"}]}, "B")])
        grouped, report = harvest.group_and_split(rows)
        self.assertEqual(grouped[0]["provenance"]["split"], grouped[1]["provenance"]["split"])
        self.assertEqual(report["near_duplicate_pairs"], 1)

    def test_artifacts_contain_only_redacted_context_and_ready_is_separate(self):
        token = "sk-" + "x" * 32
        messages = captured_messages()
        messages[0]["content"] += " Authorization: Bearer " + token
        rows, report = harvest.extract_rows([event(1, "tool_call", {"messages": messages})])
        rows, grouping = harvest.group_and_split(rows)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            receipt = harvest.write_artifacts(out, rows, {**report, "grouping": grouping})
            self.assertNotIn(token, (out / "candidates.jsonl").read_text())
            self.assertEqual(receipt["ready_rows"], 0)
            self.assertEqual((out / "ready.jsonl").read_text(), "")
            with self.assertRaises(FileExistsError):
                harvest.write_artifacts(out, rows, report)

    def test_database_connection_uses_readonly_transaction_and_parameterized_bounds(self):
        class Transaction:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
        conn = types.SimpleNamespace(set_type_codec=mock.AsyncMock(), fetchval=mock.AsyncMock(return_value=123),
                                     fetch=mock.AsyncMock(return_value=[]), transaction=mock.Mock(return_value=Transaction()),
                                     terminate=mock.Mock())
        connect = mock.AsyncMock(return_value=conn)
        with mock.patch.dict("sys.modules", {"asyncpg": types.SimpleNamespace(connect=connect)}), \
             mock.patch.dict("os.environ", {"MEMORY_DB_PASSWORD": "fixture-" + hashlib.sha256(b"test-only").hexdigest()}):
            _, query = asyncio.run(harvest.fetch_events(25, harvest.MIN_DATE, harvest.MAX_DATE, 100))
        self.assertEqual(query["query_max_event_id"], 100)
        conn.transaction.assert_called_once_with(isolation="repeatable_read", readonly=True)
        self.assertEqual(connect.call_args.kwargs["server_settings"]["statement_timeout"], "15000")
        self.assertIn("$6", conn.fetch.call_args.args[0])
        self.assertEqual(conn.fetch.call_args.args[6], 25)
        self.assertIn("33554432", conn.fetch.call_args.args[0])
        conn.terminate.assert_called_once()

    def test_body_free_diagnostics_show_missing_prompt_turn(self):
        samples = [event(1, "user_prompt", {"prompt": "never expose this body"}),
                   event(2, "session_stop", {"turn_id": "turn", "last_assistant_message": "never expose this answer"})]
        diagnostic = harvest.inspect_events(samples, {})
        self.assertEqual(diagnostic["context_diagnostics"]["user_prompt:turn_present"], 0)
        self.assertEqual(diagnostic["context_diagnostics"]["session_stop:exact_prompt_matches"], 0)
        self.assertNotIn("never expose", harvest.canonical(diagnostic))

    def test_context_identity_allows_source_space_change_but_rejects_conflicts(self):
        prompt = event(1, "user_prompt", {"prompt": "Read a value", "turn-id": "t"})
        stopped = event(2, "session_stop", {"last_assistant_message": "Read complete", "turn-id": "t"})
        stopped.update(source="other_ingress", space="other_space")
        rows, _ = harvest.extract_rows([prompt, stopped])
        self.assertEqual(rows[0]["quality"]["context_linkage"], "exact_turn")
        stopped = event(2, "session_stop", {"last_assistant_message": "Read complete", "session_id": "conflicting"})
        rows, report = harvest.extract_rows([stopped])
        self.assertFalse(rows)
        self.assertEqual(report["rejected"]["recorded_identity_conflict"], 1)

    def test_anthropic_block_order_and_failure_flag_are_preserved(self):
        messages = [{"role": "user", "content": "Read then explain"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "recorded", "name": "fixture_tool", "input": {"value": "read"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "recorded", "content": "failed read", "is_error": True},
                                                  {"type": "text", "text": "Explain the error"}]},
                    {"role": "assistant", "content": "The read failed."}]
        rows, _ = harvest.extract_rows([event(1, "tool_call", {"messages": messages})])
        self.assertEqual([m["role"] for m in rows[0]["messages"]], ["user", "assistant", "tool", "user", "assistant"])
        self.assertEqual(rows[0]["quality"]["outcomes"][0]["status"], "failure")

    def test_parallel_same_tool_failure_and_success_is_not_recovery(self):
        messages = [{"role": "user", "content": "Read two values"},
                    {"role": "assistant", "tool_calls": [call("a"), call("b")]},
                    {"role": "tool", "tool_call_id": "a", "content": '{"exit_code":1}'},
                    {"role": "tool", "tool_call_id": "b", "content": '{"exit_code":0}'}]
        rows, _ = harvest.extract_rows([event(1, "tool_call", {"messages": messages})])
        self.assertEqual(rows[0]["quality"]["failure_then_same_tool_success"], 0)

    def test_user_after_assistant_redacted_or_empty_target_never_passes_format_gate(self):
        samples = [ [{"role": "assistant", "content": "Answer"}, {"role": "user", "content": "Prompt"}],
                    [{"role": "user", "content": "Prompt"}, {"role": "assistant", "content": "Password=[REDACTED]"}],
                    [{"role": "user", "content": "Prompt"}, {"role": "assistant", "content": ""}] ]
        for messages in samples:
            rows, _ = harvest.extract_rows([event(1, "session_stop", {"messages": messages})])
            self.assertFalse(rows[0]["quality"]["format_ready"])
            self.assertFalse(rows[0]["quality"]["train_ready"])

    def test_camelcase_and_legacy_secret_keys_are_redacted(self):
        clean = harvest.redact({key: "fixture-secret" for key in ("apiKey", "clientSecret", "token", "credentials", "dsn", "passphrase")})
        self.assertTrue(all(value == harvest.REDACTED for value in clean.values()))

    def test_malformed_call_rejects_one_row_without_aborting_tranche(self):
        messages = captured_messages()
        messages[1]["tool_calls"] = ["invalid"]
        rows, report = harvest.extract_rows([event(1, "tool_call", {"messages": messages}),
                                            event(2, "session_stop", {"messages": [{"role": "user", "content": "Prompt"}, {"role": "assistant", "content": "Answer"}]})])
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["rejected"]["invalid_tool_call"], 1)

    def test_later_complete_trace_is_not_hidden_by_candidate_cap(self):
        events = [event(1, "session_stop", {"last_assistant_message": "Incomplete"}),
                  event(2, "session_stop", {"messages": [{"role": "user", "content": "Prompt"}, {"role": "assistant", "content": "Complete"}]})]
        rows, report = harvest.extract_rows(events, max_rows=1)
        self.assertEqual(rows[0]["provenance"]["source_ids"], [2])
        self.assertEqual(report["rejected"]["row_limit_excluded"], 1)

    def test_near_frozen_eval_prompt_is_excluded(self):
        prompt = " ".join(f"word{i}" for i in range(100))
        sample = event(1, "session_stop", {"messages": [{"role": "user", "content": prompt + " extra"}, {"role": "assistant", "content": "Answer"}]})
        rows, report = harvest.extract_rows([sample], excluded_prompts={harvest.normalize_text(prompt)})
        self.assertFalse(rows)
        self.assertEqual(report["rejected"]["frozen_eval_prompt_overlap"], 1)

    def test_individual_review_requires_exact_source_ids_and_archive_hashes(self):
        rows, _ = harvest.extract_rows([event(1, "session_stop", {"messages": [{"role": "user", "content": "Prompt"}, {"role": "assistant", "content": "Answer"}]})])
        row = rows[0]
        decision = {"id": row["id"], "source_ids": row["provenance"]["source_ids"],
                    "row_sha256": harvest.review_digest(row),
                    "archive_payload_sha256s": row["provenance"]["archive_payload_sha256s"], "decision": "approve", "reason": "Fixture reviewed"}
        manifest = {"schema": "recovery_memory_review.v1", "reviewer": "fixture-reviewer", "reviewed_at": "2026-10-13T00:00:00Z", "rows": [decision]}
        self.assertTrue(harvest.apply_reviews(rows, manifest)[0]["quality"]["train_ready"])
        decision["archive_payload_sha256s"] = ["0" * 64]
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            harvest.apply_reviews(rows, manifest)

    def test_legacy_json_text_secret_is_redacted_before_artifact(self):
        legacy = '{"password": "fixture-secret", "token": "fixture-token"}'
        self.assertNotIn("fixture-secret", harvest.redact(legacy))
        self.assertNotIn("fixture-token", harvest.redact(legacy))


if __name__ == "__main__":
    unittest.main()
