"""Read-only, bounded PostgreSQL trace harvest for Chimera recovery.

Never reads local harness transcripts. Credentials stay inside the process.
Archives must match their stored SHA-256 before any body is consumed.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib

EVENT_TYPES = ["user_prompt", "tool_call", "tool_call_failure", "tool_start", "assistant_message", "session_stop"]
MIN_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
MAX_DATE = datetime(2027, 1, 1, tzinfo=timezone.utc)
ARCHIVE_BYTE_BUDGET = 32 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


REDACTED = "[REDACTED]"
SECRET_KEY = re.compile(r"(?:^|[_-])(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|private[_-]?key|credential)(?:$|[_-])", re.I)
SENSITIVE_NAMES = {"apikey", "auth", "auth_token", "authorization", "bearer", "client_secret", "connection_uri",
                   "cookie", "credentials", "dsn", "passphrase", "passwd", "password", "private_key",
                   "proxy_authorization", "refresh_token", "secret", "set_cookie", "token", "access_token"}
SECRET_TEXT = [
    re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9+/=_\-.]{8,}"),
    re.compile(r"(?i)(?:authorization|x-api-key|cookie|set-cookie)\s*:\s*[^\r\n'\"]+"),
    re.compile(r"(?i)\b(?:password|passwd|pwd|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|PGPASSWORD|MEMORY_DB_PASSWORD|credentials?|dsn|passphrase|auth[_-]?token|token)[\"']?\s*[:=]\s*(?:\"[^\"]+\"|'[^']+'|[^\s,;\"'}]+)"),
    re.compile(r"(?i)--(?:password|api-key|token|secret)\s+(?:\"[^\"]+\"|'[^']+'|[^\s,;]+)"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@"),
    re.compile(r"data:[^;,\s]+;base64,[A-Za-z0-9+/=]+"),
]
MARKER = re.compile(r"\[redacted\]|<redacted(?:[-_ :][^>]*)?>", re.I)
OMISSION_MARKER = re.compile(r"<agent-memory[^>]*(?:truncat|omitted)|\"omitted_items\"\s*:", re.I)


def sensitive_key(key):
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    name = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    return name in SENSITIVE_NAMES or bool(SECRET_KEY.search(name)) or name.endswith(("_credentials", "_passphrase", "_auth_token"))


def redact(value):
    """Second redaction pass, including legacy text; no secret originals are retained."""
    if isinstance(value, dict):
        return {str(k): REDACTED if sensitive_key(k) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for pattern in SECRET_TEXT:
            value = pattern.sub(REDACTED, value)
        return value
    return value


def first_string(payload, *keys):
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def normalize_tools(tools):
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            raise ValueError("invalid_tool_schema")
        if tool.get("type") == "function":
            function = tool.get("function", {})
        elif "name" in tool and "input_schema" in tool:
            function = {"name": tool["name"], "parameters": tool["input_schema"]}
            if "description" in tool:
                function["description"] = tool["description"]
        else:
            raise ValueError("unrecognized_tool_schema")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not isinstance(function.get("parameters"), dict):
            raise ValueError("invalid_tool_schema")
        out.append({"type": "function", "function": function})
    if len({t["function"]["name"] for t in out}) != len(out):
        raise ValueError("duplicate_tool_schema")
    return out


def normalize_call(call):
    if not isinstance(call, dict):
        raise ValueError("invalid_tool_call")
    if call.get("type") == "tool_use":
        name, call_id, arguments = call.get("name"), call.get("id"), call.get("input")
    else:
        if call.get("type", "function") != "function":
            raise ValueError("unsupported_custom_tool_call")
        function = call.get("function", {})
        if not isinstance(function, dict):
            raise ValueError("invalid_tool_call")
        name, call_id, arguments = function.get("name"), call.get("id"), function.get("arguments")
    if not isinstance(name, str) or not re.fullmatch(r"[\w./-]{1,128}", name):
        raise ValueError("invalid_tool_name")
    if not isinstance(call_id, str) or not call_id.strip() or len(call_id) > 256:
        raise ValueError("missing_recorded_call_id")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool_arguments_not_json_object")
    if MARKER.search(canonical(arguments)) or OMISSION_MARKER.search(canonical(arguments)):
        raise ValueError("redacted_assistant_call")
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": canonical(arguments)}}


def normalize_messages(messages):
    """Convert only captured chat/tool blocks. Never synthesize calls or call IDs."""
    out = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError("invalid_message_role")
        role, content = message["role"], message.get("content", "")
        if isinstance(content, list):
            text, calls, results = [], [], []
            for block in content:
                if not isinstance(block, dict):
                    raise ValueError("invalid_content_block")
                kind = block.get("type")
                if kind == "text":
                    text.append(block.get("text", ""))
                elif kind == "tool_use" and role == "assistant":
                    calls.append(normalize_call(block))
                elif kind == "tool_result" and role == "user":
                    if text:
                        out.append({"role": "user", "content": "\n".join(text)})
                        text = []
                    if not isinstance(block.get("tool_use_id"), str):
                        raise ValueError("missing_recorded_result_id")
                    result_content = block.get("content", "")
                    result = {"role": "tool", "tool_call_id": block["tool_use_id"],
                              "content": result_content if isinstance(result_content, str) else canonical(result_content)}
                    if "is_error" in block:
                        result["is_error"] = block["is_error"]
                    out.append(result)
                else:
                    raise ValueError("unsupported_content_block")
            if text or calls:
                record = {"role": role, "content": "\n".join(text)}
                if calls:
                    record["tool_calls"] = calls
                out.append(record)
            out.extend(results)
            continue
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("message_content_not_text")
        record = {"role": role, "content": content}
        if role == "assistant" and message.get("tool_calls"):
            record["tool_calls"] = [normalize_call(c) for c in message["tool_calls"]]
        if role == "tool":
            if not isinstance(message.get("tool_call_id"), str):
                raise ValueError("missing_recorded_result_id")
            record["tool_call_id"] = message["tool_call_id"]
            if "is_error" in message:
                record["is_error"] = message["is_error"]
        out.append(record)
    return out


def pairing_and_schema(messages, tools):
    pending, seen, call_count, parallel_groups = {}, set(), 0, 0
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    missing_schemas, invalid_schemas = set(), set()
    for message in messages:
        if message["role"] == "assistant":
            if pending:
                raise ValueError("assistant_before_all_results")
            calls = message.get("tool_calls", [])
            parallel_groups += int(len(calls) > 1)
            for call in calls:
                call_id, function = call["id"], call["function"]
                if call_id in seen:
                    raise ValueError("duplicate_call_id")
                pending[call_id] = function["name"]
                seen.add(call_id)
                call_count += 1
                schema = schemas.get(function["name"])
                if schema is None:
                    missing_schemas.add(function["name"])
                else:
                    try:
                        from jsonschema import Draft202012Validator
                        Draft202012Validator.check_schema(schema)
                        # References must not fetch remote schemas or documents.
                        if "$ref" in canonical(schema) or "$dynamicRef" in canonical(schema):
                            invalid_schemas.add("external_or_unresolved_schema_reference")
                        elif not Draft202012Validator(schema).is_valid(json.loads(function["arguments"])):
                            invalid_schemas.add("arguments_fail_captured_schema")
                    except ImportError:
                        invalid_schemas.add("schema_validator_unavailable")
                    except Exception:
                        invalid_schemas.add("invalid_captured_schema")
        elif message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending:
                raise ValueError("orphan_or_duplicate_tool_result")
            del pending[call_id]
        elif pending:
            raise ValueError("interrupted_tool_exchange")
    if pending:
        raise ValueError("missing_tool_result")
    return {"call_count": call_count, "parallel_groups": parallel_groups,
            "schema_complete": not missing_schemas and not invalid_schemas,
            "missing_schema_names": sorted(missing_schemas), "schema_errors": sorted(invalid_schemas)}


def session_key(event):
    identity = event.get("metadata", {}).get("identity", {})
    session = event.get("session_id") or identity.get("session_id")
    if not session:
        raise ValueError("missing_session_identity")
    # Ingress source and project space can change during one recorded session.
    # Keep every occurrence of an agent/session identity in one split.
    return sha([event.get("agent"), session])


def event_turn(event, payload):
    return first_string(payload, "turn_id", "turnId", "turn-id") or first_string(event.get("metadata", {}), "turn_id", "turnId", "turn-id")


def verify_recorded_identity(event, payload):
    """Reject disagreeing archive/event identity fields without exposing values."""
    metadata = event.get("metadata", {})
    identity = metadata.get("identity", {})
    groups = [
        [event.get("session_id"), first_string(payload, "session_id", "sessionId", "conversation_id", "conversationId"), identity.get("session_id")],
        [first_string(payload, "turn_id", "turnId", "turn-id"), first_string(metadata, "turn_id", "turnId", "turn-id")],
        [first_string(payload, "thread_id", "threadId", "thread-id"), first_string(metadata, "thread_id", "threadId", "thread-id"), identity.get("thread_id")],
    ]
    if any(len({v for v in values if isinstance(v, str) and v}) > 1 for values in groups):
        raise ValueError("recorded_identity_conflict")


def recorded_date(event, payload):
    for key in ("timestamp", "recorded_at", "created_at"):
        value = payload.get(key)
        if isinstance(value, str):
            try:
                observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if observed.tzinfo is not None:
                    if not MIN_DATE <= observed < MAX_DATE:
                        raise ValueError("source_date_outside_2025_2026")
                    return observed.isoformat(), "payload_timestamp"
            except ValueError as exc:
                if str(exc) == "source_date_outside_2025_2026":
                    raise
    created = event["created_at"]
    if isinstance(created, str):
        created = datetime.fromisoformat(created.replace("Z", "+00:00"))
    if not MIN_DATE <= created < MAX_DATE:
        raise ValueError("source_date_outside_2025_2026")
    return created.isoformat(), "ingestion_timestamp_only"


def event_provenance(events):
    dates = [recorded_date(e, p) for e, p in events]
    return {"source": "agent_memory_postgres", "source_verified": True,
            "source_ids": sorted({e["id"] for e, _ in events}),
            "archive_ids": sorted({e["archive_id"] for e, _ in events}),
            "archive_payload_sha256s": sorted({e["payload_sha256"] for e, _ in events}),
            "archive_integrity": "event_relation_and_sha256_verified",
            "archive_redacted_flags": sorted({bool(e.get("redacted")) for e, _ in events}),
            "archive_redaction_versions": sorted({str((e.get("archive_metadata") or {}).get("redaction_version", "unrecorded")) for e, _ in events}),
            "archive_payload_kinds": sorted({e.get("payload_kind", "hook_payload") for e, _ in events}),
            "original_context_verified": False,
            "source_dates": sorted({d for d, _ in dates}),
            "source_date_basis": sorted({basis for _, basis in dates}),
            "ingress_sources": sorted({e.get("source", "") for e, _ in events}),
            "spaces": sorted({e.get("space", "") for e, _ in events}),
            "session_group_sha256s": sorted({session_key(e) for e, _ in events}),
            "group_id": "", "split": ""}


def result_status(result, failed=False):
    if failed:
        return "failure"
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return "unknown"
    if isinstance(result, dict):
        if result.get("is_error") is True or result.get("isError") is True or result.get("error"):
            return "failure"
        code = result.get("exit_code", result.get("exitCode", result.get("returncode")))
        if isinstance(code, int) and not isinstance(code, bool):
            return "success" if code == 0 else "failure"
        if result.get("success") is True:
            return "success"
    return "unknown"


def make_row(messages, tools, sources, *, context_linkage, capture_kind, outcomes=None, max_chars=100000):
    if len(canonical(messages)) > max_chars:
        raise ValueError("row_size_limit")
    messages, tools = normalize_messages(redact(messages)), normalize_tools(redact(tools))
    shape = pairing_and_schema(messages, tools)
    if outcomes is None:
        outcomes = [{"tool_call_id": m["tool_call_id"], "status": result_status(m["content"], m.get("is_error") is True)}
                    for m in messages if m["role"] == "tool"]
    statuses = {o["tool_call_id"]: o["status"] for o in outcomes}
    failed_tools, recoveries, call_groups = {}, 0, {}
    group_number = 0
    for message in messages:
        if message.get("tool_calls"):
            group_number += 1
        for call in message.get("tool_calls", []):
            call_groups[call["id"]] = (call["function"]["name"], group_number)
        if message["role"] == "tool":
            name, group = call_groups[message["tool_call_id"]]
            status = statuses.get(message["tool_call_id"], "unknown")
            if name in failed_tools and failed_tools[name] < group and status == "success":
                recoveries += 1
                del failed_tools[name]
            if status == "failure":
                failed_tools[name] = group
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_indices:
        raise ValueError("no_assistant_target")
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user" and m["content"].strip()), None)
    user_precedes_targets = first_user is not None and all(i > first_user for i in assistant_indices)
    # Matching a turn identifies the prompt; it does not reconstruct the model's
    # preceding system, developer, or tool history. Hook fragments stay partial.
    context_omitted = bool(OMISSION_MARKER.search(canonical(messages)))
    context_complete = user_precedes_targets and context_linkage == "captured_messages" and not context_omitted
    target_redacted = any(MARKER.search(m["content"]) for m in messages if m["role"] == "assistant")
    meaningful_targets = all(m["content"].strip() or m.get("tool_calls") for m in messages if m["role"] == "assistant")
    format_ready = bool(context_complete and shape["schema_complete"] and not target_redacted and meaningful_targets)
    route = "tool_calling" if shape["call_count"] else "text_reasoning"
    provenance = event_provenance(sources)
    row = {"schema": "recovery_corpus.v1", "id": "memory-" + sha([provenance["source_ids"], messages])[:24],
           "route": route, "messages": messages, "tools": tools, "media": [], "provenance": provenance,
           "supervision": {"policy": "assistant_only", "message_mask": [int(m["role"] == "assistant") for m in messages]},
           "supervised_assistant_messages": assistant_indices,
           "quality": {**shape, "context_complete": context_complete, "context_linkage": context_linkage,
                       "capture_kind": capture_kind, "outcomes": outcomes, "failure_then_same_tool_success": recoveries,
                       "prompt_linked": user_precedes_targets, "assistant_target_redacted": target_redacted,
                       "context_omitted": context_omitted, "format_ready": format_ready,
                       "content_review": "unreviewed",
                       "train_ready": False}}
    return row


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("operation_deadline")


def extract_rows(events, *, external_tools=None, catalog_provenance=None, max_rows=1000, excluded_prompts=None, deadline=None):
    """Pair calls within their authenticated archive, never by adjacent database rows.

    Hook rows without a matching exact-turn prompt remain explicitly non-ready.
    Retry/parallel order is preserved only from recorded message arrays.
    """
    hydrated, counts, contexts, rows = [], Counter(), {}, []
    for event in events:
        check_deadline(deadline)
        if event.get("payload") is None:
            counts["missing_or_oversize_archive"] += 1
            continue
        try:
            if event["archive_event_id"] != event["id"] or event["archive_event_uid"] != event["event_uid"]:
                raise ValueError("archive_event_identity_mismatch")
            payload = archive_payload(event)
            if not isinstance(payload, dict):
                raise ValueError("payload_not_object")
            verify_recorded_identity(event, payload)
            session = session_key(event)
            recorded_date(event, payload)
            hydrated.append((event, payload))
            if event["event_type"] == "user_prompt":
                prompt = first_string(payload, "prompt", "userPrompt", "message", "userMessage", "content")
                turn = event_turn(event, payload)
                if prompt and turn:
                    contexts.setdefault((session, turn), []).append((event, payload, prompt))
        except (ValueError, TypeError, UnicodeError, zlib.error) as exc:
            reason = str(exc) if str(exc) in ("source_date_outside_2025_2026", "archive_event_identity_mismatch", "missing_session_identity", "recorded_identity_conflict") else "archive_verification_failed"
            counts[reason] += 1
    # Inspect the full bounded window before selecting rows, so a long run of
    # incomplete hooks cannot hide a complete trace later in the same window.
    for event, payload in hydrated:
        check_deadline(deadline)
        kind = event["event_type"]
        if kind not in ("tool_call", "tool_call_failure", "assistant_message", "session_stop"):
            continue
        try:
            sources, outcomes = [(event, payload)], None
            tools = payload.get("tools", external_tools or [])
            if isinstance(payload.get("messages"), list):
                messages, linkage, capture = payload["messages"], "captured_messages", "archived_messages"
            else:
                session, turn = session_key(event), event_turn(event, payload)
                matches = contexts.get((session, turn), []) if turn else []
                prompt = first_string(payload, "prompt", "userPrompt")
                if prompt and matches and any(normalize_text(prompt) != normalize_text(match[2]) for match in matches):
                    raise ValueError("embedded_prompt_conflict")
                linkage = "embedded_prompt" if prompt else "missing"
                if not prompt and len(matches) == 1:
                    prompt = matches[0][2]
                    sources.append((matches[0][0], matches[0][1]))
                    linkage = "exact_turn"
                elif not prompt and len(matches) > 1:
                    raise ValueError("ambiguous_prompt_for_turn")
                messages = [{"role": "user", "content": prompt}] if prompt else []
                if kind in ("tool_call", "tool_call_failure"):
                    name = first_string(payload, "tool_name", "toolName") or event["metadata"].get("tool_name")
                    call_id = first_string(payload, "tool_use_id", "toolUseId", "call_id", "callId") or event["metadata"].get("tool_use_id") or event["metadata"].get("call_id")
                    arguments = payload.get("tool_input", payload.get("toolInput", payload.get("arguments")))
                    result_key = next((k for k in ("tool_response", "toolResponse", "tool_result", "toolResult", "result", "response", "error") if k in payload), None)
                    if result_key is None:
                        raise ValueError("missing_recorded_tool_outcome")
                    result = payload[result_key]
                    call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
                    messages.extend([{"role": "assistant", "content": "", "tool_calls": [call]},
                                     {"role": "tool", "tool_call_id": call_id, "content": result if isinstance(result, str) else canonical(result)}])
                    outcomes = [{"tool_call_id": call_id, "status": result_status(result, kind == "tool_call_failure")}]
                    capture = "single_hook_call_and_result"
                else:
                    text = first_string(payload, "last_assistant_message", "lastAssistantMessage", "message", "content")
                    if not text:
                        raise ValueError("missing_assistant_text")
                    messages.append({"role": "assistant", "content": text})
                    capture = "recorded_assistant_text"
            row = make_row(messages, tools, sources, context_linkage=linkage, capture_kind=capture, outcomes=outcomes)
            if "tools" not in payload and external_tools:
                row["provenance"]["tool_catalog"] = catalog_provenance
                row["quality"]["schema_source"] = "operator_reviewed_catalog"
            elif tools:
                row["quality"]["schema_source"] = "verified_archive"
            else:
                row["quality"]["schema_source"] = "missing"
            prompts = [normalize_text(m["content"]) for m in row["messages"] if m["role"] == "user"]
            if excluded_prompts and any(prompt_overlap(p, excluded_prompts) for p in prompts):
                raise ValueError("frozen_eval_prompt_overlap")
            rows.append(row)
        except (ValueError, TypeError, KeyError) as exc:
            # Whitelist reason codes; no rejected body or exception value enters evidence.
            known = {"missing_recorded_tool_outcome", "missing_recorded_call_id", "tool_arguments_not_json_object",
                     "redacted_assistant_call", "ambiguous_prompt_for_turn", "embedded_prompt_conflict", "invalid_tool_call", "missing_assistant_text", "frozen_eval_prompt_overlap",
                     "row_size_limit", "unsupported_content_block", "orphan_or_duplicate_tool_result", "missing_tool_result",
                     "assistant_before_all_results", "duplicate_call_id", "interrupted_tool_exchange", "invalid_tool_name",
                     "missing_recorded_result_id", "unrecognized_tool_schema", "invalid_tool_schema", "unsupported_custom_tool_call"}
            counts[str(exc) if str(exc) in known else "invalid_trace_structure"] += 1
    rows.sort(key=lambda row: (not row["quality"]["format_ready"], not row["quality"]["context_complete"], min(row["provenance"]["source_ids"])))
    if len(rows) > max_rows:
        counts["row_limit_excluded"] = len(rows) - max_rows
        rows = rows[:max_rows]
    return rows, {"events_considered": len(events), "archives_verified": len(hydrated), "candidate_rows": len(rows),
                  "rejected": dict(counts)}


def apply_reviews(rows, manifest):
    """Approve only individually reviewed, byte-bound rows; never approve a batch by default."""
    if manifest.get("schema") != "recovery_memory_review.v1" or not isinstance(manifest.get("reviewer"), str) or not manifest["reviewer"].strip():
        raise ValueError("review_manifest_invalid")
    reviewed_at = datetime.fromisoformat(manifest.get("reviewed_at", "").replace("Z", "+00:00"))
    if reviewed_at.tzinfo is None or not MIN_DATE <= reviewed_at < MAX_DATE:
        raise ValueError("review_date_invalid")
    decisions = manifest.get("rows")
    if not isinstance(decisions, list) or len(decisions) > 1000:
        raise ValueError("review_manifest_invalid")
    by_id = {row["id"]: row for row in rows}
    seen = set()
    for decision in decisions:
        if not isinstance(decision, dict) or decision.get("id") not in by_id or decision["id"] in seen:
            raise ValueError("review_row_not_unique_or_missing")
        seen.add(decision["id"])
        row = by_id[decision["id"]]
        provenance, quality = row["provenance"], row["quality"]
        if (decision.get("row_sha256") != review_digest(row) or sorted(decision.get("source_ids", [])) != provenance["source_ids"] or
                sorted(decision.get("archive_payload_sha256s", [])) != provenance["archive_payload_sha256s"]):
            raise ValueError("review_source_hash_mismatch")
        if decision.get("decision") not in ("approve", "reject") or not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            raise ValueError("review_decision_invalid")
        quality["content_review"] = "approved" if decision["decision"] == "approve" else "rejected"
        provenance["content_review"] = {"reviewer": redact(manifest["reviewer"]), "reviewed_at": reviewed_at.isoformat(),
                                        "reason": redact(decision["reason"]), "manifest_sha256": sha(manifest)}
        standalone = decision.get("standalone_prompt_context") is True
        quality["reviewed_standalone_context"] = standalone
        approved_context = quality["context_complete"] or (standalone and quality["prompt_linked"] and not quality["context_omitted"])
        meaningful = all(m["content"].strip() or m.get("tool_calls") for m in row["messages"] if m["role"] == "assistant")
        quality["train_ready"] = bool(quality["content_review"] == "approved" and approved_context and
                                      quality["schema_complete"] and not quality["assistant_target_redacted"] and meaningful)
    return rows


def review_digest(row):
    return sha({key: row[key] for key in ("id", "route", "messages", "tools", "media", "supervision")})


def normalize_text(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def prompt_overlap(prompt, excluded, near_threshold=0.85):
    """Protect frozen prompts from punctuation variants and near duplicates."""
    if prompt in excluded:
        return True
    words = prompt.split()
    a = {" ".join(words[j:j + 3]) for j in range(max(0, len(words) - 2))}
    if len(a) < 8:
        return False
    for other in excluded:
        other_words = other.split()
        if len(other_words) >= 8 and (" " + other + " ") in (" " + prompt + " "):
            return True
        b = {" ".join(other_words[j:j + 3]) for j in range(max(0, len(other_words) - 2))}
        if b and min(len(a), len(b)) / max(len(a), len(b)) >= near_threshold and len(a & b) / len(a | b) >= near_threshold:
            return True
    return False


def semantic_content(row):
    messages = []
    for original in row["messages"]:
        message = {k: v for k, v in original.items() if k != "tool_call_id"}
        if message.get("tool_calls"):
            message["tool_calls"] = [{"type": c["type"], "function": c["function"]} for c in message["tool_calls"]]
        messages.append(message)
    return [row["route"], messages, row["tools"]]


def group_and_split(rows, *, near_threshold=0.85, deadline=None):
    """Connected components over sessions and exact/near duplicates prevent leakage."""
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        parents[find(b)] = find(a)
    session_seen, exact_seen, prompt_seen, duplicate_of, shingles, prompt_shingles = {}, {}, {}, {}, [], []
    prompt_duplicates = 0
    for i, row in enumerate(rows):
        check_deadline(deadline)
        for session in row["provenance"]["session_group_sha256s"]:
            if session in session_seen:
                union(i, session_seen[session])
            session_seen[session] = i
        exact = sha(semantic_content(row))
        if exact in exact_seen:
            union(i, exact_seen[exact])
            duplicate_of[i] = exact_seen[exact]
        else:
            exact_seen[exact] = i
        words = normalize_text(canonical(semantic_content(row))).split()
        shingles.append({" ".join(words[j:j + 3]) for j in range(max(0, len(words) - 2))})
        row_prompt_shingles = []
        for message in row["messages"]:
            if message["role"] != "user":
                continue
            prompt = normalize_text(message["content"])
            words = prompt.split()
            if len(words) < 8:
                continue
            if prompt in prompt_seen:
                union(i, prompt_seen[prompt])
                prompt_duplicates += 1
            else:
                prompt_seen[prompt] = i
            row_prompt_shingles.append({" ".join(words[j:j + 3]) for j in range(len(words) - 2)})
        prompt_shingles.append(row_prompt_shingles)
    near_pairs = 0
    # Bounded tranche: exhaustive comparisons catch boundary cases without approximate misses.
    for i, a in enumerate(shingles):
        check_deadline(deadline)
        if len(a) < 8:
            continue
        for j in range(i):
            b = shingles[j]
            semantic_near = bool(b and min(len(a), len(b)) / max(len(a), len(b)) >= near_threshold
                                 and len(a & b) / len(a | b) >= near_threshold)
            prompt_near = any(min(len(prompt_a), len(prompt_b)) / max(len(prompt_a), len(prompt_b)) >= near_threshold
                              and len(prompt_a & prompt_b) / len(prompt_a | prompt_b) >= near_threshold
                              for prompt_a in prompt_shingles[i] for prompt_b in prompt_shingles[j])
            if semantic_near or prompt_near:
                union(i, j)
                near_pairs += 1
    components = {}
    for i, row in enumerate(rows):
        components.setdefault(find(i), set()).update(row["provenance"]["session_group_sha256s"])
    output = []
    for i, row in enumerate(rows):
        group_id = "memory-group-" + sha(sorted(components[find(i)]))[:32]
        bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
        row["provenance"].update(group_id=group_id, split="train" if bucket < 90 else "validation" if bucket < 95 else "test")
        row["quality"]["exact_duplicate_of"] = rows[duplicate_of[i]]["id"] if i in duplicate_of else None
        output.append(row)
    # Exact duplicates stay as audit candidates, but only their representative is ready.
    for i in duplicate_of:
        output[i]["quality"]["train_ready"] = False
        output[i]["quality"]["dedup_exclusion"] = "exact_duplicate"
    return output, {"groups": len(components), "exact_duplicate_rows": len(duplicate_of),
                    "repeated_prompt_occurrences": prompt_duplicates,
                    "near_duplicate_pairs": near_pairs, "split_counts": dict(Counter(r["provenance"]["split"] for r in output)),
                    "near_threshold": near_threshold, "all_related_rows_grouped": True}


def write_artifacts(out, rows, report):
    out.mkdir(parents=True, exist_ok=False)
    ready = [r for r in rows if r["quality"]["train_ready"]]
    artifacts = {}
    for name, values in (("candidates.jsonl", rows), ("ready.jsonl", ready)):
        data = "".join(canonical(r) + "\n" for r in values).encode("utf-8")
        (out / name).write_bytes(data)
        artifacts[name] = {"sha256": hashlib.sha256(data).hexdigest(), "rows": len(values), "bytes": len(data)}
    report.update(artifacts=artifacts, ready_rows=len(ready),
                  routes=dict(Counter(r["route"] for r in rows)),
                  context_complete_rows=sum(r["quality"]["context_complete"] for r in rows),
                  format_ready_rows=sum(r["quality"]["format_ready"] for r in rows),
                  content_review_counts=dict(Counter(r["quality"]["content_review"] for r in rows)),
                  schema_complete_rows=sum(r["quality"]["schema_complete"] for r in rows),
                  observed_outcomes=dict(Counter(o["status"] for r in rows for o in r["quality"]["outcomes"])),
                  parallel_groups=sum(r["quality"]["parallel_groups"] for r in rows),
                  recorded_failure_then_same_tool_success=sum(r["quality"]["failure_then_same_tool_success"] for r in rows),
                  supervision_policy="assistant_only",
                  limitations=["Trace execution is observed, not a content-quality endorsement.",
                               "Insertion time proves ingestion date only when original timestamps are absent.",
                               "Missing authenticated schemas/context are candidates, never training-ready.",
                               "Readiness additionally requires an individual source/hash-bound content review.",
                               "No retry or parallel ordering is inferred from adjacent hook rows."])
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def archive_payload(event, max_bytes=2_000_000):
    blob = bytes(event["payload"])
    if len(blob) != event["compressed_bytes"]:
        raise ValueError("archive_compressed_length_mismatch")
    if event["compression"] == "zlib":
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(blob, max_bytes + 1)
        if not decompressor.eof or decompressor.unused_data or len(raw) > max_bytes:
            raise ValueError("archive_size_or_codec_limit")
    elif event["compression"] == "none":
        raw = blob
    else:
        raise ValueError("archive_codec_unknown")
    if len(raw) > max_bytes or hashlib.sha256(raw).hexdigest() != event["payload_sha256"]:
        raise ValueError("archive_digest_or_size_mismatch")
    text = raw.decode("utf-8")
    if len(text) != event["raw_chars"]:
        raise ValueError("archive_character_count_mismatch")
    if event["payload_format"] != "json":
        raise ValueError("archive_not_json")
    return json.loads(text)


async def fetch_events(limit, since, until, before_id=None, space=None, agent=None, event_types=None):
    """Only SELECTs inside a bounded, repeatable-read, read-only transaction."""
    if not 1 <= limit <= 10000 or since.tzinfo is None or until.tzinfo is None or not MIN_DATE <= since < until <= MAX_DATE:
        raise ValueError("query_bounds_invalid")
    event_types = event_types or EVENT_TYPES
    if not event_types or any(kind not in EVENT_TYPES for kind in event_types):
        raise ValueError("event_type_filter_invalid")
    import asyncpg
    password = os.environ.get("MEMORY_DB_PASSWORD", "")
    if not password:
        manager = Path(os.environ.get("SERVER_MANAGER_SKILL_DIR", "C:/Users/cereal/.agents/skills/server-manager"))
        result = subprocess.run([sys.executable, str(manager / "credman.py"), "get",
                                 os.environ.get("MEMORY_DB_VAULT_KEY", "agent_memory_pg")],
                                capture_output=True, text=True, timeout=15, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError("vault_credential_unavailable")
        password = result.stdout.strip()
    conn = None
    try:
        conn = await asyncpg.connect(host=os.environ.get("MEMORY_DB_HOST", "192.168.50.222"),
                                    port=int(os.environ.get("MEMORY_DB_PORT", "25490")),
                                    database=os.environ.get("MEMORY_DB_NAME", "agent_memory"),
                                    user=os.environ.get("MEMORY_DB_USER", "agent_memory"), password=password,
                                    timeout=10, command_timeout=20,
                                    server_settings={"default_transaction_read_only": "on",
                                                     "application_name": "chimera_memory_harvest",
                                                     "statement_timeout": "15000", "lock_timeout": "2000"})
        password = ""
        await conn.set_type_codec("jsonb", schema="pg_catalog", encoder=json.dumps, decoder=json.loads)
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            snapshot = await conn.fetchval("SELECT COALESCE(max(id), 0) FROM agent_memory_events")
            bound = min(snapshot, before_id) if before_id is not None else snapshot
            records = await conn.fetch("""
                WITH selected AS MATERIALIZED (
                  SELECT * FROM agent_memory_events WHERE id <= $1 AND created_at >= $2 AND created_at < $3
                    AND event_type=ANY($4::text[]) AND ($5::text IS NULL OR space=$5)
                    AND ($7::text IS NULL OR agent=$7)
                  ORDER BY id DESC LIMIT $6
                )
                SELECT e.id, e.event_uid, e.created_at, e.space, e.session_id, e.agent,
                       e.event_type, e.role, e.source, e.metadata,
                       a.id AS archive_id, a.event_id AS archive_event_id, a.event_uid AS archive_event_uid,
                       a.payload_kind, a.payload_sha256, a.payload_format, a.compression,
                       a.raw_chars, a.compressed_bytes, a.redacted, a.metadata AS archive_metadata,
                       CASE WHEN a.raw_chars <= 2000000 AND a.compressed_bytes <= 1000000
                         AND sum(COALESCE(a.compressed_bytes, 0)) OVER (ORDER BY e.id DESC) <= 33554432
                         AND sum(COALESCE(a.raw_chars, 0)) OVER (ORDER BY e.id DESC) <= 33554432
                         THEN a.payload END AS payload
                FROM selected e
                LEFT JOIN LATERAL (
                    SELECT * FROM agent_memory_payload_archives a
                    WHERE a.event_id=e.id AND a.payload_kind='hook_payload'
                      AND (e.metadata->'trace_archive'->>'payload_sha256' IS NULL
                           OR a.payload_sha256=e.metadata->'trace_archive'->>'payload_sha256')
                    ORDER BY a.id DESC LIMIT 1
                ) a ON true
                ORDER BY e.id DESC
            """, bound, since, until, event_types, space, limit, agent)
        return [dict(row) for row in reversed(records)], {"snapshot_max_event_id": snapshot, "query_max_event_id": bound,
                "since": since.isoformat(), "until": until.isoformat(), "requested_limit": limit,
                "agent_filter": agent, "event_type_filter": event_types,
                "compressed_archive_budget_bytes": ARCHIVE_BYTE_BUDGET, "read_only": True}
    finally:
        password = ""
        if conn is not None:
            # A timed-out/cancelled query may make graceful close wait on cancellation.
            conn.terminate()


def inspect_events(events, query):
    keys = Counter()
    shapes = Counter()
    failures = Counter()
    context_shapes, prompt_keys, observed_turns = Counter(), Counter(), []
    for event in events:
        if event.get("payload") is None:
            failures["missing_archive"] += 1
            continue
        try:
            if event.get("archive_event_id") != event["id"] or event.get("archive_event_uid") != event["event_uid"]:
                raise ValueError("archive_event_identity_mismatch")
            payload = archive_payload(event)
            if isinstance(payload, dict):
                verify_recorded_identity(event, payload)
                kind = event["event_type"]
                turn = event_turn(event, payload)
                prompt = first_string(payload, "prompt", "userPrompt", "message", "userMessage", "content") if kind == "user_prompt" else first_string(payload, "prompt", "userPrompt")
                session = session_key(event)
                context_shapes[f"{kind}:turn_present"] += bool(turn)
                context_shapes[f"{kind}:prompt_present"] += bool(prompt)
                context_shapes[f"{kind}:captured_messages"] += isinstance(payload.get("messages"), list)
                if kind == "user_prompt" and prompt and turn:
                    prompt_keys[(session, turn)] += 1
                if kind in ("tool_call", "tool_call_failure", "assistant_message", "session_stop"):
                    observed_turns.append((kind, session, turn, bool(prompt)))
                for key, value in payload.items():
                    keys[str(key)] += 1
                    shapes[f"{key}:{type(value).__name__}"] += 1
                for container in ("payload", "raw_payload", "tool_input", "tool_response", "tool_result", "message"):
                    nested = payload.get(container)
                    if isinstance(nested, dict):
                        for key, value in nested.items():
                            shapes[f"{container}.{key}:{type(value).__name__}"] += 1
        except (ValueError, TypeError, UnicodeError, zlib.error):
            failures["archive_verification_failed"] += 1
    for kind, session, turn, embedded in observed_turns:
        if turn:
            context_shapes[f"{kind}:exact_prompt_matches"] += int(prompt_keys.get((session, turn), 0) == 1)
            context_shapes[f"{kind}:ambiguous_prompt_matches"] += int(prompt_keys.get((session, turn), 0) > 1)
    return {"query": query, "events": len(events), "event_types": dict(Counter(e["event_type"] for e in events)),
            "agents": dict(Counter(e["agent"] for e in events)), "archive_keys": dict(keys),
            "archive_shapes": dict(shapes), "context_diagnostics": dict(context_shapes),
            "verification_failures": dict(failures)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "harvest"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--since", default="2026-10-01T00:00:00+00:00")
    parser.add_argument("--until", default="2027-01-01T00:00:00+00:00")
    parser.add_argument("--before-id", type=int)
    parser.add_argument("--space")
    parser.add_argument("--agent", help="Exact recorded agent name")
    parser.add_argument("--event-types", help="Comma-separated subset of supported event types")
    parser.add_argument("--out", type=Path, help="New artifact directory; existing directories are refused")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--tool-catalog", type=Path, help="Explicit operator-reviewed schema catalog, never inferred from arguments")
    parser.add_argument("--review-manifest", type=Path, help="Individual source/hash-bound content approval; never inferred from execution")
    parser.add_argument("--exclude-prompts", type=Path, help="JSON list of frozen evaluation prompt strings")
    args = parser.parse_args(argv)
    try:
        since, until = datetime.fromisoformat(args.since), datetime.fromisoformat(args.until)
    except ValueError:
        parser.error("dates must be timezone-aware ISO timestamps")
    if since.tzinfo is None or until.tzinfo is None or not MIN_DATE <= since < until <= MAX_DATE or not 1 <= args.limit <= 10000 or not 1 <= args.max_rows <= 1000:
        parser.error("use bounded 2025-2026 UTC dates and limit 1..10000")
    if args.command == "harvest" and (args.out is None or args.out.exists()):
        parser.error("harvest needs --out pointing to a new directory")
    event_types = args.event_types.split(",") if args.event_types else EVENT_TYPES
    if not event_types or any(kind not in EVENT_TYPES for kind in event_types):
        parser.error("event-types must be a subset of " + ",".join(EVENT_TYPES))
    try:
        tools, catalog_provenance = None, None
        if args.tool_catalog:
            raw_catalog = args.tool_catalog.read_bytes()
            catalog = json.loads(raw_catalog)
            provenance = catalog.get("provenance", {})
            if catalog.get("schema") != "recovery_tool_catalog.v1" or provenance.get("authority") != "operator_reviewed" or not isinstance(provenance.get("source"), str) or not re.fullmatch(r"[0-9a-f]{64}", provenance.get("source_sha256", "")):
                raise ValueError("operator_tool_catalog_provenance_required")
            tools = normalize_tools(catalog["tools"])
            catalog_provenance = {**provenance, "catalog_sha256": hashlib.sha256(raw_catalog).hexdigest()}
        excluded = None
        excluded_sha256 = None
        if args.exclude_prompts:
            excluded_bytes = args.exclude_prompts.read_bytes()
            excluded_sha256 = hashlib.sha256(excluded_bytes).hexdigest()
            prompts = json.loads(excluded_bytes)
            if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
                raise ValueError("evaluation_prompts_must_be_string_list")
            excluded = {normalize_text(p) for p in prompts}
        deadline = time.monotonic() + 45
        events, query = asyncio.run(asyncio.wait_for(fetch_events(args.limit, since, until, args.before_id, args.space,
                                                               args.agent, event_types), timeout=45))
        if args.command == "inspect":
            print(json.dumps(inspect_events(events, query), sort_keys=True))
        else:
            rows, report = extract_rows(events, external_tools=tools, catalog_provenance=catalog_provenance,
                                        max_rows=args.max_rows, excluded_prompts=excluded, deadline=deadline)
            if args.review_manifest:
                rows = apply_reviews(rows, json.loads(args.review_manifest.read_text(encoding="utf-8")))
            rows, grouping = group_and_split(rows, deadline=deadline)
            report.update(query=query, grouping=grouping, tool_catalog_provided=bool(tools),
                          frozen_eval_prompts_excluded=len(excluded or []),
                          frozen_eval_prompt_file_sha256=excluded_sha256,
                          generated_at=datetime.now(timezone.utc).isoformat())
            report = write_artifacts(args.out, rows, report)
            print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        # Never print exception values: driver exceptions can contain credentials or bodies.
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
