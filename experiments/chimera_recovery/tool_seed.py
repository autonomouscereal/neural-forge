"""Operator-authored 2026 format episodes with actual deterministic tool results.

Supplemental format recovery only; these are not model outputs or BFCL scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import tempfile

try:
    from .memory_harvest import canonical, sha, normalize_messages, normalize_tools, pairing_and_schema, normalize_text, prompt_overlap
except ImportError:
    from memory_harvest import canonical, sha, normalize_messages, normalize_tools, pairing_and_schema, normalize_text, prompt_overlap


TOOLS = [
    {"type": "function", "function": {"name": "local_arithmetic", "description": "Exact integer arithmetic; division returns a rational string.",
     "parameters": {"type": "object", "properties": {"operation": {"enum": ["add", "subtract", "multiply", "divide"]},
                     "a": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
                     "b": {"type": "integer", "minimum": -1000000, "maximum": 1000000}},
                    "required": ["operation", "a", "b"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "local_text", "description": "Deterministic text transforms; decode_hex requires hexadecimal text.",
     "parameters": {"type": "object", "properties": {"operation": {"enum": ["upper", "lower", "reverse", "strip", "words", "decode_hex"]},
                     "text": {"type": "string", "maxLength": 512}}, "required": ["operation", "text"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "local_file", "description": "Read, write, or append a UTF-8 text file inside this episode's temporary directory.",
     "parameters": {"type": "object", "properties": {"operation": {"enum": ["read", "write", "append"]},
                     "name": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,75}\\.txt$"},
                     "text": {"type": "string", "maxLength": 512}}, "required": ["operation", "name"], "additionalProperties": False,
                    "allOf": [{"if": {"properties": {"operation": {"enum": ["write", "append"]}}}, "then": {"required": ["text"]}}]}}},
]
TOOL_MAP = {tool["function"]["name"]: tool for tool in TOOLS}

# Each actual scenario/template family belongs to one split. Variants of a
# family never become held-out examples merely by changing a parameter or label.
FAMILIES = [
    ("schema_upper", "tool_schema", "train"), ("schema_multiply", "tool_schema", "validation"), ("schema_read", "tool_schema", "test"),
    ("call_add", "tool_call", "train"), ("call_reverse", "tool_call", "validation"), ("call_write", "tool_call", "test"),
    ("result_words", "tool_result", "train"), ("result_subtract", "tool_result", "validation"), ("result_append", "tool_result", "test"),
    ("retry_division", "tool_retry", "train"), ("retry_missing_file", "tool_retry", "validation"), ("retry_hex", "tool_retry", "test"),
    ("parallel_text", "tool_parallel", "train"), ("parallel_arithmetic", "tool_parallel", "validation"), ("parallel_files", "tool_parallel", "test"),
    ("multiturn_arithmetic", "tool_multiturn", "train"), ("multiturn_text", "tool_multiturn", "validation"), ("multiturn_files", "tool_multiturn", "test"),
]


class Toolbox:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)

    def execute(self, name, arguments):
        """Validate and run the actual tool, recording genuine runtime failures."""
        from jsonschema import Draft202012Validator
        if name not in TOOL_MAP:
            raise ValueError("unknown_defined_tool")
        Draft202012Validator(TOOL_MAP[name]["function"]["parameters"]).validate(arguments)
        try:
            operation = arguments["operation"]
            if name == "local_arithmetic":
                a, b = arguments["a"], arguments["b"]
                if operation == "divide":
                    value = str(Fraction(a, b))
                elif operation == "add":
                    value = a + b
                elif operation == "subtract":
                    value = a - b
                else:
                    value = a * b
            elif name == "local_text":
                text = arguments["text"]
                if operation == "decode_hex":
                    value = bytes.fromhex(text).decode("utf-8")
                elif operation == "upper":
                    value = text.upper()
                elif operation == "lower":
                    value = text.lower()
                elif operation == "reverse":
                    value = text[::-1]
                elif operation == "strip":
                    value = text.strip()
                else:
                    value = text.split()
            else:
                # Strict filename grammar and resolved-parent check prevent
                # traversal, absolute paths, and historical command execution.
                filename = arguments["name"]
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,75}\.txt", filename):
                    raise ValueError("invalid_episode_filename")
                path = self.root / filename
                if path.resolve().parent != self.root or path.is_symlink():
                    raise ValueError("episode_path_escape")
                if operation in ("read", "append") and not path.exists():
                    raise FileNotFoundError("missing_episode_file:" + filename)
                if operation == "write":
                    path.write_text(arguments["text"], encoding="utf-8", newline="")
                elif operation == "append":
                    with path.open("a", encoding="utf-8", newline="") as stream:
                        stream.write(arguments["text"])
                text = path.read_text(encoding="utf-8")
                value = text if operation == "read" else {"characters": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
            return {"ok": True, "success": True, "value": value}
        except (ValueError, ZeroDivisionError, FileNotFoundError, UnicodeError) as exc:
            return {"ok": False, "is_error": True, "error": {"type": type(exc).__name__, "message": str(exc)}}


class Episode:
    def __init__(self, case_id, toolbox):
        self.case_id, self.toolbox, self.number = case_id, toolbox, 0
        self.messages = [{"role": "system", "content": "Use the supplied local tools. Observe recorded tool errors and follow the user's stated recovery instructions. Report values from actual tool results."}]
        self.receipts = []

    def user(self, text):
        self.messages.append({"role": "user", "content": text})

    def calls(self, requests, *, introduction="", concurrent=False):
        calls = []
        for name, arguments in requests:
            self.number += 1
            calls.append({"id": f"seed-{self.case_id}-{self.number}", "type": "function", "function": {"name": name, "arguments": canonical(arguments)}})
        self.messages.append({"role": "assistant", "content": introduction, "tool_calls": calls})
        if concurrent:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(self.toolbox.execute, name, arguments) for name, arguments in requests]
                # All calls have been dispatched before collecting any result.
                # Reverse delivery tests valid completion ordering without
                # pretending that completion time is deterministic.
                delivered = [(i, futures[i].result()) for i in reversed(range(len(futures)))]
        else:
            delivered = [(i, self.toolbox.execute(name, arguments)) for i, (name, arguments) in enumerate(requests)]
        results = {}
        for i, result in delivered:
            call = calls[i]
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": canonical(result), "is_error": not result["ok"]})
            self.receipts.append({"call_id": call["id"], "tool_name": call["function"]["name"], "arguments": requests[i][1],
                                  "result": result, "result_sha256": sha(result), "dispatch_group": self.number - len(calls) + 1,
                                  "concurrent_dispatch": concurrent})
            results[i] = result
        return [results[i] for i in range(len(requests))]

    def answer(self, results):
        if not all(result["ok"] for result in results):
            raise ValueError("final_answer_requires_successful_observations")
        self.messages.append({"role": "assistant", "content": "Recorded value" + ("s" if len(results) != 1 else "") + ": " + canonical([result["value"] for result in results])})


def build_episode(family, facet, split, case_number, family_number, toolbox):
    case_id = sha(["operator-tool-seed-2026.v1", family, case_number])[:24]
    episode = Episode(case_id, toolbox)
    offset = {"train": 10000, "validation": 200000, "test": 400000}[split]
    a, b = offset + family_number * 701 + case_number * 7, offset // 10 + case_number * 11 + family_number
    text = f"{family} case {case_number} records local results in 2026 with identifier {case_id}"
    filename = f"{family}_{case_id[:12]}.txt"
    math = lambda op, x=a, y=b: ("local_arithmetic", {"operation": op, "a": x, "b": y})
    transform = lambda op, value=text: ("local_text", {"operation": op, "text": value})
    file = lambda op, name=filename, value=text: ("local_file", {"operation": op, "name": name, **({"text": value} if op != "read" else {})})
    tag = f"Episode {case_id}; scenario {family}. "
    if family == "schema_upper":
        episode.user(tag + f"State the local_text parameter schema, then uppercase this string exactly: {text}")
        result = episode.calls([transform("upper")], introduction=canonical(TOOL_MAP["local_text"]["function"]["parameters"]))
    elif family == "schema_multiply":
        episode.user(tag + f"Show the local_arithmetic argument requirements and use its multiplication operation on {a} and {b}.")
        result = episode.calls([math("multiply")], introduction=canonical(TOOL_MAP["local_arithmetic"]["function"]["parameters"]))
    elif family == "schema_read":
        episode.user(tag + f"Write {text!r} to {filename}. State the local_file schema before reading it back to verify exact content.")
        episode.calls([file("write")])
        result = episode.calls([file("read")], introduction=canonical(TOOL_MAP["local_file"]["function"]["parameters"]))
    elif family == "call_add":
        episode.user(tag + f"Call local_arithmetic once to add {a} and {b}, then give its recorded value.")
        result = episode.calls([math("add")])
    elif family == "call_reverse":
        episode.user(tag + f"Reverse the order of characters using local_text for this exact input: {text}")
        result = episode.calls([transform("reverse")])
    elif family == "call_write":
        episode.user(tag + f"Create the temporary file {filename} containing {text!r}. Return the observed character count and content digest.")
        result = episode.calls([file("write")])
    elif family == "result_words":
        episode.user(tag + f"Use the words operation and report every observed token, preserving order, from {text!r}.")
        result = episode.calls([transform("words")])
    elif family == "result_subtract":
        episode.user(tag + f"Subtract {b} from {a} with the arithmetic tool and report the integer received in the result.")
        result = episode.calls([math("subtract")])
    elif family == "result_append":
        episode.user(tag + f"Save {text!r} as {filename}, append ' verified', then read and report the resulting complete text.")
        episode.calls([file("write")])
        episode.calls([file("append", value=" verified")])
        result = episode.calls([file("read")])
    elif family == "retry_division":
        episode.user(tag + f"Demonstrate the division error by dividing {a} by zero. After observing it, retry with divisor {b} and report the exact rational value.")
        failed = episode.calls([math("divide", y=0)])
        if failed[0]["ok"]:
            raise ValueError("expected_runtime_error_not_observed")
        result = episode.calls([math("divide")], introduction="The recorded division failed. I will use the requested nonzero divisor.")
    elif family == "retry_missing_file":
        episode.user(tag + f"Attempt reading absent {filename}. If it is missing, write {text!r}, retry the read, and return the recorded content.")
        failed = episode.calls([file("read")])
        if failed[0]["ok"]:
            raise ValueError("expected_runtime_error_not_observed")
        episode.calls([file("write")], introduction="The read reported a missing file. I will create the requested content.")
        result = episode.calls([file("read")])
    elif family == "retry_hex":
        encoded = text.encode().hex()
        episode.user(tag + f"First try decode_hex on 'not-hex-{case_id}'. On its parsing error, retry this valid hexadecimal input and report the decoded text: {encoded}")
        failed = episode.calls([transform("decode_hex", "not-hex-" + case_id)])
        if failed[0]["ok"]:
            raise ValueError("expected_runtime_error_not_observed")
        result = episode.calls([transform("decode_hex", encoded)], introduction="The recorded hexadecimal parse failed. I will use the supplied valid encoding.")
    elif family == "parallel_text":
        episode.user(tag + f"Dispatch independent upper and lower transforms of {text!r} together. Return both observations in upper/lower order.")
        result = episode.calls([transform("upper"), transform("lower")], concurrent=True)
    elif family == "parallel_arithmetic":
        episode.user(tag + f"In one concurrent batch, compute the sum and difference of {a} and {b}. Report sum then difference from the results.")
        result = episode.calls([math("add"), math("subtract")], concurrent=True)
    elif family == "parallel_files":
        second = "second_" + filename
        episode.user(tag + f"Concurrently write {text!r} into {filename} and its reversed characters into {second}. Report each write receipt in requested file order.")
        result = episode.calls([file("write"), file("write", name=second, value=text[::-1])], concurrent=True)
    elif family == "multiturn_arithmetic":
        episode.user(tag + f"Add {a} and {b} and report the result before waiting for the next request.")
        first = episode.calls([math("add")])
        episode.answer(first)
        episode.user(f"Now subtract {b} from the previously observed sum and report the new tool value for episode {case_id}.")
        result = episode.calls([math("subtract", x=first[0]["value"])])
    elif family == "multiturn_text":
        episode.user(tag + f"Strip surrounding whitespace from '  {text}  ' with the text tool, then report its value.")
        first = episode.calls([transform("strip", "  " + text + "  ")])
        episode.answer(first)
        episode.user(f"For episode {case_id}, reverse the previously observed stripped text and report that new result.")
        result = episode.calls([transform("reverse", first[0]["value"])])
    elif family == "multiturn_files":
        episode.user(tag + f"Write {text!r} to {filename} and return the creation receipt.")
        first = episode.calls([file("write")])
        episode.answer(first)
        episode.user(f"Continue episode {case_id}: append ' complete' to {filename}, read it, and report the full recorded text.")
        episode.calls([file("append", value=" complete")])
        result = episode.calls([file("read")])
    else:
        raise ValueError("unknown_scenario_family")
    episode.answer(result)
    messages = normalize_messages(episode.messages)
    names = {call["function"]["name"] for message in messages for call in message.get("tool_calls", [])}
    tools = normalize_tools([TOOL_MAP[name] for name in sorted(names)])
    shape = pairing_and_schema(messages, tools)
    if not shape["schema_complete"]:
        raise ValueError("defined_schema_validation_failed")
    targets = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    row = {"schema": "recovery_corpus.v1", "id": "tool-seed-" + case_id, "route": "tool_calling", "messages": messages,
           "tools": tools, "media": [], "supervision": {"policy": "assistant_only", "message_mask": [int(m["role"] == "assistant") for m in messages]},
           "supervised_assistant_messages": targets,
           "provenance": {"source": "operator_authored_deterministic_tools_2026", "source_verified": True,
                          "source_ids": ["operator-tool-seed-2026.v1:" + family + ":" + str(case_number)],
                          "source_code_sha256": source_sha256, "tool_catalog_sha256": sha(TOOLS), "source_year": 2026,
                          "group_id": "tool-seed-family-" + family, "split": split, "template_family": family,
                          "case_id": case_id, "execution_receipt_sha256": sha(episode.receipts),
                          "synthetic_operator_authored": True, "actual_tool_execution": True,
                          "review_method": "deterministic_defined_schema_and_observed_result_checks", "benchmark_claim": False},
           "quality": {**shape, "context_complete": True, "format_ready": True, "content_review": "deterministic_fixture_review",
                       "train_ready": True, "tool_routes": [facet], "outcomes": [{"tool_call_id": r["call_id"], "status": "success" if r["result"]["ok"] else "failure"} for r in episode.receipts]}}
    return row, {"id": row["id"], "template_family": family, "split": split, "calls": episode.receipts}


def audit_splits(rows):
    seen, conflicts = {}, Counter()
    for row in rows:
        keys = [("family", row["provenance"]["template_family"]), ("case", row["provenance"]["case_id"]),
                ("source", row["provenance"]["source_ids"][0])]
        keys += [("prompt", sha(normalize_text(m["content"]))) for m in row["messages"] if m["role"] == "user"]
        keys += [("parameters", sha([call["function"]["name"], call["function"]["arguments"]])) for m in row["messages"] for call in m.get("tool_calls", [])]
        for kind, key in keys:
            previous = seen.setdefault((kind, key), row["provenance"]["split"])
            if previous != row["provenance"]["split"]:
                conflicts[kind] += 1
    if conflicts:
        raise ValueError("cross_split_case_parameter_or_source_leakage")
    return {"cross_split_conflicts": {}, "isolation_keys": ["template_family", "case_id", "source_id", "user_prompt", "tool_parameters"]}


def generate(out, episodes=216, excluded_prompts=None):
    if not 128 <= episodes <= 512:
        raise ValueError("episodes_must_be_128_to_512")
    now = datetime.now(timezone.utc)
    if now.year != 2026:
        raise ValueError("operator_source_year_requires_2026")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, receipts = [], []
    for number in range(episodes):
        family_number, case_number = number % len(FAMILIES), number // len(FAMILIES)
        family, facet, split = FAMILIES[family_number]
        with tempfile.TemporaryDirectory(prefix="episode-", dir=out) as temporary:
            row, receipt = build_episode(family, facet, split, case_number, family_number, Toolbox(temporary))
        if excluded_prompts and any(prompt_overlap(normalize_text(m["content"]), excluded_prompts) for m in row["messages"] if m["role"] == "user"):
            raise ValueError("frozen_eval_prompt_overlap")
        rows.append(row)
        receipts.append(receipt)
    isolation = audit_splits(rows)
    artifacts = {}
    for filename, values in (("ready.jsonl", rows), ("execution_receipts.jsonl", receipts)):
        data = "".join(canonical(value) + "\n" for value in values).encode()
        (out / filename).write_bytes(data)
        artifacts[filename] = {"sha256": hashlib.sha256(data).hexdigest(), "rows": len(values), "bytes": len(data)}
    report = {"schema": "operator_tool_seed_report.v1", "generated_at": now.isoformat(), "episodes": len(rows),
              "calls": sum(len(receipt["calls"]) for receipt in receipts), "observed_failures": sum(not call["result"]["ok"] for receipt in receipts for call in receipt["calls"]),
              "parallel_groups": sum(row["quality"]["parallel_groups"] for row in rows),
              "multi_user_turn_episodes": sum(sum(m["role"] == "user" for m in row["messages"]) > 1 for row in rows),
              "split_counts": dict(Counter(row["provenance"]["split"] for row in rows)),
              "tool_route_counts": dict(Counter(facet for row in rows for facet in row["quality"]["tool_routes"])),
              "families": len(FAMILIES), "isolation": isolation, "artifacts": artifacts,
              "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "tool_catalog_sha256": sha(TOOLS),
              "frozen_eval_prompts_excluded": len(excluded_prompts or []), "supervision_policy": "assistant_only",
              "limitations": ["Operator-authored synthetic format coverage, not realistic BFCL coverage.",
                              "Actual local tool execution validates these fixtures, not model capability.",
                              "Shared tool definitions are intentional; scenario families, cases, and parameters remain split-isolated."]}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=216)
    parser.add_argument("--exclude-prompts", type=Path)
    args = parser.parse_args(argv)
    try:
        excluded = None
        if args.exclude_prompts:
            prompts = json.loads(args.exclude_prompts.read_bytes())
            if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
                raise ValueError("frozen_prompts_must_be_string_list")
            excluded = {normalize_text(prompt) for prompt in prompts}
        report = generate(args.out, args.episodes, excluded)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
