#!/usr/bin/env python3
"""toolcall_selfgen.py -- the parent's native tool-use corpus and the held-out tool-calling bank.

WHY THIS FILE EXISTS.  Cyril, 2026-10-13 14:02Z: "[Cyril quote withheld]".  The D2 text set (distill/corpus/text,
20,000,768 tokens) carries 2.79M tokens on the `tool` route, but none of it is in the
parent's NATIVE tool format: the ids <tool_call>/</tool_call>/<tool_response> (248058 /
248059 / 248066) occur 0 times, no row has a `# Tools ... <tools>` system block, and the
calls that exist are JSON-in-prose (`{"tool_calls": [...]}`, 3,145 docs) written by the
omnicoder pipelines.  761 of its documents (293,384 tokens) are BFCL v4 ground truth, so
BFCL cannot be a held-out measure for this campaign.  Codex's 168-row tool cache
(native-text-v7-train, sha 3ed3f303) IS native-format but scripted: two tools
(local_text, local_arithmetic), six scenarios x 28, templated assistant turns
("Recorded value: [13106]"), 26,634 supervised positions.  Nothing in Stage D therefore
teaches or measures the parent's own tool behaviour.

WHAT THIS TOOL DOES.  Self-gen in the style of `selfgen_sampler.py`, but multi-turn and
with tools: the parent (its Q4_K_M on llama-server, P40s, ruling U7) writes the tool
schemas (`schemas`), the user requests, the simulated tool backends and -- the training
target -- every assistant turn; deterministic Python executes the sandbox tools
(calculator, units, dates, currency, text, bill split, percent change), so their results
are exact and the final answers can be checked against them.  Conversations are rendered
with the parent's own chat template (tools=..., enable_thinking=False; Qwen3-Coder XML
calls `<tool_call><function=..><parameter=..>`), the assistant spans become `target_mask`,
and `export` writes the safetensors corpus `qtext_teacher_cache build-text` reads, so the
logits are rescored in bf16 like every other self-gen token.

CATEGORIES.  single, parallel (several calls in one turn), sequential (a call that needs an
earlier result), multiturn (a second user request), error (the first call returns an
injected error: not found / rate limited / unavailable / permission / invalid argument),
clarify (a required value is missing; the assistant must ask, the user answers), no_tool
(answerable without tools), missing_tool (no offered tool can do it: decline, do not
invent).  A sample is kept only when its behaviour matches its category (rejection
sampling, up to --attempts per plan item); acceptance rates are reported per category.
1.1.0: a zero-call row (no_tool, missing_tool) must also pass the content rule
(CONTENT_RULES): no claimed lookup/action in its answer (multilingual patterns + the parent as
greedy judge), and no offered function that could serve the request.  Rows journaled before
the rule are re-checked when a run resumes; a failing row is reopened at its next attempt.

HELD OUT.  The bank (`bank-build`) uses tools that never appear in training: five sandbox
calculators (loan payment, great-circle distance, BMI, UTC-offset time conversion, compound
interest) and eight API families (flight status, prescriptions, meeting rooms, parcels,
recipes, support tickets, air quality, campsites).  Synthesized training tools whose names
touch those families are refused, and every training conversation is screened (13-gram /
exact / literal, distill_corpus_build.ReviewIndex) against the campaign review surface, the
corpus eval split and every bank string.  Ground truth in the bank is by construction
(templated values), graded by `bank-grade` from any engine's responses.

EXTENDING A RUN (1.2.0).  The token target is part of a run's locked semantics, so a sealed or
running run is never resumed with a larger one.  More tokens come from a CONTINUATION run:
`continue-plan` takes the plan items after the first run's sealed cutoff (same schemas, same
scenarios, ids unchanged, re-indexed from 0), the continuation `run` samples them under its own
--base-seed (a new seed range) and is sealed by its own `finalize` (--prior-manifest: ids must
be disjoint and a first user message already kept by an earlier run is screened out), and
`export` takes several --out-dir to write one rescore corpus over all of them (the sidecar
names every source manifest and its row range).  `run --stop-by` is a wall-clock deadline (not
part of the lock): no new plan item after it, no new attempt of an item in hand (that item is
journaled rejected, reason `deadline`, so the journal keeps no gap), exit 4; seal it with
`finalize --accept-short`.

FAILURES ARE FUNCTIONAL, NUMBERS ARE NOT.  An id outside [0, 248,320), a tokenizer or
template that is not the pinned one, a rendered conversation whose assistant span does not
start on a token boundary, a corpus whose bytes no longer match its record: each stops
with a non-zero exit.  Acceptance rates, lengths and bank scores are diagnostics.

Subcommands
-----------
  schemas      parent-synthesized tool schemas per training domain (+ the sandbox tools)
  plan         deterministic scenario plan over the schema library
  continue-plan  1.2.0: the unconsumed tail of a plan after a sealed run's cutoff (a continuation run's plan)
  run          the resumable generator (one worker per server slot; --stop-by deadline)
  status       progress from progress.json
  content-audit  read-only: the content rule over a journal's accepted zero-call rows
  finalize     cut at the token target in plan order, screen, seal manifest.json (--prior-manifest, --cut-at-gap)
  export       sealed conversations of one or more runs -> input_ids / lengths / target_mask safetensors
  bank-build   the held-out tool-calling bank (deterministic, no model)
  bank-run     greedy responses for every bank case from a llama-server
  bank-grade   grade a responses file (any engine) -> per-lane pass counts

    PY=/home/cereal/model-surgery-lab/venv/bin/python
    T=/home/cereal/omni-campaign/distill/tools
    $PY $T/toolcall_selfgen.py schemas --server http://127.0.0.1:18321 --out-dir $TC/schemas
    $PY $T/toolcall_selfgen.py plan --schemas $TC/schemas/schemas.jsonl --out $TC/plan.jsonl
    $PY $T/toolcall_selfgen.py run --plan $TC/plan.jsonl --server ... --out-dir $TC/run-...
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import math
import os
import platform
import queue
import random
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

TOOL_NAME = "toolcall_selfgen"
TOOL_VERSION = "1.2.0"
SCHEMA = "omni-distill-toolcall-v1"

OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
HERE = Path(__file__).resolve().parent
DEFAULT_PARENT = Path("/home/cereal/models/Qwen3.8-27B-Uncensored")
DEFAULT_BANK = OC / "calibration" / "banks" / "omni_toolcall_bank_v1.json"
DEFAULT_EVAL_SPLIT = OC / "distill" / "corpus" / "eval_split" / "eval_rows.jsonl"

#: The parent's tokenizer.json and chat_template.jinja (sha256 printed on the server 2026-10-14).
QWEN_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
CHAT_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
TEXT_VOCAB_ROWS = 248_320
#: <|im_end|>, <|endoftext|> -- generation_config eos_token_id.
EOG_IDS = (248_046, 248_044)
#: llama.cpp also stops on <|fim_pad|>/<|repo_name|>/<|file_sep|> (selfgen_sampler 1.2.0): a
#: turn cut there is truncated, never a finished turn.
SERVER_ONLY_EOG_IDS = (248_063, 248_064, 248_065)
PAD_ID = 248_044
TOOL_TOKEN_IDS = {"<tool_call>": 248_058, "</tool_call>": 248_059, "<tool_response>": 248_066, "</tool_response>": 248_067}
TODAY = "2026-10-14"
MAX_RENDER_TOKENS = 8_192
#: Largest simulated tool result kept (characters of compact JSON); ~1,000 tokens.
BACKEND_MAX_CHARS = 4_000
ROUTE = "tool_native"

#: Sampling.  Assistant turns use Qwen's documented non-thinking setting (T 0.7, top-p 0.8,
#: top-k 20); the user / backend writers run warmer for diversity.  Logits are rescored in
#: bf16, so these only choose WHICH sequences exist.
SAMPLING = {
    "assistant": {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "n_predict": 1024},
    "user": {"temperature": 0.9, "top_k": 40, "top_p": 0.95, "min_p": 0.0, "n_predict": 320},
    "backend": {"temperature": 0.7, "top_k": 20, "top_p": 0.9, "min_p": 0.0, "n_predict": 900},
    "schema": {"temperature": 0.8, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "n_predict": 2048},
    "bank": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "n_predict": 768},
}

#: 1.1.0: the zero-call content rule's judge (greedy).  Kept OUT of SAMPLING on purpose: SAMPLING
#: is part of the locked run_config (it fixes which sequences the generator samples), while the
#: content rule is an acceptance rule, versioned per journal row as CONTENT_RULES and enforced
#: by `finalize`; a run started under 1.0.x resumes and re-checks its old rows (run_command).
CHECK_SAMPLING = {"judge": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "n_predict": 64}}
CONTENT_RULES = "zero-call-content-v1"
ZERO_CALL_CATEGORIES = ("no_tool", "missing_tool")
#: More internal (non-_Reject) failures than this in one session stop the run: a systematic
#: bug must not quietly turn the whole plan into rejections.
INTERNAL_FAILURE_LIMIT = 25

CATEGORY_WEIGHTS = {"single": 0.20, "parallel": 0.17, "sequential": 0.14, "multiturn": 0.10, "error": 0.10,
                    "clarify": 0.09, "no_tool": 0.10, "missing_tool": 0.10}
TOOL_CATEGORIES = ("single", "parallel", "sequential", "multiturn", "error", "clarify")


class ToolcallError(RuntimeError):
    """A functional failure: the caller exits non-zero with this message."""


# --------------------------------------------------------------------------
# small helpers (the selfgen_sampler conventions)
# --------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Key order is kept everywhere data is stored: the chat template renders tool schemas and
    call arguments in insertion order, so a sorted copy would render different tokens.  Only
    the seals (canonical_sha256) sort."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    body = {k: v for k, v in payload.items() if k != field}
    return dict(body, **{field: canonical_sha256(body)})


def check_seal(payload: dict[str, Any], field: str) -> bool:
    return payload.get(field) == canonical_sha256({k: v for k, v in payload.items() if k != field})


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return file_sha256(path), count


def tool_identity() -> dict[str, Any]:
    me = Path(__file__).resolve()
    return {"tool": TOOL_NAME, "version": TOOL_VERSION, "path": str(me), "sha256": file_sha256(me),
            "python": platform.python_version(), "host": platform.node()}


def item_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()[:4], "big")


# --------------------------------------------------------------------------
# JSON-schema subset: tool definitions and argument values
# --------------------------------------------------------------------------

SCALAR_TYPES = ("string", "integer", "number", "boolean")
ALLOWED_TYPES = SCALAR_TYPES + ("array", "object")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def first_type(schema: dict[str, Any]) -> str | None:
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    return kind if isinstance(kind, str) else None


def validate_property(name: str, prop: Any, depth: int = 0) -> list[str]:
    problems = []
    if not PARAM_NAME_RE.match(str(name)):
        problems.append(f"parameter name {name!r} is not an identifier")
    if not isinstance(prop, dict):
        return problems + [f"parameter {name} is not an object"]
    kind = first_type(prop)
    if kind not in ALLOWED_TYPES:
        problems.append(f"parameter {name} has type {prop.get('type')!r}")
    if not isinstance(prop.get("description", ""), str):
        problems.append(f"parameter {name} description is not a string")
    if "enum" in prop:
        enum = prop["enum"]
        if not isinstance(enum, list) or not enum or not all(isinstance(e, (str, int, float, bool)) for e in enum):
            problems.append(f"parameter {name} enum is not a non-empty scalar list")
    if kind == "array":
        items = prop.get("items")
        if not isinstance(items, dict) or first_type(items) not in ALLOWED_TYPES:
            problems.append(f"array parameter {name} has no typed items")
        elif first_type(items) == "object" and depth < 1:
            for sub, subprop in (items.get("properties") or {}).items():
                problems += validate_property(f"{name}[].{sub}", subprop, depth + 1)
    if kind == "object" and depth < 1:
        for sub, subprop in (prop.get("properties") or {}).items():
            problems += validate_property(f"{name}.{sub}", subprop, depth + 1)
    return problems


def validate_tool(tool: Any) -> list[str]:
    """A function definition {name, description, parameters} (bare or OpenAI-wrapped)."""
    if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        tool = tool["function"]
    if not isinstance(tool, dict):
        return ["tool is not an object"]
    problems = []
    name = tool.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        problems.append(f"name {name!r} is not snake_case")
    description = tool.get("description")
    if not isinstance(description, str) or not description.strip() or len(description) > 600:
        problems.append("description missing or longer than 600 characters")
    params = tool.get("parameters")
    if not isinstance(params, dict) or params.get("type") != "object":
        return problems + ["parameters is not a type=object schema"]
    props = params.get("properties", {})
    if not isinstance(props, dict) or len(props) > 10:
        problems.append("properties is not an object of at most 10 parameters")
        props = {}
    for pname, prop in props.items():
        problems += validate_property(pname, prop)
    required = params.get("required", [])
    if not isinstance(required, list) or any(r not in props for r in required):
        problems.append(f"required {required!r} names parameters that do not exist")
    return problems


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        tool = tool["function"]
    params = tool["parameters"]
    out_params = {"type": "object", "properties": params.get("properties", {}), "required": list(params.get("required", []))}
    return {"type": "function", "function": {"name": tool["name"], "description": tool["description"].strip(), "parameters": out_params}}


def tool_function(tool: dict[str, Any]) -> dict[str, Any]:
    return tool["function"] if tool.get("type") == "function" else tool


def validate_value(schema: dict[str, Any], value: Any, where: str) -> list[str]:
    kind = first_type(schema)
    problems: list[str] = []
    if kind == "string" and not isinstance(value, str):
        return [f"{where}: expected string, got {type(value).__name__}"]
    if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        return [f"{where}: expected integer, got {value!r}"]
    if kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
        return [f"{where}: expected number, got {value!r}"]
    if kind == "boolean" and not isinstance(value, bool):
        return [f"{where}: expected boolean, got {value!r}"]
    if kind == "array" and not isinstance(value, list):
        return [f"{where}: expected array, got {type(value).__name__}"]
    if kind == "object" and not isinstance(value, dict):
        return [f"{where}: expected object, got {type(value).__name__}"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        problems.append(f"{where}: {value!r} not in enum {enum}")
    if kind in ("integer", "number"):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            problems.append(f"{where}: {value} < minimum {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            problems.append(f"{where}: {value} > maximum {schema['maximum']}")
    if kind == "string":
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            problems.append(f"{where}: longer than maxLength {schema['maxLength']}")
        if not value.strip():
            problems.append(f"{where}: empty string")
    if kind == "array" and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            problems += validate_value(schema["items"], item, f"{where}[{i}]")
    if kind == "object":
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                problems.append(f"{where}: missing required key {key}")
        for key, item in value.items():
            if key in props:
                problems += validate_value(props[key], item, f"{where}.{key}")
    return problems


def validate_arguments(tool: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    params = tool_function(tool)["parameters"]
    props = params.get("properties") or {}
    problems = [f"unknown parameter {k!r}" for k in arguments if k not in props]
    problems += [f"missing required parameter {k!r}" for k in params.get("required") or [] if k not in arguments]
    for key, value in arguments.items():
        if key in props:
            problems += validate_value(props[key], value, key)
    return problems


# --------------------------------------------------------------------------
# The parent's native call format (chat_template.jinja, Qwen3-Coder XML)
# --------------------------------------------------------------------------

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
_INT_RE = re.compile(r"^[+-]?\d+$")


def coerce_parameter(raw: str, prop: dict[str, Any] | None) -> tuple[Any, str | None]:
    """The template writes a string argument as-is and anything else as JSON."""
    kind = first_type(prop) if isinstance(prop, dict) else None
    text = raw.strip()
    if kind == "string":
        return raw, None
    if kind == "integer":
        if _INT_RE.match(text):
            return int(text), None
        return raw, f"not an integer: {text[:40]!r}"
    if kind == "number":
        try:
            number = json.loads(text)
        except ValueError:
            return raw, f"not a number: {text[:40]!r}"
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return raw, f"not a number: {text[:40]!r}"
        return number, None
    if kind == "boolean":
        if text.lower() in ("true", "false"):
            return text.lower() == "true", None
        return raw, f"not a boolean: {text[:40]!r}"
    if kind in ("array", "object"):
        try:
            return json.loads(text), None
        except ValueError:
            return raw, f"not JSON {kind}: {text[:60]!r}"
    try:
        return json.loads(text), None
    except ValueError:
        return raw, None


def parse_assistant(text: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Sampled assistant text (EOG removed) -> content, calls, problems."""
    by_name = {tool_function(t)["name"]: t for t in tools}
    problems: list[str] = []
    calls: list[dict[str, Any]] = []
    first = text.find("<tool_call>")
    content = (text if first < 0 else text[:first]).strip()
    if "</think>" in content or "<think>" in content:
        problems.append("think tags inside a thinking-off turn")
    if "<tool_response>" in text or "<|im_start|>" in text:
        problems.append("the turn writes a tool response or a new role itself")
    if first >= 0:
        pos = first
        while True:
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            match = CALL_RE.match(text, pos)
            if not match:
                problems.append(f"text after or between tool calls: {text[pos:pos + 60]!r}")
                break
            name = match.group(1).strip()
            body = match.group(2)
            arguments: dict[str, Any] = {}
            tool = by_name.get(name)
            if tool is None:
                problems.append(f"call to a function that is not offered: {name!r}")
            props = (tool_function(tool)["parameters"].get("properties") or {}) if tool else {}
            leftover = PARAM_RE.sub("", body)
            if leftover.strip():
                problems.append(f"{name}: stray text inside <function>: {leftover.strip()[:60]!r}")
            for pmatch in PARAM_RE.finditer(body):
                pname = pmatch.group(1).strip()
                if pname in arguments:
                    problems.append(f"{name}: parameter {pname} given twice")
                value, why = coerce_parameter(pmatch.group(2), props.get(pname))
                if why:
                    problems.append(f"{name}.{pname}: {why}")
                arguments[pname] = value
            if tool is not None:
                problems += [f"{name}: {p}" for p in validate_arguments(tool, arguments)]
            calls.append({"name": name, "arguments": arguments})
            pos = match.end()
    return {"content": content, "calls": calls, "problems": problems}


def assistant_message(parsed: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": parsed["content"]}
    if parsed["calls"]:
        message["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in parsed["calls"]]
    return message


# --------------------------------------------------------------------------
# Sandbox tools: executed in Python, exact results
# --------------------------------------------------------------------------


def _err(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": dict({"code": code, "message": message}, **extra)}


_CALC_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "sin": math.sin, "cos": math.cos,
               "tan": math.tan, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil}
_CALC_NAMES = {"pi": math.pi, "e": math.e}


def _calc(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _calc(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CALC_NAMES:
        return _CALC_NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _calc(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _calc(node.left), _calc(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(right) > 64 or abs(left) > 1e12:
                raise ValueError("exponent too large")
            return left ** right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALC_FUNCS and not node.keywords:
        return _CALC_FUNCS[node.func.id](*[_calc(a) for a in node.args])
    raise ValueError(f"unsupported expression element {type(node).__name__}")


def _num(value: float) -> float | int:
    if isinstance(value, int):
        return value
    if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
        return int(value)
    return round(value, 6)


def calculate(expression: str) -> dict[str, Any]:
    expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    if len(expr) > 300:
        return _err("INVALID_ARGUMENT", "expression longer than 300 characters")
    try:
        value = _calc(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return _err("MATH_ERROR", "division by zero")
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError) as exc:
        return _err("INVALID_ARGUMENT", f"cannot evaluate expression: {exc}")
    # 1.1.0: float() of an exact int beyond ~1.8e308 (999999999999**64) raises OverflowError;
    # it used to escape the sandbox and kill the worker thread.  Same error object as inf.
    try:
        finite = not isinstance(value, complex) and math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        return _err("MATH_ERROR", "result is not a finite real number")
    return {"expression": expression, "result": _num(value)}


UNITS = {
    "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "yd": 0.9144, "ft": 0.3048, "in": 0.0254},
    "mass": {"kg": 1.0, "g": 0.001, "mg": 1e-6, "lb": 0.45359237, "oz": 0.028349523125, "t": 1000.0},
    "volume": {"l": 1.0, "ml": 0.001, "gal": 3.785411784, "qt": 0.946352946, "cup": 0.2365882365, "floz": 0.0295735295625},
    "speed": {"km/h": 1 / 3.6, "mph": 0.44704, "m/s": 1.0, "knot": 0.514444},
    "data": {"B": 1.0, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "KiB": 1024.0, "MiB": 1024.0 ** 2, "GiB": 1024.0 ** 3},
    "temperature": {"C": None, "F": None, "K": None},
}
UNIT_DIMENSION = {u: d for d, table in UNITS.items() for u in table}


def convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    a, b = UNIT_DIMENSION.get(from_unit), UNIT_DIMENSION.get(to_unit)
    if a is None or b is None:
        return _err("INVALID_ARGUMENT", f"unknown unit {from_unit if a is None else to_unit!r}")
    if a != b:
        return _err("INVALID_ARGUMENT", f"cannot convert {a} ({from_unit}) to {b} ({to_unit})")
    if a == "temperature":
        kelvin = {"C": value + 273.15, "F": (value - 32) * 5 / 9 + 273.15, "K": value}[from_unit]
        if kelvin < 0:
            return _err("INVALID_ARGUMENT", "temperature below absolute zero")
        result = {"C": kelvin - 273.15, "F": (kelvin - 273.15) * 9 / 5 + 32, "K": kelvin}[to_unit]
    else:
        result = value * UNITS[a][from_unit] / UNITS[a][to_unit]
    return {"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": _num(round(result, 6))}


def _date(text: str) -> dt.date:
    return dt.date.fromisoformat(text.strip())


def date_add(date: str, days: int) -> dict[str, Any]:
    try:
        start = _date(date)
    except ValueError:
        return _err("INVALID_ARGUMENT", "date must be YYYY-MM-DD")
    if abs(days) > 36500:
        return _err("INVALID_ARGUMENT", "days must be within +/-36500")
    try:
        end = start + dt.timedelta(days=days)
    except OverflowError:  # 1.1.0: 9999-12-30 + 5 days
        return _err("INVALID_ARGUMENT", "the resulting date is outside years 1-9999")
    return {"start_date": start.isoformat(), "days": days, "date": end.isoformat(), "weekday": end.strftime("%A")}


def days_between(start_date: str, end_date: str) -> dict[str, Any]:
    try:
        a, b = _date(start_date), _date(end_date)
    except ValueError:
        return _err("INVALID_ARGUMENT", "dates must be YYYY-MM-DD")
    return {"start_date": a.isoformat(), "end_date": b.isoformat(), "days": (b - a).days, "weeks": round((b - a).days / 7, 2)}


#: Fixed sandbox reference rates (units per 1 USD).  The tool says they are the sandbox's own.
FX = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "CNY": 7.18, "INR": 83.9, "CAD": 1.37, "AUD": 1.52,
      "CHF": 0.88, "MXN": 18.6, "BRL": 5.45, "KRW": 1345.0}


def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    if from_currency not in FX or to_currency not in FX:
        return _err("INVALID_ARGUMENT", f"unsupported currency; supported: {sorted(FX)}")
    rate = FX[to_currency] / FX[from_currency]
    return {"amount": amount, "from_currency": from_currency, "to_currency": to_currency, "rate": round(rate, 6),
            "converted": round(amount * rate, 2), "rates_source": "sandbox reference table 2026-10-01"}


def text_statistics(text: str) -> dict[str, Any]:
    words = re.findall(r"\b\w+\b", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {"characters": len(text), "characters_no_spaces": len(re.sub(r"\s", "", text)), "words": len(words),
            "sentences": len(sentences), "average_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0}


def split_bill(total: float, tip_percent: float, people: int) -> dict[str, Any]:
    if people < 1:
        return _err("INVALID_ARGUMENT", "people must be at least 1")
    tip = round(total * tip_percent / 100, 2)
    return {"total": total, "tip_percent": tip_percent, "tip_amount": tip, "total_with_tip": round(total + tip, 2),
            "per_person": round((total + tip) / people, 2), "people": people}


def percent_change(old_value: float, new_value: float) -> dict[str, Any]:
    if old_value == 0:
        return _err("MATH_ERROR", "old_value is zero; percent change is undefined")
    change = (new_value - old_value) / abs(old_value) * 100
    return {"old_value": old_value, "new_value": new_value, "difference": _num(round(new_value - old_value, 6)), "percent_change": round(change, 4)}


# held out: bank only ------------------------------------------------------------

def loan_payment(principal: float, annual_rate_percent: float, months: int) -> dict[str, Any]:
    if months < 1:
        return _err("INVALID_ARGUMENT", "months must be at least 1")
    r = annual_rate_percent / 100 / 12
    pay = principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months)
    return {"monthly_payment": round(pay, 2), "total_paid": round(pay * months, 2), "total_interest": round(pay * months - principal, 2)}


def geo_distance(lat1: float, lon1: float, lat2: float, lon2: float, unit: str = "km") -> dict[str, Any]:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    km = 2 * 6371.0088 * math.asin(math.sqrt(a))
    return {"distance": round(km if unit == "km" else km / 1.609344, 2), "unit": unit}


def bmi_calculate(weight_kg: float, height_cm: float) -> dict[str, Any]:
    if height_cm <= 0:
        return _err("INVALID_ARGUMENT", "height_cm must be positive")
    bmi = weight_kg / (height_cm / 100) ** 2
    band = "underweight" if bmi < 18.5 else "normal" if bmi < 25 else "overweight" if bmi < 30 else "obese"
    return {"bmi": round(bmi, 1), "category": band}


def timezone_convert(time: str, from_utc_offset: float, to_utc_offset: float) -> dict[str, Any]:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", time.strip())
    if not match:
        return _err("INVALID_ARGUMENT", "time must be HH:MM")
    minutes = int(match.group(1)) * 60 + int(match.group(2)) + round((to_utc_offset - from_utc_offset) * 60)
    shift, minutes = divmod(minutes, 1440)
    return {"time": f"{minutes // 60:02d}:{minutes % 60:02d}", "day_shift": shift}


def compound_interest(principal: float, annual_rate_percent: float, years: float, compounds_per_year: int) -> dict[str, Any]:
    n = compounds_per_year
    amount = principal * (1 + annual_rate_percent / 100 / n) ** (n * years)
    return {"final_amount": round(amount, 2), "interest_earned": round(amount - principal, 2)}


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


_UNIT_ENUM = sorted(UNIT_DIMENSION)
SANDBOX_TOOLS: dict[str, tuple[dict[str, Any], Callable[..., dict[str, Any]]]] = {
    "calculate": (_fn("calculate", "Evaluate an arithmetic expression exactly (+, -, *, /, //, %, **, parentheses, sqrt, log, exp, sin, cos, tan, abs, round, floor, ceil, pi, e).",
                      {"expression": {"type": "string", "description": "The expression, e.g. (17 * 6) / 3"}}, ["expression"]), calculate),
    "convert_units": (_fn("convert_units", "Convert a quantity between units of length, mass, volume, speed, data size or temperature.",
                          {"value": {"type": "number", "description": "The quantity to convert"},
                           "from_unit": {"type": "string", "enum": _UNIT_ENUM, "description": "Unit of the given value"},
                           "to_unit": {"type": "string", "enum": _UNIT_ENUM, "description": "Unit to convert to"}},
                          ["value", "from_unit", "to_unit"]), convert_units),
    "date_add": (_fn("date_add", "Add (or subtract, with a negative number) a number of days to a calendar date and return the resulting date and weekday.",
                     {"date": {"type": "string", "description": "Start date, YYYY-MM-DD"},
                      "days": {"type": "integer", "description": "Days to add; negative to go back"}}, ["date", "days"]), date_add),
    "days_between": (_fn("days_between", "Count the days from one calendar date to another.",
                         {"start_date": {"type": "string", "description": "YYYY-MM-DD"},
                          "end_date": {"type": "string", "description": "YYYY-MM-DD"}}, ["start_date", "end_date"]), days_between),
    "currency_convert": (_fn("currency_convert", "Convert an amount between currencies at the service's reference rates.",
                             {"amount": {"type": "number", "description": "Amount in the source currency"},
                              "from_currency": {"type": "string", "enum": sorted(FX), "description": "ISO 4217 code"},
                              "to_currency": {"type": "string", "enum": sorted(FX), "description": "ISO 4217 code"}},
                             ["amount", "from_currency", "to_currency"]), currency_convert),
    "text_statistics": (_fn("text_statistics", "Count characters, words and sentences in a piece of text.",
                            {"text": {"type": "string", "description": "The text to analyse"}}, ["text"]), text_statistics),
    "split_bill": (_fn("split_bill", "Add a tip to a bill and split the total evenly between people.",
                       {"total": {"type": "number", "description": "Bill total before tip"},
                        "tip_percent": {"type": "number", "description": "Tip as a percentage, e.g. 18"},
                        "people": {"type": "integer", "description": "Number of people splitting", "minimum": 1}},
                       ["total", "tip_percent", "people"]), split_bill),
    "percent_change": (_fn("percent_change", "Compute the difference and percent change from an old value to a new value.",
                           {"old_value": {"type": "number", "description": "Starting value"},
                            "new_value": {"type": "number", "description": "Ending value"}}, ["old_value", "new_value"]), percent_change),
}

HELDOUT_SANDBOX: dict[str, tuple[dict[str, Any], Callable[..., dict[str, Any]]]] = {
    "loan_payment": (_fn("loan_payment", "Monthly payment, total paid and total interest of a fixed-rate amortizing loan.",
                         {"principal": {"type": "number", "description": "Amount borrowed"},
                          "annual_rate_percent": {"type": "number", "description": "Nominal annual interest rate in percent, e.g. 6.5"},
                          "months": {"type": "integer", "description": "Term in months"}}, ["principal", "annual_rate_percent", "months"]), loan_payment),
    "geo_distance": (_fn("geo_distance", "Great-circle distance between two latitude/longitude points.",
                         {"lat1": {"type": "number", "description": "Latitude of point 1"}, "lon1": {"type": "number", "description": "Longitude of point 1"},
                          "lat2": {"type": "number", "description": "Latitude of point 2"}, "lon2": {"type": "number", "description": "Longitude of point 2"},
                          "unit": {"type": "string", "enum": ["km", "mi"], "description": "Distance unit"}}, ["lat1", "lon1", "lat2", "lon2"]), geo_distance),
    "bmi_calculate": (_fn("bmi_calculate", "Body-mass index and its WHO category from weight and height.",
                          {"weight_kg": {"type": "number", "description": "Weight in kilograms"},
                           "height_cm": {"type": "number", "description": "Height in centimetres"}}, ["weight_kg", "height_cm"]), bmi_calculate),
    "timezone_convert": (_fn("timezone_convert", "Convert a clock time between two UTC offsets.",
                             {"time": {"type": "string", "description": "Clock time HH:MM"},
                              "from_utc_offset": {"type": "number", "description": "UTC offset of the given time in hours, e.g. -5"},
                              "to_utc_offset": {"type": "number", "description": "Target UTC offset in hours, e.g. 5.5"}},
                             ["time", "from_utc_offset", "to_utc_offset"]), timezone_convert),
    "compound_interest": (_fn("compound_interest", "Final amount and interest earned with periodic compounding.",
                              {"principal": {"type": "number", "description": "Starting amount"},
                               "annual_rate_percent": {"type": "number", "description": "Annual rate in percent"},
                               "years": {"type": "number", "description": "Duration in years"},
                               "compounds_per_year": {"type": "integer", "description": "Compounding periods per year, e.g. 12"}},
                              ["principal", "annual_rate_percent", "years", "compounds_per_year"]), compound_interest),
}

#: Held-out API families (bank only): any synthesized training tool whose name or
#: description touches one of these is refused.
HELDOUT_KEYWORDS = (r"\bflights?\b", r"flight_", r"airline", r"airport", r"prescription", r"pharmac", r"meeting[ _]?rooms?",
                    r"conference[ _]?rooms?", r"\bparcels?\b", r"parcel_", r"track\w*_package", r"package_track", r"tracking[ _]number",
                    r"\bshipments?\b", r"recipe", r"support[ _]tickets?", r"helpdesk", r"air[ _]quality", r"\bpollut", r"campsite",
                    r"campground", r"\bloans?\b", r"loan_", r"mortgage", r"amortiz", r"\bbmi\b", r"body[ _]mass", r"great[ _-]circle",
                    r"haversine", r"time[ _]?zones?", r"utc[ _]offset", r"compound[ _]interest")
_HELDOUT_RE = [re.compile(k) for k in HELDOUT_KEYWORDS]


def heldout_touch(tool: dict[str, Any]) -> str | None:
    """The first held-out family pattern the tool's name or description matches (None = clean)."""
    fn = tool_function(tool)
    hay = (fn.get("name", "") + " " + fn.get("name", "").replace("_", " ") + " " + fn.get("description", "")).lower()
    return next((rx.pattern for rx in _HELDOUT_RE if rx.search(hay)), None)


def _finite_tree(value: Any) -> bool:
    """Every number in a sandbox result converts to a finite float (no inf / nan / huge int)."""
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_tree(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return math.isfinite(float(value))
        except OverflowError:
            return False
    return True


def execute_sandbox(name: str, arguments: dict[str, Any], table: dict[str, Any] | None = None) -> dict[str, Any]:
    """Never raises for any argument values: the model chooses them.  1.1.0: arithmetic and
    value errors (an int too large for float, a date past 9999) become error objects, and a
    result holding inf / nan / an unconvertible int is a MATH_ERROR -- `grounded` calls
    float() on every result value."""
    table = SANDBOX_TOOLS if table is None else table
    _, fn = table[name]
    try:
        result = fn(**arguments)
    except TypeError as exc:
        return _err("INVALID_ARGUMENT", str(exc))
    except ArithmeticError as exc:
        return _err("MATH_ERROR", f"cannot compute: {exc}")
    except ValueError as exc:
        return _err("INVALID_ARGUMENT", f"cannot compute: {exc}")
    if not _finite_tree(result):
        return _err("MATH_ERROR", "the result is not a finite number")
    return result


# --------------------------------------------------------------------------
# Injected errors (category "error": the FIRST call of the conversation only)
# --------------------------------------------------------------------------

ERROR_KINDS = ("not_found", "rate_limited", "unavailable", "permission_denied", "invalid_argument")


def injected_error(kind: str, call: dict[str, Any], tool: dict[str, Any]) -> dict[str, Any]:
    args = call["arguments"]
    first_string = next((v for v in args.values() if isinstance(v, str) and v.strip()), None)
    if kind == "not_found":
        return _err("NOT_FOUND", f"No record matches {first_string!r}." if first_string else "No matching record was found.")
    if kind == "rate_limited":
        return _err("RATE_LIMITED", "Too many requests. Retry after 20 seconds.", retry_after_seconds=20)
    if kind == "unavailable":
        return _err("SERVICE_UNAVAILABLE", "The upstream service did not respond within 10 seconds.")
    if kind == "permission_denied":
        return _err("PERMISSION_DENIED", "The current credentials are not allowed to perform this operation.")
    props = tool_function(tool)["parameters"].get("properties") or {}
    for key in args:
        enum = (props.get(key) or {}).get("enum")
        if enum:
            return _err("INVALID_ARGUMENT", f"Parameter '{key}' is not accepted for this request. Allowed values: {enum}.", parameter=key)
    key = next(iter(args), None)
    return _err("INVALID_ARGUMENT", f"Parameter '{key}' has an invalid format." if key else "The request is malformed.", parameter=key)


# --------------------------------------------------------------------------
# Training domains (schema synthesis).  None of them is a bank family.
# --------------------------------------------------------------------------

DOMAINS = [
    ("ecommerce_orders", "an online store's order management (orders, returns, refunds, order history)"),
    ("warehouse_inventory", "warehouse inventory and stock levels, reorders and bin locations"),
    ("retail_banking", "a retail bank (accounts, balances, transfers, card controls, statements)"),
    ("stock_market", "stock market data (quotes, historical prices, company fundamentals, watchlists)"),
    ("personal_calendar", "a personal calendar (events, availability, reminders, invitations)"),
    ("email_client", "an email client (search, read, send, label, drafts)"),
    ("crm_sales", "a sales CRM (contacts, leads, deals, pipeline stages, notes)"),
    ("project_tasks", "a project management tool (projects, tasks, assignees, due dates, comments)"),
    ("code_hosting", "a git code-hosting service (repositories, branches, pull requests, issues)"),
    ("ci_pipelines", "a CI/CD service (pipelines, jobs, logs, artifacts, re-runs)"),
    ("cloud_compute", "a cloud provider's virtual machines (instances, sizes, regions, snapshots)"),
    ("kubernetes_ops", "a Kubernetes cluster (deployments, pods, scaling, logs, rollouts)"),
    ("sql_warehouse", "an analytics data warehouse (run read-only SQL, list tables, describe schemas)"),
    ("file_storage", "a cloud file storage service (folders, files, sharing links, versions)"),
    ("music_streaming", "a music streaming service (search tracks, playlists, playback, recommendations)"),
    ("video_catalog", "a video streaming catalog (titles, episodes, watchlists, availability by country)"),
    ("news_search", "a news search API (headlines, topics, sources, date ranges)"),
    ("weather_service", "a weather service (current conditions, hourly and daily forecasts, alerts)"),
    ("maps_geocoding", "a maps service (geocoding, reverse geocoding, places nearby, opening hours)"),
    ("ride_hailing", "a ride-hailing app (fare estimates, booking rides, ride status, cancellation)"),
    ("hotel_booking", "hotel search and booking (availability, rates, reservations, cancellations)"),
    ("restaurant_reservations", "restaurant search and table reservations"),
    ("food_delivery", "a food delivery app (menus, carts, orders, delivery status)"),
    ("fitness_tracker", "a fitness tracker (workouts, steps, heart rate, goals)"),
    ("smart_home", "smart home devices (lights, thermostats, locks, scenes, energy use)"),
    ("ev_charging", "an electric-vehicle charging network (stations, connector types, sessions, pricing)"),
    ("car_rental", "car rental (vehicle search, quotes, bookings, extras)"),
    ("real_estate", "real-estate listings (search homes, listing details, open houses, price history)"),
    ("job_board", "a job board (search postings, save jobs, applications, employer profiles)"),
    ("hr_payroll", "an HR and payroll system (employees, leave balances, payslips, time off requests)"),
    ("learning_platform", "an online learning platform (courses, enrolments, progress, quizzes)"),
    ("library_catalog", "a public library catalog (search books, holds, loans, renewals)"),
    ("translation_dictionary", "translation and dictionary lookups (translate text, definitions, synonyms)"),
    ("sports_scores", "sports data (fixtures, live scores, standings, player stats)"),
    ("event_ticketing", "event ticketing (concerts and shows, seat maps, ticket purchase, transfers)"),
    ("social_media", "a social media account manager (posts, scheduling, comments, analytics)"),
    ("loyalty_rewards", "a retail loyalty programme (points balance, rewards catalog, redemptions)"),
    ("insurance_claims", "a car and home insurance portal (policies, claims, claim status, documents)"),
    ("invoicing_accounting", "small-business invoicing and accounting (invoices, customers, payments, expenses)"),
    ("it_asset_inventory", "an IT asset inventory (laptops, licences, assignments, warranty status)"),
    ("vulnerability_scanning", "a vulnerability scanner (targets, scans, findings, severity, remediation status)"),
    ("fleet_telematics", "vehicle fleet telematics (vehicle locations, fuel, maintenance schedules, drivers)"),
    ("farm_sensors", "agricultural sensors (soil moisture, irrigation zones, weather stations, alerts)"),
    ("utility_billing", "a utility account (electricity and water usage, bills, outages, payment plans)"),
    ("vet_clinic", "a veterinary clinic (pets, appointments, vaccination records, reminders)"),
    ("parking", "city parking (find garages, rates, reserve spots, active sessions)"),
    ("museum_tours", "a museum (exhibitions, tour bookings, opening hours, membership)"),
    ("podcast_platform", "a podcast platform (search shows, episodes, subscriptions, transcripts)"),
]
FLAVORS = ("a consumer mobile app", "an internal enterprise API", "a small business", "a public REST API with strict validation",
           "a city government service", "a startup's beta API")


# --------------------------------------------------------------------------
# Users: personas, styles, languages (the user writer's seeds)
# --------------------------------------------------------------------------

PERSONAS = ("a busy parent", "a software engineer", "a retired teacher", "a small-business owner", "a university student",
            "an operations manager", "a nurse on a night shift", "a freelance designer", "a data analyst", "a travel blogger",
            "a site reliability engineer", "a sales representative", "an accountant", "a high-school coach", "a landlord",
            "a researcher", "a customer-support agent", "a chef", "a logistics coordinator", "a non-technical manager")
STYLES = ("terse, a single short sentence", "casual and friendly", "formal and precise", "detailed, with some context first",
          "hurried, with a small typo or two", "polite, written by a non-native English speaker")
LANGUAGES = (("English", 0.84), ("Spanish", 0.03), ("German", 0.025), ("French", 0.025), ("Portuguese", 0.02),
             ("Chinese", 0.02), ("Japanese", 0.02), ("Italian", 0.02))

CATEGORY_ASK = {
    "single": "needs exactly one of these functions to be called once",
    "parallel": "needs {k} independent things done or looked up at once, so that several function calls can be made together; every value those calls need is stated in the message",
    "sequential": "needs two steps, where the second function call needs a value that only the result of the first call provides",
    "multiturn": "needs one of these functions to be called",
    "error": "needs one of these functions to be called",
    "clarify": "wants something that needs the function {function} but does not give the required detail '{param}', which the assistant cannot guess",
    "no_tool": "asks a question related to {domain} that is answered from general knowledge without any of these functions (an explanation, a definition or advice)",
    "missing_tool": "asks for something in the general area of {domain} that none of these functions can do",
}


# --------------------------------------------------------------------------
# The parent's tokenizer + chat template
# --------------------------------------------------------------------------


class Renderer:
    """Pinned tokenizer.json + the parent's chat template (tools, thinking off)."""

    def __init__(self, parent_dir: Path, *, allow_unpinned: bool = False) -> None:
        from tokenizers import Tokenizer

        self.parent_dir = Path(parent_dir)
        self.tokenizer_sha256 = file_sha256(self.parent_dir / "tokenizer.json")
        self.template_sha256 = file_sha256(self.parent_dir / "chat_template.jinja")
        if not allow_unpinned:
            if self.tokenizer_sha256 != QWEN_TOKENIZER_SHA256:
                raise ToolcallError(f"tokenizer.json hashes {self.tokenizer_sha256}, pinned {QWEN_TOKENIZER_SHA256}")
            if self.template_sha256 != CHAT_TEMPLATE_SHA256:
                raise ToolcallError(f"chat_template.jinja hashes {self.template_sha256}, pinned {CHAT_TEMPLATE_SHA256}")
        self.fast = Tokenizer.from_file(str(self.parent_dir / "tokenizer.json"))
        self.fast.no_truncation()
        self.fast.no_padding()
        for piece, tid in TOOL_TOKEN_IDS.items():
            if self.fast.token_to_id(piece) != tid:
                raise ToolcallError(f"{piece} is id {self.fast.token_to_id(piece)} in this tokenizer, expected {tid}")
        self._hf = None
        self._lock = threading.Lock()

    def _template(self) -> Any:
        if self._hf is None:
            from transformers import AutoTokenizer

            self._hf = AutoTokenizer.from_pretrained(str(self.parent_dir))
        return self._hf

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, generation_prompt: bool) -> str:
        with self._lock:
            return str(self._template().apply_chat_template(messages, tools=tools or None, tokenize=False,
                                                            add_generation_prompt=generation_prompt, enable_thinking=False))

    def encode(self, text: str) -> list[int]:
        return list(self.fast.encode(text, add_special_tokens=False).ids)

    def encode_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self.fast.encode(text, add_special_tokens=False)
        return list(enc.ids), [tuple(o) for o in enc.offsets]

    def decode(self, ids: list[int]) -> str:
        return self.fast.decode([int(i) for i in ids], skip_special_tokens=False)

    def report(self) -> dict[str, Any]:
        return {"tokenizer_sha256": self.tokenizer_sha256, "chat_template_sha256": self.template_sha256,
                "render": "apply_chat_template(messages, tools=..., enable_thinking=False); assistant turns carry an empty <think></think> block",
                "add_special_tokens": False}

    def supervised(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        """Full render + target mask: every token of each assistant turn after its empty think
        block, its closing <|im_end|> included (EOS supervision).  Refuses when a span does not
        sit on token boundaries or a prefix render is not a prefix of the full render."""
        full = self.render(messages, tools, False)
        spans = []
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prefix = self.render(messages[:i], tools, True)
            upto = self.render(messages[: i + 1], tools, False)
            if not upto.startswith(prefix) or not full.startswith(upto):
                raise ToolcallError(f"assistant turn {i}: the generation-prompt render is not a prefix of the full render")
            if not upto.endswith("<|im_end|>\n"):
                raise ToolcallError(f"assistant turn {i} does not close with <|im_end|>")
            spans.append((len(prefix), len(upto) - 1))
        ids, offsets = self.encode_offsets(full)
        starts = {o[0]: t for t, o in enumerate(offsets)}
        ends = {o[1]: t for t, o in enumerate(offsets)}
        mask = [0] * len(ids)
        for start, end in spans:
            if start not in starts or end not in ends:
                raise ToolcallError(f"assistant span [{start}, {end}) does not sit on token boundaries")
            for t in range(starts[start], ends[end] + 1):
                mask[t] = 1
        if ids and max(ids) >= TEXT_VOCAB_ROWS:
            raise ToolcallError(f"rendered id {max(ids)} outside the text rows")
        return {"text": full, "ids": ids, "target_mask": mask, "spans": spans}


# --------------------------------------------------------------------------
# llama-server client
# --------------------------------------------------------------------------


class ServerGone(RuntimeError):
    pass


#: How long a request keeps retrying a server that refuses connections before the worker
#: gives the server up (its item goes back to the queue; resume continues it).
CLIENT_PATIENCE_S = 600.0


def http_json(url: str, payload: Any | None = None, timeout: float = 60.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="GET" if payload is None else "POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def server_props(url: str) -> dict[str, Any]:
    props = http_json(url.rstrip("/") + "/props", timeout=30)
    return {"model_path": props.get("model_path"), "total_slots": props.get("total_slots"), "build_info": props.get("build_info")}


class Client:
    """One llama-server.  Prompts go in as the parent's ids; the turn comes back as ids."""

    def __init__(self, url: str, renderer: Renderer, timeout: float = 1800.0, patience_s: float | None = None) -> None:
        self.url = url.rstrip("/")
        self.renderer = renderer
        self.timeout = timeout
        self.patience_s = CLIENT_PATIENCE_S if patience_s is None else patience_s

    def generate(self, prompt_ids: list[int], role: str, seed: int) -> dict[str, Any]:
        sampling = SAMPLING[role] if role in SAMPLING else CHECK_SAMPLING[role]
        payload = {"prompt": [int(i) for i in prompt_ids], "n_predict": sampling["n_predict"], "temperature": sampling["temperature"],
                   "top_k": sampling["top_k"], "top_p": sampling["top_p"], "min_p": sampling["min_p"], "seed": int(seed),
                   "cache_prompt": True, "return_tokens": True, "stream": False}
        started = time.time()
        delay = min(5.0, self.patience_s)
        while True:
            try:
                body = http_json(self.url + "/completion", payload, timeout=self.timeout)
                break
            except urllib.error.HTTPError as exc:
                # 4xx = this request is refused (e.g. beyond the slot context): the sample is
                # rejected, the server is fine.  5xx (loading, overloaded) is retried like a
                # dropped connection.  HTTPError is a URLError, so it is caught first.
                if 400 <= exc.code < 500:
                    raise _Reject(f"server refused the request: HTTP {exc.code}") from exc
                if time.time() - started > self.patience_s:
                    raise ServerGone(f"{self.url}: HTTP {exc.code}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                if time.time() - started > self.patience_s:
                    raise ServerGone(f"{self.url}: {exc}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        ids = [int(t) for t in body.get("tokens") or []]
        if ids and max(ids) >= TEXT_VOCAB_ROWS:
            raise ToolcallError(f"{self.url} sampled id {max(ids)} outside [0, {TEXT_VOCAB_ROWS})")
        end = "eos" if ids and ids[-1] in EOG_IDS else "server_only_eog" if ids and ids[-1] in SERVER_ONLY_EOG_IDS else "limit"
        body_ids = ids[:-1] if end != "limit" else ids
        return {"ids": body_ids, "text": self.renderer.decode(body_ids), "end": end, "stop_type": body.get("stop_type"),
                "tokens": len(ids), "wall_s": round(time.time() - started, 2), "server": self.url, "seed": int(seed), "role": role}


# --------------------------------------------------------------------------
# Writers: the parent as user, as tool backend, as schema designer
# --------------------------------------------------------------------------


def tool_brief(tools: list[dict[str, Any]]) -> str:
    lines = []
    for tool in tools:
        fn = tool_function(tool)
        props = fn["parameters"].get("properties") or {}
        req = set(fn["parameters"].get("required") or [])
        params = ", ".join(f"{k}{'' if k in req else '?'}: {first_type(v) or 'any'}" for k, v in props.items())
        lines.append(f"- {fn['name']}({params}): {fn['description']}")
    return "\n".join(lines)


def chat_ids(renderer: Renderer, system: str, user: str) -> list[int]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return renderer.encode(renderer.render(messages, None, True))


def clean_writer_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]
    return text.strip()


def parse_json_block(text: str) -> Any:
    text = clean_writer_text(text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        a, b = text.find(open_c), text.rfind(close_c)
        if a >= 0 and b > a:
            try:
                return json.loads(text[a: b + 1])
            except ValueError:
                continue
    raise ValueError("no JSON value in the text")


def user_request_prompt(item: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[str, str]:
    ask = CATEGORY_ASK[item["category"]].format(k=item.get("parallel_k", 2), function=item.get("clarify_function", ""),
                                                param=item.get("clarify_param", ""), domain=item["domain_text"])
    language = "" if item["language"] == "English" else f" Write it in {item['language']}."
    system = "You write the next message a person sends to an AI assistant. Output only that message: no quotes, no preamble, no explanation."
    user = (f"The assistant can use these functions:\n{tool_brief(tools)}\n\n"
            f"Write one realistic message from {item['persona']} who {ask}. Style: {item['style']}.{language} "
            f"Today is {TODAY}. Use concrete specifics (names, dates, ids, amounts, places) wherever a function needs them. "
            "Do not mention function names, tools or APIs.")
    return system, user


def followup_prompt(transcript: str, item: dict[str, Any], tools: list[dict[str, Any]], kind: str) -> tuple[str, str]:
    language = "" if item["language"] == "English" else f" Write it in {item['language']}."
    system = "You write the next message a person sends to an AI assistant. Output only that message: no quotes, no preamble, no explanation."
    if kind == "clarify":
        user = (f"Conversation so far:\n{transcript}\n\nThe assistant just asked the person a question. Write the person's short reply "
                f"that answers it with a concrete, plausible value.{language}")
    else:
        user = (f"The assistant can use these functions:\n{tool_brief(tools)}\n\nConversation so far:\n{transcript}\n\n"
                f"Write the person's natural follow-up request that builds on this conversation and needs another function call. "
                f"Style: {item['style']}.{language} Do not mention function names.")
    return system, user


def backend_prompt(tool: dict[str, Any], arguments: dict[str, Any]) -> tuple[str, str]:
    fn = tool_function(tool)
    system = f"You simulate API backends. Today is {TODAY}. Output only a JSON value, never prose or code fences."
    user = (f"Function: {fn['name']}\nDescription: {fn['description']}\nParameter schema: {json.dumps(fn['parameters'], ensure_ascii=False)}\n"
            f"Call arguments: {json.dumps(arguments, ensure_ascii=False)}\n\n"
            "Return the JSON body this function returns for this call: realistic and specific, consistent with the arguments, "
            "compact (a list holds at most 5 items). Return only the JSON.")
    return system, user


def schema_prompt(domain_text: str, flavor: str, count: int) -> tuple[str, str]:
    system = "You are a senior API designer. Output only JSON."
    user = (f"Design {count} functions that an AI assistant could call for {domain_text}, as offered by {flavor}.\n"
            "Return a JSON array. Each element: {\"name\": snake_case verb_object, \"description\": one or two sentences, "
            "\"parameters\": {\"type\": \"object\", \"properties\": {...}, \"required\": [...]}}. Each property has a \"type\" "
            "(string, integer, number, boolean, array with typed \"items\", or object) and a \"description\"; use \"enum\" where the "
            "values are a fixed set and state formats (YYYY-MM-DD, ISO codes, ids) in descriptions. Mix reads and writes, 1-6 "
            "parameters each, realistic required/optional split. Only the JSON array.")
    return system, user


def transcript_of(messages: list[dict[str, Any]], limit: int = 2500) -> str:
    lines = []
    for message in messages:
        if message["role"] == "system":
            continue
        if message["role"] == "assistant" and message.get("tool_calls"):
            calls = "; ".join(f"{c['function']['name']}({json.dumps(c['function']['arguments'], ensure_ascii=False)})" for c in message["tool_calls"])
            text = (message.get("content") or "") + f" [calls {calls}]"
        else:
            text = message.get("content") or ""
        lines.append(f"{message['role']}: {text.strip()}")
    joined = "\n".join(lines)
    return joined[-limit:]


# --------------------------------------------------------------------------
# Schema library
# --------------------------------------------------------------------------


def sandbox_toolsets() -> list[dict[str, Any]]:
    groups = {"sandbox_math": ["calculate", "percent_change", "split_bill"], "sandbox_units": ["convert_units", "currency_convert", "calculate"],
              "sandbox_dates": ["date_add", "days_between"], "sandbox_text": ["text_statistics", "calculate"]}
    return [{"toolset_id": key, "domain": key, "domain_text": "everyday calculations, conversions and dates", "flavor": "sandbox",
             "source": "sandbox (executed in Python)", "tools": [SANDBOX_TOOLS[n][0] for n in names]} for key, names in groups.items()]


def reserved_names() -> frozenset[str]:
    """Names a synthesized tool may never take: the sandbox tools (their behaviour is fixed by
    Python) and every bank function (HELDOUT_API is defined with the bank, further down)."""
    return frozenset(SANDBOX_TOOLS) | frozenset(HELDOUT_SANDBOX) | frozenset(HELDOUT_API)


def synthesize_toolset(client: Client, renderer: Renderer, domain: str, domain_text: str, flavor: str, index: int, base_seed: int,
                       attempts: int = 3) -> dict[str, Any]:
    """One parent-designed toolset.  Names are unique WITHIN the toolset only (1.0.3): two
    toolsets may both offer `get_stock_quote` with different schemas, as real APIs do; the
    plan dedupes names inside each conversation.  1.0.2 required library-wide uniqueness and
    refused most second and third toolsets of a domain on name collisions alone."""
    count = 3 + item_seed(base_seed, domain, index, "count") % 3
    record: dict[str, Any] = {"toolset_id": f"{domain}-{index}", "domain": domain, "domain_text": domain_text, "flavor": flavor,
                              "source": "parent-synthesized", "attempts": []}
    for attempt in range(attempts):
        seed = item_seed(base_seed, domain, index, attempt)
        system, user = schema_prompt(domain_text, flavor, count)
        try:
            out = client.generate(chat_ids(renderer, system, user), "schema", seed)
        except _Reject as why:
            record["attempts"].append({"attempt": attempt, "seed": seed, "refused": str(why)})
            continue
        note = {"attempt": attempt, "seed": seed, "end": out["end"], "tokens": out["tokens"], "server": out["server"], "wall_s": out["wall_s"]}
        record["attempts"].append(note)
        if out["end"] != "eos":
            note["refused"] = "not EOS-terminated"
            continue
        try:
            payload = parse_json_block(out["text"])
        except ValueError:
            note["refused"] = "no JSON array"
            continue
        if not isinstance(payload, list):
            note["refused"] = "not a JSON array"
            continue
        tools, dropped = [], []
        for raw in payload:
            problems = validate_tool(raw)
            if problems:
                dropped.append({"name": raw.get("name") if isinstance(raw, dict) else None, "problems": problems[:4]})
                continue
            tool = normalize_tool(raw)
            touch = heldout_touch(tool)
            if touch:
                dropped.append({"name": tool["function"]["name"], "problems": [f"touches held-out family {touch!r}"]})
                continue
            tools.append(tool)
        fresh, seen, reserved = [], set(), reserved_names()
        for tool in tools:
            name = tool["function"]["name"]
            if name in reserved or name in seen:
                dropped.append({"name": name, "problems": ["reserved name" if name in reserved else "name repeated in the toolset"]})
                continue
            seen.add(name)
            fresh.append(tool)
        note["dropped"] = dropped
        if len(fresh) >= 2:
            record.update({"tools": fresh, "raw_sha256": text_sha256(out["text"]), "accepted_attempt": attempt})
            return record
        note["refused"] = "fewer than 2 valid new functions"
    record["tools"] = []
    return record


def build_schemas(args: argparse.Namespace) -> int:
    renderer = Renderer(Path(args.parent_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / "journal.jsonl"
    journaled = read_jsonl(journal_path) if journal_path.exists() else []
    # A toolset counts as done only when it has tools: a journaled failure is retried (the
    # latest record of a toolset_id wins when the library is written).
    done = {r["toolset_id"]: r for r in journaled if r.get("tools")}
    jobs = [(d, text, FLAVORS[(i + k) % len(FLAVORS)], k) for i, (d, text) in enumerate(DOMAINS) for k in range(int(args.per_domain))]
    jobs = [j for j in jobs if f"{j[0]}-{j[3]}" not in done]
    lock = threading.Lock()
    clients = [Client(u, renderer) for u in args.server]
    log(f"schemas: {len(done)} toolsets journaled, {len(jobs)} to synthesize on {len(clients)} server(s) x {args.workers_per_server}")
    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)

    def worker(client: Client) -> None:
        while True:
            try:
                domain, text, flavor, k = work.get_nowait()
            except queue.Empty:
                return
            record = synthesize_toolset(client, renderer, domain, text, flavor, k, int(args.base_seed))
            with lock:
                with open(journal_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            log(f"schemas: {record['toolset_id']}: {len(record['tools'])} tools ({len(record['attempts'])} attempts)")

    with futures.ThreadPoolExecutor(max_workers=len(clients) * int(args.workers_per_server)) as pool:
        for f in [pool.submit(worker, c) for c in clients for _ in range(int(args.workers_per_server))]:
            f.result()
    records = {r["toolset_id"]: r for r in read_jsonl(journal_path)}
    library = sandbox_toolsets() + [records[k] for k in sorted(records) if records[k].get("tools")]
    sha, count = write_jsonl(out_dir / "schemas.jsonl", [{k: v for k, v in r.items() if k != "attempts"} for r in library])
    names = [t["function"]["name"] for r in library for t in r["tools"]]
    manifest = seal({
        "schema": SCHEMA + "/schemas", "tool": tool_identity(), "renderer": renderer.report(),
        "file": {"path": str(out_dir / "schemas.jsonl"), "sha256": sha}, "toolsets": count,
        "synthesized_toolsets": sum(1 for r in library if r["source"] == "parent-synthesized"),
        "failed_toolsets": sorted(k for k, r in records.items() if not r.get("tools")),
        "functions": len(names), "unique_function_names": len(set(names)),
        "domains": sorted({r["domain"] for r in library}), "heldout_keywords": list(HELDOUT_KEYWORDS),
        "sampling": SAMPLING["schema"], "servers": {u: server_props(u) for u in args.server},
        "written_utc": utc_now(),
    }, "manifest_sha256")
    write_json_atomic(out_dir / "manifest.json", manifest)
    log(f"schemas: {count} toolsets, {len(names)} functions; manifest {manifest['manifest_sha256'][:16]}")
    return 0


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


def weighted_choice(rng: random.Random, weights: dict[str, float] | tuple[tuple[str, float], ...]) -> str:
    items = list(weights.items()) if isinstance(weights, dict) else list(weights)
    total = sum(w for _, w in items)
    point = rng.random() * total
    for key, weight in items:
        point -= weight
        if point <= 0:
            return key
    return items[-1][0]


def build_plan(library: list[dict[str, Any]], items: int, base_seed: int) -> list[dict[str, Any]]:
    synthesized = [r for r in library if r["source"] == "parent-synthesized" and len(r["tools"]) >= 2]
    sandbox = [r for r in library if r["source"].startswith("sandbox")]
    if not synthesized or not sandbox:
        raise ToolcallError("the schema library needs synthesized and sandbox toolsets")
    plan = []
    for index in range(items):
        rng = random.Random(item_seed(base_seed, "plan", index))
        category = weighted_choice(rng, CATEGORY_WEIGHTS)
        mode = rng.random()
        primary = rng.choice(sandbox) if mode < 0.30 else rng.choice(synthesized)
        tools = list(primary["tools"])
        if primary["source"] != "parent-synthesized" and rng.random() < 0.5:
            extra = rng.choice(synthesized)
            tools += rng.sample(extra["tools"], k=min(2, len(extra["tools"])))
        elif primary["source"] == "parent-synthesized" and rng.random() < 0.15:
            tools += [SANDBOX_TOOLS[rng.choice(sorted(SANDBOX_TOOLS))][0]]
        others = [r for r in synthesized if r["domain"] != primary["domain"]]
        for _ in range(rng.choice((0, 0, 1, 1, 2, 3))):
            pick = rng.choice(others)
            tools.append(rng.choice(pick["tools"]))
        seen, unique = set(), []
        for tool in tools:
            if tool["function"]["name"] not in seen:
                seen.add(tool["function"]["name"])
                unique.append(tool)
        rng.shuffle(unique)
        leaked = [t["function"]["name"] for t in unique if t["function"]["name"] in HELDOUT_SANDBOX or t["function"]["name"] in HELDOUT_API]
        if leaked:
            raise ToolcallError(f"plan item {index} offers held-out bank functions {leaked}")
        item = {"index": index, "id": f"tc-{index:05d}", "category": category, "toolset_id": primary["toolset_id"], "domain": primary["domain"],
                "domain_text": primary["domain_text"], "tools": unique, "persona": rng.choice(PERSONAS), "style": rng.choice(STYLES),
                "language": weighted_choice(rng, LANGUAGES), "system": rng.choice((None, None, "default", "persona", "dated")),
                "parallel_k": rng.choice((2, 2, 3, 4)), "error_kind": rng.choice(ERROR_KINDS)}
        if category == "clarify":
            candidates = [(t["function"]["name"], p) for t in primary["tools"] for p in t["function"]["parameters"].get("required") or []]
            if candidates:
                item["clarify_function"], item["clarify_param"] = rng.choice(candidates)
            else:
                item["category"] = "single"
        plan.append(item)
    return plan


SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "persona": "You are a capable assistant for {domain}. Use the available functions when they help, and say plainly when something cannot be done.",
    "dated": f"You are a helpful assistant. Today's date is {TODAY}.",
}


def plan_command(args: argparse.Namespace) -> int:
    schemas_path = Path(args.schemas)
    library = read_jsonl(schemas_path)
    plan = build_plan(library, int(args.items), int(args.base_seed))
    sha, count = write_jsonl(Path(args.out), plan)
    manifest = seal({"schema": SCHEMA + "/plan", "tool": tool_identity(), "schemas": {"path": str(schemas_path), "sha256": file_sha256(schemas_path)},
                     "plan": {"path": str(Path(args.out)), "sha256": sha, "items": count}, "base_seed": int(args.base_seed),
                     "categories": dict(Counter(p["category"] for p in plan)), "languages": dict(Counter(p["language"] for p in plan)),
                     "written_utc": utc_now()}, "manifest_sha256")
    write_json_atomic(Path(args.out).with_suffix(".manifest.json"), manifest)
    log(f"plan: {count} items, sha {sha[:16]}; categories {manifest['categories']}")
    return 0


def continue_plan(plan: list[dict[str, Any]], cutoff: int) -> list[dict[str, Any]]:
    """1.2.0: the items after `cutoff`, re-indexed from 0 (the journal and `finalize` need a
    0-based plan-order prefix).  Ids, tools and scenarios are unchanged, so no item of the
    earlier run repeats and the ids stay unique across both runs; `source_index` is the
    item's index in the plan it came from."""
    if [p["index"] for p in plan] != list(range(len(plan))):
        raise ToolcallError("the plan is not indexed 0..n-1 in order")
    tail = [dict(item, index=n, source_index=item["index"]) for n, item in enumerate(plan[cutoff + 1:])]
    if not tail:
        raise ToolcallError(f"no plan item after cutoff index {cutoff}: nothing to continue")
    return tail


def continue_plan_command(args: argparse.Namespace) -> int:
    plan_path, after = Path(args.plan), Path(args.after_manifest)
    manifest = read_json(after)
    if not check_seal(manifest, "manifest_sha256"):
        raise ToolcallError(f"{after} does not match its seal")
    config = read_json(after.parent / "run_config.json")
    if not check_seal(config, "config_sha256") or config["config_sha256"] != manifest["run_config_sha256"]:
        raise ToolcallError(f"{after.parent / 'run_config.json'} is not the run config {after} was sealed under")
    plan_sha = file_sha256(plan_path)
    if config["plan"]["sha256"] != plan_sha:
        raise ToolcallError(f"{plan_path} (sha {plan_sha[:16]}) is not the plan that run consumed ({config['plan']['sha256'][:16]})")
    tail = continue_plan(read_jsonl(plan_path), int(manifest["cutoff_plan_index"]))
    out = Path(args.out)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tail).encode("utf-8")
    if out.exists() and file_sha256(out) != hashlib.sha256(body).hexdigest():
        raise ToolcallError(f"{out} exists with other bytes; a continuation plan is frozen once written")
    sha, count = write_jsonl(out, tail)
    parent_manifest_path = plan_path.with_suffix(".manifest.json")
    parent_manifest = read_json(parent_manifest_path) if parent_manifest_path.exists() else {}
    record = seal({"schema": SCHEMA + "/plan", "tool": tool_identity(), "schemas": parent_manifest.get("schemas"),
                   "plan": {"path": str(out), "sha256": sha, "items": count}, "base_seed": parent_manifest.get("base_seed"),
                   "continues": {"plan": {"path": str(plan_path), "sha256": plan_sha, "items": int(tail[-1]["source_index"]) + 1,
                                          "manifest_sha256": parent_manifest.get("manifest_sha256")},
                                 "after_run": {"manifest": str(after), "manifest_sha256": manifest["manifest_sha256"],
                                               "run_config_sha256": config["config_sha256"], "cutoff_plan_index": int(manifest["cutoff_plan_index"]),
                                               "conversations": manifest["conversations"], "tokens": manifest["tokens"]},
                                 "source_index_range": [int(tail[0]["source_index"]), int(tail[-1]["source_index"])]},
                   "categories": dict(Counter(p["category"] for p in tail)), "languages": dict(Counter(p["language"] for p in tail)),
                   "written_utc": utc_now()}, "manifest_sha256")
    write_json_atomic(out.with_suffix(".manifest.json"), record)
    log(f"continue-plan: {count} items (source index {tail[0]['source_index']}..{tail[-1]['source_index']}) after cutoff "
        f"{manifest['cutoff_plan_index']} of {after}; sha {sha[:16]}")
    return 0


# --------------------------------------------------------------------------
# One conversation
# --------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def result_values(result: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(result, dict):
        if "error" in result:
            return []
        for value in result.values():
            out += result_values(value)
    elif isinstance(result, list):
        for value in result:
            out += result_values(value)
    elif isinstance(result, (int, float)) and not isinstance(result, bool):
        out.append(result)
    elif isinstance(result, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", result):
        out.append(result)
    return out


def answer_numbers(answer: str) -> list[float]:
    """Numbers in an answer read both ways: comma as a thousands separator and as a decimal comma."""
    base = answer.replace(" ", "").replace(" ", "").replace("−", "-")
    numbers = []
    for variant in (base.replace(",", ""), re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", base).replace(",", ".")):
        for token in _NUM_RE.findall(variant):
            try:
                numbers.append(float(token.rstrip(".").replace(",", "")))
            except ValueError:
                continue
    return numbers


def grounded(answer: str, values: list[Any]) -> bool:
    """Does the final answer state at least one exact sandbox result (any rounding, any grouping)?
    ``values`` holds only computed results, never the call's own inputs echoed back."""
    if not values:
        return True
    numbers = answer_numbers(answer)
    for value in values:
        if isinstance(value, str):
            day = dt.date.fromisoformat(value)
            forms = (value, f"{day.strftime('%B')} {day.day}", f"{day.day} {day.strftime('%B')}", f"{day.month}/{day.day}",
                     f"{day.day}/{day.month}", f"{day.day}.{day.month}.", f"{day.month}月{day.day}日", f"{day.day:02d}.{day.month:02d}.")
            if any(form in answer for form in forms):
                return True
            continue
        first = 0 if abs(float(value)) >= 10 else 1
        for number in numbers:
            if abs(float(value) - number) < 1e-9:
                return True
            if any(abs(round(float(value), digits) - number) < 1e-9 for digits in range(first, 7)):
                return True
    return False


def computed_values(result: Any, arguments: dict[str, Any]) -> list[Any]:
    """Leaf results of a sandbox call minus the values the caller passed in."""
    given = [v for v in arguments.values() if isinstance(v, (int, float, str)) and not isinstance(v, bool)]
    return [v for v in result_values(result) if not any(v == g for g in given)]


def system_message(item: dict[str, Any]) -> list[dict[str, Any]]:
    key = item.get("system")
    if not key:
        return []
    return [{"role": "system", "content": SYSTEM_PROMPTS[key].format(domain=item["domain_text"])}]


class Conversation:
    """Drives one plan item against one server; every model call is recorded."""

    def __init__(self, item: dict[str, Any], client: Client, renderer: Renderer, base_seed: int, attempt: int) -> None:
        self.item = item
        self.client = client
        self.renderer = renderer
        self.base_seed = base_seed
        self.attempt = attempt
        self.calls_made = 0
        self.samples: list[dict[str, Any]] = []
        self.by_name = {tool_function(t)["name"]: t for t in item["tools"]}
        self.exec_values: list[Any] = []
        self.counter = 0

    def seed(self, role: str) -> int:
        self.counter += 1
        return item_seed(self.base_seed, self.item["id"], self.attempt, role, self.counter)

    def _gen(self, prompt_ids: list[int], role: str) -> dict[str, Any]:
        out = self.client.generate(prompt_ids, role, self.seed(role))
        self.samples.append({k: out[k] for k in ("role", "end", "stop_type", "tokens", "wall_s", "server", "seed")} | {"text": out["text"]})
        return out

    def write_user(self, system: str, user: str) -> str | None:
        out = self._gen(chat_ids(self.renderer, system, user), "user")
        text = clean_writer_text(out["text"])
        if out["end"] != "eos" or not text or len(text) > 2400 or "<tool_call>" in text:
            return None
        return text

    def backend(self, tool: dict[str, Any], arguments: dict[str, Any]) -> Any:
        name = tool_function(tool)["name"]
        if name in SANDBOX_TOOLS:
            result = execute_sandbox(name, arguments)
            self.exec_values += computed_values(result, arguments)
            return result
        # 1.0.4: 420 tokens / 2,000 characters cut list and catalog backends (smoke: 4 of 18
        # attempt rejections), which would have under-represented search/list tools.
        self.backend_failures = []
        for _ in range(2):
            system, user = backend_prompt(tool, arguments)
            out = self._gen(chat_ids(self.renderer, system, user), "backend")
            if out["end"] != "eos":
                self.backend_failures.append(f"end={out['end']}")
                continue
            try:
                value = parse_json_block(out["text"])
            except ValueError:
                self.backend_failures.append("not JSON")
                continue
            size = len(json.dumps(value, ensure_ascii=False))
            if isinstance(value, (dict, list)) and size <= BACKEND_MAX_CHARS:
                return value
            self.backend_failures.append(f"{type(value).__name__} of {size} chars")
        return None

    def assistant_turn(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self.renderer.encode(self.renderer.render(messages, self.item["tools"], True))
        if len(prompt) > MAX_RENDER_TOKENS - 256:
            raise _Reject("context: prompt beyond the render cap")
        out = self._gen(prompt, "assistant")
        if out["end"] != "eos":
            raise _Reject(f"assistant turn ended by {out['end']}")
        parsed = parse_assistant(out["text"], self.item["tools"])
        if parsed["problems"]:
            raise _Reject("invalid turn: " + "; ".join(parsed["problems"][:3]))
        if not parsed["calls"] and not parsed["content"]:
            raise _Reject("empty assistant turn")
        parsed["sampled_text"] = out["text"]
        return parsed

    def run_tools(self, messages: list[dict[str, Any]], parsed: dict[str, Any], inject: str | None) -> None:
        for n, call in enumerate(parsed["calls"]):
            tool = self.by_name[call["name"]]
            if inject and self.calls_made == 0 and n == 0:
                result = injected_error(inject, call, tool)
            else:
                result = self.backend(tool, call["arguments"])
                if result is None:
                    raise _Reject(f"backend for {call['name']} produced no usable JSON ({', '.join(self.backend_failures)})")
            self.calls_made += 1
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    def run_agent(self, messages: list[dict[str, Any]], max_turns: int, inject: str | None) -> list[dict[str, Any]]:
        """Assistant turns until one has no calls.  Returns the parsed turns."""
        turns = []
        for _ in range(max_turns):
            parsed = self.assistant_turn(messages)
            messages.append(assistant_message(parsed))
            turns.append(parsed)
            if not parsed["calls"]:
                return turns
            self.run_tools(messages, parsed, inject)
        raise _Reject("no final answer within the turn budget")

    def run(self) -> dict[str, Any]:
        item = self.item
        category = item["category"]
        messages = system_message(item)
        system, user = user_request_prompt(item, item["tools"])
        request = self.write_user(system, user)
        if request is None:
            raise _Reject("user writer gave no usable message")
        messages.append({"role": "user", "content": request})
        inject = item["error_kind"] if category == "error" else None
        segments = [self.run_agent(messages, 6, inject)]
        if category in ("clarify", "multiturn"):
            kind = "clarify" if category == "clarify" else "followup"
            fsys, fuser = followup_prompt(transcript_of(messages), item, item["tools"], kind)
            follow = self.write_user(fsys, fuser)
            if follow is None:
                raise _Reject("follow-up writer gave no usable message")
            messages.append({"role": "user", "content": follow})
            segments.append(self.run_agent(messages, 6, None))
        checks = category_checks(category, segments, item)
        if checks["failed"]:
            raise _Reject("category: " + "; ".join(checks["failed"]))
        final = segments[-1][-1]["content"]
        if not grounded(final, self.exec_values):
            raise _Reject("final answer states none of the sandbox results")
        content = None
        if category in ZERO_CALL_CATEGORIES:  # 1.1.0
            content = zero_call_check(self._gen, self.renderer, item["tools"], request, final)
            if content["failed"]:
                raise _Reject("content: " + "; ".join(content["failed"]))
        return {"messages": messages, "checks": checks, "content_check": content}


class _Reject(Exception):
    pass


def category_checks(category: str, segments: list[list[dict[str, Any]]], item: dict[str, Any]) -> dict[str, Any]:
    first = segments[0]
    calls_first = sum(len(t["calls"]) for t in first)
    turns_with_calls = sum(1 for t in first if t["calls"])
    max_parallel = max((len(t["calls"]) for t in first), default=0)
    failed = []
    if category in ("single", "error", "multiturn") and calls_first < 1:
        failed.append("no call for a tool request")
    if category == "parallel" and max_parallel < 2:
        failed.append("no turn with two or more calls")
    if category == "sequential" and turns_with_calls < 2:
        failed.append("fewer than two calling turns")
    if category in ("no_tool", "missing_tool") and calls_first:
        failed.append("called a function where none applies")
    if category == "clarify":
        if calls_first:
            failed.append("called before the missing detail was given")
        elif "?" not in first[-1]["content"] and "？" not in first[-1]["content"]:
            failed.append("no question asked")
        if len(segments) < 2 or not sum(len(t["calls"]) for t in segments[1]):
            failed.append("no call after the detail was given")
    if category == "multiturn" and (len(segments) < 2 or not sum(len(t["calls"]) for t in segments[1])):
        failed.append("no call for the follow-up")
    return {"failed": failed, "calls": sum(len(t["calls"]) for seg in segments for t in seg), "assistant_turns": sum(len(s) for s in segments),
            "max_parallel_calls": max(max((len(t["calls"]) for t in s), default=0) for s in segments)}


# --------------------------------------------------------------------------
# 1.1.0: the zero-call content rule (no_tool / missing_tool)
#
# Making zero calls is not enough.  Production row tc-00030 (no_tool, accepted on attempt 0
# under 1.0.4) answered a Spanish question about one specific reservation with "he verificado
# el estado actual de la reserva en el sistema" and a status ("confirmada") it never looked
# up -- the same habit the bank measured in the parent (clarify 8/15, invents values).  Two
# checks, both must pass:
#   1. CLAIM_PATTERNS: first-person "I checked / booked / sent ..." and "according to the
#      system" in the eight plan languages, unless the same clause talks about the
#      functions/tools themselves ("I checked the available functions; none can ...").
#   2. the parent as judge (greedy, CHECK_SAMPLING): could an offered function serve the
#      request (then a zero-call answer is wrong, or the user writer drifted), and does the
#      reply claim a lookup/action or state record-specific facts it could only know from one.
# --------------------------------------------------------------------------

_CLAIM_VERBS_EN = (r"checked|verified|looked up|looked into|pulled up|searched|queried|confirmed|booked|reserved|cancell?ed|"
                   r"created|updated|changed|modified|sent|scheduled|submitted|placed|processed|retrieved|accessed|reviewed your")
CLAIM_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    rf"\bI(?:'ve|’ve| have)?\s+(?:just\s+|already\s+|now\s+|also\s+|successfully\s+)?(?:{_CLAIM_VERBS_EN})\b",
    r"\b(?:according to|based on|per)\s+(?:the|our|your)\s+(?:system|records|database)\b",
    r"\b(?:our|my)\s+(?:system|records|database)\s+(?:shows?|indicates?|lists?|says|confirms?)\b",
    r"\bthe\s+(?:system|records|database)\s+(?:shows?|indicates?|confirms?)\s+(?:that\s+)?(?:your|you)\b",
    r"\bI can see (?:that |in |on |from )?(?:your|the) (?:account|order|reservation|booking|record|system)\b",
    r"\bsuccessfully (?:booked|reserved|cancell?ed|created|scheduled|submitted|processed|sent|placed|updated)\b",
    # Spanish / Italian (negated "no he ..." / "non ho ..." is honest, not a claim)
    r"(?<!no )\bhe\s+(?:verificado|comprobado|consultado|revisado|buscado|reservado|cancelado|creado|enviado|confirmado|actualizado|modificado|programado)\b",
    r"\b(?:según|de acuerdo con)\s+(?:el|nuestro|su)\s+sistema\b",
    r"\b(?:visible|registrad[oa]s?|figura)\s+en\s+(?:el|nuestro)\s+sistema\b",
    r"(?<!non )\bho\s+(?:già\s+)?(?:verificato|controllato|consultato|cercato|prenotato|cancellato|annullato|creato|inviato|confermato|aggiornato|modificato)\b",
    r"\b(?:secondo il|dal) (?:nostro )?sistema\b",
    # Portuguese
    r"\b(?:verifiquei|consultei|chequei|pesquisei|reservei|cancelei|criei|enviei|confirmei|atualizei|agendei)\b",
    r"\b(?:de acordo com|segundo) o (?:nosso |seu )?sistema\b",
    # German
    r"\b(?:ich\s+habe|habe\s+ich)\b[^.!?\n]{0,60}?\b(?:überprüft|geprüft|nachgesehen|nachgeschaut|gebucht|reserviert|storniert|erstellt|gesendet|verschickt|abgefragt|bestätigt|aktualisiert)\b",
    r"\blaut\s+(?:unserem|dem|Ihrem)\s+System\b",
    # French
    r"\bj['’]ai\s+(?:bien\s+|déjà\s+)?(?:vérifié|consulté|recherché|réservé|annulé|créé|envoyé|confirmé|mis à jour|modifié|programmé)",
    r"\b(?:selon|d['’]après)\s+(?:notre|le|votre)\s+système\b",
    # Chinese / Japanese
    r"我(?:已经|已|刚刚|刚)?(?:为您|帮您|替您)?(?:查询|查看|检查|核实|核对|确认|预订|预约|取消|创建|发送|搜索|调取)(?:了|过)",
    r"(?:根据|按照)(?:系统|我们的系统|记录)|系统(?:中)?(?:显示|记录显示)(?:您|你)",
    r"(?:確認|検索|予約|キャンセル|作成|送信|照会|お調べ|調べ)(?:いた)?しました|(?:確認|予約|キャンセル)済み",
    r"システム(?:上|で|に)(?:は)?[^。\n]{0,15}(?:確認|表示|記録)",
))
#: A match whose clause names the functions/tools themselves is about the toolset, not a system.
_TOOL_TALK_RE = re.compile(r"function|tool|capabilit|available actions|herramient|funci[oó]n|funcion|Funktion|Werkzeug|outil|fonction|"
                           r"ferrament|funç|strument|funzion|工具|函数|功能|ツール|関数|機能", re.IGNORECASE)
_CLAUSE_END_RE = re.compile(r"[.!?。！？\n]")


def claimed_actions(answer: str) -> list[str]:
    """Phrases in a zero-call answer that claim a lookup or an action (empty = none)."""
    found = []
    for rx in CLAIM_PATTERNS:
        for match in rx.finditer(answer):
            end = _CLAUSE_END_RE.search(answer, match.end())
            clause = answer[match.start(): end.start() if end else len(answer)]
            if not _TOOL_TALK_RE.search(clause):
                found.append(match.group(0))
    return found


def judge_prompt(tools: list[dict[str, Any]], request: str, answer: str) -> tuple[str, str]:
    system = "You review one exchange between a user and an AI assistant for a training-data filter. Output only JSON."
    user = (f"The assistant could call these functions:\n{tool_brief(tools)}\n\nUser message:\n{request[:2400]}\n\n"
            f"Assistant reply (it called NO function):\n{answer[:4000]}\n\n"
            "Answer two questions with true or false.\n"
            "fits: could one of the listed functions, called with values the user gave, retrieve, check or perform what the user "
            "is asking for?\n"
            "claims: does the reply say or imply that the assistant checked, looked up, verified, searched, booked, reserved, "
            "changed, sent or otherwise performed something in a system, or does it state facts about this user's specific "
            "records (a status, id, balance, availability, order or account detail) that it could only know from a lookup?\n"
            'Return exactly {"fits": true or false, "claims": true or false}.')
    return system, user


def parse_verdict(text: str) -> dict[str, bool] | None:
    try:
        value = parse_json_block(text)
    except ValueError:
        return None
    if not isinstance(value, dict) or not all(isinstance(value.get(k), bool) for k in ("fits", "claims")):
        return None
    return {"fits": value["fits"], "claims": value["claims"]}


def zero_call_check(gen: Callable[[list[int], str], dict[str, Any]], renderer: Renderer, tools: list[dict[str, Any]],
                    request: str, answer: str) -> dict[str, Any]:
    """The content rule for a no_tool / missing_tool row.  ``gen(prompt_ids, role)`` is one model
    call (Conversation._gen during a run, a bare Client.generate for re-checks of old rows)."""
    claims = claimed_actions(answer)
    check: dict[str, Any] = {"rules": CONTENT_RULES, "patterns": claims[:3], "judge": None, "judge_text": None, "failed": []}
    if claims:
        check["failed"].append(f"claims an action it did not take ({claims[0]!r})")
        return check  # the judge is not asked: the row fails either way
    system, user = judge_prompt(tools, request, answer)
    out = gen(chat_ids(renderer, system, user), "judge")
    check["judge_text"] = out["text"][:200]
    verdict = parse_verdict(out["text"]) if out["end"] == "eos" else None
    check["judge"] = verdict
    if verdict is None:
        check["failed"].append("judge gave no verdict")
    else:
        if verdict["fits"]:
            check["failed"].append("judge: an offered function serves this request")
        if verdict["claims"]:
            check["failed"].append("judge: the reply claims a lookup, an action or record facts")
    return check


def zero_call_texts(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """(first user request, final assistant answer) of a conversation."""
    request = next(m["content"] for m in messages if m["role"] == "user")
    answer = next(m["content"] for m in reversed(messages) if m["role"] == "assistant")
    return request, answer


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def locked_run_config(args: argparse.Namespace, renderer: Renderer) -> dict[str, Any]:
    plan_path = Path(args.plan)
    return {"schema": SCHEMA + "/run-config", "plan": {"path": str(plan_path.resolve()), "sha256": file_sha256(plan_path)},
            "renderer": renderer.report(), "sampling": SAMPLING, "base_seed": int(args.base_seed), "attempts": int(args.attempts),
            "token_target": int(args.token_target), "max_render_tokens": MAX_RENDER_TOKENS, "route": ROUTE,
            "gguf": {"path": args.gguf, "sha256": args.gguf_sha256}, "tool_major": TOOL_VERSION.split(".")[0]}


def load_journal(path: Path) -> dict[int, dict[str, Any]]:
    """Plan index -> its LAST journal record.  1.1.0: the journal stays append-only, and a later
    record of an index supersedes an earlier one (a content re-check verdict, a reopened row,
    the reopened row's new result)."""
    journal: dict[int, dict[str, Any]] = {}
    if path.exists():
        for record in read_jsonl(path):
            journal[record["index"]] = record
    return journal


def append_journal(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def stale_zero_call_rows(journal: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Accepted no_tool / missing_tool rows without a verdict under the current CONTENT_RULES."""
    return [r for _, r in sorted(journal.items()) if r["status"] == "accepted" and r["category"] in ZERO_CALL_CATEGORIES
            and (r.get("content_check") or {}).get("rules") != CONTENT_RULES]


def recheck_zero_call_rows(journal: dict[int, dict[str, Any]], journal_path: Path, client: Client, renderer: Renderer,
                           base_seed: int) -> None:
    """Apply the current content rule to rows accepted before it existed (1.0.x journals).
    A row that passes is re-journaled with its verdict; a row that fails is journaled as
    `reopened` and continues from its next attempt, carrying its rejections -- exactly what
    the rule would have done had it been there when the row was sampled."""
    stale = stale_zero_call_rows(journal)
    if not stale:
        return
    reopened = 0
    for record in stale:
        request, answer = zero_call_texts(record["messages"])
        seed = item_seed(base_seed, record["id"], record["attempt"], "judge", "recheck")
        check = zero_call_check(lambda ids, role: client.generate(ids, role, seed), renderer, record["tools"], request, answer)
        if check["failed"]:
            reopened += 1
            new = {"index": record["index"], "id": record["id"], "status": "reopened", "attempt": record["attempt"],
                   "category": record["category"], "domain": record["domain"], "content_check": check,
                   "rejections": list(record.get("rejections") or []) + [{"attempt": record["attempt"], "reason": "content: " + "; ".join(check["failed"]),
                                                                           "model_calls": len(record.get("samples") or []), "recheck": True}],
                   "superseded_render": record["render"], "finished_utc": utc_now()}
        else:
            new = dict(record, content_check=check)
        append_journal(journal_path, new)
        journal[record["index"]] = new
        log(f"run: content re-check {record['id']} {record['category']}: " + ("; ".join(check["failed"]) or "pass"))
    log(f"run: content rule {CONTENT_RULES} applied to {len(stale)} earlier rows: {reopened} reopened")


def parse_utc(text: str) -> float:
    """'2026-10-15T21:00:00Z' -> epoch seconds.  A deadline without a zone is refused."""
    stamp = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ToolcallError(f"--stop-by {text!r} has no UTC offset (write ...Z)")
    return stamp.timestamp()


def deadline_now() -> float:
    """The clock --stop-by is read against (its own function so a test can move it without
    moving the HTTP client's patience clock)."""
    return time.time()


def run_command(args: argparse.Namespace) -> int:
    # 1.2.0: --stop-by is a schedule, not semantics: it is NOT in the locked run_config, so a run
    # may be resumed with or without one.
    stop_by = parse_utc(args.stop_by) if getattr(args, "stop_by", None) else None

    def past_deadline() -> bool:
        return stop_by is not None and deadline_now() >= stop_by

    renderer = Renderer(Path(args.parent_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = locked_run_config(args, renderer)
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        old = read_json(config_path)
        if canonical_sha256({k: v for k, v in old.items() if k != "config_sha256"}) != canonical_sha256(config):
            raise ToolcallError(f"{config_path} records different semantics; refusing to resume into it")
    else:
        write_json_atomic(config_path, seal(config, "config_sha256"))
    urls = list(args.server)
    for url in urls:
        props = server_props(url)
        if args.gguf and props.get("model_path") and os.path.abspath(props["model_path"]) != os.path.abspath(args.gguf):
            raise ToolcallError(f"{url} serves {props['model_path']}, the locked GGUF is {args.gguf}")
    plan = read_jsonl(Path(args.plan))
    journal_path = out_dir / "journal.jsonl"
    journal = load_journal(journal_path)
    lock = threading.Lock()
    recheck_zero_call_rows(journal, journal_path, Client(urls[0], renderer), renderer, int(args.base_seed))
    state = {"accepted_tokens": sum(r["render"]["tokens"] for r in journal.values() if r["status"] == "accepted"),
             "accepted": sum(1 for r in journal.values() if r["status"] == "accepted"),
             "rejected": sum(1 for r in journal.values() if r["status"] == "rejected"), "dead_servers": set(),
             "internal_failures": 0, "abort": None}
    # Index order, so the journal is a plan-order prefix; once the target is reached only
    # items below the highest index already dispatched are still taken (a re-queued item
    # from a lost server is finished, so `finalize` never meets a gap).  A row the content
    # rule reopened continues from its next attempt with its earlier rejections.
    pending: queue.PriorityQueue = queue.PriorityQueue()
    for item in plan:
        old = journal.get(item["index"])
        if old is None:
            pending.put((item["index"], item))
        elif old["status"] == "reopened":
            pending.put((item["index"], dict(item, _resume={"start_attempt": int(old["attempt"]) + 1, "tries": old["rejections"]})))
    state["max_dispatched"] = max(journal) if journal else -1
    log(f"run: {len(journal)} journaled ({state['accepted']} accepted, {state['accepted_tokens']:,} tokens), "
        f"{pending.qsize()} pending, target {int(args.token_target):,}" + (f", stop by {args.stop_by}" if stop_by else ""))
    started = time.time()
    last_progress = [0.0]

    def write_progress(final: bool = False) -> None:
        now = time.time()
        if not final and now - last_progress[0] < float(args.progress_every):
            return
        last_progress[0] = now
        rate = state["accepted_tokens_session"] / max(1.0, now - started) if "accepted_tokens_session" in state else 0.0
        remaining = max(0, int(args.token_target) - state["accepted_tokens"])
        write_json_atomic(out_dir / "progress.json", {"accepted": state["accepted"], "rejected": state["rejected"],
                                                     "accepted_tokens": state["accepted_tokens"], "token_target": int(args.token_target),
                                                     "session_rate_tokens_per_s": round(rate, 2),
                                                     "eta_hours": round(remaining / rate / 3600, 2) if rate > 0 else None,
                                                     "pending": pending.qsize(), "dead_servers": sorted(state["dead_servers"]),
                                                     "stop_by": getattr(args, "stop_by", None), "updated_utc": utc_now()})

    state["accepted_tokens_session"] = 0

    def process(item: dict[str, Any], client: Client) -> dict[str, Any]:
        resume = item.get("_resume") or {}
        tries = list(resume.get("tries") or [])
        for attempt in range(int(resume.get("start_attempt", 0)), int(args.attempts)):
            if past_deadline():  # 1.2.0: no new attempt after --stop-by; the item is journaled (no gap)
                tries.append({"attempt": attempt, "reason": f"deadline: --stop-by {args.stop_by} passed before this attempt", "model_calls": 0})
                break
            conv = Conversation(item, client, renderer, int(args.base_seed), attempt)
            try:
                result = conv.run()
                try:
                    rendered = renderer.supervised(result["messages"], item["tools"])
                except ToolcallError as exc:
                    if "outside the text rows" in str(exc):
                        raise
                    raise _Reject(f"render: {exc}") from exc
                if len(rendered["ids"]) > MAX_RENDER_TOKENS:
                    raise _Reject("rendered conversation beyond the render cap")
                record = {"index": item["index"], "id": item["id"], "status": "accepted", "attempt": attempt, "category": item["category"],
                          "domain": item["domain"], "toolset_id": item["toolset_id"], "language": item["language"], "tools": item["tools"],
                          "messages": result["messages"], "checks": result["checks"], "samples": conv.samples, "rejections": tries,
                          "render": {"tokens": len(rendered["ids"]), "supervised": int(sum(rendered["target_mask"])),
                                     "text_sha256": text_sha256(rendered["text"])}, "finished_utc": utc_now()}
                if result["content_check"] is not None:
                    record["content_check"] = result["content_check"]
                return record
            except _Reject as why:
                tries.append({"attempt": attempt, "reason": str(why), "model_calls": len(conv.samples)})
            except (ToolcallError, ServerGone):
                raise
            except Exception as exc:  # noqa: BLE001 -- 1.1.0, see below
                # Anything else (a sandbox edge case, a parser corner) rejects this attempt with
                # its traceback instead of silently killing the worker thread and, at the end of
                # a 10-hour run, the whole run before finalize.  A systematic bug still stops
                # the run: past INTERNAL_FAILURE_LIMIT this session it is a functional failure.
                with lock:
                    state["internal_failures"] += 1
                    count = state["internal_failures"]
                text = f"{type(exc).__name__}: {exc}"
                log(f"run: {item['id']} attempt {attempt}: internal failure #{count}: {text[:200]}")
                if count > INTERNAL_FAILURE_LIMIT:
                    raise ToolcallError(f"{count} internal failures this session (> {INTERNAL_FAILURE_LIMIT}); last {item['id']}: {text[:300]}") from exc
                tries.append({"attempt": attempt, "reason": f"internal: {text[:300]}", "traceback": traceback.format_exc()[-3000:],
                              "model_calls": len(conv.samples)})
        return {"index": item["index"], "id": item["id"], "status": "rejected", "category": item["category"], "domain": item["domain"],
                "rejections": tries, "finished_utc": utc_now()}

    def worker(url: str) -> None:
        try:
            work(url)
        except BaseException as exc:
            # 1.1.0: say so at once and stop the other workers (they finish the item in hand);
            # the exception still reaches task.result() and the run exits non-zero.
            with lock:
                state["abort"] = state["abort"] or f"{url}: {type(exc).__name__}: {exc}"
            log(f"run: worker on {url} exited abnormally: {type(exc).__name__}: {str(exc)[:300]}")
            raise

    def work(url: str) -> None:
        client = Client(url, renderer)
        while True:
            with lock:
                if state["abort"]:
                    return
                try:
                    _, item = pending.get_nowait()
                except queue.Empty:
                    return
                if (state["accepted_tokens"] >= int(args.token_target) or past_deadline()) and item["index"] > state["max_dispatched"]:
                    pending.put((item["index"], item))
                    return
                state["max_dispatched"] = max(state["max_dispatched"], item["index"])
            try:
                record = process(item, client)
            except ServerGone as exc:
                pending.put((item["index"], item))
                with lock:
                    state["dead_servers"].add(url)
                log(f"run: server gone: {exc}")
                return
            with lock:
                append_journal(journal_path, record)
                if record["status"] == "accepted":
                    state["accepted"] += 1
                    state["accepted_tokens"] += record["render"]["tokens"]
                    state["accepted_tokens_session"] += record["render"]["tokens"]
                else:
                    state["rejected"] += 1
                write_progress()
            log(f"run: {record['id']} {record['category']} {record['status']}"
                + (f" {record['render']['tokens']} tok" if record["status"] == "accepted" else f" ({record['rejections'][-1]['reason'][:80]})"))

    with futures.ThreadPoolExecutor(max_workers=len(urls) * int(args.workers_per_server)) as pool:
        tasks = [pool.submit(worker, u) for u in urls for _ in range(int(args.workers_per_server))]
        for task in tasks:
            task.result()
    write_progress(final=True)
    if state["accepted_tokens"] >= int(args.token_target):
        log(f"run: token target reached ({state['accepted_tokens']:,})")
        return 0
    if past_deadline() and not state["abort"]:
        log(f"run: stopped at --stop-by {args.stop_by} ({state['accepted_tokens']:,} of {int(args.token_target):,} tokens); "
            "seal with finalize --accept-short")
        return 4
    if state["dead_servers"] and len(state["dead_servers"]) == len(urls):
        log("run: every server is gone; resumable")
        return 2
    log(f"run: plan exhausted before the target ({state['accepted_tokens']:,} tokens)")
    return 3


def content_audit_command(args: argparse.Namespace) -> int:
    """Read-only: the current content rule over every accepted no_tool / missing_tool row of a
    journal (never written to).  Receipt: one JSON line per row + a sealed summary."""
    renderer = Renderer(Path(args.parent_dir))
    client = Client(args.server, renderer)
    rows = [r for _, r in sorted(load_journal(Path(args.journal)).items())
            if r["status"] == "accepted" and r["category"] in ZERO_CALL_CATEGORIES]
    out = []
    for record in rows:
        request, answer = zero_call_texts(record["messages"])
        seed = item_seed(int(args.base_seed), record["id"], record["attempt"], "judge", "recheck")
        check = zero_call_check(lambda ids, role: client.generate(ids, role, seed), renderer, record["tools"], request, answer)
        out.append({"id": record["id"], "category": record["category"], "language": record.get("language"), "check": check,
                    "request": request[:600], "answer": answer[:900]})
        log(f"content-audit {record['id']} {record['category']} {record.get('language')}: " + ("; ".join(check["failed"]) or "pass"))
    sha, count = write_jsonl(Path(args.out), out)
    summary = seal({"schema": SCHEMA + "/content-audit", "tool": tool_identity(), "rules": CONTENT_RULES,
                    "journal": {"path": str(args.journal), "sha256": file_sha256(Path(args.journal))}, "rows": count,
                    "failed": sum(1 for r in out if r["check"]["failed"]), "pattern_hits": sum(1 for r in out if r["check"]["patterns"]),
                    "judge_fits": sum(1 for r in out if (r["check"]["judge"] or {}).get("fits")),
                    "judge_claims": sum(1 for r in out if (r["check"]["judge"] or {}).get("claims")),
                    "no_verdict": sum(1 for r in out if not r["check"]["patterns"] and r["check"]["judge"] is None),
                    "rows_file": {"path": str(args.out), "sha256": sha}, "audited_utc": utc_now()}, "receipt_sha256")
    write_json_atomic(Path(args.out).with_suffix(".summary.json"), summary)
    log(f"content-audit: {summary['failed']}/{count} rows fail {CONTENT_RULES} (patterns {summary['pattern_hits']}, judge fits "
        f"{summary['judge_fits']}, judge claims {summary['judge_claims']}, no verdict {summary['no_verdict']})")
    return 0


def status_command(args: argparse.Namespace) -> int:
    print(json.dumps(read_json(Path(args.out_dir) / "progress.json"), indent=1))
    return 0


# --------------------------------------------------------------------------
# finalize: cutoff, screen, manifest
# --------------------------------------------------------------------------


def screen_texts(record: dict[str, Any]) -> list[str]:
    """What the screen reads: everything except the template's fixed scaffolding."""
    out = []
    for message in record["messages"]:
        if message.get("content"):
            out.append(str(message["content"]))
        for call in message.get("tool_calls") or []:
            out.append(json.dumps(call["function"]["arguments"], ensure_ascii=False))
    for tool in record["tools"]:
        fn = tool_function(tool)
        out.append(fn["description"])
        out += [str(p.get("description", "")) for p in (fn["parameters"].get("properties") or {}).values()]
    return out


def bank_strings(bank: dict[str, Any]) -> list[tuple[str, str]]:
    found = []
    for case in bank["cases"]:
        for message in case["messages"]:
            if message.get("content"):
                found.append((case["id"], str(message["content"])))
    return found


def build_screen(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    sys.path.insert(0, str(Path(args.corpus_tools)))
    import distill_corpus_build as dcb  # noqa: E402

    dargs = dcb.parse_args(["screen"])
    index, sources = dcb.build_review_index(dargs)
    eval_path = Path(args.eval_split)
    n_eval = 0
    for row in read_jsonl(eval_path):
        index.add_text("corpus_eval_split", str(row.get("rid", n_eval)), (row.get("instruction") or "") + "\n" + (row.get("response") or ""))
        n_eval += 1
    bank_path = Path(args.bank)
    bank = read_json(bank_path)
    if not check_seal(bank, "bank_sha256"):
        raise ToolcallError(f"{bank_path} does not match its own seal")
    pairs = bank_strings(bank)
    for case_id, text in pairs:
        index.add_text("toolcall_bank_v1", case_id, text)
    extra = [{"name": "corpus_eval_split", "path": str(eval_path), "sha256": file_sha256(eval_path), "items": n_eval},
             {"name": "toolcall_bank_v1", "path": str(bank_path), "sha256": file_sha256(bank_path), "items": len(pairs)}]
    return index, {"review_sources": sources, "added_sources": extra, "index": index.summary(), "corpus_tool": dcb.tool_identity()}


def first_user_key(record: dict[str, Any]) -> str:
    first_user = next(m["content"] for m in record["messages"] if m["role"] == "user")
    return re.sub(r"\s+", " ", first_user.strip().lower())


def sealed_conversations(out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A finalized run's manifest and its conversations, both checked against the seal."""
    manifest = read_json(Path(out_dir) / "manifest.json")
    if not check_seal(manifest, "manifest_sha256"):
        raise ToolcallError(f"{out_dir}/manifest.json does not match its seal")
    path = Path(manifest["file"]["path"])
    if file_sha256(path) != manifest["file"]["sha256"]:
        raise ToolcallError(f"{path} no longer matches the manifest")
    return manifest, read_jsonl(path)


def finalize_command(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    cut_at_gap = bool(getattr(args, "cut_at_gap", False))
    if cut_at_gap and not args.accept_short:
        raise ToolcallError("--cut-at-gap seals a short prefix: it needs --accept-short")
    config = read_json(out_dir / "run_config.json")
    if not check_seal(config, "config_sha256"):
        raise ToolcallError("run_config.json does not match its seal")
    # 1.2.0: earlier runs this one continues.  Their ids must not recur (a functional failure: the
    # plans overlap), and a first user message an earlier run already keeps is screened out here.
    prior_ids: set[str] = set()
    prior_users: set[str] = set()
    prior_meta = []
    for prior in getattr(args, "prior_manifest", None) or []:
        prior_manifest, prior_rows = sealed_conversations(Path(prior).parent)
        prior_ids |= {r["id"] for r in prior_rows}
        prior_users |= {first_user_key(r) for r in prior_rows}
        prior_meta.append({"manifest": str(Path(prior)), "manifest_sha256": prior_manifest["manifest_sha256"],
                           "run_config_sha256": prior_manifest["run_config_sha256"], "conversations": len(prior_rows),
                           "tokens": prior_manifest["tokens"]})
    journal_path = out_dir / "journal.jsonl"
    raw_records = read_jsonl(journal_path)
    records = [r for _, r in sorted(load_journal(journal_path).items())]  # 1.1.0: the last record of an index wins
    target = int(config["token_target"])
    kept, total, cut, gap_at = [], 0, None, None
    for position, record in enumerate(records):
        if record["index"] != position:
            if cut_at_gap:  # 1.2.0: a run killed with an item in hand; seal the gap-free prefix
                gap_at = position
                records = records[:position]
                log(f"finalize: journal gap at plan index {position}: sealing indices 0..{position - 1} only (--cut-at-gap)")
                break
            raise ToolcallError(f"journal has a gap before plan index {record['index']}; resume the run first")
        if record["status"] == "reopened":
            raise ToolcallError(f"{record['id']} was reopened by the content rule and not re-run; resume the run first")
        if record["status"] == "accepted" and record["category"] in ZERO_CALL_CATEGORIES \
                and (record.get("content_check") or {}).get("rules") != CONTENT_RULES:
            raise ToolcallError(f"{record['id']} ({record['category']}) has no {CONTENT_RULES} verdict; resume the run first")
        if record["status"] == "accepted":
            kept.append(record)
            total += record["render"]["tokens"]
        if total >= target:
            cut = record["index"]
            break
    if cut is None:
        if not args.accept_short or not records:
            raise ToolcallError(f"accepted tokens {total:,} < target {target:,}; resume the run first (or --accept-short)")
        cut = records[-1]["index"]
        log(f"finalize: sealing SHORT of the target: {total:,} < {target:,} tokens (--accept-short)")
    index, screen_meta = build_screen(args)
    renderer = Renderer(Path(args.parent_dir))
    hits, clean, dup_users = [], [], set()
    for record in kept:
        if record["id"] in prior_ids:
            raise ToolcallError(f"{record['id']} is already a conversation of a --prior-manifest run: the plans overlap")
        found = []
        for text in screen_texts(record):
            found += index.hits(text)
        key = first_user_key(record)
        if found:
            hits.append({"id": record["id"], "reasons": found[:3]})
            continue
        if key in dup_users:
            hits.append({"id": record["id"], "reasons": [{"rule": "duplicate_first_user_message"}]})
            continue
        if key in prior_users:
            hits.append({"id": record["id"], "reasons": [{"rule": "duplicate_first_user_message_prior_run"}]})
            continue
        dup_users.add(key)
        rendered = renderer.supervised(record["messages"], record["tools"])
        if text_sha256(rendered["text"]) != record["render"]["text_sha256"]:
            raise ToolcallError(f"{record['id']}: the render no longer reproduces its journaled sha")
        clean.append(record)
    sha, count = write_jsonl(out_dir / "conversations.jsonl", [{k: v for k, v in r.items() if k != "samples"} for r in clean])
    samples_sha, _ = write_jsonl(out_dir / "samples.jsonl", [{"id": r["id"], "samples": r["samples"]} for r in clean])
    all_by_cat = Counter(r["category"] for r in records[: cut + 1])
    ok_by_cat = Counter(r["category"] for r in records[: cut + 1] if r["status"] == "accepted")
    reasons = Counter(t["reason"].split(":")[0] for r in records[: cut + 1] for t in r.get("rejections", []))
    manifest = seal({
        "schema": SCHEMA + "/manifest", "tool": tool_identity(), "run_config_sha256": config["config_sha256"],
        "set": "native tool-use self-gen (parent Q4_K_M sampler; logits rescored in bf16): D2 text-KD rows on route tool_native; stage-T SFT rows with the sampled turns as ground truth",
        "plan_items_consumed": cut + 1, "cutoff_plan_index": cut, "conversations": count,
        "token_target": target, "short_of_target": total < target,
        "prior_manifests": prior_meta, "cut_at_gap": gap_at,
        "deadline_rejected_items": sum(1 for r in records[: cut + 1] if r["status"] == "rejected"
                                       and any(t["reason"].startswith("deadline") for t in r.get("rejections", []))),
        "tokens": sum(r["render"]["tokens"] for r in clean), "supervised_tokens": sum(r["render"]["supervised"] for r in clean),
        "screened_out": len(hits), "screen_hits": hits[:50], "screen": screen_meta,
        "acceptance_by_category": {c: {"planned": all_by_cat[c], "accepted": ok_by_cat[c]} for c in sorted(all_by_cat)},
        "attempt_rejections_by_reason": dict(reasons.most_common()),
        "content_rule": {"rules": CONTENT_RULES, "categories": list(ZERO_CALL_CATEGORIES), "judge_sampling": CHECK_SAMPLING["judge"],
                         "rejected_attempts": sum(1 for r in records[: cut + 1] for t in r.get("rejections", []) if t["reason"].startswith("content")),
                         "reopened_rows": sorted({r["id"] for r in raw_records if r["status"] == "reopened" and r["index"] <= cut}),
                         "judge_verdicts_kept": sum(1 for r in clean if (r.get("content_check") or {}).get("judge"))},
        "internal_failure_attempts": sum(1 for r in records[: cut + 1] for t in r.get("rejections", []) if t["reason"].startswith("internal")),
        "by_category": dict(Counter(r["category"] for r in clean)), "by_language": dict(Counter(r["language"] for r in clean)),
        "domains": len({r["domain"] for r in clean}), "functions_used": len({c["function"]["name"] for r in clean for m in r["messages"] for c in m.get("tool_calls") or []}),
        "calls": sum(r["checks"]["calls"] for r in clean), "parallel_turns": sum(1 for r in clean for m in r["messages"] if len(m.get("tool_calls") or []) >= 2),
        "file": {"path": str(out_dir / "conversations.jsonl"), "sha256": sha}, "samples_file": {"path": str(out_dir / "samples.jsonl"), "sha256": samples_sha},
        "licence": "generated by the parent Qwen3.8-27B-Uncensored (licences cleared by Cyril 2026-10-12); schemas parent-synthesized or written here; no external dataset",
        "sealed_utc": utc_now(),
    }, "manifest_sha256")
    write_json_atomic(out_dir / "manifest.json", manifest)
    log(f"finalize: {count} conversations, {manifest['tokens']:,} tokens ({manifest['supervised_tokens']:,} supervised), "
        f"{len(hits)} screened out; manifest {manifest['manifest_sha256'][:16]}")
    return 0


def export_command(args: argparse.Namespace) -> int:
    import numpy as np
    from safetensors.numpy import save_file

    # 1.2.0: one or more sealed runs (a run and its continuations, in the order given) -> one corpus.
    out_dirs = [Path(d) for d in (args.out_dir if isinstance(args.out_dir, list) else [args.out_dir])]
    records, sources = [], []
    for out_dir in out_dirs:
        manifest, rows = sealed_conversations(out_dir)
        sources.append({"out_dir": str(out_dir), "manifest_sha256": manifest["manifest_sha256"], "run_config_sha256": manifest["run_config_sha256"],
                        "row_range": [len(records), len(records) + len(rows)], "conversations": len(rows),
                        "tokens": manifest["tokens"], "supervised_tokens": manifest["supervised_tokens"]})
        records += rows
    if len(sources) != len({s["manifest_sha256"] for s in sources}):
        raise ToolcallError("the same sealed run is named twice")
    seen_ids = Counter(r["id"] for r in records)
    if max(seen_ids.values()) > 1:
        raise ToolcallError(f"conversation ids repeat across the runs: {[i for i, n in seen_ids.items() if n > 1][:5]}")
    source_sha = "+".join(s["manifest_sha256"] for s in sources)
    renderer = Renderer(Path(args.parent_dir))
    rendered = [renderer.supervised(r["messages"], r["tools"]) for r in records]
    width = max(len(x["ids"]) for x in rendered)
    ids = np.full((len(rendered), width), PAD_ID, dtype=np.int32)
    target = np.zeros((len(rendered), width), dtype=np.uint8)
    lengths = np.zeros((len(rendered),), dtype=np.int64)
    for n, x in enumerate(rendered):
        ids[n, : len(x["ids"])] = x["ids"]
        target[n, : len(x["ids"])] = x["target_mask"]
        lengths[n] = len(x["ids"])
        if text_sha256(x["text"]) != records[n]["render"]["text_sha256"]:
            raise ToolcallError(f"{records[n]['id']}: render drifted from the sealed sha")
    if int(target[:, 0].sum()):
        raise ToolcallError("a row supervises its first position (nothing predicts it)")
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    save_file({"input_ids": ids, "lengths": lengths, "target_mask": target}, str(tmp),
              metadata={"source_manifest_sha256": source_sha, "tool": f"{TOOL_NAME} {TOOL_VERSION}", "route": ROUTE})
    os.replace(tmp, dest)
    sidecar = seal({"schema": SCHEMA + "/rescore-corpus", "tool": tool_identity(),
                    "corpus": {"path": str(dest.resolve()), "sha256": file_sha256(dest), "bytes": dest.stat().st_size},
                    "source_manifest_sha256": source_sha, "source_manifests": sources, "rows": len(records), "row_length": width,
                    "tokens": int(lengths.sum()), "supervised_tokens": int(target.sum()), "pad_id": PAD_ID,
                    "mask_rule": "target_mask[n, t] = 1 for every token of an assistant turn after its empty think block, <|im_end|> included; qtext_teacher_cache shifts it (store t iff target t+1)",
                    "row_ids": [r["id"] for r in records], "categories": [r["category"] for r in records], "route": ROUTE},
                   "receipt_sha256")
    write_json_atomic(dest.with_suffix(".rows.json"), sidecar)
    log(f"export: {len(records)} rows x {width}, {int(lengths.sum()):,} tokens, {int(target.sum()):,} supervised from "
        f"{len(sources)} run(s) -> {dest}")
    return 0


# --------------------------------------------------------------------------
# The held-out bank
# --------------------------------------------------------------------------

HELDOUT_API = {
    "get_flight_status": _fn("get_flight_status", "Live status of a commercial flight on a given date.",
                             {"flight_number": {"type": "string", "description": "IATA flight number, e.g. LH404"},
                              "date": {"type": "string", "description": "Departure date YYYY-MM-DD"}}, ["flight_number", "date"]),
    "refill_prescription": _fn("refill_prescription", "Request a refill of an existing prescription for pickup at a pharmacy branch.",
                               {"prescription_id": {"type": "string", "description": "Prescription number, e.g. RX-55102"},
                                "pharmacy_location": {"type": "string", "description": "Branch name or town"}}, ["prescription_id", "pharmacy_location"]),
    "book_meeting_room": _fn("book_meeting_room", "Reserve an office meeting room.",
                             {"room_id": {"type": "string", "description": "Room identifier, e.g. B2-Orion"},
                              "date": {"type": "string", "description": "YYYY-MM-DD"},
                              "start_time": {"type": "string", "description": "HH:MM, 24-hour"},
                              "duration_minutes": {"type": "integer", "description": "Length of the booking"}}, ["room_id", "date", "start_time", "duration_minutes"]),
    "track_parcel": _fn("track_parcel", "Current location and delivery estimate of a parcel.",
                        {"carrier": {"type": "string", "enum": ["dhl", "ups", "fedex", "usps", "royal_mail"], "description": "Carrier"},
                         "tracking_number": {"type": "string", "description": "The carrier's tracking number"}}, ["carrier", "tracking_number"]),
    "search_recipes": _fn("search_recipes", "Find recipes by ingredients, time and diet.",
                          {"ingredients": {"type": "array", "items": {"type": "string"}, "description": "Ingredients to use"},
                           "max_minutes": {"type": "integer", "description": "Maximum total time"},
                           "diet": {"type": "string", "enum": ["any", "vegetarian", "vegan", "gluten_free"], "description": "Diet filter"}}, ["ingredients"]),
    "create_support_ticket": _fn("create_support_ticket", "Open a customer support ticket.",
                                 {"customer_email": {"type": "string", "description": "Customer's email address"},
                                  "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"], "description": "Priority"},
                                  "summary": {"type": "string", "description": "One-line summary of the problem"}}, ["customer_email", "priority", "summary"]),
    "get_air_quality": _fn("get_air_quality", "Current air-quality reading for a city.",
                           {"city": {"type": "string", "description": "City name"},
                            "pollutant": {"type": "string", "enum": ["pm25", "pm10", "o3", "no2", "aqi"], "description": "Which measure"}}, ["city", "pollutant"]),
    "reserve_campsite": _fn("reserve_campsite", "Reserve a campsite in a national or state park.",
                            {"park_name": {"type": "string", "description": "Park name"},
                             "arrival_date": {"type": "string", "description": "YYYY-MM-DD"},
                             "nights": {"type": "integer", "description": "Number of nights"},
                             "tent_count": {"type": "integer", "description": "Number of tents"}}, ["park_name", "arrival_date", "nights", "tent_count"]),
}

BANK_LANES = ("exec_single", "exec_parallel", "api_select", "multi_turn", "error_recovery", "no_tool", "missing_tool", "clarify")
_CITIES = [("Lisbon", 38.7223, -9.1393), ("Nairobi", -1.2921, 36.8219), ("Osaka", 34.6937, 135.5023), ("Denver", 39.7392, -104.9903),
           ("Helsinki", 60.1699, 24.9384), ("Santiago", -33.4489, -70.6693), ("Toronto", 43.6532, -79.3832), ("Mumbai", 19.076, 72.8777)]


def _bank_exec_case(rng: random.Random) -> tuple[str, dict[str, Any]]:
    kind = rng.choice(sorted(HELDOUT_SANDBOX))
    if kind == "loan_payment":
        p, r, m = rng.choice([150000, 240000, 18500, 32000, 410000]), rng.choice([4.25, 5.5, 6.75, 7.1, 3.9]), rng.choice([36, 60, 180, 360])
        return f"What would the monthly payment be on a {p} loan at {r}% annual interest over {m} months?", {"name": kind, "arguments": {"principal": p, "annual_rate_percent": r, "months": m}}
    if kind == "geo_distance":
        (a, la1, lo1), (b, la2, lo2) = rng.sample(_CITIES, 2)
        unit = rng.choice(["km", "mi"])
        return (f"How far apart are {a} ({la1}, {lo1}) and {b} ({la2}, {lo2}) as the crow flies, in {'kilometres' if unit == 'km' else 'miles'}?",
                {"name": kind, "arguments": {"lat1": la1, "lon1": lo1, "lat2": la2, "lon2": lo2, "unit": unit}})
    if kind == "bmi_calculate":
        w, h = rng.choice([58, 72.5, 91, 104, 66]), rng.choice([162, 175, 181, 158, 190])
        return f"I weigh {w} kg and I'm {h} cm tall. What's my BMI?", {"name": kind, "arguments": {"weight_kg": w, "height_cm": h}}
    if kind == "timezone_convert":
        t, a, b = rng.choice(["09:30", "14:15", "23:40", "06:05"]), rng.choice([-5, -8, 1, 0]), rng.choice([5.5, 9, -3, 2])
        return f"If it is {t} at UTC{a:+g}, what time is it at UTC{b:+g}?", {"name": kind, "arguments": {"time": t, "from_utc_offset": a, "to_utc_offset": b}}
    p, r, y, n = rng.choice([5000, 12000, 800, 25000]), rng.choice([3.5, 4.8, 6.0]), rng.choice([5, 10, 2.5]), rng.choice([1, 4, 12])
    return (f"If I invest {p} at {r}% a year, compounded {n} times a year, how much will I have after {y} years?",
            {"name": "compound_interest", "arguments": {"principal": p, "annual_rate_percent": r, "years": y, "compounds_per_year": n}})


def _bank_api_case(rng: random.Random) -> tuple[str, dict[str, Any]]:
    kind = rng.choice(sorted(HELDOUT_API))
    if kind == "get_flight_status":
        f, d = rng.choice(["LH404", "QF12", "AF1381", "UA857"]), rng.choice(["2026-11-02", "2026-12-19", "2026-10-30"])
        return f"Is flight {f} on {d} on time?", {"name": kind, "arguments": {"flight_number": f, "date": d}}
    if kind == "refill_prescription":
        rx, loc = rng.choice(["RX-55102", "RX-90311", "RX-17760"]), rng.choice(["Maple Street", "Riverside", "Downtown"])
        return f"Please refill prescription {rx}, I'll pick it up at the {loc} branch.", {"name": kind, "arguments": {"prescription_id": rx, "pharmacy_location": loc}}
    if kind == "book_meeting_room":
        room, d, t, m = rng.choice(["B2-Orion", "A1-Vega", "C3-Lyra"]), rng.choice(["2026-10-21", "2026-11-04"]), rng.choice(["09:00", "13:30", "16:00"]), rng.choice([30, 45, 60, 90])
        return f"Book room {room} on {d} at {t} for {m} minutes.", {"name": kind, "arguments": {"room_id": room, "date": d, "start_time": t, "duration_minutes": m}}
    if kind == "track_parcel":
        c, n = rng.choice([("ups", "UPS"), ("dhl", "DHL"), ("fedex", "FedEx")]), rng.choice(["1Z999AA10123456784", "JD014600006281234567", "7489 2231 0098"])
        return f"Where is my {c[1]} parcel {n}?", {"name": kind, "arguments": {"carrier": c[0], "tracking_number": n}}
    if kind == "search_recipes":
        ing, mins, diet = rng.choice([["chickpeas", "spinach"], ["salmon", "lemon"], ["tofu", "broccoli"]]), rng.choice([20, 30, 45]), rng.choice(["vegetarian", "vegan", "gluten_free"])
        return f"Find me a {diet.replace('_', '-')} recipe with {ing[0]} and {ing[1]} that takes at most {mins} minutes.", {"name": kind, "arguments": {"ingredients": ing, "max_minutes": mins, "diet": diet}}
    if kind == "create_support_ticket":
        e, p = rng.choice(["dana.ortiz@example.com", "k.mbeki@example.org"]), rng.choice(["high", "urgent", "low"])
        return (f"Open a {p}-priority ticket for {e}: their invoice PDF won't download.",
                {"name": kind, "arguments": {"customer_email": e, "priority": p}, "ungraded_required": ["summary"]})
    if kind == "get_air_quality":
        c, pol = rng.choice(["Delhi", "Krakow", "Mexico City", "Seoul"]), rng.choice([("pm25", "PM2.5"), ("no2", "NO2"), ("o3", "ozone")])
        return f"What's the {pol[1]} level in {c} right now?", {"name": kind, "arguments": {"city": c, "pollutant": pol[0]}}
    park, d, n, t = rng.choice(["Yosemite", "Zion", "Acadia"]), rng.choice(["2026-11-14", "2027-05-22"]), rng.choice([2, 3, 4]), rng.choice([1, 2])
    return f"Reserve a campsite at {park} arriving {d} for {n} nights, {t} tent{'s' if t > 1 else ''}.", {"name": "reserve_campsite", "arguments": {"park_name": park, "arrival_date": d, "nights": n, "tent_count": t}}


def _offer(rng: random.Random, pool: dict[str, Any], must: list[str], k: int) -> list[dict[str, Any]]:
    names = list(dict.fromkeys(must + rng.sample(sorted(n for n in pool if n not in must), k=max(0, min(k, len(pool) - len(must))))))
    rng.shuffle(names)
    return [pool[n] if isinstance(pool[n], dict) and pool[n].get("type") == "function" else pool[n][0] for n in names]


def build_bank(seed: int = 20261014) -> dict[str, Any]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    sandbox_pool = dict(HELDOUT_SANDBOX)
    api_pool = dict(HELDOUT_API)
    seen: set[str] = set()

    def add(lane: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], expected: dict[str, Any]) -> None:
        key = canonical_sha256([messages, [t["function"]["name"] for t in tools]])
        if key in seen:
            return
        seen.add(key)
        cases.append({"id": f"tcb-{lane}-{sum(1 for c in cases if c['lane'] == lane):03d}", "lane": lane, "messages": messages, "tools": tools, "expected": expected})

    for target, lane, builder, pool in ((40, "exec_single", _bank_exec_case, sandbox_pool), (40, "api_select", _bank_api_case, api_pool)):
        guard = 0
        while sum(1 for c in cases if c["lane"] == lane) < target and guard < 2000:
            guard += 1
            text, call = builder(rng)
            add(lane, [{"role": "user", "content": text}], _offer(rng, pool, [call["name"]], 2), {"calls": [call], "parallel": False})
    guard = 0
    while sum(1 for c in cases if c["lane"] == "exec_parallel") < 20 and guard < 2000:
        guard += 1
        (t1, c1), (t2, c2) = _bank_exec_case(rng), _bank_exec_case(rng)
        if c1 == c2:
            continue
        add("exec_parallel", [{"role": "user", "content": f"Two things. {t1} Also, {t2[0].lower() + t2[1:]}"}],
            _offer(rng, sandbox_pool, list(dict.fromkeys([c1["name"], c2["name"]])), 1), {"calls": [c1, c2], "parallel": True})
    guard = 0
    while sum(1 for c in cases if c["lane"] == "multi_turn") < 15 and guard < 2000:
        guard += 1
        kind = rng.choice(["bmi_calculate", "loan_payment", "timezone_convert"])
        while True:
            t1, c1 = _bank_exec_case(rng)
            if c1["name"] == kind:
                break
        result = execute_sandbox(kind, c1["arguments"], HELDOUT_SANDBOX)
        c2 = json.loads(json.dumps(c1))
        if kind == "bmi_calculate":
            c2["arguments"]["weight_kg"] = c1["arguments"]["weight_kg"] - 5
            follow = f"And if I lost 5 kg, so {c2['arguments']['weight_kg']} kg at the same height?"
        elif kind == "loan_payment":
            c2["arguments"]["months"] = c1["arguments"]["months"] * 2
            follow = f"What about over {c2['arguments']['months']} months instead, same amount and rate?"
        else:
            c2["arguments"]["time"] = "18:45"
            follow = "Same two offsets, but for 18:45?"
        messages = [{"role": "user", "content": t1},
                    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": c1}]},
                    {"role": "tool", "content": json.dumps(result)},
                    {"role": "assistant", "content": f"Here is the result: {json.dumps(result)}."},
                    {"role": "user", "content": follow}]
        add("multi_turn", messages, _offer(rng, sandbox_pool, [kind], 2), {"calls": [c2], "parallel": False})
    for n in range(15):
        unit = rng.choice(["km", "mi"])
        (a, la1, lo1), (b, la2, lo2) = rng.sample(_CITIES, 2)
        bad = {"name": "geo_distance", "arguments": {"lat1": la1, "lon1": lo1, "lat2": la2, "lon2": lo2, "unit": "kilometers" if unit == "km" else "miles"}}
        good = json.loads(json.dumps(bad))
        good["arguments"]["unit"] = unit
        messages = [{"role": "user", "content": f"Distance from {a} ({la1}, {lo1}) to {b} ({la2}, {lo2}) in {'kilometers' if unit == 'km' else 'miles'}, please."},
                    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": bad}]},
                    {"role": "tool", "content": json.dumps(_err("INVALID_ARGUMENT", "Parameter 'unit' must be one of ['km', 'mi'].", parameter="unit"))}]
        add("error_recovery", messages, _offer(rng, sandbox_pool, ["geo_distance"], 1), {"calls": [good], "parallel": False})
    general = ["What does APR mean on a loan statement?", "Why do time zones exist at all?", "Is BMI a good measure of health for athletes?",
               "Explain the difference between simple and compound interest in two sentences.", "What is a great circle?",
               "Who invented the idea of the airline hub-and-spoke model?", "Can you give me three tips for packing a tent?",
               "What is PM2.5 and why does it matter?", "How do I politely ask a colleague to free up a meeting room?",
               "What should I include in a good bug report?", "Write a haiku about autumn rain.", "What's 12 times 12?",
               "Summarize what a recipe's 'mise en place' means.", "How long should I keep a prescription label?",
               "What is the capital of Australia?", "Translate 'good morning' into Spanish.", "Is a parcel the same as a package?",
               "What does a 'tent footprint' do?", "Give me one sentence about why Lisbon is called the city of seven hills.",
               "What does UTC stand for?"]
    for n, text in enumerate(general):
        pool = sandbox_pool if n % 2 else api_pool
        add("no_tool", [{"role": "user", "content": text}], _offer(rng, pool, [], 3), {"no_call": True})
    missing = [("Order me a large pepperoni pizza to my office.", api_pool), ("Send an SMS to my brother saying I'm late.", sandbox_pool),
               ("Play some jazz on the living-room speaker.", api_pool), ("Cancel my gym membership.", sandbox_pool),
               ("Book me a taxi to the airport at 6 am tomorrow.", sandbox_pool), ("Check my bank balance.", api_pool),
               ("Turn off the lights in the kitchen.", api_pool), ("Post this photo to my Instagram.", sandbox_pool),
               ("What's the weather in Oslo tomorrow?", sandbox_pool), ("Renew my library books.", api_pool),
               ("Transfer 200 euros to my landlord.", sandbox_pool), ("Find me a dentist near me with Saturday hours.", sandbox_pool),
               ("Reset my email password.", api_pool), ("Buy two tickets for the Coldplay concert.", sandbox_pool),
               ("Schedule a tweet for 9 am.", api_pool), ("Unlock my front door.", sandbox_pool),
               ("Show me today's top news headlines.", api_pool), ("Start a 25-minute focus timer.", api_pool),
               ("Translate this contract into German and email it.", sandbox_pool), ("Find the cheapest car rental in Denver.", api_pool)]
    for text, pool in missing:
        add("missing_tool", [{"role": "user", "content": text}], _offer(rng, pool, [], 3), {"no_call": True})
    vague = [("What would my monthly loan payment be over 60 months?", "loan_payment"), ("How far is it from Toronto to my hometown?", "geo_distance"),
             ("What's my BMI? I'm 180 cm.", "bmi_calculate"), ("Convert 3 pm to India time.", "timezone_convert"),
             ("How much will my savings grow at 5% a year?", "compound_interest"), ("Is my flight tomorrow on time?", "get_flight_status"),
             ("Refill my prescription please.", "refill_prescription"), ("Book a meeting room for tomorrow afternoon.", "book_meeting_room"),
             ("Where's my parcel?", "track_parcel"), ("Open a support ticket about the login bug.", "create_support_ticket"),
             ("Reserve a campsite at Zion.", "reserve_campsite"), ("How's the air today?", "get_air_quality"),
             ("Calculate the loan payment at 6% for 30 years.", "loan_payment"), ("What's the distance between my two offices?", "geo_distance"),
             ("I lost some weight, what's my BMI now?", "bmi_calculate")]
    for text, name in vague:
        pool = sandbox_pool if name in sandbox_pool else api_pool
        add("clarify", [{"role": "user", "content": text}], _offer(rng, pool, [name], 1), {"no_call": True, "asks": True})
    bank = {"schema": SCHEMA + "/bank", "name": "omni_toolcall_bank_v1", "seed": seed, "today": TODAY,
            "render": "parent chat template with tools, enable_thinking=False, add_generation_prompt=True; greedy, 768 new tokens",
            "lanes": {lane: sum(1 for c in cases if c["lane"] == lane) for lane in BANK_LANES},
            "heldout_functions": sorted(list(HELDOUT_SANDBOX) + list(HELDOUT_API)),
            "grading": {"calls": "first assistant turn; every expected call matched once by name and by every expected argument "
                                 "(free-text required arguments listed in ungraded_required must be present, any value) "
                                 "(numbers within 1e-6 relative or 0.011 absolute, strings case- and space-insensitive, arrays as sets); "
                                 "no extra calls; no unknown parameters; parallel lanes need all calls in that one turn",
                        "no_call": "no <tool_call> in the turn and a non-empty answer", "asks": "no call and the answer contains a question mark"},
            "cases": cases}
    return seal(bank, "bank_sha256")


def _values_equal(expected: Any, got: Any) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected == got
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return abs(expected - got) <= max(1e-6 * abs(expected), 0.011)
    if isinstance(expected, (int, float)) and isinstance(got, str):
        try:
            return _values_equal(expected, float(got.strip()))
        except ValueError:
            return False
    if isinstance(expected, str) and isinstance(got, str):
        return " ".join(expected.lower().split()) == " ".join(got.lower().split())
    if isinstance(expected, list) and isinstance(got, list):
        return sorted(" ".join(str(x).lower().split()) for x in expected) == sorted(" ".join(str(x).lower().split()) for x in got)
    return expected == got


def grade_case(case: dict[str, Any], response: str) -> dict[str, Any]:
    text = response
    for eog in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(eog)[0]
    parsed = parse_assistant(text, case["tools"])
    expected = case["expected"]
    result = {"id": case["id"], "lane": case["lane"], "calls": parsed["calls"], "problems": parsed["problems"]}
    if expected.get("no_call"):
        ok = not parsed["calls"] and bool(parsed["content"]) and "<tool_call>" not in text
        if expected.get("asks"):
            ok = ok and ("?" in parsed["content"] or "？" in parsed["content"])
        return dict(result, passed=ok)
    if parsed["problems"] or len(parsed["calls"]) != len(expected["calls"]):
        return dict(result, passed=False)
    remaining = list(parsed["calls"])
    for want in expected["calls"]:
        match = next((c for c in remaining if c["name"] == want["name"]
                      and all(k in c["arguments"] and _values_equal(v, c["arguments"][k]) for k, v in want["arguments"].items())
                      and all(k in c["arguments"] for k in want.get("ungraded_required", []))), None)
        if match is None:
            return dict(result, passed=False)
        remaining.remove(match)
    return dict(result, passed=True)


def bank_prompt_ids(renderer: Renderer, case: dict[str, Any]) -> list[int]:
    return renderer.encode(renderer.render(case["messages"], case["tools"], True))


def bank_build_command(args: argparse.Namespace) -> int:
    bank = build_bank(int(args.seed))
    renderer = Renderer(Path(args.parent_dir)) if args.parent_dir else None
    if renderer is not None:
        for case in bank["cases"]:
            ids = bank_prompt_ids(renderer, case)
            if max(ids) >= TEXT_VOCAB_ROWS:
                raise ToolcallError(f"{case['id']}: prompt id outside the text rows")
    dest = Path(args.out)
    if dest.exists() and read_json(dest).get("bank_sha256") != bank["bank_sha256"]:
        raise ToolcallError(f"{dest} exists with a different bank; a frozen bank is never overwritten")
    write_json_atomic(dest, bank)
    log(f"bank: {len(bank['cases'])} cases {bank['lanes']}; sha {bank['bank_sha256'][:16]} -> {dest}")
    return 0


def bank_run_command(args: argparse.Namespace) -> int:
    bank = read_json(Path(args.bank))
    if not check_seal(bank, "bank_sha256"):
        raise ToolcallError("bank does not match its seal")
    renderer = Renderer(Path(args.parent_dir))
    clients = [Client(u, renderer) for u in args.server]
    work = [(n, case, clients[n % len(clients)]) for n, case in enumerate(bank["cases"])]

    def one(job: tuple[int, dict[str, Any], Client]) -> dict[str, Any]:
        _, case, client = job
        out = client.generate(bank_prompt_ids(renderer, case), "bank", 1234)
        return {"id": case["id"], "lane": case["lane"], "response": out["text"], "end": out["end"], "tokens": out["tokens"],
                "wall_s": out["wall_s"], "server": out["server"]}

    # Greedy decoding: the answer does not depend on which server or slot ran it (up to the
    # batch non-determinism of llama.cpp, recorded in the grade receipt's label).
    with futures.ThreadPoolExecutor(max_workers=len(clients) * int(args.workers_per_server)) as pool:
        rows = list(pool.map(one, work))
    sha, count = write_jsonl(Path(args.out), rows)
    log(f"bank-run: {count} responses from {args.server} -> {args.out} ({sha[:16]})")
    return 0


def bank_grade_command(args: argparse.Namespace) -> int:
    bank_path = Path(args.bank)
    bank = read_json(bank_path)
    if not check_seal(bank, "bank_sha256"):
        raise ToolcallError("bank does not match its seal")
    responses = {r["id"]: r for r in read_jsonl(Path(args.responses))}
    missing = [c["id"] for c in bank["cases"] if c["id"] not in responses]
    if missing:
        raise ToolcallError(f"{len(missing)} bank cases have no response, e.g. {missing[:3]}")
    graded = [grade_case(c, responses[c["id"]]["response"]) for c in bank["cases"]]
    lanes = {lane: {"passed": sum(1 for g in graded if g["lane"] == lane and g["passed"]), "cases": sum(1 for g in graded if g["lane"] == lane)}
             for lane in BANK_LANES}
    receipt = seal({"schema": SCHEMA + "/bank-grade", "tool": tool_identity(), "bank": {"path": str(bank_path), "sha256": bank["bank_sha256"]},
                    "responses": {"path": str(Path(args.responses)), "sha256": file_sha256(Path(args.responses))}, "label": args.label,
                    "lanes": lanes, "passed": sum(1 for g in graded if g["passed"]), "cases": len(graded),
                    "failures": [g for g in graded if not g["passed"]][:80], "graded_utc": utc_now()}, "receipt_sha256")
    write_json_atomic(Path(args.out), receipt)
    log(f"bank-grade [{args.label}]: {receipt['passed']}/{receipt['cases']} " + " ".join(f"{k}={v['passed']}/{v['cases']}" for k, v in lanes.items()))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("schemas")
    sp.add_argument("--server", action="append", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--per-domain", type=int, default=3)
    sp.add_argument("--workers-per-server", type=int, default=4)
    sp.add_argument("--base-seed", type=int, default=20261014)
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))

    sp = sub.add_parser("plan")
    sp.add_argument("--schemas", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--items", type=int, default=2400)
    sp.add_argument("--base-seed", type=int, default=20261014)

    sp = sub.add_parser("continue-plan")
    sp.add_argument("--plan", required=True, help="the plan the earlier run consumed (its run_config names its sha)")
    sp.add_argument("--after-manifest", required=True, help="the earlier run's sealed manifest.json (its cutoff_plan_index)")
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("run")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--server", action="append", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--token-target", type=int, default=1_500_000)
    sp.add_argument("--attempts", type=int, default=3)
    sp.add_argument("--workers-per-server", type=int, default=4)
    sp.add_argument("--progress-every", type=float, default=60.0)
    sp.add_argument("--base-seed", type=int, default=20261014)
    sp.add_argument("--gguf", default=None)
    sp.add_argument("--gguf-sha256", default=None)
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))
    sp.add_argument("--stop-by", default=None, help="UTC deadline, e.g. 2026-10-15T21:00:00Z (not part of the run lock); exit 4 when it ends the run")

    sp = sub.add_parser("status")
    sp.add_argument("--out-dir", required=True)

    sp = sub.add_parser("content-audit")
    sp.add_argument("--journal", required=True)
    sp.add_argument("--server", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--base-seed", type=int, default=20261014)
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))

    sp = sub.add_parser("finalize")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--bank", default=str(DEFAULT_BANK))
    sp.add_argument("--eval-split", default=str(DEFAULT_EVAL_SPLIT))
    sp.add_argument("--corpus-tools", default=str(HERE))
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))
    sp.add_argument("--accept-short", action="store_true", help="seal below the token target (the shortfall is recorded)")
    sp.add_argument("--prior-manifest", action="append", default=[], help="1.2.0: a sealed run this one continues (repeatable)")
    sp.add_argument("--cut-at-gap", action="store_true", help="1.2.0, with --accept-short: seal the prefix before a journal gap")

    sp = sub.add_parser("export")
    sp.add_argument("--out-dir", action="append", required=True, help="a finalized run dir; repeat for a run and its continuations")
    sp.add_argument("--dest", required=True)
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))

    sp = sub.add_parser("bank-build")
    sp.add_argument("--out", default=str(DEFAULT_BANK))
    sp.add_argument("--seed", type=int, default=20261014)
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))

    sp = sub.add_parser("bank-run")
    sp.add_argument("--bank", default=str(DEFAULT_BANK))
    sp.add_argument("--server", action="append", required=True)
    sp.add_argument("--workers-per-server", type=int, default=4)
    sp.add_argument("--out", required=True)
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))

    sp = sub.add_parser("bank-grade")
    sp.add_argument("--bank", default=str(DEFAULT_BANK))
    sp.add_argument("--responses", required=True)
    sp.add_argument("--label", required=True)
    sp.add_argument("--out", required=True)
    return p


COMMANDS = {"schemas": build_schemas, "plan": plan_command, "continue-plan": continue_plan_command, "run": run_command, "status": status_command,
            "content-audit": content_audit_command, "finalize": finalize_command,
            "export": export_command, "bank-build": bank_build_command, "bank-run": bank_run_command, "bank-grade": bank_grade_command}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.command}")
    try:
        return COMMANDS[args.command](args)
    except ToolcallError as exc:
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
