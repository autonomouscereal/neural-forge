#!/usr/bin/env python3
"""quality_scorer.py -- quality scorer v1: per-row data-quality scores for the registry, the D2 text set and intake batches.

WHY THIS FILE EXISTS.  Research lane (Claude), 2026-10-15, pilot "Data 1" of the data/training program
(research/DATA_AND_TRAINING_PROGRAM_2026-10-14.md section 2.2, as amended by the adversarial critique of the same day):
D2 spends 20.0M tokens of which about 46.6 % fall in hard-defect classes (data-audit/full/audit4_report.json), and
nothing in the pipeline scores a row before it is sealed into a teacher cache.  This tool is that scoring stage, CPU only,
deterministic, no model.  It READS sealed corpora and never writes next to them: every output goes to --out-dir
(default under /mnt/lexar/omni-campaign/research/data/).

WHAT IT IS NOT.  It is not the promote/repair stage: whether a row is dropped, masked or repaired is L-DATA's call (Codex,
STAGE_D_RECONCILED_PLAN_2026-10-14 section 7).  `action` in every record is ADVISORY, computed from the decision table of
the program doc so L-DATA can read one column instead of four audit scripts.  `score` is a ranking aid, never a gate.
No PostgreSQL ledger yet (the critique deferred it until approval); files are the record.

CHECKS (task list: dedup/near-dup, structure validity, length/truncation, repetition, tool-call validity, verifiable
answers; plus the eval screen and the audit's disjoint class).
  class      the audit4 disjoint problem class (data_audit4.py priority order, same regexes) so counts reconcile with
             audit4_report.json.
  screen     distill_corpus_build.build_review_index (imported by path, sha recorded) + the corpus eval split + the
             toolcall bank + Codex's corpus-schema-v2 validation/test splits.  A hit is a hard reject.  `--no-screen`
             for fixtures.
  trunc      dataset cap signature (audit2 rule: max >= 500 chars and >= max(20, 10 %) of rows within max(30, 1 %) of it)
             and the row sitting at it; mid-word / mid-sentence endings; unclosed <think>.
  struct     empty, mojibake, flattened hex transcript, newline-collapsed code, JSON parse of JSON-shaped responses,
             balanced code fences, Python ast.parse of python blocks (parse only, nothing is executed).
  rep        zlib ratio, duplicate-line fraction, top word-4-gram share; `degenerate` only outside ARC grids and JSON
             (the audit showed those are false positives).
  tool       tool calls in the response (Qwen XML <tool_call><function=..>, Hermes <tool_call>{json}, JSON-in-prose
             {"tool_calls": [...]}): count, parse errors, argument objects, and -- when the instruction carries function
             specs -- name offered / required parameters present.
  verify     math: a STRICT reference (last balanced \\boxed{} after </think>, else an "Answer:" / "Final answer:" /
             "the answer is:" line of <= 60 chars) and whether the problem is in the instruction (not a teacher-JSON
             wrapper, not a generic header); arithmetic claims "a op b = c" re-computed exactly (Fraction); ARC grid
             rows; code rows with tests.  audit4's looser FINAL regex counted 515 "extractable" math rows; most of its
             captures are "." or a teacher-JSON number, which is why this rule is strict.
  dedup      exact (instruction+response) and response-only sha256; MinHash word-5-gram, 128 permutations, LSH 16 x 8,
             verified estimated Jaccard >= 0.8; union-find clusters; one representative per cluster (train split,
             fewest flags, closed ending, longest).  Batch mode dedups against a ledger's signatures (--ledger), so a
             new batch is compared with the whole promoted set, not only with itself.

D2 EOS CHECK (registry mode).  For every D2 document whose row ends cut (class 06 / 08 / 14) the last four packed ids
of the document are read from the sealed text shards and decoded: this prints whether the packer gives a cut document
an <|im_end|> + separator target (the program doc's amendment A question).

Subcommands
-----------
  registry   staging/rows.jsonl + carve.json + text/documents.jsonl (+ text shards for the EOS check) -> --out-dir
  batch      an intake batch dir (manifest.json schema omni.intake.batch.v1 + rows.jsonl) -> <batch>/quality/<version>/
             (or --out-dir); --ledger DIR dedups against a previous run's minhash.npz
  summary    re-print a run's summary.json

Outputs (<out-dir>):  scores.jsonl (one record per row), clusters.jsonl (near-dup clusters of size >= 2),
minhash.npz (signatures + exact keys: the ledger for later batches), d2_eos_check.json (registry mode), summary.json,
receipt.json (every input sha256, this tool's sha256, the screen sources).

FAILURES ARE FUNCTIONAL, NUMBERS ARE NOT.  A batch whose rows do not hash to its manifest, an input file that is
missing, a screen source that cannot be built (unless --no-screen): non-zero exit.  Rates and scores are diagnostics.

    PY=/home/cereal/model-surgery-lab/venv/bin/python
    R=/mnt/lexar/omni-campaign/research
    nice -n 15 env CUDA_VISIBLE_DEVICES= $PY $R/tools/quality_scorer.py registry --out-dir $R/data/quality-v1-registry
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator

TOOL_NAME = "quality_scorer"
TOOL_VERSION = "1.0.4"   # 1.0.4 (2026-10-15): arithmetic skips LaTeX chains, refutations, truncated decimals, last-place rounding; 1.0.3: letter-x product chains skipped by the arithmetic check; 1.0.2: answer forms (units, backticks, percent, choice letters); 1.0.1: arithmetic thousands / percent false positives, split screen by near-dup fraction; 1.0.0 first version
SCORE_SCHEMA = "omni-research.quality-score.v1"
INTAKE_SCHEMA = "omni.intake.batch.v1"

OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
DEFAULT_ROWS = OC / "distill" / "corpus" / "staging" / "rows.jsonl"
DEFAULT_CARVE = OC / "distill" / "corpus" / "carve.json"
DEFAULT_DOCS = OC / "distill" / "corpus" / "text" / "documents.jsonl"
DEFAULT_TEXT_MANIFEST = OC / "distill" / "corpus" / "text" / "manifest.json"
DEFAULT_CORPUS_TOOLS = OC / "distill" / "tools"
DEFAULT_EVAL_SPLIT = OC / "distill" / "corpus" / "eval_split" / "eval_rows.jsonl"
DEFAULT_TOOLCALL_BANK = OC / "calibration" / "banks" / "omni_toolcall_bank_v1.json"
DEFAULT_CODEX_SPLITS = (OC / "distill" / "codex-preparation-01a1326f-20261013" / "corpus-schema-v2" / "validation.jsonl",
                        OC / "distill" / "codex-preparation-01a1326f-20261013" / "corpus-schema-v2" / "test.jsonl")
DEFAULT_BFCL_HOLDOUT = OC / "distill" / "corpus" / "holdouts" / "bfcl62_rids.json"

#: Parent tokenizer ids used by the D2 EOS check (selfgen_sampler / distill_corpus_build pins).
IM_END_ID = 248_046
SEPARATOR_ID = 248_044

# ------------------------------------------------------------------------------------------- audit4 class regexes
GEN = ("model this high-quality text sample", "model this high-quality long-context document")
TEACH = "you are a teacher model producing compact structured distillation targets"
GENERIC_MATH = ("solve the source-grounded math problem",)
HEX = re.compile(r"^\s*[0-9a-f]{32}\s+(user|system)\b")
CODEISH = re.compile(r"(```|\bdef \w+\(|\bimport \w+|\breturn\b|#include|function\s*\(|\bclass \w+|\bpublic \w+|SELECT .* FROM)")
OMNI = re.compile(r"(omnicoder|G:/omnicoder|mega_lane_variant|resume_cursor|active_v9_mutation|training_orchestration)", re.I)
PROSE_END = re.compile(r"[.!?)\]\"'`*:;>]\s*$")
WEBNAV = re.compile(r"(Log Out|Log in|Sign up|Subscribe|Click here|All rights reserved|Copyright ©|Privacy Policy|Terms of (Use|Service)|Share this|Related Posts|Leave a (comment|reply)|Posted by|Read more|cookies)", re.I)
ADULT = re.compile(r"\b(porn|xxx|nude|erotic|sex(y|ual)? (video|scene)|escort|viagra|casino|payday loan)\b", re.I)
ASSERT = re.compile(r"^\s*assert\s|def test_|unittest|pytest|assert\s+\w+\(", re.M)
TEMPLATED_DS = re.compile(r"(lowio|kepler_|dirac_|curie_|erdos_|bounded_|socrates_|goodall_|lagrange_|herschel_|feynman_|noether_|turing_|gauss_)")
SPLIT_TRAILER = re.compile(r"\s(train|validation|test)\s*$")
MOJIBAKE = re.compile(r"(\ufffd|Ã[\x80-\xbf©¨¢£¤¥¦§ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿]|â€[™œ\x9d“”˜¦]|Â[\xa0-\xbf])")
FENCE = re.compile(r"^\s*```", re.M)
PY_BLOCK = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S | re.I)
WORD = re.compile(r"[a-z0-9]+")

CLASS_LABELS = {
    "01": "bfcl_groundtruth_format", "02": "teacher_json_wrapper", "03": "flattened_hex_transcript",
    "04": "execute_structured_tool_call_json", "05": "arc_grid_answer_only", "06": "think_unclosed_no_answer",
    "07": "newline_collapsed_code", "08": "generic_web_cut_midsentence", "09": "generic_web_complete",
    "10": "omnicoder_pipeline_selfreference", "11": "templated_synthetic_family", "12": "memory_trace_rewrites",
    "13": "empty", "14": "other_cut_midword", "15": "other_unflagged",
}

# ------------------------------------------------------------------------------------------- small helpers


class ScorerError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def tool_identity() -> dict[str, Any]:
    me = Path(__file__).resolve()
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "path": str(me), "sha256": file_sha256(me)}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


# ------------------------------------------------------------------------------------------- math answers
ANSWER_LINE = re.compile(r"(?im)^\s*(?:\*\*)?\s*(?:(?:therefore|thus|so|hence),?\s+)?(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(.{1,60}?)\s*(?:\*\*)?\s*$")


def boxed_all(text: str) -> list[str]:
    """Every \\boxed{...} / \\fbox{...} body with balanced braces, in order."""
    out = []
    for marker in ("\\boxed{", "\\fbox{"):
        i = 0
        while True:
            j = text.find(marker, i)
            if j < 0:
                break
            k = j + len(marker)
            depth, start = 1, k
            while k < len(text) and depth:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                k += 1
            if depth == 0:
                out.append((j, text[start:k - 1]))
            i = k
    return [body for _, body in sorted(out)]


def after_think(text: str) -> str:
    return text.split("</think>")[-1]


def extract_reference(response: str) -> tuple[str | None, str]:
    """(answer, rule) from a source response; None when no strict rule applies."""
    tail = after_think(response)
    boxed = boxed_all(tail)
    if boxed:
        return boxed[-1].strip(), "boxed"
    lines = ANSWER_LINE.findall(tail)
    if lines:
        value = lines[-1].strip().rstrip(".").strip()
        if value and re.search(r"[0-9A-Za-z\\]", value):
            return value, "answer_line"
    return None, "none"


def extract_final_answer(text: str) -> tuple[str | None, str]:
    """(answer, rule) from a generated response: post-think boxed, else an answer line, else Codex's last-line numeric
    rule (evaluation.grade_arithmetic: a final line ending with a number / fraction)."""
    if "<think>" in text and "</think>" not in text:
        return None, "unclosed_think"
    answer, rule = extract_reference(text)
    if answer is not None:
        return answer, rule
    tail = after_think(text).strip()
    line = tail.splitlines()[-1] if tail else ""
    m = re.search(r"(?<![\w.])([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)\s*[.!]?\s*$", line)
    if m:
        return m.group(1), "last_line_numeric"
    return None, "none"


_LATEX_DROP = ("\\left", "\\right", "\\displaystyle", "\\!", "\\,", "\\;", "\\:", "\\ ", "$", "\\quad", "\\qquad")


def normalize_answer(value: str) -> str:
    s = value.strip()
    s = re.sub(r"\\text(?:bf|rm|it|normal)?\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    for token in _LATEX_DROP:
        s = s.replace(token, "")
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\^\{?\\circ\}?|°|\\degree", "", s)
    s = s.replace("\\%", "%").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"^(?:[a-zA-Z]\w*|\\[a-zA-Z]+(?:\{[^{}]*\})?)\s*=\s*", "", s)   # "x = 3" -> "3"
    s = re.sub(r"\s+", "", s).rstrip(".").lower()
    s = re.sub(r"^\(?[a-e]\)(?=\S)", "", s)                                # "(C) 8" -> "8"
    m = re.fullmatch(r"\(?\(?([a-e])\)?\)?(?:\S*)?", s)
    if m and len(s) <= 6 and not re.search(r"\d", s):
        return m.group(1)
    return s


def number_parts(value: str) -> tuple[Fraction, bool] | None:
    """(number as written, percent-marked) or None.  1.0.2: backticks, a "~" spacer and a unit suffix of >= 2 letters
    ("6\\mathrm{~}", "180 days", "5cm", "24 percent") are dropped; a single trailing letter ("2x") is not a unit."""
    s = normalize_answer(value).replace(",", "").strip("`'\"")
    s = re.sub(r"^\((-?[\d.]+)\)/\((-?[\d.]+)\)$", r"\1/\2", s)
    s = s.replace("~", "")
    pct = False
    if s.endswith("%") or s.endswith("percent"):
        pct = True
        s = s[: -len("percent")] if s.endswith("percent") else s[:-1]
    m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)([a-z]{2,})?", s)
    if not m:
        return None
    try:
        if "/" in m.group(1):
            a, b = m.group(1).split("/")
            out = Fraction(a) / Fraction(b)
        else:
            out = Fraction(m.group(1))
    except (ValueError, ZeroDivisionError):
        return None
    return out, pct


def as_number(value: str) -> Fraction | None:
    parts = number_parts(value)
    if parts is None:
        return None
    return parts[0] / 100 if parts[1] else parts[0]


def choice_letter(value: str) -> str | None:
    m = re.search(r"\(\s*([A-Ea-e])\s*\)", value)
    return m.group(1).lower() if m else None


def answers_equal(left: str | None, right: str | None) -> bool | None:
    """True / False when decidable; None when the two forms cannot be compared by these rules (never counted as a
    pass).  Numbers: exact Fraction equality (Codex grade_arithmetic's tolerance 0), a percent-marked side also matching
    its bare number ("24 percent" == "24"); a bare choice letter matches a reference "(C) ..." label; else normalized
    string equality."""
    if left is None or right is None:
        return None
    pa, pb = number_parts(left), number_parts(right)
    if pa is not None and pb is not None:
        va = pa[0] / 100 if pa[1] else pa[0]
        vb = pb[0] / 100 if pb[1] else pb[0]
        return va == vb or ((pa[1] != pb[1]) and pa[0] == pb[0])
    na, nb = normalize_answer(left), normalize_answer(right)
    if not na or not nb:
        return None
    if na == nb:
        return True
    for x, y in ((left, nb), (right, na)):
        if re.fullmatch(r"[a-e]", y) and choice_letter(x) == y:
            return True
    if (pa is None) != (pb is None):
        return False
    return False if re.fullmatch(r"[a-e]", na) and re.fullmatch(r"[a-e]", nb) else None


#: 1.0.1: ',' joins the operand boundaries (a thousands separator "33,000/3530" or "= 1,260" is never one operand; the
#: first registry pass counted those as wrong), and a result followed by "%" / "percent" is compared as a percentage.
ARITH = re.compile(r"(?<![\w.,^/])(\d{1,7}(?:\.\d{1,4})?)\s*([+\-*/×÷])\s*(\d{1,7}(?:\.\d{1,4})?)\s*=\s*(-?\d{1,10}(?:\.\d{1,6})?)(?!\w|[.,]\d|[\^/]|\s*[+\-*/×÷]\s*\d)")
PERCENT_AFTER = re.compile(r"\s*(%|\\%|percent\b)")


def arithmetic_claims(text: str, limit: int = 200) -> tuple[int, int, list[str]]:
    """(checked, wrong, examples-with-context) for simple 'a op b = c' claims; chains ('2 + 3 * 4 = 14') are skipped."""
    checked = wrong = 0
    examples: list[str] = []
    for m in ARITH.finditer(text):
        before = text[: m.start()].rstrip()
        if before and before[-1] in "+-*/×÷^)":
            continue
        if re.search(r"(?:^|[\s\d)])[xX]$", before):   # 1.0.3: "13 x 3 + 3 = 42" is a chain with a letter-x product
            continue
        if re.search(r"\\(?:cdot|times|div)$", before):  # 1.0.4: LaTeX product / quotient chains
            continue
        after = text[m.end(): m.end() + 48].lower()
        if after.startswith(("...", "…")) or re.search(r"impossib|contradict|not possible|absurd|cannot|can't", after):
            continue                                     # 1.0.4: a truncated decimal, or a claim set up to be refuted
        a, op, b, c = Fraction(m.group(1)), m.group(2), Fraction(m.group(3)), Fraction(m.group(4))
        if op in "/÷" and b == 0:
            continue
        if op == "+":
            value = a + b
        elif op == "-":
            value = a - b
        elif op in "*×":
            value = a * b
        else:
            value = a / b
        decimals = len(m.group(4).split(".")[1]) if "." in m.group(4) else 0
        # 1.0.4: a rounded result may be off by one unit in its last place (4.1853 written 4.18): a slip, not wrong math
        tolerance = Fraction(1, 10 ** decimals) if (op in "/÷" or decimals) else Fraction(0)
        if PERCENT_AFTER.match(text, m.end()):
            value *= 100
        checked += 1
        if abs(value - c) > tolerance:
            wrong += 1
            if len(examples) < 3:
                examples.append(text[max(0, m.start() - 40): m.end() + 20].replace("\n", " "))
        if checked >= limit:
            break
    return checked, wrong, examples


def problem_in_instruction(instruction: str) -> bool:
    il = instruction.strip().lower()
    if il.startswith(TEACH) or any(il.startswith(g) for g in GEN + GENERIC_MATH):
        return False
    return len(instruction.strip()) >= 40


# ------------------------------------------------------------------------------------------- tool calls
XML_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
XML_FUNC = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
XML_PARAM = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)


def _raw_json_objects(text: str, limit: int = 20, max_scan: int = 200_000) -> list[Any]:
    """JSON objects/arrays decodable at '{' / '[' positions (outermost first)."""
    decoder = json.JSONDecoder()
    out: list[Any] = []
    i = 0
    text = text[:max_scan]
    while i < len(text) and len(out) < limit:
        j = min([p for p in (text.find("{", i), text.find("[", i)) if p >= 0], default=-1)
        if j < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, j)
            out.append(obj)
            i = end
        except ValueError:
            i = j + 1
    return out


def offered_functions(instruction: str) -> dict[str, dict[str, Any]]:
    """name -> parameters for function specs written as JSON in the instruction (empty when none)."""
    if '"name"' not in instruction or '"parameters"' not in instruction:
        return {}
    out: dict[str, dict[str, Any]] = {}

    def visit(node: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(node, dict):
            fn = node.get("function") if isinstance(node.get("function"), dict) else node
            if isinstance(fn.get("name"), str) and isinstance(fn.get("parameters"), dict):
                out[fn["name"]] = fn["parameters"]
                return
            for value in node.values():
                visit(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                visit(value, depth + 1)

    for obj in _raw_json_objects(instruction):
        visit(obj)
    return out


def parse_tool_calls(response: str) -> dict[str, Any] | None:
    """Tool calls in a response, or None when the response carries no call syntax."""
    calls: list[dict[str, Any]] = []
    errors = 0
    formats: set[str] = set()
    opens, closes = response.count("<tool_call>"), response.count("</tool_call>")
    if opens or closes:
        errors += abs(opens - closes)
        for body in XML_CALL.findall(response):
            funcs = XML_FUNC.findall(body)
            if funcs:
                formats.add("qwen_xml")
                for name, inner in funcs:
                    calls.append({"name": name.strip(), "arguments": {k.strip(): v.strip() for k, v in XML_PARAM.findall(inner)}})
                continue
            try:
                obj = json.loads(body.strip())
                formats.add("hermes_json")
                calls.append({"name": obj.get("name"), "arguments": obj.get("arguments", obj.get("parameters"))})
            except (ValueError, AttributeError):
                errors += 1
    elif '"tool_calls"' in response or '"function_call"' in response:
        formats.add("json_in_prose")
        found = False
        for obj in _raw_json_objects(response, limit=5):
            items = obj.get("tool_calls") if isinstance(obj, dict) else None
            if items is None and isinstance(obj, dict) and isinstance(obj.get("function_call"), dict):
                items = [obj["function_call"]]
            if not isinstance(items, list):
                continue
            found = True
            for item in items:
                fn = item.get("function", item) if isinstance(item, dict) else {}
                args = fn.get("arguments", fn.get("parameters"))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        errors += 1
                        args = None
                calls.append({"name": fn.get("name"), "arguments": args})
        if not found:
            errors += 1
    else:
        return None
    bad_args = sum(1 for c in calls if not isinstance(c.get("arguments"), dict))
    no_name = sum(1 for c in calls if not isinstance(c.get("name"), str) or not c.get("name"))
    return {"calls": len(calls), "parse_errors": errors, "args_not_object": bad_args, "no_name": no_name,
            "formats": sorted(formats), "names": [c.get("name") for c in calls][:12], "_calls": calls}


def check_offered(calls: list[dict[str, Any]], offered: dict[str, dict[str, Any]]) -> dict[str, int]:
    not_offered = missing = unknown = 0
    for call in calls:
        name = call.get("name")
        if name not in offered:
            not_offered += 1
            continue
        params = offered[name] or {}
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        props = params.get("properties") or {}
        missing += any(r not in args for r in params.get("required") or [])
        unknown += bool(props) and any(k not in props for k in args)
    return {"name_not_offered": not_offered, "missing_required": missing, "unknown_param": unknown}


# ------------------------------------------------------------------------------------------- per-row features
def repetition(text: str) -> dict[str, float]:
    b = text.encode("utf-8", "replace")
    ratio = len(zlib.compress(b, 6)) / max(1, len(b))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    dup_lines = 1.0 - len(set(lines)) / len(lines) if len(lines) >= 4 else 0.0
    toks = words(text)
    grams = collections.Counter(tuple(toks[i:i + 4]) for i in range(len(toks) - 3))
    top = grams.most_common(1)[0][1] / max(1, sum(grams.values())) if grams else 0.0
    return {"zlib": round(ratio, 4), "dup_line_frac": round(dup_lines, 4), "top_4gram_share": round(top, 4)}


def audit4_class(row: dict[str, Any], collapsed: bool) -> str:
    ins = row.get("instruction") or ""
    il = ins.lower().strip()
    resp = row.get("response") or ""
    ds = row.get("dataset_id") or ""
    if "bfcl" in ds:
        return "01"
    if il.startswith(TEACH):
        return "02"
    if HEX.search(resp):
        return "03"
    if il.startswith("execute the structured tool call"):
        return "04"
    if "arc_agi" in ds:
        return "05"
    if "<think>" in resp and "</think>" not in resp:
        return "06"
    if collapsed:
        return "07"
    if any(il.startswith(g) for g in GEN):
        return "08" if not PROSE_END.search(resp.rstrip()[-2:] or "x") else "09"
    if OMNI.search(resp) or OMNI.search(ins):
        return "10"
    if TEMPLATED_DS.search(ds):
        return "11"
    if "memory" in ds:
        return "12"
    if not resp.strip():
        return "13"
    if len(resp) > 200 and resp.rstrip()[-1:].isalpha() and row.get("route") not in ("code", "tool", "long_context"):
        return "14"
    return "15"


def newline_collapsed(row: dict[str, Any]) -> bool:
    resp = row.get("response") or ""
    il = (row.get("instruction") or "").lower().strip()
    return (row.get("route") in ("code", "text", "reasoning", "math", "tool") and len(resp) > 300 and resp.count("\n") <= 1
            and resp.lstrip()[:1] not in "{[" and bool(CODEISH.search(resp)) and not any(il.startswith(g) for g in GEN))


def template_key(instruction: str) -> str:
    return re.sub(r"\d+", "#", instruction.strip().lower())[:60]


def python_blocks_ok(row: dict[str, Any]) -> tuple[int, int]:
    resp = row.get("response") or ""
    blocks = PY_BLOCK.findall(resp)
    if not blocks and row.get("route") == "code" and resp.lstrip().startswith(("def ", "import ", "from ", "class ")):
        blocks = [resp]
    ok = bad = 0
    for block in blocks[:8]:
        try:
            ast.parse(block)
            ok += 1
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            bad += 1
    return ok, bad


def score_row(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Every row-local check.  ctx: cap_signature {dataset: max_chars}, template_counts {key: n}, screen (ReviewIndex or
    None), split / d2_kept lookups.  Dedup fields are filled later (they need every row)."""
    ins = row.get("instruction") or ""
    resp = row.get("response") or ""
    ds = row.get("dataset_id") or ""
    route = row.get("route") or ""
    rid = row.get("rid")
    flags: list[str] = []
    collapsed = newline_collapsed(row)
    cls = audit4_class(row, collapsed)

    # ---- screen
    screen_hits: list[dict[str, str]] = []
    if ctx.get("screen") is not None:
        screen_hits = ctx["screen"].hits(ins + "\n" + resp)
        if screen_hits:
            flags.append("screen_hit")
    split = ctx.get("split", {}).get(rid, row.get("split"))
    if split in ("eval", "eval_near_duplicate", "eval_media_shared"):
        flags.append(f"split_{split}")
    if rid in ctx.get("holdout_rids", ()):
        flags.append("bfcl62_holdout")

    # ---- structure
    st: dict[str, Any] = {}
    if not resp.strip():
        flags.append("empty_response")
    moj = len(MOJIBAKE.findall(ins + resp))
    st["mojibake"] = moj
    if moj >= 3:
        flags.append("mojibake")
    if HEX.search(resp):
        flags.append("hex_flattened_transcript")
    if SPLIT_TRAILER.search(resp):
        flags.append("split_label_trailer")
    if collapsed:
        flags.append("newline_collapsed_code")
    fences = len(FENCE.findall(resp))
    st["code_fences"] = fences
    if fences % 2:
        flags.append("unbalanced_code_fence")
    stripped = resp.strip()
    if stripped[:1] in "{[" and stripped[-1:] in "}]":
        try:
            json.loads(stripped)
            st["json_ok"] = True
        except ValueError:
            st["json_ok"] = False
            flags.append("json_invalid")
    py_ok, py_bad = python_blocks_ok(row)
    if py_ok or py_bad:
        st["python_blocks_ok"], st["python_blocks_bad"] = py_ok, py_bad
        if py_bad:
            flags.append("python_syntax_error")
    il = ins.lower().strip()
    if il.startswith(TEACH):
        flags.append("teacher_json_wrapper")
    if OMNI.search(resp) or OMNI.search(ins):
        flags.append("pipeline_selfreference")
    if any(il.startswith(g) for g in GEN):
        flags.append("generic_instruction")
        if WEBNAV.search(resp):
            flags.append("web_boilerplate")
        if ADULT.search(resp):
            flags.append("adult_spam_keywords")

    # ---- truncation
    tr: dict[str, Any] = {}
    cap = ctx.get("cap_signature", {}).get(ds)
    tr["cap_cut"] = bool(cap and len(resp) >= cap - max(30, cap // 100))
    if tr["cap_cut"]:
        flags.append("cap_cut")
    tr["unclosed_think"] = "<think>" in resp and "</think>" not in resp
    if tr["unclosed_think"]:
        flags.append("unclosed_think")
    end = resp.rstrip()[-2:] or "x"
    tr["mid_sentence"] = bool(resp.strip()) and not PROSE_END.search(end) and route not in ("code", "tool", "long_context") \
        and stripped[-1:] not in "}]"
    tr["mid_word"] = len(resp) > 200 and resp.rstrip()[-1:].isalpha() and route not in ("code", "tool", "long_context")
    if tr["mid_word"]:
        flags.append("ends_mid_word")
    elif tr["mid_sentence"] and len(resp) > 200:
        flags.append("ends_mid_sentence")

    # ---- repetition
    rep = repetition(resp) if resp else {"zlib": 1.0, "dup_line_frac": 0.0, "top_4gram_share": 0.0}
    rep["degenerate"] = (len(resp) > 400 and rep["zlib"] < 0.12 and "arc_agi" not in ds and stripped[:1] not in "{[")
    if rep["degenerate"]:
        flags.append("degenerate_repetition")

    # ---- template family
    tkey = template_key(ins)
    family_size = ctx.get("template_counts", {}).get(tkey, 0)
    if TEMPLATED_DS.search(ds) or family_size >= ctx.get("template_family_min", 50):
        flags.append("templated_family")

    # ---- tool calls
    tool = parse_tool_calls(resp)
    if tool is not None:
        offered = offered_functions(ins)
        calls = tool.pop("_calls")
        tool["offered_functions"] = len(offered)
        if offered:
            tool.update(check_offered(calls, offered))
        if tool["parse_errors"] or tool["args_not_object"] or tool["no_name"]:
            flags.append("tool_call_malformed")
        if tool.get("name_not_offered") or tool.get("missing_required"):
            flags.append("tool_call_schema_mismatch")
    if "bfcl" in ds:
        flags.append("bfcl_eval_overlap")

    # ---- verifiable
    ver: dict[str, Any] = {"kind": "none"}
    if route == "math" or "math" in ds or route == "reasoning":
        ref, rule = extract_reference(resp)
        has_problem = problem_in_instruction(ins)
        if ref is not None:
            ver = {"kind": "math_ref" if has_problem else "math_ref_no_problem", "reference": ref, "rule": rule}
        else:
            ver = {"kind": "math_noref" if has_problem else "math_noref_no_problem"}
        ver["problem_in_instruction"] = has_problem
    elif "arc_agi" in ds:
        ver = {"kind": "arc_grid"}
    elif route in ("code", "text", "tool") and ASSERT.search(resp + "\n" + ins) and not collapsed:
        ver = {"kind": "code_tests_present", "executed": False}
    checked, wrong, ex = arithmetic_claims(resp)
    if checked:
        ver["arith_checked"], ver["arith_wrong"] = checked, wrong
        if wrong:
            ver["arith_wrong_examples"] = ex
            flags.append("arithmetic_wrong")

    out = {
        "schema": SCORE_SCHEMA,
        "rid": rid,
        "dataset_id": ds,
        "route": route,
        "family": row.get("family"),
        "split": split,
        "d2_kept": int(ctx.get("d2_kept", {}).get(rid, 0)),
        "class": cls,
        "class_label": CLASS_LABELS[cls],
        "len": {"instr_chars": len(ins), "resp_chars": len(resp), "tok_response": _int(row.get("tok_response"))},
        "flags": flags,
        "struct": st,
        "trunc": tr,
        "rep": rep,
        "template": {"key_sha16": text_sha256(tkey)[:16], "family_size": family_size},
        "tool": tool,
        "verify": ver,
        "screen": screen_hits[:3],
    }
    return out


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------- score + advisory action
PENALTY = {
    "cap_cut": 0.4, "unclosed_think": 0.3, "ends_mid_word": 0.5, "ends_mid_sentence": 0.7, "hex_flattened_transcript": 0.3,
    "split_label_trailer": 0.7, "newline_collapsed_code": 0.5, "teacher_json_wrapper": 0.5, "pipeline_selfreference": 0.5,
    "templated_family": 0.7, "web_boilerplate": 0.8, "adult_spam_keywords": 0.5, "degenerate_repetition": 0.5,
    "tool_call_malformed": 0.5, "tool_call_schema_mismatch": 0.6, "json_invalid": 0.6, "unbalanced_code_fence": 0.8,
    "python_syntax_error": 0.7, "arithmetic_wrong": 0.6, "near_duplicate_member": 0.2, "exact_duplicate_member": 0.0,
}
HARD_REJECT = ("screen_hit", "empty_response", "mojibake", "bfcl62_holdout", "exact_duplicate_member")


def finish(record: dict[str, Any]) -> dict[str, Any]:
    """score = product of penalties (hard defects -> 0); action = the program doc's decision table, ADVISORY."""
    flags = set(record["flags"])
    score = 0.0 if flags & set(HARD_REJECT) else 1.0
    for flag in flags:
        score *= PENALTY.get(flag, 1.0)
    reasons: list[str] = []
    if flags & set(HARD_REJECT):
        action = "reject"
        reasons = sorted(flags & set(HARD_REJECT))
    elif "near_duplicate_member" in flags:
        action, reasons = "reject", ["near_duplicate_member"]
    elif "hex_flattened_transcript" in flags:
        action, reasons = "repair_deterministic", ["hex_flattened_transcript"]
    elif flags & {"cap_cut", "unclosed_think", "ends_mid_word"}:
        action, reasons = "repair_with_parent_or_trunc_mask", sorted(flags & {"cap_cut", "unclosed_think", "ends_mid_word"})
    elif flags & {"teacher_json_wrapper", "pipeline_selfreference"}:
        action, reasons = "exclude_stage_t_gt_mask_d2", sorted(flags & {"teacher_json_wrapper", "pipeline_selfreference"})
    elif record["class"] == "05":
        action, reasons = "exclude_kd_verified_lanes_only", ["arc_grid_answer_only"]
    elif flags & {"tool_call_malformed", "arithmetic_wrong", "degenerate_repetition", "python_syntax_error"}:
        action, reasons = "review", sorted(flags & {"tool_call_malformed", "arithmetic_wrong", "degenerate_repetition", "python_syntax_error"})
    else:
        action = "keep"
    if "templated_family" in flags and action == "keep":
        action, reasons = "keep_capped_per_family", ["templated_family"]
    record["score"] = round(score, 4)
    record["action"] = action
    record["action_reasons"] = reasons
    record["action_note"] = "advisory (program doc 2.2 decision table); L-DATA signs off promote/repair"
    return record


# ------------------------------------------------------------------------------------------- MinHash + clusters
N_PERM = 128
BANDS, ROWS_PER_BAND = 16, 8
SHINGLE = 5
NEAR_JACCARD = 0.8
_MERSENNE = (1 << 61) - 1


def _perm_params(seed: int = 20261015) -> tuple[Any, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    a = rng.integers(1, (1 << 31) - 1, size=N_PERM, dtype=np.uint64)
    b = rng.integers(0, (1 << 31) - 1, size=N_PERM, dtype=np.uint64)
    return a, b


_PRIME32 = 4_294_967_311  # smallest prime above 2**32


def minhash(text: str, params: tuple[Any, Any]) -> Any:
    """128 x uint32 signature over word 5-gram shingles (crc32-hashed); a text under 5 words is one shingle."""
    import numpy as np

    toks = words(text)
    if len(toks) >= SHINGLE:
        shingles = {zlib.crc32(" ".join(toks[i:i + SHINGLE]).encode()) for i in range(len(toks) - SHINGLE + 1)}
    else:
        shingles = {zlib.crc32(" ".join(toks).encode())}
    h = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
    a, b = params
    sig = np.empty(N_PERM, dtype=np.uint32)
    for start in range(0, len(h), 50_000):          # bounded temporaries on very long documents
        chunk = h[start:start + 50_000]
        vals = ((a[:, None] * chunk[None, :] + b[:, None]) % np.uint64(_PRIME32)).min(axis=1)
        sig = vals.astype(np.uint32) if start == 0 else np.minimum(sig, vals.astype(np.uint32))
    return sig


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def near_duplicate_clusters(sigs: Any, threshold: float = NEAR_JACCARD, big_bucket: int = 64) -> tuple[UnionFind, int]:
    """LSH 16 x 8 candidates, verified by the estimated Jaccard (fraction of equal signature slots).  Buckets larger than
    `big_bucket` are compared star-wise (every member vs the first) plus consecutive pairs, so a 5,000-row template
    family costs O(n), not O(n^2); a missed edge there only under-merges (never a false merge)."""
    import numpy as np

    n = len(sigs)
    uf = UnionFind(n)
    checked = 0
    for band in range(BANDS):
        lo = band * ROWS_PER_BAND
        buckets: dict[bytes, list[int]] = collections.defaultdict(list)
        block = np.ascontiguousarray(sigs[:, lo:lo + ROWS_PER_BAND])
        for i in range(n):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) <= big_bucket:
                pairs = ((members[x], members[y]) for x in range(len(members)) for y in range(x + 1, len(members)))
            else:
                pairs = [(members[0], m) for m in members[1:]] + list(zip(members[1:], members[2:]))
            for i, j in pairs:
                if uf.find(i) == uf.find(j):
                    continue
                checked += 1
                if float(np.mean(sigs[i] == sigs[j])) >= threshold:
                    uf.union(i, j)
    return uf, checked


def representative_key(rec: dict[str, Any]) -> tuple:
    hard = sum(1 for f in rec["flags"] if f in PENALTY or f in HARD_REJECT)
    closed = not rec["trunc"]["cap_cut"] and not rec["trunc"]["unclosed_think"] and not rec["trunc"]["mid_word"]
    return (0 if rec.get("split") in (None, "train") else 1, 0 if not rec.get("ledger") else -1, hard,
            0 if closed else 1, -rec["len"]["resp_chars"], str(rec["rid"]))


def apply_dedup(records: list[dict[str, Any]], texts: list[tuple[str, str]], sigs: Any,
                ledger_count: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fill dedup fields.  The first `ledger_count` entries of `sigs` belong to the ledger (already promoted rows); they
    are never written out but win every cluster they are in."""
    n_ledger = ledger_count
    exact: dict[str, int] = {}
    resp_only: dict[str, int] = {}
    exact_dupes = resp_dupes = 0
    for k, rec in enumerate(records):
        ins, resp = texts[k]
        key = text_sha256(ins.strip() + "\x00" + resp.strip())
        rec["dedup"] = {"exact_sha16": key[:16]}
        if key in exact:
            rec["flags"].append("exact_duplicate_member")
            rec["dedup"]["exact_of"] = records[exact[key]]["rid"] if exact[key] >= 0 else "ledger"
            exact_dupes += 1
        else:
            exact[key] = k
        if len(resp.strip()) >= 50:
            rkey = text_sha256(resp.strip())
            if rkey in resp_only:
                rec["dedup"]["response_dup_of"] = records[resp_only[rkey]]["rid"]
                resp_dupes += 1
            else:
                resp_only[rkey] = k
    uf, checked = near_duplicate_clusters(sigs)
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for idx in range(len(sigs)):
        groups[uf.find(idx)].append(idx)
    clusters = []
    near_members = 0
    mixed_split = 0
    for root, members in groups.items():
        if len(members) < 2:
            continue
        batch_members = [m - n_ledger for m in members if m >= n_ledger]
        if not batch_members:
            continue
        has_ledger = any(m < n_ledger for m in members)
        cid = f"nd{root:07d}"
        ordered = sorted(batch_members, key=lambda m: representative_key(records[m]))
        rep = None if has_ledger else ordered[0]
        splits = {records[m].get("split") for m in batch_members}
        if "train" in splits and splits & {"eval", "eval_near_duplicate"}:
            mixed_split += 1
        for m in batch_members:
            rec = records[m]
            rec["dedup"]["near_cluster"] = cid
            rec["dedup"]["cluster_size"] = len(members)
            rec["dedup"]["representative"] = m == rep
            if m != rep and "exact_duplicate_member" not in rec["flags"]:
                rec["flags"].append("near_duplicate_member")
                near_members += 1
        clusters.append({"cluster": cid, "size": len(members), "ledger_members": sum(1 for m in members if m < n_ledger),
                         "representative": records[rep]["rid"] if rep is not None else "ledger",
                         "rids": [records[m]["rid"] for m in ordered][:200], "splits": sorted(str(s) for s in splits)})
    stats = {"exact_duplicate_members": exact_dupes, "response_only_duplicates": resp_dupes, "near_clusters": len(clusters),
             "near_duplicate_members_non_representative": near_members, "candidate_pairs_checked": checked,
             "clusters_mixing_train_and_eval_splits": mixed_split,
             "rule": f"MinHash word-{SHINGLE}-gram, {N_PERM} perms, LSH {BANDS}x{ROWS_PER_BAND}, est. Jaccard >= {NEAR_JACCARD}"}
    return clusters, stats


# ------------------------------------------------------------------------------------------- screen
def load_by_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ScorerError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FractionIndex:
    """Near-duplicate containment for large held-out SPLITS (not sealed banks): a text is a hit when >= `threshold` of
    its word 13-grams that are not boilerplate occur in the split.  Boilerplate = a 13-gram present in >= `boiler_df`
    split items (shared system prompts, the teacher-JSON wrapper, generic headers).  1.0.0's first smoke used the bank
    rule (any shared 13-gram) on these splits and flagged 1,397 of 2,000 registry rows through boilerplate alone; the
    threshold is distill_corpus_build.NEAR_DUP_FRACTION (0.5), the rule carve.json's eval_near_duplicate used."""

    def __init__(self, n: int = 13, threshold: float = 0.5, boiler_df: int = 20, min_grams: int = 5) -> None:
        self.n, self.threshold, self.boiler_df, self.min_grams = n, threshold, boiler_df, min_grams
        self.df: collections.Counter = collections.Counter()
        self.owner: dict[int, tuple[str, str]] = {}
        self.items: collections.Counter = collections.Counter()

    def grams(self, text: str) -> set[int]:
        toks = words(text)
        return {hash(tuple(toks[i:i + self.n])) for i in range(len(toks) - self.n + 1)}

    def add_text(self, source: str, item_id: str, text: str) -> None:
        for g in self.grams(text):
            self.df[g] += 1
            self.owner.setdefault(g, (source, item_id))
        self.items[source] += 1

    def hits(self, text: str) -> list[dict[str, str]]:
        grams = [g for g in self.grams(text) if self.df.get(g, 0) < self.boiler_df]
        if len(grams) < self.min_grams:
            return []
        found = [g for g in grams if g in self.df]
        frac = len(found) / len(grams)
        if frac < self.threshold:
            return []
        owners = collections.Counter(self.owner[g] for g in found)
        (source, item), _ = owners.most_common(1)[0]
        return [{"rule": f"near{self.n}_frac{frac:.2f}", "source": source, "item": item}]


class Screen:
    """Sealed review banks: any shared 13-gram / exact / short item / literal (distill_corpus_build.ReviewIndex).
    Held-out splits: near-duplicate fraction (FractionIndex)."""

    def __init__(self, banks: Any, splits: FractionIndex) -> None:
        self.banks, self.splits = banks, splits

    def hits(self, text: str) -> list[dict[str, str]]:
        return self.banks.hits(text) + self.splits.hits(text)


def build_screen(corpus_tools: Path, eval_split: Path, toolcall_bank: Path, codex_splits: Iterable[Path]) -> tuple[Any, dict[str, Any]]:
    """distill_corpus_build's review index (+ the toolcall bank, a sealed bank) and the held-out splits the data audit
    listed as missing (corpus eval split, Codex corpus-schema-v2 validation / test) under the near-dup fraction rule."""
    sys.path.insert(0, str(corpus_tools))
    dcb = load_by_path("distill_corpus_build", corpus_tools / "distill_corpus_build.py")
    index, sources = dcb.build_review_index(dcb.parse_args(["screen"]))
    added = []
    bank = json.loads(toolcall_bank.read_text(encoding="utf-8"))
    strings: list[str] = []
    dcb.harvest_strings(bank, strings)
    for i, text in enumerate(strings):
        index.add_text("toolcall_bank_v1", f"tc#{i}", text)
    added.append({"name": "toolcall_bank_v1", "rule": "bank (any 13-gram)", "path": str(toolcall_bank),
                  "sha256": file_sha256(toolcall_bank), "items": len(strings)})
    splits = FractionIndex(n=dcb.NGRAM, threshold=float(getattr(dcb, "NEAR_DUP_FRACTION", 0.5)))
    n = 0
    for row in read_jsonl(eval_split):
        splits.add_text("corpus_eval_split", str(row.get("rid", n)), (row.get("instruction") or "") + "\n" + (row.get("response") or ""))
        n += 1
    added.append({"name": "corpus_eval_split", "rule": "near-dup fraction", "path": str(eval_split), "sha256": file_sha256(eval_split), "items": n})
    for path in codex_splits:
        k = 0
        for row in read_jsonl(path):
            text = "\n".join(str(m.get("content") or "") for m in row.get("messages") or [] if m.get("role") in ("user", "assistant"))
            splits.add_text(f"codex_{path.stem}", str(row.get("id", k)), text)
            k += 1
        added.append({"name": f"codex_corpus_schema_v2_{path.stem}", "rule": "near-dup fraction", "path": str(path),
                      "sha256": file_sha256(path), "items": k})
    return Screen(index, splits), {"review_sources": sources, "added_sources": added, "corpus_tool": dcb.tool_identity(),
                                   "index": index.summary(), "split_rule": {"ngram": splits.n, "threshold": splits.threshold,
                                                                            "boilerplate_df": splits.boiler_df, "distinct_grams": len(splits.df)}}


# ------------------------------------------------------------------------------------------- context builders
def cap_signatures(rows: list[dict[str, Any]]) -> dict[str, int]:
    """audit2's cap rule: dataset max >= 500 chars and >= max(20, 10 %) of its rows within max(30, 1 %) of the max."""
    lens: dict[str, list[int]] = collections.defaultdict(list)
    for r in rows:
        lens[r.get("dataset_id") or ""].append(len(r.get("response") or ""))
    out = {}
    for ds, values in lens.items():
        mx = max(values)
        if mx < 500:
            continue
        near = sum(1 for v in values if v >= mx - max(30, mx // 100))
        if near >= max(20, 0.1 * len(values)):
            out[ds] = mx
    return out


def build_context(rows: list[dict[str, Any]], *, split: dict[str, str] | None, d2_kept: dict[str, int] | None, screen: Any,
                  holdout_rids: Iterable[str] = (), template_family_min: int = 50) -> dict[str, Any]:
    return {"cap_signature": cap_signatures(rows),
            "template_counts": collections.Counter(template_key(r.get("instruction") or "") for r in rows),
            "template_family_min": template_family_min, "split": split or {}, "d2_kept": d2_kept or {},
            "screen": screen, "holdout_rids": set(holdout_rids)}


def score_all(rows: list[dict[str, Any]], ctx: dict[str, Any], ledger: dict[str, Any] | None = None,
              progress_every: int = 10_000) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Any]:
    """Score every row, then dedup (within the rows, and against `ledger` = {"sigs": uint32 [n, 128], "exact16": set of
    16-hex exact keys} when given: a ledger row wins every cluster and every exact match)."""
    import numpy as np

    params = _perm_params()
    records = []
    texts = []
    sigs = np.zeros((len(rows), N_PERM), dtype=np.uint32)
    t0 = time.time()
    for i, row in enumerate(rows):
        records.append(score_row(row, ctx))
        ins, resp = row.get("instruction") or "", row.get("response") or ""
        texts.append((ins, resp))
        sigs[i] = minhash(ins + "\n" + resp, params)
        if progress_every and (i + 1) % progress_every == 0:
            log(f"scored {i + 1:,}/{len(rows):,} rows ({time.time() - t0:.0f} s)")
    n_ledger = 0
    all_sigs = sigs
    if ledger is not None and len(ledger["sigs"]):
        n_ledger = len(ledger["sigs"])
        all_sigs = np.concatenate([ledger["sigs"].astype(np.uint32), sigs])
    clusters, dstats = apply_dedup(records, texts, all_sigs, ledger_count=n_ledger)
    if ledger is not None:
        hits = 0
        for rec in records:
            if rec["dedup"]["exact_sha16"] in ledger["exact16"] and "exact_duplicate_member" not in rec["flags"]:
                rec["flags"].append("exact_duplicate_member")
                rec["dedup"]["exact_of"] = "ledger"
                hits += 1
        dstats["ledger_rows"] = n_ledger
        dstats["ledger_exact_hits"] = hits
    for rec in records:
        finish(rec)
    return records, clusters, dstats, sigs


# ------------------------------------------------------------------------------------------- summary
def summarize(records: list[dict[str, Any]], dstats: dict[str, Any]) -> dict[str, Any]:
    def tally(selector) -> dict[str, Any]:
        rows = collections.Counter()
        d2 = collections.Counter()
        for r in records:
            for key in selector(r):
                rows[key] += 1
                d2[key] += r["d2_kept"]
        return {k: {"rows": rows[k], "d2_tokens": d2[k]} for k in sorted(rows)}

    d2_total = sum(r["d2_kept"] for r in records)
    in_d2 = [r for r in records if r["d2_kept"]]
    scores = sorted(r["score"] for r in records)
    deciles = [scores[min(len(scores) - 1, int(q * len(scores)))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)] if scores else []
    return {
        "rows": len(records), "rows_in_d2": len(in_d2), "d2_tokens": d2_total,
        "by_family": tally(lambda r: [str(r.get("family"))]),
        "by_split": tally(lambda r: [str(r.get("split"))]),
        "by_class": tally(lambda r: [f"{r['class']}_{r['class_label']}"]),
        "by_flag": tally(lambda r: r["flags"]),
        "by_action": tally(lambda r: [r["action"]]),
        "by_verify_kind": tally(lambda r: [r["verify"]["kind"]]),
        "verify_math_ref_by_split": dict(collections.Counter(str(r["split"]) for r in records if r["verify"]["kind"] == "math_ref")),
        "arithmetic": {"rows_with_claims": sum(1 for r in records if r["verify"].get("arith_checked")),
                       "claims_checked": sum(r["verify"].get("arith_checked", 0) for r in records),
                       "claims_wrong": sum(r["verify"].get("arith_wrong", 0) for r in records),
                       "rows_with_wrong": sum(1 for r in records if r["verify"].get("arith_wrong"))},
        "tool_rows": {"with_calls": sum(1 for r in records if r["tool"]), "calls": sum((r["tool"] or {}).get("calls", 0) for r in records),
                      "malformed_rows": sum(1 for r in records if "tool_call_malformed" in r["flags"]),
                      "schema_mismatch_rows": sum(1 for r in records if "tool_call_schema_mismatch" in r["flags"]),
                      "rows_with_offered_specs": sum(1 for r in records if (r["tool"] or {}).get("offered_functions"))},
        "screen_hits": {"rows": sum(1 for r in records if r["screen"]),
                        "by_source": dict(collections.Counter(h["source"] for r in records for h in r["screen"][:1]))},
        "score_quantiles_p10_p25_p50_p75_p90": deciles,
        "d2_tokens_by_action": {a: v["d2_tokens"] for a, v in tally(lambda r: [r["action"]]).items()},
        "d2_token_share_not_keep": round(sum(r["d2_kept"] for r in in_d2 if not r["action"].startswith("keep")) / d2_total, 4) if d2_total else None,
        "dedup": dstats,
    }


# ------------------------------------------------------------------------------------------- D2 EOS check
def d2_eos_check(docs_path: Path, text_manifest: Path, by_rid: dict[str, dict[str, Any]], parent_dir: Path,
                 sample_limit: int = 12) -> dict[str, Any]:
    """For D2 documents whose row is cut (class 06 / 08 / 14): the last 4 packed ids of the document."""
    from safetensors import safe_open

    manifest = json.loads(text_manifest.read_text(encoding="utf-8"))
    seq = int(manifest["sequence_length"])
    shards = [(s["path"], s["rows"][0], s["rows"][1]) for s in manifest["shards"]]
    handles: dict[str, Any] = {}
    tok = None
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(parent_dir / "tokenizer.json"))
    except Exception:  # noqa: BLE001 - decode is a convenience; ids are the evidence
        tok = None

    def ids_at(pos_lo: int, pos_hi: int) -> list[int]:
        out = []
        for pos in range(pos_lo, pos_hi):
            row, col = divmod(pos, seq)
            for path, lo, hi in shards:
                if lo <= row < hi:
                    if path not in handles:
                        handles[path] = safe_open(path, framework="np")
                    out.append(int(handles[path].get_slice("input_ids")[row - lo: row - lo + 1, col: col + 1][0, 0]))
                    break
        return out

    tails = collections.Counter()
    by_class = collections.defaultdict(collections.Counter)
    samples = []
    n = 0
    for d in read_jsonl(docs_path):
        rec = by_rid.get(d["rid"])
        if rec is None or rec["class"] not in ("06", "08", "14"):
            continue
        start, end, kept, length = int(d["start"]), int(d["end"]), int(d["kept"]), int(d["length"])
        if kept != length:      # a document cut by the stream end: its tail is not in the set
            continue
        ids = ids_at(max(start, end - 4), end)
        tail = ("im_end+sep" if ids[-2:] == [IM_END_ID, SEPARATOR_ID] else
                "im_end+nl+sep" if len(ids) >= 3 and ids[-3] == IM_END_ID and ids[-1] == SEPARATOR_ID else
                "sep_only" if ids[-1:] == [SEPARATOR_ID] else "other")
        tails[tail] += 1
        by_class[rec["class"]][tail] += 1
        n += 1
        if len(samples) < sample_limit:
            samples.append({"doc": d["doc"], "rid": d["rid"], "class": rec["class"], "last_ids": ids,
                            "decoded": tok.decode(ids, skip_special_tokens=False) if tok else None})
    return {"cut_documents_checked": n, "tails": dict(tails), "tails_by_class": {k: dict(v) for k, v in by_class.items()},
            "samples": samples,
            "reading": "im_end+(nl)+sep means the packed target after a cut row is <|im_end|> then the separator: D2 "
                       "supervises an end-of-turn right after the cut unless a mask (A2 gt_mask / 4.1 A trunc_mask) removes it"}


# ------------------------------------------------------------------------------------------- commands
def write_outputs(out_dir: Path, records: list[dict[str, Any]], clusters: list[dict[str, Any]], sigs: Any,
                  summary: dict[str, Any], receipt: dict[str, Any]) -> None:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ".scores.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, out_dir / "scores.jsonl")
    with open(out_dir / "clusters.jsonl", "w", encoding="utf-8") as handle:
        for c in sorted(clusters, key=lambda c: -c["size"]):
            handle.write(json.dumps(c, ensure_ascii=False) + "\n")
    exact = np.array([r["dedup"]["exact_sha16"] for r in records], dtype="U16")
    np.savez_compressed(out_dir / "minhash.npz", sigs=sigs, rids=np.array([str(r["rid"]) for r in records]), exact16=exact)
    write_json_atomic(out_dir / "summary.json", summary)
    receipt["outputs"] = {name: {"path": str(out_dir / name), "sha256": file_sha256(out_dir / name)}
                          for name in ("scores.jsonl", "clusters.jsonl", "minhash.npz", "summary.json")}
    receipt["receipt_sha256"] = canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    write_json_atomic(out_dir / "receipt.json", receipt)


def cmd_registry(args: argparse.Namespace) -> int:
    t0 = time.time()
    rows_path, carve_path, docs_path = Path(args.rows), Path(args.carve), Path(args.documents)
    for p in (rows_path, carve_path, docs_path):
        if not p.is_file():
            raise ScorerError(f"input {p} is missing")
    out_dir = Path(args.out_dir)
    log(f"{TOOL_NAME} {TOOL_VERSION} registry -> {out_dir}")
    rows = list(read_jsonl(rows_path))
    if args.limit:
        rows = rows[: int(args.limit)]
    carve = json.loads(carve_path.read_text(encoding="utf-8"))["assignment"]
    d2_kept: dict[str, int] = collections.Counter()
    for d in read_jsonl(docs_path):
        d2_kept[d["rid"]] += int(d["kept"])
    holdout = []
    if Path(args.bfcl_holdout).is_file():
        payload = json.loads(Path(args.bfcl_holdout).read_text(encoding="utf-8"))
        holdout = payload.get("rids", payload) if isinstance(payload, dict) else payload
        holdout = [h if isinstance(h, str) else h.get("rid") for h in holdout]
    screen = None
    screen_info: dict[str, Any] = {"enabled": False}
    if not args.no_screen:
        log("building the review-surface screen")
        screen, screen_info = build_screen(Path(args.corpus_tools), Path(args.eval_split), Path(args.toolcall_bank),
                                           [Path(p) for p in args.codex_split])
        screen_info["enabled"] = True
    log(f"{len(rows):,} rows; scoring")
    ctx = build_context(rows, split=carve, d2_kept=d2_kept, screen=screen, holdout_rids=holdout)
    records, clusters, dstats, sigs = score_all(rows, ctx)
    summary = summarize(records, dstats)
    summary["cap_signature_datasets"] = len(ctx["cap_signature"])
    eos = None
    if not args.no_eos_check:
        log("D2 EOS check on cut documents")
        eos = d2_eos_check(docs_path, Path(args.text_manifest), {r["rid"]: r for r in records}, Path(args.parent_dir))
        write_json_atomic(out_dir / "d2_eos_check.json", eos)
        summary["d2_eos_check"] = {k: v for k, v in eos.items() if k != "samples"}
    audit4 = Path(args.audit4)
    if audit4.is_file():
        ref = json.loads(audit4.read_text(encoding="utf-8"))["disjoint_classes"]
        mine = {}
        for r in records:
            if r["d2_kept"]:
                mine.setdefault(r["class"], [0, 0])
                mine[r["class"]][0] += 1
                mine[r["class"]][1] += r["d2_kept"]
        summary["audit4_reconciliation"] = {
            key.split("_")[0]: {"audit4_d2_rows": v["d2_rows"], "audit4_d2_tok": v["d2_tok"],
                                "here_d2_rows": mine.get(key.split("_")[0], [0, 0])[0],
                                "here_d2_tok": mine.get(key.split("_")[0], [0, 0])[1]} for key, v in ref.items()}
    summary["wall_s"] = round(time.time() - t0, 1)
    receipt = {"schema": "omni-research.quality-run.v1", "mode": "registry", "at_utc": utc_now(), "tool": tool_identity(),
               "inputs": {"rows": {"path": str(rows_path), "sha256": file_sha256(rows_path), "rows": len(rows)},
                          "carve": {"path": str(carve_path), "sha256": file_sha256(carve_path)},
                          "documents": {"path": str(docs_path), "sha256": file_sha256(docs_path)}},
               "screen": screen_info, "limit": args.limit, "wall_s": summary["wall_s"],
               "sealed_inputs_untouched": "read only; every output is under --out-dir"}
    write_outputs(out_dir, records, clusters, sigs, summary, receipt)
    log(f"done: {len(records):,} rows, {summary['rows_in_d2']:,} in D2; actions {dict((k, v['rows']) for k, v in summary['by_action'].items())}; "
        f"D2 token share not 'keep*' {summary['d2_token_share_not_keep']}; {out_dir}")
    return 0


def batch_rows(batch_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path, rows_path = batch_dir / "manifest.json", batch_dir / "rows.jsonl"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise ScorerError(f"{batch_dir} is not an intake batch (needs manifest.json + rows.jsonl)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INTAKE_SCHEMA:
        raise ScorerError(f"{manifest_path} schema {manifest.get('schema')!r}, expected {INTAKE_SCHEMA}")
    sha = file_sha256(rows_path)
    if manifest.get("rows_sha256") != sha:
        raise ScorerError(f"{rows_path} hashes {sha}; the manifest records {manifest.get('rows_sha256')}")
    rows = []
    for row in read_jsonl(rows_path):
        if "instruction" not in row and isinstance(row.get("messages"), list):
            msgs = row["messages"]
            row = dict(row)
            row["instruction"] = "\n\n".join(str(m.get("content") or "") for m in msgs if m.get("role") in ("system", "user"))
            row["response"] = "\n\n".join(str(m.get("content") or "") for m in msgs if m.get("role") == "assistant")
        row.setdefault("family", "text")
        rows.append(row)
    if manifest.get("rows") not in (None, len(rows)):
        raise ScorerError(f"{rows_path} has {len(rows)} rows; the manifest says {manifest['rows']}")
    return manifest, rows


def cmd_batch(args: argparse.Namespace) -> int:
    import numpy as np

    t0 = time.time()
    batch_dir = Path(args.batch_dir)
    manifest, rows = batch_rows(batch_dir)
    out_dir = Path(args.out_dir) if args.out_dir else batch_dir / "quality" / TOOL_VERSION
    screen = None
    screen_info: dict[str, Any] = {"enabled": False}
    if not args.no_screen:
        screen, screen_info = build_screen(Path(args.corpus_tools), Path(args.eval_split), Path(args.toolcall_bank),
                                           [Path(p) for p in args.codex_split])
        screen_info["enabled"] = True
    ledger = None
    if args.ledger:
        led = np.load(Path(args.ledger) / "minhash.npz")
        ledger = {"sigs": led["sigs"], "exact16": {str(x) for x in led["exact16"]}, "path": str(args.ledger),
                  "sha256": file_sha256(Path(args.ledger) / "minhash.npz")}
    ctx = build_context(rows, split={}, d2_kept={}, screen=screen)
    records, clusters, dstats, sigs = score_all(rows, ctx, ledger=ledger, progress_every=0)
    summary = summarize(records, dstats)
    summary["wall_s"] = round(time.time() - t0, 1)
    receipt = {"schema": "omni-research.quality-run.v1", "mode": "batch", "at_utc": utc_now(), "tool": tool_identity(),
               "batch": {"dir": str(batch_dir), "manifest_sha256": file_sha256(batch_dir / "manifest.json"),
                         "batch_id": manifest.get("batch_id"), "rows_sha256": manifest.get("rows_sha256"), "rows": len(rows)},
               "ledger": {k: v for k, v in (ledger or {}).items() if k in ("path", "sha256")} or None,
               "screen": screen_info, "wall_s": summary["wall_s"]}
    write_outputs(out_dir, records, clusters, sigs, summary, receipt)
    log(f"batch {manifest.get('batch_id')}: {len(records)} rows; actions {dict((k, v['rows']) for k, v in summary['by_action'].items())}; {out_dir}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    print(json.dumps(json.loads((Path(args.out_dir) / "summary.json").read_text(encoding="utf-8")), indent=1)[:20000])
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    def screen_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--no-screen", action="store_true", help="fixtures only: skip the review-surface screen")
        sp.add_argument("--corpus-tools", default=str(DEFAULT_CORPUS_TOOLS))
        sp.add_argument("--eval-split", default=str(DEFAULT_EVAL_SPLIT))
        sp.add_argument("--toolcall-bank", default=str(DEFAULT_TOOLCALL_BANK))
        sp.add_argument("--codex-split", action="append", default=[str(p) for p in DEFAULT_CODEX_SPLITS])

    sp = sub.add_parser("registry")
    sp.add_argument("--rows", default=str(DEFAULT_ROWS))
    sp.add_argument("--carve", default=str(DEFAULT_CARVE))
    sp.add_argument("--documents", default=str(DEFAULT_DOCS))
    sp.add_argument("--text-manifest", default=str(DEFAULT_TEXT_MANIFEST))
    sp.add_argument("--bfcl-holdout", default=str(DEFAULT_BFCL_HOLDOUT))
    sp.add_argument("--parent-dir", default="/home/cereal/models/Qwen3.8-27B-Uncensored")
    sp.add_argument("--audit4", default="/mnt/lexar/omni-campaign/research/data-audit/full/audit4_report.json")
    sp.add_argument("--no-eos-check", action="store_true")
    sp.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke)")
    sp.add_argument("--out-dir", required=True)
    screen_args(sp)

    sp = sub.add_parser("batch")
    sp.add_argument("--batch-dir", required=True)
    sp.add_argument("--ledger", default=None, help="a previous run dir (minhash.npz + scores.jsonl) to dedup against")
    sp.add_argument("--out-dir", default=None)
    screen_args(sp)

    sp = sub.add_parser("summary")
    sp.add_argument("--out-dir", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "registry":
            return cmd_registry(args)
        if args.command == "batch":
            return cmd_batch(args)
        if args.command == "summary":
            return cmd_summary(args)
    except ScorerError as exc:
        log(f"REFUSED: {exc}")
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
