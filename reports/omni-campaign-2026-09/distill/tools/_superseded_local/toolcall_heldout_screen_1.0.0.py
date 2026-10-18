#!/usr/bin/env python3
"""toolcall_heldout_screen.py -- the D2-X1 13-gram held-out screen of the tool-call export (C1's corpus).

WHY THIS FILE EXISTS.  D2_DECISIONS_2026-10-15 (sha c8d47f22..., rule D2-X1, per_input toolcall_selfgen) and
prereg A2 item 3(d): the tool-call self-gen rows are synthetic (plan tc-*), so no id join can exclude Codex's
held-out rows from them; instead every accepted row is screened (13-gram) against the rendered text of the 5,391
held-out rows, the 62 BFCL tasks and Codex's 274-task set, a hit is DROPPED and counted, and the screen receipt
must name the C1 corpus with 0 hits.  d2_plan 1.3.0 (screen_state) refuses D2's data config until
$OC/distill/toolcall/rescore/heldout_screen.json (schema omni-d2x1-toolcall-screen-v1: decisions_sha256,
corpus_sha256 = the C1 corpus file, ngram 13, hits_in_corpus 0) exists.  toolcall_selfgen's own finalize screen
covers the campaign review surface, the corpus eval split and the bank strings, NOT the D2-X1 sets (review of
lane refill, 2026-10-15, finding 1), and the refill run 2 is exported together with run 1 into one corpus.

WHAT IT SCREENS.  Every row of the export (run 1 + run 2; the sidecar's source_manifests, each re-read through
toolcall_selfgen.sealed_conversations and bound to its corpus row by length and supervised count) against:
  * the campaign review surface, exactly as finalize builds it (toolcall_selfgen.build_screen ->
    distill_corpus_build.build_review_index: bank v1, bank v2 + its materialised haystacks, S8 media prompts (the
    bank v2 media cases), probe / smoke packs, R-Q PROMPT_PACK, the 64-prompt probe pack, needle sets and codes,
    export parity bank, extension bank v2, paraphrase triplets) + the corpus eval split + the toolcall bank's
    message strings, with ReviewIndex's rules (exact, 13-gram, short item, literal);
  * the D2-X1 held-out sets, 13-gram (verbatim-verified):
      codex_heldout   the six held-out packs of the decisions file's canonical_split (native-text-v3/v4/v7
                      validation + test; manifest and corpus shas pinned there), every row decoded with the
                      parent's pinned tokenizer (rendered text): 5,391 distinct row ids or refused.  Fidelity: the
                      v3 + v4 ids must equal corpus-schema-v2's validation + test ids, and each schema-v2 row's
                      longest message must be caught by the decoded index (the decode is the text, not noise);
      bfcl62          the 62 BFCL held-out source records + the 10 twin records (bfcl62_rids.json twin_exposure:
                      registry text, K3, self-gen and native-train twins), messages and tools, from Codex's
                      normalized corpus (sha pinned by the decisions file);
      codex_274       Codex's sealed 274-task set (combined_oct13_v1 tasks.jsonl, sha pinned): input + oracle;
      toolcall_bank   omni_toolcall_bank_v1 whole cases (messages, tool descriptions, expected calls) and its
                      held-out function names (literal containment in any screened text or tool / call name).
A row with any hit is dropped from the export BEFORE C1 rescores it, never silently: the dropped list
(heldout_screen.dropped.json: row, id, category, run, tokens, reasons with the matched 13-gram) is always
written, the unscreened export is copied to --backup-root first, the corpus and its sidecar are rewritten in place
(the sidecar keeps every toolcall_selfgen field, re-counted, and records the drop), and the receipt is written
last, naming the corpus sha C1 will rescore.  Screening is liveness-checked (every source must fire on planted
items, verbatim and re-wrapped; 12-word fragments and clean filler must not).

FAILURES ARE FUNCTIONAL, NUMBERS ARE NOT.  A pinned input with another sha, a decode that does not reproduce the
schema-v2 text, a sidecar that does not name the corpus, a row not bound to its sealed conversation, a dead screen,
or a drop above --max-drop-fraction (a screen that matches most of a synthetic corpus is matching boilerplate,
not held-out text: the lead reads it) -> exit 2, nothing changed.  Not enough root space or a busy screen lock ->
exit 75 (post_d1_queue defers).  The number of rows dropped is a diagnostic.

  screen   screen the export, drop hits, write dropped list + receipt (idempotent: a valid receipt for the current
           corpus sha is kept unless --force)
  check    exit 0 when the receipt is valid for the corpus as it is now (d2_plan screen_state's conditions + the
           seal + the sidecar names the corpus), else 10

    PY=/home/cereal/model-surgery-lab/venv/bin/python; T=$OC/distill/tools
    CUDA_VISIBLE_DEVICES= PYTHONPATH=$T nice -n 10 $PY $T/toolcall_heldout_screen.py screen \\
        --corpus $OC/distill/toolcall/rescore/toolcall-q38-2026-10-14.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import distill_corpus_build as dcb  # noqa: E402
import toolcall_selfgen as tc  # noqa: E402

try:
    import fcntl
except ImportError:  # local Windows test runs only; the server always locks
    fcntl = None

TOOL_NAME = "toolcall_heldout_screen"
TOOL_VERSION = "1.0.0"  # 2026-10-15: first version (lane toolscreen, Claude lead)
SCHEMA = "omni-d2x1-toolcall-screen-v1"
DROPPED_SCHEMA = "omni-d2x1-toolcall-screen-dropped-v1"
NGRAM = 13

OC = tc.OC
DEFAULT_DECISIONS = OC / "distill" / "queues" / "D2_DECISIONS_2026-10-15.json"
DEFAULT_CORPUS = OC / "distill" / "toolcall" / "rescore" / "toolcall-q38-2026-10-14.safetensors"
DEFAULT_EVAL_TASKS = Path("/mnt/lexar/omni-campaign/distill/d2/inputs/codex/combined_oct13_v1.tasks.jsonl")
DEFAULT_BACKUP_ROOT = Path("/mnt/lexar/omni-campaign/distill/toolcall/unscreened")

#: the pinned inputs (d2_plan.DECISIONS_SHA; bfcl62_holdout PINS eval_tasks; d2_bfcl_native_copy source eval_tasks_sha256)
DECISIONS_SHA = "c8d47f22784d265a2e9239f9274c062298d25278497ba21d01102aa45274d90d"
EVAL_TASKS_SHA = "3795f333998627bb3b2d6fd460c19228469b2286284dfaadf30c5e12d23713f8"
EVAL_TASKS_ROWS = 274
BFCL_TASK_PREFIX = "bfcl-heldout:"
BFCL_TASKS = 62
MAX_DROP_FRACTION = 0.05
ROOT_FLOOR_BYTES = 60 * 10**9  # the campaign's root floor (60 GB)
LIVENESS_PROBES = 8
FIDELITY_MIN_WORDS = NGRAM


class Refusal(SystemExit):
    """A functional failure: exit 2, nothing past this point is written."""

    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


class Deferral(SystemExit):
    """Not now (root space, a busy lock): exit 75, nothing written; the queue defers."""

    def __init__(self, message: str) -> None:
        super().__init__(75)
        self.message = message


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def tool_identity() -> dict[str, Any]:
    me = Path(__file__).resolve()
    return {"tool": TOOL_NAME, "version": TOOL_VERSION, "path": str(me), "sha256": tc.file_sha256(me)}


def pinned(path: Path, want: str, what: str) -> str:
    path = Path(path)
    if not path.is_file():
        raise Refusal(f"{what} missing at {path}")
    got = tc.file_sha256(path)
    if got != want:
        raise Refusal(f"{what} {path} hashes {got}, pinned {want}")
    return got


# --------------------------------------------------------------------------
# the D2-X1 index: 13-grams (verified verbatim on every hit) + literal names
# --------------------------------------------------------------------------


class HeldoutIndex:
    """distill_corpus_build's word rule (NFKC, lower case, letter / digit runs, every CJK character a word) and its
    13-gram rule, keyed by the in-process tuple hash to keep millions of grams small; a key hit counts only after
    the 13 words are found verbatim in the held-out item, so a hash collision can never drop a row."""

    def __init__(self, n: int = NGRAM) -> None:
        self.n = int(n)
        self.grams: dict[int, int] = {}
        self.items: list[tuple[str, str, str]] = []
        self.literals: dict[str, int] = {}
        self.by_source: Counter = Counter()
        self.below_n: Counter = Counter()
        self._words: dict[int, str] = {}

    def add_text(self, source: str, item_id: str, text: str) -> None:
        toks = dcb.words(text)
        index = len(self.items)
        self.items.append((source, str(item_id), text))
        self.by_source[source] += 1
        if len(toks) < self.n:
            self.below_n[source] += 1
            return
        grams = self.grams
        for i in range(len(toks) - self.n + 1):
            grams.setdefault(hash(tuple(toks[i : i + self.n])), index)

    def add_literal(self, source: str, item_id: str, literal: str) -> None:
        index = len(self.items)
        self.items.append((source, str(item_id), literal))
        self.by_source[source] += 1
        self.literals.setdefault(literal.lower(), index)

    def _verbatim(self, index: int, gram: tuple[str, ...]) -> bool:
        joined = self._words.get(index)
        if joined is None:
            joined = " " + " ".join(dcb.words(self.items[index][2])) + " "
            self._words[index] = joined
        return (" " + " ".join(gram) + " ") in joined

    def hits(self, text: str) -> list[dict[str, str]]:
        """The first verified 13-gram hit and the first literal hit of ``text``."""

        found: list[dict[str, str]] = []
        toks = dcb.words(text)
        n, grams = self.n, self.grams
        for i in range(len(toks) - n + 1):
            gram = tuple(toks[i : i + n])
            index = grams.get(hash(gram))
            if index is not None and self._verbatim(index, gram):
                source, item, _ = self.items[index]
                found.append({"rule": f"ngram{n}", "source": source, "item": item, "gram": " ".join(gram)})
                break
        if self.literals:
            lowered = text.lower()
            for literal, index in self.literals.items():
                if literal in lowered:
                    source, item, _ = self.items[index]
                    found.append({"rule": "literal", "source": source, "item": item, "gram": literal})
                    break
        return found

    def summary(self) -> dict[str, Any]:
        return {"ngram": self.n, "ngram_keys": len(self.grams), "literal_names": len(self.literals),
                "items_by_source": dict(self.by_source), "items_below_ngram_by_source": dict(self.below_n),
                "rule": f"any word {self.n}-gram of a screened text found verbatim in a held-out item; held-out function names by literal containment"}


def liveness(index: HeldoutIndex, probes: int = LIVENESS_PROBES) -> dict[str, Any]:
    """Prove the D2-X1 index fires on REAL held-out items of every source (13 words cut from the item, inside
    filler; and re-wrapped upper case one word per line), on every literal name, and stays quiet on a 12-word
    fragment between sentinels and on clean filler (distill_corpus_build.screen_liveness, same probes)."""

    by_source: dict[str, list[tuple[str, str]]] = {}
    for source, item, text in index.items:
        if text.lower() in index.literals:
            continue
        if len(dcb.words(text)) >= index.n:
            by_source.setdefault(source, []).append((item, text))
    rows, totals = [], Counter()
    for source in sorted(index.by_source):
        picked = sorted(by_source.get(source, []), key=lambda it: (-len(it[1]), it[0]))[:probes]
        row = Counter()
        for _, text in picked:
            toks = dcb.words(text)
            middle = max(0, (len(toks) - index.n) // 2)
            cut = toks[middle : middle + index.n]
            row["probes"] += 1
            if index.hits("The following passage continues a longer maintenance document.\n" + " ".join(cut) + "\nThe document then moves on."):
                row["caught"] += 1
            # upper case only where it round-trips through the word rule (German sharp s -> SS would not)
            wrapped = [w.upper() if dcb.words(w.upper()) == [w] else w for w in cut]
            if index.hits("Exported log:\n" + "\n".join(wrapped) + "\nEnd."):
                row["rewrapped_caught"] += 1
            fragment = " ".join(cut[: index.n - 1])
            row["fragment_controls"] += 1
            if index.hits(f"Entry note: the technician wrote qqzxv {fragment} vxzqq and closed it."):
                row["fragment_falsely_caught"] += 1
        literal_items = [(item, text) for s, item, text in index.items if s == source and text.lower() in index.literals]
        for _, literal in literal_items:
            row["literal_probes"] += 1
            if index.hits(f"maintenance ticket reference {literal} closed"):
                row["literal_caught"] += 1
        totals.update(row)
        rows.append({"source": source, **dict(row)})
    clean = 32
    clean_caught = sum(1 for i in range(clean) if index.hits(
        f"Batch {i}: the pump was serviced on the {i % 28 + 1}th and the reading settled at {i * 3 % 97} units after the valve was reseated."))
    empty = [r["source"] for r in rows if not r.get("probes") and not r.get("literal_probes")]
    live = (totals["probes"] > 0 and totals["caught"] == totals["probes"] and totals["rewrapped_caught"] == totals["probes"]
            and totals["fragment_falsely_caught"] == 0 and totals["literal_caught"] == totals["literal_probes"]
            and clean_caught == 0 and not empty)
    return {"rule": "every source fires on 13 words cut from its own items (verbatim and re-wrapped) and on every literal name; "
                    "no 12-word fragment and no clean filler is caught; no source without a probe",
            "totals": dict(totals), "clean_controls": clean, "clean_controls_falsely_caught": clean_caught,
            "sources_without_probe": empty, "by_source": rows, "screen_is_live": live}


# --------------------------------------------------------------------------
# held-out sources (every input pinned by the decisions file or a constant here)
# --------------------------------------------------------------------------


def load_decisions(path: Path) -> dict[str, Any]:
    pinned(path, DECISIONS_SHA, "decisions file")
    return tc.read_json(path)


def pack_decoder(parent_dir: Path) -> Callable[[list[list[int]]], list[str]]:
    """The parent's pinned tokenizer.json (toolcall_selfgen.QWEN_TOKENIZER_SHA256), special tokens kept: the rendered
    text of a packed row.  Ids outside the tokenizer are refused."""

    from tokenizers import Tokenizer

    pinned(Path(parent_dir) / "tokenizer.json", tc.QWEN_TOKENIZER_SHA256, "parent tokenizer")
    tok = Tokenizer.from_file(str(Path(parent_dir) / "tokenizer.json"))
    vocab = tok.get_vocab_size(with_added_tokens=True)

    def decode(rows: list[list[int]]) -> list[str]:
        for row in rows:
            if row and (max(row) >= vocab or min(row) < 0):
                raise Refusal(f"a held-out row holds id {max(row)} outside the tokenizer's {vocab}")
        return tok.decode_batch(rows, skip_special_tokens=False)

    return decode


def add_codex_heldout(index: HeldoutIndex, decisions: dict[str, Any],
                      decode: Callable[[list[list[int]]], list[str]]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """The six canonical held-out packs, every row decoded; 5,391 distinct ids and the schema-v2 fidelity check."""

    import numpy as np
    from safetensors.numpy import load_file

    split = decisions["canonical_split"]
    want_total = int(split["cross_checks"]["heldout_ids_total"])
    packs, ids_by_pack = [], {}
    for pack in split["heldout_packs"]:
        manifest_path = Path(pack["manifest"])
        pinned(manifest_path, pack["manifest_sha256"], f"held-out pack manifest {pack['name']}")
        manifest = tc.read_json(manifest_path)
        receipts = manifest.get("row_receipts") or []
        if manifest.get("sha256") != pack["corpus_sha256"] or int(manifest.get("rows", -1)) != int(pack["rows"]) or len(receipts) != int(pack["rows"]):
            raise Refusal(f"held-out pack {pack['name']}: manifest sha / rows / row_receipts disagree with the decisions file")
        shard = Path(manifest["shard"])
        pinned(shard, pack["corpus_sha256"], f"held-out pack shard {pack['name']}")
        tensors = load_file(str(shard))
        ids, lengths = tensors["input_ids"], tensors["lengths"].astype(np.int64).reshape(-1)
        if ids.shape[0] != len(receipts) or lengths.shape[0] != len(receipts):
            raise Refusal(f"held-out pack {pack['name']}: shard rows {ids.shape[0]} != {len(receipts)} row receipts")
        texts = decode([ids[r, : int(lengths[r])].astype(np.int64).tolist() for r in range(ids.shape[0])])
        row_ids = [str(rr["row_id"]) for rr in receipts]
        for row_id, text in zip(row_ids, texts):
            index.add_text("codex_heldout", f"{pack['name']}:{row_id}", text)
        ids_by_pack[pack["name"]] = row_ids
        packs.append({"name": pack["name"], "manifest": str(manifest_path), "manifest_sha256": pack["manifest_sha256"],
                      "corpus": str(shard), "corpus_sha256": pack["corpus_sha256"], "rows": len(row_ids),
                      "tokens": int(lengths.sum())})
    every = [i for ids in ids_by_pack.values() for i in ids]
    if len(set(every)) != len(every) or len(every) != want_total:
        raise Refusal(f"held-out packs give {len(every)} rows / {len(set(every))} distinct ids, the decisions file says {want_total}")
    return {"name": "codex_heldout", "role": "Codex canonical validation + test rows (D2-X1), rendered text decoded from the six pinned packs",
            "rows": len(every), "packs": packs, "ids_by_pack": {k: len(v) for k, v in ids_by_pack.items()}}, ids_by_pack


def fidelity_check(index: HeldoutIndex, decisions: dict[str, Any], ids_by_pack: dict[str, list[str]]) -> dict[str, Any]:
    """The decoded rows ARE the held-out text: corpus-schema-v2's validation + test ids equal the v3 + v4 pack ids,
    and every schema-v2 row's longest message (>= 13 words) is caught by the decoded index."""

    lists = [e for e in decisions["canonical_split"]["row_id_lists"] if Path(e["path"]).name in ("validation.jsonl", "test.jsonl")]
    if len(lists) != 2:
        raise Refusal("the decisions file does not name corpus-schema-v2 validation.jsonl and test.jsonl")
    pack_ids = {i for name, ids in ids_by_pack.items() if not name.startswith("executed_tools") for i in ids}
    schema_ids, probes, caught, missed = set(), 0, 0, []
    for entry in lists:
        pinned(Path(entry["path"]), entry["sha256"], f"corpus-schema-v2 {Path(entry['path']).name}")
        for row in tc.read_jsonl(Path(entry["path"])):
            schema_ids.add(str(row["id"]))
            contents = [str(m.get("content") or "") for m in row.get("messages") or []]
            longest = max(contents, key=lambda c: len(dcb.words(c)), default="")
            if len(dcb.words(longest)) < FIDELITY_MIN_WORDS:
                continue
            probes += 1
            if any(h["rule"].startswith("ngram") and h["source"] == "codex_heldout" for h in index.hits(longest)):
                caught += 1
            elif len(missed) < 20:
                missed.append(str(row["id"]))
    if schema_ids != pack_ids:
        raise Refusal(f"corpus-schema-v2 validation + test ids ({len(schema_ids)}) != the v3 + v4 pack ids ({len(pack_ids)})")
    return {"rule": "schema-v2 validation + test ids == v3 + v4 pack ids; each schema-v2 row's longest message (>= 13 words) caught by the decoded index",
            "ids": len(schema_ids), "probes": probes, "caught": caught, "missed_first": missed, "ok": probes > 0 and caught == probes}


_ID_RE = re.compile(r'"id"\s*:\s*"([0-9a-f]{64})"')


def add_bfcl62(index: HeldoutIndex, decisions: dict[str, Any]) -> dict[str, Any]:
    """The 62 BFCL held-out source records and every twin record (bfcl62_rids.json), messages + tools."""

    spec = decisions["exclusion"]["sets"]["bfcl62"]
    pinned(Path(spec["path"]), spec["sha256"], "bfcl62_rids.json")
    holdout = tc.read_json(Path(spec["path"]))
    exposure = holdout["twin_exposure"]
    twins = {str(h["uid"]): h for h in exposure["hits"]}
    twin_rids = {str(h["rid"]) for h in exposure["hits"]}
    by_source = exposure["twin_ids_by_source"]
    for kind in ("text", "k3", "selfgen", "long"):
        missing = [r for r in by_source.get(kind) or [] if r not in twin_rids]
        if missing:
            raise Refusal(f"bfcl62 {kind} twin {missing[:3]} has no twin_exposure hit (no uid to read its text by)")
    missing = [u for u in by_source.get("native_train") or [] if u not in twins]
    if missing:
        raise Refusal(f"bfcl62 native-train twin {missing[:3]} is not a twin_exposure hit uid")
    sources = {str(u): "bfcl62" for u in holdout["uids"]}
    if len(sources) != BFCL_TASKS:
        raise Refusal(f"bfcl62_rids.json lists {len(sources)} uids, expected {BFCL_TASKS}")
    for uid in twins:
        sources.setdefault(uid, "bfcl62_twin")
    normalized = next((e for e in decisions["canonical_split"]["row_id_lists"] if Path(e["path"]).name == "normalized.jsonl"), None)
    if normalized is None:
        raise Refusal("the decisions file names no normalized.jsonl")
    pinned(Path(normalized["path"]), normalized["sha256"], "Codex normalized.jsonl")
    found: dict[str, dict[str, Any]] = {}
    with open(normalized["path"], encoding="utf-8") as handle:
        for line in handle:
            if not any(m in sources for m in _ID_RE.findall(line)):
                continue
            record = json.loads(line)
            if str(record.get("id")) in sources:
                found[str(record["id"])] = record
    absent = sorted(set(sources) - set(found))
    if absent:
        raise Refusal(f"{len(absent)} BFCL-62 / twin records absent from {normalized['path']} (first {absent[:3]})")
    for uid, record in sorted(found.items()):
        strings: list[str] = []
        dcb.harvest_strings({"messages": record.get("messages") or [], "tools": record.get("tools") or []}, strings)
        for i, text in enumerate(strings):
            index.add_text(sources[uid], f"{uid}#{i}", text)
    counts = Counter(sources.values())
    return {"name": "bfcl62", "role": "Codex's 62 BFCL held-out source records + their twins (registry text, K3, self-gen, native train), messages + tools",
            "path": spec["path"], "sha256": spec["sha256"], "normalized": normalized["path"], "normalized_sha256": normalized["sha256"],
            "records": dict(counts), "twin_rids_by_source": {k: len(v or []) for k, v in by_source.items()}}


def add_codex_274(index: HeldoutIndex, path: Path) -> dict[str, Any]:
    pinned(path, EVAL_TASKS_SHA, "Codex combined_oct13_v1 tasks.jsonl")
    tasks = tc.read_jsonl(path)
    if len(tasks) != EVAL_TASKS_ROWS or sum(1 for t in tasks if str(t["id"]).startswith(BFCL_TASK_PREFIX)) != BFCL_TASKS:
        raise Refusal(f"{path}: {len(tasks)} tasks, expected {EVAL_TASKS_ROWS} with {BFCL_TASKS} BFCL")
    for task in tasks:
        strings: list[str] = []
        dcb.harvest_strings({"input": task.get("input"), "oracle": task.get("oracle")}, strings)
        for i, text in enumerate(strings):
            index.add_text("codex_274", f"{task['id']}#{i}", text)
    return {"name": "codex_274", "role": "Codex's sealed 274-task set (combined_oct13_v1): input messages + tools + oracle", "path": str(path),
            "sha256": EVAL_TASKS_SHA, "tasks": len(tasks)}


def add_toolcall_bank(index: HeldoutIndex, path: Path) -> dict[str, Any]:
    bank = tc.read_json(path)
    if not tc.check_seal(bank, "bank_sha256"):
        raise Refusal(f"{path} does not match its own seal")
    names = sorted(bank.get("heldout_functions") or [])
    if names != sorted(list(tc.HELDOUT_SANDBOX) + list(tc.HELDOUT_API)):
        raise Refusal(f"{path} heldout_functions {names} != toolcall_selfgen's held-out tools")
    for case in bank["cases"]:
        strings: list[str] = []
        dcb.harvest_strings(case, strings)
        for i, text in enumerate(strings):
            index.add_text("toolcall_bank", f"{case['id']}#{i}", text)
    for name in names:
        index.add_literal("toolcall_bank_functions", name, name)
    return {"name": "toolcall_bank", "role": "omni_toolcall_bank_v1 whole cases (messages, tool descriptions, expected calls) + held-out function names (literal)",
            "path": str(path), "sha256": tc.file_sha256(path), "bank_sha256": bank["bank_sha256"], "cases": len(bank["cases"]), "heldout_functions": names}


def build_heldout_index(args: argparse.Namespace, decisions: dict[str, Any]) -> tuple[HeldoutIndex, list[dict[str, Any]], dict[str, Any]]:
    index = HeldoutIndex()
    decode = pack_decoder(Path(args.parent_dir))
    codex, ids_by_pack = add_codex_heldout(index, decisions, decode)
    sources = [codex, add_bfcl62(index, decisions), add_codex_274(index, Path(args.eval_tasks)), add_toolcall_bank(index, Path(args.bank))]
    fidelity = fidelity_check(index, decisions, ids_by_pack)
    return index, sources, fidelity


def build_review_surface(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """toolcall_selfgen finalize's own screen (the campaign review surface + eval split + bank strings)."""

    ns = argparse.Namespace(corpus_tools=str(HERE), eval_split=str(args.eval_split), bank=str(args.bank))
    try:
        return tc.build_screen(ns)
    except tc.ToolcallError as exc:
        raise Refusal(f"review surface: {exc}") from exc
    except dcb.Refusal as exc:
        raise Refusal(f"review surface: {exc}") from exc


# --------------------------------------------------------------------------
# the export: sidecar, sealed runs, rows bound to records
# --------------------------------------------------------------------------


def sidecar_path(corpus: Path) -> Path:
    return Path(corpus).with_suffix(".rows.json")


def load_export(corpus: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.numpy import load_file

    corpus = Path(corpus)
    side_path = sidecar_path(corpus)
    if not corpus.is_file() or not side_path.is_file():
        raise Refusal(f"no export at {corpus} (+ {side_path.name})")
    side = tc.read_json(side_path)
    if not tc.check_seal(side, "receipt_sha256"):
        raise Refusal(f"{side_path} does not match its own seal")
    corpus_sha = tc.file_sha256(corpus)
    if (side.get("corpus") or {}).get("sha256") != corpus_sha:
        raise Refusal(f"{side_path} names corpus {str((side.get('corpus') or {}).get('sha256'))[:16]}, the file is {corpus_sha[:16]}")
    by_id: dict[str, dict[str, Any]] = {}
    run_of: dict[str, int] = {}
    concat: list[str] = []
    for k, source in enumerate(side["source_manifests"]):
        try:
            manifest, rows = tc.sealed_conversations(Path(source["out_dir"]))
        except tc.ToolcallError as exc:
            raise Refusal(f"source run {source['out_dir']}: {exc}") from exc
        if manifest["manifest_sha256"] != source["manifest_sha256"]:
            raise Refusal(f"source run {source['out_dir']} is sealed {manifest['manifest_sha256'][:16]}, the sidecar names {source['manifest_sha256'][:16]}")
        for record in rows:
            if record["id"] in by_id:
                raise Refusal(f"conversation {record['id']} repeats across the source runs")
            by_id[record["id"]] = record
            run_of[record["id"]] = k
            concat.append(record["id"])
    row_ids = [str(i) for i in side["row_ids"]]
    prior = side.get("heldout_screen_drop")
    if prior is None:
        if row_ids != concat:
            raise Refusal("the sidecar's row_ids are not the source runs' conversations in order")
    else:  # an export this tool already screened: its rows are the runs' conversations minus the dropped ones, in order
        kept = set(row_ids)
        if [i for i in concat if i in kept] != row_ids or len(kept) != len(row_ids):
            raise Refusal("the screened sidecar's row_ids are not an ordered subset of the source runs' conversations")
    tensors = load_file(str(corpus))
    with safe_open(str(corpus), "np") as handle:
        metadata = dict(handle.metadata() or {})
    ids, lengths, target = tensors["input_ids"], tensors["lengths"], tensors["target_mask"]
    if ids.shape[0] != len(row_ids) or int(side["rows"]) != len(row_ids):
        raise Refusal(f"corpus rows {ids.shape[0]} / sidecar rows {side['rows']} / row_ids {len(row_ids)} disagree")
    supervised = target.sum(axis=1)
    for n, row_id in enumerate(row_ids):
        render = by_id[row_id]["render"]
        if int(lengths[n]) != int(render["tokens"]) or int(supervised[n]) != int(render["supervised"]):
            raise Refusal(f"corpus row {n} ({row_id}): length {int(lengths[n])} / supervised {int(supervised[n])}, "
                          f"its sealed conversation renders {render['tokens']} / {render['supervised']}")
    return {"corpus": corpus, "sidecar": side_path, "side": side, "corpus_sha256": corpus_sha, "records": [by_id[i] for i in row_ids],
            "run_of": run_of, "tensors": tensors, "metadata": metadata}


def screened_texts(record: dict[str, Any]) -> list[str]:
    """toolcall_selfgen.screen_texts (contents, call arguments, tool and parameter descriptions) + one line of every
    tool and call name (the held-out function names are literals)."""

    names = [tc.tool_function(t).get("name", "") for t in record.get("tools") or []]
    names += [c["function"]["name"] for m in record["messages"] for c in m.get("tool_calls") or []]
    return tc.screen_texts(record) + [" ".join(n for n in names if n)]


def screen_rows(records: list[dict[str, Any]], review: Any, heldout: HeldoutIndex) -> list[dict[str, Any]]:
    hits = []
    for n, record in enumerate(records):
        reasons: list[dict[str, str]] = []
        for text in screened_texts(record):
            for hit in review.hits(text):
                reasons.append({**hit, "index": "review_surface"})
            for hit in heldout.hits(text):
                reasons.append({**hit, "index": "d2x1_heldout"})
            if len(reasons) >= 5:
                break
        if reasons:
            hits.append({"row": n, "id": record["id"], "reasons": reasons[:5]})
    return hits


# --------------------------------------------------------------------------
# the drop: backup, rewritten corpus + sidecar
# --------------------------------------------------------------------------


def write_dropped_corpus(export: dict[str, Any], drop_rows: set[int], backup_root: Path) -> dict[str, Any]:
    import numpy as np
    from safetensors.numpy import save_file

    corpus, side_path, side = export["corpus"], export["sidecar"], export["side"]
    tensors = export["tensors"]
    keep = [n for n in range(len(export["records"])) if n not in drop_rows]
    if not keep:
        raise Refusal("the screen would drop every row of the export")
    size = corpus.stat().st_size
    free = shutil.disk_usage(corpus.parent).free
    if free - 2 * size < ROOT_FLOOR_BYTES:
        raise Deferral(f"{corpus.parent}: {free / 1e9:.1f} GB free; rewriting the corpus would cross the {ROOT_FLOOR_BYTES / 1e9:.0f} GB floor")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = Path(backup_root) / f"{stamp}-{export['corpus_sha256'][:16]}"
    backup.mkdir(parents=True, exist_ok=False)
    for src in (corpus, side_path):
        shutil.copy2(src, backup / src.name)
        if tc.file_sha256(backup / src.name) != tc.file_sha256(src):
            raise Refusal(f"backup copy of {src} does not reproduce its sha")
    lengths = tensors["lengths"][keep]
    width = int(lengths.max())
    ids = np.ascontiguousarray(tensors["input_ids"][keep])
    target = np.ascontiguousarray(tensors["target_mask"][keep])
    if (ids[:, width:] != tc.PAD_ID).any() or target[:, width:].any():
        raise Refusal("a kept row holds tokens or targets beyond its length")
    ids, target = np.ascontiguousarray(ids[:, :width]), np.ascontiguousarray(target[:, :width])
    records = [export["records"][n] for n in keep]
    dropped_ids = {export["records"][n]["id"] for n in drop_rows}
    metadata = dict(export["metadata"])
    metadata.update({"heldout_screen": f"{TOOL_NAME} {TOOL_VERSION}", "unscreened_sha256": export["corpus_sha256"],
                     "heldout_screen_dropped_rows": str(len(drop_rows))})
    tmp = corpus.with_name(f".{corpus.name}.tmp-{os.getpid()}")
    save_file({"input_ids": ids, "lengths": lengths.astype(np.int64), "target_mask": target}, str(tmp), metadata=metadata)
    new_sha = tc.file_sha256(tmp)
    sources, start = [], 0
    for k, source in enumerate(side["source_manifests"]):
        mine = [r for r in records if export["run_of"][r["id"]] == k]
        gone = sorted(i for i in dropped_ids if export["run_of"][i] == k)
        sources.append({**source, "row_range": [start, start + len(mine)], "conversations_kept": len(mine), "dropped": len(gone),
                        "tokens_kept": sum(int(r["render"]["tokens"]) for r in mine),
                        "supervised_tokens_kept": sum(int(r["render"]["supervised"]) for r in mine),
                        "row_range_before_screen": (source.get("row_range_before_screen") or source["row_range"])})
        start += len(mine)
    body = {k: v for k, v in side.items() if k != "receipt_sha256"}
    body.update({"tool": tool_identity(), "corpus": {"path": str(corpus.resolve()), "sha256": new_sha, "bytes": tmp.stat().st_size},
                 "source_manifests": sources, "rows": len(records), "row_length": width, "tokens": int(lengths.sum()),
                 "supervised_tokens": int(target.sum()), "row_ids": [r["id"] for r in records],
                 "categories": [r["category"] for r in records],
                 "derived_from": {"tool": side.get("tool"), "receipt_sha256": side["receipt_sha256"], "corpus_sha256": export["corpus_sha256"],
                                  "rows": len(export["records"]), "backup_dir": str(backup)},
                 "heldout_screen_drop": {"rows": len(drop_rows), "ids": sorted(dropped_ids), "unscreened_corpus_sha256": export["corpus_sha256"],
                                         "prior": side.get("heldout_screen_drop")}})
    new_side = tc.seal(body, "receipt_sha256")
    os.replace(tmp, corpus)
    tc.write_json_atomic(side_path, new_side)
    log(f"rewrote {corpus.name}: {len(records)} rows x {width} (was {len(export['records'])}), sha {new_sha[:16]}; unscreened copy in {backup}")
    return {"backup_dir": str(backup), "corpus_sha256": new_sha, "sidecar_receipt_sha256": new_side["receipt_sha256"]}


# --------------------------------------------------------------------------
# receipt
# --------------------------------------------------------------------------


def receipt_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    corpus = Path(args.corpus)
    receipt = Path(args.receipt) if getattr(args, "receipt", None) else corpus.parent / "heldout_screen.json"
    dropped = Path(args.dropped) if getattr(args, "dropped", None) else corpus.parent / "heldout_screen.dropped.json"
    return receipt, dropped


def check_receipt(receipt: Path, corpus: Path) -> tuple[bool, str]:
    """d2_plan.screen_state's conditions (schema, decisions sha, corpus sha = the corpus as it is now, ngram 13,
    hits_in_corpus 0) + the receipt's seal + its dropped list's sha + the sidecar naming the corpus."""

    if not Path(receipt).is_file():
        return False, f"no receipt {receipt}"
    try:
        r = tc.read_json(receipt)
    except (OSError, ValueError) as exc:
        return False, f"unreadable receipt: {exc}"
    if not Path(corpus).is_file():
        return False, f"no corpus {corpus}"
    sha = tc.file_sha256(corpus)
    problems = []
    if r.get("schema") != SCHEMA:
        problems.append(f"schema {r.get('schema')!r}")
    if not tc.check_seal(r, "receipt_sha256"):
        problems.append("receipt does not match its seal")
    if r.get("decisions_sha256") != DECISIONS_SHA:
        problems.append("another decisions file")
    if r.get("corpus_sha256") != sha:
        problems.append(f"names corpus {str(r.get('corpus_sha256'))[:16]}, the file is {sha[:16]}")
    if r.get("ngram") != NGRAM or r.get("hits_in_corpus") != 0:
        problems.append(f"ngram {r.get('ngram')} hits_in_corpus {r.get('hits_in_corpus')}")
    dropped = (r.get("dropped") or {})
    if not dropped.get("path") or not Path(dropped["path"]).is_file() or tc.file_sha256(Path(dropped["path"])) != dropped.get("sha256"):
        problems.append("dropped list missing or changed")
    side = sidecar_path(corpus)
    try:
        if (tc.read_json(side).get("corpus") or {}).get("sha256") != sha:
            problems.append("the sidecar does not name the corpus")
    except (OSError, ValueError):
        problems.append("no readable sidecar")
    return not problems, "; ".join(problems) or f"valid: {sha[:16]} screened, 0 hits"


class ScreenLock:
    def __init__(self, path: Path, wait_s: float) -> None:
        self.path, self.wait_s, self.handle = Path(path), float(wait_s), None

    def __enter__(self) -> "ScreenLock":
        if fcntl is None:
            return self
        self.handle = open(self.path, "a")
        deadline = time.time() + self.wait_s
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.time() >= deadline:
                    self.handle.close()
                    raise Deferral(f"{self.path} held by another screen for {self.wait_s:.0f} s")
                time.sleep(5)

    def __exit__(self, *exc: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def screen_command(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    receipt_path, dropped_path = receipt_paths(args)
    with ScreenLock(corpus.parent / ".heldout_screen.lock", args.lock_wait):
        ok, why = check_receipt(receipt_path, corpus)
        if ok and not args.force:
            log(f"already screened: {why} ({receipt_path}); nothing written (--force re-screens)")
            return 0
        decisions = load_decisions(Path(args.decisions))
        export = load_export(corpus)
        log(f"export {corpus.name} sha {export['corpus_sha256'][:16]}: {len(export['records'])} rows from "
            f"{len(export['side']['source_manifests'])} run(s){' (already screened once)' if export['side'].get('heldout_screen_drop') else ''}")
        started = time.time()
        heldout, heldout_sources, fidelity = build_heldout_index(args, decisions)
        if not fidelity["ok"]:
            raise Refusal(f"decode fidelity: {fidelity['caught']}/{fidelity['probes']} schema-v2 rows caught by the decoded held-out index "
                          f"(first missed {fidelity['missed_first'][:3]})")
        live = liveness(heldout)
        if not live["screen_is_live"]:
            raise Refusal(f"the D2-X1 index is not live: {json.dumps(live['totals'])} clean {live['clean_controls_falsely_caught']} "
                          f"no-probe {live['sources_without_probe']}")
        review, review_meta = build_review_surface(args)
        review_live = dcb.screen_liveness(review)
        if not review_live["screen_is_live"]:
            raise Refusal(f"the review-surface index is not live: {json.dumps(review_live['totals'])}")
        log(f"indexes built in {time.time() - started:.0f} s: D2-X1 {heldout.summary()['ngram_keys']:,} 13-gram keys over "
            f"{dict(heldout.by_source)}; fidelity {fidelity['caught']}/{fidelity['probes']}; review surface {review_meta['index']['ngram_count']:,} grams")
        hits = screen_rows(export["records"], review, heldout)
        records = export["records"]
        fraction = len(hits) / max(1, len(records))
        dropped_doc = {"schema": DROPPED_SCHEMA, "tool": tool_identity(), "corpus": str(corpus), "screened_corpus_sha256": export["corpus_sha256"],
                       "decisions_sha256": DECISIONS_SHA, "rows_screened": len(records), "applied": False,
                       "dropped": [{**h, "category": records[h["row"]]["category"], "run": export["side"]["source_manifests"][export["run_of"][h["id"]]]["out_dir"],
                                    "tokens": int(records[h["row"]]["render"]["tokens"]), "supervised": int(records[h["row"]]["render"]["supervised"])}
                                   for h in hits],
                       "written_utc": tc.utc_now()}
        if fraction > args.max_drop_fraction:
            tc.write_json_atomic(dropped_path, dropped_doc)
            raise Refusal(f"{len(hits)} of {len(records)} rows hit ({fraction:.1%} > --max-drop-fraction {args.max_drop_fraction:.1%}): "
                          f"nothing dropped; the would-drop list is {dropped_path} (applied false) -- a screen this wide is matching boilerplate")
        drop_meta = None
        if hits:
            drop_meta = write_dropped_corpus(export, {h["row"] for h in hits}, Path(args.backup_root))
            after = load_export(corpus)
            if after["corpus_sha256"] != drop_meta["corpus_sha256"] or [r["id"] for r in after["records"]] != \
                    [r["id"] for n, r in enumerate(records) if n not in {h["row"] for h in hits}]:
                raise Refusal("the rewritten corpus does not hold exactly the kept rows")
            export = after
        recheck = screen_rows(export["records"], review, heldout)
        if recheck:
            raise Refusal(f"{len(recheck)} rows of the corpus still hit after the drop (first {recheck[0]['id']})")
        dropped_doc["applied"] = True
        dropped_doc["corpus_sha256_after"] = export["corpus_sha256"]
        dropped_doc["prior"] = export["side"].get("heldout_screen_drop") if not hits else None
        tc.write_json_atomic(dropped_path, dropped_doc)
        if drop_meta:  # beside the unscreened copy too: a later --force re-screen rewrites the list, never these reasons
            shutil.copy2(dropped_path, Path(drop_meta["backup_dir"]) / dropped_path.name)
        side = export["side"]
        by_run = Counter(d["run"] for d in dropped_doc["dropped"])
        body = {
            "schema": SCHEMA, "tool": tool_identity(), "toolcall_selfgen": {"version": tc.TOOL_VERSION, "sha256": tc.file_sha256(Path(tc.__file__))},
            "decisions": str(Path(args.decisions)), "decisions_sha256": DECISIONS_SHA, "rule": "D2-X1 per_input toolcall_selfgen; prereg A2 3(d)",
            "corpus": str(corpus.resolve()), "corpus_sha256": export["corpus_sha256"], "corpus_bytes": corpus.stat().st_size,
            "sidecar": str(export["sidecar"]), "sidecar_receipt_sha256": side["receipt_sha256"],
            "rows": len(export["records"]), "tokens": int(side["tokens"]), "supervised_tokens": int(side["supervised_tokens"]),
            "source_manifests": side["source_manifests"],
            "ngram": NGRAM, "hits_in_corpus": 0,
            "screened_export": {"corpus_sha256": dropped_doc["screened_corpus_sha256"], "rows": len(records),
                                "tokens": sum(int(r["render"]["tokens"]) for r in records),
                                "supervised_tokens": sum(int(r["render"]["supervised"]) for r in records)},
            "hits_in_export": len(hits),
            "dropped": {"path": str(dropped_path), "sha256": tc.file_sha256(dropped_path), "rows": len(hits),
                        "tokens": sum(d["tokens"] for d in dropped_doc["dropped"]), "supervised_tokens": sum(d["supervised"] for d in dropped_doc["dropped"]),
                        "by_run": dict(by_run), "by_index_source": dict(Counter(f"{r['index']}:{r['source']}" for d in dropped_doc["dropped"] for r in d["reasons"][:1])),
                        "prior": side.get("heldout_screen_drop") if not hits else None},
            "rewrite": drop_meta,
            "screen": {"heldout_index": heldout.summary(), "heldout_sources": heldout_sources, "decode_fidelity": fidelity, "heldout_liveness": live,
                       "review_surface": review_meta, "review_liveness": {k: v for k, v in review_live.items() if k != "by_source"},
                       "screened_text": "toolcall_selfgen.screen_texts (message contents, call arguments, tool + parameter descriptions) + the tool / call names line, per piece",
                       "max_drop_fraction": args.max_drop_fraction},
            "written_utc": tc.utc_now(),
        }
        receipt = tc.seal(body, "receipt_sha256")
        tc.write_json_atomic(receipt_path, receipt)
        ok, why = check_receipt(receipt_path, corpus)
        if not ok:
            raise Refusal(f"the receipt just written does not check: {why}")
        log(f"SCREENED {corpus} sha {export['corpus_sha256'][:16]}: {len(hits)} of {len(records)} rows hit and dropped "
            f"({sum(d['tokens'] for d in dropped_doc['dropped']):,} tokens); receipt {receipt_path} ({tc.file_sha256(receipt_path)[:16]}), "
            f"dropped list {dropped_path}")
        return 0


def check_command(args: argparse.Namespace) -> int:
    receipt_path, _ = receipt_paths(args)
    ok, why = check_receipt(receipt_path, Path(args.corpus))
    print(f"HELDOUT-SCREEN {'OK' if ok else 'NOT-OK'} {why}", flush=True)
    return 0 if ok else 10


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("screen")
    sp.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    sp.add_argument("--receipt", default=None, help="default: heldout_screen.json beside the corpus (d2_plan reads it there)")
    sp.add_argument("--dropped", default=None, help="default: heldout_screen.dropped.json beside the corpus")
    sp.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    sp.add_argument("--eval-tasks", default=str(DEFAULT_EVAL_TASKS))
    sp.add_argument("--bank", default=str(tc.DEFAULT_BANK))
    sp.add_argument("--eval-split", default=str(tc.DEFAULT_EVAL_SPLIT))
    sp.add_argument("--parent-dir", default=str(tc.DEFAULT_PARENT))
    sp.add_argument("--backup-root", default=str(DEFAULT_BACKUP_ROOT))
    sp.add_argument("--max-drop-fraction", type=float, default=MAX_DROP_FRACTION)
    sp.add_argument("--lock-wait", type=float, default=3600.0)
    sp.add_argument("--force", action="store_true", help="re-screen even when the receipt is valid for the corpus")
    sp = sub.add_parser("check")
    sp.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    sp.add_argument("--receipt", default=None)
    return p


COMMANDS = {"screen": screen_command, "check": check_command}


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(list(argv) if argv is not None else None)
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.command}")
    try:
        return COMMANDS[args.command](args)
    except (Refusal, Deferral) as exc:
        log(f"{'REFUSED' if isinstance(exc, Refusal) else 'DEFERRED'}: {exc.message}")
        return int(exc.code)


if __name__ == "__main__":
    sys.exit(main())
