#!/usr/bin/env python3
"""d2_refill_build.py -- D2-X1 refill producers for ranks 2 and 3 (D2_DECISIONS_2026-10-15 refill_policy) and the
eligibility screen receipt d2_plan (>= 1.4.0, refill_screen_state) needs before a refill corpus supplies a D2 token.

Lane refill (L-TRAIN), Claude lead, 2026-10-15.  Inputs are read-only; every output is a new file under
$OC/distill/corpus/d2_refill/ (a corpus is never rewritten with other bytes: rebuild into a new directory instead).

  codex-native --pack v7|v8   Codex's native train pack (native-text-v7-train: 168 executed-tool rows; native-text-v8-train:
                              1,536 verified-intelligence rows; both "not yet in D2") -> codex-native-<pack>-train.safetensors.
                              The rows are COPIED, never re-tokenised, re-rendered or re-masked (the d2_bfcl_native_copy
                              convention): input_ids int32 [N, W], lengths int64 [N], target_mask bool [N, W] (target-indexed:
                              token t is an assistant target; qtext_teacher_cache shifts it), assistant_mask bool [N, W];
                              W = the longest kept row.  D2 trains them supervision=assistant_only (d2_plan REFILL_CANDIDATES).
  selfgen --run DIR [...]     sealed selfgen_sampler runs -> selfgen-refill.safetensors in the D2 self-gen corpus's own format
                              (rescore_target.safetensors 7b73d3d9: input_ids int32, lengths int64, store_mask u8 [prompt_len-1,
                              length-2], target_mask u8 [prompt_len, length), eog_terminated u8), only rows NOT already in D2
                              (the rids of rescore_export.rows.json) that end at a parent EOG id.  Nothing eligible: no corpus,
                              a receipt saying so, exit 3.
  state                       every rank 2-3 candidate: corpus sha, eligibility receipt, Q38 cache (one JSON line; the runner
                              d2refill_recache.py reads it).
  totals                      expected supervised (KD-carrying) tokens per refill source against the refill target ->
                              $OC/distill/corpus/d2_refill/REFILL_TOTALS.json, for L-TRAIN's applied receipt.

ELIGIBILITY (decisions refill_policy.eligibility, all three, checked here and written into the receipt):
  1. train split only; never one of the 5,391 codex_heldout_ids, a BFCL-62 rid / uid or a twin (bfcl62_rids.json 538678aa);
     never in a held-out recovery group; never a record whose registry origin is a held-out rid.  Any such row is a
     refusal (the row lists come from Codex's split, so a hit means the split is not what the decisions file says).
  2. a 13-gram screen against the rendered text of the 5,391 held-out rows (every message, every role, every tool call),
     the BFCL-62 tasks and Codex's 274-task set (combined_oct13_v1, every string of every task): distill_corpus_build's
     own ReviewIndex at n = 13 (exact-text and 13-gram rules; short items off); plus (1.0.2) the campaign review surface
     exactly as toolcall_selfgen finalize builds it (distill_corpus_build.build_review_index: bank v1 / v2, probe and
     smoke packs, R-Q prompts, needle sets, ...; all of ReviewIndex's rules), so no refill row can carry a promotion-gate
     item either.  The C1 receipt is lane toolscreen's (toolcall_heldout_screen.py).  A row with a hit is DROPPED and counted
     (the toolcall_selfgen finalize convention); the kept rows are screened again and the receipt's hits_in_corpus is
     that second count (0 or no corpus).  Screen surface of a candidate row: every message and tool call; a system
     message is left out ONLY in a pack where one identical system text opens every row (the tool-seed instruction that
     also opens every held-out executed-tool row: fixed scaffolding, never a loss position under assistant_only); the
     receipt names that text's sha256.
  3. verified rows only: Codex rows from an admitted train pack (split train, 0 quarantined, no cross-split group, its
     corpus sha = the manifest's); self-gen rows ending at a parent EOG id from a sealed run whose shards verify.

The receipt (<corpus file>.heldout_screen.json, schema omni-d2x1-refill-screen-v1) carries what d2_plan 1.4.0 checks
(decisions_sha256, corpus_sha256, ngram 13, hits_in_corpus 0, split "train", excluded_id_hits 0, verified_rows_only)
plus the evidence: input shas, the index summary, every dropped row and why.

Usage (server, CPU, lab venv):
  CUDA_VISIBLE_DEVICES= nice -n 15 $PY d2_refill_build.py codex-native --pack v7
  CUDA_VISIBLE_DEVICES= nice -n 15 $PY d2_refill_build.py selfgen --run $OC/distill/selfgen/run-q38-27bu-q4km-2026-10-13
  $PY d2_refill_build.py state ; $PY d2_refill_build.py totals
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TOOL_NAME = "d2_refill_build"
TOOL_VERSION = "1.0.3"  # 2026-10-15: an existing corpus whose TENSORS equal the new build is kept (its metadata names
# the tool version that wrote it, so a patch-version rebuild differs in bytes only there) and the receipt is re-written
# for it; any other difference is still a refusal.  1.0.2 refused the re-screen of the 1.0.0 corpora on exactly that.
# 1.0.2 2026-10-15: the campaign review surface joins the screen (as toolcall finalize and the C1
# screen build it); `screen-toolcall` removed: lane toolscreen's toolcall_heldout_screen.py owns the C1 receipt (one writer)
#   (_superseded/d2_refill_build_1.0.2.py)
# 1.0.1 2026-10-15: totals also print d2_plan's clean-form target (if D2 takes text_clean_v1); _superseded/d2_refill_build_1.0.1.py
# 1.0.0 2026-10-15: first version (lane refill, Claude lead); built the canonical d2_refill corpora; _superseded/d2_refill_build_1.0.0.py

HERE = Path(__file__).resolve().parent
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
LEX = Path(os.environ.get("LEX", "/mnt/lexar/omni-campaign"))
CODEX = OC / "distill" / "codex-preparation-01a1326f-20261013"
REFILL_DIR = OC / "distill" / "corpus" / "d2_refill"
DECISIONS = OC / "distill" / "queues" / "D2_DECISIONS_2026-10-15.json"
DECISIONS_SHA = "c8d47f22784d265a2e9239f9274c062298d25278497ba21d01102aa45274d90d"
BFCL62 = OC / "distill" / "corpus" / "holdouts" / "bfcl62_rids.json"
BFCL62_SHA = "538678aacbdd1e9dfe83253b30d2276f4da3d3349724c7e7c78a3cbf5d594cb0"
TASKS274 = LEX / "distill" / "d2" / "inputs" / "codex" / "combined_oct13_v1.tasks.jsonl"
TASKS274_SHA = "3795f333998627bb3b2d6fd460c19228469b2286284dfaadf30c5e12d23713f8"  # bfcl_native_v1/rows.json eval_tasks_sha256
TASKS274_ROWS = 274
HELDOUT_RIDS = OC / "distill" / "corpus" / "d2_masks_v2" / "heldout_rids.json"  # registry rids of the 5,055 joined ids
BFCL_NATIVE_ROWS = OC / "distill" / "corpus" / "bfcl_native_v1" / "rows.json"  # the BFCL-native train rows already in D2
SELFGEN_D2_ROWS = LEX / "teacher_cache" / "qtext" / "inputs" / "selfgen" / "rescore_export.rows.json"
SELFGEN_D2_MANIFEST = "fe1f0784777b8fdfc5cbae75ee8895bba3ad8e997f8c54b1b41aa086987c313f"  # the sealed run D2 already reads
#: held-out record files: Codex's text records (corpus-schema-v2, pinned by the decisions file's row_id_lists) and its
#: executed-tool records (corpus-tool-v3: the 168 + 168 ids of native-text-v7-validation / -test)
HELDOUT_RECORDS = (
    ("corpus-schema-v2/validation.jsonl", "ffba4ea37346b1fa26a7816fa356e16f711f118b6364e4738a5caa0d431b36c4"),
    ("corpus-schema-v2/test.jsonl", "8544046b39b9457e3c7fbead9839718567f0599ba6989de071610a63badb9151"),
    ("corpus-tool-v3/validation.jsonl", "61526c6239c3892afa36bf407b480953b18d5694272121b3a9bf219238d7cb02"),
    ("corpus-tool-v3/test.jsonl", "9ac970fa341d390453a46d85fc8739c26534ba9a6fd9d744dd6def155849e065"),
)
HELDOUT_IDS_TOTAL = 5391
HELDOUT_REGISTRY_RIDS = 5055  # the joined registry documents (decisions cross_checks.joined_to_registry_rids)
#: rank 3 packs (decisions refill_policy.order[rank 3].rows): pack dir, its records, the corpus name d2_plan reads
PACKS = {
    "v7": {"key": "executed_tools_train", "dir": "native-text-v7-train", "out": "codex-native-v7-train.safetensors",
           "records": ("corpus-tool-v3/train.jsonl", "743ef837c36667b410c9cbaf3e2a25a7f41f3098730bdc6bd5d1d68938b62db3"),
           "system_scaffold": True},
    "v8": {"key": "verified_intelligence_train", "dir": "native-text-v8-train", "out": "codex-native-v8-train.safetensors",
           "records": ("intelligence-seed-v1/train.jsonl", "7a5d4690b120520456a8f40ca3063abd971bec8ce6286ca5dabdc0e92376e1e7"),
           "system_scaffold": False},
}
SELFGEN_OUT = "selfgen-refill.safetensors"
SCREEN_SCHEMA = "omni-d2x1-refill-screen-v1"  # d2_plan 1.4.0 REFILL_SCREEN_SCHEMA
TOTALS_SCHEMA = "omni-d2x1-refill-totals-v1"
NGRAM = 13
TEXT_VOCAB_ROWS = 248_320
PAD_ID = 248_044
EOG_IDS = (248_046, 248_044)
METADATA_KEY = "omni_d2_refill"
EXIT_NOTHING = 3


class Refusal(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json_atomic(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def pinned(path: Path, want: str | None, what: str, no_expect: bool = False) -> str:
    if not Path(path).is_file():
        raise Refusal(f"{what} {path} does not exist")
    got = sha256_file(Path(path))
    if want and got != want and not no_expect:
        raise Refusal(f"{what} {path} is {got[:16]}, not the pinned {want[:16]}")
    return got


def tool_identity() -> dict[str, Any]:
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))}


def screen_path(corpus: Path) -> Path:
    """d2_plan 1.4.0 refill_screen_state: the receipt sits beside the corpus as <file>.heldout_screen.json."""
    return Path(corpus).with_name(Path(corpus).name + ".heldout_screen.json")


def receipt_problems(receipt: dict[str, Any] | None, corpus_sha: str) -> list[str]:
    """The fields d2_plan 1.4.0 refill_screen_state accepts (test_d2_refill_build pins this against d2_plan's own check
    whenever that function exists, so the two cannot drift apart unnoticed)."""
    if receipt is None:
        return ["no eligibility screen receipt"]
    problems = []
    if receipt.get("schema") != SCREEN_SCHEMA:
        problems.append(f"schema {receipt.get('schema')!r}")
    if receipt.get("decisions_sha256") != DECISIONS_SHA:
        problems.append("another decisions file")
    if receipt.get("corpus_sha256") != corpus_sha:
        problems.append(f"screened corpus {str(receipt.get('corpus_sha256'))[:16]} is not {corpus_sha[:16]}")
    if receipt.get("ngram") != NGRAM or receipt.get("hits_in_corpus") != 0:
        problems.append(f"ngram {receipt.get('ngram')} hits_in_corpus {receipt.get('hits_in_corpus')}")
    if receipt.get("split") != "train" or receipt.get("excluded_id_hits") != 0 or receipt.get("verified_rows_only") is not True:
        problems.append(f"split {receipt.get('split')!r} excluded_id_hits {receipt.get('excluded_id_hits')} "
                        f"verified_rows_only {receipt.get('verified_rows_only')}")
    return problems


# --------------------------------------------------------------------------
# the held-out screen
# --------------------------------------------------------------------------


def import_corpus_tools(tools: Path) -> Any:
    sys.path.insert(0, str(tools))
    import distill_corpus_build as dcb  # noqa: E402  (the one ReviewIndex / words rule of every screen in this campaign)

    return dcb


def message_texts(record: dict[str, Any], *, skip_system: str | None = None) -> list[tuple[str, str]]:
    """(role, text) of every message content and every tool call of a Codex record (all roles unless the one scaffold
    system text is named)."""
    out = []
    for message in record.get("messages") or []:
        role = str(message.get("role"))
        content = message.get("content")
        if content:
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, ensure_ascii=False)
            if not (role == "system" and skip_system is not None and text == skip_system):
                out.append((role, text))
        for call in message.get("tool_calls") or []:
            out.append((f"{role}:tool_call", json.dumps(call.get("function"), sort_keys=True, ensure_ascii=False)))
    return out


class Heldout:
    """The 5,391 held-out records, the BFCL-62 tasks + twins and the 274-task set: ids, groups, and the 13-gram index."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.dcb = import_corpus_tools(Path(args.tools))
        self.inputs: dict[str, Any] = {}
        self.inputs["decisions"] = {"path": str(args.decisions), "sha256": pinned(args.decisions, DECISIONS_SHA, "decisions file")}
        self.decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
        # 1. the held-out ids: the row_receipts of the six packs the decisions file names, manifests at their pinned shas
        ids: set[str] = set()
        packs = []
        for pack in self.decisions["canonical_split"]["heldout_packs"]:
            got = pinned(Path(pack["manifest"]), pack["manifest_sha256"], f"held-out pack {pack['name']} manifest")
            receipts = json.loads(Path(pack["manifest"]).read_text(encoding="utf-8"))["row_receipts"]
            if len(receipts) != int(pack["rows"]):
                raise Refusal(f"{pack['name']}: {len(receipts)} row receipts, the decisions file says {pack['rows']}")
            ids |= {r["row_id"] for r in receipts}
            packs.append({"name": pack["name"], "manifest_sha256": got, "rows": len(receipts)})
        if len(ids) != HELDOUT_IDS_TOTAL:
            raise Refusal(f"{len(ids)} distinct held-out ids, the decisions file says {HELDOUT_IDS_TOTAL}")
        self.ids = ids
        self.inputs["heldout_packs"] = packs
        # 2. their records (texts and groups); every held-out id must be found exactly once
        index = self.dcb.ReviewIndex(ngram=NGRAM, short_min=10**9)  # exact + 13-gram rules; short items off
        found: Counter = Counter()
        self.groups: set[str] = set()
        self.registry_rids: set[str] = set()
        self.records: dict[str, dict[str, Any]] = {}
        files = []
        for rel, want in HELDOUT_RECORDS:
            path = Path(args.codex_dir) / rel
            files.append({"path": str(path), "sha256": pinned(path, want, "held-out records", args.no_expect)})
            for r in read_jsonl(path):
                if r["id"] not in ids:
                    continue
                found[r["id"]] += 1
                self.records[r["id"]] = r
                p = r.get("provenance") or {}
                for key in ("group_id", "recovery_group_id"):
                    if p.get(key):
                        self.groups.add(str(p[key]))
                origin = p.get("registry_origin") or {}
                if origin.get("rid"):
                    self.registry_rids.add(str(origin["rid"]))
                for role, text in message_texts(r):
                    index.add_text(f"heldout:{rel.split('/')[0]}", r["id"], text)
                if r.get("tools"):  # an executed-tool row's catalog (a schema answer must not slip past as "scaffolding")
                    index.add_text(f"heldout:{rel.split('/')[0]}", r["id"], json.dumps(r["tools"], sort_keys=True, ensure_ascii=False))
        missing = sorted(ids - set(found))
        twice = sorted(k for k, n in found.items() if n > 1)
        if missing or twice:
            raise Refusal(f"held-out records: {len(missing)} ids not found (first {missing[:2]}), {len(twice)} found twice")
        self.inputs["heldout_records"] = files
        # 3. registry rids of the joined held-out documents (d2_mask_build 1.2.0 heldout_rids.json)
        hr = load_json(Path(args.heldout_rids))
        if not hr or hr.get("schema") != "omni-d2-heldout-rids-v1" or len(hr.get("rids") or []) != HELDOUT_REGISTRY_RIDS:
            raise Refusal(f"{args.heldout_rids}: not a d2_mask_build heldout_rids file with {HELDOUT_REGISTRY_RIDS:,} rids")
        self.registry_rids |= set(hr["rids"])
        self.inputs["heldout_rids"] = {"path": str(args.heldout_rids), "sha256": sha256_file(Path(args.heldout_rids)),
                                       "rids": len(hr["rids"])}
        # 4. BFCL-62: rids, uids (= Codex source row ids, all inside the held-out ids), task ids, twins
        sha = pinned(Path(args.bfcl62), BFCL62_SHA, "bfcl62_rids.json", args.no_expect)
        b = json.loads(Path(args.bfcl62).read_text(encoding="utf-8"))
        self.bfcl_rids = set(b["rids"])
        self.bfcl_uids = set(b["uids"])
        self.bfcl_tasks = set(b["task_ids"])
        twins: set[str] = set()
        for values in ((b.get("twin_exposure") or {}).get("twin_ids_by_source") or {}).values():
            twins |= {str(v) for v in values}
        for key in ("twin_rids", "twin_uids"):
            twins |= {str(v) for v in (b.get("twin_exposure") or {}).get(key) or []}
        self.twins = twins
        if not self.bfcl_uids <= ids:
            raise Refusal(f"{len(self.bfcl_uids - ids)} BFCL-62 source uids are not held-out ids")
        from bfcl62_holdout import expected_calls  # noqa: E402  (the one twin rule)

        self.bfcl_calls = {uid: expected_calls(self.records[uid]) for uid in self.bfcl_uids}
        # the recovery-schema records may drop the BFCL adaptation block; d2_bfcl_native_copy reads the calls from
        # Codex's corpus/test.jsonl, so the 62 tasks' answers come from there when they are missing here
        lacking = {uid for uid, calls in self.bfcl_calls.items() if not calls}
        calls_file = Path(args.codex_dir) / "corpus" / "test.jsonl"
        if lacking and calls_file.is_file():
            for r in read_jsonl(calls_file):
                if r.get("id") in lacking:
                    self.bfcl_calls[r["id"]] = expected_calls(r)
        self.inputs["bfcl62"] = {"path": str(args.bfcl62), "sha256": sha, "rids": len(self.bfcl_rids), "twin_ids": len(twins),
                                 "with_expected_calls": sum(1 for c in self.bfcl_calls.values() if c),
                                 "calls_file": str(calls_file) if lacking else None}
        # 5. Codex's 274-task set: every string of every task (dcb.harvest_strings) AND the whole task as canonical JSON:
        # an oracle answer stored as a nested object (a tool's parameter schema) lives in its keys, which harvest_strings
        # leaves out (trial run 2026-10-15: 28 tool-seed train rows answer one held-out task's schema question verbatim
        # and only the JSON form finds them); the 62 BFCL task ids must be in the set
        sha = pinned(Path(args.tasks), TASKS274_SHA, "combined_oct13_v1 tasks", args.no_expect)
        task_ids = []
        for t in read_jsonl(Path(args.tasks)):
            task_ids.append(str(t.get("id")))
            strings: list[str] = []
            self.dcb.harvest_strings(t, strings)
            for s in strings + [json.dumps(t, sort_keys=True, ensure_ascii=False)]:
                index.add_text("combined_oct13_v1", str(t.get("id")), s)
        if len(task_ids) != TASKS274_ROWS and not args.no_expect:
            raise Refusal(f"{args.tasks}: {len(task_ids)} tasks, not {TASKS274_ROWS}")
        if not self.bfcl_tasks <= set(task_ids):
            raise Refusal(f"{len(self.bfcl_tasks - set(task_ids))} BFCL-62 task ids are not in the 274-task set")
        self.inputs["tasks274"] = {"path": str(args.tasks), "sha256": sha, "tasks": len(task_ids)}
        self.index = index
        self.summary = index.summary()
        # 6. (1.0.2) the campaign review surface, as toolcall_selfgen.build_screen builds it for finalize
        self.review, self.review_sources = None, []
        if getattr(args, "review_surface", True):
            self.review, self.review_sources = self.dcb.build_review_index(self.dcb.parse_args(["screen"]))
        log(f"held-out screen: {len(ids):,} held-out ids, {len(self.groups):,} groups, {len(self.registry_rids):,} registry rids, "
            f"{len(task_ids)} tasks, {self.summary['ngram_count']:,} 13-grams")

    def screen(self, texts: list[tuple[str, str]]) -> list[dict[str, str]]:
        hits = []
        for role, text in texts:
            for h in self.index.hits(text):
                hits.append({**h, "role": role})
            if self.review is not None:
                for h in self.review.hits(text):
                    hits.append({**h, "source": f"review:{h['source']}", "role": role})
        return hits

    def excluded_id(self, row_id: str, group: str | None, rid: str | None) -> str | None:
        if row_id in self.ids:
            return "codex_heldout_id"
        if row_id in self.bfcl_uids or row_id in self.twins:
            return "bfcl62_uid_or_twin"
        if rid and (rid in self.registry_rids or rid in self.bfcl_rids or rid in self.twins):
            return "heldout_or_bfcl62_registry_rid"
        if group and group in self.groups:
            return "heldout_recovery_group"
        return None

    def evidence(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "index": self.summary,
                "surface": ("held-out side: every message content of every role and every tool call of the 5,391 records, every "
                            "string of the 274 tasks and each task as canonical JSON (BFCL-62 included); candidate side: every "
                            "message and tool call, a system message left out only as the one scaffold text of its pack"),
                "rules": "distill_corpus_build.ReviewIndex(ngram=13, short items off): exact normalised text (>= its exact_min "
                         "words) or any shared word 13-gram",
                "review_surface": ({"sources": self.review_sources, "index": self.review.summary(),
                                    "rules": "every ReviewIndex rule (exact, 13-gram, short item, literal), as finalize"}
                                   if self.review is not None else "not screened (--no-review-surface)"),
                "corpus_tool": self.dcb.tool_identity()}


def build_heldout(args: argparse.Namespace) -> Heldout:
    return Heldout(args)


def receipt_body(*, rank: int, source: str, corpus: Path, held: Heldout, counts: dict[str, Any], dropped: list[dict[str, Any]],
                 hits_in_corpus: int, excluded_id_hits: int, verified: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    return {"schema": SCREEN_SCHEMA, "tool": tool_identity(), "utc": utc(), "argv": sys.argv,
            "decisions_sha256": DECISIONS_SHA, "rule_id": "D2-X1-refill", "rank": rank, "source": source,
            "corpus": str(corpus), "corpus_sha256": sha256_file(corpus), "ngram": NGRAM, "hits_in_corpus": hits_in_corpus,
            "split": "train", "excluded_id_hits": excluded_id_hits, "verified_rows_only": True, "verified": verified,
            "counts": counts, "dropped_by_screen": dropped, "screen": held.evidence(), **extra}


# --------------------------------------------------------------------------
# rank 3: Codex native train packs
# --------------------------------------------------------------------------


def rank3_entry(decisions: dict[str, Any], key: str) -> dict[str, Any]:
    for item in decisions["refill_policy"]["order"]:
        if item.get("rank") == 3:
            return item["rows"][key]
    raise Refusal("the decisions file has no rank 3 refill rows")


def scaffold_system(records: list[dict[str, Any]]) -> str | None:
    """The one system text that opens every record of a pack, or a refusal (anything else is not scaffolding)."""
    texts = set()
    for r in records:
        systems = [m.get("content") for m in r.get("messages") or [] if m.get("role") == "system"]
        if len(systems) != 1:
            raise Refusal(f"{r['id']}: {len(systems)} system messages; the scaffold rule needs exactly one per row")
        texts.add(systems[0])
    if len(texts) != 1:
        raise Refusal(f"{len(texts)} distinct system texts in the pack: not one fixed scaffold")
    return texts.pop()


def check_rows(ids: Any, lengths: Any, target: Any, assistant: Any) -> list[str]:
    """d2_bfcl_native_copy.check_rows, imported (one rule for every copied assistant_only corpus)."""
    from d2_bfcl_native_copy import check_rows as bfcl_check_rows  # noqa: E402

    return bfcl_check_rows(ids, lengths, target, assistant)


def same_tensors(a: Path, b: Path) -> bool:
    """Equal tensor names, dtypes, shapes and bytes (the safetensors metadata may differ: it names the writing tool)."""
    from safetensors.numpy import load_file

    x, y = load_file(str(a)), load_file(str(b))
    return sorted(x) == sorted(y) and all(x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and x[k].tobytes() == y[k].tobytes()
                                          for k in x)


def settle_output(path: Path, tmp: Path) -> str:
    """Put a freshly written corpus in place and return the sha the receipt must name.  Corpora are immutable: an existing
    file with other TENSORS is a refusal; one with the same tensors (only the writer's metadata differs) is kept as it is."""
    new_sha = sha256_file(tmp)
    if path.exists() and sha256_file(path) != new_sha:
        if same_tensors(path, tmp):
            tmp.unlink()
            log(f"{path} already holds these exact tensors (sha {sha256_file(path)[:16]}, written by another tool version): kept")
            return sha256_file(path)
        tmp.unlink(missing_ok=True)
        raise Refusal(f"{path} exists with other tensors ({sha256_file(path)[:16]}); corpora are immutable -- build into a new --out")
    os.replace(tmp, path)
    return new_sha


def cmd_codex_native(args: argparse.Namespace) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    spec = PACKS[args.pack]
    held = build_heldout(args)
    want = rank3_entry(held.decisions, spec["key"])
    manifest_path = Path(args.codex_dir) / spec["dir"] / "manifest.json"
    manifest_sha = pinned(manifest_path, want["manifest_sha256"], f"{spec['dir']} manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    src = Path(manifest["shard"])
    corpus_sha = sha256_file(src)
    problems = []
    if manifest.get("sha256") != corpus_sha:
        problems.append(f"manifest sha256 {str(manifest.get('sha256'))[:16]} does not name {src} ({corpus_sha[:16]})")
    if manifest.get("split") != "train" or manifest.get("quarantined_count") != 0 or manifest.get("cross_split_groups"):
        problems.append(f"not an admitted train pack: split {manifest.get('split')!r}, quarantined {manifest.get('quarantined_count')}, "
                        f"cross-split groups {len(manifest.get('cross_split_groups') or [])}")
    if int(manifest["rows"]) != int(want["rows"]) or int(manifest["supervised_tokens"]) != int(want["supervised_tokens"]):
        problems.append(f"{manifest['rows']} rows / {manifest['supervised_tokens']} supervised; the decisions file says "
                        f"{want['rows']} / {want['supervised_tokens']}")
    if problems:
        raise Refusal("; ".join(problems))
    rec_rel, rec_want = spec["records"]
    rec_path = Path(args.codex_dir) / rec_rel
    rec_sha = pinned(rec_path, rec_want, f"{args.pack} records", args.no_expect)
    records = {r["id"]: r for r in read_jsonl(rec_path)}
    receipts = manifest["row_receipts"]
    joined = [records.get(r["row_id"]) for r in receipts]
    if any(r is None for r in joined):
        raise Refusal(f"{sum(r is None for r in joined)} row receipts do not join a record of {rec_path}")
    # eligibility 1: train split, no excluded id / group / rid, not already in D2 -- a hit is a refusal, not a drop
    in_d2 = {row["row_id"] for row in (load_json(Path(args.bfcl_native_rows)) or {}).get("rows", [])}
    excluded, already = [], []
    for receipt, record in zip(receipts, joined):
        p = record.get("provenance") or {}
        if p.get("split") != "train":
            excluded.append((receipt["row_id"], f"record split {p.get('split')!r}"))
            continue
        why = held.excluded_id(receipt["row_id"], p.get("recovery_group_id") or p.get("group_id"),
                               (p.get("registry_origin") or {}).get("rid"))
        if why:
            excluded.append((receipt["row_id"], why))
        if receipt["row_id"] in in_d2 or (p.get("registry_origin") or {}).get("rid"):
            already.append(receipt["row_id"])  # a registry record is already a D2 registry document; BFCL-native is C2
    if excluded:
        raise Refusal(f"{len(excluded)} rows of a train pack are excluded ids / groups (first {excluded[:3]}): the split is not "
                      "what the decisions file says; nothing written")
    if already:
        raise Refusal(f"{len(already)} rows are already D2 inputs (first {already[:3]})")
    # twin rule (bfcl62_holdout.same_answer) against the 62 tasks' expected calls
    from bfcl62_holdout import expected_calls, same_answer  # noqa: E402

    twins = []
    with_calls = 0
    known = sum(1 for c in held.bfcl_calls.values() if c)
    if any(expected_calls(r) for r in joined) and known != len(held.bfcl_uids):
        raise Refusal(f"rows carry expected calls but only {known} of the {len(held.bfcl_uids)} BFCL-62 answers are known: "
                      "the twin rule cannot be applied")
    for receipt, record in zip(receipts, joined):
        mine = expected_calls(record)
        if mine:
            with_calls += 1
            tasks = sorted(u for u, calls in held.bfcl_calls.items() if same_answer(calls, mine))
            if tasks:
                twins.append((receipt["row_id"], tasks))
    if twins:
        raise Refusal(f"{len(twins)} rows are BFCL-62 task twins (first {twins[:2]}); nothing written")
    # eligibility 2: the 13-gram screen; a hit drops the row
    system = scaffold_system(joined) if spec["system_scaffold"] else None
    keep, dropped = [], []
    for i, (receipt, record) in enumerate(zip(receipts, joined)):
        hits = held.screen(message_texts(record, skip_system=system))
        if hits:
            dropped.append({"source_row": i, "row_id": receipt["row_id"], "route": receipt.get("route"),
                            "family": (record.get("provenance") or {}).get("template_family"), "hits": hits[:4]})
        else:
            keep.append(i)
    log(f"{args.pack}: {len(receipts)} train rows; screen drops {len(dropped)}; keeps {len(keep)}")
    if not keep:
        raise Refusal("every row hit the held-out screen")
    index = torch.tensor(list(range(len(receipts))), dtype=torch.long)
    with safe_open(str(src), framework="pt", device="cpu") as handle:
        ids_all = handle.get_tensor("input_ids").index_select(0, index)
        lengths_all = handle.get_tensor("lengths").index_select(0, index).to(torch.int64)
        target_all = handle.get_tensor("target_mask").index_select(0, index).to(torch.bool)
        assistant_all = handle.get_tensor("assistant_mask").index_select(0, index).to(torch.bool)
    selection = {"rows": len(receipts), "tokens": int(lengths_all.sum()), "supervised": int(target_all.sum())}
    if not args.no_expect and (selection["tokens"] != int(manifest["tokens"]) or selection["supervised"] != int(want["supervised_tokens"])):
        raise Refusal(f"the pack reads {selection}, its manifest says tokens {manifest['tokens']} / the decisions file "
                      f"supervised {want['supervised_tokens']}")
    k = torch.tensor(keep, dtype=torch.long)
    ids, lengths = ids_all.index_select(0, k), lengths_all.index_select(0, k)
    target, assistant = target_all.index_select(0, k), assistant_all.index_select(0, k)
    width = int(lengths.max())
    if int(target[:, width:].sum()) + int(assistant[:, width:].sum()):
        raise Refusal(f"mask positions past the longest kept row ({width})")
    ids, target, assistant = ids[:, :width].contiguous(), target[:, :width].contiguous(), assistant[:, :width].contiguous()
    problems = check_rows(ids, lengths, target, assistant)
    if problems:
        raise Refusal("; ".join(problems))
    counts = {"selection": selection, "rows": len(keep), "tokens": int(lengths.sum()), "supervised": int(target.sum()),
              "width": width, "dropped_rows": len(dropped),
              "dropped_supervised": selection["supervised"] - int(target.sum()),
              "dropped_by_family": dict(Counter(d["family"] for d in dropped)),
              "rows_with_expected_calls": with_calls, "routes": dict(Counter(receipts[i].get("route") for i in keep))}
    for key, value in counts.items():
        log(f"COUNT {key} {value}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / spec["out"]
    tmp = out_dir / f".{spec['out']}.tmp-{os.getpid()}"
    metadata = {"schema": SCREEN_SCHEMA.replace("screen", "corpus"), "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
                "source_corpus_sha256": corpus_sha, "source_manifest_sha256": manifest_sha, "template_sha256": str(manifest.get("template_sha256")),
                "pack": spec["dir"], "dropped_by_screen": str(len(dropped)),
                "mask_convention": "target_mask[r, t] = token t is an assistant target (target-indexed)"}
    save_file({"input_ids": ids.to(torch.int32).contiguous(), "lengths": lengths.contiguous(), "target_mask": target,
               "assistant_mask": assistant}, str(tmp), metadata={METADATA_KEY: canonical_json(metadata)})
    new_sha = settle_output(out, tmp)
    # the kept rows screened again: hits_in_corpus is a measurement, not an assumption
    again = sum(1 for i in keep if held.screen(message_texts(joined[i], skip_system=system)))
    rows = [{"out_row": j, "source_row": i, "row_id": receipts[i]["row_id"], "route": receipts[i].get("route"),
             "rendered_sha256": receipts[i].get("rendered_sha256"), "length": int(lengths[j]), "targets": int(target[j].sum())}
            for j, i in enumerate(keep)]
    sidecar = {"schema": "omni-d2x1-refill-rows-v1", "tool": tool_identity(), "pack": spec["dir"],
               "source": {"corpus": str(src), "corpus_sha256": corpus_sha, "manifest": str(manifest_path), "manifest_sha256": manifest_sha,
                          "records": str(rec_path), "records_sha256": rec_sha},
               "output": {"path": str(out), "sha256": new_sha, "width": width}, "counts": counts, "rows": rows}
    write_json_atomic(out.with_suffix(".rows.json"), sidecar)
    verified = {"rule": "Codex native train rows that passed Codex's admission", "admission": manifest.get("admission"),
                "split": manifest.get("split"), "quarantined_count": manifest.get("quarantined_count"),
                "cross_split_groups": manifest.get("cross_split_groups"), "source_verified_rows":
                sum(1 for i in keep if (joined[i].get("provenance") or {}).get("source_verified") is True)}
    extra = {"system_scaffold": {"left_out_of_candidate_screen": system is not None,
                                 "sha256": text_sha256(system) if system is not None else None,
                                 "held_out_rows_opening_with_it": sum(1 for r in held.records.values()
                                                                      if system is not None and any(m.get("role") == "system" and m.get("content") == system for m in r.get("messages") or []))},
             "excluded_id_rule": "row_id not a codex_heldout_id / BFCL-62 uid / twin; group not a held-out recovery group; "
                                 "no registry origin (not already a D2 registry document); not a BFCL-native C2 row",
             "twin_rule": f"bfcl62_holdout.same_answer against the 62 tasks' expected calls: {with_calls} rows carry expected calls, 0 twins"}
    body = receipt_body(rank=3, source=f"refill-codex-{args.pack}-train", corpus=out, held=held, counts=counts, dropped=dropped,
                        hits_in_corpus=again, excluded_id_hits=0, verified=verified, extra=extra)
    sha = write_json_atomic(screen_path(out), body)
    log(f"WROTE {out} sha256 {new_sha}")
    log(f"WROTE {screen_path(out)} sha256 {sha} (hits_in_corpus {again})")
    return 0 if again == 0 else 2


# --------------------------------------------------------------------------
# rank 2: self-gen rows not already in D2
# --------------------------------------------------------------------------


def import_sampler(tools: Path) -> Any:
    sys.path.insert(0, str(tools))
    import selfgen_sampler as sg  # noqa: E402

    return sg


def sealed_run_rows(sg: Any, run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A finalized selfgen_sampler run's manifest and its rows (ids, labels), every shard re-verified."""
    manifest = sg.read_json(run / "manifest.json")
    if not sg.check_seal(manifest, "manifest_sha256"):
        raise Refusal(f"{run}/manifest.json does not match its own seal")
    if sg.verify_run(run) != 0:
        raise Refusal(f"{run}: selfgen_sampler verify fails")
    rows = []
    for name in manifest["shard_read_order"]:
        loaded = sg.read_shard(run / name)
        if sg.check_shard_structure(loaded):
            raise Refusal(f"{run}/{name} fails its structural check")
        for row, sample in zip(loaded["record"]["rows"], map(sg.relabel, loaded["samples"])):
            ids = list(loaded["tokens"][row["offset"]: row["offset"] + row["prompt_len"] + row["response_len"]])
            rows.append({"index": row["index"], "rid": row["rid"], "prompt_len": row["prompt_len"], "response_len": row["response_len"],
                         "ids": ids, "eog_terminated": bool(sample["eog_terminated"]), "end": sample["end"], "text": sample.get("text") or "",
                         "run_manifest_sha256": manifest["manifest_sha256"]})
    return manifest, rows


def cmd_selfgen(args: argparse.Namespace) -> int:
    import numpy as np
    from safetensors.numpy import save_file

    sg = import_sampler(Path(args.tools))
    held = build_heldout(args)
    d2 = load_json(Path(args.d2_rows))
    if not d2 or d2.get("source_manifest_sha256") != SELFGEN_D2_MANIFEST and not args.no_expect:
        raise Refusal(f"{args.d2_rows}: not the rows file of the self-gen run D2 reads ({SELFGEN_D2_MANIFEST[:16]})")
    in_d2 = set(d2["row_ids"])
    runs, rows, seen = [], [], set()
    counts: Counter = Counter()
    for run in map(Path, args.run):
        manifest, run_rows = sealed_run_rows(sg, run)
        prompts_path = Path(manifest["prompts"]["path"])
        if sha256_file(prompts_path) != manifest["prompts"]["sha256"]:
            raise Refusal(f"{prompts_path} no longer matches run {run}")
        prompts = list(read_jsonl(prompts_path))
        runs.append({"run": str(run), "manifest_sha256": manifest["manifest_sha256"], "rows": len(run_rows),
                     "prompts": str(prompts_path), "prompts_sha256": manifest["prompts"]["sha256"]})
        for row in run_rows:
            counts["rows_read"] += 1
            if row["rid"] in in_d2:
                counts["already_in_d2"] += 1
                continue
            if row["rid"] in seen:
                counts["duplicate_rid_across_runs"] += 1
                continue
            if not row["eog_terminated"]:
                counts["not_eog_terminated"] += 1
                continue
            why = held.excluded_id("", None, row["rid"])
            if why:
                counts[f"excluded:{why}"] += 1
                continue
            prompt = prompts[row["index"]]
            if prompt.get("rid") != row["rid"]:
                raise Refusal(f"{run}: row {row['index']} rid {row['rid']} is not prompt {row['index']}'s rid {prompt.get('rid')}")
            texts = [(m.get("role", "?"), str(m.get("content") or "")) for m in prompt.get("messages") or []] + [("assistant", row["text"])]
            hits = held.screen(texts)
            if hits:
                counts["dropped_by_screen"] += 1
                row["hits"] = hits[:4]
                continue
            row["texts"] = texts
            seen.add(row["rid"])
            rows.append(row)
    counts["eligible_rows"] = len(rows)
    log(f"selfgen: {dict(counts)}")
    out_dir = Path(args.out)
    out = out_dir / SELFGEN_OUT
    if not rows:
        receipt = {"schema": "omni-d2x1-refill-selfgen-none-v1", "tool": tool_identity(), "utc": utc(), "runs": runs,
                   "counts": dict(counts), "d2_rows": {"path": str(args.d2_rows), "sha256": sha256_file(Path(args.d2_rows)), "rids": len(in_d2)},
                   "why": "no eligible self-gen row: every row of the named sealed run(s) is already a D2 input, truncated or "
                          "excluded; rank 2 needs NEW sampling over prompts outside the 1,168 rescored rows (decisions "
                          "refill_policy rank 2 'how')"}
        sha = write_json_atomic(out_dir / "receipts" / f"selfgen-refill-none-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json", receipt)
        log(f"NOTHING: no eligible self-gen row ({dict(counts)}); receipt sha256 {sha}; no corpus written")
        return EXIT_NOTHING
    width = max(len(r["ids"]) for r in rows)
    n = len(rows)
    input_ids = np.full((n, width), PAD_ID, dtype=np.int32)
    lengths = np.zeros((n,), dtype=np.int64)
    store = np.zeros((n, width), dtype=np.uint8)
    target = np.zeros((n, width), dtype=np.uint8)
    for i, r in enumerate(rows):
        L = len(r["ids"])
        input_ids[i, :L] = np.asarray(r["ids"], dtype=np.int64).astype(np.int32)
        lengths[i] = L
        store[i, r["prompt_len"] - 1: L - 1] = 1
        target[i, r["prompt_len"]: L] = 1
        if r["ids"][-1] not in EOG_IDS or int(input_ids[i, :L].max()) >= TEXT_VOCAB_ROWS:
            raise Refusal(f"row {r['rid']}: does not end at an EOG id or has an id outside the text vocab")
    eog = np.ones((n,), dtype=np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".{SELFGEN_OUT}.tmp-{os.getpid()}"
    metadata = {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "source_manifests": "+".join(r["manifest_sha256"] for r in runs),
                "target_rule": "target_mask[n, t] = 1 for t in [prompt_len, length); store_mask = target_mask shifted by one"}
    save_file({"input_ids": input_ids, "lengths": lengths, "store_mask": store, "target_mask": target, "eog_terminated": eog},
              str(tmp), metadata={METADATA_KEY: canonical_json(metadata)})
    new_sha = settle_output(out, tmp)
    again = sum(1 for r in rows if held.screen(r["texts"]))
    c = {"rows": n, "tokens": int(lengths.sum()), "supervised": int(target.sum()), "width": width, **dict(counts)}
    write_json_atomic(out.with_suffix(".rows.json"), {
        "schema": "omni-d2x1-refill-rows-v1", "tool": tool_identity(), "runs": runs, "output": {"path": str(out), "sha256": new_sha},
        "counts": c, "row_ids": [r["rid"] for r in rows], "row_index": [r["index"] for r in rows],
        "row_run": [r["run_manifest_sha256"] for r in rows]})
    body = receipt_body(rank=2, source="refill-selfgen", corpus=out, held=held, counts=c, dropped=[], hits_in_corpus=again,
                        excluded_id_hits=0, verified={"rule": "self-gen rows ending at a parent EOG id from sealed runs whose "
                                                      "shards verify (selfgen_sampler verify)", "runs": runs},
                        extra={"not_in_d2": f"rid not among the {len(in_d2)} rows of {args.d2_rows}"})
    sha = write_json_atomic(screen_path(out), body)
    log(f"WROTE {out} sha256 {new_sha}; receipt {screen_path(out)} sha256 {sha} (hits_in_corpus {again})")
    return 0 if again == 0 else 2


# --------------------------------------------------------------------------
# state / totals
# --------------------------------------------------------------------------


def import_plan(tools: Path) -> Any:
    sys.path.insert(0, str(tools))
    import d2_plan  # noqa: E402

    return d2_plan


def candidates(plan: Any) -> list[dict[str, Any]]:
    return [dict(c) for c in plan.REFILL_CANDIDATES]


def corpus_state(plan: Any, refill_dir: Path, cache_root: Path, cand: dict[str, Any]) -> dict[str, Any]:
    corpus = Path(refill_dir) / cand["file"]
    out = {"rank": cand["rank"], "name": cand["name"], "file": cand["file"], "corpus": str(corpus), "exists": corpus.is_file()}
    if not corpus.is_file():
        return {**out, "eligible": False, "why": "corpus file does not exist yet", "cache_sealed": False}
    sha = sha256_file(corpus)
    receipt = load_json(screen_path(corpus))
    problems = receipt_problems(receipt, sha)
    via = "d2_refill_build.receipt_problems"
    if hasattr(plan, "refill_screen_state"):  # d2_plan >= 1.4.0: its own decision is the one that counts
        st = plan.refill_screen_state(None, corpus, sha)
        via = f"d2_plan {plan.TOOL_VERSION} refill_screen_state"
        if bool(st["ok"]) != (not problems):
            raise Refusal(f"d2_plan and d2_refill_build disagree on {screen_path(corpus)}: {st['why']} vs {problems}")
    m = load_json(Path(cache_root) / sha / "manifest.json") or {}
    sealed = bool(m.get("all_finite")) and bool(m.get("all_structural_ok")) and bool((m.get("teacher") or {}).get("equal_before_and_after_every_session"))
    return {**out, "corpus_sha256": sha, "eligible": not problems, "why": "; ".join(problems) or "eligible", "checked_by": via,
            "receipt_sha256": sha256_file(screen_path(corpus)) if receipt is not None else None,
            "supervised": (receipt or {}).get("counts", {}).get("supervised"), "tokens": (receipt or {}).get("counts", {}).get("tokens"),
            "width": (receipt or {}).get("counts", {}).get("width"),
            "cache_dir": str(Path(cache_root) / sha), "cache_sealed": sealed}


def cmd_state(args: argparse.Namespace) -> int:
    plan = import_plan(Path(args.tools))
    rows = [corpus_state(plan, Path(args.out), Path(args.cache_root), c) for c in candidates(plan)]
    print("D2REFILL state " + json.dumps({"refill_dir": args.out, "d2_plan": plan.TOOL_VERSION, "candidates": rows}, sort_keys=True), flush=True)
    return 0


def toolcall_rank1(args: argparse.Namespace) -> dict[str, Any]:
    """Rank 1 (C1): the supervised tokens of the tool-call rows beyond A2 2(a)'s 1.5M.  Measured from the export's sidecar
    once it exists (source_manifests[1:]); before that, run 2's live progress and run 1's supervised share."""
    tcdir = Path(args.toolcall_dir)
    sidecar = load_json(tcdir / "rescore" / "toolcall-q38-2026-10-14.rows.json")
    out: dict[str, Any] = {"source": "tool-call self-gen beyond A2 2(a) (C1 runs 2..)"}
    if sidecar and sidecar.get("source_manifests"):
        sm = sidecar["source_manifests"]
        out.update(state="exported", runs=[{k: s.get(k) for k in ("manifest_sha256", "row_range", "tokens", "supervised_tokens")} for s in sm],
                   supervised_beyond_a2_2a=sum(int(s["supervised_tokens"]) for s in sm[1:]),
                   d2_plan_formula=("d2_plan refill_plan counts sup x (real - 1,500,000) / real over the whole C1 corpus; "
                                    "the run-2 rows' own count is this value"))
        return out
    run1 = load_json(tcdir / "run-q38-q4km-2026-10-14" / "manifest.json") or {}
    prog = load_json(tcdir / "run-q38-q4km-2026-10-15-cont" / "progress.json") or {}
    share = (int(run1["supervised_tokens"]) / int(run1["tokens"])) if run1.get("tokens") else None
    target2 = 3_000_000 - int(run1.get("tokens") or 0)
    done = None
    for key in ("tokens_accepted", "accepted_tokens", "tokens_done", "tokens"):
        if isinstance(prog.get(key), int):
            done = int(prog[key])
            break
    out.update(state="sampling (run 2)" if prog else "unknown", run1={"tokens": run1.get("tokens"), "supervised": run1.get("supervised_tokens"),
                                                                       "manifest_sha256": run1.get("manifest_sha256")},
               supervised_share_run1=round(share, 6) if share else None, run2_token_target=target2, run2_tokens_so_far=done,
               run2_progress_utc=prog.get("updated_utc"),
               supervised_beyond_a2_2a=int(round(share * target2)) if share else 0,
               basis="projection: run 2 reaches its token target at run 1's supervised share (assistant positions only)")
    return out


def clean_form_target(plan: Any, args: argparse.Namespace, refill: dict[str, Any], f_v1: float, draw: int,
                      manifest: dict[str, Any]) -> dict[str, Any] | None:
    """d2_plan refill_plan's target if D2 takes lane d2clean's clean registry (text_clean_v1): KD of the v1 draw minus KD
    of the clean draw.  None while no clean version exists; d2_plan decides the form when it writes the config."""
    clean_dir = Path(args.text_manifest).parent.parent / "text_clean_v1"
    man = load_json(clean_dir / "manifest.json")
    kd_v1 = refill.get("kd_positions_v1")
    if not man or not isinstance(kd_v1, int):
        return None
    kd_clean = int((man.get("counts") or {}).get("kd_positions_clean_total") or 0)
    excl = load_json(clean_dir / "clean_exclude_rows.json") or {}
    rows = sum(int(s["rows"][1]) - int(s["rows"][0]) - len((excl.get("corpora", {}).get(s["sha256"]) or {}).get("rows", []))
               for s in manifest.get("shards") or [])
    rows += sum(int(s["rows"][1]) - int(s["rows"][0]) for s in man.get("shards") or [])
    f_clean = min(1.0, draw / max(rows * plan.SEQ, 1))
    return {"target_tokens": max(0, int(round(kd_v1 * f_v1 - kd_clean * f_clean))), "kd_positions_v1": kd_v1,
            "kd_positions_clean_total": kd_clean, "registry_positions": rows * plan.SEQ, "f_clean": round(f_clean, 6),
            "clean_manifest_sha256": man.get("manifest_sha256"),
            "note": "used only if the clean registry is built, verified and every re-pack cache is sealed when the config is written"}


def cmd_totals(args: argparse.Namespace) -> int:
    plan = import_plan(Path(args.tools))
    v2 = load_json(Path(args.masks_v2) / "build_receipt.json") or {}
    refill = v2.get("refill") or {}
    removed, after_v1 = refill.get("heldout_trained_positions_removed"), refill.get("positions_after_v1_exclusion")
    if not isinstance(removed, int) or not isinstance(after_v1, int):
        raise Refusal(f"{args.masks_v2}/build_receipt.json has no refill block")
    draw = plan.REGISTRY_DRAW + plan.ONPOLICY_FALLBACK
    f = min(1.0, draw / after_v1)
    target = int(round(removed * f))  # d2_plan refill_plan's span-form target, same inputs, same formula
    rank1 = toolcall_rank1(args)
    ranks: list[dict[str, Any]] = [{"rank": 1, **rank1, "supervised": int(rank1.get("supervised_beyond_a2_2a") or 0)}]
    for cand in candidates(plan):
        st = corpus_state(plan, Path(args.out), Path(args.cache_root), cand)
        sup = int(st.get("supervised") or 0) if st["eligible"] else 0
        ranks.append({"rank": cand["rank"], "source": cand["name"], "supervised": sup, "corpus_sha256": st.get("corpus_sha256"),
                      "eligible": st["eligible"], "why": st["why"], "cache_sealed": st["cache_sealed"],
                      "counts_only_when": "its Q38 cache is sealed when the short data config is written (d2_plan)"})
    excl = load_json(Path(args.masks_v2) / "exclude_rows.json") or {}
    manifest = load_json(Path(args.text_manifest)) or {}
    rows = sum(int(s["rows"][1]) - int(s["rows"][0]) - len((excl.get("corpora", {}).get(s["sha256"]) or {}).get("rows", []))
               for s in manifest.get("shards") or [])
    undrawn = max(0, rows * plan.SEQ - draw)
    ranks.append({"rank": 4, "source": "undrawn registry positions (span-mask form)", "supervised": undrawn,
                  "basis": f"{rows:,} registry rows after d2_masks_v2/exclude_rows x {plan.SEQ} - the {draw:,} draw"})
    expected = sum(r["supervised"] for r in ranks)
    rank2 = next(r for r in ranks if r["rank"] == 2)
    clean_form = clean_form_target(plan, args, refill, f, draw, manifest)
    body = {"schema": TOTALS_SCHEMA, "tool": tool_identity(), "utc": utc(), "decisions_sha256": DECISIONS_SHA,
            "unit": "supervised (KD-carrying) tokens, the unit of D2_DECISIONS refill_policy.target",
            "target_tokens": target, "target_basis": f"{removed:,} x min(1, {draw:,} / {after_v1:,}) = {f:.4f} (d2_plan refill_plan, "
                                                      "span-mask form; the decisions file's 'about 2.96M' assumed an 18.85M draw)",
            "ranks": ranks, "expected_supervised_total": expected, "expected_shortfall": max(0, target - expected),
            "target_met": expected >= target, "clean_form": clean_form,
            "rank2_next": None if rank2["eligible"] else {
                "why": "the sealed self-gen run (fe1f0784, cutoff 1,168) is wholly a D2 input: 0 rows are left for the refill",
                "how": "NEW selfgen_sampler sampling over prompts outside the 1,168 rescored rows (decisions rank 2 'how'): the "
                       "1,364 unsampled prompts of distill/corpus/selfgen/prompts.jsonl (indices 1168..2531), screened, then this "
                       "tool's `selfgen --run <new run>` and its Q38 cache",
                "estimate": "about 1.18M response tokens at run 1's 864 tokens/row; about 8 h on GPU1+GPU2 at run 1's 41 tok/s; "
                            "needs the parent Q4_K_M (its scratch copies are removed after use) and P40 time the d2clean caches "
                            "and the D1 bank v1 review also want"},
            "d2_never_waits": "D2 uses only the refill sources whose caches are sealed when the short data config is written; "
                              "the remainder is the recorded shortfall (decisions refill_policy.timing)",
            "inputs": {"masks_v2_build_receipt": sha256_file(Path(args.masks_v2) / "build_receipt.json"),
                       "exclude_rows": sha256_file(Path(args.masks_v2) / "exclude_rows.json") if (Path(args.masks_v2) / "exclude_rows.json").is_file() else None,
                       "text_manifest": sha256_file(Path(args.text_manifest)) if Path(args.text_manifest).is_file() else None,
                       "d2_plan": {"version": plan.TOOL_VERSION, "sha256": sha256_file(Path(plan.__file__))}}}
    out = Path(args.out) / "REFILL_TOTALS.json"
    sha = write_json_atomic(out, body)
    for r in ranks:
        log(f"RANK {r['rank']} {r.get('source')}: supervised {r['supervised']:,}")
    if clean_form:
        log(f"CLEAN-FORM TARGET {clean_form['target_tokens']:,} (if D2 takes text_clean_v1); shortfall there "
            f"{max(0, clean_form['target_tokens'] - expected):,}")
    log(f"TARGET {target:,}; EXPECTED {expected:,}; SHORTFALL {max(0, target - expected):,}; wrote {out} sha256 {sha}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser, *, screen: bool) -> None:
        sp.add_argument("--tools", default=str(HERE))
        sp.add_argument("--out", default=str(REFILL_DIR))
        if screen:
            sp.add_argument("--codex-dir", default=str(CODEX))
            sp.add_argument("--decisions", default=str(DECISIONS))
            sp.add_argument("--bfcl62", default=str(BFCL62))
            sp.add_argument("--tasks", default=str(TASKS274))
            sp.add_argument("--heldout-rids", default=str(HELDOUT_RIDS))
            sp.add_argument("--no-expect", action="store_true", help="tests only: skip the file pins")
            sp.add_argument("--no-review-surface", dest="review_surface", action="store_false",
                            help="tests only: leave the campaign review surface out of the screen")

    c = sub.add_parser("codex-native")
    common(c, screen=True)
    c.add_argument("--pack", choices=sorted(PACKS), required=True)
    c.add_argument("--bfcl-native-rows", default=str(BFCL_NATIVE_ROWS))
    s = sub.add_parser("selfgen")
    common(s, screen=True)
    s.add_argument("--run", action="append", required=True, help="a sealed selfgen_sampler run directory (repeatable)")
    s.add_argument("--d2-rows", default=str(SELFGEN_D2_ROWS), help="the rows file of the self-gen corpus D2 already reads")
    for name in ("state", "totals"):
        x = sub.add_parser(name)
        common(x, screen=False)
        x.add_argument("--cache-root", default=str(LEX / "teacher_cache" / "qtext" / "q38-27bu"))
        x.add_argument("--toolcall-dir", default=str(OC / "distill" / "toolcall"))
        x.add_argument("--masks-v2", default=str(OC / "distill" / "corpus" / "d2_masks_v2"))
        x.add_argument("--text-manifest", default=str(OC / "distill" / "corpus" / "text" / "manifest.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "codex-native":
            return cmd_codex_native(args)
        if args.command == "selfgen":
            return cmd_selfgen(args)
        if args.command == "state":
            return cmd_state(args)
        if args.command == "totals":
            return cmd_totals(args)
    except Refusal as refusal:
        print(f"REFUSED: {refusal.message}", file=sys.stderr, flush=True)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
