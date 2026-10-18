#!/usr/bin/env python3
"""d2_bfcl_native_copy.py -- D2-M2: copy Codex's 506 BFCL-native train rows into a Claude-owned D2 corpus file.

Card D2-M2 of STAGE_D_RECONCILED_PLAN_2026-10-14 (section 3.3 item 2, amendment A2 item 2b).  Codex re-rendered the
BFCL v4 records with structured `tool_calls` through the PARENT's own chat template (`template_sha256` 924d3fcc, the
canonical-JSON digest of the parent chat_template.jinja) and packed them with an assistant target mask.  Those rows sit
inside Codex's 19,463-row `native-text-v3-train/assistant_text.safetensors` (corpus sha 8d815f32...), which is
READ-ONLY for Claude.  This tool selects the BFCL rows by joining the manifest's `row_receipts[i].row_id` to Codex's
corpus records (`corpus/{train,validation,test}.jsonl`, `provenance.registry_origin.dataset_id`) and writes:

  <out>/bfcl-native-506.safetensors   input_ids int32 [506, W], lengths int64 [506], target_mask bool [506, W],
                                      assistant_mask bool [506, W]; W = the longest selected row (columns past W hold
                                      only Codex's padding; nothing is re-tokenised, re-rendered or re-masked)
  <out>/rows.json                     the source corpus / manifest shas, the selected source row indices and row ids,
                                      per-row lengths and supervised counts, and the output file's sha256

`target_mask` is TARGET-indexed (token t is a target), the convention qtext_teacher_cache build-text shifts into its
store mask and streamed_kd_trainer >= 1.3.4 reads for `supervision: assistant_only`.

Refused (exit 2, nothing written): a source file whose sha256 is not the pinned corpus sha; a selection whose
counts differ from the X05 reading (506 rows / 420,725 tokens / 49,213 targets / longest 2,109) unless --no-expect;
a selected row whose target mask marks column 0, padding, or nothing; target_mask != assistant_mask on a selected
row; an id at or past 248,320; any selected row whose recovery group is a group of Codex's BFCL held-out tasks
(--eval-tasks), i.e. a BFCL-62 source must never become D2 data.

Usage (server, CPU):
  python d2_bfcl_native_copy.py --codex-dir $OC/distill/codex-preparation-01a1326f-20261013 \\
      --eval-tasks <combined_oct13_v1 tasks.jsonl> --out $OC/distill/corpus/bfcl_native_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TOOL_NAME = "d2_bfcl_native_copy"
TOOL_VERSION = "1.0.0"  # 2026-10-14: first version (lane d2prep, card D2-M2)
ROWS_SCHEMA = "omni-d2-bfcl-native-rows-v1"
TEXT_VOCAB_ROWS = 248_320
BFCL_MARK = "bfcl"
BFCL_TASK_PREFIX = "bfcl-heldout:"
EXPECT = {"corpus_sha256": "8d815f32b8d4dc82890dd5b9e2e4bcd99dbe9d2d337604f3a5f6f787f49ff107",
          "rows": 506, "tokens": 420725, "targets": 49213, "max_length": 2109}
OUT_NAME = "bfcl-native-506.safetensors"


class Refusal(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


def log(message: str) -> None:
    print(f"[{TOOL_NAME}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def select_rows(row_receipts: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> tuple[list[int], Counter]:
    """Source row indices whose record's registry dataset id names BFCL, in source order."""

    seen: Counter = Counter()
    picked = []
    for index, receipt in enumerate(row_receipts):
        record = records.get(receipt["row_id"])
        if record is None:
            seen["no_join"] += 1
            continue
        is_bfcl = BFCL_MARK in str(record["dataset_id"]).lower()
        seen[(record["split"], is_bfcl)] += 1
        if is_bfcl:
            picked.append(index)
    return picked, seen


def check_rows(ids: Any, lengths: Any, target: Any, assistant: Any) -> list[str]:
    """Functional checks on the selected rows (torch tensors)."""

    import torch

    problems = []
    width = ids.shape[1]
    cols = torch.arange(width).unsqueeze(0)
    real = cols < lengths.unsqueeze(1)
    if bool(target[:, 0].any()):
        problems.append(f"target_mask marks column 0 in {int(target[:, 0].sum())} rows")
    if bool((target & ~real).any()):
        problems.append(f"target_mask marks {int((target & ~real).sum())} padding positions")
    if bool((target.sum(dim=1) == 0).any()):
        problems.append(f"{int((target.sum(dim=1) == 0).sum())} rows have no target token")
    if not torch.equal(target, assistant):
        problems.append("target_mask != assistant_mask")
    if int(ids[real].max()) >= TEXT_VOCAB_ROWS or int(ids[real].min()) < 0:
        problems.append(f"ids outside [0, {TEXT_VOCAB_ROWS}) on real positions")
    return problems


def run(args: argparse.Namespace) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    codex = Path(args.codex_dir)
    src = codex / "native-text-v3-train" / "assistant_text.safetensors"
    manifest_path = codex / "native-text-v3-train" / "manifest.json"
    corpus_sha = sha256_file(src)
    manifest_sha = sha256_file(manifest_path)
    log(f"source {src} sha256 {corpus_sha}")
    log(f"manifest {manifest_path} sha256 {manifest_sha}")
    if not args.no_expect and corpus_sha != EXPECT["corpus_sha256"]:
        raise Refusal(f"source sha256 {corpus_sha[:16]} is not the pinned Codex corpus {EXPECT['corpus_sha256'][:16]}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != corpus_sha:
        raise Refusal(f"manifest sha256 field {str(manifest.get('sha256'))[:16]} does not name the source file {corpus_sha[:16]}")
    records: dict[str, dict[str, Any]] = {}
    corpus_files = {}
    for split in ("train", "validation", "test"):
        path = codex / "corpus" / f"{split}.jsonl"
        corpus_files[split] = {"path": str(path), "sha256": sha256_file(path)}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                p = r.get("provenance") or {}
                origin = p.get("registry_origin") or {}
                records[r["id"]] = {"split": split, "dataset_id": origin.get("dataset_id", ""), "rid": origin.get("rid"),
                                    "group": p.get("recovery_group_id") or p.get("group_id")}
    picked, seen = select_rows(manifest["row_receipts"], records)
    log(f"join {dict((str(k), v) for k, v in seen.items())}")
    heldout_groups = set()
    if args.eval_tasks:
        with open(args.eval_tasks, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    t = json.loads(line)
                    if str(t.get("id", "")).startswith(BFCL_TASK_PREFIX):
                        heldout_groups.add(t["group_id"])
    leaked = [i for i in picked if records[manifest["row_receipts"][i]["row_id"]]["group"] in heldout_groups]
    splits = Counter(records[manifest["row_receipts"][i]["row_id"]]["split"] for i in picked)
    log(f"selected {len(picked)} rows; splits {dict(splits)}; BFCL held-out groups {len(heldout_groups)}; selected rows in them {len(leaked)}")
    if leaked:
        raise Refusal(f"{len(leaked)} selected rows belong to a BFCL held-out task group (first source row {leaked[0]})")
    if set(splits) != {"train"}:
        raise Refusal(f"selected rows come from splits {dict(splits)}; only train rows may become D2 data")
    index = torch.tensor(picked, dtype=torch.long)
    with safe_open(str(src), framework="pt", device="cpu") as handle:
        ids = handle.get_tensor("input_ids").index_select(0, index)
        lengths = handle.get_tensor("lengths").index_select(0, index).to(torch.int64)
        target = handle.get_tensor("target_mask").index_select(0, index).to(torch.bool)
        assistant = handle.get_tensor("assistant_mask").index_select(0, index).to(torch.bool)
    width = int(lengths.max())
    beyond = int(target[:, width:].sum()) + int(assistant[:, width:].sum())
    if beyond:
        raise Refusal(f"{beyond} mask positions past the longest row ({width})")
    ids, target, assistant = ids[:, :width].contiguous(), target[:, :width].contiguous(), assistant[:, :width].contiguous()
    problems = check_rows(ids, lengths, target, assistant)
    if problems:
        raise Refusal("; ".join(problems))
    counts = {"rows": len(picked), "tokens": int(lengths.sum()), "targets": int(target.sum()), "max_length": width,
              "min_length": int(lengths.min()), "routes": dict(Counter(manifest["row_receipts"][i]["route"] for i in picked)),
              "datasets": dict(Counter(records[manifest["row_receipts"][i]["row_id"]]["dataset_id"] for i in picked))}
    for key, value in counts.items():
        log(f"COUNT {key} {value}")
    if not args.no_expect:
        bad = {k: (counts[k], EXPECT[k]) for k in ("rows", "tokens", "targets", "max_length") if counts[k] != EXPECT[k]}
        if bad:
            raise Refusal(f"counts differ from the X05 reading: {bad}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{OUT_NAME}.tmp-{os.getpid()}"
    metadata = {"schema": ROWS_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "source_corpus_sha256": corpus_sha,
                "source_manifest_sha256": manifest_sha, "template_sha256": str(manifest.get("template_sha256")),
                "mask_convention": "target_mask[r, t] = token t is an assistant target (target-indexed)", "route": "tool_native"}
    save_file({"input_ids": ids.to(torch.int32).contiguous(), "lengths": lengths.contiguous(), "target_mask": target,
               "assistant_mask": assistant}, str(tmp), metadata=metadata)
    os.replace(tmp, out / OUT_NAME)
    out_sha = sha256_file(out / OUT_NAME)
    rows = [{"out_row": j, "source_row": i, "row_id": manifest["row_receipts"][i]["row_id"],
             "route": manifest["row_receipts"][i]["route"], "rendered_sha256": manifest["row_receipts"][i]["rendered_sha256"],
             "length": int(lengths[j]), "targets": int(target[j].sum())} for j, i in enumerate(picked)]
    body = {"schema": ROWS_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(Path(__file__)),
            "argv": sys.argv, "source": {"corpus": str(src), "corpus_sha256": corpus_sha, "manifest": str(manifest_path),
                                         "manifest_sha256": manifest_sha, "template_sha256": manifest.get("template_sha256"),
                                         "records": corpus_files, "eval_tasks": str(args.eval_tasks) if args.eval_tasks else None,
                                         "eval_tasks_sha256": sha256_file(Path(args.eval_tasks)) if args.eval_tasks else None},
            "selection": f"row_receipts[i].row_id -> Codex corpus record; registry dataset id contains {BFCL_MARK!r}",
            "source_rows_sha256": canonical_sha256(picked), "counts": counts,
            "output": {"path": str(out / OUT_NAME), "sha256": out_sha, "width": width,
                       "tensors": {"input_ids": "int32 [rows, W]", "lengths": "int64 [rows]", "target_mask": "bool [rows, W]",
                                   "assistant_mask": "bool [rows, W]"}},
            "bfcl_heldout_groups_checked": len(heldout_groups), "rows": rows}
    rtmp = out / f".rows.json.tmp-{os.getpid()}"
    rtmp.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(rtmp, out / "rows.json")
    log(f"WROTE {out / OUT_NAME} sha256 {out_sha}")
    log(f"WROTE {out / 'rows.json'} sha256 {sha256_file(out / 'rows.json')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--codex-dir", required=True)
    p.add_argument("--eval-tasks", help="Codex combined_oct13_v1 tasks.jsonl: selected rows must avoid its BFCL groups")
    p.add_argument("--out", required=True)
    p.add_argument("--no-expect", action="store_true")
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    args = p.parse_args(argv)
    try:
        return run(args)
    except Refusal as refusal:
        print(f"REFUSED: {refusal.message}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
