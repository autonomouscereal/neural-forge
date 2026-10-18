#!/usr/bin/env python3
"""d2_plan.py -- the decisions behind queues/d2_launch.sh (Stage D2), as small testable commands.

Lane L-TRAIN (Claude, d2prep), 2026-10-14.  STAGE_D_RECONCILED_PLAN_2026-10-14 sections 3-4, prereg D2 + draft A2.
d2_launch.sh owns the waiting, the locks and the processes; this file owns what is read and written:

  inventory     every D2 input: the 20 registry Q38 caches, self-gen, the tool-call self-gen (C1) and BFCL-native
                (C2) caches, the 4 K3-row Q38 caches, the long cache, the d2_masks_v1 files (pinned shas + a passed
                doc_index check), and the K3 cache passes.  Exit 0 = every START input is sealed, 10 = waiting.
  verdict       the P2 engine verdict: $Q/p2.verdict if it exists, else derived from the P2 summaries under the
                standing GO (STAGE_D_RULINGS G5-1: >= 151 tok/s -> streamed (upside), else the split cell (planning))
                and written to $Q/p2.verdict with `by=auto`; exit 10 while no P2 cell has completed.
  data-config   the D2 short (or long) data config, deterministic from the inventory (budgets in the units the
                trainer counts; K3 rows deferred to the 20M rung behind the K3 seal marker).
  a2-state      is amendment A2 still a DRAFT (placeholders <C1>/<C2>)?  `a2-finalize` fills them from the sealed C1 /
                C2 manifests, drops DRAFT, backs the file up and re-runs check-prereg (refuses if the pin moves).
  k3-state      how many K3 passes are sealed by the B2-proven frozen tool; `k3-seal` builds each pass's xtok
                target with that frozen tool, verifies it against its Q38 cache through streamed_kd_trainer's own
                XtokTargets.verify_against, and only then writes the seal marker the D2 data config waits on.
  memory-fit    (1.2.0) the verdict's micro-batch at the data config's widest row, through the trainer's own memory
                model calibrated on the P2 cell's measured peak; halves the micro-batch (same global tokens) until it
                fits the card, and carries the knobs (FFN slice, row group, head chunk) of the P2 attempts that completed.

Every command prints one `D2PLAN <command> {...}` JSON line; nothing here starts a GPU job.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "d2_plan"
TOOL_VERSION = "1.2.0"  # 2026-10-15 (adversarial review): memory-fit -- P2 measured its cells at width 4,096, D2 pads a
# micro-batch to its widest row (self-gen 4,841) and the trainer's defaults are the knobs P2's OOM retry abandons
# 1.1.0 2026-10-14: BFCL-62 twin decision (76 excluded registry rows, K3/self-gen exclusions, 500 BFCL rows)
# 1.0.0 2026-10-14: first version (lane d2prep), never deployed; _superseded/d2_plan_1.0.0.py

EXIT_WAITING = 10
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
LEX = Path(os.environ.get("LEX", "/mnt/lexar/omni-campaign"))
SEQ = 4096
PAD_ID = 248044

#: the D2 short draw (prereg D2 data + A2): the registry's off-policy draw plus the on-policy share, which falls back
#: to the registry while no on-policy sampler exists (prereg section 3 D2 "Data"); capped at one epoch of the rows
#: left after the BFCL-62 + twin exclusion (76 rows -> 19,689,472 positions), so no registry row is repeated
REGISTRY_DRAW = 18_850_000
ONPOLICY_FALLBACK = 1_150_000
K3_DEFER_TOKENS = 20_000_000
RUNGS = (5_000_000, 12_000_000, 20_000_000)

#: d2_masks_v1 (d2_mask_build.py 1.1.0 on the pinned X04 inputs + bfcl62_rids.json 538678aa, 2026-10-14)
MASK_SHAS = {
    "exclude_rows.json": "55d9c51030f742a4f5969ae8abd736b4226727f24f57113941da51f12d85c864",
    "all_mask.json": "23b0186e5104fc34c9437e4f3051059797caa36212f9c55ec1dbc0b7aabcf598",
    "gt_mask.json": "148bbebc3ab16e6469d271b41d088aabc36430714817f9106ce6d1cf8d4dfda2",
}
#: pinned-by-content corpora (the cache directory name is the corpus sha256)
SELFGEN_CORPUS_SHA = "7b73d3d9321f134986bd90c2388802fe080725c6f16ec9705fa4a45629566f3a"
LONG_CORPUS_SHA = "918f36f8e6042d5de2e9304704ff8e8168ad34c5e7d821b785d5b253036c60de"
BFCL_EXPECT = {"rows": 500, "tokens": 415513}  # d2_bfcl_native_copy 1.1.0: 506 rows / 420,725 tokens minus 6 task twins / 5,212

STREAMED_MIN = 151.0
CELL_SPLITS = {"L4096-mb8": (8, 1), "L4096-mb4": (4, 2)}  # per card: (micro, accumulation) for 32,768 tokens
GLOBAL_TOKENS_PER_STEP = 65_536


class Refusal(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit(command: str, body: dict[str, Any]) -> None:
    print(f"D2PLAN {command} " + json.dumps(body, sort_keys=True, default=str), flush=True)


def write_json_atomic(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


class Layout:
    def __init__(self, oc: Path = OC, lex: Path = LEX) -> None:
        self.oc, self.lex = Path(oc), Path(lex)
        self.q = self.oc / "distill" / "queues"
        self.d2 = self.lex / "distill" / "d2"
        self.cache_root = self.lex / "teacher_cache" / "qtext" / "q38-27bu"
        self.text_manifest = self.oc / "distill" / "corpus" / "text" / "manifest.json"
        self.masks = self.oc / "distill" / "corpus" / "d2_masks_v1"
        self.toolcall_corpus = self.oc / "distill" / "toolcall" / "rescore" / "toolcall-q38-2026-10-14.safetensors"
        self.bfcl_corpus = self.oc / "distill" / "corpus" / "bfcl_native_v1" / "bfcl-native-500.safetensors"
        self.k3_corpus = self.oc / "distill" / "corpus" / "k3"
        self.k3_frozen_tool = self.q / "frozen" / "k3_teacher_cache-gpu6.py"
        self.k3_marker = self.d2 / "k3.sealed"
        self.prereg = self.oc / "distill" / "PREREGISTRATION_OMNI_STAGE_D_v1.md"
        self.p2_verdict = self.q / "p2.verdict"
        self.p2_summaries = self.oc / "distill" / "tps" / "p2-v133"

    def cache_dir(self, corpus_sha: str) -> Path:
        return self.cache_root / corpus_sha

    def k3_out(self) -> Path:
        manifest = json.loads((self.k3_corpus / "manifest.json").read_text(encoding="utf-8"))
        return self.lex / "teacher_cache" / "k3" / str(manifest["manifest_sha256"])[:16]


def cache_state(cache: Path) -> dict[str, Any]:
    """A Q38 cache directory: sealed (manifest present, finite, structural, teacher steady) or not."""

    manifest = cache / "manifest.json"
    if not manifest.is_file():
        return {"dir": str(cache), "sealed": False, "reason": "no manifest.json"}
    m = json.loads(manifest.read_text(encoding="utf-8"))
    ok = bool(m.get("all_finite")) and bool(m.get("all_structural_ok")) and bool((m.get("teacher") or {}).get("equal_before_and_after_every_session"))
    corpus = m.get("corpus") or {}
    return {"dir": str(cache), "sealed": ok, "manifest_sha256": sha256_file(manifest), "corpus_sha256": corpus.get("sha256"),
            "corpus_path": corpus.get("path"), "tokens": corpus.get("tokens"), "rows": corpus.get("rows"),
            "row_length": corpus.get("row_length"), "reason": None if ok else "manifest not finite / structural / teacher-steady"}


def corpus_cache(layout: Layout, corpus: Path) -> dict[str, Any]:
    if not corpus.is_file():
        return {"corpus": str(corpus), "sealed": False, "reason": "corpus file does not exist yet"}
    sha = sha256_file(corpus)
    return {"corpus": str(corpus), **cache_state(layout.cache_dir(sha)), "corpus_sha256": sha}


def k3_state(layout: Layout) -> dict[str, Any]:
    """K3 passes sealed by the B2-proven frozen tool, xtok targets built, and the seal marker."""

    tool_sha = sha256_file(layout.k3_frozen_tool) if layout.k3_frozen_tool.is_file() else None
    out = layout.k3_out()
    passes = []
    for k in range(4):
        pd = out / f"pass-{k}-of-4"
        man = pd / "manifest.json"
        sealed_by = None
        if man.is_file():
            m = json.loads(man.read_text(encoding="utf-8"))
            sealed_by = m.get("tool_sha256") if m.get("verdict") == "sealed" else None
        xt = out / "xtok" / f"pass-{k}" / "xtok_target.safetensors"
        passes.append({"pass": k, "dir": str(pd), "sealed": sealed_by is not None and sealed_by == tool_sha,
                       "sealed_by": sealed_by, "xtok": str(xt), "xtok_built": xt.is_file() and (xt.parent / "xtok_receipt.json").is_file()})
    return {"frozen_tool": str(layout.k3_frozen_tool), "frozen_tool_sha256": tool_sha, "out": str(out), "passes": passes,
            "sealed": sum(1 for p in passes if p["sealed"]), "marker": str(layout.k3_marker), "marker_exists": layout.k3_marker.is_file()}


def masks_state(layout: Layout) -> dict[str, Any]:
    files = {}
    ok = True
    for name, want in MASK_SHAS.items():
        path = layout.masks / name
        got = sha256_file(path) if path.is_file() else None
        files[name] = {"path": str(path), "sha256": got, "pinned": want, "ok": got == want}
        ok = ok and got == want
    receipt = layout.masks / "build_receipt.json"
    check = None
    if receipt.is_file():
        check = (json.loads(receipt.read_text(encoding="utf-8")).get("doc_index_check") or {})
    doc_ok = isinstance(check, dict) and check.get("ok") is True
    return {"files": files, "doc_index_check_ok": doc_ok, "ok": ok and doc_ok}


def inventory(layout: Layout) -> dict[str, Any]:
    manifest = json.loads(layout.text_manifest.read_text(encoding="utf-8"))
    registry = []
    for entry in sorted(manifest["shards"], key=lambda e: e["rows"][0]):
        registry.append({**cache_state(layout.cache_dir(entry["sha256"])), "file": Path(entry["path"]).name,
                         "row_range": [int(entry["rows"][0]), int(entry["rows"][1])], "corpus_sha256": entry["sha256"]})
    k3_q38 = []
    for k in range(4):
        k3_q38.append({"pass": k, **corpus_cache(layout, layout.k3_corpus / f"pass-{k}" / "qwen_input_ids.safetensors")})
    inv = {
        "registry": registry,
        "selfgen": cache_state(layout.cache_dir(SELFGEN_CORPUS_SHA)),
        "toolcall_c1": corpus_cache(layout, layout.toolcall_corpus),
        "bfcl_c2": corpus_cache(layout, layout.bfcl_corpus),
        "k3_q38": k3_q38,
        "long": cache_state(layout.cache_dir(LONG_CORPUS_SHA)),
        "masks": masks_state(layout),
        "k3": k3_state(layout),
    }
    waiting = [f"registry {r['file']}" for r in registry if not r["sealed"]]
    for key in ("selfgen", "toolcall_c1", "bfcl_c2", "long"):
        if not inv[key]["sealed"]:
            waiting.append(f"{key}: {inv[key].get('reason')}")
    waiting += [f"k3_q38 pass {c['pass']}: {c.get('reason')}" for c in k3_q38 if not c["sealed"]]
    if not inv["masks"]["ok"]:
        waiting.append("masks: d2_masks_v1 files / doc_index check not as pinned")
    inv["start_inputs_waiting"] = waiting
    inv["k3_rows_ready"] = inv["k3"]["marker_exists"]
    return inv


# --------------------------------------------------------------------------
# P2 verdict
# --------------------------------------------------------------------------


def read_verdict_file(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    if out.get("engine") not in ("streamed", "split"):
        raise Refusal(f"{path}: engine must be streamed or split, not {out.get('engine')!r}")
    for key in ("micro_batch", "accumulation"):
        if not str(out.get(key, "")).isdigit() or int(out[key]) < 1:
            raise Refusal(f"{path}: {key} must be a positive integer")
    return out


def derive_verdict(summaries: list[Path]) -> dict[str, str] | None:
    """The G5-1 rule on every completed P2 cell of every summary; None when no cell completed."""

    best = None
    long_ok = False
    sources = []
    for path in summaries:
        s = json.loads(path.read_text(encoding="utf-8"))
        sources.append(f"{path}:{sha256_file(path)[:16]}")
        for cell, entry in (s.get("cells") or {}).items():
            if entry.get("state") != "completed":
                continue
            if cell.startswith("L32768"):
                long_ok = True
                continue
            rate = entry.get("steady_tok_per_s_mean_steps_2_on")
            if rate is not None and cell in CELL_SPLITS and (best is None or rate > best[0]):
                best = (float(rate), cell, int(s.get("gpu_index", -1)))
    if best is None:
        return None
    rate, cell, gpu = best
    if rate >= STREAMED_MIN and gpu in (0, 4):
        micro, acc = CELL_SPLITS[cell]
        engine = "streamed"
    else:
        engine, micro, acc = "split", 4, 4
    return {"engine": engine, "micro_batch": str(micro), "accumulation": str(acc), "best_cell": cell, "best_tok_per_s": f"{rate:.2f}",
            "best_gpu": str(gpu), "long_rung_cell_completed": "yes" if long_ok else "no",
            "rule": "STAGE_D_RULINGS G5-1: >= 151 tok/s on a 3090 -> streamed two-replica DP (GPU0+GPU4); else the split cell (GPU6+GPU4, fp16)",
            "by": "auto (d2_plan verdict under the standing GO 2026-10-12; recorded for later review)",
            "summaries": ";".join(sources), "utc": utc()}


def cmd_verdict(args: argparse.Namespace, layout: Layout) -> int:
    path = Path(args.verdict) if args.verdict else layout.p2_verdict
    if path.is_file():
        verdict = read_verdict_file(path)
        emit("verdict", {"source": str(path), "sha256": sha256_file(path), **verdict})
        return 0
    summaries = sorted(Path(p) for p in glob.glob(str(layout.p2_summaries / "p2-summary-v133-gpu*.json")))
    derived = derive_verdict(summaries)
    if derived is None:
        emit("verdict", {"source": None, "waiting": f"no completed P2 cell in {layout.p2_summaries} (summaries: {len(summaries)})"})
        return EXIT_WAITING
    if args.write:
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        tmp.write_text("".join(f"{k}={v}\n" for k, v in derived.items()), encoding="utf-8")
        os.replace(tmp, path)
        emit("verdict", {"source": str(path), "written": True, "sha256": sha256_file(path), **derived})
    else:
        emit("verdict", {"source": "derived (not written: --write not given)", **derived})
    return 0


def current_verdict(layout: Layout, verdict_path: str | None = None) -> dict[str, str] | None:
    """The verdict `verdict` would print: the file if it exists, else derived (never written here); None = waiting."""

    path = Path(verdict_path) if verdict_path else layout.p2_verdict
    if path.is_file():
        return read_verdict_file(path)
    return derive_verdict(sorted(Path(p) for p in glob.glob(str(layout.p2_summaries / "p2-summary-v133-gpu*.json"))))


# --------------------------------------------------------------------------
# memory fit (1.2.0): the P2 cell's memory geometry at D2's real width
# --------------------------------------------------------------------------

#: the trainer's 1.3.3 memory-granularity knobs.  They are NOT in the run's config digest (they regroup the same sums),
#: so the queue may set them from the P2 attempt that completed: P2 retries an OOM'd cell with --ffn-chunk-tokens 2048
#: --head-chunk-mib 128 (run_p2_v133.sh), and the trainer's defaults are the geometry that OOM'd.
KNOBS = (("ffn_chunk_tokens", "--ffn-chunk-tokens"), ("recompute_tokens", "--recompute-tokens"),
         ("head_chunk_mib", "--head-chunk-mib"), ("head_chunk_positions", "--head-chunk-positions"))
LONG_CELL = "L32768-mb1"
MEM_HEADROOM_GIB = 1.5  # CUDA context + allocator slack not counted in peak_allocated (P2's 3090: 24,576 MiB)
GIB = float(1 << 30)


def load_trainer(tools: Path) -> Any:
    sys.path.insert(0, str(Path(tools)))
    import streamed_kd_trainer as T

    return T


def p2_cells(summaries: list[Path]) -> list[dict[str, Any]]:
    """Every completed P2 cell: {cell, gpu, entry, summary}."""

    out = []
    for path in summaries:
        s = json.loads(path.read_text(encoding="utf-8"))
        for cell, entry in (s.get("cells") or {}).items():
            if entry.get("state") == "completed":
                out.append({"cell": cell, "gpu": int(s.get("gpu_index", -1)), "entry": entry, "summary": str(path)})
    return out


def calibration_cell(cells: list[dict[str, Any]], verdict: dict[str, str]) -> dict[str, Any] | None:
    """The completed cell the verdict was read from (derived: best_cell on best_gpu); for a hand-written verdict, the
    fastest completed 4k cell of the verdict's micro-batch on a 3090 (GPU0 / GPU4)."""

    if verdict.get("best_cell"):
        for c in cells:
            if c["cell"] == verdict["best_cell"] and str(c["gpu"]) == str(verdict.get("best_gpu", c["gpu"])):
                return c
    micro = int(verdict["micro_batch"])
    pool = [c for c in cells if c["cell"] in CELL_SPLITS and CELL_SPLITS[c["cell"]][0] == micro and c["gpu"] in (0, 4)]
    return max(pool, key=lambda c: float(c["entry"].get("steady_tok_per_s_mean_steps_2_on") or 0.0)) if pool else None


def receipt_namespace(T: Any, receipt: dict[str, Any]) -> argparse.Namespace:
    argv = list(receipt.get("argv") or [])
    if "train" not in argv:
        raise ValueError("receipt has no `train` argv")
    return T.parser().parse_args(argv[argv.index("train"):])


def knobs_of(ns: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(ns, key) for key, _flag in KNOBS}


def default_config_loader(T: Any, ns: argparse.Namespace) -> Any:
    """The text config of the argv's source, from headers and config.json only: a P1 source is opened WITHOUT its
    plan (a plan replaces values, never shapes, and opening it would re-digest the whole root)."""

    ns = argparse.Namespace(**{**vars(ns), "p1_plan": None})
    return T.text_config_from_source(T.build_source(ns), T.resolve_dtype(ns.dtype), ns.attn_implementation)


def geometry(config: Any) -> dict[str, Any]:
    keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "linear_key_head_dim", "linear_num_key_heads", "linear_value_head_dim", "linear_num_value_heads",
            "vocab_size")
    g = {k: getattr(config, k, None) for k in keys}
    g["layer_types"] = [str(t) for t in config.layer_types[: int(config.num_hidden_layers)]]
    return g


def trainable_parameters(T: Any, config: Any, ns: argparse.Namespace) -> int:
    """The AdapterBank census without building the bank: every part of the site plan, the K3 gain only when trained."""

    plan = T.build_site_plan(config, T.site_plan_from_args(ns))
    frozen_gain = bool(getattr(ns, "freeze_k3_gain", False)) or bool(getattr(ns, "no_k3_gain", False))
    total = 0
    for layer_type in [str(t) for t in config.layer_types[: int(config.num_hidden_layers)]]:
        for spec in plan[layer_type]:
            for part, shape in spec.part_shapes().items():
                if part == "gain" and frozen_gain:
                    continue
                total += math.prod(int(n) for n in shape)
    return total


def predicted_peak(T: Any, ns: argparse.Namespace, config: Any, kernels: dict[str, Any], trainable: int, *, micro: int, length: int) -> int:
    text_rows = min(int(ns.text_vocab_rows), int(config.vocab_size))
    objective = T.objective_from_args(ns, text_rows)
    dtype = T.resolve_dtype(ns.dtype)
    return int(T.predict_memory(ns, config, objective, dtype, kernels, int(trainable), batch=int(micro), length=int(length),
                                optimizer=True)["peak_bytes"])


def memory_fit(layout: Layout, *, width: int, card_mib: int, tools: Path, headroom_gib: float = MEM_HEADROOM_GIB,
               d2_source: dict[str, str] | None = None, d2_ranks: dict[str, int] | None = None, verdict_path: str | None = None,
               config_loader: Any = None) -> tuple[int, dict[str, Any]]:
    """(exit code, fit).  The verdict's micro-batch, checked at the D2 data config's widest row.

    P2 measured its cells at width 4,096; a D2 micro-batch is padded to its widest row (self-gen: 4,841), so the
    trainer's own memory model is evaluated at (micro, width) with the P2 attempt's knobs and D2's ranks, and calibrated
    on P2's measured peak: estimate = max(measured + (pred(micro, width) - pred(P2 cell)), pred(micro, width) x
    measured / pred(P2 cell)).  It fits when the estimate is <= max(measured, card - headroom) -- what P2 ran is proven
    to run.  If it does not fit, the micro-batch halves (accumulation doubles: the same global tokens per step, prereg
    section 6) until it does; nothing fitting is a refusal (exit 2).  A split engine or a missing P2 receipt is not
    modelled: the verdict's micro-batch is kept and the reason recorded.  The knobs of the P2 attempt that completed are
    carried either way (short cell and the long cell), since they are not part of the run's config digest.
    """

    verdict = current_verdict(layout, verdict_path)
    if verdict is None:
        return EXIT_WAITING, {"waiting": "no P2 verdict yet"}
    micro, acc = int(verdict["micro_batch"]), int(verdict["accumulation"])
    fit: dict[str, Any] = {"schema": "omni-d2-memory-fit-v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "utc": utc(),
                           "engine": verdict["engine"], "verdict_micro_batch": micro, "verdict_accumulation": acc,
                           "micro_batch": micro, "accumulation": acc, "width": int(width), "card_mib": int(card_mib),
                           "headroom_gib": float(headroom_gib), "knobs": None, "long_knobs": None, "modeled": False}
    T = load_trainer(tools)
    loader = config_loader or (lambda ns: default_config_loader(T, ns))
    cells = p2_cells(sorted(Path(p) for p in glob.glob(str(layout.p2_summaries / "p2-summary-v133-gpu*.json"))))
    # the long rung runs exactly the P2 long cell's geometry (micro 1, 32,768): only its knobs are carried
    longs = [c for c in cells if c["cell"] == LONG_CELL and c["entry"].get("receipt") and Path(c["entry"]["receipt"]).is_file()]
    if longs:
        lc = sorted(longs, key=lambda c: (c["gpu"] not in (0, 4), c["gpu"]))[0]
        try:
            fit["long_knobs"] = knobs_of(receipt_namespace(T, json.loads(Path(lc["entry"]["receipt"]).read_text(encoding="utf-8"))))
            fit["long_cell"] = {"gpu": lc["gpu"], "used_attempt": lc["entry"].get("used_attempt"), "receipt": lc["entry"]["receipt"]}
        except (ValueError, SystemExit) as exc:
            fit["long_cell"] = {"error": f"long cell receipt unreadable: {exc}"}
    if verdict["engine"] != "streamed":
        fit["reason"] = "split engine: one replica over two cards; the per-device streamed memory model does not apply"
        return 0, fit
    cell = calibration_cell(cells, verdict)
    if cell is None or not cell["entry"].get("receipt") or not Path(cell["entry"]["receipt"]).is_file():
        fit["reason"] = "no completed P2 cell with a receipt for the verdict's micro-batch: not modelled, verdict kept"
        return 0, fit
    receipt_path = Path(cell["entry"]["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    try:
        p2 = receipt_namespace(T, receipt)
    except (ValueError, SystemExit) as exc:
        fit["reason"] = f"P2 receipt {receipt_path} argv unreadable ({exc}): not modelled, verdict kept"
        return 0, fit
    fit["knobs"] = knobs_of(p2)
    measured = cell["entry"].get("measured_peak_gib") or (receipt.get("memory_engine") or {}).get("peak_allocated_gib")
    kernels = receipt.get("kernels") or {}
    receipt_trainable = (receipt.get("adapter_census") or {}).get("parameters_trainable")
    fit["calibration"] = {"cell": cell["cell"], "gpu": cell["gpu"], "used_attempt": cell["entry"].get("used_attempt"),
                          "summary": cell["summary"], "receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
                          "p2_micro_batch": int(p2.micro_batch), "p2_seq_len": int(p2.seq_len), "measured_peak_gib": measured,
                          "receipt_predicted_peak_gib": round(float((receipt.get("memory_model") or {}).get("peak_bytes", 0)) / GIB, 3),
                          "kernel_family": "fla" if "fla" in str(kernels.get("chunk_gated_delta_rule", "")) else "torch"}
    if not isinstance(measured, (int, float)) or measured <= 0:
        fit["reason"] = "the P2 cell has no measured peak: not modelled, verdict kept (knobs carried)"
        return 0, fit
    try:
        config_p2 = loader(p2)
    except Exception as exc:  # noqa: BLE001 -- a source that cannot be opened is a reason not to model, not to stop D2
        fit["reason"] = f"the P2 cell's source cannot be opened ({type(exc).__name__}: {str(exc)[:200]}): not modelled, verdict kept"
        return 0, fit
    p2_trainable = trainable_parameters(T, config_p2, p2)
    if receipt_trainable is not None and int(receipt_trainable) != p2_trainable:
        fit["reason"] = (f"the site plan gives {p2_trainable:,} trainable parameters for the P2 argv, its receipt says "
                         f"{int(receipt_trainable):,}: the model is not reproducing P2, so not modelled, verdict kept")
        return 0, fit
    d2 = argparse.Namespace(**{**vars(p2), **(d2_ranks or {})})
    config_d2 = config_p2
    if d2_source:
        try:
            config_d2 = loader(argparse.Namespace(**{**vars(d2), "p1_chimera": d2_source["chimera"],
                                                     "p1_parent_config": d2_source.get("parent_config") or p2.p1_parent_config,
                                                     "model": None, "source_plugin": None}))
        except Exception as exc:  # noqa: BLE001
            fit["reason"] = f"the D2 base {d2_source['chimera']} cannot be opened ({type(exc).__name__}: {str(exc)[:200]}): not modelled, verdict kept"
            return 0, fit
    fit["geometry_equal_to_p2"] = geometry(config_d2) == geometry(config_p2)
    d2_trainable = trainable_parameters(T, config_d2, d2)
    pred_p2 = predicted_peak(T, p2, config_p2, kernels, p2_trainable, micro=int(p2.micro_batch), length=int(p2.seq_len))
    measured = float(measured)
    budget = max(measured, int(card_mib) / 1024.0 - float(headroom_gib))
    fit.update(modeled=True, budget_gib=round(budget, 3), d2_trainable=d2_trainable, p2_trainable=p2_trainable,
               p2_predicted_peak_gib=round(pred_p2 / GIB, 3), ratio_measured_to_predicted=round(measured * GIB / pred_p2, 4))
    rows = []
    chosen = None
    m = micro
    while m >= 1:
        pred = predicted_peak(T, d2, config_d2, kernels, d2_trainable, micro=m, length=int(width))
        additive = measured + (pred - pred_p2) / GIB
        scaled = pred / GIB * measured * GIB / pred_p2
        estimate = max(additive, scaled)
        rows.append({"micro_batch": m, "accumulation": micro * acc // m, "predicted_gib": round(pred / GIB, 3),
                     "estimate_gib": round(estimate, 3), "fits": estimate <= budget})
        if estimate <= budget and chosen is None:
            chosen = rows[-1]
            break
        if m % 2:
            break
        m //= 2
    fit["rows"] = rows
    if chosen is None:
        fit["reason"] = f"no micro-batch <= {micro} fits {budget:.2f} GiB at width {width}"
        return 2, fit
    fit["micro_batch"], fit["accumulation"] = chosen["micro_batch"], chosen["accumulation"]
    fit["reason"] = ("the verdict's micro-batch fits at the D2 width" if chosen["micro_batch"] == micro else
                     f"micro {micro} does not fit at width {width}: {chosen['micro_batch']} x accumulation {chosen['accumulation']} "
                     "(same global tokens per step, prereg section 6)")
    return 0, fit


def cmd_memory_fit(args: argparse.Namespace, layout: Layout) -> int:
    config = json.loads(Path(args.data).read_text(encoding="utf-8"))
    width = int(config.get("max_width") or max_width(config))
    d2_source = {"chimera": args.d2_chimera, "parent_config": args.d2_parent_config} if args.d2_chimera else None
    ranks = {"lora_rank": args.lora_rank, "k3_lora_rank": args.k3_lora_rank, "trunk_ffn_width": args.trunk_ffn_width}
    code, fit = memory_fit(layout, width=width, card_mib=args.card_mib, tools=Path(args.tools), headroom_gib=args.headroom_gib,
                           d2_source=d2_source, d2_ranks=ranks, verdict_path=args.verdict)
    fit["data_config"] = {"path": args.data, "sha256": sha256_file(Path(args.data))}
    if args.out and code == 0:
        emit("memory-fit", {**fit, "out": args.out, "out_sha256": write_json_atomic(Path(args.out), fit)})
    else:
        emit("memory-fit", fit)
    if code == 2:
        raise Refusal(f"memory-fit: {fit.get('reason')}")
    return code


# --------------------------------------------------------------------------
# data configs
# --------------------------------------------------------------------------


def safetensors_shape(path: Path, key: str) -> list[int]:
    with open(path, "rb") as handle:
        n = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(n))
    return list(header[key]["shape"])


def real_tokens_of(path: Path, exclude: set[int] | frozenset[int] = frozenset()) -> int:
    """Sum of a corpus's real lengths (`lengths`, else `real_lengths`, else full rows) over the rows not excluded."""

    from safetensors.numpy import load_file

    t = load_file(str(path))
    for key in ("lengths", "real_lengths"):
        if key in t:
            lengths = t[key].astype("int64")
            return int(sum(int(n) for i, n in enumerate(lengths) if i not in exclude))
    rows, width = t["input_ids"].shape
    return int((rows - len([r for r in exclude if 0 <= r < rows])) * width)


def build_data_config(layout: Layout, inv: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "long":
        long_cache = inv["long"]
        return {"schema": "omni-d2-data-config-v1", "kind": "long", "written_by": f"{TOOL_NAME} {TOOL_VERSION}",
                "sources": [{"name": "long", "kind": "text", "qtext_cache": long_cache["dir"]}]}
    excl = layout.masks / "exclude_rows.json"
    exclusion = json.loads(excl.read_text(encoding="utf-8"))
    excluded_by_sha = {sha: len(e["rows"]) for sha, e in exclusion["corpora"].items()}
    sources = []
    registry_rows = sum((r["row_range"][1] - r["row_range"][0]) - excluded_by_sha.get(r["corpus_sha256"], 0) for r in inv["registry"])
    registry_budget = min(REGISTRY_DRAW + ONPOLICY_FALLBACK, registry_rows * SEQ)
    for r in inv["registry"]:
        rows = (r["row_range"][1] - r["row_range"][0]) - excluded_by_sha.get(r["corpus_sha256"], 0)
        sources.append({"name": f"registry-{r['file'].split('.')[0]}", "kind": "text", "qtext_cache": r["dir"],
                        "tokens": int(round(rows * SEQ * registry_budget / (registry_rows * SEQ))), "budget_unit": "positions",
                        "exclude_rows": str(excl), "gt_mask": str(layout.masks / "gt_mask.json"),
                        "all_mask": str(layout.masks / "all_mask.json")})
    selfgen = inv["selfgen"]
    # the twin exclusion file names every D2 input's twin rows by corpus sha; a source with no entry loses nothing
    # (the self-gen twin prompts were never rescored), and the trainer records that per source
    sources.append({"name": "selfgen", "kind": "text", "qtext_cache": selfgen["dir"], "budget_unit": "real_tokens",
                    "tokens": int(selfgen["tokens"]), "exclude_rows": str(excl)})
    for key, name in (("toolcall_c1", "tool-selfgen"), ("bfcl_c2", "tool-bfcl-native")):
        c = inv[key]
        sources.append({"name": name, "kind": "text", "qtext_cache": c["dir"], "supervision": "assistant_only", "w_ul": 0.0,
                        "budget_unit": "real_tokens", "tokens": int(c["tokens"])})
    k3_out = Path(inv["k3"]["out"])
    deferred_rows = 0
    for c in inv["k3_q38"]:
        k = c["pass"]
        corpus = layout.k3_corpus / f"pass-{k}" / "qwen_input_ids.safetensors"
        gone = {int(r) for r in (exclusion["corpora"].get(c["corpus_sha256"]) or {}).get("rows", [])}  # K3 twin rows
        deferred_rows += safetensors_shape(corpus, "input_ids")[0] - len(gone)  # one epoch of the pass (budget = its real tokens)
        sources.append({"name": f"k3-pass-{k}", "kind": "k3", "qtext_cache": c["dir"],
                        "xtok": str(k3_out / "xtok" / f"pass-{k}" / "xtok_target.safetensors"),
                        "xtok_qwen_file": "qwen_input_ids", "budget_unit": "real_tokens", "tokens": real_tokens_of(corpus, gone),
                        "exclude_rows": str(excl),
                        "defer_until_tokens": K3_DEFER_TOKENS, "sealed_marker": str(layout.k3_marker)})
    return {"schema": "omni-d2-data-config-v1", "kind": "short", "written_by": f"{TOOL_NAME} {TOOL_VERSION}",
            "registry_budget_positions": registry_budget, "registry_positions_after_exclusion": registry_rows * SEQ,
            "onpolicy": ("no on-policy sampler exists: the 1.15M on-policy tokens come from the off-policy registry "
                         "(prereg section 3 D2 Data), capped at one epoch of the rows left after the BFCL-62 exclusion"),
            "rungs": list(RUNGS), "k3_defer_until_tokens": K3_DEFER_TOKENS, "deferred_rows": deferred_rows, "sources": sources}


def max_width(config: dict[str, Any]) -> int:
    widths = [SEQ]
    for s in config["sources"]:
        manifest = Path(s["qtext_cache"]) / "manifest.json"
        if manifest.is_file():
            corpus = Path(json.loads(manifest.read_text(encoding="utf-8"))["corpus"]["path"])
            widths.append(safetensors_shape(corpus, "input_ids")[1])
    return max(widths)


def cmd_data_config(args: argparse.Namespace, layout: Layout) -> int:
    inv = inventory(layout)
    if args.kind == "short" and inv["start_inputs_waiting"]:
        emit("data-config", {"waiting": inv["start_inputs_waiting"]})
        return EXIT_WAITING
    if args.kind == "long" and not inv["long"]["sealed"]:
        emit("data-config", {"waiting": ["long cache"]})
        return EXIT_WAITING
    config = build_data_config(layout, inv, args.kind)
    width = max_width(config)
    config["max_width"] = width
    sha = write_json_atomic(Path(args.out), config)
    emit("data-config", {"out": args.out, "sha256": sha, "sources": len(config["sources"]), "max_width": width,
                         "registry_budget_positions": config.get("registry_budget_positions"),
                         "deferred_rows": config.get("deferred_rows", 0)})
    return 0


def cmd_inventory(args: argparse.Namespace, layout: Layout) -> int:
    inv = inventory(layout)
    if args.out:
        write_json_atomic(Path(args.out), inv)
    brief = {"registry_sealed": sum(1 for r in inv["registry"] if r["sealed"]), "registry": len(inv["registry"]),
             "selfgen": inv["selfgen"]["sealed"], "toolcall_c1": inv["toolcall_c1"]["sealed"], "bfcl_c2": inv["bfcl_c2"]["sealed"],
             "k3_q38_sealed": sum(1 for c in inv["k3_q38"] if c["sealed"]), "long": inv["long"]["sealed"],
             "masks_ok": inv["masks"]["ok"], "k3_passes_sealed": inv["k3"]["sealed"], "k3_marker": inv["k3"]["marker_exists"],
             "waiting": inv["start_inputs_waiting"]}
    emit("inventory", brief)
    return 0 if not inv["start_inputs_waiting"] else EXIT_WAITING


# --------------------------------------------------------------------------
# amendment A2
# --------------------------------------------------------------------------


A2_DRAFT_HEADING = re.compile(r"^### A2, DRAFT \(written 2026-10-14; NOT IN FORCE\)", re.M)


def a2_state(prereg: Path) -> dict[str, Any]:
    text = prereg.read_text(encoding="utf-8")
    present = "\n### A2" in text
    draft = bool(A2_DRAFT_HEADING.search(text))
    return {"prereg": str(prereg), "sha256": sha256_file(prereg), "a2_present": present, "a2_draft": draft,
            "placeholders": sorted(set(re.findall(r"`<(C1|C2)>`", text))), "in_force": present and not draft}


def check_prereg(prereg: Path, tools: Path) -> dict[str, Any]:
    run = subprocess.run([sys.executable, str(tools / "stage2_training_receipt.py"), "check-prereg", str(prereg)],
                         capture_output=True, text=True, timeout=120)
    line = next((l for l in run.stdout.splitlines() if l.startswith("PREREG OK")), None)
    if run.returncode != 0 or line is None:
        raise Refusal(f"check-prereg failed on {prereg}: rc {run.returncode} {run.stdout[-300:]} {run.stderr[-300:]}")
    return json.loads(line[len("PREREG OK "):])


def cmd_a2(args: argparse.Namespace, layout: Layout) -> int:
    prereg = Path(args.prereg) if args.prereg else layout.prereg
    state = a2_state(prereg)
    if args.command == "a2-state":
        emit("a2-state", state)
        return 0 if state["in_force"] else EXIT_WAITING
    if state["in_force"]:
        emit("a2-finalize", {**state, "note": "already in force; nothing written"})
        return 0
    if not state["a2_draft"]:
        raise Refusal(f"{prereg} has no A2 draft block to finalize")
    inv = inventory(layout)
    c1, c2 = inv["toolcall_c1"], inv["bfcl_c2"]
    if not (c1["sealed"] and c2["sealed"]):
        emit("a2-finalize", {**state, "waiting": "C1 and C2 caches must be sealed first"})
        return EXIT_WAITING
    bfcl_tokens = int(c2.get("tokens") or 0)
    if bfcl_tokens != BFCL_EXPECT["tokens"]:
        raise Refusal(f"C2 cache corpus holds {bfcl_tokens} tokens, not the copied {BFCL_EXPECT['tokens']}")
    before = check_prereg(prereg, Path(args.tools))
    backup = prereg.with_name(prereg.name + f".pre-A2final-{state['sha256'][:8]}")
    shutil.copy2(prereg, backup)
    text = prereg.read_text(encoding="utf-8")
    if text.count("`<C1>`") != 1 or text.count("`<C2>`") != 1:
        raise Refusal(f"{prereg}: expected exactly one `<C1>` and one `<C2>` placeholder in the A2 draft")
    text = text.replace("`<C1>`", f"`{c1['manifest_sha256']}` (cache {c1['dir']})", 1)
    text = text.replace("`<C2>`", f"`{c2['manifest_sha256']}` (cache {c2['dir']})", 1)
    text = A2_DRAFT_HEADING.sub(f"### A2, in force {utc()} (finalized by {TOOL_NAME} {TOOL_VERSION} from the sealed C1/C2 manifests)", text, count=1)
    text = text.replace("**Status: DRAFT, pending the cache shas C1 and C2.**",
                        "**Status: in force (was a DRAFT pending the cache shas C1 and C2; filled mechanically, no other word changed).**", 1)
    tmp = prereg.with_name(f".{prereg.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    after = check_prereg(tmp, Path(args.tools))
    if after["preregistration_sha256"] != before["preregistration_sha256"]:
        tmp.unlink()
        raise Refusal("the frozen-body pin moved while finalizing A2; nothing written")
    os.replace(tmp, prereg)
    emit("a2-finalize", {**a2_state(prereg), "backup": str(backup), "pin": after["preregistration_sha256"],
                         "c1_manifest_sha256": c1["manifest_sha256"], "c2_manifest_sha256": c2["manifest_sha256"]})
    return 0


# --------------------------------------------------------------------------
# K3 seal (xtok targets with the B2-proven frozen tool, verified through the trainer's reader)
# --------------------------------------------------------------------------


def cmd_k3(args: argparse.Namespace, layout: Layout) -> int:
    state = k3_state(layout)
    if args.command == "k3-state":
        emit("k3-state", state)
        return 0 if state["marker_exists"] else EXIT_WAITING
    if state["marker_exists"]:
        emit("k3-seal", {**state, "note": "marker exists; nothing to do"})
        return 0
    if state["frozen_tool_sha256"] is None or state["sealed"] < 4:
        emit("k3-seal", {**state, "waiting": f"{state['sealed']} of 4 K3 passes sealed by the frozen tool"})
        return EXIT_WAITING
    sys.path.insert(0, str(Path(args.tools)))
    import streamed_kd_trainer as T  # the trainer's own reader proves the join

    built = []
    for p in state["passes"]:
        k = p["pass"]
        src = layout.k3_corpus / f"pass-{k}"
        out = Path(p["xtok"]).parent
        if not p["xtok_built"]:
            cmd = [sys.executable, str(layout.k3_frozen_tool), "xtok", "--cache-dir", p["dir"],
                   "--boundary-map", str(src / "boundary_map.json"), "--k3-ids", str(src / "k3_input_ids.safetensors"),
                   "--qwen-ids", str(src / "qwen_input_ids.safetensors"), "--out", str(out)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            (out.parent / f"xtok-pass-{k}.log").parent.mkdir(parents=True, exist_ok=True)
            (out.parent / f"xtok-pass-{k}.log").write_text(run.stdout + run.stderr, encoding="utf-8")
            if run.returncode != 0:
                raise Refusal(f"xtok pass {k} failed rc {run.returncode}: {run.stderr[-400:]}")
        cache = T.QTextCache(layout.cache_dir(sha256_file(src / "qwen_input_ids.safetensors")))
        xtok = T.XtokTargets(Path(p["xtok"]), "qwen_input_ids")
        proof = xtok.verify_against(cache)
        built.append({"pass": k, "xtok_sha256": xtok.sha256, "pairs": xtok.pairs, "provenance": proof.get("method"),
                      "q38_cache_manifest_sha256": cache.manifest_sha256})
    marker = {"schema": "omni-d2-k3-seal-v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "utc": utc(),
              "frozen_k3_tool_sha256": state["frozen_tool_sha256"], "passes": built,
              "rule": "A2 item 8 (draft): the K3 cache is sealed before the first K3 row"}
    sha = write_json_atomic(layout.k3_marker, marker)
    emit("k3-seal", {"marker": str(layout.k3_marker), "sha256": sha, "passes": built})
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    p.add_argument("--oc", default=str(OC))
    p.add_argument("--lex", default=str(LEX))
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inventory")
    i.add_argument("--out")
    v = sub.add_parser("verdict")
    v.add_argument("--verdict", help="verdict file (default $Q/p2.verdict)")
    v.add_argument("--write", action="store_true", help="write a derived verdict to the verdict file")
    d = sub.add_parser("data-config")
    d.add_argument("--kind", choices=("short", "long"), default="short")
    d.add_argument("--out", required=True)
    for name in ("a2-state", "a2-finalize"):
        a = sub.add_parser(name)
        a.add_argument("--prereg")
        a.add_argument("--tools", default=str(Path(__file__).resolve().parent))
    for name in ("k3-state", "k3-seal"):
        k = sub.add_parser(name)
        k.add_argument("--tools", default=str(Path(__file__).resolve().parent))
    m = sub.add_parser("memory-fit", help="1.2.0: the verdict's micro-batch at the data config's widest row")
    m.add_argument("--data", required=True, help="a D2 data config (its max_width, else its sources' widest corpus)")
    m.add_argument("--card-mib", type=int, required=True, help="the smallest memory.total of the engine's cards (nvidia-smi)")
    m.add_argument("--headroom-gib", type=float, default=MEM_HEADROOM_GIB)
    m.add_argument("--d2-chimera", help="the D2 base's root Chimera directory (geometry; default: the P2 cell's source)")
    m.add_argument("--d2-parent-config")
    m.add_argument("--lora-rank", type=int, default=64)
    m.add_argument("--k3-lora-rank", type=int, default=32)
    m.add_argument("--trunk-ffn-width", type=int, default=4352)
    m.add_argument("--verdict", help="verdict file (default $Q/p2.verdict, else derived)")
    m.add_argument("--out", help="write the fit here (only when it fits)")
    m.add_argument("--tools", default=str(Path(__file__).resolve().parent))
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    layout = Layout(Path(args.oc), Path(args.lex))
    try:
        if args.command == "inventory":
            return cmd_inventory(args, layout)
        if args.command == "verdict":
            return cmd_verdict(args, layout)
        if args.command == "data-config":
            return cmd_data_config(args, layout)
        if args.command in ("a2-state", "a2-finalize"):
            return cmd_a2(args, layout)
        if args.command in ("k3-state", "k3-seal"):
            return cmd_k3(args, layout)
        if args.command == "memory-fit":
            return cmd_memory_fit(args, layout)
    except Refusal as refusal:
        print(f"REFUSED: {refusal.message}", file=sys.stderr, flush=True)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
