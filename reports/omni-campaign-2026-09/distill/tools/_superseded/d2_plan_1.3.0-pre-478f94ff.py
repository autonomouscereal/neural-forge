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
  k3-state      (1.3.0) which K3 passes count as sealed (A2 item 8a): a pass sealed by a pinned K3 tool, recorded in
                $Q/k3_pass_tools.json (k3_lib.k3_record_pass) with its card class and a PASSING gate receipt, checked
                here: the frozen 1.2.4 (2aae4dc3...) on GPU6's class behind the B2 PASS; the frozen 1.3.0 (434ced6d...)
                on GPU6's class behind the k3-v130-gate bit-identity PASS (its compare receipts re-hashed, the +1-ulp
                negative control DIFFER); either tool on another card class behind that class's card-class equivalence
                PASS (ruling k3-card-class-equivalence-2026-10-15).  ONE geometry per pass: <pass>.geometry, the
                manifest and every runner-log attempt line must name one tool, one card class and one group-token
                count.  A pass sealed by the frozen 1.2.4 on GPU6 that no record names (a queue <= 1.3.x sealed it) is
                accepted on the same checks and marked legacy.  Anything else is refused.  `k3-seal` builds each pass's
                xtok target with the frozen 1.2.4 (its xtok code is byte-identical in 1.3.0), verifies it against its
                Q38 cache through streamed_kd_trainer's own XtokTargets.verify_against, and only then writes the seal
                marker the D2 data config waits on (each pass's manifest, xtok and gate shas in it).
  memory-fit    (1.2.0) the verdict's micro-batch at the data config's widest row, through the trainer's own memory
                model calibrated on the P2 cell's measured peak; halves the micro-batch (same global tokens) until it
                fits the card, and carries the knobs (FFN slice, row group, head chunk) of the P2 attempts that completed.
  decisions-state / decisions-apply   (1.3.0) D2_DECISIONS_2026-10-15 ($Q/D2_DECISIONS_2026-10-15.json, pinned sha):
                the d2_masks_v2 build (d2_mask_build 1.2.0 `decisions`: D2-X1 held-out all-terms spans, D2-X2 end-of-turn
                gt spans, no-loss rows, its counts against expected_counts, the 0-hit re-checks, a passed check-decisions)
                and the tool-call export's 13-gram screen against the held-out text must be in place before a short data
                config is written.  The registry variant: lane d2clean's CLEAN rows (d2_clean_repack 1.x,
                $OC/distill/corpus/text_clean_v1: the sealed rows no held-out document touches + the re-packed rows, with
                its clean_* masks) when that version is built, verified and every re-pack Q38 cache is sealed, else the
                span-mask rows (the 20 sealed registry caches + d2_masks_v2); the choice and its reason are recorded in
                the data config.  `decisions-apply --data` writes the applied receipt
                (distill/corpus/d2_masks_v2/D2_DECISIONS_APPLIED.json) for exactly that config; d2_launch.sh refuses to
                train without it.  The D2-X1 refill (ranks 1-4) is declared in the config; D2 never waits on it.

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
TOOL_VERSION = "1.3.0"  # 2026-10-15 (lane d2plan, Claude lead): k3-state accepts passes by $Q/k3_pass_tools.json + their
# gate receipts (1.2.4 B2 / 1.3.0 k3-v130-gate / card-class equivalence), one geometry per pass, xtok checked in the
# marker; D2_DECISIONS_2026-10-15 applied (d2_masks_v2, the d2clean preference, the refill, the applied receipt)
# 1.2.0 2026-10-15 (adversarial review): memory-fit -- P2 measured its cells at width 4,096, D2 pads a
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

#: K3 (1.3.0, A2 item 8a; ruling k3-card-class-equivalence-2026-10-15)
K3_V124_SHA = "2aae4dc3134dda725e3ee35572584fbf35532e0c4a7a5721a0b0c817fc2c4f9d"  # frozen 1.2.4, B2-proven on GPU6
K3_V130_SHA = "434ced6d7e8e22e4179bb2332d4695803dd89a38407805126261500c112fa3c7"  # frozen 1.3.0 (k3-v130-gate on GPU6)
K3_TOOLS = {K3_V124_SHA: "1.2.4", K3_V130_SHA: "1.3.0"}
K3_HOME_CLASS = "quadro-rtx-8000-sm75"  # GPU6, the card B2 ran on (k3_lib k3_card_class)
K3_OUT_OF_SCOPE_CC = ("-sm61",)  # the P40s and GPU5 (ruling part 4)
K3_CARD_CLASS_RULING = "k3-card-class-equivalence-2026-10-15"
K3_V130_GATE_EXITS = {"w1": "0", "w2p": "0", "w4": "0", "w1_NEG": "3"}  # queue_gpu6 1.4.0 v130_gate: A-B EQUAL, NEG DIFFER
K3_V130_COMPARES = {"compare_w1_A_B.json": True, "compare_w2p_A_B.json": True, "compare_w4_A_B.json": True,
                    "compare_w1_A_NEG.json": False}  # bit_identical expected
K3_ATTEMPT = re.compile(r"^== k3-pass-(\d) attempt (\d+) group_tokens (\d+)(?: tool ([0-9a-f]{16}))?(?: \((v\d+)\) class (\S+))?")

#: D2_DECISIONS_2026-10-15 (1.3.0)
DECISIONS_SHA = "c8d47f22784d265a2e9239f9274c062298d25278497ba21d01102aa45274d90d"
APPLIED_SUFFIX = "distill/corpus/d2_masks_v2/D2_DECISIONS_APPLIED.json"
MASK_BUILD_MIN = (1, 2, 0)  # d2_mask_build `decisions`
CLEAN_REPACK_SCHEMA = "omni-d2-clean-repack-v1"  # d2_clean_repack.py (lane d2clean)
CLEAN_MASKS = {"exclude_rows": "clean_exclude_rows.json", "all_mask": "clean_all_mask.json", "gt_mask": "clean_gt_mask.json"}
SCREEN_SCHEMA = "omni-d2x1-toolcall-screen-v1"
TOOL_SELFGEN_BASE_TOKENS = 1_500_000  # A2 2(a): the tool-call self-gen already in the plan; refill rank 1 is what is beyond
#: refill ranks 2 and 3 (decisions refill_policy): corpora a refill lane writes here, used once their Q38 cache is sealed
REFILL_CANDIDATES = (
    {"rank": 2, "name": "refill-selfgen", "file": "selfgen-refill.safetensors", "supervision": "all_tokens", "cap": "remainder"},
    {"rank": 3, "name": "refill-codex-v7-train", "file": "codex-native-v7-train.safetensors", "supervision": "assistant_only", "cap": None},
    {"rank": 3, "name": "refill-codex-v8-train", "file": "codex-native-v8-train.safetensors", "supervision": "assistant_only", "cap": None},
)

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
        # 1.3.0: K3 pass acceptance
        self.k3_pass_tools = self.q / "k3_pass_tools.json"
        self.k3_b2_base = self.lex / "distill" / "parity" / "k3cache-2026-10-14-b2-gpu6"
        self.k3_rebase_ruling = self.q / "k3_b2_rebaseline_ruling.txt"
        self.k3_v130_gate = self.q / "k3_v130_gate"
        self.k3_cardclass = self.q / "k3_cardclass"
        self.k3_logs = self.q / "logs"
        # 1.3.0: D2_DECISIONS_2026-10-15
        self.decisions = self.q / "D2_DECISIONS_2026-10-15.json"
        self.masks_v2 = self.oc / "distill" / "corpus" / "d2_masks_v2"
        self.applied = self.oc / APPLIED_SUFFIX
        self.clean_dir = self.oc / "distill" / "corpus" / "text_clean_v1"
        self.toolcall_screen = self.toolcall_corpus.parent / "heldout_screen.json"
        self.refill_dir = self.oc / "distill" / "corpus" / "d2_refill"

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


def load_json(path: Path) -> Any:
    """A JSON file, or None when it is missing or unreadable (a file mid-write reads as missing)."""

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def card_class_of(name: str, capability: str) -> str | None:
    """k3_lib.sh k3_card_class: lower-cased name, runs of non-alphanumerics -> '-', then '-sm' + the capability digits."""

    n = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower())
    c = re.sub(r"[^0-9]", "", str(capability or ""))
    return f"{n}-sm{c}" if n and c else None


def manifest_card_class(m: dict[str, Any]) -> str | None:
    """The card a K3 manifest ran on: the `devices` entry whose index is the manifest's `device` (cuda:<i>)."""

    dev = str(m.get("device") or "")
    idx = int(dev.split(":", 1)[1]) if dev.startswith("cuda:") and dev.split(":", 1)[1].isdigit() else None
    devs = [d for d in (m.get("devices") or []) if idx is not None and int(d.get("index", -1)) == idx]
    return card_class_of(devs[0].get("name"), devs[0].get("capability")) if len(devs) == 1 else None


def b2_state(layout: Layout) -> dict[str, Any]:
    """queue_gpu6 / k3_lib b2_ok for the frozen 1.2.4: the B2 parity PASS, its sealed bf16 review pass and (under the
    rebase ruling) the sealed fp32 pass and the 93-layer fold-identity PASS, every one by exactly that tool."""

    base = layout.k3_b2_base
    b2 = base
    qt = base / "queue_tool.sha256"
    if qt.is_file():
        if qt.read_text(encoding="utf-8").strip() != K3_V124_SHA:
            b2 = base.with_name(f"{base.name}-tool{K3_V124_SHA[:16]}")
    elif base.is_dir() and any(base.iterdir()):
        b2 = base.with_name(f"{base.name}-tool{K3_V124_SHA[:16]}")
    rebase = layout.k3_rebase_ruling.is_file() and layout.k3_rebase_ruling.stat().st_size > 0
    checks: dict[str, Any] = {}

    def record(name: str, path: Path, ok: bool) -> None:
        checks[name] = {"path": str(path), "ok": ok, "sha256": sha256_file(path) if path.is_file() else None}

    def sealed_by_v124(path: Path) -> bool:
        m = load_json(path) or {}
        return m.get("verdict") == "sealed" and m.get("tool_sha256") == K3_V124_SHA

    parity = load_json(b2 / "parity_b2_receipt.json") or {}
    record("parity", b2 / "parity_b2_receipt.json", parity.get("verdict") == "pass")
    record("review_pass", b2 / "review-pass" / "manifest.json", sealed_by_v124(b2 / "review-pass" / "manifest.json"))
    if rebase:
        record("review_pass_fp32", b2 / "review-pass-fp32" / "manifest.json", sealed_by_v124(b2 / "review-pass-fp32" / "manifest.json"))
        ident = load_json(b2 / "identity" / "b2_identity.json") or {}
        v = ident.get("verdict") or {}
        record("identity", b2 / "identity" / "b2_identity.json",
               ident.get("mode") == "identity" and v.get("pass") is True and (ident.get("frozen_tool") or {}).get("sha256") == K3_V124_SHA
               and v.get("layers_checked") == v.get("layers_expected") == 93)
    return {"dir": str(b2), "rebase_ruling": rebase, "ok": all(c["ok"] for c in checks.values()), "checks": checks}


def k3_gate_check(layout: Layout, tool: str, cls: str, gate_path: str | None, b2: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """(problems, gate record) for a pass sealed by `tool` on card class `cls` (ruling k3-card-class-equivalence-2026-10-15)."""

    problems: list[str] = []
    if not b2["ok"]:
        problems.append(f"no B2 PASS for the frozen 1.2.4 in {b2['dir']} ({[n for n, c in b2['checks'].items() if not c['ok']]})")
    if any(cls.endswith(cc) for cc in K3_OUT_OF_SCOPE_CC):
        return problems + [f"card class {cls} is out of scope for K3 (ruling part 4)"], {"kind": "out-of-scope"}
    if cls == K3_HOME_CLASS and tool == K3_V124_SHA:
        if gate_path and Path(gate_path).resolve() != (Path(b2["dir"]) / "parity_b2_receipt.json").resolve():
            problems.append(f"the record names gate {gate_path}, not the B2 receipt (the 1.2.4 passes on GPU6 run behind B2)")
        return problems, {"kind": "b2", "receipts": b2["checks"]}
    if cls == K3_HOME_CLASS and tool == K3_V130_SHA:
        want = layout.k3_v130_gate / cls / "verdict.json"
        if not gate_path or Path(gate_path).resolve() != want.resolve():
            return problems + [f"a 1.3.0 pass on {cls} needs the k3-v130-gate verdict {want}, the record names {gate_path}"], {"kind": "k3-v130-gate"}
        v = load_json(want) or {}
        if not (v.get("schema") == "omni-k3-v130-gate-v1" and v.get("verdict") == "pass" and v.get("tool_sha256") == K3_V130_SHA
                and v.get("reference_tool_sha256") == K3_V124_SHA and v.get("card_class") == cls):
            problems.append(f"k3-v130-gate {want}: not a PASS of 1.3.0 {K3_V130_SHA[:16]} vs 1.2.4 {K3_V124_SHA[:16]} on {cls} "
                            f"(verdict {v.get('verdict')!r}, tool {str(v.get('tool_sha256'))[:16]}, class {v.get('card_class')})")
        exits = dict(x.split("=", 1) for x in str(v.get("compare_exits") or "").split() if "=" in x)
        if exits != K3_V130_GATE_EXITS:
            problems.append(f"k3-v130-gate compare exits {exits} != {K3_V130_GATE_EXITS} (every window EQUAL, the +1-ulp control DIFFER)")
        recorded = v.get("compare_receipts_sha256") or {}
        for name, identical in K3_V130_COMPARES.items():
            path = Path(str(v.get("run_dir", ""))) / name
            if not path.is_file() or sha256_file(path) != recorded.get(name):
                problems.append(f"k3-v130-gate compare receipt {path} missing or not the sha the verdict recorded")
                continue
            c = load_json(path) or {}
            if c.get("bit_identical") is not identical or (c.get("a") or {}).get("sha256") != K3_V124_SHA \
                    or (c.get("b") or {}).get("sha256") != K3_V130_SHA or not c.get("tensors"):
                problems.append(f"k3-v130-gate {name}: bit_identical {c.get('bit_identical')} (want {identical}) of 1.2.4 vs 1.3.0")
        return problems, {"kind": "k3-v130-gate", "path": str(want), "sha256": sha256_file(want) if want.is_file() else None}
    # another card class: the equivalence check by the very tool its passes use
    want = layout.k3_cardclass / cls / "verdict.json"
    if not gate_path or Path(gate_path).resolve() != want.resolve():
        return problems + [f"a pass on {cls} needs its card-class verdict {want}, the record names {gate_path}"], {"kind": "card-class"}
    v = load_json(want) or {}
    if not (v.get("schema") == "omni-k3-cardclass-v1" and v.get("ruling") == K3_CARD_CLASS_RULING and v.get("verdict") == "pass"
            and v.get("tool_sha256") == tool and v.get("card_class") == cls and v.get("parity_code_tool_sha256") == K3_V124_SHA):
        problems.append(f"card-class verdict {want}: not a PASS of tool {tool[:16]} on {cls} under {K3_CARD_CLASS_RULING} "
                        f"(verdict {v.get('verdict')!r}, tool {str(v.get('tool_sha256'))[:16]}, class {v.get('card_class')})")
    gate = v.get("gate") or {}
    gp = Path(str(gate.get("path", "")))
    if not gp.is_file() or sha256_file(gp) != gate.get("sha256") or (load_json(gp) or {}).get("verdict") != "pass":
        problems.append(f"card-class gate receipt {gp} missing, changed, or not a parity PASS against the B2 fp32 reference")
    return problems, {"kind": "card-class", "path": str(want), "sha256": sha256_file(want) if want.is_file() else None}


def k3_attempts(layout: Layout, k: int) -> list[dict[str, Any]]:
    """Every runner-log attempt line of pass k (queue_gpu6 $Q/logs/gpu6, k3_multicard $Q/logs/k3mc-gpu<N>)."""

    out = []
    for path in sorted(glob.glob(str(layout.k3_logs / "*" / f"k3-pass-{k}.log"))):
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                m = K3_ATTEMPT.match(line)
                if m and int(m.group(1)) == k:
                    out.append({"log": path, "attempt": int(m.group(2)), "group_tokens": int(m.group(3)), "tool16": m.group(4),
                                "card_class": m.group(6)})
    return out


def k3_geometry_check(layout: Layout, k: int, pd: Path, m: dict[str, Any], tool: str, cls: str, *, legacy: bool) -> tuple[list[str], dict[str, Any]]:
    """ONE tool, ONE card class, ONE group-token count over the whole pass (ruling part 3)."""

    problems = []
    corpus_gt = (m.get("corpus") or {}).get("group_tokens")
    arg_gt = (m.get("arguments") or {}).get("group_tokens")
    if corpus_gt is None or str(corpus_gt) != str(arg_gt):
        problems.append(f"manifest group tokens {corpus_gt} (corpus) vs {arg_gt} (arguments)")
    geo = load_json(Path(f"{pd}.geometry"))
    if geo is None and not legacy:
        problems.append(f"{pd}.geometry is missing (k3_lib records it at the pass's first start)")
    if geo is not None:
        for key, want in (("tool_sha256", tool), ("card_class", cls), ("group_tokens", corpus_gt)):
            if str(geo.get(key)) != str(want):
                problems.append(f"{pd}.geometry {key} {geo.get(key)} != {want}")
    attempts = k3_attempts(layout, k)
    gts = sorted({a["group_tokens"] for a in attempts})
    if gts and gts != [int(corpus_gt or -1)]:
        problems.append(f"mixed geometry within the pass: runner logs show group tokens {gts}, the manifest {corpus_gt}")
    tools16 = sorted({a["tool16"] for a in attempts if a["tool16"]})
    if tools16 and tools16 != [tool[:16]]:
        problems.append(f"mixed tools within the pass: runner logs show {tools16}")
    classes = sorted({a["card_class"] for a in attempts if a["card_class"]})
    if classes and classes != [cls]:
        problems.append(f"mixed card classes within the pass: runner logs show {classes}")
    return problems, {"group_tokens": corpus_gt, "geometry_file": geo, "attempts": len(attempts), "attempt_group_tokens": gts}


def k3_pass_state(layout: Layout, out: Path, k: int, records: dict[str, Any], b2: dict[str, Any]) -> dict[str, Any]:
    pd = out / f"pass-{k}-of-4"
    man = pd / "manifest.json"
    xt = out / "xtok" / f"pass-{k}" / "xtok_target.safetensors"
    st: dict[str, Any] = {"pass": k, "dir": str(pd), "xtok": str(xt), "state": "pending", "sealed": False, "problems": []}
    m = load_json(man)
    if m is None:
        st["why"] = "no manifest.json yet"
        return st
    if m.get("verdict") != "sealed":
        st["why"] = f"manifest verdict {m.get('verdict')!r}"
        return st
    tool = str(m.get("tool_sha256"))
    man_sha = sha256_file(man)
    st.update(sealed_by=tool, manifest_sha256=man_sha, tool_version=K3_TOOLS.get(tool))
    if tool not in K3_TOOLS:
        st.update(state="refused", problems=[f"sealed by {tool[:16]}, not a pinned K3 tool ({', '.join(t[:16] for t in K3_TOOLS)})"])
        return st
    ran_on = manifest_card_class(m)
    rec = (records.get("passes") or {}).get(str(k))
    legacy = rec is None
    if legacy:
        if not (tool == K3_V124_SHA and ran_on == K3_HOME_CLASS):
            st["why"] = f"sealed by {tool[:16]} on {ran_on} but not (yet) recorded in {layout.k3_pass_tools}"
            return st
        cls, gate_path, source = K3_HOME_CLASS, None, "legacy: the frozen 1.2.4 on GPU6, sealed by a queue that kept no record"
    else:
        cls, gate_path, source = str(rec.get("card_class")), rec.get("gate_receipt"), "k3_pass_tools.json"
        if rec.get("tool_sha256") != tool:
            st["problems"].append(f"record tool {str(rec.get('tool_sha256'))[:16]} != manifest tool {tool[:16]}")
        if rec.get("manifest_sha256") != man_sha:
            st["problems"].append(f"record manifest sha {str(rec.get('manifest_sha256'))[:16]} != the manifest now {man_sha[:16]}")
        if Path(str(rec.get("dir"))).resolve() != pd.resolve():
            st["problems"].append(f"record dir {rec.get('dir')} is not {pd}")
        if ran_on != cls:
            st["problems"].append(f"record card class {cls} != the class the manifest ran on ({ran_on})")
    gate_problems, gate = k3_gate_check(layout, tool, cls, gate_path, b2)
    geo_problems, geometry = k3_geometry_check(layout, k, pd, m, tool, cls, legacy=legacy)
    st["problems"] += gate_problems + geo_problems
    xr = load_json(xt.parent / "xtok_receipt.json") or {}
    xtok_built = xt.is_file() and bool(xr)
    st.update(card_class=cls, source=source, gate=gate, geometry=geometry, xtok_built=xtok_built,
              xtok_matches_pass=xtok_built and [c.get("sha256") for c in xr.get("cache") or []] == [man_sha]
              and (xr.get("target") or {}).get("sha256") == sha256_file(xt))
    if st["problems"]:
        st["state"] = "refused"
    else:
        st.update(state="accepted", sealed=True)
    return st


def k3_marker_check(layout: Layout, passes: list[dict[str, Any]]) -> tuple[bool, str]:
    """The seal marker is valid only for the passes it was written for: every pass accepted, its manifest and xtok
    files unchanged since k3-seal verified them."""

    marker = load_json(layout.k3_marker)
    if marker is None:
        return False, "no marker"
    by_pass = {int(p["pass"]): p for p in marker.get("passes") or []}
    if sorted(by_pass) != [0, 1, 2, 3]:
        return False, f"the marker names passes {sorted(by_pass)}"
    for p in passes:
        rec = by_pass[p["pass"]]
        if p["state"] != "accepted":
            return False, f"pass {p['pass']} is {p['state']}"
        if rec.get("k3_manifest_sha256") != p.get("manifest_sha256"):
            return False, f"pass {p['pass']}: the manifest changed since the seal (or the marker predates d2_plan 1.3.0)"
        if not Path(p["xtok"]).is_file() or sha256_file(Path(p["xtok"])) != rec.get("xtok_sha256"):
            return False, f"pass {p['pass']}: the xtok target changed since the seal"
    return True, "every pass accepted; manifests and xtok targets unchanged since the seal"


def k3_state(layout: Layout) -> dict[str, Any]:
    """1.3.0: the K3 passes A2 item 8a counts as sealed, the xtok targets, and the seal marker (see the module doc)."""

    tool_sha = sha256_file(layout.k3_frozen_tool) if layout.k3_frozen_tool.is_file() else None
    out = layout.k3_out()
    records = load_json(layout.k3_pass_tools) or {}
    b2 = b2_state(layout)
    passes = [k3_pass_state(layout, out, k, records, b2) for k in range(4)]
    marker_exists = layout.k3_marker.is_file()
    valid, why = k3_marker_check(layout, passes) if marker_exists else (False, "no marker")
    return {"frozen_tool": str(layout.k3_frozen_tool), "frozen_tool_sha256": tool_sha, "frozen_tool_is_v124": tool_sha == K3_V124_SHA,
            "out": str(out), "pass_tools": str(layout.k3_pass_tools), "b2": {"dir": b2["dir"], "ok": b2["ok"]}, "passes": passes,
            "sealed": sum(1 for p in passes if p["state"] == "accepted"),
            "refused": [{"pass": p["pass"], "problems": p["problems"]} for p in passes if p["state"] == "refused"],
            "pending": [{"pass": p["pass"], "why": p.get("why")} for p in passes if p["state"] == "pending"],
            "marker": str(layout.k3_marker), "marker_exists": marker_exists, "marker_valid": valid, "marker_check": why}


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


# --------------------------------------------------------------------------
# D2_DECISIONS_2026-10-15 (1.3.0): the v2 masks, the d2clean rows, the tool-call screen
# --------------------------------------------------------------------------


def version_tuple(v: Any) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)


def masks_v2_state(layout: Layout) -> dict[str, Any]:
    """d2_masks_v2 as d2_mask_build >= 1.2.0 `decisions` + `check-decisions` left it: ok / waiting / refused."""

    d = layout.masks_v2
    r = load_json(d / "build_receipt.json")
    if r is None:
        return {"state": "waiting", "dir": str(d), "why": f"{d} is not built (d2_mask_build >= 1.2.0 `decisions`)"}
    problems = []
    if r.get("kind") != "d2-decisions" or version_tuple(r.get("tool_version")) < MASK_BUILD_MIN:
        problems.append(f"build receipt kind {r.get('kind')!r} / version {r.get('tool_version')} is not a d2_mask_build >= 1.2.0 decisions build")
    if (r.get("decisions") or {}).get("sha256") != DECISIONS_SHA:
        problems.append(f"built for decisions {str((r.get('decisions') or {}).get('sha256'))[:16]}, not {DECISIONS_SHA[:16]}")
    files = {}
    for name in ("exclude_rows.json", "all_mask.json", "gt_mask.json", "heldout_rids.json"):
        path = d / name
        want = (r.get("outputs") or {}).get(name)
        got = sha256_file(path) if path.is_file() else None
        files[name] = {"path": str(path), "sha256": got}
        if got is None or got != want:
            problems.append(f"{path}: sha {str(got)[:16]} is not the build receipt's {str(want)[:16]}")
    for name, pin in MASK_SHAS.items():
        if (r.get("inputs") or {}).get(f"v1_{name}") != pin:
            problems.append(f"built on a v1 {name} other than the A2 pin {pin[:16]}")
    rechecks = r.get("rechecks") or {}
    if rechecks.get("hits") != 0:
        problems.append(f"the 0-hit re-checks found {rechecks.get('hits')} held-out row(s) in another D2 input")
    missing = [k for k in ("heldout_trained_positions_removed", "positions_after_v1_exclusion", "kd_positions_v1")
               if not isinstance((r.get("refill") or {}).get(k), int)]
    if missing:
        problems.append(f"the build receipt's refill block lacks {missing}")
    cc = r.get("count_check") or {}
    if not cc.get("ok") and not cc.get("reason"):
        problems.append(f"counts differ from expected_counts without a written reason: {cc.get('differences')}")
    out = {"dir": str(d), "files": files, "receipt_sha256": sha256_file(d / "build_receipt.json"), "tool_version": r.get("tool_version"),
           "counts": r.get("counts"), "count_check": cc, "rechecks": rechecks, "refill": r.get("refill")}
    if problems:
        return {**out, "state": "refused", "why": "; ".join(problems)}
    dic = r.get("doc_index_check")
    if not isinstance(dic, dict):
        return {**out, "state": "waiting", "why": f"check-decisions has not run on {d}"}
    if dic.get("ok") is not True:
        return {**out, "state": "refused", "why": f"check-decisions FAILED: {dic.get('problems')}"}
    return {**out, "state": "ok", "why": "built, counts as decided" + ("" if cc.get("ok") else f" (with reason: {cc.get('reason')})")
            + ", 0-hit re-checks, doc_index check OK"}


def clean_state(layout: Layout, inv: dict[str, Any]) -> dict[str, Any]:
    """Lane d2clean's clean registry version (d2_clean_repack.py, $OC/distill/corpus/text_clean_v1): the sealed registry
    rows that hold no held-out / BFCL-62 / twin document (used as they are, with their sealed Q38 caches) plus the other
    documents of the affected rows re-packed whole into text-clean-*.safetensors, each with its own Q38 cache; masks
    clean_exclude_rows.json (the affected sealed rows + v1's non-registry rows), clean_all_mask.json (A2 3b + the kept
    rows' X / re-packed positions), clean_gt_mask.json (A2 3c + D2-X2).  Usable ("sealed") only when -- the rule of
    d2_clean_repack `state` plus what d2_plan pins -- the version is built for the pinned decisions file and for the very
    sealed registry manifest D2 reads, verify_receipt.json is OK for this manifest file, the three masks are the files
    the manifest names, every re-pack Q38 cache and every sealed registry cache is sealed.  Anything else falls back to
    the span-mask rows (recorded), never a refusal: D2 never waits for this lane."""

    d = layout.clean_dir
    out: dict[str, Any] = {"dir": str(d), "sealed": False}
    man_path = d / "manifest.json"
    man = load_json(man_path)
    if man is None:
        return {**out, "why": f"no clean registry version at {d} (lane d2clean)"}
    problems = []
    out["manifest_sha256"] = man.get("manifest_sha256")
    if man.get("repack_schema") != CLEAN_REPACK_SCHEMA:
        problems.append(f"repack_schema {man.get('repack_schema')!r} != {CLEAN_REPACK_SCHEMA}")
    pins = man.get("inputs") or {}
    if (pins.get("decisions") or {}).get("sha256") != DECISIONS_SHA:
        problems.append(f"built for decisions {str((pins.get('decisions') or {}).get('sha256'))[:16]}, not {DECISIONS_SHA[:16]}")
    sealed_man = load_json(layout.text_manifest) or {}
    if (man.get("sealed_corpus") or {}).get("manifest_sha256") != sealed_man.get("manifest_sha256"):
        problems.append("built on another sealed registry manifest than distill/corpus/text/manifest.json")
    v = load_json(d / "verify_receipt.json") or {}
    if not (v.get("ok") is True and v.get("manifest_sha256") == man.get("manifest_sha256")
            and v.get("manifest_file_sha256") == sha256_file(man_path)):
        problems.append("verify_receipt.json is missing, not OK, or for another manifest")
    masks = {}
    for key, name in CLEAN_MASKS.items():
        m = (man.get("masks") or {}).get(name) or {}
        path = Path(str(m.get("path", "")))
        if path.resolve() != (d / name).resolve() or not path.is_file() or sha256_file(path) != m.get("sha256"):
            problems.append(f"{name}: missing or not the sha the manifest records")
        masks[key] = {"path": str(d / name), "sha256": m.get("sha256")}
    kd = (man.get("counts") or {}).get("kd_positions_clean_total")
    if not isinstance(kd, int) or kd <= 0:
        problems.append(f"counts.kd_positions_clean_total {kd!r}")
    out.update(masks=masks, kd_positions_clean_total=kd)
    if problems:
        return {**out, "invalid": True, "why": "clean registry not usable: " + "; ".join(problems)}
    registry, pending = [], []
    for r in inv["registry"]:  # the sealed rows the clean version keeps (its exclusion drops the affected ones)
        registry.append(r)
        if not r["sealed"]:
            pending.append(r["file"])
    for f in sorted(man.get("shards") or [], key=lambda f: int(f["rows"][0])):
        path = Path(f["path"])
        if not path.is_file() or sha256_file(path) != f["sha256"]:
            return {**out, "invalid": True, "why": f"clean registry not usable: {path} missing or not {f['sha256'][:16]}"}
        cs = cache_state(layout.cache_dir(f["sha256"]))
        registry.append({**cs, "file": f["file"], "row_range": [int(f["rows"][0]), int(f["rows"][1])], "corpus_sha256": f["sha256"]})
        if not cs["sealed"]:
            pending.append(f["file"])
    out["files"] = len(man.get("shards") or [])
    if pending:
        return {**out, "why": f"{len(pending)} Q38 cache(s) of the clean registry not sealed (first {pending[:3]})"}
    return {**out, "sealed": True, "registry": registry,
            "why": f"built, verified, {out['files']} re-pack files + the kept sealed rows, every Q38 cache sealed"}


def screen_state(layout: Layout, inv: dict[str, Any]) -> dict[str, Any]:
    """D2-X1 per_input toolcall_selfgen: the tool-call export (C1) screened (13-gram) against the rendered text of the
    5,391 held-out rows, the BFCL-62 tasks and Codex's 274 tasks; receipt beside the C1 corpus (schema
    omni-d2x1-toolcall-screen-v1: decisions_sha256, corpus_sha256 = the C1 corpus file, ngram 13, hits_in_corpus 0)."""

    c1 = inv["toolcall_c1"]
    s = load_json(layout.toolcall_screen)
    if s is None:
        return {"ok": False, "path": str(layout.toolcall_screen),
                "why": f"no 13-gram held-out screen receipt {layout.toolcall_screen} for the tool-call export (L-TOOL finalize)"}
    problems = []
    if s.get("schema") != SCREEN_SCHEMA:
        problems.append(f"schema {s.get('schema')!r}")
    if s.get("decisions_sha256") != DECISIONS_SHA:
        problems.append("another decisions file")
    if not c1.get("corpus_sha256") or s.get("corpus_sha256") != c1.get("corpus_sha256"):
        problems.append(f"screened corpus {str(s.get('corpus_sha256'))[:16]} is not the C1 corpus {str(c1.get('corpus_sha256'))[:16]}")
    if s.get("ngram") != 13 or s.get("hits_in_corpus") != 0:
        problems.append(f"ngram {s.get('ngram')} hits_in_corpus {s.get('hits_in_corpus')}")
    return {"ok": not problems, "path": str(layout.toolcall_screen), "sha256": sha256_file(layout.toolcall_screen),
            "why": "; ".join(problems) or "screened, 0 hits in the corpus"}


def decisions_state(layout: Layout, inv: dict[str, Any]) -> dict[str, Any]:
    """ok / waiting / refused: may a short data config that applies D2_DECISIONS_2026-10-15 be written now?"""

    st: dict[str, Any] = {"path": str(layout.decisions), "pinned_sha256": DECISIONS_SHA, "applied_receipt": str(layout.applied)}
    if not layout.decisions.is_file():
        return {**st, "state": "waiting", "why": f"no decisions file {layout.decisions}"}
    st["sha256"] = sha256_file(layout.decisions)
    if st["sha256"] != DECISIONS_SHA:
        return {**st, "state": "refused", "why": f"{layout.decisions} is {st['sha256'][:16]}, not the pinned {DECISIONS_SHA[:16]} (a new decision needs a new d2_plan)"}
    d = load_json(layout.decisions) or {}
    if not str(d.get("applied_receipt", "")).endswith(APPLIED_SUFFIX):
        return {**st, "state": "refused", "why": f"the decisions file names applied receipt {d.get('applied_receipt')}, not .../{APPLIED_SUFFIX}"}
    st.update(id=d.get("id"), rule_ids=(d.get("rulings") or {}).get("ids"))
    masks = masks_v2_state(layout)
    clean = clean_state(layout, inv)
    screen = screen_state(layout, inv)
    variant = "clean" if clean["sealed"] else "span-mask"
    reason = ("lane d2clean's clean registry is built, verified and cached: no held-out text even as attention context"
              if clean["sealed"] else
              f"span-mask rows (the sealed registry + d2_masks_v2): {clean['why']}; held-out text stays attention context in "
              "the shared packed rows (decisions residual_exposure)")
    st.update(masks_v2=masks, clean=clean, toolcall_screen=screen, variant=variant, variant_reason=reason)
    if masks["state"] == "refused":
        return {**st, "state": "refused", "why": f"d2_masks_v2: {masks['why']}"}
    if masks["state"] != "ok":
        return {**st, "state": "waiting", "why": f"d2_masks_v2: {masks['why']}"}
    if not screen["ok"]:
        return {**st, "state": "waiting", "why": f"tool-call screen: {screen['why']}"}
    return {**st, "state": "ok", "why": f"applicable ({variant})"}


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
    # 1.3.0: D2_DECISIONS_2026-10-15 is a start input: the v2 masks (or the clean rows), the re-checks, the tool-call screen
    inv["decisions"] = decisions_state(layout, inv)
    if inv["decisions"]["state"] != "ok":
        waiting.append(f"decisions ({inv['decisions']['state']}): {inv['decisions']['why']}")
    inv["start_inputs_waiting"] = waiting
    inv["k3_rows_ready"] = inv["k3"]["marker_valid"]
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


def supervised_tokens_of(path: Path) -> tuple[int, int, str]:
    """(real tokens, supervised tokens, unit) of a corpus: supervised = its target_mask's marked targets when it has
    one (an assistant_only source trains only there), else every real token."""

    from safetensors.numpy import load_file

    t = load_file(str(path))
    real = real_tokens_of(path)
    if "target_mask" in t:
        return real, int(t["target_mask"].astype(bool).sum()), "target_mask"
    return real, real, "real_tokens"


def registry_plan(layout: Layout, inv: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The registry entries and mask files of the variant decisions_state chose (1.3.0)."""

    dec = inv["decisions"]
    if dec.get("variant") == "clean":
        return dec["clean"]["registry"], {k: v["path"] for k, v in dec["clean"]["masks"].items()}
    files = dec["masks_v2"]["files"]
    return inv["registry"], {k: files[f"{k}.json"]["path"] for k in ("exclude_rows", "all_mask", "gt_mask")}


def refill_plan(layout: Layout, inv: dict[str, Any], *, registry_positions: int, registry_budget: int) -> dict[str, Any]:
    """D2-X1-refill (decisions refill_policy): the target, what ranks 1-4 supply when the config is written, the shortfall.
    D2 never waits on a refill source: a rank whose cache is not sealed now supplies nothing (trainer 1.3.4 cannot skip
    an unsealed deferred source at its threshold -- it checkpoints and waits -- so no refill source is deferred)."""

    dec = inv["decisions"]
    extra_registry_positions = registry_positions - registry_budget
    refill = dec["masks_v2"]["refill"]
    after_v1 = int(refill["positions_after_v1_exclusion"])
    draw = REGISTRY_DRAW + ONPOLICY_FALLBACK
    f_v1 = min(1.0, draw / after_v1)
    if dec["variant"] == "clean":
        kd_v1, kd_clean = int(refill["kd_positions_v1"]), int(dec["clean"]["kd_positions_clean_total"])
        f_clean = min(1.0, draw / max(registry_positions, 1))
        target = max(0, int(round(kd_v1 * f_v1 - kd_clean * f_clean)))
        basis = (f"KD-carrying registry positions of the A2 (v1) draw {kd_v1:,} x {f_v1:.4f} minus those of the clean draw "
                 f"{kd_clean:,} x {f_clean:.4f} (d2_clean_repack counts.kd_positions_clean_total)")
    else:
        removed = int(refill["heldout_trained_positions_removed"])
        target = int(round(removed * f_v1))
        basis = (f"{removed:,} trained registry positions D2-X1 removes x min(1, planned registry draw {draw:,} (18.85M "
                 f"off-policy + 1.15M on-policy fallback) / {after_v1:,} positions after A2 3(a)) = {f_v1:.4f}; the decisions "
                 "file's 'about 2.96M' assumed an 18.85M draw")
    ranks: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    c1 = inv["toolcall_c1"]
    real, sup, unit = supervised_tokens_of(Path(c1["corpus"])) if c1.get("sealed") else (0, 0, "not sealed")
    beyond = max(0, real - TOOL_SELFGEN_BASE_TOKENS)
    r1 = int(round(sup * beyond / real)) if real else 0
    ranks.append({"rank": 1, "source": "tool-selfgen (C1, already a D2 source)", "real_tokens": real, "supervised": sup, "unit": unit,
                  "beyond_a2_2a": beyond, "supplied": r1})
    left = max(0, target - r1)
    for cand in REFILL_CANDIDATES:
        corpus = layout.refill_dir / cand["file"]
        entry = {"rank": cand["rank"], "source": cand["name"], "corpus": str(corpus), "supplied": 0}
        state = corpus_cache(layout, corpus)
        if not state["sealed"]:
            ranks.append({**entry, "why": f"not used: {state.get('reason')}"})
            continue
        real, sup, unit = supervised_tokens_of(corpus)
        tokens = real
        if cand["cap"] == "remainder":
            if left <= 0:
                ranks.append({**entry, "why": "not used: the target is already met"})
                continue
            tokens = min(real, int(math.ceil(left * real / max(sup, 1))))
        supplied = int(round(sup * tokens / real)) if real else 0
        src = {"name": cand["name"], "kind": "text", "qtext_cache": state["dir"], "budget_unit": "real_tokens", "tokens": tokens,
               "exclude_rows": str(layout.masks / "exclude_rows.json")}
        if cand["supervision"] == "assistant_only":
            src.update(supervision="assistant_only", w_ul=0.0)
        sources.append(src)
        ranks.append({**entry, "real_tokens": real, "supervised": sup, "unit": unit, "tokens_budgeted": tokens, "supplied": supplied})
        left = max(0, left - supplied)
    r4 = min(left, max(0, extra_registry_positions))
    ranks.append({"rank": 4, "source": "undrawn registry positions", "available": max(0, extra_registry_positions), "supplied": r4,
                  "why": "the registry budget is one epoch of the rows left after the exclusion" if extra_registry_positions <= 0 else
                         "added to the registry budget"})
    left = max(0, left - r4)
    supplied = sum(r["supplied"] for r in ranks)
    return {"rule_id": "D2-X1-refill", "target_tokens": target, "target_basis": basis, "ranks": ranks, "sources": sources,
            "supplied_estimate": supplied, "shortfall_estimate": max(0, target - supplied), "rank4_positions": r4,
            "census": "the trainer census (supervised_positions_applied) proves the totals; this block is the plan"}


def build_data_config(layout: Layout, inv: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "long":
        long_cache = inv["long"]
        return {"schema": "omni-d2-data-config-v1", "kind": "long", "written_by": f"{TOOL_NAME} {TOOL_VERSION}",
                "sources": [{"name": "long", "kind": "text", "qtext_cache": long_cache["dir"]}]}
    dec = inv["decisions"]
    registry, mask_files = registry_plan(layout, inv)
    excl = Path(mask_files["exclude_rows"])
    exclusion = json.loads(excl.read_text(encoding="utf-8"))
    excluded_by_sha = {sha: len(e["rows"]) for sha, e in exclusion["corpora"].items()}
    sources = []
    registry_rows = sum((r["row_range"][1] - r["row_range"][0]) - excluded_by_sha.get(r["corpus_sha256"], 0) for r in registry)
    registry_budget = min(REGISTRY_DRAW + ONPOLICY_FALLBACK, registry_rows * SEQ)
    refill = refill_plan(layout, inv, registry_positions=registry_rows * SEQ, registry_budget=registry_budget)
    registry_budget += refill["rank4_positions"]
    dropped = []
    for r in registry:
        rows = (r["row_range"][1] - r["row_range"][0]) - excluded_by_sha.get(r["corpus_sha256"], 0)
        if rows <= 0:  # every row excluded (the clean version drops whole sealed files): the trainer refuses an empty source
            dropped.append(f"registry-{r['file'].split('.')[0]}")
            continue
        sources.append({"name": f"registry-{r['file'].split('.')[0]}", "kind": "text", "qtext_cache": r["dir"],
                        "tokens": int(round(rows * SEQ * registry_budget / (registry_rows * SEQ))), "budget_unit": "positions",
                        "exclude_rows": str(excl), "gt_mask": mask_files["gt_mask"], "all_mask": mask_files["all_mask"]})
    excl = layout.masks / "exclude_rows.json"  # A2 3(a) (v1): the K3 / self-gen twin rows, unchanged by D2_DECISIONS
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
    sources += refill["sources"]
    masks_used = {k: {"path": p, "sha256": sha256_file(Path(p))} for k, p in mask_files.items()}
    decisions = {"id": dec.get("id"), "path": dec["path"], "sha256": dec["sha256"], "rule_ids": dec.get("rule_ids"),
                 "variant": dec["variant"], "variant_reason": dec["variant_reason"], "registry_masks": masks_used,
                 "a2_3a_exclude_rows_for_k3_and_selfgen": {"path": str(excl), "sha256": sha256_file(excl)},
                 "mask_build_receipt_sha256": dec["masks_v2"]["receipt_sha256"],
                 "clean_manifest_sha256": dec["clean"].get("manifest_sha256") if dec["variant"] == "clean" else None,
                 "toolcall_screen_sha256": dec["toolcall_screen"].get("sha256"),
                 "applied_receipt": str(layout.applied)}
    return {"schema": "omni-d2-data-config-v1", "kind": "short", "written_by": f"{TOOL_NAME} {TOOL_VERSION}",
            "registry_budget_positions": registry_budget, "registry_positions_after_exclusion": registry_rows * SEQ,
            "onpolicy": ("no on-policy sampler exists: the 1.15M on-policy tokens come from the off-policy registry "
                         "(prereg section 3 D2 Data), capped at one epoch of the rows left after the exclusions"),
            "rungs": list(RUNGS), "k3_defer_until_tokens": K3_DEFER_TOKENS, "deferred_rows": deferred_rows,
            "decisions": decisions, "refill": {k: v for k, v in refill.items() if k != "sources"},
            "registry_sources_dropped": dropped,
            "sources": sources}


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
    if args.kind == "short" and inv["decisions"]["state"] == "refused":
        raise Refusal(f"D2_DECISIONS_2026-10-15 cannot be applied: {inv['decisions']['why']}")
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
             "masks_ok": inv["masks"]["ok"], "k3_passes_sealed": inv["k3"]["sealed"], "k3_marker": inv["k3"]["marker_valid"],
             "decisions": inv["decisions"]["state"], "registry_variant": inv["decisions"].get("variant"),
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
        if state["refused"] or (state["marker_exists"] and not state["marker_valid"]):
            return 2
        return 0 if state["marker_valid"] else EXIT_WAITING
    if state["marker_valid"]:
        emit("k3-seal", {**state, "note": "marker exists and is valid; nothing to do"})
        return 0
    if state["marker_exists"]:
        raise Refusal(f"{layout.k3_marker} exists but does not match the passes: {state['marker_check']} (the D2 data config "
                      "would load K3 rows the seal never verified; move it aside after a decision)")
    if state["refused"]:
        raise Refusal("K3 pass(es) sealed but not acceptable (A2 item 8a): " + json.dumps(state["refused"])[:1500])
    if state["sealed"] < 4:
        emit("k3-seal", {**state, "waiting": f"{state['sealed']} of 4 K3 passes accepted (A2 item 8a)"})
        return EXIT_WAITING
    if not state["frozen_tool_is_v124"]:
        raise Refusal(f"{layout.k3_frozen_tool} is {str(state['frozen_tool_sha256'])[:16]}, not the B2-proven 1.2.4 "
                      f"{K3_V124_SHA[:16]} that builds the xtok targets")
    sys.path.insert(0, str(Path(args.tools)))
    import streamed_kd_trainer as T  # the trainer's own reader proves the join

    built = []
    for p in state["passes"]:
        k = p["pass"]
        src = layout.k3_corpus / f"pass-{k}"
        out = Path(p["xtok"]).parent
        if p["xtok_built"] and not p["xtok_matches_pass"]:
            stale = out.with_name(f"{out.name}.stale-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
            os.replace(out, stale)  # built from another sealing of this pass: kept, never overwritten
            p["xtok_built"] = False
            p["xtok_moved_aside"] = str(stale)
        if not p["xtok_built"]:
            # the frozen 1.2.4 builds every pass's xtok: cmd_xtok and everything it calls are byte-identical in 1.3.0
            cmd = [sys.executable, str(layout.k3_frozen_tool), "xtok", "--cache-dir", p["dir"],
                   "--boundary-map", str(src / "boundary_map.json"), "--k3-ids", str(src / "k3_input_ids.safetensors"),
                   "--qwen-ids", str(src / "qwen_input_ids.safetensors"), "--out", str(out)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            (out.parent / f"xtok-pass-{k}.log").parent.mkdir(parents=True, exist_ok=True)
            (out.parent / f"xtok-pass-{k}.log").write_text(run.stdout + run.stderr, encoding="utf-8")
            if run.returncode != 0:
                raise Refusal(f"xtok pass {k} failed rc {run.returncode}: {run.stderr[-400:]}")
        receipt = load_json(out / "xtok_receipt.json") or {}
        if [c.get("sha256") for c in receipt.get("cache") or []] != [p["manifest_sha256"]]:
            raise Refusal(f"xtok pass {k}: its receipt names K3 manifest(s) {[c.get('sha256') for c in receipt.get('cache') or []]}, "
                          f"not the accepted pass {p['manifest_sha256'][:16]}")
        cache = T.QTextCache(layout.cache_dir(sha256_file(src / "qwen_input_ids.safetensors")))
        xtok = T.XtokTargets(Path(p["xtok"]), "qwen_input_ids")
        proof = xtok.verify_against(cache)
        built.append({"pass": k, "xtok_sha256": xtok.sha256, "xtok_receipt_sha256": sha256_file(out / "xtok_receipt.json"),
                      "pairs": xtok.pairs, "provenance": proof.get("method"), "q38_cache_manifest_sha256": cache.manifest_sha256,
                      "k3_manifest_sha256": p["manifest_sha256"], "k3_tool_sha256": p["sealed_by"], "card_class": p["card_class"],
                      "source": p["source"], "gate": p["gate"], "group_tokens": p["geometry"]["group_tokens"]})
    marker = {"schema": "omni-d2-k3-seal-v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "utc": utc(),
              "frozen_k3_tool_sha256": state["frozen_tool_sha256"], "xtok_tool_sha256": state["frozen_tool_sha256"],
              "pass_tools": {"path": str(layout.k3_pass_tools),
                             "sha256": sha256_file(layout.k3_pass_tools) if layout.k3_pass_tools.is_file() else None},
              "b2": state["b2"], "passes": built,
              "rule": "A2 items 8 + 8a (draft): every pass sealed by a pinned K3 tool behind its passing gate, one geometry per "
                      "pass, xtok verified against its Q38 cache, before the first K3 row"}
    sha = write_json_atomic(layout.k3_marker, marker)
    emit("k3-seal", {"marker": str(layout.k3_marker), "sha256": sha, "passes": built})
    return 0


# --------------------------------------------------------------------------
# D2_DECISIONS applied receipt (1.3.0)
# --------------------------------------------------------------------------


def check_config_applies(layout: Layout, cfg: dict[str, Any]) -> list[str]:
    """What a short data config must carry for D2_DECISIONS_2026-10-15 to count as applied by it."""

    problems = []
    if not str(cfg.get("written_by", "")).startswith(f"{TOOL_NAME} ") or version_tuple(str(cfg.get("written_by")).split()[-1]) < (1, 3, 0):
        problems.append(f"written by {cfg.get('written_by')!r}, not d2_plan >= 1.3.0")
    dec = cfg.get("decisions") or {}
    if dec.get("sha256") != DECISIONS_SHA:
        problems.append(f"the config applies decisions {str(dec.get('sha256'))[:16]}, not {DECISIONS_SHA[:16]}")
    masks = dec.get("registry_masks") or {}
    for key in ("exclude_rows", "all_mask", "gt_mask"):
        m = masks.get(key) or {}
        path = Path(str(m.get("path", "")))
        if not path.is_file() or sha256_file(path) != m.get("sha256"):
            problems.append(f"registry {key} {path} missing or changed since the config was written")
    if dec.get("variant") == "span-mask":
        v2 = masks_v2_state(layout)
        if v2["state"] != "ok":
            problems.append(f"span-mask config but d2_masks_v2 is {v2['state']}: {v2['why']}")
        elif any((masks.get(k) or {}).get("sha256") != v2["files"][f"{k}.json"]["sha256"] for k in ("exclude_rows", "all_mask", "gt_mask")):
            problems.append("the config's registry masks are not the d2_masks_v2 files")
    elif dec.get("variant") != "clean":
        problems.append(f"variant {dec.get('variant')!r}")
    registry = [s for s in cfg.get("sources") or [] if str(s.get("name", "")).startswith("registry-")]
    if not registry:
        problems.append("no registry source")
    for s in registry:
        for key in ("exclude_rows", "all_mask", "gt_mask"):
            if s.get(key) != (masks.get(key) or {}).get("path"):
                problems.append(f"{s['name']}: {key} {s.get(key)} is not the variant's {(masks.get(key) or {}).get('path')}")
        if not cache_state(Path(s["qtext_cache"]))["sealed"]:
            problems.append(f"{s['name']}: Q38 cache {s['qtext_cache']} is not sealed")
    if not isinstance(cfg.get("refill"), dict) or cfg["refill"].get("rule_id") != "D2-X1-refill":
        problems.append("no D2-X1-refill block")
    return problems


def applied_check(layout: Layout, data: Path) -> tuple[int, dict[str, Any]]:
    """(rc, detail): 0 = the applied receipt is for exactly this config and the pinned decisions; 10 = no receipt; 2 = mismatch."""

    r = load_json(layout.applied)
    if r is None:
        return EXIT_WAITING, {"applied_receipt": str(layout.applied), "why": "no applied receipt (d2_plan decisions-apply --data)"}
    problems = []
    if (r.get("decisions") or {}).get("sha256") != DECISIONS_SHA or not layout.decisions.is_file() \
            or sha256_file(layout.decisions) != DECISIONS_SHA:
        problems.append("the receipt or the decisions file is not the pinned decisions")
    want = sha256_file(data) if data.is_file() else None
    if (r.get("data_config") or {}).get("sha256") != want:
        problems.append(f"the receipt is for data config {str((r.get('data_config') or {}).get('sha256'))[:16]}, not {str(want)[:16]} ({data})")
    for key, m in ((r.get("data_config_decisions") or {}).get("registry_masks") or {}).items():
        if not Path(m["path"]).is_file() or sha256_file(Path(m["path"])) != m["sha256"]:
            problems.append(f"registry {key} {m['path']} changed since the receipt")
    detail = {"applied_receipt": str(layout.applied), "sha256": sha256_file(layout.applied), "variant": r.get("variant"),
              "problems": problems}
    return (2 if problems else 0), detail


def cmd_decisions(args: argparse.Namespace, layout: Layout) -> int:
    if args.command == "decisions-state" and args.data:
        rc, detail = applied_check(layout, Path(args.data))
        emit("decisions-state", detail)
        return rc
    inv = inventory(layout)
    dec = inv["decisions"]
    brief = {k: dec.get(k) for k in ("state", "why", "sha256", "variant", "variant_reason")}
    brief.update(masks_v2=(dec.get("masks_v2") or {}).get("state"), clean=(dec.get("clean") or {}).get("why"),
                 toolcall_screen=(dec.get("toolcall_screen") or {}).get("why"))
    if args.command == "decisions-state":
        emit("decisions-state", brief)
        return {"ok": 0, "waiting": EXIT_WAITING}.get(dec["state"], 2)
    # decisions-apply --data CONFIG
    if dec["state"] == "refused":
        raise Refusal(f"decisions-apply: {dec['why']}")
    if dec["state"] != "ok":
        emit("decisions-apply", {**brief, "waiting": dec["why"]})
        return EXIT_WAITING
    data = Path(args.data)
    cfg = json.loads(data.read_text(encoding="utf-8"))
    problems = check_config_applies(layout, cfg)
    if problems:
        raise Refusal("decisions-apply: the data config does not apply D2_DECISIONS_2026-10-15: " + "; ".join(problems)[:1500])
    data_sha = sha256_file(data)
    old = load_json(layout.applied)
    if old is not None and (old.get("data_config") or {}).get("sha256") != data_sha:
        if not args.replace:
            raise Refusal(f"{layout.applied} is for data config {str((old.get('data_config') or {}).get('sha256'))[:16]}, not "
                          f"{data_sha[:16]}; give --replace (only before D2's first step) to archive it")
        os.replace(layout.applied, layout.applied.with_name(f"{layout.applied.name}.superseded-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
    m = dec["masks_v2"]
    variant = cfg["decisions"]["variant"]
    receipt = {"schema": "omni-d2-decisions-applied-v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
               "tool_sha256": sha256_file(Path(__file__)), "utc": utc(),
               "decisions": {"path": str(layout.decisions), "sha256": DECISIONS_SHA, "id": dec.get("id"), "rule_ids": dec.get("rule_ids")},
               "data_config": {"path": str(data), "sha256": data_sha}, "data_config_decisions": cfg["decisions"],
               "variant": variant, "variant_reason": cfg["decisions"]["variant_reason"],
               "variant_now": {"variant": dec["variant"], "note": None if dec["variant"] == variant else
                               "the preferred variant changed after the config was written; the config (written once) keeps its own"},
               "mask_build": {"dir": m["dir"], "receipt_sha256": m["receipt_sha256"], "tool_version": m["tool_version"],
                              "files": m["files"]},
               "counts_measured": m["counts"], "count_check": m["count_check"], "rechecks": m["rechecks"],
               "toolcall_screen": dec["toolcall_screen"], "clean": {k: v for k, v in dec["clean"].items() if k != "registry"},
               "refill": cfg["refill"],
               "apply_before_first_step": {
                   "1_spans_built_and_checked": f"d2_mask_build {m['tool_version']} decisions + check-decisions OK ({m['dir']})",
                   "2_rechecks_and_toolcall_screen": f"0 hits; screen {dec['toolcall_screen']['why']}",
                   "3_no_loss_rows_dropped_refill_target": f"{(m['counts'] or {}).get('no_loss_rows')} rows; refill target "
                                                          f"{cfg['refill']['target_tokens']:,}",
                   "4_refill_declared": f"supplied {cfg['refill']['supplied_estimate']:,}, shortfall {cfg['refill']['shortfall_estimate']:,} (plan)",
                   "5_a2_revised": a2_state(layout.prereg) if layout.prereg.is_file() else {"prereg": str(layout.prereg), "missing": True},
                   "6_this_receipt": "d2_launch.sh trains only when decisions-state --data says this receipt is for its config"}}
    sha = write_json_atomic(layout.applied, receipt)
    emit("decisions-apply", {"applied_receipt": str(layout.applied), "sha256": sha, "variant": variant, "data_config_sha256": data_sha,
                             "refill_target": cfg["refill"]["target_tokens"], "refill_shortfall": cfg["refill"]["shortfall_estimate"]})
    return 0


def partial_data_config(layout: Layout) -> dict[str, Any]:
    """d2_launch preflight's check config while inputs are pending: the sealed registry sources with the masks the run
    will use (d2_masks_v2 once it is built, else the v1 masks) and self-gen."""

    inv = inventory(layout)
    v2 = inv["decisions"].get("masks_v2") or {}
    if v2.get("state") == "ok":
        mask_files = {k: v2["files"][f"{k}.json"]["path"] for k in ("exclude_rows", "all_mask", "gt_mask")}
    else:
        mask_files = {k: str(layout.masks / f"{k}.json") for k in ("exclude_rows", "all_mask", "gt_mask")}
    sources = [{"name": f"registry-{r['file'].split('.')[0]}", "kind": "text", "qtext_cache": r["dir"], "budget_unit": "positions",
                "tokens": (r["row_range"][1] - r["row_range"][0]) * SEQ, **mask_files}
               for r in inv["registry"] if r["sealed"]]
    if inv["selfgen"]["sealed"]:
        sources.append({"name": "selfgen", "kind": "text", "qtext_cache": inv["selfgen"]["dir"], "budget_unit": "real_tokens",
                        "tokens": int(inv["selfgen"]["tokens"]), "exclude_rows": str(layout.masks / "exclude_rows.json")})
    return {"sources": sources}


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
    ds = sub.add_parser("decisions-state", help="1.3.0: D2_DECISIONS applicable now (no --data), or applied to --data")
    ds.add_argument("--data", help="a short data config: is the applied receipt for exactly this file?")
    da = sub.add_parser("decisions-apply", help="1.3.0: write the D2_DECISIONS applied receipt for a short data config")
    da.add_argument("--data", required=True)
    da.add_argument("--replace", action="store_true", help="archive a receipt for another config (only before D2 step 1)")
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
        if args.command in ("decisions-state", "decisions-apply"):
            return cmd_decisions(args, layout)
    except Refusal as refusal:
        print(f"REFUSED: {refusal.message}", file=sys.stderr, flush=True)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
