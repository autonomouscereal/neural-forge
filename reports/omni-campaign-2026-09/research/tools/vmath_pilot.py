#!/usr/bin/env python3
"""vmath_pilot.py -- the verified-math synthetic-generation pilot (G2, k = 2, thinking on) on a guest P40 lane.

WHY THIS FILE EXISTS.  Research lane (Claude), 2026-10-15, pilot "Data 3" of the data/training program as amended by the
adversarial critique (research/DATA_AND_TRAINING_PROGRAM_2026-10-14.md section 6 step 6): the parent's own Q4_K_M samples
k = 2 thinking-on solutions per math prompt; only verified samples become training rows.  Verifier-filtered best-of-k is
the one lane that can add signal beyond the parent's average output, and closed reasoning traces are the scarcest class
in the registry (150 rows).  The pilot measures yield and acceptance before any growth budget is set.

WHAT IS REUSED, NOT RE-IMPLEMENTED.
  * Sampling is `selfgen_sampler.py run` 1.2.0 unchanged (journal, resume, shards, EOG labels, manifest): this tool only
    writes a prompt file in its schema (rid, route, prompt_text, prompt_tokens, max_new_tokens) + a manifest whose
    file.sha256 / prompts count its `load_prompts` checks.  k samples of one problem are k prompt records with rids
    <source>~k0..k<k-1>, so each gets its own seed sha256(base_seed:rid) and the k records stay adjacent (a token-target
    cut keeps whole problems).
  * Serving is `selfgen_sampler.py serve` / `stop` (card <= 2,000 MiB, launch + allocator receipts); the claim protocol is
    `d1_seal_lib.claim_state` (imported by path), the same rule every Stage D queue reads p40.claims with.
  * The eval screen is quality_scorer.build_screen = distill_corpus_build.build_review_index + eval split + toolcall
    bank + Codex corpus-schema-v2 validation/test.
  * Answers: quality_scorer.extract_reference / extract_final_answer / answers_equal (numbers: exact Fraction, Codex
    evaluation.grade_arithmetic's tolerance 0; else normalized LaTeX string equality; anything else is "undecidable" and
    never counted as a pass).

THE PROMPT SET (measured 2026-10-15, not assumed).  The program doc planned "the 515 registry math rows with extractable
answers".  That count came from audit4's loose FINAL regex; read row by row, 127 of them are teacher-JSON wrappers and 333
carry the generic instruction "Solve the source-grounded math problem ..." with the problem inside the response, so no
prompt can be built from them.  Two strata are therefore sampled, and reported apart:
  R  "referenced": the problem is in the instruction AND the source response gives a strict reference (balanced
     \\boxed{} or an "Answer:" line).  Accepted = closed think + EOS + extracted answer equal to the reference.
  A  "agreement": route math, problem in the instruction, no strict reference.  Accepted = closed + EOS + every one of
     the k answers extracted and pairwise equal (k-sample agreement: weaker evidence than a reference; labelled so).
Both strata: carve split `train` only, no screen hit, not a duplicate member, prompt <= --max-prompt-tokens.  The user
turn is the instruction + "Please reason step by step, and put your final answer within \\boxed{}." rendered by the
parent's pinned chat template with enable_thinking=True and the generation prompt.

QUEUE DECISIONS (testable here, called by vmath_queue.sh):
  gate         0 = start on the printed GPUs; 10 = wait (every reason printed).  A GPU is usable when no OTHER lane's
               live claim names it (own-lane lines are ignored so a crashed session can resume), its gpuN.lock is free,
               nvidia-smi shows < 2,000 MiB and no compute app.  The queue also needs: MemAvailable >= mem-base + per-card
               x cards, no YIELD file, and NOT (d1.sealed present and d1.reviewed absent): D1's review may need P40 1+2+3
               and v1 comes first.
  yield-check  0 = keep running; 20 = yield now (YIELD file, the D1-review condition above, a foreign claim line naming
               a held GPU, or MemAvailable below the floor).

Subcommands
-----------
  prompts      scores.jsonl (quality_scorer registry run) + staging rows -> prompts.jsonl, references.jsonl, manifest.json
  gate         see above
  yield-check  see above
  verify       a selfgen_sampler run dir -> verified.jsonl, per-sample verdicts, summary.json, and an intake batch
               (manifest.json schema omni.intake.batch.v1 + rows.jsonl) for quality_scorer `batch`

FAILURES ARE FUNCTIONAL, NUMBERS ARE NOT.  A template or tokenizer that is not the pinned one, a run journal whose rids
are not in the prompt file, a prompt that re-tokenizes to a different length: non-zero exit.  Acceptance and accuracy
are diagnostics.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

TOOL_NAME = "vmath_pilot"
TOOL_VERSION = "1.0.2"   # 1.0.2 (2026-10-15): the gate serves as many cards as memory fits; 1.0.1: harness hooks --meminfo / --gpu-state-json; 1.0.0 first version, pilot Data 3
PROMPT_SCHEMA = "omni-research.vmath-prompts.v1"
SUMMARY_SCHEMA = "omni-research.vmath-summary.v1"
INTAKE_SCHEMA = "omni.intake.batch.v1"

HERE = Path(__file__).resolve().parent
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
DEFAULT_PARENT = Path("/home/cereal/models/Qwen3.8-27B-Uncensored")
DEFAULT_ROWS = OC / "distill" / "corpus" / "staging" / "rows.jsonl"
DEFAULT_CLAIMS = OC / "distill" / "queues" / "p40.claims"
DEFAULT_QUEUES = OC / "distill" / "queues"
DEFAULT_SEAL_LIB = OC / "distill" / "tools" / "d1_seal_lib.py"

QWEN_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
CHAT_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
BOXED_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
ROUTE = "math_verified"
DATASET_ID = "selfgen_vmath_q38_2026-10-15"
LANE = "research-vmath"
CARD_BUSY_MIB = 2000
EXCLUDE_FLAGS = ("screen_hit", "exact_duplicate_member", "near_duplicate_member", "teacher_json_wrapper", "empty_response",
                 "mojibake", "bfcl62_holdout", "pipeline_selfreference")


class PilotError(RuntimeError):
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return file_sha256(path), n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tool_identity() -> dict[str, Any]:
    me = Path(__file__).resolve()
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "path": str(me), "sha256": file_sha256(me)}


def load_by_path(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise PilotError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def qs() -> Any:
    return load_by_path("quality_scorer", HERE / "quality_scorer.py")


# ------------------------------------------------------------------------------------------- prompts
class Renderer:
    """Pinned tokenizer.json + chat template, thinking ON with the generation prompt."""

    def __init__(self, parent_dir: Path, allow_unpinned: bool = False) -> None:
        from tokenizers import Tokenizer

        self.parent_dir = Path(parent_dir)
        self.tokenizer_sha256 = file_sha256(self.parent_dir / "tokenizer.json")
        self.template_sha256 = file_sha256(self.parent_dir / "chat_template.jinja")
        if not allow_unpinned and (self.tokenizer_sha256 != QWEN_TOKENIZER_SHA256 or self.template_sha256 != CHAT_TEMPLATE_SHA256):
            raise PilotError(f"tokenizer {self.tokenizer_sha256[:16]} / template {self.template_sha256[:16]} are not the pinned parent's")
        self.fast = Tokenizer.from_file(str(self.parent_dir / "tokenizer.json"))
        self.fast.no_truncation()
        self.fast.no_padding()
        from transformers import AutoTokenizer

        self.hf = AutoTokenizer.from_pretrained(str(self.parent_dir))

    def render(self, user: str) -> str:
        return str(self.hf.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True))

    def count(self, text: str) -> int:
        return len(self.fast.encode(text, add_special_tokens=False).ids)

    def report(self) -> dict[str, Any]:
        return {"tokenizer_sha256": self.tokenizer_sha256, "chat_template_sha256": self.template_sha256,
                "render": "apply_chat_template([user], add_generation_prompt=True, enable_thinking=True)",
                "add_special_tokens": False}


def select_sources(scores: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Stratum R (referenced) and A (agreement) candidates from quality_scorer records."""
    out: dict[str, list[dict[str, Any]]] = {"R": [], "A": []}
    for rec in scores:
        ver = rec.get("verify") or {}
        if rec.get("split") != "train" or not ver.get("problem_in_instruction"):
            continue
        if set(rec.get("flags") or []) & set(EXCLUDE_FLAGS):
            continue
        if ver.get("kind") == "math_ref":
            out["R"].append(rec)
        elif ver.get("kind") == "math_noref" and rec.get("route") == "math":
            out["A"].append(rec)
    return out


def build_prompts(scores_path: Path, rows_path: Path, renderer: Any, *, k: int, max_prompts: int, seed: int,
                  max_prompt_tokens: int, max_new_tokens: int, screen: Any = None) -> dict[str, Any]:
    q = qs()
    strata = select_sources(q.read_jsonl(scores_path))
    wanted = {r["rid"]: s for s, recs in strata.items() for r in recs}
    rows = {}
    for row in q.read_jsonl(rows_path):
        if row["rid"] in wanted:
            rows[row["rid"]] = row
    rng = random.Random(seed)
    ordered: list[tuple[str, dict[str, Any]]] = []
    drop = collections.Counter()
    for stratum in ("R", "A"):
        recs = sorted(strata[stratum], key=lambda r: r["rid"])
        rng.shuffle(recs)
        for rec in recs:
            row = rows.get(rec["rid"])
            if row is None:
                drop["row_missing"] += 1
                continue
            ordered.append((stratum, row))
    prompts: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    taken = collections.Counter()
    for stratum, row in ordered:
        if len(refs) >= max_prompts:
            break
        user = (row["instruction"] or "").strip() + "\n\n" + BOXED_SUFFIX
        if screen is not None and screen.hits(user):
            drop["screen_hit_prompt"] += 1
            continue
        text = renderer.render(user)
        n = renderer.count(text)
        if n > max_prompt_tokens:
            drop["prompt_too_long"] += 1
            continue
        ref, rule = q.extract_reference(row.get("response") or "")
        if stratum == "R" and ref is None:
            drop["reference_lost"] += 1
            continue
        refs.append({"source_rid": row["rid"], "stratum": stratum, "reference": ref, "reference_rule": rule,
                     "dataset_id": row.get("dataset_id")})
        taken[stratum] += 1
        for j in range(k):
            prompts.append({"rid": f"{row['rid']}~k{j}", "source_rid": row["rid"], "sample_index": j, "stratum": stratum,
                            "route": ROUTE, "dataset_id": row.get("dataset_id"), "uid": row.get("uid"),
                            "record_id": row.get("record_id"), "prompt_text": text, "prompt_tokens": n,
                            "max_new_tokens": int(max_new_tokens), "messages": [{"role": "user", "content": user}]})
    return {"prompts": prompts, "references": refs, "taken": dict(taken), "dropped": dict(drop),
            "candidates": {s: len(v) for s, v in strata.items()}}


def cmd_prompts(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    if (out_dir / "manifest.json").exists():
        raise PilotError(f"{out_dir} already has a sealed prompt set (never overwritten)")
    renderer = Renderer(Path(args.parent_dir), allow_unpinned=args.allow_unpinned)
    screen, screen_info = None, {"enabled": False}
    if not args.no_screen:
        q = qs()
        screen, screen_info = q.build_screen(Path(args.corpus_tools), Path(args.eval_split), Path(args.toolcall_bank),
                                             [Path(p) for p in args.codex_split])
        screen_info["enabled"] = True
    built = build_prompts(Path(args.scores), Path(args.rows), renderer, k=args.k, max_prompts=args.max_prompts,
                          seed=args.seed, max_prompt_tokens=args.max_prompt_tokens, max_new_tokens=args.max_new_tokens,
                          screen=screen)
    if not built["prompts"]:
        raise PilotError(f"no prompt survived selection: {built}")
    psha, n = write_jsonl(out_dir / "prompts.jsonl", built["prompts"])
    rsha, _ = write_jsonl(out_dir / "references.jsonl", built["references"])
    manifest = {"schema": PROMPT_SCHEMA, "created_utc": utc_now(), "tool": tool_identity(),
                "file": {"path": str((out_dir / "prompts.jsonl").resolve()), "sha256": psha},
                "references": {"path": str((out_dir / "references.jsonl").resolve()), "sha256": rsha},
                "prompts": n, "problems": len(built["references"]), "k": args.k, "taken_by_stratum": built["taken"],
                "candidates_by_stratum": built["candidates"], "dropped": built["dropped"],
                "inputs": {"scores": {"path": str(args.scores), "sha256": file_sha256(Path(args.scores))},
                           "rows": {"path": str(args.rows), "sha256": file_sha256(Path(args.rows))}},
                "selection": {"seed": args.seed, "order": "stratum R shuffled, then stratum A shuffled; k adjacent records per problem",
                              "exclude_flags": list(EXCLUDE_FLAGS), "split": "train", "max_prompt_tokens": args.max_prompt_tokens,
                              "user_suffix": BOXED_SUFFIX},
                "max_new_tokens": args.max_new_tokens, "renderer": renderer.report(), "screen": screen_info,
                "prompt_tokens_total": sum(p["prompt_tokens"] for p in built["prompts"])}
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(out_dir / "manifest.json", manifest)
    log(f"prompts: {n} records = {len(built['references'])} problems x k={args.k}; strata {built['taken']}; "
        f"candidates {built['candidates']}; dropped {built['dropped']}; {out_dir}")
    return 0


# ------------------------------------------------------------------------------------------- queue decisions
def seal_lib(path: Path) -> Any:
    return load_by_path("d1_seal_lib", path)


def foreign_lines(lines: list[str], lane: str) -> list[str]:
    return [ln for ln in lines if f"lane={lane}" not in ln]


def gpu_rows(csv_text: str) -> dict[int, dict[str, int]]:
    """nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"""
    out = {}
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            out[int(parts[0])] = {"used_mib": int(float(parts[1]))}
    return out


def apps_by_gpu(csv_text: str, uuid_to_index: dict[str, int]) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for line in csv_text.splitlines():
        uuid = line.split(",")[0].strip()
        if uuid in uuid_to_index:
            c[uuid_to_index[uuid]] += 1
    return c


def live_gpu_state() -> dict[int, dict[str, int]]:
    used = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,uuid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=60).stdout
    rows = gpu_rows(used)
    uuids = {line.split(",")[2].strip(): int(line.split(",")[0]) for line in used.splitlines() if line.count(",") >= 2}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=60).stdout
    counts = apps_by_gpu(apps, uuids)
    for g in rows:
        rows[g]["apps"] = counts.get(g, 0)
    return rows


def mem_available_gib(meminfo: str | None = None) -> float:
    text = meminfo if meminfo is not None else Path("/proc/meminfo").read_text()
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, re.M)
    return int(m.group(1)) / 1048576 if m else 0.0


def lock_free(path: Path) -> bool:
    """True when nobody holds an flock on `path` (probe with a non-blocking shared lock, released at once)."""
    import fcntl

    if not path.exists():
        return True
    fd = os.open(str(path), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def v1_review_pending(queues: Path) -> bool:
    return (queues / "d1.sealed").exists() and not (queues / "d1.reviewed").exists()


def gate(*, claims_lines: list[str], gpus: list[int], gpu_state: dict[int, dict[str, int]], locks_free: dict[int, bool],
         mem_gib: float, mem_base_gib: float, mem_per_card_gib: float, min_cards: int, yield_file_exists: bool,
         review_pending: bool, claim_state: Any, lane: str = LANE) -> tuple[int, list[int], list[str]]:
    """(0 open / 10 wait, usable GPUs, reasons).  A card that is claimed / locked / busy is only a note while enough
    other cards are usable; the global blockers (YIELD, D1 review pending, too few cards, memory) always wait."""
    blockers: list[str] = []
    notes: list[str] = []
    if yield_file_exists:
        blockers.append("YIELD file present")
    if review_pending:
        blockers.append("d1.sealed without d1.reviewed: D1's bank review may need P40 1+2+3 (v1 first)")
    lines = foreign_lines(claims_lines, lane)
    usable = []
    for g in gpus:
        held, line = claim_state(lines, g)
        state = gpu_state.get(g)
        if held:
            notes.append(f"GPU{g} claimed: {line[:160]}")
        elif not locks_free.get(g, False):
            notes.append(f"GPU{g} gpu{g}.lock held")
        elif state is None or state["used_mib"] >= CARD_BUSY_MIB or state.get("apps", 0):
            notes.append(f"GPU{g} busy ({state})")
        else:
            usable.append(g)
    if len(usable) < min_cards:
        blockers.append(f"{len(usable)} usable card(s) < {min_cards}")
    # 1.0.2: serve as many cards as host memory allows (each llama-server holds ~9.3 GiB anonymous host RAM, measured)
    fit = int((mem_gib - mem_base_gib) // mem_per_card_gib) if mem_gib > mem_base_gib else 0
    if usable and fit < len(usable):
        if fit >= max(1, min_cards):
            notes.append(f"memory fits {fit} of {len(usable)} usable card(s): serving GPU{','.join(str(g) for g in usable[:fit])}")
            usable = usable[:fit]
        else:
            need = mem_base_gib + mem_per_card_gib * max(1, min_cards)
            blockers.append(f"MemAvailable {mem_gib:.1f} GiB < {need:.1f} GiB ({mem_base_gib} + {mem_per_card_gib} x {max(1, min_cards)})")
    return (10 if blockers else 0), usable, blockers + notes


def yield_check(*, claims_lines: list[str], held: list[int], since_line: int, mem_gib: float, mem_floor_gib: float,
                yield_file_exists: bool, review_pending: bool, lane: str = LANE) -> tuple[int, str]:
    if yield_file_exists:
        return 20, "YIELD file present"
    if review_pending:
        return 20, "d1.sealed without d1.reviewed: yielding the P40s to D1's review"
    for line in claims_lines[since_line:]:
        if f"lane={lane}" in line:
            continue
        head = line.split()
        tokens = " ".join(head[1:3]) if len(head) > 2 else ""
        if any(re.search(rf"\bGPU{g}\b", tokens) for g in held) and not re.search(r"\b(released|RELEASE)\b", line):
            return 20, f"another lane claimed a held card: {line[:160]}"
    if mem_gib < mem_floor_gib:
        return 20, f"MemAvailable {mem_gib:.1f} GiB < floor {mem_floor_gib} GiB"
    return 0, "keep"


def gpu_state_for(args: argparse.Namespace) -> dict[int, dict[str, int]]:
    """nvidia-smi, or --gpu-state-json (queue harness only: {"1": {"used_mib": 5, "apps": 0}, ...})."""
    if getattr(args, "gpu_state_json", None):
        return {int(k): v for k, v in json.loads(Path(args.gpu_state_json).read_text(encoding="utf-8")).items()}
    return live_gpu_state()


def mem_for(args: argparse.Namespace) -> float:
    """/proc/meminfo, or --meminfo FILE (queue harness only)."""
    if getattr(args, "meminfo", None):
        return mem_available_gib(Path(args.meminfo).read_text(encoding="utf-8"))
    return mem_available_gib()


def cmd_gate(args: argparse.Namespace) -> int:
    lib = seal_lib(Path(args.seal_lib))
    claims = Path(args.claims)
    lines = claims.read_text(encoding="utf-8", errors="replace").splitlines() if claims.is_file() else []
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    queues = Path(args.queues_dir)
    code, usable, reasons = gate(
        claims_lines=lines, gpus=gpus, gpu_state=gpu_state_for(args), locks_free={g: lock_free(queues / f"gpu{g}.lock") for g in gpus},
        mem_gib=mem_for(args), mem_base_gib=args.mem_base_gib, mem_per_card_gib=args.mem_per_card_gib,
        min_cards=args.min_cards, yield_file_exists=Path(args.yield_file).exists(), review_pending=v1_review_pending(queues),
        claim_state=lib.claim_state, lane=args.lane)
    print(("GATE OPEN " + ",".join(str(g) for g in usable)) if code == 0 else "GATE WAIT " + " | ".join(reasons))
    return code


def cmd_yield_check(args: argparse.Namespace) -> int:
    claims = Path(args.claims)
    lines = claims.read_text(encoding="utf-8", errors="replace").splitlines() if claims.is_file() else []
    code, why = yield_check(claims_lines=lines, held=[int(g) for g in args.held.split(",") if g.strip()],
                            since_line=args.since_line, mem_gib=mem_for(args), mem_floor_gib=args.mem_floor_gib,
                            yield_file_exists=Path(args.yield_file).exists(), review_pending=v1_review_pending(Path(args.queues_dir)),
                            lane=args.lane)
    print(("YIELD " if code else "KEEP ") + why)
    return code


# ------------------------------------------------------------------------------------------- verify
def journal_records(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Last record per prompt index (selfgen_sampler journal; a torn last line is skipped as the sampler does)."""
    out: dict[int, dict[str, Any]] = {}
    path = run_dir / "journal.jsonl"
    if not path.is_file():
        raise PilotError(f"{path} does not exist")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            out[int(rec["index"])] = rec
    return out


def judge_samples(prompts: list[dict[str, Any]], refs: dict[str, dict[str, Any]], records: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    q = qs()
    by_rid = {p["rid"]: (i, p) for i, p in enumerate(prompts)}
    verdicts = []
    for index, rec in sorted(records.items()):
        if rec["rid"] not in by_rid:
            raise PilotError(f"journal index {index} rid {rec['rid']} is not in the prompt file")
        i, prompt = by_rid[rec["rid"]]
        if i != index:
            raise PilotError(f"journal index {index} carries rid {rec['rid']}, which is prompt {i}")
        text = rec.get("text") or ""
        full = "<think>\n" + text
        closed = "</think>" in text
        eog = rec.get("end") == "eog"
        answer, rule = q.extract_final_answer(full)
        ref = refs[prompt["source_rid"]]
        verdicts.append({"index": index, "rid": rec["rid"], "source_rid": prompt["source_rid"], "stratum": prompt["stratum"],
                         "sample_index": prompt["sample_index"], "closed_think": closed, "eog": eog, "end": rec.get("end"),
                         "response_tokens": rec.get("response_tokens"), "answer": answer, "answer_rule": rule,
                         "reference": ref.get("reference"),
                         "equal_reference": q.answers_equal(answer, ref.get("reference")) if prompt["stratum"] == "R" else None})
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for v in verdicts:
        groups[v["source_rid"]].append(v)
    for v in verdicts:
        base = v["closed_think"] and v["eog"] and v["answer"] is not None
        if v["stratum"] == "R":
            v["accepted"] = bool(base and v["equal_reference"] is True)
            v["verified_by"] = "reference"
        else:
            sibs = groups[v["source_rid"]]
            complete = len(sibs) >= 2 and all(s["closed_think"] and s["eog"] and s["answer"] is not None for s in sibs)
            agree = complete and all(q.answers_equal(sibs[0]["answer"], s["answer"]) is True for s in sibs[1:])
            v["accepted"] = bool(base and agree)
            v["verified_by"] = "k_agreement"
    return verdicts


def summarize_verdicts(verdicts: list[dict[str, Any]], k: int) -> dict[str, Any]:
    def rate(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    out: dict[str, Any] = {}
    for stratum in ("R", "A"):
        vs = [v for v in verdicts if v["stratum"] == stratum]
        probs = collections.defaultdict(list)
        for v in vs:
            probs[v["source_rid"]].append(v)
        complete = {s: g for s, g in probs.items() if len(g) >= k}
        extractable = [v for v in vs if v["closed_think"] and v["eog"] and v["answer"] is not None]
        tokens = [v["response_tokens"] or 0 for v in vs]
        acc_tokens = [v["response_tokens"] or 0 for v in vs if v["accepted"]]
        block = {
            "samples": len(vs), "problems_sampled": len(probs), "problems_complete_k": len(complete),
            "closed_think_rate": rate(sum(v["closed_think"] for v in vs), len(vs)),
            "eog_rate": rate(sum(v["eog"] for v in vs), len(vs)),
            "closed_eog_extractable_rate": rate(len(extractable), len(vs)),
            "accepted_samples": sum(v["accepted"] for v in vs),
            "sample_acceptance_rate": rate(sum(v["accepted"] for v in vs), len(vs)),
            "problem_yield_any_accepted": rate(sum(any(v["accepted"] for v in g) for g in complete.values()), len(complete)),
            "problem_yield_closed_extractable": rate(sum(any(v in extractable for v in g) for g in complete.values()), len(complete)),
            "mean_response_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
            "tokens_per_accepted_sample": round(sum(tokens) / len(acc_tokens), 1) if acc_tokens else None,
            "end_types": dict(collections.Counter(v["end"] for v in vs)),
            "answer_rules": dict(collections.Counter(v["answer_rule"] for v in vs)),
        }
        if stratum == "R":
            decided = [v for v in extractable if v["equal_reference"] is not None]
            block["accuracy_vs_reference_on_extractable_decided"] = rate(sum(v["equal_reference"] is True for v in decided), len(decided))
            block["undecidable_answer_forms"] = sum(1 for v in extractable if v["equal_reference"] is None)
        out[stratum] = block
    allv = verdicts
    probs_all = collections.defaultdict(list)
    for v in allv:
        probs_all[v["source_rid"]].append(v)
    complete_all = [g for g in probs_all.values() if len(g) >= k]
    out["program_doc_criteria"] = {
        "closed_extractable_problem_yield": rate(sum(any(v["closed_think"] and v["eog"] and v["answer"] is not None for v in g)
                                                     for g in complete_all), len(complete_all)),
        "closed_extractable_target": ">= 0.60 of prompts (program doc 6.6)",
        "referenced_accuracy": out["R"].get("accuracy_vs_reference_on_extractable_decided"),
        "referenced_accuracy_target": ">= 0.80 (program doc 6.6)",
        "note": "diagnostics: these targets set the G2 budget, they gate nothing",
    }
    return out


def cmd_verify(args: argparse.Namespace) -> int:
    run_dir, pdir, out_dir = Path(args.run_dir), Path(args.prompts_dir), Path(args.out_dir)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    if file_sha256(pdir / "prompts.jsonl") != manifest["file"]["sha256"]:
        raise PilotError(f"{pdir / 'prompts.jsonl'} does not hash to its manifest")
    prompts = read_jsonl(pdir / "prompts.jsonl")
    refs = {r["source_rid"]: r for r in read_jsonl(pdir / "references.jsonl")}
    records = journal_records(run_dir)
    verdicts = judge_samples(prompts, refs, records)
    summary = {"schema": SUMMARY_SCHEMA, "at_utc": utc_now(), "tool": tool_identity(), "run_dir": str(run_dir),
               "prompts_manifest_sha256": manifest.get("manifest_sha256"), "k": manifest.get("k"),
               "journal_records": len(records), "run_finalized": (run_dir / "manifest.json").exists(),
               "strata": summarize_verdicts(verdicts, int(manifest.get("k") or 2))}
    progress = run_dir / "progress.json"
    if progress.is_file():
        p = json.loads(progress.read_text(encoding="utf-8"))
        summary["throughput"] = {k: p.get(k) for k in ("rate_tokens_per_s_session", "rate_tokens_per_s_recent", "response_tokens_done",
                                                       "prompts_done", "updated_utc")}
        summary["throughput"]["servers"] = {u: {k: s.get(k) for k in ("done", "tokens", "errors")} for u, s in (p.get("servers") or {}).items()}
    write_jsonl(out_dir / "verdicts.jsonl", verdicts)
    accepted = [v for v in verdicts if v["accepted"]]
    by_index = {v["index"]: v for v in accepted}
    rows = []
    for index in sorted(by_index):
        rec, prompt, v = records[index], prompts[index], by_index[index]
        user = prompt["messages"][0]["content"]
        response = "<think>\n" + (rec.get("text") or "")
        rows.append({"rid": prompt["rid"], "dataset_id": DATASET_ID, "route": ROUTE, "family": "text",
                     "instruction": user, "response": response,
                     "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": response}],
                     "provenance": {"source_rid": prompt["source_rid"], "stratum": prompt["stratum"], "verified_by": v["verified_by"],
                                    "answer": v["answer"], "reference": v["reference"], "seed": rec.get("seed"),
                                    "response_tokens": rec.get("response_tokens"), "end": rec.get("end"),
                                    "run_dir": str(run_dir), "journal_index": index}})
    batch = out_dir / "batch"
    rsha, n = write_jsonl(batch / "rows.jsonl", rows)
    bman = {"schema": INTAKE_SCHEMA, "batch_id": f"{time.strftime('%Y-%m-%d', time.gmtime())}-selfgen-vmath-0001",
            "source_kind": "selfgen", "producer": {"sampler": "selfgen_sampler 1.2.0 (run dir journal)", "verifier": tool_identity()},
            "rows_sha256": rsha, "rows": n, "tokens_response": sum(r["provenance"]["response_tokens"] or 0 for r in rows),
            "licence_note": "parent self-generation (Qwen3.8-27B-Uncensored Q4_K_M sampler); prompts from registry train rows",
            "screen_receipt_sha256": manifest.get("manifest_sha256"), "parent_batch_ids": [],
            "rescore_note": "logits NOT rescored yet: selfgen_sampler export-rescore exports a whole sealed run; a filtered "
                            "export of accepted indices is not implemented (left out on purpose)"}
    write_json_atomic(batch / "manifest.json", bman)
    write_json_atomic(out_dir / "summary.json", summary)
    r, a = summary["strata"]["R"], summary["strata"]["A"]
    log(f"verify: {len(verdicts)} samples; R accepted {r['accepted_samples']}/{r['samples']} (accuracy on extractable "
        f"{r.get('accuracy_vs_reference_on_extractable_decided')}, closed+eog+answer {r['closed_eog_extractable_rate']}); "
        f"A accepted {a['accepted_samples']}/{a['samples']} (problem yield {a['problem_yield_any_accepted']}); batch {n} rows -> {batch}")
    return 0


# ------------------------------------------------------------------------------------------- CLI
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prompts")
    sp.add_argument("--scores", required=True)
    sp.add_argument("--rows", default=str(DEFAULT_ROWS))
    sp.add_argument("--parent-dir", default=str(DEFAULT_PARENT))
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--k", type=int, default=2)
    sp.add_argument("--max-prompts", type=int, default=500, help="problems (each becomes k prompt records)")
    sp.add_argument("--seed", type=int, default=20261015)
    sp.add_argument("--max-prompt-tokens", type=int, default=2000)
    sp.add_argument("--max-new-tokens", type=int, default=6144)
    sp.add_argument("--allow-unpinned", action="store_true", help="fixtures only")
    sp.add_argument("--no-screen", action="store_true", help="fixtures only")
    sp.add_argument("--corpus-tools", default=str(OC / "distill" / "tools"))
    sp.add_argument("--eval-split", default=str(OC / "distill" / "corpus" / "eval_split" / "eval_rows.jsonl"))
    sp.add_argument("--toolcall-bank", default=str(OC / "calibration" / "banks" / "omni_toolcall_bank_v1.json"))
    sp.add_argument("--codex-split", action="append", default=[
        str(OC / "distill" / "codex-preparation-01a1326f-20261013" / "corpus-schema-v2" / "validation.jsonl"),
        str(OC / "distill" / "codex-preparation-01a1326f-20261013" / "corpus-schema-v2" / "test.jsonl")])

    for name in ("gate", "yield-check"):
        sp = sub.add_parser(name)
        sp.add_argument("--claims", default=str(DEFAULT_CLAIMS))
        sp.add_argument("--queues-dir", default=str(DEFAULT_QUEUES))
        sp.add_argument("--yield-file", required=True)
        sp.add_argument("--lane", default=LANE)
        sp.add_argument("--meminfo", default=None, help="queue harness only: read MemAvailable from this file")
        sp.add_argument("--gpu-state-json", default=None, help="queue harness only: card state instead of nvidia-smi")
        if name == "gate":
            sp.add_argument("--gpus", default="1,2")
            sp.add_argument("--min-cards", type=int, default=1)
            sp.add_argument("--mem-base-gib", type=float, default=70.0)
            sp.add_argument("--mem-per-card-gib", type=float, default=9.5)
            sp.add_argument("--seal-lib", default=str(DEFAULT_SEAL_LIB))
        else:
            sp.add_argument("--held", required=True)
            sp.add_argument("--since-line", type=int, default=0, help="claims lines before our own claim are not news")
            sp.add_argument("--mem-floor-gib", type=float, default=70.0)

    sp = sub.add_parser("verify")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--prompts-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prompts":
            return cmd_prompts(args)
        if args.command == "gate":
            return cmd_gate(args)
        if args.command == "yield-check":
            return cmd_yield_check(args)
        if args.command == "verify":
            return cmd_verify(args)
    except PilotError as exc:
        log(f"REFUSED: {exc}")
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
