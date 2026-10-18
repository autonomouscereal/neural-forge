#!/usr/bin/env python3
"""fold_eval.py -- behaviour-first evaluation harness for qwen3_5 ladder children (0.8B / 4B parents and their folds).

  prepare  (CPU)  freeze the eval pack: Codex's held oracle tasks (24 arithmetic, 12 code / 48 cases, 8 strict-JSON,
                  61 BFCL single + 57 BFCL parallel call proposals; grader = Codex evaluation.py, pinned), 64 held
                  free-running retention prompts from the corpus carve's eval split (never in any training set built
                  by distill_corpus_build), the 15-case bank v1 for a human reader; plus eval_surface.jsonl, the text
                  every training pool of this lane is screened against.
  run      (GPU)  greedy, thinking-off generation for one checkpoint (+ optional MLP overlay) on every block.
  grade    (CPU)  content block (primary): Codex oracles, artifact-tolerant; each miss classed artifact-miss
                  (empty / cap / loop) or content-miss.  Diagnostics: folding_retention.py 5/8/16/32 vs the parent,
                  empty / cap / repetition counts, bank v1 terminations.  Bank v1 TEXT is for a human; never scored here.
  pair     (CPU)  paired candidate-vs-baseline: discordant counts + exact McNemar on content passes, pooled and by
                  route; retention and artifact deltas as diagnostics.

Grading contract: token agreement is never a gate (owner rule).  Promotion questions are answered on content.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_common as fc  # noqa: E402

TOOL = "fold_eval"
VERSION = "1.0.0"
PACK_SCHEMA = "fold-research.eval-pack.v1"
SEMANTIC_ROUTES = ("text_arithmetic", "text_code", "text_instruction", "tool_call", "tool_parallel")
MEDIA_ROUTES = {"audio", "image", "music", "video", "speech", "tts", "sound", "ocr", "multimodal", "media_flow",
                "long_context", "vision", "image_edit"}
RETENTION_MAX_NEW = 64
SEMANTIC_MAX_NEW = 512
BANK_MAX_NEW = 1024
SALT = "fold-research-retention-v1"


# ------------------------------------------------------------------------------------------------------- prepare

def cmd_prepare(args) -> None:
    pins = {"vendor": fc.verify_vendor_pins(), "tools": fc.own_shas()}
    tasks = fc.read_jsonl(args.codex_tasks)
    manifest = json.loads(Path(args.codex_manifest).read_text(encoding="utf-8"))
    if manifest.get("grader_sha256") != pins["vendor"]["chimera_recovery/evaluation.py"]:
        raise SystemExit("Codex eval manifest names a different grader than the vendored evaluation.py")
    semantic = [t for t in tasks if t["route"] in SEMANTIC_ROUTES]
    route_counts = Counter(t["route"] for t in semantic)
    tok, eos = fc.load_tokenizer(args.tokenizer_dir)

    rows = fc.read_jsonl(args.eval_rows)
    keys_seen = sorted({k for r in rows[:50] for k in r})
    pool = defaultdict(list)
    for r in rows:
        route = r.get("route") or r.get("raw_route") or ""
        if r.get("render") != "chat" or route in MEDIA_ROUTES or r.get("sidecars") or r.get("media"):
            continue
        instruction = (r.get("instruction") or "").strip()
        if not 40 <= len(instruction) <= 3000:
            continue
        ident = r.get("record_id") or r.get("uid") or r.get("rid")
        pool[route].append((fc.salted_rank(SALT, str(ident)), ident, instruction, r))
    for route in pool:
        pool[route].sort(key=lambda x: x[0])
    retention, cursor = [], {route: 0 for route in pool}
    routes = sorted(pool)
    while len(retention) < args.n_retention and any(cursor[r] < len(pool[r]) for r in routes):
        for route in routes:
            if len(retention) >= args.n_retention or cursor[route] >= len(pool[route]):
                continue
            _, ident, instruction, row = pool[route][cursor[route]]
            cursor[route] += 1
            messages = [{"role": "user", "content": instruction}]
            ids = fc.chat_ids(tok, messages)
            if len(ids) > 1024:
                continue
            retention.append({"id": f"ret:{ident}", "source_group_id": f"carve-eval:{ident}", "split": "validation",
                              "route": route, "dataset_id": row.get("dataset_id"), "messages": messages,
                              "prompt_ids": ids})
    if len(retention) < args.n_retention:
        raise SystemExit(f"only {len(retention)} retention prompts qualified")

    bank = json.loads(Path(args.bank_v1).read_text(encoding="utf-8"))
    bank_sha = fc.file_sha256(args.bank_v1)
    if bank_sha != args.bank_v1_sha256:
        raise SystemExit(f"bank v1 sha {bank_sha} != expected {args.bank_v1_sha256}")
    bank_cases = [{"id": p["id"], "category": p.get("category"), "messages": p["messages"], "context": p.get("context")}
                  for p in bank["prompts"]]

    surface = []
    for t in tasks:                                   # ALL 274 Codex tasks, not only the graded routes
        for i, m in enumerate(t["input"]["messages"]):
            surface.append({"source": "codex274", "id": f"{t['id']}#{i}", "text": m.get("content") or ""})
    for r in retention:
        surface.append({"source": "retention64", "id": r["id"], "text": r["messages"][0]["content"]})
    for c in bank_cases:
        for i, m in enumerate(c["messages"]):
            surface.append({"source": "bank_v1", "id": f"{c['id']}#{i}", "text": m["content"]})
    extra_pins = {}
    for path in args.screen_extra:
        extra_pins[path] = fc.file_sha256(path)
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        for i, text in enumerate(_strings(body)):
            if len(fc.words(text)) >= fc.SHORT_MIN_WORDS:
                surface.append({"source": Path(path).name, "id": str(i), "text": text})

    out = Path(args.out)
    if (out / "pack.json").exists():
        raise SystemExit(f"{out}/pack.json exists; a frozen pack is never rewritten")
    surface_sha = fc.write_jsonl(out / "eval_surface.jsonl", surface)
    pack = {"schema": PACK_SCHEMA, "tool": TOOL, "version": VERSION, "created_utc": _utc(),
            "tokenizer_dir": str(args.tokenizer_dir), "tokenizer_sha256": fc.file_sha256(Path(args.tokenizer_dir) / "tokenizer.json"),
            "chat_template_sha256": fc.file_sha256(Path(args.tokenizer_dir) / "chat_template.jinja"),
            "eos_ids": eos, "thinking": False,
            "decoding": {"strategy": "greedy", "do_sample": False},
            "max_new": {"retention": RETENTION_MAX_NEW, "semantic": SEMANTIC_MAX_NEW, "bank": BANK_MAX_NEW},
            "semantic": semantic, "semantic_routes": dict(route_counts),
            "semantic_excluded_routes": dict(Counter(t["route"] for t in tasks if t["route"] not in SEMANTIC_ROUTES)),
            "semantic_excluded_reason": "tool_schema / tool_result / tool_retry / tool_multiturn need executed tool "
                                        "outcomes or scripted follow-up turns; this single-turn runner records none",
            "retention": retention, "retention_rule": f"eval_split rows, render=chat, non-media route, 40-3000 chars, "
                                                      f"<=1024 prompt tokens, salted rank '{SALT}', round-robin by route",
            "retention_routes": dict(Counter(r["route"] for r in retention)),
            "eval_rows_keys_seen": keys_seen,
            "bank_v1": bank_cases,
            "pins": {"codex_tasks": {"path": str(args.codex_tasks), "sha256": fc.file_sha256(args.codex_tasks)},
                     "codex_manifest": {"path": str(args.codex_manifest), "sha256": fc.file_sha256(args.codex_manifest),
                                        "holdout_sha256": manifest.get("holdout_sha256")},
                     "eval_rows": {"path": str(args.eval_rows), "sha256": fc.file_sha256(args.eval_rows)},
                     "bank_v1": {"path": str(args.bank_v1), "sha256": bank_sha},
                     "screen_extra": extra_pins, "eval_surface_sha256": surface_sha, **pins}}
    pack["retention_prompts_sha256"] = fc.digest([r["prompt_ids"] for r in retention])
    sha = fc.write_json(out / "pack.json", pack)
    print(json.dumps({"pack": str(out / "pack.json"), "sha256": sha, "semantic": dict(route_counts),
                      "retention": len(retention), "retention_routes": pack["retention_routes"],
                      "bank_v1": len(bank_cases), "surface_items": len(surface)}))


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------------------------------------------------- run

def cmd_run(args) -> None:
    import torch
    pack_path = Path(args.pack)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    out = Path(args.out)
    if (out / "run.json").exists():
        print(f"SKIP {out}/run.json exists")
        return
    fc.verify_vendor_pins()
    started = time.time()
    identity = fc.checkpoint_identity(args.model_dir, args.overlay)
    tok, eos = fc.load_tokenizer(args.model_dir)
    if fc.file_sha256(Path(args.model_dir) / "tokenizer.json") != pack["tokenizer_sha256"] or eos != pack["eos_ids"]:
        raise SystemExit("model tokenizer differs from the pack's")
    model, load = fc.load_model(args.model_dir, device=args.device, dtype=args.dtype, overlay=args.overlay)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    rows, timings = [], {}
    blocks = args.blocks.split(",")
    if "retention" in blocks:
        t0 = time.time()
        got = fc.generate(model, [r["prompt_ids"] for r in pack["retention"]], max_new=pack["max_new"]["retention"],
                          eos_ids=eos, pad_id=pad, batch_size=1)
        for r, g in zip(pack["retention"], got):
            rows.append({"block": "retention", "id": r["id"], **g})
        timings["retention_s"] = round(time.time() - t0, 1)
    if "semantic" in blocks:
        t0 = time.time()
        prompts = [fc.chat_ids(tok, t["input"]["messages"]) for t in pack["semantic"]]
        got = fc.generate(model, prompts, max_new=pack["max_new"]["semantic"], eos_ids=eos, pad_id=pad,
                          batch_size=args.batch)
        for t, g in zip(pack["semantic"], got):
            rows.append({"block": "semantic", "id": t["id"], "text": tok.decode(_content(g, eos)), **g})
        timings["semantic_s"] = round(time.time() - t0, 1)
    if "bank" in blocks:
        t0 = time.time()
        prompts = [fc.chat_ids(tok, c["messages"]) for c in pack["bank_v1"]]
        got = fc.generate(model, prompts, max_new=pack["max_new"]["bank"], eos_ids=eos, pad_id=pad,
                          batch_size=min(args.batch, 8))
        for c, g in zip(pack["bank_v1"], got):
            rows.append({"block": "bank", "id": c["id"], "text": tok.decode(_content(g, eos)), **g})
        timings["bank_s"] = round(time.time() - t0, 1)
    rollouts_sha = fc.write_jsonl(out / "rollouts.jsonl", rows)
    run = {"schema": "fold-research.eval-run.v1", "tool": TOOL, "version": VERSION, "created_utc": _utc(),
           "pack": str(pack_path), "pack_sha256": fc.file_sha256(pack_path), "checkpoint": identity, "load": load,
           "label": args.label, "device": args.device, "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
           "batch": args.batch, "blocks": blocks, "rollouts_sha256": rollouts_sha, "timings": timings,
           "wall_s": round(time.time() - started, 1), "peak_rss_gib": fc.peak_rss_gib(), "tools": fc.own_shas()}
    fc.write_json(out / "run.json", run)
    print(json.dumps({"run": str(out / "run.json"), "label": args.label, "timings": timings, "peak_rss_gib": run["peak_rss_gib"]}))


def _content(g: dict, eos: list[int]) -> list[int]:
    ids = g["generated_ids"]
    return ids[:-1] if g["stop_reason"] == "eos" else ids


# --------------------------------------------------------------------------------------------------------- grade

def grade_semantic_task(task: dict, row: dict) -> dict:
    stop = "eos" if row["stop_reason"] == "eos" else "length"
    text = row["text"]
    tool = task["oracle"].get("kind") == "tool"
    if tool:
        messages, parsed = fc.tool_messages(text)
    else:
        messages, parsed = [{"role": "assistant", "content": fc.strip_think(text)}], None
    response = {"messages": messages, "generation": {"stop_reason": stop, "token_ids": row["generated_ids"]}}
    strict = fc.cx_eval.grade_semantics(task, response)
    tolerant = None
    kind = task["oracle"].get("kind")
    if strict["status"] != "pass" and kind == "arithmetic":
        tolerant = fc.cx_eval.grade_arithmetic(fc.tolerant_text(text), task["oracle"])
    elif strict["status"] != "pass" and kind == "instruction_json":
        fences = fc.FENCE_RE.findall(fc.strip_think(text))
        if len(fences) == 1:
            tolerant = fc.cx_eval.grade_semantics(task, {"messages": [{"role": "assistant", "content": fences[0].strip()}]})
    diag = fc.cx_eval.generation_diagnostics({"messages": [{"role": "assistant", "content": fc.strip_think(text)}],
                                              "generation": response["generation"]})
    artifact = diag["empty_assistant"] or diag["cap_reached"] or diag["loop_candidate"] is not None
    if strict["status"] == "pass":
        cls = "pass"
    elif tolerant and tolerant["status"] == "pass":
        cls = "artifact_pass"          # right content behind a formatting artifact
    elif strict["status"] == "ungraded":
        cls = "ungraded"
    else:
        cls = "artifact_miss" if artifact else "content_miss"
    return {"id": task["id"], "route": task["route"], "class": cls, "content_pass": cls in ("pass", "artifact_pass"),
            "strict": strict, "tolerant": tolerant, "transport": parsed, "diagnostics": diag}


def cmd_grade(args) -> None:
    pack = json.loads(Path(args.pack).read_text(encoding="utf-8"))
    fc.verify_vendor_pins()
    run = json.loads((Path(args.run) / "run.json").read_text(encoding="utf-8"))
    rows = {(r["block"], r["id"]): r for r in fc.read_jsonl(Path(args.run) / "rollouts.jsonl")}
    report = {"schema": "fold-research.grade.v1", "tool": TOOL, "version": VERSION, "created_utc": _utc(),
              "pack_sha256": fc.file_sha256(args.pack), "run": run["checkpoint"], "label": run.get("label"),
              "grading_contract": "content (Codex oracles, artifact-tolerant) is primary; retention / artifacts / bank "
                                  "terminations are diagnostics; bank v1 text is read by a human",
              "tools": fc.own_shas(), "vendor": fc.verify_vendor_pins()}
    sem = [grade_semantic_task(t, rows[("semantic", t["id"])]) for t in pack["semantic"] if ("semantic", t["id"]) in rows]
    if sem:
        by_route = defaultdict(Counter)
        for g in sem:
            by_route[g["route"]][g["class"]] += 1
            by_route[g["route"]]["content_pass"] += g["content_pass"]
            by_route[g["route"]]["n"] += 1
        report["semantic"] = {"n": len(sem), "content_pass": sum(g["content_pass"] for g in sem),
                              "strict_pass": sum(g["class"] == "pass" for g in sem),
                              "classes": dict(Counter(g["class"] for g in sem)),
                              "by_route": {k: dict(v) for k, v in sorted(by_route.items())},
                              "tasks": sem}
    if args.parent_run and any(b == "retention" for b, _ in rows):
        parent_run = json.loads((Path(args.parent_run) / "run.json").read_text(encoding="utf-8"))
        prow = {(r["block"], r["id"]): r for r in fc.read_jsonl(Path(args.parent_run) / "rollouts.jsonl")}
        training_groups = json.loads(Path(args.training_groups).read_text(encoding="utf-8")) if args.training_groups else []
        contract = {"schema": "chimera_recovery.fold_rollout_contract.v1",
                    "decoding": {"strategy": "greedy", "do_sample": False}, "vocab_size": fc.VOCAB,
                    "max_new_tokens": pack["max_new"]["retention"], "tokenizer_sha256": pack["tokenizer_sha256"],
                    "evaluation_source_sha256": pack["retention_prompts_sha256"],
                    "parent_checkpoint_sha256": parent_run["checkpoint"]["checkpoint_sha256"],
                    "candidate_checkpoint_sha256": run["checkpoint"]["checkpoint_sha256"],
                    "eos_ids": pack["eos_ids"], "training_source_groups": training_groups}
        records = []
        for r in pack["retention"]:
            prompt_sha = fc.digest(r["prompt_ids"])
            def side(row, sha):
                return {"prompt_sha256": prompt_sha, "checkpoint_sha256": sha, "mode": "free_running",
                        "generated_ids": row["generated_ids"], "stop_reason": row["stop_reason"]}
            records.append({"id": r["id"], "source_group_id": r["source_group_id"], "split": r["split"],
                            "prompt_ids": r["prompt_ids"],
                            "parent": side(prow[("retention", r["id"])], contract["parent_checkpoint_sha256"]),
                            "candidate": side(rows[("retention", r["id"])], contract["candidate_checkpoint_sha256"])})
        if contract["parent_checkpoint_sha256"] == contract["candidate_checkpoint_sha256"]:
            report["retention"] = {"self": True, "note": "parent graded against itself: retention is trivially exact"}
        else:
            scored = fc.cx_ret.score(records, contract)
            report["retention"] = {"summary": scored["summary"], "seal_sha256": scored["seal_sha256"],
                                   "diagnostic_only": True}
            fc.write_json(Path(args.out).with_name("retention_report.json"), scored)
    bank = [rows[("bank", c["id"])] for c in pack["bank_v1"] if ("bank", c["id"]) in rows]
    if bank:
        terms = []
        lines = [f"# Bank v1 answers -- {run.get('label')}  (checkpoint {run['checkpoint']['checkpoint_sha256'][:16]})",
                 "", "Verbatim, thinking off, greedy, cap 1,024. For a human reader; this file grades nothing.", ""]
        for c, r in zip(pack["bank_v1"], bank):
            d = fc.cx_eval.generation_diagnostics({"messages": [{"role": "assistant", "content": r["text"]}],
                                                   "generation": {"stop_reason": "eos" if r["stop_reason"] == "eos" else "length",
                                                                  "token_ids": r["generated_ids"]}})
            terms.append({"id": c["id"], "stop": r["stop_reason"], "tokens": len(r["generated_ids"]),
                          "empty": d["empty_assistant"], "loop": d["loop_candidate"] is not None})
            lines += [f"## {c['id']} ({c.get('category')})", "", f"**Prompt:** {c['messages'][-1]['content']}", "",
                      f"**Reader context:** {c.get('context')}", "",
                      f"**Termination:** {r['stop_reason']}, {len(r['generated_ids'])} tokens, empty={d['empty_assistant']}, "
                      f"loop={d['loop_candidate'] is not None}", "", "```text", r["text"], "```", ""]
        Path(args.out).with_name("bank_v1_answers.md").write_text("\n".join(lines), encoding="utf-8")
        report["bank_v1"] = {"n": len(terms), "eos": sum(t["stop"] == "eos" for t in terms),
                             "capped": sum(t["stop"] == "max_tokens" for t in terms),
                             "empty": sum(t["empty"] for t in terms), "loop": sum(t["loop"] for t in terms),
                             "cases": terms, "content": "READ-REQUIRED (bank_v1_answers.md)"}
    sha = fc.write_json(args.out, report)
    brief = {"grade": str(args.out), "sha256": sha, "label": run.get("label")}
    if "semantic" in report:
        brief["content_pass"] = f"{report['semantic']['content_pass']}/{report['semantic']['n']}"
        brief["classes"] = report["semantic"]["classes"]
    if "retention" in report and "summary" in report["retention"]:
        s = report["retention"]["summary"]
        brief["retention_diag"] = {h: f"{v['matches']}/{v['eligible']}" for h, v in s["consecutive_prefix"].items()}
    if "bank_v1" in report:
        brief["bank_terminations"] = {k: report["bank_v1"][k] for k in ("eos", "capped", "empty", "loop")}
    print(json.dumps(brief))


# ---------------------------------------------------------------------------------------------------------- pair

def pair_report(base: dict, cand: dict) -> dict:
    if base["pack_sha256"] != cand["pack_sha256"]:
        raise SystemExit("different eval packs: not a paired comparison")
    bt = {t["id"]: t for t in base["semantic"]["tasks"]}
    ct = {t["id"]: t for t in cand["semantic"]["tasks"]}
    if set(bt) != set(ct):
        raise SystemExit("task sets differ")
    def block(ids):
        b = sum(1 for i in ids if bt[i]["content_pass"] and not ct[i]["content_pass"])
        c = sum(1 for i in ids if ct[i]["content_pass"] and not bt[i]["content_pass"])
        return {"n": len(ids), "base": sum(bt[i]["content_pass"] for i in ids), "cand": sum(ct[i]["content_pass"] for i in ids),
                "lost": b, "gained": c, "mcnemar_p": round(fc.mcnemar_exact(b, c), 6)}
    routes = sorted({t["route"] for t in bt.values()})
    out = {"schema": "fold-research.pair.v1", "base": base["label"], "cand": cand["label"],
           "base_checkpoint": base["run"]["checkpoint_sha256"], "cand_checkpoint": cand["run"]["checkpoint_sha256"],
           "content_pooled": block(list(bt)),
           "content_text": block([i for i in bt if bt[i]["route"].startswith("text_")]),
           "content_tool": block([i for i in bt if bt[i]["route"].startswith("tool_")]),
           "content_by_route": {r: block([i for i in bt if bt[i]["route"] == r]) for r in routes},
           "artifacts_diag": {k: {"base": base["semantic"]["classes"].get(k, 0), "cand": cand["semantic"]["classes"].get(k, 0)}
                              for k in ("artifact_miss", "content_miss", "artifact_pass", "ungraded")}}
    if "summary" in base.get("retention", {}) and "summary" in cand.get("retention", {}):
        bs, cs = base["retention"]["summary"], cand["retention"]["summary"]
        out["retention_diag"] = {h: {"base": bs["consecutive_prefix"][h]["matches"], "cand": cs["consecutive_prefix"][h]["matches"],
                                     "eligible": cs["consecutive_prefix"][h]["eligible"]} for h in cs["consecutive_prefix"]}
        out["retention_diag"]["empty"] = {"base": bs["candidate_empty"], "cand": cs["candidate_empty"]}
        out["retention_diag"]["capped"] = {"base": bs["candidate_capped"], "cand": cs["candidate_capped"]}
    if "bank_v1" in base and "bank_v1" in cand:
        out["bank_terminations_diag"] = {k: {"base": base["bank_v1"][k], "cand": cand["bank_v1"][k]} for k in ("eos", "capped", "empty", "loop")}
    return out


def cmd_pair(args) -> None:
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.cand).read_text(encoding="utf-8"))
    out = pair_report(base, cand)
    fc.write_json(args.out, out)
    print(json.dumps({"pair": str(args.out), "pooled": out["content_pooled"], "text": out["content_text"], "tool": out["content_tool"]}))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--codex-tasks", required=True)
    a.add_argument("--codex-manifest", required=True)
    a.add_argument("--eval-rows", required=True)
    a.add_argument("--bank-v1", required=True)
    a.add_argument("--bank-v1-sha256", default="0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b")
    a.add_argument("--screen-extra", nargs="*", default=[])
    a.add_argument("--tokenizer-dir", required=True)
    a.add_argument("--n-retention", type=int, default=64)
    a.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pack", required=True)
    r.add_argument("--model-dir", required=True)
    r.add_argument("--overlay", default=None)
    r.add_argument("--label", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--dtype", default="float32")
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--blocks", default="retention,semantic,bank")
    r.add_argument("--out", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--pack", required=True)
    g.add_argument("--run", required=True)
    g.add_argument("--parent-run", default=None)
    g.add_argument("--training-groups", default=None)
    g.add_argument("--out", required=True)
    q = sub.add_parser("pair")
    q.add_argument("--base", required=True)
    q.add_argument("--cand", required=True)
    q.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"prepare": cmd_prepare, "run": cmd_run, "grade": cmd_grade, "pair": cmd_pair}[args.cmd](args)


if __name__ == "__main__":
    main()
